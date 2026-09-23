#!/usr/bin/env bash
# ===================================================================
# LoRA-capable vLLM launcher for the adapterCL project.
#
# Derived from vlm/startVLMmodel_q35.sh (Qwen3.5 / Gated-DeltaNet flags) and
# vlm/startVLMmodel_bw.sh (Blackwell sm_120 profile). Same CLI surface as both --
# only --model and --port -- so it drops straight into the repo's eval driver:
#
#   VLM_SCRIPT=adapter_project/scripts/startVLM_lora.sh \
#   LORA_MODULES='wiki_e3=/path/to/adapter' MAX_LORA_RANK=16 \
#   MODEL_PATH=Qwen/Qwen3.5-9B OUTPUT_ROOT=... bash run_v6_eval_q35_sonnet.sh
#
# run_v6_eval_q35_sonnet.sh:166 invokes it as
#   ( CUDA_VISIBLE_DEVICES=$EVAL_CUDA_DEVICES bash "$VLM_SCRIPT" --model M --port P )
# and its stop_vlm (lines 188-223) finds the workers by port + by matching
# CUDA_VISIBLE_DEVICES inside /proc/<pid>/environ, so this script must not fork
# the server into a different CVD -- hence `exec`, exactly as the originals do.
#
# ---------------------------------------------------------------- flags -----
#   --enable-lora            turn on the LoRA punica path. WITHOUT IT, --lora-modules
#                            is a startup error; WITH it but with a mis-named module,
#                            vLLM logs "will be ignored" and serves the BASE MODEL.
#                            That silent failure is adapterCL.md 4.5 / 9's headline
#                            risk -- always follow a first launch with
#                            `python -m adaptercl.verify_adapter vllm --log <log>`.
#   --max-lora-rank R        vLLM only accepts 1,8,16,32,64,128,256,320,512
#                            (config/lora.py:26). We round UP to the next legal
#                            value; a server rank >= the adapter rank is correct,
#                            a rank below it fails to load.
#   --max-loras N            how many DISTINCT adapters may be live in one batch.
#                            The 4.3 per-episode granularity needs N >= the number
#                            of cells served concurrently; joblib runs
#                            EVAL_N_JOBS episodes at once, so N=1 with two cells
#                            serialises the batch.
#   --max-cpu-loras M        adapter slots held in host RAM (must be >= max-loras).
#   --lora-modules NAME=PATH one per adapter. NAME is what benchmark_adapter.py
#                            sends as the OpenAI `model` field (--served-name).
#   --served-model-name      keeps the BASE reachable under its own name on the
#                            same server, so base-vs-adapter is one process, one
#                            GPU, one KV cache -- and a paired A/B (6.6) never
#                            confounds the comparison with a server restart.
#   --gdn-prefill-backend triton
#                            Qwen3.5 only. Avoids the FlashInfer GDN-prefill CUDA
#                            JIT, which cannot compile in this env (libcu++/CCCL
#                            C++17). Auto-omitted for non-GDN architectures --
#                            that is the one flag startVLMmodel_bw.sh drops.
#   --enforce-eager          skip cudagraph capture (both originals).
#   VLLM_ATTENTION_BACKEND=FLASH_ATTN, VLLM_USE_FLASHINFER_SAMPLER=0
#                            avoid the flashinfer JITs; verified 2026-07-04.
#
# ------------------------------------------------- per-episode granularity ---
# 4.3 freezes the adapter for a whole episode but a study sweeps many cells. Two
# ways to serve that:
#   (a) STATIC -- list every cell up front:
#         LORA_MODULES='wiki_e1=/o/cells/wiki_e1 wiki_e2=/o/cells/wiki_e2 ...'
#         MAX_LORAS=8
#       and have each episode address its cell by served-name. Preferred: no
#       mutation of a running server, so concurrent joblib workers cannot race.
#   (b) DYNAMIC -- ALLOW_RUNTIME_LORA_UPDATING=1 exports
#       VLLM_ALLOW_RUNTIME_LORA_UPDATING=True, which attaches
#         POST /v1/load_lora_adapter    {"lora_name": "...", "lora_path": "..."}
#         POST /v1/unload_lora_adapter  {"lora_name": "..."}
#       (vllm/entrypoints/serve/lora/api_router.py:43-66; the request model is
#       vllm/entrypoints/serve/lora/protocol.py:7). vLLM itself logs that this
#       "should ONLY be used for local development". Needed when the hypernetwork
#       materialises a fresh adapter per episode (Phase 3) and the set is not
#       known at launch. Load is NOT atomic w.r.t. in-flight requests, so only use
#       it with EVAL_N_JOBS=1 or a name-per-episode scheme.
#
# Self-contained: absolute vllm_q35 paths, no `conda activate` (a scratch purge
# breaks activation but leaves the env bin usable).
# ===================================================================
set -uo pipefail

VLLM_Q35_BIN="${VLLM_Q35_BIN:-${ADAPTERCL_ENVS_ROOT:-${HOME}/miniconda3/envs}/vllm_q35/bin}"
export PATH="$VLLM_Q35_BIN:$PATH"                       # ninja + JIT tools resolve
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

PORT=8001
MODEL_PATH="${ADAPTERCL_BASE_MODEL:-Qwen/Qwen3.5-9B}"
# 65536 matches the training cutoff_len. The old 40960 was inherited from a
# Qwen3-4B recipe; Qwen3.5-9B supports 262144. At 40960 the agent's own budget
# (40960) plus max_new_tokens (2048) exceeds the server limit and EVERY request
# 400s once a TimeWarp AXTree prompt gets long -- the eval then reports 0.0 with
# no crash, which is indistinguishable from a model that simply fails.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
# HARD FLOOR. Every result in this study was produced at 65536, and context
# length changes agent behaviour: at 40960 the prompt overflowed and every
# request 400'd, which the harness recorded as avg_reward=0.0 rather than as an
# error. Serving any unit at a smaller context would silently make it
# non-comparable to the rest while still producing plausible-looking numbers.
# If a card cannot hold 65536, the correct action is to not use that card.
# Override only with an explicit, deliberate ADAPTERCL_ALLOW_SHORT_CTX=1.
if [ "${MAX_MODEL_LEN}" -lt 65536 ] && [ "${ADAPTERCL_ALLOW_SHORT_CTX:-0}" != "1" ]; then
  echo "REFUSING: MAX_MODEL_LEN=$MAX_MODEL_LEN < 65536." >&2
  echo "  Results at a reduced context are not comparable to this study's." >&2
  echo "  Use a larger card, or set ADAPTERCL_ALLOW_SHORT_CTX=1 to override." >&2
  exit 2
fi
GPU_UTIL="${VLM_GPU_UTIL:-0.85}"
GDN_PREFILL_BACKEND="${GDN_PREFILL_BACKEND:-triton}"     # 'none'/'' to omit
ENFORCE_EAGER="${VLM_ENFORCE_EAGER:-1}"

LORA_MODULES="${LORA_MODULES:-}"                         # 'name=path name=path' or comma-separated
MAX_LORA_RANK="${MAX_LORA_RANK:-16}"
MAX_LORAS="${MAX_LORAS:-1}"
MAX_CPU_LORAS="${MAX_CPU_LORAS:-}"                       # defaults to MAX_LORAS
SERVED_MODEL_NAMES="${SERVED_MODEL_NAMES:-}"             # defaults to MODEL_PATH
ALLOW_RUNTIME_LORA_UPDATING="${ALLOW_RUNTIME_LORA_UPDATING:-0}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"

die() { echo "startVLM_lora.sh: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
    case $1 in
        --port)  PORT="$2"; shift 2 ;;
        --model) MODEL_PATH="$2"; shift 2 ;;
        --lora)  LORA_MODULES="${LORA_MODULES:+$LORA_MODULES }$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--port PORT] [--model MODEL] [--lora NAME=PATH]"
            echo "Env: LORA_MODULES MAX_LORA_RANK MAX_LORAS MAX_CPU_LORAS"
            echo "     SERVED_MODEL_NAMES ALLOW_RUNTIME_LORA_UPDATING MAX_MODEL_LEN"
            echo "     VLM_GPU_UTIL VLM_ENFORCE_EAGER GDN_PREFILL_BACKEND EXTRA_VLLM_ARGS"
            exit 0 ;;
        *) die "Unknown option: $1" ;;
    esac
done

# ---- Blackwell (sm_120) detection ------------------------------------------
# startVLMmodel_bw.sh exists because tw_r1_q3 cannot serve sm_120; vllm_q35 can.
# We are already on vllm_q35, so the only thing left to inherit from the bw
# profile is its *conditional* --enforce-eager, which VLM_ENFORCE_EAGER gives us
# on every GPU. Recorded and echoed so a run log says which silicon it ran on.
COMPUTE_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')"
IS_BLACKWELL=0
case "${COMPUTE_CAP:-}" in 12.*) IS_BLACKWELL=1 ;; esac

# ---- GDN prefill backend ----------------------------------------------------
# The one real difference between the two upstream launchers: the bw script drops
# --gdn-prefill-backend because it serves DENSE Qwen3, which has no Gated-DeltaNet
# layers and rejects the flag. Decide from the checkpoint, not the GPU: a Qwen3.5
# on Blackwell still needs triton prefill, and a dense Qwen3 anywhere still must
# not get the flag.
MODEL_TYPE=""
if [ -f "$MODEL_PATH/config.json" ]; then
    MODEL_TYPE="$(sed -n 's/.*"model_type"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' \
                    "$MODEL_PATH/config.json" | head -1)"
fi
IS_GDN=0
case "${MODEL_TYPE:-}${MODEL_PATH}" in
    *qwen3_5*|*qwen3_next*|*Qwen3.5*|*Qwen3-Next*|*qwen3.5*) IS_GDN=1 ;;
esac
case "$(printf '%s' "$GDN_PREFILL_BACKEND" | tr 'A-Z' 'a-z')" in
    ""|none|off|0) IS_GDN=0 ;;
esac

# ---- LoRA args --------------------------------------------------------------
# Split on whitespace AND commas so both 'a=/x b=/y' and 'a=/x,b=/y' work.
#
# vLLM's LoRAParserAction ends with `setattr(namespace, self.dest, lora_list)`
# (entrypoints/openai/cli_args.py:66), so REPEATING --lora-modules silently
# discards every earlier occurrence. All specs must go after ONE flag:
#   --lora-modules a=/x b=/y
# It also treats any item containing a comma as JSON (cli_args.py:52), so a
# comma in a path would be parsed as JSON and error out.
LORA_SPECS=()
LORA_NAMES=""
if [ -n "$LORA_MODULES" ]; then
    _spec="$(printf '%s' "$LORA_MODULES" | tr ',' ' ')"
    for _m in $_spec; do
        case "$_m" in
            *=*) : ;;
            *) die "LORA_MODULES entry '$_m' is not NAME=PATH" ;;
        esac
        _name="${_m%%=*}"; _path="${_m#*=}"
        [ -n "$_name" ] || die "LORA_MODULES entry '$_m' has an empty name"
        case "$_path" in
            *,*) die "adapter path '$_path' contains a comma; vLLM would parse the spec as JSON" ;;
        esac
        # A typo'd path is EXACTLY the 4.5 silent failure -- vLLM would fail the
        # load and the benchmark would fall back to reading like the base model.
        # Refuse to start rather than produce base-model numbers under an
        # adapter's name.
        [ -d "$_path" ] || die "adapter dir does not exist: $_path (from '$_m')"
        [ -f "$_path/adapter_config.json" ] || \
            die "no adapter_config.json in $_path -- not a PEFT adapter dir"
        [ -f "$_path/adapter_model.safetensors" ] || [ -f "$_path/adapter_model.bin" ] || \
            die "no adapter_model.{safetensors,bin} in $_path"
        LORA_SPECS+=("$_name=$_path")
        LORA_NAMES="${LORA_NAMES:+$LORA_NAMES }$_name"
    done
fi
LORA_ARGS=()
[ "${#LORA_SPECS[@]}" -gt 0 ] && LORA_ARGS=(--lora-modules "${LORA_SPECS[@]}")

# ---- rank / slot validation -------------------------------------------------
# vllm/config/lora.py:26 -- MaxLoRARanks is a Literal, so an illegal value dies
# in pydantic validation several seconds into startup with an opaque message.
_legal_rank=""
for _r in 1 8 16 32 64 128 256 320 512; do
    if [ "$MAX_LORA_RANK" -le "$_r" ]; then _legal_rank="$_r"; break; fi
done
[ -n "$_legal_rank" ] || die "MAX_LORA_RANK=$MAX_LORA_RANK exceeds vLLM's maximum of 512"
if [ "$_legal_rank" != "$MAX_LORA_RANK" ]; then
    echo "  note: MAX_LORA_RANK=$MAX_LORA_RANK is not one of vLLM's legal values;"
    echo "        rounding the SERVER's max rank up to $_legal_rank (adapters keep their own r)."
    MAX_LORA_RANK="$_legal_rank"
fi
[ -z "$MAX_CPU_LORAS" ] && MAX_CPU_LORAS="$MAX_LORAS"
[ "$MAX_CPU_LORAS" -ge "$MAX_LORAS" ] || \
    die "MAX_CPU_LORAS=$MAX_CPU_LORAS must be >= MAX_LORAS=$MAX_LORAS (config/lora.py:110-115)"

ENABLE_LORA_ARGS=()
if [ "${#LORA_SPECS[@]}" -gt 0 ] || [ "$ALLOW_RUNTIME_LORA_UPDATING" = "1" ]; then
    ENABLE_LORA_ARGS=(--enable-lora
                      --max-lora-rank "$MAX_LORA_RANK"
                      --max-loras "$MAX_LORAS"
                      --max-cpu-loras "$MAX_CPU_LORAS")
else
    echo "  WARNING: no LORA_MODULES and ALLOW_RUNTIME_LORA_UPDATING=0 --"
    echo "           serving the BASE MODEL ONLY. If you expected an adapter,"
    echo "           this is the 4.5 silent failure and every number from this"
    echo "           server is a base-model number."
fi

if [ "$ALLOW_RUNTIME_LORA_UPDATING" = "1" ]; then
    export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
    echo "  runtime LoRA updating ENABLED: POST /v1/load_lora_adapter"
    echo "    {\"lora_name\":\"cell\",\"lora_path\":\"/abs/dir\"}   (and /v1/unload_lora_adapter)"
fi

[ -z "$SERVED_MODEL_NAMES" ] && SERVED_MODEL_NAMES="$MODEL_PATH"

GDN_ARGS=()
[ "$IS_GDN" = "1" ] && GDN_ARGS=(--gdn-prefill-backend "$GDN_PREFILL_BACKEND")
EAGER_ARGS=()
[ "$ENFORCE_EAGER" = "1" ] && EAGER_ARGS=(--enforce-eager)

# shellcheck disable=SC2206
EXTRA_ARGS=($EXTRA_VLLM_ARGS)
# shellcheck disable=SC2206
SERVED_NAME_ARGS=($SERVED_MODEL_NAMES)

CMD=("$VLLM_Q35_BIN/vllm" serve "$MODEL_PATH"
     --port "$PORT" --host 0.0.0.0
     --trust-remote-code
     --gpu-memory-utilization "$GPU_UTIL"
     --max-model-len "$MAX_MODEL_LEN"
     "${EAGER_ARGS[@]}"
     "${GDN_ARGS[@]}"
     "${ENABLE_LORA_ARGS[@]}"
     "${LORA_ARGS[@]}"
     --served-model-name "${SERVED_NAME_ARGS[@]}"
     "${EXTRA_ARGS[@]}")

echo "=========================================="
echo "adapterCL LoRA vLLM server (vllm_q35)"
echo "  model      = $MODEL_PATH   (model_type=${MODEL_TYPE:-unknown}, gdn=$IS_GDN)"
echo "  port       = $PORT  max_model_len=$MAX_MODEL_LEN  gpu_util=$GPU_UTIL  eager=$ENFORCE_EAGER"
echo "  served-as  = $SERVED_MODEL_NAMES"
echo "  loras      = ${LORA_NAMES:-<none>}   max_rank=$MAX_LORA_RANK max_loras=$MAX_LORAS"
echo "  host       = $(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}  compute_cap=${COMPUTE_CAP:-unknown} blackwell=$IS_BLACKWELL"
echo "  vllm       = $("$VLLM_Q35_BIN/python" -c 'import vllm;print(vllm.__version__)' 2>/dev/null)"
echo "  command    :"
printf '    %q' "${CMD[@]}"; echo
if [ -n "$LORA_NAMES" ]; then
    echo "  AFTER /health: verify the adapter is really applied (4.5 gate) --"
    echo "    python -m adaptercl.verify_adapter vllm --port $PORT --lora-name ${LORA_NAMES%% *} --log <this log>"
fi
echo "=========================================="

exec "${CMD[@]}"
