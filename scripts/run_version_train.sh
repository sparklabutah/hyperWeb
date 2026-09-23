#!/usr/bin/env bash
# run_version_train.sh -- train one LoRA per UI VERSION (the default adapter unit).
#
# 2 makes version-level CL the setting, so the adapter unit is the era, pooled
# across all three sites: 6 adapters, not the 18 (environment x era) cells. The
# cell grid exists only for 6.2's crossed transfer matrix.
#
# Pooling also removes the data blocker outright: wiki_e1 has 0 BC episodes and
# shop_e1 has 2, so 2 of 18 cells cannot be trained at all -- while every
# version has >= 30 verified episodes (v1 30, v2-v6 122-127).
#
#   bash scripts/run_version_train.sh                       # plan only
#   GO=1 SLOTS="1724294:0 1724294:1 ..." bash scripts/run_version_train.sh
#
# One version per slot. Requires the corpora + YAMLs to exist already:
#   python3 -m adaptercl.bcdata cell-corpora versions --tag ver
# Idempotent: a version whose adapter_config.json exists is skipped.

set -uo pipefail
ulimit -Sn 131072 2>/dev/null || true

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(dirname "$PROJECT")"
ENVS="${ADAPTERCL_ENVS_ROOT:-${HOME}/miniconda3/envs}"
PY=/usr/bin/python3
LMF="$ENVS/llamafactory/bin/lmf"

# DeepSpeed's CPUAdam JIT needs a CUDA toolkit matching torch's cu130 build, and
# tilelang (Qwen3.5 FLA kernels) needs C++17 -- system g++ 8.5 is too old.
# Sourced from adaptercl.paths, NOT duplicated here. A hand-copied pair in this
# script silently overrode the corrected constants in paths.py and cost a fourth
# launch round: one definition, or you fix it twice and miss once.
SPACK_GCC="$($PY -c "import sys;sys.path.insert(0,'$PROJECT');from adaptercl import paths;print(paths.SPACK_GCC_BIN)")"
SPACK_CUDA="$($PY -c "import sys;sys.path.insert(0,'$PROJECT');from adaptercl import paths;print(paths.SPACK_CUDA_HOME)")"
[ -x "$SPACK_CUDA/bin/nvcc" ] || { echo "BLOCKED: no nvcc at $SPACK_CUDA/bin/nvcc"; exit 1; }
[ -x "$SPACK_GCC/g++" ]      || { echo "BLOCKED: no g++ at $SPACK_GCC/g++"; exit 1; }

# The env's OWN bin must be on PATH even though we call `lmf` by absolute path:
# with FORCE_TORCHRUN=1 the launcher shells out to plain `torchrun`
# (LLaMA-Factory/src/llamafactory/launcher.py:115), which is resolved via PATH.
# Calling lmf absolutely is not enough -- this cost one full launch round.
ENV_BIN="$ENVS/llamafactory/bin"

GO="${GO:-0}"
SLOTS="${SLOTS:-}"
TAG="${TAG:-ver}"
VERSIONS="${VERSIONS:-v1 v2 v3 v4 v5 v6}"
CPUS="${CPUS:-16}"
STAGGER="${STAGGER:-45}"

YAML_DIR="$PROJECT/out/cells/$TAG/yaml"
LOG_DIR="$PROJECT/out/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/vertrain_$(date +%Y%m%d-%H%M%S).log"
log(){ echo "[vertrain $(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
die(){ log "BLOCKED: $*"; exit 1; }

cd "$PROJECT"
export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"

log "=== adapterCL: per-VERSION LoRA training ==="
log "tag=$TAG versions=[$VERSIONS] go=$GO"

TODO=()
for v in $VERSIONS; do
  y="$YAML_DIR/_gen_${v}.yaml"
  out="$PROJECT/out/cells/$TAG/$v"
  [ -f "$y" ] || die "missing $y -- run: $PY -m adaptercl.bcdata cell-corpora versions --tag $TAG, then regenerate the YAMLs"
  if [ -f "$out/adapter_config.json" ]; then
    log "  skip $v (adapter already at $out)"
    continue
  fi
  TODO+=("$v")
done
log "to train: ${#TODO[@]} -> ${TODO[*]:-none}"
[ "${#TODO[@]}" -eq 0 ] && { log "nothing to do"; exit 0; }

if [ "$GO" != "1" ] || [ -z "$SLOTS" ]; then
  log "NOT LAUNCHING (need GO=1 and SLOTS=\"<jobid>:<gpu> ...\"). Would run, per version:"
  for v in "${TODO[@]}"; do
    log "  # $v"
    log "  export PATH=$SPACK_GCC:$ENV_BIN:\$PATH CUDA_HOME=$SPACK_CUDA FORCE_TORCHRUN=1 NPROC_PER_NODE=1"
    log "  cd $REPO/LLaMA-Factory && $LMF train $YAML_DIR/_gen_${v}.yaml"
  done
  exit 0
fi

read -r -a SLOT_ARR <<< "$SLOTS"
[ "${#SLOT_ARR[@]}" -ge 1 ] || die "SLOTS is empty"
log "slots: ${#SLOT_ARR[@]}"

pids=(); i=0
for v in "${TODO[@]}"; do
  slot="${SLOT_ARR[$((i % ${#SLOT_ARR[@]}))]}"
  job="${slot%%:*}"; gpu="${slot##*:}"
  y="$YAML_DIR/_gen_${v}.yaml"
  # TAG in the name, not just the version: the pooled-LOO arms are both called
  # "pooled" (out/cells/po1_9b/pooled and po6_9b/pooled), so a tag-less name
  # interleaves two concurrent runs into one unreadable file and makes a
  # failure impossible to attribute.
  vlog="$LOG_DIR/vertrain_${TAG}_${v}.log"
  log "  [$v] job=$job gpu=$gpu -> $vlog"
  srun --jobid="$job" --overlap --nodes=1 --ntasks=1 --cpus-per-task="$CPUS" \
    bash -c "export PATH=$SPACK_GCC:$ENV_BIN:\$PATH; export CUDA_HOME=$SPACK_CUDA; \
             export CUDA_VISIBLE_DEVICES=$gpu; export FORCE_TORCHRUN=1; \
             export NPROC_PER_NODE=1; export DISABLE_VERSION_CHECK=1; \
             export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
             cd $REPO/LLaMA-Factory && $LMF train $y" \
    > "$vlog" 2>&1 </dev/null &
  pids+=($!); i=$((i+1))
  sleep "$STAGGER"
done

log "launched ${#pids[@]} training job(s); waiting"
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
log "done; $fail job(s) returned non-zero"

log ""
log "--- adapters produced"
for v in $VERSIONS; do
  d="$PROJECT/out/cells/$TAG/$v"
  if [ -f "$d/adapter_config.json" ]; then
    log "  $($PY -m adaptercl.materialize 2>/dev/null >/dev/null; $PY -c "
import sys;sys.path.insert(0,'$PROJECT')
from adaptercl import materialize
print(materialize.describe('$d'))" 2>/dev/null || echo "$v ok")"
  else
    log "  $v MISSING -- see $LOG_DIR/vertrain_${TAG}_${v}.log"
  fi
done
log "log: $LOG"
