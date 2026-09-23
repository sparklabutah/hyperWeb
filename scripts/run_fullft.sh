#!/usr/bin/env bash
# run_fullft.sh -- one FULL fine-tuning run (tag, version) on a given allocation.
#
# Full FT is memory-bound in a way LoRA is not: bf16 weights + fp32 master +
# Adam moments is ~14x the parameter count in bytes before a single activation,
# which is why these YAMLs use ZeRO-3 WITH CPU offload. At cutoff_len 65536 the
# activations still dominate, so each run wants several GPUs.
#
#   TAG=ft4b V=v3 J=<jobid> G=0,1 bash scripts/run_fullft.sh
set -uo pipefail
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; REPO="$(dirname "$PROJECT")"
cd "$PROJECT"; export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
TAG="${TAG:?}"; V="${V:?}"; J="${J:?}"; G="${G:?}"
NG="$(echo "$G" | tr ',' '\n' | grep -c .)"
CPUS="${CPUS:-7}"
ENVS="${ADAPTERCL_ENVS_ROOT:-${HOME}/miniconda3/envs}"
ENV_BIN="$ENVS/llamafactory/bin"; PY=/usr/bin/python3
SPACK_GCC="$($PY -c "import sys;sys.path.insert(0,'$PROJECT');from adaptercl import paths;print(paths.SPACK_GCC_BIN)")"
SPACK_CUDA="$($PY -c "import sys;sys.path.insert(0,'$PROJECT');from adaptercl import paths;print(paths.SPACK_CUDA_HOME)")"
LOG="$PROJECT/out/logs/fullft_${TAG}_${V}.log"
CLAIM_DIR="$PROJECT/out/logs/gpu_claims"; mkdir -p "$CLAIM_DIR"
NODE="$(squeue -u "$USER" -h -o "%i %R" | awk -v j="$J" '$1==j{print $2}' | head -1)"
if [ -n "$NODE" ]; then
  for g in $(echo "$G" | tr ',' ' '); do echo $$ > "$CLAIM_DIR/$NODE.$g"; done
  trap 'for g in $(echo "$G" | tr "," " "); do rm -f "$CLAIM_DIR/$NODE.$g"; done' EXIT
fi
Y="$PROJECT/out/cells/$TAG/yaml/_gen_${V}.yaml"
OUT="$PROJECT/out/fullft/${TAG}_${V}"
[ -f "$Y" ] || { echo "missing $Y"; exit 1; }
[ -f "$OUT/config.json" ] && { echo "$TAG $V already trained"; exit 0; }
echo "$(date '+%F %H:%M:%S') START $TAG $V on ${NODE:-?} gpus=$G (${NG}-way)" | tee -a "$LOG"
srun --jobid="$J" --overlap --nodes=1 --ntasks=1 --cpus-per-task="$(( CPUS * NG ))" \
  bash -c "export PATH=$SPACK_GCC:$ENV_BIN:\$PATH; export CUDA_HOME=$SPACK_CUDA; \
           export FORCE_TORCHRUN=1 NPROC_PER_NODE=$NG CUDA_VISIBLE_DEVICES=$G; \
           export DISABLE_VERSION_CHECK=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
           cd $REPO/LLaMA-Factory && $ENV_BIN/python -m llamafactory.cli train $Y" \
  >> "$LOG" 2>&1 </dev/null
rc=$?
for t in 1 2 3 4 5 6; do [ -f "$OUT/config.json" ] && break; sleep 5; done
ok=NO; [ -f "$OUT/config.json" ] && ok=yes
echo "$(date '+%F %H:%M:%S') DONE  $TAG $V rc=$rc model=$ok" | tee -a "$LOG"
[ "$ok" = "NO" ] && grep -aE "CUDA out of memory|Error|Traceback" "$LOG" | tail -4
