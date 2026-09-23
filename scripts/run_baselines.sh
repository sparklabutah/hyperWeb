#!/usr/bin/env bash
# run_baselines.sh -- adapterCL.md 6.4's Tier-1 baselines, "the paper is not
# reviewable without these".
#
# Four of them, all thin wrappers over the same eval bridge so every number
# lands in one table with everything else (out/eval/baselines.csv, one row per
# (baseline, cell) through evalbridge.write_row):
#
#   frozen        Qwen3.5-9B zero-shot, no adapter                      (6.4.1)
#   multiversion  ONE LoRA trained on the split's training cells POOLED (6.4.4)
#                 -- and this is also the stand-in for 6.4.3's "TimeTraj
#                 multi-version BC, retrained on your split", which is the
#                 number Phase 3's kill criterion is stated against. It is a
#                 stand-in, not the published system; say so in the paper.
#   nearest       reuse the per-cell adapter of the VISUALLY most similar
#                 training cell (6.4.5). Similarity comes from
#                 out/style/era_style.json -- the same signature the 6.2
#                 precondition was checked with -- and NOT from adapter weight
#                 distance, which 4.4 forbids (BA = (BR)(R^-1 A), so distances
#                 between independently trained adapters measure symmetry-group
#                 arbitrariness). NN_MATRIX picks which view; see below.
#   oracle        the held-out cell's OWN per-cell adapter (6.4.7). A ceiling,
#                 never a method; it is also the diagonal of the 6.2 matrix.
#
# Not covered here, and deliberately: 6.4.2 (TimeWarp-BC on the training
# versions) is the incumbent's own training protocol, 6.4.6 (mixture of trained
# adapters with learned coefficients) is a Phase-2 model rather than a wrapper
# around an eval, and Tier 2 (LoRA merging, in-context adaptation) is later.
# The report prints this list so the gap is visible rather than assumed.
#
# ---------------------------------------------------------------- invoking ---
# NOTHING LAUNCHES BY DEFAULT: you get the unit list, the exact commands, and a
# cost. Launching needs GO=1 plus SLOTS= naming an allocation you already hold
# (srun --overlap; never sbatch, never salloc).
#
#   bash scripts/run_baselines.sh                          # plan + report
#   GO=1 STAGE=corpus bash scripts/run_baselines.sh        # pooled BC corpus
#   GO=1 STAGE=train SLOTS="1724294:0" bash scripts/run_baselines.sh
#   GO=1 STAGE=eval  SLOTS="1724294:0 1724294:1 1724294:2 1724294:3" \
#       bash scripts/run_baselines.sh
#   STAGE=report bash scripts/run_baselines.sh
#
# The three adapter-bearing baselines are gated on 4.5's adapter-application
# artifact (out/eval/adapter_verification.json), because an ignored adapter
# makes them silently identical to the frozen row -- which would read as "the
# method does not help" while measuring nothing. `frozen` needs no adapter and
# is never gated.
#
# Idempotent: a cell whose eval dir already holds result_df*.csv is skipped.
#
# Knobs: SPLIT TRAIN_CELLS EVAL_CELLS BASELINES TAG PERCELL_TAG TARGET_SET RANK
# EPOCHS GRAD_ACCUM N_REPEATS DETERMINISTIC JUDGE_PROVIDER JUDGE_MODEL
# NN_MATRIX CPUS_TRAIN CPUS_EVAL STAGGER RUN_DIR SHOW_MAX ALLOW_UNVERIFIED.

set -uo pipefail
ulimit -Sn 131072 2>/dev/null || true    # Playwright fd leak (findings.md:258-262)

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(dirname "$PROJECT")"
ENVS="${ADAPTERCL_ENVS_ROOT:-${HOME}/miniconda3/envs}"
PY=/usr/bin/python3

STAGE="${STAGE:-all}"                    # all|corpus|train|eval|report
GO="${GO:-0}"
SLOTS="${SLOTS:-}"

SPLIT="${SPLIT:-forward}"                # 6.1's headline fold
TRAIN_CELLS="${TRAIN_CELLS:-}"           # default: the split's train cells
EVAL_CELLS="${EVAL_CELLS:-}"             # default: the split's held-out cells
BASELINES="${BASELINES:-frozen multiversion nearest oracle}"
TAG="${TAG:-base}"
PERCELL_TAG="${PERCELL_TAG:-percell}"    # where Phase 2's per-cell adapters live
TARGET_SET="${TARGET_SET:-attn_mlp}"
RANK="${RANK:-16}"
EPOCHS="${EPOCHS:-3}"                    # pooled corpus is ~9x a cell's size
GRAD_ACCUM="${GRAD_ACCUM:-4}"
# era_style.json's similarity matrices. `raw` is the default because 6.4.5's
# baseline is a DEPLOYMENT-TIME lookup -- "which training version does this page
# look most like" -- and raw cosine over the style signature is that question.
# `env` (per-site mean removed) and `grand` are the analysis views 6.2 uses to
# test correspondence with site identity taken out; they systematically pick
# cross-site neighbours, which makes a weaker baseline for a reason that has
# nothing to do with the method.
NN_MATRIX="${NN_MATRIX:-raw}"            # raw|env|grand

N_REPEATS="${N_REPEATS:-1}"
DETERMINISTIC="${DETERMINISTIC:-0}"      # baselines sit next to headline numbers
JUDGE_PROVIDER="${JUDGE_PROVIDER:-openai}"
JUDGE_MODEL="${JUDGE_MODEL:-}"
ALLOW_UNVERIFIED="${ALLOW_UNVERIFIED:-0}"
CPUS_TRAIN="${CPUS_TRAIN:-16}"
CPUS_EVAL="${CPUS_EVAL:-8}"
STAGGER="${STAGGER:-45}"
SHOW_MAX="${SHOW_MAX:-3}"
SEC_PER_EPISODE="${SEC_PER_EPISODE:-120}"

RUN_DIR="${RUN_DIR:-$PROJECT/out/baselines/$TAG}"
UNITS="$RUN_DIR/units.tsv"
TODO="$RUN_DIR/eval.todo"
MV_DIR="$PROJECT/out/cells/${TAG}_multiversion_${SPLIT}"
MV_CORPUS="$PROJECT/out/corpus/${TAG}/multiversion_${SPLIT}.json"
MV_YAML="$RUN_DIR/_gen_multiversion_${SPLIT}.yaml"
RESULTS_CSV="$PROJECT/out/eval/baselines.csv"
VERIFICATION="$PROJECT/out/eval/adapter_verification.json"

LOG_DIR="$PROJECT/out/logs"
mkdir -p "$LOG_DIR" "$RUN_DIR" "$RUN_DIR/eval" "$RUN_DIR/logs" "$PROJECT/out/corpus/$TAG"
LOG="$LOG_DIR/baselines_$(date +%Y%m%d-%H%M%S).log"
log(){ echo "[baselines $(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
die(){ log "BLOCKED: $*"; log "log: $LOG"; exit 1; }

cd "$PROJECT"
export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"

want(){ [ "$STAGE" = "all" ] || [ "$STAGE" = "$1" ]; }

# Split membership, resolved once.
if [ -z "$TRAIN_CELLS" ]; then
  TRAIN_CELLS=$("$PY" -c "import sys;sys.path.insert(0,'$PROJECT');from adaptercl import cells;print(','.join(c.key for c in cells.split_by_name('$SPLIT').train_cells()))")
fi
if [ -z "$EVAL_CELLS" ]; then
  EVAL_CELLS=$("$PY" -c "import sys;sys.path.insert(0,'$PROJECT');from adaptercl import cells;print(','.join(c.key for c in cells.split_by_name('$SPLIT').test_cells()))")
fi

log "=== adapterCL 6.4 Tier-1 baselines ==="
log "stage=$STAGE go=$GO split=$SPLIT baselines=[$BASELINES]"
log "train cells: $TRAIN_CELLS"
log "eval  cells: $EVAL_CELLS"
log "run_dir=$RUN_DIR  csv=$RESULTS_CSV  log=$LOG"

# --------------------------------------------------------------------------
# [1] Unit list: one (baseline, cell) per row
# --------------------------------------------------------------------------
log ""
log "--- [1/5] units"
ADAPTERCL_UNITS="$UNITS" ADAPTERCL_TRAIN="$TRAIN_CELLS" ADAPTERCL_EVAL="$EVAL_CELLS" \
ADAPTERCL_BASELINES="$BASELINES" ADAPTERCL_PERCELL_TAG="$PERCELL_TAG" \
ADAPTERCL_MV_DIR="$MV_DIR" ADAPTERCL_ROOT="$RUN_DIR/eval" ADAPTERCL_NN="$NN_MATRIX" \
"$PY" - <<'PY' 2>&1 | tee -a "$LOG"
import json, os, sys
sys.path.insert(0, os.environ["PYTHONPATH"].split(os.pathsep)[0])
from adaptercl import cells as cells_mod
from adaptercl import materialize, paths

train = [c for c in os.environ["ADAPTERCL_TRAIN"].split(",") if c]
evalc = [c for c in os.environ["ADAPTERCL_EVAL"].split(",") if c]
kinds = os.environ["ADAPTERCL_BASELINES"].split()
tag = os.environ["ADAPTERCL_PERCELL_TAG"]
root = os.environ["ADAPTERCL_ROOT"]

# 6.4.5's "visually most similar training version". era_style's env-centred
# matrix removes the per-site mean, so the neighbour is chosen on STYLE rather
# than on "which site is this", which is exactly the confound 6.2 is about.
nn, nn_note = {}, ""
style = os.path.join(paths.OUT_STYLE, "era_style.json")
if os.path.exists(style):
    rep = json.load(open(style))
    mat = rep["matrices"][os.environ["ADAPTERCL_NN"]]
    keys, M = mat["keys"], mat["matrix"]
    idx = dict((k, i) for i, k in enumerate(keys))
    for c in evalc:
        cands = [(M[idx[c]][idx[t]], t) for t in train if t != c]
        if cands:
            s, t = max(cands)
            nn[c] = (t, s)
    nn_note = "era_style.json matrices[%s]" % os.environ["ADAPTERCL_NN"]
else:
    # Fallback: nearest nominal year inside the same environment. Stated in the
    # log so a run can never present it as the visual-similarity baseline.
    for c in evalc:
        cc = cells_mod.Cell.parse(c)
        cands = []
        for t in train:
            tc = cells_mod.Cell.parse(t)
            if tc.env != cc.env or tc.year is None or cc.year is None:
                continue
            cands.append((-abs(tc.year - cc.year), t))
        if cands:
            s, t = max(cands)
            nn[c] = (t, s)
    nn_note = ("FALLBACK nearest nominal YEAR within the environment -- "
               "out/style/era_style.json is missing, so this is NOT the visual "
               "similarity 6.4.5 asks for. Run `python3 -m adaptercl style`.")

rows = []
for kind in kinds:
    for c in evalc:
        if kind == "frozen":
            ad, name, note = "-", "-", "no adapter"
        elif kind == "multiversion":
            ad, name, note = os.environ["ADAPTERCL_MV_DIR"], "multiversion", \
                             "pooled over %d training cells" % len(train)
        elif kind == "nearest":
            if c not in nn:
                print("  skip nearest/%s: no training neighbour" % c)
                continue
            t, s = nn[c]
            ad = materialize.cell_adapter_dir(cells_mod.Cell.parse(t), tag=tag)
            name = "nn_%s" % t
            note = "neighbour=%s sim=%.4f" % (t, s)
        elif kind == "oracle":
            ad = materialize.cell_adapter_dir(cells_mod.Cell.parse(c), tag=tag)
            name = "oracle_%s" % c
            note = "the cell's OWN adapter -- ceiling only (6.4.7)"
        else:
            print("  unknown baseline %r, skipped" % kind)
            continue
        unit = "%s__%s" % (kind, c)
        rows.append((unit, kind, c, ad, name, os.path.join(root, unit), note))

with open(os.environ["ADAPTERCL_UNITS"], "w") as fh:
    fh.write("\t".join(["unit", "baseline", "cell", "adapter_dir", "lora_name",
                        "eval_dir", "note"]) + "\n")
    for r in rows:
        fh.write("\t".join(r) + "\n")

print("  nearest-neighbour source: %s" % nn_note)
for c, (t, s) in sorted(nn.items()):
    print("    %-9s -> %-9s (%.4f)" % (c, t, s))
print("  %d unit(s) -> %s" % (len(rows), os.environ["ADAPTERCL_UNITS"]))
PY

N_UNITS=$(( $(wc -l < "$UNITS") - 1 ))
N_EPISODES=$("$PY" -c "
import sys;sys.path.insert(0,'$PROJECT')
from adaptercl import cells
tot=0
for i,line in enumerate(open('$UNITS')):
    if i==0: continue
    c=line.split('\t')[2]
    tot+=len(cells.tasks_for_env(c.split('_')[0], split='test'))
print(tot*$N_REPEATS)")
log "units: $N_UNITS   cost if all run: ~$N_EPISODES episodes = ~$(( N_EPISODES * SEC_PER_EPISODE / 3600 )) GPU-hours serial"

# --------------------------------------------------------------------------
# [2] The pooled multi-version corpus (6.4.4 / the 6.4.3 stand-in)
# --------------------------------------------------------------------------
if want corpus && printf '%s' "$BASELINES" | grep -qw multiversion; then
  log ""
  log "--- [2/5] pooled BC corpus over the training cells"
  if [ -f "$MV_CORPUS" ]; then
    log "  exists: $MV_CORPUS"
  elif [ "$GO" != "1" ]; then
    log "  NOT BUILDING (need GO=1). Would run:"
    log "    $PY -m adaptercl census sharegpt $TRAIN_CELLS -o $MV_CORPUS"
  else
    "$PY" -m adaptercl census sharegpt "$TRAIN_CELLS" -o "$MV_CORPUS" 2>&1 | tee -a "$LOG"
    [ -f "$MV_CORPUS" ] || log "  WARNING: the corpus was not written"
  fi
fi

# --------------------------------------------------------------------------
# [3] Train the single multi-version LoRA
# --------------------------------------------------------------------------
if want train && printf '%s' "$BASELINES" | grep -qw multiversion; then
  log ""
  log "--- [3/5] the single multi-version LoRA (training cells pooled)"
  if [ -f "$MV_DIR/adapter_config.json" ]; then
    log "  exists: $MV_DIR"
  elif [ ! -f "$MV_CORPUS" ]; then
    log "  NOT TRAINING: the pooled corpus is missing ($MV_CORPUS) -- STAGE=corpus first."
  else
    MV_CMD=$(ADAPTERCL_YAML="$MV_YAML" ADAPTERCL_OUT="$MV_DIR" ADAPTERCL_CORPUS="$MV_CORPUS" \
      ADAPTERCL_TS="$TARGET_SET" ADAPTERCL_RANK="$RANK" ADAPTERCL_EPOCHS="$EPOCHS" \
      ADAPTERCL_GA="$GRAD_ACCUM" ADAPTERCL_TRAIN="$TRAIN_CELLS" \
      "$PY" - <<'PY'
import json, os, sys
sys.path.insert(0, os.environ["PYTHONPATH"].split(os.pathsep)[0])
from adaptercl import bcdata, cells, percell

corpus = os.environ["ADAPTERCL_CORPUS"]
stats_path = corpus[:-5] + ".stats.json" if corpus.endswith(".json") else corpus + ".stats.json"
n_samples, dataset = 0, "adaptercl_" + os.path.basename(corpus)[:-5]
if os.path.exists(stats_path):
    st = json.load(open(stats_path))
    n_samples = int(st.get("n_samples") or 0)
    dataset = st.get("dataset_name") or dataset

# write_cell_yaml wants a Cell for the header comment only; the dataset and the
# output dir are what actually differ from a per-cell run.
rep = cells.Cell.parse(os.environ["ADAPTERCL_TRAIN"].split(",")[0])
path = percell.write_cell_yaml(
    rep, dataset, target_set=os.environ["ADAPTERCL_TS"],
    rank=int(os.environ["ADAPTERCL_RANK"]),
    out_dir=os.environ["ADAPTERCL_OUT"], yaml_path=os.environ["ADAPTERCL_YAML"],
    epochs=float(os.environ["ADAPTERCL_EPOCHS"]),
    grad_accum=int(os.environ["ADAPTERCL_GA"]), n_samples=n_samples)
pre = percell.env_preamble(["export NPROC_PER_NODE=1"])
cmd = "; ".join(pre + ['cd "%s"' % __import__("adaptercl").paths.LLAMA_FACTORY,
                       '"%s" train "%s"' % (__import__("adaptercl").paths.LMF, path)])
sys.stderr.write("  yaml: %s  dataset=%s  samples=%d\n" % (path, dataset, n_samples))
print(cmd)
PY
)
    log "  $MV_CMD"
    if [ "$GO" != "1" ] || [ -z "$SLOTS" ]; then
      log "  NOT LAUNCHING (need GO=1 and SLOTS=\"<jobid>:<gpu> ...\")."
    else
      set -- $SLOTS; IFS=: read -r JOB GPU _ <<<"$1"
      log "  training on job $JOB gpu $GPU -> $MV_DIR"
      srun --jobid="$JOB" --overlap --nodes=1 --ntasks=1 --cpus-per-task="$CPUS_TRAIN" \
        bash -c "export CUDA_VISIBLE_DEVICES=$GPU; $MV_CMD" \
        > "$RUN_DIR/logs/train_multiversion.log" 2>&1 </dev/null
      [ -f "$MV_DIR/adapter_config.json" ] && log "  multiversion LoRA OK" \
        || log "  multiversion LoRA FAILED (see $RUN_DIR/logs/train_multiversion.log)"
    fi
  fi
fi

# --------------------------------------------------------------------------
# [4] Evaluate every unit
# --------------------------------------------------------------------------
declare -a SLOT_JOB=() SLOT_GPU=()
parse_slots(){
  SLOT_JOB=(); SLOT_GPU=()
  local s job gpu pin
  for s in $SLOTS; do
    IFS=: read -r job gpu pin <<<"$s"
    [ -n "${job:-}" ] && [ -n "${gpu:-}" ] || die "bad slot '$s': want <jobid>:<gpu>"
    SLOT_JOB+=("$job"); SLOT_GPU+=("$gpu")
  done
  [ "${#SLOT_JOB[@]}" -gt 0 ] || die "SLOTS is empty"
}

eval_cmd_for(){   # $1 unit, $2 port index, $3 gpu
  local u="$1" pi="$2" gpu="$3" line kind cell ad name ev args
  line=$(awk -F'\t' -v n="$u" '$1==n {print; exit}' "$UNITS")
  kind=$(printf '%s' "$line" | cut -f2); cell=$(printf '%s' "$line" | cut -f3)
  ad=$(printf '%s' "$line" | cut -f4);   name=$(printf '%s' "$line" | cut -f5)
  ev=$(printf '%s' "$line" | cut -f6)
  if [ "$kind" = "frozen" ]; then args="--cell $cell"
  else args="--cell $cell --adapter-dir $ad --lora-name $name"; fi
  printf '%s -m adaptercl eval plan %s --n-repeats %s --port-index %s --cuda-devices %s --output-root %s --label %s --judge-provider %s%s%s' \
    "$PY" "$args" "$N_REPEATS" "$pi" "$gpu" "$ev" "bl_$u" "$JUDGE_PROVIDER" \
    "$( [ -n "$JUDGE_MODEL" ] && printf ' --judge-model %s' "$JUDGE_MODEL" )" \
    "$( [ "$DETERMINISTIC" = "1" ] && printf ' --deterministic-judge' )"
}

if want eval; then
  log ""
  log "--- [4/5] evaluate"
  # 4.5's gate, using the same predicate `adaptercl status` uses so the two can
  # never disagree. `frozen` is exempt: it serves no adapter.
  GATE_OK=1; GATE_WHY="ok"
  if [ ! -f "$VERIFICATION" ]; then
    GATE_OK=0; GATE_WHY="$VERIFICATION is absent"
  else
    GATE_OK=$("$PY" -c "
import sys; sys.path.insert(0, '$PROJECT')
from adaptercl.cli import adapter_verification_state
passed, detail = adapter_verification_state()
print('1' if passed else '0')
print(detail)
" 2>/dev/null | sed -n 1p)
    GATE_WHY=$("$PY" -c "
import sys; sys.path.insert(0, '$PROJECT')
from adaptercl.cli import adapter_verification_state
print(adapter_verification_state()[1])
" 2>/dev/null || echo unreadable)
    [ -n "$GATE_OK" ] || GATE_OK=0
  fi
  if [ "$GATE_OK" != "1" ]; then
    log "GATE (4.5): adapter application is NOT verified ($GATE_WHY)."
    log "  An ignored adapter reads exactly like the frozen row, so the three"
    log "  adapter baselines would silently measure nothing. Fix with:"
    log "    VERIFY=1 GPU_JOB=<jobid> GPU=0 bash scripts/run_phase0.sh"
    if [ "$ALLOW_UNVERIFIED" = "1" ]; then
      log "  ALLOW_UNVERIFIED=1 -- running them anyway; the rows are UNTRUSTED."
      GATE_OK=1
    else
      log "  'frozen' will still run; the adapter baselines are held back."
    fi
  fi

  : > "$TODO"
  while IFS=$'\t' read -r unit kind cell ad name ev note; do
    [ "$unit" = "unit" ] && continue
    if find "$ev" -name 'result_df*.csv' 2>/dev/null | grep -q .; then
      log "  skip $unit (eval exists)"
    elif [ "$kind" != "frozen" ] && [ ! -f "$ad/adapter_config.json" ]; then
      log "  skip $unit (no adapter at $ad)"
    elif [ "$kind" != "frozen" ] && [ "$GATE_OK" != "1" ]; then
      log "  hold $unit (4.5 gate)"
    else
      echo "$unit" >> "$TODO"
    fi
  done < "$UNITS"
  N_TODO=$(wc -l < "$TODO" 2>/dev/null || echo 0)
  log "units to evaluate: $N_TODO"

  if [ "$N_TODO" -eq 0 ]; then
    log "nothing to do"
  elif [ "$GO" != "1" ] || [ -z "$SLOTS" ]; then
    log "NOT LAUNCHING (need GO=1 and SLOTS=\"<jobid>:<gpu> ...\"). Would run:"
    i=0
    while read -r u; do
      [ "$i" -lt "$SHOW_MAX" ] && { log "  # $u"; echo "  $(eval_cmd_for "$u" "$i" "<gpu>") --launch" | tee -a "$LOG"; }
      i=$((i+1))
    done < "$TODO"
    [ "$N_TODO" -gt "$SHOW_MAX" ] && log "  (+$(( N_TODO - SHOW_MAX )) more; full list in $UNITS)"
  else
    parse_slots
    n=${#SLOT_JOB[@]}
    for ((k=0; k<n; k++)); do : > "$RUN_DIR/eval.slot${k}"; done
    i=0
    while read -r u; do echo "$u" >> "$RUN_DIR/eval.slot$(( i % n ))"; i=$((i+1)); done < "$TODO"
    pids=()
    for ((k=0; k<n; k++)); do
      [ -s "$RUN_DIR/eval.slot${k}" ] || continue
      (
        while read -r u; do
          ev=$(awk -F'\t' -v n="$u" '$1==n {print $6; exit}' "$UNITS")
          cmd="$(eval_cmd_for "$u" "$k" "${SLOT_GPU[$k]}") --launch"
          log "  [slot $k gpu ${SLOT_GPU[$k]} band $k] $u"
          srun --jobid="${SLOT_JOB[$k]}" --overlap --nodes=1 --ntasks=1 \
               --cpus-per-task="$CPUS_EVAL" \
               bash -c "cd $PROJECT && PYTHONPATH=$PROJECT $cmd" \
               > "$RUN_DIR/logs/eval_${u}.log" 2>&1 </dev/null
          # `eval plan --launch` exits 0 even when the driver fails.
          if find "$ev" -name 'result_df*.csv' 2>/dev/null | grep -q .; then
            log "  [slot $k] $u OK"
          else
            log "  [slot $k] $u produced NO result_df (see $RUN_DIR/logs/eval_${u}.log)"
          fi
        done < "$RUN_DIR/eval.slot${k}"
      ) &
      pids+=($!)
      sleep "$STAGGER"
    done
    log "launched ${#pids[@]} eval worker(s); waiting"
    for p in "${pids[@]}"; do wait "$p"; done
    log "eval stage done"
  fi
fi

# --------------------------------------------------------------------------
# [5] Report -- one CSV row per (baseline, cell), plus the published figures
# --------------------------------------------------------------------------
log ""
log "--- [5/5] report"
ADAPTERCL_UNITS="$UNITS" ADAPTERCL_CSV="$RESULTS_CSV" ADAPTERCL_SPLIT="$SPLIT" \
ADAPTERCL_TRAIN="$TRAIN_CELLS" \
"$PY" - <<'PY' 2>&1 | tee -a "$LOG"
import os, sys
sys.path.insert(0, os.environ["PYTHONPATH"].split(os.pathsep)[0])
from adaptercl import evalbridge, paths

units = []
with open(os.environ["ADAPTERCL_UNITS"]) as fh:
    head = fh.readline().rstrip("\n").split("\t")
    for line in fh:
        units.append(dict(zip(head, line.rstrip("\n").split("\t"))))

split = os.environ["ADAPTERCL_SPLIT"]
csv_path = os.environ["ADAPTERCL_CSV"]
by = {}
print("  %-22s %-9s %6s %9s %9s  %s"
      % ("baseline", "cell", "n", "success", "avg_steps", "note"))
print("  " + "-" * 92)
for u in units:
    parsed = evalbridge.read_results(u["eval_dir"])
    agg = parsed["aggregate"]
    print("  %-22s %-9s %6d %9s %9s  %s"
          % (u["baseline"], u["cell"], agg["n"],
             "-" if agg["success_rate"] is None else "%.3f" % agg["success_rate"],
             "-" if agg["avg_steps"] is None else "%.1f" % agg["avg_steps"],
             u["note"]))
    if not agg["n"]:
        continue
    by.setdefault(u["baseline"], []).append(agg["success_rate"])
    evalbridge.write_row(csv_path, {
        "baseline": u["baseline"], "cell": u["cell"], "split": split,
        "unit": u["unit"], "adapter": u["adapter_dir"],
        "lora_name": u["lora_name"], "n": agg["n"],
        "success_rate": agg["success_rate"], "avg_reward": agg["avg_reward"],
        "std_err": agg["std_err"], "avg_steps": agg["avg_steps"],
        "n_err": agg["n_err"], "note": u["note"], "status": parsed["status"],
        "output_root": u["eval_dir"]})

print("")
if by:
    print("  per-baseline mean over the cells that have episodes")
    print("  (6.6: report per environment and per era too -- NEVER means only):")
    for k in sorted(by):
        v = by[k]
        print("    %-14s %.3f over %d cell(s)" % (k, sum(v) / len(v), len(v)))
else:
    print("  no baseline has produced episodes yet.")

print("")
print("  6.4 Tier-1 items this script does NOT produce:")
print("    2. TimeWarp-BC on the training versions (the incumbent's own protocol)")
print("    3. TimeTraj multi-version BC as published -- 'multiversion' above is a")
print("       stand-in retrained on this split, which is what Phase 3's kill")
print("       criterion is evaluated against; label it as such in the paper")
print("    6. mixture of trained adapters with learned coefficients (6.5, Phase 2)")
print("  Tier 2 (LoRA merging, in-context adaptation) is not started.")
print("")
print("  rows -> %s" % csv_path)
PY

log ""
log "log: $LOG"
