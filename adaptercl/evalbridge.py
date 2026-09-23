"""Programmatic interface to the repo's eval harness.

`run_v6_eval_q35_sonnet.sh` is already fully parameterised by environment
variables, so adapterCL treats it as a black box rather than reimplementing the
flask-app startup, the vLLM lifecycle, the judge wiring and the four operational
traps its comments record. This module builds the environment, builds the
command, and parses what comes back.

Call chain it drives (verified 2026-08-06):

    run_v6_eval_q35_sonnet.sh
      -> starts wiki/news/webshop flask apps      (:125-153, ENV_PORT_BASE)
      -> bash $VLM_SCRIPT --model M --port P      (:166, VLM_PORT_BASE)
      -> cd $OUTPUT_ROOT && $BENCH_PY $BENCH_SCRIPT --port P --version V --model M   (:233-239)
      -> $BENCH_PY $AGG_SCRIPT --vdir $OUTPUT_ROOT ...                               (:242)

    benchmarkGeneral.py -> make_study(...).run(n_jobs=EVAL_N_JOBS, "joblib")
      -> $OUTPUT_ROOT/results/<basename(model)>/<version>_<TW_RESULTS_TAG|all>/
           result_df_trial_<i>_of_<n>.csv
           summary_df_trial_<i>_of_<n>.csv
           <one dir per episode>/summary_info.json

Three things the driver does NOT set, which we therefore must (they are read
straight out of the inherited environment by the python it spawns):

    EVAL_N_JOBS     benchmarkGeneral.py:124   default 8
    TW_RESULTS_TAG  benchmarkGeneral.py:114   default "all"
    TW_TASK_DATA    browsergym/timewarp/task.py:61  default test.raw.json (LLM judge)

And one place the driver's default actively fights us: it passes
``EVAL_TASK_LIMIT="${EVAL_TASK_LIMIT:-309}"`` (:238), which would truncate a
train-split run (128 tasks x 3 = 384 experiments) to its first 309. `build_env`
therefore always sets EVAL_TASK_LIMIT explicitly; 0 means "no cap"
(benchmarkGeneral.py:100-101 maps <=0 to the full list).

Nothing here starts a GPU job. `run()` is dry-run by default and refuses to
launch unless `dry_run=False` is passed explicitly.

Pure stdlib -- importable from the system python (3.6+).
"""

from __future__ import print_function

import collections
import csv
import glob
import json
import math
import os
import random
import re
import subprocess
import sys

from . import cells, paths

# --------------------------------------------------------------------------
# Ports
# --------------------------------------------------------------------------

#: Chromium refuses these with ERR_UNSAFE_PORT regardless of what is listening.
#: Copied verbatim from run_v6_eval_q35_sonnet.sh:104. Getting this wrong once
#: cost 219/309 errored episodes and looked exactly like "the method does not
#: generalize".
UNSAFE_PORTS = (5060, 5061, 6000, 6566, 6665, 6666, 6667, 6668, 6669, 6697, 10080)

#: The env band needs three consecutive free ports (wiki/news/webshop, :136);
#: free_port() walks upward from the base, so we keep a little headroom.
ENV_BAND_WIDTH = 8
ENV_PORT_START = 5100
ENV_PORT_STRIDE = 100
VLM_PORT_START = 8100
VLM_PORT_STRIDE = 10


def _band_is_safe(base, width):
    for p in range(base, base + width):
        if p in UNSAFE_PORTS:
            return False
    return True


def port_bands(index, env_start=ENV_PORT_START, env_stride=ENV_PORT_STRIDE,
               vlm_start=VLM_PORT_START, vlm_stride=VLM_PORT_STRIDE):
    """Disjoint (ENV_PORT_BASE, VLM_PORT_BASE) for the `index`-th concurrent eval.

    The driver's `free_port` only tests "is anyone LISTENING", so two evals
    starting in the same second both see 5000-5002 free, both claim them, and
    the loser dies 180s later in wait_for_url (:131-134). Give each eval its own
    band and the race disappears.

    Bands that would collide with UNSAFE_PORTS are skipped, not shifted, so the
    mapping index -> band stays stable as more evals are added.
    """
    if index < 0:
        raise ValueError("port band index must be >= 0")
    env_base, taken = env_start, 0
    while True:
        if _band_is_safe(env_base, ENV_BAND_WIDTH):
            if taken == index:
                break
            taken += 1
        env_base += env_stride
    vlm_base, taken = vlm_start, 0
    while True:
        if _band_is_safe(vlm_base, 2):
            if taken == index:
                break
            taken += 1
        vlm_base += vlm_stride
    return {"ENV_PORT_BASE": env_base, "VLM_PORT_BASE": vlm_base}


# --------------------------------------------------------------------------
# Spec
# --------------------------------------------------------------------------

class EvalSpec(object):
    """One eval of one (model, adapter, cell) through run_v6_eval_q35_sonnet.sh.

    Parameters
    ----------
    model : checkpoint path or hub id. Becomes MODEL_PATH -- what the vLLM
        launcher serves and what names the results directory.
    adapter_dir : PEFT adapter directory to serve as a LoRA, or None for the
        base model. Implies LORA_MODULES and a served-name of `lora_name`.
    lora_name : the vLLM served-name for the adapter. Defaults to the cell key
        (or the adapter dir's basename).
    served_name : override for the OpenAI `model` field the agent sends. Almost
        never needed: it is `lora_name` when an adapter is set, `model`
        otherwise.
    cell : cells.Cell. Sets VERSION from `cell.era` and, unless `task_ids` says
        otherwise, restricts the run to the cell's single-site tasks.
    version : UI era to serve. Derived from `cell` when given.
    split : "test" (ids 1-103) or "train" (104-231). Train tasks carry
        `additional_instructions` that leak the answer -- only use them for the
        6.6 drift gap's train-side term, never as a headline number.
    task_ids : explicit task subset. `None` + `cell` -> the cell's tasks;
        `False` -> no filter at all (whole split).
    deterministic_judge : True selects TW_TASK_DATA=test.raw.v2.json, i.e.
        string/number/list matchers instead of the LLM judge (164/44/29, with 2
        stragglers still on llm_judge). Cheap and reproducible; use it for
        Phase-1/2 sweeps and keep the LLM judge for any number that goes in the
        paper next to a published TimeWarp figure.
    port_index : index into `port_bands`, for concurrent evals.
    """

    def __init__(self, model=None, adapter_dir=None, lora_name=None,
                 served_name=None, cell=None, version=None, split="test",
                 task_ids=None, single_site=True,
                 n_repeats=1, n_jobs=8, cuda_devices="0", port_index=0,
                 env_port_base=None, vlm_port_base=None,
                 judge_provider="openai", judge_model=None,
                 output_root=None, results_tag=None, results_dir=None,
                 deterministic_judge=False, task_limit=0, seed=None,
                 use_thinking=True, use_memory=True, use_plan=True,
                 max_lora_rank=None, max_loras=1,
                 allow_runtime_lora_updating=False, label=None,
                 extra_env=None):
        self.model = model or paths.model_dir_or_id()
        self.adapter_dir = os.path.abspath(adapter_dir) if adapter_dir else None
        self.cell = cell
        if version is None:
            version = cell.era if cell is not None else 6
        self.version = int(version)
        self.split = split
        self.single_site = single_site
        self._task_ids = task_ids
        self.n_repeats = int(n_repeats)
        self.n_jobs = int(n_jobs)
        self.cuda_devices = str(cuda_devices)
        self.port_index = int(port_index)
        bands = port_bands(self.port_index)
        self.env_port_base = int(env_port_base or bands["ENV_PORT_BASE"])
        self.vlm_port_base = int(vlm_port_base or bands["VLM_PORT_BASE"])
        self.judge_provider = judge_provider
        self.judge_model = judge_model
        self.deterministic_judge = bool(deterministic_judge)
        # TW_SEED: a real replicate seed. AgentLab's task-seed RNG is
        # hardcoded to RandomState(42), so without this N independent
        # runs at n_repeats=1 are byte-identical and any error bar
        # computed across them is zero by construction.
        self.seed = seed
        self.task_limit = int(task_limit)
        self.use_thinking = bool(use_thinking)
        self.use_memory = bool(use_memory)
        self.use_plan = bool(use_plan)
        self.max_loras = int(max_loras)
        self.allow_runtime_lora_updating = bool(allow_runtime_lora_updating)
        self.extra_env = dict(extra_env or {})

        if lora_name is None and self.adapter_dir:
            lora_name = (cell.key if cell is not None
                         else os.path.basename(self.adapter_dir.rstrip("/")))
        self.lora_name = lora_name
        self.served_name = served_name or self.lora_name or self.model

        self.label = label or self.default_label()
        self.output_root = os.path.abspath(
            output_root or os.path.join(paths.OUT_EVAL, self.label))
        self.results_tag = results_tag or self.label
        self.results_dir = results_dir

        if max_lora_rank is None and self.adapter_dir:
            max_lora_rank = self._rank_from_adapter()
        self.max_lora_rank = max_lora_rank

    # -- derived ----------------------------------------------------------
    @staticmethod
    def short_model_name(model):
        """A readable name for a checkpoint.

        paths.model_dir_or_id() resolves to an HF snapshot dir whose basename is
        a commit hash, which makes for unreadable labels and result dirs. Walk
        back to `models--Org--Name` when that is the shape.
        """
        p = str(model).rstrip("/")
        base = os.path.basename(p)
        parent = os.path.basename(os.path.dirname(p))
        if parent == "snapshots":
            slug = os.path.basename(os.path.dirname(os.path.dirname(p)))
            if slug.startswith("models--"):
                return slug[len("models--"):].split("--")[-1]
        return base

    def default_label(self):
        parts = [self.short_model_name(self.model)]
        if self.cell is not None:
            parts.append(self.cell.key)
        else:
            parts.append("v%d" % self.version)
        if self.adapter_dir:
            parts.append("ad-" + os.path.basename(self.adapter_dir.rstrip("/")))
        else:
            parts.append("base")
        if self.split != "test":
            parts.append(self.split)
        return "_".join(re.sub(r"[^A-Za-z0-9._-]+", "-", p) for p in parts)

    def _rank_from_adapter(self):
        cfg_path = os.path.join(self.adapter_dir, "adapter_config.json")
        if not os.path.exists(cfg_path):
            raise IOError(
                "%s has no adapter_config.json -- the launcher will refuse to "
                "start and, worse, a run that skipped that check would silently "
                "read as the base model (adapterCL.md 4.5)." % self.adapter_dir)
        with open(cfg_path) as fh:
            return int(json.load(fh)["r"])

    def task_ids(self):
        """The task-id subset for this run, or None for 'whole split'."""
        if self._task_ids is False:
            return None
        if self._task_ids is not None:
            return sorted(int(t) for t in self._task_ids)
        if self.cell is None:
            return None
        return cells.tasks_for_env(self.cell.env, split=self.split,
                                   single_site=self.single_site)

    def as_dict(self):
        d = collections.OrderedDict()
        for k in ("model", "adapter_dir", "lora_name", "served_name", "version",
                  "split", "n_repeats", "n_jobs", "cuda_devices", "port_index",
                  "env_port_base", "vlm_port_base", "judge_provider",
                  "judge_model", "deterministic_judge", "task_limit", "seed",
                  "use_thinking", "use_memory", "use_plan", "max_lora_rank",
                  "max_loras", "label", "output_root", "results_tag"):
            d[k] = getattr(self, k)
        d["cell"] = self.cell.key if self.cell is not None else None
        # scout_plan.md Part E #3: an eval whose log does not say which extra
        # env vars it ran with cannot be audited after the fact -- and the site
        # manual (TW_SITE_MANUAL) changes the PROMPT, so it is exactly the kind
        # of thing that must not be invisible.
        d["extra_env"] = ",".join("%s=%s" % (k, v)
                                  for k, v in sorted(self.extra_env.items())) or None
        ids = self.task_ids()
        d["n_tasks"] = len(ids) if ids else None
        return d

    def __repr__(self):
        return "EvalSpec(%s)" % self.label


# --------------------------------------------------------------------------
# Environment / command
# --------------------------------------------------------------------------

def build_env(spec):
    """The env var dict to hand run_v6_eval_q35_sonnet.sh.

    Every name here was read off the driver (or off the python it spawns); see
    the module docstring for the three the driver does not itself set.
    """
    env = collections.OrderedDict()

    # -- driver knobs (run_v6_eval_q35_sonnet.sh line numbers) -------------
    env["MODEL_PATH"] = str(spec.model)                       # :25
    env["OUTPUT_ROOT"] = spec.output_root                     # :26 (required)
    env["VERSION"] = str(spec.version)                        # :27
    env["RUN_TAG"] = spec.label                               # :28
    if getattr(spec, "seed", None) is not None:
        env["TW_SEED"] = str(spec.seed)
    env["TW_USE_THINKING"] = "1" if spec.use_thinking else "0"   # :31
    env["TW_USE_MEMORY"] = "1" if spec.use_memory else "0"       # :32
    env["TW_USE_PLAN"] = "1" if spec.use_plan else "0"           # :33
    env["EVAL_N_REPEATS"] = str(spec.n_repeats)               # :36
    env["EVAL_CUDA_DEVICES"] = spec.cuda_devices              # :38
    env["TW_JUDGE_PROVIDER"] = spec.judge_provider            # :48
    if spec.judge_model:
        env["JUDGE_MODEL"] = spec.judge_model                 # :70 / :76
    env["ENV_PORT_BASE"] = str(spec.env_port_base)            # :135
    env["VLM_PORT_BASE"] = str(spec.vlm_port_base)            # :230
    # :238 forces 309 when unset, which would truncate a 384-experiment train
    # split. 0 == no cap (benchmarkGeneral.py:100-101).
    env["EVAL_TASK_LIMIT"] = str(spec.task_limit)

    # -- our replacements for the three overridable script hooks -----------
    env["VLM_SCRIPT"] = paths.VLM_LORA_LAUNCHER               # :56
    env["BENCH_SCRIPT"] = paths.BENCH_ADAPTER                 # :57
    env["AGG_SCRIPT"] = paths.AGGREGATE                       # :58 (unchanged)

    # -- read by the spawned python, never by the driver -------------------
    env["EVAL_N_JOBS"] = str(spec.n_jobs)                     # benchmarkGeneral.py:124
    env["TW_RESULTS_TAG"] = spec.results_tag                  # benchmarkGeneral.py:114
    if spec.deterministic_judge:
        env["TW_TASK_DATA"] = "test.raw.v2.json"              # timewarp/task.py:61

    # -- benchmark_adapter.py's additions ----------------------------------
    env["TW_SERVED_NAME"] = str(spec.served_name)
    env["TW_SPLIT"] = spec.split
    ids = spec.task_ids()
    if ids:
        env["TW_TASK_IDS"] = ",".join(str(i) for i in ids)
    if spec.results_dir:
        env["TW_RESULTS_DIR"] = spec.results_dir

    # -- startVLM_lora.sh --------------------------------------------------
    if spec.adapter_dir:
        env["LORA_MODULES"] = "%s=%s" % (spec.lora_name, spec.adapter_dir)
        env["MAX_LORA_RANK"] = str(spec.max_lora_rank or 16)
        env["MAX_LORAS"] = str(spec.max_loras)
    if spec.allow_runtime_lora_updating:
        env["ALLOW_RUNTIME_LORA_UPDATING"] = "1"

    env.update(spec.extra_env)
    return env


def build_command(spec, driver=None):
    """(argv, copy-pasteable string) to run this eval.

    argv is what `subprocess` wants; the string is what goes in a run log or a
    tmux window (STANDING ORDER: experiments run in tmux on an allocation, never
    on the login node).
    """
    driver = driver or paths.EVAL_DRIVER
    env = build_env(spec)
    argv = ["bash", driver]
    try:
        from shlex import quote
    except ImportError:                                  # pragma: no cover
        from pipes import quote
    parts = ["%s=%s" % (k, quote(str(v))) for k, v in env.items()]
    parts += ["bash", quote(driver)]
    return argv, " \\\n  ".join(parts)


def preflight(spec):
    """Everything that must be true before a launch. Returns a list of problems."""
    problems = []
    if not os.path.exists(paths.EVAL_DRIVER):
        problems.append("eval driver missing: %s" % paths.EVAL_DRIVER)
    if not os.path.exists(paths.VLM_LORA_LAUNCHER):
        problems.append("VLM_SCRIPT missing: %s" % paths.VLM_LORA_LAUNCHER)
    if not os.path.exists(paths.BENCH_ADAPTER):
        problems.append("BENCH_SCRIPT missing: %s" % paths.BENCH_ADAPTER)
    # A missing manual would not crash the bench script into a loud failure --
    # `benchmark_adapter.py` opens it, so the run would die 10 minutes in, after
    # the vLLM server was up. Catch it here, before anything is launched.
    manual = spec.extra_env.get("TW_SITE_MANUAL")
    if manual and not os.path.exists(manual):
        problems.append("TW_SITE_MANUAL does not exist: %s" % manual)
    if spec.adapter_dir:
        if not os.path.isdir(spec.adapter_dir):
            problems.append("adapter dir missing: %s" % spec.adapter_dir)
        elif not os.path.exists(os.path.join(spec.adapter_dir,
                                             "adapter_config.json")):
            problems.append("not a PEFT adapter dir (no adapter_config.json): %s"
                            % spec.adapter_dir)
        if spec.served_name == spec.model:
            problems.append(
                "adapter_dir is set but served_name == model, so the agent "
                "would query the BASE served-name and the adapter would do "
                "nothing (adapterCL.md 4.5).")
    if os.path.isdir(spec.output_root) and _existing_trials(spec.output_root):
        problems.append(
            "OUTPUT_ROOT %s already contains result_df_trial_*.csv. AgentLab "
            "resumes into a reused study dir and inflates n_completed; clear it "
            "first (rm -rf %s)." % (spec.output_root, spec.output_root))
    ids = spec.task_ids()
    if ids is not None and not ids:
        problems.append("task subset is empty for %r / split=%s"
                        % (spec.cell, spec.split))
    return problems


def _existing_trials(root):
    return sorted(glob.glob(os.path.join(root, "**", "result_df_trial_*.csv"),
                            recursive=True))


def run(spec, dry_run=True, driver=None, check_preflight=True, fh=None):
    """Print the command; only launch when `dry_run=False` is passed explicitly.

    This codebase must never start a GPU job as a side effect of being imported,
    inspected, or called with defaults -- hence the inverted default. Launch from
    a tmux window on a real allocation, not the login node.
    """
    fh = fh or sys.stdout
    argv, pretty = build_command(spec, driver=driver)
    problems = preflight(spec) if check_preflight else []

    print("=" * 72, file=fh)
    print("EvalSpec %s" % spec.label, file=fh)
    for k, v in spec.as_dict().items():
        print("  %-22s %s" % (k, v), file=fh)
    print("", file=fh)
    print(pretty, file=fh)
    print("", file=fh)
    if problems:
        print("PREFLIGHT PROBLEMS:", file=fh)
        for p in problems:
            print("  - %s" % p, file=fh)
    print("=" * 72, file=fh)

    if dry_run:
        return {"dry_run": True, "argv": argv, "command": pretty,
                "env": build_env(spec), "problems": problems,
                "output_root": spec.output_root}
    if problems:
        raise RuntimeError("refusing to launch with %d preflight problem(s):\n  %s"
                           % (len(problems), "\n  ".join(problems)))
    env = dict(os.environ)
    env.update(dict((k, str(v)) for k, v in build_env(spec).items()))
    if not os.path.isdir(spec.output_root):
        os.makedirs(spec.output_root)
    proc = subprocess.Popen(argv, env=env, cwd=paths.REPO)
    rc = proc.wait()
    return {"dry_run": False, "argv": argv, "command": pretty,
            "returncode": rc, "output_root": spec.output_root}


# --------------------------------------------------------------------------
# Reading results
# --------------------------------------------------------------------------

_TRIAL_RE = re.compile(r"result_df_trial_(\d+)_of_(\d+)\.csv$")
_TASK_RE = re.compile(r"timewarp\.(\d+)")
#: AgentLab prefixes every episode dir with the study's start timestamp, and
#: appends `_<k>` when a (task, seed) pair repeats within one study.
_TS_PREFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_")
_EPISODE_RE = re.compile(r"_on_timewarp\.(\d+)_(\d+)(?:_(\d+))?$")


def episode_key(exp_dir):
    """Retry-stable, repeat-distinguishing identity for one episode.

    `exp_dir` basenames look like

        2026-07-21_00-43-37_GenericAgentWithTraining-<model>_on_timewarp.3_10
        2026-07-21_00-43-37_GenericAgentWithTraining-<model>_on_timewarp.3_10_1

    Two independent facts make the obvious keys wrong, both verified on
    runs/exp1_all6_v1/results/qwen3-5-4b/1_all (309 episodes, 3 trial files):

    * **exp_dir is not stable across trials.** A retried episode gets a NEW
      timestamp prefix, so the same episode appears under 3 exp_dirs and naive
      exp_dir dedupe reports 315 episodes for a 309-episode study.
    * **(task_id, seed) is not unique within a trial.** AgentLab draws the
      EVAL_N_REPEATS seeds at random and they collide -- 9 collisions in 309
      here -- so keying on (task, seed) collapses 309 real episodes to 300.
      The colliding repeat is disambiguated by the `_1` dir suffix.

    Stripping only the timestamp gives exactly 309 keys, in every trial file.
    """
    return _TS_PREFIX_RE.sub("", os.path.basename(str(exp_dir).rstrip("/")))


def _task_seed_repeat(exp_dir, fallback_task=None, fallback_seed=-1):
    """(task_id, seed, repeat) parsed from an exp_dir basename.

    `repeat` is the `_<k>` suffix (0 when absent). Unlike `episode_key` this is
    portable ACROSS runs -- the basename also embeds the model name, so
    episode_key can only be compared within one study.
    """
    m = _EPISODE_RE.search(episode_key(exp_dir))
    if not m:
        return fallback_task, fallback_seed, 0
    return int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)


def _as_float(s, default=0.0):
    try:
        return float(s)
    except (TypeError, ValueError):
        return default


def _as_int(s, default=0):
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return default


def find_trial_csvs(output_root):
    """All result_df_trial_*.csv under a run root, oldest trial first.

    Sorted by (directory mtime, trial index) so a later trial's row for an
    episode overwrites an earlier one -- AgentLab re-dumps the FULL result frame
    after each retry round (verified: three 309-row trial files in
    runs/exp1_all6_v1/results/qwen3-5-4b/1_all/), so the last trial is the
    authoritative one.
    """
    # Targeted globs FIRST. The recursive "**" walk visits every file under the
    # run root -- ~10k+ once a run has 103 episode dirs -- and measured 120.8s
    # per unit on NFS, against 0.9s to actually parse the CSV it finds. At ~130
    # units that is four hours of pure directory walking. The layout is
    # <root>/results/<model-sha>/<n>_<tag>/result_df_trial_*.csv, so name the
    # depths explicitly and keep the recursive walk only as a fallback for
    # layouts these patterns miss.
    hits = []
    for pat in ("result_df_trial_*.csv",
                os.path.join("*", "result_df_trial_*.csv"),
                os.path.join("results", "*", "*", "result_df_trial_*.csv"),
                os.path.join("*", "*", "*", "result_df_trial_*.csv")):
        hits += glob.glob(os.path.join(output_root, pat))
    if not hits:
        hits = glob.glob(os.path.join(output_root, "**", "result_df_trial_*.csv"),
                         recursive=True)
    seen, out = set(), []
    for h in hits:
        h = os.path.abspath(h)
        if h in seen:
            continue
        seen.add(h)
        m = _TRIAL_RE.search(os.path.basename(h))
        idx = int(m.group(1)) if m else 0
        try:
            mtime = os.path.getmtime(os.path.dirname(h))
        except OSError:
            mtime = 0.0
        out.append((mtime, os.path.dirname(h), idx, h))
    out.sort()
    return [(d, i, p) for _mt, d, i, p in out]


def _read_summary_df(study_dir):
    """The harness's own aggregate, as a cross-check on ours."""
    cands = sorted(glob.glob(os.path.join(study_dir, "summary_df*.csv")))
    exact = [c for c in cands if os.path.basename(c) == "summary_df.csv"]
    chosen = exact or cands
    if not chosen:
        return None
    path = max(chosen, key=os.path.getmtime)
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return None
    r = dict(rows[-1])
    r["_path"] = path
    return r


def read_results(output_root):
    """Parse a finished (or partial) eval into tidy per-episode records.

    Returns
    -------
    {"records": [ {task_id, task_name, seed, reward, n_steps, exp_dir, err,
                   trial, study_dir, n_attempts} ],
     "aggregate": {n, avg_reward, std_err, solved, avg_steps, n_err,
                   answered, answer_rate, conditional_success},
     "summary_df": the harness's own summary row (or None),
     "study_dirs": [...], "status": "ok"|"empty"|"missing"}

    Episodes are keyed by `episode_key(exp_dir)` -- the exp_dir basename with the
    study timestamp stripped -- because neither exp_dir nor (task_id, seed) is a
    correct identity here; see `episode_key` for the two measurements that show
    it. The LAST trial's row wins, which is the completed retry. `exp_dir` is
    still on every record (it is how you reach that episode's
    summary_info.json) and `n_attempts` says how many exp_dirs it burned.

    Never raises on a missing or half-written run: a crashed eval must be
    reportable, not fatal. stdlib csv only -- no pandas in any of the three
    project envs is guaranteed.
    """
    out = {"output_root": output_root, "records": [], "study_dirs": [],
           "summary_df": None, "aggregate": empty_aggregate(),
           "status": "missing", "n_rows_read": 0, "n_retried_episodes": 0}
    if not os.path.isdir(output_root):
        return out
    trials = find_trial_csvs(output_root)
    if not trials:
        out["status"] = "empty"
        return out

    by_episode = collections.OrderedDict()
    attempts = collections.defaultdict(set)
    study_dirs = []
    n_rows = 0
    for study_dir, trial_idx, path in trials:
        if study_dir not in study_dirs:
            study_dirs.append(study_dir)
        try:
            with open(path) as fh:
                rows = list(csv.DictReader(fh))
        except (IOError, csv.Error):
            continue
        for row in rows:
            n_rows += 1
            exp_dir = (row.get("exp_dir") or "").strip()
            name = row.get("env.task_name") or row.get("exp_name") or ""
            m = _TASK_RE.search(name)
            task_id = int(m.group(1)) if m else None
            seed = _as_int(row.get("env.task_seed"), -1)
            if exp_dir:
                task_id, seed, repeat = _task_seed_repeat(exp_dir, task_id, seed)
                key = (study_dir, episode_key(exp_dir))
                attempts[key].add(exp_dir)
            else:
                # No exp_dir at all (a half-written frame). Fall back to
                # (task, seed) so a malformed row never merges with a good one.
                repeat = 0
                key = (study_dir, "_noexpdir", task_id, seed, n_rows)
            err = bool((row.get("err_msg") or "").strip())
            by_episode[key] = {
                "task_id": task_id,
                "task_name": name,
                "seed": seed,
                "repeat": repeat,
                "reward": _as_float(row.get("cum_reward"), 0.0),
                "raw_reward": _as_float(row.get("cum_raw_reward"), 0.0),
                "n_steps": _as_int(row.get("n_steps"), 0),
                # Whether the episode ENDED by answering. TimeWarp reads the
                # last assistant chat message and scores 0 if there is none
                # (browsergym/timewarp/task.py:193-221), so an agent that never
                # emits send_msg_to_user scores 0 however well it navigated.
                # Measured on the frozen Qwen3.5-9B baseline: 4/71 episodes
                # terminated, the rest truncated at the 30-step cap. Without
                # this column a gain from "learned to stop" is indistinguishable
                # from a gain from "learned the interface" -- which is exactly
                # what Phase 1's kill criterion would otherwise reward.
                "terminated": _as_bool(row.get("terminated")),
                "truncated": _as_bool(row.get("truncated")),
                "exp_dir": exp_dir,
                "err": err,
                "err_msg": (row.get("err_msg") or "").strip()[:400],
                "trial": trial_idx,
                "study_dir": study_dir,
            }

    for key, rec in by_episode.items():
        rec["n_attempts"] = len(attempts.get(key)) or 1

    records = list(by_episode.values())
    records.sort(key=lambda r: (r["task_id"] if r["task_id"] is not None else 1 << 30,
                                r["seed"], r["repeat"]))
    out["records"] = records
    out["study_dirs"] = study_dirs
    out["n_rows_read"] = n_rows
    out["n_retried_episodes"] = sum(1 for r in records if r["n_attempts"] > 1)
    out["aggregate"] = aggregate(records)
    for d in study_dirs:
        s = _read_summary_df(d)
        if s is not None:
            out["summary_df"] = s
            break
    out["status"] = "ok" if records else "empty"
    return out


def empty_aggregate():
    return {"n": 0, "avg_reward": None, "std_err": None, "solved": 0,
            "success_rate": None, "avg_steps": None, "n_err": 0,
            "answered": 0, "answer_rate": None, "conditional_success": None,
            "by_site": {}}


def _as_bool(v):
    """CSV round-trips booleans as 'True'/'False'/''/'1'/'0'."""
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    return str(v).strip().lower() in ("true", "1", "yes")


def site_groups(records, task_data=None):
    """{group: [record]} for group in wiki|news|shop|multi.

    The columns of the site transfer matrix. Every eval unit already runs the
    whole 103-task test split, so ONE unit yields the whole row of the matrix
    and this grouping is free -- it is post hoc, never a filter on the run
    (domain_transfer_timewarp_plan.md 1.1/2.2).

    A record whose task_id is not in the task list at all lands in `_unknown`
    rather than being dropped: a silently shrinking denominator is exactly how
    a per-group mean stops being comparable across arms.
    """
    gmap = cells.task_site_group_map(task_data=task_data)
    out = collections.OrderedDict((g, []) for g in cells.SITE_GROUPS)
    for r in records:
        g = gmap.get(r.get("task_id"))
        if g is None:
            out.setdefault("_unknown", []).append(r)
        else:
            out[g].append(r)
    return out


def aggregate_by_site(records, threshold=1.0, task_data=None):
    """{group: aggregate} -- success, answer_rate and conditional_success per
    site group, which is what the matrix's cells are made of.

    `answer_rate` is in every cell on purpose: a site gap that is entirely
    answer rate is a *protocol* gap, not a domain one, and reporting only
    success cannot tell them apart (the plan's 4th trap; §1 of adapterCL 6.6).
    """
    out = collections.OrderedDict()
    for g, recs in site_groups(records, task_data=task_data).items():
        out[g] = aggregate(recs, threshold=threshold, by_site=False)
    return out


def aggregate(records, threshold=1.0, by_site=True, task_data=None):
    """{n, avg_reward, std_err, solved, success_rate, avg_steps, n_err}.

    std_err is the SEM over episodes, matching the harness's own summary_df
    column, so the two can be compared directly.

    `by_site` adds the per-group breakdown under the "by_site" key (the site
    transfer matrix's row). It is on by default because it is cheap and because
    an aggregate that lacks it silently downgrades every consumer to the pooled
    number; pass by_site=False for the inner recursion and for callers that only
    want the scalar.
    """
    n = len(records)
    if n == 0:
        return empty_aggregate()
    rewards = [r["reward"] for r in records]
    mean = sum(rewards) / float(n)
    if n > 1:
        var = sum((x - mean) ** 2 for x in rewards) / float(n - 1)
        sem = math.sqrt(var / n)
    else:
        sem = 0.0
    ok = [r for r in records if not r["err"]]
    steps = [r["n_steps"] for r in records]
    solved = sum(1 for r in records if r["reward"] >= threshold)
    answered = sum(1 for r in records if r.get("terminated"))
    solved_answered = sum(1 for r in records
                          if r.get("terminated") and r["reward"] >= threshold)
    return {
        "n": n,
        "avg_reward": mean,
        "std_err": sem,
        "solved": solved,
        "success_rate": solved / float(n),
        "avg_steps": (sum(steps) / float(n)) if steps else None,
        "n_err": n - len(ok),
        # 6.6: near the 0% floor, success alone cannot tell "cannot do the task"
        # from "never emitted send_msg_to_user". Report both.
        "answered": answered,
        "answer_rate": answered / float(n),
        "conditional_success": (solved_answered / float(answered)
                                if answered else None),
        # The domain axis's row. Empty dict, never absent, so a consumer can
        # test `agg["by_site"]` without a KeyError on an old cached aggregate.
        "by_site": (aggregate_by_site(records, threshold=threshold,
                                      task_data=task_data)
                    if by_site else {}),
    }


def success_rate(records, threshold=1.0):
    """Fraction of episodes whose terminal reward reaches `threshold`.

    TimeWarp's judge is binary in practice (verified over 768 episodes: only 0.0
    and 1.0 appear), so this equals avg_reward there. It is kept separate because
    webshop's verifier can in principle return partial credit, and 6.6 asks for
    *task success rate*, not mean reward.
    """
    if not records:
        return None
    return sum(1 for r in records if r["reward"] >= threshold) / float(len(records))


def drift_gap(train_records, test_records, threshold=1.0):
    """6.6's headline metric: success on training versions minus held-out ones.

    Positive == the agent is worse on the interfaces it was not trained on,
    which is the whole quantity TimeWarp exists to measure. `se` is the
    unpaired SEM of the difference (the two sets are different cells, so a
    paired test does not apply -- use `paired_bootstrap` for same-task
    comparisons instead).
    """
    a = success_rate(train_records, threshold)
    b = success_rate(test_records, threshold)
    if a is None or b is None:
        return {"train": a, "test": b, "gap": None, "se": None,
                "n_train": len(train_records), "n_test": len(test_records)}
    na, nb = len(train_records), len(test_records)
    se = math.sqrt(a * (1 - a) / max(na, 1) + b * (1 - b) / max(nb, 1))
    return {"train": a, "test": b, "gap": a - b, "se": se,
            "n_train": na, "n_test": nb}


def _pair_key(rec, by_seed=True):
    """Cross-run episode identity.

    `episode_key` cannot be used here: the exp_dir basename embeds the model
    name, so two runs of different checkpoints never share one. (task_id, seed,
    repeat) is the portable form -- and `repeat` matters, because AgentLab's
    random repeat seeds collide (see `episode_key`), so dropping it silently
    merges two distinct episodes of the same task.
    """
    if by_seed:
        return (rec["task_id"], rec["seed"], rec.get("repeat", 0))
    return (rec["task_id"],)


def _grouped_means(records, key_fn, val_fn):
    """key -> mean of val over that key's records (never last-write-wins)."""
    acc = collections.OrderedDict()
    for r in records:
        acc.setdefault(key_fn(r), []).append(val_fn(r))
    return collections.OrderedDict(
        (k, sum(v) / float(len(v))) for k, v in acc.items())


def paired_bootstrap(a, b, n=10000, seed=0, by_seed=True, threshold=1.0,
                     metric="success"):
    """Paired bootstrap of (mean_a - mean_b) over episodes present in both runs.

    6.6 warns that "TimeWarp baselines include a 0% floor, so near-floor effects
    need care". Three consequences are baked in here:

    * Only episodes matched by `_pair_key` are used, so the comparison is on an
      identical task set and the difference is not driven by which tasks each
      run happened to complete.
    * Resampling is over *pairs*, so a task both runs fail contributes an exact
      zero rather than variance. Near the floor this is the difference between a
      usable interval and a meaningless one.
    * When a key covers several episodes (by_seed=False, i.e. matching on
      task_id across repeats) the value is that key's MEAN, not an arbitrary
      last row.

    Returns {n_pairs, mean_a, mean_b, diff, ci_low, ci_high, p_two_sided,
             n_discordant}. `p_two_sided` is the bootstrap sign test:
    2 * min(P(d* <= 0), P(d* >= 0)).
    """
    def val(r):
        if metric == "success":
            return 1.0 if r["reward"] >= threshold else 0.0
        if metric == "reward":
            return r["reward"]
        if metric == "steps":
            return float(r["n_steps"])
        raise ValueError("unknown metric %r" % (metric,))

    kf = lambda r: _pair_key(r, by_seed)      # noqa: E731
    ma = _grouped_means(a, kf, val)
    mb = _grouped_means(b, kf, val)
    keys = sorted(set(ma) & set(mb))
    k = len(keys)
    if k == 0:
        return {"n_pairs": 0, "mean_a": None, "mean_b": None, "diff": None,
                "ci_low": None, "ci_high": None, "p_two_sided": None,
                "n_discordant": 0,
                "note": "no (task_id, seed, repeat) pairs in common -- were "
                        "these two runs on the same task subset with the same "
                        "EVAL_N_REPEATS? AgentLab draws repeat seeds at random "
                        "per study, so two runs only share seeds if they were "
                        "built from the same benchmark object. Pass "
                        "by_seed=False to match on task_id alone."}
    diffs = [ma[key] - mb[key] for key in keys]
    mean_a = sum(ma[key] for key in keys) / float(k)
    mean_b = sum(mb[key] for key in keys) / float(k)
    obs = sum(diffs) / float(k)

    rng = random.Random(seed)
    boots = []
    for _ in range(int(n)):
        s = 0.0
        for _i in range(k):
            s += diffs[rng.randrange(k)]
        boots.append(s / float(k))
    boots.sort()
    lo = boots[int(0.025 * len(boots))]
    hi = boots[min(len(boots) - 1, int(0.975 * len(boots)))]
    le = sum(1 for d in boots if d <= 0.0) / float(len(boots))
    ge = sum(1 for d in boots if d >= 0.0) / float(len(boots))
    p = min(1.0, 2.0 * min(le, ge))
    return {"n_pairs": k, "mean_a": mean_a, "mean_b": mean_b, "diff": obs,
            "ci_low": lo, "ci_high": hi, "p_two_sided": p,
            "n_discordant": sum(1 for d in diffs if d != 0.0),
            "n_boot": int(n), "metric": metric}


# --------------------------------------------------------------------------
# Findings tables
# --------------------------------------------------------------------------

def write_row(csv_path, row, fieldnames=None):
    """Append one result row to a CSV, writing the header if the file is new.

    Every findings table in this repo is built this way (see
    runs/edit_live_q35/eval_results_q35.csv), so a long sweep survives a crash
    with everything it had already finished.

    If the file exists with a different header, the row is projected onto the
    existing header and any new keys are reported in the return value rather
    than silently dropped -- a column that quietly vanishes is how a sweep ends
    up unanalysable.
    """
    row = collections.OrderedDict(row)
    d = os.path.dirname(os.path.abspath(csv_path))
    if d and not os.path.isdir(d):
        os.makedirs(d)
    existing = None
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        with open(csv_path) as fh:
            existing = next(csv.reader(fh), None)
    if existing:
        fieldnames = existing
    elif fieldnames is None:
        fieldnames = list(row.keys())
    dropped = [k for k in row if k not in fieldnames]
    out = dict((k, row.get(k, "")) for k in fieldnames)
    write_header = not existing
    with open(csv_path, "a") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow(out)
    return {"path": csv_path, "fieldnames": fieldnames, "dropped_keys": dropped,
            "wrote_header": write_header}


def result_row(spec, parsed, extra=None):
    """A flat, CSV-ready row for one eval. `parsed` is read_results()'s output."""
    agg = parsed["aggregate"]
    row = collections.OrderedDict()
    row["label"] = spec.label
    row["model"] = spec.model
    row["adapter"] = spec.adapter_dir or ""
    row["served_name"] = spec.served_name
    row["cell"] = spec.cell.key if spec.cell is not None else ""
    row["env"] = spec.cell.env if spec.cell is not None else ""
    row["era"] = spec.version
    row["split"] = spec.split
    row["judge"] = ("deterministic" if spec.deterministic_judge
                    else "%s:%s" % (spec.judge_provider, spec.judge_model or "default"))
    row["n"] = agg["n"]
    row["avg_reward"] = agg["avg_reward"]
    row["std_err"] = agg["std_err"]
    row["success_rate"] = agg["success_rate"]
    row["solved"] = agg["solved"]
    row["avg_steps"] = agg["avg_steps"]
    row["answered"] = agg.get("answered")
    row["answer_rate"] = agg.get("answer_rate")
    row["conditional_success"] = agg.get("conditional_success")
    row["n_err"] = agg["n_err"]
    row["status"] = parsed["status"]
    row["output_root"] = parsed["output_root"]
    if extra:
        row.update(extra)
    return row


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _spec_from_args(args):
    # `--cell` is a Cell; a version-level run passes `--version` + `--task-ids`
    # and no cell at all.
    cell = cells.Cell.parse(args.cell) if args.cell else None
    task_ids = None
    if getattr(args, "task_ids", None):
        task_ids = [int(t) for t in re.split(r"[,\s]+", args.task_ids.strip()) if t]
        if not task_ids:
            raise SystemExit("--task-ids was given but parsed to nothing: %r"
                             % (args.task_ids,))
    elif args.all_tasks:
        task_ids = False
    return EvalSpec(
        model=args.model, adapter_dir=args.adapter_dir, cell=cell,
        version=args.version, split=args.split,
        task_ids=task_ids,
        n_repeats=args.n_repeats, n_jobs=args.n_jobs,
        cuda_devices=args.cuda_devices, port_index=args.port_index,
        judge_provider=args.judge_provider, judge_model=args.judge_model,
        output_root=args.output_root, results_tag=args.results_tag,
        deterministic_judge=args.deterministic_judge, label=args.label,
        seed=args.seed,
        extra_env=_extra_env_from_args(args),
        lora_name=args.lora_name, max_loras=args.max_loras)


def _extra_env_from_args(args):
    """`--env K=V` (repeatable) plus the `--site-manual` sugar.

    `EvalSpec.extra_env` has existed since the start and `build_env` applies it
    last, but no CLI flag reached it, so every caller that needed one extra
    variable had to set it in the dispatching shell instead -- which the run log
    then does not record. scout_plan.md's arm A needs TW_SITE_MANUAL recorded.
    """
    env = {}
    for item in getattr(args, "env", None) or []:
        if "=" not in item:
            raise SystemExit("--env wants KEY=VALUE, got %r" % (item,))
        k, _, v = item.partition("=")
        if not k.strip():
            raise SystemExit("--env has an empty key: %r" % (item,))
        env[k.strip()] = v
    manual = getattr(args, "site_manual", None)
    if manual:
        env["TW_SITE_MANUAL"] = os.path.abspath(manual)
    return env


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(
        description="Bridge to run_v6_eval_q35_sonnet.sh. Dry-run by default; "
                    "launches nothing unless --launch is passed.")
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("plan", help="print the env + command for one eval")
    p.add_argument("--model", default=None)
    p.add_argument("--adapter-dir", default=None)
    p.add_argument("--cell", default=None, help="e.g. wiki_e3")
    p.add_argument("--version", type=int, default=None)
    p.add_argument("--split", default="test")
    p.add_argument("--all-tasks", action="store_true",
                   help="do not restrict to the cell's single-site tasks")
    p.add_argument("--seed", type=int, default=None,
                   help="replicate seed -> TW_SEED. AgentLab draws task "
                        "seeds from a hardcoded RandomState(42), so N runs "
                        "without this are byte-identical and any error bar "
                        "across them is zero by construction.")
    p.add_argument("--task-ids", default=None,
                   help="explicit comma/space-separated task ids. Required for "
                        "a cross-application matrix, where every pair must run "
                        "the SAME tasks (6.6's paired comparison) and the unit "
                        "is a version rather than a cell.")
    p.add_argument("--n-repeats", type=int, default=1)
    p.add_argument("--n-jobs", type=int, default=8)
    p.add_argument("--cuda-devices", default="0")
    p.add_argument("--port-index", type=int, default=0)
    p.add_argument("--judge-provider", default="openai")
    p.add_argument("--judge-model", default=None)
    p.add_argument("--deterministic-judge", action="store_true")
    p.add_argument("--output-root", default=None)
    p.add_argument("--results-tag", default=None)
    p.add_argument("--label", default=None)
    p.add_argument("--lora-name", default=None)
    p.add_argument("--max-loras", type=int, default=1)
    p.add_argument("--env", action="append", default=None, metavar="KEY=VALUE",
                   help="extra environment variable for the eval, repeatable. "
                        "Applied last, so it wins over everything build_env "
                        "computes; recorded in the run log.")
    p.add_argument("--site-manual", default=None, metavar="PATH",
                   help="scout_plan.md arm A: sets TW_SITE_MANUAL to this "
                        "path, which benchmark_adapter.py appends to the "
                        "agent's extra_instructions. Refused by preflight if "
                        "the file does not exist.")
    p.add_argument("--launch", action="store_true",
                   help="actually run it (default: dry-run)")

    r = sub.add_parser("read", help="parse a finished/partial run")
    r.add_argument("output_root")
    r.add_argument("--json", action="store_true")
    r.add_argument("--records", action="store_true")

    c = sub.add_parser("compare", help="paired bootstrap between two runs")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--n", type=int, default=10000)
    c.add_argument("--metric", default="success",
                   choices=("success", "reward", "steps"))
    c.add_argument("--by-task", action="store_true",
                   help="match on task_id only, ignoring seed")

    b = sub.add_parser("ports", help="show disjoint port bands")
    b.add_argument("--n", type=int, default=8)

    args = ap.parse_args(argv)
    if args.cmd is None:
        ap.print_help()
        return 2

    if args.cmd == "plan":
        spec = _spec_from_args(args)
        out = run(spec, dry_run=not args.launch)
        if out.get("problems"):
            return 1
        # A launched run has no `problems` key, so testing only for that made
        # `--launch` exit 0 even when the eval driver died. A caller that treats
        # exit 0 as success would then read an empty results dir and report a
        # zero success rate as a finding.
        rc = out.get("returncode")
        if rc:
            print("eval driver exited %s -- see the log above and %s"
                  % (rc, spec.output_root), file=sys.stderr)
            return 1
        return 0

    if args.cmd == "read":
        parsed = read_results(args.output_root)
        if args.json:
            print(json.dumps(parsed if args.records
                             else dict((k, v) for k, v in parsed.items()
                                       if k != "records"),
                             indent=2, sort_keys=True, default=str))
            return 0
        a = parsed["aggregate"]
        print("status      : %s" % parsed["status"])
        print("study dirs  : %s" % (", ".join(parsed["study_dirs"]) or "(none)"))
        print("episodes    : %d  (%d errored, from %d csv rows, %d retried)"
              % (a["n"], a["n_err"], parsed["n_rows_read"],
                 parsed["n_retried_episodes"]))
        if a["n"]:
            print("avg_reward  : %.4f +/- %.4f" % (a["avg_reward"], a["std_err"]))
            print("success     : %d/%d = %.4f" % (a["solved"], a["n"],
                                                  a["success_rate"]))
            print("avg_steps   : %.2f" % a["avg_steps"])
            # 6.6's floor caveat, made unavoidable: an agent that never emits
            # send_msg_to_user scores 0 by construction (task.py:193-221).
            ca = a.get("conditional_success")
            print("answer rate : %d/%d = %.4f%s"
                  % (a["answered"], a["n"], a["answer_rate"],
                     "   <-- near-zero: success is measuring FORMAT, not skill"
                     if a["answer_rate"] < 0.25 else ""))
            print("cond. acc   : %s (of the episodes that answered)"
                  % ("n/a -- never answered" if ca is None else "%.4f" % ca))
        if parsed["summary_df"]:
            s = parsed["summary_df"]
            print("harness says: avg_reward=%s std_err=%s n_completed=%s n_err=%s"
                  % (s.get("avg_reward"), s.get("std_err"),
                     s.get("n_completed"), s.get("n_err")))
        return 0 if parsed["status"] == "ok" else 1

    if args.cmd == "compare":
        ra = read_results(args.a)["records"]
        rb = read_results(args.b)["records"]
        res = paired_bootstrap(ra, rb, n=args.n, metric=args.metric,
                               by_seed=not args.by_task)
        print(json.dumps(res, indent=2, sort_keys=True))
        return 0

    if args.cmd == "ports":
        for i in range(args.n):
            band = port_bands(i)
            print("  %2d  ENV_PORT_BASE=%d  VLM_PORT_BASE=%d"
                  % (i, band["ENV_PORT_BASE"], band["VLM_PORT_BASE"]))
        print("\n  unsafe (Chromium ERR_UNSAFE_PORT): %s"
              % (", ".join(str(p) for p in UNSAFE_PORTS),))
        return 0

    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
