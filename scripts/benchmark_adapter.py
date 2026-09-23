"""adapterCL fork of BrowserGym-TimeWarp/benchmarkGeneral.py.

Drop-in replacement, selected with `BENCH_SCRIPT=` in run_v6_eval_q35_sonnet.sh
(:57). Everything that touches comparability with the repo's existing numbers --
prompt flags, agent class, SelfHostedModelArgs token budgets, temperature,
extra_instructions text, EVAL_N_REPEATS / EVAL_TASK_LIMIT / EVAL_N_JOBS
handling, make_study(), the joblib backend -- is copied verbatim. Only the four
things adapterCL needs are added.

DIFF vs BrowserGym-TimeWarp/benchmarkGeneral.py (original line numbers on the
left; everything not listed is byte-identical apart from this docstring):

  L16-21  argparse   ADDED --served-name, --results-dir, --task-ids, --split.
                     The eval driver calls the bench script with a fixed argv --
                     `--port --version --model` only (run_v6_eval_q35_sonnet.sh:239)
                     -- so each new flag also reads an env fallback
                     (TW_SERVED_NAME / TW_RESULTS_DIR / TW_TASK_IDS / TW_SPLIT).
                     That is what makes this usable as a black-box BENCH_SCRIPT.
  L38-46  model args CHANGED `model_name=args.model` -> `model_name=SERVED_NAME`.
                     AgentLab passes model_name straight through to VLLMChatModel
                     and out as the OpenAI `model` field (AgentLab/src/agentlab/
                     llm/chat_api.py:163-170), which is exactly the vLLM LoRA
                     served-name. --model keeps its old meaning: the checkpoint
                     identity used for the results directory.
  L94     benchmark  CHANGED subset_from_split("test") -> subset_from_split(SPLIT).
  L98-104 limit      ADDED a task-id filter BEFORE the EVAL_TASK_LIMIT slice, so
                     a run can be restricted to one cell's tasks
                     (adaptercl.cells.tasks_for_env). Also made the empty-list
                     case fail loudly instead of IndexError-ing on
                     env_args_list[0].
  L113-115 study.dir CHANGED to honour --results-dir; the
                     results/<model>/<version>_<TW_RESULTS_TAG> default is
                     unchanged so existing aggregation keeps working.
  (new)   banner     ADDED a startup banner naming the served model and the task
                     subset. adapterCL.md 9's silent-adapter risk is why: a run
                     whose log does not say which served-name it queried cannot
                     be audited after the fact.

Nothing else. In particular the judge, the reward path (TW_TASK_DATA is read
inside browsergym.timewarp.task, not here) and study.run's parallelism are
untouched.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path

from agentlab.experiments.study import make_study
from browsergym.experiments.benchmark.configs import DEFAULT_BENCHMARKS

from agentlab.agents.generic_agent import AGENT_LLAMA3_70B
from agentlab.agents.generic_agent.generic_agent import GenericAgentArgs
from agentlab.agents.generic_agent.generic_agent_with_training import GenericAgentWithTrainingArgs
from agentlab.llm.chat_api import SelfHostedModelArgs

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run Qwen Thinking benchmark on TimeWarp")
parser.add_argument("--port", type=int, default=8000, help="Port number for vLLM API (default: 8000)")
parser.add_argument("--version", type=str, required=True, help="Version string for results directory (e.g., '1', '2', 'v1.0')")
parser.add_argument("--model", type=str, required=True, help="Model name/path (e.g., '/path/to/model' or 'LLaMA-Factory/saves/qwen3-4b-thinking/full/sft')")
# ---- adapterCL additions (env fallbacks: the driver's argv is fixed) --------
parser.add_argument("--served-name", type=str, default=os.environ.get("TW_SERVED_NAME"),
                    help="vLLM served-model name to send as the OpenAI `model` field. "
                         "For a LoRA run this is the --lora-modules NAME, not the base "
                         "checkpoint. Defaults to --model. Env: TW_SERVED_NAME")
parser.add_argument("--results-dir", type=str, default=os.environ.get("TW_RESULTS_DIR"),
                    help="Explicit study directory. Default keeps the repo layout: "
                         "results/<basename(--model)>/<version>_<TW_RESULTS_TAG|all>. "
                         "Env: TW_RESULTS_DIR")
parser.add_argument("--task-ids", type=str, default=os.environ.get("TW_TASK_IDS"),
                    help="Comma/space-separated TimeWarp task ids to keep (e.g. the "
                         "output of adaptercl.cells.tasks_for_env). Env: TW_TASK_IDS")
parser.add_argument("--split", type=str, default=os.environ.get("TW_SPLIT", "test"),
                    help="browsergym_split to evaluate: test (ids 1-103) or train "
                         "(104-231). NOTE: train tasks carry additional_instructions "
                         "that leak the answer. Env: TW_SPLIT")
args = parser.parse_args()

SERVED_NAME = args.served_name or args.model

# Set environment variables for vLLM connection
# For local vLLM servers, the API key can be any non-empty string
os.environ["VLLM_API_KEY"] = "EMPTY"
os.environ["VLLM_API_URL"] = f"http://localhost:{args.port}/v1"

# Set TimeWarp environment variables if not already set
# Default to localhost ports (adjust if your TimeWarp instance uses different ports)

print(f"TimeWarp URLs:")
print(f"  TW_WIKI: {os.environ.get('TW_WIKI', 'not set')}")
print(f"  TW_WEBSHOP: {os.environ.get('TW_WEBSHOP', 'not set')}")
print(f"  TW_NEWS: {os.environ.get('TW_NEWS', 'not set')}")
print(f"  TW_HOME: {os.environ.get('TW_HOME', 'not set')}")

# Define a vLLM-hosted Qwen Thinking backend. Update host/port via env vars as needed.
qwen_thinking_vllm = SelfHostedModelArgs(
    model_name=SERVED_NAME,   # adapterCL: was args.model
    # Must be <= (server max_model_len - max_new_tokens): AgentLab shrinks the
    # prompt to max_total_tokens and THEN asks for max_new_tokens on top, so
    # budget == server length is off-by-max_new_tokens and 400s on long pages.
    max_total_tokens=63_000,  # server 65536 - 2048 generation - margin
    max_input_tokens=40_960 - 2048,
    max_new_tokens=2048,
    backend="vllm",
    temperature=0.01,
    n_retry_server=1,
)

# Reuse the stock GenericAgent prompt flags that work well for 70B-class models.
# The TimeWarp benchmark will automatically set the correct action subset (includes send_msg_to_user)
flags_70b_custom = AGENT_LLAMA3_70B.flags.copy()
flags_70b_custom.obs.use_error_logs = True

# Agent reasoning flags. Defaults match FLAGS_LLAMA3_70B (thinking on, memory &
# plan off) so existing benchmarks are unchanged; the v6-epochs orchestrator sets
# TW_USE_MEMORY / TW_USE_PLAN so eval uses the SAME thinking+memory+plan agent the
# model was GRPO-trained with.
def _envflag(name: str, default: str) -> bool:
    return os.environ.get(name, default) in ("1", "true", "True")

flags_70b_custom.use_thinking = _envflag("TW_USE_THINKING", "1")
flags_70b_custom.use_memory = _envflag("TW_USE_MEMORY", "0")
flags_70b_custom.use_plan = _envflag("TW_USE_PLAN", "0")


# Get TimeWarp URLs from environment variables (must be explicitly set)
tw_wiki = os.environ["TW_WIKI"]
tw_webshop = os.environ["TW_WEBSHOP"]
tw_news = os.environ["TW_NEWS"]
tw_home = os.environ["TW_HOME"]

flags_70b_custom.extra_instructions = f"""
IMPORTANT: You must only navigate to URLs within the TimeWarp environment.
Do NOT navigate to external websites.
Only use the URLs provided in the TimeWarp environment.
WIKI URL: {tw_wiki}
NEWS URL: {tw_news}
SHOP URL: {tw_webshop}
For instance, goto("{tw_wiki}/") to navigate to WIKI. If you need to access Wikipedia content, use the local TimeWarp wiki instance instead.
Strictly follows the instructions provided in the task description.
You MUST output EXACTLY ONE action per response. Do NOT attempt multiple actions at once.
CRITICAL: Output the <action> tag ONLY ONCE, at the end of your response, OUTSIDE of any <think> tags.
Do NOT include <action> tags inside your reasoning section.
"""

# ---- adapterCL: in-context ERA baseline -----------------------------------
# TW_ERA_FILE points at a plain-text description of the UI era the task runs in
# (out/era_text/v*.txt, the SAME strings T2L uses as its text conditioning). When
# set, it is appended to extra_instructions so the policy is *told* what era the
# page is, rather than having that information delivered through adapter weights.
#
# Read from a FILE rather than an env string on purpose: the descriptions are
# multi-line prose and the scheduler dispatches through `srun bash -c "..."`,
# where embedded newlines and quotes would be mangled.
#
# Unset (the control arm) leaves extra_instructions byte-identical to every
# previous run, so the with/without comparison differs in exactly one thing.
_era_file = os.environ.get("TW_ERA_FILE", "").strip()
if _era_file:
    with open(_era_file) as _fh:
        _era = _fh.read().strip()
    if _era:
        flags_70b_custom.extra_instructions += (
            "\n\nCONTEXT ON THIS SITE'S VISUAL ERA:\n" + _era + "\n"
        )
        print("[adapterCL] era text injected from %s (%d chars)" % (_era_file, len(_era)))

# ---- scout_plan.md arm A: the SITE MANUAL in the prompt --------------------
# TW_SITE_MANUAL points at the scout's manual for the version this run is on
# (out/scout/crawl/v*/manual_{full,affordance,style}.md). When set, it is
# appended to extra_instructions, which AgentLab renders under
# `## Extra instructions:` (dynamic_prompting.py:499-511) -- the same prompt
# slot `bcdata.inject_manual` writes into for arm A's training corpus, so train
# and eval prompts differ only in URLs and the manual body.
#
# The framing string is `adaptercl.scout.manual_block`, duplicated here rather
# than imported: this file runs under the bench interpreter with the repo root
# as cwd and must not depend on `adaptercl` being importable.
# `tests/test_scout.py` asserts the two agree.
#
# A file, not an env string, for the same reason TW_ERA_FILE is: the manual is
# multi-line prose and the scheduler dispatches through `srun bash -c "..."`.
#
# Unset (every existing caller) leaves extra_instructions byte-identical.
_manual_file = os.environ.get("TW_SITE_MANUAL", "").strip()
if _manual_file:
    with open(_manual_file) as _fh:
        _manual = _fh.read().strip()
    if _manual:
        import hashlib as _hashlib
        flags_70b_custom.extra_instructions += (
            "\n# Site manuals\n" + _manual + "\n"
        )
        print("[adapterCL] site manual injected from %s (%d chars, sha1 %s)"
              % (_manual_file, len(_manual),
                 _hashlib.sha1(_manual.encode("utf-8")).hexdigest()[:12]))
    else:
        raise SystemExit("TW_SITE_MANUAL=%s is empty" % _manual_file)

generic_agent_args = GenericAgentWithTrainingArgs(
    chat_model_args=qwen_thinking_vllm,
    flags=flags_70b_custom,
)

# EVAL_N_REPEATS: repeats per test task (default 3, preserving prior behavior).
# Set EVAL_N_REPEATS=1 for a single run per task (103 experiments).
_n_repeats = int(os.environ.get("EVAL_N_REPEATS", "3"))
timewarp_benchmark = DEFAULT_BENCHMARKS["timewarp"](n_repeats=_n_repeats)
test_benchmark = timewarp_benchmark.subset_from_split(args.split)   # adapterCL: was "test"

# ---- adapterCL: TW_SEED -- real seed control ------------------------------
# AgentLab draws its task seeds from a HARDCODED RandomState(42)
# (browsergym/experiments/.../benchmark/configs.py:111), so N independent runs
# at EVAL_N_REPEATS=1 are byte-identical -- "3 seeds" would silently be one seed
# measured three times, and the error bars would be zero by construction.
# TW_SEED re-draws each episode's task_seed from RandomState(TW_SEED), giving
# genuinely independent replicates at n_repeats=1.
_tw_seed = os.environ.get("TW_SEED")
if _tw_seed not in (None, ""):
    import numpy as _np
    _rng = _np.random.RandomState(int(_tw_seed))
    _old = [getattr(e, "task_seed", None) for e in test_benchmark.env_args_list]
    for _e in test_benchmark.env_args_list:
        _e.task_seed = int(_rng.randint(0, 2 ** 31 - 1))
    _new = [getattr(e, "task_seed", None) for e in test_benchmark.env_args_list]
    print(f"TW_SEED={_tw_seed}: re-drew {len(_new)} task seeds "
          f"(e.g. {_old[:3]} -> {_new[:3]})")


# ---- adapterCL: restrict to one cell's tasks -------------------------------
# cells.tasks_for_env returns the single-site task ids for an environment; a
# per-cell adapter must only ever be scored on tasks attributable to its own
# site, or the 6.2 transfer matrix is confounded by multi-site episodes.
def _parse_task_ids(raw: str | None) -> set[int] | None:
    if not raw:
        return None
    ids = {int(tok) for tok in re.split(r"[,\s]+", raw.strip()) if tok}
    if not ids:
        return None
    return ids


_task_ids = _parse_task_ids(args.task_ids)
if _task_ids is not None:
    def _task_id_of(env_args) -> int | None:
        m = re.search(r"timewarp\.(\d+)$", str(env_args.task_name))
        return int(m.group(1)) if m else None

    _before = len(test_benchmark.env_args_list)
    test_benchmark.env_args_list = [
        e for e in test_benchmark.env_args_list if _task_id_of(e) in _task_ids
    ]
    _kept_ids = sorted({_task_id_of(e) for e in test_benchmark.env_args_list})
    _absent = sorted(_task_ids - set(_kept_ids))
    print(f"task-id filter: {_before} -> {len(test_benchmark.env_args_list)} experiments "
          f"({len(_kept_ids)} distinct tasks of {len(_task_ids)} requested)")
    if _absent:
        print(f"  WARNING: {len(_absent)} requested id(s) are not in the "
              f"'{args.split}' split and were dropped: {_absent[:20]}")

if not test_benchmark.env_args_list:
    raise SystemExit(
        f"No experiments left after filtering. split={args.split!r} "
        f"task_ids={sorted(_task_ids)[:20] if _task_ids else None}. "
        "Test ids are 1-103, train ids are 104-231 "
        "(browsergym/experiments/.../benchmark/metadata/timewarp.csv)."
    )

# Full test split = 103 tasks x EVAL_N_REPEATS (default 3 => 309) experiments.
# EVAL_TASK_LIMIT (>0) further caps this for smoke tests; unset / <=0 keeps the full split.
_full = len(test_benchmark.env_args_list)
_eval_limit = int(os.environ.get("EVAL_TASK_LIMIT", str(_full)))
if _eval_limit <= 0:
    _eval_limit = _full
test_benchmark.env_args_list = test_benchmark.env_args_list[0:_eval_limit]
print(f"Running {len(test_benchmark.env_args_list)} experiments (n_repeats={_n_repeats}, EVAL_TASK_LIMIT={_eval_limit})")
print(f"Running on task: {test_benchmark.env_args_list[0].task_name}")

# Create study with Qwen Thinking 4B model
study = make_study(
    benchmark=test_benchmark,
    agent_args=[generic_agent_args],
    comment="Qwen3-4B-Thinking on TimeWarp test split (joblib backend)",
)
# Extract model name for results directory (use last component of path)
model_name_for_dir = Path(args.model).name if "/" in args.model or "\\" in args.model else args.model
_results_tag = os.environ.get("TW_RESULTS_TAG", "all")  # override to keep distinct runs (e.g. plan+memory) separate
# adapterCL: --results-dir wins, so one OUTPUT_ROOT can hold many cells/adapters
# without them colliding in results/<model>/<version>_<tag>. Relative paths stay
# relative to cwd, which the driver sets to $OUTPUT_ROOT (run_v6_eval_q35_sonnet.sh:233).
if args.results_dir:
    study.dir = Path(args.results_dir)
else:
    study.dir = Path(f"results/{model_name_for_dir}/{args.version}_{_results_tag}")  # Separate directory to avoid mixing with other results

# adapterCL: make the served-name auditable. A run whose log does not record
# which vLLM model string it queried cannot be told apart, after the fact, from
# a run whose adapter was silently ignored (adapterCL.md 9).
print("=" * 60)
print("adapterCL benchmark_adapter.py")
print(f"  checkpoint (--model)  : {args.model}")
print(f"  served-name sent to vLLM: {SERVED_NAME}"
      + ("   <- LoRA served-name" if SERVED_NAME != args.model else "   (== --model, no adapter)"))
print(f"  vLLM                  : {os.environ['VLLM_API_URL']}")
print(f"  split / version       : {args.split} / v{args.version}")
print(f"  study dir             : {study.dir}  (cwd={os.getcwd()})")
print(f"  agent flags           : thinking={flags_70b_custom.use_thinking} "
      f"memory={flags_70b_custom.use_memory} plan={flags_70b_custom.use_plan}")
print(f"  task data / judge     : TW_TASK_DATA={os.environ.get('TW_TASK_DATA', 'test.raw.json (LLM judge)')}"
      f"  provider={os.environ.get('TW_JUDGE_PROVIDER', 'unset')}"
      f"  model={os.environ.get('TW_JUDGE_MODEL', 'unset')}")
print("=" * 60)
sys.stdout.flush()

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    # Note: joblib uses multiprocessing, which automatically inherits environment variables
    # No need for explicit environment variable setup like with Ray

    # Run study with joblib backend
    study.run(n_jobs=int(os.environ.get("EVAL_N_JOBS", "8")), parallel_backend="joblib")
