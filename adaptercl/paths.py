"""Repository paths and configurable interpreters for AdapterCL.

Paths are derived from this checkout or environment variables.

Nothing here imports torch/transformers, so this module is importable from the
system python.
"""

from __future__ import print_function

import os
import sys

# --------------------------------------------------------------------------
# Repo layout
# --------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
REPO = os.path.abspath(os.environ.get("ADAPTERCL_TIMEWARP_ROOT", os.path.dirname(PROJECT)))

BGYM = os.path.join(REPO, "BrowserGym-TimeWarp")
TIMEWARP_ENV = os.path.join(REPO, "TimeWarp")      # the flask env repo
LLAMA_FACTORY = os.path.join(REPO, "LLaMA-Factory")
TIMETRAJ = os.path.join(REPO, "TimeTraj-Trajectories")
VLM_DIR = os.path.join(REPO, "vlm")

TIMEWARP_PKG = os.path.join(
    BGYM, "browsergym", "timewarp", "src", "browsergym", "timewarp"
)
TASK_DATA_DIR = os.path.join(TIMEWARP_PKG, "data")
TASK_DATA_LLM_JUDGE = os.path.join(TASK_DATA_DIR, "test.raw.json")
TASK_DATA_DETERMINISTIC = os.path.join(TASK_DATA_DIR, "test.raw.v2.json")
SPLIT_CSV = os.path.join(
    BGYM, "browsergym", "experiments", "src", "browsergym", "experiments",
    "benchmark", "metadata", "timewarp.csv",
)

# Project outputs. Kept inside adapter_project so nothing pollutes the repo root.
OUT = os.path.join(PROJECT, "out")
OUT_SURVEY = os.path.join(OUT, "survey")
OUT_CAPTURE = os.path.join(OUT, "capture")
OUT_STYLE = os.path.join(OUT, "style")
OUT_CELLS = os.path.join(OUT, "cells")          # per-cell LoRA adapters
OUT_TRANSFER = os.path.join(OUT, "transfer")    # transfer matrices
OUT_HYPERNET = os.path.join(OUT, "hypernet")
OUT_EVAL = os.path.join(OUT, "eval")
OUT_CORPUS = os.path.join(OUT, "corpus")
OUT_LOGS = os.path.join(OUT, "logs")

ALL_OUT_DIRS = (
    OUT, OUT_SURVEY, OUT_CAPTURE, OUT_STYLE, OUT_CELLS, OUT_TRANSFER,
    OUT_HYPERNET, OUT_EVAL, OUT_CORPUS, OUT_LOGS,
)


def ensure_out_dirs():
    """Create every project output dir. Idempotent."""
    for d in ALL_OUT_DIRS:
        if not os.path.isdir(d):
            os.makedirs(d)
    return OUT


# --------------------------------------------------------------------------
# Interpreters / envs
#
# Override ADAPTERCL_ENVS_ROOT or individual executable paths for your setup.
# --------------------------------------------------------------------------

ENVS = os.environ.get("ADAPTERCL_ENVS_ROOT", os.path.join(os.path.expanduser("~"), "miniconda3", "envs"))

PY_TRAIN = os.path.join(ENVS, "llamafactory", "bin", "python")   # torch 2.12 / tf 5.6 / peft 0.19
PY_VLLM = os.path.join(ENVS, "vllm_q35", "bin", "python")        # vllm 0.23.1rc1 / tf 5.13
PY_BENCH = os.path.join(ENVS, "tw_r1_q3", "bin", "python")       # agentlab / browsergym
PY_WEB = os.path.join(ENVS, "tw_web", "bin", "python")           # wiki + news flask apps
PY_SHOP = os.path.join(ENVS, "tw_webshop", "bin", "python")      # webshop flask app (bundled JVM)
LMF = os.path.join(ENVS, "llamafactory", "bin", "lmf")           # llamafactory-cli

# Set these only when the local trainer requires a separate compiler/toolkit.
SPACK_GCC_BIN = os.environ.get("ADAPTERCL_GCC_BIN", "")
SPACK_CUDA_HOME = os.environ.get("ADAPTERCL_CUDA_HOME", "")

# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

HF_HOME = os.environ.get(
    "HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
)
BASE_MODEL = os.environ.get("ADAPTERCL_BASE_MODEL", "Qwen/Qwen3.5-9B")
BASE_MODEL_SMALL = "Qwen/Qwen3.5-2B"   # for CPU-ish smoke tests

def hf_snapshot(repo_id):
    """Resolve a local HF hub snapshot dir, or None if not cached.

    >>> hf_snapshot("Qwen/Qwen3.5-9B")  # doctest: +SKIP
    '~/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/<revision>'
    """
    slug = "models--" + repo_id.replace("/", "--")
    snap = os.path.join(HF_HOME, "hub", slug, "snapshots")
    if not os.path.isdir(snap):
        return None
    entries = [os.path.join(snap, e) for e in sorted(os.listdir(snap))]
    entries = [e for e in entries if os.path.isdir(e)]
    if not entries:
        return None
    # Prefer a snapshot that actually has a config.json.
    for e in entries:
        if os.path.exists(os.path.join(e, "config.json")):
            return e
    return entries[0]


def model_dir_or_id(repo_id=None):
    """Local snapshot dir if cached, else the hub id (so from_pretrained still works)."""
    repo_id = repo_id or BASE_MODEL
    return hf_snapshot(repo_id) or repo_id


# --------------------------------------------------------------------------
# Reused parent-repo entry points
# --------------------------------------------------------------------------

EVAL_DRIVER = os.path.join(REPO, "run_v6_eval_q35_sonnet.sh")
BENCH_GENERAL = os.path.join(BGYM, "benchmarkGeneral.py")
BENCH_TRAIN_SPLIT = os.path.join(
    BGYM, "collectTrainingTrajectories", "benchmarkGeneralNoPlan.py"
)
CONVERT_SGPT = os.path.join(REPO, "dataCreationScripts", "convert2sgptArgs.py")
FILTER_TRAINING_DATA = os.path.join(REPO, "filter_training_data.py")
AGGREGATE = os.path.join(REPO, "curriculum_aggregate.py")
DATASET_INFO = os.path.join(LLAMA_FACTORY, "data", "dataset_info.json")
RUN_ALL_ENV = os.path.join(TIMEWARP_ENV, "run_all_env.sh")
STOP_ALL_PORTS = os.path.join(TIMEWARP_ENV, "stop_all_ports.sh")

# Our own scripts
SCRIPTS = os.path.join(PROJECT, "scripts")
VLM_LORA_LAUNCHER = os.path.join(SCRIPTS, "startVLM_lora.sh")
BENCH_ADAPTER = os.path.join(SCRIPTS, "benchmark_adapter.py")


def add_timewarp_to_syspath():
    """Make `browsergym.timewarp` importable without installing anything.

    Only needed by tools that read the task JSON through the package; the
    task-data readers in cells.py go straight to the JSON file instead.
    """
    p = os.path.join(TIMEWARP_PKG, "..", "..", "..")
    p = os.path.abspath(p)
    if p not in sys.path:
        sys.path.insert(0, p)
    return p


def describe():
    """Human-readable existence check of everything this module points at."""
    rows = []
    for name in sorted(dir(sys.modules[__name__])):
        if name.startswith("_") or not name.isupper():
            continue
        val = getattr(sys.modules[__name__], name)
        if not isinstance(val, str) or not val.startswith("/"):
            continue
        rows.append((name, val, os.path.exists(val)))
    return rows


if __name__ == "__main__":
    missing = 0
    for name, val, ok in describe():
        print("%-24s %-5s %s" % (name, "ok" if ok else "MISS", val))
        missing += 0 if ok else 1
    snap = hf_snapshot(BASE_MODEL)
    print("%-24s %-5s %s" % ("BASE_MODEL", "ok" if snap else "MISS", snap or BASE_MODEL))
    print("\n%d missing path(s)" % missing)
