"""adapterCL -- conditioned adapter synthesis for TimeWarp web agents.

Implements the research plan in `adapter_project/adapterCL.md`: a frozen
Qwen3.5-9B web agent whose LoRA adapter is generated on the fly by a
hypernetwork conditioned on the interface the agent is looking at.

Nothing heavyweight is imported at package level -- the modules span three
conda envs (analysis on the system python, torch under `llamafactory`,
playwright under `tw_r1_q3`), so importing `adaptercl` must stay free. Import
the module you want, or go through `python -m adaptercl <stage>`, which
re-execs under the right interpreter for you.
"""

__version__ = "0.1.0"

#: Which interpreter each module needs. `None` == any python 3.6+.
#: Kept here (not in cli.py) so a caller can check before importing.
MODULE_ENV = {
    "paths": None,
    "cells": None,
    "targets": None,
    "era_style": None,
    "transfer": None,
    "probe": None,
    "bcdata": None,
    "percell": None,
    "evalbridge": None,
    "materialize": None,       # stdlib to import; torch only inside functions
    "verify_adapter": None,    # torch imported lazily for the HF path
    "inject": "PY_TRAIN",
    "survey": "PY_TRAIN",
    "encoder": "PY_TRAIN",
    "hypernet": "PY_TRAIN",
    "diagnostics": "PY_TRAIN",
    "train_hypernet": "PY_TRAIN",
    "toy": "PY_TRAIN",
    "capture": "PY_BENCH",
}
