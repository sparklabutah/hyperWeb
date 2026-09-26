# WebMix

This repository contains code and reproducible interfaces. It does not contain datasets, checkpoints, run logs, or measured results.

## Navigate the code

| Component | Entry point | Purpose |
|---|---|---|
| Adapter units and data | `adaptercl/cells.py`, `adaptercl/bcdata.py` | Define version, site, and site × version units; build behavior cloning corpora. |
| Version and domain scout | `adaptercl/scout.py` | Deterministic crawl, task-grounded scouting, manual generation, and leakage checks. |
| Mixture of adapters | `adaptercl/hypernet.py`, `scripts/make_mixture.py` | Learn routing weights or exactly combine LoRA weight deltas. |
| KL anchor | `scripts/build_offdomain_anchor.py`, `scripts/gen_yaml_anchor.py`, `integrations/llamafactory-kl.patch` | Build disjoint supervised and anchor rows and enable the reference-policy loss in the TimeWarp LLaMA-Factory fork. |
| Adapter training and serving | `adaptercl/percell.py`, `adaptercl/train_hypernet.py`, `adaptercl/materialize.py`, `scripts/startVLM_lora.sh` | Train, export, verify, and serve adapters. |
| Baselines | `scripts/run_baselines.sh`, `scripts/gen_yaml_fullft.py`, `scripts/gen_yaml_qlora.py`, `scripts/train_t2l.py` | Frozen model, pooled and nearest adapters, oracle adapter, full fine-tuning, QLoRA, and T2L code. |
| Evaluation | `adaptercl/evalbridge.py`, `scripts/benchmark_adapter.py` | Interface to the TimeWarp evaluation harness. |
| Proofs | `lean/` | Lean 4 + Mathlib proofs of the paper's theoretical results; see `lean/README.md`. |
| Tests | `tests/` | CPU contracts for data, scouting, and adapter operations. |

## Setup

Use Python 3.10 or newer. The modelling path needs PyTorch, Transformers, PEFT, and the TimeWarp LLaMA-Factory fork. Scouting needs Playwright and the TimeWarp browser environment. Evaluation needs the TimeWarp benchmark harness and a vLLM server. Install those projects separately; this repository intentionally contains no vendored dependencies or environment-specific paths.

AdapterCL finds its neighboring TimeWarp repositories from its parent directory. Override that location when the checkout is elsewhere:

```bash
export ADAPTERCL_TIMEWARP_ROOT=/path/to/TimeWarp
export ADAPTERCL_ENVS_ROOT=/path/to/conda/envs
export ADAPTERCL_BASE_MODEL=Qwen/Qwen3.5-4B
```

`ADAPTERCL_ENVS_ROOT` is optional if the named environments live under `~/miniconda3/envs`. The launcher uses `ADAPTERCL_GCC_BIN` and `ADAPTERCL_CUDA_HOME` only when the local trainer needs a separate compiler or CUDA toolkit. `HF_HOME` controls the model cache.

## Inspect and prepare

From this repository root:

```bash
python -m adaptercl                 # available stages
python -m adaptercl paths           # check configured paths
python -m adaptercl scout --help    # version and task-grounded scout commands
python -m adaptercl cells           # adapter units
python -m adaptercl targets         # LoRA injection sites
```

For a version scout, use `scout crawl`, `scout summarize`, then `scout gates`. For a task-grounded domain scout, use `scout task-pick`, collect the selected training-task episodes, then use `scout task-collect`, `scout task-summarize`, and `scout task-gates`. The task path checks train/test separation and answer leakage. See each command's `--help` for its inputs and output location. Generated files go under `out/`.

The mixture generator in `adaptercl/hypernet.py` learns coefficients over a bank. `scripts/make_mixture.py` exports a static mixture as an ordinary PEFT adapter. It concatenates LoRA factors along rank so the resulting weight delta is the weighted sum of component deltas; it verifies this equality before writing.

The KL path uses supervised rows from the current source site and anchor rows from another source site. `build_offdomain_anchor.py` emits a role mask, and `gen_yaml_anchor.py` emits the matching training recipe. The trainer integration is a patch for the TimeWarp LLaMA-Factory fork used by this project:

```bash
cd /path/to/TimeWarp/LLaMA-Factory
git apply --check /path/to/AdapterCL/integrations/llamafactory-kl.patch
git apply /path/to/AdapterCL/integrations/llamafactory-kl.patch
```

Check the patch against the exact fork before applying it; it is not an upstream LLaMA-Factory patch. Set the `TW_KL_*` variables emitted in the generated `.env` file when training. The loss computes supervised cross entropy on source rows and a reference-policy KL anchor on disjoint off-domain rows.

## Baselines and checks

`bash scripts/run_baselines.sh` prints its planned baseline units. Training or evaluation requires the runner's explicit `GO=1` and an existing allocation. The baseline generators and T2L implementation remain available independently of that runner.

Run CPU checks with the appropriate installed environment:

```bash
python tests/test_scout.py
python tests/test_contracts.py
```

The contract suite also checks local TimeWarp dependencies and optional compiler/CUDA settings, so configure those before running the full suite. Some modelling tests need the training environment and local model files. No test command in this README starts a GPU experiment.
