<h1 align="center">
  🎛️&nbsp;Don’t Retrain, Remix: Adaptation of Web Agents using Mixture Hypernetworks
</h1>

<div align="center">

[![project](https://img.shields.io/badge/Project%20Page-4285F4?style=flat&logo=homeassistant&logoColor=white&color=006A4E&labelColor=gray)](https://webmix-hyper.github.io)
[![code](https://img.shields.io/badge/GitHub-sparklabutah/hyperweb-blue?logo=GitHub&labelColor=black)](https://github.com/sparklabutah/hyperweb)
</div>

tldr. WebMix helps fine-tuned web agents reuse and combine the skills they have already learned, so they generalize to unseen web domains and to new versions of familiar interfaces without retraining. A **Scout** explores a web domain and writes a manual of how it works. A **Mixture Hypernetwork** reads the manual and mixes a bank of trained LoRA adapters into a single adapter. The **Web Agent** then acts on the domain with the mixed adapter. WebMix is built on the [TimeWarp](https://github.com/sparklabutah/timewarp) benchmark.

> [!NOTE]
> This repository contains code and reproducible interfaces. It does not contain datasets, checkpoints, run logs, or measured results.

---

## Table of Contents

- [Installation](#-installation)
  - [Environment Variables](#environment-variables)
  - [Inspect and Prepare](#inspect-and-prepare)
- [Navigating the Code](#-navigating-the-code)
- [Scouting a Web Domain](#-scouting-a-web-domain)
- [Mixing LoRA Adapters](#-mixing-lora-adapters)
- [Training with a KL Anchor](#-training-with-a-kl-anchor)
- [Evaluating your Web Agent](#-evaluating-your-web-agent)
- [Baselines and Checks](#-baselines-and-checks)
- [Formal Proofs](#-formal-proofs)
- [Citation](#citation)

---

## 📦 Installation

⚠️ Use Python 3.10 or newer. This repository intentionally contains no vendored dependencies or environment-specific paths, so install the projects below separately. ⚠️

| Path | Needs |
|------|-------|
| Modelling | PyTorch, Transformers, PEFT, and the TimeWarp LLaMA-Factory fork |
| Scouting | Playwright and the [TimeWarp](https://github.com/sparklabutah/timewarp) browser environments |
| Evaluation | The TimeWarp benchmark harness and a vLLM server |

Every stage runs through one entry point, `python -m adaptercl <stage>`, which re-runs itself under the conda environment that stage needs. It looks for these environments under `ADAPTERCL_ENVS_ROOT`:

| Conda environment | Used for |
|-------------------|----------|
| `llamafactory` | Adapter training, the encoder, and the hypernetwork (PyTorch, Transformers, PEFT) |
| `tw_r1_q3` | Scouting, page capture, and benchmarking (AgentLab, BrowserGym, Playwright) |
| `vllm_q35` | Serving the base model and adapters with vLLM |
| `tw_web`, `tw_webshop` | The TimeWarp Wiki/News and Shop servers |

### Environment Variables

WebMix finds its neighboring TimeWarp checkouts (for example `TimeWarp/`, `BrowserGym-TimeWarp/`, and `LLaMA-Factory/`) from its parent directory. Override any of these when your setup differs:

| Variable | Default | Description |
|----------|---------|-------------|
| `ADAPTERCL_TIMEWARP_ROOT` | parent of this checkout | Directory that holds the TimeWarp checkouts |
| `ADAPTERCL_ENVS_ROOT` | `~/miniconda3/envs` | Directory that holds the conda environments above |
| `ADAPTERCL_BASE_MODEL` | `Qwen/Qwen3.5-9B` | Base model that adapters are trained for and served on |
| `ADAPTERCL_GCC_BIN` | unset | Separate compiler, only if the local trainer needs one |
| `ADAPTERCL_CUDA_HOME` | unset | Separate CUDA toolkit, only if the local trainer needs one |
| `HF_HOME` | `~/.cache/huggingface` | Model cache |

Example:

```sh
export ADAPTERCL_TIMEWARP_ROOT=/path/to/TimeWarp
export ADAPTERCL_ENVS_ROOT=/path/to/conda/envs
export ADAPTERCL_BASE_MODEL=Qwen/Qwen3.5-4B
```

### Inspect and Prepare

From the repository root:

```sh
python -m adaptercl                 # list every stage
python -m adaptercl paths           # check configured paths
python -m adaptercl status          # what is built, what has run, what is blocking
python -m adaptercl cells           # adapter units
python -m adaptercl targets         # LoRA injection sites
```

Pass `--help` to any stage for its options, or `--where` to print the interpreter and command instead of running it. Generated files go under `out/`.

---

## 🧭 Navigating the Code

| Component | Entry point | Purpose |
|---|---|---|
| Adapter units and data | [`adaptercl/cells.py`](adaptercl/cells.py), [`adaptercl/bcdata.py`](adaptercl/bcdata.py) | Define version, site, and site × version units; build behavior cloning corpora. |
| Version and domain scout | [`adaptercl/scout.py`](adaptercl/scout.py) | Deterministic crawl, task-grounded scouting, manual generation, and leakage checks. |
| Mixture of adapters | [`adaptercl/hypernet.py`](adaptercl/hypernet.py), [`scripts/make_mixture.py`](scripts/make_mixture.py) | Learn routing weights or exactly combine LoRA weight deltas. |
| KL anchor | [`scripts/build_offdomain_anchor.py`](scripts/build_offdomain_anchor.py), [`scripts/gen_yaml_anchor.py`](scripts/gen_yaml_anchor.py), [`integrations/llamafactory-kl.patch`](integrations/llamafactory-kl.patch) | Build disjoint supervised and anchor rows and enable the reference-policy loss in the TimeWarp LLaMA-Factory fork. |
| Adapter training and serving | [`adaptercl/percell.py`](adaptercl/percell.py), [`adaptercl/train_hypernet.py`](adaptercl/train_hypernet.py), [`adaptercl/materialize.py`](adaptercl/materialize.py), [`scripts/startVLM_lora.sh`](scripts/startVLM_lora.sh) | Train, export, verify, and serve adapters. |
| Baselines | [`scripts/run_baselines.sh`](scripts/run_baselines.sh), [`scripts/gen_yaml_fullft.py`](scripts/gen_yaml_fullft.py), [`scripts/gen_yaml_qlora.py`](scripts/gen_yaml_qlora.py), [`scripts/train_t2l.py`](scripts/train_t2l.py) | Frozen model, pooled and nearest adapters, oracle adapter, full fine-tuning, QLoRA, and T2L code. |
| Evaluation | [`adaptercl/evalbridge.py`](adaptercl/evalbridge.py), [`scripts/benchmark_adapter.py`](scripts/benchmark_adapter.py) | Interface to the TimeWarp evaluation harness. |
| Proofs | [`lean/`](lean/) | Lean 4 + Mathlib proofs of the paper's theoretical results. |
| Tests | [`tests/`](tests/) | CPU contracts for data, scouting, and adapter operations. |

---

## 🔭 Scouting a Web Domain

The Scout visits a site with no task and writes a manual of how it works. A scripted crawler explores the site, and an LLM only summarizes the crawl, which keeps the crawl reproducible. Scout commands are dry runs unless `GO=1` is set.

**Version scout.** Crawl, summarize the crawl into a manual, then run the gates:

```sh
python -m adaptercl scout crawl --versions 1,6 --envs wiki,news
python -m adaptercl scout summarize
python -m adaptercl scout gates
```

The summarizer calls an OpenAI-compatible endpoint, set with `--endpoint` or `SCOUT_ENDPOINT` (model with `--model` or `SCOUT_MODEL`).

**Task-grounded domain scout.** Pick training tasks, collect their episodes, then summarize and gate:

```sh
python -m adaptercl scout task-pick
# collect the selected training-task episodes
python -m adaptercl scout task-collect
python -m adaptercl scout task-summarize
python -m adaptercl scout task-gates
```

The task path checks train/test separation and answer leakage. See each command's `--help` for its inputs and output location.

---

## 🔀 Mixing LoRA Adapters

The mixture generator in [`adaptercl/hypernet.py`](adaptercl/hypernet.py) learns coefficients $c_k$ over a bank of $K$ adapters. Averaging the LoRA factors would add cross terms between one adapter's $B$ and another's $A$, so WebMix mixes the weight deltas instead. Concatenating the factors along the rank axis represents that sum exactly:

```math
\Delta W \;=\; \sum_{k=1}^{K} c_k\, B_k A_k \;=\; \underbrace{\begin{bmatrix} B_1 & \cdots & B_K \end{bmatrix}}_{B_{\text{cat}}}\;\underbrace{\begin{bmatrix} c_1 A_1 \\ \vdots \\ c_K A_K \end{bmatrix}}_{A_{\text{cat}}}
```

[`scripts/make_mixture.py`](scripts/make_mixture.py) exports a static mixture as an ordinary PEFT adapter and verifies this equality before writing:

```sh
python scripts/make_mixture.py \
  --components path/to/adapter1 path/to/adapter2 path/to/adapter3 \
  --weights 0.5,0.3,0.2 \
  --out path/to/mixed_adapter
```

Weights default to uniform $1/K$ and are not renormalized.

---

## ⚓ Training with a KL Anchor

The KL path uses supervised rows from the current source site and anchor rows from another source site. [`build_offdomain_anchor.py`](scripts/build_offdomain_anchor.py) emits a role mask, and [`gen_yaml_anchor.py`](scripts/gen_yaml_anchor.py) emits the matching training recipe. The trainer integration is a patch for the TimeWarp LLaMA-Factory fork used by this project:

```sh
cd /path/to/TimeWarp/LLaMA-Factory
git apply --check /path/to/hyperweb/integrations/llamafactory-kl.patch
git apply /path/to/hyperweb/integrations/llamafactory-kl.patch
```

⚠️ Check the patch against the exact fork before applying it; it is not an upstream LLaMA-Factory patch. ⚠️

Set the `TW_KL_*` variables emitted in the generated `.env` file when training. The loss computes supervised cross entropy on source rows and a reference-policy KL anchor on disjoint off-domain rows. After applying the patch, check the integration on CPU:

```sh
python tests/test_kl_disjoint.py
```

---

## 📏 Evaluating your Web Agent

WebMix evaluates through the TimeWarp harness. Start the TimeWarp environments as described in the [TimeWarp README](https://github.com/sparklabutah/timewarp#-running-environments), then:

**1. Serve the base model with an adapter.** [`startVLM_lora.sh`](scripts/startVLM_lora.sh) takes the same `--model` and `--port` flags as TimeWarp's `startVLMmodel.sh` and refuses to start if an adapter path is not a PEFT adapter:

```sh
LORA_MODULES="mix=path/to/mixed_adapter" MAX_LORA_RANK=64 \
  bash scripts/startVLM_lora.sh --model Qwen/Qwen3.5-4B --port 8000
```

A mixture of $K$ rank-$r$ adapters has rank $K \cdot r$, so set `MAX_LORA_RANK` at least that high.

**2. Run the benchmark.** [`benchmark_adapter.py`](scripts/benchmark_adapter.py) is TimeWarp's benchmark script with four additions: `--served-name` (the LoRA name to query), `--results-dir`, `--task-ids`, and `--split`:

```sh
python scripts/benchmark_adapter.py \
  --port 8000 \
  --version v1 \
  --model Qwen/Qwen3.5-4B \
  --served-name mix
```

---

## 🧪 Baselines and Checks

`bash scripts/run_baselines.sh` prints its planned baseline units. Training or evaluation requires the runner's explicit `GO=1` and an existing allocation. The baseline generators and T2L implementation remain available independently of that runner.

Run CPU checks with the appropriate installed environment:

```sh
python tests/test_scout.py
python tests/test_contracts.py
```

The contract suite also checks local TimeWarp dependencies and optional compiler/CUDA settings, so configure those before running the full suite. Some modelling tests need the training environment and local model files. No test command in this README starts a GPU experiment.

---

## 📐 Formal Proofs

Machine-checked Lean 4 + Mathlib proofs of the paper's theoretical results live in [`lean/`](lean/). Every statement is proved without `sorry`. See [`lean/README.md`](lean/README.md) for details.

```sh
cd lean
lake exe cache get        # fetch the Mathlib cache once
lake build                # build all modules
lake env lean Audit.lean  # print the axioms each headline theorem depends on
```

---

## Citation

Don't forget to cite all the repos that have helped us!

### BrowserGym and AgentLab
```bibtex
@article{
    chezelles2025browsergym,
    title={The BrowserGym Ecosystem for Web Agent Research},
    author={Thibault Le Sellier de Chezelles and Maxime Gasse and Alexandre Lacoste and Massimo Caccia and Alexandre Drouin and L{\'e}o Boisvert and Megh Thakkar and Tom Marty and Rim Assouel and Sahar Omidi Shayegan and Lawrence Keunho Jang and Xing Han L{\`u} and Ori Yoran and Dehan Kong and Frank F. Xu and Siva Reddy and Graham Neubig and Quentin Cappart and Russ Salakhutdinov and Nicolas Chapados},
    journal={Transactions on Machine Learning Research},
    issn={2835-8856},
    year={2025},
    url={https://openreview.net/forum?id=5298fKGmv3},
    note={Expert Certification}
}
```

### TimeWarp
```bibtex
@inproceedings{ishmam2026timewarp,
  title         = {{TimeWarp}: Evaluating Web Agents by Revisiting the Past},
  author        = {Ishmam, Md Farhan and Marino, Kenneth},
  booktitle     = {Fortieth Conference on Neural Information Processing Systems Evaluations and Datasets Track},
  year          = {2026},
  note          = {To appear},
  eprint        = {2603.04949},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2603.04949}
}
```

If you enjoyed using this repo, also consider citing us! 😊

### WebMix
```bibtex
@misc{ishmam2026webmix,
  title  = {Don't Retrain, Remix: Adaptation of Web Agents using Mixture Hypernetworks},
  author = {Ishmam, Md Farhan and Pham-Dinh, Minh and Marino, Kenneth},
  year   = {2026},
  url    = {https://webmix-hyper.github.io}
}
```
