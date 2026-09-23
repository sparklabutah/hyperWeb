#!/usr/bin/env python3
"""Generate per-version LoRA YAMLs for a second backbone (default Qwen3.5-4B).

The oracle adapters are the reconstruction targets for T2L, and dW is defined
against a specific set of weight matrices -- so a 4B run needs its own oracles;
the 9B ones are not merely suboptimal for it, they have the wrong shapes
entirely (2560-wide vs 4096-wide hidden).

Everything except the backbone is held fixed against the 9B recipe: same BC
corpora (they are plain text, so model-agnostic), same rank, same alpha, same
target set, same epochs and learning rate. A difference between backbones should
be attributable to the backbone.

    python scripts/gen_yaml_4b.py --model Qwen/Qwen3.5-4B --tag ver4b
    TAG=ver4b GO=1 SLOTS="<jobid>:0 ..." bash scripts/run_version_train.sh
"""
from __future__ import print_function

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT)
os.chdir(PROJECT)

from adaptercl import cells, percell, targets  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--tag", default="ver4b")
    ap.add_argument("--src-tag", default="ver",
                    help="tag whose BC datasets are reused (corpora are text)")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--target-set", default="attn_mlp")
    args = ap.parse_args(argv)

    cfg = targets.load_text_config(args.model)
    sites = targets.enumerate_sites(cfg, args.target_set)
    print("backbone %s: %d layers, hidden=%d, %d sites, %s adapter params"
          % (args.model, cfg["num_hidden_layers"], cfg["hidden_size"],
             len(sites), "{:,}".format(targets.site_budget(sites, args.rank))))

    made = []
    for v in cells.ALL_VERSIONS:
        # Reuse the 9B run's dataset names: the corpora are (system,
        # conversations) text and carry no tokenizer or model assumption.
        dataset = "adaptercl_%s_%s" % (args.src_tag, v.key)
        y = percell.write_cell_yaml(
            v, dataset, target_set=args.target_set, rank=args.rank,
            alpha=args.alpha, tag=args.tag, model=args.model)
        made.append(y)
        print("  %-4s -> %s" % (v.key, y))
    print("\n%d YAML(s) under out/cells/%s/yaml" % (len(made), args.tag))
    print("next: TAG=%s GO=1 SLOTS=\"<jobid>:<gpu> ...\" bash scripts/run_version_train.sh"
          % args.tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
