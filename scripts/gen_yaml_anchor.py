#!/usr/bin/env python3
"""Emit the training YAML (+ sibling .env) for one K1 off-domain-anchor component
(ablation K, scout_task_plan.md 10.2/10.3).

Reads the ``.anchorstats.json`` side-car ``scripts/build_offdomain_anchor.py`` already wrote (this
script builds NO corpus itself -- run that first) and emits:

    <out-root>/cells/anc6_4b/yaml/_gen_<unit>.yaml   (the LoRA training config)
    <out-root>/cells/anc6_4b/yaml/_gen_<unit>.env    (TW_KL_* exports run_anchor_train.sh sources)

THE TEMPLATE is ``adaptercl.percell.write_cell_yaml`` -- the EXACT SAME function and template
string ``gen_yaml_site.py`` uses for ``site6_4b`` -- with the recipe pinned to site6_4b's own
values (rank 16, alpha 32, lr 1e-4, 6 epochs, cutoff 65536; seed follows site6_4b's OWN convention
of ``seed_base(42) + draw``, which is 43 for draw 1 -- the literal "seed 43" scout_task_plan.md
10.3 names is that convention's draw-1 value, not a constant across draws; see out/cells/site6_4b/
yaml/_gen_<site>_v6_d<draw>.yaml for drawS 2/3 -> seed 44/45).

ONE DELIBERATE OVERRIDE of that shared template: ``enable_liger_kernel``. site6_4b's YAML sets it
`true` (fine for a plain LoRA CE loss); the KL anchor path
(``CustomSeq2SeqTrainer._tw_kl_compute_loss``) needs the REAL, materialised ``outputs["logits"]``
to recompute per-token log p under both the student and the frozen reference, which a fused-CE
kernel does not necessarily hand back. The existing KL-anchor recipes under `self-correct/`
(``examples/train_full/sc_cell_q35.yaml``) already run with liger OFF for exactly this reason ("NO
liger -- liger has no kernel for the new qwen3_5 arch; would error"); this script matches that
choice by flipping the one line ``percell.write_cell_yaml`` writes, post-hoc (percell.py's template
hard-codes ``true`` and is out of this package's file list -- see PATCH.diff / open_issues if a
first-class parameter is wanted instead).

    python scripts/gen_yaml_anchor.py --fold shop --site wiki --draw 1 --btag b0p1
"""
from __future__ import print_function

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT)

from adaptercl import cells, paths, percell  # noqa: E402
from scripts.build_offdomain_anchor import BASE_SITES, BTAG_BETA  # noqa: E402

#: Verbatim site6_4b recipe (scout_task_plan.md 10.3: "Same rank/alpha/lr/epochs/seed as
#: site6_4b"). Duplicated rather than imported from gen_yaml_site.py -- scripts/ is not a package
#: and every other launcher here is a stand-alone file by the same convention.
PV = {"rank": 16, "alpha": 32.0, "target_set": "attn_mlp", "epochs": 6.0,
      "lr": "1.0e-4", "cutoff": 65536, "grad_accum": 4}
SEED_BASE = 42
BACKBONE = {"4b": "Qwen/Qwen3.5-4B", "9b": "Qwen/Qwen3.5-9B"}

LIGER_ON = ("enable_liger_kernel: true                 # as in the base recipe: fused CE keeps "
            "the 152k-vocab logits off-peak")
LIGER_OFF = ("enable_liger_kernel: false                # DEVIATION from the site6_4b template: "
             "the KL anchor path needs real, materialised logits (see this script's module "
             "docstring; matches self-correct/examples/train_full/sc_cell_q35.yaml)")


class AnchorYamlError(Exception):
    pass


def _anchorstats_path(out_root, unit_key):
    # out_root/corpus_anc6_4b/, NOT out_root/corpus/anc6_4b/ -- see build_offdomain_anchor.py's
    # module docstring (REGISTRY ISOLATION: this whole tree, including dataset_info.json, is
    # private to K1 and never the shared out_root/corpus/ another session's live trainings use).
    return os.path.join(out_root, "corpus_anc6_4b", unit_key + ".anchorstats.json")


def _unit_key(site, other, btag, draw):
    return "%s_v6_anc-%s_%s_d%d" % (site, other, btag, draw)


def _disable_liger(yaml_text, yaml_path):
    if LIGER_ON not in yaml_text:
        raise AnchorYamlError(
            "expected the literal line %r in %s (from adaptercl.percell.YAML_TEMPLATE) but did not "
            "find it -- the template changed underneath this override; fix this script's LIGER_ON "
            "string before trusting the yaml it just wrote" % (LIGER_ON, yaml_path)
        )
    return yaml_text.replace(LIGER_ON, LIGER_OFF, 1)


def build_yaml(fold, site, draw, btag, out_root=None, model=None, template=None):
    if fold not in BASE_SITES:
        raise AnchorYamlError("--fold %r must be one of %r" % (fold, BASE_SITES))
    if site not in BASE_SITES:
        raise AnchorYamlError("--site %r must be one of %r" % (site, BASE_SITES))
    if site == fold:
        raise AnchorYamlError("--site %r == --fold %r" % (site, fold))
    if btag not in BTAG_BETA:
        raise AnchorYamlError("--btag %r must be one of %r" % (btag, sorted(BTAG_BETA)))
    others = [s for s in BASE_SITES if s not in (fold, site)]
    other = others[0]
    draw = int(draw)

    out_root = out_root or paths.OUT
    unit_key = _unit_key(site, other, btag, draw)
    stats_path = _anchorstats_path(out_root, unit_key)
    if not os.path.isfile(stats_path):
        raise AnchorYamlError(
            "missing %s -- run scripts/build_offdomain_anchor.py --fold %s --site %s --draw %d "
            "--btag %s first" % (stats_path, fold, site, draw, btag)
        )
    with open(stats_path) as fh:
        anchor_stats = json.load(fh)
    if anchor_stats.get("unit") != unit_key:
        raise AnchorYamlError(
            "%s is for unit %r, expected %r -- stale or mismatched side-car"
            % (stats_path, anchor_stats.get("unit"), unit_key)
        )

    model = model or BACKBONE["4b"]
    seed = SEED_BASE + draw
    u = cells.Site(site, draw=draw, eras=(6,))  # provenance object only; out_dir/yaml_path below
    #                                             are the anchor tree's own paths, not u.key's.
    out_dir = os.path.join(out_root, "cells", "anc6_4b", unit_key)
    yaml_dir = os.path.join(out_root, "cells", "anc6_4b", "yaml")
    yaml_path = os.path.join(yaml_dir, "_gen_%s.yaml" % unit_key)
    env_path = os.path.join(yaml_dir, "_gen_%s.env" % unit_key)

    # dataset_dir points at K1's OWN registry (out_root/corpus_anc6_4b/), never the shared
    # out_root/corpus/ dataset_info.json another session's live trainings import (see
    # build_offdomain_anchor.py's REGISTRY ISOLATION note). LLaMA-Factory resolves
    # `<dataset_dir>/dataset_info.json` from this yaml's `dataset_dir` field -- verified against
    # a real generated site6_4b yaml, which does the analogous thing for the shared registry.
    dataset_dir = os.path.join(out_root, "corpus_anc6_4b")
    written = percell.write_cell_yaml(
        u, anchor_stats["dataset_name"], target_set=PV["target_set"], rank=PV["rank"],
        alpha=PV["alpha"], out_dir=out_dir, tag="anc6_4b", model=model, epochs=PV["epochs"],
        lr=PV["lr"], cutoff=PV["cutoff"], grad_accum=PV["grad_accum"], seed=seed,
        dataset_dir=dataset_dir, n_samples=anchor_stats["n_samples"],
        yaml_path=yaml_path, template=template,
    )
    with open(written) as fh:
        text = fh.read()
    text = _disable_liger(text, written)
    with open(written, "w") as fh:
        fh.write(text)

    beta = BTAG_BETA[btag]
    env_lines = [
        "# GENERATED by scripts/gen_yaml_anchor.py for %s -- source before `lmf train %s`"
        % (unit_key, os.path.basename(written)),
        "export TW_KL_BETA=%g" % beta,
        "export TW_KL_REF=%s" % model,
        "export TW_KL_GAMMA_WEIGHTS=%s" % anchor_stats["gamma_json"],
        "export TW_KL_DISJOINT=1",
        # 2026-09-21: the b1p0 bank's 18 env files had this line added BY HAND on 09-19 03:14,
        # after the first GPU run OOM'd an 80 GB card on full-sequence logits (248k vocab x ~38k
        # tokens). The generator was never taught it, so the freshly generated b0p1 envs lacked it
        # and all 18 units OOM'd on 09-21. Same arithmetic (see trainer.py "TAIL LOGITS"); without
        # it a sweep level is not trained under the conditions of the published bank.
        "export TW_KL_TAIL_LOGITS=1",
        "",
    ]
    with open(env_path, "w") as fh:
        fh.write("\n".join(env_lines))

    return {"unit": unit_key, "yaml": written, "env": env_path, "seed": seed, "beta": beta,
            "model": model, "dataset": anchor_stats["dataset_name"],
            "n_samples": anchor_stats["n_samples"], "output_dir": out_dir}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fold", required=True, choices=BASE_SITES)
    ap.add_argument("--site", required=True, choices=BASE_SITES)
    ap.add_argument("--draw", required=True, type=int)
    ap.add_argument("--btag", required=True, choices=sorted(BTAG_BETA))
    ap.add_argument("--backbone", default="4b", choices=sorted(BACKBONE))
    ap.add_argument("--model", default=None, help="override the backbone id")
    ap.add_argument("--out-root", default=None,
                    help="project 'out' directory (default: the real out/; tests must override)")
    ap.add_argument("--template", default=None)
    args = ap.parse_args(argv)

    try:
        result = build_yaml(
            fold=args.fold, site=args.site, draw=args.draw, btag=args.btag,
            out_root=args.out_root, model=args.model or BACKBONE[args.backbone],
            template=args.template,
        )
    except AnchorYamlError as exc:
        print("BLOCKED: %s" % exc, file=sys.stderr)
        return 1

    print("=== %s ===" % result["unit"])
    print("yaml   : %s" % result["yaml"])
    print("env    : %s" % result["env"])
    print("seed=%d beta=%g model=%s dataset=%s n_samples=%d"
          % (result["seed"], result["beta"], result["model"], result["dataset"],
             result["n_samples"]))
    print("output_dir: %s" % result["output_dir"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
