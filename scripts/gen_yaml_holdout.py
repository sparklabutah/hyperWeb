#!/usr/bin/env python3
"""Generate a CONTINUAL LoRA chain over five versions, holding one out entirely.

This is the baseline the leave-one-out generators have never been tested against.
`loo6_4b` synthesises a v6 adapter from the other five versions and scores 0.3236
on v6 without ever seeing it; the obvious question -- "could you just fine-tune
one adapter forward through those same five versions and do as well?" -- has no
answer in the study. This produces that answer.

Design points that are NOT incidental:

* **The held-out version is refused if it appears in the chain.** Same leak guard
  as materialize_generated_any.py: a chain that trains on the version it is later
  scored on is not a held-out test, it is an in-domain one, and the failure is
  silent because the numbers still look plausible.
* **Every hyperparameter is copied from the independent per-version YAML** (via
  gen_yaml_seq4b.py, which this wraps), so the only differences from the `pv`
  arms are the training ORDER, the number of epochs, and which version is absent.
* **The two folds are not symmetric, deliberately.** Holding out v6 leaves a chain
  ending on v5, the version temporally adjacent to the target. Holding out v1
  leaves a chain ending on v6, the version furthest from it. Section 10's recency
  gradient says the last stage dominates, so these two folds pull recency and
  similarity apart: if recency governs, v6-held-out should do much better than
  v1-held-out; if similarity governs, both should track their nearest trained
  neighbour. Do not average the folds together -- that is the comparison.

    python scripts/gen_yaml_holdout.py --holdout v6 --backbone 4b --epochs 1

**`--pooled`** generates a different baseline entirely, and the one T2L_PLAN E5
says every generator has to beat before "conditioned generation" means anything:
ONE ordinary LoRA trained on the five training versions' corpora POOLED, with
no conditioning, no bank and no routing. It is dose-matched to the bank (five
corpora x the per-version epoch count), it is what `hash ~ vision ~ none` would
reduce every generator to, and the study has never had a valid number for it --
`out/cells/pooled/all` saw all six versions and its eval errored on 39 of 40
episodes. Unlike the chain it has no stage order, so it also separates "training
on everything" from "training on everything in temporal order" (12).

    python scripts/gen_yaml_holdout.py --holdout v1 --backbone 9b --pooled
"""
from __future__ import print_function

import argparse
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT)
os.chdir(PROJECT)

ALL = ["v1", "v2", "v3", "v4", "v5", "v6"]
SRC = {"4b": "ver4b", "9b": "ver9b6"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--holdout", required=True, choices=ALL)
    ap.add_argument("--backbone", required=True, choices=sorted(SRC))
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--grad-accum", type=int, default=None,
                    help="per-version recipe is batch 1 x accum 4 (effective 4). "
                         "On N GPUs pass 4/N to hold the effective batch fixed, or "
                         "the chain is not step-matched to the pv arms it is being "
                         "compared against.")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--pooled", action="store_true",
                    help="one LoRA on the five training corpora POOLED, "
                         "instead of a five-stage chain (T2L_PLAN E5 baseline)")
    ap.add_argument("--dataset-prefix", default="adaptercl_ver_%s",
                    help="dataset_info name template, one %%s for the version "
                         "key. scout_plan.md arm A pools the MANUAL-injected "
                         "corpora instead of the plain ones: "
                         "adaptercl_scout_ver_%%s. The leak check runs on the "
                         "written yaml either way.")
    args = ap.parse_args(argv)

    order = [v for v in ALL if v != args.holdout]          # temporal order preserved
    if args.pooled:
        return gen_pooled(args, order)
    if args.holdout in order:                              # unreachable, kept explicit
        raise SystemExit("refusing: held-out version is in the chain")
    tag = args.tag or "ho%s_%s" % (args.holdout[1:], args.backbone)

    print("holdout   : %s  (evaluated on this version ONLY)" % args.holdout)
    print("chain     : %s" % " -> ".join(order))
    print("tag       : %s" % tag)
    print("epochs    : %g per stage" % args.epochs)
    print("src recipe: %s" % SRC[args.backbone])

    cmd = [sys.executable, "scripts/gen_yaml_seq4b.py",
           "--src-tag", SRC[args.backbone], "--tag", tag,
           "--order", ",".join(order), "--epochs", "%g" % args.epochs]
    if args.grad_accum is not None:
        cmd += ["--grad-accum", str(args.grad_accum)]
    rc = subprocess.call(cmd)
    if rc != 0:
        return rc

    # Post-hoc leak check on what was actually WRITTEN, not on what we intended:
    # read every generated yaml back and confirm the held-out version appears in
    # no dataset line. The intent check above cannot catch a bad --src-tag whose
    # per-version yaml points at the wrong dataset.
    ydir = os.path.join(PROJECT, "out", "cells", tag, "yaml")
    bad = []
    for f in sorted(os.listdir(ydir)):
        if not f.endswith(".yaml"):
            continue
        for ln in open(os.path.join(ydir, f)):
            if ln.startswith("dataset:") and args.holdout in ln:
                bad.append((f, ln.strip()))
    if bad:
        for f, ln in bad:
            print("LEAK: %s -> %s" % (f, ln))
        raise SystemExit("refusing: held-out version %s appears in a dataset line"
                         % args.holdout)
    print("leak check: OK -- %s appears in no dataset line" % args.holdout)
    print("final adapter will be: out/cells/%s/s%d_%s" % (tag, len(order), order[-1]))
    return 0


def gen_pooled(args, order):
    """One unconditioned LoRA over the five training corpora."""
    tag = args.tag or "po%s_%s" % (args.holdout[1:], args.backbone)
    src_dir = os.path.join(PROJECT, "out", "cells", SRC[args.backbone], "yaml")
    src = os.path.join(src_dir, "_gen_%s.yaml" % order[0])
    if not os.path.isfile(src):
        raise SystemExit("missing %s -- generate the per-version YAMLs first"
                         % src)
    body = open(src).read()
    out_dir = os.path.join(PROJECT, "out", "cells", tag, "pooled")
    if "%s" not in args.dataset_prefix:
        raise SystemExit("--dataset-prefix needs exactly one %s for the "
                         "version key, got %r" % ("%s", args.dataset_prefix))
    datasets = ",".join(args.dataset_prefix % v for v in order)

    print("holdout   : %s  (evaluated on this version ONLY)" % args.holdout)
    print("pooled on : %s" % ", ".join(order))
    print("tag       : %s" % tag)
    print("src recipe: %s" % SRC[args.backbone])
    print("datasets  : %s" % (args.dataset_prefix % "<v>"))

    # Copy the per-version recipe and change exactly three things, so the only
    # differences from the `pv` arms are WHICH corpora and how many epochs.
    body, n = re.subn(r"^dataset:.*$", "dataset: " + datasets, body, count=1,
                      flags=re.M)
    if n != 1:
        raise SystemExit("could not rewrite dataset in %s" % src)
    body, n = re.subn(r"^output_dir:.*$", "output_dir: " + out_dir, body,
                      count=1, flags=re.M)
    if n != 1:
        raise SystemExit("could not rewrite output_dir in %s" % src)
    if args.epochs is not None:
        body, n = re.subn(r"^num_train_epochs:.*$",
                          "num_train_epochs: %g" % args.epochs, body, count=1,
                          flags=re.M)
        if n != 1:
            raise SystemExit("could not rewrite num_train_epochs in %s" % src)
    if args.grad_accum is not None:
        body = re.sub(r"^gradient_accumulation_steps:.*$",
                      "gradient_accumulation_steps: %d" % args.grad_accum,
                      body, count=1, flags=re.M)
    body = re.sub(r"^(model_name_or_path:.*\n)",
                  r"\1### POOLED-LOO: one LoRA over %s; %s is HELD OUT "
                  "entirely.\n" % (", ".join(order), args.holdout),
                  body, count=1, flags=re.M)

    ydir = os.path.join(PROJECT, "out", "cells", tag, "yaml")
    if not os.path.isdir(ydir):
        os.makedirs(ydir)
    path = os.path.join(ydir, "_gen_pooled.yaml")
    with open(path, "w") as fh:
        fh.write(body)

    # Post-hoc leak check on what was WRITTEN, not on what we intended -- the
    # same rule as the chain path: a bad --src-tag would otherwise put the
    # held-out version's corpus in a dataset line without anyone noticing.
    held = args.dataset_prefix % args.holdout
    bad = [ln.strip() for ln in open(path)
           if ln.startswith("dataset") and held in ln]
    if bad:
        for ln in bad:
            print("LEAK: %s" % ln)
        raise SystemExit("refusing: held-out version %s appears in a dataset "
                         "line" % args.holdout)
    n_ds = len([ln for ln in open(path) if ln.startswith("dataset:")])
    print("wrote %s" % path)
    print("leak check: OK -- %s appears in no dataset line (%d dataset line, "
          "%d corpora)" % (args.holdout, n_ds, len(order)))
    print("final adapter will be: %s" % out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
