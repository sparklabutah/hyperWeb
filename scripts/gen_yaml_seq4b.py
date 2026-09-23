#!/usr/bin/env python3
"""Generate the SEQUENTIAL (continual) LoRA chain on Qwen3.5-4B: v1 -> v2 -> ... -> v6.

One adapter is carried through all six versions in order, each stage resuming
from the previous stage's weights and training for the SAME number of epochs the
independent per-version LoRAs used. The final adapter is then evaluated on all
six versions.

This is the continual-learning baseline the rest of the study is implicitly
argued against: if one adapter could simply be fine-tuned forward through the
version history, neither per-version adapters nor a generator would be needed.
The expected outcome is catastrophic forgetting -- strong on v6 (trained last),
weak on v1-v5 -- and that is the point, so no attempt is made to mitigate it
(no replay, no regularisation, no LR decay across stages).

Each stage's YAML is a byte-for-byte copy of the corresponding independent
`ver4b` YAML with exactly two lines changed:

    output_dir:            -> out/cells/seq4b/s<i>_v<i>
    adapter_name_or_path:  -> previous stage's output_dir   (absent for stage 1)

Copying rather than regenerating is deliberate: it makes it impossible for a
hyperparameter to drift between the sequential arm and the independent arm, so
any difference in results is attributable to the training ORDER alone.

    python scripts/gen_yaml_seq4b.py
    bash scripts/run_seq4b.sh <jobid> <gpu>

`--order` takes ARBITRARY unit keys, so the same generator builds the SITE
stream that domain_transfer_timewarp_plan.md §3 and the companion plan's arm A1
share -- wiki -> news -> shop -> multi and the reverse, on the dose-matched
site corpora rather than on the era ones:

    python scripts/gen_yaml_seq4b.py --src-tag site4b --draw 1 \
        --order wiki,news,shop,multi --tag siteseq4b --epochs 6

`--draw` appends the source tag's draw suffix (`wiki` -> `wiki_d1`) to every
element and to the output tag, because the site YAMLs are per-draw: a chain
built from `--order wiki,news,...` against a per-draw tag would look for
`_gen_wiki.yaml` and die. One chain per draw is one seed, matching the `S_*`
arms it is compared against; FINDINGS §12's chain was NOT dose-matched and this
one is, which is the only reason the comparison is readable.
"""
from __future__ import print_function

import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT)
os.chdir(PROJECT)

ORDER = ["v1", "v2", "v3", "v4", "v5", "v6"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-tag", default="ver4b", help="tag holding the independent YAMLs")
    ap.add_argument("--tag", default="seq4b")
    ap.add_argument("--order", default=",".join(ORDER),
                    help="comma-separated unit keys in training order. Era keys "
                         "(v1..v6) or SITE keys (wiki,news,shop,multi,all).")
    ap.add_argument("--draw", type=int, default=None,
                    help="site chains only: append _d<draw> to every unit key "
                         "and to the tag, so one chain is one seed against the "
                         "matching per-draw site corpora")
    ap.add_argument("--lr", default=None,
                    help="override learning_rate. Large steps are the main lever "
                         "for inducing interference: at lr=1e-4 the chain showed "
                         "positive backward transfer, so a destructive regime "
                         "needs a materially bigger update per stage.")
    ap.add_argument("--lora-rank", type=int, default=None,
                    help="override lora_rank. More capacity lets each stage carve "
                         "out version-specific directions instead of sharing one "
                         "low-rank subspace, which is a precondition for forgetting.")
    ap.add_argument("--grad-accum", type=int, default=None,
                    help="override gradient_accumulation_steps. seqHI -- the arm "
                         "that produced forgetting -- ran 8 GPUs x batch 1 x accum 1, "
                         "i.e. EFFECTIVE BATCH 8. A chain on N GPUs must use "
                         "accum=8/N to keep the same effective batch, or the step "
                         "size is not actually being held constant and the "
                         "comparison to seqHI is void.")
    ap.add_argument("--epochs", type=float, default=None,
                    help="override num_train_epochs per stage (default: keep the "
                         "independent recipe's value). Everything else is copied "
                         "verbatim so ONLY epochs and the resume chain differ.")
    args = ap.parse_args(argv)

    order = [v.strip() for v in args.order.split(",") if v.strip()]
    tag = args.tag
    if args.draw is not None:
        order = ["%s_d%d" % (v, args.draw) if not v.endswith("_d%d" % args.draw)
                 else v for v in order]
        if tag == "seq4b":                       # the era default; name the draw
            tag = "siteseq4b_d%d" % args.draw
        elif not tag.endswith("_d%d" % args.draw):
            tag = "%s_d%d" % (tag, args.draw)
    src_dir = os.path.join(PROJECT, "out", "cells", args.src_tag, "yaml")
    out_yaml = os.path.join(PROJECT, "out", "cells", tag, "yaml")
    if not os.path.isdir(src_dir):
        raise SystemExit("missing %s -- run gen_yaml_4b.py (eras) or "
                         "gen_yaml_site.py (sites) first" % src_dir)
    # Name every missing stage up front. Reporting them one at a time costs one
    # round trip per typo, and a site chain has four stages whose keys carry a
    # draw suffix that is easy to get wrong.
    missing = [v for v in order
               if not os.path.isfile(os.path.join(src_dir, "_gen_%s.yaml" % v))]
    if missing:
        have = sorted(f[5:-5] for f in os.listdir(src_dir)
                      if f.startswith("_gen_") and f.endswith(".yaml"))
        raise SystemExit("missing stage YAML(s) %s in %s (have: %s)"
                         % (", ".join(missing), src_dir, ", ".join(have)))
    if not os.path.isdir(out_yaml):
        os.makedirs(out_yaml)

    print("sequential chain: %s" % " -> ".join(order))
    prev_out = None
    made = []
    for i, v in enumerate(order, 1):
        src = os.path.join(src_dir, "_gen_%s.yaml" % v)
        if not os.path.isfile(src):
            raise SystemExit("missing %s" % src)
        body = open(src).read()

        stage_out = os.path.join(PROJECT, "out", "cells", tag, "s%d_%s" % (i, v))
        n_before = len(body)
        body, n = re.subn(r"^output_dir:.*$",
                          "output_dir: %s" % stage_out, body, count=1, flags=re.M)
        if n != 1:
            raise SystemExit("could not rewrite output_dir in %s" % src)

        if prev_out is not None:
            # Continue training the SAME adapter. create_new_adapter must stay
            # false (the default) -- setting it true would start a fresh adapter
            # on top of the old one and silently turn this into six independent
            # LoRAs stacked, not one adapter trained sequentially.
            hdr = ("### SEQUENTIAL stage %d/%d: resumes the adapter trained on %s\n"
                   "adapter_name_or_path: %s\n"
                   "create_new_adapter: false\n" % (i, len(order), order[i - 2], prev_out))
            body = re.sub(r"^(model_name_or_path:.*\n)", r"\1" + hdr, body, count=1, flags=re.M)
        else:
            body = re.sub(r"^(model_name_or_path:.*\n)",
                          r"\1### SEQUENTIAL stage 1/%d: fresh adapter\n" % len(order),
                          body, count=1, flags=re.M)

        # Apply EVERY edit before writing. An earlier version substituted the
        # epoch count after the write and then reported the in-memory value, so
        # the files said 6 while the log said 1 -- a silent, self-confirming
        # mismatch. Mutate, then write, then read the file back to report.
        # The replacement must reuse the YAML KEY, not the CLI flag name. An
        # earlier version wrote "lr: ..." over the learning_rate line, which
        # deletes the real key -- the trainer then falls back to its default and
        # the run looks like it used the requested LR when it did not.
        for key, val in (("learning_rate", args.lr), ("lora_rank", args.lora_rank),
                         ("gradient_accumulation_steps", args.grad_accum)):
            if val is None:
                continue
            body, nn = re.subn(r"^%s:.*$" % key, "%s: %s" % (key, val),
                               body, count=1, flags=re.M)
            if nn != 1:
                raise SystemExit("could not rewrite %s in %s" % (key, src))
        if args.epochs is not None:
            body, ne = re.subn(r"^num_train_epochs:.*$",
                               "num_train_epochs: %g" % args.epochs, body,
                               count=1, flags=re.M)
            if ne != 1:
                raise SystemExit("could not rewrite num_train_epochs in %s" % src)

        dst = os.path.join(out_yaml, "_seq_%d_%s.yaml" % (i, v))
        with open(dst, "w") as fh:
            fh.write(body)
        made.append(dst)
        txt = open(dst).read()
        def _f(pat):
            m = re.search(pat, txt, flags=re.M)
            return m.group(1) if m else "?"
        print("  stage %d  %-3s epochs=%s lr=%s rank=%s accum=%s  resume=%s"
              % (i, v, _f(r"^num_train_epochs:\s*(\S+)"), _f(r"^learning_rate:\s*(\S+)"),
                 _f(r"^lora_rank:\s*(\S+)"),
                 _f(r"^gradient_accumulation_steps:\s*(\S+)"),
                 os.path.basename(prev_out) if prev_out else "(fresh)"))
        prev_out = stage_out

    print("\nfinal adapter will be: %s" % prev_out)
    print("%d YAML(s) under %s" % (len(made), out_yaml))
    return 0


if __name__ == "__main__":
    sys.exit(main())
