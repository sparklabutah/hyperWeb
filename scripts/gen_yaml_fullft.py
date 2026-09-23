#!/usr/bin/env python3
"""Generate per-version FULL fine-tuning YAMLs from the per-version LoRA YAMLs.

The point of this arm is to answer "does a full fine-tune beat a rank-16 adapter
on the same data?", so everything that is not the finetuning method is copied
verbatim from the LoRA YAML: same dataset (`adaptercl_ver_vN`), same
cutoff_len 65536, same batch/accum, same epochs.

Three things DO change, each for a stated reason:

  finetuning_type: lora -> full      the thing being tested
  lora_rank / lora_alpha  removed    meaningless without an adapter
  learning_rate 1e-4 -> 1e-5         the LoRA YAML carries an explicit deviation
                                     comment: "1.0e-5 -> 1.0e-4 (standard LoRA
                                     lr; the full-SFT lr underfits an adapter)".
                                     Full fine-tuning wants the original 1e-5;
                                     running a full FT at 1e-4 would be a
                                     10x-too-large step, which section 10 shows
                                     is destructive on this benchmark.
  deepspeed -> ZeRO-3 WITH offload   a LoRA optimiser state is tiny, a full one
                                     is not: 4B needs ~48 GB of fp32 master +
                                     Adam moments before activations, 9B ~110 GB.

    python3 scripts/gen_yaml_fullft.py --src-tag ver4b  --tag ft4b
    python3 scripts/gen_yaml_fullft.py --src-tag ver9b6 --tag ft9b
"""
from __future__ import print_function
import argparse, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT); os.chdir(PROJECT)
REPO = os.path.dirname(PROJECT)
DS_OFFLOAD = os.path.join(REPO, "LLaMA-Factory", "examples", "deepspeed",
                          "ds_z3_offload_config.json")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-tag", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--lr", default="1.0e-5")
    ap.add_argument("--versions", default="v1,v2,v3,v4,v5,v6")
    ap.add_argument("--cutoff", default=None,
                    help="override cutoff_len. REQUIRED for the 9B: at 65536 the "
                         "per-GPU activation exceeds a 140GB H200 even with ZeRO-3 "
                         "offload, gradient checkpointing, fa2, Liger fused-CE and "
                         "micro-batch 1 -- adding ranks does not help because each "
                         "rank still holds batch-1 activations for the full "
                         "sequence. 32768 is the project's own full-SFT precedent "
                         "(qwen3-5-4b-v6single). This is a REAL deviation from the "
                         "LoRA arms at 65536 and must be reported as one.")
    args = ap.parse_args(argv)

    if not os.path.isfile(DS_OFFLOAD):
        raise SystemExit("missing deepspeed config %s" % DS_OFFLOAD)
    src_dir = os.path.join(PROJECT, "out", "cells", args.src_tag, "yaml")
    out_dir = os.path.join(PROJECT, "out", "cells", args.tag, "yaml")
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    for v in [x.strip() for x in args.versions.split(",") if x.strip()]:
        src = os.path.join(src_dir, "_gen_%s.yaml" % v)
        if not os.path.isfile(src):
            raise SystemExit("missing %s" % src)
        body = open(src).read()
        body, n = re.subn(r"^finetuning_type:.*$", "finetuning_type: full", body, count=1, flags=re.M)
        if n != 1:
            raise SystemExit("no finetuning_type in %s" % src)
        body = re.sub(r"^lora_\w+:.*\n", "", body, flags=re.M)          # drop adapter-only keys
        body = re.sub(r"^learning_rate:.*$", "learning_rate: %s" % args.lr, body, count=1, flags=re.M)
        body = re.sub(r"^deepspeed:.*$", "deepspeed: %s" % DS_OFFLOAD, body, count=1, flags=re.M)
        if args.cutoff:
            body, nc = re.subn(r"^cutoff_len:.*$", "cutoff_len: %s" % args.cutoff,
                               body, count=1, flags=re.M)
            if nc != 1:
                raise SystemExit("could not rewrite cutoff_len in %s" % src)
        out = os.path.join(PROJECT, "out", "fullft", "%s_%s" % (args.tag, v))
        body = re.sub(r"^output_dir:.*$", "output_dir: %s" % out, body, count=1, flags=re.M)

        dst = os.path.join(out_dir, "_gen_%s.yaml" % v)
        open(dst, "w").write(body)
        txt = open(dst).read()          # report from the FILE, never the buffer
        g = lambda p, d="?": (re.search(p, txt, re.M).group(1) if re.search(p, txt, re.M) else d)
        print("  %-3s type=%s lr=%s ep=%s ctx=%s ds=%s lora_keys=%d -> %s"
              % (v, g(r"^finetuning_type:\s*(\S+)"), g(r"^learning_rate:\s*(\S+)"),
                 g(r"^num_train_epochs:\s*(\S+)"), g(r"^cutoff_len:\s*(\S+)"),
                 os.path.basename(g(r"^deepspeed:\s*(\S+)")),
                 len(re.findall(r"^lora_", txt, re.M)), os.path.basename(out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
