#!/usr/bin/env python3
"""Generate per-version QLoRA YAMLs from the existing per-version LoRA YAMLs.

QLoRA = the SAME LoRA adapter trained on top of a base model held in 4-bit NF4
instead of bf16. It is the standard memory-efficient baseline, and here it also
probes something the study cares about: if the adapter is mostly teaching an
output protocol (findings section 1) rather than fine interface knowledge, then
crushing the base model to 4 bits should cost little. If instead QLoRA collapses,
the frozen base's precision is carrying more of the task than section 1 implies.

Each YAML is a byte-for-byte copy of the corresponding full-precision YAML with
only these changes:

    output_dir:            -> out/cells/<tag>/<v>
    + quantization_bit: 4          (NF4, double-quantised)
    + quantization_method: bitsandbytes
    - deepspeed:                   (REMOVED -- see below)

Copying rather than regenerating keeps rank/alpha/lr/epochs/batch identical to
the full-precision arm, so QLoRA-vs-LoRA is attributable to the quantisation
alone. In particular BOTH backbones already sit at 6 epochs, so this inherits
the epoch match that section 5 had to be corrected for.

DeepSpeed ZeRO-3 is removed deliberately, not incidentally: ZeRO-3 shards
parameters across ranks, and bitsandbytes 4-bit params are opaque uint8 blobs
with their quantisation state attached, which ZeRO-3 cannot shard or gather
correctly. The combination either errors at init or silently trains on corrupted
weights. Each QLoRA run is therefore single-GPU -- which is fine, because 4-bit
is precisely what makes a single card enough.

    python scripts/gen_yaml_qlora.py --src-tag ver4b  --tag ver4bq
    python scripts/gen_yaml_qlora.py --src-tag ver9b6 --tag ver9b6q
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

VERSIONS = ["v1", "v2", "v3", "v4", "v5", "v6"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-tag", required=True, help="tag holding the full-precision YAMLs")
    ap.add_argument("--tag", required=True, help="tag to write QLoRA YAMLs under")
    ap.add_argument("--versions", default=",".join(VERSIONS))
    ap.add_argument("--bits", type=int, default=4, choices=(4, 8))
    ap.add_argument("--quant-type", default="nf4", choices=("nf4", "fp4"))
    args = ap.parse_args(argv)

    versions = [v.strip() for v in args.versions.split(",") if v.strip()]
    src_dir = os.path.join(PROJECT, "out", "cells", args.src_tag, "yaml")
    out_dir = os.path.join(PROJECT, "out", "cells", args.tag, "yaml")
    if not os.path.isdir(src_dir):
        raise SystemExit("missing %s" % src_dir)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    made = []
    for v in versions:
        src = os.path.join(src_dir, "_gen_%s.yaml" % v)
        if not os.path.isfile(src):
            raise SystemExit("missing %s" % src)
        body = open(src).read()

        stage_out = os.path.join(PROJECT, "out", "cells", args.tag, v)
        body, n = re.subn(r"^output_dir:.*$", "output_dir: %s" % stage_out,
                          body, count=1, flags=re.M)
        if n != 1:
            raise SystemExit("could not rewrite output_dir in %s" % src)

        # Drop ZeRO-3: incompatible with 4-bit params (see module docstring).
        body, n_ds = re.subn(r"^deepspeed:.*\n", "", body, count=1, flags=re.M)

        # Insert the quantisation block right after the base model line so it is
        # visible at the top of the file rather than buried.
        qblock = (
            "### QLoRA: base model in %d-bit %s, adapter trained in bf16 on top.\n"
            "### Everything else is copied verbatim from %s/_gen_%s.yaml so that\n"
            "### any difference from the full-precision arm is the quantisation.\n"
            "quantization_bit: %d\n"
            "quantization_method: bitsandbytes\n"
            "quantization_type: %s\n"
            "double_quantization: true\n"
            % (args.bits, args.quant_type, args.src_tag, v, args.bits, args.quant_type)
        )
        body, n2 = re.subn(r"^(model_name_or_path:.*\n)", r"\1" + qblock,
                           body, count=1, flags=re.M)
        if n2 != 1:
            raise SystemExit("could not find model_name_or_path in %s" % src)

        dst = os.path.join(out_dir, "_gen_%s.yaml" % v)
        with open(dst, "w") as fh:
            fh.write(body)
        made.append(dst)

        # Read the file back and report from THAT, never from the in-memory
        # string: a previous generator in this project substituted after writing
        # and then printed the in-memory value, so the log and the file disagreed
        # and the mismatch was self-confirming.
        txt = open(dst).read()

        def _f(pat, d="?"):
            m = re.search(pat, txt, flags=re.M)
            return m.group(1) if m else d

        print("  %-3s bits=%s type=%s rank=%s alpha=%s lr=%s epochs=%s deepspeed=%s"
              % (v, _f(r"^quantization_bit:\s*(\S+)"), _f(r"^quantization_type:\s*(\S+)"),
                 _f(r"^lora_rank:\s*(\S+)"), _f(r"^lora_alpha:\s*(\S+)"),
                 _f(r"^learning_rate:\s*(\S+)"), _f(r"^num_train_epochs:\s*(\S+)"),
                 "REMOVED" if n_ds else "absent"))
        if re.search(r"^deepspeed:", txt, flags=re.M):
            raise SystemExit("deepspeed survived in %s -- refusing (ZeRO-3 + 4-bit)" % dst)

    print("\n%d QLoRA YAML(s) under %s" % (len(made), out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
