#!/usr/bin/env python3
"""Rescale a generated adapter's delta-W to a target magnitude, keeping its direction.

This is the control that separates two explanations that are otherwise perfectly
confounded in our T2L results:

  (a) text conditioning is a WORSE conditioner -> a worse adapter direction
  (b) text conditioning produces a SMALLER delta-W -> a milder intervention on
      the policy, which reverts toward the frozen base's failure to terminate

In every fold, the text-conditioned run has both the lower reconstruction loss
AND the smaller |dW| (26-68% of oracle), so nothing in the existing data can
tell (a) from (b). Rescaling changes ONLY the magnitude and leaves the direction
untouched: dW = (alpha/r) B A, so multiplying B by k multiplies dW by exactly k
while every singular direction is preserved.

If the rescaled text adapter recovers the vision arm's score, magnitude is the
mediator and the reconstruction objective (L1, which hedges toward the
conditional mean) is what is costing performance. If it does not, the
conditioning modality is what matters and the magnitude story is wrong.

    python scripts/rescale_adapter.py \
        --src out/t2l/t2lL_txt_v6/generated/v6 \
        --ref out/cells/ver/v6 \
        --out out/t2l/rescaled/txt_v6
"""
from __future__ import print_function

import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT)
os.chdir(PROJECT)

from adaptercl import materialize, targets  # noqa: E402


def mean_abs_dw(factors, names, scale):
    tot = n = 0.0
    for k in names:
        A, B = factors[k]
        A = torch.as_tensor(A, dtype=torch.float32)
        B = torch.as_tensor(B, dtype=torch.float32)
        tot += float(((B @ A) * scale).abs().mean())
        n += 1
    return tot / max(n, 1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="generated adapter to rescale")
    ap.add_argument("--ref", default=None, help="reference adapter whose |dW| to match")
    ap.add_argument("--scale", type=float, default=None,
                    help="explicit scale factor k (mutually exclusive with --ref). "
                         "Matching the oracle is NOT known to be optimal -- on v6 "
                         "k=6.13 raised answer rate but collapsed accuracy -- so an "
                         "explicit sweep is how the usable band gets measured.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-set", default="attn_mlp")
    ap.add_argument("--base-model", default=None)
    ap.add_argument("--probe-sites", type=int, default=24,
                    help="sites sampled to estimate |dW| (full set is slow and unnecessary)")
    args = ap.parse_args(argv)

    cfg = targets.load_text_config(args.base_model)
    sites = targets.enumerate_sites(cfg, args.target_set)

    if args.ref is None and args.scale is None:
        raise SystemExit("need --ref or --scale")
    sf, scfg, smeta = materialize.read_adapter(args.src)
    s_scale = float(scfg.get("lora_alpha", scfg.get("r"))) / float(scfg.get("r"))
    if args.ref:
        rf, rcfg, _ = materialize.read_adapter(args.ref)
        r_scale = float(rcfg.get("lora_alpha", rcfg.get("r"))) / float(rcfg.get("r"))
        common = [s.rel_name for s in sites
                  if s.rel_name in sf and s.rel_name in rf][: args.probe_sites]
    else:
        rf, r_scale = {}, s_scale
        common = [s.rel_name for s in sites if s.rel_name in sf][: args.probe_sites]
    if not common:
        raise SystemExit("no overlapping sites between src and ref")

    m_src = mean_abs_dw(sf, common, s_scale)
    if m_src <= 0:
        raise SystemExit("source |dW| is zero; nothing to rescale")
    if args.scale is not None:
        k = float(args.scale); m_ref = m_src * k
    else:
        m_ref = mean_abs_dw(rf, common, r_scale)
        k = m_ref / m_src

    print("  probe sites      : %d" % len(common))
    print("  src  mean|dW|    : %.4e  (%s)" % (m_src, args.src))
    print("  ref  mean|dW|    : %.4e  (%s)" % (m_ref, args.ref))
    print("  scale factor k   : %.4f" % k)

    # Apply to B only: dW = (alpha/r) B A scales by exactly k, direction intact.
    out_factors = {}
    for name, (A, B) in sf.items():
        A = torch.as_tensor(A, dtype=torch.float32)
        B = torch.as_tensor(B, dtype=torch.float32) * k
        out_factors[name] = (A, B)

    site_by_name = {s.rel_name: s for s in sites}
    keep = [site_by_name[n] for n in out_factors if n in site_by_name]
    materialize.write_adapter(
        args.out, out_factors, keep, int(scfg["r"]), float(scfg.get("lora_alpha", scfg["r"])),
        sorted(set(s.module for s in keep)),
        meta={"rescaled_from": args.src, "reference": args.ref,
              "scale_factor": k, "src_mean_abs_dw": m_src, "ref_mean_abs_dw": m_ref,
              "note": "magnitude-only control: B scaled by k, direction unchanged",
              "parent_meta": smeta})

    chk = mean_abs_dw(materialize.read_adapter(args.out)[0], common, s_scale)
    print("  written          : %s" % args.out)
    print("  verify mean|dW|  : %.4e  (target %.4e, ratio %.3f)" % (chk, m_ref, chk / m_ref))
    if abs(chk / m_ref - 1.0) > 0.02:
        print("  WARNING: rescaled magnitude is off target by >2%")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
