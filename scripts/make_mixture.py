#!/usr/bin/env python3
"""Build a MIXTURE-OF-ADAPTERS LoRA from K trained per-site adapters.

The comparison this serves is "one pooled LoRA vs a mixture of per-domain
LoRAs" on the four leave-one-site-out settings. The pooled arms already exist
(`out/cells/loso6_4b/*`); this writes their mixture counterparts.

WHY CONCATENATION AND NOT AVERAGING THE FACTORS. The thing a mixture should
combine is the weight DELTAS, not the factors. Averaging factors gives

    A = sum_k c_k A_k,  B = sum_k c_k B_k
    =>  dW = B A = sum_{j,k} c_j c_k B_j A_k

which is quadratic in c and carries cross terms `B_j A_k` pairing one adapter's
B with another's A -- a product no trained adapter ever contained. The honest
object is

    dW = sum_k c_k (alpha_k / r_k) B_k A_k

and it is represented EXACTLY by stacking along the rank axis:

    A_cat = concat_k(c_k A_k)   (K*r, d_in)
    B_cat = concat_k(B_k)       (d_out, K*r)

because `B_cat @ A_cat = sum_k c_k B_k A_k` -- the cross terms are structurally
absent. This is `hypernet.MixtureGenerator`'s `mode="delta"`, done offline so
the result is an ordinary PEFT adapter vLLM can serve with no new code.

THE SCALING, which is the one thing that silently ruins this. PEFT applies
`alpha/r`, so writing rank `K*r` while keeping `alpha = 32` would scale the
delta by `32/(K*16)` instead of the `32/16 = 2.0` every component was trained
with -- a K-fold shrink that reads as "the mixture is weak" rather than as a
bug. The output alpha is therefore set so `alpha_out / r_out` equals the
components' own `alpha/r`:

    r_out = K * r,  alpha_out = (alpha_k / r_k) * r_out

For K=2 that is r=32 / alpha=64; for K=3, r=48 / alpha=96. `--verify` checks the
materialised delta against the intended mixture directly, per site, and is on by
default because getting this wrong does not error: vLLM would serve a quietly
mis-scaled adapter (adapterCL.md 4.5).

    python scripts/make_mixture.py \
        --components out/cells/site6_4b/wiki_v6_d1 out/cells/site6_4b/news_v6_d1 \
        --out out/cells/mix6_4b/mixL_shop_d1
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


def load_components(dirs):
    """Read K adapters and refuse anything that would make the mixture invalid."""
    comps = []
    for d in dirs:
        if not os.path.isfile(os.path.join(d, "adapter_config.json")):
            raise SystemExit("BLOCKED: %s has no adapter_config.json" % d)
        f, cfg, meta = materialize.read_adapter(d)
        comps.append({"dir": d, "factors": f, "cfg": cfg, "meta": meta,
                      "scaling": float(cfg["lora_alpha"]) / float(cfg["r"])})
    ref = comps[0]
    for c in comps[1:]:
        if set(c["factors"]) != set(ref["factors"]):
            raise SystemExit(
                "BLOCKED: %s covers %d sites, %s covers %d -- a mixture over "
                "different injection sites is not a mixture"
                % (c["dir"], len(c["factors"]), ref["dir"], len(ref["factors"])))
        if int(c["cfg"]["r"]) != int(ref["cfg"]["r"]):
            raise SystemExit("BLOCKED: rank %s != %s (%s vs %s)"
                             % (c["cfg"]["r"], ref["cfg"]["r"], c["dir"], ref["dir"]))
        if abs(c["scaling"] - ref["scaling"]) > 1e-9:
            raise SystemExit(
                "BLOCKED: %s was trained at alpha/r=%g, %s at %g. Mixing them "
                "would silently reweight one component."
                % (c["dir"], c["scaling"], ref["dir"], ref["scaling"]))
        if c["cfg"]["base_model_name_or_path"] != ref["cfg"]["base_model_name_or_path"]:
            raise SystemExit("BLOCKED: different base models (%s vs %s)"
                             % (c["cfg"]["base_model_name_or_path"],
                                ref["cfg"]["base_model_name_or_path"]))
    return comps


def build(comps, weights):
    """Stack into (factors, r_out, alpha_out). Exact in delta space."""
    r = int(comps[0]["cfg"]["r"])
    scaling = comps[0]["scaling"]
    r_out = r * len(comps)
    alpha_out = scaling * r_out
    out = {}
    for key in comps[0]["factors"]:
        As, Bs = [], []
        for c, w in zip(comps, weights):
            A, B = c["factors"][key]
            As.append(float(w) * A.to(torch.float32))
            Bs.append(B.to(torch.float32))
        out[key] = (torch.cat(As, dim=0), torch.cat(Bs, dim=1))
    return out, r_out, alpha_out


def verify(comps, weights, factors, r_out, alpha_out, n_check=8, tol=2e-2):
    """The materialised delta must equal sum_k w_k * (alpha/r) * B_k A_k.

    Checked in float32 on a sample of sites, as a RELATIVE error against the
    intended delta's own magnitude -- the components are bf16, so an absolute
    tolerance would either pass everything or fail everything depending on the
    layer's scale.
    """
    s_out = alpha_out / float(r_out)
    keys = sorted(factors)[:n_check]
    worst = 0.0
    for key in keys:
        A, B = factors[key]
        got = s_out * (B @ A)
        want = torch.zeros_like(got)
        for c, w in zip(comps, weights):
            Ak, Bk = c["factors"][key]
            want += float(w) * c["scaling"] * (Bk.to(torch.float32)
                                               @ Ak.to(torch.float32))
        denom = float(want.abs().mean()) or 1.0
        err = float((got - want).abs().mean()) / denom
        worst = max(worst, err)
        if err > tol:
            raise SystemExit(
                "BLOCKED: site %s materialises a delta %.3g relative error from "
                "the intended mixture (tol %.3g). The adapter was NOT written."
                % (key, err, tol))
    return worst, len(keys)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--components", nargs="+", required=True,
                    help="the per-site adapter dirs to mix, in a fixed order")
    ap.add_argument("--out", required=True, help="adapter dir to write")
    ap.add_argument("--weights", default=None,
                    help="comma-separated mixture weights (default: uniform "
                         "1/K). They are NOT renormalised -- pass what you mean.")
    ap.add_argument("--target-set", default="attn_mlp")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the delta-space check (do not)")
    ap.add_argument("--bf16", action="store_true",
                    help="store bf16 instead of fp32 (halves the files)")
    args = ap.parse_args(argv)

    comps = load_components(args.components)
    K = len(comps)
    if args.weights:
        weights = [float(x) for x in args.weights.replace(" ", ",").split(",") if x]
        if len(weights) != K:
            raise SystemExit("BLOCKED: %d weights for %d components"
                             % (len(weights), K))
    else:
        weights = [1.0 / K] * K

    factors, r_out, alpha_out = build(comps, weights)

    print("=== mixture of %d adapters ===" % K)
    for c, w in zip(comps, weights):
        print("  w=%.4f  r=%s alpha=%s  %s"
              % (w, c["cfg"]["r"], c["cfg"]["lora_alpha"],
                 os.path.relpath(c["dir"], PROJECT)))
    print("  -> r_out=%d alpha_out=%g (scaling %g, components' own %g)"
          % (r_out, alpha_out, alpha_out / r_out, comps[0]["scaling"]))

    if not args.no_verify:
        worst, n = verify(comps, weights, factors, r_out, alpha_out)
        print("  verify: %d sites checked, worst relative delta error %.2e" % (n, worst))

    base = comps[0]["cfg"]["base_model_name_or_path"]
    cfg = targets.load_text_config(base)
    sites = targets.enumerate_sites(cfg, args.target_set)
    have = set(factors)
    sites = [s for s in sites if s.rel_name in have]
    if len(sites) != len(have):
        raise SystemExit("BLOCKED: %d factors but only %d match enumerated "
                         "sites for target_set=%s"
                         % (len(have), len(sites), args.target_set))

    meta = {"kind": "mixture_of_adapters", "mode": "delta_concat",
            "n_components": K, "weights": list(weights),
            "components": [os.path.relpath(c["dir"], PROJECT) for c in comps],
            "component_rank": int(comps[0]["cfg"]["r"]),
            "component_scaling": comps[0]["scaling"],
            "built_by": "scripts/make_mixture.py"}
    materialize.write_adapter(
        args.out, factors, sites, r_out, alpha_out, args.target_set,
        base_model=base, meta=meta,
        dtype=(torch.bfloat16 if args.bf16 else None))
    print("  wrote %s" % os.path.relpath(args.out, PROJECT))
    print("  NOTE: serve with MAX_LORA_RANK >= %d" % r_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
