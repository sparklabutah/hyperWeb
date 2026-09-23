#!/usr/bin/env python3
"""Train a Text-to-LoRA style hypernetwork and materialise a held-out adapter.

Reconstruction mode, following the reference implementation's primary recipe:
the hypernetwork regresses the *delta weights* of adapters trained
independently per version ("oracle" adapters), conditioned only on a natural
language description of the target interface.

    L(theta) = E_{v in train, s in sites} | dW_v,s  -  Bhat(v,s) Ahat(v,s) |

Two properties of this objective are worth stating because they are not
incidental:

* The target is the **product** dW = BA, never the factors. (RA, BR^-1) induces
  the same dW for any invertible R, so regressing the factors would fit an
  arbitrary gauge -- and our oracles were each trained from their own random
  init, so they share no gauge at all.
* It requires **no forward pass through the 9B policy**. Training touches only
  the oracle adapters and the hypernetwork, which is why this arm costs minutes
  where the behaviour-cloning arm costs an hour.

Leave-one-out is enforced the same way as for the mixture generator: the
held-out version contributes neither a description nor an oracle adapter to
training. Unlike the mixture arm there is no bank to leak through, since
nothing is retrieved -- the weights are regressed from scratch.

    python scripts/train_t2l.py --train-cells v2,v3,v4,v5,v6 --eval-cells v1 \
        --run-name t2l_loo_v1 --steps 2000
"""
from __future__ import print_function

import argparse
import json
import math
import os
import random
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT)
os.chdir(PROJECT)

from adaptercl import materialize, t2l, targets  # noqa: E402

LOSS_SCALE = 1e4    # |dW| ~ 5e-5; without this the L1 loss underflows fp32 logs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-cells", required=True)
    ap.add_argument("--eval-cells", required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--oracle-root", default="out/cells/ver",
                    help="comma-separated adapter roots. More than one gives "
                         "the same version several targets (T2L_PLAN E6: the "
                         "3-epoch and 6-epoch banks as 2 targets/version), "
                         "which is the cheapest way to stop the regressor "
                         "fitting one arbitrary member of the gauge class.")
    ap.add_argument("--descriptions", default=None, help="JSON override")
    # Conditioning is a separate axis from architecture. T2L's own choice is
    # text, but our mixture arm is vision-conditioned, so comparing T2L-text
    # against mixture-vision would confound "direct generation vs routing" with
    # "text vs pixels". Running T2L on the SAME vision embeddings the mixture
    # arm used isolates the architecture change.
    ap.add_argument("--conditioning", default="text", choices=["text", "vision"])
    ap.add_argument("--embeddings", default="out/hypernet/cond_ver.npz",
                    help="vision conditioning .npz (used when --conditioning vision)")
    ap.add_argument("--target-set", default="attn_mlp")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=16.0,
                    help="alpha the GENERATED adapter is written with; the "
                         "served scale is alpha/rank")
    # T2L_PLAN E4a. The oracle bank is written alpha 32 / r 16 and therefore
    # SERVED at alpha/r = 2, but this script has always regressed it at 1 --
    # so the target was half the adapter the policy was actually trained with,
    # and the shrunken output was then served at 1 as well. `auto` reads the
    # oracle's own config so target scale == the scale that oracle earns its
    # numbers at; pair it with --alpha 32 to serve on the same convention.
    ap.add_argument("--oracle-scale", default="1.0",
                    help="alpha/r at which the oracle dW target is formed: a "
                         "float, or 'auto' to read each oracle's own config")
    ap.add_argument("--allow-scale-mismatch", action="store_true",
                    help="permit served alpha/r != oracle target alpha/r "
                         "(reproduces the pre-E4a runs on purpose)")
    ap.add_argument("--variant", default="M", choices=["M", "L"],
                    help="M = one output head per weight SHAPE (the paper's, "
                         "and what every existing checkpoint used); L = one "
                         "per module type")
    ap.add_argument("--fixed-a", default=None,
                    choices=["init", "bank_mean"],
                    help="T2L_PLAN E3a: hold A frozen and regress B only. "
                         "'bank_mean' averages the TRAIN oracles' A factors "
                         "(cosine 0.90-0.95, so this loses little) and fixes "
                         "the gauge across the whole bank.")
    ap.add_argument("--loss", default="l1",
                    choices=["l1", "mse", "cosine_logmag", "activation"],
                    help="reconstruction loss (T2L_PLAN E6). cosine_logmag "
                         "scores direction and magnitude separately, which is "
                         "the pair 10 showed are traded off against each other "
                         "and that a single L1 conflates. `activation` scores "
                         "|(BhatAhat - BA)x| on cached real inputs -- the only "
                         "one of the four that does not reward reproducing an "
                         "arbitrary member of the gauge class; needs "
                         "--act-cache from scripts/cache_site_inputs.py.")
    ap.add_argument("--act-cache", default=None,
                    help="site-input cache for --loss activation. It must have "
                         "been built on the TRAIN versions only; this script "
                         "refuses a cache that saw the held-out version.")
    ap.add_argument("--logmag-weight", type=float, default=1.0,
                    help="weight on the |log magnitude ratio| term of "
                         "--loss cosine_logmag")
    ap.add_argument("--learn-scale", action="store_true",
                    help="add a trainable per-module-type log-gain on B")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--sites-per-step", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--d-hidden", type=int, default=128)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args(argv)

    train = [c.strip() for c in args.train_cells.split(",") if c.strip()]
    evalc = [c.strip() for c in args.eval_cells.split(",") if c.strip()]
    leak = sorted(set(train) & set(evalc))
    if leak:
        raise SystemExit("refusing: %s appear in BOTH train and eval; the "
                         "held-out version must contribute no description and "
                         "no oracle adapter" % leak)

    out_dir = args.out_dir or os.path.join("out", "t2l", args.run_name)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print("=" * 66)
    print("T2L reconstruction  run=%s" % args.run_name)
    print("  train : %s" % ", ".join(train))
    print("  held  : %s   (no description, no oracle adapter in training)"
          % ", ".join(evalc))
    print("=" * 66)

    # -- conditioning -------------------------------------------------------
    keys = train + evalc
    desc = t2l.load_descriptions(args.descriptions)
    if args.conditioning == "vision":
        # The same frozen vision embeddings the mixture generator consumed.
        # Precomputing them is memoisation of a deterministic frozen tower, not
        # supervision: producing one needs a screenshot and nothing else, so a
        # held-out version having an entry does not leak its demonstrations.
        from adaptercl import encoder as _enc
        emb_all, _ = _enc.load_embeddings(args.embeddings)
        missing = [c for c in keys if c not in emb_all]
        if missing:
            raise SystemExit("no vision embedding for %s in %s"
                             % (missing, args.embeddings))
        print("\nvision conditioning from %s" % args.embeddings)
        H = {k: torch.as_tensor(emb_all[k]).float() for k in keys}
        cond_name = "vision/" + os.path.basename(args.embeddings)
    else:
        missing = [c for c in keys if c not in desc]
        if missing:
            raise SystemExit("no description for %s" % missing)
        print("\nencoding descriptions with %s ..." % t2l.DEFAULT_TEXT_ENCODER)
        tc = t2l.TextConditioner(device=args.device)
        emb = tc.encode([desc[k] for k in keys])
        H = {k: emb[i] for i, k in enumerate(keys)}
        cond_name = "text/" + t2l.DEFAULT_TEXT_ENCODER
        del tc
    print("  d_cond=%d" % cond_dim(H))
    C = torch.stack([H[k] for k in keys])
    S = (C @ C.T)
    off = S[~torch.eye(len(keys), dtype=torch.bool)]
    print("  pairwise cosine: mean %.4f  min %.4f  max %.4f"
          % (float(off.mean()), float(off.min()), float(off.max())))
    if args.device != "cpu":
        torch.cuda.empty_cache()

    # -- oracle targets, TRAIN versions only --------------------------------
    tcfg = targets.load_text_config()
    sites = targets.enumerate_sites(tcfg, args.target_set)
    roots = [r.strip() for r in args.oracle_root.split(",") if r.strip()]
    serve_scale = args.alpha / float(args.rank)
    print("\nloading oracle deltas for %d train version(s) over %d sites "
          "from %d bank(s) ..." % (len(train), len(sites), len(roots)))
    # {version: [ {rel_name: (A, B, scale)} per bank ]}. A version with two
    # banks has two equally valid targets; sampling between them is E6's
    # "both banks as 2 targets/version".
    oracle, target_scales = {}, set()
    for v in train:
        oracle[v] = []
        for root in roots:
            d = os.path.join(root, v)
            if not os.path.isdir(d):
                raise SystemExit("missing oracle adapter %s" % d)
            sc = oracle_scale_for(d, args.oracle_scale, args.rank)
            target_scales.add(round(sc, 9))
            # Factored, not materialised: see t2l.oracle_factors -- holding
            # every dW is ~21 GB per version and OOM-kills the step.
            oracle[v].append(t2l.oracle_factors(d, sites, alpha_over_r=sc))
            mags = [float((B @ A).abs().mean() * s2)
                    for (A, B, s2) in list(oracle[v][-1].values())[:16]]
            print("  %-4s %-28s %d sites, target alpha/r %.3g, mean|dW| %.3e"
                  % (v, root, len(oracle[v][-1]), sc,
                     sum(mags) / max(len(mags), 1)))

    # ONE alpha convention (T2L_PLAN Part D / E4a): what we regress and what we
    # serve must be the same scale, or the emitted adapter is a fixed multiple
    # of the thing that was learned and every magnitude conclusion is off by it.
    if len(target_scales) > 1:
        raise SystemExit("oracle banks disagree on alpha/r (%s); pass an "
                         "explicit --oracle-scale" % sorted(target_scales))
    target_scale = target_scales.pop()
    if abs(serve_scale / target_scale - 1.0) > 1e-6:
        msg = ("alpha convention mismatch: regressing an oracle at alpha/r "
               "%.4g but writing the adapter at alpha/r %.4g (x%.4g)."
               % (target_scale, serve_scale, serve_scale / target_scale))
        if not args.allow_scale_mismatch:
            raise SystemExit(msg + " Pass --alpha %g, or "
                             "--allow-scale-mismatch to do it on purpose."
                             % (target_scale * args.rank))
        print("\nWARNING: " + msg + "  (--allow-scale-mismatch)")

    fixed_kw = {}
    if args.fixed_a:
        fixed_kw["fixed_A"] = args.fixed_a
        if args.fixed_a == "bank_mean":
            # TRAIN roots only -- the held-out version contributes no A either.
            fixed_kw["bank_adapters"] = [os.path.join(roots[0], v)
                                         for v in train]
            fixed_kw["bank_keys"] = list(train)
    gen = t2l.T2LHypernet(sites, rank=args.rank, d_cond=cond_dim(H),
                          d_hidden=args.d_hidden, depth=args.depth,
                          variant=args.variant, learn_scale=args.learn_scale,
                          n_layers=tcfg["num_hidden_layers"],
                          **fixed_kw).to(args.device)
    n = sum(p.numel() for p in gen.parameters() if p.requires_grad)
    print("\nT2L-{} generator: {:,} trainable ({:.1f}M){}".format(
        args.variant, n, n / 1e6,
        "  [A frozen, B only]" if args.fixed_a else ""))

    opt = torch.optim.AdamW([p for p in gen.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=args.weight_decay)
    usable = [s for s in sites
              if all(s.rel_name in f for v in train for f in oracle[v])]
    print("  usable sites (present in every oracle): %d" % len(usable))
    print("  loss: %s" % args.loss)

    act = None
    if args.loss == "activation":
        if not args.act_cache:
            raise SystemExit("--loss activation needs --act-cache (build it "
                             "with scripts/cache_site_inputs.py)")
        blob = torch.load(args.act_cache, map_location="cpu",
                          weights_only=False)
        leak = sorted(set(blob.get("cells", [])) & set(evalc))
        if leak:
            raise SystemExit(
                "refusing: the activation cache %s was built on %s, which "
                "includes the held-out version. Rebuild it with --cells %s."
                % (args.act_cache, blob.get("cells"), ",".join(train)))
        act = blob["inputs"]
        usable = [s for s in usable if s.rel_name in act]
        n_rows = int(next(iter(act.values())).shape[0])
        print("  activation cache: %s, %d sites, %d rows, built on %s"
              % (args.act_cache, len(act), n_rows,
                 ",".join(blob.get("cells", []))))
        print("  usable sites after intersecting with the cache: %d"
              % len(usable))
        if not usable:
            raise SystemExit("the activation cache covers none of the sites")

    hist, t0 = [], time.time()
    for step in range(args.steps):
        lr = args.lr * (step + 1) / args.warmup if step < args.warmup else \
            0.5 * args.lr * (1 + math.cos(math.pi * min(1.0, (step - args.warmup) /
                                                        max(1, args.steps - args.warmup))))
        for gparam in opt.param_groups:
            gparam["lr"] = lr

        opt.zero_grad(set_to_none=True)
        v = random.choice(train)
        # Do not draw when there is only one bank: an extra RNG call would
        # shift the site sequence and make this script irreproducible against
        # every run recorded before multi-bank targets existed.
        fac = oracle[v][0] if len(oracle[v]) == 1 else random.choice(oracle[v])
        chosen = random.sample(usable, min(args.sites_per_step, len(usable)))
        h = H[v].to(args.device)
        tot = 0.0
        # One site at a time with immediate backward: a single dW can be
        # 12288x4096, so materialising the whole graph would not fit.
        for s in chosen:
            A, B = gen.factors_for_site(h, s)
            if act is not None:
                # Never form dW: with x cached, B(Ax) is (d_out, r) @ (r, n),
                # so the whole term is two thin matmuls instead of a
                # 12288x4096 product. This is also why activation matching is
                # CHEAPER than the L1 loss it replaces, not more expensive.
                x = act[s.rel_name].to(args.device, torch.float32).t()
                Ao, Bo, sc = fac[s.rel_name]
                Ao = Ao.to(args.device)
                Bo = Bo.to(args.device)
                loss = ((B @ (A @ x)) - (Bo @ (Ao @ x)) * sc).abs().mean() \
                    * LOSS_SCALE / len(chosen)
            else:
                dw = B @ A
                tgt = t2l.oracle_delta_at(fac, s.rel_name, args.device)
                loss = recon_loss(dw, tgt, args.loss, args.logmag_weight) \
                    / len(chosen)
            loss.backward()
            tot += float(loss.detach())
        gnorm = torch.nn.utils.clip_grad_norm_(gen.parameters(), args.grad_clip)
        opt.step()
        hist.append({"step": step, "loss": tot, "lr": lr, "gnorm": float(gnorm)})
        if step % args.log_every == 0 or step == args.steps - 1:
            print("  step %5d  loss %8.4f  lr %.2e  |g| %.3f  %6.1fs"
                  % (step, tot, lr, float(gnorm), time.time() - t0))
            sys.stdout.flush()

    torch.save({"generator": gen.state_dict(), "step": args.steps,
                "rank": args.rank, "alpha": args.alpha,
                "target_set": args.target_set, "train_cells": train,
                "eval_cells": evalc, "d_cond": cond_dim(H),
                "conditioning": cond_name, "variant": args.variant,
                "fixed_A": args.fixed_a, "loss": args.loss,
                "act_cache": args.act_cache,
                "oracle_roots": roots,
                "target_alpha_over_r": target_scale,
                "served_alpha_over_r": serve_scale,
                "sites": [s.as_dict() for s in sites]},
               os.path.join(out_dir, "checkpoints", "final.pt"))
    with open(os.path.join(out_dir, "history.json"), "w") as fh:
        json.dump(hist, fh)

    # -- baseline for the loss: how well does the *mean* oracle do? ---------
    # A reconstruction loss with no reference is uninterpretable. The natural
    # floor is predicting the mean of the training oracles, ignoring the
    # description entirely: if the trained model does not beat that, the
    # conditioning is doing nothing.
    print("\nreference: predict the mean training oracle (ignores conditioning)")
    with torch.no_grad():
        num = den = 0.0
        for s in usable[:32]:
            if act is not None:
                x = act[s.rel_name].float().t()
                ys = []
                for v in train:
                    Ao, Bo, sc = oracle[v][0][s.rel_name]
                    ys.append((Bo @ (Ao @ x)) * sc)
                mean_y = torch.stack(ys).mean(0)
                for y in ys:
                    num += float((mean_y - y).abs().mean()) * LOSS_SCALE
                    den += 1
                continue
            dws = [t2l.oracle_delta_at(oracle[v][0], s.rel_name) for v in train]
            mean_dw = torch.stack(dws).mean(0)
            for dw in dws:
                num += float(recon_loss(mean_dw, dw, args.loss,
                                        args.logmag_weight))
                den += 1
        print("  mean-oracle %s loss: %.4f" % (args.loss, num / den))

    # -- generate + materialise the held-out adapter ------------------------
    gen.eval()
    for v in evalc:
        with torch.no_grad():
            h = H[v].to(args.device)
            factors = {}
            for s in sites:
                A, B = gen.factors_for_site(h, s)
                factors[s.rel_name] = (A.detach().cpu(), B.detach().cpu())
        d = os.path.join(out_dir, "generated", v)
        materialize.write_adapter(
            d, factors, sites, args.rank, args.alpha,
            sorted(set(s.module for s in sites)),
            meta={"generator": "t2l", "run": args.run_name,
                  "train_cells": train, "held_out": v,
                  "conditioning": cond_name, "variant": args.variant,
                  "fixed_A": args.fixed_a, "loss": args.loss,
                  "oracle_roots": roots,
                  "trained_alpha_over_r": target_scale,
                  "served_alpha_over_r": serve_scale,
                  "description": desc.get(v)})
        mag = sum(float((B @ A).abs().mean()) for A, B in
                  [factors[s.rel_name] for s in usable[:16]]) / 16
        print("  wrote %s\n    mean|dW| raw %.3e, SERVED at alpha/r %.3g -> "
              "%.3e  (oracle at this convention ~%.3e)"
              % (d, mag, serve_scale, mag * serve_scale, 5e-5 * target_scale))
    print("\ndone in %.1f min" % ((time.time() - t0) / 60))
    return 0


def cond_dim(H):
    return int(next(iter(H.values())).numel())


def oracle_scale_for(adapter_dir, spec, rank):
    """alpha/r at which an oracle's dW target is formed."""
    if str(spec).lower() != "auto":
        return float(spec)
    with open(os.path.join(adapter_dir, "adapter_config.json")) as fh:
        cfg = json.load(fh)
    r = float(cfg.get("r", rank))
    return float(cfg.get("lora_alpha", r)) / r


def recon_loss(dw, tgt, kind="l1", logmag_weight=1.0):
    """Reconstruction loss between a generated and an oracle delta (E6).

    `l1` / `mse` are the usual elementwise regressions and both hedge toward
    the conditional mean, whose magnitude is 0.64-0.66 of a member (A.3) --
    which is precisely the 26-68 % magnitude collapse 10 measured.

    `cosine_logmag` scores the two things separately: `1 - cos` on the
    flattened deltas, plus `|log(|dW_hat| / |dW|)|`. Averaging over targets no
    longer shrinks the prediction, because the magnitude term is scale-free and
    the direction term is normalised -- the conditional mean is still the
    optimum for the direction, but it no longer drags the norm down with it.
    All are multiplied by LOSS_SCALE so the numbers stay readable next to the
    published L1 curves.
    """
    if kind == "l1":
        return (dw - tgt).abs().mean() * LOSS_SCALE
    if kind == "mse":
        return ((dw - tgt) ** 2).mean() * (LOSS_SCALE ** 2)
    if kind == "cosine_logmag":
        a, b = dw.reshape(-1), tgt.reshape(-1)
        cos = torch.nn.functional.cosine_similarity(a, b, dim=0, eps=1e-12)
        na = a.norm().clamp_min(1e-20)
        nb = b.norm().clamp_min(1e-20)
        return (1.0 - cos) + logmag_weight * (na / nb).log().abs()
    raise ValueError("unknown loss %r" % (kind,))


if __name__ == "__main__":
    sys.exit(main())
