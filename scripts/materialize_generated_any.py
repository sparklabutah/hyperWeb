#!/usr/bin/env python3
"""Materialize generated adapters from a trained hypernetwork -- any backbone.

scripts/materialize_generated.py hardcodes the 9B bank (out/cells/ver/) and the
default base model, so it cannot emit the 4B mixture adapters. This is that
script with two things parameterised and NOTHING else changed:

    BANK_TAG   which per-version adapter bank the generator was trained against
               (ver for 9B, ver4b for 4B). Must match the generator's --bank-tag,
               or the mixture coefficients index a different set of adapters.
    ADAPTERCL_BASE_MODEL   picked up by targets.load_text_config(); decides
               hidden size and therefore the site geometry.

The DEFAULT is deliberately PRESERVED, not "fixed": the handle is built with
alpha=32.0 while the generators are trained with --alpha 16. That 2x scaling on
the emitted dW is already baked into every published 9B mixture/LOO number, and
section 8 of the findings shows results are strongly magnitude-sensitive (an
inverted U in |dW|). Changing the default here would silently make the 4B arm
incomparable to the 9B arm it exists to be compared against.

T2L_PLAN E4c revisits it explicitly instead. `ALPHA` and `OUT_SUB` make the
convention an argument rather than a constant, so the fair 1x comparator can be
built ALONGSIDE the published 2x adapters rather than replacing them:

    # the published arm, unchanged (served at 2x its trained scale)
    python3 scripts/materialize_generated_any.py loo_v1 mixture v2,v3,v4,v5,v6 v1

    # E4c: the same generator at the scale it was TRAINED with
    ALPHA=16 OUT_SUB=generated_x1 \\
      python3 scripts/materialize_generated_any.py loo_v1 mixture v2,v3,v4,v5,v6 v1

`TRAINED_ALPHA` (default 16, matching every --alpha 16 run) is asserted against
the served scale by `hypernet.materialize_generated`; a mismatch needs an
explicit `ALLOW_SCALE_MISMATCH=1`, which is what the legacy default sets.

    BANK_TAG=ver4b ADAPTERCL_BASE_MODEL=Qwen/Qwen3.5-4B \\
      python3 scripts/materialize_generated_any.py mix4b_all mixture
    BANK_TAG=ver4b ADAPTERCL_BASE_MODEL=Qwen/Qwen3.5-4B \\
      python3 scripts/materialize_generated_any.py mix4b_loo_v1 mixture v2,v3,v4,v5,v6 v1
"""
import glob
import os
import sys

import torch

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)
os.chdir(PROJECT)

from adaptercl import encoder, targets, hypernet, cells, materialize  # noqa: E402,F401

run = sys.argv[1]
gen_kind = sys.argv[2]
BANK_TAG = os.environ.get("BANK_TAG", "ver")
RANK = int(os.environ.get("RANK", "16"))
ALPHA = float(os.environ.get("ALPHA", "32.0"))
TRAINED_ALPHA = float(os.environ.get("TRAINED_ALPHA", "16.0"))
OUT_SUB = os.environ.get("OUT_SUB", "generated")
EMBEDDINGS = os.environ.get("EMBEDDINGS", "out/hypernet/cond_ver.npz")
ALLOW_MISMATCH = os.environ.get("ALLOW_SCALE_MISMATCH",
                                "1" if ALPHA != TRAINED_ALPHA else "0") == "1"

bank_keys = sys.argv[3].split(",") if len(sys.argv) > 3 else [v.key for v in cells.ALL_VERSIONS]
gen_keys = sys.argv[4].split(",") if len(sys.argv) > 4 else [v.key for v in cells.ALL_VERSIONS]

# Leak guard: a LOO generator's bank holds only its five TRAIN adapters. If the
# held-out version were also in the bank, "zero-shot generation" would just be a
# lookup of the adapter we withheld.
leaked = [k for k in gen_keys if k in bank_keys]
if len(sys.argv) > 4 and leaked:
    raise SystemExit("refusing: %s are in the bank AND being generated -- "
                     "that is a lookup, not synthesis" % leaked)

cks = sorted(glob.glob("out/hypernet/%s/checkpoints/*.pt" % run))
if not cks:
    raise SystemExit("no checkpoints under out/hypernet/%s" % run)
# Prefer final.pt; sorted() puts stepNNN before final only by luck of the name.
final = [c for c in cks if c.endswith("final.pt")]
ck = final[0] if final else cks[-1]
sd = torch.load(ck, map_location="cpu")
print("checkpoint", ck, "step", sd.get("step"))
print("bank tag  ", BANK_TAG, "| base model", os.environ.get("ADAPTERCL_BASE_MODEL", "(default)"))

cfg = targets.load_text_config()
sites = targets.enumerate_sites(cfg, "attn_mlp")
kw = dict(n_layers=cfg["num_hidden_layers"])
if gen_kind == "mixture":
    kw["adapters"] = ["out/cells/%s/%s" % (BANK_TAG, k) for k in bank_keys]
    kw["keys"] = list(bank_keys)
    missing = [a for a in kw["adapters"] if not os.path.isfile(os.path.join(a, "adapter_config.json"))]
    if missing:
        raise SystemExit("bank adapters missing (wrong BANK_TAG?): %s" % missing)

# The run's own config.json records the --generator-kwargs it was TRAINED with
# (e.g. {"head_rank": 32}, which makes each head a 2-layer Sequential). Building
# the generator without them yields a different module tree and load_state_dict
# fails with "Missing key heads.q_proj.a.weight / Unexpected key heads.q_proj.a.0
# .weight". Merge them in; explicit kwargs above still win.
_cfgp = "out/hypernet/%s/config.json" % run
if os.path.isfile(_cfgp):
    import json as _json
    _gk = (_json.load(open(_cfgp)) or {}).get("generator_kwargs") or {}
    for _k, _v in _gk.items():
        kw.setdefault(_k, _v)
    if _gk:
        print("generator_kwargs from config.json:", _gk)

emb, _ = encoder.load_embeddings(EMBEDDINGS)
d_cond = int(len(next(iter(emb.values()))))
gen = hypernet.build_generator(gen_kind, sites, RANK, d_cond, **kw)
gen.load_state_dict(sd["generator"])
gen.eval()


class H(object):
    """Minimal handle: generate_for_cells needs site geometry only, no weights."""

    def __init__(self, sites, rank, alpha):
        self.sites, self.rank, self.alpha = sites, rank, alpha

    def query_ids(self, device=None):
        li = torch.tensor([s.layer for s in self.sites], dtype=torch.long)
        mi = torch.tensor([s.module_id for s in self.sites], dtype=torch.long)
        if device is not None:
            li, mi = li.to(device), mi.to(device)
        return li, mi

    def __len__(self):
        return len(self.sites)


h = H(sites, RANK, ALPHA)   # default alpha 32.0 matches the 9B pipeline
print("served alpha/r %.4g | trained alpha/r %.4g | out %s"
      % (ALPHA / RANK, TRAINED_ALPHA / RANK, OUT_SUB))
factors = hypernet.generate_for_cells(gen, {k: emb[k] for k in gen_keys},
                                      handle=h, keys=list(gen_keys))
written = hypernet.materialize_generated(factors, h,
                                         "out/hypernet/%s/%s" % (run, OUT_SUB),
                                         run="", generator=gen, base_model=None,
                                         trained_scaling=TRAINED_ALPHA / RANK,
                                         allow_scale_mismatch=ALLOW_MISMATCH,
                                         meta={"generator": gen_kind, "run": run,
                                               "bank_tag": BANK_TAG,
                                               "embeddings": EMBEDDINGS,
                                               "base_model": os.environ.get("ADAPTERCL_BASE_MODEL")})
for k, d in sorted(written.items()):
    print(" ", k, "->", d)
print("emitted rank:", gen.output_rank)
