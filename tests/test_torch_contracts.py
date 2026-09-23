#!/usr/bin/env python3
"""Cross-module contract tests for the modelling layer (torch, CPU, toy model only).

Run: python \
         adapter_project/tests/test_torch_contracts.py

No GPU, no checkpoint, no network: everything runs against `toy.tiny_model()`,
whose module tree is byte-for-byte the one `targets.enumerate_sites` predicts.

The contracts, in order of how much they cost if they break:

  1. GENERATE -> SERVE. inject -> generator -> set_factors -> forward ->
     write_from_handle -> read_adapter -> load_into_handle -> forward again must
     produce IDENTICAL logits, for all three generator families. This is what
     makes a generated adapter servable at all; if it drifts, every eval number
     in Phase 3 is measuring a different adapter than the one that was trained.
  2. MERGE == SERVE. materialize.merge_into_model reproduces the InjectedLoRALinear
     forward, so merge-then-serve and LoRA-then-serve are the same model.
  3. Per-example factors (Bsz, r, d_in) equal the unbatched path example by
     example -- a BC batch mixes cells, so this path is the training path.
  4. inject() freezes the base, and gradients reach the generator and only the
     generator (4.1: "only the generator trains").
  5. The lora_alpha rescale for a rank-K*r generator (mixture in mode="delta").
     This was a real, silent, already-fixed numerical bug; this test is what
     keeps it fixed. It asserts BOTH that the rescaled adapter matches the live
     generator AND that the un-rescaled one would not.
  6. Zero-init really is a no-op: base and adapted logits are bitwise equal at
     init for `free` and `basis` (and MixtureGenerator's starts_as_noop=False is
     honest about not being one).
"""
from __future__ import print_function

import collections
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT)

import torch
import torch.nn as nn

from adaptercl import hypernet, inject, materialize, targets, toy

PASS = [0]
FAIL = [0]

D_COND = 12
RANK = 2
TARGET_SET = "attn_mlp"


def check(name, cond, extra=""):
    if cond:
        PASS[0] += 1
        print("  ok   %s %s" % (name, extra))
    else:
        FAIL[0] += 1
        print("  FAIL %s %s" % (name, extra))


class _Quiet(object):
    def write(self, s):
        pass

    def flush(self):
        pass


def _silent(fn, *a, **kw):
    """Call `fn` with stdout swallowed (materialize prints a note per write)."""
    old = sys.stdout
    sys.stdout = _Quiet()
    try:
        return fn(*a, **kw)
    finally:
        sys.stdout = old


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

def _toy(rank=RANK, alpha=None, target_set=TARGET_SET, seed=0):
    model, cfg = toy.tiny_model(seed=seed)
    model.eval()
    handle = inject.inject(model, target_set, rank=rank, alpha=alpha, cfg=cfg)
    return model, cfg, handle


def _cond(seed=3, batch=None):
    g = torch.Generator().manual_seed(seed)
    shape = (D_COND,) if batch is None else (batch, D_COND)
    return torch.randn(*shape, generator=g)


def _fake_bank(sites, rank, k=3, seed=1):
    """K in-memory 'trained' adapters, in the {rel_name: (A, B)} form
    `_load_adapter_bank` accepts -- so the mixture needs no adapters on disk."""
    g = torch.Generator().manual_seed(seed)
    bank = collections.OrderedDict()
    for i in range(k):
        d = collections.OrderedDict()
        for s in sites:
            d[s.rel_name] = (torch.randn(rank, s.d_in, generator=g) * 0.10,
                             torch.randn(s.d_out, rank, generator=g) * 0.10)
        bank["cell_%d" % i] = d
    return bank


def _build(kind, handle, rank=RANK, **kw):
    if kind == "mixture":
        kw.setdefault("adapters", _fake_bank(handle.sites, rank))
    torch.manual_seed(11)
    return hypernet.build_generator(kind, handle.sites, rank, D_COND, **kw)


def _wake_up(gen):
    """Take a zero-init generator out of its no-op start.

    Every generator here begins as an exact no-op by design (8.3), so a round
    trip on a fresh one would pass trivially: 0 == 0. This is the post-first-step
    state, done by hand so the test needs no optimiser.
    """
    with torch.no_grad():
        if gen.kind == "free":
            for head in gen.heads.values():
                last = head.b[-1] if isinstance(head.b, nn.Sequential) else head.b
                last.weight.normal_(0.0, 0.05)
                if last.bias is not None:
                    last.bias.normal_(0.0, 0.05)
        elif gen.kind == "basis":
            gen.out_scale.fill_(0.7)
    return gen


def _logits(model, ids):
    with torch.no_grad():
        return model(ids).logits


# --------------------------------------------------------------------------
# 1. generate -> write -> read -> load -> identical logits
# --------------------------------------------------------------------------

def test_generate_serve_round_trip(tmp):
    for kind in hypernet.GENERATOR_KINDS:
        model, cfg, handle = _toy()
        ids, labels = toy.tiny_batch(cfg, batch=2, seq=5)
        base_logits = _logits(model, ids)

        # mode="factor" so the emitted rank equals the handle's rank; the
        # rank-K*r default (mode="delta") cannot go through a handle at all and
        # is covered by test_alpha_rescale below.
        kw = {"mode": "factor"} if kind == "mixture" else {}
        gen = _wake_up(_build(kind, handle, **kw))
        check("%s: emitted rank == handle rank (round trip is possible)" % kind,
              gen.output_rank == handle.rank)

        factors = hypernet.generate_for_cells(gen, {"wiki_e5": _cond()}, handle)
        handle.load_state(factors["wiki_e5"])
        logits1 = _logits(model, ids)
        check("%s: the generated adapter actually changes the model" % kind,
              not inject.adapter_is_noop(handle)
              and not torch.equal(base_logits, logits1))

        out_dir = os.path.join(tmp, "rt_%s" % kind)
        _silent(materialize.write_from_handle, out_dir, handle,
                base_model="toy/tiny", meta={"kind": kind})
        read, cfg_json, meta = materialize.read_adapter(out_dir)
        check("%s: every site survived the write" % kind,
              set(read) == set(handle.rel_names) and len(read) == len(handle))
        check("%s: adapter_config records the handle's r/alpha" % kind,
              int(cfg_json["r"]) == handle.rank
              and abs(float(cfg_json["lora_alpha"]) - handle.alpha) < 1e-9)

        handle.clear()
        check("%s: cleared handle is back to the base model" % kind,
              torch.equal(base_logits, _logits(model, ids)))

        info = materialize.load_into_handle(handle, out_dir)
        logits2 = _logits(model, ids)
        check("%s: load_into_handle filled every site" % kind,
              info["missing"] == [] and info["loaded"] == len(handle))
        check("%s: GENERATE->WRITE->READ->LOAD gives IDENTICAL logits" % kind,
              torch.equal(logits1, logits2))

        # ...and the same adapter written under a DIFFERENT module prefix still
        # loads into this handle (materialize.py:190-196's rel_name claim).
        alt_sites = targets.enumerate_sites(
            cfg, TARGET_SET, prefix="base_model.model.model.language_model")
        alt_dir = os.path.join(tmp, "rt_%s_altprefix" % kind)
        _silent(materialize.write_adapter, alt_dir, factors["wiki_e5"], alt_sites,
                handle.rank, handle.alpha,
                sorted(set(s.module for s in handle.sites)))
        handle.clear()
        materialize.load_into_handle(handle, alt_dir)
        check("%s: an adapter written under another module prefix still loads" % kind,
              torch.equal(logits1, _logits(model, ids)))

        if kind == "free":
            # A partially-filled handle writes a PARTIAL adapter rather than
            # refusing (materialize.py:118-121 skips sites with no factors).
            # 4.5 warns that a partial LoRA over the DeltaNet group can crash
            # vLLM's expand_packed_lora, so the one thing that must hold is that
            # the provenance says so out loud.
            handle.clear()
            some = dict(list(factors["wiki_e5"].items())[:4])
            handle.load_state(some, strict=False)
            part_dir = os.path.join(tmp, "rt_partial")
            _silent(materialize.write_from_handle, part_dir, handle)
            _f, _c, part_meta = materialize.read_adapter(part_dir)
            check("partial: a partly-filled handle is written as a PARTIAL "
                  "adapter and the provenance says so",
                  part_meta["n_sites"] == 4 < len(handle.sites)
                  and len(part_meta["sites"]) == 4)
        handle.remove()


# --------------------------------------------------------------------------
# 2. merge-then-serve == LoRA-then-serve
# --------------------------------------------------------------------------

def test_merge_equals_injected(tmp):
    model, cfg, handle = _toy()
    ids, _labels = toy.tiny_batch(cfg, batch=2, seq=5)
    gen = _wake_up(_build("free", handle))
    factors = hypernet.generate_for_cells(gen, {"c": _cond()}, handle)
    handle.load_state(factors["c"])
    adapted = _logits(model, ids)

    out_dir = os.path.join(tmp, "merge")
    _silent(materialize.write_from_handle, out_dir, handle)

    # a pristine copy of the same base weights (tiny_model is seeded)
    fresh, _ = toy.tiny_model(seed=0)
    fresh.eval()
    check("merge: the fresh copy starts identical to the base",
          torch.equal(_logits(fresh, ids), _logits_without_adapter(model, handle, ids)))
    before = dict((n, p.detach().clone()) for n, p in fresh.named_parameters())
    n = materialize.merge_into_model(fresh, out_dir)
    merged = _logits(fresh, ids)

    # merge_into_model matches on `.layers.` anywhere in the module path, so it
    # must not reach anything outside the decoder stack. (Safe on Qwen3.5: the
    # vision tower is `.visual.blocks.N.*`, verified in transformers 5.6
    # modeling_qwen3_5.py:1020. A vision tower named `.layers.` would collide.)
    changed = set(nm for nm, p in fresh.named_parameters()
                  if not torch.equal(p, before[nm]))
    want_changed = set(s.name + ".weight" for s in handle.sites)
    check("merge: every site was merged", n == len(handle))
    check("merge: exactly the injected sites changed -- nothing outside them",
          changed == want_changed,
          "unexpected=%r" % sorted(changed - want_changed)[:3])
    check("merge: merged weights reproduce the injected forward",
          torch.allclose(adapted, merged, atol=1e-5, rtol=1e-4),
          "max|d|=%.2e" % float((adapted - merged).abs().max()))
    check("merge: and the merge really did something",
          not torch.allclose(merged, _logits(toy.tiny_model(seed=0)[0].eval(), ids),
                             atol=1e-5))

    # InjectedLoRALinear.merge_into_base is the in-place version of the same math
    site0 = handle.modules[0]
    dw = site0.delta_weight()
    check("merge: delta_weight == scaling * B @ A",
          torch.allclose(dw, site0.scaling * (site0.B @ site0.A), atol=0, rtol=0))
    handle.remove()


def _logits_without_adapter(model, handle, ids):
    saved = [(m.A, m.B) for m in handle.modules]
    handle.clear()
    out = _logits(model, ids)
    for m, (a, b) in zip(handle.modules, saved):
        m.set_factors(a, b)
    return out


# --------------------------------------------------------------------------
# 3. per-example factors == the unbatched path, example by example
# --------------------------------------------------------------------------

def test_per_example_factors():
    bsz = 3
    model, cfg, handle = _toy()
    ids, _labels = toy.tiny_batch(cfg, batch=bsz, seq=5)
    gen = _wake_up(_build("free", handle))
    layer_ids, module_ids = handle.query_ids()

    H = _cond(seed=5, batch=bsz)
    with torch.no_grad():
        A_list, B_list = gen(H, layer_ids, module_ids)
    check("batched: generator emits (Bsz, r, d_in) / (Bsz, d_out, r)",
          all(tuple(a.shape) == (bsz, handle.rank, d_in)
              and tuple(b.shape) == (bsz, d_out, handle.rank)
              for (d_in, d_out), a, b in zip(handle.shapes(), A_list, B_list)))
    check("batched: the three conditionings gave three different adapters",
          not torch.allclose(A_list[0][0], A_list[0][1]))

    handle.set_factors(A_list, B_list)
    batched = _logits(model, ids)

    worst = 0.0
    for i in range(bsz):
        handle.set_factors([a[i] for a in A_list], [b[i] for b in B_list])
        one = _logits(model, ids[i:i + 1])
        worst = max(worst, float((one[0] - batched[i]).abs().max()))
    check("batched: per-example factors match the unbatched path",
          worst < 1e-5, "max|d|=%.2e" % worst)

    # and the guard that stops a silently mis-aligned batch
    handle.set_factors(A_list, B_list)
    raised = False
    try:
        _logits(model, ids[:2])
    except ValueError:
        raised = True
    check("batched: a batch/factor size mismatch raises instead of broadcasting",
          raised)
    handle.remove()


# --------------------------------------------------------------------------
# 4. the base is frozen and only the generator learns
# --------------------------------------------------------------------------

def test_freeze_and_gradient_flow():
    model, cfg, handle = _toy()
    ids, labels = toy.tiny_batch(cfg, batch=2, seq=5)
    frozen = inject.check_base_frozen(model)
    check("freeze: every parameter is frozen after inject()",
          frozen["ok"], "leaked=%r" % frozen["leaked"][:3])

    gen = _wake_up(_build("free", handle))
    A_list, B_list = gen(_cond(), *handle.query_ids())
    handle.set_factors(A_list, B_list)

    res = inject.check_gradient_flow(handle, gen,
                                     lambda: model(ids, labels=labels).loss)
    check("grad: the task loss reaches the generator",
          res["ok"] and res["n_with_grad"] > 0,
          "%d/%d tensors" % (res["n_with_grad"], res["n_params"]))
    check("grad: it reaches the trunk, not only the output head",
          any(v > 0 for k, v in res["grad_norms"].items() if k.startswith("trunk")))
    check("grad: it reaches the layer and module embeddings",
          res["grad_norms"].get("layer_emb.weight", 0) > 0
          and res["grad_norms"].get("module_emb.weight", 0) > 0)
    with_grad = [n for n, p in model.named_parameters() if p.grad is not None]
    check("grad: NO base parameter received a gradient", not with_grad,
          "%r" % with_grad[:3])

    # static mode is ordinary LoRA: the only unfrozen tensors must be the factors
    model2, cfg2, handle2 = _toy()
    handle2.make_static()
    leaked = inject.check_base_frozen(model2)["leaked"]
    check("freeze: in static mode only the LoRA factors are trainable",
          leaked and all(n.endswith(".A") or n.endswith(".B") for n in leaked),
          "%d tensors" % len(leaked))
    check("freeze: static factors are exactly the handle's trainable set",
          len(handle2.trainable_parameters()) == 2 * len(handle2))
    handle.remove()
    handle2.remove()


# --------------------------------------------------------------------------
# 5. the lora_alpha rescale for a rank-K*r generator
# --------------------------------------------------------------------------

def test_alpha_rescale_for_delta_mixture(tmp):
    K = 3
    # alpha != rank so a dropped rescale cannot hide behind scaling == 1
    model, cfg, handle = _toy(alpha=2 * RANK)
    ids, _labels = toy.tiny_batch(cfg, batch=2, seq=5)
    gen = _build("mixture", handle, adapters=_fake_bank(handle.sites, RANK, k=K))
    check("alpha: a delta mixture emits rank K*r",
          gen.mode == "delta" and gen.output_rank == K * RANK
          and gen.n_basis == K)

    factors = hypernet.generate_for_cells(gen, {"wiki_e5": _cond()}, handle)
    handle.set_factors(*_unzip(factors["wiki_e5"], handle))
    live = _logits(model, ids)
    live_dw = [m.delta_weight() for m in handle.modules]

    written = _silent(hypernet.materialize_generated, factors, handle,
                      os.path.join(tmp, "gen"), run="r1", generator=gen,
                      base_model="toy/tiny")
    read, cfg_json, meta = materialize.read_adapter(written["wiki_e5"])
    served_scaling = float(cfg_json["lora_alpha"]) / float(cfg_json["r"])

    check("alpha: adapter_config carries r=K*r and alpha=K*handle.alpha",
          int(cfg_json["r"]) == K * RANK
          and abs(float(cfg_json["lora_alpha"]) - handle.alpha * K) < 1e-9)
    check("alpha: the served scaling equals the live scaling",
          abs(served_scaling - handle.modules[0].scaling) < 1e-12,
          "%.4f vs %.4f" % (served_scaling, handle.modules[0].scaling))
    check("alpha: the rescale is recorded in the provenance",
          meta["emitted_rank"] == K * RANK and meta["handle_rank"] == handle.rank
          and abs(meta["alpha_rescaled_from"] - handle.alpha) < 1e-9)

    worst, worst_wrong = 0.0, 1e30
    for s, dw in zip(handle.sites, live_dw):
        A, B = read[s.rel_name]
        served = served_scaling * (B @ A)
        worst = max(worst, float((served - dw).abs().max()))
        # what the bug looked like: use the handle's alpha with the emitted rank
        wrong = (handle.alpha / float(K * RANK)) * (B @ A)
        worst_wrong = min(worst_wrong, float((wrong - dw).abs().max()))
    check("alpha: the materialised adapter reproduces the live dW",
          worst < 1e-6, "max|d|=%.2e" % worst)
    check("alpha: ...and dropping the rescale would NOT (so this test bites)",
          worst_wrong > 1e-4, "min|d|=%.2e" % worst_wrong)

    # end to end: merging the written adapter must reproduce the live forward
    fresh, _ = toy.tiny_model(seed=0)
    fresh.eval()
    n = materialize.merge_into_model(fresh, written["wiki_e5"])
    check("alpha: merge-then-serve reproduces the live delta mixture",
          n == len(handle)
          and torch.allclose(live, _logits(fresh, ids), atol=1e-5, rtol=1e-4),
          "max|d|=%.2e" % float((live - _logits(fresh, ids)).abs().max()))
    handle.remove()


def _unzip(factors_by_rel, handle):
    A = [factors_by_rel[s.rel_name][0] for s in handle.sites]
    B = [factors_by_rel[s.rel_name][1] for s in handle.sites]
    return A, B


# --------------------------------------------------------------------------
# 6. zero-init is an exact no-op
# --------------------------------------------------------------------------

def test_zero_init_is_a_noop():
    for kind in ("free", "basis"):
        model, cfg, handle = _toy()
        ids, _labels = toy.tiny_batch(cfg, batch=2, seq=5)
        base = _logits(model, ids)
        gen = _build(kind, handle)
        check("%s: declares starts_as_noop" % kind, gen.starts_as_noop is True)

        factors = hypernet.generate_for_cells(gen, {"c": _cond()}, handle)
        handle.load_state(factors["c"])
        check("%s: every generated B is EXACTLY zero at init" % kind,
              all(float(B.abs().sum()) == 0.0 for _A, B in factors["c"].values()))
        check("%s: adapter_is_noop agrees" % kind, inject.adapter_is_noop(handle))
        check("%s: base and adapted logits are BITWISE equal at init" % kind,
              torch.equal(base, _logits(model, ids)))
        handle.remove()

    # the mixture is honest about not being one: it starts at the mean adapter
    model, cfg, handle = _toy()
    ids, _labels = toy.tiny_batch(cfg, batch=2, seq=5)
    base = _logits(model, ids)
    gen = _build("mixture", handle)
    factors = hypernet.generate_for_cells(gen, {"c": _cond()}, handle)
    handle.set_factors(*_unzip(factors["c"], handle))
    check("mixture: starts_as_noop is False and that is the truth",
          gen.starts_as_noop is False and not torch.equal(base, _logits(model, ids)))
    coeffs = gen.coefficients(_cond())
    check("mixture: uniform coefficients at init (starts at the mean adapter)",
          float((coeffs - 1.0 / gen.n_basis).abs().max()) < 1e-6,
          "%r" % [round(float(c), 3) for c in coeffs])
    handle.remove()


# --------------------------------------------------------------------------

def main():
    torch.manual_seed(0)
    tmp = tempfile.mkdtemp(prefix="adaptercl_torch_contracts_")
    try:
        test_generate_serve_round_trip(tmp)
        test_merge_equals_injected(tmp)
        test_per_example_factors()
        test_freeze_and_gradient_flow()
        test_alpha_rescale_for_delta_mixture(tmp)
        test_zero_init_is_a_noop()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\n%d passed, %d failed" % (PASS[0], FAIL[0]))
    sys.exit(1 if FAIL[0] else 0)


if __name__ == "__main__":
    main()
