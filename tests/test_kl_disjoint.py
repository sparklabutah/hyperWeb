#!/usr/bin/env python3
"""CPU contract checks for the optional disjoint KL anchor integration.

Run after applying integrations/llamafactory-kl.patch to the TimeWarp
LLaMA-Factory fork. Uses small CPU tensors; no model checkpoint is loaded.
"""
from __future__ import print_function

import importlib.util
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
REPO = os.path.dirname(PROJECT)


def _lmf_file(*rel):
    root = os.environ.get("ADAPTERCL_LLAMA_FACTORY_ROOT",
                          os.path.join(os.environ.get("ADAPTERCL_TIMEWARP_ROOT", REPO),
                                       "LLaMA-Factory"))
    return os.path.join(root, "src", "llamafactory", *rel)


TW_PATH = _lmf_file("data", "processor", "tw_token_weights.py")
TRAINER_PATH = _lmf_file("train", "sft", "trainer.py")

PASS = [0]
FAIL = [0]


print("test_kl_disjoint: exercising %s" % TW_PATH)
print("test_kl_disjoint: exercising %s" % TRAINER_PATH)


def check(name, cond, extra=""):
    if cond:
        PASS[0] += 1
        print("  ok   %s%s" % (name, (" " + extra) if extra else ""))
    else:
        FAIL[0] += 1
        print("  FAIL %s%s" % (name, (" " + extra) if extra else ""))


def _load_tw():
    if not os.path.isfile(TW_PATH):
        raise RuntimeError("staged tw_token_weights.py not found at %s" % TW_PATH)
    spec = importlib.util.spec_from_file_location("tw_token_weights_staged", TW_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TW = _load_tw()

torch.manual_seed(0)


def _k3(ref_logp, logp):
    d = (ref_logp - logp).clamp(-20.0, 8.0)
    return torch.exp(d) - d - 1.0


def historical_ce_kl(logp, ref_logp, mask, gamma, ce_w):
    """Byte-for-byte copy of the OLD (pre-K1) `_tw_kl_compute_loss` arithmetic.

    Copied rather than imported: it is the formula the patch REPLACES, so it must not live in the
    staged file (which now only carries the fixed version) -- test (d) exists precisely to pin the
    bug this reproduces.
    """
    n = mask.sum().clamp(min=1)
    if ce_w is None:
        ce = -(logp * mask.float()).sum() / n
    else:
        w = ce_w.float() * mask.float()
        ce = -(logp * w).sum() / w.sum().clamp(min=1e-6)

    kl = _k3(ref_logp, logp)
    g = mask.float() if gamma is None else (gamma.float() * mask.float())
    kl_term = (kl * g).sum() / g.sum().clamp(min=1e-6)
    return ce, kl_term


def _random_batch(B=4, T=6, own_rows=None, kl_rows=None, requires_grad=True):
    """A random (B, T) logp/ref_logp/mask/gamma batch. own_rows/kl_rows are explicit row index
    sets; every row not named is a CE (own) row."""
    logp = (-torch.rand(B, T, dtype=torch.float64) * 3.0).requires_grad_(requires_grad)
    ref_logp = -torch.rand(B, T, dtype=torch.float64) * 3.0
    mask = torch.ones(B, T, dtype=torch.bool)
    gamma = torch.zeros(B, T, dtype=torch.float64)
    kl_rows = kl_rows or set()
    for r in kl_rows:
        gamma[r, :] = 1.0
    return logp, ref_logp, mask, gamma


def test_all_anchor_batch():
    B, T = 3, 5
    logp, ref_logp, mask, gamma = _random_batch(B, T, kl_rows=set(range(B)))
    loss, ce, kl_term, n_ce, n_kl = TW.disjoint_kl_loss(logp, ref_logp, mask, gamma, None, beta=1.0)
    check("all-anchor: n_ce_rows == 0", int(n_ce.item()) == 0)
    check("all-anchor: n_kl_rows == B", int(n_kl.item()) == B)
    check("all-anchor: ce == 0 exactly", ce.item() == 0.0, "ce=%r" % ce.item())

    expect_kl = _k3(ref_logp, logp)
    expect_kl_term = expect_kl.sum() / gamma.sum()
    check("all-anchor: kl_term matches k3 mean", torch.allclose(kl_term, expect_kl_term))
    check("all-anchor: loss == beta * kl_term (ce contributes nothing)",
          torch.allclose(loss, 1.0 * kl_term))

    (grad,) = torch.autograd.grad(loss, logp, retain_graph=True)
    # d(loss)/d(logp) should be exactly the KL gradient (no CE term reaches this batch at all):
    # recompute the KL-only quantity from scratch on a fresh leaf and compare gradients.
    logp2 = logp.detach().clone().requires_grad_(True)
    kl2 = _k3(ref_logp, logp2)
    kl_term2 = kl2.sum() / gamma.sum()
    (expect_grad,) = torch.autograd.grad(kl_term2, logp2)
    check("all-anchor: gradient equals the KL-only gradient", torch.allclose(grad, expect_grad))


def test_all_own_batch():
    B, T = 3, 5
    logp, ref_logp, mask, gamma = _random_batch(B, T, kl_rows=set())
    loss, ce, kl_term, n_ce, n_kl = TW.disjoint_kl_loss(logp, ref_logp, mask, gamma, None, beta=1.0)
    check("all-own: n_kl_rows == 0", int(n_kl.item()) == 0)
    check("all-own: n_ce_rows == B", int(n_ce.item()) == B)
    check("all-own: kl_term == 0 exactly", kl_term.item() == 0.0, "kl=%r" % kl_term.item())

    expect_ce = -(logp * mask.float()).sum() / mask.float().sum()
    check("all-own: ce matches plain uniform mean CE", torch.allclose(ce, expect_ce))
    check("all-own: loss == ce (kl contributes nothing)", torch.allclose(loss, expect_ce))


def test_mixed_batch():
    B, T = 5, 4
    kl_rows = {1, 3}
    logp, ref_logp, mask, gamma = _random_batch(B, T, kl_rows=kl_rows)
    loss, ce, kl_term, n_ce, n_kl = TW.disjoint_kl_loss(logp, ref_logp, mask, gamma, None, beta=0.7)
    check("mixed: n_ce_rows + n_kl_rows == B", int(n_ce.item()) + int(n_kl.item()) == B)
    check("mixed: n_kl_rows == 2", int(n_kl.item()) == 2)

    ce_rows = [r for r in range(B) if r not in kl_rows]
    ce_mask = torch.zeros(B, T, dtype=torch.float64)
    ce_mask[ce_rows, :] = 1.0
    expect_ce = -(logp * ce_mask).sum() / ce_mask.sum()
    check("mixed: ce == mean CE over CE rows only", torch.allclose(ce, expect_ce))

    kl_mask = torch.zeros(B, T, dtype=torch.float64)
    kl_mask[sorted(kl_rows), :] = 1.0
    expect_kl = _k3(ref_logp, logp)
    expect_kl_term = (expect_kl * kl_mask).sum() / kl_mask.sum()
    check("mixed: kl_term == k3 mean over KL rows only", torch.allclose(kl_term, expect_kl_term))
    check("mixed: loss == ce + beta*kl_term", torch.allclose(loss, expect_ce + 0.7 * expect_kl_term))


def test_env_off_reproduces_old_formula_on_a_ce_only_batch():
    """(a) With no anchor rows at all, disjoint_kl_loss and the historical formula must agree --
    this is the "the flag being off changes nothing" case for the arithmetic itself."""
    B, T = 4, 6
    logp, ref_logp, mask, gamma = _random_batch(B, T, kl_rows=set())
    loss, ce, kl_term, _n_ce, _n_kl = TW.disjoint_kl_loss(logp, ref_logp, mask, gamma, None, beta=1.0)
    hist_ce, hist_kl = historical_ce_kl(logp, ref_logp, mask, None, None)
    check("env-off proxy: ce matches historical formula on a no-anchor batch",
          torch.allclose(ce, hist_ce))
    # kl_term differs by construction (historical uses gamma=None -> every token; disjoint scopes
    # to KL rows, of which there are none) -- that divergence is exactly what "disjoint" means, so
    # only ce (the part that must not silently change) is compared here.
    check("env-off proxy: disjoint kl_term is 0 with no KL rows (vs historical's whole-mask kl)",
          kl_term.item() == 0.0)


def test_regression_old_formula_trains_anchor_at_full_strength():
    """(d) THE REGRESSION THE PLAN NAMES. One row of a size-1 micro-batch (per_device_train_batch_size
    = 1, the K1 recipe) is an "anchor" row: its CE weights are all epsilon. Under the OLD formula
    (ce = -(logp*w).sum()/w.sum()) the epsilon cancels and the row trains at FULL CE strength --
    identical to giving it weight 1.0 everywhere. This is exactly why K1 cannot express "CE 0 on
    anchor rows" through TW_TOKEN_WEIGHTS epsilon weights and needs disjoint_kl_loss instead.
    """
    T = 8
    logp = -torch.rand(1, T, dtype=torch.float64) * 3.0
    ref_logp = -torch.rand(1, T, dtype=torch.float64) * 3.0
    mask = torch.ones(1, T, dtype=torch.bool)

    epsilon = 1e-4
    eps_weights = torch.full((1, T), epsilon, dtype=torch.float64)
    hist_ce_eps, _ = historical_ce_kl(logp, ref_logp, mask, None, eps_weights)

    uniform_weights = torch.ones(1, T, dtype=torch.float64)
    hist_ce_uniform, _ = historical_ce_kl(logp, ref_logp, mask, None, uniform_weights)

    check("regression: OLD formula + epsilon CE weights == full-strength (uniform) CE",
          torch.allclose(hist_ce_eps, hist_ce_uniform),
          "eps_ce=%.6f uniform_ce=%.6f" % (hist_ce_eps.item(), hist_ce_uniform.item()))

    # And disjoint_kl_loss actually fixes it: the same row, correctly marked as a KL row (gamma=1
    # everywhere), contributes NO ce at all.
    gamma_all = torch.ones(1, T, dtype=torch.float64)
    _loss, ce_fixed, _kl, n_ce, n_kl = TW.disjoint_kl_loss(logp, ref_logp, mask, gamma_all, None, beta=1.0)
    check("regression: disjoint_kl_loss gives the same row ce == 0 (fixed)", ce_fixed.item() == 0.0)
    check("regression: disjoint_kl_loss classifies it as a KL row", int(n_kl.item()) == 1 and int(n_ce.item()) == 0)


def test_gamma_floor_not_lifted():
    """(f) target_kl_gamma on an "own" row (entry default_weight=0.0) must return exact 0.0 for
    every content token, not DEFAULT_FLOOR (0.1) -- scout_task_plan.md 10.2/10.3's floor-lift bug.

    Goes through the REAL env-var-driven gamma table (TW_KL_GAMMA_WEIGHTS -> a side-car on disk),
    the same path the trainer uses, not an injected table -- `_gamma_table()` has no `set_table`
    equivalent, by design (module docstring: it must not thrash the CE weighting table's cache).
    """
    import json
    import tempfile

    class _FakeTokenizer(object):
        is_fast = True

        def __call__(self, content, add_special_tokens=False, return_offsets_mapping=True):
            # One "token" per character, offsets [i, i+1) -- simplest possible fast-tokenizer stub.
            ids = [ord(c) % 1000 for c in content]
            offsets = [(i, i + 1) for i in range(len(content))]
            return {"input_ids": ids, "offset_mapping": offsets}

    tok = _FakeTokenizer()
    content = "own site content"
    target_ids = [ord(c) % 1000 for c in content]
    target_label = list(target_ids)  # nothing masked
    key = TW.content_key(content)

    old_env = os.environ.get(TW.ENV_KL_GAMMA)
    old_disjoint = os.environ.get(TW.ENV_KL_DISJOINT)
    tmpdir = tempfile.mkdtemp(prefix="tw_kl_gamma_test_")
    try:
        own_path = os.path.join(tmpdir, "own.klgamma.json")
        own_entry = {"len": len(content), "default_weight": 0.0, "envelope_weight": 0.0}
        with open(own_path, "w") as fh:
            json.dump({"version": 2, "entries": {key: own_entry}}, fh)
        os.environ[TW.ENV_KL_GAMMA] = own_path

        # SWITCH OFF: the gamma path must be argument-for-argument what it was before ablation K.
        # A floor-less side-car entry of 0.0 therefore still floors to DEFAULT_FLOOR (0.1), exactly
        # as the pre-patch code did. The relaxations below are gated on TW_KL_DISJOINT, so a re-run
        # of an existing (non-disjoint) KL recipe cannot change under this patch.
        os.environ.pop(TW.ENV_KL_DISJOINT, None)
        weights_off = TW.target_kl_gamma(tok, content, target_ids, target_label)
        check("gamma floor: switch OFF -> floor-less 0.0 entry still floors to 0.1 (pre-patch behaviour)",
              all(abs(w - TW.DEFAULT_FLOOR) < 1e-9 for w in weights_off), "weights=%r" % weights_off)

        os.environ[TW.ENV_KL_DISJOINT] = "1"
        weights = TW.target_kl_gamma(tok, content, target_ids, target_label)
        check("gamma floor: own row -> gamma is EXACTLY 0.0 everywhere (not the 0.1 floor)",
              all(w == 0.0 for w in weights), "weights=%r" % weights)

        anchor_path = os.path.join(tmpdir, "anchor.klgamma.json")
        anchor_entry = {"len": len(content), "default_weight": 1.0, "envelope_weight": 1.0}
        with open(anchor_path, "w") as fh:
            json.dump({"version": 2, "entries": {key: anchor_entry}}, fh)
        os.environ[TW.ENV_KL_GAMMA] = anchor_path
        weights = TW.target_kl_gamma(tok, content, target_ids, target_label)
        check("gamma floor: anchor row -> gamma is 1.0 everywhere", all(w == 1.0 for w in weights))
    finally:
        if old_env is None:
            os.environ.pop(TW.ENV_KL_GAMMA, None)
        else:
            os.environ[TW.ENV_KL_GAMMA] = old_env
        if old_disjoint is None:
            os.environ.pop(TW.ENV_KL_DISJOINT, None)
        else:
            os.environ[TW.ENV_KL_DISJOINT] = old_disjoint
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    # And the ordinary CE-weighting table is UNCHANGED: an entry weight of 0.0 there still floors
    # to DEFAULT_FLOOR (0.1) exactly as before this patch -- byte-identical default behaviour.
    ce_entry = {"len": len(content), "default_weight": 0.0}
    weights_ce, _labels = TW.target_token_weights(
        tok, content, target_ids, target_label, tables=({key: ce_entry}, {})
    )
    check("gamma floor: the CE weighting table's own floor (0.1) is untouched",
          all(abs(w - TW.DEFAULT_FLOOR) < 1e-9 for w in weights_ce), "weights=%r" % weights_ce)


def test_assert_disjoint_requires_gamma():
    try:
        TW.assert_disjoint_requires_gamma(True, False, beta=1.0)
        ok = False
    except ValueError:
        ok = True
    check("assert_disjoint_requires_gamma: refuses disjoint=1 without gamma", ok)

    try:
        TW.assert_disjoint_requires_gamma(True, True, beta=1.0)
        ok = True
    except ValueError:
        ok = False
    check("assert_disjoint_requires_gamma: disjoint=1 WITH gamma AND beta>0 is fine", ok)

    try:
        TW.assert_disjoint_requires_gamma(False, False, beta=1.0)
        ok = True
    except ValueError:
        ok = False
    check("assert_disjoint_requires_gamma: disjoint=0 is always a no-op", ok)

    # Regression: a disjoint config with a real gamma side-car but TW_KL_BETA unset/<=0 never
    # reaches compute_loss's disjoint branch at all (compute_loss only calls _tw_kl_compute_loss
    # when beta > 0) and would silently train every row -- anchor rows included -- at plain
    # uniform CE. Must be refused at construction, the same as the missing-gamma case.
    for bad_beta in (0.0, -1.0):
        try:
            TW.assert_disjoint_requires_gamma(True, True, beta=bad_beta)
            ok = False
        except ValueError:
            ok = True
        check("assert_disjoint_requires_gamma: refuses disjoint=1 + gamma with "
             "beta=%r (silent full-CE-on-anchor-rows regression via a different path)"
             % bad_beta, ok)

    try:
        TW.assert_disjoint_requires_gamma(True, True)   # beta omitted -> defaults to 0.0
        ok = False
    except ValueError:
        ok = True
    check("assert_disjoint_requires_gamma: beta defaults to 0.0, so omitting it under "
         "disjoint=1 is ALSO refused, not silently accepted", ok)

    try:
        TW.assert_disjoint_requires_gamma(False, False)   # disjoint off: beta never checked
        ok = True
    except ValueError:
        ok = False
    check("assert_disjoint_requires_gamma: disjoint=0 ignores beta entirely (still a no-op)", ok)


def test_tail_logits_window_is_exact():
    """TW_KL_TAIL_LOGITS: cutting the forward to the supervised tail must leave EVERY supervised
    token's log-probability bit-identical, and with it ce, kl_term and the loss. The "model" here is
    a fixed (B, T, V) logits tensor; `logits_to_keep=k` is emulated exactly as HF does it
    (`hidden[:, -k:, :]`), so this pins the index arithmetic: start, the label/gamma cut, the shift."""
    torch.manual_seed(7)
    B, T, V = 2, 23, 11
    full = torch.randn(B, T, V)
    ref_full = torch.randn(B, T, V)
    labels = torch.full((B, T), -100, dtype=torch.long)
    labels[0, 15:] = torch.randint(0, V, (8,))      # first supervised token at 15
    labels[1, 12:] = torch.randint(0, V, (11,))     # ...and at 12: the batch window must start at 11
    gamma = torch.zeros(B, T); gamma[1, 12:] = 1.0  # row 1 is an anchor row, row 0 an own row

    def logp(lg, tg, mk):
        safe = tg.masked_fill(~mk, 0)
        return -torch.nn.functional.cross_entropy(lg.reshape(-1, V).float(), safe.reshape(-1), reduction="none").view_as(safe)

    start = TW.supervised_tail_start(labels)
    check("tail: start is (earliest first-supervised) - 1", start == 11, "start=%r" % start)
    keep = T - start
    # full path
    sl, m = labels[..., 1:], labels[..., 1:].ne(-100)
    lp_full, rp_full = logp(full[..., :-1, :], sl, m), logp(ref_full[..., :-1, :], sl, m)
    out_full = TW.disjoint_kl_loss(lp_full, rp_full, m, gamma[..., 1:], None, 0.7)
    # tail path
    lab_t, gam_t = labels[..., start:], gamma[..., start:]
    slt, mt = lab_t[..., 1:], lab_t[..., 1:].ne(-100)
    lp_tail, rp_tail = logp(full[:, -keep:, :][..., :-1, :], slt, mt), logp(ref_full[:, -keep:, :][..., :-1, :], slt, mt)
    out_tail = TW.disjoint_kl_loss(lp_tail, rp_tail, mt, gam_t[..., 1:], None, 0.7)
    check("tail: same number of supervised tokens", int(m.sum()) == int(mt.sum()) == 19)
    check("tail: per-token log-probs bit-identical on every supervised token",
          torch.equal(lp_full[m], lp_tail[mt]) and torch.equal(rp_full[m], rp_tail[mt]))
    # The per-token values above are BIT-identical; the reductions sum a shorter tensor, so float32
    # summation order differs in the last digit (3.0281522 vs 3.0281525). That is rounding, not a
    # different loss: require agreement to 1e-6 relative, and exact row counts.
    check("tail: loss, ce, kl_term agree to 1e-6 relative; row counts exact",
          all(torch.allclose(a.float(), b.float(), rtol=1e-6, atol=0.0) for a, b in zip(out_full[:3], out_tail[:3]))
          and all(torch.equal(a, b) for a, b in zip(out_full[3:], out_tail[3:])),
          "full=%r tail=%r" % ([float(x) for x in out_full], [float(x) for x in out_tail]))
    nolab = torch.full((2, 5), -100, dtype=torch.long); nolab[0, 3] = 1
    check("tail: a row with no supervised token -> keep everything (start 0)", TW.supervised_tail_start(nolab) == 0)
    first0 = torch.zeros(1, 4, dtype=torch.long)
    check("tail: supervised from position 0 -> start 0", TW.supervised_tail_start(first0) == 0)
    old = os.environ.pop(TW.ENV_KL_TAIL_LOGITS, None)
    check("tail: switch is OFF by default", TW.kl_tail_logits_enabled() is False)
    os.environ[TW.ENV_KL_TAIL_LOGITS] = "1"; check("tail: TW_KL_TAIL_LOGITS=1 turns it on", TW.kl_tail_logits_enabled() is True)
    if old is None: os.environ.pop(TW.ENV_KL_TAIL_LOGITS, None)
    else: os.environ[TW.ENV_KL_TAIL_LOGITS] = old


def test_trainer_passes_beta_to_the_gate():
    """Static source check: the STAGED trainer.py's construction-time call must actually pass
    ``self._tw_kl_beta`` as the gate's third argument -- fixing the pure function alone does
    nothing if the one real caller keeps calling it with the old 2-argument form (which now
    silently defaults beta to 0.0 and would refuse EVERY disjoint run, gamma or not)."""
    src = open(TRAINER_PATH).read()
    i_call = src.index("assert_disjoint_requires_gamma(")
    call_text = src[i_call:i_call + 120]
    check("trainer.py's assert_disjoint_requires_gamma call passes self._tw_kl_beta",
         "self._tw_kl_beta" in call_text, " ".join(call_text.split()))


def test_kl_disjoint_env_helper():
    old = os.environ.pop("TW_KL_DISJOINT", None)
    try:
        check("kl_disjoint_enabled: unset -> False", TW.kl_disjoint_enabled() is False)
        os.environ["TW_KL_DISJOINT"] = "1"
        check("kl_disjoint_enabled: '1' -> True", TW.kl_disjoint_enabled() is True)
        os.environ["TW_KL_DISJOINT"] = "0"
        check("kl_disjoint_enabled: '0' -> False", TW.kl_disjoint_enabled() is False)
    finally:
        if old is None:
            os.environ.pop("TW_KL_DISJOINT", None)
        else:
            os.environ["TW_KL_DISJOINT"] = old


def main():
    test_all_anchor_batch()
    test_all_own_batch()
    test_mixed_batch()
    test_env_off_reproduces_old_formula_on_a_ce_only_batch()
    test_regression_old_formula_trains_anchor_at_full_strength()
    test_gamma_floor_not_lifted()
    test_assert_disjoint_requires_gamma()
    test_tail_logits_window_is_exact()
    test_trainer_passes_beta_to_the_gate()
    test_kl_disjoint_env_helper()
    print("\n%d passed, %d failed" % (PASS[0], FAIL[0]))
    sys.exit(1 if FAIL[0] else 0)


if __name__ == "__main__":
    main()
