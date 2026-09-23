"""6.7's diagnostics -- the checks that gate every downstream number.

6.7 is titled "Diagnostics -- run before trusting any result", and each of its
three checks has a failure mode that is *silent*: the numbers still come out,
they are just meaningless.

  identity probe     A linear probe from the conditioning embedding to cell
                     identity. Near-perfect accuracy means a lookup table is
                     learnable, i.e. the appendix's "encoder routes to one of N
                     adapters" critique in a new costume. 6.7 prescribes
                     pixel-space augmentation and a re-probe -- `augmentation_
                     report`.
  diversity collapse Pairwise distance + effective rank over *generated*
                     adapters (valid: shared parametrisation, 6.2's last
                     paragraph). "Hypernetworks at this input:output ratio
                     routinely converge to mean adapter plus noise."
  functional sanity  Generated adapter vs. a random adapter of matched norm.
                     "Indistinguishable success means the adapter does nothing
                     and everything downstream is noise."

`run_all` bundles them into one JSON with `trustworthy` and `blocking`, so a
Phase-3 driver can refuse to report a result the diagnostics do not support.

Two things here go beyond 6.7's sketch and both matter:

* **The identity probe is decomposed.** Probing era, environment and joint cell
  identity separately is more informative than one accuracy: 3 predicts that
  appearance (era) is recoverable while site function (environment) is not, so
  "era-identifiable, environment-confusable" *supports* the claim while
  "both near-perfect" is the lookup table. One scalar cannot tell those apart.
* **Grouped splits.** A stratified random split over embeddings leaks when
  several embeddings come from the same page: the probe then memorises pages,
  not cells, and reads as near-perfect for the wrong reason. Pass `groups=` and
  the split holds out whole groups; the report records which split ran.

Statistics helpers (sign-flip test, bootstrap, effective rank, strict-JSON
coercion) live in transfer.py so both analysis modules use one implementation.
sklearn is used when importable and a documented numpy implementation otherwise;
the two are cross-checked in `--selftest`.

Pure numpy + stdlib; runs on the system python (3.6.8, numpy 1.19.5).
"""

from __future__ import print_function

import collections
import json
import math
import os

import numpy as np

from . import cells as cells_mod
from . import paths
from .transfer import (bootstrap_mean_ci, json_safe, sign_flip_test,
                       spectrum_effective_rank)

# --------------------------------------------------------------------------
# Thresholds. Every one of these is a judgement call, so every one is named,
# defaulted here, overridable, and echoed into the report.
# --------------------------------------------------------------------------

#: Probe test accuracy at or above this is "near-perfect": a lookup table from
#: conditioning embedding to cell identity is learnable. Not 1.0, because with
#: ~18 classes and a few hundred embeddings a couple of genuinely ambiguous
#: screenshots should not rescue the result.
NEAR_PERFECT = 0.95

#: How far above the majority-class baseline counts as "identifiable at all".
IDENTIFIABLE_MARGIN = 0.20

#: diversity_collapse: RMS deviation from the mean adapter, as a fraction of the
#: mean adapter's norm. Below this the generator is emitting one adapter with a
#: perturbation on top -- 6.7's "mean adapter plus noise", literally.
COLLAPSE_VARIANCE_RATIO = 0.10

#: diversity_collapse: centred effective rank as a fraction of its maximum
#: (min(n-1, d)). With 18 conditioning inputs the ceiling is 17, so 0.25 means
#: fewer than ~4.3 independent directions -- below what the 3-environment x
#: 6-era grid needs (6 era directions alone) before you can claim the generator
#: distinguishes cells rather than groups them.
COLLAPSE_RANK_FRACTION = 0.25

#: random_adapter_control: |Cohen's d_z| below this is "indistinguishable" even
#: if the permutation test happens to clear alpha.
MIN_EFFECT_DZ = 0.20

ALPHA = 0.05

#: Deterministic optimiser settings for the numpy logistic regression. Full
#: batch, zero init, adaptive step size (grow 1.1x on improvement, halve on
#: overshoot) -- no learning-rate tuning per dataset, no RNG in the optimiser,
#: so two runs on the same array give bit-identical coefficients.
PROBE_L2 = 1e-3
PROBE_STEPS = 600
PROBE_LR0 = 1.0
PROBE_TOL = 1e-9


# --------------------------------------------------------------------------
# Multinomial logistic regression (numpy; sklearn if available)
# --------------------------------------------------------------------------

def _softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def fit_multinomial(X, y, n_classes, l2=PROBE_L2, n_steps=PROBE_STEPS,
                    lr0=PROBE_LR0, tol=PROBE_TOL):
    """Softmax regression by full-batch gradient descent with an adaptive step.

    Deterministic: weights start at exactly zero, the data order is fixed, and
    the step size is chosen by the loss itself (never by an RNG). Returns
    (W, b, info); `info["converged"]` says whether the improvement fell below
    `tol` rather than the step budget running out.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    n, d = X.shape
    W = np.zeros((d, n_classes))
    b = np.zeros(n_classes)
    Y = np.zeros((n, n_classes))
    Y[np.arange(n), y] = 1.0

    def _lg(W, b):
        P = _softmax(X.dot(W) + b)
        nll = -float(np.log(np.clip(P[np.arange(n), y], 1e-12, None)).mean())
        loss = nll + 0.5 * l2 * float((W * W).sum())
        G = (P - Y) / float(n)
        return loss, X.T.dot(G) + l2 * W, G.sum(axis=0), nll

    loss, gW, gb, nll = _lg(W, b)
    lr = float(lr0)
    converged, used = False, 0
    for step in range(n_steps):
        used = step + 1
        nW, nb = W - lr * gW, b - lr * gb
        nloss, ngW, ngb, nnll = _lg(nW, nb)
        if nloss <= loss:
            improved = loss - nloss
            W, b, gW, gb, loss, nll = nW, nb, ngW, ngb, nloss, nnll
            lr *= 1.1
            if improved < tol:
                converged = True
                break
        else:
            lr *= 0.5
            if lr < 1e-14:
                converged = True
                break
    return W, b, {"loss": loss, "nll": nll, "steps": used,
                  "converged": converged, "final_lr": lr, "l2": l2,
                  "backend": "numpy"}


def _predict(X, W, b):
    return np.argmax(np.asarray(X, dtype=np.float64).dot(W) + b, axis=1)


def _fit_predict(Xtr, ytr, Xte, n_classes, l2, backend):
    """One (fit, predict-train, predict-test), via sklearn or numpy."""
    if backend in ("auto", "sklearn"):
        try:
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(C=1.0 / max(l2, 1e-12), solver="lbfgs",
                                     multi_class="multinomial", max_iter=2000)
            clf.fit(Xtr, ytr)
            return (clf.predict(Xtr), clf.predict(Xte),
                    {"backend": "sklearn", "l2": l2,
                     "n_iter": int(np.max(clf.n_iter_))})
        except ImportError:
            if backend == "sklearn":
                raise ImportError(
                    "backend='sklearn' requested but sklearn is not importable "
                    "on this interpreter; use backend='numpy' or 'auto'")
    W, b, info = fit_multinomial(Xtr, ytr, n_classes, l2=l2)
    return _predict(Xtr, W, b), _predict(Xte, W, b), info


# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------

def _split_indices(y, test_frac, seed, groups=None):
    """Deterministic train/test split.

    With `groups` (e.g. one id per screenshot/page/episode) whole groups go to
    one side, so a probe cannot win by memorising a page that appears on both
    sides. Without them the split is stratified by class -- every class keeps at
    least one test example, so the reported accuracy is comparable across
    classes.
    """
    y = np.asarray(y, dtype=int)
    rs = np.random.RandomState(seed)
    if groups is not None:
        g = np.asarray(groups)
        uniq = np.array(sorted(set(g.tolist())))
        if uniq.size < 4:
            raise ValueError("grouped split needs >=4 groups, got %d" % uniq.size)
        perm = rs.permutation(uniq.size)
        n_test = max(1, int(round(test_frac * uniq.size)))
        test_groups = set(uniq[perm[:n_test]].tolist())
        mask = np.array([gg in test_groups for gg in g.tolist()])
        tr, te = np.where(~mask)[0], np.where(mask)[0]
        kind = "grouped (held-out groups, no page shared across the split)"
        if len(set(y[tr].tolist())) < len(set(y.tolist())):
            kind += " -- WARNING: some classes are absent from train"
        return tr, te, kind
    tr, te = [], []
    for c in sorted(set(y.tolist())):
        idx = np.where(y == c)[0]
        if idx.size < 2:
            raise ValueError(
                "class %d has %d example(s); a probe needs >=2 per class. "
                "Collect more conditioning embeddings per cell, or probe a "
                "coarser target (era/environment) instead of joint identity."
                % (c, idx.size))
        idx = idx[rs.permutation(idx.size)]
        n_test = max(1, int(round(test_frac * idx.size)))
        n_test = min(n_test, idx.size - 1)
        te.extend(idx[:n_test].tolist())
        tr.extend(idx[n_test:].tolist())
    return (np.array(sorted(tr)), np.array(sorted(te)),
            "stratified random (WARNING: leaks if several embeddings come from "
            "one page -- pass groups= to hold out whole pages)")


def _labels_for(cell_like, target):
    """cells/keys -> integer labels for one of era / environment / cell."""
    cs = []
    for c in cell_like:
        if isinstance(c, cells_mod.Cell):
            cs.append(c)
        elif isinstance(c, str):
            cs.append(cells_mod.Cell.parse(c))
        elif isinstance(c, (tuple, list)) and len(c) == 2:
            cs.append(cells_mod.Cell(c[0], int(c[1])))
        else:
            raise TypeError("cannot interpret %r as a Cell" % (c,))
    if target == "era":
        names = [str(c.era) for c in cs]
    elif target in ("env", "environment"):
        names = [c.env for c in cs]
    elif target == "cell":
        names = [c.key for c in cs]
    else:
        raise ValueError("target must be 'era', 'env' or 'cell', got %r" % target)
    uniq = sorted(set(names))
    lut = dict((k, i) for i, k in enumerate(uniq))
    return np.array([lut[n] for n in names], dtype=int), uniq


# --------------------------------------------------------------------------
# 6.7: version-identity memorisation probe
# --------------------------------------------------------------------------

def identity_probe(embeddings, labels, target="cell", test_frac=0.3, seed=0,
                   l2=PROBE_L2, backend="auto", groups=None,
                   near_perfect=NEAR_PERFECT, standardize=True):
    """Linear probe from conditioning embedding -> identity (6.7).

    `labels` is one Cell / cell key per embedding row; `target` selects what is
    predicted (`'cell'`, `'era'`, `'env'`). Chance is reported two ways --
    1/n_classes and the majority-class rate in the test split -- and the
    verdict compares against the majority baseline, which is the honest one when
    classes are unbalanced.
    """
    X = np.asarray(embeddings, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("embeddings must be (n_samples, d), got %r" % (X.shape,))
    y, class_names = _labels_for(labels, target)
    if X.shape[0] != y.shape[0]:
        raise ValueError("%d embeddings but %d labels" % (X.shape[0], y.shape[0]))
    n_classes = len(class_names)
    if n_classes < 2:
        raise ValueError("target %r has a single class -- nothing to probe"
                         % target)

    tr, te, split_kind = _split_indices(y, test_frac, seed, groups=groups)
    Xtr, Xte = X[tr], X[te]
    if standardize:
        mu = Xtr.mean(axis=0, keepdims=True)
        sd = Xtr.std(axis=0, keepdims=True)
        sd[sd < 1e-9] = 1.0
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd

    ptr, pte, info = _fit_predict(Xtr, y[tr], Xte, n_classes, l2, backend)
    acc_tr = float((ptr == y[tr]).mean())
    acc_te = float((pte == y[te]).mean())
    counts = collections.Counter(y[te].tolist())
    majority = max(counts.values()) / float(len(te)) if len(te) else float("nan")
    uniform = 1.0 / n_classes

    identifiable = acc_te - majority >= IDENTIFIABLE_MARGIN
    near = acc_te >= near_perfect
    if near:
        verdict = "near_perfect"
        reading = (
            "NEAR-PERFECT (%.3f): a lookup table from conditioning embedding to "
            "%s is learnable, so 'the hypernetwork is doing version-ID "
            "inference' is live -- 6.7's discrete-routing critique in a new "
            "costume. 6.7's prescription is pixel-space augmentation and a "
            "re-probe: see augmentation_report()." % (acc_te, target))
    elif identifiable:
        verdict = "identifiable"
        reading = ("%s is recoverable from the embedding (%.3f vs %.3f majority "
                   "baseline) but not memorised outright." % (target, acc_te, majority))
    else:
        verdict = "confusable"
        reading = ("%s is NOT recoverable above the majority baseline (%.3f vs "
                   "%.3f)." % (target, acc_te, majority))

    return {
        "target": target,
        "n_samples": int(X.shape[0]), "d": int(X.shape[1]),
        "n_classes": n_classes, "classes": class_names,
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "split": split_kind, "test_frac": test_frac, "seed": seed,
        "grouped": groups is not None,
        "standardized": bool(standardize),
        "train_accuracy": acc_tr, "test_accuracy": acc_te,
        "chance_uniform": uniform, "chance_majority": majority,
        "margin_over_majority": acc_te - majority,
        "overfit_gap": acc_tr - acc_te,
        "near_perfect_threshold": near_perfect,
        "identifiable": bool(identifiable), "near_perfect": bool(near),
        "verdict": verdict, "reading": reading,
        "fit": info,
    }


def identity_probes(embeddings, labels, targets=("era", "env", "cell"),
                    **kw):
    """Run the probe for era, environment and joint cell identity.

    The decomposition is the informative part, not the joint number:

      era identifiable + environment confusable -> exactly what 3 predicts.
        Appearance is in the embedding, site function is not, and the adapter
        that gets generated from it is a UI-affordance adapter.
      both near-perfect  -> the lookup-table reading; 6.7 blocking.
      environment identifiable + era confusable -> the embedding carries site
        identity and not style; 6.2 will likely come back "organizes_by_
        environment" and 3 is in trouble.
      neither            -> the encoder is not seeing the interface at all;
        check the capture protocol (4.2) before anything else.
    """
    out = collections.OrderedDict()
    for t in targets:
        out[t] = identity_probe(embeddings, labels, target=t, **kw)
    era = out.get("era")
    env = out.get("env")
    structure, note = "not_evaluated", ""
    if era is not None and env is not None:
        e_id, v_id = era["identifiable"], env["identifiable"]
        if e_id and not v_id:
            structure = "era_identifiable_env_confusable"
            note = ("the structure 3 predicts: appearance is recoverable, site "
                    "function is not")
        elif e_id and v_id:
            structure = "both_identifiable"
            note = ("the embedding carries both; check the joint 'cell' probe "
                    "for the lookup-table reading")
        elif v_id and not e_id:
            structure = "env_identifiable_era_confusable"
            note = ("the embedding encodes which SITE this is, not what era it "
                    "looks like -- contradicts 3 and predicts 6.2 will "
                    "organize by environment")
        else:
            structure = "neither_identifiable"
            note = ("nothing about the cell is linearly recoverable -- suspect "
                    "the capture protocol (4.2) or the encoder before reading "
                    "any downstream result")
    res = collections.OrderedDict()
    res["probes"] = out
    res["structure"] = structure
    res["structure_note"] = note
    res["blocking"] = bool(out.get("cell", {}).get("near_perfect"))
    return res


def augmentation_report(plain_embeddings, augmented_embeddings, labels,
                        targets=("era", "env", "cell"), **kw):
    """6.7's "mitigate with pixel-space augmentation and re-probe".

    Both embedding sets must be row-aligned with `labels` -- same conditioning
    inputs, the augmented one captured through the pixel-space augmentation
    pipeline. Reports the accuracy drop per target and whether the augmentation
    actually pushed the joint cell probe below near-perfect. A drop that is
    large but still lands at 0.97 has mitigated nothing.
    """
    a = np.asarray(plain_embeddings, dtype=np.float64)
    b = np.asarray(augmented_embeddings, dtype=np.float64)
    if a.shape[0] != b.shape[0]:
        raise ValueError("plain has %d rows, augmented has %d -- they must be "
                         "row-aligned with the same labels"
                         % (a.shape[0], b.shape[0]))
    plain = identity_probes(a, labels, targets=targets, **kw)
    aug = identity_probes(b, labels, targets=targets, **kw)
    drops = collections.OrderedDict()
    for t in targets:
        p, q = plain["probes"][t], aug["probes"][t]
        drops[t] = {"plain": p["test_accuracy"], "augmented": q["test_accuracy"],
                    "drop": p["test_accuracy"] - q["test_accuracy"],
                    "still_near_perfect": bool(q["near_perfect"]),
                    "chance_majority": q["chance_majority"]}
    cell = drops.get("cell") or list(drops.values())[-1]
    was = plain["probes"].get("cell", {}).get("near_perfect", False)
    mitigated = bool(was and not cell["still_near_perfect"])
    if not was:
        verdict = "not_needed"
        reading = ("the un-augmented probe was already below near-perfect, so "
                   "there is nothing for augmentation to mitigate")
    elif mitigated:
        verdict = "mitigated"
        reading = ("augmentation moved the joint cell probe from %.3f to %.3f, "
                   "below the %.2f near-perfect line: the conditioning signal is "
                   "no longer a lookup key" % (cell["plain"], cell["augmented"],
                                               NEAR_PERFECT))
    else:
        verdict = "not_mitigated"
        reading = ("augmentation left the joint cell probe at %.3f (from %.3f). "
                   "6.7's mitigation failed -- the embedding is still a version "
                   "id, and any 'generalisation' result is a routing result"
                   % (cell["augmented"], cell["plain"]))
    return {"drops": drops, "plain": plain, "augmented": aug,
            "verdict": verdict, "reading": reading,
            "mitigated": mitigated, "blocking": bool(was and not mitigated)}


# --------------------------------------------------------------------------
# 6.7: diversity collapse
# --------------------------------------------------------------------------

def effective_rank(X, center=True):
    """exp(entropy of the normalised singular-value spectrum).

    `center=True` (the default) subtracts the mean row first, so the number
    answers "how many independent directions does this set VARY along" rather
    than "how many directions does it occupy". That distinction is the whole
    diagnostic: a collapsed generator emitting `mean + eps` has an uncentred
    effective rank near 1 and a centred one near the noise dimension, and only
    the pair of them together tells you which you are looking at -- so
    `diversity_collapse` reports both.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("effective_rank wants a 2-D (n, d) array, got %r"
                         % (X.shape,))
    if center:
        X = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(X, full_matrices=False, compute_uv=False)
    return spectrum_effective_rank(s)


def pairwise_distance_stats(X, metric="euclidean"):
    """Pairwise distance summary over the rows of X (6.7)."""
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    if n < 2:
        raise ValueError("need >=2 rows, got %d" % n)
    norms = np.linalg.norm(X, axis=1)
    eu, cos = [], []
    for i in range(n):
        for j in range(i + 1, n):
            eu.append(float(np.linalg.norm(X[i] - X[j])))
            den = norms[i] * norms[j]
            cos.append(1.0 - float(X[i].dot(X[j]) / den) if den > 0 else float("nan"))
    d = np.array(eu if metric == "euclidean" else cos, dtype=float)
    d = d[np.isfinite(d)]
    return {"metric": metric, "n": int(n), "n_pairs": int(d.size),
            "mean": float(d.mean()), "sd": float(d.std(ddof=1)) if d.size > 1 else 0.0,
            "min": float(d.min()), "median": float(np.median(d)),
            "max": float(d.max()),
            "mean_euclidean": float(np.mean(eu)),
            "mean_cosine_distance": float(np.nanmean(cos)),
            "mean_row_norm": float(norms.mean()),
            "distance_to_norm_ratio": float(np.mean(eu) / norms.mean())
            if norms.mean() > 0 else float("nan")}


def diversity_collapse(generated, reference=None,
                       variance_ratio_threshold=COLLAPSE_VARIANCE_RATIO,
                       rank_fraction_threshold=COLLAPSE_RANK_FRACTION,
                       keys=None):
    """6.7's "mean adapter plus noise" test on hypernetwork-generated adapters.

    `generated` is (n_adapters, d) -- flattened generated adapters, which are a
    valid thing to compare because they share a parametrisation (6.2's last
    paragraph). Use transfer.generated_adapter_distances() to build it from
    adapter directories.

    Two independent collapse signals, both reported and either sufficient:

      variance_ratio = RMS(x_i - mean) / ||mean||
        Below `variance_ratio_threshold` (default 0.10) the spread across
        conditioning inputs is under a tenth of the shared component -- the
        generator is emitting one adapter with a perturbation, which is 6.7's
        phrasing taken literally. 0.10 rather than something smaller because a
        LoRA whose per-cell variation is a 10% effect on top of a common
        direction cannot express the 18 distinct behaviours 6.1 needs.

      rank_fraction = effective_rank(centred) / min(n-1, d)
        Below `rank_fraction_threshold` (default 0.25) the set uses under a
        quarter of the directions available to it. At 18 conditioning inputs the
        ceiling is 17, so 0.25 is ~4.3 effective directions -- fewer than the
        6 era directions the grid alone demands, let alone era x environment.

    `reference` (e.g. the independently trained per-cell adapters) is reported
    as a CEILING, not a target: independently trained adapters differ by random
    init and optimisation path as well as by function (4.4), so their spread is
    inflated and a generator that matched it would be suspicious, not good.
    """
    X = np.asarray(generated, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("generated must be (n_adapters, d), got %r" % (X.shape,))
    n, d = X.shape
    if n < 2:
        raise ValueError("need >=2 generated adapters, got %d" % n)
    mean = X.mean(axis=0)
    mean_norm = float(np.linalg.norm(mean))
    dev = X - mean[None, :]
    within_rms = float(np.sqrt((dev * dev).sum(axis=1).mean()))
    var_ratio = within_rms / mean_norm if mean_norm > 0 else float("inf")

    er_c = effective_rank(X, center=True)
    er_r = effective_rank(X, center=False)
    max_rank = float(min(n - 1, d))
    rank_frac = er_c / max_rank if max_rank > 0 else float("nan")

    pd = pairwise_distance_stats(X)
    by_ratio = var_ratio < variance_ratio_threshold
    by_rank = np.isfinite(rank_frac) and rank_frac < rank_fraction_threshold
    collapsed = bool(by_ratio or by_rank)

    out = {
        "n_adapters": int(n), "dim": int(d), "keys": list(keys) if keys else None,
        "mean_adapter_norm": mean_norm,
        "within_set_rms_deviation": within_rms,
        "variance_ratio": var_ratio,
        "variance_ratio_threshold": variance_ratio_threshold,
        "effective_rank_centered": er_c,
        "effective_rank_raw": er_r,
        "max_effective_rank": max_rank,
        "rank_fraction": rank_frac,
        "rank_fraction_threshold": rank_fraction_threshold,
        "pairwise": pd,
        "collapsed": collapsed,
        "collapsed_by": ([] + (["variance_ratio"] if by_ratio else [])
                         + (["rank_fraction"] if by_rank else [])),
        "reading": ("COLLAPSED: %s. 6.7's 'mean adapter plus noise' -- the "
                    "generator is not producing per-interface adapters and every "
                    "downstream generalisation number is measuring one adapter."
                    % (" and ".join(["variance_ratio %.3f < %.2f" % (var_ratio, variance_ratio_threshold)] if by_ratio else [])
                       + (" " if by_ratio and by_rank else "")
                       + (" ".join(["rank_fraction %.3f < %.2f" % (rank_frac, rank_fraction_threshold)]) if by_rank else ""))
                   if collapsed else
                   "not collapsed: variance ratio %.3f (>= %.2f) and %.1f of %.0f "
                   "effective directions used (%.0f%% >= %.0f%%)"
                   % (var_ratio, variance_ratio_threshold, er_c, max_rank,
                      100 * rank_frac, 100 * rank_fraction_threshold)),
        "blocking": collapsed,
    }
    if reference is not None:
        R = np.asarray(reference, dtype=np.float64)
        rmean = R.mean(axis=0)
        rdev = R - rmean[None, :]
        r_rms = float(np.sqrt((rdev * rdev).sum(axis=1).mean()))
        r_norm = float(np.linalg.norm(rmean))
        r_er = effective_rank(R, center=True)
        out["reference"] = {
            "n": int(R.shape[0]),
            "variance_ratio": r_rms / r_norm if r_norm > 0 else float("inf"),
            "effective_rank_centered": r_er,
            "relative_variance_ratio": (var_ratio / (r_rms / r_norm))
            if r_norm > 0 and r_rms > 0 else float("nan"),
            "relative_effective_rank": er_c / r_er if r_er > 0 else float("nan"),
            "caveat": ("independently trained reference adapters differ by "
                       "random init and optimisation path as well as by "
                       "function (4.4), so this is an inflated CEILING, not a "
                       "target -- matching it would be suspicious."),
        }
    return out


# --------------------------------------------------------------------------
# 6.7: functional sanity vs a random adapter of matched norm
# --------------------------------------------------------------------------

def random_adapter_control(generated_scores, random_scores, alpha=ALPHA,
                           n_perm=10000, seed=0, min_dz=MIN_EFFECT_DZ,
                           paired=True):
    """6.7: "generated adapter vs random adapter of matched norm".

    Inputs are per-unit outcomes -- one entry per (task, seed) evaluated under
    both adapters, in the same order. Paired, because the same task set is run
    twice and the between-task variance is enormous relative to the effect; an
    unpaired test on ~20 tasks would need a much larger gap to reach the same
    p-value. For binary outcomes the discordant-pair counts (McNemar's b and c)
    are reported as well, since they are what the sign-flip test is actually
    driven by.

    "Indistinguishable success means the adapter does nothing and everything
    downstream is noise" -- so `verdict == "adapter_does_nothing"` is blocking.
    """
    g = np.asarray(generated_scores, dtype=float)
    r = np.asarray(random_scores, dtype=float)
    if paired and g.shape != r.shape:
        raise ValueError(
            "paired control needs matched arrays, got %r vs %r. Evaluate both "
            "adapters on the SAME task/seed list and pass them in the same "
            "order (or set paired=False and accept the power loss)."
            % (g.shape, r.shape))
    if g.size < 2 or r.size < 2:
        raise ValueError("need >=2 evaluation units per arm")

    res = {"n_generated": int(g.size), "n_random": int(r.size),
           "mean_generated": float(np.nanmean(g)),
           "mean_random": float(np.nanmean(r)),
           "paired": bool(paired), "alpha": alpha, "min_dz": min_dz}
    res["difference"] = res["mean_generated"] - res["mean_random"]

    if paired:
        d = g - r
        ok = np.isfinite(d)
        d = d[ok]
        test = sign_flip_test(d, n_perm=n_perm, seed=seed)
        res["test"] = test
        res["p_value"] = test["p_value"]
        res["effect_size_dz"] = test["effect_size_dz"]
        res["ci"] = bootstrap_mean_ci(d, n_boot=n_perm, seed=seed, alpha=alpha)
        binary = bool(np.all(np.isin(g[np.isfinite(g)], (0.0, 1.0)))
                      and np.all(np.isin(r[np.isfinite(r)], (0.0, 1.0))))
        if binary:
            b = int(((g == 1) & (r == 0)).sum())
            c = int(((g == 0) & (r == 1)).sum())
            res["mcnemar"] = {"generated_only": b, "random_only": c,
                              "discordant": b + c,
                              "note": ("the sign-flip test is driven entirely by "
                                       "these %d discordant pairs" % (b + c))}
    else:
        # Unpaired permutation on the difference of means.
        pooled = np.concatenate([g, r])
        obs = float(np.nanmean(g) - np.nanmean(r))
        rs = np.random.RandomState(seed)
        hits = 0
        for _ in range(n_perm):
            p = rs.permutation(pooled)
            if abs(float(p[:g.size].mean() - p[g.size:].mean())) >= abs(obs) - 1e-15:
                hits += 1
        sd = math.sqrt((float(np.nanstd(g, ddof=1)) ** 2
                        + float(np.nanstd(r, ddof=1)) ** 2) / 2.0)
        res["test"] = {"method": "unpaired permutation of arm labels",
                       "observed": obs, "n_perm": n_perm}
        res["p_value"] = (1.0 + hits) / (1.0 + n_perm)
        res["effect_size_dz"] = obs / sd if sd > 0 else float("nan")

    p = res["p_value"]
    dz = res["effect_size_dz"]
    if not np.isfinite(p):
        verdict, reading = "inconclusive", "test could not be computed"
    elif p >= alpha or (np.isfinite(dz) and abs(dz) < min_dz):
        verdict = "adapter_does_nothing"
        reading = ("generated and random-matched-norm adapters are "
                   "INDISTINGUISHABLE (diff %+.3f, p=%.4f, d_z=%.2f). 6.7: "
                   "'the adapter does nothing and everything downstream is "
                   "noise'. Before concluding that, rule out the 4.5 silent-"
                   "failure mode -- an adapter the serving stack ignored also "
                   "looks exactly like this."
                   % (res["difference"], p, dz))
    elif res["difference"] > 0:
        verdict = "adapter_functional"
        reading = ("the generated adapter beats a random adapter of matched "
                   "norm (diff %+.3f, p=%.4f, d_z=%.2f)"
                   % (res["difference"], p, dz))
    else:
        verdict = "adapter_harmful"
        reading = ("the generated adapter is significantly WORSE than a random "
                   "adapter of matched norm (diff %+.3f, p=%.4f). The generator "
                   "is actively damaging the policy, not failing to help."
                   % (res["difference"], p))
    res["verdict"] = verdict
    res["reading"] = reading
    res["blocking"] = verdict in ("adapter_does_nothing", "adapter_harmful")
    return res


# --------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------

#: Checks that must have run and passed for `trustworthy` to be True.
REQUIRED_CHECKS = ("identity_probe", "diversity_collapse", "random_control")


def run_all(embeddings=None, labels=None, augmented_embeddings=None,
            generated=None, reference=None, generated_scores=None,
            random_scores=None, groups=None, require=REQUIRED_CHECKS,
            strict=True, out_path=None, seed=0, backend="auto", **thresholds):
    """Every 6.7 diagnostic in one JSON, with a `trustworthy` gate.

    A check that was not supplied is recorded as "skipped" and, when
    `strict=True` (the default) and it is in `require`, it BLOCKS: 6.7's title
    is "run before trusting any result", so "we did not run it" and "it failed"
    are the same thing as far as trusting the result goes. Pass
    `require=("random_control",)` to gate on a subset during development.
    """
    rep = collections.OrderedDict()
    rep["schema"] = "adaptercl.probe.run_all/1"
    rep["thresholds"] = {
        "near_perfect": thresholds.get("near_perfect", NEAR_PERFECT),
        "identifiable_margin": IDENTIFIABLE_MARGIN,
        "collapse_variance_ratio": thresholds.get("variance_ratio_threshold",
                                                  COLLAPSE_VARIANCE_RATIO),
        "collapse_rank_fraction": thresholds.get("rank_fraction_threshold",
                                                 COLLAPSE_RANK_FRACTION),
        "alpha": thresholds.get("alpha", ALPHA),
        "min_effect_dz": thresholds.get("min_dz", MIN_EFFECT_DZ),
    }
    rep["checks"] = collections.OrderedDict()
    blocking, warnings = [], []

    def _skip(name, why):
        rep["checks"][name] = {"status": "skipped", "reason": why}
        if strict and name in require:
            blocking.append({"check": name, "reason": "not run: " + why})

    # -- identity probe --------------------------------------------------
    if embeddings is not None and labels is not None:
        probes = identity_probes(embeddings, labels, seed=seed,
                                 backend=backend, groups=groups,
                                 near_perfect=rep["thresholds"]["near_perfect"])
        probes["status"] = "ok"
        rep["checks"]["identity_probe"] = probes
        if not probes["probes"].get("cell", {}).get("grouped"):
            warnings.append({"check": "identity_probe",
                             "reason": "stratified (ungrouped) split -- pass "
                                       "groups= if several embeddings share a page"})
        if probes["structure"] == "neither_identifiable":
            blocking.append({"check": "identity_probe",
                             "reason": probes["structure_note"]})
    else:
        _skip("identity_probe", "no embeddings/labels supplied")

    # -- augmentation ----------------------------------------------------
    if (embeddings is not None and augmented_embeddings is not None
            and labels is not None):
        aug = augmentation_report(embeddings, augmented_embeddings, labels,
                                  seed=seed, backend=backend, groups=groups)
        aug["status"] = "ok"
        rep["checks"]["augmentation"] = aug
        if aug["blocking"]:
            blocking.append({"check": "augmentation", "reason": aug["reading"]})
    else:
        rep["checks"]["augmentation"] = {"status": "skipped",
                                         "reason": "no augmented embeddings"}
        ip = rep["checks"].get("identity_probe", {})
        if isinstance(ip, dict) and ip.get("blocking"):
            blocking.append({"check": "augmentation",
                             "reason": ("the joint cell probe is near-perfect and "
                                        "6.7's augmentation mitigation was not "
                                        "run -- the lookup-table reading stands")})

    # -- diversity collapse ----------------------------------------------
    if generated is not None:
        dc = diversity_collapse(
            generated, reference=reference,
            variance_ratio_threshold=rep["thresholds"]["collapse_variance_ratio"],
            rank_fraction_threshold=rep["thresholds"]["collapse_rank_fraction"])
        dc["status"] = "ok"
        rep["checks"]["diversity_collapse"] = dc
        if dc["blocking"]:
            blocking.append({"check": "diversity_collapse", "reason": dc["reading"]})
    else:
        _skip("diversity_collapse", "no generated adapter matrix supplied")

    # -- random-adapter control -------------------------------------------
    if generated_scores is not None and random_scores is not None:
        rc = random_adapter_control(generated_scores, random_scores, seed=seed,
                                    alpha=rep["thresholds"]["alpha"],
                                    min_dz=rep["thresholds"]["min_effect_dz"])
        rc["status"] = "ok"
        rep["checks"]["random_control"] = rc
        if rc["blocking"]:
            blocking.append({"check": "random_control", "reason": rc["reading"]})
    else:
        _skip("random_control", "no generated/random score arrays supplied")

    rep["blocking"] = blocking
    rep["warnings"] = warnings
    rep["trustworthy"] = not blocking
    rep["summary"] = ("all 6.7 diagnostics pass -- downstream results may be "
                      "reported" if not blocking else
                      "%d blocking diagnostic(s); downstream results are not "
                      "interpretable until they are cleared" % len(blocking))
    if out_path:
        d = os.path.dirname(os.path.abspath(out_path))
        if d and not os.path.isdir(d):
            os.makedirs(d)
        with open(out_path, "w") as fh:
            json.dump(json_safe(rep), fh, indent=2, sort_keys=True)
    return rep


def format_report(rep, fh=None):
    import sys
    fh = fh or sys.stdout
    print("=" * 78, file=fh)
    print("adapterCL 6.7 -- diagnostics (run before trusting any result)", file=fh)
    print("=" * 78, file=fh)
    ip = rep["checks"].get("identity_probe", {})
    if ip.get("status") == "ok":
        any_probe = list(ip["probes"].values())[0]
        print("identity probe (%s):" % any_probe["split"], file=fh)
        print("  %-6s %8s %8s %8s %8s   %s"
              % ("target", "train", "test", "major", "margin", "verdict"), file=fh)
        for t, p in ip["probes"].items():
            print("  %-6s %8.3f %8.3f %8.3f %+8.3f   %s"
                  % (t, p["train_accuracy"], p["test_accuracy"],
                     p["chance_majority"], p["margin_over_majority"],
                     p["verdict"]), file=fh)
        print("  structure: %s -- %s" % (ip["structure"], ip["structure_note"]),
              file=fh)
    else:
        print("identity probe: SKIPPED (%s)" % ip.get("reason"), file=fh)
    ag = rep["checks"].get("augmentation", {})
    if ag.get("status") == "ok":
        print("augmentation: %s" % ag["verdict"], file=fh)
        for t, d in ag["drops"].items():
            print("  %-6s %.3f -> %.3f (accuracy drop %+.3f)%s"
                  % (t, d["plain"], d["augmented"], d["drop"],
                     "   <-- still near-perfect" if d["still_near_perfect"] else ""),
                  file=fh)
    else:
        print("augmentation: SKIPPED (%s)" % ag.get("reason"), file=fh)
    dc = rep["checks"].get("diversity_collapse", {})
    if dc.get("status") == "ok":
        print("diversity: n=%d dim=%d  var_ratio %.3f  eff_rank %.2f/%.0f "
              "(raw %.2f)  -> %s"
              % (dc["n_adapters"], dc["dim"], dc["variance_ratio"],
                 dc["effective_rank_centered"], dc["max_effective_rank"],
                 dc["effective_rank_raw"],
                 "COLLAPSED" if dc["collapsed"] else "ok"), file=fh)
        print("  %s" % dc["reading"], file=fh)
    else:
        print("diversity: SKIPPED (%s)" % dc.get("reason"), file=fh)
    rc = rep["checks"].get("random_control", {})
    if rc.get("status") == "ok":
        print("random control: generated %.3f vs random %.3f (diff %+.3f, "
              "p=%.4f, d_z=%.2f) -> %s"
              % (rc["mean_generated"], rc["mean_random"], rc["difference"],
                 rc["p_value"], rc["effect_size_dz"], rc["verdict"]), file=fh)
    else:
        print("random control: SKIPPED (%s)" % rc.get("reason"), file=fh)
    print("", file=fh)
    for w in rep["warnings"]:
        print("WARNING  [%s] %s" % (w["check"], w["reason"]), file=fh)
    for b in rep["blocking"]:
        print("BLOCKING [%s] %s" % (b["check"], b["reason"]), file=fh)
    print("", file=fh)
    print("TRUSTWORTHY: %s -- %s" % (rep["trustworthy"], rep["summary"]), file=fh)
    return rep


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------

def _synth_embeddings(cells, per_cell=12, d=32, era_scale=1.0, env_scale=1.0,
                      noise=0.15, seed=0):
    """Embeddings with a known, tunable amount of era and environment signal."""
    rs = np.random.RandomState(seed)
    eras = sorted(set(c.era for c in cells))
    envs = sorted(set(c.env for c in cells))
    era_dir = dict((e, rs.normal(size=d)) for e in eras)
    env_dir = dict((v, rs.normal(size=d)) for v in envs)
    X, y, groups = [], [], []
    for c in cells:
        for k in range(per_cell):
            X.append(era_scale * era_dir[c.era] + env_scale * env_dir[c.env]
                     + rs.normal(0.0, noise, size=d))
            y.append(c)
            groups.append("%s_page%d" % (c.key, k % 4))
    return np.array(X), y, groups


def selftest(verbose=True):
    import sys
    fh = sys.stdout
    cs = list(cells_mod.ALL_CELLS)
    ok = []

    def _say(msg):
        if verbose:
            print("  %s" % msg, file=fh)
        ok.append(msg)

    print("probe.py selftest", file=fh)

    # 1. separable embeddings -> near-perfect on every target --------------
    X, y, groups = _synth_embeddings(cs, per_cell=12, noise=0.10, seed=0)
    p = identity_probe(X, y, target="cell", backend="numpy")
    assert p["n_classes"] == 18
    assert abs(p["chance_uniform"] - 1.0 / 18) < 1e-12
    assert p["test_accuracy"] > 0.95, p
    assert p["near_perfect"] and p["verdict"] == "near_perfect"
    assert "lookup table" in p["reading"] and "6.7" in p["reading"]
    _say("linearly separable embeddings -> cell probe %.3f (chance %.3f) "
         "-> %s" % (p["test_accuracy"], p["chance_majority"], p["verdict"]))

    # 2. pure noise -> chance ----------------------------------------------
    rs = np.random.RandomState(1)
    Xn = rs.normal(size=(18 * 12, 32))
    pn = identity_probe(Xn, y, target="cell", backend="numpy")
    assert pn["test_accuracy"] < 0.30, pn["test_accuracy"]
    assert not pn["identifiable"] and pn["verdict"] == "confusable"
    _say("pure-noise embeddings   -> cell probe %.3f (majority %.3f) -> %s"
         % (pn["test_accuracy"], pn["chance_majority"], pn["verdict"]))

    # 3. the decomposition 3 predicts: era yes, environment no --------------
    Xe, ye, _ = _synth_embeddings(cs, per_cell=14, era_scale=1.0, env_scale=0.0,
                                  noise=0.35, seed=2)
    dec = identity_probes(Xe, ye, backend="numpy")
    assert dec["structure"] == "era_identifiable_env_confusable", dec["structure"]
    assert dec["probes"]["era"]["test_accuracy"] > 0.9
    assert not dec["probes"]["env"]["identifiable"]
    _say("era-only embeddings -> structure=%s (era %.3f, env %.3f) -- '%s'"
         % (dec["structure"], dec["probes"]["era"]["test_accuracy"],
            dec["probes"]["env"]["test_accuracy"], dec["structure_note"][:52]))

    # and the mirror image, which contradicts 3
    Xv, yv, _ = _synth_embeddings(cs, per_cell=14, era_scale=0.0, env_scale=1.0,
                                  noise=0.35, seed=3)
    dec2 = identity_probes(Xv, yv, backend="numpy")
    assert dec2["structure"] == "env_identifiable_era_confusable", dec2["structure"]
    _say("env-only embeddings -> structure=%s (contradicts 3)" % dec2["structure"])

    # 4. grouped split is honest about page leakage ------------------------
    pg = identity_probe(X, y, target="cell", groups=groups, backend="numpy")
    assert pg["grouped"] and "grouped" in pg["split"]
    assert "WARNING" in p["split"] and "groups=" in p["split"]
    _say("grouped split honoured (%d train / %d test); ungrouped split carries "
         "its leakage warning" % (pg["n_train"], pg["n_test"]))

    # 5. sklearn/numpy agreement (only if sklearn is importable) -----------
    try:
        import sklearn                                    # noqa: F401
        ps = identity_probe(X, y, target="cell", backend="sklearn")
        assert abs(ps["test_accuracy"] - p["test_accuracy"]) < 0.10
        _say("sklearn backend agrees with numpy (%.3f vs %.3f)"
             % (ps["test_accuracy"], p["test_accuracy"]))
    except ImportError:
        assert _fit_predict(X[:100], np.zeros(100, int), X[:10], 1, 1e-3,
                            "auto")[2]["backend"] == "numpy"
        _say("sklearn absent -> backend='auto' falls back to numpy cleanly")

    # 6. determinism --------------------------------------------------------
    W1, b1, _ = fit_multinomial(X[:200], np.arange(200) % 5, 5)
    W2, b2, _ = fit_multinomial(X[:200], np.arange(200) % 5, 5)
    assert np.array_equal(W1, W2) and np.array_equal(b1, b2)
    _say("fit_multinomial is bit-deterministic across runs")

    # 7. augmentation report -------------------------------------------------
    Xaug = X * 0.05 + rs.normal(0.0, 1.0, size=X.shape)     # augmentation destroys it
    ar = augmentation_report(X, Xaug, y, backend="numpy")
    assert ar["verdict"] == "mitigated" and ar["mitigated"] and not ar["blocking"]
    assert ar["drops"]["cell"]["drop"] > 0.5
    ar2 = augmentation_report(X, X * 1.001, y, backend="numpy")
    assert ar2["verdict"] == "not_mitigated" and ar2["blocking"]
    _say("augmentation_report: effective augmentation -> mitigated (drop %.2f); "
         "cosmetic augmentation -> not_mitigated + blocking"
         % ar["drops"]["cell"]["drop"])

    # 8. effective_rank on known spectra --------------------------------------
    rank1 = np.outer(np.arange(1.0, 11.0), np.ones(6))
    assert abs(effective_rank(rank1, center=False) - 1.0) < 1e-6
    E = np.zeros((8, 8))
    for i in range(4):
        E[i, i] = 1.0
        E[i + 4, i] = -1.0
    er = effective_rank(E, center=False)
    assert abs(er - 4.0) < 1e-6, er
    _say("effective_rank: rank-1 matrix -> %.3f, 4 equal orthogonal directions "
         "-> %.3f" % (effective_rank(rank1, center=False), er))

    # 9. diversity collapse ----------------------------------------------------
    base = rs.normal(size=512) * 3.0
    collapsed_set = np.array([base + rs.normal(0, 0.005, 512) for _ in range(18)])
    dc = diversity_collapse(collapsed_set)
    assert dc["collapsed"] and dc["blocking"], dc
    assert "variance_ratio" in dc["collapsed_by"]
    assert "mean adapter plus noise" in dc["reading"]
    diverse = np.array([base + rs.normal(0, 3.0, 512) for _ in range(18)])
    dv = diversity_collapse(diverse)
    assert not dv["collapsed"], dv
    # low-rank but high-variance: caught by rank_fraction, not variance_ratio
    B = rs.normal(size=(2, 512))
    lowrank = np.array([base + rs.normal(size=2).dot(B) * 3.0 for _ in range(18)])
    dl = diversity_collapse(lowrank)
    assert dl["collapsed"] and dl["collapsed_by"] == ["rank_fraction"], dl
    _say("diversity_collapse: mean+noise -> collapsed (var_ratio %.4f); diverse "
         "-> ok (rank %.1f/17); rank-2 set -> collapsed by rank_fraction only"
         % (dc["variance_ratio"], dv["effective_rank_centered"]))

    # 10. reference ceiling ----------------------------------------------------
    dcr = diversity_collapse(collapsed_set, reference=diverse)
    assert dcr["reference"]["relative_variance_ratio"] < 0.05
    assert "CEILING" in dcr["reference"]["caveat"]
    _say("reference arm reports relative spread %.4f and flags itself as an "
         "inflated ceiling (4.4)" % dcr["reference"]["relative_variance_ratio"])

    # 11. random-adapter control ------------------------------------------------
    rs2 = np.random.RandomState(5)
    same = rs2.binomial(1, 0.3, 60).astype(float)
    rc_null = random_adapter_control(same, same.copy(), n_perm=2000)
    assert rc_null["verdict"] == "adapter_does_nothing" and rc_null["blocking"]
    assert "4.5" in rc_null["reading"]        # points at the silent-failure mode
    gen = same.copy()
    flip = np.where(same == 0)[0][:14]
    gen[flip] = 1.0
    rc = random_adapter_control(gen, same, n_perm=2000)
    assert rc["verdict"] == "adapter_functional" and not rc["blocking"]
    assert rc["mcnemar"]["generated_only"] == 14 and rc["mcnemar"]["random_only"] == 0
    rc_bad = random_adapter_control(same, gen, n_perm=2000)
    assert rc_bad["verdict"] == "adapter_harmful" and rc_bad["blocking"]
    try:
        random_adapter_control(gen[:10], same)
        raise AssertionError("mismatched arms should raise")
    except ValueError as exc:
        assert "SAME task/seed list" in str(exc)
    _say("random_adapter_control: identical arms -> %s; +14 wins -> %s "
         "(p=%.4f, McNemar b/c=%d/%d); reversed -> %s"
         % (rc_null["verdict"], rc["verdict"], rc["p_value"],
            rc["mcnemar"]["generated_only"], rc["mcnemar"]["random_only"],
            rc_bad["verdict"]))

    # 12. run_all gating ---------------------------------------------------------
    good = run_all(embeddings=X, labels=y, augmented_embeddings=Xaug,
                   generated=diverse, generated_scores=gen, random_scores=same,
                   backend="numpy")
    assert good["trustworthy"], good["blocking"]
    bad = run_all(embeddings=X, labels=y, generated=collapsed_set,
                  generated_scores=same, random_scores=same.copy(),
                  backend="numpy")
    assert not bad["trustworthy"]
    names = sorted(b["check"] for b in bad["blocking"])
    assert names == ["augmentation", "diversity_collapse", "random_control"], names
    partial = run_all(generated=diverse, backend="numpy")
    assert not partial["trustworthy"]
    assert sorted(b["check"] for b in partial["blocking"]) == [
        "identity_probe", "random_control"]
    assert all("not run" in b["reason"] for b in partial["blocking"])
    relaxed = run_all(generated=diverse, require=("diversity_collapse",),
                      backend="numpy")
    assert relaxed["trustworthy"]
    txt = json.dumps(json_safe(good), sort_keys=True)
    assert "NaN" not in txt and "Infinity" not in txt
    _say("run_all: clean inputs -> trustworthy; collapsed+null inputs -> "
         "blocking %s; missing checks block unless `require` narrows them; "
         "output is strict JSON" % ",".join(names))

    print("OK -- %d checks" % len(ok), file=fh)
    return True


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

_NPZ_KEYS = ("embeddings", "labels", "augmented_embeddings", "generated",
             "reference", "generated_scores", "random_scores", "groups")


def _main():
    import argparse

    ap = argparse.ArgumentParser(
        description="adapterCL 6.7 diagnostics: identity probe, diversity "
                    "collapse, random-adapter control")
    ap.add_argument("--selftest", action="store_true",
                    help="run the synthetic known-answer checks and exit")
    ap.add_argument("--npz", default=None,
                    help="npz with any of: %s (labels are cell keys, "
                         "row-aligned with embeddings)" % ", ".join(_NPZ_KEYS))
    ap.add_argument("--json", default=None, help="write the report JSON here")
    ap.add_argument("--backend", default="auto",
                    choices=("auto", "numpy", "sklearn"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--require", default=",".join(REQUIRED_CHECKS),
                    help="comma-separated checks that must pass for "
                         "trustworthy=true (default: all)")
    ap.add_argument("--non-strict", action="store_true",
                    help="a skipped check no longer blocks (development only)")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return 0
    if not args.npz:
        ap.error("nothing to do: pass --npz with the diagnostic arrays, or "
                 "--selftest. Expected keys: %s" % ", ".join(_NPZ_KEYS))

    data = np.load(args.npz, allow_pickle=True)
    kw = {}
    for k in _NPZ_KEYS:
        if k in data.files:
            v = data[k]
            kw[k] = [str(x) for x in v.tolist()] if k in ("labels", "groups") else v
    if not kw:
        raise SystemExit("%s has none of the expected keys (%s); it has %s"
                         % (args.npz, ", ".join(_NPZ_KEYS), ", ".join(data.files)))
    paths.ensure_out_dirs()
    rep = run_all(seed=args.seed, backend=args.backend,
                  require=tuple(s.strip() for s in args.require.split(",") if s.strip()),
                  strict=not args.non_strict,
                  out_path=args.json, **kw)
    format_report(rep)
    if args.json:
        print("\nwrote %s" % args.json)
    return 0 if rep["trustworthy"] else 2


if __name__ == "__main__":
    raise SystemExit(_main())
