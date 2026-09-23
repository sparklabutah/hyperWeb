"""Phase-3 checks on the *generator* (6.7, 8.3, 9's risk register).

6.7 is explicit that these run "before trusting any result", and three of its
four failure modes are only visible on the generator side:

* the adapter is a genuine no-op at init, so training starts from the base
  policy (8.3) -- `noop_at_init`;
* generated adapters are distinguishable from a random adapter of matched norm,
  or "everything downstream is noise" -- `generated_vs_random`;
* generated adapters have not collapsed to "mean adapter plus noise" --
  `collapse_report`;
* and the generator is not simply ignoring its conditioning -- `sensitivity`,
  which is the cheap precursor to both.

Everything here is *generator-specific*. The version-identity memorisation probe
of 6.7 belongs to the conditioning embedding, not the adapter, and lives in
`probe.py`; where that module already implements a statistic (effective rank,
pairwise distances, "mean plus noise") we delegate to it and record which source
produced the number, so the same statistic never has two implementations that
quietly disagree. probe.py is written by another agent, so the import is lazy
and every statistic has a local fallback.

Weight-space comparison is valid here and only here. 4.4/6.2: independently
trained adapters cannot be compared by weight distance (`BA = (BR)(R^-1 A)`),
but adapters from one generator share a parametrisation, so distances between
them mean something. `collapse_report(..., dissociate=True)` therefore also
reports the 6.2 contrast -- within-era-across-environment vs
within-environment-across-era similarity -- which is the cross-check 6.2 asks
for against the Phase-2 functional transfer matrix.

Requires torch + numpy. Import from the `llamafactory` env python (paths.PY_TRAIN).
"""

from __future__ import print_function

import collections

import numpy as np
import torch

from . import cells as cells_mod
from . import hypernet
from . import inject as inject_mod

# --------------------------------------------------------------------------
# probe.py delegation
# --------------------------------------------------------------------------

def _probe_module():
    """probe.py, or None. Imported lazily: it is stdlib-only and torch-free, so
    it must not be dragged in at import time by a module that needs torch."""
    try:
        from . import probe
    except Exception:
        return None
    return probe


def probe_view(X, keys=None):
    """Run probe.py's 6.7 statistics over a set of adapter signatures.

    `probe.diversity_collapse` already owns the whole "mean adapter plus noise"
    test -- variance ratio, centred and raw effective rank, pairwise summary,
    documented thresholds, and a `collapsed` / `collapsed_by` / `reading`
    verdict. Rather than re-deriving any of that here, this returns its output
    verbatim so there is exactly one definition of "collapsed" in the project.

    Note the return types differ from the local fallbacks and must not be
    confused: `probe.effective_rank(X, center=True)` returns a **float**
    (exp-entropy of the singular spectrum), while the local `effective_rank`
    below returns a dict carrying both the participation ratio and the entropy
    form. Both are reported, under distinct keys, so a divergence between the
    two definitions is visible rather than silent.

    Returns (view, source) where view is a dict or None.
    """
    probe = _probe_module()
    if probe is None:
        return None, "unavailable (probe.py did not import)"
    X = np.asarray(X, dtype=np.float64)
    try:
        view = collections.OrderedDict([
            ("diversity_collapse", probe.diversity_collapse(X, keys=keys)),
            ("effective_rank_centered", float(probe.effective_rank(X, center=True))),
            ("effective_rank_raw", float(probe.effective_rank(X, center=False))),
            ("pairwise", probe.pairwise_distance_stats(X)),
        ])
    except Exception as exc:                    # noqa: BLE001
        return None, "unavailable (probe raised: %s)" % (exc,)
    return view, "probe.diversity_collapse"


# --------------------------------------------------------------------------
# Local statistics
# --------------------------------------------------------------------------

def _as_matrix(X):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("expected a (n, d) matrix, got %r" % (X.shape,))
    return X


def effective_rank(X):
    """Participation ratio and spectral entropy of the row set.

    Both answer "how many directions does this set actually span"; the
    participation ratio `(sum s)^2 / sum s^2` is the one 6.7's "effective rank"
    usually means, the entropy form is the information-theoretic sibling.
    Reported on the *centred* rows, because an uncentred set of near-identical
    adapters has a large first singular value that hides the collapse.
    """
    X = _as_matrix(X)
    Xc = X - X.mean(0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    s = s[s > 0]
    if s.size == 0:
        return {"participation_ratio": 0.0, "entropy_rank": 0.0,
                "n_rows": int(X.shape[0]), "singular_values": []}
    pr = float(s.sum() ** 2 / (s ** 2).sum())
    p = (s ** 2) / (s ** 2).sum()
    ent = float(np.exp(-(p * np.log(p)).sum()))
    return {"participation_ratio": pr, "entropy_rank": ent,
            "n_rows": int(X.shape[0]),
            "singular_values": [float(v) for v in s[:8]]}


def pairwise_stats(X):
    """Cosine and Euclidean distance summaries over the rows."""
    X = _as_matrix(X)
    n = X.shape[0]
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / np.clip(norms, 1e-12, None)
    C = Xn @ Xn.T
    D = np.sqrt(np.clip(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1), 0, None))
    off = ~np.eye(n, dtype=bool)
    scale = float(np.mean(norms)) or 1.0
    return {"cosine": C, "distance": D,
            "mean_cosine": float(C[off].mean()) if n > 1 else float("nan"),
            "min_cosine": float(C[off].min()) if n > 1 else float("nan"),
            "max_cosine": float(C[off].max()) if n > 1 else float("nan"),
            "mean_distance": float(D[off].mean()) if n > 1 else float("nan"),
            "mean_relative_distance": float(D[off].mean() / scale) if n > 1
            else float("nan")}


def mean_plus_noise(X):
    """How much of each adapter is just the *shared* adapter (6.7).

    `mean_energy_frac` near 1 and `ratio` >> 1 is the collapse signature: every
    generated adapter is the mean plus a small, near-orthogonal wiggle.
    """
    X = _as_matrix(X)
    mu = X.mean(0)
    resid = X - mu
    mean_norm = float(np.linalg.norm(mu))
    resid_rms = float(np.sqrt((resid ** 2).sum(1).mean()))
    energy = float((X ** 2).sum(1).mean())
    return {"mean_norm": mean_norm, "resid_rms": resid_rms,
            "ratio": mean_norm / resid_rms if resid_rms > 0 else float("inf"),
            "mean_energy_frac": (mean_norm ** 2 / energy) if energy > 0 else 1.0}


def delta_norms(A_list, B_list):
    """Per-site ||scaling-free dW||_F, computed without materialising dW.

    ||BA||_F^2 = tr(A^T B^T B A) = sum((B^T B) * (A A^T)) -- two r x r products
    instead of a d_out x d_in matrix.
    """
    out = []
    for A, B in zip(A_list, B_list):
        if A.dim() == 3:
            AAt = torch.matmul(A, A.transpose(-1, -2))
            BtB = torch.matmul(B.transpose(-1, -2), B)
            v = (BtB * AAt).sum(dim=(-1, -2)).clamp_min(0).sqrt()
        else:
            AAt = A @ A.transpose(0, 1)
            BtB = B.transpose(0, 1) @ B
            v = float((BtB * AAt).sum().clamp_min(0).sqrt())
        out.append(v)
    return out


# --------------------------------------------------------------------------
# 1. No-op at init (8.3)
# --------------------------------------------------------------------------

def _generator_device(generator, fallback=None):
    """The device a generator's parameters live on, or `fallback`.

    Every diagnostic here builds index tensors and a probe vector and feeds them
    to the generator. On a CUDA run those default to CPU while the generator is
    on the GPU, and the whole diagnostic suite dies on a device mismatch --
    precisely when it is most needed (6.7: "run before trusting any result").
    So device is inferred, never assumed.
    """
    for p in generator.parameters():
        return p.device
    for b in generator.buffers():
        return b.device
    return fallback if fallback is not None else torch.device("cpu")


def noop_at_init(generator, handle=None, tolerance=0.0, h=None, seed=0,
                 device=None):
    """Every generated B must be exactly zero before training (8.3).

    8.3's note -- "the generated adapter starts as identity, so training begins
    from the base policy's behavior rather than destroying it" -- is the reason
    this is a named test rather than a comment. `tolerance=0.0` means exactly
    zero, which is what a zeroed head (or a zero `out_scale`) gives; anything
    else means an init changed.

    `MixtureGenerator` sets `starts_as_noop = False` on purpose (it starts at
    the mean of the trained bank), so this reports rather than fails: `ok`
    tracks the generator's declared intent, `is_noop` the measurement.
    """
    src = handle if handle is not None else generator
    device = device or _generator_device(generator)
    layer_ids, module_ids = src.query_ids(device)
    if h is None:
        g = torch.Generator().manual_seed(seed)
        h = torch.randn(generator.d_cond, generator=g).to(device)
    else:
        h = torch.as_tensor(h).to(device)
    with torch.no_grad():
        A_list, B_list = generator(h, layer_ids, module_ids)
    max_b = max(float(b.abs().max()) for b in B_list)
    max_a = max(float(a.abs().max()) for a in A_list)
    nonzero = [s.rel_name for s, b in zip(src.sites, B_list)
               if float(b.abs().max()) > tolerance]
    is_noop = not nonzero
    expected = bool(getattr(generator, "starts_as_noop", True))
    res = {"kind": generator.kind, "expected_noop": expected,
           "is_noop": is_noop, "ok": (is_noop == expected),
           "max_abs_B": max_b, "max_abs_A": max_a,
           "n_nonzero_sites": len(nonzero), "n_sites": len(B_list),
           "nonzero_examples": nonzero[:5], "tolerance": tolerance}
    if expected and not is_noop:
        res["message"] = (
            "B is not zero at init: %d/%d sites nonzero (max |B| = %.3e). The "
            "generated adapter perturbs the base policy before a single "
            "gradient step, so any Phase-3 number is measured from a different "
            "starting policy than the baselines (8.3)."
            % (len(nonzero), len(B_list), max_b))
    elif not expected:
        res["message"] = (
            "%s does not start as a no-op by design -- with uniform "
            "coefficients it starts at the mean of the trained bank, which is "
            "the intended 6.5 baseline." % generator.kind)
    if handle is not None:
        handle.set_factors(A_list, B_list)
        res["handle_reports_noop"] = bool(inject_mod.adapter_is_noop(handle))
        handle.clear()
    return res


# --------------------------------------------------------------------------
# 2. Matched-norm random control (6.7 "functional sanity")
# --------------------------------------------------------------------------

def generated_vs_random(generator, handle, embeddings=None, key=None, h=None,
                        seed=0, device=None):
    """Generated adapter + a random adapter matched in ||dW||_F, per site.

    6.7: "Generated adapter vs random adapter of matched norm. Indistinguishable
    success means the adapter does nothing and everything downstream is noise."

    `inject.random_factors` takes a single global `scale`, which cannot match a
    generator whose per-site magnitudes differ by orders of magnitude (`k_proj`
    is 1024x4096, `down_proj` 4096x12288). So we draw at scale 1 and rescale
    each site so that ||B_r A_r||_F == ||B_g A_g||_F exactly -- the functional
    norm, not the factor norm, because the factors are only identified up to
    `A -> RA, B -> BR^-1` (4.4).

    Returns both factor sets ready to hand to `handle.set_factors`.
    """
    layer_ids, module_ids = handle.query_ids(device)
    if h is None:
        if embeddings:
            key = key or sorted(embeddings)[0]
            h = torch.as_tensor(embeddings[key]).float()
        else:
            g = torch.Generator().manual_seed(seed)
            h = torch.randn(generator.d_cond, generator=g)
    if device is not None:
        h = h.to(device)
    with torch.no_grad():
        A_gen, B_gen = generator(h, layer_ids, module_ids)
    if A_gen[0].dim() != 2:
        raise ValueError("pass a single conditioning vector; the control is "
                         "defined per adapter")
    target = delta_norms(A_gen, B_gen)

    g = torch.Generator(device="cpu").manual_seed(seed)
    A_rnd, B_rnd = inject_mod.random_factors(
        handle, scale=1.0, device=torch.device("cpu"), generator=g)
    # our A_gen may have a different rank than handle.rank (mixture delta mode)
    rank = A_gen[0].shape[0]
    if rank != handle.rank:
        A_rnd, B_rnd = [], []
        for s, A in zip(handle.sites, A_gen):
            A_rnd.append(torch.randn(rank, s.d_in, generator=g) / float(np.sqrt(s.d_in)))
            B_rnd.append(torch.randn(s.d_out, rank, generator=g) / float(np.sqrt(rank)))
    cur = delta_norms(A_rnd, B_rnd)
    A_out, B_out = [], []
    for A, B, t, c in zip(A_rnd, B_rnd, target, cur):
        f = float(np.sqrt(t / c)) if c > 0 and t > 0 else 0.0
        A_out.append(A * f)
        B_out.append(B * f)
    got = delta_norms(A_out, B_out)
    total_gen = float(np.sqrt(sum(x ** 2 for x in target)))
    res = {"generated": (A_gen, B_gen), "random": (A_out, B_out),
           "key": key,
           "delta_norm_generated": target, "delta_norm_random": got,
           "total_delta_norm": total_gen,
           "max_norm_mismatch": float(max(abs(a - b) for a, b in zip(target, got))
                                      if target else 0.0),
           "matched": all(abs(a - b) <= 1e-4 * max(1.0, a)
                          for a, b in zip(target, got))}
    if total_gen == 0.0:
        res["message"] = (
            "the generated adapter is identically zero, so the matched-norm "
            "control is zero too. This control is only meaningful after "
            "training -- at init the generator is a no-op by construction (8.3).")
    return res


# --------------------------------------------------------------------------
# 3. Diversity collapse (6.7, 9)
# --------------------------------------------------------------------------

def generated_signatures(generator, embeddings, handle=None, n_probe=8, seed=0,
                         keys=None, device=None):
    """(keys, (N, D) signature matrix) over one conditioning per cell."""
    src = handle if handle is not None else generator
    layer_ids, module_ids = src.query_ids(device)
    keys = list(keys or sorted(embeddings))
    H = torch.stack([torch.as_tensor(embeddings[k]).float() for k in keys], 0)
    if device is not None:
        H = H.to(device)
    with torch.no_grad():
        A_list, B_list = generator(H, layer_ids, module_ids)
        sig = hypernet.adapter_signature(A_list, B_list, n_probe=n_probe, seed=seed)
    return keys, sig.cpu().numpy(), (A_list, B_list)


def dissociation_contrast(keys, X):
    """6.2's contrast, in weight space, where it IS valid.

    Valid only because these adapters come from one generator and share a
    parametrisation (6.2: "Weight-space clustering is valid in one place:
    hypernetwork-generated adapters"). Compare the sign of this against the
    Phase-2 functional transfer matrix; agreement is evidence the generator
    learned the right organisation, disagreement is a red flag worth chasing.

    Returns mean cosine similarity within-era-across-environment vs
    within-environment-across-era, or None if the keys are not cell keys.
    """
    parsed = []
    for k in keys:
        try:
            parsed.append(cells_mod.Cell.parse(k))
        except Exception:
            return None
    stats = pairwise_stats(X)
    C = stats["cosine"]
    same_era, same_env = [], []
    for i, ci in enumerate(parsed):
        for j, cj in enumerate(parsed):
            if i == j:
                continue
            if ci.era == cj.era and ci.env != cj.env:
                same_era.append(C[i, j])
            if ci.env == cj.env and ci.era != cj.era:
                same_env.append(C[i, j])
    if not same_era or not same_env:
        return None
    a, b = float(np.mean(same_era)), float(np.mean(same_env))
    return {"within_era_across_env": a, "within_env_across_era": b,
            "delta": a - b, "n_era_pairs": len(same_era),
            "n_env_pairs": len(same_env),
            "organises_by": "era/appearance" if a > b else "environment/function"}


def collapse_report(generator, embeddings, handle=None, n_probe=8, seed=0,
                    keys=None, device=None, dissociate=True, fh=None):
    """Pairwise distances, effective rank, and the mean-plus-noise ratio (6.7).

    "Hypernetworks at this input:output ratio routinely converge to 'mean
    adapter plus noise'" -- this is the measurement that says whether yours did.

    The verdict is `probe.diversity_collapse`'s, not a second opinion invented
    here: probe.py owns the thresholds and the reasoning behind them, and having
    two definitions of "collapsed" in one project is how a sweep ends up
    reporting both. The local statistics below are computed too, and reported
    alongside, but they only *decide* the verdict when probe.py is unavailable.
    """
    keys, X, (A_list, B_list) = generated_signatures(
        generator, embeddings, handle=handle, n_probe=n_probe, seed=seed,
        keys=keys, device=device)
    view, source = probe_view(X, keys=keys)
    norms = delta_norms(A_list, B_list)
    per_cell = torch.stack([n if torch.is_tensor(n) else torch.tensor(n)
                            for n in norms], 0)  # (n_sites, N)
    res = collections.OrderedDict([
        ("keys", keys), ("n_cells", len(keys)), ("signature_dim", X.shape[1]),
        ("probe", view), ("probe_source", source),
        ("local_pairwise", pairwise_stats(X)),
        ("local_effective_rank", effective_rank(X)),
        ("local_mean_plus_noise", mean_plus_noise(X)),
        ("delta_norm_per_cell", per_cell.sum(0).tolist()),
        ("signatures", X),
    ])
    if dissociate:
        res["dissociation"] = dissociation_contrast(keys, X)

    if view is not None:
        dc = view["diversity_collapse"]
        res["collapsed"] = bool(dc["collapsed"])
        res["collapsed_by"] = dc.get("collapsed_by")
        res["reading"] = dc.get("reading")
        res["verdict_source"] = "probe.diversity_collapse"
    else:
        # Fallback only. Deliberately blunt, and labelled as such, so nobody
        # mistakes it for probe.py's calibrated thresholds.
        mn, er = res["local_mean_plus_noise"], res["local_effective_rank"]
        res["collapsed"] = bool(mn.get("mean_energy_frac", 0) > 0.99
                                and er.get("participation_ratio", 9) < 1.5)
        res["collapsed_by"] = ["local_heuristic"] if res["collapsed"] else []
        res["reading"] = "probe.py unavailable; local heuristic used"
        res["verdict_source"] = "local heuristic (probe.py unavailable)"
    if fh is not None:
        _print_collapse(res, fh)
    return res


def _print_collapse(res, fh):
    print("collapse report over %d conditionings (signature dim %d)"
          % (res["n_cells"], res["signature_dim"]), file=fh)
    view = res.get("probe")
    if view is not None:
        dc, pw = view["diversity_collapse"], view["pairwise"]
        print("  pairwise          mean %.4f  min %.4f  max %.4f  (%s)"
              % (pw["mean"], pw["min"], pw["max"], pw["metric"]), file=fh)
        print("  effective rank    centred %.2f  raw %.2f  (max %s, frac %.3f)"
              % (dc["effective_rank_centered"], dc["effective_rank_raw"],
                 dc["max_effective_rank"], dc["rank_fraction"]), file=fh)
        print("  mean+noise        within-set rms %.4g / ||mean|| %.4g "
              "-> variance ratio %.4f"
              % (dc["within_set_rms_deviation"], dc["mean_adapter_norm"],
                 dc["variance_ratio"]), file=fh)
    else:
        pw, er, mn = (res["local_pairwise"], res["local_effective_rank"],
                      res["local_mean_plus_noise"])
        print("  pairwise cosine   mean %.4f  min %.4f  max %.4f   [local]"
              % (pw["mean_cosine"], pw["min_cosine"], pw["max_cosine"]), file=fh)
        print("  effective rank    participation %.2f  entropy %.2f  (of %d) [local]"
              % (er["participation_ratio"], er["entropy_rank"], res["n_cells"]),
              file=fh)
        print("  mean+noise        ||mean||/rms(resid) %.2f  mean energy %.3f [local]"
              % (mn["ratio"], mn["mean_energy_frac"]), file=fh)
    d = res.get("dissociation")
    if d:
        print("  6.2 contrast      within-era %.4f vs within-env %.4f -> %s"
              % (d["within_era_across_env"], d["within_env_across_era"],
                 d["organises_by"]), file=fh)
    print("  VERDICT           %s   [%s]"
          % ("COLLAPSED" if res["collapsed"] else "not collapsed",
             res.get("verdict_source", "?")), file=fh)
    if res.get("reading"):
        print("  reading           %s" % (res["reading"],), file=fh)


# --------------------------------------------------------------------------
# 4. Sensitivity to the conditioning
# --------------------------------------------------------------------------

def sensitivity(generator, embeddings, handle=None, eps=1e-2, n_dirs=4, seed=0,
                n_probe=8, keys=None, device=None):
    """How much does the generated adapter move per unit move in conditioning?

    Jacobian-free: perturb `h` by `eps * ||h|| * u` for random unit `u` and
    measure the relative change in the adapter's functional signature. The
    reported `elasticity` is dimensionless --
    `(||ds||/||s||) / (||dh||/||h||)` -- so it is comparable across modalities
    and conditioning scales.

    A dead generator is one whose output barely moves: `elasticity` near 0 means
    the conditioning is decorative and every 6.5 conditioning-modality result is
    measuring nothing.

    Two very different failures both drive the response to exactly zero, and
    they must not be reported as one number:

    `adapter_is_zero`
        The generated adapter is identically zero, so of course it does not
        move. True by construction at init (B == 0, 8.3) and therefore
        *expected* before training -- but a red flag after it.
    `conditioning_independent`
        The adapter is non-zero but does not depend on `h` at all. This is the
        6.7 failure that matters: the generator has collapsed to a single
        adapter and the conditioning is decorative. It is invisible if you only
        look at whether the output moved.

    `dead_at_init` is kept as an alias of `adapter_is_zero` for callers that
    only care about the init check.
    """
    src = handle if handle is not None else generator
    layer_ids, module_ids = src.query_ids(device)
    keys = list(keys or sorted(embeddings))
    g = torch.Generator().manual_seed(seed)
    rows = []
    with torch.no_grad():
        for k in keys:
            h = torch.as_tensor(embeddings[k]).float()
            if device is not None:
                h = h.to(device)
            A0, B0 = generator(h, layer_ids, module_ids)
            s0 = hypernet.adapter_signature(A0, B0, n_probe=n_probe, seed=seed)
            n0 = float(s0.norm())
            hn = float(h.norm()) or 1.0
            for _ in range(n_dirs):
                u = torch.randn(h.shape, generator=g).to(h.device)
                u = u / u.norm().clamp_min(1e-12)
                dh = eps * hn * u
                A1, B1 = generator(h + dh, layer_ids, module_ids)
                s1 = hypernet.adapter_signature(A1, B1, n_probe=n_probe, seed=seed)
                ds = float((s1 - s0).norm())
                rows.append({
                    "key": k, "d_sig": ds, "d_h": float(dh.norm()),
                    "sig_norm": n0,
                    "abs_ratio": ds / max(float(dh.norm()), 1e-12),
                    # A zero-norm signature has no scale to normalise by, so
                    # elasticity is genuinely undefined there -- but the adapter
                    # being zero is itself the finding, reported separately.
                    "elasticity": ((ds / n0) / eps) if n0 > 0 else 0.0,
                })
    el = np.array([r["elasticity"] for r in rows], dtype=np.float64)
    ab = np.array([r["abs_ratio"] for r in rows], dtype=np.float64)
    norms = np.array([r["sig_norm"] for r in rows], dtype=np.float64)
    adapter_is_zero = bool(norms.size and np.all(norms == 0.0))
    unresponsive = bool(ab.size and np.all(ab == 0.0))
    conditioning_independent = bool(unresponsive and not adapter_is_zero)
    dead_at_init = adapter_is_zero

    # between-conditioning response: how far apart are two real interfaces?
    between = None
    if len(keys) > 1:
        _, X, _ = generated_signatures(generator, embeddings, handle=handle,
                                       n_probe=n_probe, seed=seed, keys=keys,
                                       device=device)
        H = np.stack([np.asarray(embeddings[k], dtype=np.float64) for k in keys], 0)
        dS = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
        dH = np.linalg.norm(H[:, None, :] - H[None, :, :], axis=-1)
        off = ~np.eye(len(keys), dtype=bool)
        ok = off & (dH > 0)
        between = {"mean_ratio": float((dS[ok] / dH[ok]).mean()) if ok.any() else float("nan"),
                   "mean_d_sig": float(dS[off].mean()),
                   "mean_d_h": float(dH[off].mean())}
    res = {"n_points": len(rows), "eps": eps, "n_dirs": n_dirs,
           "elasticity_mean": float(np.nanmean(el)) if el.size else float("nan"),
           "elasticity_median": float(np.nanmedian(el)) if el.size else float("nan"),
           "elasticity_min": float(np.nanmin(el)) if el.size else float("nan"),
           "elasticity_max": float(np.nanmax(el)) if el.size else float("nan"),
           "abs_ratio_mean": float(ab.mean()) if ab.size else float("nan"),
           "adapter_is_zero": adapter_is_zero,
           "conditioning_independent": conditioning_independent,
           "unresponsive": unresponsive,
           "dead_at_init": dead_at_init,
           "between_conditionings": between, "rows": rows}
    if adapter_is_zero:
        res["message"] = ("the generated adapter is identically zero -- expected "
                          "before training (B == 0 by construction, 8.3), a red "
                          "flag after it")
    elif conditioning_independent:
        res["message"] = ("the adapter is non-zero but does not depend on the "
                          "conditioning at all: the generator has collapsed to a "
                          "single adapter and every 6.5 conditioning-modality "
                          "result would be measuring nothing (6.7)")
    return res


# --------------------------------------------------------------------------
# Convenience
# --------------------------------------------------------------------------

def run_all(generator, handle, embeddings, fh=None, n_probe=8, seed=0,
            device=None):
    """Every Phase-3 diagnostic, printed. Returns the raw dicts.

    `device` is inferred from the generator when omitted, so this works on a
    CUDA run without the caller threading it through. Note that `noop_at_init`
    is only meaningful on a *freshly built* generator -- running it on a trained
    one measures nothing useful, and `expected_noop` will disagree.
    """
    import sys
    fh = fh or sys.stdout
    device = device or _generator_device(generator)
    out = collections.OrderedDict()
    out["noop"] = noop_at_init(generator, handle, device=device)
    print("noop_at_init: expected=%s measured=%s max|B|=%.3e %s"
          % (out["noop"]["expected_noop"], out["noop"]["is_noop"],
             out["noop"]["max_abs_B"], out["noop"].get("message", "")), file=fh)
    out["random"] = generated_vs_random(generator, handle, embeddings, seed=seed,
                                        device=device)
    print("generated_vs_random: ||dW|| total %.4e, matched=%s %s"
          % (out["random"]["total_delta_norm"], out["random"]["matched"],
             out["random"].get("message", "")), file=fh)
    out["collapse"] = collapse_report(generator, embeddings, handle,
                                      n_probe=n_probe, seed=seed, fh=fh,
                                      device=device)
    out["sensitivity"] = sensitivity(generator, embeddings, handle,
                                     n_probe=n_probe, seed=seed, device=device)
    s = out["sensitivity"]
    print("sensitivity: elasticity mean %.4f (median %.4f), dead=%s"
          % (s["elasticity_mean"], s["elasticity_median"], s["dead_at_init"]),
          file=fh)
    return out


# --------------------------------------------------------------------------
# Selftest
# --------------------------------------------------------------------------

def _selftest():
    from . import targets, toy

    ok = []

    def check(name, cond, extra=""):
        ok.append(bool(cond))
        print("  %-52s %s %s" % (name, "PASS" if cond else "FAIL", extra))

    torch.manual_seed(0)
    model, cfg = toy.tiny_model()
    sites = targets.enumerate_sites(cfg, "attn_mlp")
    handle = inject_mod.inject(model, "attn_mlp", rank=2, cfg=cfg)
    d_cond = 12
    free = hypernet.FreeHypernet(sites, 2, d_cond, d_hidden=32, d_emb=8)
    basis = hypernet.BasisGenerator(sites, 2, d_cond, n_basis=4, d_hidden=16,
                                    d_emb=8)
    bank = {}
    for i, key in enumerate(["wiki_e1", "wiki_e2", "news_e1"]):
        torch.manual_seed(10 + i)
        bank[key] = dict((s.rel_name, (torch.randn(2, s.d_in) * 0.05,
                                       torch.randn(s.d_out, 2) * 0.05))
                         for s in sites)
    mix = hypernet.MixtureGenerator(sites, 2, d_cond, adapters=bank,
                                    d_hidden=16, d_emb=8)

    print("[1] noop_at_init")
    r = noop_at_init(free, handle)
    check("free hypernet is an exact no-op at init",
          r["ok"] and r["is_noop"] and r["max_abs_B"] == 0.0
          and r["handle_reports_noop"])
    check("... and A is non-zero (only B is zeroed)", r["max_abs_A"] > 0)
    r = noop_at_init(basis, handle)
    check("basis generator is an exact no-op at init",
          r["ok"] and r["is_noop"] and r["max_abs_B"] == 0.0)
    r = noop_at_init(mix, handle)
    check("mixture reports not-a-no-op, and that is expected",
          r["ok"] and not r["is_noop"] and not r["expected_noop"]
          and "mean of the trained bank" in r["message"])
    broken = hypernet.FreeHypernet(sites, 2, d_cond, d_hidden=32, d_emb=8)
    with torch.no_grad():
        broken.heads["q_proj"].b.bias.fill_(0.01)
    r = noop_at_init(broken, handle)
    check("a broken zero-init is caught",
          not r["ok"] and r["n_nonzero_sites"] == 2 and "8.3" in r["message"],
          "%d nonzero site(s)" % r["n_nonzero_sites"])

    print("[2] generated_vs_random")
    torch.manual_seed(1)
    with torch.no_grad():
        basis.out_scale.fill_(0.7)
    emb = dict((c.key, torch.randn(d_cond).numpy())
               for c in cells_mod.cells_for(None, [1, 2, 3]))
    r = generated_vs_random(basis, handle, emb, key="wiki_e1")
    check("returns both factor sets ready for set_factors",
          len(r["generated"][0]) == len(handle) and len(r["random"][0]) == len(handle))
    check("per-site ||dW||_F matched exactly", r["matched"],
          "max mismatch %.2e" % r["max_norm_mismatch"])
    handle.set_factors(*r["random"])
    ids, labels = toy.tiny_batch(cfg)
    l_rand = float(model(ids, labels=labels).loss)
    handle.set_factors(*r["generated"])
    l_gen = float(model(ids, labels=labels).loss)
    handle.clear()
    l_base = float(model(ids, labels=labels).loss)
    check("both control and generated actually change the model",
          abs(l_rand - l_base) > 1e-6 and abs(l_gen - l_base) > 1e-6,
          "base %.4f gen %.4f rand %.4f" % (l_base, l_gen, l_rand))
    r0 = generated_vs_random(free, handle, emb)
    check("no-op generator gets an honest warning",
          r0["total_delta_norm"] == 0.0 and "identically zero" in r0["message"])
    rm = generated_vs_random(mix, handle, emb)
    check("mixture (rank K*r) gets a rank-matched control",
          rm["matched"] and rm["random"][0][0].shape[0] == mix.output_rank,
          "rank %d" % rm["random"][0][0].shape[0])

    print("[3] collapse_report")
    live = collapse_report(basis, emb, handle, n_probe=4)
    check("live generator is not flagged collapsed", not live["collapsed"],
          "mean cos %.3f, part.rank %.2f"
          % (live["local_pairwise"]["mean_cosine"],
             live["local_effective_rank"]["participation_ratio"]))
    dead = hypernet.BasisGenerator(sites, 2, d_cond, n_basis=4, per_site=False,
                                   d_hidden=16, d_emb=8)
    with torch.no_grad():
        dead.out_scale.fill_(1.0)
        dead.coeff.out.weight.zero_()        # coefficients ignore h -> one adapter
        dead.coeff.out.bias.copy_(torch.tensor([0.4, 0.3, 0.2, 0.1]))
    col = collapse_report(dead, emb, handle, n_probe=4)
    check("conditioning-independent generator IS flagged collapsed",
          col["collapsed"] and col["local_pairwise"]["mean_cosine"] > 0.999,
          "mean cos %.4f, mean energy %.4f"
          % (col["local_pairwise"]["mean_cosine"],
             col["local_mean_plus_noise"]["mean_energy_frac"]))
    check("the verdict comes from probe.py, not a second local definition",
          col["verdict_source"] == "probe.diversity_collapse"
          and col["probe"] is not None,
          col["verdict_source"])
    check("probe's collapse reasons are surfaced",
          bool(col.get("collapsed_by")) and bool(col.get("reading")),
          "collapsed_by=%s" % (col.get("collapsed_by"),))
    check("probe and local effective rank both reported, not conflated",
          isinstance(col["probe"]["effective_rank_centered"], float)
          and isinstance(col["local_effective_rank"], dict),
          "probe %.3f vs local part. %.3f"
          % (col["probe"]["effective_rank_centered"],
             col["local_effective_rank"]["participation_ratio"]))
    d = live["dissociation"]
    check("6.2 weight-space contrast computed from cell keys",
          d is not None and set(d) >= set(["within_era_across_env",
                                           "within_env_across_era", "delta"]),
          d["organises_by"] if d else "None")
    check("effective rank of N identical adapters is ~0",
          effective_rank(np.ones((5, 7)))["participation_ratio"] == 0.0)
    check("effective rank of an orthogonal set is ~N",
          abs(effective_rank(np.eye(6))["participation_ratio"] - 5.0) < 0.5,
          "%.2f" % effective_rank(np.eye(6))["participation_ratio"])

    print("[4] sensitivity")
    s = sensitivity(basis, emb, handle, n_dirs=2, n_probe=4)
    check("live generator has non-zero elasticity",
          not s["dead_at_init"] and s["elasticity_mean"] > 1e-6,
          "%.4f" % s["elasticity_mean"])
    check("between-conditioning response reported",
          s["between_conditionings"]["mean_ratio"] > 0)
    s0 = sensitivity(free, emb, handle, n_dirs=2, n_probe=4)
    check("zero adapter flagged adapter_is_zero (and explained)",
          s0["adapter_is_zero"] and s0["dead_at_init"]
          and not s0["conditioning_independent"] and "8.3" in s0["message"])
    s1 = sensitivity(dead, emb, handle, n_dirs=2, n_probe=4)
    check("conditioning-independent generator has ~0 elasticity",
          s1["elasticity_mean"] < 1e-6 and not s1["adapter_is_zero"],
          "%.3e" % s1["elasticity_mean"])
    check("... and is distinguished from a merely-zero adapter",
          s1["conditioning_independent"] and "collapsed to a single adapter"
          in s1["message"], s1["message"][:48])

    print("[5] run_all")
    import io as _io
    buf = _io.StringIO()
    out = run_all(basis, handle, emb, fh=buf)
    text = buf.getvalue()
    check("run_all prints every diagnostic",
          all(k in out for k in ("noop", "random", "collapse", "sensitivity"))
          and "VERDICT" in text and "elasticity" in text)

    n_fail = len([x for x in ok if not x])
    print("\n%d/%d checks passed" % (len(ok) - n_fail, len(ok)))
    return n_fail


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Phase-3 generator diagnostics "
                                             "(adapterCL 6.7)")
    ap.add_argument("--selftest", action="store_true", help="CPU-only checks")
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(1 if _selftest() else 0)
    ap.print_help()
