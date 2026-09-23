"""The functional transfer matrix and the dissociation experiment (6.2, 8.5).

6.2 is the paper's centerpiece: train one LoRA per (environment, era) cell,
cross-apply every adapter to every cell, and ask whether transfer is organised
by *theme* (appearance -> 3 holds) or by *environment* (site function -> 3 is
refuted). This module owns that matrix, the statistics that decide it, and the
figure.

Vocabulary, and it is a real trap
---------------------------------
The plan's "theme" is this codebase's **era**. `cells.Cell.theme` is something
else -- the per-site theme *directory* ("3-2003-4" on wiki, "3-2008s" on news,
"webshop2010" on shop, cells.py:48-55) -- which differs across sites for the
same era. Grouping on `Cell.theme` would put every cell in its own group and
silently produce an empty within-theme set. Everything here groups on
`Cell.era` and says "theme" only in the *output*, where it is flagged with
`theme_is_era`.

Three places this is stronger than 8.5's sketch, all deliberate
---------------------------------------------------------------
1. **Paired, not unpaired.** The two means share cells: the same 18 adapters
   generate both. `paired_permutation_test` sign-flips the per-cell difference
   (exact null space 2^18); `label_permutation_test` is the omnibus companion.
2. **Normalised is the headline.** The per-cell pairing contrasts two
   *different* column sets -- a wiki adapter's within-theme columns are
   {news,shop} while its within-environment columns are all wiki -- so per-cell
   difficulty does **not** cancel inside a pair. `normalized_transfer` divides
   by the oracle diagonal M[j,j], making each entry "fraction of the ceiling
   achievable on that cell". Raw is reported alongside and is unbiased only at
   the pair-level grand mean of a *complete* matrix (there the column marginal
   is uniform in both groups: 2 hits/column within-theme, 5 within-environment).
   See `HEADLINE_NOTE`.
3. **Guardrail, not a warning.** `weight_space_distance` refuses independently
   trained adapters outright (4.4: BA = (BR)(R^-1 A), so distances measure
   symmetry-group arbitrariness). Weight space is legal only for
   hypernetwork-generated adapters, which share a parametrisation -- 6.2's last
   paragraph -- via `generated_adapter_distances`.

Pure numpy + stdlib; importable from the system python (3.6.8, numpy 1.19.5).
torch is imported lazily and only when adapter tensors are actually read.
"""

from __future__ import print_function

import collections
import json
import math
import os

import numpy as np

from . import cells as cells_mod
from . import paths

# --------------------------------------------------------------------------
# Knobs, all documented where they are used
# --------------------------------------------------------------------------

#: Two-sided significance level for every test in this module.
DEFAULT_ALPHA = 0.05

#: Smallest difference in success rate we are willing to call a dissociation.
#: TimeWarp success rates sit between a 0% floor and ~40% (6.4), and 6.6 warns
#: that near-floor effects need care, so a 1-point gap is not a result even if
#: it clears p<0.05 with 18 paired cells.
DEFAULT_MIN_EFFECT = 0.02

#: Columns whose oracle (diagonal) success is below this are dropped from the
#: normalised matrix: dividing by ~0 manufactures enormous "transfer".
MIN_DIAG = 0.05

#: Default permutation/bootstrap resamples.
DEFAULT_N_PERM = 10000

HEADLINE_NOTE = (
    "Headline the NORMALISED dissociation. The paired unit is the source cell, "
    "and a source cell's within-theme targets (other environments, same era) "
    "are a different set of columns from its within-environment targets (same "
    "environment, other eras) -- so intrinsic cell difficulty does not cancel "
    "inside a pair and a merely hard environment masquerades as poor transfer. "
    "M[i,j]/M[j,j] is 'fraction of the oracle ceiling on cell j', which is what "
    "'transfer' should mean. Report raw alongside: at the pair-level grand mean "
    "of a COMPLETE matrix raw is unbiased (both groups have a uniform column "
    "marginal), so raw is the right sanity check and normalised is the right "
    "headline. If many diagonals are near zero (6.6's floor), normalisation is "
    "unstable -- `normalized_transfer` reports dropped columns and the verdict "
    "falls back to raw with a note."
)

PAIR_DIR_FMT = "%s__on__%s"   # <train cell>__on__<eval cell>


# --------------------------------------------------------------------------
# Small numeric helpers (shared with probe.py -- single implementation)
# --------------------------------------------------------------------------

def _as_cells(cells):
    """Accept training units: Cells, Versions, 'wiki_e3'/'v3' keys, or (env, era).

    A 6x6 matrix over Versions is the version-level experiment (2): the
    diagonal is each adapter on its own era, the off-diagonal is transfer to
    other eras. An 18x18 matrix over Cells is 6.2's crossing, which is the only
    thing that can ask the theme-vs-environment question -- `dissociation_scores`
    needs both factors and will refuse a Version matrix.
    """
    out = []
    for c in cells:
        if isinstance(c, (cells_mod.Cell, cells_mod.Version)):
            out.append(c)
        elif isinstance(c, str):
            out.append(cells_mod.parse_unit(c))
        elif isinstance(c, (tuple, list)) and len(c) == 2:
            out.append(cells_mod.Cell(c[0], int(c[1])))
        else:
            raise TypeError("cannot interpret %r as a training unit" % (c,))
    return out


def _finite(vals):
    a = np.asarray(list(vals), dtype=float)
    if a.size == 0:
        return a
    return a[np.isfinite(a)]


def _stats(vals):
    """{'mean','sd','n'} over the finite entries. NaN mean when empty."""
    a = _finite(vals)
    if a.size == 0:
        return {"mean": float("nan"), "sd": float("nan"), "n": 0}
    return {"mean": float(a.mean()),
            "sd": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "n": int(a.size)}


def rankdata(a):
    """Ranks 1..n with ties averaged. numpy-only stand-in for scipy.rankdata."""
    a = np.asarray(a, dtype=float)
    n = a.size
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    srt = a[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and srt[j + 1] == srt[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return ranks


def _pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xc, yc = x - x.mean(), y - y.mean()
    den = math.sqrt(float((xc * xc).sum()) * float((yc * yc).sum()))
    if den == 0.0:
        return float("nan")
    return float((xc * yc).sum() / den)


def spearman(x, y, n_perm=0, seed=0):
    """Spearman rho, ties averaged. Optional permutation p-value.

    numpy-only on purpose: scipy 1.0.0 happens to be installed on the system
    python but nothing else in this project depends on it.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("spearman: shape mismatch %r vs %r" % (x.shape, y.shape))
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3:
        return {"rho": float("nan"), "n": int(x.size), "p_value": float("nan")}
    rx, ry = rankdata(x), rankdata(y)
    rho = _pearson(rx, ry)
    p = float("nan")
    if n_perm:
        rs = np.random.RandomState(seed)
        hits = 0
        for _ in range(n_perm):
            if abs(_pearson(rx, rs.permutation(ry))) >= abs(rho) - 1e-15:
                hits += 1
        p = (1.0 + hits) / (1.0 + n_perm)
    return {"rho": float(rho), "n": int(x.size), "p_value": p,
            "n_perm": int(n_perm)}


def sign_flip_test(diffs, n_perm=DEFAULT_N_PERM, seed=0):
    """Paired permutation test on per-unit differences.

    The exact null is "the label 'within-theme' vs 'within-environment' is
    exchangeable within each cell", i.e. each d_i may flip sign. With k cells
    the null space is 2^k (262144 at k=18), so `n_perm` samples it rather than
    enumerating. Two-sided p uses the (1+hits)/(1+n) convention, which never
    reports p=0 -- an honest floor of 1/(n+1).
    """
    d = _finite(diffs)
    k = d.size
    if k < 2:
        raise ValueError("sign_flip_test needs >=2 paired units, got %d" % k)
    obs = float(d.mean())
    rs = np.random.RandomState(seed)
    signs = rs.randint(0, 2, size=(n_perm, k)) * 2 - 1
    null = (signs * d[None, :]).mean(axis=1)
    hits_two = int((np.abs(null) >= abs(obs) - 1e-15).sum())
    hits_up = int((null >= obs - 1e-15).sum())
    return {
        "method": "paired sign-flip permutation over units",
        "observed": obs,
        "n_units": int(k),
        "n_perm": int(n_perm),
        "exact_null_space": float(2.0 ** k),
        "p_value": (1.0 + hits_two) / (1.0 + n_perm),
        "p_one_sided_positive": (1.0 + hits_up) / (1.0 + n_perm),
        "null": {"mean": float(null.mean()), "sd": float(null.std(ddof=1)),
                 "q025": float(np.percentile(null, 2.5)),
                 "q975": float(np.percentile(null, 97.5)),
                 "max_abs": float(np.abs(null).max())},
        # sd can be numerically zero when every unit gives the same difference
        # (it happens on the normalised matrix, where the contrast is exactly
        # constant): report inf rather than 1e16.
        "effect_size_dz": (obs / float(d.std(ddof=1))
                           if d.std(ddof=1) > 1e-12 * max(1.0, abs(obs))
                           else (float("inf") if obs > 0 else
                                 float("-inf") if obs < 0 else float("nan"))),
    }


def bootstrap_mean_ci(vals, n_boot=DEFAULT_N_PERM, seed=0, alpha=DEFAULT_ALPHA):
    """Percentile bootstrap CI for a mean, resampling units with replacement."""
    a = _finite(vals)
    if a.size < 2:
        return {"mean": float(a.mean()) if a.size else float("nan"),
                "lo": float("nan"), "hi": float("nan"), "n": int(a.size)}
    rs = np.random.RandomState(seed)
    idx = rs.randint(0, a.size, size=(n_boot, a.size))
    boots = a[idx].mean(axis=1)
    return {"mean": float(a.mean()),
            "lo": float(np.percentile(boots, 100.0 * alpha / 2.0)),
            "hi": float(np.percentile(boots, 100.0 * (1.0 - alpha / 2.0))),
            "n": int(a.size), "n_boot": int(n_boot)}


def json_safe(obj):
    """Replace NaN/Inf with None so the output is *strict* JSON.

    Empty buckets legitimately produce NaN means (a cell pair that was never
    evaluated has no transfer), and `json.dump` writes those as the bare tokens
    `NaN`/`Infinity`, which Python re-reads but jq, JS and strict parsers
    reject. Everything this module writes to disk goes through here.
    """
    if isinstance(obj, dict):
        return dict((k, json_safe(v)) for k, v in obj.items())
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if not np.isfinite(v) else v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def spectrum_effective_rank(svals, eps=1e-12):
    """exp(H) of the normalised singular-value spectrum (Roy & Vetterli).

    Lives here so probe.effective_rank and generated_adapter_distances share one
    implementation. 1.0 == a single direction; k == k equally-weighted ones.
    """
    s = np.asarray(svals, dtype=float)
    s = s[s > eps]
    if s.size == 0:
        return 0.0
    p = s / s.sum()
    h = -float((p * np.log(p)).sum())
    return float(math.exp(h))


def ols(x, y):
    """Least-squares fit y = a + b x, plus r and r^2. numpy-only."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = x.size
    if n < 3 or x.std() == 0:
        return {"slope": float("nan"), "intercept": float("nan"),
                "r": float("nan"), "r2": float("nan"), "n": int(n)}
    b = float(((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum())
    a = float(y.mean() - b * x.mean())
    r = _pearson(x, y)
    return {"slope": b, "intercept": a, "r": float(r), "r2": float(r * r),
            "n": int(n)}


def _ols_perm_p(x, y, n_perm, seed):
    fit = ols(x, y)
    if not np.isfinite(fit["slope"]) or not n_perm:
        return float("nan")
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    rs = np.random.RandomState(seed)
    obs = abs(fit["slope"])
    hits = 0
    xc = x - x.mean()
    denom = float((xc * xc).sum())
    for _ in range(n_perm):
        yp = rs.permutation(y)
        b = float((xc * (yp - yp.mean())).sum() / denom)
        if abs(b) >= obs - 1e-15:
            hits += 1
    return (1.0 + hits) / (1.0 + n_perm)


# --------------------------------------------------------------------------
# The matrix
# --------------------------------------------------------------------------

class TransferMatrix(object):
    """N x N success rates: M[i, j] = success on cell j with cell i's adapter.

    NaN means "not run yet" -- the sweep is 324 evaluations (6.1) and will not
    finish in one sitting, so `set()` + `save()` are incremental by design and
    `load()` resumes. Per-entry provenance (episode counts, output dir) is kept
    beside the value so a half-finished matrix can be audited.
    """

    def __init__(self, cells, path=None, meta=None):
        self.cells = _as_cells(cells)
        if len(set(c.key for c in self.cells)) != len(self.cells):
            raise ValueError("duplicate cells in %r" % ([c.key for c in self.cells],))
        self.keys = [c.key for c in self.cells]
        self._index = dict((k, i) for i, k in enumerate(self.keys))
        self.n = len(self.cells)
        self.M = np.full((self.n, self.n), np.nan, dtype=float)
        self.entries = {}          # "src|dst" -> provenance dict
        self.path = path
        self.meta = dict(meta or {})

    # -- indexing ---------------------------------------------------------
    def index(self, cell):
        if isinstance(cell, (int, np.integer)):
            i = int(cell)
            if not 0 <= i < self.n:
                raise IndexError("cell index %d out of range 0..%d" % (i, self.n - 1))
            return i
        key = cell.key if isinstance(cell, cells_mod.Cell) else str(cell)
        if key not in self._index:
            raise KeyError("cell %r is not in this matrix (have %r)"
                           % (key, self.keys))
        return self._index[key]

    def set(self, i, j, value, **prov):
        """M[i, j] = value. i/j may be an index, a Cell, or a 'wiki_e3' key."""
        ii, jj = self.index(i), self.index(j)
        self.M[ii, jj] = float("nan") if value is None else float(value)
        if prov:
            self.entries["%s|%s" % (self.keys[ii], self.keys[jj])] = dict(prov)
        return self.M[ii, jj]

    def get(self, i, j):
        return float(self.M[self.index(i), self.index(j)])

    def provenance(self, i, j):
        return self.entries.get("%s|%s" % (self.keys[self.index(i)],
                                           self.keys[self.index(j)]))

    # -- completeness -----------------------------------------------------
    def missing_pairs(self, include_diagonal=True):
        out = []
        for i in range(self.n):
            for j in range(self.n):
                if i == j and not include_diagonal:
                    continue
                if not np.isfinite(self.M[i, j]):
                    out.append((self.cells[i], self.cells[j]))
        return out

    def n_filled(self):
        return int(np.isfinite(self.M).sum())

    def is_complete(self, include_diagonal=True):
        return not self.missing_pairs(include_diagonal=include_diagonal)

    def coverage(self):
        total = self.n * self.n
        return {"filled": self.n_filled(), "total": total,
                "fraction": self.n_filled() / float(total) if total else 0.0}

    # -- io ---------------------------------------------------------------
    def to_dict(self):
        rows = []
        for i in range(self.n):
            rows.append([None if not np.isfinite(v) else float(v)
                         for v in self.M[i]])
        return {"schema": "adaptercl.transfer.TransferMatrix/1",
                "cells": list(self.keys), "values": rows,
                "entries": self.entries, "meta": self.meta,
                "coverage": self.coverage()}

    @classmethod
    def from_dict(cls, d, path=None):
        tm = cls(d["cells"], path=path, meta=d.get("meta"))
        vals = d["values"]
        if len(vals) != tm.n:
            raise ValueError("matrix is %dx? but has %d cells" % (len(vals), tm.n))
        for i, row in enumerate(vals):
            if len(row) != tm.n:
                raise ValueError("row %d has %d entries, expected %d"
                                 % (i, len(row), tm.n))
            for j, v in enumerate(row):
                tm.M[i, j] = np.nan if v is None else float(v)
        tm.entries = dict(d.get("entries") or {})
        return tm

    def save(self, path=None):
        """Atomic incremental save. NaN is written as JSON null, not NaN."""
        path = path or self.path
        if path is None:
            raise ValueError("TransferMatrix.save() needs a path (none was set "
                             "at construction time)")
        d = os.path.dirname(os.path.abspath(path))
        if d and not os.path.isdir(d):
            os.makedirs(d)
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2, sort_keys=True)
        os.replace(tmp, path)
        self.path = path
        return path

    @classmethod
    def load(cls, path):
        if not os.path.exists(path):
            raise IOError("no transfer matrix at %s (run the sweep first, or "
                          "point --matrix somewhere else)" % path)
        with open(path) as fh:
            return cls.from_dict(json.load(fh), path=path)

    def __repr__(self):
        cov = self.coverage()
        return "TransferMatrix(%d cells, %d/%d filled)" % (
            self.n, cov["filled"], cov["total"])


def default_matrix_path(tag="percell"):
    return os.path.join(paths.OUT_TRANSFER, "transfer_%s.json" % tag)


# --------------------------------------------------------------------------
# Sweep planning (6.1: "an 18 x 18 transfer matrix is 324 evaluations")
# --------------------------------------------------------------------------

def build_pairs(cells, include_diagonal=True):
    """Every (train_cell, eval_cell) pair, row-major over `cells`.

    The diagonal is not optional in practice: M[j,j] is the oracle ceiling that
    `normalized_transfer` divides by and that 6.6 calls the oracle gap. Keeping
    it is 18 of 324 evaluations.
    """
    cs = _as_cells(cells)
    out = []
    for a in cs:
        for b in cs:
            if a == b and not include_diagonal:
                continue
            out.append((a, b))
    return out


def plan(cells=None, n_repeats=1, tasks_per_cell=None, include_diagonal=True,
         split="test", seconds_per_episode=120.0, n_parallel=1):
    """What the full matrix costs, before it is spent.

    `tasks_per_cell` defaults to the real per-environment task count from
    cells.cell_tasks(); pass an int to override (or when the task JSON is not
    readable). Episode counts are per *evaluation cell*, since that is what is
    actually rolled out.
    """
    cs = _as_cells(cells or cells_mod.ALL_CELLS)
    pairs = build_pairs(cs, include_diagonal=include_diagonal)

    per_cell, task_note = {}, "explicit override"
    if isinstance(tasks_per_cell, dict):
        per_cell = dict((k if isinstance(k, str) else k.key, int(v))
                        for k, v in tasks_per_cell.items())
    elif tasks_per_cell is not None:
        per_cell = dict((c.key, int(tasks_per_cell)) for c in cs)
    else:
        task_note = "cells.cell_tasks(split=%r, single_site=True)" % split
        try:
            for c in cs:
                per_cell[c.key] = len(cells_mod.cell_tasks(c, split=split))
        except (IOError, OSError, KeyError, ValueError) as exc:
            per_cell = dict((c.key, 20) for c in cs)
            task_note = ("FALLBACK 20 tasks/cell -- could not read the task "
                         "JSON (%s)" % exc)

    episodes = sum(per_cell[b.key] * n_repeats for _, b in pairs)
    gpu_hours = episodes * seconds_per_episode / 3600.0
    rep = {
        "n_cells": len(cs),
        "n_pairs": len(pairs),
        "include_diagonal": bool(include_diagonal),
        "plan_reference_324": len(cs) == 18 and len(pairs) == 324,
        "tasks_per_cell": per_cell,
        "tasks_per_cell_source": task_note,
        "n_repeats": int(n_repeats),
        "episodes": int(episodes),
        "adapter_swaps": len(pairs),
        "seconds_per_episode": float(seconds_per_episode),
        "wall_clock_hours_serial": gpu_hours,
        "wall_clock_hours_at_parallel": gpu_hours / float(max(1, n_parallel)),
        "n_parallel": int(n_parallel),
        "scaling": {},
    }
    for r in (1, 3, 5):
        eps = sum(per_cell[b.key] * r for _, b in pairs)
        rep["scaling"]["n_repeats=%d" % r] = {
            "episodes": int(eps),
            "hours_serial": eps * seconds_per_episode / 3600.0,
        }
    rep["note_seeds"] = ("6.6 asks for >=3 seeds per cell; the n_repeats=3 row "
                         "is that matrix. 6.2 needs only the mean per pair, so "
                         "seeds buy CI width, not the point estimate.")
    return rep


def format_plan(rep, fh=None):
    import sys
    fh = fh or sys.stdout
    print("transfer-matrix sweep plan", file=fh)
    print("  cells        : %d" % rep["n_cells"], file=fh)
    print("  pairs        : %d%s" % (rep["n_pairs"],
                                     "  (== 6.1's 18x18=324)" if rep["plan_reference_324"] else ""),
          file=fh)
    print("  tasks/cell   : %s" % rep["tasks_per_cell_source"], file=fh)
    ks = sorted(rep["tasks_per_cell"])
    print("                 %s" % ", ".join("%s=%d" % (k, rep["tasks_per_cell"][k])
                                            for k in ks), file=fh)
    print("  n_repeats    : %d" % rep["n_repeats"], file=fh)
    print("  episodes     : %s" % "{:,}".format(rep["episodes"]), file=fh)
    print("  adapter swaps: %d" % rep["adapter_swaps"], file=fh)
    print("  wall clock   : %.1f h serial, %.1f h at %dx parallel (@%.0fs/episode)"
          % (rep["wall_clock_hours_serial"], rep["wall_clock_hours_at_parallel"],
             rep["n_parallel"], rep["seconds_per_episode"]), file=fh)
    print("  scaling:", file=fh)
    for k in sorted(rep["scaling"]):
        s = rep["scaling"][k]
        print("    %-14s %9s episodes  %8.1f h serial"
              % (k, "{:,}".format(s["episodes"]), s["hours_serial"]), file=fh)
    print("  %s" % rep["note_seeds"], file=fh)
    return rep


# --------------------------------------------------------------------------
# Filling the matrix
# --------------------------------------------------------------------------

def functional_transfer(train_cell, eval_cell, evaluate, n_repeats=1):
    """One entry of the matrix -- the *only* valid way to compare independently
    trained adapters (4.4, 6.2).

    `evaluate(eval_cell, train_cell)` must return a success rate in [0, 1] (or a
    dict with a "success_rate" key). Averaged over `n_repeats` calls, so the
    caller can vary the seed inside `evaluate`.
    """
    vals = []
    for _ in range(max(1, int(n_repeats))):
        v = evaluate(eval_cell, train_cell)
        if isinstance(v, dict):
            v = v.get("success_rate")
        if v is None:
            raise ValueError("evaluate(%s, %s) returned no success rate"
                             % (eval_cell.key, train_cell.key))
        vals.append(float(v))
    return float(np.mean(vals))


def fill_matrix(tm, evaluate, pairs=None, n_repeats=1, save_every=1,
                verbose=True, skip_filled=True):
    """Populate a TransferMatrix, saving as it goes so the sweep can resume."""
    import sys
    pairs = pairs or build_pairs(tm.cells)
    done = 0
    for a, b in pairs:
        if skip_filled and np.isfinite(tm.M[tm.index(a), tm.index(b)]):
            continue
        val = functional_transfer(a, b, evaluate, n_repeats=n_repeats)
        tm.set(a, b, val, n_repeats=int(n_repeats), source="fill_matrix")
        done += 1
        if verbose:
            print("  %-10s -> %-10s  %.3f" % (a.key, b.key, val), file=sys.stderr)
        if tm.path and save_every and done % save_every == 0:
            tm.save()
    if tm.path:
        tm.save()
    return tm


def pair_dir_name(train_cell, eval_cell):
    """Directory convention for a cross-application eval run.

    `<root>/<train cell>__on__<eval cell>/`, e.g. `wiki_e3__on__shop_e3`. The
    train cell comes first because that is what varies per adapter load, so the
    directory sorts by adapter and a partially finished sweep reads cleanly.
    """
    return PAIR_DIR_FMT % (train_cell.key, eval_cell.key)


def parse_pair_dir(name):
    """'wiki_e3__on__shop_e3' -> (Cell(wiki,3), Cell(shop,3))."""
    if "__on__" not in name:
        raise ValueError("%r is not a pair directory (expected %r)"
                         % (name, PAIR_DIR_FMT % ("<train>", "<eval>")))
    a, b = name.split("__on__", 1)
    return cells_mod.Cell.parse(a), cells_mod.Cell.parse(b)


def _success_rate_from_results(res, threshold=1.0):
    """Coerce whatever evalbridge.read_results returns into a success rate.

    Per-episode records are preferred over the pre-computed aggregate, because
    evalbridge.aggregate() bakes in its own success threshold (1.0) and this
    module lets the caller pick one; recomputing from the (already deduped)
    records keeps the two in step. The aggregate is used only when there are no
    records, and then only at threshold 1.0 -- silently applying the wrong
    threshold is the kind of thing that shifts a whole matrix by a few points
    with nothing to show for it.

    Three shapes are accepted -- {"records":..,"aggregate":..} (what evalbridge
    returns), (records, aggregate), and a bare record list -- so this keeps
    working if the bridge's return shape is revised.
    """
    records, aggregate = None, None
    if isinstance(res, tuple) and len(res) == 2:
        records, aggregate = res
    elif isinstance(res, dict):
        if "records" in res or "episodes" in res:
            records = res.get("records") or res.get("episodes")
            aggregate = res.get("aggregate") or res
        else:
            aggregate = res
    elif isinstance(res, list):
        records = res
    else:
        raise TypeError("evalbridge.read_results returned %r, which is neither "
                        "a record list, a dict, nor (records, aggregate)"
                        % type(res))

    if records:
        rewards = [float(r.get("reward", 0.0) or 0.0) for r in records]
        ok = [1.0 if r >= threshold else 0.0 for r in rewards]
        return float(np.mean(ok)), len(ok)

    if isinstance(aggregate, dict):
        for k in ("success_rate", "success", "mean_success"):
            if aggregate.get(k) is not None:
                if abs(threshold - 1.0) > 1e-12:
                    raise ValueError(
                        "this eval dir has no per-episode records, only an "
                        "aggregate computed at evalbridge's threshold=1.0, but "
                        "success_threshold=%r was requested. Re-run with "
                        "success_threshold=1.0 or point at a run whose trial "
                        "CSVs are readable." % (threshold,))
                return float(aggregate[k]), int(aggregate.get("n")
                                                or aggregate.get("n_episodes") or 0)
    raise ValueError("no episode records and no success rate in the aggregate "
                     "(status=%r) -- the eval is empty or half-written"
                     % (res.get("status") if isinstance(res, dict) else None))


def from_eval_dirs(root, cells=None, tag_fn=None, success_threshold=1.0,
                   matrix=None, strict=False, verbose=False):
    """Build/refresh a TransferMatrix by scanning per-pair eval output dirs.

    Convention: `<root>/<train_cell>__on__<eval_cell>/` (see `pair_dir_name`).
    Pass `tag_fn(train_cell, eval_cell) -> str` to override it. Missing
    directories leave NaN unless `strict=True`.
    """
    try:
        from . import evalbridge
    except ImportError as exc:
        raise ImportError(
            "from_eval_dirs needs adaptercl.evalbridge (read_results); it is "
            "not importable yet: %s. Until it lands, fill the matrix with "
            "fill_matrix(tm, evaluate=...) or TransferMatrix.set()." % exc)
    if not hasattr(evalbridge, "read_results"):
        raise ImportError("adaptercl.evalbridge has no read_results(); this "
                          "module needs read_results(output_root).")

    cs = _as_cells(cells or cells_mod.ALL_CELLS)
    tm = matrix or TransferMatrix(cs)
    tag_fn = tag_fn or pair_dir_name
    found, missing, failed = 0, [], []
    for a, b in build_pairs(cs):
        d = os.path.join(root, tag_fn(a, b))
        if not os.path.isdir(d):
            missing.append(os.path.basename(d))
            continue
        try:
            rate, n = _success_rate_from_results(
                evalbridge.read_results(d), threshold=success_threshold)
        except Exception as exc:                      # noqa: BLE001 -- report, don't die
            failed.append((os.path.basename(d), "%s: %s" % (type(exc).__name__, exc)))
            if strict:
                raise
            continue
        tm.set(a, b, rate, n_episodes=n, exp_dir=d, source="from_eval_dirs")
        found += 1
        if verbose:
            print("  %-24s %.3f (n=%d)" % (os.path.basename(d), rate, n))
    if strict and missing:
        raise IOError("%d pair directories missing under %s: %s%s"
                      % (len(missing), root, ", ".join(missing[:6]),
                         " ..." if len(missing) > 6 else ""))
    tm.meta.setdefault("from_eval_dirs", {})
    tm.meta["from_eval_dirs"].update({"root": root, "found": found,
                                      "missing": missing, "failed": failed,
                                      "success_threshold": success_threshold})
    return tm


# --------------------------------------------------------------------------
# Matrix transforms
# --------------------------------------------------------------------------

def _matrix(M):
    """Accept a TransferMatrix or an array; return (array, cells_or_None)."""
    if isinstance(M, TransferMatrix):
        return np.array(M.M, dtype=float), list(M.cells)
    return np.asarray(M, dtype=float), None


def self_transfer(M):
    """The diagonal: the oracle per-cell adapter, 6.6's ceiling."""
    A, _ = _matrix(M)
    return np.array(np.diag(A), dtype=float)


def normalized_transfer(M, min_diag=MIN_DIAG):
    """M[i,j] / M[j,j] -- transfer as a fraction of the oracle on cell j.

    Returns (normalised array, info). Columns whose diagonal is below
    `min_diag` become all-NaN and are listed in info["dropped_columns"]: with a
    0% floor in play (6.4/6.6), dividing by a near-zero oracle turns noise into
    spectacular "transfer". Values may exceed 1.0 -- an adapter really can beat
    the cell's own adapter -- and that is not clipped.
    """
    A, cs = _matrix(M)
    d = np.diag(A).astype(float)
    out = np.full(A.shape, np.nan, dtype=float)
    dropped = []
    for j in range(A.shape[1]):
        if not np.isfinite(d[j]) or d[j] < min_diag:
            dropped.append(cs[j].key if cs else j)
            continue
        out[:, j] = A[:, j] / d[j]
    info = {"min_diag": float(min_diag), "dropped_columns": dropped,
            "n_dropped": len(dropped), "n_columns": int(A.shape[1]),
            "usable": len(dropped) < A.shape[1] // 2,
            "note": "values >1 mean the foreign adapter beat the cell's own"}
    return out, info


def asymmetry(M):
    """Directional asymmetry of the matrix: mean |M[i,j] - M[j,i]| off-diagonal."""
    A, _ = _matrix(M)
    n = A.shape[0]
    vals = [abs(A[i, j] - A[j, i]) for i in range(n) for j in range(i + 1, n)
            if np.isfinite(A[i, j]) and np.isfinite(A[j, i])]
    return _stats(vals)


def directional_asymmetry(M, cells):
    """Old-adapter-on-new-UI vs new-adapter-on-old-UI, within environment.

    6.1 makes forward extrapolation (train old, test new) the headline split and
    the reverse the symmetric check. If the matrix says one direction transfers
    better, the two splits are not interchangeable and the paper should say so.
    Uses cells.NOMINAL_YEAR, so the neutral era 6 (year None) is excluded.
    """
    A, cs = _matrix(M)
    cs = _as_cells(cells) if cells is not None else cs
    fwd, bwd = [], []
    for i, a in enumerate(cs):
        for j, b in enumerate(cs):
            if i == j or a.env != b.env:
                continue
            ya, yb = a.year, b.year
            if ya is None or yb is None or not np.isfinite(A[i, j]):
                continue
            (fwd if ya < yb else bwd).append(A[i, j])
    f, b_ = _stats(fwd), _stats(bwd)
    return {"older_adapter_on_newer_ui": f,
            "newer_adapter_on_older_ui": b_,
            "difference": f["mean"] - b_["mean"],
            "note": ("positive => forward extrapolation (6.1's headline split) "
                     "is the easier direction")}


# --------------------------------------------------------------------------
# The dissociation (6.2, 8.5)
# --------------------------------------------------------------------------

THEME_NOTE = ("'theme' is the plan's word for what cells.py calls 'era'. "
              "Cell.theme is the per-site theme directory and differs across "
              "sites for the same era -- never group on it.")


def _group_of(a, b):
    """Which contrast an ordered off-diagonal pair belongs to."""
    if a.era == b.era and a.env != b.env:
        return "theme"      # within-theme, across-environment
    if a.env == b.env and a.era != b.era:
        return "env"        # within-environment, across-theme
    return "neither"        # nothing shared -- the floor


def dissociation_scores(M, cells=None, min_diag=MIN_DIAG):
    """8.5's two means, done properly.

    Returns pair-level grand means (what 8.5 computes) *and* cell-level means of
    per-cell means (what the paired test uses); with a complete matrix the two
    agree only up to the 2-vs-5 imbalance in group sizes, so both are reported.
    """
    A, cs = _matrix(M)
    cs = _as_cells(cells) if cells is not None else cs
    if cs is None:
        raise ValueError("dissociation_scores needs cells (pass a "
                         "TransferMatrix or an explicit cell list)")
    if A.shape[0] != len(cs) or A.shape[1] != len(cs):
        raise ValueError("matrix is %r but %d cells were given"
                         % (A.shape, len(cs)))

    # 6.2's contrast needs BOTH factors to vary. A Version matrix has one era
    # per unit and no environment factor at all, so every within-theme and
    # within-environment bucket comes out empty and the means are NaN -- which
    # previously returned a well-formed dict of Nones rather than refusing.
    # A silently-empty dissociation is exactly the failure 9 calls out as
    # "weight-space adapter comparison used by mistake: silent invalidity".
    envs = set(getattr(c, "env", None) for c in cs)
    eras = set(c.era for c in cs)
    if len(envs) < 2 or None in envs:
        raise ValueError(
            "dissociation_scores needs an environment x era grid (6.2), but "
            "these %d unit(s) span environments=%r. A Version matrix cannot "
            "answer the theme-vs-environment question -- it has no environment "
            "factor. Build the matrix over cells.ALL_CELLS, or use "
            "distance_decay() for the version-level 1-D manifold test (6.1)."
            % (len(cs), sorted(str(e) for e in envs)))
    if len(eras) < 2:
        raise ValueError("dissociation_scores needs >= 2 eras, got %r"
                         % (sorted(eras),))

    buckets = {"theme": [], "env": [], "neither": []}
    per_src = collections.OrderedDict()
    per_dst = collections.OrderedDict()
    for c in cs:
        per_src[c.key] = {"theme": [], "env": []}
        per_dst[c.key] = {"theme": [], "env": []}
    for i, a in enumerate(cs):
        for j, b in enumerate(cs):
            if i == j:
                continue
            g = _group_of(a, b)
            v = A[i, j]
            buckets[g].append(v)
            if g in ("theme", "env"):
                per_src[a.key][g].append(v)
                per_dst[b.key][g].append(v)

    def _pack(d):
        rows = []
        for key, g in d.items():
            t, e = _stats(g["theme"]), _stats(g["env"])
            rows.append({"cell": key, "theme_mean": t["mean"], "theme_n": t["n"],
                         "env_mean": e["mean"], "env_n": e["n"],
                         "difference": t["mean"] - e["mean"]})
        return rows

    src_rows = _pack(per_src)
    dst_rows = _pack(per_dst)
    cell_diffs = [r["difference"] for r in src_rows]

    theme, env = _stats(buckets["theme"]), _stats(buckets["env"])
    out = collections.OrderedDict()
    out["theme_is_era"] = True
    out["note"] = THEME_NOTE
    out["within_theme_across_env"] = theme
    out["within_env_across_theme"] = env
    out["neither_shared"] = _stats(buckets["neither"])
    out["diagonal_oracle"] = _stats(np.diag(A))
    out["difference"] = theme["mean"] - env["mean"]
    out["cell_level"] = {
        "within_theme_across_env": _stats([r["theme_mean"] for r in src_rows]),
        "within_env_across_theme": _stats([r["env_mean"] for r in src_rows]),
        "difference": _stats(cell_diffs)["mean"],
        "n_cells": int(_finite(cell_diffs).size),
    }
    out["per_source_cell"] = src_rows
    out["per_target_cell"] = dst_rows
    out["coverage"] = {
        "theme_pairs_filled": theme["n"],
        "theme_pairs_total": sum(1 for i, a in enumerate(cs)
                                 for j, b in enumerate(cs)
                                 if i != j and _group_of(a, b) == "theme"),
        "env_pairs_filled": env["n"],
        "env_pairs_total": sum(1 for i, a in enumerate(cs)
                               for j, b in enumerate(cs)
                               if i != j and _group_of(a, b) == "env"),
    }
    return out


def paired_permutation_test(M, cells=None, n=DEFAULT_N_PERM, seed=0,
                            direction="source"):
    """Paired permutation over the environment/era labels, not a t-test.

    The two means are computed from the *same* 18 adapters, so an unpaired
    two-sample test (8.5's implicit comparison of two lists of pair means)
    both ignores the pairing and treats the 36 within-theme and 90
    within-environment entries as independent observations, which they are not:
    each row shares one adapter and each column shares one task set. The unit
    here is the cell, and the exchangeable label within a cell is
    theme-vs-environment -- hence a sign-flip on the per-cell difference.
    """
    sc = dissociation_scores(M, cells)
    rows = sc["per_source_cell"] if direction == "source" else sc["per_target_cell"]
    diffs = [r["difference"] for r in rows]
    res = sign_flip_test(diffs, n_perm=n, seed=seed)
    res["direction"] = direction
    res["unit"] = "cell (%s of the ordered pair)" % direction
    res["statistic"] = ("mean over cells of (mean within-theme transfer - "
                        "mean within-environment transfer)")
    return res


def label_permutation_test(M, cells=None, n=DEFAULT_N_PERM, seed=0):
    """Omnibus companion: shuffle which (env, era) label each matrix row/column
    carries and recompute the pair-level difference.

    This tests "the matrix has no environment/era structure at all", which is a
    different and weaker null than the paired test's "the two contrasts are
    exchangeable within a cell". Report both: the paired test is the headline,
    this one guards against a matrix whose apparent structure is an artefact of
    the grouping itself.
    """
    A, cs = _matrix(M)
    cs = _as_cells(cells) if cells is not None else cs
    n_cells = len(cs)

    def _diff(perm):
        th, en = [], []
        for i in range(n_cells):
            for j in range(n_cells):
                if i == j:
                    continue
                a, b = cs[perm[i]], cs[perm[j]]
                v = A[i, j]
                if not np.isfinite(v):
                    continue
                g = _group_of(a, b)
                if g == "theme":
                    th.append(v)
                elif g == "env":
                    en.append(v)
        if not th or not en:
            return float("nan")
        return float(np.mean(th) - np.mean(en))

    identity = np.arange(n_cells)
    obs = _diff(identity)
    rs = np.random.RandomState(seed)
    null = np.array([_diff(rs.permutation(n_cells)) for _ in range(n)])
    null = null[np.isfinite(null)]
    hits = int((np.abs(null) >= abs(obs) - 1e-15).sum())
    return {"method": "permutation of (env, era) labels over matrix positions",
            "observed": obs, "n_perm": int(null.size),
            "p_value": (1.0 + hits) / (1.0 + null.size),
            "null": {"mean": float(null.mean()), "sd": float(null.std(ddof=1)),
                     "q025": float(np.percentile(null, 2.5)),
                     "q975": float(np.percentile(null, 97.5))}}


def bootstrap_ci(M, cells=None, n=DEFAULT_N_PERM, seed=0, alpha=DEFAULT_ALPHA,
                 direction="source"):
    """Cluster bootstrap over cells for both means and their difference.

    Cells are the resampling unit (not pairs): pairs inside a cell share an
    adapter, so a pair-level bootstrap would understate the interval.
    """
    sc = dissociation_scores(M, cells)
    rows = sc["per_source_cell"] if direction == "source" else sc["per_target_cell"]
    theme = np.array([r["theme_mean"] for r in rows], dtype=float)
    env = np.array([r["env_mean"] for r in rows], dtype=float)
    ok = np.isfinite(theme) & np.isfinite(env)
    theme, env = theme[ok], env[ok]
    if theme.size < 2:
        raise ValueError("bootstrap_ci needs >=2 usable cells, got %d" % theme.size)
    rs = np.random.RandomState(seed)
    idx = rs.randint(0, theme.size, size=(n, theme.size))
    bt, be = theme[idx].mean(axis=1), env[idx].mean(axis=1)
    bd = bt - be
    lo, hi = 100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)

    def _ci(obsv, boots):
        return {"mean": float(obsv), "lo": float(np.percentile(boots, lo)),
                "hi": float(np.percentile(boots, hi))}

    return {"unit": "cell", "n_cells": int(theme.size), "n_boot": int(n),
            "alpha": alpha, "direction": direction,
            "within_theme_across_env": _ci(theme.mean(), bt),
            "within_env_across_theme": _ci(env.mean(), be),
            "difference": _ci(theme.mean() - env.mean(), bd),
            "excludes_zero": bool(np.percentile(bd, lo) > 0
                                  or np.percentile(bd, hi) < 0)}


def by_era(M, cells=None):
    """Per-era breakdown. 6.6: 'per environment and per era, never means only'."""
    A, cs = _matrix(M)
    cs = _as_cells(cells) if cells is not None else cs
    out = collections.OrderedDict()
    for era in cells_mod.ERAS:
        th, en, dg = [], [], []
        for i, a in enumerate(cs):
            for j, b in enumerate(cs):
                if i == j:
                    if a.era == era:
                        dg.append(A[i, j])
                    continue
                g = _group_of(a, b)
                if g == "theme" and a.era == era:
                    th.append(A[i, j])
                elif g == "env" and a.era == era:
                    en.append(A[i, j])
        out["era_%d" % era] = {
            "era": era, "is_temporal": era in cells_mod.TEMPORAL_ERAS,
            "year_spread": cells_mod.ERA_YEAR_SPREAD.get(era),
            "within_theme_across_env": _stats(th),
            "within_env_across_theme": _stats(en),
            "diagonal": _stats(dg),
            "difference": _stats(th)["mean"] - _stats(en)["mean"],
        }
    out["_note"] = ("year_spread is cells.ERA_YEAR_SPREAD -- how far the three "
                    "sites disagree about what this era looks like. 6.2's "
                    "precondition is style correspondence, so a large spread "
                    "predicts a weak within-theme mean for that era.")
    return out


def by_env(M, cells=None):
    """Per-environment breakdown."""
    A, cs = _matrix(M)
    cs = _as_cells(cells) if cells is not None else cs
    out = collections.OrderedDict()
    for env in cells_mod.ENVIRONMENTS:
        th, en, dg = [], [], []
        for i, a in enumerate(cs):
            for j, b in enumerate(cs):
                if i == j:
                    if a.env == env:
                        dg.append(A[i, j])
                    continue
                g = _group_of(a, b)
                if g == "theme" and a.env == env:
                    th.append(A[i, j])
                elif g == "env" and a.env == env:
                    en.append(A[i, j])
        out[env] = {"env": env,
                    "within_theme_across_env": _stats(th),
                    "within_env_across_theme": _stats(en),
                    "diagonal": _stats(dg),
                    "difference": _stats(th)["mean"] - _stats(en)["mean"]}
    return out


def distance_decay(M, cells=None, n_perm=2000, seed=0, unordered=True):
    """Does within-environment transfer fall off with era distance? (6.1)

    6.1's whole scale argument is that the conditioning manifold is ~1-D along
    the era axis, which makes extrapolation from 12 points well-posed. That is a
    testable prediction about *this* matrix: within an environment, transfer
    should decay monotonically with era distance. A flat profile means eras are
    not on a line and the extrapolation splits are not the well-posed problem
    6.1 claims.

    Primary regressor is the nominal-year gap (cells.NOMINAL_YEAR), which
    excludes the style-neutral era 6 (year None, cells.py:59-63); the era-index
    gap |era_i - era_j| is reported too but is not a time axis across era 6.
    `unordered=True` averages M[i,j] with M[j,i] so each unordered pair is one
    observation -- ordered pairs double-count and understate the standard error.
    """
    A, cs = _matrix(M)
    cs = _as_cells(cells) if cells is not None else cs
    n = len(cs)

    def _collect(same_env, use_year):
        xs, ys = [], []
        for i in range(n):
            start = i + 1 if unordered else 0
            for j in range(start, n):
                if i == j:
                    continue
                a, b = cs[i], cs[j]
                if same_env and a.env != b.env:
                    continue
                if not same_env and a.env == b.env:
                    continue
                if use_year:
                    if a.year is None or b.year is None:
                        continue
                    gap = abs(a.year - b.year)
                else:
                    gap = abs(a.era - b.era)
                    if gap == 0:
                        continue
                if unordered:
                    vals = [v for v in (A[i, j], A[j, i]) if np.isfinite(v)]
                    if not vals:
                        continue
                    y = float(np.mean(vals))
                else:
                    if not np.isfinite(A[i, j]):
                        continue
                    y = float(A[i, j])
                xs.append(float(gap))
                ys.append(y)
        return np.array(xs), np.array(ys)

    def _fit(xs, ys, tag):
        fit = ols(xs, ys)
        fit["p_value"] = _ols_perm_p(xs, ys, n_perm, seed)
        fit["regressor"] = tag
        buckets = collections.OrderedDict()
        for g in sorted(set(xs.tolist())):
            buckets["%g" % g] = _stats(ys[xs == g])
        fit["by_gap"] = buckets
        return fit

    xw, yw = _collect(True, True)
    xe, ye = _collect(True, False)
    xa, ya = _collect(False, True)
    within = _fit(xw, yw, "|year_i - year_j| (era 6 excluded)")
    idxfit = _fit(xe, ye, "|era_i - era_j| (all eras; not a time axis at era 6)")
    across = _fit(xa, ya, "|year_i - year_j|, across environments")

    decays = (np.isfinite(within["slope"]) and within["slope"] < 0
              and within["p_value"] < DEFAULT_ALPHA)
    return {
        "within_environment_year_gap": within,
        "within_environment_era_index_gap": idxfit,
        "across_environment_year_gap": across,
        "unordered": bool(unordered),
        "monotone_decay": bool(decays),
        "verdict": ("consistent with the 1-D era manifold (6.1)" if decays else
                    "no reliable decay with era distance -- 6.1's 1-D manifold "
                    "argument is not supported by this matrix"),
        "note": ("slope is in success-rate per year. Compare the within- and "
                 "across-environment slopes: if only the within-environment one "
                 "decays, era distance matters inside a site but style "
                 "correspondence across sites does not track years."),
    }


def verdict(M, cells=None, alpha=DEFAULT_ALPHA, min_effect=DEFAULT_MIN_EFFECT,
            n_perm=DEFAULT_N_PERM, seed=0, min_diag=MIN_DIAG):
    """6.2's decision, on both raw and normalised matrices.

    Returns "organizes_by_theme" (appearance/affordance -- 3 holds and the
    "you're just doing version-ID inference" critique dies), "organizes_by_
    environment" (site function -- 3 is refuted, pivot to a limits-of-pixel-
    conditioning paper per 7/Phase 2), or "inconclusive".
    """
    A, cs = _matrix(M)
    cs = _as_cells(cells) if cells is not None else cs
    Anorm, norm_info = normalized_transfer(A, min_diag=min_diag)

    def _one(mat, tag):
        sc = dissociation_scores(mat, cs)
        try:
            test = paired_permutation_test(mat, cs, n=n_perm, seed=seed)
        except ValueError as exc:
            test = {"p_value": float("nan"), "error": str(exc)}
        try:
            ci = bootstrap_ci(mat, cs, n=n_perm, seed=seed, alpha=alpha)
        except ValueError as exc:
            ci = {"error": str(exc)}
        diff = sc["cell_level"]["difference"]
        p = test.get("p_value", float("nan"))
        if not np.isfinite(diff) or not np.isfinite(p):
            v, why = "inconclusive", "not enough filled entries to decide"
        elif p >= alpha:
            v, why = "inconclusive", "p=%.4f >= alpha=%.3f" % (p, alpha)
        elif abs(diff) < min_effect:
            v, why = ("inconclusive",
                      "|difference|=%.4f below min_effect=%.3f (6.6: near-floor "
                      "effects need care)" % (abs(diff), min_effect))
        elif diff > 0:
            v, why = "organizes_by_theme", "within-theme transfer exceeds within-environment"
        else:
            v, why = "organizes_by_environment", "within-environment transfer exceeds within-theme"
        return {"matrix": tag, "verdict": v, "reason": why,
                "difference_cell_level": diff,
                "difference_pair_level": sc["difference"],
                "within_theme_across_env": sc["within_theme_across_env"]["mean"],
                "within_env_across_theme": sc["within_env_across_theme"]["mean"],
                "test": test, "bootstrap": ci}

    raw = _one(A, "raw")
    nrm = _one(Anorm, "normalized")
    headline_key = "normalized" if norm_info["usable"] else "raw"
    headline = nrm if headline_key == "normalized" else raw
    notes = [HEADLINE_NOTE]
    if not norm_info["usable"]:
        notes.append("FALLBACK to raw: %d/%d columns have an oracle below %.2f, "
                     "so the normalised matrix is mostly NaN."
                     % (norm_info["n_dropped"], norm_info["n_columns"], min_diag))
    if raw["verdict"] != nrm["verdict"]:
        notes.append("raw and normalised DISAGREE (%s vs %s). That is exactly "
                     "the cell-difficulty confound normalisation exists to "
                     "remove; report both and lead with normalised."
                     % (raw["verdict"], nrm["verdict"]))
    outcome = {
        "organizes_by_theme": ("6.2 outcome A: the adapter encodes appearance / "
                               "affordance. 3 holds; the version-ID-inference "
                               "critique dies."),
        "organizes_by_environment": ("6.2 outcome B: the adapter encodes site "
                                     "function. 3 is REFUTED -- Phase 2's "
                                     "redirect applies: pivot to a paper about "
                                     "what pixel conditioning does and does not "
                                     "buy."),
        "inconclusive": ("neither outcome is supported at this power. More "
                         "seeds (6.6) or a completed matrix before deciding."),
    }[headline["verdict"]]
    return {"verdict": headline["verdict"], "headline_matrix": headline_key,
            "plan_outcome": outcome, "alpha": alpha, "min_effect": min_effect,
            "raw": raw, "normalized": nrm, "normalization": norm_info,
            "notes": notes}


def analyze(M, cells=None, n_perm=DEFAULT_N_PERM, seed=0, alpha=DEFAULT_ALPHA,
            min_effect=DEFAULT_MIN_EFFECT, min_diag=MIN_DIAG):
    """Everything 6.2 needs, as one JSON-able dict."""
    A, cs = _matrix(M)
    cs = _as_cells(cells) if cells is not None else cs
    Anorm, norm_info = normalized_transfer(A, min_diag=min_diag)
    rep = collections.OrderedDict()
    rep["schema"] = "adaptercl.transfer.analyze/1"
    rep["cells"] = [c.key for c in cs]
    rep["coverage"] = {"filled": int(np.isfinite(A).sum()),
                       "total": int(A.size),
                       "complete": bool(np.isfinite(A).all())}
    rep["self_transfer"] = {
        "per_cell": dict((c.key, None if not np.isfinite(v) else float(v))
                         for c, v in zip(cs, self_transfer(A))),
        "summary": _stats(self_transfer(A)),
        "note": "the diagonal is the oracle ceiling of 6.6's oracle gap",
    }
    rep["normalization"] = norm_info
    rep["dissociation_raw"] = dissociation_scores(A, cs)
    rep["dissociation_normalized"] = dissociation_scores(Anorm, cs)
    rep["label_permutation_raw"] = label_permutation_test(
        A, cs, n=min(n_perm, 2000), seed=seed)
    rep["by_era"] = by_era(Anorm if norm_info["usable"] else A, cs)
    rep["by_env"] = by_env(Anorm if norm_info["usable"] else A, cs)
    rep["distance_decay"] = distance_decay(
        Anorm if norm_info["usable"] else A, cs, n_perm=min(n_perm, 2000), seed=seed)
    rep["asymmetry"] = asymmetry(A)
    rep["directional_asymmetry"] = directional_asymmetry(A, cs)
    rep["verdict"] = verdict(A, cs, alpha=alpha, min_effect=min_effect,
                             n_perm=n_perm, seed=seed, min_diag=min_diag)
    return rep


def format_report(rep, fh=None):
    """Readable rendering of `analyze`."""
    import sys
    fh = fh or sys.stdout
    v = rep["verdict"]
    cov = rep["coverage"]
    print("=" * 78, file=fh)
    print("adapterCL 6.2 -- functional transfer / dissociation", file=fh)
    print("=" * 78, file=fh)
    print("cells      : %d   filled %d/%d%s"
          % (len(rep["cells"]), cov["filled"], cov["total"],
             "" if cov["complete"] else "   <-- INCOMPLETE"), file=fh)
    st = rep["self_transfer"]["summary"]
    print("diagonal   : mean %.3f  sd %.3f  (oracle ceiling)"
          % (st["mean"], st["sd"]), file=fh)
    ni = rep["normalization"]
    print("normalised : %d/%d columns dropped (oracle < %.2f)%s"
          % (ni["n_dropped"], ni["n_columns"], ni["min_diag"],
             "" if ni["usable"] else "   <-- normalisation unusable"), file=fh)
    print("", file=fh)
    for tag in ("raw", "normalized"):
        d = rep["dissociation_%s" % tag]
        print("%-11s within-theme(across env) %.3f (n=%d) | "
              "within-env(across theme) %.3f (n=%d) | diff %+.3f"
              % (tag + ":", d["within_theme_across_env"]["mean"],
                 d["within_theme_across_env"]["n"],
                 d["within_env_across_theme"]["mean"],
                 d["within_env_across_theme"]["n"], d["difference"]), file=fh)
        sub = v[tag]
        t = sub.get("test", {})
        ci = sub.get("bootstrap", {}).get("difference", {})
        print("            cell-level diff %+.3f  p=%.4f  95%% CI [%s, %s]  -> %s"
              % (sub["difference_cell_level"], t.get("p_value", float("nan")),
                 "%.3f" % ci["lo"] if "lo" in ci else "n/a",
                 "%.3f" % ci["hi"] if "hi" in ci else "n/a",
                 sub["verdict"]), file=fh)
    print("", file=fh)
    print("per era (%s matrix):" % v["headline_matrix"], file=fh)
    print("  %-8s %10s %10s %8s %8s" % ("era", "theme", "env", "diff", "spread"),
          file=fh)
    for k in sorted(rep["by_era"]):
        if k.startswith("_"):
            continue
        e = rep["by_era"][k]
        sp = e["year_spread"]
        print("  %-8s %10.3f %10.3f %+8.3f %8s"
              % (k, e["within_theme_across_env"]["mean"],
                 e["within_env_across_theme"]["mean"], e["difference"],
                 "n/a" if sp is None else "%.1f y" % sp), file=fh)
    print("", file=fh)
    print("per environment:", file=fh)
    for k in sorted(rep["by_env"]):
        e = rep["by_env"][k]
        print("  %-8s theme %.3f  env %.3f  diff %+.3f  diag %.3f"
              % (k, e["within_theme_across_env"]["mean"],
                 e["within_env_across_theme"]["mean"], e["difference"],
                 e["diagonal"]["mean"]), file=fh)
    print("", file=fh)
    dd = rep["distance_decay"]["within_environment_year_gap"]
    print("distance decay (6.1's 1-D manifold): slope %+.5f /yr  r2=%.3f  "
          "p=%.4f  n=%d" % (dd["slope"], dd["r2"], dd["p_value"], dd["n"]), file=fh)
    print("  %s" % rep["distance_decay"]["verdict"], file=fh)
    da = rep["directional_asymmetry"]
    print("direction  : old->new %.3f vs new->old %.3f (diff %+.3f)"
          % (da["older_adapter_on_newer_ui"]["mean"],
             da["newer_adapter_on_older_ui"]["mean"], da["difference"]), file=fh)
    print("", file=fh)
    print("VERDICT: %s   (headline matrix: %s)"
          % (v["verdict"].upper(), v["headline_matrix"]), file=fh)
    print("  %s" % v["plan_outcome"], file=fh)
    for note in v["notes"]:
        print("  - %s" % note, file=fh)
    return rep


# --------------------------------------------------------------------------
# The figure (6.2: "the paper's mechanistic evidence")
# --------------------------------------------------------------------------

def heatmap_markdown(M, cells=None, scale=100.0, normalized=False,
                     min_diag=MIN_DIAG):
    """Markdown table of the matrix, rows = adapter, cols = evaluation cell."""
    A, cs = _matrix(M)
    cs = _as_cells(cells) if cells is not None else cs
    info = None
    if normalized:
        A, info = normalized_transfer(A, min_diag=min_diag)
    n = len(cs)
    lines = []
    lines.append("### Transfer matrix (%s)  rows = adapter, cols = evaluated on"
                 % ("normalized M[i,j]/M[j,j]" if normalized else "raw success rate"))
    lines.append("")
    lines.append("| adapter \\ eval | " + " | ".join(c.key for c in cs) + " |")
    lines.append("|" + "---|" * (n + 1))
    for i, a in enumerate(cs):
        row = ["**%s**" % a.key]
        for j in range(n):
            v = A[i, j]
            if not np.isfinite(v):
                row.append(".")
            elif i == j:
                row.append("_%.0f_" % (v * scale))
            else:
                row.append("%.0f" % (v * scale))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    d = dissociation_scores(A, cs)
    lines.append("Values are success rate x %g; _italic_ = diagonal (oracle). "
                 "within-theme(across env) **%.1f** vs within-env(across theme) "
                 "**%.1f**, difference %+.1f."
                 % (scale, d["within_theme_across_env"]["mean"] * scale,
                    d["within_env_across_theme"]["mean"] * scale,
                    d["difference"] * scale))
    if info and info["dropped_columns"]:
        lines.append("Dropped columns (oracle < %.2f): %s."
                     % (info["min_diag"], ", ".join(map(str, info["dropped_columns"]))))
    lines.append("")
    lines.append("*'theme' == era; %s*" % THEME_NOTE.split(". ", 1)[1])
    return "\n".join(lines)


# --------------------------------------------------------------------------
# The SITE transfer matrix (domain_transfer_timewarp_plan.md 2.3)
#
# Deliberately NOT the Cell path. `dissociation_scores` needs an
# environment x era grid and refuses anything else, and bending it to accept a
# Site matrix would mean inventing an era factor that a Site unit does not have.
# The two matrices also differ in shape: the site matrix's rows are ARMS (which
# include `S_all`, `pv4b` and `frozen` -- rows with no own column) and its
# columns are the four post-hoc task groups, so it is rectangular, its diagonal
# is defined only on the single-site sub-block, and every cell pools several
# (version, seed) eval units.
# --------------------------------------------------------------------------

#: The square sub-block the gap statistic is defined on: rows whose own column
#: exists. `multi` is in it because `S_multi` has a `multi` column; `all`,
#: `pv4b` and `frozen` are rows only.
SITE_BLOCK = cells_mod.SITE_GROUPS

#: Rows that are controls rather than site adapters -- they have no own column,
#: so they never enter the diagonal and never enter the off-diagonal either.
#: Kept as a name so a new control does not silently become a matrix row.
SITE_CONTROL_ROWS = ("all", "pv4b", "pv9b6", "frozen", "pool")

SITE_GAP_NOTE = (
    "The gap is mean(diagonal) - mean(off-diagonal) over the single-site block "
    "only. Two SEs are reported and they answer different questions. The "
    "EPISODE-level SE is what the plan pre-registers (roughly 0.02-0.03 on a "
    "site-cell mean at 22-31 tasks x 3 seeds x 2 versions), and it treats "
    "episodes as independent -- which they are not across rows, because every "
    "row is evaluated on the SAME tasks. The UNIT-level SE is over the "
    "(version, seed) eval units and is the paired quantity; the sign-flip test "
    "on the per-column differences is the actual inference. A gap of 0.05 is "
    "detectable in Wave 1 and a gap of 0.03 is not -- stated before running."
)


def arm_own_site(arm):
    """Which column an arm's own site is, or None for a control row.

    Accepts the arm names the drivers use interchangeably: `S_wiki`, `wiki`,
    `wiki_d1`, `site4b/wiki_d2`. A name that resolves to a control row (or to
    nothing) returns None, and a None row is excluded from BOTH the diagonal and
    the off-diagonal -- counting a control as off-diagonal would deflate the
    off-diagonal mean and manufacture a gap.
    """
    if not arm:
        return None
    tail = str(arm).rstrip("/").split("/")[-1]
    tail = tail[2:] if tail.startswith("S_") else tail
    base = tail.split("_d")[0]
    if base in SITE_CONTROL_ROWS or base not in SITE_BLOCK:
        return None
    return base


def site_matrix(units, rows=None, cols=None, threshold=1.0, own_site=None,
                task_data=None):
    """Build the site transfer matrix from eval units.

    Parameters
    ----------
    units : iterable of dicts, one per EVAL UNIT (one arm x version x seed):
            {"arm": str, "version": int, "seed": int, "records": [record]}
            where `records` are `evalbridge.read_results()["records"]`. One unit
            supplies a whole ROW because every unit runs all 103 test tasks.
    rows  : arm order (default: as encountered, controls last)
    cols  : column order (default cells.SITE_GROUPS)
    own_site : {arm: group|None} override for the diagonal. The default is
            `arm_own_site`, which reads it off the arm name; pass this
            explicitly if an arm is named something the parser cannot know.

    Returns a dict with a `cells` map keyed (row, col) holding both the pooled
    episode-level mean and the per-(version, seed) unit means -- the gap needs
    the first and the sign-flip test needs the second.
    """
    cols = list(cols or cells_mod.SITE_GROUPS)
    gmap = cells_mod.task_site_group_map(task_data=task_data)
    acc = collections.OrderedDict()
    seen_rows, versions, seeds = [], set(), set()
    dropped = collections.Counter()

    for u in units:
        arm = u["arm"]
        if arm not in seen_rows:
            seen_rows.append(arm)
        ver, seed = u.get("version"), u.get("seed")
        versions.add(ver)
        seeds.add(seed)
        for rec in u["records"]:
            g = gmap.get(rec.get("task_id"))
            if g is None or g not in cols:
                dropped[arm] += 1
                continue
            k = (arm, g)
            a = acc.setdefault(k, {"vals": [], "per_unit": collections.OrderedDict()})
            v = 1.0 if float(rec.get("reward") or 0.0) >= threshold else 0.0
            a["vals"].append(v)
            a["per_unit"].setdefault((ver, seed), []).append(v)

    if rows is None:
        site_rows = [r for r in seen_rows if arm_own_site(r) is not None]
        rows = site_rows + [r for r in seen_rows if arm_own_site(r) is None]
    rows = list(rows)

    override = own_site or {}
    own = dict((r, override.get(r, arm_own_site(r))) for r in rows)

    out = {"rows": rows, "cols": cols, "own": own,
           "versions": sorted(v for v in versions if v is not None),
           "seeds": sorted(s for s in seeds if s is not None),
           "cells": collections.OrderedDict(), "dropped": dict(dropped)}
    for r in rows:
        for c in cols:
            a = acc.get((r, c))
            if not a:
                out["cells"][(r, c)] = {"success": float("nan"), "n": 0,
                                        "se": float("nan"), "unit_means": {},
                                        "n_units": 0}
                continue
            vals = np.asarray(a["vals"], dtype=float)
            p = float(vals.mean())
            # Binomial episode-level SE. Not the harness's SEM by accident:
            # rewards here are 0/1 (verified over 768 episodes), so the two
            # agree up to the ddof, and the closed form is stable at n=22.
            se = math.sqrt(max(0.0, p * (1.0 - p)) / vals.size) if vals.size else float("nan")
            um = collections.OrderedDict(
                (k, float(np.mean(v))) for k, v in a["per_unit"].items())
            out["cells"][(r, c)] = {"success": p, "n": int(vals.size), "se": se,
                                    "unit_means": um, "n_units": len(um),
                                    "n_solved": int(vals.sum())}
    return out


def _site_pool(M, keys):
    """Pooled episode-level mean/SE over a set of (row, col) cells."""
    n = sum(M["cells"][k]["n"] for k in keys)
    s = sum(M["cells"][k].get("n_solved", 0) for k in keys)
    if not n:
        return float("nan"), float("nan"), 0
    p = float(s) / n
    return p, math.sqrt(max(0.0, p * (1.0 - p)) / n), n


def site_gap(M, version=None):
    """mean(diagonal) - mean(off-diagonal) on the single-site block.

    `version` restricts to one eval version by recomputing from the unit means
    (the episode-level SE is then unavailable for that slice and comes back as
    the unit-level SE instead, which is the honest thing at 3 units).
    """
    block = [r for r in M["rows"] if M["own"].get(r) is not None
             and M["own"][r] in M["cols"]]
    # Only columns that HAVE an own adapter are in the square block. A column
    # without one has no diagonal entry, so including it would put its whole
    # column in the off-diagonal and deflate that mean.
    block_cols = set(M["own"][r] for r in block)
    diag, off = [], []
    for r in block:
        o = M["own"][r]
        for c in M["cols"]:
            if c not in block_cols:
                continue
            (diag if c == o else off).append((r, c))

    if version is None:
        dm, dse, dn = _site_pool(M, diag)
        om, ose, on = _site_pool(M, off)
        gap = dm - om
        ep_se = math.sqrt(dse ** 2 + ose ** 2)
    else:
        def _slice(keys):
            vs = [v for k in keys
                  for (ver, _s), v in M["cells"][k]["unit_means"].items()
                  if ver == version]
            return _stats(vs)
        d, o = _slice(diag), _slice(off)
        dm, om, dn, on = d["mean"], o["mean"], d["n"], o["n"]
        gap = dm - om
        ep_se = float("nan")

    # Unit-level (paired) SE: per (version, seed) eval unit, the difference of
    # that unit's diagonal mean and its off-diagonal mean. This is the quantity
    # the sign-flip test permutes, and the one that respects the fact that every
    # row is scored on the same tasks.
    per_unit = collections.OrderedDict()
    for keys, tag in ((diag, "diag"), (off, "off")):
        for k in keys:
            for u, v in M["cells"][k]["unit_means"].items():
                if version is not None and u[0] != version:
                    continue
                per_unit.setdefault(u, {"diag": [], "off": []})[tag].append(v)
    unit_diffs = [float(np.mean(d["diag"])) - float(np.mean(d["off"]))
                  for d in per_unit.values() if d["diag"] and d["off"]]
    us = _stats(unit_diffs)
    return {
        "block": block,
        "version": version,
        "gap": gap,
        "diagonal_mean": dm, "off_diagonal_mean": om,
        "n_diagonal_cells": len(diag), "n_off_diagonal_cells": len(off),
        "n_episodes_diagonal": dn, "n_episodes_off_diagonal": on,
        "episode_se": ep_se,
        "unit_level": {"mean": us["mean"], "sd": us["sd"], "n": us["n"],
                       "se": (us["sd"] / math.sqrt(us["n"])
                              if us["n"] > 1 else float("nan"))},
        "gap_over_episode_se": (gap / ep_se) if ep_se and ep_se == ep_se else float("nan"),
        "note": SITE_GAP_NOTE,
    }


def site_column_tests(M, n_perm=DEFAULT_N_PERM, seed=0):
    """Per column: the own-site adapter minus each other row, sign-flipped.

    The paired unit is one (version, seed) eval unit, which is exactly what
    makes this paired -- both rows saw the same tasks in the same version at the
    same seed. Rows with no own column (`S_all`, `pv4b`, `frozen`) ARE tested
    against the own-site adapter, because "does the pooled control match the
    site adapter?" is the plan's first hypothesis row; they are simply never the
    reference themselves.
    """
    out = collections.OrderedDict()
    own_of = M["own"]
    for c in M["cols"]:
        ref = [r for r in M["rows"] if own_of.get(r) == c]
        if not ref:
            continue
        ref = ref[0]
        col = collections.OrderedDict()
        for r in M["rows"]:
            if r == ref:
                continue
            a = M["cells"][(ref, c)]["unit_means"]
            b = M["cells"][(r, c)]["unit_means"]
            shared = [u for u in a if u in b]
            if len(shared) < 2:
                col[r] = {"n_units": len(shared),
                          "skipped": "fewer than 2 paired units"}
                continue
            diffs = [a[u] - b[u] for u in shared]
            t = sign_flip_test(diffs, n_perm=n_perm, seed=seed)
            t["n_units"] = len(shared)
            t["reference"] = ref
            col[r] = t
        out[c] = {"reference": ref, "vs": col}
    return out


def site_column_winners(M):
    """Per column: which row is best, and whether it is that column's own row.

    Reported over the single-site columns (the plan's "2/3 columns" rule) AND
    over all four, because `multi` is a column whose own adapter exists but
    whose reading is the composition test, not the domain test.
    """
    rows_with_own = [r for r in M["rows"] if M["own"].get(r) is not None]
    per_col = collections.OrderedDict()
    for c in M["cols"]:
        vals = [(M["cells"][(r, c)]["success"], r) for r in M["rows"]
                if M["cells"][(r, c)]["n"]]
        vals = [(v, r) for v, r in vals if np.isfinite(v)]
        if not vals:
            continue
        best = max(vals)
        ref = [r for r in rows_with_own if M["own"][r] == c]
        per_col[c] = {"best_row": best[1], "best": best[0],
                      "own_row": ref[0] if ref else None,
                      "own": (M["cells"][(ref[0], c)]["success"] if ref
                              else float("nan")),
                      "own_is_best": bool(ref) and best[1] == ref[0]}
    single = [c for c in per_col if c in cells_mod.ENVIRONMENTS]
    return {
        "per_column": per_col,
        "single_site_columns": len(single),
        "single_site_winners": sum(1 for c in single
                                   if per_col[c]["own_is_best"]),
        "all_columns": len(per_col),
        "all_winners": sum(1 for c in per_col if per_col[c]["own_is_best"]),
    }


def site_analyze(M, n_perm=DEFAULT_N_PERM, seed=0):
    """gap (pooled + per version), column tests, winners -- one dict."""
    rep = collections.OrderedDict()
    rep["rows"] = list(M["rows"])
    rep["cols"] = list(M["cols"])
    rep["own"] = dict(M["own"])
    rep["versions"] = list(M["versions"])
    rep["seeds"] = list(M["seeds"])
    rep["gap"] = site_gap(M)
    rep["gap_by_version"] = collections.OrderedDict(
        (v, site_gap(M, version=v)) for v in M["versions"])
    rep["columns"] = site_column_tests(M, n_perm=n_perm, seed=seed)
    rep["winners"] = site_column_winners(M)
    g = rep["gap"]
    se = g["episode_se"]
    if not (se == se) or se <= 0:
        rep["verdict"] = "insufficient data"
    elif g["gap"] >= 2 * se and rep["winners"]["single_site_winners"] >= 2:
        rep["verdict"] = ("site knowledge exists: gap %.4f >= 2 SE (%.4f) and "
                          "%d/%d single-site columns are won by their own "
                          "adapter" % (g["gap"], se,
                                       rep["winners"]["single_site_winners"],
                                       rep["winners"]["single_site_columns"]))
    elif abs(g["gap"]) < se:
        rep["verdict"] = ("adapters are protocol on the site axis too: |gap| "
                          "%.4f < 1 SE (%.4f). The negative IS the result -- "
                          "stop after writing it up (plan 2.4)."
                          % (g["gap"], se))
    else:
        rep["verdict"] = ("indeterminate: gap %.4f is between 1 and 2 SE "
                          "(%.4f); the plan's decision rule does not fire."
                          % (g["gap"], se))
    return rep


def format_site_report(rep, fh=None):
    """Readable version of `site_analyze`."""
    import sys as _sys
    fh = fh or _sys.stdout
    print("=== site transfer matrix ===", file=fh)
    print("rows=%s" % ", ".join(rep["rows"]), file=fh)
    print("cols=%s   versions=%s   seeds=%s"
          % (", ".join(rep["cols"]), rep["versions"], rep["seeds"]), file=fh)
    print("", file=fh)
    g = rep["gap"]
    print("gap (diag - off, single-site block %s):" % ", ".join(g["block"]),
          file=fh)
    print("  diagonal      %.4f over %d cell(s) / %d episode(s)"
          % (g["diagonal_mean"], g["n_diagonal_cells"],
             g["n_episodes_diagonal"]), file=fh)
    print("  off-diagonal  %.4f over %d cell(s) / %d episode(s)"
          % (g["off_diagonal_mean"], g["n_off_diagonal_cells"],
             g["n_episodes_off_diagonal"]), file=fh)
    print("  GAP           %+.4f   episode SE %.4f (%.2f SE)   unit SE %s (n=%d)"
          % (g["gap"], g["episode_se"], g["gap_over_episode_se"],
             ("%.4f" % g["unit_level"]["se"]
              if g["unit_level"]["se"] == g["unit_level"]["se"] else "--"),
             g["unit_level"]["n"]), file=fh)
    print("  per version (means over CELLS, equal weight per cell -- the "
          "pooled row above weights by task count):", file=fh)
    for v, gv in rep["gap_by_version"].items():
        print("    v%-2s gap %+.4f (diag %.4f / off %.4f)"
              % (v, gv["gap"], gv["diagonal_mean"], gv["off_diagonal_mean"]),
              file=fh)
    print("", file=fh)
    w = rep["winners"]
    print("column winners: %d/%d single-site, %d/%d overall"
          % (w["single_site_winners"], w["single_site_columns"],
             w["all_winners"], w["all_columns"]), file=fh)
    for c, r in w["per_column"].items():
        print("  %-6s best=%-10s %.4f   own=%-10s %s"
              % (c, r["best_row"], r["best"], r["own_row"] or "-",
                 ("%.4f%s" % (r["own"], "  <-- own is best"
                              if r["own_is_best"] else ""))
                 if r["own"] == r["own"] else "--"), file=fh)
    print("", file=fh)
    print("paired sign-flip, own-site adapter minus each other row:", file=fh)
    for c, blk in rep["columns"].items():
        print("  column %s (reference %s)" % (c, blk["reference"]), file=fh)
        for r, t in blk["vs"].items():
            if "skipped" in t:
                print("    vs %-12s skipped: %s" % (r, t["skipped"]), file=fh)
                continue
            print("    vs %-12s diff %+.4f  p=%.4f  dz=%s  n_units=%d"
                  % (r, t["observed"], t["p_value"],
                     ("%.2f" % t["effect_size_dz"])
                     if np.isfinite(t["effect_size_dz"]) else "inf",
                     t["n_units"]), file=fh)
    print("", file=fh)
    print("VERDICT: %s" % rep["verdict"], file=fh)
    print("", file=fh)
    print(SITE_GAP_NOTE, file=fh)
    return rep


def site_heatmap_markdown(M, scale=100.0):
    """Markdown table of the site matrix; rows = arm, cols = task group."""
    lines = ["### Site transfer matrix  rows = adapter, cols = task group",
             "",
             "| adapter \\ group | " + " | ".join(M["cols"]) + " | n units |",
             "|" + "---|" * (len(M["cols"]) + 2)]
    for r in M["rows"]:
        row = ["**%s**" % r]
        nu = 0
        for c in M["cols"]:
            cell = M["cells"][(r, c)]
            nu = max(nu, cell["n_units"])
            if not cell["n"] or not np.isfinite(cell["success"]):
                row.append(".")
            elif M["own"].get(r) == c:
                row.append("_%.1f_" % (cell["success"] * scale))
            else:
                row.append("%.1f" % (cell["success"] * scale))
        lines.append("| " + " | ".join(row + [str(nu)]) + " |")
    g = site_gap(M)
    lines += ["",
              "Values are success rate x %g; _italic_ = the row's own site "
              "(the diagonal). Gap = %+.1f (diag %.1f - off %.1f), episode SE "
              "%.1f." % (scale, g["gap"] * scale, g["diagonal_mean"] * scale,
                         g["off_diagonal_mean"] * scale,
                         g["episode_se"] * scale),
              "",
              "*Rows with no own column (%s) are controls: they enter neither "
              "the diagonal nor the off-diagonal.*"
              % ", ".join(r for r in M["rows"] if M["own"].get(r) is None)]
    return "\n".join(lines)


#: 9-stop viridis approximation. Perceptually uniform-ish and legible in
#: greyscale, which matters for a print figure.
_VIRIDIS = ((68, 1, 84), (72, 40, 120), (62, 74, 137), (49, 104, 142),
            (38, 130, 142), (31, 158, 137), (53, 183, 121), (109, 205, 89),
            (253, 231, 37))


def _colormap(t):
    """t in [0,1] -> (r, g, b)."""
    if not np.isfinite(t):
        return (226, 226, 226)
    t = min(1.0, max(0.0, float(t)))
    k = t * (len(_VIRIDIS) - 1)
    i = int(math.floor(k))
    if i >= len(_VIRIDIS) - 1:
        return _VIRIDIS[-1]
    f = k - i
    a, b = _VIRIDIS[i], _VIRIDIS[i + 1]
    return tuple(int(round(a[c] + f * (b[c] - a[c]))) for c in range(3))


def _luminance(rgb):
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]


def _esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def heatmap_svg(M, cells=None, path=None, normalized=False, vmin=None,
                vmax=None, cell_px=34, title=None, min_diag=MIN_DIAG,
                annotate=True, scale=100.0):
    """Self-contained SVG heatmap -- no matplotlib, no external fonts.

    Legibility choices, because 6.2 calls this "the paper's mechanistic
    evidence": rows and columns are grouped by environment with heavy
    separators between the three blocks, the diagonal is outlined, every cell
    carries its value, annotation colour flips on background luminance, and the
    caption states the two means and the verdict so the figure stands alone.
    """
    A, cs = _matrix(M)
    cs = _as_cells(cells) if cells is not None else cs
    info = None
    if normalized:
        A, info = normalized_transfer(A, min_diag=min_diag)
    n = len(cs)
    finite = A[np.isfinite(A)]
    if vmin is None:
        vmin = float(finite.min()) if finite.size else 0.0
    if vmax is None:
        vmax = float(finite.max()) if finite.size else 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-6

    # -- layout ---------------------------------------------------------
    left, top, right, bottom = 118, 118, 26, 118
    grid_w = n * cell_px
    width = left + grid_w + right
    height = top + grid_w + bottom

    # environment blocks, assuming cells are grouped by environment
    blocks, start = [], 0
    for i in range(1, n + 1):
        if i == n or cs[i].env != cs[start].env:
            blocks.append((cs[start].env, start, i))
            start = i
    grouped = len(blocks) == len(set(b[0] for b in blocks))

    out = []
    out.append('<svg xmlns="http://www.w3.org/2000/svg" '
               'xmlns:xlink="http://www.w3.org/1999/xlink" '
               'width="%d" height="%d" viewBox="0 0 %d %d">'
               % (width, height, width, height))
    out.append('<style>'
               'text{font-family:"DejaVu Sans",Helvetica,Arial,sans-serif}'
               '.lbl{font-size:11px;fill:#222}'
               '.val{font-size:%dpx}'
               '.blk{font-size:12px;font-weight:bold;fill:#111}'
               '.ttl{font-size:15px;font-weight:bold;fill:#111}'
               '.cap{font-size:11px;fill:#333}'
               '</style>' % (10 if cell_px >= 30 else 8))
    out.append('<rect width="%d" height="%d" fill="#ffffff"/>' % (width, height))

    ttl = title or ("Functional transfer%s: adapter (row) evaluated on cell "
                    "(column)" % (" [normalised by oracle]" if normalized else ""))
    out.append('<text class="ttl" x="%d" y="26">%s</text>' % (14, _esc(ttl)))

    # -- cells ----------------------------------------------------------
    for i in range(n):
        for j in range(n):
            x = left + j * cell_px
            y = top + i * cell_px
            v = A[i, j]
            t = (v - vmin) / (vmax - vmin) if np.isfinite(v) else float("nan")
            rgb = _colormap(t)
            out.append('<rect x="%d" y="%d" width="%d" height="%d" '
                       'fill="rgb(%d,%d,%d)" stroke="#ffffff" stroke-width="0.6"/>'
                       % (x, y, cell_px, cell_px, rgb[0], rgb[1], rgb[2]))
            if annotate:
                if np.isfinite(v):
                    txt = "%.0f" % (v * scale)
                    fill = "#ffffff" if _luminance(rgb) < 140 else "#111111"
                else:
                    txt, fill = "·", "#888888"
                out.append('<text class="val" x="%.1f" y="%.1f" fill="%s" '
                           'text-anchor="middle">%s</text>'
                           % (x + cell_px / 2.0, y + cell_px / 2.0 + 3.5,
                              fill, _esc(txt)))
            if i == j:
                out.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" '
                           'fill="none" stroke="#d62728" stroke-width="1.8"/>'
                           % (x + 0.9, y + 0.9, cell_px - 1.8, cell_px - 1.8))

    # -- labels ---------------------------------------------------------
    for i, c in enumerate(cs):
        y = top + i * cell_px + cell_px / 2.0 + 4
        out.append('<text class="lbl" x="%.1f" y="%.1f" text-anchor="end">%s</text>'
                   % (left - 8, y, _esc(c.key)))
    for j, c in enumerate(cs):
        x = left + j * cell_px + cell_px / 2.0
        out.append('<text class="lbl" x="%.1f" y="%.1f" text-anchor="start" '
                   'transform="rotate(-60 %.1f %.1f)">%s</text>'
                   % (x, top - 8, x, top - 8, _esc(c.key)))

    # -- environment blocks + separators --------------------------------
    if grouped:
        for env, s, e in blocks:
            ys, ye = top + s * cell_px, top + e * cell_px
            xs, xe = left + s * cell_px, left + e * cell_px
            out.append('<rect x="%d" y="%d" width="6" height="%d" fill="#444"/>'
                       % (left - 74, ys, ye - ys))
            out.append('<text class="blk" x="%d" y="%.1f" text-anchor="middle" '
                       'transform="rotate(-90 %d %.1f)">%s</text>'
                       % (left - 84, (ys + ye) / 2.0, left - 84,
                          (ys + ye) / 2.0, _esc(env.upper())))
            out.append('<rect x="%d" y="%d" width="%d" height="6" fill="#444"/>'
                       % (xs, top - 68, xe - xs))
            out.append('<text class="blk" x="%.1f" y="%d" text-anchor="middle">'
                       '%s</text>' % ((xs + xe) / 2.0, top - 76, _esc(env.upper())))
            for pos in (s, e):
                gx = left + pos * cell_px
                gy = top + pos * cell_px
                out.append('<line x1="%d" y1="%d" x2="%d" y2="%d" '
                           'stroke="#111" stroke-width="2"/>'
                           % (gx, top, gx, top + grid_w))
                out.append('<line x1="%d" y1="%d" x2="%d" y2="%d" '
                           'stroke="#111" stroke-width="2"/>'
                           % (left, gy, left + grid_w, gy))
    out.append('<rect x="%d" y="%d" width="%d" height="%d" fill="none" '
               'stroke="#111" stroke-width="2"/>' % (left, top, grid_w, grid_w))

    # -- colour scale ---------------------------------------------------
    bar_y = top + grid_w + 30
    bar_w = min(300, grid_w)
    out.append('<defs><linearGradient id="cs" x1="0" y1="0" x2="1" y2="0">')
    for k in range(len(_VIRIDIS)):
        r, g, b = _VIRIDIS[k]
        out.append('<stop offset="%.3f" stop-color="rgb(%d,%d,%d)"/>'
                   % (k / float(len(_VIRIDIS) - 1), r, g, b))
    out.append('</linearGradient></defs>')
    out.append('<rect x="%d" y="%d" width="%d" height="12" fill="url(#cs)" '
               'stroke="#111" stroke-width="0.8"/>' % (left, bar_y, bar_w))
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        val = vmin + frac * (vmax - vmin)
        out.append('<text class="cap" x="%.1f" y="%d" text-anchor="middle">%.0f</text>'
                   % (left + frac * bar_w, bar_y + 26, val * scale))
    out.append('<text class="cap" x="%d" y="%d">%s (x%g)</text>'
               % (left + bar_w + 12, bar_y + 11,
                  _esc("normalised transfer" if normalized else "success rate"),
                  scale))

    # -- caption --------------------------------------------------------
    d = dissociation_scores(A, cs)
    cap = ("within-theme / across-environment %.1f (n=%d)   vs   "
           "within-environment / across-theme %.1f (n=%d)   difference %+.1f"
           % (d["within_theme_across_env"]["mean"] * scale,
              d["within_theme_across_env"]["n"],
              d["within_env_across_theme"]["mean"] * scale,
              d["within_env_across_theme"]["n"], d["difference"] * scale))
    out.append('<text class="cap" x="14" y="%d">%s</text>'
               % (bar_y + 52, _esc(cap)))
    out.append('<text class="cap" x="14" y="%d">%s</text>'
               % (bar_y + 68,
                  _esc("red outline = diagonal (oracle). 'theme' == era. "
                       "%s" % ("dropped columns: %s" % ", ".join(map(str, info["dropped_columns"]))
                               if info and info["dropped_columns"] else
                               "dot = pair not yet evaluated."))))
    out.append('</svg>')
    svg = "\n".join(out)
    if path:
        d_ = os.path.dirname(os.path.abspath(path))
        if d_ and not os.path.isdir(d_):
            os.makedirs(d_)
        with open(path, "w") as fh:
            fh.write(svg)
    return svg


# --------------------------------------------------------------------------
# Weight space -- the guardrail (4.4 / 6.2)
# --------------------------------------------------------------------------

_WEIGHT_SPACE_REFUSAL = (
    "weight_space_distance() refuses to compare INDEPENDENTLY TRAINED adapters.\n"
    "\n"
    "adapterCL.md 4.4 / 6.2: a LoRA is identified only up to an invertible R --\n"
    "  BA = (BR)(R^-1 A) -- plus permutation and scaling of the rank axis. Two\n"
    "  functionally near-identical adapters therefore sit in unrelated corners of\n"
    "  weight space, and any distance between per-cell adapters measures\n"
    "  symmetry-group arbitrariness rather than function. This is the same defect\n"
    "  that makes reconstruction training fail, and 9's risk register lists\n"
    "  'weight-space adapter comparison used by mistake' as a *silent* invalidity.\n"
    "\n"
    "Do this instead:\n"
    "  * independently trained per-cell adapters -> functional_transfer() /\n"
    "    TransferMatrix / from_eval_dirs(). Cross-apply and measure success.\n"
    "  * hypernetwork-generated adapters -> pass shared_parametrization=True.\n"
    "    One generator, one parametrisation, so distances are meaningful\n"
    "    (6.2, last paragraph). generated_adapter_distances() is the batch form.\n"
    "  * if you really want a weight-space number for independently trained\n"
    "    adapters, the only symmetry-invariant one is on the PRODUCT dW = B @ A\n"
    "    (invariant under A -> R^-1 A, B -> B R): metric='delta_w_cosine' with\n"
    "    allow_delta_w=True. It is still not what 6.2 asks for -- init and\n"
    "    optimisation-path variance dominate at rank 16 -- so it is opt-in and\n"
    "    the result carries a caveat.\n"
)


def _to_numpy(t):
    """torch tensor / numpy array / nested list -> float64 numpy array."""
    if isinstance(t, np.ndarray):
        return t.astype(np.float64)
    if hasattr(t, "detach"):          # torch, possibly bf16
        return t.detach().to("cpu").float().numpy().astype(np.float64)
    return np.asarray(t, dtype=np.float64)


def _adapter_factors(obj):
    """Coerce an adapter dir / factor dict into {rel_name: (A, B)} numpy."""
    if isinstance(obj, str):
        from . import materialize
        factors, _cfg, _meta = materialize.read_adapter(obj)
    elif isinstance(obj, dict):
        factors = obj
    else:
        raise TypeError("expected an adapter directory or a {rel_name: (A, B)} "
                        "dict, got %r" % type(obj))
    return dict((k, (_to_numpy(v[0]), _to_numpy(v[1]))) for k, v in factors.items())


def _common_sites(a, b):
    common = sorted(set(a) & set(b))
    if not common:
        raise ValueError("the two adapters share no injection sites (%d vs %d); "
                         "were they trained on different target sets?"
                         % (len(a), len(b)))
    return common


def _flat_ab(factors, sites):
    parts = []
    for s in sites:
        A, B = factors[s]
        parts.append(A.ravel())
        parts.append(B.ravel())
    return np.concatenate(parts)


def _flat_delta_w(factors, sites):
    parts = []
    for s in sites:
        A, B = factors[s]
        parts.append((B.dot(A)).ravel())
    return np.concatenate(parts)


def _cos_frob(u, v):
    nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    cos = float(u.dot(v) / (nu * nv)) if nu > 0 and nv > 0 else float("nan")
    return {"cosine_distance": 1.0 - cos if np.isfinite(cos) else float("nan"),
            "cosine_similarity": cos,
            "frobenius_distance": float(np.linalg.norm(u - v)),
            "norm_a": nu, "norm_b": nv}


def weight_space_distance(adapter_a, adapter_b, shared_parametrization=False,
                          metric="cosine", allow_delta_w=False):
    """Distance between two adapters in weight space. GUARDED ON PURPOSE.

    Raises ValueError unless `shared_parametrization=True` (hypernetwork-
    generated adapters from one generator) or `allow_delta_w=True` with a
    `delta_w_*` metric. See `_WEIGHT_SPACE_REFUSAL` for the full reasoning.
    """
    delta_w = metric.startswith("delta_w")
    if not shared_parametrization and not (delta_w and allow_delta_w):
        raise ValueError(_WEIGHT_SPACE_REFUSAL)
    fa = _adapter_factors(adapter_a)
    fb = _adapter_factors(adapter_b)
    sites = _common_sites(fa, fb)
    if delta_w:
        u, v = _flat_delta_w(fa, sites), _flat_delta_w(fb, sites)
    else:
        u, v = _flat_ab(fa, sites), _flat_ab(fb, sites)
    res = _cos_frob(u, v)
    res["metric"] = metric
    res["n_sites"] = len(sites)
    res["dim"] = int(u.size)
    res["shared_parametrization"] = bool(shared_parametrization)
    res["caveats"] = []
    if not shared_parametrization:
        res["caveats"].append(
            "independently trained adapters compared on dW = B@A: symmetry-"
            "invariant, but init and optimisation-path variance still dominate. "
            "Not a substitute for 6.2's functional transfer matrix.")
    if metric in ("cosine", "frobenius"):
        res["distance"] = res["cosine_distance" if metric == "cosine"
                              else "frobenius_distance"]
    else:
        res["distance"] = res["cosine_distance" if metric.endswith("cosine")
                              else "frobenius_distance"]
    return res


def _check_shared(dirs, require_shared):
    """Refuse a batch of adapters that look independently trained."""
    if not require_shared:
        return {"checked": False}
    from . import materialize
    verdicts = []
    for d in dirs:
        meta = {}
        if isinstance(d, str):
            mp = os.path.join(d, "adaptercl_meta.json")
            if os.path.exists(mp):
                with open(mp) as fh:
                    meta = json.load(fh)
        shared = bool(meta.get("shared_parametrization")
                      or meta.get("generator") or meta.get("hypernet")
                      or meta.get("run"))
        verdicts.append((d if isinstance(d, str) else "<dict>", shared, meta))
    bad = [v[0] for v in verdicts if not v[1]]
    if bad:
        raise ValueError(
            _WEIGHT_SPACE_REFUSAL +
            "\n%d of %d adapters carry no generator provenance in "
            "adaptercl_meta.json (no 'generator'/'hypernet'/'run'/"
            "'shared_parametrization' key):\n  %s\n"
            "If these really did come from one generator, set "
            "shared_parametrization=true in their meta (materialize."
            "write_adapter(meta=...)) or pass require_shared=False."
            % (len(bad), len(verdicts), "\n  ".join(bad[:6])))
    return {"checked": True, "n": len(verdicts)}


def average_linkage(D, labels=None):
    """UPGMA agglomerative clustering. numpy only -- no scipy.

    Returns {"merges": [(a, b, distance, size), ...], "order": leaf order,
    "labels": ...}. Cluster ids >= n are the merged clusters, scipy-style, so
    the output can be handed to a scipy dendrogram later without translation.
    """
    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    if D.shape[0] != D.shape[1]:
        raise ValueError("average_linkage needs a square distance matrix, got %r"
                         % (D.shape,))
    labels = list(labels) if labels is not None else [str(i) for i in range(n)]
    active = list(range(n))
    size = dict((i, 1) for i in range(n))
    members = dict((i, [i]) for i in range(n))
    dist = {}
    for i in range(n):
        for j in range(i + 1, n):
            dist[(i, j)] = float(D[i, j])

    def _d(a, b):
        return dist[(a, b)] if a < b else dist[(b, a)]

    merges = []
    nxt = n
    while len(active) > 1:
        best, ba, bb = None, None, None
        for x in range(len(active)):
            for y in range(x + 1, len(active)):
                a, b = active[x], active[y]
                v = _d(a, b)
                if best is None or v < best:
                    best, ba, bb = v, a, b
        sa, sb = size[ba], size[bb]
        for c in active:
            if c in (ba, bb):
                continue
            nd = (sa * _d(ba, c) + sb * _d(bb, c)) / float(sa + sb)
            key = (min(nxt, c), max(nxt, c))
            dist[key] = nd
        active = [c for c in active if c not in (ba, bb)]
        active.append(nxt)
        size[nxt] = sa + sb
        members[nxt] = members[ba] + members[bb]
        merges.append((ba, bb, float(best), sa + sb))
        nxt += 1
    order = members[nxt - 1] if merges else [0]
    return {"merges": merges, "order": order, "labels": labels,
            "members": dict((k, v) for k, v in members.items() if k >= n)}


def cut_clusters(link, n_leaves, k):
    """Cut a linkage into k clusters; returns a leaf -> cluster-id array."""
    if k < 1 or k > n_leaves:
        raise ValueError("k must be in 1..%d, got %d" % (n_leaves, k))
    assign = list(range(n_leaves))
    members = dict((i, [i]) for i in range(n_leaves))
    nxt = n_leaves
    n_clusters = n_leaves
    for a, b, _dst, _sz in link["merges"]:
        if n_clusters <= k:
            break
        members[nxt] = members[a] + members[b]
        for leaf in members[nxt]:
            assign[leaf] = nxt
        nxt += 1
        n_clusters -= 1
    remap, out = {}, []
    for a in assign:
        if a not in remap:
            remap[a] = len(remap)
        out.append(remap[a])
    return np.array(out, dtype=int)


def cluster_purity(assign, cells):
    """How well a clustering lines up with environment vs era labels.

    The qualitative half of 6.2's cross-check: if generated adapters cluster by
    era the generator learned appearance, if by environment it learned site
    identity. Purity = sum over clusters of the modal label count / n.
    """
    cs = _as_cells(cells)
    assign = np.asarray(assign, dtype=int)

    def _purity(get):
        tot = 0
        for c in sorted(set(assign.tolist())):
            idx = np.where(assign == c)[0]
            counts = collections.Counter(get(cs[i]) for i in idx)
            tot += counts.most_common(1)[0][1]
        return tot / float(len(cs))

    pe, pv = _purity(lambda c: c.env), _purity(lambda c: c.era)
    return {"purity_by_environment": pe, "purity_by_era": pv,
            "prefers": ("era" if pv > pe else
                        "environment" if pe > pv else "tie"),
            "n_clusters": int(len(set(assign.tolist())))}


def generated_adapter_distances(adapter_dirs, keys=None, metric="cosine",
                                require_shared=True, cells=None, k=None):
    """Pairwise weight-space structure over HYPERNETWORK-GENERATED adapters.

    The one valid use of weight space (6.2, last paragraph): all of these come
    from one generator, so they share a parametrisation and their distances mean
    something. `require_shared` checks each adapter's adaptercl_meta.json for
    generator provenance and refuses otherwise -- the silent-invalidity risk in
    9 is exactly "used weight-space comparison by mistake".

    Returns cosine and Frobenius distance matrices, per-adapter norms, the
    effective rank of the set (diversity, 6.7), an average-linkage clustering,
    and -- when `cells` is given -- how that clustering lines up with the
    environment/era grid.
    """
    prov = _check_shared(adapter_dirs, require_shared)
    fac = [_adapter_factors(d) for d in adapter_dirs]
    if len(fac) < 2:
        raise ValueError("need >=2 adapters, got %d" % len(fac))
    sites = sorted(set(fac[0]))
    for f in fac[1:]:
        sites = sorted(set(sites) & set(f))
    if not sites:
        raise ValueError("the adapters share no injection sites -- they were "
                         "not produced by one generator over one target set")
    flat = "delta_w" in metric
    X = np.stack([(_flat_delta_w(f, sites) if flat else _flat_ab(f, sites))
                  for f in fac])
    n = X.shape[0]
    keys = list(keys) if keys is not None else [
        os.path.basename(str(d).rstrip("/")) for d in adapter_dirs]

    norms = np.linalg.norm(X, axis=1)
    Dc = np.zeros((n, n))
    Df = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            r = _cos_frob(X[i], X[j])
            Dc[i, j] = Dc[j, i] = r["cosine_distance"]
            Df[i, j] = Df[j, i] = r["frobenius_distance"]

    Xc = X - X.mean(axis=0, keepdims=True)
    svals = np.linalg.svd(Xc, full_matrices=False, compute_uv=False)
    er = spectrum_effective_rank(svals)
    svals_raw = np.linalg.svd(X, full_matrices=False, compute_uv=False)

    D = Dc if metric.endswith("cosine") else Df
    link = average_linkage(D, labels=keys)
    out = {"keys": keys, "n": n, "n_sites": len(sites), "dim": int(X.shape[1]),
           "metric": metric, "flattened": "delta_w" if flat else "concat(A,B)",
           "provenance_check": prov,
           "cosine_distance": Dc.tolist(), "frobenius_distance": Df.tolist(),
           "norms": norms.tolist(),
           "distance_stats": _stats([Dc[i, j] for i in range(n)
                                     for j in range(i + 1, n)]),
           "effective_rank_centered": er,
           "effective_rank_raw": spectrum_effective_rank(svals_raw),
           "max_effective_rank": float(min(n - 1, X.shape[1])),
           "singular_values": svals.tolist(),
           "linkage": {"merges": link["merges"], "order": link["order"],
                       "labels": link["labels"]}}
    if cells is not None:
        cs = _as_cells(cells)
        kk = k or len(set(c.env for c in cs))
        out["clustering"] = {}
        for kk_ in sorted(set([kk, len(set(c.era for c in cs))])):
            if 1 < kk_ <= n:
                assign = cut_clusters(link, n, kk_)
                out["clustering"]["k=%d" % kk_] = dict(
                    cluster_purity(assign, cs), assignment=assign.tolist())
    return out


def agreement(functional_M, weight_M, cells=None, n_perm=2000, seed=0,
              symmetrize=True, threshold=-0.3):
    """Do the generated adapters' weight-space distances agree with 6.2's
    functional transfer matrix?

    SIGN CONVENTION, and it matters: `weight_M` is a *distance* and
    `functional_M` is a *success rate*, so agreement means a NEGATIVE Spearman
    rho -- adapters that sit far apart in weight space should transfer to each
    other badly. Positive rho is the red flag 6.2 says is "worth chasing".

    The functional matrix is asymmetric and the distance matrix is not, so the
    functional one is symmetrised ((M + M.T)/2) and only the upper triangle is
    used: 153 unordered pairs at 18 cells. Using all 306 ordered off-diagonal
    entries would duplicate every observation and halve the effective p-value.
    """
    F, cs = _matrix(functional_M)
    W, _ = _matrix(weight_M)
    if F.shape != W.shape:
        raise ValueError("functional matrix is %r but weight matrix is %r"
                         % (F.shape, W.shape))
    if symmetrize:
        F = (F + F.T) / 2.0
    n = F.shape[0]
    iu = np.triu_indices(n, 1)
    x, y = W[iu], F[iu]
    res = spearman(x, y, n_perm=n_perm, seed=seed)
    rho = res["rho"]
    agrees = bool(np.isfinite(rho) and rho <= threshold
                  and res["p_value"] < DEFAULT_ALPHA)
    res.update({
        "agrees": agrees,
        "threshold": threshold,
        "symmetrized": bool(symmetrize),
        "sign_convention": ("negative rho == agreement (distance up => transfer "
                            "down)"),
        "interpretation": (
            "weight-space structure agrees with functional transfer -- 6.2's "
            "good sign that the hypernetwork learned the right organisation"
            if agrees else
            "weight-space structure does NOT track functional transfer (rho=%.3f, "
            "p=%.4f). 6.2 calls disagreement a red flag worth chasing: either the "
            "generator's variation is in directions that do not change behaviour, "
            "or the functional matrix is noise-dominated." % (rho, res["p_value"]))
    })
    return res


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------

def _synth_matrix(cells, mode="theme", difficulty=None, era_decay=0.0,
                  noise=0.0, seed=0, weights=None):
    """A matrix with a known answer, for --selftest.

    mode='theme'  : transfer driven by era match  -> organizes_by_theme
    mode='env'    : transfer driven by env match  -> organizes_by_environment
    mode='null'   : neither                       -> inconclusive

    `difficulty` scales whole columns and is keyed by environment *or* by cell
    key. The distinction is the point:

      * environment-level difficulty cannot flip the sign of the raw per-cell
        paired difference. With three environments the "other environments"
        average is symmetric, so the mean over source cells works out to
        (w_era - w_env) * sum(difficulty) / 3 -- the sign is the truth's and the
        difficulty only shrinks the magnitude (toward `min_effect`, which near
        6.6's floor is exactly how a real effect gets called inconclusive).
      * cell-level difficulty *can* flip it, because a source cell's two
        within-theme columns and five within-environment columns are then drawn
        from different difficulty distributions.

    Normalising by the diagonal removes both.
    """
    cs = _as_cells(cells)
    n = len(cs)
    rs = np.random.RandomState(seed)
    w_era, w_env = weights or {"theme": (0.55, 0.10), "env": (0.10, 0.55),
                               "null": (0.0, 0.0)}[mode]

    def _d(cell):
        if not difficulty:
            return 1.0
        if cell.key in difficulty:
            return float(difficulty[cell.key])
        return float(difficulty[cell.env])

    M = np.zeros((n, n))
    for i, a in enumerate(cs):
        for j, b in enumerate(cs):
            base = 0.15
            if a.era == b.era:
                base += w_era
            if a.env == b.env:
                base += w_env
            if era_decay and a.env == b.env and a.year and b.year:
                base -= era_decay * abs(a.year - b.year)
            v = base * _d(b)
            if noise:
                v += rs.normal(0.0, noise)
            M[i, j] = min(1.0, max(0.0, v))
    return M


def _no_constants(tok):
    raise AssertionError("non-strict JSON token %r survived json_safe()" % tok)


def selftest(verbose=True):
    import sys
    fh = sys.stdout
    cs = list(cells_mod.ALL_CELLS)
    ok = []

    def _say(msg):
        if verbose:
            print("  %s" % msg, file=fh)
        ok.append(msg)

    print("transfer.py selftest", file=fh)

    # 1. TransferMatrix mechanics ---------------------------------------
    tm = TransferMatrix(cs)
    assert tm.n == 18 and not tm.is_complete()
    assert len(tm.missing_pairs()) == 324
    tm.set("wiki_e1", cs[5], 0.5, n_episodes=7)
    assert abs(tm.get(0, 5) - 0.5) < 1e-12
    assert tm.provenance(0, 5)["n_episodes"] == 7
    assert tm.n_filled() == 1
    import tempfile
    tmpd = tempfile.mkdtemp(prefix="adaptercl_selftest_")
    p = os.path.join(tmpd, "m.json")
    tm.save(p)
    tm2 = TransferMatrix.load(p)
    assert tm2.keys == tm.keys
    assert abs(tm2.get(0, 5) - 0.5) < 1e-12
    assert not np.isfinite(tm2.get(0, 0))
    assert tm2.provenance(0, 5)["n_episodes"] == 7
    _say("TransferMatrix: 18 cells, 324 pairs, NaN<->null round trip, resume OK")

    # 2. build_pairs / plan ---------------------------------------------
    assert len(build_pairs(cs)) == 324
    assert len(build_pairs(cs, include_diagonal=False)) == 306
    pl = plan(cs, n_repeats=1, tasks_per_cell=20)
    assert pl["plan_reference_324"] and pl["episodes"] == 324 * 20
    _say("plan(): 324 pairs == 6.1's 18x18; 6480 episodes at 20 tasks/cell")

    # 3. theme-organised matrix (no confound) ----------------------------
    Mt = _synth_matrix(cs, "theme")
    d = dissociation_scores(Mt, cs)
    assert d["within_theme_across_env"]["n"] == 36, d["within_theme_across_env"]
    assert d["within_env_across_theme"]["n"] == 90
    assert d["difference"] > 0
    v = verdict(Mt, cs, n_perm=2000)
    assert v["verdict"] == "organizes_by_theme", v
    assert v["raw"]["verdict"] == "organizes_by_theme"
    _say("theme-organised synthetic -> %s (diff %+.3f, p=%.4f)"
         % (v["verdict"], v["normalized"]["difference_cell_level"],
            v["normalized"]["test"]["p_value"]))

    # 4. environment-organised matrix ------------------------------------
    Me = _synth_matrix(cs, "env")
    ve = verdict(Me, cs, n_perm=2000)
    assert ve["verdict"] == "organizes_by_environment", ve
    assert "REFUTED" in ve["plan_outcome"]
    _say("env-organised synthetic   -> %s (diff %+.3f)"
         % (ve["verdict"], ve["normalized"]["difference_cell_level"]))

    # 5. null matrix ------------------------------------------------------
    Mn = _synth_matrix(cs, "null", noise=0.02, seed=3)
    vn = verdict(Mn, cs, n_perm=2000)
    assert vn["verdict"] == "inconclusive", vn
    _say("null synthetic            -> %s" % vn["verdict"])

    # 6. env-level difficulty: sign survives, magnitude does not ----------
    hard = {"wiki": 1.0, "news": 0.45, "shop": 0.30}
    Mc = _synth_matrix(cs, "theme", difficulty=hard)
    vc = verdict(Mc, cs, n_perm=2000)
    assert vc["headline_matrix"] == "normalized"
    assert vc["verdict"] == "organizes_by_theme", vc
    raw_diff = vc["raw"]["difference_cell_level"]
    nrm_diff = vc["normalized"]["difference_cell_level"]
    assert nrm_diff > raw_diff, (raw_diff, nrm_diff)
    _say("env-level difficulty: raw cell-level diff %+.3f vs normalised %+.3f "
         "(sign survives, effect is shrunk %.0f%%)"
         % (raw_diff, nrm_diff, 100.0 * (1 - raw_diff / nrm_diff)))

    # 6b. cell-level difficulty + a weak true effect: raw goes inconclusive
    #     while the normalised matrix recovers the truth. This is the concrete
    #     case behind HEADLINE_NOTE.
    rs6 = np.random.RandomState(1)
    cell_hard = dict((c.key, float(rs6.uniform(0.10, 1.0))) for c in cs)
    Mw = _synth_matrix(cs, "theme", difficulty=cell_hard, weights=(0.06, 0.02))
    vw = verdict(Mw, cs, n_perm=2000)
    assert vw["raw"]["verdict"] == "inconclusive", vw["raw"]
    assert vw["verdict"] == "organizes_by_theme", vw
    assert any("DISAGREE" in n for n in vw["notes"])
    _say("cell-level difficulty, weak truth: raw %+.4f p=%.3f -> %s BUT "
         "normalised %+.4f -> %s (headline is normalised, disagreement noted)"
         % (vw["raw"]["difference_cell_level"], vw["raw"]["test"]["p_value"],
            vw["raw"]["verdict"], vw["normalized"]["difference_cell_level"],
            vw["normalized"]["verdict"]))

    # 7. self / normalised transfer --------------------------------------
    diag = self_transfer(Mc)
    assert abs(diag[0] - 0.80 * hard["wiki"]) < 1e-9
    Nm, info = normalized_transfer(Mc)
    assert info["n_dropped"] == 0 and info["usable"]
    assert abs(Nm[0, 0] - 1.0) < 1e-9
    tmz = TransferMatrix(cs)
    tmz.M = np.array(Mc, copy=True)
    tmz.M[:, 3] = 0.0
    _, info2 = normalized_transfer(tmz)
    assert info2["n_dropped"] == 1 and info2["dropped_columns"] == ["wiki_e4"], info2
    _say("normalized_transfer: diag==1, near-zero oracle column dropped by key")

    # 8. distance decay ---------------------------------------------------
    Md = _synth_matrix(cs, "theme", era_decay=0.012)
    dd = distance_decay(Md, cs, n_perm=500)
    assert dd["within_environment_year_gap"]["slope"] < 0, dd
    assert dd["monotone_decay"], dd
    flat = distance_decay(_synth_matrix(cs, "theme"), cs, n_perm=500)
    assert not flat["monotone_decay"]
    _say("distance_decay: slope %+.5f/yr p=%.4f on a decaying matrix; flat "
         "matrix -> monotone_decay=False"
         % (dd["within_environment_year_gap"]["slope"],
            dd["within_environment_year_gap"]["p_value"]))

    # 9. by_era / by_env / asymmetry --------------------------------------
    be = by_era(Mt, cs)
    assert all(be["era_%d" % e]["difference"] > 0 for e in cells_mod.ERAS)
    bv = by_env(Mt, cs)
    assert set(bv) == set(cells_mod.ENVIRONMENTS)
    da = directional_asymmetry(_synth_matrix(cs, "theme", era_decay=0.01), cs)
    assert np.isfinite(da["difference"])
    _say("by_era/by_env/directional_asymmetry populated")

    # 10. permutation + bootstrap ----------------------------------------
    pt = paired_permutation_test(Mt, cs, n=2000)
    assert pt["p_value"] < 0.01 and pt["n_units"] == 18
    assert abs(pt["exact_null_space"] - 262144.0) < 1.0
    lp = label_permutation_test(Mt, cs, n=300)
    assert lp["p_value"] < 0.05, lp
    ci = bootstrap_ci(Mt, cs, n=2000)
    assert ci["excludes_zero"] and ci["difference"]["lo"] > 0
    _say("paired sign-flip p=%.4f (2^18 null space); label-perm p=%.4f; "
         "bootstrap CI [%.3f, %.3f]"
         % (pt["p_value"], lp["p_value"], ci["difference"]["lo"],
            ci["difference"]["hi"]))

    # 11. figures ----------------------------------------------------------
    md = heatmap_markdown(Mt, cs)
    assert "wiki_e1" in md and "shop_e6" in md and md.count("\n") > 18
    svg_path = os.path.join(tmpd, "heat.svg")
    svg = heatmap_svg(Mt, cs, path=svg_path, title="selftest")
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
    for probe_str in ("wiki_e1", "shop_e6", "WIKI", "SHOP", "linearGradient"):
        assert probe_str in svg, probe_str
    assert os.path.getsize(svg_path) > 4000
    nan_svg = heatmap_svg(tm, path=None)
    assert "·" in nan_svg
    import xml.etree.ElementTree as ET
    ET.fromstring(svg.encode("utf-8"))          # must be well-formed XML
    ET.fromstring(nan_svg.encode("utf-8"))
    _say("heatmap_markdown + heatmap_svg (%d bytes, well-formed XML, env blocks "
         "+ legend + NaN dots)" % os.path.getsize(svg_path))

    # 12. weight-space guardrail ------------------------------------------
    rsw = np.random.RandomState(0)
    A0, B0 = rsw.normal(size=(2, 4)), rsw.normal(size=(4, 2))
    fa = {"layers.0.mlp.up_proj": (A0, B0)}
    fb = {"layers.0.mlp.up_proj": (A0 * 2.0, B0 * 2.0)}    # same direction
    # the R-symmetry 4.4 warns about: A -> R^-1 A, B -> B R leaves B@A alone
    R = np.array([[0.0, 1.0], [1.0, 0.0]])                 # a rank permutation
    fc = {"layers.0.mlp.up_proj": (np.linalg.inv(R).dot(A0), B0.dot(R))}
    raised = False
    try:
        weight_space_distance(fa, fb)
    except ValueError as exc:
        raised = "4.4" in str(exc) and "functional_transfer" in str(exc)
    assert raised, "weight_space_distance did not refuse independent adapters"
    r = weight_space_distance(fa, fb, shared_parametrization=True)
    assert abs(r["cosine_distance"]) < 1e-9      # same direction, different scale
    # the whole point of 4.4, demonstrated: an exact functional twin looks far
    # away in (A, B) space but identical on the product dW = B @ A
    r_sym = weight_space_distance(fa, fc, shared_parametrization=True)
    r_dw = weight_space_distance(fa, fc, metric="delta_w_cosine",
                                 allow_delta_w=True)
    assert r_sym["cosine_distance"] > 0.5, r_sym
    assert abs(r_dw["cosine_distance"]) < 1e-9, r_dw
    assert r_dw["caveats"]
    _say("weight_space_distance refuses independent adapters (names 4.4 and "
         "functional_transfer); a functional twin under A->R^-1 A, B->BR reads "
         "cos-dist %.2f in (A,B) space but %.0e on dW -- 4.4 demonstrated"
         % (r_sym["cosine_distance"], abs(r_dw["cosine_distance"])))

    # 13. linkage / purity / spearman / agreement ---------------------------
    pts = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1],
                    [5.0, 5.0], [5.1, 5.0], [5.0, 5.1]])
    D = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    link = average_linkage(D)
    assert len(link["merges"]) == 5
    assign = cut_clusters(link, 6, 2)
    assert len(set(assign[:3].tolist())) == 1 and len(set(assign[3:].tolist())) == 1
    assert assign[0] != assign[3]
    sp = spearman([1, 2, 3, 4, 5], [5, 4, 3, 2, 1])
    assert abs(sp["rho"] + 1.0) < 1e-12
    sp2 = spearman([1, 2, 3, 4], [1, 2, 2, 4])
    assert abs(sp2["rho"] - 0.9486832980505138) < 1e-9
    W = 1.0 - (Mt + Mt.T) / 2.0
    ag = agreement(Mt, W, n_perm=300)
    assert ag["rho"] < -0.9 and ag["agrees"], ag
    pur = cluster_purity([0, 0, 0, 1, 1, 1], cs[:6])
    assert pur["prefers"] in ("era", "environment", "tie")

    # 13b. generated_adapter_distances on synthetic "generated" adapters:
    #      one shared mean plus an era-structured component -> should cluster by
    #      era, which is what a working hypernetwork would look like (6.2).
    rsg = np.random.RandomState(7)
    era_dir = dict((e, rsg.normal(size=(3, 5))) for e in cells_mod.ERAS)
    gen = []
    for c in cs:
        A = np.ones((3, 5)) * 2.0 + era_dir[c.era] + rsg.normal(0, 0.01, (3, 5))
        B = np.ones((5, 3)) * 2.0 + rsg.normal(0, 0.01, (5, 3))
        gen.append({"layers.0.mlp.up_proj": (A, B)})
    gd = generated_adapter_distances(gen, keys=[c.key for c in cs],
                                     require_shared=False, cells=cs)
    assert gd["n"] == 18 and gd["dim"] == 30
    assert gd["clustering"]["k=6"]["prefers"] == "era", gd["clustering"]
    assert gd["clustering"]["k=6"]["purity_by_era"] > 0.9
    assert 4.0 < gd["effective_rank_centered"] < 7.0, gd["effective_rank_centered"]
    # and the guardrail fires when provenance is missing
    try:
        generated_adapter_distances(gen[:3], require_shared=True)
        raise AssertionError("require_shared did not refuse")
    except ValueError as exc:
        assert "4.4" in str(exc) and "adaptercl_meta.json" in str(exc)
    _say("generated_adapter_distances: era-structured generated set clusters by "
         "era (purity %.2f), effective rank %.1f of max %.0f; require_shared "
         "refuses adapters without generator provenance"
         % (gd["clustering"]["k=6"]["purity_by_era"],
            gd["effective_rank_centered"], gd["max_effective_rank"]))
    _say("average_linkage recovers 2 clusters; spearman exact on ties; "
         "agreement rho=%.3f (negative == agrees)" % ag["rho"])

    # 14. fill_matrix + the evalbridge boundary ----------------------------
    tmf = TransferMatrix(cs[:4], path=os.path.join(tmpd, "resume.json"))
    calls = []

    def _fake_eval(eval_cell, train_cell):
        calls.append((train_cell.key, eval_cell.key))
        return 0.9 if train_cell.era == eval_cell.era else 0.1

    fill_matrix(tmf, _fake_eval, verbose=False, save_every=3)
    assert len(calls) == 16 and tmf.is_complete()
    tmf2 = TransferMatrix.load(tmf.path)
    fill_matrix(tmf2, _fake_eval, verbose=False)      # resume: nothing to do
    assert len(calls) == 16, "fill_matrix re-ran already-filled pairs"

    # the record/aggregate shapes evalbridge can hand back
    recs = [{"task_id": 1, "seed": 0, "reward": 1.0},
            {"task_id": 2, "seed": 0, "reward": 0.0},
            {"task_id": 3, "seed": 0, "reward": 0.5}]
    assert _success_rate_from_results({"records": recs, "aggregate": {}})[0] == 1 / 3.0
    assert _success_rate_from_results(recs, threshold=0.5)[0] == 2 / 3.0
    assert _success_rate_from_results((recs, None))[0] == 1 / 3.0
    assert _success_rate_from_results({"success_rate": 0.25, "n": 8}) == (0.25, 8)
    try:
        _success_rate_from_results({"records": [], "aggregate":
                                    {"success_rate": 0.25}}, threshold=0.5)
        raise AssertionError("threshold mismatch should raise")
    except ValueError as exc:
        assert "threshold=1.0" in str(exc)
    try:
        from . import evalbridge
        real = evalbridge.aggregate(
            [{"reward": 1.0, "n_steps": 3, "err": None},
             {"reward": 0.0, "n_steps": 5, "err": None}])
        assert _success_rate_from_results({"records": [], "aggregate": real}) == (0.5, 2)
        assert _success_rate_from_results(
            evalbridge.read_results(os.path.join(tmpd, "does_not_exist"))
        ) is None
    except ImportError:
        _say("evalbridge not importable yet; from_eval_dirs would raise an "
             "actionable ImportError")
    except ValueError as exc:
        assert "empty or half-written" in str(exc), exc
        _say("evalbridge present: its real aggregate() and a missing-run "
             "read_results() are both handled at the boundary")
    _say("fill_matrix fills 16 pairs, saves incrementally and resumes without "
         "re-evaluating; all four evalbridge return shapes decode")

    # 14b. from_eval_dirs end to end over a real AgentLab-shaped tree -------
    try:
        from . import evalbridge          # noqa: F811
        root = os.path.join(tmpd, "sweep")
        quad = cs[:2] + [c for c in cs if c.env == "news"][:2]
        for a in quad:
            for b in quad:
                d = os.path.join(root, pair_dir_name(a, b), "study")
                os.makedirs(d)
                hit = 1.0 if a.era == b.era else 0.0
                with open(os.path.join(d, "result_df_trial_1.csv"), "w") as cfh:
                    cfh.write("exp_dir,env.task_name,env.task_seed,cum_reward,"
                              "n_steps,err_msg\n")
                    for t in (11, 12, 13, 14):
                        cfh.write("%s,timewarp.%d,0,%.1f,4,\n"
                                  % (os.path.join(d, "timewarp.%d_0" % t), t,
                                     hit if t % 2 else 0.0))
        tme = from_eval_dirs(root, quad)
        assert tme.is_complete(), tme.missing_pairs()
        assert abs(tme.get(quad[0], quad[0]) - 0.5) < 1e-9      # 2 of 4 solved
        assert abs(tme.get(quad[0], quad[1])) < 1e-9            # era mismatch
        assert tme.provenance(quad[0], quad[0])["n_episodes"] == 4
        # a pair that was never run stays NaN and is reported, not invented
        import shutil as _sh
        _sh.rmtree(os.path.join(root, pair_dir_name(quad[0], quad[1])))
        tmp2 = from_eval_dirs(root, quad)
        assert not tmp2.is_complete()
        assert tmp2.meta["from_eval_dirs"]["missing"] == [
            pair_dir_name(quad[0], quad[1])]
        _say("from_eval_dirs reads a real <train>__on__<eval>/ tree through "
             "evalbridge (4x4 filled, missing pair left NaN and listed)")
    except ImportError:
        pass

    # 15. analyze() end to end ---------------------------------------------
    rep = analyze(Mc, cs, n_perm=1000)
    txt = json.dumps(json_safe(rep), sort_keys=True)
    assert "NaN" not in txt and "Infinity" not in txt   # strict JSON
    assert json.loads(txt, parse_constant=_no_constants)
    assert rep["verdict"]["verdict"] == "organizes_by_theme"
    # a half-filled matrix must still analyse rather than explode
    partial = np.array(Mc, copy=True)
    partial[2, :] = np.nan
    rep2 = analyze(partial, cs, n_perm=500)
    assert not rep2["coverage"]["complete"]
    json.dumps(json_safe(rep2), sort_keys=True)
    _say("analyze() -> strict JSON (no NaN/Infinity tokens), and survives a "
         "half-filled matrix")

    import shutil
    shutil.rmtree(tmpd, ignore_errors=True)
    print("OK -- %d checks" % len(ok), file=fh)
    return True


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _main():
    import argparse
    import sys

    ap = argparse.ArgumentParser(
        description="adapterCL 6.2: functional transfer matrix + dissociation")
    ap.add_argument("--selftest", action="store_true",
                    help="run the synthetic known-answer checks and exit")
    ap.add_argument("--plan", action="store_true",
                    help="print the sweep cost before spending it")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--tasks-per-cell", type=int, default=None)
    ap.add_argument("--seconds-per-episode", type=float, default=120.0)
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--matrix", default=None,
                    help="transfer matrix JSON to analyse")
    ap.add_argument("--from-eval-dirs", default=None,
                    help="scan <root>/<train>__on__<eval>/ and build the matrix")
    ap.add_argument("--svg", default=None, help="write the heatmap SVG here")
    ap.add_argument("--md", action="store_true", help="print a markdown heatmap")
    ap.add_argument("--json", default=None, help="write the analysis JSON here")
    ap.add_argument("--normalized", action="store_true",
                    help="draw the normalised matrix instead of raw")
    ap.add_argument("--n-perm", type=int, default=DEFAULT_N_PERM)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return 0

    if args.plan or not (args.matrix or args.from_eval_dirs):
        format_plan(plan(None, n_repeats=args.repeats,
                         tasks_per_cell=args.tasks_per_cell,
                         seconds_per_episode=args.seconds_per_episode,
                         n_parallel=args.parallel))
        if not (args.matrix or args.from_eval_dirs):
            return 0

    if args.from_eval_dirs:
        tm = from_eval_dirs(args.from_eval_dirs, verbose=True)
        if args.matrix:
            tm.save(args.matrix)
    else:
        tm = TransferMatrix.load(args.matrix)

    print("", file=sys.stdout)
    rep = analyze(tm, n_perm=args.n_perm, seed=args.seed)
    format_report(rep)
    if args.md:
        print("")
        print(heatmap_markdown(tm, normalized=args.normalized))
    if args.svg:
        heatmap_svg(tm, path=args.svg, normalized=args.normalized)
        print("\nwrote %s" % args.svg)
    if args.json:
        paths.ensure_out_dirs()
        with open(args.json, "w") as fh:
            json.dump(json_safe(rep), fh, indent=2, sort_keys=True)
        print("wrote %s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


# ===========================================================================
# Leave-one-site-out on one era (LOSO_HANDOFF.md)
# ===========================================================================
#
# The site matrix above asks "does an adapter trained on ONE site beat the
# others on that site?". The LOSO design asks the domain-GENERALIZATION
# question: "given every site EXCEPT x, how much of x do you get for free?".
# Three quantities, all read against the matched-dose, same-era reference
# `M_all` so that dose and era are held fixed:
#
#   penalty(x)   = M_all[x] - L_x[x]              the OOD cost on the held-out column
#   retention(x) = L_x[x] / M_all[x]              the fraction of in-dist success kept
#   DiD(x)       = penalty on x  -  (M_all - L_x) on the SEEN columns
#
# The DiD is the load-bearing one: L_x and M_all differ in composition AND in
# what the 385 samples were spent on, so a generic offset (L_x is simply a
# worse or better adapter) shows up on every column; only the part of the
# held-out drop that the seen columns do NOT share is a domain effect.
#
# `L_multi` is the same row with `multi` held out: its DiD is the single-site
# -> multi-site composition test.

import fractions as _fractions

LOSO_ARMS = cells_mod.LOSO_ARMS
LOSO_REF_ARM = cells_mod.LOSO_REF_ARM
LOSO_SINGLE_ARMS = tuple(a for a in LOSO_ARMS if cells_mod.loso_held_out(a)
                         in cells_mod.ENVIRONMENTS)

LOSO_NOTE = (
    "Read every LOSO number as a DIFFERENCE against M_all at the same dose "
    "and era; the DiD (held-out delta minus seen-columns delta) is the domain "
    "effect, the raw held-out delta is domain + whatever the corpus swap did "
    "to the adapter in general. Per-column cells are 22-31 tasks x 3 seeds: "
    "no single-cell claims; the pooled rows carry the decision.")


def _pool_cells(M, keys):
    """Pooled (mean, binomial SE, n, n_solved) over a set of (row, col) cells."""
    n = sum(M["cells"][k]["n"] for k in keys if k in M["cells"])
    s = sum(M["cells"][k].get("n_solved", 0) for k in keys if k in M["cells"])
    if not n:
        return float("nan"), float("nan"), 0, 0
    p = float(s) / n
    return p, math.sqrt(max(0.0, p * (1.0 - p)) / n), n, s


def _paired_unit_diffs(M, keys_a, keys_b):
    """Per (version, seed) unit: mean over keys_a minus mean over keys_b.

    Pairs a LOSO arm's unit at seed s with the reference's unit at the SAME
    seed -- the seed indexes the corpus draw on both sides, so this is the
    paired quantity a sign-flip test may permute.
    """
    per = collections.OrderedDict()
    for keys, tag in ((keys_a, "a"), (keys_b, "b")):
        for k in keys:
            if k not in M["cells"]:
                continue
            for u, v in M["cells"][k]["unit_means"].items():
                per.setdefault(u, {"a": [], "b": []})[tag].append(v)
    diffs = collections.OrderedDict()
    for u, d in per.items():
        if d["a"] and d["b"]:
            diffs[u] = float(np.mean(d["a"])) - float(np.mean(d["b"]))
    return diffs


def _sign_flip_or_skip(diffs, n_perm, seed):
    vals = list(diffs.values()) if isinstance(diffs, dict) else list(diffs)
    if len(vals) < 2:
        return {"n_units": len(vals), "skipped": "fewer than 2 paired units",
                "observed": (float(vals[0]) if vals else float("nan"))}
    t = sign_flip_test(vals, n_perm=n_perm, seed=seed)
    t["n_units"] = len(vals)
    if len(vals) <= 3:
        t["note"] = ("%d paired units: the smallest attainable p is %.3f, so "
                     "this p is descriptive, not decisive; the episode-level "
                     "SE carries the precision" % (len(vals),
                                                   1.0 / 2 ** (len(vals) - 1)))
    return t


def _paired_with_per_unit(diffs, n_perm, seed):
    """Sign-flip result plus the per-(version, seed) differences it was run on,
    keyed 'v6_s1', so a report can show which seed carried an effect."""
    t = _sign_flip_or_skip(diffs, n_perm, seed)
    t["per_unit"] = collections.OrderedDict(
        ("v%s_s%s" % u, float(v)) for u, v in diffs.items())
    return t


def loso_row(M, arm, ref_arm=LOSO_REF_ARM, extra_refs=("pv4b", "frozen"),
             n_perm=DEFAULT_N_PERM, seed=0):
    """One LOSO arm against the reference: held-out column, seen columns, DiD."""
    held = cells_mod.loso_held_out(arm)
    seen = sorted(cells_mod.loso_seen_sites(arm), key=cells_mod.ENVIRONMENTS.index)
    if held is None:
        raise ValueError("%r is not a LOSO arm (%s)" % (arm, list(LOSO_ARMS)))
    if arm not in M["rows"] or ref_arm not in M["rows"]:
        return None

    ho_L = [(arm, held)]
    ho_R = [(ref_arm, held)]
    sn_L = [(arm, c) for c in seen]
    sn_R = [(ref_arm, c) for c in seen]

    pL, seL, nL, _ = _pool_cells(M, ho_L)
    pR, seR, nR, _ = _pool_cells(M, ho_R)
    qL, sqL, mL, _ = _pool_cells(M, sn_L)
    qR, sqR, mR, _ = _pool_cells(M, sn_R)

    penalty = pR - pL
    pen_se = math.sqrt(seR ** 2 + seL ** 2)
    seen_delta = qR - qL
    seen_se = math.sqrt(sqR ** 2 + sqL ** 2)
    did = penalty - seen_delta
    did_se = math.sqrt(pen_se ** 2 + seen_se ** 2)
    retention = (pL / pR) if pR and pR == pR else float("nan")

    # Paired, per seed: the held-out delta, the seen delta and the DiD. The
    # LOSO unit at seed s is paired with the reference unit at seed s -- the
    # seed indexes the corpus draw on both sides.
    d_ho = _paired_unit_diffs(M, ho_R, ho_L)
    d_sn = _paired_unit_diffs(M, sn_R, sn_L)
    d_did = collections.OrderedDict(
        (u, d_ho[u] - d_sn[u]) for u in d_ho if u in d_sn)

    extras = collections.OrderedDict()
    for r in extra_refs:
        if r in M["rows"]:
            e, ese, en, _ = _pool_cells(M, [(r, held)])
            extras[r] = {"success": e, "se": ese, "n": en}

    return {
        "arm": arm, "held_out": held, "seen": seen, "reference": ref_arm,
        "corpus": LOSO_ARMS[arm],
        "held_out_col": {"loso": pL, "loso_se": seL, "n": nL,
                         "ref": pR, "ref_se": seR, "n_ref": nR,
                         "penalty": penalty, "penalty_se": pen_se,
                         "retention": retention,
                         "paired": _paired_with_per_unit(d_ho, n_perm, seed)},
        "seen_cols": {"loso": qL, "loso_se": sqL, "n": mL,
                      "ref": qR, "ref_se": sqR, "n_ref": mR,
                      "delta": seen_delta, "delta_se": seen_se,
                      "paired": _sign_flip_or_skip(d_sn, n_perm, seed)},
        "did": {"value": did, "se": did_se,
                "over_se": (did / did_se) if did_se else float("nan"),
                "paired": _paired_with_per_unit(d_did, n_perm, seed)},
        "extra_refs": extras,
    }


def loso_table(units, ref_arm=LOSO_REF_ARM, extra_refs=("pv4b", "frozen"),
               threshold=1.0, n_perm=DEFAULT_N_PERM, seed=0, task_data=None):
    """The LOSO table: one row per LOSO arm present, plus the pooled rows.

    `units` are eval units as for `site_matrix` (one per arm x version x seed).
    Pooling over the three single-site hold-outs weights each column by its
    task count on BOTH sides (the same tasks, the same seeds), so the pooled
    penalty is a paired quantity even though it is computed from pooled cells.
    """
    M = site_matrix(units, threshold=threshold, task_data=task_data)
    rows = collections.OrderedDict()
    for arm in LOSO_ARMS:
        r = loso_row(M, arm, ref_arm=ref_arm, extra_refs=extra_refs,
                     n_perm=n_perm, seed=seed)
        if r is not None:
            rows[arm] = r

    pooled = None
    singles = [a for a in LOSO_SINGLE_ARMS if a in rows]
    if singles and ref_arm in M["rows"]:
        ho_L = [(a, cells_mod.loso_held_out(a)) for a in singles]
        ho_R = [(ref_arm, cells_mod.loso_held_out(a)) for a in singles]
        sn_L = [(a, c) for a in singles for c in rows[a]["seen"]]
        sn_R = [(ref_arm, c) for a in singles for c in rows[a]["seen"]]
        pL, seL, nL, _ = _pool_cells(M, ho_L)
        pR, seR, nR, _ = _pool_cells(M, ho_R)
        qL, sqL, mL, _ = _pool_cells(M, sn_L)
        qR, sqR, mR, _ = _pool_cells(M, sn_R)
        penalty, pen_se = pR - pL, math.sqrt(seR ** 2 + seL ** 2)
        sdelta, s_se = qR - qL, math.sqrt(sqR ** 2 + sqL ** 2)
        did, did_se = penalty - sdelta, math.sqrt(pen_se ** 2 + s_se ** 2)
        # Paired over (arm, seed): each LOSO arm's unit against the reference
        # unit at the same seed -- up to 9 pairs.
        d_did = collections.OrderedDict()
        for a in singles:
            for u, v in rows[a]["did"]["paired"].get("per_unit", {}).items():
                d_did["%s_%s" % (a, u)] = v
        pooled = {
            "arms": singles,
            "held_out": {"loso": pL, "ref": pR, "penalty": penalty,
                         "penalty_se": pen_se, "n": nL,
                         "retention": (pL / pR) if pR else float("nan")},
            "seen": {"loso": qL, "ref": qR, "delta": sdelta,
                     "delta_se": s_se, "n": mL},
            "did": {"value": did, "se": did_se,
                    "over_se": (did / did_se) if did_se else float("nan"),
                    "paired": _sign_flip_or_skip(d_did, n_perm, seed)},
            "detectable_at_2se": 2 * did_se,
        }

    return {"rows": rows, "pooled": pooled, "reference": ref_arm,
            "versions": M["versions"], "seeds": M["seeds"],
            "matrix": M, "note": LOSO_NOTE}


# ---------------------------------------------------------------------------
# The OOD gradient: unseen fraction of a task's sites -> penalty
# ---------------------------------------------------------------------------

def _frac_key(f):
    return str(_fractions.Fraction(f).limit_denominator(12))


def ood_gradient(units, ref_arm=LOSO_REF_ARM, arms=None, threshold=1.0,
                 n_perm=DEFAULT_N_PERM, seed=0, task_data=None):
    """Success as a function of how much of a task the adapter never saw.

    For each single-site LOSO arm a with seen set S_a, every test task t gets
    f = |sites(t) - S_a| / |sites(t)|: 0 for seen single sites and fully-seen
    multi tasks, 1/3 and 1/2 for mixed multi tasks, 1 for the held-out site.
    Records are pooled across arms into bins by f. The reference is re-binned
    PER ARM with that arm's S_a, so the reference bin holds exactly the same
    (task, seed) multiset as the LOSO bin -- the comparison is paired by
    construction, and column difficulty cancels.

    Also returned: the per-(arm, subgroup) breakdown that the bins are made of
    (subgroup = single site or `canonical_site_key` of a multi task), and a
    Spearman test of f against the per-(arm, seed, bin) delta, which is the
    monotonicity check ("does the penalty grow with the unseen fraction?").
    """
    arms = list(arms or LOSO_SINGLE_ARMS)
    tsm = cells_mod.task_sites_map(task_data=task_data)

    by_arm = collections.defaultdict(list)
    for u in units:
        by_arm[u["arm"]].append(u)
    have = [a for a in arms if by_arm.get(a)]
    if not have or not by_arm.get(ref_arm):
        return {"bins": collections.OrderedDict(), "arms": have,
                "reference": ref_arm, "subgroups": collections.OrderedDict(),
                "monotone": None, "spearman": None,
                "reason": "need >= 1 single-site LOSO arm and the reference"}

    # bins[f] = {"loso": [0/1...], "ref": [...], "per_unit": {(arm, ver, seed): {"loso": [...], "ref": [...]}}}
    bins = collections.OrderedDict()
    sub = collections.OrderedDict()
    ref_by_unit = collections.defaultdict(list)
    for u in by_arm[ref_arm]:
        ref_by_unit[(u.get("version"), u.get("seed"))].extend(u["records"])

    def _hit(rec):
        return 1.0 if float(rec.get("reward") or 0.0) >= threshold else 0.0

    for a in have:
        seen = cells_mod.loso_seen_sites(a)
        for u in by_arm[a]:
            key_u = (u.get("version"), u.get("seed"))
            refs = ref_by_unit.get(key_u, [])
            ref_by_task = collections.defaultdict(list)
            for r in refs:
                ref_by_task[r.get("task_id")].append(_hit(r))
            for r in u["records"]:
                ts = tsm.get(r.get("task_id"))
                if not ts:
                    continue
                f = cells_mod.unseen_fraction(ts, seen)
                fk = _frac_key(f)
                b = bins.setdefault(fk, {"f": float(f), "loso": [], "ref": [],
                                         "per_unit": collections.OrderedDict()})
                b["loso"].append(_hit(r))
                rv = ref_by_task.get(r.get("task_id"))
                pu = b["per_unit"].setdefault((a,) + key_u, {"loso": [], "ref": []})
                pu["loso"].append(_hit(r))
                if rv:
                    b["ref"].extend(rv[:1])   # one record per task per unit
                    pu["ref"].extend(rv[:1])
                g = (cells_mod.multi_subgroup(ts) if len(ts) >= 2
                     else sorted(ts)[0])
                s = sub.setdefault((a, g), {"arm": a, "subgroup": g, "f": float(f),
                                            "loso": [], "ref": [], "tasks": set()})
                s["loso"].append(_hit(r))
                s["tasks"].add(r.get("task_id"))
                if rv:
                    s["ref"].extend(rv[:1])

    out_bins = collections.OrderedDict()
    xs, ys = [], []
    for fk in sorted(bins, key=lambda k: bins[k]["f"]):
        b = bins[fk]
        L = np.asarray(b["loso"], dtype=float)
        R = np.asarray(b["ref"], dtype=float)
        pL = float(L.mean()) if L.size else float("nan")
        pR = float(R.mean()) if R.size else float("nan")
        seL = math.sqrt(max(0.0, pL * (1 - pL)) / L.size) if L.size else float("nan")
        seR = math.sqrt(max(0.0, pR * (1 - pR)) / R.size) if R.size else float("nan")
        diffs = collections.OrderedDict()
        for k, d in b["per_unit"].items():
            if d["loso"] and d["ref"]:
                diffs[k] = float(np.mean(d["ref"])) - float(np.mean(d["loso"]))
                xs.append(b["f"])
                ys.append(diffs[k])
        out_bins[fk] = {
            "f": b["f"], "n": int(L.size), "n_ref": int(R.size),
            "loso": pL, "loso_se": seL, "ref": pR, "ref_se": seR,
            "penalty": pR - pL, "penalty_se": math.sqrt(seL ** 2 + seR ** 2),
            "retention": (pL / pR) if pR else float("nan"),
            "n_unit_pairs": len(diffs),
            "paired": _sign_flip_or_skip(diffs, n_perm, seed),
        }

    pens = [out_bins[k]["penalty"] for k in out_bins]
    # Non-decreasing penalty in the unseen fraction.
    monotone = (all(pens[i] <= pens[i + 1] + 1e-12 for i in range(len(pens) - 1))
                if len(pens) >= 2 else None)
    sp = spearman(xs, ys, n_perm=n_perm, seed=seed) if len(xs) >= 3 else None

    subgroups = collections.OrderedDict()
    for (a, g), s in sorted(sub.items(), key=lambda kv: (kv[1]["f"], kv[0])):
        L = np.asarray(s["loso"], dtype=float)
        R = np.asarray(s["ref"], dtype=float)
        subgroups["%s|%s" % (a, g)] = {
            "arm": a, "subgroup": g, "f": s["f"], "n_tasks": len(s["tasks"]),
            "n": int(L.size), "loso": float(L.mean()) if L.size else float("nan"),
            "ref": float(R.mean()) if R.size else float("nan"),
        }

    return {"bins": out_bins, "arms": have, "reference": ref_arm,
            "subgroups": subgroups, "monotone": monotone, "spearman": sp,
            "n_pairs_for_spearman": len(xs)}


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _f(x, w=7, d=4):
    return ("%*.*f" % (w, d, x)) if (x == x and x is not None) else "%*s" % (w, "--")


def format_loso_report(tab, grad=None, fh=None):
    """Readable LOSO table + OOD gradient, decision rule printed not implied."""
    import sys as _sys
    fh = fh or _sys.stdout
    print("=== leave-one-site-out (era %d only) ===" % cells_mod.LOSO_ERA, file=fh)
    print("reference=%s   versions=%s   seeds=%s"
          % (tab["reference"], tab["versions"], tab["seeds"]), file=fh)
    print("", file=fh)
    print("%-8s %-15s %-5s | %7s %7s %8s %7s | %7s %7s %8s | %8s %6s"
          % ("arm", "corpus", "held", "L[held]", "M[held]", "penalty",
             "retain", "L[seen]", "M[seen]", "delta", "DiD", "DiD/SE"), file=fh)
    for a, r in tab["rows"].items():
        h, s, d = r["held_out_col"], r["seen_cols"], r["did"]
        print("%-8s %-15s %-5s | %s %s %s %s | %s %s %s | %s %6s"
              % (a, r["corpus"], r["held_out"],
                 _f(h["loso"]), _f(h["ref"]), _f(h["penalty"], 8),
                 _f(h["retention"], 7, 3),
                 _f(s["loso"]), _f(s["ref"]), _f(s["delta"], 8),
                 _f(d["value"], 8), _f(d["over_se"], 6, 2)), file=fh)
        ex = "  ".join("%s[%s]=%s" % (k, r["held_out"], _f(v["success"], 6))
                       for k, v in r["extra_refs"].items())
        if ex:
            print("%-8s %s" % ("", ex), file=fh)
        pp = d["paired"]
        if "skipped" not in pp:
            print("%-8s paired DiD over %d seed(s): mean %+.4f  p=%.3f%s"
                  % ("", pp["n_units"], pp["observed"], pp["p_value"],
                     "  (%s)" % pp["note"] if pp.get("note") else ""), file=fh)
    if tab["pooled"]:
        p = tab["pooled"]
        print("", file=fh)
        print("POOLED over %s:" % ", ".join(p["arms"]), file=fh)
        print("  held-out  L=%.4f  M=%.4f  penalty %+.4f (SE %.4f)  retention %.3f  n=%d"
              % (p["held_out"]["loso"], p["held_out"]["ref"],
                 p["held_out"]["penalty"], p["held_out"]["penalty_se"],
                 p["held_out"]["retention"], p["held_out"]["n"]), file=fh)
        print("  seen      L=%.4f  M=%.4f  delta   %+.4f (SE %.4f)  n=%d"
              % (p["seen"]["loso"], p["seen"]["ref"], p["seen"]["delta"],
                 p["seen"]["delta_se"], p["seen"]["n"]), file=fh)
        print("  DiD       %+.4f  (SE %.4f, %.2f SE)   detectable at 2 SE: %.3f"
              % (p["did"]["value"], p["did"]["se"], p["did"]["over_se"],
                 p["detectable_at_2se"]), file=fh)
    print("", file=fh)
    print("VERDICT: %s" % loso_verdict(tab), file=fh)
    if "L_multi" in tab["rows"]:
        print("COMPOSITION: %s" % composition_verdict(tab["rows"]["L_multi"]),
              file=fh)
    if grad is not None:
        print("", file=fh)
        format_gradient(grad, fh=fh)
    print("", file=fh)
    print(LOSO_NOTE, file=fh)


def loso_verdict(tab):
    p = tab.get("pooled")
    if not p or not (p["did"]["se"] == p["did"]["se"]) or p["did"]["se"] <= 0:
        return "insufficient data (need the three single-site LOSO arms and M_all)"
    z = p["did"]["over_se"]
    if z >= 2:
        return ("site-specific knowledge: the held-out site pays a penalty of "
                "%+.3f beyond the seen columns (DiD %.2f SE); retention %.2f. "
                "Report per column -- see which sites carry it."
                % (p["did"]["value"], z, p["held_out"]["retention"]))
    if abs(z) < 1:
        return ("domain-general: the held-out site costs nothing the seen "
                "columns do not also pay (DiD %+.3f, %.2f SE); retention %.2f. "
                "Adapters carry no site-specific knowledge on this axis at "
                "this dose." % (p["did"]["value"], z, p["held_out"]["retention"]))
    return ("indeterminate: DiD %+.3f is between 1 and 2 SE (%.2f); neither "
            "rule fires. Do not add seeds to chase it unless pre-registered."
            % (p["did"]["value"], z))


def composition_verdict(row):
    d = row["did"]
    if not (d["se"] == d["se"]) or d["se"] <= 0:
        return "insufficient data"
    z = d["over_se"]
    h = row["held_out_col"]
    if z >= 2:
        return ("multi-site tasks need multi-site data: L_multi (singles only) "
                "loses %+.3f on multi beyond its seen columns (DiD %.2f SE); "
                "single-site data does not compose." % (d["value"], z))
    if abs(z) < 1:
        return ("single-site data composes: L_multi matches M_all on multi up "
                "to noise (penalty %+.3f, DiD %.2f SE)." % (h["penalty"], z))
    return "indeterminate (DiD %.2f SE)" % z


def format_gradient(grad, fh=None):
    import sys as _sys
    fh = fh or _sys.stdout
    print("=== OOD gradient: unseen fraction of a task's sites -> penalty ===",
          file=fh)
    if not grad["bins"]:
        print("  %s" % grad.get("reason", "no data"), file=fh)
        return
    print("arms=%s  reference=%s  (reference re-binned PER ARM, so each bin is "
          "the same (task, seed) multiset on both sides)"
          % (", ".join(grad["arms"]), grad["reference"]), file=fh)
    print("  %-6s %6s %7s %7s %8s %8s %7s %6s %7s"
          % ("unseen", "n", "LOSO", "M_all", "penalty", "SE", "retain",
             "pairs", "p"), file=fh)
    for fk, b in grad["bins"].items():
        pp = b["paired"]
        print("  %-6s %6d %s %s %s %s %s %6d %7s"
              % (fk, b["n"], _f(b["loso"]), _f(b["ref"]), _f(b["penalty"], 8),
                 _f(b["penalty_se"], 8), _f(b["retention"], 7, 3),
                 b["n_unit_pairs"],
                 ("%.3f" % pp["p_value"]) if "p_value" in pp else "--"),
              file=fh)
    sp = grad.get("spearman")
    print("  monotone (penalty non-decreasing in unseen fraction): %s"
          % grad["monotone"], file=fh)
    if sp:
        print("  Spearman(unseen fraction, per-unit penalty): rho %+.3f  p=%.4f  "
              "n=%d" % (sp["rho"], sp["p_value"], sp["n"]), file=fh)
    print("", file=fh)
    print("  per (arm, subgroup):", file=fh)
    print("  %-8s %-16s %6s %6s %5s %7s %7s"
          % ("arm", "subgroup", "unseen", "tasks", "n", "LOSO", "M_all"), file=fh)
    for k, s in grad["subgroups"].items():
        print("  %-8s %-16s %6s %6d %5d %s %s"
              % (s["arm"], s["subgroup"], _frac_key(s["f"]), s["n_tasks"],
                 s["n"], _f(s["loso"]), _f(s["ref"])), file=fh)
    print("  Subgroups are 3-9 tasks: read the BINS (pooled over arms and "
          "seeds), never a subgroup cell.", file=fh)


def loso_markdown(tab, grad=None):
    """Markdown deliverable: the LOSO table and the gradient bins."""
    lines = ["### Leave-one-site-out (era %d), rows = arm, reference = %s"
             % (cells_mod.LOSO_ERA, tab["reference"]), "",
             "| arm | corpus | held out | L[held] | M[held] | penalty | retention "
             "| L[seen] | M[seen] | seen delta | DiD | DiD/SE |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for a, r in tab["rows"].items():
        h, s, d = r["held_out_col"], r["seen_cols"], r["did"]
        lines.append("| %s | %s | %s | %.3f | %.3f | %+.3f | %.2f | %.3f | %.3f "
                     "| %+.3f | %+.3f | %.2f |"
                     % (a, r["corpus"], r["held_out"], h["loso"], h["ref"],
                        h["penalty"], h["retention"], s["loso"], s["ref"],
                        s["delta"], d["value"], d["over_se"]))
    if tab["pooled"]:
        p = tab["pooled"]
        lines += ["", "Pooled over %s: penalty %+.3f (SE %.3f), retention %.2f, "
                  "seen delta %+.3f, **DiD %+.3f (%.2f SE)**."
                  % (", ".join(p["arms"]), p["held_out"]["penalty"],
                     p["held_out"]["penalty_se"], p["held_out"]["retention"],
                     p["seen"]["delta"], p["did"]["value"], p["did"]["over_se"])]
    lines += ["", "**Verdict:** %s" % loso_verdict(tab)]
    if "L_multi" in tab["rows"]:
        lines += ["", "**Composition:** %s"
                  % composition_verdict(tab["rows"]["L_multi"])]
    if grad and grad["bins"]:
        lines += ["", "### OOD gradient (pooled over %s)" % ", ".join(grad["arms"]),
                  "", "| unseen fraction | n | LOSO | M_all | penalty | SE | retention |",
                  "|---|---|---|---|---|---|---|"]
        for fk, b in grad["bins"].items():
            lines.append("| %s | %d | %.3f | %.3f | %+.3f | %.3f | %.2f |"
                         % (fk, b["n"], b["loso"], b["ref"], b["penalty"],
                            b["penalty_se"], b["retention"]))
        sp = grad.get("spearman")
        lines += ["", "Monotone: %s.%s" % (
            grad["monotone"],
            (" Spearman rho %+.3f, p=%.4f (n=%d unit-bin pairs)."
             % (sp["rho"], sp["p_value"], sp["n"])) if sp else "")]
    lines += ["", "*%s*" % LOSO_NOTE]
    return "\n".join(lines)
