"""Shared helpers for the numeric falsification tests (Gate A, item 8).

Each falsify/<label>.py samples random instances that satisfy the hypotheses of
WebMix.Statements.<label>, including edge cases (K = 1, duplicate or collinear bank members,
targets inside and far outside the hull, near-singular matrices), and checks the conclusion with
the paper's exact constants. The tests are independent of Lean.

Run one test with `python3 falsify/<label>.py`, or all of them with `python3 falsify/run_all.py`.
Exit status 0 means no violation was found.
"""
import itertools
import sys

import numpy as np

RNG = np.random.default_rng(20261014)
TOL = 1e-8


class Report:
    """Counts checks and violations; prints a one-line verdict per test."""

    def __init__(self, label):
        self.label = label
        self.checks = 0
        self.violations = []

    def check(self, ok, what):
        self.checks += 1
        if not ok:
            self.violations.append(what)

    def done(self):
        status = "PASS" if not self.violations else "FAIL"
        print(f"[{status}] {self.label}: {self.checks} checks, {len(self.violations)} violations")
        for v in self.violations[:10]:
            print("   ", v)
        sys.exit(0 if not self.violations else 1)


def random_simplex_point(K, rng=RNG):
    """A point of the simplex: Dirichlet, a vertex, or near a vertex/face (edge cases)."""
    kind = rng.integers(4)
    if kind == 0:
        return rng.dirichlet(np.ones(K))
    if kind == 1:
        c = np.zeros(K)
        c[rng.integers(K)] = 1.0
        return c
    if kind == 2:
        c = rng.dirichlet(0.05 * np.ones(K))  # sparse, close to a face
        return c / c.sum()
    return np.ones(K) / K


def random_bank(K, d, rng=RNG):
    """K bank updates in R^d (rows), with edge cases: duplicates, collinear, tiny, huge."""
    kind = rng.integers(5)
    bank = rng.normal(size=(K, d))
    if kind == 1 and K >= 2:
        bank[1] = bank[0]  # duplicate member
    elif kind == 2 and K >= 2:
        bank[1] = 3.0 * bank[0]  # collinear members
    elif kind == 3:
        bank *= 1e-4  # tiny updates (the oracle scale of Appendix B is about 5e-5)
    elif kind == 4:
        bank *= 1e3
    return bank


def dist_to_hull(bank, x):
    """Exact Euclidean distance from x to conv(rows of bank), with the minimising coefficients.

    Enumerates the faces (subsets S of the bank). On each, it solves the affine least squares
    min_t ||(x - b_0) - sum_j t_j (b_j - b_0)|| (so c = (1 - sum t, t) sums to one exactly) and
    keeps the best solution with c >= 0. The minimum over the hull is attained on an affinely
    independent face, where this solution is unique, so the enumeration is exact (2^K - 1 faces)."""
    K = bank.shape[0]
    best, best_c = np.inf, None
    for size in range(1, K + 1):
        for S in itertools.combinations(range(K), size):
            S = list(S)
            P = bank[S]
            if size == 1:
                cS = np.ones(1)
            else:
                D = (P[1:] - P[0]).T  # d x (|S| - 1)
                t = np.linalg.lstsq(D, x - P[0], rcond=None)[0]
                cS = np.concatenate([[1.0 - t.sum()], t])
            if np.any(cS < -1e-10):
                continue
            c = np.zeros(K)
            c[S] = np.clip(cS, 0, None)
            c /= c.sum()
            dist = np.linalg.norm(x - c @ bank)
            if dist < best:
                best, best_c = dist, c
    return best, best_c


def project_simplex(v):
    """Euclidean projection onto the probability simplex (Duchi et al., 2008)."""
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(v) + 1) > (css - 1))[0][-1]
    theta = (css[rho] - 1) / (rho + 1.0)
    return np.maximum(v - theta, 0)


def minimize_convex_on_simplex(f, grad, K, iters=4000, starts=4, rng=RNG):
    """Projected gradient descent with backtracking, from several starts; returns (value, c)."""
    best_val, best_c = np.inf, None
    inits = [np.ones(K) / K] + [rng.dirichlet(np.ones(K)) for _ in range(starts - 1)]
    for c in inits:
        step = 1.0
        val = f(c)
        for _ in range(iters):
            g = grad(c)
            while True:
                c_new = project_simplex(c - step * g)
                v_new = f(c_new)
                if v_new <= val - 1e-4 * np.dot(g, c - c_new) or step < 1e-12:
                    break
                step *= 0.5
            if abs(val - v_new) < 1e-15:
                c, val = c_new, v_new
                break
            c, val = c_new, v_new
            step = min(step * 2, 1e6)
        if val < best_val:
            best_val, best_c = val, c
    return best_val, best_c


def lse(z):
    m = np.max(z)
    return m + np.log(np.sum(np.exp(z - m)))
