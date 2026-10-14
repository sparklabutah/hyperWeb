"""Numeric falsification of Proposition prop:wellposed (WebMix.Statements.prop_wellposed).

(a) Re-gauging each bank member at each matrix by an invertible Q (well- and ill-conditioned)
    leaves the served update sum_k c_k ρ B^(k) A^(k) unchanged.
(b) For a linearly independent bank, distinct coefficient vectors give distinct mixtures.
(c) The factor loss ||A - t A*||_1 + ||B - B*/t||_1 differs between t = 1 and some t > 0, while
    every (t A*, B*/t) gives the same product B* A*.
"""
import numpy as np

from _common import RNG, Report, random_simplex_point

rep = Report("prop_wellposed")
for trial in range(800):
    r, dout, din, K = (int(RNG.integers(1, 5)) for _ in range(4))
    rho = float(RNG.normal())
    A = RNG.normal(size=(K, r, din))
    B = RNG.normal(size=(K, dout, r))
    c = RNG.normal(size=K)
    served = sum(c[k] * rho * B[k] @ A[k] for k in range(K))
    Qs = []
    for k in range(K):
        Q = RNG.normal(size=(r, r))
        if trial % 4 == 0:
            U, _, Vt = np.linalg.svd(Q)
            Q = U @ np.diag(np.logspace(0, -6, r)) @ Vt  # ill-conditioned
        Qs.append(Q)
    regauged = sum(c[k] * rho * (B[k] @ np.linalg.inv(Qs[k])) @ (Qs[k] @ A[k]) for k in range(K))
    rep.check(np.allclose(served, regauged, rtol=1e-5, atol=1e-6), f"(a) trial={trial}")
for trial in range(800):
    K = int(RNG.integers(1, 6))
    d = K + int(RNG.integers(0, 4))
    bank = RNG.normal(size=(K, d))
    if np.linalg.matrix_rank(bank) < K:
        continue
    c1, c2 = random_simplex_point(K), random_simplex_point(K)
    if np.linalg.norm(c1 - c2) > 1e-9:
        rep.check(np.linalg.norm((c1 - c2) @ bank) > 0, "(b) distinct c with equal mixture")
for trial in range(800):
    r, dout, din = (int(RNG.integers(1, 4)) for _ in range(3))
    A, B = RNG.normal(size=(r, din)), RNG.normal(size=(dout, r))
    As, Bs = RNG.normal(size=(r, din)), RNG.normal(size=(dout, r))
    loss = lambda t: np.abs(A - t * As).sum() + np.abs(B - Bs / t).sum()
    ts = np.logspace(-3, 3, 61)
    rep.check(any(abs(loss(t) - loss(1.0)) > 1e-9 for t in ts), "(c) loss constant in t")
    for t in (0.1, 7.0):
        rep.check(np.allclose((Bs / t) @ (t * As), Bs @ As), "(c) same model")
rep.done()
