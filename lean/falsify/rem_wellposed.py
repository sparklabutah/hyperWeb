"""Numeric falsification of Remark rem:wellposed (WebMix.Statements.rem_wellposed).

With the Gram matrix G_jk = <ΔW^(j), ΔW^(k)>: det G != 0 iff the bank is linearly independent
(random full-rank and rank-deficient banks); ||M(c) - M(c')||^2 = (c - c')^T G (c - c') for all
coefficient vectors.
"""
import numpy as np

from _common import RNG, Report

rep = Report("rem_wellposed")
for trial in range(1500):
    K = int(RNG.integers(1, 6))
    d = int(RNG.integers(1, 8))
    bank = RNG.normal(size=(K, d))
    if trial % 3 == 0 and K >= 2:
        bank[-1] = bank[:-1].T @ RNG.normal(size=K - 1)  # force linear dependence
    G = bank @ bank.T
    independent = np.linalg.matrix_rank(bank) == K
    nonsingular = np.linalg.matrix_rank(G) == K
    rep.check(independent == nonsingular, f"det G != 0 iff LI fails: trial={trial}")
    c, c2 = RNG.normal(size=K), RNG.normal(size=K)
    lhs = np.linalg.norm((c - c2) @ bank) ** 2
    rhs = (c - c2) @ G @ (c - c2)
    rep.check(abs(lhs - rhs) <= 1e-9 * (1 + abs(lhs)), "Gram identity")
rep.done()
