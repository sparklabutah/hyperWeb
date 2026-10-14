"""Numeric falsification of Corollary cor:growth (WebMix.Statements.cor_growth).

Appending one update to a bank of K >= 1: M_{K+1}((c, 0)) = M_K(c); dist(x, H_{K+1}) <=
dist(x, H_K) for random x; and for convex risks (exact minimisation over each simplex) the best
achievable risk does not increase, R*(K+1) <= R*(K). Also checks nested sequences of banks
K = 1, ..., 6 (the bank-size curve of Remark rem:growth is non-increasing).
"""
import numpy as np

from _common import RNG, Report, dist_to_hull, random_bank, random_simplex_point

rep = Report("cor_growth")
for trial in range(800):
    K = int(RNG.choice([1, 2, 3, 5]))
    d = int(RNG.choice([1, 2, 5, 12]))
    bank = random_bank(K, d)
    new = RNG.normal(size=d) * np.max(np.abs(bank)) * RNG.choice([0.0, 1.0, 3.0])
    big = np.vstack([bank, new])
    c = random_simplex_point(K)
    rep.check(np.allclose(np.append(c, 0.0) @ big, c @ bank), "M_{K+1}((c,0)) != M_K(c)")
    x = RNG.normal(size=d) * np.max(np.abs(big)) * RNG.choice([0.1, 1, 10])
    d_small, _ = dist_to_hull(bank, x)
    d_big, _ = dist_to_hull(big, x)
    rep.check(d_big <= d_small + 1e-9 * (1 + d_small), f"dist grew: {d_big} > {d_small}")
    # Convex risk R(z) = ||z - p||: R* is the distance from p to the hull, computed exactly.
    p = RNG.normal(size=d) * np.max(np.abs(big))
    r_small, _ = dist_to_hull(bank, p)
    r_big, _ = dist_to_hull(big, p)
    rep.check(r_big <= r_small + 1e-9 * (1 + r_small), f"R* grew: {r_big} > {r_small}")

# Nested banks: the best-mixture risk on a held-out point is non-increasing in K.
for trial in range(300):
    d = int(RNG.choice([2, 5, 10]))
    full = RNG.normal(size=(6, d))
    p = RNG.normal(size=d) * 2
    curve = [dist_to_hull(full[:K], p)[0] for K in range(1, 7)]
    rep.check(all(b <= a + 1e-9 * (1 + a) for a, b in zip(curve, curve[1:])),
              f"nested curve not monotone: {np.round(curve, 4)}")
rep.done()
