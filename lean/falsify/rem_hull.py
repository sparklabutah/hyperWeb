"""Numeric falsification of Remark rem:hull (WebMix.Statements.rem_hull).

dist(ΔW^(u), H_K) is a K-variable least-squares problem over the simplex: it is attained at the
exact face-enumeration solution c_P, no simplex point (random, vertex, near-face) does better,
and the uniform mixture M(1/K) lies in H_K (its distance to the hull is 0). In the Euclidean case
the squared distance to M(c) is the quadratic ||u||^2 - 2 sum_k c_k <u, ΔW^(k)> + c^T G c.
"""
import numpy as np

from _common import RNG, Report, dist_to_hull, random_bank, random_simplex_point

rep = Report("rem_hull")
for trial in range(1500):
    K = int(RNG.choice([1, 2, 3, 5, 7]))
    d = int(RNG.choice([1, 2, 4, 16]))
    bank = random_bank(K, d)
    scale = np.max(np.abs(bank)) + 1e-12
    u = random_simplex_point(K) @ bank + scale * RNG.normal(size=d) * RNG.choice([0, 0.1, 5])
    dist, c_p = dist_to_hull(bank, u)
    rep.check(abs(np.linalg.norm(u - c_p @ bank) - dist) <= 1e-9 * (1 + dist), "attained at c_P")
    for _ in range(20):
        c = random_simplex_point(K)
        rep.check(dist <= np.linalg.norm(u - c @ bank) + 1e-9 * (1 + dist),
                  f"trial={trial}: a simplex point beats the computed distance")
    G = bank @ bank.T
    for _ in range(5):
        c = RNG.normal(size=K)
        lhs = np.linalg.norm(u - c @ bank) ** 2
        rhs = u @ u - 2 * c @ (bank @ u) + c @ G @ c
        rep.check(abs(lhs - rhs) <= 1e-8 * (1 + abs(lhs) + u @ u + abs(c @ G @ c)), "quadratic form")
    d_unif, _ = dist_to_hull(bank, np.ones(K) / K @ bank)
    rep.check(d_unif <= 1e-7 * scale, f"uniform mixture not in hull: {d_unif}")
rep.done()
