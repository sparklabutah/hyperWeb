"""Numeric falsification of Corollary cor:scale (WebMix.Statements.cor_scale).

Banks of per-matrix blocks: ||M(c)|| <= max_k ||ΔW^(k)|| and, for every block s,
||M(c)_s||_F <= max_k ||ΔW^(k)_s||_F, for c in the simplex (vertices, faces, interior), with
banks containing duplicate, collinear, tiny and huge members. In the twin, the feature vector
t ξ_j yields the output t ΔW^(j), so the Euclidean norm grows linearly in t.
"""
import numpy as np

from _common import RNG, Report, random_bank, random_simplex_point

rep = Report("cor_scale")
for trial in range(2000):
    K = int(RNG.choice([1, 2, 3, 6]))
    sizes = RNG.integers(1, 5, size=int(RNG.integers(1, 4)))  # entries per block
    d = int(sizes.sum())
    bank = random_bank(K, d)
    blocks = np.split(np.arange(d), np.cumsum(sizes)[:-1])
    for _ in range(10):
        c = random_simplex_point(K)
        m = c @ bank
        top = max(np.linalg.norm(bank[k]) for k in range(K))
        rep.check(np.linalg.norm(m) <= top * (1 + 1e-12) + 1e-300, "whole-update bound")
        for idx in blocks:
            top_s = max(np.linalg.norm(bank[k][idx]) for k in range(K))
            rep.check(np.linalg.norm(m[idx]) <= top_s * (1 + 1e-12) + 1e-300, "per-block bound")
for trial in range(300):
    K = int(RNG.integers(1, 4))
    n = K + int(RNG.integers(0, 3))
    Xi = RNG.normal(size=(n, K))
    B = RNG.normal(size=(int(RNG.integers(1, 5)), K))
    G = B @ np.linalg.solve(Xi.T @ Xi, Xi.T)
    j = int(RNG.integers(K))
    for t in (0.5, 2.0, 1e3):
        rep.check(np.allclose(G @ (t * Xi[:, j]), t * B[:, j], rtol=1e-6, atol=1e-6), "t ξ_j")
rep.done()
