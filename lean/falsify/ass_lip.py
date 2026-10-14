"""Numeric falsification of the justification of Assumption ass:lip (WebMix.Statements.ass_lip).

For C^1 risks whose gradient norm is a convex function of x (R(x) = x^T Q x / 2 + g^T x with Q
PSD, gradient Qx + g), the largest gradient norm on K_u = conv(bank ∪ {ΔW^(u)}) is attained at a
generating point, so Λ_u is computed exactly; then |R(x) - R(z)| <= Λ_u ||x - z|| is checked for
random pairs x, z in K_u (convex combinations of the generating points, including vertices).
"""
import numpy as np

from _common import RNG, Report, random_bank, random_simplex_point

rep = Report("ass_lip")
for trial in range(1500):
    K = int(RNG.choice([1, 2, 3, 5]))
    d = int(RNG.choice([1, 2, 4, 10]))
    bank = random_bank(K, d)
    scale = np.max(np.abs(bank)) + 1e-12
    u = RNG.normal(size=d) * scale * RNG.choice([0.1, 1, 5])
    gens = np.vstack([bank, u])  # K_u = conv of these K + 1 points
    Qh = RNG.normal(size=(d, d)) / scale
    Q = Qh @ Qh.T
    g = RNG.normal(size=d)
    R = lambda x: float(0.5 * x @ Q @ x + g @ x)
    grad_norm = lambda x: float(np.linalg.norm(Q @ x + g))
    lam = max(grad_norm(p) for p in gens)  # sup over K_u of a convex function: at a generator
    for _ in range(20):
        x = random_simplex_point(K + 1) @ gens
        z = random_simplex_point(K + 1) @ gens
        rep.check(grad_norm(x) <= lam * (1 + 1e-9) + 1e-12, "grad norm above the max at a vertex")
        lhs, rhs = abs(R(x) - R(z)), lam * np.linalg.norm(x - z)
        rep.check(lhs <= rhs + 1e-9 * (1 + abs(lhs)), f"Lipschitz trial={trial}: {lhs} > {rhs}")
rep.done()
