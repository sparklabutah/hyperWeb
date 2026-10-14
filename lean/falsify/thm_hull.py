"""Numeric falsification of Theorem thm:hull (WebMix.Statements.thm_hull).

Samples banks of K >= 1 updates in R^d, an oracle ΔW^(u) (inside, on, or far outside the hull),
and risks with a known Lipschitz constant Λ on all of R^d (hence on K_u):
  R(x) = Λ ||x - p|| + a         (convex, exactly Λ-Lipschitz; c* computable exactly),
  R(x) = <a, x>                  (Λ = ||a||),
  R(x) = lse(A x + b)            (convex, Λ = ||A||_op),
  R(x) = sum_i s_i sin(<w_i, x>) (nonconvex, Λ = sum_i |s_i| ||w_i||).
Checks, with the paper's constants:
  * bound 1 at the projection c_Π (the inequality the proof uses) and at c*:
        R(M(c)) - R(ΔW^(u)) <= Λ dist(ΔW^(u), H_K),
  * bound 2: R(M(c*)) <= min_k R(ΔW^(k)),
  * routing term >= 0 for random router outputs ĉ in the simplex,
  * the decomposition identity.
c* is computed exactly (first family) or by projected gradient (convex families); for the
nonconvex family only the c_Π form of bound 1 is checked.
"""
import numpy as np

from _common import RNG, Report, dist_to_hull, lse, minimize_convex_on_simplex, random_bank, \
    random_simplex_point

rep = Report("thm_hull")


def tol(*vals):
    return 1e-7 * (1 + max(abs(v) for v in vals))


for trial in range(1500):
    K = int(RNG.choice([1, 1, 2, 3, 4, 6]))
    d = int(RNG.choice([1, 2, 3, 8, 30]))
    bank = random_bank(K, d)
    scale = np.max(np.abs(bank)) + 1e-12
    where = RNG.integers(3)
    if where == 0:      # inside the hull
        u = random_simplex_point(K) @ bank
    elif where == 1:    # near the hull
        u = random_simplex_point(K) @ bank + 0.1 * scale * RNG.normal(size=d)
    else:               # far outside
        u = 10 * scale * RNG.normal(size=d)
    dist, c_pi = dist_to_hull(bank, u)
    M = lambda c: c @ bank
    family = trial % 4
    if family == 0:
        lam = float(RNG.exponential(2.0))
        p = u + scale * RNG.normal(size=d) * RNG.choice([0.0, 1.0])
        a0 = float(RNG.normal())
        R = lambda x: lam * np.linalg.norm(x - p) + a0
        _, c_star = dist_to_hull(bank, p)  # argmin of ||M(c) - p|| over the simplex
        convex = True
    elif family == 1:
        a = RNG.normal(size=d) / scale
        lam = float(np.linalg.norm(a))
        R = lambda x: float(a @ x)
        Rg = lambda c: bank @ a
        _, c_star = minimize_convex_on_simplex(lambda c: R(M(c)), Rg, K)
        convex = True
    elif family == 2:
        V = int(RNG.integers(2, 7))
        A = RNG.normal(size=(V, d)) / scale
        b = RNG.normal(size=V)
        lam = float(np.linalg.norm(A, 2))

        def R(x, A=A, b=b):
            return float(lse(A @ x + b))

        def Rg(c, A=A, b=b):
            z = A @ M(c) + b
            s = np.exp(z - z.max())
            s /= s.sum()
            return bank @ (A.T @ s)

        _, c_star = minimize_convex_on_simplex(lambda c: R(M(c)), Rg, K)
        convex = True
    else:
        J = int(RNG.integers(1, 5))
        Wf = RNG.normal(size=(J, d)) / scale
        s = RNG.normal(size=J)
        lam = float(np.sum(np.abs(s) * np.linalg.norm(Wf, axis=1)))
        R = lambda x, Wf=Wf, s=s: float(s @ np.sin(Wf @ x))
        c_star = None
        convex = False

    # Bound 1 in the form the proof uses: at the projection c_Π.
    lhs = R(M(c_pi)) - R(u)
    rep.check(lhs <= lam * dist + tol(lhs, lam * dist),
              f"bound1@c_pi trial={trial} fam={family} lhs={lhs:.3e} rhs={lam * dist:.3e}")
    if not convex:
        continue
    # c* is (numerically) the global minimiser: keep the best of c*, the vertices and c_Π.
    cands = [c_star, c_pi] + [np.eye(K)[k] for k in range(K)]
    c_star = min(cands, key=lambda c: R(M(c)))
    r_star = R(M(c_star))
    approx = r_star - R(u)
    rep.check(approx <= lam * dist + tol(approx, lam * dist),
              f"bound1@c* trial={trial} fam={family} approx={approx:.3e} rhs={lam * dist:.3e}")
    rmin = min(R(bank[k]) for k in range(K))
    rep.check(r_star <= rmin + tol(r_star, rmin), f"bound2 trial={trial} {r_star} > {rmin}")
    for _ in range(5):
        c_hat = random_simplex_point(K)
        routing = R(M(c_hat)) - r_star
        rep.check(routing >= -1e-6 * (1 + abs(r_star)), f"routing<0 trial={trial} {routing:.3e}")
        total = R(M(c_hat)) - R(u)
        rep.check(abs(total - (routing + approx)) <= tol(total), "decomposition identity")

rep.done()
