"""Numeric falsification of Proposition prop:twin (WebMix.Statements.prop_twin).

Random features Ξ (d_ξ x K, full column rank, some nearly collinear) and banks 𝓑 (D x K):
  * Γ̂ = 𝓑 Ξ⁺ interpolates, Γ̂ Ξ = 𝓑;
  * every other interpolant Γ̂ + N (N Ξ = 0) has a larger Frobenius norm;
  * gradient descent Γ <- Γ - η 2(ΓΞ - 𝓑)Ξᵀ from 0 with 0 < η < 1/λ_max(ΞΞᵀ) converges to Γ̂
    (checked at η = 0.5/λ_max and 0.99/λ_max), and the error norm never increases;
  * teeth: with η = 1.05/λ_max the iteration diverges;
  * Eq. prop:twin: Γ̂ ξ = 𝓑 Ξ⁺ ξ; (c) η(ξ_1 ± ξ_2) = e_1 ± e_2 without orthonormality.
"""
import numpy as np

from _common import RNG, Report

rep = Report("prop_twin")
for trial in range(400):
    K = int(RNG.integers(1, 5))
    n = K + int(RNG.integers(0, 4))
    D = int(RNG.integers(1, 7))
    Xi = RNG.normal(size=(n, K))
    if K >= 2 and trial % 5 == 0:
        Xi[:, 1] = Xi[:, 0] + 0.2 * RNG.normal(size=n)  # nearly collinear features
    if np.linalg.matrix_rank(Xi) < K:
        continue
    B = RNG.normal(size=(D, K))
    pinv = np.linalg.solve(Xi.T @ Xi, Xi.T)
    G = B @ pinv
    rep.check(np.allclose(G @ Xi, B, atol=1e-8), "Γ̂ does not interpolate")
    for _ in range(5):
        N = RNG.normal(size=(D, n))
        N = N - N @ Xi @ pinv  # project rows onto the orthogonal complement of span{ξ_k}
        rep.check(np.allclose(N @ Xi, 0, atol=1e-8), "N Ξ != 0")
        gn, nn = np.linalg.norm(G) ** 2, np.linalg.norm(N) ** 2
        rep.check(np.linalg.norm(G + N) ** 2 >= gn - 1e-9 * (gn + nn) - 1e-12,
                  "another interpolant has a smaller norm")
    lam_max = float(np.linalg.eigvalsh(Xi @ Xi.T).max())
    lam_min = float(np.linalg.eigvalsh(Xi.T @ Xi).min())
    for frac in (0.5, 0.99):
        # The error contracts by ρ = max(|1 - 2η λ_min(ΞᵀΞ)|, |1 - 2η λ_max|) per step; budget the
        # iterations for a 1e-9 reduction (a test-budget limit, not a hypothesis of the result).
        eta = frac / lam_max
        rho = max(abs(1 - 2 * eta * lam_min), abs(1 - 2 * eta * lam_max))
        iters = int(np.log(1e9) / (1 - rho)) + 100
        if iters > 400000:
            continue
        Gam = np.zeros((D, n))
        prev = np.linalg.norm(Gam - G)
        monotone = True
        for t in range(iters):
            Gam = Gam - eta * 2 * (Gam @ Xi - B) @ Xi.T
            if t % 50 == 0:
                cur = np.linalg.norm(Gam - G)
                monotone &= cur <= prev * (1 + 1e-12) + 1e-12 * (1 + np.linalg.norm(G))
                prev = cur
        err = np.linalg.norm(Gam - G) / (1 + np.linalg.norm(G))
        rep.check(err < 1e-6, f"GD did not converge (frac={frac}, err={err:.2e}, K={K})")
        rep.check(monotone, "GD error increased")
    Gam = np.zeros((D, n))
    for t in range(3000):
        Gam = Gam - (1.05 / lam_max) * 2 * (Gam @ Xi - B) @ Xi.T
    rep.check(np.linalg.norm(Gam - G) > 1e3 * (1 + np.linalg.norm(G)),
              "teeth: η above 1/λ_max should diverge")
    xi = RNG.normal(size=n)
    rep.check(np.allclose(G @ xi, B @ (pinv @ xi)), "Eq. prop:twin")
    if K >= 2:
        e = np.eye(K)
        rep.check(np.allclose(pinv @ (Xi[:, 0] + Xi[:, 1]), e[0] + e[1], atol=1e-7), "η(ξ1+ξ2)")
        rep.check(np.allclose(pinv @ (Xi[:, 0] - Xi[:, 1]), e[0] - e[1], atol=1e-7), "η(ξ1-ξ2)")
rep.done()
