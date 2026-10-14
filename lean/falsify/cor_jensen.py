"""Numeric falsification of Corollary cor:jensen (WebMix.Statements.cor_jensen).

Convex risks (behaviour-cloning risks with logits affine in the update, R(x) = sum_i w_i
nll(y_i, A_i x + b_i), and quadratics): for random c in the simplex,
R(M(c)) <= sum_k c_k R(ΔW^(k)), and R(M(1/K)) <= (1/K) sum_k R(ΔW^(k)).

Also prints the counterexample behind the review finding on the success-level reading of the
corollary: the risk inequality holds, yet with the natural link success = exp(-risk) (the
likelihood of reproducing the demonstration, a decreasing *convex* function of the risk), the
uniform mixture's success is far below the mean success of the bank. That reading needs success
to be a nonincreasing *concave* function of risk; this script checks that the counterexample is
real and that a concave link restores the inequality.
"""
import numpy as np

from _common import RNG, Report, lse, random_bank, random_simplex_point

rep = Report("cor_jensen")


def make_bc_risk(d, rng=RNG):
    V = int(rng.integers(2, 6))
    N = int(rng.integers(1, 6))
    A = rng.normal(size=(N, V, d))
    b = rng.normal(size=(N, V))
    y = rng.integers(V, size=N)
    w = rng.exponential(size=N) * rng.choice([0.0, 1.0], size=N, p=[0.2, 0.8])

    def R(x):
        return float(sum(w[i] * (lse(A[i] @ x + b[i]) - (A[i] @ x + b[i])[y[i]]) for i in range(N)))

    return R


for trial in range(1500):
    K = int(RNG.choice([1, 2, 3, 5]))
    d = int(RNG.choice([1, 3, 8]))
    bank = random_bank(K, d) / (1 + np.max(np.abs(random_bank(K, d))))
    if trial % 2:
        R = make_bc_risk(d)
    else:
        Q = RNG.normal(size=(d, d))
        Q = Q @ Q.T
        g = RNG.normal(size=d)
        R = lambda x, Q=Q, g=g: float(0.5 * x @ Q @ x + g @ x)
    vals = np.array([R(bank[k]) for k in range(K)])
    for _ in range(10):
        c = random_simplex_point(K)
        lhs, rhs = R(c @ bank), float(c @ vals)
        rep.check(lhs <= rhs + 1e-9 * (1 + abs(rhs)), f"Jensen trial={trial}: {lhs} > {rhs}")
    lhs, rhs = R(np.ones(K) / K @ bank), float(vals.mean())
    rep.check(lhs <= rhs + 1e-9 * (1 + abs(rhs)), f"uniform trial={trial}: {lhs} > {rhs}")

# Review finding: a monotone link does not carry the uniform-mixture inequality to success.
# Two adapters with risks 0 and 10; a convex (here affine) risk along the segment gives the
# uniform mixture risk 5 <= mean risk 5, but exp(-5) << (exp(0) + exp(-10)) / 2.
risks = np.array([0.0, 10.0])
r_unif = risks.mean()  # affine (hence convex) risk on the segment
succ = lambda r: np.exp(-r)
print(f"  counterexample: risk(uniform)={r_unif:.1f} <= mean risk={risks.mean():.1f}, "
      f"but success(uniform)={succ(r_unif):.4f} < mean success={succ(risks).mean():.4f}")
rep.check(r_unif <= risks.mean() and succ(r_unif) < succ(risks).mean(),
          "the success-level counterexample should exist")
# With a nonincreasing concave link the success-level inequality does follow.
for _ in range(2000):
    K = int(RNG.integers(1, 6))
    r = RNG.exponential(size=K) * 3
    r_mix = r.mean() - RNG.exponential()  # any value <= the mean risk
    a, bb = RNG.exponential(), RNG.exponential()
    concave = lambda t: -a * t - bb * t ** 2  # nonincreasing and concave on t >= 0
    if r_mix < 0:
        continue
    rep.check(concave(r_mix) >= concave(r).mean() - 1e-9, "concave link")
rep.done()
