"""Numeric falsification of Remark rem:jensen (WebMix.Statements.rem_jensen).

If the logits of every demonstrated token are an affine function of the update, the BC risk
sum_i w_i nll(y_i, A_i x + b_i) with w_i >= 0 is convex: checked along random segments
(R((1-t)x + t z) <= (1-t)R(x) + t R(z)), including large logits and zero weights.

Also checks the review finding on the scope wording: logits that are affine in each adapted
matrix *separately* (here bilinear in two matrices, logits = (W1 + D1)(W2 + D2) v) can give a
nonconvex risk, so the condition must be joint affinity in the whole update.
"""
import numpy as np

from _common import RNG, Report, lse

rep = Report("rem_jensen")
for trial in range(3000):
    d = int(RNG.choice([1, 2, 6]))
    V = int(RNG.integers(1, 7))
    N = int(RNG.integers(1, 6))
    A = RNG.normal(size=(N, V, d)) * RNG.choice([1, 30])
    b = RNG.normal(size=(N, V)) * RNG.choice([1, 30])
    y = RNG.integers(V, size=N)
    w = RNG.exponential(size=N) * RNG.choice([0.0, 1.0], size=N)
    R = lambda x: float(sum(w[i] * (lse(A[i] @ x + b[i]) - (A[i] @ x + b[i])[y[i]])
                            for i in range(N)))
    x, z = RNG.normal(size=d) * 3, RNG.normal(size=d) * 3
    for t in (0.1, 0.5, 0.9):
        lhs, rhs = R((1 - t) * x + t * z), (1 - t) * R(x) + t * R(z)
        rep.check(lhs <= rhs + 1e-9 * (1 + abs(rhs)), f"convexity trial={trial}: {lhs} > {rhs}")

# Separate affinity is not enough: one token, two classes, logits (w1 + a)(w2 + b) * [1, -1]
# depend bilinearly on the update (a, b). Midpoint convexity fails somewhere.
def R_bilinear(a, bb, y=0):
    s = (1.0 + a) * (1.0 + bb)
    logits = np.array([s, -s])
    return float(lse(logits) - logits[y])

found = False
for _ in range(20000):
    p, q = RNG.normal(size=2) * 3, RNG.normal(size=2) * 3
    mid = (p + q) / 2
    if R_bilinear(*mid) > (R_bilinear(*p) + R_bilinear(*q)) / 2 + 1e-6:
        found = True
        print(f"  separate affinity, nonconvex risk: p={np.round(p, 3)}, q={np.round(q, 3)}")
        break
rep.check(found, "expected a nonconvex example for logits affine in each matrix separately")
rep.done()
