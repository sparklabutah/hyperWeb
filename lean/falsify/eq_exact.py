"""Numeric falsification of Eqs. delta/exact (WebMix.Statements.eq_exact).

Concatenating [c_1 A^(1); ...; c_K A^(K)] and [B^(1), ..., B^(K)] gives B A = sum_k c_k B^(k) A^(k);
with r_out = K r and α_out = K α, (α_out / r_out) B A = sum_k c_k (α/r) B^(k) A^(k); averaging
the factors separately gives sum_{k,j} c_k c_j B^(k) A^(j) instead, which differs in general.
"""
import numpy as np

from _common import RNG, Report

rep = Report("eq_exact")
differs = 0
for trial in range(1500):
    r, dout, din, K = (int(RNG.integers(1, 6)) for _ in range(4))
    A = RNG.normal(size=(K, r, din))
    B = RNG.normal(size=(K, dout, r))
    c = RNG.dirichlet(np.ones(K)) if trial % 2 else RNG.normal(size=K)
    A_cat = np.concatenate([c[k] * A[k] for k in range(K)], axis=0)
    B_cat = np.concatenate([B[k] for k in range(K)], axis=1)
    target = sum(c[k] * B[k] @ A[k] for k in range(K))
    rep.check(np.allclose(B_cat @ A_cat, target), "concatenation identity")
    alpha = float(RNG.exponential() * 32)
    served = (K * alpha) / (K * r) * (B_cat @ A_cat)
    rep.check(np.allclose(served, sum(c[k] * (alpha / r) * B[k] @ A[k] for k in range(K))),
              "served update with α_out = Kα, r_out = Kr")
    avg = sum(c[k] * B[k] for k in range(K)) @ sum(c[j] * A[j] for j in range(K))
    cross = sum(c[k] * c[j] * B[k] @ A[j] for k in range(K) for j in range(K))
    rep.check(np.allclose(avg, cross), "cross-term expansion")
    differs += int(K >= 2 and not np.allclose(avg, target))
rep.check(differs > 0, "averaging factors should differ from the exact mixture in general")
rep.done()
