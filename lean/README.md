# Web2LoRA theory, machine-checked in Lean 4

Lean 4 + Mathlib formalisation of the theoretical claims of the Web2LoRA paper (Section
"Theoretical Analysis" and Appendix "Proofs"). Every statement below is proved without
`sorry`; `Audit.lean` prints the axioms each headline result depends on (only `propext`,
`Classical.choice`, `Quot.sound`).

## Build

```
lake exe cache get        # fetch the Mathlib cache once (a few minutes)
lake build                # builds all modules
lake env lean Audit.lean  # axiom audit of the headline theorems
```

Toolchain: `lean-toolchain` pins Lean `v4.34.0-rc2`, Mathlib at the matching tag.

## Modelling choices

* The adapter space `𝒲 = ∏_s ℝ^{d_out(s) × d_in(s)}` with `‖ΔW‖ = (∑_s ‖ΔW_s‖_F²)^{1/2}` is a
  finite-dimensional real normed space. The hull results are stated for an arbitrary real normed
  space `W`; the paper's space is the instance `EuclideanSpace ℝ (Σ s, Fin (d_out s) × Fin (d_in s))`
  (`paper_space_instance`). No inner product or finite dimension is needed for them.
* The risk `ℛ_u : 𝒲 → ℝ` is an arbitrary function; each result carries exactly the hypothesis the
  paper invokes (Lipschitz on `𝒦_u`, continuity on the hull, or convexity on `𝒦_u`).
* `Δ^{K-1}` is Mathlib's `stdSimplex ℝ (Fin K)`; `dist(·, ℋ_K)` is `Metric.infDist`;
  `min_k` statements are given as bounds for every `k`.
* In the twin, `Ξ : Matrix n k ℝ` holds the features `ξ_1, …, ξ_K` as columns, `𝓑 : Matrix m k ℝ`
  the vectorised oracles, `rank Ξ = K` is injectivity of `y ↦ Ξy` (`FullColumnRank`), and
  `λ_max(ΞΞᵀ) ≤ L` is the Rayleigh-quotient bound `∀ w, ‖wΞ‖² ≤ L‖w‖²`. The step-size
  condition `η < 1/λ_max(ΞΞᵀ)` is `η·L < 1`.

## Paper ↔ Lean map

| Paper | Lean (namespace `Web2LoRA`) |
|---|---|
| Eq. thm:mix, `ℋ_K = M(Δ^{K-1}) = conv{ΔW^{(k)}}`, compact, convex | `hull_eq_convexHull`, `isCompact_hull`, `convex_hull'` (Hull.lean) |
| `M(e_k) = ΔW^{(k)}` | `mix_vertex`, `bank_mem_hull` |
| Assumption ass:lip, justification "`Λ_u` = largest gradient norm on `𝒦_u`" | `lipschitzOnWith_of_contDiff`, `assumption_lip_of_contDiff` (Lipschitz.lean); needs convexity of `𝒦_u` (`convex_Ku`) |
| Theorem thm:hull, existence of `c⋆` | `exists_best_mixture` |
| Theorem thm:hull, Eq. thm:decomp identity and routing `≥ 0` | `risk_decomposition`, `routing_nonneg` |
| Theorem thm:hull, first bound of Eq. thm:bounds (`Λ_u · dist(ΔW^{(u)}, ℋ_K)`) | `approximation_bound` |
| Theorem thm:hull, second bound (`≤ min_k ℛ_u(ΔW^{(k)})`) | `best_mixture_le_bank` |
| Theorem thm:hull assembled | `hull_decomposition` |
| Remark rem:hull, `dist` is a `K`-variable least squares over the simplex | `infDist_hull_attained` |
| Corollary cor:growth, `M_{K+1}((c,0)) = M_K(c)`, `ℋ_K ⊆ ℋ_{K+1}` | `mix_snoc_zero`, `hull_subset_hull_snoc` |
| Corollary cor:growth, distance inequality | `infDist_hull_snoc_le` |
| Corollary cor:growth, `ℛ⋆_v(K+1) ≤ ℛ⋆_v(K)` (continuity only) | `Rstar_snoc_le`, `Rstar_eq_of_isMinOn`, `best_mixture_snoc_le` |
| Corollary cor:jensen, Eq. cor:jensen | `jensen_mixture`, `jensen_uniform` |
| Remark rem:jensen / proof: NLL convex in logits, affine logits ⟹ convex risk | `convexOn_lse`, `convexOn_nll`, `convexOn_affineRisk` (Softmax.lean) |
| Assumption ass:twin: `ΞᵀΞ` invertible, `Ξ⁺Ξ = I` | `Twin.gram_isUnit_det`, `Twin.pinv_mul` (Twin.lean) |
| Proposition prop:twin: minimiser set `{Γ : ΓΞ = 𝓑}`, `Γ̂ = 𝓑Ξ⁺` a minimiser | `Twin.isMinOn_loss_iff`, `Twin.loss_Γhat` |
| Proof of prop:twin, `∇L(Γ) = 2(ΓΞ − 𝓑)Ξᵀ` (exact quadratic expansion) | `Twin.loss_expand` |
| Proposition prop:twin, unique minimum-norm interpolant | `Twin.Γhat_min_norm` |
| Proposition prop:twin, gradient descent from `0` with `η < 1/λ_max(ΞΞᵀ)` converges to `Γ̂` | `Twin.gd_tendsto` (Lyapunov argument: `frob2_step_le`, `err_mul_proj`) |
| Eq. prop:twin, `vec ΔW_{Γ̂}(h) = 𝓑 β(h)` | `Twin.output_Γhat` |
| Proposition prop:twin (a), output in the span of the bank | `Twin.output_mem_span` |
| Proposition prop:twin (b), `β` depends on `(Ξ, ξ(h))` only | `Twin.beta` mentions no bank; `Twin.beta_independent_of_bank` |
| Proposition prop:twin (c), `β(ξ_1 ± ξ_2) = e_1 ± e_2 ∉ Δ^{K-1}`, outputs `ΔW^{(1)} ± ΔW^{(2)}` | `Twin.beta_add_cols`, `Twin.beta_sub_cols`, `Twin.beta_add_cols_not_mem_simplex`, `Twin.beta_sub_cols_not_mem_simplex`, `Twin.output_add_cols`, `Twin.output_sub_cols`, `Twin.add_cols_eq_two_smul_mean`, `Twin.beta_of_orthonormal` |
| Corollary cor:scale, `‖M(c)‖ ≤ max_k ‖ΔW^{(k)}‖` and per site | `norm_mix_le`, `norm_mix_le_sup`, `norm_site_mix_le` |
| Corollary cor:scale, twin: `ξ(h) = tξ_k ↦ tΔW^{(k)}`, unbounded | `Twin.output_smul_col`, `Twin.output_unbounded` |
| Eq. bg:gauge, `(BQ⁻¹)(QA) = BA` | `Gauge.gauge_invariance` (Gauge.lean) |
| Eq. exact / Eq. bg:concat, concatenation is the exact convex combination | `Gauge.concat_mul`, `Gauge.concat_mul_unweighted` |
| Cross terms of averaging factors separately | `Gauge.avg_factors_cross_terms` |
| Proposition prop:wellposed (a) | `Gauge.mixture_gauge_invariant`, `Gauge.objective_gauge_invariant` |
| Proposition prop:wellposed (b), Remark rem:wellposed | `mix_injective`, `mix_injOn_simplex`, `mix_eq_iff` |
| Proposition prop:wellposed (c) | `Gauge.factor_loss_not_gauge_invariant`, `Gauge.factor_loss_not_constant` |

Not formalised: the bridge assumptions that are stated as such in the paper (behaviour-cloning
risk and task success move together; the twin idealises T2L), and the claim that the base network
is `C¹` in its weights (it enters `assumption_lip_of_contDiff` as the hypothesis `ContDiff ℝ 1 R`).
The gradient-descent convergence is proved directly (a Lyapunov/summability argument in the row
space of `Ξᵀ`) rather than by citing the general convex-quadratic result the paper's proof invokes;
the statement proved is the one the paper makes.

## Discrepancies found and corrected in `main.tex` (purple edits, explained at the top of the Proofs appendix)

1. Remark rem:jensen said "Convexity holds *exactly when* the adapted sites feed the logits
   affinely". Only sufficiency is proved (`convexOn_affineRisk`); the converse was never argued.
   Now "whenever …; the converse is not claimed".
2. Remark rem:hull and the takeaway of Corollary cor:jensen read predictions off "column `u` of the
   transfer matrix", but that column also contains the diagonal entry (the oracle `ΔW^{(u)}`), which
   is not a bank member. `best_mixture_le_bank` bounds by the bank adapters only, so the text now
   says "off-diagonal entries"; the ceiling on the approximation term is the gap between the best
   off-diagonal entry and the diagonal entry.
3. Remark rem:hull said that beating every column entry means the head "is using the interior of the
   hull". What follows from `best_mixture_le_bank` is only that the mixture is not a vertex (it may
   lie on a face); the text now says so. "Fixed point of `ℋ_K`" was reworded to "fixed element".
4. Remark rem:growth predicted a non-increasing bank-size curve for "banks of size 2,…,5 drawn from
   the training eras"; `Rstar_snoc_le` needs each bank to extend the previous one. Now "nested".
5. The justification of Assumption ass:lip used the mean value inequality on `𝒦_u` without stating
   that `𝒦_u` is convex (it is, as a convex hull: `convex_Ku`); `lipschitzOnWith_of_contDiff` takes
   convexity as a hypothesis, and the text now mentions it.
