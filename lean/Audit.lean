import Web2LoRA

/-!
Axiom audit of the headline results.  Run with `lake env lean Audit.lean`.
Every result should depend only on `propext`, `Classical.choice`, `Quot.sound`.
-/

open Web2LoRA

-- Theorem thm:hull and its pieces
#print axioms hull_decomposition
#print axioms exists_best_mixture
#print axioms approximation_bound
#print axioms best_mixture_le_bank
#print axioms hull_eq_convexHull
#print axioms infDist_hull_attained
-- Corollary cor:growth
#print axioms mix_snoc_zero
#print axioms hull_subset_hull_snoc
#print axioms infDist_hull_snoc_le
#print axioms Rstar_snoc_le
#print axioms best_mixture_snoc_le
-- Corollary cor:jensen and its scope
#print axioms jensen_mixture
#print axioms jensen_uniform
#print axioms convexOn_lse
#print axioms convexOn_nll
#print axioms convexOn_affineRisk
-- Corollary cor:scale
#print axioms norm_mix_le
#print axioms norm_mix_le_sup
#print axioms norm_site_mix_le
#print axioms Twin.output_smul_col
#print axioms Twin.output_unbounded
-- Proposition prop:twin
#print axioms Twin.gram_isUnit_det
#print axioms Twin.pinv_mul
#print axioms Twin.isMinOn_loss_iff
#print axioms Twin.loss_expand
#print axioms Twin.Γhat_min_norm
#print axioms Twin.gd_tendsto
#print axioms Twin.output_Γhat
#print axioms Twin.output_mem_span
#print axioms Twin.beta_add_cols
#print axioms Twin.beta_sub_cols
#print axioms Twin.output_add_cols
#print axioms Twin.output_sub_cols
#print axioms Twin.beta_add_cols_not_mem_simplex
#print axioms Twin.beta_sub_cols_not_mem_simplex
#print axioms Twin.beta_of_orthonormal
-- Proposition prop:wellposed
#print axioms Gauge.gauge_invariance
#print axioms Gauge.concat_mul
#print axioms Gauge.avg_factors_cross_terms
#print axioms Gauge.mixture_gauge_invariant
#print axioms Gauge.objective_gauge_invariant
#print axioms mix_injective
#print axioms mix_eq_iff
#print axioms Gauge.factor_loss_not_gauge_invariant
#print axioms Gauge.factor_loss_not_constant
-- Assumption ass:lip
#print axioms lipschitzOnWith_of_contDiff
#print axioms assumption_lip_of_contDiff
