import WebMix

/-!
Axiom audit: every paper result and every lemma of the paper ↔ Lean map in `README.md`.
Run with `lake env lean Audit.lean`. Each line should report only `propext`, `Classical.choice`
and `Quot.sound`.
-/

open WebMix

-- The paper's results (each has type `WebMix.Statements.<label>`)
#print axioms WebMix.ass_lip
#print axioms WebMix.thm_hull
#print axioms WebMix.rem_hull
#print axioms WebMix.cor_growth
#print axioms WebMix.cor_jensen
#print axioms WebMix.rem_jensen
#print axioms WebMix.prop_twin
#print axioms WebMix.cor_scale
#print axioms WebMix.prop_wellposed
#print axioms WebMix.rem_wellposed
#print axioms WebMix.eq_exact
-- Theorem thm:hull, step by step
#print axioms hull_eq_convexHull
#print axioms isCompact_hull
#print axioms hull_convex
#print axioms convex_Ku
#print axioms isCompact_Ku
#print axioms mix_vertex
#print axioms exists_best_mixture
#print axioms risk_decomposition
#print axioms routing_nonneg
#print axioms approximation_bound
#print axioms best_mixture_le_bank
#print axioms hull_decomposition
-- Assumption ass:lip
#print axioms lipschitzOnWith_of_contDiff
#print axioms assumption_lip_of_contDiff
-- Remark rem:hull
#print axioms infDist_hull_attained
#print axioms norm_target_sub_mix_sq
#print axioms mix_uniform_mem_hull
-- Corollary cor:growth
#print axioms mix_snoc_zero
#print axioms hull_subset_hull_snoc
#print axioms infDist_hull_snoc_le
#print axioms best_mixture_snoc_le
#print axioms snoc_zero_mem_simplex
#print axioms realized_rise_le_routing_rise
-- Corollary cor:jensen and Remark rem:jensen
#print axioms jensen_mixture
#print axioms jensen_uniform
#print axioms sumExp_combo_le
#print axioms convexOn_lse
#print axioms convexOn_nll
#print axioms convexOn_bcRisk
#print axioms WebMix.Sanity.separate_affinity_not_convex
-- Proposition prop:twin and the twin half of Corollary cor:scale
#print axioms Twin.gram_isUnit_det
#print axioms Twin.pinv_mul
#print axioms Twin.Γhat_mul
#print axioms Twin.isMinOn_loss_iff
#print axioms Twin.loss_expand
#print axioms Twin.Γhat_min_norm
#print axioms Twin.gdStep_sub_Γhat
#print axioms Twin.err_mul_proj
#print axioms Twin.frob2_mul_transpose_le
#print axioms Twin.frob2_step_le
#print axioms Twin.tendsto_err_mul
#print axioms Twin.err_eq
#print axioms Twin.gd_tendsto
#print axioms Twin.output_Γhat
#print axioms Twin.output_mem_span
#print axioms Twin.beta_col
#print axioms Twin.beta_add_cols
#print axioms Twin.beta_sub_cols
#print axioms Twin.output_add_cols
#print axioms Twin.output_sub_cols
#print axioms Twin.add_cols_eq_two_smul_mean
#print axioms Twin.beta_add_cols_not_mem_simplex
#print axioms Twin.beta_sub_cols_not_mem_simplex
#print axioms Twin.beta_smul_col
#print axioms Twin.output_smul_col
#print axioms Twin.output_unbounded_euc
-- Corollary cor:scale, mixture half
#print axioms norm_mix_le
#print axioms exists_norm_mix_le
#print axioms exists_norm_proj_mix_le
-- Eqs. bg:gauge, delta, exact and Proposition prop:wellposed
#print axioms Gauge.gauge_invariance
#print axioms Gauge.concat_mul
#print axioms Gauge.served_update
#print axioms Gauge.avg_factors_cross_terms
#print axioms Gauge.served_update_gauge_invariant
#print axioms Gauge.objective_gauge_invariant
#print axioms mix_injective
#print axioms mix_injOn_simplex
#print axioms Gauge.l1_sub_smul_ge
#print axioms Gauge.factor_loss_not_gauge_invariant
#print axioms Gauge.factor_loss_not_constant
-- Remark rem:wellposed
#print axioms mix_eq_iff
#print axioms norm_mix_sq
#print axioms norm_mix_sub_mix_sq
