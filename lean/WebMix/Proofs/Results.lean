import WebMix.Proofs.Hull
import WebMix.Proofs.Lipschitz
import WebMix.Proofs.Softmax
import WebMix.Proofs.Twin
import WebMix.Proofs.Gauge

/-!
# The paper's results

Each theorem `WebMix.<label>` below has type `WebMix.Statements.<label>`, the statement frozen at
Gate A, so the kernel checks that what is proved is exactly what was approved. Each proof only
assembles the lemmas that carry the steps of the paper's proof (Appendix D); the step table in
`README.md` maps every paper step to its lemma.
-/

namespace WebMix

/-- **Assumption ass:lip, justification**: a `C¹` risk is Lipschitz on `𝒦_u` with constant the
largest derivative norm on `𝒦_u`. -/
theorem ass_lip : Statements.ass_lip := by
  intro W _ _ K ΔW ΔWu R hR
  exact assumption_lip_of_contDiff ΔW ΔWu R hR

/-- **Theorem thm:hull** (hull decomposition of the held-out risk). -/
theorem thm_hull : Statements.thm_hull := by
  intro W _ _ K ΔW ΔWu R Λ hK hLip
  exact hull_decomposition ΔW ΔWu R Λ hLip hK

/-- **Remark rem:hull**: the distance to the hull is attained, is a least-squares problem in an
inner-product space, and the uniform mixture lies in the hull. -/
theorem rem_hull : Statements.rem_hull := by
  refine ⟨fun W _ _ K ΔW ΔWu hK => ⟨(infDist_hull_attained ΔW ΔWu hK).1,
      (infDist_hull_attained ΔW ΔWu hK).2, mix_uniform_mem_hull ΔW hK⟩,
    fun W _ _ K ΔW ΔWu c => norm_target_sub_mix_sq ΔW ΔWu c⟩

/-- **Corollary cor:growth** (growing the bank). -/
theorem cor_growth : Statements.cor_growth := by
  intro W _ _ K ΔW new hK
  refine ⟨mix_snoc_zero ΔW new, hull_subset_hull_snoc ΔW new, infDist_hull_snoc_le ΔW new hK,
    fun R hR => ⟨?_, ?_, ?_⟩⟩
  · -- the old bank: `R` is continuous on `ℋ_K ⊆ ℋ_{K+1}`
    exact exists_best_mixture ΔW R (hR.mono (hull_subset_hull_snoc ΔW new)) hK
  · exact exists_best_mixture (Fin.snoc ΔW new) R hR (Nat.succ_pos K)
  · intro c' _ hmin' c hc
    exact best_mixture_snoc_le R ΔW new hc hmin'

/-- **Corollary cor:jensen** (convex twin). -/
theorem cor_jensen : Statements.cor_jensen := by
  intro W _ _ K ΔW ΔWu R hconv
  exact ⟨fun c hc => jensen_mixture ΔW ΔWu R hconv hc, fun hK => jensen_uniform ΔW ΔWu R hconv hK⟩

/-- **Remark rem:jensen** (scope of the convex twin): affine logits make the risk convex. -/
theorem rem_jensen : Statements.rem_jensen := by
  intro W _ _ ι _ _ N A y w hw
  exact convexOn_bcRisk A y w hw

/-- **Proposition prop:twin** (direct generation is unnormalised routing). -/
theorem prop_twin : Statements.prop_twin := by
  refine ⟨fun m n k _ _ _ _ Ξ h => ?_,
    fun n k _ _ _ Ξ _hrank => ⟨Twin.beta Ξ, fun m B ξ => ?_⟩⟩
  · exact ⟨fun B Γ => Twin.isMinOn_loss_iff Ξ B h Γ,
      fun B => ⟨Twin.Γhat_mul Ξ B h, fun Γ hΓ => Twin.Γhat_min_norm Ξ B h Γ hΓ⟩,
      fun B L hL0 hL η hη hηL => Twin.gd_tendsto Ξ B h hL0 hL hη hηL,
      fun B ξ => ⟨Twin.output_Γhat Ξ B ξ, Twin.output_mem_span Ξ B ξ⟩,
      fun B j₁ j₂ hj => ⟨Twin.beta_add_cols Ξ h j₁ j₂, Twin.beta_add_cols_not_mem_simplex Ξ h j₁ j₂,
        Twin.output_add_cols Ξ B h j₁ j₂, Twin.beta_sub_cols Ξ h j₁ j₂,
        Twin.beta_sub_cols_not_mem_simplex Ξ h hj, Twin.output_sub_cols Ξ B h j₁ j₂⟩⟩
  · -- statement (b): `Γ̂ξ = (𝓑Ξ⁺)ξ = 𝓑(Ξ⁺ξ)` for every bank `𝓑` of every size `m`
    exact Twin.output_Γhat Ξ B ξ

/-- **Corollary cor:scale** (scale control; no bound in the twin over feature vectors). -/
theorem cor_scale : Statements.cor_scale := by
  refine ⟨fun W _ _ K ΔW c hc => exists_norm_mix_le ΔW hc,
    fun W Ws _ _ _ _ P K ΔW c hc => exists_norm_proj_mix_le P ΔW hc,
    fun m n k _ _ _ _ Ξ B h => ?_⟩
  exact ⟨fun t j => Twin.output_smul_col Ξ B h t j,
    fun j hj M => Twin.output_unbounded_euc Ξ B h j hj M⟩

/-- **Proposition prop:wellposed** (gauge invariance and coefficient identifiability). -/
theorem prop_wellposed : Statements.prop_wellposed := by
  refine ⟨fun r Ω _ _ dout din K ρ A B Q hQ c s =>
      Gauge.served_update_gauge_invariant ρ A B Q hQ c s,
    fun r Ω _ _ dout din F A B Q hQ => Gauge.objective_gauge_invariant F A B Q hQ,
    fun W _ _ K ΔW h => mix_injOn_simplex ΔW h,
    fun r dout din _ _ _ A Astar B Bstar hA => ?_⟩
  exact ⟨(Gauge.factor_loss_not_gauge_invariant A Astar B Bstar hA).1,
    Gauge.factor_loss_not_constant A Astar B Bstar hA⟩

/-- **Remark rem:wellposed** (reading the coefficients). -/
theorem rem_wellposed : Statements.rem_wellposed := by
  intro W _ _ K ΔW
  exact ⟨Matrix.det_gram_ne_zero_iff_linearIndependent, fun h c c' => mix_eq_iff ΔW h c c',
    fun c c' => norm_mix_sub_mix_sq ΔW c c'⟩

/-- **Eqs. delta and exact** (exact composition, served scale, cross terms of factor averaging). -/
theorem eq_exact : Statements.eq_exact := by
  intro r dout din _ K c A B
  exact ⟨Gauge.concat_mul c A B, fun α hK => Gauge.served_update hK α c A B,
    Gauge.avg_factors_cross_terms c A B⟩

end WebMix
