import Mathlib
import Web2LoRA.Hull

/-!
# Justification of Assumption `ass:lip`

The paper argues: "The set `𝒦_u` is compact, and the base network is continuously differentiable
in its weights, so the assumption holds with `Λ_u` equal to the largest gradient norm of `ℛ_u` on
`𝒦_u`."  This is the mean value inequality on a convex compact set: convexity of `𝒦_u` (it is a
convex hull) is what makes the argument go through, and the paper's text was amended to say so.
-/

open Set

namespace Web2LoRA

variable {W : Type*} [NormedAddCommGroup W] [NormedSpace ℝ W]

/-- A `C¹` function on a convex compact set is Lipschitz there with constant equal to the largest
norm of its derivative on the set. -/
theorem lipschitzOnWith_of_contDiff (R : W → ℝ) (hR : ContDiff ℝ 1 R) (S : Set W)
    (hconv : Convex ℝ S) (hcomp : IsCompact S) (hne : S.Nonempty) :
    ∃ x₀ ∈ S, (∀ x ∈ S, ‖fderiv ℝ R x‖ ≤ ‖fderiv ℝ R x₀‖) ∧
      LipschitzOnWith ‖fderiv ℝ R x₀‖₊ R S := by
  obtain ⟨x₀, hx₀, hmax⟩ :=
    hcomp.exists_isMaxOn hne ((hR.continuous_fderiv one_ne_zero).norm.continuousOn)
  refine ⟨x₀, hx₀, fun x hx => hmax hx, ?_⟩
  refine hconv.lipschitzOnWith_of_nnnorm_fderiv_le
    (fun x _ => (hR.differentiable one_ne_zero) x) (fun x hx => ?_)
  rw [← NNReal.coe_le_coe, coe_nnnorm, coe_nnnorm]
  exact hmax hx

/-- Assumption `ass:lip` holds for every `C¹` risk, with `Λ_u` the largest gradient norm on
`𝒦_u = conv(ℋ_K ∪ {ΔW^{(u)}})`. -/
theorem assumption_lip_of_contDiff {K : ℕ} (ΔW : Fin K → W) (ΔWu : W) (R : W → ℝ)
    (hR : ContDiff ℝ 1 R) :
    ∃ x₀ ∈ Ku ΔW ΔWu, (∀ x ∈ Ku ΔW ΔWu, ‖fderiv ℝ R x‖ ≤ ‖fderiv ℝ R x₀‖) ∧
      LipschitzOnWith ‖fderiv ℝ R x₀‖₊ R (Ku ΔW ΔWu) :=
  lipschitzOnWith_of_contDiff R hR _ (convex_Ku ΔW ΔWu) (isCompact_Ku ΔW ΔWu)
    ⟨ΔWu, target_mem_Ku ΔW ΔWu⟩

end Web2LoRA
