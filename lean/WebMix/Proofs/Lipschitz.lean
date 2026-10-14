import WebMix.Proofs.Hull
import Mathlib.Analysis.Calculus.MeanValue

/-!
# Justification of Assumption ass:lip

The paper argues: `𝒦_u` is convex and compact and the base network is continuously
differentiable in its weights, so by the mean value inequality the assumption holds with `Λ_u`
equal to the largest gradient norm of `ℛ_u` on `𝒦_u`.

Strategy. The derivative norm `x ↦ ‖Dℛ(x)‖` is continuous (the risk is `C¹`), so it attains its
maximum on the compact set `𝒦_u` at some `x₀`. On the convex set `𝒦_u` the mean value inequality
turns this bound on the derivative into a Lipschitz bound with the same constant. Convexity of
`𝒦_u` is essential to the mean value step; it holds because `𝒦_u` is a convex hull.
-/

open Set

namespace WebMix

variable {W : Type*} [NormedAddCommGroup W] [NormedSpace ℝ W]

/-- A `C¹` function on a convex compact set is Lipschitz there, with constant the largest norm of
its derivative on the set. -/
theorem lipschitzOnWith_of_contDiff (R : W → ℝ) (hR : ContDiff ℝ 1 R) (S : Set W)
    (hconv : Convex ℝ S) (hcomp : IsCompact S) (hne : S.Nonempty) :
    ∃ x₀ ∈ S, (∀ x ∈ S, ‖fderiv ℝ R x‖ ≤ ‖fderiv ℝ R x₀‖) ∧
      LipschitzOnWith ‖fderiv ℝ R x₀‖₊ R S := by
  -- the largest derivative norm on the compact set `S`
  obtain ⟨x₀, hx₀, hmax⟩ :=
    hcomp.exists_isMaxOn hne ((hR.continuous_fderiv one_ne_zero).norm.continuousOn)
  refine ⟨x₀, hx₀, fun x hx => hmax hx, ?_⟩
  -- the mean value inequality on the convex set `S`
  refine hconv.lipschitzOnWith_of_nnnorm_fderiv_le
    (fun x _ => (hR.differentiable one_ne_zero) x) (fun x hx => ?_)
  rw [← NNReal.coe_le_coe, coe_nnnorm, coe_nnnorm]
  exact hmax hx

/-- Assumption ass:lip holds for every `C¹` risk, with `Λ_u` the largest derivative norm on
`𝒦_u = conv(ℋ_K ∪ {ΔW^{(u)}})`, which is convex, compact and nonempty. -/
theorem assumption_lip_of_contDiff {K : ℕ} (ΔW : Fin K → W) (ΔWu : W) (R : W → ℝ)
    (hR : ContDiff ℝ 1 R) :
    ∃ x₀ ∈ Ku ΔW ΔWu, (∀ x ∈ Ku ΔW ΔWu, ‖fderiv ℝ R x‖ ≤ ‖fderiv ℝ R x₀‖) ∧
      LipschitzOnWith ‖fderiv ℝ R x₀‖₊ R (Ku ΔW ΔWu) :=
  lipschitzOnWith_of_contDiff R hR _ (convex_Ku ΔW ΔWu) (isCompact_Ku ΔW ΔWu)
    ⟨ΔWu, target_mem_Ku ΔW ΔWu⟩

end WebMix
