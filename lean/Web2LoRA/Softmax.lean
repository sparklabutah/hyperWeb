import Mathlib

/-!
# Convexity of the negative log-likelihood in the logits

Formalises the scope statement of Remark `rem:jensen` / the second half of the proof of
Corollary `cor:jensen`: the per-token negative log-likelihood `ℓ ↦ log ∑_j exp ℓ_j − ℓ_y` is
convex in the logits, the composition of a convex function with an affine map is convex, and a
finite average of convex functions is convex.  Hence, when every adapted site feeds the logits
affinely, the behaviour-cloning risk `ℛ_u` is convex on all of `𝒲`.

Only sufficiency is proved (affine logits ⟹ convex risk).  The paper's original wording
"convexity holds exactly when …" claimed the converse as well; the text was corrected to
"convexity holds when …".
-/

open Finset Real

namespace Web2LoRA

section LogSumExp

variable {ι : Type*} [Fintype ι]

/-- log-sum-exp of the logits. -/
noncomputable def lse (ℓ : ι → ℝ) : ℝ := Real.log (∑ j, Real.exp (ℓ j))

/-- Per-token negative log-likelihood of the softmax at target `y`. -/
noncomputable def nll (y : ι) (ℓ : ι → ℝ) : ℝ := lse ℓ - ℓ y

lemma sumExp_pos [Nonempty ι] (ℓ : ι → ℝ) : 0 < ∑ j, Real.exp (ℓ j) :=
  Finset.sum_pos (fun _ _ => Real.exp_pos _) Finset.univ_nonempty

/-- Hölder's inequality gives `∑ exp(a x_j + b y_j) ≤ (∑ exp x_j)^a (∑ exp y_j)^b`. -/
lemma sumExp_combo_le (x y : ι → ℝ) {a b : ℝ} (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
    ∑ j, Real.exp (a * x j + b * y j)
      ≤ (∑ j, Real.exp (x j)) ^ a * (∑ j, Real.exp (y j)) ^ b := by
  have ha1 : a < 1 := by linarith
  have hpq : Real.HolderConjugate (1 / a) (1 / b) := by
    rw [Real.holderConjugate_iff]
    refine ⟨one_lt_one_div ha ha1, ?_⟩
    simp only [one_div, inv_inv]
    exact hab
  have h := Real.inner_le_Lp_mul_Lq_of_nonneg (s := Finset.univ)
    (f := fun j => Real.exp (a * x j)) (g := fun j => Real.exp (b * y j)) hpq
    (fun j _ => (Real.exp_pos _).le) (fun j _ => (Real.exp_pos _).le)
  have hf : ∀ j, Real.exp (a * x j) ^ (1 / a) = Real.exp (x j) := fun j => by
    rw [← Real.exp_mul]; congr 1; field_simp
  have hg : ∀ j, Real.exp (b * y j) ^ (1 / b) = Real.exp (y j) := fun j => by
    rw [← Real.exp_mul]; congr 1; field_simp
  simp only [hf, hg, one_div_one_div, ← Real.exp_add] at h
  exact h

/-- log-sum-exp is convex on `ℝ^ι`. -/
theorem convexOn_lse [Nonempty ι] : ConvexOn ℝ Set.univ (lse (ι := ι)) := by
  refine convexOn_iff_forall_pos.2 ⟨convex_univ, ?_⟩
  intro x _ y _ a b ha hb hab
  have hx := sumExp_pos x
  have hy := sumExp_pos y
  have h := sumExp_combo_le x y ha hb hab
  have hpos : 0 < ∑ j, Real.exp (a * x j + b * y j) :=
    Finset.sum_pos (fun j _ => Real.exp_pos _) Finset.univ_nonempty
  simp only [smul_eq_mul, lse, Pi.add_apply, Pi.smul_apply]
  calc Real.log (∑ j, Real.exp (a * x j + b * y j))
      ≤ Real.log ((∑ j, Real.exp (x j)) ^ a * (∑ j, Real.exp (y j)) ^ b) :=
        Real.log_le_log hpos h
    _ = a * Real.log (∑ j, Real.exp (x j)) + b * Real.log (∑ j, Real.exp (y j)) := by
        rw [Real.log_mul (Real.rpow_pos_of_pos hx a).ne' (Real.rpow_pos_of_pos hy b).ne',
          Real.log_rpow hx, Real.log_rpow hy]

/-- The per-token negative log-likelihood is convex in the logits. -/
theorem convexOn_nll [Nonempty ι] (y : ι) : ConvexOn ℝ Set.univ (nll y) := by
  have hlin : ConcaveOn ℝ Set.univ (fun ℓ : ι → ℝ => ℓ y) :=
    (LinearMap.proj y : (ι → ℝ) →ₗ[ℝ] ℝ).concaveOn convex_univ
  exact convexOn_lse.sub hlin

end LogSumExp

section AffineRisk

variable {W : Type*} [AddCommGroup W] [Module ℝ W]

/-- A finite sum of convex functions is convex. -/
theorem convexOn_finset_sum {ι' : Type*} {s : Set W} (hs : Convex ℝ s) (t : Finset ι')
    {f : ι' → W → ℝ} (h : ∀ i ∈ t, ConvexOn ℝ s (f i)) :
    ConvexOn ℝ s (fun x => ∑ i ∈ t, f i x) := by
  classical
  induction t using Finset.induction_on with
  | empty => simpa using convexOn_const (0 : ℝ) hs
  | insert a t ha ih =>
    simp_rw [Finset.sum_insert ha]
    exact (h a (Finset.mem_insert_self _ _)).add (ih fun i hi => h i (Finset.mem_insert_of_mem hi))

variable {ι : Type*} [Fintype ι]

/-- The behaviour-cloning risk of a corpus of `N` tokens whose logits `A i ΔW` are affine in the
installed update `ΔW`, with targets `y i`: `ℛ(ΔW) = (1/N) ∑_i nll_{y_i}(A_i ΔW)`. -/
noncomputable def affineRisk {N : ℕ} (A : Fin N → (W →ᵃ[ℝ] (ι → ℝ))) (y : Fin N → ι)
    (ΔW : W) : ℝ :=
  (1 / N : ℝ) * ∑ i, nll (y i) (A i ΔW)

/-- Scope of Corollary `cor:jensen`: when the adapted sites feed the logits affinely, the risk is
convex on all of `𝒲`. -/
theorem convexOn_affineRisk [Nonempty ι] {N : ℕ} (A : Fin N → (W →ᵃ[ℝ] (ι → ℝ)))
    (y : Fin N → ι) : ConvexOn ℝ Set.univ (affineRisk A y) := by
  have h : ∀ i ∈ (Finset.univ : Finset (Fin N)),
      ConvexOn ℝ Set.univ (fun ΔW : W => nll (y i) (A i ΔW)) := fun i _ => by
    have := (convexOn_nll (y i)).comp_affineMap (A i)
    rw [Set.preimage_univ] at this
    exact this
  have hsum := convexOn_finset_sum convex_univ Finset.univ h
  have h2 : ConvexOn ℝ Set.univ (fun ΔW : W => (1 / N : ℝ) • ∑ i, nll (y i) (A i ΔW)) :=
    hsum.smul (by positivity)
  exact h2

end AffineRisk

end Web2LoRA
