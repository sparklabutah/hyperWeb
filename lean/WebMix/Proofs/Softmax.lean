import WebMix.Statements
import Mathlib.Analysis.MeanInequalities
import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-!
# Convexity of the behaviour-cloning risk when the logits are affine in the update

Proof of Remark rem:jensen (the second half of the proof of Corollary cor:jensen in the paper).

Strategy. The per-token negative log-likelihood is `nll_y(ℓ) = lse(ℓ) − ℓ_y`. Log-sum-exp is convex
by Hölder's inequality, `∑_j e^{a x_j + b y_j} ≤ (∑_j e^{x_j})^a (∑_j e^{y_j})^b` for `a + b = 1`,
and `ℓ ↦ ℓ_y` is linear, so `nll_y` is convex in the logits. Composing a convex function with an
affine map keeps it convex, and so does a nonnegative combination of convex functions. Hence the
risk is convex on all of `𝒲` whenever the logits are an affine function of the whole update.
-/

open Finset Real

namespace WebMix

section LogSumExp

variable {ι : Type*} [Fintype ι]

/-- `∑_j exp ℓ_j > 0` for a nonempty vocabulary. -/
lemma sumExp_pos [Nonempty ι] (ℓ : ι → ℝ) : 0 < ∑ j, Real.exp (ℓ j) :=
  Finset.sum_pos (fun _ _ => Real.exp_pos _) Finset.univ_nonempty

/-- Hölder's inequality: `∑_j exp(a x_j + b y_j) ≤ (∑_j exp x_j)^a (∑_j exp y_j)^b` for
`a, b > 0`, `a + b = 1`. -/
lemma sumExp_combo_le (x y : ι → ℝ) {a b : ℝ} (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
    ∑ j, Real.exp (a * x j + b * y j)
      ≤ (∑ j, Real.exp (x j)) ^ a * (∑ j, Real.exp (y j)) ^ b := by
  have ha1 : a < 1 := by linarith
  -- conjugate exponents `1/a` and `1/b`
  have hpq : Real.HolderConjugate (1 / a) (1 / b) := by
    rw [Real.holderConjugate_iff]
    refine ⟨one_lt_one_div ha ha1, ?_⟩
    simp only [one_div, inv_inv]
    exact hab
  -- Hölder: `∑_j f_j g_j ≤ (∑_j f_j^p)^{1/p} (∑_j g_j^q)^{1/q}` with `p = 1/a`, `q = 1/b`,
  -- `f_j = exp(a x_j)` and `g_j = exp(b y_j)`
  have h := Real.inner_le_Lp_mul_Lq_of_nonneg (s := Finset.univ)
    (f := fun j => Real.exp (a * x j)) (g := fun j => Real.exp (b * y j)) hpq
    (fun j _ => (Real.exp_pos _).le) (fun j _ => (Real.exp_pos _).le)
  have hf : ∀ j, Real.exp (a * x j) ^ (1 / a) = Real.exp (x j) := fun j => by
    rw [← Real.exp_mul]; congr 1; field_simp
  have hg : ∀ j, Real.exp (b * y j) ^ (1 / b) = Real.exp (y j) := fun j => by
    rw [← Real.exp_mul]; congr 1; field_simp
  -- `f_j^p = exp x_j`, `g_j^q = exp y_j`, `1/(1/a) = a`, and `f_j g_j = exp(a x_j + b y_j)`
  simp only [hf, hg, one_div_one_div, ← Real.exp_add] at h
  exact h

/-- log-sum-exp is convex on `ℝ^ι`: take logarithms in `sumExp_combo_le`. -/
theorem convexOn_lse [Nonempty ι] : ConvexOn ℝ Set.univ (lse (ι := ι)) := by
  refine convexOn_iff_forall_pos.2 ⟨convex_univ, ?_⟩
  intro x _ y _ a b ha hb hab
  have hx := sumExp_pos x
  have hy := sumExp_pos y
  have hpos : 0 < ∑ j, Real.exp (a * x j + b * y j) :=
    Finset.sum_pos (fun j _ => Real.exp_pos _) Finset.univ_nonempty
  simp only [smul_eq_mul, lse, Pi.add_apply, Pi.smul_apply]
  calc Real.log (∑ j, Real.exp (a * x j + b * y j))
      ≤ Real.log ((∑ j, Real.exp (x j)) ^ a * (∑ j, Real.exp (y j)) ^ b) :=
        Real.log_le_log hpos (sumExp_combo_le x y ha hb hab)
    _ = a * Real.log (∑ j, Real.exp (x j)) + b * Real.log (∑ j, Real.exp (y j)) := by
        rw [Real.log_mul (Real.rpow_pos_of_pos hx a).ne' (Real.rpow_pos_of_pos hy b).ne',
          Real.log_rpow hx, Real.log_rpow hy]

/-- The per-token negative log-likelihood `lse(ℓ) − ℓ_y` is convex in the logits. -/
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

/-- Remark rem:jensen: with logits `A_i ΔW` affine in the whole update and weights `w_i ≥ 0`, the
risk `∑_i w_i nll(y_i, A_i ΔW)` is convex on all of `𝒲`. -/
theorem convexOn_bcRisk [Nonempty ι] {N : ℕ} (A : Fin N → (W →ᵃ[ℝ] (ι → ℝ))) (y : Fin N → ι)
    (w : Fin N → ℝ) (hw : ∀ i, 0 ≤ w i) : ConvexOn ℝ Set.univ (bcRisk A y w) := by
  have hterm : ∀ i ∈ (Finset.univ : Finset (Fin N)),
      ConvexOn ℝ Set.univ (fun ΔW : W => w i * nll (y i) (A i ΔW)) := fun i _ => by
    -- `nll ∘ A_i` is convex (convex after affine), and `w_i ≥ 0` keeps it convex
    have hcomp : ConvexOn ℝ Set.univ (fun ΔW : W => nll (y i) (A i ΔW)) := by
      have := (convexOn_nll (y i)).comp_affineMap (A i)
      rwa [Set.preimage_univ] at this
    exact hcomp.smul (hw i)
  exact convexOn_finset_sum convex_univ Finset.univ hterm

end AffineRisk

end WebMix
