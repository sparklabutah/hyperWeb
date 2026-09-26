import Mathlib

/-!
# Gauge invariance, exact composition, and factor-space losses

Formalises

* Eq. bg:gauge — `(QA, BQ⁻¹)` induces the same update as `(A, B)`;
* Eq. exact / Eq. bg:concat — concatenation along the rank axis realises the convex combination
  of the bank updates exactly;
* the cross-term formula for averaging the factors separately (Section "Composition");
* Proposition `prop:wellposed` (a): the objective sees the induced updates only, so re-gauging any
  bank member leaves `M(c)` and the loss unchanged;
* Proposition `prop:wellposed` (c): an `ℓ₁` loss on the factors is not a function of the target
  model.
-/

open Matrix

namespace Web2LoRA.Gauge

variable {r : Type*} [Fintype r] [DecidableEq r]

/-- Eq. bg:gauge: for invertible `Q`, `(BQ⁻¹)(QA) = BA`. -/
theorem gauge_invariance {dout din : Type*} (A : Matrix r din ℝ) (B : Matrix dout r ℝ) (Q : Matrix r r ℝ)
    (hQ : IsUnit Q.det) : (B * Q⁻¹) * (Q * A) = B * A := by
  rw [Matrix.mul_assoc, Matrix.nonsing_inv_mul_cancel_left _ _ hQ]

/-- The factors emitted by Eq. delta: `A_s = [c_1 A^{(1)}_s; …; c_K A^{(K)}_s]` (rows indexed by
`Fin K × r`). -/
def concatA {din : Type*} {K : ℕ} (c : Fin K → ℝ) (A : Fin K → Matrix r din ℝ) : Matrix (Fin K × r) din ℝ :=
  Matrix.of fun p j => c p.1 * A p.1 p.2 j

/-- `B_s = [B^{(1)}_s, …, B^{(K)}_s]` (columns indexed by `Fin K × r`). -/
def concatB {dout : Type*} {K : ℕ} (B : Fin K → Matrix dout r ℝ) : Matrix dout (Fin K × r) ℝ :=
  Matrix.of fun o p => B p.1 o p.2

/-- Eq. exact: `B_s A_s = ∑_k c_k B^{(k)}_s A^{(k)}_s`. -/
theorem concat_mul {dout din : Type*} {K : ℕ} (c : Fin K → ℝ) (A : Fin K → Matrix r din ℝ)
    (B : Fin K → Matrix dout r ℝ) :
    concatB B * concatA c A = ∑ k, c k • (B k * A k) := by
  ext o j
  simp only [concatA, concatB, Matrix.mul_apply, Matrix.of_apply, Fintype.sum_prod_type,
    Matrix.sum_apply, Matrix.smul_apply, smul_eq_mul, Finset.mul_sum]
  refine Finset.sum_congr rfl fun k _ => Finset.sum_congr rfl fun i _ => by ring

/-- Eq. bg:concat: unweighted concatenation adds the updates exactly. -/
theorem concat_mul_unweighted {dout din : Type*} {K : ℕ} (A : Fin K → Matrix r din ℝ)
    (B : Fin K → Matrix dout r ℝ) :
    concatB B * concatA (fun _ => 1) A = ∑ k, B k * A k := by
  simpa using concat_mul (fun _ => (1 : ℝ)) A B

/-- Averaging the factors separately is quadratic in `c` with cross terms. -/
theorem avg_factors_cross_terms {dout din : Type*} {K : ℕ} (c : Fin K → ℝ) (A : Fin K → Matrix r din ℝ)
    (B : Fin K → Matrix dout r ℝ) :
    (∑ k, c k • B k) * (∑ j, c j • A j) = ∑ k, ∑ j, (c k * c j) • (B k * A j) := by
  rw [Matrix.sum_mul]
  refine Finset.sum_congr rfl fun k _ => ?_
  rw [Matrix.mul_sum]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [Matrix.smul_mul, Matrix.mul_smul, smul_smul]

/-- Proposition `prop:wellposed` (a), mixture part: re-gauging bank member `k` by any invertible
`Q_k` leaves `M(c)` unchanged for every `c`. -/
theorem mixture_gauge_invariant {dout din : Type*} {K : ℕ} (c : Fin K → ℝ) (A : Fin K → Matrix r din ℝ)
    (B : Fin K → Matrix dout r ℝ) (Q : Fin K → Matrix r r ℝ) (hQ : ∀ k, IsUnit (Q k).det) :
    ∑ k, c k • ((B k * (Q k)⁻¹) * (Q k * A k)) = ∑ k, c k • (B k * A k) := by
  refine Finset.sum_congr rfl fun k _ => ?_
  rw [gauge_invariance _ _ _ (hQ k)]

/-- Proposition `prop:wellposed` (a), objective part: any functional `F` of the induced per-site
updates (in particular the behaviour-cloning loss, which evaluates the backbone with
`W_s + (α/r) B_s A_s` installed) is invariant under a per-site gauge change. -/
theorem objective_gauge_invariant {Ω : Type*} {dout din : Ω → Type*}
    (F : ((s : Ω) → Matrix (dout s) (din s) ℝ) → ℝ)
    (A : (s : Ω) → Matrix r (din s) ℝ) (B : (s : Ω) → Matrix (dout s) r ℝ)
    (Q : Ω → Matrix r r ℝ) (hQ : ∀ s, IsUnit (Q s).det) :
    F (fun s => (B s * (Q s)⁻¹) * (Q s * A s)) = F (fun s => B s * A s) := by
  congr 1
  funext s
  exact gauge_invariance _ _ _ (hQ s)

/-! ### Proposition `prop:wellposed` (c) -/

/-- Element-wise `ℓ₁` norm. -/
def l1 {p q : Type*} [Fintype p] [Fintype q] (A : Matrix p q ℝ) : ℝ := ∑ i, ∑ j, |A i j|

lemma l1_nonneg {p q : Type*} [Fintype p] [Fintype q] (A : Matrix p q ℝ) : 0 ≤ l1 A :=
  Finset.sum_nonneg fun _ _ => Finset.sum_nonneg fun _ _ => abs_nonneg _

lemma l1_eq_zero_iff {p q : Type*} [Fintype p] [Fintype q] (A : Matrix p q ℝ) :
    l1 A = 0 ↔ A = 0 := by
  constructor
  · intro h
    have h1 := (Finset.sum_eq_zero_iff_of_nonneg
      (fun i _ => Finset.sum_nonneg fun j _ => abs_nonneg (A i j))).1 h
    ext i j
    have h2 := (Finset.sum_eq_zero_iff_of_nonneg (fun j _ => abs_nonneg (A i j))).1
      (h1 i (Finset.mem_univ _)) j (Finset.mem_univ _)
    simpa using h2
  · rintro rfl
    simp [l1]

lemma l1_pos_of_ne_zero {p q : Type*} [Fintype p] [Fintype q] {A : Matrix p q ℝ} (hA : A ≠ 0) :
    0 < l1 A :=
  lt_of_le_of_ne (l1_nonneg A) (Ne.symm (mt (l1_eq_zero_iff A).1 hA))

/-- Reverse triangle inequality, summed: `‖A − t A⋆‖₁ ≥ t ‖A⋆‖₁ − ‖A‖₁` for `t ≥ 0`. -/
lemma l1_sub_smul_ge {p q : Type*} [Fintype p] [Fintype q] (A Astar : Matrix p q ℝ) {t : ℝ}
    (ht : 0 ≤ t) : t * l1 Astar - l1 A ≤ l1 (A - t • Astar) := by
  unfold l1
  rw [Finset.mul_sum, ← Finset.sum_sub_distrib]
  refine Finset.sum_le_sum fun i _ => ?_
  rw [Finset.mul_sum, ← Finset.sum_sub_distrib]
  refine Finset.sum_le_sum fun j _ => ?_
  simp only [Matrix.sub_apply, Matrix.smul_apply, smul_eq_mul]
  calc t * |Astar i j| - |A i j| = |t * Astar i j| - |A i j| := by
        rw [abs_mul, abs_of_nonneg ht]
    _ ≤ |t * Astar i j - A i j| := abs_sub_abs_le_abs_sub _ _
    _ = |A i j - t * Astar i j| := abs_sub_comm _ _

/-- Proposition `prop:wellposed` (c).  With `Q = t I_r`, every re-gauged target
`(t A⋆, B⋆/t)` induces the same update `B⋆ A⋆`, yet the factor-space loss
`‖A − t A⋆‖₁ + ‖B − B⋆/t‖₁` is unbounded in `t` (so not constant in `t`) as soon as `A⋆ ≠ 0`. -/
theorem factor_loss_not_gauge_invariant {dout din : Type*} [Fintype dout] [Fintype din]
    (A Astar : Matrix r din ℝ) (B Bstar : Matrix dout r ℝ)
    (hA : Astar ≠ 0) :
    (∀ t : ℝ, 0 < t → (t⁻¹ • Bstar) * (t • Astar) = Bstar * Astar) ∧
    (∀ M : ℝ, ∃ t : ℝ, 0 < t ∧ M < l1 (A - t • Astar) + l1 (B - t⁻¹ • Bstar)) := by
  refine ⟨fun t ht => ?_, fun M => ?_⟩
  · rw [Matrix.smul_mul, Matrix.mul_smul, smul_smul, inv_mul_cancel₀ ht.ne', one_smul]
  · have hpos : 0 < l1 Astar := l1_pos_of_ne_zero hA
    set t : ℝ := (|M| + l1 A + 1) / l1 Astar with ht
    have htpos : 0 < t := div_pos (by linarith [l1_nonneg A, abs_nonneg M]) hpos
    refine ⟨t, htpos, ?_⟩
    have h1 := l1_sub_smul_ge A Astar htpos.le
    have h2 := l1_nonneg (B - t⁻¹ • Bstar)
    have h3 : t * l1 Astar = |M| + l1 A + 1 := by
      rw [ht]; field_simp
    have h4 := le_abs_self M
    linarith

/-- Consequence: the factor-space loss takes different values at `t = 1` and at some `t > 0`,
although both targets are the same model. -/
theorem factor_loss_not_constant {dout din : Type*} [Fintype dout] [Fintype din]
    (A Astar : Matrix r din ℝ) (B Bstar : Matrix dout r ℝ)
    (hA : Astar ≠ 0) :
    ∃ t : ℝ, 0 < t ∧
      l1 (A - t • Astar) + l1 (B - t⁻¹ • Bstar) ≠ l1 (A - (1 : ℝ) • Astar) + l1 (B - (1 : ℝ)⁻¹ • Bstar) := by
  obtain ⟨t, ht, hlt⟩ := (factor_loss_not_gauge_invariant A Astar B Bstar hA).2
    (l1 (A - (1 : ℝ) • Astar) + l1 (B - (1 : ℝ)⁻¹ • Bstar))
  exact ⟨t, ht, ne_of_gt hlt⟩

end Web2LoRA.Gauge
