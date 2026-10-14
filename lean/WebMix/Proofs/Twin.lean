import WebMix.Statements
import Mathlib.Algebra.Order.BigOperators.Ring.Finset
import Mathlib.Topology.Algebra.InfiniteSum.Real
import Mathlib.Topology.Instances.Matrix
import Mathlib.Topology.MetricSpace.Pseudo.Lemmas

/-!
# Direct generation in a frozen-trunk twin

Proofs for Assumption ass:twin, Proposition prop:twin and the twin half of Corollary cor:scale.

Notation. `Ξ : Matrix n k ℝ` holds the training features `ξ_1, …, ξ_K` as columns, `B : Matrix m k ℝ`
holds the vectorised bank updates `𝓑` as columns, and a head is `Γ : Matrix m n ℝ`. The loss is
`L(Γ) = ‖ΓΞ − 𝓑‖_F²`, and `rank Ξ = K` is injectivity of `y ↦ Ξy`.

Strategy (Appendix D, proof of prop:twin). Full column rank makes `ΞᵀΞ` invertible, so
`Ξ⁺ = (ΞᵀΞ)⁻¹Ξᵀ` satisfies `Ξ⁺Ξ = I` and `Γ̂ = 𝓑Ξ⁺` interpolates; the minimisers of `L` are exactly
the interpolants. `Γ̂` has rows in `span{ξ_k}` while any other interpolant differs from it by `N`
with `NΞ = 0`, so the two are Frobenius-orthogonal and `Γ̂` has the least norm. For gradient descent,
the error `E_t = Γ_t − Γ̂` obeys `E_{t+1} = E_t(I − 2ηΞΞᵀ)`: (i) its rows stay in `span{ξ_k}`;
(ii) each step lowers `‖E_t‖_F²` by at least `4η(1 − ηλ_max)‖E_tΞ‖_F²`; (iii) summing gives
`E_tΞ → 0`; (iv) by (i), `E_t = (E_tΞ)Ξ⁺ → 0`. The output at a new conditioning is
`Γ̂ξ = 𝓑(Ξ⁺ξ)`, and `Ξ⁺ξ_j = e_j` gives the examples of statement (c) and the twin half of
cor:scale by linearity.

Names. `L` in the lemma hypotheses is an upper bound on `λ_max(ΞΞᵀ)` (the loss itself is `loss`),
`η` is the step size the paper calls `γ`, and `beta Ξ ξ = Ξ⁺ξ` is the paper's coefficient map `η(h)`.
-/

open Matrix Finset Filter Topology

namespace WebMix.Twin

variable {m n k : Type*} [Fintype m] [Fintype n] [Fintype k] [DecidableEq k]

/-! ### The Frobenius inner product -/

/-- `⟨A, B⟩_F = tr(ABᵀ)`. -/
lemma finner_eq_trace (A B : Matrix m n ℝ) : finner A B = Matrix.trace (A * Bᵀ) := by
  simp [finner, Matrix.trace, Matrix.mul_apply]

/-- `‖A‖_F² ≥ 0`. -/
lemma frob2_nonneg (A : Matrix m n ℝ) : 0 ≤ frob2 A :=
  Finset.sum_nonneg fun _ _ => Finset.sum_nonneg fun _ _ => mul_self_nonneg _

/-- `‖A‖_F² = 0` iff `A = 0`. -/
lemma frob2_eq_zero_iff (A : Matrix m n ℝ) : frob2 A = 0 ↔ A = 0 := by
  constructor
  · intro h
    have h1 := (Finset.sum_eq_zero_iff_of_nonneg
      (fun i _ => Finset.sum_nonneg fun j _ => mul_self_nonneg (A i j))).1 h
    ext i j
    have h2 := (Finset.sum_eq_zero_iff_of_nonneg (fun j _ => mul_self_nonneg (A i j))).1
      (h1 i (Finset.mem_univ _)) j (Finset.mem_univ _)
    exact mul_self_eq_zero.1 h2
  · rintro rfl
    simp [frob2, finner]

/-- `‖A + B‖_F² = ‖A‖_F² + 2⟨A, B⟩_F + ‖B‖_F²`. -/
lemma frob2_add (A B : Matrix m n ℝ) : frob2 (A + B) = frob2 A + 2 * finner A B + frob2 B := by
  simp only [frob2, finner, Matrix.add_apply, Finset.mul_sum, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => by ring

/-- `‖A − cB‖_F² = ‖A‖_F² − 2c⟨A, B⟩_F + c²‖B‖_F²`. -/
lemma frob2_sub_smul (A B : Matrix m n ℝ) (c : ℝ) :
    frob2 (A - c • B) = frob2 A - 2 * c * finner A B + c ^ 2 * frob2 B := by
  simp only [frob2, finner, Matrix.sub_apply, Matrix.smul_apply, smul_eq_mul, Finset.mul_sum,
    ← Finset.sum_sub_distrib, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => by ring

/-- Each squared entry is at most `‖A‖_F²`. -/
lemma entry_sq_le_frob2 (A : Matrix m n ℝ) (i : m) (j : n) : A i j ^ 2 ≤ frob2 A := by
  unfold frob2 finner
  calc A i j ^ 2 = A i j * A i j := sq _
    _ ≤ ∑ j', A i j' * A i j' :=
        Finset.single_le_sum (fun j' _ => mul_self_nonneg (A i j')) (Finset.mem_univ j)
    _ ≤ ∑ i', ∑ j', A i' j' * A i' j' :=
        Finset.single_le_sum (f := fun i' => ∑ j', A i' j' * A i' j')
          (fun i' _ => Finset.sum_nonneg fun j' _ => mul_self_nonneg _) (Finset.mem_univ i)

/-- `⟨E, HΞ⟩_F = ⟨EΞᵀ, H⟩_F`. -/
lemma finner_mul_right {p q : Type*} [Fintype p] [Fintype q] (E : Matrix m q ℝ)
    (H : Matrix m p ℝ) (Ξ : Matrix p q ℝ) : finner E (H * Ξ) = finner (E * Ξᵀ) H := by
  rw [finner_eq_trace, finner_eq_trace, Matrix.transpose_mul, ← Matrix.mul_assoc]

/-- Cauchy–Schwarz for the Frobenius inner product. -/
lemma finner_sq_le (A B : Matrix m n ℝ) : finner A B ^ 2 ≤ frob2 A * frob2 B := by
  have h := Finset.sum_mul_sq_le_sq_mul_sq (Finset.univ : Finset (m × n))
    (fun p => A p.1 p.2) (fun p => B p.1 p.2)
  simpa [finner, frob2, Fintype.sum_prod_type, sq] using h

omit [Fintype m] in
/-- Rows of a product: `(W Ξ)_i = W_i Ξ`. -/
lemma mul_row {p : Type*} [Fintype p] (Wm : Matrix m n ℝ) (Ξ : Matrix n p ℝ) (i : m) :
    (Wm * Ξ) i = Wm i ᵥ* Ξ := by
  funext j
  simp [Matrix.mul_apply, Matrix.vecMul, dotProduct]

/-- `‖A‖_F²` is the sum of the squared row norms. -/
lemma frob2_eq_sum_dot (A : Matrix m n ℝ) : frob2 A = ∑ i, A i ⬝ᵥ A i := by
  simp [frob2, finner, dotProduct]

/-! ### Interpolation and the minimum-norm head -/

variable (Ξ : Matrix n k ℝ) (B : Matrix m k ℝ)

/-- Under `rank Ξ = K`, the Gram matrix `ΞᵀΞ` is invertible. -/
lemma gram_isUnit_det (h : FullColumnRank Ξ) : IsUnit (gram Ξ).det := by
  -- a square matrix is invertible iff `y ↦ (ΞᵀΞ)y` is injective
  rw [← Matrix.isUnit_iff_isUnit_det, ← Matrix.mulVec_injective_iff_isUnit]
  intro y z hyz
  have h0 : gram Ξ *ᵥ (y - z) = 0 := by rw [Matrix.mulVec_sub, hyz, sub_self]
  -- `(y − z)ᵀ ΞᵀΞ (y − z) = (Ξ(y − z))ᵀ Ξ(y − z) = ‖Ξ(y − z)‖²`, and it is `0`
  have h1 : (Ξ *ᵥ (y - z)) ⬝ᵥ (Ξ *ᵥ (y - z)) = 0 := by
    have : (y - z) ⬝ᵥ (gram Ξ *ᵥ (y - z)) = (Ξ *ᵥ (y - z)) ⬝ᵥ (Ξ *ᵥ (y - z)) := by
      rw [gram, ← Matrix.mulVec_mulVec, Matrix.dotProduct_mulVec (y - z) Ξᵀ (Ξ *ᵥ (y - z)),
        Matrix.vecMul_transpose Ξ (y - z)]
    rw [← this, h0, dotProduct_zero]
  -- so `Ξ(y − z) = 0`, and `rank Ξ = K` gives `y = z`
  have h2 : Ξ *ᵥ (y - z) = 0 := dotProduct_self_eq_zero.1 h1
  have h3 : Ξ *ᵥ (y - z) = Ξ *ᵥ 0 := by rw [h2, Matrix.mulVec_zero]
  exact sub_eq_zero.1 (h h3)

/-- `Ξ⁺Ξ = I`. -/
lemma pinv_mul (h : FullColumnRank Ξ) : pinv Ξ * Ξ = 1 := by
  rw [pinv, Matrix.mul_assoc]
  exact Matrix.nonsing_inv_mul _ (gram_isUnit_det Ξ h)

omit [Fintype m] in
/-- `Γ̂Ξ = 𝓑`: the head `Γ̂ = 𝓑Ξ⁺` interpolates the bank. -/
lemma Γhat_mul (h : FullColumnRank Ξ) : Γhat Ξ B * Ξ = B := by
  rw [Γhat, Matrix.mul_assoc, pinv_mul Ξ h, Matrix.mul_one]

omit [DecidableEq k] in
/-- `L ≥ 0`. -/
lemma loss_nonneg (Γ : Matrix m n ℝ) : 0 ≤ loss Ξ B Γ := frob2_nonneg _

omit [DecidableEq k] in
/-- `L(Γ) = 0` iff `Γ` interpolates the bank, `ΓΞ = 𝓑`. -/
lemma loss_eq_zero_iff (Γ : Matrix m n ℝ) : loss Ξ B Γ = 0 ↔ Γ * Ξ = B := by
  rw [loss, frob2_eq_zero_iff, sub_eq_zero]

/-- `L(Γ̂) = 0`. -/
lemma loss_Γhat (h : FullColumnRank Ξ) : loss Ξ B (Γhat Ξ B) = 0 :=
  (loss_eq_zero_iff Ξ B _).2 (Γhat_mul Ξ B h)

/-- The minimiser set of `L` is the affine space `{Γ : ΓΞ = 𝓑}` of interpolants (`L ≥ 0` and
`L(Γ̂) = 0`). -/
theorem isMinOn_loss_iff (h : FullColumnRank Ξ) (Γ : Matrix m n ℝ) :
    IsMinOn (loss Ξ B) Set.univ Γ ↔ Γ * Ξ = B := by
  constructor
  · intro hmin
    have := isMinOn_iff.1 hmin _ (Set.mem_univ (Γhat Ξ B))
    rw [loss_Γhat Ξ B h] at this
    exact (loss_eq_zero_iff Ξ B Γ).1 (le_antisymm this (loss_nonneg Ξ B Γ))
  · intro hΓ
    rw [isMinOn_iff]
    intro x _
    rw [(loss_eq_zero_iff Ξ B Γ).2 hΓ]
    exact loss_nonneg Ξ B x

omit [DecidableEq k] in
/-- The gradient: `L(Γ + H) = L(Γ) + ⟨2(ΓΞ − 𝓑)Ξᵀ, H⟩_F + ‖HΞ‖_F²`, so `∇L(Γ) = 2(ΓΞ − 𝓑)Ξᵀ`. -/
theorem loss_expand (Γ H : Matrix m n ℝ) :
    loss Ξ B (Γ + H) = loss Ξ B Γ + finner ((2 : ℝ) • ((Γ * Ξ - B) * Ξᵀ)) H + frob2 (H * Ξ) := by
  have h1 : (Γ + H) * Ξ - B = (Γ * Ξ - B) + H * Ξ := by rw [Matrix.add_mul]; abel
  have h2 : finner ((2 : ℝ) • ((Γ * Ξ - B) * Ξᵀ)) H = 2 * finner ((Γ * Ξ - B) * Ξᵀ) H := by
    simp [finner, Finset.mul_sum, mul_assoc]
  rw [loss, h1, frob2_add, finner_mul_right, h2]
  -- what remains is `frob2 (ΓΞ − 𝓑) = loss Ξ 𝓑 Γ`, the definition of `loss`
  rfl

/-- `Γ̂` is the unique minimum-norm interpolant: `Γ = Γ̂ + N` with `NΞ = 0`, the rows of `Γ̂` lie in
`span{ξ_k}`, so `⟨Γ̂, N⟩_F = 0` and `‖Γ‖_F² = ‖Γ̂‖_F² + ‖N‖_F²`. -/
theorem Γhat_min_norm (h : FullColumnRank Ξ) (Γ : Matrix m n ℝ) (hΓ : Γ * Ξ = B) :
    frob2 (Γhat Ξ B) ≤ frob2 Γ ∧ (frob2 Γ = frob2 (Γhat Ξ B) → Γ = Γhat Ξ B) := by
  set N := Γ - Γhat Ξ B with hN
  have hNΞ : N * Ξ = 0 := by rw [hN, Matrix.sub_mul, hΓ, Γhat_mul Ξ B h, sub_self]
  have hΓ' : Γ = Γhat Ξ B + N := by rw [hN]; abel
  have hΞN : Ξᵀ * Nᵀ = 0 := by rw [← Matrix.transpose_mul, hNΞ, Matrix.transpose_zero]
  -- `⟨Γ̂, N⟩_F = tr(𝓑(ΞᵀΞ)⁻¹ΞᵀNᵀ) = 0`
  have hinner : finner (Γhat Ξ B) N = 0 := by
    rw [finner_eq_trace, Γhat, pinv, Matrix.mul_assoc, Matrix.mul_assoc, hΞN, Matrix.mul_zero,
      Matrix.mul_zero, Matrix.trace_zero]
  rw [hΓ', frob2_add, hinner]
  constructor
  · linarith [frob2_nonneg N]
  · intro heq
    have : frob2 N = 0 := by linarith
    rw [(frob2_eq_zero_iff N).1 this, add_zero]

/-! ### Gradient descent -/

omit [Fintype m] [DecidableEq k] in
/-- `Γ_0 = 0`. -/
lemma gd_zero (η : ℝ) : gd Ξ B η 0 = 0 := rfl

omit [Fintype m] [DecidableEq k] in
/-- `Γ_{t+1}` is one gradient step from `Γ_t`. -/
lemma gd_succ (η : ℝ) (t : ℕ) : gd Ξ B η (t + 1) = gdStep Ξ B η (gd Ξ B η t) := by
  rw [gd, Function.iterate_succ_apply', ← gd]

omit [Fintype m] in
/-- The error recursion: `Γ_{t+1} − Γ̂ = (Γ_t − Γ̂)(I − 2ηΞΞᵀ)`, because `Γ̂Ξ = 𝓑`. -/
lemma gdStep_sub_Γhat [DecidableEq n] (h : FullColumnRank Ξ) (η : ℝ) (Γ : Matrix m n ℝ) :
    gdStep Ξ B η Γ - Γhat Ξ B = (Γ - Γhat Ξ B) * (1 - (2 * η) • (Ξ * Ξᵀ)) := by
  have : Γ * Ξ - B = (Γ - Γhat Ξ B) * Ξ := by rw [Matrix.sub_mul, Γhat_mul Ξ B h]
  rw [gdStep, this, Matrix.mul_sub, Matrix.mul_one, Matrix.mul_smul, Matrix.mul_assoc]
  abel

/-- `P := ΞΞ⁺ = Ξ(ΞᵀΞ)⁻¹Ξᵀ`, the orthogonal projector onto `span{ξ_1, …, ξ_K}`. -/
noncomputable def proj : Matrix n n ℝ := Ξ * pinv Ξ

omit [Fintype m] in
/-- `Γ̂P = Γ̂`: the rows of `Γ̂` lie in `span{ξ_k}`. -/
lemma Γhat_mul_proj (h : FullColumnRank Ξ) : Γhat Ξ B * proj Ξ = Γhat Ξ B := by
  rw [Γhat, proj, Matrix.mul_assoc, ← Matrix.mul_assoc (pinv Ξ), pinv_mul Ξ h, Matrix.one_mul]

/-- `ΞΞᵀP = ΞΞᵀ`. -/
lemma S_mul_proj (h : FullColumnRank Ξ) : (Ξ * Ξᵀ) * proj Ξ = Ξ * Ξᵀ := by
  -- `ΞΞᵀ · Ξ(ΞᵀΞ)⁻¹Ξᵀ = Ξ (ΞᵀΞ)(ΞᵀΞ)⁻¹ Ξᵀ = ΞΞᵀ`: unfold `proj`, `pinv`, `gram` and cancel
  show Ξ * Ξᵀ * (Ξ * ((gram Ξ)⁻¹ * Ξᵀ)) = Ξ * Ξᵀ
  rw [Matrix.mul_assoc, ← Matrix.mul_assoc Ξᵀ Ξ]
  change Ξ * (gram Ξ * ((gram Ξ)⁻¹ * Ξᵀ)) = Ξ * Ξᵀ
  rw [Matrix.mul_nonsing_inv_cancel_left _ _ (gram_isUnit_det Ξ h)]

omit [Fintype m] in
/-- One error step preserves `EP = E` (rows in `span{ξ_k}`). -/
lemma mul_step_mul_proj [DecidableEq n] (h : FullColumnRank Ξ) (η : ℝ) (E : Matrix m n ℝ)
    (hE : E * proj Ξ = E) :
    E * (1 - (2 * η) • (Ξ * Ξᵀ)) * proj Ξ = E * (1 - (2 * η) • (Ξ * Ξᵀ)) := by
  calc E * (1 - (2 * η) • (Ξ * Ξᵀ)) * proj Ξ
      = E * ((1 - (2 * η) • (Ξ * Ξᵀ)) * proj Ξ) := Matrix.mul_assoc _ _ _
    _ = E * (proj Ξ - (2 * η) • (Ξ * Ξᵀ)) := by
        rw [Matrix.sub_mul, Matrix.one_mul, Matrix.smul_mul, S_mul_proj Ξ h]
    _ = E * proj Ξ - (2 * η) • (E * (Ξ * Ξᵀ)) := by rw [Matrix.mul_sub, Matrix.mul_smul]
    _ = E - (2 * η) • (E * (Ξ * Ξᵀ)) := by rw [hE]
    _ = E * (1 - (2 * η) • (Ξ * Ξᵀ)) := by rw [Matrix.mul_sub, Matrix.mul_one, Matrix.mul_smul]

omit [Fintype m] in
/-- Step (i) of the proof of prop:twin: every error `E_t = Γ_t − Γ̂` satisfies `E_tP = E_t`. It
holds at `t = 0` (`E_0 = −Γ̂` and `Γ̂P = Γ̂`) and is preserved by each step. -/
lemma err_mul_proj (h : FullColumnRank Ξ) (η : ℝ) :
    ∀ t, (gd Ξ B η t - Γhat Ξ B) * proj Ξ = gd Ξ B η t - Γhat Ξ B := by
  classical
  intro t
  induction t with
  | zero =>
    rw [gd_zero, zero_sub, Matrix.neg_mul, Γhat_mul_proj Ξ B h]
  | succ t ih =>
    rw [gd_succ, gdStep_sub_Γhat Ξ B h]
    exact mul_step_mul_proj Ξ h η _ ih

omit [Fintype m] in
/-- Step (iv) of the proof of prop:twin: by step (i), `E_t = E_tP = (E_tΞ)Ξ⁺`. -/
lemma err_eq (h : FullColumnRank Ξ) (η : ℝ) (t : ℕ) :
    gd Ξ B η t - Γhat Ξ B = ((gd Ξ B η t - Γhat Ξ B) * Ξ) * pinv Ξ := by
  conv_lhs => rw [← err_mul_proj Ξ B h η t]
  rw [proj, Matrix.mul_assoc]

omit [DecidableEq k] in
/-- The Rayleigh bound `λ_max(ΞΞᵀ) ≤ L` in Frobenius form: `‖WΞ‖_F² ≤ L ‖W‖_F²`. -/
lemma frob2_mul_le {L : ℝ} (hL : ∀ w : n → ℝ, (w ᵥ* Ξ) ⬝ᵥ (w ᵥ* Ξ) ≤ L * (w ⬝ᵥ w))
    (Wm : Matrix m n ℝ) : frob2 (Wm * Ξ) ≤ L * frob2 Wm := by
  rw [frob2_eq_sum_dot, frob2_eq_sum_dot, Finset.mul_sum]
  refine Finset.sum_le_sum fun i _ => ?_
  rw [mul_row]
  exact hL _

omit [DecidableEq k] in
/-- The same bound for the transposed factor, `‖ZΞᵀ‖_F² ≤ L ‖Z‖_F²` (`λ_max(ΞᵀΞ) = λ_max(ΞΞᵀ)`),
via Cauchy–Schwarz: `‖ZΞᵀ‖_F⁴ = ⟨ZΞᵀΞ, Z⟩_F² ≤ ‖ZΞᵀΞ‖_F² ‖Z‖_F² ≤ L ‖ZΞᵀ‖_F² ‖Z‖_F²`. -/
lemma frob2_mul_transpose_le {L : ℝ} (hL0 : 0 ≤ L)
    (hL : ∀ w : n → ℝ, (w ᵥ* Ξ) ⬝ᵥ (w ᵥ* Ξ) ≤ L * (w ⬝ᵥ w))
    (Z : Matrix m k ℝ) : frob2 (Z * Ξᵀ) ≤ L * frob2 Z := by
  set a := frob2 (Z * Ξᵀ) with ha
  set b := frob2 Z with hb
  have ha0 : 0 ≤ a := frob2_nonneg _
  have hb0 : 0 ≤ b := frob2_nonneg _
  have h1 : a = finner ((Z * Ξᵀ) * Ξ) Z := by
    rw [ha, frob2, finner_mul_right, Matrix.transpose_transpose]
  have h2 : finner ((Z * Ξᵀ) * Ξ) Z ^ 2 ≤ frob2 ((Z * Ξᵀ) * Ξ) * b := finner_sq_le _ _
  have h3 : frob2 ((Z * Ξᵀ) * Ξ) ≤ L * a := frob2_mul_le Ξ hL _
  have h4 : a ^ 2 ≤ L * a * b := by
    calc a ^ 2 = finner ((Z * Ξᵀ) * Ξ) Z ^ 2 := by rw [h1]
      _ ≤ frob2 ((Z * Ξᵀ) * Ξ) * b := h2
      _ ≤ L * a * b := mul_le_mul_of_nonneg_right h3 hb0
  -- divide `a² ≤ L a b` by `a > 0`; the case `a = 0` is immediate
  rcases ha0.lt_or_eq with hpos | hzero
  · nlinarith
  · rw [← hzero]; positivity

omit [DecidableEq k] in
/-- Step (ii) of the proof of prop:twin, the display of the paper:
`‖E(I − 2ηΞΞᵀ)‖_F² = ‖E‖_F² − 4η‖EΞ‖_F² + 4η²‖EΞΞᵀ‖_F² ≤ ‖E‖_F² − 4η(1 − ηL)‖EΞ‖_F²`. -/
lemma frob2_step_le [DecidableEq n] {L : ℝ} (hL0 : 0 ≤ L)
    (hL : ∀ w : n → ℝ, (w ᵥ* Ξ) ⬝ᵥ (w ᵥ* Ξ) ≤ L * (w ⬝ᵥ w)) {η : ℝ}
    (E : Matrix m n ℝ) :
    frob2 (E * (1 - (2 * η) • (Ξ * Ξᵀ))) ≤ frob2 E - 4 * η * (1 - η * L) * frob2 (E * Ξ) := by
  -- `E(I − 2ηΞΞᵀ) = E − 2η (EΞ)Ξᵀ`
  have hsplit : E * (1 - (2 * η) • (Ξ * Ξᵀ)) = E - (2 * η) • ((E * Ξ) * Ξᵀ) := by
    rw [Matrix.mul_sub, Matrix.mul_one, Matrix.mul_smul, Matrix.mul_assoc]
  -- `⟨E, (EΞ)Ξᵀ⟩_F = ‖EΞ‖_F²`
  have hcross : finner E ((E * Ξ) * Ξᵀ) = frob2 (E * Ξ) := by
    rw [finner_mul_right, Matrix.transpose_transpose]; rfl
  -- `‖(EΞ)Ξᵀ‖_F² ≤ L ‖EΞ‖_F²`
  have hray : frob2 ((E * Ξ) * Ξᵀ) ≤ L * frob2 (E * Ξ) := frob2_mul_transpose_le Ξ hL0 hL _
  calc frob2 (E * (1 - (2 * η) • (Ξ * Ξᵀ)))
      = frob2 E - 2 * (2 * η) * finner E ((E * Ξ) * Ξᵀ)
          + (2 * η) ^ 2 * frob2 ((E * Ξ) * Ξᵀ) := by rw [hsplit, frob2_sub_smul]
    _ = frob2 E - 4 * η * frob2 (E * Ξ) + 4 * η ^ 2 * frob2 ((E * Ξ) * Ξᵀ) := by
        rw [hcross]; ring
    _ ≤ frob2 E - 4 * η * frob2 (E * Ξ) + 4 * η ^ 2 * (L * frob2 (E * Ξ)) := by gcongr
    _ = frob2 E - 4 * η * (1 - η * L) * frob2 (E * Ξ) := by ring

/-- Entrywise convergence from convergence of the squared Frobenius norm. -/
lemma tendsto_of_frob2 (F : ℕ → Matrix m n ℝ) (hF : Tendsto (fun t => frob2 (F t)) atTop (𝓝 0)) :
    Tendsto F atTop (𝓝 0) := by
  -- convergence of matrices is entrywise; each squared entry is squeezed by `‖F t‖_F²`
  refine tendsto_pi_nhds.2 fun i => tendsto_pi_nhds.2 fun j => ?_
  have h1 : Tendsto (fun t => (F t i j) ^ 2) atTop (𝓝 0) :=
    squeeze_zero (fun t => sq_nonneg _) (fun t => entry_sq_le_frob2 (F t) i j) hF
  -- take square roots: `√(x²) = |x|`
  have h2 : Tendsto (fun t => ‖F t i j‖) atTop (𝓝 0) := by
    have := h1.sqrt
    simpa [Real.sqrt_sq_eq_abs, Real.norm_eq_abs] using this
  simpa using tendsto_zero_iff_norm_tendsto_zero.2 h2

/-- Steps (ii) and (iii) of the proof of prop:twin: each step lowers `‖E_t‖_F²` by at least
`κ‖E_tΞ‖_F²` with `κ = 4η(1 − ηL) > 0`, so `∑_t κ‖E_tΞ‖_F² ≤ ‖E_0‖_F² < ∞` and `E_tΞ → 0`. -/
lemma tendsto_err_mul (h : FullColumnRank Ξ) {L : ℝ} (hL0 : 0 ≤ L)
    (hL : ∀ w : n → ℝ, (w ᵥ* Ξ) ⬝ᵥ (w ᵥ* Ξ) ≤ L * (w ⬝ᵥ w)) {η : ℝ} (hη : 0 < η)
    (hηL : η * L < 1) :
    Tendsto (fun t => (gd Ξ B η t - Γhat Ξ B) * Ξ) atTop (𝓝 0) := by
  classical
  set E : ℕ → Matrix m n ℝ := fun t => gd Ξ B η t - Γhat Ξ B with hE
  set κ : ℝ := 4 * η * (1 - η * L) with hκ
  have hκpos : 0 < κ := mul_pos (mul_pos four_pos hη) (by linarith)
  -- step (ii) along the iterates, through the error recursion
  have hstep : ∀ t, frob2 (E (t + 1)) ≤ frob2 (E t) - κ * frob2 (E t * Ξ) := fun t => by
    have hrec : E (t + 1) = E t * (1 - (2 * η) • (Ξ * Ξᵀ)) := by
      simp only [hE]
      rw [gd_succ, gdStep_sub_Γhat Ξ B h]
    rw [hrec]
    exact frob2_step_le Ξ hL0 hL (E t)
  -- step (iii): the partial sums telescope
  have hpartial : ∀ N, ∑ t ∈ Finset.range N, κ * frob2 (E t * Ξ) + frob2 (E N) ≤ frob2 (E 0) := by
    intro N
    induction N with
    | zero => simp
    | succ N ih =>
      rw [Finset.sum_range_succ]
      have := hstep N
      linarith
  -- nonnegative terms with bounded partial sums are summable
  have hsummable : Summable fun t => κ * frob2 (E t * Ξ) :=
    summable_of_sum_range_le (c := frob2 (E 0)) (fun t => mul_nonneg hκpos.le (frob2_nonneg _))
      (fun N => by linarith [hpartial N, frob2_nonneg (E N)])
  -- the terms of a convergent series tend to `0`; divide by `κ > 0`
  have hsq : Tendsto (fun t => frob2 (E t * Ξ)) atTop (𝓝 0) := by
    have := hsummable.tendsto_atTop_zero.const_mul κ⁻¹
    simpa [← mul_assoc, inv_mul_cancel₀ hκpos.ne'] using this
  exact tendsto_of_frob2 _ hsq

/-- **Proposition prop:twin, gradient-descent part.** Under `rank Ξ = K`, with
`λ_max(ΞΞᵀ) ≤ L` and step `0 < η < 1/L`, gradient descent from `Γ_0 = 0` converges to
`Γ̂ = 𝓑Ξ⁺`: by steps (ii)–(iii) `E_tΞ → 0`, and by step (iv) `E_t = (E_tΞ)Ξ⁺ → 0`. -/
theorem gd_tendsto (h : FullColumnRank Ξ) {L : ℝ} (hL0 : 0 ≤ L)
    (hL : ∀ w : n → ℝ, (w ᵥ* Ξ) ⬝ᵥ (w ᵥ* Ξ) ≤ L * (w ⬝ᵥ w)) {η : ℝ} (hη : 0 < η)
    (hηL : η * L < 1) :
    Tendsto (gd Ξ B η) atTop (𝓝 (Γhat Ξ B)) := by
  have hmul := tendsto_err_mul Ξ B h hL0 hL hη hηL
  -- step (iv): multiply by the fixed matrix `Ξ⁺` and use `E_t = (E_tΞ)Ξ⁺`
  have hpinv : Tendsto (fun t => ((gd Ξ B η t - Γhat Ξ B) * Ξ) * pinv Ξ) atTop (𝓝 0) := by
    have hc : Continuous fun X : Matrix m k ℝ => X * pinv Ξ :=
      continuous_id.matrix_mul continuous_const
    have := (hc.tendsto 0).comp hmul
    rw [Matrix.zero_mul] at this
    exact this
  have herr : Tendsto (fun t => gd Ξ B η t - Γhat Ξ B) atTop (𝓝 0) :=
    hpinv.congr fun t => (err_eq Ξ B h η t).symm
  have hlim := herr.add_const (Γhat Ξ B)
  rw [zero_add] at hlim
  exact hlim.congr fun t => sub_add_cancel _ _

/-! ### The output at a new conditioning -/

omit [Fintype m] in
/-- Eq. prop:twin: `vec ΔW_{Γ̂}(h) = Γ̂ξ(h) = 𝓑 (Ξ⁺ ξ(h)) = 𝓑 η(h)`. -/
theorem output_Γhat (ξ : n → ℝ) : output (Γhat Ξ B) ξ = B *ᵥ beta Ξ ξ := by
  simp [output, Γhat, beta, Matrix.mulVec_mulVec]

omit [Fintype m] [DecidableEq k] in
/-- `𝓑 β = ∑_k β_k vec ΔW^{(k)}` (the columns of `𝓑`). -/
lemma mulVec_eq_sum_cols (β : k → ℝ) : B *ᵥ β = ∑ j, β j • Bᵀ j := by
  ext i
  simp [Matrix.mulVec, dotProduct, Finset.sum_apply, mul_comm]

omit [Fintype m] in
/-- Statement (a): the output at any conditioning lies in the linear span of the bank. -/
theorem output_mem_span (ξ : n → ℝ) :
    output (Γhat Ξ B) ξ ∈ Submodule.span ℝ (Set.range Bᵀ) := by
  rw [output_Γhat, mulVec_eq_sum_cols]
  exact Submodule.sum_mem _ fun j _ =>
    Submodule.smul_mem _ _ (Submodule.subset_span (Set.mem_range_self j))

/-- The `j`-th column of `M` is `M e_j`. -/
lemma col_eq_mulVec_single {p : Type*} (M : Matrix p k ℝ) (j : k) :
    Mᵀ j = M *ᵥ Pi.single j 1 := by
  ext i
  simp [Matrix.mulVec, dotProduct, Pi.single_apply]

/-- `η(ξ_j) = Ξ⁺Ξe_j = e_j`. -/
lemma beta_col (h : FullColumnRank Ξ) (j : k) : beta Ξ (Ξᵀ j) = Pi.single j 1 := by
  rw [beta, col_eq_mulVec_single Ξ, Matrix.mulVec_mulVec, pinv_mul Ξ h, Matrix.one_mulVec]

/-- Statement (c): `η(ξ_1 + ξ_2) = e_1 + e_2` by linearity. -/
theorem beta_add_cols (h : FullColumnRank Ξ) (j₁ j₂ : k) :
    beta Ξ (Ξᵀ j₁ + Ξᵀ j₂) = Pi.single j₁ 1 + Pi.single j₂ 1 := by
  rw [beta, Matrix.mulVec_add, ← beta, ← beta, beta_col Ξ h, beta_col Ξ h]

/-- Statement (c): `η(ξ_1 − ξ_2) = e_1 − e_2` by linearity. -/
theorem beta_sub_cols (h : FullColumnRank Ξ) (j₁ j₂ : k) :
    beta Ξ (Ξᵀ j₁ - Ξᵀ j₂) = Pi.single j₁ 1 - Pi.single j₂ 1 := by
  rw [beta, Matrix.mulVec_sub, ← beta, ← beta, beta_col Ξ h, beta_col Ξ h]

omit [Fintype m] in
/-- The corresponding outputs: `ΔW^{(1)} + ΔW^{(2)}`. -/
theorem output_add_cols (h : FullColumnRank Ξ) (j₁ j₂ : k) :
    output (Γhat Ξ B) (Ξᵀ j₁ + Ξᵀ j₂) = Bᵀ j₁ + Bᵀ j₂ := by
  rw [output_Γhat, beta_add_cols Ξ h, Matrix.mulVec_add, ← col_eq_mulVec_single B,
    ← col_eq_mulVec_single B]

omit [Fintype m] in
/-- … and `ΔW^{(1)} − ΔW^{(2)}`. -/
theorem output_sub_cols (h : FullColumnRank Ξ) (j₁ j₂ : k) :
    output (Γhat Ξ B) (Ξᵀ j₁ - Ξᵀ j₂) = Bᵀ j₁ - Bᵀ j₂ := by
  rw [output_Γhat, beta_sub_cols Ξ h, Matrix.mulVec_sub, ← col_eq_mulVec_single B,
    ← col_eq_mulVec_single B]

omit [Fintype m] [Fintype k] [DecidableEq k] in
/-- `ΔW^{(1)} + ΔW^{(2)}` is twice the mean of the two. -/
theorem add_cols_eq_two_smul_mean (j₁ j₂ : k) :
    Bᵀ j₁ + Bᵀ j₂ = (2 : ℝ) • ((1 / 2 : ℝ) • (Bᵀ j₁ + Bᵀ j₂)) := by
  rw [smul_smul]; norm_num

/-- `e_1 + e_2 ∉ Δ^{K-1}`: its coordinates sum to `2`. -/
theorem beta_add_cols_not_mem_simplex (h : FullColumnRank Ξ) (j₁ j₂ : k) :
    beta Ξ (Ξᵀ j₁ + Ξᵀ j₂) ∉ stdSimplex ℝ k := by
  rw [beta_add_cols Ξ h]
  rintro ⟨-, hsum⟩
  simp [Finset.sum_add_distrib, Pi.single_apply] at hsum

/-- `e_1 − e_2 ∉ Δ^{K-1}`: coordinate `j₂` is `−1`. -/
theorem beta_sub_cols_not_mem_simplex (h : FullColumnRank Ξ) {j₁ j₂ : k} (hj : j₁ ≠ j₂) :
    beta Ξ (Ξᵀ j₁ - Ξᵀ j₂) ∉ stdSimplex ℝ k := by
  rw [beta_sub_cols Ξ h]
  rintro ⟨hnn, -⟩
  have := hnn j₂
  rw [Pi.sub_apply, Pi.single_eq_of_ne hj.symm, Pi.single_eq_same] at this
  norm_num at this

/-! ### Corollary cor:scale, twin half -/

/-- `η(t ξ_j) = t e_j`. -/
theorem beta_smul_col (h : FullColumnRank Ξ) (t : ℝ) (j : k) :
    beta Ξ (t • Ξᵀ j) = t • Pi.single j 1 := by
  rw [beta, Matrix.mulVec_smul, ← beta, beta_col Ξ h]

omit [Fintype m] in
/-- The feature vector `ξ(h) = t ξ_j` yields the output `t ΔW^{(j)}`. -/
theorem output_smul_col (h : FullColumnRank Ξ) (t : ℝ) (j : k) :
    output (Γhat Ξ B) (t • Ξᵀ j) = t • Bᵀ j := by
  rw [output_Γhat, beta_smul_col Ξ h, Matrix.mulVec_smul, ← col_eq_mulVec_single B]

/-- `‖t v‖₂ = |t| ‖v‖₂`. -/
lemma eucNorm_smul {p : Type*} [Fintype p] (t : ℝ) (v : p → ℝ) :
    eucNorm (t • v) = |t| * eucNorm v := by
  have h : (t • v) ⬝ᵥ (t • v) = t ^ 2 * (v ⬝ᵥ v) := by
    rw [smul_dotProduct, dotProduct_smul, smul_eq_mul, smul_eq_mul]; ring
  rw [eucNorm, eucNorm, h, Real.sqrt_mul (sq_nonneg t), Real.sqrt_sq_eq_abs]

/-- `‖v‖₂ > 0` for `v ≠ 0`. -/
lemma eucNorm_pos {p : Type*} [Fintype p] {v : p → ℝ} (hv : v ≠ 0) : 0 < eucNorm v := by
  rw [eucNorm]
  apply Real.sqrt_pos.2
  have hnn : 0 ≤ v ⬝ᵥ v := Finset.sum_nonneg fun i _ => mul_self_nonneg (v i)
  rcases hnn.lt_or_eq with hpos | hzero
  · exact hpos
  · exact absurd (dotProduct_self_eq_zero.1 hzero.symm) hv

/-- No scale bound holds in the twin over feature vectors: for `ΔW^{(j)} ≠ 0`, the Euclidean norm
of the output at `ξ(h) = t ξ_j`, which is `t ‖ΔW^{(j)}‖`, exceeds any `M` for some `t > 0`. -/
theorem output_unbounded_euc (h : FullColumnRank Ξ) (j : k) (hj : Bᵀ j ≠ 0) (M : ℝ) :
    ∃ t : ℝ, 0 < t ∧ M < eucNorm (output (Γhat Ξ B) (t • Ξᵀ j)) := by
  have hnorm : 0 < eucNorm (Bᵀ j) := eucNorm_pos hj
  have ht : 0 < (|M| + 1) / eucNorm (Bᵀ j) := div_pos (by positivity) hnorm
  refine ⟨(|M| + 1) / eucNorm (Bᵀ j), ht, ?_⟩
  rw [output_smul_col Ξ B h, eucNorm_smul, abs_of_pos ht, div_mul_cancel₀ _ hnorm.ne']
  linarith [le_abs_self M]

end WebMix.Twin
