import Mathlib

/-!
# Direct generation in a frozen-trunk twin

Formalisation of Assumption `ass:twin`, Proposition `prop:twin` and the twin half of
Corollary `cor:scale`.

Notation.  `Ξ : Matrix n k ℝ` holds the training features `ξ_1, …, ξ_K` as columns
(`n = ℝ^{d_ξ}`, `k = {1,…,K}`), `B : Matrix m k ℝ` holds the vectorised oracle updates as
columns (`m = ℝ^{D_Ω}`), and a head is `Γ : Matrix m n ℝ`.  The loss is
`L(Γ) = ‖ΓΞ − B‖_F²`.  `rank Ξ = K` is expressed as injectivity of `y ↦ Ξ y`
(`FullColumnRank`).

Results.
* `gram_isUnit_det`: `ΞᵀΞ` is invertible;  `pinv_mul`: `Ξ⁺Ξ = I`.
* `isMinOn_loss_iff`: the minimiser set of `L` is `{Γ : ΓΞ = B}`; `Γ̂ = BΞ⁺` is one of them.
* `loss_expand`: `L(Γ + H) = L(Γ) + ⟨2(ΓΞ − B)Ξᵀ, H⟩_F + ‖HΞ‖_F²`, certifying the gradient.
* `Γhat_min_norm`: `Γ̂` is the unique minimum-Frobenius-norm minimiser.
* `gd_tendsto`: gradient descent from `Γ = 0` with step `η ∈ (0, 1/λ_max(ΞΞᵀ))` converges
  to `Γ̂`.  The hypothesis `hL` says `λ_max(ΞΞᵀ) ≤ L` (Rayleigh quotient form).
* `output_Γhat`, `output_mem_span`: Eq. prop:twin and statement (a).
* `beta_add_cols`, `beta_sub_cols`, `beta_add_cols_not_mem_simplex`, …: statement (c).
* `output_smul_col`, `output_unbounded`: the twin half of Corollary `cor:scale`.
-/

open Matrix Finset Filter Topology

namespace Web2LoRA.Twin

variable {m n k : Type*} [Fintype m] [Fintype n] [Fintype k] [DecidableEq k]

/-! ### Frobenius inner product -/

/-- Frobenius inner product `⟨A, B⟩_F = ∑_{ij} A_{ij} B_{ij} = tr(A Bᵀ)`. -/
def finner (A B : Matrix m n ℝ) : ℝ := ∑ i, ∑ j, A i j * B i j

/-- Squared Frobenius norm `‖A‖_F²`. -/
def frob2 (A : Matrix m n ℝ) : ℝ := finner A A

lemma finner_eq_trace (A B : Matrix m n ℝ) : finner A B = Matrix.trace (A * Bᵀ) := by
  simp [finner, Matrix.trace, Matrix.mul_apply]

lemma finner_comm (A B : Matrix m n ℝ) : finner A B = finner B A := by
  simp [finner, mul_comm]

lemma frob2_nonneg (A : Matrix m n ℝ) : 0 ≤ frob2 A :=
  Finset.sum_nonneg fun _ _ => Finset.sum_nonneg fun _ _ => mul_self_nonneg _

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

lemma frob2_add (A B : Matrix m n ℝ) : frob2 (A + B) = frob2 A + 2 * finner A B + frob2 B := by
  simp only [frob2, finner, Matrix.add_apply, Finset.mul_sum, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => by ring

lemma frob2_sub_smul (A B : Matrix m n ℝ) (c : ℝ) :
    frob2 (A - c • B) = frob2 A - 2 * c * finner A B + c ^ 2 * frob2 B := by
  simp only [frob2, finner, Matrix.sub_apply, Matrix.smul_apply, smul_eq_mul, Finset.mul_sum,
    ← Finset.sum_sub_distrib, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => by ring

lemma entry_sq_le_frob2 (A : Matrix m n ℝ) (i : m) (j : n) : A i j ^ 2 ≤ frob2 A := by
  unfold frob2 finner
  calc A i j ^ 2 = A i j * A i j := sq _
    _ ≤ ∑ j', A i j' * A i j' :=
        Finset.single_le_sum (fun j' _ => mul_self_nonneg (A i j')) (Finset.mem_univ j)
    _ ≤ ∑ i', ∑ j', A i' j' * A i' j' :=
        Finset.single_le_sum (f := fun i' => ∑ j', A i' j' * A i' j')
          (fun i' _ => Finset.sum_nonneg fun j' _ => mul_self_nonneg _) (Finset.mem_univ i)

/-- `⟨E, HΞ⟩_F = ⟨EΞᵀ, H⟩_F`. -/
lemma finner_mul_right {p q : Type*} [Fintype p] [Fintype q] (E : Matrix m q ℝ) (H : Matrix m p ℝ)
    (Ξ : Matrix p q ℝ) : finner E (H * Ξ) = finner (E * Ξᵀ) H := by
  rw [finner_eq_trace, finner_eq_trace, Matrix.transpose_mul, ← Matrix.mul_assoc]

/-- Cauchy–Schwarz for the Frobenius inner product. -/
lemma finner_sq_le (A B : Matrix m n ℝ) : finner A B ^ 2 ≤ frob2 A * frob2 B := by
  have h := Finset.sum_mul_sq_le_sq_mul_sq (Finset.univ : Finset (m × n))
    (fun p => A p.1 p.2) (fun p => B p.1 p.2)
  simpa [finner, frob2, Fintype.sum_prod_type, sq] using h

/-- Rows of a product: `(W Ξ)_i = W_i Ξ`. -/
lemma mul_row {p : Type*} [Fintype p] (Wm : Matrix m n ℝ) (Ξ : Matrix n p ℝ) (i : m) :
    (Wm * Ξ) i = Wm i ᵥ* Ξ := by
  funext j
  simp [Matrix.mul_apply, Matrix.vecMul, dotProduct]

lemma frob2_eq_sum_dot (A : Matrix m n ℝ) : frob2 A = ∑ i, A i ⬝ᵥ A i := by
  simp [frob2, finner, dotProduct]

/-! ### The twin -/

variable [DecidableEq n] (Ξ : Matrix n k ℝ) (B : Matrix m k ℝ)

/-- `rank Ξ = K`: the columns of `Ξ` are linearly independent. -/
def FullColumnRank : Prop := Function.Injective Ξ.mulVec

/-- The Gram matrix `ΞᵀΞ`. -/
def gram : Matrix k k ℝ := Ξᵀ * Ξ

lemma gram_isUnit_det (h : FullColumnRank Ξ) : IsUnit (gram Ξ).det := by
  rw [← Matrix.isUnit_iff_isUnit_det, ← Matrix.mulVec_injective_iff_isUnit]
  intro y z hyz
  have h0 : gram Ξ *ᵥ (y - z) = 0 := by rw [Matrix.mulVec_sub, hyz, sub_self]
  have h1 : (Ξ *ᵥ (y - z)) ⬝ᵥ (Ξ *ᵥ (y - z)) = 0 := by
    have : (y - z) ⬝ᵥ (gram Ξ *ᵥ (y - z)) = (Ξ *ᵥ (y - z)) ⬝ᵥ (Ξ *ᵥ (y - z)) := by
      rw [gram, ← Matrix.mulVec_mulVec, Matrix.dotProduct_mulVec (y - z) Ξᵀ (Ξ *ᵥ (y - z)),
        Matrix.vecMul_transpose Ξ (y - z)]
    rw [← this, h0, dotProduct_zero]
  have h2 : Ξ *ᵥ (y - z) = 0 := dotProduct_self_eq_zero.1 h1
  have h3 : Ξ *ᵥ (y - z) = Ξ *ᵥ 0 := by rw [h2, Matrix.mulVec_zero]
  exact sub_eq_zero.1 (h h3)

/-- `Ξ⁺ := (ΞᵀΞ)⁻¹ Ξᵀ`. -/
noncomputable def pinv : Matrix k n ℝ := (gram Ξ)⁻¹ * Ξᵀ

lemma pinv_mul (h : FullColumnRank Ξ) : pinv Ξ * Ξ = 1 := by
  rw [pinv, Matrix.mul_assoc]
  exact Matrix.nonsing_inv_mul _ (gram_isUnit_det Ξ h)

/-- `Γ̂ := B Ξ⁺`. -/
noncomputable def Γhat : Matrix m n ℝ := B * pinv Ξ

lemma Γhat_mul (h : FullColumnRank Ξ) : Γhat Ξ B * Ξ = B := by
  rw [Γhat, Matrix.mul_assoc, pinv_mul Ξ h, Matrix.mul_one]

/-- The reconstruction loss `L(Γ) = ‖ΓΞ − B‖_F² = ∑_k ‖Γ ξ_k − vec ΔW^{(k)}‖²`. -/
def loss (Γ : Matrix m n ℝ) : ℝ := frob2 (Γ * Ξ - B)

lemma loss_nonneg (Γ : Matrix m n ℝ) : 0 ≤ loss Ξ B Γ := frob2_nonneg _

lemma loss_eq_zero_iff (Γ : Matrix m n ℝ) : loss Ξ B Γ = 0 ↔ Γ * Ξ = B := by
  rw [loss, frob2_eq_zero_iff, sub_eq_zero]

lemma loss_Γhat (h : FullColumnRank Ξ) : loss Ξ B (Γhat Ξ B) = 0 :=
  (loss_eq_zero_iff Ξ B _).2 (Γhat_mul Ξ B h)

/-- The minimiser set of `L` is the affine space `{Γ : ΓΞ = B}` of interpolants. -/
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

/-- Gradient certificate: `L(Γ + H) = L(Γ) + ⟨2(ΓΞ − B)Ξᵀ, H⟩_F + ‖HΞ‖_F²`, so the Frobenius
gradient of `L` at `Γ` is `∇L(Γ) = 2(ΓΞ − B)Ξᵀ`. -/
theorem loss_expand (Γ H : Matrix m n ℝ) :
    loss Ξ B (Γ + H) = loss Ξ B Γ + finner ((2 : ℝ) • ((Γ * Ξ - B) * Ξᵀ)) H + frob2 (H * Ξ) := by
  have h1 : (Γ + H) * Ξ - B = (Γ * Ξ - B) + H * Ξ := by rw [Matrix.add_mul]; abel
  have h2 : finner ((2 : ℝ) • ((Γ * Ξ - B) * Ξᵀ)) H = 2 * finner ((Γ * Ξ - B) * Ξᵀ) H := by
    simp [finner, Finset.mul_sum, mul_assoc]
  rw [loss, h1, frob2_add, finner_mul_right, h2]
  rfl

/-- `Γ̂` is the unique minimum-norm interpolant: every interpolant `Γ` has
`‖Γ̂‖_F² ≤ ‖Γ‖_F²`, with equality only at `Γ = Γ̂`. -/
theorem Γhat_min_norm (h : FullColumnRank Ξ) (Γ : Matrix m n ℝ) (hΓ : Γ * Ξ = B) :
    frob2 (Γhat Ξ B) ≤ frob2 Γ ∧ (frob2 Γ = frob2 (Γhat Ξ B) → Γ = Γhat Ξ B) := by
  set N := Γ - Γhat Ξ B with hN
  have hNΞ : N * Ξ = 0 := by rw [hN, Matrix.sub_mul, hΓ, Γhat_mul Ξ B h, sub_self]
  have hΓ' : Γ = Γhat Ξ B + N := by rw [hN]; abel
  have hΞN : Ξᵀ * Nᵀ = 0 := by rw [← Matrix.transpose_mul, hNΞ, Matrix.transpose_zero]
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

/-- One gradient step `Γ ↦ Γ − η ∇L(Γ)` with `∇L(Γ) = 2(ΓΞ − B)Ξᵀ`. -/
def gdStep (η : ℝ) (Γ : Matrix m n ℝ) : Matrix m n ℝ := Γ - (2 * η) • ((Γ * Ξ - B) * Ξᵀ)

/-- Gradient descent from `Γ_0 = 0`. -/
def gd (η : ℝ) (t : ℕ) : Matrix m n ℝ := (gdStep Ξ B η)^[t] 0

lemma gd_zero (η : ℝ) : gd Ξ B η 0 = 0 := rfl

lemma gd_succ (η : ℝ) (t : ℕ) : gd Ξ B η (t + 1) = gdStep Ξ B η (gd Ξ B η t) := by
  rw [gd, Function.iterate_succ_apply', ← gd]

/-- Error recursion: `Γ_{t+1} − Γ̂ = (Γ_t − Γ̂)(I − 2η ΞΞᵀ)`. -/
lemma gdStep_sub_Γhat (h : FullColumnRank Ξ) (η : ℝ) (Γ : Matrix m n ℝ) :
    gdStep Ξ B η Γ - Γhat Ξ B = (Γ - Γhat Ξ B) * (1 - (2 * η) • (Ξ * Ξᵀ)) := by
  have : Γ * Ξ - B = (Γ - Γhat Ξ B) * Ξ := by rw [Matrix.sub_mul, Γhat_mul Ξ B h]
  rw [gdStep, this, Matrix.mul_sub, Matrix.mul_one, Matrix.mul_smul, Matrix.mul_assoc]
  abel

/-- `P := Ξ Ξ⁺ = Ξ (ΞᵀΞ)⁻¹ Ξᵀ`, the orthogonal projector onto `span{ξ_1, …, ξ_K}`. -/
noncomputable def proj : Matrix n n ℝ := Ξ * pinv Ξ

lemma Γhat_mul_proj (h : FullColumnRank Ξ) : Γhat Ξ B * proj Ξ = Γhat Ξ B := by
  rw [Γhat, proj, Matrix.mul_assoc, ← Matrix.mul_assoc (pinv Ξ), pinv_mul Ξ h, Matrix.one_mul]

lemma S_mul_proj (h : FullColumnRank Ξ) : (Ξ * Ξᵀ) * proj Ξ = Ξ * Ξᵀ := by
  show Ξ * Ξᵀ * (Ξ * ((gram Ξ)⁻¹ * Ξᵀ)) = Ξ * Ξᵀ
  rw [Matrix.mul_assoc, ← Matrix.mul_assoc Ξᵀ Ξ]
  change Ξ * (gram Ξ * ((gram Ξ)⁻¹ * Ξᵀ)) = Ξ * Ξᵀ
  rw [Matrix.mul_nonsing_inv_cancel_left _ _ (gram_isUnit_det Ξ h)]

/-- The row space of `Ξᵀ` is invariant under one error step. -/
lemma mul_step_mul_proj (h : FullColumnRank Ξ) (η : ℝ) (E : Matrix m n ℝ) (hE : E * proj Ξ = E) :
    E * (1 - (2 * η) • (Ξ * Ξᵀ)) * proj Ξ = E * (1 - (2 * η) • (Ξ * Ξᵀ)) := by
  calc E * (1 - (2 * η) • (Ξ * Ξᵀ)) * proj Ξ
      = E * ((1 - (2 * η) • (Ξ * Ξᵀ)) * proj Ξ) := Matrix.mul_assoc _ _ _
    _ = E * (proj Ξ - (2 * η) • (Ξ * Ξᵀ)) := by
        rw [Matrix.sub_mul, Matrix.one_mul, Matrix.smul_mul, S_mul_proj Ξ h]
    _ = E * proj Ξ - (2 * η) • (E * (Ξ * Ξᵀ)) := by rw [Matrix.mul_sub, Matrix.mul_smul]
    _ = E - (2 * η) • (E * (Ξ * Ξᵀ)) := by rw [hE]
    _ = E * (1 - (2 * η) • (Ξ * Ξᵀ)) := by rw [Matrix.mul_sub, Matrix.mul_one, Matrix.mul_smul]

/-- Every iterate's error has rows in `span{ξ_k}`: `(Γ_t − Γ̂) P = Γ_t − Γ̂`. -/
lemma err_mul_proj (h : FullColumnRank Ξ) (η : ℝ) :
    ∀ t, (gd Ξ B η t - Γhat Ξ B) * proj Ξ = gd Ξ B η t - Γhat Ξ B := by
  intro t
  induction t with
  | zero =>
    rw [gd_zero, zero_sub, Matrix.neg_mul, Γhat_mul_proj Ξ B h]
  | succ t ih =>
    rw [gd_succ, gdStep_sub_Γhat Ξ B h]
    exact mul_step_mul_proj Ξ h η _ ih

lemma err_eq (h : FullColumnRank Ξ) (η : ℝ) (t : ℕ) :
    gd Ξ B η t - Γhat Ξ B = ((gd Ξ B η t - Γhat Ξ B) * Ξ) * pinv Ξ := by
  conv_lhs => rw [← err_mul_proj Ξ B h η t]
  rw [proj, Matrix.mul_assoc]

/-- Rayleigh bound on `ΞΞᵀ` transferred to the Frobenius norm: `‖WΞ‖_F² ≤ L ‖W‖_F²`. -/
lemma frob2_mul_le {L : ℝ} (hL : ∀ w : n → ℝ, (w ᵥ* Ξ) ⬝ᵥ (w ᵥ* Ξ) ≤ L * (w ⬝ᵥ w))
    (Wm : Matrix m n ℝ) : frob2 (Wm * Ξ) ≤ L * frob2 Wm := by
  rw [frob2_eq_sum_dot, frob2_eq_sum_dot, Finset.mul_sum]
  refine Finset.sum_le_sum fun i _ => ?_
  rw [mul_row]
  exact hL _

/-- The same bound for the transposed factor, `‖ZΞᵀ‖_F² ≤ L ‖Z‖_F²`, via Cauchy–Schwarz
(this is `λ_max(ΞᵀΞ) = λ_max(ΞΞᵀ)`). -/
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
  rcases ha0.lt_or_eq with hpos | hzero
  · nlinarith
  · rw [← hzero]; positivity

/-- One descent step: `‖E(I − 2ηΞΞᵀ)‖_F² ≤ ‖E‖_F² − 4η(1 − ηL)‖EΞ‖_F²`. -/
lemma frob2_step_le {L : ℝ} (hL0 : 0 ≤ L)
    (hL : ∀ w : n → ℝ, (w ᵥ* Ξ) ⬝ᵥ (w ᵥ* Ξ) ≤ L * (w ⬝ᵥ w)) {η : ℝ} (hη : 0 ≤ η)
    (E : Matrix m n ℝ) :
    frob2 (E * (1 - (2 * η) • (Ξ * Ξᵀ))) ≤ frob2 E - 4 * η * (1 - η * L) * frob2 (E * Ξ) := by
  have h1 : E * (1 - (2 * η) • (Ξ * Ξᵀ)) = E - (2 * η) • ((E * Ξ) * Ξᵀ) := by
    rw [Matrix.mul_sub, Matrix.mul_one, Matrix.mul_smul, Matrix.mul_assoc]
  have h2 : finner E ((E * Ξ) * Ξᵀ) = frob2 (E * Ξ) := by
    rw [finner_mul_right, Matrix.transpose_transpose]; rfl
  have h3 : frob2 ((E * Ξ) * Ξᵀ) ≤ L * frob2 (E * Ξ) := frob2_mul_transpose_le Ξ hL0 hL _
  have h4 : 0 ≤ (2 * η) ^ 2 := sq_nonneg _
  have h5 := mul_le_mul_of_nonneg_left h3 h4
  rw [h1, frob2_sub_smul, h2]
  nlinarith [frob2_nonneg (E * Ξ)]

/-- Entrywise convergence from convergence of the squared Frobenius norm. -/
lemma tendsto_of_frob2 (F : ℕ → Matrix m n ℝ) (hF : Tendsto (fun t => frob2 (F t)) atTop (𝓝 0)) :
    Tendsto F atTop (𝓝 0) := by
  refine tendsto_pi_nhds.2 fun i => tendsto_pi_nhds.2 fun j => ?_
  have h1 : Tendsto (fun t => (F t i j) ^ 2) atTop (𝓝 0) :=
    squeeze_zero (fun t => sq_nonneg _) (fun t => entry_sq_le_frob2 (F t) i j) hF
  have h2 : Tendsto (fun t => ‖F t i j‖) atTop (𝓝 0) := by
    have := h1.sqrt
    simpa [Real.sqrt_sq_eq_abs, Real.norm_eq_abs] using this
  simpa using tendsto_zero_iff_norm_tendsto_zero.2 h2

/-- **Proposition `prop:twin`, gradient-descent part.** Under `rank Ξ = K`, with
`λ_max(ΞΞᵀ) ≤ L` and step `0 < η < 1/L`, gradient descent from `Γ_0 = 0` converges to
`Γ̂ = BΞ⁺`. -/
theorem gd_tendsto (h : FullColumnRank Ξ) {L : ℝ} (hL0 : 0 ≤ L)
    (hL : ∀ w : n → ℝ, (w ᵥ* Ξ) ⬝ᵥ (w ᵥ* Ξ) ≤ L * (w ⬝ᵥ w)) {η : ℝ} (hη : 0 < η)
    (hηL : η * L < 1) :
    Tendsto (gd Ξ B η) atTop (𝓝 (Γhat Ξ B)) := by
  set E : ℕ → Matrix m n ℝ := fun t => gd Ξ B η t - Γhat Ξ B with hE
  set κ : ℝ := 4 * η * (1 - η * L) with hκ
  have hκpos : 0 < κ := by
    have : 0 < 1 - η * L := by linarith
    rw [hκ]
    exact mul_pos (mul_pos four_pos hη) this
  have hstep : ∀ t, frob2 (E (t + 1)) ≤ frob2 (E t) - κ * frob2 (E t * Ξ) := fun t => by
    have := frob2_step_le Ξ hL0 hL hη.le (E t)
    have hrec : E (t + 1) = E t * (1 - (2 * η) • (Ξ * Ξᵀ)) := by
      simp only [hE]
      rw [gd_succ, gdStep_sub_Γhat Ξ B h]
    rw [hrec]
    exact this
  have hpartial : ∀ N, ∑ t ∈ Finset.range N, κ * frob2 (E t * Ξ) + frob2 (E N) ≤ frob2 (E 0) := by
    intro N
    induction N with
    | zero => simp
    | succ N ih =>
      rw [Finset.sum_range_succ]
      have := hstep N
      linarith
  have hsummable : Summable fun t => κ * frob2 (E t * Ξ) :=
    summable_of_sum_range_le (c := frob2 (E 0)) (fun t => mul_nonneg hκpos.le (frob2_nonneg _))
      (fun N => by linarith [hpartial N, frob2_nonneg (E N)])
  have h1 : Tendsto (fun t => κ * frob2 (E t * Ξ)) atTop (𝓝 0) := hsummable.tendsto_atTop_zero
  have h2 : Tendsto (fun t => frob2 (E t * Ξ)) atTop (𝓝 0) := by
    have := h1.const_mul κ⁻¹
    simpa [← mul_assoc, inv_mul_cancel₀ hκpos.ne'] using this
  have h3 : Tendsto (fun t => E t * Ξ) atTop (𝓝 0) := tendsto_of_frob2 _ h2
  have h4 : Tendsto (fun t => (E t * Ξ) * pinv Ξ) atTop (𝓝 0) := by
    have hc : Continuous fun X : Matrix m k ℝ => X * pinv Ξ :=
      continuous_id.matrix_mul continuous_const
    have := (hc.tendsto 0).comp h3
    rw [Matrix.zero_mul] at this
    exact this
  have h5 : Tendsto E atTop (𝓝 0) := by
    refine h4.congr fun t => ?_
    simp only [hE]
    exact (err_eq Ξ B h η t).symm
  have h6 := h5.add_const (Γhat Ξ B)
  rw [zero_add] at h6
  refine h6.congr fun t => ?_
  simp only [hE]
  abel

/-! ### The output at a new conditioning -/

/-- `β(h) := Ξ⁺ ξ(h)`. -/
noncomputable def beta (ξ : n → ℝ) : k → ℝ := pinv Ξ *ᵥ ξ

/-- The generated (vectorised) update `Γ ξ(h)`. -/
def output (Γ : Matrix m n ℝ) (ξ : n → ℝ) : m → ℝ := Γ *ᵥ ξ

/-- Eq. prop:twin: `vec ΔW_{Γ̂}(h) = B β(h)`. -/
theorem output_Γhat (ξ : n → ℝ) : output (Γhat Ξ B) ξ = B *ᵥ beta Ξ ξ := by
  simp [output, Γhat, beta, Matrix.mulVec_mulVec]

/-- `B β = ∑_k β_k vec ΔW^{(k)}` (columns of `B`). -/
lemma mulVec_eq_sum_cols (β : k → ℝ) : B *ᵥ β = ∑ j, β j • Bᵀ j := by
  ext i
  simp [Matrix.mulVec, dotProduct, Finset.sum_apply, mul_comm]

/-- Statement (a): the output at any conditioning lies in the linear span of the bank. -/
theorem output_mem_span (ξ : n → ℝ) :
    output (Γhat Ξ B) ξ ∈ Submodule.span ℝ (Set.range Bᵀ) := by
  rw [output_Γhat, mulVec_eq_sum_cols]
  exact Submodule.sum_mem _ fun j _ =>
    Submodule.smul_mem _ _ (Submodule.subset_span (Set.mem_range_self j))

/-- Statement (b): `β(h)` is a function of `(Ξ, ξ(h))` alone — `beta` does not mention `B`, so
two twins with the same features and different banks route with the same coefficients. -/
theorem beta_independent_of_bank (B' : Matrix m k ℝ) (ξ : n → ℝ) :
    output (Γhat Ξ B) ξ = B *ᵥ beta Ξ ξ ∧ output (Γhat Ξ B') ξ = B' *ᵥ beta Ξ ξ :=
  ⟨output_Γhat Ξ B ξ, output_Γhat Ξ B' ξ⟩

lemma col_eq_mulVec_single {p : Type*} (M : Matrix p k ℝ) (j : k) :
    Mᵀ j = M *ᵥ Pi.single j 1 := by
  ext i
  simp [Matrix.mulVec, dotProduct, Pi.single_apply]

/-- `β(ξ_j) = e_j`. -/
lemma beta_col (h : FullColumnRank Ξ) (j : k) : beta Ξ (Ξᵀ j) = Pi.single j 1 := by
  rw [beta, col_eq_mulVec_single Ξ, Matrix.mulVec_mulVec, pinv_mul Ξ h, Matrix.one_mulVec]

/-- Statement (c), first example: `β(ξ_1 + ξ_2) = e_1 + e_2`. -/
theorem beta_add_cols (h : FullColumnRank Ξ) (j₁ j₂ : k) :
    beta Ξ (Ξᵀ j₁ + Ξᵀ j₂) = Pi.single j₁ 1 + Pi.single j₂ 1 := by
  rw [beta, Matrix.mulVec_add, ← beta, ← beta, beta_col Ξ h, beta_col Ξ h]

/-- Statement (c), second example: `β(ξ_1 − ξ_2) = e_1 − e_2`. -/
theorem beta_sub_cols (h : FullColumnRank Ξ) (j₁ j₂ : k) :
    beta Ξ (Ξᵀ j₁ - Ξᵀ j₂) = Pi.single j₁ 1 - Pi.single j₂ 1 := by
  rw [beta, Matrix.mulVec_sub, ← beta, ← beta, beta_col Ξ h, beta_col Ξ h]

/-- The corresponding outputs: `ΔW^{(1)} + ΔW^{(2)}` and `ΔW^{(1)} − ΔW^{(2)}`. -/
theorem output_add_cols (h : FullColumnRank Ξ) (j₁ j₂ : k) :
    output (Γhat Ξ B) (Ξᵀ j₁ + Ξᵀ j₂) = Bᵀ j₁ + Bᵀ j₂ := by
  rw [output_Γhat, beta_add_cols Ξ h, Matrix.mulVec_add, ← col_eq_mulVec_single B,
    ← col_eq_mulVec_single B]

theorem output_sub_cols (h : FullColumnRank Ξ) (j₁ j₂ : k) :
    output (Γhat Ξ B) (Ξᵀ j₁ - Ξᵀ j₂) = Bᵀ j₁ - Bᵀ j₂ := by
  rw [output_Γhat, beta_sub_cols Ξ h, Matrix.mulVec_sub, ← col_eq_mulVec_single B,
    ← col_eq_mulVec_single B]

/-- `ΔW^{(1)} + ΔW^{(2)}` is twice the mean of the two. -/
theorem add_cols_eq_two_smul_mean (j₁ j₂ : k) :
    Bᵀ j₁ + Bᵀ j₂ = (2 : ℝ) • ((1 / 2 : ℝ) • (Bᵀ j₁ + Bᵀ j₂)) := by
  rw [smul_smul]; norm_num

/-- `e_1 + e_2 ∉ Δ^{K-1}` (its coordinates sum to `2`). -/
theorem beta_add_cols_not_mem_simplex (h : FullColumnRank Ξ) (j₁ j₂ : k) :
    beta Ξ (Ξᵀ j₁ + Ξᵀ j₂) ∉ stdSimplex ℝ k := by
  rw [beta_add_cols Ξ h]
  rintro ⟨-, hsum⟩
  simp [Finset.sum_add_distrib, Pi.single_apply] at hsum

/-- `e_1 − e_2 ∉ Δ^{K-1}` (coordinate `j₂` is `−1`). -/
theorem beta_sub_cols_not_mem_simplex (h : FullColumnRank Ξ) {j₁ j₂ : k} (hj : j₁ ≠ j₂) :
    beta Ξ (Ξᵀ j₁ - Ξᵀ j₂) ∉ stdSimplex ℝ k := by
  rw [beta_sub_cols Ξ h]
  rintro ⟨hnn, -⟩
  have := hnn j₂
  rw [Pi.sub_apply, Pi.single_eq_of_ne hj.symm, Pi.single_eq_same] at this
  norm_num at this

/-- Orthonormal features, as in the paper's example: then `Ξ⁺ = Ξᵀ` and `β(h) = Ξᵀ ξ(h)`. -/
theorem beta_of_orthonormal (horth : Ξᵀ * Ξ = 1) (ξ : n → ℝ) : beta Ξ ξ = Ξᵀ *ᵥ ξ := by
  simp [beta, pinv, gram, horth]

/-! ### Corollary `cor:scale`, twin half -/

/-- `β(t ξ_k) = t e_k`. -/
theorem beta_smul_col (h : FullColumnRank Ξ) (t : ℝ) (j : k) :
    beta Ξ (t • Ξᵀ j) = t • Pi.single j 1 := by
  rw [beta, Matrix.mulVec_smul, ← beta, beta_col Ξ h]

/-- The feature vector `ξ(h) = t ξ_k` yields the output `t ΔW^{(k)}`. -/
theorem output_smul_col (h : FullColumnRank Ξ) (t : ℝ) (j : k) :
    output (Γhat Ξ B) (t • Ξᵀ j) = t • Bᵀ j := by
  rw [output_Γhat, beta_smul_col Ξ h, Matrix.mulVec_smul, ← col_eq_mulVec_single B]

/-- No scale bound holds in the twin: for `ΔW^{(k)} ≠ 0`, the norm of the generated update at
`ξ(h) = t ξ_k` exceeds any prescribed `M` for some `t > 0`. -/
theorem output_unbounded (h : FullColumnRank Ξ) (j : k) (hj : Bᵀ j ≠ 0) (M : ℝ) :
    ∃ t : ℝ, 0 < t ∧ M < ‖output (Γhat Ξ B) (t • Ξᵀ j)‖ := by
  have hnorm : 0 < ‖Bᵀ j‖ := norm_pos_iff.2 hj
  refine ⟨(|M| + 1) / ‖Bᵀ j‖, by positivity, ?_⟩
  rw [output_smul_col Ξ B h, norm_smul, Real.norm_eq_abs, abs_of_pos (by positivity),
    div_mul_cancel₀ _ hnorm.ne']
  linarith [le_abs_self M]

end Web2LoRA.Twin
