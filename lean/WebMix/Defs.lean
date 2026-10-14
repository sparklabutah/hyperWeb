import Mathlib.Analysis.InnerProductSpace.PiL2

/-!
# Definitions used in the paper's statements

Every object that appears in `WebMix/Statements.lean`, with a docstring citing the equation
or label of the WebMix paper (Appendix C, "Theoretical Analysis"; Section 3.2 for the
composition). This file holds definitions only: proofs live in `WebMix/Proofs/`.
It is frozen together with `Statements.lean` once the statements are approved (Gate A).

Conventions.
* The adapter space `𝒲 = ∏_{s ∈ Ω} ℝ^{d_out(s) × d_in(s)}` with
  `‖ΔW‖ = (∑_s ‖ΔW_s‖_F²)^{1/2}` is a real normed (indeed Euclidean) space, `PaperSpace`.
  The hull results are stated for every real normed space `W`, so they include `PaperSpace`.
* `Δ^{K-1}` is Mathlib's `stdSimplex ℝ (Fin K)`, and `dist(·, ℋ)` is `Metric.infDist`.
* In the frozen-trunk twin, matrices act on column vectors indexed by finite types:
  `m ≃ {1, …, D_Ω}` (vectorised updates), `n ≃ {1, …, d_ξ}` (features), `k ≃ {1, …, K}`
  (source interfaces). Mathlib's default norm on `m → ℝ` is the sup norm, so Euclidean norms
  are written out (`Twin.frob2`, `Twin.eucNorm`).
-/

open Matrix

namespace WebMix

/-! ### Mixtures and the hull (Appendix C, Setting) -/

/-- The probability simplex `Δ^{K-1} = {c ∈ ℝ^K : c_k ≥ 0, ∑_k c_k = 1}`. -/
abbrev simplex (K : ℕ) : Set (Fin K → ℝ) := stdSimplex ℝ (Fin K)

/-- The mixture `M(c) := ∑_{k=1}^{K} c_k ΔW^{(k)}` of Eq. thm:mix. By Eq. exact it is the update
WebMix serves under coefficients `c`. -/
def mix {W : Type*} [AddCommGroup W] [Module ℝ W] {K : ℕ} (ΔW : Fin K → W) (c : Fin K → ℝ) : W :=
  ∑ k, c k • ΔW k

/-- The reachable set `ℋ_K := M(Δ^{K-1})` of Eq. thm:mix. -/
def hull {W : Type*} [AddCommGroup W] [Module ℝ W] {K : ℕ} (ΔW : Fin K → W) : Set W :=
  mix ΔW '' simplex K

/-- `𝒦_u := conv(ℋ_K ∪ {ΔW^{(u)}})`, the set on which Assumption ass:lip is stated. -/
def Ku {W : Type*} [AddCommGroup W] [Module ℝ W] {K : ℕ} (ΔW : Fin K → W) (ΔWu : W) : Set W :=
  convexHull ℝ (hull ΔW ∪ {ΔWu})

/-- The uniform coefficient vector `𝟏/K` (Remark rem:hull, Corollary cor:jensen). -/
noncomputable def uniform (K : ℕ) : Fin K → ℝ := fun _ => 1 / (K : ℝ)

/-- The paper's adapter space `𝒲 = ∏_{s ∈ Ω} ℝ^{d_out(s) × d_in(s)}` with
`‖ΔW‖ = (∑_s ‖ΔW_s‖_F²)^{1/2}`: the Euclidean space on all entries of all adapted matrices. -/
abbrev PaperSpace (Ω : Type) (dout din : Ω → ℕ) : Type :=
  EuclideanSpace ℝ (Σ s : Ω, Fin (dout s) × Fin (din s))

/-! ### Behaviour-cloning risk with logits affine in the update (Remark rem:jensen) -/

/-- log-sum-exp of a logit vector, `lse(ℓ) = log ∑_j exp ℓ_j`. -/
noncomputable def lse {ι : Type*} [Fintype ι] (ℓ : ι → ℝ) : ℝ := Real.log (∑ j, Real.exp (ℓ j))

/-- Per-token negative log-likelihood `−log softmax(ℓ)_y = lse(ℓ) − ℓ_y` (proof of
Corollary cor:jensen). -/
noncomputable def nll {ι : Type*} [Fintype ι] (y : ι) (ℓ : ι → ℝ) : ℝ := lse ℓ - ℓ y

/-- The behaviour-cloning risk of Eq. thm:risk when the logits of demonstrated token `i` are the
affine function `A i` of the installed update `ΔW`: `ℛ(ΔW) = ∑_i w_i · nll(y_i, A_i ΔW)` over the
`N` demonstrated (response) tokens, with vocabulary `ι`, targets `y i` and weights `w i ≥ 0`.
Eq. thm:risk is the case `w_i = 1/|𝒟_u|` for every response token of `𝒟_u`; Eq. obj restricted
to `𝒟_u` is the case `w_i = 1/N_u`, with `N_u` the number of response tokens of `𝒟_u`. -/
noncomputable def bcRisk {W ι : Type*} [AddCommGroup W] [Module ℝ W] [Fintype ι] {N : ℕ}
    (A : Fin N → (W →ᵃ[ℝ] (ι → ℝ))) (y : Fin N → ι) (w : Fin N → ℝ) (ΔW : W) : ℝ :=
  ∑ i, w i * nll (y i) (A i ΔW)

/-! ### The frozen-trunk twin (Assumption ass:twin, Proposition prop:twin) -/

namespace Twin

/-- Frobenius inner product `⟨A, B⟩_F = ∑_{ij} A_{ij} B_{ij}`. -/
def finner {m n : Type*} [Fintype m] [Fintype n] (A B : Matrix m n ℝ) : ℝ :=
  ∑ i, ∑ j, A i j * B i j

/-- Squared Frobenius norm `‖A‖_F² = ⟨A, A⟩_F`. -/
def frob2 {m n : Type*} [Fintype m] [Fintype n] (A : Matrix m n ℝ) : ℝ := finner A A

/-- Euclidean norm `‖v‖₂ = (∑_i v_i²)^{1/2}` of a vectorised update; on `vec ΔW` it is the norm
`‖ΔW‖ = (∑_s ‖ΔW_s‖_F²)^{1/2}` of the paper. -/
noncomputable def eucNorm {m : Type*} [Fintype m] (v : m → ℝ) : ℝ := Real.sqrt (v ⬝ᵥ v)

/-- `rank Ξ = K` (Assumption ass:twin): the columns `ξ_1, …, ξ_K` of `Ξ` are linearly
independent, that is, `y ↦ Ξ y` is injective. -/
def FullColumnRank {n k : Type*} [Fintype k] (Ξ : Matrix n k ℝ) : Prop :=
  Function.Injective Ξ.mulVec

/-- The Gram matrix `ΞᵀΞ` of the training features. -/
def gram {n k : Type*} [Fintype n] (Ξ : Matrix n k ℝ) : Matrix k k ℝ := Ξᵀ * Ξ

/-- `Ξ⁺ := (ΞᵀΞ)⁻¹ Ξᵀ` (Proposition prop:twin). -/
noncomputable def pinv {n k : Type*} [Fintype n] [Fintype k] [DecidableEq k]
    (Ξ : Matrix n k ℝ) : Matrix k n ℝ :=
  (gram Ξ)⁻¹ * Ξᵀ

/-- `Γ̂ := 𝓑 Ξ⁺`, the head that gradient descent reaches (Proposition prop:twin). Here `B` is
`𝓑 = [vec ΔW^{(1)}, …, vec ΔW^{(K)}]`. -/
noncomputable def Γhat {m n k : Type*} [Fintype n] [Fintype k] [DecidableEq k]
    (Ξ : Matrix n k ℝ) (B : Matrix m k ℝ) : Matrix m n ℝ :=
  B * pinv Ξ

/-- The reconstruction loss of Assumption ass:twin,
`L(Γ) = ∑_k ‖Γ ξ_k − vec ΔW^{(k)}‖² = ‖ΓΞ − 𝓑‖_F²`. -/
def loss {m n k : Type*} [Fintype m] [Fintype n] [Fintype k]
    (Ξ : Matrix n k ℝ) (B : Matrix m k ℝ) (Γ : Matrix m n ℝ) : ℝ :=
  frob2 (Γ * Ξ - B)

/-- One gradient step `Γ ↦ Γ − η ∇L(Γ)` with `∇L(Γ) = 2(ΓΞ − 𝓑)Ξᵀ` and step size `η`. -/
def gdStep {m n k : Type*} [Fintype n] [Fintype k]
    (Ξ : Matrix n k ℝ) (B : Matrix m k ℝ) (η : ℝ) (Γ : Matrix m n ℝ) : Matrix m n ℝ :=
  Γ - (2 * η) • ((Γ * Ξ - B) * Ξᵀ)

/-- Gradient descent from `Γ_0 = 0`: `gd Ξ B η t = Γ_t`. -/
def gd {m n k : Type*} [Fintype n] [Fintype k]
    (Ξ : Matrix n k ℝ) (B : Matrix m k ℝ) (η : ℝ) (t : ℕ) : Matrix m n ℝ :=
  (gdStep Ξ B η)^[t] 0

/-- The coefficients `η(h) := Ξ⁺ ξ(h)` of Eq. prop:twin, as a function of the feature vector
`ξ(h)` (named `beta` because `η` is the step size here). -/
noncomputable def beta {n k : Type*} [Fintype n] [Fintype k] [DecidableEq k]
    (Ξ : Matrix n k ℝ) (ξ : n → ℝ) : k → ℝ :=
  pinv Ξ *ᵥ ξ

/-- The vectorised update `vec ΔW_Γ(h) = Γ ξ(h)` that the head `Γ` emits for the feature vector
`ξ(h)` (Assumption ass:twin). -/
def output {m n : Type*} [Fintype n] (Γ : Matrix m n ℝ) (ξ : n → ℝ) : m → ℝ := Γ *ᵥ ξ

end Twin

/-! ### LoRA factors (Eqs. bg:gauge, delta, exact; Proposition prop:wellposed) -/

namespace Gauge

/-- The composed `A`-factor of Eq. delta, `A_s = [c_1 A^{(1)}_s; …; c_K A^{(K)}_s]`, with rows
indexed by `Fin K × r` (rank `K r`). -/
def concatA {r din : Type*} {K : ℕ} (c : Fin K → ℝ) (A : Fin K → Matrix r din ℝ) :
    Matrix (Fin K × r) din ℝ :=
  Matrix.of fun p j => c p.1 * A p.1 p.2 j

/-- The composed `B`-factor of Eq. delta, `B_s = [B^{(1)}_s, …, B^{(K)}_s]`, with columns indexed
by `Fin K × r`. -/
def concatB {r dout : Type*} {K : ℕ} (B : Fin K → Matrix dout r ℝ) : Matrix dout (Fin K × r) ℝ :=
  Matrix.of fun o p => B p.1 o p.2

/-- Entrywise `ℓ₁` norm `‖A‖₁ = ∑_{ij} |A_{ij}|` (Proposition prop:wellposed (c)). -/
def l1 {p q : Type*} [Fintype p] [Fintype q] (A : Matrix p q ℝ) : ℝ := ∑ i, ∑ j, |A i j|

end Gauge

end WebMix
