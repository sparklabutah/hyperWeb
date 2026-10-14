import WebMix.Defs
import Mathlib.Analysis.Calculus.ContDiff.Defs
import Mathlib.Analysis.Calculus.FDeriv.Basic
import Mathlib.Analysis.Convex.Function
import Mathlib.Analysis.InnerProductSpace.GramMatrix
import Mathlib.Order.Filter.Extr
import Mathlib.Topology.EMetricSpace.Lipschitz
import Mathlib.Topology.MetricSpace.HausdorffDistance

/-!
# The paper's results as closed propositions

One `def <label> : Prop` per result of Appendix C of the WebMix paper (and the composition
identity of Section 3.2), named after its LaTeX label (`thm:hull` ↦ `thm_hull`). Every hypothesis
is written inside the proposition: there are no `variable`s, axioms, notations or instances.
The theorem `WebMix.<label>` in `WebMix/Proofs/Results.lean` has type
`WebMix.Statements.<label>`, so the kernel checks that what is proved is what is stated here.

This file and `Defs.lean` are frozen after the statements are approved (Gate A);
`.statements.sha256` records their hashes.
-/

open Filter Topology Matrix

namespace WebMix.Statements

/-- **Assumption ass:lip, justification** (Appendix C.1). If the risk `ℛ_u` is continuously
differentiable, Assumption ass:lip holds on `𝒦_u = conv(ℋ_K ∪ {ΔW^{(u)}})` with `Λ_u` equal to
the largest gradient norm of `ℛ_u` on `𝒦_u`, attained at some `x₀ ∈ 𝒦_u`. -/
def ass_lip : Prop :=
  ∀ (W : Type) [NormedAddCommGroup W] [NormedSpace ℝ W] (K : ℕ) (ΔW : Fin K → W) (ΔWu : W)
    (R : W → ℝ), ContDiff ℝ 1 R →
    ∃ x₀ ∈ Ku ΔW ΔWu,
      (∀ x ∈ Ku ΔW ΔWu, ‖fderiv ℝ R x‖ ≤ ‖fderiv ℝ R x₀‖) ∧
      LipschitzOnWith ‖fderiv ℝ R x₀‖₊ R (Ku ΔW ΔWu)

/-- **Theorem thm:hull** (hull decomposition of the held-out risk). For a bank of `K ≥ 1`
updates, an oracle `ΔW^{(u)}` and a risk `ℛ_u` that is `Λ_u`-Lipschitz on `𝒦_u`
(Assumption ass:lip): a best mixture `c⋆ ∈ argmin_{Δ^{K-1}} ℛ_u ∘ M` exists; for every routed
`ĉ ∈ Δ^{K-1}` the excess risk is the routing term plus the approximation term (Eq. thm:decomp)
and the routing term is nonnegative; and the approximation term satisfies both bounds of
Eq. thm:bounds. -/
def thm_hull : Prop :=
  ∀ (W : Type) [NormedAddCommGroup W] [NormedSpace ℝ W] (K : ℕ) (ΔW : Fin K → W) (ΔWu : W)
    (R : W → ℝ) (Λ : NNReal), 0 < K → LipschitzOnWith Λ R (Ku ΔW ΔWu) →
    ∃ cstar ∈ simplex K, IsMinOn (fun c => R (mix ΔW c)) (simplex K) cstar ∧
      (∀ chat ∈ simplex K,
        R (mix ΔW chat) - R ΔWu
            = (R (mix ΔW chat) - R (mix ΔW cstar)) + (R (mix ΔW cstar) - R ΔWu) ∧
          0 ≤ R (mix ΔW chat) - R (mix ΔW cstar)) ∧
      R (mix ΔW cstar) - R ΔWu ≤ Λ * Metric.infDist ΔWu (hull ΔW) ∧
      ∀ k, R (mix ΔW cstar) ≤ R (ΔW k)

/-- **Remark rem:hull.** For `K ≥ 1`, `dist(ΔW^{(u)}, ℋ_K)` is attained at some `c_P ∈ Δ^{K-1}`
and is at most `‖ΔW^{(u)} − M(c)‖` for every `c ∈ Δ^{K-1}`, and the uniform mixture `M(𝟏/K)` is an
element of `ℋ_K`. In an inner-product adapter space the squared distance to `M(c)` is the
quadratic `‖ΔW^{(u)}‖² − 2 ∑_k c_k ⟨ΔW^{(u)}, ΔW^{(k)}⟩ + cᵀ G c` in `c`, with Gram matrix
`G_{jk} = ⟨ΔW^{(j)}, ΔW^{(k)}⟩`. Together the two parts make `dist(ΔW^{(u)}, ℋ_K)²` the minimum of
this quadratic over the simplex: a `K`-variable least-squares problem whose data are inner products
of the stored adapters. -/
def rem_hull : Prop :=
  (∀ (W : Type) [NormedAddCommGroup W] [NormedSpace ℝ W] (K : ℕ) (ΔW : Fin K → W) (ΔWu : W),
    0 < K →
    (∃ cP ∈ simplex K, ‖ΔWu - mix ΔW cP‖ = Metric.infDist ΔWu (hull ΔW)) ∧
    (∀ c ∈ simplex K, Metric.infDist ΔWu (hull ΔW) ≤ ‖ΔWu - mix ΔW c‖) ∧
    mix ΔW (uniform K) ∈ hull ΔW) ∧
  ∀ (W : Type) [NormedAddCommGroup W] [InnerProductSpace ℝ W] (K : ℕ) (ΔW : Fin K → W)
    (ΔWu : W) (c : Fin K → ℝ),
    ‖ΔWu - mix ΔW c‖ ^ 2
      = ‖ΔWu‖ ^ 2 - 2 * ∑ k, c k * inner ℝ ΔWu (ΔW k) + c ⬝ᵥ (Matrix.gram ℝ ΔW *ᵥ c)

/-- **Corollary cor:growth** (growing the bank). Extend a bank of `K ≥ 1` updates by one update
`new` (`Fin.snoc` appends it as member `K+1`). Then `M_{K+1}((c, 0)) = M_K(c)`,
`ℋ_K ⊆ ℋ_{K+1}`, `dist(ΔW, ℋ_{K+1}) ≤ dist(ΔW, ℋ_K)` for every `ΔW`, and for every risk `ℛ_v`
that is continuous on `ℋ_{K+1}` both best-mixture minima exist and
`ℛ⋆_v(K+1) ≤ ℛ⋆_v(K)`. -/
def cor_growth : Prop :=
  ∀ (W : Type) [NormedAddCommGroup W] [NormedSpace ℝ W] (K : ℕ) (ΔW : Fin K → W) (new : W),
    0 < K →
    (∀ c : Fin K → ℝ, mix (Fin.snoc ΔW new) (Fin.snoc c 0) = mix ΔW c) ∧
    hull ΔW ⊆ hull (Fin.snoc ΔW new) ∧
    (∀ x : W, Metric.infDist x (hull (Fin.snoc ΔW new)) ≤ Metric.infDist x (hull ΔW)) ∧
    ∀ R : W → ℝ, ContinuousOn R (hull (Fin.snoc ΔW new)) →
      (∃ c ∈ simplex K, IsMinOn (fun c => R (mix ΔW c)) (simplex K) c) ∧
      (∃ c' ∈ simplex (K + 1),
        IsMinOn (fun c => R (mix (Fin.snoc ΔW new) c)) (simplex (K + 1)) c') ∧
      ∀ c' ∈ simplex (K + 1),
        IsMinOn (fun c => R (mix (Fin.snoc ΔW new) c)) (simplex (K + 1)) c' →
        ∀ c ∈ simplex K, R (mix (Fin.snoc ΔW new) c') ≤ R (mix ΔW c)

/-- **Corollary cor:jensen** (convex twin). If `ℛ_u` is convex on `𝒦_u`, then
`ℛ_u(M(c)) ≤ ∑_k c_k ℛ_u(ΔW^{(k)})` for every `c ∈ Δ^{K-1}`, and for `K ≥ 1`
`ℛ_u(M(𝟏/K)) ≤ (1/K) ∑_k ℛ_u(ΔW^{(k)})` (Eq. cor:jensen). -/
def cor_jensen : Prop :=
  ∀ (W : Type) [NormedAddCommGroup W] [NormedSpace ℝ W] (K : ℕ) (ΔW : Fin K → W) (ΔWu : W)
    (R : W → ℝ), ConvexOn ℝ (Ku ΔW ΔWu) R →
    (∀ c ∈ simplex K, R (mix ΔW c) ≤ ∑ k, c k * R (ΔW k)) ∧
    (0 < K → R (mix ΔW (uniform K)) ≤ (1 / (K : ℝ)) * ∑ k, R (ΔW k))

/-- **Remark rem:jensen** (scope of the convex twin; second half of the proof of
Corollary cor:jensen). If the logits of every demonstrated token are an affine function of the
update `ΔW`, the behaviour-cloning risk, a nonnegative combination of per-token negative
log-likelihoods, is convex on all of `𝒲`. -/
def rem_jensen : Prop :=
  ∀ (W : Type) [AddCommGroup W] [Module ℝ W] (ι : Type) [Fintype ι] [Nonempty ι] (N : ℕ)
    (A : Fin N → (W →ᵃ[ℝ] (ι → ℝ))) (y : Fin N → ι) (w : Fin N → ℝ),
    (∀ i, 0 ≤ w i) → ConvexOn ℝ Set.univ (bcRisk A y w)

/-- **Proposition prop:twin** (with Assumption ass:twin). Features `Ξ = [ξ_1, …, ξ_K]` of full
column rank; bank `𝓑 = [vec ΔW^{(1)}, …, vec ΔW^{(K)}]` (written `B`); loss `L(Γ) = ‖ΓΞ − 𝓑‖_F²`.
* The minimisers of `L` are exactly the interpolants `ΓΞ = 𝓑`.
* `Γ̂ = 𝓑Ξ⁺` interpolates and is the unique minimum-Frobenius-norm interpolant.
* Gradient descent from `Γ = 0` with step `η`, `0 < η < 1/λ_max(ΞΞᵀ)`, converges to `Γ̂`
  (`λ_max(ΞΞᵀ) ≤ L` is written as the Rayleigh bound `‖Ξᵀw‖² ≤ L‖w‖²`).
* Eq. prop:twin: `vec ΔW_{Γ̂}(h) = 𝓑 η(h)` with `η(h) = Ξ⁺ξ(h)`, and (a) it lies in the span of
  the bank.
* (b) One coefficient map, chosen from `Ξ` before the bank and its dimension, serves every
  bank `𝓑`.
* (c) For `j₁ ≠ j₂`, `η(ξ_{j₁} ± ξ_{j₂}) = e_{j₁} ± e_{j₂} ∉ Δ^{K-1}`, with outputs
  `ΔW^{(j₁)} ± ΔW^{(j₂)}`. -/
def prop_twin : Prop :=
  (∀ (m n k : Type) [Fintype m] [Fintype n] [Fintype k] [DecidableEq k]
    (Ξ : Matrix n k ℝ), Twin.FullColumnRank Ξ →
    (∀ (B : Matrix m k ℝ) (Γ : Matrix m n ℝ),
        IsMinOn (Twin.loss Ξ B) Set.univ Γ ↔ Γ * Ξ = B) ∧
    (∀ B : Matrix m k ℝ, Twin.Γhat Ξ B * Ξ = B ∧
        ∀ Γ : Matrix m n ℝ, Γ * Ξ = B →
          Twin.frob2 (Twin.Γhat Ξ B) ≤ Twin.frob2 Γ ∧
            (Twin.frob2 Γ = Twin.frob2 (Twin.Γhat Ξ B) → Γ = Twin.Γhat Ξ B)) ∧
    (∀ (B : Matrix m k ℝ) (L : ℝ), 0 ≤ L →
        (∀ w : n → ℝ, (w ᵥ* Ξ) ⬝ᵥ (w ᵥ* Ξ) ≤ L * (w ⬝ᵥ w)) →
        ∀ η : ℝ, 0 < η → η * L < 1 →
          Tendsto (Twin.gd Ξ B η) atTop (𝓝 (Twin.Γhat Ξ B))) ∧
    (∀ (B : Matrix m k ℝ) (ξ : n → ℝ),
        Twin.output (Twin.Γhat Ξ B) ξ = B *ᵥ Twin.beta Ξ ξ ∧
          Twin.output (Twin.Γhat Ξ B) ξ ∈ Submodule.span ℝ (Set.range Bᵀ)) ∧
    ∀ (B : Matrix m k ℝ) (j₁ j₂ : k), j₁ ≠ j₂ →
      Twin.beta Ξ (Ξᵀ j₁ + Ξᵀ j₂) = Pi.single j₁ 1 + Pi.single j₂ 1 ∧
      Twin.beta Ξ (Ξᵀ j₁ + Ξᵀ j₂) ∉ stdSimplex ℝ k ∧
      Twin.output (Twin.Γhat Ξ B) (Ξᵀ j₁ + Ξᵀ j₂) = Bᵀ j₁ + Bᵀ j₂ ∧
      Twin.beta Ξ (Ξᵀ j₁ - Ξᵀ j₂) = Pi.single j₁ 1 - Pi.single j₂ 1 ∧
      Twin.beta Ξ (Ξᵀ j₁ - Ξᵀ j₂) ∉ stdSimplex ℝ k ∧
      Twin.output (Twin.Γhat Ξ B) (Ξᵀ j₁ - Ξᵀ j₂) = Bᵀ j₁ - Bᵀ j₂) ∧
  ∀ (n k : Type) [Fintype n] [Fintype k] [DecidableEq k] (Ξ : Matrix n k ℝ),
    Twin.FullColumnRank Ξ →
    ∃ coeff : (n → ℝ) → (k → ℝ), ∀ (m : Type) (B : Matrix m k ℝ) (ξ : n → ℝ),
      Twin.output (Twin.Γhat Ξ B) ξ = B *ᵥ coeff ξ

/-- **Corollary cor:scale** (scale control). Every mixture is no larger than some bank member,
hence than the largest one, both for the whole update and for every linear block projection `P`
(such as `ΔW ↦ ΔW_s`). In the twin, the feature vector `t ξ_j` yields the output `t ΔW^{(j)}`,
so the Euclidean norm of the output is unbounded in `t > 0` whenever `ΔW^{(j)} ≠ 0`. -/
def cor_scale : Prop :=
  (∀ (W : Type) [NormedAddCommGroup W] [NormedSpace ℝ W] (K : ℕ) (ΔW : Fin K → W),
    ∀ c ∈ simplex K, ∃ k, ‖mix ΔW c‖ ≤ ‖ΔW k‖) ∧
  (∀ (W Ws : Type) [NormedAddCommGroup W] [NormedSpace ℝ W] [NormedAddCommGroup Ws]
    [NormedSpace ℝ Ws] (P : W →ₗ[ℝ] Ws) (K : ℕ) (ΔW : Fin K → W),
    ∀ c ∈ simplex K, ∃ k, ‖P (mix ΔW c)‖ ≤ ‖P (ΔW k)‖) ∧
  (∀ (m n k : Type) [Fintype m] [Fintype n] [Fintype k] [DecidableEq k]
    (Ξ : Matrix n k ℝ) (B : Matrix m k ℝ), Twin.FullColumnRank Ξ →
    (∀ (t : ℝ) (j : k), Twin.output (Twin.Γhat Ξ B) (t • Ξᵀ j) = t • Bᵀ j) ∧
    ∀ j : k, Bᵀ j ≠ 0 → ∀ M : ℝ, ∃ t : ℝ, 0 < t ∧
      M < Twin.eucNorm (Twin.output (Twin.Γhat Ξ B) (t • Ξᵀ j)))

/-- **Proposition prop:wellposed** (gauge invariance and coefficient identifiability).
* (a) Re-gauging bank member `k` at adapted matrix `s` by any invertible `Q_{k,s}` leaves the
  served update `∑_k c_k ρ B^{(k)}_s A^{(k)}_s` (`ρ = α/r`) unchanged for every `c` and `s`; and any
  objective that sees a generator's factors only through the installed updates `B_s A_s` is
  unchanged by a per-matrix gauge change.
* (b) If the bank updates are linearly independent, `c ↦ M(c)` is injective on `Δ^{K-1}`.
* (c) With `Q = t I`, every target `(t A⋆, B⋆/t)` is the same model `B⋆A⋆`, yet the factor loss
  `‖A − t A⋆‖₁ + ‖B − B⋆/t‖₁` is not constant in `t > 0` when `A⋆ ≠ 0`. -/
def prop_wellposed : Prop :=
  (∀ (r Ω : Type) [Fintype r] [DecidableEq r] (dout din : Ω → Type) (K : ℕ) (ρ : ℝ)
    (A : Fin K → (s : Ω) → Matrix r (din s) ℝ) (B : Fin K → (s : Ω) → Matrix (dout s) r ℝ)
    (Q : Fin K → Ω → Matrix r r ℝ), (∀ k s, IsUnit (Q k s).det) →
    ∀ (c : Fin K → ℝ) (s : Ω),
      ∑ k, c k • (ρ • ((B k s * (Q k s)⁻¹) * (Q k s * A k s)))
        = ∑ k, c k • (ρ • (B k s * A k s))) ∧
  (∀ (r Ω : Type) [Fintype r] [DecidableEq r] (dout din : Ω → Type)
    (F : ((s : Ω) → Matrix (dout s) (din s) ℝ) → ℝ)
    (A : (s : Ω) → Matrix r (din s) ℝ) (B : (s : Ω) → Matrix (dout s) r ℝ)
    (Q : Ω → Matrix r r ℝ), (∀ s, IsUnit (Q s).det) →
    F (fun s => (B s * (Q s)⁻¹) * (Q s * A s)) = F (fun s => B s * A s)) ∧
  (∀ (W : Type) [AddCommGroup W] [Module ℝ W] (K : ℕ) (ΔW : Fin K → W),
    LinearIndependent ℝ ΔW → Set.InjOn (mix ΔW) (simplex K)) ∧
  (∀ (r dout din : Type) [Fintype r] [Fintype dout] [Fintype din]
    (A Astar : Matrix r din ℝ) (B Bstar : Matrix dout r ℝ), Astar ≠ 0 →
    (∀ t : ℝ, 0 < t → (t⁻¹ • Bstar) * (t • Astar) = Bstar * Astar) ∧
    ∃ t : ℝ, 0 < t ∧
      Gauge.l1 (A - t • Astar) + Gauge.l1 (B - t⁻¹ • Bstar)
        ≠ Gauge.l1 (A - Astar) + Gauge.l1 (B - Bstar))

/-- **Remark rem:wellposed** (reading the coefficients), in an inner-product adapter space with
Gram matrix `G_{jk} = ⟨ΔW^{(j)}, ΔW^{(k)}⟩`: the bank is linearly independent iff `G` is
nonsingular; under it, `M(c) = M(c') ↔ c = c'`; and always
`‖M(c) − M(c')‖² = (c − c')ᵀ G (c − c')`. -/
def rem_wellposed : Prop :=
  ∀ (W : Type) [NormedAddCommGroup W] [InnerProductSpace ℝ W] (K : ℕ) (ΔW : Fin K → W),
    ((Matrix.gram ℝ ΔW).det ≠ 0 ↔ LinearIndependent ℝ ΔW) ∧
    (LinearIndependent ℝ ΔW → ∀ c c' : Fin K → ℝ, mix ΔW c = mix ΔW c' ↔ c = c') ∧
    ∀ c c' : Fin K → ℝ,
      ‖mix ΔW c - mix ΔW c'‖ ^ 2 = (c - c') ⬝ᵥ (Matrix.gram ℝ ΔW *ᵥ (c - c'))

/-- **Eqs. delta and exact** (Section 3.2). Concatenating the factors along the rank axis gives
`B_s A_s = ∑_k c_k B^{(k)}_s A^{(k)}_s`, and with `r_out = K r`, `α_out = K α` the served update
`(α_out / r_out) B_s A_s` is `∑_k c_k ΔW^{(k)}_s` with `ΔW^{(k)}_s = (α/r) B^{(k)}_s A^{(k)}_s`
(the rank `r` is the size of the index type `r`). Averaging the factors separately gives instead
`∑_{k,j} c_k c_j B^{(k)}_s A^{(j)}_s`. -/
def eq_exact : Prop :=
  ∀ (r dout din : Type) [Fintype r] (K : ℕ) (c : Fin K → ℝ)
    (A : Fin K → Matrix r din ℝ) (B : Fin K → Matrix dout r ℝ),
    Gauge.concatB B * Gauge.concatA c A = ∑ k, c k • (B k * A k) ∧
    (∀ α : ℝ, 0 < K →
      (((K : ℝ) * α) / ((K : ℝ) * (Fintype.card r : ℝ))) • (Gauge.concatB B * Gauge.concatA c A)
        = ∑ k, c k • ((α / (Fintype.card r : ℝ)) • (B k * A k))) ∧
    (∑ k, c k • B k) * (∑ j, c j • A j) = ∑ k, ∑ j, (c k * c j) • (B k * A j)

end WebMix.Statements
