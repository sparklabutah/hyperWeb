import WebMix.Statements
import Mathlib.Analysis.Calculus.ContDiff.Operations
import Mathlib.Analysis.Convex.Mul

/-!
# Non-vacuity witnesses and convention checks (Gate A)

For every statement in `Statements.lean`, a concrete instance that satisfies all of its
hypotheses, chosen to avoid the degenerate cases of its conclusion (a target outside the hull,
a strictly growing hull, a nonconstant risk, a nonzero bank, a nonzero factor). Contradictory
hypotheses would make every conclusion provable, so each statement needs one.

Also: the teeth check of Theorem thm:hull (the Lipschitz hypothesis on `𝒦_u`, not only on the
hull, is needed for the first bound), and checks of the Lean conventions that could change what
a statement means (`1/K` is real division, the norm of `PaperSpace` is Euclidean, `eucNorm` is
the Euclidean norm).
-/

open Matrix Set

namespace WebMix.Sanity

/-! ### A two-adapter bank on the real line -/

/-- The bank `{ΔW^{(1)}, ΔW^{(2)}} = {0, 1}` in `W = ℝ`. -/
def bank2 : Fin 2 → ℝ := ![0, 1]

/-- `ℋ_2 = M(Δ^1) ⊆ [0, 1]` for the bank `{0, 1}`. -/
lemma hull_bank2_subset : hull bank2 ⊆ Icc 0 1 := by
  rintro _ ⟨c, hc, rfl⟩
  have h0 := hc.1 0
  have h1 := hc.1 1
  have hs : c 0 + c 1 = 1 := by simpa [Fin.sum_univ_two] using hc.2
  simp only [mix, Fin.sum_univ_two, bank2, Matrix.cons_val_zero, Matrix.cons_val_one,
    Matrix.cons_val_fin_one, smul_eq_mul, mul_zero, mul_one, zero_add, mem_Icc]
  constructor <;> linarith

/-- Both bank members lie in their hull. -/
lemma bank2_mem_hull (k : Fin 2) : bank2 k ∈ hull bank2 :=
  ⟨Pi.single k 1, single_mem_stdSimplex ℝ k, by simp [mix, Pi.single_apply]⟩

/-- The target `ΔW^{(u)} = 2` lies at distance at least `1` from `ℋ_2 = [0, 1]`. -/
lemma one_le_infDist_two : 1 ≤ Metric.infDist (2 : ℝ) (hull bank2) := by
  rw [Metric.le_infDist ⟨_, bank2_mem_hull 0⟩]
  intro y hy
  have := hull_bank2_subset hy
  rw [Real.dist_eq, abs_of_pos (by linarith [this.2])]
  linarith [this.2]

/-- `thm_hull`, `rem_hull`: `K = 2 > 0`, the nonconstant risk `R = id` is `1`-Lipschitz on
`𝒦_u`, and the target is outside the hull (`dist > 0`), so neither bound is trivial. -/
example : 0 < 2 ∧ LipschitzOnWith 1 (fun x : ℝ => x) (Ku bank2 2) ∧
    0 < Metric.infDist (2 : ℝ) (hull bank2) :=
  ⟨two_pos, LipschitzWith.id.lipschitzOnWith, lt_of_lt_of_le one_pos one_le_infDist_two⟩

/-- `ass_lip`: the nonconstant risk `R(x) = x²` is `C¹`. -/
example : ContDiff ℝ 1 (fun x : ℝ => x ^ 2) := contDiff_id.pow 2

/-- `cor_jensen`: the strictly convex risk `R(x) = x²` is convex on `𝒦_u`. -/
example : ConvexOn ℝ (Ku bank2 2) (fun x : ℝ => x ^ 2) :=
  (Even.convexOn_pow even_two).subset (subset_univ _) (convex_convexHull ℝ _)

/-- `cor_growth`: `K = 2 > 0`, the risk `R = id` is continuous, and appending `2` strictly
enlarges the hull, so the inclusion `ℋ_2 ⊆ ℋ_3` is not an equality. -/
example : 0 < 2 ∧ ContinuousOn (fun x : ℝ => x) (hull (Fin.snoc bank2 2)) ∧
    (2 : ℝ) ∈ hull (Fin.snoc bank2 2) ∧ (2 : ℝ) ∉ hull bank2 := by
  refine ⟨two_pos, continuousOn_id, ⟨Pi.single (Fin.last 2) 1,
    single_mem_stdSimplex ℝ _, ?_⟩, fun h => ?_⟩
  · simp [mix, Pi.single_apply]
    exact Fin.snoc_last (α := fun _ => ℝ) (x := (2 : ℝ)) (p := bank2)
  · have := (hull_bank2_subset h).2
    norm_num at this

/-! ### Affine logits -/

/-- `rem_jensen`: nonnegative weights, a nonempty vocabulary, and logits that depend on the
update (`ΔW ↦ ΔW · e_0`). -/
example : ∃ (A : Fin 1 → (ℝ →ᵃ[ℝ] (Fin 2 → ℝ))) (w : Fin 1 → ℝ),
    (∀ i, 0 ≤ w i) ∧ A 0 1 ≠ A 0 0 := by
  refine ⟨fun _ => (LinearMap.smulRight LinearMap.id (Pi.single 0 1 : Fin 2 → ℝ)).toAffineMap,
    fun _ => 1, fun _ => zero_le_one, fun h => ?_⟩
  have := congrFun h 0
  simp at this

/-! ### The twin -/

/-- `prop_twin`, `cor_scale`: the identity feature matrix has full column rank, satisfies the
Rayleigh bound with `L = 1`, the step `η = 1/2` obeys `ηL < 1`, two distinct interfaces exist,
and the bank `𝓑 = I` has a nonzero column. -/
example : Twin.FullColumnRank (1 : Matrix (Fin 2) (Fin 2) ℝ) ∧
    (∀ w : Fin 2 → ℝ, (w ᵥ* (1 : Matrix (Fin 2) (Fin 2) ℝ)) ⬝ᵥ
      (w ᵥ* (1 : Matrix (Fin 2) (Fin 2) ℝ)) ≤ 1 * (w ⬝ᵥ w)) ∧
    (0 : ℝ) < 1 / 2 ∧ (1 / 2 : ℝ) * 1 < 1 ∧ (0 : Fin 2) ≠ 1 ∧
    (1 : Matrix (Fin 2) (Fin 2) ℝ)ᵀ 0 ≠ 0 := by
  refine ⟨fun x y hxy => by simpa using hxy, fun w => by simp, by norm_num, by norm_num,
    Fin.zero_ne_one', fun h => ?_⟩
  have := congrFun h 0
  simp at this

/-! ### Gauge -/

/-- `prop_wellposed`: an invertible gauge, a linearly independent bank, and `A⋆ ≠ 0`. -/
example : IsUnit (1 : Matrix (Fin 2) (Fin 2) ℝ).det ∧
    LinearIndependent ℝ (Pi.basisFun ℝ (Fin 2)) ∧ (1 : Matrix (Fin 1) (Fin 1) ℝ) ≠ 0 :=
  ⟨by simp, (Pi.basisFun ℝ (Fin 2)).linearIndependent, one_ne_zero⟩

/-- `rem_wellposed`: the paper's space is an inner-product space, and it has a linearly
independent bank (its standard basis). -/
example : LinearIndependent ℝ (EuclideanSpace.basisFun (Fin 2) ℝ) :=
  (EuclideanSpace.basisFun (Fin 2) ℝ).toBasis.linearIndependent

/-- `eq_exact`: its only hypothesis, `0 < K`, holds for a two-adapter bank. -/
example : 0 < 2 := two_pos

/-! ### Teeth of Theorem thm:hull -/

/-- Without the Lipschitz hypothesis on `𝒦_u` the first bound of Eq. thm:bounds fails. The bank
`{1}` in `ℝ`, the target `0` and the risk `R(x) = 10x`: `R` is `1`-Lipschitz on the hull `{1}`
(any function is Lipschitz on a point), yet the best mixture has approximation term
`R(1) − R(0) = 10 > 1 · dist(0, ℋ_1) = 1`. (`R` is `10`-Lipschitz on `𝒦_u = [0, 1]`, not
`1`-Lipschitz.) -/
example : LipschitzOnWith 1 (fun x : ℝ => 10 * x) (hull ![(1 : ℝ)]) ∧
    ∀ c ∈ simplex 1, ¬ ((fun x : ℝ => 10 * x) (mix ![(1 : ℝ)] c) - (fun x : ℝ => 10 * x) 0
      ≤ (1 : NNReal) * Metric.infDist (0 : ℝ) (hull ![(1 : ℝ)])) := by
  have hhull : hull ![(1 : ℝ)] = {1} := by
    ext x
    constructor
    · rintro ⟨c, hc, rfl⟩
      have hs : c 0 = 1 := by simpa using hc.2
      simp [mix, hs]
    · rintro rfl
      exact ⟨Pi.single 0 1, single_mem_stdSimplex ℝ 0, by simp [mix]⟩
  refine ⟨?_, fun c hc h => ?_⟩
  · rw [hhull]
    intro x hx y hy
    rw [mem_singleton_iff.1 hx, mem_singleton_iff.1 hy]
    simp
  · have hs : c 0 = 1 := by simpa using hc.2
    rw [hhull, Metric.infDist_singleton] at h
    simp [mix, hs] at h

/-! ### Teeth of Remark rem:jensen -/

/-- Logits that are affine in each of two adapted parameters separately, but bilinear jointly, can
make the risk nonconvex. With logits `ℓ(a, b) = (ab, 0)` and target `1`, the risk is
`log(e^{ab} + 1)`: at the midpoint `(0, 0)` of `(1, −1)` and `(−1, 1)` it is `log 2`, more than its
value `log(e^{-1} + 1)` at both endpoints. So the convexity condition of rem:jensen needs the
logits to be affine in the whole update. -/
theorem separate_affinity_not_convex :
    ¬ ConvexOn ℝ Set.univ (fun p : ℝ × ℝ => nll (1 : Fin 2) ![p.1 * p.2, 0]) := by
  intro h
  -- convexity between `(1, −1)` and `(−1, 1)` with weights `1/2, 1/2`
  have hmid := h.2 (Set.mem_univ ((1 : ℝ), (-1 : ℝ))) (Set.mem_univ ((-1 : ℝ), (1 : ℝ)))
    (show (0 : ℝ) ≤ 1 / 2 by norm_num) (show (0 : ℝ) ≤ 1 / 2 by norm_num) (by norm_num)
  -- evaluate: `log 2` at the midpoint, `log(e^{-1} + 1)` at both endpoints
  norm_num [nll, lse, Fin.sum_univ_two] at hmid
  -- `e^{-1} + 1 < 2`, so `log(e^{-1} + 1) < log 2`
  have hlt : Real.log (Real.exp (-1) + 1) < Real.log 2 :=
    Real.log_lt_log (by positivity)
      (by linarith [Real.exp_lt_one_iff.2 (show (-1 : ℝ) < 0 by norm_num)])
  linarith

/-! ### Lean conventions that could change a statement's meaning -/

/-- `𝟏/K` is real division: `uniform 2` is `1/2` in every coordinate, not `0`. -/
example : uniform 2 0 = 1 / 2 := by simp [uniform]

/-- The paper's adapter space is an inner-product (hence normed) space over `ℝ`. -/
example (Ω : Type) [Fintype Ω] (dout din : Ω → ℕ) :
    Nonempty (InnerProductSpace ℝ (PaperSpace Ω dout din)) :=
  ⟨inferInstance⟩

/-- Its norm is the Euclidean norm on all entries, `‖ΔW‖ = (∑_s ‖ΔW_s‖_F²)^{1/2}`. -/
example (Ω : Type) [Fintype Ω] (dout din : Ω → ℕ) (x : PaperSpace Ω dout din) :
    ‖x‖ = Real.sqrt (∑ i, ‖x.ofLp i‖ ^ 2) :=
  EuclideanSpace.norm_eq x

/-- `Twin.eucNorm` is the Euclidean norm (Mathlib's default norm on `Fin 2 → ℝ` is the sup
norm, which would give `4` here). -/
example : Twin.eucNorm (![3, 4] : Fin 2 → ℝ) = 5 := by
  rw [Twin.eucNorm, show (![3, 4] : Fin 2 → ℝ) ⬝ᵥ ![3, 4] = 5 ^ 2 by
    simp [dotProduct, Fin.sum_univ_two]; norm_num]
  exact Real.sqrt_sq (by norm_num)

/-- `Twin.loss` sums the squared errors of all entries: for `Ξ = I`, `𝓑 = 0`, `Γ = [[3, 4]]`
it is `3² + 4² = 25`. -/
example : Twin.loss (1 : Matrix (Fin 2) (Fin 2) ℝ) (0 : Matrix (Fin 1) (Fin 2) ℝ)
    (Matrix.of ![![3, 4]]) = 25 := by
  simp [Twin.loss, Twin.frob2, Twin.finner, Fin.sum_univ_two]
  norm_num

end WebMix.Sanity
