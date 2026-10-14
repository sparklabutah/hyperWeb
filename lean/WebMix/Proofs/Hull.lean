import WebMix.Statements
import Mathlib.Analysis.Convex.Jensen
import Mathlib.Analysis.Convex.Topology
import Mathlib.Topology.Order.Compact

/-!
# The hull of the bank

Proofs for Theorem thm:hull, Remark rem:hull, Corollaries cor:growth and cor:jensen, the mixture
half of Corollary cor:scale, Proposition prop:wellposed (b) and Remark rem:wellposed (Appendix D
of the paper).

Strategy. The mixture `M` is linear, so `ℋ_K = M(Δ^{K-1})` is the image of the compact convex
simplex: it is compact, convex and equal to `conv{ΔW^{(k)}}`, and the vertex `e_k` maps to
`ΔW^{(k)}`. Theorem thm:hull follows the paper's proof step by step: existence by compactness, the
decomposition by adding and subtracting `ℛ_u(M(c⋆))`, the first bound by projecting `ΔW^{(u)}` onto
`ℋ_K` and applying the Lipschitz assumption once, the second bound by evaluating at the vertices.
Growing the bank embeds `c ↦ (c, 0)`. The convex twin is Jensen's inequality. Scale control is the
triangle inequality. Identifiability is linear independence, and the Gram identities expand the
inner product of two mixtures.
-/

open Set Filter Topology Matrix

namespace WebMix

/-! ### The simplex and the mixture (no topology) -/

/-- The vertex `e_k` of the simplex. -/
lemma vertex_mem_simplex {K : ℕ} (k : Fin K) : (Pi.single k (1 : ℝ) : Fin K → ℝ) ∈ simplex K :=
  single_mem_stdSimplex ℝ k

/-- The uniform vector `𝟏/K` lies in the simplex when `K ≥ 1`. -/
lemma uniform_mem_simplex {K : ℕ} (hK : 0 < K) : uniform K ∈ simplex K := by
  refine ⟨fun _ => by unfold uniform; positivity, ?_⟩
  have : (K : ℝ) ≠ 0 := by exact_mod_cast hK.ne'
  simp [uniform, Finset.sum_const, Finset.card_univ, Fintype.card_fin, this]

/-- `(c, 0)` lies in `Δ^K` when `c ∈ Δ^{K-1}`. -/
lemma snoc_zero_mem_simplex {K : ℕ} {c : Fin K → ℝ} (hc : c ∈ simplex K) :
    (Fin.snoc c 0 : Fin (K + 1) → ℝ) ∈ simplex (K + 1) := by
  refine ⟨fun i => ?_, ?_⟩
  · refine Fin.lastCases ?_ (fun j => ?_) i
    · simp
    · simpa using hc.1 j
  · simpa [Fin.sum_univ_castSucc] using hc.2

section Mixture

variable {W : Type*} [AddCommGroup W] [Module ℝ W]

/-- `M` as a linear map. -/
def mixL {K : ℕ} (ΔW : Fin K → W) : (Fin K → ℝ) →ₗ[ℝ] W := Fintype.linearCombination ℝ ΔW

/-- `mixL ΔW c = M(c)`. -/
lemma mixL_apply {K : ℕ} (ΔW : Fin K → W) (c : Fin K → ℝ) : mixL ΔW c = mix ΔW c := rfl

/-- `M(e_k) = ΔW^{(k)}`. -/
lemma mix_vertex {K : ℕ} (ΔW : Fin K → W) (k : Fin K) : mix ΔW (Pi.single k 1) = ΔW k := by
  simp [mix, Pi.single_apply, ite_smul]

/-- Every bank update lies in the hull. -/
lemma bank_mem_hull {K : ℕ} (ΔW : Fin K → W) (k : Fin K) : ΔW k ∈ hull ΔW :=
  ⟨Pi.single k 1, vertex_mem_simplex k, mix_vertex ΔW k⟩

/-- The hull is nonempty when `K ≥ 1`. -/
lemma hull_nonempty {K : ℕ} (ΔW : Fin K → W) (hK : 0 < K) : (hull ΔW).Nonempty :=
  ⟨_, bank_mem_hull ΔW ⟨0, hK⟩⟩

/-- Eq. thm:mix: `ℋ_K = M(Δ^{K-1}) = conv{ΔW^{(1)}, …, ΔW^{(K)}}`. -/
theorem hull_eq_convexHull {K : ℕ} (ΔW : Fin K → W) :
    hull ΔW = convexHull ℝ (Set.range ΔW) := by
  -- `Δ^{K-1}` is the convex hull of the basis vectors `e_k`, a linear map sends a convex hull to
  -- the convex hull of the images, and `M(e_k) = ΔW^{(k)}`
  have h1 : simplex K = convexHull ℝ (Set.range fun i j : Fin K => if i = j then (1 : ℝ) else 0) :=
    (convexHull_basis_eq_stdSimplex ℝ (Fin K)).symm
  have h2 : hull ΔW = mixL ΔW '' simplex K := rfl
  have h3 : (mixL ΔW ∘ fun i j : Fin K => if i = j then (1 : ℝ) else 0) = ΔW := by
    funext i
    simp [Function.comp, mixL_apply, mix, ite_smul]
  rw [h2, h1, LinearMap.image_convexHull, ← Set.range_comp, h3]

/-- `ℋ_K` is convex. -/
theorem hull_convex {K : ℕ} (ΔW : Fin K → W) : Convex ℝ (hull ΔW) := by
  rw [hull_eq_convexHull]; exact convex_convexHull ℝ _

/-- Remark rem:hull: the uniform mixture `M(𝟏/K)` is an element of `ℋ_K`. -/
lemma mix_uniform_mem_hull {K : ℕ} (ΔW : Fin K → W) (hK : 0 < K) : mix ΔW (uniform K) ∈ hull ΔW :=
  ⟨uniform K, uniform_mem_simplex hK, rfl⟩

/-- `M(c) − M(c') = M(c − c')`. -/
lemma mix_sub {K : ℕ} (ΔW : Fin K → W) (c c' : Fin K → ℝ) :
    mix ΔW c - mix ΔW c' = mix ΔW (c - c') := by
  simp [mix, sub_smul, Finset.sum_sub_distrib]

/-! #### Proposition prop:wellposed (b): coefficient identifiability -/

/-- If the bank updates are linearly independent, `c ↦ M(c)` is injective on all of `ℝ^K`. -/
theorem mix_injective {K : ℕ} (ΔW : Fin K → W) (h : LinearIndependent ℝ ΔW) :
    Function.Injective (mix ΔW) := by
  intro c c' hcc'
  have hsum : ∑ k, (c k - c' k) • ΔW k = 0 := by
    simp only [sub_smul, Finset.sum_sub_distrib]
    exact sub_eq_zero.2 hcc'
  have := Fintype.linearIndependent_iff.1 h _ hsum
  funext k
  exact sub_eq_zero.1 (this k)

/-- Proposition prop:wellposed (b): in particular `c ↦ M(c)` is injective on `Δ^{K-1}`. -/
theorem mix_injOn_simplex {K : ℕ} (ΔW : Fin K → W) (h : LinearIndependent ℝ ΔW) :
    Set.InjOn (mix ΔW) (simplex K) :=
  (mix_injective ΔW h).injOn

/-- Remark rem:wellposed: two generated adapters differ iff their coefficient vectors differ. -/
theorem mix_eq_iff {K : ℕ} (ΔW : Fin K → W) (h : LinearIndependent ℝ ΔW) (c c' : Fin K → ℝ) :
    mix ΔW c = mix ΔW c' ↔ c = c' :=
  (mix_injective ΔW h).eq_iff

/-! #### Corollary cor:growth: the embedding `c ↦ (c, 0)` -/

/-- `M_{K+1}((c, 0)) = M_K(c)`. -/
theorem mix_snoc_zero {K : ℕ} (ΔW : Fin K → W) (new : W) (c : Fin K → ℝ) :
    mix (Fin.snoc ΔW new) (Fin.snoc c 0) = mix ΔW c := by
  simp [mix, Fin.sum_univ_castSucc]

/-- `ℋ_K ⊆ ℋ_{K+1}`. -/
theorem hull_subset_hull_snoc {K : ℕ} (ΔW : Fin K → W) (new : W) :
    hull ΔW ⊆ hull (Fin.snoc ΔW new) := by
  rintro _ ⟨c, hc, rfl⟩
  exact ⟨Fin.snoc c 0, snoc_zero_mem_simplex hc, mix_snoc_zero ΔW new c⟩

end Mixture

section Topology

variable {W : Type*} [AddCommGroup W] [Module ℝ W] [TopologicalSpace W]
  [ContinuousAdd W] [ContinuousSMul ℝ W]

/-- `M` is continuous. -/
lemma continuous_mix {K : ℕ} (ΔW : Fin K → W) : Continuous (mix ΔW) :=
  continuous_finsetSum _ fun k _ => (continuous_apply k).smul continuous_const

/-- `ℋ_K` is compact, as the continuous image of the compact simplex. -/
theorem isCompact_hull {K : ℕ} (ΔW : Fin K → W) : IsCompact (hull ΔW) :=
  (isCompact_stdSimplex ℝ (Fin K)).image (continuous_mix ΔW)

end Topology

section Risk

variable {W : Type*} [NormedAddCommGroup W] [NormedSpace ℝ W]

/-! #### The set `𝒦_u` of Assumption ass:lip -/

/-- `𝒦_u = conv(ℋ_K ∪ {ΔW^{(u)}}) = conv({ΔW^{(u)}} ∪ {ΔW^{(k)}})`. -/
lemma Ku_eq_convexHull_insert {K : ℕ} (ΔW : Fin K → W) (ΔWu : W) :
    Ku ΔW ΔWu = convexHull ℝ (insert ΔWu (Set.range ΔW)) := by
  rw [Ku, hull_eq_convexHull, convexHull_convexHull_union_left, Set.union_singleton]

/-- `ℋ_K ⊆ 𝒦_u`. -/
lemma hull_subset_Ku {K : ℕ} (ΔW : Fin K → W) (ΔWu : W) : hull ΔW ⊆ Ku ΔW ΔWu := by
  unfold Ku
  exact Set.subset_union_left.trans (subset_convexHull ℝ _)

/-- `ΔW^{(u)} ∈ 𝒦_u`. -/
lemma target_mem_Ku {K : ℕ} (ΔW : Fin K → W) (ΔWu : W) : ΔWu ∈ Ku ΔW ΔWu := by
  unfold Ku
  exact subset_convexHull ℝ _ (Set.mem_union_right _ (Set.mem_singleton _))

/-- `ΔW^{(k)} ∈ 𝒦_u`. -/
lemma bank_mem_Ku {K : ℕ} (ΔW : Fin K → W) (ΔWu : W) (k : Fin K) : ΔW k ∈ Ku ΔW ΔWu :=
  hull_subset_Ku ΔW ΔWu (bank_mem_hull ΔW k)

/-- `𝒦_u` is convex, being a convex hull. -/
lemma convex_Ku {K : ℕ} (ΔW : Fin K → W) (ΔWu : W) : Convex ℝ (Ku ΔW ΔWu) := by
  unfold Ku
  exact convex_convexHull ℝ _

/-- `𝒦_u` is compact, being the convex hull of finitely many points. -/
lemma isCompact_Ku {K : ℕ} (ΔW : Fin K → W) (ΔWu : W) : IsCompact (Ku ΔW ΔWu) := by
  rw [Ku_eq_convexHull_insert]
  exact ((Set.finite_range ΔW).insert ΔWu).isCompact_convexHull ℝ

/-! #### Theorem thm:hull -/

/-- Step "existence" of the proof of thm:hull: a best mixture `c⋆ ∈ argmin_{Δ^{K-1}} ℛ_u ∘ M`
exists. Only continuity of the risk on the hull is used. -/
theorem exists_best_mixture {K : ℕ} (ΔW : Fin K → W) (R : W → ℝ)
    (hR : ContinuousOn R (hull ΔW)) (hK : 0 < K) :
    ∃ cstar ∈ simplex K, IsMinOn (fun c => R (mix ΔW c)) (simplex K) cstar :=
  (isCompact_stdSimplex ℝ (Fin K)).exists_isMinOn ⟨_, vertex_mem_simplex ⟨0, hK⟩⟩
    (hR.comp (continuous_mix ΔW).continuousOn (Set.mapsTo_image _ _))

/-- Step "decomposition" (Eq. thm:decomp): adding and subtracting `ℛ_u(M(c⋆))`. -/
theorem risk_decomposition {K : ℕ} (ΔW : Fin K → W) (ΔWu : W) (R : W → ℝ)
    (cstar chat : Fin K → ℝ) :
    R (mix ΔW chat) - R ΔWu
      = (R (mix ΔW chat) - R (mix ΔW cstar)) + (R (mix ΔW cstar) - R ΔWu) := by ring

/-- The routing term is nonnegative because `ĉ ∈ Δ^{K-1}` and `c⋆` minimises over `Δ^{K-1}`. -/
theorem routing_nonneg {K : ℕ} (ΔW : Fin K → W) (R : W → ℝ) {cstar chat : Fin K → ℝ}
    (hmin : IsMinOn (fun c => R (mix ΔW c)) (simplex K) cstar) (hc : chat ∈ simplex K) :
    0 ≤ R (mix ΔW chat) - R (mix ΔW cstar) :=
  sub_nonneg.2 (isMinOn_iff.1 hmin _ hc)

/-- Step "first bound" of the proof of thm:hull: with `Π = M(c_Π)` the projection of `ΔW^{(u)}`
onto the compact set `ℋ_K`, `ℛ_u(M(c⋆)) ≤ ℛ_u(Π) ≤ ℛ_u(ΔW^{(u)}) + Λ_u ‖Π − ΔW^{(u)}‖` and
`‖Π − ΔW^{(u)}‖ = dist(ΔW^{(u)}, ℋ_K)`. -/
theorem approximation_bound {K : ℕ} (ΔW : Fin K → W) (ΔWu : W) (R : W → ℝ) (Λ : NNReal)
    (hLip : LipschitzOnWith Λ R (Ku ΔW ΔWu)) (hK : 0 < K) {cstar : Fin K → ℝ}
    (hmin : IsMinOn (fun c => R (mix ΔW c)) (simplex K) cstar) :
    R (mix ΔW cstar) - R ΔWu ≤ Λ * Metric.infDist ΔWu (hull ΔW) := by
  -- the projection `Π = M(c_Π)` of `ΔW^{(u)}` onto `ℋ_K`, which attains the distance
  obtain ⟨_, ⟨cP, hcP, rfl⟩, hdist⟩ :=
    (isCompact_hull ΔW).exists_infDist_eq_dist (hull_nonempty ΔW hK) ΔWu
  -- `c⋆` is at least as good as `c_Π`
  have hstar : R (mix ΔW cstar) ≤ R (mix ΔW cP) := isMinOn_iff.1 hmin _ hcP
  -- the Lipschitz assumption between the two points `Π, ΔW^{(u)} ∈ 𝒦_u`
  have hlip : |R (mix ΔW cP) - R ΔWu| ≤ Λ * dist (mix ΔW cP) ΔWu := by
    simpa [Real.dist_eq] using
      hLip.dist_le_mul _ (hull_subset_Ku ΔW ΔWu ⟨cP, hcP, rfl⟩) _ (target_mem_Ku ΔW ΔWu)
  calc R (mix ΔW cstar) - R ΔWu
      ≤ R (mix ΔW cP) - R ΔWu := by linarith
    _ ≤ Λ * dist (mix ΔW cP) ΔWu := (le_abs_self _).trans hlip
    _ = Λ * Metric.infDist ΔWu (hull ΔW) := by rw [hdist, dist_comm]

/-- Step "second bound" of the proof of thm:hull: `M(e_k) = ΔW^{(k)}` with `e_k ∈ Δ^{K-1}`, so
`ℛ_u(M(c⋆)) ≤ ℛ_u(ΔW^{(k)})` for every `k`. -/
theorem best_mixture_le_bank {K : ℕ} (ΔW : Fin K → W) (R : W → ℝ) {cstar : Fin K → ℝ}
    (hmin : IsMinOn (fun c => R (mix ΔW c)) (simplex K) cstar) (k : Fin K) :
    R (mix ΔW cstar) ≤ R (ΔW k) := by
  simpa [mix_vertex] using isMinOn_iff.1 hmin _ (vertex_mem_simplex k)

/-- **Theorem thm:hull**, assembled from the four steps above. -/
theorem hull_decomposition {K : ℕ} (ΔW : Fin K → W) (ΔWu : W) (R : W → ℝ) (Λ : NNReal)
    (hLip : LipschitzOnWith Λ R (Ku ΔW ΔWu)) (hK : 0 < K) :
    ∃ cstar ∈ simplex K, IsMinOn (fun c => R (mix ΔW c)) (simplex K) cstar ∧
      (∀ chat ∈ simplex K,
        R (mix ΔW chat) - R ΔWu
          = (R (mix ΔW chat) - R (mix ΔW cstar)) + (R (mix ΔW cstar) - R ΔWu) ∧
        0 ≤ R (mix ΔW chat) - R (mix ΔW cstar)) ∧
      R (mix ΔW cstar) - R ΔWu ≤ Λ * Metric.infDist ΔWu (hull ΔW) ∧
      ∀ k, R (mix ΔW cstar) ≤ R (ΔW k) := by
  -- existence uses continuity on `ℋ_K ⊆ 𝒦_u`, which the Lipschitz assumption provides
  obtain ⟨cstar, hcstar, hmin⟩ :=
    exists_best_mixture ΔW R (hLip.continuousOn.mono (hull_subset_Ku ΔW ΔWu)) hK
  exact ⟨cstar, hcstar, hmin,
    fun chat hc => ⟨risk_decomposition ΔW ΔWu R cstar chat, routing_nonneg ΔW R hmin hc⟩,
    approximation_bound ΔW ΔWu R Λ hLip hK hmin, best_mixture_le_bank ΔW R hmin⟩

/-! #### Remark rem:hull -/

/-- The distance to the hull is attained at some `c_P ∈ Δ^{K-1}` and bounds every
`‖ΔW^{(u)} − M(c)‖`, `c ∈ Δ^{K-1}`, from below. -/
theorem infDist_hull_attained {K : ℕ} (ΔW : Fin K → W) (ΔWu : W) (hK : 0 < K) :
    (∃ cP ∈ simplex K, ‖ΔWu - mix ΔW cP‖ = Metric.infDist ΔWu (hull ΔW)) ∧
      ∀ c ∈ simplex K, Metric.infDist ΔWu (hull ΔW) ≤ ‖ΔWu - mix ΔW c‖ := by
  refine ⟨?_, fun c hc => ?_⟩
  · obtain ⟨_, ⟨cP, hcP, rfl⟩, hdist⟩ :=
      (isCompact_hull ΔW).exists_infDist_eq_dist (hull_nonempty ΔW hK) ΔWu
    exact ⟨cP, hcP, by rw [hdist, dist_eq_norm]⟩
  · rw [← dist_eq_norm]
    exact Metric.infDist_le_dist_of_mem ⟨c, hc, rfl⟩

/-! #### Corollary cor:growth -/

/-- `dist(ΔW, ℋ_{K+1}) ≤ dist(ΔW, ℋ_K)`: an infimum over a superset is no larger. -/
theorem infDist_hull_snoc_le {K : ℕ} (ΔW : Fin K → W) (new : W) (hK : 0 < K) (x : W) :
    Metric.infDist x (hull (Fin.snoc ΔW new)) ≤ Metric.infDist x (hull ΔW) :=
  Metric.infDist_le_infDist_of_subset (hull_subset_hull_snoc ΔW new) (hull_nonempty ΔW hK)

/-- `ℛ⋆_v(K+1) ≤ ℛ⋆_v(K)`: a best mixture over the grown bank is at least as good as any mixture
over the old bank, since `(c, 0) ∈ Δ^K`. -/
theorem best_mixture_snoc_le {K : ℕ} (R : W → ℝ) (ΔW : Fin K → W) (new : W)
    {c : Fin K → ℝ} (hc : c ∈ simplex K) {c' : Fin (K + 1) → ℝ}
    (hmin' : IsMinOn (fun c => R (mix (Fin.snoc ΔW new) c)) (simplex (K + 1)) c') :
    R (mix (Fin.snoc ΔW new) c') ≤ R (mix ΔW c) := by
  have := isMinOn_iff.1 hmin' _ (snoc_zero_mem_simplex hc)
  simpa [mix_snoc_zero] using this

/-- Remark rem:growth: when the bank grows, the realised risk on an interface can rise only through
the routing term. For router outputs `ĉ` (old bank) and `ĉ'` (grown bank), any `c⋆ ∈ Δ^{K-1}` and a
best mixture `c⋆'` of the grown bank, the rise of the realised risk is at most the rise of the
routing term, because `ℛ(M_{K+1}(c⋆')) ≤ ℛ(M_K(c⋆))`. -/
theorem realized_rise_le_routing_rise {K : ℕ} (R : W → ℝ) (ΔW : Fin K → W) (new : W)
    {chat cstar : Fin K → ℝ} (hcstar : cstar ∈ simplex K) {chat' cstar' : Fin (K + 1) → ℝ}
    (hmin' : IsMinOn (fun c => R (mix (Fin.snoc ΔW new) c)) (simplex (K + 1)) cstar') :
    R (mix (Fin.snoc ΔW new) chat') - R (mix ΔW chat)
      ≤ (R (mix (Fin.snoc ΔW new) chat') - R (mix (Fin.snoc ΔW new) cstar'))
        - (R (mix ΔW chat) - R (mix ΔW cstar)) := by
  have hbest := best_mixture_snoc_le R ΔW new hcstar hmin'
  linarith

/-! #### Corollary cor:jensen -/

/-- Jensen's inequality on `𝒦_u`: every mixture is at least as good as the corresponding average
of the transferred bank adapters. -/
theorem jensen_mixture {K : ℕ} (ΔW : Fin K → W) (ΔWu : W) (R : W → ℝ)
    (hconv : ConvexOn ℝ (Ku ΔW ΔWu) R) {c : Fin K → ℝ} (hc : c ∈ simplex K) :
    R (mix ΔW c) ≤ ∑ k, c k * R (ΔW k) := by
  have := hconv.map_sum_le (t := Finset.univ) (w := c) (p := ΔW)
    (fun k _ => hc.1 k) hc.2 (fun k _ => bank_mem_Ku ΔW ΔWu k)
  simpa [mix] using this

/-- The case `c = 𝟏/K`. -/
theorem jensen_uniform {K : ℕ} (ΔW : Fin K → W) (ΔWu : W) (R : W → ℝ)
    (hconv : ConvexOn ℝ (Ku ΔW ΔWu) R) (hK : 0 < K) :
    R (mix ΔW (uniform K)) ≤ (1 / (K : ℝ)) * ∑ k, R (ΔW k) := by
  have := jensen_mixture ΔW ΔWu R hconv (uniform_mem_simplex hK)
  simpa [uniform, Finset.mul_sum] using this

/-! #### Corollary cor:scale, mixture half -/

/-- Triangle inequality: any common bound `m` on the bank bounds every mixture. -/
theorem norm_mix_le {K : ℕ} (ΔW : Fin K → W) {c : Fin K → ℝ} (hc : c ∈ simplex K)
    {m : ℝ} (hm : ∀ k, ‖ΔW k‖ ≤ m) : ‖mix ΔW c‖ ≤ m := by
  calc ‖∑ k, c k • ΔW k‖ ≤ ∑ k, ‖c k • ΔW k‖ := norm_sum_le _ _
    _ = ∑ k, c k * ‖ΔW k‖ := by simp [norm_smul, abs_of_nonneg (hc.1 _)]
    _ ≤ ∑ k, c k * m := Finset.sum_le_sum fun k _ => mul_le_mul_of_nonneg_left (hm k) (hc.1 k)
    _ = m := by rw [← Finset.sum_mul, hc.2, one_mul]

/-- The simplex is empty for `K = 0`, so `c ∈ Δ^{K-1}` forces a bank member to exist. -/
lemma univ_nonempty_of_mem_simplex {K : ℕ} {c : Fin K → ℝ} (hc : c ∈ simplex K) :
    (Finset.univ : Finset (Fin K)).Nonempty := by
  rcases (Finset.univ : Finset (Fin K)).eq_empty_or_nonempty with h | h
  · have h2 := hc.2
    rw [h, Finset.sum_empty] at h2
    exact absurd h2 zero_ne_one
  · exact h

/-- `‖M(c)‖ ≤ max_k ‖ΔW^{(k)}‖`, written as `‖M(c)‖ ≤ ‖ΔW^{(k)}‖` for a largest member `k`. -/
theorem exists_norm_mix_le {K : ℕ} (ΔW : Fin K → W) {c : Fin K → ℝ} (hc : c ∈ simplex K) :
    ∃ k, ‖mix ΔW c‖ ≤ ‖ΔW k‖ := by
  obtain ⟨k, -, hk⟩ :=
    Finset.exists_max_image Finset.univ (fun k => ‖ΔW k‖) (univ_nonempty_of_mem_simplex hc)
  exact ⟨k, norm_mix_le ΔW hc fun j => hk j (Finset.mem_univ j)⟩

/-- Per-matrix version: for any linear projection `P` (such as `ΔW ↦ ΔW_s` with the Frobenius
norm), `‖P(M(c))‖ ≤ ‖P(ΔW^{(k)})‖` for a member `k` with the largest projected norm. -/
theorem exists_norm_proj_mix_le {Ws : Type*} [NormedAddCommGroup Ws] [NormedSpace ℝ Ws]
    (P : W →ₗ[ℝ] Ws) {K : ℕ} (ΔW : Fin K → W) {c : Fin K → ℝ} (hc : c ∈ simplex K) :
    ∃ k, ‖P (mix ΔW c)‖ ≤ ‖P (ΔW k)‖ := by
  have hP : P (mix ΔW c) = mix (fun k => P (ΔW k)) c := by simp [mix, map_sum, map_smul]
  rw [hP]
  exact exists_norm_mix_le (fun k => P (ΔW k)) hc

end Risk

/-! ### Inner-product identities (Remarks rem:hull and rem:wellposed) -/

section Inner

variable {W : Type*} [NormedAddCommGroup W] [InnerProductSpace ℝ W]

/-- `‖∑_k a_k ΔW^{(k)}‖² = aᵀ G a` with the Gram matrix `G_{jk} = ⟨ΔW^{(j)}, ΔW^{(k)}⟩`. -/
lemma norm_mix_sq {K : ℕ} (ΔW : Fin K → W) (a : Fin K → ℝ) :
    ‖mix ΔW a‖ ^ 2 = a ⬝ᵥ (Matrix.gram ℝ ΔW *ᵥ a) := by
  -- Mathlib: `star x ⬝ᵥ G y = ⟨∑_i x_i ΔW^{(i)}, ∑_i y_i ΔW^{(i)}⟩`, and `star x = x` over `ℝ`
  have h := Matrix.star_dotProduct_gram_mulVec (𝕜 := ℝ) ΔW a a
  rw [star_trivial] at h
  rw [h, ← real_inner_self_eq_norm_sq]
  -- `M(a)` is by definition `∑_k a_k ΔW^{(k)}`
  rfl

/-- Remark rem:wellposed: `‖M(c) − M(c')‖² = (c − c')ᵀ G (c − c')`. -/
theorem norm_mix_sub_mix_sq {K : ℕ} (ΔW : Fin K → W) (c c' : Fin K → ℝ) :
    ‖mix ΔW c - mix ΔW c'‖ ^ 2 = (c - c') ⬝ᵥ (Matrix.gram ℝ ΔW *ᵥ (c - c')) := by
  rw [mix_sub, norm_mix_sq]

/-- Remark rem:hull: `‖ΔW^{(u)} − M(c)‖² = ‖ΔW^{(u)}‖² − 2∑_k c_k⟨ΔW^{(u)}, ΔW^{(k)}⟩ + cᵀGc`, a
quadratic in `c` whose data are inner products of the stored adapters (least squares). -/
theorem norm_target_sub_mix_sq {K : ℕ} (ΔW : Fin K → W) (ΔWu : W) (c : Fin K → ℝ) :
    ‖ΔWu - mix ΔW c‖ ^ 2
      = ‖ΔWu‖ ^ 2 - 2 * ∑ k, c k * inner ℝ ΔWu (ΔW k) + c ⬝ᵥ (Matrix.gram ℝ ΔW *ᵥ c) := by
  have hcross : inner ℝ ΔWu (mix ΔW c) = ∑ k, c k * inner ℝ ΔWu (ΔW k) := by
    simp [mix, inner_sum, real_inner_smul_right]
  rw [norm_sub_sq_real, hcross, norm_mix_sq]

end Inner

end WebMix
