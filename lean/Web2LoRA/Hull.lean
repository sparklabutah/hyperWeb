import Mathlib

/-!
# The convex hull of the bank

Formalisation of Section "What the hull can reach" of the paper:

* the mixture `M(c) = ∑ c_k ΔW^{(k)}` and its reachable set `ℋ_K = M(Δ^{K-1})`;
* `ℋ_K = conv{ΔW^{(1)}, …, ΔW^{(K)}}` (Eq. thm:mix), compact and convex;
* Theorem `thm:hull` (hull decomposition of the held-out risk);
* Corollary `cor:growth` (growing the bank);
* Corollary `cor:jensen` (convex twin, Jensen);
* Corollary `cor:scale`, mixture half (scale control);
* Proposition `prop:wellposed` (b) (coefficient identifiability).

The adapter space `𝒲 = ∏_s ℝ^{d_out(s) × d_in(s)}` with the norm
`‖ΔW‖ = (∑_s ‖ΔW_s‖_F²)^{1/2}` is a finite-dimensional real normed space.  Every result
below is stated for an arbitrary real normed space `W` (the hull results need neither an
inner product nor finite dimension), so the paper's `𝒲` is an instance
(`EuclideanSpace ℝ (Σ s, Fin (d_out s) × Fin (d_in s))`, see `paper_space_instance`).
-/

open Set Filter Topology

namespace Web2LoRA

/-- The probability simplex `Δ^{K-1} = {c ∈ ℝ^K_{≥0} : ∑ c_k = 1}`. -/
abbrev simplex (K : ℕ) : Set (Fin K → ℝ) := stdSimplex ℝ (Fin K)

section Mixture

variable {W : Type*} [AddCommGroup W] [Module ℝ W]

/-- The mixture `M(c) := ∑_k c_k ΔW^{(k)}` (Eq. thm:mix). -/
def mix {K : ℕ} (ΔW : Fin K → W) (c : Fin K → ℝ) : W := ∑ k, c k • ΔW k

/-- `M` as a linear map. -/
def mixL {K : ℕ} (ΔW : Fin K → W) : (Fin K → ℝ) →ₗ[ℝ] W := Fintype.linearCombination ℝ ΔW

lemma mixL_apply {K : ℕ} (ΔW : Fin K → W) (c : Fin K → ℝ) : mixL ΔW c = mix ΔW c := rfl

/-- The reachable set `ℋ_K := M(Δ^{K-1})`. -/
def hull {K : ℕ} (ΔW : Fin K → W) : Set W := mix ΔW '' simplex K

/-- The vertex `e_k` of the simplex. -/
lemma vertex_mem_simplex {K : ℕ} (k : Fin K) : (Pi.single k (1 : ℝ) : Fin K → ℝ) ∈ simplex K := by
  refine ⟨fun j => ?_, ?_⟩
  · by_cases h : j = k
    · subst h; simp
    · simp [h]
  · simp

/-- `M(e_k) = ΔW^{(k)}`. -/
lemma mix_vertex {K : ℕ} (ΔW : Fin K → W) (k : Fin K) : mix ΔW (Pi.single k 1) = ΔW k := by
  simp [mix, Pi.single_apply, ite_smul]

lemma bank_mem_hull {K : ℕ} (ΔW : Fin K → W) (k : Fin K) : ΔW k ∈ hull ΔW :=
  ⟨Pi.single k 1, vertex_mem_simplex k, mix_vertex ΔW k⟩

lemma hull_nonempty {K : ℕ} (ΔW : Fin K → W) (hK : 0 < K) : (hull ΔW).Nonempty :=
  ⟨_, bank_mem_hull ΔW ⟨0, hK⟩⟩

/-- Eq. thm:mix: `ℋ_K = conv{ΔW^{(1)}, …, ΔW^{(K)}}`. -/
theorem hull_eq_convexHull {K : ℕ} (ΔW : Fin K → W) :
    hull ΔW = convexHull ℝ (Set.range ΔW) := by
  have h1 : simplex K = convexHull ℝ (Set.range fun i j : Fin K => if i = j then (1 : ℝ) else 0) :=
    (convexHull_basis_eq_stdSimplex ℝ (Fin K)).symm
  have h2 : hull ΔW = mixL ΔW '' simplex K := rfl
  have h3 : (mixL ΔW ∘ fun i j : Fin K => if i = j then (1 : ℝ) else 0) = ΔW := by
    funext i
    simp [Function.comp, mixL_apply, mix, ite_smul]
  rw [h2, h1, LinearMap.image_convexHull, ← Set.range_comp, h3]

theorem convex_hull' {K : ℕ} (ΔW : Fin K → W) : Convex ℝ (hull ΔW) := by
  rw [hull_eq_convexHull]; exact convex_convexHull ℝ _

end Mixture

section Topology

variable {W : Type*} [AddCommGroup W] [Module ℝ W] [TopologicalSpace W]
  [ContinuousAdd W] [ContinuousSMul ℝ W]

lemma continuous_mix {K : ℕ} (ΔW : Fin K → W) : Continuous (mix ΔW) :=
  continuous_finsetSum _ fun k _ => (continuous_apply k).smul continuous_const

theorem isCompact_hull {K : ℕ} (ΔW : Fin K → W) : IsCompact (hull ΔW) :=
  (isCompact_stdSimplex ℝ (Fin K)).image (continuous_mix ΔW)

end Topology

section Risk

variable {W : Type*} [NormedAddCommGroup W] [NormedSpace ℝ W]

/-- `𝒦_u := conv(ℋ_K ∪ {ΔW^{(u)}})`, the set on which Assumption `ass:lip` is stated. -/
def Ku {K : ℕ} (ΔW : Fin K → W) (ΔWu : W) : Set W := convexHull ℝ (insert ΔWu (Set.range ΔW))

lemma hull_subset_Ku {K : ℕ} (ΔW : Fin K → W) (ΔWu : W) : hull ΔW ⊆ Ku ΔW ΔWu := by
  rw [hull_eq_convexHull]; exact convexHull_mono (Set.subset_insert _ _)

lemma target_mem_Ku {K : ℕ} (ΔW : Fin K → W) (ΔWu : W) : ΔWu ∈ Ku ΔW ΔWu :=
  subset_convexHull ℝ _ (Set.mem_insert _ _)

lemma bank_mem_Ku {K : ℕ} (ΔW : Fin K → W) (ΔWu : W) (k : Fin K) : ΔW k ∈ Ku ΔW ΔWu :=
  subset_convexHull ℝ _ (Set.mem_insert_of_mem _ (Set.mem_range_self k))

lemma convex_Ku {K : ℕ} (ΔW : Fin K → W) (ΔWu : W) : Convex ℝ (Ku ΔW ΔWu) := convex_convexHull ℝ _

lemma isCompact_Ku {K : ℕ} (ΔW : Fin K → W) (ΔWu : W) : IsCompact (Ku ΔW ΔWu) :=
  ((Set.finite_range ΔW).insert ΔWu).isCompact_convexHull ℝ

/-! ### Theorem `thm:hull` -/

/-- Existence of a best mixture `c⋆ ∈ argmin_{c ∈ Δ^{K-1}} ℛ_u(M(c))`; only continuity of the
risk on the hull is used. -/
theorem exists_best_mixture {K : ℕ} (ΔW : Fin K → W) (R : W → ℝ)
    (hR : ContinuousOn R (hull ΔW)) (hK : 0 < K) :
    ∃ cstar ∈ simplex K, IsMinOn (fun c => R (mix ΔW c)) (simplex K) cstar :=
  (isCompact_stdSimplex ℝ (Fin K)).exists_isMinOn ⟨_, vertex_mem_simplex ⟨0, hK⟩⟩
    (hR.comp (continuous_mix ΔW).continuousOn (Set.mapsTo_image _ _))

/-- The decomposition Eq. thm:decomp is an identity. -/
theorem risk_decomposition {K : ℕ} (ΔW : Fin K → W) (ΔWu : W) (R : W → ℝ)
    (cstar chat : Fin K → ℝ) :
    R (mix ΔW chat) - R ΔWu
      = (R (mix ΔW chat) - R (mix ΔW cstar)) + (R (mix ΔW cstar) - R ΔWu) := by ring

/-- The routing term is non-negative. -/
theorem routing_nonneg {K : ℕ} (ΔW : Fin K → W) (R : W → ℝ) {cstar chat : Fin K → ℝ}
    (hmin : IsMinOn (fun c => R (mix ΔW c)) (simplex K) cstar) (hc : chat ∈ simplex K) :
    0 ≤ R (mix ΔW chat) - R (mix ΔW cstar) :=
  sub_nonneg.2 (isMinOn_iff.1 hmin _ hc)

/-- First bound of Eq. thm:bounds: the approximation term is at most
`Λ_u · dist(ΔW^{(u)}, ℋ_K)` under Assumption `ass:lip`. -/
theorem approximation_bound {K : ℕ} (ΔW : Fin K → W) (ΔWu : W) (R : W → ℝ) (Λ : NNReal)
    (hLip : LipschitzOnWith Λ R (Ku ΔW ΔWu)) (hK : 0 < K) {cstar : Fin K → ℝ}
    (hmin : IsMinOn (fun c => R (mix ΔW c)) (simplex K) cstar) :
    R (mix ΔW cstar) - R ΔWu ≤ Λ * Metric.infDist ΔWu (hull ΔW) := by
  obtain ⟨Pi₀, hPi₀, hdist⟩ := (isCompact_hull ΔW).exists_infDist_eq_dist (hull_nonempty ΔW hK) ΔWu
  obtain ⟨cP, hcP, rfl⟩ := hPi₀
  have h1 : R (mix ΔW cstar) ≤ R (mix ΔW cP) := isMinOn_iff.1 hmin _ hcP
  have h2 : dist (R (mix ΔW cP)) (R ΔWu) ≤ Λ * dist (mix ΔW cP) ΔWu :=
    hLip.dist_le_mul _ (hull_subset_Ku ΔW ΔWu ⟨cP, hcP, rfl⟩) _ (target_mem_Ku ΔW ΔWu)
  rw [Real.dist_eq] at h2
  have h3 := le_abs_self (R (mix ΔW cP) - R ΔWu)
  rw [hdist, dist_comm]
  linarith

/-- Second bound of Eq. thm:bounds: the best mixture is at least as good as every bank adapter,
hence `ℛ_u(M(c⋆)) ≤ min_k ℛ_u(ΔW^{(k)})`. -/
theorem best_mixture_le_bank {K : ℕ} (ΔW : Fin K → W) (R : W → ℝ) {cstar : Fin K → ℝ}
    (hmin : IsMinOn (fun c => R (mix ΔW c)) (simplex K) cstar) (k : Fin K) :
    R (mix ΔW cstar) ≤ R (ΔW k) := by
  simpa [mix_vertex] using isMinOn_iff.1 hmin _ (vertex_mem_simplex k)

/-- The distance to the hull is a `K`-variable least-squares problem over the simplex
(Remark `rem:hull`): it is attained at some `c_P ∈ Δ^{K-1}` and bounds every `‖ΔW^{(u)} − M(c)‖`
from below. -/
theorem infDist_hull_attained {K : ℕ} (ΔW : Fin K → W) (ΔWu : W) (hK : 0 < K) :
    (∃ cP ∈ simplex K, ‖ΔWu - mix ΔW cP‖ = Metric.infDist ΔWu (hull ΔW)) ∧
      ∀ c ∈ simplex K, Metric.infDist ΔWu (hull ΔW) ≤ ‖ΔWu - mix ΔW c‖ := by
  refine ⟨?_, fun c hc => ?_⟩
  · obtain ⟨Pi₀, ⟨cP, hcP, rfl⟩, hdist⟩ :=
      (isCompact_hull ΔW).exists_infDist_eq_dist (hull_nonempty ΔW hK) ΔWu
    exact ⟨cP, hcP, by rw [hdist, dist_eq_norm]⟩
  · rw [← dist_eq_norm]
    exact Metric.infDist_le_dist_of_mem ⟨c, hc, rfl⟩

/-- **Theorem `thm:hull`** (hull decomposition of the held-out risk), assembled.
Under Assumption `ass:lip` (`R` is `Λ`-Lipschitz on `𝒦_u`) a best mixture `c⋆` exists, the
excess risk of any routed `ĉ ∈ Δ^{K-1}` splits into a non-negative routing term and an
approximation term, and the approximation term obeys both bounds of Eq. thm:bounds. -/
theorem hull_decomposition {K : ℕ} (ΔW : Fin K → W) (ΔWu : W) (R : W → ℝ) (Λ : NNReal)
    (hLip : LipschitzOnWith Λ R (Ku ΔW ΔWu)) (hK : 0 < K) :
    ∃ cstar ∈ simplex K, IsMinOn (fun c => R (mix ΔW c)) (simplex K) cstar ∧
      (∀ chat ∈ simplex K,
        R (mix ΔW chat) - R ΔWu
          = (R (mix ΔW chat) - R (mix ΔW cstar)) + (R (mix ΔW cstar) - R ΔWu) ∧
        0 ≤ R (mix ΔW chat) - R (mix ΔW cstar)) ∧
      R (mix ΔW cstar) - R ΔWu ≤ Λ * Metric.infDist ΔWu (hull ΔW) ∧
      ∀ k, R (mix ΔW cstar) ≤ R (ΔW k) := by
  obtain ⟨cstar, hcstar, hmin⟩ :=
    exists_best_mixture ΔW R (hLip.continuousOn.mono (hull_subset_Ku ΔW ΔWu)) hK
  exact ⟨cstar, hcstar, hmin,
    fun chat hc => ⟨risk_decomposition ΔW ΔWu R cstar chat, routing_nonneg ΔW R hmin hc⟩,
    approximation_bound ΔW ΔWu R Λ hLip hK hmin, best_mixture_le_bank ΔW R hmin⟩

/-! ### Corollary `cor:growth` (growing the bank) -/

/-- `M_{K+1}((c, 0)) = M_K(c)`. -/
theorem mix_snoc_zero {K : ℕ} (ΔW : Fin K → W) (new : W) (c : Fin K → ℝ) :
    mix (Fin.snoc ΔW new) (Fin.snoc c 0) = mix ΔW c := by
  simp [mix, Fin.sum_univ_castSucc]

lemma snoc_zero_mem_simplex {K : ℕ} {c : Fin K → ℝ} (hc : c ∈ simplex K) :
    (Fin.snoc c 0 : Fin (K + 1) → ℝ) ∈ simplex (K + 1) := by
  refine ⟨fun i => ?_, ?_⟩
  · refine Fin.lastCases ?_ (fun j => ?_) i
    · simp
    · simpa using hc.1 j
  · simpa [Fin.sum_univ_castSucc] using hc.2

/-- `ℋ_K ⊆ ℋ_{K+1}`. -/
theorem hull_subset_hull_snoc {K : ℕ} (ΔW : Fin K → W) (new : W) :
    hull ΔW ⊆ hull (Fin.snoc ΔW new) := by
  rintro _ ⟨c, hc, rfl⟩
  exact ⟨Fin.snoc c 0, snoc_zero_mem_simplex hc, mix_snoc_zero ΔW new c⟩

/-- `dist(ΔW, ℋ_{K+1}) ≤ dist(ΔW, ℋ_K)` for every `ΔW ∈ 𝒲`. -/
theorem infDist_hull_snoc_le {K : ℕ} (ΔW : Fin K → W) (new : W) (hK : 0 < K) (x : W) :
    Metric.infDist x (hull (Fin.snoc ΔW new)) ≤ Metric.infDist x (hull ΔW) :=
  Metric.infDist_le_infDist_of_subset (hull_subset_hull_snoc ΔW new) (hull_nonempty ΔW hK)

/-- `ℛ⋆_v(K) := min_{c ∈ Δ^{K-1}} ℛ_v(M_K(c)) = inf ℛ_v(ℋ_K)`. -/
noncomputable def Rstar {K : ℕ} (R : W → ℝ) (ΔW : Fin K → W) : ℝ := sInf (R '' hull ΔW)

/-- `ℛ⋆_v(K)` is attained by any best mixture. -/
theorem Rstar_eq_of_isMinOn {K : ℕ} (R : W → ℝ) (ΔW : Fin K → W) {cstar : Fin K → ℝ}
    (hcstar : cstar ∈ simplex K)
    (hmin : IsMinOn (fun c => R (mix ΔW c)) (simplex K) cstar) :
    Rstar R ΔW = R (mix ΔW cstar) := by
  refine IsLeast.csInf_eq ⟨⟨_, ⟨cstar, hcstar, rfl⟩, rfl⟩, ?_⟩
  rintro _ ⟨_, ⟨c, hc, rfl⟩, rfl⟩
  exact isMinOn_iff.1 hmin _ hc

/-- `ℛ⋆_v(K+1) ≤ ℛ⋆_v(K)` for every version `v`; only continuity of `ℛ_v` is used. -/
theorem Rstar_snoc_le {K : ℕ} (R : W → ℝ) (ΔW : Fin K → W) (new : W) (hK : 0 < K)
    (hR : ContinuousOn R (hull (Fin.snoc ΔW new))) :
    Rstar R (Fin.snoc ΔW new) ≤ Rstar R ΔW :=
  csInf_le_csInf ((isCompact_hull _).bddBelow_image hR)
    ((hull_nonempty ΔW hK).image R) (Set.image_mono (hull_subset_hull_snoc ΔW new))

/-- The same statement through explicit minimisers: any best mixture over the grown bank is at
least as good as any best mixture over the old bank. -/
theorem best_mixture_snoc_le {K : ℕ} (R : W → ℝ) (ΔW : Fin K → W) (new : W)
    {c : Fin K → ℝ} (hc : c ∈ simplex K) {c' : Fin (K + 1) → ℝ}
    (hmin' : IsMinOn (fun c => R (mix (Fin.snoc ΔW new) c)) (simplex (K + 1)) c') :
    R (mix (Fin.snoc ΔW new) c') ≤ R (mix ΔW c) := by
  have := isMinOn_iff.1 hmin' _ (snoc_zero_mem_simplex hc)
  simpa [mix_snoc_zero] using this

/-! ### Corollary `cor:jensen` (convex twin) -/

/-- Jensen: if `ℛ_u` is convex on `𝒦_u`, every mixture is at least as good as the corresponding
average of transferred bank adapters. -/
theorem jensen_mixture {K : ℕ} (ΔW : Fin K → W) (ΔWu : W) (R : W → ℝ)
    (hconv : ConvexOn ℝ (Ku ΔW ΔWu) R) {c : Fin K → ℝ} (hc : c ∈ simplex K) :
    R (mix ΔW c) ≤ ∑ k, c k * R (ΔW k) := by
  have := hconv.map_sum_le (t := Finset.univ) (w := c) (p := ΔW)
    (fun k _ => hc.1 k) hc.2 (fun k _ => bank_mem_Ku ΔW ΔWu k)
  simpa [mix] using this

/-- The uniform coefficient vector `𝟏/K`. -/
noncomputable def uniform (K : ℕ) : Fin K → ℝ := fun _ => 1 / K

lemma uniform_mem_simplex {K : ℕ} (hK : 0 < K) : uniform K ∈ simplex K := by
  refine ⟨fun _ => by unfold uniform; positivity, ?_⟩
  have : (K : ℝ) ≠ 0 := by exact_mod_cast hK.ne'
  simp [uniform, Finset.sum_const, Finset.card_univ, Fintype.card_fin, this]

/-- The uniform mixture is at least as good as the mean of the transferred bank adapters. -/
theorem jensen_uniform {K : ℕ} (ΔW : Fin K → W) (ΔWu : W) (R : W → ℝ)
    (hconv : ConvexOn ℝ (Ku ΔW ΔWu) R) (hK : 0 < K) :
    R (mix ΔW (uniform K)) ≤ (1 / K) * ∑ k, R (ΔW k) := by
  have := jensen_mixture ΔW ΔWu R hconv (uniform_mem_simplex hK)
  simpa [uniform, Finset.mul_sum] using this

/-! ### Corollary `cor:scale`, mixture half -/

/-- `‖M(c)‖ ≤ max_k ‖ΔW^{(k)}‖`: any common bound on the bank bounds every mixture. -/
theorem norm_mix_le {K : ℕ} (ΔW : Fin K → W) {c : Fin K → ℝ} (hc : c ∈ simplex K)
    {m : ℝ} (hm : ∀ k, ‖ΔW k‖ ≤ m) : ‖mix ΔW c‖ ≤ m := by
  calc ‖∑ k, c k • ΔW k‖ ≤ ∑ k, ‖c k • ΔW k‖ := norm_sum_le _ _
    _ = ∑ k, c k * ‖ΔW k‖ := by simp [norm_smul, abs_of_nonneg (hc.1 _)]
    _ ≤ ∑ k, c k * m := Finset.sum_le_sum fun k _ => mul_le_mul_of_nonneg_left (hm k) (hc.1 k)
    _ = m := by rw [← Finset.sum_mul, hc.2, one_mul]

/-- The same with the maximum written out. -/
theorem norm_mix_le_sup {K : ℕ} (ΔW : Fin K → W) {c : Fin K → ℝ} (hc : c ∈ simplex K)
    (hK : 0 < K) :
    ‖mix ΔW c‖ ≤ Finset.univ.sup' ⟨⟨0, hK⟩, Finset.mem_univ _⟩ (fun k => ‖ΔW k‖) :=
  norm_mix_le ΔW hc fun k => Finset.le_sup' (fun k => ‖ΔW k‖) (Finset.mem_univ k)

/-- Per-site version: for any linear site projection `P : 𝒲 → ℝ^{d_out × d_in}`,
`‖P(M(c))‖ ≤ max_k ‖P(ΔW^{(k)})‖`. -/
theorem norm_site_mix_le {Ws : Type*} [NormedAddCommGroup Ws] [NormedSpace ℝ Ws]
    (P : W →ₗ[ℝ] Ws) {K : ℕ} (ΔW : Fin K → W) {c : Fin K → ℝ} (hc : c ∈ simplex K)
    {m : ℝ} (hm : ∀ k, ‖P (ΔW k)‖ ≤ m) : ‖P (mix ΔW c)‖ ≤ m := by
  have : P (mix ΔW c) = mix (fun k => P (ΔW k)) c := by simp [mix, map_sum, map_smul]
  rw [this]
  exact norm_mix_le _ hc hm

/-! ### Proposition `prop:wellposed` (b): coefficient identifiability -/

/-- If the bank updates are linearly independent, `c ↦ M(c)` is injective (on all of `ℝ^K`,
in particular on `Δ^{K-1}`). -/
theorem mix_injective {K : ℕ} (ΔW : Fin K → W) (h : LinearIndependent ℝ ΔW) :
    Function.Injective (mix ΔW) := by
  intro c c' hcc'
  have hsum : ∑ k, (c k - c' k) • ΔW k = 0 := by
    simp only [sub_smul, Finset.sum_sub_distrib]
    exact sub_eq_zero.2 hcc'
  have := Fintype.linearIndependent_iff.1 h _ hsum
  funext k
  exact sub_eq_zero.1 (this k)

theorem mix_injOn_simplex {K : ℕ} (ΔW : Fin K → W) (h : LinearIndependent ℝ ΔW) :
    Set.InjOn (mix ΔW) (simplex K) :=
  (mix_injective ΔW h).injOn

/-- Remark `rem:wellposed`: two generated adapters differ iff their coefficient vectors differ. -/
theorem mix_eq_iff {K : ℕ} (ΔW : Fin K → W) (h : LinearIndependent ℝ ΔW) (c c' : Fin K → ℝ) :
    mix ΔW c = mix ΔW c' ↔ c = c' :=
  (mix_injective ΔW h).eq_iff

end Risk

/-- The paper's adapter space `𝒲 = ∏_s ℝ^{d_out(s) × d_in(s)}` with
`‖ΔW‖ = (∑_s ‖ΔW_s‖_F²)^{1/2}` is the Euclidean space on all entries, hence an instance of the
hypotheses used above. -/
noncomputable def paper_space_instance {Ω : Type*} [Fintype Ω] (dout din : Ω → ℕ) :
    NormedSpace ℝ (EuclideanSpace ℝ (Σ s : Ω, Fin (dout s) × Fin (din s))) := inferInstance

end Web2LoRA
