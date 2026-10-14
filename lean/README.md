# WebMix theory, machine-checked in Lean 4

Lean 4 + Mathlib proofs of the theoretical results of the WebMix paper ("Don't Retrain, Remix:
Adaptation of Web Agents using Mixture Hypernetworks", Appendix C "Theoretical Analysis" and
Appendix D "Proofs", plus the composition identity of Section 3.2). Every result is stated once,
as a closed proposition in `WebMix/Statements.lean`, and proved by a theorem whose type *is* that
proposition, so the Lean kernel checks that what is proved is exactly what is stated. No proof uses
`sorry`, and every result depends only on Lean's three standard axioms.

## Build and check

```sh
lake exe cache get          # fetch the prebuilt Mathlib once
lake build                  # build every module
lake env lean Audit.lean    # print the axioms of every result and every mapped lemma
python3 falsify/run_all.py  # numeric falsification tests (numpy, BSD-3-Clause), independent of Lean
```

Toolchain: Lean `v4.34.0-rc2` (`lean-toolchain`), Mathlib at the matching tag
(commit `85e3a25e006c35636f0e53b0e9296caca2685bc0`, `lake-manifest.json`).

## Layout

| File | Contents |
|---|---|
| `WebMix/Defs.lean` | Every object the statements use (mixtures, the hull, `𝒦_u`, the BC risk with affine logits, the frozen-trunk twin, LoRA factor concatenation), each citing the paper's equation |
| `WebMix/Statements.lean` | One closed `def <label> : Prop` per paper result, named after its LaTeX label |
| `WebMix/Sanity.lean` | For every statement, an instance satisfying all hypotheses (non-vacuity), the teeth example for Theorem thm:hull, and checks of Lean conventions |
| `WebMix/Proofs/*.lean` | The proofs: one lemma per step of the paper's proof |
| `WebMix/Proofs/Results.lean` | `theorem WebMix.<label> : WebMix.Statements.<label>` for every result |
| `Audit.lean` | `#print axioms` for every result and every lemma in the map below |
| `falsify/` | Numeric falsification tests with the paper's exact constants |
| `VERIFICATION.md` | The verification log: statement fidelity, proof alignment, readability, trust |

`Defs.lean` and `Statements.lean` were approved by the authors and then frozen;
`.statements.sha256` records their hashes.

## Paper ↔ Lean map

| Paper result | Lean theorem | Status |
|---|---|---|
| Assumption ass:lip, justification (`C¹` risk ⇒ Lipschitz on `𝒦_u` with the largest gradient norm) | `WebMix.ass_lip` | verified |
| Theorem thm:hull (hull decomposition of the held-out risk) | `WebMix.thm_hull` | verified |
| Remark rem:hull (distance attained; least squares; uniform mixture in the hull) | `WebMix.rem_hull` | verified |
| Corollary cor:growth (growing the bank) | `WebMix.cor_growth` | verified |
| Corollary cor:jensen (convex twin) | `WebMix.cor_jensen` | verified |
| Remark rem:jensen and proof of cor:jensen (affine logits ⇒ convex risk) | `WebMix.rem_jensen` | verified |
| Assumption ass:twin + Proposition prop:twin (direct generation is unnormalised routing) | `WebMix.prop_twin` | verified |
| Corollary cor:scale (scale control; no bound in the twin over feature vectors) | `WebMix.cor_scale` | verified |
| Proposition prop:wellposed (gauge invariance, identifiability, factor losses) | `WebMix.prop_wellposed` | verified |
| Remark rem:wellposed (Gram matrix; distances between mixtures) | `WebMix.rem_wellposed` | verified |
| Eqs. delta and exact (exact composition, served scale, cross terms) | `WebMix.eq_exact` | verified |

Two remark-level claims are backed by lemmas outside the frozen statements (axioms checked in
`Audit.lean`): Remark rem:growth, "any rise of the realised risk after the bank grows comes from
the routing term" (`realized_rise_le_routing_rise`), and Remark rem:jensen, "affinity in each
adapted matrix separately is not enough" (`WebMix.Sanity.separate_affinity_not_convex`, an explicit
nonconvex risk with logits bilinear in two parameters).

### Proof steps

Each step of the paper's proof (Appendix D) is one Lean lemma (namespace `WebMix`).

| Paper step | Lean lemma | File |
|---|---|---|
| `ℋ_K = M(Δ^{K-1}) = conv{ΔW^{(k)}}`, compact, convex (Eq. thm:mix) | `hull_eq_convexHull`, `isCompact_hull`, `hull_convex` | `Proofs/Hull.lean` |
| `𝒦_u` convex and compact | `convex_Ku`, `isCompact_Ku` | `Proofs/Hull.lean` |
| Ass. ass:lip: max of `‖Dℛ‖` on compact `𝒦_u`, then the mean value inequality on convex `𝒦_u` | `lipschitzOnWith_of_contDiff`, `assumption_lip_of_contDiff` | `Proofs/Lipschitz.lean` |
| thm:hull, existence of `c⋆` (compactness; continuity from Ass. ass:lip) | `exists_best_mixture` | `Proofs/Hull.lean` |
| thm:hull, Eq. thm:decomp (add and subtract), routing `≥ 0` | `risk_decomposition`, `routing_nonneg` | `Proofs/Hull.lean` |
| thm:hull, first bound (projection `Π = M(c_Π)`, then Lipschitz) | `approximation_bound` | `Proofs/Hull.lean` |
| thm:hull, second bound (vertices `M(e_k) = ΔW^{(k)}`) | `best_mixture_le_bank`, `mix_vertex` | `Proofs/Hull.lean` |
| thm:hull assembled | `hull_decomposition` | `Proofs/Hull.lean` |
| rem:hull, distance attained / least squares / `M(𝟏/K) ∈ ℋ_K` | `infDist_hull_attained`, `norm_target_sub_mix_sq`, `mix_uniform_mem_hull` | `Proofs/Hull.lean` |
| cor:growth, `(c, 0) ∈ Δ^K`, `M_{K+1}((c,0)) = M_K(c)`, `ℋ_K ⊆ ℋ_{K+1}` | `snoc_zero_mem_simplex`, `mix_snoc_zero`, `hull_subset_hull_snoc` | `Proofs/Hull.lean` |
| cor:growth, minimum over a superset (distance; best risk) | `infDist_hull_snoc_le`, `best_mixture_snoc_le` | `Proofs/Hull.lean` |
| rem:growth, the rise of the realised risk is at most the rise of the routing term | `realized_rise_le_routing_rise` | `Proofs/Hull.lean` |
| cor:jensen, Jensen's inequality and the uniform case | `jensen_mixture`, `jensen_uniform` | `Proofs/Hull.lean` |
| proof of cor:jensen: log-sum-exp convex (Hölder), NLL convex, affine composition, nonnegative combination | `sumExp_combo_le`, `convexOn_lse`, `convexOn_nll`, `convexOn_bcRisk` | `Proofs/Softmax.lean` |
| rem:jensen, separate affinity is not enough | `Sanity.separate_affinity_not_convex` | `Sanity.lean` |
| prop:twin: `ΞᵀΞ` invertible, `Ξ⁺Ξ = I`, `Γ̂Ξ = 𝓑` | `Twin.gram_isUnit_det`, `Twin.pinv_mul`, `Twin.Γhat_mul` | `Proofs/Twin.lean` |
| prop:twin: minimisers = interpolants; `∇L(Γ) = 2(ΓΞ − 𝓑)Ξᵀ` | `Twin.isMinOn_loss_iff`, `Twin.loss_expand` | `Proofs/Twin.lean` |
| prop:twin: minimum norm (`⟨Γ̂, N⟩_F = 0`) | `Twin.Γhat_min_norm` | `Proofs/Twin.lean` |
| prop:twin: error recursion `X_{t+1} = X_t(I − 2γΞΞᵀ)` | `Twin.gdStep_sub_Γhat` | `Proofs/Twin.lean` |
| prop:twin step (i): rows of `X_t` stay in `span{ξ_k}` | `Twin.err_mul_proj` | `Proofs/Twin.lean` |
| prop:twin step (ii): one-step descent, with `‖ZΞᵀ‖_F² ≤ λ_max‖Z‖_F²` | `Twin.frob2_step_le`, `Twin.frob2_mul_transpose_le` | `Proofs/Twin.lean` |
| prop:twin step (iii): summability, `X_tΞ → 0` | `Twin.tendsto_err_mul` | `Proofs/Twin.lean` |
| prop:twin step (iv): `X_t = (X_tΞ)Ξ⁺ → 0` | `Twin.err_eq`, `Twin.gd_tendsto` | `Proofs/Twin.lean` |
| Eq. prop:twin and (a): `Γ̂ξ = 𝓑(Ξ⁺ξ)` in the span of the bank | `Twin.output_Γhat`, `Twin.output_mem_span` | `Proofs/Twin.lean` |
| prop:twin (c): `η(ξ_k) = e_k`, `η(ξ_1 ± ξ_2) = e_1 ± e_2 ∉ Δ^{K-1}`, outputs | `Twin.beta_col`, `Twin.beta_add_cols`, `Twin.beta_sub_cols`, `Twin.beta_add_cols_not_mem_simplex`, `Twin.beta_sub_cols_not_mem_simplex`, `Twin.output_add_cols`, `Twin.output_sub_cols`, `Twin.add_cols_eq_two_smul_mean` | `Proofs/Twin.lean` |
| cor:scale, triangle inequality (whole update, per matrix) | `norm_mix_le`, `exists_norm_mix_le`, `exists_norm_proj_mix_le` | `Proofs/Hull.lean` |
| cor:scale, twin: `η(tξ_k) = te_k`, output `tΔW^{(k)}`, unbounded | `Twin.beta_smul_col`, `Twin.output_smul_col`, `Twin.output_unbounded_euc` | `Proofs/Twin.lean` |
| Eq. bg:gauge, `(BQ⁻¹)(QA) = BA` | `Gauge.gauge_invariance` | `Proofs/Gauge.lean` |
| Eqs. delta/exact, concatenation and served scale `α_out/r_out = α/r`; cross terms | `Gauge.concat_mul`, `Gauge.served_update`, `Gauge.avg_factors_cross_terms` | `Proofs/Gauge.lean` |
| prop:wellposed (a) | `Gauge.served_update_gauge_invariant`, `Gauge.objective_gauge_invariant` | `Proofs/Gauge.lean` |
| prop:wellposed (b) and rem:wellposed | `mix_injective`, `mix_injOn_simplex`, `mix_eq_iff`, `norm_mix_sq`, `norm_mix_sub_mix_sq` | `Proofs/Hull.lean` |
| prop:wellposed (c): `Q = tI`, reverse triangle inequality | `Gauge.l1_sub_smul_ge`, `Gauge.factor_loss_not_gauge_invariant`, `Gauge.factor_loss_not_constant` | `Proofs/Gauge.lean` |

## Conventions for reading the statements

* **Spaces.** The hull results hold for every real normed space `W`. The paper's adapter space
  `𝒲 = ∏_s ℝ^{d_out(s) × d_in(s)}` with `‖ΔW‖ = (∑_s ‖ΔW_s‖_F²)^{1/2}` is `PaperSpace`, a Euclidean
  space, so it is an instance. Remarks rem:hull (least squares) and rem:wellposed need an
  inner-product space, which `PaperSpace` is.
* **Risk.** `ℛ` is any function `W → ℝ` satisfying the stated hypothesis (Lipschitz on `𝒦_u`,
  continuity, or convexity). For Remark rem:jensen the risk is `∑_i w_i nll(y_i, A_i ΔW)` with
  `w_i ≥ 0`, which covers both the per-sequence average of Eq. thm:risk and the token average of
  Eq. obj.
* **Simplex and distance.** `Δ^{K-1}` is `stdSimplex ℝ (Fin K)`; `dist(·, ℋ)` is `Metric.infDist`;
  "`≤ min_k`" and "`≤ max_k`" are stated as "`≤` for every `k`" and "`≤` for some `k`".
* **Twin.** Features are the columns of `Ξ : Matrix n k ℝ`, bank updates the columns of
  `B : Matrix m k ℝ`; `rank Ξ = K` is injectivity of `y ↦ Ξy`; `λ_max(ΞΞᵀ) ≤ L` is the Rayleigh
  bound `‖Ξᵀw‖² ≤ L‖w‖²`, and the step condition `γ < 1/λ_max` is `ηL < 1` (Lean calls the step `η`
  and the coefficients `beta`). Norms are written out (`frob2`, `eucNorm`) because Mathlib's default
  norm on `m → ℝ` is the sup norm. Convergence is in the entrywise topology, which equals
  Frobenius convergence.
* **Gradient step.** `Twin.gdStep` implements `Γ − 2η(ΓΞ − 𝓑)Ξᵀ`; `Twin.loss_expand` proves the exact
  expansion `L(Γ + H) = L(Γ) + ⟨2(ΓΞ − 𝓑)Ξᵀ, H⟩_F + ‖HΞ‖_F²`, which certifies that `2(ΓΞ − 𝓑)Ξᵀ` is
  the gradient of `L`. The frozen statement takes `gdStep` as the definition of one gradient step.
* **Lean junk values** are excluded by the hypotheses: `infDist` of `∅` is `0` (so `K ≥ 1`), the
  inverse of a singular matrix is `0` (so full column rank), `1/K` is real division.

### Names that differ from the paper

| Lean | Paper | Note |
|---|---|---|
| `η` (step size) | `γ` | the paper's `η(h)` is the coefficient map |
| `Twin.beta Ξ ξ` | `η(h) = Ξ⁺ξ(h)` | in the paper `β` is the weight of the KL anchor |
| `L` (in the twin lemmas) | `λ_max(ΞΞᵀ)` bound | in the paper `L(Γ)` is the training loss, `Twin.loss` here |
| `ρ` (in `prop_wellposed`) | `α/r` | in Appendix B `ρ` is the role of a factor |
| `s ∈ Ω` | `ℓ ∈ Ω` in Section 3 | Appendix C also writes `s` |
| `E t` (local name in `Twin.lean`) | `X_t = Γ_t − Γ̂` | the gradient-descent error |

## Side conditions

Conditions the Lean statements carry beyond the paper's wording, each harmless:

* `K ≥ 1` (a nonempty bank) for Theorem thm:hull, Remark rem:hull and Corollary cor:growth; the paper
  now states it in the Setting of Appendix C.
* Corollary cor:growth needs `ℛ_v` continuous only on `ℋ_{K+1}`; the paper's `ℛ_v` is continuous on
  all of `𝒲`.
* Remark rem:jensen assumes a nonempty vocabulary; the paper's vocabulary is nonempty.
* Corollary cor:scale (twin half): unboundedness needs `ΔW^{(k)} ≠ 0`; the paper now says "for a
  nonzero bank".
* Proposition prop:twin (c) takes two distinct training features `j₁ ≠ j₂`, and the gradient-descent
  part takes `L ≥ 0`; both are implicit in the paper (two features; a bound on `λ_max ≥ 0`).
* Proposition prop:wellposed (a) models "`ℒ_router` depends on the generator only through the updates
  `B_sA_s`" as "is a function `F` of the per-matrix updates", which is how the policy uses an adapter.

## What is not formalised

* The bridge from behaviour-cloning risk to task success (Remarks rem:hull, rem:growth,
  rem:jensen), which the paper states as an assumption.
* That the base network is `C¹` in its weights: it enters `WebMix.ass_lip` as the hypothesis
  `ContDiff ℝ 1 R`.
* That the frozen-trunk twin idealises T2L (Remark rem:twin), and the examples in remarks (which
  matrices make the logits affine).

## Trust

> Trusted: Lean kernel `v4.34.0-rc2`, Mathlib `85e3a25e006c35636f0e53b0e9296caca2685bc0`, axioms
> `propext`, `Classical.choice`, `Quot.sound`. Kernel replay: pass (`--fresh`: yes, every constant
> including Mathlib). Independent kernel: none (single-kernel trust). Numeric falsification: pass
> (225,087 checks, 0 violations). Statements human-approved: yes.

The audit (`lean_audit.py` of the `/theory` workflow) recompiles every module from source,
checks that each `WebMix.<label>` proves `WebMix.Statements.<label>`, scans the sources for `sorry`,
`axiom`, native evaluation and metaprogramming (none), prints the axioms of every result and every
mapped lemma, and replays every declaration through the kernel. Residual risks: a single kernel
(Lean v4.35.0 ships the independent checkers `lean4lean` and `nanoda`); toolchain `v4.34.0-rc2` lacks
the later reference-count-overflow fixes of v4.34.1 (#15241, #15288, #15289), which concern
subterms shared more than about 2³¹ times, far beyond this development. See `VERIFICATION.md`.
