# Verification log: WebMix theory (Appendices C and D of the paper)

Statements: `WebMix/Statements.lean`, one closed `Prop` per result, named after its LaTeX label.
Definitions they use: `WebMix/Defs.lean`. Proofs: `WebMix/Proofs/`. Numeric tests: `falsify/`.

Frozen on 2026-10-14 after the authors approved the statements (`lean_audit.py --freeze`):

```
150fa1cf81fc346a1f75a2da49eb822f507d81e7ef821d3f0dbad2ff7efb21cc  WebMix/Defs.lean
4d464fe4f7392b56b907a1715e04b485c4a315b5fc2c1353f3e2625e4580f440  WebMix/Statements.lean
```

| Label | Paper result | Priority |
|---|---|---|
| `ass_lip` | Assumption ass:lip, justification (C¹ risk ⇒ Lipschitz on `𝒦_u` with the largest gradient norm) | support |
| `thm_hull` | Theorem thm:hull (hull decomposition, both bounds) | HIGH |
| `rem_hull` | Remark rem:hull (distance to the hull is attained, is least squares, uniform mixture in the hull) | support |
| `cor_growth` | Corollary cor:growth (growing the bank) | MEDIUM |
| `cor_jensen` | Corollary cor:jensen (convex twin) | MEDIUM |
| `rem_jensen` | Remark rem:jensen and proof of cor:jensen (affine logits ⇒ convex risk) | support |
| `prop_twin` | Assumption ass:twin + Proposition prop:twin (direct generation is unnormalized routing) | MEDIUM |
| `cor_scale` | Corollary cor:scale (scale control; no bound in the twin) | MEDIUM |
| `prop_wellposed` | Proposition prop:wellposed (gauge invariance, identifiability, factor losses) | LOW |
| `rem_wellposed` | Remark rem:wellposed (Gram matrix, distances between mixtures) | LOW |
| `eq_exact` | Eqs. delta and exact of Section 3.2 (exact composition, served scale, cross terms) | support |

## Gate A: does each Lean statement say what the paper says?

### Blind back-translation

A fresh agent read only `Defs.lean` and `Statements.lean` (no `.tex`, no proofs, Mathlib source only to
look up definitions) and wrote every statement back as mathematics, with quantifier order,
hypotheses, conclusion and Lean conventions. It found no contradictory hypotheses and no statement
that is false or weaker than intended. Its notes and what was done:

- `rem_hull`: the docstring said "least-squares problem", which the term did not express (and which
  is false for a non-Euclidean norm). Added a second conjunct: in an inner-product space,
  `‖ΔW^{(u)} − M(c)‖² = ‖ΔW^{(u)}‖² − 2∑_k c_k⟨ΔW^{(u)}, ΔW^{(k)}⟩ + cᵀGc`.
- `prop_twin` (b): the coefficient map was chosen after the bank's dimension `m`. It is now its own
  conjunct with `∃ coeff` before `∀ m, ∀ B`, so one map serves every bank of every size.
- `[DecidableEq n]` was unused in `prop_twin` and `cor_scale`; removed.
- Confirmed: `1/K` is real division; `Λ` is coerced `NNReal → ℝ` in `thm_hull`; `infDist` is a
  genuine infimum because `K ≥ 1`; the matrix inverse is genuine under full column rank; `Tendsto`
  uses `atTop` and the entrywise (= Frobenius) topology; no statement takes Mathlib's sup norm of a
  vector (Euclidean norms are written out as `frob2`, `eucNorm`).
- Noted, no change needed: the decomposition identity and the routing sign in `thm_hull` are
  tautologies given `c⋆` (the paper calls the decomposition an identity); `prop_wellposed` (a2) is the
  definition of "depends on the factors only through `B_sA_s`"; `ass_lip`'s "gradient norm" is the
  operator norm of the derivative, which equals the Euclidean gradient norm on the paper's `𝒲`.

### Clause-by-clause comparison

Paper clauses refer to the released version of the paper. "Stronger" means the Lean statement
implies the paper's.

| Result | Paper clause | Lean clause | Match |
|---|---|---|---|
| all hull results | `𝒲 = ∏_s ℝ^{d_out×d_in}`, `‖ΔW‖ = (∑‖ΔW_s‖_F²)^{1/2}` | any real normed space `W` (`PaperSpace` is an instance, `Sanity.lean`) | stronger |
| all hull results | `ℛ_v` is the BC risk of Eq. thm:risk | any `R : W → ℝ` with the stated hypothesis | stronger |
| `ass_lip` | network `C¹` in its weights ⇒ Ass. ass:lip with `Λ_u` = largest gradient norm on `𝒦_u` | `ContDiff ℝ 1 R` ⇒ `∃ x₀ ∈ 𝒦_u`, `‖DR(x)‖ ≤ ‖DR(x₀)‖` on `𝒦_u` and `R` is `‖DR(x₀)‖`-Lipschitz on `𝒦_u` | yes |
| `thm_hull` | bank `K ≥ 1` (stated in the Setting) | `0 < K` | yes |
| `thm_hull` | Assumption ass:lip | `LipschitzOnWith Λ R (Ku ΔW ΔWu)`, `𝒦_u = conv(ℋ_K ∪ {ΔW^{(u)}})` literally | yes |
| `thm_hull` | a minimiser `c⋆` exists | `∃ cstar ∈ simplex K, IsMinOn …` | yes |
| `thm_hull` | decomposition for the router output `ĉ`, routing `≥ 0` | for every `ĉ ∈ Δ^{K-1}` | stronger |
| `thm_hull` | approx. `≤ Λ_u dist(ΔW^{(u)}, ℋ_K)` | `≤ Λ * Metric.infDist ΔWu (hull ΔW)` | yes |
| `thm_hull` | `ℛ_u(M(c⋆)) ≤ min_k ℛ_u(ΔW^{(k)})` | `∀ k, R (mix ΔW cstar) ≤ R (ΔW k)` | yes (`K ≥ 1`) |
| `rem_hull` | dist attained by some `c ∈ Δ^{K-1}` | `∃ cP ∈ simplex K, ‖ΔWu − M(cP)‖ = infDist`, and `infDist ≤ ‖ΔWu − M(c)‖` for all `c` | yes |
| `rem_hull` | "a `K`-variable least-squares problem" from the stored adapters | quadratic identity in `c` with the Gram matrix (inner-product space) | yes |
| `rem_hull` | `M(𝟏/K)` is an element of `ℋ_K` | `mix ΔW (uniform K) ∈ hull ΔW` | yes |
| `cor_growth` | bank of `K+1` extends bank of `K` by one adapter | `Fin.snoc ΔW new` (appended last) | yes |
| `cor_growth` | `M_{K+1}((c,0)) = M_K(c)` for `c ∈ Δ^{K-1}` | for every `c ∈ ℝ^K` | stronger |
| `cor_growth` | `ℋ_K ⊆ ℋ_{K+1}`; `dist(ΔW, ℋ_{K+1}) ≤ dist(ΔW, ℋ_K)` for every `ΔW` | same | yes |
| `cor_growth` | `ℛ⋆_v(K+1) ≤ ℛ⋆_v(K)`, minima of continuous `ℛ_v` | for every `R` continuous on `ℋ_{K+1}`: both minima exist, and every minimal value over `Δ^K` is `≤` every value over `Δ^{K-1}` | yes |
| `cor_jensen` | `ℛ_u` convex on `𝒦_u` | `ConvexOn ℝ (Ku ΔW ΔWu) R` | yes |
| `cor_jensen` | `ℛ_u(M(c)) ≤ ∑ c_k ℛ_u(ΔW^{(k)})`; uniform case | same; uniform case with `0 < K` and `1/(K:ℝ)` | yes |
| `rem_jensen` | logits an affine function of the whole update ⇒ `ℛ_u` convex on `𝒲` | `A i : W →ᵃ[ℝ] (ι → ℝ)` (jointly affine), weights `w_i ≥ 0` | yes |
| `rem_jensen` | `ℛ_u` weights each response token by `1/|𝒟_u|` | arbitrary `w_i ≥ 0` | stronger |
| `prop_twin` | `rank Ξ = K` | `FullColumnRank Ξ` (`y ↦ Ξy` injective) | yes |
| `prop_twin` | GD from `Γ = 0` on `∑_k ‖Γξ_k − vec ΔW^{(k)}‖²` | `gd` from `0`, `loss = frob2(ΓΞ − B)`, step `Γ − 2η(ΓΞ − B)Ξᵀ` | yes |
| `prop_twin` | step `< 1/λ_max(ΞΞᵀ)` | `∀ L ≥ 0` with `‖Ξᵀw‖² ≤ L‖w‖²`, `∀ η > 0` with `ηL < 1` | equivalent |
| `prop_twin` | converges to `Γ̂ = 𝓑Ξ⁺`, the unique min-norm interpolant | `Tendsto (gd …) atTop (𝓝 Γ̂)`; `Γ̂Ξ = 𝓑`; `‖Γ̂‖_F² ≤ ‖Γ‖_F²` for every interpolant, equality only at `Γ̂`; minimisers = interpolants | yes |
| `prop_twin` | Eq. prop:twin and (a) span | `output Γ̂ ξ = 𝓑 (Ξ⁺ξ)` and `∈ span{columns of 𝓑}` | yes |
| `prop_twin` | (b) `η(h)` depends on `(Ξ, ξ(h))` only | `∃ coeff, ∀ m B ξ, output (Γ̂ Ξ B) ξ = B (coeff ξ)` | yes |
| `prop_twin` | (c) for two training features | any `j₁ ≠ j₂`, any full-rank `Ξ` | yes |
| `cor_scale` | `‖M(c)‖ ≤ max_k ‖ΔW^{(k)}‖`, and per matrix | `∃ k, ‖M(c)‖ ≤ ‖ΔW k‖`; any linear `P` (block projections included) | yes / stronger |
| `cor_scale` | no such bound in the twin over feature vectors: `ξ(h) = tξ_k ↦ tΔW^{(k)}` | exact identity, and Euclidean norm unbounded in `t > 0` when `ΔW^{(k)} ≠ 0` | yes |
| `prop_wellposed` | (a) re-gauging member `k` by `Q_k` leaves `M(c)` unchanged | per member and per matrix `Q_{k,s}`, any scale `ρ` | stronger |
| `prop_wellposed` | (a) `ℒ_router` sees `F_φ(E_v)` only through `B_sA_s` | any `F` of the per-matrix updates is gauge invariant (definition of "only through") | yes (modelled) |
| `prop_wellposed` | (b) independent bank ⇒ `c ↦ M(c)` injective on `Δ^{K-1}` | `Set.InjOn (mix ΔW) (simplex K)` | yes |
| `prop_wellposed` | (c) factor ℓ₁ loss not constant over `(QA⋆, B⋆Q⁻¹)`, `A⋆ ≠ 0` | `Q = tI`: all targets give `B⋆A⋆`, and `∃ t > 0` with a different loss than at `t = 1` | yes |
| `rem_wellposed` | (b) checkable as nonsingularity of the Gram matrix | `det G ≠ 0 ↔ LinearIndependent` | yes |
| `rem_wellposed` | outputs differ iff coefficients differ; `‖M(c) − M(c')‖² = (c−c')ᵀG(c−c')` | same, for all `c, c' ∈ ℝ^K` | yes |
| `eq_exact` | `B_vA_v = ∑ c_k B^{(k)}A^{(k)}`; `α_out/r_out = α/r` gives the served update; averaging factors gives cross terms | the three identities (`r = card` of the rank index type) | yes |

### Lean-convention traps checked

- Division: `uniform K = 1/(K:ℝ)` (`Sanity.lean` checks `uniform 2 0 = 1/2`); `(Kα)/(K·card r)` in
  `eq_exact` is `0 = 0` when `card r = 0` (harmless); `t⁻¹` only for `t > 0`.
- Junk values: `Metric.infDist` of `∅` is `0`, excluded by `0 < K`; the matrix inverse of a singular
  matrix is `0`, excluded by full column rank; `fderiv` is only used for a `C¹` function; `sInf` is not
  used (minima are stated with `IsMinOn`); `Real.log` gets a positive argument (`[Nonempty ι]`).
- Norms: hull results use the norm of `W`; the paper's `𝒲` is `PaperSpace`, whose norm is the
  Euclidean norm on all entries (`Sanity.lean`); twin norms are `frob2` and `eucNorm`, not Mathlib's
  sup norm (`Sanity.lean` checks `eucNorm ![3,4] = 5` and a `loss` value).
- Coercions: `ℕ → ℝ` in `uniform`, `cor_jensen`, `eq_exact`; `NNReal → ℝ` for `Λ` in `thm_hull`
  (visible as `↑Λ` in the elaborated goal).
- Filters: one `Tendsto`, `atTop` on `ℕ` to `𝓝 Γ̂` in the entrywise topology.
- `autoImplicit` is now off in `lakefile.toml`, so a typo in a statement is an error, not a new
  universally quantified variable.

### Definitions (`Defs.lean`) against the paper

| Lean | Paper | Agree |
|---|---|---|
| `simplex K = stdSimplex ℝ (Fin K)` | `Δ^{K-1}` | yes |
| `mix ΔW c = ∑ c_k • ΔW k` | `M(c)`, Eq. thm:mix | yes |
| `hull ΔW = mix ΔW '' simplex K` | `ℋ_K = M(Δ^{K-1})` | yes |
| `Ku ΔW ΔWu = convexHull ℝ (hull ΔW ∪ {ΔWu})` | `𝒦_u = conv(ℋ_K ∪ {ΔW^{(u)}})` | yes |
| `uniform K = fun _ => 1/(K:ℝ)` | `𝟏/K` | yes |
| `PaperSpace Ω dout din = EuclideanSpace ℝ (Σ s, Fin dout × Fin din)` | `𝒲` with its norm | yes |
| `lse`, `nll y ℓ = lse ℓ − ℓ_y` | `−log softmax(ℓ)_y` | yes |
| `bcRisk A y w ΔW = ∑ w_i nll(y_i, A_i ΔW)` | `ℛ_u` with affine logits | yes |
| `Twin.finner`, `frob2`, `eucNorm` | Frobenius inner product, `‖·‖_F²`, Euclidean norm | yes |
| `FullColumnRank`, `gram`, `pinv`, `Γhat` | `rank Ξ = K`, `ΞᵀΞ`, `Ξ⁺ = (ΞᵀΞ)⁻¹Ξᵀ`, `Γ̂ = 𝓑Ξ⁺` | yes |
| `loss`, `gdStep`, `gd` | `L(Γ)`, one GD step with `∇L = 2(ΓΞ − 𝓑)Ξᵀ`, GD from `0` | yes |
| `beta Ξ ξ = Ξ⁺ξ`, `output Γ ξ = Γξ` | `η(h)`, `vec ΔW_Γ(h)` | yes |
| `Gauge.concatA`, `concatB` | Eq. delta | yes |
| `Gauge.l1` | entrywise `‖·‖₁` | yes |

### Non-vacuity, triviality, teeth, numeric falsification

- **Witnesses** (`Sanity.lean`, compiles): for every statement, an instance satisfying all hypotheses
  that avoids the degenerate case (target outside the hull with `dist ≥ 1`; a strictly growing hull;
  the strictly convex `x²`; logits that depend on the update; identity features with `L = 1`,
  `η = 1/2`; a nonzero bank column; an invertible gauge; an independent bank; `A⋆ ≠ 0`).
- **Not trivially true**: `aesop` and `simp` were run on all eleven statements (14 attempts); none
  closed a goal.
- **Teeth** (HIGH result `thm_hull`): with `R` Lipschitz only on the hull, not on `𝒦_u`, the first
  bound fails (`Sanity.lean`: bank `{1}`, target `0`, `R = 10x`: `10 > 1`). Also numeric teeth: a GD
  step above `1/λ_max` diverges (`falsify/prop_twin.py`); logits affine in each matrix separately can
  give a nonconvex risk (`falsify/rem_jensen.py`, and in Lean `Sanity.separate_affinity_not_convex`).
- **Numeric falsification** (`python3 falsify/run_all.py`, independent of Lean): see the counts below.

| Test | Checks | Violations |
|---|---|---|
| `falsify/ass_lip.py` | 60000 | 0 |
| `falsify/thm_hull.py` | 15000 | 0 |
| `falsify/rem_hull.py` | 40500 | 0 |
| `falsify/cor_growth.py` | 2700 | 0 |
| `falsify/cor_jensen.py` | 18205 | 0 |
| `falsify/rem_jensen.py` | 9001 | 0 |
| `falsify/prop_twin.py` | 7398 | 0 |
| `falsify/cor_scale.py` | 61020 | 0 |
| `falsify/prop_wellposed.py` | 3762 | 0 |
| `falsify/rem_wellposed.py` | 3000 | 0 |
| `falsify/eq_exact.py` | 4501 | 0 |

`cor_jensen.py` also prints an example showing that the risk inequality of the corollary does not
carry over to task success under a convex risk-to-success link (bank risks 0 and 10, uniform risk 5:
`e^{-5} < (1 + e^{-10})/2`), and `rem_jensen.py` prints a nonconvex risk whose logits are affine in
each of two matrices separately.

### Human checkpoint

The authors reviewed the clause table, the back-translation summary and the convention traps
above and approved all eleven statements on 2026-10-14. The package was renamed from `Web2LoRA` to
`WebMix` and the two files were frozen; they have not changed since, and the audit's freeze check
verifies this.

## Gate B: is the Lean proof the paper's proof?

- **Map.** `README.md` maps every numbered result to its theorem and every step of the paper's
  proofs (Appendix D) to its lemma; `Audit.lean` prints the axioms of all of them (84 lines). Two
  remark-level claims have their own lemmas: rem:growth (`realized_rise_le_routing_rise`) and
  rem:jensen (`Sanity.separate_affinity_not_convex`).
- **Side conditions.** `K ≥ 1` is stated in the paper's Setting, and the paper states the conditions
  of cor:scale (a nonzero bank) and prop:wellposed (a) (the policy uses `W_{θ,s} + (α/r)B_sA_s`). The
  remaining ones are listed in `README.md` as harmless (nonempty vocabulary, `j₁ ≠ j₂`, `L ≥ 0`,
  continuity only on `ℋ_{K+1}`).
- **Twin honesty.** The paper states the twin results for the twin (Assumption ass:twin) and about
  conditionings with given features, as Lean does; the hull results are stated for the paper's
  space, which Lean covers by proving them for every normed space.
- **Proof alignment.** The paper's proofs and the Lean proofs follow the same decomposition: for
  thm:hull, existence (continuity on `ℋ_K` from the Lipschitz assumption), add and subtract, a nearest
  point of the hull then Lipschitz, the vertices; for cor:growth, the embedding `c ↦ (c, 0)`; for
  cor:jensen, Jensen then Hölder, log-sum-exp, the NLL, affine composition and a nonnegative sum; for
  prop:twin, the minimum norm by orthogonality and the four gradient-descent steps (rows in the span,
  one-step descent, summability, `X_t = (X_tΞ)Ξ⁺`), with `Ξ⁺Ξ = I` for (c); for cor:scale, the
  triangle inequality; for prop:wellposed, `(BQ⁻¹)(QA) = BA`, linear independence and the reverse
  triangle inequality.
- **Claim scan.** "Lean", "formal", "machine-checked" and "verified" appear in the paper's
  Reproducibility Statement and LLM-usage statement; both rest on this folder, and the
  Reproducibility Statement cites `README.md` for the numbered results.
- **Independent audit.** A fresh agent compared the paper's appendices with the Lean files. It found
  no unsound Lean and no unresolved reference; the differences in wording it listed were resolved in
  the paper or in Lean before release (the two remark-level lemmas above came from this check).

## Gate C: can a reader follow the proof?

- **Structure.** Every proof file opens with a module docstring that names the paper results and
  states the strategy. Each paper step is one named lemma (step table in `README.md`); chains of
  (in)equalities are `calc` blocks (`approximation_bound`, `frob2_step_le`, `norm_mix_le`,
  `mul_step_mul_proj`, `l1_sub_smul_ge`); intermediate facts are named `have`s; the longest tactic
  block (`tendsto_err_mul`) is about 30 lines after the convergence proof was split into steps
  (ii)–(iii) and (iv). Every definition, statement, lemma and theorem has a docstring citing the
  paper.
- **Automation.** `simp`, `ring`, `linarith`, `nlinarith`, `positivity`, `field_simp` and `gcongr` do
  routine algebra; where one carries a step (Hölder's inequality reshaped into the convexity bound,
  the division in `frob2_mul_transpose_le`, the `rfl` that unfolds a definition), a comment says
  what it proves. No heartbeat limit is raised.
- **Blind explanation.** A fresh agent given only `Defs.lean` and the proof files (not the paper,
  not `Statements.lean`) explained all eight groups of results. Its explanations follow the paper's
  proofs step by step: for thm:hull, existence by compactness with continuity from the Lipschitz
  assumption, add and subtract, projection then Lipschitz, vertices; for prop:twin, the orthogonal
  decomposition `Γ = Γ̂ + N` for the minimum norm and the four steps (i)–(iv) for gradient descent;
  for rem:jensen, Hölder, logarithm, linear subtraction, affine composition, nonnegative sum.
  It found these proofs hard to follow, all fixed: `Twin.S_mul_proj`, `Twin.gram_isUnit_det`,
  `sumExp_combo_le` and the `rfl` steps of `norm_mix_sq` and `Twin.loss_expand` lacked comments
  (comments added); `hull_eq_convexHull` (comment added); the near-identical names `norm_sub_mix_sq`
  and `norm_mix_sub_sq` (renamed `norm_target_sub_mix_sq` and `norm_mix_sub_mix_sq`) and the primed
  `convex_hull'` (renamed `hull_convex`); `prop_twin` (b) repeated a `simp` call (now cites
  `Twin.output_Γhat`); `L` names the eigenvalue bound while the loss is `loss` (explained in the
  module docstring of `Twin.lean`). It also noted that `Twin.loss_expand` is what certifies the
  gradient formula built into `Twin.gdStep`; that link is stated in the step table.
- **Paper side.** Appendix C gives each result's proof sketch in words, Appendix D gives complete
  prose proofs that can be checked without Lean, and the step table in `README.md` maps each paper
  step to its lemma (the authors chose a one-sentence pointer in the paper rather than a Lean
  appendix).

## Gate D: what is trusted?

Run: `python3 ~/.claude/skills/theory/scripts/lean_audit.py lean` (and `--fresh`), 2026-10-14.

- **Forbidden constructs**: none anywhere in the project (`sorry`, `admit`, `axiom`, native
  evaluation, `implemented_by`, `@[extern]`, `unsafe`, `debug.skipKernelTC`, trust axioms, `#exit`,
  metaprogramming). **Flagged constructs**: none (no custom `instance`/`notation`, `opaque`,
  `decide`, raised heartbeats or `import Lean`); the one `decide` in an early draft of
  `Sanity.lean` was replaced by `Fin.zero_ne_one'`.
- **Freeze**: `Defs.lean` and `Statements.lean` match the approved hashes.
- **Fresh compile**: all 10 modules recompiled from source into a temporary directory, so no stale
  build artifact counts.
- **Statement binding**: each `WebMix.<label>` proves `WebMix.Statements.<label>` (11 of 11).
- **Axioms**: every result, and every one of the 84 `#print axioms` lines of `Audit.lean`, depends
  only on `propext`, `Classical.choice`, `Quot.sound`.
- **Kernel replay** of the package's own declarations: pass (67 s, non-fresh run).
- **Fresh replay** (every constant the package sees, Mathlib included, into an empty environment):
  pass (627 s, 2026-10-14). The final run was `--fresh`; every result reads `VERIFIED` and the
  verdict is `PASS`.
- **Independent kernel**: none installed (`lean4lean`, `nanoda`, Comparator are not on `PATH`), so
  trust rests on a single kernel. Lean v4.35.0 ships `lean4lean` and `nanoda` with the toolchain and
  `lake check --paranoid` runs them; upgrading the toolchain and Mathlib to v4.35.0 is the way to add
  a second kernel. Comparator needs Linux.
- **Version hygiene**: toolchain `leanprover/lean4:v4.34.0-rc2` (commit `6a10ac8c`), Mathlib
  `85e3a25e006c35636f0e53b0e9296caca2685bc0`. Releases after it: v4.34.0, v4.34.1, v4.35.0. The kernel
  soundness fixes listed for v4.34.0 (#14582, #14806, #14807, #14808, #14833, #14838, #14849) were
  backported into v4.34.0-rc2 (checked against the tag's commit history). Not in rc2: #15241, #15288,
  #15289 (v4.34.1, v4.35.0), which stop reference counts from overflowing when one subterm is shared
  more than about 2³¹ times, and #14953 (v4.35.0), which removes the native-reduction axioms that
  this project never uses and the audit forbids. The toolchain was not upgraded: the overflow fixes
  concern terms with billions of references to one subterm, far beyond this development. This is
  recorded as a residual risk.
- **Redundancy outside Lean**: the prose proofs passed Gates B and C, and the numeric falsification
  tests passed (225,087 checks, 0 violations), independently of Lean.
- **Suspicion triggers**: the theorems in `Results.lean` are short because they only assemble the
  step lemmas; no proof relies on kernel computation (`decide`, `rfl` on large terms) or an unusual
  `set_option`. One hypothesis is unused by design: full column rank in `prop_twin` (b), which holds
  for every `Ξ` because it is the associativity `(𝓑Ξ⁺)ξ = 𝓑(Ξ⁺ξ)`; the paper states (b) under
  Assumption ass:twin, which is harmless.

**Trust statement.** Trusted: Lean kernel `v4.34.0-rc2`, Mathlib `85e3a25e`, axioms `propext`,
`Classical.choice`, `Quot.sound`. Kernel replay: pass (`--fresh`: yes). Independent kernel:
none. Numeric falsification: pass. Statements human-approved: yes.
