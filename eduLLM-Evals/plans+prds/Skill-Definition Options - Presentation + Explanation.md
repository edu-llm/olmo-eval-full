# Skill-Definition Options: "Presentation" and "Explanation"

**Status:** Directional investigation to inform (not execute) an IRT re-parameterization.
**Author:** analysis run `scripts/investigate_skill_options.py`
**Date:** 2026-07-28
**Do NOT** treat any number here as a decision. See "Data caveats" first.

---

## 0. Executive summary

We evaluated two proposed changes to the tutor-skill latent structure using the only
graded data available today — the **1/4 judge run on the OLD (pre-curation) criteria**
(`staging/response_matrix.csv`, 82 model rows / 6462 criteria, ~25% filled, **~25–28
usable models**). The curated one-per-scenario "presentation" criterion was **not graded
in this run**, so we built a **presentation proxy** from the OLD all-zero-Q style/format
orphans (organization/markdown/LaTeX + persona/second-person items; affect excluded),
which curation later consolidated into "presentation".

| Hypothesis | Recommendation |
|---|---|
| **H1** — make "presentation" its own latent dimension, possibly replacing "diagnosis" | **ADD as a candidate 4th dimension (defer to confirmatory); do NOT use it to replace diagnosis.** Presentation looks like a real, measurable, separable trait, but it is a *surface/style* construct — a statistically cleaner factor than diagnosis, yet a poor pedagogical substitute for it. |
| **H2** — reframe "content" as "explanation quality" to make "diagnosis" standalone | **DEFER — not testable from current data.** Diagnosis is currently ~fused with content (latent r≈0.90; per-model r≈0.97). Whether an "explanation"-only redefinition would separate diagnosis is a *relabeling* question that cannot be answered without re-deriving Q-labels and regrading under the new definitions. |

Both recommendations are conditional on the confirmatory analyses pre-registered in §6.

---

## 1. Data & method caveats (read before any number)

- **Single, biased, tiny sample.** All empirical numbers come from one judge run
  (Qwen strict judge, canonical r1) over the OLD criteria. After dropping empty
  rows/cols and zero-variance items, **only ~25–28 models are usable** — a biased
  small-model subsample on a ~1/4-filled matrix.
- **Column keying.** The matrix columns are OLD `criterion_id`s, so we map them with the
  **OLD** Q-matrix `data/rubrics_qmatrix_final.jsonl` — **not** the curated file.
- **Presentation was never graded as curated.** The normalized one-per-scenario
  presentation criterion does not exist in this matrix. We use the **granular OLD orphan
  items** curation later merged into "presentation" as the empirical proxy
  (428 items: 299 organization/format + 129 persona/second-person; affect items
  deliberately excluded — affect is a *different* candidate skill).
- **Directional only.** Any dimensionality / factor / latent-correlation result at this N
  is a *directional* read. **AIC/BIC are power artifacts at this N and are deliberately not
  used to adjudicate anything.** The multidimensional M2PL is **not identifiable** at
  N≈28 (the calibrator itself warns below 150 persons); its latent correlations below are
  a structure *hint*, not an estimate.
- **General-ability halo.** Per-model group pass-rate correlations are inflated by overall
  model capability (bigger/instruct-tuned models pass *everything* more often). Where the
  aggregate-level and item-level reads disagree, trust the **item-level phi/tetrachoric**
  and the **latent-correlation** reads over the raw group-score correlations.
- **The definitive test** is the full curated-criteria matrix the team is about to
  produce. See §6 pre-registration.

---

## 2. H1 — Is "presentation" its own latent skill dimension?

### 2.1 Is presentation a *measurable trait*? (reliability / variance) — YES (directional)

| Set | # items (gradeable) | # models | mean pass | std | range |
|---|---|---|---|---|---|
| presentation proxy (org ∪ persona) | 428 | 25 | 0.182 | 0.143 | 0.00–0.46 |
| presentation — org/format only | 299 | 25 | 0.178 | 0.148 | 0.00–0.52 |
| presentation — persona/2nd-person only | 129 | 23 | 0.203 | 0.163 | 0.00–0.63 |
| (contrast) affect orphans | 476 | 25 | 0.137 | 0.140 | 0.00–0.45 |

- **Not floor/ceiling.** Presentation pass rates vary substantially across models
  (std ≈ 0.14, full 0→0.5+ range) — there is a real between-model trait to measure.

**Inter-item coherence** (mean pairwise phi / approx-tetrachoric, min overlap 10):

| Set | mean φ | mean tetrachoric~ |
|---|---|---|
| presentation proxy | 0.119 | 0.337 |
| presentation org-only | 0.138 | 0.379 |
| content-primary (ref) | 0.155 | 0.503 |
| diagnosis-primary (ref) | 0.220 | 0.590 |
| scaffolding (ref) | 0.058 | 0.137 |

- Presentation items cohere **about as well as content items** and clearly better than
  scaffolding — consistent with a coherent underlying trait, not noise.

### 2.2 Is it *separable* from content / diagnosis / scaffolding? (mixed → separable from scaffolding, entangled with competence)

**Item-level cross-correlation** (the cleaner read):

| pair | mean φ | tetrachoric~ |
|---|---|---|
| presentation × scaffolding | **0.018** | 0.109 |
| presentation × content-primary | 0.105 | 0.369 |
| presentation × diagnosis-primary | 0.126 | 0.397 |

**Group-score EFA (varimax, 3-factor, 25 listwise models)** — dominant factor per group:

| group | F0 (competence) | F1 | F2 | → factor |
|---|---|---|---|---|
| content-primary | −0.86 | −0.29 | −0.40 | competence |
| diagnosis-primary | −0.90 | −0.21 | −0.37 | competence |
| scaffolding | −0.22 | **−0.97** | −0.08 | scaffolding |
| presentation | −0.44 | −0.09 | **−0.89** | **presentation** |

**Directional M2PL latent correlations** (4-dim, N≈28, NOT identifiable — hint only):

| pair | latent r |
|---|---|
| scaffolding ↔ presentation | **−0.15** |
| diagnosis ↔ presentation | 0.75 |
| content ↔ presentation | 0.81 |
| content ↔ diagnosis | 0.90 |

Reading:
- **Presentation is clearly distinct from scaffolding** (item φ≈0.02, latent r≈−0.15) and
  forms **its own factor** in the group-score EFA.
- **Presentation is entangled with general competence / content** (latent r≈0.75–0.81;
  group-score r≈0.75). Much of this is the capability halo, but it means presentation is
  **not orthogonal** to the existing content/diagnosis axis.
- **Net:** presentation is separable *enough* to be a plausible **added** dimension,
  primarily because it captures something scaffolding and diagnosis do not.

### 2.3 "Replace diagnosis" assessment — statistically tempting, construct-invalid

Compare the current structure against the alternative that frees diagnosis's slot:

| structure | key latent correlations |
|---|---|
| current {content, diagnosis, scaffolding} | content↔diagnosis **0.90**, content↔scaff −0.17, diag↔scaff −0.12 |
| alt {content+diagnosis, scaffolding, presentation} | (c+d)↔presentation **0.70**, (c+d)↔scaff −0.26, scaff↔presentation −0.06 |

- **The statistical question:** *Is presentation a cleaner factor than diagnosis?*
  **Yes.** Diagnosis is essentially fused with content (latent r≈0.90; structural nesting
  r≈0.945; per-model diagnosis-only vs content-only r≈0.97). In the alt structure the three
  factors (content+diagnosis / scaffolding / presentation) are markedly more distinct than
  the current three. So *if the only goal were three well-separated factors,* replacing
  diagnosis with presentation "wins".
- **The construct-validity question:** *Is that a good idea?* **No.** Presentation ≈
  markdown / LaTeX / persona = a formatting & instruct-tuning artifact. Curation
  deliberately tagged it **style_surface / non-gating / excluded from calibration**.
  Diagnosis, though statistically collinear, is a **pedagogically central** construct
  (does the tutor address the student's *specific* error?). Replacing it would optimize a
  statistic (factor separability) by discarding a construct we care about and promoting a
  surface trait — the classic construct-validity trap.

**H1 conclusion:** **Presentation is a plausible ADDED dimension (distinct from
scaffolding, measurable, coherent) but a POOR REPLACEMENT for diagnosis.** Treat it as a
candidate 4th latent skill to test confirmatorily; do not delete diagnosis for it. Keep the
"style_surface / non-gating" framing so it does not silently gate tutoring-quality scores.

---

## 3. H2 — Would reframing "content" as "explanation" make "diagnosis" standalone?

### 3.1 Current empirical structure — diagnosis is NOT standalone

- **Nesting / collinearity (already established):** content ⊃ diagnosis, structural
  r≈0.945; P(content | diagnosis)=0.82.
- **This run:** per-model **diagnosis-only vs content-only r ≈ 0.97**; M2PL latent
  **content↔diagnosis ≈ 0.90–0.91**.
- **EFA (2/3/4-factor, group scores):** `content_only`, `content_diagnosis`, and
  `diagnosis_only` **all load on the same "competence" factor** (loadings 0.88–0.95) at
  every factor count. **Diagnosis never splits off.** Only scaffolding (F1) and
  presentation (F2) form their own factors.

### 3.2 A weak hint that diagnosis carries *some* unique signal

- Within-**diagnosis** inter-item coherence (φ≈0.221, tet~0.586) is **higher** than the
  **content×diagnosis cross** coherence (φ≈0.179, tet~0.544) and higher than within-content
  (φ≈0.145). This is a *faint* signature that diagnosis items share something beyond raw
  content — but the margin (0.22 vs 0.18) is small and well within noise at this N.

### 3.3 Can existing data tell us whether "explanation" reframing helps? — NO

- The current "content" label **bundles raw correctness with explanation**, and diagnosis
  (addressing the student's specific reasoning error) plausibly overlaps with the
  *explanation* portion of content more than with the *correctness* portion. Narrowing
  content to **"explanation quality (independent of correctness)"** could **either**:
  - **reduce** content↔diagnosis overlap (if diagnosis is mostly error-identification,
    which is somewhat orthogonal to explaining the concept), **or**
  - **increase** it (if "explanation quality" and "diagnosis" both tap the same
    reasoning/communication ability).
- The faint within-diagnosis>cross-diagnosis margin (§3.2) leans *weakly* toward "reduction
  is possible," but this is **not decisive**.
- **Critically, this is a relabeling hypothesis.** The matrix is graded under the OLD
  "content = correctness + explanation" definition. You **cannot** re-derive
  "explanation-only" verdicts or new Q-labels post hoc from these columns. Any claim that
  reframing *does* separate diagnosis requires **new Q-labels + regrading** under the new
  definitions.

**H2 conclusion:** **DEFER.** The reframing is a reasonable, testable hypothesis, but it is
**unresolvable from the existing matrix**. It must be tested on the full curated matrix
after content/explanation and diagnosis are re-labeled under the new definitions.

---

## 4. Evidence index (where the numbers live)

- Script: `scripts/investigate_skill_options.py` (self-contained, re-runnable).
- Machine-readable results: `staging/skill_options_report.json`.
- Per-model group pass-rate table: `staging/skill_options_report.csv`.
- Group-score correlation matrix: `staging/skill_options_corr.csv`.
- Prior context reused (not modified): `staging/collinearity_report.json`,
  `staging/orphan_criteria.{csv,json}`.

---

## 5. What is NOT computable from current data (stated explicitly)

- The behavior of the **curated one-per-scenario, normalized** presentation criterion
  (we only have the granular OLD orphans as a proxy).
- Any **relabeling** result for H2 (explanation-only content, re-scoped diagnosis).
- Identifiable multidimensional IRT parameters or trustworthy AIC/BIC — N is far too small.
- Reliable factor structure at the **item** level (N≪#items); we use group-score EFA +
  item-pair coherence + the directional M2PL latent correlation instead.

---

## 6. PRE-REGISTRATION — confirmatory analyses for the full curated-criteria matrix

Run these **once** on the full curated matrix (curated criterion_ids, curated Q-matrix,
full model fleet). Fix the analysis plan and thresholds *before* looking at the results.

### 6.1 Sample / power gates (must pass before interpreting factor results)
- **G1.** ≥ 150 usable persons (models) after zero-variance/empty removal — the
  calibrator's identifiability floor for a 3-dim M2PL; ≥ 250 preferred for 4-dim.
- **G2.** Each candidate dimension (incl. presentation) has ≥ 30 gradeable items with
  between-model variance (not floor/ceiling).
- **G3.** Report per-item observation counts; drop items with < 30 observed persons from
  loading interpretation.

### 6.2 H1 — presentation as an ADDED 4th dimension
Fit the confirmatory M2PL with the curated Q-matrix plus a presentation column
(`calibrate_mirt.py --estimate-latent-corr`, curated rubrics).
- **A1 (measurable):** presentation item-set reliability (e.g., tetrachoric ω or mean
  inter-item tetrachoric) **≥ 0.6**, and per-model presentation θ variance not degenerate.
- **A2 (separable from scaffolding):** latent |r(presentation, scaffolding)| **< 0.4**.
- **A3 (distinct from competence):** latent r(presentation, content) **< 0.7** — i.e.,
  meaningfully below the content↔diagnosis fusion level. If r ≥ 0.85, presentation is a
  competence/halo restatement, not a new axis.
- **A4 (added value):** 4-dim {content, diagnosis, scaffolding, presentation} improves fit
  over 3-dim by likelihood-ratio (report LRT + ΔAIC/ΔBIC *only now that N is powered*),
  **and** presentation items show clean simple structure in EFA (primary loading > 0.4,
  max cross-loading < 0.3).
- **Decision:** ADD presentation iff A1 ∧ A2 ∧ A3 ∧ A4. Regardless of A-results, keep its
  **style_surface / non-gating** flag: presentation must not gate tutoring-quality scoring.

### 6.3 H1 — presentation as a REPLACEMENT for diagnosis (construct gate)
- **B1 (statistical):** compare {content, diagnosis, scaffolding} vs
  {content+diagnosis, scaffolding, presentation} on fit + factor separability.
- **B2 (construct veto):** *Even if B1 favors the replacement,* require an explicit
  construct-validity sign-off that dropping diagnosis is acceptable. Pre-committed default:
  **do not replace diagnosis with presentation** — presentation is surface/style;
  diagnosis is pedagogically central. Replacement only if the team documents that diagnosis
  is redundant with content *and* presentation is re-scoped to a substantive construct.

### 6.4 H2 — content→explanation, diagnosis separability (requires relabeling)
- **C0 (prerequisite):** re-derive Q-labels under the new definitions — "content" restricted
  to **explanation quality (independent of correctness)**, diagnosis re-scoped to
  error/reasoning identification — and **regrade** (or re-map existing verdicts only if the
  criteria are genuinely unchanged in scope). Without C0, H2 is not testable.
- **C1 (primary):** latent r(explanation, diagnosis) drops to **< 0.7** (from the current
  ≈0.90) — the key sign that reframing separated the constructs.
- **C2 (structure):** in EFA, diagnosis items form a factor distinct from explanation
  items (diagnosis primary loading > 0.4, cross-loading on explanation < 0.3).
- **C3 (guardrail):** confirm the reframing did not simply move the collinearity elsewhere
  (e.g., explanation↔scaffolding) — check all off-diagonal latent correlations.
- **Decision:** adopt the content→explanation reframing iff C1 ∧ C2 ∧ C3. If r stays
  ≥ 0.85 after relabeling, diagnosis is genuinely nested in the reasoning axis and should be
  collapsed, not separated.

### 6.5 Reporting requirements for the confirmatory run
- Report point estimates **with uncertainty** (bootstrap latent correlations over models).
- Pre-commit to the thresholds above; report pass/fail per criterion, not just p-values.
- Keep affect/motivation as a **separate** candidate skill probe (it also shows spread here)
  — do not fold it into presentation.
