# 00 — Skill Definitions: Synthesis + Pre-registration

**Status:** consolidation / decision memo. **Changes nothing.** It does not touch
`data/scenarios.jsonl`, `data/curated/rubrics_qmatrix_curated.jsonl`, any `q_mapping`, the skill
definitions, the IRT parameters, or any calibration/judge code. It **synthesizes** six prior
memos into a single decision doc for a human to drive discussion while the full judging run
completes.

**What this memo is for.** The multidimensional-IRT (M2PL) model over three latent tutor skills —
**content** (domain correctness), **diagnosis** (identifying *this* student's specific
error/confusion), **scaffolding** (structuring/sequencing help) — needs a settled skill-space
*direction* before the full-matrix fit. Three independent lines of evidence have now been worked
out. This memo states where they converge, what to pre-register, and what the human must decide.

**Source memos (cross-linked by relative path — read for detail, this doc does not restate them):**
- [`01 Tutoring Frameworks + Content-Diagnosis Boundary.md`](01%20Tutoring%20Frameworks%20+%20Content-Diagnosis%20Boundary.md) — classic teaching/teacher-knowledge frameworks.
- [`02 Candidate Additional Dimensions.md`](02%20Candidate%20Additional%20Dimensions.md) — candidate 4th-axis assessment.
- [`03 LLM-Tutor Evaluation Rubrics.md`](03%20LLM-Tutor%20Evaluation%20Rubrics.md) — how modern LLM-tutor benchmarks score skill.
- [`../Collinearity Analysis + Skill-Definition Options.md`](../Collinearity%20Analysis%20+%20Skill-Definition%20Options.md) — our own structural + EFA/latent-r analysis.
- [`../Skill-Definition Options - Presentation + Explanation.md`](../Skill-Definition%20Options%20-%20Presentation%20+%20Explanation.md) — presentation/explanation options + A/B/C pre-registration.
- [`../Tutor Skill Taxonomy - Theory + Candidate Dimensions.md`](../Tutor%20Skill%20Taxonomy%20-%20Theory%20+%20Candidate%20Dimensions.md) — why content↔diagnosis overlap; affect-EFA gate.

> ⚠️ **Read this before any number.** Every empirical figure below is **DIRECTIONAL**. It comes
> from a **N≈25–28** biased small-model subsample on the **OLD (pre-curation)** ~1/4-filled
> criteria matrix. At that N the M2PL is **not identifiable** (calibrator floor ≈150 persons) and
> AIC/BIC are **power artifacts**. **The full curated-matrix run is the only adjudicator.** Nothing
> here is a decision; it is a pre-registered direction plus the gate table that will decide it.

---

## 1. Executive summary

Three independent lines of evidence — **classic teaching/teacher-knowledge frameworks**,
**modern LLM-tutor evaluation rubrics**, and **our own structural + EFA/collinearity analysis** —
converge on the same skill-space verdict:

- **Diagnosis is a real, field-recognized competency but is fundamentally NESTED in content**, not
  a separable latent axis in our instrument. All three lines agree: no validated teaching-quality
  model isolates diagnosis as its own factor; PCK is defined over CK and fuses toward r≈1 in
  expert/bundled regimes; our directional latent r = 0.945 (structural co-occurrence r≈0.945;
  P(content|diagnosis)=81.9%) sits squarely inside that envelope. External LLM rubrics *do* score
  mistake-identification standalone — but on datasets where a student mistake is present ~100% of
  the time, which validates diagnosis as a **label**, not as a separable **latent axis** on our
  mixed bank.
- **Scaffolding is the cleanest, most consensus-backed axis.** It separates cleanly in our data
  (latent r ≈ −0.19 with content/diagnosis) and "withhold / don't reveal the answer" is the single
  most universal external tutoring sub-criterion.
- **Affect/motivation is the strongest genuine candidate for an added dimension** — the only
  candidate with *external factor-separability* evidence (CLASS Emotional Support), recurring
  across 5/7 LLM rubrics, and domain-independent by construction. Its binding risk is **variance**
  (is it discriminating on a strong fleet?), **not collinearity**.
- **Everything else is a subtype or cross-cut, not a new axis:** metacognition, communication
  clarity, adaptivity/contingency, Socratic questioning, and feedback quality all decompose into or
  re-collide with {correctness/diagnosis, scaffolding, affect}. Presentation is a `style_surface`
  reporting facet, not a skill.

**Recommended direction (pending the full-matrix fit):** model **two latent axes —
{correctness/diagnosis (collapsed), scaffolding}** — and **PRE-REGISTER affect/motivation** as a
candidate third axis to be adopted only on evidence. Retain **diagnosis, presentation,
adaptability, and actionability as reported (non-latent) facets.** Collapse content+diagnosis *for
calibration*; retain diagnosis *for reporting*.

---

## 2. Convergent conclusions (evidence from each source line)

### 2.1 Diagnosis is real but FUNDAMENTALLY NESTED in content

| Line | Supporting evidence |
|---|---|
| **Classic frameworks** ([01](01%20Tutoring%20Frameworks%20+%20Content-Diagnosis%20Boundary.md), [taxonomy](../Tutor%20Skill%20Taxonomy%20-%20Theory%20+%20Candidate%20Dimensions.md)) | **No validated multidimensional teaching-quality model isolates diagnosis as its own factor.** Diagnosis is a PCK facet ("knowledge of students' (mis)conceptions") *defined over* content (Shulman 1986/1987) — CK ⊃ PCK. CK–PCK correlation rises to **statistically indistinguishable from 1** in expert (academic-track) teachers (COACTIV; Krauss et al. 2008), is **inseparable** for elementary specialized CK, and **Danielson's 4 FfT domains collapsed at r > .90** ("nearly redundant," REL-West 2016) — a direct precedent for our proposed collapse. Where teaching instruments *do* separate a factor, it is affect or scaffolding, never diagnosis. |
| **LLM-tutor rubrics** ([03](03%20LLM-Tutor%20Evaluation%20Rubrics.md)) | External rubrics **DO** score mistake-ID standalone (MRBench "mistake identification" + "mistake location"; Bridge Step A "identify the student's error"; TutorBench "identifying core misconceptions"; LearnLM "guide mistake discovery"). **BUT** these run on mistake-remediation datasets where **diagnosis base-rate ≈ 100% by construction**, so diagnosis co-varies less with raw content than in our mixed bank. This validates diagnosis as a **conceptual/labeling axis**, and is explicitly agnostic about *empirical separability* on our instrument. |
| **Our EFA/collinearity** ([collinearity](../Collinearity%20Analysis%20+%20Skill-Definition%20Options.md), [presentation](../Skill-Definition%20Options%20-%20Presentation%20+%20Explanation.md)) | **Structural near-nesting:** P(content\|diagnosis)=**81.9%** vs P(diagnosis\|content)=42.9% — diagnosis behaves as a *sub-region* of content. **Directional latent r = 0.945** (per-model content-only vs diagnosis-only r ≈ 0.97; item-level cross-skill phi ≈ within-skill phi). EFA: content-only, content+diagnosis, and diagnosis-only **all load one "competence" factor** at every factor count — **diagnosis never splits off**. Narrowing the definition dissolves only **1.7%** of the overlap ⇒ the entanglement is intrinsic, not a labeling artifact. |

**Verdict:** diagnosis is a genuine competency (keep the label; the judge still scores mistake-ID)
but is not expected to hold up as a separate latent dimension. Danielson is the template: **keep
the rubric language, report the collapsed dimension.**

### 2.2 Scaffolding is the cleanest, most consensus-backed axis

| Line | Supporting evidence |
|---|---|
| **Classic frameworks** | Where teaching-quality models cleanly separate a *non-affect* factor, it is scaffolding / cognitive activation (Three Basic Dimensions; VanLehn locates the tutoring *effect* in interaction granularity, not raw content). Contingency/fading/transfer-of-responsibility are the *defining* features **of** scaffolding (van de Pol et al. 2010). |
| **LLM-tutor rubrics** | **"Withhold / don't reveal the answer" is the single most universal external sub-criterion** — scored by LearnLM, MRBench, MathDial (Telling = failure mode), Tutor CoPilot (Provide-Answer = Low), and Pedagogical-Alignment. Guiding-questions/prompt-reasoning recurs across ≥5 rubrics. |
| **Our EFA/collinearity** | Scaffolding separates cleanly: latent r ≈ **−0.19** with content/diagnosis; forms its own EFA factor (F1) distinct from competence; item φ with presentation ≈ 0.02. (Human κ 0.59 for scaffolding vs 0.365 for diagnosis; scaffolding items cohere less tightly, matching the field's "scaffolding is hard to measure" note, but it is unambiguously a distinct axis.) |

### 2.3 Affect/motivation is the strongest genuine candidate for an added dimension

| Line | Supporting evidence |
|---|---|
| **Classic frameworks** | **The only candidate with external factor-separability evidence:** CLASS "Emotional Support" is a factor **empirically separable from Instructional Support** (CFA over 4,000+ classrooms; 26-matrix meta-analysis). Three Basic Dimensions (student support) and INSPIRE (Nurturant/Encouraging) also peel off affect. Domain-independent by construction. |
| **LLM-tutor rubrics** | Tutor tone / motivation / affect recurs in **5 of 7 rubrics** (LearnLM, MRBench, TutorBench style+emotional, Tutor CoPilot, Bridge "care") — our single biggest omission. Nuance to import: Tutor CoPilot scores *generic* encouragement as **low-quality**, echoing our v2 rule that generic acknowledgement is all-zero (support ≠ mere sentiment). |
| **Our EFA/collinearity** | Affect is a genuine **unmodeled orphan** — plentiful in the bank and currently all-zero (not absorbed like adaptability). Its binding risk is **variance, not collinearity**: affective competence may be near-ceiling/low-variance on a strong fleet, sinking discrimination regardless of correlation. This is the make-or-break unknown that only the powered EFA can settle. |

### 2.4 The rest are subtypes / cross-cuts, not new axes

- **Metacognition / self-regulated-learning support** — framed as a *subtype of scaffolding*
  ("process/metacognitive scaffolding," Azevedo). Our reflection/self-check items are guiding
  questions ⇒ already mapped to scaffolding; would re-collide.
- **Communication / explanation clarity** — domain-independent but frequently **co-authored with
  affect** in the same criterion and likely low-variance ⇒ at best a sub-facet of a delivery/affect
  axis, not standalone.
- **Adaptivity / contingency** — *constitutive of* scaffolding (van de Pol et al. 2010); already
  decided as a **reporting facet**, absorbed into scaffolding+diagnosis.
- **Socratic questioning / elicitation** — definitionally the "guiding-question / withhold-the-
  answer" form of scaffolding (INSPIRE "S"; AutoTutor EMT).
- **Feedback quality** — **decomposes** into the existing axes (Hattie & Timperley task/process =
  correctness/diagnosis+scaffolding; self-reg = metacognition; self = affect); not a fourth skill.
- **Presentation** — a coherent, separable-from-scaffolding **surface/style** trait (markdown/
  LaTeX/persona), but a *poor pedagogical substitute* for diagnosis. Keep as a `style_surface`,
  **non-gating reporting facet**, not a latent skill.

---

## 3. Recommended skill-space direction (pending full-matrix confirmation)

- **LATENT axes:**
  - **{correctness/diagnosis (collapsed), scaffolding}** — the likely 2-dim structure.
  - **PRE-REGISTER `affect/motivation`** as a candidate third latent axis, adopted **only if** the
    gate in §7 fires (coherent factor + meaningful variance + latent r < ~0.9 with both others).
- **REPORTED (non-latent) facets:** **diagnosis** (judge still scores mistake-ID), **presentation**
  (style_surface, non-gating), **adaptability** (via `use_case` stratification), and
  **actionability** (see §5).

This preserves every construct the benchmark wants to speak to (diagnosis, presentation,
adaptability, actionability) while calibrating on a clean, identifiable latent space.

---

## 4. Correctness/Diagnosis collapse — framing

**What the collapse means.** We fold **content + diagnosis** into ONE latent axis best named
**"Correctness & Diagnosis"** (alt: **"Domain Competence"** / "subject-matter competence-in-
context"). Rationale, drawn from all three lines: **diagnosis is applying content knowledge to the
student's specific work** — PCK nested in CK — so a criterion that reads "correct the student's
specific error" genuinely requires *both* abilities and cannot be attributed to one on a single
pass/fail item (the measurement confound). Prerequisite structure guarantees P(correct diagnosis |
wrong content) ≈ 0.

**Mechanics of the collapse.** Every content-or-diagnosis criterion loads the single merged axis —
including the **~401 diagnosis-only** curated items (a merged item loads the combined dim iff it
loaded either content *or* diagnosis). There is **no separate diagnosis θ**.

**What we keep.** `diagnosis` stays a **reported sub-facet**: the judge continues to score
mistake-identification per-criterion, and we report error-reading as a distinct tutoring
competency via tagging/`use_case`. This is the Danielson precedent (keep the language, report the
merged dimension) and is fully reversible — Q-matrix regen + re-fit only; tutor responses and
judge verdicts are reused.

**Net: collapse for calibration, retain for reporting.**

---

## 5. "Actionability" definition

**Actionability** = whether the response gives the student a **concrete, usable NEXT STEP**, versus
vague/abstract feedback. It recurs across **MRBench** (dim 5, "clear what to do next"),
**TutorBench** (conciseness & relevance / implicit), **LearnLM** ("guides appropriately"), and
**Bridge** ("usefulness"). It is distinct from **content** (a next step can be actionable without
re-deriving the answer) and from **withhold-the-answer scaffolding** (a good hint can be actionable
*without* revealing the solution). Because it partially rides inside scaffolding and has no clean
external factor-separability evidence, **recommend tracking actionability as a reported facet, not
a clean latent axis.**

---

## 6. Affect/motivation prevalence in the curated bank

Exact heuristic counts from
[`../../staging/curated_orphan/orphan_criteria.json`](../../staging/curated_orphan/orphan_criteria.json)
(`data/curated/rubrics_qmatrix_curated.jsonl`, **6,845** curated criteria):

| Measure | Count | Share |
|---|---:|---:|
| `affect_motivation` — single-label (precedence) | **581** | ~8.5% of 6,845 |
| `affect_motivation` — multi-label | **605** | ~8.8% of 6,845 |
| `affect_motivation` within the all-zero pool (precedence) | **460** | ~33% of the 1,412 all-zero pool |

**Curation side-effect to keep in mind.** Curation's normalized **presentation** criterion
inflated the **all-zero pool 1,107 → 1,412** and **organization_structure 408 → 756** — so the
all-zero pool is now dominated by presentation/organization plus affect, not by hidden pedagogical
signal.

**Caveat.** These are a **heuristic keyword match** over free text (needs hand audit; see the CSV),
and the affect items are **mostly non-critical** ("be encouraging" boilerplate). **Viability of an
affect axis hinges on variance**, not prevalence — a large but near-ceiling bucket still fails to
identify a latent dimension. This is the empirical question deferred to the full-matrix EFA.

---

## 7. PRE-REGISTRATION — consolidated gate table (run on the full curated-criteria matrix)

Fix these thresholds **before** looking at results. Consolidated from the presentation/explanation
memo (A1–A4, B1–B2, C0–C3) and the taxonomy/affect-EFA gate. Report pass/fail per gate **with
bootstrap uncertainty**, not just p-values.

| Gate | Test | Threshold / rule |
|---|---|---|
| **G1 — power** | Usable persons (models) after zero-variance/empty removal | **≥ 150** for a 3-dim M2PL (calibrator identifiability floor); **≥ 250** preferred for 4-dim. *Below this, ignore all factor/AIC/BIC results.* |
| **G2 — item variance** | Each candidate dimension has gradeable items with between-model variance | **≥ 30** items, not floor/ceiling |
| **G3 — obs count** | Per-item observed persons | drop items with **< 30** observed persons from loading interpretation |
| **CD1 — collapse: latent r** | content↔diagnosis M2PL latent correlation (`--estimate-latent-corr`) | **adopt collapse iff r > ~0.9** |
| **CD2 — collapse: fit** | collapsed 2-dim vs 3-dim on AIC/BIC (`--collapse content,diagnosis`) | collapsed model wins on AIC/BIC |
| **CD3 — collapse: structure** | EFA / scree | no separate content/diagnosis factor emerges |
| — | **Collapse decision** | **Collapse content+diagnosis iff CD1 ∧ CD2 ∧ CD3.** Otherwise keep 3 dims. Scaffolding stays distinct in both branches. |
| **AF1 — affect: coherence** | residual EFA factor after the collapse | loads **coherently** on the `affect_motivation` items (primary loading > 0.4, cross-loading < 0.3) |
| **AF2 — affect: variance** | affect θ / item variance on the **full** fleet (incl. strong models) | **meaningful, not near-ceiling/degenerate** |
| **AF3 — affect: separability** | affect latent correlation with **both** other axes | **latent r < ~0.9 with BOTH** correctness/diagnosis **and** scaffolding |
| — | **Affect decision** | **Adopt `affect/motivation` as the third axis iff AF1 ∧ AF2 ∧ AF3.** Otherwise keep affect as descriptive all-zero, reported via `use_case`/critical-failure monitoring. |
| **PR1–PR4 — presentation (optional 4th)** | measurable (tetrachoric ω ≥ 0.6, θ non-degenerate); separable from scaffolding (\|r\| < 0.4); distinct from competence (r(pres,content) < 0.7; if ≥ 0.85 it is a halo restatement); added value (LRT + clean simple structure) | **ADD presentation iff PR1 ∧ PR2 ∧ PR3 ∧ PR4** — and **regardless, keep its `style_surface` / non-gating flag.** Do **not** use presentation to *replace* diagnosis (construct veto). |
| **EX1–EX3 — content→explanation (H2, requires relabel)** | prerequisite: re-derive Q-labels + regrade under explanation-only content; then r(explanation, diagnosis) < 0.7; diagnosis forms distinct EFA factor; collinearity not merely displaced | adopt reframing iff all hold; if r stays ≥ 0.85, diagnosis is nested ⇒ collapse, not separate. **Not testable without relabeling.** |

**Re-run commands (on the FULL matrix):**

```bash
python scripts/calibrate_mirt.py --estimate-latent-corr --efa --matrix <FULL_MATRIX.csv>
python scripts/calibrate_mirt.py --collapse content,diagnosis --estimate-latent-corr --efa --matrix <FULL_MATRIX.csv>
```

Then use `staging/curated_orphan/orphan_criteria.csv` to identify each bucket's items for a
throwaway candidate-loading regen + re-fit (affect / presentation), per AF/PR gates.

---

## 8. Open decisions for the human

- **Keep `diagnosis` as a reported facet after the collapse?** (Recommended: yes — judge still
  scores mistake-ID; preserves the benchmark's error-reading story at zero re-grade cost.)
- **Add `actionability` as a reported facet?** (Recommended: yes as a facet, not a latent axis —
  recurs in ≥3 external rubrics but is entangled with scaffolding.)
- **Affect/motivation — full latent dimension vs reported-facet-first?** (Recommended:
  reported-facet-first, promote to a latent axis **only** if the §7 AF gate fires — its risk is
  variance, and adopting a degenerate axis is a false-1 hazard hard to detect at finite N.)

---

> **Caveat, restated for prominence.** All empirical numbers here are **DIRECTIONAL**: N≈25–28, a
> biased small-model subsample on the OLD criteria, ~1/4-filled matrix, non-identifiable M2PL,
> AIC/BIC as power artifacts. This memo pre-registers a direction and the gates that will decide
> it; **the full curated-matrix run is the adjudicator.**
