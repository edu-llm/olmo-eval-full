# Skill-Definition Design History (pre-empirical)

Consolidated pre-decision design intent, skill-taxonomy theory, and candidate-dimension
analyses that led to the 3-skill instrument. Supersedes: `Implementation Strategy + Success
Metrics`, `LLM-JUDGE Implementation Strategy + Success Metrics`, `IRT Calibration PRD`,
`Collinearity Analysis + Skill-Definition Options`, `Skill-Definition Options - Presentation +
Explanation`, `TutorBench Use Cases + Adaptability`, `Tutor Skill Taxonomy - Theory + Candidate
Dimensions`, the `research/00-03` literature threads, and the two research dumps (`Adaptive
Testing + Tiny Benchmark`, `LLM-As-A-Judge Alternatives`).

**Canonical current definitions live in `Skill Definitions v2 (Q-matrix + Judge).md` (kept).**
The empirical outcomes that these design hypotheses were tested against live in
`Calibration Log.md` (Runs 1-6). Note the divergence: this design phase flagged
**motivation/affect** as the strongest candidate 3rd axis, but the empirical fits later
**deferred motivation (Run 2) and adopted presentation (Runs 5-6)** instead.

All memos below were read-only (no bank/scenario/fitter/code changes); every pre-empirical
number was directional (N approx 25-28 on the OLD ~1/4-filled pre-curation matrix, below the
~150-person identifiability floor).

---

## 1. Original CAT/MIRT design intent (`Implementation Strategy`)

Goal: a skill-based Computerized Adaptive Test targeting a ~100x reduction in items needed to
predict a model's per-skill benchmark accuracy at similar precision. Built on ATLAS's CAT
framework (cf. tinyBenchmarks, FLUID), extended two ways: (1) calibrate on education/tutoring
benchmarks, not just general ones; (2) **multidimensional** ability (per-skill), not one overall
score.

- **Original skill vector was 4:** content, diagnosis, scaffolding, **adaptability** (change
  explanations given prior work/confusions). Adaptability was later dropped -> canonical 3-skill.
- **Q-matrix:** item->skill map; a skill marked 1 only if strictly necessary. Synthetic
  generation via an LLM (given a TutorBench row + skill defs + labeling examples; each `1`
  requires evidence + counterfactual), then two independent LLM verifiers + a human sample,
  unanimous-positive kept; a sample of `0`s checked for false-negative rate. Then compare
  multidim vs unidim and alternative Q-mappings.
- **CAT:** random uniform start, then Fisher information; multidimensional -> **D-optimality**
  (maximize `det` of the updated dxd information matrix); select the highest-SE skill to drive
  the next item. (Note: the shipped harness uses the info trace, not full D-optimality - a spec
  deviation flagged in `Analysis Notes.md` sec 1.)
- **Stopping rule (design):** each skill >=15 observations and SE<=0.3, plus >=40 total obs and
  overall SE<=0.2.
- **Success metrics (design):** correlation 0.7-0.85 with TutorBench, 0.4-0.7 with general
  frontier benchmarks; median length <=75 items, p90 <=100. Ideal length/correlation: 10->0.7,
  25->0.85, 50->0.92, 75->0.95, 100->0.97.

## 2. Original benchmark/skill list (`IRT Calibration PRD`)

Early, informal PRD listing the intended calibration banks and per-bank Q-matrix skills:
- MCQ: ARC (Easy/Challenge), OpenBookQA, Pedagogy Benchmark, EducationQ (MMLU-Pro-stratified) -
  each with tentative per-bank skill lists (fact-recall, explanatory/multi-hop/experimental
  reasoning, teaching strategies, assessment creation, etc.). "Probably a separate Q-matrix skill
  set per benchmark."
- Open-ended: TutorBench (~600), TutorEval (~800), EduBench (~9,600).
- A ~100-model calibration roster of small open models (gemma-3-270m through the low-billions),
  the "common persons" for IRT linking.

## 3. LLM-judge integration intent (`LLM-JUDGE Implementation Strategy`)

Extends the original MCQ-only proposal to **free-response (FRQ)** grading via an LLM judge.
- **Reframing vs TutorBench:** no negative IRT weights + a **separate critical-failure flag**
  (e.g. "avoids revealing the answer" = +1 rather than "reveals answer" = -5), so higher ability
  monotonically raises pass probability and critical failures are not compensated by good
  performance elsewhere.
- **Preprocessing:** split each dataset item into a **scenario schema** (prompt, conversation
  context, reference solution, criterion_ids) and per-criterion **rubric schemas** (criterion,
  expected_evidence, q_mapping, criticality, objectivity, explicitness). Plus **judge-result** and
  **running-MIRT-ability** schemas. (These are now `tutor_cat/schemas.py`.)
- **CAT loop:** tutor answers scenario -> judge pass/fails each criterion -> update MIRT vector ->
  pick highest-SE skill -> choose next scenario by Fisher info averaged over that skill's criteria,
  top-n random pick. Stop when every skill < max SE and >= 15 evaluations.
- **Judge workflow:** criteria batched by skill; one judge scores all, a second re-checks critical
  failures / repeated-run disagreements / a random sample.
- **Judge acceptance criteria:** Macro-F1 >= 0.80; critical-failure sensitivity >= 0.90;
  test-retest agreement >= 0.90; no per-skill F1 < 0.70; prompt-consistency variation <= 0.1;
  marginal reliability >= 0.7.
- **5-candidate judge shortlist:** Prometheus 2 7B (specialist, self-host), GPT-5.6 Luna, Claude
  Haiku 4.5, Gemini 3.5 Flash-Lite (efficient API generalists), Qwen3.5-9B (open generalist).
  **Qwen3.5-9B was selected** as the frozen judge (see `docs/judge_validation.md` and the grading
  runs). Note this doc's tail was marked "UNDER CONSTRUCTION."

## 4. Skill taxonomy theory: why content and diagnosis overlap (`Tutor Skill Taxonomy`, `research/01`)

Frames the 3 skills against Shulman's teacher-knowledge partition: our **content** approx Content
Knowledge (CK); **diagnosis** is a **Pedagogical Content Knowledge (PCK)** facet ("knowledge of
students' misconceptions"), which is *defined over* content; **scaffolding** straddles PCK
representations and general pedagogical knowledge.

The content<->diagnosis overlap is **overwhelmingly FUNDAMENTAL, not a labeling artifact**:
1. **PCK nested in CK** (Shulman) - diagnosis is defined over content, can't be defined away.
2. **Prerequisite/conditional dependence** - diagnosing *this* specific error presupposes knowing
   the correct answer; P(correct diagnosis | wrong content) approx 0. Matches the near-nesting
   footprint P(content|diagnosis)=81.9% vs P(diagnosis|content)=42.9%.
3. **Expert-regime collinearity** - COACTIV measured teacher CK<->PCK manifest r=0.60, but latent
   r "indistinguishable from 1" in the high-expertise (academic-track) subgroup (Krauss/Baumert
   2008). Our bundled-item, weak-fleet regime is exactly that condition, so directional latent
   r=0.945 is *expected*, not anomalous. Baumert 2010: CK/PCK still distinct in principle, PCK
   predicts learning gains.
4. **Bundled pass/fail items** - one dichotomous item can't attribute a pass to CK vs PCK; only
   ~1.7% of the overlap dissolves under a narrower diagnosis definition.

Human raters agreed well on content (Fleiss kappa 0.85) and scaffolding (0.59) but **poorly on
diagnosis (0.365)** - independent support that diagnosis is not cleanly separable. External
LLM-tutor rubrics (`research/03`: MRBench, Bridge, MathDial, LearnLM, Tutor CoPilot) *do* score
mistake-identification standalone - but on datasets where a student mistake is present ~100% of
the time, which validates diagnosis as a **label**, not as a separable **latent axis** on our
mixed bank.

## 5. content/diagnosis collapse: options + decision rule (`Collinearity Analysis`)

Structural (6,462 criteria): content 4,524, diagnosis 2,372, scaffolding 1,136; content^diagnosis
1,943; diagnosis-without-content 429 (18.1% of diagnosis) - splitting into genuine error-detection
(separable) vs acknowledgement (borderline). Directional 1/4-matrix: per-model content<->diagnosis
r=0.9665, latent r=0.945, scaffolding r approx -0.19.

- **Option A (recommended pre-empirically):** keep 3 skills, decide from the full-data fit
  (latent r + EFA + 3-dim-vs-collapsed AIC/BIC); add a `--collapse content,diagnosis` fit-time flag.
- **Option B:** collapse content+diagnosis -> "correctness", keep scaffolding.
- **Option C (off the table):** narrow the diagnosis definition - touches only ~1.7% and re-opens
  the false-0 risk v2.1 closed deliberately.
- **Decision rule:** collapse IFF latent r >~ 0.9 AND collapsed-2-dim wins AIC/BIC AND EFA finds no
  separate factor. (Re-work is cheap/reversible: Q-matrix regen + re-fit; tutor responses and judge
  verdicts reused.) -> Empirically **Option B fired** (see `Calibration Log.md` Run 1/5).

## 6. Presentation + Explanation options (`Skill-Definition Options`)

- **H1 - presentation as a latent dimension:** ADD as a candidate (distinct from scaffolding: item
  phi approx 0.02, latent r approx -0.15; coheres as its own EFA factor), but it is a **surface/style**
  construct - a statistically cleaner factor than diagnosis yet a **poor pedagogical replacement**
  for it (construct-validity veto). Keep the **style_surface / non-gating** flag so it never gates
  tutoring-quality scores. (Empirically presentation *was* adopted as the optional 3rd axis, Runs 5-6.)
- **H2 - reframe "content" as "explanation quality" to free diagnosis:** DEFER - not testable from
  existing data; it is a relabeling hypothesis requiring new Q-labels + regrading. Pre-registered
  gates C0-C3 (needs latent r(explanation,diagnosis) < 0.7 after relabel).

## 7. Adaptability = use case, not a skill (`TutorBench Use Cases`)

TutorBench's three use cases are task types, not latent skills: Adaptive Explanation Generation
(`adaptive_explanation`, 329/662 approx 50%), Assessment and Feedback (`feedback`, 167), Active
Learning Support (`hint_generation`, 166). Adaptability was the original dropped 4th skill; its
criteria (analogies, relatable examples, tailoring) are **absorbed into scaffolding (+ content/
diagnosis)**, not an unmodeled orphan. **Recommendation: report `use_case` as a stratification
facet, do not add adaptability as a latent axis** (it would be even more collinear with
scaffolding+diagnosis). Revisit only if post-collapse EFA surfaces a residual factor loading on
adaptive-explanation criteria.

## 8. Candidate 3rd/4th axes, ranked (`Tutor Skill Taxonomy` Q2, `research/02`)

A candidate must be (1) conceptually distinct, (2) empirically separable (low latent r, the hard
bar - domain-independence is the key predictor), and (3) present as discriminating items.

| Rank | Candidate | Verdict |
|---|---|---|
| 1 | **Motivation / affective support** | Strongest: domain-independent (can't be dragged into correctness), plentiful, and a genuine **unmodeled orphan** (currently all-zero, ~500-600 of the 1,107 all-zero criteria). Risk is **variance** (tone is easy/near-ceiling for capable models), not collinearity. Evidence-gated via EFA. |
| 2 | Metacognition / self-regulation | Likely **re-collides with scaffolding** (reflection/self-check prompts are guiding questions); conceptually strong, empirically weak on our instrument. |
| 3 | Communication / explanation clarity | Domain-independent but often co-authored with affect and likely low-variance; not its own axis. |
| 4-7 | Socratic questioning, examples/analogies, feedback quality, cognitive-load/pacing | Not separable - definitionally scaffolding, or decompose into the existing content/diagnosis/scaffolding axes. |

**Orphan probe** (`classify_orphan_criteria.py`): the all-zero pool splits ~half affect/tone,
~third formatting/organization, thin clarity tail; the scaffolding-only pool is
questioning/reflection (already scaffolding). A **broad `communication/delivery` axis re-collides
with scaffolding** (its pedagogically-active items are already scaffolding) or collapses to
low-signal formatting; only the **narrow affect** axis is a viable evidence-gated candidate.
Pre-registered rule: adopt narrow affect IFF a residual EFA factor loads coherently on affect
items with real variance AND latent r < ~0.9 with both other axes.

## 9. Literature/design basis (research threads + dumps)

- `research/00` - synthesis + pre-registration: three lines (teacher-knowledge frameworks,
  LLM-tutor rubrics, our own EFA/collinearity) converge - diagnosis nested in content, scaffolding
  cleanest axis, motivation the best orphan candidate; states the pre-registered gate tables.
- `research/01` - canonical teaching/teacher-knowledge frameworks (Shulman CK/PCK/GPK, COACTIV,
  Fleiss kappa evidence) refereeing the content<->diagnosis boundary.
- `research/02` - candidate additional dimensions with citable sources (INSPIRE, control-value
  theory, Affective AutoTutor for motivation; Zimmerman SRL; ICAP; Hattie/Timperley feedback).
- `research/03` - how external LLM-tutor benchmarks (LearnLM, MRBench, MathDial, Bridge, Tutor
  CoPilot) decompose tutoring skill; diagnosis scored standalone but on ~100%-mistake datasets.
- Research dumps - raw notes on adaptive-testing/tinyBenchmarks methodology (the source of the
  3-use-case framing) and LLM-as-judge alternatives; superseded by the formal memos above,
  `docs/judge_validation.md`, and Skill Definitions v2.
