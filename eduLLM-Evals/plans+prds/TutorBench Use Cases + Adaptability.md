# TutorBench Use Cases + Adaptability (is it a missing skill dimension?)

**Status:** analysis memo. **Changes nothing.** Does not touch `data/scenarios.jsonl`,
`data/rubrics_qmatrix_final.jsonl`, the `q_mapping`s, or the skill definitions in
`Skill Definitions v2 (Q-matrix + Judge).md`. It answers: what are TutorBench's three use
cases, how are our scenarios distributed across them, how do they relate to our three skills
(content / diagnosis / scaffolding), and — given the content↔diagnosis collinearity
(latent r = 0.945, see `Collinearity Analysis + Skill-Definition Options.md`) — whether
**adaptability** should become a modeled skill dimension.

**TL;DR:** *Adaptability* in TutorBench is a **use case (task type)**, not a latent skill, and
it is the **single most-represented use case in our calibration bank** (`adaptive_explanation`
≈ 50%). But as a *skill*, adaptability is **not separable in our current rubric bank** — its
criteria (analogies, relatable/accessible examples, tailoring to the student's stated
confusion) are already absorbed into **scaffolding** (often + content, sometimes + diagnosis).
**Recommendation: do NOT add adaptability as a 4th latent dimension now.** Use `use_case` as a
**stratification/reporting facet** instead, and only reconsider a dedicated dimension if the
full-matrix EFA (post content+diagnosis collapse) surfaces a residual factor on the
adaptive-explanation / analogy criteria. All estimates below are caveated.

---

## 1. The three TutorBench use cases (confirmed, with sources)

TutorBench draws its samples from **three tutoring tasks / use cases** (identical wording
across the dataset card, the paper §2.1, and the Scale blog):

| # | Use case (paper name) | What it tests | Our local `use_case` value |
|---|---|---|---|
| 1 | **Adaptive Explanation Generation** ("adaptability" / "Adaptive") | Personalized instruction: identify the student's core misconceptions/knowledge gaps from a follow-up question and adapt the explanation so it is easy for *this* student to understand. | `adaptive_explanation` |
| 2 | **Assessment and Feedback** ("Assessment") | Shown a student's (often incorrect) solution; analyze the work, identify and *classify* mistakes (arithmetic/factual/conceptual…), and give corrective feedback. Many upstream samples are multimodal (images of handwritten work). | `feedback` |
| 3 | **Active Learning Support** ("Active Learning") | Promote engagement without giving away the answer: generate hints / analogies / intermediate steps that let the student take the next step. | `hint_generation` |

**Sources:**
- Dataset card — `ScaleAI/TutorBench` on Hugging Face: *"examples drawn from three common tutoring tasks: (i) generating adaptive explanations tailored to a student's confusion, (ii) providing actionable feedback on a student's work, and (iii) promoting active learning through effective hint generation."*
- Paper — *TutorBench* (arXiv:2510.02663) §2.1 "Tutoring Use Cases": Use Case 1 Adaptive Explanation Generation, Use Case 2 Assessment and Feedback, Use Case 3 Active Learning Support. Abstract confirms the same three.
- Scale AI blog (`scale.com/blog/tutorbench`): "Adaptive Explanation Generation," "Feedback and Assessment," "Active Learning Support."
- Local corroboration: `plans+prds/Adaptive Testing + Tiny Benchmark Research Dump.md` ("Three use cases:" Adaptive Explanation / Assessment and feedback / Active Learning Support); `plans+prds/TutorBench Multimodal Transcription Plan.md` and `TutorEval Integration Spec.md` both name the local vocabulary **`adaptive_explanation` / `feedback` / `hint_generation`**.

> The user's recollection was essentially right: (i) help/hint = **Active Learning Support**,
> (ii) assessing student work = **Assessment and Feedback**, (iii) adaptability/personalization
> = **Adaptive Explanation Generation**. Note "adaptability" is the paper's *framing word* for
> use case 1 (the abstract lists "be adaptive / provide personalized guidance" as a goal); the
> official use-case name is **Adaptive Explanation Generation**.

---

## 2. Distribution of our scenarios across the three use cases

Our calibration bank is **662 scenarios / 6,462 criteria, text-only** (the text half of
TutorBench; the 817 multimodal items are un-ingested — see the Multimodal Transcription Plan).
Each scenario in `data/scenarios.jsonl` carries an explicit `use_case` field, so this is a
**hard field, not an inference**.

| Use case (`use_case`) | Scenarios | Share (of 662) |
|---|---:|---:|
| `adaptive_explanation` | **329** | ≈ 49.7% |
| `feedback` | **167** | ≈ 25.2% |
| `hint_generation` | **166** | ≈ 25.1% |
| **Total** | **662** | 100% |

- **Adaptability IS represented — in fact it is the plurality use case (~half the bank).** Every
  one of the ~329 `adaptive_explanation` scenarios is an adaptivity/personalization task.
- Confirmed present in the local file: a `use_case` grep matches `feedback` and
  `hint_generation` rows as well as `adaptive_explanation`.

> ⚠️ **Caveat on the exact counts.** The 329 / 167 / 166 split is the number recorded in
> `TutorBench Multimodal Transcription Plan.md` (measured against the source parquet; it also
> matches the upstream `USE_CASE_*_TEXT` batch sizes 327/165/164). I could **not independently
> re-count** them here because the search tool's `count` mode caps at 100 (a known issue also
> flagged in the collinearity memo), and the shell in this environment returns no exit status,
> so I cannot execute code. **To verify exactly**, run:
>
> ```bash
> python - <<'PY'
> import json, collections
> c = collections.Counter(json.loads(l)["use_case"] for l in open("data/scenarios.jsonl", encoding="utf-8"))
> print(dict(c), "total", sum(c.values()))
> PY
> ```

---

## 3. Crosswalk: TutorBench use cases ↔ our three skills

**Key framing distinction.** TutorBench **use cases are task types** (what the scenario asks
for). Our **skills (content / diagnosis / scaffolding) are cross-cutting latent abilities** a
response exercises. They are not the same axis: a single use case exercises *several* skills,
and each skill appears across *all* use cases. TutorBench itself does **not** treat its three
use cases as latent skills — it tags each rubric criterion with its own per-criterion
`tutoring_skill` / `eval_dimension` attributes (e.g. "student_level_calibration"), which our
final bank does **not** retain (our `q_mapping` is only content/diagnosis/scaffolding).

| TutorBench use case | content | diagnosis | scaffolding | adaptability (unmodeled) | Notes |
|---|:--:|:--:|:--:|:--:|---|
| **Adaptive Explanation Generation** | ✓ (explanation must be correct) | ✓✓ (read the student's specific confusion) | ✓ (accessible framing, analogy, "explain not state") | ✓✓ **(this is where adaptability lives)** | The personalization task. In our bank its criteria distribute mainly to **scaffolding + diagnosis (+ content)**. |
| **Assessment and Feedback** | ✓✓ (know the correct answer) | ✓✓ (identify & classify the mistake) | ~ (feedback framing, optional) | ~ | Strongly a **content + diagnosis** task — exactly the pair that is collinear (r = 0.945). |
| **Active Learning Support (hints)** | ✓ (the hint must be correct) | ✓ (target where the student is stuck) | ✓✓ (withhold the answer; hint/step form) | ~ | Strongly a **scaffolding (+ content/diagnosis)** task. |

Legend: ✓✓ central, ✓ commonly required, ~ sometimes/optional.

**Where does "adaptability" sit in the rubric bank?** A sample read of criteria carrying
adaptivity/personalization language (analogies, "relatable/everyday example", "simpler /
not too technical", "tailored to the student's stated confusion") shows they are **almost
always mapped to `scaffolding`** as `primary_skill`, frequently co-loading `content`, and
occasionally `diagnosis` when the criterion is tied to the student's specific confusion.
Examples: `tb_0011_c07` (distinct relatable analogy → content+scaffolding), `tb_0037_c02`
/ `tb_0040_c05` (relatable analogy → scaffolding-only), `tb_0021_c03` (example tailored to
the student's confusion → content+diagnosis+scaffolding), `tb_0006_c06` (honor the student's
"explain the concept, not the formula" constraint → diagnosis+scaffolding).

⇒ **Adaptability is currently an *absorbed* construct, not an unmodeled orphan.** Our
instrument already scores it — it just books it under **scaffolding** (personalization of the
*form* of help) and **diagnosis** (reading *this* student). It is **not separately
identified**, and it was **never tagged as its own axis** in the final bank.

> ⚠️ **Caveat:** this is a keyword/sample characterization over free text, not an adjudicated
> re-labeling. The exact scaffolding/diagnosis split of adaptivity criteria would need the
> per-item audit; treat the "mostly scaffolding" claim as directional.

---

## 4. Historical note: adaptability was once a proposed 4th skill

`plans+prds/Implementation Strategy + Success Metrics.md` originally proposed a **4-skill**
MIRT vector — **Content, Diagnosis, Scaffolding, Adaptability** — where *Adaptability* =
"change explanations and strategies given prior work, stated confusions, unsuccessful prior
explanations to ensure personalized learning." That 4th dimension was **dropped** on the way
to the canonical **v2 three-skill** taxonomy (`Skill Definitions v2`): the final
`q_mapping` schema is `{content, diagnosis, scaffolding}` only, verified across all 6,462
criteria in `data/rubrics_qmatrix_final.jsonl`. So adaptability is not a new idea — it was
considered and folded (mostly into scaffolding) before calibration.

---

## 5. Implications given the collinearity finding

From `Collinearity Analysis + Skill-Definition Options.md` (§6–7): on the ~1/4 matrix
(non-identifiable, N≈28), **content ↔ diagnosis latent r = 0.945** (likely collapse to one
"correctness/diagnosis" dimension) while **scaffolding is distinct** (r ≈ −0.19). If that
holds on the full matrix, the empirical latent structure is **~2 dimensions**:

1. **Correctness / diagnosis** (was content + diagnosis)
2. **Pedagogy / scaffolding**

**Does adaptability belong as a *replacement* 3rd dimension?** Tempting — it would move our
taxonomy closer to TutorBench's own three use cases (adaptive / assessment / active-learning ≈
adaptivity / correctness+diagnosis / scaffolding). But three cautions:

- **Adaptability is entangled with the very dimensions it would sit beside.** Per §3, our
  adaptivity criteria already load **scaffolding** (form of help) and **diagnosis** (reading
  the student). A dedicated "adaptability" axis would very likely be **more** collinear with
  scaffolding+diagnosis than content/diagnosis are with each other — reintroducing the exact
  identifiability problem we're trying to shed. The false-1 asymmetry from `Skill Definitions
  v2` ("a spurious dimension is systemic and hard to detect at finite N") applies squarely.
- **A "collapse frees a slot" argument is not evidence.** Collapsing content+diagnosis does not
  create statistical room for adaptability unless the *data* show a residual factor loading on
  the adaptive-explanation criteria after the collapse. That is an empirical question for EFA,
  not a design choice.
- **Use case ≠ skill.** Even if we want to *report* adaptivity, the cleaner instrument is to
  keep skills as cross-cutting abilities and use `use_case` as a **stratifier** — i.e. report
  each skill's θ *within* the `adaptive_explanation` slice — which needs **no** new dimension,
  no re-labeling, and no re-fit beyond what we already run.

**Could the "right" taxonomy be closer to TutorBench's 3 use cases?** Possibly, but only if the
full-matrix EFA says so. The honest position: our skills and TutorBench's use cases are
**orthogonal descriptions** (ability vs. task). The benchmark's own design tags skills *per
criterion*, not per use case — which is the same stance as our Q-matrix. Reframing skills = use
cases would conflate the two.

**Cost of a redefinition (if ever justified):** low and bounded — **Q-matrix regen + verify
+ re-fit only**. Tutor responses **and** judge verdicts are **reused** (grading is
per-criterion pass/fail, independent of skill labels) — **no re-grade, no fleet re-run**. This
is the same reversible economics as the content/diagnosis collapse.

---

## 6. Options

- **Option A — Keep 3 skills; treat `use_case` as a reporting/stratification facet
  (RECOMMENDED).** Report each skill θ overall *and* sliced by `use_case`
  (adaptive / feedback / hint). Zero data mutation, zero new dimension, immediately answers
  "how adaptive is model X" without an identifiability gamble. Adaptivity stays booked under
  scaffolding+diagnosis, which the data already support.
- **Option B — Add `adaptability` as a 4th (or, post-collapse, 3rd) latent dimension.** Regen
  the Q-matrix to tag adaptivity criteria, re-fit. **High identifiability risk** (see §5), and
  the ~1/4-matrix evidence points the *opposite* way (dimensions are collapsing, not
  multiplying). Only pursue if EFA demands it.
- **Option C — Defer, but pre-register a test.** After the full-matrix content+diagnosis
  collapse (see collinearity memo §7), run EFA and check whether a residual factor loads on the
  adaptive-explanation / analogy criteria. If yes → revisit Option B *with evidence*; if no →
  adaptability is confirmed as absorbed and Option A stands.

---

## 7. Recommendation

**Do Option A now; hold Option C as the trigger for any future change.**

1. **Confirmed:** TutorBench's three use cases are Adaptive Explanation Generation, Assessment
   and Feedback, Active Learning Support (§1). Our data uses `adaptive_explanation` / `feedback`
   / `hint_generation`.
2. **Adaptability is well represented** in our calibration bank — it is the **plurality use
   case (≈329/662 ≈ 50%)** (§2, count caveated).
3. **Adaptability is already *modeled*, just not *isolated*:** its criteria load
   **scaffolding** (+ content/diagnosis) in the current bank (§3). It is **not** an unmodeled
   orphan dimension.
4. **Do not add it as a separate latent skill now.** Given content≈diagnosis is *collapsing*
   toward fewer dimensions and adaptability is entangled with scaffolding+diagnosis, a dedicated
   adaptability axis would likely be **even more collinear** and hurt identifiability (§5).
   Use `use_case` for stratified reporting instead.
5. **Revisit only on evidence:** after the full-matrix collapse, run EFA; add adaptability as a
   dimension **iff** a residual factor loads on the adaptive-explanation criteria. Cost if ever
   done: Q-matrix regen/verify + re-fit only — responses and judge verdicts reused, **no
   re-grade** — and fully reversible.

### Pre-registered test (run on the FULL matrix, after the content+diagnosis collapse)

```bash
python scripts/calibrate_mirt.py --collapse content,diagnosis --efa --estimate-latent-corr --matrix <FULL_MATRIX.csv>
```

**Decision rule for a dedicated adaptability dimension:** promote adaptability to its own axis
**iff** EFA (after collapse) shows a **distinct factor** whose top-loading items are
predominantly `adaptive_explanation` scenarios / analogy-personalization criteria, **and** that
factor's latent correlation with scaffolding is **< ~0.9**. Otherwise keep it absorbed in
scaffolding and report it via `use_case` stratification.

---

## Appendix — sources & verification

- **Use cases:** `ScaleAI/TutorBench` dataset card; paper arXiv:2510.02663 §2.1 & abstract;
  `scale.com/blog/tutorbench`; local `Adaptive Testing + Tiny Benchmark Research Dump.md`,
  `TutorBench Multimodal Transcription Plan.md`, `TutorEval Integration Spec.md`.
- **Distribution:** `data/scenarios.jsonl` (`use_case` field, 662 rows);
  counts 329/167/166 recorded in `TutorBench Multimodal Transcription Plan.md`. Verify with the
  Python snippet in §2 (search-tool `count` caps at 100; shell here returns no exit status).
- **Skill mapping / adaptability absorption:** `data/rubrics_qmatrix_final.jsonl`
  (`q_mapping` = content/diagnosis/scaffolding only; sample criteria in §3);
  `Skill Definitions v2 (Q-matrix + Judge).md`.
- **Historical 4th skill:** `Implementation Strategy + Success Metrics.md` (Content / Diagnosis
  / Scaffolding / Adaptability proposal, later reduced to 3).
- **Collinearity context:** `Collinearity Analysis + Skill-Definition Options.md` §6–7.
