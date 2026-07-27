# Tutor Skill Taxonomy — Theory + Candidate Dimensions

**Status:** conceptual / literature memo. **Changes nothing.** It does not touch
`data/scenarios.jsonl`, `data/rubrics_qmatrix_final.jsonl`, the `q_mapping`s, the skill
definitions in `Skill Definitions v2 (Q-matrix + Judge).md`, or any judge/calibration code.
It exists to (a) explain, from the teacher-knowledge and tutoring literature, **why our
`content` and `diagnosis` skills empirically overlap** (latent r = 0.945, directional, N≈28),
and (b) survey what **other** tutor skills the literature treats as genuinely distinguishable,
judging for each whether it would be *empirically separable* in our instrument or would simply
re-collide the way `adaptability` did.

**Companion memos (do not contradict):** `Collinearity Analysis + Skill-Definition Options.md`
(content↔diagnosis collapse; Option C off the table at 1.7%), `TutorBench Use Cases +
Adaptability.md` (adaptability = reporting facet, not a dimension), `Skill Definitions v2
(Q-matrix + Judge).md` (canonical definitions).

**Empirical-claim caveat.** All correlations/effect sizes from the literature are cited inline
with source. Our own numbers (r = 0.945; 1.7%; the marginals) are **directional** — the latent
correlation is from the ~1/4 matrix at **N≈28** (< the ~150 needed for a powered latent-structure
call) and our overlap figures are structural/heuristic (see the collinearity memo's caveats).
The literature is used to *interpret* our numbers, not to license a decision on its own.

---

## 0. TL;DR

- **Q1 (why content & diagnosis overlap):** Our `content` ≈ Shulman's **content knowledge (CK)**;
  our `diagnosis` is a **component of pedagogical content knowledge (PCK)** — "knowledge of
  students' (mis)conceptions" — which is *defined* as knowledge **about** content and is
  therefore built **on top of** CK. Three forces drive the overlap: (i) PCK is theoretically an
  "amalgam"/transformation of CK, and empirically CK–PCK correlate ~0.6–0.8, rising toward **1
  in high-expertise populations** (COACTIV: manifest r = 0.60; latent r *statistically
  indistinguishable from 1* among academic-track teachers — Krauss et al., 2008; Baumert et al.,
  2010); (ii) a **prerequisite/conditional-dependence** structure — diagnosing *this* student's
  *specific* error logically presupposes knowing the correct content, so the two co-vary
  regardless of definitional care; (iii) a **measurement confound** — our pass/fail criteria
  bundle "know the right answer" and "spot the student's specific error" into a single item, so
  one item cannot separate the two. **Verdict: the overlap is mostly FUNDAMENTAL, not fixable by
  definition.** Only ~1.7% of the overlap dissolves under a narrower diagnosis definition;
  definitional sharpening has a low ceiling. Our r = 0.945 is not an anomaly — it is what CK–PCK
  looks like measured on a *high-expertise, bundled-item* instrument, i.e. exactly the COACTIV
  "indistinguishable-from-1" regime.

- **Q2 (other separable skills):** The one skill the literature treats as robustly predictive
  **and** least dependent on domain content — and which our bank *already contains but does not
  model* — is **motivational/affective support** (encouragement, framing effort, reducing
  anxiety). In our bank these criteria are almost uniformly **all-zero** (`primary_skill: null`),
  so unlike `adaptability` (absorbed into scaffolding) motivation is a genuine **unmodeled
  orphan** sitting in the ~1,107 all-zero items. It is the strongest candidate for a genuinely
  orthogonal 3rd/4th axis. **Metacognitive/self-regulation support** is a plausible but weaker
  second: its criteria exist but are sparse and are currently mapped to **scaffolding**
  (reflection prompts = guiding questions), so it would likely **re-collide with scaffolding**
  the way adaptability did. Every other candidate (Socratic questioning, examples/analogies,
  feedback quality/timing, cognitive-load/pacing) is either *definitionally part of* scaffolding
  or *bundled with* content/diagnosis, and would not add a separable axis.

- **Recommendation:** Keep the just-decided plan — let the full-matrix fit **collapse
  content+diagnosis** to one "correctness/diagnosis" axis and keep **scaffolding** as the second;
  treat `adaptability` as a reporting facet, not a dimension. Do **not** add a new latent skill
  pre-emptively. **The single change worth pre-registering a test for is a `motivation/affect`
  axis**, because it is the only candidate that is (i) domain-independent, (ii) plentiful in the
  bank, and (iii) currently *unmodeled* rather than absorbed. Its cost, if ever adopted, is the
  same bounded/reversible economics as any taxonomy change: **Q-matrix regen + verify + re-fit
  only — tutor responses and judge verdicts reused, no re-grade.** Its likely ceiling is
  discussed in §2.4 (LLM tone may be low-variance/easy → weak discrimination).

---

## Q1 — Why do content knowledge and diagnosis overlap?

The user's intuition is that `content` (domain correctness) and `diagnosis` (identify the
student's specific misconception) *should* be more definitionally distinct than r = 0.945
suggests. The teacher-knowledge literature explains why that intuition, while reasonable, runs
into a well-documented wall: **diagnostic knowledge is not a sibling of content knowledge; it is
a child of it.**

### 1.1 The Shulman frame — where "diagnosis" sits

Shulman (1986, *Educational Researcher* 15(2):4–14; 1987, *Harvard Educational Review*
57(1):1–22) partitioned teacher knowledge into, among others:

- **Content knowledge (CK)** — the subject matter itself: facts, concepts, procedures, and
  *why* they are so ("the teacher need not only understand *that* something is so, the teacher
  must further understand *why* it is so", Shulman 1986, p. 9). **This is our `content`.**
- **General pedagogical knowledge (GPK)** — classroom management, generic instructional moves,
  broad principles independent of subject. (Roughly: our all-zero conversational/tone moves plus
  the domain-free part of scaffolding.)
- **Pedagogical content knowledge (PCK)** — the "amalgam of content and pedagogy" (Shulman 1987):
  the subject-specific knowledge of *how to teach this topic* — including "the conceptions and
  preconceptions that students … bring", "the most useful forms of representation … analogies,
  illustrations, examples", and **what makes specific topics easy or difficult, i.e. the
  misconceptions students are prone to.**

**`diagnosis` is a PCK component, specifically "knowledge of students' (mis)conceptions and
errors."** This placement is decisive: PCK is, by Shulman's own definition, knowledge *about the
content* — you cannot have "knowledge of the typical student error on quadratic factoring"
without having the content knowledge of quadratic factoring. Diagnosis is therefore **not a
skill that sits *beside* content; it is a skill that is *defined over* content.** That is the
first and most fundamental reason the two are not cleanly separable — the taxonomy itself nests
one inside the other.

(Our `scaffolding` straddles PCK's "representations/instructional moves" and GPK's generic
structuring; §2 shows why it separates cleanly from content while diagnosis does not.)

### 1.2 Empirical CK–PCK correlations — and why r = 0.945 is *expected*, not anomalous

If diagnosis (a PCK facet) were merely conceptually related to content, one might still expect
low-to-moderate empirical correlation. The teacher-knowledge measurement literature shows the
opposite:

- **COACTIV** (the large German study that built direct tests of secondary-math teachers' CK and
  PCK): the **manifest bivariate CK–PCK correlation was r = 0.60** (Krauss, Baumert, & Blum,
  2008, *ZDM* 40(5):873–892; Krauss et al., 2008, *J. Educational Psychology* 100(3):716–725).
  Crucially, **in the high-expertise subgroup (academic-track "Gymnasium" teachers), modeling CK
  and PCK as latent constructs yielded a latent correlation "no longer statistically
  distinguishable from 1"** (Krauss et al., 2008, as reported in the ZDM validation paper). In
  other words: *the more expert the population, the more CK and PCK fuse into a single
  dimension.* Baumert et al. (2010, *American Educational Research Journal* 47(1):133–180)
  nonetheless found CK and PCK "theoretically and empirically distinguishable" **and** that PCK
  (not raw CK) predicted student learning gains — so the constructs are *distinct in principle
  and in their consequences*, even when they are *nearly collinear in measurement*.

- **The "integration vs. transformation" debate about PCK** (Gess-Newsome, 1999, in *Examining
  Pedagogical Content Knowledge*): the *integrative* model treats PCK as CK + pedagogy held
  simultaneously (predicts high CK–PCK correlation and weak separability); the *transformative*
  model treats PCK as a genuinely new amalgam distinct from its ingredients (predicts
  separability). The empirical record (COACTIV, and reviews by Depaepe, Verschaffel, & Kelchtermans,
  2013, *Teaching and Teacher Education* 34:12–25) sits **between**: PCK is separable *sometimes*,
  and collapses into CK precisely when (a) the population is expert and (b) the instrument bundles
  the two. **Both conditions describe us.**

**Relating our r = 0.945 to this:** our graded fleet on the ~1/4 matrix is dominated by weak
models that fail broadly (the collinearity memo's guardrail), which inflates any correlation; and
our items *bundle* content and diagnosis (§1.4). So we are measuring CK–PCK on effectively an
"expert-regime + bundled-item" instrument — exactly the COACTIV academic-track condition where
the latent correlation goes to ≈1. **r = 0.945 is therefore consistent with, not contradictory
to, the finding that CK and PCK are "distinct in principle."** The literature's own strongest
instruments hit the same wall.

### 1.3 The prerequisite / hierarchical argument (conditional dependence)

Independent of any labeling choice, there is a **logical entailment**: to identify that *this*
student made *this specific* error, the tutor must (implicitly) know what the correct content is
— the error is defined as a deviation from the correct answer. Formally, diagnosis of a specific
error is **conditionally dependent** on content: P(correct diagnosis | wrong content) ≈ 0. A
response that gets the chemistry wrong cannot reliably flag the student's chemistry error as an
error. This produces **high positive correlation between the two abilities regardless of how
carefully we word the definitions** — it is a property of the *task*, not of the *rubric*. This
is the same structure the collinearity memo identified empirically as **near-nesting**:
P(content | diagnosis) = 81.9% but P(diagnosis | content) = 42.9% — diagnosis behaves like a
*sub-region* of content, which is exactly the footprint of a prerequisite relationship.

### 1.4 The measurement / operationalization confound

Even setting theory aside, our **instrument cannot separate what it bundles into one item.** A
representative diagnosis-loaded criterion — e.g. `tb_0047_c04` "correct the student's wrong
y-intercept term → a(V−b)/(RV²)" — requires the response to *both* (a) know the correct term
(content) *and* (b) pinpoint the student's specific wrong term (diagnosis), and it is scored
**pass/fail as a single unit.** A single dichotomous item loads on whatever mixture of latent
abilities its passage requires; it has **no way to attribute the pass to content vs. diagnosis.**
Psychometrically, to *identify* two correlated dimensions you need items that load them **purely
and separately** — pure-content items (which we have in abundance: 2,096 content-only) and
**pure-diagnosis items** (which we have few of: 399 diagnosis-only + 30 diagnosis+scaffolding =
429, and a chunk of those are borderline acknowledgement items near the all-zero line). With so
few "diagnose-without-needing-the-content" items, the dimension **rarely varies independently**,
and a latent-variable model with little independent variation returns a near-1 correlation. This
is a **confound of item design**, and it is the part of the overlap that is *in principle*
fixable — but see §1.5 for how little it buys.

### 1.5 Verdict — how much is FUNDAMENTAL vs. FIXABLE?

| Source of overlap | Mechanism | Fundamental or fixable? |
|---|---|---|
| PCK-nested-in-CK (Shulman) | Diagnosis is *defined over* content | **Fundamental** — cannot define away without mislabeling |
| Prerequisite / conditional dependence (§1.3) | Diagnosing a specific error presupposes the correct content | **Fundamental** — property of the task, not the rubric |
| Expertise-regime collinearity (§1.2) | CK–PCK fuse toward r≈1 in expert/bundled measurement | **Fundamental-ish** — inherent to measuring able tutors on bundled items |
| Bundled pass/fail items (§1.4) | One item can't attribute a pass to CK vs PCK | **Partly fixable** — but needs more pure-diagnosis items |
| Broad "address-the-error" labeling rule (v2.1) | Any "address the student's error" phrasing loads diagnosis | **Fixable, but tiny** — only **1.7%** of the overlap (collinearity memo §6.2) |

**Bottom line: the overlap is overwhelmingly FUNDAMENTAL.** The collinearity memo's exact finding
— that narrowing the `diagnosis` definition to "explicit specific-misconception only" would
dissolve diagnosis from just **~33 items (1.7%)** of the 1,943 content∩diagnosis items — is the
empirical fingerprint of this: there is almost no "labeling slack" to reclaim. The remaining ~98%
of the overlap is items that genuinely require both abilities, exactly as the prerequisite
argument predicts.

**Concrete definitional-sharpening options and their honest ceilings:**

1. **Narrow `diagnosis` to explicit specific-misconception naming only** (old Option C). *Ceiling:
   ~1.7% of overlap removed.* Re-opens the false-0 risk v2.1 deliberately closed. **Not worth it**
   — the collinearity memo already ruled this off the table.
2. **Author more pure-diagnosis items** (diagnose-without-needing-content: "which of the student's
   steps is the *first* wrong one?" style, where the correct answer is given). *Ceiling: could
   meaningfully lower the latent correlation* — this is the *only* lever that attacks the §1.4
   confound at its root — **but it is out of scope** (it changes scenarios/rubrics, which this
   analysis is forbidden from doing, and TutorBench items are fixed). It also fights the §1.3
   prerequisite structure, so even a pure-diagnosis item is only partly independent of content.
3. **Split `diagnosis` into "error detection" vs. "state acknowledgement"** (the collinearity
   memo's two sub-constructs). *Ceiling: sharpens meaning, does not reduce collinearity* — the
   detection sub-construct is the one that co-fires with content.

**So: definitional care cannot rescue separability here.** The reason is not sloppy definitions;
it is that diagnosis is a PCK facet defined over content, measured on bundled items in an
expert-ish regime. The right move is the one already decided: **collapse content+diagnosis into a
single "correctness/diagnosis" axis** and stop treating the overlap as a bug to be defined away.

---

## Q2 — What other tutor skills are genuinely distinguishable?

### 2.1 The bar for "separable"

A candidate dimension must clear **three** hurdles, not one:

1. **Conceptually distinct** (necessary, easy) — the literature names it as its own competency.
2. **Empirically separable** (necessary, hard) — it should vary *across responders* largely
   *independently* of content/diagnosis/scaffolding, i.e. a low latent correlation, not merely a
   different label. The `adaptability` case is the cautionary tale: conceptually distinct, but its
   criteria load scaffolding+diagnosis, so it re-collides.
3. **Present as *discriminating* items in our bank** — criteria must exist **and** be mapped to a
   loading that actually varies. A construct whose criteria are all mapped **all-zero** currently
   contributes *zero* discrimination (it is invisible to MIRT), and one whose criteria are all
   mapped to an existing skill is *absorbed* (invisible as a separate axis).

The decisive lens for hurdle 2 is **domain-dependence**: the reason content and diagnosis
collapse is that both are defined over the subject matter. A skill is a good bet for separability
**exactly to the extent that it does not depend on domain content** — because then it cannot be
dragged into the correctness/diagnosis axis. This is why the user's prior (motivation and
metacognition) is well-aimed: those are the least domain-bound skills in the tutoring literature.

### 2.2 Ranked candidate table

Presence-in-bank is from a keyword/sample probe of `data/rubrics_qmatrix_final.jsonl` (directional,
not an adjudicated re-label — the search tool's counts are unreliable, so treat magnitudes as
approximate). "Current mapping" = how such criteria are labeled *today* under v2.

| Rank | Candidate dimension | Definition | Literature evidence it's separable/predictive | In our bank? Current mapping | Empirically separable here, or re-collides? |
|---|---|---|---|---|---|
| **1** | **Motivational / affective support** | Encouragement, effort/growth framing, reducing anxiety & frustration, protecting confidence; managing the student's emotional state so they stay engaged. | Strong & largely **domain-independent**. Lepper's **INSPIRE** model of expert tutors (Lepper & Woolverton, 2002) makes motivation co-equal with cognition; **control-value theory of achievement emotions** (Pekrun, 2006, *Educ. Psych. Review* 18:315–341) shows emotions predict achievement partly independently of ability; affect is a first-class channel in AutoTutor/**Affective AutoTutor** (D'Mello & Graesser, 2012). Motivation is not reducible to being correct. | **Yes, abundant** — "encouraging/positive/warm/growth-oriented tone", "praise the student's insight", "not be discouraged". **Currently almost all `primary_skill: null` / all-zero** (part of the ~1,107 all-zero items). **Unmodeled orphan, not absorbed.** | **Most likely separable.** Loads *no* content by construction, so it cannot be dragged into the correctness/diagnosis axis. The main risk is *low variance/easy* (hurdle 2 via ceiling, not collinearity) — see §2.4. |
| **2** | **Metacognitive / self-regulation support** | Prompting the student to plan, monitor, self-check, and reflect on their own reasoning; building strategy awareness (SRL). | **Zimmerman** (2002, *Theory Into Practice* 41(2):64–70) SRL; strong meta-analytic support for metacognition/self-regulation interventions on achievement (e.g. Dignath & Büttner, 2008). Distinct construct, moderately domain-general. | **Yes, but sparse** — "prompt the student to reflect on their answer", "metacognitive learning tip", "reflect on the sign relationship". **Currently mapped to `scaffolding`** (± content), because a reflection prompt *is* a guiding question. | **Likely re-collides with scaffolding.** A "get the student to reflect/self-check" prompt is operationally a guiding question = the core of scaffolding. Same failure mode as adaptability. Separable only if the *self-regulation* content (plan/monitor) were isolated from the *question-form* — which our items don't do. |
| **3** | **Communication / explanation clarity** | Is the explanation clear, concise, well-organized, jargon-appropriate, readable — independent of whether it's correct or pedagogically structured. | Formative-feedback theory stresses clarity/comprehensibility (**Shute, 2008**, *Review of Educ. Research* 78(1):153–189); readability & coherence matter for learning from text. Distinct from correctness. | **Yes, plentiful** — "clear, student-friendly language", "concise", "avoid jargon". **Mixed:** often bundled *with* encouraging tone into the same **all-zero** criterion; sometimes overlaps scaffolding's "explain, not just state". | **Weak/ambiguous.** Domain-independent (good) but (a) frequently *co-authored with* the affective/tone bucket, so it may not separate from motivation, and (b) likely **low variance** — most competent LLMs are clear — so weak discrimination. Could ride along as part of a broader "delivery/affect" axis rather than its own. |
| 4 | **Socratic questioning / elicitation** | Draw the answer out via questions rather than telling. | Chi & Wylie **ICAP** (2014, *Educ. Psychologist* 49(4):219–243): interactive/constructive > passive; Graesser's **AutoTutor** Expectation–Misconception Tailored (EMT) dialogue is built on elicitation (Graesser et al., 2004). | **Yes** — "give a hint via a guiding question", "ask the student to consider…". **This *is* `scaffolding`** (the "guiding question / withhold the answer" form). | **Not separable — it's definitionally scaffolding.** Adding it would just re-label part of scaffolding. |
| 5 | **Use of examples / representations / analogies (adaptive explanation)** | Choose apt analogies, worked examples, multiple representations tailored to the student. | Core PCK facet (Shulman 1986); representational fluency predicts learning. | **Yes** — "relatable analogy", "everyday example". **Absorbed into `scaffolding` (+content/diagnosis)** — this is exactly `adaptability`. | **Not separable — already shown to re-collide** (see `TutorBench Use Cases + Adaptability.md`). Decided: reporting facet, not a dimension. |
| 6 | **Feedback quality / timing** | Right amount, right level, right time; feed-up/feed-back/feed-forward. | **Hattie & Timperley** (2007, *Review of Educ. Research* 77(1):81–112); Shute (2008). Strong effects, but the *construct* is about correctness+specificity+structure. | **Yes** — but "feedback" is a **use_case** (`feedback`), and quality criteria bundle **content** (right answer) + **diagnosis** (specific error) + **scaffolding** (delivery). | **Not separable — it decomposes *into* the existing axes.** High-quality feedback = correct + specific-to-the-error + well-structured; it *is* the content/diagnosis/scaffolding blend. |
| 7 | **Cognitive-load / pacing management** | Chunking, sequencing, not overwhelming; managing intrinsic/extraneous load. | Sweller cognitive-load theory; VanLehn's **interaction-granularity** result (2011, d≈0.76–0.79) is about step size/pacing. | **Sparse** — "break into steps", "don't overwhelm". **Mapped to `scaffolding`** (decomposition/sequencing *is* scaffolding). | **Not separable — it's a facet of scaffolding.** |

*Broader tutoring-effectiveness anchors cited above:* Bloom's **2-sigma** (1984, *Educational
Researcher* 13(6):4–16) motivates the whole enterprise (individual tutoring ≈ +2σ under mastery
conditions); VanLehn (2011, *Educ. Psychologist* 46(4):197–221) tempers it (human tutoring
d = 0.79 ≈ ITS d = 0.76), and locates much of the effect in **interaction granularity**
(scaffolding/pacing) rather than raw content delivery — consistent with scaffolding being our
cleanly-separable axis.

### 2.3 The 1–3 most promising, judged honestly

**#1 Motivational / affective support — the strongest genuinely-separable candidate.**
It clears all three hurdles better than any other: (1) conceptually distinct and literature-backed
(INSPIRE, control-value theory, Affective AutoTutor); (2) **structurally guaranteed low
domain-dependence** — its criteria load *no* content, so it *cannot* be pulled into the
correctness/diagnosis axis the way adaptability was pulled into scaffolding; and (3) it is
**already present in bulk** in the bank. The critical difference from adaptability: adaptability's
criteria were **absorbed** into an existing skill (scaffolding), whereas motivation's criteria are
currently **all-zero orphans** — we author them, keep them for critical-failure monitoring, and
then throw away their signal in MIRT. Promoting affect to a modeled dimension would *recover*
existing, discarded information rather than manufacture a new label.

**#2 Metacognitive / self-regulation support — plausible but likely re-collides.**
The construct is real and domain-general (Zimmerman SRL), *but* our operationalization of it is a
**reflection/self-check prompt**, which is structurally a guiding question — i.e. **scaffolding**.
Our bank already maps these to scaffolding. So on our instrument metacognition would most likely
behave like a high-correlation sibling of scaffolding, repeating the adaptability failure. It
becomes separable only if items isolated the *self-regulation object* (plan/monitor your own
process) from the *question form* — which they do not. **Downgrade the user's prior here:
metacognition is conceptually strong but empirically weak *in our instrument*.**

**#3 Communication clarity — real but probably not its own axis.**
Domain-independent (good) but (a) often bundled with affective tone in the same all-zero criterion
and (b) likely **low-variance** across capable models. More realistic as *part of* a combined
"delivery/affect" axis than as a standalone dimension.

### 2.4 The honest ceiling on a `motivation/affect` axis

Even the strongest candidate has a real risk — and it is **not** collinearity, it is **variance**:

- Affective-tone criteria may be **easy and near-all-pass** for capable models (most frontier LLMs
  produce encouraging tone effortlessly), giving the dimension **low discrimination** — a
  different way to fail hurdle 2. On our current fleet (weak small models), tone may vary more, but
  on a strong fleet it could flatten.
- These items are currently `primary_skill: null` **by design** (v2 treats tone/affect/empathy as
  all-zero). Modeling motivation means **re-mapping** a chunk of the ~1,107 all-zero items to a new
  loading — a deliberate reversal of a v2 decision, and a definitional judgment (which affective
  criteria are "support" vs. mere "tone"?). That is a real design cost, not just a re-fit.

So the fair verdict on motivation is: **the most likely-to-be-orthogonal addition we have, and the
only one recovering discarded signal — but its payoff is capped by whether affective competence
actually *varies* across the models we score.** That is an empirical question for EFA, not a
foregone win.

---

## Recommendation

**Keep the just-decided 2-dimensional plan; do not add a latent skill now; pre-register one test
for a `motivation/affect` axis.**

1. **Stay the course on content+diagnosis.** Q1 shows their overlap is fundamental (PCK-nested-in-CK
   + prerequisite structure + expert-regime bundled measurement), not a labeling artifact. Let the
   full-matrix fit collapse them into one **correctness/diagnosis** axis if the pre-registered rule
   fires (latent r ≳ 0.9 **and** collapsed-2-dim wins AIC/BIC **and** EFA finds no separate factor;
   see collinearity memo §7). **Scaffolding stays a distinct axis** (it separates cleanly, r ≈ −0.19).
   This yields the likely **2-skill** structure: *correctness/diagnosis* + *pedagogy/scaffolding*.

2. **Adaptability = reporting facet, not a dimension** — unchanged from the use-cases memo. It is
   absorbed into scaffolding+diagnosis; report it via `use_case` stratification.

3. **The one well-supported case to ADD a genuinely orthogonal dimension is `motivation/affect`
   — but only on evidence.** It is the sole candidate that is domain-independent, plentiful, and
   *unmodeled* (not absorbed). Do **not** add it pre-emptively (the false-1 asymmetry in
   `Skill Definitions v2` — a spurious dimension is systemic and hard to detect at finite N —
   applies). Instead, **pre-register an EFA test**: after the content+diagnosis collapse, check
   whether a residual factor loads on the affective/encouragement (currently all-zero) criteria,
   and whether that factor has meaningful item variance and a latent correlation < ~0.9 with the
   other two axes. If yes → add `motivation` (likely as the taxonomy's *third* axis in a 3-skill
   model: correctness/diagnosis + scaffolding + motivation). If no (e.g. affect is near-all-pass /
   low-variance) → it stays all-zero and is reported descriptively.

   > Pre-registered probe (illustrative — mirrors the adaptability trigger in the use-cases memo;
   > would require first defining a candidate `motivation` loading in a *throwaway* Q-matrix regen,
   > not touching the canonical bank):
   > `python scripts/calibrate_mirt.py --collapse content,diagnosis --efa --estimate-latent-corr --matrix <FULL_MATRIX.csv>`
   > then inspect whether the top-loading items on any residual factor are the affective/encouragement
   > criteria.

4. **Metacognition and clarity: hold.** Both are conceptually attractive but our items map them
   into scaffolding / tone respectively, so they would most likely re-collide. Revisit only if the
   motivation EFA also surfaces a distinct metacognitive factor (unlikely given current mappings).

**Cost of any taxonomy change (all options above).** Bounded and reversible, identical to the
economics in the companion memos: **Q-matrix regeneration + verification + a calibration re-fit
only.** Tutor responses **and** judge verdicts are **reused** (grading is per-criterion pass/fail,
independent of skill labels) — **no re-grade, no fleet re-run.** A `motivation` axis additionally
requires a **one-time re-mapping decision** for the affective all-zero items (which count as
"support" vs. mere "tone"), which is a definitional judgment but still cheap to execute and fully
reversible.

---

## Appendix — sources

- **Shulman, L. S.** (1986). Those who understand: Knowledge growth in teaching. *Educational
  Researcher* 15(2):4–14. — (1987). Knowledge and teaching. *Harvard Educational Review* 57(1):1–22.
- **Krauss, S., Baumert, J., & Blum, W.** (2008). Secondary mathematics teachers' PCK and CK:
  validation of the COACTIV constructs. *ZDM* 40(5):873–892. (Manifest CK–PCK r = 0.60; latent
  ≈ 1 in academic-track group.) **Krauss et al.** (2008). *J. Educational Psychology* 100(3):716–725.
- **Baumert, J., Kunter, M., et al.** (2010). Teachers' math knowledge, cognitive activation, and
  student progress. *American Educational Research Journal* 47(1):133–180. (CK & PCK distinguishable;
  PCK predicts learning gains.)
- **Gess-Newsome, J.** (1999). Pedagogical content knowledge: an introduction and orientation
  (integration vs. transformation models). — **Depaepe, Verschaffel, & Kelchtermans** (2013).
  *Teaching and Teacher Education* 34:12–25 (PCK review).
- **VanLehn, K.** (2011). The relative effectiveness of human tutoring, ITS, and other tutoring
  systems. *Educational Psychologist* 46(4):197–221. (Human d = 0.79 ≈ ITS d = 0.76;
  interaction-granularity.)
- **Bloom, B. S.** (1984). The 2 sigma problem. *Educational Researcher* 13(6):4–16.
- **Chi, M. T. H., & Wylie, R.** (2014). The ICAP framework. *Educational Psychologist*
  49(4):219–243.
- **Graesser, A. C., et al.** (2004). AutoTutor (Expectation–Misconception Tailored dialogue).
  *Behavior Research Methods* 36:180–192. — **D'Mello, S., & Graesser, A.** (2012). Affective
  AutoTutor.
- **Lepper, M. R., & Woolverton, M.** (2002). The wisdom of practice (INSPIRE model of expert
  tutors), in *Improving Academic Achievement*.
- **Pekrun, R.** (2006). Control-value theory of achievement emotions. *Educational Psychology
  Review* 18:315–341.
- **Zimmerman, B. J.** (2002). Becoming a self-regulated learner. *Theory Into Practice* 41(2):64–70.
  — **Dignath, C., & Büttner, G.** (2008). Meta-analysis of self-regulated learning interventions.
- **Hattie, J., & Timperley, H.** (2007). The power of feedback. *Review of Educational Research*
  77(1):81–112. — **Shute, V. J.** (2008). Focus on formative feedback. *Review of Educational
  Research* 78(1):153–189.
- **Local:** `Skill Definitions v2 (Q-matrix + Judge).md`; `Collinearity Analysis + Skill-Definition
  Options.md` (r = 0.945, N≈28; 1.7%; marginals); `TutorBench Use Cases + Adaptability.md`;
  `data/rubrics_qmatrix_final.jsonl` (affective criteria → all-zero; metacognitive/reflection
  criteria → scaffolding, sampled).

---

## Orphan-criteria probe: affect vs communication vs scaffolding

**Status:** analysis addendum. **Changes nothing** — no scenario, rubric, `q_mapping`, or skill
definition is touched; no dataset was downloaded. It refereess one specific question left open
by §2.3/§2.4: is the currently-**unmodeled** tutor signal best described as a **narrow
`motivation/affect`** axis, a **broad `communication/delivery`** axis (clarity + tone +
encouragement + organization), or **neither** — and would either stay distinct from
`scaffolding` or re-collide with it (as content↔diagnosis collapsed at latent r = 0.945 and
adaptability re-collided with scaffolding)?

**Method.** The "unmodeled" pool is the two Q-patterns that contribute **zero discrimination**
to the current 3-skill MIRT: the **all-zero (0,0,0)** criteria (**1,107**, per the collinearity
memo's exact structural count) and the **scaffolding-only (0,0,1)** criteria (**402**). A new
read-only script, `scripts/classify_orphan_criteria.py`, partitions the whole bank by exact
Q-pattern and runs a **transparent keyword/regex classifier** over each `criterion` text into six
candidate buckets — `affect_motivation`, `communication_clarity`, `organization_structure`,
`metacognitive`, `socratic_questioning`, `other_uncertain` — emitting exact per-bucket counts
(overall / within-all-zero / within-scaffolding-only), per-item labels for hand audit, and a
directional variance check. **Run:**

```bash
python scripts/classify_orphan_criteria.py
# structural/bucket counts only (skip the N≈28 variance block):
python scripts/classify_orphan_criteria.py --structural-only
```

Outputs: `staging/orphan_criteria.json`, `staging/orphan_criteria.csv` (per-item labels), plus a
printed summary. **⚠️ Every count below is a keyword HEURISTIC over free text and needs a hand
audit** (read the CSV); the exact machine counts come from the script. The magnitudes here are
what a **stratified sample read** of the two pools supports, quoted as ranges.

### A. What is actually in the two unmodeled pools (heuristic, read-estimated)

**All-zero pool (1,107).** From a stratified sample read it is dominated by two clean, mutually
distinct constructs, plus a thin tail:

| Bucket (within all-zero) | Read-estimated share | Read-estimated count | Character |
|---|---:|---:|---|
| `affect_motivation` | **~45–55%** | ~500–600 | "encouraging/positive/empathetic tone", "acknowledge the student's confusion/frustration", "praise the student's insight". Clean, domain-free, plentiful. |
| `organization_structure` | **~30–40%** | ~330–440 | "headings/bold/bullets/Markdown/LaTeX", "format the code in code blocks", "clearly organized using section headings". Pure presentation. |
| `communication_clarity` (standalone) | **~5–10%** | ~55–110 | "concise, not verbose", "clear, student-friendly language". **Rarely standalone — frequently co-authored with `affect` in the same criterion** (e.g. `tb_0012_c06` "clear, student-friendly language **and** an encouraging tone"). |
| `other_uncertain` / conversational | **~5–10%** | ~55–110 | "invite further questions if needed", scope constraints ("avoid mentioning Experiment 3"), "check for understanding". |
| `metacognitive` / `socratic_questioning` | **small** | few | Mostly live in the scaffolding-only pool, not here. |

**Scaffolding-only pool (402).** Sampling shows this pool is, as expected, **the delivery/
questioning construct already modeled as `scaffolding`**: `socratic_questioning` items ("ask a
guiding question", "prompt the student to consider…", "invite the student to try/verify") are the
plurality, and `metacognitive` prompts ("invite a quick self-check", "encourage reflection on
their understanding", "invite the student to rerun/unit-test") are a substantial second — **both
currently labeled scaffolding.** Pure `affect`/`organization` are a small minority here.

### B. Qualitative distinctness judgments (the referee call)

**(a) How much of the all-zero pool is genuinely affect vs communication vs junk?**
The pool is **mostly a clean two-way split: affect/tone (~half) and formatting/organization
(~a third).** Genuine standalone *communication-clarity* ("concise", "plain language",
"readable") is a **thin slice (~5–10%)**, and roughly half of even that is **co-authored with
affect in the same criterion**, so it does not stand on its own as a construct. There is real
near-duplicate boilerplate ("use an encouraging tone", "acknowledge the student's confusion"
recur near-verbatim across scenarios), which matters for variance (§C).

**(b) Is "communication/clarity" DISTINCT from scaffolding, or the same delivery construct —
would a broad `communication` axis re-collide with scaffolding?**
**A broad `communication/delivery` axis would re-collide with scaffolding.** The reason is
structural, not incidental: once you widen "communication" past *pure formatting* to include the
things that make communication *pedagogically* effective — engaging the student, asking rather
than telling, prompting reflection, inviting a self-check — **you are describing the items the
bank already labels `scaffolding`.** The evidence:
- Guiding-question / elicitation items ("ask a guiding question", "prompt the student to
  consider…", `tb_0513_c04`, `tb_0517_c04`) are **scaffolding**, not all-zero.
- Reflection / self-check prompts ("invite a quick self-check" `tb_0309_c08`; "conclude with a
  reflective question" `tb_0136_c10`; "invite the student to rerun/unit-test" `tb_0448_c07`) are
  **scaffolding**.
- The all-zero/scaffolding boundary is **already fuzzy for engagement moves**: near-identical
  "invite the student to ask follow-up questions" is scaffolding in one place (`tb_0042_c09`) and
  all-zero in another (`tb_0111_c05`). A broad communication axis would sweep exactly this
  contested strip **into** its column — i.e. it would load the same items as scaffolding and drive
  the two latent axes toward collinearity. This is the adaptability failure mode again: a
  conceptually-broad label whose operational items are already someone else's dimension.

  What is *cleanly* non-scaffolding in the "communication" family is only the **residue** —
  pure visual **formatting** and bare **concision/clarity-of-language**. That residue is (i)
  small, (ii) not a coherent "communication competency" (formatting ≠ clarity ≠ tone), and (iii)
  low-variance (§C). So the broad axis either **re-collides with scaffolding** (if it includes the
  engagement/questioning content) or **collapses to low-signal formatting** (if it excludes it).
  Neither is a viable third dimension.

**(c) Do affect items have real variance, or are they near-all-"be nice" low-signal?**
Affect is the **one** unmodeled construct that is both plentiful **and** structurally
domain-free (it loads no content, so — unlike a broad communication axis — it *cannot* be dragged
into either the correctness/diagnosis axis **or** scaffolding). Its risk is **not collinearity, it
is variance** (§2.4): much of it is easy "be encouraging / acknowledge the confusion" boilerplate
that capable models pass trivially. Whether it has enough spread across responders to identify a
latent axis is exactly the empirical question the variance check flags and the full-matrix EFA
must settle — it is **not** answerable by reading.

### C. Variance flags (DIRECTIONAL, N≈28 — do not over-read)

`classify_orphan_criteria.py` computes, per bucket, each model's pass rate on that bucket's
gradeable items and then the mean / std / variance **across the ~28 graded models**, flagging
`near_ceiling` (mean ≥ 0.9), `near_floor`, or `low_variance` (std < 0.1). **⚠️ N≈28 on a biased
small-model subsample — directional only; the flags identify *weak-discrimination* buckets, they
decide nothing.** Read the emitted numbers from `staging/orphan_criteria.json`; the *a priori*
expectation (to be confirmed, not asserted) is:
- `organization_structure` and `communication_clarity`: **likely near-ceiling / low-variance** —
  formatting and concision are easy for most models ⇒ weak discrimination even before the strong
  fleet is added. This is a second, independent reason the broad-communication axis is unattractive.
- `affect_motivation`: **the bucket most likely to show real spread on the current weak fleet**
  (small models genuinely vary on tone), but §2.4's ceiling risk means this spread may flatten on
  a strong fleet — hence the axis is *evidence-gated*, not a foregone win.

### D. Recommendation and the pre-registered full-matrix test

**Recommendation: pursue the NARROW `affect/motivation` axis on evidence; do NOT adopt a broad
`communication/delivery` axis; and add nothing pre-emptively.** A broad communication axis fails
the separability bar because its pedagogically-active content (questioning, elicitation,
reflection, engagement prompts) **is already `scaffolding`** and would re-collide with it, while
its clean remainder (formatting + concision) is incoherent and low-variance. Narrow affect remains
the single candidate that is domain-free, plentiful, and genuinely *unmodeled* (not absorbed) —
unchanged from §2.3/Recommendation §3. This is a referee call, not a taxonomy change: it stays
consistent with the decided plan (content+diagnosis collapse; adaptability = reporting facet;
affect = strongest orphan candidate).

**Pre-registered EFA/MIRT test (decide among A = add narrow affect / B = add broad communication /
C = add neither).** On the FULL graded matrix, after the content+diagnosis collapse:

```bash
python scripts/calibrate_mirt.py --collapse content,diagnosis --efa --estimate-latent-corr \
    --matrix <FULL_MATRIX.csv>
```

Then, using `staging/orphan_criteria.csv` to identify each bucket's items, run a throwaway
Q-matrix regen that assigns a candidate loading to the bucket in question and re-fit. **Decision
rule** (mirrors the collinearity memo §7 collapse rule and the §2.4 affect trigger):

- **Adopt (A) narrow `affect` IFF** a residual EFA factor loads **coherently on the
  `affect_motivation` items**, that factor has **meaningful item variance** (not near-ceiling on
  the full fleet), **AND** its latent correlation is **< ~0.9 with BOTH** the correctness/diagnosis
  axis **and** scaffolding.
- **Adopt (B) broad `communication` ONLY IF** a residual factor loads coherently across the
  *combined* affect+communication+organization items **AND** — the load-bearing extra test —
  its latent correlation is **< ~0.9 with `scaffolding` specifically** (not just with
  correctness). Given §B, expect this test to **fail on the scaffolding collinearity**: the broad
  axis should show high latent r with scaffolding because it re-loads the same
  questioning/engagement items. **A broad communication axis must be tested for collinearity WITH
  scaffolding, not merely with correctness.**
- **Otherwise (C) add neither** — keep affect/organization/communication as descriptive all-zero
  items reported via `use_case`/critical-failure monitoring, exactly as today.

**Deliverables of this probe:** `scripts/classify_orphan_criteria.py` (+ run command above);
`staging/orphan_criteria.{json,csv}` (exact counts + auditable per-item labels). Every count and
variance figure is a heuristic or underpowered (N≈28) and must be hand-audited / deferred to the
full-matrix fit before it is quoted as fact.
