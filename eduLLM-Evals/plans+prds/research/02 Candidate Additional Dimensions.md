# 02 — Candidate Additional Tutor-Skill Dimensions (beyond content / diagnosis / scaffolding)

**Status:** literature-research memo (thread 2 of 3). **Changes nothing** — it does not touch
`data/scenarios.jsonl`, `data/rubrics_qmatrix_final.jsonl`, the `q_mapping`s, the skill
definitions, or any calibration/judge code. It exists to rigorously assess **candidate
additional tutor-skill dimensions** for a possible 4th (or, post content+diagnosis collapse,
3rd) latent axis in the multidimensional-IRT model, using real citable sources.

**Scope of the modeled taxonomy today:** three latent skills — **content** (domain
correctness), **diagnosis** (identifying the student's specific error/confusion), and
**scaffolding** (structuring/sequencing pedagogical help). The companion collinearity work
finds content↔diagnosis nearly fused (directional latent r = 0.945, N≈28) and likely to collapse
into one "correctness/diagnosis" axis, while scaffolding separates cleanly (r ≈ −0.19). The live
question for *this* memo: **what, if anything, should be the axis that sits beside scaffolding as
a genuinely separate, measurable, domain-independent tutor competency?**

**Internal findings taken as given** (from the four companion memos): motivation/affect is the
strongest candidate additional dimension (domain-independent, plentiful in the ~1,107 "all-zero"
criteria pool, currently *unmodeled* rather than absorbed); metacognition and communication-clarity
likely re-collide with scaffolding or are low-variance; presentation is a surface/style construct
(separable but non-gating — treated in memo 01/H1, not re-litigated here).

**Citation policy.** Every literature claim below is tagged with author, year, venue, and a link,
and was verified by web lookup during this research pass **unless explicitly flagged
"(unverified — from companion memo)"**. Our own numbers (r = 0.945; the marginals; N≈28) are
directional and underpowered; the literature is used to *interpret*, not to *decide*.

---

## (a) Executive summary — candidates ranked by strength of evidence as a SEPARATE, measurable, domain-independent tutor dimension

Ranking criterion = the conjunction the companion memos require: (1) a named, well-defined
construct; (2) **domain-independence** (does not load on subject-matter correctness, so it cannot
be dragged into the correctness/diagnosis axis); (3) evidence it has been **measured as a distinct,
reliable factor separable from instructional support/scaffolding**; and (4) presence as
*discriminating* signal in our bank.

| Rank | Candidate | Separable from scaffolding? | One-line verdict |
|---|---|---|---|
| **1** | **Motivational / affective support** (encouragement, effort/growth framing, anxiety/frustration reduction, autonomy/competence/relatedness support) | **YES** | The strongest case: a construct **directly validated as a factor distinct from instructional support** by the CLASS "Emotional Support" domain (CFA over 4,000+ classrooms; meta-analysis of 26 matrices), domain-independent by construction, and a plentiful *unmodeled orphan* in our bank. **Pre-register this as the 4th dimension.** |
| **2** | **Metacognition / self-regulated-learning support** (prompting planning, monitoring, self-checking, reflection) | **PARTLY / mostly NO** | A real, domain-general construct (Zimmerman SRL; Azevedo's explicit *metacognitive* vs *cognitive/content* scaffold distinction). BUT its operational form in our bank is a reflection/self-check *prompt* = a guiding question = scaffolding. Literature treats it as a **subtype of scaffolding** ("process/metacognitive scaffolding"), so on our instrument it most likely re-collides. |
| **3** | **Communication / explanation clarity** (clear, concise, well-organized, jargon-appropriate delivery) | **PARTLY** | A validated, domain-independent construct with a large meta-analysis (Titsworth et al. 2015). But in our bank it is (i) frequently co-authored with affective tone into the same all-zero criterion and (ii) likely low-variance across capable models ⇒ weak discrimination. More realistic as part of a broader delivery/affect axis than a standalone dimension. |
| 4 | **Contingency / adaptivity** (tailoring support to the student's current level; fading) | **NO** | The literature (van de Pol et al. 2010) names contingency as *the* core defining feature *of* scaffolding — not a separate axis. Confirms the companion adaptability memo: absorbed into scaffolding+diagnosis. |
| 5 | **Questioning / Socratic elicitation** (draw the answer out via questions) | **NO** | Definitionally the "guiding-question / withhold-the-answer" form of scaffolding (INSPIRE's "S"; AutoTutor EMT). Re-labels part of scaffolding. |
| 6 | **Feedback quality** (Hattie & Timperley feed-up/back/forward × task/process/self-reg/self levels) | **NO** (decomposes) | High-quality feedback = correct (content) + specific-to-the-error (diagnosis) + well-structured/process-level (scaffolding). It **decomposes into the existing axes** rather than adding one; note its self-regulation level *is* the metacognition candidate and its "self/affect" level *is* the motivation candidate. |

**Single strongest recommendation:** **pre-register `motivation/affect` as the candidate 4th
dimension for the full-matrix EFA.** It is the only candidate that (i) has been *directly
validated as a factor statistically separable from instructional support* (CLASS Emotional
Support), (ii) is domain-independent by construction so it cannot fuse into the
correctness/diagnosis axis, and (iii) already exists in bulk in our bank as discarded (all-zero)
signal we would be *recovering*, not manufacturing. Its one real risk is **low variance** (capable
models may all be encouraging), not collinearity — an empirical question for the powered EFA.

---

## (b) Per-candidate synthesis with citations and separability verdicts

### 1. Motivational / affective support — **separable from scaffolding? YES**

**Construct definition.** Support directed at the student's *motivational and emotional state*
rather than at the correctness or structure of the domain content: encouragement, framing effort
and mistakes as productive, protecting confidence, reducing anxiety/frustration, sustaining
engagement, and (in the self-determination framing) supporting the student's autonomy, sense of
competence, and relatedness.

**Evidence it is a distinct, measurable, domain-independent competency:**

- **CLASS "Emotional Support" domain — the decisive validation.** The Classroom Assessment Scoring
  System / Teaching Through Interactions framework organizes teacher–student interaction into
  **three** domains — **Emotional Support, Classroom Organization, and Instructional Support** —
  and confirmatory factor analysis over **4,000+ preschool–grade-5 classrooms** shows the
  **three-factor structure fits better than one- or two-factor alternatives**, and generalizes
  across grades (Hamre, Pianta, Downer, et al., 2013, *The Elementary School Journal* 113(4):461–487,
  doi:10.1086/669616; "Building a Science of Classrooms," Pianta & Hamre, FCD report). A
  **meta-analysis of 26 CLASS correlation matrices** (two-stage SEM) independently supports the
  three-factor model (2019, *Journal of Experimental Education*, doi:10.1080/00220973.2018.1551184).
  The secondary-school replication (Hafen et al., 2015, *J. of Psychoeducational Assessment*, PMC5319784)
  again finds three domains, with Emotional Support comprising Positive Climate, Negative Climate,
  Teacher Sensitivity, and Regard for Adolescent Perspectives. **This is the single best piece of
  external evidence in the whole memo: it is a direct demonstration that "emotional/affective
  support" is a factor empirically separable from "instructional support" (≈ our scaffolding+content)
  in a large, well-powered sample.**
- **Expert-tutor practice (INSPIRE).** Lepper & Woolverton (2002), "The Wisdom of Practice," in
  Aronson (ed.), *Improving Academic Achievement*, pp. 135–158, Academic Press
  (doi:10.1016/B978-012064455-1/50010-5). Expert tutors run a **dual diagnostic model —
  cognitive AND motivational — simultaneously**, and the "N" (Nurturant) and "E" (Encouraging)
  components are explicitly about rapport, empathy, and bolstering competence/confidence, treated
  as *co-equal with* cognition, and hardest exactly when cognitive and motivational indicators
  *point in different directions* (i.e., the two are not redundant).
- **Affective states are real and dynamic in tutoring.** D'Mello & Graesser (2012), "Dynamics of
  Affective States during Complex Learning," *Learning and Instruction* 22(2):145–157
  (doi:10.1016/j.learninstruc.2011.10.001) — boredom/confusion/flow/frustration oscillate during
  tutoring; and D'Mello & Graesser (2012), "Language and Discourse Are Powerful Signals of Student
  Emotions during Tutoring," *IEEE Transactions on Learning Technologies* 5(4):304–317
  (doi:10.1109/tlt.2012.10) — affect is text-detectable and a first-class channel in Affective
  AutoTutor.
- **Self-determination theory (why affective support has motivational leverage).** Ryan & Deci
  (2000), "Self-Determination Theory and the Facilitation of Intrinsic Motivation…," *American
  Psychologist* 55(1):68–78 (doi:10.1037/0003-066X.55.1.68); Niemiec & Ryan (2009), "Autonomy,
  competence, and relatedness in the classroom," *Theory and Research in Education* 7(2):133–144
  (doi:10.1177/1477878509104318). Teacher support of the three needs (autonomy, competence,
  relatedness) predicts engagement, performance, and well-being — and autonomy support is
  experimentally manipulable/trainable (Cheon, Reeve, & Moon, 2012, *J. of Sport & Exercise
  Psychology* 34(3):365–396, doi:10.1123/jsep.34.3.365).
- **Achievement emotions predict learning partly independent of ability.** Pekrun (2006),
  control-value theory of achievement emotions, *Educational Psychology Review* 18:315–341
  *(unverified this pass — from companion memo; DOI 10.1007/s10648-006-9029-9)*.

**Separability verdict — YES.** Two independent arguments: (i) **structural** — affective-support
criteria load *no* domain content, so unlike adaptability (which loaded scaffolding+diagnosis) they
cannot be pulled into the correctness/diagnosis axis; and (ii) **empirical, external** — the CLASS
CFA literature *directly* shows Emotional Support separates from Instructional Support as its own
factor. In our bank these criteria are currently `primary_skill: null` / all-zero (an *unmodeled
orphan*, ~45–55% of the 1,107 all-zero pool per the taxonomy memo's orphan probe), so promoting
them **recovers discarded signal** rather than re-labeling scaffolding.

**Honest caveat (the real risk is variance, not collinearity).** CLASS domains, though separable,
are still *positively correlated* (Emotional and Instructional Support commonly correlate
moderately), so expect a positive — not zero — latent correlation with the pedagogy axis; the
pre-registered bar is < ~0.9, not ≈0. And affective competence may be **near-ceiling/low-variance**
on capable frontier models ("be encouraging" is easy), which would sink discrimination independent
of collinearity. This is the decisive empirical uncertainty and is *not* resolvable by reading.

### 2. Metacognition / self-regulated-learning support — **separable from scaffolding? PARTLY (mostly NO on our instrument)**

**Construct definition.** Prompting the student to *plan, monitor, self-check/evaluate, and reflect*
on their own reasoning and strategy — supporting the regulation of learning rather than the content
of it.

**Evidence:**

- **The construct is real and domain-general.** Zimmerman (2002), "Becoming a Self-Regulated
  Learner: An Overview," *Theory Into Practice* 41(2):64–70 *(unverified this pass — from companion
  memo; DOI 10.1207/s15430421tip4102_2)*.
- **Azevedo's cognitive-vs-metacognitive scaffolding distinction — the key nuance.** Azevedo &
  Hadwin (2005), "Scaffolding self-regulated learning and metacognition — Implications for the
  design of computer-based scaffolds," *Instructional Science* 33(5–6):367–379
  (doi:10.1007/s11251-005-1272-9); Greene, Moos, & Azevedo (2011); Azevedo et al. adaptive
  content-vs-process scaffolding studies (Azevedo, Cromley, & Seibert, 2004, *Contemporary
  Educational Psychology* 29:344–370). **Critically, this literature frames metacognitive support
  as a *type of scaffolding* — "process scaffolding" / "metacognitive scaffolding" set explicitly
  in contrast to "content/cognitive scaffolding," both delivered by the same tutor as adaptive
  guidance.** So the field's own taxonomy nests metacognitive support *inside* scaffolding as a
  subtype, not beside it as an orthogonal axis.

**Separability verdict — PARTLY, but mostly NO in our instrument.** Conceptually distinguishable
from *content* scaffolding, yes; but a "get the student to self-check / reflect / plan" move is
operationally a **guiding question**, which is the core of our `scaffolding` skill, and our bank
already maps these criteria to scaffolding. This is the exact failure mode `adaptability` showed:
a conceptually-broad label whose operational items already belong to scaffolding, driving the two
axes toward collinearity. It becomes separable **only if** items isolated the *self-regulation
object* (plan/monitor/evaluate your own process) from the *question form* — which our items do not.
**Downgrade relative to the naive prior:** conceptually strong, empirically weak on our instrument.
(Note: this is *also* Hattie & Timperley's "self-regulation level" of feedback — see candidate 6.)

### 3. Communication / explanation clarity — **separable from scaffolding? PARTLY**

**Construct definition.** Whether the response is clear, concise, well-organized, appropriately
jargon-free, and readable — independent of whether it is *correct* (content) or *pedagogically
sequenced* (scaffolding).

**Evidence:**

- **Teacher clarity is a validated, separately-measured construct with real effects.** Titsworth,
  Mazer, Goodboy, Bolkan, & Myers (2015), "Two Meta-analyses Exploring the Relationship between
  Teacher Clarity and Student Learning," *Communication Education* 64(4):385–418
  (doi:10.1080/03634523.2015.1041998). Clarity accounts for **~13% of variance** in learning across
  144 effects (N = 73,281); notably the effect is **larger for affective learning (r ≈ .52) than
  cognitive (r ≈ .34)** — which is itself a hint that clarity and affect are entangled outcomes.
  The authors flag heterogeneity and urge a return to **low-inference behavioral** measures over
  high-inference perceptual ones (relevant to how we would label it).
- Formative-feedback theory stresses clarity/comprehensibility: Shute (2008), "Focus on Formative
  Feedback," *Review of Educational Research* 78(1):153–189 *(unverified this pass — from companion
  memo)*.

**Separability verdict — PARTLY.** Domain-independent (good, like affect), and *conceptually*
distinct from scaffolding (clarity ≠ sequencing). But two practical problems in our bank: (i)
clarity criteria are **frequently co-authored with affective tone** into a single all-zero
criterion ("clear, student-friendly language *and* an encouraging tone"), so it may not separate
from motivation; and (ii) it is likely **low-variance** — most competent LLMs write clearly — so
weak discrimination. **Best treated as a possible sub-facet of a broader delivery/affect axis, not
its own dimension.** (This overlaps memo 01's "presentation," which found the *formatting* residue
separable-but-surface; clarity-of-language is the non-formatting sibling.)

### 4. (Also asked) Are we missing a well-supported dimension entirely?

Three families were checked; **none clears the separable-from-scaffolding bar**, and one confirms
scaffolding's own definition:

- **Contingency / adaptivity — NO (it IS scaffolding).** van de Pol, Volman, & Beishuizen (2010),
  "Scaffolding in Teacher–Student Interaction: A Decade of Research," *Educational Psychology
  Review* 22(3):271–296 (doi:10.1007/s10648-010-9127-6) identify **contingency, fading, and
  transfer of responsibility** as the **three defining characteristics of scaffolding itself**.
  Contingency (tailoring support to the student's current level) is therefore *constitutive of*
  scaffolding, not a separate axis — this externally validates the companion adaptability memo's
  conclusion that adaptivity is absorbed into scaffolding+diagnosis. (This review also flags that
  *measuring* scaffolding is the field's hard problem — consistent with our own scaffolding items
  cohering less tightly than content/diagnosis.)
- **Questioning / Socratic elicitation — NO (it is the "guiding-question" form of scaffolding).**
  The "S" of INSPIRE (Lepper & Woolverton 2002; >90% of expert-tutor moves are questions) and
  AutoTutor's Expectation–Misconception-Tailored dialogue (Graesser et al. 2004) *(from companion
  memo)*. In our bank this is already `scaffolding`.
- **Feedback quality — NO (it decomposes into the existing axes).** Hattie & Timperley (2007), "The
  Power of Feedback," *Review of Educational Research* 77(1):81–112 (doi:10.3102/003465430298487).
  Their model is a *cross-cut*, not a fourth skill: effective feedback answers three questions
  (feed-up / feed-back / feed-forward) at four levels (**task, process, self-regulation, self**).
  Mapping onto our axes: the *task/process* levels = correctness/diagnosis + scaffolding; the
  *self-regulation* level = candidate 2 (metacognition); the *self* level (affect about the learner)
  = candidate 1 (motivation). So "feedback quality" is not a missing axis — it is a lens whose
  components are exactly {content, diagnosis, scaffolding, metacognition, affect}. Usefully, Hattie
  & Timperley find the **self level (praise about the person) least effective** — a caution for how
  we would define the motivation axis (support competence via effort/strategy, not empty praise).

---

## (c) Top candidate — proposed sharp definition for reliable human + LLM labeling

**Dimension name:** `motivation` (motivational / affective support).

**Core question the labeler answers (per criterion):** *Does this criterion require the tutor to
support the student's motivational or emotional state — encouragement, effort/growth framing,
confidence protection, anxiety/frustration reduction, or autonomy/competence/relatedness support —
independent of the correctness or structure of the domain help?*

**INCLUDE (loads `motivation` = 1) — criterion primarily requires:**
- Encouragement / expressed confidence in the student's ability to succeed ("reassure the student
  they can solve this," "affirm the student's capability").
- Effort-/process-/growth framing of mistakes ("frame the error as a normal, productive part of
  learning," "praise the student's *strategy/effort*," not merely the person).
- Explicit emotional acknowledgement/validation aimed at *state regulation* ("acknowledge the
  student's frustration/anxiety before proceeding," "normalize the confusion").
- Autonomy/relatedness support ("offer the student a choice of how to proceed," "invite the student
  to set the pace," warmth/rapport moves).

**EXCLUDE (does NOT load `motivation`):**
- **Correctness or error-identification** content → that is content/diagnosis (even if phrased
  warmly). *"Warm delivery of a correct diagnosis" loads content/diagnosis, not motivation.*
- **Guiding questions / hints / step-sequencing / reflection prompts** → that is scaffolding
  (candidate 2/5). *A reflection or self-check prompt is scaffolding even though it is
  "metacognitive."*
- **Pure formatting / clarity-of-language** ("use headings," "be concise," "avoid jargon") → that
  is presentation/clarity (memo 01), not motivation.
- **Empty person-praise** with no effort/strategy/competence content ("great job!") → per Hattie &
  Timperley (2007) the least-effective "self level"; treat as **low-value/borderline** and, to keep
  the construct sharp and gradeable, **exclude** unless it is tied to effort/competence.

**Tie-breakers for reliability:**
- **Primary-purpose rule.** Load `motivation` only when affective/motivational support is the
  criterion's *primary* demand. If a criterion bundles "encouraging tone *and* correct explanation,"
  the affect clause loads `motivation` only if it is separately gradeable; a single inseparable
  pass/fail unit defaults to its cognitive component (mirrors the content/diagnosis bundling
  problem).
- **State-regulation, not sentiment-detection.** The move must aim to *change/support the student's
  motivational-emotional state*, not merely mention feelings.
- **Domain-independence check.** If you cannot decide pass/fail *without* knowing the subject-matter
  answer, it is not `motivation`.

This yields a construct that is (a) domain-independent by construction, (b) disjoint from
scaffolding's question/sequencing form, and (c) aligned to the CLASS Emotional Support and SDT
need-support literature — i.e., reliably labelable by both humans and an LLM judge.

---

## (d) Expected correlations with existing dimensions + identifiability risk

- **`motivation` × correctness/diagnosis:** expected **low** (near 0), because affective criteria
  load no content. This is the whole point — it is the property that keeps it out of the fused
  correctness/diagnosis axis. **Directional support:** memo 01's proxy analysis found affect
  orphans behave as their own group; CLASS shows Emotional vs Instructional Support separate.
- **`motivation` × scaffolding:** expected **low-to-moderate positive**. Warning: warmth and
  guiding-questions co-occur in good tutors (INSPIRE bundles N/E with S), and CLASS Emotional and
  Instructional Support are positively correlated. So do **not** expect ≈0; the pre-registered
  separability bar should be **latent r < ~0.9 with scaffolding** (same threshold used for the
  content/diagnosis collapse and the adaptability/broad-communication tests), not r ≈ 0.
- **Identifiability risk — variance, not collinearity, is the binding constraint.** Unlike
  adaptability/metacognition (which risk *re-collision* with scaffolding), `motivation`'s risk is a
  **degenerate/low-variance factor**: if affective competence is near-ceiling on the scored fleet,
  the dimension carries little discrimination and the MIRT θ variance collapses, making the axis
  unidentifiable regardless of correlation. The false-1 asymmetry (a spurious dimension is systemic
  and hard to detect at finite N) means we must gate adoption on *positive* evidence of both
  meaningful item variance and sub-0.9 latent correlations, not adopt pre-emptively.

**Pre-registered adoption rule (consistent with the companion memos' §7 / §2.4 triggers).** On the
full graded matrix, after the content+diagnosis collapse, run EFA + latent-correlation estimation
and re-fit with a candidate `motivation` loading on the affect/encouragement criteria. **Adopt
`motivation` as the taxonomy's third axis (correctness/diagnosis + scaffolding + motivation) IFF:**
1. a residual EFA factor loads **coherently on the affect_motivation items**, **AND**
2. that factor has **meaningful item variance** (not near-ceiling on the *full* fleet, incl. strong
   models), **AND**
3. its latent correlation is **< ~0.9 with BOTH** the correctness/diagnosis axis **and** scaffolding.

Otherwise keep affect as descriptive all-zero items reported via `use_case` / critical-failure
monitoring. Cost if adopted: Q-matrix regen + verify + re-fit only; tutor responses and judge
verdicts reused (no re-grade), plus a one-time re-mapping of affective all-zero items ("support" vs
mere "tone") — bounded and reversible, per the companion memos.

---

## (e) Established findings vs. speculation, and where our small-N data cannot adjudicate

**Established (external, verified literature):**
- Emotional/affective support is an **empirically separable factor** from instructional support in
  large samples (CLASS 3-factor CFA over 4,000+ classrooms; 26-matrix meta-analysis). *This is the
  firmest claim in the memo.*
- Expert tutors run **dual cognitive+motivational diagnosis** (INSPIRE); affect is dynamic and
  measurable in tutoring dialogue (D'Mello & Graesser 2012); need-support (autonomy/competence/
  relatedness) predicts engagement and learning and is trainable (SDT).
- **Contingency is constitutive of scaffolding** (van de Pol et al. 2010) — so adaptivity is not a
  separate axis.
- **Metacognitive support is framed as a *subtype* of scaffolding** ("process/metacognitive
  scaffolding," Azevedo) — so on-instrument separability is unlikely.
- **Feedback quality decomposes** across task/process/self-reg/self levels (Hattie & Timperley
  2007) — not a standalone fourth skill; the "self level" (person-praise) is *least* effective.
- Teacher **clarity** is a real construct explaining ~13% of learning variance (Titsworth et al.
  2015), but its effect is *larger on affective than cognitive* outcomes — evidence clarity and
  affect are entangled, arguing against clarity as an independent axis.

**Speculation / model-specific (flag before quoting):**
- That `motivation` will show a *sub-0.9* latent correlation with scaffolding **on our instrument**
  — plausible (structural domain-independence) but unproven; CLASS domains are positively
  correlated, so a moderate positive r is likely.
- That affective competence will have **enough between-model variance** on a *strong* fleet to
  identify the axis — this is the make-or-break unknown. Our current fleet (weak small models) may
  show spread that flattens once frontier models are added (memo taxonomy §2.4).
- The orphan-pool magnitudes (affect ≈ 45–55% of the 1,107 all-zero criteria) are **keyword
  heuristics over free text**, not adjudicated labels — must be hand-audited before quoting.

**Where our small-N data cannot adjudicate:** every latent-structure question here needs the full
graded matrix. The current directional numbers come from an **N≈28** biased small-model subsample
on a ~1/4-filled matrix (< the ~150 persons the calibrator needs for a 3-dim M2PL; ~250 for 4-dim).
At that N, AIC/BIC dimensionality comparisons are power artifacts and must be ignored. **No decision
on a 4th dimension can be made from present data** — this memo pre-registers the test, it does not
resolve it.

---

## Appendix — sources (verified this pass unless flagged)

- **CLASS / Teaching Through Interactions:** Hamre, Pianta, Downer, DeCoster, Mashburn, Jones, et
  al. (2013). Teaching through Interactions. *The Elementary School Journal* 113(4):461–487.
  doi:10.1086/669616. — Pianta & Hamre, "Building a Science of Classrooms" (FCD report, 4,000+
  classrooms). — Hafen et al. (2015), CLASS-S secondary factor structure, PMC5319784. —
  Meta-analysis of CLASS factor structure (2019), *J. of Experimental Education*,
  doi:10.1080/00220973.2018.1551184.
- **INSPIRE / expert tutors:** Lepper, M. R., & Woolverton, M. (2002). The Wisdom of Practice. In
  Aronson (ed.), *Improving Academic Achievement*, pp. 135–158. Academic Press.
  doi:10.1016/B978-012064455-1/50010-5.
- **Affective tutoring:** D'Mello, S., & Graesser, A. (2012). Dynamics of Affective States during
  Complex Learning. *Learning and Instruction* 22(2):145–157. doi:10.1016/j.learninstruc.2011.10.001.
  — D'Mello & Graesser (2012). Language and Discourse Are Powerful Signals of Student Emotions
  during Tutoring. *IEEE Trans. Learning Technologies* 5(4):304–317. doi:10.1109/tlt.2012.10.
- **Self-determination theory:** Ryan, R. M., & Deci, E. L. (2000). *American Psychologist*
  55(1):68–78. doi:10.1037/0003-066X.55.1.68. — Niemiec & Ryan (2009). *Theory and Research in
  Education* 7(2):133–144. doi:10.1177/1477878509104318. — Cheon, Reeve, & Moon (2012). *J. of Sport
  & Exercise Psychology* 34(3):365–396. doi:10.1123/jsep.34.3.365.
- **Metacognition / SRL scaffolding:** Azevedo, R., & Hadwin, A. F. (2005). *Instructional Science*
  33(5–6):367–379. doi:10.1007/s11251-005-1272-9. — Azevedo, Cromley, & Seibert (2004).
  *Contemporary Educational Psychology* 29:344–370. — Zimmerman (2002). *Theory Into Practice*
  41(2):64–70 *(unverified this pass — companion memo)*.
- **Clarity:** Titsworth, Mazer, Goodboy, Bolkan, & Myers (2015). *Communication Education*
  64(4):385–418. doi:10.1080/03634523.2015.1041998.
- **Scaffolding definition / contingency:** van de Pol, Volman, & Beishuizen (2010). *Educational
  Psychology Review* 22(3):271–296. doi:10.1007/s10648-010-9127-6.
- **Feedback:** Hattie, J., & Timperley, H. (2007). The Power of Feedback. *Review of Educational
  Research* 77(1):81–112. doi:10.3102/003465430298487.
- **Unverified this pass (cited from companion memos):** Pekrun (2006) control-value theory,
  *Educ. Psych. Review* 18:315–341; Shute (2008) formative feedback, *RER* 78(1):153–189; Graesser
  et al. (2004) AutoTutor, *Behavior Research Methods* 36:180–192; Chi & Wylie (2014) ICAP; Dignath
  & Büttner (2008) SRL meta-analysis; Bloom (1984) 2-sigma; VanLehn (2011) tutoring effectiveness.
- **Local:** `plans+prds/Tutor Skill Taxonomy - Theory + Candidate Dimensions.md`;
  `plans+prds/Collinearity Analysis + Skill-Definition Options.md`;
  `plans+prds/Skill-Definition Options - Presentation + Explanation.md`;
  `plans+prds/TutorBench Use Cases + Adaptability.md`.
