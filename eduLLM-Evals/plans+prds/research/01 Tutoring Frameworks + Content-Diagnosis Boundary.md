# 01 — Canonical Tutoring/Teaching-Quality Frameworks & the Content↔Diagnosis Boundary

**Status:** literature-research memo (thread 1 of 3). **Changes nothing** — it does not touch
`data/scenarios.jsonl`, `data/rubrics_qmatrix_final.jsonl`, the `q_mapping`s, the skill
definitions, or any calibration/judge code. It surveys the **canonical teaching/tutoring
competence and teaching-quality literature** to referee one specific question our own data can't
settle at N: **is "diagnosis" (reading the student's specific error) empirically separable from
"content" (domain correctness), or is it fundamentally nested inside it?** — and to derive
sharper, more rater-reliable inclusion/exclusion rules for our three skills.

**Scope of the modeled taxonomy:** three latent skills — **content** (domain correctness),
**diagnosis** (identifying *this* student's specific error/confusion), **scaffolding**
(structuring/sequencing pedagogical help).

**Internal findings taken as given** (from companion memos):
- content↔diagnosis are nearly fused: structural co-occurrence r≈0.945-adjacent overlap
  (P(content|diagnosis)=81.9%), directional latent r≈0.945 at N≈28; near-nesting signature
  (diagnosis ≈ sub-region of content).
- The overlap is mostly FUNDAMENTAL (CK→PCK nesting + prerequisite structure + bundled items),
  not a labeling artifact (only ~1.7% dissolves under a narrower diagnosis definition).
- scaffolding separates cleanly (latent r ≈ −0.19 with content/diagnosis).
- Human raters had LOW agreement on `diagnosis` labels (Fleiss κ = 0.365) vs content (0.85),
  scaffolding (0.59).
- Companion memo 03: external LLM-tutor rubrics DO score mistake-identification as a standalone
  dimension — but on datasets where the diagnosis base rate ≈100% by construction.

**Citation policy.** Every literature claim is tagged author/year/venue/link and flagged
**(verified this pass)** or **(unverified — carried from companion memo / to check)**.

---

## (a) Executive summary

1. **The canonical literature treats "diagnosis" as a real, nameable competency — but NOT as an
   empirically separable dimension from content.** Across Shulman/PCK, COACTIV, TEDS-M, diagnostic
   competence, teacher noticing, tutoring-move taxonomies, and validated observation instruments,
   error-reading is consistently *defined over* the content and measured as near-collinear with
   it. **No validated multidimensional teaching-quality model isolates diagnosis as its own
   factor.** This corroborates our directional latent r = 0.945.

2. **The CK↔PCK correlation is a MIXED, method-dependent number that lands at its fused extreme in
   exactly our regime.** COACTIV: manifest r = 0.60 but latent **≈1 among expert (academic-track)
   teachers**; TEDS-M: r≈.6 individual (→≈.3 after controls); elementary specialized-CK
   **inseparable** from PCK; Depaepe et al. (2013): separability depends on operationalization
   (integrative vs transformative). Our instrument — capable tutors + single pass/fail items that
   *bundle* "know the answer" and "spot the error" — is the worst case for separation, so
   r = 0.945 is **consistent with, not contradictory to,** "CK and PCK are distinct in principle."

3. **Danielson's Framework for Teaching is a direct precedent for our proposed collapse.** A
   REL-West validity study found the 4 FfT domains correlated **r > .90 ("nearly redundant") and
   recommended a single rating** — a theoretically-multidimensional teaching instrument whose
   factors empirically collapse to one, and whose field response is exactly our content+diagnosis
   collapse (keep the language, report the merged dimension).

4. **Where teaching-quality models DO cleanly separate a factor, it is affect or
   scaffolding/cognitive activation — never diagnosis.** CLASS (Emotional Support separate from
   Instructional Support), Three Basic Dimensions (student support; cognitive activation), and
   INSPIRE (Nurturant/Encouraging) all peel off **affect**; VanLehn locates the tutoring *effect*
   in **interaction granularity** (scaffolding). This triangulates our two companion findings:
   scaffolding separates cleanly, and motivation/affect is the strongest unmodeled-orphan
   candidate.

5. **Our κ = 0.365 on diagnosis is a normal, remediable measurement problem, not evidence the
   construct is unreal.** The MET project shows even professionally-trained observers are
   unreliable on a *single* rating and need **multiple raters + averaging + anchors**; Südkamp
   shows diagnostic accuracy rises when judges are **task-informed**; noticing is trained via
   intensive protocols. Fix reliability with anchored inclusion/exclusion rules (§e) + more
   raters — not by deleting or narrowing the construct.

**Bottom line:** keep `diagnosis` as a labeled/reported construct; **expect and accept its
collapse into a single "correctness/diagnosis" competence axis** in the latent model; invest the
definitional effort in the attend-vs-respond boundary and concrete-referent anchoring to lift
rater reliability.

---

## (b) Per-framework synthesis

### 1. Shulman's knowledge base & the empirical CK↔PCK separability question

**The taxonomy places diagnosis inside content.** Shulman (1986, *Educational Researcher*
15(2):4–14; 1987, *Harvard Educational Review* 57(1):1–22) partitions teacher knowledge into
**content knowledge (CK)**, **general pedagogical knowledge (GPK)**, and **pedagogical content
knowledge (PCK)** — the latter being the "amalgam of content and pedagogy," explicitly including
*"the conceptions and preconceptions that students bring"* and the misconceptions they are prone
to. Our `content` = CK; our `diagnosis` (reading *this* student's specific error) is a **PCK
facet — "knowledge of students' (mis)conceptions."** By definition PCK is knowledge *about the
content*: you cannot know the typical student error on a topic without the content knowledge of
that topic. So the taxonomy itself **nests diagnosis inside content** — the first structural
reason they don't cleanly separate. (This is the companion-memo finding; restated here as the
frame for the empirical question below.)

**The empirical separability question — a genuinely MIXED literature.** Whether PCK is
*measurably* distinct from CK, or collapses into it, has been tested by three large programs.
The answer is **"separable in principle, but the correlation is high and rises toward 1 exactly
in our regime (expert responders + bundled items)."**

- **COACTIV** (German secondary-math teachers). Manifest CK–PCK correlation **r = 0.60**;
  crucially, in the **high-expertise academic-track ("Gymnasium") subgroup the latent CK–PCK
  correlation was "no longer statistically distinguishable from 1"** (Krauss, Baumert, & Blum,
  2008, *ZDM* 40(5):873–892; Krauss et al., 2008, *J. Educational Psychology* 100(3):716–725).
  Yet **Baumert et al. (2010, *American Educational Research Journal* 47(1):133–180,
  doi:10.3102/0002831209345157)** found CK and PCK "theoretically and empirically
  distinguishable" **and** that *PCK*, not raw CK, predicted student learning gains — so the two
  are distinct *in principle and in consequence* even when nearly collinear *in measurement*.
  **(verified this pass — Baumert et al. AERJ, doi confirmed via Praetorius reference list.)**
- **TEDS-M** (international, 17 countries, pre-service teachers). Models **Mathematics Content
  Knowledge (MCK)** and **Mathematics Pedagogical Content Knowledge (MPCK)** as **distinct but
  correlated factors** — a "two-dimensional structure … PCK constitutes a distinct, but
  correlated dimension of subject-matter knowledge" (Kaiser & Blömeke lineage; Taiwan–Germany
  replication, *Teaching and Teacher Education*, doi:10.1016/j.tate.2014.09.003). Individual-level
  MCK–MPCK **r ≈ 0.6**, and **program-level r ≈ 0.7** but the individual correlation **drops to
  ≈0.3 after controlling for program effects** in the Russian sub-sample (Blömeke et al.; TEDS-M
  correlation-methods paper, *Large-scale Assessments in Education* 1:7,
  doi:10.1186/2196-0739-1-7). Blömeke's throughline: **CK is necessary but not sufficient for
  PCK.** Note the important boundary case: for **elementary** teachers, *specialized* content
  knowledge is **"inseparable from" PCK** (Blömeke et al. 2014; Copur-Gencturk et al. 2019) —
  i.e., the more the CK measure is itself pedagogically specialized, the less it separates from
  PCK. **(verified this pass.)**
- **Depaepe, Verschaffel, & Kelchtermans (2013)**, "Pedagogical content knowledge: A systematic
  review …," *Teaching and Teacher Education* 34:12–25, doi:10.1016/j.tate.2013.03.001
  (**verified this pass** — authors, venue, year, DOI confirmed). The review's relevant verdict:
  PCK's *relationship to CK* is unsettled across studies, split between an **integrative** view
  (PCK = CK + pedagogy held together ⇒ high correlation, weak separability) and a
  **transformative** view (PCK a genuinely new amalgam ⇒ separable). The empirical record sits
  **between**, and separability depends heavily on **how the constructs are operationalized** —
  exactly the measurement-confound lever.

**Net for us.** The canonical teacher-knowledge literature is **not** a clean "diagnosis is its
own factor" result. It is: *distinguishable in principle, predictively distinct, but empirically
near-collinear — and driven toward r ≈ 1 precisely by (i) expert respondents and (ii)
pedagogically-bundled items.* Our instrument (able-ish tutors, single pass/fail items bundling
"know the answer" + "spot the student's error") is the **worst case** for separation. Our
directional latent r = 0.945 is therefore **consistent with**, not anomalous against, this
literature — it is the COACTIV academic-track regime reproduced.

### 2. Diagnostic competence as a construct + teacher "noticing"

Does the field measure *error-diagnosis* as its own reliable factor, or is it fused with content
and — importantly for us — is it *reliably ratable*? The evidence says diagnosis is a **real,
named competency that is nonetheless hard to measure reliably**, which directly mirrors our
`diagnosis` κ = 0.365 problem.

- **Teacher "diagnostic competence" (judgment accuracy).** Südkamp, Kaiser, & Möller (2012),
  "Accuracy of teachers' judgments of students' academic achievement: A meta-analysis," *Journal
  of Educational Psychology* 104(3):743–762, doi:10.1037/a0027627 (**verified this pass**). 75
  studies; mean teacher-judgment↔test correlation **r = .63** (≈40% of variance), **higher when
  judgments are *informed* about the specific test/task** than when uninformed. A 2024
  psychometric-meta-analytic replication (*PLOS ONE*, doi:10.1371/journal.pone.0307594) puts it a
  touch lower (mean r ≈ .58, median .53). **Reading for us:** even the *global* form of diagnostic
  competence leaves ~60% of variance unexplained and is **contingent on how much task-specific
  information the judge has** — i.e., diagnosis is measurable but noisy, and its accuracy is
  *conditional on content/task information* (the informed-judgment moderator is the
  content-dependence showing up empirically).
- **Teacher "noticing."** van Es & Sherin (2002), "Learning to notice: Scaffolding new teachers'
  interpretations of classroom interactions," *Journal of Technology and Teacher Education*
  10(4):571–596 (**verified this pass**; framework text confirmed). Noticing = **(a) identifying
  what is noteworthy, (b) using knowledge of context to reason about it, (c) connecting specifics
  to broader principles.** Jacobs, Lamb, & Philipp (2010), "Professional noticing of children's
  mathematical thinking," *JRME* 41(2):169–202, decompose it into **attending → interpreting →
  deciding how to respond** (**verified this pass** via the ZDM review). This is the closest
  external analogue to our `diagnosis`: *attending/interpreting* = reading the specific error;
  *deciding how to respond* straddles diagnosis→scaffolding. **Two facts matter:** (i) the field
  treats "attend to and interpret the student's specific thinking" as a nameable skill distinct
  from delivering content — supporting a `diagnosis` *label*; but (ii) it is **defined over the
  content** (you interpret *mathematical* thinking) and is documented as **hard to develop and to
  score reliably**, which is why it is trained via intensive video-club protocols rather than
  read off cheaply — mirroring why our raters agreed least on `diagnosis`.

**Net for section 2.** Diagnosis/noticing is a **first-class construct in the teaching
literature** (so keeping a `diagnosis` label is well-motivated), but it is (a) measured with
**moderate-at-best reliability**, (b) **conditional on content/task information**, and (c)
operationally entangled with "what you then do about it" (scaffolding). This triangulates our own
result: a real but noisy, content-nested axis.

### 3. Human/ITS tutoring-move taxonomies

These taxonomies describe the *actions* a tutor takes, and they consistently name a
**diagnose→respond** loop in which reading the student's state is a distinct step from the
domain-help step — but always operating *over* the content.

- **VanLehn (2011)**, "The relative effectiveness of human tutoring, ITS, and other tutoring
  systems," *Educational Psychologist* 46(4):197–221, doi:10.1080/00461520.2011.611369
  (**verified this pass**). Human tutoring **d = 0.79**, ITS **d = 0.76** (nearly equal; both far
  below the folklore 2σ). VanLehn frames tutoring as a two-loop structure — an **outer loop**
  (task selection) and an **inner loop** that, per step, (i) *gives feedback on the step*, (ii)
  *gives hints on the next step*, (iii) *assesses knowledge*, all keyed to **interaction
  granularity** (answer/step/substep). Effectiveness plateaus at **step-based** granularity. For
  us: the inner loop's "assess the student's step / give error feedback" = **diagnosis+content
  fused at the step level**, while "hint on next step / granularity" = **scaffolding** — and
  VanLehn locates the *effect* in granularity (scaffolding), not raw content delivery, matching
  our clean-scaffolding result.
- **Chi & Wylie ICAP** (2014, *Educational Psychologist* 49(4):219–243) — Interactive >
  Constructive > Active > Passive. This is a taxonomy of **student cognitive engagement elicited
  by the tutor**; it maps onto **scaffolding** (getting the student to construct/interact rather
  than receive), not onto the content/diagnosis boundary. *(carried from companion memos; ICAP
  venue/year consistent across memos.)*
- **Graesser/AutoTutor Expectation–Misconception-Tailored (EMT) dialogue** (Graesser et al.,
  2004, *Behavior Research Methods* 36:180–192). The EMT engine is *built on* a
  content/diagnosis pairing: **expectations** (correct content the student should articulate) and
  **misconceptions** (specific student errors to detect and correct) — i.e., the system
  operationalizes diagnosis *as a list defined over the content*, the clearest possible
  demonstration of the nesting. Delivery is via hints/prompts (scaffolding). *(carried from
  companion memos.)*
- **Lepper INSPIRE / expert human tutors** (Lepper & Woolverton, 2002). Expert tutors run a
  **dual cognitive + motivational diagnosis**; the model's letters mix scaffolding ("S"ocratic),
  affect ("N"urturant, "E"ncouraging), and content. Relevant here: even the expert-tutor
  literature treats **diagnosis as a running read of the student that feeds the response**, not
  as a separable "get the answer right" competency. *(carried from companion memos.)*

**Net for section 3.** Every tutoring-move taxonomy contains a recognizable
**read-the-student (diagnosis)** action distinct from **deliver-domain-help (content)** and
**structure-the-help (scaffolding)** — so a three-way *conceptual* decomposition is well
supported. But diagnosis is always the *inner-loop assessment over the content*, never a
free-standing axis; and the measured *effect* concentrates in the scaffolding/granularity axis.

### 4. Empirically validated dimensional models of teaching quality

This is where "how many dimensions actually validate, and do the diagnosis-like ones separate?"
is answered most directly — and the answer is a **cautionary tale of near-collinear factors**.

- **CLASS / Teaching Through Interactions** — the strongest *positive* separability result, but
  **not** for the content/diagnosis split. CFA over 4,000+ classrooms validates **three**
  domains — **Emotional Support, Classroom Organization, Instructional Support** — fitting better
  than 1- or 2-factor alternatives (Hamre, Pianta, et al., 2013, *Elementary School Journal*
  113(4):461–487, doi:10.1086/669616); a 26-matrix meta-analysis concurs
  (doi:10.1080/00220973.2018.1551184). *(verified in companion memo 02.)* **What CLASS separates
  is affect vs. instruction — it does NOT split "diagnosis" out of "instructional support";**
  error-reading lives inside Instructional Support (Content Understanding / Analysis & Problem
  Solving / Quality of Feedback). So CLASS's own validated structure keeps diagnosis+content
  bundled and pulls *affect* out as the clean separate axis — echoing our result that scaffolding
  and (prospectively) motivation separate while content↔diagnosis do not.
- **Three Basic Dimensions (Praetorius et al., 2018)**, "Generic dimensions of teaching quality:
  the German framework of Three Basic Dimensions," *ZDM* 50(3):407–426,
  doi:10.1007/s11858-018-0918-4 (**verified this pass**). Three dimensions — **classroom
  management, student support, cognitive activation** — originating in a factor analysis of the
  TIMSS video study (Klieme, Schümer & Knoll, 2001). Reported evidence: **good reliability, MIXED
  predictive validity.** Notably, **Kleickmann et al. (2020)** find the 3-factor model
  under-represents *cognitive support* and argue for a **4th dimension** in science (two-level
  CFA, 2,659 students) — i.e., the "right" dimensionality is itself contested. **Crosswalk:**
  cognitive activation ≈ scaffolding (+content); student support ≈ motivation/affect (our
  unmodeled orphan); classroom management is off-scope for one-on-one tutoring. **None of the
  three is a diagnosis axis** — again, error-reading is folded into the cognitive/content work.
- **Danielson Framework for Teaching (FfT)** — **the closest external replica of our
  content↔diagnosis collapse.** The FfT posits **four domains** (planning; classroom
  environment; instruction; professional responsibilities). But a REL-West validity study of the
  instrument in one district (713 teachers; Washoe County, 2016, IES/REL 2016-135) found that
  while the four-factor CFA *fit*, **the factors were correlated so highly (all but one pair
  r > .90; the 2-factor solution r = .93) as to be "nearly redundant," and recommended reporting
  a SINGLE rating** ("the groups appear to be measuring one aspect of teaching rather than
  different domains"). *(verified this pass.)* **This is exactly our situation:** a
  theoretically-multidimensional teaching instrument whose factors empirically collapse toward
  one because the constructs co-vary in practice — and the field's response (report one dimension)
  is precisely our proposed content+diagnosis collapse.
- **MET project (Kane & Staiger, 2012, "Gathering Feedback for Teaching," Gates Foundation)** —
  speaks to **reliability**, our most acute `diagnosis` problem (κ = 0.365). Findings
  (**verified this pass**): (i) the five observation instruments' scores were **highly correlated
  with each other** (little discriminant separation among frameworks); (ii) **a single
  observation by a single observer is an unreliable estimate** — a teacher's scores vary lesson-
  to-lesson and observer-to-observer; reliability only reaches ~.65–.67 by **averaging over
  multiple lessons AND multiple observers**, and **combining measures** (observation + student
  survey + achievement) beat any single measure. **Reading for us:** low single-rater agreement
  on a subtle construct is *expected and normal* even for professionally-trained observers; the
  remedy is **more raters/items + explicit anchors**, not abandoning the construct. This
  reframes our κ = 0.365 as an under-powered/under-anchored measurement problem, not proof that
  diagnosis is unreal.

---

## (c) Crosswalk — external constructs → {content, diagnosis, scaffolding}

● = primary map · ◐ = partial/secondary · ○ = not this skill. "Other" flags a construct with no
home in our three (candidate omission, mostly affect — treated in memos 02/03).

| Framework | Construct | content | diagnosis | scaffolding | other | Separates diagnosis from content? |
|---|---|:--:|:--:|:--:|---|---|
| **Shulman** | Content knowledge (CK) | ● | ○ | ○ | — | — |
| | PCK: knowledge of students' (mis)conceptions | ◐ | ● | ○ | — | Conceptually yes; **defined over CK** |
| | PCK: representations/analogies | ◐ | ○ | ● | — | — |
| | General pedagogical knowledge (GPK) | ○ | ○ | ◐ | ● mgmt/affect | — |
| **COACTIV** | CK test | ● | ○ | ○ | — | Latent r→1 in expert group (**no**) |
| | PCK test (tasks, student errors, representations) | ◐ | ● | ◐ | — | Distinguishable in principle; predicts gains |
| **TEDS-M** | MCK | ● | ○ | ○ | — | 2-factor MCK/MPCK; **r≈.6 (partly)** |
| | MPCK | ◐ | ● | ◐ | — | Separable but correlated; inseparable at elementary |
| **Diagnostic competence** (Südkamp) | Judgment accuracy | ○ | ● | ○ | — | Own construct; r=.63 vs test, **content/task-conditional** |
| **Teacher noticing** (van Es & Sherin; Jacobs) | Attend → interpret → decide | ◐ | ● (attend/interpret) | ◐ (decide) | — | Named skill; **over content**, low reliability |
| **VanLehn** inner loop | Assess step / error feedback | ● | ● | ○ | — | Fused at step level (**no**) |
| | Hint next step / granularity | ○ | ○ | ● | — | — |
| **ICAP** (Chi & Wylie) | Interactive/Constructive engagement | ○ | ○ | ● | — | — |
| **AutoTutor EMT** | Expectations | ● | ○ | ◐ | — | — |
| | Misconceptions | ◐ | ● | ○ | — | Diagnosis = list **defined over content** |
| **INSPIRE** | Socratic ("S") | ○ | ◐ | ● | — | — |
| | Nurturant/Encouraging ("N/E") | ○ | ○ | ○ | ● affect | — |
| **CLASS** | Instructional Support | ● | ● | ● | — | **Keeps content+diagnosis bundled** |
| | Emotional Support | ○ | ○ | ○ | ● affect | (separates affect, not diagnosis) |
| | Classroom Organization | ○ | ○ | ◐ | ● mgmt | — |
| **Three Basic Dims** (Praetorius) | Cognitive activation | ◐ | ○ | ● | — | No diagnosis axis |
| | Student support | ○ | ○ | ◐ | ● affect | — |
| | Classroom management | ○ | ○ | ○ | ● mgmt | — |
| **Danielson FfT** | 4 domains (instruction etc.) | ● | ◐ | ● | ● mgmt/prof | **Factors r>.90 → collapse to 1** |
| **MET** | (multi-instrument; reliability lens) | ● | ◐ | ● | ● | Instruments mutually r-high; reliability needs many raters |

**Pattern across the whole table:** the field routinely names a **diagnosis-like construct**
(PCK-misconceptions, noticing, EMT-misconceptions, VanLehn step-assessment) — so a `diagnosis`
*label* is well-founded — but **no validated dimensional model isolates diagnosis as its own
empirical factor separate from content**. Where instruments *do* cleanly separate a factor, it is
**affect** (CLASS Emotional Support; 3BD student support; INSPIRE N/E) or **scaffolding/cognitive
activation** — never diagnosis. And two flagship instruments (Danielson, COACTIV-expert)
**empirically collapse** their fine-grained factors toward a single dimension, exactly as our
data do.

---

## (d) Verdict — is diagnosis separable from content, or fundamentally nested?

**Verdict: In the canonical literature, diagnosis is a real, nameable competency but is
FUNDAMENTALLY NESTED in content — conceptually distinguishable and predictively meaningful, yet
empirically near-collinear with content and never validated as its own standalone dimension in a
multidimensional teaching-quality model.** Four convergent strands:

1. **Taxonomic nesting (Shulman/AutoTutor).** Diagnosis is a PCK facet *defined over* content;
   AutoTutor operationalizes it as a misconception list attached to the content. You cannot
   define it free of content without mislabeling it.
2. **Measurement near-collinearity that rises in our exact regime (COACTIV/TEDS-M/Danielson).**
   CK–PCK is distinguishable *in principle* but latent r climbs toward 1 with **expert
   respondents + bundled items** (COACTIV academic-track), sits at r≈.6 and drops further once
   shared causes are controlled (TEDS-M), is **inseparable** when the content measure is itself
   pedagogically specialized (elementary), and **collapses to one factor at r>.90** in a full
   observation instrument (Danielson). Our directional latent r = 0.945 is squarely inside this
   envelope, not an outlier.
3. **Diagnosis-as-judgment is content/task-conditional (Südkamp).** Judgment accuracy is *higher
   when informed about the specific task* — the empirical fingerprint of content-dependence.
4. **The clean separations in the literature are affect and scaffolding, not diagnosis** (CLASS,
   3BD, INSPIRE, VanLehn granularity) — matching our own result (scaffolding separates; affect is
   the orphan candidate; content↔diagnosis fuse).

**Implication for keeping `diagnosis` vs folding it into a competence axis.** The literature
**supports keeping `diagnosis` as a reported/labeled construct** (it is a genuine, field-recognized
tutoring competency, and external LLM-tutor rubrics score it — memo 03) **but does NOT support
expecting it to hold up as a separate *latent dimension* on a bundled-item instrument measuring
capable tutors.** This is fully consistent with the companion decision: **let the full-matrix fit
collapse content+diagnosis into one "correctness/diagnosis" competence axis**, while continuing to
**report diagnosis descriptively** (via `use_case`/tagging) so the benchmark can still speak to
error-reading as a skill. The literature moves the prior *toward* collapse-of-the-latent-axis and
*against* narrowing/deleting the label. Danielson is the precedent: keep the rubric language,
report the collapsed dimension.

---

## (e) Definitional-boundary suggestions (explicit inclusion/exclusion rules)

Goal: raise `diagnosis` rater reliability above the κ = 0.365 floor **without** changing the
conservative v2 intent. The literature suggests the reliability problem is *under-anchoring of a
subtle, content-entangled construct* (MET: even trained observers need anchors + multiple raters;
Südkamp: accuracy rises with task-specific information; noticing: hard to score without protocol).
Concrete rules:

1. **Separate "attend/interpret" from "decide/respond" (van Es & Sherin / Jacobs boundary).** Load
   `diagnosis` for *reading* the student's specific error (attend + interpret). Do **not** let the
   *response* move (hint, correction delivery) pull the label — that is `scaffolding`/`content`.
   This directly attacks the audit's core ambiguity ("does acting on the error count?").
2. **Require a concrete referent (anti-vagueness anchor).** `diagnosis = 1` only if the criterion
   points to a **specific, identifiable** student object — a named step, value, term, or
   proposition ("the student used 0.0017 for [H₂O₂]", "added the denominators"). A generic
   "address the student's error" with no locatable referent is a **weaker** load; keep v2's broad
   rule but tag these "broad" so raters aren't forced to guess specificity.
3. **Codify the acknowledgement exclusion (the single biggest κ driver, already v2).** Merely
   *acknowledging a feeling* ("acknowledge the student's confusion") is **all-zero**, not
   diagnosis — reinforced by Südkamp's *state-regulation vs. sentiment* distinction and Tutor
   CoPilot's "generic encouragement = low quality." Give raters 2–3 near-verbatim worked examples
   on each side of this line.
4. **Informed-judgment framing for raters (Südkamp moderator).** Give raters the *correct answer
   and the student's work* alongside the criterion, since diagnosis accuracy is empirically higher
   when the judge is task-informed; withholding it manufactures avoidable disagreement.
5. **Domain-independence probe for the OTHER skills (keep boundaries crisp).** For any criterion,
   ask "could I decide pass/fail without knowing the subject-matter answer?" — if **no**, it is
   content and/or diagnosis, not scaffolding/affect. This keeps the clean axes clean.
6. **Reliability by aggregation, not redefinition (MET).** Expect single-rater κ on diagnosis to
   stay modest; drive reliability with **≥3 raters + adjudication** and the anchored examples
   above, rather than by re-drawing the construct. Report agreement *after* anchoring.

---

## (f) Where the literature is mixed / our small-N data can't adjudicate

**Well-established (external, verified):**
- Diagnosis-type knowledge (PCK-misconceptions, noticing, judgment accuracy) is a **named,
  field-recognized construct** — keeping a `diagnosis` label is well-motivated.
- CK↔PCK are **distinguishable in principle and predictively** (Baumert et al. 2010) but
  **empirically near-collinear**, trending to r≈1 with expert respondents + bundled items
  (COACTIV) and collapsing to one factor in a full observation instrument (Danielson r>.90).
- The **cleanly separable** teaching-quality factors are **affect** and **scaffolding/cognitive
  activation**, not diagnosis (CLASS, 3BD, VanLehn).
- Low single-rater reliability on subtle constructs is **normal** and remediated by
  raters/anchors, not construct deletion (MET).

**Mixed / contested in the literature itself:**
- **How separable CK/PCK are depends entirely on operationalization** (Depaepe et al. 2013;
  integrative vs transformative). TEDS-M individual-level r ranges ~.3–.6 depending on controls;
  elementary specialized-CK is *inseparable* from PCK. So the literature does **not** give a single
  separability number — it gives a *range keyed to method*, and our method sits at the fused end.
- **The "right" number of teaching-quality dimensions is unsettled** (3BD vs Kleickmann's 4th
  cognitive-support dimension) — a caution against over-confidence in *any* fixed dimensionality,
  including ours.

**Where our small-N data cannot adjudicate (flag before quoting):**
- Our latent r = 0.945 is **N≈28, ~1/4 matrix, biased weak-model fleet** — directional only. The
  literature tells us this magnitude is *plausible* for a bundled/expert instrument, but it does
  **not** prove the true latent structure; only the full-matrix M2PL + EFA + AIC/BIC (companion
  memo §7 rule) can.
- Whether the **separable error-detection subset** (the ~399 "diagnosis-without-content"
  items, companion collinearity §1 group 1) is *strong and prevalent enough* to identify a thin
  third axis is unanswerable from present data and from the literature — it is an empirical
  question for the powered fit.
- External LLM-tutor rubrics *do* score diagnosis standalone (memo 03) but on **~100%-base-rate
  mistake datasets**, so their success does **not** transfer to our mixed bank as evidence of
  empirical separability.

---

## Appendix — sources

**Verified this pass (author/venue/year/link confirmed via web lookup):**
- **Shulman, L. S.** (1986). Those who understand: Knowledge growth in teaching. *Educational
  Researcher* 15(2):4–14. — (1987). Knowledge and teaching. *Harvard Educational Review* 57(1):1–22.
  *(framework carried from companion memos; foundational.)*
- **Baumert, J., Kunter, M., et al.** (2010). Teachers' Mathematical Knowledge, Cognitive
  Activation in the Classroom, and Student Progress. *American Educational Research Journal*
  47(1):133–180. doi:10.3102/0002831209345157. (verified via Praetorius 2018 reference list.)
- **Depaepe, F., Verschaffel, L., & Kelchtermans, G.** (2013). Pedagogical content knowledge: A
  systematic review… *Teaching and Teacher Education* 34:12–25. doi:10.1016/j.tate.2013.03.001.
- **TEDS-M:** Kaiser & Blömeke lineage — CK vs PCK two-dimensional structure, Taiwan–Germany
  study, *Teaching and Teacher Education* (2015), doi:10.1016/j.tate.2014.09.003; TEDS-M
  correlation-methods paper (MCK/MPCK r≈.6 individual, ≈.7 program), *Large-scale Assessments in
  Education* 1:7, doi:10.1186/2196-0739-1-7; Blömeke et al. (2014) / Copur-Gencturk et al. (2019)
  on elementary specialized-CK ≈ PCK inseparability (via ScienceDirect S0742051X14001498).
- **Südkamp, A., Kaiser, J., & Möller, J.** (2012). Accuracy of teachers' judgments of students'
  academic achievement: A meta-analysis. *Journal of Educational Psychology* 104(3):743–762.
  doi:10.1037/a0027627. (r=.63; informed>uninformed.) Replication: *PLOS ONE* (2024),
  doi:10.1371/journal.pone.0307594 (mean r≈.58).
- **van Es, E. A., & Sherin, M. G.** (2002). Learning to notice… *Journal of Technology and
  Teacher Education* 10(4):571–596. — **Jacobs, V. R., Lamb, L. L. C., & Philipp, R. A.** (2010).
  Professional noticing of children's mathematical thinking. *JRME* 41(2):169–202 (verified via
  ZDM review doi:10.1007/s11858-020-01216-z).
- **VanLehn, K.** (2011). The relative effectiveness of human tutoring, ITS, and other tutoring
  systems. *Educational Psychologist* 46(4):197–221. doi:10.1080/00461520.2011.611369. (human
  d=0.79, ITS d=0.76; interaction granularity.)
- **Praetorius, A.-K., Klieme, E., Herbert, B., & Pinger, P.** (2018). Generic dimensions of
  teaching quality: the German framework of Three Basic Dimensions. *ZDM* 50(3):407–426.
  doi:10.1007/s11858-018-0918-4. — **Kleickmann et al.** (2020), 4th "cognitive support" dimension
  (pedocs 25862). — origin: Klieme, Schümer & Knoll (2001), TIMSS-video factor analysis.
- **CLASS:** Hamre, Pianta, et al. (2013). *Elementary School Journal* 113(4):461–487.
  doi:10.1086/669616; 26-matrix meta-analysis doi:10.1080/00220973.2018.1551184 (verified in
  companion memo 02).
- **Danielson FfT validity:** REL West / IES (2016). Examining the validity of ratings from a
  classroom observation instrument… REL 2016-135 (Washoe County, 713 teachers; 4-factor CFA
  fit but inter-factor r>.90, "nearly redundant," recommend single rating).
- **MET project:** Kane, T. J., & Staiger, D. O. (2012). Gathering Feedback for Teaching. Bill &
  Melinda Gates Foundation (MET Project). (Instruments mutually correlated; single-observer
  unreliable; combine measures + multiple raters → reliability ~.65–.67.)

**Carried from companion memos (not independently re-verified this pass):**
- **Krauss, Baumert, & Blum** (2008), *ZDM* 40(5):873–892 (COACTIV CK–PCK manifest r=0.60; latent
  ≈1 in academic-track). — **Krauss et al.** (2008), *J. Educational Psychology* 100(3):716–725.
- **Chi, M. T. H., & Wylie, R.** (2014). ICAP. *Educational Psychologist* 49(4):219–243.
- **Graesser, A. C., et al.** (2004). AutoTutor (EMT dialogue). *Behavior Research Methods*
  36:180–192.
- **Lepper, M. R., & Woolverton, M.** (2002). The Wisdom of Practice (INSPIRE). In *Improving
  Academic Achievement*, pp. 135–158. doi:10.1016/B978-012064455-1/50010-5.

**Local:** `Skill Definitions v2 (Q-matrix + Judge).md`; `Collinearity Analysis + Skill-Definition
Options.md` (r=0.945, N≈28; 1.7%; marginals; P(content|diagnosis)=81.9%); `Tutor Skill Taxonomy -
Theory + Candidate Dimensions.md`; `plans+prds/research/02 Candidate Additional Dimensions.md`;
`plans+prds/research/03 LLM-Tutor Evaluation Rubrics.md`.

**Citation-verifiability notes:**
- COACTIV's "latent CK–PCK ≈ 1 in academic-track teachers" is cited **from the companion memo**
  (Krauss et al. 2008); I confirmed the surrounding COACTIV/Baumert facts this pass but did **not**
  re-open the ZDM 40(5) article to re-verify that specific latent figure — treat as high-confidence
  but memo-sourced.
- MET reliability figures (~.65–.67; combined-measure ranges) are verified from the MET
  practitioner brief / "Gathering Feedback" report text; the exact per-instrument value-added
  reliabilities (e.g., 0.48 math / 0.20 ELA) are quoted from those PDFs and should be re-checked
  against the original if quoted precisely.
- Danielson r>.90 collapse is from the **single-district** REL 2016-135 study (Washoe County); the
  UAE validation (Educ. Sci. 2024) reports good 4-domain fit — so the *collapse* result is
  context-specific, not universal (see §f "mixed").
- ICAP, AutoTutor-EMT, INSPIRE, and Krauss COACTIV latent-≈1 were **not re-fetched this pass**
  (carried from companion memos 02/03, which verified several of them).
