# 03 — LLM-Tutor Evaluation Rubrics: How the Field Decomposes Tutoring Skill

**Status:** literature/research memo (thread 3 of 3). **Changes nothing** — no scenario, rubric,
`q_mapping`, skill definition, or code is touched; the full TutorBench dataset was **not**
downloaded (only `data/scenarios.jsonl` was read for scenario shape). All external claims are
web-sourced with citable links (§7). Our own numbers (r = 0.945, 1.7%, N≈28) come from the
companion memos and are directional.

**Purpose.** Survey how the contemporary (2023–2026) LLM-tutor / AI-tutoring **evaluation**
literature breaks tutoring skill into *scored dimensions*, and crosswalk those onto our three
latent skills — **content** (domain correctness), **diagnosis** (reading *this* student's
specific error/confusion), **scaffolding** (structuring/sequencing pedagogical help) — to inform
two live decisions: (i) whether **diagnosis** deserves to remain its own dimension, and (ii)
whether we are omitting any dimension the field consistently scores.

---

## (a) Executive summary

1. **Every major external rubric decomposes tutoring into 4–9 scored dimensions, and all of them
   contain recognizable analogues of our three skills** — a *correctness* axis (content), a
   *mistake/misconception* axis (diagnosis), and a *guidance/withhold-the-answer* axis
   (scaffolding). Our three skills are not idiosyncratic; they recur under different names in
   LearnLM, MRBench, MathDial, Bridge, Tutor CoPilot, and TutorBench's own tags.

2. **Diagnosis is treated as its OWN scored dimension, separate from correctness, in the majority
   of independent rubrics — this is the single most decision-relevant finding.** MRBench scores
   *mistake identification* and *mistake location* as two dedicated dimensions distinct from
   *providing (correct) guidance*; Bridge makes "identify the student's error" the first explicit
   stage of its decision model, separate from "strategy"; TutorBench tags *identifying core
   misconceptions* and *recognizing correct/incorrect steps* as tutoring-skills distinct from
   *truthfulness* and *stating definitions/theorems*; LearnLM scores *guide mistake discovery* /
   *identify and address misconceptions*. **The field's consensus is that error-reading is a
   first-class evaluation target, not a byproduct of being correct.** (Caveat: those benchmarks
   are built on datasets where *every* item contains a known student mistake, so their base rate
   for diagnosis is ~100% by construction — different from our mixed bank.)

3. **The dimension we most conspicuously omit — and that recurs across the most independent
   rubrics — is tutor tone / motivation / affect.** It appears as a first-class scored axis in
   LearnLM ("motivate and stimulate curiosity", "communicate with positive tone"), MRBench
   ("tutor tone"), TutorBench ("style/tone", "emotional component"), Tutor CoPilot ("encourage
   generically" — notably scored as *low*-quality when non-specific), and Bridge ("care"). This
   corroborates the companion taxonomy memo's finding that **motivation/affect is our strongest
   unmodeled-orphan candidate** (currently all-zero in our bank).

4. **Two further recurring dimensions we don't isolate:** *actionability* (is it clear what the
   student should do next — MRBench, TutorBench, LearnLM "guides appropriately") and
   *coherence / human-likeness* (MRBench, Wang-et-al. lineage). *Adaptivity / student-level
   calibration* is also widely scored (LearnLM, TutorBench) but our companion memo already
   decided to treat it as a **reporting facet**, not a dimension.

5. **Withholding-the-answer / not-revealing is the most universal scaffolding sub-criterion** in
   the field (LearnLM, MRBench, MathDial, Tutor CoPilot, Pedagogical-Alignment all score it), which
   validates scaffolding as our cleanest, most consensus-backed axis.

---

## (b) The big crosswalk

**Legend for mapping columns.** ● = the external dimension maps *primarily* to our skill;
◐ = partial / secondary load; ○ = not this skill. "Other/uncovered" flags dimensions with **no
home** in {content, diagnosis, scaffolding} (i.e., candidate omissions).

### B.1 Per-rubric dimension inventory, mapped to {content / diagnosis / scaffolding / other}

| External benchmark / rubric | Its scored dimension (verbatim) | content | diagnosis | scaffolding | other / uncovered |
|---|---|:--:|:--:|:--:|---|
| **LearnLM pedagogy rubric** (Google DeepMind) | Manage cognitive load (stay on topic, appropriate length, manageable chunks, straightforward) | ○ | ○ | ● | ◐ conciseness/pacing |
| | Encourage active learning (do not reveal answer; guide toward answer; ask questions; promote engagement) | ○ | ○ | ● | — |
| | Deepen metacognition (guide mistake discovery; constructive feedback; acknowledge correctness; communicate plan; **identify & address misconceptions**) | ◐ | ● (mistake discovery/misconceptions) | ◐ | ◐ metacognition |
| | Motivate & stimulate curiosity (positive tone; respond to affect cues; stimulate interest) | ○ | ○ | ○ | **● motivation/affect** |
| | Adapt to learners' goals & needs (adapt to level; adapt to affect; guide appropriately) | ○ | ◐ | ◐ | **● adaptivity** |
| | (Correctness — scored as a separate axis) | ● | ○ | ○ | — |
| **MRBench / Unifying AI-Tutor Eval** (8 dims) | 1. Mistake identification | ◐ | ● | ○ | — |
| | 2. Mistake location | ◐ | ● | ○ | — |
| | 3. Revealing of the answer (desideratum: **No**) | ○ | ○ | ● | — |
| | 4. Providing guidance (correct, relevant hint/explanation) | ● | ◐ | ● | — |
| | 5. Actionability (clear what to do next) | ○ | ○ | ◐ | **● actionability** |
| | 6. Coherence (consistent with student's prior turn) | ○ | ◐ | ○ | **● coherence** |
| | 7. Tutor tone (encouraging / neutral / offensive) | ○ | ○ | ○ | **● motivation/affect** |
| | 8. Human-likeness (natural vs robotic) | ○ | ○ | ○ | **● human-likeness** |
| **MathDial** teacher-move taxonomy (4 moves) | Focus (constrain toward solution) | ○ | ◐ | ● | — |
| | Probing (explore underlying concept) | ○ | ◐ | ● | — |
| | Telling (reveal parts of answer — used when scaffolding fails) | ● | ○ | ◐ (anti-scaffold) | — |
| | Generic (conversational filler) | ○ | ○ | ○ | ● conversational/other |
| **Bridge** decision model (Wang et al.) | (A) Identify the student's **error** (error-type taxonomy) | ◐ | ● | ○ | — |
| | (B) Remediation **strategy** (z_what) | ○ | ◐ | ● | — |
| | (C) **Intention** (z_why) | ○ | ◐ | ● | ◐ |
| | (derived response-eval dims via Wang'24a lineage: care / usefulness / human-likeness) | ○ | ○ | ◐ | **● affect + actionability + human-likeness** |
| **Tutor CoPilot** high/low strategy taxonomy | Prompt Student to Explain (High) | ○ | ○ | ● | — |
| | Ask Question to Guide Thinking (High) | ○ | ◐ | ● | — |
| | Affirm Student's Correct Attempt (High) | ● | ◐ | ○ | — |
| | Ask Student to Retry (Low) | ○ | ○ | ◐ | — |
| | Provide the Answer or Explanation (Low) | ● | ○ | ◐ (anti-scaffold) | — |
| | Provide Problem-Specific Solution Strategy (Low) | ● | ○ | ◐ (anti-scaffold) | — |
| | Encourage Student in Generic Way (Low) | ○ | ○ | ○ | **● motivation/affect (scored *low* when generic)** |
| **Pedagogical Alignment of LLMs** (Sonkar et al.) | Scaffolded guidance vs direct answer (single perplexity-based axis) | ◐ | ○ | ● | — |
| **TutorBench** — evaluation-dimension tags (8) | Instruction following | ○ | ○ | ◐ | **● instruction-following** |
| | Truthfulness | ● | ○ | ○ | — |
| | Style & tone | ○ | ○ | ○ | **● motivation/affect** |
| | Conciseness & relevance | ○ | ○ | ◐ | ◐ conciseness/clarity |
| | Student-level calibration | ○ | ◐ | ◐ | **● adaptivity** |
| | Emotional component | ○ | ○ | ○ | **● motivation/affect** |
| | Visual reasoning / Visual perception | ◐ | ○ | ○ | ● multimodal (out of our text scope) |
| **TutorBench** — tutoring-skill tags (7) | Identifying core misconceptions | ◐ | ● | ○ | — |
| | Recognizing correct/incorrect student steps | ● | ● | ○ | — |
| | Asking guiding questions | ○ | ○ | ● | — |
| | Including examples or analogies | ◐ | ○ | ● | ◐ adaptivity |
| | Providing alternative solutions | ● | ○ | ● | — |
| | Stating definitions/theorems (knowledge) | ● | ○ | ○ | — |
| | Providing step-by-step help | ◐ | ○ | ● | — |

### B.2 Condensed view — which of OUR skills each rubric covers, and what it adds

| Rubric | content covered? | diagnosis covered as its **own** dim? | scaffolding covered? | Dimensions it scores that we OMIT |
|---|:--:|:--:|:--:|---|
| LearnLM | ✔ (correctness axis) | ✔ (mistake discovery / misconceptions, under "metacognition") | ✔✔ (active learning + cognitive load) | motivation/affect, adaptivity, metacognition, curiosity, conciseness |
| MRBench | ✔ (via guidance) | **✔✔ (two dedicated dims: identification + location)** | ✔✔ (reveal-answer + guidance) | tone/affect, actionability, coherence, human-likeness |
| MathDial | ◐ (ground-truth grounded) | ◐ (confusion annotated, not a scored dim) | ✔✔ (Focus/Probing/Telling) | conversational "Generic" bucket |
| Bridge | ◐ | **✔✔ (Step A = identify error, first-class stage)** | ✔ (strategy + intention) | affect ("care"), human-likeness, actionability ("usefulness") |
| Tutor CoPilot | ✔ (affirm/provide answer) | ◐ (implicit in affirming correctness) | ✔✔ (the whole high/low axis) | motivation ("generic encouragement", scored *negatively*) |
| Pedagogical Alignment | ◐ | ○ | ✔✔ (the entire metric) | — (single-axis) |
| **TutorBench (native tags)** | **✔ (truthfulness + knowledge)** | **✔✔ (identify misconceptions + recognize steps)** | **✔✔ (guiding questions, step-by-step, alternatives, examples)** | tone, emotional component, calibration, conciseness, instruction-following |

---

## (c) Dimensions that RECUR across multiple independent rubrics (real signal) vs one-off

**Strongly recurring (≥4 independent rubrics) — high confidence these are "real" tutoring axes:**

| Recurring dimension | Where it appears | Maps to our skill |
|---|---|---|
| **Withhold / don't reveal the answer** | LearnLM, MRBench, MathDial (Telling=failure mode), Tutor CoPilot (Provide-Answer=Low), Pedagogical-Alignment | **scaffolding** (core) |
| **Guiding questions / prompt reasoning** | LearnLM, MRBench (guidance), MathDial (Probing/Focus), Tutor CoPilot (Ask/Prompt), TutorBench (guiding questions) | **scaffolding** (core) |
| **Mistake identification / location** | MRBench (×2), Bridge (Step A), TutorBench (misconceptions + steps), LearnLM (mistake discovery) | **diagnosis** (standalone — see (d)) |
| **Correctness / truthfulness** | LearnLM, MRBench (via guidance), MathDial (ground truth), Tutor CoPilot, TutorBench (truthfulness) | **content** (core) |
| **Tutor tone / encouragement / affect** | LearnLM, MRBench, TutorBench (style + emotional), Tutor CoPilot, Bridge (care) | **OMITTED (other)** — see (e) |

**Moderately recurring (2–3 rubrics):**

| Dimension | Where | Maps to |
|---|---|---|
| Step-by-step help / providing guidance | LearnLM, MRBench, TutorBench, Tutor CoPilot | scaffolding |
| Adaptivity / student-level calibration | LearnLM, TutorBench, (Bridge context-sensitivity) | omitted → we treat as **reporting facet** |
| Actionability (clear next step) | MRBench, TutorBench, LearnLM ("guides appropriately"), Bridge ("usefulness") | **OMITTED (other)** |
| Conciseness / length / cognitive load | LearnLM, TutorBench | partial scaffolding / omitted clarity |
| Metacognition (guide reflection/self-check) | LearnLM (own axis), MRBench (mistake discovery adjacent) | our bank folds into scaffolding |

**One-off (single rubric / lineage) — weaker signal:**

- **Human-likeness / naturalness** — MRBench + Wang-2024a lineage only.
- **Coherence** (consistency with prior turn) — MRBench primarily.
- **Curiosity / interest stimulation** — LearnLM only.
- **Instruction-following** — TutorBench only (a general-purpose LLM-eval tag, not tutor-specific).
- **Visual reasoning / perception** — TutorBench only (multimodal; out of our text-only scope).

---

## (d) Is "mistake identification / location" scored as its OWN dimension, separate from correctness? — **YES, and this is the strongest external support for keeping `diagnosis`.**

The literature **overwhelmingly separates error-reading from being correct**:

- **MRBench (Maurya et al., NAACL 2025)** — the clearest case. Its 8-dimension taxonomy makes
  **"Mistake identification"** (dim 1: *has the tutor identified a mistake?*) and **"Mistake
  location"** (dim 2: *does the response accurately point to the genuine mistake and its
  location?*) **two dedicated dimensions**, explicitly distinct from **"Providing guidance"**
  (dim 4, which carries the correctness/helpfulness load). The paper says mistake identification
  "corresponds to *student understanding* in Tack & Piech (2022) and *correctness* in Macina et
  al. (2023) and Daheim et al. (2024)" — i.e., the field's *correctness* label was, in earlier
  schemata, doing double duty for "spot the student's error," and MRBench deliberately splits it
  out.

- **Bridge (Wang et al., NAACL 2024)** — the decision model's **Step (A) "identify the student's
  error"** is a distinct, separately-annotated stage (with its own error-type taxonomy `e`),
  upstream of and separate from **(B) strategy** and **(C) intention**. Their headline finding —
  "better-ranked models engage more with the **student's process**, rather than just the
  student's **answer**" — is precisely a content-vs-diagnosis distinction, and they show
  supplying the error/strategy decision makes GPT-4 responses **+76%** more preferred.

- **TutorBench (Scale AI, 2025)** — its own *tutoring-skill* tags list **"identifying core
  misconceptions"** and **"recognizing correct or incorrect student steps"** as skills separate
  from **"stating definitions or theorems (knowledge)"**, and separate again from the
  *evaluation-dimension* tag **"truthfulness."** So the benchmark our bank is drawn from **itself
  distinguishes diagnosis from content at the tagging level** (a distinction our final
  `q_mapping` collapsed away — see `TutorBench Use Cases + Adaptability.md` §3).

- **LearnLM** — scores **"Guide Mistake Discovery"** and **"identify and address misconceptions"**
  as rubric items (under *deepen metacognition*), separate from its correctness axis.

**Implication for our decision.** Conceptually, the field is unanimous that **diagnosis is a
legitimate standalone scored construct** — this is external validation for *defining* it as a
skill. **However**, this does *not* by itself resolve our empirical collinearity problem, for two
reasons the companion memos already flagged: (i) these benchmarks operate on **mistake-remediation
datasets where every item contains a known student error**, so diagnosis has a ~100% base rate and
co-varies less with raw content than in our mixed bank; and (ii) MRBench itself notes "inherent
interdependencies among the dimensions." So the literature supports **keeping diagnosis as a
conceptual/labeling axis** while remaining agnostic about whether it is *empirically separable* on
our specific instrument (where content↔diagnosis latent r ≈ 0.945). The two questions —
"is diagnosis a real dimension in the field?" (yes) and "is it separable in our N≈28 fit?" (no) —
have different answers, and the field speaks only to the first.

---

## (e) Dimensions commonly scored that we currently OMIT

Ranked by cross-rubric recurrence:

1. **Tutor tone / motivation / affect — scored in 5 of 7 rubrics (LearnLM, MRBench, TutorBench,
   Tutor CoPilot, Bridge).** This is our single biggest omission. Two nuances worth importing:
   - LearnLM and MRBench treat *encouraging tone* as a **positive** desideratum.
   - **Tutor CoPilot scores *generic* encouragement as a *low-quality* strategy** ("That's a good
     try!" with no specificity), and its RCT found treatment tutors *reduced* generic praise. This
     is a direct external echo of our v2 rule that **generic acknowledgement/empathy is all-zero**
     — i.e., the field also distinguishes *specific, contingent* affect support from cheap praise.
   - Corroborates the companion `Tutor Skill Taxonomy` memo: motivation/affect is the strongest
     domain-independent, currently-unmodeled ("orphan") candidate for a future axis.

2. **Actionability — "is it clear what the student should do next?" (MRBench dim 5; TutorBench
   "conciseness & relevance"/implicit; LearnLM "guides appropriately"; Bridge "usefulness").**
   We have no dedicated construct for this; it partially rides inside scaffolding. Recurs in ≥3
   independent rubrics.

3. **Adaptivity / student-level calibration (LearnLM, TutorBench "student-level calibration").**
   Widely scored, but we have *already decided* (companion memo) to treat this as a **`use_case`
   reporting facet**, not a latent dimension, because its criteria are absorbed into
   scaffolding+diagnosis. No change recommended; noted here only for completeness.

4. **Coherence / conversational consistency (MRBench dim 6).** Mostly one rubric; low priority.

5. **Human-likeness / naturalness (MRBench dim 8; Wang-2024a lineage).** Mostly the MRBench/Wang
   lineage; arguably an artifact of dialogue-turn evaluation and low priority for a rubric-scored
   response benchmark like ours.

6. **Metacognition (LearnLM's own axis).** Conceptually distinct but, per the companion taxonomy
   memo, our reflection/self-check items map to **scaffolding** and would likely re-collide.

**Net recommendation for this thread (consistent with companion memos):** the only omission worth
*pre-registering a test* for is **motivation/affect** (recurs most, domain-independent,
currently all-zero orphan). *Actionability* is a credible second candidate but is more entangled
with scaffolding. Everything else is either already handled (adaptivity = facet), likely to
re-collide (metacognition), or low-signal/one-off (coherence, human-likeness).

---

## (f) Citations — established/peer-reviewed vs preprint/blog

### Peer-reviewed / archival (*ACL-anthology or equivalent*)

- **Macina, J., Daheim, N., Chowdhury, S., et al. (2023). "MathDial: A Dialogue Tutoring Dataset
  with Rich Pedagogical Properties Grounded in Math Reasoning Problems." *Findings of EMNLP
  2023*.** Teacher-move taxonomy (Focus / Probing / Telling / Generic). Inter-annotator κ = 0.60;
  **merging Focus+Probing into a single "scaffolding" category raised agreement to κ = 0.67 /
  0.75 / 0.55** (Probing vs Focus hard to distinguish). https://aclanthology.org/2023.findings-emnlp.372/
- **Wang, R. E., Zhang, Q., Robinson, C., Loeb, S., & Demszky, D. (2024). "Bridging the
  Novice-Expert Gap via Models of Decision-Making: A Case Study on Remediating Math Mistakes."
  *NAACL 2024*.** Decision model = (A) student error, (B) remediation strategy, (C) intention;
  700 expert-annotated conversations; response quality via human pairwise preference.
  https://aclanthology.org/2024.naacl-long.120/ · arXiv:2310.10648
- **Maurya, K. K., et al. (2025). "Unifying AI Tutor Evaluation: An Evaluation Taxonomy for
  Pedagogical Ability Assessment of LLM-Powered AI Tutors." *NAACL 2025* (long).** Introduces
  **MRBench** (192 conversations, 1,596 responses, 7 tutors) and the **8-dimension taxonomy**
  (mistake identification, mistake location, revealing of the answer, providing guidance,
  actionability, coherence, tutor tone, human-likeness), each rated Yes / To-some-extent / No.
  **Reliability approach:** gold human annotations; evaluates **Prometheus2** and
  **Llama-3.1-8B** as automatic evaluators against them. https://aclanthology.org/2025.naacl-long.57/
  · arXiv:2412.09416
- **Sonkar, S., et al. (2024). "Pedagogical Alignment of Large Language Models." *Findings of
  EMNLP 2024*.** Single scored axis — *scaffolded guidance vs. direct answers* — via a
  perplexity-based metric and pedagogical-alignment accuracy/F1; LHP (DPO/IPO/KTO) vs SFT.
  https://aclanthology.org/2024.findings-emnlp.797/ · arXiv:2402.05000

### Preprint / tech report / working paper / blog

- **Jurenka, I., et al. / Google DeepMind (2024). "Towards Responsible Development of Generative AI
  for Education: An Evaluation-Driven Approach."** arXiv:2407.12687 (preprint; revised 2025 after
  peer feedback). Introduces **LearnLM-Tutor** and the **high-level pedagogy rubric — 5
  principles**: *encourage active learning; manage cognitive load; deepen metacognition; motivate
  & stimulate curiosity; adapt to learners' goals and needs* (+ **correctness** as a separate
  axis). Each principle broken into measurable items across automatic (LLM-critic / "LME"),
  conversation-level, and turn-level human evaluations. https://arxiv.org/abs/2407.12687
- **LearnLM Team / Google (2024). "LearnLM: Improving Gemini for Learning."** arXiv:2412.16429
  (tech report). Reports the operational rubric sub-dimensions used by experts on a 7-point scale
  across **~29 rubric questions**, e.g. Cognitive Load {appropriate response length, manageable
  chunks, straightforward response, stay on topic}; Active Learning {do-not-reveal-answer,
  guides-to-answer, asks questions, active engagement}; Metacognition {guide mistake discovery,
  constructive feedback, acknowledge correctness, communicates plan}; Curiosity {stimulates
  interest}; Adaptivity {adapts to affect, adapts to needs, guides appropriately}.
  https://arxiv.org/abs/2412.16429
- **Wang, R. E., Ribeiro, A. T., Robinson, C. D., Loeb, S., & Demszky, D. (2024/2025). "Tutor
  CoPilot: A Human-AI Approach for Scaling Real-Time Expertise."** arXiv:2410.03017; Annenberg/Brown
  EdWorkingPaper 24-1056. Behavior taxonomy scored by RoBERTa classifiers over 240k+ tutor
  messages — **High:** Prompt Student to Explain (F1 0.89), Ask Question to Guide Thinking (0.90),
  Affirm Student's Correct Attempt (0.65); **Low:** Ask Student to Retry (0.73), Provide the
  Answer or Explanation (0.76), Provide Problem-Specific Solution Strategy (0.76), Encourage
  Student in Generic Way (0.81). RCT: +4 p.p. topic mastery; treatment tutors used more probing
  questions and *less generic praise*. https://arxiv.org/abs/2410.03017
- **Scale AI (TutorBench team) (2025). "TutorBench: A Benchmark To Assess Tutoring Capabilities Of
  Large Language Models."** arXiv:2510.02663; OpenReview `NIhIpxykLK`; dataset `ScaleAI/TutorBench`;
  blog labs.scale.com/leaderboard/tutorbench. 1,490 samples across three use cases; **15,220
  sample-specific rubric criteria** (3–39/example), pass/fail with weights **{+5, +1, −5}**. Two
  tag axes: **evaluation dimensions** {instruction following, style & tone, truthfulness, visual
  reasoning, visual perception, conciseness & relevance, student-level calibration, emotional
  component} and **tutoring skills** {asking guiding questions, identifying core misconceptions,
  recognizing correct/incorrect steps, including examples/analogies, providing alternative
  solutions, stating definitions/theorems, providing step-by-step help}; plus explicit/implicit and
  objective/subjective. **Reliability:** inter-human agreement **0.75**; LLM-judge (Claude-4-Sonnet)
  vs human **0.78**; majority-label F1 **0.81** (F1 on critical rubrics **0.82**).
  https://arxiv.org/abs/2510.02663

---

## (g) Reliability / agreement approaches at a glance

| Rubric | Rating scale | Agreement / reliability method | Reported figure |
|---|---|---|---|
| MathDial | move labels | Cohen's κ, 2 annotators + teacher | κ = 0.60 (pairs); Focus/Probing hard to split; merged "scaffolding" → κ 0.67/0.75/0.55 |
| MRBench | Yes / to-some-extent / No | Gold human labels; test Prometheus2 & Llama-3.1-8B as evaluators | (per-dimension evaluator reliability; DAMR analyses) |
| Bridge | pairwise human preference | Human preference win-rate | +76% preferred w/ expert decisions; −97% w/ random |
| Tutor CoPilot | binary behavior classifiers | RoBERTa F1 vs verified labels | F1 0.65–0.90 per strategy |
| Pedagogical Alignment | accuracy / F1 + perplexity | Held-out pedagogical-alignment acc/F1 | +13.1% / +8.7% / +50% over SFT |
| LearnLM | 7-point expert scale + LLM critics | Expert human eval + LLM-critic (LME) triangulation | large effect sizes; 29 rubric qs |
| **TutorBench** | pass/fail weighted (+5/+1/−5) | Inter-human vs LLM-judge agreement | human 0.75 · judge 0.78 · F1 0.81–0.82 |

---

## Notes on citation verifiability

- **Author lists partially unverified.** For **MRBench** ("Maurya et al., 2025"),
  **Pedagogical Alignment** ("Sonkar et al., 2024"), **MathDial** (co-authors beyond Macina &
  Daheim), and **LearnLM / Towards-Responsible** ("Jurenka et al." / "LearnLM Team"), I confirmed
  the paper title, venue, year, and content, but did **not** independently verify the complete
  ordered author lists from a canonical bibliographic record — treat first-author attributions as
  high-confidence and the "et al." tails as to-be-checked against the ACL Anthology / arXiv pages
  linked above. **Bridge** and **Tutor CoPilot** author lists were confirmed from source.
- **TutorBench individual authors** were not surfaced (OpenReview submission; cited institutionally
  as "Scale AI"). The paper ID (arXiv:2510.02663), the three use cases, the 15,220-criterion
  count, the tag taxonomies, and the agreement figures (0.75 / 0.78 / 0.81) are all verified from
  the paper text and the Scale leaderboard page.
- **"Wang et al. 2024a"** referenced by MRBench for the *care / usefulness / human-likeness*
  dimensions is the **Bridge** lineage (Rose E. Wang et al.); the care/usefulness attribution is
  reported *via MRBench's* text, not read from a separate Wang rubric, so treat that specific
  mapping as second-hand.
