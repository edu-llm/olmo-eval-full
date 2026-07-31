# Gate B — Proposed Criteria / Rubric Revisions (FOR REVIEW)

> ## ⚠️ THESE ARE PROPOSALS. NOTHING HAS BEEN APPLIED.
> - **No** edits to `data/TutorBench/curated/rubrics_qmatrix_curated.jsonl` (the bank) — git: **clean**.
> - **No** edits to `scenarios_curated.jsonl`, `staging/response_matrix.csv`, or `staging/calibration_mirt_full2skill.csv` (fitter inputs) — git: **clean**.
> - **Nothing** committed.
> - Only files written: this memo + `staging/revision_evidence/proposed_criteria_edits.csv` / `.jsonl` (plus read-only analysis helpers in `staging/revision_evidence/_gateb_*.py`).

**Method.** Applied the **Gate A lesson** as the default prior: TutorBench criteria are overwhelmingly `implicit`/`subjective` **by design** — they test whether the tutor *proactively* surfaces a point. A criterion failing across all 82 models usually means the task is legitimately hard, **not** that the criterion is broken. Every statistical flag was treated as a *hypothesis*, verified against the bank text + scenario context, and — where it mattered — the raw per-model response matrix (point-biserial of item-pass vs. model ability) or the ScaleAI/TutorBench design intent. **Default = KEEP.** Only concrete, content-level defects were promoted to an edit.

---

## Executive summary

| Task | Raw flags | Verified genuine action | What survives |
|---|---:|---:|---|
| **1 — reverse-behaving (a<0)** | 60 | **1 reword** (+59 keep/exclude-from-fit) | `MIS_KEYED 0 · JUDGE_INVERTED 1 · DEGENERATE_FIT 29 · FINE 30` |
| **2 — mis_specified (345) + judge_biased (107+3)** | 455 | **~0–1 defect** in a 21-item sample (≈**4.8%** upper bound) | 20/21 KEEP (intentionally implicit / working-as-intended); 1 low-confidence verify |
| **3 — missing_Q_row (640)** | 640 | **~0 text edits**; ~625 **recover** via Q-loading, ~15 trim | presentation-recover ~600 · scaffolding-recover ~11–15 · trim minority |
| **4 — under-labeling (11)** | 11 | **0 add** | all 11 REJECT (no genuine scaffolding elicitation) |

**Bottom line up front:** across **1,166 statistically-flagged rows**, the genuinely-recommended **criterion-text edits number 2** (one medium-confidence judge-anchor reword, one low-confidence answer-key reword pending source verification). Everything else is either **KEEP** (the benchmark working as intended) or a **difficulty-neutral psychometric re-loading** (`add_q` / exclude-from-fit) that does **not** change any criterion text. This mirrors Gate A almost exactly: raw flags are dominated by "hard, implicit, by-design," not "broken."

---

## TASK 1 — Reverse-behaving items (fitted a<0) — the 60

**Reconciliation.** `sharpening_candidates.csv` has 147 rows with `a_value<0`, but only **60** carry the `negative_a` flag (28 correctness + 31 scaffolding + 1 presentation). The other ~87 are low-info scaffolding items whose slightly-negative collapsed-axis loading is not a reverse-behaving flag. **The canonical reverse set = the 60 `negative_a` items**, matching the pack.

**The decisive evidence is the raw point-biserial (r_pb) of item-pass vs. model ability** (mean pass across all items), computed directly from `response_matrix.csv`. If abler models genuinely fail *more*, r_pb is clearly negative with real variance. If r_pb ≥ 0, the negative fitted `a` is an artifact of the 2-skill collapse, not reverse content.

### Classification counts

| Classification | n | Meaning | Recommendation |
|---|---:|---|---|
| **MIS_KEYED** | **0** | polarity inverted / rewards wrong behavior | — |
| **JUDGE_INVERTED** | **1** | judge scores it backwards | reword judge anchor + exclude-from-fit |
| **DEGENERATE_FIT** | **29** | extreme (a≈−4.09, b≈18.3) or near-zero-variance / sparse-cell / no-data artifacts | leave-and-exclude-from-fit |
| **FINE** | **30** | spurious flag; raw r_pb ≥ 0 (positive discrimination) — negative `a` is a collapse artifact | leave-and-exclude-from-fit |

- **No item is mis-keyed.** Not one of the 60 rewards the wrong behavior on inspection.
- **The 6 headline "extreme" items** (`tb_0112_c02`, `tb_0273_c03`, `tb_0274_c07`, `tb_0464_c01`, `tb_0641_c06`, `tb_0043_c01`; all a≈−4.09, b≈18.3) each have exactly **1/82 passers** → the fit blew up on near-zero variance. Their content is legitimate (e.g. `tb_0464_c01` = "acknowledge the student's frustration"). **DEGENERATE_FIT, keep.**
- **28 of 31 flagged "scaffolding-reverse" items actually have positive r_pb** (e.g. `tb_0617_c06` r_pb=+0.70, `tb_0598_c05` +0.73, `tb_0185_c08` +0.69, `tb_0220_c05` +0.70): abler models pass them **more**. These are legitimate hint criteria whose negative 2-skill loading is a **collapse artifact of the weak, correlated scaffolding axis** — the 3-skill run should re-estimate them cleanly. **FINE, keep.**
- **Only 2 items have substantial variance AND strongly-negative r_pb** — the only genuine reverse behavior:

| item | pass | r_pb | verdict | why |
|---|---:|---:|---|---|
| `tb_0532_c06` | 0.63 | −0.57 | **JUDGE_INVERTED → reword + exclude** | Implicit "must NOT falsely credit the student" guard. The student got stuck and never stated the v(t)<0⇒left fact; abler models proactively supply it (good tutoring) and the judge scores presence-of-fact as a fail. Legitimate design; disambiguate for the judge. |
| `tb_0455_c07` | 0.54 | −0.42 | **FINE → exclude** | "Don't state the distance formula without briefly explaining its origin." Terse expert models skip the derivation, so it anti-correlates with raw math ability. Valid pedagogy; reverse is a style/ability confound, not a defect. |

**Reword (the only content edit from Task 1) — `tb_0532_c06`:**
- **BEFORE:** *"The response must not state that the student has correctly identified that the particle moves to the left when v(t) < 0."*
- **AFTER:** *"The response must not falsely credit the student with having determined that the particle moves left when v(t)<0 (the student got stuck before reaching this and never stated it). Stating the v(t)<0 → leftward-motion fact itself, e.g. as a hint, is acceptable; only ATTRIBUTING that (unmade) determination to the student fails this criterion."*
- *Confidence: medium. Separates "mention the fact" (allowed) from "credit the student" (not allowed) so the judge stops penalizing good tutoring. Also exclude from the 2-skill fit.*

**Net Task 1: 1 reword, 59 leave-and-exclude-from-fit, 0 drops.** These are difficulty-neutral psychometric hygiene, exactly as billed — but the hygiene is *"exclude the artifacts from the fit,"* not *"rewrite the bank."*

---

## TASK 2 — mis_specified (345) + judge_biased (107) + judge_biased_negative (3)

**Structural tell first:** **all 345 mis_specified are `objective + implicit`**, and 82/107 judge_biased are `implicit`. That is *precisely* the Gate A "intentionally-implicit proactive-teaching target" signature. The cluster label "mis_specified" is a misnomer for "the tutor didn't proactively surface an implicit point."

**Sample:** 21 items, stratified across all 6 subjects, all 3 use-cases, and both criticality levels (14 mis_specified spanning subjects, 4 judge_biased, 3 judge_biased_negative). Each verified against criterion text + scenario prompt + sibling criteria (and math checked where possible).

### Result: measured true-defect rate ≈ **0–4.8%** (0 clear defects, 1 borderline in 21)

| verdict | n | examples |
|---|---:|---|
| **KEEP — intentionally implicit** (proactive-teaching target; coherent, answerable prompt) | 17 | `tb_0155_c07`, `tb_0020_c10`, `tb_0243_c06`, `tb_0141_c05` (correct rounding trick), `tb_0006_c08` (analogy), `tb_0002_c15` (bond strength HI<HCl<HF — *correct* content, mis-labeled subjective) |
| **KEEP — working-as-intended negative** (hint-task "don't give away the answer"; all models over-helped) | 3 | `tb_0528_c11`, `tb_0550_c05`, `tb_0569_c08` |
| **POSSIBLE DEFECT — verify against source** (low confidence) | 1 | `tb_0370_c11` |

**The single candidate — `tb_0370_c11`** (calculus feedback, "must explicitly state x=11.75 in"): the optimization ((x−2)(y−4)=30, minimize xy) gives one dimension ≈5.87 and the other ≈11.75. The rubric's answer-key criteria assign **x=11.75 (x=height)** while the **student prompt defines x=width**, and sibling `tb_0370_c13` ("A=11.75×5.87 ≠ 56 in²") is internally garbled. The variable-convention clash is the likely reason for 0% pass.
- **Proposed (low confidence, verify first):** name the *dimension* not the ambiguous variable — *"…state the optimal page HEIGHT (~11.75 in) and WIDTH (~5.87 in) using one consistent variable convention."*
- **Caveat:** under the rubric's own `x=height` convention the value is internally consistent, so this may be **working-as-intended** for a messy feedback item. **Verify against `ScaleAI/TutorBench` (match by content) before any edit.**

### Extrapolation to the full clusters (with caveat)
A 4.8% upper-bound sample rate over the **452** mis_specified + judge_biased items ⇒ on the order of **~0–22 genuine defects**, most likely **<15**, and even those are "verify against source," not "auto-rewrite." **Do NOT mass-rewrite the 455.** A 21-item sample is small; the extrapolation is a ceiling, not a work order. The dominant reality is: these clusters are the benchmark's implicit-teaching signal.

---

## TASK 3 — missing_Q_row (640): recover-via-Q-loading vs. trim

These 640 load **no** skill in the Q-matrix, so they never enter the fit. They are dominated by tone/affect. **No text edits are needed** — the action is a difficulty-neutral **Q-loading** (or trim). Categorized by content:

| bucket | n | recommendation |
|---|---:|---|
| **affect / tone / acknowledgement** ("acknowledge feelings", "encouraging tone", "compliment the student") | **450** | **RECOVER** → presentation/affect Q-loading |
| **presentation / formatting / verbosity / 2nd-person** | **~180** (32 explicit + ~150 in "other") | **RECOVER** → presentation(style) Q-loading; a minority that duplicate the scenario's canonical "follow presentation conventions" item are **trim** candidates |
| **scaffolding-flavored** (guiding/eliciting follow-ups, "check for understanding", jargon-gating hints) | **~11–15** | **RECOVER** → scaffolding Q-loading |
| **pure noise / degenerate** (e.g. `tb_0036_c07` "state the given problem …") | small (~10–15) | **TRIM** |

**Split:** ≈ **625 recover / ≈15 trim**; of the recovers, ≈**610 presentation(affect+style)** and ≈**11–15 scaffolding**. This confirms the pack's read: the Q-less set is a *coverage/loading* problem (affect and presentation behavior are real but unmeasured), not a *quality* problem. Representative rows are in the proposal CSV (`cluster=missing_Q_row`); the 640 are **not** enumerated individually.

Representative recover samples:
- scaffolding: `tb_0521_c03` ("hint should not use jargon without first defining it"), `tb_0564_c05` ("written as teacher→student, guiding language").
- presentation/affect: `tb_0002_c09` ("acknowledge the student's feelings of being overwhelmed"), `tb_0009_c02` ("acknowledge that its initial explanation contained an error").

---

## TASK 4 — under-labeling (11 hypotheses): hand-verified

**Verdict: 0 confirmed / 11 REJECT.** None warrants adding a scaffolding Q-loading — confirming the pack's own "0 high-confidence" finding.

| criterion | why REJECT |
|---|---|
| `tb_0558_c04` | "guide…**by explaining**" — explain-dominant (tells the mechanism). |
| `tb_0564_c08` | "clearly **explain OR** prompt" — telling satisfies it; scaffolding not required. |
| `tb_0249_c03` | Pure sign-error diagnosis; "student arrive" is a false keyword match. |
| `tb_0268_c04` | Pure diagnosis (omitted factors). |
| `tb_0278_c10` | Diagnosis/affirmation of a correct step. |
| `tb_0330_c02` / `tb_0330_c04` | Pure error-identification (diagnosis). |
| `tb_0397_c02` | Diagnosis+content; "questioning" refers to the *student's* questioning. |
| `tb_0448_c06` | "scaffold" match = "**loop scaffolding**" (a code term); this is an affect/praise criterion. |
| `tb_0635_c02` | Diagnosis/affect ("acknowledge why struggling"). |
| `tb_0577_c01` | **Explicitly anti-scaffolding:** "state the rule … **not just ask a question** about the rule." |

The two least-clear rejects (`tb_0558_c04`, `tb_0564_c08`) sit in `hint_generation` scenarios and could be revisited, but both are explanation-dominant, so the default (reject) holds.

---

## Honest bottom line

- **Reverse-behaving (60):** `MIS_KEYED 0 / JUDGE_INVERTED 1 / DEGENERATE_FIT 29 / FINE 30`. **1 reword** (`tb_0532_c06`); the rest are fit artifacts → **exclude from fit, keep the text**.
- **mis_specified + judge_biased (455):** measured true-defect rate ≈ **0–4.8%** (0 clear + 1 verify in 21). **Do not mass-rewrite.**
- **missing_Q_row (640):** ≈ **625 recover** (≈610 presentation/affect + ≈11–15 scaffolding) / **≈15 trim**; **0 text edits**.
- **under-labeling (11):** **0 add / 11 reject.**

**Genuinely-recommended criterion-text edits: 2** — `tb_0532_c06` (reword, medium confidence) and `tb_0370_c11` (reword, low confidence, verify against the HF original first). Everything else is KEEP or a **no-text-change Q-loading / exclude-from-fit** hygiene action. The safe, high-value slice is **psychometric** (exclude 60 artifacts from the fit; recover ~625 affect/presentation criteria via Q-loadings), **not** a rubric rewrite. As in Gate A, the raw flag counts (60 / 455 / 640 / 11 = 1,166) collapse to a **handful** of genuine content changes once each flag is checked against the actual dataset.

---

## Output paths
- **This memo:** `plans+prds/Gate B - Proposed Criteria Revisions (for review).md`
- **Structured proposal (111 rows):** `staging/revision_evidence/proposed_criteria_edits.csv` / `.jsonl`
  - Columns: `criterion_id, scenario_id, cluster, classification, current_text, proposed_action (drop/reword/add_q/keep), proposed_text, evidence, confidence`
  - Action tally: **keep 91 · add_q 18 (representative Q-loadings) · reword 2 · drop 0**

*All analysis read-only over the 82-model pilot + bank. Bank and fitter files untouched; nothing committed.*
