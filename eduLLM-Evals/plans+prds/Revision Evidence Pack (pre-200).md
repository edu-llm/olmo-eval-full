# Revision Evidence Pack (pre-200-run)

**Status:** READ-ONLY evidence pack. No bank / scenario / fitter files were modified; nothing committed.
**Purpose:** Tell the team exactly *which scenarios/questions* and *which criteria* to consider changing before the 200-run, ranked by priority.
**Gate A (URGENT — must land before the imminent tutor run):** scenario / question-text edits + coverage-gap new questions → **Section A**.
**Gate B (can land in the tutor→judge pause):** rubric revision → **Sections B, C, D**.
**Section E** (coverage) feeds both gates.

**Inputs (read-only):**
- `staging/calibration_mirt_full2skill.csv` (2-skill collapsed M2PL: `a_correctness`, `a_scaffolding`, `b`) — 3,497 items fit.
- `staging/run6_presentation/calibration_mirt.csv` + `presentation_health.txt` (3-skill; presentation = the `a_scaffolding` column in that run, 656 items, median a≈1.09).
- `staging/calibration_mirt_full2skill_manifest.json` (drops: 2,043 all-fail + 640 missing-Q-row).
- `staging/response_matrix.csv` (82 models × 6,180 non-optional criteria → per-criterion pass rate / all-fail).
- `data/TutorBench/curated/rubrics_qmatrix_curated.jsonl` (6,845 criteria) and `scenarios_curated.jsonl` (662 scenarios).

**Supporting CSVs (in `staging/revision_evidence/`):** `scenario_candidates.csv`, `allfail_triage.csv`, `sharpening_candidates.csv`, `underlabeling_candidates.csv`, `coverage_gaps.csv`. Builder script: `staging/revision_evidence/_build_evidence.py` (read-only); machine summary: `_summary.json`.

---

## Executive summary

| Section | What | Count (candidates) | Priority split | Gate |
|---|---|---|---|---|
| **A** | Scenario / question-text review + coverage-gap new questions | **443 scenarios** (+3 new-question cells) | 143 H / 154 M / 146 L | **A (urgent)** |
| **B** | All-fail + Q-less dropped-criteria triage | **2,683** (2,043 all-fail + 640 Q-less) | 455 H / 2,100 M / 128 L | B |
| **C** | Low-discrimination sharpening / merge / trim | **584** | 192 H / 189 M / 203 L | B |
| **D** | Under-labeling (hidden scaffolding) — hypothesis list | **11** | 0 H / 10 M / 1 L | B |
| **E** | Per-skill / per-scenario-type coverage | 255 rows | — | A + B |

**Headlines:**
- **Gate A is the bottleneck.** 143 scenarios have the majority of their rubric criteria fail across **all 82** models (near-zero yield). That pattern usually means the *question/context as posed* is mis-scoped or the rubric expects content the prompt never asks for — either way it should be eyeballed before the tutor run. These are ranked in `scenario_candidates.csv`; the top 15 are listed below.
- **The benchmark is genuinely hard** (median scenario mean-pass ≈ 0.10, median all-fail fraction ≈ 0.25), so Section A ranks scenarios *relative* to that baseline and by absolute text-quality signals — it does **not** filter to "critical only," per the brief.
- **Gate B, most actionable slice:** 455 dropped criteria are **judge-biased or mis-specified** rewrites (not "too hard"). The other 1,588 all-fail are **legit-but-unattainable** ("too hard") — mostly *keep/monitor*, optionally partial-credit. 640 are **missing-Q-row** (no skill loads → never enter the fit).
- **Presentation axis** is the main CAT-length cost: 218/656 presentation criteria sit below the axis median info, 37 have a<0.5, and 1 is reverse-behaving. Trimming/merging these is the lever to shorten the ~40-item presentation test length if the optional 3rd axis is kept for the 200-run.
- **60 items are reverse-behaving (fitted a<0)** — higher-ability models fail them *more*. These are the highest-value single fixes (likely mis-keyed / judge-inverted).
- **Scaffolding coverage is lopsided:** 46% of `hint_generation` criteria load scaffolding, but only ~8.6% of `adaptive_explanation` and ~6% of `feedback`. 241/662 scenarios have **no** scaffolding criterion at all.

---

## SECTION A — Scenario / question-text review (GATE A, URGENT, RANKED)

**Method.** For each of the 662 scenarios we combined *text-quality* signals (prompt word-count; use-case) with *response-quality* signals from the 82-model matrix (fraction of the scenario's criteria that are all-fail; scenario mean pass-rate; fraction of criteria dropped from calibration). Baselines: median scenario mean-pass = 0.102, median all-fail fraction = 0.25. A scenario is flagged if it shows at least one signal; **all flagged scenarios are listed and ranked** (not filtered to critical).

- **high (143):** `ungradable_majority_all_fail` (≥50% of criteria all-fail across all models) **or** `near_zero_yield` (mean pass < 0.05), i.e. the whole task fails broadly → the question/context or its rubric scope needs review before the tutor run.
- **medium (154):** broadly failing (≥35% all-fail with <10% pass), or an under-specified/short prompt, or heavy criterion drop.
- **low (146):** elevated-but-not-extreme all-fail or low-ish yield — monitor, lower urgency.

> Interpretation note: a high all-fail fraction can be a **Gate A** problem (question mis-scoped) *or* a **Gate B** problem (criteria too hard / mis-written). Section A flags the scenario for a human read; the criterion-level fix is cross-referenced in `allfail_triage.csv`. Review top-down.

### TOP scenario/question-text candidates (time-critical — all `high`)

| # | scenario | subj / use-case | words | crit | mean-pass | all-fail | one-line reason |
|---|---|---|---|---|---|---|---|
| 1 | `tb_0014` | biology / adaptive_expl | 71 | 10 | 0.00 | 90% | 9/10 criteria unsatisfiable by any model → task/rubric scope mismatch |
| 2 | `tb_0064` | statistics / adaptive_expl | 136 | 11 | 0.001 | 82% | near-zero yield; 91% of criteria dropped |
| 3 | `tb_0108` | biology / adaptive_expl | 32 | 11 | 0.023 | 82% | 82% all-fail; likely expects unstated content |
| 4 | `tb_0027` | biology / adaptive_expl | 66 | 13 | 0.012 | 77% | emotive student prompt, 85% criteria dropped |
| 5 | `tb_0136` | calculus / adaptive_expl | 255 | 11 | 0.004 | 73% | long prompt yet 73% all-fail → rubric over-reaches |
| 6 | `tb_0141` | computer_science / adaptive_expl | 141 | 12 | 0.003 | 75% | near-zero yield; 83% dropped |
| 7 | `tb_0228` | biology / adaptive_expl | 51 | 14 | 0.02 | 71% | 71% all-fail |
| 8 | `tb_0087` | physics / adaptive_expl | 76 | 7 | 0.016 | 71% | 71% all-fail |
| 9 | `tb_0216` | biology / adaptive_expl | 54 | 12 | 0.033 | 67% | 67% all-fail |
| 10 | `tb_0201` | chemistry / adaptive_expl | 36 | 17 | 0.041 | 65% | 65% all-fail across 17 criteria |
| 11 | `tb_0146` | biology / adaptive_expl | 84 | 11 | 0.031 | 64% | 64% all-fail |
| 12 | `tb_0112` | chemistry / adaptive_expl | 47 | 8 | 0.007 | 62% | near-zero yield |
| 13 | `tb_0207` | chemistry / adaptive_expl | 35 | 10 | 0.05 | 60% | 60% all-fail |
| 14 | `tb_0037` | physics / adaptive_expl | 43 | 5 | 0.009 | 60% | near-zero yield on a small rubric |
| 15 | `tb_0222` | physics / adaptive_expl | **15** | 9 | 0.015 | 56% | **short + ambiguous prompt** AND broad failure → reword+rescope |

Full ranked list (443 rows) in `staging/revision_evidence/scenario_candidates.csv` (columns: signals, score, priority, recommendation, prompt_preview). Note the concentration in `adaptive_explanation`: these are follow-up-confusion prompts where the rubric frequently demands a specific misconception-correction that no model produced — the prime candidates for question-text/scope tightening.

### A2 — Coverage-gap NEW questions (from Section E)
| priority | gap | recommendation |
|---|---|---|
| **high** | Scaffolding thin outside hint-gen (241/662 scenarios have no scaffolding criterion) | Author NEW scaffolding-eliciting scenarios (hint_generation / step-by-step guidance) with explicit tutoring-support criteria |
| medium | Presentation axis sparse (656 criteria) if retained | Add presentation-focused items or trim low-info ones (see Section C) |
| low | `computer_science` thinnest subject (84 vs 120 max) | Add CS scenarios to balance subject coverage |

---

## SECTION B — All-fail + Q-less triage (GATE B)

**Method.** Joined the two drop lists (2,043 all-fail + 640 missing-Q-row) to bank text, criticality, objectivity, explicitness, and pass rate, then clustered. Full row-per-criterion output in `allfail_triage.csv`.

| cluster | n | recommend | priority | why |
|---|---|---|---|---|
| **too_hard** | 1,588 | **keep / partial-credit** (mostly monitor) | M (critical) / L (not) | objective + explicit + 0% pass = legit-but-unattainable at current fleet ability |
| **mis_specified** | 345 | **reword** | high | implicit/under-specified criterion never satisfied → make expected evidence explicit |
| **judge_biased** | 107 | **reword + add anchors** | high | subjective + 0% pass → judge cannot award reliably |
| **judge_biased_negative** | 3 | **reword** | high | "must NOT do X" negatives tripped by *all* models → over-triggering anchor |
| **missing_Q_row** | 640 | **add q-loading or trim** | medium | no skill loads → criterion cannot enter the calibration at all |

Priority totals: **455 high / 2,100 medium / 128 low.**

**Recover vs trim vs reword — recommended play:**
- **Reword (highest ROI, 455):** the `mis_specified` + `judge_biased` clusters. Examples: `tb_0006_c08` ("use a simple analogy…" — subjective, unbounded), `tb_0008_c01` (implicit "identify that the student misunderstands…"), `tb_0002_c02/_c05/_c12` (compound multi-clause "must state…" that judges scored 0 across the board).
- **Recover via Q-row (640):** the missing-Q-row set is dominated by *tone/affect* criteria that currently load no skill — e.g. `tb_0002_c01` ("begin with a compliment/acknowledgement"), `tb_0002_c09` ("acknowledge the student's feelings"), `tb_0004_c08` ("encouraging teacher tone… invite the student to re-check"). These are plausibly **presentation** or **scaffolding** loadings; assigning a Q-row recovers them into the fit (see also Section D/E). Trim the ones that are pure formatting noise.
- **Keep/partial-credit (1,588 too_hard):** don't mass-edit; these preserve difficulty. Consider partial-credit rewrites only for the *critical* subset that blocks scoring (examples `tb_0003_c11`, `tb_0004_c06/_c07`).

---

## SECTION C — Sharpening / merge candidates (GATE B)

**Method.** From the 2-skill fit we flagged correctness-loading items with `a_correctness < 0.35` and scaffolding-loading items with `|a_scaffolding| < 0.35`; from the run6 presentation fit we flagged presentation items below the axis median info (a < ~0.8). Full output in `sharpening_candidates.csv` (criterion_id, skill_axis, a, b, pass_rate, recommendation).

Reference medians: correctness a≈1.07, scaffolding a≈0.86 (2-skill, positive items), presentation a≈1.09.

| axis | flagged | of which reverse-behaving (a<0) | note |
|---|---|---|---|
| correctness | 203 | 28 | low/near-zero info; 28 are actively reverse-keyed |
| scaffolding | 163 | 31 | axis already weak; nearly a fifth reverse-behaving |
| **presentation** | **218** | 1 | **drives ~40-item CAT length** — the trim lever |
| **total** | **584** | **60** | — |

**Priority calls:**
- **Reverse-behaving (60 items, a<0) → highest value.** These are almost certainly mis-keyed or judge-inverted, not merely low-info; reword or drop. Extreme examples: `tb_0112_c02`, `tb_0273_c03`, `tb_0043_c01` (a≈−4.09, b≈18.3 — degenerate). Also `tb_0532_c06`, `tb_0371_c12`, `tb_0455_c07` (negative-worded "must not…" criteria that invert).
- **Presentation trim (218; 37 with a<0.5).** Merging/trimming the low-info presentation items is the direct way to cut the presentation test length (the axis's *only* real cost). Example: `tb_0510_c11` ("follow tutoring presentation conventions: use clear Markdown…", a≈−0.40) is both low-info and reverse-behaving — a clean drop/merge candidate. **Recommendation:** if presentation is kept as the optional 3rd axis for the 200-run, prune the bottom ~third first.
- **Correctness/scaffolding low-info merges.** The many `a≈0.35` items with 1–5% pass rates are candidates to merge with a sibling criterion in the same scenario rather than keep as standalone near-zero-info items.

---

## SECTION D — Under-labeling candidates (GATE B, HYPOTHESIS ONLY)

**Method.** Scanned content/diagnosis-only (non-scaffolding-loaded) criteria for tutoring-support language: explicit phrases ("guide the student", "prompt the student", "without giving away the answer", Socratic/guiding questions) plus a student-as-agent elicitation regex (ask/guide/encourage/prompt/invite + "student", or "student to try/derive/discover…"). Tell-style verbs (remind/explain/state) were excluded. Output in `underlabeling_candidates.csv`.

**Finding: under-labeling is NOT widespread — only 11 candidates, 0 high-confidence.** This is itself useful evidence: the scaffolding Q-column is not systematically missing obvious cases. Do **not** auto-apply; review the shortlist:

| criterion | current skills | why flagged |
|---|---|---|
| `tb_0558_c04` | content | "**guide the student** to understand StringBuilder's efficiency…" — elicitation phrasing |
| `tb_0564_c08` | content+diagnosis | "clearly explain or **prompt the student** to recognize…" |
| `tb_0448_c06` | content+diagnosis | "praise at least one correct aspect of the student's code" — affective/support move |
| `tb_0635_c02` | diagnosis | "acknowledge why the student is struggling…" |
| 7 more | content+diagnosis | student-agent phrasing (see CSV) |

Recommendation: hand-review these 11; if confirmed, add a scaffolding Q-loading. Treat as a hypothesis list, not an auto-apply set.

---

## SECTION E — Coverage gaps (feeds A1 new-questions and B1 new-criteria)

**Method.** Counted, per skill and per scenario-type, how many criteria load each collapsed axis (correctness = content∪diagnosis; scaffolding; presentation) and which scenarios lack each axis. Full breakdown (global, per use-case, per subject, per-scenario gaps) in `coverage_gaps.csv`.

**Global (6,845 criteria):** correctness-loading 5,029 · scaffolding-loading 1,142 (16.7%) · presentation 656 (9.6%). Raw Q: content 4,599 · diagnosis 2,412 · scaffolding 1,142 · **none 1,412**. (The 1,412 no-skill criteria overlap the 640 missing-Q-row drops and the tone/affect items in Section B.)

**Scaffolding is concentrated in one use-case:**

| use-case | scenarios | criteria | scaffolding share | presentation share | gap |
|---|---|---|---|---|---|
| hint_generation | 166 | 1,602 | **0.461** | 0.102 | — |
| adaptive_explanation | 329 | 3,413 | **0.086** | 0.096 | thin_scaffolding |
| feedback | 167 | 1,830 | **0.060** | 0.091 | thin_scaffolding |

→ **241/662 scenarios have no scaffolding criterion; 6 have no presentation criterion; 0 lack a correctness criterion.** Scaffolding ability is currently measured almost entirely through hint-generation tasks — a validity risk for a "tutoring-support" axis. **B1 (new criteria):** add scaffolding criteria to `adaptive_explanation` / `feedback` scenarios. **A1 (new questions):** author new scaffolding-eliciting scenarios.

**Subject balance** is even (share of scaffolding 0.11–0.23, presentation ~0.09–0.11 across subjects); `computer_science` is the thinnest cell (84 scenarios vs 120 for calculus). `chemistry` has the lowest scaffolding share (0.107) among subjects.

---

## Recommended sequencing

1. **Before the tutor run (Gate A):** work `scenario_candidates.csv` top-down from the 143 `high` rows; decide per scenario whether to (a) reword/rescope the question, (b) tighten rubric scope, or (c) leave as a hard item. Kick off authoring of new scaffolding-eliciting scenarios (A2).
2. **In the tutor→judge pause (Gate B):** apply the 455 `high` rewrites in `allfail_triage.csv` (judge-biased + mis-specified) and the 60 reverse-behaving fixes in `sharpening_candidates.csv`; decide on presentation-axis trimming; hand-review the 11 under-labeling hypotheses and the 640 missing-Q-row recoveries.

*All figures are read-only diagnostics over the 82-model pilot; no bank/scenario/fitter files were changed.*
