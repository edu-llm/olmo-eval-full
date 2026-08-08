# Benchmark Revisions Log (pre-200-run)

Consolidated record of the pre-200-run TutorBench revision pass: the evidence pack that
generated candidates, and the two review gates (Gate A = scenario/question text, Gate B =
criteria/rubric + psychometrics). Supersedes the separate `Revision Evidence Pack`,
`Gate A`, `Gate B - Proposed Criteria Revisions`, and `Gate B - Q-Loading Recovery` memos.

All analysis was read-only over the 82-model pilot (`staging/response_matrix.csv`) and the
curated bank (`data/TutorBench/curated/rubrics_qmatrix_curated.jsonl`,
`scenarios_curated.jsonl`). Structured proposals live in `staging/revision_evidence/`
(`proposed_scenario_edits`, `proposed_criteria_edits`, `qloading_recovery_plan`,
`scenario_candidates`, `allfail_triage`, `sharpening_candidates`,
`underlabeling_candidates`, `coverage_gaps` + `_build_evidence.py`).

## Guiding lesson (verified twice)

Raw statistical flags are dominated by "hard, implicit, by-design," not "broken." TutorBench
criteria are overwhelmingly `implicit`/`subjective` on purpose: they test whether the tutor
*proactively* surfaces a point. **Default = KEEP.** A criterion failing across all 82 models
usually means the task is legitimately hard. Across ~1,600 flagged rows (443 scenarios +
1,166 criteria), genuinely-recommended text edits collapse to **4** (2 scenario typos + 2
criterion rewords); everything else is KEEP or a difficulty-neutral psychometric action
(exclude-from-fit / Q-loading) that changes no text. Modeled axes: **correctness**
(content ∪ diagnosis, collapsed), **scaffolding**, **presentation** (optional 3rd). No new
"affect" axis - affect/tone maps to presentation.

## Evidence pack candidate counts (pre-gate)

| Section | What | Candidates | Priority (H/M/L) |
|---|---|---|---|
| A | Scenario / question-text review (+3 new-question cells) | 443 scenarios | 143 / 154 / 146 |
| B | All-fail (2,043) + Q-less (640) dropped-criteria triage | 2,683 | 455 / 2,100 / 128 |
| C | Low-discrimination sharpening / merge / trim | 584 | 192 / 189 / 203 |
| D | Under-labeling (hidden scaffolding) hypotheses | 11 | 0 / 10 / 1 |
| E | Per-skill / per-scenario-type coverage | 255 rows | - |

Baselines: median scenario mean-pass approx 0.10, median all-fail fraction approx 0.25 (the
benchmark is genuinely hard). Coverage (6,845 criteria): correctness-loading 5,029,
scaffolding 1,142 (16.7%), presentation 656 (9.6%), no-skill 1,412. Scaffolding is
concentrated in hint_generation (46.1% of its criteria) vs adaptive_explanation (8.6%) and
feedback (6.0%); **241/662 scenarios have no scaffolding criterion**, 6 have no presentation
criterion, 0 lack correctness. CS is the thinnest subject (84 scenarios).

## Gate A - scenario / question-text edits (verified vs `ScaleAI/TutorBench`)

v1 proposed 201+ edits; every reword was re-verified against the authoritative HF original
(matched by follow-up text, since our `tb_XXXX` ids do not exist upstream). **2 edits
survive**, both one-word source typos; all context-turn edits, all 11 `mis_specified` full
rewrites, all 15 scaffolding adds, and 169 short suggestions were dropped/skipped as
implicit-by-design. For all 6 rewords our ingested and curated layers are byte-identical to
HF (no ingestion/curation corruption).

Applied edits (field = `prompt`):
- `tb_0222` (physics, HF row 365): `...greater than than each individual torque?` ->
  `...greater than each individual torque (the applied-force torque and the friction
  torque)?` Fix `than than` + `means`->`mean`; add a value-free referent only (do not reveal
  that the torques subtract - that is the graded catch).
- `tb_0207` (chemistry, HF row 350): `What do mean by 0.25 mol/L x 0.100 L = 0.025 mol??` ->
  `What do you mean by 0.25 mol/L x 0.100 L = 0.025 mol?` Fix `What do mean` -> `What do you
  mean` + the doubled `??`. Rest of the student phrasing preserved.

Key DO-NOT-EDIT reversals (the flagged "defects" are the graded item by design):
- `tb_0319` (calc): the incorrect sqrt(34) chord tutor turn is in the HF original; the rubric
  grades catching it and deriving approx 11.19 m. Editing hands over the answer.
- `tb_0347` (cs): the `word1.lenth()` typo is a rubric target (crit[0] grades identifying the
  compile error). "Fixing" it deletes a criterion.
- `tb_0201` (chem): tutor turn already defines terms; standard usage, gradable -> leave.
- `tb_0230` (calc): benign malformed `r(\theta)` LaTeX in the source model output; ungraded ->
  leave.

## Gate B - criteria/rubric text edits

Across 1,166 flagged rows (60 reverse-behaving + 455 mis_specified/judge_biased + 640
missing-Q + 11 under-labeling), genuine criterion-text edits = **2**:

- `tb_0532_c06` (reword, medium confidence; also exclude-from-fit). JUDGE_INVERTED: the
  student never stated the v(t)<0 => leftward fact; abler models proactively supply it (good
  tutoring) and the judge scored presence-of-fact as a fail.
  - BEFORE: *"The response must not state that the student has correctly identified that the
    particle moves to the left when v(t) < 0."*
  - AFTER: *"The response must not falsely credit the student with having determined that the
    particle moves left when v(t)<0 (the student got stuck before reaching this and never
    stated it). Stating the v(t)<0 -> leftward-motion fact itself, e.g. as a hint, is
    acceptable; only ATTRIBUTING that (unmade) determination to the student fails this
    criterion."*
- `tb_0370_c11` (reword, LOW confidence - verify against HF before trusting). Variable-
  convention clash: rubric answer-key uses `x=height` (~11.75 in) while the prompt defines
  `x=width` (~5.87 in); sibling `tb_0370_c13` is internally garbled. May be working-as-intended
  for a messy feedback item. (The bank also carries a related "56 in^2" -> "69 in^2" numeric fix.)

Everything else in the criteria clusters is KEEP: mis_specified (345) + judge_biased (110)
measured true-defect rate approx 0-4.8% on a 21-item stratified sample -> do NOT mass-rewrite;
under-labeling 11 -> 0 add / 11 reject (all explanation/diagnosis-dominant, e.g. `tb_0577_c01`
is explicitly anti-scaffolding). Proposal tally (`proposed_criteria_edits`, 111 rows):
keep 91 / add_q 18 / reword 2 / drop 0.

## Gate B - psychometric actions (no text change)

### Fit-exclusion: 60 reverse-behaving (`a<0`) items masked from the M2PL fit, kept in the bank
| Tag | n | Meaning |
|---|---:|---|
| degenerate | 29 | extreme fit (a approx -4.09, b approx 18.3) or near-zero-variance/sparse (mostly 1/82 passers) |
| low-info-fine | 30 | recomputed raw r_pb >= 0 (abler models pass more); negative `a` is a 2-skill collapse artifact |
| judge-inverted | 1 | `tb_0532_c06` (r_pb=-0.57) |
| mis-keyed | 0 | none reward the wrong behavior |

Decisive evidence was the raw point-biserial of item-pass vs model ability from
`response_matrix.csv`. The 30 "low-info-fine" scaffolding items (e.g. `tb_0617_c06` r_pb=+0.70)
should re-estimate cleanly in the 3-skill fit.

### Q-loading recovery for the 640 `missing_Q_row` criteria (load no skill -> never enter fit)
Ordered keyword classifier over full criterion text (trim -> scaffolding-strong ->
presentation -> scaffolding-weak -> default presentation; presentation checked before the weak
"hint" mention because many hint items grade tone/format).

| Assignment | n | variance-pass |
|---|---:|---:|
| presentation | 634 | 607 |
| scaffolding | 5 | 5 |
| trim | 1 | - |

108 low-confidence defaults (no keyword) went to presentation. The 5 scaffolding items:
`tb_0521_c03`, `tb_0564_c05`, `tb_0598_c09`, `tb_0006_c09`, `tb_0564_c14`. The 1 trim:
`tb_0036_c07` (echoes the prompt).

### Fit-eligibility delta
| Axis | Current | Excluded | Recovered | Projected | Delta |
|---|---:|---:|---:|---:|---:|
| correctness | 3,035 | -51 | +0 | 2,984 | -51 |
| scaffolding | 738 | -30 | +5 | 713 | -25 |
| presentation | 0 | 0 | +607 | 607 | +607 |

Non-obvious finding: **presentation is the win** (a previously-empty pre-registered axis
becomes 607 fit-usable items). Scaffolding is NOT rescued by Q-recovery (only 5 genuine items;
net -25 because 30 excluded artifacts were scaffolding-loaded) - the scaffolding rescue must
come from re-estimating the 30 low-info-fine items in the 3-skill fit.

## Sequencing (as executed)
1. Gate A (before tutor run): reworked the 2 typo scenarios; authored new scaffolding-eliciting
   scenarios for the coverage gap.
2. Gate B (tutor->judge pause): 2 criterion rewords; 60 items excluded from fit; 640 Q-less
   criteria reloaded (634 presentation / 5 scaffolding / 1 trim). See `Pilot Baseline Freeze`
   for the frozen post-revision calibration.
