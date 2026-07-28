# Rubric Curation Policy (`curation_v1`)

Context doc for any agent/human working on TutorBench criterion quality in this repo.
Read this before editing criteria or scaling the curation beyond the pilot.

## Why this exists

Human grading of frontier tutors against the TutorBench-derived criterion bank
(`data/rubrics_qmatrix_final.jsonl`, 6,462 criteria) surfaced recurring quality
problems. TutorBench rubrics are *human-generated*, but "human-generated"
guarantees pedagogical **intent**, not measurement-ready **atomicity/scope**. The
authors wrote *descriptions of an ideal answer keyed to the reference solution*,
not *atomic, phrasing-agnostic, turn-scoped pass/fail conditions*. Our ingest
(`scripts/test.py`) copies criterion text verbatim (only dropping `deleted`
items), so authoring artifacts flow straight into the measurement instrument.

An audit (`scripts/audit_rubric_quality.py` -> `staging/audit_rubric_quality.*`)
found **1,820 / 6,462 (28.2%)** criteria trip >=1 quality flag.

## The three failure modes we are clearing

1. **Non-atomic (bundled)** — one criterion smuggles in multiple independently
   gradeable requirements, so a single P/F cannot honestly represent it.
   Signatures: multiple requirement verbs, conjoined actions
   (`and then`, `followed by`, `concluding with`), enumerated sub-steps,
   two action verbs joined by `and`, multiple errors bundled.
   *Bank count: 724 (11.2%). Highest-precision signal; safe to act on.*

2. **Irrelevant / mis-scoped** — grades something not responsive to the
   *follow-up turn actually being evaluated*. Two sub-types:
   (a) **restatement**: demands re-emitting a formula/answer already delivered in
       a prior turn; (b) **surface/persona orphan**: pure format/second-person/tone
       with no skill loading (`q_mapping` all-zero).
   *Bank count: 407 (6.3%).*

3. **Rigid / verbatim** — phrased so a correct-but-differently-worded answer
   fails: `must say/state <exact wording>`, `identify that <X> is incorrect`,
   long `i.e.`-anchored targets, absolute failure language
   (`... causes failure`, `... in full`), or very long over-specified criteria.
   *Bank count: 956 (14.8%). NOISIEST signal — needs human review, not auto-action
   (see caveats).*

## Additional patterns (approved, to clear in the scaled pass)

Confirmed with the data owner after the pilot. Fold detection into
`audit_rubric_quality.py` and fixes into the edit spec before scaling.

| Pattern | Signature | Fix |
|---------|-----------|-----|
| **Identify+correct pairs** | one diagnosis criterion bundling "detect the error" AND "provide the correction" | split into detect vs. correct |
| **Reference-solution leakage** | criterion cites "the Golden Response" / the hidden reference (e.g. `tb_0011_c08`) | rephrase to describe the required structure directly, without citing the reference |
| **Absolute-failure language** | "causes failure", "leads to failure", "must ... in full", "Any missing ... leads to failure" (e.g. `tb_0019_c02/c03/c04/c08`) | soften; remove absolute all-or-nothing phrasing |
| **Numeric exactness** | answer criteria with no rounding tolerance (e.g. `tb_0001_c01`) | add "accept minor rounding / equivalent value" |
| **Prescribed-pedagogy wording** | criterion dictates an exact phrase or metaphor (e.g. `tb_0340_c08` "similar to: 'Totally!...'", `tb_0072_c03` "six-brick LEGO or a six-car toy train") | soften to accept any equivalent move/metaphor |

## What is NOT a problem (do not "fix" these)

- **Withhold-answer criteria** (`must not provide/reveal ...`) that quote the
  answer or formula *only to forbid revealing it* (common in `scaffolding` /
  `critical_negative` hint-generation items). The audit's `restatement` heuristic
  flags ~20 of these as false positives. **Keep as single criteria** (op `keep`).
- **Atomic factual criteria** where stating the specific fact *is* the criterion
  (e.g., "H bonded to more electronegative elements is +1"). The
  `demands_specific_wording` flag over-fires here — the fix is to *soften wording
  acceptance*, not delete.

## Heuristic caveats

- `restatement_of_formula_or_answer` co-fires on withhold criteria (false pos).
- `demands_specific_wording` (697 hits) is the noisiest reason; treat as
  "needs review", not an auto-action trigger.
- All audit counts are heuristic; the per-criterion CSV must be hand-audited.

### New-detector precision (first run, pre-tightening) and status

| Detector | 1st-run count | Precision | Status | Auto-fix? |
|----------|---------------|-----------|--------|-----------|
| `absolute_failure` | 9 | 8/9 (dropped `in full` FP) | tightened | yes (soften) |
| `prescribed_wording` | 2 | 2/2 (low recall) | ok, widen later | yes (soften) |
| `reference_leak` | 3 | 1/3 (dropped bare `the reference`; "reference type" was FP) | tightened | yes (rephrase) |
| `identify_correct` | 315 | LOW (fires on "correctly", on already-atomic "correct the error" halves) | tightened; **REVIEW-ONLY** | no |
| `numeric_no_tolerance` | 133 | MEDIUM (caught diagnosis "student's answer of X", withhold "must not compute") | tightened (exclude diagnosis + negatives) | no |

Re-run `audit_rubric_quality.py` after tightening for trustworthy counts. Only
the three high-precision detectors are safe to auto-fix; `identify_correct` and
`numeric_no_tolerance` are decision-support for hand review.

## Calibrated fix decisions (baseline, approved by data owner)

| Issue | Decision |
|-------|----------|
| Non-atomic bundle | **Split** at natural conceptual boundaries (not maximally fine). |
| Rigid content fact | **Split** rule vs computation **and soften** to accept semantic equivalents. |
| Mis-scoped restatement | **Rescope to conditional** ("only if the response re-derives"), set `not_critical`, set `optional: true`, add `judge_guidance` telling the judge to mark **N/A (blank), not fail** when it does not apply. |
| Surface/persona orphan | **Consolidate** all format/persona orphans in a scenario into ONE reusable standard criterion (`meta.standard_format_criterion`). |
| Verbatim negation | **Soften**: accept an explicit "incorrect" label **OR** an equivalent correction. |
| Withhold-answer | **Keep** as a single criterion (correctly scoped). |
| Prohibition + positive req bundled | **Split** the "must not reveal" from the "must instead do X". |

## Atomization granularity rule

Split down to the finest level where each child is **(a) conceptually distinct
AND (b) can realistically be satisfied independently** of its siblings.

- **Distinct errors / distinct conceptual acts -> SPLIT.** e.g. detecting a
  circle-equation error vs. a chain-rule error are independent events; "declare
  the distribution" vs. "define its parameters" are distinct acts.
- **Mechanically-linked sub-parts of ONE co-satisfied step -> KEEP TOGETHER.**
  e.g. defining `n` and `p`, or three unit conversions in one setup step, are
  near-perfectly correlated (a response that gets one gets them all).
- **Why the caveat matters:** over-splitting correlated sub-checks creates
  *locally dependent* items that (i) add judge cost / test length for ~zero new
  information and (ii) **bias MIRT calibration**, which assumes criteria are
  conditionally independent given ability. Atomize for honesty, not maximally.

## Metadata rules on edit

- **Split children** copy the parent's `criticality`, `q_mapping`, `difficulty`,
  `discrimination`, `objectivity`, `explicitness`, `irt_params` (override only when
  it does not make sense). `q_rationale` is **regenerated** per child from the
  child's own text (no longer quotes the old bundled criterion).
- **Renumber**: any scenario that changes has ALL its criteria renumbered
  sequentially `{scenario}_c01..cN`, and `scenario.criterion_ids` is regenerated.
  Scenarios whose only op is `keep` are left byte-identical.
- Every produced/edited record carries a `curation` provenance block
  (`policy_version`, `op`, `from`, `original_criterion_id`).
- **IRT params**: split children currently COPY parent params. These are synthetic
  (`metadata_heuristic_v1`) and MUST be re-derived via `scripts/assign_irt_params.py`
  before any real calibration. Do not trust copied params for calibration.
- **Optional criteria**: `optional: true` + `judge_guidance` require the LLM judge
  prompt to support an N/A verdict. Ensure judge integration honors this before
  optional criteria affect scoring.

## Files

| Path | Role |
|------|------|
| `scripts/audit_rubric_quality.py` | Read-only audit; emits `staging/audit_rubric_quality.{json,csv}`. |
| `curation/pilot_edits.json` | Hand-authored edit spec (ops keyed by `criterion_id`). |
| `scripts/apply_curation.py` | Applies a spec to a CLONE; never mutates source. |
| `data/curated/rubrics_qmatrix_curated.jsonl` | Curated clone (edits applied). |
| `data/curated/scenarios_curated.jsonl` | Curated scenarios (renumbered `criterion_ids`). |
| `curation/pilot_change_report.{md,csv}` | Before/after review report. |

## How to run

```
python scripts/audit_rubric_quality.py          # refresh audit counts
python scripts/apply_curation.py                 # clone + apply curation/pilot_edits.json
```

## Status

Curated bank: **6,462 source -> 6,845 criteria** (`data/curated/`). All tranches
below verified via `scripts/verify_curated.py` (0 dup ids / empty texts / missing
rationales / dangling refs). **Criterion bank is FINALIZED** (curation_v1); IRT
params are the separate downstream step. See `curation/CHANGES.md` for the
teammate-facing before/after summary.

### Tranche 1 - non_atomic (critical) — DONE
Mechanical auto-fixes (softens, consolidations) + 142 atomic splits across the
critical non-atomic pool; the rest KEEP (flag precision ~7-15%). See
`pilot_change_report.md`.

### Tranche 2 - rigid_verbatim — DONE (no per-criterion edits)
The `rigid_verbatim` flag is ~95% false positives ("state that <fact>" content).
A full-bank scan for criteria that genuinely *forbid a correct equivalent* found
only substantive requirements, so **no text edits were needed**. The
"accept semantic equivalents" concern is handled bank-wide by
`curation/grading_notes.md`, which is wired into every judge path:
- canonical frozen judge already enforced it (`EVIDENCE_DECISION_POLICY` item 7);
- local smoke judge aligned (`tutor_cat/judge.py`, `judge-v2` -> `judge-v3`).

### Tranche 3 - irrelevant_misscoped — DONE
- **surface_format_or_persona (all-zero Q, ~306)**: bank-wide **aspect-aware
  presentation consolidation** in `apply_curation.py`. Each scenario's style
  orphans merge into ONE `optional` criterion tagged `dimension: style_surface`
  whose text covers only the aspects actually present (persona / structure /
  math / code). All are `not_critical`, carry `judge_guidance`, and are already
  excluded from MIRT (all-zero Q rows are dropped in `calibrate_mirt.align_q_rows`).
- **restatement_of_formula_or_answer (~90)**: KEEP all. Context review showed the
  follow-up turn almost always calls for the answer/formula (student asks, asks to
  verify, or is confused) -> legit content, not mis-scoped. Withhold variants
  ("must not give the answer") were already excluded.
- **numeric_no_tolerance (80)**: no per-criterion action; covered by the bank-wide
  grading note (notation equivalence + rounding / significant figures).

### Presentation coverage normalization — DONE
`apply_curation.py --ensure-presentation` (default on) appends one optional
`style_surface` presentation criterion (subject-appropriate default aspects) to
every scenario lacking one. Result: **all 662 scenarios** have exactly one
presentation criterion (313 consolidated + 349 added).

### Cross-scenario coverage check — no gaps to add
A per-`use_case` pattern-coverage scan (`scripts/coverage_by_usecase.py`) confirmed
each use_case's characteristic patterns are 77-96% present; spot-checks showed the
residual "missing" cases are detector artifacts (plural/adverb forms, skill-tag
differences) or intentional per-scenario omissions, not real gaps. No substantive
criteria were added by pattern transfer.
