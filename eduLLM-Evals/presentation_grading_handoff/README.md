# presentation_grading_handoff

Targeted add-on to the full Qwen calibration run: grade the **662 per-scenario
presentation (`style_surface`) criteria** that were `optional`/non-gating in the
main run and never scored, so we can test presentation as a candidate skill axis.

## Contents
- `rubrics_presentation_gating.jsonl` — the 662 presentation criteria, flipped to
  gating (`optional: false`, pass/fail `judge_guidance`). One per scenario.
- `RUN_INSTRUCTIONS.md` — exact prepare / commands / status / merge steps.

## Prerequisite
The cluster runner reuses the existing `qwen_calibration_handoff_32k_20260728/`
packet (scripts, `tutor_cat`, responses zip, env, cohort policy, judge config).
The **only** change is `--rubrics rubrics_presentation_gating.jsonl`.

## Scope / expected
82 models × 662 criteria = **54,284 cells** (~11% of the main run). Same frozen
judge (`Qwen/Qwen3.5-9B`, 32K ctx, `p_fail>=0.33`) so verdicts merge with the
existing `run_data.jsonl`.

## Provenance / reversibility
Built by `scripts/build_presentation_gating_bank.py` from
`data/TutorBench/curated/rubrics_qmatrix_curated.jsonl` (read-only). The official curated
bank is **not** modified; flipping is confined to this experimental bank. To make
the change permanent later, flip `optional: false` on the 662 `style_surface`
criteria in the official bank.

## What we need back
The de-blinded `run_data.jsonl` (with a `tutor_model` column), the merged dir, and
prep manifests. If no de-blinded jsonl is produced, send
`presentation_prepared/private/case_index.jsonl` instead.
