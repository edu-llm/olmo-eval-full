# InFoBench calibration report — 2026-08-04

This directory is the shareable, curated snapshot of the completed InFoBench
calibration and CAT study. It contains aggregate results, selection provenance, and
figures. Large run logs, raw judge/tutor responses, individual-model tables, bootstrap
draws, and the provisional fitted bank remain in external run storage rather than Git.

## Headline result

- The one-standard-error rule selected one latent dimension:
  `instruction_following`.
- The fitted-only export retained 2,096 of 2,250 criteria.
- The selected CAT configuration used trace selection, no scenario floor, and an
  SE stopping target of 0.25.
- In the same-cohort comparison it administered 5.29 scenarios on average versus
  23.69 for random selection.

## Important limitations

This snapshot is **provisional**. The reference EAP scores used five quadrature
points, so ability values collapsed onto five discrete theta nodes. Dense
one-dimensional rescoring is required before final ability-recovery or CAT-efficiency
claims. The selected Qwen judge was validated on TutorBench, while an
InFoBench-specific human agreement audit is still pending.

The checked-in study configuration references two frozen inputs that are intentionally
not tracked: `response_matrix.csv` and `judge_manifest.json` under
`runs/calibration/InFoBench_full_20260804/inputs/`. Retrieve those from the team's
shared artifact storage to reproduce the full run. Their SHA-256 hashes are
`087948fcaa884cde6660df1fb4964072ec60fe8aee398e293ed5f74db8f3f27c`
and `d2f19c55b8ebb8f3ebe0c641485e0b694ce861f2740835da760b7b51b40698e7`,
respectively. Paths in this report were converted from workstation-specific absolute
paths to repository-relative paths; hashes and other provenance fields were retained.

## Contents

- `CALIBRATION_STUDY_SUMMARY.md`: plain-language result summary.
- `structure_comparison.csv`, `selection.json`, and `structure_selection.json`:
  latent-structure evidence and selection.
- `sensitivity_*` and `ridge_sensitivity_selection.csv`: numerical grid/ridge
  checks.
- `cat_*` and `cat_vs_random/`: CAT configuration, efficiency, and paired metrics.
- `estimator_oos_*`: held-out EAP/MWLE/online recovery and pass-rate calibration.
- `order_stability*` and `parameter_uncertainty*`: order and uncertainty checks.
- `selected_bank/export_manifest.json`: fitted-bank inclusion/exclusion provenance;
  the bank itself is not frozen or tracked here.
- `figures/`: the four summary figures and eleven detailed diagnostic figures.
