# InFoBench full calibration study

This command executes the complete full-data study defined by
`plans+prds/Calibration Playbook - Cross-Benchmark.md`. It is not a pilot or a
subset run.

The runner starts at the calibration boundary: it consumes the already frozen
judge-response matrix and native InFoBench Q-matrix. It hash-checks and validates
those inputs, but it does not call the judge again or regenerate the Q-matrix.

```bash
.venv/bin/python scripts/run_infobench_calibration_study.py \
  --config configs/infobench_calibration_study.json \
  --out-dir runs/calibration/InFoBench_playbook_full
```

## Reproducibility requirements

The checked-in configuration deliberately references the frozen response matrix and
judge manifest under `runs/calibration/InFoBench_full_20260804/inputs/`. Those two
artifacts contain run data and are not stored in Git. A fresh clone alone cannot
reproduce the study. Retrieve the frozen calibration-input bundle from the team's
shared artifact storage and verify:

- `response_matrix.csv`:
  `087948fcaa884cde6660df1fb4964072ec60fe8aee398e293ed5f74db8f3f27c`
- `judge_manifest.json`:
  `d2f19c55b8ebb8f3ebe0c641485e0b694ce861f2740835da760b7b51b40698e7`

Place the files at the configured paths, or make a local copy of the configuration
that points to their downloaded locations. The runner verifies input dimensions and
judge provenance before fitting; it does not silently regenerate either input.

If a long run is interrupted, repeat the same command with `--resume`. A stage is
resumed only when its saved command is identical and all declared output artifacts
exist. To inspect commands without fitting anything, add `--plan-only`.

## What the runner covers

- Hashes and validates the complete 52-model, 2,250-criterion input artifacts.
- Fits and compares predeclared 5D, 4D, 3D, 2D, and 1D latent structures while
  preserving InFoBench's five native Q-matrix labels.
- Uses five person folds and disjoint scoring/evaluation scenarios for the primary
  held-out metric. Same-cell reconstruction is retained only as a secondary
  diagnostic.
- Applies the one-standard-error rule, then prefers fewer latent dimensions.
- Repeats the selected structure over the configured grid/ridge sensitivity set.
- Exports a fitted-only bank; unfitted, nonpositive-loading, and extreme-loading
  criteria are excluded explicitly and synthetic parameters are never copied.
- Compares online ability updates, correlated-prior batch EAP, and MWLE out of
  sample.
- Runs the minimum-scenario, SE-target, trace-versus-D-opt, seed/order, parameter
  uncertainty, and adaptive-versus-random studies.
- Selects settings sequentially: scenario floor first, then SE target at that floor,
  then selector at the selected floor and SE. It uses that same configuration for
  held-out estimator recovery, the final CAT comparison, order study, and
  parameter-uncertainty study, and validates the within-study selected replay. This
  configuration remains provisional until dense-grid scoring and independent-cohort
  validation are complete.
- Writes model-bootstrap confidence intervals and a plain-language summary.

## Main outputs

- `structure_comparison.csv` and `selection.json`
- `CALIBRATION_STUDY_SUMMARY.md`
- `study_results.json` and `figures/`
- `selected_bank/` (fitted-only rubrics, filtered scenarios, export manifest)
- `estimator_oos_kfold/`
- `estimator_oos_kfold/pass_rate_calibration.csv` (p-IRT predicted versus observed)
- `cat_sweeps/`, `cat_vs_random/`, and `order_stability/`
- `parameter_uncertainty/`
- `parameter_uncertainty/total_se_vs_target.csv`
- `cat_sweep_metrics.csv` and `cat_sweep_paired_differences.csv`
- `commands.json`, `study_manifest.json`, and `playbook_coverage.json`

## Status of the August 4 result

The completed run selected the one-dimensional `instruction_following` structure and
exported 2,096 fitted criteria. In the same-cohort CAT comparison, the selected policy
used a mean of 5.29 scenarios versus 23.69 for the random baseline. Treat that CAT
efficiency result as provisional: the selected 1D fit and reference EAP ability
estimates used the configured five-point quadrature grid and therefore collapsed onto
five discrete theta nodes. Recovery measured against that reference—and CAT settings
chosen using those recovery values—must be checked with dense one-dimensional
rescoring. The observed replay test lengths are real, but the CAT-versus-random study
reused the 52-model calibration cohort rather than an independent external cohort.

The fitted item parameters are also provisional at 52 calibration models.

The InFoBench-specific human audit of Qwen is also still pending. The curated report
under `reports/infobench_calibration_20260804/` preserves this result and its figures
without committing the raw response matrix, per-model output tables, or provisional
fitted bank.

The InFoBench-specific human audit of Qwen remains an external validation task and
is recorded as pending in `playbook_coverage.json`; calibration does not silently
claim that TutorBench judge validation transfers to InFoBench.

The current source bank contains no predeclared `exclude_from_fit` flags. The runner
therefore records that limitation and removes unfitted, nonpositive-loading, and
extreme-loading criteria at the safe fitted-bank export gate. It does not silently
rewrite the frozen source Q-matrix or pretend those criteria were excluded from the
initial structure fits.
