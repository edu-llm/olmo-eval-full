# InFoBench remediation figures

These figures use only the completed remediation artifacts from `runs/calibration/InFoBench_remediation_v1_quadrature_resolution/`.

## What is complete

Phase 2 is complete: the 61-point fit grid passed common-cell equivalence, the 401-node normal-trapezoid EAP grid passed numerical checks, and ridge selection was run inside the nested cross-validation design.

## What is incomplete

Phase 3 did not produce finalists in folds 2 and 3. Outer CAT diagnostics therefore cover 32 of 52 models in only three folds. There is no pooled outer estimate, no frozen final CAT policy, and no Phase-4 parameter-bootstrap or order-stability result. The Qwen-on-InFoBench human audit also remains incomplete.

## Figure-by-figure guide

### `01_item_parameter_distributions.png` — Provisional item-parameter distributions

The locked grid-61/ridge-0.1 fit contains 2,105 criteria: 2,097 currently exportable and 8 with nonpositive discrimination. This is a Phase-2 diagnostic, not a frozen deployment bank.

### `02_fit_grid_common_cell_equivalence.png` — Fit-grid common-cell equivalence

Held-out log-loss and Brier-score changes are shown with family-cluster bootstrap 95% intervals. Both adjacent grid comparisons stay inside the frozen equivalence margin on the same 21,329 evaluation cells.

### `03_eap_numerical_stability.png` — EAP numerical stability gates

Theta shifts are far below the locked tolerances across denser EAP grids, a wider integration bound, and an independent quadrature family. The 401-node normal-trapezoid grid therefore passed Phase 2.

### `04_nested_ridge_selection.png` — Nested-CV ridge selection

Ridge was selected inside each outer training fold using disjoint inner predictions and the one-standard-error rule. All five folds selected 0.1.

### `05_cat_gate_outcomes.png` — CAT gate outcomes by fold

Only outer folds 0, 1, and 4 produced any Phase-3 finalist. Failure counts overlap because a candidate can fail more than one gate.

### `06_inner_cv_cat_tradeoff_landscape.png` — Inner-CV CAT tradeoff landscape

All 240 candidate-fold rows are development evidence. Points outlined in black pass every absolute gate; this is not outer-test performance.

### `07_minimum_scenario_sweep_DIAGNOSTIC.png` — Minimum-scenario sweep diagnostic

A frozen trace/SE=0.20 slice shows how the minimum-scenario floor changes recovery slope and pass-rate MAE inside the outer training folds.

### `08_se_target_sweep_DIAGNOSTIC.png` — Conditional-SE target sweep diagnostic

A frozen trace/floor=12 slice shows the inner-CV test-length and recovery tradeoff. It does not select a final stopping target.

### `09_partial_outer_recovery_INCOMPLETE.png` — Partial outer-fold recovery

Only the 32 models in folds 0, 1, and 4 were scored because folds 2 and 3 had no finalist. No pooled recovery estimate or final policy exists.

### `10_partial_outer_cat_vs_random_INCOMPLETE.png` — Partial outer CAT-versus-random diagnostics

Per-fold CAT and random-baseline results are shown separately for the three scored folds. They must not be interpreted as an all-52-model result.

## Deliberately not reproduced

The obsolete run included total-SE, order/seed stability, final-policy recovery, complete CAT-versus-random, and model-leaderboard figures. Those require Phase 4 and/or a final policy and would be misleading for the current run, so they are not present here.

## Reproduction

```bash
.venv/bin/python scripts/plot_infobench_remediation_figures.py
```

`figure_source_manifest.json` records SHA-256 hashes for every source artifact and rendered PNG.
