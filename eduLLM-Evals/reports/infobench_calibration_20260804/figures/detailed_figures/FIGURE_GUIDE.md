# InFoBench detailed calibration and CAT figures

These figures mirror the analytical style used on `origin/frq/biggen` while using only the completed InFoBench run artifacts. No files on the BigGen branch were changed.

## Strongest figures to share now

1. `01_item_parameter_distributions.png` — distributions of the 2,096 exported difficulty and discrimination estimates.
2. `02_grid_ridge_validation.png` — held-out predictive loss and item-parameter stability across numerical grids and ridge values.
3. `04_oos_pirt_predicted_vs_actual.png` — honest five-fold held-out predicted versus observed pass rates.
4. `05_mwle_uncertainty_selected_setting.png` — ability, item-parameter, and total uncertainty at the selected CAT setting.
5. `06_mwle_total_se_vs_target.png` — the test-length/total-uncertainty tradeoff across stopping targets.
6. `07_order_seed_stability.png` — how much MWLE theta changes when CAT sees scenarios in different orders.

## Useful, but explicitly provisional or exploratory

- `03_cat_vs_random_efficiency.png` — the scenario and criterion savings are descriptive, but this comparison reuses the 52-model calibration cohort. Its recovery panel also uses the coarse five-node EAP reference.
- `08_minimum_scenario_sweep_DIAGNOSTIC.png` and `09_se_target_sweep_DIAGNOSTIC.png` — same-cohort configuration diagnostics, not held-out evidence.
- `10_oos_estimator_recovery_PROVISIONAL.png` — intentionally exposes the vertical stripes caused by the five-node reference theta. Do not use as a final recovery claim until dense one-dimensional scoring is run.
- `11_model_leaderboard_EXPLORATORY.png` — uses one CAT path on the calibration cohort; its intervals include ability and item-parameter uncertainty but not order/path variability.

## BigGen figures not reproduced

The BigGen branch contains factorial floor-by-SE plots and an OOS CAT-versus-random comparison. The current InFoBench study swept floors and SE targets separately and compared CAT with random on the calibration cohort, so exact equivalents would require additional experiments. They were not fabricated from incomplete data.

## Reproduction

Run from `eduLLM-Evals/`:

```bash
.venv/bin/python scripts/plot_infobench_detailed_figures.py
```
