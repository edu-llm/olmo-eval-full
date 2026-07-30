# AdaptiveTesting experiments

Curated diagnostic / calibration experiments (scripts, figures, tables).
Runtime inference dumps stay under `AdaptiveTesting/Outputs/` (gitignored).

| Experiment | Path | What it tests |
|---|---|---|
| 200-model roster | [`models_200/`](models_200/) | Size-balanced 0.2–7B HF roster |
| Local ARC diagnostic | [`local_diagnostic_arc/`](local_diagnostic_arc/) | Same-matrix 2PL/3PL CAT vs full score |
| ATLAS transfer (published) | [`atlas_transfer_published/`](atlas_transfer_published/) | Published ATLAS 3PL → our held-out models |
| ATLAS recalibrate 0.5–7B | [`atlas_recalibrate_0p5_7b/`](atlas_recalibrate_0p5_7b/) | Recalibrated bank on small models only |

## Held-out transfer, side by side

Same 60 held-out models for both banks, computed from each experiment's own
`results/*.csv`. `baseline MAE` is the error from predicting the constant mean
accuracy for every model.

| Bank | SE stop | r | p-IRT MAE | baseline MAE | mean items |
|---|---|---|---|---|---|
| Published | 0.2 | **0.830** | 0.147 | 0.084 | 21.1 |
| Published | 0.3 | **0.738** | 0.172 | 0.084 | 9.9 |
| Recalibrated 0.5–7B | 0.2 | 0.589 | 0.147 | 0.084 | 12.5 |
| Recalibrated 0.5–7B | 0.3 | 0.585 | 0.150 | 0.084 | 8.3 |

Two things to read off this table. The **published bank transfers better than the
recalibrated one**; restricting the calibration population to 0.5–7B models cost
roughly 0.24 in r. And **p-IRT accuracy loses to the constant baseline on absolute
error** in every configuration, so θ ranking is the usable output today while the
reconstructed accuracy level is not. See
[`../docs/02_atlas_and_adaptive_testing.md`](../docs/02_atlas_and_adaptive_testing.md)
§2 for the sub-chance floor effect behind that.
