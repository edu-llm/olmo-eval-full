# AdaptiveTesting experiments

Curated diagnostic / calibration experiments (scripts, figures, tables).
Runtime inference dumps stay under `AdaptiveTesting/Outputs/` (gitignored).

| Experiment | Path | What it tests |
|---|---|---|
| 200-model roster | [`models_200/`](models_200/) | Size-balanced 0.2–7B HF roster |
| Local ARC diagnostic | [`local_diagnostic_arc/`](local_diagnostic_arc/) | Same-matrix 2PL/3PL CAT vs full score |
| ATLAS transfer (published) | [`atlas_transfer_published/`](atlas_transfer_published/) | Published ATLAS 3PL → our held-out models |
| ATLAS recalibrate 0.5–7B | [`atlas_recalibrate_0p5_7b/`](atlas_recalibrate_0p5_7b/) | Recalibrated bank on small models only |
