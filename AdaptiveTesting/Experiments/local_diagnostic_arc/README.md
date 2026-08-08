# Local ARC-Challenge diagnostic validation

Calibrate 2PL/3PL on a train split of our MCQ response matrices, then run a
Fisher-information CAT on held-out models and scatter diagnostic-predicted vs
full-benchmark accuracy.

## Results (held-out)

| Setting | Figure |
|---|---|
| 2PL, SE&lt;0.15 | `figures/diag_validation_arc_challenge.png` |
| 2PL, SE&lt;0.3 | `figures/diag_validation_arc_challenge_se0.3.png` |
| 3PL, SE&lt;0.3 | `figures/diag_validation_arc_challenge_3pl_se0.3.png` |

Matching CSVs are under `results/`.

## Reproduce

```bash
uv run python AdaptiveTesting/Experiments/local_diagnostic_arc/diagnostic_validation.py arc_challenge 0.15 2pl
uv run python AdaptiveTesting/Experiments/local_diagnostic_arc/diagnostic_validation.py arc_challenge 0.3 2pl
uv run python AdaptiveTesting/Experiments/local_diagnostic_arc/diagnostic_validation.py arc_challenge 0.3 3pl
```
