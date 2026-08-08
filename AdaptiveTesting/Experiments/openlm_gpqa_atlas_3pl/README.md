# OpenLM GPQA — ATLAS unidimensional 3PL diagnostic

90/10 model split on OpenLM GPQA responses → mirt **3PL** on ~100-item chunks
with mean–σ linking → Fisher CAT (SE≤0.3) + p-IRT on held-out 10%.

## Held-out (SE≤0.3)

| Metric | Value |
|---|---|
| Pearson r | **0.737** |
| MAE | 0.070 |
| RMSE | 0.083 |
| avg CAT items | 13.0 |

See `figures/openlm_gpqa_atlas3pl_se0.3.png` and `results/`. A 2PL re-fit
(same split) is under `calibration_2pl/` for comparison
(`figures/compare_2pl_vs_3pl_se0.3.png`).

## Reproduce

```bash
# Needs AdaptiveTesting/Inputs/OpenLM/gpqa/*.csv locally
uv run python prepare_matrix.py
bash run_calibration.sh
uv run python diagnostic_validation.py
```
