# ATLAS ARC recalibrated on 0.5–7B models

Re-ran ATLAS’s chunked mirt 3PL + mean–σ linking on ATLAS train models with
heuristic size in [0.5, 7]B, then repeated the same held-out p-IRT diagnostic.

## Calibration

- Filtered train matrix: `AdaptiveTesting/Inputs/ATLAS/data/gaussian_sampled_arc_response_matrix_train_with_scores_0p5_7b.csv`
- Custom fit/link scripts: `AdaptiveTesting/Inputs/ATLAS/scripts/01_fit_irt_custom.r`, `02_link_chunks_custom.r`
- Linked params snapshot: `calibration/irt_item_parameters_combined.csv`

## Results

| SE stop | Figure | Approx. r |
|---|---|---|
| 0.2 | `figures/atlas_arc_0p5_7b_heldout_se0.2.png` | 0.59 |
| 0.3 | `figures/atlas_arc_0p5_7b_heldout_se0.3.png` | 0.59 |

## Reproduce diagnostic

```bash
ATLAS_SE_STOP=0.2 uv run python \
  AdaptiveTesting/Experiments/atlas_recalibrate_0p5_7b/atlas_diagnostic_validation.py
```
