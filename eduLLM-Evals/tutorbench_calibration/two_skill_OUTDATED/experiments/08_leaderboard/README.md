# 08 — leaderboard @ locked EAP 20/0.27

**Status: EAP-regenerated @20/0.27 (stop-dependent, of-record).**

Per-model deployed CAT ability at the locked operating point (floor=20, SE_ability=0.27), EAP
stop, MWLE θ at stop, with SE_total error bars. 115 models; the 2 weakly-calibrated models are
flagged `weak_calibrated=True` (excluded from the headline; they land at the low-θ tail).

## Files

- `model_leaderboard_f20se27.csv` — one row per model, ranked by `theta_stop_correctness` desc.
  Columns: θ_stop (corr, scaff), SD (corr, scaff), SE_total (corr, scaff), length, stop_reason,
  reached (both SD ≤ 0.27), weak_calibrated. (Also copied to package root as `model_leaderboard.csv`.)
- `figures/leaderboard_correctness_f20se27.png` — ranked correctness θ with SE_total bars
  (grey = weak-calibrated).
- `metrics.json` — of-record summary (config, headline numbers, discrepancy check).

## Provenance

`scripts/of_record_f20se27.py` from `06_floor_se_grid/oos_per_model_per_cell.csv` (row 20/0.27).
Original: `reports/eap_oos_grid_tutorbench_2skill/of_record_f20se27/`.
