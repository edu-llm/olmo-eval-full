# 06 — floor × SE_ability OOS grid (operating-point selection basis)

**Status: EAP-regenerated @ full grid (stop-dependent, of-record).** This is the OUT-OF-SAMPLE
grid from which the locked **floor=20 / SE=0.27** operating point was SELECTED.

The entire 35-cell (floor {10,12,15,20,25} × SE_ability {0.15,0.20,0.22,0.25,0.27,0.30,0.32})
grid is scored OUT-OF-SAMPLE over all **115** models with a k=5 (seed 20260729), refit-per-fold
protocol, under the EAP-posterior honest per-skill marginal-SD stop (161 nodes/dim, info-plateau
δ=0.005/W=3, cap 70, MWLE θ at stop). Headline excludes the 2 weak models → N=113.

## Key finding

Recovery is excellent and essentially **flat across the whole grid** (corr r ≈ 0.957–0.971,
slope ≈ 1.0); it does not degrade OOS. What moves is length vs %reach. The floor only bites at
20/25 with loose SE (≥0.27). **SE_param (median ~0.21 corr) is the binding precision ceiling** —
median SE_total_corr floors at ~0.36 regardless of the SD target. Cap(70) never binds.

The locked row **20/0.27** gives median length 25, corr r=0.967 / scaff r=0.932, both-skills
SE_ability reach 33.6%, plateau 66.4%.

## Files

- `oos_per_cell_grid.csv` — 35 rows, all aggregate OOS metrics per cell.
- `oos_per_model_per_cell.csv` — per model × cell (θ_ref, MWLE θ, SD, SE_total, length, stop_reason).
- `weak_models_per_cell.csv` — per-cell fate of the 2 excluded weak models.
- `summary.json` — config, protocol, fold diagnostics, SE_param offsets.
- `figures/oos_grid_heatmaps.png` — r_corr / median SE_total / median length / %reach heatmaps.
- `figures/oos_recovery_scatter_f12se27.png` — recovery scatter at 12/0.27 (context).

## Provenance

`scripts/eap_oos_grid_study.py --workers 6`. Original: `reports/eap_oos_grid_tutorbench_2skill/`.
