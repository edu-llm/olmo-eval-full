# 06b — operating-point landscape (in-sample EAP stop grid)

**Status: EAP-regenerated @ full grid (stop-dependent, in-sample landscape).** Companion to
`06_floor_se_grid/` (the OOS selection basis). This is the in-sample op-point derivation on the
33-model TB33 set — the landscape / heatmaps that motivate the SE-target elbow and document the
two-floor (asymptotic vs within-budget plateau) distinction.

Same EAP-posterior stop machinery (MWLE at stop, 161-node/dim dense marginal SD, info-plateau
δ=0.005/W=3, cap 70). Floors {6,8,10,12,15} × SE targets {0.15…0.32}.

## Key points

- Floor is inert here (stops land at 20–33 scenarios, above every floor) — the **SE target** is
  the knob. Median SE_total for correctness never drops below ~0.35: **SE_param (~0.21) is the
  binding precision ceiling**, exactly as the OOS grid confirms.
- **Two floors:** the *asymptotic* full-bank SD floor (correctness median 0.087, max 0.234) is
  far below the *within-budget plateau* floor (~0.32–0.45 for weak models). "Bank-limited" means
  diminishing returns within a feasible test length, not a hard information ceiling.
- Cap(70) never binds; info-plateau is the effective secondary stop.

## Files

- `per_cell_grid.csv`, `per_model_per_cell.csv` — in-sample grid metrics.
- `asymptotic_sd_floor.csv` — per-model full-bank SD floor.
- `recovery_r_correctness_grid.csv`, `recovery_r_scaffolding_grid.csv` — recovery-r landscape.
- `se_total_mean_sd_by_target_all33.csv`, `..._excl2.csv` — SE_total by SE target.
- `summary.json` — config + plateau sensitivity + asymptotic-floor summary.
- `heatmap_*.png`, `heatmap_combined_se_len_recovery.png` — landscape heatmaps.

## Provenance

`scripts/eap_stop_grid_study.py --workers 6` + `scripts/plot_eap_stop_grid.py` /
`scripts/plot_eap_stop_grid_recovery.py`. Original: `reports/eap_stop_grid_tutorbench_2skill/`.
