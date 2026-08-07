# 03 — dimensionality / skill structure (1 vs 2 vs 3 skill)

**Status: carried over (STOP-INDEPENDENT).** The choice of latent structure is a calibration /
fit decision independent of the CAT stop rule, so the of-record 2-skill/115 comparison is carried
over unchanged and remains valid under the EAP-posterior stop.

## Decision

**2-skill (correctness + scaffolding) is canonical.** `content` + `diagnosis` collapse into
`correctness`; `scaffolding` as-is; `presentation` is NOT modeled. The **3-skill** variant (adds
`presentation`) is a secondary/exploratory track: it only reaches precision for 58/115 models at
SE=0.30 and needs ~41 scenarios (vs ~22 for 2-skill), so it is not the shipped instrument. The
**1-skill (unidim)** variant recovers well (r≈0.949) but discards the scaffolding axis the
instrument is built to report.

## Comparison (MWLE, OOS)

| structure | corr r | corr slope | scaff r | pres r | precision | mean scen |
|---|---|---|---|---|---|---|
| 1-skill (unidim) | 0.949 | 0.958 | — | — | 115/115 | 12.1 |
| **2-skill** | 0.945 | 0.981 | 0.902 | — | 115/115 | 21.9 |
| 3-skill (SE=0.30) | 0.969 | 1.115 | 0.916 | 0.965 | 58/115 | 41.2 |

(from `model_comparison_1v2v3.csv`; per-structure OOS k-fold recovery detail in
`oos_recovery/{unidim,2_skills,3_skills}/`.)

## Files

- `model_comparison_1v2v3.csv` — wide 1 vs 2 vs 3-skill MWLE recovery + precision + length.
- `oos_recovery/{unidim,2_skills,3_skills}/{metrics.json,oos_per_model.csv}` — per-structure
  OOS k-fold recovery (online/batch/MWLE).
- `figures/composite_vs_axes.png` — composite score vs per-axis abilities.

## Provenance

`scripts/scenario_kfold_estimator_cv.py`. Original:
`regenerated_figures/scenario_level_115_min12/{kfold,model_comparison_1v2v3_mwle_wide.csv,figures}`.
