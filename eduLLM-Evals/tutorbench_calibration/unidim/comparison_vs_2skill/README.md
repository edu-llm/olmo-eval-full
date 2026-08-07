# TutorBench scale comparison: 2-skill vs correctness-only unidim @ floor 20 / SE_ability 0.27

**STUDY-ONLY / LOCAL side-experiment.** Nothing committed or pushed. The production engine
(`tutor_cat/`, `scripts/scenario_cat_lib.py`) was only read/imported/called, never modified. All
outputs live in this folder. Both scales are compared at the **same locked operating point:
floor 20 / SE_ability 0.27**.

- **2-skill** (correctness + scaffolding) — the of-record scale.
- **correctness-only unidim** — the alternative.

## The tradeoff in one line

**Correctness-only unidim is the *same* correctness axis, only cleaner.** It wins on
SE_param / SE_total / test length / reach with essentially equal recovery, and models rank
identically (full-bank EAP θ correlation **r = 0.9994, ρ = 0.9993**). The entire, honest cost is
that **scaffolding stops being a measured construct at all** — the 2-skill scale still delivers a
strong, separate scaffolding axis (OOS r 0.932, θMAE 0.256) that unidim throws away.

## Comparison table @ 20/0.27

| metric | 2-skill correctness | 2-skill scaffolding | correctness-only unidim |
|---|---:|---:|---:|
| SE_ability (median, param-unc. bootstrap) | 0.3337 | 0.3073 | **0.3258** |
| SE_param (median) | 0.2074 | 0.1749 | **0.1126** |
| SE_total (median) | 0.4117 | 0.3592 | **0.3441** |
| OOS recovery r @20/0.27 | 0.9667 | 0.9321 | **0.9550** |
| slope @20/0.27 | 1.0187 | 0.9527 | **1.0055** |
| θ MAE @20/0.27 | 0.4368 | 0.2564 | **0.4159** |
| median test length (scenarios) | 25 | 25 | **20** |
| %reach SE_ability≤0.27 | 43.4% | 69.0% | **59.6%** |

Notes:
- **SE_ability / SE_param / SE_total** are medians from the scenario-level parameter-uncertainty
  bootstrap (N=115; eap-grid 61, n-boot 150, min-evals-per-skill 12, max-se 0.30). SE_ability is the
  posterior SE; SE_total = √(SE_ability² + SE_param²). Sources:
  `regenerated_figures/scenario_level_115_min12/param_uncertainty/2_skills/metrics.json` and
  `../experiments/07_parameter_uncertainty/param_uncertainty_metrics.json`.
- **OOS r / slope / θMAE / length / %reach** are the k=5 refit-per-fold OOS grid rows at
  floor 20 / SE_ability 0.27 (headline pool excludes flagged models). Sources:
  `reports/eap_oos_grid_tutorbench_2skill/oos_per_cell_grid.csv` (row 20/0.27) and
  `../experiments/06_floor_se_grid/oos_per_cell_grid_excl.csv` (row 20/0.27).
- **median test length** is the *joint* administered test. A 2-skill test measures both skills
  simultaneously in 25 scenarios; unidim measures the single axis in 20.
- **%reach** shown is SE_ability≤0.27 per axis. The 2-skill *both-skills* reach is only **33.6%**;
  the unidim *combined* reach (SE_ability≤0.27 **and** SE_total≤0.30) is **57.9%**. Either way unidim
  reaches the target for far more models than the 2-skill correctness axis.

## What unidim wins, and what it gives up

**Wins (vs 2-skill correctness):**
- **SE_param 0.207 → 0.113 (−46%)** — the dominant improvement. Dropping the reverse-loading
  scaffolding items removes the calibration instability that inflated the 2-skill correctness axis
  (clamped negative loadings fall 349 → 33; worst-case error-bar inflation 3.22× → 1.56×).
- **SE_total 0.412 → 0.344 (−16%)**, entirely on the back of the SE_param gain.
- **Shorter tests**: median 20 vs 25 scenarios.
- **Higher reach**: 59.6% vs 43.4% (SE_ability), i.e. it hits the precision target for ~16 pp more
  models on the correctness axis (and 57.9% vs 33.6% both-skills combined).
- **Recovery essentially unchanged**: OOS r within 0.012 (0.955 vs 0.967), slope 1.006 vs 1.019,
  and **lower** θMAE (0.416 vs 0.437).
- **Rehabilitates `salamandra-7b-instruct`** (a 2-skill weak-flagged model): under unidim it lands
  on the recovery line at θ ≈ −0.4 (green ring in the scatter), so the headline pool no longer has to
  special-case it.

**Gives up:**
- **Scaffolding as a measured construct — entirely.** The 2-skill scale carries a genuinely separate
  scaffolding axis that recovers well (OOS r 0.932, slope 0.953, θMAE 0.256, 69% reach). Unidim does
  not estimate it at all. If reporting a scaffolding ability is a product requirement, unidim cannot
  supply it.
- **SE_ability (information) is basically identical** (0.334 → 0.326, −2.4%): unidim is not adding
  correctness information, it is removing fit noise. So the win is fit-stability, not measurement power.

## Rank agreement: same axis or different axis?

Full-bank EAP θ (every graded criterion, **stop-independent**) computed from each fitted bank with the
shared production estimator `scenario_cat_lib.eap_all_models` (clamp negative-loading policy; 2-skill
grid 61²/dim → correctness column; unidim grid 241):

- **Headline pool (excl `Qwen/Qwen1.5-1.8B`, N=114): Pearson r = 0.9994, Spearman ρ = 0.9993.**
- All 115 models: Pearson r = 0.9994, Spearman ρ = 0.9993. OLS slope 0.967.

**Read: "same axis, cleaner."** θ correlates ~1.0 and models rank the same — correctness-only unidim
is not a different construct, it is the 2-skill correctness axis with the scaffolding-induced
calibration noise stripped out.

## Decision framing

If the of-record deliverable needs **only a correctness ability**, the correctness-only unidim scale
strictly dominates the 2-skill correctness axis: same ranking, same recovery, tighter error bars,
shorter tests, more models reaching target. If a **separate scaffolding ability** is a required
output, that is the one thing unidim cannot provide — and that is the whole decision.

## Files

```
../scripts/build_comparison.py            # this comparison builder (read-only on engine + banks)
comparison_table_20_0.27.csv / .md        # the 20/0.27 metric table
fullbank_eap_theta_per_model.csv          # per-model 2-skill vs unidim full-bank EAP theta
comparison_summary.json                   # all numbers + rank-agreement stats + provenance
figures/se_components_grouped_bar.png     # [2-skill, unidim] x [SE_ability, SE_param, SE_total], 0.27 target line, SE_param gap
figures/rank_agreement_scatter.png        # 2-skill correctness theta vs unidim theta (r, rho)
figures/recovery_efficiency_comparison.png# OOS r / thetaMAE / length / %reach, side by side
```

## Reproduce

```powershell
uv run python tutorbench_calibration\unidim\scripts\build_comparison.py --workers 6
```

Reads (read-only): the two fitted banks
(`data/TutorBench/rubrics_qmatrix_calibrated_2skill_115_fitted.jsonl`,
`../bank/rubrics_qmatrix_correctness_only_unidim_115_fitted.jsonl`), the response matrix
(`staging/response_matrix_full_nonopt_115.csv`), the two param-uncertainty `metrics.json`, and the two
OOS per-cell grids. Nothing else is touched.
