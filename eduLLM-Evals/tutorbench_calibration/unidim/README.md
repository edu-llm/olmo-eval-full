# TutorBench — correctness-only unidimensional scale (CANONICAL)

**correctness-only unidim, floor 20 / SE 0.27, N=114 headline.**

This is the **canonical TutorBench scale**. A single latent **correctness** ability is fit as a
unidimensional 2PL over the TutorBench criteria that load on correctness (scaffolding is dropped as
a measured construct). The scale is the 2-skill correctness axis with the scaffolding-induced
calibration noise stripped out — same ranking, cleaner error bars, shorter tests, higher reach.
See `comparison_vs_2skill/` for the head-to-head, and `../two_skill_OUTDATED/` for the superseded
2-skill package.

**Status:** STUDY / reporting only, LOCAL. The production engine (`tutor_cat/`,
`scripts/scenario_cat_lib.py`) and the banks were only read/called, never modified. Nothing committed.

---

## Locked operating point + methodology

- **Scale:** correctness-only **unidimensional 2PL**. Bank built by
  `scripts/build_correctness_only_bank.py` from the of-record 2-skill fitted bank, keeping only
  criteria that load on correctness and dropping every scaffolding-loading criterion (2842 kept of
  3666; 824 dropped), then refitting with the exact config below on
  `staging/response_matrix_full_nonopt_115.csv`. Clamped negative loadings fall **349 → 33**.
- **Fit:** `calibrate_mirt.fit_m2pl_em`, 1 dim, `fit_grid=7`, `ridge=1e-2`, `negative_policy=clamp`.
- **CAT stop:** EAP-posterior honest **1-D marginal SD**, dense **321-node** grid, info-plateau
  **δ=0.005 / W=3**, **cap 70**, **MWLE θ** at stop.
- **Operating point (LOCKED): floor(min_scenarios)=20, SE_ability target=0.27** — directly
  comparable to the locked 2-skill 20/0.27; selected out-of-sample on the floor×SE grid.
- **OOS:** k=5 model-fold, seed 20260729, refit-per-fold.
- **N=115 models; headline N=114 EXCLUDES only `Qwen/Qwen1.5-1.8B`** (a genuinely info-limited weak
  1.8B outlier). `BSC-LT/salamandra-7b-instruct` is **rehabilitated** under this scale and kept in
  the headline (it lands on the recovery line).

## Headline numbers @ 20/0.27 (N=114)

| quantity | value |
|---|---|
| OOS recovery r | **0.955** |
| slope | **1.006** |
| θ-MAE | **0.416** |
| median test length (scenarios) | **20** |
| %reach (SE_ability≤0.27) | **59.6%** |
| median SE_total | **0.275** |
| median SE_ability | 0.258 |
| fixed SE_param offset | **0.113** |
| adaptive vs random to reach SE≤0.27 | 20 vs 68 scenarios (**~3.4×** fewer) |
| p-IRT pass-rate MAE | **0.020** (r=0.975) |
| leaderboard top (headline) | `Qwen/Qwen3-1.7B` θ=2.60, SE_total=0.275 |
| leaderboard bottom (headline) | `openai-community/gpt2-large` θ=−5.78 |

Machine-readable: `summary.json`.

## Why unidim is canonical (vs 2-skill)

- **Same axis, cleaner:** full-bank EAP θ correlates with the 2-skill correctness axis at
  **r = 0.9994 (ρ = 0.9993)** — models rank identically, so this is not a different construct.
- **SE_param 0.207 → 0.113 (−46%)** and **SE_total 0.412 → 0.344 (−16%)**: the win is fit-stability,
  from dropping the reverse-loading scaffolding items that destabilized the 2-skill calibration.
- **Shorter tests** (median 20 vs 25 scenarios) and **higher reach** (59.6% vs 43.4% correctness).
- **Recovery essentially unchanged** (r within 0.012; slope 1.006 vs 1.019; **lower** θMAE).
- **The one thing it gives up:** scaffolding as a separately-measured construct (the 2-skill scale
  still delivers a strong, separate scaffolding axis). Full head-to-head: `comparison_vs_2skill/`.

## Layout

```
unidim/
  README.md                 this index
  summary.json              machine-readable op-point / config / all headline numbers
  bank/
    rubrics_qmatrix_correctness_only_unidim_115_fitted.jsonl   the fitted bank (2842 criteria; dim=correctness)
    bank_fit_summary.json                                      kept/dropped, clamped-neg, max-a, loglik/AIC/BIC
  scripts/                  build + reproduction scripts (provenance; paths reflect the original build layout)
    build_correctness_only_bank.py      bank builder (reuses calibrate_mirt, read-only)
    oos_grid_full_unidim_correctness_only.py   full floor×SE OOS grid (exp 06)
    build_of_record_f20se27.py          of-record recovery/leaderboard/SE @20/0.27 (exp 04,05,07,08,09)
    make_se_comparison.py               3-scale SE-component comparison
    build_comparison.py                 2-skill vs unidim comparison (comparison_vs_2skill/)
  experiments/
    04_efficiency_vs_random/  adaptive vs random efficiency (~20 vs 68 scenarios, ~3.4×)
    05_oos_recovery/          headline OOS recovery (r 0.955 / slope 1.006 / θMAE 0.416)
    06_floor_se_grid/         floor×SE OOS grid CSVs + heatmaps (op-point selection basis)
    07_parameter_uncertainty/ SE components + SE_ability-vs-SE_total bar (SE_param 0.113)
    08_leaderboard/           deployed leaderboard @ 20/0.27
    09_pirt_mae/              p-IRT / pass-rate MAE (0.020)
  comparison_vs_2skill/       2-skill vs correctness-only unidim @ 20/0.27 (rank agreement r=0.9994)
```

## Experiment index

| slot | description | headline | path |
|---|---|---|---|
| `04_efficiency_vs_random` | adaptive vs random efficiency | 20 vs 68 scenarios (~3.4×) | `experiments/04_efficiency_vs_random/adaptive_vs_random_efficiency_f20se27.png` |
| `05_oos_recovery` | OOS recovery scatter (MWLE θ vs OOS reference) | r 0.955 / slope 1.006 / θMAE 0.416 | `experiments/05_oos_recovery/oos_recovery_correctness_f20se27.png` |
| `06_floor_se_grid` | floor×SE OOS grid (selection basis) | row 20/0.27 selected | `experiments/06_floor_se_grid/figures/oos_grid_full_heatmaps.png` |
| `07_parameter_uncertainty` | SE components / SE_ability vs SE_total | SE_ability 0.258, SE_total 0.275, SE_param 0.113 | `experiments/07_parameter_uncertainty/se_ability_vs_total_bars_f20se27.png` |
| `08_leaderboard` | deployed leaderboard @ 20/0.27 | top Qwen3-1.7B θ=2.60 | `experiments/08_leaderboard/leaderboard_correctness_f20se27.png` |
| `09_pirt_mae` | p-IRT / pass-rate MAE | MAE 0.020 (r=0.975) | `experiments/09_pirt_mae/pirt_pred_vs_actual_f20se27.png` |
| `comparison_vs_2skill` | 2-skill vs unidim @ 20/0.27 | rank agreement r=0.9994 | `comparison_vs_2skill/README.md` |

## Bank

`bank/rubrics_qmatrix_correctness_only_unidim_115_fitted.jsonl` (2842 fitted criteria; dim =
`correctness`; negatives preserved in-file, clamp applied at consume time). Fit provenance:
`bank/bank_fit_summary.json`.

## Notes on scripts

`scripts/*.py` are retained as **provenance / reproduction record**. Their internal default paths
reflect the original `reports/eap_unidim_correctness_only_tutorbench/` build layout (e.g. a sibling
`oos_grid_full/` and `param_uncertainty/`, and an `import oos_grid_unidim_correctness_only`), so
re-running them as-is from this new location requires repointing those paths. The **artifacts** they
produced are the of-record deliverables under `experiments/`, `bank/`, and `comparison_vs_2skill/`.
