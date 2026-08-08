# InfoBench calibration — CANONICAL unidimensional scale (floor 15 / SE 0.25)

**unidim `instruction_following`, floor 15 / SE_post 0.25, N=52; headline excludes 4 weakly-identified models (N=48).**

This is the **canonical (and only) InfoBench scale**: a single latent **instruction_following**
ability fit as a unidimensional 2PL over the InfoBench criteria (source skills `content`, `format`,
`number`, `style`, `linguistic` collapse into one axis). A scenario is administered as a **testlet
bundle** of its per-criterion items; the real production engine
(`tutor_cat.engine.run_evaluation` via `scripts/scenario_cat_lib.py`) selects a whole scenario per
CAT step and updates ability. Nothing here reimplements CAT selection or the item-response update.

This package mirrors the sibling `tutoreval_calibration/` layout (which is itself the unidim mirror of
`tutorbench_calibration/unidim/`). InfoBench has exactly one canonical scale — unidim — so this
package root *is* the unidim scale.

**Status:** STUDY / reporting only, LOCAL. The production engine (`tutor_cat/`,
`scripts/scenario_cat_lib.py`) and the calibration inputs were only read/called, never modified.
Nothing committed by this study.

---

## Locked operating point + methodology

- **Scale:** unidimensional 2PL, single latent `instruction_following`
  (`q_modeled = {"instruction_following": 1}`, `axis: "unidim"`).
- **Fit:** `calibrate_mirt.fit_m2pl_em`, 1 dim, **log-shrinkage-2PL λ16** (`log_a_prior_sd = 0.25`),
  **401-node** `normal_trapezoid` (bound 8), `returned_iterate` convergence (`parameter_tol 5e-5`,
  2 consecutive passes), export drops nonpositive / `a > 6`. Config
  `configs/infobench_calibration_cat_2pl_only_v1.json` (spec `log_shrinkage_2pl_lambda16`).
  Matrix N=52 models × 2250 criteria × 500 scenarios, fill 0.9998.
- **CAT stop:** EAP-posterior honest **1-D marginal SD**, dense **801-node** `normal_trapezoid`
  grid (bound 8), info-plateau **δ = 0.005 / W = 3**, **cap 70**, **MWLE θ** at stop. Stop priority:
  precision → plateau → cap.
- **Operating point (LOCKED): floor(min_scenarios) = 15, SE_post target = 0.25** — selected
  out-of-sample on the floor×SE grid (experiment 06).
- **OOS:** k = 5 model-fold, seed 20260729, refit-per-fold (log-shrinkage-2PL λ16, 401-node
  `normal_trapezoid`, `returned_iterate`). Reference θ = full fold-bank EAP on fold params
  (§8.3 held-out recovery).
- **N = 52 models; headline N = 48 EXCLUDES 4 weakly-identified models** (see weak-exclusion note).
  The weak set is identified **from the run** (by SE, not θ), not assumed.

## Headline numbers @ 15/0.25 (excl-weak, N=48)

| quantity | value |
|---|---|
| OOS recovery r | **0.972** |
| slope | **1.070**  (>1 = mild scale **expansion**; see note) |
| θ-MAE | **0.430** |
| median test length (scenarios) | **17** |
| length mean ± SD (max; # overran floor) | 17.7 ± 3.3 (max 26; 27/48 overran) |
| %reach (SE_post ≤ 0.25) | **85.4%** |
| SE_post mean ± SD (median) | 0.242 ± 0.022 (0.243) |
| SE_total mean ± SD (median) | **0.260 ± 0.024** (0.258) |
| SE_param (sound, CAT-admin median) | **≈ 0.085** |
| p-IRT pass-rate MAE | **0.050** (r = 0.973, OLS slope 1.043) |
| adaptive vs random to reach median SE_post ≤ 0.25 | **~17 vs ~30** scenarios (~1.76×, 13 saved) |
| leaderboard top (headline) | `meta-llama/Llama-3.2-3B-Instruct` θ = 3.764 (SE_total 0.277) |
| leaderboard bottom (headline) | `HuggingFaceTB/SmolLM2-360M` θ = −3.896 |

**All-52 (weak included):** r = **0.977**, slope = **1.140**, θ-MAE = 0.507, median length 17
(mean **17.8 ± 3.2**, max 26, **31/52** overran the floor), %reach = **78.8%**,
SE_total = **0.276** (median 0.259), p-IRT MAE = **0.047** (r = 0.977). Machine-readable:
`summary.json`.

**%reach is SE_post-only:** reach = SE_post (EAP posterior SD at stop) ≤ target **ONLY**. SE_total is
reported separately as a precision number, **not** a gate. InfoBench used this SE_post-only reach
rule *from the start*, so — unlike the sibling TutorEval package — there is **no combined-gate
"reach-corrected" pass and no `*_reachfix` heatmap** here; the grid's `%reach` is already correct by
construction. See `experiments/06_floor_se_grid/`.

---

## ⚠️ SLOPE-EXPANSION note (read before quoting absolute θ)

The OOS recovery **slope is ~1.07 (excl-weak) / ~1.14 (all-52)** — i.e. **> 1**, a **mild scale
EXPANSION** of the ability axis. This is the **opposite** of TutorEval's ~0.80 compression:

- **Direction:** InfoBench's bank carries *more* per-scenario information than the reference scale
  implies, so recovered θ is mildly stretched away from 0 rather than shrunk toward it.
- **What is still faithful:** **rankings and pass-rate prediction are unaffected** — recovery
  r = 0.97–0.98 and p-IRT pass-rate MAE = 0.047–0.050 (OLS slope 1.043, r 0.973). The expansion is
  mild and uniform; only the **absolute-θ units** are slightly stretched.
- **If you need absolute θ:** apply an affine rescale (divide the CAT θ by the slope, ≈ ÷1.07). For
  ranking, pass-rate, or relative comparisons, use θ as-is.

This distinguishes InfoBench (slope ≈ 1.07–1.14) from **TutorEval** (slope ≈ 0.80, compression) and
places it close to **TutorBench** (slope ≈ 1.0). See `experiments/05_oos_recovery/`.

---

## Weak-exclusion note (headline excludes N=4 by SE, not θ)

Four models are excluded from the headline **by their SE (never by their θ / rank)**:

- `allenai/OLMo-1B-hf`
- `ai-forever/mGPT`
- `BEE-spoke-data/smol_llama-220M-GQA-fineweb_edu`
- `BEE-spoke-data/smol_llama-220M-openhermes`

Criterion = **high SE_total / plateaus that never reach the SE target**. These four sit at the
extreme low tail (θ ≤ −4.4) and are **borderline @ SE_total ≈ 0.375**; per the decision they are
**kept excluded** from the headline. They remain in the leaderboard (greyed) and in the all-52
numbers. Excluding them tightens the recovery slope from 1.140 → 1.070 and lifts %reach from
78.8% → 85.4%.

---

## SE_param correction note (IMPORTANT)

The of-record `SE_total = √(SE_post² + SE_param²)` uses the **SOUND observed-information parametric
bootstrap** for `SE_param`: per-item parameter covariance from the observed information at the frozen
fit, perturb the administered item params, re-EAP, take the SD across draws — no person resampling,
no refit (full-data, reused across folds). Its **CAT-admin median ≈ 0.085** (full-admin floor median
≈ 0.041).

The earlier teammate-reported **`SE_param ≈ 0.19` is STALE and superseded.** It came from a
**nonparametric person-resample-refit** bootstrap (`scripts/scenario_param_uncertainty.py` as it
stands on `frq/infobench`), which resamples calibration models with replacement and refits the whole
bank per replicate — a different, inflated quantity. Use the sound ≈ 0.085. See **Part 2 / script
drift** below.

> **Script status (frq/infobench):** the intended swap of
> `scripts/scenario_param_uncertainty.py` to the sound observed-information version from `frq/bridge`
> was **NOT applied** because that version is **incompatible with this branch** (different CLI,
> different output files, and it depends on `scenario_cat_lib` functions absent here — which must not
> be modified). The stale script was **left in place** to avoid breaking the InfoBench study
> pipeline and the tracked test suite. The *sound* SE_param used throughout this package instead comes
> from the grid's own in-process observed-information bootstrap (`scripts/run_oos_grid.py`, class
> `SEParam`, mirroring `recompute_se_param.py`), NOT from the stale CLI script.

---

## Layout

```
infobench_calibration/
  README.md                 this index (canonical)
  FLOW_PACKAGE.md           package manifest -> flow/uni-frq (unidim)
  summary.json              machine-readable op-point / config / all headline numbers (all-52 AND excl-weak)
  bank/
    response_matrix.csv     calibration INPUT the grid fit from (N=52 x 2250; sha256 087948fc…)
    judge_manifest.json     frozen-judge provenance for the matrix (Qwen labels)
    bank_fit_summary.json   fit provenance (matrix sha, λ16 / 401-node config, loglik note, low-n + no-persisted-param-bank caveat)
  scripts/                  reproduction scripts (provenance; paths reflect the original build layout)
    run_oos_grid.py               full floor x SE OOS grid + sound SE_param (exp 06)
    build_of_record_f15se25.py    of-record recovery/leaderboard/SE/p-IRT @ 15/0.25 (exp 04,05,07,08,09)
  experiments/
    04_efficiency_vs_random/  adaptive vs random efficiency (~17 vs 30 scenarios, ~1.76x)
    05_oos_recovery/          headline OOS recovery (r 0.972 / slope 1.070 / theta-MAE 0.430; slope>1 EXPANSION)
    06_floor_se_grid/         floor x SE OOS grid CSVs + heatmaps + DECISION_TABLE (op-point selection basis)
    07_parameter_uncertainty/ SE components + SE_post-vs-SE_total bars (sound SE_param median 0.085) + length_and_se_distribution.csv
    08_leaderboard/           deployed leaderboard @ 15/0.25 (weak greyed)
    09_pirt_mae/              p-IRT / pass-rate MAE (0.050 excl-weak; OLS fit-line slope 1.043)
```

## Experiment index

| slot | description | headline | path |
|---|---|---|---|
| `04_efficiency_vs_random` | adaptive vs random efficiency | ~17 vs 30 scenarios (~1.76×) | `experiments/04_efficiency_vs_random/adaptive_vs_random_efficiency_f15se25.png` |
| `05_oos_recovery` | OOS recovery scatter (MWLE θ vs OOS reference) | excl-weak r 0.972 / slope 1.070 / θMAE 0.430 | `experiments/05_oos_recovery/oos_recovery_ability_f15se25.png` |
| `06_floor_se_grid` | floor×SE OOS grid (selection basis) | row 15/0.25 selected; %reach 85.4% (excl-weak) | `experiments/06_floor_se_grid/figures/oos_grid_heatmaps.png` |
| `07_parameter_uncertainty` | SE components / SE_post vs SE_total (mean±SD + median+IQR) | SE_post 0.242, SE_total 0.260, sound SE_param 0.085 | `experiments/07_parameter_uncertainty/se_ability_vs_total_bars_f15se25.png` |
| `08_leaderboard` | deployed leaderboard @ 15/0.25 | top Llama-3.2-3B-Instruct θ=3.764 | `experiments/08_leaderboard/leaderboard_f15se25.png` |
| `09_pirt_mae` | p-IRT / pass-rate MAE (with OLS fit-line) | MAE 0.050 (r 0.973, OLS slope 1.043) | `experiments/09_pirt_mae/pirt_pred_vs_actual_f15se25.png` |

## Bank

InfoBench has **no persisted per-item fitted-param jsonl**: the of-record OOS grid refits the
log-shrinkage-2PL bank on-the-fly from the frozen **response matrix** (per-fold for OOS; full-data
for the sound SE_param). The `bank/` here therefore ships the exact **calibration input** the grid
consumed — `response_matrix.csv` (sha256 `087948fcaa884cde…`) and its `judge_manifest.json` — plus
`bank_fit_summary.json` (fit config, shas, loglik note, low-n and no-persisted-param-bank caveats).
To materialise deployment item params, re-run `scripts/run_oos_grid.py`, which fits with the λ16 /
401-node config above. The scenarios the tutor model responds to are `data/InFoBench/scenarios.jsonl`
(not copied here; referenced from `FLOW_PACKAGE.md`).

## Notes on scripts

`scripts/*.py` are retained as **provenance / reproduction record**. Their internal default paths
reflect the original `reports/infobench_oos_grid_unidim/` build layout (they import the production
`scripts/scenario_cat_lib.py` + `scripts/calibrate_mirt.py` read-only and read the matrix from
`runs/calibration/InFoBench_full_20260804/inputs/response_matrix.csv`, which is `runs/`-gitignored),
so re-running them as-is from this new location requires repointing those paths. The **artifacts**
they produced are the of-record deliverables under `experiments/`, `summary.json`, and `bank/`.
