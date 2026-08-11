# TutorEval calibration — CANONICAL unidimensional scale (floor 15 / SE 0.22)

**unidim `ability`, floor 15 / SE_ability 0.22, N=52; headline excludes 4 weakly-identified models.**

This is the **canonical (and only) TutorEval scale**: a single latent **ability** fit as a
unidimensional 2PL over the TutorEval criteria. A scenario is administered as a **testlet bundle** of
its per-criterion items; the real production engine
(`tutor_cat.engine.run_evaluation` via `scripts/scenario_cat_lib.py`) selects a whole scenario per
CAT step and updates ability. Nothing here reimplements CAT selection or the item-response update.

Unlike the sibling `tutorbench_calibration/` (which carries a `unidim/` scale plus a superseded
`two_skill_OUTDATED/` package), **TutorEval has exactly one canonical scale — unidim — so this
package root *is* the unidim scale** (it mirrors `tutorbench_calibration/unidim/`). A teammate's
2-skill EAP prototype was audited and **rejected** (wrong dimensionality); see the note below.

**Status:** STUDY / reporting only, LOCAL. The production engine (`tutor_cat/`,
`scripts/scenario_cat_lib.py`) and the banks were only read/called, never modified. Nothing committed.

---

## Locked operating point + methodology

- **Scale:** unidimensional 2PL, single latent `ability` (`q_modeled = {"ability": 1}`, `axis: "unidim"`).
- **Fit:** `calibrate_mirt.fit_m2pl_em`, 1 dim, `fit_grid = 7` nodes/dim, `ridge = 1e-2`,
  `negative_policy = clamp` (negatives preserved in-file, clamped at consume time). 1,186 fitted
  items, `n_persons = 52`. Reproduction loglik = −17111.06 (matches the calibration manifest).
- **CAT stop:** EAP-posterior honest **1-D marginal SD**, dense **321-node** grid (spacing 0.05 over
  [−8, 8]), info-plateau **δ = 0.005 / W = 3**, **cap 70**, **MWLE θ** at stop. Stop priority:
  precision → plateau → cap → bank_exhausted.
- **Operating point (LOCKED): floor(min_scenarios) = 15, SE_ability target = 0.22** — selected
  out-of-sample on the floor×SE grid (experiment 06).
- **OOS:** k = 5 model-fold, seed 20260729, refit-per-fold (1-D, ridge 0.01, fit_grid 7, clamp).
- **N = 52 models; headline N = 48 EXCLUDES 4 weakly-identified models** (best-achievable SE_total at
  the cap still > 0.30): `BEE-spoke-data/smol_llama-220M-GQA-fineweb_edu`, `ai-forever/mGPT`,
  `allenai/OLMo-1B-hf`, `ibm-granite/granite-3.1-2b-instruct`. The weak set is identified **from the
  run**, not assumed.

## Headline numbers @ 15/0.22 (excl-weak, N=48)

| quantity | value |
|---|---|
| OOS recovery r | **0.950** |
| slope | **0.801** |
| θ-MAE | **0.333** |
| median test length (scenarios) | **15** |
| median SE_ability | 0.177 |
| median SE_total | **0.217** |
| fixed SE_param offset | **0.112** |
| corrected %reach (SE_ability ≤ 0.22) | **97.9%** |
| adaptive vs random to reach SE ≤ 0.22 | 10 vs 79 scenarios (**~7.9×** fewer) |
| p-IRT pass-rate MAE | **0.036** (r = 0.953) |
| leaderboard top (headline) | `tiiuae/Falcon3-3B-Instruct` θ = 2.60 |
| leaderboard bottom (headline) | `BEE-spoke-data/smol_llama-220M-openhermes` θ = −2.98 |

All-52 (weak included): r = 0.970, slope = 0.813, θ-MAE = 0.359, median SE_total = 0.223,
corrected %reach = 92.3%. Machine-readable: `summary.json`.

**%reach is corrected:** reach = SE_ability (EAP posterior SD at stop) ≤ target **ONLY**. SE_total is
reported separately as a precision number, **not** a gate. The old combined-gate definition
(SE_ability ≤ target AND SE_total ≤ 0.30) understated reach by ~4–6 pp and produced a spurious
"reach peaks then drops at SE = 0.30" artifact; that is resolved here. See
`experiments/06_floor_se_grid/` (`oos_per_cell_grid_reach_corrected*.csv`,
`figures/oos_grid_heatmaps_reachfix.png`).

---

## ⚠️ SLOPE CAVEAT (read before quoting absolute θ)

The OOS recovery **slope is ~0.80** — a uniform **~20% absolute-scale compression** of the ability
axis. This is a **genuine bank property, not tail noise or an artifact**:

- **Not tail-driven:** the core slope (≈ 0.82) matches the tail slope (≈ 0.82) — the compression is
  uniform across the range, not caused by a few extreme models.
- **Structural, not sample noise:** the slope only creeps up with a longer/stricter test — ≈ **0.84**
  at floor 20, ≈ **0.86** at floor 25 — and **never reaches ~1.0 even at full length**. It is a
  property of how much per-scenario information the TutorEval bank carries, not of the stop rule.
- **What is still faithful:** **rankings and pass-rate prediction are unaffected** — recovery
  r = 0.95–0.98 and p-IRT pass-rate MAE = 0.036. Only the **absolute-θ units** are shrunk toward 0.
- **If you need absolute θ:** apply an affine rescale (divide the CAT θ by the slope, ≈ ÷0.80).
  For ranking, pass-rate, or relative comparisons, use θ as-is.

This distinguishes TutorEval from **TutorBench**, whose canonical unidim scale recovers at
**slope ≈ 1.0** (`tutorbench_calibration/unidim/`, slope 1.006). TutorEval's shorter, lower-information
bank compresses the absolute scale; TutorBench's does not.

---

## Rejected 2-skill prototype (do NOT cite as current)

A teammate's EAP-posterior stop prototype at
`reports/eap_stop_prototype_20260806/TutorEval/` (commit `ea65c15`) ran TutorEval against a **2-skill
bank** (`conceptual_understanding` + `quantitative_procedural`). That is **NOT TutorEval's canonical
scale**: the flow-package owner graduates **only the unidimensional scale** (the N=52 sample is far
below the persons needed to identify multi-dimensional loadings — see the identifiability caveat in
`bank/bank_fit_summary.json`). The 2-skill prototype was **audited and rejected for wrong
dimensionality**. Those files are committed on this branch and left in place as a rejected record;
this canonical unidim package supersedes them. Point readers here.

---

## Layout

```
tutoreval_calibration/
  README.md                 this index (canonical)
  FLOW_PACKAGE.md           graduation manifest -> flow/uni-frq (unidim)
  summary.json              machine-readable op-point / config / all headline numbers
  bank/
    rubrics_qmatrix_final_unidim_fitted.jsonl   fitted bank (1,186 criteria; dim = ability)
    bank_fit_summary.json                       fit provenance (matrix sha, loglik, config, low-n caveat)
  scripts/                  reproduction scripts (provenance; paths reflect the original build layout)
    run_oos_grid_unidim.py       full floor x SE OOS grid (exp 06)
    build_of_record_f15se22.py   of-record recovery/leaderboard/SE/p-IRT @ 15/0.22 (exp 04,05,07,08,09)
  experiments/
    04_efficiency_vs_random/  adaptive vs random efficiency (~10 vs 79 scenarios, ~7.9x)
    05_oos_recovery/          headline OOS recovery (r 0.950 / slope 0.801 / theta-MAE 0.333)
    06_floor_se_grid/         floor x SE OOS grid CSVs + reach-corrected heatmaps + DECISION_TABLE
    07_parameter_uncertainty/ SE components + SE_ability-vs-SE_total bar (SE_param 0.112)
    08_leaderboard/           deployed leaderboard @ 15/0.22
    09_pirt_mae/              p-IRT / pass-rate MAE (0.036)
```

## Experiment index

| slot | description | headline | path |
|---|---|---|---|
| `04_efficiency_vs_random` | adaptive vs random efficiency | 10 vs 79 scenarios (~7.9×) | `experiments/04_efficiency_vs_random/adaptive_vs_random_efficiency_f15se22.png` |
| `05_oos_recovery` | OOS recovery scatter (MWLE θ vs OOS reference) | r 0.950 / slope 0.801 / θMAE 0.333 | `experiments/05_oos_recovery/oos_recovery_ability_f15se22.png` |
| `06_floor_se_grid` | floor×SE OOS grid (selection basis) + reach fix | row 15/0.22 selected; corrected %reach 97.9% | `experiments/06_floor_se_grid/figures/oos_grid_heatmaps_reachfix.png` |
| `07_parameter_uncertainty` | SE components / SE_ability vs SE_total | SE_ability 0.177, SE_total 0.217, SE_param 0.112 | `experiments/07_parameter_uncertainty/se_ability_vs_total_bars_f15se22.png` |
| `08_leaderboard` | deployed leaderboard @ 15/0.22 | top Falcon3-3B-Instruct θ=2.60 | `experiments/08_leaderboard/leaderboard_f15se22.png` |
| `09_pirt_mae` | p-IRT / pass-rate MAE | MAE 0.036 (r=0.953) | `experiments/09_pirt_mae/pirt_pred_vs_actual_f15se22.png` |

## Bank

`bank/rubrics_qmatrix_final_unidim_fitted.jsonl` — 1,186 fitted criteria; dim = `ability`
(`q_modeled = {"ability": 1}`); negatives preserved in-file, clamp applied at consume time. Fit
provenance: `bank/bank_fit_summary.json` (matrix sha256, reproduction loglik −17111.06, low-n
identifiability caveat). The scenarios the tutor model responds to are
`data/TutorEval/scenarios_final.jsonl` (not copied here; referenced from `FLOW_PACKAGE.md`).

## Notes on scripts

`scripts/*.py` are retained as **provenance / reproduction record**. Their internal default paths
reflect the original `reports/eap_oos_grid_tutoreval_unidim/` build layout (e.g. the sibling
`_input/response_matrix_tutoreval_unidim.csv` and a `figures/` output dir), so re-running them
as-is from this new location requires repointing those paths — and requires the of-record response
matrix `runs/judge/TutorEval/response_matrix.csv` (S3-origin; `runs/` is gitignored). The
**artifacts** they produced are the of-record deliverables under `experiments/`, `summary.json`,
and `bank/`.
