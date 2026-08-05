# Bridge calibration — SCENARIO-LEVEL study

Scenario-level recalibration of the Bridge benchmark, built to mirror the TutorBench
scenario CAT pipeline (`scripts/scenario_*` + `tutor_cat.engine`). This is now the canonical
Bridge calibration study, and **supersedes the earlier item/criterion-level study**
(recoverable at git commit `7a394af`), which was developed and compared against before being
retired in favour of this scenario-level design.

## Design (locked)

- **Testlet bundle administration.** A scenario is administered as a *bundle* of its
  per-criterion IRT items (grouped by scenario). The real engine
  (`tutor_cat.engine.run_evaluation` via `scripts/scenario_cat_lib.py`) selects a whole
  scenario per CAT step and updates ability from all its criteria. No polytomous collapse.
- **Bank = all 250 scenarios** (no source dedup). `source_id` is used only for leakage-safe
  fold grouping (recorded in `build_manifest.json`).
- **Dimensionality: 1D** (derived; see below). 5D kept exploratory only.
- **CAT-pool exclusions:** the A3 affective safety-gate criterion (`criterion_code=="A3"`,
  the `c15` slot, 250 of them) and the extreme_a criteria (|a|>`EXTREME_A`=6 or non-finite
  in the 1D fit) are excluded from **administration and θ scoring**, but kept in the
  **calibration fit** and reporting (`exclusion_mask.json`).
- **Recovery CV:** OOS **model/person folds**, k=5, seed 20260729.
- **Deployment scoring seed: FIXED production seed** (locked). Single-run 12-scenario CAT
  scores pin the RNG so first-item/tie-break choices are reproducible and every model is scored
  under the same selection RNG; the ~0.10 measurement SE is still reported as the honest
  uncertainty. (Seed-averaging over K runs was considered and rejected for K× cost.) Affects
  only deployed single-run CAT scoring, not the full-bank leaderboard.

### Differences from the earlier item-level study (git `7a394af`)

| aspect | item-level | scenario-level (this) |
|---|---|---|
| unit of administration/scoring/stop | one **criterion** at a time | whole **scenario** (testlet of ~18 criteria) |
| bank | source-deduped to ~162 scenarios' criteria | **all 250 scenarios**, source only for folds |
| operating point | SE 0.15 / floor **20 criteria** (≈1 scenario) — VOID | **min_scenarios=12 / SE 0.15** (re-derived) |
| `SE_param` | per-criterion (~0.02) | scenario administration, full-bank floor ~0.033 |
| engine | item-level harness | production `tutor_cat` engine (unmodified) |

## Bank build (`build_manifest.json`)

- 250 scenarios / **162 unique sources** / 51 models, matrix 100% filled.
- 4795 criteria → **4197** after zero-variance filtering (559 all-fail + 39 all-pass dropped).
- ~19–20 criteria/scenario (mean 19.18); after ZV, 228 scenarios retain ≥1 fittable criterion
  (**22 all-ZV scenarios are un-administrable**, per the TutorBench `if not cids: continue`
  convention — omitted from the per-model bank, no dedup).
- Source→fold grouping: 5 folds of 50 scenarios, sources never split.

### CAT-pool exclusion counts (`exclusion_mask.json`)

- A3: 250 total (225 survive ZV); extreme_a in the 1D fit: **11** (the item-level "~44" does
  not carry over — the scenario 1D fit at ridge 1e-2 has far fewer extreme loadings).
- Administrable after ZV + A3 + extreme_a: **3961 criteria across 228 scenarios**.

## Dimensionality (`experiments/03_structures/`) — 1D selected

Confirmatory M2PL at 1/2/3/4/5 latent skills, all 250 scenarios; selection = lowest held-out
marginal log-loss, keep within 1 SE, then BIC + parsimony (NOT AIC).

| structure | dims | AIC | BIC | OOS log-loss | max\|latent corr\| |
|---|---|---|---|---|---|
| **overall_1d** | **1** | 184532 | **270771** | 0.5124 ± 0.0125 | 0.000 |
| cognitive_vs_relational | 2 | 185099 | 279547 | 0.5111 ± 0.0127 | 0.777 |
| correlation_3d | 3 | 184666 | 281796 | 0.5123 ± 0.0141 | 0.791 |
| merge_affect_comm | 4 | 184336 | 282709 | 0.5127 ± 0.0135 | 0.873 |
| full_5d | 5 | 182594 | 283350 | 0.5098 ± 0.0142 | 0.967 |

All structures within 1 SE on held-out log-loss; the latent correlation collapses toward 1.0
as dims increase (non-identifiable at N=51). BIC + parsimony → **1D**. Per-axis 5D
discriminations are all positive (0.82–0.98) — the skills carry signal but are not separable.

## Operating point (`experiments/06_floor_se_grid/`, `experiments/06b_operating_point/`)

The item-level "SE 0.15 / floor 20 criteria" is **void**. Re-derived in scenario units over a
7×7 grid (floors {0,4,6,8,12,15,20} × SE {0.08–0.30}), OOS model-fold recovery. Because a
Bridge scenario is a heavy ~18-criterion testlet, the `min_scenarios` floor binds and the SE
target is moot once floor ≥ ~12; SE=0.08 is near-unachievable (ability-SE floor ≈ 0.076).

**Locked: `min_scenarios = 12, SE target 0.15`** → ~12 scenarios / ~212 criteria, OOS
r ≈ 0.952. No low-floor + tight-SE cell reaches r ≥ 0.95 at fewer scenarios (tightening SE at
a low floor just lengthens the test to the same place). Higher precision is available at
floor 15 (r 0.959) / 20 (r 0.968) for longer tests.

## SE_param regime (`experiments/07_parameter_uncertainty/`)

Observed-information parametric bootstrap (B=300, NOT jackknife). Full administrable-bank
**SE_param floor: mean 0.033, median 0.027** (tiny — 1D θ is pinned by ~3961 criteria). It
grows for shorter tests: ~0.05 at the locked point, ~0.13 at a 1-scenario test. vs item-level
~0.02 (same order at full bank) and TutorBench ~0.2 (only matched by Bridge at short lengths).
`SE_total = √(SE_ability² + SE_param²)`.

## Headline results

- **Recovery (exp 05, OOS, locked point, MWLE):** r = **0.952** [0.923, 0.974], slope 0.877,
  θ-MAE 0.285; p-IRT pass r = **0.935**, pass-MAE 0.047; mean length 12 scenarios / 212 criteria.
  Reference θ uses a fine uniform EAP grid (321 nodes over ±8; continuous, no GH quantization).
- **Efficiency (exp 04):** adaptive ≥ random at **every** test length (mean r gap +0.023).
  Adaptive reaches r ≈ 0.95 at ~4 scenarios; random needs ~6. Online SE at L=4: 0.171 (adaptive)
  vs 0.310 (random).
- **Estimator (exp 10):** online r=0.940/slope 0.810, batch-EAP 0.952/0.866, MWLE 0.952/0.877,
  MLE 0.952/0.879. MLE's slope is nominally closest to 1 but ties MWLE within noise; **MWLE**
  recommended (robust — plain MLE diverges on all-pass/all-fail administrations).
- **Order/seed (exp 11):** across 8 seeds at the locked point, mean θ SD = **0.107**
  (median 0.095, max 0.443), ≈ the achieved ability SE (~0.10). Seed/order adds variance on the
  order of measurement error; a fixed production seed removes it. Affects only the deployment
  CAT, not the full-bank leaderboard.
- **Ridge sensitivity (exp 12):** θ rank highly stable across ridge {1e-3,1e-2,1e-1} × grid
  {5,7,9} (min corr 0.988). **Keep ridge = 1e-2** (within noise of the best; keeps extreme_a
  low). Optional: ridge=0.1 drains extreme_a to 0 with marginally higher OOS r (within noise).

## Leaderboard (`experiments/08_leaderboard/`, `model_leaderboard.csv`)

**Full administrable bank** scoring (all 228 scenarios / 3961 non-excluded criteria per model),
NOT the 12-scenario CAT. 1D fine-EAP θ with SE_total bars.

- θ range [-4.42, 2.50]; SE_total range [~0.00, 0.148] (small — full-bank θ is well-pinned).
- Top: meta-llama/Llama-3.2-3B-Instruct (2.50), tiiuae/Falcon3-3B-Instruct (2.18),
  pankajmathur/orca_mini_v9_5_3B-Instruct (1.65).
- Bottom: allenai/OLMo-1B-hf (-3.84), Qwen/Qwen1.5-1.8B (-4.42).

## Layout

```
bridge_calibration/
  build_manifest.json                 bank build + source->fold grouping
  fit_manifest.json                   core 1D/5D fit (loglik/AIC/BIC, latent corr)
  exclusion_mask.json                 A3 + extreme_a CAT-pool mask
  item_params_5d.csv                  per-criterion 5D loadings (exploratory)
  bridge_scenario_fitted_1d.jsonl     1D fitted bank (calibration; all 4197)
  bridge_scenario_fitted_5d.jsonl     5D fitted bank (exploratory)
  bridge_scenario_fitted_1d_catpool.jsonl   1D CAT pool (3961; A3+extreme_a removed)
  model_leaderboard.csv               top-level full-bank leaderboard
  experiments/
    03_structures/                    dimensionality 1-5 + selection.json
    04_efficiency_vs_random/          CAT vs random
    05_oos_recovery/                  headline recovery (locked point)
    06_floor_se_grid/                 Phase-1 SE-sweep (convergence/length)
    06b_operating_point/              7x7 recovery x op-point grid + heatmaps
    07_parameter_uncertainty/         SE_param bootstrap + SE_total leaderboard
    08_leaderboard/                   full-bank leaderboard + bars
    09_pirt_mae/                      standalone p-IRT pass-rate MAE (BiGGen parity)
    10_estimator_comparison/          EAP vs MWLE vs MLE
    11_order_seed/                    order/seed stability
    12_ridge_grid_sensitivity/        ridge/grid robustness
  scripts/                            bridge_scenario_lib + per-experiment drivers
```

## Reproduce

```
# BLAS pinned; run from eduLLM-Evals/. Order matters (1->2->3->mask->4..12).
python bridge_calibration/scripts/build_scenario_bank.py
python bridge_calibration/scripts/scenario_dimensionality.py
python bridge_calibration/scripts/build_catpool.py
python bridge_calibration/scripts/scenario_floor_se_grid.py
python bridge_calibration/scripts/scenario_param_uncertainty.py
python bridge_calibration/scripts/scenario_recovery_grid.py
python bridge_calibration/scripts/scenario_recovery_final.py   # exp 05 + 10
python bridge_calibration/scripts/scenario_efficiency.py       # exp 04
python bridge_calibration/scripts/scenario_leaderboard.py      # exp 08
python bridge_calibration/scripts/scenario_pirt_mae.py         # exp 09 (reuses exp 05)
python bridge_calibration/scripts/scenario_order_seed.py       # exp 11
python bridge_calibration/scripts/scenario_ridge_sensitivity.py # exp 12
```
