# BiGGen calibration (branch `frq/biggen`)

Primary model: **UNIDIMENSIONAL** ("general" skill), MWLE estimator, trace selection, scenario-level CAT.
Matrix: `staging/biggen_response_matrix.csv` (52 models x 2,678 atoms). Hygiene bank: `staging/biggen_rubrics_hygiene.jsonl`
(2,389 modeled + 289 `exclude_from_fit` = 160 zero-variance + 90 |b|>6 + 40 safety/moral_belief).
Calibrated bank candidate: `bank/biggen_unidim_calibrated.jsonl`.

**LOCKED config:** `SE target=0.12`, `min_scenarios=8`, `ridge=0.01`, MWLE, trace, k=5, seed 20260729 (unidimensional).
- **OOS-of-record (05, re-run at locked config):** r=0.970, slope=0.812, mean 9.4 scenarios / 37.4 criteria, 100% convergence (n=52).
- Experiments **01/02/03/06 are the decision-evidence sweeps** and are left AS-IS at their sweep configs (not re-run).
- The calibrated bank (`bank/biggen_unidim_calibrated.jsonl`) is **config-independent** — item params don't depend on the stopping rule — so it is unchanged.
- History: earlier passes floated floor-12/SE-0.30 then floor-10; the fine-SE knee (06c) motivated the final SE=0.12/floor-8 lock (r 0.970 at ~9 scenarios, 100% convergence, lower-variance than SE 0.11: max 17 vs 24).

**Mean-scenario reconciliation (audit, no bug).** Two mean test-lengths coexist at floor 8 / SE 0.12 for legitimate reasons — NOT a floor or stopping bug:
- **OOS k-fold convention (05, 06, 04, 08-10 recovery): ~9.4 scen.** Each fold refits item params on ~42 train models; small-N MML **upward-biases discriminations** (fold mean|a| 1.44-1.48 vs full-data 1.35), so the engine hits SE 0.12 in fewer scenarios.
- **Deployed frozen bank (07, 07b): ~13.6 scen.** A genuinely new model is scored with the shipped 52-model params (`biggen_unidim_modeled.jsonl`, mean|a| 1.35) -> ~13.6 scen to reach SE 0.12. **This is the true in-deployment length.**
- Verified: `min_evals_per_skill` 15 vs 1 gives identical 13.56 (ruled out); floor 8 enforced identically (min administered = 8) in both paths (ruled out); the only differing input is the item params. So the floor is applied correctly and **the sweeps + the SE=0.12/floor-8 lock stand — nothing needs re-running.** The 05/06 "9.4" is a slightly optimistic CV artifact; the deployed length is ~13.6.
- Total SE is ~0.19 at **both** operating points (decomposition differs): OOS-refit 9.4-scen = SE_ability 0.153 / SE_param 0.116 / SE_total 0.193 (ability-SE looks smaller only because the fold `a` are inflated; param-SE correspondingly larger). See `07_parameter_uncertainty/{se_by_operating_point.csv, figures/se_by_operating_point.png}`.

| # | experiment | headline | figure |
|---|---|---|---|
| 01 | min_scenarios {0,4,6,8,10} (OOS) | OOS r rises 0.907->0.977 (slope 0.67->0.80) | experiments/01_min_scenarios/figures/recovery_r_slope_vs_min_scenarios.png |
| 02 | SE target {0.20..0.35} | convergence 100%; mean scen 10.0; in-sample r~0.973 (SE non-binding at floor 10) | experiments/02_se_target/figures/convergence_and_length_vs_se_target.png |
| 03 | ridge {1e-3..2e-2} (OOS) | OOS r peaks 0.977 @ ridge 0.01; a-stability 0.59->0.71; recommend ridge 0.01 | experiments/03_ridge/figures/recovery_r_and_a_stability_vs_ridge.png |
| 04 | efficiency vs random, **OOS** @ locked (random uncapped 693) | adaptive **9.4 scen / OOS r 0.970** vs random **112 scen / r 0.994** (random's higher r is just ~12x more items, ~full bank; 2 models non-converge). Adaptive ~12x more efficient. (Stale in-sample per-arm scatter subdirs `adaptive/` `random/` removed; aggregate bar is the deliverable.) | experiments/04_efficiency_vs_random/figures/efficiency_adaptive_vs_random.png |
| 05 | **OOS of record @ LOCKED** (SE 0.12 / floor 8) | OOS **r=0.970, slope=0.812**, mean 9.4 scen / 37.4 crit, 100% conv (n=52). 3-panel estimator comparison (online 0.937/0.626, batch 0.962/0.771, **MWLE 0.970/0.812**) regenerated at the locked config — MWLE panel matches the single scatter. | experiments/05_oos_recovery/figures/oos_recovery_scatter_general.png ; oos_recovery_general.png (3-panel) |
| 06 | floor x SE grid (5x4=20, OOS) | SE non-binding at SE>=0.15 (only SE=0.15 lifts length ~0.5 scen at floor 6); OOS r 0.950(f6)->0.984(f15), plateau ~f12 | experiments/06_floor_se_grid/figures/oos_r_vs_floor_by_se.png |
| 06b | tight-SE extension {0.08,0.10} x floor {0,6,8,10,12,15} (OOS) | **convergence cliff at ~0.08**: SE=0.10 -> 100% converge (mean ~13 scen); SE=0.08 -> 2 models (3.8%) hit cap=50, both lowest-ability (theta_ref -4.5). Efficiency frontier: SE=0.10 var-length reaches r=0.985 @ ~13 mean scen vs fixed floor 15's 0.984 @ 15 (marginal gain); SE=0.08 balloons to 24 scen for +0.004 r. | experiments/06_floor_se_grid/figures/efficiency_frontier_r_vs_mean_scenarios.png, convergence_vs_se_target.png |

06b outputs: `results_tight_se.csv` (12 cells w/ scen min/med/max + non-converged counts + per-arm mean theta).
Verdict: tight SE does NOT beat the fixed floor enough to switch; keep floor 12 / SE 0.30 (predictable length, 100% convergence). Avoid SE <= 0.08 (non-convergence for low-ability models + length blowup).

06c — fine SE band {0.11,0.12,0.13,0.14} x floor {0,6,8} (`results_fine_se.csv`; 100% convergence everywhere).
Smooth interpolation (floor 0): SE 0.14 r0.950 @mean6.2/max12 -> 0.13 r0.955 @7.3/16 -> 0.12 r0.963 @8.5/17 -> 0.11 r0.975 @10.4/max24 -> (0.10 r0.985 @13.3/**max46**).
**Length/variance knee at SE ~0.11**: recovers r~0.975 (= fixed floor-10) at ~10 mean scen (max 24); tightening to 0.10 nearly DOUBLES worst-case length (max 46) for +0.01 r. SE 0.12 is the lower-variance pick (max 17, r 0.963).
Caveat: at N=52 the r spread across SE 0.11-0.15 (~0.95-0.975) is within the ~+/-0.02-0.03 bootstrap-CI noise, so 0.11-0.15 is a length/variance-preference call, not a clear optimum. Recommendation unchanged (floor 12 / SE 0.30).

### Locked-config deliverables (07-10, all at SE=0.12 / floor 8 / MWLE)

| # | experiment | headline | figure |
|---|---|---|---|
| 07 | parameter uncertainty (obs-info bootstrap, n_boot=150) | mean SE_ability 0.173 -> **SE_total 0.197** (+SE_param 0.094); inflation median x1.14, max x1.26 — calibration error is MODEST at N=52 | experiments/07_parameter_uncertainty/figures/mean_total_se.png |
| 07b | **SE-vs-floor sensitivity** (exploratory; locked floor stays 8) | more scenarios -> lower SE: floor 8/10/12 = SE_total **0.197 / 0.190 / 0.186** (SE_ability 0.173/0.166/0.162, SE_param ~0.092-0.094), mean scen 13.6/14.2/15.1. Gains are small (SE 0.12 already pushes past the floor to ~14 scen) | experiments/07_parameter_uncertainty/figures/se_vs_floor.png |
| 08 | general-ability leaderboard (52 models, ranked by locked-config theta) | instruct 2-6B on top, tiny base at bottom; **ALL 51 adjacent pairs' +/-1.96 SE_total bars overlap** -> only coarse ability bands are distinguishable at N=52 | experiments/08_leaderboard/figures/leaderboard_general.png |
| 09 | p-IRT predicted-vs-actual pass rate, **OOS** (k=5; TRAIN-fit params + held-out MWLE theta) | pass-rate MAE **0.047** [0.038,0.057] (LOO 0.045), theta-MAE **0.341** [0.265,0.433], r **0.973** [0.961,0.982], slope **1.127** [1.056,1.202] (bootstrap CIs, B=2000). _(in-sample, optimistic: MAE 0.032, theta-MAE 0.259, r=0.986, slope=0.999)_ | experiments/09_pirt_mae/figures/pirt_pred_vs_actual.png |
| 10 | order/seed dependence (8 seeds) | across-seed theta SD mean **0.179** (~1.5x the SE target, 0.74x SE_total), max 1.50 -> short-test run-to-run variability is non-trivial; SE_total captures most of it | experiments/10_order_seed/figures/seed_spread_general.png |

**Bootstrap 95% CI bands (B=2000, seed 0, resample 52 models)** overlay the recovery figures: the 05 three-panel `oos_recovery_general.png` (OLS fit + band + r/slope CIs per panel, matching the single scatter); the sweep r-lines `01_min_scenarios/recovery_r_slope_vs_min_scenarios.png`, `03_ridge/recovery_r_and_a_stability_vs_ridge.png`, `02_se_target/recovery_r_vs_se_target.png` (bands overlap across configs ~+/-0.02-0.03, reinforcing "preference call, not a clear optimum"); and `09_pirt_mae/pirt_pred_vs_actual.png` (bootstrap CI band around the OLS fit line; per-model pairs persisted to `09_pirt_mae/oos_per_model_pirt.csv`). Skipped: `06/oos_r_vs_floor_by_se.png` r-bands (per-cell OOS θ not persisted; would require re-running the full grid).

Each `experiments/NN_*/results.csv` has the full per-config rows; `runs/` holds the raw per-config engine/kfold outputs.
Estimator = MWLE (won here: OOS slope ~0.80 vs batch/online shrink lower). N=52 is below the MIRT identifiability floor — treat as provisional.
