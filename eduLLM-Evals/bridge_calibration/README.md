# Bridge calibration study summary (preliminary)

**Model:** unidimensional 2PL (single tutoring-ability axis), fit with the pure-numpy
Bock-Aitkin EM. Preliminary: 51 models (below the ~150 needed for a stable
5-skill MIRT). Bridge `source_id` grouping applied (duplicate-source scenarios collapsed).

## Item parameters
- Criteria calibrated: **2684** (dropped 400 all-fail + 22 all-pass constants).
- Discrimination `a`: median **1.12**, mean 1.32, range [-1.64, 9.52].
- 2415 of 2684 criteria discriminate usefully (a >= 0.3); 44 flagged `extreme_a`.
- Full per-criterion table: `item_params.csv`. Calibrated bank: `rubrics_calibrated.jsonl`.

## Model leaderboard (ability theta)
- theta range [-3.03, 2.41]. Full table: `model_leaderboard.csv`.
- Top: meta-llama/Llama-3.2-3B-Instruct (2.41), tiiuae/Falcon3-3B-Instruct (2.01), nvidia/AceInstruct-1.5B (1.61), pankajmathur/orca_mini_v9_5_3B-Instruct (1.61), LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct (1.57)
- Bottom: BSC-LT/salamandra-2b (-1.61), Qwen/Qwen2-0.5B (-1.61), ai-forever/mGPT (-2.01), allenai/OLMo-1B-hf (-2.44), Qwen/Qwen1.5-1.8B (-3.03)

## Held-out recovery (5-fold, models excluded from their own scoring)
- **r = 0.226**, slope = 0.201, MAE = 1.011 (theta recovered on unseen models).
- This is the validity check: how well the calibrated items recover a model's ability
  when that model was NOT used to fit the items.

## Caveats
- Preliminary (N=51, unidimensional). The committed 5-skill numbers come from the
  200-model run. Exclude the `extreme_a` items and A3 (safety tripwire) from CAT use.

---

## Study contents (reproducible via `scripts/run_bridge_study.sh`)

Locked CAT operating point: **SE target 0.15 / floor 20** (experiment 06).

| # | experiment | question | key output |
|---|---|---|---|
| 03 | `experiments/03_structures` | 1D vs 2/3/4/5-skill dimensionality | `structure_comparison.csv`, `selection.json` |
| 04 | `experiments/04_efficiency_vs_random` | does adaptive selection save criteria? | `results.csv`, `summary.json` |
| 05 | `experiments/05_oos_recovery` | held-out theta + p-IRT recovery @ locked point | `recovery_metrics.json` (headline r) |
| 06 | `experiments/06_floor_se_grid` | SE x floor efficiency frontier | `sweep_results.csv`, `best.json` |
| 07 | `experiments/07_parameter_uncertainty` | ability vs calibration SE (total-SE bars) | `leaderboard_se_components.csv` |
| 08 | `experiments/08_leaderboard` | leaderboard with honest total-SE bars | `leaderboard_general.png` |
| 10 | `experiments/10_estimator_comparison` | EAP vs MWLE vs MLE | `estimator_comparison.json` |
| 11 | `experiments/11_order_seed` | is theta robust to administration order? | `order_seed_stability.json` |
| 12 | `experiments/12_ridge_grid_sensitivity` | fit robustness to ridge/grid | `ridge_grid_sensitivity.csv` |

**Headline recovery** is experiment 05 (held-out MWLE-CAT theta at the locked point);
the top-level `recovery.json` is a simpler EAP k-fold sanity check. **Leaderboard SE**
in `model_leaderboard.csv` is the *total* SE (ability (+) calibration) from experiment 07;
at N=51 the calibration term dominates. **Dimensionality (03) is exploratory**: a 5-skill
MIRT does not identify at this N (latent correlations collapse toward 1; the affective axis
has no informative anchor) -- the committed multi-skill numbers require the 200-model run.
