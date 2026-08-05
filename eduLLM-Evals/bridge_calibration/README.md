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

## Held-out recovery (5-fold OOS, models excluded from their own scoring)
Locked operating point: **SE 0.15 / floor 20** (chosen from the SE x floor sweep in
`experiments/06_floor_se_grid/` — the knee: near-max recovery at ~24 criteria, 100% convergence).
- theta recovery: **r = 0.944** [0.911, 0.966], slope = 0.866, theta-MAE = 0.267 (CAT-MWLE vs full-bank EAP).
- p-IRT pass-rate: **r = 0.910**, passMAE = 0.0485.
- Method: per fold, items fit on TRAIN models; each held-out model scored with a CAT (MWLE,
  SE 0.15, floor 20) vs its full-bank EAP reference. Supersedes the earlier crude number.

## Caveats
- Preliminary (N=51, unidimensional). The committed 5-skill numbers come from the
  200-model run. Exclude the `extreme_a` items and A3 (safety tripwire) from CAT use.
