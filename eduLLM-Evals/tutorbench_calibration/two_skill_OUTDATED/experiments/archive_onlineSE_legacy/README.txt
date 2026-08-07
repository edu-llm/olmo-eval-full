ARCHIVE — legacy ONLINE-SE stop, 2-skill/115. SUPERSEDED.
=========================================================

Everything in this folder was generated under the OLD engine online normal-approx SE stop
(the production engine's max_se rule), BEFORE the of-record EAP-posterior honest per-skill
marginal-SD stop was adopted. It is retained for provenance and cross-checking ONLY.

For all STOP-DEPENDENT results (recovery, leaderboard, efficiency, p-IRT MAE, achieved SE,
operating point) the canonical numbers are the EAP-posterior-stop package at floor=20 /
SE_ability=0.27 in ../05_oos_recovery, ../06_floor_se_grid, ../06b_operating_point,
../08_leaderboard. DO NOT quote the numbers in this archive as current.

Contents (from regenerated_figures/scenario_level_115_min12/):
  leaderboard_2skill/   legacy CAT leaderboard @ online-SE (cat_per_model.csv, metrics.json, figs)
  kfold_2skill/         legacy OOS k-fold recovery @ online-SE (metrics.json, oos_per_model.csv, figs)
  se_sweep_2skill/      legacy SE-target sweep {0.20,0.25,0.30,0.35} recovery @ online-SE
  frq_mae/              legacy p-IRT / pass-rate MAE + adaptive-vs-random efficiency @ online-SE
                        (this is the legacy source for the TODO slots 04 and 09)
  figures/              legacy composite/se-tradeoff/total-SE figures @ online-SE

Original (non-destructive copy): eduLLM-Evals/regenerated_figures/scenario_level_115_min12/
Once the EAP package is confirmed, the original regenerated_figures/scenario_level_115_min12/
tree may be pruned at commit.
