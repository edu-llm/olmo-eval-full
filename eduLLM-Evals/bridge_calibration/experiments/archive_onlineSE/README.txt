Archived online-SE-stop of-record outputs (provenance), superseded by the EAP-posterior stop
adoption at floor 12 / SE 0.12. See ../../README.md.

These are the previous of-record artifacts under the ONLINE normal-approx SE stop at SE target
0.15 (floor 12): exp-04 efficiency, exp-05 recovery, exp-06 / 06b operating-point grids, exp-09
p-IRT, exp-10 estimator comparison, exp-11 order/seed, plus the online-SE leaderboard weak-flag
snapshot (model_leaderboard_onlineSE_weakflag.csv).

Stop-INDEPENDENT experiments are NOT archived here (unchanged by the stop rule): exp-03
dimensionality, exp-07 full-bank SE_param floor (parameter_uncertainty metrics.json +
leaderboard_se_components.csv), exp-08 full-bank leaderboard theta, exp-12 ridge sensitivity.

Superseded because (a) the deployed stop moved from the engine's online normal-approx SE to the
honest EAP posterior SD, resolving the online-vs-posterior estimator mismatch, and (b) the SE
target moved 0.15 -> 0.12 to control the SE_total tail (SE_total adds SE_param ~0.05, so a 0.15
ability target left ~14 models with SE_total > 0.15; 0.12 collapses that tail to 2) and to align
Bridge's SE target with the WildBench / BiGGen adoptions.
