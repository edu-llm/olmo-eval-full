# 10 — estimator comparison (online vs batch-EAP vs MWLE)

**Status: carried over (STOP-INDEPENDENT-ish); MWLE locked.** The estimator ranking is a property
of the estimator + calibrated bank, not the CAT stop rule. **MWLE** is the locked deployment
instrument and is what every EAP experiment here (05/06/06b/08) uses at stop.

## OOS k-fold recovery (2-skill, N=115)

| estimator | corr r | corr slope | scaff r | scaff slope |
|---|---|---|---|---|
| online (Laplace) | 0.921 | 0.644 | 0.903 | 0.875 |
| batch-EAP | 0.939 | 0.755 | 0.913 | 0.918 |
| **MWLE** | **0.945** | **0.981** | **0.902** | **1.032** |

MWLE's slope is closest to 1.0 on both skills, and its Warm/Firth penalty stays finite on
all-pass/all-fail testlets where plain MLE diverges. **Locked: MWLE** (consistent with
Bridge/BiGGen/WildBench).

## Files

- `estimator_comparison.json` — distilled online/batch/MWLE recovery + lock rationale.
- `kfold_2skill_estimators_metrics.json` — full source metrics.

## Provenance

`scripts/scenario_kfold_estimator_cv.py`. Original:
`regenerated_figures/scenario_level_115_min12/kfold/2_skills/metrics.json`.
