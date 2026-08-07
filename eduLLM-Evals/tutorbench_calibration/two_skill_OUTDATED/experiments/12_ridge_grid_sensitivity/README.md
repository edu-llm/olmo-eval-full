# 12 — ridge / grid sensitivity

**Status: documented / locked (STOP-INDEPENDENT).** Ridge is a calibration-fit hyperparameter,
independent of the CAT stop rule. Unlike the sibling benches, no standalone TutorBench ridge-sweep
figure was produced; the value is **locked at ridge = 1e-2** and documented from the bank
provenance rather than re-swept here.

## Locked value

- **ridge = 1e-2**, GH fit grid = 7 (M2PL confirmatory MML-EM), negative_policy = clamp.
- Adopted **2026-07-31** (Scaffolding-Hygiene audit plateau; previously 1e-3). Recorded in the
  of-record bank provenance: `data/TutorBench/rubrics_qmatrix_calibrated_2skill_115_fitted.jsonl`
  → `irt_params.provenance.ridge = 0.01`, `ridge_note = "ridge=1e-2 adopted 2026-07-31 …"`.
- The pre-adoption 1e-3 fit manifest is retained locally at
  `staging/_before_ridge1e3/cal_2skill_1e3_manifest.json` for comparison.
- This same ridge=1e-2 is used in every EAP fold refit (`eap_oos_grid_study.py --ridge 1e-2`),
  so all of-record recovery numbers already reflect it.

## TODO (optional)

A full {1e-3, 1e-2, 1e-1} × grid {7, 21} sweep in the sibling style (θ-rank stability, extreme_a
count) has not been run for TutorBench. If desired, adapt `scripts/scenario_kfold_estimator_cv.py`
/ `scripts/calibrate_mirt.py` to sweep ridge and confirm θ-rank stability. Not blocking: the value
is locked and consistent with the shared-toolchain default.
