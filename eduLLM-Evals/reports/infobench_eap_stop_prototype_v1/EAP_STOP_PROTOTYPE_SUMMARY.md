# InFoBench EAP-posterior stopping prototype

**Decision:** `do_not_adopt`

This offline experiment changed only the CAT stopping statistic. It reused the same 25 family-held-out shrinkage-2PL banks, frozen response matrix, scenario split, online selection state, policy, and seeds. No calibration, tutor calls, or judge calls ran.

## Primary results

| Metric | Online SE stop | EAP posterior-SD stop | EAP - online |
|---|---:|---:|---:|
| Honest EAP target rate | 0.138 | 0.858 | 0.719 |
| Arm median of model-average scenarios | 23.1 | 26.5 | 3.4 |
| Median paired per-model increase | — | — | 2.6 |
| Mean scenarios | 25.79 | 29.92 | 4.13 |
| Recovery r | 0.986 | 0.986 | 0.001 |
| Recovery slope | 1.096 | 1.090 | — |
| Theta MAE | 0.399 | 0.380 | -0.019 |

The family-bootstrap 95% CI for the median paired per-model increase was [1.8, 4.2] scenarios.

## Preregistered acceptance gates

- PASS — `online_reproduction_exact`
- PASS — `honest_target_rate_strictly_higher`
- PASS — `honest_target_improvement_ci_nonnegative`
- PASS — `recovery_r_noninferior`
- PASS — `theta_mae_noninferior`
- PASS — `eap_recovery_slope_in_range`
- FAIL — `median_length_not_increased`
- PASS — `dense_grid_stop_agreement`
- PASS — `dense_grid_final_se_stable`

## Interpretation

Even a passing prototype would not authorize deployment because the underlying CAT policy previously failed Phase-3 validation.
The handoff's historical floor-0/SE-0.25 setting was not used. This run uses the current floor-15/SE-0.20/trace 2PL candidate and the V4-validated 801-node normal-trapezoid grid; 1601 nodes are included as a numerical sensitivity.

## Artifacts

Raw rows and machine-readable metrics are in `runs/calibration/InFoBench_eap_stop_prototype_v1/`; figures and this summary are in `reports/infobench_eap_stop_prototype_v1/`.
