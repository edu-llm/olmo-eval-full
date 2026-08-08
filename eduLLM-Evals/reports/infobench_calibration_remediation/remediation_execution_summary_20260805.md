# InFoBench calibration remediation execution summary

**Run date:** 2026-08-05  
**Status:** Phase 2 passed; Phase 3 failed the complete-policy gate; Phase 4 not authorized  
**Interpretation:** Provisional until the independent two-human InFoBench Qwen audit passes

## Bottom line

The numerical quadrature problem is fixed. The fitted one-dimensional bank is
stable at fit grid 61, and a 401-point normal-trapezoid EAP grid removes the
five-value theta stripes without materially changing under denser grids, wider
bounds, or an independent integration family.

The CAT policy-selection problem is not yet solved. In leakage-free nested
cross-validation, only three of five outer folds produced any CAT configuration
that passed every preregistered inner gate. The other two folds produced no
finalist, and the no-fallback rule was honored. Therefore there is no complete
five-fold CAT policy, Phase 4 correctly refuses to run, and no efficiency or
final-policy claim should be released from this study.

## Phase 2 — numerical lock

Artifacts are under:

`runs/calibration/InFoBench_remediation_v1_quadrature_resolution/`

The study manifest has status `phase2_complete` and locks:

- fit grid: 61;
- EAP method: normal trapezoid;
- EAP grid: 401 points;
- integration bounds: `[-8, 8]`;
- all 52 model theta estimates unique at six decimal places.

| Check | Observed result | Frozen limit | Result |
|---|---:|---:|---|
| Common-cell 41-vs-61 log-loss shift | -0.000642 | absolute shift and full family-bootstrap CI within ±0.005 | Pass |
| Common-cell 61-vs-81 log-loss shift | +0.000545 | absolute shift and full family-bootstrap CI within ±0.005 | Pass |
| 401-vs-1601 maximum theta shift | 3.37e-8 | 0.005 | Pass |
| Wider-bound maximum held-out theta shift | 1.39e-9 | 0.005 | Pass |
| Maximum posterior tail mass | 3.70e-7 | 1.00e-5 | Pass |
| Independent-family maximum full theta shift | 0.002647 | 0.005 | Pass |
| Independent-family maximum held-out theta shift | 0.001056 | 0.005 | Pass |

Ridge candidates 0.001, 0.01, and 0.1 were all fit only after the numerical
lock. Ridge selection remained an inner-CV decision in Phase 3 rather than a
same-cohort Phase-2 choice.

## Phase 3 — nested OOS CAT selection

Artifacts are under:

`runs/calibration/InFoBench_remediation_v1_quadrature_resolution/nested_cat_cv/`

The run evaluated all 48 combinations of minimum-scenario floor, conditional-SE
target, and selector inside each outer fold. Ridge 0.1 was selected from inner
data in every fold. All five fold-local shortlists were frozen before any outer
panel was scored; candidates were never unioned across folds.

| Outer fold | Models | Candidates passing every inner gate | Fold-local finalist |
|---:|---:|---:|---|
| 0 | 11 | 3 | floor 12, SE 0.20, trace |
| 1 | 10 | 5 | floor 8, SE 0.25, D-opt |
| 2 | 10 | 0 | None |
| 3 | 10 | 0 | None |
| 4 | 11 | 1 | floor 15, SE 0.20, trace |

Across the 240 candidate-fold rows, the most frequent failed gates were:

| Gate | Candidate-fold failures |
|---|---:|
| Recovery slope outside 0.90–1.10 | 161 |
| Disjoint pass-rate MAE above 0.07 | 134 |
| Scenario reduction versus random below 50% | 128 |
| Recovery-correlation lower CI below 0.85 | 31 |
| Absolute pass-rate bias above 0.03 | 13 |
| Paired interval did not favor CAT | 8 |

A row may fail more than one gate, so these counts do not sum to 240. Replay
success, MWLE convergence, and nominal conditional precision were not the
dominant problems.

The closest fold-2 configurations missed only recovery slope: their slopes were
approximately 0.810–0.840. In fold 3, the floor-15/SE-0.20/D-opt candidate
missed only the upper slope limit (1.1069 versus 1.10), while the floor-20
candidates missed the required 50% scenario reduction. These are diagnostics,
not permission to relax a threshold after seeing the data.

## Partial outer-fold diagnostics

Only the three folds with finalists were scored. These results cover 32 of 52
models and must not be presented as a complete nested estimate.

| Fold | Finalist | Outer recovery r | Outer recovery slope | Mean scenarios | Reduction vs random | Pass-rate MAE |
|---:|---|---:|---:|---:|---:|---:|
| 0 | floor 12, SE 0.20, trace | 0.951 | 1.033 | 12 | 62.5% | 0.0795 |
| 1 | floor 8, SE 0.25, D-opt | 0.981 | 0.727 | 8 | 64.1% | 0.0881 |
| 4 | floor 15, SE 0.20, trace | 0.984 | 0.839 | 15 | 51.2% | 0.0435 |

The disagreement in selected settings and the weak outer slopes in folds 1 and
4 reinforce that the current cohort does not support a stable frozen policy.

## Phase 4 and later phases

Phase-4 plan validation exits closed with:

`outer fold 2 has no Phase-3 finalist panel`

No parameter-bootstrap, total-SE, or 20-seed order-stability run was launched.
Running it on only the successful folds would break the frozen five-fold design
and create selection-conditioned evidence. In addition, the frozen remediation
configuration never defined a distinct total-SE acceptance tolerance, so even a
complete Phase-4 run could rank candidates but could not honestly freeze a final
policy without a prospective decision.

Final full-cohort fitting, CAT-versus-random release claims, deployment replay,
and final release figures remain blocked by:

1. incomplete Phase-3 coverage;
2. the missing prospectively defined total-SE tolerance; and
3. the incomplete two-human InFoBench Qwen audit.

## Recommended prospective follow-up

Preserve this run as the failed preregistered result. Before generating another
official response-dependent output, freeze a new follow-up design that:

1. completes and adjudicates the InFoBench-specific Qwen audit;
2. defines a distinct total-SE acceptance tolerance;
3. compares the current 2PL bank with a constrained-discrimination/1PL or
   shrinkage model using the same outer folds, because recovery-slope instability
   is the dominant failure;
4. decides whether the 50% reduction gate is a hard product requirement or a
   preference, using a rationale independent of these observed near misses; and
5. expands the calibration cohort if the simpler model still cannot produce
   stable OOS slope and parameter uncertainty.

Threshold-only tuning on the same 52 models would turn the failed holdout result
into a development set and would not constitute honest confirmation.

## Validation performed

- all eight Phase-3 output hashes recorded in the manifest reproduce;
- 240/240 inner candidate rows are present;
- all 15 ridge-evidence rows are present;
- outer output contains exactly one row per model, with 32 scored rows and 20
  explicit `not_run_no_phase3_finalist` rows;
- no duplicate outer fold/candidate/model keys were found;
- the focused Phase-2/3/4 suite passed (65 tests);
- Ruff, compilation, and `git diff --check` passed before the official run.

## Frozen evidence bundle

The complete failed-run evidence, relevant configs/code/tests, the Phase-0 input
bundle, and the still-unfilled judge-audit packet were archived at:

`runs/calibration/InFoBench_remediation_evidence_20260805.tar.gz`

SHA256:

`d1748dd77289be9998a52a272cdd677cd1446f30ae7566452bc74a58b86e017f`

The archive is 44 MB and contains 593 entries. It is a local frozen artifact;
publishing this exact hash to team-controlled immutable storage remains pending.
