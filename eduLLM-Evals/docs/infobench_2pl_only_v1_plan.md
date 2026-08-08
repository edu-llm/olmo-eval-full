# InFoBench 2PL-only follow-up plan

**Frozen before any 2PL-only outer CAT outcome is opened:** 2026-08-06

**Status:** Secondary same-cohort internal-development follow-up

## Purpose

The completed V3 study selected 1PL in all 25 nested-CV panels under its
one-standard-error simplicity rule.  The team requires the calibration family
to remain 2PL for consistency with its other benchmark pipelines.  This
append-only follow-up therefore repeats calibration and CAT with only the two
2PL specifications that already passed the independent V4 dense-grid numerical
lock.

This run does not overwrite or relabel V3.  Because the V3 outcome is already
known, this follow-up is not independent confirmation.

## Frozen 2PL candidates

1. `log_shrinkage_2pl_lambda16` (stronger regularization; simpler rank 0)
2. `log_shrinkage_2pl_lambda4` (moderate regularization; simpler rank 1)

Both estimate item difficulty and discrimination, constrain discrimination to
be positive, and use the V4-locked 401-node normal-trapezoid fit on `[-8, 8]`
with 801-node EAP scoring.  The unrestricted free-2PL ridge variants are not
eligible because they do not have a passing dense numerical lock for this
52-tutor InFoBench cohort.

## Unchanged design

- frozen 52 tutors in 22 model families and frozen Qwen response matrix;
- the same deterministic five repetitions, five outer folds, and four inner
  folds, with related model families grouped together;
- inner-only common-cell held-out log loss and the one-standard-error rule;
- all 25 calibration selections locked before any outer CAT outcome is opened;
- one-dimensional `instruction_following` ability and MWLE estimation;
- primary CAT policy: floor 15, conditional SE 0.20, trace selector;
- diagnostic-only floor-12/trace and floor-15/D-opt sensitivities;
- paired random baseline and every existing absolute acceptance gate;
- no fallback, no sensitivity promotion, and no threshold changes.

## Execution gates

1. Validate the unchanged V4 lock and exact parent input/split/policy hashes.
2. Complete all `25 panels x 4 inner folds x 2 specs = 200` inner fits.  A
   missing or invalid fit blocks the run before CAT.
3. Require a unique exact 2PL specification to be selected in at least 80% of
   panels, using inner evidence only.
4. Evaluate the frozen CAT policies versus paired random order on all 25 outer
   panels and all 52 tutors per repetition.
5. Run total-uncertainty/order Phase 4 only if every Phase-3 primary-policy gate
   passes.  Run the final all-52 fit/export/replay only if Phase 4 passes.
6. Render the terminal result reached, including failed-validation results.

## Versioned outputs

- config: `configs/infobench_calibration_cat_2pl_only_v1.json`
- Phase 3: `runs/calibration/InFoBench_2pl_only_v1/phase3/`
- Phase 4: `runs/calibration/InFoBench_2pl_only_v1/phase4/`
- final fit: `runs/calibration/InFoBench_2pl_only_v1/final_fit/`
- report: `reports/infobench_calibration_cat_2pl_only_v1/`

All conclusions remain conditional on the frozen Qwen labels, which were not
independently human-validated on InFoBench, and on this same 52-tutor cohort.
