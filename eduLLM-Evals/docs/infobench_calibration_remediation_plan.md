# InFoBench calibration and CAT remediation plan

**Status:** Numerically blocked after the completed Phase-3 run; no CAT policy is frozen

**Baseline:** `frq/infobench` at `c321919`

**Baseline report:** `reports/infobench_calibration_20260804/`

**Primary objective:** Replace the provisional five-node, same-cohort CAT result
with a dense-grid, leakage-free, uncertainty-aware, reproducible result.

## Executive summary

The existing response matrix can be reused for most of this work. The plan does
not require regenerating tutor responses or rerunning Qwen unless the
InFoBench-specific judge audit fails or the calibration cohort is expanded.

Work will proceed in this order:

1. Freeze the current result and preregister the new experiment.
2. Validate Qwen on InFoBench in parallel with numerical work.
3. Replace the five-node calibration and EAP reference with a demonstrably stable
   dense one-dimensional grid;
4. Select CAT settings inside nested cross-validation rather than on the same
   models used to evaluate them;
5. Choose among the surviving configurations using total uncertainty and
   question-order stability, not conditional CAT SE alone;
6. Fit the final bank on all calibration models, evaluate CAT versus random using
   held-out evidence, and separately measure a full-bank deployment proxy;
7. Freeze reproducible artifacts, reconcile counts, and regenerate figures with
   confidence intervals.

No final CAT-efficiency or ability-recovery claim will be made until the gates in
this document pass.

## Execution update — 2026-08-05

The official Phase-2 quadrature-resolution study completed successfully under
`runs/calibration/InFoBench_remediation_v1_quadrature_resolution/`. It retained
fit grid 61 and locked 401-point normal-trapezoid EAP integration on `[-8, 8]`.
All common-cell fit-grid, primary-grid, wider-bound, posterior-tail, and
independent SciPy Gauss-Hermite checks passed. All 52 full-bank theta estimates
were distinct at six decimal places, so the former five-node stripes are gone.

The official Phase-3 nested study also completed. Ridge 0.1 was selected from
inner data in all five outer folds. Folds 0, 1, and 4 produced one fold-local
finalist each, but folds 2 and 3 produced no configuration that passed every
absolute inner gate. The no-fallback rule was honored. Consequently, a complete
five-fold policy cannot be formed and Phase 4 correctly refuses to start. The
most common failed gates were recovery slope, disjoint pass-rate MAE, and the
required 50% scenario reduction versus random order.

This is an evidence gate, not a software failure. Thresholds will not be relaxed
after seeing these results. Phase 4, final-bank fitting, final CAT-versus-random
claims, and release figures remain unauthorized. The separate two-human
InFoBench Qwen audit is also still incomplete, so all numerical results remain
provisional even if the CAT gate is later resolved.

## What remains fixed

- The source benchmark remains InFoBench with 500 scenarios and 2,250 criteria.
- The selected latent structure remains the one-dimensional
  `instruction_following` model for this remediation study. Reopening the skill
  structure would confound the fixes below and requires a separate decision.
- The current 52-model binary response matrix remains frozen unless the judge
  audit fails.
- MWLE remains the primary final-theta estimator unless dense-grid held-out
  results show that another estimator clearly performs better.
- The original run remains untouched under
  `runs/calibration/InFoBench_playbook_full/` and in the existing report.

## Reproducibility rules for the new run

- Create a new configuration, tentatively
  `configs/infobench_calibration_remediation_v1.json`.
- Write all generated outputs to a new ignored directory, tentatively
  `runs/calibration/InFoBench_remediation_v1/`; never resume into or overwrite the
  original run.
- Record the response-matrix, bank, scenario, and judge-manifest hashes before any
  fitting.
- Precompute and freeze outer and inner fold assignments before examining new
  metrics.
- Cache every item-parameter fit and reuse it across CAT configurations. Changing
  a stopping rule must not trigger a different calibration fit.
- Record exact commands, seeds, code commit, package versions, convergence status,
  and output hashes.

## Phase 0 — Baseline audit and preregistration

### Work

1. Verify the frozen inputs against the existing hashes:
   - response matrix:
     `087948fcaa884cde6660df1fb4964072ec60fe8aee398e293ed5f74db8f3f27c`
   - judge manifest:
     `d2f19c55b8ebb8f3ebe0c641485e0b694ce861f2740835da760b7b51b40698e7`
2. Reproduce the current summary metrics from stored artifacts without refitting.
3. Preserve an immutable input bundle before more analysis. It must include the
   response matrix, judge manifest, normalized verdicts, and the 52 InFoBench tutor-
   response shards, with a stable storage URI and inner-file hashes.
4. Add an artifact manifest recording storage URIs, archive/file hashes, source
   commits, exact commands/configuration, and the Python environment reference.
5. Either freeze the matrix-ingestion code in Git or explicitly declare the matrix
   itself to be the immutable calibration boundary.
6. Create the new configuration and frozen fold-assignment files.
7. Add an automated leakage audit that proves no outer-test model appears in item
   fitting or CAT-setting selection for that fold.
8. Add an explicit count flow:
   `2,250 source -> 2,105 fitted -> 2,096 exported`, with the current exclusion
   reasons shown as `145 unfitted + 9 nonpositive discrimination`.

### Gate

Do not proceed if the input hashes, model/criterion counts, judge provenance, or
baseline metrics cannot be reproduced exactly.

### Deliverables

- remediation configuration;
- retrievable immutable input bundle and artifact manifest;
- immutable outer/inner fold manifest;
- baseline-reproduction report;
- leakage-audit tests;
- criterion-count reconciliation table.

## Phase 1 — InFoBench-specific Qwen audit (parallel workstream)

The response matrix depends on Qwen judgments, but Qwen was selected using
TutorBench. That validation does not automatically transfer to InFoBench.

### Work

1. Draw a deterministic blinded sample of roughly 300–400 response-criterion cells
   covering:
   - Qwen pass and fail verdicts;
   - all five native InFoBench constraint labels;
   - Easy and Hard subsets;
   - short and long tutor responses;
   - a broad range of tutor models and observed pass rates;
   - Qwen-family and non-Qwen-family tutor models.
2. Add targeted risk cases: all 27 current `no_decision` cells, examples from the
   145 universally failed criteria, and the 9 nonpositive-discrimination criteria.
3. Have two blinded humans grade each sampled cell independently, then adjudicate
   disagreements.
4. Compare the frozen Qwen verdicts with the adjudicated human labels.
5. Report accuracy, macro-F1, fail sensitivity/recall, per-label F1, coverage,
   Cohen's kappa between humans, and a small repeat-consistency check for Qwen.

### Proposed acceptance gate

- macro-F1 at least 0.80;
- fail sensitivity at least 0.90;
- no mapped constraint category below F1 0.70;
- at least 0.90 repeat agreement on the repeated subset;
- no systematic error pattern that would materially change item pass rates.

These are the previously used judge-selection thresholds. The team should approve
them before labels are revealed.

### Decision

- **Pass:** retain the existing matrix and continue.
- **Fail:** revise the judging policy, regrade the full matrix, and restart every
  calibration phase from the new matrix. Numerical reruns performed before the
  audit finishes remain engineering diagnostics only.

### Durable audit files

- `docs/infobench_judge_audit_protocol.md`;
- `scripts/prepare_infobench_judge_audit.py`;
- `scripts/score_infobench_judge_audit.py`;
- `tests/test_infobench_judge_audit.py`;
- `reports/infobench_judge_audit_v1/`.

## Phase 2 — Dense-grid calibration and reference scoring

The current fit grid and EAP scoring grid are both five nodes. Fixing only the
final plot or only the EAP call would leave part of the problem in place.

### Work

1. Keep the initial 1D-versus-multidimensional structure screen at grid 5. A global
   grid 41 would make the five-dimensional arm require `41^5` nodes and is neither
   necessary nor practical.
2. After the one-dimensional structure is selected, treat its **fit grid** and
   **EAP scoring grid** as separate settings. Increasing only the fit grid will not
   remove the quantized reference theta; increasing only the EAP grid will not
   establish that the fitted item parameters are numerically stable.
3. For the selected 1D fit, run grid values `5, 7, 15, 25, 41`. Because this stage
   is one-dimensional, grid 41 is computationally practical.
4. For EAP reference scoring, run `21, 41, 81` nodes while holding the item fit and
   scenario paths fixed. Use grid 81 as the stability comparison, not automatically
   as the production choice.
5. For numerical convergence, hold the current ridge fixed first. Select the
   smallest grid that is stable relative to the next denser grid; do not select a
   quadrature grid merely because it has the best observed test loss.
6. At the selected dense fit grid, rerun the ridge comparison using held-out log loss
   and item-parameter stability. Ridge is a statistical hyperparameter and must be
   selected from training/validation data, unlike the numerical grid.
7. Recompute full-bank reference EAP, MWLE, p-IRT, and item parameters using the
   selected dense settings.
8. Add a diagnostic reporting the number of unique reference-theta values and the
   largest concentration of models at one value, so coarse-grid collapse cannot
   recur silently.

### Proposed numerical-stability gate

For EAP grid 41 versus 81, require:

- median absolute reference-theta shift no more than 0.02 and 95th percentile no
  more than 0.05;
- recovery changes no larger than 0.01 in correlation, 0.02 in slope, and 0.02 in
  theta MAE;
- pass-rate MAE changes no more than 0.005;
- the node-pileup diagnostic no longer shows coarse quantization;
- fixed-bank scenario orders are identical across scoring grids.

For fit grid 25 versus 41, require:

- all full and fold fits converge with finite parameters;
- item-parameter Spearman correlation at least 0.99 for both discrimination and
  difficulty;
- at least 99.5% agreement on which criteria are exportable;
- for both held-out log loss and Brier score, the absolute pooled change is at
  most 0.005 or at most two paired-fold standard errors. A large improvement and
  a large worsening both count as numerical instability requiring investigation.

If either comparison fails, continue to a denser grid or investigate the fitter
rather than freezing a grid. Dense-grid recovery is not required to improve over
the coarse result; it is required to become numerically stable.

### Deliverables

- dense-grid sensitivity table and figure;
- selected numerical-grid manifest;
- dense reference-theta table;
- updated fitted-item parameters for each fold;
- tests ensuring fit and EAP grids are propagated correctly.

## Phase 3 — Nested out-of-sample CAT configuration selection

### Phase 2 numerical follow-up amendment (frozen before follow-up results)

The original Phase-2 run completed every preregistered fit grid and failed the
`25_vs_41` stability gate. The only failing component was discrimination-parameter
rank stability: the minimum fold-level Spearman correlation was `0.9819`, below
the frozen `0.9900` threshold. Convergence, difficulty-parameter stability,
exportability agreement, held-out log loss, and held-out Brier stability passed.

This failed result remains immutable. The append-only follow-up is specified by
`configs/infobench_dense_grid_followup_v1.json` and adds fit grids 61 and 81. It
reuses the completed grid-41 artifacts by verified copy rather than refitting
them, runs both `41_vs_61` and `61_vs_81`, and applies the original thresholds
unchanged. Grid 41 may be locked only if both comparisons pass. If `41_vs_61`
fails but `61_vs_81` passes, grid 61 may be locked. Any other result leaves the
study numerically blocked. EAP-grid stability and ridge sensitivity are scheduled
only after that fit-grid lock.

That follow-up locked fit grid 61: `41_vs_61` still missed the discrimination
rank threshold (`0.9887 < 0.9900`), while `61_vs_81` passed (`0.9940`). Its
separate `41_vs_81` EAP gate then failed because the median and 95th-percentile
absolute reference-theta shifts were `0.1032` and `0.2414`, respectively, versus
limits of `0.02` and `0.05`. All recovery-summary shifts, pass-rate error,
scenario-order identity, and node-pileup checks passed. This result also remains
immutable.

The second append-only amendment is frozen in
`configs/infobench_eap_grid_followup_v1.json`. It reuses the completed grid-61
fit and EAP-81 score by verified copy, adds EAP grids 161 and 321, and changes no
threshold. EAP 81 may be locked only if both `81_vs_161` and `161_vs_321` pass.
If the first fails and the second passes, EAP 161 may be locked. Otherwise Phase
2 remains blocked and no ridge, nested CAT, or final-policy run is authorized.

The 81/161/321 follow-up also failed the original theta-shift limits, although
the shifts decreased monotonically and all 52 model estimates became unique at
321 nodes. Investigation showed that the full-bank posterior is narrower than
the central spacing of practical fixed Gauss-Hermite rules. The next official
study is therefore frozen in `configs/infobench_quadrature_resolution_v1.json`.
It keeps the grid-61 EM fit unchanged and changes only EAP integration to an
explicit normal-trapezoid grid on `[-8,8]`, tested at 401/801/1601 points. It
also adds a same-step wider-bound check, an independent SciPy Gauss-Hermite
cross-check, a maximum per-model theta-shift gate, and recomputes fit-grid
held-out metrics on identical response cells with a family-cluster bootstrap.
The diagnostic evidence that motivated this design is disclosed in the config;
this is not represented as an independently timestamped preregistration.

“CAT settings” means the minimum scenario floor, conditional SE stopping target,
trace versus D-opt selection rule, and related fixed runtime limits. These must be
chosen without looking at the models used for final evaluation.

### Design

Use nested person-level cross-validation:

1. Freeze five grouped, stratified outer folds using the existing seed. Keep base,
   instruction-tuned, and sibling models from the same model family together where
   possible, so the evaluation better represents a genuinely new family. Each outer
   fold holds out approximately 10–11 tutor models for final evaluation.
2. For each outer fold, fit item parameters only on the other approximately
   41–42 models.
3. Inside those outer-training models, use four inner folds to choose ridge and
   CAT settings.
4. Lock one configuration for that outer fold using inner-fold metrics only.
5. Within each held-out model, administer CAT from a frozen 80% scenario pool and
   evaluate predictions on the other 20%. Primary selection metrics are disjoint-
   scenario log loss, Brier score, and pass-rate error; full-bank theta recovery is
   secondary.
6. Run that locked configuration once on the untouched outer-test models.
7. Pool the five outer-test predictions for the headline OOS metrics and bootstrap
   confidence intervals over tutor models.

The held-out model's administration-pool responses may be used after selection to
simulate its CAT path. A full-response reference ability may be computed afterward
as a secondary diagnostic, but it cannot affect the selected path or configuration.
Outer-test outcomes may not affect item fitting or configuration choice, and
evaluation-pool outcomes may not affect the CAT path.

### Candidate CAT grid

- minimum scenarios: `0, 5, 8, 12, 15, 20`;
- conditional SE target: `0.20, 0.25, 0.30, 0.35`;
- selector: `trace, dopt`;
- fixed unless separately preregistered: MWLE, `top_n=5`, maximum 50 adaptive
  scenarios, and the current missing-response policy.

Run the floor × SE × selector combinations jointly. The current sequential sweep
can miss interactions between the floor and SE target.

### Inner-selection rule

Use absolute preregistered gates and select no fallback when every candidate fails.
Proposed starting gates are:

- at least 99% successful replays;
- at least 95% MWLE convergence;
- at least 95% nominal precision reached, with a lower 95% confidence bound of at
  least 90%;
- held-out recovery-correlation lower confidence bound at least 0.85;
- recovery slope between 0.90 and 1.10;
- held-out pass-rate MAE no more than 0.07 and absolute bias no more than 0.03;
- at least 50% fewer scenarios than paired random order, with the paired confidence
  interval still favoring CAT;
- at least 90% valid parameter-bootstrap draws;
- preferably the same configuration selected in at least four of five outer folds.

The team must ratify these numbers before outer-fold results are examined. Among
configurations that pass every applicable gate, shortlist by lowest 90th-percentile
scenario count, then mean scenario count. Use lower total uncertainty and the simpler
trace selector as tie-breakers. Send those finalists to Phase 4 rather than declaring
a winner from conditional SE alone.

### Required tests

- outer-test model IDs never enter a fit or inner metric;
- changing outer-test outcomes cannot change that fold's selected configuration;
- evaluation-scenario responses cannot change the CAT administration path;
- related model families do not cross grouped-fold boundaries;
- fold assignments and results are deterministic under the frozen seed;
- every candidate in a fold uses the identical cached item fit;
- missing responses remain missing rather than becoming failures.

### Deliverables

- per-fold fit and selection manifests;
- inner-fold candidate table;
- outer-fold per-model results;
- pooled OOS recovery, p-IRT, length, convergence, and bootstrap CIs;
- explicit optimism gap versus the original same-cohort sweep.

The nested stage should write at least:

- `nested_cat_cv/fold_assignments.json`;
- `nested_cat_cv/inner_candidate_results.csv`;
- `nested_cat_cv/outer_fold_choices.json`;
- `nested_cat_cv/outer_oof_per_model.csv`;
- `nested_cat_cv/disjoint_prediction_metrics.csv`;
- `nested_cat_cv/outer_metrics.json`;
- `nested_cat_cv/config_selection_frequency.csv`;
- `nested_cat_cv/outer_total_se.csv`;
- `nested_cat_cv/outer_order_stability.csv`;
- `nested_cat_cv/manifest.json`.

## Phase 4 — Honest uncertainty, floor selection, and order stability

The engine's stopping SE is conditional on treating difficulty and discrimination
as known. It is not the same as total uncertainty.

### Work

1. For each shortlisted CAT configuration, bootstrap only that fold's outer-training
   models and refit item parameters.
2. Reuse each bootstrap fit across all floor/SE/selector candidates; do not refit
   separately for identical calibration data.
3. Rerun the held-out model's CAT path under each bootstrap bank when feasible, so
   item-parameter changes are allowed to alter item selection rather than keeping an
   artificially fixed path.
4. On held-out models, calculate:
   - `SE_ability`: conditional ability uncertainty;
   - `SE_param`: uncertainty from estimated item parameters;
   - total variance using the law of total variance: mean bootstrap conditional
     variance plus variance of bootstrap theta estimates;
   - `SE_total`: the square root of that total variance;
   - order/path SD across at least 20 deterministic seeds, reported separately.
5. Compare average, median, 90th percentile, maximum, and the proportion below the
   preregistered total-SE tolerance alongside
   test length and recovery.
6. Use total SE as an offline configuration-selection criterion. The online engine
   may still stop on conditional SE, but its floor/target must be chosen so that
   held-out total SE is acceptable.

The current run demonstrates the gap: at nominal online SE 0.25, median MWLE total
SE was 0.537, the 90th percentile was 0.938, and no model achieved total SE at or
below 0.25. Future reports must call 0.25 the nominal online stopping threshold, not
achieved total precision.

### Final configuration rule

1. Start with candidates that passed every absolute OOS gate.
2. Reject candidates that fail the preregistered total-SE tolerance or hide poor
   tail behavior behind a good median.
3. Among the remaining candidates, choose the lowest 90th-percentile scenario count,
   then the lowest mean scenario count.
4. Use lower total uncertainty and trace selection as tie-breakers.
5. Require median order/path SD no greater than the candidate's nominal conditional
   SE target. If none passes, do not freeze a CAT configuration.

The complete tradeoff table remains the primary evidence; the rule must not hide
large tail uncertainty or nonconvergence.

### N=52 contingency

If `SE_param` still dominates after dense scoring and a reasonable scenario floor:

1. Compare the current 2PL with a simpler constrained-discrimination/1PL model or
   a clearly specified shrinkage model using outer-fold log loss, recovery, and
   item-parameter stability.
2. If simplification does not produce acceptable OOS uncertainty, expand the
   calibration cohort. That expansion requires new tutor responses, Qwen judgments,
   and a full restart from the response matrix.

Threshold tuning alone cannot repair underidentified item parameters.

### Deliverables

- OOS `SE_ability`, `SE_param`, and `SE_total` tables for finalists;
- order/path stability table from at least 20 seeds;
- floor/SE/selector Pareto frontier;
- final configuration decision or a documented “insufficient calibration data”
  result.

## Phase 5 — Final fit, CAT-versus-random evaluation, and deployment replay

Only begin this phase after the Qwen audit and Phases 2–4 pass.

### Work

1. Run the same preregistered inner-CV selection procedure once across all 52 models
   to select the production configuration. Do not change gates based on outer-fold
   results.
2. Lock the dense grid, ridge, estimator, floor, SE target, selector, and all seeds.
3. Refit the final item bank using all 52 calibration models.
4. Evaluate the selection procedure using the pooled outer-fold results. These are the
   headline generalization metrics.
5. Compare adaptive CAT with random order on the same outer-test models, frozen
   item parameters, stopping rule, and paired seeds. Use multiple random seeds so
   one favorable ordering cannot determine the result.
6. Replay the locked policy against the final all-52-model bank to estimate
   operational/deployment test length. Label this replay as same-cohort and do not
   use it as the headline recovery estimate.

### Required reporting nuance

K-fold item banks are trained on approximately 41–42 models; the deployed bank is
trained on all 52. Their discrimination estimates and therefore stopping lengths
can differ. Report both:

- **OOS length/recovery:** honest expected generalization from outer folds;
- **full-bank replay length:** operational diagnostic for the shipped bank.

Do not assume one is numerically interchangeable with the other.

Write a `cat_length_evidence.csv` with three explicitly labeled rows:

- `same_cohort_final_bank`;
- `cross_fitted_proxy`;
- `external_frozen_bank` (pending until a new model cohort is available).

Nested CV is the strongest internal estimate available from the existing cohort,
but these 52 models have already informed exploratory decisions. A genuinely
untouched model cohort remains the cleanest final external confirmation.

### Gate

Freeze the policy only if it passes the preregistered OOS recovery, convergence,
total-uncertainty, and order-stability gates. A longer but stable CAT is preferable
to a three-to-five-scenario CAT whose apparent precision is not real.

## Phase 6 — Figures, artifacts, and release hygiene

### Figures

Regenerate all summary and detailed figures from the final outputs. In particular:

- add model-bootstrap confidence intervals to the minimum-scenario and SE sweeps;
  with four sweep points, prefer honest error bars over a smooth ribbon;
- add model-bootstrap CIs for correlation, slope, and MAE to the recovery figure,
  clearly labeled as excluding item-parameter uncertainty;
- remove the five-node stripes rather than merely hiding them;
- build empirical leaderboard intervals from a joint parameter-bootstrap × CAT-
  order replay when feasible. Otherwise retain approximate total-SE bars, label
  them as excluding path/order uncertainty, and overlay the observed order range;
- label every figure as OOS, same-cohort diagnostic, or deployment replay.

Figures 08–11 must not imply more precision than the underlying CSVs support.

### Frozen artifacts

Create a versioned release directory containing:

- fitted-only rubric bank and filtered scenarios;
- final response matrix or a privacy-safe/anonymized equivalent;
- Q-matrix/skill provenance;
- judge manifest and Qwen-audit report;
- configuration, fold assignments, seeds, commands, hashes, and package versions;
- aggregate and per-fold metrics;
- figures and plain-language summary.

If the response matrix is safe and small enough, track it with the calibrated bank.
Otherwise use Git LFS, DVC, or immutable shared object storage with a stable retrieval
command and checksum. A hash with no retrievable artifact is not sufficient.

### Count reconciliation

Every report must distinguish:

- **2,250 source criteria**;
- **2,105 criteria with fitted parameter rows**;
- **2,096 exported CAT criteria**;
- **145 unfitted exclusions** and **9 nonpositive-discrimination exclusions**.

### Final tests

- focused unit/integration suite passes;
- dense-grid and nested-CV leakage tests pass;
- all manifests and JSON/CSV schemas validate;
- a clean environment can retrieve inputs and reproduce the selected outputs;
- no absolute workstation paths, credentials, raw model secrets, or synthetic IRT
  parameters appear in the frozen bank;
- regenerated summary numbers match source tables exactly.

## Implementation map

The exact split may change during implementation, but the expected code surface is:

- extend `configs/infobench_calibration_study.json` through a new remediation
  config rather than changing the historical configuration;
- extend `scripts/run_infobench_calibration_study.py` or add a separate remediation
  orchestrator so baseline behavior remains reproducible;
- reuse and extend `scripts/calibrate_mirt.py` and `scripts/kfold_cv_mirt.py` for
  dense 1D fits and fit-grid propagation;
- reuse `scripts/scenario_kfold_estimator_cv.py` for fixed-config OOS scoring;
- add a nested-CV CAT configuration-selection driver with explicit leakage checks;
- tentatively name that isolated driver `scripts/nested_scenario_cat_cv.py` rather
  than changing the historical same-cohort selection semantics in place;
- extend `scripts/scenario_param_uncertainty.py` to operate on fold-specific cached
  fits and finalist configurations;
- extend `scripts/scenario_order_experiment.py` for the final seed panel;
- update `scripts/aggregate_calibration_study.py` and
  `scripts/plot_infobench_detailed_figures.py` only after final metrics exist;
- add an InFoBench judge-audit preparation/comparison path using the existing
  deterministic judge-comparison approach;
- add tests for every new selection, propagation, leakage, and freeze invariant.

## Step-by-step execution checklist

- [x] Step 0: approve this plan, candidate grids, folds, and decision gates.
- [x] Step 1: preserve the immutable input bundle, then implement Phase 0
  provenance, fold manifests, count flow, and tests.
  The content-addressed archive is frozen and verified locally; publishing that
  exact hash to team-controlled immutable storage remains a release-hygiene action.
- [ ] Step 2: prepare the blinded Qwen audit packet and begin human grading; in
  parallel, implement dense-grid support. The packet is prepared; independent
  human grading and adjudication remain outstanding.
- [x] Step 3: run only the dense-grid study. The append-only numerical follow-ups
  ultimately locked fit grid 61 and normal-trapezoid EAP grid 401.
- [ ] Step 4: review the dense-grid result and complete the Qwen audit. Lock the
  numerical grid only if it is stable, and proceed only if the judge audit passes.
  The numerical review passed and the grid is locked; the human Qwen audit remains
  incomplete, so release claims remain blocked.
- [x] Step 5: implement nested OOS CAT configuration selection.
- [x] Step 6: run the nested candidate sweep and produce the finalist shortlist.
  The result is incomplete by design: three folds have one finalist and two folds
  have none.
- [ ] Step 7: run fold-specific parameter uncertainty and 20-seed order stability
  for finalists. Blocked because Phase 3 did not produce finalists in all five
  folds; the Phase-4 preflight refuses this handoff.
- [ ] Step 8: select a final policy or trigger the N=52 contingency. No policy is
  frozen; the team must decide on a prospective follow-up design without tuning
  thresholds to this failed result.
- [ ] Step 9: fit the final full-cohort bank and run paired CAT-versus-random plus
  deployment replay.
- [ ] Step 10: regenerate figures, freeze artifacts, run reproducibility checks,
  and publish a final report.

At the end of each numbered step, stop and review its outputs before starting the
next experiment. This prevents later stages from inheriting an unnoticed error.
