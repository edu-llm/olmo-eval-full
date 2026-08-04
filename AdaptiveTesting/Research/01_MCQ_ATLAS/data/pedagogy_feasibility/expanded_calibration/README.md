# Pedagogy MCQ CAT: expanded calibration (78 models)

This folder re-runs the pedagogy MCQ IRT/CAT feasibility experiment after 26 more models
were scored on the pedagogy benchmark. The calibration pool goes from 52 models to 78. The
question is whether a larger, wider calibration set changes the adaptive-test recovery
numbers (Pearson r against full-bank accuracy, mean items to the SE stopping rule, MAE, and
the linear recalibration fit). It does for the 2PL bank. It does not for the 1PL/Rasch bank.

Nothing in the original `pedagogy_feasibility/` folder was modified. All outputs here are new.

## Final model set

- Final N used: 78 pedagogy-scored models.
- Newly added: 26 (scored 2026-08-03; see the S3 manifest dates).
- Original baseline: 52 (scored 2026-08-01).
- Excluded / unscorable: 0 (see "Scorability" below).

The 26 new models are pedagogy-only. Their S3 manifests exist for `pedagogy` but not for
`piqa` or `socialiqa`, so PIQA and SocialIQa are unchanged and were not re-run.

## Data source

Per-item pedagogy responses were pulled read-only from
`s3://edullm-adaptive-inference-056956104102/full200/results/Outputs/mcq/pedagogy/` via the
sb_aws broker (account `sbsandbox`, id `056956104102`). The 78 completion markers under
`.../Outputs/_manifests/pedagogy__<org>__<model>.done` were used to enumerate the models. The
downloaded CSVs live in `_mcq_data/pedagogy/` and carry the same schema the loader expects
(`question_id, model, benchmark, predicted, gold, result, scoring_method`), one file per model,
920 items each.

## Methodology

The three analyses reuse the original code paths without changes:
`tutor_cat.mcq_irt.load_benchmark` and `filter_items`, the girth `twopl_mml` (2PL) and
`rasch_mml` (1PL) fitters, the EAP-theta Fisher-information CAT with an 8-item floor, the
mean-probability predictor, the seed-7 split with `n_test = min(12, n_models // 3)`, and the
least-squares linear map from raw predicted accuracy to true accuracy. `run_expanded_calibration.py`
covers the single split, the controlled comparison, and the linear recalibration.
`run_kfold_cv.py` covers the cross-validation.

OLD is reproduced by subsetting the 78-model matrix to the 52 originally-scored models. Every
model answers the identical 920 items in the identical order, so that subset is the exact
matrix the original scripts saw. The reproduction is a check on faithfulness, and it matches
the published numbers to four decimals (2PL SE<=0.3 r=0.7163, MAE=0.063, 10.25 items; 1PL
SE<=0.3 r=0.8903, 43.83 items).

### Why three lenses instead of one

The original study used a single seed-7 held-out split of 12 models. That is faithful but
noisy, and it has a specific problem for an OLD-versus-NEW comparison: the seed-7 permutation
over 52 models and over 78 models selects different held-out sets, so the two r values are not
measured on the same targets. The NEW split happens to draw 12 models whose true accuracy
spans only 0.252 to 0.287, and Pearson r collapses when the target range is that narrow (the
NEW single-split 1PL r of 0.457 is almost entirely this artifact). To get a fair read, this
folder adds two designs on top of the faithful single split:

1. Controlled: fix the evaluation set to the 12 models OLD held out, then compare a bank
   calibrated on the 40 OLD-train models against a bank calibrated on those same 40 plus the
   26 new models (66 total). The evaluated targets are identical, so any change is the
   calibration set.
2. K-fold cross-validation (K=6): every model is held out once and predicted from a bank fit
   on the rest. Pooling all out-of-sample predictions gives a low-variance recovery number.
   `NEW_on_old52` restricts the pooled NEW predictions to the original 52 models, so recovery
   of the same models can be read under the old and the expanded regimes.

## Results

### OLD reproduction (single seed-7 split, n_test=12)

| model | SE   | r      | MAE_raw | avg items | bank items |
|-------|------|--------|---------|-----------|------------|
| 2PL   | 0.3  | 0.7163 | 0.0630  | 10.25     | 318        |
| 2PL   | 0.15 | 0.8996 | 0.0798  | 275.50    | 318        |
| 1PL   | 0.3  | 0.8903 | 0.1011  | 43.83     | 318        |
| 1PL   | 0.15 | 0.9407 | 0.0973  | 318.00    | 318        |

### Faithful single split, OLD vs NEW

Same protocol on both sets. Read this with the caveat above: the NEW held-out 12 are a
different, range-compressed draw, so the NEW single-split r (especially 1PL) understates
recovery. The bank grows from 318 to 384 kept items because more models expose more
discriminating items.

| model | SE   | r (OLD) | r (NEW) | MAE_raw (OLD to NEW) | avg items (OLD to NEW) |
|-------|------|---------|---------|----------------------|------------------------|
| 2PL   | 0.3  | 0.7163  | 0.8238  | 0.0630 to 0.0724     | 10.25 to 13.33         |
| 2PL   | 0.15 | 0.8996  | 0.7272  | 0.0798 to 0.0842     | 275.50 to 384.00       |
| 1PL   | 0.3  | 0.8903  | 0.4571  | 0.1011 to 0.0648     | 43.83 to 42.67         |
| 1PL   | 0.15 | 0.9407  | 0.7238  | 0.0973 to 0.0743     | 318.00 to 274.58       |

### Controlled comparison (identical 12-model eval set, calibrate on 40 vs 66)

This is the clean read on the calibration set. The 2PL improves markedly and its prediction
spread widens (the extreme models are pulled toward the mean less). The 1PL does not improve.

| model | SE   | r (40) | r (66) | pred spread (40 to 66) | MAE calibrated (40 to 66) | avg items (40 to 66) |
|-------|------|--------|--------|------------------------|---------------------------|----------------------|
| 2PL   | 0.3  | 0.7163 | 0.9049 | 0.230 to 0.333         | 0.0151 to 0.0084          | 10.25 to 12.25       |
| 2PL   | 0.15 | 0.8996 | 0.9580 | 0.242 to 0.339         | 0.0121 to 0.0062          | 275.50 to 294.75     |
| 1PL   | 0.3  | 0.8903 | 0.7340 | 0.283 to 0.167         | 0.0104 to 0.0144          | 43.83 to 43.17       |
| 1PL   | 0.15 | 0.9407 | 0.9255 | 0.245 to 0.184         | 0.0095 to 0.0092          | 318.00 to 288.50     |

### K-fold cross-validation (K=6, every model held out once)

The population-level, low-variance comparison. `NEW_on_old52` holds the evaluated models fixed
at the original 52. The 2PL improves on both the full population and the fixed 52, and reaches
SE<=0.3 in fewer items. The 1PL is flat to slightly lower.

| model | SE   | OLD (52) | NEW (78) | NEW_on_old52 | avg items OLD to NEW | MAE_loo_cal OLD to NEW |
|-------|------|----------|----------|--------------|----------------------|------------------------|
| 2PL   | 0.3  | 0.6846   | 0.7654   | 0.7463       | 17.81 to 14.64       | 0.0115 to 0.0121       |
| 2PL   | 0.15 | 0.8338   | 0.8782   | 0.8331       | 319.58 to 295.97     | 0.0086 to 0.0089       |
| 1PL   | 0.3  | 0.7835   | 0.7577   | 0.7239       | 43.38 to 43.33       | 0.0097 to 0.0124       |
| 1PL   | 0.15 | 0.8819   | 0.8931   | 0.8528       | 330.73 to 297.74     | 0.0075 to 0.0082       |

### Linear recalibration on the expanded set

The raw predicted accuracy sits on a steep, shifted line relative to `y = x`, so the raw MAE
is dominated by an affine offset, not by rank error. A single least-squares map fit on the
train models and applied to held-out models removes most of it while leaving r unchanged, the
same finding as the original study. On the expanded set the slope rises (the raw predicted
axis is less over-wide than before) and the calibrated MAE stays small.

Single-split fit (fit on train, applied to held-out test):

| model | SE   | slope (OLD to NEW) | intercept (OLD to NEW) | MAE raw to calibrated (NEW) |
|-------|------|--------------------|------------------------|-----------------------------|
| 2PL   | 0.3  | 0.152 to 0.243     | 0.214 to 0.187         | 0.0724 to 0.0049 (93.3%)    |
| 2PL   | 0.15 | 0.169 to 0.244     | 0.208 to 0.186         | 0.0842 to 0.0057 (93.2%)    |
| 1PL   | 0.3  | 0.171 to 0.239     | 0.208 to 0.191         | 0.0648 to 0.0097 (85.0%)    |
| 1PL   | 0.15 | 0.188 to 0.269     | 0.201 to 0.179         | 0.0743 to 0.0060 (91.9%)    |

The controlled fit on the identical 12-model eval set tells the same story for the 2PL: the
slope goes from 0.152 to 0.242 at SE<=0.3, and the calibrated MAE drops from 0.0151 to 0.0084.

## What adding the 26 models changed

- Does r improve? For the 2PL bank, yes, and consistently. Controlled r goes from 0.716 to
  0.905 at SE<=0.3 and from 0.900 to 0.958 at SE<=0.15. K-fold r goes from 0.685 to 0.765 on
  the full population and from 0.685 to 0.746 on the same 52 old models. For the 1PL bank r is
  flat to slightly lower (k-fold 0.783 to 0.724 on the same 52 at SE<=0.3). The 2PL is the
  primary bank in the original study, so the headline is an improvement.
- Does scale compression ease? For the 2PL, yes. The predicted spread on the fixed eval set
  widens from 0.230 to 0.333 at SE<=0.3, so the top and bottom models are pulled toward the
  mean less. The k-fold 2PL spread also widens (0.297 to 0.353). The 1PL is mixed and does not
  show a clean easing.
- Do standard errors drop? At a fixed SE stopping rule the achieved SE is capped at the
  target, so this shows up as items needed to reach the target. The k-fold 2PL reaches SE<=0.3
  in fewer items (17.81 to 14.64), which means the larger bank is more informative per item.
  On the fixed 12-model controlled eval the 2PL item count moves the other way by a small
  amount (10.25 to 12.25), within the noise of 12 models.
- Does the true-accuracy range widen? Yes, at the top. The pool range goes from 0.225 to 0.312
  (span 0.087) to 0.225 to 0.338 (span 0.113). The new 7B instruct models set the new top:
  Intel/neural-chat-7b-v3-2 (0.338), argilla/notus-7b-v1 (0.337), ibm/merlinite-7b (0.331),
  Open-Orca/Mistral-7B-OpenOrca (0.317), allenai/OLMo-2-1124-7B-Instruct (0.315). The bottom is
  unchanged at 0.225.

Why the 2PL benefits and the 1PL does not: pedagogy items have very heterogeneous
discriminations, and true accuracy is squeezed into a narrow 0.22 to 0.34 band. The 2PL can
fit per-item discrimination, so more (and higher-scoring) calibration models let it place item
difficulties and slopes more precisely and spread its predictions. The 1PL fixes every
discrimination at 1, so extra models cannot buy it the same resolution, and adding the higher
7B scorers shifts the single difficulty scale without improving rank recovery.

## Scorability

All 78 models produced a complete 920-item response vector with no missing predictions. None
is a degenerate single-answer or repetitive-loop case: the most any model repeats one option
is 34 percent of items (`model_inventory.csv`, column `top_pred_frac`). Four small models score
just below the 0.25 random baseline (bigscience/bloom-560m 0.229, BEE-spoke smol_llama-220M
variants 0.225 to 0.229, facebook/opt-350m 0.241), but a below-chance score is a valid low
score, not a scoring failure. Following the original methodology, which applies no model-level
exclusion beyond complete-case filtering, none were excluded. Count excluded: 0.

## Files

Scripts:

- `run_expanded_calibration.py`: single split, controlled comparison, linear recalibration,
  model inventory, and the figures below. CPU-only, `uv run`, local `.mplcache`.
- `run_kfold_cv.py`: K-fold cross-validation (default K=6), reusing the same machinery.

Result tables:

- `expanded_diagnostic_summary.csv`: single-split r, MAE_raw, mean items, bank size, OLD and NEW.
- `expanded_linear_calibration.csv`: linear map (train_fit, loo_test, insample_naive) per set.
- `controlled_comparison.csv`: fixed 12-model eval, calibrate on 40 vs 66.
- `kfold_cv_summary.csv`: pooled CV recovery for OLD, NEW, and NEW_on_old52.
- `old_vs_new_comparison.csv`: single-split deltas.
- `model_inventory.csv`: per-model accuracy, is_new flag, and the degeneracy screen.
- `results_snapshot.json`, `kfold_cv_snapshot.json`: machine-readable snapshots.
- `diag_pedagogy_{2pl,1pl}_se{0.3,0.15}_{old,new}.csv`: per-model held-out predictions.

Figures:

- `diag_pedagogy_{2pl,1pl}_se{0.3,0.15}_new.png`: NEW recovery scatter, predicted vs actual.
- `calib_pedagogy_{2pl,1pl}_se{0.3,0.15}_new.png`: raw-with-fit and calibrated, side by side.
- `controlled_calib40_vs_66_se0.3.png`: same 12 targets, 40-model vs 66-model bank.
- `compare_old_vs_new_se0.3.png`: single-split OLD and NEW recovery overlay.
- `kfold_old_vs_new_se0.3.png`: 6-fold CV recovery, OLD 52 vs NEW 78.

Input data:

- `_mcq_data/pedagogy/*.csv`: 78 per-item response files downloaded from S3.

## Reproduce

```
cd AdaptiveTesting/Research/01_MCQ_ATLAS/data/pedagogy_feasibility/expanded_calibration
uv run python run_expanded_calibration.py
uv run python run_kfold_cv.py 6
```
