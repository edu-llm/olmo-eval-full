# Experiment 3 (expanded): 1PL vs 2PL vs 3PL on the 78-model pedagogy set

The original Experiment 3 (`../run_pl_comparison.py`) compared 1PL / 2PL / 3PL on a
52-model pedagogy pool with a 40-model calibration split. We have since scored 26
more models on the identical 920 pedagogy items, taking the pool to 78 models. This
folder re-runs the 1PL / 2PL / 3PL comparison on the expanded set and reports it
under three sampling designs so the calibration-set effect is read fairly.

## Headline verdict

A bigger calibration pool does not rescue 3PL. On the primary SE<=0.3 operating
point, held-out recovery for 3PL falls when the pool grows from 40 to 66 models
(controlled r 0.820 to 0.507; k-fold r 0.625 to 0.499). The extra models do reduce
3PL discrimination pinning (68 percent of items pinned at 40 models, 41 percent at
66), but the guessing floor keeps pinning (45 then 51 items at the c ceiling) and the
held-out order still collapses. At 66 models 3PL is the worst of the three at SE<=0.3
(2PL 0.905, 1PL 0.734, 3PL 0.507). The clear winner from more data is 2PL, which
gains in both fair designs; 1PL is roughly flat. 3PL remains not usable at this pool
size.

## What is new relative to the original Experiment 3

- The pool grows from 52 to 78 models. The 26 new models keep the same 920 pedagogy
  items in the same order, so restricting the 78-model matrix to the original 52 is
  bit-identical to the matrix the original scripts saw (verified).
- The prior expanded run (`../../expanded_calibration/`) already did the fair designs
  for 1PL and 2PL only. This deliverable adds 3PL to every design and packages a
  clean 1PL / 2PL / 3PL comparison.
- pedagogy only. PIQA and SocialIQa were not re-scored and are left alone.

## Designs

1. Controlled (fair). Fix the eval set to the original seed-7 held-out 12 models.
   Calibrate a bank on the original 40 train models (calib40), then on those same 40
   plus the 26 new models (calib66). Identical eval set and item axis, so the only
   thing that changes is how many models the bank saw. This is the clean read on
   whether more calibration models help each model order.
2. K-fold CV (fair, K=6). Every model is held out once and predicted from a bank fit
   on the rest, then all out-of-sample predictions are pooled into one lower-variance
   recovery r. Run on OLD (52) and NEW (78). The `r_new_on_old52` column restricts the
   NEW pooled predictions to the original 52 models, so the calibration-set effect can
   be read with the evaluated models held fixed.
3. Naive single split (caveated). The seed-7 draw of 12 held-out from 78 with 66
   train. This draw spans a very narrow accuracy band (about 0.25 to 0.29), which
   deflates Pearson r. It is reported for completeness and flagged in the CSV. The
   fair designs lead the conclusions.

## Machinery (reused, not reimplemented)

- Fitters via `se_sweep_small_pool.fit_bank`: 1PL girth `rasch_mml`, 2PL girth
  `twopl_mml`, 3PL the self-contained MML-EM `fit_3pl_mml` (girth `threepl_mml` is
  broken under scipy>=1.15).
- CAT via `se_sweep.full_cat_traces` with `pred_meanprob`, `MIN_ITEMS=8`, stop at
  `SE<=target`. At c=0 this reduces exactly to the 2PL / 1PL CAT, so 1PL / 2PL
  reproduce the prior numbers.
- Item filter `filter_items` on the train slice (ATLAS point-biserial rules).
- Uniform usability filter (finite, a>0, 0<=c<1) on every fitted bank. It is a no-op
  for 1PL / 2PL and, because the 3PL EM lower-bounds a at 0.01, drops nothing for 3PL,
  so 3PL degeneracy shows up as bounds-pinning rather than as dropped items.
- The old / new model partition (`NEW_STEMS`) and the seed-7 split are imported from
  `../../expanded_calibration/run_expanded_calibration.py`, so the partitions match.

## Validation (all passed)

1PL and 2PL reproduce the prior fair-design numbers exactly (tolerance 5e-3):

| design | model | SE | recomputed | reference | source |
|---|---|---|---|---|---|
| controlled r_40 | 2PL | 0.30 | 0.7163 | 0.7163 | `controlled_comparison.csv` |
| controlled r_66 | 2PL | 0.30 | 0.9049 | 0.9049 | `controlled_comparison.csv` |
| controlled r_40 | 1PL | 0.30 | 0.8903 | 0.8903 | `controlled_comparison.csv` |
| controlled r_66 | 1PL | 0.30 | 0.7340 | 0.7340 | `controlled_comparison.csv` |
| controlled r_40 | 2PL | 0.15 | 0.8996 | 0.8996 | `controlled_comparison.csv` |
| controlled r_40 | 1PL | 0.15 | 0.9407 | 0.9407 | `controlled_comparison.csv` |
| k-fold OLD52 | 2PL | 0.30 | 0.6846 | 0.6846 | `kfold_cv_summary.csv` |
| k-fold NEW78 | 2PL | 0.30 | 0.7654 | 0.7654 | `kfold_cv_summary.csv` |
| k-fold OLD52 | 1PL | 0.30 | 0.7835 | 0.7835 | `kfold_cv_summary.csv` |
| k-fold NEW78 | 1PL | 0.30 | 0.7577 | 0.7577 | `kfold_cv_summary.csv` |

The naive single split also reproduces the prior 1PL / 2PL numbers
(`../../expanded_calibration/expanded_diagnostic_summary.csv`): 1PL SE0.3 r=0.4571,
2PL SE0.3 r=0.8238, 1PL SE0.15 r=0.7238, 2PL SE0.15 r=0.7272.

Note on the 3PL 40-pool baseline. The committed `../pl_1_2_3_comparison.csv` reports
3PL SE0.3 r=0.793 (113.8 items). Under the current library stack the same fitter on
the same data gives 3PL SE0.3 r=0.8195 (90.9 items). This was checked directly: the
original full200 matrix and the expanded `_mcq_data` matrix restricted to 52 models
are bit-identical, the train and test splits match, and both data sources produce the
identical 0.8195 in this environment. The 3PL discrimination pinning count is the same
as the committed run (215 of 318 items), so this is the same degenerate fit shifted by
floating-point differences across library versions, amplified by the unstable CAT
trajectory. 1PL and 2PL are stable and reproduce to four decimals; 3PL at 40 models is
not even reproducible across environments at fixed data. The controlled r_40 and r_66
here come from one consistent environment, so their delta is a valid comparison.

## Results

### Controlled: same 12 held-out, calibrate on 40 vs 66 (`pl_1_2_3_expanded_controlled.csv`)

| model | SE | r (40 pool) | r (66 pool) | delta | items 40 | items 66 |
|---|---|---|---|---|---|---|
| 1PL | 0.30 | 0.890 | 0.734 | -0.156 | 43.8 | 43.2 |
| 2PL | 0.30 | 0.716 | 0.905 | +0.189 | 10.2 | 12.2 |
| 3PL | 0.30 | 0.820 | 0.507 | -0.313 | 90.9 | 45.0 |
| 1PL | 0.15 | 0.941 | 0.925 | -0.015 | 318.0 | 288.5 |
| 2PL | 0.15 | 0.900 | 0.958 | +0.058 | 275.5 | 294.8 |
| 3PL | 0.15 | 0.542 | 0.575 | +0.033 | 298.4 | 200.8 |

### K-fold CV, K=6, pooled out-of-sample r (`pl_1_2_3_expanded_kfold.csv`)

| model | SE | r (OLD 52) | r (NEW 78) | r (NEW on old 52) |
|---|---|---|---|---|
| 1PL | 0.30 | 0.7835 | 0.7577 | 0.7239 |
| 2PL | 0.30 | 0.6846 | 0.7654 | 0.7463 |
| 3PL | 0.30 | 0.6245 | 0.4989 | 0.3703 |
| 1PL | 0.15 | 0.8819 | 0.8931 | 0.8528 |
| 2PL | 0.15 | 0.8338 | 0.8782 | 0.8331 |
| 3PL | 0.15 | 0.6761 | 0.6263 | 0.5123 |

### Naive single split on 78 (`pl_1_2_3_expanded_single_split.csv`, caveated)

| model | SE | r | items | frac reached |
|---|---|---|---|---|
| 1PL | 0.30 | 0.4571 | 42.7 | 1.00 |
| 2PL | 0.30 | 0.8238 | 13.3 | 1.00 |
| 3PL | 0.30 | -0.0519 | 19.5 | 1.00 |
| 1PL | 0.15 | 0.7238 | 274.6 | 0.92 |
| 2PL | 0.15 | 0.7272 | 384.0 | 0.00 |
| 3PL | 0.15 | 0.0934 | 324.3 | 0.17 |

The seed-7 draw of 12 from 78 spans a narrow accuracy band, so these r values are
deflated (1PL collapses to 0.457 here for that reason). Read the fair designs above,
not this table.

## How the 40 to 66 change lands per model order

- 2PL is the winner from more calibration data. Controlled SE0.3 r goes 0.716 to
  0.905, and k-fold SE0.3 r goes 0.685 to 0.765. The tight SE0.15 target also
  improves (controlled 0.900 to 0.958). More models sharpen the 2PL discriminations
  that the thin 40-model pool over-fit.
- 1PL is roughly flat. The controlled SE0.3 drop (0.890 to 0.734) is within the noise
  of a 12-model eval; the lower-variance k-fold moves only 0.784 to 0.758 at SE0.3 and
  slightly up at SE0.15 (0.882 to 0.893). 1PL neither gains much nor collapses.
- 3PL is not rescued. See the verdict below.

## Verdict on 3PL

The larger 66-model pool does not rescue 3PL, and on the primary operating point it
makes 3PL worse.

- SE<=0.3, controlled: 3PL r falls 0.820 to 0.507 as the pool grows 40 to 66.
- SE<=0.3, k-fold: 3PL r falls 0.625 to 0.499.
- At 66 models and SE<=0.3, 3PL (0.507) is the worst of the three (2PL 0.905, 1PL
  0.734).
- SE<=0.15 nudges up (controlled 0.542 to 0.575, k-fold 0.676 to 0.626 which is a
  drop), but 3PL stays far below 1PL and 2PL (about 0.93 to 0.96), and most models
  never reach SE<=0.15 (frac reached 0.08 then 0.50 for controlled 3PL), so the
  tight-SE rows mix reached-SE with ran-out-of-items and are not a clean comparison.
- Instability persists (see `fit_diagnostics.csv`). Discrimination pinning drops from
  215 of 318 items (68 percent) at 40 models to 150 of 366 (41 percent) at 66, so
  more data helps identifiability, but 41 percent is still severe and the guessing
  floor pinning rises (45 then 51 items at the c ceiling of 0.5). The held-out order
  collapses despite the reduced a-pinning.

Bottom line: for small-pool skill banks, 2PL benefits most from adding calibration
models, 1PL stays stable, and 3PL stays unstable and non-competitive. Do not use 3PL
at this pool size.

## Files

- `run_pl_expanded78.py` produces everything in this folder. It does not touch the
  52-model comparison or the `expanded_calibration/` outputs.
- `pl_1_2_3_expanded_controlled.csv` model x SE with r_40, r_66, delta, items, mae,
  frac_reached, kept bank sizes.
- `pl_1_2_3_expanded_kfold.csv` model x SE with r_old52, r_new78, r_new_on_old52,
  items, mae, n_eval.
- `pl_1_2_3_expanded_single_split.csv` naive design with a caveat column.
- `fit_diagnostics.csv` per fit at the 40 and 66 pools: bank size, a and c ranges,
  bounds-pinning counts, fit seconds.
- `per_model_predictions.csv` per held-out model predictions across all designs.
- `pl_1_2_3_expanded78.png` grouped bars of held-out Pearson r for 1PL / 2PL / 3PL,
  40 vs 66 pool, at SE<=0.3 and SE<=0.15.

## Reproduce

CPU-local, no network:

```bash
uv run python AdaptiveTesting/Research/01_MCQ_ATLAS/data/pedagogy_feasibility/pl_1_2_3_comparison/pedagogy_expanded78/run_pl_expanded78.py --workers 6
```

The run falls back to sequential automatically if a process pool is unavailable.
Runtime is dominated by girth `twopl_mml` on the larger banks; 1PL and the 3PL EM fit
in seconds.
