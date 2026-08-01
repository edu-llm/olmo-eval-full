# Pedagogy MCQ CAT diagnostic — linear calibration (predicted → actual)

## The problem: high MAE from a slope/intercept offset, not noise

The pedagogy CAT diagnostic predicts each model's pedagogy accuracy from an IRT θ /
p-IRT mapping (mean of item success probabilities at the EAP θ). Predictions track true
accuracy well in **rank** (Pearson r up to ≈0.94), but the raw predicted values sit on a
much steeper, shifted line than `y = x`:

- The IRT mean-probability predictions live in a compressed θ→p range but are read off
  against the full 0–1 accuracy axis, so the predicted spread is far wider than the true
  accuracy spread (true pedagogy accuracy is squeezed into ≈0.22–0.31).
- Result: the raw cloud has the right ordering but a systematic **slope ≈ 6× too steep and
  a large positive intercept** → the diagnostic over-predicts, especially the 1PL/Rasch
  bank (raw MAE ≈ 0.10, the worst of the four cases).

Because the error is a deterministic affine offset, a single linear map
`y_hat = a·x + b` (x = raw predicted, y = true accuracy) removes almost all of it while
leaving r **exactly** unchanged.

## Method (honest, no in-sample circularity)

Same response matrix, same 40-train / 12-test split (seed 7), same item-keep filter, and
the same bank calibration + EAP-θ + Fisher-info CAT as `scripts/mcq_diagnostic.py` (2PL)
and `rasch_1pl/run_pedagogy_1pl.py` (1PL) — reused directly, IRT not reinvented.

- **Headline `train_fit`**: fit `(a, b)` by least squares on the **40 calibration/train
  models'** (predicted, actual) pairs, then apply to the **12 held-out test models**.
  This is the correct, honest number.
- **`loo_test`**: leave-one-out on the 12 test models (fit on 11, predict the 1 left out).
- **`insample_naive`**: fit and evaluate on the same 12 test points — optimistic reference
  only, **not** a validity claim.

## Results — honest (train-fit) headline

| model | SE   | r     | MAE_raw | MAE_calibrated | % reduction | a (slope) | b (intercept) |
|-------|------|-------|---------|----------------|-------------|-----------|---------------|
| 2PL   | 0.3  | 0.716 | 0.0630  | **0.0151**     | **76.1%**   | 0.152     | 0.214         |
| 2PL   | 0.15 | 0.900 | 0.0798  | **0.0121**     | **84.8%**   | 0.169     | 0.208         |
| 1PL   | 0.3  | 0.890 | 0.1011  | **0.0104**     | **89.7%**   | 0.171     | 0.208         |
| 1PL   | 0.15 | 0.941 | 0.0973  | **0.0095**     | **90.3%**   | 0.188     | 0.200         |

Fitted equations (true_accuracy ≈ a·predicted + b):

- 2PL SE 0.3 :  `y = 0.152·x + 0.214`
- 2PL SE 0.15:  `y = 0.169·x + 0.208`
- 1PL SE 0.3 :  `y = 0.171·x + 0.208`
- 1PL SE 0.15:  `y = 0.188·x + 0.200`

The small slope (≈0.15–0.19) is exactly the point: the raw predicted axis is ~6× too wide,
and calibration compresses it back onto the true 0.20–0.31 pedagogy range.

## Takeaway

Linear recalibration closes the MAE gap: honest test-set MAE drops **76–90%** across all
four cases, and r is unchanged. The 1PL/Rasch over-prediction — the worst raw case
(MAE ≈ 0.10) — is fixed, ending up with the *lowest* calibrated MAE (≈0.010), on par with
or better than the 2PL. Cross-checks agree: LOO (0.0084–0.0162) closely matches train-fit,
confirming the improvement is real and not fit noise.

## Files

- `run_linear_calibration.py` — reproducer (CPU-only, `uv run`, local `.mplcache`).
- `pedagogy_linear_calibration.csv` — one row per (model, se, calibration_method).
- `calib_pedagogy_{2pl,1pl}_se{0.3,0.15}.png` — before/after scatter (raw with fitted line;
  calibrated vs y=x).

## Footnote — optimistic in-sample reference (not a validity claim)

Fitting and evaluating on the same 12 test points gives MAE 0.0132 / 0.0079 (2PL 0.3 /
0.15) and 0.0095 / 0.0070 (1PL 0.3 / 0.15), i.e. 79–93% reduction. These are upper bounds;
the honest train-fit / LOO numbers above are the ones to quote.
