# Parameter-band extrapolation of an ATLAS 3PL item bank

**Question.** Does an IRT (3PL) item bank calibrated on models from ONE parameter range
extrapolate to models in a DIFFERENT, non-overlapping range? We test BOTH directions with
a deliberate GAP in the middle (no MIDDLE-band models used anywhere), plus in-range
controls so extrapolation can be compared against interpolation.

- **LOW→HIGH** (extrapolate up): calibrate on LOW-param models, test on HIGH-param models.
- **HIGH→LOW** (extrapolate down): calibrate on HIGH-param models, test on LOW-param models.
- **LOW→LOW** / **HIGH→HIGH** (controls): calibrate and test within the same band (held-out).

Each cross condition shares its held-out test set with its in-range control (LOW→HIGH and
HIGH→HIGH share the HIGH test set; HIGH→LOW and LOW→LOW share the LOW test set), so the
**extrapolation penalty** is a clean bank-only comparison:

```
penalty(LOW→HIGH) = r(HIGH→HIGH) − r(LOW→HIGH)     # transfer onto the HIGH test set
penalty(HIGH→LOW) = r(LOW→LOW)   − r(HIGH→LOW)     # transfer onto the LOW  test set
```

## Method (reused, unchanged)

Calibration + CAT reuse the exact validated ATLAS pipeline in
`AdaptiveTesting/Research/scripts/openlm_trainsize_sweep.py`: chunked 3PL fit in R `mirt`,
mean-sigma chunk linking with polarity flip, the **critical a>0 positive-discrimination
item filter** (a≤0 items are mis-linked/degenerate and poison the CAT — see
`link_failure_diagnosis/`), and EAP / Fisher-information CAT with p-IRT accuracy recovery.
**Only the model-SELECTION logic changed**: parameter-band bands instead of random k-subset.
Model→params from `05_Data_Availability/data/openlm_download_status.csv` (`status==ok`).
Runner: `AdaptiveTesting/Research/scripts/param_extrapolation_sweep.py`.

## Band definition & counts

- **LOW = params_b ≤ 2.0 B**, **HIGH = params_b ≥ 3.5 B**, **MIDDLE = (2.0, 3.5) B EXCLUDED**.
- Thresholds chosen so both bands have ≥100 calibration models while keeping a real 1.5 B
  gap. (The suggested HIGH ≥ 4.0 B leaves only 137 HIGH models → <110 after the test
  hold-out; HIGH ≥ 3.5 B is healthier.)
- OpenLM `status==ok` counts (n=1101): **LOW=567, MIDDLE=310 (excluded), HIGH=224.**
- Per-benchmark counts inside the cleaned response matrix, with the **matched** calibration N
  (larger band subsampled to the smaller band's available N; `n_test=30` held out per band):

| bench  | LOW in matrix | HIGH in matrix | matched N_calib | N_test/band | seeds |
|--------|---------------|----------------|-----------------|-------------|-------|
| ifeval | 567           | 224            | **194**         | 30          | 3     |
| math   | 429           | 199            | **169**         | 30          | 3     |
| gpqa   | 567           | 224            | **194**         | 30          | 3     |

Histogram: `param_band_histogram.png`.

## Results — 4-condition Pearson r (mean ± sd over 3 seeds)

### SE ≤ 0.3 (primary)

| bench  | LOW→LOW (ctrl) | HIGH→HIGH (ctrl) | LOW→HIGH (extrap ↑) | HIGH→LOW (extrap ↓) | penalty LOW→HIGH | penalty HIGH→LOW |
|--------|----------------|------------------|---------------------|---------------------|------------------|------------------|
| ifeval | 0.910 ± 0.005  | 0.955 ± 0.015    | **0.917 ± 0.022**   | **0.919 ± 0.023**   | **+0.038**       | **−0.009**       |
| math   | 0.891 ± 0.036  | 0.907 ± 0.055    | **0.789 ± 0.079**   | **0.859 ± 0.104**   | **+0.118**       | **+0.032**       |
| gpqa   | 0.854 ± 0.042  | 0.835 ± 0.035    | **0.389 ± 0.156**   | **0.540 ± 0.101**   | **+0.445**       | **+0.314**       |

### SE ≤ 0.2

| bench  | LOW→LOW (ctrl) | HIGH→HIGH (ctrl) | LOW→HIGH (extrap ↑) | HIGH→LOW (extrap ↓) | penalty LOW→HIGH | penalty HIGH→LOW |
|--------|----------------|------------------|---------------------|---------------------|------------------|------------------|
| ifeval | 0.933 ± 0.013  | 0.955 ± 0.013    | 0.935 ± 0.011       | 0.925 ± 0.028       | +0.020           | +0.007           |
| math   | 0.900 ± 0.031  | 0.907 ± 0.055    | 0.796 ± 0.077       | 0.862 ± 0.106       | +0.111           | +0.038           |
| gpqa   | 0.861 ± 0.046  | 0.858 ± 0.024    | 0.393 ± 0.113       | 0.603 ± 0.113       | +0.465           | +0.258           |

Mean CAT items (SE ≤ 0.3): ifeval 26–50; math 87–292; gpqa 12–32. Full per-seed detail
(r, MAE, mean/median items, %bank, n_calib, n_test, n_bank_items, band thresholds) in
`param_extrapolation_results.csv`. Figures: `<bench>_param_extrapolation.png` and
`param_extrapolation_combined.png`.

## Interpretation

**Does the bank extrapolate out of range?** Yes for strongly-linking benchmarks, with a
direction-dependent cost:

- **ifeval** — extrapolation is essentially free. Both out-of-range conditions (0.917 / 0.919)
  match the LOW→LOW control (0.910); penalties are within noise (+0.038 / −0.009). The item
  bank transfers across the parameter gap in both directions.
- **math** — extrapolation works but with a real, **asymmetric** penalty. LOW→HIGH costs
  +0.118 r (0.907→0.789) while HIGH→LOW costs only +0.032. Calibrating on small models and
  testing large is the harder direction.
- **gpqa** — out-of-range transfer largely **fails** (r drops to 0.39 / 0.54 from ~0.85
  controls). gpqa links weakly even in-range and is near guessing for small models, so few
  items carry positive discrimination in the LOW bank (LOW bank ≈446 usable items vs HIGH
  bank ≈771), and the mismatch is severe out of range. Treat gpqa extrapolation as
  unreliable, not as a property of parameter range alone.

**Is it symmetric?** **No — consistently asymmetric.** In all three benchmarks the LOW→HIGH
penalty exceeds the HIGH→LOW penalty (ifeval +0.038 vs −0.009; math +0.118 vs +0.032; gpqa
+0.445 vs +0.314). Extrapolating **up** (calibrate small, test large) is the weaker
direction. This **supports the stated hypothesis**: a bank calibrated on small models is
dominated by items that are easy/saturated for large models and therefore lose
discrimination at the top, so it predicts large-model accuracy less well than a large-model
bank predicts small-model accuracy.

## Verdict

> **The ATLAS 3PL bank extrapolates across a parameter gap for strongly-linking benchmarks
> (ifeval near-perfectly, math with a modest penalty), but transfer is asymmetric: LOW→HIGH
> (small→large) always costs more than HIGH→LOW, consistent with easy items saturating at the
> top. For weakly-linking gpqa, out-of-range transfer breaks down.**

## bbh: not delivered (honest note)

bbh was attempted (optional) but **could not be calibrated** with the standard pipeline: the
HIGH-band 3PL fit is unstable on one 100-item chunk (`mirt` EM log-likelihood decreasing,
`fscores` returns NA → `Error in if (nc == 0)`), which is deterministic across retries and
blocks chunk linking. 57 of 58 chunks fit; chunk 301 does not. No bbh numbers are reported
rather than fabricating around a failed calibration. musr was skipped per the task.

## Reproduce

```bash
export PYTHONPATH=/Users/arhant/Documents/EDLM/olmo-eval-full/eduLLM-Evals
cd AdaptiveTesting/Research/scripts
uv run python param_extrapolation_sweep.py --bench ifeval --seeds 3 --workers 6
uv run python param_extrapolation_sweep.py --bench math   --seeds 3 --workers 6
uv run python param_extrapolation_sweep.py --bench gpqa   --seeds 3 --workers 6
uv run python param_extrapolation_sweep.py --plot   # rebuild figures from the CSV
```
