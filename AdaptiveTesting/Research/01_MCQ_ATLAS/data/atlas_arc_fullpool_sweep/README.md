# Full-pool ATLAS ARC: random train-size sweep + reverse-engineering oracle greedy

## What this is

Two questions on the BALANCED, full-range ATLAS ARC calibration pool (0.5B-59B,
Gaussian-sampled over ability; the pool behind ATLAS's published ARC r~0.83), at
calibration-pool steps of 50, using the ENTIRE ~3.7k-model train pool as the
reference (no 500-model cap):

1. **random train-size sweep** -- how the NUMBER of calibration models affects
   held-out accuracy recovery (r vs N, 5 seeds, mean +/- sd).
2. **reverse-engineering oracle greedy subset selection** -- calibrate the whole
   pool, read each model's fitted latent ability (EAP theta), then greedily add the
   model that most improves held-out r (`oracle_greedy`). This uses post-hoc
   information, so it is an UPPER BOUND, not a deployable rule.

This deliberately uses the balanced pool and NOT the size-skewed 0.5-7B ARC pool
(~95% 7B), which topped out near r~0.59.

## Method (reused, unchanged)

Same validated 3PL pipeline as `openlm_trainsize_sweep.py` (chunked R mirt 3PL,
mean-sigma linking with polarity flip, a>0 item filter) + `oracle_subset_selection.py`
selection strategies + SE<=0.3 EAP/Fisher adaptive CAT (MIN_ITEMS=8). Only the data
source changes: the ARC train matrix is the model x item frame, and the pre-split
417 ARC test models are the FIXED held-out set for every strategy and N.

The reference is the ENTIRE 3747-model train pool (no cap): the full-pool
calibration is fit on all of it, random subsets are drawn from it, and greedy
candidates come from it. Latent groups are 5 quantile bins of the
full-pool EAP theta (`latent_groups_arc.csv`). Reported at SE<=0.3.

## Full-pool reference

- Pool used: 3747 models (the whole train pool). **Full-pool reference r = 0.9296** (SE<=0.3), MAE = 0.0314, bank = 662 items.
- Contrast: ATLAS full ARC recovers r~0.83; the size-skewed 0.5-7B pool topped near r~0.59.

## (a) random correlation-vs-N

| N | r (mean +/- sd over 5 seeds) |
|---|---|
| 50 | 0.9040 +/- 0.0077 |
| 100 | 0.9082 +/- 0.0225 |
| 150 | 0.9218 +/- 0.0067 |
| 200 | 0.9128 +/- 0.0227 |
| 250 | 0.9285 +/- 0.0049 |
| 300 | 0.9168 +/- 0.0076 |
| 350 | 0.9250 +/- 0.0156 |
| 400 | 0.9222 +/- 0.0243 |
| 450 | 0.9167 +/- 0.0091 |
| 500 | 0.9117 +/- 0.0155 |

Shape: per-step gains [+50@N=100:+0.004, +50@N=150:+0.014, +50@N=200:-0.009, +50@N=250:+0.016, +50@N=300:-0.012, +50@N=350:+0.008, +50@N=400:-0.003, +50@N=450:-0.006, +50@N=500:-0.005]. random r plateaus (per-step gain < 0.01) by N~100.

## (b) reverse-engineering oracle greedy

### oracle_greedy (forward greedy on held-out r)

| N | r |
|---|---|
| 50 | 0.9223 |

## Takeaway

On the balanced full-range ARC pool the full 3747-model calibration recovers held-out accuracy at r=0.930, versus ATLAS's published ~0.83 and the size-skewed 0.5-7B pool's ~0.59. So a balanced full-range pool recovers much better than the skewed 0.5-7B pool. Random sweep: r rises from 0.904 at N=50 to 0.912 at N=500. oracle_greedy reaches within 0.02 of the reference at N=50 (random needs N~150).

## Outputs

- `results.csv`: strategy, N, seed, se_target, r, mae, items, n_bank_items, n_pool, n_test, calib_s.
- `latent_groups_arc.csv`: group id, theta bounds, size, mean theta.
- `selected_subsets.csv`: strategy, N, model, latent_group, theta (greedy picks per N).
- `oracle_curve_arc.png`: r vs N (random band, greedy line, reference lines).
- `latent_axis_arc.png`: selected subset positions along the latent-ability axis.

## Caveats

- Oracle/upper-bound only: `oracle_greedy` peeks at the held-out accuracies to pick
  each next model, so it is NOT a deployable rule.
- 3PL on very small model subsets is unstable (many items go constant / a<=0 and are
  dropped, shrinking the bank); treat small-N points as noisy. See `n_bank_items`.
- Random subsets are genuine 5-seed draws from the full 3747-model pool, so
  N=500 has seed-to-seed variance (it is a 500-of-pool subset, not the whole pool).