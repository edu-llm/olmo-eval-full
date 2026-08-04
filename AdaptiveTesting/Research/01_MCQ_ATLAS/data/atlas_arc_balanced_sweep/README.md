# Balanced ATLAS ARC: random train-size sweep + reverse-engineering oracle subsets

## What this is

Two questions on the BALANCED, full-range ATLAS ARC calibration pool (0.5B-59B,
Gaussian-sampled over ability; the pool behind ATLAS's published ARC r~0.83), at
calibration-pool steps of 100:

1. **random train-size sweep** -- how the NUMBER of calibration models affects
   held-out accuracy recovery (r vs N, 5 seeds, mean +/- sd).
2. **reverse-engineering oracle subset selection** -- calibrate the whole capped
   pool, read each model's fitted latent ability (EAP theta), then choose N models
   that span the latent-ability space (`oracle_stratified`) or greedily add the
   model that most improves held-out r (`oracle_greedy`). Both use post-hoc
   information, so they are an UPPER BOUND, not a deployable rule.

This deliberately uses the balanced pool and NOT the size-skewed 0.5-7B ARC pool
(~95% 7B), which topped out near r~0.59.

## Method (reused, unchanged)

Same validated 3PL pipeline as `openlm_trainsize_sweep.py` (chunked R mirt 3PL,
mean-sigma linking with polarity flip, a>0 item filter) + `oracle_subset_selection.py`
selection strategies + SE<=0.3 EAP/Fisher adaptive CAT (MIN_ITEMS=8). Only the data
source changes: the ARC train matrix is the model x item frame, and the pre-split
417 ARC test models are the FIXED held-out set for every strategy and N.

Calibration pool capped at 500 models, drawn once with a fixed seed
(pool_seed=13) from the ~3.7k-model train pool. That capped pool IS
"the full pool" here. Latent groups are 5 quantile bins of the full-pool
EAP theta (`latent_groups_arc.csv`). Reported at SE<=0.3.

## Full-pool reference

- Pool used: 500 models. **Full-pool reference r = 0.9124** (SE<=0.3), MAE = 0.0340, bank = 669 items.
- Contrast: ATLAS full ARC recovers r~0.83; the size-skewed 0.5-7B pool topped near r~0.59.

## (a) random correlation-vs-N

| N | r (mean +/- sd over 5 seeds) |
|---|---|
| 50 | 0.9003 +/- 0.0103 |
| 100 | 0.9043 +/- 0.0094 |
| 150 | 0.9135 +/- 0.0086 |
| 200 | 0.9208 +/- 0.0051 |
| 250 | 0.9231 +/- 0.0101 |
| 300 | 0.9261 +/- 0.0036 |
| 350 | 0.9240 +/- 0.0133 |
| 400 | 0.9205 +/- 0.0058 |
| 450 | 0.9229 +/- 0.0028 |
| 500 | 0.9124 +/- 0.0000 |

Shape: per-+100 gains [+100@N=100:+0.004, +100@N=150:+0.009, +100@N=200:+0.007, +100@N=250:+0.002, +100@N=300:+0.003, +100@N=350:-0.002, +100@N=400:-0.003, +100@N=450:+0.002, +100@N=500:-0.010]. random r plateaus (per-+100 gain < 0.01) by N~100.

## (b) reverse-engineering oracle subsets

### oracle_stratified (latent-group spread)

| N | r |
|---|---|

### oracle_greedy (forward greedy on held-out r; capped)

| N | r |
|---|---|
| 100 | 0.9300 |

## Takeaway

On the balanced full-range ARC pool the full 500-model calibration recovers held-out accuracy at r=0.912 -- close to ATLAS's ~0.83 and far above the size-skewed 0.5-7B pool's ~0.59. So a balanced full-range pool DOES recover much better than the skewed 0.5-7B pool. Random sweep: r rises from 0.900 at N=50 to 0.912 at N=500. oracle_greedy reaches within 0.02 of the reference at N=100 (random needs N~50).

## Outputs

- `results.csv`: strategy, N, seed, se_target, r, mae, items, n_bank_items, n_pool, n_test, calib_s.
- `latent_groups_arc.csv`: group id, theta bounds, size, mean theta.
- `selected_subsets.csv`: strategy, N, model, latent_group, theta (oracle picks per N).
- `oracle_curve_arc.png`: r vs N (random band, oracle lines, reference lines).
- `latent_axis_arc.png`: selected subset positions along the latent-ability axis.

## Caveats

- Oracle/upper-bound only: `oracle_stratified` uses the final fitted abilities and
  `oracle_greedy` peeks at the held-out accuracies. Neither is a deployable rule.
- 3PL on very small model subsets is unstable (many items go constant / a<=0 and are
  dropped, shrinking the bank); treat small-N points as noisy. See `n_bank_items`.
- Transparency: one N=450 random fit (seed=1) collapsed to a degenerate 3PL solution
  (r=0.637, versus about 0.92 for its 4 sibling seeds) and was excluded as a fit failure,
  so N=450 aggregates over the remaining 4 seeds; the value is recorded here so it is not lost.
- `oracle_stratified` is deterministic (one curve, no band).