# Oracle (upper-bound) calibration-model subset selection

## What this is (read first)

This experiment finds the THEORETICAL-BEST small subset of calibration models
that keeps roughly the same held-out correlation as the full pool, by
REVERSE-ENGINEERING the choice from the final calibration. We calibrate on the
whole pool, read each model's fitted latent ability (EAP theta) from that bank,
split models into latent-ability groups, and select representatives that span
the ability axis. The greedy variant goes further and directly maximizes the
held-out correlation.

**These are ORACLE numbers / an UPPER BOUND.** They use information not
available before calibration (the final fitted abilities, and for greedy the
held-out accuracies themselves). They are NOT a deployable selection rule; they
set the theoretical floor that practical a-priori strategies (random, and the
diversity strategies in `data/model_diversity_selection/`) are compared against.

## Method

Same validated OpenLM 3PL pipeline as `openlm_trainsize_sweep.py` (chunked R
mirt 3PL, mean-sigma linking with polarity flip, a>0 item filter, EAP/Fisher
p-IRT CAT). Only the choice of the N calibration models changes. The held-out
test set is the same fixed seed-7 10% split for every
strategy and N, so all curves are comparable. Reported at SE<=0.3.

Initial calibration pool is capped at 300 models, drawn once with a
fixed seed from the cleaned train pool (after holding out the test set). That
capped pool IS "the full pool" here: oracle abilities are fit on it and the
full-pool reference correlation uses all of it. Latent groups are
5 quantile bins of the full-pool EAP theta (latent_groups_<bench>.csv).

Strategies: `random` (mean +/- sd over seeds), `oracle_stratified` (spread N
models evenly across the latent groups and across theta within each group),
`oracle_greedy` (forward greedy that adds the model most improving held-out
Pearson r, seeded at the two theta extremes; the strongest oracle).

## Headline: minimum N to hold correlation

### ifeval

- Pool used: 300 models. Full-pool reference r = 0.9499 (SE<=0.3).
- Within 0.02 r of the reference (r >= 0.9299):
    - random needs N ~ 156.3.
    - oracle-stratified: min N = 300 (r = 0.9499), 0.52x fewer than random.
    - oracle-greedy: min N = 50 (r = 0.9413), 3.13x fewer than random.
- Within 0.05 r of the reference (r >= 0.8999):
    - random needs N ~ 88.2.
    - oracle-stratified: min N = 150 (r = 0.9261), 0.59x fewer than random.
    - oracle-greedy: min N = 50 (r = 0.9413), 1.76x fewer than random.

See `best_subset_ifeval.csv` for the theoretical-best subset (models + latent group + theta) at the smallest holding N.

### math

- Pool used: 300 models. Full-pool reference r = 0.9327 (SE<=0.3).
- Within 0.02 r of the reference (r >= 0.9127):
    - random needs N ~ 279.6.
    - oracle-stratified: min N = 100 (r = 0.9213), 2.8x fewer than random.
    - oracle-greedy: min N = 100 (r = 0.9435), 2.8x fewer than random.
- Within 0.05 r of the reference (r >= 0.8827):
    - random needs N ~ 199.3.
    - oracle-stratified: min N = 100 (r = 0.9213), 1.99x fewer than random.
    - oracle-greedy: min N = 100 (r = 0.9435), 1.99x fewer than random.

See `best_subset_math.csv` for the theoretical-best subset (models + latent group + theta) at the smallest holding N.

## Verdict

An oracle subset that spans the latent-ability space keeps the full-pool correlation with far fewer calibration models (ifeval: oracle holds within 0.02 r at N=50, ~3.13x fewer than random; math: oracle holds within 0.02 r at N=100, ~2.8x fewer than random). This is an upper bound; practical strategies fall between this and random.

## Outputs

- `results.csv`: benchmark, strategy (full_pool/random/oracle_stratified/oracle_greedy), N, seed, se_target, r, mae, items, n_bank_items, n_pool, n_test.
- `latent_groups_<bench>.csv`: group id, theta bounds, size, mean theta.
- `best_subset_<bench>.csv`: chosen models + latent group + theta at the smallest holding N.
- `oracle_curve_<bench>.png`: r vs N, random band, oracle lines, reference line.
- `latent_axis_<bench>.png`: selected subset positions along the latent-ability axis.

## Caveats

- Oracle/upper-bound only (uses final fitted abilities; greedy peeks at held-out r).
- Random savings factors use linear interpolation of the random mean curve to the
  tolerance threshold; treat single small-N points as noisy.
- oracle_stratified is deterministic (one curve, no band).
