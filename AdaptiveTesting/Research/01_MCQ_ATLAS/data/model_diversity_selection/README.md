# Model-selection sweep: fewer calibration models by choosing them well

Question: can a smart/diverse choice of calibration models reach the same
held-out correlation as random sampling, but with fewer models?

Method: same validated OpenLM 3PL pipeline as the train-size sweep
(`openlm_trainsize_sweep.py`: chunked R mirt 3PL, mean-sigma linking, a>0
item filter, EAP/Fisher p-IRT CAT). Only the choice of the N calibration
models changes. The held-out test set is the same fixed seed-7 10% split for
every strategy and N, so all curves are comparable. Reported at SE<=0.3.

Strategies: `random` (5 seeds, mean +/- sd), `ability_spread` (models
covering the accuracy range evenly), `diverse` (farthest-point k-center on the
model-by-item correctness matrix, Hamming distance).

## Headline numbers

### ifeval

- Random plateau: r = 0.919 at N = 120 (seed sd = 0.008); match threshold r >= 0.911.
- ability spread: matches the plateau at N = 80 (r = 0.921); random needs N ~ 112.7 for the same r -> 1.41x fewer models.
- response diversity: matches the plateau at N = 120 (r = 0.912); random needs N ~ 112.7 for the same r -> 0.94x fewer models.

### math

- Random plateau: r = 0.865 at N = 120 (seed sd = 0.043); match threshold r >= 0.822.
- ability spread: matches the plateau at N = 50 (r = 0.859); random needs N ~ 47.7 for the same r -> 0.95x fewer models.
- response diversity: matches the plateau at N = 30 (r = 0.837); random needs N ~ 47.7 for the same r -> 1.59x fewer models.

## Verdict

Yes: choosing calibration models well reaches random's plateau correlation with fewer models (ifeval: ability spread 1.41x fewer; math: response diversity 1.59x fewer). Best overall: diversity-aware selection.

## Figures

- `selection_curve_ifeval.png`: r vs N per strategy, random error band, plateau line, and markers where smart strategies hit it.
- `selection_curve_math.png`: r vs N per strategy, random error band, plateau line, and markers where smart strategies hit it.
- `selection_curve_combined.png`: all benchmarks side by side.

## Data

- `results.csv`: benchmark, strategy, N, seed, se_target, r, mae, items, n_bank_items.

## Caveats

- Small-N 3PL fits (N=10-30 respondents against ~500-1000 items) are
  heavily over-parameterized; item banks shrink as constant items drop.
  Treat the smallest-N points as noisy and read the trend, not single points.
- Smart strategies are deterministic (one curve, no band). Random shows the
  seed spread. Savings factors use linear interpolation of the random mean
  curve to the match threshold.
