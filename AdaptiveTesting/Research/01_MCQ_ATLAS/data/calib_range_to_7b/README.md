# Calibration parameter range vs 7B-model recovery

Fixed TEST set = 7B models (params_b >= 6.5). Two calibration pools, matched N, same test set. Config A calibrates on [0,5)B (upward extrapolation); Config B on [3,7]B with the >=6.5 test models removed so its band is [3,6.5)B (in-range / interpolation). p-IRT + Fisher CAT recover full-benchmark accuracy.

Matched N equals pool B's size in every benchmark, so B uses its entire pool each seed
(its r has near-zero sd); only the larger pool A is subsampled to N across the 5 seeds.

## Pool sizes (per benchmark, models present in cleaned matrix)

| benchmark | 7B test (>=6.5) | pool A [0,5) | pool B [3,6.5) | matched N |
|---|---|---|---|---|
| bbh | 59 | 1012 | 367 | 367 |
| gpqa | 59 | 1012 | 367 | 367 |
| ifeval | 59 | 1012 | 367 | 367 |
| math | 57 | 817 | 324 | 324 |

## Matched-N Pearson r (SE<=0.2)

| benchmark | A r (mean+/-sd) | B r (mean+/-sd) | B - A | A MAE | B MAE |
|---|---|---|---|---|---|
| bbh | 0.702+/-0.117 | 0.515+/-0.007 | -0.187 | 0.096 | 0.106 |
| gpqa | 0.633+/-0.147 | 0.623+/-0.000 | -0.010 | 0.208 | 0.185 |
| ifeval | 0.921+/-0.010 | 0.876+/-0.000 | -0.045 | 0.036 | 0.047 |
| math | 0.959+/-0.027 | 0.880+/-0.000 | -0.079 | 0.007 | 0.009 |

## Matched-N Pearson r (SE<=0.3)

| benchmark | A r (mean+/-sd) | B r (mean+/-sd) | B - A | A MAE | B MAE |
|---|---|---|---|---|---|
| bbh | 0.697+/-0.120 | 0.512+/-0.007 | -0.185 | 0.094 | 0.105 |
| gpqa | 0.632+/-0.152 | 0.594+/-0.000 | -0.038 | 0.244 | 0.196 |
| ifeval | 0.907+/-0.009 | 0.871+/-0.000 | -0.036 | 0.042 | 0.049 |
| math | 0.961+/-0.020 | 0.831+/-0.000 | -0.130 | 0.008 | 0.012 |

## Answer

Calibrating on 3-7B vs 0-5B, tested on 7B models: at SE<=0.3, in-range calibration (B) gives Pearson r that is on average -0.083 (lower) than upward extrapolation (A) across ifeval and math (per-benchmark B-A: -0.036, -0.130). See table above for exact per-benchmark values.
