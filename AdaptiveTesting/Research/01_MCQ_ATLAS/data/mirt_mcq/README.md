# MIRT on MCQ (OpenLM): multidimensional vs unidimensional CAT

**Takeaway.** At SE 0.3, MIRT matches-or-beats the unidimensional baseline on accuracy recovery for 4/4 benchmarks (most dramatically on the near-floor MATH dimension, which the unidimensional 2PL barely recovers), by borrowing strength across correlated skills; but it needs MORE total items to satisfy an all-dimensions SE target, because the weakest/slowest dimension gates the single interleaved test.

Pools 4 MCQ benchmarks into one between-item multidimensional bank (each benchmark = a latent dimension; Q-matrix = benchmark membership), calibrates a confirmatory M2PL, runs a multidimensional adaptive test, and compares to separate per-benchmark unidimensional CATs.

## Setup

- Data: OpenLM (status==ok models). Common models across all 4 benches: 1101 (train 991 / test 110, seed 20260802).
- Dimensions / Q-matrix (simple structure): ifeval (150 items), gpqa (150 items), math (77 items), bbh (150 items).
- Item hygiene: a>0 positive-discrimination filter (unidim 2PL) then subsample to n_sub=150 per benchmark.
- Fitter: shared FRQ MIRT EM (`eduLLM-Evals/scripts/calibrate_mirt.py fit_m2pl_em`), confirmatory M2PL, grid=5 nodes/dim (625 total nodes), latent-corr estimated. CAT via `tutor_cat.mirt.update`/`standard_errors`, initialised with the estimated latent correlation R as the prior covariance so the update SHARES information across correlated dimensions (the multidimensional mechanism under test).
- MIRT fit: converged=True, n_iter=79, loglik=-267733.2.

## Comparison table (predicted vs actual per-benchmark accuracy)

`items` semantics differ by model type: for **unidim** it is that benchmark's own CAT length (items administered from that benchmark); for **mirt** it is the POOLED test length at which that dimension first reached the SE target (items of ANY benchmark administered so far in the single interleaved test). The apples-to-apples efficiency number is the total-items comparison in the Efficiency section.

| SE | benchmark | model | r | MAE | items |
|---|---|---|---:|---:|---:|
| 0.3 | ifeval | mirt | 0.985 | 0.0360 | 91.1 |
| 0.3 | ifeval | unidim | 0.962 | 0.0533 | 25.6 |
| 0.3 | gpqa | mirt | 0.881 | 0.0509 | 185.5 |
| 0.3 | gpqa | unidim | 0.874 | 0.0550 | 40.3 |
| 0.3 | math | mirt | 0.997 | 0.0202 | 179.6 |
| 0.3 | math | unidim | 0.903 | 0.1726 | 43.3 |
| 0.3 | bbh | mirt | 0.974 | 0.0337 | 201.6 |
| 0.3 | bbh | unidim | 0.900 | 0.0545 | 32.0 |
| 0.2 | ifeval | mirt | 0.990 | 0.0323 | 327.2 |
| 0.2 | ifeval | unidim | 0.976 | 0.0509 | 70.5 |
| 0.2 | gpqa | mirt | 0.913 | 0.0440 | 400.0 |
| 0.2 | gpqa | unidim | 0.912 | 0.0472 | 83.8 |
| 0.2 | math | mirt | 0.998 | 0.0185 | 227.7 |
| 0.2 | math | unidim | 0.945 | 0.1710 | 54.2 |
| 0.2 | bbh | mirt | 0.983 | 0.0273 | 400.0 |
| 0.2 | bbh | unidim | 0.963 | 0.0353 | 90.0 |

## Latent inter-dimension correlation matrix

| | ifeval | gpqa | math | bbh |
|---|---|---|---|---|
| ifeval | 1.000 | 0.093 | 0.760 | 0.463 |
| gpqa | 0.093 | 1.000 | 0.280 | 0.417 |
| math | 0.760 | 0.280 | 1.000 | 0.608 |
| bbh | 0.463 | 0.417 | 0.608 | 1.000 |

Off-diagonal correlations range 0.09..0.76 (mean 0.44). High (~0.8+) across the board would indicate the skills effectively collapse to one factor; moderate/mixed values indicate genuine multidimensionality.

## Efficiency

- SE 0.3: MIRT total items to reach target on ALL dims = 316.6 (mean/model); sum of 4 separate unidim CATs = 141.2. MIRT does not use fewer total items.
- SE 0.2: MIRT total items to reach target on ALL dims = 400.0 (mean/model); sum of 4 separate unidim CATs = 298.5. MIRT does not use fewer total items.

## Reference: existing published unidimensional 3PL numbers

From `data/atlas_replication/summary_pirt_mae_sd_se.csv` (separate 3PL CATs, full item banks):

| benchmark | SE | r | MAE | items |
|---|---|---:|---:|---:|
| ifeval | 0.3 | 0.921 | 0.0503 | 18.2 |
| ifeval | 0.2 | 0.927 | 0.0465 | 43.8 |
| ifeval | 0.1 | 0.955 | 0.0345 | 161.0 |
| gpqa | 0.3 | 0.737 | 0.0702 | 13.0 |
| gpqa | 0.2 | 0.757 | 0.0661 | 26.1 |
| gpqa | 0.1 | 0.710 | 0.0345 | 154.2 |
| math | 0.3 | 0.878 | 0.0245 | 77.3 |
| math | 0.2 | 0.892 | 0.0227 | 100.9 |
| math | 0.1 | 0.948 | 0.0158 | 188.9 |
| bbh | 0.3 | 0.674 | 0.1145 | 8.2 |
| bbh | 0.2 | 0.677 | 0.1118 | 8.8 |
| bbh | 0.1 | 0.695 | 0.1091 | 15.4 |

> Note: this MIRT experiment subsamples items and uses a 2PL EM fitter, so the re-run unidim numbers above are the apples-to-apples baseline; the published 3PL full-bank numbers use different item counts.
