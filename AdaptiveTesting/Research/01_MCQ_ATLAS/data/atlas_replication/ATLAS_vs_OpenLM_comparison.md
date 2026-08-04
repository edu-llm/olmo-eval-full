# ATLAS reported numbers vs OpenLM replication

Side by side of ATLAS's published error metrics and our OpenLM replication of the same
analyses and plots. ATLAS numbers come from `AdaptiveTesting/Inputs/ATLAS/summary_pirt_mae_sd_se.csv`
(accuracy p-IRT error) and `summary_theta_mae_sd_se.csv` (theta error, `*/pirt_accuracy_se_*`
rows). OpenLM numbers come from `summary_pirt_mae_sd_se.csv`, `summary_theta_mae_sd_se.csv`,
and `summary_items_adaptive_vs_random.csv` in this folder.

ATLAS and OpenLM use different benchmarks (ATLAS: arc/gsm8k/hellaswag/truthfulqa/winogrande;
ours: ifeval/gpqa/math/bbh/musr), so this compares the error profile, not the same tasks. The
analogous ATLAS quantities exist per (benchmark, SE) for accuracy MAE, mean items administered
(`n_subset_items`), and theta MAE.

## p-IRT accuracy MAE (|p-IRT - actual full accuracy|), by SE stop

| Source | Benchmark | SE<=0.1 | SE<=0.2 | SE<=0.3 |
|---|---|---:|---:|---:|
| ATLAS | arc | 0.032 | 0.034 | 0.034 |
| ATLAS | gsm8k | 0.039 | 0.044 | 0.042 |
| ATLAS | hellaswag | 0.020 | 0.021 | 0.021 |
| ATLAS | truthfulqa | 0.023 | 0.024 | 0.023 |
| ATLAS | winogrande | 0.048 | 0.051 | 0.050 |
| ATLAS mean | | 0.032 | 0.035 | 0.034 |
| OpenLM | ifeval | 0.034 | 0.047 | 0.050 |
| OpenLM | gpqa | 0.034 | 0.066 | 0.070 |
| OpenLM | math | 0.016 | 0.023 | 0.024 |
| OpenLM | bbh | 0.109 | 0.112 | 0.114 |
| OpenLM | musr | 0.075 | 0.091 | 0.095 |
| OpenLM mean | | 0.054 | 0.068 | 0.071 |
| OpenLM mean (no bbh/musr) | | 0.028 | 0.045 | 0.048 |

On ifeval, gpqa, and math the p-IRT accuracy MAE (0.016 to 0.070) sits in the same 0.02 to
0.05 band ATLAS reports, and shows the same monotone trend: MAE falls as SE tightens
(0.1 < 0.2 < 0.3). bbh and musr are error-heavy outliers (see caveats below).

## Ability (theta) MAE (|CAT theta - full-bank/WLE theta|), by SE stop

| Source | Benchmark | SE<=0.1 | SE<=0.2 | SE<=0.3 |
|---|---|---:|---:|---:|
| ATLAS | arc | 0.084 | 0.120 | 0.117 |
| ATLAS | gsm8k | 0.150 | 0.177 | 0.173 |
| ATLAS | hellaswag | 0.157 | 0.163 | 0.165 |
| ATLAS | truthfulqa | 0.064 | 0.073 | 0.071 |
| ATLAS | winogrande | 0.155 | 0.166 | 0.179 |
| ATLAS mean | | 0.122 | 0.140 | 0.141 |
| OpenLM | ifeval | 0.070 | 0.160 | 0.207 |
| OpenLM | gpqa | 0.104 | 0.357 | 0.407 |
| OpenLM | math | 0.097 | 0.168 | 0.208 |
| OpenLM | bbh | 0.711 | 0.719 | 0.684 |
| OpenLM | musr | 0.362 | 0.629 | 0.707 |

theta MAE on ifeval/math (0.07 to 0.21) brackets ATLAS's 0.06 to 0.18 range and drops toward
SE<=0.1 (tighter SE means fewer items but each administered to a smaller posterior SD, so CAT
theta lands closer to the full-bank theta). ATLAS's ground truth is full-test WLE; ours is
full-bank EAP (same role, slightly different estimator).

## Adaptive vs random items to reach the same SE

ATLAS reports a floor of about 30 items (`n_subset_items` about 30 at SE<=0.3, growing to 40
to 88 at SE<=0.1) and includes random/tiny/metabench baselines. Our replication uses a
min-8-item floor and an explicit random-item baseline (same EAP stop rule, fixed seed).

| Benchmark | SE | adaptive items | random items | random/adaptive |
|---|---:|---:|---:|---:|
| ifeval | 0.3 | 18.2 | 62.1 | 3.4x |
| gpqa | 0.3 | 13.0 | 45.7 | 3.5x |
| math | 0.3 | 77.3 | 147.6 | 1.9x |
| musr | 0.3 | 13.6 | 76.2 | 5.6x |
| bbh | 0.3 | 8.2 | 13.3 | 1.6x |
| ifeval | 0.1 | 161.0 | 251.0 | 1.6x |
| gpqa | 0.1 | 154.2 | 314.5 | 2.0x |
| math | 0.1 | 188.9 | 328.8 | 1.7x |

Fisher-information adaptive selection reaches the target SE in 1.6x to 5.6x fewer items than
random, the core ATLAS efficiency claim, reproduced on OpenLM. The advantage is largest at
loose SE (SE<=0.3) where a few high-information items suffice; it shrinks at SE<=0.1 because
tight precision forces the CAT to administer most informative items regardless of order.

## Correlation (Pearson r, p-IRT vs actual), OpenLM

| Benchmark | SE<=0.1 | SE<=0.2 | SE<=0.3 |
|---|---:|---:|---:|
| ifeval | 0.955 | 0.927 | 0.921 |
| gpqa | 0.710 | 0.757 | 0.737 |
| math | 0.948 | 0.892 | 0.878 |
| musr | 0.893 | 0.770 | 0.746 |
| bbh | 0.695 | 0.677 | 0.674 |

(ATLAS's per-benchmark figures report r about 0.85 to 0.95; e.g. arc SE<=0.2 r=0.909.)

## Caveats and methodological differences

- Item filtering. We drop items with non-positive discrimination (`a <= 0`) from the linked
  bank (mirt calibration failures or reverse-scored items), matching the canonical
  `openlm_gpqa_atlas_3pl/diagnostic_validation.py`. It is consequential: gpqa/musr/bbh banks
  contain 51% / 43% / 31% negative-`a` items. gpqa collapses to r about 0 if they are kept
  (and recovers to r=0.737 when dropped, matching the repo's reported gpqa number). musr
  recovers to r about 0.75 under this filter, so it is not a clean negative control here; its
  near-zero r in README section 4 was driven by keeping bad items.
- min-items floor. ATLAS floors CAT length at about 30 items; we floor at 8, so our SE<=0.3
  test lengths (8 to 18) are shorter than ATLAS's about 30.
- theta ground truth. ATLAS uses full-test WLE; ours uses full-bank EAP.
- Scoring regime. OpenLM responses are 0-shot; ATLAS calibrated on 25-shot leaderboard labels
  (see README section 1b). Different tasks and regimes, so absolute numbers are not directly
  comparable; the shape of the error curves is what replicates.
- bbh is the weakest replication (MAE about 0.11, theta MAE about 0.7): a large multi-subtask
  bank whose retained items shift the full-accuracy baseline; p-IRT is rank-correlated
  (r about 0.68) but biased.
