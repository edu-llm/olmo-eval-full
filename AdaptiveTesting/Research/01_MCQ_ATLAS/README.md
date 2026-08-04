# MCQ / ATLAS recreation

Evidence for the MCQ, ATLAS recreation, feasibility, experiments, and MIRT sections of
`Outline.md`. Every correlation is Pearson r between the CAT (or diagnostic) predicted
benchmark score and the true full-benchmark accuracy on held-out models.

## 1. The published ATLAS bank transfers to our ARC models; a range-restricted refit does not

We run ATLAS-style adaptive testing on ARC two ways and score both on the same 60 held-out
models: the published ATLAS 3PL bank, and the same pipeline refit on only the 0.5 to 7B
model slice.

| Condition | SE stop | Pearson r | MAE | mean CAT items | data | figure |
|---|---:|---:|---:|---:|---|---|
| Published ATLAS 3PL bank | 0.2 | 0.830 | 0.147 | 21.1 | `data/atlas_arc_heldout_se0.2.csv` | `figures/atlas_arc_heldout_se0.2.png` |
| Published ATLAS 3PL bank | 0.3 | 0.738 | 0.172 | 9.9 | `data/atlas_arc_heldout_se0.3.csv` | `figures/atlas_arc_heldout_se0.3.png` |
| Refit on 0.5 to 7B models | 0.2 | 0.589 | 0.147 | 12.5 | `data/atlas_arc_0p5_7b_heldout_se0.2.csv` | `figures/atlas_arc_0p5_7b_heldout_se0.2.png` |
| Refit on 0.5 to 7B models | 0.3 | 0.585 | 0.150 | 8.3 | `data/atlas_arc_0p5_7b_heldout_se0.3.csv` | `figures/atlas_arc_0p5_7b_heldout_se0.3.png` |

The published bank (thousands of models, wide parameter range) recovers held-out ARC
accuracy at r=0.83 using about 21 of 1,172 items at SE<=0.2. Refitting on the 0.5 to 7B
slice (about 1,686 models, 95% of them 7B) drops transfer to r=0.59. A range-matched bank
is cheaper to build but discriminates worse when the calibration pool is small and skewed
toward one size.

### 1b. Prompt-matched responses recover near-ceiling transfer

The numbers above replay our 0-shot loglikelihood responses through a bank ATLAS fit on
25-shot leaderboard labels, a prompt mismatch. Re-running the same 3PL CAT on ATLAS's own
ARC test response matrix
(`AdaptiveTesting/Inputs/ATLAS/data/gaussian_sampled_arc_response_matrix_test.csv`, aligned
to the bank by ATLAS item index) removes the mismatch. Script:
`../scripts/atlas_selfresp_diagnostic.py`; data and figures in `data/atlas_selfresp/`.

| Held-out responses | SE stop | r | MAE | mean CAT items |
|---|---:|---:|---:|---:|
| Our 0-shot responses | 0.2 | 0.830 | 0.147 | 21.1 |
| Our 0-shot responses | 0.3 | 0.738 | 0.172 | 9.9 |
| ATLAS's own responses | 0.2 | 0.915 | 0.031 | 13.0 |
| ATLAS's own responses | 0.3 | 0.906 | 0.034 | 9.2 |

With prompt-matched responses the published bank reaches r=0.91 in under 10 items and MAE
drops about 5x. Most of the earlier degradation was scoring-protocol mismatch, not IRT
transfer error. (The 60 held-out models here are a seed-7 sample from ATLAS's 417-model ARC
test set, which spans a wider parameter range than our 0.2 to 7B pool; 650 bank items align
to the test matrix.)

## 2. On a thin in-house pool, 2PL beats 3PL on accuracy and test length

We calibrate our own bank on about 50 in-house models
(`AdaptiveTesting/Inputs/Open/LLM-Judge/mcq`), ARC-Challenge.

| Fitter | SE stop | r | MAE | mean CAT items | data |
|---|---:|---:|---:|---:|---|
| 2PL (girth MML) | 0.15 | 0.954 | 0.065 | 149.8 | `data/diag_validation_arc_challenge.csv` |
| 2PL (girth MML) | 0.3 | 0.908 | 0.062 | 33.8 | `data/diag_validation_arc_challenge_se0.3.csv` |
| 3PL (py-irt Bayesian) | 0.3 | 0.731 | 0.086 | 492.8 | `data/diag_validation_arc_challenge_3pl_se0.3.csv` |

The 3PL's guessing parameter is poorly identified with few persons, so its CAT rarely
reaches the SE target and administers about 14x more items for a lower correlation. ATLAS
had enough models to fit a stable 3PL; with a small calibration fleet, prefer 2PL. Figures:
`figures/diag_validation_arc_challenge*.png`.

### 2b. The discrimination parameter buys shorter tests, and needs regularization on thin pools

To measure what the discrimination parameter `a` adds, we run 1PL (Rasch, `a` fixed at 1)
and 2PL under the same girth fitter, split (50 train / 13 test, seed 7), and CAT. Script:
`../scripts/mcq_diagnostic.py --method {rasch,girth}`; data in `data/local_1pl_vs_2pl/`.

| Model (girth) | SE stop | r | MAE | mean CAT items | % of bank |
|---|---:|---:|---:|---:|---:|
| 1PL (Rasch) | 0.3 | 0.976 | 0.050 | 43.4 | 5.7% |
| 2PL | 0.3 | 0.862 | 0.083 | 8.0 | 1.1% |
| 1PL (Rasch) | 0.15 | 0.987 | 0.060 | 182.0 | 24.0% |
| 2PL | 0.15 | 0.920 | 0.074 | 32.9 | 4.3% |
| 2PL (py-irt, regularized) | 0.3 | 0.908 | 0.062 | 33.8 | |
| 3PL (py-irt) | 0.3 | 0.731 | 0.086 | 492.8 | |

The discrimination parameter makes the CAT about 5x shorter at a fixed SE target (8 vs 43
items at SE<=0.3; 33 vs 182 at SE<=0.15), because high-`a` items concentrate information and
the posterior SE collapses fast. But on this 63-model pool the raw MML discriminations are
noisy, so the unregularized 2PL's accuracy recovery drops (r 0.976 to 0.862 at SE<=0.3). The
parameter-free Rasch model has nothing to overfit and predicts best, at the cost of much
longer tests. The regularized (Bayesian py-irt) 2PL is the best of both: short like a 2PL
(about 34 items) and accurate like the Rasch (r=0.908), because its priors tame the
thin-sample discriminations. The parameter pays off only with enough calibration models, or
enough regularization, to estimate `a` stably.

## 3. Calibration correlation stabilizes at about 25 to 40 models

`data/trainsize_corr.csv` (and `_ext`, which adds the n=60 point) sweeps the number of
calibration models on ARC-Challenge with a fixed 13-model held-out set, SE<=0.3.

- Correlation is noisy below about 15 models (r swings 0.45 to 0.92), then stabilizes in the
  0.85 to 0.94 band from about 25 models up. Peak in this run: n=25 gives r=0.938 (about 4.7
  CAT items); n=50 gives r=0.898; n=60 gives r=0.846.
- CAT length stays about 4 to 5 items regardless of train size. Train size buys calibration
  stability, not shorter tests.

The multi-benchmark version (`data/bigrun_summary.csv`, 5 shuffles x 8 sizes x 3 benchmarks;
`figures/bigrun_corr.png`) confirms this. Mean r at n=60: arc_challenge 0.897, sciq 0.845,
openbookqa 0.806. Figures: `figures/trainsize_corr.png`, `figures/corr_three_benchmarks.png`.

About 25 to 40 diverse calibration models hold correlation near 0.85 to 0.9 on these MCQ
benchmarks; below about 15 models the correlation collapses.

### 3b. Correlation plateaus by a few hundred models, even at 1,000-model scale

Section 3 tops out at 50 to 60 in-house models. Here we ask whether correlation keeps
rising as we add models, using the OpenLM per-question dumps (about 1,102 models per
benchmark) and the full ATLAS 3PL calibrator. Script: `../scripts/openlm_trainsize_sweep.py`;
data and figures in `data/openlm_trainsize/`.

Setup. The fitter is the ATLAS methodology used elsewhere in this repo: a k-subset 3PL fit
in R `mirt` (100-item chunks, EM up to 500 cycles) followed by mean-sigma chunk linking with
polarity flip, matching `Experiments/openlm_atlas_3pl/run_3pl_diagnostic.py`. The bank keeps
only positive-discrimination items (`a>0`), the convention validated in
`Experiments/openlm_gpqa_atlas_3pl/diagnostic_validation.py`; items that link to `a<=0` have
reversed response curves and poison the Fisher-information CAT if kept. The held-out set is a
fixed seed-7 10% split (110 held-out; math 90), never used in calibration. Train subsets are
nested: the first `n_train` models of one fixed seed-7 permutation, one calibration per step.
Sweep: `n_train` = 100, 200, up to `total - n_test` (step 100; bbh coarsened to 200 for its
about 3,900-item bank). CAT is the ATLAS p-IRT Fisher-information selector; we report SE<=0.3
(primary) and SE<=0.2 (in the CSVs).

| Benchmark | n_test | bank items | r @ n_train=100 | r @ max n_train | plateau | mean CAT items (SE<=0.3) |
|---|---:|---:|---:|---:|---|---:|
| ifeval | 110 | ~511 | 0.908 | 0.923 (n=992) | flat from ~100, 0.90 to 0.95 | ~20 to 24 |
| math | 90 | ~1,087 | 0.889 | 0.880 (n=811) | flat from ~100, 0.86 to 0.93 | ~130 to 180 |
| gpqa | 110 | ~825 | 0.604 | 0.592 (n=992) | noisy, no trend, ~0.60 to 0.71 | ~15 to 23 |
| bbh | 110 | ~3,900 | 0.845 (n=200) | 0.671 (n=992) | peaks ~n=600 (0.86) then declines | ~8 |

(Full per-step numbers including SE<=0.2, median items, %-of-bank, MAE:
`data/openlm_trainsize/<bench>_openlm_trainsize.csv` and the concatenated
`data/openlm_trainsize/openlm_trainsize_summary.csv`.)

- ifeval and math confirm section 3 at 20x the models: correlation saturates by 100 to 200
  models (ifeval about 0.92, math about 0.88) and stays flat to n about 1,000. The full fleet
  adds nothing once you have a couple hundred diverse models.
- Test length is decoupled from n_train. Mean CAT length barely moves across the sweep (bbh
  about 8, gpqa about 15 to 23, ifeval about 20, math about 130 to 180 at SE<=0.3). math's
  long CATs reflect a bank of only positively-discriminating items; the shorter counts in
  section 4 came partly from retained reversed-curve items that falsely collapsed the SE.
- bbh is a cautionary case: correlation peaks near n=600 (r about 0.86) then falls to 0.67 at
  the full pool. A single-factor 3PL mis-fits BBH's multi-subtask mixture more as the pool
  grows and spans a wider ability range, so more data makes the unidimensional bank worse.
  gpqa stays weak and noisy (r about 0.6 to 0.7, MAE about 0.2) throughout: near-chance
  responses give IRT little signal, though `a>0` filtering rescues it from the r about 0
  collapse seen when reversed-curve items are kept.

Figures: per-benchmark dual-axis (correlation and mean items vs n_train)
`data/openlm_trainsize/<bench>_openlm_trainsize.png`; combined correlation vs n_train
`data/openlm_trainsize/openlm_trainsize_corr_combined.png`. musr is omitted (known linking
failure, section 4, r about -0.04).

On OpenLM, ATLAS-style CAT reaches its accuracy ceiling with a few hundred calibration
models; scaling to about 1,000 adds stability but not correlation, and for a
multi-dimensional benchmark (bbh) more models can hurt a unidimensional 3PL. Benchmark
dimensionality, not calibration-pool size, is the ceiling on transfer quality here.

## 4. ATLAS-style 3PL on newer OpenLM benchmarks

Using the OpenLM per-question dumps (0.2 to 7B models) we build ATLAS-format matrices and
run the published 3PL CAT pipeline on benchmarks ATLAS never covered
(`data/openlm_atlas3pl_results_summary.csv`, SE<=0.3, about 110 held-out models).

| Benchmark | r | MAE | mean CAT items | figure |
|---|---:|---:|---:|---|
| ifeval | 0.921 | 0.048 | 9.2 | `figures/ifeval_atlas3pl_se0.3.png` |
| math | 0.878 | 0.024 | 43.8 | `figures/math_atlas3pl_se0.3.png` |
| bbh | 0.753 | 0.035 | 8.1 | `figures/bbh_atlas3pl_se0.3.png` |
| gpqa | 0.737 | 0.070 | 13.0 | `figures/openlm_gpqa_atlas3pl_se0.3.png` |
| musr | -0.036 | 0.109 | 8.0 | `figures/musr_atlas3pl_se0.3.png` |

GPQA also has an explicit 2PL vs 3PL comparison (`data/compare_2pl_vs_3pl_se0.3.csv`,
`figures/compare_2pl_vs_3pl_se0.3.png`): with the large OpenLM pool the 3PL wins (r=0.74 vs
about 0.10 for 2PL), the mirror image of finding 2, so the crossover is driven by
calibration-pool size. The musr r near zero here turns out to be a bad-item artifact, not a
benchmark property (see section 4b and `data/link_failure_diagnosis/`).

## 4b. Replicating ATLAS's error analyses on OpenLM

Section 4 shows one p-IRT-vs-actual scatter per benchmark at SE<=0.3. Here we reproduce
ATLAS's full error-analysis figure set on OpenLM at ATLAS's own SE thresholds (0.1 / 0.2 /
0.3), using the same frozen 3PL banks and Fisher-information p-IRT CAT (single calibration,
fixed 90/10 split, seed 7, no shuffles). Script: `../scripts/atlas_error_plots_openlm.py`.
Outputs: `data/atlas_replication/` (per-benchmark CSVs and figures, cross-benchmark
summaries, and `ATLAS_vs_OpenLM_comparison.md`); key PNGs copied to `figures/atlasrep_*`.

ATLAS's error reporting is one 2x2 panel figure per (benchmark, SE),
`Inputs/ATLAS/<bench>/pirt_vs_actual_se_<SE>.png`, produced by
`Inputs/ATLAS/scripts/analysis/compare_pirt_actual.r`, plus cross-benchmark error-vs-SE
curves in `Inputs/ATLAS/summary_pirt_mae_sd_se.csv`, `summary_theta_mae_sd_se.csv`, and
`rmse_cat_summary.csv`. The four error views are:

1. p-IRT predicted vs actual accuracy scatter (y=x line, linear fit, Pearson r, RMSE),
   top-left. Cross-benchmark this becomes accuracy MAE vs SE stopping threshold.
2. Error distribution histogram of `error = p-IRT - actual` (mean, MAE), top-right.
3. Absolute error vs number of items administered, bottom-left. Cross-benchmark this is mean
   items vs SE threshold (ATLAS adaptive vs a random-item baseline).
4. Error vs actual accuracy (loess), bottom-right. ATLAS also tracks theta (ability) MAE vs
   the WLE full-test theta by SE.

Per-benchmark 2x2 error figures: `figures/atlasrep_<bench>_pirt_vs_actual_se_<SE>.png` (and
`data/atlas_replication/<bench>/`). Cross-benchmark curves:

![accuracy MAE vs SE](figures/atlasrep_mae_vs_se.png)

![items adaptive vs random vs SE](figures/atlasrep_items_vs_se.png)

![theta MAE vs SE](figures/atlasrep_theta_mae_vs_se.png)

| Benchmark | r (SE 0.1/0.2/0.3) | acc-MAE (0.1/0.2/0.3) | theta-MAE (0.1/0.2/0.3) | adaptive vs random items @0.3 |
|---|---|---|---|---|
| ifeval | 0.955 / 0.927 / 0.921 | 0.034 / 0.047 / 0.050 | 0.070 / 0.160 / 0.207 | 18 vs 62 (3.4x) |
| gpqa | 0.710 / 0.757 / 0.737 | 0.034 / 0.066 / 0.070 | 0.104 / 0.357 / 0.407 | 13 vs 46 (3.5x) |
| math | 0.948 / 0.892 / 0.878 | 0.016 / 0.023 / 0.024 | 0.097 / 0.168 / 0.208 | 77 vs 148 (1.9x) |
| musr | 0.893 / 0.770 / 0.746 | 0.075 / 0.091 / 0.095 | 0.362 / 0.629 / 0.707 | 14 vs 76 (5.6x) |
| bbh | 0.695 / 0.677 / 0.674 | 0.109 / 0.112 / 0.114 | 0.711 / 0.719 / 0.684 | 8 vs 13 (1.6x) |

Full side-by-side against ATLAS's reported numbers (arc/gsm8k/hellaswag/truthfulqa/
winogrande) is in `data/atlas_replication/ATLAS_vs_OpenLM_comparison.md`.

- Scatter and accuracy MAE vs SE: on ifeval/gpqa/math the p-IRT accuracy MAE (0.016 to
  0.070) lands in ATLAS's own 0.02 to 0.05 band and falls monotonically as SE tightens
  (0.1 < 0.2 < 0.3), matching ATLAS. gpqa reproduces the repo's r=0.737 at SE<=0.3.
- Items, adaptive vs random: Fisher-information selection reaches the same SE in 1.6x to 5.6x
  fewer items than random, the core ATLAS efficiency result. The gap is widest at loose SE
  and narrows at SE<=0.1.
- theta MAE vs SE: ability recovery on ifeval/math (0.07 to 0.21) brackets ATLAS's 0.06 to
  0.18 and improves toward SE<=0.1.
- Item filtering matters (the key divergence from section 4). We drop non-positive
  discrimination items (`a<=0`, mirt failures), matching gpqa's canonical loader. gpqa/musr/
  bbh banks are 51% / 43% / 31% such items; keeping them collapses gpqa to r about 0. Under
  the filter musr recovers to r about 0.75, so it is not the clean negative control section 4
  described; its earlier near-zero r came from retaining bad items. bbh stays the weakest
  (r about 0.68, MAE about 0.11): rank-correlated but biased on its large multi-subtask bank.

## 5. Feasibility on benchmarks with no public response data (Pedagogy, PIQA, SocialIQA)

The outline calls for calibrating Pedagogy, a skill-specific benchmark with no public
response matrix, so we generated the responses ourselves (52 models, full200 run) and fit a
2PL plus CAT for the first time. Script: `../scripts/mcq_diagnostic.py`. Summary:
`data/pedagogy_feasibility/mcq_diagnostic_summary.csv`.

| Benchmark | items | SE stop | r | MAE | mean CAT items | % of bank | figure |
|---|---:|---:|---:|---:|---:|---:|---|
| Pedagogy | 920 | 0.3 | 0.716 | 0.063 | 10.2 | 3.2% | `figures/diag_pedagogy_se0.3.png` |
| Pedagogy | 920 | 0.15 | 0.900 | 0.080 | 275.5 | 86.6% | `figures/diag_pedagogy_se0.15.png` |
| PIQA | 1,838 | 0.3 | 0.937 | 0.040 | 10.6 | 1.3% | `figures/diag_piqa_se0.3.png` |
| PIQA | 1,838 | 0.15 | 0.954 | 0.038 | 194.1 | 23.4% | `figures/diag_piqa_se0.15.png` |
| SocialIQA | 1,954 | 0.3 | 0.271 | 0.093 | 8.0 | 0.8% | `figures/diag_socialiqa_se0.3.png` |
| SocialIQA | 1,954 | 0.15 | 0.430 | 0.080 | 14.3 | 1.5% | `figures/diag_socialiqa_se0.15.png` |

(Source of truth: `data/pedagogy_feasibility/mcq_diagnostic_summary.csv`.)

A benchmark with no public response data (Pedagogy) becomes a working CAT once you generate a
modest in-house response matrix: at SE<=0.3 we recover full accuracy at r=0.72 (Pedagogy) and
r=0.94 (PIQA) using 1 to 3% of the items. Two caveats:

- Tight SE needs a fat bank. At SE<=0.15 a thin 2PL-only bank cannot reach precision
  efficiently; it administers most of the bank (about 87% for Pedagogy). This is the regime
  where ATLAS's large 3PL pools pay off, and it argues for more calibration models or a
  looser SE target for checkpoint-time diagnostics.
- Not every benchmark discriminates. SocialIQA hits SE<=0.3 in the minimum 8 items but
  recovers true accuracy at only r=0.27, because the 52 in-house models are tightly clustered
  in socialiqa ability (little variance for IRT to use). A benchmark must spread the target
  model population for CAT to beat a fixed short test.

Two follow-ups on Pedagogy live under `data/pedagogy_feasibility/`: a 1PL vs 2PL comparison
(`rasch_1pl/`) and a linear recalibration that fixes the predicted-to-actual offset
(`linear_calibration/`, see its README).

## 6. Effect of the SE stopping requirement on test length and correlation

A dedicated sweep of the CAT stopping SE. For each SE target we record the mean items
administered and the Pearson r of predicted vs actual full accuracy on held-out models.
Because the CAT administers items in the same order regardless of the stop threshold, we run
one full CAT per model and read off (items, prediction) at each SE. Script:
`../scripts/se_sweep.py`. Data: `data/se_sweep/` (per-config CSVs and `se_sweep_combined.csv`).

Correlation vs SE:

![corr vs se](figures/se_sweep_corr_vs_se.png)

Test length vs SE:

![items vs se](figures/se_sweep_items_vs_se.png)

| Config | SE<=0.5 | SE<=0.3 | SE<=0.2 | SE<=0.15 | SE<=0.12 |
|---|---|---|---|---|---|
| ATLAS 3PL (own responses) | r=0.905 / 8 it | r=0.906 / 9 it | r=0.915 / 13 it | r=0.925 / 32 it | r=0.931 / 47 it |
| local ARC 2PL (girth) | r=0.862 / 8 it | r=0.862 / 8 it | r=0.842 / 15 it | r=0.920 / 33 it | r=0.965 / 71 it |
| local ARC 1PL (Rasch) | r=0.729 / 14 it | r=0.976 / 43 it | r=0.977 / 100 it | r=0.987 / 182 it | r=0.985 / 304 it |

- Test length grows super-linearly as SE tightens for every model form. Going from SE<=0.3 to
  SE<=0.12 multiplies test length about 5x (ATLAS), 9x (2PL), 7x (1PL).
- A well-calibrated bank plateaus early. The ATLAS 3PL bank already sits at r about 0.90 with
  about 8 items at SE<=0.5; tightening SE buys only +0.03 r for 6x the items. You can run a
  very short diagnostic when the bank is good.
- A thin bank needs a tighter SE (more items) to pay off. The local 1PL climbs from r=0.73
  (SE<=0.5, 14 items) to about 0.98 by SE<=0.30 to 0.35 (about 31 to 43 items), then
  flattens; the knee is around SE<=0.3. The local 2PL is non-monotone (noisy thin-sample
  discriminations) and only overtakes at very tight SE.
- Practical guidance: for a trusted large-pool bank, SE<=0.3 to 0.4 (about 8 to 10 items) is
  plenty; for a thin in-house bank, budget SE<=0.3 (about 30 to 45 items) to reach the
  correlation plateau.

## Reproduce

```bash
export REPO=/Users/arhant/Documents/EDLM/olmo-eval-full
export PYTHONPATH=$REPO/eduLLM-Evals
# SE-requirement sweep (correlation and #items vs SE):
uv run python $REPO/AdaptiveTesting/Research/scripts/se_sweep.py --source atlas \
  --n-test 60 --tag atlas3pl_own_resp --out-dir <outdir>
uv run python $REPO/AdaptiveTesting/Research/scripts/se_sweep.py --source local \
  --mcq-dir $REPO/AdaptiveTesting/Inputs/Open/LLM-Judge/mcq --bench arc_challenge \
  --method girth --tag local_arc_2pl --out-dir <outdir>
# Feasibility diagnostics (Pedagogy/PIQA/SocialIQA):
uv run python $REPO/AdaptiveTesting/Research/scripts/mcq_diagnostic.py \
  --mcq-dir $REPO/AdaptiveTesting/Outputs/full200_results/Outputs/mcq \
  --bench pedagogy --se-stop 0.3 --out-dir <outdir>
# ATLAS transfer / recalibration / local 2PL-3PL / sweeps: see the scripts under
# AdaptiveTesting/Outputs/ (atlas_diagnostic_validation.py, diagnostic_validation.py,
# cat_trainsize_sweep.py, bigrun_3bench.py) and AdaptiveTesting/Experiments/openlm_atlas_3pl/.
```
