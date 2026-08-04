# FRQ MIRT results (TutorBench, 2-skill)

Evidence for the *MIRT results* and *Limitations* subsections of `Outline.md`. The FRQ
pipeline grades each model's free responses criterion by criterion with the frozen Qwen
judge (see `../02_FRQ_Judge`), builds a model by criterion binary response matrix, applies
the TutorBench Q-matrix, and fits a confirmatory multidimensional 2PL (M2PL). We collapse
content and diagnosis into one "correctness" axis and keep scaffolding as a second axis (a
2-skill model), then run a multidimensional CAT.

## Two calibrations

| Run | Models | Source | Purpose |
|---|---:|---|---|
| 82-model | 82 | original TutorBench calibration | prior definitive fit |
| 115-model | 115 | 82 original plus 33 non-overlapping AWS Qwen-judge models | the outline's "115 models" calibration |

The 33 AWS models are disjoint from the original 82, so the merge is a clean 115-model
union with 0 overlaps. Merged matrix: `data/tb115/response_matrix_115.csv` (115 models by
6,507 criteria, 95.1% filled). Merge script: `../scripts/merge_tb115.py`.

## Why 2 skills

The full 3-skill M2PL fit on the 115 models gives this latent correlation matrix
(`data/tb115/calibration_mirt_manifest.json`):

```
              content  diagnosis  scaffolding
content         1.00      0.94       -0.70
diagnosis       0.94      1.00       -0.68
scaffolding    -0.70     -0.68        1.00
```

Content and diagnosis correlate at 0.94, so collapsing them into one "correctness" axis is
justified (the same held at N=82). The 3-way comparison (unidimensional, collapsed 2-skill,
full 3-skill) selects the collapsed 2-skill model by AIC (ΔAIC vs full = +3,251). Scaffolding
is a distinct axis and, as at N=82, correlates negatively with correctness (about -0.70):
models graded more correct are graded scaffolding less on this bank. Flag this in the
discussion.

## Headline CAT results (2-skill, SE target 0.3)

Full numbers: `data/tb115/cat_eval/cat_metrics.json`, `data/mirt_n82_vs_n115_summary.csv`.

| Metric | N=82 | N=115 |
|---|---:|---:|
| Items fit | 3,443 | 3,669 |
| Observed cells | 276,587 | 415,118 |
| Recovery r, correctness | 0.960 | 0.970 |
| Recovery r, scaffolding | 0.850 | 0.923 |
| Recovery r, overall (stacked) | 0.941 | 0.955 |
| CAT length, overall (mean / median) | 23.6 / 26 | 44.9 / 44 |
| CAT length, correctness (mean / median) | n/a | 38.0 / 35 |
| CAT length, scaffolding (mean / median) | n/a | 21.7 / 18 |
| pIRT MAE (CAT) | 0.0349 | 0.0216 |
| pIRT MAE (full-bank ceiling) | 0.0176 | 0.0165 |
| median discrimination a, correctness / scaffolding | 1.06 / 0.90 | 0.98 / 0.80 |

Figures for the 115-model run carry the `tb115_` prefix in `figures/`:
`tb115_recovery_scatter_correctness.png`, `tb115_recovery_scatter_scaffolding.png`,
`tb115_cat_length_hist.png`, `tb115_se_reduction_curve.png`, `tb115_pirt_calibration.png`,
`tb115_item_info_by_skill.png`. The un-prefixed versions are the 82-model reference.

Adding 33 models improves recovery on both axes, most on scaffolding (0.85 to 0.92, the
weaker axis), and roughly halves CAT prediction error (pIRT MAE 0.035 to 0.022, near the
full-bank ceiling of 0.017). The CAT reaches SE below 0.3 on correctness for 110 of 115
models and on scaffolding for all 115. CAT length grows at N=115 (about 45 vs about 24
items) because the merged bank is harder and more informative and the stopping rule waits
for both axes to cross the target. Scaffolding alone converges in about 18 items.

> Note: `cat_metrics.json -> bank` echoes the 82-run reference constants (`n_items_fit
> 3497 / 6180`, "scaffolding needs refit") hard-coded in
> `eduLLM-Evals/scripts/cat_eval_tutorbench.py`. The 115-model fit actually used 3,669
> items (`data/tb115/calibration_mirt_115_2skill_manifest.json`). The recovery, CAT, and
> pIRT numbers above are computed on the 115 models.

## Limitation: parameter SE vs. N

The outline's headline limitation is that item-parameter uncertainty at N about 115
inflates every ability SE, and more calibration models would shrink it. We measure this
with a matched asymptotic (observed-information) loading-SE estimate computed the same way
at both N (`../scripts/mirt_2skill_bank.py`):

| | N=82 | N=115 | change |
|---|---:|---:|---:|
| median loading SE (all) | 0.411 | 0.337 | -18% |
| median loading SE, correctness | 0.419 | 0.350 | -16% |
| median loading SE, scaffolding | 0.409 | 0.287 | -30% |

Going from 82 to 115 models cuts median loading SE by about 18% (scaffolding by about 30%),
which supports the outline's claim and its remedy of more calibration models. The N=82
bootstrap estimate (`data/param_uncertainty/metrics.json`) reports median item loading SE of
0.41 (correctness) and 0.60 (scaffolding) and a median ability-SE inflation from parameter
uncertainty of about 4 to 5%: small relative to the posterior SE, but not negligible for
scaffolding. The bank-health cross-fold check still flags scaffolding as "needs refit"
(cross-fold discrimination r about 0.48 vs 0.80 for correctness). Scaffolding is the axis
most in need of more data.

Second limitation (unchanged): the judge is imperfect and does not always agree with human
graders (see `../02_FRQ_Judge`). The smallest models also fall into repetitive loops the
judge marks unscorable, and long inputs are capped at 32k tokens.

## Reproduce

```bash
export REPO=/Users/arhant/Documents/EDLM/olmo-eval-full
export PYTHONPATH=$REPO/eduLLM-Evals
D=$REPO/AdaptiveTesting/Research/03_FRQ_MIRT/data/tb115
# 1) merge 82 + 33 -> 115
uv run python $REPO/AdaptiveTesting/Research/scripts/merge_tb115.py $D/response_matrix_115.csv
# 2) 3-way skill-collapse model comparison + latent correlation
uv run python $REPO/eduLLM-Evals/scripts/calibrate_mirt.py --matrix $D/response_matrix_115.csv \
  --rubrics $REPO/eduLLM-Evals/data/TutorBench/curated/rubrics_qmatrix_curated.jsonl \
  --collapse content,diagnosis --estimate-latent-corr --grid 7 --min-persons-identifiable 100 \
  --out-dir $D --write-params --out-rubrics $D/rubrics_qmatrix_115_mirt.jsonl
# 3) definitive 2-skill bank (correctness+scaffolding) + matched loading SE
uv run python $REPO/AdaptiveTesting/Research/scripts/mirt_2skill_bank.py \
  --matrix $D/response_matrix_115.csv --out-bank $D/calibration_mirt_115_2skill.csv \
  --out-manifest $D/calibration_mirt_115_2skill_manifest.json
# 4) CAT evaluation (recovery, CAT length, pIRT MAE, figures)
uv run python $REPO/eduLLM-Evals/scripts/cat_eval_tutorbench.py \
  --bank $D/calibration_mirt_115_2skill.csv --matrix $D/response_matrix_115.csv \
  --no-oos --se-target 0.3 --out-dir $D/cat_eval
```
