# FRQ MIRT results (TutorBench, 2-skill)

Evidence for the *MIRT results* and *Limitations* subsections of `Outline.md`. The FRQ
pipeline grades each model's free responses criterion-by-criterion with the frozen Qwen
judge (see `../02_FRQ_Judge`), builds a model × criterion binary response matrix, applies
the TutorBench Q-matrix, and fits a confirmatory multidimensional 2PL (M2PL). Following the
outline we collapse **content + diagnosis → "correctness"** and keep **scaffolding** as the
second axis (a 2-skill model), then run a multidimensional CAT.

## Two calibrations

| Run | Models | Source | Purpose |
|---|---:|---|---|
| 82-model (existing) | 82 | original TutorBench calibration | the prior definitive fit |
| **115-model (NEW)** | **115** | 82 original **+ 33 non-overlapping** AWS Qwen-judge models | the outline's "115 Models ran" calibration |

The 33 AWS models were selected to be disjoint from the original 82, so the merge is a
clean 115-model union (0 overlaps). Merged matrix: `data/tb115/response_matrix_115.csv`
(115 models × 6,507 criteria, 95.1% filled). Merge script: `../scripts/merge_tb115.py`.

## Model-selection evidence: is a 2-skill model the right choice?

The full 3-skill M2PL fit on the 115 models reports the latent correlation matrix
(`data/tb115/calibration_mirt_manifest.json`):

```
              content  diagnosis  scaffolding
content         1.00      0.94       -0.70
diagnosis       0.94      1.00       -0.68
scaffolding    -0.70     -0.68        1.00
```

- **content ↔ diagnosis = 0.94** → the two are near-collinear, so collapsing them into a
  single "correctness" axis is justified (the same conclusion reached at N=82, where the
  content↔diagnosis correlation was also high).
- The 3-way model comparison (unidimensional vs collapsed-2-skill vs full-3-skill) selects
  the **collapsed 2-skill model by AIC** (ΔAIC vs full = +3,251 in favor of collapsed).
- **correctness ↔ scaffolding ≈ −0.70**: scaffolding is a genuinely distinct axis (and, as
  at N=82, negatively related to correctness — models that are more correct are, on this
  bank, graded as scaffolding *less*, an effect worth flagging in the discussion).

## Headline CAT results (2-skill, SE target 0.3)

Full numbers: `data/tb115/cat_eval/cat_metrics.json`, `data/mirt_n82_vs_n115_summary.csv`.

| Metric | N=82 | **N=115 (new)** |
|---|---:|---:|
| Items fit | 3,443 | **3,669** |
| Observed cells | 276,587 | **415,118** |
| Recovery r — correctness | 0.960 | **0.970** |
| Recovery r — scaffolding | 0.850 | **0.923** |
| Recovery r — overall (stacked) | 0.941 | **0.955** |
| CAT length — overall (mean / median) | 23.6 / 26 | 44.9 / 44 |
| CAT length — correctness (mean / median) | — | 38.0 / 35 |
| CAT length — scaffolding (mean / median) | — | 21.7 / 18 |
| pIRT MAE (CAT) | 0.0349 | **0.0216** |
| pIRT MAE (full-bank ceiling) | 0.0176 | 0.0165 |
| median discrimination a — correctness / scaffolding | 1.06 / 0.90 | 0.98 / 0.80 |

Figures (115-model, prefixed `tb115_`) in `figures/`:
`tb115_recovery_scatter_correctness.png`, `tb115_recovery_scatter_scaffolding.png`,
`tb115_cat_length_hist.png`, `tb115_se_reduction_curve.png`, `tb115_pirt_calibration.png`,
`tb115_item_info_by_skill.png`. The un-prefixed versions are the 82-model reference.

**Reading.** Adding 33 models improves recovery on *both* axes — especially scaffolding
(0.85 → 0.92), the weaker axis — and roughly **halves the CAT prediction error** (pIRT MAE
0.035 → 0.022, essentially at the full-bank ceiling of 0.017). The CAT reaches SE<0.3 on
correctness for 110/115 models and on scaffolding for all 115. The longer CAT length at
N=115 (≈45 vs ≈24 items) reflects a harder/more-informative merged bank and the two-axis
stopping rule (stop only when *both* SEs cross the target); scaffolding alone converges in
~18 items.

> Note: `cat_metrics.json → bank` echoes the 82-run reference constants (`n_items_fit
> 3497 / 6180`, "scaffolding needs refit") that are hard-coded in
> `eduLLM-Evals/scripts/cat_eval_tutorbench.py`. The *actual* 115-model fit used **3,669**
> items (`data/tb115/calibration_mirt_115_2skill_manifest.json`); the recovery / CAT /
> pIRT numbers above ARE computed on the 115 models.

## Limitations, quantified (parameter SE vs. N)

The outline's headline limitation is that item-parameter uncertainty at N≈115 (median
loading SE ≈0.2 was the aspiration) inflates every ability SE, and more calibration models
would shrink it. We quantify this with a **matched** asymptotic (observed-information)
loading-SE estimate computed the same way at both N (`../scripts/mirt_2skill_bank.py`):

| | N=82 | N=115 | change |
|---|---:|---:|---:|
| median loading SE (all) | 0.411 | **0.337** | −18% |
| median loading SE — correctness | 0.419 | 0.350 | −16% |
| median loading SE — scaffolding | 0.409 | 0.287 | −30% |

So going 82 → 115 models already cuts median loading SE by ~18% (scaffolding by ~30%),
confirming the outline's claim and its remedy (more calibration models). For context, the
existing bootstrap-based estimate at N=82
(`data/param_uncertainty/metrics.json`) reports median item loading SE of 0.41
(correctness) / 0.60 (scaffolding) and a median ability-SE inflation from parameter
uncertainty of ~4–5% — small relative to the posterior SE, but not negligible for the
scaffolding axis. The bank-health cross-fold check still flags **scaffolding as "needs
refit"** (cross-fold discrimination r≈0.48 vs 0.80 for correctness): scaffolding is the
axis most in need of more data.

Second limitation (unchanged): the judge is imperfect and does not always agree with human
graders (see `../02_FRQ_Judge`); the smallest models also fall into repetitive loops the
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
