# Within-benchmark MIRT on bbh: internal multidimensionality

**Verdict.** bbh subtasks are genuinely multidimensional (mean inter-subtask ability correlation 0.26); the within-bbh MIRT CAT recovers overall accuracy better than the unidimensional CAT (r 0.973 vs 0.845 at SE 0.3).

Fits MIRT WITHIN a single benchmark (subtasks as latent dimensions) to test whether that benchmark is internally multidimensional and whether modeling it recovers the true benchmark accuracy better than a unidimensional CAT. This is distinct from the earlier between-benchmark pooling (mirt_mcq_test.py), which only showed that different benchmarks measure different things.

## Setup

- Data: OpenLM `bbh` (status==ok models). Complete matrix after coverage filter: 1101 models x 3759 items (train 991 / test 110, seed 7, 10% held out).
- Subtasks (latent dimensions): 24 with >= 3 usable items each.
- Item hygiene: per-subtask oriented unidimensional 2PL, keep a>0 items (pass-rate window [0.05, 0.95] on train).
- Fitter: shared FRQ MIRT EM (`eduLLM-Evals/scripts/calibrate_mirt.py fit_m2pl_em`), Bock-Aitkin marginal-ML. CAT via `tutor_cat.mirt.update` / `standard_errors`.

### Missing-data / intractability fallback (read this)

A joint confirmatory M2PL with all 24 subtasks as dimensions is computationally intractable with the shared Gauss-Hermite EM: the fixed grid has `grid**n_dims` nodes (grid**24). We therefore estimate the confirmatory simple-structure latent correlation via PER-SUBTASK unidimensional 2PL calibration followed by the EAP-ability correlation across subtasks (raw and reliability-disattenuated), and assemble the simple-structure MIRT CAT bank from those per-subtask item parameters (each item loads only on its subtask dimension), with the estimated latent correlation R as the CAT prior covariance. Per-item subtask labels ARE present in the OpenLM dump, so no exploratory-only fallback was needed for the Q-matrix.

## 1. Exploratory dimensionality: how many factors does bbh need?

Exploratory M2PL (all-ones Q, k free loadings/item) on a balanced a>0 subsample (500 persons x 288 items, grid=4 nodes/dim, param count rotation-corrected).

| factors k | loglik | n_params | AIC | BIC | converged |
|---:|---:|---:|---:|---:|---|
| 1 | -74168.2 | 576 | 149488.4 | 155177.9 | True |
| 2 | -67109.0 | 863 | 135944.0 | 144468.4 | True |
| 3 | -64096.7 | 1149 | 130491.3 | 141840.6 | True |
| 4 | -60887.6 | 1434 | 124643.1 | 138807.5 | True |
| 5 | -58547.5 | 1718 | 120530.9 | 137500.6 | False |
| 6 | -56672.7 | 2001 | 117347.4 | 137112.4 | False |

AIC is minimised at k=6, BIC at k=6. Both criteria are still decreasing at the largest k tested (k=6), so the effective dimensionality is at LEAST 6; it did not plateau within the tested range, consistent with a suite of ~24 heterogeneous subtasks. The inter-item correlation matrix has a dominant first eigenvalue with a long tail (top eigenvalues: 41.2, 26.4, 13.2, 12.4, 9.3, 8.3, 7.7, 6.3); 53 exceed 1, though the Kaiser>1 count over-states factor count for binary items and is shown only as a scree. The unidimensional 1-factor model is decisively the worst fit (highest AIC/BIC), so the suite is multidimensional.

## 2. Subtask correlation structure: one factor or many?

Latent inter-subtask ability correlation across 24 subtasks (per-subtask EAP theta on the full model set).

- Raw off-diagonal correlation: mean 0.26, range -0.63 to 0.89.
- Reliability-disattenuated off-diagonal: mean 0.27, range -0.65 to 0.92.
- Interpretation: the mean raw correlation 0.26 is well below the 0.8 one-factor threshold, so subtasks are genuinely multidimensional.

## 3. Within-BBH MIRT vs unidimensional CAT (recovering overall accuracy)

Both CATs predict the OVERALL benchmark accuracy on the SAME held-out models. `items` = mean number administered before the SE stopping rule. The unidimensional CAT stops when a single theta's SE < target (documented early stop); the MIRT CAT stops when the slowest subtask dimension's SE < target, so it cannot early-stop on the whole heterogeneous suite.

| SE | model | r | MAE | mean items | median items | bank items |
|---|---|---:|---:|---:|---:|---:|
| 0.3 | unidim | 0.845 | 0.0543 | 13.4 | 8 | 3207 |
| 0.3 | mirt_within | 0.973 | 0.0230 | 500.0 | 500 | 3759 |
| 0.2 | unidim | 0.855 | 0.0545 | 37.3 | 23 | 3207 |
| 0.2 | mirt_within | 0.973 | 0.0230 | 500.0 | 500 | 3759 |

### Matched-item-budget control (isolates dimensionality from item count)

Both CATs forced to the SAME number of items (no early stop), so any gap is due to modeling multidimensionality, not to administering more items.

| # items | unidim r | MIRT r | unidim MAE | MIRT MAE |
|---:|---:|---:|---:|---:|
| 25 | 0.855 | 0.890 | 0.0532 | 0.0513 |
| 50 | 0.861 | 0.924 | 0.0525 | 0.0396 |
| 100 | 0.876 | 0.944 | 0.0526 | 0.0320 |
| 200 | 0.892 | 0.960 | 0.0508 | 0.0279 |
| 400 | 0.915 | 0.969 | 0.0485 | 0.0244 |

## Reference: published unidimensional 3PL numbers

From `data/atlas_replication/summary_pirt_mae_sd_se.csv` (full-bank 3PL R-mirt CAT):

| SE | r | MAE | items |
|---|---:|---:|---:|
| 0.3 | 0.674 | 0.1145 | 8 |
| 0.2 | 0.677 | 0.1118 | 9 |
| 0.1 | 0.695 | 0.1091 | 15 |

> The unidim CAT here is a 2PL EM re-run on the SAME items/split as the MIRT bank (apples-to-apples); the published 3PL full-bank numbers use different item counts and a 3PL fitter, echoed for reference.

## Files

- `bbh_dimensionality.csv` -- factors, loglik, AIC, BIC, eigenvalues.
- `bbh_subtask_correlations.csv` / `_disattenuated.csv` -- inter-subtask ability correlations.
- `bbh_cat_comparison.csv` -- unidim vs within-BBH MIRT (r, MAE, items) at each SE target.
- `bbh_cat_budget.csv` -- matched-item-budget r/MAE for both models.
- figures/: `bbh_dimensionality.png`, `bbh_subtask_correlation_heatmap.png`, `bbh_cat_recovery.png`, `bbh_matched_budget.png`.

## Per-skill recovery

Does the single-ability CAT that recovers OVERALL BBH accuracy also recover each SUBTASK's accuracy? For every held-out model we compute, per subtask, the actual accuracy on that subtask's full item set and the predicted accuracy from (a) the unidimensional CAT's single theta and (b) the within-BBH MIRT CAT's per-subtask dimension theta, then correlate predicted vs actual across the 110 test models. Primary read-out uses full-information theta (all items); a matched 100-item budget is the secondary read-out.

**Mean per-subtask recovery r (full information): unidimensional 0.578 vs within-BBH MIRT 0.945** (mean gain +0.367). At the matched 100-item budget: unidim 0.524 vs MIRT 0.811. A single latent ability recovers overall accuracy (full-info overall r: unidim 0.966, MIRT 0.989) yet cannot separate per-skill performance, so its per-subtask recovery collapses; the MIRT per-subtask dimensions hold up across skills.

Skills where the unidimensional CAT drops most (lowest unidim r) are also where MIRT gains most (delta r), because MIRT sits near the per-skill ceiling everywhere:

| subtask | n items | actual acc | unidim r | MIRT r | delta r |
|---|---:|---:|---:|---:|---:|
| causal_judgement | 97 | 0.571 | -0.014 | 0.999 | +1.013 |
| web_of_lies | 128 | 0.463 | 0.100 | 0.999 | +0.899 |
| formal_fallacies | 133 | 0.430 | 0.153 | 0.998 | +0.845 |
| snarks | 96 | 0.496 | 0.265 | 0.998 | +0.733 |
| hyperbaton | 129 | 0.704 | 0.338 | 1.000 | +0.661 |

Skills where a single ability already suffices (highest unidim r, smallest MIRT gain):

| subtask | unidim r | MIRT r | delta r |
|---|---:|---:|---:|
| reasoning_about_colored_objects | 0.951 | 0.976 | +0.026 |
| logical_deduction_seven_objects | 0.889 | 0.965 | +0.076 |
| penguins_in_a_table | 0.884 | 0.968 | +0.085 |
| logical_deduction_five_objects | 0.872 | 0.904 | +0.032 |
| date_understanding | 0.847 | 0.970 | +0.123 |

Overall-consistency check: the item-count-weighted average of the per-subtask predictions is by construction the overall predicted accuracy, and at matched budgets it reproduces the existing overall recovery numbers exactly:

| budget | unidim r (here / existing) | MIRT r (here / existing) |
|---:|---|---|
| 100 | 0.876 / 0.876 | 0.944 / 0.944 |
| 400 | 0.915 / 0.915 | 0.969 / 0.969 |

Files: `bbh_per_skill_recovery.csv` (per-subtask r/MAE, sorted by unidim r ascending), `figures/bbh_per_skill_recovery.png`.
