# Calibration — K-Fold Cross-Validation (2-skill M2PL)

**Status:** complete (read-only study; nothing written back to the bank).
**Date:** 2026-07-30
**Harness:** `scripts/kfold_cv_mirt.py` (+ optional `scripts/kfold_cat_pilot.py`)
**Outputs:** `staging/kfold/` (k=5) and `staging/kfold/k10/` (k=10)

---

## 1. Motivation — the circularity this addresses

The frozen 2-skill M2PL item bank was calibrated on the **same 82 TutorBench models**
that the adaptive test (CAT) then scores. That is circular: the CAT test-takers **are**
the calibration crowd, so in-sample fit statistics (loglik / AIC / BIC from Run 1) say
nothing about how the calibration **generalises to an unseen model**. AIC/BIC reward fit
to *these* 82 persons; they do not measure held-out prediction.

This study answers the honest question directly with **person-level k-fold
cross-validation**:

> Fit the 2-skill M2PL on a TRAIN subset of models, freeze the item params, then score
> the HELD-OUT models from their own responses (EAP) and measure how well the frozen
> items predict the held-out cells.

Because the folds partition **models (persons)**, every metric is computed on models the
calibration never saw — the exact generalisation question the circularity obscures.

---

## 2. Design

### Inputs (existing, unchanged)
- **Response matrix:** `staging/response_matrix.csv` — 82 models × 6,180 non-optional
  criteria; cells ∈ {0, 1, NaN}; loaded via `calibrate_partial.load_matrix`.
- **Q-matrix:** `data/TutorBench/curated/rubrics_qmatrix_curated.jsonl` (post-reorg
  path); `q_mapping` over content / diagnosis / scaffolding (6,845 criteria).

### 2-skill structure (built exactly as Run 1)
The 3-skill confirmatory Q `(content, diagnosis, scaffolding)` is collapsed with
`calibrate_mirt.collapse_q_matrix(Q, ["content", "diagnosis"])` — the logical-OR of the
content & diagnosis columns — into the 2-d layout `[content+diagnosis, scaffolding]`.
Dim 0 is labelled **correctness** (merged content+diagnosis), dim 1 **scaffolding**.
This is the same in-memory Q transform Run 1's `--collapse content,diagnosis` used; the
rubric bank and skill definitions are never touched.

### Reused fitter internals (no reimplementation, no edits)
`scripts/kfold_cv_mirt.py` **imports** `scripts/calibrate_mirt.py` and reuses:
- `prepare_block()` — `select_sparse` + `split_zero_variance` + `align_q_rows`
  (identical empty-row/col, all-fail, all-pass, and all-zero-Q item dropping);
- `collapse_q_matrix()` — the 2-skill collapse;
- `fit_m2pl_em()` — the Bock–Aitkin EM / MML fitter (same ridge=1e-3, grid=7,
  tol=1e-4, max_iter=200);
- `build_grid()`, `base_log_weights()`, `prior_log_weights()` — the same fixed
  Gauss–Hermite quadrature (49 nodes for the 2-d model) and standard-normal prior.

The **EAP scoring** of held-out persons is the only genuinely new code, and it reuses
the fitter's exact E-step math (per-person log-likelihood over the grid using only
observed cells → posterior → posterior-mean ability). No M2PL math is re-derived.

### Per-fold procedure
For each fold `f` (folds = a seeded random shuffle of the 82 models, seed 20260729):
1. **TRAIN** = models not in `f`; **TEST** = models in `f`.
2. Fit the 2-skill M2PL on **TRAIN persons only** → frozen `(a_correctness,
   a_scaffolding, b)` for the items that survive the TRAIN-fold variance filter.
   (~3.1k–3.3k items survive per fold; the surviving set differs slightly by fold,
   which is expected and handled by the stability analysis below.)
3. For each **TEST** person, holding item params fixed, estimate the 2-d ability
   **θ via EAP** over the quadrature grid from that person's **observed** responses on
   the fitted items, then predict `P(pass) = σ(a·θ − b)` for every observed
   (test-person, fitted-item) cell.
4. Compute out-of-sample metrics on those held-out cells (observed, non-NaN only):
   **log-loss, accuracy@0.5, AUC, Brier**.

Metrics are **pooled** across all held-out cells (primary) and reported **per fold**
(spread). k = 5 is the headline; k = 10 is reported as a robustness check.

### In-sample baseline (optimism gap)
Fit the 2-skill M2PL once on **all 82** models; EAP-score every person from their own
responses under those all-82 params; compute the same metrics. `gap = OOS − in-sample`.

### Item-param cross-fold stability
For items fit in ≥ 2 folds, correlate per-item `a_correctness`, `a_scaffolding`, and
`b` between every pair of folds (Pearson, on the items common to the pair), and against
the all-82 reference. The **median** cross-fold correlation is a direct, honest read on
how reproducible the calibration is at N = 82.

---

## 3. Results

### 3.1 Out-of-sample metrics (pooled) vs in-sample

| Metric      | k=5 OOS | k=10 OOS | In-sample (all 82) | Gap (k=5 OOS − in-sample) |
|-------------|--------:|---------:|-------------------:|--------------------------:|
| Log-loss ↓  | 0.2635  | 0.2548   | 0.2160             | **+0.0475** (OOS worse)   |
| Accuracy ↑  | 0.8943  | 0.8966   | 0.9060             | **−0.0118** (OOS worse)   |
| AUC ↑       | 0.9045  | 0.9087   | 0.9312             | **−0.0267** (OOS worse)   |
| Brier ↓     | 0.0765  | 0.0748   | 0.0672             | **+0.0093** (OOS worse)   |

Held-out cells: 259,199 (k=5) / 264,404 (k=10). Positive class (pass) rate ≈ 14.1%.

**Reading:** the optimism gap is **small**. Out of sample the model still discriminates
well (AUC ≈ 0.90) and is well-calibrated (Brier ≈ 0.076, log-loss ≈ 0.26). The gap
between fitting on 65–66 models and being scored on unseen ones is only ~0.01 in
accuracy, ~0.027 in AUC, and ~0.05 in log-loss — i.e. the in-sample numbers were not
badly inflated by the circularity. k=10 (larger train folds) closes the gap slightly, as
expected.

### 3.2 Per-fold spread (k=5)

| Fold | Train | Test | Items | Cells  | Log-loss | Acc    | AUC    | Brier  |
|-----:|------:|-----:|------:|-------:|---------:|-------:|-------:|-------:|
| 0    | 65    | 17   | 3117  | 52,073 | 0.2292   | 0.9063 | 0.9208 | 0.0679 |
| 1    | 65    | 17   | 3313  | 55,035 | 0.2436   | 0.9036 | 0.8903 | 0.0702 |
| 2    | 66    | 16   | 3179  | 49,831 | 0.3284   | 0.8681 | 0.8859 | 0.0946 |
| 3    | 66    | 16   | 3291  | 51,490 | 0.2592   | 0.8970 | 0.9051 | 0.0748 |
| 4    | 66    | 16   | 3237  | 50,770 | 0.2609   | 0.8946 | 0.9155 | 0.0762 |

Per-fold spread: AUC std ≈ 0.014, accuracy std ≈ 0.014, log-loss std ≈ 0.034 (k=5).
k=10 fold spread is wider (AUC std ≈ 0.021, log-loss std ≈ 0.051) — smaller test folds
(8–9 models) are noisier per fold, but the pooled numbers agree with k=5.

### 3.3 Item-parameter cross-fold stability

Median cross-fold Pearson correlation of per-item estimates (items common to a fold
pair; median ~3,100 shared items/pair at k=5, ~3,250 at k=10):

| Parameter        | k=5 median pairwise | k=5 median vs all-82 | k=10 median pairwise | k=10 median vs all-82 |
|------------------|--------------------:|---------------------:|---------------------:|----------------------:|
| `b` (difficulty) | **0.853**           | 0.907                | **0.919**            | 0.958                 |
| `a_correctness`  | **0.800**           | 0.877                | **0.888**            | 0.931                 |
| `a_scaffolding`  | **0.478**           | 0.515                | **0.725**            | 0.772                 |

**Reading:** difficulty `b` and the correctness discrimination `a_correctness` are
**reproducible** across folds even at N ≈ 65 train persons (median r ≈ 0.80–0.85 at k=5,
rising to ≈ 0.89–0.92 at k=10 with ~74 train persons — exactly the "more persons → more
stable" pattern you want to see). The **scaffolding discrimination is the weak link**
(median r ≈ 0.48 at k=5), which is consistent with the known content↔diagnosis collapse
rationale and with scaffolding being the sparsest, least-identified axis. It stabilises
markedly at k=10 (r ≈ 0.73), i.e. its instability is a small-train-N effect, not a
structural non-identifiability.

### 3.4 Optional CAT pilot (fold 0 held-out models) — PILOT ONLY

`scripts/kfold_cat_pilot.py` runs a max-Fisher-information adaptive test on the 17
**unseen** fold-0 models using fold 0's frozen params, with the MIRT Newton/Laplace
update reused verbatim from `tutor_cat.mirt` (dimension-agnostic; used here in 2-d).
Stopping rule: both per-dim SE < 0.40 after ≥ 5 items, cap 60.

| Pilot metric (fold 0, n=17)                 | Value |
|---------------------------------------------|------:|
| Median items to converge (SE < 0.40)        | 20    |
| Median predicted-score abs. error vs full   | 0.029 |
| Max predicted-score abs. error vs full      | 0.163 |
| Median ability L2 error vs full-response θ  | 1.09  |

**Reading (pilot):** ~20 adaptively-chosen items recover the **predicted score** of an
unseen model to within ~0.03 on average (a very usable ranking signal), while the raw 2-d
**ability vector** is recovered more loosely (L2 ≈ 1.1) — the looseness is concentrated in
the weakly-identified scaffolding axis, matching §3.3. Treat this as a feasibility sanity
check, not a tuned CAT evaluation.

---

## 4. Honest N-per-fold caveat

- At **k=5**, each fit sees **65–66 train persons**; at **k=10**, **73–74**. Both are far
  below the `--min-persons-identifiable=150` heuristic that `calibrate_mirt.py` itself
  warns about for the full 3-dim M2PL. The 2-skill collapse is easier to identify than
  the 3-dim model, and difficulty/correctness are clearly stable, but **scaffolding
  discrimination at this N should be treated as provisional** (its cross-fold r and its
  contribution to ability recovery are the least reliable numbers here).
- Test folds are tiny (16–17 models at k=5, 8–9 at k=10), so **per-fold** metrics are
  noisy; the **pooled** cell-level metrics (>250k cells) are the trustworthy summary.
- The surviving item set differs slightly per fold (variance filter runs per TRAIN
  subset); stability correlations are computed only on items common to each pair, and the
  shared-item counts are reported alongside.
- Metrics are on **observed cells only** (holes are marginalised in EAP, never imputed).

---

## 5. Interpretation — does the 2-skill calibration generalise at N=82?

**Yes, with one caveat.** The out-of-sample gap is small (AUC drops only ~0.027 from
0.931 in-sample to 0.905 held-out; accuracy ~0.012; log-loss ~0.05), so the in-sample
Run-1 fit was **not materially inflated by the circularity** — the calibrated bank
predicts unseen models about as well as it predicts the calibration crowd. The item
**difficulty** and **correctness discrimination** parameters are **reproducible** across
folds and tighten as train-N grows (k=5 → k=10), which is the signature of a genuinely
identified, generalising calibration rather than an over-fit one.

**The caveat is the scaffolding loading.** It is the only parameter that is unstable at
k=5 (median cross-fold r ≈ 0.48) and it dominates the ability-recovery error in the CAT
pilot. This is a small-sample identifiability limit for the *second* latent dimension, not
a failure of the correctness axis. Practically: trust rank-ordering and pass-probability
predictions (driven by `b` and `a_correctness`); treat the scaffolding-specific ability
component as lower-confidence until more persons are added.

---

## 6. Reproduce

From repo root `eduLLM-Evals/` (Python: `..\.venv\Scripts\python.exe`):

```bash
# headline 5-fold CV (grid 7 = 49 nodes for the 2-d model), seed 20260729:
python scripts/kfold_cv_mirt.py --k 5 --grid 7 --seed 20260729 --out-dir staging/kfold

# 10-fold robustness check:
python scripts/kfold_cv_mirt.py --k 10 --grid 7 --seed 20260729 --out-dir staging/kfold/k10

# optional CAT pilot on fold 0's held-out models:
python scripts/kfold_cat_pilot.py --fold 0 --kfold-dir staging/kfold
```

Defaults match Run 1: `--ridge 1e-3`, `--max-iter 200`, `--tol 1e-4`, latent correlation
fixed to identity (pass `--estimate-latent-corr` to re-estimate the 2×2 each EM iter).
All runs are READ-ONLY: no `--write-params`, no edits to `calibrate_mirt.py` /
`calibrate_partial.py`, no changes to the rubric bank.

## 7. Output files

`staging/kfold/` (k=5) and `staging/kfold/k10/` (k=10) each contain:
- `fold_assignments.json` / `.csv` — model → fold map.
- `fold_<f>_item_params.csv` — frozen `a_correctness`, `a_scaffolding`, `b` per fold.
- `insample_baseline_item_params.csv`, `insample_baseline.json` — all-82 fit.
- `metrics_per_fold.csv` — per-fold OOS metrics.
- `metrics_aggregate.json` — pooled OOS, per-fold spread, in-sample, and the gap.
- `item_param_stability.csv` / `.json` — cross-fold correlation tables.
- `kfold_summary.json` — everything + run provenance.

CAT pilot: `staging/kfold/cat_pilot_fold_0.csv` / `.json`.
Console logs: `staging/kfold_k5_console.txt`, `staging/kfold/k10_console.txt`,
`staging/kfold/cat_pilot_fold0_console.txt`.
