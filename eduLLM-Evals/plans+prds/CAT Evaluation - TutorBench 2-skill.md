# CAT Evaluation — TutorBench definitive 2-skill M2PL

**Status:** complete (read-only study; nothing written back to the bank or fitters).
**Date:** 2026-07-30
**Harness:** `scripts/cat_eval_tutorbench.py` (imports `scripts/calibrate_mirt.py`,
reuses `scripts/kfold_cat_pilot.py`'s CAT pattern and `tutor_cat.mirt`).
**Outputs:** written to `staging/cat_eval/` (scratch, git-ignored). The shareable
deliverables (six figures, summary tables, `cat_metrics.json`, per-model CSVs, and the
item-bank `_manifest.json`) are copied to the tracked **`reports/cat_eval/`** for review.

---

## 1. What this is

A **full Computerized-Adaptive-Testing (CAT) evaluation** of the definitive 2-skill
TutorBench item bank over **all 82 models**, structured to mirror the unidimensional
MCQ-CAT reference report (`tutor_cat.mcq_irt.pipeline`) — same columns (Items kept /
Bank health / `a` median / Recovery r / CAT items / pIRT MAE) — but adapted to the
**two-dimensional** TutorBench latent space.

The reference is **unidimensional** (one ability per benchmark). TutorBench has **two**
latent abilities, so every ability-based metric is reported **per dimension** with a
combined TutorBench overall row:

* **correctness** — the collapsed content+diagnosis axis (dim 0);
* **scaffolding** — the tutoring-support axis (dim 1).

### Multidimensional adaptation (explicit)

| Reference (unidimensional MCQ-CAT) | This study (2-dimensional TutorBench) |
|---|---|
| Ability `θ` is a scalar; EAP over a 1-D normal grid | Ability `θ = (θ_correctness, θ_scaffolding)`; EAP over a 2-D Gauss–Hermite grid (49 nodes), reusing `calibrate_mirt.build_grid/base_log_weights/prior_log_weights` |
| Item info = 2PL Fisher info `a²·p(1−p)` (scalar) | **Multidimensional Fisher info** `p(1−p)·(m·m)` with `m = q⊙a` the Q-masked 2-vector loading; next item = max total info (reused from `kfold_cat_pilot`) |
| Ability update = EAP re-estimate | **Exact MIRT Newton/Laplace update** (PRD Eqs 1–3) via `tutor_cat.mirt.update`: `U⁻¹ ← U⁻¹ + p(1−p)·mmᵀ`, `θ ← θ + U·m·(y−p)` |
| Stop when posterior SD `< se_target` | Stop when **both** per-dim SEs `= √diag(U) < se_target`; record the first-crossing item count **per dimension** and **overall** |
| Recovery r on one ability | Recovery r **per dimension**, plus an overall (stacked) r |
| pIRT: predicted vs actual accuracy | Same, but predicted accuracy uses the 2-D `σ(a·θ − b)` |

No M2PL math is re-implemented — the pass-probability, Fisher information, Newton/Laplace
update, quadrature grid, and EAP are all imported from the existing fitter / CAT modules.

### Inputs (definitive artifacts, unchanged)

- **Item bank:** `staging/calibration_mirt_full2skill.csv` (+ `_manifest.json`) — 3,497
  fitted items with `a_correctness`, `a_scaffolding`, `b` (collapse content+diagnosis →
  correctness).
- **Response matrix:** `staging/response_matrix_full_nonopt.csv` — 82 models × 6,180
  non-optional criteria (the fit block is the 3,497-item subset; 98.0 % of the block's
  cells are observed).
- **Q / structure:** the 2-skill collapse is baked into the bank (`collapse_q_matrix`).
- **OOS params:** `staging/kfold_full/` (k=5 fold-trained item params + assignments).

---

## 2. Summary table (headline = in-sample)

| Benchmark | Models | Items kept (fit_block / total) | Bank (healthy / needs refit) | a median | Recovery r | CAT items | pIRT MAE |
|---|---|---|---|---|---|---|---|
| **TutorBench (2-skill, overall)** | 82 | 3497 / 6180 | correctness healthy / scaffolding needs refit | corr 1.07 / scaff 0.94 | corr **0.960** / scaff **0.850** | 23.6 (median 26; both dims, 82/82 conv.) | **0.0349** |
| &nbsp;&nbsp;— correctness | 82 | 3108 loading | healthy | 1.068 | **0.960** | 20.4 (median 20; 82/82 conv.) | 0.0349 (overall) |
| &nbsp;&nbsp;— scaffolding | 82 | 482 loading | needs refit | 0.938 | **0.850** | 12.4 (median 11; 82/82 conv.) | — |

- **Recovery r** = Pearson corr between the short adaptive test's ability and the
  FULL-bank ability, over all 82 models (1.0 = identical). Overall stacked r = **0.941**.
- **CAT items** = mean #items to reach per-dim SE < 0.3 (cap 100). Overall = #items for
  **both** dims to be under target.
- **pIRT MAE** = mean over 82 models of |predicted accuracy − actual observed pass-rate|,
  predicted from the CAT ability. Full-bank ceiling (predicted from the full-bank ability)
  = **0.0176**.
- **a median** = median of the **non-zero** fitted loadings per axis (3,108 items load
  correctness, 482 load scaffolding).
- **Items kept** = 3,497 of 6,180 non-optional criteria survive the variance + Q-row
  filters (2,043 all-fail dropped, 640 without a Q-row).

### Out-of-sample honesty check (k-fold fold-trained params)

For each model we re-run the CAT with the item params from the **k-fold fold that held
that model out** (`staging/kfold_full/fold_<f>_item_params.csv`) and correlate the OOS CAT
ability against the model's full-bank ability.

| Recovery r | In-sample (headline) | Out-of-sample (honest) |
|---|---:|---:|
| correctness | 0.960 | **0.852** |
| scaffolding | 0.850 | **0.713** |

The in-sample numbers are somewhat **circular** (the bank was fit on the same 82 models it
now scores). The OOS drop is modest for correctness (0.96 → 0.85) and larger for
scaffolding (0.85 → 0.71) — exactly the axis the k-fold stability study flagged as weak.

---

## 3. Figures

The tracked copies live in [`reports/cat_eval/figures/`](../reports/cat_eval/figures/)
(the script writes them to the git-ignored `staging/cat_eval/figures/`).

- **[`recovery_scatter_correctness.png`](../reports/cat_eval/figures/recovery_scatter_correctness.png)** — full-bank vs CAT correctness ability, 82 points,
  y=x, r = 0.960. Tight along the diagonal.
- **[`recovery_scatter_scaffolding.png`](../reports/cat_eval/figures/recovery_scatter_scaffolding.png)** — same for scaffolding, r = 0.850. The full-bank
  scaffolding abilities **cluster at a few values** (≈ −1.2 / 0 / +1.1): most models carry
  little scaffolding signal so the posterior mean is pulled toward the prior — a direct
  visual of the weak second axis.
- **[`cat_length_hist.png`](../reports/cat_eval/figures/cat_length_hist.png)** — histogram of #items to reach correctness SE < 0.3
  (mean 20.4, median 20; 82/82 converged, cap 100).
- **[`se_reduction_curve.png`](../reports/cat_eval/figures/se_reduction_curve.png)** — mean posterior SE vs #items administered, both dimensions,
  with the SE = 0.3 crossing. Scaffolding SE drops *faster on average* than correctness —
  see the caveat in §4.
- **[`pirt_calibration.png`](../reports/cat_eval/figures/pirt_calibration.png)** — predicted (from CAT ability) vs actual accuracy, 82 points,
  y=x, MAE = 0.0349 (corr 0.935).
- **[`item_info_by_skill.png`](../reports/cat_eval/figures/item_info_by_skill.png)** — distribution of non-zero `a_correctness` vs
  `a_scaffolding` loadings; shows scaffolding's heavier tail of extreme loadings.

---

## 4. Interpretation

**Correctness is a strong, well-behaved CAT axis.** ~20 adaptively-chosen items recover
the full-bank correctness ability at r = 0.96 in-sample and 0.85 out-of-sample, and the
overall pass-rate is reconstructed to a mean absolute error of **3.5 %** (ceiling 1.8 %).
This is a usable adaptive test: a fifth of the items reaches the SE < 0.3 target for all
82 models.

**Scaffolding is the weaker axis, and its CAT behaviour reflects that — in two ways.**

1. **Lower recovery.** Scaffolding recovery is r = 0.85 in-sample but falls to **0.71**
   out-of-sample, the largest optimism gap in the study. The full-bank scaffolding
   abilities themselves are quantized toward the prior (see the scatter), because only
   482 items carry a non-zero scaffolding loading and most models express little
   scaffolding variance.
2. **Deceptively fast "convergence."** Scaffolding SE crosses 0.3 in a *median of 11*
   items — faster than correctness — which is **not** a sign of strength. It is driven by
   a handful of **extreme-discrimination** scaffolding items (max `a_scaffolding` = 18.2;
   120 items with `a > 1.5`), a classic symptom of near-separation / thin-sample
   over-fitting on a poorly-identified axis. A single extreme-`a` item collapses the
   scaffolding posterior variance almost instantly, so the SE stopping rule "converges"
   onto **unstable** parameter estimates. This is precisely why the scaffolding recovery
   r — the metric that actually checks the *value*, not the *uncertainty* — is the one
   that degrades out-of-sample.

**Bank health.** From the k-fold cross-validation study
(`plans+prds/Calibration - K-Fold Cross-Validation (2-skill).md`), the item parameters
that drive correctness (`b`, `a_correctness`) are reproducible across folds (cross-fold
r ≈ 0.85 / 0.80), while `a_scaffolding` is the weak link (cross-fold r ≈ 0.48 at k=5,
recovering to ≈ 0.73 at k=10). We therefore label **correctness healthy / scaffolding
needs refit** (more persons, or regularisation of the extreme scaffolding loadings).

**Bottom line.** Trust the CAT for **correctness / pass-rate ranking** (that is what `b`
and `a_correctness` support, and both in-sample and OOS confirm it). Treat the
**scaffolding-specific ability** as lower-confidence: its adaptive test converges quickly
only because a few over-fit items dominate its information, and its held-out recovery is
the softest number here.

---

## 5. Reproduce

From repo root `eduLLM-Evals/` (Python: `..\.venv\Scripts\python.exe`):

```bash
# full CAT eval over all 82 models (in-sample headline + OOS check), default SE target 0.3:
python scripts/cat_eval_tutorbench.py

# tune the stopping rule / cap / seed:
python scripts/cat_eval_tutorbench.py --se-target 0.3 --max-items 100 --seed 20260730

# skip the out-of-sample k-fold pass:
python scripts/cat_eval_tutorbench.py --no-oos
```

All runs are **READ-ONLY**: the script only *imports* `calibrate_mirt.py` /
`tutor_cat.mirt`; it never edits the fitters, the item bank, or the rubric bank, and
writes only under `staging/cat_eval/`.

---

## 6. Output files

The script writes everything below to the git-ignored `staging/cat_eval/`; the shareable
subset is mirrored to the tracked [`reports/cat_eval/`](../reports/cat_eval/) (all files
except `cat_eval_console.txt`). The large fitted-item bank
`staging/calibration_mirt_full2skill.csv` stays git-ignored (regenerable); only its
`calibration_mirt_full2skill_manifest.json` is tracked under `reports/cat_eval/`.

`staging/cat_eval/`:
- `cat_per_model.csv` — per-model full-bank vs CAT ability (both dims), #items to each
  per-dim + overall SE crossing, final SEs, predicted/actual accuracy.
- `cat_per_model_oos.csv` — the same for the out-of-sample (fold-trained) CAT.
- `cat_metrics.json` — aggregate in-sample + OOS metrics, `a` medians, bank health, config,
  and figure paths.
- `cat_summary_table.md` / `cat_summary_table.csv` — the reference-shaped table above.
- `cat_eval_console.txt` — run log.
- `figures/*.png` — the six figures listed in §3.
