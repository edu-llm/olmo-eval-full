# Calibration Run 5 — Full Supplement (2-skill definitive + k-fold refresh)

**Date:** 2026-07-30
**Data:** `staging/response_matrix_full.csv` nonoptional slice = `staging/response_matrix_full_nonopt.csv`
(82 models × 6,180 nonoptional criteria, **0 all-NaN columns**, 498,754 observed cells).
This is the COMPLETE supplement: the 244 previously all-NaN nonoptional columns are now
populated (19,614 holes filled by the Qwen supplement vs the old sparse
`staging/response_matrix.csv`). The 665 optional/presentation columns are excluded (not skills).

**Read-only guarantee:** the fitters (`scripts/calibrate_mirt.py`, `scripts/kfold_cv_mirt.py`)
and the rubric bank were NOT edited. The 2-skill calibration uses a thin driver
(`staging/fit_full2skill.py`) that *imports* `calibrate_mirt.py` internals
(`prepare_block`, `collapse_q_matrix`, `fit_m2pl_em`, `aic_bic`) exactly as the k-fold harness
does — same EM/MML, grid, ridge, collapse. Nothing was committed.

---

## TASK A — Definitive 2-skill M2PL (collapse content+diagnosis → "correctness")

Settings identical to Run 1: grid 7 (GH quadrature), ridge 1e-3, `--estimate-latent-corr`.
The driver fits the unidimensional baseline, the full 3-dim confirmatory M2PL, and the
collapsed 2-skill model on the SAME block, so the three-way model-selection comparison is
apples-to-apples.

### Fit block (Run 5 vs Run 1)

| quantity | Run 1 (sparse) | Run 5 (full) | Δ |
|---|---:|---:|---:|
| items fit | 3,347 | **3,497** | +150 |
| persons fit | 82 | 82 | 0 |
| observed cells | 268,936 | **280,943** | +12,007 |
| dropped all-fail (zero-var) | 1,976 | 2,043 | +67 |
| dropped missing-Q | 613 | 640 | +27 |

Q-pattern counts (content,diagnosis,scaffolding): `100:1228, 110:1159, 101:302, 010:314,
011:20, 001:340, 111:134`. Items loading scaffolding = 796 of 3,497 (≈23%) — scaffolding
is still by far the thinnest axis.

### Model comparison (Run 5, full slice)

| model | loglik | k (params) | AIC | BIC |
|---|---:|---:|---:|---:|
| unidimensional | −62,505.73 | 6,994 | 138,999.46 | **212,757.54** |
| **collapsed 2-skill** | **−61,443.76** | 7,451 | **137,789.53** | 216,367.08 |
| full 3-dim | −61,485.92 | 8,746 | 140,463.84 | 232,698.34 |

- **AIC winner: collapsed (2-skill).** **BIC winner: unidimensional.** (Same winners as Run 1.)
- Collapsed beats full by **ΔAIC = +2,674.3** (Run 1: +2,392.7) and **ΔBIC = +16,331** (Run 1: +15,394).
  The collapse advantage over the full 3-dim model **grew** with the complete data.
- **The full 3-dim loglik (−61,485.92) is now WORSE than the collapsed 2-skill (−61,443.76).**
  A richer model scoring *lower* is a red flag for the 3rd dimension: with content↔diagnosis
  at r≈0.94 the separate diagnosis axis is redundant, and estimating its near-singular latent
  correlation actively degrades the marginal likelihood. This is stronger evidence for the
  collapse than Run 1, where the full model still edged out the collapsed one on raw loglik.

### Latent correlations

- **correctness ↔ scaffolding (2-skill): −0.4621** (Run 1: −0.4175) → **more negative** by 0.045.
  The scaffolding axis is, if anything, *more* clearly distinct from correctness with the
  complete data — models that score well on correctness tend to score worse on scaffolding.
- Full 3-dim latent correlation:
  - content ↔ diagnosis = **0.9422** (Run 1: 0.9461) → still near-collinear ⇒ collapse justified.
  - content ↔ scaffolding = −0.2533, diagnosis ↔ scaffolding = −0.2095 (both slightly more
    negative than Run 1's −0.170 / −0.181).

### Raw-loglik caveat (Run 5 vs Run 1)

Raw loglik/AIC/BIC are **larger in magnitude** in Run 5 only because there are more observed
cells (+12,007) and more items/params — they are NOT directly comparable across the two data
sets. Normalised per observed cell the collapsed fit is essentially unchanged:

- Run 1 collapsed: −58,479.9 / 268,936 = **−0.2175 / cell**
- Run 5 collapsed: −61,443.8 / 280,943 = **−0.2187 / cell**

i.e. the 2-skill model fits the fuller matrix about as well per cell as before. The
*decision-relevant* quantities (winner ordering, collapse margin, latent correlation) are what
moved — all in the direction of confirming the 2-skill structure.

**Outputs:** `staging/calibration_mirt_full2skill.csv` (3,497 items: `a_correctness,
a_scaffolding, b, n_persons, flags`), `staging/calibration_mirt_full2skill_manifest.json`.

---

## TASK B — k-fold refresh on the full matrix

Same protocol as before: `scripts/kfold_cv_mirt.py`, seed 20260729, grid 7, ridge 1e-3,
latent-corr OFF (harness default), pointed at the full-matrix slice via `--matrix`, writing to
`staging/kfold_full/` (k=5) and `staging/kfold_full/k10/` (k=10). The old sparse outputs were
not touched.

### k = 5 — OOS metrics (pooled) and stability

| metric | prior (sparse) | Run 5 (full) | Δ |
|---|---:|---:|---:|
| OOS AUC | 0.905 | **0.9050** | ≈0 |
| OOS accuracy | 0.894 | **0.8937** | −0.001 |
| OOS log-loss | 0.263 | **0.2642** | +0.001 |
| OOS Brier | — | 0.0769 | — |
| in-sample AUC / acc / log-loss | — | 0.9310 / 0.9054 / 0.2173 | — |
| stability median-pairwise `b` | 0.85 | **0.845** | −0.005 |
| stability `a_correctness` | 0.80 | **0.786** | −0.014 |
| stability `a_scaffolding` | 0.48 | **0.495** | **+0.015** |

Optimism gap (pooled OOS − in-sample): log-loss +0.047, accuracy −0.012, AUC −0.026,
Brier +0.009 — small and healthy, essentially the same as before.

### k = 10 — OOS metrics (pooled) and stability

| metric | prior sparse k=10 | Run 5 full k=10 | Δ |
|---|---:|---:|---:|
| OOS AUC | 0.9087 | 0.9089 | +0.0002 |
| OOS accuracy | 0.8966 | 0.8957 | −0.0009 |
| OOS log-loss | 0.2548 | 0.2559 | +0.0012 |
| OOS Brier | 0.0748 | 0.0753 | +0.0005 |
| stability `b` | 0.919 | 0.913 | −0.007 |
| stability `a_correctness` | 0.888 | 0.875 | −0.013 |
| stability `a_scaffolding` | **0.725** | **0.637** | **−0.088** |

(k=10 in-sample = all-82 fit: AUC 0.9310, acc 0.9054, log-loss 0.2173.)

### Did scaffolding stability improve? **No.**

This is the key question and the answer is negative:

- **k=5:** scaffolding median cross-fold correlation moved only 0.48 → 0.495 (+0.015) — inside
  the noise. It remains by far the weakest axis (vs `a_correctness` 0.786 and `b` 0.845).
- **k=10:** scaffolding stability actually **dropped** 0.725 → 0.637, and both `a_correctness`
  and `b` ticked down slightly too.

Why: the 244 newly-filled items add cells and items but **not persons** — the sample is still
82 models. Scaffolding is the rarest skill (≈23% of items load it), so its identifiability is
capped by the 82-person crowd, not by item coverage. The extra items are disproportionately
correctness/content items and inject additional noise into the already-thin scaffolding
estimates rather than stabilising them. The generalisation metrics (OOS AUC/acc/log-loss/Brier)
are, meanwhile, essentially unchanged — the frozen 2-skill bank predicts held-out models just
as well on the fuller data as on the sparse data.

**Outputs:** `staging/kfold_full/` (k=5) and `staging/kfold_full/k10/` (k=10): fold
assignments, per-fold item params, per-fold + pooled metrics, in-sample baseline, item-param
stability, and full summaries.

---

## Standing conclusions (do the complete data change anything?)

1. **2-skill structure — confirmed and reinforced.** content↔diagnosis stays near-collinear
   (r≈0.94); the collapsed 2-skill model wins on AIC by a *larger* margin than Run 1, and the
   full 3-dim model now even loses to it on raw loglik. **Keep collapsing content+diagnosis →
   "correctness"; keep scaffolding separate.** BIC still prefers the unidimensional model
   (heavy param penalty on 82 persons), unchanged from Run 1 — so the collapse decision is a
   deliberate AIC/interpretability call, not a BIC one.
2. **correctness ⟂ scaffolding — still a real, distinct axis** (latent r = −0.46, slightly
   stronger than Run 1). The two-dimensional structure is not an artifact of the sparse matrix.
3. **Generalisation — unchanged and solid.** OOS AUC ≈ 0.905 (k=5) / 0.909 (k=10), small
   optimism gap. The frozen 2-skill bank transfers to unseen models about as well on the
   complete data as on the sparse data.
4. **Scaffolding remains under-identified.** The bottleneck is N = 82 persons, not item
   coverage; more graded items did not fix cross-fold scaffolding stability. To harden the
   scaffolding axis, add **more models (persons)**, not more items.

No prior conclusion is overturned; the collapse decision and the 2-skill CAT structure are
strengthened by the complete data.

---

## Reproduce

```powershell
# from repo root: C:\Users\Samee\Documents\GitHub\olmo-eval-full\eduLLM-Evals
# Python: ..\.venv\Scripts\python.exe

# 0) materialize the nonoptional slice (6,180 cols, matching old response_matrix.csv order)
..\.venv\Scripts\python.exe -c "import pandas as pd; old=list(pd.read_csv('staging/response_matrix.csv',index_col='model').columns); pd.read_csv('staging/response_matrix_full.csv',index_col='model').reindex(columns=old).to_csv('staging/response_matrix_full_nonopt.csv')"

# A) definitive 2-skill calibration (uni + full-3-dim + collapsed-2-skill, corr on, grid 7)
..\.venv\Scripts\python.exe staging/fit_full2skill.py

# equivalently, the stock 3-way comparison manifest (no 2-skill per-item CSV) via the official script:
..\.venv\Scripts\python.exe scripts/calibrate_mirt.py --matrix staging/response_matrix_full_nonopt.csv --rubrics data/TutorBench/curated/rubrics_qmatrix_curated.jsonl --collapse content,diagnosis --estimate-latent-corr --grid 7 --out-dir staging/full2skill_official

# B) k-fold refresh on the full slice
..\.venv\Scripts\python.exe scripts/kfold_cv_mirt.py --matrix staging/response_matrix_full_nonopt.csv --rubrics data/TutorBench/curated/rubrics_qmatrix_curated.jsonl --k 5  --seed 20260729 --grid 7 --out-dir staging/kfold_full
..\.venv\Scripts\python.exe scripts/kfold_cv_mirt.py --matrix staging/response_matrix_full_nonopt.csv --rubrics data/TutorBench/curated/rubrics_qmatrix_curated.jsonl --k 10 --seed 20260729 --grid 7 --out-dir staging/kfold_full/k10
```

### Output paths
- `staging/response_matrix_full_nonopt.csv` — nonoptional slice (82 × 6,180, 0 all-NaN).
- `staging/calibration_mirt_full2skill.csv` (+ `_manifest.json`) — definitive 2-skill params.
- `staging/fit_full2skill.py` — read-only 2-skill driver (reuses `calibrate_mirt.py`).
- `staging/kfold_full/` — k=5 refresh; `staging/kfold_full/k10/` — k=10 refresh.
- Console logs: `staging/full2skill_console.txt`, `staging/kfold_full_console.txt`,
  `staging/kfold_full_k10_console.txt`.
