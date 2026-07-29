# Calibration Run 1 — Results + Collapse Decision

**Status:** read-only analysis (no params written). Produced `staging/` outputs + this memo.
**Date:** 2026-07-29
**Analyst run:** `scripts/calibrate_mirt.py` + `scripts/calibrate_partial.py` (both unchanged), on `staging/response_matrix.csv`.

---

## TL;DR (decision-ready)

- **Empirical latent r(content, diagnosis) = 0.946** (full 3-dim confirmatory M2PL, `--estimate-latent-corr`). This is essentially identical to the prior **synthetic r = 0.945**.
- **The pre-registered collapse rule FIRES:** content and diagnosis are not empirically separable. The collapsed 2-dim model (content+diagnosis merged, scaffolding kept separate) **beats the full 3-dim model on both AIC (Δ = +2392.7) and BIC (Δ = +15394.4)** — the third dimension does not earn its parameters.
- **Scaffolding is a distinct axis** and must NOT be collapsed: r(content, scaffolding) = −0.170, r(diagnosis, scaffolding) = −0.181, and in the collapsed model r(content+diagnosis, scaffolding) = −0.418.
- **Recommendation: PROVISIONALLY COLLAPSE content+diagnosis → one skill; keep scaffolding separate (2-skill Q-matrix).** But treat as provisional — see caveats: **N = 82 persons ≪ 150 identifiability floor**, 244 empty columns, `no_decision → NaN`. Re-run on a ≥150-person matrix before writing any params.
- **No Q-matrix / column misalignment:** all 6,180 matrix columns exist in the curated bank (0 absent). The 613 "missing Q-row" drops are legitimately all-zero `q_mapping` rows (no skill assigned), not misalignment.

---

## 1. Matrix provenance + items dropped

**Source matrix:** `staging/response_matrix.csv` (sha256 `21e9f254e0da3dd69967e7668746e9fdb78a92864ad38067f9fb2a840591e897`), manifest `staging/response_matrix_manifest.json` (generated 2026-07-29T16:38:36Z).

- **Shape:** 82 models (persons) × 6,180 curated non-optional criteria (items) = 506,760 cells.
- **Fill:** 479,140 observed cells = **94.55%**. Cells ∈ {0, 1, NaN}. `pass=1`, `fail=0`, `no_decision → NaN` (indistinguishable from absent), absent-from-CSV → NaN.
- Columns = non-optional curated `criterion_id`s in bank order from `data/curated/rubrics_qmatrix_curated.jsonl`; rows = `tutor_model` (cohort order).
- 244 columns entirely NaN (27 absent scenarios `tb_0487…tb_0513`); least-covered model `Qwen/Qwen2.5-1.5B` (5,504 obs), best `bigscience/bloomz-1b7` (5,906 obs).

**Item drop accounting (of 6,180 columns → 3,347 fitted):**

| Reason | Count |
|---|---:|
| All-NaN (empty columns) | 244 |
| Zero-variance all-fail (observed but all 0) | 1,976 |
| Zero-variance all-pass | 0 |
| Missing / all-zero Q-row (no skill assigned) | 613 |
| **Total dropped** | **2,833** |
| **Items fitted** | **3,347** |

Observed cells among fitted items (BIC sample size) = **268,936**.

**Zero-variance skew:** the whole matrix is heavily fail-dominated (427,560 fail vs 51,580 pass), so 1,976 criteria are all-fail — no all-pass items exist. These carry no information for a 2PL/1PL fit and are dropped before fitting.

## Q-matrix alignment (curated bank vs matrix columns)

- **6,180 / 6,180 matrix columns are present in the curated bank → 0 absent, no misalignment.**
- **750 columns have an all-zero `q_mapping`** (assigned to no skill). Of these, 137 were already removed as empty/zero-variance; the remaining **613 are dropped at the Q-alignment stage** (unfittable under confirmatory masking, where `a_k` is free iff `q_k == 1`). Reported by the script as `dropped_missing_qrow`.
- **Q-pattern counts over the 3,347 fitted items** (content|diagnosis|scaffolding):

  | pattern | meaning | n |
  |---|---|---:|
  | `100` | content only | 1,206 |
  | `110` | content+diagnosis | 1,112 |
  | `101` | content+scaffolding | 283 |
  | `010` | diagnosis only | 291 |
  | `011` | diagnosis+scaffolding | 19 |
  | `001` | scaffolding only | 312 |
  | `111` | all three | 124 |

  Loading totals: content on 2,725 items, diagnosis on 1,546, scaffolding on 738.

---

## 2. Unidimensional 2PL / Rasch (`calibrate_partial.py`) — did not fit

Both `--method girth` (2PL) and `--method rasch` (1PL) **exit 3 without fitting** on this matrix, for two independent reasons:

1. **Dense-block path is empty.** The girth/rasch path keeps columns observed by ≥15 persons (5,936 columns survive) then requires **hole-free rows** over that block. Max per-model coverage is 5,906 < 5,936, so **no model is complete → block = 0 rows**. Script message: *"dense block is empty or trivially small (rows=0, cols=5936)."*
2. **`girth` is not installed** in the venv (`No module named 'girth'`) — the girth/rasch fitter could not run even on a valid block. `py-irt` (the sparse alternative) is also absent.

**Consequence:** the standalone unidimensional 2PL CSV was not produced. The authoritative unidimensional fit is instead the **pure-numpy EM 1-dim baseline inside `calibrate_mirt.py`** (native missing-data handling, apples-to-apples with the multidimensional fits). Its likelihood/AIC/BIC are in the table below. Per-item a/b for the 1-dim baseline are not persisted to CSV by the script; the discrimination/difficulty distributions reported below are from the **full 3-dim** `staging/calibration_mirt.csv`.

### Discrimination (a) + difficulty (b) distributions — full 3-dim M2PL

Per-skill loadings, computed over items that load on that skill (`a_k ≠ 0`):

| skill | n loading | min | p25 | median | p75 | max | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| content | 2,725 | −13.35 | 0.672 | 1.143 | 1.947 | 30.48 | 1.941 |
| diagnosis | 1,546 | −29.78 | 0.385 | 0.973 | 1.797 | 10.74 | 1.503 |
| scaffolding | 738 | −9.87 | −0.287 | 0.432 | 1.30 | 9.63 | 0.455 |

Difficulty `b` (offset scale, all 3,347 items): min −9.14, median 2.582, max 36.77, mean 4.281.

**Degenerate / unstable items:**
- **341 items flagged `extreme_a`** (`|a| > 6`) — thin-sample / near-separation artifacts (note the negative and >20 loadings above). Scaffolding shows the weakest, most centered-on-zero discriminations (median 0.43, p25 negative), consistent with a weakly-populated axis.
- **All 3,347 items flagged `low_n`** — every item has < 150 observing persons because N = 82. This is the identifiability warning, universal by construction here.

---

## 3. MIRT model comparison (unidim vs collapsed-2dim vs full-3dim)

Single run: `calibrate_mirt.py --rubrics data/curated/rubrics_qmatrix_curated.jsonl --collapse content,diagnosis --estimate-latent-corr --grid 7 --efa`. EM: unidim converged in 17 iters, full-3dim in 27 iters (collapsed converged too). Grid = 7 nodes/dim (343 nodes for 3-dim). BIC sample size = 268,936 observed cells.

| Model | dims | loglik | k (params) | AIC | BIC |
|---|---:|---:|---:|---:|---:|
| Unidimensional (1-dim) | 1 | −59,501.64 | 6,694 | **132,391.27** | **202,693.19** |
| Collapsed (content+diagnosis, scaffolding) | 2 | −58,479.93 | 7,121 | **131,201.85** | 205,988.23 |
| Full confirmatory M2PL | 3 | −58,438.26 | 8,359 | 133,594.53 | 221,382.66 |

- **AIC winner: collapsed 2-dim** (lowest AIC).
- **BIC winner: unidimensional** (BIC's heavier parameter penalty at n=268,936 favors the most parsimonious model).
- **Multi beats uni?** No — neither AIC nor BIC (full 3-dim's extra params are not justified).
- **Collapsed beats full?** **Yes on both** (ΔAIC full−collapsed = +2,392.7; ΔBIC = +15,394.4). The content/diagnosis split adds parameters without a commensurate likelihood gain.

### Estimated latent correlation R (full 3-dim)

|  | content | diagnosis | scaffolding |
|---|---:|---:|---:|
| **content** | 1.000 | **0.946** | −0.170 |
| **diagnosis** | 0.946 | 1.000 | −0.181 |
| **scaffolding** | −0.170 | −0.181 | 1.000 |

Collapsed 2-dim R: r(content+diagnosis, scaffolding) = **−0.418**.

**Per-skill discrimination summaries:** see §2 table. content is the strongest/most-populated axis (median a 1.14, 2,725 items); diagnosis is comparable in strength but with large thin-N outliers; scaffolding is weak and centered near zero.

### EFA diagnostics

**Not available** — `factor_analyzer` is not installed (`efa.available = false`). The `girth` dense-block cross-check is likewise unavailable (`No module named 'girth'`). No scree/eigenvalue evidence this run; the dimensionality read rests on the latent-R and AIC/BIC comparison above.

---

## 4. Collapse decision

**Pre-registered rule:** collapse content + diagnosis into one latent skill **if** the estimated latent r(content, diagnosis) stays high (prior synthetic reference r = 0.945).

**Empirical result:** r(content, diagnosis) = **0.946** ≈ 0.945, and the collapsed 2-dim model dominates the full 3-dim model on AIC and BIC. **The rule fires.**

### Recommendation: **COLLAPSE content + diagnosis (→ 2-skill model), keep scaffolding separate — PROVISIONALLY.**

Rationale:
- content and diagnosis are statistically indistinguishable (r ≈ 0.95); the third dimension is not earning its parameters.
- scaffolding is clearly a separate construct (r ≈ −0.17 to −0.18 with the other two; −0.42 with the merged dimension) and must stay its own axis.

### Caveats (why "provisional", not "commit now")

1. **N = 82 persons ≪ 150 identifiability floor.** The script explicitly warns the 3-dim confirmatory M2PL is **not identifiable** at this N (`identifiable: false`). A high latent r at small N can be partly a non-identifiability artifact, not only true collinearity. **Re-run on a ≥150-person matrix to confirm before acting.**
2. **BIC actually prefers unidimensional**, i.e. at maximum parsimony even the content+diagnosis vs scaffolding split is borderline. The collapse-vs-keep call for content/diagnosis is robust (AIC + latent-r + BIC all disfavor the full 3-dim); the 2-dim-vs-1-dim call is not settled.
3. **244 empty columns + 1,976 all-fail + heavy fail skew** shrink the effective bank to 3,347 items; the fit speaks only to that variant subset.
4. **`no_decision → NaN`** is indistinguishable from absent in the matrix; systematic no-decision patterns could bias item difficulty.
5. **No EFA / girth cross-check** this run (optional deps absent) — the dimensionality conclusion has no independent confirmation.

---

## 5. Reproduce commands (read-only, no params written)

Run from `eduLLM-Evals/` (Python: `..\.venv\Scripts\python.exe`):

```powershell
# Coverage report only
..\.venv\Scripts\python.exe scripts\calibrate_partial.py --report-only

# Unidimensional 2PL / Rasch attempts (both exit 3: no hole-free rows + girth absent)
..\.venv\Scripts\python.exe scripts\calibrate_partial.py --method girth
..\.venv\Scripts\python.exe scripts\calibrate_partial.py --method rasch

# Full 3-way comparison: unidim vs collapsed-2dim vs full-3dim, latent corr, EFA (skipped)
..\.venv\Scripts\python.exe scripts\calibrate_mirt.py --rubrics data\curated\rubrics_qmatrix_curated.jsonl --collapse content,diagnosis --estimate-latent-corr --grid 7 --efa
```

Outputs (all under `staging/`): `coverage_report.csv`, `coverage_report.json`, `calibration_mirt.csv`, `calibration_mirt_manifest.json`, `mirt_run1_console.txt` (captured console).

---

## 6. Exact `--write-params` command (run ONCE a choice is approved)

> Do NOT run until the decision is approved **and** (per caveat 1) ideally re-confirmed on a ≥150-person matrix. `--write-params` always writes a NEW file and never touches `data/rubrics_qmatrix_final.jsonl`.

**Important:** `--collapse` is a **fit-time diagnostic only** — it does not change what `--write-params` persists. `calibrate_mirt.py --write-params` always writes the **full 3-vector** M2PL discriminations. So:

**(a) If the decision is to KEEP the 3-skill bank** (persist the full M2PL fit):

```powershell
..\.venv\Scripts\python.exe scripts\calibrate_mirt.py --rubrics data\curated\rubrics_qmatrix_curated.jsonl --estimate-latent-corr --grid 7 --write-params --out-rubrics data\rubrics_qmatrix_mirt.jsonl
```

**(b) If the decision is to APPLY the content+diagnosis collapse:** this is **not** a `--write-params` operation — it requires an upstream change to the skill set / Q-matrix (merging the content and diagnosis columns), which is **out of scope for this read-only run** and must be a separate, reviewed change. Only after that Q-matrix change would you re-fit and `--write-params` the resulting 2-skill bank.

---

## Output paths

- `staging/coverage_report.csv`, `staging/coverage_report.json`
- `staging/calibration_mirt.csv` (full 3-dim per-criterion `a_content, a_diagnosis, a_scaffolding, b, n_persons, flags`)
- `staging/calibration_mirt_manifest.json` (full comparison, collapse block, latent R, identifiability warning, provenance)
- `staging/mirt_run1_console.txt` (captured console of the main run)
- `plans+prds/Calibration Run 1 - Results + Collapse Decision.md` (this memo)
