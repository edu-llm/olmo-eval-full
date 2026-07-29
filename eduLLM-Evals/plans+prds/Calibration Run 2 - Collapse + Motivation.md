# Calibration Run 2 — Collapse + Motivation

**Status:** read-only candidate test (NO params written; curated bank untouched). Produced `staging/run2*` outputs + this memo.
**Date:** 2026-07-29
**Analyst run:** `scripts/calibrate_mirt.py` (UNCHANGED — no source edit was needed), on `staging/response_matrix.csv` with an EXPERIMENTAL Q-matrix variant.

---

## TL;DR (decision-ready)

- **Candidate structure tested:** 3 latent skills = **[correctness, scaffolding, motivation]**, where `correctness = content OR diagnosis`, `scaffolding` = the existing bit, `motivation = 1` for criteria the affect/motivation text classifier tagged (`hit_affect_motivation == 1`).
- **Motivation is a healthy, well-populated item cluster.** 605 criteria assigned, **485 survived** the fit; their discriminations are **median a ≈ 0.98** (comparable to correctness ≈ 1.03, and *stronger* than scaffolding ≈ 0.42) with real pass-rate spread (mean ≈ 0.27). The dimension does **not** collapse to zero.
- **Motivation is clearly distinct from scaffolding** (latent r = **−0.47**), and splitting it out of scaffolding improves fit on the same data (ΔAIC = +1616, ΔBIC = +1340 in favor of the 3-dim).
- **But motivation is nearly collinear with correctness: latent r = +0.90.** This is the same red flag that fired the content↔diagnosis collapse (r = 0.946). Even so, keeping motivation separate still edges out folding it into correctness (ΔAIC = +1489, but ΔBIC only **+460** — thin).
- **Verdict: DEFER motivation.** On THIS fleet the data mildly favor a motivation axis and it is well-measured and separable from scaffolding — but its **r = 0.90 with correctness at a non-identifiable N = 82** means we cannot yet call it an independent competency. Re-test on a ≥150-person matrix with more affect-capable (instruct/RLHF) tutors that can dissociate "gets it right" from "is encouraging."
- **N = 82 ≪ 150 identifiability floor** — everything below is provisional / machinery-exercise only, exactly as in Run 1.

---

## 0. Method — how the candidate structure was fit without editing the fitter

`scripts/calibrate_mirt.py` reads each criterion's Q row as `[q_mapping.get(s,0) for s in SKILLS]` with `SKILLS = (content, diagnosis, scaffolding)`, and is otherwise **agnostic to what those slot names mean** (labels are cosmetic; `--collapse` validates against the same names). So no source edit was required. The experimental bank **repurposes the three slots**:

| q_mapping slot (what the fitter reads) | packed candidate skill |
|---|---|
| `content` | **correctness** = content OR diagnosis |
| `diagnosis` | **scaffolding** (unchanged bit) |
| `scaffolding` | **motivation** (classifier tag) |

Therefore in every Run-2 output the columns/labels map as: `a_content → a_correctness`, `a_diagnosis → a_scaffolding`, `a_scaffolding → a_motivation`, and the printed `latent correlation (content, diagnosis, scaffolding)` is actually **(correctness, scaffolding, motivation)**. **No fitter edits were made.** (`calibrate_partial.py` / `calibrate_mirt.py` are byte-for-byte unchanged.)

**Experimental Q rule (documented exactly).** `motivation = 1` iff the heuristic classifier `scripts/classify_orphan_criteria.py` fired its affect bank on the criterion text — i.e. the per-item CSV column `hit_affect_motivation == 1` (the MULTI-LABEL affect hit, so a criterion counts as motivation even when it also loads correctness/scaffolding, per the task's "even if it already loads content/scaffolding" clause). This is a HEURISTIC keyword tag; it needs a hand audit before anything is committed. Builder: `scripts/build_motivation_qmatrix.py` → `data/experimental/rubrics_qmatrix_collapse_motivation.jsonl` (a copy of the curated bank with only `q_mapping` replaced; the stale `discrimination`/`irt_params` fields are carried over verbatim and are **ignored** by the fit).

---

## 1. Experimental Q-matrix — item counts

Over the **6,180 response-matrix columns** (non-optional curated criteria):

| skill (candidate) | criteria with bit = 1 |
|---|---:|
| correctness (content OR diagnosis) | 5,026 |
| scaffolding | 1,142 |
| motivation | 605 |
| **all-zero Q** (no skill) | **285** |

Motivation **rescues 465** criteria that were all-zero (unfittable) under the curated content/diagnosis/scaffolding scheme (matrix-column all-zero-Q drops fall 750 → 285). After the fitter's empty / zero-variance / all-zero-Q filters, **3,725 items** are fitted (vs 3,347 in Run 1); the extra 378 are motivation-only survivors. Fitted Q-pattern counts (correctness | scaffolding | motivation):

| pattern | meaning | n |
|---|---|---:|
| `100` | correctness only | 2,526 |
| `110` | correctness + scaffolding | 414 |
| `101` | correctness + motivation | 83 |
| `010` | scaffolding only | 300 |
| `001` | **motivation only** | 378 |
| `011` | scaffolding + motivation | 12 |
| `111` | all three | 12 |

---

## 2. Four-model comparison (grid = 7, `--estimate-latent-corr`)

> **⚠ Item-set caveat (read this before comparing rows).** The prior-run models (Run 1) were fit on **item set A = 3,347 items / 268,936 observed cells**. The NEW model rescues 378 motivation-only orphans, so it is fit on **item set B = 3,725 items / 299,163 cells = A ∪ {378 motivation-only}**. **Log-lik / AIC / BIC are only comparable WITHIN the same item set.** Rows 1–3 (set A) are mutually comparable; row 4 (set B) is not directly comparable to rows 1–3 on raw AIC/BIC (it scores more observed cells). The decision-relevant, same-data motivation tests are in §3.

| # | Model | dims | loglik | k | AIC | BIC | item set (items / obs cells) |
|---|---|---:|---:|---:|---:|---:|---|
| 1 | Unidimensional | 1 | −59,501.64 | 6,694 | 132,391.27 | 202,693.19 | A (3,347 / 268,936) |
| 2 | Collapsed-2dim: **correctness + scaffolding** (= Run 1 content+diagnosis collapse) | 2 | −58,479.93 | 7,121 | 131,201.85 | 205,988.23 | A (3,347 / 268,936) |
| 3 | Full-3dim: content / diagnosis / scaffolding (Run 1) | 3 | −58,438.26 | 8,359 | 133,594.53 | 221,382.66 | A (3,347 / 268,936) |
| 4 | **NEW 3-dim: correctness / scaffolding / motivation** | 3 | −68,380.04 | 7,986 | 152,732.08 | 237,453.50 | B (3,725 / 299,163) |

(Rows 1–3 pulled verbatim from `plans+prds/Calibration Run 1 - Results + Collapse Decision.md` and `staging/calibration_mirt_manifest.json`; row 4 from `staging/run2/calibration_mirt_manifest.json`. Same grid = 7, same EM tol, same BIC sample convention = observed cells.)

---

## 3. Does motivation earn its own axis? (same-data tests, item set B = 3,725 items / 299,163 cells)

Two `--collapse` runs on the **identical** new item set (collapse is an in-memory Q transform, so the item set is unchanged — a true apples-to-apples test):

| Model (all on set B) | dims | loglik | k | AIC | BIC |
|---|---:|---:|---:|---:|---:|
| Unidimensional | 1 | −69,772.84 | 7,450 | 154,445.68 | **233,480.82** |
| Collapsed: [correctness, **scaffolding∪motivation**] | 2 | −69,214.04 | 7,960 | 154,348.07 | 238,793.67 |
| Collapsed: [**correctness∪motivation**, scaffolding] | 2 | −69,221.58 | 7,889 | 154,221.16 | 237,913.54 |
| **NEW full-3dim [correctness, scaffolding, motivation]** | 3 | **−68,380.04** | 7,986 | **152,732.08** | 237,453.50 |

- **Motivation vs folding it into scaffolding:** the 3-dim beats the merge on **AIC (Δ +1,616)** and **BIC (Δ +1,340)** → motivation is **not** scaffolding.
- **Motivation vs folding it into correctness:** the 3-dim still beats the merge on **AIC (Δ +1,489)** and **BIC (Δ +460)** → motivation is not *statistically* redundant with correctness either, but the **BIC margin is thin (460)** given the r = 0.90 collinearity.
- **AIC winner among all set-B candidates: the full 3-dim (motivation separate).** **BIC winner: unidimensional** (heavy parsimony penalty at n = 299,163, same pattern as Run 1).

### Latent correlation R (NEW 3-dim), order = (correctness, scaffolding, motivation)

|  | correctness | scaffolding | motivation |
|---|---:|---:|---:|
| **correctness** | 1.000 | −0.383 | **0.900** |
| **scaffolding** | −0.383 | 1.000 | −0.474 |
| **motivation** | **0.900** | −0.474 | 1.000 |

- **r(correctness, motivation) = 0.90** — nearly collinear (cf. the content↔diagnosis r = 0.946 that triggered the Run-1 collapse). This is the core reason to defer.
- **r(scaffolding, motivation) = −0.47** — motivation is distinct from scaffolding.
- **r(correctness, scaffolding) = −0.38** — consistent with Run 1's finding that scaffolding is a separate (mildly negatively-loaded) axis on this fail-dominated fleet.

---

## 4. Motivation-dimension health

From `scripts/analyze_motivation_dim.py` (→ `staging/run2/motivation_health.txt`):

- **Assigned motivation = 1 (matrix columns): 605. Survived to the fit: 485. Dropped: 120** (100 of them all-fail zero-variance; rest empty / all-zero-Q). So the dimension is **well-populated**, not near-empty.
- **Pass-rate across the 82-model fleet** (per-item mean over observed cells): assigned mean 0.222 (median 0.185); survived mean 0.267 (median 0.244), range 0.012–0.753. Real spread, **not** near-ceiling or near-floor. (100 assigned items are all-fail — every model fails them — and are correctly dropped.)
- **Fitted motivation discrimination `a_motivation`** (n = 485): min −3.61, p25 0.62, **median 0.98**, p75 1.40, max 14.15, mean 1.16. Only **28/485 negative**, **24/485 near-zero (|a|<0.2)**, **10/485 extreme (|a|>6)**. For context: correctness median 1.03 (n = 3,035), scaffolding median 0.42 (n = 738). **Motivation discriminates better than scaffolding and about as well as correctness.**

**Health read:** the motivation *items* are not the problem — they survive, they spread, they discriminate. The problem is purely the **latent ability axis being ~collinear with correctness (r = 0.90)** at this N.

---

## 5. Verdict

**Does adding motivation improve fit (AIC/BIC) beyond collapse-only?**
Yes on AIC, marginally on BIC, on the same item set: the 3-dim [correctness, scaffolding, motivation] beats every 2-dim collapse of it (whether motivation is folded into scaffolding or into correctness). But among *all* candidates BIC still prefers the unidimensional model, and the motivation-vs-correctness BIC margin is only ~460.

**Is motivation identifiable / separable in THIS fleet?**
Partially. It is **cleanly separable from scaffolding** (r = −0.47) and is a healthy, well-discriminating item cluster. It is **not** demonstrably separable from **correctness** (r = 0.90 ≈ the 0.946 collapse threshold), at an N that the script itself flags as **non-identifiable**.

**Recommendation: DEFER motivation** pending (a) a **≥ 150-person** matrix, and (b) a cohort with **more affect-capable tutors** (instruct/RLHF models) that can dissociate "answers correctly" from "responds supportively." On the current fail-dominated fleet of mostly small base models, weak models fail affect criteria for the same reason they fail correctness criteria, which is exactly what an r = 0.90 correctness↔motivation correlation looks like. Do **not** collapse-and-adopt now; also do **not** discard motivation — the item cluster is real. **Interim skill set stays: collapse content+diagnosis → correctness; keep scaffolding separate (2-skill), per Run 1**, with motivation parked for the powered re-test.

**Repeat identifiability caveat:** N = 82 ≪ 150. The 3-dim confirmatory M2PL is **NOT identifiable** at this sample size (`identifiable: false` in every manifest); a high latent r at small N can be partly a non-identifiability artifact. All numbers here are provisional / machinery-exercise only.

---

## 6. Exact reproduce commands (read-only; no `--write-params`; curated bank never written)

Run from `eduLLM-Evals/` (Python: `..\.venv\Scripts\python.exe`):

```powershell
# 1. Refresh the affect/motivation text classifier on the CURATED bank (structural only)
..\.venv\Scripts\python.exe scripts\classify_orphan_criteria.py --rubrics data\curated\rubrics_qmatrix_curated.jsonl --out-dir staging\run2_orphan --structural-only

# 2. Build the EXPERIMENTAL [correctness, scaffolding, motivation] Q variant (copy of curated; q_mapping replaced)
..\.venv\Scripts\python.exe scripts\build_motivation_qmatrix.py

# 3a. NEW 3-dim fit + same-set collapse of motivation INTO scaffolding
..\.venv\Scripts\python.exe scripts\calibrate_mirt.py --rubrics data\experimental\rubrics_qmatrix_collapse_motivation.jsonl --collapse diagnosis,scaffolding --estimate-latent-corr --grid 7 --efa --out-dir staging\run2

# 3b. Same-set collapse of motivation INTO correctness (the r=0.90 test)
..\.venv\Scripts\python.exe scripts\calibrate_mirt.py --rubrics data\experimental\rubrics_qmatrix_collapse_motivation.jsonl --collapse content,scaffolding --estimate-latent-corr --grid 7 --out-dir staging\run2_corrmerge

# 4. Motivation-dimension health
..\.venv\Scripts\python.exe scripts\analyze_motivation_dim.py
```

> `--collapse` names are the raw slot names `content/diagnosis/scaffolding`; under the repurposing they mean `correctness/scaffolding/motivation`. So `--collapse diagnosis,scaffolding` merges **scaffolding+motivation**, and `--collapse content,scaffolding` merges **correctness+motivation**.

---

## Output paths

- `data/experimental/rubrics_qmatrix_collapse_motivation.jsonl` — experimental Q variant (curated copy, `q_mapping` replaced). **Curated bank NOT touched.**
- `scripts/build_motivation_qmatrix.py`, `scripts/analyze_motivation_dim.py` — new helper scripts (no fitter edits).
- `staging/run2_orphan/orphan_criteria.{json,csv}` — refreshed classifier labels on the curated bank.
- `staging/run2/calibration_mirt.csv`, `staging/run2/calibration_mirt_manifest.json`, `staging/run2/mirt_run2_console.txt`, `staging/run2/motivation_health.txt`, `staging/run2/coverage_report.{csv,json}` — NEW 3-dim run (+ scaffolding∪motivation collapse).
- `staging/run2_corrmerge/calibration_mirt_manifest.json`, `staging/run2_corrmerge/mirt_run2_corrmerge_console.txt`, … — correctness∪motivation collapse test.
- `plans+prds/Calibration Run 2 - Collapse + Motivation.md` — this memo.

**Fitter edits made: NONE.** `--write-params` NOT used. `data/curated/rubrics_qmatrix_curated.jsonl` NOT modified.
