# Calibration Run 5 — Full Supplement (presentation axis + 2-skill refresh)

**Status:** read-only calibration on the COMPLETE data (NO params written; official curated bank + `staging/response_matrix.csv` untouched; nothing committed). Supersedes **Run 4** (which ran on a ~50% partial supplement). Produced a rebuilt combined matrix `staging/response_matrix_full.csv`, `staging/run5_*` fit outputs, and this memo.
**Date:** 2026-07-29
**Method:** `scripts/calibrate_mirt.py` (UNCHANGED — no fitter edit) on the rebuilt full combined matrix, with the same read-only slot-repurposing protocol as Runs 2–4. New helper `scripts/merge_full_matrix.py` builds the combined matrix from the existing big-run matrix + the supplement matrix (prefer-big-run hole-fill).

> **This time the data is COMPLETE.** The supplement (`supplement_v2/run_data_supp_v2.jsonl`, 74,329 rows) grades **all 662/662** style_surface presentation criteria + 3 `rescope_optional` + 244 previously-ungraded nonoptional hole-fills. The combined matrix is **98.1% filled** with **0 all-NaN columns**. N is still **82 persons** (< 150 identifiability floor), so the absolute-N caveat from Run 4 persists, but — see §3 — the two *specific* technical blockers that forced Run 4 to DEFER are now **resolved**.

---

## TL;DR (decision-ready)

- **Full-supplement coverage:** the supplement fills **909 columns** — 244 nonoptional hole-fills + **662/662 presentation** (`style_surface`) + 3 `rescope_optional`. Verdicts: fail 55,470 / pass 16,283 / no_decision 2,576 (→ NaN). 662/662 scenarios present.
- **Rebuilt combined matrix:** **82 × 6,845** (6,180 nonoptional + 665 optional). 550,893 observed (**98.1%**); pass 67,863 / fail 483,030 / NaN 10,397. Provenance: **479,140 cells from the big run, 71,753 from the supplement** (19,614 nonoptional hole-fills + 52,139 new optional cells). **Overlap between the two runs = 0 cells → 0 disagreements** (the 244 supplement-nonoptional criteria are all genuinely previously-ungraded holes). Every category is now fully populated: nonoptional 6,180/6,180, presentation 662/662, optional_other 3/3.
- **Presentation axis (the headline):** **The FULL data FLIPS the Run-4 verdict.** On complete data the **full 3-dim `[correctness, scaffolding, presentation]` is the AIC winner**, beating fold-into-scaffolding by **ΔAIC +1,238** and fold-into-correctness by **ΔAIC +1,772**. Run-4's under-identification red flag is **gone**: the full 3-dim loglik (−78,679.83) now cleanly **exceeds both nested folds** (−79,300.95 / −79,567.79), as a correct optimum must. Presentation is a healthy, high-variance, well-discriminating cluster (656/662 survive, pass-rate 0.01–0.70, median a = 1.09). → **Presentation now EARNS a separate third axis under AIC** (the program's operative criterion). Residual caveats: N = 82 still below the nominal 150 floor, and BIC still prefers unidimensional (a program-wide parsimony artifact at ~330k cells that also "rejects" the accepted correctness/scaffolding split).
- **2-skill refresh (core structure):** on the now-hole-filled nonoptional matrix the **collapsed `[correctness, scaffolding]` is the AIC winner**, beating the full 3-way `[content, diagnosis, scaffolding]` by **ΔAIC +2,674 / ΔBIC +16,331** and beating unidimensional on AIC (ΔAIC +1,210). Content↔diagnosis latent r = **0.942** (merge justified); collapsed correctness↔scaffolding latent r = **−0.462** (distinct). **The 2-skill `[correctness, scaffolding]` structure HOLDS on complete data.**

---

## 0. Method — slot-repurposing (no fitter edits), on the rebuilt combined matrix

`scripts/calibrate_mirt.py` reads each criterion's Q row as `[q_mapping.get(s,0) for s in SKILLS]` with `SKILLS = (content, diagnosis, scaffolding)` and is agnostic to what the slots MEAN. As in Runs 2–4 the experimental presentation bank repurposes the three slots:

| q_mapping slot (what the fitter reads) | packed skill |
|---|---|
| `content` | **correctness** = content OR diagnosis |
| `diagnosis` | **scaffolding** (unchanged bit) |
| `scaffolding` | **presentation** (optional `dimension=='style_surface'` criteria) |

So `a_content → a_correctness`, `a_diagnosis → a_scaffolding`, `a_scaffolding → a_presentation`, and the printed `latent correlation (content, diagnosis, scaffolding)` is actually **(correctness, scaffolding, presentation)**. Under this repurposing `--collapse diagnosis,scaffolding` folds presentation **INTO scaffolding** and `--collapse content,scaffolding` folds presentation **INTO correctness**.

**Presentation tag rule:** `presentation = 1` iff the curated record has `dimension == "style_surface"` (all 662 are `optional: True`). This is a **bank-structural** tag (not a heuristic keyword classifier), so it needs no hand audit. The 3 remaining optional rows are `rescope_optional` conditionals and are NOT presentation.

**No fitter edits.** `calibrate_partial.py` / `calibrate_mirt.py` are byte-for-byte unchanged. `--write-params` was NOT used. The official curated bank and `staging/response_matrix.csv` were NOT modified, and nothing was committed. The experimental presentation Q variant is written under **`staging/`** (not `data/`) this run, so no file under `data/` is touched.

**Helper scripts (read-only w.r.t. official data):**
- `scripts/ingest_calibration_csv.py`: reused unchanged (already supports `--jsonl`, `--include-optional/--full-bank`, `--matrix-basename`, `--audit-basename`, per-category/per-source fill).
- `scripts/build_candidate_qmatrix.py`: reused unchanged (`--select-dimension style_surface`, `--out` to a staging path).
- `scripts/merge_full_matrix.py`: **NEW** this run — merges the big-run nonoptional matrix with the supplement full-bank matrix into the combined matrix using prefer-big-run hole-fill semantics, and emits the provenance + disagreement audit. It only READS `staging/response_matrix.csv`.

---

## 1. Full-supplement coverage (`staging/response_matrix_supp.*`)

Ingested `supplement_v2/run_data_supp_v2.jsonl` over the **full 6,845-criterion** curated bank (`--include-optional`). `pass→1, fail→0, no_decision→NaN, absent→NaN`.

- **Supplement matrix:** 82 models × 6,845 criteria; **71,753 observed** cells (pass 16,283 / fail 55,470), **2,576 no_decision → NaN**, rest absent-NaN. Fill of the supplement-only matrix = 12.78% (it only carries the 909 supplement columns). 0 off-bank criteria.
- **Populated columns = 909:** nonoptional **244/6,180** (previously-ungraded hole-fills) + presentation `style_surface` **662/662** + optional_other (`rescope_optional`) **3/3**.
- **Scenarios present / curated = 662 / 662** (complete — vs Run 4's 638/662 overall, 427/662 for presentation specifically).
- `no_decision → NaN` confirmed: 16,283 + 55,470 + 2,576 = 74,329 source rows; the 2,576 no_decision are left NaN (indistinguishable from absent), never written as 0/1.

Artifacts: `staging/response_matrix_supp.csv` / `.npy` / `_manifest.json`, `staging/ingest_supp_audit.json` / `.md`, console `staging/ingest_supp_console.txt`.

---

## 2. Rebuilt combined matrix — provenance + disagreement audit (`staging/response_matrix_full.csv`)

Built by `scripts/merge_full_matrix.py` from the big-run nonoptional matrix (`staging/response_matrix.csv`, 82×6,180) + the supplement full-bank matrix (`staging/response_matrix_supp.csv`, 82×6,845). Rows in big-run order; columns in curated bank file order (all 6,845).

**Merge rule:** nonoptional columns — big-run cell wins wherever observed; the supplement only fills NaN holes. Optional columns — taken entirely from the supplement (new columns). On any model×criterion graded by BOTH and disagreeing, the big-run value is kept and the disagreement is logged.

- **Shape:** **82 × 6,845** = 561,290 cells.
- **Overall fill:** 550,893 observed (**98.1%**); pass **67,863** / fail **483,030** / NaN **10,397**. **0 all-NaN columns.**
- **Provenance (per-cell source):** big run **479,140** | supplement **71,753** | empty **10,397**.
  - supplement contribution = **19,614 nonoptional hole-fills** + **52,139 new optional cells**.
- **Disagreement audit:** cells graded by BOTH runs = **0** → **0 disagreements** (nothing to resolve). The 244 supplement-nonoptional criteria fill genuine holes that the big run never graded; there is no big-run↔supplement overlap, so "prefer big-run" never had to override anything.
- **Per-category fill (combined):** nonoptional **6,180/6,180**, presentation `style_surface` **662/662**, optional_other **3/3** — all fully populated.
- **Remaining NaN (10,397):** 8,006 nonoptional cells still ungraded by either run + 2,391 optional cells (sparse per-scenario presentation gaps). These are honest holes, marginalised natively by the EM.

Artifacts: `staging/response_matrix_full.csv` / `.npy` / `_manifest.json`, per-column provenance `staging/response_matrix_full_provenance.csv`, nonoptional-only slice `staging/response_matrix_full_nonoptional.csv`, console `staging/merge_full_console.txt`.

---

## 3. Presentation-axis calibration on FULL data (supersedes Run 4)

Fitted item block after selection (identical across the presentation collapse runs): **4,156 items × 82 persons**, 7 nodes/dim (343 grid nodes), **332,637 observed cells**. Dropped 2,049 all-fail zero-variance + 640 all-zero-Q. Fitted Q-pattern counts `[correctness, scaffolding, presentation]`: `100` = 2,704, `110` = 456, `010` = 340, **`001` = 656** → **656 items load presentation** (up from Run 4's 423).

### 3a. Presentation-dimension HEALTH

- **Assigned = 662; survived = 656; dropped = 6** (all-fail). With full grading, essentially every presentation item survives (vs 423/662 in Run 4).
- **Pass-rate across the 82-model fleet** (per-item mean), survived: **mean 0.264, median 0.247, min 0.012, max 0.699**; only **6** assigned items are all-fail. Genuine spread — neither near-floor nor near-ceiling.
- **Fitted discrimination `a_presentation`** (n=656): min −0.40, p25 0.82, **median 1.09**, p75 1.43, max 7.50, mean 1.30. **2 negative, 1 near-zero (|a|<0.2), 9 extreme (|a|>6).** A substantive, well-behaved cluster — nearly identical to Run 4's shape (median a 1.08) but now over 656 items instead of 423.
- Context (same fit): correctness `a` median 1.08 (n=3,160), scaffolding `a` median 0.43 (n=796).

### 3b. Latent correlation R (full 3-dim), order = (correctness, scaffolding, presentation)

|  | correctness | scaffolding | presentation |
|---|---:|---:|---:|
| **correctness** | 1.000 | −0.484 | **0.945** |
| **scaffolding** | −0.484 | 1.000 | −0.559 |
| **presentation** | **0.945** | −0.559 | 1.000 |

- **r(correctness, presentation) = +0.945** — high person-level collinearity with correctness, but **lower than Run 4's +0.979** and now estimated over the full 656-item cluster (well clear of the ±0.999 clip).
- `r(scaffolding, presentation) = −0.559`, `r(correctness, scaffolding) = −0.484` (the persistent negative correctness↔scaffolding seen in every run).

### 3c. SAME-ITEM-SET collapse table (item set = 4,156 items / 332,637 observed cells) — THE VERDICT

All four models are fit on the **identical** presentation item set (collapse is an in-memory Q-OR transform), so AIC/BIC are directly comparable. `uni` and `full-3-dim` are shared across both collapse runs.

| Model | dims | loglik | k | AIC | BIC |
|---|---:|---:|---:|---:|---:|
| Unidimensional | 1 | −80,665.29 | 8,312 | 177,954.59 | **267,016.07** |
| Fold presentation INTO **scaffolding** `[corr, scaff∪pres]` | 2 | −79,300.95 | 8,769 | 176,139.90 | 270,098.05 |
| Fold presentation INTO **correctness** `[corr∪pres, scaff]` | 2 | −79,567.79 | 8,769 | 176,673.57 | 270,631.71 |
| **Full 3-dim** `[corr, scaff, presentation]` | 3 | **−78,679.83** | 8,771 | **174,901.67** | 268,881.24 |

- **Full 3-dim vs fold into scaffolding:** ΔAIC **+1,238.24**, ΔBIC **+1,216.81** — **both favour the FULL 3-dim.** (Run 4: the fold WON AIC by +75.1. **Reversed.**)
- **Full 3-dim vs fold into correctness:** ΔAIC **+1,771.90**, ΔBIC **+1,750.47** — **both favour the FULL 3-dim.** Presentation is decidedly **not** interchangeable with correctness (consistent with Run 4, now even stronger).
- **AIC winner: full 3-dim. BIC winner: unidimensional.** (Δ convention: positive = favours full 3-dim, i.e. Δ = collapsed_metric − full_metric.)
- **Identifiability red flag RESOLVED.** In Run 4 the 3-dim loglik (−70,088.78) was *worse* than its own nested fold-into-scaffolding submodel (−70,053.25) — mathematically impossible at a true optimum, i.e. the 3-dim was stuck in a local optimum at ~50% coverage. On full data the 3-dim loglik (**−78,679.83**) **cleanly exceeds both** nested folds (−79,300.95, −79,567.79), as a correctly-optimised nesting model must. The full model now identifies.

### 3d. Compare to Run 4 + verdict

| | **Run 4 (partial, ~50%)** | **Run 5 (full)** |
|---|---|---|
| Presentation scenarios graded | 427 / 662 | **662 / 662** |
| Presentation items surviving fit | 423 | **656** |
| r(correctness, presentation) | +0.979 | **+0.945** |
| Full-3-dim vs fold-into-scaffolding (AIC) | fold wins (ΔAIC +75) | **full 3-dim wins (ΔAIC +1,238)** |
| Full-3-dim vs fold-into-correctness (AIC) | full wins (ΔAIC +561) | full wins (ΔAIC +1,772) |
| 3-dim vs its nested submodel | 3-dim WORSE (local optimum) | **3-dim BETTER (clean optimum)** |
| AIC winner | fold-into-scaffolding | **full 3-dim** |
| BIC winner | unidimensional | unidimensional |
| Verdict | **DEFER** | **EARNS a separate axis (under AIC)** |

**Verdict: the FULL data CHANGES the Run-4 verdict.** Run 4 deferred on two specific technical grounds — (a) folding presentation into scaffolding beat the 3-dim on AIC, and (b) the 3-dim was under-identified (fit below its own nested submodel). **Both are now resolved:** the full 3-dim is the clean AIC winner and dominates both nested folds. Presentation was already established as real, high-variance, and distinct from correctness in Run 4; on complete data it now **earns inclusion as a separable third axis by the AIC criterion** — the same criterion that certifies the accepted correctness/scaffolding split (see §4).

**Residual caveats (why this is a recommendation, not an auto-commit):**
1. **N = 82 is still below the nominal 150 identifiability floor** and the script still prints the warning. The fit is now a *clean* optimum (full > nested), which is the qualitative fix, but a powered N ≥ 150 confirmation would remove the last statistical caveat. Note the identical N = 82 underlies the accepted 2-skill decision.
2. **BIC still prefers unidimensional** everywhere — a program-wide parsimony artifact at ~330k cells that also "rejects" the accepted correctness↔scaffolding 2-skill split, so BIC is not the operative criterion in this program.
3. r(correctness, presentation) = 0.945 is high; presentation ability tracks correctness ability closely at the person level even though the confirmatory item-loading fold decisively rejects merging them.

**Recommendation:** promote presentation from "defer" to a **candidate third axis `[correctness, scaffolding, presentation]`**, pending the usual sign-off and (ideally) an N ≥ 150 confirmation. This is a read-only finding; no params were written.

### 3e. EFA
`--efa` requested but `factor_analyzer` is not installed → scree diagnostic skipped cleanly (as in Runs 2–4).

---

## 4. Core 2-skill refresh on the hole-filled nonoptional matrix

Rerun of the content+diagnosis collapse decision on `staging/response_matrix_full_nonoptional.csv` (82 × 6,180, the hole-filled nonoptional slice) with the **official curated bank** as the Q source. Fitted **3,497 items × 82 persons**, **280,943 observed cells**. Q-pattern counts `[content, diagnosis, scaffolding]`: `100`=1,228, `110`=1,159, `101`=302, `010`=314, `011`=20, `001`=340, `111`=134.

| Model | dims | loglik | k | AIC | BIC |
|---|---:|---:|---:|---:|---:|
| Unidimensional | 1 | −62,505.73 | 6,994 | 138,999.46 | **212,757.54** |
| **Collapsed `[correctness=content∪diagnosis, scaffolding]`** | 2 | **−61,443.76** | 7,451 | **137,789.53** | 216,367.08 |
| Full 3-way `[content, diagnosis, scaffolding]` | 3 | −61,485.92 | 8,746 | 140,463.84 | 232,698.34 |

- **Collapsed beats full 3-way:** AIC **True**, BIC **True** — ΔAIC **+2,674.31**, ΔBIC **+16,331.26** in favour of the collapse. Content and diagnosis should be merged into a single **correctness** dimension.
- **Collapsed beats unidimensional on AIC:** 137,789.53 < 138,999.46 (ΔAIC +1,209.93) — correctness and scaffolding **are** separable (2 dims beats 1).
- **AIC winner: collapsed 2-skill. BIC winner: unidimensional** (the same parsimony pattern as every run).
- **Latent correlations:** full 3-way `r(content, diagnosis) = 0.942` (near-collinear → merge justified), `r(content, scaffolding) = −0.253`, `r(diagnosis, scaffolding) = −0.210`. In the collapsed 2-dim model the **correctness↔scaffolding latent r = −0.462** (clearly distinct dimensions).

**The 2-skill `[correctness, scaffolding]` structure HOLDS on complete data** — content+diagnosis collapse is confirmed (r = 0.942; collapsed beats full by ΔAIC +2,674 / ΔBIC +16,331), and correctness vs scaffolding remain separable (collapsed beats unidim on AIC; latent r = −0.462).

---

## 5. Exact reproduce commands (read-only; no `--write-params`; official bank + `response_matrix.csv` never written)

Run from `eduLLM-Evals/` (Python: `..\.venv\Scripts\python.exe`):

```powershell
# STEP 1 — ingest the FULL supplement over the full 6,845-criterion bank -> NEW supp matrix
..\.venv\Scripts\python.exe scripts\ingest_calibration_csv.py `
  --jsonl supplement_v2\run_data_supp_v2.jsonl `
  --include-optional `
  --matrix-basename response_matrix_supp `
  --audit-basename ingest_supp_audit

# STEP 2 — rebuild the combined matrix (prefer-big-run hole-fill + provenance/disagreement audit)
#          reads staging\response_matrix.csv (big run) + staging\response_matrix_supp.csv; never writes response_matrix.csv
..\.venv\Scripts\python.exe scripts\merge_full_matrix.py

# STEP 3 — experimental presentation Q variant (presentation = bank style_surface), written under staging\
..\.venv\Scripts\python.exe scripts\build_candidate_qmatrix.py `
  --candidate presentation --select-dimension style_surface `
  --matrix staging\response_matrix_full.csv `
  --out staging\rubrics_qmatrix_collapse_presentation_full.jsonl

# STEP 4a — full 3-dim + fold-INTO-scaffolding + latent R (unmodified fitter, full combined matrix)
..\.venv\Scripts\python.exe scripts\calibrate_mirt.py `
  --rubrics staging\rubrics_qmatrix_collapse_presentation_full.jsonl `
  --matrix staging\response_matrix_full.csv `
  --collapse diagnosis,scaffolding --estimate-latent-corr --grid 7 --efa `
  --out-dir staging\run5_presentation

# STEP 4b — fold-INTO-correctness (same item set)
..\.venv\Scripts\python.exe scripts\calibrate_mirt.py `
  --rubrics staging\rubrics_qmatrix_collapse_presentation_full.jsonl `
  --matrix staging\response_matrix_full.csv `
  --collapse content,scaffolding --estimate-latent-corr --grid 7 `
  --out-dir staging\run5_presentation_corrmerge

# STEP 4c — presentation-dimension health
..\.venv\Scripts\python.exe scripts\analyze_candidate_dim.py `
  --candidate presentation `
  --bank staging\rubrics_qmatrix_collapse_presentation_full.jsonl `
  --matrix staging\response_matrix_full.csv `
  --csv staging\run5_presentation\calibration_mirt.csv

# STEP 5 — core 2-skill refresh on the hole-filled nonoptional matrix (official curated bank Q)
..\.venv\Scripts\python.exe scripts\calibrate_mirt.py `
  --rubrics data\TutorBench\curated\rubrics_qmatrix_curated.jsonl `
  --matrix staging\response_matrix_full_nonoptional.csv `
  --collapse content,diagnosis --estimate-latent-corr --grid 7 `
  --out-dir staging\run5_2skill
```

> Under the slot-repurposing, `--collapse diagnosis,scaffolding` folds presentation INTO scaffolding, and `--collapse content,scaffolding` folds presentation INTO correctness. In STEP 5 the collapse acts on the *real* skills, folding content+diagnosis into correctness.

---

## 6. Output paths

- `staging/response_matrix_supp.csv` / `.npy` / `_manifest.json` — full-supplement matrix (82×6,845, 909 populated columns); audit `staging/ingest_supp_audit.json` / `.md`; console `staging/ingest_supp_console.txt`.
- `staging/response_matrix_full.csv` / `.npy` / `_manifest.json` — **REBUILT** combined 82×6,845 matrix (overwrote the old partial Run-4 file). `staging/response_matrix.csv` **untouched**.
- `staging/response_matrix_full_provenance.csv` — per-column big/supp/combined observed counts, hole-fills, disagreements.
- `staging/response_matrix_full_nonoptional.csv` — 6,180 nonoptional columns of the combined (hole-filled) matrix, for the 2-skill refresh.
- `staging/merge_full_console.txt` — merge console.
- `staging/rubrics_qmatrix_collapse_presentation_full.jsonl` — experimental presentation Q variant (curated copy, `q_mapping` replaced), written **under `staging/`**. Official bank + `data/` NOT touched.
- `staging/run5_presentation/` — full 3-dim + fold-into-scaffolding fit (`calibration_mirt.csv`, `calibration_mirt_manifest.json`, `coverage_report.*`, `presentation_health.txt`); console `staging/run5_presentation_console.txt`.
- `staging/run5_presentation_corrmerge/` — fold-into-correctness fit; console `staging/run5_presentation_corrmerge_console.txt`.
- `staging/run5_2skill/` — core 2-skill refresh fit; console `staging/run5_2skill_console.txt`.
- `scripts/merge_full_matrix.py` — **NEW** combined-matrix merge helper (read-only w.r.t. inputs).
- `plans+prds/Calibration Run 5 - Full Supplement (presentation + 2-skill).md` — this memo.

**Fitter edits made: NONE.** `--write-params` NOT used. `data/TutorBench/curated/rubrics_qmatrix_curated.jsonl`, `data/TutorBench/experimental/*`, and `staging/response_matrix.csv` NOT modified. Nothing committed.
