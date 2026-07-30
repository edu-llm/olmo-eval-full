# Calibration Run 4 — Presentation Axis (partial supplement)

**Status:** read-only candidate test (NO params written; official curated bank + `staging/response_matrix.csv` untouched). Produced a NEW combined matrix `staging/response_matrix_full.csv`, `staging/run4_presentation*` fit outputs, and this memo.
**Date:** 2026-07-29
**Method:** `scripts/calibrate_mirt.py` (UNCHANGED — no fitter edit), on the NEW combined matrix `staging/response_matrix_full.csv` with an EXPERIMENTAL presentation Q-matrix variant. Same read-only slot-repurposing protocol as Runs 2–3.

> **PARTIAL DATA WARNING (read first).** The presentation verdicts are a **~50% supplement** (`run_data_supp.jsonl`, 37,070 rows): only **427 of 662** style_surface presentation criteria are graded (**427/662 scenarios**), plus 24 recovered-nonoptional. **N = 82 persons ≪ 150** identifiability floor. `no_decision → NaN`. Every number below is **indicative / machinery-exercise only**, and — see §2 — the full 3-dim EM did **not** reach a clean optimum at this N. Do not treat this as the final presentation decision.

---

## TL;DR (decision-ready, with caveats)

- **Presentation is now a HEALTHY, high-variance item cluster** — the opposite of Run 3's metacognition/communication probes. **423 presentation items survive** the fit (vs the earlier 24), pass-rate **mean 0.266 / median 0.247, range 0.012–0.662** (real spread, not near-floor), and fitted discrimination is strong (**median a = 1.08, mean 1.27; 0 negative, 0 near-zero, 5 extreme/423**). So for the first time we can actually *test* presentation instead of guessing from ~24 items.
- **Latent r(correctness, presentation) = +0.979 — now IDENTIFIED** (not pinned to the ±0.999 clip that Run 3's 24-item proxies hit). Presentation ability is highly collinear with correctness ability at the person level. `r(scaffolding, presentation) = −0.481`, `r(correctness, scaffolding) = −0.452`.
- **Same-item-set collapse (the real verdict):**
  - **Fold presentation INTO scaffolding beats the full 3-dim** on both criteria (ΔAIC **+75.1**, ΔBIC **+96.3** in favour of the fold) → a separate presentation axis **does not earn its parameters** relative to scaffolding.
  - **Fold presentation INTO correctness is much WORSE than the 3-dim** (ΔAIC **+560.7**, ΔBIC **+539.5** in favour of 3-dim) → presentation is **clearly NOT the same as correctness** (despite the high latent r).
  - **AIC winner = fold-into-scaffolding; BIC winner = unidimensional** (heavy parsimony penalty at ~304k fitted cells, same pattern as Runs 1–3).
- **Identifiability red flag:** the full 3-dim converged to a loglik (−70,088.78) **below** the *nested* fold-into-scaffolding 2-dim model (−70,053.25). That is mathematically impossible at a global optimum → the 3-dim fit is stuck in a **local optimum / is under-identified** at N=82. The collapse verdict must be read through that lens.
- **Verdict:** on this partial data, **presentation does NOT earn inclusion as a separable third axis** (folding into scaffolding matches/beats it), but it is **not redundant with correctness** and — decisively different from Run 3 — **it is a real, well-behaved, high-variance competency cluster, not an empty tag.** This is **strong enough to promote presentation to a powered, full-data re-test**, and NOT strong enough to add it to the instrument now. **DEFER (test properly at full coverage + N ≥ 150).** The interim `[correctness, scaffolding]` structure still stands.

---

## 0. Method — slot-repurposing (no fitter edits), on the NEW combined matrix

`scripts/calibrate_mirt.py` reads each criterion's Q row as `[q_mapping.get(s,0) for s in SKILLS]` with `SKILLS = (content, diagnosis, scaffolding)` and is agnostic to what the slots mean. As in Runs 2–3 the experimental bank repurposes the three slots:

| q_mapping slot (what the fitter reads) | packed skill |
|---|---|
| `content` | **correctness** = content OR diagnosis |
| `diagnosis` | **scaffolding** (unchanged bit) |
| `scaffolding` | **presentation** (optional `dimension=='style_surface'` criteria) |

So `a_content → a_correctness`, `a_diagnosis → a_scaffolding`, `a_scaffolding → a_presentation`, and the printed `latent correlation (content, diagnosis, scaffolding)` is actually **(correctness, scaffolding, presentation)**. Under this repurposing `--collapse diagnosis,scaffolding` folds presentation **INTO scaffolding**, and `--collapse content,scaffolding` folds presentation **INTO correctness**.

**Presentation tag rule:** `presentation = 1` iff the curated record has `dimension == "style_surface"` (all 662 are `optional: True`; the 3 remaining optional rows are `rescope_optional` conditionals and are NOT presentation). This is a **bank-structural** tag (not a heuristic keyword classifier), so it needs no hand audit.

**No fitter edits.** `calibrate_partial.py` / `calibrate_mirt.py` are byte-for-byte unchanged. `--write-params` was NOT used. The official curated bank and `staging/response_matrix.csv` were NOT modified.

**Ingester / helper edits made (read-only w.r.t. official data):**
- `scripts/ingest_calibration_csv.py`: (a) `--jsonl` now accepts **multiple** files (merged into one matrix; pass/fail beats no_decision on any cross-file duplicate); (b) new `--include-optional/--full-bank` flag uses **all 6,845** curated criteria as columns (so presentation columns exist); (c) new `--matrix-basename` / `--audit-basename` so the combined run writes to **new** files without touching the existing `response_matrix.csv`; (d) new per-category / per-source fill report. The default (single `--jsonl`, nonoptional-only, `response_matrix.*`) behaviour is unchanged.
- `scripts/build_candidate_qmatrix.py`: new `--select-dimension` / `--select-optional` flags select the candidate directly from the curated bank (no classifier CSV needed); `--hit-col` is now optional (either/or).
- **Path fix:** both helpers already default to the post-reorg canonical bank `data/TutorBench/curated/rubrics_qmatrix_curated.jsonl`; the experimental variant is written under `data/TutorBench/experimental/` (the reorg's canonical experimental location, alongside the Run-3 variants). No old `data/curated/...` path was used.

---

## 1. Combined matrix provenance + fill (`staging/response_matrix_full.csv`)

Built by streaming **both** `run_data.jsonl` (485,615 rows) **and** `run_data_supp.jsonl` (37,070 rows) over the **full 6,845-criterion** curated bank (nonoptional + optional). `pass→1, fail→0, no_decision→NaN, absent→NaN`; 0 off-bank, 0 cross-file overlap.

- **Shape:** **82 models × 6,845 criteria** = 561,290 cells.
- **Overall fill:** 514,790 observed (**91.72%**); pass 60,747 / fail 454,043 / NaN 46,500 (of which no_decision = 7,895). **455 all-NaN columns.**
- **Populated columns: 6,390 of 6,845** (455 never graded).

| Category | Populated / total columns | Source | Notes |
|---|---:|---|---|
| Nonoptional | **5,960 / 6,180** | big run (5,936) + supp (24) | 220 nonoptional never graded |
| Optional presentation (`style_surface`) | **427 / 662** | supplement | the presentation axis; 235 not yet graded |
| Optional other (`rescope_optional`) | **3 / 3** | supplement | conditional re-derive criteria; not presentation |
| **Per source** | | | |
| `run_data` (big) | 5,936 populated | — | all nonoptional |
| `run_data_supp` (partial) | 454 populated | — | 24 nonoptional (recovered) + 427 presentation + 3 optional_other |

Scenarios present / curated overall = **638 / 662**; presentation coverage specifically = **427 / 662 scenarios** (~64.5%). This matches the stated supplement contract (430 of 665 optional graded; ~50% cutoff of the presentation grading).

---

## 2. Presentation-axis calibration test (mirror of Run 3)

Fitted item block after selection (identical across all models in the collapse table): **3,787 items × 82 persons**, 7 nodes/dim (343 grid nodes). Dropped 1,984 all-fail zero-variance + 619 all-zero-Q. Fitted Q-pattern counts `[correctness, scaffolding, presentation]`: `100`=2,616, `110`=431, `010`=317, **`001`=423** → **423 items load presentation.**

### 2a. Presentation-dimension HEALTH

- **Assigned (matrix columns) = 662; survived the fit = 423; dropped = 239** (235 all-NaN not-yet-graded + 4 all-fail). With 427 graded columns, **423 survive** — a few *hundred* items, exactly as hoped vs the earlier 24.
- **Pass-rate across the 82-model fleet** (per-item mean): survived **mean 0.266, median 0.247, min 0.012, max 0.662**; only **4** assigned items are all-fail. Genuine spread — presentation is neither near-floor (metacognition) nor near-ceiling.
- **Fitted discrimination `a_presentation`** (n=423): min 0.21, p25 0.79, **median 1.08**, p75 1.42, max 7.05, mean 1.27. **0 negative, 0 near-zero (|a|<0.2), 5 extreme (|a|>6).** The items discriminate well — this is a substantive, well-behaved cluster.
- Context (same fit): correctness `a` median 1.11 (n=3,047), scaffolding `a` median 0.49 (n=748).

### 2b. Latent correlation R (full 3-dim), order = (correctness, scaffolding, presentation)

|  | correctness | scaffolding | presentation |
|---|---:|---:|---:|
| **correctness** | 1.000 | −0.452 | **0.979** |
| **scaffolding** | −0.452 | 1.000 | −0.481 |
| **presentation** | **0.979** | −0.481 | 1.000 |

- **r(correctness, presentation) = +0.979 — IDENTIFIED** (below the ±0.999 clip). Unlike Run 3's metacognition/communication (both pinned to +0.999 off ~24 items), the 423-item presentation dimension yields a genuine, un-clipped estimate. High person-level collinearity with correctness.
- `r(scaffolding, presentation) = −0.481`, `r(correctness, scaffolding) = −0.452` (consistent negative correctness↔scaffolding seen in Runs 1–3).

### 2c. SAME-ITEM-SET collapse table (item set = 3,787 items / 303,540 observed cells) — THE VERDICT

All four models are fit on the **identical** presentation item set (collapse is an in-memory Q-OR transform), so AIC/BIC are directly comparable within this table. `uni` and `full-3-dim` are shared across both collapse runs (identical numbers).

| Model | dims | loglik | k | AIC | BIC |
|---|---:|---:|---:|---:|---:|
| Unidimensional | 1 | −71,226.43 | 7,574 | 157,600.85 | **238,061.49** |
| Fold presentation INTO **scaffolding** `[corr, scaff∪pres]` | 2 | −70,053.25 | 8,006 | **156,118.50** | 241,168.39 |
| Fold presentation INTO **correctness** `[corr∪pres, scaff]` | 2 | −70,371.14 | 8,006 | 156,754.27 | 241,804.16 |
| Full 3-dim `[corr, scaff, presentation]` | 3 | −70,088.78 | 8,008 | 156,193.56 | 241,264.70 |

- **3-dim vs fold into scaffolding:** ΔAIC **+75.1**, ΔBIC **+96.3** — **both favour the FOLD.** A separate presentation axis does not beat simply letting presentation items load scaffolding.
- **3-dim vs fold into correctness:** ΔAIC **−560.7**, ΔBIC **−539.5** — **both strongly favour the 3-dim.** Presentation is decidedly **not** interchangeable with correctness.
- **AIC winner: fold-into-scaffolding. BIC winner: unidimensional.** (Δ convention: positive = favours full 3-dim, i.e. Δ = collapsed_metric − full_metric.)

**Identifiability caveat on this table.** The full 3-dim loglik (−70,088.78) is **worse than** the *nested, more-constrained* fold-into-scaffolding model (−70,053.25). A correctly-optimised 3-dim model can never fit worse than a model it nests, so the 3-dim EM is sitting in a **local optimum** at N=82. Consequently "fold-into-scaffolding wins" is partly an artifact of the 3-dim's failure to identify — read it as "**no clear evidence presentation needs its own axis on this partial, under-powered data**," not as proof it belongs inside scaffolding (note the *negative* latent r(scaffolding, presentation) = −0.481 argues against a literal scaffolding≈presentation identity).

### 2d. EFA

`--efa` requested but `factor_analyzer` is not installed in this environment → scree diagnostic skipped cleanly (as in Runs 2–3).

---

## 3. Verdict

**DEFER — do not add presentation as a separable axis now; promote it to a powered, full-data re-test.**

1. **It is real and well-behaved** — 423 surviving items, healthy pass-rate spread (0.01–0.66), strong discrimination (median a 1.08, 0 negative/near-zero). This is a categorically stronger candidate than Run 3's metacognition (20 items) or communication (24 items), whose verdicts were "too sparse to test." Presentation is **testable**, and the machinery works end-to-end on real presentation data.
2. **It is NOT redundant with correctness** — folding into correctness costs ΔAIC +560 / ΔBIC +540, despite the high latent r = 0.979. (The latent-ability correlation and the confirmatory item-loading fold measure different things; the fold is decisive here.)
3. **A separate axis is NOT yet earned** — folding into scaffolding matches/beats the 3-dim (ΔAIC +75 / ΔBIC +96), and the 3-dim fit is under-identified (local optimum) at N=82.

**Is this partial result strong enough to decide?** It is strong enough to (a) **kill the "presentation is an empty/degenerate tag" worry** — it is not, unlike the Run-3 candidates — and (b) establish presentation is **distinct from correctness**. It is **NOT** strong enough to settle whether presentation is a genuine third axis or is absorbed by scaffolding: that hinges on a 3-dim fit that this N=82, ~50%-graded matrix cannot identify (the full model converged below its own nested submodel). **The remaining ~half of presentation grading (235 more criteria, → 662/662) and N ≥ 150 are required for a decisive call.** Recommendation: finish grading the presentation criteria, re-run this exact test on the completed matrix, and only then decide inclusion.

---

## 4. Exact reproduce commands (read-only; no `--write-params`; official bank + `response_matrix.csv` never written)

Run from `eduLLM-Evals/` (Python: `..\.venv\Scripts\python.exe`):

```powershell
# STEP 1 — combined matrix over the FULL bank from BOTH verdict files -> NEW files only
..\.venv\Scripts\python.exe scripts\ingest_calibration_csv.py `
  --jsonl run_data.jsonl run_data_supp.jsonl `
  --include-optional `
  --matrix-basename response_matrix_full `
  --audit-basename ingest_full_audit

# STEP 2a — experimental presentation Q variant (presentation = bank style_surface)
..\.venv\Scripts\python.exe scripts\build_candidate_qmatrix.py `
  --candidate presentation --select-dimension style_surface `
  --matrix staging\response_matrix_full.csv

# STEP 2b — full 3-dim + fold-INTO-scaffolding + latent R (unmodified fitter, full matrix)
..\.venv\Scripts\python.exe scripts\calibrate_mirt.py `
  --rubrics data\TutorBench\experimental\rubrics_qmatrix_collapse_presentation.jsonl `
  --matrix staging\response_matrix_full.csv `
  --collapse diagnosis,scaffolding --estimate-latent-corr --grid 7 --efa `
  --out-dir staging\run4_presentation

# STEP 2c — fold-INTO-correctness (same item set)
..\.venv\Scripts\python.exe scripts\calibrate_mirt.py `
  --rubrics data\TutorBench\experimental\rubrics_qmatrix_collapse_presentation.jsonl `
  --matrix staging\response_matrix_full.csv `
  --collapse content,scaffolding --estimate-latent-corr --grid 7 `
  --out-dir staging\run4_presentation_corrmerge

# STEP 2d — presentation-dimension health
..\.venv\Scripts\python.exe scripts\analyze_candidate_dim.py `
  --candidate presentation `
  --bank data\TutorBench\experimental\rubrics_qmatrix_collapse_presentation.jsonl `
  --matrix staging\response_matrix_full.csv `
  --csv staging\run4_presentation\calibration_mirt.csv
```

> Under the slot-repurposing, `--collapse diagnosis,scaffolding` folds presentation INTO scaffolding, and `--collapse content,scaffolding` folds presentation INTO correctness.

---

## 5. Output paths

- `staging/response_matrix_full.csv` / `.npy` / `response_matrix_full_manifest.json` — **NEW** combined 82×6,845 matrix (big + supplement, full bank). Existing `staging/response_matrix.csv` **untouched**.
- `staging/ingest_full_audit.json` / `ingest_full_audit.md` — combined-ingest audit + per-category / per-source fill.
- `data/TutorBench/experimental/rubrics_qmatrix_collapse_presentation.jsonl` — experimental presentation Q variant (curated copy, `q_mapping` replaced). **Official bank NOT touched.**
- `staging/run4_presentation/` — full 3-dim + fold-into-scaffolding fit (`calibration_mirt.csv`, `calibration_mirt_manifest.json`, `coverage_report.*`, `presentation_health.txt`); console `staging/run4_presentation_console.txt`.
- `staging/run4_presentation_corrmerge/` — fold-into-correctness fit; console `staging/run4_presentation_corrmerge_console.txt`.
- `scripts/ingest_calibration_csv.py`, `scripts/build_candidate_qmatrix.py` — extended (multi-jsonl / full-bank / bank-select); **no fitter edits.**
- `plans+prds/Calibration Run 4 - Presentation Axis (partial supplement).md` — this memo.

**Fitter edits made: NONE.** `--write-params` NOT used. `data/TutorBench/curated/rubrics_qmatrix_curated.jsonl` and `staging/response_matrix.csv` NOT modified.
