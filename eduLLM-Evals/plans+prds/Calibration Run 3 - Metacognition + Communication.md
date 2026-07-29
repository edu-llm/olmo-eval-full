# Calibration Run 3 — Metacognition + Communication

**Status:** read-only candidate test (NO params written; curated bank untouched). Produced `staging/run3_*` outputs + this memo.
**Date:** 2026-07-29
**Method:** `scripts/calibrate_mirt.py` (UNCHANGED — no source edit), on `staging/response_matrix.csv` with two EXPERIMENTAL Q-matrix variants. Same read-only protocol as Run 2 (motivation).

---

## TL;DR (decision-ready)

- **Two candidate axes tested, each as the 3rd latent skill in `[correctness, scaffolding, <candidate>]`** where `correctness = content OR diagnosis`, `scaffolding` = the existing bit. Candidate tags come from the heuristic text classifier `scripts/classify_orphan_criteria.py`.
- **Candidate 1 — metacognition** (`hit_metacognitive == 1`): **too low-variance to be a distinct axis.** Only **34** criteria tagged over the 6,180 fitted matrix columns; **20 survive** the fit; they are **near-floor** (survived pass-rate mean 0.094). Latent **r(correctness, metacognition) = +0.999** (pinned to the clip boundary — see caveat). Folding metacognition **into correctness improves both AIC and BIC** (ΔAIC −20.6, ΔBIC −209.7). **Verdict: redundant with correctness / too low-variance → DO NOT include.**
- **Candidate 2 — communication clarity** (`hit_communication_clarity == 1`): **also too low-variance on the fitted set.** The classifier tags **586** criteria in the curated bank, but **575 of them live in the all-zero (mostly optional / non-graded) pool** — on the 6,180 fitted matrix columns only **33** remain, **24 survive**. Latent **r(correctness, communication) = +0.999** (clip boundary). It *is* separable from scaffolding (folding into scaffolding costs ΔAIC +371 / ΔBIC +297), but folding **into correctness is a wash** (ΔAIC +3.4, ΔBIC −59.7). **Verdict: redundant with correctness / too low-variance on the graded item set → DO NOT include (defer, powered re-test).**
- **Overall:** neither candidate earns inclusion. Combined with Run 1 (content+diagnosis collapse = YES) and Run 2 (motivation = DEFER), **the interim 2-skill structure `[correctness, scaffolding]` still stands.**
- **N = 82 ≪ 150 identifiability floor** — everything below is provisional / machinery-exercise only, and for these two candidates the candidate dimension has only ~20–24 fitted items, so its latent correlation is **essentially unidentified** (both pin to the +0.999 clip).

---

## 0. Method — same slot-repurposing trick as Run 2 (no fitter edits)

`scripts/calibrate_mirt.py` reads each criterion's Q row as `[q_mapping.get(s,0) for s in SKILLS]` with `SKILLS = (content, diagnosis, scaffolding)`, and is otherwise **agnostic to what those slot names mean**. So, exactly as in Run 2, the experimental bank **repurposes the three slots**:

| q_mapping slot (what the fitter reads) | packed candidate skill |
|---|---|
| `content` | **correctness** = content OR diagnosis |
| `diagnosis` | **scaffolding** (unchanged bit) |
| `scaffolding` | **&lt;candidate&gt;** (classifier tag) |

So in every Run-3 output the columns map as `a_content → a_correctness`, `a_diagnosis → a_scaffolding`, `a_scaffolding → a_<candidate>`, and the printed `latent correlation (content, diagnosis, scaffolding)` is actually **(correctness, scaffolding, candidate)**. Under this repurposing, `--collapse diagnosis,scaffolding` merges **scaffolding+candidate** (fold the candidate INTO scaffolding) and `--collapse content,scaffolding` merges **correctness+candidate** (fold the candidate INTO correctness).

**No fitter edits were made.** `calibrate_partial.py` / `calibrate_mirt.py` are byte-for-byte unchanged. `--write-params` was NOT used. `data/curated/rubrics_qmatrix_curated.jsonl` was NOT modified.

**Generalized helpers.** The Run-2 motivation helpers were **parametrized** rather than rewritten:
- `scripts/build_candidate_qmatrix.py --candidate <name> --hit-col <classifier column>` — generalization of `build_motivation_qmatrix.py`.
- `scripts/analyze_candidate_dim.py --candidate <name> --bank <jsonl> --csv <fit csv>` — generalization of `analyze_motivation_dim.py`.

**Classifier tags used (report exactly which flag):** candidate 1 = **`hit_metacognitive`**, candidate 2 = **`hit_communication_clarity`**, from `staging/run2_orphan/orphan_criteria.csv` (classifier run on the curated bank; reused from Run 2). This is a HEURISTIC keyword tag — fine for a candidate test, would need a hand audit before any commit.

---

## Candidate 1 — METACOGNITION / self-regulation

### 1a. Experimental Q rule + item counts

`metacognition = 1` iff `hit_metacognitive == 1`. Over the curated bank (6,845 records) and, in parentheses, restricted to the **6,180 response-matrix columns** actually fitted:

| skill | criteria with bit = 1 (curated) | (matrix columns) |
|---|---:|---:|
| correctness (content OR diagnosis) | 5,029 | 5,026 |
| scaffolding | 1,142 | 1,142 |
| **metacognition** | **34** | **34** |
| all-zero Q | 1,411 | 749 |

Metacognition **rescues just 1** previously-all-zero criterion over the matrix columns. Fitted item set = **3,348 items / 269,018 observed cells**. Fitted Q-pattern counts (correctness | scaffolding | metacognition): `100`=2,597, `110`=422, `010`=309, `101`=12, `111`=4, `011`=3, `001`=1 → **only ~20 fitted items load metacognition at all.**

### 1b. Same-item-set collapse tests (item set = 3,348 items / 269,018 cells) — THE VERDICT

All three models below are fit on the **identical** metacognition item set (collapse is an in-memory Q transform), so AIC/BIC are directly comparable *within this table*.

| Model (all on the metacognition item set) | dims | loglik | k | AIC | BIC |
|---|---:|---:|---:|---:|---:|
| Unidimensional | 1 | −59,504.14 | 6,696 | 132,400.29 | **202,725.25** |
| Fold candidate INTO scaffolding `[correctness, scaff∪metacog]` | 2 | −58,475.76 | 7,135 | 131,221.53 | 206,157.11 |
| Fold candidate INTO correctness `[corr∪metacog, scaffolding]` | 2 | −58,473.04 | 7,126 | **131,198.07** | 206,039.13 |
| Full 3-dim `[correctness, scaffolding, metacognition]` | 3 | −58,465.35 | 7,144 | 131,218.71 | 206,248.81 |

- **3-dim vs fold into scaffolding:** ΔAIC **+2.8** (barely favors 3-dim), ΔBIC **−91.7** (favors the fold). Essentially a tie — no evidence metacognition needs its own axis apart from scaffolding.
- **3-dim vs fold into correctness:** ΔAIC **−20.6**, ΔBIC **−209.7** — **folding metacognition into correctness improves both criteria.** Metacognition is statistically redundant with correctness.
- **AIC winner across all candidates: fold-into-correctness. BIC winner: unidimensional** (heavy parsimony penalty at n≈269k cells, same pattern as Runs 1–2).

(Δ convention: positive = favors the full 3-dim; i.e. Δ = collapsed_metric − full_metric.)

### 1c. Latent correlation R (full 3-dim), order = (correctness, scaffolding, metacognition)

|  | correctness | scaffolding | metacognition |
|---|---:|---:|---:|
| **correctness** | 1.000 | −0.418 | **0.999** |
| **scaffolding** | −0.418 | 1.000 | −0.418 |
| **metacognition** | **0.999** | −0.418 | 1.000 |

- **r(correctness, metacognition) = +0.999** — pinned to the estimator's clip boundary (±0.999). With only ~20 items loading the 3rd dim at N=82, this correlation is **not identified**; treat it as "no separable signal," not a precise 0.999.
- r(scaffolding, metacognition) = −0.42; r(correctness, scaffolding) = −0.42 (consistent with Runs 1–2).

### 1d. Metacognition-dimension health

- **Assigned = 34; survived = 20; dropped = 14** (13 all-fail zero-variance).
- **Pass-rate:** assigned mean 0.057 (median 0.024); survived mean 0.094 (median 0.087), range 0.012–0.256 — **near-floor** (nearly every model fails these; 13 are all-fail).
- **Fitted discrimination `a_metacognition`** (n=20): min 0.29, median 0.77, mean 1.11, max 5.91; **0 negative, 0 near-zero, 0 extreme.** The individual items discriminate fine — there are just far too few of them.

### 1e. Verdict — metacognition

**DO NOT include (redundant with correctness / too low-variance).** Only ~20 fitted, near-floor items; folding into correctness *improves* fit; latent r pins to the clip. The item cluster is real but far too small and too correctness-collinear on this fleet to be an independent competency. **Defer** to a powered, more-metacognition-rich cohort if ever revisited — but the tag is so sparse (34/6,180) that a bigger *bank*, not just a bigger person sample, is the real prerequisite.

---

## Candidate 2 — COMMUNICATION CLARITY

### 2a. Experimental Q rule + item counts

`communication = 1` iff `hit_communication_clarity == 1`. Over the curated bank (6,845 records) and, restricted to the **6,180 fitted matrix columns**:

| skill | criteria with bit = 1 (curated) | (matrix columns) |
|---|---:|---:|
| correctness (content OR diagnosis) | 5,029 | 5,026 |
| scaffolding | 1,142 | 1,142 |
| **communication** | **586** | **33** |
| all-zero Q | 837 | 728 |

**Key structural finding:** the classifier tags 586 communication criteria in the *curated bank*, but **575 of them sit in the all-zero (largely optional / never-graded) pool** — they are **not columns of the response matrix**. On the 6,180 fitted columns only **33 communication criteria** exist, and it **rescues 22** previously-all-zero criteria. Fitted item set = **3,365 items / 270,386 observed cells**. Fitted Q-pattern counts (correctness | scaffolding | communication): `100`=2,608, `110`=423, `010`=310, `001`=18, `111`=3, `011`=2, `101`=1 → **only ~24 fitted items load communication.** So although communication looks populous in the bank, it is **near-empty on the graded matrix**.

### 2b. Same-item-set collapse tests (item set = 3,365 items / 270,386 cells) — THE VERDICT

| Model (all on the communication item set) | dims | loglik | k | AIC | BIC |
|---|---:|---:|---:|---:|---:|
| Unidimensional | 1 | −60,024.73 | 6,730 | 133,509.45 | **204,225.64** |
| Fold candidate INTO scaffolding `[correctness, scaff∪comm]` | 2 | −59,179.85 | 7,158 | 132,675.71 | 207,889.15 |
| Fold candidate INTO correctness `[corr∪comm, scaffolding]` | 2 | −58,995.11 | 7,159 | 132,308.23 | 207,532.18 |
| Full 3-dim `[correctness, scaffolding, communication]` | 3 | −58,987.43 | 7,165 | **132,304.87** | 207,591.86 |

- **3-dim vs fold into scaffolding:** ΔAIC **+370.8**, ΔBIC **+297.3** — the 3-dim clearly beats folding communication into scaffolding on both. **Communication is NOT scaffolding** (same qualitative result as motivation in Run 2).
- **3-dim vs fold into correctness:** ΔAIC **+3.4** (negligible), ΔBIC **−59.7** (favors the fold). **A wash / lean toward redundant** — communication does not earn a separate axis apart from correctness.
- **AIC winner across all candidates: full 3-dim (by a hair over fold-into-correctness, 3.4). BIC winner: unidimensional.**

### 2c. Latent correlation R (full 3-dim), order = (correctness, scaffolding, communication)

|  | correctness | scaffolding | communication |
|---|---:|---:|---:|
| **correctness** | 1.000 | −0.418 | **0.999** |
| **scaffolding** | −0.418 | 1.000 | −0.418 |
| **communication** | **0.999** | −0.418 | 1.000 |

- **r(correctness, communication) = +0.999** — again pinned to the clip boundary (only ~24 items load the 3rd dim). Not identified; read as "collinear with correctness / no separable signal."
- r(scaffolding, communication) = −0.42; r(correctness, scaffolding) = −0.42.

> Note the latent-R matrix is **numerically identical** across both candidates and both collapse directions. That is itself the tell: with a ~20–24-item 3rd dimension at N=82 the posterior second-moment estimate is driven entirely by the correctness/scaffolding block and pins the candidate to the +0.999 clip against correctness. Contrast Run 2's motivation, whose 485 surviving items yielded a genuine (un-clipped) r = 0.90.

### 2d. Communication-dimension health

- **Assigned = 33; survived = 24; dropped = 9** (8 all-fail).
- **Pass-rate:** assigned mean 0.151 (median 0.129); survived mean 0.202 (median 0.216), range 0.025–0.494 — low but with some spread (better than metacognition's near-floor).
- **Fitted discrimination `a_communication`** (n=24): min 0.08, median 0.77, mean 0.81, max 1.39; **0 negative, 1 near-zero, 0 extreme.** Items discriminate reasonably — but again there are only 24.

### 2e. Verdict — communication clarity

**DO NOT include now (defer / redundant with correctness on the graded set).** Communication is cleanly separable from scaffolding, but it is **not** separable from correctness (fold-into-correctness is a wash; r pins to the clip), and — decisively — the graded matrix contains only ~24 fittable communication items because the tag lands overwhelmingly on **optional, un-graded** criteria. To ever test communication properly you must first **grade the communication criteria** (get them into the response matrix), then re-fit at N≥150. Until then it cannot be an independent axis.

---

## Overall recommendation (across all candidates tested so far)

| Candidate | Run | fitted items (assigned→survived) | r(correctness, cand) | same-set verdict | decision |
|---|---|---:|---:|---|---|
| content ↔ diagnosis collapse | 1 | (full bank) | 0.946 | collapse improves fit | **ADOPT: merge → correctness** |
| motivation / affect | 2 | 605 → 485 | 0.90 (identified) | beats both folds on AIC; thin BIC vs correctness | **DEFER** (powered re-test) |
| metacognition | 3 | 34 → 20 | 0.999 (clip) | folding into correctness *improves* fit | **DO NOT include** (redundant / too sparse) |
| communication clarity | 3 | 33 → 24 | 0.999 (clip) | distinct from scaffolding, but wash vs correctness | **DEFER / do not include** (must grade the items first) |

**The interim 2-skill structure `[correctness, scaffolding]` still stands.** Content+diagnosis collapse into `correctness` (Run 1); keep `scaffolding` separate; motivation parked for a powered re-test (Run 2); metacognition and communication do not earn inclusion on this fleet (Run 3). No candidate tested so far dislodges the 2-skill instrument.

**Recurring identifiability caveat.** N = 82 ≪ 150. Every Run-3 manifest reports `identifiable: false`. For metacognition and communication the candidate dimension has only ~20–24 fitted items, so its latent correlation is essentially **unidentified** and pins to the ±0.999 clip — an even weaker basis than Run 2's motivation (r = 0.90 from 485 items). Treat all numbers as machinery-exercise / directional only.

---

## Exact reproduce commands (read-only; no `--write-params`; curated bank never written)

Run from `eduLLM-Evals/` (Python: `..\.venv\Scripts\python.exe`):

```powershell
# (classifier already refreshed on the curated bank in Run 2 -> staging/run2_orphan/;
#  re-run only if that is missing:)
# ..\.venv\Scripts\python.exe scripts\classify_orphan_criteria.py --rubrics data\curated\rubrics_qmatrix_curated.jsonl --out-dir staging\run2_orphan --structural-only

# ---- Candidate 1: metacognition ----
..\.venv\Scripts\python.exe scripts\build_candidate_qmatrix.py --candidate metacognition --hit-col hit_metacognitive
..\.venv\Scripts\python.exe scripts\calibrate_mirt.py --rubrics data\experimental\rubrics_qmatrix_collapse_metacognition.jsonl --collapse diagnosis,scaffolding --estimate-latent-corr --grid 7 --efa --out-dir staging\run3_metacognition
..\.venv\Scripts\python.exe scripts\calibrate_mirt.py --rubrics data\experimental\rubrics_qmatrix_collapse_metacognition.jsonl --collapse content,scaffolding --estimate-latent-corr --grid 7 --out-dir staging\run3_metacognition_corrmerge
..\.venv\Scripts\python.exe scripts\analyze_candidate_dim.py --candidate metacognition --bank data\experimental\rubrics_qmatrix_collapse_metacognition.jsonl --csv staging\run3_metacognition\calibration_mirt.csv

# ---- Candidate 2: communication clarity ----
..\.venv\Scripts\python.exe scripts\build_candidate_qmatrix.py --candidate communication --hit-col hit_communication_clarity
..\.venv\Scripts\python.exe scripts\calibrate_mirt.py --rubrics data\experimental\rubrics_qmatrix_collapse_communication.jsonl --collapse diagnosis,scaffolding --estimate-latent-corr --grid 7 --efa --out-dir staging\run3_communication
..\.venv\Scripts\python.exe scripts\calibrate_mirt.py --rubrics data\experimental\rubrics_qmatrix_collapse_communication.jsonl --collapse content,scaffolding --estimate-latent-corr --grid 7 --out-dir staging\run3_communication_corrmerge
..\.venv\Scripts\python.exe scripts\analyze_candidate_dim.py --candidate communication --bank data\experimental\rubrics_qmatrix_collapse_communication.jsonl --csv staging\run3_communication\calibration_mirt.csv
```

> `--collapse` names are the raw slot names; under the repurposing `--collapse diagnosis,scaffolding` = fold candidate INTO scaffolding, and `--collapse content,scaffolding` = fold candidate INTO correctness.
> EFA (`--efa`) is requested but `factor_analyzer` is not installed in this environment, so the scree diagnostic is skipped cleanly (as noted in each console log).

---

## Output paths

- `data/experimental/rubrics_qmatrix_collapse_metacognition.jsonl`, `data/experimental/rubrics_qmatrix_collapse_communication.jsonl` — experimental Q variants (curated copies, `q_mapping` replaced). **Curated bank NOT touched.**
- `scripts/build_candidate_qmatrix.py`, `scripts/analyze_candidate_dim.py` — new generalized helpers (no fitter edits).
- `staging/run3_metacognition/` — 3-dim fit + fold-into-scaffolding (`calibration_mirt.csv`, `calibration_mirt_manifest.json`, `coverage_report.*`, `metacognition_health.txt`); console `staging/run3_metacognition_console.txt`.
- `staging/run3_metacognition_corrmerge/` — fold-into-correctness test; console `staging/run3_metacognition_corrmerge_console.txt`.
- `staging/run3_communication/` — 3-dim fit + fold-into-scaffolding (+ `communication_health.txt`); console `staging/run3_communication_console.txt`.
- `staging/run3_communication_corrmerge/` — fold-into-correctness test; console `staging/run3_communication_corrmerge_console.txt`.
- `plans+prds/Calibration Run 3 - Metacognition + Communication.md` — this memo.

**Fitter edits made: NONE.** `--write-params` NOT used. `data/curated/rubrics_qmatrix_curated.jsonl` NOT modified.
