# Calibration Run 6 — Presentation Axis (complete supplement)

**Status:** READ-ONLY gate re-test (NO params written; fitters + curated bank untouched; nothing committed). Outputs under `staging/run6_presentation/` (+ `staging/run6_presentation_corrmerge/`) and this memo.
**Date:** 2026-07-30
**Purpose:** Re-run the presentation-axis test on the **COMPLETE** supplement (`staging/response_matrix_full.csv`, 82×6,845, **0 all-NaN columns**) to confirm or overturn Run 4's **DEFER** verdict *before* freezing the 2-skill structure. This is a **gate**.
**Method:** `scripts/calibrate_mirt.py` (UNCHANGED), same read-only slot-repurposing protocol as Runs 2–4, same fit settings as Run 5 (grid 7 GH quadrature, ridge 1e-3, `--estimate-latent-corr`). Experimental presentation Q rebuilt from the current curated bank via `scripts/build_candidate_qmatrix.py`.

---

## TL;DR — the complete data OVERTURNS Run 4's defer rationale

- **Presentation earns a separate axis on complete data.** The full 3-dim `[correctness, scaffolding, presentation]` now **beats BOTH 2-skill folds on AIC *and* BIC** (vs fold-into-scaffolding ΔAIC **+1,238** / ΔBIC **+1,217**; vs fold-into-correctness ΔAIC **+1,772** / ΔBIC **+1,750**). In Run 4 (partial) fold-into-scaffolding *beat* the 3-dim (ΔAIC +75 / ΔBIC +96). **This is a clean reversal.**
- **The Run 4 identifiability red flag is GONE.** Run 4's 3-dim converged to a loglik *below* its own nested submodel (a local optimum → under-identified at N=82). Run 6's 3-dim loglik (**−78,679.83**) is now correctly **above** both nested folds (−79,300.95 / −79,567.79) → a clean optimum.
- **`presentation ↔ correctness = 0.945`** (Run 4: 0.979 — now *lower* and un-clipped/identified). **`presentation ↔ scaffolding = −0.559`** (Run 4: −0.481 — more negative/distinct). `correctness ↔ scaffolding = −0.484`.
- **Low-variance concern: resolved / never applied.** 656/662 presentation items survive, pass-rate spread **0.012–0.699** (median 0.247), fitted **median a_presentation = 1.09** (mean 1.30; only 2 negative, 1 near-zero, 9 extreme of 656). A healthy, strongly-discriminating cluster — not the near-floor Run-3 metacognition/communication case.
- **Cross-fold stability (k=5): `a_presentation` median 0.712 (min 0.630) — MORE stable than `a_scaffolding` (0.488) at the same N=82.** Presentation is *better* identified than the axis already in the instrument.
- **VERDICT: REVISIT — do not freeze presentation out.** The 2-skill *collapse* decision (content+diagnosis→correctness; scaffolding separate) is unchanged and reinforced (Run 5). But a presentation-**excluding** instrument is no longer justified: adopt presentation as a 3rd (optional) axis, structure `[correctness, scaffolding, presentation]`, to be **confirmed** at N≥150–200 rather than **gated** by it. See §5.

---

## 0. Method — slot-repurposing (no fitter edits) on the COMPLETE matrix

`scripts/calibrate_mirt.py` reads each Q row as `[q_mapping.get(s,0) for s in SKILLS]`, `SKILLS=(content, diagnosis, scaffolding)`, agnostic to slot meaning. The experimental bank repurposes the slots:

| q_mapping slot (fitter reads) | packed skill |
|---|---|
| `content` | **correctness** = content OR diagnosis |
| `diagnosis` | **scaffolding** (unchanged bit) |
| `scaffolding` | **presentation** (`dimension=='style_surface'`, 662 optional criteria) |

So `a_content→a_correctness`, `a_diagnosis→a_scaffolding`, `a_scaffolding→a_presentation`, and the printed latent correlation `(content, diagnosis, scaffolding)` is actually **(correctness, scaffolding, presentation)**. Under this repurposing `--collapse diagnosis,scaffolding` folds presentation **INTO scaffolding**, and `--collapse content,scaffolding` folds presentation **INTO correctness**.

**Presentation tag rule (bank-structural, no hand audit needed):** `presentation=1` iff curated `dimension=='style_surface'` — **662** criteria, all `optional:True`, all purely presentation (0 overlap with correctness/scaffolding: 662/662 "rescues"). The 3 remaining optional rows (`dimension=None`, `rescope_optional`) are NOT presentation. Rebuilt from the current curated bank; matches Run 4's set.

**Read-only guarantee.** Fitters and `data/TutorBench/curated/rubrics_qmatrix_curated.jsonl` byte-for-byte unchanged. `--write-params` NOT used. Nothing committed. The experimental Q was written under `staging/run6_presentation/` (not `data/`).

---

## 1. Data — the COMPLETE supplement

`staging/response_matrix_full.csv`: **82 models × 6,845 criteria**, **0 all-NaN columns** (98.1% fill, 550,893 observed cells). All 665 optional columns now graded (662 style_surface presentation + 3 rescope_optional). This is the completion of Run 4's ~50% partial matrix (which had 455 all-NaN columns; only 427/662 presentation graded).

**Fitted block (identical across all four models in the comparison):** **4,156 items × 82 persons**, 332,637 observed cells, 7 nodes/dim (343 grid nodes). Dropped 2,049 all-fail zero-variance + 640 all-zero-Q. Q-pattern counts `[correctness, scaffolding, presentation]`: `100`=2,704, `110`=456, `010`=340, **`001`=656** → **656 items load presentation.**

---

## 2. Presentation-axis HEALTH (complete data)

- **Assigned 662 → survived 656; dropped 6** (all-fail). Nearly the entire graded presentation pool survives (vs 423 in Run 4's partial).
- **Pass-rate across the 82-model fleet** (per-item mean), survived items: **mean 0.264, median 0.247, min 0.012, max 0.699.** Genuine spread — neither near-floor nor near-ceiling. Only 6 assigned items all-fail. **The low-variance problem (Run 3 metacognition/communication) does not apply to presentation, and complete grading confirms it.**
- **Fitted discrimination `a_presentation`** (n=656): min −0.399, p25 0.816, **median 1.090**, p75 1.430, max 7.499, mean 1.300. **2 negative, 1 near-zero (|a|<0.2), 9 extreme (|a|>6).** Strong, well-behaved discrimination.
- Context (same fit): correctness `a` median 1.08 (n=3,160), scaffolding `a` median 0.43 (n=796).

---

## 3. Latent correlations (full 3-dim), order = (correctness, scaffolding, presentation)

|  | correctness | scaffolding | presentation |
|---|---:|---:|---:|
| **correctness** | 1.000 | −0.484 | **0.945** |
| **scaffolding** | −0.484 | 1.000 | −0.559 |
| **presentation** | **0.945** | −0.559 | 1.000 |

- **`r(correctness, presentation) = +0.945`** — high person-level collinearity, but **lower than Run 4's 0.979** and cleanly identified (not clipped). Models good at correctness tend to be good at presentation, but less tightly than the partial data implied.
- **`r(scaffolding, presentation) = −0.559`** (Run 4: −0.481) — *more* negative; presentation and scaffolding are clearly distinct, and (as in Run 4) the negative sign argues against a literal presentation≈scaffolding identity even though scaffolding is the less-bad fold home.
- `r(correctness, scaffolding) = −0.484` (Run 5 definitive 2-skill: −0.462; Run 4: −0.452) — consistent.

> **Tension noted:** `r(correctness, presentation)=0.945` is about the same magnitude as `content↔diagnosis=0.942`, which we *did* collapse. The latent (person-ability) correlation and the confirmatory item-loading fold measure different things — and the fold is decisive: folding presentation into correctness is the **worst** model (§4), whereas collapsing content+diagnosis *improved* AIC. So the high latent r does **not** imply presentation should collapse into correctness.

---

## 4. SAME-ITEM-SET model comparison — THE VERDICT NUMBERS

All four models fit on the **identical** 4,156-item / 332,637-cell block (folds are in-memory Q-OR transforms), so AIC/BIC are directly comparable. Δ convention: **positive = favours full 3-dim** (Δ = collapsed_metric − full_metric).

| Model | dims | loglik | k | AIC | BIC |
|---|---:|---:|---:|---:|---:|
| Unidimensional | 1 | −80,665.29 | 8,312 | 177,954.59 | **267,016.07** |
| Fold presentation → **scaffolding** `[corr, scaff∪pres]` | 2 | −79,300.95 | 8,769 | 176,139.90 | 270,098.05 |
| Fold presentation → **correctness** `[corr∪pres, scaff]` | 2 | −79,567.79 | 8,769 | 176,673.57 | 270,631.71 |
| **Full 3-dim `[corr, scaff, presentation]`** | 3 | **−78,679.83** | 8,771 | **174,901.67** | 268,881.24 |

- **3-dim vs fold-into-scaffolding:** ΔAIC **+1,238.24**, ΔBIC **+1,216.81** — **both favour the 3-dim.** (Run 4: fold *won*, ΔAIC +75 / ΔBIC +96.)
- **3-dim vs fold-into-correctness:** ΔAIC **+1,771.90**, ΔBIC **+1,750.47** — **both strongly favour the 3-dim.** Presentation is not interchangeable with correctness.
- **AIC winner: full 3-dim. BIC winner: unidimensional.** BIC still prefers unidim under its heavy param penalty at N=82 (same pattern as Runs 1/5 — where BIC also preferred unidim over the accepted 2-skill model). So BIC-prefers-unidim is *not* a presentation-specific mark against it; the *nested* 3-dim-vs-fold BIC comparison favours the 3-dim.
- **Clean optimum:** the 3-dim loglik (−78,679.83) is now **above** both nested folds — the Run 4 "3-dim below its own submodel" under-identification artifact is resolved.

### Definitive 2-skill reference (Run 5, NOT same item set)
Run 5's `[correctness, scaffolding]` was fit on the **nonoptional slice** (3,497 items / 280,943 cells; presentation excluded) → AIC 137,789.53 / BIC 216,367.08, `r(corr,scaff)=−0.462`. Its raw AIC/BIC are **not** comparable to the run6 table (fewer items/cells). The apples-to-apples "2-skill vs 3-skill" test is the **fold-into-scaffolding row above** — i.e. `[correctness, scaffolding]` re-fit *with presentation items folded into scaffolding* on the same block — and the 3-dim beats it.

---

## 5. Cross-fold stability at N=82 (k=5, seed 20260729) — identifiability tiebreaker

Full 3-dim re-fit on each of 5 person-folds; median pairwise Pearson correlation of per-item loadings across folds:

| loading | median pairwise corr | min pairwise corr |
|---|---:|---:|
| `a_correctness` | 0.772 | 0.432 |
| `a_scaffolding` | 0.488 | 0.257 |
| **`a_presentation`** | **0.712** | **0.630** |

- **`a_presentation` (0.712) is far more stable than the already-accepted `a_scaffolding` (0.488)** and approaches `a_correctness` (0.772). Its *minimum* pairwise corr (0.630) is the **highest** of the three — the most consistent axis fold-to-fold.
- The correctness/scaffolding stabilities reproduce Run 5's definitive 2-skill k-fold (a_correctness 0.786, a_scaffolding 0.495), validating the driver.
- **Reading:** the N=82 under-identification that drove Run 4's defer is *not* the binding constraint for presentation — it is better identified than scaffolding, which we are freezing.

---

## 6. VERDICT & recommendation on the freeze

**Overturn Run 4's defer rationale.** Every pillar of the Run 4 "DEFER" call has flipped on complete data:

| Run 4 (partial, ~50%) | Run 6 (complete) |
|---|---|
| fold-into-scaffolding *beat* 3-dim (ΔAIC +75) | 3-dim *beats* both folds (ΔAIC +1,238 / +1,772) on AIC **and** BIC |
| 3-dim under-identified (loglik below nested submodel) | clean optimum (loglik above nested submodels) |
| r(corr,pres)=0.979 (near-collinear) | r=0.945 (lower, identified) |
| 423 items, "promising but unpowered" | 656 items; a_presentation cross-fold-stable (0.712 > scaffolding 0.488) |

**Does this change the 2-skill decision?** The **collapse** decision — content+diagnosis→correctness, scaffolding separate — is **unchanged and reinforced** (Run 5; content↔diagnosis still ≈0.94). Presentation lives in a *separate* (optional) criterion pool; adding it is **additive**, not a re-litigation of the skill collapse.

**Should it gate / block the pre-200 freeze?**
- It should **not block** freezing the **correctness+scaffolding skill core** — that structure stands on its own.
- It **should block** freezing a presentation-**excluding**, strictly-2-dimensional instrument as *final*. On complete data presentation is a separable, well-identified 3rd axis that is *more* stable than scaffolding; freezing it out would lock in a demonstrably worse-fitting structure.

**Recommendation:** Freeze the skill core now, but **adopt presentation as a 3rd (optional) axis** — treat the instrument structure as **`[correctness, scaffolding, presentation]`**. Use the N→200 expansion to **confirm/lock** presentation's item params (and to further harden scaffolding), **not** as a prerequisite for admitting the axis. The two residual caveats — high person-level r=0.945 with correctness (limited *incremental* person-ordering information for a CAT that already measures correctness) and N=82<150 — are reasons to (a) confirm at higher N and (b) weigh presentation's CAT value, **not** reasons to keep it deferred. If a single go/no-go is forced before N=200: **admit presentation** (statistical case is now decisive) with a flag to re-confirm loadings at N≥150.

---

## 7. Reproduce (read-only; no `--write-params`; bank + `response_matrix*.csv` never written)

Run from `eduLLM-Evals/` (Python: `..\.venv\Scripts\python.exe`):

```powershell
# 1 - experimental presentation Q from the CURRENT curated bank (style_surface = presentation)
..\.venv\Scripts\python.exe scripts\build_candidate_qmatrix.py `
  --candidate presentation --select-dimension style_surface `
  --matrix staging\response_matrix_full.csv `
  --out staging\run6_presentation\rubrics_qmatrix_collapse_presentation.jsonl

# 2 - full 3-dim + fold-INTO-scaffolding + latent R (grid 7, ridge, corr) on COMPLETE matrix
..\.venv\Scripts\python.exe scripts\calibrate_mirt.py `
  --rubrics staging\run6_presentation\rubrics_qmatrix_collapse_presentation.jsonl `
  --matrix staging\response_matrix_full.csv `
  --collapse diagnosis,scaffolding --estimate-latent-corr --grid 7 `
  --out-dir staging\run6_presentation

# 3 - fold-INTO-correctness (same item set)
..\.venv\Scripts\python.exe scripts\calibrate_mirt.py `
  --rubrics staging\run6_presentation\rubrics_qmatrix_collapse_presentation.jsonl `
  --matrix staging\response_matrix_full.csv `
  --collapse content,scaffolding --estimate-latent-corr --grid 7 `
  --out-dir staging\run6_presentation_corrmerge

# 4 - presentation-dimension health
..\.venv\Scripts\python.exe scripts\analyze_candidate_dim.py `
  --candidate presentation `
  --bank staging\run6_presentation\rubrics_qmatrix_collapse_presentation.jsonl `
  --matrix staging\response_matrix_full.csv `
  --csv staging\run6_presentation\calibration_mirt.csv

# 5 - cross-fold stability of a_presentation (k=5, full 3-dim per fold)
..\.venv\Scripts\python.exe staging\run6_presentation\kfold_presentation.py
```

> Under the slot-repurposing, `--collapse diagnosis,scaffolding` folds presentation INTO scaffolding; `--collapse content,scaffolding` folds presentation INTO correctness.

---

## 8. Output paths

- `staging/run6_presentation/rubrics_qmatrix_collapse_presentation.jsonl` — experimental presentation Q (curated copy, `q_mapping` replaced). **Official bank NOT touched.**
- `staging/run6_presentation/calibration_mirt.csv` (+ `_manifest.json`) — full 3-dim per-item params (`a_content/a_diagnosis/a_scaffolding` = `a_correctness/a_scaffolding/a_presentation`) + uni/fold-scaffolding/full 3-way comparison + latent R.
- `staging/run6_presentation_corrmerge/calibration_mirt.csv` (+ `_manifest.json`) — fold-into-correctness fit (same item set).
- `staging/run6_presentation/comparison.json` — machine-readable summary of every number above + verdict.
- `staging/run6_presentation/presentation_health.txt` — §2 health console.
- `staging/run6_presentation/kfold_presentation.py` + `kfold_presentation_stability.json` + `console_kfold.txt` — §5 cross-fold stability (read-only driver reusing `calibrate_mirt.py` internals).
- `staging/run6_presentation/console_scaffmerge.txt`, `console_corrmerge.txt` — full run consoles.
- `plans+prds/Calibration Run 6 - Presentation Axis (complete supplement).md` — this memo.

**Fitter edits: NONE. `--write-params`: NOT used. Curated bank + `response_matrix*.csv`: NOT modified. Nothing committed.**
