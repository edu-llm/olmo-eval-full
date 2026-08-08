# Ablation Pre-Registration — 200-run attribution (leave-one-out)

**Status:** DESIGN / PRE-REGISTRATION ONLY. No harness code, no fits, nothing committed.
**Date:** 2026-07-31
**Author context:** written BEFORE the 200-model run exists, so the attribution rules
below cannot be reverse-engineered from results.

**Purpose in one line:** the pilot baseline bundled ~5 changes at once; after the 200-run
we want a table that says *"change X moved metric Y by Δ"* for each change, honestly.

---

## 1. Purpose + scope

We froze several changes into the pilot baseline simultaneously (ridge 1e-2, exclude-60,
CAT policy, 2-vs-3-skill presentation, Gate-B bank edits). After the 200-model calibration
run it will be hard to attribute any observed movement to a specific change. Every bundled
change is a toggle or a scalar parameter, so we can isolate each one's **marginal effect**
with a **leave-one-out (LOO)** ablation: start from the full "all-on" baseline and revert
exactly one factor at a time.

**Why LOO, not full factorial.** With `k` factors, LOO costs `k + 1` runs (baseline +
one per reverted factor), optionally `+1` for an all-off reference. A full factorial costs
`2^k` runs. Here `k = 5` → LOO is **6–7 runs**; factorial is **32 runs**. Each run is a fit
+ k-fold CV + CAT eval, so factorial is ~5× the compute for interaction structure we mostly
don't need. LOO answers the actual question ("what did each change buy at the current
operating point?"). The two interactions we *do* care about get a small dedicated 2×2 (§7).

**Scope guardrails.** Read-only w.r.t. the authoritative bank and the stock fitters, exactly
as Runs 5/6, the CAT-policy work, and the scaffolding-hygiene audit were. All ablation
artifacts land under a fresh `staging/ablation/` tree. Nothing overwrites
`data/TutorBench/curated/rubrics_qmatrix_curated.jsonl`,
`data/rubrics_qmatrix_final.jsonl`, or `config.yaml`.

---

## 2. Baseline ("all-on") config — exact commands

The pilot baseline is the **2-skill** instrument (Run 5 definitive collapse); the **3-skill**
instrument (Run 6 presentation axis) is carried as a reported presentation dimension (factor
D). All commands below are templated on the **pilot** inputs; the only thing that changes for
the real ablation is the response-matrix path, which becomes the **200-run matrix**. Those
substitution points are marked `<<200-run …>>`.

**Pilot response-matrix inputs (from Run 5/6), which become the 200-run matrices:**
- 2-skill (nonoptional slice): `staging/response_matrix_full_nonopt.csv` (pilot: 82 × 6,180)
  → `<<200-run nonopt slice>>`
- 3-skill (full, incl. presentation): `staging/response_matrix_full.csv` (pilot: 82 × 6,845)
  → `<<200-run full matrix>>`

**Frozen shared settings (all-on):** grid 7 (GH quadrature), `--estimate-latent-corr` ON for
the bank fit, EM `max_iter 200` / `tol 1e-4`, **ridge 1e-2**, k-fold `k=5 seed 20260729`
(corr OFF), CAT `--se-target 0.3 --max-items 100 --min-items 1 --grid 7 --seed 20260730`,
CAT policy ON with **floor 15** / pbis 0.05 / content-balance ON / exposure-top-n 5.

### 2a. Baseline bank fit — 2-skill (collapse content+diagnosis → correctness)

The CAT 2-skill harness reads a bank CSV with columns `a_correctness, a_scaffolding, b`
(`skillset_config(2)` in `scripts/cat_eval_tutorbench_multiskill.py`). The **stock**
`scripts/calibrate_mirt.py --collapse content,diagnosis` writes the *full 3-dim* CSV
(`a_content, a_diagnosis, a_scaffolding`), NOT that 2-skill schema — so the 2-skill bank is
produced by the collapse **driver** that imports `calibrate_mirt` internals
(`prepare_block` → `collapse_q_matrix(["content","diagnosis"])` → `fit_m2pl_em`). The
existing `staging/scaff_hygiene/` driver (`scaff_common.fit_bank`, invoked by
`staging/scaff_hygiene/experiment.py`) already does exactly this and already parameterizes
`ridge`, so it is the natural template.

Conceptual command (driver, ridge explicit for provenance even though 1e-2 is now the fit default):

```powershell
..\.venv\Scripts\python.exe <collapse-2skill driver> `
  --matrix <<200-run nonopt slice>> `
  --rubrics data\TutorBench\curated\rubrics_qmatrix_curated.jsonl `
  --collapse content,diagnosis --estimate-latent-corr --grid 7 --ridge 1e-2 `
  --out staging\ablation\baseline_2skill\bank_baseline.csv
```

Stock 3-way comparison manifest (uni / collapsed-2 / full-3), for the model-selection numbers:

```powershell
..\.venv\Scripts\python.exe scripts\calibrate_mirt.py `
  --matrix <<200-run nonopt slice>> `
  --rubrics data\TutorBench\curated\rubrics_qmatrix_curated.jsonl `
  --collapse content,diagnosis --estimate-latent-corr --grid 7 --ridge 1e-2 `
  --out-dir staging\ablation\baseline_2skill_official
```

### 2b. Baseline bank fit — 3-skill (presentation axis; reported dimension, factor D)

Uses the presentation-repurposed Q built from the curated bank
(`scripts/build_candidate_qmatrix.py --candidate presentation --select-dimension
style_surface`, → `staging/run6_presentation/rubrics_qmatrix_collapse_presentation.jsonl`).
The stock `calibrate_mirt.py` CSV (`a_content/a_diagnosis/a_scaffolding` = correctness /
scaffolding / presentation under Run 6 slot-repurposing) is consumed directly by the 3-skill
CAT harness.

```powershell
..\.venv\Scripts\python.exe scripts\calibrate_mirt.py `
  --rubrics staging\run6_presentation\rubrics_qmatrix_collapse_presentation.jsonl `
  --matrix <<200-run full matrix>> `
  --estimate-latent-corr --grid 7 --ridge 1e-2 `
  --out-dir staging\ablation\baseline_3skill
```

### 2c. Baseline k-fold CV — 2-skill

> **CRITICAL PREP:** `scripts/kfold_cv_mirt.py` still has `--ridge` **default 1e-3**
> (line 545), whereas `scripts/calibrate_mirt.py` default is now **1e-2** (line 886). The
> k-fold harness does **not** inherit the new default. You **must** pass `--ridge 1e-2`
> explicitly, or the k-fold OOS numbers silently belong to factor-A-reverted, not baseline.

```powershell
..\.venv\Scripts\python.exe scripts\kfold_cv_mirt.py `
  --matrix <<200-run nonopt slice>> `
  --rubrics data\TutorBench\curated\rubrics_qmatrix_curated.jsonl `
  --k 5 --seed 20260729 --grid 7 --ridge 1e-2 `
  --out-dir staging\ablation\kfold_baseline
```

### 2d. Baseline CAT eval — 2-skill (policy ON, floor 15)

```powershell
..\.venv\Scripts\python.exe scripts\cat_eval_tutorbench_multiskill.py `
  --skills 2 `
  --bank staging\ablation\baseline_2skill\bank_baseline.csv `
  --matrix <<200-run nonopt slice>> `
  --kfold-dir staging\ablation\kfold_baseline `
  --curated data\TutorBench\curated\rubrics_qmatrix_curated.jsonl `
  --se-target 0.3 --max-items 100 --min-items 1 --grid 7 --seed 20260730 `
  --min-items-per-skill 15 --min-point-biserial 0.05 --exposure-top-n 5 `
  --out-dir staging\ablation\cat_baseline
```

(Policy knobs default ON; floor default is already 15. They are written out explicitly so
each LOO row is a one-token diff off this line.)

---

## 3. Leave-one-out run matrix

Each row = the **baseline** with **exactly one** factor reverted. "Delta vs baseline" is the
precise flag/data change; everything else stays at the §2 all-on values. Factor D is a
reported *mode* (2- vs 3-skill), not a strict revert, so it is marked accordingly.

| Row | Factor reverted | Exact flag / data delta vs all-on baseline | Touches |
|---|---|---|---|
| **R0** | — (all-on baseline) | none — ridge 1e-2, exclude-60 ON, policy ON (floor 15), 2-skill collapse, Gate-B (curated) bank | fit + kfold + CAT |
| **A** | ridge → 1e-3 (was 1e-2) | fit driver `--ridge 1e-3`; **k-fold `--ridge 1e-3`**; (scaff_common driver `ridge=1e-3`) | fit + kfold |
| **B** | exclude-60 OFF (fit **with** the 60 items) | fit `--ignore-exclude-flag` (**MUST BE ADDED**, see §6); k-fold `--ignore-exclude-flag`; CAT `--curated <bank w/o exclude flags>` (or a `--no-exclude` switch, must add) | fit + kfold + CAT |
| **C** | CAT policy → legacy/off | CAT harness `--legacy` (all 5 knobs off: pbis off, floor 0, content-balance off, exposure-top-n 1; deterministic tie-break stays on) | CAT only |
| **D** | presentation mode: 3-skill instead of 2 *(reported dimension, not strict LOO)* | fit §2b (3-skill Q + full matrix); CAT `--skills 3` + 3-skill bank + `<<200-run full matrix>>` | fit + CAT |
| **E** | Gate-B bank edits reverted (2 text rewords + 5 scaffolding relabels) | fit/kfold/CAT `--rubrics <pre-Gate-B bank snapshot>`; see §6 for the git recipe **and the B/E entanglement caveat** | fit + kfold + CAT |
| **R_off** | *(optional)* all-off reference | ridge 1e-3 **+** exclude OFF **+** `--legacy` **+** pre-Gate-B bank, simultaneously | fit + kfold + CAT |

**Reading the matrix.** Compare each of A, B, C, E against **R0** to read that factor's
marginal effect. D is compared as a *pair* (R0 2-skill vs the 3-skill run) and reported as a
dimension/column, not folded into the single-factor attribution column. R_off is a sanity
anchor: if the sum of the individual LOO deltas is far from `(R0 − R_off)`, interactions are
large and the §7 2×2 matters.

---

## 4. Metrics captured per run (and where each number comes from)

For every row, capture the following. Sources are the exact JSON/CSV each script writes.

| Metric | Source (file → field) |
|---|---|
| **OOS k-fold recovery r** — correctness, scaffolding (+ presentation if 3-skill) | CAT eval `cat_metrics.json` → `out_of_sample.recovery_r.{correctness,scaffolding[,presentation]}` (written by `run_oos`/`aggregate`) |
| In-sample recovery r (context, not headline) | `cat_metrics.json` → `in_sample.recovery_r.*` |
| **Cross-fold parameter stability** (median pairwise r of a_correctness, a_scaffolding, b) | k-fold `item_param_stability.csv` (`median_pairwise_corr` per parameter) / `item_param_stability.json` (`stability.*`) |
| **# positive / # negative scaffolding loaders** | bank CSV (`bank_*.csv`): count `a_scaffolding > 0` vs `< 0`. (Audit template: `staging/scaff_hygiene/experiment.py` prints these.) |
| **# extreme-a (`\|a\| ≥ 6`)** | bank CSV: count `\|a_scaffolding\| ≥ 6.0` (`EXTREME_A = 6.0` in `calibrate_mirt.py`); also surfaced via the `flags` column (`extreme_a`) |
| **Latent correlation** (correctness↔scaffolding; 3-skill: full 3×3) | fit manifest `calibration_mirt_manifest.json` → `latent_correlation` (needs `--estimate-latent-corr`) |
| **CAT median test length** (overall + per skill) | `cat_metrics.json` → `in_sample.cat_items.{overall,<skill>}.{median,mean}` |
| **Per-skill items-administered** (mean / median / min) | `cat_metrics.json` → `in_sample.items_administered_by_skill.<skill>.{mean,median,min}` |
| **CAT convergence / N** (models reaching SE<0.3 after floor, of N) | `cat_metrics.json` → `in_sample.cat_items.<skill>.converged` and `n_models`; per-model detail in `cat_per_model.csv` |
| **pIRT calibration MAE** | `cat_metrics.json` → `in_sample.pirt_mae_cat` (ceiling ref: `pirt_mae_full`) |
| Model-selection (loglik / AIC / BIC; uni vs collapsed vs full) | fit manifest → `comparison.*` and `collapse.three_way.*` |
| OOS k-fold pooled cell metrics (logloss/acc/AUC/Brier) | k-fold `metrics_aggregate.json` → `pooled_oos.*`, `oos_minus_insample_gap.*` |

Notes:
- The **primary decision metric** (per the Scaffolding-Hygiene precedent) is **OOS k-fold CAT
  scaffolding recovery r**, with **OOS correctness recovery as the guardrail** (must not
  regress). Everything else is supporting.
- "recovery r (scaffolding)" from the CAT harness and "cross-fold a_scaffolding stability"
  from the k-fold harness are *different* numbers measuring related things — capture both.

---

## 5. Attribution table — final deliverable format (mocked)

The deliverable is one table: **rows = factors**, **columns = Δ(metric) vs the all-on
baseline** (reverted-minus-baseline). Each change's marginal contribution is a single row.
Δ sign convention: **Δ = (factor-reverted value) − (all-on baseline value)**, so a *negative*
Δ on OOS scaffolding recovery means "reverting this factor makes scaffolding worse" ⇒ the
factor was *helping* ⇒ evidence to KEEP it.

**Absolute reference (baseline R0) — mocked placeholders:**

| Baseline R0 (all-on) | OOS scaff r | OOS corr r | stab a_scaff | #neg scaff | #extreme-a | latent r(corr,scaff) | CAT median len | pIRT MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| value | 0.667 | 0.931 | 0.618 | 326 | 6 | −0.462 | 24 | 0.0341 |

**Attribution table (Δ vs R0) — mocked placeholders, illustrating output shape only:**

| Factor reverted | ΔOOS scaff r | ΔOOS corr r | Δstab a_scaff | Δ#neg scaff | Δ#extreme-a | Δlatent r | ΔCAT median len | ΔpIRT MAE | KEEP/REVERT (pre-registered §8) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **A** ridge→1e-3 | −0.131 | −0.015 | −0.060 | −33 | +16 | +0.044 | +0 | +0.001 | KEEP 1e-2 unless plateau moved |
| **B** exclude-60 OFF | −0.02 | −0.00 | −0.01 | +40 | +5 | +0.01 | +1 | +0.000 | KEEP exclude unless items gain variance |
| **C** policy→legacy | +0.14* | +0.00 | 0 | 0 | 0 | 0 | −18 | +0.000 | KEEP policy (legacy r is deceptive) |
| **D** 3-skill (mode) | *(reported separately)* | | | | | | | | dimension column, not a keep/revert |
| **E** pre-Gate-B bank | ~0.00 | ~0.00 | ~0.00 | +0/−0 | +0 | ~0.00 | +0 | ~0.000 | KEEP (negligible: 2 edits+5 relabels) |

\* For factor C the "+0.14" on scaffolding is the *deceptive* legacy lift documented in the
CAT-policy memo (legacy scaffolding "converges" on as few as 1 item). A positive Δ here is
**not** a reason to revert — see the §8 decision rule for C. This is exactly the kind of
sign trap the pre-registered rules exist to defuse.

(All numbers above are placeholders showing the intended shape; real values come from §4
sources after the 200-run.)

---

## 6. Toggles / prep needed before the run

| # | Prep item | Factor | Status | Where |
|---|---|---|---|---|
| 1 | Add `--ignore-exclude-flag` to the fitter | B | **MUST ADD** | `scripts/calibrate_mirt.py::load_q_matrix` — the `if rec.get("exclude_from_fit"): continue` at **lines 185–186** is the only gate. Thread a param `ignore_exclude=False` through `load_q_matrix(rubrics_path, ignore_exclude=...)` and register a `--ignore-exclude-flag` argparse arg (`action="store_true"`, default off) in `main()`. When set, do **not** `continue` on the flag. |
| 2 | Thread the same flag into k-fold | B | **MUST ADD** | `scripts/kfold_cv_mirt.py` calls `cm.load_q_matrix(args.rubrics)` (line 328) — add a matching `--ignore-exclude-flag` to its argparser and pass it through. |
| 3 | Bypass CAT bank exclusion for the B row | B | **MUST ADD (or work around)** | `scripts/cat_eval_tutorbench_multiskill.py::load_excluded_criteria` (lines 192–206) + main (lines 960–967) unconditionally drop `exclude_from_fit` items from the CAT bank. Either add a `--no-exclude`/`--ignore-exclude-flag` switch here, or (workaround, no code) point `--curated` at a copy of the curated bank with the 60 flags stripped. |
| 4 | Obtain pre-Gate-B bank snapshot | E | **RECIPE ONLY (do not fetch now)** | See git recipe + caveat below. |
| 5 | Confirm `--ridge` exists on the fitter | A | **ALREADY POSSIBLE** | `calibrate_mirt.py` `--ridge` default **1e-2** (line 886). |
| 6 | Confirm `--ridge` on k-fold + pass 1e-2 explicitly | A | **ALREADY POSSIBLE, but default is 1e-3** | `kfold_cv_mirt.py` `--ridge` default **1e-3** (line 545). Baseline runs must pass `--ridge 1e-2`; A-row passes `--ridge 1e-3`. |
| 7 | Confirm `--legacy` exists | C | **ALREADY POSSIBLE** | `cat_eval_tutorbench_multiskill.py` `--legacy` (line 918) → `Policy.legacy()`. Per-knob flags also exist: `--min-items-per-skill`, `--no-pbis-filter`/`--min-point-biserial`, `--no-content-balance`, `--exposure-top-n`. |
| 8 | Confirm `--skills` / `--collapse` exist | D | **ALREADY POSSIBLE** | fitter `--collapse content,diagnosis` (line 879); CAT `--skills {2,3}` (line 901). |

**Factor E — pre-Gate-B snapshot git recipe (documented, not run):**

Git history for the curated bank has exactly two commits:
- `4137990` ("edubench") — **PRE-Gate-B**: curated bank has **0** `exclude_from_fit` flags.
- `306c86e` ("stuff", = HEAD) — **POST-Gate-B**: 60 exclude flags + the 2 text rewords
  (`tb_0532_c06`, `tb_0370_c13`) + the 5 scaffolding relabels. Working tree == HEAD (clean).

Recipe to materialize the snapshot to a scratch file (from repo root
`C:\Users\Samee\Documents\GitHub\olmo-eval-full`):

```powershell
git show 4137990:eduLLM-Evals/data/TutorBench/curated/rubrics_qmatrix_curated.jsonl `
  > eduLLM-Evals/staging/ablation/scratch/rubrics_qmatrix_pre_gateB.jsonl
```

To see exactly which criteria Gate-B changed (to build a surgical E-only revert):

```powershell
git diff 4137990 306c86e -- eduLLM-Evals/data/TutorBench/curated/rubrics_qmatrix_curated.jsonl
```

(Do NOT use `git checkout 4137990 -- <path>` — that overwrites the working-tree curated bank
in place, violating the read-only guarantee. Use `git show > scratch` instead.)

> **⚠ B/E entanglement caveat (important).** Gate-B bundled the **exclude-60 flags AND the
> text/relabel edits into the same commit `306c86e`**. Therefore the pre-Gate-B snapshot
> (`4137990`) has **neither** — checking it out reverts **factor E and factor B together**.
> A naïve "E row = fit on the `4137990` snapshot" is really an **E+B** row, not a clean LOO
> of E. To isolate E cleanly, build a snapshot that reverts **only** the 2 text rewords + 5
> relabels while **keeping** the 60 exclude flags (apply those specific record reversions on
> top of the current curated bank, e.g. via the `git diff` above restricted to the 7
> non-exclude records). Given E is expected to be **negligible** (2 text edits + 5 relabels
> out of ~6,845 records / ~6,180–6,845 criteria in the fit block), the pragmatic call is to
> either (a) accept the E+B combined row and lean on the clean B row to net it out, or (b)
> do the surgical 7-record revert. Decide before running; don't discover it in the numbers.

---

## 7. Interaction caveat

LOO measures each factor's **marginal effect at the all-on operating point** only. It does
**not** isolate interactions: if two factors both act on the same underlying problem, their
individual LOO deltas can under- or over-state their combined effect, and won't sum to
`(R0 − R_off)`.

Two interactions are worth a dedicated **2×2** (4 runs) if budget allows:

1. **ridge (A) × exclude-60 (B)** — *the priority.* Both target the **same noisy-item
   problem**: ridge shrinks the runaway/extreme-a scaffolding loadings; exclude-60 removes
   the reverse/degenerate items outright. They are partial substitutes, so at the all-on
   point (ridge already 1e-2) removing the 60 items may buy little, and the LOO-B delta will
   understate what exclusion would do at ridge 1e-3. A 2×2 `{ridge 1e-3, 1e-2} × {exclude
   on, off}` cleanly separates "regularize the noise" from "delete the noise."
2. **ridge (A) × skill-mode (D)** — the Scaffolding-Hygiene ridge sweep was validated on the
   **2-skill** fit only; the 3-skill scaffolding axis uses the same `ridge 1e-3→1e-2`
   mechanism but was **never re-validated** (explicit caveat in that memo). A `{ridge} ×
   {2-skill, 3-skill}` 2×2 confirms the ridge win transfers to the presentation-axis
   instrument before we rely on it there.

If only one 2×2 is run, do **ridge × exclude-60**.

---

## 8. Pre-registered decision rules (KEEP vs REVERT, stated NOW)

Stated before results exist, so they can't be fit to the outcome.

- **A. ridge = 1e-2.** KEEP 1e-2 **iff** the 200-run *re-run of the k-fold OOS-recovery ridge
  sweep* (baseline → 5e-3 → 1e-2 → 2e-2, per the Scaffolding-Hygiene recommendation) still
  plateaus at ≈1e-2 on OOS scaffolding recovery with no correctness regression. **REVERT (or
  re-tune)** to the new plateau point if the 200-model plateau moves — optimal ridge is a
  function of person-N (more persons → less shrinkage), so 1e-2 is explicitly *not* a
  permanent default. Concretely: pick the smallest ridge whose OOS scaffolding recovery is
  within noise of the max, subject to OOS correctness ≥ baseline.
- **B. exclude-60.** KEEP the exclusion **unless** the 200 models give those 60 items *real
  variance and positive, non-artifact loadings* — i.e. with more persons they stop being
  all-fail/near-separation/reverse-behaving and fit stable positive `a` with a healthy
  point-biserial. Operational test: on the 200-run, re-audit the 60; **REVERT (fit them in)**
  any item that now has pass-rate variance in a non-degenerate band **and** `a_scaffolding > 0`
  with pbis(scaffolding ability) above the audit's ~0.3 bar. Keep excluding the rest.
- **C. CAT policy (floor 15 + pbis + balancing + exposure).** KEEP policy ON. Do **not** be
  moved by legacy showing *higher* scaffolding recovery — that lift is the documented
  deceptive convergence (scaffolding "converges" on as few as 1 administered item). The
  policy's *lower* scaffolding recovery is the honest number. The only genuinely tunable knob
  is the **floor**: re-run the floor sweep {10…15(+)} on 200 models and re-pick the floor at
  the convergence knee; the floor value may change even though "policy ON" does not.
- **D. 2-skill vs 3-skill (presentation).** Not a keep/revert — a structural decision already
  made (Run 6: admit presentation as a 3rd optional axis, confirm at N≥150). Use the 200-run
  to **confirm/lock presentation's loadings** (target: `a_presentation` cross-fold stability
  stays > `a_scaffolding`, as it was at N=82: 0.712 vs 0.488) and re-check `r(corr,pres)` for
  incremental CAT value. Report 2- and 3-skill as parallel columns; do not collapse D into
  the single-factor attribution.
- **E. Gate-B bank edits (2 rewords + 5 relabels).** KEEP. Expected impact is negligible
  (7 records out of ~6,845). **REVERT** a specific edit only if the isolated E row (or the
  surgical 7-record revert) moves any headline metric beyond fold noise (say |ΔOOS r| > 0.01)
  — which would itself be a surprising finding worth its own look, not a routine revert.

---

## Appendix — exact flag/mechanism inventory (verified against code)

- **A · ridge:** `scripts/calibrate_mirt.py` `--ridge` (float, **default 1e-2**, line 886;
  penalizes loadings not intercept, `_item_neg_loglik`). `scripts/kfold_cv_mirt.py` `--ridge`
  (float, **default 1e-3**, line 545) — *mismatched default, must pass 1e-2 for baseline.*
- **B · exclude-60:** honored in `calibrate_mirt.py::load_q_matrix` lines 185–186
  (`if rec.get("exclude_from_fit"): continue`). No bypass flag exists → add
  `--ignore-exclude-flag`. Independently enforced for CAT in
  `cat_eval_tutorbench_multiskill.py::load_excluded_criteria` (lines 192–206) + main
  (960–967) via `--curated`.
- **C · CAT policy:** `cat_eval_tutorbench_multiskill.py` `--legacy` (line 918) →
  `Policy.legacy()` (pbis off, `min_items_per_skill=0`, content-balance off,
  `exposure_top_n=1`; deterministic tie-break stays on). Per-knob: `--min-items-per-skill`
  (default 15, line 922), `--no-pbis-filter` / `--min-point-biserial` (0.05, lines 920–921),
  `--no-content-balance` (line 924), `--exposure-top-n` (default 5, line 925).
- **D · skill mode:** fitter `--collapse content,diagnosis` (line 879, in-memory Q-OR
  transform, bank untouched); CAT `--skills {2,3}` / `--skill-set` (line 901). 3-skill Q via
  `scripts/build_candidate_qmatrix.py` presentation slot-repurposing (Run 6).
- **E · Gate-B revert:** pre-Gate-B commit `4137990`; recipe
  `git show 4137990:eduLLM-Evals/data/TutorBench/curated/rubrics_qmatrix_curated.jsonl > <scratch>`.
  Entangled with B (both in commit `306c86e`).
