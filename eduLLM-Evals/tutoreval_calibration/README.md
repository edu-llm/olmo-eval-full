# TutorEval calibration — CANONICAL gemini-3 unidim scale (floor 15 / SE_ability 0.25)

**unidim `ability`, floor 15 / SE_ability 0.25, N=52; headline excludes 3 weakly-identified
models. Judge = `gemini-3-flash-preview`.**

This is the **canonical (and only) TutorEval scale**, recalibrated on **gemini-3-flash-preview**
judge labels: a single latent **ability** fit as a unidimensional 2PL over the TutorEval criteria.
A scenario is administered as a **testlet bundle** of its per-criterion items; the real production
engine (`tutor_cat.engine.run_evaluation` via `scripts/scenario_cat_lib.py`) selects a whole
scenario per CAT step and updates ability. Nothing here reimplements CAT selection or the
item-response update.

This package **supersedes the prior Qwen-judge of-record** (floor 15 / SE 0.22), which is archived
under [`qwen_superseded/`](qwen_superseded/) — moved, not deleted. The calibration is on **OBSERVED
gemini-3 labels (Rule 0, no pre-correction)**; the judge measurement-error (SE_judge / de-bias)
layer is quantified and reported alongside but is NOT folded back into the fit.

**Status:** STUDY / reporting only, LOCAL. The production engine (`tutor_cat/`,
`scripts/scenario_cat_lib.py`, `scripts/calibrate_mirt.py`) and the banks were only read/called,
never modified. Nothing committed.

---

## Locked operating point + methodology

- **Scale:** unidimensional 2PL, single latent `ability` (`q_modeled = {"ability": 1}`, `axis: "unidim"`).
- **Judge:** `gemini-3-flash-preview`. Fit uses OBSERVED gemini-3 labels (Rule 0).
- **Curation — MINIMAL Rule-0:** auto zero-variance drop only (542 all-fail columns dropped →
  **1244 fitted** criteria). NO near-separation / pass-imbalance exclusion. Fit:
  `calibrate_mirt.fit_m2pl_em`, 1 dim, `fit_grid = 7`, `ridge = 1e-2`, `negative_policy = clamp`
  (35 negatives preserved in-file, clamped at consume time). `n_persons = 52`, fit loglik
  −18071.21 (converged, 13 iters).
- **CAT stop:** EAP-posterior honest **1-D marginal SD**, dense **321-node** grid over [−8, 8],
  info-plateau **δ = 0.005 / W = 3**, **cap 70**, **MWLE θ at stop**. Stop priority:
  precision → plateau → cap → bank_exhausted.
- **Operating point (LOCKED): floor(min_scenarios) = 15, SE_ability target = 0.25.** Deployed
  headline θ = **op-point MWLE-at-stop @ f15/se0.25** (leaderboard `theta`, length ≥ 15). Reach =
  SE_ability (EAP posterior SD at stop) ≤ target ONLY.
- **OOS:** k = 5 model-fold, seed 20260729, refit-per-fold (1-D, ridge 0.01, fit_grid 7, clamp,
  within-fold zero-variance filter).
- **N = 52 models; headline N = 49 EXCLUDES 3 weakly-identified models** (best-achievable SE_total
  at the cap still > 0.30): `BEE-spoke-data/smol_llama-220M-GQA-fineweb_edu`, `ai-forever/mGPT`,
  `allenai/OLMo-1B-hf`. The weak set is identified **from the run**, not assumed.

## De-bias / SE_judge convention = HYBRID

gemini-3 is **strict** on TutorEval (α ≈ 0.45 false-fail; β = 0 false-pass — zero in 100 gold
cells), so **SE_judge is the DOMINANT uncertainty floor** and observed θ is biased **down**. We
publish under a hybrid convention:

- **Of-record leaderboard (of-record headline):** publish **observed θ** (op-point MWLE-at-stop)
  with **SE_total error bars** (`SE_total = √(SE_ability² + SE_param² + SE_judge²)`; SE_judge
  dominant), the 3 weak models greyed, plus an annotated **1.83× cohort pass-rate de-bias caveat**
  (posterior CI **1.33–2.97×**). The correction is **NOT** applied per-model in the headline.
- **Secondary diagnostic:** a **per-model θ_debiased BAND from the CLEAN full-bank regime**
  (Spearman **0.993**, gate passes, **0 violations**) — see `experiments/11_debias_band/`. This is
  **NOT** the of-record headline.
- **Per-model op-point θ_debiased is directional-only for the weak tail** (op-point Spearman
  **0.778**; weak-tail bands span 2–3 θ units, SE_judge up to 1.04). Do NOT present it as precise
  per-model estimates.

## Headline numbers @ 15/0.25

| quantity | excl-weak (N=49) | all-52 |
|---|---|---|
| OOS recovery r | **0.954** | 0.966 |
| slope | **0.879** | 0.916 |
| θ-MAE | **0.292** | 0.303 |
| median test length (scenarios) | **15** | 15 |
| %reach (SE_ability ≤ 0.25) | **95.9%** | 90.4% |
| median SE_total (incl SE_judge) | **0.437** | 0.450 |
| p-IRT pass-rate MAE | **0.036** (r = 0.935) | 0.035 (r = 0.942) |
| adaptive vs random to reach SE ≤ 0.25 | **7 vs 70 scenarios (~10×)** | — |
| leaderboard top (headline) | `tiiuae/Falcon3-3B-Instruct` θ = **2.71** (SE_total 0.37) | |
| leaderboard bottom (headline) | `BEE-spoke-data/smol_llama-220M-openhermes` θ = **−3.64** | |

**SE_judge layer (Tier A, frozen bank):** op-point median SE_judge **0.381** (var-share ≈ **69%**
of SE_total²), SE_total median **0.450**; full-bank median SE_judge **0.358** (var-share ≈ **89%**),
SE_total **0.378**. SE_judge does NOT shrink with more models (dominates 96% full-bank / 75%
op-point). **Cohort de-bias 1/(1−α) = 1.83× [1.33–2.97×]** (UP, direction firm, **no crossover**).
Machine-readable: `summary.json`.

## ⚠️ Caveats (read before quoting)

- **Absolute θ is judge-biased DOWN.** The of-record leaderboard is **observed** θ; the cohort
  carries the 1.83× de-bias band. Rankings survive (full-bank Spearman 0.993), pass-rate prediction
  is faithful (p-IRT MAE 0.036), but absolute θ / pass-rate should be read with the band.
- **SLOPE ≈ 0.88** — a mild (~12%) absolute-scale compression, a genuine bank property (CAT MWLE
  pulls extremes toward centre). Rankings and pass-rate are unaffected; only absolute-θ units are
  shrunk. (gemini-3's slope is higher than the Qwen of-record's ≈ 0.80.)
- **Precision lever:** tightening α (hence the whole de-bias band) needs a **judge-FAIL-enriched
  gold audit** (more true-passes gemini wrongly failed), **not more cells** — the α CI is wide
  because the gold is fail-heavy (only 20 gold-PASS cells constrain α).
- **Tier B ≤ Tier A:** refitting the bank each replicate absorbs judge noise into the item
  parameters, so the **frozen-bank Tier A is the conservative of-record** (item-parameter
  re-estimation does NOT amplify SE_judge).

---

## Layout

```
tutoreval_calibration/
  README.md                 this index (gemini-3 canonical)
  FLOW_PACKAGE.md           graduation manifest -> flow/uni-frq (unidim, gemini-3 judge)
  summary.json              machine-readable op-point / config / all headline numbers + SE_judge
  bank/
    rubrics_qmatrix_final_unidim_gemini3_fitted.jsonl   fitted bank (1244 criteria; dim = ability)
    bank_fit_summary.json                               fit provenance (sha, loglik, config, low-n)
  se_judge/                 judge measurement-error layer (Phase 3)
    summary.json            alpha posteriors, both-regime SE stats, de-bias gate, Tier B
    per_model_full_bank.csv theta/theta_debiased + SE_ability/param/judge + band + SE_total
    per_model_oppoint_f15se25.csv  deployed op-point per-model (headline theta + SE_total)
    tierB_sensitivity.csv   Tier A vs Tier B SE_judge per model
    CHECKPOINT.md           Phase-3 interpretation write-up
  scripts/                  reproduction scripts (provenance; paths reflect the build layout)
  experiments/
    04_efficiency_vs_random/  adaptive vs random efficiency (~7 vs 70 scenarios, ~10x)
    05_oos_recovery/          headline OOS recovery (r 0.954 / slope 0.879 / theta-MAE 0.292)
    06_floor_se_grid/         floor x SE OOS grid CSVs + DECISION_TABLE (op-point selection basis)
    07_se_components/         SE_ability/param/judge bars (SE_judge dominant), both regimes
    08_leaderboard/           OF-RECORD leaderboard @ 15/0.25 (observed theta + SE_total + caveat)
    09_pirt_mae/              p-IRT / pass-rate MAE (0.036)
    10_judge_error/           per-stratum alpha/beta + cohort de-bias 1.83x [1.33,2.97]
    11_debias_band/           SECONDARY full-bank theta_debiased band (NOT of-record headline)
  qwen_superseded/          the prior Qwen-judge of-record (floor 15 / SE 0.22) — archived
```

## Experiment index

| slot | description | headline | path |
|---|---|---|---|
| `04_efficiency_vs_random` | adaptive vs random efficiency | 7 vs 70 scenarios (~10×) | `experiments/04_efficiency_vs_random/adaptive_vs_random_efficiency_f15se25.png` |
| `05_oos_recovery` | OOS recovery scatter (MWLE θ vs reference) | r 0.954 / slope 0.879 / θMAE 0.292 | `experiments/05_oos_recovery/oos_recovery_ability_f15se25.png` |
| `06_floor_se_grid` | floor×SE OOS grid (op-point selection basis) | row 15/0.25 selected | `experiments/06_floor_se_grid/DECISION_TABLE.md` |
| `07_se_components` | SE_ability vs SE_param vs SE_judge | SE_judge dominant (op var-share 69%) | `experiments/07_se_components/se_components_f15se25.png` |
| `08_leaderboard` | OF-RECORD leaderboard @ 15/0.25 (observed θ) | top Falcon3-3B-Instruct θ=2.71 | `experiments/08_leaderboard/leaderboard_f15se25.png` |
| `09_pirt_mae` | p-IRT / pass-rate MAE | MAE 0.036 (r=0.935) | `experiments/09_pirt_mae/pirt_pred_vs_actual_f15se25.png` |
| `10_judge_error` | per-stratum α/β + cohort de-bias | α≈0.45 strict, β=0; 1.83× [1.33,2.97] | `experiments/10_judge_error/judge_error_debias_f15se25.png` |
| `11_debias_band` | SECONDARY full-bank θ_debiased band | Spearman 0.993, 0 violations | `experiments/11_debias_band/debias_band_full_bank_f15se25.png` |

## Bank

`bank/rubrics_qmatrix_final_unidim_gemini3_fitted.jsonl` — 1244 fitted criteria; dim = `ability`
(`q_modeled = {"ability": 1}`); negatives preserved in-file, clamp applied at consume time. Fit
provenance: `bank/bank_fit_summary.json`. The scenarios the tutor model responds to are
`data/TutorEval/scenarios_final.jsonl` (not copied here; referenced from `FLOW_PACKAGE.md`). The
gemini-3 response matrix (`api_judge_pilot/grading_tutoreval/response_matrix.csv`) is the
calibration input (not needed at flow runtime).

## Notes on scripts

`scripts/*.py` are retained as **provenance / reproduction record**. Their internal default paths
reflect the original `reports/tutoreval_gemini3_recal/` build layout (e.g. `se_judge/`,
`oppoint_grid/`, `bank/` siblings), so re-running them as-is from this package requires repointing
those paths — and requires the gemini-3 response matrix. The **artifacts** they produced are the
of-record deliverables under `experiments/`, `se_judge/`, `summary.json`, and `bank/`.

## Superseded Qwen of-record

The prior canonical scale (Qwen-judge, floor 15 / SE 0.22) is archived under `qwen_superseded/`
(its own `README.md`, `FLOW_PACKAGE.md`, `summary.json`, `bank/`, `scripts/`, `experiments/`). It
is retained as a record and is superseded by this gemini-3 package. Point readers here.
