# BiGGen calibration — CANONICAL unidimensional scale (gemini-3 judge, floor 10 / SE_post 0.12)

**unidim `general`, judge = `gemini-3-flash-preview`, op-point floor 10 / SE_post 0.12, N=52; headline excludes 3 weakly-identified models.**

This is the **canonical (of-record) BiGGen scale**: a single latent **general** ability fit as a
unidimensional 2PL over the BiGGen criteria, **recalibrated on the frontier `gemini-3-flash-preview`
judge** and carrying an explicit **SE_judge** (judge-error) uncertainty layer. A scenario is
administered as a testlet bundle of its per-criterion items; the real production engine
(`scripts/scenario_cat_lib.py` → `tutor_cat` / `scripts/calibrate_mirt.py`) selects a whole scenario
per CAT step and updates ability. Nothing here reimplements CAT selection or the item-response update.

The **prior Qwen-judge of-record** (unidim `general`, floor 8 / SE 0.12) is archived under
[`qwen_superseded/`](qwen_superseded/README.md) — retained for provenance/diff only. **Do not use it
for of-record reporting.**

**Status:** STUDY / reporting only, **LOCAL**. The production engine (`tutor_cat/`,
`scripts/scenario_cat_lib.py`, `scripts/calibrate_mirt.py`) and the banks were only read/called,
never modified by this package. **Nothing committed by the packaging step.** Mirrors the sibling
`tutoreval_calibration/` / `tutorbench_calibration/unidim/` numbered-experiment layout, plus the
SE_judge additions (experiments 07 + 10).

---

## Locked operating point + methodology

- **Scale:** unidimensional 2PL, single latent `general` (`q_modeled = {"general": 1}`, `axis: "unidim"`).
- **Judge:** `gemini-3-flash-preview` (frontier). The bank was re-graded and re-fit on this judge;
  the SE_judge layer quantifies the residual judge error against 250 human-gold cells.
- **Fit:** production M2PL MML-EM (`calibrate_mirt.fit_m2pl_em`), 1 dim, `fit_grid = 7` GH nodes,
  `ridge = 0.01`, clamp; skill = `general`; `n_models = 52`.
- **Pass-imbalance curation:** administrable pool 2337 criteria → **322 columns excluded** (train
  `pass_count ≤ 3` **OR** `fail_count ≤ 3`, applied per-fold OOS and full-data) → **2015 fitted
  criteria** in the shipped bank (`bank/biggen_unidim_modeled_gemini3_curated.jsonl`).
- **CAT stop:** dense **3201-node** EAP-posterior marginal-SD grid over [−8, 8], info-plateau
  δ = 0.005 / W = 3, cap 40/50, **MWLE θ at stop** (CAT ability). **Reach = SE_post ≤ target ONLY.**
- **Leaderboard θ:** **full-bank fine-EAP** posterior mean (stop-independent), reported **as
  observed** (NOT per-model de-biased). Judge strictness biases *absolute* pass-rates down; the
  honest correction is the **cohort pass-rate de-bias factor** (see SE_judge below), not a per-model
  θ shift.
- **Operating point (of-record): floor(min_scenarios) = 10, SE_post target = 0.12** — chosen off the
  floor × SE grid (experiment 06 / `DECISION_TABLE.md`).
- **OOS:** k = 5 model-fold, seed 20260729, refit-per-fold (1-D, ridge 0.01, GH 7, clamp; per-fold
  pass-imbalance exclusion).
- **N = 52 models; headline N = 49 EXCLUDES 3 weakly-identified models** whose EAP posterior SD at the
  forced-long cap still exceeds the loosest grid target (identified **from the run**, not assumed):
  `ai-forever/mGPT`, `BEE-spoke-data/smol_llama-220M-GQA-fineweb_edu`, `allenai/OLMo-1B-hf`.

## Headline numbers @ 10 / 0.12

| quantity | excl-weak (N=49) | all-52 |
|---|---|---|
| OOS recovery r | **0.956** | 0.9533 |
| slope | **0.882** | 0.7791 |
| θ-MAE | **0.257** | 0.3407 |
| median test length (scenarios) | **13** | 13 |
| length mean ± SD | 13.18 ± 2.8 | 13.13 ± 2.73 |
| %reach (SE_post ≤ 0.12) | **69.4%** | 65.4% |
| median SE_total (ability+param) | 0.144 | 0.145 |
| p-IRT pass-rate MAE | **0.050** (r = 0.947) | 0.049 (r = 0.956) |
| adaptive vs random to reach SE_post ≤ 0.12 | **13 vs 87 scenarios (~6.7×)** | — |
| leaderboard top (headline, observed θ) | `LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct` θ = 2.65 (SE_total 0.253) | — |
| leaderboard bottom (headline, observed θ) | `BEE-spoke-data/smol_llama-220M-openhermes` θ = −2.79 (SE_total 0.276) | — |

Machine-readable: `summary.json`. **%reach is corrected** — reach is gated on `SE_post` (EAP
posterior SD at stop) ≤ target **ONLY**; SE_total (incl. SE_param, SE_judge) is a reported precision
number, **not** a gate.

**Slope caveat:** OOS recovery slope ≈ **0.88** (excl-weak) — a mild uniform ~12% absolute-scale
compression (CAT MWLE pulls extremes toward centre). **Rankings and pass-rate prediction are
unaffected** (r = 0.956, p-IRT MAE = 0.050); only absolute-θ units are shrunk toward 0. For absolute
θ, apply an affine rescale (÷ slope); for ranking / pass-rate, use θ as-is.

---

## ⚠️ SE_judge — the key story (judge-error uncertainty layer)

The BiGGen labels come from a fallible judge, so the calibration carries a third uncertainty
component, **SE_judge**, alongside SE_ability (finite test) and SE_param (finite calibration sample).
Estimated from **250 human-gold cells** vs the `gemini-3-flash-preview` verdicts
(`bank/biggen_gemini3_confusion.json`, `bank/gold_labels.jsonl`):

- **Direction (firm): the judge is STRICT.** False-fail **α ≈ 22.5%** (judge fails a good answer) vs
  false-pass **β ≈ 2.2%** (judge passes a bad answer). β small ⇒ **certification-safe** (a passed
  answer is almost always genuinely good); α large ⇒ observed θ / pass-rates are biased **down**.
- **De-bias is a COHORT pass-rate factor, not a per-model θ shift: ×1.33 [1.17, 1.57]**
  (closed form `p_true ≈ (p_obs − β)/(1 − α − β)`). Direction is firm (α strict ⇒ absolute
  pass-rates biased down); magnitude band is wide on 250 gold cells. **A per-model `theta_debiased`
  is NOT of-record** — the earlier per-column-prevalence-prior resample is a *shrinkage estimator*
  that compresses the ability scale (top models spuriously move down, low tail pulled up), so it is
  retained only as a labeled diagnostic (`experiments/08_leaderboard/leaderboard_f10se12_diagnostic_shrinkage.csv`).
  A proper per-model noisy-label IRT de-bias is **future work**.
- **SE_judge is the DOMINANT, correlated floor.** It is **~88% of full-bank SE_total²** (full-bank
  median SE_judge **0.146**, dominates in 94% of models); at the op-point it is heavier-tailed
  (median **0.164**, dominates in ~60%). It does **NOT shrink across models** — there is one fixed
  judge, so its error is correlated across the whole leaderboard rather than averaging out.
- **Frozen Tier A is the conservative of-record.** Tier B (refit the bank on each resampled judge
  replicate) gives SE_judge ≤ Tier A (B/A median 0.64) — item-parameter re-estimation **absorbs**,
  not amplifies, judge noise — so the frozen-bank Tier A layer we ship is the conservative choice.
- **Rankings are robust to the judge layer** (the judge error is systematic/correlated, so it does
  not reshuffle the ordering); **absolute θ / pass-rate carry the cohort bias** (the ×1.33 factor
  above). The of-record fig-08 leaderboard ranks by **observed θ** with **SE_total** error bars (no
  per-model bias band). Per-stratum α/β vary: **theory_of_mind is strictest (α ≈ 0.42)**;
  **instruction_following carries the highest β (≈ 0.11)**.

See experiments **07** (3-component SE decomposition) and **10** (judge-error / de-bias), plus
`se_judge/` inputs summarized in `summary.json` → `se_components` / `se_judge_regimes` / `judge_error`.

---

## Experiments (numbered set)

| # | experiment | headline | figure |
|---|---|---|---|
| 04 | efficiency vs random (OOS) | adaptive reaches median SE_post ≤ 0.12 at **13** vs random at **87** scenarios (**~6.7×** fewer) | `experiments/04_efficiency_vs_random/adaptive_vs_random_efficiency_f10se12.png` |
| 05 | OOS recovery @ 10/0.12 | excl-weak **r = 0.956, slope = 0.882, θ-MAE = 0.257**; all-52 r = 0.953 (slope < 1 = mild compression) | `experiments/05_oos_recovery/oos_recovery_ability_f10se12.png` |
| 06 | floor × SE operating-point grid | decision table across floor {6,8,10,12,15} × SE {0.10–0.25}; floor 10 / 0.12 = r 0.956 / %reach 69 (excl-weak); floor is length knob, SE the precision knob | `experiments/06_floor_se_grid/figures/oos_grid_heatmaps.png` + `DECISION_TABLE.md` |
| 07 | parameter uncertainty (3-component SE: ability, param, **judge**) | full-bank + op-point mean ± SD; **SE_judge ≈ 88% of full-bank SE_total²**; op-point SE_total median 0.211 | `experiments/07_parameter_uncertainty/se_components_f10se12.png` |
| 08 | general-ability leaderboard (52 models, full-bank **observed θ** + SE_total incl. SE_judge) | top `LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct` θ = 2.65; bottom headline `BEE-spoke-data/smol_llama-220M-openhermes` θ = −2.79; 3 weak at the extreme low tail greyed. Per-model θ_debiased/bias-band moved to `leaderboard_f10se12_diagnostic_shrinkage.csv` (NOT of-record; shrinkage estimator) | `experiments/08_leaderboard/leaderboard_f10se12.png` |
| 09 | p-IRT predicted-vs-actual pass rate (OOS) | pass-rate MAE **0.050** (r = 0.947) excl-weak; all-52 MAE 0.049 | `experiments/09_pirt_mae/pirt_pred_vs_actual_f10se12.png` |
| 10 | judge error profile + cohort de-bias | per-stratum α/β (ToM strictest α ≈ 0.42; IF highest β ≈ 0.11); **cohort pass-rate** de-bias **×1.33 [1.17, 1.57]** only (no per-model θ de-bias — shrinkage; future work) | `experiments/10_judge_error/judge_error_debias_f10se12.png` |

Experiments 01–03 (min-scenarios / SE-target / ridge decision sweeps) and 11–12 (EAP-stop adoption
study) from the Qwen package are **not re-run** on the gemini-3 judge; they live under
`qwen_superseded/experiments/` for methodology provenance. The gemini-3 of-record here is the
04–10 set at the locked 10/0.12 operating point.

## Layout

```
biggen_calibration/
  README.md                 <- this file (canonical gemini-3 of-record)
  FLOW_PACKAGE.md           <- flow/uni-frq graduation payload (gemini-3)
  summary.json              <- machine-readable op-point / config / headline (both subsets) + SE_judge
  bank/
    biggen_unidim_modeled_gemini3_curated.jsonl   <- 2015 fitted criteria (deployment params)
    response_matrix.csv                            <- calibration input (52 x 2337)
    biggen_gemini3_confusion.json                  <- SE_judge alpha/beta from gold
    gold_labels.jsonl                              <- human gold backing the confusion
    PROVENANCE.json                                <- sha256 + source paths for the above
  scripts/                  <- repro drivers (read-only vs the production engine)
    run_oos_grid.py  build_of_record_f10se12.py  se_judge_lib.py  se_judge_gemini3.py  curate_and_refit_gemini3.py
  experiments/
    04_efficiency_vs_random/  05_oos_recovery/  06_floor_se_grid/
    07_parameter_uncertainty/ 08_leaderboard/   09_pirt_mae/  10_judge_error/
  qwen_superseded/          <- ⚠️ prior Qwen of-record (bank + experiments 01–12 + README/FLOW), provenance only
```

## Caveats

- **N = 52 is provisional** (below the ideal ~150 / the MIRT identifiability floor). Treat item
  params as provisional; a larger rerun is a drop-in bank swap.
- Leaderboard adjacent-pair SE bands overlap once SE_judge is included — only coarse ability bands are
  distinguishable at N = 52.
- The 3 weakly-identified tiny base models sit at the extreme low tail and are excluded from the
  headline (kept in the all-52 numbers and flagged as upper bounds on the leaderboard).
