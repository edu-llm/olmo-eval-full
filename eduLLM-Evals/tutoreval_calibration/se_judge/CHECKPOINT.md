# TutorEval gemini-3 recal — Phase 3 (SE_judge + de-bias): REVIEW CHECKPOINT

**LOCAL / STUDY ONLY.** Engine untouched (`tutor_cat/`, `scripts/scenario_cat_lib.py`,
`scripts/calibrate_mirt.py` imported read-only). Calibrated on OBSERVED gemini-3 labels (Rule 0,
no pre-correction). Nothing committed/staged. **STOP here for de-bias interpretation review — do
NOT proceed to Phase 4.** No of-record figures built; `tutoreval_calibration/` untouched.

- Op-point (LOCKED): **floor 15 / SE_ability 0.25**, dense-EAP posterior-SD stop (321-node grid
  [-8,8]), plateau δ0.005/W3, cap 70, MWLE-at-stop. Deployed order frozen at the observed-label
  adaptive trace (engine seed 42); the resample moves the stop along that order.
- Confusion (Beta-from-gold-counts, Jeffreys; **β pinned to 0** — zero false-pass in 100 gold
  cells): `conceptual_understanding` OWN α = Beta(7.5, 8.5), mean 0.469, CI95 [0.24, 0.71]
  (15 gold-PASS ≥ 8); `quantitative_procedural` (3 gold-PASS) + missing/unknown POOL to the
  OVERALL α = Beta(9.5, 11.5), mean 0.452, CI95 [0.25, 0.66].
- Tier A B=500 (frozen bank); Tier B B=50 (refit); bias band 100 draws/corner. Runtime ~12.7 min.

## 1. SE_judge magnitude — it is the DOMINANT uncertainty floor

| regime | SE_ability (med) | SE_param (med) | **SE_judge (med)** | SE_total (med) | SE_judge var-share (med) | SE_judge dominates |
|---|---|---|---|---|---|---|
| full-bank (cohort) | 0.096 | 0.077 | **0.358** (max 0.504) | 0.378 | **0.887** | 96% of models |
| op-point f15/se0.25 (headline) | 0.204 | 0.101 | **0.381** (max 1.038) | 0.450 | **0.690** | 75% of models |

- SE_judge is **~2× the next-largest term** and the dominant variance component in both regimes.
  At the op-point its median variance share is 69% (ability ≈ 21%, param ≈ 5%); full-bank 89%.
- It is **~2.5× the BiGGen SE_judge** (0.146) — direct consequence of α=0.45 here vs 0.225 there.
  gemini-3's over-strictness on TutorEval is the single largest source of ability uncertainty,
  larger than either the adaptive-test posterior SE or the item-calibration SE.
- op/full SE_judge ratio ≈ 1.07 (fewer administered cells → slightly more per-label leverage).

## 2. De-bias direction & magnitude — clean, firm UP, NO shrinkage crossover

β=0 makes this qualitatively different from BiGGen: the resampled TRUE matrix is **elementwise ≥
the observed matrix** (observed passes never flip; only observed FAILs can flip UP), so a monotone
estimator's θ_debiased is ≥ observed θ **by construction**. The empirical gate confirms it:

| regime | frac monotone (θ_deb ≥ θ) | violations | min shift | median shift | Spearman(obs, deb) |
|---|---|---|---|---|---|
| full-bank | **1.00** | 0 | +0.056 | **+1.41** | **0.993** |
| op-point | **1.00** | 0 | +0.008 | **+1.04** | **0.778** |

- **No top-model decrease** (min shift positive in both regimes) → the BiGGen shrinkage-crossover
  trap does **not** occur here. The aggregate shift is a true de-bias, not shrinkage.
- Aggregate pass-rate de-bias factor `1/(1−α)` = **1.83×** (posterior median 1.82×, CI95
  **[1.33×, 2.97×]**; α-CI corner band **[1.25×, 3.33×]**). Direction firm; magnitude band wide
  (only 20 gold-PASS cells constrain α).
- The shift is **non-uniform (scale compression), not a rank flip**: top models barely move
  (Falcon3-3B-Instruct +0.10 θ, band [2.40, 2.60]), the weak tail is pulled up hard
  (OLMo-1B-hf −5.40 → +0.44, band [−1.11, +1.41]). This compresses the bottom but preserves order
  (full-bank Spearman 0.993; Pearson 0.816 reflects the nonlinearity).

## 3. Rank stability

- **Full-bank: Spearman 0.993** — the de-bias is order-preserving; the cohort leaderboard ranks
  survive the correction essentially intact.
- **Op-point: Spearman 0.778** — moderate. The short (15-scenario) test + a *moving* stop point
  amplify weak-tail reshuffling: the bottom ~15 models' de-biased θ bunch into ~[+0.2, +0.6] with
  SE_judge up to 1.04, so their relative order is not robust under the correction.

## 4. SE_total breakdown at the op-point (headline θ)

`SE_total = √(SE_ability² + SE_param² + SE_judge²)`, same MWLE-at-stop estimator as the of-record θ.

- Medians: SE_ability **0.204**, SE_param **0.101**, SE_judge **0.381** → **SE_total 0.450**.
- Variance shares (median): **judge ≈ 69%**, ability ≈ 21%, param ≈ 5% (remaining from covariance
  of the median-of-ratios). SE_judge is the term that sets the floor on model separability at the
  op-point; SE_param (the sound observed-info parametric-bootstrap offset from Phase-1) is minor.

## 5. Tier B (item-parameter sensitivity)

- SE_judge_full Tier B (refit each draw) median **0.159** ≪ Tier A (frozen) **0.358**.
- Refitting absorbs the resampled judge noise into the item parameters ⇒ **frozen-bank Tier A is
  the conservative of-record**; item-parameter re-estimation does NOT amplify SE_judge.

## 6. Recommendation — publishable as a per-model BAND (with a weak-tail caveat)

**The direction/monotonicity gate PASSES** (0 violations, no crossover, firm UP), so per-model
**θ_debiased IS publishable as a band** — this is the genuine β=0 payoff the plan anticipated.
Concretely I recommend:

1. **Cohort / de-bias band → anchor on the FULL-BANK regime.** It is monotone, crossover-free,
   rank-preserving (Spearman 0.993), with tight bands for the top/mid cohort. Report θ_debiased
   with the α-CI band and the 1.83× [1.25–3.33×] pass-rate de-bias factor.
2. **Deployed op-point HEADLINE θ → publish observed θ with SE_total error bars** (SE_judge is the
   acknowledged dominant, correlated/systematic term — it does NOT shrink with more models), plus
   the aggregate cohort de-bias caveat. Per-model op-point θ_debiased is a valid band for the
   **well-identified** models but is **uninformative for the weak tail** (bands span 2–3 θ units,
   SE_judge up to 1.04, op-point Spearman 0.78) — flag those as directional-only.
3. **Tighten before quoting a point de-bias:** the α CI (hence the whole band) is wide because the
   gold is fail-heavy (20 gold-PASS cells). A judge-FAIL-enriched gold audit (more true-passes
   gemini wrongly failed) would narrow α and the band; more representative cells would not.

### Outputs (all under `reports/tutoreval_gemini3_recal/se_judge/`)
- `summary.json` — full config, α posteriors, both regime stats, de-bias gate, Tier B.
- `per_model_full_bank.csv` — θ (EAP), θ_mwle, θ_debiased, SE_ability/param/judge, band, SE_total.
- `per_model_oppoint_f15se25.csv` — deployed op-point θ (MWLE), θ_debiased, SE components, band,
  admin lengths, stop reason, SE_total.
- `tierB_sensitivity.csv` — Tier A vs Tier B SE_judge per model.
- `scratch/se_judge_lib.py`, `scratch/se_judge_tutoreval.py` — the (local) judge-error layer.
- Quadrature helper added to `scripts/frq_total_se.py` (`se_quadrature`, `regime_stats`).
