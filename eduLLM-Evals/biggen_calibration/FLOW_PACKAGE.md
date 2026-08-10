# BiGGen -> flow/uni-frq graduation package (gemini-3 of-record, INTERIM)

Interim, self-contained payload for porting BiGGen's finalized **gemini-3-flash-preview** calibration
into the FRQ eval flow. **Layout is provisional**: conform the file arrangement to the FRQ diagnostic
loader (`diagnostics/frq_cat/common/irt_params.py` on `CheckpointFlows`, mirroring
`diagnostics/mcq_cat/`) before committing into `flow/uni-frq`.

This supersedes the Qwen-judge payload archived under `qwen_superseded/FLOW_PACKAGE.md`.

## Status

- **PROVISIONAL — 52 models** (below the ideal ~150). The item bank is refit per judge; a larger
  rerun is a drop-in swap of the bank files (see Rerun protocol).
- **Judge = `gemini-3-flash-preview`** (frontier). The bank was re-graded + re-fit on this judge and
  carries an explicit **SE_judge** layer (judge-error uncertainty) on top of SE_ability + SE_param.
- Snapshot tag: `biggen-cal-gemini3-52models`.

## Target

`flow/uni-frq` — unidimensional (`general` skill), MWLE estimator, scenario-level CAT.

## Locked CAT config

`SE_post target = 0.12`, `min_scenarios (floor) = 10`, `ridge = 0.01`, estimator = **MWLE-at-stop**,
selection = trace, `k = 5` CV, seed `20260729`. **CAT stop = dense 3201-node EAP-posterior marginal
SD** over [−8, 8] (plateau δ = 0.005 / W = 3, cap 40/50); **reach = SE_post ≤ target ONLY**.
Leaderboard θ = **full-bank fine-EAP** posterior mean (**observed**, not per-model de-biased).
Deployed test length ~**20
scenarios** mean (median 13 OOS); ~6 lowest-ability tiny base models cap. **Pass-imbalance curation**:
2337 administrable → 2015 fitted (drop train `pass_count ≤ 3` OR `fail_count ≤ 3`).

## Payload

| Artifact | Path | Role | Tracked |
|---|---|---|---|
| Deployment params bank | `bank/biggen_unidim_modeled_gemini3_curated.jsonl` | 2015 fitted items (a/b per criterion) the engine scores a new model with | yes |
| Response matrix (calibration input) | `bank/response_matrix.csv` (52 × 2337) | reproducibility input only; NOT needed at flow runtime | yes (shipped as provenance) |
| Judge confusion (SE_judge) | `bank/biggen_gemini3_confusion.json` | α/β from 250 human-gold cells vs gemini-3 verdicts | yes |
| Human gold labels | `bank/gold_labels.jsonl` | gold backing the confusion / de-bias | yes |
| Provenance | `bank/PROVENANCE.json` | sha256 + source paths for the above | yes |
| Items / questions | `data/BiGGen/scenarios.jsonl` | prompts + conversation_context the tutor model responds to | yes |
| Rubric criteria | `data/BiGGen/rubrics.jsonl` | per-scenario criteria (also embedded in the bank) | yes |
| Judge config | `gemini-3-flash-preview` (frozen prompt/config in `api_judge_pilot/`) | frozen judge for grading responses identically | see api_judge_pilot |

## Runtime data flow (scoring a new checkpoint)

1. Generate tutor responses from the checkpoint on `data/BiGGen/scenarios.jsonl`.
2. Grade responses per criterion with the frozen `gemini-3-flash-preview` judge.
3. Feed graded per-criterion outcomes + `biggen_unidim_modeled_gemini3_curated.jsonl` (a/b params)
   into the MWLE/CAT loop at the locked config to estimate ability θ with SE.
4. Report **observed θ** with SE_total = √(SE_ability² + SE_param² + SE_judge²). For judge
   strictness, apply the **cohort pass-rate de-bias factor ×1.33 [1.17, 1.57]** (closed form
   `(p_obs−β)/(1−α−β)`) to *cohort* pass-rates only. Do **not** apply a per-model `theta_debiased`
   (the prevalence-prior resample is a shrinkage estimator, not of-record; per-model noisy-label
   IRT de-bias is future work).

## Provenance + headline metrics

- Judge: `gemini-3-flash-preview`. Fit: `calibrate_mirt.fit_m2pl_em`, skill = `general`, ridge 0.01,
  n_models = 52, 2015 fitted criteria (pass-imbalance curated).
- Matrix sha256: `2bee8b127b18be4674201138972462862153cb282b5b31d2346040340039fba5`.
  Bank sha256: `544e8fab71bcf6b36c5fe62bfebd593fc52a9c15a4d2a47a5a0a701ede374b60`.
- OOS of record @ 10/0.12 (excl 3 weakly-identified): **r = 0.956, slope = 0.882, θ-MAE = 0.257**,
  median 13 scen, %reach(SE_post ≤ 0.12) = 69.4% (all-52 r = 0.953, %reach 65.4%).
- p-IRT pass-rate MAE **0.050** (r = 0.947). Adaptive vs random: **13 vs 87 scenarios (~6.7×)**.
- **SE_judge (judge-error layer):** α ≈ 22.5% strict / β ≈ 2.2% cert-safe ⇒ absolute pass-rates
  biased **down**; **cohort pass-rate de-bias ×1.33 [1.17, 1.57]** (closed form). SE_judge is the
  **dominant, correlated floor** (~88% of full-bank SE_total²; full-bank median 0.146, op-point
  0.164) and does **not** shrink across models; frozen Tier A is the conservative of-record (Tier B
  refit ≤ Tier A). **Rankings robust** (judge error systematic/correlated). Absolute θ / pass-rate
  carry the cohort bias. **No per-model `theta_debiased` is of-record** — the prevalence-prior
  resample is a shrinkage estimator (compresses the scale); per-model noisy-label IRT de-bias is
  future work.
- Dimensionality: **unidimensional** (single `general` skill), unchanged from the Qwen study.
- Caveats: N = 52 is provisional; leaderboard adjacent-pair SE bands overlap. 3 weakly-identified
  models excluded from the headline: `ai-forever/mGPT`,
  `BEE-spoke-data/smol_llama-220M-GQA-fineweb_edu`, `allenai/OLMo-1B-hf`.
- Supersedes the Qwen-judge of-record (`qwen_superseded/`).

## Rerun / swap protocol (~150 models)

1. Recalibrate on the frozen `gemini-3-flash-preview` judge; re-emit
   `biggen_unidim_modeled_gemini3_curated.jsonl` (+ refreshed confusion/gold if the gold set grows).
2. Replace the bank + provenance files here; bump `n_models` + `matrix_sha256` in `bank/PROVENANCE.json`.
3. Re-tag `biggen-cal-gemini3-<N>models`; re-port to `flow/uni-frq`.
4. Items / rubrics / judge config are unchanged unless new scenarios are added, so the flow harness
   itself does not change — only these data artifacts.
