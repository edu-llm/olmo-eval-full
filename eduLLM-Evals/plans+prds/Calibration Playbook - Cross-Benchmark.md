# Calibration Playbook — Cross-Benchmark

**Purpose.** A benchmark-agnostic, runnable recipe for everything we did at *calibration
time* on TutorBench, so you can repeat it on a new benchmark (InfoBench / WildBench /
BiGGen / Bridge, etc.). Replace "your benchmark" with the one you're calibrating; the
TutorBench numbers are quoted only as worked examples.

> ### Note on the two script layers (read first)
> The per-benchmark branches are **intentionally pruned** to *their own benchmark + the shared
> toolchain* (e.g. commit `28f1e79` pruned `frq/infobench` to InFoBench + shared tooling), so this
> doc tags every script by layer:
> - **✅ SHARED TOOLCHAIN** — the core IRT calibration, OOS, and CAT engine. Present on every branch
>   under `scripts/` or `tutor_cat/`; the flags/columns below are verified against it.
> - **⚠️ SCENARIO-LEVEL WRAPPER** — the TutorBench analysis + figure scripts (and `regenerated_figures/`).
>   These were pruned off the per-benchmark branches by design; they live in pre-prune git history (or on
>   the TutorBench branch). Recover with `git log --all -- scripts/<name>.py` then
>   `git checkout <sha> -- <path>`, or rebuild on the present engine. Every ⚠️ step names the ✅ primitive
>   it was built on, so you are never blocked.

---

## 0. TL;DR pipeline

```
judge verdicts ──▶ response matrix ──▶ Q-matrix/skills ──▶ fitted bank
   (run_judge_grading)  (merge_full_matrix)  (generate/verify/finalize_qmatrix)  (calibrate_mirt --write-params)
        │
        ├─▶ core fit (calibrate_mirt.py): dims, latent-corr, AIC/BIC, collapse
        ├─▶ OOS k-fold (kfold_cv_mirt.py): generalization to unseen models
        ├─▶ sweeps: dimensionality → estimator → min_scenarios → SE target
        ├─▶ parameter-uncertainty / total SE
        └─▶ full CAT leaderboard + efficiency-vs-random (tutor_cat.engine)
```

---

## 1. Inputs

### 1.1 Response matrix (models × criteria)
- **Rows** = calibration models (the "persons"; TutorBench had 82 → 115). **Cols** = criteria.
  **Cells** = `1` pass / `0` fail; `no_decision`, absent, or optional-not-applicable → **NaN**.
- **✅ Build it:** grade with `scripts/run_judge_grading.py` (single judge) or
  `scripts/run_all_judge_grading.py` (fleet), stage inputs with `scripts/stage_judge_inputs.py`,
  ingest external CSVs with `scripts/ingest_calibration_csv.py`, then assemble the matrix with
  **`scripts/merge_full_matrix.py`**:
  - flags: `--big <matrix.csv> --supp <supplement.csv> --curated <bank.jsonl> --out-dir staging --basename response_matrix_full`
  - emits: `response_matrix_full.csv` (**all** criteria), `response_matrix_full_nonoptional.csv`
    (**gating only** — TutorBench: 6,180 cols), `response_matrix_full_provenance.csv`, and a manifest.
  - the **optional-vs-gating split** is read straight from the bank: a column is optional iff its
    record has `optional: true`. **Fit/score on the non-optional matrix**; the full matrix is only for
    optional-axis (e.g. presentation) analyses.
- **✅ Sanity/audit:** `scripts/audit_judge_verdicts.py`, `scripts/renormalize_judge_results.py`,
  `scripts/validate_responses.py`.
- **⚙️ Adapt:** confirm your benchmark's judge encodes `no_decision` distinctly from `fail`
  (both must map to NaN, not 0), and confirm its optional convention (see §6 adapt-notes).

### 1.2 Q-matrix / skills (the fitted bank)
- **Skills per benchmark** are defined by hand (a small set of latent tutoring/answer abilities),
  then each criterion is mapped to the skill(s) it requires.
- **✅ Generate + verify the mapping:**
  - `scripts/generate_qmatrix.py` — one LLM label per criterion under a strict rubric
    (SYSTEM_PROMPT; **default every skill 0**, mark 1 only if unsatisfiable without it; emits
    per-skill evidence/why/counterfactual + `explicitness`/`objectivity`/`criticality`).
  - `scripts/verify_qmatrix.py` — **two blind, cross-provider verifiers** re-label from scratch →
    3-rater panel → raw agreement + Cohen κ (per verifier) + **Fleiss κ** + a non-unanimous
    **human-review queue** + a 0-label false-negative audit.
  - `scripts/prepare_qmatrix_human_review.py` → blind reviewers A/B/C →
    `scripts/analyze_qmatrix_human_review.py` (Fleiss κ per skill, AI-vs-consensus accuracy,
    adjudication queue) → `scripts/finalize_qmatrix.py`.
  - curation/renumber: `scripts/apply_curation.py` + `scripts/verify_curated.py`;
    freeze bank via `scripts/build_final_rubrics.py`, `scripts/assign_irt_params.py`
    (or `estimate_placeholder_params.py`), `scripts/export_calibrated_bank.py`.
- **Fitted-bank JSONL** (one record per criterion) — the format every downstream tool expects:
  ```json
  {"criterion_id": "...", "scenario_id": "...",
   "discrimination": {"correctness": 1.61, "scaffolding": 0.0},   // per-skill a_k
   "difficulty": -0.25,                                            // b (offset scale)
   "q_modeled": {"correctness": 1, "scaffolding": 0},              // confirmatory 0/1 mask
   "optional": false,                                              // gating vs optional
   "exclude_from_fit": false}                                      // keep in bank, drop from the fit
  ```
- **`exclude_from_fit`**: a criterion stays in the bank (and can be administered) but is dropped from
  the parameter fit — used for the reverse-behaving / degenerate items we didn't want polluting the
  calibration. `calibrate_mirt.py` honors this flag natively.

---

## 2. Core calibration (confirmatory M2PL)

- **✅ Script:** `scripts/calibrate_mirt.py` — confirmatory **multidimensional 2PL** (M2PL), pure
  numpy/scipy **Bock–Aitkin EM/MML** over a Gauss–Hermite grid, loadings masked by the Q-matrix
  (`a_k` free iff `q_modeled[k]==1`, else exactly 0).
- **Key flags (verified):**

  | flag | default | meaning |
  |---|---|---|
  | `--matrix` | `staging/response_matrix.csv` | response matrix (use your **non-optional** matrix) |
  | `--rubrics` | `data/TutorBench/curated/rubrics_qmatrix_curated.jsonl` | bank with `q_mapping`/`q_modeled` |
  | `--grid` | `7` | GH nodes per latent dim (7 → 343 nodes for 3-D) |
  | `--ridge` | **`1e-2`** | L2 ridge on loadings (M-step stability) — **we used 1e-2** |
  | `--estimate-latent-corr` | off | estimate the K×K latent correlation from the posterior |
  | `--collapse a,b` | off | fit-time diagnostic: merge skills `a,b` into one dim (bank untouched) |
  | `--efa` | off | scree/EFA diagnostic (needs `factor_analyzer`; guarded) |
  | `--min-persons-identifiable` | `150` | warns when N < this (identifiability floor) |
  | `--report-only` / `--dry-run` | off | analysis only |
  | `--write-params --out-rubrics <f>` | `data/rubrics_qmatrix_mirt.jsonl` | write a **new** fitted bank (never edits the input) |

- **Missing data:** holes are marginalised in the E-step (never imputed).
- **Zero-variance & Q-less dropping:** all-NaN columns, observed-but-all-fail / all-pass items, and
  all-zero-`q_modeled` rows are dropped **before** fitting (they carry no information). `exclude_from_fit`
  rows are dropped too. TutorBench: **3,666 of 6,180** non-optional criteria actually entered the 2-skill fit.
- **Everything is scenario-level in spirit** — criteria are grouped under their `scenario_id`; the CAT
  administers whole scenarios (see §4/§5) even though the *fit* is per-criterion. If your criteria are
  **not** nested in scenarios, see §6 adapt-notes.
- **Emits:** `calibration_mirt.csv` (per-item `a_<skill>`, `b`, `n_persons`, flags) +
  `calibration_mirt_manifest.json` (dropped-item accounting, uni-vs-multi loglik/AIC/BIC, latent
  correlation, and the `--collapse` comparison block).

---

## 3. Out-of-sample generalization (person-level k-fold)

- **✅ Script (present, cell-level):** `scripts/kfold_cv_mirt.py`
  - flags: `--k 5 --seed 20260729 --grid 7 --ridge 1e-3 --matrix <...> --rubrics <...> --out-dir staging/kfold`
    (`--estimate-latent-corr` optional). Note this file's `--ridge` **default is 1e-3** — pass `--ridge 1e-2`
    to match the core fit if you want identical settings.
  - What it does: partition models into `k` folds; per fold **refit `(a,b)` on TRAIN models**, freeze,
    then **EAP-score held-out TEST models** from their own responses and predict every held-out cell.
  - Emits: `metrics_per_fold.csv`, `metrics_aggregate.json` (**pooled OOS log-loss / accuracy / AUC /
    Brier**, per-fold spread, and the **OOS − in-sample optimism gap**), `fold_<f>_item_params.csv`,
    `item_param_stability.csv` (cross-fold Pearson corr of `a_correctness`/`a_scaffolding`/`b` — a direct
    read on identifiability at your N).
- **⚠️ ABSENT (the θ-recovery OOS):** `scripts/scenario_kfold_estimator_cv.py` produced the
  `oos_per_model.csv` with columns **`theta_ref_{skill}`** and **`theta_{online,batch,mwle}_{skill}`** that
  ALL our OOS *recovery-r / recovery-slope* numbers came from (same k=5, seed=20260729, refit-per-fold,
  then run the real engine on held-out models and score MWLE θ vs the fold's full-bank EAP reference).
  It's not in this checkout. **Restore it** (it imported the present `tutor_cat.engine` + the scenario CAT
  lib) or reproduce its θ-recovery from `kfold_cv_mirt.py`'s fold params + `tutor_cat.engine`.

> **In-sample vs OOS:** in-sample recovery reuses the same models for fit and scoring (optimistic;
> TutorBench 2-skill correctness r ≈ 0.982). OOS (k-fold) is the honest number (correctness r ≈ 0.945,
> scaffolding ≈ 0.90). **Report OOS for headline recovery claims.**

---

## 4. Experiments to run (each: decides → script → key flags → the metric that settles it)

| # | Decision | Script | Key flags | Settling metric(s) |
|---|---|---|---|---|
| 4.1 | **# of skill dimensions** (1 vs 2 vs 3) | ✅ `calibrate_mirt.py` | `--estimate-latent-corr --collapse a,b --efa` | latent-corr, **AIC/BIC** (collapsed vs full), + recovery r/slope, precision %, mean scenarios |
| 4.2 | **>1 dim justified?** | ⚠️ `fig_composite_vs_axes.py` | — | composite (overall) θ correlates ≈ +1 with primary skill but **negatively** with the 2nd → axes aren't redundant |
| 4.3 | **Collapse correlated skills?** | ⚠️ `analyze_collinearity.py` (+ `calibrate_mirt --collapse`) | — | latent r between skills; **collapse if r ≳ 0.9 AND collapsed model wins AIC/BIC AND EFA shows no 2nd factor** |
| 4.4 | **Estimator** (online/Gaussian vs batch-EAP vs MWLE) | ⚠️ `scenario_kfold_estimator_cv.py` (θ-recovery OOS) | k=5, seed=20260729 | recovery **r + slope** per skill from `oos_per_model.csv` |
| 4.5 | **`min_scenarios` floor** (0/12/15/20) | ⚠️ `offline_engine_driver.py` (per floor) | `--min-scenarios {0,12,15,20}` | recovery r/slope + **mean scenarios** |
| 4.6 | **SE target** (0.20/0.25/0.30/0.35) | ⚠️ driver + ✅ `se_sweep_aggregate.py` + ⚠️ `fig_se_tradeoff.py` / `fig_total_se_vs_target.py` | `--max-se {..}` | **convergence %** + mean scenarios + total SE vs floor |
| 4.7 | **Selection rule** (trace vs D-opt) | ✅ `tutor_cat/selector.py` via ⚠️ `offline_engine_driver.py --selection {trace,dopt}` | `--selection` | test size + recovery |
| 4.8 | **Order/seed dependence** | ⚠️ `scenario_order_experiment.py` | multiple seeds | across-seed **SD/range of θ** as a fraction of the SE target |
| 4.9 | **Parameter (calibration) uncertainty** | ⚠️ `scenario_param_uncertainty.py` + ⚠️ `frq_total_se.py` | bootstrap B | per-model `se_param`; **SE_total = √(se_ability² + se_param²)** |
| 4.10 | **Full CAT leaderboard** | ✅ `tutor_cat.engine.run_evaluation` (via ⚠️ `offline_engine_driver.py` / `cat_eval_tutorbench_multiskill.py`) | RunConfig | per-model θ, SE, scenarios/criteria administered, `precision_reached` |

**Present primitives the ABSENT wrappers were built on** (so you can rebuild/restore any row above):
- `tutor_cat/engine.py` → `run_evaluation(bank, tutor, judge, cfg, mode="cat"|"baseline", run_id)`.
  `RunConfig(seed, top_n, max_se={skill:0.30}, min_evals_per_skill, min_scenarios, max_scenarios,
  selection="trace"|"dopt", skills, unmapped_criteria, write_logs)`. Writes `steps.jsonl` (per-scenario
  θ + `se` + counts), `criterion_updates.jsonl` (`se_after`), `final_result.json`
  (`theta`, `se`, `stop_reason`, `precision_reached`, `scenarios_administered`).
- `tutor_cat/selector.py` → `select_next(..., selection="trace"|"dopt")` (trace = PRD per-criterion Fisher
  rule for the argmax-SE skill; dopt = D-optimality log-det gain).
- `calibrate_mirt.py` internals reused by the wrappers: `collapse_q_matrix`, `prepare_block`,
  `fit_m2pl_em`, `build_grid`, `prior_log_weights`.
- **MCQ track (✅ present, separate):** `tutor_cat/mcq_irt/` — unidimensional 2PL calibration + CAT over
  right/wrong grids (`pipeline.py`, `calibrate.py`, `cat.py`, `ability.py`: `eap`, `fisher_information`,
  `prob_2pl`; `matrix.py`, `report.py`). Default MCQ benchmarks: `arc_challenge, arc_easy, openbookqa, sciq`.

**TutorBench results (worked example):** collapsed content+diagnosis at latent **r ≈ 0.95** (AIC/BIC
favored collapse) → `[correctness, scaffolding]`; chose **MWLE** (EAP estimators shrink to the prior →
slope 0.64–0.76; MWLE restores slope ≈ 1.0 without losing r); locked **min_scenarios = 12** and
**SE target = 0.30**; **D-opt did not shorten** scenario-level tests → kept **trace**.

---

## 5. Metrics glossary (units; in-sample vs OOS; producing script)

| Metric | Definition / units | Notes | Produced by |
|---|---|---|---|
| **recovery correlation r** | corr(θ_est, θ_ref) | in-sample (fit=score models) vs OOS (k-fold) — always label which | ⚠️`frq_mae_analysis.py` / metrics.json |
| **recovery slope** | OLS slope of θ_est ~ θ_ref | **<1 = compression** (shrinkage); ≈1 ideal | ⚠️`frq_mae_analysis.py` |
| **θ-MAE** | mean\|θ_est − θ_ref\| in **logits** | ability-scale error | ⚠️`frq_mae_analysis.py` |
| **pass-rate MAE** | mean\|p̂ − p_obs\|, **0–1** | interpretable (observable) calibration error | ⚠️`frq_mae_analysis.py` |
| **p-IRT predicted-vs-actual pass rate** | model-predicted vs observed pass rate | calibration-to-observable check, **distinct from θ-recovery** | ⚠️`frq_mae_analysis.py` |
| **SE_post / ability-only SE** | posterior SD of θ, **conditional on item params** | overconfident (ignores calibration error) | ✅ engine `final_se_*`; ⚠️`scenario_param_uncertainty.py` |
| **SE_param** | SD of θ across a parameter bootstrap | calibration (item-parameter) uncertainty | ⚠️`scenario_param_uncertainty.py` |
| **SE_total** | √(SE_ability² + SE_param²) | quadrature-combined honest SE | ⚠️`frq_total_se.py` |
| **convergence % / `precision_reached`** | fraction of models hitting the SE target before the cap | | ✅ engine `final_result.json`; ✅`se_sweep_aggregate.py` |
| **mean scenarios / criteria administered** | test length (count) | efficiency | ✅ engine; ✅`se_sweep_aggregate.py` |
| **efficiency vs random baseline** | adaptive vs random-order (**same stop rule**) length & recovery | run random with `--mode baseline`, **uncapped** (`--max-scenarios` = full bank) | ⚠️`offline_engine_driver.py --mode baseline` |
| **cell-level OOS** (log-loss / acc / AUC / Brier) | held-out cell prediction quality + optimism gap | the *present* k-fold's outputs | ✅`kfold_cv_mirt.py` |
| **item-param cross-fold stability** | Pearson corr of `a`,`b` across folds | identifiability at your N | ✅`kfold_cv_mirt.py` |
| **bootstrap CIs** | 95% CIs by resampling the models, **B=2000, seed=0** | on r / slope / MAE and on the figures' bands/error bars | ⚠️`frq_mae_analysis.py` |

---

## 6. Locked defaults (from TutorBench) + what to RE-SWEEP

| Setting | TutorBench-locked | Reuse as-is? |
|---|---|---|
| estimator | **MWLE** | ✅ reuse (EAP shrinks; MWLE de-biases the slope) |
| selection rule | **trace** (Fisher) | ✅ reuse (D-opt didn't help at scenario level) |
| k (folds) | **5**, seed 20260729 | ✅ reuse |
| `top_n` (selection candidate pool) | **5** | ✅ reuse |
| `min_evals_per_skill` | **15** | ✅ reuse (re-check if a skill is very thin) |
| GH grid | **7 nodes/dim** | ✅ reuse (drop for high dim cost) |
| **ridge** | **1e-2** | ⚠️ **RE-SWEEP** — stability depends on N & sparsity |
| **skill count / definitions** | 2 (correctness, scaffolding) + optional presentation | ⚠️ **RE-DERIVE per benchmark** |
| **collapse decision** | collapse @ latent r ≈ 0.95 | ⚠️ **RE-DECIDE** (latent-corr + AIC/BIC + EFA) |
| **`min_scenarios` floor** | **12** | ⚠️ **RE-SWEEP** (0/12/15/20) |
| **SE target** | **0.30** | ⚠️ **RE-SWEEP** (0.20–0.35) |
| `max_scenarios` cap | 50 for adaptive; **uncapped (= full bank)** for the random baseline | ⚙️ set to your bank size |

**Adapt-notes for a different benchmark structure:**
- **Criteria NOT nested in scenarios** (e.g. one item = one prompt, no scenario grouping): the M2PL fit
  and `kfold_cv_mirt.py` are unaffected (they're per-item), but the CAT "administer a whole scenario"
  logic collapses to "administer one item" — set the scenario granularity to the item, and the testlet /
  within-scenario local-dependence concern disappears. The MCQ track (`tutor_cat/mcq_irt`) is exactly this
  single-item-per-scenario regime and is the cleaner template for such benchmarks.
- **Different optional convention:** `merge_full_matrix.py` keys "optional" off `rec["optional"] is True`.
  If your benchmark marks non-gating criteria differently (e.g. a `criticality`/`weight` field, or none),
  adjust the gating filter so the fit uses only gating criteria and the full matrix is used only for the
  optional axis.
- **Testlet / local dependence:** if criteria ARE nested in scenarios, within-scenario pairs are locally
  dependent (TutorBench: within-scenario response-correlation ≈ 2× across-scenario). This inflates any
  limited-information fit statistic; prefer a scenario-aggregated or testlet/bifactor model if you report M2/C2.

---

## 7. Recommended order of operations

1. **Build the response matrix** from judge verdicts → `run_judge_grading.py` → `merge_full_matrix.py`
   (get both `_full` and `_nonoptional`). Audit with `audit_judge_verdicts.py`.
2. **Define skills + Q-matrix** → `generate_qmatrix.py` → `verify_qmatrix.py` → human review
   (`prepare_/analyze_qmatrix_human_review.py`) → `finalize_qmatrix.py` → freeze the fitted bank
   (`build_final_rubrics.py` / `assign_irt_params.py` / `export_calibrated_bank.py`).
3. **Core M2PL fit** on the non-optional matrix → `calibrate_mirt.py --estimate-latent-corr` (+ `--efa`).
4. **OOS k-fold** → `kfold_cv_mirt.py` (cell-level optimism gap + item-param stability); restore
   `scenario_kfold_estimator_cv.py` for θ-recovery-r if you need it.
5. **Dimensionality:** composite-vs-axes (`fig_composite_vs_axes.py`) + collinearity
   (`analyze_collinearity.py`) + `calibrate_mirt --collapse` AIC/BIC → decide skill count / collapses.
6. **Estimator** (MWLE vs EAP): confirm from the θ-recovery OOS (default: keep MWLE).
7. **Floors:** sweep **`min_scenarios`** then **SE target**; aggregate with `se_sweep_aggregate.py`;
   pick the smallest floor / loosest SE that reaches ~100% convergence without lengthening tests.
8. **Parameter uncertainty / total SE** → `scenario_param_uncertainty.py` → `frq_total_se.py`
   (report SE_total, not just ability-only SE).
9. **Full CAT leaderboard + efficiency vs random** → `tutor_cat.engine.run_evaluation` (adaptive) and
   `--mode baseline` **uncapped** (random) → per-model θ/SE/length/precision, and the adaptive-vs-random
   efficiency figure.

---

## Appendix — quick script index

**✅ present:** `run_judge_grading.py`, `run_all_judge_grading.py`, `stage_judge_inputs.py`,
`ingest_calibration_csv.py`, `audit_judge_verdicts.py`, `renormalize_judge_results.py`,
`merge_full_matrix.py`, `generate_qmatrix.py`, `verify_qmatrix.py`,
`prepare_qmatrix_human_review.py`, `analyze_qmatrix_human_review.py`, `finalize_qmatrix.py`,
`apply_curation.py`, `verify_curated.py`, `build_final_rubrics.py`, `assign_irt_params.py`,
`estimate_placeholder_params.py`, `export_calibrated_bank.py`, `calibrate_mirt.py`,
`calibrate_partial.py`, `kfold_cv_mirt.py`, `se_sweep_aggregate.py`,
`tutor_cat/engine.py`, `tutor_cat/selector.py`, `tutor_cat/mirt.py`, `tutor_cat/cli.py`,
`tutor_cat/mcq_irt/*`.

**⚠️ scenario-level wrappers (pruned from the per-benchmark branches by design; recover from pre-prune git history or the TutorBench branch):** `offline_engine_driver.py`,
`scenario_cat_lib.py`, `scenario_kfold_estimator_cv.py`, `frq_mae_analysis.py`, `frq_total_se.py`,
`scenario_param_uncertainty.py`, `scenario_order_experiment.py`, `analyze_collinearity.py`,
`fig_composite_vs_axes.py`, `fig_se_tradeoff.py`, `fig_total_se_vs_target.py`,
`cat_eval_tutorbench_multiskill.py`, and the `regenerated_figures/` outputs.

---

## 8. Cross-benchmark lessons & recommended defaults

Hard-won rules that generalize across benchmarks. Follow them by default; deviate only with a
recorded reason. Numbers in parentheses are worked examples (mostly Bridge, N≈51 models) — treat
them as illustrations of the failure mode, not as targets.

### 8.1 Response matrix & folds
- **Fit and score on a dense matrix, and re-filter zero-variance items *within every fold*.** An item
  that varies across the full sample can be all-pass or all-fail *inside a given train fold*; such items
  carry no information there and, if left in, silently corrupt that fold's fit. Apply the same
  all-NaN / all-pass / all-fail / all-zero-`q_modeled` drop **independently on each fold's training rows**,
  not once globally. Skipping this is the single most damaging small-N bug we hit: it produced a bogus
  headline held-out recovery of **r = 0.226** that reconciled to **r ≈ 0.967** once within-fold filtering
  was applied consistently. If your OOS number is wildly worse than in-sample, suspect this first.
- **Use source-grouped, leakage-safe folds.** Partition so that responses sharing a leakage channel —
  the same underlying prompt/source, model family, or judge instance — never straddle the train/test
  boundary. Random cell-level splits leak and inflate OOS; group by the person (model) and by source.
- **Audit the input matrix before fitting.** Print and check its dimensions (rows × cols), the missingness
  pattern ("holes"), and the **judge / frozen-config hash** that produced it. Confirm `no_decision` maps to
  NaN (not 0) and that the optional/gating split is what you expect. Fitting a matrix you have not audited
  is how silent shape and provenance bugs enter the leaderboard.

### 8.2 Standard errors & uncertainty
- **Report `SE_total = SE_ability ⊕ SE_param` (quadrature) on every leaderboard**, never ability-only SE.
  At small N the **calibration (parameter) term dominates** — item parameters are themselves poorly pinned,
  so an ability-only SE is badly overconfident. Ranking and "is model A ≠ model B" claims must use SE_total.
- **Estimate `SE_param` with an observed-information / parametric bootstrap, not jackknife.** Jackknife
  (leave-one-model-out) goes **degenerate at small N** — it produced zero or near-zero `se_param` for a large
  fraction of models (e.g. 13 of 51 Bridge models), which then vanish from SE_total and fake precision.
  A resampling/observed-information bootstrap stays well-defined in the same regime.

### 8.3 OOS / recovery discipline
- **Headline recovery must be held-out.** Never quote an in-sample recovery r (fit and score the same
  models) as the validity number; it is optimistic by construction.
- **Keep two distinct OOS artifacts and make them reconcile.** (a) a *simple EAP full-bank k-fold sanity
  check* (refit per fold, EAP-score held-out models on all items) and (b) the *deployed MWLE-CAT recovery*
  (the estimator + selection + stop rule you actually ship). These answer different questions but **must
  use identical within-fold filtering and the identical operating point** (same folds/seed, same item
  drops, same `min_scenarios`/SE target). When they do, they line up (Bridge: EAP sanity r ≈ 0.967,
  deployed MWLE-CAT recovery in the same neighborhood); when they diverge, you have a filtering or
  operating-point mismatch, not two legitimately different truths.
- **Distinguish deployment length from k-fold length.** The number of scenarios/criteria the CAT
  administers under the deployed stop rule is *not* the same quantity as the fixed length used inside a
  k-fold recovery experiment. Label every "test length" number with which regime produced it; do not
  compare a k-fold length against a deployed length.

### 8.4 Dimensionality at small N
- **Do not choose dimensionality by AIC alone.** AIC's penalty is too weak at small N and will
  over-select dimensions. Decide with **BIC + held-out log-loss + a parsimony bias**; add an axis only if
  it earns its keep out of sample.
- **Watch for latent-correlation collapse toward 1 as you add dimensions.** The diagnostic failure mode at
  small N is that the estimated inter-axis correlations run to ±1 — the axes are not separately identified
  (Bridge: max off-diagonal latent correlation climbed **0.77 → 0.967 as dims went 2 → 5**). This
  collapse, **not** "missing anchor items", is the identification problem to name in the writeup.
- **Report the per-axis discriminations (`a_k`) before making any (non-)identification claim.** Whether an
  axis is real is a statement about its loadings; show them (and their spread/CIs) to support or retract the
  claim rather than asserting it from fit indices alone.

### 8.5 Order/seed robustness & figures
- **Assess order/seed robustness from the production start, not a random first item.** Seed the CAT with
  the real **max-information first-item** rule you deploy, then perturb order/seed; a random first item
  measures a device you never ship and overstates instability.
- **Put CI bands on every line figure, and report slope/bias — not just r.** A high r can hide a
  compressed (slope < 1) or biased estimator; recovery figures must show the band and the slope/intercept,
  and leaderboard figures must show SE_total error bars.

### 8.6 Reproducibility & study layout
- **Record config-decision provenance and commit the runner.** Persist every locked choice
  (dims, ridge, estimator, floors, SE target, folds/seed, judge hash) to a `study_config.json` /
  `study_manifest.json`, and commit the driver script (`run_<bench>_study.sh` + the per-experiment scripts)
  so the whole study **reproduces from the committed tree** — not from an uncommitted working copy.
- **Use the standard study folder convention.** A benchmark study is a base package
  (`<bench>_calibration/` with `study_config.json`, `study_manifest.json`, top-level
  `model_leaderboard.csv` / `recovery.json`, a `scripts/` dir, and a `README.md`) plus numbered
  `experiments/` directories `03_structures`, `04_efficiency_vs_random`, `05_oos_recovery`,
  `06_floor_se_grid`, `07_parameter_uncertainty`, `08_leaderboard`, `10_estimator_comparison`,
  `11_order_seed` (03..12). Each experiment writes its metrics/CSVs at its own root and its plots under a
  `figures/` subdirectory. Keeping this layout stable is what lets one review script and this playbook
  apply unchanged to a new benchmark.

### 8.7 CAT stopping rule — use the EAP posterior SD (suite default)
- **Stop on the grid-integrated EAP posterior SD, not the engine's online/Laplace SE.** The online
  normal-approx SE (curvature at the point estimate) is **optimistic**: on skewed posteriors (few
  informative items / extreme-ability models) it declares "converged" while the true posterior SD is
  still above target. At each candidate stop step, integrate the posterior over a **dense** θ grid from
  the *administered* items and stop when its SD ≤ target (AND `min_scenarios`); this is cheap
  (grid × items-so-far, ms/step) and reuses the same estimator as the recovery reference, resolving the
  online-vs-posterior estimator mismatch. **Multidimensional:** take each skill's *marginal* SD from the
  **joint** grid (not independent 1-D integrals — skills are correlated).
- **Adopted across the in-house suite (bank UNCHANGED — stop-rule change only, no re-fit):** the honest
  "% reaching target" jumped everywhere — BiGGen 28.8→88.5%, WildBench 65→86.5%, Bridge (SE-ability)
  78→100% — recovery held or improved (e.g. BiGGen r 0.972→0.982, WildBench 0.957→0.975), and the SE_total
  tail (SD/max) shrank. Cost is length: median often unchanged, mean **+2–6 scenarios** as only the tail
  models administer more. Locked op-points: **WildBench floor 8 / SE 0.12, BiGGen floor 8 / SE 0.12,
  Bridge floor 12 / SE 0.12.**
- **Target SE-ability ~0.12 to keep SE_total under ~0.15.** The stop targets **SE_ability**, but the
  honest bar is **SE_total = √(SE_ability² + SE_param²)** with SE_param ≈ 0.05 at these N. So an
  SE-ability target of 0.15 leaves a fat SE_total tail (Bridge: 14 models > 0.15); tightening to **0.12**
  collapses it (→ 2). Pick the SE target against SE_total, not SE_ability.
- **The SE target is a TAIL lever, not a median lever.** On heavy-testlet benchmarks the median model is
  already below target once the floor is met, so `se_total_median` is ~flat across SE targets and hides
  the benefit. **Report per-cell mean / SD / max SE_total and #(SE_total > threshold)** — the SE target's
  effect lives in the tail.
- **Adoption recompute-vs-keep:** re-run only the **stop-dependent** experiments (recovery, efficiency,
  estimator, order/seed, deployed-SE + **SE_param re-bootstrapped on the EAP-administered sets**); **keep**
  the stop-independent ones (dimensionality, ridge, the **full-bank** SE_param floor, and the **full-bank**
  leaderboard θ — full-bank scoring doesn't use the CAT stop). **Archive the online-SE of-record**
  (`experiments/archive_onlineSE/`) and keep the online path behind a `--stop-se {online,eap}` flag.
- **Prerequisite:** the reference/scoring EAP must already be a **dense** grid. A coarse quadrature grid
  quantizes θ onto discrete nodes and makes the posterior SD meaningless (e.g. InfoBench's 5-node grid) —
  do the dense-grid rescore **first**, then adopt the EAP stop.
- **Deployment:** keep the fixed-production-seed policy; models with no bank information at their θ
  (near-all-fail extremes) still cap out and stay flagged "weakly identified" (θ bound only) regardless of
  the stop rule.
