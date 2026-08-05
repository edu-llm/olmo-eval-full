# Bridge Scenario-Level Rerun Plan

**Status:** planning memo (read-only audit; no code changed, no experiments run).
**Author:** calibration audit pass.
**Scope:** redo the entire Bridge calibration study at **scenario granularity**, matching the
TutorBench scenario-level pipeline (`scripts/scenario_*` + `tutor_cat.engine`), and scrap the
existing **item/criterion-level** Bridge study under `eduLLM-Evals/bridge_calibration/`.

---

## 0. One-paragraph problem statement

Bridge is scenario-structured: **250 scenarios / 4795 criteria** (`data/Bridge/rubrics.jsonl`
maps `criterion_id -> scenario_id`; `data/Bridge/scenarios.jsonl` maps `scenario_id -> source_id`;
**162 unique `source_id`s**, 176 scenarios share a source). The committed Bridge study
(`bridge_calibration/scripts/*`) fits and administers at the **criterion (item) level**: the
response matrix columns are individual criteria (`bridge_0000_c01` … 51×4795), the CAT selects
**one criterion at a time** by Fisher info, and `a/b/info/SE` are all per-criterion. TutorBench
was instead calibrated **scenario-level**: the real engine (`tutor_cat.engine.run_evaluation` +
`selector.select_next`) administers a **whole scenario** (all its criteria graded together) per
CAT step. All Bridge item-level experiment outputs (03–12) and every operating-point decision
(SE 0.15 / floor 20, item-level `SE_param`) must be scrapped and re-derived scenario-level.

---

## Part A — Per-script verdict (audit result)

**VERDICT: ALL Bridge experiments are item/criterion-level. No exceptions.**

The unit of the response-matrix columns is the **criterion**; the unit of CAT selection is the
**criterion**; and `a`, `b`, Fisher information, and SE are all computed **per criterion**.
`source_id` grouping appears only as a *column dedup filter* (keep one scenario's criteria per
source), which is **not** scenario-level administration or scoring.

| Script | Level | Code evidence |
|---|---|---|
| `bridge_calibration_study.py` | **item** | Matrix columns are criteria; `fit_items()` builds `Q = np.ones((Y.shape[1],1))` per **column=criterion** and calls `cm.fit_m2pl_em`; `eap_theta` sums Fisher `a^2·P·Q` over criterion columns; `item_params.csv` header is `criterion_id,a,b`. `dedupe_by_source` only **drops criterion columns** whose scenario isn't the source-representative (still per-criterion). |
| `bridge_dimensionality.py` | **item** | `items = list(sub.columns)` = criteria; `build_Q` makes one row **per criterion**; `fit_m2pl_em(Yk, Mk, Qk,…)` fits per-criterion loadings; held-out log-loss `marginal_logloss` is per-criterion-cell. Dedup is column-level. |
| `bridge_cat_vs_random.py` | **item** | `adaptive_theta` maintains `pool = list(obs_idx)` of **criterion indices**, picks `k = pool[argmax(a^2·P·Q)]` one **criterion** per step; "criteria administered" x-axis; `random_theta` samples criteria. Pure item-level CAT. |
| `bridge_recovery_plots.py` | **item** | `run_cat(y,a,b,min_items,se_target)` administers **one criterion** per step (`j = argmax(info)`, `info[used]=-1`), `min_items`/`se_target` counted in **criteria**; MWLE over criterion subset. |
| `bridge_se_floor_sweep.py` | **item** | `cat_trajectory` greedy Fisher over **criteria**; `read_off(ths,ses,se_target,floor)` where `floor = min criteria`; sweep grid `floor-list "0,4,8,12,16,20"` is **criteria counts**, `se-list` is criterion-level SE. This is where the item-level operating point comes from. |
| `bridge_param_uncertainty.py` | **item** | `Q=np.ones((J,1))` (J=criteria); `SE_ability = 1/sqrt(sum a^2 P Q)` over criteria; parametric bootstrap redraws each **criterion's** `(a,-b)` from per-criterion observed info; leaderboard `se_param` is item-level. |
| `bridge_param_uncertainty_bootstrap.py` | **item** | `item_param_cov_chol` inverts a **per-criterion** 2×2 Hessian; `bootstrap_se_param` redraws the full **criterion**-parameter set and re-EAPs. Identical item-level data prep. |
| `bridge_sensitivity.py` | **item** | `prep()` returns criterion columns; `cat_order`/`exp_estimator` reuse `brp.run_cat` (item-level CAT); `exp_sensitivity` fits `np.ones((J,1))` per criterion; `exp_order_seed` perturbs the **criterion** administration order. |
| `bridge_pick_operating_point.py` | **item** | Reads `sweep_results.csv` (`floor` = min criteria) and stamps `SE 0.15 / floor 20` — an **item-level** operating point. No scenario concept. |
| `run_bridge_study.sh` | **item** | Orchestrates all of the above with `--min-items "$FLOOR"` (FLOOR=20 **criteria**); banner "Locked CAT operating point: SE 0.15 / floor 20"; passes the 51×4795 **criterion** matrix throughout. |

**Ingest path — `scripts/ingest_bridge.py`:** emits `scenarios.jsonl` (one row per scenario,
`criterion_ids` list) and `rubrics.jsonl` (one row per **criterion**, each with its `scenario_id`,
`q_mapping` over 5 skills). The response matrix `bridgegrade/response_matrix.csv` (**51 models ×
4795 criteria**, confirmed header `bridge_0000_c01,…`) therefore has **per-criterion columns**.
The docstring itself flags the scenario structure and the `source_id` local dependence:
"one scenario per ROW, so … `source_id` is intentionally NON-unique … That is genuine local
dependence: group or hold out on `source_id` at calibration."

**On `source_id` grouping (folds/dedupe):** every Bridge script's `dedupe_by_source(...)` groups
scenarios by `source_id`, keeps the first scenario per source, and retains only that scenario's
**criterion columns**. This is a *leakage-safe dedup of duplicated stimuli* — it is **NOT**
scenario-level administration or scoring. After dedup the matrix is still a wide table of
individual criteria, fit and administered one criterion at a time.

---

## Part B — How the TutorBench scenario-level template works (mechanics)

Reference files: `scripts/scenario_cat_lib.py`, `scripts/offline_engine_driver.py`,
`scripts/scenario_selection_experiment.py`, `scripts/scenario_order_experiment.py`,
`scripts/scenario_param_uncertainty.py`, `scripts/scenario_kfold_estimator_cv.py`, and the
production engine `tutor_cat/engine.py` + `tutor_cat/selector.py`.

**1. A scenario is a bundle administered together — NOT one aggregated/polytomous item.**
`tutor_cat.engine.run_evaluation` picks a `scenario_id`, then grades **every criterion in that
scenario** (`for rubric in bank.rubrics_for(sid): … update(theta,U,rubric.a,rubric.q,rubric.b,y)`).
Each criterion keeps its own binary `(a,b,q)` and updates the multidimensional ability `theta` and
posterior covariance `U` via the M2PL update. So the *fit* is still per-criterion (`calibrate_mirt`
is per-item), but the **unit of administration/scoring/stopping is the whole scenario** — all of a
scenario's criteria arrive as one testlet. There is no per-scenario score collapse and no polytomous
recoding; the criteria are administered *as a group*.

**2. Fisher information and CAT selection at scenario granularity** (`selector.py`):
- Per-criterion Fisher info for the target skill `k`: `V_kc = q_kc · P_c(1-P_c) · a_kc²`.
- **ScenarioValue(S,k) = (Σ_{c∈S} V_kc) / #{c∈S : q_kc=1}** — total info about skill `k` from the
  scenario's criteria, **normalised per applicable criterion** (info per unit judging cost).
- `select_next`: target skill = `argmax(se)`; rank scenarios by ScenarioValue descending; take
  `top_n` (=5); pick one with the seeded RNG. `dopt` alternative scores the **log-det gain** of
  administering the whole scenario against the posterior covariance `U` (also per-scorable-criterion).
- Fallback when no unused scenario touches the target skill: `total_information_value` summed across
  all skills, per criterion.

**3. SE target and the min-scenarios floor** (`engine.py` `RunConfig`):
- Stop rule (checked **between scenarios**): `precision_reached()` iff `(se < max_se).all()` **and**
  `(counts >= min_evals_per_skill).all()`, AND `len(administered) >= min_scenarios`; else stop at
  `max_scenarios`.
- **`max_se`** is a per-skill SE **target** (TutorBench-locked **0.30**). **`min_scenarios`** is the
  minimum **number of scenarios** administered before a precision stop is allowed (TutorBench-locked
  **12**). This floor is a stronger minimum-test-length guarantee than `min_evals_per_skill` (which
  counts criteria). `top_n=5`, `min_evals_per_skill=15`, GH grid 7 nodes/dim.
- **How TutorBench chose the operating point** (Calibration Playbook §4.5–4.7, §6): sweep
  `min_scenarios ∈ {0,12,15,20}` then `SE target ∈ {0.20,0.25,0.30,0.35}` with the real engine
  (`offline_engine_driver.py --min-scenarios/--max-se`), aggregate with `se_sweep_aggregate.py`,
  and **pick the smallest floor / loosest SE that reaches ~100% convergence without lengthening
  tests**. TutorBench also compared SE-target rows against the **`SE_param` floor** (the
  calibration-uncertainty floor, ~0.21/0.25) so it didn't chase an ability-SE tighter than the
  irreducible parameter noise. Estimator = **MWLE** (EAP shrinks slope to ~0.64–0.76; MWLE restores
  slope ≈ 1 at equal r); selection = **trace** (D-opt didn't shorten scenario-level tests).

**4. Artifacts each scenario script produces:**
- `offline_engine_driver.py` → `cat_per_model.csv` (per-model θ_full / θ_cat / θ_batch / θ_mwle,
  final SE, scenarios & criteria administered, stop reason) + `metrics.json` (recovery r/slope,
  stop reasons, lengths) + recovery scatters.
- `scenario_selection_experiment.py` → `per_model_{trace,dopt}.csv`, `metrics.json` (length,
  recovery, per-skill convergence).
- `scenario_kfold_estimator_cv.py` → `oos_per_model.csv` (`theta_ref_{d}`, `theta_{online,batch,
  mwle}_{d}`) + `metrics.json` (**OOS** recovery r/slope per estimator) — the honed held-out number.
- `scenario_param_uncertainty.py` → `leaderboard_se_components.csv` (`se_posterior`, `se_param`,
  `se_total`, bar inflation per skill) + `metrics.json`.
- `scenario_order_experiment.py` → `per_model_spread.csv` + `metrics.json` (across-seed θ SD/range).
- Fit provenance: fitted bank JSONL keyed by **modeled-skill** names with `discrimination{}`,
  `difficulty`, `q_modeled{}`, and an `irt_params.provenance.matrix_sha256` that
  `verify_provenance()` cross-checks against the response matrix.

---

## Part C — Ordered rerun plan

### C.1 Reusable vs scrap

**Reuse as-is (raw inputs — the judge verdicts are level-agnostic):**
- `bridgegrade/response_matrix.csv` (51×4795 **per-criterion** 0/1 matrix) and
  `bridgegrade/verdicts.jsonl` / `manifest.json` — the criterion verdicts are the raw signal;
  scenario-level administration is just a *different grouping/selection* over the same cells.
- `data/Bridge/rubrics.jsonl` (criterion → scenario_id, `q_mapping` over the 5 skills) and
  `data/Bridge/scenarios.jsonl` (scenario_id → source_id, `criterion_ids`) — the maps that make
  scenario-level grouping possible.
- The whole TutorBench scenario toolchain: `scenario_cat_lib.py`, `offline_engine_driver.py`,
  `scenario_selection_experiment.py`, `scenario_kfold_estimator_cv.py`,
  `scenario_param_uncertainty.py`, `scenario_order_experiment.py`, plus `tutor_cat/{engine,
  selector,mirt,dataio,schemas}.py` and `calibrate_mirt.py` / `kfold_cv_mirt.py`.

**Scrap / do not cite (all item-level, superseded):**
- Every `experiments/03…12` output under `bridge_calibration/` (dimensionality, CAT-vs-random,
  OOS recovery, floor/SE grid, param-uncertainty, leaderboard, estimator, order/seed, sensitivity).
- The **item-level operating point `SE 0.15 / floor 20`** (`06_floor_se_grid/best.json`,
  `bridge_pick_operating_point.py`) — void; floor was **20 criteria**, not scenarios.
- The item-level `SE_param` / `se_total` leaderboard (`07_parameter_uncertainty/*`,
  `model_leaderboard.csv`) and the item-level `recovery.json` headline (r=0.945 @ item CAT).
- `bridge_calibration/scripts/bridge_*` — replaced by scenario scripts/adapters below. Keep on disk
  for provenance only; they are not part of the new study.

### C.2 Build step — scenario-level representation

The engine consumes **exactly the same three inputs** as TutorBench, so *no aggregation into a
single per-scenario score is needed*: the bundle is administered together and the per-criterion
verdicts come straight from the existing matrix.

1. **Fit a scenario-level fitted bank** with `calibrate_mirt.py --write-params` on the **criterion**
   matrix, with the Q-matrix taken from `rubrics.jsonl` (`q_mapping` over the 5 Bridge skills, or a
   collapsed skill set — see C.4). Output must be the **fitted-bank schema** the scenario lib expects:
   per-criterion `{criterion_id, scenario_id, discrimination{skill:a_k}, difficulty:b,
   q_modeled{skill:0/1}}` + an `irt_params.provenance.matrix_sha256` of the matrix. Drop
   zero-variance / all-zero-`q` criteria **within each fit** (and within each fold).
2. **Source-dedup at the SCENARIO level** for the bank/administration set: keep one scenario per
   `source_id` → **~162 scenarios survive** (from 250; 176 currently share a source). Hold out on
   `source_id` in folds regardless (§8.1 of the Playbook). *Decision to confirm:* whether to dedup to
   162 or keep all 250 and only fold-group by source (see C.6 risks).
3. Point the scenario scripts at `--bank <fitted.jsonl> --matrix bridgegrade/response_matrix.csv
   --scenarios data/Bridge/scenarios.jsonl`. The engine's `bank_for_model` already filters each
   scenario's `criterion_ids` to graded cells, so ~98% fill is handled natively.

> Note: TutorBench's scenarios each carry ~1 criterion per skill; **Bridge scenarios carry 19–20
> criteria across 5 skills**, so a Bridge scenario is a *much heavier testlet*. This changes test-length
> economics (few scenarios buy a lot of criteria) and strengthens within-scenario local dependence —
> flagged in C.6.

### C.3 Ordered experiment sequence (dependency-ordered)

Each row: **script to reuse/adapt → inputs → outputs**. "adapt" = the existing `scenario_*` script
runs against Bridge by pointing `--bank/--matrix/--scenarios` at Bridge files (its `--scenarios`
default is TutorBench, so pass Bridge's explicitly); "new" = a thin Bridge wrapper.

1. **Ingest / aggregate (build)** — *reuse* `calibrate_mirt.py --write-params` (+ a small
   `build_bridge_fitted_bank` adapter to emit the fitted-bank schema with provenance and apply
   scenario source-dedup). **In:** `response_matrix.csv`, `rubrics.jsonl`, `scenarios.jsonl`.
   **Out:** `bridge_scenario_fitted.jsonl` (+ manifest).
   *Depends on:* nothing. *Blocks:* everything below.

2. **Scenario-level calibration (M2PL fit) + core diagnostics** — *reuse* `calibrate_mirt.py
   --estimate-latent-corr`. **In:** matrix + Q. **Out:** `calibration_mirt.csv`,
   `calibration_mirt_manifest.json` (loglik/AIC/BIC, latent-corr, dropped-item accounting).
   *Depends on:* 1.

3. **Dimensionality (1D vs multi at scenario level)** — *reuse* `calibrate_mirt.py --collapse
   --efa` + *adapt* `analyze_collinearity.py` / `fig_composite_vs_axes.py`. Decide 1 vs 2…5 skills by
   **BIC + held-out log-loss + parsimony** (not AIC), watching latent-corr collapse toward ±1 and
   reporting per-axis `a_k`. **Out:** structure comparison + `selection.json`.
   *Depends on:* 2. *Parallelizable with* nothing before the skill set is fixed (it **decides** the
   skill set that all later steps use).

4. **Operating-point re-derivation (SE-target × min-scenarios floor sweep)** — *adapt*
   `offline_engine_driver.py --min-scenarios {0,12,15,20} --max-se {0.20,0.25,0.30,0.35}` (or a new
   `bridge_scenario_floor_se_grid.py` that loops the engine over the grid) + *reuse*
   `se_sweep_aggregate.py`. **The item-level SE 0.15 / floor 20 is VOID and must be re-swept in
   SCENARIO units.** Pick the smallest floor / loosest SE at ~100% convergence without lengthening
   tests, compared against the scenario `SE_param` floor from step 7. **Out:** `sweep_results.csv`,
   `best.json` (locked scenario operating point). *Depends on:* 3 (skill set) + a first pass of 7's
   `SE_param` floor (can bootstrap: run 7 once at SE 0.30 to get the floor, then finalize 4).

5. **OOS recovery at the new locked point** — *adapt* `scenario_kfold_estimator_cv.py --k 5 --seed
   20260729 --min-scenarios <locked> --max-se <locked>`. Refit `(a,b)` per fold (within-fold
   zero-variance drop), run the real scenario engine on held-out models, score MWLE θ vs the fold's
   full-bank EAP reference. **Out:** `oos_per_model.csv`, `metrics.json` (headline held-out r/slope).
   *Depends on:* 4.

6. **CAT-vs-random efficiency** — *adapt* `offline_engine_driver.py` twice (`--mode cat` and
   `--mode baseline` **uncapped** `--max-scenarios <bank size>`), same stop rule, or a
   `bridge_scenario_cat_vs_random.py` wrapper. **Out:** length + recovery, adaptive-vs-random figure.
   *Depends on:* 4. *Parallelizable with* 5, 7, 8.

7. **Parameter uncertainty (scenario-level observed-info bootstrap)** — *adapt*
   `scenario_param_uncertainty.py` (administered set from the real engine; per-criterion observed-info
   covariance → parametric bootstrap → `SE_param` per skill; `SE_total = √(SE_ability²+SE_param²)`;
   **observed-info bootstrap, not jackknife** — jackknife went degenerate 13/51 on Bridge item-level).
   **Out:** `leaderboard_se_components.csv`, `metrics.json` (feeds the `SE_param` floor into step 4).
   *Depends on:* 3 (skills) + a locked-ish operating point (iterate with 4).

8. **Estimator comparison (EAP vs MWLE vs MLE)** — the estimator columns are already emitted by
   `scenario_kfold_estimator_cv.py` (`theta_online/batch/mwle`); read them off, or extend with MLE.
   Confirm MWLE (slope ≈ 1). **Out:** estimator recovery table. *Depends on:* 5.

9. **Order / seed stability** — *adapt* `scenario_order_experiment.py --n-seeds 8 --min-scenarios
   <locked> --max-se <locked>`. Across-seed θ SD/range vs the SE target. **Out:** `per_model_spread.csv`,
   `metrics.json`. *Depends on:* 4. *Parallelizable with* 5–8.

10. **Ridge / grid sensitivity** — *reuse* `calibrate_mirt.py` across `ridge ∈ {1e-3,1e-2,1e-1}` and
    GH grid, scoring fit stability + OOS r (a `bridge_scenario_sensitivity.py` loop, or reuse
    `kfold_cv_mirt.py --ridge`). **Ridge must be RE-SWEPT** (Playbook §6). *Depends on:* 2/3.

11. **Leaderboard** — final `model_leaderboard.csv` with **`SE_total` bars** from step 7 (never
    ability-only SE). *Depends on:* 4,5,7.

### C.4 Decisions that MUST be re-derived from scratch (item-level values don't transfer)

| Decision | Item-level value (VOID) | Why it doesn't transfer |
|---|---|---|
| **SE target** | 0.15 (per-criterion Fisher SE) | Scenario-level SE lives on a different information scale (a whole 19–20-criterion testlet per step); a per-criterion 0.15 says nothing about per-scenario convergence. Re-sweep 0.20–0.35 (or wider) in engine `max_se` units against the scenario `SE_param` floor. |
| **Min-test-length floor** | floor = **20 criteria** | The scenario floor is a count of **scenarios** (`min_scenarios`), and one Bridge scenario ≈ 19–20 criteria, so "floor 20 criteria" ≈ **1 scenario**. Re-sweep `min_scenarios ∈ {0,12,15,20}` (cap-aware: ~162 scenario bank). |
| **`SE_param` / `SE_total`** | item-level bootstrap | Parameter uncertainty must be propagated through the **scenario** administered set and the chosen skill model; the item-level per-criterion `se_param` is on the wrong administration and (likely) wrong dimensionality. |
| **Dimensionality / skill structure** | unidim 2PL (item study) / Bridge's 5-skill axis | Must be re-decided at scenario level by BIC + held-out log-loss + parsimony + latent-corr collapse + per-axis `a_k`, on the same fit. At N=51 the 5-skill MIRT is unlikely to identify (latent corr collapses toward 1) — expect a collapse to 1–2 skills, but *derive* it. |
| **Ridge** | 1e-2 | Stability depends on N and sparsity; re-sweep {1e-3,1e-2,1e-1} (Playbook §6). |

Reusable-as-is from TutorBench defaults (still confirm): estimator **MWLE**, selection **trace**,
**k=5 / seed 20260729**, `top_n=5`, `min_evals_per_skill=15`, GH grid 7 nodes/dim.

### C.5 Effort / compute per step & parallelism

| Step | Rough effort | Compute | Parallelizable? |
|---|---|---|---|
| 1 build fitted bank | S (adapter) | seconds | — (gate) |
| 2 core fit | S | seconds–1 min | — |
| 3 dimensionality | M (decide skills) | minutes (a few EM fits + EFA) | after 2 |
| 4 floor×SE sweep | **L** (16 engine runs × 51 models) | minutes–low-hours; engine runs parallel across models (`--workers`) | grid cells embarrassingly parallel |
| 5 OOS k-fold | M | minutes (5 folds × engine) | folds parallel; parallel with 6–9 |
| 6 CAT-vs-random | S–M | minutes (2 engine passes) | parallel with 5,7,8,9 |
| 7 param bootstrap | **L** | B×51 EAPs; serial-ish over models, coarse `--eap-grid` for multi-skill | parallel with 5,6,8,9 |
| 8 estimator | S | free (reads step-5 columns) | after 5 |
| 9 order/seed | M | n_seeds × engine | parallel with 5–8 |
| 10 ridge/grid | M | several EM fits | after 2/3 |
| 11 leaderboard | S | seconds | after 4,5,7 |

**Critical path:** 1 → 2 → 3 → (7 preliminary for the SE_param floor) → 4 → 5 → 11. Steps 6, 8, 9,
10 hang off 3/4 and run in parallel. All engine steps parallelize across the 51 models with
`--workers`; the sweep (4) parallelizes across grid cells; k-fold (5) across folds.

### C.6 Risks (small bank, small N)

- **Small scenario bank (~162 after source-dedup, 250 raw).** A CAT can administer at most the bank
  size; with `min_scenarios` floors of 12–20 and heavy 19–20-criterion testlets, a large fraction of
  the bank may be consumed before the SE target is hit, so **the efficiency story ("adaptive saves
  scenarios") may be weak** and some SE targets may be **unachievable at 100% convergence**. The
  sweep must report convergence % honestly and may land on a looser SE target than TutorBench's 0.30.
- **CAT length floor interacts with testlet size.** Because each Bridge scenario grades ~19–20
  criteria, even a *small* `min_scenarios` floor administers many criteria; "floor = a few scenarios"
  already exceeds the item-level "floor 20 criteria." Choose the floor in scenario units and report
  both scenarios and criteria administered (label which regime — deployment vs k-fold length).
- **Dimensionality identifiability at N=51.** A 5-skill (or even multi-skill) M2PL is very unlikely
  to identify at 51 models; expect latent correlations to collapse toward ±1 (the item study already
  saw 0.77→0.97 across 2→5 dims). Name the **correlation collapse**, not "missing anchors," and
  likely **lock 1D (or 2D at most)** — but derive it, and report per-axis `a_k` with spread before any
  (non-)identification claim.
- **Within-scenario local dependence (testlet effect).** Criteria in one scenario are locally
  dependent; administering the whole testlet means ability SE from independent-item Fisher info is
  **optimistic**. Keep `SE_total` (with `SE_param`) as the honest bar, group/hold-out folds on
  `source_id`, and consider noting a testlet caveat on any limited-information fit statistic.
- **`SE_param` dominates at N=51.** As on the item study, calibration error likely dominates ability
  error; use the observed-info **bootstrap** (jackknife was degenerate 13/51) and report `SE_total`.

### C.7 Open questions needing a human decision

1. **Scenario aggregation choice.** Confirm the engine's *administer-the-bundle* model (each
   scenario = a testlet of its 19–20 criteria, per-criterion `(a,b,q)` updates) is the intended
   representation — **not** collapsing each scenario into a single polytomous/aggregated score. The
   TutorBench pipeline uses the bundle model; adopting it keeps the whole toolchain reusable. If a
   per-scenario *score* (e.g. sum/mean pass) is instead wanted, that is a different (and heavier)
   build and departs from the TutorBench template.
2. **Dedup vs keep-all.** Dedup scenarios to **162** (one per `source_id`) for the administered bank,
   or keep all **250** and only *fold-group* by source? Dedup shrinks an already-small bank; keep-all
   preserves length but carries duplicated stimuli into administration.
3. **Skill structure.** Attempt a multi-skill (2–5) scenario MIRT, or lock **unidimensional** given
   N=51? Recommendation: derive via step 3 and most likely lock 1D, but the user may want the
   multi-skill attempt documented as exploratory (as the item study did).
4. **SE target range.** If no SE target reaches ~100% convergence on the small bank, is a looser
   target (e.g. 0.35–0.40) acceptable, or should the floor/`max_scenarios` policy change instead?
5. **A3 safety tripwire & `extreme_a` items.** The item study excluded A3 (constant safety criterion)
   and `extreme_a` items from CAT use; confirm the same exclusions carry into the scenario bank.
