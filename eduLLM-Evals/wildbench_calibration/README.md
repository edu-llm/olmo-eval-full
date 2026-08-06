# WildBench calibration — SCENARIO-LEVEL study

Scenario-level IRT recalibration of the WildBench benchmark, mirroring the scenario-level
Bridge study (`bridge_calibration/`) and the BiGGen p-IRT parity. A scenario is administered
as a **testlet bundle** of its per-criterion M2PL items (grouped by scenario); the real
production engine (`tutor_cat.engine.run_evaluation` via `scripts/scenario_cat_lib.py`)
selects a whole scenario per CAT step and updates ability from all its criteria. Nothing here
reimplements CAT selection or the M2PL update.

Inputs: 52 models × 11,416 criteria (`WildBenchGrade/response_matrix.csv`, 99.99% filled);
1,001 scenarios (`data/WildBench/scenarios.jsonl`, ~11.4 criteria/scenario); each criterion
is **one-hot** mapped to one of **11 task-category skills** (advice_seeking, brainstorming,
coding_debugging, creative_writing, data_analysis, editing, information_seeking, math,
planning, reasoning, role_playing). The rubrics' synthetic placeholder IRT params were
IGNORED and fit fresh.

## Design (locked)

- **Dimensionality: 1-D** (unidimensional "general ability"). The 11 task categories are
  reporting labels, not separable latent abilities at N=52 — see below.
- **CAT stop rule: EAP-POSTERIOR SE (of-record).** Stop when the EAP posterior SD over the
  administered items (on the fine θ grid) ≤ 0.12 AND `min_scenarios ≥ 8`, else cap. This
  replaced the earlier online normal-approx SE stop (archived under
  `experiments/archive_onlineSE/`; still available via `--stop-rule online`). The posterior SD
  is the honest measurement SE and resolves the online-vs-posterior estimator mismatch (adopted
  after the exp-13 prototype: recovery r 0.957→0.975, precision-reached 65%→~87%, order/seed
  variance halved). The adaptive scenario *selection* and the production engine are UNCHANGED;
  only the stop rule differs (applied on a forced-long adaptive order).
- **Operating point: `min_scenarios = 8`, SE target `0.12`** (re-confirmed under the EAP stop;
  r = 0.975 at floor 8).
- **Estimator: MWLE** (multidimensional Warm's weighted-likelihood) for the deployment
  instrument (consistent with Bridge/BiGGen).
- **Calibration: ridge `1e-2`, production GH fit grid `7`** (shared-toolchain default).
- **CAT-pool exclusion: extreme-|a| items only** (|a| > 6 in the fit) — WildBench has no
  affective safety gate (Bridge's A3). At the locked ridge/grid this is **only 13 items**.
- **Reference θ: fine uniform EAP grid** (321 nodes over ±8, continuous — no coarse GH
  quantization) everywhere recovery is scored.
- **Recovery CV: OOS model/person folds, k=5, seed 20260729.**
- **Deployment scoring: FIXED production seed** (see order/seed below).

## Bank build (`build_manifest.json`)

- 1,001 scenarios / **1,001 unique sources** (each scenario is its own source — source→fold
  grouping is a no-op, recorded for parity) / 52 models, matrix 99.99% filled.
- 11,416 criteria → **8,358** after zero-variance filtering (3,058 all-fail dropped, 0
  all-pass); ~11.4 criteria/scenario, **985 administrable scenarios** (16 all-ZV).

## Dimensionality — the "collapse the 11" decision (`experiments/03_structures/`)

There was no pre-existing WildBench collapse doc; decided empirically. Confirmatory M2PL at
1-D, semantic 2/3/4-D super-skill groupings, an EFA-derived grouping, and the full 11-skill
model (fit as the block-diagonal independent model — an 11-dim product GH grid is
infeasible, and one-hot loadings make the identity-R model factorise; the 11×11 inter-skill
correlation is recovered via a plug-in on per-skill EAP abilities).

| structure | dims | AIC | BIC | OOS log-loss (±SE) | max\|latent r\| |
|---|---|---|---|---|---|
| **overall_1d** | **1** | 270,661 | **454,239** | 0.48139 ± 0.08013 | 0.000 |
| gen_vs_rest | 2 | 279,652 | 463,241 | 0.49032 ± 0.08021 | 0.957 |
| efa_2d (math vs rest) | 2 | 278,720 | 462,309 | 0.49009 ± 0.07954 | 0.906 |
| gen/tech/info | 3 | 278,705 | 462,316 | 0.49094 ± 0.07945 | 0.974 |
| gen/tech/info/edit | 4 | 278,778 | 462,422 | 0.49074 ± 0.07976 | 0.986 |
| full_11d (independent) | 11 | 267,994 | 451,572 | 0.47756 ± 0.07937 | 0.984 |

- **11-skill latent-ability correlation**: max off-diag **0.984**, mean **0.849** (`math` is
  the least-correlated skill, ~0.87). **Scree**: first eigenvalue = **94%** of skill-ability
  variance; **only 1 Kaiser factor** (≥1). Every candidate grouping's super-skills correlate
  ~0.91–0.99.
- **Selection** (lowest held-out log-loss within 1 SE → BIC + parsimony): **all structures
  are within 1 SE on OOS log-loss**, so parsimony selects **1-D**.
- **AIC/BIC caveat** (`selection.json → aic_bic_caveat`): full_11d is nominally AIC/BIC-best,
  but that is an artifact of the identity-R block-diagonal parameterization (11 free
  per-person abilities at the same parameter count as 1-D) plus the coarse product grid
  handicapping the 2–4d models. The grid-fair arbiter — OOS log-loss — separates nothing, and
  with r≈0.98 + 94% first-factor variance the well-specified conclusion is **unidimensional
  collapse**. **Recommendation: collapse all 11 → one general-ability axis.**

## Ridge / grid sensitivity (`experiments/12_ridge_grid_sensitivity/`) — ridge 1e-2 kept

Swept ridge {1e-3, 1e-2, 1e-1} × grid {7, 21}. **θ rank is highly stable** (min corr vs the
production baseline = 0.990). The **extreme_a set is strongly grid-sensitive** — 13 (grid 7)
vs 378 (grid 21) at ridge 1e-2 — so the production fit grid is **fixed at 7** (the
shared-toolchain default) and documented. Ridge 1e-2 is within noise of the best OOS
(ridge 0.1 gives r 0.997 vs 0.996 but drains extreme_a to 0); **kept ridge 1e-2** (Bridge
convention). Finalized extreme_a = **13**; `exclusion_mask.json` + the catpool bank were
re-derived at (ridge 1e-2, grid 7). exp-05/07/10 were **re-run** at these params.

## CAT-pool (`exclusion_mask.json`, `wildbench_scenario_fitted_1d_catpool.jsonl`)

After ZV + extreme_a: **8,345 administrable criteria across 985 scenarios** (only 13
excluded). Excluded from administration + θ scoring, kept in calibration + reporting.

## Headline recovery (`experiments/05_oos_recovery/`) — locked 8/0.12, MWLE, EAP stop

- **r = 0.975** [0.957, 0.986], slope **0.921**, θ-MAE **0.357**; mean length **13.2
  scenarios / 129 criteria**.
- p-IRT pass calibration: **pass-r = 0.951**, pass-MAE **0.047** (see also exp-09).
- Reference θ = fine uniform EAP (321 nodes over ±8).
- (Under the earlier online-SE stop, archived: r 0.957 / slope 0.890 / θ-MAE 0.428 at ~9
  scenarios. The EAP stop trades ~+4 scenarios for materially better recovery + honest SE.)

## Estimator comparison (`experiments/10_estimator_comparison/`) — MWLE locked

| estimator | r | slope | θ-MAE |
|---|---|---|---|
| batch-EAP | 0.975 | 0.883 | — |
| **MWLE** | **0.975** | **0.921** | **0.357** |
| MLE | 0.975 | 0.927 | — |

(Under the EAP stop the estimators converge — the longer administration pins θ well.) MLE's
slope is marginally closest to 1 but **MWLE is locked** as the deployment instrument (Warm's
penalty stays finite where plain MLE diverges on all-pass/all-fail administrations),
consistent with Bridge & BiGGen.

## Efficiency (`experiments/04_efficiency_vs_random/`) — adaptive vs random

- Locked L=8: adaptive **r = 0.940** vs random **r = 0.898** (gap **+0.042**); mean r gap
  over all lengths **+0.050**. Random needs **12** scenarios to match adaptive's L=8 recovery.
- **SE-vs-length panel** (now the **EAP posterior SD**, the of-record SE metric): adaptive's
  posterior SD falls steadily with length while random's stays high; random needs many more
  scenarios to reach the 0.12 posterior-SD target. Adaptive selection is what makes the
  SE-target stop rule viable. (Recovery curves are fixed-length and identical under either stop
  rule; only the SE metric on the panel changed from online SE to EAP posterior SD.)
- (One minor r reversal at L=3 where random edges adaptive by ~0.008 — noise.)

## Order / seed stability (`experiments/11_order_seed/`) — fixed production seed

- Across 8 seeds at the locked point (EAP stop): mean θ SD = **0.134** (median 0.099, max
  0.61), i.e. **~1.1× the SE target** — down from **0.292** under the online-SE stop. The
  EAP stop administers to a real posterior-SD target, so different seeds converge to similar
  precision; order/seed variance drops to ~the measurement SE (now in line with Bridge ~0.11).
- **The variance is concentrated in the near-all-fail bottom models**, NOT typical mid-range
  models (`experiments/11_order_seed/order_seed_diagnostics.{json,csv}`):
  corr(θ_SD, |θ|) = **+0.836**, corr(θ_SD, %-informative-items) = **−0.721**. A typical model
  has ~44% of CAT-pool items informative at its θ (median); the unstable tail has 1–11%. The
  four largest-SD models are all near-all-fail (pass ≤ 0.8%, θ ≤ −4.6, < 3% informative) —
  their θ is only *bounded below*, so no adaptive order or test length can point-identify it.
- **Decision (locked): accept + document.** Use the **FIXED production seed** policy
  (consistent with Bridge/BiGGen) so single-run CAT scoring is reproducible, and **flag the
  bottom near-degenerate models as `weakly_identified`** (θ reported as an upper bound; rule in
  the leaderboard section) rather than implying precise θ. Seed-averaging (K× cost) and global
  test-lengthening were both rejected: they would not point-identify a model that fails almost
  everything, and would penalize the stable ~90% of models. The large SE_total bars on these
  models already carry the honest uncertainty.

## SE_param regime (`experiments/07_parameter_uncertainty/`)

Observed-information parametric bootstrap (B=200, NOT jackknife) over the full administrable
bank. **SE_param floor: mean 0.039, median 0.0385** (max 0.104); SE_posterior median 0.031;
**SE_total median 0.046**. `SE_total = √(SE_ability² + SE_param²)`. Comparable to Bridge
scenario-level (~0.027–0.033) and ~5× smaller than TutorBench (~0.21); well below the 0.12
SE target, so calibration noise is not the binding constraint. Deployed-CAT SE bars
(ability-only vs total, mean ± SD across models) are in `figures/se_ability_vs_total.png`
(WildBench analog of the BiGGen chart): mean SE_ability **0.137** / SE_total **0.164**
(inflated by the low-θ tail; medians **0.111 / 0.124**).

**Deployed precision** (`experiments/07_parameter_uncertainty/precision_reached.{json,csv}`,
EAP stop). **45/52 (86.5%) reach SE_ability ≤ 0.12** (median SE_ability **0.108** ≈ target,
IQR [0.082, 0.117]; median SE_total **0.128**, IQR [0.106, 0.134]) — up from **34/52 (65%)**
under the archived online-SE stop, because the EAP stop administers to the real posterior-SD
target. The 7 that still miss: the **4 `weakly_identified`** hit-cap models (OLMo-1B-hf,
smol_llama-220M-GQA, mGPT, smol_llama-220M-openhermes; SE_ability 0.22–0.31 even at the L=40
cap) + 3 narrow misses (gemma-2-2b, Qwen2-0.5B, Yi-6B-200K at ~0.122). Standard
variable-length-with-cap CAT behavior: the bank lacks informative items at very low θ, not a
calibration defect. Deployed SE bars (mean ± SD) in `figures/se_ability_vs_total.png`: mean
SE_ability **0.109** / SE_total **0.138** (medians 0.108 / 0.128).

## Leaderboard (`experiments/08_leaderboard/`, `model_leaderboard.csv`)

**Full administrable-bank** scoring (all 985 scenarios / 8,345 CAT-pool criteria per model),
NOT the locked CAT. 1-D fine-EAP θ with SE_total bars.

- θ range **[-6.75, 2.70]**; SE_total range **[0.000, 0.357]** (the large SE_total tail is
  the near-all-fail tiny models where SE_param inflates).
- **Top**: LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct (2.70), ibm-granite/granite-3.1-2b-instruct
  (2.40), meta-llama/Llama-3.2-3B-Instruct (2.29), tiiuae/Falcon3-3B-Instruct (2.20),
  01-ai/Yi-1.5-6B-Chat (2.10).
- **Bottom**: allenai/OLMo-1B-hf (-6.75), ai-forever/mGPT (-6.22),
  BEE-spoke-data/smol_llama-220M-GQA-fineweb_edu (-5.32),
  BEE-spoke-data/smol_llama-220M-openhermes (-4.64), openbmb/MiniCPM-1B-sft-bf16 (-2.89).
- **`weakly_identified` flag** (column in `model_leaderboard.csv`; rendered as an upper-bound
  arrow "θ ≲ value" in `leaderboard_general.png`): **4 models** — OLMo-1B-hf, mGPT,
  smol_llama-220M-GQA-fineweb_edu, smol_llama-220M-openhermes (leaderboard ranks 49–52). These
  are near-all-fail / very-low-information models whose ability is only an **upper bound**, not
  a point estimate. They keep their (large) SE_total bars but should not be read as precise θ.
  **Rule (EAP-of-record)**: `weakly_identified = deployed EAP hit_cap` — the model cannot reach
  SE_ability ≤ 0.12 even at the forced L=40 cap (< ~2–3% of CAT-pool items are informative at
  its θ). Derivation: `experiments/11_order_seed/order_seed_diagnostics.{json,csv}` (which also
  records the structural `frac_informative` / order-seed-SD context).

## p-IRT MAE (`experiments/09_pirt_mae/`) — BiGGen/Bridge parity

Standalone predicted-vs-actual pass-rate MAE from the exp-05 OOS data: **pass-r = 0.937**
[0.887, 0.970], **pass-MAE = 0.051** [0.039, 0.064], slope 0.808, θ-MAE 0.428 (n=52).

## Differences from Bridge / BiGGen

- **Skill structure**: WildBench has **11 one-hot task categories collapsed to 1-D** (Bridge:
  5 tutoring skills → 1-D; both collapse, but WildBench starts from a much wider one-hot axis
  whose abilities are ~0.85–0.98 correlated).
- **Testlet weight → length control**: WildBench scenarios are **light (~11 criteria)** vs
  Bridge's heavy ~18-criterion testlets. Consequently the **SE target (not a scenario floor)
  controls test length** — in the Phase-2a op-point grid, efficiency favored a **low-floor +
  tight-SE** point (r≥0.95 at ~10.6 scenarios at floor 0), the **opposite of Bridge** (where
  the heavy testlet made the floor bind). The user nonetheless locked a modest floor (8) for a
  minimum-length guard.
- **Order/seed variance is larger** (~0.29 vs Bridge ~0.11): a short test over a 985-scenario
  bank is more sensitive to the seeded max-info scenario choice — making the fixed-seed policy
  more important here.
- **Extreme_a is tiny and grid-sensitive** (13 at grid 7 vs 378 at grid 21) — WildBench needs
  the production grid fixed to avoid an inflated exclusion set.
- **Stop rule**: WildBench's of-record CAT stop is the **EAP posterior SD** (honest measurement
  SE). **Bridge and BiGGen still use the engine's online normal-approx SE stop** pending their
  own migration; the online-SE variant remains available here (`--stop-rule online`, archived
  under `experiments/archive_onlineSE/`) for cross-benchmark comparison.

## Resolved decisions / notes

- **CAT stop rule — ADOPTED: EAP posterior SE** (of-record), replacing the online normal-approx
  SE (archived under `experiments/archive_onlineSE/`). Timing gate: the EAP posterior-SD compute
  is negligible per step; the of-record re-run was ~Phase-2b scale. Net effect: recovery r
  0.957→0.975, deployed precision-reached 65%→87%, order/seed θ-SD 0.29→0.13 — at ~+4 scenarios
  mean length. The exp-13 prototype that motivated this is now of-record.
- **Order/seed variance — RESOLVED.** Under the EAP stop it drops to mean 0.13 (~1.1× target),
  in line with Bridge. Fixed production seed retained; the 4 hit-cap models are flagged
  `weakly_identified` (θ as an upper bound). Seed-averaging / global lengthening not needed.
- **Op-point — re-confirmed at 8/0.12 under the EAP stop** (`experiments/06_floor_se_grid/`):
  r **0.975** at floor 8 / SE-target 0.12 (~13.8 scenarios / 136 criteria). The auto-recommended
  "shortest r≥0.95" cell is at a looser SE target (0.25); we keep 8/0.12 for its tight 0.12
  posterior-SD precision + minimum-length guard.

## Layout

```
wildbench_calibration/
  build_manifest.json                         bank build + source->fold grouping
  fit_manifest.json                           core 1D fit (ridge 1e-2, grid 7)
  exclusion_mask.json                         extreme_a CAT-pool mask (13 items)
  wildbench_scenario_fitted_1d.jsonl          1D fitted bank (all 8,358)
  wildbench_scenario_fitted_1d_catpool.jsonl  1D CAT pool (8,345; extreme_a removed)
  model_leaderboard.csv                       full-bank leaderboard (top-level copy)
  experiments/
    03_structures/            dimensionality 1D vs 11-skill vs groupings + EFA/scree
    04_efficiency_vs_random/  CAT vs random (+ SE-vs-length panel)
    05_oos_recovery/          headline recovery at locked 8/0.12 (MWLE)
    06_floor_se_grid/         Phase-2a recovery x (floor x SE) grid
    06b_operating_point/      Phase-2a heatmaps
    07_parameter_uncertainty/ SE_param bootstrap floor + SE components
    08_leaderboard/           full-bank leaderboard + SE_total bars
    09_pirt_mae/              standalone p-IRT pass-rate MAE (BiGGen parity)
    10_estimator_comparison/  EAP vs MWLE vs MLE (MWLE locked)
    11_order_seed/            order/seed stability + fixed-seed policy + weakly_identified
    12_ridge_grid_sensitivity/ ridge/grid robustness + extreme_a re-derivation
    13_eap_stop_prototype/    EAP-vs-online stop prototype (now of-record; kept for provenance)
    archive_onlineSE/         archived online-SE-stop of-record outputs (superseded)
  scripts/                    wildbench_scenario_lib (+ EAP stop helpers) + per-experiment drivers
                              (all stop-dependent drivers take --stop-rule {eap,online}; eap = of-record)
```

## Reproduce

```
# BLAS pinned; run from repo root with: uv run --no-sync python -u <script>
python wildbench_calibration/scripts/build_scenario_bank.py --grid-1d 7
python wildbench_calibration/scripts/scenario_dimensionality.py        # exp 03
python wildbench_calibration/scripts/scenario_ridge_sensitivity.py --grids 7,21  # exp 12
python wildbench_calibration/scripts/build_catpool.py                  # extreme_a mask
python wildbench_calibration/scripts/scenario_param_uncertainty.py     # exp 07
python wildbench_calibration/scripts/scenario_recovery_final.py --fit-grid 7  # exp 05 + 10
python wildbench_calibration/scripts/scenario_efficiency.py            # exp 04
python wildbench_calibration/scripts/scenario_order_seed.py            # exp 11
python wildbench_calibration/scripts/scenario_leaderboard.py           # exp 08
python wildbench_calibration/scripts/scenario_pirt_mae.py              # exp 09
python wildbench_calibration/scripts/make_exp03_figures.py             # exp 03 figures
```
