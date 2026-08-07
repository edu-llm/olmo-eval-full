> ⚠️ **OUTDATED / SUPERSEDED** — the canonical TutorBench scale is now the correctness-only unidim package in `../unidim/`. This 2-skill (correctness+scaffolding) calibration is retained for reference and possible revisit after more models are graded. Do not cite these as current.

# TutorBench calibration — SCENARIO-LEVEL 2-skill study (EAP-posterior stop, of-record)

Scenario-level MIRT recalibration of the TutorBench benchmark, mirroring the sibling
scenario-level packages (`bridge_calibration/`, `wildbench_calibration/`, `biggen_calibration/`).
A scenario is administered as a **testlet bundle** of its per-criterion M2PL items; the real
production engine (`tutor_cat.engine.run_evaluation` via `scripts/scenario_cat_lib.py`) selects a
whole scenario per CAT step and updates ability from all its criteria. Nothing here reimplements
CAT selection or the M2PL update.

**Status: STUDY / reporting ONLY. Production engine (`tutor_cat/`, `scenario_cat_lib.py`)
untouched. Nothing committed.** N=115 models; 2 latent skills (correctness, scaffolding).

---

## ⭐ WHERE THE UP-TO-DATE FIGURES LIVE (read this first)

**The canonical TutorBench results are the 2-skill EAP-posterior-stop package at floor(min_scenarios)=20 / SE_ability=0.27.**
They live under `tutorbench_calibration/experiments/`:

| What | Canonical location |
|---|---|
| Headline OOS recovery (scatters, per-model) | `experiments/05_oos_recovery/` |
| Operating-point selection grid (OOS floor×SE) | `experiments/06_floor_se_grid/` |
| Operating-point landscape (in-sample heatmaps) | `experiments/06b_operating_point/` |
| Deployed leaderboard @ 20/0.27 (+ top-level `model_leaderboard.csv`) | `experiments/08_leaderboard/` |
| SE_param precision floor | `experiments/07_parameter_uncertainty/` |

**`regenerated_figures/scenario_level_115_min12/` is LEGACY (engine online-SE stop) and is
SUPERSEDED for every STOP-DEPENDENT result** (recovery, leaderboard, efficiency, p-IRT MAE,
achieved SE, operating point). Its 2-skill outputs are archived here under
`experiments/archive_onlineSE_legacy/` and must NOT be quoted as current. Two stop-dependent slots
(**04 efficiency**, **09 p-IRT MAE**) are not yet regenerated at 20/0.27 and are marked **TODO** —
their slots contain a regeneration recipe, not a stale figure.

Superseded originals the user may prune at commit:
`reports/eap_oos_grid_tutorbench_2skill/` and `reports/eap_stop_grid_tutorbench_2skill/` (now
copied into this package) and `regenerated_figures/scenario_level_115_min12/` (2-skill parts
archived here; the rest is legacy).

---

## Locked of-record configuration (2-skill is canonical)

- **Skills:** correctness + scaffolding (content+diagnosis → correctness; presentation not
  modeled). 3-skill is a SECONDARY/exploratory track (see `experiments/03_structures/`), NOT shipped.
- **CAT stop:** EAP-posterior honest per-skill **MARGINAL SD** (marginal of the joint 2-D
  posterior), dense grid **161 nodes/dim** (25,921 joint), info-plateau **δ=0.005 / W=3**, **cap 70**,
  **MWLE θ** at stop. Priority: precision → info_plateau → cap → bank_exhausted.
- **Operating point (LOCKED): floor(min_scenarios)=20, SE_ability target=0.27** — SELECTED
  out-of-sample on the floor×SE grid over all 115 models.
- **Fit:** `calibrate_mirt.fit_m2pl_em`, fit_grid=7, ridge=1e-2, negative_policy=clamp.
- **OOS:** k=5 model-fold, seed 20260729, refit-per-fold.
- **N=115 models; headline EXCLUDES 2 weakly-calibrated models:** `Qwen/Qwen1.5-1.8B`,
  `BSC-LT/salamandra-7b-instruct` (parameter-limited; land at the low-θ tail; reported separately).

### Headline OOS numbers @ 20/0.27 (N=113 excl-2 — must match `06_floor_se_grid/oos_per_cell_grid.csv` row 20/0.27)

| skill | r | slope | θ-MAE | median SD | median SE_total | %reach SE_ability≤0.27 |
|---|---|---|---|---|---|---|
| correctness | **0.967** | **1.019** | **0.437** | 0.280 | **0.371** | 43.4% |
| scaffolding | **0.932** | **0.953** | **0.256** | 0.265 | **0.320** | 69.0% |

- Median length **25** scenarios (mean 26.54). Both-skills SE_ability reach **34%**. Stop reasons:
  info_plateau **66%**, cap **0%**.
- The modest 34% both-skills reach is an **SE_param precision-ceiling artifact** (SE_param ~0.21
  correctness), NOT a stop-rule defect: OOS ability recovery is strong (r≈0.97, slope≈1.0).

## Experiment index

| slot | description | status | figures / path |
|---|---|---|---|
| `03_structures` | dimensionality: 1 vs 2 vs 3-skill (2-skill canonical) | **carried-over (stop-independent)** | `experiments/03_structures/` |
| `04_efficiency_vs_random` | adaptive vs random efficiency / SE-vs-length | **TODO (regen @20/0.27)** | `experiments/04_efficiency_vs_random/README.md` (legacy in archive) |
| `05_oos_recovery` | headline OOS recovery scatters (corr+scaff) | **EAP-regenerated @20/0.27** | `experiments/05_oos_recovery/figures/` |
| `06_floor_se_grid` | OOS floor×SE grid (op-point selection basis) | **EAP-regenerated @grid** | `experiments/06_floor_se_grid/figures/oos_grid_heatmaps.png` |
| `06b_operating_point` | in-sample op-point landscape / heatmaps | **EAP-regenerated @grid** | `experiments/06b_operating_point/*.png` |
| `07_parameter_uncertainty` | SE_param precision floor (fixed offset) | **carried-over (stop-independent)** | `experiments/07_parameter_uncertainty/` |
| `08_leaderboard` | deployed leaderboard @20/0.27 + SE_total bars | **EAP-regenerated @20/0.27** | `experiments/08_leaderboard/figures/` |
| `09_pirt_mae` | p-IRT / pass-rate MAE | **TODO (regen @20/0.27)** | `experiments/09_pirt_mae/README.md` (legacy in archive) |
| `10_estimator_comparison` | online vs batch-EAP vs MWLE (MWLE locked) | **carried-over (stop-independent-ish)** | `experiments/10_estimator_comparison/` |
| `11_order_seed` | order / seed stability (fixed-seed policy) | **carried-over (stop-independent)** | `experiments/11_order_seed/figures/` |
| `12_ridge_grid_sensitivity` | ridge locked at 1e-2 (documented) | **carried-over / documented** | `experiments/12_ridge_grid_sensitivity/README.md` |
| `archive_onlineSE_legacy` | legacy online-SE stop outputs | **legacy-superseded** | `experiments/archive_onlineSE_legacy/` |

**Stop-dependent** slots (must reflect EAP 20/0.27): 04, 05, 06, 06b, 08, 09.
**Stop-independent** slots (carried over from the of-record 2-skill work): 03, 07, 10, 11, 12.

## Layout

```
tutorbench_calibration/
  README.md                    this file
  FLOW_PACKAGE.md              graduation manifest -> flow/mirt-frq
  fit_manifest.json            fit + locked-config + headline provenance
  model_leaderboard.csv        of-record deployed leaderboard @ 20/0.27 (top-level copy)
  experiments/
    03_structures/             dimensionality 1 vs 2 vs 3-skill        [carried-over]
    04_efficiency_vs_random/   adaptive vs random                      [TODO @20/0.27]
    05_oos_recovery/           headline recovery @ 20/0.27             [EAP of-record]
    06_floor_se_grid/          OOS floor x SE grid (selection basis)   [EAP of-record]
    06b_operating_point/       in-sample op-point landscape/heatmaps   [EAP of-record]
    07_parameter_uncertainty/  SE_param precision floor                [carried-over]
    08_leaderboard/            deployed leaderboard @ 20/0.27          [EAP of-record]
    09_pirt_mae/               p-IRT / pass-rate MAE                    [TODO @20/0.27]
    10_estimator_comparison/   online vs batch vs MWLE (MWLE locked)   [carried-over]
    11_order_seed/             order/seed stability + fixed-seed       [carried-over]
    12_ridge_grid_sensitivity/ ridge locked 1e-2 (documented)          [carried-over/doc]
    archive_onlineSE_legacy/   legacy online-SE 2-skill outputs        [SUPERSEDED]
  scripts/                     driving scripts (pointer copies; run from repo root)
```

## Driving scripts (`scripts/`, run from `eduLLM-Evals/` with `uv run --no-sync python -u`)

- `eap_oos_grid_study.py --workers 6` — OOS floor×SE grid (exp 06); ~45 min. Pin BLAS to 1
  thread/worker and use `--workers 6` (the machine OOMs at the default worker count).
- `of_record_f20se27.py` — of-record recovery + leaderboard @ 20/0.27 (exp 05, 08) from the grid.
- `eap_stop_grid_study.py --workers 6` + `plot_eap_stop_grid.py` / `plot_eap_stop_grid_recovery.py`
  — in-sample op-point landscape (exp 06b).
- `scenario_kfold_estimator_cv.py` (exp 03, 10), `scenario_param_uncertainty.py` (exp 07),
  `scenario_order_experiment.py` (exp 11), `scenario_selection_experiment.py` (selection/D-opt).
- `scenario_cat_lib.py` — shared library (drives the UNMODIFIED production engine). `eap_stop_prototype.py`
  — original teammate prototype (superseded by the grid studies; kept for provenance).

## Of-record bank (graduates to flow/mirt-frq)

`data/TutorBench/rubrics_qmatrix_calibrated_2skill_115_fitted.jsonl` (3,666 fitted items;
ridge 1e-2, grid 7, clamp; matrix sha256 `152286b6...cccc0eb`). See `FLOW_PACKAGE.md`.
