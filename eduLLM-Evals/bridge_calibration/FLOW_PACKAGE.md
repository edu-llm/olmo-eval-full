# Bridge -> flow/uni-frq graduation package (INTERIM)

Interim, self-contained payload for porting Bridge's canonical **scenario-level 1D**
calibration into the FRQ eval flow. **Layout is provisional**: conform to the FRQ diagnostic
loader (`diagnostics/frq_cat/common/irt_params.py` on `CheckpointFlows`, mirroring
`diagnostics/mcq_cat/`) before committing into `flow/uni-frq`.

## Status

- Canonical per owner: **scenario-level 1D (`overall`)**. Supersedes the earlier
  item/criterion-level study (recoverable at git `7a394af`). The 5D fit is **exploratory only**
  (`bridge_scenario_fitted_5d.jsonl`) and does NOT graduate.
- **PROVISIONAL - N=51 models** (multi-dim non-identifiable at this N; latent corr collapses
  toward 1.0, which is why 1D was selected by BIC + parsimony).
- Snapshot tag: `bridge-scenario-1d-51models`.

## Target

`flow/uni-frq` - unidimensional, single latent axis `overall`. **Testlet administration**: a
scenario is administered as a bundle of its per-criterion items; the CAT engine selects a whole
scenario per step and updates ability from all its criteria (production `tutor_cat.engine`
via `scripts/scenario_cat_lib.py`; no polytomous collapse).

## Locked config

**Stop rule: EAP-posterior SD** (integrate the posterior over the fine 321-node theta grid;
stop when posterior SD <= target AND floor met, else cap). `min_scenarios = 12`,
`SE-ability target = 0.12`, ridge 1e-2, grid 7 (1D), estimator **MWLE** (plain MLE diverges on
all-pass/all-fail testlets). Deployment single-run CAT uses a **fixed production seed** so
first-item/tie-break choices are reproducible across models; the honest measurement SE is still
reported. Recovery CV = OOS model/person folds, k=5, seed 20260729.

The SE target moved **0.15 -> 0.12** at adoption: SE_total adds SE_param (~0.05), so a 0.15
ability target left 14 models with SE_total > 0.15; 0.12 collapses that tail to 2 (see README
"Operating point"). The prior online-SE @0.15 of-record is archived at
`experiments/archive_onlineSE/`. Bank unchanged (no re-fit).

## Payload

| Artifact | Path | Role | Tracked |
|---|---|---|---|
| CAT administration/scoring pool (canonical runtime) | `bridge_calibration/bridge_scenario_fitted_1d_catpool.jsonl` | 3,961 items across 228 scenarios (A3 safety-gate + extreme_a removed); what the engine selects/scores from | yes |
| Full 1D calibration bank | `bridge_calibration/bridge_scenario_fitted_1d.jsonl` | 4,197 calibrated items (superset; reporting/leaderboard) | yes |
| CAT-pool exclusion mask | `bridge_calibration/exclusion_mask.json` | defines catpool = full bank minus A3 + extreme_a | yes |
| Items / questions | `data/Bridge/scenarios.jsonl` | 250 scenarios the tutor model responds to | yes |
| Judge config | `judge_frozen.yaml` | frozen judge for grading responses | yes |
| Fit / build manifests | `bridge_calibration/{fit_manifest.json,build_manifest.json}` | provenance, dimensionality, source->fold grouping | yes |
| Response matrix (calibration input) | `eduLLM-Evals/bridgegrade/response_matrix.csv` | reproducibility input only; NOT needed at flow runtime | no (local -> S3) |

**Bank schema note:** `discrimination` is keyed by the modeled skill name `overall` (single
axis) + scalar `difficulty`; `q_modeled = {"overall": 1}`; `irt_params.structure = "overall_1d"`.
Records are per-criterion but administration/scoring is per-scenario testlet.

## Runtime data flow (scoring a new checkpoint)

1. Generate tutor responses from the checkpoint on `data/Bridge/scenarios.jsonl`.
2. Grade per criterion with `judge_frozen.yaml`.
3. Run the testlet CAT (whole scenario per step) via `tutor_cat.engine` / `scenario_cat_lib`
   against the **catpool** at the locked config to estimate `overall` ability with SE.

## Provenance + metrics

- Fit: `calibrated-m2pl-scenario` confirmatory-m2pl-mml-em, `overall_1d`; `n_models = 51`;
  250 scenarios / 162 unique sources; 4,795 criteria -> 4,197 after zero-variance filtering;
  ridge 0.01; calibrated 2026-08-05.
- Matrix sha256: `db27d62c42834abae4d5733bfd8aeb45dddda18157cab40393a448c5a61f623e`.
- Recovery (exp 05, OOS locked, EAP stop @ 12/0.12, MWLE): r = **0.958** [0.929, 0.977],
  slope 0.883, theta-MAE 0.273; p-IRT pass r = 0.936, pass-MAE 0.047; OOS mean length
  14.4 scenarios / 252 criteria (deployed full-data ~17 scenarios).
- Deployed SE (exp 07, EAP admin @ 12/0.12): 96.1% of models reach SE-ability <= 0.12;
  SE_total mean 0.117 / max 0.195; #(SE_total > 0.15) = 2, #(> 0.20) = 0. 2 weakly-identified
  EAP-native caps (Qwen/Qwen1.5-1.8B, allenai/OLMo-1B-hf).
- Estimator (exp 10): MWLE r 0.958 / slope 0.883 (recommended). Ridge stable across
  {1e-3,1e-2,1e-1} (min corr 0.988) - keep 1e-2.
- **Caveats:** N=51 is small (multi-dim non-identifiable -> 1D shipped; 5D exploratory only).
  22 all-zero-variance scenarios are un-administrable. Full details in `README.md` (exp 03-12).

## Rerun / swap protocol (larger cohort)

1. Recalibrate at the larger cohort; re-emit `bridge_scenario_fitted_1d.jsonl` + `_1d_catpool.jsonl`
   (+ `exclusion_mask.json`).
2. Replace them here; bump `n_models` + `matrix_sha256` in `fit_manifest.json` / bank provenance.
3. Re-tag `bridge-scenario-1d-<N>models`; re-port to `flow/uni-frq`.
4. Items / judge config unchanged unless criteria are revised.
