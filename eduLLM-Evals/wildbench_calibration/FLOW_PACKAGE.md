# WildBench -> flow/uni-frq graduation package (INTERIM)

Interim, self-contained payload for porting WildBench's canonical **scenario-level 1D**
calibration into the FRQ eval flow. **Layout is provisional**: conform to the FRQ diagnostic
loader (`diagnostics/frq_cat/common/irt_params.py` on `CheckpointFlows`, mirroring
`diagnostics/mcq_cat/`) before committing into `flow/uni-frq`.

## Status

- Canonical: **scenario-level 1D (`overall`)** — 11 WildBench task categories collapsed to one
  general-ability axis (94% first-factor variance; skill abilities r ~0.85-0.98; only 1 Kaiser
  factor). Testlet-bundle administration via the production `tutor_cat.engine` /
  `scenario_cat_lib`.
- **Of-record CAT stop = EAP posterior SD** (posterior SD <= 0.12 AND min_scenarios >= 8).
  This SUPERSEDES the earlier online normal-approx SE stop.
- **ARCHIVED / do NOT port:** `experiments/archive_onlineSE/` and the `--stop-rule online`
  path are the older, superseded methodology (kept for cross-benchmark comparison only).
  `experiments/13_eap_stop_prototype/` is the prototype that motivated the EAP stop and is now
  of-record. Only the EAP-stop artifacts graduate.
- **PROVISIONAL - N=52 models** (small, but 1D is strongly supported at this N).
- Snapshot tag: `wildbench-scenario-1d-52models`.

## Target

`flow/uni-frq` - unidimensional, single latent axis `overall`. Testlet administration (whole
scenario per CAT step, ability updated from all its criteria).

## Locked config

`min_scenarios = 8`, **EAP posterior SD target 0.12** (honest measurement SE; NOT the online
normal-approx SE), ridge 1e-2, GH fit grid 7, estimator **MWLE**, fixed production seed for
deployment single-run scoring. Near-all-fail models that cannot reach SE_ability <= 0.12 even
at the L=40 cap are flagged **`weakly_identified`** (theta reported as an upper bound).

## Payload

| Artifact | Path | Role | Tracked |
|---|---|---|---|
| CAT administration/scoring pool (canonical runtime) | `wildbench_calibration/wildbench_scenario_fitted_1d_catpool.jsonl` | 8,345 items across 985 scenarios (13 extreme_a removed); what the engine selects/scores from | yes |
| Full 1D calibration bank | `wildbench_calibration/wildbench_scenario_fitted_1d.jsonl` | 8,358 calibrated items (superset; reporting/leaderboard) | yes |
| CAT-pool exclusion mask | `wildbench_calibration/exclusion_mask.json` | catpool = full bank minus 13 extreme_a | yes |
| Items / questions | `data/WildBench/scenarios.jsonl` | 1,001 scenarios (~11.4 criteria each) | yes |
| Judge config | `judge_frozen.yaml` | frozen judge for grading responses | yes |
| Fit / build manifests | `wildbench_calibration/{fit_manifest.json,build_manifest.json}` | provenance, dimensionality, fit params | yes |
| Response matrix (calibration input) | `eduLLM-Evals/WildBenchGrade/response_matrix.csv` | reproducibility input only; NOT needed at flow runtime | no (local -> S3) |

**Bank schema note:** `discrimination` keyed by the modeled skill name `overall` (single axis)
+ scalar `difficulty`; `q_modeled = {"overall": 1}`; `irt_params.structure = "overall_1d"`.
Per-criterion records; administration/scoring is per-scenario testlet.

## Runtime data flow (scoring a new checkpoint)

1. Generate tutor responses from the checkpoint on `data/WildBench/scenarios.jsonl`.
2. Grade per criterion with `judge_frozen.yaml`.
3. Run the testlet CAT (whole scenario per step) via `tutor_cat.engine` / `scenario_cat_lib`
   against the **catpool**, stopping on **EAP posterior SD <= 0.12 / floor 8**, to estimate
   `overall` ability with SE.

## Provenance + metrics

- Fit: `calibrated-m2pl-scenario` confirmatory-m2pl-mml-em, `overall_1d`; **n_models = 52**;
  1,001 scenarios; 11,416 criteria -> 8,358 after zero-variance filtering; ridge 0.01, grid 7;
  calibrated 2026-08-06.
- Matrix sha256: `e6542f2025eaf4aeeefff5050074264ebefd9a21406cd98f5c4678515bffeb08`.
- Recovery (exp 05, OOS locked 8/0.12, MWLE, EAP stop): r = **0.975** [0.957, 0.986], slope
  0.921, theta-MAE 0.357; mean length 13.2 scenarios / 129 criteria. p-IRT pass-r 0.951,
  pass-MAE 0.047.
- Deployed precision: **45/52 (86.5%)** reach SE_ability <= 0.12; SE_total median 0.046.
- **Caveats:** N=52 (1D well-supported: 94% first-factor variance). **4 `weakly_identified`**
  near-all-fail models (OLMo-1B-hf, mGPT, smol_llama-220M x2) have theta as an upper bound, not
  a point estimate. Full detail in `README.md` (exp 03-13).

## Cross-benchmark integration note (important for the flow harness)

WildBench's of-record stop is the **EAP posterior SD**, whereas Bridge and BiGGen currently use
the engine's **online normal-approx SE** stop (pending their own migration). The FRQ flow loader
must therefore support a per-benchmark stop rule (WildBench = `eap`), not assume a single global
one, when these graduate into `flow/uni-frq`.

## Rerun / swap protocol (larger cohort)

1. Recalibrate at the larger cohort; re-emit `wildbench_scenario_fitted_1d.jsonl` + `_catpool.jsonl`
   (+ `exclusion_mask.json`) at ridge 1e-2 / grid 7.
2. Replace them here; bump `n_models` + `matrix_sha256` in `fit_manifest.json` / bank provenance.
3. Re-tag `wildbench-scenario-1d-<N>models`; re-port to `flow/uni-frq`.
4. Items / judge config unchanged unless criteria are revised.
