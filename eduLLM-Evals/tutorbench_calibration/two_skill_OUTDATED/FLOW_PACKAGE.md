> ⚠️ **OUTDATED / SUPERSEDED** — this is the 2-skill graduation manifest. The canonical TutorBench scale is now the correctness-only unidim package in `../unidim/`; see the repointed top-level `../../FLOW_PACKAGE.md`. Retained for reference only.

# TutorBench -> flow/mirt-frq graduation package (INTERIM)

Interim, self-contained payload for porting TutorBench's canonical **2-skill** calibration into
the FRQ eval flow. **Layout is provisional**: conform the file arrangement to the FRQ diagnostic
loader (`diagnostics/frq_cat/common/irt_params.py` on `CheckpointFlows`, mirroring
`diagnostics/mcq_cat/`) before committing into `flow/mirt-frq`.

## Status

- **PILOT BASELINE - NOT final** (criteria to be revised for the 200-run). Canonical instrument
  per owner: **2-skill (correctness, scaffolding), 115-model cohort**.
- **Of-record CAT stop = EAP-posterior honest per-skill MARGINAL SD** (marginal of the joint 2-D
  posterior), floor(min_scenarios)=20 / SE_ability target=0.27. This SUPERSEDES the earlier
  engine online normal-approx SE stop.
- **3-skill (adds presentation) is SECONDARY/exploratory** and does NOT graduate. The other bank
  variants (unidim, 82-model, `_excl60`, full non-fitted) are NOT the shipped instrument.
- Snapshot tag: `tutorbench-2skill-115models`.

## Target

`flow/mirt-frq` - multidimensional, 2 latent dims: **correctness, scaffolding** (content+diagnosis
collapsed -> correctness; presentation not modeled). Testlet administration (whole scenario per
CAT step; ability updated from all its criteria) via the production `tutor_cat.engine` /
`scenario_cat_lib`.

## Locked config

M2PL confirmatory MML-EM, `ridge = 1e-2`, GH fit grid 7, negative_policy = clamp. CAT stop:
**EAP-posterior marginal SD**, 161 nodes/dim, info-plateau delta=0.005 / W=3, cap 70, estimator
**MWLE** at stop. Operating point **floor = 20, SE_ability target = 0.27** (SELECTED out-of-sample
on the floor x SE grid over all 115 models, k=5 seed 20260729, refit-per-fold). Headline EXCLUDES
2 weakly-calibrated models (`Qwen/Qwen1.5-1.8B`, `BSC-LT/salamandra-7b-instruct`).

## Payload

| Artifact | Path | Role | Tracked |
|---|---|---|---|
| Deployment params bank (canonical) | `data/TutorBench/rubrics_qmatrix_calibrated_2skill_115_fitted.jsonl` | 3,666 fitted items; a per {correctness,scaffolding} + difficulty b | yes |
| Items / questions | `data/TutorBench/scenarios.jsonl` | prompts the tutor model responds to | yes |
| Curated scenarios (ref) | `data/TutorBench/curated/scenarios_curated.jsonl` | curated bank the fit was built on | yes |
| Judge config | `judge_frozen.yaml` | frozen judge for grading responses | yes |
| Of-record leaderboard | `tutorbench_calibration/model_leaderboard.csv` | per-model deployed CAT theta @ 20/0.27 with SE_total | yes |
| Fit manifest | `tutorbench_calibration/fit_manifest.json` | provenance, locked config, headline numbers | yes |
| Response matrix (calibration input) | `staging/response_matrix_full_nonopt_115.csv` | reproducibility input only; NOT needed at flow runtime | no (staging, gitignored -> S3) |

**Bank schema note:** the fitted bank keys `discrimination` by the modeled skill names
(`correctness`, `scaffolding`) with a scalar `difficulty`, and restates the Q-matrix in
`q_modeled`. This differs from the full 3-slot `{content,diagnosis,scaffolding}` layout that
`tutor_cat/schemas.py::Rubric.from_json` reads, so the FRQ flow loader must read the modeled-name
schema (or emit a 3-slot companion at port time).

## Runtime data flow (scoring a new checkpoint)

1. Generate tutor responses from the checkpoint on `data/TutorBench/scenarios.jsonl`.
2. Grade per criterion with `judge_frozen.yaml`.
3. Run the testlet CAT (whole scenario per step) via `tutor_cat.engine` / `scenario_cat_lib`
   against the fitted bank at the locked EAP-posterior stop (floor 20 / SE_ability 0.27) to
   estimate correctness + scaffolding ability with SE_total.

## Provenance + metrics

- Fit: `calibrate_mirt` confirmatory-m2pl-mml-em; `n_persons = 115`; ridge 0.01 (adopted
  2026-07-31); date 2026-08-02. Matrix sha256 `152286b6...cccc0eb`.
- Recovery (OOS, EAP stop @ 20/0.27, MWLE, N=113 excl-2): correctness **r=0.967 slope=1.019
  theta-MAE=0.437 median SE_total=0.371**; scaffolding **r=0.932 slope=0.953 theta-MAE=0.256
  median SE_total=0.320**. Median length 25 scenarios; both-skills SE_ability reach 34%; plateau
  66%; cap 0.
- **Caveats:** pilot baseline (criteria change at 200-run). ~34% both-skills SE_ability reach is
  an SE_param precision-ceiling artifact (SE_param ~0.21 corr), NOT a stop defect. 2
  weakly-calibrated models excluded from the headline. Full detail in `README.md` (exp 03-12) and
  `experiments/*/README.md`.

## Cross-benchmark integration note

TutorBench's of-record stop is the **EAP posterior marginal SD** (honest measurement SE). The FRQ
flow loader must support a per-benchmark stop rule (TutorBench = `eap`, 2-D marginal), not assume
a single global one, when these graduate into `flow/mirt-frq`.

## Rerun / swap protocol (200-run)

1. Recalibrate at the 200-model cohort; re-emit `rubrics_qmatrix_calibrated_2skill_<N>_fitted.jsonl`.
2. Re-run `scripts/eap_oos_grid_study.py --workers 6` + `scripts/of_record_f20se27.py`; refresh
   `tutorbench_calibration/{model_leaderboard.csv,fit_manifest.json}` + `experiments/`.
3. Re-tag `tutorbench-2skill-<N>models`; re-port to `flow/mirt-frq`.
4. Items / judge config unchanged unless criteria are revised (they will be for the 200-run).
