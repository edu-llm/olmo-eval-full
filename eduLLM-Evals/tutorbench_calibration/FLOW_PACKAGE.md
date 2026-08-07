# TutorBench -> flow/mirt-frq graduation package (INTERIM)

Interim, self-contained payload for porting TutorBench's canonical **correctness-only unidimensional**
calibration into the FRQ eval flow. **Layout is provisional**: conform the file arrangement to the FRQ
diagnostic loader (`diagnostics/frq_cat/common/irt_params.py` on `CheckpointFlows`, mirroring
`diagnostics/mcq_cat/`) before committing into `flow/mirt-frq`.

## Status

- **PILOT BASELINE - NOT final** (criteria to be revised for the 200-run). Canonical instrument:
  **correctness-only unidim (single latent: correctness), 115-model cohort** (headline N=114).
- **Of-record CAT stop = EAP-posterior honest 1-D MARGINAL SD**, floor(min_scenarios)=20 /
  SE_ability target=0.27.
- The prior **2-skill (correctness + scaffolding)** instrument is **SUPERSEDED** and retained under
  `two_skill_OUTDATED/` (with its own `FLOW_PACKAGE.md`). Correctness-only unidim is the *same*
  correctness axis (full-bank EAP θ rank agreement r=0.9994), cleaner (SE_param −46%, SE_total −16%)
  and shorter (median 20 vs 25 scenarios); it drops scaffolding as a measured construct.
- Snapshot tag: `tutorbench-unidim-correctness-only-115models`.

## Target

`flow/mirt-frq` - unidimensional, 1 latent dim: **correctness** (scaffolding dropped as a measured
construct). Testlet administration (whole scenario per CAT step; ability updated from all its
correctness-loading criteria) via the production `tutor_cat.engine` / `scenario_cat_lib`.

## Locked config

Unidimensional 2PL, MML-EM, `ridge = 1e-2`, GH fit grid 7, `negative_policy = clamp`. Bank built by
`unidim/scripts/build_correctness_only_bank.py` (keep correctness-loading criteria, drop every
scaffolding-loading criterion: 2842 kept of 3666). CAT stop: **EAP-posterior 1-D marginal SD**, dense
321-node grid, info-plateau delta=0.005 / W=3, cap 70, estimator **MWLE** at stop. Operating point
**floor = 20, SE_ability target = 0.27**. OOS = k=5 refit-per-fold, seed 20260729. Headline EXCLUDES
1 weakly-calibrated model (`Qwen/Qwen1.5-1.8B`); `salamandra-7b-instruct` is rehabilitated and kept.

## Payload

| Artifact | Path | Role | Tracked |
|---|---|---|---|
| Deployment params bank (canonical) | `tutorbench_calibration/unidim/bank/rubrics_qmatrix_correctness_only_unidim_115_fitted.jsonl` | 2842 fitted criteria; dim = correctness (a) + difficulty (b) | yes |
| Bank fit summary | `tutorbench_calibration/unidim/bank/bank_fit_summary.json` | kept/dropped counts, clamped-neg, max-a, loglik/AIC/BIC | yes |
| Items / questions | `data/TutorBench/scenarios.jsonl` | prompts the tutor model responds to | yes |
| Curated scenarios (ref) | `data/TutorBench/curated/scenarios_curated.jsonl` | curated bank the fit was built on | yes |
| Judge config | `judge_frozen.yaml` | frozen judge for grading responses | yes |
| Of-record leaderboard | `tutorbench_calibration/unidim/experiments/08_leaderboard/leaderboard_f20se27.csv` | per-model deployed CAT theta @ 20/0.27 with SE_total | yes |
| Headline summary | `tutorbench_calibration/unidim/summary.json` | op-point, locked config, all headline numbers | yes |
| Response matrix (calibration input) | `staging/response_matrix_full_nonopt_115.csv` | reproducibility input only; NOT needed at flow runtime | no (staging, gitignored -> S3) |

**Bank schema note:** the fitted bank keys `discrimination` by the single modeled dim
(`correctness`) with a scalar `difficulty`, and restates the Q-matrix in `q_modeled`. This differs
from the full 3-slot `{content,diagnosis,scaffolding}` layout that `tutor_cat/schemas.py::Rubric.from_json`
reads, so the FRQ flow loader must read the modeled-name schema (or emit a 3-slot companion at port time).

## Runtime data flow (scoring a new checkpoint)

1. Generate tutor responses from the checkpoint on `data/TutorBench/scenarios.jsonl`.
2. Grade per criterion with `judge_frozen.yaml`.
3. Run the testlet CAT (whole scenario per step) via `tutor_cat.engine` / `scenario_cat_lib`
   against the fitted unidim bank at the locked EAP-posterior stop (floor 20 / SE_ability 0.27) to
   estimate correctness ability with SE_total.

## Provenance + metrics

- Fit: `calibrate_mirt` unidimensional-2PL-mml-em; `n_persons = 115`; 1 dim; ridge 0.01; grid 7;
  clamp. Source = the of-record 2-skill fitted bank filtered to correctness-loading criteria.
- Recovery (OOS, EAP stop @ 20/0.27, MWLE, N=114): correctness **r=0.955 slope=1.006
  theta-MAE=0.416 median SE_total=0.275**. Median length 20 scenarios; %reach SE_ability≤0.27 = 59.6%;
  SE_param offset 0.113. p-IRT pass-rate MAE 0.020 (r=0.975). Adaptive reaches SE≤0.27 in 20 vs 68
  scenarios random (~3.4×).
- **Caveats:** pilot baseline (criteria change at 200-run). Scaffolding is not measured on this scale.
  1 weakly-calibrated model excluded from the headline. Full detail in `unidim/README.md`,
  `unidim/experiments/*/`, and the 2-skill/unidim head-to-head in `unidim/comparison_vs_2skill/`.

## Cross-benchmark integration note

TutorBench's of-record stop is the **EAP posterior marginal SD** (honest measurement SE). The FRQ
flow loader must support a per-benchmark stop rule (TutorBench = `eap`, 1-D marginal), not assume a
single global one, when these graduate into `flow/mirt-frq`.

## Rerun / swap protocol (200-run)

1. Recalibrate at the 200-model cohort; re-emit
   `rubrics_qmatrix_correctness_only_unidim_<N>_fitted.jsonl` via
   `unidim/scripts/build_correctness_only_bank.py`.
2. Re-run `unidim/scripts/oos_grid_full_unidim_correctness_only.py --workers 6` +
   `unidim/scripts/build_of_record_f20se27.py`; refresh `unidim/summary.json` + `unidim/experiments/`.
3. Re-tag `tutorbench-unidim-correctness-only-<N>models`; re-port to `flow/mirt-frq`.
4. Items / judge config unchanged unless criteria are revised (they will be for the 200-run).
