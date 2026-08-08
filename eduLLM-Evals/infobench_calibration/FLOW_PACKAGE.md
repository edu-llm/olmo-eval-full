# InfoBench calibration → flow/uni-frq — package manifest

Pointer manifest for this self-contained **canonical unidim** calibration package. It maps the
graduation artifacts onto what ships **in this package** so the package stands alone.

## Target

`flow/uni-frq` — unidimensional 2PL, single latent axis `instruction_following`. Testlet
administration (whole scenario per CAT step; ability updated from all its criteria) via the
production `tutor_cat.engine` / `scripts/scenario_cat_lib.py`.

## Status

- **PROVISIONAL — N=52 persons** (`low_n`; below the ~150 identifiability floor). Unidimensional was
  chosen **deliberately**: the five InfoBench source skills (`content`, `format`, `number`, `style`,
  `linguistic`) collapse onto a single `instruction_following` axis at this sample size.
- Canonical instrument: **unidimensional (`instruction_following`)**.
- Snapshot tag: `infobench-unidim-52models`.
- STUDY / reporting only. Production engine (`tutor_cat/`, `scenario_cat_lib.py`) untouched; only the calibration artifacts + exported param bank are committed.

## Locked config

Unidimensional 2PL, **log-shrinkage-2PL λ16** (`log_a_prior_sd 0.25`), **401-node** `normal_trapezoid`
(bound 8), `returned_iterate`. CAT stop = **EAP-posterior 1-D marginal SD**, dense **801-node** grid,
plateau δ = 0.005 / W = 3, cap 70, MWLE θ at stop. Operating point **floor = 15 / SE_post target =
0.25**. OOS = k=5 refit-per-fold, seed 20260729. Full config + provenance:
`bank/bank_fit_summary.json`, `summary.json`.

## Payload (in this package unless noted)

| Artifact | Path | Role | Tracked |
|---|---|---|---|
| Deployment params bank (canonical) | `bank/rubrics_qmatrix_instruction_following_unidim_fitted.jsonl` | 2,105 fitted criteria; `discrimination.instruction_following` (a) + `difficulty` (b) | yes |
| Calibration input matrix | `bank/response_matrix.csv` | N=52 × 2250 frozen judge verdicts; the grid refits the bank from this (sha256 087948fc…) | yes |
| Judge provenance | `bank/judge_manifest.json` | frozen Qwen labels behind the matrix | yes |
| Bank fit provenance | `bank/bank_fit_summary.json` | matrix sha256, λ16 / 401-node config, loglik note, low-n + no-persisted-param-bank caveat | yes |
| Headline summary | `summary.json` | op-point, locked config, all headline numbers (all-52 + excl-weak) | yes |
| Of-record leaderboard | `experiments/08_leaderboard/leaderboard_f15se25.csv` | per-model deployed CAT θ @ 15/0.25 with SE_total | yes |
| Floor×SE selection grid | `experiments/06_floor_se_grid/oos_per_cell_grid*.csv` | op-point selection basis (SE_post-only %reach) | yes |
| Items / questions | `data/InFoBench/scenarios.jsonl` | prompts the tutor model responds to | yes (in `data/`) |
| Rubrics / criteria | `data/InFoBench/rubrics.jsonl` | per-scenario criteria the judge scores | yes (in `data/`) |
| Response matrix (source) | `runs/calibration/InFoBench_full_20260804/inputs/response_matrix.csv` | original of-record location; copied into `bank/` here | no (`runs/` gitignored) |

## Runtime data flow (scoring a new checkpoint)

1. Generate tutor responses from the checkpoint on `data/InFoBench/scenarios.jsonl`.
2. Grade per criterion with the frozen judge (see `bank/judge_manifest.json`).
3. Run the testlet CAT (whole scenario per step) via `tutor_cat.engine` / `scenario_cat_lib` against
   the persisted log-shrinkage-2PL bank `bank/rubrics_qmatrix_instruction_following_unidim_fitted.jsonl`
   at the locked EAP-posterior stop (floor 15 / SE_post 0.25) to
   estimate `instruction_following` with SE_total.

## Provenance + metrics

- Fit: `calibrate_mirt.fit_m2pl_em` unidim, log-shrinkage-2PL λ16, 401-node `normal_trapezoid`,
  `returned_iterate`; `n_persons = 52`. No single full-data loglik is persisted (per-fold refits ran
  to `max_iter=200` returned-iterate; n_kept ∈ [2088, 2101]); reproduce via `scripts/run_oos_grid.py`.
- Recovery (OOS, EAP stop @ 15/0.25, MWLE, excl-weak N=48): **r = 0.972, slope = 1.070,
  θ-MAE = 0.430, median SE_total = 0.258**, median length 17 scenarios, %reach 85.4%, sound SE_param
  ≈ 0.085. p-IRT pass-rate MAE 0.050 (r 0.973, OLS slope 1.043). Adaptive reaches SE_post ≤ 0.25 in
  ~17 vs ~30 scenarios random (~1.76×).
- **⚠️ SLOPE-EXPANSION caveat:** OOS slope ≈ 1.07–1.14 (> 1) = mild absolute-scale **expansion**
  (opposite of TutorEval's ~0.80 compression; close to TutorBench's ~1.0). **Rankings and pass-rate
  prediction are faithful** (r 0.97–0.98, p-IRT MAE 0.047–0.050); only absolute-θ units are slightly
  stretched. If absolute θ is needed at port, apply an affine rescale (÷ slope ≈ ÷1.07). Full
  discussion in `README.md`.
- **SE_param is the sound observed-information bootstrap (~0.085)**, NOT the stale nonparametric
  ~0.19; see the README "SE_param correction note".

## Cross-benchmark integration note

InfoBench's of-record stop is the **EAP posterior marginal SD** (honest measurement SE). The FRQ flow
loader must support a **per-benchmark stop rule** (InfoBench = `eap`, 1-D marginal) and carry the
**slope caveat direction per benchmark**: InfoBench expands (÷1.07) whereas TutorEval compresses
(÷0.80) and TutorBench is ≈1.0. Do not assume a single global slope correction.

## Rerun / swap protocol (next cohort)

1. Recalibrate at the larger cohort; re-emit the fitted params via
   `scripts/build_infobench_unidim_bank.py` (re-fit from the refreshed `response_matrix.csv`).
2. Re-run `scripts/run_oos_grid.py --workers 6` + `scripts/build_of_record_f15se25.py` (repointing
   their default paths — see `README.md` "Notes on scripts"); refresh `summary.json` + `experiments/`.
3. Re-tag `infobench-unidim-<N>models`; re-port to `flow/uni-frq`.
4. Items / rubrics / judge config unchanged unless criteria are revised.
