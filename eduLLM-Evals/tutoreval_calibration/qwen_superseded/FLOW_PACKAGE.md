# TutorEval calibration → flow/uni-frq — package manifest

Pointer manifest for this self-contained **canonical unidim** calibration package. The authoritative
graduation manifest (target loader, judge config, snapshot tag, runtime data flow) is
`data/TutorEval/FLOW_PACKAGE.md`; this file just maps that manifest onto the artifacts shipped **in
this package** so the package stands alone.

## Target

`flow/uni-frq` — unidimensional 2PL, single latent axis `ability`. Testlet administration (whole
scenario per CAT step; ability updated from all its criteria) via the production
`tutor_cat.engine` / `scripts/scenario_cat_lib.py`.

## Status

- **PROVISIONAL — N=52 persons** (`low_n`; below the ~150 identifiability floor). Unidimensional was
  chosen **deliberately**: at N=52 the multi-dim loadings and the ~0.98 latent correlation are
  unstable, so a 2-skill variant is not the shipped instrument (the teammate's 2-skill prototype was
  audited and rejected — see `README.md`).
- Canonical instrument per the flow-package owner: **unidimensional (`ability`)**.
- Snapshot tag: `tutoreval-unidim-52models`.

## Locked config

Unidimensional 2PL, MML-EM, `ridge = 0.01`, `fit_grid = 7`, `negative_policy = clamp` (at consume
time). CAT stop = **EAP-posterior 1-D marginal SD**, dense 321-node grid, plateau δ = 0.005 / W = 3,
cap 70, MWLE θ at stop. Operating point **floor = 15 / SE_ability target = 0.22**. OOS = k=5
refit-per-fold, seed 20260729. Full config + provenance: `bank/bank_fit_summary.json`,
`summary.json`.

## Payload (in this package unless noted)

| Artifact | Path | Role | Tracked |
|---|---|---|---|
| Deployment params bank (canonical) | `bank/rubrics_qmatrix_final_unidim_fitted.jsonl` | 1,186 fitted criteria; `discrimination.ability` (a) + `difficulty` (b) | yes |
| Bank fit provenance | `bank/bank_fit_summary.json` | matrix sha256, reproduction loglik, fit config, low-n caveat | yes |
| Headline summary | `summary.json` | op-point, locked config, all headline numbers | yes |
| Of-record leaderboard | `experiments/08_leaderboard/leaderboard_f15se22.csv` | per-model deployed CAT θ @ 15/0.22 with SE_total | yes |
| Floor×SE selection grid | `experiments/06_floor_se_grid/oos_per_cell_grid_reach_corrected*.csv` | op-point selection basis (corrected %reach) | yes |
| Items / questions | `data/TutorEval/scenarios_final.jsonl` | prompts the tutor model responds to (matches the `_final` bank) | yes (in `data/`) |
| Judge config | `judge_frozen.yaml` | frozen judge for grading responses | yes (per data/ manifest) |
| Response matrix (calibration input) | `runs/judge/TutorEval/response_matrix.csv` | reproducibility input only; NOT needed at flow runtime | no (S3-origin; `runs/` gitignored) |

## Runtime data flow (scoring a new checkpoint)

1. Generate tutor responses from the checkpoint on `data/TutorEval/scenarios_final.jsonl`.
2. Grade per criterion with the frozen judge (`judge_frozen.yaml`).
3. Run the testlet CAT (whole scenario per step) via `tutor_cat.engine` / `scenario_cat_lib` against
   `bank/rubrics_qmatrix_final_unidim_fitted.jsonl` at the locked EAP-posterior stop (floor 15 /
   SE_ability 0.22) to estimate `ability` with SE_total.

## Provenance + metrics

- Fit: `calibrate_mirt` unidimensional-2PL-mml-em; `n_persons = 52`; 1 dim; ridge 0.01; grid 7;
  clamp. Reproduction loglik = −17111.06 (matches manifest).
- Recovery (OOS, EAP stop @ 15/0.22, MWLE, excl-weak N=48): **r = 0.950, slope = 0.801,
  θ-MAE = 0.333, median SE_total = 0.217**, median length 15 scenarios, corrected %reach 97.9%,
  SE_param offset 0.112. p-IRT pass-rate MAE 0.036 (r = 0.953). Adaptive reaches SE ≤ 0.22 in 10 vs
  79 scenarios random (~7.9×).
- **⚠️ SLOPE CAVEAT:** OOS slope ≈ 0.80 = a uniform ~20% absolute-scale compression (a genuine bank
  property, not tail noise; rises only to ~0.84/0.86 at floor 20/25, never ~1.0). **Rankings and
  pass-rate prediction are faithful** (r 0.95–0.98, p-IRT MAE 0.036); only absolute-θ units are
  shrunk. If absolute θ is needed at port, apply an affine rescale (÷ slope). This differs from
  TutorBench (slope ≈ 1.0). Full discussion in `README.md`.

## Cross-benchmark integration note

TutorEval's of-record stop is the **EAP posterior marginal SD** (honest measurement SE). The FRQ flow
loader must support a **per-benchmark stop rule** (TutorEval = `eap`, 1-D marginal), not assume a
single global one, when these graduate into `flow/uni-frq`. It must also carry the slope caveat: if a
scale is later needed in absolute-θ units, TutorEval needs the affine rescale that TutorBench does not.

## Rerun / swap protocol (next cohort)

1. Recalibrate at the larger cohort; re-emit `rubrics_qmatrix_final_unidim_fitted.jsonl` via the
   `staging/tutoreval_calibration/` export.
2. Re-run `scripts/run_oos_grid_unidim.py --workers 6` + `scripts/build_of_record_f15se22.py`
   (repointing their default paths — see `README.md` "Notes on scripts"); refresh `summary.json` +
   `experiments/`.
3. Re-tag `tutoreval-unidim-<N>models`; re-port to `flow/uni-frq`.
4. Items / judge config unchanged unless criteria are revised.
