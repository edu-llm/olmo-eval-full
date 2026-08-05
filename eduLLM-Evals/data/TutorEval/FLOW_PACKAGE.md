# TutorEval -> flow/uni-frq graduation package (INTERIM)

Interim, self-contained payload for porting TutorEval's canonical calibration into the
FRQ eval flow. **Layout is provisional**: conform the file arrangement to the FRQ
diagnostic loader (`diagnostics/frq_cat/common/irt_params.py` on `CheckpointFlows`,
mirroring `diagnostics/mcq_cat/`) before committing into `flow/uni-frq`.

## Status

- **PROVISIONAL - N=52 persons** (`low_n`; below the identifiability floor of ~150).
  Unidimensional was chosen deliberately: at N=52 the multi-dim loadings and the ~0.98
  latent correlation are unstable, so the 2-skill variant is not the shipped instrument.
- Canonical instrument per owner: **unidimensional ("ability")**. Only
  `rubrics_qmatrix_final_unidim_fitted.jsonl` graduates; `..._2skill_fitted.jsonl` does not.
- Snapshot tag: `tutoreval-unidim-52models`.

## Target

`flow/uni-frq` - unidimensional 2PL, single latent axis `ability`.

## Locked config

Unidimensional 2PL, MML-EM, `ridge = 0.01`, grid 7 nodes/dim. CAT stopping/selection policy
per `staging/tutoreval_calibration/calibration_mirt_manifest.json` (not restated here to avoid
drift). The item bank is config-independent (params do not depend on the stopping rule).

## Payload

| Artifact | Path | Role | Tracked |
|---|---|---|---|
| Deployment params bank (canonical) | `data/TutorEval/rubrics_qmatrix_final_unidim_fitted.jsonl` | 1,186 fitted items; `discrimination.ability` (a) + `difficulty` (b) | yes |
| Items / questions | `data/TutorEval/scenarios_final.jsonl` | prompts the tutor model responds to (matches the `_final` bank; confirm scenario_id resolution at port) | yes |
| Judge config | `judge_frozen.yaml` | frozen judge for grading responses | yes |
| Response matrix (calibration input) | `runs/judge/TutorEval/response_matrix.csv` | reproducibility input only; NOT needed at flow runtime | (calibration input -> S3) |
| Dimensionality analysis (evidence) | `regenerated_figures/tutoreval_skillfit/` + `plot_tutoreval_skillfit.py` | why unidim was chosen | yes |

**Bank schema note:** `discrimination` is keyed by the modeled skill name `ability`
(single axis) with a scalar `difficulty`; `q_modeled = {"ability": 1}`.

## Runtime data flow (scoring a new checkpoint)

1. Generate tutor responses from the checkpoint on `data/TutorEval/scenarios_final.jsonl`.
2. Grade per criterion with `judge_frozen.yaml`.
3. Feed graded outcomes + the fitted bank (a/b params) into the unidimensional 2PL CAT loop
   to estimate `ability` with SE.

## Provenance + caveats

- Fit: `unidimensional-2pl-mml-em` (re-run of calibrate_tutoreval's EM; item params
  recomputed by `staging/tutoreval_calibration/export_tutoreval_fitted_bank.py`), `n_persons = 52`,
  ridge 0.01, reproduction loglik matches manifest.
- Matrix sha256: `f7830280a40f6bb774116bb6e7a44f9840073c88d7ce8fbd538b357cdcbd4781`.
- Negative loadings preserved (not floored) in this fitted export.
- **Caveats:** N=52 is below the ~150 identifiability floor (`low_n`); treat abilities as
  coarse. Recovery/coverage numbers should be read from
  `staging/tutoreval_calibration/coverage_report.json` and the `tutoreval_skillfit` figures,
  not assumed here.

## Rerun / swap protocol (larger cohort)

1. Recalibrate at the larger cohort; re-emit `rubrics_qmatrix_final_unidim_fitted.jsonl`.
2. Replace the bank here; bump `n_persons` + `matrix_sha256` in its provenance.
3. Re-tag `tutoreval-unidim-<N>models`; re-port to `flow/uni-frq`.
4. Items / judge config unchanged unless criteria are revised - then expect a criteria
   refresh, not just a param swap.
