# TutorBench -> flow/mirt-frq graduation package (INTERIM)

Interim, self-contained payload for porting TutorBench's canonical calibration into the
FRQ eval flow. **Layout is provisional**: conform the file arrangement to the FRQ
diagnostic loader (`diagnostics/frq_cat/common/irt_params.py` on `CheckpointFlows`,
mirroring `diagnostics/mcq_cat/`) before committing into `flow/mirt-frq`.

## Status

- **PILOT BASELINE - NOT final** (criteria to be revised for the 200-run). Canonical
  instrument per owner: **2-skill, 115-model cohort**.
- Only `rubrics_qmatrix_calibrated_2skill_115_fitted.jsonl` graduates. The other tracked
  variants (unidim, 3-skill, 82-model `_fitted`, `_excl60`, full non-fitted banks) are NOT
  the shipped instrument and must not be ported.
- Snapshot tag: `tutorbench-2skill-115models`.

## Target

`flow/mirt-frq` - multidimensional, 2 latent dims: **correctness, scaffolding**
(content+diagnosis collapsed -> correctness; presentation not modeled).

## Locked config (2-skill instrument)

M2PL confirmatory MML-EM, `ridge = 1e-2`, grid 7, k=5 CV seed `20260729`. CAT policy:
`min-items (floor) = 15`, pbis >= 0.05, content-balanced selection, exposure top-5, policy
seed `20260730`; SE target / max-items per instrument default
(`scripts/cat_eval_tutorbench_multiskill.py --skills 2`).

## Payload

| Artifact | Path | Role | Tracked |
|---|---|---|---|
| Deployment params bank (canonical) | `data/TutorBench/rubrics_qmatrix_calibrated_2skill_115_fitted.jsonl` | 3,666 fitted items; a per `{correctness,scaffolding}` + difficulty b | yes |
| Items / questions | `data/TutorBench/scenarios.jsonl` | prompts the tutor model responds to | yes |
| Curated scenarios (ref) | `data/TutorBench/curated/scenarios_curated.jsonl` | curated bank the fit was built on | yes |
| Judge config | `judge_frozen.yaml` | frozen judge for grading responses | yes |
| Response matrix (calibration input) | `staging/response_matrix_full_nonopt_115.csv` | reproducibility input only; NOT needed at flow runtime | no (staging, gitignored -> S3) |

**Bank schema note:** the fitted bank keys `discrimination` by the *modeled* skill names
(`correctness`, `scaffolding`) with a scalar `difficulty`, and restates the Q-matrix in
`q_modeled`. This differs from the full 3-slot `{content,diagnosis,scaffolding}` layout that
`tutor_cat/schemas.py::Rubric.from_json` reads, so the FRQ flow loader must read the
modeled-name schema (or we emit a 3-slot companion at port time).

## Runtime data flow (scoring a new checkpoint)

1. Generate tutor responses from the checkpoint on `data/TutorBench/scenarios.jsonl`.
2. Grade per criterion with `judge_frozen.yaml`.
3. Feed graded outcomes + the fitted bank (a/b params) into the 2-dim M2PL CAT loop at the
   locked config to estimate correctness + scaffolding ability with SE.

## Provenance + metrics

- Fit: `calibrate_mirt` confirmatory-m2pl-mml-em; `n_persons = 115`
  (82-model pilot + 33 net-new tb33 from full-200 grading); ridge 0.01; date 2026-08-02.
- Matrix sha256: `152286b6ded6561aca84e4b90cb0abbf065634d374217f91c9edcc783cccc0eb`.
- Negative-loading policy: 349 negative fitted loadings clamped to 0.0 (runtime requires a>=0).
- Recovery: the documented pilot-baseline (N=82) figures are correctness r 0.960/0.852 (in/OOS),
  scaffolding 0.850/0.713, mean ~23.6 CAT items, pIRT MAE 0.0349 (see
  `plans+prds/Pilot Baseline Freeze (2 and 3 skill).md`). **115-cohort recovery numbers should
  be pulled from the regenerated 2-skill/115 CAT report before publishing** rather than assumed
  equal to the 82-model baseline.

## Rerun / swap protocol (200-run)

1. Recalibrate at the 200-model cohort; re-emit `rubrics_qmatrix_calibrated_2skill_<N>_fitted.jsonl`.
2. Replace the bank here; bump `n_persons` + `matrix_sha256` + `date` in its provenance.
3. Re-tag `tutorbench-2skill-<N>models`; re-port to `flow/mirt-frq`.
4. Items / judge config unchanged unless criteria are revised (they will be for the 200-run,
   per the pilot-baseline note - so expect a criteria refresh, not just a param swap, at 200).
