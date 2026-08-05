# BiGGen -> flow/uni-frq graduation package (INTERIM)

Interim, self-contained payload for porting BiGGen's finalized calibration into the
FRQ eval flow. **Layout is provisional**: conform the file arrangement to the FRQ
diagnostic loader (`diagnostics/frq_cat/common/irt_params.py` on `CheckpointFlows`,
mirroring `diagnostics/mcq_cat/`) before committing into `flow/uni-frq`.

## Status

- **PROVISIONAL - 52 models** (below the ideal ~150). A larger rerun is expected (~1-2
  days out). The item bank is **config-independent** (item params do not depend on the
  stopping rule), so a rerun is a drop-in swap of the bank files (see Rerun protocol).
- Snapshot tag: `biggen-cal-52models`.

## Target

`flow/uni-frq` - unidimensional ("general" skill), MWLE estimator, scenario-level CAT.

## Locked CAT config

`SE target = 0.12`, `min_scenarios (floor) = 8`, `ridge = 0.01`, estimator = **MWLE**,
selection = trace, `k = 5` CV, seed `20260729`. Deployed test length ~13.6 scenarios to
reach SE 0.12 (see README experiment 07b).

## Payload

| Artifact | Path | Role | Tracked |
|---|---|---|---|
| Deployment params bank | `biggen_calibration/bank/biggen_unidim_modeled.jsonl` | 2,389 fitted items (a/b per criterion) the engine scores a new model with | yes (copied here from `staging/`) |
| Calibrated bank (full) | `biggen_calibration/bank/biggen_unidim_calibrated.jsonl` | full calibrated rubric+params bank (incl. metadata) | yes |
| Items / questions | `data/BiGGen/scenarios.jsonl` | prompts + conversation_context the tutor model responds to | yes |
| Rubric criteria | `data/BiGGen/rubrics.jsonl` | per-scenario criteria (also embedded in the banks) | yes |
| Judge config | `judge_frozen.yaml` | frozen judge for grading responses identically | yes |
| Response matrix (calibration input) | `staging/biggen_response_matrix.csv` (52 x 2,678) | reproducibility input only; NOT needed at flow runtime | no (staging, gitignored -> keep in S3) |

## Runtime data flow (scoring a new checkpoint)

1. Generate tutor responses from the checkpoint on `data/BiGGen/scenarios.jsonl`.
2. Grade responses per criterion with `judge_frozen.yaml`.
3. Feed graded per-criterion outcomes + `biggen_unidim_modeled.jsonl` (a/b params) into the
   MWLE/CAT loop at the locked config to estimate ability theta with SE.

## Provenance + headline metrics

- Fit: `calibrate_mirt.fit_m2pl_em`, skill=`general`, ridge=0.01, n_models=52.
- Matrix sha256: `86e9516fcb123df7df761743d35286079325b360a7db4236d4594369016fccd7`.
- OOS of record @ locked config: **r = 0.970, slope = 0.812**, 100% convergence (n=52).
- p-IRT pass-rate MAE **0.047**; parameter uncertainty SE_total ~0.197 (SE_param ~0.094).
- Caveats: N=52 is provisional; leaderboard adjacent-pair SE bands overlap (only coarse
  ability bands distinguishable); across-seed theta SD ~0.179. See `README.md` for the full
  experiment log (01-10).

## Rerun / swap protocol (~150 models)

1. Recalibrate; re-emit `biggen_unidim_modeled.jsonl` (+ calibrated bank).
2. Replace the two bank files here; bump `n_models` + `matrix_sha256` in their provenance.
3. Re-tag `biggen-cal-<N>models`; re-port to `flow/uni-frq`.
4. Items / rubrics / judge config are unchanged unless new scenarios are added, so the flow
   harness itself does not change - only these data artifacts.
