# TutorEval -> flow/uni-frq graduation package (gemini-3 of-record, INTERIM)

Authoritative graduation manifest for TutorEval's canonical **unidimensional** calibration,
recalibrated on the **`gemini-3-flash-preview`** judge. The self-contained package (bank +
experiments + SE_judge layer) lives under `eduLLM-Evals/tutoreval_calibration/` (see its
`FLOW_PACKAGE.md`); this file is the graduation-facing summary (target loader, judge config,
snapshot tag, runtime data flow). **Layout is provisional**: conform to the FRQ diagnostic loader
(`diagnostics/frq_cat/common/irt_params.py` on `CheckpointFlows`, mirroring `diagnostics/mcq_cat/`)
before committing into `flow/uni-frq`.

This supersedes the Qwen-judge of-record archived under `tutoreval_calibration/qwen_superseded/`.

## Status

- **PROVISIONAL - N=52 persons** (`low_n`; below the ~150 identifiability floor). Unidimensional
  chosen deliberately: at N=52 the multi-dim loadings / ~0.98 latent correlation are unstable.
- **Judge = `gemini-3-flash-preview`** (frontier); calibrated on OBSERVED gemini-3 labels. Carries
  an explicit **SE_judge** (judge-error) layer on top of SE_ability + SE_param.
- Canonical instrument: **unidimensional (`ability`)**.
- Snapshot tag: `tutoreval-unidim-gemini3-52models`.

## Target

`flow/uni-frq` - unidimensional 2PL, single latent axis `ability`. Testlet administration (whole
scenario per CAT step) via the production `tutor_cat.engine` / `scripts/scenario_cat_lib.py`.

## Locked config

Unidimensional 2PL, MML-EM, `ridge = 0.01`, `fit_grid = 7`, `negative_policy = clamp`. Rule-0
curation: 542 all-fail columns dropped -> **1,244 fitted criteria**. CAT stop = **EAP-posterior
1-D marginal SD**, dense 321-node grid (plateau delta=0.005 / W=3, cap 70), MWLE theta at stop.
Operating point **floor = 15 / SE_ability target = 0.25**. OOS = k=5 refit-per-fold, seed 20260729.
Full config/provenance: `tutoreval_calibration/bank/bank_fit_summary.json`,
`tutoreval_calibration/summary.json`, `tutoreval_calibration/se_judge/summary.json`.

## Payload

| Artifact | Path | Role | Tracked |
|---|---|---|---|
| Deployment params bank (canonical) | `tutoreval_calibration/bank/rubrics_qmatrix_final_unidim_gemini3_fitted.jsonl` | 1,244 fitted criteria; `discrimination.ability` (a) + `difficulty` (b) | yes |
| Bank fit provenance | `tutoreval_calibration/bank/bank_fit_summary.json` | matrix sha256, fit config, loglik, low-n caveat | yes |
| SE_judge layer | `tutoreval_calibration/se_judge/summary.json` (+ per-model CSVs) | judge-error alpha/beta, SE_judge stats, de-bias band | yes |
| Items / questions | `data/TutorEval/scenarios_final.jsonl` | prompts the tutor model responds to | yes |
| Judge config | frozen `gemini-3-flash-preview` (prompt/config in `api_judge_pilot/`) | grading responses per criterion | see api_judge_pilot |
| Response matrix (calibration input) | `api_judge_pilot/grading_tutoreval/response_matrix.csv` | reproducibility input only; NOT needed at flow runtime | built/committable |

**Bank schema note:** `discrimination` keyed by the modeled skill name `ability` (single axis) +
scalar `difficulty`; `q_modeled = {"ability": 1}`.

## Runtime data flow (scoring a new checkpoint)

1. Generate tutor responses from the checkpoint on `data/TutorEval/scenarios_final.jsonl`.
2. Grade per criterion with the frozen **`gemini-3-flash-preview`** judge.
3. Feed graded outcomes + `tutoreval_calibration/bank/rubrics_qmatrix_final_unidim_gemini3_fitted.jsonl`
   into the unidim 2PL EAP-posterior CAT loop (floor 15 / SE_ability 0.25, MWLE-at-stop) to estimate
   `ability`.
4. Report **observed theta** with **SE_total = sqrt(SE_ability^2 + SE_param^2 + SE_judge^2)** -
   SE_judge is the dominant floor (op-point median ~0.381). Attach the **cohort pass-rate de-bias
   band 1.83x [1.33-2.97x] UP** (alpha ~0.45 strict, beta=0) to cohort pass-rates only; do NOT apply
   it per-model in the headline.

## Provenance + metrics

- Fit: `calibrate_mirt` unidim-2PL-mml-em, `n_persons = 52`, ridge 0.01, grid 7, clamp; 1,244 fitted
  (542 all-fail dropped); fit loglik -18071.21. Matrix sha256 in `bank/bank_fit_summary.json`.
- Recovery (OOS, EAP @ 15/0.25, MWLE, excl-weak N=49): **r = 0.954, slope = 0.879, theta-MAE = 0.292**,
  median 15 scenarios, %reach 95.9%; all-52 r = 0.966. p-IRT pass-rate MAE 0.036 (r = 0.935).
- **Judge-bias / slope:** observed theta biased DOWN (alpha ~0.45 strict); OOS slope ~0.88 (mild
  compression). Rankings + pass-rate prediction faithful (full-bank Spearman 0.993); absolute-theta /
  pass-rate carry the de-bias band. If absolute units are needed at port, apply an affine rescale.
- Supersedes the Qwen-judge of-record (`tutoreval_calibration/qwen_superseded/`).

## Cross-benchmark integration note

Of-record stop = EAP posterior marginal SD; the FRQ flow loader must support a per-benchmark stop
rule (TutorEval = `eap`, 1-D marginal), carry the SE_judge dominant-floor convention + cohort de-bias
band, and the slope caveat. Judge = gemini-3 (differs from the Qwen-judge benchmarks until they migrate).

## Rerun / swap protocol (next cohort or judge)

1. Recalibrate on the frozen judge; re-emit `rubrics_qmatrix_final_unidim_gemini3_fitted.jsonl`.
2. Re-run the op-point grid + of-record build + SE_judge layer; refresh `summary.json` + `se_judge/` +
   `experiments/`.
3. Re-tag `tutoreval-unidim-<judge>-<N>models`; re-port to `flow/uni-frq`.
4. Items / judge config unchanged unless criteria are revised.
