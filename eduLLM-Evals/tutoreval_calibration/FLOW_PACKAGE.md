# TutorEval calibration → flow/uni-frq — package manifest (gemini-3 canonical)

Pointer manifest for this self-contained **canonical unidim** calibration package, recalibrated on
the **`gemini-3-flash-preview`** judge. The authoritative graduation manifest (target loader, judge
config, snapshot tag, runtime data flow) is `data/TutorEval/FLOW_PACKAGE.md`; this file maps that
manifest onto the artifacts shipped **in this package** so the package stands alone. Supersedes the
Qwen-judge of-record (archived under `qwen_superseded/`).

## Target

`flow/uni-frq` — unidimensional 2PL, single latent axis `ability`. Testlet administration (whole
scenario per CAT step; ability updated from all its criteria) via the production
`tutor_cat.engine` / `scripts/scenario_cat_lib.py`.

## Status

- **PROVISIONAL — N=52 persons** (`low_n`; below the ~150 identifiability floor). Unidimensional
  was chosen **deliberately** (the 2-skill variant is not the shipped instrument at N=52).
- Judge = **`gemini-3-flash-preview`**; calibrated on OBSERVED gemini-3 labels (Rule 0).
- Canonical instrument per the flow-package owner: **unidimensional (`ability`)**.
- Snapshot tag: `tutoreval-unidim-gemini3-52models`.

## Locked config

Unidimensional 2PL, MML-EM, `ridge = 0.01`, `fit_grid = 7`, `negative_policy = clamp` (at consume
time). MINIMAL Rule-0 curation (542 all-fail columns auto-dropped → 1244 fitted criteria; no
near-sep exclusion). CAT stop = **EAP-posterior 1-D marginal SD**, dense 321-node grid, plateau
δ = 0.005 / W = 3, cap 70, MWLE θ at stop. Operating point **floor = 15 / SE_ability target = 0.25**.
OOS = k=5 refit-per-fold, seed 20260729. Full config + provenance: `bank/bank_fit_summary.json`,
`summary.json`, `se_judge/summary.json`.

## Payload (in this package unless noted)

| Artifact | Path | Role | Tracked |
|---|---|---|---|
| Deployment params bank (canonical) | `bank/rubrics_qmatrix_final_unidim_gemini3_fitted.jsonl` | 1244 fitted criteria; `discrimination.ability` (a) + `difficulty` (b) | yes |
| Bank fit provenance | `bank/bank_fit_summary.json` | matrix sha256, reproduction loglik, fit config, low-n caveat | yes |
| Headline summary | `summary.json` | op-point, locked config, all headline numbers + SE_judge layer | yes |
| SE_judge layer | `se_judge/summary.json` + per-model CSVs | α posteriors, both-regime SE stats, de-bias gate, Tier B | yes |
| Of-record leaderboard | `experiments/08_leaderboard/leaderboard_f15se25.csv` | per-model deployed OBSERVED θ @ 15/0.25 with SE_total (incl SE_judge) | yes |
| Secondary de-bias band | `experiments/11_debias_band/debias_band_full_bank_f15se25.csv` | full-bank per-model θ_debiased band (NOT of-record headline) | yes |
| Floor×SE selection grid | `experiments/06_floor_se_grid/decision_table.csv` | op-point selection basis | yes |
| Items / questions | `data/TutorEval/scenarios_final.jsonl` | prompts the tutor model responds to | yes (in `data/`) |
| Judge config | frozen `gemini-3-flash-preview` | grading responses per criterion | yes (per data/ manifest) |
| Response matrix (calibration input) | `api_judge_pilot/grading_tutoreval/response_matrix.csv` | reproducibility input only; NOT needed at flow runtime | committable (built) |

## Runtime data flow (scoring a new checkpoint)

1. Generate tutor responses from the checkpoint on `data/TutorEval/scenarios_final.jsonl`.
2. Grade per criterion with the frozen **gemini-3-flash-preview** judge.
3. Run the testlet CAT (whole scenario per step) via `tutor_cat.engine` / `scenario_cat_lib`
   against `bank/rubrics_qmatrix_final_unidim_gemini3_fitted.jsonl` at the locked EAP-posterior
   stop (floor 15 / SE_ability 0.25) to estimate `ability` (MWLE-at-stop) with SE_total.

## De-bias / SE_judge convention (HYBRID) — carry to the loader

- Deployed **headline θ = observed** op-point MWLE-at-stop; report **SE_total = √(SE_ability² +
  SE_param² + SE_judge²)** error bars — **SE_judge is the dominant floor** (op-point median 0.381).
- Attach the **cohort pass-rate de-bias caveat 1.83× [1.33–2.97×]** (UP; α ≈ 0.45 strict, β = 0).
  Do NOT apply it per-model in the headline.
- The **full-bank per-model θ_debiased band** (`se_judge/per_model_full_bank.csv`) is a **secondary
  diagnostic** (Spearman 0.993, 0 violations). Op-point per-model θ_debiased is **directional-only**
  for the weak tail (op-point Spearman 0.778).

## Provenance + metrics

- Fit: `calibrate_mirt` unidimensional-2PL-mml-em; `n_persons = 52`; 1 dim; ridge 0.01; grid 7;
  clamp. 1244 fitted criteria (542 all-fail auto-dropped). Fit loglik −18071.21.
- Recovery (OOS, EAP stop @ 15/0.25, MWLE, excl-weak N=49): **r = 0.954, slope = 0.879,
  θ-MAE = 0.292**, median length 15 scenarios, %reach 95.9%, median SE_total 0.437. all-52:
  r = 0.966, slope = 0.916. p-IRT pass-rate MAE 0.036 (r = 0.935). Adaptive reaches SE ≤ 0.25 in
  7 vs 70 scenarios random (~10×).
- **⚠️ Judge-bias / slope:** observed θ is biased DOWN (α ≈ 0.45 strict; cohort de-bias 1.83×
  [1.33–2.97×] UP, no crossover). OOS slope ≈ 0.88 = mild absolute-scale compression. **Rankings
  and pass-rate prediction are faithful** (full-bank Spearman 0.993, p-IRT MAE 0.036); absolute-θ /
  pass-rate carry the de-bias band. **Precision lever:** a judge-FAIL-enriched gold audit (not more
  cells) tightens α and the band.

## Cross-benchmark integration note

TutorEval's of-record stop is the **EAP posterior marginal SD** (honest measurement SE). The FRQ
flow loader must support a **per-benchmark stop rule** (TutorEval = `eap`, 1-D marginal). It must
also carry (a) the **SE_judge** dominant-floor SE_total convention and the **cohort de-bias band**,
and (b) the slope caveat (an affine rescale if absolute-θ units are needed).

## Rerun / swap protocol (next cohort or judge)

1. Recalibrate at the larger cohort / new judge; re-emit
   `rubrics_qmatrix_final_unidim_gemini3_fitted.jsonl`.
2. Re-run `scripts/run_oppoint_grid_gemini3.py --workers 6` + `scripts/build_of_record_f15se25.py`
   (repointing default paths — see `README.md` "Notes on scripts"); re-run the SE_judge layer
   (`scripts/se_judge_tutoreval.py`); refresh `summary.json` + `se_judge/` + `experiments/`.
3. Re-tag `tutoreval-unidim-<judge>-<N>models`; re-port to `flow/uni-frq`.
4. Items / judge config unchanged unless criteria are revised.
