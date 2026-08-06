# EAP-posterior stop rule — prototype (TutorBench + TutorEval)

**Status:** offline prototype, of-record CAT results untouched. Implements the
[`EAP-Posterior Stop Rule` handoff](../../plans+prds/) for the multidimensional (2-skill)
benchmarks. Per-branch: TutorBench results live on `frq/tutorbench`, TutorEval on
`frq/tutoreval`; the method + script are identical.

## What it tests

The engine currently stops on the **online / normal-approx SE** (`sqrt(diag(U))` from the
Laplace update in `tutor_cat/mirt.py`). That is optimistic when the ability posterior is
skewed. This prototype compares it head-to-head against stopping on the honest
**grid-integrated EAP posterior SD** — for the multidim case, the **per-skill marginal SD of
the JOINT posterior** over a dense product grid (`scenario_cat_lib.eap_subset_mean_var`).

Locked operating point: `min_scenarios = 12`, `SE = 0.30`, trace selection, seed 20260729.

## Method (offline, faithful)

Scenario selection does not depend on the stop rule, so per model we drive the **real**
engine (`scenario_cat_lib.run_one_model` → `tutor_cat.engine`) to the cap once to capture the
exact administration order, replay it through the real M2PL update, and truncate that one
sequence at the online-SE point vs the EAP-SD point. Fitted bank is frozen (no re-fit).

## Results (online → EAP)

**TutorBench** (correctness / scaffolding), 33 models:
- honest convergence, all skills: **30% → 79%**
- correctness: %reach 36%→79%, recovery r 0.924→0.961, θ-MAE 0.575→0.437
- scaffolding: %reach 39%→100%, recovery r 0.892→0.938, θ-MAE 0.344→0.278
- median length 14 → 23 (mean 14.6 → 26.4)

**TutorEval** (conceptual_understanding / quantitative_procedural), 52 models:
- honest convergence, all skills: **23% → 60%**
- conceptual: %reach 37%→64%, recovery r 0.836→0.881, θ-MAE 0.611→0.539
- quantitative: %reach 31%→67%, recovery r 0.927→0.965, θ-MAE 0.446→0.349
- median length 14 → 29 (mean 16.0 → 31.6)

## Read against the acceptance bar

Bar: raise %-reaching-target **and** not hurt recovery **without inflating median length**.
- Honest convergence: passes decisively (the online SE was badly optimistic here).
- Recovery: passes (every skill's r up, θ-MAE down).
- Median length: **fails** — it inflates (14→23 / 14→29), unlike the WildBench pilot.

Conclusion: the EAP stop is clearly more honest and improves recovery, but at SE 0.30 it
~doubles the test. **Adopt the EAP stop and re-lock the operating point** (re-run the SE×floor
sweep under the EAP stop so median length stays near today's ~14) rather than keep 12/0.30.
A few models genuinely cap within 50 scenarios → correctly flagged weakly identified.

## Reproduce

```bash
python eduLLM-Evals/scripts/eap_stop_prototype.py \
  --benchmark TutorBench \
  --bank eduLLM-Evals/data/TutorBench/rubrics_qmatrix_calibrated_2skill_fitted.jsonl \
  --matrix <TutorBench response_matrix.csv> \
  --scenarios eduLLM-Evals/data/TutorBench/scenarios.jsonl \
  --out-dir <out>
# TutorEval: bank rubrics_qmatrix_final_2skill_fitted.jsonl + TutorEval response_matrix.csv
```

## Caveats

- Offline prototype, not the live engine flag yet. Adoption still wants `--stop-se {online,eap}`
  wired into `tutor_cat/engine.py` (accumulate administered `(a,q,b,y)`; per-skill marginal SD).
- TutorEval matrix is an exact sha match to the bank's calibration matrix; TutorBench uses the
  33-model grading restricted to the frozen bank's criteria (params frozen, so valid).
- EAP-SD grid = 25 nodes/dim (denser than the handoff's 7/dim example) for an honest SD.
