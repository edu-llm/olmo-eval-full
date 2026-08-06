# EAP-Posterior Stop Rule — Handoff (TutorBench + InfoBench)

**Audience:** whoever runs the CAT-eval stop-rule change on `frq/tutorbench` / `frq/tutoreval`
(multidimensional) and `frq/infobench` (1-D).
**Author context:** piloted + adopted on WildBench (`frq/wildbench`). Reference implementation:
`eduLLM-Evals/wildbench_calibration/scripts/prototype_eap_stop.py` (1-D case).

---

## 1. What this is and why

**The problem (estimator mismatch).** The CAT engine currently stops on an **online /
normal-approximation SE** — the posterior SD from the single Laplace/covariance update (curvature at
the point estimate). That is **optimistic**: when the ability posterior is skewed (few informative
items / extreme-ability models), the true posterior SD is larger, so the engine can call a model
"converged" (online SE ≤ target) when its **honest EAP posterior SD is still above target**. Same
issue documented in `plans+prds/CAT Uncertainty Audit - Selection, Parameter Error, Estimator
Mismatch.md`.

**The fix.** Stop on the **grid-integrated EAP posterior SD over the administered items so far**,
not the normal-approx SE. Same target, same floor — just the honest uncertainty number.

**WildBench pilot result (1-D), for expectations:** % reaching SE ≤ target **60% → 92%**; **median
test length unchanged** (only borderline models extend → mean +~4 scen); OOS recovery **improved**
(r 0.957 → 0.975, slope 0.890 → 0.921, θ-MAE 0.428 → 0.357); only the genuinely-degenerate near-all-fail
models still cap (correctly "weakly identified"). Per-step cost is negligible.

**Acceptance criterion:** adopt only if it **raises %-reaching-target and doesn't hurt recovery**,
**without inflating median length**. Prototype first, adopt second.

---

## 2. Per-benchmark parameters

| | **TutorBench / TutorEval** | **InFoBench** |
|---|---|---|
| Dimensionality | **Multidimensional** — 2-skill `[correctness, scaffolding]` (opt. 3-skill +presentation) | **1-D** (`instruction_following`; 5 one-hot skills content/format/number/style/linguistic collapsed to 1-D by the 1-SE rule) |
| Posterior SD to stop on | **Per-skill marginal SD** from the JOINT multidim posterior (product grid) | **Single** 1-D posterior SD (like WildBench) |
| Locked operating point | **min_scenarios = 12, SE = 0.30** | **floor 0 (no floor), SE = 0.25**, trace selection |
| ⚠️ Prerequisite | none extra | **Dense-grid EAP rescore REQUIRED FIRST** (see §6) — their report used a 5-node grid so θ collapsed to 5 nodes |
| Reference generality | generalize the 1-D pilot → per-skill marginal SDs | **directly analogous to the WildBench 1-D pilot** |
| Study location | scenario-level TutorBench harness (playbook ⚠️ layer) | `reports/infobench_calibration_20260804/` on `frq/infobench` |

InfoBench is the easy case (1-D, same as WildBench). TutorBench is the harder case (multidim).

---

## 3. The code change

Stop decision lives in the engine's per-step SE computation: `tutor_cat/mirt.py` (ability/covariance
update + Fisher info → the current normal-approx SE) and `tutor_cat/engine.py` → `run_evaluation`
(stops when per-skill SE meets `RunConfig.max_se` + floors → `precision_reached`).

Add an EAP-posterior-SD stop behind a flag (e.g. `--stop-se {online,eap}`; keep `online` default until
adoption). At each candidate stop step, integrate the posterior over a θ grid from the administered
responses and use its SD:

```
posterior(θ) ∝ L(administered responses | θ) · prior(θ)     # θ over a FIXED DENSE grid
EAP  = ∫ θ · posterior dθ
SE   = sqrt( ∫ (θ − EAP)^2 · posterior dθ )                 # honest posterior SD
stop when  SE ≤ target  AND  min_scenarios met ;  else continue to the cap
```

- **InFoBench (1-D):** single integral — copy the WildBench `prototype_eap_stop.py` posterior-SD logic
  almost verbatim; use a **dense** grid (e.g. ~3201 nodes over [−8, 8], std-normal prior).
- **TutorBench (multidim):** integrate the **joint** posterior over a product grid (e.g. 7 nodes/dim,
  7^2 = 49 for 2-skill) and take the **marginal SD of each modeled skill**; stop when *every* modeled
  skill's marginal SD ≤ its `max_se`. Do NOT use independent per-skill 1-D integrals — the skills are
  correlated (correctness↔scaffolding ≈ −0.4), so the marginal from the joint matters.
- **Fitted bank (a_k, b) does NOT change** for either — stop-rule change only, no re-fit / no re-collapse.

---

## 4. Validation protocol (prototype BEFORE adopting)

1. Build the EAP-posterior stop behind a flag.
2. **Prototype run:** both stop rules head-to-head at the benchmark's locked op-point (TutorBench
   12/0.30; InFoBench floor-0/0.25), same folds/seed, same bank.
3. **Compare** into a separate `experiments/NN_eap_stop_prototype/` dir (of-record untouched):
   mean/median length, % reaching target (per skill for TutorBench), SE distribution, OOS recovery
   r/slope/θ-MAE.
4. **Adopt only if** it clears the acceptance bar (§1).

---

## 5. If adopted: RE-RUN vs KEEP

| Experiment / artifact | Re-run? | Why |
|---|---|---|
| Operating-point sweep (min_scenarios × SE grid) | **RE-RUN (full grid)** | Stop rule changes lengths → re-lock op-point; run the full grid so tradeoffs stay comparable |
| OOS recovery (θ-recovery, p-IRT) | **RE-RUN** | Administered sets change |
| CAT-vs-random efficiency | **RE-RUN** | Length/precision change |
| Estimator comparison (EAP/MWLE/MLE) | **RE-RUN** | Scored on the administered set |
| Order/seed stability | **RE-RUN** | Administration paths change |
| Deployed SE (SE_ability, SE_total, precision_reached, SE-post-vs-total) | **RE-RUN** | These ARE the stop-dependent numbers |
| SE_param (parameter uncertainty) | **RE-BOOTSTRAP on EAP-administered sets** | deployed SE_param depends on administered items (full-bank floor unchanged) |
| Dimensionality / collapse decision | **KEEP** | bank-level (TutorBench: content+diagnosis→correctness stands; InFoBench: 1-D stands) |
| Ridge / grid sensitivity | **KEEP** | bank-level |
| Full-bank leaderboard θ + full-bank SE_total | **KEEP** | full-bank scoring doesn't use the CAT stop rule |
| Fitted bank / calibrated params | **KEEP** | no re-fit |

**Provenance:** archive the online-SE of-record (e.g. `*_onlineSE.*`) before overwriting.

---

## 6. ⚠️ InFoBench prerequisite: fix the coarse-grid θ quantization FIRST

InFoBench's report states its **reference EAP used 5 quadrature points, so θ collapsed onto 5 discrete
nodes** — the identical coarse-grid quantization we fixed on Bridge/BiGGen/WildBench. This must be fixed
**before** the EAP-posterior stop is meaningful (a posterior SD computed on a 5-node grid is garbage):
1. **Dense 1-D rescore:** switch the reference/scoring EAP to a **fine uniform grid** (~3201 nodes over
   [−8, 8], std-normal prior), mirroring the WildBench/BiGGen fix. Their report already flags "dense
   one-dimensional rescoring is required before final recovery / CAT-efficiency claims."
2. Re-do the deployed SE / recovery / op-point on the dense grid (this is needed regardless of the stop
   rule; it's on their limitations list).
3. **Then** layer the EAP-posterior stop on top (it uses the same dense grid).

For TutorBench, confirm its reference grid is already dense (it uses `calibrate_mirt`'s grid; verify the
recovery reference isn't coarse-quantized) — if it is coarse, apply the same dense-grid fix first.

---

## 7. Scripts / entry points
- Stop-rule change: `tutor_cat/engine.py` + `tutor_cat/mirt.py` (per-skill posterior SD for multidim).
- Scenario-level wrappers (playbook ⚠️ layer): `offline_engine_driver.py` (add `--stop-se eap`),
  `cat_eval_tutorbench_multiskill.py`, `scenario_kfold_estimator_cv.py`, `se_sweep_aggregate.py`,
  `scenario_param_uncertainty.py`.
- Keep folds + CAT seed fixed at **20260729** (fold partition AND administration seed held constant
  across k-fold — intentional for reproducibility; order/seed experiment quantifies seed sensitivity).
- InFoBench study dir: `reports/infobench_calibration_20260804/`; its frozen inputs
  (`response_matrix.csv`, `judge_manifest.json`) are in external storage (SHA-256 hashes in that
  README) — retrieve to reproduce.

---

## 8. Gotchas
- **Multidim (TutorBench):** marginal SD from the JOINT posterior, not independent 1-D integrals.
- **InFoBench:** it's 1-D — the WildBench prototype is a near drop-in — but do the dense-grid fix (§6) first.
- **Weak/degenerate models still cap** (no bank information at their θ) — flag "weakly identified"
  (θ bound only); don't chase precision the bank can't provide.
- **One change at a time** — stop-rule only (plus the InFoBench dense-grid prerequisite, which is a
  separate, already-required fix). Don't co-mingle with bank/Q-matrix edits.
- SE targets differ (TutorBench 0.30, InFoBench 0.25, WildBench 0.12) → the online-vs-posterior gap
  bites differently; the prototype comparison shows how much per benchmark.
- Once TutorBench + InFoBench adopt, update Calibration Playbook §8 to make the EAP-posterior stop the
  documented suite default.
