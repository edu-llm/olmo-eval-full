# Judge Error — Quantification + Mitigation Plan

**Status:** draft / pre-work. No code changes yet.
**Branch:** `frq/tutorbench` (the human-label + recovered judge-validation data lives here).
**Scope:** the single frozen LLM judge (`Qwen/Qwen3.5-9B`, `generic-binary`, `judge-validation-v3` +
`criterion-evidence-gate-v1`) used to grade all six benchmarks — InfoBench, WildBench, TutorBench,
TutorEval, Bridge, BiGGen.

---

## 1. Problem statement

We run an LLM-as-judge to grade model responses on six benchmarks, then calibrate an IRT bank and
run CAT evaluations off those judge labels. We have tuned stop rules, ridges, SE targets, etc. — but
we have **never propagated the judge's own error** into the numbers we report. Two goals:

- **A. Quantify** the error the judge injects, so we can (i) report it as an error term when we
  certify other teams' models, and (ii) *correct* for it, not just caveat it.
- **B. Mitigate** it via prompting / operating-point / (last resort) fine-tuning.

Working assumption from the team: **we will not switch the judge** even with more human labels. That
does not lower the value of labels — it shifts them from *judge selection* to *quantify + correct a
judge we are committed to*.

---

## 2. What we already know (recovered v3 baseline — TutorBench only)

Recovered from `graderValidationStuff/judge-v3-results/` (5 judges × 6 waves × 261 cases) compared
against the reconstructed human gold (`graderValidationStuff/_recovery/human_labels.csv`, rebuilt
from the `frq/tutorbench` grader packets). Reproduce with
`graderValidationStuff/_recovery/run_compare.py`.

**Acceptance gates (all judges FAIL; Qwen is best):**

| Judge | macro-F1 (≥0.80) | crit-fail sens. (≥0.90) | test-retest (≥0.90) | prompt-flip (≤0.10) | per-skill F1 (≥0.70) | Overall |
|---|---|---|---|---|---|---|
| **qwen** | **0.719** ✗ | **0.637** ✗ | **0.985** ✓ | **0.050** ✓ | content .67✗ / diag .72✓ / scaff .62✗ | **FAIL** |
| flow | 0.621 ✗ | 0.569 ✗ | 0.996 ✓ | 0.096 ✓ | all <0.70 ✗ | FAIL |
| selene | 0.542 ✗ | 0.304 ✗ | 0.728 ✗ | 0.437 ✗ | ✗ | FAIL |
| gemma | 0.534 ✗ | 0.245 ✗ | 0.812 ✗ | 0.215 ✗ | ✗ | FAIL |
| prometheus | 0.430 ✗ | 0.275 ✗ | 0.985 ✓ | 0.100 ✓ | ✗ | FAIL |

**Qwen error budget (canonical_r1 confusion, vs 261 raw human labels):**

- accuracy 0.724, MCC 0.458, coverage 0.985 (4 `no_decision`).
- Humans PASS (145): judge → 119 pass / 25 fail → **false-fail rate α ≈ 25/145 = 0.17**.
- Humans FAIL (116): judge → 43 pass / 70 fail → **false-pass rate β ≈ 43/116 = 0.37**.
- Critical failures: 102 critical human-fails, **37 missed** → sensitivity 0.64
  (`critical` 0.59, `critical_negative` 1.00).

**Key characterization:**

1. **Qwen's error is BIAS, not NOISE.** test-retest 0.985 and prompt-flip 0.05 mean it is nearly
   deterministic. The failures are systematic (accuracy/bias), not stochastic. → methods that only
   reduce variance (self-consistency / k-sample majority vote) have almost nothing to fix here.
2. **Directionally lenient.** β (0.37) ≫ α (0.17): the judge passes ~37% of true failures and misses
   ~37% of *critical* failures. Used to certify a model, it **over-credits**. This is a directional
   bias to correct, not just a variance to widen.
3. **Correlated / systematic across models.** One fixed judge grades everyone, so per-criterion
   judge error is shared across all graded models. It does **not** shrink as we add candidate models
   → it is a *floor* on model separability, and it biases the calibrated item parameters (`a`,`b`)
   themselves, not just the scores.
4. **Only TutorBench is validated.** The 0.72 / 0.37 numbers do **not** transfer to the other five
   benchmarks (different domains/rubric styles). We currently have *zero* judge-quality evidence for
   InfoBench, WildBench, TutorEval, Bridge, BiGGen.

**Caveats on the baseline:** computed against **raw, unadjudicated** human labels (includes the
documented likely-wrong Fails `tb_0001_c02/c03`); v3 thresholds are **diagnostic** (the v3 prompt was
developed on the v2 cases), not a clean acceptance test.

---

## 3. Framing that drives the plan

- **Judge error decomposes into:** (a) **systematic per-criterion bias** — the judge grades a given
  criterion wrong for everyone; does not cancel, biases absolute θ / pass-rate; and (b) **stochastic
  per-cell noise** — test-retest / prompt-flip; partly cancels, inflates variance. For Qwen, (a)
  dominates and (b) is near-zero.
- **Two deployment stages with different cost budgets:**
  - **Calibration-time (offline, one-time batch):** grading the calibration cohort to fit the bank.
    Cost/latency don't matter → expensive judge configs are affordable here.
  - **CAT-eval-time (online, per candidate):** adaptive evaluation of a new model. Latency/cost
    matter; human-in-the-loop is impossible.
  - **The judge config need not be identical at both stages.** We can use a stronger/slower judge at
    calibration time and a cheap single-pass judge online, *provided we characterize the online
    judge's error* (which is exactly goal A). Qwen's near-perfect repeatability makes this safe.
- **Two distinct deliverables for reporting:** *relative ranking robustness* (does judge error
  reorder the leaderboard? — mostly the stochastic part; tends to be robust) vs *absolute score
  error* (systematic; the thing that bites when certifying another team's number). Report both as
  separate quantities.

---

## 4. What per-benchmark human labels buy us (beyond SE_judge)

Even with the judge fixed:

1. **Per-benchmark error rates** (α, β) — not transferable from TutorBench; the only way to know the
   judge isn't catastrophic on a given benchmark.
2. **De-biasing**, not just error bars — invert the measured confusion to report a *corrected*
   pass-rate/θ.
3. **Bank validity audit** — detect where calibrated `a`,`b` are judge artifacts (items the judge
   grades leniently look artificially easy).
4. **Eval harness for goal B** — you can't measure a prompt/threshold/fine-tune improvement on a
   benchmark without that benchmark's human labels.
5. **Risk surfacing** — discover an unusable-on-benchmark-X judge before certifying on X.

Sampling note: a **probability sample (~80–150 cells/benchmark) with defined inclusion
probabilities** is enough for usable α/β + CIs — do **not** repeat the InfoBench-style risk-enriched
audit (non-representative by construction). Prioritize the tutoring-focused benchmarks first
(TutorEval, Bridge).

---

## 5. Phased plan

### Phase 0 — Reproducibility + label integrity (prerequisite)
- **0.1** Freeze the recovered gold: commit `human_labels.csv` + the build/compare scripts so the v3
  numbers reproduce from the committed tree.
- **0.2** Run the documented **adjudication gate** on disputed human labels (start with
  `tb_0001_c02/c03`): blind re-review against criterion + response only, record reviewer + resolved
  label + rationale, version the adjudicated file. Re-run the comparison on adjudicated labels to get
  the corrected baseline.
- **Exit:** a reproducible, adjudicated TutorBench baseline + committed pipeline.

### Phase 1 — QUANTIFY (goal A)
- **1.1 Per-benchmark human-label sampling design.** Pre-register a probability sample per benchmark
  (inclusion probabilities, size, stratification by skill/criticality/pass-rate). Prioritize
  TutorEval + Bridge.
- **1.2 Collect labels + compute confusion.** Reuse `run_judge_validation.py` /
  `compare_judge_reliability.py`. Output α, β, critical sensitivity, and CIs **per benchmark** (and
  per skill/criticality where n allows).
- **1.3 SE_judge propagation.** Monte-Carlo / parametric bootstrap: resample response-matrix cells
  from the estimated per-stratum confusion (and per-cell flip prob), **refit `calibrate_mirt` and
  re-score θ inside every replicate** (same within-fold discipline as the calibration playbook §8.3).
  Spread of θ across replicates = `SE_judge`. Report
  `SE_total = √(SE_ability² + SE_param² + SE_judge²)` on the leaderboard.
- **1.4 Bias band + de-bias.** Because β≠α, also report a **systematic band** (best/worst-case θ +
  leaderboard rank shifts under the plausible confusion range), and a **de-biased** point estimate
  (measurement-error inversion using α, β).
- **1.5 Report split.** Deliver *relative ranking robustness* and *absolute score error* as separate
  numbers.
- **Exit:** every reported score carries a judge-error term (variance + systematic band) and a
  de-biased estimate; per-benchmark α/β documented.

### Phase 2 — MITIGATE (goal B) — tiered by cost / where it runs

> Ordered by payoff-for-effort given Qwen's *bias-not-noise* profile. All Tier-1 items are online-safe
> (no CAT-time cost spike). Each improvement is measured on the same held-out human labels from Phase 1.

**Tier 1 — free, online-compatible (do first):**
- **2.1 Decision-threshold calibration.** The v4 atomic runner emits per-atomic `p_fail`
  (`failure_probability_threshold` default 0.5). Move the threshold to trade β↓ vs α↑ — directly
  attacks the critical-failure-sensitivity gap (0.64) at **zero** added online cost. Pick the
  operating point certification needs (e.g. minimize letting failures through).
- **2.2 Prompt specialization.** Per-benchmark prompt/rubric decomposition. The teammate's
  **v4.3.2 atomic-decomposition** run (`graderValidationStuff/outputs/…`) is exactly this — same
  single pass, changes accuracy not cost. Finish/validate it (its 237-case dev run currently fails
  validation: incomplete manifest + a `tb_0355_c06` checklist-text drift + a few parse failures).
- **2.3 Post-hoc de-bias.** The Phase-1.4 correction is itself a mitigation applied *after* grading —
  zero online cost; complements 2.1/2.2.

**Tier 2 — offline-only (calibration-time / audit; NOT in the online CAT loop):**
- **2.4 Two-judge consensus** — only helps if the second judge's errors are *uncorrelated* (a
  frontier judge, not another open 9B). Use at calibration time to clean the bank, or as an audit.
- **2.5 Human-review-on-disagreement** — impossible online; use offline to build/expand gold and to
  clean the calibration bank.
- **2.6 Stronger judge at calibration-time only** — e.g. a frontier judge to fit a cleaner bank,
  while keeping cheap Qwen online (relies on Phase-1 characterization to reconcile the two).

**Tier 3 — last resort:**
- **2.7 Fine-tune / distill the judge.** *Targets the systematic accuracy/bias ceiling* (macro-F1,
  critical sensitivity) — the gates prompting can't close. SFT on (rubric, response) → verdict +
  evidence, or DPO on human-adjudicated disagreements, or distill a frontier judge into the 9B.
  Heavyweight: needs enough labels, an **untouched per-benchmark holdout**, and re-validation of a new
  frozen artifact across all six benchmarks. **Gate:** only if Tier-1 (2.1–2.3) plateaus short of
  targets.

**Explicitly de-prioritized:**
- **Self-consistency / k-sample majority vote** — low value here: Qwen's error is systematic, not
  random (test-retest 0.985), so resampling returns the same biased answer. Not worth the CAT-time
  cost.

### Phase 3 — Re-validate, freeze, integrate
- Re-run the acceptance gates on any changed judge config against the **untouched holdout** (never the
  tuning set — v3 was already tuned on v2; do not repeat that contamination).
- Freeze the winning config; wire `SE_judge` + de-bias into the standard leaderboard reporting so
  every future certification carries the judge-error term automatically.

---

## 6. Decision gates
- **G0 → G1:** adjudicated, reproducible baseline exists.
- **G1 → report:** per-benchmark α/β + SE_judge propagation validated on ≥ the priority benchmarks.
- **Tier1 → Tier3:** proceed to fine-tuning **only** if threshold + prompt specialization plateau
  below targets on the held-out labels.
- **Any judge-config change → freeze:** must clear acceptance on an untouched holdout before deploy.

## 7. Open questions / dependencies
- Human-labeling budget + which benchmarks first (leaning TutorEval, Bridge).
- Whether v3 numbers should be re-quoted only after adjudication (recommended).
- Whether calibration-time and CAT-time judge configs are allowed to differ (recommended: yes).
- Recover/confirm the v4.3.2 study's held-out **test** split + inputs from the cluster
  (`/home/ikchen/orcd/scratch/edu-judge-v4/`) before using it for goal B.

## 8. Artifacts / pointers
- Recovered v3 judge waves: `graderValidationStuff/judge-v3-results/<judge>/<wave>/…` (gitignored — large).
- v4.3.2 atomic dev run: `graderValidationStuff/outputs/…` (gitignored — large).
- Reconstructed gold + recovery pipeline: `graderValidationStuff/_recovery/` (`human_labels.csv`,
  `reliability_report.json`, `reliability_summary.csv`, `build_human_labels.py`, `run_compare.py`,
  `summarize.py`).
- Grader packets (source of gold): `eduLLM-Evals/grader_packets/grader_0{1..6}.csv` (tracked here).
- Comparison code: `scripts/compare_judge_reliability.py`, `scripts/run_judge_validation.py`.
- Calibration/propagation primitives: `scripts/calibrate_mirt.py`, `scripts/scenario_cat_lib.py`,
  `plans+prds/Calibration Playbook - Cross-Benchmark.md` (§8.2 SE_total).
