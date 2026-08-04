# FRQ LLM-as-judge selection and skills / Q-matrix

Evidence for the FRQ LLM-as-judge and skills/rubrics subsections of `Outline.md`. This section holds the judge-validation design, thresholds, blinded case set, and frozen config, plus the skill definitions and Q-matrix that the MIRT calibration (`../03_FRQ_MIRT`) depends on.

> **Data-availability caveat (see also `../GAPS.md`).** The judge-validation pipeline, study design, blinded cases, and frozen config are in-repo. The computed per-judge metric tables (macro-F1, critical-failure sensitivity, per-skill F1, test-retest agreement, prompt-flip rate) are not. Those tables live in a gitignored `runs/` tree and S3. The outcomes below come from `Outline.md` and the study design; the numeric per-judge scoreboard has to be regenerated from the wave JSONLs with `eduLLM-Evals/scripts/compare_judge_reliability.py`.

## Study design (`data/STUDY_DESIGN.json`)

- **261 blinded response-criterion cases** (`data/judge_cases.blinded.jsonl`, verified 261 lines): 10 scenarios by 3 tutor responses, expanded to criterion level. The judge sees the scenario, the tutor response, and one criterion. It does not see the human label or tutor identity (blinded for bias and anonymity).
- **Six waves per judge**: `canonical_r1/r2/r3` for test-retest reliability, plus three meaning-preserving prompt perturbations `whitespace_r1`, `header_synonyms_r1`, `instruction_politeness_r1` for prompt stability. 5 judges by 6 waves is the full grid.
- **Judges compared**: Prometheus (1 to 5), Flow (binary), Selene (Yes/No), Qwen (JSON pass/fail), Gemma (JSON pass/fail). Each ran in its native output format, then results were normalized.
- **Acceptance thresholds**:

  | Metric | Threshold |
  |---|---|
  | criterion macro-F1 | ≥ 0.80 |
  | critical-failure sensitivity | ≥ 0.90 |
  | test-retest strict agreement (worst pair) | ≥ 0.90 |
  | mapped primary-skill macro-F1 (each skill) | ≥ 0.70 |
  | prompt-variant flip rate (worst) | ≤ 0.10 |

- These 261 cases are a development/recalibration set used to diagnose v2, not an untouched final-acceptance holdout. Formal acceptance needs a separate unseen scenario-level holdout.

## Outcomes (qualitative, from `Outline.md`)

- **No judge passed every threshold.** Qwen was the strongest practical local candidate.
- Frontier judges were each restricted to **174 of 261 cases**: a judge was blocked from grading tutor responses from its own provider family, to avoid self-preference bias.
- The strongest local judges (Qwen, Flow, Selene) were re-run under stricter evidence requirements. A 96-case pilot compared zero-shot, few-shot, and countercheck prompting.
- Qwen's disagreements were reviewed by hand. Some were real judge errors. Others exposed questionable human labels or unclear grading policy, so the ground truth itself is noisy.

## Frozen production judge (`data/judge_frozen.yaml`)

- **Qwen/Qwen3.5-9B**, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
- **Zero-shot binary**, one criterion at a time, evidence required from the tutor response, `p_fail ≥ 0.33 → fail`, temperature 0, seed 42.
- Prompt `judge-validation-v3`, normalization `judge-normalization-v3`, evidence gate `criterion-evidence-gate-v1`.
- Chosen over frontier judges for reproducibility (self-hosted frozen checkpoint), privacy, deployment control, and low marginal cost across the hundreds of thousands of judgments the calibration needs. This is the judge used for the `../03_FRQ_MIRT` matrices.

## Skills and Q-matrix

- **Skill definitions**: `data/skill_definitions_v2.md`, the content / diagnosis / scaffolding v2 definitions used to label criteria.
- **Human Q-label audit** (`data/qmatrix_human_review_summary.json`, 25 criteria, 3 raters): inter-rater agreement is strong for content (unanimous 84%, pairwise Cohen's κ ≈ 0.86) and scaffolding (unanimous 80%), but weaker for diagnosis (unanimous 64%). This matches the MIRT finding that content and diagnosis are hard to separate, and it motivates collapsing content and diagnosis into "correctness" during calibration.
- Q-matrix banks live under `eduLLM-Evals/data/<Benchmark>/`: TutorBench (`rubrics_qmatrix_final.jsonl`, 6,462 criteria across 662 scenarios), TutorEval (828 scenarios, 1,786 criteria), WildBench (1,001 scenarios), Bridge (250 scenarios, 5 skills), EduBench (9 task types). These are large and version-controlled in place rather than copied here. `../03_FRQ_MIRT` uses the curated TutorBench bank.

## Regenerate the per-judge scoreboard (the missing numbers)

```bash
cd /Users/arhant/Documents/EDLM/olmo-eval-full/eduLLM-Evals
# needs the six-wave verdict JSONLs per judge + human_labels.csv (kept out of the repo):
scripts/compare_reliability_suite.sh human_labels.csv runs/judge_reliability_v3/waves \
  reliability_report.json reliability_summary.csv
```
