# Qwen judge grading results

- **Judge:** qwen / Qwen/Qwen3.5-9B @ `c202236235762e1c871ad0ccb60c8ee5ba337b9a`

## TutorBench
- models: 33, criteria: 6462
- coverage: complete=False, holes=17, filled=213229
- verdicts: {'fail': 190501, 'pass': 22728, 'no_decision': 17}
- policy: {'prompt_version': 'judge-validation-v3', 'normalization_version': 'judge-normalization-v3', 'evidence_policy_version': 'criterion-evidence-gate-v1'}

## TutorEval
- models: 52, criteria: 1786
- coverage: complete=False, holes=1, filled=92871
- verdicts: {'fail': 78488, 'pass': 14383, 'no_decision': 1}
- policy: {'prompt_version': 'judge-validation-v3', 'normalization_version': 'judge-normalization-v3', 'evidence_policy_version': 'criterion-evidence-gate-v1'}

## Files
- `<Bench>/response_matrix.csv` — models x (scenario_id, criterion_id) 0/1 matrix
- `<Bench>/verdicts.jsonl` — one normalized row per (model, scenario, criterion)
- `<Bench>/manifest.json` — provenance + coverage
- `<Bench>/cases_index.jsonl` — de-blind map (case_id -> model/scenario/criterion)
- `raw/<Bench>.verdicts_full.jsonl` — full judge output incl. rationale / evidence / raw_output

- `responses/<Bench>/<model>.responses.jsonl` — tutor models open-ended answers (Output) + prompts
