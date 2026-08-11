# TutorEval gemini-3 calibration — inputs (committed for reproducibility)

Small copies of the calibration inputs so the package reproduces without the raw
109 MB grading dump (which is gitignored and Slack/S3-sourced).

- `response_matrix.csv` / `criteria_index.csv` — 52 models × 1786 criteria pass/fail
  matrix (100% coverage, pass-rate 13.9%). Built from the gemini-3 grading verdicts
  (`TutorEvalGeminiGrade/verdicts.jsonl`, judge `gemini-3-flash-preview`,
  `generic-binary-strict`, max_tokens 6144) via
  `api_judge_pilot/build_response_matrix.py` (pass→1, else→0).
- `gold/` — the 100-cell human gold used to measure the judge confusion (α=0.45,
  β=0.00). Recovered from `frq/biggen` (commit "finalize tutoreval judge gemini-3 on
  gold set"): 80 fail / 20 pass; `stratum`=primary_skill, `criticality` present.

Judge-error rates derived from these are in `../experiments/10_judge_error/` and
`../se_judge/`.
