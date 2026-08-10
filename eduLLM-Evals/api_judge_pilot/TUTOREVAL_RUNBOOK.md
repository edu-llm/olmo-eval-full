# TutorEval frontier-judge production run

This branch grades the 52-model TutorEval response cohort with the frozen
frontier-judge selection from `origin/frq-lab`:

- judge: `gemini-group/gemini-3-flash-preview`
- adapter: `generic-binary-strict` v1 (evidence first)
- response format: JSON object
- temperature: 0
- maximum completion tokens: 6,144

The runtime prompt is the `generic-binary-strict` builder in
`run_api_judge_pilot.py`. The runner records its rendered-template hash and the
full frozen configuration in `manifest.json`.

## Inputs

- responses: `Full200Run/full200_results/Outputs/open/tutoreval/*.responses.jsonl`
- scenarios: `data/TutorEval/scenarios_final.jsonl`
- rubrics: `data/TutorEval/rubrics_qmatrix_final.jsonl`
- gateway credentials: `TFY_BASE_URL` + `TFY_API_KEY` (or the legacy
  `MODEL_API_BASE` + `MODEL_API_KEY`) in an ignored `.env`

Expected matrix: 52 models × 1,786 criteria = 92,872 cells. The frozen blank
response policy auto-fails 434 cells without an API call; 92,438 cells are sent
to the judge.

Use the clean bank on this branch. The copied gold-source bank on `frq-lab`
contains a UTF-8 BOM and mojibake introduced during extraction. Its 100-cell
gold study remains the selection evidence, but those corrupted files are not
production inputs.

## Commands

Run a one-model smoke into a separate output directory:

```bash
uv run --no-project --with httpx python eduLLM-Evals/api_judge_pilot/regrade_benchmark.py \
  --responses-dir eduLLM-Evals/Full200Run/full200_results/Outputs/open/tutoreval \
  --rubrics eduLLM-Evals/data/TutorEval/rubrics_qmatrix_final.jsonl \
  --scenarios eduLLM-Evals/data/TutorEval/scenarios_final.jsonl \
  --env /path/to/private/.env \
  --model gemini-group/gemini-3-flash-preview \
  --judge-name tutoreval-gemini3-v1 \
  --adapter generic-binary-strict \
  --prompt-version generic-binary-strict-v1 \
  --max-tokens 6144 --concurrency 128 --batch-size 512 \
  --model-limit 1 --expected-models 1 --expected-scenarios 828 --expected-criteria 1786 \
  --out-dir eduLLM-Evals/api_judge_pilot/grading_tutoreval_smoke
```

Run all 52 models into the production directory:

```bash
uv run --no-project --with httpx python eduLLM-Evals/api_judge_pilot/regrade_benchmark.py \
  --responses-dir eduLLM-Evals/Full200Run/full200_results/Outputs/open/tutoreval \
  --rubrics eduLLM-Evals/data/TutorEval/rubrics_qmatrix_final.jsonl \
  --scenarios eduLLM-Evals/data/TutorEval/scenarios_final.jsonl \
  --env /path/to/private/.env \
  --model gemini-group/gemini-3-flash-preview \
  --judge-name tutoreval-gemini3-v1 \
  --adapter generic-binary-strict \
  --prompt-version generic-binary-strict-v1 \
  --max-tokens 6144 --concurrency 128 --batch-size 1024 \
  --model-limit 999 --expected-models 52 --expected-scenarios 828 --expected-criteria 1786 \
  --out-dir eduLLM-Evals/api_judge_pilot/grading_tutoreval
```

The canonical outputs are `verdicts.jsonl`, `summary.json`, and
`manifest.json`. Exhausted request failures go to `errors.jsonl`, remain absent
from canonical verdicts, and are retried by rerunning the identical command.
The command exits successfully only at exact 92,872-cell coverage.
