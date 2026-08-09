# Grading RUNBOOK — frontier-API judge (production)

Finalized judges (see `checkpoints/`), all with the `generic-binary-strict` (v1) prompt +
robust regex parser, JSON mode, temperature 0:

| benchmark | model (gateway slug) | max_tokens | who |
| --- | --- | --- | --- |
| biggen | gemini-group/gemini-3-flash-preview | 6144 (auto) | owner |
| tutoreval | gemini-group/gemini-3-flash-preview | 6144 (auto) | teammate |
| tutorbench | claude-group/claude-haiku-4-5 | 4096 | (deferred) |

## Prereqs
- `eduLLM-Evals/.env` with `MODEL_API_KEY` + `MODEL_API_BASE` (TrueFoundry gateway). Not in git.
- Inputs are LARGE/LOCAL and are NOT all in git — provision on the run machine:
  - Tutor responses: `Full200Run/full200_results/Outputs/open/<benchmark>/*.responses.jsonl`.
  - biggen bank: `.edullm_gen/biggen/bank/{rubrics,scenarios}.jsonl` (gitignored; regenerate with
    `conform_flow_package_bank.py` from the `biggen-cal-52models` tag if missing).
  - tutoreval bank: `gold/tutoreval_src/{rubrics,scenarios}.jsonl` (committed).
- gemini-3 auto-bumps to `max_tokens 6144` in `regrade_benchmark.py` (verbose; lower caps
  truncate ~2.4% of cells -> fake fails). Do not override below 6144 for gemini-3.
- Runs are resumable: same `--out-dir` skips already-graded (model, criterion) via
  incremental `verdicts.jsonl`. Interrupt-safe.

## SMOKE FIRST (1 model), then full. Confirm `unscorable` is near-zero on the smoke.

### biggen (owner)
```bash
# smoke (1 model): expect errors=0, unscorable ~0 (<1%)
uv run --no-project --with httpx python eduLLM-Evals/api_judge_pilot/regrade_benchmark.py \
  --responses-dir eduLLM-Evals/Full200Run/full200_results/Outputs/open/biggen \
  --rubrics .edullm_gen/biggen/bank/rubrics.jsonl \
  --scenarios .edullm_gen/biggen/bank/scenarios.jsonl \
  --model gemini-group/gemini-3-flash-preview --adapter generic-binary-strict \
  --max-tokens 6144 --concurrency 128 --model-limit 1 \
  --out-dir eduLLM-Evals/api_judge_pilot/grading_biggen
# full: same command with --model-limit 999 and the SAME --out-dir (resumes past the smoke)
```

### tutoreval (teammate)
```bash
uv run --no-project --with httpx python eduLLM-Evals/api_judge_pilot/regrade_benchmark.py \
  --responses-dir eduLLM-Evals/Full200Run/full200_results/Outputs/open/tutoreval \
  --rubrics eduLLM-Evals/api_judge_pilot/gold/tutoreval_src/rubrics.jsonl \
  --scenarios eduLLM-Evals/api_judge_pilot/gold/tutoreval_src/scenarios.jsonl \
  --model gemini-group/gemini-3-flash-preview --adapter generic-binary-strict \
  --max-tokens 6144 --concurrency 128 --model-limit 1 \
  --out-dir eduLLM-Evals/api_judge_pilot/grading_tutoreval
# full: --model-limit 999, same --out-dir.
```

### tutorbench (deferred; haiku)
Same pattern with `--model claude-group/claude-haiku-4-5 --max-tokens 4096` and the
TutorBench responses + rubrics/scenarios. Glance at `unscorable` on the smoke (haiku is less
verbose than gemini-3, but tutorbench responses are long).

## After a run
- Output: `<out-dir>/verdicts.jsonl` (model, criterion_id, verdict, status, reason, tokens)
  + `<out-dir>/summary.json` (pass rate, unscorable, throughput, token totals).
- Sanity vs the frozen judge budget: false-pass ~0% (biggen/tutoreval), macro-F1/MCC per the
  benchmark checkpoint under `checkpoints/`.
- Throughput: gemini-3 ~19-21 calls/s at c=128 (verbose); full biggen (52 models x 2,337) is
  a few hours. Cost is gemini-flash-tier (cheap); watch token volume on long-response cells.
