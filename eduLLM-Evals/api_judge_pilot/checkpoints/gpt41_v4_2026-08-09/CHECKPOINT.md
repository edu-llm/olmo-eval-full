# Checkpoint: gpt-4.1 + generic-binary-strict-v4 (2026-08-09)

Frozen best-so-far judge config on the tutorbench 261-case human set. This is the
regression baseline: if later prompt/parser changes regress the metrics below, revert to
this config (prompt text is frozen in `prompt.txt` independent of the code).

## Config
- Model: `openai-group/gpt-4.1` (TrueFoundry gateway; `MODEL_API_BASE`/`MODEL_API_KEY`).
- Prompt: `generic-binary-strict-v4` (verbatim in `prompt.txt`): analysis -> verdict ->
  evidence field order (reason before deciding for quality; verdict before the verbose
  evidence for truncation-robustness).
- Decoding: temperature 0, `response_format={"type":"json_object"}`, max_tokens 2048.
- Parser: regex verdict extraction with `json.loads` fallback, fail-closed.
- Auto-fail policy: generation error / empty / missing -> fail without a judge call.
- Optimal concurrency: ~256.

## Baseline metrics (see metrics.json)
accuracy 0.751 | balanced 0.751 | F1 0.770 | MCC 0.499 | false-pass 25.0% |
critical-failure sensitivity 0.784 | critical false-pass 21.6% | content false-pass 25.0% |
unscorable 0.

Regression = critical-sensitivity < 0.76, false-pass > 28%, content false-pass > 28%, or
unscorable > 3.

## Reference verdicts
`gpt-4.1.v4.verdicts.jsonl` -- the exact per-case verdicts that produced the baseline.
Diff a new run against this to localize any regression.

## Reproduce / verify
```bash
uv run --no-project --with httpx python eduLLM-Evals/api_judge_pilot/run_api_judge_pilot.py \
  --labels eduLLM-Evals/api_judge_pilot/human_labels.csv \
  --models openai-group/gpt-4.1 --adapter generic-binary-strict-v4 \
  --json-mode --max-tokens 2048 --concurrency 64 \
  --out-dir eduLLM-Evals/api_judge_pilot/results_verify
uv run --no-project python eduLLM-Evals/api_judge_pilot/audit_compare.py \
  --results-dir eduLLM-Evals/api_judge_pilot/results_verify
```

## Caveats
261 cases with known label noise -> directional. The winning config may differ per
benchmark; re-select on the biggen/tutoreval gold sets. Keep the robust regex parser in
all configs.
