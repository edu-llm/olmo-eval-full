# FINALIZED tutorbench judge: claude-haiku-4-5 + generic-binary-strict (v1) — 2026-08-09

Cost-conscious pick on the tutorbench 261-cell human set (`gold/tutorbench_core/`). ~94% of
sonnet's MCC at the lowest false-pass and ~1/3 the cost. Frozen prompt in `prompt.txt`
(same v1 template as biggen/tutoreval checkpoints).

## Config
- Model: `claude-group/claude-haiku-4-5` (TrueFoundry gateway).
- Prompt: `generic-binary-strict` (v1, evidence-first; see `prompt.txt`).
- Decoding: temperature 0, `response_format={"type":"json_object"}`, **max_tokens 4096**.
- Parser: regex verdict extraction with `json.loads` fallback, fail-closed (REQUIRED).
- Auto-fail policy: generation error / empty / missing -> fail without a judge call.

## Baseline metrics (tutorbench gold, 261 cells; see metrics.json)
false-pass 22.4% | macro-F1 74.6 | MCC 0.497 | accuracy 74.7 | false-fail 27.6% |
test-retest flip 1.5% | prompt-flip 4.2% | unscorable 0. Frozen Qwen baseline: FP 37.1%,
macro-F1 72.7, MCC 0.461 -> the API judge beats Qwen.

Regression = false-pass > 28%, macro-F1 < 0.71, MCC < 0.45, test-retest flip > 6%, unscorable > 3.

Quality alternative: sonnet-4-6 + v1 (MCC 0.529, macro-F1 76.4) at ~3x the cost. Cheap GPT
tiers (gpt-4.1-nano 60% false-pass, gpt-5-mini) are too weak; gemini-3 truncates on
tutorbench's long responses under v1.

## Reference verdicts
`haiku.v1.gold_verdicts.jsonl` — per-cell verdicts behind the baseline.

## Reproduce / verify
```bash
uv run --no-project --with httpx python eduLLM-Evals/api_judge_pilot/score_gold.py \
  --gold eduLLM-Evals/api_judge_pilot/gold/tutorbench_core/gold_labels.jsonl \
  --sample eduLLM-Evals/api_judge_pilot/gold/tutorbench_core/sample.jsonl \
  --sample-manifest eduLLM-Evals/api_judge_pilot/gold/tutorbench_core/sample_manifest.json \
  --models claude-group/claude-haiku-4-5 --adapter generic-binary-strict \
  --max-tokens 4096 --replicates 2 --variants whitespace politeness \
  --out-dir eduLLM-Evals/api_judge_pilot/gold/results_verify
```

## Caveats
261 cells with known label noise; tutorbench is hard for all judges (MCC ~0.5) vs biggen
(~0.87) / tutoreval (~0.70). Config is benchmark-specific: biggen + tutoreval use
gemini-3-flash-preview + v1. Keep the robust regex parser everywhere.
