# FINALIZED tutoreval judge: gemini-3-flash-preview + generic-binary-strict (v1) — 2026-08-09

Selected on the tutoreval 100-cell human gold set (`gold/tutoreval_core/`). Best MCC + 0%
false-pass + 0 unscorable + cheapest, and identical to the finalized biggen config (both
scaling targets converge on gemini-3 + v1). Frozen prompt in `prompt.txt` (same v1 template
as the biggen checkpoint).

## Config
- Model: `gemini-group/gemini-3-flash-preview` (TrueFoundry gateway).
- Prompt: `generic-binary-strict` (v1, evidence-first; see `prompt.txt`).
- Decoding: temperature 0, `response_format={"type":"json_object"}`, **max_tokens 4096**.
- Parser: regex verdict extraction with `json.loads` fallback, fail-closed (REQUIRED).
- Auto-fail policy: generation error / empty / missing -> fail without a judge call.

## Baseline metrics (tutoreval gold, 100 cells; see metrics.json)
false-pass 0.0% | macro-F1 82.8 | MCC 0.703 | accuracy 91.0 | false-fail 45.0% |
test-retest flip 2.0% | prompt-flip 3.0% | unscorable 0. No Qwen baseline (no local matrix).

Regression = false-pass > 5%, macro-F1 < 0.78, MCC < 0.65, test-retest flip > 6%, unscorable > 2.

Caveat: over-strict (false-fail 45%) but on only 20 gold-pass cells (noisy); 0% false-pass
is the headline. gpt-4.1 + v1 is the less-over-strict alternative (FF 30%, but MCC 0.60, FP 8.8%).

## Reference verdicts
`gemini3.v1.gold_verdicts.jsonl` — per-gold-cell verdicts behind the baseline.

## Reproduce / verify
```bash
uv run --no-project --with httpx python eduLLM-Evals/api_judge_pilot/score_gold.py \
  --gold eduLLM-Evals/api_judge_pilot/gold/tutoreval_core/gold_labels.jsonl \
  --sample eduLLM-Evals/api_judge_pilot/gold/tutoreval_core/sample.jsonl \
  --sample-manifest eduLLM-Evals/api_judge_pilot/gold/tutoreval_core/sample_manifest.json \
  --models gemini-group/gemini-3-flash-preview --adapter generic-binary-strict \
  --max-tokens 4096 --replicates 2 --variants whitespace politeness \
  --out-dir eduLLM-Evals/api_judge_pilot/gold/results_verify
```

## Caveats
100 cells, AI-assisted single-annotator gold -> directional. Config is benchmark-specific;
biggen also = gemini-3 + v1, but tutorbench differs (haiku/sonnet + v1). Keep the robust
regex parser everywhere.
