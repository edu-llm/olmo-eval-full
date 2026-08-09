# FINALIZED biggen judge: gemini-3-flash-preview + generic-binary-strict (v1) — 2026-08-09

Selected on the biggen 100-cell human gold set (`gold/biggen_core/`). Lowest false-pass
(the HANDOFF headline) + highest MCC + cheapest + 0 unscorable. Regression baseline; frozen
prompt text in `prompt.txt`.

## Config
- Model: `gemini-group/gemini-3-flash-preview` (TrueFoundry gateway).
- Prompt: `generic-binary-strict` (v1, evidence-first; verbatim in `prompt.txt`).
- Decoding: temperature 0, `response_format={"type":"json_object"}`, **max_tokens 4096**
  (gemini-3 is verbose; >=4096 gives 0 unscorable on biggen).
- Parser: regex verdict extraction with `json.loads` fallback, fail-closed (REQUIRED).
- Auto-fail policy: generation error / empty / missing -> fail without a judge call.
- Bulk-run concurrency: ~256 (gateway soft-throttles past that).

## Baseline metrics (biggen gold, 100 cells; see metrics.json)
false-pass 0.0% (weighted 0.0%) | macro-F1 92.8 | MCC 0.866 | accuracy 93.0 |
false-fail 15.6% | balanced 90.2 | test-retest flip 4.0% | prompt-flip 6.0% | unscorable 0.
Frozen Qwen baseline on the same cells: FP 12.7%, macro-F1 88.9, MCC 0.780 -> the API judge
beats Qwen.

Regression = false-pass > 5%, macro-F1 < 0.90, MCC < 0.83, test-retest flip > 8%, or
unscorable > 2.

Tradeoff accepted: gemini-3 over-fails good responses (false-fail 15.6%). If balanced errors
matter more later, gpt-4.1 + v1 (macro-F1 92.9, FP 5.5%, false-fail 8.9%) is the alternative.

## Reference verdicts
`gemini3.v1.gold_verdicts.jsonl` — per-gold-cell verdicts behind the baseline.

## Reproduce / verify
```bash
uv run --no-project --with httpx python eduLLM-Evals/api_judge_pilot/score_gold.py \
  --gold eduLLM-Evals/api_judge_pilot/gold/biggen_core/gold_labels.jsonl \
  --sample eduLLM-Evals/api_judge_pilot/gold/biggen_core/sample.jsonl \
  --sample-manifest eduLLM-Evals/api_judge_pilot/gold/biggen_core/sample_manifest.json \
  --models gemini-group/gemini-3-flash-preview --adapter generic-binary-strict \
  --replicates 2 --variants whitespace politeness \
  --out-dir eduLLM-Evals/api_judge_pilot/gold/results_verify
```

## Caveats
100 cells, AI-assisted single-annotator gold -> directional. This config is biggen-specific;
tutorbench's best was gpt-4.1 + v4. Re-select for tutoreval on its gold set. Keep the robust
regex parser everywhere.
