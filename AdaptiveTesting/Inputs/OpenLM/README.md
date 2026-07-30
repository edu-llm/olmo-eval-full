# OpenLM — Open LLM Leaderboard per-question responses

Per-question correctness CSVs for Open LLM Leaderboard v2 models in the
**0.2–7B** parameter range (benches: `ifeval`, `bbh`, `math`, `gpqa`, `musr`).

## Layout

```
OpenLM/
  download_openlm_responses.py   # scrape script
  models_selected.csv            # roster used for the scrape
  download_status.csv            # per-model download outcome
  <benchmark>/<org>__<model>.csv # per-question rows (local; gitignored)
```

CSV columns: `question_id, model, benchmark, subtask, question, response,
predicted, gold, result, grade, scoring_method`.

## Regenerate

Requires `HF_TOKEN` (gated `*-details` datasets):

```bash
uv run python AdaptiveTesting/Inputs/OpenLM/download_openlm_responses.py \
  --min-params 0.2 --max-params 7 --benchmarks ifeval,bbh,math,gpqa,musr \
  --workers 16
```

Raw response CSVs are large (~26GB) and are **not** committed; re-download locally
before running calibration experiments under `AdaptiveTesting/Experiments/openlm_*`.
