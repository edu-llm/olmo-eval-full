# OpenLM multi-bench ATLAS 3PL diagnostics

Same pipeline as `openlm_gpqa_atlas_3pl/` for **ifeval**, **math**, **musr**,
and **bbh**: chunked unidimensional 3PL (capped EM), polarity-aware mean–σ
linking, Fisher CAT + p-IRT on a 10% held-out model split (seed=7).

## Summary (SE≤0.3)

| Bench | r | MAE | avg items | notes |
|---|---|---|---|---|
| ifeval | **0.921** | 0.048 | 9.2 | strong |
| math | **0.878** | 0.024 | 43.8 | strong; longer CAT |
| bbh | **0.753** | 0.035 | 8.1 | ok |
| gpqa | **0.737** | 0.070 | 13.0 | see sibling experiment |
| musr | −0.036 | 0.109 | 8.0 | failed (weak linking) |

Full table: `results_summary.csv`. Per-bench plots under `<bench>/figures/`.

## Reproduce

```bash
# Needs AdaptiveTesting/Inputs/OpenLM/<bench>/*.csv locally
uv run python run_3pl_diagnostic.py \
  --benchmarks ifeval,math,musr,bbh --workers 8 --ncycles 500 --se-stop 0.3
```
