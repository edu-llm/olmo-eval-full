# 200-model roster (0.2–7B)

Builds a size-balanced, org-diverse Hugging Face roster for AdaptiveTesting sweeps.

## Artifacts

- `build_models_200.py` — selector
- `figures/models_200_param_dist.png` — parameter histogram
- `results/models_200.yaml` — snapshot of the roster
- `results/models_200_stats.txt` — bin / org / family counts

Canonical runtime path for inference jobs: `AdaptiveTesting/Inputs/Models/models_200.yaml`.

## Reproduce

```bash
uv run python AdaptiveTesting/Experiments/models_200/build_models_200.py
```
