# ATLAS published 3PL → held-out transfer

Uses published ATLAS ARC item parameters (`AdaptiveTesting/Inputs/ATLAS/arc/`)
mapped to Open LLM Leaderboard `harness_arc_challenge_25` question order, then
runs p-IRT CAT diagnostics on models we scored that ATLAS never saw.

Note: ATLAS labels are 25-shot; our MCQ CSVs are 0-shot log-likelihood.

## Results

| SE stop | Figure | Approx. r |
|---|---|---|
| 0.2 | `figures/atlas_arc_heldout_se0.2.png` | 0.83 |
| 0.3 | `figures/atlas_arc_heldout_se0.3.png` | 0.74 |

## Reproduce

```bash
ATLAS_SE_STOP=0.2 uv run python \
  AdaptiveTesting/Experiments/atlas_transfer_published/atlas_diagnostic_validation.py
```
