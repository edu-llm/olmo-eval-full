# 200-model roster (0.2–7B)

Builds a size-balanced, org-diverse Hugging Face roster for AdaptiveTesting sweeps.

## Artifacts

- `curate_models_200.py` — **current** selector: English-first, vLLM-clean,
  no mirrors/merges/quantisations, one checkpoint per (family, size, tuning)
- `build_models_200.py` — original size/org-balanced selector, kept for
  provenance (its roster leaned on language-targeted checkpoints to fill bins)
- `figures/models_200_param_dist.png` — parameter histogram
- `results/models_200.yaml` — snapshot of the earlier roster
- `results/models_200_stats.txt` — bin / org / family counts

Canonical runtime path for inference jobs: `AdaptiveTesting/Inputs/Models/models_200.yaml`.
The curator also emits a matching generate-mode manifest for TutorBench
response generation at `eduLLM-Evals/models_200.yaml`.

## Reproduce

```bash
uv run python AdaptiveTesting/Experiments/models_200/curate_models_200.py
```

`--dry-run` reports the selection without writing. `--allow-vllm-broken`
re-admits the families that fail under vLLM — valid for the CPU sweep, which
runs `--backend hf` — and grows the roster from 159 to 197 by restoring the
low-ability floor (Pythia, OPT, GPT-Neo, TinyLlama, phi-2, MPT, InternLM,
classic Falcon, Cerebras, RedPajama).
