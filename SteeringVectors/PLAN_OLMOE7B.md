# TracingLLM on OLMoE-7B — 3-checkpoint full protocol

Paper: [TracingLLM (arXiv 2402.19465)](https://arxiv.org/abs/2402.19465).  
Model: **OLMoE-7B** (MoE), one pre-training run, **exactly three checkpoints**.

## Checkpoint roles (3 total)

| Role | Purpose | Selection rule |
|------|---------|----------------|
| **early** | Probing + steering-vector source | First scheduled save after warmup (`fixed_steps[0]`, or manifest override) |
| **chinchilla** | Probing + steering-vector source | Step whose tokens seen ≈ **20 × non-embedding params** (Chinchilla-optimal) |
| **final** | Probing + **evaluation target** for all steered runs | Last save / `max_duration` step |

Steering builds **two** vectors (from **early** and **chinchilla**) and applies each to the **final** model — same cross-checkpoint logic as the paper’s pretrain→SFT design, with final substituting for SFT.

Tokens at step *s* = `s × global_batch_size` (from final checkpoint `config.json`).

## What this job runs (one platform submission)

Single entry point: `SteeringVectors/run_tracingllm_olmoe7b.py`

| Phase | Paper section | Scope |
|-------|---------------|--------|
| **A. Linear probing** | §2 | 3 ckpts × 5 dims × all layers (4:1 split) |
| **B. HSIC / MI** | §4 (lite) | 3 ckpts × 5 dims × middle layers |
| **C. Steering sweep** | §3 | 2 sources × 5 dims × layer/α grid on **final** |
| **D. Trustworthiness eval** | §3.3 | All 5 benchmarks on final (baseline + steered) |
| **E. General eval** | §3.3 tables | ARC, MMLU (subset), MathQA, RACE on final |
| **F. Figures** | Figs 1, 3, 5 | `plot_tracingllm_figures.py` → PNG under output prefix |

## Deviations from Amber/LLM360 (documented)

| Paper | This run |
|-------|----------|
| 360 checkpoints | **3** (early / chinchilla / final) |
| AmberChat SFT target | **final** pretrain checkpoint |
| TruthfulQA GPT-judge | **MC1/MC2** log-likelihood (no OpenAI key) |
| Full MMLU | **8-subject subset** (cost cap) |
| sklearn probes | torch LBFGS logistic regression |

## Prerequisites

1. **Three OLMoE-7B OLMo-core checkpoints** staged under  
   `s3://sbsandbox-intern-edullm-outputs/teams/eval-inference/runs/olmoe7b-staged/{early,chinchilla,final}/`  
   (GPU role cannot read `edullm-checkpoints` directly).
2. Fill `SteeringVectors/checkpoints_olmoe7b.yaml` with URIs (or rely on auto step pick after staging).
3. Copy `SteeringVectors/platform-run-olmoe7b.yaml` → `.edullm/run.yaml` on branch `edullm/steering-olmoe7b`, push, `edullm submit`.

## Compute

- **Profile:** `olmo-eval-sweep`, `gpu-8xl40s` (7B MoE + conversion; 48 GB may be tight on 1×L40S).
- **Order of magnitude:** ~10¹–10² GPU-hours depending on caps in manifest; tune `max_statements`, `eval_limit`, `mmlu_subjects`.

## Outputs

```
outputs/<run_name>/
  probe_accuracy.json
  mi_hsic.json
  steering_metrics.json
  general_metrics.json
  figures/
    fig1_probe_dynamics.png
    fig3_alpha_ppl_toxic.png
    fig5_capabilities.png
```

Uploaded to `$EDULLM_OUTPUT_PREFIX/<run_name>/` when `--results-s3` is set.
