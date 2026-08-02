# Cost per inference vs. model size

Measured single-inference latency for the full200 candidate models, converted to
dollars, split by task type. **MCQ** and **open-ended** are reported separately;
open-ended is further split into **3 input-token buckets**.

## Cost basis
- Hardware: **g6.xlarge** (1× NVIDIA **L4**), on-demand us-east-1 = **$0.8048/hr**
  (the GPU the full200 inference actually ran on).
- `cost_per_inference_usd = Latency(s) × 0.8048 / 3600`.
- `cost_per_1k_usd` = the same × 1000 (easier to read).
- Raw `mean_latency_s` / `median_latency_s` are included so cost can be rescaled to
  any other GPU price without re-measuring.
- Latency is per-request wall time under batched vLLM (temperature 0). It reflects
  effective throughput cost, not isolated single-stream latency.

## Data sources
- **Open-ended** (real, measured): `full200_results/Outputs/open/*/*.responses.jsonl`
  — 52 models × 6 open benchmarks (biggen, bridge, infobench, tutorbench, tutoreval,
  wildbench), 206,307 inferences with per-item `Prompt Tokens`, `Output Tokens`,
  `Latency (s)`. All 6 benchmarks are **pooled**; rows are grouped per (model, bucket).
- **MCQ** (real, measured): `eduLLM-Evals/mcq_runs/run_20260724_211626_mcq/responses.jsonl`
  — the purely-MCQ generative sweep. Not all models finished; the **24 that completed**
  are used (benchmark `cdpk_main`, 25 items each) with per-item `latency_sec`.
  NOTE: the full200 MCQ (pedagogy/piqa/socialiqa) is loglikelihood-scored and records
  **no latency**, so it cannot provide cost-per-inference — hence this separate run.

## Input-token buckets (open-ended)
Chosen at ~tertiles of the observed prompt-token distribution (p33≈117, p66≈530):
- **Short**: `< 128` input tokens  (75,484 inferences)
- **Medium**: `128–511` input tokens  (58,592 inferences)
- **Long**: `>= 512` input tokens  (72,231 inferences)

## Files
- `data/mcq_cost_per_inference.csv` — per model: params_b, n_items, mean/median latency, cost.
- `data/open_ended_cost_per_inference_by_token_bucket.csv` — per (model, bucket): params_b,
  n, mean prompt/output tokens, mean/median latency, cost.
- `graphs/mcq_cost_vs_params.png`
- `graphs/open_cost_vs_params_by_bucket.png`

## Headline numbers (mean across models, cost per 1,000 inferences)
| Task | Cost / 1k inferences |
|---|---|
| MCQ (cdpk_main, 24 models) | ~$0.009 |
| Open-ended — Short input | ~$0.20 |
| Open-ended — Medium input | ~$0.21 |
| Open-ended — Long input | ~$0.23 |

## Interpretation caveats
- **Open-ended cost is driven by OUTPUT tokens, not input length**: the 3 input
  buckets differ by only ~15% (Short→Long), because decode dominates latency. Scatter
  within a size is large because different models generate very different output lengths.
- **MCQ is ~20–25× cheaper** than open-ended: the generative MCQ answer is a few tokens,
  so there is almost no decode cost.
- MCQ and open-ended use different item sets / model sets and are **not** directly
  comparable beyond order-of-magnitude.
