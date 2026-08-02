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
- `graphs/individual_latency_by_param_range.png` — the featured figure (see below).

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

## Individual (single-stream) latency by parameter range — the outline's tutor-cost table

`data/individual_latency_by_param_range.csv` + `graphs/individual_latency_by_param_range.png`
(script `../scripts/param_range_latency.py`). For each band we report the measured mean
prompt/output tokens, the measured **amortized (batched)** per-request latency, and an
estimated **individual (single-stream)** latency from a memory-bandwidth decode roofline
(L4: 300 GB/s, 60% efficiency, 2 bytes/param).

| Parameter range | Individual latency (s) | Amortized/batched (s) | batch speed-up | mean out tok |
|---|---:|---:|---:|---:|
| 0–1B | **4.0** | 0.25 | ~16× | 878 |
| 1–2B | **10.3** | 0.74 | ~14× | 648 |
| 2–5B | **19.7** | 1.13 | ~17× | 658 |
| 5–7B | **34.7** | 1.64 | ~21× | 489 |

Finer split (`band_set=fine` in the CSV): 2–3B = 18.2 s, 3–5B = 20.2 s.

**These individual-latency numbers (≈4 / 10 / 20 / 35 s) are the values in the outline's
"Inference Cost For Tutor Models" table.** Two clarifications for the paper:

1. **Individual vs. batched.** In production the full200 inference ran batched under vLLM,
   so the *measured* per-request wall time is only ~0.25–1.6 s (right column). The 4–35 s
   figures are the *single-stream* latency you would see issuing one request at a time; the
   ~14–21× gap is the batching throughput win. Be explicit in the paper about which one a
   given claim uses (checkpoint-time CAT with a warm batched server → use the batched
   numbers; a cold single-request diagnostic → use the individual numbers).
2. **Hardware label.** The measurements and roofline here are for the **L4** GPU the
   full200 sweep actually used. The outline table is captioned "L40S"; an L40S (≈864 GB/s)
   would be ~2–3× faster on decode. Either re-label the table **L4**, or re-measure on an
   L40S — see `../GAPS.md`.

