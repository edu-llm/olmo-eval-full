# Availability & quality of response data; parameter distribution

Evidence for the *Availability and Quality of Response Data* and *Parameter Distribution*
sections of `Outline.md`.

## Response-data sources present locally

| Source | Benchmarks | #models | Per-item detail | Notes |
|---|---|---:|---|---|
| **OpenLM** (Open LLM Leaderboard v2 dumps) | bbh, gpqa, math, ifeval, musr | **1,102** | question, response text, predicted, gold, result, scoring_method | 0.213–7.0B; ~28 GB. `data/openlm_download_status.csv` (ok=1,101 / no_samples=84 / err=3). |
| **ATLAS** | ARC (full response matrix), + IRT item banks for gsm8k/hellaswag/truthfulqa/winogrande/math/ifeval | 3,747 train / 417 test (ARC) | 0/1 response matrix + published 3PL params | Full response matrix present **only for ARC**; other benches ship item params + accuracy summaries. |
| **Local full200** (in-house inference) | MCQ: pedagogy, piqa, socialiqa; Open: biggen, bridge, infobench, tutorbench, tutoreval, wildbench | 52 completed (of 156 roster) | MCQ = loglikelihood (no latency); Open = responses.jsonl with tokens + latency | Controlled/reproducible; the only source with per-item latency. |
| **Local LLM-Judge MCQ** | arc_challenge, arc_easy, openbookqa, sciq (+ small pilots) | 62–63 | loglikelihood correctness | The in-house calibration pool for §1–3 of `../01_MCQ_ATLAS`. |
| **OpenHelm** | — | — | — | **Not present locally** (cited in outline only). |

### Quality / coverage observations (for the outline's discussion)

- **Prompting/scoring variability.** OpenLM mixes scoring methods across benchmarks
  (`acc_norm` for bbh/gpqa/musr, `exact_match` for math, `ifeval_prompt_strict`) and keeps
  the raw generated text; the in-house MCQ dumps strip to a loglikelihood choice with no
  latency. Any cross-source pooling has to reconcile these.
- **Benchmarks are slow to reach OpenLM.** The two datasets are almost complementary:
  OpenLM has the *newer* benches (gpqa, musr, math, ifeval, bbh) but **none** of ATLAS's 5
  core MCQ benches (arc, hellaswag, gsm8k, truthfulqa, winogrande). A brand-new,
  skill-specific benchmark (e.g. Pedagogy) has **no** public response data at all — which
  is exactly why `../01_MCQ_ATLAS §5` had to generate its own response matrix.
- **Held-out realism.** ATLAS's local held-out ARC set (417 models) spans well beyond 7B
  (Mixtral, 13B, …), whereas the EDU-LLM regime is 0.2–7B; the range-restricted
  recalibration in `../01_MCQ_ATLAS §1` is the honest apples-to-apples comparison.

## Parameter distribution

- Model rosters live in `AdaptiveTesting/Inputs/Models/`. The main curated roster,
  `models_200.yaml`, currently holds **156** checkpoints spanning **0.218–7.0B** (mean ≈
  3.6B), binned roughly: 0.2–1B: 18, 1–2B: 42, 2–3B: 13, 3–4B: 22, 4–5B: 9, 6–7B+: 52.
  (`data/models_200_stats.txt` still reports the earlier 200-target counts — stale vs the
  156-entry YAML; noted in `../GAPS.md`.)
- Distribution figure: `figures/models_200_param_dist.png`.
- **Calibration-pool skew (important caveat).** The ATLAS 0.5–7B recalibration set
  (`cal_models_0p5_7b.csv`, 1,686 models) is ~95% 7B — very poor mid-range coverage. This
  size-skew is part of why the range-restricted recalibration in `../01_MCQ_ATLAS`
  under-performs the wide-range published bank; a genuinely *uniform* 0.5–7B calibration
  pool would be a better (and currently missing) comparison.

## Files
- `data/openlm_download_status.csv` — per-model OpenLM download status + params_b + average.
- `data/models_200_stats.txt` — roster stats.
- `figures/models_200_param_dist.png` — parameter histogram.
