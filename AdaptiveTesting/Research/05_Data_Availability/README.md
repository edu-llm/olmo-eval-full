# Response-data availability and parameter distribution

Evidence for the response-data and parameter-distribution sections of `Outline.md`.

## Response-data sources present locally

Five sources were checked. Four are on disk; OpenHelm is not.

| Source | Benchmarks | Models | Per-item detail | Notes |
|---|---|---:|---|---|
| OpenLM (Open LLM Leaderboard v2 dumps) | bbh, gpqa, math, ifeval, musr | 1,102 | question, response text, predicted, gold, result, scoring_method | 0.213 to 7.0B, about 28 GB. `data/openlm_download_status.csv`: ok=1,101, no_samples=84, err=3. |
| ATLAS | ARC (full response matrix), plus IRT item banks for gsm8k, hellaswag, truthfulqa, winogrande, math, ifeval | 3,747 train / 417 test (ARC) | 0/1 response matrix plus published 3PL params | Full response matrix present only for ARC. Other benchmarks ship item params and accuracy summaries. |
| Local full200 (in-house inference) | MCQ: pedagogy, piqa, socialiqa. Open: biggen, bridge, infobench, tutorbench, tutoreval, wildbench | 52 done of 156 roster | MCQ = loglikelihood (no latency). Open = responses.jsonl with tokens and latency | The only source with per-item latency. |
| Local LLM-Judge MCQ | arc_challenge, arc_easy, openbookqa, sciq (plus small pilots) | 62 to 63 | loglikelihood correctness | The in-house calibration pool for sections 1 to 3 of `../01_MCQ_ATLAS`. |
| OpenHelm | (none) | (none) | (none) | Not present locally. Cited in the outline only. |

## Quality and coverage

Scoring methods differ across sources. OpenLM mixes `acc_norm` (bbh, gpqa, musr), `exact_match` (math), and `ifeval_prompt_strict`, and keeps the raw generated text. The in-house MCQ dumps strip each item to a loglikelihood choice with no latency. Any cross-source pooling has to reconcile these first.

The two large sources barely overlap. OpenLM has the newer benchmarks (gpqa, musr, math, ifeval, bbh) but none of the five ATLAS core MCQ benchmarks (arc, hellaswag, gsm8k, truthfulqa, winogrande). A new skill-specific benchmark such as Pedagogy has no public response data at all, which is why `../01_MCQ_ATLAS §5` had to generate its own response matrix.

The held-out sets sit at different scales. The local held-out ARC set (417 models) reaches well past 7B (Mixtral, 13B, and larger), while the EDU-LLM models run 0.2 to 7B. The range-restricted recalibration in `../01_MCQ_ATLAS §1` is the matched comparison.

## Parameter distribution

Rosters live in `AdaptiveTesting/Inputs/Models/`. The main curated roster, `models_200.yaml`, holds 156 checkpoints from 0.218 to 7.0B (mean 3.6B, median 3.0B) across 50 organizations. Size bins: 0.2 to 1B: 18, 1 to 2B: 42, 2 to 3B: 13, 3 to 4B: 22, 4 to 5B: 9, 5 to 6B: 0, 6 to 7B: 52. See `figures/models_200_param_dist.png` (rebuild with `figures/regen_param_dist.py`).

The stale file `data/models_200_stats.txt` still reports the earlier 200-target counts (200 models, mean 2.42B). Treat it as stale against the 156-entry YAML. This is logged in `../GAPS.md` #15.

The ATLAS recalibration set skews large. `cal_models_0p5_7b.csv` has 1,686 models, about 95% of them 7B, so mid-range coverage is thin. That skew is part of why the range-restricted recalibration in `../01_MCQ_ATLAS` trails the wide-range published bank. A uniform 0.5 to 7B calibration pool would be a better comparison and does not yet exist.

## Files
- `data/openlm_download_status.csv`: per-model OpenLM download status, params_b, and average.
- `data/models_200_stats.txt`: roster stats. Stale (200-target counts); use the YAML for live numbers.
- `figures/models_200_param_dist.png`: size histogram and top organizations for the 156-model roster.
- `figures/regen_param_dist.py`: rebuilds the figure from the live `models_200.yaml`.
