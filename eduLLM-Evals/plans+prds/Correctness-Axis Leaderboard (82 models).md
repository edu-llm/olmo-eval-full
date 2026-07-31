# Correctness-Axis Leaderboard (82 tutor models)

**Status:** deliverable — a usable ranking of the 82 pilot tutor models on the
**correctness** latent axis, produced by *scoring* the frozen 2-skill M2PL bank.
No re-fit, no parameter changes: this consumes the frozen calibration only.

Provenance: frozen 2-skill calibrated bank at **ridge = 1e-2, floor = 15**
(`data/TutorBench/rubrics_qmatrix_calibrated_2skill.jsonl`, 3443 calibrated items,
60 `exclude_from_fit` flags honoured), response matrix
`staging/response_matrix_full_nonopt.csv` (82 models), CAT policy defaults
(point-biserial ≥ 0.05, per-skill floor 15, content balancing, exposure top-5,
seed 20260730). Same bank + matrix + policy as the
[Pilot Baseline Freeze](Pilot%20Baseline%20Freeze%20(2%20and%203%20skill).md)
definitive 2-skill run.

## What the axis means

**Correctness** is the collapsed **content + diagnosis** latent trait: does the
tutor say true, on-topic, mathematically/factually correct things and correctly
diagnose the student? In the 2-skill instrument the fit's `a_content` and
`a_diagnosis` loadings are collapsed into a single `a_correctness`; the second
axis (`a_scaffolding`) is modelled separately and reported here only as a
secondary reference column. Correctness is our **strong, trustworthy axis**:
honest out-of-sample (k-fold, fold-trained params) recovery is **r ≈ 0.93**,
versus scaffolding's ~0.67. The headline ranking is correctness only.

Higher θ (in logits) = more able. θ is on the standard-normal latent scale
(prior N(0, 1)), so θ = 0 is the pilot-fleet centre and z ≈ θ.

## How θ was estimated

Two abilities per model, both from the frozen bank, both reusing the harness math
(`scripts/cat_eval_tutorbench_multiskill.py`, `tutor_cat/mirt.py`):

1. **Full-response θ (`theta_correctness_full`)** — the reference / sort key. EAP
   (posterior mean) over **all** of a model's observed rubric responses under the
   frozen item parameters (`eap_full`). Computed on a **fine** Gauss–Hermite grid
   (41 nodes/dim) so the reference is a smooth, continuous ability. (The pilot
   in-run EAP uses only 7 nodes/dim, which quantises θ into ~7 tiers; that coarse
   value is retained as `theta_correctness_full_grid7` for provenance. Grid size
   is a numerical-integration choice, not a parameter change.)
2. **CAT θ (`theta_correctness_cat`)** — the adaptive-test estimate. The
   operational CAT (frozen params, policy defaults) administers items adaptively
   and Newton-updates θ; we record the final correctness θ, its posterior SE
   (`cat_se_correctness`), and the number of items administered.

`items_administered` is the **full 2-skill test length** (correctness **and**
scaffolding). It is long (mean 47, median 40) because the scaffolding floor of 15
keeps the test running on a scarce skill; **correctness alone** reaches SE < 0.3
much sooner (median ≈ 24 items, 81/82 models). If you only care about the
correctness score, the correctness axis is essentially settled by ~2 dozen items.

## How to read the table / figures

- **CSV:** `reports/leaderboard_correctness_2skill/leaderboard_correctness_2skill.csv`
  — one row per model, sorted by `theta_correctness_full` (desc), ties broken by
  CAT θ. Columns: `rank`, `model_name`, `theta_correctness_full` (sort key),
  `theta_correctness_cat`, `cat_se_correctness`, `items_administered`,
  `cat_items_loading_correctness` (# administered items that load correctness),
  `theta_scaffolding_full` (secondary reference), `z_correctness`,
  `percentile_correctness`, `correctness_converged`, and
  `theta_correctness_full_grid7` (pilot-grid provenance value).
- **Caterpillar** (`figures/leaderboard_correctness_caterpillar.png`): every model
  ranked on the correctness axis; blue diamond = full-response θ (sort key),
  orange dot + whiskers = CAT θ ± 1.96·SE. Read top-to-bottom for the ranking and
  the whisker width for CAT measurement precision.
- **Recovery scatter** (`figures/cat_vs_full_correctness_recovery.png`): CAT θ vs
  full-response θ with y = x; shows the adaptive test reproduces the full-response
  ranking. Mild tail shrinkage (extreme models pulled toward 0) is the expected
  effect of the N(0, 1) prior on a short adaptive test.

## Top 10 / bottom 5 (by full-response correctness θ)

| rank | model | θ_full | θ_CAT | CAT SE | items |
|---:|---|---:|---:|---:|---:|
| 1 | Qwen/Qwen3-4B | 3.46 | 2.89 | 0.18 | 17 |
| 2 | Qwen/Qwen3-1.7B | 2.45 | 2.43 | 0.18 | 55 |
| 3 | Qwen/Qwen2.5-7B-Instruct | 2.45 | 1.86 | 0.18 | 19 |
| 4 | tiiuae/Falcon3-7B-Base | 2.45 | 2.07 | 0.19 | 17 |
| 5 | Qwen/Qwen2.5-3B-Instruct | 1.96 | 2.13 | 0.22 | 17 |
| 6 | Qwen/Qwen2.5-Math-7B | 1.96 | 1.93 | 0.20 | 85 |
| 7 | Qwen/Qwen2.5-7B | 1.96 | 2.01 | 0.23 | 17 |
| 8 | Qwen/Qwen2.5-Coder-7B | 1.96 | 1.95 | 0.19 | 46 |
| 9 | Qwen/Qwen2-7B | 1.47 | 1.07 | 0.24 | 17 |
| 10 | Qwen/Qwen2.5-3B | 1.47 | 1.23 | 0.23 | 17 |
| … | … | … | … | … | … |
| 78 | bigscience/bloomz-1b7 | -5.49 | -2.12 | 0.29 | 27 |
| 79 | bigscience/bloom-560m | -5.64 | -1.41 | 0.30 | 31 |
| 80 | EleutherAI/pythia-160m | -5.65 | -2.63 | 0.29 | 37 |
| 81 | openai-community/gpt2-medium | -5.74 | -2.00 | 0.30 | 35 |
| 82 | EleutherAI/pythia-70m | -5.86 | -2.16 | 0.30 | 35 |

The ranking is face-valid: the Qwen2.5/Qwen3 family and Falcon3-7B top the axis;
small/old base models (pythia-70m/160m, gpt2, bloom-560m) sit at the bottom.

## CAT-vs-full recovery (does the adaptive test recover the ranking?)

- **In-sample, this leaderboard** (fine-grid reference): Pearson **r = 0.934**,
  **Spearman ρ = 0.953** (n = 82). The adaptive test reproduces both the score
  and the rank order of the full-response reference.
- **Pilot canonical CAT recovery** (grid-7, as cited in the freeze memo): in-sample
  correctness **r = 0.964**.
- **Honest out-of-sample** (k-fold, fold-trained params vs full-bank ability):
  correctness **r ≈ 0.929** — the number to quote for generalisation, since the
  in-sample figures reuse the same responses the bank was fit on.

## Caveats

- **In-sample vs OOS.** In-sample recovery (0.93–0.96) is optimistic — the bank
  was fit on these 82 models' responses. The honest generalisation number is the
  k-fold OOS **r ≈ 0.93** (correctness). Cite OOS, not in-sample, for claims about
  new models.
- **N = 82, pilot.** This is the pilot calibration, not the final 200-run bank.
  Absolute θ values are on the pilot latent scale; treat this as a working
  ranking, not a locked benchmark.
- **Tail shrinkage.** CAT θ for very weak models is pulled toward ~-2 by the
  prior + short test, so `theta_correctness_full` (the reference) separates the
  bottom of the fleet better than CAT θ does. Rank order is preserved
  (ρ = 0.95); absolute CAT θ in the tails is regularised.
- **Scaffolding is not the focus.** `theta_scaffolding_full` is included only for
  reference; its OOS recovery (~0.67) is much weaker than correctness, so do not
  rank on it.
- **Frozen bank equivalence.** The harness consumes
  `staging/calibration_mirt_full2skill.csv`, which carries the identical fit as
  the frozen JSONL bank (`b` bit-identical; `a` differs only where the JSONL
  clamps negative loadings to 0 — on the correctness axis just 23 items, all
  |Δa| ≤ 0.17, all removed anyway by the point-biserial filter). So the
  correctness leaderboard is faithful to the frozen bank.

## Reproduce

From `eduLLM-Evals/` (Windows PowerShell, `uv`):

```powershell
# 1) definitive 2-skill CAT run on the frozen bank + matrix, policy defaults
uv run python scripts/cat_eval_tutorbench_multiskill.py --skills 2
#    -> reports/cat_eval_policy/2skill/cat_per_model.csv (+ metrics, figures)

# 2) build the correctness-axis leaderboard + figures from that per-model output
uv run python scripts/build_correctness_leaderboard.py
#    -> reports/leaderboard_correctness_2skill/leaderboard_correctness_2skill.csv
#    -> reports/leaderboard_correctness_2skill/figures/{leaderboard_correctness_caterpillar,cat_vs_full_correctness_recovery}.png
```
