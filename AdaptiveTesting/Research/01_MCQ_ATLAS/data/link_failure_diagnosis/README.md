# Link-failure diagnosis: the apparent gpqa/musr failure was a bad-item (a≤0) artifact; bbh is the true weak case

**Headline.** The near-zero p-IRT correlation on musr (and the collapse you get on
gpqa if you skip the filter) was **not** a property of the benchmark — it was
caused by leaving non-positive-discrimination items (`a ≤ 0`, mirt fit
failures / polarity flips) in the 3PL bank. These banks are heavily polluted
(gpqa 51%, musr 43%, bbh 31% of fitted items have `a ≤ 0`). Dropping them — the
canonical ATLAS loader behaviour, already implemented in
`../../../scripts/atlas_error_plots_openlm.py` — restores the link: musr's CAT
correlation jumps from **r = −0.036 → 0.746** and gpqa holds at **r = 0.737**.
**bbh is the genuinely weak benchmark (r ≈ 0.67, worst MAE), and for a completely
different reason: early-stopping under-sampling of a heterogeneous suite, not bad
items.**

Inputs are the exact frozen banks + held-out matrices the replication used
(`AdaptiveTesting/Experiments/openlm_atlas_3pl/<bench>/…` and
`…/openlm_gpqa_atlas_3pl/…`). Script: `../../../scripts/link_failure_diagnosis.py`.

## 1. The item-quality mechanism

`r_theta_score` below is the Pearson r between the **full-bank EAP θ** and the
**true full-benchmark accuracy** (mean over every usable item), computed twice:
once with the polluted bank (`a ≤ 0` kept) and once filtered (`a > 0`). The target
is held fixed, so only the estimation bank changes — this isolates the effect of
the bad items.

| bench | % a≤0 | % 0<a<0.2 | a median (unfilt → filt) | **θ-r unfilt → filt** | **CAT r pre→post filter** |
|---|---:|---:|---|---:|---:|
| math   |  1.9% | 1.2% | 2.17 → 2.21 | 0.83 → 0.84 | 0.878 → 0.878 |
| ifeval |  4.5% | 0.6% | 2.67 → 2.79 | 0.86 → 0.85 | 0.921 → 0.921 |
| bbh    | 31.2% | 2.4% | 1.60 → 3.99 | 0.69 → 0.86 | 0.753 → 0.674 |
| musr   | 42.7% | 2.5% | 0.47 → 1.39 | 0.49 → 0.62 | **−0.036 → 0.746** |
| gpqa   | 51.4% | 3.3% | **−0.11** → 1.44 | **−0.07 → 0.32** | 0.737 → 0.737 |

Reading:

- **musr** is the clean before/after case in the pipeline itself: the pre-filter
  CAT scored **r = −0.036** (predictions barely varied — corrupted θ), the
  post-filter CAT scores **r = 0.746**. Same benchmark, same models; the only
  change is dropping 43% junk items.
- **gpqa** already used the canonical `a > 0` loader, so its published CAT r never
  changed (0.737). But its bank is the *most* polluted: over half the fitted items
  have `a ≤ 0` (unfiltered a-median is literally **negative, −0.11**). The
  counterfactual confirms the dependence — full-bank θ recovery **flips sign**
  from −0.07 (polluted) to +0.32 (filtered). Without the filter, gpqa would look
  exactly like the old musr "failure."
- **math / ifeval** have almost no bad items (2–5%), so filtering is a no-op and
  they link strongly throughout. This is the control that shows the effect is
  item quality, not benchmark identity.
- Panel (C) of `filter_mechanism.png` shows the raw cause directly: the gpqa and
  musr discrimination distributions have a large mass of items at `a < 0` (left of
  the red line) — polarity-flipped/failed mirt fits that make "more able" models
  look *less* able on those items, scrambling θ.

**Conclusion for gpqa/musr:** neither is intrinsically uninformative. Both carry a
real, positive latent-ability signal once the fit-failure items are removed; the
"link failure" was a bank-hygiene bug.

Figure: `filter_mechanism.png`
- (A) θ-recovery r, unfiltered vs filtered (same target) — gpqa sign-flips, musr/bbh gain.
- (B) bank pollution: % `a ≤ 0` (solid) and % `0<a<0.2` (hatched) per bench.
- (C) unfiltered discrimination distributions for gpqa/musr (large `a ≤ 0` mass).
- (D) gpqa θ-vs-score scatter, unfiltered (grey, r=−0.07) vs filtered (red, r=0.32).

## 2. Why bbh stays weak (r ≈ 0.67) even after filtering

bbh's problem is **not** bad items (only 31% `a ≤ 0`, and filtering *helps* its
full-bank θ recovery: 0.69 → 0.86) and **not** low score variance (bbh has a
*healthy* per-model score SD of 0.117 on scored items — the widest of the MCQ-style
set). The failure is a **CAT early-stopping / under-sampling** effect:

- bbh's filtered discriminations are **inflated**: a-median **3.99**, the highest
  of any benchmark (vs ~1.4 for gpqa/musr, ~2.2 for math). High `a` items collapse
  the posterior SE very fast, so the CAT hits the SE≤0.3 target at the **8-item
  minimum** (`cat_mean_items = 8.2`, fewest of all benches).
- Those ~8 items badly under-sample a **heterogeneous, effectively
  multidimensional** suite (BBH bundles ~27 disparate subtasks). One unidimensional
  θ fit from 8 items does not generalise: θ recovery error stays high
  (`theta_mae = 0.68`, the worst) and *does not improve* even at SE≤0.1.
- The result is **over-dispersed, high-error predictions**: p-IRT prediction SD is
  **0.186 vs actual SD 0.116** (predictions swing wider than reality; slope of
  pred-on-actual = 1.08), and bbh's accuracy MAE is **0.114 — 2–5× every other
  benchmark**. See panel (3) of `bbh_diagnosis.png`.
- The tell-tale contrast: bbh's **full-bank θ recovers accuracy at r = 0.86**, but
  the **early-stopped CAT plateaus at r = 0.67** — the single biggest full-bank→CAT
  drop of any benchmark (panel (2)). bbh has the signal; the SE stopping rule is
  overconfident (inflated `a`) and quits before the CAT samples enough of the suite.

**One-line reason bbh is weak:** its inflated/heterogeneous discriminations make the
SE criterion overconfident, so the CAT stops at the ~8-item minimum and under-samples
a multidimensional benchmark — the full 3965-item bank recovers accuracy at r≈0.86,
but the early-stopped CAT is stuck at r≈0.67 with the highest MAE.

Figure: `bbh_diagnosis.png`
- (1) score spread by bench (bbh spread is healthy → variance is not the issue).
- (2) full-bank θ-r vs CAT-r with mean #items (bbh's big drop, 8-item stop).
- (3) CAT prediction vs actual (bbh predictions over-dispersed, highest MAE).
- (4) item p-value + a-median (bbh a-median ~4 → overconfident SE).

## Files

- `diagnosis_summary.csv` — one row per benchmark (math, bbh, gpqa, musr, ifeval):
  `%a≤0`, `%0<a<0.2`, a-median filtered/unfiltered, θ-score r filtered/unfiltered,
  pre/post-filter CAT r, score mean/SD/range, item p-value stats, and the bbh CAT
  diagnostics (mean items, MAE, prediction SD, θ-MAE).
- `filter_mechanism.png` — the a≤0 mechanism (section 1).
- `bbh_diagnosis.png` — the bbh early-stop diagnosis (section 2).

## Reproduce

```bash
export REPO=/Users/arhant/Documents/EDLM/olmo-eval-full
export PYTHONPATH=$REPO/eduLLM-Evals
uv run python $REPO/AdaptiveTesting/Research/scripts/link_failure_diagnosis.py
```

## Notes / honest caveats

- The replication's headline CAT r (e.g. gpqa 0.737, musr 0.746) is measured against
  the *scored (a>0) item subset*, matching `atlas_error_plots_openlm.py`. The
  `θ-r filtered` column here uses the stricter *full-benchmark* accuracy target, so
  its gpqa value is lower (0.32): once you demand recovery of accuracy over the whole
  item pool, gpqa's near-4-choice-chance compression (score SD 0.021 full-bank, mean
  only ~0.05 above the 0.25 chance line) limits the ceiling. That is a precision
  limit, **not** the a≤0 artifact and **not** a claim that gpqa is uninformative —
  the sign flip (−0.07 → +0.32) is the artifact's signature.
- `cat_r_prefilter_old` is read from `../openlm_atlas3pl_results_summary.csv`
  (the pre-corrected run); `cat_r_postfilter_new` from
  `../atlas_replication/summary_pirt_mae_sd_se.csv` (SE≤0.3).
