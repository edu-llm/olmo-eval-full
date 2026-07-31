# Exclusion Experiment — Relabel-31 (Variant A) vs Exclude-All-60 (Variant B)

**Date:** 2026-07-31
**Status:** exploratory A/B (nothing committed; authoritative bank untouched — all work on copies)
**Question:** For the 60 "reverse-behaving" criteria currently held out of the M2PL fit
(`exclude_from_fit: true`), is it better to **relabel the ~31 non-degenerate ones as
scaffolding-primary and re-enter them** (Variant A), or to **keep all 60 excluded**
(Variant B)?

**Headline: Exclude-all (Variant B) wins. Relabeling does NOT rescue the 31 items — 26 of 31
come out with a *negative* scaffolding loading under Variant A, degrading scaffolding-axis
health for ~zero out-of-sample benefit.** The only items that turn positive are the 3 genuinely
reverse-behaving ones (negative raw point-biserial); the ~28 low-info items have *positive* raw
point-biserial and therefore load *negative* on the anti-correlated scaffolding axis.

---

## Setup (identical for both variants — apples-to-apples)

- **Response matrix:** `staging/response_matrix_full_nonopt.csv` (82 models × 6,180 nonoptional
  criteria) — same slice as Run 5.
- **Fitter / hyperparameters (Run-5 settings):** `scripts/calibrate_mirt.py`
  `--collapse content,diagnosis --estimate-latent-corr --grid 7` (343 nodes 3-dim / 49 nodes
  2-dim), ridge `1e-3`, EM `tol 1e-4`, `max-iter 200`.
- **k-fold:** `scripts/kfold_cv_mirt.py --k 5 --seed 20260729 --grid 7` (latent-corr off = harness
  default, as in Run 5).
- **Banks (copies, never in place):**
  - Variant B → `data/experimental/rubrics_qmatrix_excludeall.jsonl` (byte-identical to curated;
    all 60 `exclude_from_fit: true`).
  - Variant A → `data/experimental/rubrics_qmatrix_relabel31.jsonl` (29 degenerate stay excluded;
    31 non-degenerate re-entered with `content=0, diagnosis=0, scaffolding=1`,
    `primary_skill=scaffolding`).
- **The 3 groups of the 60:** `degenerate` = 29 (kept excluded in BOTH — zero-variance, no
  relabel can help), `low-info-fine` = 30, `judge-inverted` = 1. Variant A re-enters the 31
  (`low-info-fine` ∪ `judge-inverted`). Confirmed: Variant A adds exactly +31 scaffolding-only
  (`001`) Q-rows (340 → 371) and re-enters 31 items (3,443 → 3,474 items fit).

---

## Side-by-side comparison

| metric | Variant B (exclude-all-60) | Variant A (relabel-31) | Δ (A − B) | verdict |
|---|---:|---:|---:|---|
| **items fit (collapsed)** | 3,443 | 3,474 | +31 | set differs (see caveat) |
| collapsed loglik | −60,630.83 | −61,961.37 | −1,330.5 | not comparable¹ |
| collapsed AIC | 135,891.66 | 138,676.74 | +2,785 | not comparable¹ |
| collapsed BIC | 212,920.67 | 216,425.10 | +3,504 | not comparable¹ |
| AIC winner (uni/coll/full) | **collapsed** | unidim | — | flip is a set artifact¹ |
| **latent corr (correctness ↔ scaffolding)** | −0.418 | −0.409 | +0.009 | unchanged (good) |
| **scaffolding items (nonzero a_scaff)** | 768 | 799 | +31 | — |
| **# POSITIVE-loading scaffolding items** | 475 | 483 | +8 | — |
| # NEGATIVE-loading scaffolding items | 293 | 316 | **+23** | **A adds net negatives** |
| median a_scaffolding (all nonzero) | 0.386 | 0.249 | **−0.137** | **A worse** |
| median a_scaffolding (positive only) | 0.917 | 0.974 | +0.057 | ~unchanged |
| **cross-fold stability a_scaffolding** | 0.558 | 0.576 | +0.018 | ~equal² |
| cross-fold stability a_correctness | 0.790 | 0.797 | +0.007 | unchanged |
| cross-fold stability b | 0.848 | 0.852 | +0.004 | unchanged |
| **pooled OOS AUC** | 0.9045 | 0.9031 | −0.0014 | equal |
| pooled OOS accuracy | 0.8930 | 0.8920 | −0.0010 | equal |
| pooled OOS log-loss | 0.2654 | 0.2674 | +0.0020 | equal (A hair worse) |
| pooled OOS Brier | 0.0773 | 0.0780 | +0.0007 | equal |

¹ **AIC/BIC/loglik caveat.** The item SET differs by 31 items, so global fit statistics are **not**
directly comparable across variants — Variant A must "pay" for 31 extra items/params and their
raw likelihood. This is exactly why the brief says to judge on **per-axis health + OOS**, not
global fit. The AIC-winner flip (collapsed → unidim) is a consequence of adding 31 weak
scaffolding-only items, not evidence about the 2-skill structure. Variant B reproduces the Run-5
definitive picture (collapsed wins AIC; latent corr ≈ −0.42; OOS AUC ≈ 0.905).

² **Stability "improvement" is illusory.** Cross-fold correlation measures *consistency* of the
loading across folds, not *correctness of sign*. The 26 relabeled items load **consistently
negative** across folds, so they nudge the scaffolding-stability correlation up (0.558 → 0.576)
while actually making the axis less healthy (median a_scaffolding 0.386 → 0.249, +23 net negative
loaders).

**Correctness axis is essentially untouched:** corr(a_correctness_B, a_correctness_A) = **0.987**
over 3,443 common items; median a_correctness 1.052 (B) vs 1.061 (A). The relabel touched only the
scaffolding side, as intended.

---

## Per-item table — the 31 relabeled items (Variant A fitted a_scaffolding)

`status`: **PASS** = positive a_scaffolding under A (relabel "worked"); **FLAG(negative)** = still
negative under A (relabel did NOT rescue — a genuine problem). `r_pb` = recomputed point-biserial
of the item with (unidimensional) ability from the exclusion plan.

| criterion_id | rule | old_skills | pass | r_pb | a_scaffolding (A) | status |
|---|---|---|---:|---:|---:|---|
| tb_0432_c01 | low-info-fine | diagnosis | 7/82 | −0.094 | **+2.448** | PASS |
| tb_0532_c06 | **judge-inverted** | diagnosis | 50/82 | −0.571 | **+1.196** | PASS |
| tb_0455_c07 | low-info-fine (**must-not guard**) | content\|scaffolding | 43/82 | −0.415 | **+0.279** | PASS |
| tb_0647_c03 | low-info-fine | content\|diagnosis\|scaffolding | 5/82 | +0.488 | +0.259 | PASS |
| tb_0647_c10 | low-info-fine | scaffolding | 9/82 | +0.628 | +0.075 | PASS |
| tb_0185_c08 | low-info-fine | scaffolding | 16/82 | +0.688 | −0.025 | FLAG(negative) |
| tb_0082_c09 | low-info-fine | content\|scaffolding | 2/82 | +0.290 | −0.141 | FLAG(negative) |
| tb_0517_c11 | low-info-fine | content\|diagnosis\|scaffolding | 2/82 | +0.345 | −0.142 | FLAG(negative) |
| tb_0517_c06 | low-info-fine | content\|diagnosis\|scaffolding | 5/82 | +0.578 | −0.147 | FLAG(negative) |
| tb_0531_c11 | low-info-fine | content\|scaffolding | 5/82 | +0.571 | −0.147 | FLAG(negative) |
| tb_0643_c06 | low-info-fine | content\|scaffolding | 5/82 | +0.535 | −0.147 | FLAG(negative) |
| tb_0641_c08 | low-info-fine | scaffolding | 9/82 | +0.112 | −0.295 | FLAG(negative) |
| tb_0593_c05 | low-info-fine | scaffolding | 8/82 | +0.160 | −0.336 | FLAG(negative) |
| tb_0357_c08 | low-info-fine | content\|scaffolding | 10/82 | +0.686 | −0.374 | FLAG(negative) |
| tb_0605_c04 | low-info-fine | content\|diagnosis\|scaffolding | 9/82 | +0.414 | −0.392 | FLAG(negative) |
| tb_0655_c03 | low-info-fine | scaffolding | 9/82 | +0.491 | −0.392 | FLAG(negative) |
| tb_0220_c05 | low-info-fine | content\|diagnosis\|scaffolding | 18/82 | +0.684 | −0.450 | FLAG(negative) |
| tb_0097_c06 | low-info-fine | content\|scaffolding | 18/82 | +0.627 | −0.452 | FLAG(negative) |
| tb_0581_c07 | low-info-fine | content\|scaffolding | 20/82 | +0.542 | −0.484 | FLAG(negative) |
| tb_0587_c09 | low-info-fine | content\|scaffolding | 20/82 | +0.678 | −0.510 | FLAG(negative) |
| tb_0265_c05 | low-info-fine | content\|scaffolding | 13/82 | +0.521 | −0.517 | FLAG(negative) |
| tb_0587_c02 | low-info-fine | content\|diagnosis\|scaffolding | 13/82 | +0.470 | −0.556 | FLAG(negative) |
| tb_0617_c06 | low-info-fine | content\|scaffolding | 17/82 | +0.694 | −0.606 | FLAG(negative) |
| tb_0583_c06 | low-info-fine | content\|scaffolding | 15/82 | +0.355 | −0.645 | FLAG(negative) |
| tb_0595_c04 | low-info-fine | content\|scaffolding | 15/82 | +0.204 | −0.650 | FLAG(negative) |
| tb_0570_c04 | low-info-fine | content\|scaffolding | 13/82 | +0.434 | −0.686 | FLAG(negative) |
| tb_0581_c09 | low-info-fine | content\|scaffolding | 15/82 | +0.391 | −0.698 | FLAG(negative) |
| tb_0546_c02 | low-info-fine | content\|scaffolding | 21/82 | +0.616 | −0.695 | FLAG(negative) |
| tb_0635_c03 | low-info-fine | content\|diagnosis\|scaffolding | 17/82 | +0.411 | −0.768 | FLAG(negative) |
| tb_0598_c05 | low-info-fine | content\|scaffolding | 31/82 | +0.730 | −0.871 | FLAG(negative) |
| tb_0658_c06 | low-info-fine | content\|diagnosis\|scaffolding | 49/82 | +0.590 | −1.481 | FLAG(negative) |

**All 31 new skills = `scaffolding` (content=0, diagnosis=0). Result: 5 PASS, 26 FLAG(negative), 0 dropped.**

### Why the relabel backfires (the key mechanism)

The scaffolding latent axis is **negatively correlated with correctness** (r ≈ −0.41). The `r_pb`
in the plan is the point-biserial with *overall* (correctness-dominated) ability. So:

- An item with **positive r_pb** (models good at correctness pass it) *must* load **negative** on
  the anti-correlated scaffolding axis when forced to be scaffolding-only. That is exactly the ~28
  `low-info-fine` items — their positive r_pb guarantees a negative scaffolding loading. The
  "negative loading is a collapse artifact against the dominant correctness axis" hypothesis is
  **false** for these: they behave like (rare-pass, low-information) correctness items, not
  pedagogical/scaffolding items.
- An item with **negative r_pb** (good solvers *fail* it) loads **positive** on the scaffolding
  axis — these are the genuine reverse-behavers: `tb_0532_c06` (judge-inverted, r_pb −0.57 →
  a_scaff +1.20), `tb_0455_c07` (must-not guard, r_pb −0.42 → +0.28), and `tb_0432_c01`
  (r_pb −0.09 → +2.45). For these three the relabel "works" in the sign sense.

So relabeling sorts the 31 by the sign of their raw r_pb: it rescues the 3 truly reverse items and
mislabels the 28 low-info correctness-flavored items as negative-scaffolding.

---

## Recommendation

**Adopt Variant B (exclude-all-60) for the 2-skill pilot. Do NOT relabel the 31 as a block.**

1. **Relabeling fails its own test.** 26 of 31 stay negative on scaffolding under Variant A; the
   scaffolding axis gets *more* negative loaders (293 → 316) and a *lower* median a_scaffolding
   (0.386 → 0.249). The apparent stability bump (0.558 → 0.576) is consistency of a wrong sign, not
   health.
2. **No OOS payoff.** Pooled OOS is statistically identical (AUC −0.0014, acc −0.0010, log-loss
   +0.0020, Brier +0.0007 — all in the noise), because these are rare-pass, low-information items
   that carry almost no held-out signal either way. There is no generalization reason to re-enter
   them.
3. **Correctness axis + latent corr are unchanged** (a_correctness corr 0.987; latent r −0.42 vs
   −0.41), confirming the relabel is a scaffolding-side-only intervention — so its downside is not
   offset by any correctness-side gain.

### Handle the genuine reverse-behavers separately (the 3 with negative r_pb)

`tb_0532_c06` (judge-inverted), `tb_0455_c07` (must-not guard), and `tb_0432_c01` are the only
items that legitimately load positive scaffolding when relabeled. But three items cannot and should
not move an axis, and a positive scaffolding loading is not the same as a *correct* pedagogical
measurement:

- **Must-not guards** (`tb_0455_c07`, 43/82 pass, r_pb −0.42; conceptually also `tb_0532_c06`):
  good solvers "fail" them because they skip the required pedagogical explanation. The clean fix is
  **reverse-scoring** (flip the response so "correct behavior" = pass) rather than a scaffolding
  relabel, which only launders the reverse behavior into a positive loading on an axis it doesn't
  really belong to. Keep them **excluded** from the pilot fit and flag for reverse-scoring review.
- Do **not** relabel the 28 positive-r_pb `low-info-fine` items to scaffolding. If anything they
  are near-degenerate correctness items (mostly 5–20/82 pass) and belong with the `degenerate` set:
  keep excluded.

### Interaction with the 3-skill / 200-run fit

This is a **2-skill pilot** decision and should be read as one. Two forward-looking notes:

- The failure mode here is **structural**, driven by the correctness↔scaffolding anti-correlation
  (−0.42) that the collapse creates. In a genuine **3-skill** fit (content, diagnosis, scaffolding
  kept separate) the mechanism could differ — some `content|scaffolding` items might split their
  loading rather than flip sign. Re-run this exact A/B on the 3-skill Q before making any 3-skill
  relabel decision; do not carry the 2-skill relabel forward by default.
- The real bottleneck is **N = 82 persons** (per Run 5): scaffolding is under-identified regardless
  of item labeling. The 200-model run is the right lever for scaffolding health — not relabeling
  rare-pass items. Keep the 60 excluded going into the 200-run, and revisit the 3 reverse-behavers
  (reverse-scoring vs scaffolding relabel) once N is larger.

---

## Reproduce

```powershell
# from eduLLM-Evals/ ; python = ..\.venv\Scripts\python.exe
..\.venv\Scripts\python.exe staging/_exp_build_variants.py   # writes data/experimental/*.jsonl

# calibrations (2-skill collapse, corr on, grid 7, ridge 1e-3)
..\.venv\Scripts\python.exe scripts/calibrate_mirt.py --matrix staging/response_matrix_full_nonopt.csv --rubrics data/experimental/rubrics_qmatrix_excludeall.jsonl  --collapse content,diagnosis --estimate-latent-corr --grid 7 --out-dir staging/_exp/calib_B
..\.venv\Scripts\python.exe scripts/calibrate_mirt.py --matrix staging/response_matrix_full_nonopt.csv --rubrics data/experimental/rubrics_qmatrix_relabel31.jsonl --collapse content,diagnosis --estimate-latent-corr --grid 7 --out-dir staging/_exp/calib_A

# k-fold (k=5, seed 20260729, grid 7)
..\.venv\Scripts\python.exe scripts/kfold_cv_mirt.py --matrix staging/response_matrix_full_nonopt.csv --rubrics data/experimental/rubrics_qmatrix_excludeall.jsonl  --k 5 --seed 20260729 --grid 7 --out-dir staging/_exp/kfold_B
..\.venv\Scripts\python.exe scripts/kfold_cv_mirt.py --matrix staging/response_matrix_full_nonopt.csv --rubrics data/experimental/rubrics_qmatrix_relabel31.jsonl --k 5 --seed 20260729 --grid 7 --out-dir staging/_exp/kfold_A

..\.venv\Scripts\python.exe staging/_exp_analyze.py         # side-by-side + per-item table
```

**Artifacts (all under `staging/_exp/`, uncommitted):** `calib_{A,B}/calibration_mirt_manifest.json`,
`kfold_{A,B}/` (metrics_aggregate.json, item_param_stability.json, insample_baseline_item_params.csv),
`staging/_exp_summary.json`, `staging/_exp_relabel_table.csv`.
