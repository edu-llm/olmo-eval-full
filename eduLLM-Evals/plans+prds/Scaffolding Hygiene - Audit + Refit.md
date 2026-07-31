# Scaffolding Hygiene — Audit + Refit (2-skill pilot)

> **ADOPTED 2026-07-31.** `ridge = 1e-2` is now the M2PL fit default in
> `scripts/calibrate_mirt.py` (argparse) and the frozen pilot baseline artifacts (2- and
> 3-skill calibrated JSONLs, k-fold, CAT reports) were regenerated at 1e-2. The guardrail
> held on the standard floor=15 pipeline: OOS **correctness** recovery *improved* (2-skill
> 0.919→0.929, 3-skill 0.921→0.934) and the 3-skill fit stayed a clean optimum. See the
> `Pilot Baseline Freeze (2 and 3 skill).md` before/after table.

**Date:** 2026-07-31
**Question:** Can a calibration-hygiene move (heavier ridge, a loading cap, or excluding
extreme-a items) improve the health of the **scaffolding** axis in the 2-skill (collapse
content+diagnosis → "correctness"; scaffolding separate) M2PL fit, *without* hurting the
correctness axis? Primary decision metric: **out-of-sample (k-fold) CAT recovery `r` for
scaffolding**; guardrail: OOS correctness recovery must not regress.

**Data / settings (apples-to-apples, all variants):** matrix
`staging/response_matrix_full_nonopt.csv` (82 models × 6,180 nonoptional criteria); Q from
`data/TutorBench/curated/rubrics_qmatrix_curated.jsonl` (the current curated bank, **60
`exclude_from_fit` flags kept as baseline — none removed**); collapse `content,diagnosis`;
grid 7 (GH); EM `max_iter 200`, `tol 1e-4`. Bank fit uses `--estimate-latent-corr` (ON);
k-fold uses corr OFF, **k=5, seed 20260729** (Run 5 protocol). OOS recovery is produced by
the stock CAT policy harness (`scripts/cat_eval_tutorbench_multiskill.py --skills 2`, policy
ON: pbis≥0.05, min-items-per-skill 10, content-balance, exposure-top-n 5, seed 20260730).

**Read-only / non-destructive.** The authoritative bank and `scripts/calibrate_mirt.py` were
NOT edited. All work lives under `staging/scaff_hygiene/` (audit, patched-fitter module,
per-variant bank CSVs, k-fold dirs, CAT eval dirs). The cap / scaffolding-ridge variants use a
drop-in fitter (`staging/scaff_hygiene/scaff_common.py`) whose *stock* code path reproduces
`calibrate_mirt.fit_m2pl_em` **exactly** (verified: identical loglik −60630.83 and identical
475/293/22 loading counts). Nothing was committed.

> **Baseline note (important for reading the deltas).** Every number below is regenerated from
> scratch with the *current* curated bank (60 flags → 3,443 fitted items). This **fresh
> baseline** (OOS scaffolding `r` = **0.536**) is lower than the previously-reported 2-skill
> figure (`reports/cat_eval_policy/2skill`, OOS scaffolding = 0.644), which was computed on a
> *stale* bank snapshot (3,497 items, an earlier curation). The two are not directly
> comparable; all variant deltas here are measured against the fresh baseline so the pipeline
> is byte-for-byte identical across variants. See "Baseline sensitivity caveat" below.

---

## Part A — Audit of the scaffolding axis (baseline fit, ridge 1e-3)

Fit block: **3,443 items × 82 persons**, 276,587 observed cells; latent
correctness↔scaffolding `r = −0.418`. Full per-item table:
`staging/scaff_hygiene/audit.csv` (+ `audit_summary.json`).

**Loading distribution (768 Q-scaffolding items):**

| quantity | value |
|---|---|
| items with a Q-scaffolding entry | **768** |
| positive loaders (`a_scaffolding > 0`) | **475** |
| **negative loaders (`a_scaffolding < 0`)** | **293** (≈38%) |
| extreme (`\|a\| ≥ 6.0`) | **22** |
| a_scaffolding, positive loaders (min / q25 / median / q75 / max) | 0.003 / 0.508 / **0.917** / 1.482 / 14.98 |
| a_scaffolding, negative loaders (min / median / max) | −9.27 / −0.856 / −0.0001 |
| low-info `\|a\| < 0.2` | 87 |
| extreme difficulty `\|b\| > 6` | 140 |

**The 22 EXTREME_A items are fit artifacts, not signal.** They are near-separation items:
almost no model passes them (pass-rate ≈ 0.01–0.09; 1–7 passes out of ~80), so the loading
`a` runs away while the difficulty `b` inflates in lock-step (b up to 34). Their
point-biserial with full-bank scaffolding ability is *weak or ~zero* (0.16–0.32, several at
0.005/0.003/0.000), confirming they carry essentially no genuine scaffolding information.
Several are exact duplicates of the same degenerate solution (a = 9.18, b = 25.10, 1 pass
each). The negative-extreme trio (a = −6.0 … −9.3, negative pbis) are reverse-behaving items.
One near-all-pass mirror exists (tb_0588_c06: 77/81 pass, a = 6.50).

| criterion_id | a_scaffolding | b | n | passes | pass_rate | pbis(scaff ability) |
|---|---:|---:|---:|---:|---:|---:|
| tb_0576_c04 | 14.98 | 8.24 | 82 | 7 | 0.085 | 0.190 |
| tb_0601_c04 | 12.42 | 13.44 | 81 | 3 | 0.037 | 0.279 |
| tb_0187_c07 / _0517_c04 / _0045_c06 / _0556_c05 / _0601_c03 / _0325_c06 / _0549_c06 | 9.18 | 25.10 | ~82 | 1 | 0.012 | 0.159 |
| tb_0505_c01 | 8.82 | 14.51 | 81 | 2 | 0.025 | 0.226 |
| tb_0638_c03 | 8.81 | 14.50 | 80 | 2 | 0.025 | 0.226 |
| tb_0480_c13 | 8.38 | 20.23 | 79 | 6 | 0.076 | 0.005 |
| tb_0517_c07 / _0547_c04 / _0099_c12 / _0648_c03 / _0077_c02 | 7.39 | 34.47 | ~81 | 2 | 0.024 | ~0.00 |
| tb_0588_c06 | 6.50 | −8.89 | 81 | 77 | 0.951 | 0.324 |
| tb_0513_c05 | 6.31 | 8.82 | 74 | 3 | 0.041 | 0.303 |
| tb_0165_c11 | −6.03 | 9.19 | 82 | 2 | 0.024 | −0.221 |
| tb_0231_c10 | −6.26 | 10.32 | 82 | 4 | 0.049 | −0.316 |
| tb_0191_c07 | −9.27 | 18.78 | 82 | 4 | 0.049 | −0.316 |

**Headline audit finding:** the scaffolding axis is not just thin — it is *noisy*. ~38% of
its 768 items fit a negative loading, and 22 blow up on near-separation. Both are classic
symptoms of estimating discriminations for the rarest skill on only 82 persons.

---

## Part B — Refit variants vs the fresh baseline

Each variant = full re-fit of the 2-skill bank **+** fresh 5-fold CV **+** CAT OOS recovery,
identical pipeline, only the hygiene knob changes. Full table:
`staging/scaff_hygiene/variant_comparison.csv`.

| variant | ridge | scaff-ridge | cap | pos / neg loaders | \|a\|≥6 | median a⁺ | stab a_corr | **stab a_scaff** | stab b | in-sample scaff | **OOS corr** | **OOS scaff** | Δ OOS scaff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **baseline** | 1e-3 | – | – | 475 / 293 | 22 | 0.917 | 0.790 | 0.558 | 0.848 | 0.817 | 0.9159 | **0.5357** | — |
| ridge 5e-3 | 5e-3 | – | – | 466 / 302 | 14 | 0.910 | 0.809 | 0.598 | 0.869 | 0.842 | 0.9226 | **0.6453** | **+0.110** |
| **ridge 1e-2** | 1e-2 | – | – | 442 / 326 | 6 | 0.899 | 0.818 | 0.618 | 0.879 | 0.845 | **0.9305** | **0.6665** | **+0.131** |
| ridge 2e-2 | 2e-2 | – | – | 442 / 326 | 1 | 0.898 | 0.829 | 0.631 | 0.889 | 0.859 | 0.9292 | **0.6697** | **+0.134** |
| scaff-ridge 5e-2 | 1e-3 | 5e-2 | – | 442 / 326 | 0 | 0.812 | 0.814 | **0.777** | 0.865 | 0.839 | 0.9274 | 0.6634 | +0.128 |
| scaff-ridge 1e-1 | 1e-3 | 1e-1 | – | 442 / 326 | 0 | 0.637 | 0.812 | **0.801** | 0.866 | 0.788 | 0.9215 | 0.6341 | +0.098 |
| cap \|a_scaff\|≤4 | 1e-3 | – | 4.0 | 435 / 333 | 0 | 0.509 | 0.750 | 0.722 | 0.843 | 0.780 | 0.9293 | **0.5324** | **−0.003** |
| exclude 22 extreme-a | 1e-3 | – | – | 485 / 261 | 19* | 0.931 | 0.791 | 0.523 | 0.849 | 0.805 | 0.9303 | 0.6202 | +0.085 |

`stab_*` = median cross-fold (k=5) Pearson correlation of the per-item parameter (Run-5
"needs-refit" health metric). *exclude-extreme's residual `\|a\|≥6` count is new
near-separation items that re-emerge among the survivors once the original 22 are removed.

### What the variants show

1. **Global ridge is a clean, monotone win.** Raising the loadings-ridge 1e-3 → 5e-3 → 1e-2
   → 2e-2 monotonically improves **all three** decision metrics at once — OOS scaffolding
   (0.536 → 0.645 → 0.667 → 0.670), OOS correctness (0.916 → 0.931), and cross-fold
   scaffolding stability (0.558 → 0.631) — and it drains the extreme-a artifacts (22 → 1).
   OOS scaffolding **plateaus** at ridge ≈ 1e-2 (2e-2 adds only +0.003 and costs a hair of
   correctness). A monotone response across three ridge levels, in three independent metrics,
   is strong evidence this is real regularization signal, not fold noise.

2. **Scaffolding-specific ridge trades interpretability for stability, no OOS edge.** Extra
   L2 on only the scaffolding column drives cross-fold stability far higher (0.558 → **0.777**
   at 5e-2, 0.801 at 1e-1) — the best stability of any variant — but its OOS recovery (0.663)
   is **no better** than plain global ridge 1e-2 (0.667), and pushing it to 1e-1
   over-shrinks (median a⁺ collapses to 0.64) and OOS recovery *falls back* to 0.634. Classic
   bias/variance turnover: shrink buys stability past the point where it still buys recovery.

3. **A hard cap does not help recovery.** Clamping `|a_scaffolding| ≤ 4.0` in the M-step kills
   the artifacts and lifts stability (0.722), but OOS scaffolding recovery is **flat**
   (0.532, ≈ baseline). Capping touches only the ~22 extreme items and distorts their
   difficulty pairing while leaving the 293 negative loaders and the bulk thin-N noise
   untouched — so the held-out ability estimates don't actually get better. **Not recommended.**

4. **Excluding the 22 extreme-a items helps modestly but treats the symptom.** OOS scaffolding
   +0.085 (0.620) and correctness improves, but cross-fold **stability does not** (0.523,
   ≈ baseline) and fresh near-separation items re-emerge among the survivors. Removing the
   worst offenders is strictly worse than shrinking the whole column with ridge.

---

## Recommendation

**Adopt `ridge = 1e-2` for the 2-skill M2PL calibration** (up from the current `1e-3`
default in `scripts/calibrate_mirt.py`). Among all moves tested it gives:

- the **best OOS scaffolding recovery** (0.667 vs 0.536 baseline; tied with 2e-2), the
  **primary decision metric**;
- **no correctness regression** — correctness recovery *improves* to 0.931 (best of any
  variant), satisfying the guardrail;
- improved cross-fold scaffolding stability (0.558 → 0.618) and near-elimination of the
  extreme-a artifacts (22 → 6);
- **zero new machinery** — a one-line default change; no per-column ridge, no clamp, no bank
  edits, and it is a global change that also benefits correctness.

Runner-up: **scaffolding-specific ridge 5e-2** if maximizing *cross-fold stability* (0.777) is
the explicit goal — but it needs the non-standard per-column penalty and offers no OOS-recovery
advantage over global 1e-2, so it is not worth the extra complexity for the pilot. `ridge 2e-2`
is essentially tied with 1e-2 on scaffolding and marginally worse on correctness/interpretability;
1e-2 is the balanced pick. **Do not** adopt the cap (no OOS benefit) or extreme-a exclusion
(smaller benefit, no stability gain) as the primary lever.

### Should this become the default in `calibrate_mirt.py` for the 200-run?

Adopt it **now for the 82-person 2-skill pilot bank**, but **do not hard-code `1e-2` as the
permanent default.** The optimal ridge is a function of person-N: more persons → less shrinkage
needed. The concrete recommendation is to (a) set the pilot bank's ridge to 1e-2, and (b) at
the 200-run, **re-run this exact k-fold OOS-recovery ridge sweep** (baseline → 5e-3 → 1e-2 →
2e-2) on the new data and pick the plateau point empirically, rather than freezing 1e-2.

---

## Caveats

- **N = 82 is still the bottleneck; ridge is mitigation, not a cure.** Even the best variant
  caps OOS scaffolding recovery at ~0.67, far below correctness (~0.93). Ridge recovers a
  chunk of the loss the artifacts were causing, but it cannot manufacture scaffolding signal
  that the 82-person crowd does not contain. The Run 5 conclusion stands: **the durable fix is
  more models (persons), not calibration tweaks.** Ridge 1e-2 is a free, honest hygiene
  improvement to bank *before* the 200-run, not a substitute for it.
- **Baseline sensitivity caveat.** The fresh-baseline OOS scaffolding (0.536) sits below the
  previously-reported 0.644 (stale 3,497-item bank). A meaningful share of the headline
  "+0.131" is measured against a fresh baseline that is itself on the low side; against the
  older reported baseline the ridge-1e-2 gain is a more modest ~+0.02. The *robust* claims —
  independent of which baseline you anchor to — are (i) the monotone 3-point ridge trend, and
  (ii) ridge lands OOS scaffolding at ~0.66–0.67 while *improving* correctness. Those hold
  regardless of the baseline point.
- **This is a 2-skill pilot decision.** The 3-skill fit (correctness / scaffolding /
  presentation, OOS scaffolding ~0.56) uses the same `ridge 1e-3` and shows the same scaffolding
  weakness; the ridge mechanism is axis- and dimension-agnostic and should transfer, but it was
  **not** re-validated here. Before making `1e-2` a global default, run the same sweep on the
  3-skill instrument. The collapse decision (content+diagnosis → correctness) and the 2-skill
  CAT structure are unaffected by this change.

---

## Reproduce

```powershell
# from repo root: ...\eduLLM-Evals ; Python: ..\.venv\Scripts\python.exe
..\.venv\Scripts\python.exe staging/scaff_hygiene/audit.py                       # Part A
..\.venv\Scripts\python.exe staging/scaff_hygiene/experiment.py --variant baseline
..\.venv\Scripts\python.exe staging/scaff_hygiene/experiment.py --variant ridge5e3
..\.venv\Scripts\python.exe staging/scaff_hygiene/experiment.py --variant ridge1e2
..\.venv\Scripts\python.exe staging/scaff_hygiene/experiment.py --variant ridge2e2
..\.venv\Scripts\python.exe staging/scaff_hygiene/experiment.py --variant scaffridge
..\.venv\Scripts\python.exe staging/scaff_hygiene/experiment.py --variant scaffridge1e1
..\.venv\Scripts\python.exe staging/scaff_hygiene/experiment.py --variant cap4
..\.venv\Scripts\python.exe staging/scaff_hygiene/experiment.py --variant excludeXA
```

**Artifacts (`staging/scaff_hygiene/`):** `audit.csv`, `audit_summary.json`,
`variant_comparison.csv`, `scaff_common.py` (patched drop-in fitter), `experiment.py`,
per-variant `bank_<v>.csv`, `kfold_<v>/`, `cat_<v>/` (incl. `cat_metrics.json`),
`result_<v>.json`, `rubrics_excludeXA.jsonl` (modified curated for the exclusion variant only;
the authoritative curated bank is untouched).
