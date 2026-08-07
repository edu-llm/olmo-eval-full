# TutorBench calibration

Scenario-level MIRT recalibration of the TutorBench benchmark, mirroring the sibling scenario-level
packages (`bridge_calibration/`, `wildbench_calibration/`, `biggen_calibration/`). A scenario is
administered as a **testlet bundle** of its per-criterion items; the real production engine
(`tutor_cat.engine.run_evaluation` via `scripts/scenario_cat_lib.py`) selects a whole scenario per
CAT step and updates ability. Nothing here reimplements CAT selection or the item-response update.

**Status: STUDY / reporting ONLY. Production engine (`tutor_cat/`, `scenario_cat_lib.py`) untouched.
Nothing committed.**

---

## ⭐ CANONICAL SCALE — `unidim/` (correctness-only, floor 20 / SE 0.27)

**The canonical TutorBench scale is the correctness-only unidimensional package in [`unidim/`](unidim/README.md).**
A single latent **correctness** ability, fit as a unidimensional 2PL on the correctness-loading
criteria (scaffolding dropped as a measured construct), at the locked operating point
**floor(min_scenarios)=20 / SE_ability target=0.27**.

### Headline numbers @ 20/0.27 (N=114)

| quantity | value |
|---|---|
| OOS recovery r | **0.955** |
| slope | **1.006** |
| θ-MAE | **0.416** |
| median test length | **20** scenarios |
| %reach (SE_ability≤0.27) | **59.6%** |
| median SE_total | **0.275** |
| fixed SE_param offset | **0.113** |

(N=114 excludes only `Qwen/Qwen1.5-1.8B`; `salamandra-7b-instruct` is rehabilitated under this scale
and kept. Full detail + machine-readable `summary.json` in `unidim/`.)

### Why unidim is canonical (rationale)

- **Same axis, cleaner.** Full-bank EAP θ correlates with the previous 2-skill correctness axis at
  **r = 0.9994** (ρ = 0.9993) — models rank identically, so unidim is the *same* correctness
  construct, not a new one, with the scaffolding-induced calibration noise removed.
- **Tighter error bars.** SE_param **0.207 → 0.113 (−46%)** and SE_total **0.412 → 0.344 (−16%)**.
  The gain is fit-stability (dropping the reverse-loading scaffolding items; clamped negative
  loadings 349 → 33), not new information (SE_ability essentially unchanged).
- **Shorter tests** (median 20 vs 25 scenarios) and **higher reach** (59.6% vs 43.4% correctness),
  with recovery essentially unchanged (r within 0.012, better slope, lower θMAE).
- **Trade-off:** scaffolding is **dropped as a measured construct**. If a separate scaffolding
  ability is a product requirement, unidim cannot supply it.

Full head-to-head (table, rank-agreement scatter, SE bars, recovery/efficiency figures):
[`unidim/comparison_vs_2skill/`](unidim/comparison_vs_2skill/README.md).

See [`unidim/README.md`](unidim/README.md) for the locked config, methodology, experiment index
(04 efficiency, 05 recovery, 06 floor×SE grid, 07 parameter uncertainty, 08 leaderboard, 09 p-IRT),
the fitted bank, and the build scripts.

---

## Superseded — `two_skill_OUTDATED/`

The previous **2-skill (correctness + scaffolding)** calibration package now lives under
[`two_skill_OUTDATED/`](two_skill_OUTDATED/README.md). It is **retained for reference and possible
revisit after more models are graded**, but is superseded by `unidim/` and should not be cited as
current. It still carries the one thing unidim drops — a separately-measured scaffolding axis
(OOS r 0.932, θMAE 0.256) — which is why it is kept rather than deleted.

## Graduation manifest

`FLOW_PACKAGE.md` (top level) is the graduation manifest for porting the **canonical unidim** scale
into the FRQ eval flow (`flow/mirt-frq`). The superseded 2-skill manifest is preserved at
`two_skill_OUTDATED/FLOW_PACKAGE.md`.

## Layout

```
tutorbench_calibration/
  README.md              this file
  FLOW_PACKAGE.md        graduation manifest -> flow/mirt-frq (unidim)
  unidim/                ⭐ CANONICAL correctness-only unidim scale (20/0.27)
    README.md, summary.json, bank/, scripts/, experiments/{04,05,06,07,08,09}/, comparison_vs_2skill/
  two_skill_OUTDATED/    SUPERSEDED 2-skill (correctness+scaffolding) package (retained)
    README.md, FLOW_PACKAGE.md, fit_manifest.json, model_leaderboard.csv, experiments/, scripts/
```
