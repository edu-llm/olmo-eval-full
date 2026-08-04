# Analysis Notes (CAT uncertainty audit, exclusion A/B, curation regrade impact)

Consolidated record of three read-only analyses done around the pilot freeze (2026-07-31).
Supersedes the separate `CAT Uncertainty Audit`, `Exclusion Experiment`, and
`Curation - Human Regrading Impact` memos. Nothing here changed code or the authoritative bank.

---

## 1. CAT uncertainty audit - what the harness actually models

Prompted by a claim that "uncertainty wasn't taken into account." **Verdict: false as stated
for ability estimation, true in three narrower places (one a spec deviation).** All three biases
point in the **optimistic** direction, which matters for the 200-run and the pre-registered ablation.

**Handled correctly (claim is false here):** `run_cat_person` carries a full dense posterior
covariance `U` (seeded N(0,I), updated in `tutor_cat.mirt.update`), including off-diagonal
cross-skill covariance (not diagonal-only). Stopping is uncertainty-driven (`sqrt(diag(U)) <
se_target` per skill, default 0.3, AND `min_items_per_skill` administered, default 15). Per-model
`final_se_{dim}` are persisted and the leaderboard already draws +/-1.96*SE intervals.

**Three real gaps (all optimistic):**
1. **Selection ignores the posterior (spec deviation).** Item scoring uses `||m||^2` = the
   *trace* of the item information at the plug-in point estimate; `U` never enters selection.
   `Implementation Strategy` sec 86 specifies **D-optimality** (`det(U^-1 + p(1-p) m m^T)`), which
   shrinks the currently-widest posterior direction. Uniform across all three harnesses
   (`cat_eval_tutorbench.py`, `_3skill.py`, `_multiskill.py`); `slogdet` appears only in
   `calibrate_mirt.py` (latent R), never in selection. Mitigation: content-balancing already
   targets the max-SE skill's pool, so the residual gap is within-pool ranking.
2. **Item parameters treated as known.** `load_bank` reads only `a`,`b` (no SEs); ability SEs
   condition on the calibrated params being exactly right. Standard operational practice, but the
   pilot bank is N=82 with 293/768 scaffolding items negative-loading and 22 extreme -> the
   leaderboard error bars are **too narrow** (widening would collapse some adjacent-rank distinctions).
3. **Estimator vs reference mismatch.** CAT ability = single Newton/Laplace step; the recovery
   reference = 41-node Gauss-Hermite EAP (7-node in the harness). So part of the reported recovery
   shortfall is method mismatch, not adaptivity information loss.

**Not gaps:** variable-length (SE+floor driven), exposure control (top-5), content balancing,
point-biserial>=0.05 + 60 exclude_from_fit filtering, dense `U`, observed-cells-only (holes never
scored as 0), deterministic seeded selection.

**Expected direction if closed:** D-optimality -> modest efficiency gain (larger in 3-skill);
param-SE propagation -> SEs widen, tests lengthen, convergence counts fall (a correction, not a
regression). `min_items_per_skill=15` is effectively a heuristic patch for the param-uncertainty
problem and may become principled once SEs widen. **Suggested order:** (1) D-optimality selection,
(2) estimator consistency - both safe on the pilot now; (3) item-parameter SE propagation - after
the 200-run recalibration, pre-registered as an ablation factor. Caveat on `recovery_r`: both
sides share the same params, so high r shows the adaptive subset reproduces the full-bank estimate,
not that abilities are correct.

---

## 2. Exclusion experiment - relabel-31 (A) vs exclude-all-60 (B)

For the 60 reverse-behaving (`a<0`) criteria held out of the fit: relabel the ~31 non-degenerate
ones as scaffolding-primary and re-enter (A), or keep all 60 excluded (B)? Setup identical
(Run-5 settings, `response_matrix_full_nonopt.csv`, k=5 seed 20260729, banks on copies). The 60 =
29 degenerate (excluded in both) + 30 low-info-fine + 1 judge-inverted; A re-enters the 31.

**Headline: exclude-all (B) wins. Relabeling backfires - 26 of 31 come out with a NEGATIVE
scaffolding loading under A.** Only the 3 genuine reverse-behavers (negative raw point-biserial)
turn positive; the ~28 low-info items have positive r_pb and therefore load negative on the
anti-correlated scaffolding axis.

| metric | B (exclude-all) | A (relabel-31) | verdict |
|---|---:|---:|---|
| latent r(correctness,scaffolding) | -0.418 | -0.409 | unchanged |
| negative-loading scaffolding items | 293 | 316 | A adds net negatives |
| median a_scaffolding (all nonzero) | 0.386 | 0.249 | A worse |
| cross-fold stability a_scaffolding | 0.558 | 0.576 | illusory (consistency of a wrong sign) |
| pooled OOS AUC | 0.9045 | 0.9031 | equal (in the noise) |

**Mechanism:** the scaffolding axis is negatively correlated with correctness (r approx -0.41).
An item with positive r_pb (correctness-able models pass it) *must* load negative when forced
scaffolding-only -> that is the ~28 low-info-fine items (they behave like rare-pass low-info
correctness items, not scaffolding). An item with negative r_pb (good solvers fail it) loads
positive -> the 3 genuine reverse-behavers: `tb_0532_c06` (judge-inverted, r_pb -0.57 ->
a_scaff +1.20), `tb_0455_c07` (must-not guard, -0.42 -> +0.28), `tb_0432_c01` (-0.09 -> +2.45).
Correctness axis untouched (a_correctness corr 0.987).

**Recommendation: adopt B (exclude-all-60) for the 2-skill pilot; do NOT relabel the 31 as a
block** (fails its own test, no OOS payoff, correctness/latent-r unchanged). Handle the 3 genuine
reverse-behavers by **reverse-scoring** (flip so correct behavior = pass), not scaffolding
relabel; keep them excluded from the pilot fit and flag for review. Forward: the failure is
structural (correctness<->scaffolding anti-correlation from the collapse) - **re-run this A/B on
the 3-skill Q before any 3-skill relabel**; the real fix for scaffolding is N=200 persons, not
relabeling rare-pass items. Keep the 60 excluded into the 200-run.

---

## 3. Curation -> human-regrading impact

Which human labels must be redone for `curation_v1` (`data/rubrics_qmatrix_final.jsonl` ->
`data/curated/rubrics_qmatrix_curated.jsonl`)? Id-keyed exactly via each record's `curation`
provenance block (not heuristic matching; renumbered/unchanged records verified 0 text/q diffs).

**Delta (6,462 -> 6,845, net +383):** unchanged 2,987; renumbered-unchanged 3,045;
text-modified **9** (6 soften + 3 rescope_optional: `tb_0001_c02/c07/c08`); split 64 parents ->
142 children; merged 357 style orphans -> 313 optional `style_surface`; added 349 optional
presentation; removed 0; q_mapping-only 0. "Changed" set for human impact = 430 original ids.

**Cross-referenced against the two human samples:**
- **Q-matrix human review (25 criteria):** only **1** changed - `tb_0020_c07` (split; children
  inherit the parent `q_mapping`, so Q-label re-review is optional).
- **Judge-selection grades (87 criteria / 261 cases):** **15** changed, only **6 substantive** -
  3 rescopes (`tb_0001_c02/c07/c08`) + 3 split parents (`tb_0003_c09, tb_0336_c03, tb_0340_c08` ->
  7 new atomic children needing fresh P/F). The other 9 are non-gating presentation orphans
  (reported `unmapped`, excluded from the per-skill gate).

**Bottom line:** no human action is strictly required to trust prior conclusions. Judge selection
is a *relative* ranking over an identical case set, so 15/87 equally-affected criteria cannot flip
the winner -> **stable**. 414 of 430 changed criteria are outside both human samples (need only
downstream judge re-grade + later Q-relabel, no human regrading). Highest-value optional re-grades
if the curated bank becomes the graded instrument: the 7 split children + 3 tb_0001 rescopes.
