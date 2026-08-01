# CAT Uncertainty Audit — Selection, Parameter Error, Estimator Mismatch

> **Read-only audit, 2026-07-31.** No code was changed. Prompted by a collaborator's claim
> that "uncertainty wasn't being taken into account by the scripts" when we ran CAT on
> TutorBench and that "our scripts were more simplified." This memo establishes what the
> code actually does, so the fix effort lands on the real gaps.

**Verdict: partly true.** The claim is **false** as stated about ability estimation — the
CAT loop carries a full posterior covariance, stops on posterior SE, and the published
leaderboard already draws ±1.96·SE intervals. The claim is **true** in three narrower
places, one of which is a deviation from our own PRD:

1. **Item selection ignores the posterior** and uses the *trace* of the information matrix
   at a plug-in point estimate, where `plans+prds/Implementation Strategy + Success
   Metrics.md` §86 specifies **D-optimality**. This is a spec deviation, not a style choice.
2. **Item parameters are treated as known**, with no calibration error propagated. Standard
   operational practice, but it means the leaderboard's error bars are **too narrow**.
3. **The CAT estimator and its own recovery reference disagree** — a single Newton/Laplace
   step versus 41-node Gauss–Hermite quadrature — so part of the reported recovery shortfall
   is method mismatch rather than information loss from adaptivity.

None of the three is a correctness bug. All three bias results in the **optimistic**
direction, which matters because the 200-model run and the pre-registered ablation are about
to be built on these numbers.

---

## 1. What the code demonstrably does handle

Ability uncertainty is modelled properly. `run_cat_person` seeds a full covariance prior at
`scripts/cat_eval_tutorbench_multiskill.py:314` (`U = np.eye(n_dims)`, i.e. N(0, I)) and
updates it through `tutor_cat.mirt.update`:

```58:63:tutor_cat/mirt.py
    info = p * (1.0 - p) * np.outer(m, m)
    U_new = np.linalg.inv(np.linalg.inv(U) + info)
    # Symmetrize to wash out floating-point asymmetry from the inversions.
    U_new = (U_new + U_new.T) / 2.0

    theta_new = theta + U_new @ m * (float(y) - p)
```

Because `m = q ⊙ a` is dense across the skills an item loads, the outer product is dense and
the **off-diagonal cross-skill covariance is genuinely carried** — this is not a
diagonal-only approximation treating the skills as independent.

Three consequences follow, all of which contradict the blanket form of the claim:

- **Stopping is uncertainty-driven, not fixed-length.** Convergence requires
  `√diag(U) < se_target` per skill (default `0.3`, `multiskill.py:911`) *and* at least
  `min_items_per_skill` administered items actually loading that skill
  (`multiskill.py:158`, default 15). Both conditions are checked at `multiskill.py:365-373`.
  This is already documented in `CAT Evaluation - TutorBench 2-skill.md` §35.
- **Per-model SEs are persisted**, not discarded — `final_se_{dim}` at `multiskill.py:481`,
  written to `cat_per_model.csv`.
- **The leaderboard publishes intervals.** `build_correctness_leaderboard.py:135` carries
  `cat_se_correctness` into the CSV, and the caterpillar figure draws them:

```169:174:scripts/build_correctness_leaderboard.py
    ci = 1.96 * order["cat_se_correctness"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(8.5, max(10, 0.22 * n)))
    ax.errorbar(
        order["theta_correctness_cat"], ypos, xerr=ci, fmt="none",
        ecolor="#9bb0d6", elinewidth=1.4, capsize=2, zorder=1,
        label="CAT theta \u00b1 1.96\u00b7SE",
```

---

## 2. Gap 1 — selection discards the posterior (deviates from the PRD)

Every candidate item is scored like this:

```348:351:scripts/cat_eval_tutorbench_multiskill.py
        p = expit(m @ theta - b[rem])
        info = p * (1.0 - p) * np.sum(m * m, axis=1)
        # deterministic: information descending, ascending bank index (criterion_id) on ties
        order = np.lexsort((rem, -info))
```

`np.sum(m * m, axis=1)` is `‖m‖²`, the **trace** of the item's information matrix, evaluated
at the current point estimate `theta`. The posterior `U` does not appear anywhere in the
selection step. The practical effect is that selection prefers items with large overall
loading magnitude regardless of which ability directions are already well determined.

Our own specification asks for something else:

```86:86:plans+prds/Implementation Strategy + Success Metrics.md
Note that since we are extending this framework to be multidimensional, fisher information can be represented by a dxd matrix. The next item chosen will be one that maximizes the determinant of the updated information matrix. This criterion is called D-optimality.
```

D-optimality maximizes `det(U⁻¹ + p(1−p)·mmᵀ)`, which by construction favours items that
shrink whichever posterior direction is currently widest. That is precisely the
uncertainty-awareness the trace criterion drops.

**Scope of the deviation.** It is uniform and original, not a regression introduced by the
newer harness — the identical line appears at `cat_eval_tutorbench.py:152`,
`cat_eval_tutorbench_3skill.py:196`, and `cat_eval_tutorbench_multiskill.py:349`. Confirming
the absence from the other direction: `slogdet` occurs exactly once in the entire codebase,
at `calibrate_mirt.py:330`, where it is used for the latent correlation matrix `R` during
fitting and has nothing to do with item selection.

**Mitigating factor.** Content balancing (`multiskill.py:330-342`) already targets the skill
with the largest SE and restricts the candidate pool to items loading it, so the policy layer
recovers heuristically some of what D-optimality would do principledly. The residual gap is
*within-pool*: once the pool is fixed, ranking is uncertainty-blind.

---

## 3. Gap 2 — item parameters treated as known

`load_bank` reads discriminations and difficulty and nothing else:

```209:214:scripts/cat_eval_tutorbench_multiskill.py
def load_bank(path: Path, a_cols: list[str]) -> tuple[list[str], np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    items = df["criterion_id"].astype(str).tolist()
    A = df[a_cols].to_numpy(dtype=float)
    b = df["b"].to_numpy(dtype=float)
    return items, A, b
```

No standard errors are loaded, and `tutor_cat/mirt.py` describes `a`, `b`, and `q` as
"frozen" in its module docstring. Ability SEs therefore condition on the calibrated
parameters being exactly right.

This is defensible — nearly all operational CAT freezes the bank after calibration — but the
pilot bank was fitted on **82 models**, and the scaffolding hygiene audit found 293 of 768
Q-scaffolding items fitting negative with 22 at extreme `a`. Parameter error at that scale is
not a rounding concern.

**The user-visible consequence** is the leaderboard error bars in §1: they are computed from
`√diag(U)` alone and are consequently **too narrow**. Widening them would collapse some
adjacent rank distinctions that currently look separated.

---

## 4. Gap 3 — the CAT estimator disagrees with its own reference

The recovery reference integrates properly over the ability grid:

```280:286:scripts/cat_eval_tutorbench_multiskill.py
    ym = np.where(mask, y_obs, 0.0)
    nm = np.where(mask, 1.0 - y_obs, 0.0)
    eta = A @ grid.T - b[:, None]
    ll = ym @ log_expit(eta) + nm @ log_expit(-eta)
    joint = ll + log_prior
    post = np.exp(joint - logsumexp(joint))
    return post @ grid
```

The CAT side, by contrast, takes a **single** Newton/Fisher-scoring step per item
(`mirt.py:63`) with no iteration to convergence — the Laplace approximation already noted in
`Calibration - K-Fold Cross-Validation (2-skill).md` §146.

So `recovery_r` compares a quadrature EAP against a Laplace estimate. In the harness the grid
is 7 nodes per dimension (`multiskill.py:909`); in the leaderboard the reference is refit at
**41** nodes (`build_correctness_leaderboard.py:106-107`), making the mismatch wider there
than in the harness that produced the headline table.

---

## 5. What is *not* a gap

Worth stating so effort does not land here:

| concern | status |
|---|---|
| Fixed-length testing | Not the case — variable-length, SE + floor driven |
| Exposure control | Present — top-N = 5 uniform pick (`multiskill.py:352-353`) |
| Content balancing | Present — targets max-SE skill (`multiskill.py:330-342`) |
| Non-discriminating items administered | Filtered — point-biserial ≥ 0.05, plus the 60 `exclude_from_fit` items removed |
| Diagonal-only covariance | Not the case — full dense `U` |
| Missing / `no_decision` cells scored as 0 | Not the case — only observed items are administrable (`multiskill.py:311`) |
| Non-reproducible selection | Deterministic — `lexsort` tie-break, per-model seeded RNG |

---

## 6. Expected direction if the gaps are closed

| change | effect on reported numbers |
|---|---|
| D-optimal selection | Modest efficiency gain: slightly shorter tests, or slightly better recovery at fixed length. Larger in the 3-skill instrument, where ability directions differ more |
| Propagate item-parameter SE | **SEs widen → tests get longer, convergence counts fall.** Current length and convergence figures are optimistic |
| Match the CAT estimator to quadrature EAP | Recovery `r` likely rises slightly; removes an artifact rather than a real effect |

If the collaborator lands the parameter-uncertainty work, **expect convergence rates to drop
and test lengths to grow.** That is the correction functioning, not a regression, and it
should not be read as the CAT getting worse.

---

## 7. The optimism to watch before the 200-run

`recovery_r` compares CAT `theta` against a full-bank EAP computed with **the same item
parameters**. Both sides inherit identical parameter error, so a high `r` establishes that
the adaptive subset reproduces the full-bank estimate — *not* that the abilities are correct.
Given the scaffolding fit pathologies, that distinction is load-bearing.

The OOS check is honest about the bank but not fully out-of-sample about the target: the CAT
runs on fold-trained parameters (`multiskill.py:585`) while the comparison target
`full_theta` is drawn from the **in-sample** full-bank EAP (`multiskill.py:582`, used at
`:608`). It answers "does CAT with held-out parameters reproduce the in-sample full-bank
ranking," which is the right question for CAT length, and a weaker claim than out-of-sample
ability validation.

Finally, `min_items_per_skill = 15` reads, in this light, as a **heuristic patch for a
parameter-uncertainty problem**. The harness docstring already names the failure it fixes —
scaffolding SE collapsing below target on 3–4 items, flattering its recovery. If SEs widen
once calibration error is propagated, the floor may become derivable on principled grounds
rather than chosen by sweep, and the provisional floor decision should be revisited then.

---

## 8. Suggested order of work

1. **D-optimality in selection.** Smallest change, clearest justification (it is the spec),
   and isolated to the scoring line in three harnesses. Lands as a clean ablation factor.
2. **Estimator consistency.** Either iterate the Newton step to convergence or run the CAT
   ability through the same quadrature EAP as the reference, so `recovery_r` measures
   adaptivity rather than estimator choice.
3. **Item-parameter SE propagation.** Largest effort and largest downstream disruption to
   published numbers. Best sequenced *after* the 200-run recalibration, when the bank is
   fitted on 200 rather than 82 persons and the parameter error is smaller to begin with.

Items 1 and 2 are safe to do on the pilot now. Item 3 changes the meaning of every published
SE and should be pre-registered as an ablation factor rather than folded in silently.
