# InFoBench V4 dense-fitter remediation plan

**Status:** Prospective append-only design; V1--V3 remain immutable failed evidence

**Prepared:** 2026-08-05

**Branch:** `frq/infobench`

## Objective

Resolve the numerical failure that blocked the InFoBench V2 CAT study without
loosening a scientific gate merely to obtain a pass. The existing 52-tutor
cohort and frozen Qwen labels remain fixed. No human audit and no cohort
expansion are part of this run, so every result remains an internal-development
result conditional on those labels.

V4 is numerical verification only. It may authorize a later Phase-3 CAT run,
but it cannot run CAT, select a CAT policy, or make a CAT-efficiency claim.

## What V3 established

V3 correctly blocked Phase 3. All 72 fits converged and held-out prediction
equivalence passed, but all 12 refitted-bank theta panels failed. The failure is
not explained by the strict per-model maximum alone: every panel also exceeded
the original p95 tolerance.

The audit found that the V3 fitter used a fixed global Gauss--Hermite grid whose
central spacing remained about 0.35, 0.31, and 0.29 theta units at 81, 101, and
121 nodes. Full-bank posterior SE was about 0.06, so a tutor's posterior was
usually concentrated on one quadrature node. Changing the grid therefore moved
the fitted solution instead of producing monotone numerical convergence.

An exploratory full-cohort 1PL diagnostic using bounded normal-trapezoid
integration showed that 401 and 801 fit nodes agreed to numerical precision.
This diagnostic chose the remediation family; it is not itself a promotable V4
result.

## Fitter remediation

- Keep historical Gauss--Hermite behavior available and unchanged by default.
- Add an explicit one-dimensional `normal_trapezoid` calibration path on
  `[-8, 8]`.
- Require convergence of the returned parameter iterate, rather than declaring
  convergence before an unchecked final M-step.
- Record penalized-objective and maximum-parameter-change traces.
- Warm-start each item's M-step from its current parameters on the new path.
- Use a vectorized 1PL M-step so dense integration is practical.
- Use a vectorized one-dimensional shrinkage-2PL M-step with an audited Newton/
  Fisher fallback and item-wise line search.
- Fail closed on non-finite objectives, non-monotone objective movement beyond
  numerical tolerance, invalid parameters, or an unfinished inner optimizer.

The frozen returned-iterate stopping contract is a maximum of 1,500 M-steps,
absolute penalized-objective change below `1e-4`, maximum parameter-coordinate
change below `5e-5`, maximum inner-optimizer absolute gradient at most `1e-6`,
and two consecutive qualifying returned iterates. These values were fixed after
implementation/synthetic checks and an explicitly disclosed full-cohort
engineering timing diagnostic, before the official V4 output leaf was opened.

The production comparison uses 401 versus 801 fitting nodes. Grid 401 is the
smallest candidate and may be locked only if it is equivalent to grid 801.

## Eligible calibration specifications

The production panel is frozen to:

1. `1pl_fixed_a1`;
2. `log_shrinkage_2pl_lambda16`; and
3. `log_shrinkage_2pl_lambda4`.

The three unrestricted free-2PL ridge variants are not production candidates in
V4. With only 52 tutor response vectors per item, their per-item discrimination
parameters are weakly identified; V3 already preserved their diagnostic result,
including exportability failures for ridge 0.01 and 0.001. Excluding them now is
a prospective simplification based on that failed numerical study, not a
post-hoc choice made from V4 or CAT outcomes.

## Frozen V4 schedule

For each of the three eligible specifications and each of six scopes (full plus
five outer folds):

1. fit 401 nodes from the common deterministic initialization;
2. fit 801 nodes from the same deterministic initialization;
3. fit 801 nodes by continuation from that scope's completed 401-node fit; and
4. compare parameters, exportability, held-out predictions, and common-support
   theta values.

Both 801 starts must be individually valid and must agree; continuation cannot
rescue a failed cold fit. The deterministic higher penalized objective is used,
with cold winning an exact tie. This gives 54 fresh fits:
`3 specs x 6 scopes x 3 fits`. No V1--V3 fit or checkpoint is reused.

## Gate separation

### Independently refitted banks: 401 versus 801

Require, for every eligible specification:

- all cold fits converge under returned-iterate checks;
- full-scope cold and continuation fits agree;
- item difficulty and nonconstant discrimination rank agreement at least 0.99;
- exportability agreement at least 0.995;
- held-out log-loss and Brier differences remain within the existing `0.005`
  equivalence margin, including family-clustered intervals; and
- common-support theta median absolute shift at most `0.02` and p95 shift at
  most `0.05`.

Maximum theta shift is still recorded as a diagnostic but is not a refit-bank
gate. The tracked original remediation plan specified median and p95 here; the
later `0.005` maximum came from a same-bank numerical-integration check and made
those two limits redundant when copied into V2/V3.

No post-hoc affine linking can make a panel pass. Every fit uses the same one-
dimensional, positively oriented, standard-normal latent scale; linking is
report-only because it could hide solver drift.

### One fixed fitted bank: scoring resolution and bounds

After the refit gate passes, hold each 401-node bank fixed and compare:

- bound-8 normal-trapezoid EAP at 801 versus 1601 nodes; and
- equal-step bound-8/801 versus bound-10/1001.

For these same-bank checks, retain the strict maximum absolute theta shift of
`0.005` and maximum posterior tail mass of `1e-5`. This is the scientifically
appropriate place for a pure integration-accuracy maximum.

## Lock and downstream rule

V4 writes a numerical lock only if all three eligible specifications pass every
required fit, continuation, refit-bank, fixed-bank, and tail gate. The lock is:

- calibration integration: normal trapezoid, bound 8, 401 nodes;
- reference EAP scoring: normal trapezoid, bound 8, 801 nodes; and
- eligible specifications: exactly the three listed above.

Any failure produces a terminal blocked decision and no lock. Thresholds and
candidates cannot be changed inside the same output leaf.

Only a valid V4 lock may be consumed by a new append-only Phase-3 driver. That
driver retains the already-frozen primary CAT policy (floor 15, conditional SE
0.20, trace selector), its two non-promotable sensitivities, grouped repeated
nested cross-validation, the no-fallback rule, and all existing CAT acceptance
gates. Phase 3 runs only after V4 passes; figures are regenerated from whichever
terminal outcome is reached.
