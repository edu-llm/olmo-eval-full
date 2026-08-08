"""Unidimensional IRT math for the ``uni_mcq`` CAT style.

**Ported verbatim** from ``src/olmo_eval/adaptive/irt.py`` at commit
``6998270f51af47ffc5623ced960461ef28557118`` on the ``Research`` branch, which is the
implementation every published ATLAS CAT result was produced with. This branch has no
``src/olmo_eval/adaptive/`` to import, so the math is copied rather than reused. Do not
"improve" it: ``tests/test_irt_parity.py`` pins these values, because a silent drift
here would invalidate comparison against those results.

3PL item response function:

    P(correct | theta) = c + (1 - c) * sigmoid(a * (theta - b))

where ``a`` is discrimination, ``b`` difficulty, ``c`` pseudo-guessing. Upstream banks
store mirt-style ``(a1, d, g)``; the conversion ``a = a1``, ``b = -d / a1``, ``c = g``
happens once at vendoring time, so a loaded bank is already in ``(a, b, c)`` form.

**One implementation serves both 2PL and 3PL.** 2PL is the ``c = 0`` case of 3PL
exactly, for both quantities the CAT uses. The response function collapses to the bare
sigmoid, and Fisher information reduces algebraically::

    I_3pl = a^2 * (1-P)/P * ((P-c)/(1-c))^2
          = a^2 * (1-s)/s * s^2          when c = 0, so P = s
          = a^2 * s * (1-s)
          = I_2pl

which is the 2PL Fisher formula Research uses in ``Research/scripts/mcq_diagnostic.py``.
So there is no 2PL branch here, and ``tests/test_irt_parity.py`` asserts the reduction
numerically so nobody adds one. What is *not* interchangeable is the parameters: never
force ``c = 0`` on a bank whose ``a`` and ``b`` were estimated jointly with ``g``.

**Everything below :func:`eap_theta_se` is new and is not part of that port.** The MWLE
block adds a second point estimator over the same items and the same ``(a, b, c)``;
nothing above it changed, so every pinned value in ``tests/test_irt_parity.py`` still
holds and a run that does not ask for MWLE computes none of it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Fixed quadrature grid for EAP ability estimation (matches the ATLAS
# diagnostic reference implementation: 81 nodes on [-4, 4], standard-normal prior).
THETA_NODES: np.ndarray = np.linspace(-4.0, 4.0, 81)
_PRIOR_WEIGHTS: np.ndarray = np.exp(-0.5 * THETA_NODES**2)
_PRIOR_WEIGHTS = _PRIOR_WEIGHTS / _PRIOR_WEIGHTS.sum()

_EPS = 1e-6


def prob(theta: float, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """3PL success probability at a scalar ``theta`` for each item."""
    z = np.clip(a * (theta - b), -30, 30)
    return np.clip(c + (1 - c) / (1.0 + np.exp(-z)), _EPS, 1 - _EPS)


def fisher_info(theta: float, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Per-item Fisher information at ``theta`` under the 3PL model.

    ``I(theta) = a^2 * (1 - p) / p * ((p - c) / (1 - c))^2``. Items with the
    highest information are the most useful next questions in a CAT.
    """
    p = prob(theta, a, b, c)
    return (a**2) * ((1 - p) / p) * ((p - c) / (1 - c + 1e-9)) ** 2


def eap_theta_se(
    resp: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> tuple[float, float]:
    """Expected-a-posteriori ability estimate and its standard error.

    Args:
        resp: Binary responses (0/1) for the administered items.
        a, b, c: 3PL parameters for those same items (aligned to ``resp``).

    Returns:
        ``(theta, se)`` -- posterior mean and standard deviation on the grid.
    """
    z = np.clip(a[None, :] * (THETA_NODES[:, None] - b[None, :]), -30, 30)
    p = np.clip(c[None, :] + (1 - c[None, :]) / (1.0 + np.exp(-z)), _EPS, 1 - _EPS)
    ll = (resp[None, :] * np.log(p) + (1 - resp)[None, :] * np.log(1 - p)).sum(axis=1)
    w = np.exp(ll - ll.max()) * _PRIOR_WEIGHTS
    w = w / w.sum()
    mean = float((THETA_NODES * w).sum())
    sd = float(np.sqrt(((THETA_NODES - mean) ** 2 * w).sum()))
    return mean, sd


# -- MWLE ------------------------------------------------------------------------
#
# Warm's weighted likelihood estimator, in the Firth/Jeffreys form: maximise
#
#     ln L(theta) + 1/2 * ln I(theta)
#
# EAP's prior pulls every estimate toward 0, and the pull is largest exactly where the
# response pattern is most extreme -- so a weak checkpoint's theta is compressed upward
# and a strong one's downward. MWLE drops the prior and replaces it with a penalty on
# the *information the administered items carry* at the candidate ability. That penalty
# is what keeps the estimate finite where plain MLE diverges: on an all-wrong pattern
# ln L keeps rising as theta runs to -inf, but I(theta) collapses there, because every
# item has become a foregone conclusion and none of them can tell -8 from -12. The
# difference from a prior is what gets discouraged: a prior discourages abilities that
# are unusual in the population, this discourages abilities the instrument cannot
# resolve.
#
# The bank here is unidimensional, so "multidimensional WLE" degenerates to Warm's
# original: the penalty gradient d/dtheta [1/2 ln I] is J(theta) / (2 I(theta)), which is
# precisely the correction Warm's weight w(theta) = sqrt(I(theta)) encodes. The estimator
# is therefore the root of
#
#     S(theta) + J(theta) / (2 * (I(theta) + ridge)) = 0
#
# where S is the score, I the summed Fisher information over the administered items, and
# J = dI/dtheta.
#
# **Pure numpy, and a root find rather than L-BFGS-B.** The reference implementation this
# was recovered from (eduLLM-Evals ``scripts/regen_cat_figures.py``) solved a 2-3
# dimensional problem with ``scipy.optimize.minimize``. scipy is not a dependency of this
# package and the runtime image does not carry it, and in one dimension a bracketed
# bisection on the score equation is both shorter and unconditionally convergent, so it
# is what runs here. Same objective, same ridge, same bound, same fallback semantics.

#: The two values ``--ability-estimator`` accepts. ``batch_eap`` is the default and
#: reproduces today's reported number exactly: the online estimate already *is* EAP over
#: the whole administered set (see :meth:`..style.UniMcqStyle.estimate_ability`), so
#: re-running it at report time is the same arithmetic over the same inputs.
ESTIMATOR_BATCH_EAP = "batch_eap"
ESTIMATOR_BATCH_EAP_MWLE = "batch_eap+mwle"
ESTIMATORS = (ESTIMATOR_BATCH_EAP, ESTIMATOR_BATCH_EAP_MWLE)

#: Added to ``I(theta)`` before it is logged or inverted. Carried over from the recovered
#: implementation's ``mwle_ridge`` default, and it does the same job in one dimension as
#: it did in three: keeps the penalty finite where the administered set has no
#: information left, so the search does not divide by zero at the edge of its bracket.
MWLE_RIDGE = 1e-6

#: How far from 0 the estimate may travel. Also from the recovered implementation. The
#: information penalty is what actually pins the maximum; this is the backstop for a
#: pattern where it does not, and reaching it is reported as non-convergence rather than
#: returned as an answer -- a theta of -12 logits is not a measurement, it is the search
#: running out of room.
MWLE_BOUND = 12.0

#: Bracket width at which bisection stops, in logits. Four orders below anything a CAT
#: standard error resolves, and reached in about 55 halvings of the full [-12, 12] range.
MWLE_TOLERANCE = 1e-10

#: Caps both phases of the search. Bracketing doubles its step, so ~40 steps covers the
#: bound from any interior start; bisection needs ~55. Neither can loop forever, and a
#: run that somehow exhausts this is reported as non-convergence.
MWLE_MAX_STEPS = 200

#: First outward step when bracketing the root, before doubling.
_MWLE_FIRST_STEP = 0.25


@dataclass(frozen=True, slots=True)
class MwleResult:
    """What the MWLE solver found for one administered set.

    ``converged`` is the whole point of this being a record rather than two floats. When
    it is ``False`` the estimator did not produce an answer and ``theta`` is the
    ``theta0`` it was handed back unchanged -- which is a perfectly ordinary-looking
    number, indistinguishable from a converged one by inspection. Callers must branch on
    ``converged``, report the fallback as the fallback, and never publish ``theta`` from
    a record whose ``note`` is set.
    """

    theta: float
    standard_error: float
    converged: bool
    note: str = ""


def likelihood_slope(
    resp: np.ndarray, theta: float, a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> float:
    """``d/dtheta ln L(theta)`` for the 3PL, summed over the administered items.

    The statistical name for this is the *score*, which is taken in this package: to
    score an item is to grade it. Named for what it is instead.

    Differentiated from :func:`prob`, so the two cannot disagree about the model. Writing
    ``P = c + (1 - c) s`` with ``s`` the sigmoid gives ``s = (P - c)/(1 - c)`` and
    ``1 - s = (1 - P)/(1 - c)``, so::

        dP/dtheta      = a (P - c)(1 - P) / (1 - c)
        d ln L/dtheta  = sum_i a_i (y_i - P_i)(P_i - c_i) / (P_i (1 - c_i))

    At ``c = 0`` this is the familiar 2PL form ``sum_i a_i (y_i - P_i)``.
    """
    p = prob(theta, a, b, c)
    return float((a * (resp - p) * (p - c) / (p * (1 - c + 1e-9))).sum())


def information_slope(theta: float, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """``dI/dtheta`` at ``theta``, summed over the items -- Warm's ``J(theta)``.

    Differentiating :func:`fisher_info` through ``P``. With
    ``I_i = [a^2/(1 - c)^2] g(P)`` and ``g(P) = (1 - P)(P - c)^2 / P``::

        g(P)           = (P - c)^2 / P - (P - c)^2
        g'(P)          = 1 - c^2/P^2 - 2 (P - c)
        dI_i/dtheta    = a^3 (P - c)(1 - P) g'(P) / (1 - c)^3

    At ``c = 0`` this collapses to ``a^3 P (1 - P)(1 - 2P)``, which is the
    ``w (1 - 2p) a^3`` weighting the multidimensional reference implementation used.
    """
    p = prob(theta, a, b, c)
    slope = (1.0 - (c**2) / (p**2) - 2.0 * (p - c)) / ((1 - c + 1e-9) ** 3)
    return float(((a**3) * (p - c) * (1 - p) * slope).sum())


def penalized_slope(
    resp: np.ndarray,
    theta: float,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    *,
    ridge: float = MWLE_RIDGE,
) -> float:
    """Warm's bias-corrected score, whose root is the MWLE estimate.

    ``S(theta) + J(theta) / (2 (I(theta) + ridge))`` -- the derivative of
    ``ln L(theta) + 1/2 ln(I(theta) + ridge)``. The second term is what removes the
    O(1/n) bias that makes plain MLE overshoot on short tests, and what keeps the root
    finite on an all-correct or all-incorrect pattern.
    """
    info = float(fisher_info(theta, a, b, c).sum()) + ridge
    return likelihood_slope(resp, theta, a, b, c) + information_slope(theta, a, b, c) / (2.0 * info)


def mwle_theta_se(
    resp: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    *,
    theta0: float,
    ridge: float = MWLE_RIDGE,
    bound: float = MWLE_BOUND,
    tolerance: float = MWLE_TOLERANCE,
) -> MwleResult:
    """Solve Warm's score equation over the administered items, started from ``theta0``.

    ``theta0`` is the batch EAP estimate -- which is what makes the estimator "batch EAP
    + MWLE" rather than plain MWLE. It carries no evidential weight: it only decides
    which side of the root the bracket search starts on, and the solution is a property
    of ``(resp, a, b, c)`` alone. MWLE re-derives theta from the raw responses and does
    not consume the online estimate at all.

    The search is two bracketed phases, both bounded by :data:`MWLE_MAX_STEPS`. First
    walk outward from ``theta0`` in the direction the score points, doubling the step,
    until the score changes sign -- which brackets a downward crossing, hence a maximum
    of the penalized likelihood and not a minimum. Then bisect that bracket to
    ``tolerance``.

    Returns:
        An :class:`MwleResult`. ``converged`` is ``False``, ``theta`` is ``theta0``
        unchanged and ``note`` says why, in each of the cases the caller must not
        publish as an MWLE estimate:

        * nothing was administered, so there is no likelihood to maximize;
        * the administered items carry no more information at the root than the ridge
          does, so the objective is flat and theta is not identified -- the one-
          dimensional form of the reference implementation's near-singularity fallback;
        * the slope never changes sign inside ``+/- bound``, so the penalty failed to pin
          an interior maximum and the best point available is the edge of the search;
        * the arithmetic went non-finite, which on a bank with a degenerate ``a`` or a
          ``c`` at 1 it can;
        * the step budget ran out.
    """
    resp = np.asarray(resp, dtype=float)
    theta0 = float(theta0)
    if resp.size == 0:
        return MwleResult(theta0, float("inf"), False, "no items were administered")

    def se_at(theta: float) -> float:
        info = float(fisher_info(theta, a, b, c).sum()) + ridge
        return float(np.sqrt(1.0 / info)) if info > 0 else float("inf")

    def fail(note: str) -> MwleResult:
        return MwleResult(theta0, se_at(theta0), False, note)

    def solved(theta: float) -> MwleResult:
        """A root, once it has been shown to be one the administered set identifies.

        The reference implementation's near-singularity fallback, in one dimension. Its
        multidimensional form refused when ``I(theta)`` was too close to singular to
        penalize -- a subset that loaded on almost nothing in some direction leaves that
        direction unidentified, and the maximum it reports there is the ridge's, not the
        data's. Here the whole information matrix is one number, so "singular" means
        the administered items carry no more information than the ridge does: the
        objective is flat, every theta fits equally well, and the root the search
        happened to land on is the seed rather than a measurement.
        """
        info = float(fisher_info(theta, a, b, c).sum())
        if info <= ridge:
            return fail(
                f"the administered items carry information {info:.3g} at theta="
                f"{theta:.6g}, no more than the ridge {ridge:g}, so the penalized "
                f"likelihood is flat and theta is not identified"
            )
        return MwleResult(float(theta), se_at(theta), True)

    start = float(np.clip(theta0, -bound, bound))
    first = penalized_slope(resp, start, a, b, c, ridge=ridge)
    if not np.isfinite(first):
        return fail(f"the penalized slope is not finite at theta0={theta0:.6g}")
    if first == 0.0:
        return solved(start)

    # Phase 1: walk outward from theta0 in the direction the slope points, until it
    # flips sign. A positive slope means the penalized likelihood is still climbing, so
    # the maximum lies to the right; a negative one, to the left.
    rising = first > 0.0
    direction = 1.0 if rising else -1.0
    step = _MWLE_FIRST_STEP
    bracket: tuple[float, float] | None = None
    for _ in range(MWLE_MAX_STEPS):
        far = float(np.clip(start + direction * step, -bound, bound))
        far_slope = penalized_slope(resp, far, a, b, c, ridge=ridge)
        if not np.isfinite(far_slope):
            return fail(f"the penalized slope is not finite at theta={far:.6g}")
        if far_slope == 0.0:
            return solved(far)
        if (far_slope > 0.0) is not rising:
            # lo always carries the positive slope, hi the negative one, so the root
            # between them is a downward crossing -- a maximum of ln L + 1/2 ln I and
            # not a minimum.
            bracket = (start, far) if rising else (far, start)
            break
        if abs(far) >= bound:
            break
        step *= 2.0

    if bracket is None:
        return fail(
            f"the penalized slope does not change sign inside +/-{bound:g}, so the "
            f"information penalty did not pin an interior maximum for this response "
            f"pattern"
        )

    # Phase 2: bisect the bracket. lo < hi numerically and slope(lo) > 0 > slope(hi).
    lo, hi = bracket
    for _ in range(MWLE_MAX_STEPS):
        if hi - lo <= tolerance:
            break
        mid = 0.5 * (lo + hi)
        mid_slope = penalized_slope(resp, mid, a, b, c, ridge=ridge)
        if not np.isfinite(mid_slope):
            return fail(f"the penalized slope is not finite at theta={mid:.6g}")
        if mid_slope > 0.0:
            lo = mid
        else:
            hi = mid
    else:
        return fail(f"bisection did not reach {tolerance:g} within {MWLE_MAX_STEPS} steps")

    theta = 0.5 * (lo + hi)
    if not (np.isfinite(theta) and np.isfinite(se_at(theta))):
        return fail("the solved estimate or its standard error is not finite")
    return solved(theta)
