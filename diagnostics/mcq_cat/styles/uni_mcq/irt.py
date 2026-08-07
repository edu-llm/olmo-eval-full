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
"""

from __future__ import annotations

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
