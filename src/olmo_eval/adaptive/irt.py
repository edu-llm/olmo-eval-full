"""3PL Item Response Theory math for adaptive testing (ATLAS).

Pure numpy, provider-agnostic. Shared by the offline (Task) and online
(ExternalEval) adaptive-testing entry points so the two can never drift.

3PL item response function:

    P(correct | theta) = c + (1 - c) * sigmoid(a * (theta - b))

where ``a`` is discrimination, ``b`` difficulty, ``c`` pseudo-guessing.
ATLAS stores mirt-style ``(a1, d, g)``; convert with ``a = a1``,
``b = -d / a1``, ``c = g`` (see ``bank.py``).
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
        ``(theta, se)`` — posterior mean and standard deviation on the grid.
    """
    z = np.clip(a[None, :] * (THETA_NODES[:, None] - b[None, :]), -30, 30)
    p = np.clip(c[None, :] + (1 - c[None, :]) / (1.0 + np.exp(-z)), _EPS, 1 - _EPS)
    ll = (resp[None, :] * np.log(p) + (1 - resp)[None, :] * np.log(1 - p)).sum(axis=1)
    w = np.exp(ll - ll.max()) * _PRIOR_WEIGHTS
    w = w / w.sum()
    mean = float((THETA_NODES * w).sum())
    sd = float(np.sqrt(((THETA_NODES - mean) ** 2 * w).sum()))
    return mean, sd
