"""Provider-independent multidimensional item-response-theory primitives.

This module is a behavior-preserving port of
``eduLLM-Evals/tutor_cat/mirt.py`` from ``origin/frq/infobench`` (Git blob
``44c13d5029536569d149ff69857571bfd46f58b7``).  It deliberately has no
benchmark, model-provider, or runner dependencies so OLMo Eval can call the
same numerical core from any EduLLM benchmark adapter.

The fitted-bank parameterization is ``sigmoid((q * a) @ theta - b)``.  The
online update functions implement the historical covariance/ability update;
``posterior_moments`` performs joint (not dimension-by-dimension) posterior
integration over an explicit grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# The historical TutorBench API defaulted to three skills.  Callers using a
# dynamic bank pass ``n_skills`` explicitly; retaining this fallback preserves
# compatibility without importing a benchmark-specific package constant.
_DEFAULT_N_SKILLS = 3

# Clip the logit to keep exp() finite; sigmoid(60) is 1.0 to double precision.
_Z_CLIP = 60.0


@dataclass(frozen=True)
class PosteriorMoments:
    """Grid-integrated posterior moments for an administered response set."""

    theta: np.ndarray
    covariance: np.ndarray
    se: np.ndarray
    log_marginal: float


def sigmoid(z: float | np.ndarray) -> float | np.ndarray:
    """Return a numerically stable logistic transform."""

    z = np.clip(z, -_Z_CLIP, _Z_CLIP)
    return 1.0 / (1.0 + np.exp(-z))


def masked_discrimination(a: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Return the Q-masked discrimination vector, ``q * a``."""

    return q * a


def pass_probability(theta: np.ndarray, a: np.ndarray, q: np.ndarray, b: float) -> float:
    """Return ``sigmoid((q * a) @ theta - b)`` for one fitted criterion."""

    loading = masked_discrimination(a, q)
    return float(sigmoid(float(loading @ theta) - b))


def update(
    theta: np.ndarray,
    covariance: np.ndarray,
    a: np.ndarray,
    q: np.ndarray,
    b: float,
    y: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Apply the historical online MIRT update for one binary verdict.

    The uncertainty update is applied first::

        U_new^-1 = U^-1 + p(1-p) outer(q * a, q * a)

    The ability update then uses that new covariance::

        theta_new = theta + U_new @ (q * a) * (y - p)

    Returns ``(theta_new, covariance_new, p)`` where ``p`` is the pre-update
    pass probability.
    """

    loading = masked_discrimination(a, q)
    p = pass_probability(theta, a, q, b)

    information = p * (1.0 - p) * np.outer(loading, loading)
    covariance_new = np.linalg.inv(np.linalg.inv(covariance) + information)
    # Wash out floating-point asymmetry introduced by matrix inversion.
    covariance_new = (covariance_new + covariance_new.T) / 2.0

    theta_new = theta + covariance_new @ loading * (float(y) - p)
    return theta_new, covariance_new, p


def standard_errors(covariance: np.ndarray) -> np.ndarray:
    """Return marginal standard errors from a posterior covariance matrix."""

    return np.sqrt(np.diag(covariance))


def posterior_moments(
    responses: np.ndarray,
    loadings: np.ndarray,
    difficulties: np.ndarray,
    grid: np.ndarray,
    log_prior: np.ndarray,
) -> PosteriorMoments:
    """Integrate a joint IRT posterior on an explicit fixed grid.

    ``loadings`` is already Q-masked and the likelihood therefore uses
    ``sigmoid(loadings @ theta - difficulties)``.  Empty response arrays return
    the prior moments represented by the supplied grid.
    """

    y = np.asarray(responses, dtype=float)
    A = np.asarray(loadings, dtype=float)
    b = np.asarray(difficulties, dtype=float)
    nodes = np.asarray(grid, dtype=float)
    prior = np.asarray(log_prior, dtype=float)

    if nodes.ndim != 2 or nodes.shape[0] < 2 or nodes.shape[1] < 1:
        raise ValueError("grid must have shape (n_nodes >= 2, n_dimensions >= 1)")
    n_items, n_dims = A.shape if A.ndim == 2 else (-1, -1)
    if A.ndim != 2 or n_dims != nodes.shape[1]:
        raise ValueError("loadings must have shape (n_items, grid_dimensions)")
    if y.shape != (n_items,) or b.shape != (n_items,):
        raise ValueError("responses and difficulties must align with loading rows")
    if prior.shape != (nodes.shape[0],):
        raise ValueError("log_prior must align with grid nodes")
    if not (
        np.all(np.isfinite(y))
        and np.all(np.isfinite(A))
        and np.all(np.isfinite(b))
        and np.all(np.isfinite(nodes))
        and not np.any(np.isnan(prior))
        and not np.any(np.isposinf(prior))
        and np.any(np.isfinite(prior))
    ):
        raise ValueError("posterior inputs must be finite (log_prior may contain -inf)")
    if not set(np.unique(y).tolist()) <= {0.0, 1.0}:
        raise ValueError("responses must be binary")

    log_posterior = prior.copy()
    if n_items:
        eta = nodes @ A.T - b[None, :]
        log_pass = -np.logaddexp(0.0, -eta)
        log_fail = -np.logaddexp(0.0, eta)
        log_posterior += (log_pass * y[None, :] + log_fail * (1.0 - y)[None, :]).sum(axis=1)

    peak = float(np.max(log_posterior))
    weights = np.exp(log_posterior - peak)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("posterior normalization failed")
    weights /= total
    # Explicit weighted reductions avoid spurious floating-point warnings from
    # some macOS Accelerate BLAS builds on row-vector matrix multiplication.
    theta = np.sum(weights[:, None] * nodes, axis=0)
    centered = nodes - theta
    covariance = np.einsum("gi,gj,g->ij", centered, centered, weights)
    covariance = (covariance + covariance.T) / 2.0
    se = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    return PosteriorMoments(
        theta=np.asarray(theta, dtype=float),
        covariance=np.asarray(covariance, dtype=float),
        se=np.asarray(se, dtype=float),
        log_marginal=peak + float(np.log(total)),
    )


def initial_state(
    theta_init: list[float] | None = None,
    u_init_diag: list[float] | None = None,
    n_skills: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the default zero ability and identity uncertainty state.

    ``n_skills`` defaults to the historical three-dimensional API.  Dynamic
    benchmark callers should pass the fitted bank's dimension count explicitly.
    """

    n = _DEFAULT_N_SKILLS if n_skills is None else int(n_skills)
    theta = np.zeros(n) if theta_init is None else np.asarray(theta_init, dtype=float)
    diag = np.ones(n) if u_init_diag is None else np.asarray(u_init_diag, dtype=float)
    if theta.shape != (n,) or diag.shape != (n,):
        raise ValueError(f"theta_init and u_init_diag must each have length {n}")
    return theta, np.diag(diag)
