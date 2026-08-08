"""Core MIRT math: PRD Equations 1-3.

Notation (PRD "Updating Running MIRT Vector"):
    theta : (3,) ability vector (content, diagnosis, scaffolding)
    a     : (3,) calibrated discrimination vector of a criterion (frozen)
    q     : (3,) Q-matrix row of a criterion, entries in {0,1} (frozen)
    b     : scalar calibrated difficulty (frozen)
    y     : judge outcome, 1 = pass, 0 = fail
    U     : (3,3) uncertainty (posterior covariance) matrix; SE_k = sqrt(U_kk)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import N_SKILLS

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
    z = np.clip(z, -_Z_CLIP, _Z_CLIP)
    return 1.0 / (1.0 + np.exp(-z))


def masked_discrimination(a: np.ndarray, q: np.ndarray) -> np.ndarray:
    """q ⊙ a — the Q-masked discrimination vector."""
    return q * a


def pass_probability(theta: np.ndarray, a: np.ndarray, q: np.ndarray, b: float) -> float:
    """Equation 1:  p = sigmoid( (q ⊙ a) · theta − b )"""
    m = masked_discrimination(a, q)
    return float(sigmoid(float(m @ theta) - b))


def update(
    theta: np.ndarray,
    U: np.ndarray,
    a: np.ndarray,
    q: np.ndarray,
    b: float,
    y: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Apply Equations 1-3 for one criterion verdict.

    Equation 2 (uncertainty update, applied FIRST):
        U_new^-1 = U^-1 + p(1−p) · outer(q ⊙ a)
    Equation 3 (ability update, uses U_new):
        theta_new = theta + U_new · (q ⊙ a) · (y − p)

    Returns (theta_new, U_new, p) where p is the pre-update pass probability.
    """
    m = masked_discrimination(a, q)
    p = pass_probability(theta, a, q, b)

    info = p * (1.0 - p) * np.outer(m, m)
    U_new = np.linalg.inv(np.linalg.inv(U) + info)
    # Symmetrize to wash out floating-point asymmetry from the inversions.
    U_new = (U_new + U_new.T) / 2.0

    theta_new = theta + U_new @ m * (float(y) - p)
    return theta_new, U_new, p


def standard_errors(U: np.ndarray) -> np.ndarray:
    """SE_k = sqrt(U_kk) for each of the 3 skills."""
    return np.sqrt(np.diag(U))


def posterior_moments(
    responses: np.ndarray,
    loadings: np.ndarray,
    difficulties: np.ndarray,
    grid: np.ndarray,
    log_prior: np.ndarray,
) -> PosteriorMoments:
    """Integrate the joint IRT posterior on an explicit fixed grid.

    The scenario-level bank uses ``sigmoid(A @ theta - b)`` where ``A`` is
    already Q-masked.  ``grid`` therefore has shape ``(G, D)`` and the returned
    standard errors are the marginal posterior SDs from the *joint* posterior.
    Empty response arrays return the prior moments represented by the grid.
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
        log_posterior += (log_pass * y[None, :] + log_fail * (1.0 - y)[None, :]).sum(
            axis=1
        )

    peak = float(np.max(log_posterior))
    weights = np.exp(log_posterior - peak)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("posterior normalization failed")
    weights /= total
    theta = weights @ nodes
    centered = nodes - theta
    covariance = (centered * weights[:, None]).T @ centered
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
    """PRD initial conditions: theta_0 = 0, U_0 = I over ``n_skills`` dimensions.

    ``n_skills`` defaults to the package's ``N_SKILLS`` (3) so the standard run is
    unchanged; callers modelling a different latent dimensionality pass it explicitly.
    """
    n = N_SKILLS if n_skills is None else int(n_skills)
    theta = np.zeros(n) if theta_init is None else np.asarray(theta_init, dtype=float)
    diag = np.ones(n) if u_init_diag is None else np.asarray(u_init_diag, dtype=float)
    if theta.shape != (n,) or diag.shape != (n,):
        raise ValueError(f"theta_init and u_init_diag must each have length {n}")
    return theta, np.diag(diag)
