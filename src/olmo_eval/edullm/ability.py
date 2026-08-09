"""Provider-independent quadrature, EAP, MWLE, and p-IRT diagnostics.

This module is a focused, behavior-preserving extraction from
``eduLLM-Evals/scripts/scenario_cat_lib.py`` on ``origin/frq/infobench``
(Git blob ``20bb1f6177e2ff95506f7a3a55d20d78c2a3f679``).  Only the numerical
ability-estimation primitives are included; file loading, benchmark schemas,
judge calls, and runner orchestration intentionally remain outside this layer.

All estimators use the fitted-bank convention ``sigmoid(A @ theta - b)``.
Missing response cells are excluded rather than converted to failures.  EAP is
computed from a joint posterior and supports a correlated multivariate-normal
prior; MWLE uses the source Warm/Firth weighted-likelihood objective.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log_expit, logsumexp, roots_hermite


class NumericalCoreError(RuntimeError):
    """Raised when a requested numerical rule is invalid or cannot be built."""


@dataclass(frozen=True)
class Quadrature:
    """Numerical ability grid and normalized prior-integration weights."""

    grid: np.ndarray
    log_prior: np.ndarray
    correlation: np.ndarray
    nodes_per_dim: int
    method: str = "gauss_hermite"
    lower_bound: float | None = None
    upper_bound: float | None = None


@dataclass
class AbilityEstimate:
    """An ability point estimate with marginal and joint uncertainty."""

    theta: np.ndarray
    se: np.ndarray
    covariance: np.ndarray
    method: str
    n_items: int
    converged: bool
    message: str = ""
    log_marginal: float | None = None


def _validate_correlation(value: np.ndarray, n_dims: int) -> np.ndarray:
    corr = np.asarray(value, dtype=float)
    if corr.shape != (n_dims, n_dims):
        raise NumericalCoreError(f"latent correlation shape {corr.shape} != ({n_dims}, {n_dims})")
    if not np.all(np.isfinite(corr)) or not np.allclose(corr, corr.T, atol=1e-9):
        raise NumericalCoreError("latent correlation must be finite and symmetric")
    if not np.allclose(np.diag(corr), 1.0, atol=1e-6):
        raise NumericalCoreError("latent correlation must have a unit diagonal")
    if float(np.linalg.eigvalsh(corr).min()) <= 1e-10:
        raise NumericalCoreError("latent correlation must be positive definite")
    return corr


def build_quadrature(
    n_dims: int,
    nodes_per_dim: int,
    correlation: np.ndarray | None = None,
    *,
    max_nodes: int = 50_000,
    method: str = "gauss_hermite",
    linear_bound: float = 8.0,
) -> Quadrature:
    """Build an explicit numerical rule for a standard-normal latent prior.

    ``gauss_hermite`` preserves the source NumPy tensor-product implementation.
    ``gauss_hermite_scipy`` uses SciPy's stable high-order root construction.
    ``normal_trapezoid`` provides a bounded, evenly spaced one-dimensional rule
    for posteriors narrower than practical Gauss-Hermite node spacing.

    The tensor grid grows as ``nodes_per_dim ** n_dims``; ``max_nodes`` guards
    against accidental, infeasibly large requests.
    """

    if n_dims < 1 or nodes_per_dim < 2:
        raise ValueError("n_dims must be >=1 and nodes_per_dim must be >=2")
    if method not in {
        "gauss_hermite",
        "gauss_hermite_scipy",
        "normal_trapezoid",
    }:
        raise ValueError(
            "quadrature method must be gauss_hermite, gauss_hermite_scipy, or normal_trapezoid"
        )
    total = nodes_per_dim**n_dims
    if total > max_nodes:
        raise NumericalCoreError(
            f"quadrature request {nodes_per_dim}^{n_dims}={total:,} exceeds max_nodes={max_nodes:,}"
        )
    corr = _validate_correlation(
        np.eye(n_dims) if correlation is None else np.asarray(correlation, dtype=float),
        n_dims,
    )

    if method == "normal_trapezoid":
        if n_dims != 1:
            raise NumericalCoreError("normal_trapezoid quadrature is supported only for 1D")
        if not math.isfinite(linear_bound) or linear_bound <= 0:
            raise ValueError("linear_bound must be a finite positive number")
        axis = np.linspace(-linear_bound, linear_bound, nodes_per_dim, dtype=float)
        step = float(axis[1] - axis[0])
        trapezoid = np.full(nodes_per_dim, step, dtype=float)
        trapezoid[[0, -1]] *= 0.5
        log_weights = -0.5 * axis * axis - 0.5 * math.log(2.0 * math.pi) + np.log(trapezoid)
        log_weights -= logsumexp(log_weights)
        return Quadrature(
            grid=axis[:, None],
            log_prior=log_weights,
            correlation=corr,
            nodes_per_dim=nodes_per_dim,
            method=method,
            lower_bound=-float(linear_bound),
            upper_bound=float(linear_bound),
        )

    if method == "gauss_hermite_scipy":
        nodes, weights = roots_hermite(nodes_per_dim)
        # High-order tail weights may underflow to zero.  Such nodes have no
        # numerical contribution and cannot be logged, so remove only those.
        positive = weights > 0
        nodes = nodes[positive]
        weights = weights[positive]
    else:
        nodes, weights = np.polynomial.hermite.hermgauss(nodes_per_dim)
    if not np.all(np.isfinite(nodes)) or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise NumericalCoreError(
            "Gauss-Hermite construction produced non-finite/nonpositive weights; "
            "use a supported node count or an explicitly configured method"
        )

    axis = math.sqrt(2.0) * nodes
    one_log_weight = np.log(weights) - 0.5 * math.log(math.pi)
    meshes = np.meshgrid(*([axis] * n_dims), indexing="ij")
    grid = np.stack([mesh.reshape(-1) for mesh in meshes], axis=1)
    weight_meshes = np.meshgrid(*([one_log_weight] * n_dims), indexing="ij")
    log_weights = np.sum(np.stack([mesh.reshape(-1) for mesh in weight_meshes], axis=1), axis=1)

    if not np.allclose(corr, np.eye(n_dims)):
        inverse = np.linalg.inv(corr)
        sign, log_determinant = np.linalg.slogdet(corr)
        if sign <= 0:
            raise NumericalCoreError("latent correlation determinant is not positive")
        ratio_quadratic = np.einsum("gi,ij,gj->g", grid, inverse - np.eye(n_dims), grid)
        # Importance reweighting: correlated MVN / independent MVN density.
        log_weights = log_weights - 0.5 * ratio_quadratic - 0.5 * log_determinant
    log_weights -= logsumexp(log_weights)
    return Quadrature(
        grid=grid,
        log_prior=log_weights,
        correlation=corr,
        nodes_per_dim=nodes_per_dim,
        method=method,
    )


def _observed_subset(
    responses: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    item_indices: Sequence[int] | np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(responses, dtype=float)
    loadings = np.asarray(A, dtype=float)
    difficulties = np.asarray(b, dtype=float)
    if loadings.ndim != 2:
        raise ValueError("A must be a two-dimensional loading matrix")
    if difficulties.shape != (loadings.shape[0],):
        raise ValueError("b must be a vector aligned to all rows of A")
    if values.ndim != 1 or values.shape[0] != loadings.shape[0]:
        raise ValueError("responses must be a vector aligned to all rows of A")
    indices = (
        np.arange(loadings.shape[0], dtype=int)
        if item_indices is None
        else np.asarray(item_indices, dtype=int)
    )
    if indices.ndim != 1 or (
        indices.size and (indices.min() < 0 or indices.max() >= loadings.shape[0])
    ):
        raise ValueError("item_indices are outside the fitted bank")
    keep = np.isfinite(values[indices])
    indices = indices[keep]
    y = values[indices]
    if not set(np.unique(y).tolist()) <= {0.0, 1.0}:
        raise ValueError("observed responses must be binary")
    return indices, y, loadings[indices], difficulties[indices]


def batch_eap(
    responses: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    quadrature: Quadrature,
    item_indices: Sequence[int] | np.ndarray | None = None,
    *,
    node_chunk: int = 4096,
) -> AbilityEstimate:
    """Return an order-invariant joint EAP estimate over observed items."""

    indices, y, loadings, difficulties = _observed_subset(responses, A, b, item_indices)
    if node_chunk < 1:
        raise ValueError("node_chunk must be >= 1")
    if quadrature.grid.ndim != 2 or quadrature.grid.shape[1] != np.asarray(A).shape[1]:
        raise ValueError("quadrature dimensions must match A")
    if quadrature.log_prior.shape != (quadrature.grid.shape[0],):
        raise ValueError("quadrature log_prior must align with its grid")

    n_nodes = quadrature.grid.shape[0]
    log_likelihood = np.zeros(n_nodes, dtype=float)
    for start in range(0, n_nodes, node_chunk):
        stop = min(start + node_chunk, n_nodes)
        eta = loadings @ quadrature.grid[start:stop].T - difficulties[:, None]
        log_likelihood[start:stop] = y @ log_expit(eta) + (1.0 - y) @ log_expit(-eta)
    joint = log_likelihood + quadrature.log_prior
    log_marginal = float(logsumexp(joint))
    posterior = np.exp(joint - log_marginal)
    # Explicit weighted reductions avoid spurious floating-point warnings from
    # some macOS Accelerate BLAS builds on row-vector matrix multiplication.
    theta = np.sum(posterior[:, None] * quadrature.grid, axis=0)
    centered = quadrature.grid - theta
    covariance = np.einsum("gi,gj,g->ij", centered, centered, posterior)
    covariance = (covariance + covariance.T) / 2.0
    se = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    return AbilityEstimate(
        theta=theta,
        se=se,
        covariance=covariance,
        method="batch_eap",
        n_items=int(indices.size),
        converged=True,
        log_marginal=log_marginal,
    )


def _mwle_neg_obj_grad(
    theta: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    y: np.ndarray,
    ridge: float,
) -> tuple[float, np.ndarray]:
    """Return the negative Warm/Firth objective and analytic gradient."""

    n_dims = theta.shape[0]
    eta = A @ theta - b
    probability = expit(eta)
    weights = probability * (1.0 - probability)
    information = (A * weights[:, None]).T @ A + ridge * np.eye(n_dims)
    sign, log_determinant = np.linalg.slogdet(information)
    if sign <= 0 or not np.isfinite(log_determinant):
        return 1e100, np.zeros(n_dims)
    log_likelihood = float(y @ log_expit(eta) + (1.0 - y) @ log_expit(-eta))
    objective = log_likelihood + 0.5 * log_determinant
    inverse_information = np.linalg.inv(information)
    curvature = weights * (1.0 - 2.0 * probability)
    leverage = np.einsum("ij,jk,ik->i", A, inverse_information, A)
    gradient = A.T @ (y - probability) + 0.5 * (A * (curvature * leverage)[:, None]).sum(axis=0)
    return -objective, -gradient


def mwle(
    responses: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    item_indices: Sequence[int] | np.ndarray | None = None,
    *,
    theta0: np.ndarray | None = None,
    ridge: float = 1e-6,
    bound: float = 12.0,
) -> AbilityEstimate:
    """Estimate multidimensional ability with a Warm/Firth adjustment.

    A rank-deficient administered set returns ``converged=False`` and an
    explicit message.  Its ``theta`` is the supplied start value and must not be
    treated as a successfully optimized estimate.
    """

    indices, y, loadings, difficulties = _observed_subset(responses, A, b, item_indices)
    n_dims = np.asarray(A).shape[1]
    start = np.zeros(n_dims) if theta0 is None else np.asarray(theta0, dtype=float)
    if start.shape != (n_dims,):
        raise ValueError(f"theta0 shape {start.shape} != ({n_dims},)")
    rank = int(np.linalg.matrix_rank(loadings)) if loadings.size else 0
    if indices.size < n_dims or rank < n_dims:
        return AbilityEstimate(
            theta=start.copy(),
            se=np.full(n_dims, np.nan),
            covariance=np.full((n_dims, n_dims), np.nan),
            method="mwle",
            n_items=int(indices.size),
            converged=False,
            message=f"administered loading matrix is rank {rank}/{n_dims}",
        )
    try:
        result = minimize(
            _mwle_neg_obj_grad,
            start,
            args=(loadings, difficulties, y, ridge),
            jac=True,
            method="L-BFGS-B",
            bounds=[(-bound, bound)] * n_dims,
        )
    except Exception as exc:  # pragma: no cover - SciPy failures are environment-specific
        return AbilityEstimate(
            theta=start.copy(),
            se=np.full(n_dims, np.nan),
            covariance=np.full((n_dims, n_dims), np.nan),
            method="mwle",
            n_items=int(indices.size),
            converged=False,
            message=f"optimizer raised {type(exc).__name__}: {exc}",
        )

    theta = np.asarray(result.x, dtype=float)
    probability = expit(loadings @ theta - difficulties)
    information = (
        loadings * (probability * (1.0 - probability))[:, None]
    ).T @ loadings + ridge * np.eye(n_dims)
    covariance = np.linalg.pinv(information)
    covariance = (covariance + covariance.T) / 2.0
    se = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    converged = bool(result.success and np.all(np.isfinite(theta)))
    return AbilityEstimate(
        theta=theta if np.all(np.isfinite(theta)) else start.copy(),
        se=se,
        covariance=covariance,
        method="mwle",
        n_items=int(indices.size),
        converged=converged,
        message="" if converged else str(result.message),
    )


def pass_rate_check(
    responses: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    theta: np.ndarray,
    item_indices: Sequence[int] | np.ndarray | None = None,
) -> dict[str, float | int]:
    """Compare observed and p-IRT-predicted pass rates on the same items.

    Only observed binary cells contribute, so missing judgments never become
    implicit failures and never alter the denominator.
    """

    indices, y, loadings, difficulties = _observed_subset(responses, A, b, item_indices)
    ability = np.asarray(theta, dtype=float)
    n_dims = np.asarray(A).shape[1]
    if ability.shape != (n_dims,):
        raise ValueError(f"theta shape {ability.shape} != ({n_dims},)")
    if indices.size == 0:
        return {
            "n_items": 0,
            "observed_pass_rate": float("nan"),
            "predicted_pass_rate": float("nan"),
            "error": float("nan"),
        }
    observed = float(y.mean())
    predicted = (
        float(expit(loadings @ ability - difficulties).mean())
        if np.all(np.isfinite(ability))
        else float("nan")
    )
    return {
        "n_items": int(indices.size),
        "observed_pass_rate": observed,
        "predicted_pass_rate": predicted,
        "error": predicted - observed,
    }
