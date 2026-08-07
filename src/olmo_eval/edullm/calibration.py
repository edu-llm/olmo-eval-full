"""Provider-independent confirmatory MIRT calibration for EduLLM.

The numerical implementation is a behavior-preserving library extraction of the
returned-iterate fitter in
``origin/frq/infobench:eduLLM-Evals/scripts/calibrate_mirt.py`` at commit
``b4ea2e8e4b1da75067dd5e38ea5e83270776a07f`` (Git blob
``7e2b05ee0996818e6d27aaa3be0b05e18721d7a7``).

The model is ``sigmoid(A @ theta - b)``.  ``A`` is masked by the modeled Q matrix,
so inactive loadings remain exactly zero.  Observed Pass/Fail cells are encoded as
1/0; missing or no-decision cells are excluded from every likelihood calculation.

This module deliberately owns no files, dataframes, model providers, CLI, reports,
or plots.  Scientific choices are explicit inputs: callers must provide a model
family, skill structure, quadrature specification, convergence policy, and missing
cell policy.  In particular, there is no default calibration family.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, cast

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log_expit, logsumexp

from .skill_structure import SkillStructure

SOURCE_BRANCH = "origin/frq/infobench"
SOURCE_COMMIT = "b4ea2e8e4b1da75067dd5e38ea5e83270776a07f"
SOURCE_BLOB = "7e2b05ee0996818e6d27aaa3be0b05e18721d7a7"

FREE_2PL = "free-2pl"
ONE_PL = "1pl"
LOG_SHRINKAGE_2PL = "log-shrinkage-2pl"
CalibrationFamily = Literal["free-2pl", "1pl", "log-shrinkage-2pl"]
CALIBRATION_FAMILIES: tuple[CalibrationFamily, ...] = (
    FREE_2PL,
    ONE_PL,
    LOG_SHRINKAGE_2PL,
)

GAUSS_HERMITE_QUADRATURE = "gauss_hermite"
NORMAL_TRAPEZOID_QUADRATURE = "normal_trapezoid"
QuadratureMethod = Literal["gauss_hermite", "normal_trapezoid"]
QUADRATURE_METHODS: tuple[QuadratureMethod, ...] = (
    GAUSS_HERMITE_QUADRATURE,
    NORMAL_TRAPEZOID_QUADRATURE,
)

LEGACY_PRE_MSTEP_CONVERGENCE = "legacy_pre_mstep"
RETURNED_ITERATE_CONVERGENCE = "returned_iterate"
ConvergenceMode = Literal["legacy_pre_mstep", "returned_iterate"]
CONVERGENCE_MODES: tuple[ConvergenceMode, ...] = (
    LEGACY_PRE_MSTEP_CONVERGENCE,
    RETURNED_ITERATE_CONVERGENCE,
)

MissingCellPolicy = Literal["exclude"]
EXCLUDE_MISSING: MissingCellPolicy = "exclude"
DEFAULT_MAX_GRID_NODES = 5_000


class CalibrationError(RuntimeError):
    """Raised when calibration inputs or numerical state violate the contract."""


@dataclass(frozen=True, slots=True)
class QuadratureConfig:
    """An explicit calibration integration rule with an allocation guard."""

    method: QuadratureMethod
    nodes_per_dimension: int
    linear_bound: float | None
    max_total_nodes: int = DEFAULT_MAX_GRID_NODES
    allow_large_grid: bool = False

    def __post_init__(self) -> None:
        if self.method not in QUADRATURE_METHODS:
            raise CalibrationError(
                f"unknown calibration quadrature {self.method!r}; "
                f"choose one of {list(QUADRATURE_METHODS)}"
            )
        if (
            isinstance(self.nodes_per_dimension, bool)
            or not isinstance(self.nodes_per_dimension, (int, np.integer))
            or self.nodes_per_dimension < 2
        ):
            raise CalibrationError("nodes_per_dimension must be an integer of at least 2")
        if (
            isinstance(self.max_total_nodes, bool)
            or not isinstance(self.max_total_nodes, (int, np.integer))
            or self.max_total_nodes < 1
        ):
            raise CalibrationError("max_total_nodes must be a positive integer")
        if self.method == NORMAL_TRAPEZOID_QUADRATURE:
            if self.linear_bound is None:
                raise CalibrationError("normal_trapezoid requires an explicit finite linear_bound")
            if not np.isfinite(self.linear_bound) or self.linear_bound <= 0:
                raise CalibrationError("linear_bound must be finite and strictly positive")
        elif self.linear_bound is not None:
            raise CalibrationError("linear_bound applies only to normal_trapezoid")

    def total_nodes(self, n_dimensions: int) -> int:
        """Validate dimensions and return the number of allocated quadrature nodes."""

        if (
            isinstance(n_dimensions, bool)
            or not isinstance(n_dimensions, (int, np.integer))
            or n_dimensions < 1
        ):
            raise CalibrationError("n_dimensions must be a positive integer")
        if self.method == NORMAL_TRAPEZOID_QUADRATURE:
            if n_dimensions != 1:
                raise CalibrationError(
                    "normal_trapezoid calibration quadrature is supported only for 1D"
                )
            total = self.nodes_per_dimension
        else:
            total = self.nodes_per_dimension**n_dimensions
        if total > self.max_total_nodes and not self.allow_large_grid:
            raise CalibrationError(
                f"requested quadrature has {total:,} nodes, above the safety cap "
                f"{self.max_total_nodes:,}; reduce nodes/dimensions or explicitly allow it"
            )
        return total

    def as_dict(self, n_dimensions: int) -> dict[str, Any]:
        return {
            "method": self.method,
            "nodes_per_dimension": self.nodes_per_dimension,
            "total_nodes": self.total_nodes(n_dimensions),
            "linear_bound": self.linear_bound,
            "max_total_nodes": self.max_total_nodes,
            "allow_large_grid": self.allow_large_grid,
        }


@dataclass(frozen=True, slots=True)
class ConvergencePolicy:
    """Explicit EM termination policy."""

    mode: ConvergenceMode
    max_iterations: int
    objective_tolerance: float
    parameter_tolerance: float | None = None
    consecutive_passes: int = 1

    def __post_init__(self) -> None:
        if self.mode not in CONVERGENCE_MODES:
            raise CalibrationError(
                f"unknown convergence mode {self.mode!r}; choose one of {list(CONVERGENCE_MODES)}"
            )
        if (
            isinstance(self.max_iterations, bool)
            or not isinstance(self.max_iterations, (int, np.integer))
            or self.max_iterations < 1
        ):
            raise CalibrationError("max_iterations must be a positive integer")
        if not np.isfinite(self.objective_tolerance) or self.objective_tolerance <= 0:
            raise CalibrationError("objective_tolerance must be finite and strictly positive")
        if self.parameter_tolerance is not None and (
            not np.isfinite(self.parameter_tolerance) or self.parameter_tolerance <= 0
        ):
            raise CalibrationError("parameter_tolerance must be finite and strictly positive")
        if (
            isinstance(self.consecutive_passes, bool)
            or not isinstance(self.consecutive_passes, (int, np.integer))
            or self.consecutive_passes < 1
        ):
            raise CalibrationError("consecutive_passes must be a positive integer")

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "max_iterations": self.max_iterations,
            "objective_tolerance": self.objective_tolerance,
            "parameter_tolerance": self.parameter_tolerance,
            "consecutive_passes": self.consecutive_passes,
        }


@dataclass(frozen=True, slots=True)
class CalibrationConfig:
    """Scientific and numerical choices for one calibration fit.

    ``model_family``, ``quadrature``, and ``convergence`` intentionally have no
    defaults.  ``ridge`` is used only by free-2PL; ``log_a_shrinkage`` is used only
    by positive log-shrinkage 2PL.
    """

    model_family: CalibrationFamily
    quadrature: QuadratureConfig
    convergence: ConvergencePolicy
    estimate_latent_correlation: bool
    ridge: float = 1e-3
    log_a_shrinkage: float = 1.0

    def __post_init__(self) -> None:
        if self.model_family not in CALIBRATION_FAMILIES:
            raise CalibrationError(
                f"unknown calibration family {self.model_family!r}; "
                f"choose one of {list(CALIBRATION_FAMILIES)}"
            )
        if not np.isfinite(self.ridge) or self.ridge < 0:
            raise CalibrationError("ridge must be finite and non-negative")
        if not np.isfinite(self.log_a_shrinkage) or self.log_a_shrinkage <= 0:
            raise CalibrationError("log_a_shrinkage must be finite and strictly positive")


@dataclass(frozen=True, slots=True)
class CalibrationProvenance:
    """Auditable provenance for one in-memory fit."""

    source_branch: str
    source_commit: str
    source_blob: str
    parameterization: str
    model_specification: Mapping[str, Any]
    skill_structure: Mapping[str, Any]
    quadrature: Mapping[str, Any]
    convergence_policy: Mapping[str, Any]
    missing_cell_policy: MissingCellPolicy
    n_persons_supplied: int
    n_persons_fitted: int
    n_all_missing_persons_excluded: int
    n_items: int
    n_observed_cells: int
    n_missing_cells: int
    specification_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_branch": self.source_branch,
            "source_commit": self.source_commit,
            "source_blob": self.source_blob,
            "parameterization": self.parameterization,
            "model_specification": dict(self.model_specification),
            "skill_structure": dict(self.skill_structure),
            "quadrature": dict(self.quadrature),
            "convergence_policy": dict(self.convergence_policy),
            "missing_cell_policy": self.missing_cell_policy,
            "n_persons_supplied": self.n_persons_supplied,
            "n_persons_fitted": self.n_persons_fitted,
            "n_all_missing_persons_excluded": self.n_all_missing_persons_excluded,
            "n_items": self.n_items,
            "n_observed_cells": self.n_observed_cells,
            "n_missing_cells": self.n_missing_cells,
            "specification_sha256": self.specification_sha256,
        }


@dataclass(frozen=True, slots=True)
class CalibrationFitResult:
    """Fitted parameters plus an explicit convergence/returned-iterate status."""

    loadings: np.ndarray
    difficulties: np.ndarray
    latent_correlation: np.ndarray
    marginal_log_likelihood: float
    penalized_objective: float
    n_parameters: int
    n_free_loadings: int
    n_fixed_loadings: int
    iterations: int
    converged: bool
    termination_reason: Literal["converged", "max_iterations"]
    returned_iterate_status: Literal[
        "converged_returned_iterate",
        "converged_legacy_iterate",
        "last_valid_returned_iterate",
    ]
    convergence_diagnostics: Mapping[str, Any]
    provenance: CalibrationProvenance

    def __post_init__(self) -> None:
        for name in ("loadings", "difficulties", "latent_correlation"):
            value = np.asarray(getattr(self, name), dtype=float).copy()
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "convergence_diagnostics",
            MappingProxyType(dict(self.convergence_diagnostics)),
        )


def _calibration_specification(
    calibration_model: CalibrationFamily,
    *,
    ridge: float,
    log_a_shrinkage: float,
) -> dict[str, Any]:
    """Return the canonical source-compatible item calibration specification."""

    if calibration_model == FREE_2PL:
        discrimination = {
            "mode": "estimated",
            "parameterization": "unconstrained-linear-loading",
            "constraint": None,
        }
        regularization = {
            "kind": "loading-l2-ridge",
            "ridge": float(ridge),
            "log_a_shrinkage": None,
            "log_a_prior_sd": None,
        }
    elif calibration_model == ONE_PL:
        discrimination = {
            "mode": "fixed",
            "parameterization": "fixed-loading",
            "constraint": "a=1 for every active Q loading",
            "fixed_a": 1.0,
        }
        regularization = {
            "kind": "none-fixed-loading",
            "ridge": None,
            "log_a_shrinkage": None,
            "log_a_prior_sd": None,
        }
    else:
        discrimination = {
            "mode": "estimated",
            "parameterization": "a=exp(log_a)",
            "constraint": "strictly-positive",
        }
        regularization = {
            "kind": "zero-mean-gaussian-prior-on-log-a",
            "ridge": None,
            "log_a_shrinkage": float(log_a_shrinkage),
            "log_a_prior_sd": float(1.0 / np.sqrt(log_a_shrinkage)),
            "penalty": "0.5 * log_a_shrinkage * sum(log(a)^2)",
            "center": "log(a)=0 (a=1)",
        }
    semantic = {
        "schema_version": 1,
        "family": calibration_model,
        "estimation": "bock-aitkin-marginal-maximum-likelihood-em",
        "discrimination": discrimination,
        "difficulty": {
            "mode": "estimated",
            "parameterization": "unpenalized-offset-b",
        },
        "regularization": regularization,
    }
    encoded = json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode()
    return {
        **semantic,
        "cache_key": f"calibration-spec-v1-{hashlib.sha256(encoded).hexdigest()}",
    }


def _build_grid(n_dims: int, nodes_per_dim: int) -> np.ndarray:
    axis, _ = np.polynomial.hermite_e.hermegauss(nodes_per_dim)
    mesh = np.meshgrid(*([axis] * n_dims), indexing="ij")
    return np.stack([part.reshape(-1) for part in mesh], axis=1).astype(float)


def _base_log_weights(n_dims: int, nodes_per_dim: int) -> np.ndarray:
    _, weights = np.polynomial.hermite_e.hermegauss(nodes_per_dim)
    log_weights = np.log(weights / np.sqrt(2.0 * np.pi))
    mesh = np.meshgrid(*([log_weights] * n_dims), indexing="ij")
    return np.sum(np.stack([part.reshape(-1) for part in mesh], axis=1), axis=1)


def _build_quadrature(
    n_dims: int, specification: QuadratureConfig
) -> tuple[np.ndarray, np.ndarray]:
    specification.total_nodes(n_dims)
    nodes = specification.nodes_per_dimension
    if specification.method == GAUSS_HERMITE_QUADRATURE:
        return _build_grid(n_dims, nodes), _base_log_weights(n_dims, nodes)

    assert specification.linear_bound is not None
    axis = np.linspace(-specification.linear_bound, specification.linear_bound, nodes)
    step = float(axis[1] - axis[0])
    trapezoid = np.full(nodes, step, dtype=float)
    trapezoid[[0, -1]] *= 0.5
    log_weights = -0.5 * axis * axis - 0.5 * np.log(2.0 * np.pi) + np.log(trapezoid)
    log_weights -= logsumexp(log_weights)
    return axis[:, None], log_weights


def _prior_log_weights(
    grid: np.ndarray, base_log_weights: np.ndarray, correlation: np.ndarray
) -> np.ndarray:
    n_dims = grid.shape[1]
    correlation = np.asarray(correlation, dtype=float)
    if correlation.shape != (n_dims, n_dims) or not np.allclose(
        correlation, correlation.T, atol=1e-10
    ):
        raise CalibrationError("latent correlation has the wrong shape or is not symmetric")
    if np.allclose(correlation, np.eye(n_dims)):
        log_weights = base_log_weights
    else:
        if float(np.min(np.linalg.eigvalsh(correlation))) <= 0:
            raise CalibrationError("latent correlation is not positive definite")
        inverse = np.linalg.inv(correlation)
        sign, log_determinant = np.linalg.slogdet(correlation)
        if sign <= 0 or not np.isfinite(log_determinant):
            raise CalibrationError("latent correlation has a non-positive determinant")
        quadratic = np.einsum("gi,ij,gj->g", grid, inverse - np.eye(n_dims), grid)
        log_weights = base_log_weights - 0.5 * quadratic - 0.5 * log_determinant
    return log_weights - logsumexp(log_weights)


def _project_correlation(matrix: np.ndarray, eigen_floor: float = 1e-6) -> np.ndarray:
    raw = np.asarray(matrix, dtype=float)
    symmetric = (raw + raw.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    positive = (eigenvectors * np.maximum(eigenvalues, eigen_floor)) @ eigenvectors.T
    scale = np.sqrt(np.clip(np.diag(positive), eigen_floor, None))
    correlation = positive / np.outer(scale, scale)
    correlation = (correlation + correlation.T) / 2.0
    np.fill_diagonal(correlation, 1.0)
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    if float(eigenvalues.min()) <= 0:
        correlation = (eigenvectors * np.maximum(eigenvalues, eigen_floor)) @ eigenvectors.T
        scale = np.sqrt(np.clip(np.diag(correlation), eigen_floor, None))
        correlation = correlation / np.outer(scale, scale)
        correlation = (correlation + correlation.T) / 2.0
        np.fill_diagonal(correlation, 1.0)
    return correlation


def _item_neg_log_likelihood(
    beta: np.ndarray,
    design: np.ndarray,
    successes: np.ndarray,
    counts: np.ndarray,
    ridge: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    eta = design @ beta
    negative_log_likelihood = -(
        np.dot(successes, log_expit(eta)) + np.dot(counts - successes, log_expit(-eta))
    )
    probability = expit(eta)
    gradient = design.T @ (counts * probability - successes)
    weights = counts * probability * (1.0 - probability)
    hessian = (design * weights[:, None]).T @ design
    if ridge > 0:
        penalty_mask = np.ones(beta.shape[0])
        penalty_mask[-1] = 0.0
        negative_log_likelihood += 0.5 * ridge * float(np.dot(penalty_mask * beta, beta))
        gradient += ridge * penalty_mask * beta
        hessian += ridge * np.diag(penalty_mask)
    return float(negative_log_likelihood), gradient, hessian


def _fit_free_item(
    design: np.ndarray,
    successes: np.ndarray,
    counts: np.ndarray,
    ridge: float,
    *,
    initial: np.ndarray | None,
    return_diagnostics: bool,
    max_newton: int = 50,
    tolerance: float = 1e-8,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    n_parameters = design.shape[1]
    if initial is None:
        beta = np.zeros(n_parameters)
        total = float(counts.sum())
        if total > 0:
            pass_rate = np.clip(float(successes.sum()) / total, 1e-3, 1 - 1e-3)
            beta[-1] = np.log(pass_rate / (1.0 - pass_rate))
    else:
        beta = np.asarray(initial, dtype=float)
        if beta.shape != (n_parameters,) or not np.all(np.isfinite(beta)):
            raise CalibrationError(f"item warm start must be finite with shape {(n_parameters,)}")
        beta = beta.copy()

    objective, gradient, hessian = _item_neg_log_likelihood(beta, design, successes, counts, ridge)
    converged = False
    line_search_failed = False
    n_iterations = 0
    for iteration in range(max_newton):
        n_iterations = iteration + 1
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        alpha = 1.0
        for _ in range(30):
            candidate = beta - alpha * step
            new_objective, new_gradient, new_hessian = _item_neg_log_likelihood(
                candidate, design, successes, counts, ridge
            )
            if np.isfinite(new_objective) and new_objective <= objective + 1e-12:
                break
            alpha *= 0.5
        else:
            line_search_failed = True
            break
        if (
            abs(objective - new_objective) < tolerance
            and np.max(np.abs(beta - candidate)) < tolerance
        ):
            beta = candidate
            objective = new_objective
            gradient = new_gradient
            hessian = new_hessian
            converged = True
            break
        beta = candidate
        objective = new_objective
        gradient = new_gradient
        hessian = new_hessian

    if not return_diagnostics:
        return beta
    return beta, {
        "converged": converged,
        "line_search_failed": line_search_failed,
        "n_iter": n_iterations,
        "objective": float(objective),
        "max_abs_gradient": float(np.max(np.abs(gradient))),
    }


def _fit_fixed_loading_item(
    offset: np.ndarray,
    successes: np.ndarray,
    counts: np.ndarray,
    *,
    max_newton: int = 50,
    tolerance: float = 1e-8,
) -> float:
    offset = np.asarray(offset, dtype=float)
    successes = np.asarray(successes, dtype=float)
    counts = np.asarray(counts, dtype=float)
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    pass_rate = np.clip(float(successes.sum()) / total, 1e-3, 1 - 1e-3)
    intercept = float(np.log(pass_rate / (1.0 - pass_rate)) - np.average(offset, weights=counts))

    def objective(value: float) -> float:
        eta = offset + value
        return float(
            -(np.dot(successes, log_expit(eta)) + np.dot(counts - successes, log_expit(-eta)))
        )

    negative_log_likelihood = objective(intercept)
    for _ in range(max_newton):
        eta = offset + intercept
        probability = expit(eta)
        gradient = float(np.dot(counts, probability) - successes.sum())
        hessian = float(np.dot(counts, probability * (1.0 - probability)))
        if not np.isfinite(hessian) or hessian <= 1e-14:
            break
        step = gradient / hessian
        alpha = 1.0
        for _ in range(30):
            candidate = intercept - alpha * step
            candidate_objective = objective(candidate)
            if (
                np.isfinite(candidate_objective)
                and candidate_objective <= negative_log_likelihood + 1e-12
            ):
                break
            alpha *= 0.5
        else:
            break
        if (
            abs(negative_log_likelihood - candidate_objective) < tolerance
            and abs(intercept - candidate) < tolerance
        ):
            intercept = candidate
            break
        intercept = candidate
        negative_log_likelihood = candidate_objective
    return float(intercept)


def _fit_fixed_loading_items_1d(
    offset: np.ndarray,
    successes: np.ndarray,
    counts: np.ndarray,
    *,
    initial_intercepts: np.ndarray,
    max_newton: int = 50,
    tolerance: float = 1e-8,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Source-parity vectorized returned-iterate 1PL M-step."""

    offset = np.asarray(offset, dtype=float)
    successes = np.asarray(successes, dtype=float)
    counts = np.asarray(counts, dtype=float)
    if offset.ndim != 1 or successes.ndim != 2 or counts.shape != successes.shape:
        raise CalibrationError("batched 1PL inputs have incompatible shapes")
    if successes.shape[1] != offset.size:
        raise CalibrationError("batched 1PL node axis does not match the offset")
    n_items = successes.shape[0]
    total = counts.sum(axis=1)
    intercept = np.asarray(initial_intercepts, dtype=float)
    if intercept.shape != (n_items,) or not np.all(np.isfinite(intercept)):
        raise CalibrationError(f"batched 1PL warm start must be finite with shape {(n_items,)}")
    intercept = intercept.copy()
    intercept[total <= 0] = 0.0

    def objective(values: np.ndarray) -> np.ndarray:
        eta = offset[None, :] + values[:, None]
        return -(
            (successes * log_expit(eta)).sum(axis=1)
            + ((counts - successes) * log_expit(-eta)).sum(axis=1)
        )

    negative_log_likelihood = objective(intercept)
    line_search_failures = np.zeros(n_items, dtype=bool)
    converged = np.zeros(n_items, dtype=bool)
    n_iterations = 0
    for iteration in range(max_newton):
        n_iterations = iteration + 1
        eta = offset[None, :] + intercept[:, None]
        probability = expit(eta)
        gradient = (counts * probability - successes).sum(axis=1)
        hessian = (counts * probability * (1.0 - probability)).sum(axis=1)
        solvable = np.isfinite(hessian) & (hessian > 1e-14) & (total > 0)
        line_search_failures |= (total > 0) & ~solvable
        step = np.divide(
            gradient,
            hessian,
            out=np.zeros_like(gradient),
            where=solvable,
        )

        alpha = np.ones(n_items, dtype=float)
        accepted = ~solvable
        candidate = intercept.copy()
        candidate_objective = negative_log_likelihood.copy()
        for _ in range(30):
            pending = ~accepted
            if not np.any(pending):
                break
            trial = intercept[pending] - alpha[pending] * step[pending]
            trial_eta = offset[None, :] + trial[:, None]
            trial_objective = -(
                (successes[pending] * log_expit(trial_eta)).sum(axis=1)
                + ((counts[pending] - successes[pending]) * log_expit(-trial_eta)).sum(axis=1)
            )
            pending_indices = np.flatnonzero(pending)
            good = np.isfinite(trial_objective) & (
                trial_objective <= negative_log_likelihood[pending] + 1e-12
            )
            if np.any(good):
                selected = pending_indices[good]
                candidate[selected] = trial[good]
                candidate_objective[selected] = trial_objective[good]
                accepted[selected] = True
            alpha[pending_indices[~good]] *= 0.5

        failed = solvable & ~accepted
        line_search_failures |= failed
        candidate[failed] = intercept[failed]
        candidate_objective[failed] = negative_log_likelihood[failed]
        objective_delta = np.abs(negative_log_likelihood - candidate_objective)
        parameter_delta = np.abs(intercept - candidate)
        converged = ((objective_delta < tolerance) & (parameter_delta < tolerance)) | (total <= 0)
        intercept = candidate
        negative_log_likelihood = candidate_objective
        if bool(np.all(converged | line_search_failures)):
            break

    final_probability = expit(offset[None, :] + intercept[:, None])
    final_gradient = (counts * final_probability - successes).sum(axis=1)
    return intercept, {
        "method": "vectorized_1d_1pl_newton",
        "n_items": int(n_items),
        "n_converged": int(converged.sum()),
        "n_failed": int(line_search_failures.sum()),
        "max_inner_iterations": int(n_iterations),
        "max_abs_gradient": float(np.max(np.abs(final_gradient))),
    }


def _fit_log_shrinkage_items_1d(
    offset: np.ndarray,
    successes: np.ndarray,
    counts: np.ndarray,
    strength: float,
    *,
    initial_log_loadings: np.ndarray,
    initial_intercepts: np.ndarray,
    max_newton: int = 50,
    tolerance: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Source-parity vectorized returned-iterate log-shrinkage M-step."""

    offset = np.asarray(offset, dtype=float)
    successes = np.asarray(successes, dtype=float)
    counts = np.asarray(counts, dtype=float)
    if offset.ndim != 1 or successes.ndim != 2 or counts.shape != successes.shape:
        raise CalibrationError("batched log-shrinkage inputs have incompatible shapes")
    if successes.shape[1] != offset.size:
        raise CalibrationError("batched log-shrinkage node axis does not match")
    if not np.isfinite(strength) or strength <= 0:
        raise CalibrationError("log-shrinkage strength must be strictly positive")

    n_items = successes.shape[0]
    total = counts.sum(axis=1)
    log_loading = np.asarray(initial_log_loadings, dtype=float)
    intercept = np.asarray(initial_intercepts, dtype=float)
    if log_loading.shape != (n_items,) or not np.all(np.isfinite(log_loading)):
        raise CalibrationError(f"batched log-a warm start must be finite with shape {(n_items,)}")
    if intercept.shape != (n_items,) or not np.all(np.isfinite(intercept)):
        raise CalibrationError(
            f"batched intercept warm start must be finite with shape {(n_items,)}"
        )
    log_loading = np.clip(log_loading.copy(), -20.0, 20.0)
    intercept = intercept.copy()
    intercept[total <= 0] = 0.0

    def objective(
        log_loading_values: np.ndarray,
        intercept_values: np.ndarray,
        indices: np.ndarray | None = None,
    ) -> np.ndarray:
        selected_successes = successes if indices is None else successes[indices]
        selected_counts = counts if indices is None else counts[indices]
        loading = np.exp(log_loading_values)
        eta = loading[:, None] * offset[None, :] + intercept_values[:, None]
        return (
            -(
                (selected_successes * log_expit(eta)).sum(axis=1)
                + ((selected_counts - selected_successes) * log_expit(-eta)).sum(axis=1)
            )
            + 0.5 * strength * log_loading_values**2
        )

    negative_log_likelihood = objective(log_loading, intercept)
    line_search_failures = np.zeros(n_items, dtype=bool)
    converged = np.zeros(n_items, dtype=bool)
    n_iterations = 0
    for iteration in range(max_newton):
        n_iterations = iteration + 1
        loading = np.exp(log_loading)
        eta = loading[:, None] * offset[None, :] + intercept[:, None]
        probability = expit(eta)
        residual = counts * probability - successes
        weight = counts * probability * (1.0 - probability)
        gradient_loading = loading * (residual @ offset) + strength * log_loading
        gradient_intercept = residual.sum(axis=1)
        hessian_loading = (
            loading * (residual @ offset) + loading**2 * (weight @ (offset**2)) + strength
        )
        hessian_cross = loading * (weight @ offset)
        hessian_intercept = weight.sum(axis=1)
        determinant = hessian_loading * hessian_intercept - hessian_cross**2

        bad_hessian = (
            ~np.isfinite(determinant)
            | (determinant <= 1e-12)
            | (hessian_loading <= 1e-12)
            | (hessian_intercept <= 1e-12)
        )
        if np.any(bad_hessian):
            hessian_loading[bad_hessian] = (
                loading[bad_hessian] ** 2 * (weight[bad_hessian] @ (offset**2)) + strength
            )
            determinant[bad_hessian] = (
                hessian_loading[bad_hessian] * hessian_intercept[bad_hessian]
                - hessian_cross[bad_hessian] ** 2
            )
        solvable = np.isfinite(determinant) & (determinant > 1e-12) & (total > 0)
        line_search_failures |= (total > 0) & ~solvable
        step_loading = np.divide(
            hessian_intercept * gradient_loading - hessian_cross * gradient_intercept,
            determinant,
            out=np.zeros(n_items, dtype=float),
            where=solvable,
        )
        step_intercept = np.divide(
            hessian_loading * gradient_intercept - hessian_cross * gradient_loading,
            determinant,
            out=np.zeros(n_items, dtype=float),
            where=solvable,
        )

        alpha = np.ones(n_items, dtype=float)
        accepted = ~solvable
        candidate_loading = log_loading.copy()
        candidate_intercept = intercept.copy()
        candidate_objective = negative_log_likelihood.copy()
        for _ in range(30):
            pending = ~accepted
            if not np.any(pending):
                break
            trial_loading = np.clip(
                log_loading[pending] - alpha[pending] * step_loading[pending],
                -20.0,
                20.0,
            )
            trial_intercept = intercept[pending] - alpha[pending] * step_intercept[pending]
            pending_indices = np.flatnonzero(pending)
            trial_objective = objective(trial_loading, trial_intercept, indices=pending_indices)
            good = np.isfinite(trial_objective) & (
                trial_objective <= negative_log_likelihood[pending] + 1e-12
            )
            if np.any(good):
                selected = pending_indices[good]
                candidate_loading[selected] = trial_loading[good]
                candidate_intercept[selected] = trial_intercept[good]
                candidate_objective[selected] = trial_objective[good]
                accepted[selected] = True
            alpha[pending_indices[~good]] *= 0.5

        failed = solvable & ~accepted
        line_search_failures |= failed
        objective_delta = np.abs(negative_log_likelihood - candidate_objective)
        parameter_delta = np.maximum(
            np.abs(log_loading - candidate_loading),
            np.abs(intercept - candidate_intercept),
        )
        converged = ((objective_delta < tolerance) & (parameter_delta < tolerance)) | (total <= 0)
        log_loading = candidate_loading
        intercept = candidate_intercept
        negative_log_likelihood = candidate_objective
        if bool(np.all(converged | line_search_failures)):
            break

    loading = np.exp(log_loading)
    probability = expit(loading[:, None] * offset[None, :] + intercept[:, None])
    residual = counts * probability - successes
    final_gradient_loading = loading * (residual @ offset) + strength * log_loading
    final_gradient_intercept = residual.sum(axis=1)
    max_gradient = np.maximum(np.abs(final_gradient_loading), np.abs(final_gradient_intercept))
    return (
        loading,
        intercept,
        {
            "method": "vectorized_1d_log_shrinkage_newton",
            "n_items": int(n_items),
            "n_converged": int(converged.sum()),
            "n_failed": int(line_search_failures.sum()),
            "max_inner_iterations": int(n_iterations),
            "max_abs_gradient": float(np.max(max_gradient)),
        },
    )


def _log_shrinkage_item_objective(
    parameters: np.ndarray,
    loading_design: np.ndarray,
    successes: np.ndarray,
    counts: np.ndarray,
    strength: float,
) -> tuple[float, np.ndarray]:
    log_loadings = np.asarray(parameters[:-1], dtype=float)
    intercept = float(parameters[-1])
    loadings = np.exp(log_loadings)
    eta = loading_design @ loadings + intercept
    negative_log_likelihood = -(
        np.dot(successes, log_expit(eta)) + np.dot(counts - successes, log_expit(-eta))
    ) + 0.5 * strength * float(np.dot(log_loadings, log_loadings))
    residual = counts * expit(eta) - successes
    gradient_loading = loadings * (loading_design.T @ residual) + strength * log_loadings
    gradient = np.concatenate([gradient_loading, np.array([float(residual.sum())])])
    return float(negative_log_likelihood), gradient


def _fit_log_shrinkage_item(
    loading_design: np.ndarray,
    successes: np.ndarray,
    counts: np.ndarray,
    strength: float,
    *,
    initial: np.ndarray | None,
    return_diagnostics: bool,
    max_iterations: int = 200,
    tolerance: float = 1e-8,
) -> tuple[np.ndarray, float] | tuple[np.ndarray, float, dict[str, Any]]:
    n_loadings = loading_design.shape[1]
    if initial is None:
        optimizer_start = np.zeros(n_loadings + 1, dtype=float)
        total = float(counts.sum())
        if total > 0:
            pass_rate = np.clip(float(successes.sum()) / total, 1e-3, 1 - 1e-3)
            unit_offset = loading_design @ np.ones(n_loadings, dtype=float)
            optimizer_start[-1] = np.log(pass_rate / (1.0 - pass_rate)) - np.average(
                unit_offset, weights=counts
            )
    else:
        optimizer_start = np.asarray(initial, dtype=float)
        expected_shape = (n_loadings + 1,)
        if optimizer_start.shape != expected_shape or not np.all(np.isfinite(optimizer_start)):
            raise CalibrationError(
                f"log-shrinkage warm start must be finite with shape {expected_shape}"
            )
        optimizer_start = optimizer_start.copy()
    bounds = [(-20.0, 20.0)] * n_loadings + [(None, None)]
    result = minimize(
        _log_shrinkage_item_objective,
        optimizer_start,
        args=(loading_design, successes, counts, strength),
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={
            "maxiter": int(max_iterations),
            "ftol": float(tolerance),
            "gtol": float(tolerance),
        },
    )
    if not np.all(np.isfinite(result.x)):
        raise CalibrationError("log-shrinkage optimization returned non-finite parameters")
    loadings = np.exp(result.x[:-1])
    intercept = float(result.x[-1])
    if not return_diagnostics:
        return loadings, intercept
    return (
        loadings,
        intercept,
        {
            "converged": bool(result.success),
            "line_search_failed": bool(result.status == 2),
            "n_iter": int(result.nit),
            "objective": float(result.fun),
            "max_abs_gradient": float(np.max(np.abs(result.jac))),
            "status": int(result.status),
            "message": str(result.message),
        },
    )


def _calibration_penalty(
    loadings: np.ndarray,
    q_matrix: np.ndarray,
    calibration_model: CalibrationFamily,
    *,
    ridge: float,
    log_a_shrinkage: float,
) -> float:
    active = loadings[q_matrix.astype(bool)]
    if calibration_model == FREE_2PL:
        return float(0.5 * ridge * np.dot(active, active))
    if calibration_model == LOG_SHRINKAGE_2PL:
        if np.any(active <= 0):
            raise CalibrationError("log-shrinkage loadings must remain strictly positive")
        log_loadings = np.log(active)
        return float(0.5 * log_a_shrinkage * np.dot(log_loadings, log_loadings))
    return 0.0


def _marginal_e_step(
    loadings: np.ndarray,
    difficulties: np.ndarray,
    correlation: np.ndarray,
    grid: np.ndarray,
    base_log_weights: np.ndarray,
    observed_successes: np.ndarray,
    observed_failures: np.ndarray,
) -> tuple[float, np.ndarray]:
    log_prior = _prior_log_weights(grid, base_log_weights, correlation)
    eta = loadings @ grid.T - difficulties[:, None]
    # macOS Accelerate can emit spurious overflow/divide warnings for these finite
    # matrix products. The returned arrays are checked for finiteness at the API
    # boundary, so suppress only the operation-local warnings.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        likelihood = observed_successes @ log_expit(eta) + observed_failures @ log_expit(-eta)
    joint = likelihood + log_prior[None, :]
    person_log_likelihood = logsumexp(joint, axis=1)
    posterior = np.exp(joint - person_log_likelihood[:, None])
    return float(person_log_likelihood.sum()), posterior


def _maximum_parameter_change(
    old_loadings: np.ndarray,
    old_difficulties: np.ndarray,
    old_correlation: np.ndarray,
    loadings: np.ndarray,
    difficulties: np.ndarray,
    correlation: np.ndarray,
) -> dict[str, float]:
    components = {
        "A": float(np.max(np.abs(loadings - old_loadings))),
        "b": float(np.max(np.abs(difficulties - old_difficulties))),
        "R": float(np.max(np.abs(correlation - old_correlation))),
    }
    return {**components, "overall": max(components.values())}


def _aggregate_optimizer_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "method": None,
            "n_items": 0,
            "n_converged": 0,
            "n_failed": 0,
            "max_inner_iterations": 0,
            "max_abs_gradient": 0.0,
        }
    return {
        "method": "per_item_warm_started",
        "n_items": len(rows),
        "n_converged": sum(bool(row["converged"]) for row in rows),
        "n_failed": sum(
            bool(row["line_search_failed"]) or not bool(row["converged"]) for row in rows
        ),
        "max_inner_iterations": max(int(row["n_iter"]) for row in rows),
        "max_abs_gradient": max(float(row["max_abs_gradient"]) for row in rows),
    }


def _fit_m2pl_em(
    responses: np.ndarray,
    observed: np.ndarray,
    q_matrix: np.ndarray,
    config: CalibrationConfig,
    *,
    initial_loadings: np.ndarray | None,
    initial_difficulties: np.ndarray | None,
) -> dict[str, Any]:
    """Run the source returned-iterate Bock--Aitkin EM implementation."""

    responses = np.asarray(responses, dtype=float)
    observed = np.asarray(observed, dtype=bool)
    q_raw = np.asarray(q_matrix)
    if responses.ndim != 2 or observed.shape != responses.shape:
        raise CalibrationError(
            "responses and observed must be same-shape 2-D arrays; "
            f"got {responses.shape}, {observed.shape}"
        )
    if q_raw.ndim != 2 or q_raw.shape[0] != responses.shape[1] or q_raw.shape[1] < 1:
        raise CalibrationError(
            f"Q must have shape (n_items, n_dims>=1); got {q_raw.shape} "
            f"for responses {responses.shape}"
        )
    if not np.isin(q_raw, (0, 1)).all():
        raise CalibrationError("Q must contain only 0/1 values")
    q_matrix = q_raw.astype(int)
    empty_dimensions = np.where(q_matrix.sum(axis=0) == 0)[0].tolist()
    if empty_dimensions:
        raise CalibrationError(
            f"Q has latent dimension(s) with no loading items: {empty_dimensions}"
        )
    if np.any(q_matrix.sum(axis=1) == 0):
        raise CalibrationError("Q has item row(s) with no configured loading")
    observed_values = responses[observed]
    if observed_values.size and not np.isin(observed_values, (0.0, 1.0)).all():
        raise CalibrationError("observed response values must be binary 0/1")

    family = config.model_family
    convergence = config.convergence
    n_persons, n_items = responses.shape
    n_dimensions = q_matrix.shape[1]
    grid, base_log_weights = _build_quadrature(n_dimensions, config.quadrature)
    n_nodes = grid.shape[0]

    observed_successes = np.where(observed, responses, 0.0)
    observed_failures = np.where(observed, 1.0 - responses, 0.0)
    observed_float = observed.astype(float)

    loadings = np.zeros((n_items, n_dimensions))
    loadings[q_matrix == 1] = 1.0
    observations_per_item = observed_float.sum(axis=0)
    pass_rate = np.where(
        observations_per_item > 0,
        observed_successes.sum(axis=0) / np.maximum(observations_per_item, 1),
        0.5,
    )
    pass_rate = np.clip(pass_rate, 1e-3, 1 - 1e-3)
    difficulties = -np.log(pass_rate / (1.0 - pass_rate))

    if initial_loadings is not None:
        candidate = np.asarray(initial_loadings, dtype=float)
        if candidate.shape != loadings.shape or not np.all(np.isfinite(candidate)):
            raise CalibrationError(
                f"initial_loadings must be finite with shape {loadings.shape}; "
                f"got {candidate.shape}"
            )
        if np.any(candidate[q_matrix == 0] != 0.0):
            raise CalibrationError("initial_loadings must be exactly zero outside Q")
        active = candidate[q_matrix == 1]
        if family == ONE_PL and np.any(active != 1.0):
            raise CalibrationError("1PL initial_loadings must equal 1 on active Q entries")
        if family == LOG_SHRINKAGE_2PL and np.any(active <= 0.0):
            raise CalibrationError(
                "log-shrinkage initial_loadings must be positive on active Q entries"
            )
        loadings = candidate.copy()
    if initial_difficulties is not None:
        candidate_difficulties = np.asarray(initial_difficulties, dtype=float)
        if candidate_difficulties.shape != difficulties.shape or not np.all(
            np.isfinite(candidate_difficulties)
        ):
            raise CalibrationError(
                f"initial_difficulties must be finite with shape {difficulties.shape}; "
                f"got {candidate_difficulties.shape}"
            )
        difficulties = candidate_difficulties.copy()
    correlation = np.eye(n_dimensions)

    free_dimensions = [np.where(q_matrix[item_index] == 1)[0] for item_index in range(n_items)]
    constant = np.ones((n_nodes, 1))
    designs = {
        tuple(free.tolist()): np.hstack([grid[:, free], constant]) for free in free_dimensions
    }

    previous_log_likelihood = -np.inf
    previous_returned_objective: float | None = None
    initial_evaluation: dict[str, float] | None = None
    iteration_trace: list[dict[str, Any]] = []
    returned_mode = convergence.mode == RETURNED_ITERATE_CONVERGENCE
    converged = False
    n_iterations = 0
    pending_parameter_change: dict[str, float] | None = None
    pending_optimizer_diagnostics: dict[str, Any] | None = None
    consecutive_passes = 0

    while n_iterations < convergence.max_iterations or (
        returned_mode and n_iterations == convergence.max_iterations
    ):
        marginal_log_likelihood, posterior = _marginal_e_step(
            loadings,
            difficulties,
            correlation,
            grid,
            base_log_weights,
            observed_successes,
            observed_failures,
        )

        if returned_mode:
            penalty = _calibration_penalty(
                loadings,
                q_matrix,
                family,
                ridge=config.ridge,
                log_a_shrinkage=config.log_a_shrinkage,
            )
            returned_objective = marginal_log_likelihood - penalty
            if n_iterations == 0:
                initial_evaluation = {
                    "marginal_loglik": marginal_log_likelihood,
                    "penalty": penalty,
                    "penalized_objective": returned_objective,
                }
            else:
                assert pending_parameter_change is not None
                assert pending_optimizer_diagnostics is not None
                assert previous_returned_objective is not None
                objective_change = returned_objective - previous_returned_objective
                objective_converged = bool(
                    objective_change >= -1e-8
                    and abs(objective_change) < convergence.objective_tolerance
                )
                parameter_converged = bool(
                    convergence.parameter_tolerance is None
                    or pending_parameter_change["overall"] < convergence.parameter_tolerance
                )
                if objective_converged and parameter_converged:
                    consecutive_passes += 1
                else:
                    consecutive_passes = 0
                iteration_trace.append(
                    {
                        "iteration": n_iterations,
                        "marginal_loglik": marginal_log_likelihood,
                        "penalty": penalty,
                        "penalized_objective": returned_objective,
                        "objective_change": objective_change,
                        "objective_decreased_beyond_tolerance": (objective_change < -1e-8),
                        "max_abs_parameter_change": pending_parameter_change,
                        "objective_converged": objective_converged,
                        "parameter_converged": parameter_converged,
                        "consecutive_passes": consecutive_passes,
                        "required_consecutive_passes": convergence.consecutive_passes,
                        "mstep_optimizer": pending_optimizer_diagnostics,
                    }
                )
                if consecutive_passes >= convergence.consecutive_passes:
                    converged = True
                    break
            previous_returned_objective = returned_objective
            if n_iterations >= convergence.max_iterations:
                break
            old_loadings = loadings.copy()
            old_difficulties = difficulties.copy()
            old_correlation = correlation.copy()

        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            expected_successes = observed_successes.T @ posterior
            expected_counts = observed_float.T @ posterior
        optimizer_rows: list[dict[str, Any]] = []
        if returned_mode and family == ONE_PL and n_dimensions == 1:
            intercepts, optimizer_diagnostics = _fit_fixed_loading_items_1d(
                grid[:, 0],
                expected_successes,
                expected_counts,
                initial_intercepts=-difficulties,
            )
            loadings.fill(0.0)
            loadings[q_matrix == 1] = 1.0
            difficulties = -intercepts
        elif returned_mode and family == LOG_SHRINKAGE_2PL and n_dimensions == 1:
            fitted_loadings, intercepts, optimizer_diagnostics = _fit_log_shrinkage_items_1d(
                grid[:, 0],
                expected_successes,
                expected_counts,
                config.log_a_shrinkage,
                initial_log_loadings=np.log(loadings[:, 0]),
                initial_intercepts=-difficulties,
            )
            loadings[:, 0] = fitted_loadings
            difficulties = -intercepts
        else:
            for item_index in range(n_items):
                free = free_dimensions[item_index]
                pattern = tuple(free.tolist())
                if family == FREE_2PL:
                    fitted = _fit_free_item(
                        designs[pattern],
                        expected_successes[item_index],
                        expected_counts[item_index],
                        config.ridge,
                        initial=(
                            np.concatenate(
                                [
                                    loadings[item_index, free],
                                    np.array([-difficulties[item_index]]),
                                ]
                            )
                            if returned_mode
                            else None
                        ),
                        return_diagnostics=returned_mode,
                    )
                    if returned_mode:
                        beta, optimizer = cast(tuple[np.ndarray, dict[str, Any]], fitted)
                        optimizer_rows.append(optimizer)
                    else:
                        beta = cast(np.ndarray, fitted)
                    loadings[item_index] = 0.0
                    loadings[item_index, free] = beta[:-1]
                    difficulties[item_index] = -beta[-1]
                elif family == ONE_PL:
                    loadings[item_index] = 0.0
                    loadings[item_index, free] = 1.0
                    intercept = _fit_fixed_loading_item(
                        grid[:, free].sum(axis=1),
                        expected_successes[item_index],
                        expected_counts[item_index],
                    )
                    difficulties[item_index] = -intercept
                else:
                    fitted = _fit_log_shrinkage_item(
                        grid[:, free],
                        expected_successes[item_index],
                        expected_counts[item_index],
                        config.log_a_shrinkage,
                        initial=(
                            np.concatenate(
                                [
                                    np.log(loadings[item_index, free]),
                                    np.array([-difficulties[item_index]]),
                                ]
                            )
                            if returned_mode
                            else None
                        ),
                        return_diagnostics=returned_mode,
                    )
                    if returned_mode:
                        positive, intercept, optimizer = cast(
                            tuple[np.ndarray, float, dict[str, Any]], fitted
                        )
                        optimizer_rows.append(optimizer)
                    else:
                        positive, intercept = cast(tuple[np.ndarray, float], fitted)
                    loadings[item_index] = 0.0
                    loadings[item_index, free] = positive
                    difficulties[item_index] = -intercept
            optimizer_diagnostics = _aggregate_optimizer_diagnostics(optimizer_rows)

        if config.estimate_latent_correlation and n_dimensions > 1:
            node_weights = posterior.sum(axis=0)
            covariance = (grid.T * node_weights) @ grid / n_persons
            standard_deviation = np.sqrt(np.clip(np.diag(covariance), 1e-8, None))
            correlation = _project_correlation(
                covariance / np.outer(standard_deviation, standard_deviation)
            )

        n_iterations += 1
        if returned_mode:
            pending_parameter_change = _maximum_parameter_change(
                old_loadings,
                old_difficulties,
                old_correlation,
                loadings,
                difficulties,
                correlation,
            )
            pending_optimizer_diagnostics = optimizer_diagnostics
        else:
            if (
                abs(marginal_log_likelihood - previous_log_likelihood)
                < convergence.objective_tolerance
            ):
                converged = True
                previous_log_likelihood = marginal_log_likelihood
                break
            previous_log_likelihood = marginal_log_likelihood

    loadings[q_matrix == 0] = 0.0
    final_log_likelihood, _ = _marginal_e_step(
        loadings,
        difficulties,
        correlation,
        grid,
        base_log_weights,
        observed_successes,
        observed_failures,
    )
    n_free_loadings = 0 if family == ONE_PL else int(q_matrix.sum())
    n_parameters = n_free_loadings + n_items
    if config.estimate_latent_correlation and n_dimensions > 1:
        n_parameters += n_dimensions * (n_dimensions - 1) // 2
    final_penalty = _calibration_penalty(
        loadings,
        q_matrix,
        family,
        ridge=config.ridge,
        log_a_shrinkage=config.log_a_shrinkage,
    )

    objective_changes = [float(row["objective_change"]) for row in iteration_trace]
    diagnostics: dict[str, Any] = {
        "objective": ("penalized_marginal_loglik" if returned_mode else "marginal_loglik"),
        "objective_tolerance": convergence.objective_tolerance,
        "parameter_tolerance": convergence.parameter_tolerance,
        "required_consecutive_passes": convergence.consecutive_passes,
        "final_consecutive_passes": consecutive_passes,
        "initial_evaluation": initial_evaluation,
        "warm_start": {
            "initial_loadings_supplied": initial_loadings is not None,
            "initial_difficulties_supplied": initial_difficulties is not None,
            "per_item_mstep_from_current_iterate": returned_mode,
        },
        "returned_iterate_matches_last_trace": bool(
            returned_mode
            and iteration_trace
            and final_log_likelihood == iteration_trace[-1]["marginal_loglik"]
            and final_log_likelihood - final_penalty == iteration_trace[-1]["penalized_objective"]
        ),
        "stopped_before_extra_mstep": bool(returned_mode and converged),
        "final_exact_recomputation": True,
        "all_objective_changes_monotone_within_tolerance": bool(
            all(change >= -1e-8 for change in objective_changes)
        ),
        "trace": iteration_trace,
    }
    return {
        "A": loadings,
        "b": difficulties,
        "R": correlation,
        "loglik": final_log_likelihood,
        "penalized_objective": final_log_likelihood - final_penalty,
        "n_params": n_parameters,
        "n_free_loadings": n_free_loadings,
        "n_fixed_loadings": int(q_matrix.sum()) if family == ONE_PL else 0,
        "n_iter": n_iterations,
        "converged": converged,
        "convergence_diagnostics": diagnostics,
        "calibration_specification": _calibration_specification(
            family,
            ridge=config.ridge,
            log_a_shrinkage=config.log_a_shrinkage,
        ),
    }


def fit_calibration(
    responses: np.ndarray | Sequence[Sequence[float | None]],
    source_q: np.ndarray | Sequence[Sequence[int]],
    *,
    skill_structure: SkillStructure,
    config: CalibrationConfig,
    missing_policy: MissingCellPolicy,
    observed_mask: np.ndarray | Sequence[Sequence[bool]] | None = None,
    item_ids: Sequence[str] | None = None,
    initial_loadings: np.ndarray | None = None,
    initial_difficulties: np.ndarray | None = None,
) -> CalibrationFitResult:
    """Fit a confirmatory MIRT bank from an in-memory binary response matrix.

    Args:
        responses: Person-by-item Pass/Fail matrix (1/0). ``NaN`` or ``None`` is a
            missing/no-decision cell and is excluded. Values hidden by an explicit
            ``observed_mask`` are excluded as well.
        source_q: Item-by-source-skill binary Q matrix.
        skill_structure: Explicit source axis and modeled dimension partition.
        config: Explicit model family and numerical policy.
        missing_policy: Must be ``"exclude"``. It is required so callers cannot
            silently reinterpret no-decisions as failures.
        observed_mask: Optional explicit person-by-item mask. If omitted, finite
            cells are observed and ``NaN`` cells are missing.
        item_ids: Optional aligned stable IDs, included in the specification hash.
        initial_loadings: Optional modeled-Q-aligned deterministic warm start.
        initial_difficulties: Optional item-aligned deterministic warm start.

    Returns:
        A structured fit whose ``converged`` and ``termination_reason`` distinguish
        a tolerance-qualified result from the last valid iterate at the iteration
        limit. Parameters are always recomputed and validated at the returned
        iterate, including for nonconverged fits.
    """

    if missing_policy != EXCLUDE_MISSING:
        raise CalibrationError("missing_policy must be the explicit value 'exclude'")
    try:
        values = np.asarray(responses, dtype=float)
    except (TypeError, ValueError) as error:
        raise CalibrationError(
            "responses must contain numeric 0/1 values or missing cells"
        ) from error
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise CalibrationError("responses must have shape (n_persons>=1, n_items>=1)")
    if np.any(np.isinf(values)):
        raise CalibrationError("responses cannot contain infinity")

    if observed_mask is None:
        observed = np.isfinite(values)
    else:
        observed_raw = np.asarray(observed_mask)
        if observed_raw.shape != values.shape or not all(
            value in (0, 1, False, True) for value in observed_raw.reshape(-1).tolist()
        ):
            raise CalibrationError("observed_mask must be a binary array aligned to responses")
        observed = observed_raw.astype(bool)
        if np.any(observed & ~np.isfinite(values)):
            raise CalibrationError("observed_mask marks a missing response as observed")
    if not np.isin(values[observed], (0.0, 1.0)).all():
        raise CalibrationError("observed responses must be binary 0/1")

    n_persons_supplied, n_items = values.shape
    if item_ids is None:
        normalized_item_ids = tuple(str(index) for index in range(n_items))
    else:
        normalized_item_ids = tuple(str(item_id).strip() for item_id in item_ids)
        if len(normalized_item_ids) != n_items:
            raise CalibrationError("item_ids must align with response columns")
        if any(not item_id for item_id in normalized_item_ids):
            raise CalibrationError("item_ids must be non-empty")
        if len(set(normalized_item_ids)) != len(normalized_item_ids):
            raise CalibrationError("item_ids must be unique")

    source_q_array = np.asarray(source_q)
    expected_q_shape = (n_items, len(skill_structure.source_skills))
    if source_q_array.shape != expected_q_shape:
        raise CalibrationError(
            f"source_q must have shape {expected_q_shape} for this skill structure; "
            f"got {source_q_array.shape}"
        )
    try:
        modeled_q = skill_structure.transform_q(source_q_array)
    except ValueError as error:
        raise CalibrationError(str(error)) from error
    if np.any(modeled_q.sum(axis=1) == 0):
        ids = [normalized_item_ids[index] for index in np.where(modeled_q.sum(axis=1) == 0)[0][:10]]
        raise CalibrationError(f"Q has item rows with no modeled skill; first item IDs: {ids}")
    empty_dimensions = np.where(modeled_q.sum(axis=0) == 0)[0]
    if empty_dimensions.size:
        labels = [skill_structure.labels[index] for index in empty_dimensions]
        raise CalibrationError(f"Q has modeled dimensions with no loading items: {labels}")
    config.quadrature.total_nodes(skill_structure.n_dims)

    n_observed_supplied = int(observed.sum())
    all_missing_person = ~observed.any(axis=1)
    fitted_values = np.where(observed, values, 0.0)[~all_missing_person]
    fitted_observed = observed[~all_missing_person]
    if fitted_values.shape[0] < 1:
        raise CalibrationError("no respondent has an observed Pass/Fail cell")
    observations_per_item = fitted_observed.sum(axis=0)
    if np.any(observations_per_item == 0):
        ids = [normalized_item_ids[index] for index in np.where(observations_per_item == 0)[0][:10]]
        raise CalibrationError(f"items with no observed responses cannot be fit: {ids}")

    core = _fit_m2pl_em(
        fitted_values,
        fitted_observed,
        modeled_q,
        config,
        initial_loadings=initial_loadings,
        initial_difficulties=initial_difficulties,
    )
    loadings = np.asarray(core["A"], dtype=float)
    difficulties = np.asarray(core["b"], dtype=float)
    latent_correlation = np.asarray(core["R"], dtype=float)
    if not (
        np.all(np.isfinite(loadings))
        and np.all(np.isfinite(difficulties))
        and np.all(np.isfinite(latent_correlation))
        and np.isfinite(core["loglik"])
        and np.isfinite(core["penalized_objective"])
    ):
        raise CalibrationError("calibration returned non-finite parameters or objective")
    if np.any(loadings[modeled_q == 0] != 0.0):
        raise CalibrationError("calibration returned nonzero loadings outside Q")
    if config.model_family == ONE_PL and np.any(loadings[modeled_q == 1] != 1.0):
        raise CalibrationError("1PL calibration did not preserve fixed active loadings")
    if config.model_family == LOG_SHRINKAGE_2PL and np.any(loadings[modeled_q == 1] <= 0.0):
        raise CalibrationError("log-shrinkage calibration returned a nonpositive loading")
    if float(np.min(np.linalg.eigvalsh(latent_correlation))) <= 0:
        raise CalibrationError("calibration returned a non-positive-definite correlation")

    model_specification = cast(Mapping[str, Any], core["calibration_specification"])
    structure_dict = skill_structure.as_dict()
    quadrature_dict = config.quadrature.as_dict(skill_structure.n_dims)
    convergence_dict = config.convergence.as_dict()
    hash_payload = {
        "schema_version": 1,
        "source_blob": SOURCE_BLOB,
        "parameterization": "sigmoid(A @ theta - b)",
        "model_specification": model_specification,
        "skill_structure": structure_dict,
        "quadrature": quadrature_dict,
        "convergence_policy": convergence_dict,
        "estimate_latent_correlation": config.estimate_latent_correlation,
        "missing_cell_policy": missing_policy,
        "item_ids": list(normalized_item_ids),
        "modeled_q_sha256": hashlib.sha256(
            np.ascontiguousarray(modeled_q, dtype=np.int8).tobytes()
        ).hexdigest(),
    }
    specification_hash = hashlib.sha256(
        json.dumps(hash_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    provenance = CalibrationProvenance(
        source_branch=SOURCE_BRANCH,
        source_commit=SOURCE_COMMIT,
        source_blob=SOURCE_BLOB,
        parameterization="sigmoid(A @ theta - b)",
        model_specification=MappingProxyType(dict(model_specification)),
        skill_structure=MappingProxyType(structure_dict),
        quadrature=MappingProxyType(quadrature_dict),
        convergence_policy=MappingProxyType(convergence_dict),
        missing_cell_policy=missing_policy,
        n_persons_supplied=n_persons_supplied,
        n_persons_fitted=int(fitted_values.shape[0]),
        n_all_missing_persons_excluded=int(all_missing_person.sum()),
        n_items=n_items,
        n_observed_cells=n_observed_supplied,
        n_missing_cells=int(values.size - n_observed_supplied),
        specification_sha256=specification_hash,
    )
    converged = bool(core["converged"])
    if converged and config.convergence.mode == RETURNED_ITERATE_CONVERGENCE:
        iterate_status = "converged_returned_iterate"
    elif converged:
        iterate_status = "converged_legacy_iterate"
    else:
        iterate_status = "last_valid_returned_iterate"
    return CalibrationFitResult(
        loadings=loadings,
        difficulties=difficulties,
        latent_correlation=latent_correlation,
        marginal_log_likelihood=float(core["loglik"]),
        penalized_objective=float(core["penalized_objective"]),
        n_parameters=int(core["n_params"]),
        n_free_loadings=int(core["n_free_loadings"]),
        n_fixed_loadings=int(core["n_fixed_loadings"]),
        iterations=int(core["n_iter"]),
        converged=converged,
        termination_reason="converged" if converged else "max_iterations",
        returned_iterate_status=iterate_status,
        convergence_diagnostics=cast(Mapping[str, Any], core["convergence_diagnostics"]),
        provenance=provenance,
    )
