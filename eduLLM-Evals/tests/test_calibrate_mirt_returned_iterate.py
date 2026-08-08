"""Focused tests for opt-in returned-iterate EM diagnostics and warm starts."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
MODULE_PATH = ROOT / "scripts" / "calibrate_mirt.py"
SPEC = importlib.util.spec_from_file_location(
    "calibrate_mirt_returned_iterate_tests", MODULE_PATH
)
assert SPEC and SPEC.loader
cm = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cm
SPEC.loader.exec_module(cm)


def _synthetic_1d(seed: int = 20260805, n_people: int = 100, n_items: int = 8):
    rng = np.random.default_rng(seed)
    theta = rng.standard_normal(n_people)
    difficulty = np.linspace(-1.25, 1.25, n_items)
    probability = 1.0 / (1.0 + np.exp(-(theta[:, None] - difficulty[None, :])))
    responses = (rng.random(probability.shape) < probability).astype(float)
    observed = np.ones_like(responses, dtype=bool)
    q_matrix = np.ones((n_items, 1), dtype=int)
    return responses, observed, q_matrix


def test_vectorized_1pl_mstep_matches_scalar_item_solves() -> None:
    rng = np.random.default_rng(11)
    offset = np.linspace(-4.0, 4.0, 81)
    expected_counts = rng.uniform(0.01, 2.0, size=(7, offset.size))
    success_probability = 1.0 / (
        1.0 + np.exp(-(offset[None, :] + np.linspace(-1.0, 1.0, 7)[:, None]))
    )
    expected_successes = expected_counts * success_probability

    batched, diagnostics = cm._fit_fixed_loading_items_1d(
        offset,
        expected_successes,
        expected_counts,
        initial_intercepts=np.zeros(7),
    )
    scalar = np.array(
        [
            cm._fit_fixed_loading_item(offset, expected_successes[j], expected_counts[j])
            for j in range(7)
        ]
    )

    assert np.allclose(batched, scalar, atol=1e-8, rtol=0.0)
    assert diagnostics["method"] == "vectorized_1d_1pl_newton"
    assert diagnostics["n_items"] == 7
    assert diagnostics["n_failed"] == 0


def test_vectorized_log_shrinkage_mstep_matches_scalar_item_solves() -> None:
    rng = np.random.default_rng(19)
    offset = np.linspace(-4.0, 4.0, 81)
    expected_counts = rng.uniform(0.01, 2.0, size=(6, offset.size))
    true_loading = np.linspace(0.65, 1.45, 6)
    true_intercept = np.linspace(-0.8, 0.8, 6)
    probability = 1.0 / (
        1.0
        + np.exp(
            -(
                true_loading[:, None] * offset[None, :]
                + true_intercept[:, None]
            )
        )
    )
    expected_successes = expected_counts * probability
    strength = 4.0

    loading, intercept, diagnostics = cm._fit_log_shrinkage_items_1d(
        offset,
        expected_successes,
        expected_counts,
        strength,
        initial_log_a=np.zeros(6),
        initial_intercepts=np.zeros(6),
    )
    scalar = [
        cm._fit_item_log_shrinkage(
            offset[:, None], expected_successes[j], expected_counts[j], strength
        )
        for j in range(6)
    ]

    assert np.allclose(loading, [row[0][0] for row in scalar], atol=2e-6, rtol=0.0)
    assert np.allclose(intercept, [row[1] for row in scalar], atol=5e-6, rtol=0.0)
    assert diagnostics["method"] == "vectorized_1d_log_shrinkage_newton"
    assert diagnostics["n_failed"] == 0


def test_returned_iterate_mode_traces_exact_returned_1pl_objective() -> None:
    responses, observed, q_matrix = _synthetic_1d()
    initial_A = np.ones_like(q_matrix, dtype=float)
    initial_b = np.zeros(q_matrix.shape[0], dtype=float)
    original_A = initial_A.copy()
    original_b = initial_b.copy()

    fit = cm.fit_m2pl_em(
        responses,
        observed,
        q_matrix,
        nodes_per_dim=101,
        max_iter=300,
        tol=1e-5,
        calibration_model=cm.ONE_PL,
        quadrature_method=cm.NORMAL_TRAPEZOID_QUADRATURE,
        linear_bound=8.0,
        convergence_mode=cm.RETURNED_ITERATE_CONVERGENCE,
        consecutive_convergence_passes=2,
        initial_A=initial_A,
        initial_b=initial_b,
    )

    diagnostics = fit["convergence_diagnostics"]
    trace = diagnostics["trace"]
    assert fit["converged"] is True
    assert fit["convergence_mode"] == cm.RETURNED_ITERATE_CONVERGENCE
    assert len(trace) == fit["n_iter"]
    assert trace[-1]["marginal_loglik"] == fit["loglik"]
    assert trace[-1]["penalized_objective"] == fit["penalized_objective"]
    assert trace[-1]["penalty"] == 0.0
    assert diagnostics["returned_iterate_matches_last_trace"] is True
    assert diagnostics["all_objective_changes_monotone_within_tolerance"] is True
    assert diagnostics["required_consecutive_passes"] == 2
    assert diagnostics["final_consecutive_passes"] >= 2
    assert diagnostics["stopped_before_extra_mstep"] is True
    assert all(
        row["mstep_optimizer"]["method"] == "vectorized_1d_1pl_newton"
        for row in trace
    )
    assert np.array_equal(initial_A, original_A)
    assert np.array_equal(initial_b, original_b)


def test_returned_iterate_free_2pl_uses_penalized_trace_and_warm_starts() -> None:
    responses, observed, q_matrix = _synthetic_1d(seed=31, n_items=5)
    initial_A = np.full(q_matrix.shape, 0.9, dtype=float)
    initial_b = np.linspace(-0.2, 0.2, q_matrix.shape[0])
    ridge = 0.2

    fit = cm.fit_m2pl_em(
        responses,
        observed,
        q_matrix,
        nodes_per_dim=81,
        max_iter=4,
        tol=1e-8,
        ridge=ridge,
        calibration_model=cm.FREE_2PL,
        quadrature_method=cm.NORMAL_TRAPEZOID_QUADRATURE,
        convergence_mode=cm.RETURNED_ITERATE_CONVERGENCE,
        initial_A=initial_A,
        initial_b=initial_b,
    )

    expected_penalty = 0.5 * ridge * float(np.sum(fit["A"] ** 2))
    trace = fit["convergence_diagnostics"]["trace"]
    assert trace[-1]["penalty"] == pytest.approx(expected_penalty)
    assert fit["penalized_objective"] == pytest.approx(
        fit["loglik"] - expected_penalty
    )
    assert all(
        row["mstep_optimizer"]["method"] == "per_item_warm_started"
        and row["mstep_optimizer"]["n_items"] == q_matrix.shape[0]
        for row in trace
    )
    assert fit["convergence_diagnostics"]["warm_start"] == {
        "initial_A_supplied": True,
        "initial_b_supplied": True,
        "per_item_mstep_from_current_iterate": True,
    }


def test_returned_iterate_1d_shrinkage_uses_vectorized_mstep() -> None:
    responses, observed, q_matrix = _synthetic_1d(seed=47, n_items=6)
    fit = cm.fit_m2pl_em(
        responses,
        observed,
        q_matrix,
        nodes_per_dim=81,
        max_iter=3,
        tol=1e-8,
        calibration_model=cm.LOG_SHRINKAGE_2PL,
        log_a_shrinkage=4.0,
        quadrature_method=cm.NORMAL_TRAPEZOID_QUADRATURE,
        convergence_mode=cm.RETURNED_ITERATE_CONVERGENCE,
    )

    trace = fit["convergence_diagnostics"]["trace"]
    assert all(
        row["mstep_optimizer"]["method"]
        == "vectorized_1d_log_shrinkage_newton"
        for row in trace
    )
    assert np.all(fit["A"] > 0.0)
    assert trace[-1]["penalty"] > 0.0


def test_new_convergence_and_warm_start_inputs_are_opt_in_and_validated() -> None:
    responses, observed, q_matrix = _synthetic_1d(n_items=4)
    historical = cm.fit_m2pl_em(
        responses,
        observed,
        q_matrix,
        nodes_per_dim=7,
        max_iter=2,
    )
    assert "convergence_mode" not in historical
    assert "convergence_diagnostics" not in historical

    with pytest.raises(cm.CalibrationError, match="unknown convergence mode"):
        cm.fit_m2pl_em(
            responses,
            observed,
            q_matrix,
            nodes_per_dim=7,
            convergence_mode="not-a-mode",
        )
    with pytest.raises(cm.CalibrationError, match="initial_A must be finite"):
        cm.fit_m2pl_em(
            responses,
            observed,
            q_matrix,
            nodes_per_dim=7,
            initial_A=np.full((4, 1), np.nan),
        )
    with pytest.raises(cm.CalibrationError, match="parameter_tol"):
        cm.fit_m2pl_em(
            responses,
            observed,
            q_matrix,
            nodes_per_dim=7,
            parameter_tol=0.0,
        )
    with pytest.raises(cm.CalibrationError, match="consecutive_convergence_passes"):
        cm.fit_m2pl_em(
            responses,
            observed,
            q_matrix,
            nodes_per_dim=7,
            consecutive_convergence_passes=0,
        )
