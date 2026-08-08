"""Focused tests for calibration-time quadrature choices."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
MODULE_PATH = ROOT / "scripts" / "calibrate_mirt.py"
SPEC = importlib.util.spec_from_file_location("calibrate_mirt_quadrature_tests", MODULE_PATH)
assert SPEC and SPEC.loader
cm = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cm
SPEC.loader.exec_module(cm)


def test_normal_trapezoid_is_symmetric_and_normalized() -> None:
    grid, log_weights = cm.build_calibration_quadrature(
        1,
        801,
        method=cm.NORMAL_TRAPEZOID_QUADRATURE,
        linear_bound=8.0,
    )

    assert grid.shape == (801, 1)
    assert np.allclose(grid[:, 0], -grid[::-1, 0])
    assert np.allclose(log_weights, log_weights[::-1])
    assert np.isclose(np.exp(log_weights).sum(), 1.0)
    assert abs(float(np.exp(log_weights) @ grid[:, 0])) < 1e-12


def test_default_quadrature_is_exact_historical_gauss_hermite_path() -> None:
    default_grid, default_log_weights = cm.build_calibration_quadrature(2, 5)
    explicit_grid, explicit_log_weights = cm.build_calibration_quadrature(
        2, 5, method=cm.GAUSS_HERMITE_QUADRATURE
    )

    assert np.array_equal(default_grid, cm.build_grid(2, 5))
    assert np.array_equal(default_log_weights, cm.base_log_weights(2, 5))
    assert np.array_equal(default_grid, explicit_grid)
    assert np.array_equal(default_log_weights, explicit_log_weights)

    rng = np.random.default_rng(14)
    responses = rng.integers(0, 2, size=(50, 4)).astype(float)
    observed = np.ones_like(responses, dtype=bool)
    q_matrix = np.ones((4, 1), dtype=int)
    kwargs = {"nodes_per_dim": 5, "ridge": 0.01, "max_iter": 8, "tol": 1e-5}
    historical = cm.fit_m2pl_em(responses, observed, q_matrix, **kwargs)
    explicit = cm.fit_m2pl_em(
        responses,
        observed,
        q_matrix,
        quadrature_method=cm.GAUSS_HERMITE_QUADRATURE,
        **kwargs,
    )
    assert np.array_equal(historical["A"], explicit["A"])
    assert np.array_equal(historical["b"], explicit["b"])
    assert historical["loglik"] == explicit["loglik"]
    assert historical["n_iter"] == explicit["n_iter"]


def test_normal_trapezoid_rejects_multidimensional_calibration() -> None:
    with pytest.raises(cm.CalibrationError, match="supported only for 1D"):
        cm.build_calibration_quadrature(
            2,
            41,
            method=cm.NORMAL_TRAPEZOID_QUADRATURE,
            linear_bound=8.0,
        )


def test_basic_synthetic_1pl_fit_with_normal_trapezoid() -> None:
    rng = np.random.default_rng(20260805)
    n_people = 160
    difficulty = np.linspace(-1.25, 1.25, 7)
    theta = rng.standard_normal(n_people)
    probability = 1.0 / (1.0 + np.exp(-(theta[:, None] - difficulty[None, :])))
    responses = (rng.random(probability.shape) < probability).astype(float)
    observed = np.ones_like(responses, dtype=bool)
    q_matrix = np.ones((len(difficulty), 1), dtype=int)

    fit = cm.fit_m2pl_em(
        responses,
        observed,
        q_matrix,
        nodes_per_dim=161,
        max_iter=100,
        tol=1e-5,
        calibration_model=cm.ONE_PL,
        quadrature_method=cm.NORMAL_TRAPEZOID_QUADRATURE,
        linear_bound=8.0,
    )

    assert fit["converged"] is True
    assert fit["quadrature_method"] == cm.NORMAL_TRAPEZOID_QUADRATURE
    assert fit["quadrature_linear_bound"] == 8.0
    assert np.array_equal(fit["A"], np.ones_like(fit["A"]))
    assert np.isfinite(fit["loglik"])
    assert np.all(np.isfinite(fit["b"]))
    assert np.corrcoef(fit["b"], difficulty)[0, 1] > 0.85
