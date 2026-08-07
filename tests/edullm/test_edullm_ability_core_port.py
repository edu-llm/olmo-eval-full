"""Parity tests adapted from the authoritative InFoBench CAT estimators."""

import numpy as np
import pytest
from scipy.optimize import approx_fprime

from olmo_eval.edullm.ability import (
    NumericalCoreError,
    _mwle_neg_obj_grad,
    batch_eap,
    build_quadrature,
    mwle,
    pass_rate_check,
)


def test_edullm_eap_uses_correlated_joint_prior_without_observations() -> None:
    correlation = np.array([[1.0, 0.65], [0.65, 1.0]])
    quadrature = build_quadrature(2, 11, correlation)

    estimate = batch_eap(
        np.array([np.nan]),
        np.array([[1.0, 0.0]]),
        np.array([0.0]),
        quadrature,
    )

    assert estimate.n_items == 0
    assert np.allclose(estimate.theta, 0.0, atol=1e-8)
    assert np.allclose(np.diag(estimate.covariance), 1.0, atol=0.03)
    assert estimate.covariance[0, 1] == pytest.approx(0.65, abs=0.04)


def test_edullm_eap_excludes_missing_response_instead_of_failing_it() -> None:
    quadrature = build_quadrature(1, 11)
    loadings = np.array([[1.2], [1.2]])
    difficulties = np.array([0.0, 0.0])

    with_missing = batch_eap(np.array([1.0, np.nan]), loadings, difficulties, quadrature)
    first_only = batch_eap(np.array([1.0, 0.0]), loadings, difficulties, quadrature, [0])

    assert with_missing.n_items == 1
    assert np.allclose(with_missing.theta, first_only.theta)
    assert np.allclose(with_missing.covariance, first_only.covariance)


def test_edullm_pass_rate_check_uses_same_observed_item_denominator() -> None:
    responses = np.array([1.0, np.nan, 0.0])
    loadings = np.array([[1.0], [2.0], [1.0]])
    difficulties = np.array([-0.4, 9.0, 0.4])

    check = pass_rate_check(responses, loadings, difficulties, np.array([0.0]))

    expected = float(
        np.mean(
            [
                1.0 / (1.0 + np.exp(-0.4)),
                1.0 / (1.0 + np.exp(0.4)),
            ]
        )
    )
    assert check["n_items"] == 2
    assert check["observed_pass_rate"] == pytest.approx(0.5)
    assert check["predicted_pass_rate"] == pytest.approx(expected)
    assert check["error"] == pytest.approx(expected - 0.5)


def test_edullm_mwle_analytic_gradient_matches_finite_difference() -> None:
    loadings = np.array([[1.0, 0.0], [0.0, 1.2], [0.8, 0.6], [1.1, -0.3]])
    difficulties = np.array([-0.2, 0.1, 0.3, -0.1])
    responses = np.array([1.0, 0.0, 1.0, 0.0])
    theta = np.array([0.25, -0.35])
    ridge = 1e-5

    def objective(value: np.ndarray) -> float:
        return _mwle_neg_obj_grad(value, loadings, difficulties, responses, ridge)[0]

    _, analytic = _mwle_neg_obj_grad(theta, loadings, difficulties, responses, ridge)
    numeric = approx_fprime(theta, objective, 1e-7)
    assert np.allclose(analytic, numeric, atol=2e-5, rtol=2e-5)


def test_edullm_mwle_reports_rank_deficiency_without_false_convergence() -> None:
    start = np.array([0.2, -0.1])

    estimate = mwle(
        np.array([1.0, np.nan]),
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.array([0.0, 0.0]),
        theta0=start,
    )

    assert estimate.converged is False
    assert estimate.n_items == 1
    assert estimate.message == "administered loading matrix is rank 1/2"
    assert np.array_equal(estimate.theta, start)
    assert np.isnan(estimate.se).all()


def test_edullm_mwle_optimizes_full_rank_observations() -> None:
    estimate = mwle(
        np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
        np.ones((6, 1)),
        np.array([-1.5, -1.0, -0.5, 0.5, 1.0, 1.5]),
    )

    assert estimate.converged is True
    assert estimate.n_items == 6
    assert estimate.theta == pytest.approx([0.0], abs=1e-10)
    assert np.isfinite(estimate.se).all()


def test_edullm_normal_trapezoid_rule_is_normalized_and_symmetric() -> None:
    quadrature = build_quadrature(
        1,
        801,
        np.eye(1),
        max_nodes=1000,
        method="normal_trapezoid",
        linear_bound=8.0,
    )

    assert quadrature.method == "normal_trapezoid"
    assert quadrature.lower_bound == -8.0
    assert quadrature.upper_bound == 8.0
    assert quadrature.grid.shape == (801, 1)
    assert np.isclose(np.exp(quadrature.log_prior).sum(), 1.0)
    assert np.allclose(quadrature.grid[:, 0], -quadrature.grid[::-1, 0])
    assert np.allclose(quadrature.log_prior, quadrature.log_prior[::-1])


def test_edullm_normal_trapezoid_rejects_multidimensional_use() -> None:
    with pytest.raises(NumericalCoreError, match="only for 1D"):
        build_quadrature(2, 21, np.eye(2), method="normal_trapezoid")
