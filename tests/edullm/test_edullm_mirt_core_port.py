"""Parity tests adapted from the authoritative InFoBench MIRT tests."""

import numpy as np
import pytest

from olmo_eval.edullm.mirt import (
    initial_state,
    pass_probability,
    posterior_moments,
    standard_errors,
    update,
)

# Worked example from the EduLLM adaptive-evaluation specification.
THETA = np.array([0.50, 0.00, -0.20])
COVARIANCE = np.diag([0.25, 0.25, 0.25])
DISCRIMINATION = np.array([1.2, 0.8, 0.5])
Q_ROW = np.array([1, 1, 0])
DIFFICULTY = 0.3


def test_edullm_mirt_preserves_probability_parameterization() -> None:
    probability = pass_probability(THETA, DISCRIMINATION, Q_ROW, DIFFICULTY)

    # sigmoid((q * a) @ theta - b) = sigmoid(0.30).
    assert probability == pytest.approx(0.574, abs=1e-3)


def test_edullm_mirt_online_update_matches_worked_example() -> None:
    theta_new, covariance_new, _ = update(
        THETA,
        COVARIANCE,
        DISCRIMINATION,
        Q_ROW,
        DIFFICULTY,
        1,
    )

    assert covariance_new[0, 0] == pytest.approx(0.2305, abs=2e-4)
    assert covariance_new[1, 1] == pytest.approx(0.2413, abs=2e-4)
    assert covariance_new[2, 2] == pytest.approx(0.25, abs=1e-12)
    assert covariance_new[0, 1] == pytest.approx(-0.0130, abs=2e-4)
    assert theta_new == pytest.approx([0.613, 0.076, -0.200], abs=2e-3)
    assert standard_errors(covariance_new) == pytest.approx([0.480, 0.491, 0.500], abs=2e-3)


def test_edullm_mirt_initial_state_supports_dynamic_dimensions() -> None:
    theta, covariance = initial_state(n_skills=5)

    assert np.array_equal(theta, np.zeros(5))
    assert np.array_equal(covariance, np.eye(5))


def test_edullm_mirt_joint_posterior_matches_direct_calculation() -> None:
    nodes = np.linspace(-8.0, 8.0, 801)
    step = float(nodes[1] - nodes[0])
    trapezoid = np.full(len(nodes), step)
    trapezoid[[0, -1]] *= 0.5
    log_prior = -0.5 * nodes**2 - 0.5 * np.log(2.0 * np.pi) + np.log(trapezoid)
    log_prior -= np.logaddexp.reduce(log_prior)
    responses = np.array([1.0, 0.0, 1.0])
    loadings = np.array([[1.4], [0.8], [1.1]])
    difficulties = np.array([-0.3, 0.5, 0.1])

    result = posterior_moments(
        responses,
        loadings,
        difficulties,
        nodes[:, None],
        log_prior,
    )

    eta = nodes[:, None] @ loadings.T - difficulties[None, :]
    direct = log_prior + (
        responses[None, :] * -np.logaddexp(0.0, -eta)
        + (1.0 - responses[None, :]) * -np.logaddexp(0.0, eta)
    ).sum(axis=1)
    weights = np.exp(direct - np.logaddexp.reduce(direct))
    expected_theta = float(weights @ nodes)
    expected_se = float(np.sqrt(weights @ ((nodes - expected_theta) ** 2)))
    assert result.theta[0] == pytest.approx(expected_theta, abs=1e-12)
    assert result.se[0] == pytest.approx(expected_se, abs=1e-12)


def test_edullm_mirt_empty_response_set_returns_prior_moments() -> None:
    nodes = np.linspace(-6.0, 6.0, 1201)
    log_prior = -0.5 * nodes**2
    log_prior -= np.logaddexp.reduce(log_prior)

    result = posterior_moments(
        np.empty(0),
        np.empty((0, 1)),
        np.empty(0),
        nodes[:, None],
        log_prior,
    )

    assert result.theta[0] == pytest.approx(0.0, abs=1e-12)
    assert result.se[0] == pytest.approx(1.0, abs=1e-7)
