"""Contract and source-parity tests for the importable EduLLM calibrator."""

from __future__ import annotations

import numpy as np
import pytest

from olmo_eval.edullm.calibration import (
    FREE_2PL,
    GAUSS_HERMITE_QUADRATURE,
    LOG_SHRINKAGE_2PL,
    NORMAL_TRAPEZOID_QUADRATURE,
    ONE_PL,
    RETURNED_ITERATE_CONVERGENCE,
    SOURCE_BLOB,
    SOURCE_COMMIT,
    CalibrationConfig,
    CalibrationError,
    ConvergencePolicy,
    QuadratureConfig,
    fit_calibration,
)
from olmo_eval.edullm.skill_structure import SkillStructure


def _one_dimensional_structure() -> SkillStructure:
    return SkillStructure.identity(("instruction_following",), name="one_dimensional")


def _returned_policy(max_iterations: int = 100) -> ConvergencePolicy:
    return ConvergencePolicy(
        mode=RETURNED_ITERATE_CONVERGENCE,
        max_iterations=max_iterations,
        objective_tolerance=1e-5,
        consecutive_passes=2,
    )


def _one_dimensional_config(
    family: str,
    *,
    max_iterations: int = 100,
) -> CalibrationConfig:
    return CalibrationConfig(
        model_family=family,  # type: ignore[arg-type]
        quadrature=QuadratureConfig(
            method=NORMAL_TRAPEZOID_QUADRATURE,
            nodes_per_dimension=101,
            linear_bound=8.0,
        ),
        convergence=_returned_policy(max_iterations),
        estimate_latent_correlation=False,
        ridge=0.05,
        log_a_shrinkage=4.0,
    )


def _source_parity_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260805)
    theta = rng.standard_normal(100)
    difficulties = np.linspace(-1.25, 1.25, 6)
    probability = 1.0 / (1.0 + np.exp(-(theta[:, None] - difficulties[None, :])))
    responses = (rng.random(probability.shape) < probability).astype(float)
    return responses, np.ones((6, 1), dtype=int)


def test_returned_iterate_matches_authoritative_source_snapshot() -> None:
    responses, source_q = _source_parity_data()
    result = fit_calibration(
        responses,
        source_q,
        skill_structure=_one_dimensional_structure(),
        config=_one_dimensional_config(ONE_PL, max_iterations=300),
        missing_policy="exclude",
        item_ids=[f"item-{index}" for index in range(source_q.shape[0])],
        initial_loadings=np.ones_like(source_q, dtype=float),
        initial_difficulties=np.zeros(source_q.shape[0]),
    )

    # Frozen from origin/frq/infobench@b4ea2e8, blob 7e2b05ee, using the
    # identical seed/configuration above.
    expected_difficulties = np.array(
        [
            -1.28972264797593,
            -0.8883945817740629,
            -0.2355408941097086,
            0.14511980006687278,
            0.7859909695342243,
            1.4172182178303177,
        ]
    )
    assert np.array_equal(result.loadings, np.ones((6, 1)))
    assert np.allclose(result.difficulties, expected_difficulties, atol=1e-12, rtol=0)
    assert result.marginal_log_likelihood == pytest.approx(-373.5900360414086)
    assert result.converged is True
    assert result.termination_reason == "converged"
    assert result.returned_iterate_status == "converged_returned_iterate"
    assert result.iterations == 7
    assert result.convergence_diagnostics["returned_iterate_matches_last_trace"] is True
    assert result.provenance.source_commit == SOURCE_COMMIT
    assert result.provenance.source_blob == SOURCE_BLOB
    assert result.provenance.parameterization == "sigmoid(A @ theta - b)"


def test_no_decision_cells_and_all_missing_people_are_excluded() -> None:
    responses = np.array(
        [
            [1.0, 0.0, np.nan],
            [0.0, 1.0, 1.0],
            [np.nan, np.nan, np.nan],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0],
        ]
    )
    source_q = np.ones((3, 1), dtype=int)
    kwargs = {
        "skill_structure": _one_dimensional_structure(),
        "config": _one_dimensional_config(ONE_PL, max_iterations=4),
        "missing_policy": "exclude",
    }
    from_nan = fit_calibration(responses, source_q, **kwargs)  # type: ignore[arg-type]

    filled = np.nan_to_num(responses, nan=0.0)
    explicit_mask = np.isfinite(responses)
    filled[0, 2] = 1.0  # Deliberately opposite hidden values must not affect the fit.
    filled[2, :] = 1.0
    from_mask = fit_calibration(
        filled,
        source_q,
        observed_mask=explicit_mask,
        **kwargs,  # type: ignore[arg-type]
    )

    assert np.array_equal(from_nan.loadings, from_mask.loadings)
    assert np.array_equal(from_nan.difficulties, from_mask.difficulties)
    assert from_nan.marginal_log_likelihood == from_mask.marginal_log_likelihood
    assert from_nan.provenance.n_persons_supplied == 5
    assert from_nan.provenance.n_persons_fitted == 4
    assert from_nan.provenance.n_all_missing_persons_excluded == 1
    assert from_nan.provenance.n_observed_cells == 11
    assert from_nan.provenance.n_missing_cells == 4


@pytest.mark.parametrize("family", [FREE_2PL, ONE_PL, LOG_SHRINKAGE_2PL])
def test_all_three_explicit_families_preserve_q_mask(family: str) -> None:
    rng = np.random.default_rng(71)
    source_q = np.array([[1, 0], [0, 1], [1, 1], [1, 0], [0, 1], [1, 1]], dtype=int)
    responses = rng.integers(0, 2, size=(45, len(source_q))).astype(float)
    structure = SkillStructure.identity(("content", "format"), name="two_dimensional")
    config = CalibrationConfig(
        model_family=family,  # type: ignore[arg-type]
        quadrature=QuadratureConfig(
            method=GAUSS_HERMITE_QUADRATURE,
            nodes_per_dimension=3,
            linear_bound=None,
        ),
        convergence=ConvergencePolicy(
            mode=RETURNED_ITERATE_CONVERGENCE,
            max_iterations=2,
            objective_tolerance=1e-8,
        ),
        estimate_latent_correlation=True,
        ridge=0.05,
        log_a_shrinkage=4.0,
    )

    result = fit_calibration(
        responses,
        source_q,
        skill_structure=structure,
        config=config,
        missing_policy="exclude",
    )

    assert np.array_equal(result.loadings[source_q == 0], np.zeros((source_q == 0).sum()))
    if family == ONE_PL:
        assert np.array_equal(result.loadings, source_q.astype(float))
        assert result.n_free_loadings == 0
        assert result.n_fixed_loadings == int(source_q.sum())
    elif family == LOG_SHRINKAGE_2PL:
        assert np.all(result.loadings[source_q == 1] > 0)
    assert result.provenance.model_specification["family"] == family


def test_nonconvergence_returns_labeled_last_valid_iterate() -> None:
    responses, source_q = _source_parity_data()
    result = fit_calibration(
        responses,
        source_q,
        skill_structure=_one_dimensional_structure(),
        config=_one_dimensional_config(FREE_2PL, max_iterations=1),
        missing_policy="exclude",
    )

    assert result.converged is False
    assert result.termination_reason == "max_iterations"
    assert result.returned_iterate_status == "last_valid_returned_iterate"
    assert result.iterations == 1
    assert np.all(np.isfinite(result.loadings))
    assert np.all(np.isfinite(result.difficulties))
    assert np.isfinite(result.marginal_log_likelihood)


def test_tensor_quadrature_guard_fails_before_fit() -> None:
    structure = SkillStructure.identity(
        ("one", "two", "three", "four", "five"), name="five_dimensional"
    )
    source_q = np.eye(5, dtype=int)
    responses = np.tile([0.0, 1.0], (5, 3)).T[:10, :5]
    config = CalibrationConfig(
        model_family=ONE_PL,
        quadrature=QuadratureConfig(
            method=GAUSS_HERMITE_QUADRATURE,
            nodes_per_dimension=7,
            linear_bound=None,
            max_total_nodes=5_000,
        ),
        convergence=_returned_policy(max_iterations=2),
        estimate_latent_correlation=False,
    )

    with pytest.raises(CalibrationError, match="16,807 nodes"):
        fit_calibration(
            responses,
            source_q,
            skill_structure=structure,
            config=config,
            missing_policy="exclude",
        )


def test_contract_rejects_implicit_or_invalid_scientific_choices() -> None:
    with pytest.raises(TypeError):
        CalibrationConfig(  # type: ignore[call-arg]
            quadrature=QuadratureConfig(
                method=GAUSS_HERMITE_QUADRATURE,
                nodes_per_dimension=3,
                linear_bound=None,
            ),
            convergence=_returned_policy(),
            estimate_latent_correlation=False,
        )
    with pytest.raises(CalibrationError, match="missing_policy"):
        fit_calibration(
            [[0.0], [1.0]],
            [[1]],
            skill_structure=_one_dimensional_structure(),
            config=_one_dimensional_config(ONE_PL),
            missing_policy="fail",  # type: ignore[arg-type]
        )
    with pytest.raises(CalibrationError, match="observed responses"):
        fit_calibration(
            [[0.0], [2.0]],
            [[1]],
            skill_structure=_one_dimensional_structure(),
            config=_one_dimensional_config(ONE_PL),
            missing_policy="exclude",
        )


def test_count_fields_reject_bool_and_float_values() -> None:
    with pytest.raises(CalibrationError, match="nodes_per_dimension"):
        QuadratureConfig(
            method=GAUSS_HERMITE_QUADRATURE,
            nodes_per_dimension=3.0,  # type: ignore[arg-type]
            linear_bound=None,
        )
    with pytest.raises(CalibrationError, match="max_total_nodes"):
        QuadratureConfig(
            method=GAUSS_HERMITE_QUADRATURE,
            nodes_per_dimension=3,
            linear_bound=None,
            max_total_nodes=False,  # type: ignore[arg-type]
        )
    quadrature = QuadratureConfig(
        method=GAUSS_HERMITE_QUADRATURE,
        nodes_per_dimension=3,
        linear_bound=None,
    )
    with pytest.raises(CalibrationError, match="n_dimensions"):
        quadrature.total_nodes(1.0)  # type: ignore[arg-type]
    with pytest.raises(CalibrationError, match="max_iterations"):
        ConvergencePolicy(
            mode=RETURNED_ITERATE_CONVERGENCE,
            max_iterations=True,  # type: ignore[arg-type]
            objective_tolerance=1e-5,
        )
    with pytest.raises(CalibrationError, match="consecutive_passes"):
        ConvergencePolicy(
            mode=RETURNED_ITERATE_CONVERGENCE,
            max_iterations=10,
            objective_tolerance=1e-5,
            consecutive_passes=1.5,  # type: ignore[arg-type]
        )
