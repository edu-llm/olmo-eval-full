"""Focused synthetic tests for dimension-generic offline CAT replay."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import approx_fprime

from scripts import offline_engine_driver as offline_driver
from scripts import scenario_cat_lib as scat


def _five_dim_fixture(tmp_path):
    dims = ("content", "format", "number", "style", "linguistic")
    corr = np.eye(5)
    corr[0, 3] = corr[3, 0] = 0.45
    records = []
    scenarios = []
    values = {}
    # Two single-loading scenarios per dimension make the administered loading matrix full rank.
    for k, dim in enumerate(dims):
        for rep in range(2):
            sid = f"s{k}_{rep}"
            cid = f"{sid}_c01"
            q = {d: int(d == dim) for d in dims}
            records.append(
                {
                    "criterion_id": cid,
                    "scenario_id": sid,
                    "criterion": f"criterion {cid}",
                    "q_mapping": q,
                    "q_modeled": q,
                    "difficulty": -0.2 if rep == 0 else 0.2,
                    "discrimination": {d: (1.1 if d == dim else 0.0) for d in dims},
                    "irt_params": {
                        "source": "calibrated-m2pl-test",
                        "skills_order": list(dims),
                        "latent_correlation": corr.tolist(),
                    },
                    "status": "approved",
                }
            )
            scenarios.append(
                {"scenario_id": sid, "prompt": sid, "criterion_ids": [cid], "modality": "text"}
            )
            values[cid] = int((k + rep) % 2 == 0)
    bank_path = tmp_path / "bank.jsonl"
    scenario_path = tmp_path / "scenarios.jsonl"
    bank_path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    scenario_path.write_text(
        "".join(json.dumps(r) + "\n" for r in scenarios), encoding="utf-8"
    )
    return dims, corr, bank_path, scenario_path, pd.Series(values, name="model-a")


def test_correlated_prior_is_used_by_eap_without_observations():
    corr = np.array([[1.0, 0.65], [0.65, 1.0]])
    quad = scat.build_quadrature(2, 11, corr)
    estimate = scat.batch_eap(
        np.array([np.nan]), np.array([[1.0, 0.0]]), np.array([0.0]), quad
    )
    assert estimate.n_items == 0
    assert np.allclose(estimate.theta, 0.0, atol=1e-8)
    assert np.allclose(np.diag(estimate.covariance), 1.0, atol=0.03)
    assert estimate.covariance[0, 1] == pytest.approx(0.65, abs=0.04)


def test_missing_response_is_marginalized_not_converted_to_fail():
    quad = scat.build_quadrature(1, 11)
    A = np.array([[1.2], [1.2]])
    b = np.array([0.0, 0.0])
    with_missing = scat.batch_eap(np.array([1.0, np.nan]), A, b, quad)
    first_only = scat.batch_eap(np.array([1.0, 0.0]), A, b, quad, [0])
    assert with_missing.n_items == 1
    assert np.allclose(with_missing.theta, first_only.theta)
    assert np.allclose(with_missing.covariance, first_only.covariance)


def test_pirt_pass_rate_check_uses_only_observed_fitted_items():
    responses = np.array([1.0, np.nan, 0.0])
    A = np.array([[1.0], [2.0], [1.0]])
    b = np.array([-0.4, 9.0, 0.4])
    check = scat.pass_rate_check(responses, A, b, np.array([0.0]))
    expected = float(np.mean([1.0 / (1.0 + np.exp(-0.4)), 1.0 / (1.0 + np.exp(0.4))]))
    assert check["n_items"] == 2
    assert check["observed_pass_rate"] == pytest.approx(0.5)
    assert check["predicted_pass_rate"] == pytest.approx(expected)
    assert check["error"] == pytest.approx(expected - 0.5)


def test_mwle_analytic_gradient_matches_finite_difference():
    A = np.array([[1.0, 0.0], [0.0, 1.2], [0.8, 0.6], [1.1, -0.3]])
    b = np.array([-0.2, 0.1, 0.3, -0.1])
    y = np.array([1.0, 0.0, 1.0, 0.0])
    theta = np.array([0.25, -0.35])
    ridge = 1e-5

    def objective(x):
        return scat._mwle_neg_obj_grad(x, A, b, y, ridge)[0]

    _, analytic = scat._mwle_neg_obj_grad(theta, A, b, y, ridge)
    numeric = approx_fprime(theta, objective, 1e-7)
    assert np.allclose(analytic, numeric, atol=2e-5, rtol=2e-5)


def test_five_dim_recorded_replay_supports_cat_dopt_and_baseline(tmp_path):
    dims, corr, bank_path, scenario_path, row = _five_dim_fixture(tmp_path)
    fitted = scat.load_fitted_bank(bank_path)
    scenarios = scat.load_scenario_records(scenario_path)
    quad = scat.build_quadrature(5, 3, fitted.latent_correlation)
    assert fitted.dims == dims
    assert np.allclose(fitted.latent_correlation, corr)

    outputs = []
    for mode, selection in (("cat", "trace"), ("cat", "dopt"), ("baseline", "trace")):
        outputs.append(
            scat.run_recorded_model(
                "model-a",
                row,
                fitted,
                scenarios,
                quad,
                scat.RunSpec(
                    seed=7,
                    top_n=1,
                    max_se=0.01,
                    min_evals_per_skill=0,
                    max_scenarios=10,
                    mode=mode,
                    selection=selection,
                ),
            )
        )
    for result in outputs:
        assert set(result["theta_online"]) == set(dims)
        assert set(result["theta_eap"]) == set(dims)
        assert set(result["theta_mwle"]) == set(dims)
        assert result["criteria_administered"] == 10
        assert result["judge_lookups"] == 10
        assert result["mwle_converged"] is True
        assert result["n_observed_fitted_items"] == 10
        assert result["observed_fitted_pass_rate"] == pytest.approx(0.5)
        assert set(result["pass_rate_checks"]) == {"online", "eap", "mwle", "full_eap"}


def test_model_bank_drops_missing_cell_before_engine(tmp_path):
    _, _, bank_path, scenario_path, row = _five_dim_fixture(tmp_path)
    missing_cid = row.index[0]
    row.loc[missing_cid] = np.nan
    fitted = scat.load_fitted_bank(bank_path)
    scenarios = scat.load_scenario_records(scenario_path)
    quad = scat.build_quadrature(5, 3, fitted.latent_correlation)
    result = scat.run_recorded_model(
        "model-a",
        row,
        fitted,
        scenarios,
        quad,
        scat.RunSpec(
            seed=3,
            top_n=1,
            max_se=0.01,
            min_evals_per_skill=0,
            max_scenarios=10,
            mode="baseline",
        ),
    )
    assert missing_cid not in result["criterion_order"]
    assert result["criteria_administered"] == 9
    assert result["judge_lookups"] == 9


def test_loader_requires_explicit_policy_for_nonpositive_discrimination(tmp_path):
    dims, _, bank_path, _, _ = _five_dim_fixture(tmp_path)
    rows = [json.loads(line) for line in bank_path.read_text().splitlines()]
    rows[0]["discrimination"][dims[0]] = -0.5
    bank_path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    with pytest.raises(scat.OfflineStudyError, match="nonpositive"):
        scat.load_fitted_bank(bank_path)
    dropped = scat.load_fitted_bank(bank_path, negative_policy="drop")
    assert rows[0]["criterion_id"] not in dropped.criterion_ids
    assert dropped.dropped_negative_items == (rows[0]["criterion_id"],)


def test_normal_trapezoid_quadrature_is_normalized_and_symmetric():
    quadrature = scat.build_quadrature(
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


def test_normal_trapezoid_quadrature_rejects_multidimensional_use():
    with pytest.raises(scat.OfflineStudyError, match="only for 1D"):
        scat.build_quadrature(
            2,
            21,
            np.eye(2),
            method="normal_trapezoid",
        )


def test_scipy_gauss_hermite_supports_high_order_crosscheck():
    quadrature = scat.build_quadrature(
        1,
        1601,
        np.eye(1),
        max_nodes=2000,
        method="gauss_hermite_scipy",
    )
    assert quadrature.method == "gauss_hermite_scipy"
    assert quadrature.nodes_per_dim == 1601
    assert 0 < len(quadrature.grid) <= 1601
    assert np.isfinite(quadrature.grid).all()
    assert np.isfinite(quadrature.log_prior).all()
    assert np.isclose(np.exp(quadrature.log_prior).sum(), 1.0)


def test_offline_driver_writes_shareable_cat_and_random_artifacts(tmp_path):
    _, _, bank_path, scenario_path, row = _five_dim_fixture(tmp_path)
    matrix_path = tmp_path / "matrix.csv"
    pd.DataFrame([row], index=["model-a"]).rename_axis("model").to_csv(matrix_path)
    out_dir = tmp_path / "study"
    args = offline_driver.build_parser().parse_args(
        [
            "--bank",
            str(bank_path),
            "--matrix",
            str(matrix_path),
            "--scenarios",
            str(scenario_path),
            "--out-dir",
            str(out_dir),
            "--mode",
            "both",
            "--grid",
            "3",
            "--top-n",
            "1",
            "--max-se",
            "0.01",
            "--min-evals-per-skill",
            "0",
            "--max-scenarios",
            "10",
        ]
    )
    assert offline_driver.run(args) == 0
    assert {"per_model.csv", "metrics.json", "manifest.json", "paired_cat_vs_random.csv"} <= {
        path.name for path in out_dir.iterdir()
    }
    per_model = pd.read_csv(out_dir / "per_model.csv")
    assert set(per_model["mode"]) == {"cat", "baseline"}
    assert "observed_fitted_pass_rate" in per_model
    assert "pirt_predicted_pass_rate_full_eap" in per_model
    metrics = json.loads((out_dir / "metrics.json").read_text())
    assert set(metrics["modes"]["cat"]["pass_rate_calibration"]) == {
        "online",
        "eap",
        "mwle",
        "full_eap",
    }
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["bank"]["dims"] == ["content", "format", "number", "style", "linguistic"]
    assert manifest["study"]["baseline_max_scenarios"] == 10
