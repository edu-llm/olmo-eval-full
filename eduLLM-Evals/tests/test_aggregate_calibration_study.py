"""Focused synthetic tests for the calibration-study aggregator."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "aggregate_calibration_study.py"
SPEC = importlib.util.spec_from_file_location("aggregate_calibration_study", SCRIPT)
assert SPEC and SPEC.loader
agg = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = agg
SPEC.loader.exec_module(agg)


def _write_structure(
    root: Path,
    name: str,
    n_dims: int,
    fold_metric: str,
    fold_values: list[float],
) -> agg.StructureSpec:
    base = root / name
    kfold = base / "kfold"
    kfold.mkdir(parents=True)
    core = base / "calibration_mirt_manifest.json"
    core.write_text(
        json.dumps(
            {
                "skills_order": [f"d{index}" for index in range(n_dims)],
                "n_items_fit": 20,
                "latent_correlation": [
                    [1.0 if row == column else 0.4 for column in range(n_dims)]
                    for row in range(n_dims)
                ],
                "comparison": {
                    "multi": {
                        "loglik": -100.0,
                        "n_params": 30,
                        "aic": 260.0,
                        "bic": 280.0,
                    }
                },
                "em": {"multi_converged": True},
            }
        ),
        encoding="utf-8",
    )
    summary = {
        "config": {
            "structure": {
                "name": name,
                "dimensions": [{"label": f"d{index}"} for index in range(n_dims)],
            }
        },
        "metric_kind": "heldout_person_disjoint_scenario_prediction",
        "pooled_oos": {
            "log_loss": 0.5,
            "accuracy": 0.75,
            "auc": 0.8,
            "brier": 0.17,
        },
        "item_param_stability": {
            "b": {"median_pairwise_corr": 0.8},
            "a_d0": {"median_pairwise_corr": 0.7},
        },
    }
    (kfold / "kfold_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    pd.DataFrame(
        {
            "fold": range(len(fold_values)),
            "fit_converged": [True] * len(fold_values),
            fold_metric: fold_values,
        }
    ).to_csv(kfold / "metrics_per_fold.csv", index=False)
    return agg.StructureSpec(name=name, core_manifest=core, kfold_dir=kfold)


def _structure_specs(tmp_path: Path, metric: str = "log_loss") -> list[agg.StructureSpec]:
    return [
        _write_structure(
            tmp_path,
            "full_5d",
            5,
            metric,
            [0.48, 0.52, 0.50, 0.51, 0.49],
        ),
        _write_structure(
            tmp_path,
            "reduced_2d",
            2,
            metric,
            [0.504, 0.504, 0.504, 0.504, 0.504],
        ),
        _write_structure(
            tmp_path,
            "overall_1d",
            1,
            metric,
            [0.52, 0.52, 0.52, 0.52, 0.52],
        ),
    ]


def test_one_se_rule_selects_simpler_eligible_structure(tmp_path: Path) -> None:
    frame, selection = agg.compare_structures(_structure_specs(tmp_path))

    assert selection["rule"]["best_mean_structure"] == "full_5d"
    assert selection["automatic_recommendation"] == "reduced_2d"
    assert selection["selected_structure"] == "reduced_2d"
    eligible = set(frame.loc[frame["one_se_eligible"], "structure"])
    assert eligible == {"full_5d", "reduced_2d"}


def test_override_preserves_automatic_recommendation(tmp_path: Path) -> None:
    _, selection = agg.compare_structures(
        _structure_specs(tmp_path), override="full_5d"
    )

    assert selection["automatic_recommendation"] == "reduced_2d"
    assert selection["selected_structure"] == "full_5d"
    assert selection["override_used"] is True
    assert selection["selected_was_one_se_eligible"] is True


def test_maximize_direction_uses_lower_one_se_boundary(tmp_path: Path) -> None:
    specs = [
        _write_structure(
            tmp_path,
            "full_5d",
            5,
            "accuracy",
            [0.78, 0.82, 0.80, 0.81, 0.79],
        ),
        _write_structure(
            tmp_path,
            "reduced_2d",
            2,
            "accuracy",
            [0.796, 0.796, 0.796, 0.796, 0.796],
        ),
        _write_structure(
            tmp_path,
            "overall_1d",
            1,
            "accuracy",
            [0.77, 0.77, 0.77, 0.77, 0.77],
        ),
    ]
    _, selection = agg.compare_structures(
        specs, primary_metric="accuracy", direction="maximize"
    )

    assert selection["automatic_recommendation"] == "reduced_2d"
    assert selection["rule"]["boundary"] < selection["rule"]["best_mean"]


def _write_cat_csv(path: Path, *, adaptive: bool) -> None:
    reference = [-2.0, -1.0, -0.2, 0.4, 1.1, 2.0]
    if adaptive:
        estimate = [-1.9, -1.05, -0.1, 0.5, 1.0, 2.1]
        scenarios = [15, 16, 17, 18, 19, 20]
    else:
        estimate = [-1.0, -1.5, 0.3, -0.2, 1.8, 1.3]
        scenarios = [20, 21, 22, 23, 24, 25]
    pd.DataFrame(
        {
            "model": [f"m{index}" for index in range(len(reference))],
            "theta_full_content": reference,
            "theta_mwle_content": estimate,
            "scenarios_administered": scenarios,
            "criteria_administered": [value * 4 for value in scenarios],
            "precision_reached": [True, True, True, False, True, True],
        }
    ).to_csv(path, index=False)


def _cat_specs(tmp_path: Path) -> list[agg.CatSweepSpec]:
    adaptive = tmp_path / "adaptive.csv"
    random = tmp_path / "random.csv"
    _write_cat_csv(adaptive, adaptive=True)
    _write_cat_csv(random, adaptive=False)
    common = {
        "group": "selected_structure",
        "reference_prefix": "theta_full_",
        "estimate_prefixes": ("theta_mwle_",),
        "skills": ("content",),
    }
    return [
        agg.CatSweepSpec(name="adaptive", csv_path=adaptive, **common),
        agg.CatSweepSpec(name="random", csv_path=random, baseline=True, **common),
    ]


def test_paired_cat_bootstrap_is_deterministic_and_preserves_length_difference(
    tmp_path: Path,
) -> None:
    first_metrics, first_differences, _ = agg.aggregate_cat_sweeps(
        _cat_specs(tmp_path), b=250, seed=17
    )
    second_metrics, second_differences, _ = agg.aggregate_cat_sweeps(
        _cat_specs(tmp_path), b=250, seed=17
    )

    pd.testing.assert_frame_equal(first_metrics, second_metrics)
    pd.testing.assert_frame_equal(first_differences, second_differences)
    row = first_differences[
        (first_differences["run"] == "adaptive")
        & (first_differences["metric"] == "mean_scenarios_administered")
    ].iloc[0]
    assert row["run_minus_baseline"] == -5.0
    assert row["ci_lower"] == -5.0
    assert row["ci_upper"] == -5.0


def test_run_study_writes_tables_selection_and_plain_language_summary(
    tmp_path: Path,
) -> None:
    out = tmp_path / "out"
    result = agg.run_study(
        _structure_specs(tmp_path / "structures"),
        _cat_specs(tmp_path),
        out,
        bootstrap_b=50,
        bootstrap_seed=3,
        make_figures=False,
    )

    assert result["selection"]["selected_structure"] == "reduced_2d"
    expected = {
        "structure_comparison.csv",
        "selection.json",
        "CALIBRATION_STUDY_SUMMARY.md",
        "cat_sweep_metrics.csv",
        "cat_sweep_paired_differences.csv",
        "cat_sweep_bootstrap.json",
    }
    expected |= {
        "study_results.json",
        "estimator_oos_metrics.csv",
        "sensitivity_summary.csv",
        "order_stability_summary.csv",
        "parameter_uncertainty_summary.csv",
    }
    assert expected == {path.name for path in result["outputs"].values()}
    assert all((out / name).is_file() for name in expected)
    summary = (out / "CALIBRATION_STUDY_SUMMARY.md").read_text(encoding="utf-8")
    assert "reduced_2d" in summary
    assert "paired model-bootstrap 95% confidence intervals" in summary
    consolidated = json.loads((out / "study_results.json").read_text(encoding="utf-8"))
    assert consolidated["schema_version"] == "calibration-study-results-v2"
    assert set(consolidated["optional_studies_not_supplied"]) == {
        "estimator_oos_kfold",
        "sensitivity",
        "order_stability",
        "parameter_uncertainty",
    }


def _write_estimator_oos(root: Path) -> agg.EstimatorOOSSpec:
    root.mkdir(parents=True)
    reference = [-2.0, -1.4, -0.8, -0.2, 0.3, 0.9, 1.5, 2.1]
    observed = [0.10, 0.18, 0.28, 0.42, 0.55, 0.68, 0.80, 0.90]
    pd.DataFrame(
        {
            "model": [f"model-{index}" for index in range(len(reference))],
            "status": ["ok"] * len(reference),
            "mwle_converged": [True] * len(reference),
            "theta_ref_content": reference,
            "theta_online_content": [0.6 * value for value in reference],
            "theta_eap_content": [0.8 * value for value in reference],
            "theta_mwle_content": reference,
            "observed_fitted_pass_rate": observed,
            "pirt_predicted_pass_rate_online": [value + 0.05 for value in observed],
            "pirt_predicted_pass_rate_eap": [value + 0.03 for value in observed],
            "pirt_predicted_pass_rate_mwle": [value + 0.01 for value in observed],
            "pirt_predicted_pass_rate_full_eap": observed,
        }
    ).to_csv(root / "oos_per_model.csv", index=False)
    (root / "metrics_aggregate.json").write_text("{}", encoding="utf-8")
    (root / "manifest.json").write_text("{}", encoding="utf-8")
    pd.DataFrame(
        [
            {
                "fold": "pooled",
                "estimator": estimator,
                "n": len(reference),
                "mae": error,
                "bias": error,
                "r": 1.0,
            }
            for estimator, error in (
                ("online", 0.05),
                ("eap", 0.03),
                ("mwle", 0.01),
                ("full_eap", 0.0),
            )
        ]
    ).to_csv(root / "pass_rate_calibration.csv", index=False)
    return agg.EstimatorOOSSpec(root)


def _write_order_stability(root: Path) -> agg.OrderStabilitySpec:
    root.mkdir(parents=True)
    (root / "metrics.json").write_text(
        json.dumps(
            {
                "n_models": 8,
                "seeds": [1000, 1001, 1002],
                "n_run_failures": 0,
                "n_mwle_failures": 0,
                "config": {"max_se": 0.3},
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "estimator": "mwle",
                "dimension": "content",
                "n_models_with_spread": 8,
                "mean_sd": 0.02,
                "median_sd": 0.018,
                "max_sd": 0.04,
                "mean_range": 0.03,
                "median_range": 0.025,
                "max_range": 0.07,
            }
        ]
    ).to_csv(root / "spread_summary.csv", index=False)
    return agg.OrderStabilitySpec(root)


def _write_uncertainty(
    root: Path, name: str, target: float, total_se: float
) -> agg.ParameterUncertaintySpec:
    root.mkdir(parents=True)
    (root / "metrics.json").write_text(
        json.dumps(
            {
                "n_boot_requested": 100,
                "n_boot_fit_success": 98,
                "n_boot_fit_failures": 2,
                "n_base_replay_failures": 0,
                "config": {"max_se": target},
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "model": f"model-{index}",
                "se_target": target,
                "estimator": "mwle",
                "dimension": "content",
                "se_ability": target - 0.02,
                "se_param": 0.1,
                "se_total": total_se + index / 1000,
                "reliable": True,
            }
            for index in range(8)
        ]
    ).to_csv(root / "ability_se_components.csv", index=False)
    return agg.ParameterUncertaintySpec(name, root, target)


def _write_multi_target_uncertainty(root: Path) -> agg.ParameterUncertaintySpec:
    root.mkdir(parents=True)
    targets = (0.25, 0.30)
    (root / "metrics.json").write_text(
        json.dumps(
            {
                "se_targets": list(targets),
                "n_boot_requested": 100,
                "n_boot_fit_success": 99,
                "n_boot_fit_failures": 1,
                "n_base_replay_failures": 0,
                "config": {},
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "model": f"model-{index}",
                "se_target": target,
                "estimator": "mwle",
                "dimension": "content",
                "se_ability": target - 0.02,
                "se_param": 0.1,
                "se_total": target + 0.02 + index / 1000,
                "reliable": True,
            }
            for target in targets
            for index in range(8)
        ]
    ).to_csv(root / "ability_se_components.csv", index=False)
    return agg.ParameterUncertaintySpec("multi_target", root)


def test_estimator_oos_model_bootstrap_is_deterministic(tmp_path: Path) -> None:
    spec = _write_estimator_oos(tmp_path / "estimator")
    first, first_meta = agg.aggregate_estimator_oos(spec, b=200, seed=9)
    second, second_meta = agg.aggregate_estimator_oos(spec, b=200, seed=9)

    pd.testing.assert_frame_equal(first, second)
    assert first_meta == second_meta
    mwle = first[
        (first["estimator"] == "mwle")
        & (first["skill"] == "content")
        & (first["metric"] == "r")
    ].iloc[0]
    assert abs(mwle["estimate"] - 1.0) < 1e-12
    assert abs(mwle["ci_lower"] - 1.0) < 1e-12
    assert abs(mwle["ci_upper"] - 1.0) < 1e-12
    pass_rate = first[
        (first["estimator"] == "mwle")
        & (first["skill"] == "__pass_rate__")
        & (first["metric"] == "pass_rate_mae")
    ].iloc[0]
    assert abs(pass_rate["estimate"] - 0.01) < 1e-12
    assert first_meta["pass_rate_calibration"]["status"] == "available"


def test_one_parameter_bootstrap_run_preserves_multiple_se_targets(tmp_path: Path) -> None:
    frame, metadata = agg.aggregate_parameter_uncertainty(
        [_write_multi_target_uncertainty(tmp_path / "multi")]
    )

    assert set(frame["se_target"]) == {0.25, 0.30}
    assert len(frame) == 2
    assert metadata["runs"]["multi_target"]["se_targets"] == [0.25, 0.30]


def test_optional_studies_are_consolidated_without_new_model_calls(tmp_path: Path) -> None:
    estimator = _write_estimator_oos(tmp_path / "estimator")
    order = _write_order_stability(tmp_path / "order")
    uncertainties = [
        _write_uncertainty(tmp_path / "se25", "se_0p25", 0.25, 0.27),
        _write_uncertainty(tmp_path / "se30", "se_0p30", 0.30, 0.32),
    ]
    sensitivity_spec = _write_structure(
        tmp_path / "sensitivity",
        "grid3_ridge0p01",
        2,
        "log_loss",
        [0.49, 0.50, 0.51, 0.50, 0.50],
    )
    out = tmp_path / "out"
    result = agg.run_study(
        _structure_specs(tmp_path / "structures"),
        _cat_specs(tmp_path),
        out,
        bootstrap_b=100,
        bootstrap_seed=4,
        estimator_oos_spec=estimator,
        order_stability_spec=order,
        parameter_uncertainty_specs=uncertainties,
        sensitivity_specs=[
            agg.SensitivitySpec(
                "grid3_ridge0p01",
                sensitivity_spec.core_manifest,
                sensitivity_spec.kfold_dir,
                grid=3,
                ridge=0.01,
            )
        ],
        make_figures=False,
    )

    assert len(result["estimator_oos"]) > 0
    assert len(result["parameter_uncertainty"]) == 2
    consolidated = json.loads((out / "study_results.json").read_text(encoding="utf-8"))
    assert consolidated["optional_studies_not_supplied"] == []
    assert consolidated["estimator_oos_kfold"]["metadata"]["bootstrap_B"] == 100
    assert len(consolidated["parameter_uncertainty"]["rows"]) == 2
    summary = (out / "CALIBRATION_STUDY_SUMMARY.md").read_text(encoding="utf-8")
    assert "## Held-out estimator recovery" in summary
    assert "### p-IRT pass-rate calibration" in summary
    assert "## Grid and ridge sensitivity" in summary
    assert "## Scenario-order stability" in summary
    assert "## Item-parameter uncertainty and total SE" in summary


def test_extended_spec_schema_loads_optional_artifact_directories(tmp_path: Path) -> None:
    structure = _write_structure(
        tmp_path / "structures", "full_2d", 2, "log_loss", [0.5] * 5
    )
    cat_path = tmp_path / "cat.csv"
    _write_cat_csv(cat_path, adaptive=True)
    spec_path = tmp_path / "study.json"
    spec_path.write_text(
        json.dumps(
            {
                "structures": [
                    {
                        "name": "full_2d",
                        "core_manifest": str(structure.core_manifest),
                        "kfold_dir": str(structure.kfold_dir),
                    }
                ],
                "cat_sweeps": [{"name": "cat", "csv": str(cat_path)}],
                "estimator_oos_kfold": {"dir": "estimator", "name": "heldout"},
                "order_stability": {"dir": "order"},
                "parameter_uncertainty": [
                    {"name": "se25", "dir": "uncertainty/se25", "se_target": 0.25}
                ],
                "sensitivity": [
                    {"name": "grid3", "dir": "sensitivity/grid3", "grid": 3}
                ],
            }
        ),
        encoding="utf-8",
    )

    loaded = agg.load_extended_study_spec(spec_path)

    assert loaded.estimator_oos.name == "heldout"
    assert loaded.estimator_oos.dir_path == tmp_path / "estimator"
    assert loaded.order_stability.dir_path == tmp_path / "order"
    assert loaded.parameter_uncertainty[0].se_target == 0.25
    assert loaded.sensitivity[0].core_manifest == (
        tmp_path / "sensitivity/grid3/fit/calibration_mirt_manifest.json"
    )


def test_key_figures_are_written_when_matplotlib_is_available(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    comparison, _ = agg.compare_structures(_structure_specs(tmp_path / "structures"))
    cat_metrics, _, _ = agg.aggregate_cat_sweeps(_cat_specs(tmp_path), b=20, seed=2)
    estimator, _ = agg.aggregate_estimator_oos(
        _write_estimator_oos(tmp_path / "estimator"), b=20, seed=2
    )
    uncertainty_specs = [
        _write_uncertainty(tmp_path / "se25", "se25", 0.25, 0.27),
        _write_uncertainty(tmp_path / "se30", "se30", 0.30, 0.32),
    ]
    uncertainty, _ = agg.aggregate_parameter_uncertainty(uncertainty_specs)

    figures = agg.generate_key_figures(
        tmp_path / "figures_out", comparison, cat_metrics, estimator, uncertainty
    )

    assert set(figures) == {
        "structure_cv",
        "cat_length_precision_tradeoff",
        "estimator_recovery",
        "total_se_vs_target",
    }
    assert all(value["status"] == "generated" for value in figures.values())
    for value in figures.values():
        assert (tmp_path / "figures_out" / value["path"]).is_file()
