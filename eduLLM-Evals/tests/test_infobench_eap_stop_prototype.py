import json

import pandas as pd

from scripts import run_infobench_eap_stop_prototype as prototype


def test_online_reproduction_accepts_flattened_run_row() -> None:
    result = {
        "scenario_order": json.dumps(["scenario_2", "scenario_1"]),
        "scenarios_administered": 2,
        "criteria_administered": 7,
        "stop_reason": "precision",
        "precision_reached": True,
        "theta_mwle": 0.375,
    }
    historical = pd.Series(
        {
            "cat_scenario_order": json.dumps(["scenario_2", "scenario_1"]),
            "cat_scenarios_administered": 2,
            "cat_criteria_administered": 7,
            "cat_stop_reason": "precision",
            "cat_precision_reached": True,
            "theta_cat_mwle": 0.375,
            "theta_reference": 0.25,
        }
    )

    checks = prototype._check_online_reproduction(result, 0.25, historical)

    assert checks["all_exact"] is True


def test_manifest_path_preserves_external_output_path(tmp_path) -> None:
    output = tmp_path / "metrics.json"

    assert prototype._manifest_path(output) == str(output.resolve())


def test_pair_statistics_distinguishes_two_median_estimands() -> None:
    frame = pd.DataFrame(
        {
            "scenarios_mean_online": [0.0, 100.0, 101.0],
            "scenarios_mean_eap": [50.0, 51.0, 102.0],
            "scenario_difference": [50.0, -49.0, 1.0],
            "target_rate_difference": [0.0, 0.0, 0.0],
            "theta_reference_online": [-1.0, 0.0, 1.0],
            "theta_mwle_online": [-1.0, 0.0, 1.0],
            "theta_reference_eap": [-1.0, 0.0, 1.0],
            "theta_mwle_eap": [-1.0, 0.0, 1.0],
            "theta_absolute_error_online": [0.0, 0.0, 0.0],
            "theta_absolute_error_eap": [0.0, 0.0, 0.0],
        }
    )

    metrics = prototype._pair_statistics(frame)

    assert metrics["median_paired_scenario_difference"] == 1.0
    assert metrics["difference_of_arm_median_scenarios"] == -49.0
