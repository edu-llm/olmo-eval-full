"""Focused fail-closed tests for the append-only v2 numerical follow-up."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_infobench_v2_numerical_followup.py"
CONFIG = ROOT / "configs" / "infobench_v2_numerical_followup_v1.json"


def _load_module():
    spec = importlib.util.spec_from_file_location("infobench_v2_numerical_followup", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


followup = _load_module()


SPEC_IDS = [
    "1pl_fixed_a1",
    "log_shrinkage_2pl_lambda16",
    "log_shrinkage_2pl_lambda4",
    "free_2pl_ridge_0p1",
    "free_2pl_ridge_0p01",
    "free_2pl_ridge_0p001",
]


def _gates(value: bool):
    return {spec_id: {"passed": value} for spec_id in SPEC_IDS}


def test_config_freezes_all_six_and_both_successive_panels() -> None:
    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    followup._validate_exact_config(raw)
    assert raw["audit_disclosure"]["no_inherited_passes"] is True
    fit = raw["fit_grid_followup"]
    assert fit["always_run_both_fit_comparisons"] is True
    assert fit["required_comparisons"] == ["81_vs_101", "101_vs_121"]
    assert fit["lock_rule"] == {
        "both_comparisons_pass_all_six": 81,
        "first_fails_second_passes_all_six": 101,
        "otherwise": None,
    }


def test_stronger_successive_rule_locks_81_only_when_both_pass() -> None:
    result = followup.select_common_fit_grid(SPEC_IDS, _gates(True), _gates(True))
    assert result["passed"] is True
    assert result["locked_common_fit_grid"] == 81
    assert result["always_ran_both_fit_comparisons"] is True

    second_failure = _gates(True)
    second_failure[SPEC_IDS[-1]] = {"passed": False}
    blocked = followup.select_common_fit_grid(SPEC_IDS, _gates(True), second_failure)
    assert blocked["passed"] is False
    assert blocked["locked_common_fit_grid"] is None


def test_second_panel_can_lock_101_only_after_first_panel_fails() -> None:
    first = _gates(True)
    first[SPEC_IDS[0]] = {"passed": False}
    result = followup.select_common_fit_grid(SPEC_IDS, first, _gates(True))
    assert result["passed"] is True
    assert result["locked_common_fit_grid"] == 101

    incomplete = followup.select_common_fit_grid(
        SPEC_IDS, first, {spec_id: {"passed": True} for spec_id in SPEC_IDS[:-1]}
    )
    assert incomplete["passed"] is False
    assert incomplete["second_comparison_complete"] is False


def test_common_support_theta_gate_is_blocking() -> None:
    thresholds = {
        "maximum_median_absolute_theta_shift": 0.02,
        "maximum_p95_absolute_theta_shift": 0.05,
        "maximum_absolute_theta_shift": 0.005,
    }
    rows = pd.DataFrame(
        {
            "scope": ["full", "outer_0", "outer_1"],
            "model": ["m1", "m2", "m3"],
            "support": ["full_bank", "administration_only", "full_bank"],
            "theta_lower": [-1.0, 0.0, 1.0],
            "theta_upper": [-0.999, 0.001, 1.001],
        }
    )
    passed = followup.theta_shift_gate(
        rows,
        lower_column="theta_lower",
        upper_column="theta_upper",
        thresholds=thresholds,
    )
    assert passed["passed"] is True
    rows.loc[1, "theta_upper"] = 0.006
    failed = followup.theta_shift_gate(
        rows,
        lower_column="theta_lower",
        upper_column="theta_upper",
        thresholds=thresholds,
    )
    assert failed["passed"] is False
    assert failed["maximum_absolute_theta_shift"] == 0.006


def _eap_gate(theta: bool, bound8: bool, bound10: bool):
    return {
        "all_theta_comparisons_passed": theta,
        "tail_gates": {
            "bound8": {"passed": bound8},
            "bound10": {"passed": bound10},
        },
    }


def test_one_global_eap_profile_and_no_spec_selection() -> None:
    bound8 = {spec_id: _eap_gate(True, True, True) for spec_id in SPEC_IDS}
    retained = followup.select_common_eap_profile(SPEC_IDS, bound8)
    assert retained["passed"] is True
    assert retained["locked_common_eap_profile"]["linear_bound"] == 8.0
    assert retained["locked_common_eap_profile"]["eap_grid"] == 401
    assert retained["selection_performed"] is False

    bound10 = {spec_id: _eap_gate(True, False, True) for spec_id in SPEC_IDS}
    widened = followup.select_common_eap_profile(SPEC_IDS, bound10)
    assert widened["passed"] is True
    assert widened["locked_common_eap_profile"]["linear_bound"] == 10.0
    assert widened["locked_common_eap_profile"]["eap_grid"] == 501

    bound10[SPEC_IDS[-1]] = _eap_gate(False, False, True)
    blocked = followup.select_common_eap_profile(SPEC_IDS, bound10)
    assert blocked["passed"] is False
    assert blocked["locked_common_eap_profile"] is None


def test_plan_only_verifies_parent_and_schedules_both_new_grids(capsys) -> None:
    assert followup.main(["--config", str(CONFIG), "--plan-only"]) == 0
    schedule = json.loads(capsys.readouterr().out)
    assert schedule["cat_runs"] == 0
    assert schedule["calibration_model_selection_runs"] == 0
    assert schedule["required_new_grid101_fits"] == 36
    assert schedule["required_new_grid121_fits"] == 36
    assert schedule["minimum_new_fits"] == schedule["maximum_new_fits"] == 72
    assert schedule["fit_pair_gates"] == ["81_vs_101", "101_vs_121"]
