"""Synthetic blocking tests for the InFoBench v2 numerical preflight.

These tests do not fit or inspect the real InFoBench response matrix.  They
exercise the pass, fail, fixed-a, and explicit no-selection branches that guard
the official study.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_infobench_v2_numerical_checks.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("infobench_v2_numerical_checks", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


num = _load_module()


def _parameters(a, b, exportable=None):
    if exportable is None:
        exportable = [True] * len(a)
    return pd.DataFrame(
        {
            "criterion_id": [f"c{index}" for index in range(len(a))],
            "a_instruction_following": a,
            "b": b,
            "exportable": exportable,
        }
    )


def test_fixed_a_1pl_passes_without_inventing_a_spearman() -> None:
    lower = _parameters([1.0] * 5, [-2.0, -1.0, 0.0, 1.0, 2.0])
    upper = _parameters([1.0] * 5, [-2.01, -0.99, 0.01, 1.01, 2.01])
    result = num.parameter_scope_gate(
        lower,
        upper,
        family="1pl",
        source_ids=list(lower["criterion_id"]),
        minimum_spearman=0.99,
        minimum_exportability_agreement=0.995,
        lower_converged=True,
        upper_converged=True,
    )
    assert result["passed"] is True
    assert result["a_spearman"] is None
    assert result["fixed_a_exact"] is True
    assert result["a_stability_method"].startswith("fixed-a-exact")


def test_fixed_a_or_free_parameter_instability_blocks_scope() -> None:
    fixed_lower = _parameters([1.0] * 5, [-2, -1, 0, 1, 2])
    fixed_upper = _parameters([1.0, 1.0, 0.999, 1.0, 1.0], [-2, -1, 0, 1, 2])
    fixed_result = num.parameter_scope_gate(
        fixed_lower,
        fixed_upper,
        family="1pl",
        source_ids=list(fixed_lower["criterion_id"]),
        minimum_spearman=0.99,
        minimum_exportability_agreement=0.995,
        lower_converged=True,
        upper_converged=True,
    )
    assert fixed_result["passed"] is False
    assert fixed_result["fixed_a_exact"] is False

    free_lower = _parameters([0.5, 0.8, 1.0, 1.4, 2.0], [-2, -1, 0, 1, 2])
    free_upper = _parameters([2.0, 1.4, 1.0, 0.8, 0.5], [-2, -1, 0, 1, 2])
    free_result = num.parameter_scope_gate(
        free_lower,
        free_upper,
        family="free-2pl",
        source_ids=list(free_lower["criterion_id"]),
        minimum_spearman=0.99,
        minimum_exportability_agreement=0.995,
        lower_converged=True,
        upper_converged=True,
    )
    assert free_result["passed"] is False
    assert np.isclose(free_result["a_spearman"], -1.0)


def _cell_panel(probability_shift: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for fold, (model, family, label, probability) in enumerate(
        [
            ("m1", "f1", 0, 0.10),
            ("m2", "f2", 1, 0.90),
            ("m3", "f3", 0, 0.20),
            ("m4", "f4", 1, 0.80),
        ]
    ):
        rows.append(
            {
                "outer_fold": fold,
                "model": model,
                "family": family,
                "criterion_id": f"c{fold}",
                "label": label,
                "probability": probability,
            }
        )
    lower = pd.DataFrame(rows)
    upper = lower.copy()
    upper["probability"] = np.clip(
        upper["probability"].to_numpy(float) + probability_shift, 0.001, 0.999
    )
    return lower, upper


def test_common_cell_family_bootstrap_pass_and_fail() -> None:
    lower, identical = _cell_panel(0.0)
    passed, details = num.common_cell_equivalence(
        lower,
        identical,
        log_loss_margin=0.005,
        brier_margin=0.005,
        replicates=100,
        seed=123,
    )
    assert passed["passed"] is True
    assert passed["keys_identical"] is True
    assert len(details) == 4

    lower, shifted = _cell_panel(0.30)
    failed, _ = num.common_cell_equivalence(
        lower,
        shifted,
        log_loss_margin=0.005,
        brier_margin=0.005,
        replicates=100,
        seed=123,
    )
    assert failed["passed"] is False
    assert any(not value["passed"] for value in failed["metrics"].values())


def _eap_rows(shift: float, tail: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": ["m1", "m2", "m3"],
            "scope": ["full", "outer_0", "outer_1"],
            "support": ["full_bank", "administration_only", "full_bank"],
            "theta_401": [-1.0, 0.0, 1.0],
            "theta_801": [-1.0 + shift, shift, 1.0 + shift],
            "tail_mass_401": [tail] * 3,
            "tail_mass_801": [tail] * 3,
        }
    )


THRESHOLDS = {
    "maximum_median_absolute_theta_shift": 0.02,
    "maximum_p95_absolute_theta_shift": 0.05,
    "maximum_absolute_theta_shift": 0.005,
    "maximum_posterior_tail_mass": 0.00001,
}


def test_eap_shift_and_tail_mass_are_blocking() -> None:
    assert num.eap_rows_gate(_eap_rows(0.001, 1e-8), THRESHOLDS)["passed"] is True
    assert num.eap_rows_gate(_eap_rows(0.006, 1e-8), THRESHOLDS)["passed"] is False
    assert num.eap_rows_gate(_eap_rows(0.001, 2e-5), THRESHOLDS)["passed"] is False


def test_all_spec_pass_emits_lock_without_selecting_a_spec(tmp_path: Path) -> None:
    spec_ids = [f"spec_{index}" for index in range(6)]
    fits = {spec_id: {"passed": True} for spec_id in spec_ids}
    eaps = {spec_id: {"passed": True} for spec_id in spec_ids}
    decision = num.build_numerical_decision(spec_ids, fits, eaps)
    assert decision["passed"] is True
    assert decision["selection_performed"] is False
    assert decision["calibration_specification_selected"] is None

    lock = num.emit_numerical_lock(
        tmp_path,
        decision,
        {"locked_fit_grid": 61, "locked_eap_grid": 401},
    )
    assert lock == tmp_path / "numerical_lock.json"
    payload = json.loads(lock.read_text(encoding="utf-8"))
    assert payload["selection_performed"] is False
    assert payload["calibration_specification_selected"] is None
    assert (tmp_path / "numerical_lock.sha256").is_file()


def test_one_failed_or_missing_spec_emits_no_lock(tmp_path: Path) -> None:
    spec_ids = ["one", "two"]
    fits = {"one": {"passed": True}, "two": {"passed": False}}
    eaps = {"one": {"passed": True}, "two": {"passed": True}}
    failed = num.build_numerical_decision(spec_ids, fits, eaps)
    assert failed["passed"] is False
    assert failed["failure_action"] == "block_phase3_and_investigate"
    assert num.emit_numerical_lock(tmp_path, failed, {}) is None
    assert not (tmp_path / "numerical_lock.json").exists()

    incomplete = num.build_numerical_decision(
        spec_ids, {"one": {"passed": True}}, {"one": {"passed": True}}
    )
    assert incomplete["complete_exact_specification_panel"] is False
    assert incomplete["passed"] is False
    assert incomplete["selection_performed"] is False


def test_legacy_parent_lock_cannot_authorize_phase3(tmp_path: Path) -> None:
    from scripts import nested_scenario_cat_cv_v2 as phase3

    spec_ids = ["one", "two"]
    decision = num.build_numerical_decision(
        spec_ids,
        {spec_id: {"passed": True} for spec_id in spec_ids},
        {spec_id: {"passed": True} for spec_id in spec_ids},
    )
    lock = num.emit_numerical_lock(tmp_path, decision, {"locked_fit_grid": 61})
    assert lock is not None
    config = json.loads(
        (ROOT / "configs" / "infobench_calibration_cat_v2.json").read_text(
            encoding="utf-8"
        )
    )
    with pytest.raises(phase3.V2Phase3Error, match="output/lock path|follow-up"):
        phase3._validate_numerical_lock(
            config,
            phase3.load_calibration_specs(config),
            lock_path=lock,
            followup_config_path=(
                ROOT / "configs" / "infobench_v2_numerical_followup_v3.json"
            ),
        )
