from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import render_infobench_2pl_only_v1_terminal as report  # noqa: E402


def _audit(*, complete: bool = True) -> dict[str, object]:
    combinations = [
        {"inner_fold": fold, "spec_id": spec_id, "passed": True}
        for fold in range(4)
        for spec_id in report.PHASE3_SPEC_IDS
    ]
    failures: list[str] = []
    if not complete:
        combinations[-1]["passed"] = False
        failures = ["fixture_failure"]
    valid = sum(row["passed"] is True for row in combinations)
    return {
        "passed": complete,
        "require_all_specs_every_inner_fold": True,
        "survivor_selection_allowed": False,
        "expected_inner_folds": list(range(4)),
        "expected_spec_ids": list(report.PHASE3_SPEC_IDS),
        "expected_spec_fold_combinations": 8,
        "valid_spec_fold_combinations": valid,
        "failed_checks": failures,
        "combination_audits": combinations,
    }


def _panels(*, incomplete_key: tuple[int, int] | None = None) -> dict[str, object]:
    return {
        "panels": [
            {
                "repeat": repeat,
                "outer_fold": fold,
                "selected_spec_id": report.PHASE3_SPEC_IDS[0],
                "inner_selection": {
                    "complete_inner_evidence_audit": _audit(
                        complete=(repeat, fold) != incomplete_key
                    )
                },
            }
            for repeat in range(5)
            for fold in range(5)
        ]
    }


def test_versioned_paths_and_spec_rosters_are_distinct() -> None:
    assert report.DEFAULT_CONFIG.name == "infobench_calibration_cat_2pl_only_v1.json"
    assert str(report.DEFAULT_PHASE3_DIR).endswith("InFoBench_2pl_only_v1/phase3")
    assert str(report.DEFAULT_OUTPUT_DIR).endswith("infobench_calibration_cat_2pl_only_v1")
    assert report.PHASE3_SPEC_IDS == (
        "log_shrinkage_2pl_lambda16",
        "log_shrinkage_2pl_lambda4",
    )
    assert ("1pl_fixed_a1", *report.PHASE3_SPEC_IDS) == report.V4_SPEC_IDS
    assert report.EXPECTED_INNER_SUMMARY_ROWS == 50


def test_phase3_decision_name_is_canonical_and_v3_name_is_only_a_mirror() -> None:
    assert report.CANONICAL_PHASE3_DECISION_NAME == "phase3_decision.json"
    assert report.COMPATIBILITY_PHASE3_DECISION_MIRROR_NAME == "v3_decision.json"


def test_panel_audit_accepts_exact_25_by_8_contract() -> None:
    assert report._validate_panel_audits(_panels(), require_complete=True) == {
        "selection_panels_expected": 25,
        "selection_panels_complete": 25,
        "spec_fold_blocks_expected": 200,
        "spec_fold_blocks_valid": 200,
    }


@pytest.mark.parametrize("mutation", ["old_twelve", "wrong_roster", "lying_total"])
def test_panel_audit_rejects_noncanonical_contract(mutation: str) -> None:
    payload = _panels()
    audit = payload["panels"][0]["inner_selection"]["complete_inner_evidence_audit"]
    if mutation == "old_twelve":
        audit["expected_spec_fold_combinations"] = 12
    elif mutation == "wrong_roster":
        audit["expected_spec_ids"] = [*report.PHASE3_SPEC_IDS, "1pl_fixed_a1"]
    else:
        audit["valid_spec_fold_combinations"] = 7

    with pytest.raises(report.TerminalReportError):
        report._validate_panel_audits(payload, require_complete=True)


def test_panel_audit_reports_partial_evidence_but_terminal_mode_rejects_it() -> None:
    payload = _panels(incomplete_key=(4, 4))
    assert report._validate_panel_audits(payload, require_complete=False) == {
        "selection_panels_expected": 25,
        "selection_panels_complete": 24,
        "spec_fold_blocks_expected": 200,
        "spec_fold_blocks_valid": 199,
    }
    with pytest.raises(report.TerminalReportError, match="incomplete inner evidence"):
        report._validate_panel_audits(payload, require_complete=True)


def test_panel_mirrors_must_preserve_selection_and_inner_evidence() -> None:
    selected = _panels()
    choices = copy.deepcopy(selected)
    selection_lock = copy.deepcopy(selected)
    selection_lock["panels"][7]["selected_spec_id"] = report.PHASE3_SPEC_IDS[1]

    with pytest.raises(report.TerminalReportError, match="differs across panel mirrors"):
        report._validate_panel_mirrors(selected, choices, selection_lock, require_complete=True)


def test_phase3_plot_bindings_restore_inherited_globals_after_error() -> None:
    names = (
        "SPEC_IDS",
        "SPEC_LABELS",
        "V3_CONFIG_SCHEMA",
        "V3_RUN_SCHEMA",
        "V3_DECISION_SCHEMA",
        "V3_SELECTED_SPECS_SCHEMA",
        "V3_CODE_DEPENDENCIES",
        "EXPECTED_INNER_ROWS",
    )
    before = {name: getattr(report.base, name) for name in names}
    with pytest.raises(RuntimeError), report._phase3_plot_bindings():
        assert report.base.SPEC_IDS == report.PHASE3_SPEC_IDS
        assert report.base.EXPECTED_INNER_ROWS == 50
        raise RuntimeError("fixture")
    assert {name: getattr(report.base, name) for name in names} == before


def test_reproduction_commands_authorize_phase4_only_after_phase3_pass() -> None:
    blocked_v4 = report._commands(
        SimpleNamespace(terminal_state="blocked_after_v4_numerical_followup")
    )
    assert blocked_v4.count("run_infobench_v2_numerical_followup_v4.py") == 2
    assert "--resume" in blocked_v4
    assert "nested_scenario_cat_cv_2pl_only_v1.py" not in blocked_v4
    assert "nested_cat_total_uncertainty_2pl_only_v1.py" not in blocked_v4
    assert "render_infobench_2pl_only_v1_terminal.py" in blocked_v4

    blocked_phase3 = report._commands(
        SimpleNamespace(terminal_state="blocked_after_phase3_validation")
    )
    assert "nested_scenario_cat_cv_2pl_only_v1.py" in blocked_phase3
    assert "nested_cat_total_uncertainty_2pl_only_v1.py" not in blocked_phase3
    assert "InFoBench_v3/phase3" not in blocked_phase3

    authorized = report._commands(SimpleNamespace(terminal_state="phase3_pass_phase4_pending"))
    assert "nested_cat_total_uncertainty_2pl_only_v1.py" in authorized
    assert "InFoBench_2pl_only_v1/phase4" in authorized
    assert "InFoBench_v3/phase4" not in authorized
    assert "render_infobench_2pl_only_v1_terminal.py" in authorized


def test_canonical_path_pin_rejects_aliases_and_symlink_components(tmp_path: Path) -> None:
    canonical = tmp_path / "reports" / "infobench_calibration_cat_2pl_only_v1"
    assert (
        report._pin_canonical_path(
            canonical,
            canonical,
            label="fixture output",
            root=tmp_path,
        )
        == canonical
    )
    with pytest.raises(report.TerminalReportError, match="canonical"):
        report._pin_canonical_path(
            tmp_path / "reports" / "alternate",
            canonical,
            label="fixture output",
            root=tmp_path,
        )

    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "reports").symlink_to(outside, target_is_directory=True)
    with pytest.raises(report.TerminalReportError, match="alias|symlink"):
        report._pin_canonical_path(
            canonical,
            canonical,
            label="fixture output",
            root=tmp_path,
        )


@pytest.mark.parametrize("override", ["config", "v4", "phase3", "output"])
def test_renderer_rejects_noncanonical_cli_leaves_before_loading_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    override: str,
) -> None:
    called = False

    def unexpected_load(**_kwargs: object) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(report, "load_terminal_inputs", unexpected_load)
    values = {
        "config_path": report.DEFAULT_CONFIG,
        "v4_dir": report.DEFAULT_V4_DIR,
        "phase3_dir": report.DEFAULT_PHASE3_DIR,
        "output_dir": report.DEFAULT_OUTPUT_DIR,
    }
    key = {
        "config": "config_path",
        "v4": "v4_dir",
        "phase3": "phase3_dir",
        "output": "output_dir",
    }[override]
    values[key] = tmp_path / override
    with pytest.raises(report.TerminalReportError, match="canonical"):
        report.render_terminal_report(**values)
    assert called is False


def test_gate_status_keeps_v4_roster_separate_from_phase3_candidates() -> None:
    inputs = SimpleNamespace(
        terminal_state="phase3_pass_phase4_pending",
        v4_lock={"status": "passed"},
        phase3_decision={"phase3_pass": True},
    )
    status = report._gate_status(inputs)
    assert status["gates"]["v4_numerical_verification"]["spec_ids"] == list(report.V4_SPEC_IDS)
    assert status["gates"]["phase3_repeated_nested_cat"]["candidate_spec_ids"] == list(
        report.PHASE3_SPEC_IDS
    )
    assert status["gates"]["phase3_repeated_nested_cat"]["inner_spec_fold_blocks_expected"] == 200


def test_two_spec_selection_plot_renders_without_three_tick_mismatch(tmp_path: Path) -> None:
    rows = [
        {
            "spec_id": spec_id,
            "mean_log_loss": 0.35 + 0.01 * index,
        }
        for index, spec_id in enumerate(report.PHASE3_SPEC_IDS)
        for _panel in range(25)
    ]
    inputs = SimpleNamespace(
        terminal_state="phase3_pass_phase4_pending",
        phase3={"inner": pd.DataFrame(rows), "selected": _panels()},
    )
    destination = tmp_path / "selection.png"
    report._plot_calibration_selection_2pl(inputs, destination)
    assert destination.is_file()
    assert destination.stat().st_size > 2_000
