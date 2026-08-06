from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import nested_cat_total_uncertainty_2pl_only_v1 as phase4  # noqa: E402


def _complete_audit() -> dict[str, object]:
    combinations = [
        {"inner_fold": fold, "spec_id": spec_id, "passed": True}
        for fold in range(4)
        for spec_id in phase4.phase3.EXPECTED_PHASE3_SPEC_IDS
    ]
    return {
        "passed": True,
        "require_all_specs_every_inner_fold": True,
        "survivor_selection_allowed": False,
        "expected_inner_folds": list(range(4)),
        "expected_spec_ids": list(phase4.phase3.EXPECTED_PHASE3_SPEC_IDS),
        "expected_spec_fold_combinations": 8,
        "valid_spec_fold_combinations": 8,
        "failed_checks": [],
        "combination_audits": combinations,
    }


def _authorized_payloads() -> tuple[dict, dict, dict]:
    signature = "a" * 64
    selected = {
        "schema_version": phase4.phase3.SELECTED_SPECS_SCHEMA,
        "study_signature": signature,
        "panels": [
            {
                "repeat": repeat,
                "outer_fold": outer_fold,
                "selected_spec_id": phase4.phase3.EXPECTED_PHASE3_SPEC_IDS[0],
                "inner_selection": {"complete_inner_evidence_audit": _complete_audit()},
            }
            for repeat in range(5)
            for outer_fold in range(5)
        ],
    }
    decision = {
        "schema_version": phase4.phase3.DECISION_SCHEMA,
        "status": "pass",
        "phase3_pass": True,
        "phase4_authorized": True,
        "study_signature": signature,
        "failed_conditions": [],
        "inner_selection_fail_closed": {
            "expected_spec_fold_combinations_per_panel": 8,
        },
        "coverage": {
            "all_25_panels_evaluated": True,
            "all_52_models_each_repetition": True,
        },
        "calibration_stability": {
            "passed": True,
            "modal_fraction_all_25_panels": 1.0,
            "required_fraction": 0.8,
        },
        "primary_policy": {
            "all_outer_panels_pass": True,
            "all_repetitions_pass": True,
            "failed_panel_ids": [],
            "failed_repetitions": [],
        },
        "sensitivities": {"diagnostic_only": True, "can_promote": False},
    }
    manifest = {
        "schema_version": phase4.phase3.SCRIPT_SCHEMA,
        "status": "phase3_complete",
        "study_signature": signature,
    }
    return decision, selected, manifest


def test_phase4_defaults_are_isolated_from_completed_v3_leaves() -> None:
    assert "InFoBench_2pl_only_v1/phase3" in str(phase4.DEFAULT_PHASE3)
    assert "InFoBench_2pl_only_v1/phase4" in str(phase4.DEFAULT_OUTPUT)
    assert "InFoBench_v3" not in str(phase4.DEFAULT_PHASE3)
    assert "InFoBench_v3" not in str(phase4.DEFAULT_OUTPUT)
    args = phase4.build_argparser().parse_args([])
    assert args.phase3_dir == phase4.DEFAULT_PHASE3
    assert args.out_dir == phase4.DEFAULT_OUTPUT


def test_phase4_bindings_restore_every_inherited_global_after_error() -> None:
    before = {name: getattr(phase4.v3, name) for name in phase4._V3_BINDING_NAMES}
    with pytest.raises(RuntimeError), phase4._two_pl_bindings():
        assert phase4.v3.phase3 is phase4.phase3
        assert phase4.v3.DEFAULT_OUTPUT == phase4.DEFAULT_OUTPUT
        assert phase4.v3.SCRIPT_SCHEMA == phase4.SCRIPT_SCHEMA
        raise RuntimeError("fixture")
    assert {name: getattr(phase4.v3, name) for name in before} == before


def test_phase4_authorization_accepts_exact_25_by_8_handoff() -> None:
    decision, selected, manifest = _authorized_payloads()
    phase4.validate_phase3_authorization(decision, selected, manifest)


@pytest.mark.parametrize(
    "mutation",
    ["old_twelve", "duplicate_panel", "wrong_spec", "missing_block", "failed_block"],
)
def test_phase4_inner_contract_rejects_incomplete_or_old_rosters(mutation: str) -> None:
    decision, selected, _manifest = _authorized_payloads()
    first = selected["panels"][0]
    audit = first["inner_selection"]["complete_inner_evidence_audit"]
    if mutation == "old_twelve":
        decision["inner_selection_fail_closed"]["expected_spec_fold_combinations_per_panel"] = 12
    elif mutation == "duplicate_panel":
        selected["panels"][1]["repeat"] = 0
        selected["panels"][1]["outer_fold"] = 0
    elif mutation == "wrong_spec":
        first["selected_spec_id"] = "1pl_fixed_a1"
    elif mutation == "missing_block":
        audit["combination_audits"].pop()
    else:
        audit["combination_audits"][-1]["passed"] = False

    with pytest.raises(phase4.TwoPLPhase4Error):
        phase4.validate_two_pl_inner_contract(decision, selected)


def test_phase4_authorization_fails_closed_before_downstream_work() -> None:
    decision, selected, manifest = _authorized_payloads()
    decision["phase4_authorized"] = False
    with pytest.raises(phase4.TwoPLPhase4Error, match="did not pass"):
        phase4.validate_phase3_authorization(decision, selected, manifest)


def test_phase4_design_accepts_only_versioned_output_namespace() -> None:
    config = json.loads(phase4.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    phase4.validate_frozen_phase4_design(config)

    changed = copy.deepcopy(config)
    changed["outputs"]["phase4"] = "runs/calibration/InFoBench_v3/phase4"
    with pytest.raises(phase4.TwoPLPhase4Error, match="output (path|contract)|parent|differs"):
        phase4.validate_frozen_phase4_design(changed)


def test_phase4_manifest_identifies_adapter_and_denies_parent_v3_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        phase4.v3.V3Phase4Runner,
        "_base_manifest",
        lambda self, status: {"status": status},
    )
    runner = object.__new__(phase4.TwoPLPhase4Runner)
    manifest = runner._base_manifest("fixture")
    assert manifest["script"] == "scripts/nested_cat_total_uncertainty_2pl_only_v1.py"
    assert manifest["parent_v3_artifacts_read_or_written"] is False
    assert manifest["phase3_candidate_spec_ids"] == list(phase4.phase3.EXPECTED_PHASE3_SPEC_IDS)
    assert manifest["v4_numerical_spec_ids"] == list(phase4.phase3.EXPECTED_V4_SPEC_IDS)
