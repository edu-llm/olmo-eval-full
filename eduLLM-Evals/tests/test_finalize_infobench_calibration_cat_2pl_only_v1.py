from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import finalize_infobench_calibration_cat_2pl_only_v1 as final  # noqa: E402


def _complete_audit() -> dict[str, object]:
    return {
        "passed": True,
        "require_all_specs_every_inner_fold": True,
        "survivor_selection_allowed": False,
        "expected_inner_folds": list(range(4)),
        "expected_spec_ids": list(final.phase3.EXPECTED_PHASE3_SPEC_IDS),
        "expected_spec_fold_combinations": 8,
        "valid_spec_fold_combinations": 8,
        "failed_checks": [],
        "combination_audits": [
            {"inner_fold": fold, "spec_id": spec_id, "passed": True}
            for fold in range(4)
            for spec_id in final.phase3.EXPECTED_PHASE3_SPEC_IDS
        ],
    }


def _authorization_payloads() -> tuple[dict, dict, dict, dict, dict]:
    config = json.loads(final.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    specs = final.phase3.load_calibration_specs(config)
    by_id = {spec.spec_id: spec for spec in specs}
    primary = final.phase3.load_policies(config)[0]
    selected_ids = [specs[0].spec_id] * 20 + [specs[1].spec_id] * 5
    panels: list[dict[str, object]] = []
    phase4_panels: list[dict[str, object]] = []
    for position, spec_id in enumerate(selected_ids):
        repeat, outer_fold = divmod(position, 5)
        spec = by_id[spec_id]
        panels.append(
            {
                "panel_id": f"repeat_{repeat:02d}_outer_{outer_fold:02d}",
                "repeat": repeat,
                "outer_fold": outer_fold,
                "status": "evaluation_complete",
                "selected_spec_id": spec_id,
                "selected_calibration_specification": spec.canonical,
                "inner_selection": {"complete_inner_evidence_audit": _complete_audit()},
            }
        )
        phase4_panels.append(
            {
                "panel_id": f"repeat_{repeat:02d}_outer_{outer_fold:02d}",
                "repeat": repeat,
                "outer_fold": outer_fold,
                "spec_id": spec_id,
                "calibration_specification": spec.canonical,
            }
        )
    signature = "b" * 64
    selected = {
        "schema_version": final.phase3.SELECTED_SPECS_SCHEMA,
        "study_signature": signature,
        "primary_policy": final.v3.asdict(primary),
        "sensitivity_promotion_allowed": False,
        "panels": panels,
    }
    phase3_decision = {
        "schema_version": final.phase3.DECISION_SCHEMA,
        "status": "pass",
        "study_signature": signature,
        "phase3_pass": True,
        "phase4_authorized": True,
        "failed_conditions": [],
        "inner_selection_fail_closed": {
            "require_all_specs_every_inner_fold": True,
            "expected_spec_fold_combinations_per_panel": 8,
            "survivor_selection_allowed": False,
            "all_panel_selections_locked_before_outer_outcomes": True,
            "all_25_panels_complete": True,
        },
        "coverage": {
            "all_25_panels_evaluated": True,
            "all_52_models_each_repetition": True,
            "by_repetition": {
                str(repeat): {
                    "n_primary_rows": 52,
                    "n_unique_models": 52,
                    "all_52_models_exactly_once": True,
                }
                for repeat in range(5)
            },
            "model_repeat_rows_treated_as_independent": False,
        },
        "calibration_stability": {
            "passed": True,
            "unique_modal_spec_id": specs[0].spec_id,
            "tied_modal_spec_ids": [],
            "modal_count": 20,
            "modal_fraction_all_25_panels": 0.8,
            "required_fraction": 0.8,
        },
        "primary_policy": {
            "policy": final.v3.asdict(primary),
            "all_outer_panels_pass": True,
            "all_repetitions_pass": True,
            "failed_panel_ids": [],
            "failed_repetitions": [],
        },
        "sensitivities": {
            "diagnostic_only": True,
            "can_promote": False,
            "affected_phase3_decision": False,
        },
    }
    phase4_decision = {"panel_selected_specifications": phase4_panels}
    manifest = {
        "schema_version": final.phase3.SCRIPT_SCHEMA,
        "status": "phase3_complete",
        "study_signature": signature,
    }
    return config, selected, phase3_decision, phase4_decision, manifest


def test_finalizer_defaults_are_canonical_and_isolated_from_v3() -> None:
    assert "InFoBench_2pl_only_v1/phase3" in str(final.DEFAULT_PHASE3)
    assert "InFoBench_2pl_only_v1/phase4" in str(final.DEFAULT_PHASE4)
    assert "InFoBench_2pl_only_v1/final_fit" in str(final.DEFAULT_OUTPUT)
    assert all(
        "InFoBench_v3" not in str(path)
        for path in (final.DEFAULT_PHASE3, final.DEFAULT_PHASE4, final.DEFAULT_OUTPUT)
    )
    args = final.build_argparser().parse_args([])
    assert args.phase3_dir == final.DEFAULT_PHASE3
    assert args.phase4_dir == final.DEFAULT_PHASE4
    assert args.out_dir == final.DEFAULT_OUTPUT


def test_versioned_output_validator_accepts_only_exact_2pl_leaf(tmp_path: Path) -> None:
    canonical = tmp_path / "runs" / "calibration" / "InFoBench_2pl_only_v1" / "final_fit"
    configured = "runs/calibration/InFoBench_2pl_only_v1/final_fit"
    assert (
        final._validate_versioned_output(configured, canonical, root=tmp_path, canonical=canonical)
        == canonical
    )

    old_v3 = tmp_path / "runs" / "calibration" / "InFoBench_v3" / "final_fit"
    with pytest.raises(final.v3.FinalFitError, match="canonical 2PL-only"):
        final._validate_versioned_output(configured, old_v3, root=tmp_path, canonical=canonical)


def test_finalizer_bindings_restore_inherited_globals_after_error() -> None:
    before = {name: getattr(final.v3, name) for name in final._BINDINGS}
    with pytest.raises(RuntimeError), final._two_pl_bindings():
        assert final.v3.phase3 is final.phase3
        assert final.v3.phase4 is final.phase4
        assert final.v3.CANONICAL_FINAL_OUTPUT == final.CANONICAL_FINAL_OUTPUT
        assert final.v3.select_authorized_final_spec is final._select_two_pl_spec
        raise RuntimeError("fixture")
    assert {name: getattr(final.v3, name) for name in before} == before


def test_finalizer_selects_only_the_authorized_two_pl_modal_spec() -> None:
    config, selected, decision, phase4_decision, manifest = _authorization_payloads()
    original = copy.deepcopy(decision)
    selected_spec = final._select_two_pl_spec(config, selected, decision, phase4_decision, manifest)
    assert selected_spec.spec_id == "log_shrinkage_2pl_lambda16"
    assert decision == original
    assert decision["inner_selection_fail_closed"]["expected_spec_fold_combinations_per_panel"] == 8


@pytest.mark.parametrize("mutation", ["old_twelve", "one_pl", "unstable_modal"])
def test_finalizer_authorization_fails_closed(mutation: str) -> None:
    config, selected, decision, phase4_decision, manifest = _authorization_payloads()
    if mutation == "old_twelve":
        decision["inner_selection_fail_closed"]["expected_spec_fold_combinations_per_panel"] = 12
    elif mutation == "one_pl":
        selected["panels"][0]["selected_spec_id"] = "1pl_fixed_a1"
    else:
        specs = final.phase3.load_calibration_specs(config)
        for position in range(19, 20):
            panel = selected["panels"][position]
            panel["selected_spec_id"] = specs[1].spec_id
            panel["selected_calibration_specification"] = specs[1].canonical
            phase4_panel = phase4_decision["panel_selected_specifications"][position]
            phase4_panel["spec_id"] = specs[1].spec_id
            phase4_panel["calibration_specification"] = specs[1].canonical
        decision["calibration_stability"].update(
            {
                "modal_count": 19,
                "modal_fraction_all_25_panels": 19 / 25,
            }
        )

    with pytest.raises((final.phase4.TwoPLPhase4Error, final.v3.FinalFitError)):
        final._select_two_pl_spec(config, selected, decision, phase4_decision, manifest)


def test_finalizer_manifest_identifies_versioned_adapter_and_no_v3_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        final.v3.FinalFitRunner,
        "_base_manifest",
        lambda self, status: {"status": status},
    )
    runner = object.__new__(final.FinalFitRunner)
    manifest = runner._base_manifest("fixture")
    assert manifest["script"] == "scripts/finalize_infobench_calibration_cat_2pl_only_v1.py"
    assert manifest["parent_v3_artifacts_read_or_written"] is False
    assert manifest["phase3_candidate_spec_ids"] == list(final.phase3.EXPECTED_PHASE3_SPEC_IDS)
    assert manifest["v4_numerical_spec_ids"] == list(final.phase3.EXPECTED_V4_SPEC_IDS)
