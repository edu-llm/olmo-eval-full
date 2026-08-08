from __future__ import annotations

import copy
import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

from scripts import finalize_infobench_calibration_cat_v3 as final


def _authorization_payloads() -> tuple[dict, dict, dict, dict, dict]:
    config = json.loads(final.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    specs = final.phase3.load_calibration_specs(config)
    primary = final.phase3.load_policies(config)[0]
    panels: list[dict] = []
    phase4_panels: list[dict] = []
    # Stable unique modal: 20/25 1PL, then 3/25 lambda16 and 2/25 lambda4.
    selected_ids = [specs[0].spec_id] * 20 + [specs[1].spec_id] * 3 + [specs[2].spec_id] * 2
    for position, spec_id in enumerate(selected_ids):
        repeat, outer = divmod(position, 5)
        spec = next(item for item in specs if item.spec_id == spec_id)
        panels.append(
            {
                "panel_id": f"repeat_{repeat:02d}_outer_{outer:02d}",
                "repeat": repeat,
                "outer_fold": outer,
                "status": "evaluation_complete",
                "selected_spec_id": spec_id,
                "selected_calibration_specification": spec.canonical,
                "inner_selection": {
                    "complete_inner_evidence_audit": {"passed": True}
                },
            }
        )
        phase4_panels.append(
            {
                "panel_id": f"repeat_{repeat:02d}_outer_{outer:02d}",
                "repeat": repeat,
                "outer_fold": outer,
                "spec_id": spec_id,
                "calibration_specification": spec.canonical,
            }
        )
    signature = "3" * 64
    manifest = {
        "schema_version": final.phase3.SCRIPT_SCHEMA,
        "status": "phase3_complete",
        "study_signature": signature,
    }
    selected = {
        "schema_version": final.phase3.SELECTED_SPECS_SCHEMA,
        "study_signature": signature,
        "primary_policy": final.asdict(primary),
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
            "expected_spec_fold_combinations_per_panel": 12,
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
            "policy": final.asdict(primary),
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
    return config, selected, phase3_decision, phase4_decision, manifest


def test_final_spec_is_only_the_authorized_stable_exact_modal_choice() -> None:
    config, selected, p3, p4, manifest = _authorization_payloads()
    spec = final.select_authorized_final_spec(config, selected, p3, p4, manifest)
    assert spec.spec_id == "1pl_fixed_a1"

    mismatch = copy.deepcopy(p4)
    mismatch["panel_selected_specifications"][0]["spec_id"] = (
        "log_shrinkage_2pl_lambda16"
    )
    with pytest.raises(final.FinalFitError, match="exact|disagree"):
        final.select_authorized_final_spec(config, selected, p3, mismatch, manifest)


def test_final_spec_fails_closed_on_tie_or_below_eighty_percent() -> None:
    config, selected, p3, p4, manifest = _authorization_payloads()
    specs = final.phase3.load_calibration_specs(config)

    # A 12/12/1 panel is deliberately tied and cannot authorize a fallback.
    for position, panel in enumerate(selected["panels"]):
        spec = specs[0] if position < 12 else specs[1] if position < 24 else specs[2]
        panel["selected_spec_id"] = spec.spec_id
        panel["selected_calibration_specification"] = spec.canonical
        p4["panel_selected_specifications"][position]["spec_id"] = spec.spec_id
        p4["panel_selected_specifications"][position][
            "calibration_specification"
        ] = spec.canonical
    with pytest.raises(final.FinalFitError, match="unique modal"):
        final.select_authorized_final_spec(config, selected, p3, p4, manifest)

    config, selected, p3, p4, manifest = _authorization_payloads()
    p3["calibration_stability"]["modal_fraction_all_25_panels"] = 0.79
    with pytest.raises(final.FinalFitError, match="not stably authorized"):
        final.select_authorized_final_spec(config, selected, p3, p4, manifest)


def test_phase4_inventory_requires_exact_hashes_and_terminal_pass(tmp_path: Path) -> None:
    outputs: dict[str, dict[str, str]] = {}
    for name in final.phase4_engine.FINAL_OUTPUTS:
        path = tmp_path / name
        path.write_text(f"artifact:{name}\n", encoding="utf-8")
        outputs[name] = {"path": name, "sha256": final._sha256(path)}
    manifest = {
        "schema_version": final.phase4.SCRIPT_SCHEMA,
        "status": "phase4_complete_pass",
        "outputs": outputs,
        "checkpoint_tree_sha256": final._tree_hash(tmp_path / "phase4_checkpoints"),
        "bootstrap_cache_tree_sha256": final._tree_hash(tmp_path / "phase4_cache"),
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    inventory = final.validate_phase4_artifact_inventory(tmp_path, manifest)
    assert set(inventory) == {"manifest.json", *final.phase4_engine.FINAL_OUTPUTS}

    (tmp_path / "outer_total_se.csv").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(final.FinalFitError, match="hash"):
        final.validate_phase4_artifact_inventory(tmp_path, manifest)


def test_recorded_upstream_code_inventory_is_byte_verified(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    files = {str(source.resolve()): final._sha256(source)}
    provenance = {
        "files": files,
        "canonical_sha256": final._canonical_hash(files),
    }
    assert final.validate_recorded_code_inventory("fixture", provenance) == files

    source.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(final.FinalFitError, match="changed"):
        final.validate_recorded_code_inventory("fixture", provenance)


def test_phase4_decision_requires_all_five_gate_rows_and_exact_provenance(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.json"
    p3_manifest_path = tmp_path / "p3_manifest.json"
    p3_decision_path = tmp_path / "p3_decision.json"
    selected_path = tmp_path / "selected.json"
    for path, payload in (
        (config_path, {"config": True}),
        (p3_manifest_path, {"study_signature": "p3-signature"}),
        (p3_decision_path, {"phase3_pass": True}),
        (selected_path, {"panels": []}),
    ):
        path.write_text(json.dumps(payload), encoding="utf-8")
    phase3_manifest = {
        "study_signature": "p3-signature",
        "inputs": {
            "response_matrix": {"sha256": "matrix-hash"},
            "rubrics": {"sha256": "rubrics-hash"},
            "scenarios": {"sha256": "scenarios-hash"},
            "judge_manifest": {"sha256": "judge-hash"},
        },
    }
    gates = final.pd.DataFrame(
        [
            {
                "repeat": repeat,
                "complete_total_se_support": True,
                "complete_order_support": True,
                "every_model_valid_draw_gate": True,
                "p90_total_se_gate": True,
                "every_order_seed_present_and_valid": True,
                "order_stability_gate": True,
                "repetition_phase4_pass": True,
                "n_models_expected": 52,
                "n_total_se_rows": 52,
                "n_order_rows": 52,
                "minimum_valid_draw_rate_required": 0.9,
                "minimum_valid_draw_rate_observed": 0.95,
                "maximum_p90_cat_total_se": 0.5,
                "p90_cat_total_se": 0.4,
                "maximum_median_order_path_sd": 0.2,
                "median_order_path_sd": 0.1,
            }
            for repeat in range(5)
        ]
    )
    gate_path = tmp_path / "repetition_gate_results.csv"
    gates.to_csv(gate_path, index=False)
    decision = {
        "schema_version": final.phase4.DECISION_SCHEMA,
        "status": "pass",
        "phase4_pass": True,
        "all_five_repetitions_pass": True,
        "final_fit_authorized": True,
        "sensitivity_policies_considered": False,
        "model_repeat_rows_treated_as_independent": False,
        "failed_repeats": [],
        "phase3_study_signature": "p3-signature",
        "study_signature": "4" * 64,
        "primary_policy": {
            "policy_id": final.PRIMARY_POLICY_ID,
            "role": "primary",
            "minimum_scenarios": 15,
            "conditional_se_target": 0.2,
            "selector": "trace",
        },
        "input_hashes": {
            "response_matrix": "matrix-hash",
            "rubrics": "rubrics-hash",
            "scenarios": "scenarios-hash",
            "judge_manifest": "judge-hash",
        },
        "config_sha256": final._sha256(config_path),
        "phase3_manifest_sha256": final._sha256(p3_manifest_path),
        "selected_calibration_specs_sha256": final._sha256(selected_path),
        "phase3_decision_sha256": final._sha256(p3_decision_path),
        "gate_table": {
            "path": "repetition_gate_results.csv",
            "sha256": final._sha256(gate_path),
        },
    }
    manifest = {
        "study_signature": "4" * 64,
        "decision": copy.deepcopy(decision),
    }
    observed = final.validate_phase4_decision(
        decision,
        manifest,
        phase4_dir=tmp_path,
        phase3_manifest=phase3_manifest,
        config_path=config_path,
        phase3_manifest_path=p3_manifest_path,
        phase3_decision_path=p3_decision_path,
        selected_path=selected_path,
    )
    assert list(observed["repeat"]) == list(range(5))

    failed = copy.deepcopy(decision)
    failed["final_fit_authorized"] = False
    with pytest.raises(final.FinalFitError, match="authorize"):
        final.validate_phase4_decision(
            failed,
            {"decision": failed},
            phase4_dir=tmp_path,
            phase3_manifest=phase3_manifest,
            config_path=config_path,
            phase3_manifest_path=p3_manifest_path,
            phase3_decision_path=p3_decision_path,
            selected_path=selected_path,
        )


def _replay_row(model: str, seed: int, *, valid_score: bool = True) -> dict:
    return {
        "model": model,
        "replay_seed": seed,
        "status": "ok",
        "cat_replay_success": True,
        "baseline_replay_success": True,
        "cat_mwle_converged": valid_score,
        "theta_cat_mwle": 0.1 if valid_score else None,
        "cat_scenarios_administered": 10 + seed,
        "baseline_scenarios_administered": 20 + seed,
    }


def test_seed_rows_are_aggregated_within_tutor_before_family_inference() -> None:
    rows = [
        _replay_row(model, seed)
        for model in ("model-a", "model-b")
        for seed in (1, 2)
    ]
    coverage = final.replay_coverage(
        rows, expected_models=["model-a", "model-b"], expected_seeds=[1, 2]
    )
    assert coverage["attempt_grid_complete"] is True
    assert coverage["n_attempts_observed"] == 4
    assert coverage["all_cat_scores_valid"] is True

    per_model, summary = final.aggregate_paired_lengths(
        rows,
        {"model-a": "family-a", "model-b": "family-b"},
        seed=123,
        bootstrap_replicates=100,
    )
    assert len(per_model) == 2
    assert set(per_model["n_valid_seed_pairs"]) == {2}
    assert summary["seed_rows_are_independent"] is False
    assert summary["bootstrap_unit"] == "tutor_family"
    assert summary["scenario_reduction_vs_random"] == pytest.approx(10 / 21.5)


def test_complete_attempt_grid_is_distinct_from_valid_leaderboard_scores() -> None:
    rows = [_replay_row("model-a", 1, valid_score=False)]
    coverage = final.replay_coverage(
        rows, expected_models=["model-a"], expected_seeds=[1]
    )
    assert coverage["attempt_grid_complete"] is True
    assert coverage["paired_replay_success_rate"] == 1.0
    assert coverage["all_cat_scores_valid"] is False


def test_plan_only_is_read_only(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    runner = object.__new__(final.FinalFitRunner)
    runner.args = SimpleNamespace(plan_only=True)
    runner._reverify_sources = MethodType(lambda self, stage: None, runner)
    runner.plan_payload = MethodType(
        lambda self: {"read_only": True, "output": str(tmp_path / "not-created")},
        runner,
    )
    assert runner.run() == 0
    assert not (tmp_path / "not-created").exists()
    assert json.loads(capsys.readouterr().out)["read_only"] is True


def test_nonempty_output_requires_explicit_exact_resume(tmp_path: Path) -> None:
    (tmp_path / "user-evidence.txt").write_text("preserve me\n", encoding="utf-8")
    runner = object.__new__(final.FinalFitRunner)
    runner.args = SimpleNamespace(resume=False)
    runner.output_dir = tmp_path
    runner.study_signature = "signature"
    runner._already_complete = False
    runner._reverify_sources = MethodType(lambda self, stage: None, runner)
    runner._revalidate_output_leaf = MethodType(lambda self: None, runner)
    with pytest.raises(final.FinalFitError, match="not empty"):
        runner._prepare_output()
    assert (tmp_path / "user-evidence.txt").read_text(encoding="utf-8") == "preserve me\n"


def test_cli_defaults_do_not_launch() -> None:
    args = final.build_argparser().parse_args([])
    assert args.config == final.DEFAULT_CONFIG
    assert args.phase3_dir == final.DEFAULT_PHASE3
    assert args.phase4_dir == final.DEFAULT_PHASE4
    assert args.out_dir == final.DEFAULT_OUTPUT
    assert args.plan_only is False
    assert args.resume is False
