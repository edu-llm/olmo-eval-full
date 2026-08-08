from __future__ import annotations

import copy
import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from scripts import nested_cat_total_uncertainty_v2 as v2_engine
from scripts import nested_cat_total_uncertainty_v3 as phase4


def _authorized_payloads() -> tuple[dict, dict, dict]:
    signature = "v3-phase3-signature"
    manifest = {
        "schema_version": phase4.phase3.SCRIPT_SCHEMA,
        "status": "phase3_complete",
        "study_signature": signature,
    }
    selected = {
        "schema_version": phase4.phase3.SELECTED_SPECS_SCHEMA,
        "study_signature": signature,
    }
    decision = {
        "schema_version": phase4.phase3.DECISION_SCHEMA,
        "status": "pass",
        "phase3_pass": True,
        "phase4_authorized": True,
        "study_signature": signature,
        "failed_conditions": [],
        "coverage": {
            "all_25_panels_evaluated": True,
            "all_52_models_each_repetition": True,
        },
        "calibration_stability": {
            "passed": True,
            "modal_fraction_all_25_panels": 0.8,
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
    return decision, selected, manifest


def test_v3_authorization_accepts_only_passing_v3_handoff() -> None:
    decision, selected, manifest = _authorized_payloads()
    phase4.validate_phase3_authorization(decision, selected, manifest)

    wrong_schema = copy.deepcopy(manifest)
    wrong_schema["schema_version"] = "infobench-nested-scenario-cat-cv-v2-v1"
    with pytest.raises(phase4.V3Phase4Error, match="schema"):
        phase4.validate_phase3_authorization(decision, selected, wrong_schema)

    failed = copy.deepcopy(decision)
    failed["phase4_authorized"] = False
    with pytest.raises(phase4.V3Phase4Error, match="did not pass"):
        phase4.validate_phase3_authorization(failed, selected, manifest)


def test_v3_bindings_do_not_mutate_historical_v2_module() -> None:
    original_phase3 = v2_engine.phase3
    original_schema = v2_engine.SCRIPT_SCHEMA
    original_error = v2_engine.V2Phase4Error
    original_skills = phase4.cm.SKILLS
    original_n_skills = phase4.cm.N_SKILLS
    with phase4._v3_engine_bindings():
        assert v2_engine.phase3 is phase4.phase3
        assert v2_engine.SCRIPT_SCHEMA == phase4.SCRIPT_SCHEMA
        assert v2_engine.V2Phase4Error is phase4.V3Phase4Error
        assert phase4.cm.SKILLS == phase4.INFOBENCH_SKILLS
        assert phase4.cm.N_SKILLS == 5
    assert v2_engine.phase3 is original_phase3
    assert original_schema == v2_engine.SCRIPT_SCHEMA
    assert v2_engine.V2Phase4Error is original_error
    assert original_skills == phase4.cm.SKILLS
    assert original_n_skills == phase4.cm.N_SKILLS


def test_real_v3_constructor_scopes_infobench_axis_from_tutorbench_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phase4.cm.configure_skills(None)
    original_skills = phase4.cm.SKILLS
    config = json.loads(phase4.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    rubrics = phase4.engine._repo_path(config["baseline"]["rubrics"])

    def probe_base_constructor(self, args):
        del args
        assert phase4.cm.SKILLS == phase4.INFOBENCH_SKILLS
        q_by = phase4.cm.load_q_matrix(rubrics)
        assert q_by
        assert {len(row) for row in q_by.values()} == {5}
        self.preflight_axis = phase4.cm.SKILLS

    monkeypatch.setattr(
        phase4.engine.V2Phase4Runner, "__init__", probe_base_constructor
    )
    runner = phase4.V3Phase4Runner(SimpleNamespace(config=phase4.DEFAULT_CONFIG))
    assert runner.preflight_axis == phase4.INFOBENCH_SKILLS
    assert original_skills == phase4.cm.SKILLS


def test_complete_phase3_artifact_inventory_is_hash_and_path_checked(
    tmp_path: Path,
) -> None:
    outputs: dict[str, dict[str, str]] = {}
    for name in phase4.phase3.REQUIRED_OUTPUTS:
        if name == "manifest.json":
            continue
        path = tmp_path / name
        content = b"same-decision\n" if name in {
            "phase3_decision.json",
            "v3_decision.json",
        } else f"artifact:{name}\n".encode()
        path.write_bytes(content)
        outputs[name] = {
            "path": str(path.resolve()),
            "sha256": phase4.engine._sha256(path),
        }
    manifest = {
        "schema_version": phase4.phase3.SCRIPT_SCHEMA,
        "status": "phase3_complete",
        "outputs": outputs,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    inventory = phase4.validate_phase3_artifact_inventory(tmp_path, manifest)
    assert set(inventory) == set(phase4.phase3.REQUIRED_OUTPUTS)

    victim = tmp_path / "outer_oof_per_model.csv"
    victim.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(phase4.V3Phase4Error, match="failed provenance"):
        phase4.validate_phase3_artifact_inventory(tmp_path, manifest)


def test_frozen_v3_design_keeps_primary_bootstrap_order_and_no_fallback() -> None:
    config = json.loads(phase4.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    phase4.validate_frozen_phase4_design(config)

    changed = copy.deepcopy(config)
    changed["uncertainty"]["parameter_bootstrap_replicates"] = 99
    with pytest.raises(phase4.V3Phase4Error, match="bootstrap|uncertainty"):
        phase4.validate_frozen_phase4_design(changed)

    fallback = copy.deepcopy(config)
    fallback["selection_gates"]["allow_fallback_if_primary_fails"] = True
    with pytest.raises(phase4.V3Phase4Error, match="fallback"):
        phase4.validate_frozen_phase4_design(fallback)

    changed_seed = copy.deepcopy(config)
    changed_seed["uncertainty"]["order_seeds"][-1] = 2020
    with pytest.raises(phase4.V3Phase4Error, match="order|seed"):
        phase4.validate_frozen_phase4_design(changed_seed)

    changed_population_bootstrap = copy.deepcopy(config)
    changed_population_bootstrap["runtime"][
        "metric_family_bootstrap_replicates"
    ] = 1999
    with pytest.raises(phase4.V3Phase4Error, match="bootstrap"):
        phase4.validate_frozen_phase4_design(changed_population_bootstrap)


def test_phase3_cross_links_require_embedded_pass_policy_and_selected_path(
    tmp_path: Path,
) -> None:
    selected_path = tmp_path / "selected_calibration_specs.json"
    selected_path.write_text("{}\n", encoding="utf-8")
    policy = {
        "policy_id": phase4.PRIMARY_POLICY_ID,
        "role": "primary",
        "minimum_scenarios": 15,
        "conditional_se_target": 0.2,
        "selector": "trace",
    }
    manifest = {
        "phase3_decision": {
            "status": "pass",
            "phase3_pass": True,
            "phase4_authorized": True,
        }
    }
    selected = {"primary_policy": policy}
    decision = {
        "primary_policy": {"policy": policy},
        "selected_calibration_specs": {
            "path": str(selected_path),
            "sha256": phase4._REAL_SHA256(selected_path),
        },
    }
    phase4.validate_phase3_cross_links(
        manifest=manifest,
        decision=decision,
        selected=selected,
        selected_path=selected_path,
        primary_policy=policy,
    )

    bad_manifest = copy.deepcopy(manifest)
    bad_manifest["phase3_decision"]["phase4_authorized"] = False
    with pytest.raises(phase4.V3Phase4Error, match="passing authorization"):
        phase4.validate_phase3_cross_links(
            manifest=bad_manifest,
            decision=decision,
            selected=selected,
            selected_path=selected_path,
            primary_policy=policy,
        )

    bad_decision = copy.deepcopy(decision)
    bad_decision["primary_policy"]["policy"]["minimum_scenarios"] = 12
    with pytest.raises(phase4.V3Phase4Error, match="primary-policy"):
        phase4.validate_phase3_cross_links(
            manifest=manifest,
            decision=bad_decision,
            selected=selected,
            selected_path=selected_path,
            primary_policy=policy,
        )


def _valid_dense_fit() -> tuple[dict, phase4.phase3.CalibrationSpec]:
    config = json.loads(phase4.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    spec = phase4.phase3.load_calibration_specs(config)[0]
    fit = {
        "items": ["c1", "c2"],
        "A": np.ones((2, 1)),
        "b": np.zeros(2),
        "R": np.eye(1),
        "dim_labels": ["instruction_following"],
        "loglik": -1.0,
        "n_params": 2,
        "n_iter": 2,
        "converged": True,
        "grid_nodes": 401,
        "quadrature_method": phase4.cm.NORMAL_TRAPEZOID_QUADRATURE,
        "quadrature_linear_bound": 8.0,
        "convergence_mode": phase4.cm.RETURNED_ITERATE_CONVERGENCE,
        "penalized_objective": -1.0,
        "calibration_specification": spec.canonical,
        "convergence_diagnostics": {
            "objective_tolerance": 1e-4,
            "parameter_tolerance": 5e-5,
            "required_consecutive_passes": 2,
            "final_consecutive_passes": 2,
            "returned_iterate_matches_last_trace": True,
            "final_exact_recomputation": True,
            "stopped_before_extra_mstep": True,
            "all_objective_changes_monotone_within_tolerance": True,
            "trace": [
                {
                    "mstep_optimizer": {
                        "n_items": 2,
                        "n_converged": 2,
                        "n_failed": 0,
                        "max_abs_gradient": 5e-7,
                    }
                }
            ],
        },
    }
    return fit, spec


def test_dense_fit_contract_rejects_unfinished_or_high_gradient_fit() -> None:
    fit, spec = _valid_dense_fit()
    assert phase4.dense_fit_validity(fit, spec)["fit_valid"] is True

    unfinished = copy.deepcopy(fit)
    unfinished["convergence_diagnostics"]["trace"][0]["mstep_optimizer"][
        "n_converged"
    ] = 1
    assert phase4.dense_fit_validity(unfinished, spec)["fit_valid"] is False

    high_gradient = copy.deepcopy(fit)
    high_gradient["convergence_diagnostics"]["trace"][0]["mstep_optimizer"][
        "max_abs_gradient"
    ] = 2e-6
    validity = phase4.dense_fit_validity(high_gradient, spec)
    assert validity["inner_gradient_within_tolerance"] is False
    assert validity["fit_valid"] is False

    unidentified = copy.deepcopy(fit)
    unidentified["R"] = np.asarray([[0.99]])
    validity = phase4.dense_fit_validity(unidentified, spec)
    assert validity["array_shapes_and_identification_valid"] is False
    assert validity["fit_valid"] is False

    invalid_iteration = copy.deepcopy(fit)
    invalid_iteration["n_iter"] = 0
    validity = phase4.dense_fit_validity(invalid_iteration, spec)
    assert validity["valid_iteration_count"] is False
    assert validity["fit_valid"] is False


def test_captured_hashes_drive_context_and_mutation_is_rejected(tmp_path: Path) -> None:
    frozen = tmp_path / "frozen.json"
    frozen.write_text('{"value": 1}\n', encoding="utf-8")
    captured_hash = phase4._REAL_SHA256(frozen)
    runner = object.__new__(phase4.V3Phase4Runner)
    runner._captured_path_hashes = {str(frozen.resolve()): captured_hash}

    frozen.write_text('{"value": 2}\n', encoding="utf-8")
    with phase4._captured_runner_bindings(runner):
        assert phase4.engine._sha256(frozen) == captured_hash
    with pytest.raises(phase4.V3Phase4Error, match="changed after initialization"):
        phase4.validate_captured_path_hashes(runner._captured_path_hashes)


def test_code_dependency_inventory_covers_fit_and_cat_consumers() -> None:
    relative = {str(path.relative_to(phase4.ROOT)) for path in phase4.CODE_DEPENDENCIES}
    required = {
        "scripts/calibrate_partial.py",
        "tutor_cat/__init__.py",
        "tutor_cat/dataio.py",
        "tutor_cat/schemas.py",
        "tutor_cat/selector.py",
        "tutor_cat/skill_structure.py",
    }
    assert required <= relative


def test_plan_only_remains_read_only(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    runner = object.__new__(phase4.V3Phase4Runner)
    runner.args = SimpleNamespace(plan_only=True, resume=False)
    runner._captured_path_hashes = {}
    runner._revalidate_frozen_provenance = MethodType(lambda self: None, runner)
    runner.plan_payload = MethodType(
        lambda self: {"read_only": True, "output": str(tmp_path / "not-created")},
        runner,
    )
    assert runner.run() == 0
    assert not (tmp_path / "not-created").exists()
    assert json.loads(capsys.readouterr().out)["read_only"] is True


def test_v3_cli_defaults_are_versioned_and_do_not_launch() -> None:
    args = phase4.build_argparser().parse_args([])
    assert args.config == phase4.DEFAULT_CONFIG
    assert args.phase3_dir == phase4.DEFAULT_PHASE3
    assert args.out_dir == phase4.DEFAULT_OUTPUT
    assert args.plan_only is False
    assert args.resume is False
