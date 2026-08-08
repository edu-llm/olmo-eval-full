from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import calibrate_mirt as cm  # noqa: E402
from scripts import nested_scenario_cat_cv_v3 as v3  # noqa: E402


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _synthetic_v4_lock(tmp_path: Path) -> tuple[Path, Path, dict]:
    base = _load(ROOT / "configs" / "infobench_calibration_cat_v3.json")
    specs = v3.load_calibration_specs(base)
    ids = [spec.spec_id for spec in specs]
    scopes = ["full", "outer_0", "outer_1", "outer_2", "outer_3", "outer_4"]
    output = tmp_path / "InFoBench_v2_numerical_followup_v4"
    config_path = tmp_path / "infobench_v2_numerical_followup_v4.json"
    lock_path = output / "numerical_followup_lock.json"
    contract = {
        "eligible_specifications": [{"spec_id": spec_id} for spec_id in ids],
        "fit_integration": {
            "method": cm.NORMAL_TRAPEZOID_QUADRATURE,
            "linear_bound": 8.0,
            "fit_nodes": [401, 801],
        },
        "convergence": {
            "mode": cm.RETURNED_ITERATE_CONVERGENCE,
            "fit_max_iter": 1500,
            "objective_tolerance": 1e-4,
            "parameter_tolerance": 5e-5,
            "consecutive_convergence_passes": 2,
            "maximum_inner_gradient": 1e-6,
        },
        "fixed_bank_scoring_gate": {
            "profiles": {
                "bound8_801": [cm.NORMAL_TRAPEZOID_QUADRATURE, 801, 8.0, 7.5],
                "bound8_1601": [cm.NORMAL_TRAPEZOID_QUADRATURE, 1601, 8.0, 7.5],
                "bound10_1001": [cm.NORMAL_TRAPEZOID_QUADRATURE, 1001, 10.0, 9.5],
            }
        },
    }
    dependency_path = ROOT / "scripts" / "calibrate_mirt.py"
    dependencies = {
        "scripts/calibrate_mirt.py": v3._sha256(dependency_path),
    }
    history = {
        "v1_aborted": {
            "output_file_count": 1,
            "output_tree_sha256": "1" * 64,
        },
        "v2_terminal_nonpromotable": {
            "output_file_count": 2,
            "output_tree_sha256": "2" * 64,
        },
        "v3_terminal_blocked": {
            "output_file_count": 3,
            "output_tree_sha256": "3" * 64,
        },
    }
    frozen_inputs = {
        "response_matrix": base["baseline"]["response_matrix"],
        "response_matrix_sha256": base["baseline"]["response_matrix_sha256"],
        "rubrics": base["baseline"]["rubrics"],
        "rubrics_sha256": base["baseline"]["rubrics_sha256"],
        "scenarios": base["baseline"]["scenarios"],
        "scenarios_sha256": base["baseline"]["scenarios_sha256"],
        "split_manifest": base["cross_validation"]["split_manifest"],
        "split_manifest_sha256": base["cross_validation"]["split_manifest_sha256"],
    }
    followup = {
        "schema_version": v3.FOLLOWUP_CONFIG_SCHEMA,
        "status": "preregistered_before_v4_results",
        "output_dir": str(output),
        "lock_path": str(lock_path),
        "frozen_contract": contract,
        "frozen_inputs": frozen_inputs,
        "code_freeze": {"dependencies": dependencies},
        "historical_evidence": history,
    }
    _write_json(config_path, followup)

    fit_keys = {
        f"fit/{spec_id}/{scope}/{nodes}/{start}"
        for spec_id in ids
        for scope in scopes
        for nodes, start in (
            (401, "cold"),
            (801, "cold"),
            (801, "continuation"),
        )
    }
    start_keys = {f"start/{spec_id}/{scope}" for spec_id in ids for scope in scopes}
    evidence_keys = {
        *fit_keys,
        *start_keys,
        *(f"refit/{spec_id}" for spec_id in ids),
        *(f"fixed_bank/{spec_id}" for spec_id in ids),
    }
    evidence_hashes: dict[str, str] = {}
    evidence_paths: dict[str, str] = {}
    for index, key in enumerate(sorted(evidence_keys)):
        evidence_path = output / "evidence" / f"{index:03d}.json"
        _write_json(
            evidence_path,
            {"schema_version": v3.FOLLOWUP_RUN_SCHEMA, "key": key, "passed": True},
        )
        evidence_hashes[key] = v3._sha256(evidence_path)
        evidence_paths[key] = f"evidence/{index:03d}.json"

    environment_payload = {"python": "synthetic-test"}
    environment = {
        **environment_payload,
        "canonical_sha256": v3._canonical_hash(environment_payload),
    }
    expected_inputs = {
        v3._display_path(v3._resolve_repo_path(frozen_inputs[name])): frozen_inputs[
            f"{name}_sha256"
        ]
        for name in ("response_matrix", "rubrics", "scenarios", "split_manifest")
    }
    expected_history = {
        key: {
            "output_file_count": history[key]["output_file_count"],
            "output_tree_sha256": history[key]["output_tree_sha256"],
            "promotable": False,
        }
        for key in history
    }
    signature_payload = {
        "schema_version": v3.FOLLOWUP_RUN_SCHEMA,
        "config_sha256": v3._sha256(config_path),
        "frozen_contract_sha256": v3._canonical_hash(contract),
        "input_sha256": expected_inputs,
        "code_sha256": dependencies,
        "historical_evidence_sha256": expected_history,
        "environment_sha256": environment["canonical_sha256"],
        "spec_ids": ids,
        "scope_ids": scopes,
        "fit_nodes": [401, 801],
        "required_total_new_fits": 54,
        "historical_fit_or_checkpoint_artifacts_reused": 0,
        "adaptive_runs": 0,
    }
    signature_hash = v3._canonical_hash(signature_payload)
    signature = {
        **signature_payload,
        "environment": environment,
        "canonical_sha256": signature_hash,
    }
    calibration_profile = [cm.NORMAL_TRAPEZOID_QUADRATURE, 401, 8.0]
    eap_profile = [cm.NORMAL_TRAPEZOID_QUADRATURE, 801, 8.0]
    common = {
        "fit_convergence_mode": cm.RETURNED_ITERATE_CONVERGENCE,
        "fit_max_iter": 1500,
        "fit_objective_tolerance": 1e-4,
        "fit_parameter_tolerance": 5e-5,
        "fit_consecutive_convergence_passes": 2,
    }
    decision_path = output / "numerical_followup_decision.json"
    decision = {
        "schema_version": v3.FOLLOWUP_RUN_SCHEMA,
        "status": "complete_pass",
        "passed": True,
        "eligible_spec_ids": ids,
        "scope_ids": scopes,
        "all_54_fits_valid": True,
        "all_18_grid801_start_gates_passed": True,
        "all_three_refit_bank_gates_passed": True,
        "all_three_fixed_bank_scoring_and_tail_gates_passed": True,
        "fit_validity": {key: True for key in fit_keys},
        "start_validity": {key: True for key in start_keys},
        "refit_gate_passed": {spec_id: True for spec_id in ids},
        "fixed_bank_gate_passed": {spec_id: True for spec_id in ids},
        "effective_numerical_profile": {
            "calibration_integration": calibration_profile,
            "reference_eap_scoring": eap_profile,
        },
        **common,
        "maximum_inner_gradient": 1e-6,
        "config_sha256": v3._sha256(config_path),
        "frozen_contract_sha256": v3._canonical_hash(contract),
        "study_signature_sha256": signature_hash,
        "code_sha256_reverified_before_finalization": dependencies,
        "evidence_paths": evidence_paths,
        "evidence_sha256": evidence_hashes,
        "historical_fit_or_checkpoint_artifacts_reused": 0,
        "calibration_specification_selected": None,
        "calibration_model_selection_performed": False,
        "adaptive_results_inspected": False,
        "adaptive_runs": 0,
    }
    _write_json(decision_path, decision)

    lock = {
        "schema_version": v3.FOLLOWUP_LOCK_SCHEMA,
        "status": "complete_pass",
        "passed": True,
        "eligible_spec_ids": ids,
        "scope_ids": scopes,
        "calibration_integration": calibration_profile,
        "reference_eap_scoring": eap_profile,
        **common,
        "maximum_inner_gradient": 1e-6,
        "config_sha256": v3._sha256(config_path),
        "frozen_contract_sha256": v3._canonical_hash(contract),
        "study_signature_sha256": signature_hash,
        "followup_decision_sha256": v3._sha256(decision_path),
        "evidence_paths": evidence_paths,
        "evidence_sha256": evidence_hashes,
        "historical_fit_or_checkpoint_artifacts_reused": 0,
        "calibration_specification_selected": None,
        "calibration_model_selection_performed": False,
        "adaptive_results_inspected": False,
        "adaptive_runs": 0,
    }
    _write_json(lock_path, lock)
    lock_path.with_suffix(".sha256").write_text(
        f"{v3._sha256(lock_path)}  {lock_path.name}\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": v3.FOLLOWUP_RUN_SCHEMA,
        "status": "complete_pass",
        "config_sha256": v3._sha256(config_path),
        "study_signature_sha256": signature_hash,
        "study_signature": signature,
        "decision_sha256": v3._sha256(decision_path),
        "lock_sha256": v3._sha256(lock_path),
        "historical_fit_or_checkpoint_artifacts_reused": 0,
        "calibration_model_selection_performed": False,
        "adaptive_results_inspected": False,
        "adaptive_runs": 0,
    }
    _write_json(output / "study_manifest.json", manifest)
    return config_path, lock_path, base


def test_v3_freezes_three_specs_policies_and_output_leaf() -> None:
    config = _load(ROOT / "configs" / "infobench_calibration_cat_v3.json")
    v2_config = _load(ROOT / "configs" / "infobench_calibration_cat_v2.json")
    v3._check_frozen_config(config)
    specs = v3.load_calibration_specs(config)
    assert [spec.spec_id for spec in specs] == [
        "1pl_fixed_a1",
        "log_shrinkage_2pl_lambda16",
        "log_shrinkage_2pl_lambda4",
    ]
    assert [policy.policy_id for policy in v3.load_policies(config)] == [
        v3.PRIMARY_POLICY_ID,
        *v3.SENSITIVITY_POLICY_IDS,
    ]
    assert config["calibration_selection"]["allow_fallback"] is False
    assert config["calibration_selection"]["require_all_specs_every_inner_fold"] is True
    assert config["calibration_selection"]["survivor_selection_allowed"] is False
    assert config["selection_gates"]["allow_fallback_if_primary_fails"] is False
    assert config["selection_gates"]["minimum_nominal_precision_lower_95_ci"] == 0.90
    assert tuple(config["selection_gates"]["outer_panel_applied_gate_names"]) == (
        v3.PANEL_APPLIED_GATE_NAMES
    )
    assert tuple(config["selection_gates"]["outer_panel_diagnostic_gate_names"]) == (
        v3.PANEL_DIAGNOSTIC_GATE_NAMES
    )
    assert tuple(config["selection_gates"]["repetition_applied_gate_names"]) == (
        v3.REPETITION_APPLIED_GATE_NAMES
    )
    assert tuple(config["code_dependencies"]) == v3.CODE_DEPENDENCY_RELATIVE_PATHS
    assert config["outputs"]["phase3"] == "runs/calibration/InFoBench_v3/phase3"
    for frozen_section in (
        "cross_validation",
        "cat_policies",
        "uncertainty",
        "latent_structure",
        "limitations",
    ):
        assert config[frozen_section] == v2_config[frozen_section]
    for section in ("calibration_selection", "selection_gates"):
        for key, value in v2_config[section].items():
            assert config[section][key] == value


def test_v4_lock_chain_authorizes_exact_dense_profile(tmp_path: Path) -> None:
    config_path, lock_path, base = _synthetic_v4_lock(tmp_path)
    profile = v3._validate_numerical_lock(
        base,
        v3.load_calibration_specs(base),
        lock_path=lock_path,
        followup_config_path=config_path,
    )
    assert profile["status"] == "passed"
    assert profile["fit_grid"] == 401
    assert profile["fit_quadrature_method"] == cm.NORMAL_TRAPEZOID_QUADRATURE
    assert profile["fit_convergence_mode"] == cm.RETURNED_ITERATE_CONVERGENCE
    assert profile["fit_max_iter"] == 1500
    assert profile["fit_parameter_tolerance"] == 5e-5
    assert profile["fit_consecutive_convergence_passes"] == 2
    assert profile["eap_grid"] == 801


def test_v4_lock_rejects_evidence_tampering(tmp_path: Path) -> None:
    config_path, lock_path, base = _synthetic_v4_lock(tmp_path)
    lock = _load(lock_path)
    evidence = lock_path.parent / next(iter(lock["evidence_paths"].values()))
    evidence.write_text("{}\n", encoding="utf-8")
    with pytest.raises(v3.V3Phase3Error, match="evidence hash mismatch"):
        v3._validate_numerical_lock(
            base,
            v3.load_calibration_specs(base),
            lock_path=lock_path,
            followup_config_path=config_path,
        )


def test_v4_lock_rejects_signature_tampering(tmp_path: Path) -> None:
    config_path, lock_path, base = _synthetic_v4_lock(tmp_path)
    manifest_path = lock_path.parent / "study_manifest.json"
    manifest = _load(manifest_path)
    manifest["study_signature"]["spec_ids"] = ["1pl_fixed_a1"]
    _write_json(manifest_path, manifest)
    with pytest.raises(v3.V3Phase3Error, match="study-signature hash is invalid"):
        v3._validate_numerical_lock(
            base,
            v3.load_calibration_specs(base),
            lock_path=lock_path,
            followup_config_path=config_path,
        )


def test_pending_v4_lock_is_plan_only_blocker(tmp_path: Path) -> None:
    config_path, lock_path, base = _synthetic_v4_lock(tmp_path)
    lock_path.unlink()
    lock_path.with_suffix(".sha256").unlink()
    (lock_path.parent / "numerical_followup_decision.json").unlink()
    (lock_path.parent / "study_manifest.json").unlink()
    profile = v3._validate_numerical_lock(
        base,
        v3.load_calibration_specs(base),
        lock_path=lock_path,
        followup_config_path=config_path,
        allow_pending=True,
    )
    assert profile["status"] == "pending_blocker"
    with pytest.raises(v3.V3Phase3Error, match="lock is missing"):
        v3._validate_numerical_lock(
            base,
            v3.load_calibration_specs(base),
            lock_path=lock_path,
            followup_config_path=config_path,
            allow_pending=False,
        )


def test_plan_only_is_read_only_and_cannot_claim_official_leaf(
    tmp_path: Path,
) -> None:
    requested = tmp_path / "would_be_official"
    assert v3.main(["--plan-only", "--out-dir", str(requested)]) == 0
    assert not requested.exists()


def test_pending_lock_fails_before_creating_official_output(tmp_path: Path) -> None:
    runner = v3.V3Phase3Runner.__new__(v3.V3Phase3Runner)
    runner.args = argparse.Namespace(plan_only=False)
    runner.numerical = {"status": "pending_blocker"}
    runner.output_dir = tmp_path / "official_phase3"
    with pytest.raises(v3.V3Phase3Error, match="blocked by numerical verification"):
        runner.run()
    assert not runner.output_dir.exists()


def _fit_fixture(spec: v3.CalibrationSpec) -> dict:
    return {
        "items": ["criterion_a", "criterion_b"],
        "A": np.ones((2, 1)),
        "b": np.zeros(2),
        "R": np.eye(1),
        "dim_labels": ["instruction_following"],
        "collapsed_labels": ["instruction_following"],
        "loglik": -1.0,
        "n_params": 2,
        "n_iter": 10,
        "converged": True,
        "calibration_specification": spec.canonical,
        "diag": {},
        "grid_nodes": 401,
        "quadrature_method": cm.NORMAL_TRAPEZOID_QUADRATURE,
        "quadrature_linear_bound": 8.0,
        "convergence_mode": cm.RETURNED_ITERATE_CONVERGENCE,
        "penalized_objective": -1.0,
        "convergence_diagnostics": {
            "returned_iterate_matches_last_trace": True,
            "stopped_before_extra_mstep": True,
            "all_objective_changes_monotone_within_tolerance": True,
            "required_consecutive_passes": 2,
            "final_consecutive_passes": 2,
        },
    }


def test_fit_or_load_passes_complete_v4_fitter_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _load(ROOT / "configs" / "infobench_calibration_cat_v3.json")
    spec = v3.load_calibration_specs(config)[0]
    runner = v3.V3Phase3Runner.__new__(v3.V3Phase3Runner)
    runner.args = argparse.Namespace(
        max_iter=1500, tol=1e-4, negative_policy="drop", max_grid_nodes=50_000
    )
    runner.numerical = {
        "fit_grid": 401,
        "fit_quadrature_method": cm.NORMAL_TRAPEZOID_QUADRATURE,
        "fit_linear_bound": 8.0,
        "fit_convergence_mode": cm.RETURNED_ITERATE_CONVERGENCE,
        "fit_max_iter": 1500,
        "fit_objective_tolerance": 1e-4,
        "fit_parameter_tolerance": 5e-5,
        "fit_consecutive_convergence_passes": 2,
        "eap_grid": 801,
        "quadrature_method": cm.NORMAL_TRAPEZOID_QUADRATURE,
        "linear_bound": 8.0,
    }
    runner.matrix = pd.DataFrame(
        [[1.0, 0.0], [0.0, 1.0]],
        index=["model_a", "model_b"],
        columns=["criterion_a", "criterion_b"],
    )
    runner.q_by = {}
    runner.structure = SimpleNamespace(n_dims=1)
    runner.source_records = {}
    runner.split_audit = {
        "administration_scenario_ids": ["scenario_a"],
        "evaluation_scenario_ids": ["scenario_b"],
    }
    runner._fit_cache_key = lambda **_kwargs: "cache-key"
    runner._cache_dir = lambda **_kwargs: tmp_path / "cache"
    runner._load_fit_cache = lambda **_kwargs: None
    runner._write_fit_cache = lambda **_kwargs: None

    captured: dict = {}

    def fake_fit(_matrix, _q, args, _structure):
        captured.update(vars(args))
        return _fit_fixture(spec)

    bank = SimpleNamespace(
        scenario_ids=("scenario_a", "scenario_b"),
        latent_correlation=np.eye(1),
    )
    monkeypatch.setattr(v3, "_fit_structure_dense", fake_fit)
    monkeypatch.setattr(v3.scenario_cv, "build_fold_bank", lambda *_a, **_k: (bank, {}))
    monkeypatch.setattr(v3.scat, "build_quadrature", lambda *_a, **_k: "quadrature")

    bundle = runner.fit_or_load(
        repeat=0,
        outer_fold=0,
        inner_fold=0,
        role="test",
        training_model_ids=["model_a"],
        forbidden_model_ids=["model_b"],
        spec=spec,
    )
    assert bundle.quadrature == "quadrature"
    assert captured["grid"] == 401
    assert captured["quadrature_method"] == cm.NORMAL_TRAPEZOID_QUADRATURE
    assert captured["linear_bound"] == 8.0
    assert captured["convergence_mode"] == cm.RETURNED_ITERATE_CONVERGENCE
    assert captured["max_iter"] == 1500
    assert captured["tol"] == 1e-4
    assert captured["parameter_tol"] == 5e-5
    assert captured["consecutive_convergence_passes"] == 2


def test_v4_profile_rejects_legacy_convergence() -> None:
    raw = {
        "effective_global_profile": {
            "fit": {
                "grid": 401,
                "quadrature_method": cm.NORMAL_TRAPEZOID_QUADRATURE,
                "linear_bound": 8.0,
                "convergence_mode": cm.LEGACY_PRE_MSTEP_CONVERGENCE,
                "max_iter": 1500,
                "objective_tolerance": 1e-4,
                "parameter_tolerance": 5e-5,
                "consecutive_convergence_passes": 2,
            },
            "eap": {
                "grid": 801,
                "quadrature_method": cm.NORMAL_TRAPEZOID_QUADRATURE,
                "linear_bound": 8.0,
                "tail_region": 7.5,
            },
        }
    }
    with pytest.raises(v3.V3Phase3Error, match="fit_convergence_mode"):
        v3._extract_v4_profile(raw)


def test_v3_freezes_direct_runtime_dependency_inventory() -> None:
    required = {
        "tutor_cat/dataio.py",
        "tutor_cat/engine.py",
        "tutor_cat/selector.py",
        "tutor_cat/schemas.py",
    }
    assert required <= set(v3.CODE_DEPENDENCY_RELATIVE_PATHS)
    hashes = v3._code_hashes()
    assert set(hashes) == set(v3.CODE_DEPENDENCY_RELATIVE_PATHS)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("calibration_selection", "require_all_specs_every_inner_fold", False),
        ("calibration_selection", "survivor_selection_allowed", True),
        ("selection_gates", "minimum_nominal_precision_lower_95_ci", 0.89),
        ("selection_gates", "treat_260_model_repeat_rows_as_independent", True),
        ("selection_gates", "outer_panel_diagnostic_gate_names", []),
        ("selection_gates", "repetition_applied_gate_names", []),
    ],
)
def test_v3_rejects_changed_fail_closed_or_gate_scope_contract(
    section: str, field: str, value: object
) -> None:
    config = _load(ROOT / "configs" / "infobench_calibration_cat_v3.json")
    config[section][field] = value
    with pytest.raises(v3.V3Phase3Error):
        v3._check_frozen_config(config)


def test_v3_rejects_changed_code_dependency_inventory() -> None:
    config = _load(ROOT / "configs" / "infobench_calibration_cat_v3.json")
    config["code_dependencies"].remove("tutor_cat/selector.py")
    with pytest.raises(v3.V3Phase3Error, match="dependency inventory"):
        v3._check_frozen_config(config)


def test_official_output_leaf_is_immutable_but_plan_only_may_report(
    tmp_path: Path,
) -> None:
    config = _load(ROOT / "configs" / "infobench_calibration_cat_v3.json")
    configured = (ROOT / config["outputs"]["phase3"]).resolve()
    assert v3._resolve_phase3_output_dir(config, configured, plan_only=False) == configured
    arbitrary = tmp_path / "not_the_frozen_leaf"
    with pytest.raises(v3.V3Phase3Error, match="must equal the frozen"):
        v3._resolve_phase3_output_dir(config, arbitrary, plan_only=False)
    assert v3._resolve_phase3_output_dir(config, arbitrary, plan_only=True) == arbitrary.resolve()
    assert not arbitrary.exists()


def _completed_resume_runner(
    tmp_path: Path, *, status: str = "phase3_complete"
) -> tuple[v3.V3Phase3Runner, Path, dict]:
    output = tmp_path / "phase3"
    output.mkdir()
    inventory: dict[str, dict[str, str]] = {}
    required = (
        v3.SELECTION_ONLY_REQUIRED_OUTPUTS
        if status == "phase3_terminal_selection_failed"
        else v3.REQUIRED_OUTPUTS
    )
    for name in required:
        if name == "manifest.json":
            continue
        path = output / name
        path.write_text(f"fixture:{name}\n", encoding="utf-8")
        inventory[name] = {
            "path": v3._display_path(path),
            "sha256": v3._sha256(path),
        }
    manifest = {
        "schema_version": v3.SCRIPT_SCHEMA,
        "status": status,
        "study_signature": "resume-study",
        "outputs": inventory,
    }
    _write_json(output / "manifest.json", manifest)
    runner = v3.V3Phase3Runner.__new__(v3.V3Phase3Runner)
    runner.args = argparse.Namespace(resume=True)
    runner.output_dir = output
    runner.study_signature = "resume-study"
    runner._already_complete = False
    return runner, output, manifest


def test_completed_resume_requires_exact_verified_inventory(tmp_path: Path) -> None:
    runner, _output, _manifest = _completed_resume_runner(tmp_path)
    runner._prepare_output()
    assert runner._already_complete is True


def test_selection_failure_resume_uses_selection_only_inventory(tmp_path: Path) -> None:
    runner, _output, manifest = _completed_resume_runner(
        tmp_path, status="phase3_terminal_selection_failed"
    )
    assert set(manifest["outputs"]) == set(v3.SELECTION_ONLY_REQUIRED_OUTPUTS) - {"manifest.json"}
    assert (set(v3.REQUIRED_OUTPUTS) - set(v3.SELECTION_ONLY_REQUIRED_OUTPUTS)).isdisjoint(
        manifest["outputs"]
    )
    runner._prepare_output()
    assert runner._already_complete is True


@pytest.mark.parametrize("mutation", ["missing", "extra", "hash"])
def test_completed_resume_rejects_missing_extra_or_bad_hash(tmp_path: Path, mutation: str) -> None:
    runner, output, manifest = _completed_resume_runner(tmp_path)
    first = next(name for name in v3.REQUIRED_OUTPUTS if name != "manifest.json")
    if mutation == "missing":
        del manifest["outputs"][first]
    elif mutation == "extra":
        extra = output / "unexpected.csv"
        extra.write_text("unexpected\n", encoding="utf-8")
        manifest["outputs"][extra.name] = {
            "path": v3._display_path(extra),
            "sha256": v3._sha256(extra),
        }
    else:
        manifest["outputs"][first]["sha256"] = "0" * 64
    _write_json(output / "manifest.json", manifest)
    with pytest.raises(v3.V3Phase3Error, match="inventory|failed hash"):
        runner._prepare_output()


def _valid_inner_evidence() -> tuple[list[dict], list[v3.CalibrationSpec], dict[int, list[str]]]:
    config = _load(ROOT / "configs" / "infobench_calibration_cat_v3.json")
    specs = v3.load_calibration_specs(config)
    expected = {
        inner_fold: [f"model_{inner_fold}_a", f"model_{inner_fold}_b"]
        for inner_fold in range(v3.EXPECTED_INNER_FOLDS)
    }
    rows: list[dict] = []
    for inner_fold, models in expected.items():
        support_hash = v3._canonical_hash({"inner_fold": inner_fold})
        for spec in specs:
            for model in models:
                rows.append(
                    {
                        "inner_fold": inner_fold,
                        "spec_id": spec.spec_id,
                        "model": model,
                        "status": "ok",
                        "fit_eligible": True,
                        "all_specs_fitted": True,
                        "common_support_verified": True,
                        "fit_cache_key": f"fit-{inner_fold}-{spec.spec_id}",
                        "common_support_sha256": support_hash,
                        "n_common_admin_items": 10,
                        "n_common_evaluation_items": 5,
                        "n_cells": 5,
                        "log_loss_sum": 1.0,
                        "brier_sum": 0.5,
                        "model_log_loss": 0.2,
                        "observed_common_cells_sha256": v3._canonical_hash(
                            {"inner_fold": inner_fold, "model": model}
                        ),
                    }
                )
    return rows, specs, expected


def test_complete_inner_evidence_requires_all_twelve_spec_fold_blocks() -> None:
    rows, specs, expected = _valid_inner_evidence()
    audit = v3.audit_complete_inner_evidence(rows, specs, expected)
    assert audit["passed"] is True
    assert audit["expected_spec_fold_combinations"] == 12
    assert audit["valid_spec_fold_combinations"] == 12
    assert audit["failed_checks"] == []


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_spec_fold",
        "missing_validation_model",
        "duplicate_validation_model",
        "bad_status",
        "fit_ineligible",
        "not_all_specs_fitted",
        "common_support_unverified",
        "zero_admin_support",
        "zero_evaluation_support",
        "zero_cells",
        "divergent_support_hash",
        "divergent_cell_hash",
    ],
)
def test_inner_evidence_audit_fails_closed_for_every_invalidity(mutation: str) -> None:
    rows, specs, expected = _valid_inner_evidence()
    target_spec = specs[0].spec_id
    target = next(row for row in rows if row["inner_fold"] == 0 and row["spec_id"] == target_spec)
    if mutation == "missing_spec_fold":
        rows = [
            row for row in rows if not (row["inner_fold"] == 0 and row["spec_id"] == target_spec)
        ]
    elif mutation == "missing_validation_model":
        rows.remove(target)
    elif mutation == "duplicate_validation_model":
        rows.append(dict(target))
    elif mutation == "bad_status":
        target["status"] = "prediction_error"
    elif mutation == "fit_ineligible":
        target["fit_eligible"] = False
    elif mutation == "not_all_specs_fitted":
        target["all_specs_fitted"] = False
    elif mutation == "common_support_unverified":
        target["common_support_verified"] = False
    elif mutation == "zero_admin_support":
        target["n_common_admin_items"] = 0
    elif mutation == "zero_evaluation_support":
        target["n_common_evaluation_items"] = 0
    elif mutation == "zero_cells":
        target["n_cells"] = 0
    elif mutation == "divergent_support_hash":
        target["common_support_sha256"] = "a" * 64
    elif mutation == "divergent_cell_hash":
        target["observed_common_cells_sha256"] = "b" * 64
    audit = v3.audit_complete_inner_evidence(rows, specs, expected)
    assert audit["passed"] is False
    assert audit["failed_checks"]


def test_incomplete_inner_evidence_never_calls_survivor_selector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows, specs, expected = _valid_inner_evidence()
    missing_spec = specs[-1].spec_id
    rows = [row for row in rows if not (row["inner_fold"] == 3 and row["spec_id"] == missing_spec)]
    runner = v3.V3Phase3Runner.__new__(v3.V3Phase3Runner)
    runner.specs = specs
    runner.split_audit = {
        "model_to_family": {
            **{model: f"family_{model}" for models in expected.values() for model in models},
            "outer_test": "family_outer_test",
        }
    }
    runner.fit_or_load = lambda **kwargs: SimpleNamespace(
        cache_key=f"cache-{kwargs['inner_fold']}-{kwargs['spec'].spec_id}"
    )
    runner._score_inner_common_support = lambda **kwargs: [
        dict(row) for row in rows if row["inner_fold"] == kwargs["inner_fold"]
    ]
    selector_called = False

    def forbidden_selector(*_args, **_kwargs):
        nonlocal selector_called
        selector_called = True
        raise AssertionError("survivor selector must not run")

    monkeypatch.setattr(v3, "select_calibration_spec_one_se", forbidden_selector)
    outer = {
        "outer_fold": 0,
        "train_model_ids": [model for models in expected.values() for model in models],
        "test_model_ids": ["outer_test"],
        "inner_folds": [
            {
                "inner_fold": inner_fold,
                "fit_model_ids": [
                    model
                    for other_fold, models in expected.items()
                    if other_fold != inner_fold
                    for model in models
                ],
                "validation_model_ids": models,
            }
            for inner_fold, models in expected.items()
        ],
    }
    panel, _aggregate = runner._select_panel(0, outer)
    assert selector_called is False
    assert panel["selected_spec_id"] is None
    assert panel["status"] == "selection_blocked_incomplete_inner_evidence"
    assert panel["inner_selection"]["complete_inner_evidence_audit"]["passed"] is False
    assert panel["inner_selection"]["survivor_selection_used"] is False


def test_one_failed_selection_stops_all_outer_policy_evaluation(
    tmp_path: Path,
) -> None:
    runner = v3.V3Phase3Runner.__new__(v3.V3Phase3Runner)
    runner.args = argparse.Namespace(plan_only=False, resume=False)
    runner.numerical = {"status": "passed"}
    runner.output_dir = tmp_path / "phase3"
    runner.study_signature = "global-stop-study"
    runner._already_complete = False
    runner.split_audit = {
        "repetitions": [
            {
                "repeat": repeat,
                "outer_folds": [
                    {
                        "outer_fold": outer_fold,
                        "train_model_ids": [],
                        "test_model_ids": [],
                        "inner_folds": [],
                    }
                    for outer_fold in range(v3.EXPECTED_OUTER_FOLDS)
                ],
            }
            for repeat in range(v3.EXPECTED_REPEATS)
        ]
    }
    runner._fold_assignment_payload = lambda: {"schema_version": v3.SCRIPT_SCHEMA}
    runner._base_manifest = lambda status, **_kwargs: {
        "schema_version": v3.SCRIPT_SCHEMA,
        "status": status,
        "study_signature": runner.study_signature,
    }

    def fake_select(repeat, outer):
        failed = repeat == 0 and int(outer["outer_fold"]) == 0
        return (
            {
                "panel_id": f"repeat_{repeat:02d}_outer_{int(outer['outer_fold']):02d}",
                "repeat": repeat,
                "outer_fold": int(outer["outer_fold"]),
                "status": (
                    "selection_blocked_incomplete_inner_evidence"
                    if failed
                    else "selection_complete"
                ),
                "selected_spec_id": None if failed else "1pl_fixed_a1",
                "selected_calibration_specification": None if failed else {},
                "inner_selection": {},
            },
            [],
        )

    runner._select_panel = fake_select
    terminal_calls = 0

    def fake_terminal_writer(**_kwargs):
        nonlocal terminal_calls
        terminal_calls += 1
        (runner.output_dir / "terminal_selection_only.marker").write_text(
            "stopped before outer evaluation\n", encoding="utf-8"
        )

    runner._write_terminal_selection_failure_outputs = fake_terminal_writer
    outer_calls = 0

    def forbidden_outer(*_args, **_kwargs):
        nonlocal outer_calls
        outer_calls += 1
        raise AssertionError("outer policy evaluation must not be called")

    runner._evaluate_outer_panel = forbidden_outer
    runner._write_phase3_outputs = lambda **_kwargs: (_ for _ in ()).throw(
        AssertionError("full Phase-3 output writer must not be called")
    )
    assert runner.run() == 0
    assert terminal_calls == 1
    assert outer_calls == 0
    assert not list(runner.output_dir.rglob("*outer_policy_evaluation*"))
    assert not list(runner.output_dir.rglob("outer_oof_per_model.csv"))


def test_terminal_selection_failure_writes_only_honest_selection_artifacts(
    tmp_path: Path,
) -> None:
    config = _load(ROOT / "configs" / "infobench_calibration_cat_v3.json")
    runner = v3.V3Phase3Runner.__new__(v3.V3Phase3Runner)
    runner.output_dir = tmp_path / "phase3"
    runner.output_dir.mkdir()
    runner.config_path = tmp_path / "config.json"
    runner.split_path = tmp_path / "splits.json"
    _write_json(runner.config_path, config)
    _write_json(runner.split_path, {"fixture": True})
    runner.study_signature = "terminal-selection-study"
    runner.input_hashes = {"response_matrix": "a" * 64}
    runner.code_hashes = {"scripts/fixture.py": "b" * 64}
    runner.environment = {"canonical_sha256": "c" * 64}
    runner.policies = v3.load_policies(config)
    runner.config = config
    runner.args = argparse.Namespace(
        seed=20260805,
        top_n=5,
        max_scenarios=50,
        minimum_scored_criteria=15,
        max_iter=1500,
        tol=1e-4,
        mwle_ridge=1e-6,
        max_grid_nodes=50_000,
        negative_policy="drop",
        metric_bootstrap_replicates=2000,
    )
    runner.numerical = {
        "fit_parameter_tolerance": 5e-5,
        "fit_consecutive_convergence_passes": 2,
        "fit_convergence_mode": cm.RETURNED_ITERATE_CONVERGENCE,
        "fit_quadrature_method": cm.NORMAL_TRAPEZOID_QUADRATURE,
        "fit_linear_bound": 8.0,
    }
    runner._base_manifest = lambda status, **_kwargs: {
        "schema_version": v3.SCRIPT_SCHEMA,
        "status": status,
        "study_signature": runner.study_signature,
    }
    _write_json(runner.output_dir / "fold_assignments.json", {"fixture": True})
    _write_json(runner.output_dir / "pre_outer_selection_lock.json", {"fixture": True})
    panels: list[dict] = []
    inner_results: list[dict] = []
    for repeat in range(v3.EXPECTED_REPEATS):
        for outer_fold in range(v3.EXPECTED_OUTER_FOLDS):
            failed = repeat == 0 and outer_fold == 0
            valid_combinations = 11 if failed else 12
            panels.append(
                {
                    "panel_id": f"repeat_{repeat:02d}_outer_{outer_fold:02d}",
                    "repeat": repeat,
                    "outer_fold": outer_fold,
                    "status": (
                        "selection_blocked_incomplete_inner_evidence"
                        if failed
                        else "selection_complete"
                    ),
                    "error": "one missing block" if failed else "",
                    "selected_spec_id": None if failed else "1pl_fixed_a1",
                    "selected_calibration_specification": None if failed else {},
                    "inner_selection": {
                        "complete_inner_evidence_audit": {
                            "passed": not failed,
                            "valid_spec_fold_combinations": valid_combinations,
                            "expected_spec_fold_combinations": 12,
                        }
                    },
                }
            )
            inner_results.append(
                {
                    "repeat": repeat,
                    "outer_fold": outer_fold,
                    "simplicity_rank": 0,
                    "spec_id": "1pl_fixed_a1",
                }
            )
    runner._write_terminal_selection_failure_outputs(
        panels=panels,
        inner_results=inner_results,
        started_at="2026-08-05T00:00:00+00:00",
    )
    observed_files = {path.name for path in runner.output_dir.iterdir() if path.is_file()}
    assert observed_files == set(v3.SELECTION_ONLY_REQUIRED_OUTPUTS)
    assert observed_files.isdisjoint(
        set(v3.REQUIRED_OUTPUTS) - set(v3.SELECTION_ONLY_REQUIRED_OUTPUTS)
    )
    decision = _load(runner.output_dir / "phase3_decision.json")
    assert decision["status"] == "fail_selection_incomplete"
    assert decision["terminal_stop"]["outer_outcomes_opened"] is False
    assert decision["outer_evaluation"]["evaluated_outer_panels_numerator"] == 0
    assert decision["outer_evaluation"]["expected_outer_panels_denominator"] == 25
    assert decision["calibration_selection"]["complete_panels_numerator"] == 24
    assert decision["calibration_selection"]["expected_panels_denominator"] == 25
    assert decision["calibration_selection"]["valid_spec_fold_combinations_numerator"] == 299
    assert decision["calibration_selection"]["expected_spec_fold_combinations_denominator"] == 300
    manifest = _load(runner.output_dir / "manifest.json")
    assert manifest["status"] == "phase3_terminal_selection_failed"
    assert set(manifest["outputs"]) == set(v3.SELECTION_ONLY_REQUIRED_OUTPUTS) - {"manifest.json"}
    for name, provenance in manifest["outputs"].items():
        assert provenance["sha256"] == v3._sha256(runner.output_dir / name)


def _passing_gate_metrics_with_failed_nominal_ci() -> dict[str, object]:
    return {
        "replay_success_rate": 1.0,
        "mwle_convergence_rate": 1.0,
        "nominal_precision_rate": 1.0,
        "nominal_precision_lower_95_ci": 0.89,
        "recovery_correlation_lower_95_ci": 0.90,
        "recovery_slope": 1.0,
        "disjoint_pass_rate_mae": 0.01,
        "disjoint_pass_rate_bias": 0.0,
        "scenario_reduction_vs_random": 0.60,
        "paired_ci_favors_cat": True,
    }


def test_nominal_precision_ci_is_diagnostic_for_panel_but_applied_to_repetition() -> None:
    config = _load(ROOT / "configs" / "infobench_calibration_cat_v3.json")
    gates = dict(config["selection_gates"])
    gates["require_paired_ci_favors_cat"] = True
    metrics = _passing_gate_metrics_with_failed_nominal_ci()
    panel = v3.apply_scoped_absolute_gates(metrics, gates, scope="outer_panel")
    repetition = v3.apply_scoped_absolute_gates(metrics, gates, scope="repetition")
    assert panel["passed"] is True
    assert "nominal_precision_lower_95_ci" in panel["diagnostic_failures"]
    assert "nominal_precision_lower_95_ci" not in panel["failures"]
    assert repetition["passed"] is False
    assert "nominal_precision_lower_95_ci" in repetition["failures"]


def test_gate_aggregation_uses_panels_or_52_unique_oof_tutors_never_260(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _load(ROOT / "configs" / "infobench_calibration_cat_v3.json")
    policy = v3.load_policies(config)[0]
    models = [f"model_{index:02d}" for index in range(v3.EXPECTED_MODELS)]
    folds = {
        outer_fold: [model for index, model in enumerate(models) if index % 5 == outer_fold]
        for outer_fold in range(v3.EXPECTED_OUTER_FOLDS)
    }
    runner = v3.V3Phase3Runner.__new__(v3.V3Phase3Runner)
    runner.config = config
    runner.policies = [policy]
    runner.args = argparse.Namespace(seed=20260805, metric_bootstrap_replicates=100)
    runner.matrix = pd.DataFrame(index=models)
    runner.split_audit = {
        "model_to_family": {model: f"family_{model}" for model in models},
        "repetitions": [
            {
                "repeat": repeat,
                "outer_folds": [
                    {"outer_fold": outer_fold, "test_model_ids": folds[outer_fold]}
                    for outer_fold in range(v3.EXPECTED_OUTER_FOLDS)
                ],
            }
            for repeat in range(v3.EXPECTED_REPEATS)
        ],
    }
    calls: list[int] = []

    def fake_aggregate(rows, *_args, **_kwargs):
        calls.append(len(rows))
        return _passing_gate_metrics_with_failed_nominal_ci()

    monkeypatch.setattr(v3, "aggregate_policy_rows_clustered", fake_aggregate)
    monkeypatch.setattr(
        v3.v1,
        "aggregate_disjoint_predictions",
        lambda *_args, **_kwargs: {"mode": "fixture"},
    )
    outer_rows = [
        {
            "repeat": repeat,
            "outer_fold": outer_fold,
            "policy_id": policy.policy_id,
            "model": model,
        }
        for repeat in range(v3.EXPECTED_REPEATS)
        for outer_fold, fold_models in folds.items()
        for model in fold_models
    ]
    gate_rows, _prediction_rows = runner._gate_rows(outer_rows)
    assert len(outer_rows) == 260
    assert max(calls) == 52
    assert 260 not in calls
    repetition_rows = [row for row in gate_rows if row["scope"] == "repetition"]
    assert len(repetition_rows) == 5
    assert all(row["n_inference_rows"] == 52 for row in repetition_rows)
    assert all(row["n_unique_tutor_models"] == 52 for row in repetition_rows)
    assert all(row["inference_scope_complete"] is True for row in repetition_rows)
    assert all(row["model_repeat_rows_treated_as_independent"] is False for row in repetition_rows)
    panel_rows = [row for row in gate_rows if row["scope"] == "outer_panel"]
    assert all(row["all_gates_pass"] is True for row in panel_rows)
    assert all(row["all_gates_pass"] is False for row in repetition_rows)
