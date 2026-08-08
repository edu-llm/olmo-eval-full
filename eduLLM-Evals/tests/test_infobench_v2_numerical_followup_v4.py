"""Fail-closed tests for the append-only InFoBench V4 numerical study."""

from __future__ import annotations

import ast
import copy
import importlib.util
import inspect
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_infobench_v2_numerical_followup_v4.py"
CONFIG = ROOT / "configs" / "infobench_v2_numerical_followup_v4.json"
DESIGN = ROOT / "docs" / "infobench_v4_dense_fitter_remediation_plan.md"


def _load_module():
    spec = importlib.util.spec_from_file_location("infobench_v2_numerical_followup_v4", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


followup = _load_module()


def _raw_config() -> dict[str, Any]:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _write_finalization_evidence(
    context,
    *,
    refit_passed: bool = True,
    fixed_passed: bool = True,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    context.out_dir.mkdir(parents=True, exist_ok=True)
    (context.out_dir / "study_manifest.json").write_text(
        json.dumps({"schema_version": followup.RUNNER_SCHEMA, "status": "running"}),
        encoding="utf-8",
    )
    for spec in context.specs:
        for scope in context.scopes:
            for nodes, start in (
                (401, "cold"),
                (801, "cold"),
                (801, "continuation"),
            ):
                path = followup._fit_dir(context, spec, nodes, scope, start) / "fit_manifest.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps({"fit_validity": {"fit_valid": True}}),
                    encoding="utf-8",
                )
            selection = followup._selection_path(context, spec, scope)
            selection.parent.mkdir(parents=True, exist_ok=True)
            selection.write_text(json.dumps({"selection_valid": True}), encoding="utf-8")
        refit_path = followup._fit_comparison_dir(context, spec) / "fit_pair_gate.json"
        refit_path.parent.mkdir(parents=True, exist_ok=True)
        refit_path.write_text(json.dumps({"passed": refit_passed}), encoding="utf-8")
        fixed_path = context.out_dir / "fixed_bank" / spec.namespace / "fixed_bank_gate.json"
        fixed_path.parent.mkdir(parents=True, exist_ok=True)
        fixed_path.write_text(json.dumps({"passed": fixed_passed}), encoding="utf-8")
    refit = {spec.spec_id: {"passed": refit_passed} for spec in context.specs}
    fixed = {
        spec.spec_id: {
            "passed": fixed_passed,
            "numerical_scoring_gate": {
                "passed": fixed_passed,
                "tail_mass": {"passed": fixed_passed},
            },
        }
        for spec in context.specs
    }
    return refit, fixed


def test_config_contract_dependencies_and_history_are_frozen() -> None:
    raw = _raw_config()
    followup._validate_config(raw)
    assert followup._canonical_sha256(raw["frozen_contract"]) == followup.FROZEN_CONTRACT_SHA256
    assert raw["design_source"]["sha256"] == followup._sha256(DESIGN)
    for relative, digest in raw["code_freeze"]["dependencies"].items():
        assert digest == followup._sha256(ROOT / relative)
    history = followup._verify_historical_evidence(raw)
    assert set(history) == {
        "v1_aborted",
        "v2_terminal_nonpromotable",
        "v3_terminal_blocked",
    }
    assert all(record["promotable"] is False for record in history.values())
    assert raw["historical_evidence"]["reuse_policy"] == {
        "v1_fit_or_checkpoint_artifacts_reused": 0,
        "v2_fit_or_checkpoint_artifacts_reused": 0,
        "v3_fit_or_checkpoint_artifacts_reused": 0,
    }


def test_context_has_exact_three_specs_six_scopes_and_54_fit_plan() -> None:
    context = followup.load_context(CONFIG)
    assert [spec.spec_id for spec in context.specs] == list(followup.PRIMARY_SPEC_IDS)
    assert [scope.scope_id for scope in context.scopes] == list(followup.SCOPE_IDS)
    schedule = followup.runtime_schedule(context)
    assert schedule["fit_schedule"] == {
        "grid401_cold": 18,
        "grid801_cold": 18,
        "grid801_continuation_from_grid401": 18,
        "minimum_new_fits": 54,
        "maximum_new_fits": 54,
    }
    assert schedule["adaptive_runs"] == 0
    assert schedule["calibration_model_selection_runs"] == 0
    assert schedule["historical_v1_fit_or_checkpoint_artifacts_reused"] == 0
    assert schedule["historical_v2_fit_or_checkpoint_artifacts_reused"] == 0
    assert schedule["historical_v3_fit_or_checkpoint_artifacts_reused"] == 0


def test_refit_gate_producer_and_consumers_share_one_canonical_path() -> None:
    context = followup.load_context(CONFIG)
    for spec in context.specs:
        assert followup._fit_comparison_dir(context, spec) == (
            context.out_dir / "fit_comparisons" / "401_vs_801" / spec.namespace
        )
    for function in (
        followup._run_fit_comparison,
        followup._run_fixed_bank_scoring,
        followup._finalize,
    ):
        assert "_fit_comparison_dir(" in inspect.getsource(function)


def test_run_wires_exact_54_fits_and_no_adaptive_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = followup.load_context(CONFIG)
    matrix = object()
    q_by = object()
    item_to_scenario = {"criterion": "scenario"}
    fit_calls: list[dict[str, Any]] = []
    start_calls: list[tuple[str, str]] = []
    refit_calls: list[str] = []
    fixed_calls: list[str] = []

    monkeypatch.setattr(followup, "_prepare", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        followup,
        "_runtime_objects",
        lambda _context: (matrix, q_by, item_to_scenario),
    )

    def record_fit(
        observed_context,
        spec,
        scope,
        observed_matrix,
        observed_q,
        **kwargs,
    ) -> None:
        assert observed_context is context
        assert observed_matrix is matrix
        assert observed_q is q_by
        fit_calls.append({"spec": spec.spec_id, "scope": scope.scope_id, **kwargs})

    def record_start(_context, spec, scope, observed_matrix, *, resume) -> dict[str, Any]:
        assert observed_matrix is matrix
        assert resume is False
        start_calls.append((spec.spec_id, scope.scope_id))
        return {"selection_valid": True}

    def record_refit(
        _context,
        spec,
        observed_matrix,
        observed_mapping,
        *,
        resume,
    ) -> dict[str, Any]:
        assert observed_matrix is matrix
        assert observed_mapping is item_to_scenario
        assert resume is False
        refit_calls.append(spec.spec_id)
        return {"passed": True}

    def record_fixed(
        _context,
        spec,
        observed_matrix,
        observed_mapping,
        *,
        resume,
    ) -> dict[str, Any]:
        assert observed_matrix is matrix
        assert observed_mapping is item_to_scenario
        assert resume is False
        fixed_calls.append(spec.spec_id)
        return {
            "passed": True,
            "numerical_scoring_gate": {
                "passed": True,
                "tail_mass": {"passed": True},
            },
        }

    monkeypatch.setattr(followup, "_run_fit_stage", record_fit)
    monkeypatch.setattr(followup, "_run_start_selection", record_start)
    monkeypatch.setattr(followup, "_run_fit_comparison", record_refit)
    monkeypatch.setattr(followup, "_run_fixed_bank_scoring", record_fixed)
    monkeypatch.setattr(
        followup,
        "_finalize",
        lambda _context, refit, fixed: {
            "passed": set(refit) == set(fixed) == set(followup.PRIMARY_SPEC_IDS)
        },
    )

    assert followup.run(context, resume=False) == {"passed": True}
    assert len(fit_calls) == 54
    assert len(start_calls) == 18
    assert refit_calls == list(followup.PRIMARY_SPEC_IDS)
    assert fixed_calls == list(followup.PRIMARY_SPEC_IDS)
    for spec in followup.PRIMARY_SPEC_IDS:
        for scope in followup.SCOPE_IDS:
            calls = [call for call in fit_calls if call["spec"] == spec and call["scope"] == scope]
            assert [(call["nodes"], call["start"]) for call in calls] == [
                (401, "cold"),
                (801, "cold"),
                (801, "continuation"),
            ]
            assert calls[0]["continuation_source"] is None
            assert calls[1]["continuation_source"] is None
            expected = context.out_dir / "fits" / spec / "nodes_0401" / scope / "cold" / "fit.npz"
            assert calls[2]["continuation_source"] == expected


def test_runner_has_no_cat_or_adaptive_execution_import() -> None:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert not any("scenario_cat_lib" in name for name in imported)
    assert not any(name.endswith("tutor_cat.mirt") for name in imported)


def test_fit_validity_fails_unfinished_or_high_gradient_inner_optimizer() -> None:
    spec = followup.ExactSpec(
        spec_id="1pl_fixed_a1",
        family=followup.cm.ONE_PL,
        ridge=None,
        log_a_shrinkage=None,
        canonical={},
    )
    fit = {
        "A": np.ones((2, 1)),
        "b": np.zeros(2),
        "R": np.eye(1),
        "penalized_objective": -1.0,
        "converged": True,
        "convergence_diagnostics": {
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
            "returned_iterate_matches_last_trace": True,
            "final_exact_recomputation": True,
            "stopped_before_extra_mstep": True,
            "all_objective_changes_monotone_within_tolerance": True,
        },
    }
    valid = followup._fit_validity(fit, spec, maximum_inner_gradient=1e-6)
    assert valid["fit_valid"] is True

    unfinished = copy.deepcopy(fit)
    unfinished["convergence_diagnostics"]["trace"][0]["mstep_optimizer"]["n_converged"] = 1
    result = followup._fit_validity(unfinished, spec, maximum_inner_gradient=1e-6)
    assert result["all_item_optimizers_finished"] is False
    assert result["fit_valid"] is False

    high_gradient = copy.deepcopy(fit)
    high_gradient["convergence_diagnostics"]["trace"][0]["mstep_optimizer"]["max_abs_gradient"] = (
        2e-6
    )
    result = followup._fit_validity(high_gradient, spec, maximum_inner_gradient=1e-6)
    assert result["inner_gradient_within_tolerance"] is False
    assert result["fit_valid"] is False


def test_continuation_cannot_rescue_an_invalid_cold_fit() -> None:
    fit = {
        "A": np.ones((2, 1)),
        "b": np.zeros(2),
        "R": np.eye(1),
        "items": ["c1", "c2"],
        "penalized_objective": -2.0,
    }
    cold_manifest = {
        "family": followup.cm.ONE_PL,
        "n_observed_train_cells": 4,
        "fit_validity": {"fit_valid": False},
    }
    continuation_manifest = {
        "family": followup.cm.ONE_PL,
        "n_observed_train_cells": 4,
        "fit_validity": {"fit_valid": True},
    }
    gate = followup._start_agreement_gate(
        cold_fit=fit,
        continuation_fit=copy.deepcopy(fit),
        cold_manifest=cold_manifest,
        continuation_manifest=continuation_manifest,
        raw=np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        thresholds=_raw_config()["frozen_contract"]["start_agreement_gate"],
    )
    assert gate["continuation_fit_valid"] is True
    assert gate["cold_fit_valid"] is False
    assert gate["passed"] is False


def test_refit_maximum_theta_shift_is_diagnostic_not_a_gate() -> None:
    rows = pd.DataFrame(
        {
            "scope": ["full"] * 100,
            "model": [f"m{index}" for index in range(100)],
            "support": ["full_bank"] * 100,
            "theta_401": np.zeros(100),
            "theta_801": [0.5, *([0.0] * 99)],
        }
    )
    gate = followup._refit_theta_gate(rows, _raw_config()["frozen_contract"]["refit_bank_gate"])
    assert gate["maximum_absolute_theta_shift_diagnostic_only"] == 0.5
    assert gate["maximum_is_a_gate"] is False
    assert gate["passed"] is True


def test_fixed_bank_maximum_shift_and_tail_mass_are_both_gates() -> None:
    thresholds = _raw_config()["frozen_contract"]["fixed_bank_scoring_gate"]
    rows = pd.DataFrame(
        {
            "theta_bound8_801": [0.0, 0.0],
            "theta_bound8_1601": [0.004, 0.0],
            "theta_bound10_1001": [0.0, 0.0],
            "tail_bound8_801": [1e-6, 1e-6],
            "tail_bound8_1601": [1e-6, 1e-6],
            "tail_bound10_1001": [1e-6, 1e-6],
        }
    )
    gate = followup._fixed_bank_scoring_gate(rows, thresholds)
    assert gate["passed"] is True

    excessive_shift = rows.copy()
    excessive_shift.loc[0, "theta_bound8_1601"] = 0.006
    gate = followup._fixed_bank_scoring_gate(excessive_shift, thresholds)
    assert gate["comparisons"]["bound8_801_vs_bound8_1601"]["passed"] is False
    assert gate["passed"] is False

    excessive_tail = rows.copy()
    excessive_tail.loc[0, "tail_bound10_1001"] = 2e-5
    gate = followup._fixed_bank_scoring_gate(excessive_tail, thresholds)
    assert gate["tail_mass"]["passed"] is False
    assert gate["passed"] is False


def test_fit_observed_cell_denominator_uses_only_fitted_items() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "matrix.loc[list(scope.train_models), fitted_items]" in source
    assert "missing_fitted_items" in source


def test_prepare_is_append_only_and_terminal_resume_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = replace(followup.load_context(CONFIG), out_dir=tmp_path / "run")
    monkeypatch.setattr(followup, "_assert_safe_output", lambda _context: None)
    monkeypatch.setattr(followup, "_verify_code_freeze", lambda _raw: {})
    followup._prepare(context, resume=False)
    followup._prepare(context, resume=True)
    with pytest.raises(followup.V4Error, match="already exists"):
        followup._prepare(context, resume=False)
    manifest_path = context.out_dir / "study_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "blocked_numerical_followup"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(followup.V4Error, match="terminal V4 evidence"):
        followup._prepare(context, resume=True)


def test_lock_truth_table_and_evidence_path_hash_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = followup.load_context(CONFIG)

    def display(path: Path) -> str:
        return path.resolve().relative_to(tmp_path.resolve()).as_posix()

    monkeypatch.setattr(followup, "_display", display)
    passing = replace(base, out_dir=tmp_path / "passing")
    refit, fixed = _write_finalization_evidence(passing)
    decision = followup._finalize(passing, refit, fixed)
    assert decision["passed"] is True
    lock_path = passing.out_dir / "numerical_followup_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    assert lock["fit_convergence_mode"] == "returned_iterate"
    assert lock["fit_max_iter"] == 1500
    assert lock["fit_objective_tolerance"] == 1e-4
    assert lock["fit_parameter_tolerance"] == 5e-5
    assert lock["fit_consecutive_convergence_passes"] == 2
    assert lock["eligible_spec_ids"] == list(followup.PRIMARY_SPEC_IDS)
    assert lock["evidence_paths"].keys() == lock["evidence_sha256"].keys()
    assert len(lock["evidence_paths"]) == 78
    for key, relative in lock["evidence_paths"].items():
        assert followup._sha256(tmp_path / relative) == lock["evidence_sha256"][key]

    blocked = replace(base, out_dir=tmp_path / "blocked")
    refit, fixed = _write_finalization_evidence(blocked, fixed_passed=False)
    decision = followup._finalize(blocked, refit, fixed)
    assert decision["passed"] is False
    assert decision["status"] == "blocked_numerical_followup"
    assert not (blocked.out_dir / "numerical_followup_lock.json").exists()
    assert not (blocked.out_dir / "numerical_followup_lock.sha256").exists()


def test_code_mutation_blocks_finalization_and_writes_no_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = replace(followup.load_context(CONFIG), out_dir=tmp_path / "run")
    context.out_dir.mkdir()
    (context.out_dir / "study_manifest.json").write_text(
        json.dumps({"schema_version": followup.RUNNER_SCHEMA, "status": "running"}),
        encoding="utf-8",
    )

    def reject(_context) -> dict[str, str]:
        raise followup.V4Error("frozen dependency changed")

    monkeypatch.setattr(followup, "_rehash_before_finalization", reject)
    with pytest.raises(followup.V4Error, match="dependency changed"):
        followup._finalize(context, {}, {})
    assert not (context.out_dir / "numerical_followup_decision.json").exists()
    assert not (context.out_dir / "numerical_followup_lock.json").exists()


def test_cli_is_plan_or_append_only_resume_without_destructive_fresh() -> None:
    parser = followup.build_parser()
    options = {option for action in parser._actions for option in action.option_strings}
    assert {"--plan-only", "--resume"} <= options
    assert "--fresh" not in options
    assert "rmtree" not in SCRIPT.read_text(encoding="utf-8")
