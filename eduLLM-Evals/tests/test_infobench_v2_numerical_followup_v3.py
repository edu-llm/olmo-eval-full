"""Fail-closed tests for the frozen append-only numerical follow-up v3."""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_infobench_v2_numerical_followup_v3.py"
CONFIG = ROOT / "configs" / "infobench_v2_numerical_followup_v3.json"


def _load_module():
    spec = importlib.util.spec_from_file_location("infobench_v2_numerical_followup_v3", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


followup = _load_module()


def _validate_absent_or_immutable_terminal_output(output: Path) -> str:
    if not output.exists():
        return "absent_pre_run"
    manifest_path = output / "study_manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    status = manifest.get("status")
    assert manifest.get("schema_version") == followup.RUNNER_SCHEMA
    assert status in {"complete_pass", "blocked_numerical_followup"}
    assert manifest.get("selection_performed") is False
    assert manifest.get("cat_results_inspected") is False
    assert manifest.get("historical_fit_or_checkpoint_artifacts_reused") == 0

    decision_path = output / "numerical_followup_decision.json"
    assert decision_path.is_file()
    assert manifest.get("decision_sha256") == followup._sha256(decision_path)
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert decision.get("schema_version") == followup.RUNNER_SCHEMA
    assert decision.get("status") == status
    assert decision.get("passed") is (status == "complete_pass")

    lock_path = output / "numerical_followup_lock.json"
    companion_path = output / "numerical_followup_lock.sha256"
    if status == "complete_pass":
        assert lock_path.is_file() and companion_path.is_file()
        assert manifest.get("lock_sha256") == followup._sha256(lock_path)
    else:
        assert manifest.get("lock_sha256") is None
        assert not lock_path.exists() and not companion_path.exists()
    return str(status)


def test_v3_config_and_code_freeze_are_exact() -> None:
    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    followup._validate_config(raw)
    assert raw["schema_version"] == followup.CONFIG_SCHEMA
    assert raw["output_dir"].endswith("InFoBench_v2_numerical_followup_v3")
    assert raw["lock_path"].endswith(
        "InFoBench_v2_numerical_followup_v3/numerical_followup_lock.json"
    )
    assert raw["code_freeze"]["runner_sha256"] == followup._sha256(SCRIPT)
    assert (
        raw["scientific_design_source"]["frozen_contract_sha256"] == followup.FROZEN_CONTRACT_SHA256
    )


def test_v1_abort_and_terminal_v2_are_hash_preserved() -> None:
    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    source = followup._read_json(followup._resolve(raw["scientific_design_source"]["config"]))
    effective = followup.v2._materialize_effective_config(source)
    followup._verify_v1_history(raw, effective)
    evidence = followup._verify_v2_history(raw)
    assert evidence["output_file_count"] == 544
    assert evidence["promotable"] is False
    assert evidence["launched_runner_sha256"] != evidence["current_runner_sha256"]
    assert not (
        followup._resolve(raw["historical_evidence"]["v2_terminal"]["run_dir"])
        / "numerical_followup_lock.json"
    ).exists()


def test_v3_context_freezes_zero_reuse_and_all_72_new_fits() -> None:
    context = followup.load_context(CONFIG)
    signature = context.study_signature
    assert signature["schema_version"] == followup.RUNNER_SCHEMA
    assert signature["frozen_contract_sha256"] == followup.FROZEN_CONTRACT_SHA256
    assert signature["v1_fit_or_checkpoint_artifacts_reused"] == 0
    assert signature["v2_fit_or_checkpoint_artifacts_reused"] == 0
    assert signature["required_total_new_fits"] == 72
    assert len(signature["exact_specifications"]) == 6
    assert signature["comparison_schedule"] == [[81, 101], [101, 121]]
    schedule = followup.runtime_schedule(context)
    assert schedule["required_new_grid101_fits"] == 36
    assert schedule["required_new_grid121_fits"] == 36
    assert schedule["minimum_new_fits"] == schedule["maximum_new_fits"] == 72
    assert schedule["historical_v1_artifacts_reused"] == 0
    assert schedule["historical_v2_artifacts_reused"] == 0
    assert schedule["cat_runs"] == 0
    assert schedule["calibration_model_selection_runs"] == 0


def test_v3_rejects_any_frozen_runner_or_history_change() -> None:
    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    changed = json.loads(json.dumps(raw))
    changed["code_freeze"]["runner_sha256"] = "0" * 64
    followup._validate_config(changed)
    with pytest.raises(followup.v2.v1.FollowupError):
        followup._require_hash(SCRIPT, changed["code_freeze"]["runner_sha256"], "frozen v3 runner")
    changed = json.loads(json.dumps(raw))
    changed["historical_evidence"]["reuse_policy"]["v2_fit_or_checkpoint_artifacts_reused"] = 1
    with pytest.raises(followup.FollowupV3Error, match="zero-reuse"):
        followup._validate_config(changed)


def test_cli_has_no_fresh_and_v3_output_lifecycle_is_valid() -> None:
    parser = followup.build_parser()
    options = {option for action in parser._actions for option in action.option_strings}
    assert "--resume" in options
    assert "--fresh" not in options
    assert "rmtree" not in SCRIPT.read_text(encoding="utf-8")
    status = _validate_absent_or_immutable_terminal_output(
        ROOT / "runs" / "calibration" / followup.OUTPUT_LEAF
    )
    assert status in {"absent_pre_run", "complete_pass", "blocked_numerical_followup"}


def test_output_lifecycle_rejects_running_or_tampered_terminal_state(
    tmp_path: Path,
) -> None:
    output = tmp_path / followup.OUTPUT_LEAF
    output.mkdir()
    (output / "study_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": followup.RUNNER_SCHEMA,
                "status": "running",
                "selection_performed": False,
                "cat_results_inspected": False,
                "historical_fit_or_checkpoint_artifacts_reused": 0,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(AssertionError):
        _validate_absent_or_immutable_terminal_output(output)

    decision = {
        "schema_version": followup.RUNNER_SCHEMA,
        "status": "blocked_numerical_followup",
        "passed": False,
    }
    (output / "numerical_followup_decision.json").write_text(json.dumps(decision), encoding="utf-8")
    (output / "study_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": followup.RUNNER_SCHEMA,
                "status": "blocked_numerical_followup",
                "selection_performed": False,
                "cat_results_inspected": False,
                "historical_fit_or_checkpoint_artifacts_reused": 0,
                "decision_sha256": "0" * 64,
                "lock_sha256": None,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(AssertionError):
        _validate_absent_or_immutable_terminal_output(output)


def test_v3_prepare_is_append_only_and_terminal_resume_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = followup.load_context(CONFIG)
    isolated_root = tmp_path / "repo"
    output = isolated_root / "runs" / "calibration" / followup.OUTPUT_LEAF
    isolated = replace(context, out_dir=output)
    monkeypatch.setattr(followup, "ROOT", isolated_root)
    followup._prepare(isolated, resume=False)
    manifest_path = output / "study_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["historical_fit_or_checkpoint_artifacts_reused"] == 0
    followup._prepare(isolated, resume=True)
    manifest["status"] = "blocked_numerical_followup"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(followup.FollowupV3Error, match="terminal v3 evidence"):
        followup._prepare(isolated, resume=True)


def test_schema_constants_are_versioned_v3() -> None:
    assert followup.CONFIG_SCHEMA.endswith("-v3")
    assert followup.RUNNER_SCHEMA.endswith("-v3")
    assert followup.CHECKPOINT_SCHEMA.endswith("-v3")
    assert followup.LOCK_SCHEMA.endswith("-v3")
