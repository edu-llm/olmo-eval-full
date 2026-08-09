"""No-GPU tests for the ``olmo-eval run-modes`` command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from olmo_eval.cli.run_modes import run_modes
from olmo_eval.edullm.batch_resume import commit_checkpoint, sha256_file
from olmo_eval.edullm.mode import (
    BATCH_ARTIFACT_SCHEMA_VERSION,
    BATCH_PROGRESS_SCHEMA_VERSION,
    MODE_NAME,
)
from olmo_eval.inference.manager import InferenceManager
from olmo_eval.runners.mode_config import MODE_CONFIG_SCHEMA_VERSION

STATUS_FINGERPRINT = "a" * 64


def _write_standard_config(tmp_path: Path, *, run_id: str, output_dir: Path) -> Path:
    config_path = tmp_path / f"{run_id}.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": MODE_CONFIG_SCHEMA_VERSION,
                "run_id": run_id,
                "output_dir": str(output_dir),
                "harness": {
                    "name": "cli-check",
                    "provider": {"kind": "mock", "model": "fixture-candidate"},
                },
                "modes": [
                    {
                        "name": "standard_olmo",
                        "config": {"task_specs": ["arc:mc:olmo3base"]},
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return config_path


def _write_status_artifacts(
    output_dir: Path,
    *,
    run_id: str,
    fingerprint: str = STATUS_FINGERPRINT,
    checkpoint_generation: int = 7,
) -> dict[str, dict[str, object]]:
    mode_root = output_dir / "modes" / MODE_NAME
    mode_root.mkdir(parents=True)
    model_results = [
        {
            "candidate_id": "candidate-0000-success",
            "status": "succeeded",
            "output_dir": "models/candidate-0000-success/attempt-0001",
        },
        {
            "candidate_id": "candidate-0001-failure",
            "status": "failed",
            "output_dir": "models/candidate-0001-failure/attempt-0001",
        },
    ]
    attempts = [dict(row) for row in model_results]
    checkpoint = None
    for _ in range(checkpoint_generation):
        checkpoint = commit_checkpoint(
            mode_root,
            fingerprint_sha256=fingerprint,
            model_results=model_results,
            attempts=attempts,
        )
    counts = {
        "models_total": 3,
        "models_completed": 2,
        "models_succeeded": 1,
        "models_failed": 1,
        "models_cancelled": 0,
        "models_pending": 1,
    }
    model_results_path = mode_root / "model_results.jsonl"
    attempt_history_path = mode_root / "attempt_history.jsonl"
    model_results_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in model_results),
        encoding="utf-8",
    )
    attempt_history_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in attempts),
        encoding="utf-8",
    )
    artifacts: dict[str, dict[str, object]] = {
        "progress": {
            "schema_version": BATCH_PROGRESS_SCHEMA_VERSION,
            "run_id": run_id,
            "status": "running",
            "resume_contract_fingerprint": fingerprint,
            "checkpoint_generation": checkpoint_generation,
            **counts,
        },
        "batch_summary": {
            "schema_version": BATCH_ARTIFACT_SCHEMA_VERSION,
            "run_id": run_id,
            "status": "running",
            "resume_contract_fingerprint": fingerprint,
            "checkpoint_generation": checkpoint_generation,
            **counts,
            "attempts_committed": 2,
            "model_results_path": "model_results.jsonl",
            "attempt_history_path": "attempt_history.jsonl",
        },
        "batch_manifest": {
            "schema_version": BATCH_ARTIFACT_SCHEMA_VERSION,
            "mode": MODE_NAME,
            "execution": "precomputed_batch",
            "run_id": run_id,
            "status": "running",
            "resume_contract_fingerprint": fingerprint,
            "checkpoint_generation": checkpoint_generation,
            "progress": counts,
            "model_result_paths": [row["output_dir"] for row in model_results],
        },
    }
    (mode_root / "progress.json").write_text(json.dumps(artifacts["progress"]), encoding="utf-8")
    (mode_root / "batch_summary.json").write_text(
        json.dumps(artifacts["batch_summary"]), encoding="utf-8"
    )
    artifacts["batch_manifest"]["artifact_sha256"] = {
        "model_results.jsonl": sha256_file(model_results_path),
        "attempt_history.jsonl": sha256_file(attempt_history_path),
        "batch_summary.json": sha256_file(mode_root / "batch_summary.json"),
    }
    (mode_root / "manifest.json").write_text(
        json.dumps(artifacts["batch_manifest"]), encoding="utf-8"
    )
    artifacts["checkpoint"] = {
        "generation": checkpoint_generation,
        "checkpoint_sha256": None if checkpoint is None else checkpoint.checkpoint_sha256,
        "models_committed": 2,
        "attempts_committed": 2,
    }
    return artifacts


def test_check_runs_preflight_without_starting_inference_manager(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "mode-output"
    config_path = _write_standard_config(
        tmp_path,
        run_id="cli-check-fixture",
        output_dir=output_dir,
    )

    def forbidden_start(self: InferenceManager) -> dict[str, list[dict[str, object]]]:
        del self
        raise AssertionError("--check must not start an inference manager")

    monkeypatch.setattr(InferenceManager, "start", forbidden_start)

    result = CliRunner().invoke(run_modes, ["--config", str(config_path), "--check"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "status": "preflight_passed",
        "run_id": "cli-check-fixture",
        "selected_modes": ["standard_olmo"],
        "provider_names": ["candidate"],
        "available_gpu_ids": [],
        "judge_runtime": {},
    }
    assert not output_dir.exists()


def test_status_reads_persisted_artifacts_without_starting_orchestrator_or_providers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "mode-output"
    config_path = _write_standard_config(
        tmp_path,
        run_id="cli-status-fixture",
        output_dir=output_dir,
    )
    artifacts = _write_status_artifacts(
        output_dir,
        run_id="cli-status-fixture",
    )
    stale_run_manifest = {
        "run_id": "cli-status-fixture",
        "status": "failed",
        "completed_modes": [MODE_NAME],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(stale_run_manifest),
        encoding="utf-8",
    )

    def forbidden_orchestrator(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("--status must not construct the orchestrator")

    def forbidden_start(self: InferenceManager) -> dict[str, list[dict[str, object]]]:
        del self
        raise AssertionError("--status must not start an inference manager")

    monkeypatch.setattr(
        "olmo_eval.runners.mode_orchestrator.ModeRunOrchestrator",
        forbidden_orchestrator,
    )
    monkeypatch.setattr(InferenceManager, "start", forbidden_start)

    result = CliRunner().invoke(run_modes, ["--config", str(config_path), "--status"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "status": "status_available",
        "run_id": "cli-status-fixture",
        "output_dir": str(output_dir.resolve()),
        **artifacts,
    }


@pytest.mark.parametrize("artifact_name", ["progress", "batch_summary", "batch_manifest"])
def test_status_rejects_artifact_run_id_mismatch(
    tmp_path: Path,
    artifact_name: str,
) -> None:
    output_dir = tmp_path / "mode-output"
    config_path = _write_standard_config(
        tmp_path,
        run_id="configured-run",
        output_dir=output_dir,
    )
    artifacts = _write_status_artifacts(output_dir, run_id="configured-run")
    artifacts[artifact_name]["run_id"] = "different-run"
    filename = {
        "progress": "progress.json",
        "batch_summary": "batch_summary.json",
        "batch_manifest": "manifest.json",
    }[artifact_name]
    (output_dir / "modes" / MODE_NAME / filename).write_text(
        json.dumps(artifacts[artifact_name]),
        encoding="utf-8",
    )

    result = CliRunner().invoke(run_modes, ["--config", str(config_path), "--status"])

    assert result.exit_code == 1
    assert f"{artifact_name}.run_id 'different-run'" in result.output
    assert "does not match configured run_id 'configured-run'" in result.output


@pytest.mark.parametrize(
    ("field", "mismatched_value"),
    [
        ("resume_contract_fingerprint", "b" * 64),
        ("checkpoint_generation", 8),
    ],
)
def test_status_rejects_inconsistent_batch_view_identity(
    tmp_path: Path,
    field: str,
    mismatched_value: object,
) -> None:
    output_dir = tmp_path / "mode-output"
    config_path = _write_standard_config(
        tmp_path,
        run_id="configured-run",
        output_dir=output_dir,
    )
    artifacts = _write_status_artifacts(output_dir, run_id="configured-run")
    artifacts["batch_summary"][field] = mismatched_value
    (output_dir / "modes" / MODE_NAME / "batch_summary.json").write_text(
        json.dumps(artifacts["batch_summary"]),
        encoding="utf-8",
    )

    result = CliRunner().invoke(run_modes, ["--config", str(config_path), "--status"])

    assert result.exit_code == 1
    assert f"status artifacts disagree on {field}" in result.output


def test_status_rejects_inconsistent_batch_view_status(tmp_path: Path) -> None:
    output_dir = tmp_path / "mode-output"
    config_path = _write_standard_config(
        tmp_path,
        run_id="configured-run",
        output_dir=output_dir,
    )
    artifacts = _write_status_artifacts(output_dir, run_id="configured-run")
    artifacts["batch_summary"]["status"] = "succeeded"
    (output_dir / "modes" / MODE_NAME / "batch_summary.json").write_text(
        json.dumps(artifacts["batch_summary"]),
        encoding="utf-8",
    )

    result = CliRunner().invoke(run_modes, ["--config", str(config_path), "--status"])

    assert result.exit_code == 1
    assert "status artifacts disagree on status" in result.output


def test_status_rejects_unsupported_status_value(tmp_path: Path) -> None:
    output_dir = tmp_path / "mode-output"
    config_path = _write_standard_config(
        tmp_path,
        run_id="configured-run",
        output_dir=output_dir,
    )
    artifacts = _write_status_artifacts(output_dir, run_id="configured-run")
    mode_root = output_dir / "modes" / MODE_NAME
    for name, filename in (
        ("progress", "progress.json"),
        ("batch_summary", "batch_summary.json"),
        ("batch_manifest", "manifest.json"),
    ):
        artifacts[name]["status"] = "banana"
        (mode_root / filename).write_text(json.dumps(artifacts[name]), encoding="utf-8")

    result = CliRunner().invoke(run_modes, ["--config", str(config_path), "--status"])

    assert result.exit_code == 1
    assert "status is not a supported batch status" in result.output


def test_status_rejects_views_stale_behind_checkpoint(tmp_path: Path) -> None:
    output_dir = tmp_path / "mode-output"
    config_path = _write_standard_config(
        tmp_path,
        run_id="configured-run",
        output_dir=output_dir,
    )
    artifacts = _write_status_artifacts(output_dir, run_id="configured-run")
    mode_root = output_dir / "modes" / MODE_NAME
    for name, filename in (
        ("progress", "progress.json"),
        ("batch_summary", "batch_summary.json"),
        ("batch_manifest", "manifest.json"),
    ):
        artifacts[name]["checkpoint_generation"] = 6
        (mode_root / filename).write_text(json.dumps(artifacts[name]), encoding="utf-8")

    result = CliRunner().invoke(run_modes, ["--config", str(config_path), "--status"])

    assert result.exit_code == 1
    assert "views are stale relative to the authoritative checkpoint chain" in result.output


def test_status_rejects_same_generation_counts_that_contradict_checkpoint(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "mode-output"
    config_path = _write_standard_config(
        tmp_path,
        run_id="configured-run",
        output_dir=output_dir,
    )
    artifacts = _write_status_artifacts(output_dir, run_id="configured-run")
    mode_root = output_dir / "modes" / MODE_NAME
    artifacts["progress"]["models_completed"] = 1
    artifacts["progress"]["models_pending"] = 2
    artifacts["batch_summary"]["models_completed"] = 1
    artifacts["batch_summary"]["models_pending"] = 2
    manifest_progress = artifacts["batch_manifest"]["progress"]
    assert isinstance(manifest_progress, dict)
    manifest_progress["models_completed"] = 1
    manifest_progress["models_pending"] = 2
    for name, filename in (
        ("progress", "progress.json"),
        ("batch_summary", "batch_summary.json"),
        ("batch_manifest", "manifest.json"),
    ):
        (mode_root / filename).write_text(json.dumps(artifacts[name]), encoding="utf-8")

    result = CliRunner().invoke(run_modes, ["--config", str(config_path), "--status"])

    assert result.exit_code == 1
    assert "contradicts the authoritative checkpoint" in result.output


@pytest.mark.parametrize("filename", ["model_results.jsonl", "attempt_history.jsonl"])
def test_status_requires_committed_jsonl_views(tmp_path: Path, filename: str) -> None:
    output_dir = tmp_path / "mode-output"
    config_path = _write_standard_config(
        tmp_path,
        run_id="configured-run",
        output_dir=output_dir,
    )
    _write_status_artifacts(output_dir, run_id="configured-run")
    (output_dir / "modes" / MODE_NAME / filename).unlink()

    result = CliRunner().invoke(run_modes, ["--config", str(config_path), "--status"])

    assert result.exit_code == 1
    assert "JSONL artifact must be a regular file" in result.output


def test_status_rejects_malformed_or_checkpoint_divergent_jsonl_views(tmp_path: Path) -> None:
    output_dir = tmp_path / "mode-output"
    config_path = _write_standard_config(
        tmp_path,
        run_id="configured-run",
        output_dir=output_dir,
    )
    _write_status_artifacts(output_dir, run_id="configured-run")
    mode_root = output_dir / "modes" / MODE_NAME
    attempt_history_path = mode_root / "attempt_history.jsonl"
    original_attempt_history = attempt_history_path.read_text(encoding="utf-8")
    attempt_history_path.write_text("not-json\n", encoding="utf-8")

    malformed = CliRunner().invoke(run_modes, ["--config", str(config_path), "--status"])

    assert malformed.exit_code == 1
    assert "invalid JSONL artifact" in malformed.output

    attempt_history_path.write_text(original_attempt_history, encoding="utf-8")
    (mode_root / "model_results.jsonl").write_text(
        json.dumps(
            {
                "candidate_id": "candidate-9999-stale",
                "status": "succeeded",
                "output_dir": "models/candidate-9999-stale/attempt-0001",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    divergent = CliRunner().invoke(run_modes, ["--config", str(config_path), "--status"])

    assert divergent.exit_code == 1
    assert "model_results.jsonl contradicts the authoritative checkpoint" in divergent.output


def test_status_rejects_same_generation_terminal_status_that_contradicts_checkpoint(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "mode-output"
    config_path = _write_standard_config(
        tmp_path,
        run_id="configured-run",
        output_dir=output_dir,
    )
    artifacts = _write_status_artifacts(output_dir, run_id="configured-run")
    mode_root = output_dir / "modes" / MODE_NAME
    for name, filename in (
        ("progress", "progress.json"),
        ("batch_summary", "batch_summary.json"),
        ("batch_manifest", "manifest.json"),
    ):
        artifacts[name]["status"] = "succeeded"
        (mode_root / filename).write_text(json.dumps(artifacts[name]), encoding="utf-8")
    artifact_hashes = artifacts["batch_manifest"]["artifact_sha256"]
    assert isinstance(artifact_hashes, dict)
    artifact_hashes["batch_summary.json"] = sha256_file(mode_root / "batch_summary.json")
    (mode_root / "manifest.json").write_text(
        json.dumps(artifacts["batch_manifest"]), encoding="utf-8"
    )

    result = CliRunner().invoke(run_modes, ["--config", str(config_path), "--status"])

    assert result.exit_code == 1
    assert "succeeded status contradicts the authoritative checkpoint" in result.output


@pytest.mark.parametrize(
    "other_flags",
    [
        ("--check",),
        ("--resume",),
        ("--check", "--resume"),
    ],
)
def test_status_rejects_incompatible_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    other_flags: tuple[str, ...],
) -> None:
    output_dir = tmp_path / "mode-output"
    config_path = _write_standard_config(
        tmp_path,
        run_id="cli-status-flags-fixture",
        output_dir=output_dir,
    )

    def forbidden_orchestrator(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("invalid status flags must fail before orchestrator creation")

    monkeypatch.setattr(
        "olmo_eval.runners.mode_orchestrator.ModeRunOrchestrator",
        forbidden_orchestrator,
    )

    result = CliRunner().invoke(
        run_modes,
        ["--config", str(config_path), "--status", *other_flags],
    )

    assert result.exit_code == 1
    assert "--status cannot be combined with --check or --resume" in result.output
