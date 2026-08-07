"""No-GPU tests for the ``olmo-eval run-modes`` command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from olmo_eval.cli.run_modes import run_modes
from olmo_eval.inference.manager import InferenceManager
from olmo_eval.runners.mode_config import MODE_CONFIG_SCHEMA_VERSION


def test_check_runs_preflight_without_starting_inference_manager(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "mode-output"
    config_path = tmp_path / "modes.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": MODE_CONFIG_SCHEMA_VERSION,
                "run_id": "cli-check-fixture",
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
