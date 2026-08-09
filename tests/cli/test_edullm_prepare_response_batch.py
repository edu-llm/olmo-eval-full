"""CLI tests for preparing strict multi-model EduLLM response batches."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from click.testing import CliRunner

from olmo_eval.cli import main


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _inputs(tmp_path: Path, *, rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    source = tmp_path / "responses.jsonl"
    _write_jsonl(source, rows)
    manifest = tmp_path / "sources.jsonl"
    _write_jsonl(
        manifest,
        [
            {
                "format": "single-jsonl-v1",
                "path": source.name,
                "model_id": "model/a",
                "model_family": "family-a",
                "model_revision": "revision-a",
            }
        ],
    )
    scenarios = tmp_path / "scenarios.jsonl"
    _write_jsonl(scenarios, [{"scenario_id": "s1"}, {"scenario_id": "s2"}])
    return manifest, scenarios


def test_cli_group_is_registered_and_check_writes_nothing(tmp_path: Path) -> None:
    manifest, scenarios = _inputs(
        tmp_path,
        rows=[
            {"scenario_id": "s1", "response": "one"},
            {"scenario_id": "s2", "response": "two"},
        ],
    )
    output = tmp_path / "prepared.jsonl"
    report = tmp_path / "custom-report.json"

    result = CliRunner().invoke(
        main,
        [
            "edullm",
            "prepare-response-batch",
            "--source-manifest",
            str(manifest),
            "--fitted-scenarios",
            str(scenarios),
            "--output",
            str(output),
            "--report",
            str(report),
            "--check",
        ],
    )

    assert result.exit_code == 0, result.output
    summary = json.loads(result.output)
    assert summary["status"] == "checked"
    assert summary["counts"] == {
        "models": 1,
        "rows": 2,
        "scenarios_per_model": 2,
        "blank_responses": 0,
        "synthesized_blank_responses": 0,
    }
    assert summary["response_source"]["path"] == str(output.resolve())
    assert len(summary["response_source"]["sha256"]) == 64
    assert summary["report_sha256"] is None
    assert not output.exists()
    assert not report.exists()


def test_cli_missing_defaults_to_error_then_explicit_blank_writes(tmp_path: Path) -> None:
    manifest, scenarios = _inputs(
        tmp_path,
        rows=[{"scenario_id": "s1", "response": "one"}],
    )
    output = tmp_path / "prepared.jsonl"
    args = [
        "edullm",
        "prepare-response-batch",
        "--source-manifest",
        str(manifest),
        "--fitted-scenarios",
        str(scenarios),
        "--output",
        str(output),
    ]

    rejected = CliRunner().invoke(main, args)

    assert rejected.exit_code != 0
    assert "must exactly cover the fitted bank" in rejected.output
    assert not output.exists()

    accepted = CliRunner().invoke(main, [*args, "--missing", "blank"])

    assert accepted.exit_code == 0, accepted.output
    summary = json.loads(accepted.output)
    report = tmp_path / "prepared.jsonl.report.json"
    assert summary["status"] == "written"
    assert summary["counts"]["synthesized_blank_responses"] == 1
    assert output.is_file()
    assert report.is_file()
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert rows[1]["response"] == ""
    assert rows[1]["metadata"]["preparation_status"] == "missing_source_row"
    assert (
        json.loads(report.read_text())["output"]["sha256"] == summary["response_source"]["sha256"]
    )


def test_cli_refuses_to_replace_an_input_even_with_overwrite(tmp_path: Path) -> None:
    manifest, scenarios = _inputs(
        tmp_path,
        rows=[
            {"scenario_id": "s1", "response": "one"},
            {"scenario_id": "s2", "response": "two"},
        ],
    )
    source = tmp_path / "responses.jsonl"
    before = source.read_bytes()

    result = CliRunner().invoke(
        main,
        [
            "edullm",
            "prepare-response-batch",
            "--source-manifest",
            str(manifest),
            "--fitted-scenarios",
            str(scenarios),
            "--output",
            str(source),
            "--overwrite",
        ],
    )

    assert result.exit_code != 0
    assert "refusing to overwrite a source input" in result.output
    assert source.read_bytes() == before
