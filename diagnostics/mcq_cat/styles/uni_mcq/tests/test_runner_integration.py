"""The style as reached through the frozen runner CLI.

These are the checks the branch plan lists before a PR: the style is discoverable by
name, ``--dry-run`` resolves without loading a model, and a full run writes a report.
They exercise ``runner.main`` rather than the style directly, so a mismatch between the
CLI's expectations and the style's signatures would surface here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from .... import runner
from .. import resolve
from .conftest import write_bank


def test_list_styles_reports_uni_mcq(capsys: pytest.CaptureFixture[str]) -> None:
    assert runner.main(["--list-styles"]) == 0
    assert "uni_mcq" in capsys.readouterr().out


def test_dry_run_resolves_without_loading_a_model(tmp_path: Path) -> None:
    """A dry run must not touch the checkpoint, so a bogus path is fine."""
    exit_code = runner.main(
        [
            "--cat-style",
            "uni_mcq",
            "--checkpoint",
            "s3://bucket/does-not-exist/step_1000",
            "--s3-out",
            str(tmp_path / "out"),
            "--benchmark",
            "arc_challenge",
            "--dry-run",
        ]
    )
    assert exit_code == 0
    assert not (tmp_path / "out").exists()


def test_missing_required_arguments_is_rejected() -> None:
    assert runner.main(["--cat-style", "uni_mcq"]) == 2


def test_unknown_style_fails_cleanly() -> None:
    assert (
        runner.main(
            [
                "--cat-style",
                "no_such_style",
                "--checkpoint",
                "x",
                "--s3-out",
                "y",
                "--dry-run",
            ]
        )
        == 1
    )


def test_full_run_writes_a_report_locally(monkeypatch, tmp_path: Path, toy_params) -> None:
    """End to end through the CLI with the model and checkpoint stubbed out."""
    banks = tmp_path / "banks"
    write_bank(banks, dataset="arc_challenge")
    monkeypatch.setattr(resolve, "CALIBRATED_DATASETS", banks)

    from ....common import inference, s3_io
    from .conftest import SimScorer

    monkeypatch.setattr(s3_io, "resolve_checkpoint", lambda *a, **k: tmp_path / "ckpt")
    monkeypatch.setattr(inference, "load_scoring_model", lambda *a, **k: SimScorer(0.5, toy_params))

    out_dir = tmp_path / "out"
    exit_code = runner.main(
        [
            "--cat-style",
            "uni_mcq",
            "--checkpoint",
            "s3://bucket/run/step_1000",
            "--s3-out",
            str(out_dir),
            "--benchmark",
            "arc_challenge",
            "--se-threshold",
            "0.3",
            "--max-items",
            "40",
        ]
    )

    assert exit_code == 0
    report_path = out_dir / "cat_report.json"
    assert report_path.is_file()

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["cat_style"] == "uni_mcq"
    assert payload["benchmark"] == "arc_challenge"
    assert payload["run"]["checkpoint"] == "s3://bucket/run/step_1000"

    metadata = payload["metadata"]
    assert metadata["bank_size"] == 5
    assert 0.0 <= metadata["pirt_accuracy"] <= 1.0
    assert isinstance(metadata["theta"], float)


def test_unsupported_dataset_fails_before_touching_the_checkpoint(
    monkeypatch, tmp_path: Path
) -> None:
    """The allowlist check must run before anything expensive happens."""
    monkeypatch.setattr(resolve, "CALIBRATED_DATASETS", tmp_path / "banks")

    from ....common import inference, s3_io

    def _boom(*args, **kwargs):
        raise AssertionError("the checkpoint must not be fetched for a bad dataset")

    monkeypatch.setattr(s3_io, "resolve_checkpoint", _boom)
    monkeypatch.setattr(inference, "load_scoring_model", _boom)

    assert (
        runner.main(
            [
                "--cat-style",
                "uni_mcq",
                "--checkpoint",
                "s3://bucket/run/step_1000",
                "--s3-out",
                str(tmp_path / "out"),
                "--benchmark",
                "piqa",
            ]
        )
        == 1
    )
