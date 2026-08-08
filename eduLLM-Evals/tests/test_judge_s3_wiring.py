"""Offline tests for the S3 wiring in the multi-benchmark judge driver.

These check that the printed GPU hand-off command derives its ``s3://`` output
prefix from a caller-supplied ``--s3-prefix`` (never a hardcoded bucket) and that
shard tagging is threaded consistently into both the local ``--output`` path and
the ``--s3-output-prefix``. Any AWS interaction is stubbed — nothing here touches
real S3.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, rel: str):
    """Load a scripts/ module by path (mirrors test_judge_verdict_ingestion)."""
    path = ROOT / rel
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


driver = _load_module("run_all_judge_grading", "scripts/run_all_judge_grading.py")

REAL_PREFIX = "s3://my-real-bucket/edu-tutor-grading"


def test_handoff_uses_supplied_prefix_no_placeholder() -> None:
    cmd = driver.handoff_command(
        benchmark="TutorBench",
        cases_path=ROOT / "runs" / "judge" / "TutorBench" / "cases.jsonl",
        judge="qwen",
        ingest_root=ROOT / "runs" / "judge" / "_verdicts_inbox",
        s3_prefix=REAL_PREFIX,
    )
    assert "YOUR-BUCKET" not in cmd
    assert (
        f"--s3-output-prefix {REAL_PREFIX}/TutorBench/canonical_r1" in cmd
    )
    # Unsharded output is not shard-tagged.
    assert ".shard" not in cmd


def test_handoff_strips_trailing_slash_on_prefix() -> None:
    cmd = driver.handoff_command(
        benchmark="Bridge",
        cases_path=ROOT / "runs" / "judge" / "Bridge" / "cases.jsonl",
        judge="qwen",
        ingest_root=ROOT / "runs" / "judge" / "_verdicts_inbox",
        s3_prefix=REAL_PREFIX + "/",
    )
    assert f"--s3-output-prefix {REAL_PREFIX}/Bridge/canonical_r1" in cmd
    assert "edu-tutor-grading//Bridge" not in cmd


def test_handoff_shard_tagging_is_consistent() -> None:
    shard_index, num_shards = 3, 8
    cases_name = driver.cases_filename(num_shards, shard_index)
    cmd = driver.handoff_command(
        benchmark="BiGGen",
        cases_path=ROOT / "runs" / "judge" / "BiGGen" / cases_name,
        judge="qwen",
        ingest_root=ROOT / "runs" / "judge" / "_verdicts_inbox",
        s3_prefix=REAL_PREFIX,
        num_shards=num_shards,
        shard_index=shard_index,
    )
    # The cases file, local output, and s3 prefix all carry the same shard tag.
    assert cases_name == "cases.shard3.jsonl"
    assert "cases.shard3.jsonl" in cmd
    assert "canonical_r1.shard3.jsonl" in cmd
    assert (
        f"--s3-output-prefix {REAL_PREFIX}/BiGGen/canonical_r1.shard3" in cmd
    )


def test_placeholder_default_still_flags_your_bucket() -> None:
    # The ultimate fallback is a clearly-labeled placeholder, not a real bucket.
    assert "YOUR-BUCKET" in driver.S3_PREFIX_PLACEHOLDER
    cmd = driver.handoff_command(
        benchmark="TutorBench",
        cases_path=ROOT / "cases.jsonl",
        judge="qwen",
        ingest_root=ROOT / "inbox",
        s3_prefix=driver.S3_PREFIX_PLACEHOLDER,
    )
    assert "YOUR-BUCKET" in cmd


def test_is_s3_uri() -> None:
    assert driver._is_s3_uri("s3://bucket/prefix")
    assert not driver._is_s3_uri("/local/path")
    assert not driver._is_s3_uri("runs/judge/inbox")


def test_sync_s3_to_local_shells_out_without_hitting_s3(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd, check=False, **kwargs):
        calls.append(list(cmd))

        class _R:
            returncode = 0

        return _R()

    monkeypatch.setattr(driver.subprocess, "run", fake_run)

    dest = tmp_path / "inbox"
    driver._sync_s3_to_local("s3://my-real-bucket/edu-tutor-grading/verdicts", dest)

    assert dest.is_dir()  # created before syncing
    assert len(calls) == 1
    cmd = calls[0]
    assert cmd[:3] == ["aws", "s3", "sync"]
    # Source gets a trailing slash; destination is the local dir.
    assert cmd[3] == "s3://my-real-bucket/edu-tutor-grading/verdicts/"
    assert cmd[4] == str(dest)
