"""Mode-level tests for crash-safe EduLLM batch resume behavior."""

from __future__ import annotations

import asyncio
import csv
import io
import json
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from olmo_eval.edullm.batch_resume import build_resume_contract, sha256_file
from olmo_eval.edullm.mode import EduLLMAdaptiveMode, build_batch_resume_contract_payload
from olmo_eval.runners.modes import ModeStatus
from tests.edullm.test_edullm_mode import (
    _batch_judge_provider,
    _config,
    _context,
    _read_jsonl,
    _use_precomputed_response_batch,
    _write_bank,
)


def _batch_config(
    tmp_path: Path,
    model_ids: tuple[str, ...] = ("a-first", "z-second"),
) -> dict[str, Any]:
    config = _config(
        _write_bank(tmp_path / "bank", scenario_ids=("mid",)),
        max_scenarios=1,
    )
    rows = [
        {
            "model_id": model_id,
            "model_family": f"family-{model_id}",
            "model_revision": f"revision-{model_id}",
            "scenario_id": "mid",
            "response": f"{model_id} response",
        }
        for model_id in model_ids
    ]
    _use_precomputed_response_batch(config, tmp_path / "batch.jsonl", rows)
    return config


def _resume_context(tmp_path: Path, judge: Any, fingerprint: str) -> Any:
    context, _ = _context(tmp_path / "run", judge)
    return replace(
        context,
        resume=True,
        resume_contract_fingerprint=fingerprint,
    )


def _fingerprint(output: Path) -> str:
    summary = json.loads((output / "batch_summary.json").read_text(encoding="utf-8"))
    return str(summary["resume_contract_fingerprint"])


def _assert_views_reconcile(output: Path) -> None:
    summary = json.loads((output / "batch_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    progress = json.loads((output / "progress.json").read_text(encoding="utf-8"))
    report = json.loads((output / "batch_results.json").read_text(encoding="utf-8"))
    pointer = json.loads((output / "checkpoints/latest.json").read_text(encoding="utf-8"))
    checkpoint = json.loads((output / pointer["checkpoint"]).read_text(encoding="utf-8"))
    model_rows = _read_jsonl(output / "model_results.jsonl")
    attempts = _read_jsonl(output / "attempt_history.jsonl")

    generation = pointer["generation"]
    assert summary["checkpoint_generation"] == generation
    assert manifest["checkpoint_generation"] == generation
    assert progress["checkpoint_generation"] == generation
    assert checkpoint["generation"] == generation
    assert pointer["sha256"] == sha256_file(output / pointer["checkpoint"])
    assert checkpoint["model_results"] == model_rows
    assert checkpoint["attempts"] == attempts
    assert summary["models_completed"] == report["completion"]["models_completed"]
    assert summary["models_succeeded"] == report["completion"]["models_succeeded"]
    assert progress["models_completed"] == summary["models_completed"]
    assert progress["status"] == summary["status"] == manifest["status"] == "succeeded"
    assert progress["phase"] == "batch_completed"
    for name, digest in manifest["artifact_sha256"].items():
        assert sha256_file(output / name) == digest

    csv_rows = list(
        csv.DictReader(io.StringIO((output / "batch_results.csv").read_text(encoding="utf-8")))
    )
    assert len(csv_rows) == summary["models_completed"]
    assert {row["model_id"] for row in csv_rows} == {row["model_id"] for row in model_rows}


def test_batch_resume_contract_payload_binds_top_level_metadata(tmp_path: Path) -> None:
    config = _batch_config(tmp_path, model_ids=("only-model",))
    common = {
        "run_id": "metadata-fingerprint-fixture",
        "harness_config": {"name": "fixture-harness"},
        "runtime_contract": {"vllm_version": "fixture-version"},
    }

    first = build_batch_resume_contract_payload(
        config,
        metadata={"campaign": "first"},
        **common,
    )
    second = build_batch_resume_contract_payload(
        config,
        metadata={"campaign": "second"},
        **common,
    )

    assert first["metadata"] == {"campaign": "first"}
    assert (
        build_resume_contract(first)["fingerprint_sha256"]
        != build_resume_contract(second)["fingerprint_sha256"]
    )


def test_interrupted_attempt_is_preserved_and_success_is_reused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _batch_config(tmp_path)
    first_judge = _batch_judge_provider()
    first_context, _ = _context(tmp_path / "run", first_judge)
    output = tmp_path / "run/modes/edullm_adaptive"
    original = EduLLMAdaptiveMode._run_candidate
    interrupt = True

    async def interrupt_second(
        self: EduLLMAdaptiveMode,
        context: Any,
        prepared: Any,
        candidate_output: Path,
        progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> Any:
        if interrupt and prepared.tutor_identity.model == "z-second":
            candidate_output.mkdir(parents=True)
            (candidate_output / "orphaned_partial.txt").write_text(
                "preserve me\n",
                encoding="utf-8",
            )
            raise asyncio.CancelledError
        return await original(
            self,
            context,
            prepared,
            candidate_output,
            progress_callback=progress_callback,
        )

    monkeypatch.setattr(EduLLMAdaptiveMode, "_run_candidate", interrupt_second)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(EduLLMAdaptiveMode().run(first_context, config, output))

    committed_before = _read_jsonl(output / "model_results.jsonl")
    assert [(row["model_id"], row["status"]) for row in committed_before] == [
        ("a-first", "succeeded")
    ]
    first_manifest_hash = committed_before[0]["manifest_sha256"]
    assert len(first_judge.requests) == 2
    fingerprint = _fingerprint(output)

    interrupt = False
    resume_judge = _batch_judge_provider()
    result = asyncio.run(
        EduLLMAdaptiveMode().run(
            _resume_context(tmp_path, resume_judge, fingerprint),
            config,
            output,
        )
    )

    assert result.status == ModeStatus.SUCCEEDED
    assert len(resume_judge.requests) == 2
    model_rows = _read_jsonl(output / "model_results.jsonl")
    by_model = {row["model_id"]: row for row in model_rows}
    assert by_model["a-first"]["manifest_sha256"] == first_manifest_hash
    assert Path(by_model["a-first"]["output_dir"]).name == "attempt-0001"
    assert Path(by_model["z-second"]["output_dir"]).name == "attempt-0002"
    orphan = output / Path(by_model["z-second"]["output_dir"]).parent / "attempt-0001"
    assert (orphan / "orphaned_partial.txt").read_text(encoding="utf-8") == "preserve me\n"
    _assert_views_reconcile(output)


def test_failed_model_is_requeued_and_retried_while_success_is_reused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _batch_config(tmp_path, ("a-broken", "z-good"))
    first_judge = _batch_judge_provider()
    first_context, _ = _context(tmp_path / "run", first_judge)
    output = tmp_path / "run/modes/edullm_adaptive"
    original = EduLLMAdaptiveMode._run_candidate
    broken_behavior = "fail"

    async def fail_once(
        self: EduLLMAdaptiveMode,
        context: Any,
        prepared: Any,
        candidate_output: Path,
        progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> Any:
        if broken_behavior == "fail" and prepared.tutor_identity.model == "a-broken":
            candidate_output.mkdir(parents=True)
            (candidate_output / "orphaned_partial.txt").write_text("partial\n", encoding="utf-8")
            raise RuntimeError("fixture wrapper failure")
        if broken_behavior == "cancel" and prepared.tutor_identity.model == "a-broken":
            candidate_output.mkdir(parents=True)
            (candidate_output / "cancelled_partial.txt").write_text("cancelled\n", encoding="utf-8")
            raise asyncio.CancelledError
        return await original(
            self,
            context,
            prepared,
            candidate_output,
            progress_callback=progress_callback,
        )

    monkeypatch.setattr(EduLLMAdaptiveMode, "_run_candidate", fail_once)

    first_result = asyncio.run(EduLLMAdaptiveMode().run(first_context, config, output))
    assert first_result.status == ModeStatus.FAILED
    first_rows = _read_jsonl(output / "model_results.jsonl")
    assert [(row["model_id"], row["status"]) for row in first_rows] == [
        ("a-broken", "failed"),
        ("z-good", "succeeded"),
    ]
    good_hash = first_rows[1]["manifest_sha256"]

    # Interrupt after the failed row has been requeued. Its committed failure
    # history remains authoritative while the cancelled retry stays orphaned.
    broken_behavior = "cancel"
    interrupted_judge = _batch_judge_provider()
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            EduLLMAdaptiveMode().run(
                _resume_context(tmp_path, interrupted_judge, _fingerprint(output)),
                config,
                output,
            )
        )
    assert interrupted_judge.requests == []
    assert [row["model_id"] for row in _read_jsonl(output / "model_results.jsonl")] == ["z-good"]

    broken_behavior = "succeed"
    resume_judge = _batch_judge_provider()
    resumed = asyncio.run(
        EduLLMAdaptiveMode().run(
            _resume_context(tmp_path, resume_judge, _fingerprint(output)),
            config,
            output,
        )
    )

    assert resumed.status == ModeStatus.SUCCEEDED
    assert len(resume_judge.requests) == 2
    rows = _read_jsonl(output / "model_results.jsonl")
    by_model = {row["model_id"]: row for row in rows}
    assert by_model["z-good"]["manifest_sha256"] == good_hash
    assert Path(by_model["a-broken"]["output_dir"]).name == "attempt-0004"
    broken_attempts = [
        row
        for row in _read_jsonl(output / "attempt_history.jsonl")
        if row["model_id"] == "a-broken"
    ]
    assert [(row["attempt_number"], row["status"]) for row in broken_attempts] == [
        (2, "failed"),
        (4, "succeeded"),
    ]
    _assert_views_reconcile(output)


def test_tampered_successful_child_artifact_refuses_resume_before_qwen(
    tmp_path: Path,
) -> None:
    config = _batch_config(tmp_path, ("only-model",))
    first_judge = _batch_judge_provider()
    first_context, _ = _context(tmp_path / "run", first_judge)
    output = tmp_path / "run/modes/edullm_adaptive"
    result = asyncio.run(EduLLMAdaptiveMode().run(first_context, config, output))
    assert result.status == ModeStatus.SUCCEEDED

    row = _read_jsonl(output / "model_results.jsonl")[0]
    cat_result = output / row["output_dir"] / "cat_result.json"
    cat_result.write_text('{"status":"tampered"}\n', encoding="utf-8")
    resume_judge = _batch_judge_provider()

    with pytest.raises(ValueError, match="artifact SHA-256 mismatch"):
        asyncio.run(
            EduLLMAdaptiveMode().run(
                _resume_context(tmp_path, resume_judge, _fingerprint(output)),
                config,
                output,
            )
        )

    assert resume_judge.requests == []
