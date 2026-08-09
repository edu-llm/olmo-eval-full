"""Focused fail-closed tests for EduLLM batch resume primitives."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from olmo_eval.edullm.batch_resume import (
    CHECKPOINT_SCHEMA_VERSION,
    BatchCheckpointState,
    allocate_attempt_path,
    canonical_sha256,
    commit_checkpoint,
    exclusive_run_lock,
    load_checkpoint_chain,
    repair_checkpoint_pointer,
    sha256_file,
    validate_resume_contract,
    verify_hashed_artifacts,
    write_or_validate_resume_contract,
)


def _fingerprint(character: str = "a") -> str:
    return character * 64


def _model_row(index: int) -> dict[str, object]:
    return {
        "model_index": index,
        "candidate_id": f"candidate-{index:04d}",
        "model_id": f"model-{index}",
        "output_dir": f"models/candidate-{index:04d}/attempt-0001",
        "status": "succeeded",
        "manifest_sha256": _fingerprint(chr(ord("b") + index)),
    }


def _attempt_row(index: int) -> dict[str, object]:
    return {
        "model_index": index,
        "attempt": 1,
        "output_dir": f"models/candidate-{index:04d}/attempt-0001",
        "status": "committed",
    }


def test_contract_fingerprint_is_canonical_and_content_sensitive(tmp_path: Path) -> None:
    first = {"run_id": "run-1", "nested": {"b": 2, "a": 1}}
    reordered = {"nested": {"a": 1, "b": 2}, "run_id": "run-1"}
    changed = {"run_id": "run-2", "nested": {"a": 1, "b": 2}}

    assert canonical_sha256(first) == canonical_sha256(reordered)
    assert canonical_sha256(first) != canonical_sha256(changed)
    with pytest.raises(ValueError, match="non-finite"):
        canonical_sha256({"invalid": float("nan")})

    document = write_or_validate_resume_contract(tmp_path, first, resume=False)
    assert validate_resume_contract(tmp_path / "resume_contract.json", reordered) == document
    with pytest.raises(ValueError, match="fresh run refuses"):
        write_or_validate_resume_contract(tmp_path, first, resume=False)
    with pytest.raises(ValueError, match="fingerprint differs"):
        write_or_validate_resume_contract(tmp_path, changed, resume=True)


def test_resume_contract_refuses_missing_or_tampered_contract(tmp_path: Path) -> None:
    payload = {"run_id": "run-1", "response_batch_sha256": _fingerprint()}
    with pytest.raises(ValueError, match="requires an existing"):
        write_or_validate_resume_contract(tmp_path, payload, resume=True)

    write_or_validate_resume_contract(tmp_path, payload, resume=False)
    path = tmp_path / "resume_contract.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["payload"]["run_id"] = "tampered"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match its fingerprint"):
        validate_resume_contract(path, payload)


def test_exclusive_run_lock_refuses_a_concurrent_holder(tmp_path: Path) -> None:
    with exclusive_run_lock(tmp_path, "first") as lock_path:
        lock_record = json.loads(lock_path.read_text(encoding="utf-8"))
        assert lock_record["run_id"] == "first"
        with (
            pytest.raises(RuntimeError, match="already holds the run lock"),
            exclusive_run_lock(tmp_path, "second"),
        ):
            pytest.fail("a second holder must not enter the locked section")

    with exclusive_run_lock(tmp_path, "third"):
        pass


def test_run_lock_and_json_artifacts_refuse_symlinks(tmp_path: Path) -> None:
    outside_lock = tmp_path / "outside-lock.txt"
    outside_lock.write_text("do not truncate\n", encoding="utf-8")
    lock_root = tmp_path / "lock-root"
    lock_root.mkdir()
    (lock_root / ".run.lock").symlink_to(outside_lock)

    with (
        pytest.raises(RuntimeError, match="must not be a symlink"),
        exclusive_run_lock(lock_root, "unsafe"),
    ):
        pytest.fail("a symlinked lock must never be opened")
    assert outside_lock.read_text(encoding="utf-8") == "do not truncate\n"

    contract_root = tmp_path / "contract-root"
    contract_root.mkdir()
    outside_contract = tmp_path / "outside-contract.json"
    payload = {"run_id": "run-1"}
    document = write_or_validate_resume_contract(tmp_path / "source-root", payload, resume=False)
    outside_contract.write_text(json.dumps(document), encoding="utf-8")
    (contract_root / "resume_contract.json").symlink_to(outside_contract)
    with pytest.raises(ValueError, match="symlinked JSON artifact"):
        validate_resume_contract(contract_root / "resume_contract.json", payload)

    checkpoint_root = tmp_path / "checkpoint-root/checkpoints"
    checkpoint_root.mkdir(parents=True)
    (checkpoint_root / "checkpoint-000001.json").symlink_to(outside_contract)
    with pytest.raises(ValueError, match="checkpoint entry must be a file"):
        load_checkpoint_chain(tmp_path / "checkpoint-root", fingerprint_sha256=_fingerprint())


def test_checkpoint_chain_round_trip_and_hash_links(tmp_path: Path) -> None:
    fingerprint = _fingerprint()
    first = commit_checkpoint(
        tmp_path,
        fingerprint_sha256=fingerprint,
        model_results=[_model_row(0)],
        attempts=[_attempt_row(0)],
    )
    second = commit_checkpoint(
        tmp_path,
        fingerprint_sha256=fingerprint,
        model_results=[_model_row(0), _model_row(1)],
        attempts=[_attempt_row(0), _attempt_row(1)],
    )

    state = load_checkpoint_chain(tmp_path, fingerprint_sha256=fingerprint)

    assert first.generation == 1
    assert second.generation == 2
    assert state.generation == 2
    assert state.checkpoint_sha256 == sha256_file(tmp_path / "checkpoints/checkpoint-000002.json")
    assert [row["model_id"] for row in state.model_results] == ["model-0", "model-1"]
    assert state.pointer_stale is False
    checkpoint_two = json.loads(
        (tmp_path / "checkpoints/checkpoint-000002.json").read_text(encoding="utf-8")
    )
    assert checkpoint_two["schema_version"] == CHECKPOINT_SCHEMA_VERSION
    payload = {key: value for key, value in checkpoint_two.items() if key != "payload_sha256"}
    assert checkpoint_two["payload_sha256"] == canonical_sha256(payload)
    assert checkpoint_two["previous_checkpoint_sha256"] == first.checkpoint_sha256
    mutable_view = cast(dict[str, object], state.model_results[0])
    with pytest.raises(TypeError):
        mutable_view["status"] = "tampered"


def test_checkpoint_refuses_replacing_success_or_attempt_history(tmp_path: Path) -> None:
    fingerprint = _fingerprint()
    first_model = _model_row(0)
    first_attempt = _attempt_row(0)
    commit_checkpoint(
        tmp_path,
        fingerprint_sha256=fingerprint,
        model_results=[first_model],
        attempts=[first_attempt],
    )

    changed_model = {**first_model, "manifest_sha256": _fingerprint("f")}
    with pytest.raises(ValueError, match="cannot replace a committed success"):
        commit_checkpoint(
            tmp_path,
            fingerprint_sha256=fingerprint,
            model_results=[changed_model],
            attempts=[first_attempt],
        )

    changed_attempt = {**first_attempt, "status": "rewritten"}
    with pytest.raises(ValueError, match="immutable previous prefix"):
        commit_checkpoint(
            tmp_path,
            fingerprint_sha256=fingerprint,
            model_results=[first_model],
            attempts=[changed_attempt],
        )


def test_checkpoint_refuses_non_strict_json_rows(tmp_path: Path) -> None:
    invalid_row = cast(dict[str, object], {1: "non-string-key"})

    with pytest.raises(ValueError, match="non-string key"):
        commit_checkpoint(
            tmp_path,
            fingerprint_sha256=_fingerprint(),
            model_results=[invalid_row],
            attempts=[],
        )


def test_missing_or_stale_pointer_is_repairable_only_after_chain_validation(
    tmp_path: Path,
) -> None:
    fingerprint = _fingerprint()
    first = commit_checkpoint(
        tmp_path,
        fingerprint_sha256=fingerprint,
        model_results=[_model_row(0)],
        attempts=[_attempt_row(0)],
    )
    commit_checkpoint(
        tmp_path,
        fingerprint_sha256=fingerprint,
        model_results=[_model_row(0), _model_row(1)],
        attempts=[_attempt_row(0), _attempt_row(1)],
    )
    pointer_path = tmp_path / "checkpoints/latest.json"
    pointer_path.write_text(
        json.dumps(
            {
                "schema_version": "edullm-adaptive-batch-checkpoint-pointer-v1",
                "generation": 1,
                "checkpoint": "checkpoints/checkpoint-000001.json",
                "sha256": first.checkpoint_sha256,
            }
        ),
        encoding="utf-8",
    )

    stale = load_checkpoint_chain(tmp_path, fingerprint_sha256=fingerprint)
    assert stale.generation == 2
    assert stale.pointer_stale is True

    repair_checkpoint_pointer(tmp_path, stale)

    repaired = load_checkpoint_chain(tmp_path, fingerprint_sha256=fingerprint)
    assert repaired.pointer_stale is False
    assert json.loads(pointer_path.read_text(encoding="utf-8"))["generation"] == 2


def test_interrupted_atomic_checkpoint_temp_is_preserved_but_not_committed(
    tmp_path: Path,
) -> None:
    fingerprint = _fingerprint()
    committed = commit_checkpoint(
        tmp_path,
        fingerprint_sha256=fingerprint,
        model_results=[_model_row(0)],
        attempts=[_attempt_row(0)],
    )
    temporary = tmp_path / "checkpoints" / f".checkpoint-000002.json.{('d' * 32)}.tmp"
    temporary.write_text("partial", encoding="utf-8")

    state = load_checkpoint_chain(tmp_path, fingerprint_sha256=fingerprint)

    assert state.generation == committed.generation == 1
    assert temporary.read_text(encoding="utf-8") == "partial"


def test_checkpoint_chain_refuses_wrong_fingerprint_corruption_and_extra_entries(
    tmp_path: Path,
) -> None:
    fingerprint = _fingerprint()
    commit_checkpoint(
        tmp_path,
        fingerprint_sha256=fingerprint,
        model_results=[_model_row(0)],
        attempts=[_attempt_row(0)],
    )
    commit_checkpoint(
        tmp_path,
        fingerprint_sha256=fingerprint,
        model_results=[_model_row(0), _model_row(1)],
        attempts=[_attempt_row(0), _attempt_row(1)],
    )

    with pytest.raises(ValueError, match="different run fingerprint"):
        load_checkpoint_chain(tmp_path, fingerprint_sha256=_fingerprint("f"))

    first_path = tmp_path / "checkpoints/checkpoint-000001.json"
    first = json.loads(first_path.read_text(encoding="utf-8"))
    first["model_results"][0]["status"] = "tampered"
    first_path.write_text(json.dumps(first), encoding="utf-8")
    with pytest.raises(ValueError, match="payload digest does not match"):
        load_checkpoint_chain(tmp_path, fingerprint_sha256=fingerprint)

    payload = {key: value for key, value in first.items() if key != "payload_sha256"}
    first["payload_sha256"] = canonical_sha256(payload)
    first_path.write_text(json.dumps(first), encoding="utf-8")
    with pytest.raises(ValueError, match="breaks the immutable hash chain"):
        load_checkpoint_chain(tmp_path, fingerprint_sha256=fingerprint)

    (tmp_path / "checkpoints/orphan.tmp").write_text("partial", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected file in checkpoint directory"):
        load_checkpoint_chain(tmp_path, fingerprint_sha256=fingerprint)


def test_pointer_refuses_wrong_hash_or_pointer_without_checkpoints(tmp_path: Path) -> None:
    fingerprint = _fingerprint()
    commit_checkpoint(
        tmp_path,
        fingerprint_sha256=fingerprint,
        model_results=[_model_row(0)],
        attempts=[_attempt_row(0)],
    )
    pointer_path = tmp_path / "checkpoints/latest.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    pointer["sha256"] = _fingerprint("f")
    pointer_path.write_text(json.dumps(pointer), encoding="utf-8")
    with pytest.raises(ValueError, match="pointer SHA-256 does not match"):
        load_checkpoint_chain(tmp_path, fingerprint_sha256=fingerprint)

    orphan_root = tmp_path / "orphan"
    (orphan_root / "checkpoints").mkdir(parents=True)
    (orphan_root / "checkpoints/latest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="pointer exists without"):
        load_checkpoint_chain(orphan_root, fingerprint_sha256=fingerprint)


def test_orphan_attempt_is_preserved_and_next_attempt_is_never_reused(tmp_path: Path) -> None:
    candidate_root = tmp_path / "models/candidate-0000-safe"
    orphan = candidate_root / "attempt-0001"
    orphan.mkdir(parents=True)
    (orphan / "partial.json").write_text("{}", encoding="utf-8")

    attempt, relative = allocate_attempt_path(tmp_path, "candidate-0000-safe")

    assert attempt == 2
    assert relative == Path("models/candidate-0000-safe/attempt-0002")
    assert orphan.is_dir()
    assert not (tmp_path / relative).exists()

    (candidate_root / "unexpected.txt").write_text("corrupt", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected candidate attempt entry"):
        allocate_attempt_path(tmp_path, "candidate-0000-safe")
    with pytest.raises(ValueError, match="unsafe candidate_id"):
        allocate_attempt_path(tmp_path, "../escape")


def test_attempt_allocator_refuses_a_symlinked_candidate_root(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    models = tmp_path / "models"
    models.mkdir()
    (models / "candidate-0000-safe").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="candidate attempt root must be a directory"):
        allocate_attempt_path(tmp_path, "candidate-0000-safe")


def test_attempt_allocator_refuses_symlinked_models_root_and_supports_five_digits(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    linked_root = tmp_path / "linked"
    linked_root.mkdir()
    (linked_root / "models").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="candidate models root must be a directory"):
        allocate_attempt_path(linked_root, "candidate-0000-safe")

    candidate_root = tmp_path / "models/candidate-0000-safe"
    (candidate_root / "attempt-10000").mkdir(parents=True)
    attempt, relative = allocate_attempt_path(tmp_path, "candidate-0000-safe")
    assert attempt == 10001
    assert relative.name == "attempt-10001"


def test_candidate_artifact_verification_refuses_corruption_and_escape(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    artifact = candidate / "cat_result.json"
    artifact.write_text('{"status":"succeeded"}\n', encoding="utf-8")
    hashes = {"cat_result.json": sha256_file(artifact)}

    verify_hashed_artifacts(candidate, hashes)

    artifact.write_text('{"status":"tampered"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="artifact SHA-256 mismatch"):
        verify_hashed_artifacts(candidate, hashes)
    with pytest.raises(ValueError, match="unsafe relative artifact path"):
        verify_hashed_artifacts(candidate, {"../outside.json": _fingerprint()})


def test_repair_is_a_noop_without_a_committed_generation(tmp_path: Path) -> None:
    repair_checkpoint_pointer(tmp_path, BatchCheckpointState(0, None, (), ()))

    assert not (tmp_path / "checkpoints/latest.json").exists()
