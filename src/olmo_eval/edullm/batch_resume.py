"""Crash-safe, candidate-level resume primitives for EduLLM batch runs.

The immutable checkpoint chain is the only resume authority.  Public summaries,
reports, and ``progress.json`` are derived views and may be regenerated after an
interruption.  A candidate is committed only after its own manifest and artifacts
have been written; partially written attempts are preserved but never reused.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import stat
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

RESUME_CONTRACT_SCHEMA_VERSION = "edullm-adaptive-resume-contract-v1"
CHECKPOINT_SCHEMA_VERSION = "edullm-adaptive-batch-checkpoint-v2"
CHECKPOINT_POINTER_SCHEMA_VERSION = "edullm-adaptive-batch-checkpoint-pointer-v1"

_CHECKPOINT_NAME = re.compile(r"^checkpoint-([0-9]{6,})\.json$")
_CHECKPOINT_TEMP_NAME = re.compile(
    r"^\.(?:checkpoint-[0-9]{6,}\.json|latest\.json)\.[0-9a-f]{32}\.tmp$"
)
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True, slots=True)
class BatchCheckpointState:
    """The newest fully validated immutable checkpoint generation."""

    generation: int
    checkpoint_sha256: str | None
    model_results: tuple[Mapping[str, Any], ...]
    attempts: tuple[Mapping[str, Any], ...]
    pointer_stale: bool = False


def _strict_json(value: Any, *, path: str = "value") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} contains a non-string key")
            _strict_json(item, path=f"{path}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _strict_json(item, path=f"{path}[{index}]")
        return
    raise ValueError(f"{path} contains unsupported value {type(value).__name__}")


def _canonical_bytes(value: Any) -> bytes:
    _strict_json(value)
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    """Hash one strict-JSON value using the repository's canonical encoding."""

    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    try:
        directory_fd = os.open(path.parent, os.O_RDONLY)
    except OSError:  # pragma: no cover - platform/filesystem dependent.
        return
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _strict_json(value, path=str(path))
    payload = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False).encode() + b"\n"
    _atomic_write(path, payload)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _read_object(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError(f"refusing to read a symlinked JSON artifact: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must contain one object: {path}")
    _strict_json(value, path=str(path))
    return value


def read_strict_json_object(path: Path) -> dict[str, Any]:
    """Read one strict JSON object, rejecting duplicate keys and non-finite values."""

    return _read_object(path)


def read_strict_jsonl_objects(path: Path) -> tuple[Mapping[str, Any], ...]:
    """Read a strict JSONL object stream without accepting partial/corrupt rows."""

    if path.is_symlink() or not path.is_file():
        raise ValueError(f"JSONL artifact must be a regular file: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"invalid JSONL artifact {path}: {exc}") from exc
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line:
            raise ValueError(f"invalid JSONL artifact {path}: blank line {line_number}")
        try:
            value = json.loads(
                line,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_constant,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"invalid JSONL artifact {path} line {line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"JSONL artifact row must be an object: {path}:{line_number}")
        _strict_json(value, path=f"{path}:{line_number}")
        rows.append(MappingProxyType(value))
    return tuple(rows)


def _exact_keys(value: Mapping[str, Any], fields: set[str], *, path: str) -> None:
    missing = sorted(fields - set(value))
    extra = sorted(set(value) - fields)
    if missing or extra:
        raise ValueError(f"{path} fields differ; missing={missing}, extra={extra}")


def _sha(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in _HEX for char in value):
        raise ValueError(f"{path} must be a lowercase SHA-256 digest")
    return value


def build_resume_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return the immutable contract document for one exact batch run."""

    normalized = dict(payload)
    _strict_json(normalized, path="resume contract payload")
    return {
        "schema_version": RESUME_CONTRACT_SCHEMA_VERSION,
        "fingerprint_sha256": canonical_sha256(normalized),
        "payload": normalized,
    }


def validate_resume_contract(path: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an existing contract byte-independently against the expected payload."""

    observed = _read_object(path)
    _exact_keys(
        observed,
        {"schema_version", "fingerprint_sha256", "payload"},
        path="resume contract",
    )
    if observed["schema_version"] != RESUME_CONTRACT_SCHEMA_VERSION:
        raise ValueError("unsupported resume contract schema_version")
    expected_document = build_resume_contract(expected)
    observed_fingerprint = _sha(
        observed["fingerprint_sha256"], path="resume contract fingerprint_sha256"
    )
    if canonical_sha256(observed["payload"]) != observed_fingerprint:
        raise ValueError("stored resume contract payload does not match its fingerprint")
    if observed_fingerprint != expected_document["fingerprint_sha256"]:
        raise ValueError(
            "resume contract fingerprint differs from the requested run; "
            "use the original config, bank, response batch, and software version"
        )
    if observed["payload"] != expected_document["payload"]:
        raise ValueError("resume contract payload differs despite matching fingerprint")
    return observed


def write_or_validate_resume_contract(
    output_root: Path,
    payload: Mapping[str, Any],
    *,
    resume: bool,
) -> dict[str, Any]:
    """Create a fresh contract or validate the contract required for resume."""

    path = output_root / "resume_contract.json"
    if resume:
        if not path.is_file():
            raise ValueError("resume requires an existing resume_contract.json")
        return validate_resume_contract(path, payload)
    if path.exists():
        raise ValueError("fresh run refuses to overwrite an existing resume contract")
    document = build_resume_contract(payload)
    _atomic_json(path, document)
    return document


@contextmanager
def exclusive_run_lock(output_root: Path, run_id: str) -> Iterator[Path]:
    """Hold a non-blocking process lock for the complete mutable run lifecycle."""

    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / ".run.lock"
    if path.is_symlink():
        raise RuntimeError(f"run lock path must not be a symlink: {path}")
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise RuntimeError(f"could not safely open the run lock: {path}: {exc}") from exc
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise RuntimeError(f"run lock path must be a regular file: {path}")
    stream = os.fdopen(descriptor, "r+", encoding="utf-8")
    try:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another process already holds the run lock: {path}") from exc
        stream.seek(0)
        stream.truncate()
        stream.write(json.dumps({"run_id": run_id, "pid": os.getpid()}) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
        yield path
    finally:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()


def _checkpoint_document(
    *,
    generation: int,
    fingerprint_sha256: str,
    previous_checkpoint_sha256: str | None,
    model_results: Sequence[Mapping[str, Any]],
    attempts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "generation": generation,
        "fingerprint_sha256": _sha(fingerprint_sha256, path="checkpoint fingerprint"),
        "previous_checkpoint_sha256": previous_checkpoint_sha256,
        "model_results": [dict(row) for row in model_results],
        "attempts": [dict(row) for row in attempts],
    }
    return {**payload, "payload_sha256": canonical_sha256(payload)}


def _pointer_document(generation: int, relative_path: str, sha256: str) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_POINTER_SCHEMA_VERSION,
        "generation": generation,
        "checkpoint": relative_path,
        "sha256": sha256,
    }


def load_checkpoint_chain(
    mode_root: Path,
    *,
    fingerprint_sha256: str,
) -> BatchCheckpointState:
    """Validate the complete immutable chain and return its newest committed state."""

    checkpoints_dir = mode_root / "checkpoints"
    if not checkpoints_dir.exists():
        return BatchCheckpointState(0, None, (), ())
    if checkpoints_dir.is_symlink() or not checkpoints_dir.is_dir():
        raise ValueError("batch checkpoints path must be a directory")
    files: list[tuple[int, Path]] = []
    for path in checkpoints_dir.iterdir():
        if path.name == "latest.json":
            continue
        if _CHECKPOINT_TEMP_NAME.fullmatch(path.name):
            # A killed atomic write may leave an uncommitted temporary file.
            # It is not part of the immutable chain and is preserved for audit.
            continue
        match = _CHECKPOINT_NAME.fullmatch(path.name)
        if match is None:
            raise ValueError(f"unexpected file in checkpoint directory: {path.name}")
        generation = int(match.group(1))
        if generation < 1 or path.name != f"checkpoint-{generation:06d}.json":
            raise ValueError(f"non-canonical checkpoint filename: {path.name}")
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"checkpoint entry must be a file: {path}")
        files.append((generation, path))
    files.sort()
    if [generation for generation, _ in files] != list(range(1, len(files) + 1)):
        raise ValueError("batch checkpoint generations must be contiguous from 1")

    previous_sha: str | None = None
    newest: dict[str, Any] | None = None
    for generation, path in files:
        document = _read_object(path)
        _exact_keys(
            document,
            {
                "schema_version",
                "generation",
                "fingerprint_sha256",
                "previous_checkpoint_sha256",
                "model_results",
                "attempts",
                "payload_sha256",
            },
            path=f"checkpoint {generation}",
        )
        if document["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(f"checkpoint {generation} has an unsupported schema_version")
        payload = {key: value for key, value in document.items() if key != "payload_sha256"}
        if document["payload_sha256"] != canonical_sha256(payload):
            raise ValueError(f"checkpoint {generation} payload digest does not match")
        if document["generation"] != generation:
            raise ValueError(f"checkpoint {generation} declares the wrong generation")
        if document["fingerprint_sha256"] != fingerprint_sha256:
            raise ValueError(f"checkpoint {generation} belongs to a different run fingerprint")
        if document["previous_checkpoint_sha256"] != previous_sha:
            raise ValueError(f"checkpoint {generation} breaks the immutable hash chain")
        if not isinstance(document["model_results"], list) or not all(
            isinstance(row, dict) for row in document["model_results"]
        ):
            raise ValueError(f"checkpoint {generation}.model_results must be an object array")
        if not isinstance(document["attempts"], list) or not all(
            isinstance(row, dict) for row in document["attempts"]
        ):
            raise ValueError(f"checkpoint {generation}.attempts must be an object array")
        previous_sha = sha256_file(path)
        newest = document

    pointer_path = checkpoints_dir / "latest.json"
    pointer_stale = False
    if files:
        if not pointer_path.is_file():
            pointer_stale = True
        else:
            pointer = _read_object(pointer_path)
            _exact_keys(
                pointer,
                {"schema_version", "generation", "checkpoint", "sha256"},
                path="checkpoint pointer",
            )
            if pointer["schema_version"] != CHECKPOINT_POINTER_SCHEMA_VERSION:
                raise ValueError("checkpoint pointer has an unsupported schema_version")
            pointer_generation = pointer["generation"]
            if not isinstance(pointer_generation, int) or isinstance(pointer_generation, bool):
                raise ValueError("checkpoint pointer generation must be an integer")
            if pointer_generation < 1 or pointer_generation > len(files):
                raise ValueError("checkpoint pointer references an unavailable generation")
            expected_path = f"checkpoints/checkpoint-{pointer_generation:06d}.json"
            if pointer["checkpoint"] != expected_path:
                raise ValueError("checkpoint pointer contains an unsafe or inconsistent path")
            pointed_path = mode_root / expected_path
            if pointer["sha256"] != sha256_file(pointed_path):
                raise ValueError("checkpoint pointer SHA-256 does not match its checkpoint")
            pointer_stale = pointer_generation != len(files)
    elif pointer_path.exists():
        raise ValueError("checkpoint pointer exists without any checkpoint generations")

    if newest is None:
        return BatchCheckpointState(0, None, (), (), pointer_stale)
    return BatchCheckpointState(
        generation=len(files),
        checkpoint_sha256=previous_sha,
        model_results=tuple(MappingProxyType(dict(row)) for row in newest["model_results"]),
        attempts=tuple(MappingProxyType(dict(row)) for row in newest["attempts"]),
        pointer_stale=pointer_stale,
    )


def repair_checkpoint_pointer(mode_root: Path, state: BatchCheckpointState) -> None:
    """Repair only a missing/stale derived pointer after validating the full chain."""

    if state.generation < 1 or state.checkpoint_sha256 is None:
        return
    relative = f"checkpoints/checkpoint-{state.generation:06d}.json"
    _atomic_json(
        mode_root / "checkpoints/latest.json",
        _pointer_document(state.generation, relative, state.checkpoint_sha256),
    )


def commit_checkpoint(
    mode_root: Path,
    *,
    fingerprint_sha256: str,
    model_results: Sequence[Mapping[str, Any]],
    attempts: Sequence[Mapping[str, Any]],
) -> BatchCheckpointState:
    """Append one fsynced generation, then atomically advance the derived pointer."""

    previous = load_checkpoint_chain(mode_root, fingerprint_sha256=fingerprint_sha256)
    previous_attempts = [dict(row) for row in previous.attempts]
    candidate_attempts = [dict(row) for row in attempts]
    if candidate_attempts[: len(previous_attempts)] != previous_attempts:
        raise ValueError("checkpoint attempts must retain the immutable previous prefix")
    if len(candidate_attempts) < len(previous_attempts):
        raise ValueError("checkpoint attempts cannot remove committed attempt history")

    previous_successes = {
        row.get("candidate_id"): dict(row)
        for row in previous.model_results
        if row.get("status") == "succeeded"
    }
    candidate_results = {row.get("candidate_id"): dict(row) for row in model_results}
    for candidate_id, previous_row in previous_successes.items():
        if candidate_results.get(candidate_id) != previous_row:
            raise ValueError("checkpoint model results cannot replace a committed success")
    generation = previous.generation + 1
    document = _checkpoint_document(
        generation=generation,
        fingerprint_sha256=fingerprint_sha256,
        previous_checkpoint_sha256=previous.checkpoint_sha256,
        model_results=model_results,
        attempts=attempts,
    )
    relative = f"checkpoints/checkpoint-{generation:06d}.json"
    path = mode_root / relative
    if path.exists():
        raise ValueError(f"refusing to overwrite immutable checkpoint {path}")
    _atomic_json(path, document)
    digest = sha256_file(path)
    _atomic_json(
        mode_root / "checkpoints/latest.json",
        _pointer_document(generation, relative, digest),
    )
    return BatchCheckpointState(
        generation=generation,
        checkpoint_sha256=digest,
        model_results=tuple(MappingProxyType(dict(row)) for row in model_results),
        attempts=tuple(MappingProxyType(dict(row)) for row in attempts),
    )


def safe_relative_path(value: str, *, prefix: str | None = None) -> Path:
    path = Path(value)
    if not value or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe relative artifact path: {value!r}")
    if prefix is not None and (not path.parts or path.parts[0] != prefix):
        raise ValueError(f"artifact path must remain under {prefix!r}: {value!r}")
    return path


def verify_hashed_artifacts(root: Path, hashes: Mapping[str, Any]) -> None:
    """Verify every declared child artifact without permitting path traversal."""

    for name, expected in hashes.items():
        if not isinstance(name, str):
            raise ValueError("artifact hash mapping contains a non-string path")
        relative = safe_relative_path(name)
        path = (root / relative).resolve()
        if root.resolve() not in path.parents:
            raise ValueError(f"artifact escapes its candidate directory: {name!r}")
        if not path.is_file():
            raise ValueError(f"declared candidate artifact is missing: {path}")
        if sha256_file(path) != _sha(expected, path=f"artifact hash {name}"):
            raise ValueError(f"candidate artifact SHA-256 mismatch: {path}")


def allocate_attempt_path(mode_root: Path, candidate_id: str) -> tuple[int, Path]:
    """Choose a never-before-used attempt directory without creating or deleting it."""

    if not _SAFE_COMPONENT.fullmatch(candidate_id):
        raise ValueError(f"unsafe candidate_id: {candidate_id!r}")
    models_root = mode_root / "models"
    if models_root.exists():
        if models_root.is_symlink() or not models_root.is_dir():
            raise ValueError(f"candidate models root must be a directory: {models_root}")
        if mode_root.resolve() not in models_root.resolve().parents:
            raise ValueError(f"candidate models root escapes the mode directory: {models_root}")
    candidate_root = models_root / candidate_id
    maximum = 0
    if candidate_root.exists():
        if candidate_root.is_symlink() or not candidate_root.is_dir():
            raise ValueError(f"candidate attempt root must be a directory: {candidate_root}")
        resolved_mode_root = mode_root.resolve()
        if resolved_mode_root not in candidate_root.resolve().parents:
            raise ValueError(f"candidate attempt root escapes the mode directory: {candidate_root}")
        for path in candidate_root.iterdir():
            match = re.fullmatch(r"attempt-([0-9]{4,})", path.name)
            if match is None:
                raise ValueError(f"unexpected candidate attempt entry: {path}")
            if path.is_symlink() or not path.is_dir():
                raise ValueError(f"candidate attempt entry must be a directory: {path}")
            attempt_number = int(match.group(1))
            if attempt_number < 1 or path.name != f"attempt-{attempt_number:04d}":
                raise ValueError(f"non-canonical candidate attempt entry: {path}")
            maximum = max(maximum, attempt_number)
    attempt = maximum + 1
    relative = Path("models") / candidate_id / f"attempt-{attempt:04d}"
    return attempt, relative


__all__ = [
    "BatchCheckpointState",
    "CHECKPOINT_POINTER_SCHEMA_VERSION",
    "CHECKPOINT_SCHEMA_VERSION",
    "RESUME_CONTRACT_SCHEMA_VERSION",
    "allocate_attempt_path",
    "build_resume_contract",
    "canonical_sha256",
    "commit_checkpoint",
    "exclusive_run_lock",
    "load_checkpoint_chain",
    "read_strict_json_object",
    "read_strict_jsonl_objects",
    "repair_checkpoint_pointer",
    "safe_relative_path",
    "sha256_file",
    "validate_resume_contract",
    "verify_hashed_artifacts",
    "write_or_validate_resume_contract",
]
