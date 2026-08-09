"""Strict input contract for externally generated EduLLM tutor responses."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import math
import os
import stat
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

PRECOMPUTED_RESPONSES_SCHEMA_VERSION = "edullm-precomputed-tutor-responses-v1"
PRECOMPUTED_RESPONSE_BATCH_SCHEMA_VERSION = "edullm-precomputed-tutor-response-batch-v1"
RESPONSE_BATCH_PUBLICATION_LOCK_NAME = ".edullm-response-batch-publish.lock"
RESPONSE_BATCH_PUBLICATION_JOURNAL_NAME = ".edullm-response-batch-publish.transaction.json"
RESPONSE_BATCH_PUBLICATION_SCHEMA_VERSION = "edullm-response-batch-publication-v1"
_ROW_REQUIRED_FIELDS = frozenset({"scenario_id", "response"})
_ROW_ALLOWED_FIELDS = frozenset({"scenario_id", "response", "metadata"})
_BATCH_ROW_IDENTITY_FIELDS = ("model_id", "model_family", "model_revision")
_BATCH_ROW_REQUIRED_FIELDS = frozenset({*_BATCH_ROW_IDENTITY_FIELDS, "scenario_id", "response"})
_BATCH_ROW_ALLOWED_FIELDS = _BATCH_ROW_REQUIRED_FIELDS | {"metadata"}
_HEX_DIGITS = frozenset("0123456789abcdef")


@dataclass(frozen=True, slots=True)
class PrecomputedTutorResponse:
    """One uploaded tutor response associated with a fitted-bank scenario."""

    scenario_id: str
    response: str
    metadata: Mapping[str, Any]
    source_row_sha256: str


@dataclass(frozen=True, slots=True)
class PrecomputedTutorResponses:
    """A fully validated, immutable snapshot of one uploaded response file."""

    path: Path
    sha256: str
    responses: Mapping[str, PrecomputedTutorResponse]
    blank_count: int

    @property
    def row_count(self) -> int:
        return len(self.responses)


@dataclass(frozen=True, slots=True)
class PrecomputedTutorResponseBatchModel:
    """One tutor model and its complete fitted-bank response set."""

    model_id: str
    model_family: str
    model_revision: str
    responses: Mapping[str, PrecomputedTutorResponse]
    blank_count: int

    @property
    def row_count(self) -> int:
        return len(self.responses)


@dataclass(frozen=True, slots=True)
class PrecomputedTutorResponseBatch:
    """A validated multi-model response file in deterministic model order."""

    path: Path
    sha256: str
    models: tuple[PrecomputedTutorResponseBatchModel, ...]
    models_by_id: Mapping[str, PrecomputedTutorResponseBatchModel]

    @property
    def model_count(self) -> int:
        return len(self.models)

    @property
    def row_count(self) -> int:
        return sum(model.row_count for model in self.models)

    @property
    def blank_count(self) -> int:
        return sum(model.blank_count for model in self.models)


def _strict_json(value: Any, *, path: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} contains a non-string object key")
            _strict_json(item, path=f"{path}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _strict_json(item, path=f"{path}[{index}]")
        return
    raise ValueError(f"{path} contains unsupported value {type(value).__name__}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _load_publication_journal(journal: Path) -> Mapping[str, Any]:
    if journal.is_symlink() or not journal.is_file():
        raise ValueError(f"response-batch publication journal is unsafe: {journal}")
    try:
        value = json.loads(
            journal.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"response-batch publication journal is invalid: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError("response-batch publication journal must contain one object")
    if set(value) != {"schema_version", "transaction_id", "state", "entries"}:
        raise ValueError("response-batch publication journal fields differ")
    if value["schema_version"] != RESPONSE_BATCH_PUBLICATION_SCHEMA_VERSION:
        raise ValueError("response-batch publication journal has an unsupported version")
    transaction_id = value["transaction_id"]
    if (
        not isinstance(transaction_id, str)
        or len(transaction_id) != 32
        or set(transaction_id) - _HEX_DIGITS
    ):
        raise ValueError("response-batch publication transaction_id is invalid")
    if value["state"] not in {"staging", "prepared", "committed"}:
        raise ValueError("response-batch publication state is invalid")
    entries = value["entries"]
    if not isinstance(entries, list) or len(entries) != 2:
        raise ValueError("response-batch publication journal must contain exactly two entries")
    destinations: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != {
            "destination",
            "staged",
            "backup",
            "original_existed",
            "original_sha256",
            "new_sha256",
        }:
            raise ValueError("response-batch publication journal entry fields differ")
        for field in ("destination", "staged", "backup"):
            name = entry[field]
            if (
                not isinstance(name, str)
                or not name
                or name in {".", ".."}
                or "\x00" in name
                or Path(name).name != name
            ):
                raise ValueError(f"response-batch publication journal {field} is unsafe")
        destination = entry["destination"]
        if destination.casefold() in destinations:
            raise ValueError("response-batch publication destinations are duplicated")
        destinations.add(destination.casefold())
        if (
            entry["staged"] != f".{destination}.{transaction_id}.stage"
            or entry["backup"] != f".{destination}.{transaction_id}.backup"
        ):
            raise ValueError("response-batch publication temporary name is invalid")
        original_existed = entry["original_existed"]
        original_sha256 = entry["original_sha256"]
        if not isinstance(original_existed, bool):
            raise ValueError("response-batch publication original_existed must be boolean")
        if original_existed != isinstance(original_sha256, str):
            raise ValueError("response-batch publication original digest is inconsistent")
        for field in ("original_sha256", "new_sha256"):
            digest = entry[field]
            if digest is not None and (
                not isinstance(digest, str) or len(digest) != 64 or set(digest) - _HEX_DIGITS
            ):
                raise ValueError(f"response-batch publication {field} is invalid")
        if not isinstance(entry["new_sha256"], str):
            raise ValueError("response-batch publication new_sha256 is required")
    return value


def _refuse_incomplete_publication(source: Path) -> None:
    """Do not consume a destination while its paired publication is unresolved."""

    journal = source.parent / RESPONSE_BATCH_PUBLICATION_JOURNAL_NAME
    if not journal.exists() and not journal.is_symlink():
        return
    value = _load_publication_journal(journal)
    for entry in value["entries"]:
        artifact_names = [entry[field] for field in ("destination", "staged", "backup")]
        artifact_names.append(f".{entry['destination']}.{value['transaction_id']}.rollback")
        for name in artifact_names:
            artifact_path = journal.parent / name
            matches_artifact = source.name.casefold() == name.casefold()
            if not matches_artifact and artifact_path.exists():
                try:
                    matches_artifact = os.path.samefile(source, artifact_path)
                except OSError:
                    matches_artifact = False
            if matches_artifact:
                raise ValueError(
                    "response batch belongs to an incomplete paired publication; rerun "
                    "prepare-response-batch with the original arguments to recover it"
                )


@contextlib.contextmanager
def _publication_read_lock(parent: Path) -> Iterator[None]:
    """Share the publication lock for the complete input snapshot read."""

    lock_path = parent / RESPONSE_BATCH_PUBLICATION_LOCK_NAME
    if lock_path.is_symlink() or lock_path.exists() and not lock_path.is_file():
        raise ValueError(f"response-batch publication lock is unsafe: {lock_path}")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise ValueError(f"could not safely open response-batch publication lock: {exc}") from exc
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise ValueError(f"response-batch publication lock is unsafe: {lock_path}")
    stream = os.fdopen(descriptor, "rb")
    try:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"response-batch publication is active in {parent}") from exc
        yield
    finally:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()


def _read_precomputed_payload(
    path: str | Path,
    *,
    label: str,
    publication_lock_held: bool,
) -> tuple[Path, bytes]:
    raw_source = Path(path).expanduser()
    if not raw_source.is_file():
        raise ValueError(f"{label} does not exist: {raw_source}")
    source = raw_source.resolve()

    def read_payload() -> bytes:
        if not source.is_file():
            raise ValueError(f"{label} does not exist: {source}")
        if source.stat().st_nlink != 1:
            raise ValueError(f"{label} must not have hard-link aliases: {source}")
        try:
            return source.read_bytes()
        except OSError as exc:
            raise ValueError(f"could not read {label} {source}: {exc}") from exc

    if publication_lock_held:
        return source, read_payload()

    lock_path = source.parent / RESPONSE_BATCH_PUBLICATION_LOCK_NAME
    if not lock_path.exists() and not lock_path.is_symlink():
        # Standalone/read-only datasets have no publication sentinel. Read
        # optimistically, then linearize at the second absence check. Writers
        # create their persistent lock before any journal or destination edit.
        _refuse_incomplete_publication(source)
        payload = read_payload()
        if not lock_path.exists() and not lock_path.is_symlink():
            _refuse_incomplete_publication(source)
            if not lock_path.exists() and not lock_path.is_symlink():
                return source, payload

    with _publication_read_lock(source.parent):
        _refuse_incomplete_publication(source)
        return source, read_payload()


def _canonical_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_json(item) for item in value)
    return value


def load_precomputed_tutor_responses(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_scenario_ids: Sequence[str],
    _publication_lock_held: bool = False,
) -> PrecomputedTutorResponses:
    """Load a complete uploaded response set and fail closed on any mismatch."""

    source, payload = _read_precomputed_payload(
        path,
        label="precomputed tutor response file",
        publication_lock_held=_publication_lock_held,
    )
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    if observed_sha256 != expected_sha256:
        raise ValueError(
            "precomputed tutor response SHA-256 mismatch: "
            f"expected {expected_sha256}, observed {observed_sha256}"
        )
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("precomputed tutor response file must be valid UTF-8") from exc

    responses: dict[str, PrecomputedTutorResponse] = {}
    # JSONL uses literal LF delimiters. ``splitlines()`` would incorrectly
    # split on valid U+0085/U+2028/U+2029 characters inside JSON strings.
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if not lines:
        raise ValueError("precomputed tutor response file is empty")
    for line_number, line in enumerate(lines, start=1):
        label = f"precomputed tutor response line {line_number}"
        if not line.strip():
            raise ValueError(f"{label} must not be blank")
        try:
            row = json.loads(
                line,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_json_constant,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"{label} is invalid JSON: {exc}") from exc
        if not isinstance(row, Mapping):
            raise ValueError(f"{label} must be a JSON object")
        missing = sorted(_ROW_REQUIRED_FIELDS - set(row))
        unknown = sorted(set(row) - _ROW_ALLOWED_FIELDS)
        if missing or unknown:
            raise ValueError(f"{label} fields differ; missing={missing}, extra={unknown}")
        _strict_json(row, path=label)

        scenario_id = row["scenario_id"]
        if (
            not isinstance(scenario_id, str)
            or not scenario_id
            or scenario_id != scenario_id.strip()
        ):
            raise ValueError(f"{label}.scenario_id must be a non-empty, unpadded string")
        if scenario_id in responses:
            raise ValueError(f"duplicate precomputed tutor response for scenario {scenario_id!r}")
        response = row["response"]
        if not isinstance(response, str):
            raise ValueError(f"{label}.response must be a string")
        metadata = row.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError(f"{label}.metadata must be an object when supplied")
        normalized_metadata = dict(metadata)
        normalized_row = {
            "scenario_id": scenario_id,
            "response": response,
            "metadata": normalized_metadata,
        }
        responses[scenario_id] = PrecomputedTutorResponse(
            scenario_id=scenario_id,
            response=response,
            metadata=_freeze_json(normalized_metadata),
            source_row_sha256=_canonical_hash(normalized_row),
        )

    expected = set(expected_scenario_ids)
    if len(expected) != len(tuple(expected_scenario_ids)):
        raise ValueError("expected fitted-bank scenario IDs must be unique")
    actual = set(responses)
    missing_scenarios = sorted(expected - actual)
    extra_scenarios = sorted(actual - expected)
    if missing_scenarios or extra_scenarios:
        raise ValueError(
            "precomputed tutor responses must exactly cover the fitted bank; "
            f"missing={missing_scenarios}, extra={extra_scenarios}"
        )

    return PrecomputedTutorResponses(
        path=source.resolve(),
        sha256=observed_sha256,
        responses=MappingProxyType(responses),
        blank_count=sum(not row.response.strip() for row in responses.values()),
    )


def load_precomputed_tutor_response_batch(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_scenario_ids: Sequence[str],
    _publication_lock_held: bool = False,
) -> PrecomputedTutorResponseBatch:
    """Load complete response sets for one or more explicitly identified tutor models."""

    source, payload = _read_precomputed_payload(
        path,
        label="precomputed tutor response batch file",
        publication_lock_held=_publication_lock_held,
    )
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    if observed_sha256 != expected_sha256:
        raise ValueError(
            "precomputed tutor response batch SHA-256 mismatch: "
            f"expected {expected_sha256}, observed {observed_sha256}"
        )
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("precomputed tutor response batch file must be valid UTF-8") from exc

    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if not lines:
        raise ValueError("precomputed tutor response batch must contain at least one model")

    identities: dict[str, tuple[str, str]] = {}
    responses_by_model: dict[str, dict[str, PrecomputedTutorResponse]] = {}
    for line_number, line in enumerate(lines, start=1):
        label = f"precomputed tutor response batch line {line_number}"
        if not line.strip():
            raise ValueError(f"{label} must not be blank")
        try:
            row = json.loads(
                line,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_json_constant,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"{label} is invalid JSON: {exc}") from exc
        if not isinstance(row, Mapping):
            raise ValueError(f"{label} must be a JSON object")
        missing = sorted(_BATCH_ROW_REQUIRED_FIELDS - set(row))
        unknown = sorted(set(row) - _BATCH_ROW_ALLOWED_FIELDS)
        if missing or unknown:
            raise ValueError(f"{label} fields differ; missing={missing}, extra={unknown}")
        _strict_json(row, path=label)

        for field in (*_BATCH_ROW_IDENTITY_FIELDS, "scenario_id"):
            value = row[field]
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"{label}.{field} must be a non-empty, unpadded string")
        model_id = row["model_id"]
        model_family = row["model_family"]
        model_revision = row["model_revision"]
        scenario_id = row["scenario_id"]
        response = row["response"]
        if not isinstance(response, str):
            raise ValueError(f"{label}.response must be a string")
        metadata = row.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError(f"{label}.metadata must be an object when supplied")

        identity = (model_family, model_revision)
        previous_identity = identities.setdefault(model_id, identity)
        if previous_identity != identity:
            raise ValueError(
                f"inconsistent identity fields for model_id {model_id!r}: "
                f"expected model_family={previous_identity[0]!r}, "
                f"model_revision={previous_identity[1]!r}; observed "
                f"model_family={model_family!r}, model_revision={model_revision!r}"
            )
        model_responses = responses_by_model.setdefault(model_id, {})
        if scenario_id in model_responses:
            raise ValueError(
                "duplicate precomputed tutor response batch row for "
                f"model_id={model_id!r}, scenario_id={scenario_id!r}"
            )

        normalized_metadata = dict(metadata)
        normalized_row = {
            "model_id": model_id,
            "model_family": model_family,
            "model_revision": model_revision,
            "scenario_id": scenario_id,
            "response": response,
            "metadata": normalized_metadata,
        }
        model_responses[scenario_id] = PrecomputedTutorResponse(
            scenario_id=scenario_id,
            response=response,
            metadata=_freeze_json(normalized_metadata),
            source_row_sha256=_canonical_hash(normalized_row),
        )

    if not responses_by_model:
        raise ValueError("precomputed tutor response batch must contain at least one model")

    expected_scenario_ids_tuple = tuple(expected_scenario_ids)
    expected = set(expected_scenario_ids_tuple)
    if len(expected) != len(expected_scenario_ids_tuple):
        raise ValueError("expected fitted-bank scenario IDs must be unique")

    models: list[PrecomputedTutorResponseBatchModel] = []
    for model_id in sorted(responses_by_model):
        model_responses = responses_by_model[model_id]
        actual = set(model_responses)
        missing_scenarios = sorted(expected - actual)
        extra_scenarios = sorted(actual - expected)
        if missing_scenarios or extra_scenarios:
            raise ValueError(
                "precomputed tutor response batch must exactly cover the fitted bank for "
                f"model_id {model_id!r}; missing={missing_scenarios}, extra={extra_scenarios}"
            )
        model_family, model_revision = identities[model_id]
        ordered_responses = {
            scenario_id: model_responses[scenario_id] for scenario_id in sorted(model_responses)
        }
        models.append(
            PrecomputedTutorResponseBatchModel(
                model_id=model_id,
                model_family=model_family,
                model_revision=model_revision,
                responses=MappingProxyType(ordered_responses),
                blank_count=sum(
                    not tutor_response.response.strip()
                    for tutor_response in ordered_responses.values()
                ),
            )
        )

    ordered_models = tuple(models)
    return PrecomputedTutorResponseBatch(
        path=source.resolve(),
        sha256=observed_sha256,
        models=ordered_models,
        models_by_id=MappingProxyType({model.model_id: model for model in ordered_models}),
    )


__all__ = [
    "PRECOMPUTED_RESPONSE_BATCH_SCHEMA_VERSION",
    "PRECOMPUTED_RESPONSES_SCHEMA_VERSION",
    "RESPONSE_BATCH_PUBLICATION_JOURNAL_NAME",
    "RESPONSE_BATCH_PUBLICATION_LOCK_NAME",
    "RESPONSE_BATCH_PUBLICATION_SCHEMA_VERSION",
    "PrecomputedTutorResponse",
    "PrecomputedTutorResponseBatch",
    "PrecomputedTutorResponseBatchModel",
    "PrecomputedTutorResponses",
    "load_precomputed_tutor_response_batch",
    "load_precomputed_tutor_responses",
]
