"""Deterministic preparation of strict multi-model tutor response batches."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import math
import os
import stat
import tempfile
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from .precomputed import (
    PRECOMPUTED_RESPONSE_BATCH_SCHEMA_VERSION,
    RESPONSE_BATCH_PUBLICATION_JOURNAL_NAME,
    RESPONSE_BATCH_PUBLICATION_LOCK_NAME,
    RESPONSE_BATCH_PUBLICATION_SCHEMA_VERSION,
    load_precomputed_tutor_response_batch,
)

RESPONSE_BATCH_PREPARATION_REPORT_SCHEMA_VERSION = "edullm-response-batch-preparation-report-v1"
BATCH_JSONL_SOURCE_FORMAT = "batch-jsonl-v1"
SINGLE_JSONL_SOURCE_FORMAT = "single-jsonl-v1"
TUTORBENCH_OUTPUT_JSONL_SOURCE_FORMAT = "tutorbench-output-jsonl-v1"
RESPONSE_BATCH_SOURCE_FORMATS = (
    BATCH_JSONL_SOURCE_FORMAT,
    SINGLE_JSONL_SOURCE_FORMAT,
    TUTORBENCH_OUTPUT_JSONL_SOURCE_FORMAT,
)

SourceFormat = Literal[
    "batch-jsonl-v1",
    "single-jsonl-v1",
    "tutorbench-output-jsonl-v1",
]
MissingResponsePolicy = Literal["error", "blank"]

_MANIFEST_COMMON_FIELDS = frozenset({"format", "path"})
_MANIFEST_IDENTITY_FIELDS = frozenset({"model_id", "model_family", "model_revision"})
_BATCH_ROW_REQUIRED_FIELDS = frozenset(
    {"model_id", "model_family", "model_revision", "scenario_id", "response"}
)
_BATCH_ROW_ALLOWED_FIELDS = _BATCH_ROW_REQUIRED_FIELDS | {"metadata"}
_SINGLE_ROW_REQUIRED_FIELDS = frozenset({"scenario_id", "response"})
_SINGLE_ROW_ALLOWED_FIELDS = _SINGLE_ROW_REQUIRED_FIELDS | {"metadata"}
_TUTORBENCH_ROW_FIELDS = frozenset(
    {
        "Benchmark",
        "Scenario",
        "Model",
        "Model Revision",
        "Chat Template Applied",
        "Rendered Prompt",
        "Generation Params",
        "Max Model Len",
        "Prompt Tokens",
        "Output Tokens",
        "Finish Reason",
        "Truncated",
        "Latency (s)",
        "Output",
        "Issue",
        "Issue Description",
    }
)
_HEX_DIGITS = frozenset("0123456789abcdef")
_PUBLICATION_LOCK_NAME = RESPONSE_BATCH_PUBLICATION_LOCK_NAME
_PUBLICATION_JOURNAL_NAME = RESPONSE_BATCH_PUBLICATION_JOURNAL_NAME
_PUBLICATION_TRANSACTION_SCHEMA_VERSION = RESPONSE_BATCH_PUBLICATION_SCHEMA_VERSION


class ResponsePreparationError(ValueError):
    """Raised when response inputs cannot safely produce the batch contract."""


@dataclass(frozen=True, slots=True)
class ResponseBatchSource:
    """One explicitly typed source listed in a response source manifest."""

    index: int
    format: SourceFormat
    path: Path
    expected_sha256: str | None
    model_id: str | None = None
    model_family: str | None = None
    model_revision: str | None = None


@dataclass(frozen=True, slots=True)
class ResponseBatchSourceManifest:
    """Strict source manifest and the exact bytes from which it was loaded."""

    path: Path
    sha256: str
    sources: tuple[ResponseBatchSource, ...]


@dataclass(frozen=True, slots=True)
class FittedScenarioRoster:
    """Ordered scenario IDs and provenance for the fitted scenario file."""

    path: Path
    sha256: str
    scenario_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PreparedPrecomputedTutorResponseBatch:
    """Validated, deterministic bytes ready for an explicit atomic write."""

    payload: bytes
    sha256: str
    scenario_ids: tuple[str, ...]
    protected_input_paths: tuple[Path, ...]
    report: Mapping[str, Any]
    report_sha256: str
    model_count: int
    row_count: int
    blank_count: int
    synthesized_blank_count: int


@dataclass(frozen=True, slots=True)
class WrittenPrecomputedTutorResponseBatch:
    """Paths, hashes, and report from a completed atomic write."""

    output_path: Path
    report_path: Path
    sha256: str
    report_sha256: str
    report: Mapping[str, Any]


@dataclass(slots=True)
class _RollbackSnapshot:
    entry: Mapping[str, Any]
    destination: Path
    staged: Path
    backup: Path
    capture: Path
    destination_digest: str | None
    backup_digest: str | None
    capture_digest: str | None
    needs_capture: bool


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ResponsePreparationError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ResponsePreparationError(f"non-standard JSON constant {value!r}")


def _strict_json(value: Any, *, path: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ResponsePreparationError(f"{path} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ResponsePreparationError(f"{path} contains a non-string object key")
            _strict_json(item, path=f"{path}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _strict_json(item, path=f"{path}[{index}]")
        return
    raise ResponsePreparationError(f"{path} contains unsupported value {type(value).__name__}")


def _read_bytes(path: Path, label: str) -> bytes:
    if not path.is_file():
        raise ResponsePreparationError(f"{label} does not exist: {path}")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ResponsePreparationError(f"could not read {label} {path}: {exc}") from exc


def _parse_jsonl(
    payload: bytes,
    *,
    label: str,
    allow_empty: bool = False,
) -> list[dict[str, Any]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ResponsePreparationError(f"{label} must be valid UTF-8") from exc
    # JSONL records are separated by the literal LF byte.  ``str.splitlines``
    # also treats valid JSON string characters such as U+0085, U+2028, and
    # U+2029 as record boundaries, corrupting otherwise valid rows.
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if not lines:
        if allow_empty:
            return []
        raise ResponsePreparationError(f"{label} must contain at least one row")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        row_label = f"{label} line {line_number}"
        if not line.strip():
            raise ResponsePreparationError(f"{row_label} must not be blank")
        try:
            row = json.loads(
                line,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_json_constant,
            )
        except ValueError as exc:
            raise ResponsePreparationError(f"{row_label} is invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ResponsePreparationError(f"{row_label} must be a JSON object")
        _strict_json(row, path=row_label)
        rows.append(row)
    return rows


def _exact_fields(
    row: Mapping[str, Any],
    *,
    required: frozenset[str],
    allowed: frozenset[str],
    label: str,
) -> None:
    missing = sorted(required - set(row))
    extra = sorted(set(row) - allowed)
    if missing or extra:
        raise ResponsePreparationError(f"{label} fields differ; missing={missing}, extra={extra}")


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ResponsePreparationError(f"{label} must be a non-empty, unpadded string")
    return value


def _optional_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or set(value) - _HEX_DIGITS:
        raise ResponsePreparationError(f"{label} must be a lowercase 64-character SHA-256")
    return value


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def load_response_batch_source_manifest(
    path: str | Path,
) -> ResponseBatchSourceManifest:
    """Load the exact source roster; no formats or identities are inferred."""

    manifest_path = _resolved(path)
    payload = _read_bytes(manifest_path, "response batch source manifest")
    rows = _parse_jsonl(payload, label="response batch source manifest")
    sources: list[ResponseBatchSource] = []
    seen_paths: set[Path] = set()
    for index, row in enumerate(rows, start=1):
        label = f"response batch source manifest line {index}"
        source_format = row.get("format")
        if source_format not in RESPONSE_BATCH_SOURCE_FORMATS:
            raise ResponsePreparationError(
                f"{label}.format must be one of {list(RESPONSE_BATCH_SOURCE_FORMATS)!r}"
            )
        is_batch = source_format == BATCH_JSONL_SOURCE_FORMAT
        required = _MANIFEST_COMMON_FIELDS | (
            frozenset() if is_batch else _MANIFEST_IDENTITY_FIELDS
        )
        allowed = required | {"sha256"}
        _exact_fields(row, required=required, allowed=allowed, label=label)
        raw_path = _nonempty_string(row["path"], f"{label}.path")
        source_path = Path(raw_path).expanduser()
        if not source_path.is_absolute():
            source_path = manifest_path.parent / source_path
        source_path = source_path.resolve()
        if source_path == manifest_path:
            raise ResponsePreparationError(f"{label}.path must not reference the manifest itself")
        if source_path in seen_paths:
            raise ResponsePreparationError(f"duplicate response source path: {source_path}")
        if not source_path.is_file():
            raise ResponsePreparationError(f"response source does not exist: {source_path}")
        seen_paths.add(source_path)
        model_id = model_family = model_revision = None
        if not is_batch:
            model_id = _nonempty_string(row["model_id"], f"{label}.model_id")
            model_family = _nonempty_string(row["model_family"], f"{label}.model_family")
            model_revision = _nonempty_string(row["model_revision"], f"{label}.model_revision")
        sources.append(
            ResponseBatchSource(
                index=index,
                format=source_format,
                path=source_path,
                expected_sha256=(
                    _optional_sha256(row["sha256"], f"{label}.sha256") if "sha256" in row else None
                ),
                model_id=model_id,
                model_family=model_family,
                model_revision=model_revision,
            )
        )
    return ResponseBatchSourceManifest(
        path=manifest_path,
        sha256=hashlib.sha256(payload).hexdigest(),
        sources=tuple(sources),
    )


def load_fitted_scenario_roster(path: str | Path) -> FittedScenarioRoster:
    """Read the ordered scenario IDs used to define complete batch coverage."""

    scenario_path = _resolved(path)
    payload = _read_bytes(scenario_path, "fitted scenario file")
    rows = _parse_jsonl(payload, label="fitted scenario file")
    scenario_ids: list[str] = []
    seen: set[str] = set()
    for index, row in enumerate(rows, start=1):
        label = f"fitted scenario file line {index}.scenario_id"
        if "scenario_id" not in row:
            raise ResponsePreparationError(f"{label} is required")
        scenario_id = _nonempty_string(row["scenario_id"], label)
        if scenario_id in seen:
            raise ResponsePreparationError(f"duplicate fitted scenario_id {scenario_id!r}")
        seen.add(scenario_id)
        scenario_ids.append(scenario_id)
    return FittedScenarioRoster(
        path=scenario_path,
        sha256=hashlib.sha256(payload).hexdigest(),
        scenario_ids=tuple(scenario_ids),
    )


def _response_and_metadata(
    row: Mapping[str, Any],
    *,
    label: str,
) -> tuple[str, dict[str, Any]]:
    response = row["response"]
    if not isinstance(response, str):
        raise ResponsePreparationError(f"{label}.response must be a string")
    metadata = row.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ResponsePreparationError(f"{label}.metadata must be an object when supplied")
    return response, dict(metadata)


def _binary_flag(value: Any, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    raise ResponsePreparationError(f"{label} must be 0, 1, false, or true")


def _nonnegative_int_or_none(value: Any, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ResponsePreparationError(f"{label} must be a nonnegative integer or null")
    return value


def _legacy_response(
    row: Mapping[str, Any],
    *,
    source: ResponseBatchSource,
    label: str,
) -> tuple[str, str, dict[str, Any]]:
    _exact_fields(
        row,
        required=_TUTORBENCH_ROW_FIELDS,
        allowed=_TUTORBENCH_ROW_FIELDS,
        label=label,
    )
    if source.model_id is None or source.model_revision is None:
        raise ResponsePreparationError(f"internal error: {label} has no declared identity")
    scenario_id = _nonempty_string(row["Scenario"], f"{label}.Scenario")
    source_model = _nonempty_string(row["Model"], f"{label}.Model")
    if source_model != source.model_id:
        raise ResponsePreparationError(
            f"{label}.Model {source_model!r} differs from declared model_id {source.model_id!r}"
        )
    source_revision = row["Model Revision"]
    if not isinstance(source_revision, str):
        raise ResponsePreparationError(f"{label}.Model Revision must be a string")
    if source_revision and source_revision != source.model_revision:
        raise ResponsePreparationError(
            f"{label}.Model Revision {source_revision!r} differs from declared "
            f"model_revision {source.model_revision!r}"
        )
    response = row["Output"]
    if not isinstance(response, str):
        raise ResponsePreparationError(f"{label}.Output must be a string")
    benchmark = _nonempty_string(row["Benchmark"], f"{label}.Benchmark")
    rendered_prompt = row["Rendered Prompt"]
    if not isinstance(rendered_prompt, str):
        raise ResponsePreparationError(f"{label}.Rendered Prompt must be a string")
    generation_params = row["Generation Params"]
    if not isinstance(generation_params, dict):
        raise ResponsePreparationError(f"{label}.Generation Params must be an object")
    finish_reason = row["Finish Reason"]
    if finish_reason not in {"stop", "length", "error"}:
        raise ResponsePreparationError(
            f"{label}.Finish Reason must be 'stop', 'length', or 'error'"
        )
    issue = _binary_flag(row["Issue"], f"{label}.Issue")
    if finish_reason == "error" and not issue:
        raise ResponsePreparationError(f"{label} has Finish Reason='error' but Issue is false")
    if issue and response.strip():
        raise ResponsePreparationError(f"{label} has nonblank Output for a flagged issue row")
    prompt_tokens = _nonnegative_int_or_none(row["Prompt Tokens"], f"{label}.Prompt Tokens")
    output_tokens = _nonnegative_int_or_none(row["Output Tokens"], f"{label}.Output Tokens")
    if prompt_tokens is None or output_tokens is None:
        raise ResponsePreparationError(f"{label} token counts must not be null")
    max_model_len = _nonnegative_int_or_none(row["Max Model Len"], f"{label}.Max Model Len")
    latency = row["Latency (s)"]
    if latency is not None and (
        isinstance(latency, bool) or not isinstance(latency, int | float) or latency < 0
    ):
        raise ResponsePreparationError(f"{label}.Latency (s) must be nonnegative or null")
    issue_description = row["Issue Description"]
    if not isinstance(issue_description, str):
        raise ResponsePreparationError(f"{label}.Issue Description must be a string")
    if issue and issue_description.strip().lower() in {"", "n/a", "none"}:
        raise ResponsePreparationError(f"{label}.Issue Description must describe a flagged issue")
    metadata = {
        "source_format": TUTORBENCH_OUTPUT_JSONL_SOURCE_FORMAT,
        "benchmark": benchmark,
        "rendered_prompt": rendered_prompt,
        "rendered_prompt_sha256": hashlib.sha256(rendered_prompt.encode("utf-8")).hexdigest(),
        "chat_template_applied": _binary_flag(
            row["Chat Template Applied"], f"{label}.Chat Template Applied"
        ),
        "generation_params": dict(generation_params),
        "max_model_len": max_model_len,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "finish_reason": finish_reason,
        "truncated": _binary_flag(row["Truncated"], f"{label}.Truncated"),
        "latency_s": latency,
        "issue": issue,
        "issue_description": issue_description,
    }
    return scenario_id, response, metadata


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_thaw_json(item) for item in value]
    return value


def _serialize_jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    try:
        return "".join(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
            for row in rows
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ResponsePreparationError(f"prepared rows are not strict JSON: {exc}") from exc


def _canonical_json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_precomputed_tutor_response_batch(
    manifest: ResponseBatchSourceManifest,
    roster: FittedScenarioRoster,
    *,
    missing: MissingResponsePolicy = "error",
) -> PreparedPrecomputedTutorResponseBatch:
    """Normalize explicit sources into deterministic, complete batch JSONL bytes."""

    if missing not in ("error", "blank"):
        raise ResponsePreparationError("missing policy must be 'error' or 'blank'")
    if not manifest.sources:
        raise ResponsePreparationError("response source manifest must contain at least one source")
    if not roster.scenario_ids:
        raise ResponsePreparationError("fitted scenario roster must contain at least one scenario")
    for index, scenario_id in enumerate(roster.scenario_ids, start=1):
        _nonempty_string(scenario_id, f"fitted scenario roster item {index}")
    if len(set(roster.scenario_ids)) != len(roster.scenario_ids):
        raise ResponsePreparationError("fitted scenario roster IDs must be unique")
    expected = set(roster.scenario_ids)
    identities: dict[str, tuple[str, str]] = {}
    responses: dict[str, dict[str, dict[str, Any]]] = {}
    model_source_indices: dict[str, set[int]] = {}
    source_summaries: list[dict[str, Any]] = []

    def register_identity(
        model_id: str,
        model_family: str,
        model_revision: str,
        *,
        label: str,
        source_index: int,
    ) -> None:
        identity = (model_family, model_revision)
        previous = identities.setdefault(model_id, identity)
        if previous != identity:
            raise ResponsePreparationError(
                f"{label} conflicts with identity for model_id {model_id!r}: "
                f"expected model_family={previous[0]!r}, model_revision={previous[1]!r}; "
                f"observed model_family={model_family!r}, model_revision={model_revision!r}"
            )
        responses.setdefault(model_id, {})
        model_source_indices.setdefault(model_id, set()).add(source_index)

    def add_response(
        model_id: str,
        scenario_id: str,
        response: str,
        metadata: Mapping[str, Any],
        *,
        label: str,
    ) -> None:
        if scenario_id not in expected:
            raise ResponsePreparationError(
                f"{label} references scenario_id {scenario_id!r} outside the fitted bank"
            )
        model_responses = responses[model_id]
        if scenario_id in model_responses:
            raise ResponsePreparationError(
                f"duplicate response for model_id={model_id!r}, scenario_id={scenario_id!r}"
            )
        model_responses[scenario_id] = {
            "response": response,
            "metadata": dict(metadata),
            "synthesized": False,
        }

    for source in manifest.sources:
        if source.format not in RESPONSE_BATCH_SOURCE_FORMATS:
            raise ResponsePreparationError(
                f"response source {source.index} has unsupported format {source.format!r}"
            )
        if source.expected_sha256 is not None:
            _optional_sha256(
                source.expected_sha256,
                f"response source {source.index} expected_sha256",
            )
        payload = _read_bytes(source.path, f"response source {source.index}")
        observed_sha256 = hashlib.sha256(payload).hexdigest()
        if source.expected_sha256 is not None and source.expected_sha256 != observed_sha256:
            raise ResponsePreparationError(
                f"response source {source.index} SHA-256 mismatch: "
                f"expected {source.expected_sha256}, observed {observed_sha256}"
            )
        rows = _parse_jsonl(
            payload,
            label=f"response source {source.index} ({source.path})",
            allow_empty=source.format != BATCH_JSONL_SOURCE_FORMAT,
        )
        source_model_ids: set[str] = set()
        source_blank_count = 0
        if source.format != BATCH_JSONL_SOURCE_FORMAT:
            if (
                source.model_id is None
                or source.model_family is None
                or source.model_revision is None
            ):
                raise ResponsePreparationError(
                    f"internal error: response source {source.index} has no identity"
                )
            _nonempty_string(source.model_id, f"response source {source.index} model_id")
            _nonempty_string(source.model_family, f"response source {source.index} model_family")
            _nonempty_string(
                source.model_revision,
                f"response source {source.index} model_revision",
            )
            register_identity(
                source.model_id,
                source.model_family,
                source.model_revision,
                label=f"response source {source.index}",
                source_index=source.index,
            )
            source_model_ids.add(source.model_id)

        for row_index, row in enumerate(rows, start=1):
            label = f"response source {source.index} line {row_index}"
            if source.format == BATCH_JSONL_SOURCE_FORMAT:
                _exact_fields(
                    row,
                    required=_BATCH_ROW_REQUIRED_FIELDS,
                    allowed=_BATCH_ROW_ALLOWED_FIELDS,
                    label=label,
                )
                model_id = _nonempty_string(row["model_id"], f"{label}.model_id")
                model_family = _nonempty_string(row["model_family"], f"{label}.model_family")
                model_revision = _nonempty_string(row["model_revision"], f"{label}.model_revision")
                scenario_id = _nonempty_string(row["scenario_id"], f"{label}.scenario_id")
                response, metadata = _response_and_metadata(row, label=label)
                register_identity(
                    model_id,
                    model_family,
                    model_revision,
                    label=label,
                    source_index=source.index,
                )
            elif source.format == SINGLE_JSONL_SOURCE_FORMAT:
                _exact_fields(
                    row,
                    required=_SINGLE_ROW_REQUIRED_FIELDS,
                    allowed=_SINGLE_ROW_ALLOWED_FIELDS,
                    label=label,
                )
                if source.model_id is None:
                    raise ResponsePreparationError(f"internal error: {label} has no model_id")
                model_id = source.model_id
                scenario_id = _nonempty_string(row["scenario_id"], f"{label}.scenario_id")
                response, metadata = _response_and_metadata(row, label=label)
            else:
                if source.model_id is None:
                    raise ResponsePreparationError(f"internal error: {label} has no model_id")
                model_id = source.model_id
                scenario_id, response, metadata = _legacy_response(
                    row,
                    source=source,
                    label=label,
                )
            source_model_ids.add(model_id)
            add_response(
                model_id,
                scenario_id,
                response,
                metadata,
                label=label,
            )
            source_blank_count += not response.strip()

        source_summaries.append(
            {
                "index": source.index,
                "format": source.format,
                "path": str(source.path),
                "declared_sha256": source.expected_sha256,
                "observed_sha256": observed_sha256,
                "row_count": len(rows),
                "blank_count": source_blank_count,
                "model_ids": sorted(source_model_ids),
            }
        )

    if not identities:
        raise ResponsePreparationError("response sources must identify at least one model")

    synthesized_blank_count = 0
    for model_id in sorted(identities):
        model_responses = responses[model_id]
        missing_scenarios = [sid for sid in roster.scenario_ids if sid not in model_responses]
        if missing_scenarios and missing == "error":
            raise ResponsePreparationError(
                "response sources must exactly cover the fitted bank for "
                f"model_id {model_id!r}; missing={missing_scenarios}"
            )
        for scenario_id in missing_scenarios:
            model_responses[scenario_id] = {
                "response": "",
                "metadata": {"preparation_status": "missing_source_row"},
                "synthesized": True,
            }
            synthesized_blank_count += 1

    output_rows: list[dict[str, Any]] = []
    model_summaries: list[dict[str, Any]] = []
    for model_id in sorted(identities):
        model_family, model_revision = identities[model_id]
        blank_count = 0
        synthesized_for_model = 0
        for scenario_id in roster.scenario_ids:
            item = responses[model_id][scenario_id]
            response = item["response"]
            metadata = item["metadata"]
            blank_count += not response.strip()
            synthesized_for_model += item["synthesized"]
            output_row: dict[str, Any] = {
                "model_id": model_id,
                "model_family": model_family,
                "model_revision": model_revision,
                "scenario_id": scenario_id,
                "response": response,
            }
            if metadata:
                output_row["metadata"] = metadata
            output_rows.append(output_row)
        model_summaries.append(
            {
                "model_id": model_id,
                "model_family": model_family,
                "model_revision": model_revision,
                "row_count": len(roster.scenario_ids),
                "blank_count": blank_count,
                "synthesized_blank_count": synthesized_for_model,
                "all_blank": blank_count == len(roster.scenario_ids),
                "source_indices": sorted(model_source_indices[model_id]),
            }
        )

    output_payload = _serialize_jsonl(output_rows)
    output_sha256 = hashlib.sha256(output_payload).hexdigest()
    blank_count = sum(model["blank_count"] for model in model_summaries)
    all_blank_models = [model["model_id"] for model in model_summaries if model["all_blank"]]
    warnings: list[str] = []
    if synthesized_blank_count:
        warnings.append(f"synthesized {synthesized_blank_count} missing response row(s) as blank")
    if all_blank_models:
        warnings.append(f"{len(all_blank_models)} model(s) have only blank responses")
    report = {
        "schema_version": RESPONSE_BATCH_PREPARATION_REPORT_SCHEMA_VERSION,
        "response_schema_version": PRECOMPUTED_RESPONSE_BATCH_SCHEMA_VERSION,
        "source_manifest": {
            "path": str(manifest.path),
            "sha256": manifest.sha256,
        },
        "fitted_scenarios": {
            "path": str(roster.path),
            "sha256": roster.sha256,
            "count": len(roster.scenario_ids),
        },
        "inputs": source_summaries,
        "output": {
            "path": None,
            "sha256": output_sha256,
            "bytes": len(output_payload),
            "model_count": len(model_summaries),
            "row_count": len(output_rows),
            "scenario_count_per_model": len(roster.scenario_ids),
            "blank_count": blank_count,
            "synthesized_blank_count": synthesized_blank_count,
            "all_blank_model_ids": all_blank_models,
        },
        "models": model_summaries,
        "normalization": {
            "model_order": "lexicographic_model_id",
            "scenario_order": "fitted_scenario_file_order",
            "json_encoding": "utf-8-compact-sorted-keys-final-newline",
            "missing_policy": missing,
            "duplicate_policy": "error",
            "unknown_scenario_policy": "error",
            "blank_response_policy": "preserve",
        },
        "validation": {
            "status": "passed",
            "strict_json": True,
            "consistent_model_identity": True,
            "unique_model_scenario_pairs": True,
            "exact_fitted_bank_coverage_per_model": True,
        },
        "warnings": warnings,
    }
    protected = (manifest.path, roster.path, *(source.path for source in manifest.sources))
    return PreparedPrecomputedTutorResponseBatch(
        payload=output_payload,
        sha256=output_sha256,
        scenario_ids=roster.scenario_ids,
        protected_input_paths=tuple(protected),
        report=_freeze_json(report),
        report_sha256=_canonical_json_sha256(report),
        model_count=len(model_summaries),
        row_count=len(output_rows),
        blank_count=blank_count,
        synthesized_blank_count=synthesized_blank_count,
    )


def response_batch_preparation_report(
    prepared: PreparedPrecomputedTutorResponseBatch,
    output_path: str | Path,
) -> dict[str, Any]:
    """Return a plain-JSON report bound to the proposed or written output path."""

    report = _thaw_json(prepared.report)
    report["output"]["path"] = str(_resolved(output_path))
    return report


def _validate_prepared_binding(
    prepared: PreparedPrecomputedTutorResponseBatch,
) -> PreparedPrecomputedTutorResponseBatch:
    """Rebuild the preparation receipt from its protected source snapshot."""

    if hashlib.sha256(prepared.payload).hexdigest() != prepared.sha256:
        raise ResponsePreparationError("prepared response payload differs from its SHA-256")
    report = _thaw_json(prepared.report)
    normalization = report.get("normalization") if isinstance(report, Mapping) else None
    missing = normalization.get("missing_policy") if isinstance(normalization, Mapping) else None
    if missing not in ("error", "blank"):
        raise ResponsePreparationError(
            "prepared validation report has no valid missing-response policy"
        )
    if len(prepared.protected_input_paths) < 2:
        raise ResponsePreparationError(
            "prepared response batch has no authoritative manifest and scenario inputs"
        )
    try:
        manifest = load_response_batch_source_manifest(prepared.protected_input_paths[0])
        roster = load_fitted_scenario_roster(prepared.protected_input_paths[1])
        authoritative = build_precomputed_tutor_response_batch(
            manifest,
            roster,
            missing=missing,
        )
    except (OSError, TypeError, ValueError) as exc:
        if isinstance(exc, ResponsePreparationError):
            raise ResponsePreparationError(
                f"could not revalidate prepared response inputs: {exc}"
            ) from exc
        raise ResponsePreparationError(
            f"could not revalidate prepared response inputs: {exc}"
        ) from exc

    if prepared.payload != authoritative.payload or prepared.sha256 != authoritative.sha256:
        raise ResponsePreparationError(
            "authoritative response inputs no longer reproduce the prepared payload"
        )
    prepared_paths = tuple(_resolved(path) for path in prepared.protected_input_paths)
    authoritative_paths = tuple(_resolved(path) for path in authoritative.protected_input_paths)
    if prepared_paths != authoritative_paths:
        raise ResponsePreparationError(
            "prepared protected inputs differ from the authoritative source manifest"
        )
    if report != _thaw_json(authoritative.report):
        raise ResponsePreparationError(
            "prepared validation report differs from the authoritative source receipt"
        )
    if prepared.report_sha256 != authoritative.report_sha256:
        raise ResponsePreparationError(
            "prepared validation report hash differs from the authoritative source receipt"
        )
    expected_fields = (
        "scenario_ids",
        "model_count",
        "row_count",
        "blank_count",
        "synthesized_blank_count",
    )
    for field in expected_fields:
        if getattr(prepared, field) != getattr(authoritative, field):
            raise ResponsePreparationError(
                f"prepared {field} differs from the authoritative payload"
            )
    return authoritative


def _validate_output_paths(
    prepared: PreparedPrecomputedTutorResponseBatch,
    output_path: Path,
    report_path: Path,
    *,
    overwrite: bool,
) -> None:
    for path in (output_path, report_path):
        if path.is_symlink():
            raise ResponsePreparationError(f"output path must not be a symlink: {path}")
    resolved_output = _resolved(output_path)
    resolved_report = _resolved(report_path)
    if (
        resolved_output == resolved_report
        or resolved_output.name.casefold() == resolved_report.name.casefold()
        and resolved_output.parent == resolved_report.parent
        or _same_existing_file(resolved_output, resolved_report)
    ):
        raise ResponsePreparationError("response batch output and report must be different files")
    if resolved_output.parent != resolved_report.parent:
        raise ResponsePreparationError(
            "response batch output and report must use the same directory for "
            "crash-safe paired publication"
        )
    reserved_names = {
        _PUBLICATION_LOCK_NAME.casefold(),
        _PUBLICATION_JOURNAL_NAME.casefold(),
    }
    for destination in (resolved_output, resolved_report):
        if destination.name.casefold() in reserved_names:
            raise ResponsePreparationError(
                f"output path uses a reserved publication filename: {destination}"
            )
    for protected_path in prepared.protected_input_paths:
        protected = _resolved(protected_path)
        for destination in (resolved_output, resolved_report):
            if protected == destination or _same_existing_file(protected, destination):
                raise ResponsePreparationError(
                    f"refusing to overwrite a source input: {destination}"
                )
    for path in (resolved_output, resolved_report):
        if _path_exists(path) and not path.is_file():
            raise ResponsePreparationError(f"output path is not a regular file: {path}")
        if _path_exists(path) and not overwrite:
            raise ResponsePreparationError(f"output already exists: {path}")


def check_precomputed_tutor_response_batch_write(
    prepared: PreparedPrecomputedTutorResponseBatch,
    output_path: str | Path,
    report_path: str | Path,
    *,
    overwrite: bool = False,
) -> Mapping[str, Any]:
    """Validate write targets without creating directories or files."""

    output = Path(output_path).expanduser()
    report = Path(report_path).expanduser()
    authoritative = _validate_prepared_binding(prepared)
    _validate_output_paths(authoritative, output, report, overwrite=overwrite)
    return _freeze_json(response_batch_preparation_report(authoritative, output))


def _path_exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _same_existing_file(first: Path, second: Path) -> bool:
    if not _path_exists(first) or not _path_exists(second):
        return False
    try:
        return os.path.samefile(first, second)
    except OSError:
        return False


def _validate_publication_internal_paths(
    prepared: PreparedPrecomputedTutorResponseBatch,
    parent: Path,
) -> None:
    internal_paths = (
        parent / _PUBLICATION_LOCK_NAME,
        parent / _PUBLICATION_JOURNAL_NAME,
    )
    for protected_path in prepared.protected_input_paths:
        protected = _resolved(protected_path)
        for internal in internal_paths:
            if protected == internal or _same_existing_file(protected, internal):
                raise ResponsePreparationError(
                    f"source input uses a reserved publication path: {protected}"
                )


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:  # pragma: no cover - filesystem dependent.
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, payload: bytes) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)


def _write_new_file(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            path.unlink()
        raise


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextlib.contextmanager
def _publication_lock(parent: Path) -> Iterator[None]:
    parent.mkdir(parents=True, exist_ok=True)
    lock_path = parent / _PUBLICATION_LOCK_NAME
    if lock_path.is_symlink():
        raise ResponsePreparationError(f"publication lock must not be a symlink: {lock_path}")
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise ResponsePreparationError(f"could not safely open publication lock: {exc}") from exc
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise ResponsePreparationError(f"publication lock must be a regular file: {lock_path}")
    stream = os.fdopen(descriptor, "rb")
    try:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ResponsePreparationError(
                f"another response-batch publication is active in {parent}"
            ) from exc
        yield
    finally:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()


def _transaction_payload(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_transaction(journal_path: Path, transaction: Mapping[str, Any]) -> None:
    _atomic_write(journal_path, _transaction_payload(transaction))


def _transaction_entry_paths(
    parent: Path,
    entry: Mapping[str, Any],
) -> tuple[Path, Path, Path]:
    names: list[str] = []
    for field in ("destination", "staged", "backup"):
        name = entry.get(field)
        if not isinstance(name, str) or not name or Path(name).name != name:
            raise ResponsePreparationError(f"publication transaction has unsafe {field}")
        names.append(name)
    return parent / names[0], parent / names[1], parent / names[2]


def _rollback_capture_path(
    journal_path: Path,
    transaction: Mapping[str, Any],
    entry: Mapping[str, Any],
) -> Path:
    destination = _transaction_entry_paths(journal_path.parent, entry)[0]
    return journal_path.parent / (f".{destination.name}.{transaction['transaction_id']}.rollback")


def _load_transaction(journal_path: Path) -> dict[str, Any]:
    if journal_path.is_symlink() or not journal_path.is_file():
        raise ResponsePreparationError(
            f"publication transaction journal is not a regular file: {journal_path}"
        )
    try:
        value = json.loads(
            journal_path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise ResponsePreparationError(f"invalid publication transaction journal: {exc}") from exc
    if not isinstance(value, dict):
        raise ResponsePreparationError("publication transaction journal must contain one object")
    expected_fields = {"schema_version", "transaction_id", "state", "entries"}
    if set(value) != expected_fields:
        raise ResponsePreparationError("publication transaction journal fields differ")
    if value["schema_version"] != _PUBLICATION_TRANSACTION_SCHEMA_VERSION:
        raise ResponsePreparationError("unsupported publication transaction journal version")
    transaction_id = value["transaction_id"]
    if (
        not isinstance(transaction_id, str)
        or len(transaction_id) != 32
        or set(transaction_id) - _HEX_DIGITS
    ):
        raise ResponsePreparationError("publication transaction_id is invalid")
    if value["state"] not in {"staging", "prepared", "committed"}:
        raise ResponsePreparationError("publication transaction state is invalid")
    entries = value["entries"]
    if not isinstance(entries, list) or len(entries) != 2:
        raise ResponsePreparationError("publication transaction must contain two entries")
    destinations: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {
            "destination",
            "staged",
            "backup",
            "original_existed",
            "original_sha256",
            "new_sha256",
        }:
            raise ResponsePreparationError("publication transaction entry fields differ")
        destination, staged, backup = _transaction_entry_paths(journal_path.parent, entry)
        if destination.name in destinations:
            raise ResponsePreparationError("publication transaction destinations are duplicated")
        destinations.add(destination.name)
        if staged.name != f".{destination.name}.{transaction_id}.stage" or backup.name != (
            f".{destination.name}.{transaction_id}.backup"
        ):
            raise ResponsePreparationError("publication transaction temporary name is invalid")
        original_existed = entry["original_existed"]
        original_sha256 = entry["original_sha256"]
        if not isinstance(original_existed, bool):
            raise ResponsePreparationError("publication original_existed must be boolean")
        if original_existed != isinstance(original_sha256, str):
            raise ResponsePreparationError("publication original digest is inconsistent")
        for field in ("original_sha256", "new_sha256"):
            digest = entry[field]
            if digest is not None and (
                not isinstance(digest, str) or len(digest) != 64 or set(digest) - _HEX_DIGITS
            ):
                raise ResponsePreparationError(f"publication {field} is invalid")
        if not isinstance(entry["new_sha256"], str):
            raise ResponsePreparationError("publication new_sha256 is required")
    return value


def _regular_file_digest(path: Path) -> str | None:
    if not _path_exists(path):
        return None
    if path.is_symlink() or not path.is_file():
        raise ResponsePreparationError(f"publication artifact is not a regular file: {path}")
    return _sha256_file(path)


def _remove_transaction_file(path: Path) -> None:
    if not _path_exists(path):
        return
    if path.is_symlink() or not path.is_file():
        raise ResponsePreparationError(f"transaction temporary is not a regular file: {path}")
    path.unlink()


def _cleanup_committed_transaction(
    journal_path: Path,
    transaction: Mapping[str, Any],
) -> None:
    for entry in transaction["entries"]:
        destination, staged, backup = _transaction_entry_paths(journal_path.parent, entry)
        if _regular_file_digest(destination) != entry["new_sha256"]:
            raise ResponsePreparationError(
                f"committed publication artifact digest differs: {destination}"
            )
        staged_digest = _regular_file_digest(staged)
        if staged_digest not in {None, entry["new_sha256"]}:
            raise ResponsePreparationError(
                f"refusing to remove an unrelated transaction stage: {staged}"
            )
        backup_digest = _regular_file_digest(backup)
        expected_backup = entry["original_sha256"] if entry["original_existed"] else None
        if backup_digest not in {None, expected_backup}:
            raise ResponsePreparationError(
                f"refusing to remove an unrelated transaction backup: {backup}"
            )
        _remove_transaction_file(staged)
        _remove_transaction_file(backup)
    journal_path.unlink()
    _fsync_directory(journal_path.parent)


def _rollback_transaction(
    journal_path: Path,
    transaction: Mapping[str, Any],
) -> None:
    snapshots: list[_RollbackSnapshot] = []
    for entry in transaction["entries"]:
        destination, staged, backup = _transaction_entry_paths(journal_path.parent, entry)
        capture = _rollback_capture_path(journal_path, transaction, entry)
        backup_digest = _regular_file_digest(backup)
        destination_digest = _regular_file_digest(destination)
        staged_digest = _regular_file_digest(staged)
        capture_digest = _regular_file_digest(capture)
        if staged_digest not in {None, entry["new_sha256"]}:
            raise ResponsePreparationError(
                f"refusing to remove an unrelated transaction stage: {staged}"
            )
        allowed_capture_digests = {entry["new_sha256"]}
        if entry["original_existed"]:
            allowed_capture_digests.add(entry["original_sha256"])
        if capture_digest is not None and capture_digest not in allowed_capture_digests:
            raise ResponsePreparationError(
                f"refusing to remove unrelated rollback recovery data: {capture}"
            )
        needs_capture = False
        if entry["original_existed"]:
            if backup_digest is not None:
                if backup_digest != entry["original_sha256"]:
                    raise ResponsePreparationError(
                        f"publication backup digest differs; manual recovery required: {backup}"
                    )
                if destination_digest not in {
                    None,
                    entry["new_sha256"],
                    entry["original_sha256"],
                }:
                    raise ResponsePreparationError(
                        f"refusing to replace unrelated post-crash recovery data: {destination}"
                    )
                if capture_digest is not None and destination_digest not in {
                    None,
                    entry["original_sha256"],
                }:
                    raise ResponsePreparationError(
                        f"destination changed while rollback recovery was active: {destination}"
                    )
                needs_capture = (
                    capture_digest is None
                    and destination_digest == entry["new_sha256"]
                    and destination_digest != entry["original_sha256"]
                )
            elif destination_digest != entry["original_sha256"]:
                raise ResponsePreparationError(
                    f"original publication artifact cannot be recovered: {destination}"
                )
        elif destination_digest is not None and destination_digest != entry["new_sha256"]:
            raise ResponsePreparationError(
                f"refusing to remove an unrelated publication artifact: {destination}"
            )
        if not entry["original_existed"] and backup_digest is not None:
            raise ResponsePreparationError(
                f"unexpected backup exists for a new publication artifact: {backup}"
            )
        if not entry["original_existed"]:
            if capture_digest is not None and destination_digest is not None:
                raise ResponsePreparationError(
                    f"destination changed while rollback recovery was active: {destination}"
                )
            needs_capture = capture_digest is None and destination_digest is not None
        snapshots.append(
            _RollbackSnapshot(
                entry=entry,
                destination=destination,
                staged=staged,
                backup=backup,
                capture=capture,
                destination_digest=destination_digest,
                backup_digest=backup_digest,
                capture_digest=capture_digest,
                needs_capture=needs_capture,
            )
        )

    captured_now: list[_RollbackSnapshot] = []
    for snapshot in snapshots:
        if not snapshot.needs_capture:
            continue
        if _path_exists(snapshot.capture):
            raise ResponsePreparationError(
                f"rollback capture appeared during recovery: {snapshot.capture}"
            )
        os.replace(snapshot.destination, snapshot.capture)
        _fsync_directory(journal_path.parent)
        snapshot.capture_digest = _regular_file_digest(snapshot.capture)
        captured_now.append(snapshot)
        if snapshot.capture_digest != snapshot.destination_digest:
            restore_error: Exception | None = None
            for captured in reversed(captured_now):
                try:
                    if _path_exists(captured.destination):
                        raise ResponsePreparationError(
                            "could not restore a destination changed during rollback: "
                            f"{captured.destination}"
                        )
                    os.link(captured.capture, captured.destination)
                    _fsync_directory(journal_path.parent)
                    captured.capture.unlink()
                    _fsync_directory(journal_path.parent)
                    captured.capture_digest = None
                except Exception as exc:  # pragma: no cover - secondary concurrent mutation.
                    restore_error = exc
                    break
            if restore_error is not None:
                raise ResponsePreparationError(
                    "destination changed after rollback validation and its captured data "
                    f"requires manual recovery: {restore_error}"
                ) from restore_error
            raise ResponsePreparationError(
                f"destination changed after rollback validation: {snapshot.destination}"
            )

    # Install originals without replacing a name that another process may have
    # created after validation. Hard-linking is an atomic no-clobber operation;
    # the durable backup remains available until every destination is verified.
    for snapshot in reversed(snapshots):
        entry = snapshot.entry
        destination_digest = _regular_file_digest(snapshot.destination)
        if entry["original_existed"] and snapshot.backup_digest is not None:
            if destination_digest is None:
                if _regular_file_digest(snapshot.backup) != entry["original_sha256"]:
                    raise ResponsePreparationError(
                        f"publication backup changed during rollback: {snapshot.backup}"
                    )
                try:
                    os.link(snapshot.backup, snapshot.destination)
                except FileExistsError as exc:
                    raise ResponsePreparationError(
                        "refusing to replace a destination created during rollback: "
                        f"{snapshot.destination}"
                    ) from exc
                _fsync_directory(journal_path.parent)
            elif destination_digest != entry["original_sha256"]:
                raise ResponsePreparationError(
                    f"destination changed during rollback: {snapshot.destination}"
                )
        elif not entry["original_existed"] and destination_digest is not None:
            raise ResponsePreparationError(
                f"destination changed during rollback: {snapshot.destination}"
            )

    for snapshot in snapshots:
        expected = snapshot.entry["original_sha256"] if snapshot.entry["original_existed"] else None
        if _regular_file_digest(snapshot.destination) != expected:
            raise ResponsePreparationError(
                "publication rollback did not restore the original artifact: "
                f"{snapshot.destination}"
            )

    # Validate every transaction-owned temporary again before deleting any of
    # them. Destination data is already durable and no deletion targets it.
    for snapshot in snapshots:
        staged_digest = _regular_file_digest(snapshot.staged)
        if staged_digest not in {None, snapshot.entry["new_sha256"]}:
            raise ResponsePreparationError(
                f"refusing to remove an unrelated transaction stage: {snapshot.staged}"
            )
        backup_digest = _regular_file_digest(snapshot.backup)
        expected_backup = (
            snapshot.entry["original_sha256"] if snapshot.entry["original_existed"] else None
        )
        if backup_digest not in {None, expected_backup}:
            raise ResponsePreparationError(
                f"refusing to remove an unrelated transaction backup: {snapshot.backup}"
            )
        capture_digest = _regular_file_digest(snapshot.capture)
        allowed_capture_digests = {None, snapshot.entry["new_sha256"]}
        if snapshot.entry["original_existed"]:
            allowed_capture_digests.add(snapshot.entry["original_sha256"])
        if capture_digest not in allowed_capture_digests:
            raise ResponsePreparationError(
                f"refusing to remove unrelated rollback recovery data: {snapshot.capture}"
            )

    for snapshot in snapshots:
        _remove_transaction_file(snapshot.staged)
        _remove_transaction_file(snapshot.backup)
        _remove_transaction_file(snapshot.capture)
    _fsync_directory(journal_path.parent)
    journal_path.unlink()
    _fsync_directory(journal_path.parent)


def _recover_publication(journal_path: Path) -> None:
    if not _path_exists(journal_path):
        return
    transaction = _load_transaction(journal_path)
    if transaction["state"] == "committed":
        _cleanup_committed_transaction(journal_path, transaction)
        return
    final_digests = [
        _regular_file_digest(_transaction_entry_paths(journal_path.parent, entry)[0])
        for entry in transaction["entries"]
    ]
    if transaction["state"] == "prepared" and all(
        observed == entry["new_sha256"]
        for observed, entry in zip(final_digests, transaction["entries"], strict=True)
    ):
        transaction["state"] = "committed"
        _write_transaction(journal_path, transaction)
        _cleanup_committed_transaction(journal_path, transaction)
        return
    _rollback_transaction(journal_path, transaction)


def _new_transaction(
    output: Path,
    output_sha256: str,
    report: Path,
    report_sha256: str,
) -> dict[str, Any]:
    transaction_id = uuid.uuid4().hex
    entries: list[dict[str, Any]] = []
    for destination, digest in ((output, output_sha256), (report, report_sha256)):
        original_digest = _regular_file_digest(destination)
        entries.append(
            {
                "destination": destination.name,
                "staged": f".{destination.name}.{transaction_id}.stage",
                "backup": f".{destination.name}.{transaction_id}.backup",
                "original_existed": original_digest is not None,
                "original_sha256": original_digest,
                "new_sha256": digest,
            }
        )
    return {
        "schema_version": _PUBLICATION_TRANSACTION_SCHEMA_VERSION,
        "transaction_id": transaction_id,
        "state": "staging",
        "entries": entries,
    }


def _install_transaction(
    journal_path: Path,
    transaction: dict[str, Any],
    payloads: Sequence[bytes],
    scenario_ids: Sequence[str],
) -> None:
    _write_transaction(journal_path, transaction)
    for entry, payload in zip(transaction["entries"], payloads, strict=True):
        _, staged, _ = _transaction_entry_paths(journal_path.parent, entry)
        _write_new_file(staged, payload)
    _fsync_directory(journal_path.parent)
    staged_output = _transaction_entry_paths(journal_path.parent, transaction["entries"][0])[1]
    try:
        load_precomputed_tutor_response_batch(
            staged_output,
            expected_sha256=transaction["entries"][0]["new_sha256"],
            expected_scenario_ids=scenario_ids,
            _publication_lock_held=True,
        )
    except ValueError as exc:
        raise ResponsePreparationError(
            f"prepared response batch failed final contract validation: {exc}"
        ) from exc
    transaction["state"] = "prepared"
    _write_transaction(journal_path, transaction)
    for entry in transaction["entries"]:
        destination, staged, backup = _transaction_entry_paths(journal_path.parent, entry)
        if entry["original_existed"]:
            if _regular_file_digest(destination) != entry["original_sha256"]:
                raise ResponsePreparationError(f"output changed during publication: {destination}")
            os.replace(destination, backup)
            _fsync_directory(journal_path.parent)
            os.replace(staged, destination)
            _fsync_directory(journal_path.parent)
        else:
            # Hard-link installation is an atomic no-clobber operation.  It
            # closes the existence-check/replace race even without --overwrite.
            os.link(staged, destination)
            _fsync_directory(journal_path.parent)
            staged.unlink()
            _fsync_directory(journal_path.parent)
    _fsync_directory(journal_path.parent)
    for entry in transaction["entries"]:
        destination = _transaction_entry_paths(journal_path.parent, entry)[0]
        if _regular_file_digest(destination) != entry["new_sha256"]:
            raise ResponsePreparationError(
                f"published artifact failed digest verification: {destination}"
            )
    transaction["state"] = "committed"
    _write_transaction(journal_path, transaction)
    _cleanup_committed_transaction(journal_path, transaction)


def write_precomputed_tutor_response_batch(
    prepared: PreparedPrecomputedTutorResponseBatch,
    output_path: str | Path,
    report_path: str | Path,
    *,
    overwrite: bool = False,
) -> WrittenPrecomputedTutorResponseBatch:
    """Crash-safely publish a validated response batch and its bound report."""

    prepared = _validate_prepared_binding(prepared)
    raw_output = Path(output_path).expanduser()
    raw_report_output = Path(report_path).expanduser()
    for path in (raw_output, raw_report_output):
        if path.is_symlink():
            raise ResponsePreparationError(f"output path must not be a symlink: {path}")
    output = _resolved(raw_output)
    report_output = _resolved(raw_report_output)
    if output.parent != report_output.parent:
        _validate_output_paths(prepared, output, report_output, overwrite=overwrite)
    _validate_publication_internal_paths(prepared, output.parent)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = response_batch_preparation_report(prepared, output)
    report_payload = (
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    report_sha256 = hashlib.sha256(report_payload).hexdigest()
    journal_path = output.parent / _PUBLICATION_JOURNAL_NAME
    with _publication_lock(output.parent):
        _recover_publication(journal_path)
        _validate_output_paths(prepared, output, report_output, overwrite=overwrite)
        transaction = _new_transaction(
            output,
            prepared.sha256,
            report_output,
            report_sha256,
        )
        try:
            _install_transaction(
                journal_path,
                transaction,
                (prepared.payload, report_payload),
                prepared.scenario_ids,
            )
        except Exception as exc:
            try:
                _recover_publication(journal_path)
            except Exception as recovery_exc:
                raise ResponsePreparationError(
                    "response-batch publication failed and automatic recovery could not "
                    f"complete: {recovery_exc}"
                ) from exc
            if isinstance(exc, ResponsePreparationError):
                raise
            raise ResponsePreparationError(f"could not publish response batch: {exc}") from exc

    return WrittenPrecomputedTutorResponseBatch(
        output_path=output,
        report_path=report_output,
        sha256=prepared.sha256,
        report_sha256=report_sha256,
        report=_freeze_json(report),
    )


__all__ = [
    "BATCH_JSONL_SOURCE_FORMAT",
    "FittedScenarioRoster",
    "MissingResponsePolicy",
    "PreparedPrecomputedTutorResponseBatch",
    "RESPONSE_BATCH_PREPARATION_REPORT_SCHEMA_VERSION",
    "RESPONSE_BATCH_SOURCE_FORMATS",
    "ResponseBatchSource",
    "ResponseBatchSourceManifest",
    "ResponsePreparationError",
    "SINGLE_JSONL_SOURCE_FORMAT",
    "SourceFormat",
    "TUTORBENCH_OUTPUT_JSONL_SOURCE_FORMAT",
    "WrittenPrecomputedTutorResponseBatch",
    "build_precomputed_tutor_response_batch",
    "check_precomputed_tutor_response_batch_write",
    "load_fitted_scenario_roster",
    "load_response_batch_source_manifest",
    "response_batch_preparation_report",
    "write_precomputed_tutor_response_batch",
]
