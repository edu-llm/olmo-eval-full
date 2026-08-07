"""Strict input contract for externally generated EduLLM tutor responses."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

PRECOMPUTED_RESPONSES_SCHEMA_VERSION = "edullm-precomputed-tutor-responses-v1"
_ROW_REQUIRED_FIELDS = frozenset({"scenario_id", "response"})
_ROW_ALLOWED_FIELDS = frozenset({"scenario_id", "response", "metadata"})


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


def _canonical_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_precomputed_tutor_responses(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_scenario_ids: Sequence[str],
) -> PrecomputedTutorResponses:
    """Load a complete uploaded response set and fail closed on any mismatch."""

    source = Path(path).expanduser()
    if not source.is_file():
        raise ValueError(f"precomputed tutor response file does not exist: {source}")
    try:
        payload = source.read_bytes()
    except OSError as exc:
        raise ValueError(f"could not read precomputed tutor response file {source}: {exc}") from exc
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
    lines = text.splitlines()
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
            metadata=MappingProxyType(normalized_metadata),
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


__all__ = [
    "PRECOMPUTED_RESPONSES_SCHEMA_VERSION",
    "PrecomputedTutorResponse",
    "PrecomputedTutorResponses",
    "load_precomputed_tutor_responses",
]
