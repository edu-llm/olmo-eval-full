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
PRECOMPUTED_RESPONSE_BATCH_SCHEMA_VERSION = "edullm-precomputed-tutor-response-batch-v1"
_ROW_REQUIRED_FIELDS = frozenset({"scenario_id", "response"})
_ROW_ALLOWED_FIELDS = frozenset({"scenario_id", "response", "metadata"})
_BATCH_ROW_IDENTITY_FIELDS = ("model_id", "model_family", "model_revision")
_BATCH_ROW_REQUIRED_FIELDS = frozenset({*_BATCH_ROW_IDENTITY_FIELDS, "scenario_id", "response"})
_BATCH_ROW_ALLOWED_FIELDS = _BATCH_ROW_REQUIRED_FIELDS | {"metadata"}


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


def load_precomputed_tutor_response_batch(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_scenario_ids: Sequence[str],
) -> PrecomputedTutorResponseBatch:
    """Load complete response sets for one or more explicitly identified tutor models."""

    source = Path(path).expanduser()
    if not source.is_file():
        raise ValueError(f"precomputed tutor response batch file does not exist: {source}")
    try:
        payload = source.read_bytes()
    except OSError as exc:
        raise ValueError(
            f"could not read precomputed tutor response batch file {source}: {exc}"
        ) from exc
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

    lines = text.splitlines()
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
            metadata=MappingProxyType(normalized_metadata),
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
    "PrecomputedTutorResponse",
    "PrecomputedTutorResponseBatch",
    "PrecomputedTutorResponseBatchModel",
    "PrecomputedTutorResponses",
    "load_precomputed_tutor_response_batch",
    "load_precomputed_tutor_responses",
]
