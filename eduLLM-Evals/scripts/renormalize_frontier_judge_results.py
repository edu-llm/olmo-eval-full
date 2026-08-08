#!/usr/bin/env python3
"""Recover unambiguous embedded frontier judgments without calling an API.

This is a copy-on-write postprocessor for a *completed* 18-wave frontier
judge result tree.  It never changes the source tree.  Every output row is
upgraded to one v3 normalization, while accepted source rows remain
immutable.  For a non-ok current row, current and archived retry attempts are
considered in numeric attempt order.  The first safely recoverable judgment
wins.  Recovery accepts exact or terminal embedded JSON and one narrowly
specified repair: doubling illegal JSON-escape backslashes inside strings.
It never repairs structure, truncation, quotes, fields, or verdicts.  Every
candidate is passed through the frontier runner's exact parser, so the
existing schema and evidence gate stays authoritative.

Typical usage::

    python scripts/renormalize_frontier_judge_results.py \
      runs/frontier_judge_validation/results \
      runs/frontier_judge_validation/results_normalized_v3

The command is offline: it does not load ``.env``, construct an API client, or
make network requests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
import shutil
import sys
import tempfile
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

try:  # Works as both ``python scripts/...`` and a test import.
    from scripts import run_frontier_judge_validation as frontier
except ImportError:  # pragma: no cover - exercised by direct script execution.
    import run_frontier_judge_validation as frontier


SOURCE_NORMALIZATION_VERSION = "frontier-judge-normalization-v1"
NORMALIZATION_VERSION = "frontier-judge-normalization-v3"
RECOVERY_SCHEMA_VERSION = "frontier-judge-recovery-v2"
RECOVERY_POLICY_VERSION = "earliest-attempt-terminal-json-illegal-escape-v1"
SUMMARY_SCHEMA_VERSION = "frontier-judge-renormalization-summary-v2"
ILLEGAL_ESCAPE_REPAIR_VERSION = "illegal-json-string-backslash-doubling-v1"
FINAL_SOURCE_STATUSES = {"complete", "complete_with_errors"}
ROW_STATUSES = {"ok", "parse_error", "generation_error"}
JUDGMENT_KEY_PATTERN = re.compile(
    r'(?i)(?:"(?:verdict|rationale|evidence)"|\bverdict)\s*:'
)
RETRY_IDENTITY_FIELDS = (
    "case_id",
    "response_id",
    "scenario_id",
    "criterion_id",
    "candidate_family",
    "judge_name",
    "judge_family",
    "judge_model",
    "judge_revision",
    "served_model",
    "resolved_provider_model",
    "provider",
    "backend",
    "base_url",
    "api_surface",
    "request_token_param",
    "request_temperature",
    "checkpoint_provenance",
    "checkpoint_verified_by_runner",
    "adapter",
    "prompt_version",
    "evidence_policy_version",
    "prompt_variant",
    "replicate_id",
    "normalization_version",
    "routing_version",
    "configuration_hash",
    "frozen_configuration_hash",
    "input_hash",
    "prompt_hash",
)


@dataclass(frozen=True)
class JsonSpan:
    start: int
    end: int
    value: object


@dataclass(frozen=True)
class RecoveryResult:
    action: str
    candidate_count: int
    parsed: frontier.base.ParsedJudgment | None = None
    selected_start: int | None = None
    selected_end: int | None = None
    selected_sha256: str | None = None
    detail: str | None = None
    method: str | None = None
    repaired_sha256: str | None = None
    illegal_escape_offsets: tuple[int, ...] = ()


@dataclass(frozen=True)
class AttemptSource:
    row: dict
    attempt: int
    artifact_kind: str
    artifact_path: Path
    artifact_sha256: str
    row_sha256: str
    raw_output_sha256: str


@dataclass(frozen=True)
class AttemptSelection:
    action: str
    recovery: RecoveryResult
    selected: AttemptSource | None
    considered: tuple[tuple[AttemptSource, RecoveryResult], ...]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected one JSON object")
    return value


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    try:
        handle = path.open(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"could not open {path}: {exc}") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected one JSON object")
            case_id = str(row.get("case_id") or "").strip()
            if not case_id or case_id in seen:
                raise ValueError(
                    f"{path}:{line_number}: blank or duplicate case_id {case_id!r}"
                )
            seen.add(case_id)
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: contains no rows")
    return rows


def load_retry_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    try:
        handle = path.open(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"could not open {path}: {exc}") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected one JSON object")
            if not str(row.get("case_id") or "").strip():
                raise ValueError(f"{path}:{line_number}: blank case_id")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: contains no retry rows")
    return rows


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _decoded_json_spans(raw: str) -> list[JsonSpan]:
    """Return every JSON value decodable from an opening container token."""

    decoder = json.JSONDecoder()
    spans: list[JsonSpan] = []
    seen: set[tuple[int, int]] = set()
    for start, character in enumerate(raw):
        if character not in "[{":
            continue
        try:
            value, end = decoder.raw_decode(raw, start)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        key = (start, end)
        if key not in seen:
            seen.add(key)
            spans.append(JsonSpan(start=start, end=end, value=value))
    return spans


class DuplicateJsonKeyError(ValueError):
    """Raised when a candidate relies on JSON's last-key-wins behavior."""


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateJsonKeyError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _has_duplicate_keys(candidate: str) -> bool:
    try:
        json.loads(candidate, object_pairs_hook=_reject_duplicate_keys)
    except DuplicateJsonKeyError:
        return True
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    return False


def _has_unclosed_json_container(prefix: str) -> bool:
    """Conservatively reject a candidate nested in malformed outer JSON."""

    stack: list[str] = []
    in_string = False
    escaped = False
    pairs = {"}": "{", "]": "["}
    for character in prefix:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "[{":
            stack.append(character)
        elif character in "]}" and stack:
            if stack[-1] != pairs[character]:
                return True
            stack.pop()
    return bool(stack)


def _is_terminal_payload(raw: str, selected: JsonSpan) -> bool:
    prefix = raw[: selected.start]
    suffix = raw[selected.end :]
    if not suffix.strip():
        return not _has_unclosed_json_container(prefix)
    if re.fullmatch(r"\s*```\s*", suffix) is None:
        return False
    if re.search(r"```(?:json)?[ \t]*\r?\n[ \t]*$", prefix, re.IGNORECASE) is None:
        return False
    return not _has_unclosed_json_container(prefix)


def recover_embedded_judgment(raw_output: str) -> RecoveryResult:
    """Extract one unambiguous object and validate it with the exact v1 parser."""

    spans = _decoded_json_spans(raw_output)
    valid: list[tuple[JsonSpan, frontier.base.ParsedJudgment]] = []
    for span in spans:
        if not isinstance(span.value, dict):
            continue
        candidate = raw_output[span.start : span.end]
        if _has_duplicate_keys(candidate):
            continue
        parsed = frontier.parse_frontier_judgment(candidate)
        if parsed.status == "ok" and parsed.verdict in {"pass", "fail"}:
            valid.append((span, parsed))

    if not valid:
        return RecoveryResult(
            action="unresolved_no_valid_object",
            candidate_count=0,
            detail="no schema-valid embedded judgment JSON object",
        )
    if len(valid) != 1:
        return RecoveryResult(
            action="unresolved_ambiguous",
            candidate_count=len(valid),
            detail="multiple schema-valid embedded judgment JSON objects",
        )

    selected, parsed = valid[0]
    if any(
        other.start < selected.start
        and selected.end <= other.end
        and (other.start, other.end) != (selected.start, selected.end)
        for other in spans
    ):
        return RecoveryResult(
            action="unresolved_ambiguous",
            candidate_count=1,
            detail="valid judgment is nested inside another JSON value",
        )

    outside = raw_output[: selected.start] + raw_output[selected.end :]
    if JUDGMENT_KEY_PATTERN.search(outside):
        return RecoveryResult(
            action="unresolved_ambiguous",
            candidate_count=1,
            detail="another judgment-like field appears outside the valid object",
        )

    if not _is_terminal_payload(raw_output, selected):
        return RecoveryResult(
            action="unresolved_nonterminal",
            candidate_count=1,
            detail="valid judgment is not the unique terminal payload",
        )

    selected_text = raw_output[selected.start : selected.end]
    return RecoveryResult(
        action="recovered_embedded_json",
        candidate_count=1,
        parsed=parsed,
        method="embedded_terminal_json",
        selected_start=selected.start,
        selected_end=selected.end,
        selected_sha256=text_sha256(selected_text),
    )


def _lexical_container_end(raw: str, start: int) -> int | None:
    """Find a balanced JSON-like container without repairing its contents."""

    if start >= len(raw) or raw[start] not in "[{":
        return None
    stack = [raw[start]]
    pairs = {"}": "{", "]": "["}
    in_string = False
    index = start + 1
    while index < len(raw):
        character = raw[index]
        if in_string:
            if character == "\\":
                if index + 1 >= len(raw):
                    return None
                index += 2
                continue
            if character == '"':
                in_string = False
            index += 1
            continue
        if character == '"':
            in_string = True
        elif character in "[{":
            stack.append(character)
        elif character in "]}":
            if not stack or stack[-1] != pairs[character]:
                return None
            stack.pop()
            if not stack:
                return index + 1
        index += 1
    return None


def _lexical_json_object_spans(raw: str) -> list[JsonSpan]:
    spans: list[JsonSpan] = []
    seen: set[tuple[int, int]] = set()
    for start, character in enumerate(raw):
        if character != "{":
            continue
        end = _lexical_container_end(raw, start)
        if end is not None and (start, end) not in seen:
            seen.add((start, end))
            spans.append(JsonSpan(start=start, end=end, value=None))
    return spans


def _repair_illegal_json_string_escapes(
    candidate: str,
) -> tuple[str, tuple[int, ...]]:
    """Double only illegal escape backslashes occurring inside JSON strings."""

    repaired: list[str] = []
    offsets: list[int] = []
    in_string = False
    index = 0
    simple_escapes = {'"', "\\", "/", "b", "f", "n", "r", "t"}
    while index < len(candidate):
        character = candidate[index]
        if not in_string:
            repaired.append(character)
            if character == '"':
                in_string = True
            index += 1
            continue
        if character == '"':
            repaired.append(character)
            in_string = False
            index += 1
            continue
        if character != "\\":
            repaired.append(character)
            index += 1
            continue

        following = candidate[index + 1] if index + 1 < len(candidate) else None
        valid_unicode = (
            following == "u"
            and index + 5 < len(candidate)
            and all(
                character in "0123456789abcdefABCDEF"
                for character in candidate[index + 2 : index + 6]
            )
        )
        if valid_unicode:
            repaired.append(candidate[index : index + 6])
            index += 6
            continue
        if following in simple_escapes:
            repaired.append(candidate[index : index + 2])
            index += 2
            continue
        repaired.append("\\\\")
        offsets.append(index)
        index += 1
    return "".join(repaired), tuple(offsets)


def recover_illegal_escape_judgment(raw_output: str) -> RecoveryResult:
    """Repair only illegal in-string escapes, then apply the exact parser."""

    spans = _lexical_json_object_spans(raw_output)
    valid: list[
        tuple[JsonSpan, frontier.base.ParsedJudgment, str, tuple[int, ...]]
    ] = []
    for span in spans:
        candidate = raw_output[span.start : span.end]
        repaired, offsets = _repair_illegal_json_string_escapes(candidate)
        if not offsets or _has_duplicate_keys(repaired):
            continue
        parsed = frontier.parse_frontier_judgment(repaired)
        if parsed.status == "ok" and parsed.verdict in {"pass", "fail"}:
            valid.append((span, parsed, repaired, offsets))

    if not valid:
        return RecoveryResult(
            action="unresolved_no_illegal_escape_repair",
            candidate_count=0,
            detail="no terminal judgment is recoverable by illegal-escape repair",
        )
    if len(valid) != 1:
        return RecoveryResult(
            action="unresolved_ambiguous",
            candidate_count=len(valid),
            detail="multiple judgments become valid after illegal-escape repair",
        )

    selected, parsed, repaired, relative_offsets = valid[0]
    if any(
        other.start < selected.start
        and selected.end <= other.end
        and (other.start, other.end) != (selected.start, selected.end)
        for other in spans
    ):
        return RecoveryResult(
            action="unresolved_ambiguous",
            candidate_count=1,
            detail="repairable judgment is nested inside another JSON-like value",
        )
    outside = raw_output[: selected.start] + raw_output[selected.end :]
    if JUDGMENT_KEY_PATTERN.search(outside):
        return RecoveryResult(
            action="unresolved_ambiguous",
            candidate_count=1,
            detail="another judgment-like field appears outside the repairable object",
        )
    if not _is_terminal_payload(raw_output, selected):
        return RecoveryResult(
            action="unresolved_nonterminal",
            candidate_count=1,
            detail="repairable judgment is not the unique terminal payload",
        )

    original = raw_output[selected.start : selected.end]
    return RecoveryResult(
        action="recovered_illegal_escape_json",
        candidate_count=1,
        parsed=parsed,
        method="illegal_escape_repair",
        selected_start=selected.start,
        selected_end=selected.end,
        selected_sha256=text_sha256(original),
        repaired_sha256=text_sha256(repaired),
        illegal_escape_offsets=tuple(
            selected.start + offset for offset in relative_offsets
        ),
    )


def evaluate_attempt(
    raw_output: str,
    *,
    allow_illegal_escape_repair: bool = True,
) -> RecoveryResult:
    embedded = recover_embedded_judgment(raw_output)
    exact = frontier.parse_frontier_judgment(raw_output)
    if exact.status == "ok" and embedded.parsed is not None:
        return replace(
            embedded,
            action="recovered_exact_json",
            parsed=exact,
            method="exact_json",
        )
    if embedded.parsed is not None:
        return embedded
    if not allow_illegal_escape_repair:
        return embedded
    repaired = recover_illegal_escape_judgment(raw_output)
    if repaired.parsed is not None:
        return repaired
    if repaired.action in {"unresolved_ambiguous", "unresolved_nonterminal"}:
        return repaired
    return embedded


def discover_waves(
    source_root: Path,
) -> list[tuple[str, str, Path, Path, Path | None]]:
    expected = {
        (judge, wave)
        for judge in frontier.FRONTIER_JUDGES
        for wave in frontier.WAVES
    }
    discovered: dict[tuple[str, str], tuple[Path, Path]] = {}
    for source_path in sorted(source_root.glob("*/*/*.jsonl")):
        judge = source_path.parent.parent.name
        wave = source_path.parent.name
        if source_path.name != f"{wave}.jsonl":
            continue
        key = (judge, wave)
        if key not in expected:
            raise ValueError(f"unexpected frontier result wave: {source_path}")
        manifest_path = source_path.with_suffix(".manifest.json")
        if not manifest_path.is_file():
            raise ValueError(f"missing source manifest: {manifest_path}")
        discovered[key] = (source_path, manifest_path)

    actual = set(discovered)
    if actual != expected:
        missing = [f"{judge}/{wave}" for judge, wave in sorted(expected - actual)]
        extra = [f"{judge}/{wave}" for judge, wave in sorted(actual - expected)]
        raise ValueError(
            "source must contain the completed 3-judge x 6-wave frontier suite; "
            f"missing={missing}, extra={extra}"
        )
    return [
        (
            judge,
            wave,
            *discovered[(judge, wave)],
            (
                discovered[(judge, wave)][0].with_name(
                    f"{wave}.retry_history.jsonl"
                )
                if discovered[(judge, wave)][0]
                .with_name(f"{wave}.retry_history.jsonl")
                .is_file()
                else None
            ),
        )
        for judge in frontier.FRONTIER_JUDGES
        for wave in frontier.WAVES
    ]


def _require_equal(path: Path, field: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise ValueError(
            f"{path}: {field} does not match source configuration; "
            f"expected={expected!r}, got={actual!r}"
        )


def validate_source_wave(
    *,
    judge: str,
    wave: str,
    source_path: Path,
    manifest_path: Path,
    rows: Sequence[dict],
    manifest: dict,
) -> tuple[dict, str, str]:
    configuration = manifest.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError(f"{manifest_path}: missing configuration object")
    if manifest.get("status") not in FINAL_SOURCE_STATUSES:
        raise ValueError(f"{manifest_path}: source wave is not complete")

    source_configuration_hash = str(manifest.get("configuration_hash") or "")
    if frontier.base.stable_hash(configuration) != source_configuration_hash:
        raise ValueError(f"{manifest_path}: configuration_hash does not verify")
    source_frozen_hash = str(manifest.get("frozen_configuration_hash") or "")
    if frontier.base.frozen_configuration_hash(configuration) != source_frozen_hash:
        raise ValueError(f"{manifest_path}: frozen_configuration_hash does not verify")

    expected_variant, expected_replicate = frontier.WAVES[wave]
    spec = frontier.FRONTIER_JUDGES[judge]
    expected_configuration = {
        "judge_name": judge,
        "judge_family": spec.family,
        "adapter": frontier.ADAPTER,
        "prompt_version": frontier.base.PROMPT_VERSION,
        "evidence_policy_version": frontier.base.EVIDENCE_POLICY_VERSION,
        "normalization_version": SOURCE_NORMALIZATION_VERSION,
        "routing_version": frontier.ROUTING_VERSION,
        "prompt_variant": expected_variant,
        "replicate_id": expected_replicate,
    }
    for field, expected_value in expected_configuration.items():
        _require_equal(
            manifest_path, field, configuration.get(field), expected_value
        )

    manifest_checks = {
        "judge_name": judge,
        "judge_family": spec.family,
        "wave": wave,
        "prompt_variant": expected_variant,
        "replicate_id": expected_replicate,
        "configuration_hash": source_configuration_hash,
        "frozen_configuration_hash": source_frozen_hash,
    }
    for field, expected_value in manifest_checks.items():
        _require_equal(manifest_path, field, manifest.get(field), expected_value)

    status_counts: Counter[str] = Counter()
    usable = 0
    resolved_models: set[str] = set()
    for row_number, row in enumerate(rows, start=1):
        location = Path(f"{source_path}:{row_number}")
        if "normalization_recovery" in row:
            raise ValueError(f"{location}: source row is already postprocessed")
        if not isinstance(row.get("raw_output"), str):
            raise ValueError(f"{location}: raw_output must be a string")

        source_status = str(row.get("status") or "")
        source_verdict = str(row.get("verdict") or "")
        if source_status not in ROW_STATUSES:
            raise ValueError(f"{location}: unsupported status {source_status!r}")
        if source_status == "ok":
            if source_verdict not in {"pass", "fail"}:
                raise ValueError(f"{location}: status=ok requires pass/fail")
            exact = frontier.parse_frontier_judgment(row["raw_output"])
            expected_values = {
                "verdict": exact.verdict,
                "native_score": exact.native_score,
                "rationale": exact.rationale,
                "evidence": exact.evidence,
                "status": exact.status,
                "error": exact.error,
            }
            if exact.status != "ok" or any(
                row.get(field) != value for field, value in expected_values.items()
            ):
                raise ValueError(
                    f"{location}: accepted source judgment does not match the exact parser"
                )
            usable += 1
        elif source_verdict != "no_decision":
            raise ValueError(f"{location}: non-ok status requires no_decision")
        elif source_status == "parse_error":
            exact = frontier.parse_frontier_judgment(row["raw_output"])
            if exact.status == "ok":
                raise ValueError(
                    f"{location}: source parse_error is already accepted by exact parser"
                )

        status_counts[source_status] += 1
        resolved = str(row.get("resolved_provider_model") or "").strip()
        if resolved:
            resolved_models.add(resolved)

        row_checks = {
            "judge_name": judge,
            "judge_family": spec.family,
            "judge_model": configuration.get("model_id"),
            "judge_revision": configuration.get("revision"),
            "backend": configuration.get("backend"),
            "base_url": configuration.get("base_url"),
            "adapter": configuration.get("adapter"),
            "prompt_version": configuration.get("prompt_version"),
            "evidence_policy_version": configuration.get("evidence_policy_version"),
            "normalization_version": SOURCE_NORMALIZATION_VERSION,
            "routing_version": configuration.get("routing_version"),
            "prompt_variant": expected_variant,
            "replicate_id": expected_replicate,
            "configuration_hash": source_configuration_hash,
            "frozen_configuration_hash": source_frozen_hash,
        }
        for field, expected_value in row_checks.items():
            _require_equal(location, field, row.get(field), expected_value)

    no_decision = len(rows) - usable
    missing_resolved_model_rows = sum(
        not str(row.get("resolved_provider_model") or "").strip() for row in rows
    )
    count_checks = {
        "eligible_case_count": len(rows),
        "usable_decisions": usable,
        "no_decision_rows": no_decision,
        "resolved_provider_models": sorted(resolved_models),
        "missing_resolved_model_rows": missing_resolved_model_rows,
        "model_provenance_consistent": True,
    }
    for field, expected_value in count_checks.items():
        _require_equal(manifest_path, field, manifest.get(field), expected_value)
    if len(resolved_models) != 1:
        raise ValueError(f"{manifest_path}: source wave has model drift or no model ID")
    if missing_resolved_model_rows:
        raise ValueError(
            f"{manifest_path}: source wave has missing resolved model provenance"
        )
    expected_status = "complete" if no_decision == 0 else "complete_with_errors"
    if manifest.get("status") != expected_status:
        raise ValueError(
            f"{manifest_path}: final status does not match source decision counts"
        )

    source_output_hash = frontier.base.file_sha256(source_path)
    source_manifest_hash = frontier.base.file_sha256(manifest_path)
    return configuration, source_output_hash, source_manifest_hash


def _numeric_attempt(value: object, *, location: str) -> int:
    if type(value) is int and value >= 1:
        return value
    if isinstance(value, str) and re.fullmatch(r"[1-9][0-9]*", value.strip()):
        return int(value.strip())
    raise ValueError(f"{location}: attempt must be a positive integer")


def validate_retry_history(
    path: Path,
    *,
    primary_rows: Sequence[dict],
) -> tuple[dict[str, list[AttemptSource]], dict]:
    artifact_hash_before = frontier.base.file_sha256(path)
    history_rows = load_retry_jsonl(path)
    artifact_hash_after = frontier.base.file_sha256(path)
    if artifact_hash_before != artifact_hash_after:
        raise ValueError(f"retry history changed while it was read: {path}")

    primary_by_id = {str(row["case_id"]): row for row in primary_rows}
    groups: dict[str, list[AttemptSource]] = {}
    seen_attempts: set[tuple[str, int]] = set()
    for line_number, row in enumerate(history_rows, start=1):
        location = f"{path}:{line_number}"
        case_id = str(row.get("case_id") or "").strip()
        primary = primary_by_id.get(case_id)
        if primary is None:
            raise ValueError(f"{location}: retry history has unknown case_id {case_id!r}")
        for field in RETRY_IDENTITY_FIELDS:
            if field not in row or field not in primary:
                raise ValueError(f"{location}: missing retry identity field {field!r}")
            if row[field] != primary[field]:
                raise ValueError(
                    f"{location}: retry identity field {field!r} differs from primary"
                )
        if row.get("status") == "ok" or row.get("verdict") != "no_decision":
            raise ValueError(
                f"{location}: retry history must contain only archived non-ok rows"
            )
        if not isinstance(row.get("raw_output"), str):
            raise ValueError(f"{location}: raw_output must be a string")
        if not str(row.get("retry_archived_at") or "").strip():
            raise ValueError(f"{location}: retry_archived_at is blank")

        attempt = _numeric_attempt(row.get("attempt"), location=location)
        current_attempt = _numeric_attempt(
            primary.get("attempt"), location=f"{path}/{case_id}/primary"
        )
        if attempt >= current_attempt:
            raise ValueError(
                f"{location}: archived attempt {attempt} must precede current "
                f"attempt {current_attempt}"
            )
        key = (case_id, attempt)
        if key in seen_attempts:
            raise ValueError(
                f"{location}: duplicate archived attempt {attempt} for {case_id}"
            )
        seen_attempts.add(key)
        groups.setdefault(case_id, []).append(
            AttemptSource(
                row=row,
                attempt=attempt,
                artifact_kind="retry_history",
                artifact_path=path,
                artifact_sha256=artifact_hash_after,
                row_sha256=frontier.base.stable_hash(row),
                raw_output_sha256=text_sha256(row["raw_output"]),
            )
        )

    for case_id, attempts in groups.items():
        attempts.sort(key=lambda item: item.attempt)
        current_attempt = _numeric_attempt(
            primary_by_id[case_id].get("attempt"),
            location=f"{path}/{case_id}/primary",
        )
        actual_attempts = [item.attempt for item in attempts]
        expected_attempts = list(range(1, current_attempt))
        if actual_attempts != expected_attempts:
            raise ValueError(
                f"{path}/{case_id}: archived attempts must be contiguous; "
                f"expected={expected_attempts}, got={actual_attempts}"
            )
    return groups, {
        "file": str(path),
        "sha256": artifact_hash_after,
        "row_count": len(history_rows),
        "case_count": len(groups),
    }


def _attempt_audit(
    source: AttemptSource,
    evaluation: RecoveryResult,
) -> dict:
    return {
        "attempt": source.attempt,
        "artifact_kind": source.artifact_kind,
        "artifact_file": str(source.artifact_path),
        "artifact_sha256": source.artifact_sha256,
        "row_sha256": source.row_sha256,
        "raw_output_sha256": source.raw_output_sha256,
        "recorded_status": source.row.get("status"),
        "recorded_verdict": source.row.get("verdict"),
        "evaluation_action": evaluation.action,
        "recovery_method": evaluation.method,
        "normalized_verdict": (
            evaluation.parsed.verdict if evaluation.parsed is not None else None
        ),
        "candidate_count": evaluation.candidate_count,
        "selected_object_sha256": evaluation.selected_sha256,
        "repaired_object_sha256": evaluation.repaired_sha256,
        "illegal_escape_offsets": list(evaluation.illegal_escape_offsets),
    }


def select_attempt(
    primary: AttemptSource,
    history: Sequence[AttemptSource],
) -> AttemptSelection:
    if primary.row.get("status") == "ok":
        recovery = RecoveryResult(
            action="unchanged_accepted",
            candidate_count=0,
            method="source_accepted",
        )
        return AttemptSelection(
            action="unchanged_accepted",
            recovery=recovery,
            selected=primary,
            considered=((primary, recovery),),
        )

    sources = [*history, primary]
    sources.sort(key=lambda item: item.attempt)
    if len({item.attempt for item in sources}) != len(sources):
        raise ValueError(f"duplicate attempt number for {primary.row['case_id']}")
    considered = tuple(
        (
            item,
            evaluate_attempt(
                item.row["raw_output"],
                allow_illegal_escape_repair=(item.artifact_kind == "primary"),
            ),
        )
        for item in sources
    )
    for item, evaluation in considered:
        if evaluation.parsed is not None:
            return AttemptSelection(
                action=f"recovered_from_{item.artifact_kind}",
                recovery=evaluation,
                selected=item,
                considered=considered,
            )
    current_evaluation = next(
        evaluation
        for item, evaluation in considered
        if item.artifact_kind == "primary"
    )
    return AttemptSelection(
        action=current_evaluation.action,
        recovery=current_evaluation,
        selected=None,
        considered=considered,
    )


def renormalize_wave(
    *,
    judge: str,
    wave: str,
    source_path: Path,
    source_manifest_path: Path,
    retry_history_path: Path | None,
    output_path: Path,
    logical_output_path: Path,
    parser_sha256: str,
    base_dependency_sha256: str,
    postprocessor_sha256: str,
) -> dict:
    source_output_hash_before = frontier.base.file_sha256(source_path)
    source_manifest_hash_before = frontier.base.file_sha256(source_manifest_path)
    rows = load_jsonl(source_path)
    source_manifest = load_json(source_manifest_path)
    source_configuration, source_output_hash, source_manifest_hash = (
        validate_source_wave(
            judge=judge,
            wave=wave,
            source_path=source_path,
            manifest_path=source_manifest_path,
            rows=rows,
            manifest=source_manifest,
        )
    )
    source_configuration_hash = str(source_manifest["configuration_hash"])
    source_frozen_hash = str(source_manifest["frozen_configuration_hash"])
    if source_output_hash != source_output_hash_before:
        raise ValueError(f"source output changed while it was read: {source_path}")
    if source_manifest_hash != source_manifest_hash_before:
        raise ValueError(
            f"source manifest changed while it was read: {source_manifest_path}"
        )
    retry_groups: dict[str, list[AttemptSource]] = {}
    retry_metadata: dict | None = None
    if retry_history_path is not None:
        retry_groups, retry_metadata = validate_retry_history(
            retry_history_path,
            primary_rows=rows,
        )

    configuration = deepcopy(source_configuration)
    configuration.update(
        {
            "source_normalization_version": SOURCE_NORMALIZATION_VERSION,
            "normalization_version": NORMALIZATION_VERSION,
            "recovery_policy_version": RECOVERY_POLICY_VERSION,
            "retry_selection": "earliest_positive_numeric_attempt",
            "illegal_escape_repair_version": ILLEGAL_ESCAPE_REPAIR_VERSION,
            "normalization_parser_sha256": parser_sha256,
            "normalization_base_dependency_sha256": base_dependency_sha256,
            "normalization_postprocessor_sha256": postprocessor_sha256,
            "normalization_runtime": {
                "python_implementation": platform.python_implementation(),
                "python_version": platform.python_version(),
            },
        }
    )
    configuration_hash = frontier.base.stable_hash(configuration)
    frozen_configuration_hash = frontier.base.frozen_configuration_hash(configuration)

    actions: Counter[str] = Counter()
    methods: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    attempt_verdict_disagreements = 0
    output_rows: list[dict] = []
    for source_row in rows:
        source_status = str(source_row["status"])
        source_verdict = str(source_row["verdict"])
        case_id = str(source_row["case_id"])
        current_attempt = _numeric_attempt(
            source_row.get("attempt"), location=f"{source_path}/{case_id}"
        )
        primary_source = AttemptSource(
            row=source_row,
            attempt=current_attempt,
            artifact_kind="primary",
            artifact_path=source_path,
            artifact_sha256=source_output_hash,
            row_sha256=frontier.base.stable_hash(source_row),
            raw_output_sha256=text_sha256(source_row["raw_output"]),
        )
        output_row = dict(source_row)
        selection = select_attempt(
            primary_source,
            retry_groups.get(case_id, []),
        )
        action = selection.action
        recovery = selection.recovery
        selected_source = selection.selected
        considered = selection.considered
        if source_status != "ok" and recovery.parsed is not None:
            parsed = recovery.parsed
            output_row.update(
                {
                    "verdict": parsed.verdict,
                    "native_score": parsed.native_score,
                    "rationale": parsed.rationale,
                    "evidence": parsed.evidence,
                    "status": parsed.status,
                    "error": parsed.error,
                }
            )
        selected_audit = (
            {
                "attempt": selected_source.attempt,
                "artifact_kind": selected_source.artifact_kind,
                "artifact_file": str(selected_source.artifact_path),
                "artifact_sha256": selected_source.artifact_sha256,
                "row_sha256": selected_source.row_sha256,
                "raw_output_sha256": selected_source.raw_output_sha256,
                "recovery_method": recovery.method,
            }
            if selected_source is not None
            else None
        )
        recoverable_attempt_verdicts = [
            evaluation.parsed.verdict
            for _, evaluation in considered
            if evaluation.parsed is not None
        ]
        attempts_disagree = len(set(recoverable_attempt_verdicts)) > 1

        output_row.update(
            {
                "normalization_version": NORMALIZATION_VERSION,
                "configuration_hash": configuration_hash,
                "frozen_configuration_hash": frozen_configuration_hash,
                "normalization_recovery": {
                    "schema_version": RECOVERY_SCHEMA_VERSION,
                    "policy_version": RECOVERY_POLICY_VERSION,
                    "illegal_escape_repair_version": (
                        ILLEGAL_ESCAPE_REPAIR_VERSION
                    ),
                    "action": action,
                    "method": recovery.method,
                    "detail": recovery.detail,
                    "candidate_count": recovery.candidate_count,
                    "selected_start": recovery.selected_start,
                    "selected_end": recovery.selected_end,
                    "selected_sha256": recovery.selected_sha256,
                    "repaired_sha256": recovery.repaired_sha256,
                    "illegal_escape_offsets": list(
                        recovery.illegal_escape_offsets
                    ),
                    "selected_source": selected_audit,
                    "considered_attempts": [
                        _attempt_audit(item, evaluation)
                        for item, evaluation in considered
                    ],
                    "recoverable_attempt_verdicts": (
                        recoverable_attempt_verdicts
                    ),
                    "recoverable_attempts_disagree": attempts_disagree,
                    "parser_sha256": parser_sha256,
                    "base_dependency_sha256": base_dependency_sha256,
                    "postprocessor_sha256": postprocessor_sha256,
                    "source_row_sha256": frontier.base.stable_hash(source_row),
                    "source_output_sha256": source_output_hash,
                    "source_manifest_sha256": source_manifest_hash,
                    "source_normalization_version": SOURCE_NORMALIZATION_VERSION,
                    "source_configuration_hash": source_configuration_hash,
                    "source_frozen_configuration_hash": source_frozen_hash,
                    "source_status": source_status,
                    "source_verdict": source_verdict,
                    "source_error": source_row.get("error"),
                },
            }
        )
        if output_row["raw_output"] != source_row["raw_output"]:
            raise AssertionError("raw_output changed during normalization")
        actions[action] += 1
        if recovery.method is not None:
            methods[recovery.method] += 1
        attempt_verdict_disagreements += int(attempts_disagree)
        statuses[str(output_row["status"])] += 1
        output_rows.append(output_row)

    write_jsonl(output_path, output_rows)
    usable = sum(
        row["status"] == "ok" and row["verdict"] in {"pass", "fail"}
        for row in output_rows
    )
    no_decision = len(output_rows) - usable
    output_manifest = deepcopy(source_manifest)
    output_manifest.update(
        {
            "schema_version": RECOVERY_SCHEMA_VERSION,
            "status": "complete" if no_decision == 0 else "complete_with_errors",
            "renormalized_at": utc_now(),
            "case_count": len(output_rows),
            "normalization_version": NORMALIZATION_VERSION,
            "configuration": configuration,
            "configuration_hash": configuration_hash,
            "frozen_configuration_hash": frozen_configuration_hash,
            "output_file": str(logical_output_path),
            "output_sha256": frontier.base.file_sha256(output_path),
            "usable_decisions": usable,
            "no_decision_rows": no_decision,
            "model_provenance_consistent": True,
            "status_counts": dict(sorted(statuses.items())),
            "normalization_actions": dict(sorted(actions.items())),
            "normalization_methods": dict(sorted(methods.items())),
            "attempt_verdict_disagreement_rows": (
                attempt_verdict_disagreements
            ),
            "normalization_recovery": {
                "policy_version": RECOVERY_POLICY_VERSION,
                "illegal_escape_repair_version": (
                    ILLEGAL_ESCAPE_REPAIR_VERSION
                ),
                "parser_sha256": parser_sha256,
                "base_dependency_sha256": base_dependency_sha256,
                "postprocessor_sha256": postprocessor_sha256,
            },
            "source": {
                "output_file": str(source_path),
                "output_sha256": source_output_hash,
                "manifest_file": str(source_manifest_path),
                "manifest_sha256": source_manifest_hash,
                "normalization_version": SOURCE_NORMALIZATION_VERSION,
                "configuration_hash": source_configuration_hash,
                "frozen_configuration_hash": source_frozen_hash,
                "retry_history": retry_metadata,
            },
        }
    )
    write_json(output_path.with_suffix(".manifest.json"), output_manifest)
    return output_manifest


def _write_summary(root: Path, summary: dict) -> None:
    write_json(root / "renormalization_summary.json", summary)
    fieldnames = [
        "judge_name",
        "wave",
        "case_count",
        "unchanged_accepted",
        "recovered_from_primary",
        "recovered_from_retry_history",
        "unresolved_no_valid_object",
        "unresolved_ambiguous",
        "unresolved_nonterminal",
        "source_accepted",
        "exact_json",
        "embedded_terminal_json",
        "illegal_escape_repair",
        "attempt_verdict_disagreement_rows",
        "usable_decisions",
        "no_decision_rows",
    ]
    with (root / "renormalization_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for wave in summary["waves"]:
            actions = wave["normalization_actions"]
            writer.writerow(
                {
                    "judge_name": wave["judge_name"],
                    "wave": wave["wave"],
                    "case_count": wave["case_count"],
                    "unchanged_accepted": actions.get("unchanged_accepted", 0),
                    "recovered_from_primary": actions.get(
                        "recovered_from_primary", 0
                    ),
                    "recovered_from_retry_history": actions.get(
                        "recovered_from_retry_history", 0
                    ),
                    "unresolved_no_valid_object": actions.get(
                        "unresolved_no_valid_object", 0
                    ),
                    "unresolved_ambiguous": actions.get(
                        "unresolved_ambiguous", 0
                    ),
                    "unresolved_nonterminal": actions.get(
                        "unresolved_nonterminal", 0
                    ),
                    "source_accepted": wave["normalization_methods"].get(
                        "source_accepted", 0
                    ),
                    "exact_json": wave["normalization_methods"].get(
                        "exact_json", 0
                    ),
                    "embedded_terminal_json": wave[
                        "normalization_methods"
                    ].get("embedded_terminal_json", 0),
                    "illegal_escape_repair": wave[
                        "normalization_methods"
                    ].get("illegal_escape_repair", 0),
                    "attempt_verdict_disagreement_rows": wave[
                        "attempt_verdict_disagreement_rows"
                    ],
                    "usable_decisions": wave["usable_decisions"],
                    "no_decision_rows": wave["no_decision_rows"],
                }
            )


def renormalize_tree(source_root: Path, output_root: Path) -> dict:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if not source_root.is_dir():
        raise ValueError(f"source result root does not exist: {source_root}")
    if output_root.exists():
        raise FileExistsError(f"output root already exists: {output_root}")
    if source_root == output_root or source_root in output_root.parents:
        raise ValueError("output root must be a new tree outside the source result root")

    waves = discover_waves(source_root)
    parser_sha256 = frontier.base.file_sha256(frontier.__file__)
    base_dependency_sha256 = frontier.base.file_sha256(frontier.base.__file__)
    postprocessor_sha256 = frontier.base.file_sha256(__file__)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent)
    )
    source_hashes: list[tuple[Path, str]] = []
    absent_retry_paths: list[Path] = []
    try:
        manifests: list[dict] = []
        for (
            judge,
            wave,
            source_path,
            source_manifest_path,
            retry_history_path,
        ) in waves:
            output_path = temporary_root / judge / wave / f"{wave}.jsonl"
            logical_output_path = output_root / judge / wave / f"{wave}.jsonl"
            manifest = renormalize_wave(
                judge=judge,
                wave=wave,
                source_path=source_path,
                source_manifest_path=source_manifest_path,
                retry_history_path=retry_history_path,
                output_path=output_path,
                logical_output_path=logical_output_path,
                parser_sha256=parser_sha256,
                base_dependency_sha256=base_dependency_sha256,
                postprocessor_sha256=postprocessor_sha256,
            )
            manifests.append(manifest)
            source_hashes.extend(
                [
                    (source_path, manifest["source"]["output_sha256"]),
                    (source_manifest_path, manifest["source"]["manifest_sha256"]),
                ]
            )
            retry_metadata = manifest["source"].get("retry_history")
            if retry_metadata is not None:
                source_hashes.append(
                    (Path(retry_metadata["file"]), retry_metadata["sha256"])
                )
            else:
                absent_retry_paths.append(
                    source_path.with_name(f"{wave}.retry_history.jsonl")
                )

        for judge in frontier.FRONTIER_JUDGES:
            judge_manifests = [
                manifest for manifest in manifests if manifest["judge_name"] == judge
            ]
            frozen_hashes = {
                manifest["frozen_configuration_hash"] for manifest in judge_manifests
            }
            if len(frozen_hashes) != 1:
                raise ValueError(
                    f"derived frozen_configuration_hash differs across waves for {judge}"
                )
            resolved_models = {
                tuple(manifest["resolved_provider_models"])
                for manifest in judge_manifests
            }
            if len(resolved_models) != 1:
                raise ValueError(
                    f"resolved_provider_model differs across waves for {judge}"
                )

        action_totals: Counter[str] = Counter()
        method_totals: Counter[str] = Counter()
        status_totals: Counter[str] = Counter()
        attempt_verdict_disagreement_rows = 0
        wave_summaries: list[dict] = []
        for manifest in manifests:
            action_totals.update(manifest["normalization_actions"])
            method_totals.update(manifest["normalization_methods"])
            status_totals.update(manifest["status_counts"])
            attempt_verdict_disagreement_rows += manifest[
                "attempt_verdict_disagreement_rows"
            ]
            output_manifest_path = Path(manifest["output_file"]).with_suffix(
                ".manifest.json"
            )
            temporary_manifest_path = temporary_root / output_manifest_path.relative_to(
                output_root
            )
            wave_summaries.append(
                {
                    "judge_name": manifest["judge_name"],
                    "wave": manifest["wave"],
                    "case_count": manifest["case_count"],
                    "usable_decisions": manifest["usable_decisions"],
                    "no_decision_rows": manifest["no_decision_rows"],
                    "normalization_actions": manifest["normalization_actions"],
                    "normalization_methods": manifest["normalization_methods"],
                    "attempt_verdict_disagreement_rows": manifest[
                        "attempt_verdict_disagreement_rows"
                    ],
                    "status_counts": manifest["status_counts"],
                    "configuration_hash": manifest["configuration_hash"],
                    "frozen_configuration_hash": manifest[
                        "frozen_configuration_hash"
                    ],
                    "output_file": manifest["output_file"],
                    "output_sha256": manifest["output_sha256"],
                    "output_manifest_sha256": frontier.base.file_sha256(
                        temporary_manifest_path
                    ),
                    "source": manifest["source"],
                }
            )

        summary = {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "completed_at": utc_now(),
            "source_root": str(source_root),
            "output_root": str(output_root),
            "normalization_version": NORMALIZATION_VERSION,
            "recovery_policy_version": RECOVERY_POLICY_VERSION,
            "illegal_escape_repair_version": ILLEGAL_ESCAPE_REPAIR_VERSION,
            "parser_sha256": parser_sha256,
            "base_dependency_sha256": base_dependency_sha256,
            "postprocessor_sha256": postprocessor_sha256,
            "runtime": {
                "python_implementation": platform.python_implementation(),
                "python_version": platform.python_version(),
            },
            "wave_count": len(manifests),
            "case_rows": sum(manifest["case_count"] for manifest in manifests),
            "normalization_actions": dict(sorted(action_totals.items())),
            "normalization_methods": dict(sorted(method_totals.items())),
            "attempt_verdict_disagreement_rows": (
                attempt_verdict_disagreement_rows
            ),
            "status_counts": dict(sorted(status_totals.items())),
            "waves": wave_summaries,
        }
        _write_summary(temporary_root, summary)

        for source_path, expected_hash in source_hashes:
            if frontier.base.file_sha256(source_path) != expected_hash:
                raise ValueError(
                    f"source artifact changed during normalization: {source_path}"
                )
        for retry_path in absent_retry_paths:
            if retry_path.exists():
                raise ValueError(
                    f"retry history appeared during normalization: {retry_path}"
                )
        temporary_root.replace(output_root)
        return summary
    except BaseException:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("output_root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = renormalize_tree(args.source_root, args.output_root)
    except (FileExistsError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    actions = summary["normalization_actions"]
    print(
        f"Wrote {summary['case_rows']} v3 rows to {args.output_root}; "
        f"primary_recovered={actions.get('recovered_from_primary', 0)}, "
        f"retry_recovered={actions.get('recovered_from_retry_history', 0)}, "
        f"ambiguous={actions.get('unresolved_ambiguous', 0)}, "
        f"nonterminal={actions.get('unresolved_nonterminal', 0)}, "
        f"unresolved={actions.get('unresolved_no_valid_object', 0)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
