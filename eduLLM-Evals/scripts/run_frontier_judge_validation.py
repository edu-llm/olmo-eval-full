#!/usr/bin/env python3
"""Run cross-family frontier-model judges over the human-validation cases.

The script keeps identity routing separate from judge inputs. ``prepare`` reads
the existing human-label CSV once and emits (a) a still-blinded copy of the
case JSONL and (b) a minimal case-to-provider-family routing JSONL containing
no labels or model names. ``plan`` validates those artifacts without loading
credentials or making network calls. ``run`` executes one judge/wave, while
``suite`` executes the frozen six-wave reliability design.

API credentials are loaded from the repository-root ``.env`` (without
overriding already-exported values) and are never written to an artifact. The
default ``truefoundry`` backend uses ``TFY_API_KEY`` and one OpenAI-compatible
Chat Completions endpoint. The optional ``direct`` backend uses
``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``, and ``GOOGLE_API_KEY`` (or
``GEMINI_API_KEY`` as a Google fallback). Model IDs may likewise be set with
``FRONTIER_OPENAI_MODEL``, ``FRONTIER_ANTHROPIC_MODEL``, and
``FRONTIER_GOOGLE_MODEL``; an explicit CLI model flag takes precedence over
the environment and the selected backend's built-in default.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import platform
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

try:  # Works both as ``python scripts/...`` and as a test import.
    from scripts import run_judge_validation as base
except ImportError:  # pragma: no cover - exercised by direct script execution.
    import run_judge_validation as base


ROOT = Path(__file__).resolve().parents[1]
NORMALIZATION_VERSION = "frontier-judge-normalization-v1"
ROUTING_VERSION = "frontier-cross-family-routing-v1"
ADAPTER = "generic-binary"
FAMILIES = ("openai", "anthropic", "google")
BACKENDS = ("truefoundry", "direct")
DEFAULT_BACKEND = "truefoundry"
TRUEFOUNDRY_API_KEY_ENV = "TFY_API_KEY"
TRUEFOUNDRY_BASE_URL_ENV = "TFY_BASE_URL"
TRUEFOUNDRY_TOKEN_PARAM_ENV = "TFY_TOKEN_PARAM"
TRUEFOUNDRY_TEMPERATURE_MODE_ENV = "TFY_TEMPERATURE_MODE"
TRUEFOUNDRY_BASE_URL = (
    "https://tfy.promptlens.trilogy.com/api/llm/api/inference/openai"
)
TRUEFOUNDRY_TOKEN_PARAMS = ("max_completion_tokens", "max_tokens")
TRUEFOUNDRY_TEMPERATURE_MODES = ("zero", "omit")

WAVES: dict[str, tuple[str, str]] = {
    "canonical_r1": ("canonical", "r1"),
    "canonical_r2": ("canonical", "r2"),
    "canonical_r3": ("canonical", "r3"),
    "whitespace_r1": ("whitespace", "r1"),
    "header_synonyms_r1": ("header_synonyms", "r1"),
    "instruction_politeness_r1": ("instruction_politeness", "r1"),
}

JUDGMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["pass", "fail"]},
        "rationale": {"type": "string"},
        "evidence": {"type": "string"},
    },
    "required": ["verdict", "rationale", "evidence"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class FrontierJudgeSpec:
    name: str
    family: str
    provider: str
    default_model_id: str
    revision: str | None
    model_env: str
    api_key_envs: tuple[str, ...]
    api_surface: str


FRONTIER_JUDGES: dict[str, FrontierJudgeSpec] = {
    "gpt-5.5": FrontierJudgeSpec(
        name="gpt-5.5",
        family="openai",
        provider="openai",
        default_model_id="gpt-5.5-2026-04-23",
        revision="gpt-5.5-2026-04-23",
        model_env="FRONTIER_OPENAI_MODEL",
        api_key_envs=("OPENAI_API_KEY",),
        api_surface="responses",
    ),
    "opus-4.8": FrontierJudgeSpec(
        name="opus-4.8",
        family="anthropic",
        provider="anthropic",
        default_model_id="claude-opus-4-8",
        revision="claude-opus-4-8",
        model_env="FRONTIER_ANTHROPIC_MODEL",
        api_key_envs=("ANTHROPIC_API_KEY",),
        api_surface="messages",
    ),
    "gemini-3.6-flash": FrontierJudgeSpec(
        name="gemini-3.6-flash",
        family="google",
        provider="google",
        default_model_id="gemini-3.6-flash",
        revision=None,
        model_env="FRONTIER_GOOGLE_MODEL",
        api_key_envs=("GOOGLE_API_KEY", "GEMINI_API_KEY"),
        api_surface="models.generate_content",
    ),
}

TRUEFOUNDRY_DEFAULT_MODELS = {
    "gpt-5.5": "openai-group/gpt-5.5",
    "opus-4.8": "claude-group/claude-opus-4-8",
    "gemini-3.6-flash": "gemini-group/gemini-3.6-flash",
}

DIRECT_MODEL_ENVS = {
    "gpt-5.5": "FRONTIER_DIRECT_OPENAI_MODEL",
    "opus-4.8": "FRONTIER_DIRECT_ANTHROPIC_MODEL",
    "gemini-3.6-flash": "FRONTIER_DIRECT_GOOGLE_MODEL",
}


# Exact mappings are intentional. An unknown or conflicting identity fails
# preparation rather than risking same-family grading through fuzzy matching.
MODEL_TO_FAMILY = {
    "gpt-5.5": "openai",
    "opus-4.8": "anthropic",
    "claude-opus-4-8": "anthropic",
    "gemini-3.5-flash": "google",
}
MODEL_SLUG_TO_FAMILY = {
    "openai-group/gpt-5.5": "openai",
    "claude-group/claude-opus-4-8": "anthropic",
    "gemini-group/gemini-3.5-flash": "google",
}
ANONYMOUS_TUTOR_TO_FAMILY = {
    "Tutor A": "openai",
    "Tutor B": "anthropic",
    "Tutor C": "google",
}
FORBIDDEN_ROUTE_FIELDS = {
    "human_label",
    "human_notes",
    "candidate_model",
    "candidate_model_slug",
    "anonymous_tutor",
}


@dataclass(frozen=True)
class Route:
    case_id: str
    input_hash: str
    candidate_family: str


@dataclass(frozen=True)
class FrontierGenerationResult:
    text: str = ""
    raw_response: object | None = None
    resolved_model: str | None = None
    usage: object | None = None
    error: str | None = None
    latency_ms: float | None = None
    provider_attempts: int = 0


@dataclass(frozen=True)
class TrueFoundrySettings:
    base_url: str
    token_param: str
    temperature_mode: str

    @property
    def temperature(self) -> float | None:
        return 0.0 if self.temperature_mode == "zero" else None


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _write_json(path: Path, value: object) -> None:
    _atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    base.write_jsonl(path, rows)


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    for method_name, kwargs in (
        ("model_dump", {"mode": "json"}),
        ("to_dict", {}),
        ("to_json_dict", {}),
    ):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                return _json_safe(method(**kwargs))
            except (TypeError, ValueError):
                continue
    return str(value)


def _redact(text: str, secrets: Sequence[str]) -> str:
    redacted = text
    for secret in secrets:
        if secret:
            redacted = redacted.replace(secret, "[REDACTED]")
    return redacted


def _redact_json(value: object, secrets: Sequence[str]) -> object:
    """Recursively remove credentials and endpoint URLs from artifact data."""

    safe = _json_safe(value)
    if isinstance(safe, str):
        return _redact(safe, secrets)
    if isinstance(safe, dict):
        return {key: _redact_json(item, secrets) for key, item in safe.items()}
    if isinstance(safe, list):
        return [_redact_json(item, secrets) for item in safe]
    return safe


def _family_from_human_row(row: Mapping[str, str], *, location: str) -> str:
    observed: list[tuple[str, str]] = []
    lookups = (
        ("candidate_model", MODEL_TO_FAMILY),
        ("candidate_model_slug", MODEL_SLUG_TO_FAMILY),
        ("anonymous_tutor", ANONYMOUS_TUTOR_TO_FAMILY),
    )
    for field, mapping in lookups:
        value = str(row.get(field) or "").strip()
        if not value:
            continue
        family = mapping.get(value)
        if family is None:
            raise ValueError(f"{location}: unsupported {field} {value!r}")
        observed.append((field, family))
    if not observed:
        raise ValueError(f"{location}: no recognized tutor identity for family routing")
    families = {family for _, family in observed}
    if len(families) != 1:
        raise ValueError(f"{location}: conflicting tutor-family identity fields: {observed}")
    return next(iter(families))


def prepare_artifacts(
    cases_path: Path,
    human_labels_path: Path,
    out_dir: Path,
    *,
    overwrite: bool = False,
) -> dict:
    cases = base.load_jsonl(cases_path)
    if not cases:
        raise ValueError(f"{cases_path}: no cases")
    base.validate_judge_cases(cases)
    cases_by_id = {str(case["case_id"]): case for case in cases}

    with human_labels_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        required = {"case_id", "case_input_hash"}
        missing = sorted(required - fields)
        if missing:
            raise ValueError(
                f"{human_labels_path}: missing required column(s): {', '.join(missing)}"
            )
        human_rows = list(reader)

    routes: dict[str, Route] = {}
    family_counts = {family: 0 for family in FAMILIES}
    for row_number, row in enumerate(human_rows, 2):
        case_id = str(row.get("case_id") or "").strip()
        location = f"{human_labels_path}:{row_number}"
        if not case_id or case_id in routes:
            raise ValueError(f"{location}: blank or duplicate case_id {case_id!r}")
        case = cases_by_id.get(case_id)
        if case is None:
            raise ValueError(f"{location}: unknown case_id {case_id!r}")
        input_hash = str(row.get("case_input_hash") or "").strip()
        expected_hash = base.stable_hash(case)
        if input_hash != expected_hash:
            raise ValueError(f"{location}: case_input_hash mismatch for {case_id}")
        for field in ("response_id", "scenario_id", "criterion_id"):
            value = str(row.get(field) or "").strip()
            if value and value != str(case.get(field) or "").strip():
                raise ValueError(f"{location}: {field} mismatch for {case_id}")
        family = _family_from_human_row(row, location=location)
        routes[case_id] = Route(case_id, input_hash, family)
        family_counts[family] += 1

    missing_routes = sorted(set(cases_by_id) - set(routes))
    extra_routes = sorted(set(routes) - set(cases_by_id))
    if missing_routes or extra_routes:
        raise ValueError(
            "routing does not cover the blinded cases exactly; "
            f"missing={missing_routes[:5]}, extra={extra_routes[:5]}"
        )

    cases_out = out_dir / "judge_cases.blinded.jsonl"
    routes_out = out_dir / "case_routes.blinded.jsonl"
    manifest_out = out_dir / "prepare_manifest.json"
    existing = [path for path in (cases_out, routes_out, manifest_out) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"prepared artifact(s) already exist: {', '.join(map(str, existing))}; "
            "pass --overwrite"
        )

    route_rows = [
        {
            "case_id": route.case_id,
            "input_hash": route.input_hash,
            "candidate_family": route.candidate_family,
            "routing_version": ROUTING_VERSION,
        }
        for route in (routes[str(case["case_id"])] for case in cases)
    ]
    if any(FORBIDDEN_ROUTE_FIELDS & set(row) for row in route_rows):
        raise AssertionError("prepared routes unexpectedly contain identity or gold fields")

    _write_jsonl(cases_out, cases)
    _write_jsonl(routes_out, route_rows)
    summary = {
        "status": "complete",
        "prepared_at": base.utc_now(),
        "routing_version": ROUTING_VERSION,
        "case_count": len(cases),
        "candidate_family_counts": family_counts,
        "source_cases": str(cases_path),
        "source_cases_sha256": base.file_sha256(cases_path),
        "source_human_labels": str(human_labels_path),
        "source_human_labels_sha256": base.file_sha256(human_labels_path),
        "judge_cases_file": str(cases_out),
        "judge_cases_sha256": base.file_sha256(cases_out),
        "case_routes_file": str(routes_out),
        "case_routes_sha256": base.file_sha256(routes_out),
    }
    _write_json(manifest_out, summary)
    return summary


def load_routes(path: Path) -> dict[str, Route]:
    rows = base.load_jsonl(path)
    if not rows:
        raise ValueError(f"{path}: no routes")
    routes: dict[str, Route] = {}
    for index, row in enumerate(rows, 1):
        forbidden = sorted(FORBIDDEN_ROUTE_FIELDS & set(row))
        if forbidden:
            raise ValueError(
                f"{path}:{index}: route contains forbidden field(s): {', '.join(forbidden)}"
            )
        case_id = str(row.get("case_id") or "").strip()
        input_hash = str(row.get("input_hash") or "").strip()
        family = str(row.get("candidate_family") or "").strip().lower()
        if not case_id or case_id in routes:
            raise ValueError(f"{path}:{index}: blank or duplicate case_id {case_id!r}")
        if not re.fullmatch(r"[0-9a-f]{64}", input_hash):
            raise ValueError(f"{path}:{index}: invalid input_hash for {case_id}")
        if family not in FAMILIES:
            raise ValueError(f"{path}:{index}: unsupported candidate_family {family!r}")
        if row.get("routing_version") not in {None, ROUTING_VERSION}:
            raise ValueError(f"{path}:{index}: unsupported routing_version")
        routes[case_id] = Route(case_id, input_hash, family)
    return routes


def load_and_validate_inputs(
    cases_path: Path, routing_path: Path
) -> tuple[list[dict], dict[str, Route]]:
    cases = base.load_jsonl(cases_path)
    if not cases:
        raise ValueError(f"{cases_path}: no cases")
    base.validate_judge_cases(cases)
    routes = load_routes(routing_path)
    case_ids = [str(case["case_id"]) for case in cases]
    if set(case_ids) != set(routes):
        missing = sorted(set(case_ids) - set(routes))
        extra = sorted(set(routes) - set(case_ids))
        raise ValueError(
            "routes do not cover cases exactly; "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    for case in cases:
        route = routes[str(case["case_id"])]
        if base.stable_hash(case) != route.input_hash:
            raise ValueError(f"input_hash mismatch for {route.case_id}")
    return cases, routes


def eligible_cases(
    cases: Sequence[dict],
    routes: Mapping[str, Route],
    judge: FrontierJudgeSpec,
    *,
    limit: int | None = None,
) -> tuple[list[dict], int]:
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    selected = [
        case
        for case in cases
        if routes[str(case["case_id"])].candidate_family != judge.family
    ]
    excluded = len(cases) - len(selected)
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise ValueError(f"{judge.name}: no cross-family cases are eligible")
    same_family = [
        str(case["case_id"])
        for case in selected
        if routes[str(case["case_id"])].candidate_family == judge.family
    ]
    if same_family:
        raise AssertionError(
            f"{judge.name}: own-family exclusion failed for {same_family[:5]}"
        )
    return selected, excluded


def _validate_backend(backend: str) -> str:
    if backend not in BACKENDS:
        raise ValueError(
            f"unsupported backend {backend!r}; expected one of {', '.join(BACKENDS)}"
        )
    return backend


def normalize_truefoundry_base_url(value: str) -> str:
    """Canonicalize a safe OpenAI-compatible TrueFoundry base URL."""

    raw = value.strip()
    if not raw:
        raise ValueError(f"{TRUEFOUNDRY_BASE_URL_ENV} cannot be blank")
    parsed = urlsplit(raw)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise ValueError(
            f"{TRUEFOUNDRY_BASE_URL_ENV} must be an absolute http(s) URL"
        )
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{TRUEFOUNDRY_BASE_URL_ENV} must not contain userinfo")
    if parsed.query or parsed.fragment:
        raise ValueError(
            f"{TRUEFOUNDRY_BASE_URL_ENV} must not contain a query or fragment"
        )
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"{TRUEFOUNDRY_BASE_URL_ENV} has an invalid port") from exc

    hostname = parsed.hostname.lower()
    loopback_hosts = {"localhost", "127.0.0.1", "::1"}
    if parsed.scheme.lower() == "http" and hostname not in loopback_hosts:
        raise ValueError(
            f"{TRUEFOUNDRY_BASE_URL_ENV} must use HTTPS except for a loopback "
            "development endpoint"
        )

    inference_path = urlsplit(TRUEFOUNDRY_BASE_URL).path.rstrip("/")
    path = parsed.path.rstrip("/")
    chat_completions_suffix = "/chat/completions"
    if path.endswith(chat_completions_suffix):
        path = path[: -len(chat_completions_suffix)].rstrip("/")
    if path in {"", "/"}:
        path = "/v1"
    elif path not in {"/v1", inference_path}:
        raise ValueError(
            f"{TRUEFOUNDRY_BASE_URL_ENV} path must be /v1 or "
            f"{inference_path}"
        )

    if ":" in hostname:
        hostname = f"[{hostname}]"
    netloc = f"{hostname}:{port}" if port is not None else hostname
    return urlunsplit((parsed.scheme.lower(), netloc, path, "", ""))


def resolve_truefoundry_settings(
    *,
    token_param: str | None = None,
    temperature_mode: str | None = None,
) -> TrueFoundrySettings:
    base_url = normalize_truefoundry_base_url(
        os.environ.get(TRUEFOUNDRY_BASE_URL_ENV, TRUEFOUNDRY_BASE_URL)
    )
    resolved_token_param = (
        str(token_param or "").strip()
        or os.environ.get(TRUEFOUNDRY_TOKEN_PARAM_ENV, "").strip()
        or "max_completion_tokens"
    )
    if resolved_token_param not in TRUEFOUNDRY_TOKEN_PARAMS:
        raise ValueError(
            f"{TRUEFOUNDRY_TOKEN_PARAM_ENV} must be one of "
            f"{', '.join(TRUEFOUNDRY_TOKEN_PARAMS)}"
        )
    resolved_temperature_mode = (
        str(temperature_mode or "").strip().lower()
        or os.environ.get(TRUEFOUNDRY_TEMPERATURE_MODE_ENV, "").strip().lower()
        or "zero"
    )
    if resolved_temperature_mode not in TRUEFOUNDRY_TEMPERATURE_MODES:
        raise ValueError(
            f"{TRUEFOUNDRY_TEMPERATURE_MODE_ENV} must be one of "
            f"{', '.join(TRUEFOUNDRY_TEMPERATURE_MODES)}"
        )
    return TrueFoundrySettings(
        base_url=base_url,
        token_param=resolved_token_param,
        temperature_mode=resolved_temperature_mode,
    )


def _backend_default_model(spec: FrontierJudgeSpec, backend: str) -> str:
    backend = _validate_backend(backend)
    if backend == "truefoundry":
        return TRUEFOUNDRY_DEFAULT_MODELS[spec.name]
    return spec.default_model_id


def _backend_model_env(spec: FrontierJudgeSpec, backend: str) -> str:
    backend = _validate_backend(backend)
    return spec.model_env if backend == "truefoundry" else DIRECT_MODEL_ENVS[spec.name]


def _backend_api_surface(spec: FrontierJudgeSpec, backend: str) -> str:
    backend = _validate_backend(backend)
    return "chat.completions" if backend == "truefoundry" else spec.api_surface


def _backend_api_key_envs(
    spec: FrontierJudgeSpec, backend: str
) -> tuple[str, ...]:
    backend = _validate_backend(backend)
    return (
        (TRUEFOUNDRY_API_KEY_ENV,)
        if backend == "truefoundry"
        else spec.api_key_envs
    )


def _artifact_redaction_values(spec: FrontierJudgeSpec, backend: str) -> list[str]:
    _backend_api_key_envs(spec, backend)  # Validate both arguments.
    return _all_api_key_values()


def _all_api_key_values() -> list[str]:
    env_names = {TRUEFOUNDRY_API_KEY_ENV}
    for spec in FRONTIER_JUDGES.values():
        env_names.update(spec.api_key_envs)
    return [os.environ.get(env_name, "") for env_name in sorted(env_names)]


def _validate_model_for_backend(
    spec: FrontierJudgeSpec, model_id: str, backend: str
) -> str:
    if backend != "truefoundry":
        return model_id
    expected_prefix = {
        "openai": "openai-group/",
        "anthropic": "claude-group/",
        "google": "gemini-group/",
    }[spec.family]
    if not model_id.startswith(expected_prefix) or model_id == expected_prefix:
        raise ValueError(
            f"{spec.name}: TrueFoundry model {model_id!r} must start with "
            f"{expected_prefix!r}"
        )
    return model_id


def build_plan(
    cases_path: Path,
    routing_path: Path,
    *,
    backend: str = DEFAULT_BACKEND,
    tfy_token_param: str | None = None,
    tfy_temperature_mode: str | None = None,
) -> dict:
    backend = _validate_backend(backend)
    if backend == "direct" and (tfy_token_param or tfy_temperature_mode):
        raise ValueError(
            "TrueFoundry request-profile overrides require --backend truefoundry"
        )
    truefoundry = (
        resolve_truefoundry_settings(
            token_param=tfy_token_param,
            temperature_mode=tfy_temperature_mode,
        )
        if backend == "truefoundry"
        else None
    )
    cases, routes = load_and_validate_inputs(cases_path, routing_path)
    family_counts = {
        family: sum(route.candidate_family == family for route in routes.values())
        for family in FAMILIES
    }
    judges = {}
    for name, spec in FRONTIER_JUDGES.items():
        selected, excluded = eligible_cases(cases, routes, spec)
        graded_family_counts = {
            family: sum(
                routes[str(case["case_id"])].candidate_family == family
                for case in selected
            )
            for family in FAMILIES
        }
        judges[name] = {
            "judge_family": spec.family,
            "backend": backend,
            "api_surface": _backend_api_surface(spec, backend),
            "model_id": effective_model_id(spec, backend=backend)[0],
            "base_url": truefoundry.base_url if truefoundry else None,
            "eligible_cases_per_wave": len(selected),
            "own_family_excluded_per_wave": excluded,
            "graded_candidate_family_counts": graded_family_counts,
            "wave_count": len(WAVES),
            "planned_api_calls": len(selected) * len(WAVES),
        }
    return {
        "backend": backend,
        "base_url": truefoundry.base_url if truefoundry else None,
        "request_profile": (
            {
                "temperature": truefoundry.temperature,
                "token_param": truefoundry.token_param,
            }
            if truefoundry
            else None
        ),
        "routing_version": ROUTING_VERSION,
        "case_count": len(cases),
        "candidate_family_counts": family_counts,
        "waves": [
            {
                "wave": wave,
                "prompt_variant": variant,
                "replicate_id": replicate,
            }
            for wave, (variant, replicate) in WAVES.items()
        ],
        "judges": judges,
        "total_planned_api_calls": sum(
            judge["planned_api_calls"] for judge in judges.values()
        ),
    }


def parse_frontier_judgment(text: str) -> base.ParsedJudgment:
    raw = text or ""
    try:
        body = raw.strip()
        if not body:
            raise ValueError("expected a JSON object")
        if body.startswith("```"):
            fenced = re.fullmatch(
                r"```(?:json)?[ \t]*\r?\n(?P<body>\{.*\})\r?\n```",
                body,
                flags=re.DOTALL,
            )
            if fenced is None:
                raise ValueError("expected one exact JSON object fence")
            body = fenced.group("body")
        value = json.loads(body)
        if not isinstance(value, dict):
            raise ValueError("expected one top-level JSON object")
        expected_keys = {"verdict", "rationale", "evidence"}
        if set(value) != expected_keys:
            missing = sorted(expected_keys - set(value))
            extra = sorted(set(value) - expected_keys)
            raise ValueError(f"judgment fields differ; missing={missing}, extra={extra}")
        if any(not isinstance(value[key], str) for key in expected_keys):
            raise ValueError("all judgment fields must be strings")
        verdict = value["verdict"]
        rationale = value["rationale"].strip()
        evidence = value["evidence"].strip()
        if verdict not in {"pass", "fail"}:
            raise ValueError("verdict must be exactly 'pass' or 'fail'")
        if not rationale or not evidence:
            raise ValueError("rationale and evidence must be nonblank")
        if verdict == "pass" and evidence.upper() == "NONE":
            raise ValueError("a pass verdict cannot use NONE evidence")
        return base.ParsedJudgment(
            verdict=verdict,
            native_score=1 if verdict == "pass" else 0,
            rationale=rationale,
            evidence=evidence,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        return base.ParsedJudgment(
            verdict="no_decision",
            status="parse_error",
            error=str(exc),
        )


def _is_retryable_exception(exc: BaseException) -> bool:
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    try:
        status_int = int(status) if status is not None else None
    except (TypeError, ValueError):
        status_int = None
    if status_int in {408, 409, 425, 429} or (
        status_int is not None and status_int >= 500
    ):
        return True
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    transient_tokens = (
        "timeout",
        "timed out",
        "connection",
        "rate limit",
        "temporarily unavailable",
        "resource exhausted",
    )
    return any(token in name or token in message for token in transient_tokens)


class BaseFrontierGenerator:
    def __init__(
        self,
        *,
        spec: FrontierJudgeSpec,
        model_id: str,
        api_key: str,
        concurrency: int,
        timeout: float,
        max_output_tokens: int,
        reasoning_level: str,
        max_retries: int,
        retry_base_seconds: float,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.spec = spec
        self.model_id = model_id
        self.api_key = api_key
        self.concurrency = concurrency
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens
        self.reasoning_level = reasoning_level
        self.max_retries = max_retries
        self.retry_base_seconds = retry_base_seconds
        self.sleep = sleep
        self.redaction_values: tuple[str, ...] = (api_key,)

    def _call_once(
        self, messages: list[dict[str, str]]
    ) -> tuple[str, object, str | None, object | None]:
        raise NotImplementedError

    def _one(self, messages: list[dict[str, str]]) -> FrontierGenerationResult:
        started = time.monotonic()
        attempts = 0
        while True:
            attempts += 1
            try:
                text, raw, resolved_model, usage = self._call_once(messages)
                return FrontierGenerationResult(
                    text=text,
                    raw_response=_redact_json(raw, self.redaction_values),
                    resolved_model=resolved_model,
                    usage=_redact_json(usage, self.redaction_values),
                    latency_ms=(time.monotonic() - started) * 1000,
                    provider_attempts=attempts,
                )
            except Exception as exc:  # Provider SDK exception hierarchies differ.
                if attempts <= self.max_retries and _is_retryable_exception(exc):
                    delay = min(
                        30.0,
                        self.retry_base_seconds * (2 ** (attempts - 1)),
                    )
                    if delay:
                        self.sleep(delay + random.uniform(0, delay * 0.1))
                    continue
                return FrontierGenerationResult(
                    error=_redact(
                        f"{type(exc).__name__}: {exc}", self.redaction_values
                    ),
                    latency_ms=(time.monotonic() - started) * 1000,
                    provider_attempts=attempts,
                )

    def generate(
        self, message_batches: list[list[dict[str, str]]]
    ) -> list[FrontierGenerationResult]:
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            return list(executor.map(self._one, message_batches))

    def close(self) -> None:
        client = getattr(self, "client", None)
        close = getattr(client, "close", None)
        if callable(close):
            close()


class TrueFoundryGenerator(BaseFrontierGenerator):
    """OpenAI-compatible adapter shared by all three TrueFoundry models."""

    def __init__(
        self,
        *,
        settings: TrueFoundrySettings,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - environment dependent.
            raise RuntimeError(
                "install the 'openai' package to use the TrueFoundry backend"
            ) from exc
        self.settings = settings
        self.redaction_values = (self.api_key,)
        self.client = OpenAI(
            base_url=settings.base_url,
            api_key=self.api_key,
            max_retries=0,
            timeout=self.timeout,
        )

    def _call_once(
        self, messages: list[dict[str, str]]
    ) -> tuple[str, object, str | None, object | None]:
        request: dict[str, object] = {
            "model": self.model_id,
            "messages": messages,
            self.settings.token_param: self.max_output_tokens,
        }
        if self.settings.temperature is not None:
            request["temperature"] = self.settings.temperature
        response = self.client.chat.completions.create(**request)
        content = response.choices[0].message.content
        text = content if isinstance(content, str) else str(content or "")
        return (
            text,
            response,
            getattr(response, "model", None),
            getattr(response, "usage", None),
        )


class OpenAIFrontierGenerator(BaseFrontierGenerator):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - environment dependent.
            raise RuntimeError("install the 'openai' package to run gpt-5.5") from exc
        self.client = OpenAI(
            api_key=self.api_key,
            max_retries=0,
            timeout=self.timeout,
        )

    def _call_once(
        self, messages: list[dict[str, str]]
    ) -> tuple[str, object, str | None, object | None]:
        response = self.client.responses.create(
            model=self.model_id,
            input=messages,
            max_output_tokens=self.max_output_tokens,
            reasoning={"effort": self.reasoning_level},
            text={
                "format": {
                    "type": "json_schema",
                    "name": "frontier_judgment",
                    "strict": True,
                    "schema": JUDGMENT_SCHEMA,
                }
            },
            store=False,
        )
        text = response.output_text or ""
        return text, response, getattr(response, "model", None), getattr(response, "usage", None)


class AnthropicFrontierGenerator(BaseFrontierGenerator):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - environment dependent.
            raise RuntimeError("install the 'anthropic' package to run opus-4.8") from exc
        self.client = anthropic.Anthropic(
            api_key=self.api_key,
            max_retries=0,
            timeout=self.timeout,
        )

    def _call_once(
        self, messages: list[dict[str, str]]
    ) -> tuple[str, object, str | None, object | None]:
        system_parts = [message["content"] for message in messages if message["role"] == "system"]
        api_messages = [
            {"role": message["role"], "content": message["content"]}
            for message in messages
            if message["role"] in {"user", "assistant"}
        ]
        kwargs: dict[str, object] = {
            "model": self.model_id,
            "max_tokens": self.max_output_tokens,
            "messages": api_messages,
            "thinking": {"type": "adaptive", "display": "omitted"},
            "output_config": {
                "effort": self.reasoning_level,
                "format": {"type": "json_schema", "schema": JUDGMENT_SCHEMA},
            },
        }
        if system_parts:
            kwargs["system"] = "\n\n".join(system_parts)
        response = self.client.messages.create(**kwargs)
        text = "".join(
            str(block.text)
            for block in response.content
            if getattr(block, "type", None) == "text"
        )
        return text, response, getattr(response, "model", None), getattr(response, "usage", None)


class GoogleFrontierGenerator(BaseFrontierGenerator):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:  # pragma: no cover - environment dependent.
            raise RuntimeError("install the 'google-genai' package to run Gemini") from exc
        self.client = genai.Client(api_key=self.api_key)
        self.types = types

    def _call_once(
        self, messages: list[dict[str, str]]
    ) -> tuple[str, object, str | None, object | None]:
        if len(messages) != 1 or messages[0].get("role") != "user":
            raise ValueError("Gemini frontier adapter expects the one-user generic prompt")
        config = self.types.GenerateContentConfig(
            max_output_tokens=self.max_output_tokens,
            # GenerateContent exposes automatic thinking as a token-budget
            # sentinel. Gemini 3.6 Flash documents that automatic mode defaults
            # to medium, matching the explicit OpenAI/Anthropic effort below.
            thinking_config=self.types.ThinkingConfig(thinking_budget=-1),
            response_mime_type="application/json",
            response_json_schema=JUDGMENT_SCHEMA,
        )
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=messages[0]["content"],
            config=config,
        )
        text = str(getattr(response, "text", "") or "")
        resolved_model = getattr(response, "model", None) or getattr(
            response, "model_version", None
        )
        usage = getattr(response, "usage", None) or getattr(
            response, "usage_metadata", None
        )
        return text, response, resolved_model, usage


GENERATOR_TYPES = {
    "openai": OpenAIFrontierGenerator,
    "anthropic": AnthropicFrontierGenerator,
    "google": GoogleFrontierGenerator,
}


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError as exc:  # pragma: no cover - declared project dependency.
        raise RuntimeError("install python-dotenv to load the repository .env") from exc
    load_dotenv(ROOT / ".env", override=False)


def _load_model_env_only() -> None:
    """Load only non-secret frontier settings for credential-free planning."""

    try:
        from dotenv import dotenv_values
    except ImportError as exc:  # pragma: no cover - declared project dependency.
        raise RuntimeError("install python-dotenv to read repository model settings") from exc
    values = dotenv_values(ROOT / ".env")
    env_names = [spec.model_env for spec in FRONTIER_JUDGES.values()]
    env_names.extend(DIRECT_MODEL_ENVS.values())
    env_names.extend(
        (
            TRUEFOUNDRY_BASE_URL_ENV,
            TRUEFOUNDRY_TOKEN_PARAM_ENV,
            TRUEFOUNDRY_TEMPERATURE_MODE_ENV,
        )
    )
    for env_name in env_names:
        if os.environ.get(env_name):
            continue
        value = str(values.get(env_name) or "").strip()
        if value:
            os.environ[env_name] = value


def _credential(
    spec: FrontierJudgeSpec, backend: str = DEFAULT_BACKEND
) -> tuple[str, str]:
    backend = _validate_backend(backend)
    api_key_envs = _backend_api_key_envs(spec, backend)
    for env_name in api_key_envs:
        value = os.environ.get(env_name, "").strip()
        if value:
            return value, env_name
    raise ValueError(
        f"{spec.name}: set {' or '.join(api_key_envs)} in {ROOT / '.env'} "
        "or export it in the shell"
    )


def effective_model_id(
    spec: FrontierJudgeSpec,
    cli_model_id: str | None = None,
    *,
    backend: str = DEFAULT_BACKEND,
) -> tuple[str, str]:
    backend = _validate_backend(backend)
    if cli_model_id and cli_model_id.strip():
        model_id = cli_model_id.strip()
        return _validate_model_for_backend(spec, model_id, backend), "cli"
    model_env = _backend_model_env(spec, backend)
    environment_value = os.environ.get(model_env, "").strip()
    if environment_value:
        return (
            _validate_model_for_backend(spec, environment_value, backend),
            model_env,
        )
    model_id = _backend_default_model(spec, backend)
    return _validate_model_for_backend(spec, model_id, backend), "built_in_default"


def build_generator(
    spec: FrontierJudgeSpec,
    *,
    backend: str = DEFAULT_BACKEND,
    truefoundry_settings: TrueFoundrySettings | None = None,
    model_id: str,
    concurrency: int,
    timeout: float,
    max_output_tokens: int,
    reasoning_level: str,
    max_retries: int,
    retry_base_seconds: float,
) -> BaseFrontierGenerator:
    backend = _validate_backend(backend)
    api_key, _ = _credential(spec, backend)
    common_kwargs = dict(
        spec=spec,
        model_id=model_id,
        api_key=api_key,
        concurrency=concurrency,
        timeout=timeout,
        max_output_tokens=max_output_tokens,
        reasoning_level=reasoning_level,
        max_retries=max_retries,
        retry_base_seconds=retry_base_seconds,
    )
    if backend == "truefoundry":
        settings = truefoundry_settings or resolve_truefoundry_settings()
        return TrueFoundryGenerator(settings=settings, **common_kwargs)
    return GENERATOR_TYPES[spec.provider](**common_kwargs)


def _runtime_metadata() -> dict:
    packages = {}
    for package in ("openai", "anthropic", "google-genai", "python-dotenv"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }


def _configuration(
    *,
    spec: FrontierJudgeSpec,
    backend: str,
    truefoundry_settings: TrueFoundrySettings | None,
    model_id: str,
    model_id_source: str,
    wave: str,
    cases_path: Path,
    routing_path: Path,
    concurrency: int,
    timeout: float,
    max_output_tokens: int,
    reasoning_level: str,
    max_retries: int,
    retry_base_seconds: float,
) -> dict:
    backend = _validate_backend(backend)
    if backend == "truefoundry" and truefoundry_settings is None:
        raise ValueError("truefoundry settings are required for this backend")
    if backend == "direct" and truefoundry_settings is not None:
        raise ValueError("truefoundry settings must be absent for direct backend")
    prompt_variant, replicate_id = WAVES[wave]
    revision = (
        spec.revision
        if backend == "direct" and model_id == spec.default_model_id
        else None
    )
    return {
        "judge_name": spec.name,
        "judge_family": spec.family,
        "provider": spec.provider,
        "backend": backend,
        "base_url": (
            truefoundry_settings.base_url if truefoundry_settings else None
        ),
        "model_id": model_id,
        "model_id_source": model_id_source,
        "model_id_env": _backend_model_env(spec, backend),
        "revision": revision,
        "api_surface": _backend_api_surface(spec, backend),
        "api_key_envs": list(_backend_api_key_envs(spec, backend)),
        "adapter": ADAPTER,
        "prompt_version": base.PROMPT_VERSION,
        "evidence_policy_version": base.EVIDENCE_POLICY_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
        "routing_version": ROUTING_VERSION,
        "prompt_variant": prompt_variant,
        "replicate_id": replicate_id,
        "runner_sha256": base.file_sha256(__file__),
        "prompt_source_sha256": base.file_sha256(base.__file__),
        "cases_sha256": base.file_sha256(cases_path),
        "routing_sha256": base.file_sha256(routing_path),
        "local_judgment_schema_sha256": base.stable_hash(JUDGMENT_SCHEMA),
        "structured_output_schema_sha256": (
            None
            if backend == "truefoundry"
            else base.stable_hash(JUDGMENT_SCHEMA)
        ),
        "generation": {
            "max_output_tokens": max_output_tokens,
            "token_param": (
                truefoundry_settings.token_param
                if truefoundry_settings
                else "provider_native"
            ),
            "reasoning_level": (
                None if backend == "truefoundry" else reasoning_level
            ),
            "temperature": (
                truefoundry_settings.temperature
                if truefoundry_settings
                else None
            ),
        },
        "concurrency": concurrency,
        "timeout": timeout,
        "max_retries": max_retries,
        "retry_base_seconds": retry_base_seconds,
        "own_family_excluded": spec.family,
    }


def _load_existing_rows(
    path: Path,
    *,
    selected: Sequence[dict],
    routes: Mapping[str, Route],
    judge: FrontierJudgeSpec,
    configuration_hash: str,
    prompt_variant: str,
) -> dict[str, dict]:
    rows = base._load_resume_rows(path, recover_truncated_tail=True)
    cases_by_id = {str(case["case_id"]): case for case in selected}
    existing: dict[str, dict] = {}
    prompt_spec = base.JudgeSpec(
        name=judge.name,
        model_id="frontier-api",
        revision="",
        adapter=ADAPTER,
        description="frontier API judge",
    )
    for line_number, row in enumerate(rows, 1):
        case_id = str(row.get("case_id") or "").strip()
        if not case_id or case_id in existing:
            raise ValueError(f"{path}:{line_number}: blank or duplicate case_id {case_id!r}")
        case = cases_by_id.get(case_id)
        if case is None:
            raise ValueError(f"{path}:{line_number}: unexpected case_id {case_id!r}")
        route = routes[case_id]
        if route.candidate_family == judge.family:
            raise ValueError(f"{path}:{line_number}: same-family judgment for {case_id}")
        if row.get("candidate_family") != route.candidate_family:
            raise ValueError(f"{path}:{line_number}: candidate_family changed for {case_id}")
        if row.get("configuration_hash") != configuration_hash:
            raise ValueError(f"{path}:{line_number}: configuration changed for {case_id}")
        if row.get("input_hash") != route.input_hash:
            raise ValueError(f"{path}:{line_number}: input_hash changed for {case_id}")
        prompt = base.build_variant_messages(case, prompt_spec, prompt_variant)
        if row.get("prompt_hash") != base.stable_hash(prompt):
            raise ValueError(f"{path}:{line_number}: prompt changed for {case_id}")
        if row.get("judge_name") != judge.name or row.get("judge_family") != judge.family:
            raise ValueError(f"{path}:{line_number}: judge metadata changed for {case_id}")
        existing[case_id] = row
    return existing


def _archive_retry_rows(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    archive = path.with_name(f"{path.stem}.retry_history.jsonl")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with archive.open("a", encoding="utf-8") as handle:
        for source in rows:
            row = dict(source)
            row["retry_archived_at"] = base.utc_now()
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


GeneratorFactory = Callable[..., BaseFrontierGenerator]


def run_wave(
    *,
    cases_path: Path,
    routing_path: Path,
    judge_name: str,
    wave: str,
    output_path: Path,
    backend: str = DEFAULT_BACKEND,
    model_id: str | None = None,
    tfy_token_param: str | None = None,
    tfy_temperature_mode: str | None = None,
    concurrency: int = 4,
    batch_size: int = 8,
    timeout: float = 300.0,
    max_output_tokens: int = 4096,
    reasoning_level: str = "medium",
    max_retries: int = 4,
    retry_base_seconds: float = 1.0,
    resume: bool = False,
    retry_errors: bool = False,
    limit: int | None = None,
    generator_factory: GeneratorFactory = build_generator,
) -> dict:
    backend = _validate_backend(backend)
    if backend == "direct" and (tfy_token_param or tfy_temperature_mode):
        raise ValueError(
            "TrueFoundry request-profile overrides require --backend truefoundry"
        )
    truefoundry_settings = (
        resolve_truefoundry_settings(
            token_param=tfy_token_param,
            temperature_mode=tfy_temperature_mode,
        )
        if backend == "truefoundry"
        else None
    )
    if judge_name not in FRONTIER_JUDGES:
        raise ValueError(f"unknown frontier judge {judge_name!r}")
    if wave not in WAVES:
        raise ValueError(f"unknown wave {wave!r}")
    if concurrency < 1 or batch_size < 1 or max_output_tokens < 1:
        raise ValueError("concurrency, batch_size, and max_output_tokens must be positive")
    if timeout <= 0 or max_retries < 0 or retry_base_seconds < 0:
        raise ValueError("timeout must be positive; retry settings must be nonnegative")
    if reasoning_level != "medium":
        raise ValueError(
            "the frozen cross-provider study uses medium reasoning; "
            "start a separately named study to test another setting"
        )
    if retry_errors and not resume:
        raise ValueError("--retry-errors requires --resume")

    spec = FRONTIER_JUDGES[judge_name]
    effective_model, model_id_source = effective_model_id(
        spec, model_id, backend=backend
    )
    artifact_redaction_values = _artifact_redaction_values(spec, backend)
    cases, routes = load_and_validate_inputs(cases_path, routing_path)
    selected, excluded = eligible_cases(cases, routes, spec, limit=limit)
    prompt_variant, replicate_id = WAVES[wave]
    configuration = _configuration(
        spec=spec,
        backend=backend,
        truefoundry_settings=truefoundry_settings,
        model_id=effective_model,
        model_id_source=model_id_source,
        wave=wave,
        cases_path=cases_path,
        routing_path=routing_path,
        concurrency=concurrency,
        timeout=timeout,
        max_output_tokens=max_output_tokens,
        reasoning_level=reasoning_level,
        max_retries=max_retries,
        retry_base_seconds=retry_base_seconds,
    )
    configuration_hash = base.stable_hash(configuration)
    frozen_configuration_hash = base.frozen_configuration_hash(configuration)
    manifest_path = output_path.with_suffix(".manifest.json")
    if not resume and (output_path.exists() or manifest_path.exists()):
        existing = [str(path) for path in (output_path, manifest_path) if path.exists()]
        raise FileExistsError(
            f"output already exists: {', '.join(existing)}; pass --resume"
        )

    prompt_spec = base.JudgeSpec(
        name=spec.name,
        model_id=effective_model,
        revision=configuration.get("revision") or "",
        adapter=ADAPTER,
        description="frontier API judge",
    )
    existing_rows: dict[str, dict] = {}
    if output_path.exists():
        existing_rows = _load_existing_rows(
            output_path,
            selected=selected,
            routes=routes,
            judge=spec,
            configuration_hash=configuration_hash,
            prompt_variant=prompt_variant,
        )
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("configuration_hash") != configuration_hash:
            raise ValueError(f"{manifest_path}: configuration changed; use a new output")

    next_attempt: dict[str, int] = {}
    if retry_errors:
        retry_rows = [
            row
            for row in existing_rows.values()
            if row.get("status") != "ok" or row.get("verdict") not in {"pass", "fail"}
        ]
        _archive_retry_rows(output_path, retry_rows)
        for row in retry_rows:
            case_id = str(row["case_id"])
            next_attempt[case_id] = int(row.get("attempt") or 1) + 1
            existing_rows.pop(case_id)
        if retry_rows:
            _write_jsonl(
                output_path,
                [
                    existing_rows[str(case["case_id"])]
                    for case in selected
                    if str(case["case_id"]) in existing_rows
                ],
            )

    manifest = {
        "status": "starting",
        "started_at": base.utc_now(),
        "cases_file": str(cases_path),
        "routing_file": str(routing_path),
        "output_file": str(output_path),
        "judge_name": spec.name,
        "judge_family": spec.family,
        "backend": backend,
        "base_url": (
            truefoundry_settings.base_url if truefoundry_settings else None
        ),
        "request_profile": (
            {
                "temperature": truefoundry_settings.temperature,
                "token_param": truefoundry_settings.token_param,
            }
            if truefoundry_settings
            else None
        ),
        "wave": wave,
        "prompt_variant": prompt_variant,
        "replicate_id": replicate_id,
        "eligible_case_count": len(selected),
        "own_family_excluded_count": excluded,
        "configuration": configuration,
        "configuration_hash": configuration_hash,
        "frozen_configuration_hash": frozen_configuration_hash,
        "runtime": _runtime_metadata(),
    }
    _write_json(manifest_path, manifest)

    pending = [
        case for case in selected if str(case["case_id"]) not in existing_rows
    ]
    generator: BaseFrontierGenerator | None = None
    written = 0
    started = time.monotonic()
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        base._ensure_trailing_newline(output_path)
        if pending:
            generator = generator_factory(
                spec,
                backend=backend,
                truefoundry_settings=truefoundry_settings,
                model_id=effective_model,
                concurrency=concurrency,
                timeout=timeout,
                max_output_tokens=max_output_tokens,
                reasoning_level=reasoning_level,
                max_retries=max_retries,
                retry_base_seconds=retry_base_seconds,
            )
        with output_path.open("a", encoding="utf-8") as handle:
            for start in range(0, len(pending), batch_size):
                batch = pending[start : start + batch_size]
                messages = [
                    base.build_variant_messages(case, prompt_spec, prompt_variant)
                    for case in batch
                ]
                assert generator is not None
                results = generator.generate(messages)
                if len(results) != len(batch):
                    raise RuntimeError(
                        f"generator returned {len(results)} results for {len(batch)} cases"
                    )
                for case, case_messages, generated in zip(batch, messages, results):
                    case_id = str(case["case_id"])
                    route = routes[case_id]
                    if route.candidate_family == spec.family:
                        raise AssertionError(f"same-family case reached API: {case_id}")
                    if generated.error:
                        parsed = base.ParsedJudgment(
                            verdict="no_decision",
                            status="generation_error",
                            error=generated.error,
                        )
                    else:
                        parsed = parse_frontier_judgment(generated.text)
                    row = {
                        "case_id": case_id,
                        "response_id": case.get("response_id", ""),
                        "scenario_id": case.get("scenario_id", ""),
                        "criterion_id": case.get("criterion_id", ""),
                        "candidate_family": route.candidate_family,
                        "judge_name": spec.name,
                        "judge_family": spec.family,
                        "judge_model": effective_model,
                        "judge_revision": configuration.get("revision"),
                        "served_model": effective_model,
                        "resolved_provider_model": generated.resolved_model,
                        "provider": spec.provider,
                        "backend": backend,
                        "base_url": (
                            truefoundry_settings.base_url
                            if truefoundry_settings
                            else None
                        ),
                        "api_surface": _backend_api_surface(spec, backend),
                        "request_token_param": (
                            truefoundry_settings.token_param
                            if truefoundry_settings
                            else "provider_native"
                        ),
                        "request_temperature": (
                            truefoundry_settings.temperature
                            if truefoundry_settings
                            else None
                        ),
                        "checkpoint_provenance": (
                            "truefoundry_gateway_configured_model"
                            if backend == "truefoundry"
                            else "provider_api_configured_model"
                        ),
                        "checkpoint_verified_by_runner": False,
                        "adapter": ADAPTER,
                        "prompt_version": base.PROMPT_VERSION,
                        "evidence_policy_version": base.EVIDENCE_POLICY_VERSION,
                        "prompt_variant": prompt_variant,
                        "replicate_id": replicate_id,
                        "normalization_version": NORMALIZATION_VERSION,
                        "routing_version": ROUTING_VERSION,
                        "configuration_hash": configuration_hash,
                        "frozen_configuration_hash": frozen_configuration_hash,
                        "input_hash": route.input_hash,
                        "prompt_hash": base.stable_hash(case_messages),
                        "attempt": next_attempt.get(case_id, 1),
                        "provider_attempts": generated.provider_attempts,
                        "verdict": parsed.verdict,
                        "native_score": parsed.native_score,
                        "rationale": parsed.rationale,
                        "evidence": parsed.evidence,
                        "status": parsed.status,
                        "error": parsed.error,
                        "raw_output": generated.text,
                        "raw_response": _json_safe(generated.raw_response),
                        "usage": _json_safe(generated.usage),
                        "latency_ms": generated.latency_ms,
                        "created_at": base.utc_now(),
                    }
                    safe_row = _redact_json(row, artifact_redaction_values)
                    handle.write(json.dumps(safe_row, ensure_ascii=False) + "\n")
                    written += 1
                handle.flush()
                os.fsync(handle.fileno())
    except BaseException as exc:
        safe_error = _redact(
            f"{type(exc).__name__}: {exc}", artifact_redaction_values
        )
        manifest.update(
            {
                "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                "finished_at": base.utc_now(),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "error": safe_error,
            }
        )
        _write_json(manifest_path, manifest)
        if isinstance(exc, KeyboardInterrupt):
            raise
        unsafe_error = f"{type(exc).__name__}: {exc}"
        if safe_error != unsafe_error:
            raise RuntimeError(safe_error) from None
        raise
    finally:
        if generator is not None:
            generator.close()

    final_rows = _load_existing_rows(
        output_path,
        selected=selected,
        routes=routes,
        judge=spec,
        configuration_hash=configuration_hash,
        prompt_variant=prompt_variant,
    )
    usable = sum(
        row.get("status") == "ok" and row.get("verdict") in {"pass", "fail"}
        for row in final_rows.values()
    )
    no_decision = len(final_rows) - usable
    resolved_provider_models = sorted(
        {
            str(row.get("resolved_provider_model") or "").strip()
            for row in final_rows.values()
            if str(row.get("resolved_provider_model") or "").strip()
        }
    )
    missing_resolved_model_rows = sum(
        not str(row.get("resolved_provider_model") or "").strip()
        for row in final_rows.values()
    )
    completion_status = "complete" if no_decision == 0 else "complete_with_errors"
    if usable == 0:
        completion_status = "failed_no_usable_decisions"
    elif len(resolved_provider_models) > 1:
        completion_status = "failed_model_drift"
    elif no_decision == 0 and missing_resolved_model_rows:
        completion_status = "failed_missing_model_provenance"
    manifest.update(
        {
            "status": completion_status,
            "completed_at": base.utc_now(),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "new_rows": written,
            "resumed_rows": len(existing_rows),
            "usable_decisions": usable,
            "no_decision_rows": no_decision,
            "resolved_provider_models": resolved_provider_models,
            "missing_resolved_model_rows": missing_resolved_model_rows,
            "model_provenance_consistent": (
                len(resolved_provider_models) == 1
                and missing_resolved_model_rows == 0
            ),
        }
    )
    _write_json(manifest_path, manifest)
    return manifest


def run_suite(
    *,
    cases_path: Path,
    routing_path: Path,
    output_dir: Path,
    judges: Sequence[str],
    backend: str = DEFAULT_BACKEND,
    model_ids: Mapping[str, str | None] | None = None,
    generator_factory: GeneratorFactory = build_generator,
    **run_options: object,
) -> list[dict]:
    backend = _validate_backend(backend)
    unknown = sorted(set(judges) - set(FRONTIER_JUDGES))
    if unknown:
        raise ValueError(f"unknown frontier judge(s): {', '.join(unknown)}")
    reports: list[dict] = []
    overrides = model_ids or {}
    for judge_name in judges:
        for wave in WAVES:
            output = output_dir / judge_name / wave / f"{wave}.jsonl"
            print(f"Starting {judge_name}/{wave}")
            with base.output_lock(output):
                report = run_wave(
                    cases_path=cases_path,
                    routing_path=routing_path,
                    judge_name=judge_name,
                    wave=wave,
                    output_path=output,
                    backend=backend,
                    model_id=overrides.get(judge_name),
                    generator_factory=generator_factory,
                    **run_options,
                )
            reports.append(report)
            print(
                f"Completed {judge_name}/{wave}: "
                f"{report['usable_decisions']}/{report['eligible_case_count']} usable"
            )
    return reports


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def _add_run_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--concurrency", type=_positive_int, default=4)
    parser.add_argument("--batch-size", type=_positive_int, default=8)
    parser.add_argument("--timeout", type=_positive_float, default=300.0)
    parser.add_argument("--max-output-tokens", type=_positive_int, default=4096)
    parser.add_argument("--max-retries", type=_nonnegative_int, default=4)
    parser.add_argument(
        "--retry-base-seconds", type=_nonnegative_float, default=1.0
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--limit", type=_positive_int, help="smoke-test eligible cases")


def _add_truefoundry_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--tfy-token-param",
        choices=TRUEFOUNDRY_TOKEN_PARAMS,
        help=(
            "frozen gateway token-limit field; otherwise TFY_TOKEN_PARAM or "
            "max_completion_tokens"
        ),
    )
    parser.add_argument(
        "--tfy-temperature-mode",
        choices=TRUEFOUNDRY_TEMPERATURE_MODES,
        help=(
            "send temperature=0 or omit it; otherwise TFY_TEMPERATURE_MODE or zero"
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser(
        "prepare", help="create blinded cases and family-only routing", allow_abbrev=False
    )
    prepare.add_argument("--cases", required=True, type=Path)
    prepare.add_argument("--human-labels", required=True, type=Path)
    prepare.add_argument("--out-dir", required=True, type=Path)
    prepare.add_argument("--overwrite", action="store_true")

    plan = commands.add_parser(
        "plan", help="validate routing and show call counts without API keys", allow_abbrev=False
    )
    plan.add_argument("--cases", required=True, type=Path)
    plan.add_argument("--routing", required=True, type=Path)
    plan.add_argument("--json-out", type=Path)
    plan.add_argument("--backend", choices=BACKENDS, default=DEFAULT_BACKEND)
    _add_truefoundry_options(plan)

    run = commands.add_parser(
        "run", help="run one frontier judge wave", allow_abbrev=False
    )
    run.add_argument("--cases", required=True, type=Path)
    run.add_argument("--routing", required=True, type=Path)
    run.add_argument("--judge", required=True, choices=tuple(FRONTIER_JUDGES))
    run.add_argument("--wave", choices=tuple(WAVES), default="canonical_r1")
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--backend", choices=BACKENDS, default=DEFAULT_BACKEND)
    run.add_argument("--model-id")
    _add_truefoundry_options(run)
    _add_run_options(run)

    suite = commands.add_parser(
        "suite", help="run all six waves for one or all judges", allow_abbrev=False
    )
    suite.add_argument("--cases", required=True, type=Path)
    suite.add_argument("--routing", required=True, type=Path)
    suite.add_argument("--output-dir", required=True, type=Path)
    suite.add_argument(
        "--judge", choices=("all", *FRONTIER_JUDGES), default="all"
    )
    suite.add_argument("--backend", choices=BACKENDS, default=DEFAULT_BACKEND)
    suite.add_argument("--gpt-model-id")
    suite.add_argument("--opus-model-id")
    suite.add_argument("--gemini-model-id")
    _add_truefoundry_options(suite)
    _add_run_options(suite)
    return parser


def _run_options(args: argparse.Namespace) -> dict:
    return {
        "concurrency": args.concurrency,
        "batch_size": args.batch_size,
        "timeout": args.timeout,
        "max_output_tokens": args.max_output_tokens,
        "reasoning_level": "medium",
        "max_retries": args.max_retries,
        "retry_base_seconds": args.retry_base_seconds,
        "resume": args.resume,
        "retry_errors": args.retry_errors,
        "limit": args.limit,
        "tfy_token_param": args.tfy_token_param,
        "tfy_temperature_mode": args.tfy_temperature_mode,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            report = prepare_artifacts(
                args.cases,
                args.human_labels,
                args.out_dir,
                overwrite=args.overwrite,
            )
            print(json.dumps(report, indent=2))
            return 0
        if args.command == "plan":
            _load_model_env_only()
            report = build_plan(
                args.cases,
                args.routing,
                backend=args.backend,
                tfy_token_param=args.tfy_token_param,
                tfy_temperature_mode=args.tfy_temperature_mode,
            )
            if args.json_out:
                _write_json(args.json_out, report)
            print(json.dumps(report, indent=2))
            return 0

        _load_dotenv()
        if args.command == "run":
            with base.output_lock(args.output):
                report = run_wave(
                    cases_path=args.cases,
                    routing_path=args.routing,
                    judge_name=args.judge,
                    wave=args.wave,
                    output_path=args.output,
                    backend=args.backend,
                    model_id=args.model_id,
                    **_run_options(args),
                )
            print(json.dumps(report, indent=2))
            return 3 if report["status"] != "complete" else 0

        judges = (
            list(FRONTIER_JUDGES) if args.judge == "all" else [args.judge]
        )
        reports = run_suite(
            cases_path=args.cases,
            routing_path=args.routing,
            output_dir=args.output_dir,
            judges=judges,
            backend=args.backend,
            model_ids={
                "gpt-5.5": args.gpt_model_id,
                "opus-4.8": args.opus_model_id,
                "gemini-3.6-flash": args.gemini_model_id,
            },
            **_run_options(args),
        )
        return 3 if any(report["status"] != "complete" for report in reports) else 0
    except (FileExistsError, OSError, RuntimeError, ValueError) as exc:
        safe_error = _redact(str(exc), _all_api_key_values())
        print(f"error: {safe_error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
