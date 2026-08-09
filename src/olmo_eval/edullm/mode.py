"""Thin OLMo-owned execution mode for the EduLLM adaptive pipeline.

OLMo supplies and owns the inference-provider lifecycle.  This adapter validates
an explicit fitted-bank and scientific runtime configuration, obtains tutor
responses either from the primary provider or a prevalidated JSONL snapshot,
applies the frozen Qwen binary judge one criterion at a time, and delegates all
state transitions to ``run_cat``.  It never creates a model client, launches
vLLM, or calls a cloud service directly.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from olmo_eval.edullm.ability import Quadrature, build_quadrature
from olmo_eval.edullm.bank import FittedBank, Rubric, Scenario, load_fitted_bank, sha256_file
from olmo_eval.edullm.batch_report import (
    BATCH_REPORT_MARKDOWN,
    BATCH_RESULTS_CSV,
    BATCH_RESULTS_JSON,
    build_batch_report,
    write_batch_reports,
)
from olmo_eval.edullm.batch_resume import (
    BatchCheckpointState,
    allocate_attempt_path,
    commit_checkpoint,
    load_checkpoint_chain,
    read_strict_json_object,
    repair_checkpoint_pointer,
    safe_relative_path,
    verify_hashed_artifacts,
)
from olmo_eval.edullm.cat import CatConfig, CatResult, ScenarioStep, run_cat
from olmo_eval.edullm.judge import (
    ADAPTER_VERSION,
    FAILURE_PROBABILITY_THRESHOLD,
    PROMPT_PROFILE,
    PROMPT_VARIANT,
    PROMPT_VERSION,
    QWEN_JUDGE_MODEL,
    QWEN_JUDGE_REVISION,
    AtomicJudgeResult,
    AtomicRequirement,
    BlindedJudgeCase,
    CriterionJudgeResult,
    QwenZeroShotBinaryJudge,
    build_atomic_messages,
    build_classification_messages,
    judge_prompt_contract_payload,
)
from olmo_eval.edullm.precomputed import (
    PRECOMPUTED_RESPONSE_BATCH_SCHEMA_VERSION,
    PRECOMPUTED_RESPONSES_SCHEMA_VERSION,
    PrecomputedTutorResponse,
    PrecomputedTutorResponseBatch,
    PrecomputedTutorResponseBatchModel,
    PrecomputedTutorResponses,
    load_precomputed_tutor_response_batch,
    load_precomputed_tutor_responses,
)
from olmo_eval.edullm.tutor import (
    TutorGenerationConfig,
    build_tutor_messages,
    generate_tutor_response,
)
from olmo_eval.runners.modes import ModeResult, ModeRunContext, ModeStatus

MODE_NAME = "edullm_adaptive"
IMPLEMENTATION_VERSION = "edullm-adaptive-olmo-v4"
MODE_CONFIG_SCHEMA_VERSION = "edullm-adaptive-mode-config-v1"
ARTIFACT_SCHEMA_VERSION = "edullm-adaptive-artifacts-v1"
BATCH_ARTIFACT_SCHEMA_VERSION = "edullm-adaptive-batch-artifacts-v2"
BATCH_PROGRESS_SCHEMA_VERSION = "edullm-adaptive-batch-progress-v1"
ATOMIC_REQUIREMENT_POLICY = "criterion_as_single_atomic_unless_curation_finalized"
TUTOR_PROMPT_SOURCE = "origin/frq/infobench:eduLLM-Evals/tutor_cat/respgen/prompts.py"
TUTOR_PROMPT_VERSION = "frq-infobench-prompts-b4ea2e8"

_CURATION_FINALIZED_STATUS = "curation_v1_finalized"
_SCIENCE_STATUSES = frozenset({"validated", "experimental"})
_HEX_SHA256 = frozenset("0123456789abcdef")

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BankRuntimeConfig:
    rubrics_path: Path
    scenarios_path: Path
    manifest_path: Path
    skills_order: tuple[str, ...]
    benchmark_id: str
    calibration_version: str
    policy_id: str
    scientific_status: Literal["validated", "experimental"]


@dataclass(frozen=True, slots=True)
class TutorResponseSourceConfig:
    kind: Literal["provider", "precomputed_jsonl", "precomputed_batch_jsonl"]
    schema_version: str | None = None
    path: Path | None = None
    sha256: str | None = None
    provenance: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class TutorRuntimeConfig:
    expected_model: str | None
    model_family: str | None
    model_provenance: Mapping[str, Any]
    generation: TutorGenerationConfig | None
    response_source: TutorResponseSourceConfig


@dataclass(frozen=True, slots=True)
class JudgeRuntimeConfig:
    provider: str
    model: str
    model_family: str
    revision: str
    prompt_version: str
    adapter_version: str
    enable_thinking: bool
    language_model_only: bool
    pass_token_ids: tuple[int, ...] | None
    fail_token_ids: tuple[int, ...] | None
    failure_probability_threshold: float
    atomic_requirement_policy: str


@dataclass(frozen=True, slots=True)
class QuadratureRuntimeConfig:
    nodes_per_dim: int
    max_nodes: int
    method: str
    linear_bound: float


@dataclass(frozen=True, slots=True)
class EduLLMAdaptiveConfig:
    bank: BankRuntimeConfig
    tutor: TutorRuntimeConfig
    judge: JudgeRuntimeConfig
    quadrature: QuadratureRuntimeConfig
    cat: CatConfig
    raw: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class RequirementPlan:
    requirements: tuple[AtomicRequirement, ...]
    source: Literal["criterion_text", "reviewed_atomic_checklist"]
    provenance: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class TutorIdentity:
    model: str
    model_family: str
    provenance: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _PreparedRun:
    config: EduLLMAdaptiveConfig
    bank: FittedBank
    bank_provenance: Mapping[str, Any]
    quadrature: Quadrature
    judge: QwenZeroShotBinaryJudge
    requirement_plans: Mapping[str, RequirementPlan]
    tutor_identity: TutorIdentity | None
    response_rows: Mapping[str, PrecomputedTutorResponse] | None
    single_precomputed_source: PrecomputedTutorResponses | None
    precomputed_batch: PrecomputedTutorResponseBatch | None
    batch_model: PrecomputedTutorResponseBatchModel | None = None


@dataclass(frozen=True, slots=True)
class _PreparedInputs:
    config: EduLLMAdaptiveConfig
    bank: FittedBank
    bank_provenance: Mapping[str, Any]
    quadrature: Quadrature
    requirement_plans: Mapping[str, RequirementPlan]
    single_precomputed_source: PrecomputedTutorResponses | None
    precomputed_batch: PrecomputedTutorResponseBatch | None


def _json_value(value: Any, *, path: str = "value") -> Any:
    """Convert supported values to strict JSON and reject lossy fallbacks."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{path} contains a non-finite NumPy float")
        return number
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} contains a non-string key")
            result[key] = _json_value(item, path=f"{path}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item, path=f"{path}[{index}]") for index, item in enumerate(value)]
    raise ValueError(f"{path} contains unsupported value {type(value).__name__}")


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        _json_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = _json_value(value, path=path.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    payload = "".join(
        json.dumps(
            _json_value(row, path=f"{path.name}[{index}]"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
        for index, row in enumerate(rows)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], required: set[str], label: str) -> None:
    missing = sorted(required - set(value))
    extra = sorted(set(value) - required)
    if missing or extra:
        raise ValueError(f"{label} fields differ; missing={missing}, extra={extra}")


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _sha256_string(value: Any, label: str) -> str:
    result = _nonempty_string(value, label)
    if len(result) != 64 or any(character not in _HEX_SHA256 for character in result):
        raise ValueError(f"{label} must be a lowercase 64-character SHA-256 digest")
    return result


def _integer(value: Any, label: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _optional_float_tuple(value: Any, label: str) -> tuple[float, ...] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be null or an array")
    return tuple(_finite_number(item, f"{label}[{index}]") for index, item in enumerate(value))


def _optional_token_ids(value: Any, label: str) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be null or an array")
    result = tuple(
        _integer(item, f"{label}[{index}]", minimum=0) for index, item in enumerate(value)
    )
    if not result:
        raise ValueError(f"{label} must not be an empty array")
    if len(result) != len(set(result)):
        raise ValueError(f"{label} must not contain duplicate token IDs")
    return result


def _threshold(value: Any, label: str) -> float | int | Mapping[str, float]:
    if isinstance(value, Mapping):
        return {
            _nonempty_string(key, f"{label} key"): _finite_number(item, f"{label}.{key}")
            for key, item in value.items()
        }
    return _finite_number(value, label)


def _parse_config(raw_config: Mapping[str, Any]) -> EduLLMAdaptiveConfig:
    top = _mapping(raw_config, "edullm_adaptive config")
    _json_value(top, path="edullm_adaptive config")
    _exact_keys(
        top,
        {"schema_version", "bank", "tutor", "judge", "quadrature", "cat"},
        "edullm_adaptive config",
    )
    if top["schema_version"] != MODE_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema_version {top['schema_version']!r}; "
            f"expected {MODE_CONFIG_SCHEMA_VERSION!r}"
        )

    bank_value = _mapping(top["bank"], "bank")
    _exact_keys(
        bank_value,
        {
            "rubrics_path",
            "scenarios_path",
            "manifest_path",
            "skills_order",
            "benchmark_id",
            "calibration_version",
            "policy_id",
            "scientific_status",
        },
        "bank",
    )
    raw_skills = bank_value["skills_order"]
    if isinstance(raw_skills, (str, bytes)) or not isinstance(raw_skills, Sequence):
        raise ValueError("bank.skills_order must be an array")
    skills = tuple(
        _nonempty_string(skill, f"bank.skills_order[{index}]")
        for index, skill in enumerate(raw_skills)
    )
    if not skills or len(skills) != len(set(skills)):
        raise ValueError("bank.skills_order must be non-empty and unique")
    science_status = _nonempty_string(bank_value["scientific_status"], "bank.scientific_status")
    if science_status not in _SCIENCE_STATUSES:
        raise ValueError("bank.scientific_status must be explicitly 'validated' or 'experimental'")
    bank = BankRuntimeConfig(
        rubrics_path=Path(_nonempty_string(bank_value["rubrics_path"], "bank.rubrics_path")),
        scenarios_path=Path(_nonempty_string(bank_value["scenarios_path"], "bank.scenarios_path")),
        manifest_path=Path(_nonempty_string(bank_value["manifest_path"], "bank.manifest_path")),
        skills_order=skills,
        benchmark_id=_nonempty_string(bank_value["benchmark_id"], "bank.benchmark_id"),
        calibration_version=_nonempty_string(
            bank_value["calibration_version"], "bank.calibration_version"
        ),
        policy_id=_nonempty_string(bank_value["policy_id"], "bank.policy_id"),
        scientific_status=cast(Literal["validated", "experimental"], science_status),
    )

    tutor_value = _mapping(top["tutor"], "tutor")
    raw_source = tutor_value.get("response_source", {"kind": "provider"})
    source_value = _mapping(raw_source, "tutor.response_source")
    source_kind = _nonempty_string(source_value.get("kind"), "tutor.response_source.kind")
    if source_kind == "precomputed_batch_jsonl":
        _exact_keys(tutor_value, {"generation", "response_source"}, "tutor")
        _exact_keys(
            source_value,
            {"kind", "schema_version", "path", "sha256", "provenance"},
            "tutor.response_source",
        )
        if tutor_value["generation"] is not None:
            raise ValueError("tutor.generation must be null for precomputed_batch_jsonl responses")
        schema_version = _nonempty_string(
            source_value["schema_version"], "tutor.response_source.schema_version"
        )
        if schema_version != PRECOMPUTED_RESPONSE_BATCH_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported tutor.response_source.schema_version {schema_version!r}; "
                f"expected {PRECOMPUTED_RESPONSE_BATCH_SCHEMA_VERSION!r}"
            )
        batch_provenance = _mapping(source_value["provenance"], "tutor.response_source.provenance")
        _exact_keys(
            batch_provenance,
            {"source", "revision"},
            "tutor.response_source.provenance",
        )
        _nonempty_string(batch_provenance.get("source"), "tutor.response_source.provenance.source")
        _nonempty_string(
            batch_provenance.get("revision"), "tutor.response_source.provenance.revision"
        )
        response_source = TutorResponseSourceConfig(
            kind="precomputed_batch_jsonl",
            schema_version=schema_version,
            path=Path(_nonempty_string(source_value["path"], "tutor.response_source.path")),
            sha256=_sha256_string(source_value["sha256"], "tutor.response_source.sha256"),
            provenance=dict(batch_provenance),
        )
        tutor = TutorRuntimeConfig(
            expected_model=None,
            model_family=None,
            model_provenance={},
            generation=None,
            response_source=response_source,
        )
    else:
        tutor_fields = {"expected_model", "model_family", "model_provenance", "generation"}
        if "response_source" in tutor_value:
            tutor_fields.add("response_source")
        _exact_keys(tutor_value, tutor_fields, "tutor")
        provenance = _mapping(tutor_value["model_provenance"], "tutor.model_provenance")
        _exact_keys(provenance, {"source", "revision"}, "tutor.model_provenance")
        _nonempty_string(provenance.get("source"), "tutor.model_provenance.source")
        _nonempty_string(provenance.get("revision"), "tutor.model_provenance.revision")
        if source_kind == "provider":
            _exact_keys(source_value, {"kind"}, "tutor.response_source")
            response_source = TutorResponseSourceConfig(kind="provider")
            generation_value = _mapping(tutor_value["generation"], "tutor.generation")
            _exact_keys(
                generation_value,
                {
                    "max_tokens",
                    "temperature",
                    "top_p",
                    "top_k",
                    "stop_sequences",
                    "num_samples",
                    "do_sample",
                },
                "tutor.generation",
            )
            raw_stops = generation_value["stop_sequences"]
            if raw_stops is None:
                stops = None
            elif isinstance(raw_stops, Sequence) and not isinstance(raw_stops, (str, bytes)):
                stops = tuple(
                    _nonempty_string(value, f"tutor.generation.stop_sequences[{index}]")
                    for index, value in enumerate(raw_stops)
                )
            else:
                raise ValueError("tutor.generation.stop_sequences must be null or an array")
            if not isinstance(generation_value["do_sample"], bool):
                raise ValueError("tutor.generation.do_sample must be boolean")
            top_p = generation_value["top_p"]
            top_k = generation_value["top_k"]
            generation: TutorGenerationConfig | None = TutorGenerationConfig(
                max_tokens=_integer(generation_value["max_tokens"], "tutor.max_tokens", minimum=1),
                temperature=_finite_number(generation_value["temperature"], "tutor.temperature"),
                top_p=None if top_p is None else _finite_number(top_p, "tutor.top_p"),
                top_k=None if top_k is None else _integer(top_k, "tutor.top_k", minimum=1),
                stop_sequences=stops,
                num_samples=_integer(
                    generation_value["num_samples"], "tutor.num_samples", minimum=1
                ),
                do_sample=generation_value["do_sample"],
            )
        elif source_kind == "precomputed_jsonl":
            _exact_keys(
                source_value,
                {"kind", "schema_version", "path", "sha256"},
                "tutor.response_source",
            )
            if tutor_value["generation"] is not None:
                raise ValueError("tutor.generation must be null for precomputed_jsonl responses")
            schema_version = _nonempty_string(
                source_value["schema_version"], "tutor.response_source.schema_version"
            )
            if schema_version != PRECOMPUTED_RESPONSES_SCHEMA_VERSION:
                raise ValueError(
                    f"unsupported tutor.response_source.schema_version {schema_version!r}; "
                    f"expected {PRECOMPUTED_RESPONSES_SCHEMA_VERSION!r}"
                )
            response_source = TutorResponseSourceConfig(
                kind="precomputed_jsonl",
                schema_version=schema_version,
                path=Path(_nonempty_string(source_value["path"], "tutor.response_source.path")),
                sha256=_sha256_string(source_value["sha256"], "tutor.response_source.sha256"),
            )
            generation = None
        else:
            raise ValueError(
                "tutor.response_source.kind must be 'provider', 'precomputed_jsonl', "
                "or 'precomputed_batch_jsonl'"
            )
        tutor = TutorRuntimeConfig(
            expected_model=_nonempty_string(tutor_value["expected_model"], "tutor.expected_model"),
            model_family=_nonempty_string(tutor_value["model_family"], "tutor.model_family"),
            model_provenance=dict(provenance),
            generation=generation,
            response_source=response_source,
        )

    judge_value = _mapping(top["judge"], "judge")
    _exact_keys(
        judge_value,
        {
            "provider",
            "model",
            "model_family",
            "revision",
            "prompt_version",
            "adapter_version",
            "enable_thinking",
            "language_model_only",
            "pass_token_ids",
            "fail_token_ids",
            "failure_probability_threshold",
            "atomic_requirement_policy",
        },
        "judge",
    )
    judge = JudgeRuntimeConfig(
        provider=_nonempty_string(judge_value["provider"], "judge.provider"),
        model=_nonempty_string(judge_value["model"], "judge.model"),
        model_family=_nonempty_string(judge_value["model_family"], "judge.model_family"),
        revision=_nonempty_string(judge_value["revision"], "judge.revision"),
        prompt_version=_nonempty_string(judge_value["prompt_version"], "judge.prompt_version"),
        adapter_version=_nonempty_string(judge_value["adapter_version"], "judge.adapter_version"),
        enable_thinking=judge_value["enable_thinking"],
        language_model_only=judge_value["language_model_only"],
        pass_token_ids=_optional_token_ids(judge_value["pass_token_ids"], "judge.pass_token_ids"),
        fail_token_ids=_optional_token_ids(judge_value["fail_token_ids"], "judge.fail_token_ids"),
        failure_probability_threshold=_finite_number(
            judge_value["failure_probability_threshold"],
            "judge.failure_probability_threshold",
        ),
        atomic_requirement_policy=_nonempty_string(
            judge_value["atomic_requirement_policy"], "judge.atomic_requirement_policy"
        ),
    )
    if not isinstance(judge.enable_thinking, bool):
        raise ValueError("judge.enable_thinking must be boolean")
    if not isinstance(judge.language_model_only, bool):
        raise ValueError("judge.language_model_only must be boolean")
    if (judge.pass_token_ids is None) != (judge.fail_token_ids is None):
        raise ValueError("judge.pass_token_ids and judge.fail_token_ids must be supplied together")

    quadrature_value = _mapping(top["quadrature"], "quadrature")
    _exact_keys(
        quadrature_value,
        {"nodes_per_dim", "max_nodes", "method", "linear_bound"},
        "quadrature",
    )
    quadrature = QuadratureRuntimeConfig(
        nodes_per_dim=_integer(
            quadrature_value["nodes_per_dim"], "quadrature.nodes_per_dim", minimum=2
        ),
        max_nodes=_integer(quadrature_value["max_nodes"], "quadrature.max_nodes", minimum=2),
        method=_nonempty_string(quadrature_value["method"], "quadrature.method"),
        linear_bound=_finite_number(quadrature_value["linear_bound"], "quadrature.linear_bound"),
    )

    cat_value = _mapping(top["cat"], "cat")
    _exact_keys(
        cat_value,
        {
            "max_se",
            "min_evals_per_skill",
            "min_scenarios",
            "max_scenarios",
            "seed",
            "top_n",
            "selection",
            "stop_se_method",
            "theta_init",
            "covariance_init_diag",
            "mwle_ridge",
        },
        "cat",
    )
    raw_min_counts = cat_value["min_evals_per_skill"]
    if isinstance(raw_min_counts, Mapping):
        parsed_min_counts: int | dict[str, int] = {}
        for key, value in raw_min_counts.items():
            skill = _nonempty_string(key, "cat.min_evals_per_skill key")
            number = _finite_number(value, f"cat.min_evals_per_skill.{skill}")
            if not number.is_integer():
                raise ValueError("cat.min_evals_per_skill values must be integers")
            parsed_min_counts[skill] = int(number)
    else:
        number = _finite_number(raw_min_counts, "cat.min_evals_per_skill")
        if not number.is_integer():
            raise ValueError("cat.min_evals_per_skill scalar must be an integer")
        parsed_min_counts = int(number)
    selection = _nonempty_string(cat_value["selection"], "cat.selection")
    if selection not in {"trace", "dopt"}:
        raise ValueError("cat.selection must be 'trace' or 'dopt'")
    stop_se_method = _nonempty_string(cat_value["stop_se_method"], "cat.stop_se_method")
    if stop_se_method not in {"eap", "online"}:
        raise ValueError("cat.stop_se_method must be 'eap' or 'online'")
    cat = CatConfig(
        max_se=_threshold(cat_value["max_se"], "cat.max_se"),
        min_evals_per_skill=parsed_min_counts,
        min_scenarios=_integer(cat_value["min_scenarios"], "cat.min_scenarios", minimum=0),
        max_scenarios=_integer(cat_value["max_scenarios"], "cat.max_scenarios", minimum=1),
        seed=_integer(cat_value["seed"], "cat.seed", minimum=0),
        top_n=_integer(cat_value["top_n"], "cat.top_n", minimum=1),
        selection=cast(Literal["trace", "dopt"], selection),
        stop_se_method=cast(Literal["eap", "online"], stop_se_method),
        theta_init=_optional_float_tuple(cat_value["theta_init"], "cat.theta_init"),
        covariance_init_diag=_optional_float_tuple(
            cat_value["covariance_init_diag"], "cat.covariance_init_diag"
        ),
        mwle_ridge=_finite_number(cat_value["mwle_ridge"], "cat.mwle_ridge"),
    )
    if cat.stop_se_method != "eap":
        raise ValueError("EduLLM's frozen adaptive policy requires joint-EAP stopping")

    return EduLLMAdaptiveConfig(
        bank=bank,
        tutor=tutor,
        judge=judge,
        quadrature=quadrature,
        cat=cat,
        raw=dict(top),
    )


def parse_adaptive_config(raw_config: Mapping[str, Any]) -> EduLLMAdaptiveConfig:
    """Validate the complete public adaptive-mode configuration without I/O."""

    return _parse_config(raw_config)


def _record_map(bank: FittedBank) -> dict[str, Mapping[str, Any]]:
    records: dict[str, Mapping[str, Any]] = {}
    for record in bank.records:
        criterion_id = str(record.get("criterion_id") or "")
        if criterion_id:
            records[criterion_id] = record
    if set(records) != set(bank.rubrics):
        raise ValueError("fitted-bank provenance records do not align with loaded rubrics")
    return records


def _reviewed_checklist(record: Mapping[str, Any]) -> tuple[Any, Any, Any, Any, Any]:
    nested = record.get("atomic_checklist")
    legacy = record.get("atomic_requirements")
    if nested is not None and legacy is not None:
        raise ValueError("criterion declares both atomic_checklist and atomic_requirements")
    if nested is not None:
        checklist = _mapping(nested, "atomic_checklist")
        return (
            checklist.get("requirements"),
            checklist.get("review_status"),
            checklist.get("reviewed_by"),
            checklist.get("source"),
            checklist.get("sha256"),
        )
    return (
        legacy,
        record.get("atomic_requirements_review_status"),
        record.get("atomic_requirements_reviewed_by"),
        record.get("atomic_requirements_source")
        or record.get("atomic_requirements_proposal_source"),
        record.get("atomic_requirements_sha256"),
    )


def _requirement_plan(rubric: Rubric, record: Mapping[str, Any]) -> RequirementPlan:
    raw_requirements, raw_status, raw_reviewer, raw_source, raw_digest = _reviewed_checklist(record)
    if raw_requirements is None:
        return RequirementPlan(
            requirements=(AtomicRequirement("R1", rubric.criterion),),
            source="criterion_text",
            provenance={"reviewed_atomic_checklist_used": False},
        )
    if isinstance(raw_requirements, (str, bytes)) or not isinstance(raw_requirements, Sequence):
        raise ValueError(f"{rubric.criterion_id}: atomic requirements must be an array")
    normalized = tuple(str(item).strip() for item in raw_requirements)
    if not normalized or len(normalized) > 99 or any(not item for item in normalized):
        raise ValueError(
            f"{rubric.criterion_id}: atomic requirements must contain 1 through 99 nonblank items"
        )
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{rubric.criterion_id}: duplicate atomic requirements")

    status = str(raw_status or "").strip()
    if status != _CURATION_FINALIZED_STATUS:
        return RequirementPlan(
            requirements=(AtomicRequirement("R1", rubric.criterion),),
            source="criterion_text",
            provenance={
                "reviewed_atomic_checklist_used": False,
                "ignored_checklist_review_status": status or None,
            },
        )

    source = _nonempty_string(raw_source, f"{rubric.criterion_id} atomic checklist source")
    digest = _nonempty_string(raw_digest, f"{rubric.criterion_id} atomic checklist sha256")
    if len(digest) != 64 or any(character not in _HEX_SHA256 for character in digest):
        raise ValueError(f"{rubric.criterion_id}: atomic checklist sha256 is invalid")
    actual_digest = _canonical_hash(list(normalized))
    if digest != actual_digest:
        raise ValueError(f"{rubric.criterion_id}: atomic checklist sha256 mismatch")
    return RequirementPlan(
        requirements=tuple(
            AtomicRequirement(f"R{index}", text) for index, text in enumerate(normalized, 1)
        ),
        source="reviewed_atomic_checklist",
        provenance={
            "reviewed_atomic_checklist_used": True,
            "review_status": status,
            "reviewed_by": str(raw_reviewer).strip() if raw_reviewer else None,
            "source": source,
            "sha256": digest,
        },
    )


def _validate_frozen_judge_config(config: EduLLMAdaptiveConfig) -> None:
    if config.judge.provider != "judge":
        raise ValueError("frozen EduLLM mode requires the named provider 'judge'")
    if config.judge.model != QWEN_JUDGE_MODEL:
        raise ValueError(f"judge.model must be frozen to {QWEN_JUDGE_MODEL!r}")
    if config.judge.revision != QWEN_JUDGE_REVISION:
        raise ValueError(f"judge.revision must be frozen to {QWEN_JUDGE_REVISION!r}")
    if config.judge.model_family.casefold() != "qwen":
        raise ValueError("judge.model_family must explicitly be 'qwen'")
    if config.judge.prompt_version != PROMPT_VERSION:
        raise ValueError(f"judge.prompt_version must be {PROMPT_VERSION!r}")
    if config.judge.adapter_version != ADAPTER_VERSION:
        raise ValueError(f"judge.adapter_version must be {ADAPTER_VERSION!r}")
    if config.judge.enable_thinking is not False:
        raise ValueError("frozen Qwen judging requires judge.enable_thinking=false")
    if config.judge.language_model_only is not True:
        raise ValueError("frozen Qwen judging requires judge.language_model_only=true")
    if config.judge.atomic_requirement_policy != ATOMIC_REQUIREMENT_POLICY:
        raise ValueError(f"judge.atomic_requirement_policy must be {ATOMIC_REQUIREMENT_POLICY!r}")
    if not math.isclose(
        config.judge.failure_probability_threshold,
        FAILURE_PROBABILITY_THRESHOLD,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "judge.failure_probability_threshold differs from the frozen Qwen threshold"
        )


def preflight_adaptive_config(raw_config: Mapping[str, Any]) -> _PreparedInputs:
    """Validate the frozen policy, fitted bank, quadrature, and CAT setup without providers."""

    config = _parse_config(raw_config)
    _validate_frozen_judge_config(config)
    bank = load_fitted_bank(
        config.bank.rubrics_path,
        config.bank.scenarios_path,
        manifest_path=config.bank.manifest_path,
        skills=config.bank.skills_order,
    )
    if bank.skills != config.bank.skills_order:
        raise ValueError("loaded bank skills differ from the explicit skill policy")
    calibration_versions = {rubric.calibration_version for rubric in bank.rubrics.values()}
    if calibration_versions != {config.bank.calibration_version}:
        raise ValueError(
            "loaded calibration version differs from bank.calibration_version: "
            f"{sorted(calibration_versions)!r}"
        )
    declared_benchmarks = {
        scenario.benchmark for scenario in bank.scenarios.values() if scenario.benchmark
    }
    if declared_benchmarks and declared_benchmarks != {config.bank.benchmark_id}:
        raise ValueError(
            "scenario benchmark declarations differ from bank.benchmark_id: "
            f"{sorted(declared_benchmarks)!r}"
        )
    if config.cat.max_scenarios > len(bank.scenarios):
        raise ValueError("cat.max_scenarios exceeds the fitted scenario bank")
    quadrature = build_quadrature(
        bank.n_dims,
        config.quadrature.nodes_per_dim,
        bank.latent_correlation,
        max_nodes=config.quadrature.max_nodes,
        method=config.quadrature.method,
        linear_bound=config.quadrature.linear_bound,
    )
    # Constructing a session validates dimension-specific thresholds and initials.
    from olmo_eval.edullm.cat import CatSession

    CatSession(bank, quadrature, config.cat)
    records = _record_map(bank)
    plans = {
        criterion_id: _requirement_plan(rubric, records[criterion_id])
        for criterion_id, rubric in bank.rubrics.items()
    }
    single_precomputed_source: PrecomputedTutorResponses | None = None
    precomputed_batch: PrecomputedTutorResponseBatch | None = None
    if config.tutor.response_source.kind == "precomputed_jsonl":
        source = config.tutor.response_source
        if source.path is None or source.sha256 is None:
            raise ValueError("precomputed_jsonl source is missing its path or SHA-256")
        single_precomputed_source = load_precomputed_tutor_responses(
            source.path,
            expected_sha256=source.sha256,
            expected_scenario_ids=tuple(bank.scenarios),
        )
    elif config.tutor.response_source.kind == "precomputed_batch_jsonl":
        source = config.tutor.response_source
        if source.path is None or source.sha256 is None:
            raise ValueError("precomputed_batch_jsonl source is missing its path or SHA-256")
        precomputed_batch = load_precomputed_tutor_response_batch(
            source.path,
            expected_sha256=source.sha256,
            expected_scenario_ids=tuple(bank.scenarios),
        )
    return _PreparedInputs(
        config=config,
        bank=bank,
        bank_provenance=_bank_provenance_values(config.bank, bank),
        quadrature=quadrature,
        requirement_plans=plans,
        single_precomputed_source=single_precomputed_source,
        precomputed_batch=precomputed_batch,
    )


def _prepare(context: ModeRunContext, raw_config: Mapping[str, Any]) -> _PreparedRun:
    inputs = preflight_adaptive_config(raw_config)
    config = inputs.config
    tutor_identity: TutorIdentity | None = None
    if config.tutor.response_source.kind in {"provider", "precomputed_jsonl"}:
        expected_model = config.tutor.expected_model
        model_family = config.tutor.model_family
        if expected_model is None or model_family is None:
            raise ValueError("single-model tutor configuration is missing its identity")
        configured_candidate_revision = str(config.tutor.model_provenance["revision"])
        tutor_provenance = dict(config.tutor.model_provenance)
        tutor_provenance["revision"] = configured_candidate_revision
        tutor_identity = TutorIdentity(
            model=expected_model,
            model_family=model_family,
            provenance=tutor_provenance,
        )
    if config.tutor.response_source.kind == "provider":
        if tutor_identity is None:
            raise ValueError("provider tutor source is missing its identity")
        if context.provider.model_name != tutor_identity.model:
            raise ValueError(
                "primary tutor provider model does not match tutor.expected_model: "
                f"{context.provider.model_name!r} != {tutor_identity.model!r}"
            )
        runner_metadata = context.metadata.get("mode_runner")
        declared_candidate_revision: object | None = None
        if isinstance(runner_metadata, Mapping):
            declared_candidate_revision = runner_metadata.get("candidate_revision")
        provider_candidate_revision = getattr(context.provider, "_tokenizer_revision", None)
        for source, candidate_revision in (
            ("shared runner", declared_candidate_revision),
            ("candidate provider", provider_candidate_revision),
        ):
            if candidate_revision is None:
                continue
            if candidate_revision != tutor_identity.provenance["revision"]:
                raise ValueError(
                    f"{source} revision does not match tutor.model_provenance.revision: "
                    f"{candidate_revision!r} != {tutor_identity.provenance['revision']!r}"
                )

    judge_provider = context.get_provider("judge")
    if judge_provider.model_name != QWEN_JUDGE_MODEL:
        raise ValueError(
            "named judge provider does not expose the frozen Qwen model: "
            f"{judge_provider.model_name!r} != {QWEN_JUDGE_MODEL!r}"
        )
    if hasattr(judge_provider, "chat_template_kwargs"):
        raw_template_kwargs = judge_provider.chat_template_kwargs
        if not isinstance(raw_template_kwargs, Mapping):
            raise ValueError(
                "judge provider chat_template_kwargs must declare enable_thinking=false"
            )
        template_kwargs = cast(Mapping[Any, Any], raw_template_kwargs)
        if template_kwargs.get("enable_thinking") is not False:
            raise ValueError(
                "judge provider chat_template_kwargs must declare enable_thinking=false"
            )
    provider_language_model_only = getattr(judge_provider, "language_model_only", None)
    if provider_language_model_only is False:
        raise ValueError(
            "judge provider was initialized with language_model_only=false; "
            "frozen Qwen judging requires true"
        )
    for attribute in ("revision", "_tokenizer_revision"):
        if hasattr(judge_provider, attribute):
            provider_revision = getattr(judge_provider, attribute)
            if provider_revision != QWEN_JUDGE_REVISION:
                raise ValueError(
                    f"judge provider {attribute} does not match frozen revision: "
                    f"{provider_revision!r} != {QWEN_JUDGE_REVISION!r}"
                )
    judge = QwenZeroShotBinaryJudge(
        judge_provider,
        failure_probability_threshold=config.judge.failure_probability_threshold,
        pass_token_ids=config.judge.pass_token_ids,
        fail_token_ids=config.judge.fail_token_ids,
    )
    return _PreparedRun(
        config=config,
        bank=inputs.bank,
        bank_provenance=inputs.bank_provenance,
        quadrature=inputs.quadrature,
        judge=judge,
        requirement_plans=inputs.requirement_plans,
        tutor_identity=tutor_identity,
        response_rows=(
            None
            if inputs.single_precomputed_source is None
            else inputs.single_precomputed_source.responses
        ),
        single_precomputed_source=inputs.single_precomputed_source,
        precomputed_batch=inputs.precomputed_batch,
    )


def _atomic_result_row(result: AtomicJudgeResult) -> dict[str, Any]:
    return {
        "requirement_id": result.requirement_id,
        "requirement": result.requirement,
        "verdict": result.verdict,
        "status": result.status,
        "raw_output": result.raw_output,
        "finish_reason": result.finish_reason,
        "native_verdict": result.native_verdict,
        "native_score": result.native_score,
        "rationale": result.rationale,
        "evidence": result.evidence,
        "classification_text": result.classification_text,
        "classification_token_id": result.classification_token_id,
        "classification_raw_output": result.classification_raw_output,
        "classification_finish_reason": result.classification_finish_reason,
        "p_pass": result.p_pass,
        "p_fail": result.p_fail,
        "probability_source": result.probability_source,
        "native_pf_consistent": result.native_pf_consistent,
        "error": result.error,
    }


def _criterion_result_row(result: CriterionJudgeResult) -> dict[str, Any]:
    return {
        "verdict": result.verdict,
        "status": result.status,
        "p_pass": result.p_pass,
        "p_fail": result.p_fail,
        "probability_source": result.probability_source,
        "error": result.error,
        "atomic_results": [
            _atomic_result_row(atomic_result) for atomic_result in result.atomic_results
        ],
    }


def _observation_row(observation: Any) -> dict[str, Any]:
    return {
        "criterion_id": observation.criterion_id,
        "scenario_id": observation.scenario_id,
        "value": observation.value,
        "probability_before": observation.probability_before,
        "theta_after": list(observation.theta_after),
        "online_se_after": list(observation.online_se_after),
    }


def _step_row(step: ScenarioStep) -> dict[str, Any]:
    return {
        "step": step.step,
        "scenario_id": step.scenario_id,
        "selection_mode": step.selection_mode,
        "target_skill": step.target_skill,
        "selection_value": step.selection_value,
        "observations": [_observation_row(item) for item in step.observations],
        "theta_after": list(step.theta_after),
        "online_se_after": list(step.online_se_after),
        "stop_se_after": list(step.stop_se_after),
        "counts_after": dict(step.counts_after),
    }


def _cat_result_row(result: CatResult) -> dict[str, Any]:
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": "succeeded",
        "stop_reason": result.stop_reason,
        "precision_reached": result.precision_reached,
        "stop_se_method": result.stop_se_method,
        "scenarios_administered": list(result.scenarios_administered),
        "criteria_observed": result.criteria_observed,
        "criteria_no_decision": result.criteria_no_decision,
        "counts": dict(result.counts),
        "theta_online": dict(result.theta_online),
        "se_online": dict(result.se_online),
        "theta_eap": dict(result.theta_eap),
        "se_eap": dict(result.se_eap),
        "theta_mwle": None if result.theta_mwle is None else dict(result.theta_mwle),
        "se_mwle": None if result.se_mwle is None else dict(result.se_mwle),
        "mwle_converged": result.mwle_converged,
        "mwle_message": result.mwle_message,
        "critical_failures": list(result.critical_failures),
    }


def _bank_provenance_values(
    config: BankRuntimeConfig,
    bank: FittedBank,
) -> dict[str, Any]:
    irt_sources = sorted(
        {
            str(record.get("irt_params", {}).get("source") or "")
            for record in bank.records
            if isinstance(record.get("irt_params"), Mapping)
        }
        - {""}
    )
    files = {
        "rubrics": {
            "path": str(config.rubrics_path.resolve()),
            "sha256": sha256_file(config.rubrics_path),
        },
        "scenarios": {
            "path": str(config.scenarios_path.resolve()),
            "sha256": sha256_file(config.scenarios_path),
        },
        "manifest": {
            "path": str(config.manifest_path.resolve()),
            "sha256": sha256_file(config.manifest_path),
        },
    }
    return {
        "schema_version": bank.schema_version,
        "benchmark_id": config.benchmark_id,
        "calibration_version": config.calibration_version,
        "policy_id": config.policy_id,
        "scientific_status": config.scientific_status,
        "skills_order": list(bank.skills),
        "latent_correlation": bank.latent_correlation.tolist(),
        "criteria": bank.n_items,
        "scenarios": len(bank.scenarios),
        "irt_sources": irt_sources,
        "files": files,
        "bundle_sha256": _canonical_hash(files),
    }


def _bank_provenance(prepared: _PreparedRun) -> dict[str, Any]:
    # This is the file-hash snapshot captured while the in-memory bank was
    # prepared. Do not re-read mutable source paths while writing later views.
    return cast(dict[str, Any], _json_value(prepared.bank_provenance, path="bank provenance"))


def _tutor_response_source_provenance(prepared: _PreparedRun) -> dict[str, Any]:
    source = prepared.config.tutor.response_source
    if source.kind == "provider":
        return {"kind": "provider"}
    if source.kind == "precomputed_jsonl":
        uploaded = prepared.single_precomputed_source
        if uploaded is None or source.schema_version is None:
            raise ValueError("precomputed tutor response provenance is unavailable")
        return {
            "kind": "precomputed_jsonl",
            "schema_version": source.schema_version,
            "path": str(uploaded.path),
            "declared_sha256": source.sha256,
            "observed_sha256": uploaded.sha256,
            "row_count": uploaded.row_count,
            "scenario_count": uploaded.row_count,
            "blank_count": uploaded.blank_count,
            "exact_bank_coverage": True,
        }
    uploaded_batch = prepared.precomputed_batch
    batch_model = prepared.batch_model
    if uploaded_batch is None or batch_model is None or source.schema_version is None:
        raise ValueError("precomputed tutor response batch provenance is unavailable")
    return {
        "kind": "precomputed_batch_jsonl",
        "schema_version": source.schema_version,
        "path": str(uploaded_batch.path),
        "declared_sha256": source.sha256,
        "observed_sha256": uploaded_batch.sha256,
        "batch_model_count": uploaded_batch.model_count,
        "batch_row_count": uploaded_batch.row_count,
        "batch_blank_count": uploaded_batch.blank_count,
        "selected_model_id": batch_model.model_id,
        "selected_model_row_count": batch_model.row_count,
        "selected_model_blank_count": batch_model.blank_count,
        "exact_bank_coverage": True,
        "batch_provenance": dict(source.provenance or {}),
    }


def _judge_provenance(prepared: _PreparedRun) -> dict[str, Any]:
    config = prepared.config.judge
    judge_identity = {
        "provider": "judge",
        "model": config.model,
        "model_family": config.model_family,
        "revision": config.revision,
        "enable_thinking": config.enable_thinking,
        "language_model_only": config.language_model_only,
    }
    return {
        **judge_identity,
        "identity_sha256": _canonical_hash(judge_identity),
        "failure_probability_threshold": config.failure_probability_threshold,
        "atomic_requirement_policy": config.atomic_requirement_policy,
        "classification_token_ids": {
            "pass": list(prepared.judge.classification_token_ids.pass_ids),
            "fail": list(prepared.judge.classification_token_ids.fail_ids),
            "canonical_pass": prepared.judge.classification_token_ids.canonical_pass_id,
            "canonical_fail": prepared.judge.classification_token_ids.canonical_fail_id,
            "source": prepared.judge.classification_token_ids.source,
        },
    }


def _prompt_provenance() -> dict[str, Any]:
    return {
        "tutor_source": TUTOR_PROMPT_SOURCE,
        "tutor_version": TUTOR_PROMPT_VERSION,
        "judge_prompt_version": PROMPT_VERSION,
        "judge_prompt_profile": PROMPT_PROFILE,
        "judge_prompt_variant": PROMPT_VARIANT,
        "judge_adapter_version": ADAPTER_VERSION,
        "judge_contract_sha256": _canonical_hash(judge_prompt_contract_payload()),
    }


def build_batch_resume_contract_payload(
    raw_config: Mapping[str, Any],
    *,
    run_id: str,
    metadata: Mapping[str, Any],
    harness_config: Mapping[str, Any],
    runtime_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind every scientific and software input needed for safe batch resume.

    This performs only local validation and file hashing.  It intentionally does
    not construct an inference provider or inspect mutable progress artifacts.
    """

    inputs = preflight_adaptive_config(raw_config)
    batch = inputs.precomputed_batch
    if batch is None:
        raise ValueError("resume contracts are supported only for precomputed response batches")
    source = inputs.config.tutor.response_source
    bank = cast(dict[str, Any], _json_value(inputs.bank_provenance, path="bank provenance"))
    prompts = _prompt_provenance()
    scientific_contract = {
        "benchmark_id": inputs.config.bank.benchmark_id,
        "calibration_version": inputs.config.bank.calibration_version,
        "policy_id": inputs.config.bank.policy_id,
        "scientific_status": inputs.config.bank.scientific_status,
        "bank_bundle_sha256": bank["bundle_sha256"],
        "cat": inputs.config.raw["cat"],
        "quadrature": inputs.config.raw["quadrature"],
        "judge": inputs.config.raw["judge"],
        "judge_contract_sha256": prompts["judge_contract_sha256"],
    }
    return cast(
        dict[str, Any],
        _json_value(
            {
                "schema_version": "edullm-adaptive-batch-resume-payload-v1",
                "run_id": run_id,
                "mode": MODE_NAME,
                "implementation_version": IMPLEMENTATION_VERSION,
                "artifact_schema_version": BATCH_ARTIFACT_SCHEMA_VERSION,
                "adaptive_config": inputs.config.raw,
                "metadata": metadata,
                "harness_config": harness_config,
                "runtime_contract": runtime_contract,
                "response_batch": {
                    "schema_version": source.schema_version,
                    "observed_sha256": batch.sha256,
                    "provenance": dict(source.provenance or {}),
                    "model_roster": [
                        {
                            "model_id": model.model_id,
                            "model_family": model.model_family,
                            "model_revision": model.model_revision,
                            "row_count": model.row_count,
                            "blank_count": model.blank_count,
                        }
                        for model in batch.models
                    ],
                },
                "bank": bank,
                "prompts": prompts,
                "scientific_contract_sha256": _canonical_hash(scientific_contract),
            },
            path="batch resume contract",
        ),
    )


def _manifest(
    context: ModeRunContext,
    prepared: _PreparedRun,
    *,
    status: str,
    tutor_rows: Sequence[Mapping[str, Any]],
    judge_rows: Sequence[Mapping[str, Any]],
    selected_scenarios: Sequence[str],
    completed_scenarios: Sequence[str],
    artifacts: Mapping[str, str] | None = None,
    error: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config = prepared.config
    identity = prepared.tutor_identity
    if identity is None:
        raise ValueError("candidate tutor identity is unavailable")
    tutor_identity = {
        "model": identity.model,
        "model_family": identity.model_family,
        "provenance": dict(identity.provenance),
        "response_source": _tutor_response_source_provenance(prepared),
    }
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "mode": MODE_NAME,
        "run_id": context.run_id,
        "status": status,
        "config_sha256": _canonical_hash(config.raw),
        "config": config.raw,
        "context_metadata": dict(context.metadata),
        "tutor": {
            **tutor_identity,
            "identity_sha256": _canonical_hash(tutor_identity),
        },
        "judge": _judge_provenance(prepared),
        "prompts": _prompt_provenance(),
        "bank": _bank_provenance(prepared),
        "progress": {
            "selected_scenarios": list(selected_scenarios),
            "completed_scenarios": list(completed_scenarios),
            "tutor_responses": len(tutor_rows),
            "criterion_judgments": len(judge_rows),
            "no_decisions": sum(row.get("verdict") == "no_decision" for row in judge_rows),
        },
        "artifact_sha256": dict(artifacts or {}),
        "error": None if error is None else dict(error),
    }


def _batch_response_source_provenance(prepared: _PreparedRun) -> dict[str, Any]:
    batch = prepared.precomputed_batch
    source = prepared.config.tutor.response_source
    if batch is None or source.schema_version is None:
        raise ValueError("precomputed tutor response batch provenance is unavailable")
    return {
        "kind": "precomputed_batch_jsonl",
        "schema_version": source.schema_version,
        "path": str(batch.path),
        "declared_sha256": source.sha256,
        "observed_sha256": batch.sha256,
        "model_count": batch.model_count,
        "row_count": batch.row_count,
        "blank_count": batch.blank_count,
        "scenarios_per_model": len(prepared.bank.scenarios),
        "exact_bank_coverage_per_model": True,
        "provenance": dict(source.provenance or {}),
    }


def _batch_counts(
    batch: PrecomputedTutorResponseBatch,
    model_rows: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    succeeded = sum(row.get("status") == ModeStatus.SUCCEEDED.value for row in model_rows)
    failed = sum(row.get("status") == ModeStatus.FAILED.value for row in model_rows)
    cancelled = sum(row.get("status") == ModeStatus.CANCELLED.value for row in model_rows)
    metric_rows = [
        row
        for row in model_rows
        if row.get("status") == ModeStatus.SUCCEEDED.value
        and isinstance(row.get("metrics"), Mapping)
    ]
    return {
        "models_total": batch.model_count,
        "models_completed": len(model_rows),
        "models_succeeded": succeeded,
        "models_failed": failed,
        "models_cancelled": cancelled,
        "models_pending": batch.model_count - len(model_rows),
        "models_with_metrics": len(metric_rows),
        "models_precision_reached": sum(
            cast(Mapping[str, Any], row["metrics"]).get("precision_reached") is True
            for row in metric_rows
        ),
        "models_mwle_converged": sum(
            cast(Mapping[str, Any], row["metrics"]).get("mwle_converged") is True
            for row in metric_rows
        ),
        "scenario_administrations_with_metrics": sum(
            int(cast(Mapping[str, Any], row.get("metrics", {})).get("scenarios_administered", 0))
            for row in metric_rows
        ),
        "criterion_observations_with_metrics": sum(
            int(cast(Mapping[str, Any], row.get("metrics", {})).get("criteria_observed", 0))
            for row in metric_rows
        ),
        "criterion_no_decisions_with_metrics": sum(
            int(cast(Mapping[str, Any], row.get("metrics", {})).get("criteria_no_decision", 0))
            for row in metric_rows
        ),
        "partial_completed_units_failed_models": sum(
            int(row.get("completed_units") or 0)
            for row in model_rows
            if row.get("status") == ModeStatus.FAILED.value
        ),
    }


def _batch_manifest(
    context: ModeRunContext,
    prepared: _PreparedRun,
    *,
    status: str,
    model_rows: Sequence[Mapping[str, Any]],
    checkpoint_generation: int,
    artifacts: Mapping[str, str] | None = None,
    error: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    batch = prepared.precomputed_batch
    if batch is None:
        raise ValueError("batch manifest requested for a non-batch run")
    return {
        "schema_version": BATCH_ARTIFACT_SCHEMA_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "mode": MODE_NAME,
        "execution": "precomputed_batch",
        "run_id": context.run_id,
        "status": status,
        "resume_contract_fingerprint": (
            context.resume_contract_fingerprint or _batch_fingerprint(context, prepared)
        ),
        "checkpoint_generation": checkpoint_generation,
        "config_sha256": _canonical_hash(prepared.config.raw),
        "config": prepared.config.raw,
        "context_metadata": dict(context.metadata),
        "response_batch": _batch_response_source_provenance(prepared),
        "judge": _judge_provenance(prepared),
        "prompts": _prompt_provenance(),
        "bank": _bank_provenance(prepared),
        "cat_policy": _json_value(prepared.config.raw["cat"], path="cat_policy"),
        "progress": _batch_counts(batch, model_rows),
        "model_result_paths": [str(row["output_dir"]) for row in model_rows],
        "artifact_sha256": dict(artifacts or {}),
        "error": None if error is None else dict(error),
    }


def _batch_fingerprint(context: ModeRunContext, prepared: _PreparedRun) -> str:
    """Return the orchestrator fingerprint or a deterministic direct-run fallback."""

    if context.resume_contract_fingerprint is not None:
        return context.resume_contract_fingerprint
    batch = prepared.precomputed_batch
    if batch is None:
        raise ValueError("batch fingerprint requested for a non-batch run")
    return _canonical_hash(
        {
            "schema_version": "edullm-direct-batch-fingerprint-v1",
            "run_id": context.run_id,
            "implementation_version": IMPLEMENTATION_VERSION,
            "config": prepared.config.raw,
            "context_metadata": dict(context.metadata),
            "response_batch_sha256": batch.sha256,
            "bank": _bank_provenance(prepared),
            "prompts": _prompt_provenance(),
        }
    )


def _candidate_id(model_index: int, batch_model: PrecomputedTutorResponseBatchModel) -> str:
    return f"candidate-{model_index:04d}-{_text_hash(batch_model.model_id)[:12]}"


def _candidate_prepared_run(
    prepared: _PreparedRun,
    batch_model: PrecomputedTutorResponseBatchModel,
    source_provenance: Mapping[str, Any],
) -> _PreparedRun:
    batch = prepared.precomputed_batch
    if batch is None:
        raise ValueError("candidate preparation requested for a non-batch run")
    identity = TutorIdentity(
        model=batch_model.model_id,
        model_family=batch_model.model_family,
        provenance={
            "source": source_provenance["source"],
            "revision": batch_model.model_revision,
            "batch_revision": source_provenance["revision"],
            "batch_sha256": batch.sha256,
        },
    )
    return replace(
        prepared,
        tutor_identity=identity,
        response_rows=batch_model.responses,
        batch_model=batch_model,
    )


def _validate_attempt_row(
    row: Mapping[str, Any],
    *,
    run_id: str,
    config_raw: Mapping[str, Any],
    mode_root: Path,
    model_index: int,
    batch_model: PrecomputedTutorResponseBatchModel,
) -> dict[str, Any]:
    """Validate one committed candidate attempt and every artifact it names."""

    candidate_id = _candidate_id(model_index, batch_model)
    prefix = f"committed attempt for {candidate_id}"
    _exact_keys(
        row,
        {
            "schema_version",
            "model_index",
            "candidate_id",
            "attempt_number",
            "model_id",
            "model_family",
            "model_revision",
            "output_dir",
            "status",
            "metrics",
            "warnings",
            "completed_units",
            "error",
            "committed_at",
            "manifest_sha256",
        },
        prefix,
    )
    if row.get("schema_version") != BATCH_ARTIFACT_SCHEMA_VERSION:
        raise ValueError(f"{prefix} has an unsupported schema_version")
    if row.get("model_index") != model_index:
        raise ValueError(f"{prefix} has the wrong model_index")
    if row.get("candidate_id") != candidate_id:
        raise ValueError(f"{prefix} has the wrong candidate_id")
    for field, expected in (
        ("model_id", batch_model.model_id),
        ("model_family", batch_model.model_family),
        ("model_revision", batch_model.model_revision),
    ):
        if row.get(field) != expected:
            raise ValueError(f"{prefix} has the wrong {field}")

    attempt_number = row.get("attempt_number")
    if (
        isinstance(attempt_number, bool)
        or not isinstance(attempt_number, int)
        or attempt_number < 1
    ):
        raise ValueError(f"{prefix}.attempt_number must be a positive integer")
    expected_relative = Path("models") / candidate_id / f"attempt-{attempt_number:04d}"
    raw_output_dir = row.get("output_dir")
    if not isinstance(raw_output_dir, str):
        raise ValueError(f"{prefix}.output_dir must be a string")
    relative_output = safe_relative_path(raw_output_dir, prefix="models")
    if relative_output != expected_relative:
        raise ValueError(f"{prefix}.output_dir does not match its attempt number")
    candidate_output = mode_root / relative_output
    if candidate_output.is_symlink() or not candidate_output.is_dir():
        raise ValueError(f"{prefix} output directory is missing or unsafe")
    resolved_root = mode_root.resolve()
    if resolved_root not in candidate_output.resolve().parents:
        raise ValueError(f"{prefix} output directory escapes the mode root")

    status = row.get("status")
    if status not in {
        ModeStatus.SUCCEEDED.value,
        ModeStatus.FAILED.value,
        ModeStatus.CANCELLED.value,
    }:
        raise ValueError(f"{prefix}.status is not terminal")
    metrics = row.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError(f"{prefix}.metrics must be an object")
    warnings = row.get("warnings")
    if not isinstance(warnings, list) or not all(isinstance(item, str) for item in warnings):
        raise ValueError(f"{prefix}.warnings must be a string array")
    completed_units = row.get("completed_units")
    if (
        isinstance(completed_units, bool)
        or not isinstance(completed_units, int)
        or completed_units < 0
    ):
        raise ValueError(f"{prefix}.completed_units must be a non-negative integer")
    error = row.get("error")
    if status == ModeStatus.SUCCEEDED.value and error is not None:
        raise ValueError(f"{prefix} succeeded but carries an error")
    if status != ModeStatus.SUCCEEDED.value and not isinstance(error, str):
        raise ValueError(f"{prefix} failed or was cancelled without an error string")
    committed_at = row.get("committed_at")
    if not isinstance(committed_at, str):
        raise ValueError(f"{prefix}.committed_at must be an ISO-8601 string")
    try:
        committed_datetime = datetime.fromisoformat(committed_at)
    except ValueError as exc:
        raise ValueError(f"{prefix}.committed_at must be an ISO-8601 string") from exc
    if committed_datetime.tzinfo is None:
        raise ValueError(f"{prefix}.committed_at must include a timezone")

    manifest_digest = row.get("manifest_sha256")
    if (
        not isinstance(manifest_digest, str)
        or len(manifest_digest) != 64
        or any(character not in _HEX_SHA256 for character in manifest_digest)
    ):
        raise ValueError(f"{prefix}.manifest_sha256 must be a lowercase SHA-256 digest")
    manifest_path = candidate_output / "manifest.json"
    if not manifest_path.is_file() or sha256_file(manifest_path) != manifest_digest:
        raise ValueError(f"{prefix} candidate manifest hash does not match")
    manifest = read_strict_json_object(manifest_path)
    if manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(f"{prefix} candidate manifest has an unsupported schema_version")
    if manifest.get("implementation_version") != IMPLEMENTATION_VERSION:
        raise ValueError(f"{prefix} candidate manifest implementation changed")
    if manifest.get("mode") != MODE_NAME or manifest.get("run_id") != run_id:
        raise ValueError(f"{prefix} candidate manifest belongs to another run")
    if manifest.get("status") != status:
        raise ValueError(f"{prefix} status disagrees with its candidate manifest")
    if manifest.get("config_sha256") != _canonical_hash(config_raw):
        raise ValueError(f"{prefix} candidate manifest config hash changed")
    tutor = manifest.get("tutor")
    if not isinstance(tutor, Mapping):
        raise ValueError(f"{prefix} candidate manifest tutor must be an object")
    if tutor.get("model") != batch_model.model_id or tutor.get("model_family") != (
        batch_model.model_family
    ):
        raise ValueError(f"{prefix} candidate manifest tutor identity changed")
    tutor_provenance = tutor.get("provenance")
    if not isinstance(tutor_provenance, Mapping) or tutor_provenance.get("revision") != (
        batch_model.model_revision
    ):
        raise ValueError(f"{prefix} candidate manifest tutor revision changed")
    artifact_hashes = manifest.get("artifact_sha256")
    if not isinstance(artifact_hashes, Mapping):
        raise ValueError(f"{prefix} candidate manifest artifact hashes must be an object")
    required_artifacts = {
        "tutor_responses.jsonl",
        "judge_rows.jsonl",
        "cat_result.json",
        "cat_trace.jsonl",
    }
    if status != ModeStatus.SUCCEEDED.value:
        required_artifacts.add("failure.json")
    missing_artifacts = sorted(required_artifacts - set(artifact_hashes))
    if missing_artifacts:
        raise ValueError(f"{prefix} is missing artifact hash(es): {missing_artifacts}")
    verify_hashed_artifacts(candidate_output, artifact_hashes)
    if status == ModeStatus.SUCCEEDED.value:
        cat_result = read_strict_json_object(candidate_output / "cat_result.json")
        if cat_result.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
            raise ValueError(f"{prefix} CAT result has an unsupported schema_version")
        scenarios = cat_result.get("scenarios_administered")
        if not isinstance(scenarios, list):
            raise ValueError(f"{prefix} CAT scenarios_administered must be an array")
        expected_metrics = {
            "stop_reason": cat_result.get("stop_reason"),
            "precision_reached": cat_result.get("precision_reached"),
            "stop_se_method": cat_result.get("stop_se_method"),
            "scenarios_administered": len(scenarios),
            "criteria_observed": cat_result.get("criteria_observed"),
            "criteria_no_decision": cat_result.get("criteria_no_decision"),
            "counts": cat_result.get("counts"),
            "mwle_converged": cat_result.get("mwle_converged"),
            "mwle_message": cat_result.get("mwle_message"),
            "critical_failures": cat_result.get("critical_failures"),
            "theta_eap": cat_result.get("theta_eap"),
            "se_eap": cat_result.get("se_eap"),
            "theta_mwle": cat_result.get("theta_mwle"),
            "se_mwle": cat_result.get("se_mwle"),
        }
        if dict(metrics) != expected_metrics:
            raise ValueError(f"{prefix} metrics disagree with the hashed CAT result")
    return cast(dict[str, Any], _json_value(row, path=prefix))


def _validate_restored_checkpoint(
    state: BatchCheckpointState,
    *,
    run_id: str,
    config_raw: Mapping[str, Any],
    batch: PrecomputedTutorResponseBatch,
    mode_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Bind a validated checkpoint chain to the exact uploaded model roster."""

    validated_attempts: list[dict[str, Any]] = []
    attempts_by_candidate: dict[str, list[dict[str, Any]]] = {}
    seen_outputs: set[str] = set()
    last_attempt_number: dict[str, int] = {}
    roster = {
        _candidate_id(index, model): (index, model) for index, model in enumerate(batch.models)
    }
    for position, raw_attempt in enumerate(state.attempts):
        candidate_id = raw_attempt.get("candidate_id")
        if not isinstance(candidate_id, str) or candidate_id not in roster:
            raise ValueError(f"checkpoint attempt {position} references an unknown candidate")
        model_index, batch_model = roster[candidate_id]
        attempt = _validate_attempt_row(
            raw_attempt,
            run_id=run_id,
            config_raw=config_raw,
            mode_root=mode_root,
            model_index=model_index,
            batch_model=batch_model,
        )
        output_dir = cast(str, attempt["output_dir"])
        if output_dir in seen_outputs:
            raise ValueError("checkpoint attempts contain a duplicate output_dir")
        seen_outputs.add(output_dir)
        attempt_number = cast(int, attempt["attempt_number"])
        if attempt_number <= last_attempt_number.get(candidate_id, 0):
            raise ValueError("checkpoint attempt numbers must increase for each candidate")
        last_attempt_number[candidate_id] = attempt_number
        attempts_by_candidate.setdefault(candidate_id, []).append(attempt)
        validated_attempts.append(attempt)

    validated_rows: list[dict[str, Any]] = []
    seen_candidates: set[str] = set()
    for position, raw_row in enumerate(state.model_results):
        candidate_id = raw_row.get("candidate_id")
        if not isinstance(candidate_id, str) or candidate_id not in roster:
            raise ValueError(f"checkpoint model result {position} references an unknown candidate")
        if candidate_id in seen_candidates:
            raise ValueError("checkpoint model results contain a duplicate candidate")
        seen_candidates.add(candidate_id)
        model_index, batch_model = roster[candidate_id]
        row = _validate_attempt_row(
            raw_row,
            run_id=run_id,
            config_raw=config_raw,
            mode_root=mode_root,
            model_index=model_index,
            batch_model=batch_model,
        )
        candidate_attempts = attempts_by_candidate.get(candidate_id, [])
        if not candidate_attempts or row != candidate_attempts[-1]:
            raise ValueError("checkpoint model result is not the candidate's latest attempt")
        validated_rows.append(row)

    if not seen_candidates.issubset(attempts_by_candidate):
        raise ValueError("checkpoint model results are missing their committed attempt history")
    validated_rows.sort(key=lambda row: cast(int, row["model_index"]))
    return validated_rows, validated_attempts


def validate_batch_resume_artifacts(
    raw_config: Mapping[str, Any],
    *,
    run_id: str,
    mode_root: Path,
    fingerprint_sha256: str,
) -> BatchCheckpointState:
    """Validate a resume checkpoint and candidate artifacts without a provider."""

    inputs = preflight_adaptive_config(raw_config)
    batch = inputs.precomputed_batch
    if batch is None:
        raise ValueError("batch resume validation requires precomputed_batch_jsonl input")
    state = load_checkpoint_chain(mode_root, fingerprint_sha256=fingerprint_sha256)
    _validate_restored_checkpoint(
        state,
        run_id=run_id,
        config_raw=inputs.config.raw,
        batch=batch,
        mode_root=mode_root,
    )
    return state


def _attempt_result_row(
    result: ModeResult,
    *,
    context: ModeRunContext,
    prepared: _PreparedRun,
    mode_root: Path,
    model_index: int,
    batch_model: PrecomputedTutorResponseBatchModel,
    attempt_number: int,
    relative_output: Path,
) -> dict[str, Any]:
    manifest_path = mode_root / relative_output / "manifest.json"
    row = {
        "schema_version": BATCH_ARTIFACT_SCHEMA_VERSION,
        "model_index": model_index,
        "candidate_id": _candidate_id(model_index, batch_model),
        "attempt_number": attempt_number,
        "model_id": batch_model.model_id,
        "model_family": batch_model.model_family,
        "model_revision": batch_model.model_revision,
        "output_dir": relative_output.as_posix(),
        "status": result.status.value,
        "metrics": dict(result.metrics),
        "warnings": list(result.warnings),
        "completed_units": int(result.completed_units or 0),
        "error": result.error,
        "committed_at": datetime.now(UTC).isoformat(),
        "manifest_sha256": sha256_file(manifest_path),
    }
    return _validate_attempt_row(
        row,
        run_id=context.run_id,
        config_raw=prepared.config.raw,
        mode_root=mode_root,
        model_index=model_index,
        batch_model=batch_model,
    )


class EduLLMAdaptiveMode:
    """EvaluationMode implementation for adaptive EduLLM criterion grading."""

    name = MODE_NAME
    implementation_version = IMPLEMENTATION_VERSION
    required_auxiliary_providers = ("judge",)

    def __init__(self) -> None:
        # The dispatcher calls ``preflight`` immediately before its final
        # locked input validation. Retaining that exact prepared snapshot means
        # execution cannot silently reload different bank/response bytes after
        # the validation boundary.
        self._prepared_context: ModeRunContext | None = None
        self._prepared_config_sha256: str | None = None
        self._prepared_run: _PreparedRun | None = None

    def preflight_config(self, config: Mapping[str, Any]) -> None:
        """Validate bank and scientific inputs before providers are started."""

        preflight_adaptive_config(config)

    def preflight(self, context: ModeRunContext, config: Mapping[str, Any]) -> None:
        """Fail closed before generation for config, provider, bank, or CAT errors."""

        prepared = _prepare(context, config)
        self._prepared_context = context
        self._prepared_config_sha256 = _canonical_hash(prepared.config.raw)
        self._prepared_run = prepared

    async def run(
        self,
        context: ModeRunContext,
        config: Mapping[str, Any],
        output_dir: Path,
    ) -> ModeResult:
        requested_config_sha256 = _canonical_hash(_parse_config(config).raw)
        if (
            self._prepared_context is context
            and self._prepared_config_sha256 == requested_config_sha256
            and self._prepared_run is not None
        ):
            prepared = self._prepared_run
        else:
            # Direct library callers are allowed to invoke ``run`` without the
            # OLMo dispatcher's preflight phase.
            prepared = _prepare(context, config)
        self._prepared_context = None
        self._prepared_config_sha256 = None
        self._prepared_run = None
        if prepared.precomputed_batch is not None:
            return await self._run_batch(context, prepared, output_dir)
        if prepared.tutor_identity is None:
            raise RuntimeError("single-model run is missing its tutor identity")
        return await self._run_candidate(context, prepared, output_dir)

    async def _run_candidate(
        self,
        context: ModeRunContext,
        prepared: _PreparedRun,
        output_dir: Path,
        progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> ModeResult:
        paths = {
            "manifest.json": output_dir / "manifest.json",
            "tutor_responses.jsonl": output_dir / "tutor_responses.jsonl",
            "judge_rows.jsonl": output_dir / "judge_rows.jsonl",
            "cat_result.json": output_dir / "cat_result.json",
            "cat_trace.jsonl": output_dir / "cat_trace.jsonl",
        }
        tutor_rows: list[dict[str, Any]] = []
        judge_rows: list[dict[str, Any]] = []
        partial_trace: list[dict[str, Any]] = []
        selected_scenarios: list[str] = []
        completed_scenarios: list[str] = []
        output_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_jsonl(paths["tutor_responses.jsonl"], tutor_rows)
        _atomic_write_jsonl(paths["judge_rows.jsonl"], judge_rows)
        _atomic_write_jsonl(paths["cat_trace.jsonl"], partial_trace)
        _atomic_write_json(
            paths["cat_result.json"],
            {"schema_version": ARTIFACT_SCHEMA_VERSION, "status": "running"},
        )
        _atomic_write_json(
            paths["manifest.json"],
            _manifest(
                context,
                prepared,
                status="running",
                tutor_rows=tutor_rows,
                judge_rows=judge_rows,
                selected_scenarios=selected_scenarios,
                completed_scenarios=completed_scenarios,
            ),
        )

        records = _record_map(prepared.bank)

        async def responder(
            scenario: Scenario,
            rubrics: tuple[Rubric, ...],
        ) -> Mapping[str, int | None]:
            selected_scenarios.append(scenario.scenario_id)
            if progress_callback is not None:
                progress_callback(
                    {
                        "phase": "judging_scenario",
                        "scenario_id": scenario.scenario_id,
                        "selected_scenarios": len(selected_scenarios),
                        "completed_scenarios": len(completed_scenarios),
                        "criterion_judgments": len(judge_rows),
                    }
                )
            identity = prepared.tutor_identity
            if identity is None:
                raise RuntimeError("candidate tutor identity is unavailable")
            source_metadata: Mapping[str, Any] = {}
            source_row_sha256: str | None = None
            if prepared.config.tutor.response_source.kind == "provider":
                generation = prepared.config.tutor.generation
                if generation is None:
                    raise RuntimeError("provider tutor source is missing generation settings")
                generated = await generate_tutor_response(
                    context.provider,
                    scenario,
                    generation,
                )
                messages = [dict(message) for message in generated.request.messages]
                response_text = generated.output.text
                request_sha256: str | None = _canonical_hash(messages)
                request_provenance = "provider_generation_request"
                messages_provenance = "provider_generation_request"
            else:
                response_rows = prepared.response_rows
                if response_rows is None:
                    raise RuntimeError("precomputed tutor responses were not loaded")
                source_row = response_rows[scenario.scenario_id]
                messages = [dict(message) for message in build_tutor_messages(scenario)]
                response_text = source_row.response
                source_metadata = source_row.metadata
                source_row_sha256 = source_row.source_row_sha256
                rendered_prompt = source_metadata.get("rendered_prompt")
                declared_prompt_sha256 = source_metadata.get("rendered_prompt_sha256")
                if rendered_prompt is None and declared_prompt_sha256 is None:
                    request_sha256 = None
                    request_provenance = "unavailable_for_precomputed_response"
                elif isinstance(rendered_prompt, str) and isinstance(declared_prompt_sha256, str):
                    observed_prompt_sha256 = _text_hash(rendered_prompt)
                    if declared_prompt_sha256 != observed_prompt_sha256:
                        raise ValueError(
                            "precomputed rendered_prompt_sha256 does not match rendered_prompt"
                        )
                    request_sha256 = observed_prompt_sha256
                    request_provenance = "uploaded_rendered_prompt"
                else:
                    raise ValueError(
                        "precomputed rendered prompt provenance must provide both "
                        "rendered_prompt and rendered_prompt_sha256 strings"
                    )
                messages_provenance = "reconstructed_evaluation_context_not_generation_request"
            tutor_row = {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "scenario_id": scenario.scenario_id,
                "tutor_model": identity.model,
                "response_source": prepared.config.tutor.response_source.kind,
                "source_metadata": dict(source_metadata),
                "source_row_sha256": source_row_sha256,
                "messages": messages,
                "messages_provenance": messages_provenance,
                "evaluation_context_sha256": _canonical_hash(messages),
                "request_sha256": request_sha256,
                "request_provenance": request_provenance,
                "raw_output": response_text,
                "raw_output_sha256": _text_hash(response_text),
                "blank_response": not bool(response_text.strip()),
            }
            tutor_rows.append(tutor_row)
            _atomic_write_jsonl(paths["tutor_responses.jsonl"], tutor_rows)

            observations: dict[str, int | None] = {}
            for rubric in rubrics:
                plan = prepared.requirement_plans[rubric.criterion_id]
                record = records[rubric.criterion_id]
                raw_evidence = record.get("expected_evidence") or ()
                if isinstance(raw_evidence, Sequence) and not isinstance(
                    raw_evidence, (str, bytes)
                ):
                    expected_evidence = tuple(str(item) for item in raw_evidence)
                elif raw_evidence:
                    raise ValueError(f"{rubric.criterion_id}: expected_evidence must be an array")
                else:
                    expected_evidence = ()
                case = BlindedJudgeCase(
                    scenario_prompt=scenario.prompt,
                    candidate_response=response_text,
                    conversation_context=tuple(scenario.conversation_context),
                    reference_solution=scenario.reference_solution,
                    expected_evidence=expected_evidence,
                )

                if not response_text.strip():
                    result = CriterionJudgeResult(
                        verdict="no_decision",
                        status="blank_tutor_response",
                        atomic_results=(),
                        error="blank tutor response is missing data under the frozen policy",
                    )
                else:
                    result = await prepared.judge.judge_criterion(case, plan.requirements)
                if result.verdict == "pass":
                    observation: int | None = 1
                elif result.verdict == "fail":
                    observation = 0
                else:
                    observation = None
                observations[rubric.criterion_id] = observation

                atomic_messages = [
                    build_atomic_messages(case, requirement) for requirement in plan.requirements
                ]
                atomic_prompt_hashes = [_canonical_hash(messages) for messages in atomic_messages]
                classification_prompt_hashes = []
                if result.atomic_results:
                    for atomic_result, messages in zip(
                        result.atomic_results, atomic_messages, strict=True
                    ):
                        classification_prompt_hashes.append(
                            _canonical_hash(
                                build_classification_messages(
                                    messages,
                                    atomic_result.raw_output,
                                )
                            )
                        )
                row = {
                    "schema_version": ARTIFACT_SCHEMA_VERSION,
                    "scenario_id": scenario.scenario_id,
                    "criterion_id": rubric.criterion_id,
                    "criterion": rubric.criterion,
                    "criterion_sha256": _text_hash(rubric.criterion),
                    "candidate_response_sha256": _text_hash(response_text),
                    "judge_model": prepared.config.judge.model,
                    "judge_revision": prepared.config.judge.revision,
                    "adapter_version": prepared.config.judge.adapter_version,
                    "prompt_version": prepared.config.judge.prompt_version,
                    "requirement_source": plan.source,
                    "requirement_provenance": dict(plan.provenance),
                    "requirements": [
                        {
                            "requirement_id": requirement.requirement_id,
                            "text": requirement.text,
                        }
                        for requirement in plan.requirements
                    ],
                    "atomic_prompt_sha256": atomic_prompt_hashes,
                    "classification_prompt_sha256": classification_prompt_hashes,
                    **_criterion_result_row(result),
                    "observation": observation,
                }
                judge_rows.append(row)
                _atomic_write_jsonl(paths["judge_rows.jsonl"], judge_rows)

            partial_trace.append(
                {
                    "schema_version": ARTIFACT_SCHEMA_VERSION,
                    "status": "awaiting_cat_record",
                    "scenario_id": scenario.scenario_id,
                    "observations": observations,
                }
            )
            _atomic_write_jsonl(paths["cat_trace.jsonl"], partial_trace)
            completed_scenarios.append(scenario.scenario_id)
            if progress_callback is not None:
                progress_callback(
                    {
                        "phase": "scenario_completed",
                        "scenario_id": scenario.scenario_id,
                        "selected_scenarios": len(selected_scenarios),
                        "completed_scenarios": len(completed_scenarios),
                        "criterion_judgments": len(judge_rows),
                    }
                )
            return observations

        try:
            result = await run_cat(
                prepared.bank,
                prepared.quadrature,
                prepared.config.cat,
                responder,
            )
            trace = [_step_row(step) for step in result.steps]
            _atomic_write_jsonl(paths["cat_trace.jsonl"], trace)
            _atomic_write_json(paths["cat_result.json"], _cat_result_row(result))
            artifact_hashes = {
                name: sha256_file(path) for name, path in paths.items() if name != "manifest.json"
            }
            _atomic_write_json(
                paths["manifest.json"],
                _manifest(
                    context,
                    prepared,
                    status="succeeded",
                    tutor_rows=tutor_rows,
                    judge_rows=judge_rows,
                    selected_scenarios=selected_scenarios,
                    completed_scenarios=result.scenarios_administered,
                    artifacts=artifact_hashes,
                ),
            )
            warnings: list[str] = []
            if prepared.config.bank.scientific_status == "experimental":
                warnings.append("fitted bank and CAT policy are explicitly experimental")
            if result.criteria_no_decision:
                warnings.append(
                    f"{result.criteria_no_decision} criterion judgment(s) were no-decision"
                )
            return ModeResult(
                mode=self.name,
                implementation_version=self.implementation_version,
                status=ModeStatus.SUCCEEDED,
                metrics={
                    "stop_reason": result.stop_reason,
                    "precision_reached": result.precision_reached,
                    "stop_se_method": result.stop_se_method,
                    "scenarios_administered": len(result.scenarios_administered),
                    "criteria_observed": result.criteria_observed,
                    "criteria_no_decision": result.criteria_no_decision,
                    "counts": dict(result.counts),
                    "mwle_converged": result.mwle_converged,
                    "mwle_message": result.mwle_message,
                    "critical_failures": list(result.critical_failures),
                    "theta_eap": dict(result.theta_eap),
                    "se_eap": dict(result.se_eap),
                    "theta_mwle": (None if result.theta_mwle is None else dict(result.theta_mwle)),
                    "se_mwle": (None if result.se_mwle is None else dict(result.se_mwle)),
                },
                artifacts=tuple(paths),
                warnings=tuple(warnings),
                completed_units=len(result.scenarios_administered),
            )
        except asyncio.CancelledError:
            self._write_failure(
                context,
                prepared,
                paths,
                tutor_rows,
                judge_rows,
                selected_scenarios,
                completed_scenarios,
                status="cancelled",
                error=asyncio.CancelledError("evaluation mode was cancelled"),
            )
            raise
        except Exception as exc:
            failure_path = self._write_failure(
                context,
                prepared,
                paths,
                tutor_rows,
                judge_rows,
                selected_scenarios,
                completed_scenarios,
                status="failed",
                error=exc,
            )
            return ModeResult(
                mode=self.name,
                implementation_version=self.implementation_version,
                status=ModeStatus.FAILED,
                artifacts=(*tuple(paths), failure_path.name),
                completed_units=len(completed_scenarios),
                error=f"{type(exc).__name__}: {exc}",
            )

    async def _run_batch(
        self,
        context: ModeRunContext,
        prepared: _PreparedRun,
        output_dir: Path,
    ) -> ModeResult:
        batch = prepared.precomputed_batch
        source_provenance = prepared.config.tutor.response_source.provenance
        if batch is None or source_provenance is None:
            raise RuntimeError("precomputed tutor response batch is unavailable")
        if not context.resume and output_dir.exists() and any(output_dir.iterdir()):
            raise ValueError("fresh batch run refuses to overwrite a non-empty mode directory")
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "manifest.json": output_dir / "manifest.json",
            "model_results.jsonl": output_dir / "model_results.jsonl",
            "attempt_history.jsonl": output_dir / "attempt_history.jsonl",
            "batch_summary.json": output_dir / "batch_summary.json",
            "progress.json": output_dir / "progress.json",
            BATCH_RESULTS_JSON: output_dir / BATCH_RESULTS_JSON,
            BATCH_RESULTS_CSV: output_dir / BATCH_RESULTS_CSV,
            BATCH_REPORT_MARKDOWN: output_dir / BATCH_REPORT_MARKDOWN,
        }
        fingerprint = _batch_fingerprint(context, prepared)
        state = load_checkpoint_chain(output_dir, fingerprint_sha256=fingerprint)
        if not context.resume and state.generation:
            raise ValueError("fresh batch run found an existing checkpoint chain")
        if state.pointer_stale:
            if not context.resume:
                raise ValueError("fresh batch run found a stale checkpoint pointer")
            repair_checkpoint_pointer(output_dir, state)
        restored_rows, attempts = _validate_restored_checkpoint(
            state,
            run_id=context.run_id,
            config_raw=prepared.config.raw,
            batch=batch,
            mode_root=output_dir,
        )
        model_rows = list(restored_rows)
        session_started_at = datetime.now(UTC)
        session_started_monotonic = time.monotonic()
        attempts_started_this_session = 0
        attempts_completed_this_session = 0

        def ordered_rows() -> list[dict[str, Any]]:
            return sorted(model_rows, key=lambda row: cast(int, row["model_index"]))

        def write_progress(
            status: str,
            *,
            phase: str,
            current_model: Mapping[str, Any] | None = None,
            error: Mapping[str, Any] | None = None,
        ) -> None:
            elapsed = max(0.0, time.monotonic() - session_started_monotonic)
            counts = _batch_counts(batch, ordered_rows())
            models_per_hour = (
                attempts_completed_this_session / elapsed * 3600.0
                if attempts_completed_this_session and elapsed > 0
                else None
            )
            estimated_remaining_seconds = (
                counts["models_pending"] / models_per_hour * 3600.0
                if models_per_hour and models_per_hour > 0
                else None
            )
            _atomic_write_json(
                paths["progress.json"],
                {
                    "schema_version": BATCH_PROGRESS_SCHEMA_VERSION,
                    "run_id": context.run_id,
                    "status": status,
                    "phase": phase,
                    "resumed": context.resume,
                    "resume_contract_fingerprint": fingerprint,
                    "checkpoint_generation": state.generation,
                    "session_started_at": session_started_at.isoformat(),
                    "updated_at": datetime.now(UTC).isoformat(),
                    "elapsed_seconds": elapsed,
                    "attempts_started_this_session": attempts_started_this_session,
                    "attempts_completed_this_session": attempts_completed_this_session,
                    "models_per_hour_this_session": models_per_hour,
                    "estimated_remaining_seconds": estimated_remaining_seconds,
                    **counts,
                    "current_model": None if current_model is None else dict(current_model),
                    "error": None if error is None else dict(error),
                },
            )

        def write_views(
            status: str,
            *,
            error: Mapping[str, Any] | None = None,
        ) -> None:
            rows = ordered_rows()
            counts = _batch_counts(batch, rows)
            _atomic_write_jsonl(paths["model_results.jsonl"], rows)
            _atomic_write_jsonl(paths["attempt_history.jsonl"], attempts)
            summary = {
                "schema_version": BATCH_ARTIFACT_SCHEMA_VERSION,
                "run_id": context.run_id,
                "status": status,
                "checkpoint_generation": state.generation,
                "resume_contract_fingerprint": fingerprint,
                **counts,
                "response_batch_sha256": batch.sha256,
                "model_results_path": "model_results.jsonl",
                "attempt_history_path": "attempt_history.jsonl",
                "attempts_committed": len(attempts),
                "error": None if error is None else dict(error),
            }
            _atomic_write_json(paths["batch_summary.json"], summary)
            artifact_hashes = {
                name: sha256_file(paths[name])
                for name in (
                    "model_results.jsonl",
                    "attempt_history.jsonl",
                    "batch_summary.json",
                )
            }
            checkpoint_pointer = output_dir / "checkpoints/latest.json"
            if checkpoint_pointer.is_file():
                artifact_hashes["checkpoints/latest.json"] = sha256_file(checkpoint_pointer)
            manifest = _batch_manifest(
                context,
                prepared,
                status=status,
                model_rows=rows,
                checkpoint_generation=state.generation,
                artifacts=artifact_hashes,
                error=error,
            )
            _atomic_write_json(paths["manifest.json"], manifest)
            report = build_batch_report(manifest, summary, rows)
            write_batch_reports(output_dir, report)
            artifact_hashes.update(
                {
                    name: sha256_file(paths[name])
                    for name in (BATCH_RESULTS_JSON, BATCH_RESULTS_CSV, BATCH_REPORT_MARKDOWN)
                }
            )
            _atomic_write_json(
                paths["manifest.json"],
                _batch_manifest(
                    context,
                    prepared,
                    status=status,
                    model_rows=rows,
                    checkpoint_generation=state.generation,
                    artifacts=artifact_hashes,
                    error=error,
                ),
            )

        def commit_and_publish(
            status: str,
            *,
            phase: str,
            error: Mapping[str, Any] | None = None,
        ) -> None:
            nonlocal state
            state = commit_checkpoint(
                output_dir,
                fingerprint_sha256=fingerprint,
                model_results=ordered_rows(),
                attempts=attempts,
            )
            write_views(status, error=error)
            write_progress(status, phase=phase, error=error)

        if state.generation == 0:
            commit_and_publish("running", phase="batch_initialized")
        else:
            write_views("running")

        reusable_candidate_ids = {
            cast(str, row["candidate_id"])
            for row in model_rows
            if row.get("status") == ModeStatus.SUCCEEDED.value
        }
        if context.resume:
            retry_rows = [
                row for row in model_rows if row.get("status") != ModeStatus.SUCCEEDED.value
            ]
            if retry_rows:
                retry_ids = {cast(str, row["candidate_id"]) for row in retry_rows}
                model_rows[:] = [
                    row for row in model_rows if cast(str, row["candidate_id"]) not in retry_ids
                ]
                commit_and_publish("running", phase="resume_requeued_incomplete_models")
        elif any(row.get("status") != ModeStatus.SUCCEEDED.value for row in model_rows):
            raise ValueError("fresh batch run cannot inherit prior failed model results")

        write_progress("running", phase="batch_running")

        for model_index, batch_model in enumerate(batch.models):
            candidate_id = _candidate_id(model_index, batch_model)
            if candidate_id in reusable_candidate_ids:
                logger.info(
                    "Reusing verified EduLLM result for model %s (%d/%d)",
                    batch_model.model_id,
                    model_index + 1,
                    batch.model_count,
                )
                continue

            attempt_number, relative_output = allocate_attempt_path(output_dir, candidate_id)
            candidate_output = output_dir / relative_output
            candidate_prepared = _candidate_prepared_run(
                prepared,
                batch_model,
                source_provenance,
            )
            attempts_started_this_session += 1
            current_base = {
                "model_index": model_index,
                "candidate_id": candidate_id,
                "model_id": batch_model.model_id,
                "attempt_number": attempt_number,
                "output_dir": relative_output.as_posix(),
            }
            write_progress(
                "running",
                phase="candidate_started",
                current_model={**current_base, "candidate_phase": "initializing"},
            )
            logger.info(
                "Starting EduLLM model %s (%d/%d), attempt %d",
                batch_model.model_id,
                model_index + 1,
                batch.model_count,
                attempt_number,
            )

            def candidate_progress(
                event: Mapping[str, Any],
                current: Mapping[str, Any] = current_base,
            ) -> None:
                candidate_phase = str(event.get("phase") or "running")
                write_progress(
                    "running",
                    phase="candidate_running",
                    current_model={
                        **current,
                        "candidate_phase": candidate_phase,
                        "scenario_id": event.get("scenario_id"),
                        "selected_scenarios": event.get("selected_scenarios"),
                        "completed_scenarios": event.get("completed_scenarios"),
                        "criterion_judgments": event.get("criterion_judgments"),
                    },
                )

            try:
                result = await self._run_candidate(
                    context,
                    candidate_prepared,
                    candidate_output,
                    progress_callback=candidate_progress,
                )
            except asyncio.CancelledError:
                error_row = {
                    "type": "CancelledError",
                    "message": "batch evaluation was cancelled",
                }
                # The interrupted attempt is intentionally left uncommitted and
                # preserved. A resume allocates a new attempt directory.
                write_views(ModeStatus.CANCELLED.value, error=error_row)
                write_progress(
                    ModeStatus.CANCELLED.value,
                    phase="candidate_cancelled_uncommitted",
                    current_model={**current_base, "candidate_phase": "cancelled"},
                    error=error_row,
                )
                raise
            except Exception as exc:
                # Preserve the possibly partial attempt and record a separate,
                # complete failure envelope that can be safely checkpointed.
                logger.exception(
                    "Unexpected candidate wrapper failure for %s", batch_model.model_id
                )
                attempt_number, relative_output = allocate_attempt_path(output_dir, candidate_id)
                candidate_output = output_dir / relative_output
                failure_paths = {
                    "manifest.json": candidate_output / "manifest.json",
                    "tutor_responses.jsonl": candidate_output / "tutor_responses.jsonl",
                    "judge_rows.jsonl": candidate_output / "judge_rows.jsonl",
                    "cat_result.json": candidate_output / "cat_result.json",
                    "cat_trace.jsonl": candidate_output / "cat_trace.jsonl",
                }
                failure_path = self._write_failure(
                    context,
                    candidate_prepared,
                    failure_paths,
                    (),
                    (),
                    (),
                    (),
                    status=ModeStatus.FAILED.value,
                    error=exc,
                )
                result = ModeResult(
                    mode=self.name,
                    implementation_version=self.implementation_version,
                    status=ModeStatus.FAILED,
                    artifacts=(*tuple(failure_paths), failure_path.name),
                    completed_units=0,
                    error=f"{type(exc).__name__}: {exc}",
                )

            if result.status == ModeStatus.CANCELLED:
                raise asyncio.CancelledError("candidate returned cancelled status")
            row = _attempt_result_row(
                result,
                context=context,
                prepared=prepared,
                mode_root=output_dir,
                model_index=model_index,
                batch_model=batch_model,
                attempt_number=attempt_number,
                relative_output=relative_output,
            )
            model_rows[:] = [
                previous for previous in model_rows if previous.get("candidate_id") != candidate_id
            ]
            model_rows.append(row)
            attempts.append(dict(row))
            attempts_completed_this_session += 1
            commit_and_publish("running", phase="candidate_committed")
            logger.info(
                "Committed EduLLM model %s with status %s (%d/%d)",
                batch_model.model_id,
                result.status.value,
                model_index + 1,
                batch.model_count,
            )

        counts = _batch_counts(batch, ordered_rows())
        failed = counts["models_failed"]
        status = ModeStatus.FAILED if failed else ModeStatus.SUCCEEDED
        error_text = None if not failed else f"{failed} tutor model evaluation(s) failed"
        error_row = (
            None if error_text is None else {"type": "BatchModelFailure", "message": error_text}
        )
        write_views(status.value, error=error_row)
        write_progress(status.value, phase="batch_completed", error=error_row)
        warnings: list[str] = []
        if prepared.config.bank.scientific_status == "experimental":
            warnings.append("fitted bank and CAT policy are explicitly experimental")
        if failed:
            warnings.append(error_text or "one or more tutor model evaluations failed")
        if counts["criterion_no_decisions_with_metrics"]:
            warnings.append(
                f"{counts['criterion_no_decisions_with_metrics']} criterion judgment(s) "
                "were no-decision"
            )
        return ModeResult(
            mode=self.name,
            implementation_version=self.implementation_version,
            status=status,
            metrics=counts,
            artifacts=(*tuple(paths), "checkpoints/latest.json"),
            warnings=tuple(warnings),
            completed_units=len(ordered_rows()),
            error=error_text,
        )

    @staticmethod
    def _write_failure(
        context: ModeRunContext,
        prepared: _PreparedRun,
        paths: Mapping[str, Path],
        tutor_rows: Sequence[Mapping[str, Any]],
        judge_rows: Sequence[Mapping[str, Any]],
        selected_scenarios: Sequence[str],
        completed_scenarios: Sequence[str],
        *,
        status: str,
        error: BaseException,
    ) -> Path:
        error_row = {
            "type": type(error).__name__,
            "message": str(error),
        }
        failure_path = paths["manifest.json"].with_name("failure.json")
        _atomic_write_jsonl(paths["tutor_responses.jsonl"], tutor_rows)
        _atomic_write_jsonl(paths["judge_rows.jsonl"], judge_rows)
        if not paths["cat_trace.jsonl"].is_file():
            _atomic_write_jsonl(paths["cat_trace.jsonl"], ())
        _atomic_write_json(
            paths["cat_result.json"],
            {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "status": status,
                "partial": True,
                "selected_scenarios": list(selected_scenarios),
                "completed_scenarios": list(completed_scenarios),
                "error": error_row,
            },
        )
        _atomic_write_json(
            failure_path,
            {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "status": status,
                "completed_scenarios": len(completed_scenarios),
                "completed_tutor_responses": len(tutor_rows),
                "completed_criterion_judgments": len(judge_rows),
                "error": error_row,
            },
        )
        artifact_hashes = {
            name: sha256_file(path)
            for name, path in paths.items()
            if name != "manifest.json" and path.exists()
        }
        artifact_hashes[failure_path.name] = sha256_file(failure_path)
        _atomic_write_json(
            paths["manifest.json"],
            _manifest(
                context,
                prepared,
                status=status,
                tutor_rows=tutor_rows,
                judge_rows=judge_rows,
                selected_scenarios=selected_scenarios,
                completed_scenarios=completed_scenarios,
                artifacts=artifact_hashes,
                error=error_row,
            ),
        )
        return failure_path


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "ATOMIC_REQUIREMENT_POLICY",
    "BATCH_ARTIFACT_SCHEMA_VERSION",
    "EduLLMAdaptiveConfig",
    "EduLLMAdaptiveMode",
    "IMPLEMENTATION_VERSION",
    "MODE_CONFIG_SCHEMA_VERSION",
    "MODE_NAME",
    "RequirementPlan",
    "TutorResponseSourceConfig",
    "build_batch_resume_contract_payload",
    "parse_adaptive_config",
    "preflight_adaptive_config",
    "validate_batch_resume_artifacts",
]
