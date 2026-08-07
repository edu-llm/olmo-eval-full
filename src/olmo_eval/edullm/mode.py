"""Thin OLMo-owned execution mode for the EduLLM adaptive pipeline.

OLMo supplies and owns both inference providers.  This adapter validates an
explicit fitted-bank and scientific runtime configuration, generates one tutor
response for each adaptively selected scenario, applies the frozen Qwen binary
judge one criterion at a time, and delegates all state transitions to
``run_cat``.  It never creates a model client, launches vLLM, or calls a cloud
service directly.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from olmo_eval.edullm.ability import Quadrature, build_quadrature
from olmo_eval.edullm.bank import FittedBank, Rubric, Scenario, load_fitted_bank, sha256_file
from olmo_eval.edullm.cat import CatConfig, CatResult, ScenarioStep, run_cat
from olmo_eval.edullm.judge import (
    ADAPTER_VERSION,
    CLASSIFICATION_INSTRUCTION,
    CURATED_GRADING_POLICY,
    EVIDENCE_DECISION_POLICY,
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
)
from olmo_eval.edullm.tutor import (
    TutorGenerationConfig,
    generate_tutor_response,
)
from olmo_eval.runners.modes import ModeResult, ModeRunContext, ModeStatus

MODE_NAME = "edullm_adaptive"
IMPLEMENTATION_VERSION = "edullm-adaptive-olmo-v1"
MODE_CONFIG_SCHEMA_VERSION = "edullm-adaptive-mode-config-v1"
ARTIFACT_SCHEMA_VERSION = "edullm-adaptive-artifacts-v1"
ATOMIC_REQUIREMENT_POLICY = "criterion_as_single_atomic_unless_curation_finalized"
TUTOR_PROMPT_SOURCE = "origin/frq/infobench:eduLLM-Evals/tutor_cat/respgen/prompts.py"
TUTOR_PROMPT_VERSION = "frq-infobench-prompts-b4ea2e8"

_CURATION_FINALIZED_STATUS = "curation_v1_finalized"
_SCIENCE_STATUSES = frozenset({"validated", "experimental"})
_HEX_SHA256 = frozenset("0123456789abcdef")


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
class TutorRuntimeConfig:
    expected_model: str
    model_family: str
    model_provenance: Mapping[str, Any]
    generation: TutorGenerationConfig


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
class _PreparedRun:
    config: EduLLMAdaptiveConfig
    bank: FittedBank
    quadrature: Quadrature
    judge: QwenZeroShotBinaryJudge
    requirement_plans: Mapping[str, RequirementPlan]
    candidate_revision: str


@dataclass(frozen=True, slots=True)
class _PreparedInputs:
    config: EduLLMAdaptiveConfig
    bank: FittedBank
    quadrature: Quadrature
    requirement_plans: Mapping[str, RequirementPlan]


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
    _exact_keys(
        tutor_value,
        {"expected_model", "model_family", "model_provenance", "generation"},
        "tutor",
    )
    provenance = _mapping(tutor_value["model_provenance"], "tutor.model_provenance")
    _exact_keys(provenance, {"source", "revision"}, "tutor.model_provenance")
    _nonempty_string(provenance.get("source"), "tutor.model_provenance.source")
    _nonempty_string(provenance.get("revision"), "tutor.model_provenance.revision")
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
    tutor = TutorRuntimeConfig(
        expected_model=_nonempty_string(tutor_value["expected_model"], "tutor.expected_model"),
        model_family=_nonempty_string(tutor_value["model_family"], "tutor.model_family"),
        model_provenance=dict(provenance),
        generation=TutorGenerationConfig(
            max_tokens=_integer(generation_value["max_tokens"], "tutor.max_tokens", minimum=1),
            temperature=_finite_number(generation_value["temperature"], "tutor.temperature"),
            top_p=None if top_p is None else _finite_number(top_p, "tutor.top_p"),
            top_k=None if top_k is None else _integer(top_k, "tutor.top_k", minimum=1),
            stop_sequences=stops,
            num_samples=_integer(generation_value["num_samples"], "tutor.num_samples", minimum=1),
            do_sample=generation_value["do_sample"],
        ),
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
    return _PreparedInputs(
        config=config,
        bank=bank,
        quadrature=quadrature,
        requirement_plans=plans,
    )


def _prepare(context: ModeRunContext, raw_config: Mapping[str, Any]) -> _PreparedRun:
    inputs = preflight_adaptive_config(raw_config)
    config = inputs.config
    if context.provider.model_name != config.tutor.expected_model:
        raise ValueError(
            "primary tutor provider model does not match tutor.expected_model: "
            f"{context.provider.model_name!r} != {config.tutor.expected_model!r}"
        )
    configured_candidate_revision = str(config.tutor.model_provenance["revision"])
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
        if candidate_revision != configured_candidate_revision:
            raise ValueError(
                f"{source} revision does not match tutor.model_provenance.revision: "
                f"{candidate_revision!r} != {configured_candidate_revision!r}"
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
        quadrature=inputs.quadrature,
        judge=judge,
        requirement_plans=inputs.requirement_plans,
        candidate_revision=configured_candidate_revision,
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


def _bank_provenance(prepared: _PreparedRun) -> dict[str, Any]:
    bank = prepared.bank
    config = prepared.config.bank
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
    tutor_provenance = dict(config.tutor.model_provenance)
    tutor_provenance["revision"] = prepared.candidate_revision
    tutor_identity = {
        "model": context.provider.model_name,
        "model_family": config.tutor.model_family,
        "provenance": tutor_provenance,
    }
    judge_identity = {
        "provider": "judge",
        "model": config.judge.model,
        "model_family": config.judge.model_family,
        "revision": config.judge.revision,
        "enable_thinking": config.judge.enable_thinking,
        "language_model_only": config.judge.language_model_only,
    }
    prompt_contract = {
        "tutor_source": TUTOR_PROMPT_SOURCE,
        "tutor_version": TUTOR_PROMPT_VERSION,
        "judge_prompt_version": PROMPT_VERSION,
        "judge_prompt_profile": PROMPT_PROFILE,
        "judge_prompt_variant": PROMPT_VARIANT,
        "judge_adapter_version": ADAPTER_VERSION,
        "judge_contract_sha256": _canonical_hash(
            {
                "evidence": EVIDENCE_DECISION_POLICY,
                "curated": CURATED_GRADING_POLICY,
                "classification": CLASSIFICATION_INSTRUCTION,
            }
        ),
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
        "judge": {
            **judge_identity,
            "identity_sha256": _canonical_hash(judge_identity),
            "failure_probability_threshold": config.judge.failure_probability_threshold,
            "atomic_requirement_policy": config.judge.atomic_requirement_policy,
            "classification_token_ids": {
                "pass": list(prepared.judge.classification_token_ids.pass_ids),
                "fail": list(prepared.judge.classification_token_ids.fail_ids),
                "canonical_pass": prepared.judge.classification_token_ids.canonical_pass_id,
                "canonical_fail": prepared.judge.classification_token_ids.canonical_fail_id,
                "source": prepared.judge.classification_token_ids.source,
            },
        },
        "prompts": prompt_contract,
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


class EduLLMAdaptiveMode:
    """EvaluationMode implementation for adaptive EduLLM criterion grading."""

    name = MODE_NAME
    implementation_version = IMPLEMENTATION_VERSION
    required_auxiliary_providers = ("judge",)

    def preflight_config(self, config: Mapping[str, Any]) -> None:
        """Validate bank and scientific inputs before providers are started."""

        preflight_adaptive_config(config)

    def preflight(self, context: ModeRunContext, config: Mapping[str, Any]) -> None:
        """Fail closed before generation for config, provider, bank, or CAT errors."""

        _prepare(context, config)

    async def run(
        self,
        context: ModeRunContext,
        config: Mapping[str, Any],
        output_dir: Path,
    ) -> ModeResult:
        prepared = _prepare(context, config)
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
            response = await generate_tutor_response(
                context.provider,
                scenario,
                prepared.config.tutor.generation,
            )
            messages = [dict(message) for message in response.request.messages]
            tutor_row = {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "scenario_id": scenario.scenario_id,
                "tutor_model": context.provider.model_name,
                "messages": messages,
                "request_sha256": _canonical_hash(messages),
                "raw_output": response.output.text,
                "raw_output_sha256": _text_hash(response.output.text),
                "blank_response": not bool(response.output.text.strip()),
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
                    candidate_response=response.output.text,
                    conversation_context=tuple(scenario.conversation_context),
                    reference_solution=scenario.reference_solution,
                    expected_evidence=expected_evidence,
                )

                if not response.output.text.strip():
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
                    "candidate_response_sha256": _text_hash(response.output.text),
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
                    "scenarios_administered": len(result.scenarios_administered),
                    "criteria_observed": result.criteria_observed,
                    "criteria_no_decision": result.criteria_no_decision,
                    "mwle_converged": result.mwle_converged,
                    "theta_eap": dict(result.theta_eap),
                    "theta_mwle": (None if result.theta_mwle is None else dict(result.theta_mwle)),
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
    "EduLLMAdaptiveConfig",
    "EduLLMAdaptiveMode",
    "IMPLEMENTATION_VERSION",
    "MODE_CONFIG_SCHEMA_VERSION",
    "MODE_NAME",
    "RequirementPlan",
    "parse_adaptive_config",
    "preflight_adaptive_config",
]
