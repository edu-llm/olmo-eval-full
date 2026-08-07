"""Strict, side-effect-free configuration for multi-mode OLMo evaluation.

Parsing in this module creates no inference providers and performs no network
access.  It validates the public file schema, converts the shared harness to
OLMo's native :class:`HarnessConfig`, and injects that harness into the
internal ``standard_olmo`` mode configuration.
"""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

from omegaconf import DictConfig, ListConfig, OmegaConf

from olmo_eval.edullm.judge import QWEN_JUDGE_MODEL, QWEN_JUDGE_REVISION
from olmo_eval.harness.config import HarnessConfig
from olmo_eval.inference.providers.config import ProviderConfig
from olmo_eval.runners.modes import ModeSpec
from olmo_eval.runners.standard_mode import StandardOlmoConfig

MODE_CONFIG_SCHEMA_VERSION = "olmo-eval-mode-config-v1"

_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "output_dir",
        "harness",
        "modes",
        "continue_on_mode_failure",
        "metadata",
    }
)
_TOP_LEVEL_REQUIRED = frozenset({"schema_version", "run_id", "output_dir", "harness", "modes"})
_MODE_KEYS = frozenset({"name", "config"})
_HARNESS_KEYS = frozenset(
    {
        "name",
        "provider",
        "auxiliary_providers",
        "tool_names",
        "system_prompt",
        "tool_choice",
        "scaffold",
        "required_secrets",
        "max_turns",
        "max_concurrency",
        "scoring_concurrency",
        "scoring_process_pools",
        "sandboxes",
        "scaffold_kwargs",
        "sandbox_pool_instances",
        "sandbox_pool_min_instances",
        "metrics",
        "batching",
        "scorer_startup_timeout",
    }
)
_PROVIDER_KEYS = frozenset(
    {
        "kind",
        "model",
        "base_url",
        "api_base",
        "tokenizer",
        "revision",
        "force_download",
        "trust_remote_code",
        "dtype",
        "max_model_len",
        "max_concurrency",
        "num_instances",
        "required_secrets",
        "package",
        "dependencies",
        "kwargs",
    }
)
_UNRESOLVED_LOCAL_PROVIDER_KINDS = frozenset({"vllm", "hf", "olmo_core"})
_QWEN_KWARGS = frozenset(
    {
        "language_model_only",
        "tensor_parallel_size",
        "gpu_memory_utilization",
        "chat_template_kwargs",
    }
)
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _strict_json(value: Any, *, path: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} contains a non-string object key")
            _strict_json(item, path=f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _strict_json(item, path=f"{path}[{index}]")
        return
    raise ValueError(f"{path} contains non-JSON value {type(value).__name__}")


def _mapping(value: Any, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    *,
    allowed: frozenset[str],
    required: frozenset[str] = frozenset(),
    path: str,
) -> None:
    unknown = sorted(set(value) - allowed)
    missing = sorted(required - set(value))
    if unknown:
        raise ValueError(f"{path} has unknown field(s): {unknown}")
    if missing:
        raise ValueError(f"{path} is missing required field(s): {missing}")


def _provider_mapping(value: Any, *, path: str) -> Mapping[str, Any]:
    provider = _mapping(value, path=path)
    _exact_keys(
        provider,
        allowed=_PROVIDER_KEYS,
        required=frozenset({"model"}),
        path=path,
    )
    kind = provider.get("kind")
    if kind is not None and (not isinstance(kind, str) or not kind.strip()):
        raise ValueError(f"{path}.kind must be a non-blank string")
    return provider


def _parse_provider(value: Any, *, path: str) -> ProviderConfig:
    raw = _provider_mapping(value, path=path)
    try:
        provider = ProviderConfig.from_dict(copy.deepcopy(dict(raw)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {path}: {exc}") from exc

    if not isinstance(provider.model, str) or not provider.model.strip():
        raise ValueError(f"{path}.model must be non-blank")
    if (
        isinstance(provider.num_instances, bool)
        or not isinstance(provider.num_instances, int)
        or provider.num_instances != 1
    ):
        raise ValueError(f"{path}.num_instances must equal 1")
    kind = provider.get_provider_name()
    if not isinstance(kind, str) or not kind.strip():
        raise ValueError(f"{path}.kind must be a non-blank string")
    if kind in _UNRESOLVED_LOCAL_PROVIDER_KINDS:
        raise ValueError(
            f"{path}.kind={kind!r} is an unresolved local provider; "
            "mode runs require vllm_server, an API provider, or mock"
        )
    return provider


def _parse_harness(value: Any) -> tuple[HarnessConfig, Mapping[str, Any]]:
    raw = _mapping(value, path="harness")
    _exact_keys(
        raw,
        allowed=_HARNESS_KEYS,
        required=frozenset({"provider"}),
        path="harness",
    )
    _parse_provider(raw["provider"], path="harness.provider")

    auxiliaries_value = raw.get("auxiliary_providers", {})
    auxiliaries = _mapping(auxiliaries_value, path="harness.auxiliary_providers")
    for name, provider_value in auxiliaries.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("harness.auxiliary_providers names must be non-blank strings")
        _parse_provider(
            provider_value,
            path=f"harness.auxiliary_providers.{name}",
        )

    try:
        harness = HarnessConfig.from_dict(copy.deepcopy(dict(raw)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid harness: {exc}") from exc
    return harness, raw


def _validate_qwen_judge(harness: HarnessConfig, raw_harness: Mapping[str, Any]) -> None:
    try:
        judge = harness.auxiliary_providers["judge"]
    except KeyError:
        raise ValueError("edullm_adaptive requires harness.auxiliary_providers.judge") from None

    if judge.get_provider_name() != "vllm_server":
        raise ValueError("frozen Qwen judge kind must be 'vllm_server'")
    if judge.model != QWEN_JUDGE_MODEL:
        raise ValueError(f"frozen Qwen judge model must be {QWEN_JUDGE_MODEL!r}")
    if judge.revision != QWEN_JUDGE_REVISION:
        raise ValueError(f"frozen Qwen judge revision must be {QWEN_JUDGE_REVISION!r}")
    if judge.tokenizer is not None and judge.tokenizer != QWEN_JUDGE_MODEL:
        raise ValueError("frozen Qwen judge tokenizer must be null or equal to its model")
    if judge.dtype != "bfloat16":
        raise ValueError("frozen Qwen judge dtype must be 'bfloat16'")
    if judge.max_model_len != 32768:
        raise ValueError("frozen Qwen judge max_model_len must equal 32768")
    if judge.trust_remote_code is not False:
        raise ValueError("frozen Qwen judge trust_remote_code must be false")
    if isinstance(judge.num_instances, bool) or judge.num_instances != 1:
        raise ValueError("frozen Qwen judge num_instances must equal 1")

    raw_auxiliaries = _mapping(
        raw_harness.get("auxiliary_providers", {}),
        path="harness.auxiliary_providers",
    )
    raw_judge = _provider_mapping(
        raw_auxiliaries["judge"],
        path="harness.auxiliary_providers.judge",
    )
    kwargs = _mapping(
        raw_judge.get("kwargs", {}),
        path="harness.auxiliary_providers.judge.kwargs",
    )
    _exact_keys(
        kwargs,
        allowed=_QWEN_KWARGS,
        required=_QWEN_KWARGS,
        path="harness.auxiliary_providers.judge.kwargs",
    )
    if kwargs["language_model_only"] is not True:
        raise ValueError("frozen Qwen judge kwargs.language_model_only must be true")
    tensor_parallel = kwargs["tensor_parallel_size"]
    if isinstance(tensor_parallel, bool) or tensor_parallel != 1:
        raise ValueError("frozen Qwen judge kwargs.tensor_parallel_size must equal 1")
    gpu_memory = kwargs["gpu_memory_utilization"]
    if isinstance(gpu_memory, bool) or not isinstance(gpu_memory, (int, float)):
        raise ValueError("frozen Qwen judge kwargs.gpu_memory_utilization must equal 0.9")
    if float(gpu_memory) != 0.9:
        raise ValueError("frozen Qwen judge kwargs.gpu_memory_utilization must equal 0.9")

    chat_template = _mapping(
        kwargs["chat_template_kwargs"],
        path="harness.auxiliary_providers.judge.kwargs.chat_template_kwargs",
    )
    _exact_keys(
        chat_template,
        allowed=frozenset({"enable_thinking"}),
        required=frozenset({"enable_thinking"}),
        path="harness.auxiliary_providers.judge.kwargs.chat_template_kwargs",
    )
    if chat_template["enable_thinking"] is not False:
        raise ValueError(
            "frozen Qwen judge kwargs.chat_template_kwargs.enable_thinking must be false"
        )


def _run_id(value: Any) -> str:
    if not isinstance(value, str) or not _RUN_ID.fullmatch(value):
        raise ValueError(
            "run_id must start with a letter or digit and contain only letters, "
            "digits, dots, underscores, and hyphens"
        )
    return value


def _output_dir(value: Any) -> Path:
    if not isinstance(value, str) or not value.strip() or value.strip() in {".", ".."}:
        raise ValueError("output_dir must be a non-root directory path")
    try:
        path = Path(value).expanduser().resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"invalid output_dir: {exc}") from exc
    if path == Path(path.anchor):
        raise ValueError("output_dir cannot be a filesystem root")
    if path.exists() and not path.is_dir():
        raise ValueError("output_dir must be a directory path, not an existing file")
    return path


def _parse_modes(value: Any, harness: HarnessConfig) -> tuple[ModeSpec, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("modes must be a non-empty list")

    modes: list[ModeSpec] = []
    names: set[str] = set()
    for index, item in enumerate(value):
        path = f"modes[{index}]"
        raw_mode = _mapping(item, path=path)
        _exact_keys(
            raw_mode,
            allowed=_MODE_KEYS,
            required=_MODE_KEYS,
            path=path,
        )
        name = raw_mode["name"]
        if not isinstance(name, str):
            raise ValueError(f"{path}.name must be a string")
        if name in names:
            raise ValueError(f"mode {name!r} is selected more than once")
        names.add(name)

        raw_config = _mapping(raw_mode["config"], path=f"{path}.config")
        config = copy.deepcopy(dict(raw_config))
        if name == "standard_olmo":
            if "harness_config" in config:
                raise ValueError(
                    "standard_olmo.config must not contain harness_config; "
                    "the shared top-level harness is injected automatically"
                )
            config["harness_config"] = harness.to_dict()
            StandardOlmoConfig.from_mapping(config)

        try:
            modes.append(ModeSpec(name=name, config=config))
        except ValueError as exc:
            raise ValueError(f"invalid {path}: {exc}") from exc
    return tuple(modes)


@dataclass(frozen=True, slots=True)
class ModeRunConfig:
    """Validated configuration consumed by the mode-run orchestrator."""

    schema_version: str
    run_id: str
    output_dir: Path
    harness: HarnessConfig
    modes: tuple[ModeSpec, ...]
    continue_on_mode_failure: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def mode_names(self) -> tuple[str, ...]:
        return tuple(mode.name for mode in self.modes)


def parse_mapping(value: Mapping[str, Any]) -> ModeRunConfig:
    """Validate an already-loaded mode configuration without side effects."""

    raw = _mapping(value, path="config")
    _strict_json(raw, path="config")
    _exact_keys(
        raw,
        allowed=_TOP_LEVEL_KEYS,
        required=_TOP_LEVEL_REQUIRED,
        path="config",
    )
    if raw["schema_version"] != MODE_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema_version {raw['schema_version']!r}; "
            f"expected {MODE_CONFIG_SCHEMA_VERSION!r}"
        )

    continue_on_failure = raw.get("continue_on_mode_failure", True)
    if not isinstance(continue_on_failure, bool):
        raise ValueError("continue_on_mode_failure must be a boolean")
    metadata = _mapping(raw.get("metadata", {}), path="metadata")

    harness, raw_harness = _parse_harness(raw["harness"])
    modes = _parse_modes(raw["modes"], harness)
    adaptive_modes = [mode for mode in modes if mode.name == "edullm_adaptive"]
    if adaptive_modes:
        _validate_qwen_judge(harness, raw_harness)
        # Import lazily so standard-only mode files do not require the optional
        # EduLLM numerical dependency set merely to validate configuration.
        from olmo_eval.edullm.mode import parse_adaptive_config

        adaptive = parse_adaptive_config(adaptive_modes[0].config)
        provenance_source = adaptive.tutor.model_provenance.get("source")
        provenance_revision = adaptive.tutor.model_provenance.get("revision")
        if not isinstance(provenance_source, str) or not provenance_source.strip():
            raise ValueError("edullm_adaptive requires tutor.model_provenance.source")
        if not isinstance(provenance_revision, str) or not provenance_revision.strip():
            raise ValueError("edullm_adaptive requires tutor.model_provenance.revision")

        response_source_kind = adaptive.tutor.response_source.kind
        if response_source_kind == "precomputed_jsonl":
            if harness.provider.get_provider_name() != "mock":
                raise ValueError(
                    "edullm_adaptive precomputed_jsonl requires harness.provider.kind='mock'"
                )
            if any(mode.name == "standard_olmo" for mode in modes):
                raise ValueError(
                    "standard_olmo cannot run with an edullm_adaptive "
                    "precomputed_jsonl tutor response source"
                )
        elif response_source_kind == "provider":
            candidate_revision = harness.provider.revision
            if not isinstance(candidate_revision, str) or not candidate_revision.strip():
                raise ValueError(
                    "edullm_adaptive requires an explicit immutable harness.provider.revision"
                )
            if adaptive.tutor.expected_model != harness.provider.model:
                raise ValueError(
                    "edullm_adaptive tutor.expected_model must equal harness.provider.model"
                )
            if provenance_revision != candidate_revision:
                raise ValueError(
                    "edullm_adaptive tutor.model_provenance.revision must equal "
                    "harness.provider.revision"
                )
        else:  # pragma: no cover - parse_adaptive_config owns the closed enum.
            raise ValueError(
                f"unsupported edullm_adaptive tutor response source {response_source_kind!r}"
            )

    return ModeRunConfig(
        schema_version=MODE_CONFIG_SCHEMA_VERSION,
        run_id=_run_id(raw["run_id"]),
        output_dir=_output_dir(raw["output_dir"]),
        harness=harness,
        modes=modes,
        continue_on_mode_failure=continue_on_failure,
        metadata=MappingProxyType(copy.deepcopy(dict(metadata))),
    )


def _reject_interpolations(value: DictConfig | ListConfig, *, path: str = "config") -> None:
    if isinstance(value, ListConfig):
        for index in range(len(value)):
            child_path = f"{path}[{index}]"
            if OmegaConf.is_missing(value, index):
                raise ValueError(f"{child_path} cannot be missing")
            if OmegaConf.is_interpolation(value, index):
                raise ValueError(f"{child_path} cannot use OmegaConf interpolation")
            child = value[index]
            if isinstance(child, (DictConfig, ListConfig)):
                _reject_interpolations(child, path=child_path)
        return

    for key in value:
        if not isinstance(key, (str, int)):
            raise ValueError(f"{path} contains a non-string object key")
        child_path = f"{path}.{key}"
        if OmegaConf.is_missing(value, key):
            raise ValueError(f"{child_path} cannot be missing")
        if OmegaConf.is_interpolation(value, key):
            raise ValueError(f"{child_path} cannot use OmegaConf interpolation")
        child = value[key]
        if isinstance(child, (DictConfig, ListConfig)):
            _reject_interpolations(child, path=child_path)


def load_mode_config(path: str | Path) -> ModeRunConfig:
    """Load YAML or JSON through OmegaConf without resolving interpolations."""

    source = Path(path).expanduser()
    if source.suffix.lower() not in {".yaml", ".yml", ".json"}:
        raise ValueError("mode config file must use .yaml, .yml, or .json")
    if not source.is_file():
        raise ValueError(f"mode config file does not exist: {source}")
    try:
        loaded = OmegaConf.load(source)
    except Exception as exc:
        raise ValueError(f"could not load mode config {source}: {exc}") from exc
    if not isinstance(loaded, DictConfig):
        raise ValueError("mode config file must contain a top-level object")
    _reject_interpolations(loaded)
    try:
        container = OmegaConf.to_container(
            loaded,
            resolve=False,
            throw_on_missing=True,
            enum_to_str=True,
        )
    except Exception as exc:
        raise ValueError(f"could not materialize mode config {source}: {exc}") from exc
    if not isinstance(container, Mapping):
        raise ValueError("mode config file must contain a top-level object")
    return parse_mapping(cast("Mapping[str, Any]", container))


__all__ = [
    "MODE_CONFIG_SCHEMA_VERSION",
    "ModeRunConfig",
    "load_mode_config",
    "parse_mapping",
]
