"""Standard OLMo evaluation as a mode-dispatcher delegate.

This module deliberately delegates to :class:`AsyncEvalRunner`, OLMo Eval's
native task and suite runner.  It does not duplicate task preparation,
inference, scoring, suite aggregation, or artifact writing.  The mode wrapper
only validates its small configuration, binds the native runner to the mode's
isolated output directory, and indexes the native artifacts it produced.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol

from olmo_eval.harness.config import HarnessConfig
from olmo_eval.inference.providers.config import ProviderConfig
from olmo_eval.runners.modes import ModeResult, ModeRunContext, ModeStatus

if TYPE_CHECKING:
    from olmo_eval.storage import StorageBackend


MODE_NAME = "standard_olmo"
IMPLEMENTATION_VERSION = "olmo-async-runner-v1"
_CONFIG_KEYS = frozenset(
    {
        "harness_config",
        "task_specs",
        "task_overrides",
        "experiment_name",
        "experiment_group",
        "save_predictions",
        "save_requests",
        "shuffle_seed",
        "inspect_instance",
        "inspect_formatted",
        "inspect_tokens",
        "inspect_response",
        "inspect_request",
    }
)


def _string_sequence(value: Any, *, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{field_name} must be a list of task or suite names")
    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{field_name}[{index}] must be a non-blank string")
        result.append(item)
    return tuple(result)


def _task_overrides(value: Any) -> Mapping[str, Mapping[str, Any]]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise ValueError("task_overrides must be an object keyed by task or suite name")

    result: dict[str, Mapping[str, Any]] = {}
    for key, override in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("task_overrides keys must be non-blank strings")
        if not isinstance(override, Mapping):
            raise ValueError(f"task_overrides[{key!r}] must be an object")
        try:
            json.dumps(override, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"task_overrides[{key!r}] must contain strict JSON values") from exc
        result[key] = MappingProxyType(copy.deepcopy(dict(override)))
    return MappingProxyType(result)


def _bool_field(raw: Mapping[str, Any], name: str, *, default: bool) -> bool:
    value = raw.get(name, default)
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _optional_string(raw: Mapping[str, Any], name: str) -> str | None:
    value = raw.get(name)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be null or a non-blank string")
    return value


def _harness_config(value: Any) -> HarnessConfig:
    if not isinstance(value, Mapping):
        raise ValueError("harness_config must be a full HarnessConfig JSON object")
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("harness_config must contain strict JSON values") from exc
    try:
        return HarnessConfig.from_dict(copy.deepcopy(dict(value)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid harness_config: {exc}") from exc


@dataclass(frozen=True, slots=True)
class StandardOlmoConfig:
    """Validated native OLMo task/suite selection for one mode run."""

    harness_config: HarnessConfig
    task_specs: tuple[str, ...]
    task_overrides: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    experiment_name: str | None = None
    experiment_group: str | None = None
    save_predictions: bool = True
    save_requests: bool = True
    shuffle_seed: int = 42
    inspect_instance: bool = False
    inspect_formatted: bool = False
    inspect_tokens: bool = False
    inspect_response: bool = False
    inspect_request: bool = False

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> StandardOlmoConfig:
        unknown = sorted(set(raw) - _CONFIG_KEYS)
        if unknown:
            raise ValueError(f"unknown standard_olmo config field(s): {unknown}")

        if "harness_config" not in raw:
            raise ValueError("standard_olmo requires harness_config")
        specs = _string_sequence(raw.get("task_specs"), field_name="task_specs")
        if not specs:
            raise ValueError("standard_olmo requires at least one task_specs entry")

        shuffle_seed = raw.get("shuffle_seed", 42)
        if isinstance(shuffle_seed, bool) or not isinstance(shuffle_seed, int):
            raise ValueError("shuffle_seed must be an integer")

        return cls(
            harness_config=_harness_config(raw["harness_config"]),
            task_specs=tuple(specs),
            task_overrides=_task_overrides(raw.get("task_overrides")),
            experiment_name=_optional_string(raw, "experiment_name"),
            experiment_group=_optional_string(raw, "experiment_group"),
            save_predictions=_bool_field(raw, "save_predictions", default=True),
            save_requests=_bool_field(raw, "save_requests", default=True),
            shuffle_seed=shuffle_seed,
            inspect_instance=_bool_field(raw, "inspect_instance", default=False),
            inspect_formatted=_bool_field(raw, "inspect_formatted", default=False),
            inspect_tokens=_bool_field(raw, "inspect_tokens", default=False),
            inspect_response=_bool_field(raw, "inspect_response", default=False),
            inspect_request=_bool_field(raw, "inspect_request", default=False),
        )


@dataclass(frozen=True, slots=True)
class StandardOlmoRunRequest:
    """Arguments supplied by the mode wrapper to a native OLMo runner factory."""

    harness_config: HarnessConfig
    task_specs: tuple[str, ...]
    task_overrides: Mapping[str, Mapping[str, Any]]
    output_dir: Path
    experiment_id: str | None = None
    experiment_name: str | None = None
    experiment_group: str | None = None
    save_predictions: bool = True
    save_requests: bool = True
    shuffle_seed: int = 42
    inspect_instance: bool = False
    inspect_formatted: bool = False
    inspect_tokens: bool = False
    inspect_response: bool = False
    inspect_request: bool = False


class NativeOlmoRunner(Protocol):
    """Subset of the native runner contract used by the mode wrapper."""

    output_dir: str
    task_specs: list[str]
    model_name: str

    def validate(self) -> None: ...

    async def run_async(self) -> Mapping[str, Any]: ...


NativeOlmoRunnerFactory = Callable[[StandardOlmoRunRequest], NativeOlmoRunner]


@dataclass(frozen=True, slots=True)
class AsyncEvalRunnerFactory:
    """Build OLMo's real :class:`AsyncEvalRunner` for a standard mode run.

    Provider clients and cleanup remain on OLMo's existing worker path.  For a
    shared local model, the top-level orchestrator supplies a resolved
    ``vllm_server`` configuration whose ``base_url`` points at the one
    OLMo-owned server.
    """

    resolved_provider_config: ProviderConfig | None = None
    resolved_auxiliary_configs: Mapping[str, ProviderConfig] = field(default_factory=dict)
    storages: tuple[StorageBackend, ...] = ()
    attention_backend: str | None = None

    @staticmethod
    def _validate_shared_provider(provider: ProviderConfig) -> None:
        provider_kind = provider.get_provider_name()
        if provider.requires_gpu and (provider_kind != "vllm_server" or not provider.base_url):
            raise ValueError(
                "standard_olmo mode requires a shared, already-resolved GPU provider "
                "endpoint; configure vllm_server with base_url instead of starting a "
                "second local model inside AsyncEvalRunner"
            )

    def __call__(self, request: StandardOlmoRunRequest) -> NativeOlmoRunner:
        from olmo_eval.runners.asynq.runner import AsyncEvalRunner

        harness_config = request.harness_config
        if self.resolved_provider_config is not None:
            harness_config = harness_config.with_provider(self.resolved_provider_config)
        self._validate_shared_provider(harness_config.provider)

        # Auxiliary GPU servers are owned by the top-level mode orchestrator.
        # Preserve their resolved endpoints for native scorers without letting
        # the child runner start duplicate model servers.
        harness_config = replace(
            harness_config,
            auxiliary_providers=dict(self.resolved_auxiliary_configs),
        )

        attention_backend = self.attention_backend
        if attention_backend is None:
            attention_backend = harness_config.provider.kwargs.get("attention_backend")

        return AsyncEvalRunner(
            harness_config=harness_config,
            task_specs=list(request.task_specs),
            task_overrides={
                name: copy.deepcopy(dict(overrides))
                for name, overrides in request.task_overrides.items()
            },
            output_dir=str(request.output_dir),
            experiment_id=request.experiment_id,
            resolved_auxiliary_providers={
                name: [config.to_dict()] for name, config in self.resolved_auxiliary_configs.items()
            }
            or None,
            storages=list(self.storages),
            attention_backend=attention_backend,
            experiment_name=request.experiment_name,
            experiment_group=request.experiment_group,
            save_predictions=request.save_predictions,
            save_requests=request.save_requests,
            shuffle_seed=request.shuffle_seed,
            inspect_instance=request.inspect_instance,
            inspect_formatted=request.inspect_formatted,
            inspect_tokens=request.inspect_tokens,
            inspect_response=request.inspect_response,
            inspect_request=request.inspect_request,
        )


def _request(
    config: StandardOlmoConfig,
    output_dir: Path,
    *,
    experiment_id: str,
) -> StandardOlmoRunRequest:
    return StandardOlmoRunRequest(
        harness_config=config.harness_config,
        task_specs=config.task_specs,
        task_overrides=config.task_overrides,
        output_dir=output_dir.resolve(),
        experiment_id=experiment_id,
        experiment_name=config.experiment_name,
        experiment_group=config.experiment_group,
        save_predictions=config.save_predictions,
        save_requests=config.save_requests,
        shuffle_seed=config.shuffle_seed,
        inspect_instance=config.inspect_instance,
        inspect_formatted=config.inspect_formatted,
        inspect_tokens=config.inspect_tokens,
        inspect_response=config.inspect_response,
        inspect_request=config.inspect_request,
    )


def _validate_runner_binding(
    runner: NativeOlmoRunner,
    request: StandardOlmoRunRequest,
    context: ModeRunContext,
) -> None:
    actual_output = Path(runner.output_dir).expanduser().resolve()
    if actual_output != request.output_dir:
        raise ValueError(
            "native OLMo runner output_dir does not match the isolated standard_olmo directory"
        )
    if tuple(runner.task_specs) != request.task_specs:
        raise ValueError("native OLMo runner changed the selected task/suite specifications")

    context_model = getattr(context.provider, "model_name", None)
    if not isinstance(context_model, str) or not context_model:
        raise ValueError("the primary OLMo provider must expose a non-blank model_name")
    if runner.model_name != context_model:
        raise ValueError(
            "native OLMo runner model does not match the shared candidate provider: "
            f"{runner.model_name!r} != {context_model!r}"
        )


def _read_native_metrics(output_dir: Path) -> Mapping[str, Any]:
    metrics_path = output_dir / "metrics.json"
    if not metrics_path.is_file():
        raise RuntimeError("native OLMo runner completed without writing metrics.json")
    try:
        value = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("native OLMo metrics.json is not valid JSON") from exc
    if not isinstance(value, Mapping):
        raise RuntimeError("native OLMo metrics.json must contain a JSON object")
    return value


def _artifact_paths(output_dir: Path) -> tuple[str, ...]:
    root = output_dir.resolve()
    artifacts: list[str] = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"native OLMo artifact cannot be a symlink: {path}")
        if not path.is_file() or path.name == "mode_result.json":
            continue
        try:
            path.resolve().relative_to(root)
        except ValueError as exc:
            raise RuntimeError(f"native OLMo artifact escaped its mode directory: {path}") from exc
        artifacts.append(path.relative_to(output_dir).as_posix())
    return tuple(artifacts)


def _completed_units(metrics: Mapping[str, Any]) -> int | None:
    tasks = metrics.get("tasks")
    if not isinstance(tasks, list):
        return None
    counts: list[int] = []
    for task in tasks:
        if not isinstance(task, Mapping):
            return None
        count = task.get("num_instances")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            return None
        counts.append(count)
    return sum(counts)


def _warnings(metrics: Mapping[str, Any]) -> tuple[str, ...]:
    errors = metrics.get("errors")
    if not isinstance(errors, list):
        return ()
    warnings: list[str] = []
    for error in errors:
        if isinstance(error, str):
            warnings.append(error)
        else:
            warnings.append(json.dumps(error, sort_keys=True, ensure_ascii=False))
    return tuple(warnings)


class StandardOlmoMode:
    """Mode adapter that delegates unchanged task/suite execution to OLMo."""

    name = MODE_NAME
    implementation_version = IMPLEMENTATION_VERSION
    required_auxiliary_providers: tuple[str, ...] = ()

    def __init__(self, runner_factory: NativeOlmoRunnerFactory | None = None) -> None:
        self._runner_factory = runner_factory or AsyncEvalRunnerFactory()

    def preflight_config(self, config: Mapping[str, Any]) -> None:
        """Validate the native runner selection without constructing providers."""

        StandardOlmoConfig.from_mapping(config)

    def _build_runner(
        self,
        context: ModeRunContext,
        config: Mapping[str, Any],
        output_dir: Path,
    ) -> NativeOlmoRunner:
        parsed = StandardOlmoConfig.from_mapping(config)
        request = _request(parsed, output_dir, experiment_id=context.run_id)
        runner = self._runner_factory(request)
        _validate_runner_binding(runner, request, context)
        return runner

    def preflight(self, context: ModeRunContext, config: Mapping[str, Any]) -> None:
        runner = self._build_runner(context, config, context.mode_output_dir(self.name))
        runner.validate()

    async def run(
        self,
        context: ModeRunContext,
        config: Mapping[str, Any],
        output_dir: Path,
    ) -> ModeResult:
        runner = self._build_runner(context, config, output_dir)
        # Validate here as well so direct mode invocation has the same safety as
        # dispatcher invocation and cannot bypass native task/suite validation.
        runner.validate()
        native_result = await runner.run_async()
        if not isinstance(native_result, Mapping):
            raise RuntimeError("native OLMo runner returned a non-mapping result")

        metrics_document = _read_native_metrics(output_dir)
        summary = metrics_document.get("summary", {})
        if not isinstance(summary, Mapping):
            raise RuntimeError("native OLMo metrics.json summary must be an object")

        artifacts = _artifact_paths(output_dir)
        if "metrics.json" not in artifacts:
            raise RuntimeError("native OLMo metrics.json was not indexed as an artifact")

        return ModeResult(
            mode=self.name,
            implementation_version=self.implementation_version,
            status=ModeStatus.SUCCEEDED,
            metrics={"summary": copy.deepcopy(dict(summary))},
            artifacts=artifacts,
            warnings=_warnings(metrics_document),
            completed_units=_completed_units(metrics_document),
        )


__all__ = [
    "AsyncEvalRunnerFactory",
    "IMPLEMENTATION_VERSION",
    "MODE_NAME",
    "NativeOlmoRunner",
    "NativeOlmoRunnerFactory",
    "StandardOlmoConfig",
    "StandardOlmoMode",
    "StandardOlmoRunRequest",
]
