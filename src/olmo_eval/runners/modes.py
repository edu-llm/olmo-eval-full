"""Composable evaluation modes controlled by the OLMo Eval runner.

The dispatcher in this module owns run identity, sequencing, status, and artifact
namespaces.  An evaluation mode owns only its mode-specific behavior.  In
particular, an EduLLM mode receives already-created OLMo inference providers; it
does not start a second runner or construct hidden model clients.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from olmo_eval.inference.base import InferenceProvider
    from olmo_eval.inference.registry import ProviderLookup


MODE_RUN_SCHEMA_VERSION = "olmo-eval-mode-run-v1"
_MODE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class ModeStatus(StrEnum):
    """Terminal status for one evaluation mode."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BLOCKED_PREFLIGHT = "blocked_preflight"
    CANCELLED = "cancelled"


def _frozen_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))


def _json_safe(value: Any, *, path: str = "result") -> None:
    """Reject values that cannot be represented by strict, portable JSON."""

    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} contains a non-string mapping key")
            _json_safe(item, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _json_safe(item, path=f"{path}[{index}]")
        return
    raise ValueError(f"{path} contains unsupported value {type(value).__name__}")


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    _json_safe(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _validate_mode_name(name: str) -> str:
    if not _MODE_NAME.fullmatch(name):
        raise ValueError(
            "mode names must start with a letter or digit and contain only "
            "letters, digits, dots, underscores, and hyphens"
        )
    return name


def _validate_relative_artifact(value: str) -> str:
    path = Path(value)
    if not value or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"artifact path must be a safe relative path: {value!r}")
    return path.as_posix()


@dataclass(frozen=True, slots=True)
class ModeResult:
    """Small OLMo-facing envelope for one mode's native result files."""

    mode: str
    implementation_version: str
    status: ModeStatus
    metrics: Mapping[str, Any] = field(default_factory=dict)
    artifacts: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    completed_units: int | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        _validate_mode_name(self.mode)
        if not self.implementation_version.strip():
            raise ValueError("implementation_version cannot be blank")
        object.__setattr__(self, "metrics", _frozen_mapping(self.metrics))
        object.__setattr__(
            self,
            "artifacts",
            tuple(_validate_relative_artifact(value) for value in self.artifacts),
        )
        object.__setattr__(self, "warnings", tuple(str(value) for value in self.warnings))
        if self.completed_units is not None and self.completed_units < 0:
            raise ValueError("completed_units cannot be negative")
        if self.status == ModeStatus.SUCCEEDED and self.error is not None:
            raise ValueError("a succeeded mode cannot carry an error")
        if self.status in {ModeStatus.FAILED, ModeStatus.BLOCKED_PREFLIGHT} and not self.error:
            raise ValueError(f"{self.status.value} mode results require an error")
        _json_safe(self.to_dict())

    @property
    def succeeded(self) -> bool:
        return self.status == ModeStatus.SUCCEEDED

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "implementation_version": self.implementation_version,
            "status": self.status.value,
            "metrics": dict(self.metrics),
            "artifacts": list(self.artifacts),
            "warnings": list(self.warnings),
            "completed_units": self.completed_units,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class ModeRunContext:
    """Read-only resources supplied by OLMo to every selected mode."""

    run_id: str
    output_root: Path
    provider: InferenceProvider
    inference_pool: ProviderLookup | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.run_id.strip() or self.run_id in {".", ".."}:
            raise ValueError("run_id cannot be blank or a dot segment")
        root = Path(self.output_root).expanduser().resolve()
        object.__setattr__(self, "output_root", root)
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))
        _json_safe(self.metadata, path="context.metadata")

    def get_provider(self, name: str) -> InferenceProvider:
        if self.inference_pool is None:
            raise RuntimeError("No auxiliary inference providers configured.")
        return self.inference_pool.get(name)

    def mode_output_dir(self, mode: str) -> Path:
        name = _validate_mode_name(mode)
        result = (self.output_root / "modes" / name).resolve()
        modes_root = (self.output_root / "modes").resolve()
        if result.parent != modes_root:
            raise ValueError(f"unsafe mode output directory for {mode!r}")
        return result


@dataclass(frozen=True, slots=True)
class ModeSpec:
    """One selected mode and its mode-native configuration."""

    name: str
    config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_mode_name(self.name)
        object.__setattr__(self, "config", _frozen_mapping(self.config))
        _json_safe(self.config, path=f"mode.{self.name}.config")


@runtime_checkable
class EvaluationMode(Protocol):
    """Contract implemented by standard OLMo, EduLLM, and future modes."""

    @property
    def name(self) -> str: ...

    @property
    def implementation_version(self) -> str: ...

    @property
    def required_auxiliary_providers(self) -> tuple[str, ...]: ...

    def preflight_config(self, config: Mapping[str, Any]) -> None: ...

    def preflight(self, context: ModeRunContext, config: Mapping[str, Any]) -> None: ...

    async def run(
        self,
        context: ModeRunContext,
        config: Mapping[str, Any],
        output_dir: Path,
    ) -> ModeResult: ...


class EvaluationModeRegistry:
    """Registry that keeps the dispatcher independent of concrete modes."""

    def __init__(self) -> None:
        self._modes: dict[str, EvaluationMode] = {}

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._modes))

    def register(self, mode: EvaluationMode) -> None:
        name = _validate_mode_name(mode.name)
        if name in self._modes:
            raise ValueError(f"evaluation mode {name!r} is already registered")
        self._modes[name] = mode

    def get(self, name: str) -> EvaluationMode:
        try:
            return self._modes[name]
        except KeyError:
            raise KeyError(f"unknown evaluation mode {name!r}; available: {self.names}") from None


class EvaluationModeRunner:
    """Sequential OLMo-owned dispatcher with explicit partial-failure semantics."""

    def __init__(
        self,
        *,
        registry: EvaluationModeRegistry,
        context: ModeRunContext,
        modes: Sequence[ModeSpec],
        continue_on_mode_failure: bool = True,
    ) -> None:
        if not modes:
            raise ValueError("at least one evaluation mode is required")
        names = [mode.name for mode in modes]
        if len(names) != len(set(names)):
            raise ValueError("an evaluation mode cannot be selected more than once")
        self.registry = registry
        self.context = context
        self.modes = tuple(modes)
        self.continue_on_mode_failure = continue_on_mode_failure

    def preflight(self) -> None:
        """Validate shared declarations without running inference."""

        for spec in self.modes:
            self.registry.get(spec.name).preflight_config(spec.config)

    def run(self) -> dict[str, ModeResult]:
        return asyncio.run(self.run_async())

    async def run_async(self) -> dict[str, ModeResult]:
        self.context.output_root.mkdir(parents=True, exist_ok=True)
        self.preflight()
        results: dict[str, ModeResult] = {}

        # Validate every selected mode before any mode is allowed to spend
        # inference compute.  Mode preflight is deliberately read-only, so a
        # bad bank or later mode configuration cannot be discovered only after
        # an earlier mode has already run.
        preflight_failures: dict[str, ModeResult] = {}
        for spec in self.modes:
            mode = self.registry.get(spec.name)
            try:
                available = set(
                    self.context.inference_pool.names
                    if self.context.inference_pool is not None
                    else ()
                )
                missing = sorted(set(mode.required_auxiliary_providers) - available)
                if missing:
                    raise ValueError(
                        f"mode {spec.name!r} requires unavailable auxiliary "
                        f"provider(s): {missing}; available: {sorted(available)}"
                    )
                mode.preflight(self.context, spec.config)
            except Exception as exc:
                preflight_failures[spec.name] = ModeResult(
                    mode=spec.name,
                    implementation_version=mode.implementation_version,
                    status=ModeStatus.BLOCKED_PREFLIGHT,
                    error=f"{type(exc).__name__}: {exc}",
                )

        for spec in self.modes:
            mode = self.registry.get(spec.name)
            output_dir = self.context.mode_output_dir(spec.name)
            output_dir.mkdir(parents=True, exist_ok=True)

            result = preflight_failures.get(spec.name)
            if result is None:
                try:
                    result = await mode.run(self.context, spec.config, output_dir)
                except asyncio.CancelledError:
                    result = ModeResult(
                        mode=spec.name,
                        implementation_version=mode.implementation_version,
                        status=ModeStatus.CANCELLED,
                        error="evaluation mode was cancelled",
                    )
                    _atomic_write_json(output_dir / "mode_result.json", result.to_dict())
                    results[spec.name] = result
                    self._write_indexes(results)
                    raise
                except Exception as exc:
                    result = ModeResult(
                        mode=spec.name,
                        implementation_version=mode.implementation_version,
                        status=ModeStatus.FAILED,
                        error=f"{type(exc).__name__}: {exc}",
                    )

            if result.mode != spec.name:
                raise ValueError(f"mode {spec.name!r} returned a result for {result.mode!r}")
            if result.implementation_version != mode.implementation_version:
                raise ValueError(
                    f"mode {spec.name!r} returned implementation version "
                    f"{result.implementation_version!r}, expected "
                    f"{mode.implementation_version!r}"
                )
            _atomic_write_json(output_dir / "mode_result.json", result.to_dict())
            results[spec.name] = result
            # Checkpoint the shared indexes after every terminal mode. If a
            # later mode violates the dispatcher contract, earlier completed
            # work remains discoverable and can be overlaid with the shared
            # failure rather than disappearing from provenance.
            self._write_indexes(results)

            if not result.succeeded and not self.continue_on_mode_failure:
                break

        self._write_indexes(results)
        return results

    def _write_indexes(self, results: Mapping[str, ModeResult]) -> None:
        ordered = [results[name] for name in (spec.name for spec in self.modes) if name in results]
        failures = [result for result in ordered if not result.succeeded]
        if not failures:
            overall_status = "succeeded"
        elif len(failures) == len(ordered):
            overall_status = "failed"
        else:
            overall_status = "partial_failure"

        manifest = {
            "schema_version": MODE_RUN_SCHEMA_VERSION,
            "run_id": self.context.run_id,
            "status": overall_status,
            "continue_on_mode_failure": self.continue_on_mode_failure,
            "selected_modes": [spec.name for spec in self.modes],
            "completed_modes": [result.mode for result in ordered],
            "context_metadata": dict(self.context.metadata),
            "results": [result.to_dict() for result in ordered],
        }
        report_index = {
            "schema_version": MODE_RUN_SCHEMA_VERSION,
            "run_id": self.context.run_id,
            "status": overall_status,
            "modes": {
                result.mode: {
                    "status": result.status.value,
                    "result": f"modes/{result.mode}/mode_result.json",
                    "artifacts": [
                        f"modes/{result.mode}/{artifact}" for artifact in result.artifacts
                    ],
                }
                for result in ordered
            },
        }
        _atomic_write_json(self.context.output_root / "manifest.json", manifest)
        _atomic_write_json(self.context.output_root / "report_index.json", report_index)


__all__ = [
    "EvaluationMode",
    "EvaluationModeRegistry",
    "EvaluationModeRunner",
    "MODE_RUN_SCHEMA_VERSION",
    "ModeResult",
    "ModeRunContext",
    "ModeSpec",
    "ModeStatus",
]
