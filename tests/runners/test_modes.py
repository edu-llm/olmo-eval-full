"""Tests for the OLMo-owned multi-mode dispatcher."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from olmo_eval.runners.modes import (
    EvaluationModeRegistry,
    EvaluationModeRunner,
    ModeResult,
    ModeRunContext,
    ModeSpec,
    ModeStatus,
)


class _Provider:
    model_name = "fixture-primary"


class _ProviderLookup:
    def __init__(self, names: tuple[str, ...] = ()) -> None:
        self._providers = {name: _Provider() for name in names}

    @property
    def names(self) -> list[str]:
        return sorted(self._providers)

    def get(self, name: str) -> _Provider:
        return self._providers[name]


@dataclass
class _Mode:
    name: str
    implementation_version: str = "fixture-v1"
    required_auxiliary_providers: tuple[str, ...] = ()
    events: list[str] = field(default_factory=list)
    preflight_error: str | None = None
    run_error: str | None = None

    def preflight_config(self, config: dict[str, Any]) -> None:
        del config

    def preflight(self, context: ModeRunContext, config: dict[str, Any]) -> None:
        self.events.append(f"preflight:{self.name}")
        if self.preflight_error:
            raise ValueError(self.preflight_error)

    async def run(
        self,
        context: ModeRunContext,
        config: dict[str, Any],
        output_dir: Path,
    ) -> ModeResult:
        self.events.append(f"run:{self.name}")
        if self.run_error:
            raise RuntimeError(self.run_error)
        artifact = output_dir / "native.json"
        artifact.write_text(json.dumps({"mode": self.name}), encoding="utf-8")
        return ModeResult(
            mode=self.name,
            implementation_version=self.implementation_version,
            status=ModeStatus.SUCCEEDED,
            metrics={"score": float(config.get("score", 1.0))},
            artifacts=("native.json",),
            completed_units=1,
        )


def _context(tmp_path: Path, auxiliary: tuple[str, ...] = ()) -> ModeRunContext:
    return ModeRunContext(
        run_id="fixture-run",
        output_root=tmp_path,
        provider=_Provider(),  # type: ignore[arg-type]
        inference_pool=_ProviderLookup(auxiliary),  # type: ignore[arg-type]
        metadata={"revision": "abc123"},
    )


def test_registry_accepts_future_modes_without_dispatcher_changes() -> None:
    registry = EvaluationModeRegistry()
    registry.register(_Mode("standard_olmo"))
    registry.register(_Mode("edullm_adaptive"))
    registry.register(_Mode("future_design"))

    assert registry.names == ("edullm_adaptive", "future_design", "standard_olmo")
    assert registry.get("future_design").name == "future_design"


def test_dispatcher_runs_modes_in_order_and_isolates_artifacts(tmp_path: Path) -> None:
    events: list[str] = []
    standard = _Mode("standard_olmo", events=events)
    adaptive = _Mode(
        "edullm_adaptive",
        required_auxiliary_providers=("judge",),
        events=events,
    )
    registry = EvaluationModeRegistry()
    registry.register(standard)
    registry.register(adaptive)

    results = EvaluationModeRunner(
        registry=registry,
        context=_context(tmp_path, ("judge",)),
        modes=(
            ModeSpec("standard_olmo", {"score": 0.75}),
            ModeSpec("edullm_adaptive", {"score": 0.5}),
        ),
    ).run()

    assert events == [
        "preflight:standard_olmo",
        "preflight:edullm_adaptive",
        "run:standard_olmo",
        "run:edullm_adaptive",
    ]
    assert all(result.succeeded for result in results.values())
    assert (tmp_path / "modes/standard_olmo/native.json").is_file()
    assert (tmp_path / "modes/edullm_adaptive/native.json").is_file()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    index = json.loads((tmp_path / "report_index.json").read_text())
    assert manifest["status"] == "succeeded"
    assert manifest["selected_modes"] == ["standard_olmo", "edullm_adaptive"]
    assert index["status"] == "succeeded"
    assert index["modes"]["edullm_adaptive"]["artifacts"] == ["modes/edullm_adaptive/native.json"]


def test_mode_specific_preflight_failure_does_not_erase_other_mode(tmp_path: Path) -> None:
    registry = EvaluationModeRegistry()
    registry.register(_Mode("standard_olmo"))
    registry.register(_Mode("edullm_adaptive", required_auxiliary_providers=("judge",)))

    results = EvaluationModeRunner(
        registry=registry,
        context=_context(tmp_path),
        modes=(ModeSpec("edullm_adaptive"), ModeSpec("standard_olmo")),
        continue_on_mode_failure=True,
    ).run()

    assert results["edullm_adaptive"].status == ModeStatus.BLOCKED_PREFLIGHT
    assert "unavailable auxiliary" in str(results["edullm_adaptive"].error)
    assert results["standard_olmo"].status == ModeStatus.SUCCEEDED
    assert json.loads((tmp_path / "manifest.json").read_text())["status"] == "partial_failure"


def test_all_preflights_finish_before_any_mode_runs(tmp_path: Path) -> None:
    events: list[str] = []
    registry = EvaluationModeRegistry()
    registry.register(_Mode("first", events=events))
    registry.register(_Mode("later_invalid", events=events, preflight_error="bad bank"))

    results = EvaluationModeRunner(
        registry=registry,
        context=_context(tmp_path),
        modes=(ModeSpec("first"), ModeSpec("later_invalid")),
        continue_on_mode_failure=True,
    ).run()

    assert events == ["preflight:first", "preflight:later_invalid", "run:first"]
    assert results["first"].status == ModeStatus.SUCCEEDED
    assert results["later_invalid"].status == ModeStatus.BLOCKED_PREFLIGHT


def test_fail_fast_stops_after_failed_mode_but_writes_indexes(tmp_path: Path) -> None:
    registry = EvaluationModeRegistry()
    registry.register(_Mode("first", run_error="fixture failure"))
    registry.register(_Mode("second"))

    results = EvaluationModeRunner(
        registry=registry,
        context=_context(tmp_path),
        modes=(ModeSpec("first"), ModeSpec("second")),
        continue_on_mode_failure=False,
    ).run()

    assert tuple(results) == ("first",)
    assert results["first"].status == ModeStatus.FAILED
    assert (tmp_path / "manifest.json").is_file()
    assert not (tmp_path / "modes/second").exists()


@pytest.mark.parametrize("artifact", ("../escape.json", "/tmp/escape.json", ""))
def test_result_rejects_unsafe_artifact_paths(artifact: str) -> None:
    with pytest.raises(ValueError, match="safe relative path"):
        ModeResult(
            mode="safe",
            implementation_version="v1",
            status=ModeStatus.SUCCEEDED,
            artifacts=(artifact,),
        )


def test_result_rejects_nonfinite_metrics() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        ModeResult(
            mode="safe",
            implementation_version="v1",
            status=ModeStatus.SUCCEEDED,
            metrics={"bad": float("nan")},
        )


def test_cancellation_writes_terminal_mode_result(tmp_path: Path) -> None:
    class _Cancelled(_Mode):
        async def run(self, context, config, output_dir):  # type: ignore[no-untyped-def]
            raise asyncio.CancelledError

    registry = EvaluationModeRegistry()
    registry.register(_Cancelled("cancelled"))
    runner = EvaluationModeRunner(
        registry=registry,
        context=_context(tmp_path),
        modes=(ModeSpec("cancelled"),),
    )

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(runner.run_async())

    result = json.loads((tmp_path / "modes/cancelled/mode_result.json").read_text())
    assert result["status"] == "cancelled"
