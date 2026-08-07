"""No-GPU lifecycle tests for the top-level multi-mode orchestrator."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

import olmo_eval.runners.mode_orchestrator as orchestrator_module
from olmo_eval.harness.config import HarnessConfig
from olmo_eval.inference.providers.config import ProviderConfig
from olmo_eval.inference.registry import ProviderRegistry
from olmo_eval.runners.mode_config import MODE_CONFIG_SCHEMA_VERSION, ModeRunConfig
from olmo_eval.runners.mode_orchestrator import ModeRunOrchestrator
from olmo_eval.runners.modes import (
    EvaluationModeRegistry,
    ModeResult,
    ModeRunContext,
    ModeSpec,
    ModeStatus,
)
from olmo_eval.runners.standard_mode import StandardOlmoMode, StandardOlmoRunRequest

CANDIDATE_MODEL = "fixture-candidate"
JUDGE_MODEL = "fixture-judge"
RUN_ID = "shared-fixture-run"


def _provider(model: str) -> ProviderConfig:
    return ProviderConfig(kind="mock", model=model)


def _harness(*, auxiliaries: Mapping[str, ProviderConfig] | None = None) -> HarnessConfig:
    return HarnessConfig(
        name="mode-orchestrator-fixture",
        provider=_provider(CANDIDATE_MODEL),
        auxiliary_providers=dict(auxiliaries or {}),
    )


def _standard_spec(harness: HarnessConfig) -> ModeSpec:
    return ModeSpec(
        "standard_olmo",
        {
            "harness_config": harness.to_dict(),
            "task_specs": ["fixture:task"],
            "save_predictions": False,
        },
    )


def _config(
    tmp_path: Path,
    *,
    modes: tuple[ModeSpec, ...] | None = None,
    auxiliaries: Mapping[str, ProviderConfig] | None = None,
) -> ModeRunConfig:
    harness = _harness(auxiliaries=auxiliaries)
    return ModeRunConfig(
        schema_version=MODE_CONFIG_SCHEMA_VERSION,
        run_id=RUN_ID,
        output_dir=(tmp_path / "mode-run").resolve(),
        harness=harness,
        modes=modes or (_standard_spec(harness),),
        metadata={"fixture": True},
    )


@dataclass
class _TrackingManager:
    configs: dict[str, ProviderConfig]
    available_gpu_ids: list[int]
    log_dir: str
    fail_start: bool = False
    fail_shutdown: bool = False
    start_calls: int = 0
    shutdown_calls: int = 0

    def start(self) -> dict[str, list[dict[str, Any]]]:
        self.start_calls += 1
        if self.fail_start:
            raise RuntimeError("fixture manager startup failed")
        return {name: [config.to_dict()] for name, config in self.configs.items()}

    def shutdown(self) -> None:
        self.shutdown_calls += 1
        if self.fail_shutdown:
            raise RuntimeError("fixture manager shutdown failed")


@dataclass
class _ManagerFactory:
    fail_create: bool = False
    fail_start: bool = False
    fail_shutdown: bool = False
    instances: list[_TrackingManager] = field(default_factory=list)

    def __call__(self, **kwargs: Any) -> _TrackingManager:
        if self.fail_create:
            raise RuntimeError("fixture manager creation failed")
        manager = _TrackingManager(
            fail_start=self.fail_start,
            fail_shutdown=self.fail_shutdown,
            **kwargs,
        )
        self.instances.append(manager)
        return manager


class _RecordingMode:
    implementation_version = "fixture-mode-v1"

    def __init__(
        self,
        name: str,
        *,
        required_auxiliaries: tuple[str, ...] = (),
        config_preflight_error: str | None = None,
    ) -> None:
        self.name = name
        self.required_auxiliary_providers = required_auxiliaries
        self.config_preflight_error = config_preflight_error
        self.contexts: list[ModeRunContext] = []
        self.judges: list[Any] = []

    def preflight_config(self, config: Mapping[str, Any]) -> None:
        del config
        if self.config_preflight_error:
            raise ValueError(self.config_preflight_error)

    def preflight(self, context: ModeRunContext, config: Mapping[str, Any]) -> None:
        del context, config

    async def run(
        self,
        context: ModeRunContext,
        config: Mapping[str, Any],
        output_dir: Path,
    ) -> ModeResult:
        del config, output_dir
        self.contexts.append(context)
        for name in self.required_auxiliary_providers:
            self.judges.append(context.get_provider(name))
        return ModeResult(
            mode=self.name,
            implementation_version=self.implementation_version,
            status=ModeStatus.SUCCEEDED,
        )


def _registry_factory(*modes: _RecordingMode) -> Any:
    def build_registry(
        resolved_primary: ProviderConfig | None = None,
        resolved_auxiliaries: Mapping[str, ProviderConfig] | None = None,
    ) -> EvaluationModeRegistry:
        del resolved_primary, resolved_auxiliaries
        registry = EvaluationModeRegistry()
        for mode in modes:
            registry.register(mode)
        return registry

    return build_registry


class _FixtureNativeRunner:
    def __init__(self, request: StandardOlmoRunRequest) -> None:
        self.output_dir = str(request.output_dir)
        self.task_specs = list(request.task_specs)
        self.model_name = CANDIDATE_MODEL

    def validate(self) -> None:
        pass

    async def run_async(self) -> Mapping[str, Any]:
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "metrics.json").write_text(
            json.dumps({"summary": {}, "tasks": [], "errors": []}) + "\n",
            encoding="utf-8",
        )
        return {}


def test_standard_only_provisions_candidate_not_judge_and_shares_run_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    requests: list[StandardOlmoRunRequest] = []

    def runner_factory(request: StandardOlmoRunRequest) -> _FixtureNativeRunner:
        requests.append(request)
        return _FixtureNativeRunner(request)

    def registry_factory(
        resolved_primary: ProviderConfig | None = None,
        resolved_auxiliaries: Mapping[str, ProviderConfig] | None = None,
    ) -> EvaluationModeRegistry:
        del resolved_primary, resolved_auxiliaries
        registry = EvaluationModeRegistry()
        registry.register(StandardOlmoMode(runner_factory))
        return registry

    monkeypatch.setattr(orchestrator_module, "_registry", registry_factory)
    managers = _ManagerFactory()

    results = ModeRunOrchestrator(
        config,
        manager_factory=managers,
        available_gpu_ids=(),
    ).run()

    assert results["standard_olmo"].succeeded
    assert len(managers.instances) == 1
    manager = managers.instances[0]
    assert set(manager.configs) == {"candidate"}
    assert manager.start_calls == 1
    assert manager.shutdown_calls == 1
    # The native runner is built once for preflight and once for execution.
    assert len(requests) == 2
    assert {request.experiment_id for request in requests} == {RUN_ID}
    assert {request.experiment_name for request in requests} == {RUN_ID}

    manifest = json.loads((config.output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_id"] == RUN_ID
    assert manifest["selected_modes"] == ["standard_olmo"]
    assert manifest["status"] == "succeeded"


def test_combined_run_uses_one_candidate_and_judge_manager_then_cleans_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auxiliaries = {"judge": _provider(JUDGE_MODEL)}
    harness = _harness(auxiliaries=auxiliaries)
    config = _config(
        tmp_path,
        auxiliaries=auxiliaries,
        modes=(
            _standard_spec(harness),
            ModeSpec("edullm_adaptive", {"fixture": True}),
        ),
    )
    standard = _RecordingMode("standard_olmo")
    adaptive = _RecordingMode(
        "edullm_adaptive",
        required_auxiliaries=("judge",),
    )
    monkeypatch.setattr(
        orchestrator_module,
        "_registry",
        _registry_factory(standard, adaptive),
    )

    registry_closes: list[ProviderRegistry] = []
    original_aclose = ProviderRegistry.aclose

    async def tracking_aclose(self: ProviderRegistry) -> None:
        registry_closes.append(self)
        await original_aclose(self)

    monkeypatch.setattr(ProviderRegistry, "aclose", tracking_aclose)
    managers = _ManagerFactory()

    results = ModeRunOrchestrator(
        config,
        manager_factory=managers,
        runtime_checker=lambda provider: {"checked_model": provider.model},
        available_gpu_ids=(),
    ).run()

    assert list(results) == ["standard_olmo", "edullm_adaptive"]
    assert all(result.succeeded for result in results.values())
    assert len(managers.instances) == 1
    manager = managers.instances[0]
    assert set(manager.configs) == {"candidate", "judge"}
    assert manager.start_calls == 1
    assert manager.shutdown_calls == 1
    assert len(registry_closes) == 1
    assert standard.contexts[0] is adaptive.contexts[0]
    assert standard.contexts[0].provider is adaptive.contexts[0].provider
    assert adaptive.judges[0].model_name == JUDGE_MODEL


def test_manager_startup_failure_writes_shared_failure_and_always_shuts_down(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    managers = _ManagerFactory(fail_start=True)

    with pytest.raises(RuntimeError, match="fixture manager startup failed"):
        ModeRunOrchestrator(
            config,
            manager_factory=managers,
            available_gpu_ids=(),
        ).run()

    assert len(managers.instances) == 1
    manager = managers.instances[0]
    assert manager.start_calls == 1
    assert manager.shutdown_calls == 1
    manifest = json.loads((config.output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_id"] == RUN_ID
    assert manifest["status"] == "failed"
    assert manifest["completed_modes"] == []
    assert manifest["context_metadata"]["mode_runner"] == {
        "schema_version": MODE_CONFIG_SCHEMA_VERSION,
        "candidate_model": CANDIDATE_MODEL,
        "candidate_revision": None,
        "candidate_provider": "mock",
        "provider_names": ["candidate"],
        "judge_runtime": {},
    }
    assert manifest["shared_error"] == {
        "stage": "provider_startup",
        "type": "RuntimeError",
        "message": "fixture manager startup failed",
    }
    report = json.loads((config.output_dir / "report_index.json").read_text(encoding="utf-8"))
    assert report == {
        "schema_version": "olmo-eval-mode-run-v1",
        "run_id": RUN_ID,
        "status": "failed",
        "modes": {},
    }


def test_primary_failure_preserves_structured_cleanup_failure(tmp_path: Path) -> None:
    config = _config(tmp_path)
    managers = _ManagerFactory(fail_start=True, fail_shutdown=True)

    with pytest.raises(RuntimeError, match="fixture manager startup failed"):
        ModeRunOrchestrator(
            config,
            manager_factory=managers,
            available_gpu_ids=(),
        ).run()

    manifest = json.loads((config.output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["shared_error"]["stage"] == "provider_startup"
    assert manifest["shared_error"]["message"] == "fixture manager startup failed"
    assert manifest["cleanup_errors"] == [
        {
            "stage": "inference_manager_shutdown",
            "type": "RuntimeError",
            "message": "fixture manager shutdown failed",
        }
    ]


def test_manager_creation_failure_writes_shared_failure(tmp_path: Path) -> None:
    config = _config(tmp_path)
    managers = _ManagerFactory(fail_create=True)

    with pytest.raises(RuntimeError, match="fixture manager creation failed"):
        ModeRunOrchestrator(
            config,
            manager_factory=managers,
            available_gpu_ids=(),
        ).run()

    manifest = json.loads((config.output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["shared_error"] == {
        "stage": "provider_manager_creation",
        "type": "RuntimeError",
        "message": "fixture manager creation failed",
    }


def test_cleanup_failure_marks_completed_run_failed_without_losing_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    mode = _RecordingMode("standard_olmo")
    monkeypatch.setattr(orchestrator_module, "_registry", _registry_factory(mode))
    managers = _ManagerFactory(fail_shutdown=True)

    with pytest.raises(RuntimeError, match="provider cleanup failed"):
        ModeRunOrchestrator(
            config,
            manager_factory=managers,
            available_gpu_ids=(),
        ).run()

    assert managers.instances[0].shutdown_calls == 1
    manifest = json.loads((config.output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["completed_modes"] == ["standard_olmo"]
    assert manifest["results"][0]["status"] == "succeeded"
    assert manifest["shared_error"] == {
        "stage": "provider_cleanup",
        "type": "RuntimeError",
        "message": "provider cleanup failed: RuntimeError: fixture manager shutdown failed",
    }
    assert manifest["cleanup_errors"] == [
        {
            "stage": "inference_manager_shutdown",
            "type": "RuntimeError",
            "message": "fixture manager shutdown failed",
        }
    ]
    report = json.loads((config.output_dir / "report_index.json").read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["modes"]["standard_olmo"]["status"] == "succeeded"


def test_nonempty_output_root_is_rejected_before_manager_creation(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.output_dir.mkdir(parents=True)
    (config.output_dir / "existing.txt").write_text("do not overwrite\n", encoding="utf-8")

    def forbidden_manager(**kwargs: Any) -> _TrackingManager:
        del kwargs
        raise AssertionError("manager must not be created")

    with pytest.raises(ValueError, match="must be new or empty"):
        ModeRunOrchestrator(
            config,
            manager_factory=forbidden_manager,
            available_gpu_ids=(),
        ).run()

    assert (config.output_dir / "existing.txt").read_text(encoding="utf-8") == (
        "do not overwrite\n"
    )


def test_mode_config_preflight_fails_before_manager_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    mode = _RecordingMode("standard_olmo", config_preflight_error="invalid fitted bank")
    monkeypatch.setattr(orchestrator_module, "_registry", _registry_factory(mode))

    def forbidden_manager(**kwargs: Any) -> _TrackingManager:
        del kwargs
        raise AssertionError("manager must not be created")

    with pytest.raises(ValueError, match="invalid fitted bank"):
        ModeRunOrchestrator(
            config,
            manager_factory=forbidden_manager,
            available_gpu_ids=(),
        ).run()

    assert not config.output_dir.exists()


def test_dispatch_failure_preserves_earlier_completed_mode_indexes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _WrongResultMode(_RecordingMode):
        async def run(
            self,
            context: ModeRunContext,
            config: Mapping[str, Any],
            output_dir: Path,
        ) -> ModeResult:
            del context, config, output_dir
            return ModeResult(
                mode="unexpected",
                implementation_version=self.implementation_version,
                status=ModeStatus.SUCCEEDED,
            )

    first = _RecordingMode("first")
    second = _WrongResultMode("second")
    config = _config(
        tmp_path,
        modes=(ModeSpec("first"), ModeSpec("second")),
    )
    monkeypatch.setattr(
        orchestrator_module,
        "_registry",
        _registry_factory(first, second),
    )

    with pytest.raises(ValueError, match="returned a result for 'unexpected'"):
        ModeRunOrchestrator(
            config,
            manager_factory=_ManagerFactory(),
            available_gpu_ids=(),
        ).run()

    manifest = json.loads((config.output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["completed_modes"] == ["first"]
    assert [result["mode"] for result in manifest["results"]] == ["first"]
    assert manifest["shared_error"]["stage"] == "mode_dispatch"
    report = json.loads((config.output_dir / "report_index.json").read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert set(report["modes"]) == {"first"}


def test_candidate_is_reserved_and_cannot_be_declared_as_an_auxiliary(tmp_path: Path) -> None:
    config = _config(tmp_path, auxiliaries={"candidate": _provider("shadow-candidate")})

    with pytest.raises(ValueError, match="reserved for the shared candidate provider"):
        ModeRunOrchestrator(config, available_gpu_ids=()).preflight()
