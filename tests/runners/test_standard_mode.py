"""No-inference tests for the native OLMo evaluation mode."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import olmo_eval.evals  # noqa: F401
from olmo_eval.harness.config import HarnessConfig, ProviderConfig
from olmo_eval.inference.providers.mock import MockProvider
from olmo_eval.runners.asynq.runner import AsyncEvalRunner
from olmo_eval.runners.modes import ModeRunContext, ModeStatus
from olmo_eval.runners.standard_mode import (
    AsyncEvalRunnerFactory,
    StandardOlmoConfig,
    StandardOlmoMode,
    StandardOlmoRunRequest,
)

MODEL = "fixture-standard-model"
SUITE = "arc:mc:olmo3base"


def _harness_config(
    *,
    provider: ProviderConfig | None = None,
    auxiliary_providers: dict[str, ProviderConfig] | None = None,
) -> HarnessConfig:
    return HarnessConfig(
        name="standard-fixture",
        provider=provider or ProviderConfig(kind="mock", model=MODEL),
        auxiliary_providers=auxiliary_providers or {},
    )


def _mode_config(**overrides: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "harness_config": _harness_config().to_dict(),
        "task_specs": [SUITE],
    }
    config.update(overrides)
    return config


def _context(tmp_path: Path) -> ModeRunContext:
    return ModeRunContext(
        run_id="standard-mode-fixture",
        output_root=tmp_path,
        provider=MockProvider(MODEL),
        metadata={"revision": "fixture"},
    )


def _metrics_document() -> dict[str, Any]:
    return {
        "timestamp": "2026-08-07T00:00:00+00:00",
        "config": {"provider": {"kind": "mock", "model": MODEL}},
        "tasks": [
            {
                "task": "arc_challenge:mc:olmo3base",
                "metrics": {"accuracy": {"exact_match": 0.75}},
                "num_instances": 4,
                "primary_metric": "accuracy:exact_match",
            },
            {
                "task": "arc_easy:mc:olmo3base",
                "metrics": {"accuracy": {"exact_match": 1.0}},
                "num_instances": 6,
                "primary_metric": "accuracy:exact_match",
            },
        ],
        "summary": {
            SUITE: {"metric": "primary_score:average", "score": 0.875},
        },
        "errors": [],
    }


async def _deterministic_native_run(runner: AsyncEvalRunner) -> Mapping[str, Any]:
    """Write the same artifact kinds as the native runner without model calls."""

    output_dir = Path(runner.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(_metrics_document(), indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "requests.jsonl").write_text(
        json.dumps({"task": SUITE, "prompt": "fixture"}) + "\n",
        encoding="utf-8",
    )
    predictions = output_dir / "predictions" / "fixture.jsonl"
    predictions.parent.mkdir(parents=True, exist_ok=True)
    predictions.write_text(
        json.dumps({"task": SUITE, "answer": "A"}) + "\n",
        encoding="utf-8",
    )
    return {
        "tasks": {
            "arc_challenge:mc:olmo3base": {
                "metrics": {"accuracy": {"exact_match": 0.75}},
            },
        }
    }


def test_config_passes_explicit_task_and_suite_names_through_unchanged() -> None:
    parsed = StandardOlmoConfig.from_mapping(
        _mode_config(
            task_specs=["piqa:mc", SUITE],
            task_overrides={"piqa:mc": {"limit": 3}},
            save_predictions=False,
            shuffle_seed=7,
        )
    )

    assert parsed.task_specs == ("piqa:mc", SUITE)
    assert dict(parsed.task_overrides["piqa:mc"]) == {"limit": 3}
    assert parsed.harness_config.provider.model == MODEL
    assert parsed.save_predictions is False
    assert parsed.shuffle_seed == 7


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({}, "requires harness_config"),
        ({"harness_config": {}}, "at least one task_specs"),
        (
            {"harness_config": {}, "task_specs": "piqa:mc"},
            "must be a list",
        ),
        (_mode_config(unknown=True), "unknown standard_olmo"),
        (_mode_config(save_requests=1), "save_requests must be a boolean"),
    ],
)
def test_config_rejects_ambiguous_or_implicit_selection(
    config: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        StandardOlmoConfig.from_mapping(config)


def test_production_factory_builds_the_real_native_runner(tmp_path: Path) -> None:
    harness = _harness_config(
        auxiliary_providers={
            "judge": ProviderConfig(kind="mock", model="must-not-start"),
        }
    )
    request = StandardOlmoRunRequest(
        harness_config=harness,
        task_specs=(SUITE,),
        task_overrides={},
        output_dir=tmp_path.resolve(),
        experiment_name="fixture-experiment",
        save_predictions=False,
        shuffle_seed=19,
        inspect_request=True,
    )

    runner = AsyncEvalRunnerFactory()(request)

    assert isinstance(runner, AsyncEvalRunner)
    assert runner.harness_config.provider is harness.provider
    assert runner.harness_config.auxiliary_providers == {}
    assert runner.task_specs == [SUITE]
    assert Path(runner.output_dir) == tmp_path.resolve()
    assert runner.experiment_name == "fixture-experiment"
    assert runner.save_predictions is False
    assert runner.shuffle_seed == 19
    assert runner.inspect_request is True
    # This is actual native task/suite validation; it performs no inference.
    runner.validate()


def test_production_factory_rejects_a_second_local_gpu_provider() -> None:
    request = StandardOlmoRunRequest(
        harness_config=_harness_config(provider=ProviderConfig(kind="vllm_server", model=MODEL)),
        task_specs=(SUITE,),
        task_overrides={},
        output_dir=Path("/tmp/standard-mode-fixture"),
    )

    with pytest.raises(ValueError, match="shared, already-resolved GPU provider"):
        AsyncEvalRunnerFactory()(request)


def test_production_factory_replaces_provider_with_resolved_shared_endpoint(
    tmp_path: Path,
) -> None:
    resolved = ProviderConfig(
        kind="vllm_server",
        model=MODEL,
        base_url="http://127.0.0.1:8123/v1",
    )
    request = StandardOlmoRunRequest(
        harness_config=_harness_config(provider=ProviderConfig(kind="vllm_server", model=MODEL)),
        task_specs=(SUITE,),
        task_overrides={},
        output_dir=tmp_path.resolve(),
    )

    runner = AsyncEvalRunnerFactory(resolved_provider_config=resolved)(request)

    assert runner.harness_config.provider is resolved
    assert runner.provider_config.base_url == "http://127.0.0.1:8123/v1"


def test_standard_mode_matches_direct_native_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def deterministic_run(self: AsyncEvalRunner) -> Mapping[str, Any]:
        return await _deterministic_native_run(self)

    monkeypatch.setattr(AsyncEvalRunner, "run_async", deterministic_run)
    factory = AsyncEvalRunnerFactory()

    direct_dir = tmp_path / "direct"
    direct_request = StandardOlmoRunRequest(
        harness_config=_harness_config(),
        task_specs=(SUITE,),
        task_overrides={},
        output_dir=direct_dir.resolve(),
    )
    direct_runner = factory(direct_request)
    direct_runner.validate()
    asyncio.run(direct_runner.run_async())

    context = _context(tmp_path / "wrapped-run")
    wrapped_dir = context.mode_output_dir("standard_olmo")
    wrapped_dir.mkdir(parents=True)
    mode = StandardOlmoMode()
    config = _mode_config()
    mode.preflight(context, config)
    result = asyncio.run(mode.run(context, config, wrapped_dir))

    assert result.status == ModeStatus.SUCCEEDED
    assert result.metrics == {
        "summary": {SUITE: {"metric": "primary_score:average", "score": 0.875}}
    }
    assert result.completed_units == 10
    assert result.artifacts == (
        "metrics.json",
        "predictions/fixture.jsonl",
        "requests.jsonl",
    )
    for relative_path in result.artifacts:
        assert (direct_dir / relative_path).read_bytes() == (
            wrapped_dir / relative_path
        ).read_bytes()


@dataclass
class _MisboundRunner:
    output_dir: str
    task_specs: list[str]
    model_name: str = MODEL

    def validate(self) -> None:
        return None

    async def run_async(self) -> Mapping[str, Any]:
        return {}


def test_preflight_rejects_runner_output_outside_mode_directory(tmp_path: Path) -> None:
    context = _context(tmp_path / "run")

    def factory(request: StandardOlmoRunRequest) -> _MisboundRunner:
        return _MisboundRunner(
            output_dir=str(tmp_path / "wrong"),
            task_specs=list(request.task_specs),
        )

    with pytest.raises(ValueError, match="isolated standard_olmo"):
        StandardOlmoMode(factory).preflight(context, _mode_config())


def test_preflight_rejects_different_candidate_model(tmp_path: Path) -> None:
    context = _context(tmp_path / "run")

    def factory(request: StandardOlmoRunRequest) -> _MisboundRunner:
        return _MisboundRunner(
            output_dir=str(request.output_dir),
            task_specs=list(request.task_specs),
            model_name="different-model",
        )

    with pytest.raises(ValueError, match="does not match the shared candidate"):
        StandardOlmoMode(factory).preflight(context, _mode_config())


def test_mode_requires_native_metrics_artifact(tmp_path: Path) -> None:
    context = _context(tmp_path / "run")
    output_dir = context.mode_output_dir("standard_olmo")
    output_dir.mkdir(parents=True)

    def factory(request: StandardOlmoRunRequest) -> _MisboundRunner:
        return _MisboundRunner(
            output_dir=str(request.output_dir),
            task_specs=list(request.task_specs),
        )

    with pytest.raises(RuntimeError, match="without writing metrics.json"):
        asyncio.run(StandardOlmoMode(factory).run(context, _mode_config(), output_dir))
