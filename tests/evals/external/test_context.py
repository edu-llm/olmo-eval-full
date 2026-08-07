"""No-GPU tests for external-evaluation provider contexts."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from olmo_eval.common.types import ProviderKind
from olmo_eval.evals.external import ExternalEval, ExternalEvalContext, ExternalEvalResult
from olmo_eval.inference.base import InferenceProvider
from olmo_eval.inference.providers.config import ProviderConfig
from olmo_eval.inference.providers.mock import MockProvider
from olmo_eval.inference.registry import ProviderRegistry
from olmo_eval.runners.external import ExternalEvalRunner
from olmo_eval.runners.external import runner as external_runner_module


class LegacyExternalEval(ExternalEval):
    """External eval implementing only the original provider-based hook."""

    def __init__(self) -> None:
        self.call: dict[str, Any] | None = None

    @property
    def name(self) -> str:
        return "legacy"

    @property
    def description(self) -> str:
        return "Legacy single-provider test evaluation."

    @property
    def timeout_seconds(self) -> float:
        return 1.0

    async def execute(
        self,
        provider: InferenceProvider,
        args: dict[str, Any],
        output_dir: str | None = None,
        container_runtime: str = "podman",
    ) -> ExternalEvalResult:
        self.call = {
            "provider": provider,
            "args": args,
            "output_dir": output_dir,
            "container_runtime": container_runtime,
        }
        return ExternalEvalResult(name=self.name)


class ContextAwareExternalEval(LegacyExternalEval):
    """External eval that consumes both primary and named providers."""

    @property
    def name(self) -> str:
        return "context-aware"

    async def execute_with_context(
        self,
        context: ExternalEvalContext,
        args: dict[str, Any],
        output_dir: str | None = None,
        container_runtime: str = "podman",
    ) -> ExternalEvalResult:
        judge = context.get_provider("judge")
        self.call = {
            "provider": context.provider,
            "judge": judge,
            "args": args,
            "output_dir": output_dir,
            "container_runtime": container_runtime,
        }
        return ExternalEvalResult(
            name=self.name,
            metadata={
                "primary_model": context.provider.model_name,
                "judge_model": judge.model_name,
            },
        )


class ProviderOverrideExternalEval(LegacyExternalEval):
    """External eval overriding the public provider-based compatibility hook."""

    @property
    def name(self) -> str:
        return "provider-override"

    async def execute_with_provider(
        self,
        provider: InferenceProvider | None = None,
        provider_config: ProviderConfig | None = None,
        args: dict[str, Any] | None = None,
        output_dir: str | None = None,
        container_runtime: str = "podman",
        inference_pool: Any = None,
    ) -> ExternalEvalResult:
        self.call = {
            "provider": provider,
            "provider_config": provider_config,
            "args": args,
            "output_dir": output_dir,
            "container_runtime": container_runtime,
            "inference_pool": inference_pool,
        }
        return ExternalEvalResult(name=self.name)


class OriginalProviderOverrideExternalEval(LegacyExternalEval):
    """Override with the public signature from before provider pools existed."""

    @property
    def name(self) -> str:
        return "original-provider-override"

    async def execute_with_provider(
        self,
        provider: InferenceProvider | None = None,
        provider_config: ProviderConfig | None = None,
        args: dict[str, Any] | None = None,
        output_dir: str | None = None,
        container_runtime: str = "podman",
    ) -> ExternalEvalResult:
        self.call = {
            "provider": provider,
            "provider_config": provider_config,
            "args": args,
            "output_dir": output_dir,
            "container_runtime": container_runtime,
        }
        return ExternalEvalResult(name=self.name)


def make_registry() -> ProviderRegistry:
    return ProviderRegistry.from_resolved_configs(
        {
            "judge": [
                ProviderConfig(kind=ProviderKind.MOCK, model="mock-judge"),
            ]
        }
    )


def test_context_resolves_named_provider() -> None:
    primary = MockProvider("mock-tutor")
    context = ExternalEvalContext(provider=primary, inference_pool=make_registry())

    assert context.provider is primary
    assert context.get_provider("judge").model_name == "mock-judge"


def test_context_without_inference_pool_has_clear_error() -> None:
    context = ExternalEvalContext(provider=MockProvider("mock-tutor"))

    with pytest.raises(RuntimeError, match="No auxiliary inference providers configured"):
        context.get_provider("judge")


def test_context_preserves_registry_unknown_provider_error() -> None:
    context = ExternalEvalContext(
        provider=MockProvider("mock-tutor"),
        inference_pool=make_registry(),
    )

    with pytest.raises(KeyError, match="Unknown provider 'missing'"):
        context.get_provider("missing")


def test_context_default_delegates_to_legacy_execute() -> None:
    evaluator = LegacyExternalEval()
    primary = MockProvider("mock-tutor")

    result = asyncio.run(
        evaluator.execute_with_context(
            context=ExternalEvalContext(provider=primary, inference_pool=make_registry()),
            args={"limit": 2},
            output_dir="test-output",
            container_runtime="docker",
        )
    )

    assert result.success
    assert evaluator.call == {
        "provider": primary,
        "args": {"limit": 2},
        "output_dir": "test-output",
        "container_runtime": "docker",
    }


def test_execute_with_provider_remains_compatible_with_legacy_eval() -> None:
    evaluator = LegacyExternalEval()
    primary = MockProvider("mock-tutor")

    result = asyncio.run(
        evaluator.execute_with_provider(
            provider=primary,
            args={"limit": 3},
            output_dir="legacy-output",
        )
    )

    assert result.success
    assert evaluator.call is not None
    assert evaluator.call["provider"] is primary
    assert evaluator.call["args"] == {"limit": 3}
    assert evaluator.call["output_dir"] == "legacy-output"


def test_execute_with_provider_passes_inference_pool_to_context_hook() -> None:
    evaluator = ContextAwareExternalEval()
    registry = make_registry()

    result = asyncio.run(
        evaluator.execute_with_provider(
            provider_config=ProviderConfig(kind=ProviderKind.MOCK, model="mock-tutor"),
            inference_pool=registry,
            args={"limit": 1},
        )
    )

    assert result.metadata == {
        "primary_model": "mock-tutor",
        "judge_model": "mock-judge",
    }
    assert evaluator.call is not None
    assert evaluator.call["args"] == {"limit": 1}


def test_external_runner_injects_context_without_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    evaluator = ContextAwareExternalEval()
    runner = ExternalEvalRunner(
        provider_config=ProviderConfig(kind=ProviderKind.MOCK, model="mock-tutor"),
        external_eval_names=[evaluator.name],
        eval_args={"limit": 1},
        inference_pool=make_registry(),
        metrics=None,
    )
    monkeypatch.setattr(external_runner_module, "get_external_eval", lambda _name: evaluator)
    monkeypatch.setattr(runner, "_save_results", lambda _results, _duration: None)

    results = asyncio.run(runner.run_async())

    assert results[evaluator.name].metadata == {
        "primary_model": "mock-tutor",
        "judge_model": "mock-judge",
    }
    assert evaluator.call is not None
    assert evaluator.call["args"] == {"limit": 1}


def test_external_runner_preserves_execute_with_provider_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = ProviderOverrideExternalEval()
    registry = make_registry()
    runner = ExternalEvalRunner(
        provider_config=ProviderConfig(kind=ProviderKind.MOCK, model="mock-tutor"),
        external_eval_names=[evaluator.name],
        eval_args={"limit": 2},
        inference_pool=registry,
        metrics=None,
    )
    monkeypatch.setattr(external_runner_module, "get_external_eval", lambda _name: evaluator)
    monkeypatch.setattr(runner, "_save_results", lambda _results, _duration: None)

    results = asyncio.run(runner.run_async())

    assert results[evaluator.name].success
    assert evaluator.call is not None
    assert evaluator.call["provider"].model_name == "mock-tutor"
    assert evaluator.call["inference_pool"] is registry
    assert evaluator.call["args"] == {"limit": 2}


def test_external_runner_preserves_original_provider_override_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = OriginalProviderOverrideExternalEval()
    runner = ExternalEvalRunner(
        provider_config=ProviderConfig(kind=ProviderKind.MOCK, model="mock-tutor"),
        external_eval_names=[evaluator.name],
        eval_args={"limit": 4},
        metrics=None,
    )
    monkeypatch.setattr(external_runner_module, "get_external_eval", lambda _name: evaluator)
    monkeypatch.setattr(runner, "_save_results", lambda _results, _duration: None)

    results = asyncio.run(runner.run_async())

    assert results[evaluator.name].success
    assert evaluator.call is not None
    assert evaluator.call["provider"].model_name == "mock-tutor"
    assert evaluator.call["args"] == {"limit": 4}
