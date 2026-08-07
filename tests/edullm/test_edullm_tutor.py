"""Tests for source-faithful tutor prompt construction and provider binding."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest

from olmo_eval.common.types import LMOutput, LMRequest, SamplingParams
from olmo_eval.edullm.bank import Scenario
from olmo_eval.edullm.tutor import (
    CONVERSATION_OPENER,
    TutorGenerationConfig,
    build_tutor_messages,
    generate_tutor_response,
)
from olmo_eval.inference.base import InferenceProvider


class _TutorProvider(InferenceProvider):
    def __init__(self, handler: Callable[[LMRequest], str]) -> None:
        super().__init__("fixture-tutor")
        self.handler = handler
        self.requests: list[LMRequest] = []

    def generate(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        self.requests.extend(requests)
        return [[LMOutput(text=self.handler(request))] for request in requests]

    async def agenerate(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        return self.generate(requests, sampling_params)

    def logprobs(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        raise AssertionError("tutor generation must not use continuation scoring")


def _config() -> TutorGenerationConfig:
    return TutorGenerationConfig(
        max_tokens=64,
        temperature=0.0,
        top_p=1.0,
        top_k=None,
        stop_sequences=None,
        num_samples=1,
        do_sample=False,
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("max_tokens", 1.5, "max_tokens"),
        ("max_tokens", True, "max_tokens"),
        ("top_k", 1.5, "top_k"),
        ("top_k", True, "top_k"),
        ("num_samples", True, "exactly one"),
        ("do_sample", 1, "do_sample"),
    ],
)
def test_generation_config_rejects_non_integer_and_non_boolean_values(
    field: str,
    value: object,
    message: str,
) -> None:
    values = {
        "max_tokens": 64,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": None,
        "stop_sequences": None,
        "num_samples": 1,
        "do_sample": False,
    }
    values[field] = value

    with pytest.raises(ValueError, match=message):
        TutorGenerationConfig(**values)  # type: ignore[arg-type]


def test_instruction_following_benchmark_omits_system_and_coalesces_users() -> None:
    scenario = Scenario(
        scenario_id="info-1",
        prompt="Fix this response.",
        criterion_ids=("c1",),
        use_case="feedback",
        benchmark="InFoBench",
        conversation_context=({"role": "student", "content": "My answer"},),
    )

    messages = build_tutor_messages(scenario)

    assert messages == (
        {
            "role": "user",
            "content": "My answer\n\n---\nStudent's solution:\nFix this response.",
        },
    )


def test_bridge_preserves_tutor_opening_with_user_first_transport_guard() -> None:
    scenario = Scenario(
        scenario_id="bridge-1",
        prompt="What should I do next?",
        criterion_ids=("c1",),
        benchmark="Bridge",
        conversation_context=({"role": "tutor", "content": "Earlier guidance"},),
    )

    messages = build_tutor_messages(scenario)

    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": CONVERSATION_OPENER}
    assert messages[2] == {"role": "assistant", "content": "Earlier guidance"}
    assert messages[3] == {"role": "user", "content": "What should I do next?"}


def test_biggen_uses_native_per_scenario_system_prompt() -> None:
    scenario = Scenario(
        scenario_id="biggen-1",
        prompt="Answer me.",
        criterion_ids=("c1",),
        benchmark="BiGGen",
        system_prompt="Native system contract",
    )

    messages = build_tutor_messages(scenario)

    assert messages[0] == {"role": "system", "content": "Native system contract"}
    assert messages[1] == {"role": "user", "content": "Answer me."}


def test_generation_uses_supplied_olmo_provider_and_exact_messages() -> None:
    provider = _TutorProvider(lambda _request: "Tutor output")
    scenario = Scenario(
        scenario_id="scenario-1",
        prompt="Student prompt",
        criterion_ids=("c1",),
        benchmark="InFoBench",
    )

    response = asyncio.run(generate_tutor_response(provider, scenario, _config()))

    assert response.output.text == "Tutor output"
    assert provider.requests == [response.request]
    assert response.request.messages == ({"role": "user", "content": "Student prompt"},)
