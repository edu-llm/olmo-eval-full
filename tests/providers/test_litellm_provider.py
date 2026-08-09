"""Focused tests for LiteLLM generated-token logprob conversion."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from olmo_eval.common.types import LMRequest, RequestType, SamplingParams
from olmo_eval.inference.providers.litellm import LiteLLMProvider


@pytest.mark.anyio
async def test_generate_requests_and_preserves_configured_top_logprobs() -> None:
    token_logprob = SimpleNamespace(
        token="P",
        logprob=-0.2,
        bytes=[80],
        top_logprobs=[
            SimpleNamespace(token="P", logprob=-0.2, bytes=[80]),
            SimpleNamespace(token="F", logprob=-0.7, bytes=[70]),
        ],
    )
    choice = SimpleNamespace(
        message=SimpleNamespace(content="P"),
        logprobs=SimpleNamespace(content=[token_logprob]),
    )
    litellm = SimpleNamespace(acompletion=AsyncMock(return_value=SimpleNamespace(choices=[choice])))
    provider = LiteLLMProvider.__new__(LiteLLMProvider)
    provider.model_name = "openai-compatible-model"
    provider.api_base = "http://localhost:8000/v1"
    provider.api_kwargs = {}
    provider._litellm = litellm

    outputs = await provider._generate_single_impl(
        LMRequest(request_type=RequestType.CHAT, prompt="Judge"),
        SamplingParams(max_tokens=1, logprobs=20),
    )

    assert litellm.acompletion.call_args.kwargs["top_logprobs"] == 20
    assert outputs[0].logprobs == [
        {
            "token": "P",
            "logprob": -0.2,
            "bytes": [80],
            "top_logprobs": [
                {"token": "P", "logprob": -0.2, "bytes": [80]},
                {"token": "F", "logprob": -0.7, "bytes": [70]},
            ],
        }
    ]


@pytest.mark.anyio
async def test_generate_keeps_one_alternative_as_backward_compatible_default() -> None:
    choice = SimpleNamespace(message=SimpleNamespace(content="ok"), logprobs=None)
    litellm = SimpleNamespace(acompletion=AsyncMock(return_value=SimpleNamespace(choices=[choice])))
    provider = LiteLLMProvider.__new__(LiteLLMProvider)
    provider.model_name = "openai-compatible-model"
    provider.api_base = None
    provider.api_kwargs = {}
    provider._litellm = litellm

    await provider._generate_single_impl(
        LMRequest(request_type=RequestType.CHAT, prompt="Hello"),
        SamplingParams(max_tokens=1),
    )

    assert litellm.acompletion.call_args.kwargs["top_logprobs"] == 1


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("lm_request", "params", "feature"),
    [
        (
            LMRequest(request_type=RequestType.CHAT, prompt="Judge"),
            SamplingParams(max_tokens=1, logprobs=2, logprob_token_ids=(10, 11)),
            "logprob_token_ids",
        ),
        (
            LMRequest(request_type=RequestType.CHAT, prompt="Judge"),
            SamplingParams(max_tokens=1, structured_output_regex="[PF]"),
            "structured_output_regex",
        ),
        (
            LMRequest(request_type=RequestType.CHAT, prompt="Judge"),
            SamplingParams(
                max_tokens=1,
                structured_output_json_schema={"type": "object"},
            ),
            "structured_output_json_schema",
        ),
        (
            LMRequest(
                request_type=RequestType.CHAT,
                prompt="Judge",
                chat_template_kwargs={"enable_thinking": False},
            ),
            SamplingParams(max_tokens=1),
            "chat_template_kwargs",
        ),
    ],
)
async def test_generate_rejects_constraints_litellm_cannot_guarantee(
    lm_request: LMRequest,
    params: SamplingParams,
    feature: str,
) -> None:
    litellm = SimpleNamespace(acompletion=AsyncMock())
    provider = LiteLLMProvider.__new__(LiteLLMProvider)
    provider.model_name = "api-model"
    provider.api_base = None
    provider.api_kwargs = {}
    provider._litellm = litellm

    with pytest.raises(NotImplementedError, match=feature):
        await provider._generate_single_impl(lm_request, params)

    litellm.acompletion.assert_not_called()
