"""Unit tests for the inline VLLMProvider."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from olmo_eval.common.types import LMRequest, RequestType, SamplingParams
from olmo_eval.inference.providers.vllm import VLLMProvider


class FakeVllmModule(ModuleType):
    SamplingParams: object
    StructuredOutputsParams: object


class FakeSamplingModule(ModuleType):
    StructuredOutputsParams: object


def test_sampling_params_rejects_ambiguous_or_inconsistent_constraints() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        SamplingParams(
            structured_output_regex="[PF]",
            structured_output_json_schema={"type": "object"},
        )

    with pytest.raises(ValueError, match=r"logprobs must equal len\(logprob_token_ids\)"):
        SamplingParams(logprobs=20, logprob_token_ids=(10, 11))

    with pytest.raises(ValueError, match="seed must be"):
        SamplingParams(seed=-1)


def test_build_sampling_params_passes_none_max_tokens_through(monkeypatch) -> None:
    # vLLM treats max_tokens=None as "generate to the context limit", so the
    # provider must forward None unchanged rather than crashing or coercing it.
    fake_vllm = FakeVllmModule("vllm")
    fake_vllm.SamplingParams = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    provider = VLLMProvider.__new__(VLLMProvider)
    built = provider._build_sampling_params(SamplingParams(max_tokens=None, do_sample=False))

    assert built["max_tokens"] is None


def test_build_sampling_params_forwards_requested_top_logprobs(monkeypatch) -> None:
    fake_vllm = FakeVllmModule("vllm")
    fake_vllm.SamplingParams = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    provider = VLLMProvider.__new__(VLLMProvider)
    built = provider._build_sampling_params(SamplingParams(max_tokens=1, logprobs=20))

    assert built["logprobs"] == 20


class FakeTokenizer:
    def __init__(self) -> None:
        self.template_calls: list[dict[str, object]] = []
        self.encode_calls: list[dict[str, object]] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        **kwargs: object,
    ) -> str:
        call: dict[str, object] = {
            "messages": messages,
            "tokenize": tokenize,
            "add_generation_prompt": add_generation_prompt,
        }
        call.update(kwargs)
        self.template_calls.append(call)
        rendered_messages = "|".join(f"{m['role']}:{m['content']}" for m in messages)
        return f"<chat>{rendered_messages}<assistant>"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        self.encode_calls.append(
            {
                "text": text,
                "add_special_tokens": add_special_tokens,
            }
        )
        return [ord(char) for char in text]


class FakeLLM:
    def __init__(self, tokenizer: FakeTokenizer) -> None:
        self.tokenizer = tokenizer
        self.generate_calls: list[dict[str, object]] = []
        self.completion: object | None = None

    def get_tokenizer(self) -> FakeTokenizer:
        return self.tokenizer

    def generate(
        self,
        prompts: list[str] | list[dict[str, list[int]]],
        sampling_params: object,
        use_tqdm: bool = False,
    ) -> list[object]:
        self.generate_calls.append(
            {
                "prompts": prompts,
                "sampling_params": sampling_params,
                "use_tqdm": use_tqdm,
            }
        )
        completion = self.completion or SimpleNamespace(text="ok", logprobs=None)
        return [SimpleNamespace(outputs=[completion]) for _ in prompts]


@pytest.fixture
def fake_provider() -> tuple[VLLMProvider, FakeLLM, FakeTokenizer]:
    tokenizer = FakeTokenizer()
    llm = FakeLLM(tokenizer)
    provider = VLLMProvider.__new__(VLLMProvider)
    provider.model_name = "test-model"
    provider.llm = llm
    provider._add_bos_token = None
    provider._build_sampling_params = lambda params: "fake-sampling-params"
    return provider, llm, tokenizer


def test_generate_formats_chat_messages_with_template(
    fake_provider: tuple[VLLMProvider, FakeLLM, FakeTokenizer],
) -> None:
    provider, llm, tokenizer = fake_provider
    request = LMRequest(
        request_type=RequestType.CHAT,
        messages=({"role": "user", "content": "Hello"},),
    )

    outputs = provider.generate([request], SamplingParams(max_tokens=1))

    assert outputs[0][0].text == "ok"
    assert tokenizer.template_calls == [
        {
            "messages": [{"role": "user", "content": "Hello"}],
            "tokenize": False,
            "add_generation_prompt": True,
        }
    ]
    assert llm.generate_calls[0]["prompts"] == ["<chat>user:Hello<assistant>"]


def test_generate_merges_provider_and_request_chat_template_kwargs(
    fake_provider: tuple[VLLMProvider, FakeLLM, FakeTokenizer],
) -> None:
    provider, _, tokenizer = fake_provider
    provider.chat_template_kwargs = {"enable_thinking": True, "provider_only": "kept"}
    request = LMRequest(
        request_type=RequestType.CHAT,
        messages=({"role": "user", "content": "Hello"},),
        chat_template_kwargs={"enable_thinking": False},
    )

    provider.generate([request], SamplingParams(max_tokens=1))

    assert tokenizer.template_calls[0]["enable_thinking"] is False
    assert tokenizer.template_calls[0]["provider_only"] == "kept"


def test_generate_tokenizes_formatted_chat_prompt_when_bos_disabled(
    fake_provider: tuple[VLLMProvider, FakeLLM, FakeTokenizer],
) -> None:
    provider, llm, tokenizer = fake_provider
    provider._add_bos_token = False
    request = LMRequest(
        request_type=RequestType.CHAT,
        messages=({"role": "user", "content": "Hello"},),
    )

    provider.generate([request], SamplingParams(max_tokens=1))

    assert tokenizer.encode_calls == [
        {
            "text": "<chat>user:Hello<assistant>",
            "add_special_tokens": False,
        }
    ]
    assert llm.generate_calls[0]["prompts"] == [
        {"prompt_token_ids": [ord(char) for char in "<chat>user:Hello<assistant>"]}
    ]


def test_generate_keeps_completion_prompt_unchanged(
    fake_provider: tuple[VLLMProvider, FakeLLM, FakeTokenizer],
) -> None:
    provider, llm, tokenizer = fake_provider
    request = LMRequest(request_type=RequestType.COMPLETION, prompt="Complete me")

    provider.generate([request], SamplingParams(max_tokens=1))

    assert tokenizer.template_calls == []
    assert llm.generate_calls[0]["prompts"] == ["Complete me"]


def test_generate_preserves_sampled_token_and_top_logprob_alternatives(
    fake_provider: tuple[VLLMProvider, FakeLLM, FakeTokenizer],
) -> None:
    provider, llm, _ = fake_provider
    llm.completion = SimpleNamespace(
        text="P",
        token_ids=[10],
        # Deliberately put F first: the emitted token must come from token_ids,
        # not from dictionary insertion order.
        logprobs=[
            {
                11: SimpleNamespace(decoded_token="F", logprob=-0.7),
                10: SimpleNamespace(decoded_token="P", logprob=-0.2),
            }
        ],
    )

    outputs = provider.generate(
        [LMRequest(request_type=RequestType.COMPLETION, prompt="Judge")],
        SamplingParams(max_tokens=1, logprobs=20),
    )

    logprobs = outputs[0][0].logprobs
    assert logprobs is not None
    assert logprobs[0]["token"] == "P"
    assert logprobs[0]["logprob"] == pytest.approx(-0.2)
    assert logprobs[0]["top_logprobs"] == [
        {"token": "F", "logprob": -0.7, "bytes": [70], "token_id": 11},
        {"token": "P", "logprob": -0.2, "bytes": [80], "token_id": 10},
    ]
    assert logprobs[0]["token_id"] == 10
    assert outputs[0][0].token_ids == (10,)


def test_build_sampling_params_forwards_frozen_seed(monkeypatch) -> None:
    fake_vllm = FakeVllmModule("vllm")
    fake_vllm.SamplingParams = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    provider = VLLMProvider.__new__(VLLMProvider)
    built = provider._build_sampling_params(SamplingParams(max_tokens=1, seed=42))

    assert built["seed"] == 42


def test_build_sampling_params_wires_explicit_ids_and_regex(monkeypatch) -> None:
    class FakeStructuredOutputsParams:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class FakeSamplingParams:
        def __init__(
            self,
            *,
            logprob_token_ids: list[int] | None = None,
            structured_outputs: object | None = None,
            **kwargs: object,
        ) -> None:
            self.logprob_token_ids = logprob_token_ids
            self.structured_outputs = structured_outputs
            self.kwargs = kwargs

    fake_vllm = FakeVllmModule("vllm")
    fake_vllm.SamplingParams = FakeSamplingParams
    fake_vllm.StructuredOutputsParams = FakeStructuredOutputsParams
    fake_sampling_module = FakeSamplingModule("vllm.sampling_params")
    fake_sampling_module.StructuredOutputsParams = FakeStructuredOutputsParams
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", fake_sampling_module)

    provider = VLLMProvider.__new__(VLLMProvider)
    built = provider._build_sampling_params(
        SamplingParams(
            max_tokens=1,
            logprobs=2,
            logprob_token_ids=(10, 11),
            structured_output_regex="[PF]",
        )
    )

    assert built.logprob_token_ids == [10, 11]
    assert built.kwargs["logprobs"] == 2
    assert built.structured_outputs.kwargs == {"regex": "[PF]"}


def test_build_sampling_params_wires_json_schema(monkeypatch) -> None:
    class FakeStructuredOutputsParams:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class FakeSamplingParams:
        def __init__(
            self,
            *,
            structured_outputs: object | None = None,
            **kwargs: object,
        ) -> None:
            self.structured_outputs = structured_outputs
            self.kwargs = kwargs

    fake_vllm = FakeVllmModule("vllm")
    fake_vllm.SamplingParams = FakeSamplingParams
    fake_sampling_module = FakeSamplingModule("vllm.sampling_params")
    fake_sampling_module.StructuredOutputsParams = FakeStructuredOutputsParams
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", fake_sampling_module)

    provider = VLLMProvider.__new__(VLLMProvider)
    built = provider._build_sampling_params(
        SamplingParams(
            max_tokens=1,
            structured_output_json_schema={"type": "object"},
        )
    )

    assert built.structured_outputs.kwargs == {"json": {"type": "object"}}


def test_build_sampling_params_rejects_explicit_ids_on_unsupported_vllm(monkeypatch) -> None:
    fake_vllm = FakeVllmModule("vllm")
    fake_vllm.SamplingParams = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    provider = VLLMProvider.__new__(VLLMProvider)
    with pytest.raises(RuntimeError, match="requires vLLM >= 0.26.0"):
        provider._build_sampling_params(
            SamplingParams(max_tokens=1, logprobs=2, logprob_token_ids=(10, 11))
        )
