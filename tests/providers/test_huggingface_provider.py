"""Unit tests for HuggingFaceProvider generate-kwargs construction and logprob scoring."""

from types import SimpleNamespace
from typing import Any

import pytest

from olmo_eval.common.types import LMRequest, RequestType, SamplingParams
from olmo_eval.inference.providers.huggingface import HuggingFaceProvider


@pytest.fixture
def provider() -> HuggingFaceProvider:
    instance = HuggingFaceProvider.__new__(HuggingFaceProvider)
    instance.model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=2048))
    return instance


def test_finite_max_tokens_passes_through(provider: HuggingFaceProvider) -> None:
    kwargs = provider._build_generate_kwargs(SamplingParams(max_tokens=512), prompt_len=100)
    assert kwargs["max_new_tokens"] == 512


def test_uncapped_reserves_room_after_prompt(provider: HuggingFaceProvider) -> None:
    kwargs = provider._build_generate_kwargs(SamplingParams(max_tokens=None), prompt_len=2000)
    assert kwargs["max_new_tokens"] == 2048 - 2000


def test_uncapped_with_no_prompt_uses_full_context(provider: HuggingFaceProvider) -> None:
    kwargs = provider._build_generate_kwargs(SamplingParams(max_tokens=None))
    assert kwargs["max_new_tokens"] == 2048


def test_uncapped_floors_at_one_when_prompt_exceeds_context(provider: HuggingFaceProvider) -> None:
    kwargs = provider._build_generate_kwargs(SamplingParams(max_tokens=None), prompt_len=5000)
    assert kwargs["max_new_tokens"] == 1


_LABEL_TOKEN_IDS = {" A": 3, " B": 4, " C": 5, " D": 6}


class LabelTokenizer:
    """Character-level tokenizer that gives ``" A"``-style labels a single id.

    Labelled multiple choice is only a single-token continuation because real
    tokenizers spell a space-prefixed capital letter as one token, so a fake that
    split it per character could not exercise the prefix sharing at all.
    """

    pad_token_id = 0
    eos_token_id = 2
    bos_token_id = 1
    add_bos_token = False
    vocab_size = 64

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        token_ids: list[int] = []
        index = 0
        while index < len(text):
            label = text[index : index + 2]
            if label in _LABEL_TOKEN_IDS:
                token_ids.append(_LABEL_TOKEN_IDS[label])
                index += 2
                continue
            token_ids.append(ord(text[index]) % 29 + 20)
            index += 1
        if add_special_tokens:
            return [self.bos_token_id, *token_ids]
        return token_ids

    def decode(self, token_ids: int | list[int], skip_special_tokens: bool = True) -> str:
        ids = [token_ids] if isinstance(token_ids, int) else list(token_ids)
        return "".join(f"<{token_id}>" for token_id in ids)


class CountingLogitsModel:
    """Model whose logits at a position are fixed by the prefix, as a causal LM's are."""

    def __init__(self) -> None:
        self.forward_calls = 0

    def __call__(self, input_ids: Any) -> Any:
        import torch

        self.forward_calls += 1
        rows = input_ids.tolist()
        logits = []
        for row in rows:
            state = 1
            row_logits = []
            for token_id in row:
                state = (state * 1_103_515_245 + token_id + 12_345) % 2_147_483_647
                row_logits.append(
                    [
                        ((state ^ ((candidate + 1) * 2_654_435_761)) % 1_000_003) / 1_000_003 * 12.0
                        - 6.0
                        for candidate in range(LabelTokenizer.vocab_size)
                    ]
                )
            logits.append(row_logits)
        return SimpleNamespace(logits=torch.tensor(logits, dtype=torch.float32))


@pytest.fixture
def logprob_provider() -> tuple[HuggingFaceProvider, CountingLogitsModel]:
    pytest.importorskip("torch")

    instance = HuggingFaceProvider.__new__(HuggingFaceProvider)
    model = CountingLogitsModel()
    instance.model = model
    instance.tokenizer = LabelTokenizer()
    instance.device = "cpu"
    instance.share_logprob_forwards = True
    return instance, model


def _mc_request(question: str, continuations: tuple[str, ...]) -> LMRequest:
    return LMRequest(
        request_type=RequestType.LOGLIKELIHOOD,
        prompt=f"Question: {question}\n A. the sun\n B. the moon\nAnswer:",
        continuations=continuations,
    )


def test_shared_prefix_scores_match_per_continuation_forwards(
    logprob_provider: tuple[HuggingFaceProvider, CountingLogitsModel],
) -> None:
    provider, model = logprob_provider
    requests = [
        _mc_request("Which of these keeps the earth warm?", (" A", " B")),
        _mc_request("Where would you keep a spare tire?", (" A", " B", " C", " D")),
    ]

    provider.share_logprob_forwards = False
    unshared = provider.logprobs(requests)
    unshared_calls = model.forward_calls

    provider.share_logprob_forwards = True
    shared = provider.logprobs(requests)

    assert unshared_calls == 6
    assert model.forward_calls - unshared_calls == 2

    for shared_outputs, unshared_outputs in zip(shared, unshared, strict=True):
        shared_scores = [output.metadata["total_logprob"] for output in shared_outputs]
        unshared_scores = [output.metadata["total_logprob"] for output in unshared_outputs]
        assert shared_scores == pytest.approx(unshared_scores, abs=1e-12)
        assert shared_scores.index(max(shared_scores)) == unshared_scores.index(
            max(unshared_scores)
        )
        assert [output.text for output in shared_outputs] == [
            output.text for output in unshared_outputs
        ]


def test_multi_token_continuations_keep_their_own_forwards(
    logprob_provider: tuple[HuggingFaceProvider, CountingLogitsModel],
) -> None:
    provider, model = logprob_provider
    request = _mc_request("Which of these keeps the earth warm?", (" the sun", " the moon"))

    provider.share_logprob_forwards = False
    unshared = provider.logprobs([request])
    unshared_calls = model.forward_calls

    provider.share_logprob_forwards = True
    shared = provider.logprobs([request])

    assert unshared_calls == 2
    assert model.forward_calls - unshared_calls == 2
    assert [output.metadata["total_logprob"] for output in shared[0]] == pytest.approx(
        [output.metadata["total_logprob"] for output in unshared[0]], abs=1e-12
    )
