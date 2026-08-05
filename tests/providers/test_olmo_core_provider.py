"""Hotspot tests for the OLMo-core provider."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import olmo_eval.inference.providers.olmo_core_utils as olmo_core_utils
from olmo_eval.common.metrics import LogprobPerCharMCAccuracyMetric
from olmo_eval.common.types import (
    Instance,
    LMOutput,
    LMRequest,
    RequestType,
    Response,
    SamplingParams,
)
from olmo_eval.inference.providers.olmo_core import OlmoCoreProvider
from olmo_eval.inference.providers.olmo_core_utils import (
    _TRANSFORMERS_UNSET_MODEL_MAX_LENGTH,
    LogprobInput,
    _plan_logprob_forwards,
)


class FakeTokenizerConfig:
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SimpleNamespace:
        return SimpleNamespace(
            identifier=data.get("identifier"),
            pad_token_id=data.get("pad_token_id"),
            eos_token_id=data.get("eos_token_id"),
        )


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    bos_token_id = 1
    model_max_length = 32

    def __init__(self) -> None:
        self.add_bos_token = False
        self.encode_calls: list[dict[str, Any]] = []
        self.decode_calls: list[list[int]] = []
        self.vocab = {
            "": [],
            "Prompt": [10, 11],
            "Prompt!": [10, 11, 6],
            "Other": [12],
            "!": [6],
            " !": [13, 6],
            " STOP": [4],
            "STOP": [8, 9],
        }
        self.id_to_text = {
            0: "<pad>",
            1: "<bos>",
            2: "<eos>",
            4: " STOP",
            5: "hello",
            6: "!",
            7: "x",
            8: "ST",
            9: "OP",
            10: "P",
            11: "rompt",
            12: "Other",
            13: " ",
        }

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        self.encode_calls.append(
            {
                "text": text,
                "add_special_tokens": add_special_tokens,
            }
        )
        if text in self.vocab:
            token_ids = self.vocab[text]
        else:
            token_ids = [ord(char) % 13 + 3 for char in text]
        if add_special_tokens:
            return [self.bos_token_id, *token_ids]
        return token_ids

    def decode(self, token_ids: int | list[int], skip_special_tokens: bool = True) -> str:
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        self.decode_calls.append(list(token_ids))
        pieces = []
        for token_id in token_ids:
            if skip_special_tokens and token_id in {self.pad_token_id, self.eos_token_id}:
                continue
            pieces.append(self.id_to_text.get(token_id, str(token_id)))
        return "".join(pieces)


class FakeAutoTokenizer:
    @classmethod
    def from_pretrained(cls, tokenizer_path: str, **kwargs: Any) -> FakeTokenizer:
        del cls, kwargs
        assert tokenizer_path == "fake-tokenizer"
        tokenizer = FakeTokenizer()
        tokenizer.model_max_length = _TRANSFORMERS_UNSET_MODEL_MAX_LENGTH
        return tokenizer


class MissingSpecialTokenAutoTokenizer:
    @classmethod
    def from_pretrained(cls, tokenizer_path: str, **kwargs: Any) -> FakeTokenizer:
        del cls
        tokenizer = FakeAutoTokenizer.from_pretrained(tokenizer_path, **kwargs)
        tokenizer.pad_token_id = None
        tokenizer.eos_token_id = None
        return tokenizer


class FakeTensorRows:
    def __init__(self, rows: list[list[int]] | list[list[float]]) -> None:
        self._rows = rows
        self.shape = (len(rows), len(rows[0]) if rows else 0)

    def __getitem__(self, idx: int) -> Any:
        return SimpleNamespace(tolist=lambda: self._rows[idx])

    def tolist(self) -> list[list[int]] | list[list[float]]:
        return self._rows


class FakeGenerationModule:
    def __init__(self) -> None:
        self.generate_calls: list[dict[str, Any]] = []
        self.checkpoint_kwargs: dict[str, Any] = {}
        self.prepare_calls: list[tuple[int, int]] = []
        self.cache_allocated = False
        self.free_calls = 0

    @classmethod
    def from_checkpoint(cls, **kwargs: Any) -> FakeGenerationModule:
        module = cls()
        module.checkpoint_kwargs = kwargs
        return module

    def generate_batch(self, **kwargs: Any):
        self.generate_calls.append(kwargs)
        batch_size = kwargs["input_ids"].shape[0]
        if kwargs["use_cache"]:
            self.prepare_inference_cache(batch_size, kwargs["max_length"])

        completion_rows = [[5, 4, 0] if idx % 2 == 0 else [5, 6, 0] for idx in range(batch_size)]
        generated_rows = (
            completion_rows
            if kwargs["completions_only"]
            else [
                [*kwargs["input_ids"][idx].tolist(), *completion_rows[idx]]
                for idx in range(batch_size)
            ]
        )
        logprob_rows = [
            [-0.1, -0.2, -9.0] if idx % 2 == 0 else [-0.3, -0.4, -9.0] for idx in range(batch_size)
        ]
        return FakeTensorRows(generated_rows), None, FakeTensorRows(logprob_rows)

    def prepare_inference_cache(self, batch_size: int, max_seq_len: int) -> None:
        self.prepare_calls.append((batch_size, max_seq_len))
        self.cache_allocated = True

    def free_inference_cache(self) -> None:
        self.cache_allocated = False
        self.free_calls += 1


def _write_raw_checkpoint(
    checkpoint_dir: Path,
    *,
    model: dict[str, Any] | None = None,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "config.json").write_text(
        json.dumps(
            {
                "model": model if model is not None else {"d_model": 8},
                "dataset": {
                    "tokenizer": {
                        "identifier": "fake-tokenizer",
                        "vocab_size": 16,
                        "pad_token_id": 0,
                        "eos_token_id": 2,
                    }
                },
            }
        )
    )
    (checkpoint_dir / ".metadata").write_text("fake")


def _metadata_reader(path: str | Path) -> SimpleNamespace:
    path_obj = path if isinstance(path, Path) else Path(path)
    if (path_obj / ".metadata").exists():
        return SimpleNamespace(state_dict_metadata={"model.transformer.wte.weight": object()})
    raise FileNotFoundError(path)


def _fake_olmo_core_imports(
    *,
    cuda_available: bool = False,
    auto_tokenizer: type[Any] = FakeAutoTokenizer,
) -> SimpleNamespace:
    return SimpleNamespace(
        AutoTokenizer=auto_tokenizer,
        AttentionBackendName=str,
        GenerationConfig=lambda **kwargs: SimpleNamespace(**kwargs),
        TokenizerConfig=FakeTokenizerConfig,
        TransformerGenerationModule=FakeGenerationModule,
        cached_path=None,
        get_checkpoint_metadata=_metadata_reader,
        torch=SimpleNamespace(
            cuda=SimpleNamespace(
                get_device_capability=lambda: (9, 0),
                is_available=lambda: cuda_available,
            ),
            device=lambda device: device,
        ),
    )


@pytest.fixture
def fake_provider() -> tuple[OlmoCoreProvider, FakeGenerationModule, FakeTokenizer]:
    provider = OlmoCoreProvider.__new__(OlmoCoreProvider)
    tokenizer = FakeTokenizer()
    module = FakeGenerationModule()
    provider.model_name = "fake-model"
    provider.tokenizer = tokenizer
    provider.generation_module = module
    provider.pad_token_id = tokenizer.pad_token_id
    provider.eos_token_id = tokenizer.eos_token_id
    provider.use_cache = True
    provider.add_bos_token = False
    provider.share_logprob_forwards = True
    provider.batch_size = None
    provider.chat_template = None
    provider.max_length = 32

    def left_pad(sequences: list[list[int]]) -> tuple[FakeTensorRows, FakeTensorRows]:
        max_len = max(max((len(seq) for seq in sequences), default=0), 1)
        rows = []
        masks = []
        for seq in sequences:
            pad_len = max_len - len(seq)
            rows.append([tokenizer.pad_token_id] * pad_len + seq)
            masks.append([0] * pad_len + [1] * len(seq))
        return FakeTensorRows(rows), FakeTensorRows(masks)

    provider._left_pad = left_pad
    return provider, module, tokenizer


def test_provider_loads_checkpoint_with_olmes_defaults(tmp_path, monkeypatch) -> None:
    checkpoint_dir = tmp_path / "step1000"
    _write_raw_checkpoint(checkpoint_dir, model={"d_model": 8, "max_sequence_length": 4096})
    monkeypatch.setattr(
        olmo_core_utils,
        "_import_olmo_core",
        lambda: _fake_olmo_core_imports(auto_tokenizer=MissingSpecialTokenAutoTokenizer),
    )

    provider = OlmoCoreProvider(str(checkpoint_dir))

    checkpoint_kwargs = provider.generation_module.checkpoint_kwargs
    generation_config = checkpoint_kwargs["generation_config"]
    assert provider.max_length == 4096
    assert provider.add_bos_token is False
    assert provider.pad_token_id == 0
    assert provider.eos_token_id == 2
    assert provider.tokenizer.pad_token_id == 0
    assert provider.tokenizer.eos_token_id == 2
    assert checkpoint_kwargs["dtype"] == "bfloat16"
    assert "attention_backend" not in checkpoint_kwargs
    assert generation_config.pad_token_id == 0
    assert generation_config.eos_token_id == 2
    assert generation_config.use_cache is True


def test_provider_passes_explicit_attention_backend(tmp_path, monkeypatch) -> None:
    checkpoint_dir = tmp_path / "step1000"
    _write_raw_checkpoint(checkpoint_dir, model={"d_model": 8, "max_sequence_length": 4096})
    monkeypatch.setattr(
        olmo_core_utils,
        "_import_olmo_core",
        lambda: _fake_olmo_core_imports(cuda_available=True),
    )

    provider = OlmoCoreProvider(
        str(checkpoint_dir),
        attention_backend="torch",
    )

    assert provider.generation_module.checkpoint_kwargs["attention_backend"] == "torch"


def test_generate_uses_olmes_batch_contract(
    fake_provider: tuple[OlmoCoreProvider, FakeGenerationModule, FakeTokenizer],
) -> None:
    provider, module, _ = fake_provider

    outputs = provider.generate(
        [
            LMRequest(request_type=RequestType.COMPLETION, prompt="Prompt"),
            LMRequest(request_type=RequestType.COMPLETION, prompt="Other"),
        ],
        SamplingParams(
            max_tokens=3,
            num_samples=2,
            temperature=0.7,
            top_p=None,
            top_k=None,
            stop_sequences=(" STOP", "!"),
        ),
    )

    call = module.generate_calls[0]
    assert call["input_ids"].tolist() == [
        [10, 11],
        [10, 11],
        [0, 12],
        [0, 12],
    ]
    assert call["attention_mask"].tolist() == [
        [1, 1],
        [1, 1],
        [0, 1],
        [0, 1],
    ]
    assert call["return_logprobs"] is True
    assert call["completions_only"] is False
    assert call["max_length"] == 5
    assert "max_new_tokens" not in call
    assert "stop_token_ids" not in call
    assert module.prepare_calls == [(4, 5)]
    assert [output.text for output in outputs[0]] == ["hello", "hello"]
    assert outputs[0][0].metadata["sum_logits"] == pytest.approx(-0.3)
    assert outputs[1][1].metadata["num_tokens"] == 2
    assert module.cache_allocated is False
    assert module.free_calls == 1


def test_generate_uncapped_fills_context(
    fake_provider: tuple[OlmoCoreProvider, FakeGenerationModule, FakeTokenizer],
) -> None:
    provider, module, _ = fake_provider
    provider.max_length = 32

    provider.generate(
        [LMRequest(request_type=RequestType.COMPLETION, prompt="Prompt")],
        SamplingParams(max_tokens=None),
    )

    call = module.generate_calls[0]
    # The 2-token prompt isn't truncated, and generation fills the rest of the budget.
    assert call["input_ids"].tolist() == [[10, 11]]
    assert call["max_length"] == 32


def test_generate_left_truncates_to_leave_completion_room(
    fake_provider: tuple[OlmoCoreProvider, FakeGenerationModule, FakeTokenizer],
) -> None:
    provider, module, _ = fake_provider
    provider.max_length = 4

    provider.generate(
        [LMRequest(request_type=RequestType.COMPLETION, prompt="Prompt")],
        SamplingParams(max_tokens=3),
    )

    call = module.generate_calls[0]
    assert call["input_ids"].tolist() == [[11]]
    assert call["attention_mask"].tolist() == [[1]]
    assert call["max_length"] == 4

    module.generate_calls.clear()
    provider.max_length = 3
    with pytest.raises(ValueError, match=r"max_tokens \(3\) is greater than or equal"):
        provider.generate(
            [LMRequest(request_type=RequestType.COMPLETION, prompt="Prompt")],
            SamplingParams(max_tokens=3),
        )
    assert module.generate_calls == []


def test_generation_encoding_uses_provider_bos_flag(
    fake_provider: tuple[OlmoCoreProvider, FakeGenerationModule, FakeTokenizer],
) -> None:
    provider, _, tokenizer = fake_provider
    tokenizer.add_bos_token = True

    assert provider._encode_prompt("Prompt") == [10, 11]
    assert tokenizer.encode_calls[-1] == {
        "text": "Prompt",
        "add_special_tokens": False,
    }

    provider.add_bos_token = True
    assert provider._encode_prompt("Prompt") == [1, 10, 11]
    assert tokenizer.encode_calls[-1] == {
        "text": "Prompt",
        "add_special_tokens": False,
    }


def test_logprob_encoding_uses_provider_bos_flag(
    fake_provider: tuple[OlmoCoreProvider, FakeGenerationModule, FakeTokenizer],
) -> None:
    provider, _, tokenizer = fake_provider
    tokenizer.add_bos_token = True

    rows = provider._logprob_inputs_for_request(
        LMRequest(
            request_type=RequestType.LOGLIKELIHOOD,
            prompt="Prompt",
            continuations=("!",),
        )
    )
    assert rows[0].input_ids == [10, 11]
    assert rows[0].continuation_token_ids == [6]

    provider.add_bos_token = True
    rows = provider._logprob_inputs_for_request(
        LMRequest(
            request_type=RequestType.LOGLIKELIHOOD,
            prompt="Prompt",
            continuations=("!",),
        )
    )
    assert rows[0].input_ids == [1, 10, 11]
    assert rows[0].continuation_token_ids == [6]


def test_logprob_encoding_adds_prefix_when_context_tokenizes_empty(
    fake_provider: tuple[OlmoCoreProvider, FakeGenerationModule, FakeTokenizer],
) -> None:
    provider, _, _ = fake_provider

    rows = provider._logprob_inputs_for_request(
        LMRequest(
            request_type=RequestType.LOGLIKELIHOOD,
            prompt=" ",
            continuations=("!",),
        )
    )

    assert rows[0].input_ids == [1, 13]
    assert rows[0].continuation_token_ids == [13, 6]
    assert rows[0].input_length == 2
    assert rows[0].num_tokens_all == 3


def test_stop_text_postprocessing_matches_olmes(
    fake_provider: tuple[OlmoCoreProvider, FakeGenerationModule, FakeTokenizer],
) -> None:
    provider, _, tokenizer = fake_provider

    token_ids = [7, 7, 8, 9, 7]
    token_logprobs = [-0.01] * len(token_ids)
    normalized_ids, normalized_logprobs, text = provider._normalize_generation_output(
        token_ids,
        token_logprobs,
        ("STOP",),
    )
    assert normalized_ids == token_ids
    assert normalized_logprobs == token_logprobs
    assert text == "xx"

    tokenizer.id_to_text[13] = tokenizer.decode(
        [tokenizer.eos_token_id],
        skip_special_tokens=False,
    )
    normalized_ids, normalized_logprobs, text = provider._normalize_generation_output(
        [5, 13, 7],
        [-0.1, -0.2, -0.3],
        provider._stop_sequences_with_eos(None),
    )
    assert normalized_ids == [5, 13, 7]
    assert normalized_logprobs == [-0.1, -0.2, -0.3]
    assert text == "hello"


_LABEL_TOKEN_IDS = {" A": 3, " B": 4, " C": 5, " D": 6, " E": 7}
_LABEL_BY_TOKEN_ID = {token_id: label for label, token_id in _LABEL_TOKEN_IDS.items()}


class LabelTokenizer:
    """Character-level tokenizer that gives ``" A"``-style labels a single id.

    Real tokenizers spell a space-prefixed capital letter as one token, and that
    is the whole reason a labelled multiple-choice continuation is one token
    long. ``FakeTokenizer`` above splits per character, so these tests need a
    tokenizer that reproduces the property being exploited. Encoding stays a
    left-to-right scan, so it remains prefix-consistent the way a real BPE
    tokenizer is over the concatenations ``encode_context_and_continuation`` does.
    """

    pad_token_id = 0
    eos_token_id = 2
    bos_token_id = 1
    model_max_length = 512
    vocab_size = 64

    def __init__(self) -> None:
        self.add_bos_token = False

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
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        pieces = []
        for token_id in token_ids:
            if skip_special_tokens and token_id in {self.pad_token_id, self.eos_token_id}:
                continue
            pieces.append(_LABEL_BY_TOKEN_ID.get(token_id, f"<{token_id}>"))
        return "".join(pieces)


class CountingLogitsModule:
    """Generation module whose logits at a position are fixed by the prefix.

    That is the property of a causal LM that makes sharing a forward pass sound,
    so a fake that did not have it could not tell a correct sharing decision from
    an incorrect one. ``rows_forwarded`` counts sequences, which is the cost the
    change is meant to reduce; ``model_forward`` is batched, so counting calls
    alone would not show it.
    """

    vocab_size = LabelTokenizer.vocab_size

    def __init__(self) -> None:
        self.forward_calls = 0
        self.rows_forwarded = 0
        self.free_calls = 0

    def model_forward(self, *, input_ids: Any) -> Any:
        import torch

        rows = input_ids.tolist()
        self.forward_calls += 1
        self.rows_forwarded += len(rows)
        return torch.tensor(
            [
                [_position_logits(state, self.vocab_size) for state in _prefix_states(row)]
                for row in rows
            ],
            dtype=torch.float32,
        )

    def free_inference_cache(self) -> None:
        self.free_calls += 1


def _prefix_states(token_ids: list[int]) -> list[int]:
    """Fold each prefix of a row into one number, one per position."""
    states = []
    state = 1
    for token_id in token_ids:
        state = (state * 1_103_515_245 + token_id + 12_345) % 2_147_483_647
        states.append(state)
    return states


def _position_logits(state: int, vocab_size: int) -> list[float]:
    return [
        ((state ^ ((token_id + 1) * 2_654_435_761)) % 1_000_003) / 1_000_003 * 12.0 - 6.0
        for token_id in range(vocab_size)
    ]


@pytest.fixture
def label_mc_provider() -> tuple[OlmoCoreProvider, CountingLogitsModule]:
    pytest.importorskip("torch")

    provider = OlmoCoreProvider.__new__(OlmoCoreProvider)
    module = CountingLogitsModule()
    provider.model_name = "fake-model"
    provider.tokenizer = LabelTokenizer()
    provider.generation_module = module
    provider.pad_token_id = LabelTokenizer.pad_token_id
    provider.eos_token_id = LabelTokenizer.eos_token_id
    provider.use_cache = True
    provider.add_bos_token = False
    provider.share_logprob_forwards = True
    provider.batch_size = None
    provider.chat_template = None
    provider.max_length = 512
    provider.device = "cpu"
    return provider, module


def _labelled_prompt(question: str, choices: tuple[str, ...]) -> str:
    """The prompt shape the ``:mc`` variants build, ending at ``Answer:``."""
    options = "\n".join(f" {chr(ord('A') + i)}. {choice}" for i, choice in enumerate(choices))
    return f"Question: {question}\n{options}\nAnswer:"


def _mc_request(question: str, choices: tuple[str, ...]) -> LMRequest:
    """The request an ``:mc`` task emits: one prompt, one label per choice."""
    return LMRequest(
        request_type=RequestType.LOGLIKELIHOOD,
        prompt=_labelled_prompt(question, choices),
        continuations=tuple(f" {chr(ord('A') + i)}" for i in range(len(choices))),
    )


def _scores(outputs: list[LMOutput]) -> list[float]:
    return [output.metadata["total_logprob"] for output in outputs]


def _tokens(output: LMOutput) -> list[str]:
    return [entry["token"] for entry in output.logprobs or []]


def _token_logprobs(output: LMOutput) -> list[float]:
    return [entry["logprob"] for entry in output.logprobs or []]


_ARC_EASY_MC = _mc_request(
    "Which of these keeps the earth warm?",
    ("the sun", "the moon", "a nearby star", "the ocean"),
)
_CSQA_MC = _mc_request(
    "Where would you keep a spare tire?",
    ("in the trunk", "on the roof", "under a bed", "in a pond", "at the office"),
)
_SOCIALIQA_MC = _mc_request(
    "Sam handed Robin a towel. Why did Sam do this?",
    ("to be helpful", "to be rude", "to get wet"),
)


def test_plan_logprob_forwards_shares_identical_inputs_and_can_be_turned_off() -> None:
    rows = [
        LogprobInput(
            input_ids=[10, 11, 12],
            input_length=3,
            num_tokens_all=4,
            continuation_token_ids=[label],
            continuation=f" {label}",
        )
        for label in (3, 4, 5)
    ]

    assert _plan_logprob_forwards(rows) == ([[10, 11, 12]], [0, 0, 0])
    assert _plan_logprob_forwards(rows, share_identical_inputs=False) == (
        [[10, 11, 12], [10, 11, 12], [10, 11, 12]],
        [0, 1, 2],
    )


@pytest.mark.parametrize(
    ("request_", "num_choices"),
    [(_ARC_EASY_MC, 4), (_CSQA_MC, 5), (_SOCIALIQA_MC, 3)],
    ids=["arc_easy:mc", "csqa:mc", "socialiqa:mc"],
)
def test_mc_labels_cost_one_forward_per_instance(
    label_mc_provider: tuple[OlmoCoreProvider, CountingLogitsModule],
    request_: LMRequest,
    num_choices: int,
) -> None:
    provider, module = label_mc_provider

    provider.share_logprob_forwards = False
    unshared = provider.logprobs([request_])
    unshared_rows = module.rows_forwarded

    provider.share_logprob_forwards = True
    shared = provider.logprobs([request_])
    shared_rows = module.rows_forwarded - unshared_rows

    assert unshared_rows == num_choices
    assert shared_rows == 1
    assert len(shared[0]) == len(unshared[0]) == num_choices


def test_shared_forward_scores_match_per_continuation_forwards(
    label_mc_provider: tuple[OlmoCoreProvider, CountingLogitsModule],
) -> None:
    provider, module = label_mc_provider
    requests = [_ARC_EASY_MC, _CSQA_MC, _SOCIALIQA_MC]

    provider.share_logprob_forwards = False
    unshared = provider.logprobs(requests)
    unshared_rows = module.rows_forwarded

    provider.share_logprob_forwards = True
    shared = provider.logprobs(requests)
    shared_rows = module.rows_forwarded - unshared_rows

    assert unshared_rows == 12
    assert shared_rows == 3

    for shared_outputs, unshared_outputs in zip(shared, unshared, strict=True):
        shared_scores = _scores(shared_outputs)
        unshared_scores = _scores(unshared_outputs)
        assert shared_scores == pytest.approx(unshared_scores, abs=1e-12)
        assert shared_scores.index(max(shared_scores)) == unshared_scores.index(
            max(unshared_scores)
        )
        for shared_output, unshared_output in zip(shared_outputs, unshared_outputs, strict=True):
            assert shared_output.text == unshared_output.text
            assert _tokens(shared_output) == _tokens(unshared_output)
            assert _token_logprobs(shared_output) == pytest.approx(
                _token_logprobs(unshared_output), abs=1e-12
            )
            assert shared_output.metadata["num_tokens"] == 1
            assert shared_output.metadata["num_tokens_all"] == (
                unshared_output.metadata["num_tokens_all"]
            )
            assert shared_output.metadata["is_greedy"] == unshared_output.metadata["is_greedy"]


def test_multi_token_continuations_keep_their_own_forwards(
    label_mc_provider: tuple[OlmoCoreProvider, CountingLogitsModule],
) -> None:
    provider, module = label_mc_provider
    request = LMRequest(
        request_type=RequestType.LOGLIKELIHOOD,
        prompt="Question: Which of these keeps the earth warm?\nAnswer:",
        continuations=(" the sun", " the moon", " a nearby star"),
    )

    provider.share_logprob_forwards = False
    unshared = provider.logprobs([request])
    unshared_rows = module.rows_forwarded

    provider.share_logprob_forwards = True
    shared = provider.logprobs([request])

    assert unshared_rows == 3
    assert module.rows_forwarded - unshared_rows == 3
    assert _scores(shared[0]) == pytest.approx(_scores(unshared[0]), abs=1e-12)


def test_mixed_continuation_lengths_share_only_the_single_token_labels(
    label_mc_provider: tuple[OlmoCoreProvider, CountingLogitsModule],
) -> None:
    provider, module = label_mc_provider
    request = LMRequest(
        request_type=RequestType.LOGLIKELIHOOD,
        prompt=_labelled_prompt("Which of these keeps the earth warm?", ("the sun", "the moon")),
        continuations=(" A", " B", " none of these"),
    )

    provider.share_logprob_forwards = False
    unshared = provider.logprobs([request])
    unshared_rows = module.rows_forwarded

    provider.share_logprob_forwards = True
    shared = provider.logprobs([request])

    assert unshared_rows == 3
    # The two labels share the prompt; the multi-token choice puts its own first
    # token into the model input, so it cannot and does not share.
    assert module.rows_forwarded - unshared_rows == 2
    assert _scores(shared[0]) == pytest.approx(_scores(unshared[0]), abs=1e-12)


def test_equally_long_but_different_continuations_keep_their_own_forwards(
    label_mc_provider: tuple[OlmoCoreProvider, CountingLogitsModule],
) -> None:
    """Same-sized model inputs are not the same question, so they cannot be shared."""
    provider, module = label_mc_provider
    request = LMRequest(
        request_type=RequestType.LOGLIKELIHOOD,
        prompt=_labelled_prompt("Which of these keeps the earth warm?", ("the sun", "the moon")),
        continuations=(" AB", " CD"),
    )

    rows = provider._logprob_inputs_for_request(request)
    assert [len(row.input_ids) for row in rows] == [len(rows[0].input_ids)] * 2
    assert rows[0].input_ids != rows[1].input_ids

    provider.share_logprob_forwards = False
    unshared = provider.logprobs([request])
    unshared_rows = module.rows_forwarded

    provider.share_logprob_forwards = True
    shared = provider.logprobs([request])

    assert unshared_rows == 2
    assert module.rows_forwarded - unshared_rows == 2
    assert _scores(shared[0]) == pytest.approx(_scores(unshared[0]), abs=1e-12)


def test_per_continuation_prompts_keep_their_own_forwards(
    label_mc_provider: tuple[OlmoCoreProvider, CountingLogitsModule],
) -> None:
    provider, module = label_mc_provider
    request = LMRequest(
        request_type=RequestType.LOGLIKELIHOOD,
        prompt="Question: Which of these keeps the earth warm?\nAnswer:",
        continuations=(" A", " B", " A", " B"),
        continuation_prompts=(
            "Question: Which of these keeps the earth warm?\nAnswer:",
            "Question: Which of these keeps the earth warm?\nAnswer:",
            "Answer:",
            "Answer:",
        ),
    )

    provider.share_logprob_forwards = False
    unshared = provider.logprobs([request])
    unshared_rows = module.rows_forwarded

    provider.share_logprob_forwards = True
    shared = provider.logprobs([request])

    assert unshared_rows == 4
    # One forward for the conditioned pair, one for the unconditional pair.
    assert module.rows_forwarded - unshared_rows == 2
    assert _scores(shared[0]) == pytest.approx(_scores(unshared[0]), abs=1e-12)


def test_trailing_space_prompt_still_yields_single_label_tokens(
    label_mc_provider: tuple[OlmoCoreProvider, CountingLogitsModule],
) -> None:
    """A prompt ending in a space is where a hand-rolled label encoding would drift.

    ``encode_context_and_continuation`` moves the space onto the continuation, so
    the scored token is the ``" A"`` label rather than a bare ``"A"``.
    """
    provider, module = label_mc_provider
    request = LMRequest(
        request_type=RequestType.LOGLIKELIHOOD,
        prompt="Question: Which of these keeps the earth warm?\nAnswer: ",
        continuations=("A", "B", "C"),
    )

    rows = provider._logprob_inputs_for_request(request)
    assert [row.continuation_token_ids for row in rows] == [[3], [4], [5]]

    outputs = provider.logprobs([request])
    assert module.rows_forwarded == 1
    assert [output.text for output in outputs[0]] == ["A", "B", "C"]


def test_empty_continuation_scores_zero_alongside_a_shared_label(
    label_mc_provider: tuple[OlmoCoreProvider, CountingLogitsModule],
) -> None:
    provider, module = label_mc_provider
    request = LMRequest(
        request_type=RequestType.LOGLIKELIHOOD,
        prompt="Question: Which of these keeps the earth warm?\nAnswer:",
        continuations=("", " A"),
    )

    outputs = provider.logprobs([request])

    assert module.rows_forwarded == 1
    assert outputs[0][0].logprobs == []
    assert outputs[0][0].metadata["num_tokens"] == 0
    assert outputs[0][1].metadata["num_tokens"] == 1


def test_per_char_accuracy_reads_the_label_off_shared_outputs(
    label_mc_provider: tuple[OlmoCoreProvider, CountingLogitsModule],
) -> None:
    """``socialiqa:mc`` divides by ``len(output.text)``, so the label has to be there."""
    provider, _ = label_mc_provider
    outputs = provider.logprobs([_SOCIALIQA_MC])[0]
    scores = _scores(outputs)
    gold_idx = scores.index(max(scores))

    assert [output.text for output in outputs] == [" A", " B", " C"]
    response = Response(
        instance=Instance(question="", choices=("a", "b", "c"), metadata={"gold_idx": gold_idx}),
        request=_SOCIALIQA_MC,
        outputs=outputs,
    )
    assert LogprobPerCharMCAccuracyMetric().compute_instance(response) == 1.0


def test_logprobs_clears_generation_cache_before_forward(
    fake_provider: tuple[OlmoCoreProvider, FakeGenerationModule, FakeTokenizer],
) -> None:
    provider, module, _ = fake_provider
    module.cache_allocated = True
    cache_states: list[bool] = []

    def logprobs_chunk(requests: list[LMRequest]) -> list[list[LMOutput]]:
        cache_states.append(module.cache_allocated)
        return [[] for _ in requests]

    provider._logprobs_chunk = logprobs_chunk
    provider.logprobs(
        [
            LMRequest(
                request_type=RequestType.LOGLIKELIHOOD,
                prompt="Prompt",
                continuations=("!",),
            )
        ]
    )

    assert cache_states == [False]
    assert module.cache_allocated is False
    assert module.free_calls == 1
