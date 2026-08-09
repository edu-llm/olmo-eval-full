from __future__ import annotations

import asyncio
import json
import math

import pytest

from olmo_eval.common.types import LMOutput, LMRequest, SamplingParams
from olmo_eval.edullm.judge import (
    ATOMIC_JSON_SCHEMA,
    FAILURE_PROBABILITY_THRESHOLD,
    PROBABILITY_AGGREGATION,
    AtomicJudgeResult,
    AtomicRequirement,
    BlindedJudgeCase,
    ClassificationProbabilityUnavailable,
    QwenZeroShotBinaryJudge,
    aggregate_atomic_judgments,
    build_atomic_messages,
    build_classification_messages,
    classification_probabilities,
    normalize_atomic_judgment,
    parse_native_binary_output,
    resolve_classification_token_ids,
)
from olmo_eval.inference.base import InferenceProvider


def _case() -> BlindedJudgeCase:
    return BlindedJudgeCase(
        scenario_prompt="Help the student solve the equilibrium problem.",
        candidate_response="Your setup is correct; Ka is about 1.84e-5.",
        conversation_context=(
            {"role": "student", "content": "I calculated the acid concentration."},
        ),
        reference_solution="Ka = 1.847e-5.",
        expected_evidence=("State a correctly rounded Ka value.",),
    )


def _requirement(requirement_id: str = "R1") -> AtomicRequirement:
    return AtomicRequirement(requirement_id, "Reports Ka near 1.847e-5.")


def _native(verdict: str = "pass") -> str:
    return json.dumps(
        {
            "verdict": verdict,
            "rationale": "The response gives a correctly rounded value.",
            "evidence": "Ka is about 1.84e-5",
        }
    )


def _classification_output(*, hard_label: str, p_fail: float) -> LMOutput:
    p_pass = 1.0 - p_fail
    chosen_probability = p_pass if hard_label == "P" else p_fail
    return LMOutput(
        text=hard_label,
        logprobs=[
            {
                "token": f" {hard_label}",
                "logprob": math.log(chosen_probability),
                "top_logprobs": [
                    {"token": " P", "logprob": math.log(p_pass)},
                    {"token": " F", "logprob": math.log(p_fail)},
                ],
            }
        ],
    )


class FakeTokenizer:
    decoded = {
        47: "P",
        387: " P",
        9729: "P ",
        37: "F",
        426: " F",
        12362: "F ",
        99: "other",
    }

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return [{"P": 47, "F": 37}[text]]

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        assert skip_special_tokens is True
        assert clean_up_tokenization_spaces is False
        return self.decoded[token_ids[0]]

    def get_vocab(self) -> dict[str, int]:
        return {f"token-{token_id}": token_id for token_id in self.decoded}


class RemoteLikeTokenizer:
    decoded = {
        47: "P",
        387: " P",
        9729: "P ",
        37: "F",
        426: " F",
        12362: "F ",
    }

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return [{"P": 47, "F": 37}[text]]

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        assert skip_special_tokens is True
        assert clean_up_tokenization_spaces is False
        return self.decoded[token_ids[0]]


class QueueProvider(InferenceProvider):
    def __init__(
        self,
        outputs: list[LMOutput | Exception],
        *,
        tokenizer: object | None = None,
    ) -> None:
        super().__init__("Qwen/Qwen3.5-9B")
        self.outputs = list(outputs)
        self.tokenizer = FakeTokenizer() if tokenizer is None else tokenizer
        self.requests: list[LMRequest] = []
        self.params: list[SamplingParams | None] = []

    def get_tokenizer(self) -> object:
        return self.tokenizer

    def generate(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        raise AssertionError("the async path should be used")

    def logprobs(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        raise AssertionError("continuation scoring is not the frozen P/F path")

    async def agenerate(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        assert len(requests) == 1
        self.requests.extend(requests)
        self.params.append(sampling_params)
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return [[output]]


class NoTokenizerProvider(QueueProvider):
    def get_tokenizer(self) -> object:
        raise NotImplementedError


def test_blinded_mapping_rejects_human_labels_and_tutor_identity() -> None:
    with pytest.raises(ValueError, match="forbidden blinded fields"):
        BlindedJudgeCase.from_mapping(
            {
                "scenario_prompt": "Question",
                "candidate_response": "Answer",
                "human_label": "P",
                "candidate_model": "secret-model",
            }
        )


def test_prompt_is_canonical_zero_shot_and_contains_one_requirement() -> None:
    messages = build_atomic_messages(_case(), _requirement("R2"))

    assert len(messages) == 1
    prompt = messages[0]["content"]
    assert "Evaluate only the single frozen atomic requirement R2." in prompt
    assert "[R2] Reports Ka near 1.847e-5." in prompt
    assert "[R1]" not in prompt
    assert "Synthetic contrastive examples" not in prompt
    assert "Current student request:" in prompt
    assert "Prior conversation context:" in prompt
    assert "<candidate_response>" in prompt
    assert "<reference_background>" in prompt
    assert "human_label" not in prompt
    assert "candidate_model" not in prompt


def test_classification_prompt_preserves_native_turn_and_frozen_instruction() -> None:
    main = build_atomic_messages(_case(), _requirement())
    classified = build_classification_messages(main, _native())

    assert classified[:-2] == main
    assert classified[-2] == {"role": "assistant", "content": _native()}
    assert classified[-1]["role"] == "user"
    assert "Output exactly one character and nothing else" in classified[-1]["content"]
    assert "P means Pass" in classified[-1]["content"]


@pytest.mark.parametrize(("verdict", "score"), [("pass", 1), ("fail", 0)])
def test_native_parser_accepts_only_strict_binary_json(verdict: str, score: int) -> None:
    parsed = parse_native_binary_output(_native(verdict))

    assert parsed.status == "ok"
    assert parsed.verdict == verdict
    assert parsed.native_score == score


@pytest.mark.parametrize(
    "raw",
    [
        '```json\n{"verdict":"pass","rationale":"yes","evidence":"quote"}\n```',
        '{"verdict":"pass","verdict":"fail","rationale":"x","evidence":"y"}',
        '{"verdict":"pass","rationale":"x","evidence":"y","score":1}',
        '{"verdict":"PASS","rationale":"x","evidence":"y"}',
        '[{"verdict":"pass","rationale":"x","evidence":"y"}]',
    ],
)
def test_native_parser_keeps_malformed_or_noncanonical_output_as_no_decision(raw: str) -> None:
    parsed = parse_native_binary_output(raw)

    assert parsed.verdict == "no_decision"
    assert parsed.status == "parse_error"
    assert parsed.error


def test_probability_normalization_uses_p_and_f_token_text_alternatives() -> None:
    label, token_id, p_pass, p_fail = classification_probabilities(
        _classification_output(hard_label="P", p_fail=0.27)
    )

    assert label == "P"
    assert token_id is None
    assert p_pass == pytest.approx(0.73)
    assert p_fail == pytest.approx(0.27)


def test_probability_normalization_rejects_missing_label_alternative() -> None:
    output = LMOutput(
        text="P",
        logprobs=[
            {
                "token": " P",
                "logprob": -0.1,
                "top_logprobs": [{"token": " P", "logprob": -0.1}],
            }
        ],
    )

    with pytest.raises(ClassificationProbabilityUnavailable, match="both P and F"):
        classification_probabilities(output)


def test_probability_normalization_rejects_text_token_disagreement() -> None:
    output = _classification_output(hard_label="P", p_fail=0.2)
    assert output.logprobs is not None
    output.logprobs[0]["token"] = "F"

    with pytest.raises(ValueError, match="do not agree"):
        classification_probabilities(output)


def test_probability_normalization_preserves_mass_from_distinct_whitespace_ids() -> None:
    output = LMOutput(
        text="P",
        logprobs=[
            {
                "token": "P",
                "logprob": math.log(0.3),
                "top_logprobs": [
                    {"token": "P", "logprob": math.log(0.3)},
                    {"token": " P", "logprob": math.log(0.2)},
                    {"token": "F", "logprob": math.log(0.4)},
                    {"token": " F", "logprob": math.log(0.1)},
                ],
            }
        ],
    )

    label, token_id, p_pass, p_fail = classification_probabilities(output)

    assert label == "P"
    assert token_id is None
    assert p_pass == pytest.approx(0.5)
    assert p_fail == pytest.approx(0.5)


def test_token_id_discovery_scans_full_vocabulary_for_whitespace_variants() -> None:
    resolved = resolve_classification_token_ids(QueueProvider([]))

    assert resolved.pass_ids == (47, 387, 9729)
    assert resolved.fail_ids == (37, 426, 12362)
    assert resolved.canonical_pass_id == 47
    assert resolved.canonical_fail_id == 37
    assert resolved.source == "tokenizer_vocabulary"


def test_remote_token_ids_can_be_injected_without_vocabulary_access() -> None:
    provider = QueueProvider([], tokenizer=RemoteLikeTokenizer())

    resolved = resolve_classification_token_ids(
        provider,
        pass_token_ids=(9729, 47, 387),
        fail_token_ids=(12362, 37, 426),
    )

    assert resolved.pass_ids == (47, 387, 9729)
    assert resolved.fail_ids == (37, 426, 12362)
    assert resolved.source == "explicit_config"


def test_remote_provider_without_injected_token_ids_fails_closed() -> None:
    provider = QueueProvider([], tokenizer=RemoteLikeTokenizer())

    with pytest.raises(ValueError, match="does not expose get_vocab"):
        QwenZeroShotBinaryJudge(provider)


def test_provider_without_tokenizer_or_injected_ids_fails_closed() -> None:
    with pytest.raises(ValueError, match="cannot discover the complete P/F token sets"):
        QwenZeroShotBinaryJudge(NoTokenizerProvider([]))


def test_frozen_threshold_can_override_a_consistent_native_pass() -> None:
    result = normalize_atomic_judgment(
        _requirement(),
        _native("pass"),
        _classification_output(hard_label="P", p_fail=FAILURE_PROBABILITY_THRESHOLD),
    )

    assert result.status == "ok"
    assert result.native_verdict == "pass"
    assert result.classification_text == "P"
    assert result.native_pf_consistent is True
    assert result.p_fail == pytest.approx(0.33)
    assert result.verdict == "fail"


def test_native_secondary_hard_label_disagreement_is_no_decision() -> None:
    result = normalize_atomic_judgment(
        _requirement(),
        _native("fail"),
        _classification_output(hard_label="P", p_fail=0.2),
    )

    assert result.verdict == "no_decision"
    assert result.status == "native_secondary_disagreement"
    assert result.native_pf_consistent is False


def test_max_atomic_p_fail_aggregation_and_no_decision_propagation() -> None:
    low = normalize_atomic_judgment(
        _requirement("R1"),
        _native("pass"),
        _classification_output(hard_label="P", p_fail=0.1),
    )
    high = normalize_atomic_judgment(
        _requirement("R2"),
        _native("pass"),
        _classification_output(hard_label="P", p_fail=0.4),
    )
    aggregate = aggregate_atomic_judgments([low, high])

    assert aggregate.verdict == "fail"
    assert aggregate.p_fail == pytest.approx(0.4)
    assert aggregate.p_pass == pytest.approx(0.6)
    assert aggregate.probability_source == PROBABILITY_AGGREGATION

    missing = AtomicJudgeResult(
        requirement_id="R3",
        requirement="Missing result",
        verdict="no_decision",
        status="parse_error",
        raw_output="bad",
        error="invalid JSON",
    )
    unresolved = aggregate_atomic_judgments([low, missing])
    assert unresolved.verdict == "no_decision"
    assert unresolved.p_fail is None


def test_async_provider_adapter_runs_native_then_classification_calls() -> None:
    provider = QueueProvider(
        [
            LMOutput(text=_native("pass")),
            _classification_output(hard_label="P", p_fail=0.2),
        ]
    )
    judge = QwenZeroShotBinaryJudge(provider)

    result = asyncio.run(judge.judge_atomic(_case(), _requirement()))

    assert result.verdict == "pass"
    assert len(provider.requests) == 2
    assert provider.requests[0].messages == build_atomic_messages(_case(), _requirement())
    assert provider.requests[1].messages[-2] == {
        "role": "assistant",
        "content": _native("pass"),
    }
    assert provider.params[0] is not None
    assert provider.params[1] is not None
    assert provider.params[0].max_tokens == 1600
    assert provider.params[0].seed == 42
    assert provider.params[0].structured_output_json_schema == ATOMIC_JSON_SCHEMA
    assert provider.params[1].max_tokens == 1
    assert provider.params[1].temperature == 0.0
    assert provider.params[1].structured_output_regex == "[PF]"
    assert provider.params[1].logprob_token_ids == (47, 387, 9729, 37, 426, 12362)
    assert provider.params[1].logprobs == 6
    assert provider.params[1].seed == 42
    assert provider.params[1].logprobs != 20
    assert provider.requests[0].chat_template_kwargs == {"enable_thinking": False}
    assert provider.requests[1].chat_template_kwargs == {"enable_thinking": False}


def test_async_adapter_uses_explicit_remote_token_ids_exactly() -> None:
    provider = QueueProvider(
        [
            LMOutput(text=_native("pass")),
            _classification_output(hard_label="P", p_fail=0.2),
        ],
        tokenizer=RemoteLikeTokenizer(),
    )
    judge = QwenZeroShotBinaryJudge(
        provider,
        pass_token_ids=(47, 387, 9729),
        fail_token_ids=(37, 426, 12362),
    )

    result = asyncio.run(judge.judge_atomic(_case(), _requirement()))

    assert result.verdict == "pass"
    assert provider.params[1] is not None
    assert provider.params[1].logprob_token_ids == (47, 387, 9729, 37, 426, 12362)
    assert provider.params[1].logprobs == 6
    assert provider.params[1].structured_output_regex == "[PF]"


def test_native_generation_failure_is_retained_as_no_decision() -> None:
    provider = QueueProvider([RuntimeError("engine unavailable")])
    judge = QwenZeroShotBinaryJudge(provider)

    result = asyncio.run(judge.judge_atomic(_case(), _requirement()))

    assert result.verdict == "no_decision"
    assert result.status == "native_generation_error"
    assert result.error == "RuntimeError: engine unavailable"


def test_truncated_native_output_is_rejected_before_classification() -> None:
    provider = QueueProvider([LMOutput(text=_native("pass"), finish_reason="length")])
    judge = QwenZeroShotBinaryJudge(provider)

    result = asyncio.run(judge.judge_atomic(_case(), _requirement()))

    assert result.verdict == "no_decision"
    assert result.status == "native_output_truncated"
    assert result.finish_reason == "length"
    assert len(provider.requests) == 1


def test_classification_emitted_token_id_must_match_frozen_label_set() -> None:
    output = _classification_output(hard_label="P", p_fail=0.2)
    output.token_ids = (37,)
    resolved = resolve_classification_token_ids(QueueProvider([]))

    with pytest.raises(ValueError, match="is not a frozen P token"):
        classification_probabilities(output, resolved)


def test_explicit_ids_cannot_override_the_frozen_qwen_revision() -> None:
    provider = NoTokenizerProvider([])

    with pytest.raises(ValueError, match="differ from the frozen Qwen revision"):
        QwenZeroShotBinaryJudge(
            provider,
            pass_token_ids=(10,),
            fail_token_ids=(20,),
        )


def test_async_adapter_exposes_provider_logprob_gap_as_no_decision() -> None:
    provider = QueueProvider(
        [
            LMOutput(text=_native("pass")),
            LMOutput(text="P", logprobs=[{"token": "P", "logprob": -0.01}]),
        ]
    )
    judge = QwenZeroShotBinaryJudge(provider)

    result = asyncio.run(judge.judge_atomic(_case(), _requirement()))

    assert result.verdict == "no_decision"
    assert result.status == "probability_unavailable"
    assert "both P and F" in str(result.error)
