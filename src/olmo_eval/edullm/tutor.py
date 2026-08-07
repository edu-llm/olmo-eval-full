"""OLMo-provider adapter for EduLLM tutor-response generation.

Message construction is extracted from
``origin/frq/infobench:eduLLM-Evals/tutor_cat/respgen/prompts.py`` and its
``tutor_cat/chat_shape.py`` dependency.  In particular, it preserves the source
benchmark-specific system-prompt policy, dataset-role normalization, adjacent
turn coalescing, and user-first transport guard.  Rubrics and reference answers
are intentionally absent from tutor requests.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from olmo_eval.common.types import LMOutput, LMRequest, RequestType, SamplingParams
from olmo_eval.edullm.bank import Scenario
from olmo_eval.inference.base import InferenceProvider

ROLE_MAP: Mapping[str, str] = {
    "student": "user",
    "tutor": "assistant",
    "user": "user",
    "assistant": "assistant",
    "system": "system",
}

CONVERSATION_OPENER = "(Beginning of the conversation.)"

SYSTEM_PROMPTS: Mapping[str, str] = {
    "adaptive_explanation": (
        "You are an AI tutor helping a high school student understand a concept. "
        "Answer their question clearly and adjust your explanation based on what "
        "the student says they're confused about."
    ),
    "feedback": (
        "You are an AI tutor reviewing a student's answer to a question. Evaluate "
        "whether it is correct, identify any mistakes, and explain your reasoning "
        "clearly. Provide an assessment of the student incorrect solution in the "
        "first response."
    ),
    "hint_generation": (
        "You are an AI tutor helping a student who got stuck partway through a "
        "problem. Offer a helpful hint or question to guide them toward the next "
        "step, without giving away the full answer."
    ),
}

SYSTEM_PROMPTS_BY_BENCHMARK: Mapping[str, str] = {
    "TutorEval": (
        "You are an expert science tutor helping a student. Answer the student's "
        "question accurately and clearly. If reference material is provided, ground "
        "your answer in it; otherwise rely on your own knowledge."
    ),
    "WildBench": (
        "You are a helpful assistant. Respond to the user's request as helpfully, "
        "accurately, and thoroughly as you can."
    ),
    "Bridge": (
        "You are an AI math tutor. The student has just made a mistake in the "
        "conversation. Identify the specific error, then help the student correct it "
        "by guiding them toward the right approach rather than simply giving away the "
        "answer. Keep a supportive, encouraging tone."
    ),
}

NO_SYSTEM_BENCHMARKS = frozenset({"IFEval", "InFoBench", "EduBench"})
DEFAULT_USE_CASE = "adaptive_explanation"
COALESCE_LABEL: Mapping[str, str] = {
    "feedback": "Student's solution",
    "hint_generation": "Student's work so far",
}


@dataclass(frozen=True, slots=True)
class TutorGenerationConfig:
    """Fully explicit generation settings supplied by the mode configuration."""

    max_tokens: int
    temperature: float
    top_p: float | None
    top_k: int | None
    stop_sequences: tuple[str, ...] | None
    num_samples: int
    do_sample: bool

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_tokens, bool)
            or not isinstance(self.max_tokens, int)
            or self.max_tokens < 1
        ):
            raise ValueError("tutor max_tokens must be a positive integer")
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("tutor temperature must be finite and non-negative")
        if self.top_p is not None and (not math.isfinite(self.top_p) or not 0 < self.top_p <= 1):
            raise ValueError("tutor top_p must be null or in (0, 1]")
        if self.top_k is not None and (
            isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k < 1
        ):
            raise ValueError("tutor top_k must be null or a positive integer")
        if self.stop_sequences is not None and any(not value for value in self.stop_sequences):
            raise ValueError("tutor stop_sequences cannot contain an empty string")
        if (
            isinstance(self.num_samples, bool)
            or not isinstance(self.num_samples, int)
            or self.num_samples != 1
        ):
            raise ValueError("EduLLM adaptive mode requires exactly one tutor sample")
        if not isinstance(self.do_sample, bool):
            raise ValueError("tutor do_sample must be boolean")

    def sampling_params(self) -> SamplingParams:
        return SamplingParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            stop_sequences=self.stop_sequences,
            num_samples=self.num_samples,
            do_sample=self.do_sample,
        )


@dataclass(frozen=True, slots=True)
class TutorResponse:
    """One raw candidate response and the exact OLMo request that produced it."""

    scenario_id: str
    request: LMRequest
    output: LMOutput


def coalesce_adjacent(
    messages: Sequence[Mapping[str, str]],
    separator: Callable[[str], str] | None = None,
) -> list[dict[str, str]]:
    """Merge adjacent non-system turns with the same normalized role."""

    joiner = separator if separator is not None else (lambda _role: "\n\n")
    output: list[dict[str, str]] = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        if output and output[-1]["role"] == role and role != "system":
            output[-1] = {
                "role": role,
                "content": output[-1]["content"] + joiner(role) + content,
            }
        else:
            output.append({"role": role, "content": content})
    return output


def ensure_user_first(
    messages: Sequence[Mapping[str, str]],
    opener: str = CONVERSATION_OPENER,
) -> list[dict[str, str]]:
    """Insert the audited placeholder when conversation context starts with a tutor."""

    output = [dict(message) for message in messages]
    position = 1 if output and output[0]["role"] == "system" else 0
    if position < len(output) and output[position]["role"] == "assistant":
        output.insert(position, {"role": "user", "content": opener})
    return output


def normalize_messages(
    messages: Sequence[Mapping[str, str]],
    separator: Callable[[str], str] | None = None,
) -> list[dict[str, str]]:
    """Apply the source transport rules in their required order."""

    return ensure_user_first(coalesce_adjacent(messages, separator))


def system_prompt_for(use_case: str) -> str:
    """Return the TutorBench prompt, retaining its source fallback behavior."""

    return SYSTEM_PROMPTS.get(use_case, SYSTEM_PROMPTS[DEFAULT_USE_CASE])


def system_prompt_for_scenario(scenario: Scenario) -> str | None:
    """Select a system prompt by benchmark first and TutorBench use case second."""

    benchmark = scenario.benchmark or ""
    if benchmark == "BiGGen":
        return scenario.system_prompt or None
    if benchmark in NO_SYSTEM_BENCHMARKS:
        return None
    if benchmark in SYSTEM_PROMPTS_BY_BENCHMARK:
        return SYSTEM_PROMPTS_BY_BENCHMARK[benchmark]
    return system_prompt_for(scenario.use_case or DEFAULT_USE_CASE)


def _separator(use_case: str, role: str) -> str:
    label = COALESCE_LABEL.get(use_case)
    if role == "user" and label:
        return f"\n\n---\n{label}:\n"
    return "\n\n"


def build_tutor_messages(scenario: Scenario) -> tuple[dict[str, str], ...]:
    """Build the candidate prompt without leaking rubric or reference data."""

    use_case = scenario.use_case or DEFAULT_USE_CASE
    messages: list[dict[str, str]] = []
    system = system_prompt_for_scenario(scenario)
    if system is not None:
        messages.append({"role": "system", "content": system})
    for turn in scenario.conversation_context:
        role = ROLE_MAP.get(str(turn.get("role", "user")), "user")
        messages.append({"role": role, "content": str(turn.get("content", ""))})
    messages.append({"role": "user", "content": scenario.prompt})
    return tuple(
        normalize_messages(
            messages,
            separator=lambda role: _separator(use_case, role),
        )
    )


async def _generate_one(
    provider: InferenceProvider,
    request: LMRequest,
    sampling_params: SamplingParams,
) -> LMOutput:
    try:
        batches = await provider.agenerate([request], sampling_params)
    except NotImplementedError:
        batches = await asyncio.to_thread(provider.generate, [request], sampling_params)
    if len(batches) != 1 or len(batches[0]) != 1:
        raise RuntimeError("tutor provider must return exactly one output per scenario")
    return batches[0][0]


async def generate_tutor_response(
    provider: InferenceProvider,
    scenario: Scenario,
    config: TutorGenerationConfig,
) -> TutorResponse:
    """Generate one response through the OLMo-owned primary provider."""

    request = LMRequest(
        request_type=RequestType.CHAT,
        messages=build_tutor_messages(scenario),
    )
    output = await _generate_one(provider, request, config.sampling_params())
    return TutorResponse(
        scenario_id=scenario.scenario_id,
        request=request,
        output=output,
    )


__all__ = [
    "CONVERSATION_OPENER",
    "NO_SYSTEM_BENCHMARKS",
    "ROLE_MAP",
    "SYSTEM_PROMPTS",
    "SYSTEM_PROMPTS_BY_BENCHMARK",
    "TutorGenerationConfig",
    "TutorResponse",
    "build_tutor_messages",
    "coalesce_adjacent",
    "ensure_user_first",
    "generate_tutor_response",
    "normalize_messages",
    "system_prompt_for",
    "system_prompt_for_scenario",
]
