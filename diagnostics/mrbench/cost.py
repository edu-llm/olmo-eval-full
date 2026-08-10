"""Cost estimate for a full Milestone-1 judge run (no network).

Builds the actual Figure-6 prompt for every (response, dimension) over the real
MRBench V1 data, measures input size from those prompts, and translates to a
dollar range using assumed Claude Sonnet pricing. TrueFoundry passes provider
pricing through, so the underlying Anthropic Sonnet rates are used.
"""

from __future__ import annotations

from dataclasses import dataclass

from .judge_prompt import build_messages
from .reference import DIMENSIONS, PAPER_TO_DATA_KEY

# --------------------------------------------------------------------------- #
# Phase B: tutor-response generation prompt (paper Figure 2, byte-faithful).
# Kept judge-agnostic — this is the *tutor-under-test* prompt, independent of
# whatever model later acts as judge. Assistant cue folded onto the user turn,
# same convention as judge_prompt.py. NOTE: the MathDial cue's "hisotry" is a
# verbatim paper typo. When Phase B is implemented these move to a dedicated
# module; they live here now only to size the generation cost.
# --------------------------------------------------------------------------- #
_GEN_SYSTEM_BRIDGE = (
    "You are an experienced elementary school math teacher, and you are going to "
    "respond to a student's mistake in a useful and caring way"
)
_GEN_SYSTEM_MATHDIAL = (
    "You are an experienced middle school math teacher, and you are going to "
    "respond to a student's mistake in a useful and caring way"
)
_GEN_USER_BRIDGE = (
    "The problem your student is solving is on the topic: {topic}\n"
    "Conversation History: {history}\n"
    "Tutor response (maximum one sentence that is most appropriate given topic "
    "and conversation history):"
)
_GEN_USER_MATHDIAL = (
    "The conversation history of the problem your student is solving is: {history}\n"
    "Tutor response (maximum one sentence that aligns most appropriately with "
    "conversation hisotry):"
)

# Assumed pricing (USD per million tokens). Anthropic Claude Sonnet 4.x tier.
# STATED ASSUMPTION — verify against current pricing before spending.
RATE_INPUT_PER_MTOK = 3.0
RATE_OUTPUT_PER_MTOK = 15.0

# Output is a one-sentence feedback + "[RESULT] N"; small and bounded.
DEFAULT_OUTPUT_TOKENS_PER_CALL = 40

# Char-per-token heuristic and the band used to bracket the estimate.
CHARS_PER_TOKEN = 4.0
CHARS_PER_TOKEN_LOW = 4.5  # fewer tokens -> low cost
CHARS_PER_TOKEN_HIGH = 3.5  # more tokens -> high cost


@dataclass
class CostEstimate:
    n_calls: int
    total_input_chars: int
    chars_per_token: float
    input_tokens: int
    output_tokens_per_call: int
    total_output_tokens: int
    rate_input_per_mtok: float
    rate_output_per_mtok: float
    input_cost: float
    output_cost: float
    total_cost: float
    total_cost_low: float
    total_cost_high: float
    judge_samples: int = 1

    @property
    def avg_input_chars(self) -> float:
        return self.total_input_chars / self.n_calls if self.n_calls else 0.0


def _dollar(input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens / 1_000_000 * RATE_INPUT_PER_MTOK
        + output_tokens / 1_000_000 * RATE_OUTPUT_PER_MTOK
    )


def estimate_cost(
    conversations: list[dict],
    output_tokens_per_call: int = DEFAULT_OUTPUT_TOKENS_PER_CALL,
    judge_samples: int = 1,
) -> CostEstimate:
    """Measure real prompt sizes and estimate the full-run cost.

    ``judge_samples`` (self-consistency k, default 1 = paper single-sample)
    multiplies every judge call and its tokens by k; k=1 leaves the numbers
    byte-identical to the single-sample estimate.
    """
    data_keys = set(PAPER_TO_DATA_KEY.values())
    total_input_chars = 0
    n_calls = 0
    for conv in conversations:
        history = conv.get("conversation_history", "")
        for data_key, info in conv.get("anno_llm_responses", {}).items():
            if data_key not in data_keys:
                continue
            response = info.get("response", "")
            for dim in DIMENSIONS:
                messages = build_messages(history, response, dim)
                total_input_chars += sum(len(m["content"]) for m in messages)
                n_calls += 1

    judge_samples = max(1, judge_samples)
    total_input_chars *= judge_samples
    n_calls *= judge_samples
    input_tokens = round(total_input_chars / CHARS_PER_TOKEN)
    total_output_tokens = n_calls * output_tokens_per_call
    input_cost = input_tokens / 1_000_000 * RATE_INPUT_PER_MTOK
    output_cost = total_output_tokens / 1_000_000 * RATE_OUTPUT_PER_MTOK

    low = _dollar(round(total_input_chars / CHARS_PER_TOKEN_LOW), total_output_tokens)
    high = _dollar(round(total_input_chars / CHARS_PER_TOKEN_HIGH), total_output_tokens)

    return CostEstimate(
        n_calls=n_calls,
        total_input_chars=total_input_chars,
        chars_per_token=CHARS_PER_TOKEN,
        input_tokens=input_tokens,
        output_tokens_per_call=output_tokens_per_call,
        total_output_tokens=total_output_tokens,
        rate_input_per_mtok=RATE_INPUT_PER_MTOK,
        rate_output_per_mtok=RATE_OUTPUT_PER_MTOK,
        input_cost=input_cost,
        output_cost=output_cost,
        total_cost=input_cost + output_cost,
        total_cost_low=low,
        total_cost_high=high,
        judge_samples=judge_samples,
    )


# --------------------------------------------------------------------------- #
# Phase B cost: generate one tutor response per conversation, then judge it on
# all 8 dimensions. Fully parameterised by price, so it is judge-independent and
# works for any generation/judge model pairing.
# --------------------------------------------------------------------------- #
def _gen_prompt_chars(conv: dict) -> int:
    is_bridge = conv.get("Data") == "Bridge"
    system = _GEN_SYSTEM_BRIDGE if is_bridge else _GEN_SYSTEM_MATHDIAL
    history = conv.get("conversation_history", "")
    if is_bridge:
        user = _GEN_USER_BRIDGE.format(topic=conv.get("Topic", ""), history=history)
    else:
        user = _GEN_USER_MATHDIAL.format(history=history)
    return len(system) + len(user)


def _mean_response_chars(conversations: list[dict]) -> float:
    data_keys = set(PAPER_TO_DATA_KEY.values())
    total = 0
    n = 0
    for conv in conversations:
        for data_key, info in conv.get("anno_llm_responses", {}).items():
            if data_key in data_keys:
                total += len(info.get("response", ""))
                n += 1
    return total / n if n else 0.0


@dataclass
class PhaseBCostEstimate:
    n_conversations: int
    gen_calls: int
    judge_calls: int
    assumed_response_chars: float
    gen_input_tokens: int
    gen_output_tokens: int
    judge_input_tokens: int
    judge_output_tokens: int
    gen_rate_in: float
    gen_rate_out: float
    judge_rate_in: float
    judge_rate_out: float
    gen_cost: float
    judge_cost: float
    total_cost: float
    judge_samples: int = 1


def estimate_phase_b_cost(
    conversations: list[dict],
    *,
    gen_output_tokens_per_call: int | None = None,
    judge_output_tokens_per_call: int = DEFAULT_OUTPUT_TOKENS_PER_CALL,
    gen_rate_in: float = RATE_INPUT_PER_MTOK,
    gen_rate_out: float = RATE_OUTPUT_PER_MTOK,
    judge_rate_in: float = RATE_INPUT_PER_MTOK,
    judge_rate_out: float = RATE_OUTPUT_PER_MTOK,
    assumed_response_chars: float | None = None,
    judge_samples: int = 1,
) -> PhaseBCostEstimate:
    """Estimate cost of scoring ONE model-under-test over all conversations.

    calls = (#conversations generation calls) + (#conversations x 8 judge calls).
    Prices for the generation model and the judge model are independent
    parameters, so nothing here assumes a particular judge. ``judge_samples``
    (self-consistency k, default 1) multiplies only the judge side by k.
    """
    n = len(conversations)
    judge_samples = max(1, judge_samples)
    gen_calls = n
    judge_calls = n * len(DIMENSIONS) * judge_samples

    if assumed_response_chars is None:
        assumed_response_chars = _mean_response_chars(conversations)
    placeholder = "x" * int(round(assumed_response_chars))
    if gen_output_tokens_per_call is None:
        gen_output_tokens_per_call = max(1, round(assumed_response_chars / CHARS_PER_TOKEN))

    gen_input_chars = sum(_gen_prompt_chars(c) for c in conversations)
    judge_input_chars = 0
    for conv in conversations:
        history = conv.get("conversation_history", "")
        for dim in DIMENSIONS:
            messages = build_messages(history, placeholder, dim)
            judge_input_chars += sum(len(m["content"]) for m in messages)

    gen_input_tokens = round(gen_input_chars / CHARS_PER_TOKEN)
    gen_output_tokens = gen_calls * gen_output_tokens_per_call
    judge_input_tokens = round(judge_input_chars / CHARS_PER_TOKEN) * judge_samples
    judge_output_tokens = judge_calls * judge_output_tokens_per_call

    gen_cost = (
        gen_input_tokens / 1_000_000 * gen_rate_in + gen_output_tokens / 1_000_000 * gen_rate_out
    )
    judge_cost = (
        judge_input_tokens / 1_000_000 * judge_rate_in
        + judge_output_tokens / 1_000_000 * judge_rate_out
    )

    return PhaseBCostEstimate(
        n_conversations=n,
        gen_calls=gen_calls,
        judge_calls=judge_calls,
        assumed_response_chars=assumed_response_chars,
        gen_input_tokens=gen_input_tokens,
        gen_output_tokens=gen_output_tokens,
        judge_input_tokens=judge_input_tokens,
        judge_output_tokens=judge_output_tokens,
        gen_rate_in=gen_rate_in,
        gen_rate_out=gen_rate_out,
        judge_rate_in=judge_rate_in,
        judge_rate_out=judge_rate_out,
        gen_cost=gen_cost,
        judge_cost=judge_cost,
        total_cost=gen_cost + judge_cost,
        judge_samples=judge_samples,
    )
