"""TutorBench prompt construction: per-use_case system prompt + message building
with consecutive-user-turn coalescing.

Why coalescing: 333/662 scenarios (feedback + hint_generation) have a single
student context turn followed by the student's prompt — i.e. two consecutive
`user` turns. Strict chat templates (Gemma, Mistral, Llama-3) reject
non-alternating roles and raise, so the two user turns are merged into one with
a labeled separator. adaptive_explanation already alternates
(student -> tutor -> student) and is left untouched.

Pure module: uses only the scenario's .use_case / .conversation_context /
.prompt attributes (duck-typed), so message construction is testable without
torch/transformers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # import only for type hints; avoids a hard dep at runtime
    from ..schemas import Scenario

# Verbatim TutorBench system prompts (Appendix A.6, text-only variants), keyed by
# use_case. The model is TOLD the use case through this system prompt; the rubric
# and reference solution are judge-only and never appear here.
SYSTEM_PROMPTS: dict[str, str] = {
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

_DEFAULT_USE_CASE = "adaptive_explanation"

# --- per-benchmark system prompts (multi-benchmark generation) --------------
# When a scenario carries a `benchmark` label, the system prompt is chosen by
# benchmark FIRST (below), then by TutorBench use_case (above). Empty/"TutorBench"
# => the use_case behavior, so single-benchmark TutorBench runs are unchanged.
#
# Faithful to each original harness (see each data/*/README.md):
#   * IFEval / InFoBench / EduBench: instruction-following. The original harness
#     feeds the prompt with NO system prompt (the prompt is the complete, self-
#     contained instruction — EduBench prompts already embed subject/level/task and
#     the required output format). A tutor persona would change behavior and, for
#     IFEval, corrupt its deterministic verifier. => None (system message omitted).
#   * TutorEval: science tutoring; the chapter is already embedded in the prompt
#     for open-book items, so the system prompt only sets the tutor role.
#   * WildBench: open-ended chat across many task types (use_case is a content
#     tag, not a pedagogical mode) => a generic helpful-assistant prompt.
#   * Bridge: math mistake-remediation over a multi-turn tutor/student dialogue.
#   * BiGGen: every instance ships its OWN native system prompt (Scenario.
#     system_prompt), so it is used verbatim per row rather than a fixed prompt.
_NO_SYSTEM_BENCHMARKS = {"IFEval", "InFoBench", "EduBench"}

SYSTEM_PROMPTS_BY_BENCHMARK: dict[str, str] = {
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

# Dataset context roles -> chat-template roles (same map as tutors._ROLE_MAP).
_ROLE_MAP = {
    "student": "user",
    "tutor": "assistant",
    "user": "user",
    "assistant": "assistant",
    "system": "system",
}

# When two user turns are merged, label the second (the scenario prompt) so the
# model can tell the problem statement from the student's own work.
_COALESCE_LABEL = {
    "feedback": "Student's solution",
    "hint_generation": "Student's work so far",
}

# Flat role labels for base (no chat template) models.
_BASE_ROLE_LABEL = {"user": "Student", "assistant": "Tutor", "system": "System"}


def system_prompt_for(use_case: str) -> str:
    return SYSTEM_PROMPTS.get(use_case, SYSTEM_PROMPTS[_DEFAULT_USE_CASE])


def system_prompt_for_scenario(scenario: "Scenario") -> str | None:
    """The system prompt for a scenario, keyed by benchmark first then use_case.

    BiGGen uses the scenario's own native `system_prompt` (per-instance). Returns
    None for instruction-following benchmarks whose original harness uses no system
    prompt (IFEval/InFoBench/EduBench) — the caller then omits the system turn.
    Empty/"TutorBench" benchmark falls through to the use_case prompt, so the
    single-benchmark TutorBench path is unchanged."""
    benchmark = getattr(scenario, "benchmark", "") or ""
    if benchmark == "BiGGen":
        # Faithful to BiGGen: each instance carries its own native system prompt.
        return (getattr(scenario, "system_prompt", "") or "") or None
    if benchmark in _NO_SYSTEM_BENCHMARKS:
        return None
    if benchmark in SYSTEM_PROMPTS_BY_BENCHMARK:
        return SYSTEM_PROMPTS_BY_BENCHMARK[benchmark]
    use_case = getattr(scenario, "use_case", "") or _DEFAULT_USE_CASE
    return system_prompt_for(use_case)


def _separator(use_case: str, role: str) -> str:
    """Separator inserted between two merged same-role turns."""
    label = _COALESCE_LABEL.get(use_case)
    if role == "user" and label:
        return f"\n\n---\n{label}:\n"
    return "\n\n"


def _coalesce(messages: list[dict[str, str]], use_case: str) -> list[dict[str, str]]:
    """Merge adjacent same-role turns (never system) so the sequence alternates."""
    out: list[dict[str, str]] = []
    for m in messages:
        if out and out[-1]["role"] == m["role"] and m["role"] != "system":
            sep = _separator(use_case, m["role"])
            out[-1] = {
                "role": m["role"],
                "content": out[-1]["content"] + sep + m["content"],
            }
        else:
            out.append(dict(m))
    return out


# Some transcripts open on the tutor's turn, which maps to `assistant` and yields
# [system, assistant, ...]. AWS Bedrock rejects that outright ("A conversation must
# start with a user message"), so every Bedrock-routed model fails those scenarios --
# measured at 71/100 on llama3-3-70b before this existed, and 476 of Bridge's 642
# scenarios (74%) begin that way. Coalescing alone does not help: it only merges
# ADJACENT same-role turns and never repairs the leading one.
# A placeholder opener is inserted rather than dropping the turn, which would silently
# discard real conversational context the tutor is supposed to have seen.
_CONVERSATION_OPENER = "(Beginning of the conversation.)"


def _ensure_user_first(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Insert a placeholder user turn if the first non-system turn is the assistant."""
    at = 1 if messages and messages[0]["role"] == "system" else 0
    if at < len(messages) and messages[at]["role"] == "assistant":
        messages = list(messages)
        messages.insert(at, {"role": "user", "content": _CONVERSATION_OPENER})
    return messages


def build_chat_messages(scenario: "Scenario") -> list[dict[str, str]]:
    """[system?, ...role-mapped context, user(prompt)], coalesced to alternate.

    The system turn is chosen by benchmark then use_case, and is OMITTED for the
    instruction-following benchmarks whose original harness uses none (IFEval /
    InFoBench). For TutorBench: feedback / hint_generation collapse to exactly
    [system, user]; adaptive_explanation stays [system, user, assistant, user].
    """
    use_case = getattr(scenario, "use_case", "") or _DEFAULT_USE_CASE
    system = system_prompt_for_scenario(scenario)
    messages: list[dict[str, str]] = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    for turn in getattr(scenario, "conversation_context", None) or []:
        role = _ROLE_MAP.get(turn.get("role", "user"), "user")
        messages.append({"role": role, "content": turn.get("content", "")})
    messages.append({"role": "user", "content": scenario.prompt})
    return _ensure_user_first(_coalesce(messages, use_case))


def render_base_prompt(scenario: "Scenario") -> str:
    """Flat role-labeled transcript for base models with no chat template. Ends
    on 'Tutor:' so the model continues the assistant turn."""
    lines: list[str] = []
    for m in build_chat_messages(scenario):
        label = _BASE_ROLE_LABEL.get(m["role"], m["role"].title())
        lines.append(f"{label}: {m['content']}")
    lines.append("Tutor:")
    return "\n\n".join(lines)
