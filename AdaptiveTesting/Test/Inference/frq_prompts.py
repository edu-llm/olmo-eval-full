"""FRQ prompt construction (ported from eduLLM-Evals ``tutor_cat/respgen/prompts.py``
plus ``tutor_cat/chat_shape.py``).

This is the part that makes free-response pre-calibration *faithful*, and the
reason the pulled ``open_generate.py`` was replaced: an FRQ item is not a bare
prompt string. It carries a per-benchmark (or per-use_case) tutor system prompt
and a multi-turn ``conversation_context``, and the resulting message list has to
satisfy chat-template transport rules.

Rules preserved verbatim from respgen:
  * per-use_case TutorBench personas, per-benchmark prompts for TutorEval /
    WildBench / Bridge;
  * IFEval / InFoBench / EduBench get **no system turn** - their original
    harness uses none, the prompt is already a complete self-contained
    instruction, and a tutor persona would corrupt IFEval's verifier;
  * BiGGen uses each instance's OWN native ``system_prompt``;
  * adjacent same-role turns are coalesced (strict templates - Gemma, Mistral,
    Llama-3 - reject non-alternating roles) with a labeled separator;
  * the first non-system turn must be the user's (Bridge opens on a tutor turn
    in 476/642 scenarios);
  * base models with no chat template get a flat ``Student:``/``Tutor:``
    transcript ending on ``Tutor:``.

Pure module: duck-types the scenario (.use_case / .conversation_context /
.prompt / .benchmark / .system_prompt), so it is testable without torch.
"""

from __future__ import annotations

from collections.abc import Callable

# --------------------------------------------------------------------------
# transport rules (chat_shape.py)
# --------------------------------------------------------------------------

# Dataset context roles -> chat roles. Datasets say student/tutor; chat
# templates say user/assistant.
ROLE_MAP: dict[str, str] = {
    "student": "user",
    "tutor": "assistant",
    "user": "user",
    "assistant": "assistant",
    "system": "system",
}

# Inserted when a transcript opens on the assistant. A placeholder beats dropping
# the turn, which would silently discard real context the model should have seen.
CONVERSATION_OPENER = "(Beginning of the conversation.)"


def coalesce_adjacent(
    messages: list[dict[str, str]],
    separator: Callable[[str], str] | None = None,
) -> list[dict[str, str]]:
    """Merge adjacent same-role turns (never system) so the sequence alternates."""
    sep = separator if separator is not None else (lambda role: "\n\n")
    out: list[dict[str, str]] = []
    for m in messages:
        if out and out[-1]["role"] == m["role"] and m["role"] != "system":
            out[-1] = {
                "role": m["role"],
                "content": out[-1]["content"] + sep(m["role"]) + m["content"],
            }
        else:
            out.append(dict(m))
    return out


def ensure_user_first(
    messages: list[dict[str, str]],
    opener: str = CONVERSATION_OPENER,
) -> list[dict[str, str]]:
    """Insert a placeholder user turn if the first non-system turn is assistant."""
    at = 1 if messages and messages[0]["role"] == "system" else 0
    if at < len(messages) and messages[at]["role"] == "assistant":
        messages = list(messages)
        messages.insert(at, {"role": "user", "content": opener})
    return messages


def normalize(
    messages: list[dict[str, str]],
    separator: Callable[[str], str] | None = None,
    opener: str = CONVERSATION_OPENER,
) -> list[dict[str, str]]:
    """Apply every transport rule, in the order they must be applied.

    Coalescing runs first: merging same-role turns can leave the assistant
    leading, so the user-first guard has to see the merged sequence.
    """
    return ensure_user_first(coalesce_adjacent(messages, separator), opener)


# --------------------------------------------------------------------------
# system prompts
# --------------------------------------------------------------------------

# Verbatim TutorBench system prompts (paper Appendix A.6, text-only variants),
# keyed by use_case. The model is TOLD the use case through this system prompt;
# the rubric and reference solution are judge-only and never appear here.
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

# Instruction-following benchmarks whose original harness feeds the prompt with
# NO system prompt (the prompt is the complete, self-contained instruction). A
# tutor persona would change behavior and, for IFEval, corrupt its deterministic
# verifier. IFEval/EduBench are not in the active registry but are kept here so
# re-enabling them cannot silently lose this rule.
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

# When two user turns are merged, label the second (the scenario prompt) so the
# model can tell the problem statement from the student's own work.
_COALESCE_LABEL = {
    "feedback": "Student's solution",
    "hint_generation": "Student's work so far",
}

# Flat role labels for base (no chat template) models.
_BASE_ROLE_LABEL = {"user": "Student", "assistant": "Tutor", "system": "System"}

# Fallback benchmark label when a scenario carries none.
DEFAULT_BENCHMARK = "TutorBench"


def system_prompt_for(use_case: str) -> str:
    return SYSTEM_PROMPTS.get(use_case, SYSTEM_PROMPTS[_DEFAULT_USE_CASE])


def system_prompt_for_scenario(scenario) -> str | None:
    """The system prompt for a scenario, keyed by benchmark first then use_case.

    BiGGen uses the scenario's own native ``system_prompt`` (per instance).
    Returns None for the instruction-following benchmarks whose original harness
    uses no system prompt - the caller then omits the system turn entirely.
    """
    benchmark = getattr(scenario, "benchmark", "") or ""
    if benchmark == "BiGGen":
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


def build_chat_messages(scenario) -> list[dict[str, str]]:
    """[system?, ...role-mapped context, user(prompt)], coalesced to alternate."""
    use_case = getattr(scenario, "use_case", "") or _DEFAULT_USE_CASE
    system = system_prompt_for_scenario(scenario)
    messages: list[dict[str, str]] = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    for turn in getattr(scenario, "conversation_context", None) or []:
        role = ROLE_MAP.get(turn.get("role", "user"), "user")
        messages.append({"role": role, "content": turn.get("content", "")})
    messages.append({"role": "user", "content": scenario.prompt})
    return normalize(messages, separator=lambda role: _separator(use_case, role))


def render_base_prompt(scenario) -> str:
    """Flat role-labeled transcript for base models with no chat template. Ends
    on 'Tutor:' so the model continues the assistant turn."""
    lines: list[str] = []
    for m in build_chat_messages(scenario):
        label = _BASE_ROLE_LABEL.get(m["role"], m["role"].title())
        lines.append(f"{label}: {m['content']}")
    lines.append("Tutor:")
    return "\n\n".join(lines)
