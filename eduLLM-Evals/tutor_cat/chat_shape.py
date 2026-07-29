"""Transport rules every chat message list must satisfy, in one place.

Two modules build message lists independently -- ``respgen.prompts`` for the
open-weight fleet (chat templates / vLLM) and ``tutors`` for the hosted-API
adapters. They were duplicating the role map and drifting on everything else,
which is how a real bug shipped: ``tutors`` applied neither rule below, so every
Bridge scenario opening on a tutor turn (476 of 642) failed on AWS Bedrock with
"A conversation must start with a user message", and any two consecutive tutor
turns would have been rejected by Anthropic's alternation requirement.

The rules are provider requirements, not benchmark preferences, so they belong
here rather than in either caller:

  * roles must alternate -- no two adjacent turns from the same speaker
  * the first non-system turn must be the user's

Both are no-ops for data that already complies, so benchmarks whose transcripts
start with the student (TutorBench, WildBench) are unaffected.
"""

from __future__ import annotations

from typing import Callable

# Dataset context roles -> chat roles. Datasets say student/tutor; APIs and chat
# templates say user/assistant.
ROLE_MAP: dict[str, str] = {
    "student": "user",
    "tutor": "assistant",
    "user": "user",
    "assistant": "assistant",
    "system": "system",
}

# Inserted when a transcript opens on the assistant. A placeholder is used rather
# than dropping the turn, which would silently discard real conversational context
# the model is supposed to have seen.
CONVERSATION_OPENER = "(Beginning of the conversation.)"


def coalesce_adjacent(
    messages: list[dict[str, str]],
    separator: Callable[[str], str] | None = None,
) -> list[dict[str, str]]:
    """Merge adjacent same-role turns (never system) so the sequence alternates.

    ``separator`` maps a role to the text joining two merged turns; the default
    is a blank line. Callers that label the merge (e.g. "Student's solution")
    pass their own.
    """
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
    """Insert a placeholder user turn if the first non-system turn is the assistant."""
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
