"""Benign, meaning-preserving prompt perturbations for the MRBench judge (P2).

Each variant is a **pure function** ``(messages) -> messages`` that transforms the
Figure-6 judge message list produced by :func:`judge_prompt.build_messages`
*without changing its meaning*: only surface form (whitespace, a polite preamble,
or synonym swaps of fixed scaffolding) is altered. The tutor response text and the
rubric's ordinal level labels are always preserved, so any decision change the
judge makes is attributable to the reword and not to lost content.

The variants are re-implementations of the benign perturbations used by the
teammates' judge-robustness harnesses:

  * ``whitespace`` — follows Sameer's ``score_gold._perturb`` (leading newline,
    ``\\n\\n`` -> ``\\n \\n\\n`` to seed a blank line with a stray space, trailing
    whitespace). Lawrence's simpler ``\\n...\\n`` wrap is a subset of this.
  * ``politeness`` — Sameer's politeness prefix, prepended to the user turn.
  * ``rubric_synonyms`` — Lawrence's ``header_synonyms`` analog: swap fixed
    scaffolding phrases for synonyms, each guarded by a marker-presence check.
    Where Lawrence *raises* when a marker is absent, here a missing anchor makes
    that swap a no-op (and a variant whose anchors are all absent is the
    identity), so the transform never silently becomes a different intervention.

Nothing here calls the judge, the network, or reads gold labels.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from .judge_prompt import DEFINITIONS, RUBRICS

Messages = list[dict[str, str]]
Variant = Callable[[Sequence[dict[str, str]]], Messages]

# Canonical (identity) baseline name; every metric is computed against this.
CANONICAL = "canonical"


def _clone(messages: Sequence[dict[str, str]]) -> Messages:
    """Shallow-copy each message dict so a variant never mutates its input."""
    return [dict(m) for m in messages]


def _last_user_index(messages: Sequence[dict[str, str]]) -> int:
    """Index of the last ``user`` message (the payload turn), or the last message."""
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            return i
    return len(messages) - 1


def canonical(messages: Sequence[dict[str, str]]) -> Messages:
    """Identity baseline: a defensive copy of the messages, unchanged."""
    return _clone(messages)


def whitespace(messages: Sequence[dict[str, str]]) -> Messages:
    """Benign whitespace churn on the user turn (Sameer's ``_perturb``)."""
    msgs = _clone(messages)
    i = _last_user_index(msgs)
    content = msgs[i]["content"]
    msgs[i]["content"] = "\n" + content.replace("\n\n", "\n \n\n") + "\n  "
    return msgs


# Sameer's politeness prefix, verbatim in intent (re-implemented, not copied).
POLITENESS_PREFIX = (
    "Please read the following carefully and evaluate it objectively and impartially."
)


def politeness(messages: Sequence[dict[str, str]]) -> Messages:
    """Prepend a polite, meaning-neutral instruction to the user turn."""
    msgs = _clone(messages)
    i = _last_user_index(msgs)
    msgs[i]["content"] = POLITENESS_PREFIX + "\n\n" + msgs[i]["content"]
    return msgs


# Guarded scaffolding synonym swaps: ``(role, anchor, replacement)``. Anchors are
# fixed Figure-6 scaffolding — never a rubric level label, a definition, or the
# tutor response — so meaning is preserved. Each swap fires only when its anchor
# is present (marker-presence guard); absent anchors are skipped, and a message
# list with none of the anchors is returned unchanged (the identity).
RUBRIC_SYNONYM_SWAPS: tuple[tuple[str, str, str], ...] = (
    # System scaffolding: "objective" -> "impartial" (Lawrence's impartial<->neutral idea).
    (
        "system",
        "clear and objective single evaluation score",
        "clear and impartial single evaluation score",
    ),
    # User scaffolding header: "Scoring Rubric" -> "Scoring Guidelines"
    # (Lawrence's "Rules:" -> "Evaluation guidelines:").
    ("user", "# Scoring Rubric:", "# Scoring Guidelines:"),
    # User scaffolding header for the criteria block.
    ("user", "# Definitions of criteria:", "# Criteria Definitions:"),
)


def rubric_synonyms(messages: Sequence[dict[str, str]]) -> Messages:
    """Swap fixed scaffolding phrases for synonyms; no-op where anchors are absent."""
    msgs = _clone(messages)
    for role, anchor, replacement in RUBRIC_SYNONYM_SWAPS:
        for m in msgs:
            if m.get("role") == role and anchor in m["content"]:
                m["content"] = m["content"].replace(anchor, replacement, 1)
    return msgs


# Registry of benign variants (excludes the canonical identity baseline).
VARIANTS: dict[str, Variant] = {
    "whitespace": whitespace,
    "politeness": politeness,
    "rubric_synonyms": rubric_synonyms,
}


def apply_variant(name: str, messages: Sequence[dict[str, str]]) -> Messages:
    """Apply a named variant (``canonical`` or a key of :data:`VARIANTS`)."""
    if name == CANONICAL:
        return canonical(messages)
    try:
        fn = VARIANTS[name]
    except KeyError as exc:
        known = [CANONICAL, *VARIANTS]
        raise KeyError(f"unknown perturbation variant {name!r}; expected one of {known}") from exc
    return fn(messages)


def _normalize_ws(text: str) -> str:
    """Collapse every run of whitespace to a single space (for meaning checks)."""
    return " ".join(text.split())


def meaning_preserving_tokens(response: str, dimension: str) -> list[str]:
    """The semantic content a benign variant must never drop for one judge call.

    The tutor response under evaluation plus the dimension's definition and full
    scoring rubric body (which carries the ordinal level labels). Scaffolding
    headers are deliberately excluded — those are exactly what ``rubric_synonyms``
    is allowed to reword.
    """
    tokens = [t for t in (response, DEFINITIONS[dimension], RUBRICS[dimension]) if t]
    return tokens


def assert_meaning_preserved(
    perturbed: Sequence[dict[str, str]], *, response: str, dimension: str
) -> bool:
    """Assert the response, definition and rubric survive a perturbation.

    Comparison is whitespace-insensitive so the ``whitespace`` variant (which
    injects blank lines and stray spaces) is not flagged. Raises
    :class:`AssertionError` naming what went missing; returns ``True`` on success.
    """
    blob = _normalize_ws(" ".join(m["content"] for m in perturbed))
    missing = [
        tok
        for tok in meaning_preserving_tokens(response, dimension)
        if _normalize_ws(tok) and _normalize_ws(tok) not in blob
    ]
    if missing:
        preview = ", ".join(repr(m[:40]) for m in missing)
        raise AssertionError(
            f"perturbation dropped meaning-bearing content for {dimension!r}: {preview}"
        )
    return True
