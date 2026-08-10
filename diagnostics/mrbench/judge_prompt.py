"""Byte-faithful reproduction of the paper's Figure-6 LLM-as-critic prompt.

Source: Maurya et al., NAACL 2025, Appendix E / Figure 6 ("Prompt Template,
Dimension Definitions, and Rubric for LLM-based Evaluation"). The template is
adapted by the authors from the Prometheus2 guidelines (Kim et al., 2024).

The protocol is **per-dimension absolute scoring**: one judge call per (response,
dimension), each emitting ``Feedback: ... [RESULT] N`` with ``N in {1, 2, 3}``.

Fidelity notes (flagged, not silently decided):
  * REVEALING_ANSWER's "Score 1" line is reproduced exactly as printed in the
    paper PDF, which is missing its closing parenthesis
    ("...revealed answer is correct"). This may be a paper/PDF typo; it is kept
    byte-faithful and isolated here so it is trivial to correct if the user
    prefers. See ``KNOWN_PAPER_QUIRKS``.
  * Curly apostrophes in the PDF are normalised to ASCII (') for encoding
    safety. This is the only character-level normalisation applied.
"""

from __future__ import annotations

import os

# --- Figure 6: system + user template -------------------------------------

# The paper's "### System:" block.
SYSTEM_PROMPT = (
    "You are a critic evaluating a tutor's interaction with a student, responsible "
    "for providing a clear and objective single evaluation score based on specific "
    "criteria. Each assessment must accurately reflect the absolute performance "
    "standards."
)

# The paper's "### User:" block, with {history}, {definition}, {rubric},
# {response} placeholders. The trailing "# Generate Assessment Score:" line is the
# paper's "### Assistant:" generation cue; OpenAI chat has no separate assistant-
# priming slot, so it is folded onto the end of the user turn (flagged decision).
USER_TEMPLATE = """# Task Description: The assessment of the ###Tutor Response should be based on the following: ###Previous Conversation between Tutor and Student, ###Definitions of criteria and
# Scoring Rubric.
(1). Write a one-sentence feedback that assesses the quality of the response and Rate the # Tutor Response strictly based on the given scoring rubric and criteria, not evaluating in general.
(2). After writing feedback, write a score that is an integer between 1 and 3. You should refer to the scoring rubric.
(3). The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 3)"
(4). Please do not generate other opening, closing, or explanations.
# Previous Conversation between Tutor and Student: {history}
# Definitions of criteria: {definition}
# Scoring Rubric: {rubric}
# Tutor Response: {response}
# Generate Assessment Score:"""

# The assistant generation cue that was folded into USER_TEMPLATE (kept as a named
# constant so a caller could instead emit it as an assistant-role priming message
# on a backend that supports prefill).
ASSISTANT_GEN_CUE = "# Generate Assessment Score:"

KNOWN_PAPER_QUIRKS = (
    "REVEALING_ANSWER Score 1 line is missing its closing parenthesis, reproduced "
    "verbatim from the paper PDF (Figure 6).",
)

# --- Per-dimension definitions (Figure 6 `definition` dict) ----------------
# Keyed by our canonical dimension name (matching reference.DIMENSIONS).
DEFINITIONS: dict[str, str] = {
    "Mistake_Identification": "Has the tutor identified a mistake in a student's response?",
    "Mistake_Location": (
        "Does the tutor's response accurately point to a genuine mistake and its location?"
    ),
    "Revealing_of_the_Answer": "Does the tutor reveal the final answer (whether correct or not)?",
    "Providing_Guidance": (
        "Does the tutor offer correct and relevant guidance, such as an explanation, "
        "elaboration, hint, examples, and so on?"
    ),
    "Actionability": "Is it clear from the tutor's feedback what the student should do next?",
    "Coherence": (
        "Is the tutor's response logically consistent with the student's previous response?"
    ),
    "Tutor_Tone": "Is the tutor's response encouraging, neutral, or offensive?",
    "Humanlikeness": "Does the tutor's response sound natural, rather than robotic or artificial?",
}

# --- Per-dimension scoring rubrics (Figure 6, byte-faithful) ---------------
RUBRICS: dict[str, str] = {
    "Mistake_Identification": (
        "[Has the tutor identified a mistake in a student's response?]\n"
        "Score 1: Yes\n"
        "Score 2: To some extent\n"
        "Score 3: No"
    ),
    "Mistake_Location": (
        "[Does the tutor's response accurately point to a genuine mistake and its location?]\n"
        "Score 1: Yes\n"
        "Score 2: To some extent\n"
        "Score 3: No"
    ),
    # NOTE: Score 1 line is missing its closing paren in the paper; kept verbatim.
    "Revealing_of_the_Answer": (
        "[Does the tutor reveal the final answer (whether correct or not)]\n"
        "Score 1: Yes (and the revealed answer is correct\n"
        "Score 2: Yes (but the revealed answer is incorrect)\n"
        "Score 3: No"
    ),
    "Providing_Guidance": (
        "[Does the tutor offer correct and relevant guidance, such as an explanation, "
        "elaboration, hint, examples, and so on?]\n"
        "Score 1: Yes (guidance is correct and relevant to the mistake)\n"
        "Score 2: To some extent (guidance is provided but it is fully or partially "
        "incorrect or incomplete)\n"
        "Score 3: No"
    ),
    "Actionability": (
        "[Is it clear from the tutor's feedback what the student should do next?]\n"
        "Score 1: Yes\n"
        "Score 2: To some extent\n"
        "Score 3: No"
    ),
    "Coherence": (
        "[Is the tutor's response logically consistent with the student's previous response?]\n"
        "Score 1: Yes\n"
        "Score 2: To some extent\n"
        "Score 3: No"
    ),
    "Tutor_Tone": (
        "[Is the tutor's response encouraging, neutral, or offensive?]\n"
        "Score 1: Encouraging\n"
        "Score 2: Neutral\n"
        "Score 3: Offensive"
    ),
    "Humanlikeness": (
        "[Does the tutor's response sound natural rather than robotic or artificial?]\n"
        "Score 1: Yes\n"
        "Score 2: To some extent\n"
        "Score 3: No"
    ),
}

# --- Score <-> label maps (ordinal 1..3 per Figure 6 rubric) ---------------
# Canonical label per (dimension, score). Labels for score-1 in the desired
# position match reference.DESIRED_LABELS so judge-DAMR can compare directly.
SCORE_TO_LABEL: dict[str, dict[int, str]] = {
    "Mistake_Identification": {1: "Yes", 2: "To some extent", 3: "No"},
    "Mistake_Location": {1: "Yes", 2: "To some extent", 3: "No"},
    "Revealing_of_the_Answer": {
        1: "Yes (and the answer is correct)",
        2: "Yes (but the answer is incorrect)",
        3: "No",
    },
    "Providing_Guidance": {1: "Yes", 2: "To some extent", 3: "No"},
    "Actionability": {1: "Yes", 2: "To some extent", 3: "No"},
    "Coherence": {1: "Yes", 2: "To some extent", 3: "No"},
    "Tutor_Tone": {1: "Encouraging", 2: "Neutral", 3: "Offensive"},
    "Humanlikeness": {1: "Yes", 2: "To some extent", 3: "No"},
}

# Gold (human) label string -> ordinal score, matching the Figure-6 rubric order.
# Keys are the exact label strings found in MRBench_V1.json.
GOLD_LABEL_TO_SCORE: dict[str, dict[str, int]] = {
    "Mistake_Identification": {"Yes": 1, "To some extent": 2, "No": 3},
    "Mistake_Location": {"Yes": 1, "To some extent": 2, "No": 3},
    "Revealing_of_the_Answer": {
        "Yes (and the answer is correct)": 1,
        "Yes (but the answer is incorrect)": 2,
        "No": 3,
    },
    "Providing_Guidance": {"Yes": 1, "To some extent": 2, "No": 3},
    "Actionability": {"Yes": 1, "To some extent": 2, "No": 3},
    "Coherence": {"Yes": 1, "To some extent": 2, "No": 3},
    "Tutor_Tone": {"Encouraging": 1, "Neutral": 2, "Offensive": 3},
    "Humanlikeness": {"Yes": 1, "To some extent": 2, "No": 3},
}


# --- OPT-IN: reference-guided judging (default OFF) ------------------------ #
# Off-protocol accuracy lever: inject the conversation's Ground_Truth_Solution
# into the judge prompt for the *correctness-linked* dimensions only. These four
# are the ones whose rubric turns on whether the tutor's claim about the maths is
# right; tone / coherence / human-likeness are style judgements that the solution
# does not inform, so they are deliberately excluded (Actionability is a candidate
# but is left out to match the secondary set documented in Plan/mrbench/README.md).
# Default build_messages output is byte-identical to the paper's Figure 6; the
# reference block only appears when reference_guided=True AND a solution is given
# AND the dimension is correctness-linked.
REFERENCE_GUIDED_DIMENSIONS: frozenset[str] = frozenset(
    {
        "Mistake_Identification",
        "Mistake_Location",
        "Revealing_of_the_Answer",
        "Providing_Guidance",
    }
)

# Env var that flips the default at call sites (the function arg still wins).
_REFERENCE_GUIDED_ENV = "MRBENCH_JUDGE_REFERENCE_GUIDED"
_REFERENCE_LINE_PREFIX = "# Reference Solution (ground truth for correctness-linked dimensions): "
_TUTOR_RESPONSE_MARKER = "# Tutor Response:"


def reference_guided_default() -> bool:
    """Resolve the reference-guided default from the environment (OFF unless set)."""
    return os.environ.get(_REFERENCE_GUIDED_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _inject_reference(user: str, reference_solution: str) -> str:
    """Insert the reference-solution block just before the Tutor Response section."""
    block = _REFERENCE_LINE_PREFIX + reference_solution + "\n"
    return user.replace(_TUTOR_RESPONSE_MARKER, block + _TUTOR_RESPONSE_MARKER, 1)


def build_messages(
    history: str,
    response: str,
    dimension: str,
    *,
    reference_solution: str | None = None,
    reference_guided: bool = False,
) -> list[dict[str, str]]:
    """Return the OpenAI chat messages for one (response, dimension) judge call.

    ``### System:`` -> a ``system`` message; the ``### User:`` template (with the
    dimension's definition + rubric filled in) -> a ``user`` message.

    With ``reference_guided=False`` (the default) the output is byte-identical to
    the paper's Figure 6. With ``reference_guided=True`` and a non-empty
    ``reference_solution``, the solution is injected for correctness-linked
    dimensions only (see ``REFERENCE_GUIDED_DIMENSIONS``); all other dimensions
    remain byte-identical to Figure 6 even when the flag is on.
    """
    if dimension not in DEFINITIONS:
        raise KeyError(f"Unknown dimension: {dimension!r}")
    user = USER_TEMPLATE.format(
        history=history,
        definition=DEFINITIONS[dimension],
        rubric=RUBRICS[dimension],
        response=response,
    )
    if reference_guided and reference_solution and dimension in REFERENCE_GUIDED_DIMENSIONS:
        user = _inject_reference(user, reference_solution)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
