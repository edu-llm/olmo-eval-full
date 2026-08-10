"""Reference constants for the MRBench gold-DAMR sanity check.

Everything the Milestone-0 check needs to know about the MRBench taxonomy and the
NAACL-2025 paper's Table 3 lives here so the arithmetic in ``gold_damr.py`` stays
free of magic strings.

Paper: Maurya et al., "Unifying AI Tutor Evaluation: An Evaluation Taxonomy for
Pedagogical Ability Assessment of LLM-Powered AI Tutors", NAACL 2025.
Dataset: MRBench V1 (CC BY-SA 4.0), github.com/kaushal0494/UnifyingAITutorEvaluation.
"""

from __future__ import annotations

# The eight pedagogical dimensions, in a fixed reporting order. The strings are
# the annotation keys as documented in the MRBench README schema.
DIMENSIONS: tuple[str, ...] = (
    "Mistake_Identification",
    "Mistake_Location",
    "Revealing_of_the_Answer",
    "Providing_Guidance",
    "Actionability",
    "Coherence",
    "Tutor_Tone",
    "Humanlikeness",
)

# Short column headers used when printing the table.
DIMENSION_HEADERS: dict[str, str] = {
    "Mistake_Identification": "MID",
    "Mistake_Location": "MLOC",
    "Revealing_of_the_Answer": "REVEAL",
    "Providing_Guidance": "GUID",
    "Actionability": "ACT",
    "Coherence": "COH",
    "Tutor_Tone": "TONE",
    "Humanlikeness": "HUMAN",
}

# The single label per dimension that counts as "desired" for DAMR. Only the exact
# label counts; partial credit ("To some extent") does NOT count.
DESIRED_LABELS: dict[str, str] = {
    "Mistake_Identification": "Yes",
    "Mistake_Location": "Yes",
    "Revealing_of_the_Answer": "No",
    "Providing_Guidance": "Yes",
    "Actionability": "Yes",
    "Coherence": "Yes",
    "Tutor_Tone": "Encouraging",
    "Humanlikeness": "Yes",
}

# Faithful, literal mapping from the paper's tutor names (rows of Table 3) to the
# keys used inside ``anno_llm_responses`` in MRBench_V1.json. This is a 1:1 rename
# only (e.g. "GPT-4" -> "GPT4"); no tutor is swapped for another. The identity of
# each key is confirmed by the README's own worked example, which stores the
# "GPT-4" tutor under the key "GPT4".
PAPER_TO_DATA_KEY: dict[str, str] = {
    "Novice": "Novice",
    "Expert": "Expert",
    "Llama-3.1-8B": "Llama318B",
    "Phi3": "Phi3",
    "Gemini": "Gemini",
    "Sonnet": "Sonnet",
    "Mistral": "Mistral",
    "GPT-4": "GPT4",
    "Llama-3.1-405B": "Llama31405B",
}

# Row order for reporting (mirrors the task's reproduction of Table 3).
PAPER_TUTOR_ORDER: tuple[str, ...] = (
    "Novice",
    "Expert",
    "Llama-3.1-8B",
    "Phi3",
    "Gemini",
    "Sonnet",
    "Mistral",
    "GPT-4",
    "Llama-3.1-405B",
)

# Paper Table 3 DAMR percentages, keyed by paper tutor name, in DIMENSIONS order.
TABLE3_REFERENCE: dict[str, tuple[float, ...]] = {
    "Novice": (43.33, 16.67, 80.00, 11.67, 1.67, 50.00, 90.00, 35.00),
    "Expert": (76.04, 63.02, 90.62, 67.19, 76.04, 79.17, 92.19, 87.50),
    "Llama-3.1-8B": (80.21, 54.69, 73.96, 45.31, 42.71, 80.73, 19.79, 93.75),
    "Phi3": (28.65, 26.04, 73.96, 17.71, 11.98, 39.58, 45.31, 52.08),
    "Gemini": (63.02, 39.58, 67.71, 37.50, 42.71, 56.77, 21.88, 68.23),
    "Sonnet": (85.42, 69.79, 94.79, 59.38, 60.94, 88.54, 54.69, 96.35),
    "Mistral": (93.23, 73.44, 86.46, 63.54, 70.31, 86.98, 15.10, 95.31),
    "GPT-4": (94.27, 84.38, 53.12, 76.04, 46.35, 90.17, 37.50, 89.62),
    "Llama-3.1-405B": (94.27, 84.38, 80.73, 77.08, 74.48, 91.67, 16.15, 90.62),
}

# Default per-cell tolerance (percentage points) for the PASS/FAIL verdict.
DEFAULT_TOLERANCE_PP: float = 1.0

# Known data-vs-paper divergence, established by the Milestone-0 gold-DAMR check
# (see gold_damr_check.py and Plan/mrbench/README.md). ACCEPTED decision: use the
# official MRBench_V1.json as-is; treat the three tutors below as caveated for
# strict paper-comparability.
TABLE3_DIVERGENCE = """\
Milestone-0 result: the official MRBench_V1.json does NOT fully reproduce the
paper's Table 3.
  * Reproduce within ~4pp (parsing/arithmetic validated): GPT-4, Sonnet, Mistral,
    Llama-3.1-8B, Llama-3.1-405B, Phi3 (6/9 tutors).
  * DO NOT reproduce:
      - Expert : Tutor_Tone 17% Encouraging vs paper 92% (7/8 other dims ~+6pp).
      - Gemini : uniform +20-27pp across all dimensions.
      - Novice : scored on 53 Bridge dialogues, not the paper's 60; tone off ~-35pp.
  * Published V1 split is 139 MathDial / 53 Bridge (1,589 responses); the paper
    reports 132 / 60 (1,596 responses).
No faithful tutor-key remapping fixes Expert/Gemini (same pattern in V2). The
published JSON differs from the paper's frozen Table-3 snapshot. Decision:
accept V1 as-is; caveat Expert, Gemini, and Novice for strict comparability.
"""
