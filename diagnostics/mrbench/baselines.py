"""Paper baseline AC (Annotation Correlation) scores for comparison.

From Maurya et al., NAACL 2025:
  * Table 5 — Prometheus2 as LLM critic.
  * Table 6 — Llama-3.1-8B as LLM critic.

Both tables report per-tutor, per-dimension Pearson correlation between the
critic LLM's judgements and the human gold annotations. Columns are in
``reference.DIMENSIONS`` order. Novice uses 60 Bridge dialogues; all other tutors
use all 192 (per the paper's footnote).
"""

from __future__ import annotations

# Table 5: Prometheus2 critic. Values in DIMENSIONS order
# (MID, MLOC, REVEAL, GUID, ACT, COH, TONE, HUMAN).
PROMETHEUS2_AC: dict[str, tuple[float, ...]] = {
    "Novice": (-0.37, 0.09, -0.56, -0.72, 0.15, -0.15, -0.71, 0.18),
    "Expert": (-0.01, -0.25, -0.13, -0.19, -0.08, -0.11, -0.40, 0.01),
    "Phi3": (-0.67, -0.58, -0.51, -0.51, -0.46, -0.33, -0.62, 0.03),
    "Llama-3.1-8B": (-0.12, -0.37, -0.17, 0.04, -0.07, -0.16, -0.29, 0.11),
    "Gemini": (0.02, 0.09, -0.06, -0.16, -0.12, -0.07, -0.24, 0.07),
    "Sonnet": (-0.11, -0.12, -0.21, -0.11, -0.22, -0.08, -0.20, 0.07),
    "Mistral": (-0.06, -0.11, -0.10, -0.23, -0.15, -0.20, -0.19, 0.06),
    "GPT-4": (-0.07, 0.01, -0.20, -0.21, 0.02, -0.02, -0.11, 0.08),
    "Llama-3.1-405B": (-0.03, -0.08, -0.05, -0.05, 0.00, 0.06, -0.13, 0.11),
}

# Table 6: Llama-3.1-8B critic. Same column order.
LLAMA31_8B_AC: dict[str, tuple[float, ...]] = {
    "Novice": (-0.42, 0.06, -0.71, -0.80, 0.17, -0.17, -0.77, 0.14),
    "Expert": (-0.03, -0.29, -0.17, -0.23, -0.10, -0.16, -0.49, -0.01),
    "Phi3": (-0.71, -0.67, -0.77, -0.73, -0.61, -0.41, -0.62, 0.04),
    "Llama-3.1-8B": (-0.08, -0.46, -0.17, 0.09, -0.09, -0.23, -0.38, 0.09),
    "Gemini": (0.06, 0.12, -0.11, -0.27, -0.22, -0.09, -0.34, 0.09),
    "Sonnet": (-0.07, -0.17, -0.26, -0.21, -0.29, -0.08, -0.32, 0.07),
    "Mistral": (-0.16, -0.16, -0.16, -0.34, -0.27, -0.28, -0.17, 0.07),
    "GPT-4": (-0.03, 0.01, -0.13, -0.23, 0.01, -0.05, -0.06, 0.10),
    "Llama-3.1-405B": (-0.01, -0.02, -0.01, -0.07, 0.00, 0.02, -0.06, 0.09),
}

BASELINES = {"Prometheus2": PROMETHEUS2_AC, "Llama-3.1-8B": LLAMA31_8B_AC}
