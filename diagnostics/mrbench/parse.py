"""Parse the judge's ``Feedback: ... [RESULT] N`` output into a score and label."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from .judge_prompt import SCORE_TO_LABEL

# Primary: the paper's documented format, e.g. "[RESULT] 2".
_RESULT_RE = re.compile(r"\[RESULT\]\s*([123])\b")
# Fallback: a trailing standalone score if the tag is malformed but a 1-3 is present.
_FALLBACK_RE = re.compile(r"\b([123])\b(?!.*\b[123]\b)", re.DOTALL)
_FEEDBACK_RE = re.compile(r"Feedback:\s*(.*?)\s*\[RESULT\]", re.DOTALL)


@dataclass(frozen=True)
class ParsedJudgement:
    """Outcome of parsing one judge completion for one dimension."""

    score: int | None
    label: str | None
    feedback: str | None
    ok: bool
    used_fallback: bool
    raw: str

    @property
    def parse_failed(self) -> bool:
        return not self.ok


def parse_result(text: str, dimension: str) -> ParsedJudgement:
    """Extract (score, label, feedback) from a judge completion.

    ``ok`` is False when no score in {1,2,3} can be recovered; the raw text is
    always retained so parse failures can be inspected rather than silently
    dropped.
    """
    raw = text or ""
    feedback_match = _FEEDBACK_RE.search(raw)
    feedback = feedback_match.group(1).strip() if feedback_match else None

    used_fallback = False
    match = _RESULT_RE.search(raw)
    if match is None:
        match = _FALLBACK_RE.search(raw)
        used_fallback = match is not None

    if match is None:
        return ParsedJudgement(None, None, feedback, ok=False, used_fallback=False, raw=raw)

    score = int(match.group(1))
    label = SCORE_TO_LABEL[dimension][score]
    return ParsedJudgement(score, label, feedback, ok=True, used_fallback=used_fallback, raw=raw)


# --------------------------------------------------------------------------- #
# OPT-IN: self-consistency (k-sample majority vote). Default behaviour is k=1,
# i.e. a single call parsed by ``parse_result``; callers only reach this code
# when they explicitly sample k>1 times. Judge-agnostic.
# --------------------------------------------------------------------------- #
def majority_vote(scores: Sequence[int | None]) -> int | None:
    """Return the majority score in {1,2,3}; deterministic tie-break = smallest.

    ``None`` entries (parse failures) are ignored. Returns ``None`` only when no
    sample produced a valid score.
    """
    valid = [s for s in scores if s is not None]
    if not valid:
        return None
    counts = Counter(valid)
    top = max(counts.values())
    return min(score for score, count in counts.items() if count == top)


def aggregate_samples(
    raws: Sequence[str], dimension: str
) -> tuple[ParsedJudgement, list[ParsedJudgement]]:
    """Parse k judge outputs and majority-vote them into one ParsedJudgement.

    Returns ``(combined, per_sample)``. ``combined`` carries the voted score/label
    (or ``ok=False`` if no sample parsed); ``per_sample`` is every individual parse
    so callers can record raw sample outputs. With a single input this reproduces
    ``parse_result`` exactly.
    """
    per_sample = [parse_result(raw, dimension) for raw in raws]
    winner = majority_vote([s.score for s in per_sample if s.ok])
    if winner is None:
        first_feedback = per_sample[0].feedback if per_sample else None
        first_raw = per_sample[0].raw if per_sample else ""
        combined = ParsedJudgement(
            None, None, first_feedback, ok=False, used_fallback=False, raw=first_raw
        )
        return combined, per_sample

    winning = next(s for s in per_sample if s.ok and s.score == winner)
    combined = ParsedJudgement(
        score=winner,
        label=SCORE_TO_LABEL[dimension][winner],
        feedback=winning.feedback,
        ok=True,
        used_fallback=winning.used_fallback,
        raw=winning.raw,
    )
    return combined, per_sample
