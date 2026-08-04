"""Scoring base class and implementations."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, ClassVar

from olmo_eval.common.types import Instance, LMOutput


@dataclass(frozen=True)
class Scorer(ABC):
    """Abstract base class for scoring individual outputs.

    Subclasses must define:
        - name: str class attribute identifying the scorer
        - score(): method to compute score for an instance/output pair

    For scorers requiring async execution or a special runtime (e.g., sandboxed
    code execution or process-backed CPU work), extend the appropriate scorer
    subclass instead of implementing `score()` directly.
    """

    name: str = ""
    requires_async: ClassVar[bool] = False

    def __call__(self) -> Scorer:
        """Allow scorer instances to be used where classes are expected.

        When a metric's ``scorer`` field holds an instance instead of a class,
        calling ``metric.scorer()`` returns the instance itself.
        """
        return self

    @abstractmethod
    def score(self, instance: Instance, output: LMOutput) -> float:
        """Score a single output against the gold answer."""
        ...

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        return {"type": self.__class__.__name__, **asdict(self)}


@dataclass(frozen=True)
class ProcessScorer(Scorer):
    """Base class for CPU-heavy scorers that should run in a subprocess."""

    requires_async: ClassVar[bool] = True
    process_pool_name: ClassVar[str] = "cpu"

    def score(self, instance: Instance, output: LMOutput) -> float:
        raise RuntimeError(
            f"{self.__class__.__name__} requires a process scoring runtime. "
            "Ensure the task runner provides a ScoringContext with a process pool manager."
        )

    @abstractmethod
    def process_score(self, instance: Instance, output: LMOutput) -> float:
        """Score a single output in a subprocess."""
        ...


@dataclass(frozen=True, slots=True)
class ExactMatchScorer(Scorer):
    """Score 1.0 if extracted answer exactly matches gold, else 0.0."""

    name: str = "exact_match"
    case_sensitive: bool = False
    strip_whitespace: bool = True

    def score(self, instance: Instance, output: LMOutput) -> float:
        if instance.gold_answer is None or output.extracted_answer is None:
            return 0.0
        gold = instance.gold_answer
        pred = str(output.extracted_answer)
        if self.strip_whitespace:
            gold, pred = gold.strip(), pred.strip()
        if not self.case_sensitive:
            gold, pred = gold.lower(), pred.lower()
        return 1.0 if gold == pred else 0.0


@dataclass(frozen=True, slots=True)
class MultipleChoiceScorer(Scorer):
    """Score multiple choice by comparing selected index/letter."""

    name: str = "multiple_choice"

    def score(self, instance: Instance, output: LMOutput) -> float:
        if instance.gold_answer is None or output.extracted_answer is None:
            return 0.0
        # Normalize to uppercase letter
        gold = str(instance.gold_answer).strip().upper()
        pred = str(output.extracted_answer).strip().upper()
        return 1.0 if gold == pred else 0.0


def _normalize_text(text: str) -> str:
    """Normalize text for F1 computation by lowercasing and tokenizing."""
    import string

    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Normalize whitespace
    text = " ".join(text.split())
    return text


def _compute_f1(pred: str, gold: str) -> float:
    """Compute token-level F1 score between prediction and gold."""
    pred_tokens = _normalize_text(pred).split()
    gold_tokens = _normalize_text(gold).split()

    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0

    common = set(pred_tokens) & set(gold_tokens)
    num_same = sum(min(pred_tokens.count(t), gold_tokens.count(t)) for t in common)

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


@dataclass(frozen=True, slots=True)
class F1Scorer(Scorer):
    """Score using token-level F1 between prediction and gold answer."""

    name: str = "f1"

    def score(self, instance: Instance, output: LMOutput) -> float:
        if instance.gold_answer is None or output.extracted_answer is None:
            return 0.0
        return _compute_f1(str(output.extracted_answer), str(instance.gold_answer))


def _squad_normalize_answer(text: str) -> str:
    """Normalize text using SQuAD-style normalization.

    Lowercases, removes punctuation, removes articles (a, an, the),
    and normalizes whitespace. Matches the standard SQuAD evaluation script.
    """
    import re
    import string

    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = " ".join(text.split())
    return text


def _compute_squad_f1(pred: str, gold: str) -> float:
    """Compute token-level F1 using SQuAD-style normalization."""
    from collections import Counter

    pred_tokens = _squad_normalize_answer(pred).split()
    gold_tokens = _squad_normalize_answer(gold).split()

    if not gold_tokens or not pred_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


@dataclass(frozen=True, slots=True)
class SQuADF1Scorer(Scorer):
    """Score using SQuAD-style F1: max token-level F1 over all reference answers.

    Uses metadata["all_answers"] for multiple references. Falls back to
    instance.gold_answer if metadata is not present.
    """

    name: str = "f1"

    def score(self, instance: Instance, output: LMOutput) -> float:
        if output.extracted_answer is None:
            return 0.0
        pred = str(output.extracted_answer)
        all_answers = instance.metadata.get("all_answers", [])
        if not all_answers:
            if instance.gold_answer is None:
                return 0.0
            all_answers = [instance.gold_answer]
        return max(_compute_squad_f1(pred, ref) for ref in all_answers)


@dataclass(frozen=True, slots=True)
class SQuADExactMatchScorer(Scorer):
    """Score using SQuAD-style exact match over all reference answers.

    The exact-match companion to :class:`SQuADF1Scorer`, sharing its normalization
    and its multi-reference handling so the two report on the same footing. Plain
    :class:`ExactMatchScorer` would disagree with the F1 it sits beside, since it
    compares raw strings and so penalizes differences in articles, punctuation and
    casing that SQuAD scoring is defined to ignore.

    Uses metadata["all_answers"] for multiple references. Falls back to
    instance.gold_answer if metadata is not present.
    """

    name: str = "squad_exact_match"

    def score(self, instance: Instance, output: LMOutput) -> float:
        if output.extracted_answer is None:
            return 0.0
        pred = _squad_normalize_answer(str(output.extracted_answer))
        all_answers = instance.metadata.get("all_answers", [])
        if not all_answers:
            if instance.gold_answer is None:
                return 0.0
            all_answers = [instance.gold_answer]
        return 1.0 if any(pred == _squad_normalize_answer(str(ref)) for ref in all_answers) else 0.0


@dataclass(frozen=True, slots=True)
class ContainmentScorer(Scorer):
    """Score 1.0 if any reference answer occurs as a substring of the generation.

    Implements PopQA's published metric (Mallen et al. 2023, section 3.1): "We
    mark a prediction as correct if any substring of the prediction is an exact
    match of any of the gold answers."

    The lenient member of the SQuAD-normalized family, sharing normalization and
    multi-reference handling with :class:`SQuADF1Scorer` and
    :class:`SQuADExactMatchScorer`. It exists for short-answer factual tasks
    where a base model answers correctly but conversationally -- "The capital of
    France is Paris" against a gold of "Paris" is exact-match 0 and F1 0.33
    despite being right.

    Two known false positives come with the published rule, and are kept
    deliberately so scores stay comparable to the paper:

    - Matching is on raw substrings, not token boundaries. Several PopQA answer
      sets list short aliases -- "pol" for politician, "pop" for pop music -- so
      a prediction of "policy" or "popular" scores correct.
    - A model that hedges by listing candidates ("Paris, London, Rome") scores
      correct because one of them hits.

    Both inflate the score, so read it beside a stricter metric rather than
    alone. :class:`SQuADExactMatchScorer` is the intended partner.

    Uses metadata["all_answers"] for multiple references. Falls back to
    instance.gold_answer if metadata is not present.
    """

    name: str = "containment"

    def score(self, instance: Instance, output: LMOutput) -> float:
        if output.extracted_answer is None:
            return 0.0
        pred = _squad_normalize_answer(str(output.extracted_answer))
        if not pred:
            return 0.0
        all_answers = instance.metadata.get("all_answers", [])
        if not all_answers:
            if instance.gold_answer is None:
                return 0.0
            all_answers = [instance.gold_answer]
        for ref in all_answers:
            gold = _squad_normalize_answer(str(ref))
            if gold and gold in pred:
                return 1.0
        return 0.0


@dataclass(frozen=True, slots=True)
class WindowedContainmentScorer(Scorer):
    """Case-insensitive substring match within a leading window of the output.

    Reproduces the metric Co-LMLM (arXiv:2607.07707, appendix A.6) reports for
    TriviaQA: "whether any alias in the set of gold answers appears
    (case-insensitive) within the first 100 characters of the model output".
    Their ``score_popqa.py`` is the reference implementation::

        window = continuation[:_ANSWER_WINDOW_CHARS].lower()
        return any(ans.lower() in window for ans in possible_answers)

    Two things separate this from :class:`ContainmentScorer`, and both exist to
    reproduce the published rule rather than improve on it:

    - Lowercasing is the only normalization. Punctuation and articles are left
      alone, so a gold of "the Beatles" does not match a prediction of
      "Beatles" -- where every SQuAD-normalized scorer would.
    - Only the first ``window_chars`` characters are searched, so an answer
      that surfaces after a long preamble does not count.

    The paper calls this "Exact Match". It is not: it is substring containment,
    and it scores strictly higher than exact match on the same outputs. Pair it
    with a strict scorer to see how far apart they are.

    Uses metadata["all_answers"] for multiple references. Falls back to
    instance.gold_answer if metadata is not present.
    """

    name: str = "windowed_containment"
    window_chars: int = 100

    def score(self, instance: Instance, output: LMOutput) -> float:
        if output.extracted_answer is None:
            return 0.0
        window = str(output.extracted_answer)[: self.window_chars].lower()
        if not window:
            return 0.0
        all_answers = instance.metadata.get("all_answers", [])
        if not all_answers:
            if instance.gold_answer is None:
                return 0.0
            all_answers = [instance.gold_answer]
        for ref in all_answers:
            # An empty reference would match everything. The upstream code does
            # not guard against this; a blank alias is a data artifact rather
            # than an answer, so treat it as no reference at all.
            gold = str(ref).strip().lower()
            if gold and gold in window:
                return 1.0
        return 0.0


@dataclass(frozen=True, slots=True)
class BitsPerByteScorer(Scorer):
    """Compute bits per byte from logprobs.

    Bits per byte is a measure of language model performance that normalizes
    perplexity by the number of bytes in the text, making it comparable across
    different tokenizers and vocabularies.

    Formula: bits_per_byte = -sum(logprobs) / (num_bytes * log(2))
    """

    name: str = "bits_per_byte"

    def score(self, instance: Instance, output: LMOutput) -> float:
        if output.logprobs is None:
            return 0.0

        total_logprob = sum(tok.get("logprob", 0.0) for tok in output.logprobs)

        num_bytes = len(output.text.encode("utf-8")) if output.text else 0

        if num_bytes == 0:
            return 0.0

        bits_per_byte = -total_logprob / (num_bytes * math.log(2))

        return bits_per_byte


@dataclass(frozen=True, slots=True)
class PerplexityScorer(Scorer):
    """Compute perplexity from logprobs.

    Perplexity measures how well a language model predicts a sequence,
    defined as exp(-average_logprob).
    """

    name: str = "perplexity"

    def score(self, instance: Instance, output: LMOutput) -> float:
        if output.logprobs is None:
            return 0.0

        logprobs = [tok["logprob"] for tok in output.logprobs if "logprob" in tok]

        if not logprobs:
            return 0.0

        avg_logprob = sum(logprobs) / len(logprobs)
        perplexity = math.exp(-avg_logprob)

        return perplexity


@dataclass(frozen=True, slots=True)
class LogprobScorer(Scorer):
    """Compute total logprob for a sequence.

    This returns the sum of all token logprobs, useful for comparing
    continuation likelihoods.
    """

    name: str = "logprob"

    def score(self, instance: Instance, output: LMOutput) -> float:
        if output.logprobs is None:
            return float("-inf")

        logprobs = [tok["logprob"] for tok in output.logprobs if "logprob" in tok]

        if not logprobs:
            return float("-inf")

        return sum(logprobs)


@dataclass(frozen=True, slots=True)
class ExactMatchFlexScorer(Scorer):
    """Flexible exact match that checks ANY extracted answer against ANY gold answer.

    This scorer is useful for math tasks where multiple equivalent representations
    of the answer might exist. It checks if any of the extracted answers matches
    any of the gold answers.

    Expects:
        - instance.metadata["all_gold_answers"]: list of acceptable gold answers
        - output.metadata["all_extracted_answers"]: list of extracted answers from model output

    Falls back to standard exact match if these metadata fields are not present.
    """

    name: str = "exact_match_flex"
    case_sensitive: bool = False
    remove_whitespace: bool = True

    def _normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        if not self.case_sensitive:
            text = text.lower()
        if self.remove_whitespace:
            text = "".join(text.split())
        return text

    def score(self, instance: Instance, output: LMOutput) -> float:
        # Get all gold answers
        all_gold = instance.metadata.get("all_gold_answers", [])
        if not all_gold and instance.gold_answer is not None:
            all_gold = [instance.gold_answer]

        # Get all extracted answers
        all_extracted = (output.metadata or {}).get("all_extracted_answers", [])
        if not all_extracted and output.extracted_answer is not None:
            all_extracted = [output.extracted_answer]

        if not all_gold or not all_extracted:
            return 0.0

        # Check if any extracted answer matches any gold answer
        normalized_gold = {self._normalize(str(g)) for g in all_gold}
        for extracted in all_extracted:
            if self._normalize(str(extracted)) in normalized_gold:
                return 1.0

        return 0.0


@dataclass(frozen=True, slots=True)
class MinervaMathScorer(ProcessScorer):
    """Flexible math equivalence: any extracted answer vs any gold, using sympy + Hendrycks.

    Matches oe-eval Minerva MATH behavior: try sympy (minerva_is_equiv) then
    Hendrycks string normalization. Expects instance.metadata["all_gold_answers"]
    and output.metadata["all_extracted_answers"]; falls back to single gold/extracted.
    """

    name: str = "minerva_math_flex"

    def process_score(self, instance: Instance, output: LMOutput) -> float:
        from olmo_eval.evals.extract.math import is_equiv

        all_gold = instance.metadata.get("all_gold_answers", [])
        if not all_gold and instance.gold_answer is not None:
            all_gold = [instance.gold_answer]

        all_extracted = (output.metadata or {}).get("all_extracted_answers", [])
        if not all_extracted and output.extracted_answer is not None:
            all_extracted = [output.extracted_answer]

        if not all_gold or not all_extracted:
            return 0.0

        for extracted in all_extracted:
            for gold in all_gold:
                if is_equiv(str(extracted).strip(), str(gold).strip()):
                    return 1.0
        return 0.0


@dataclass(frozen=True, slots=True)
class MathVerifyScorer(Scorer):
    """Score math answers using symbolic verification via math_verify library.

    This scorer uses the math_verify package to check if the extracted answer
    is mathematically equivalent to the gold answer, handling various
    representations of mathematical expressions.

    Falls back to exact string matching if math_verify is not available.
    """

    name: str = "math_verify"
    timeout: float = 5.0

    def score(self, instance: Instance, output: LMOutput) -> float:
        if instance.gold_answer is None or output.extracted_answer is None:
            return 0.0

        gold = str(instance.gold_answer)
        pred = str(output.extracted_answer)

        # Try using math_verify first (optional dependency)
        try:
            from math_verify import verify

            result = verify(gold, pred)
            return 1.0 if result else 0.0
        except ImportError:
            pass
        except Exception:
            pass

        # Fall back to our internal equivalence check
        try:
            from olmo_eval.evals.extract.math import is_equiv

            return 1.0 if is_equiv(pred, gold) else 0.0
        except ImportError:
            pass

        # Last resort: exact string match (normalized)
        gold_norm = "".join(gold.lower().split())
        pred_norm = "".join(pred.lower().split())
        return 1.0 if gold_norm == pred_norm else 0.0
