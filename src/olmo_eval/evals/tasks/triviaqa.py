"""TriviaQA parametric fact-recall (generative EM), for the SmolLM2-135M base/split A/B.

Greedy generation with no context; scored by the Co-LMLM fact-recall EM (arXiv:2607.07707,
App. A.6, ported from ``colmlm/eval_fact_recall.py``): a prediction is correct iff any gold alias
appears (case-insensitive) within the first 100 characters of the output. This measures
*parametric* recall (no retrieval), where the expectation is base >> split.

Also defines the shared fact-recall scorers (``AliasExactMatchScorer`` for TriviaQA/PopQA and
``TrexExactMatchScorer`` for T-REx) imported by the sibling fact-recall tasks.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from olmo_eval.common.metrics import AccuracyMetric
from olmo_eval.common.scorers import Scorer
from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, SamplingParams, Split
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import Task, register, register_variant

QA_CHAR_LIMIT = 100  # TriviaQA/PopQA: a gold alias must appear within the first 100 output chars
TREX_TOKEN_LIMIT = 5  # T-REx: the reference must appear within the first 5 content tokens


def _aliases(instance: Instance) -> list[str]:
    raw = instance.metadata.get("aliases") or instance.metadata.get("all_answers") or []
    return [str(a) for a in raw if str(a).strip()]


@dataclass(frozen=True, slots=True)
class AliasExactMatchScorer(Scorer):
    """1.0 iff any gold alias appears (case-insensitive) in the first ``QA_CHAR_LIMIT`` output chars."""

    name: str = "alias_em"

    def score(self, instance: Instance, output: LMOutput) -> float:
        if output.extracted_answer is None:
            return 0.0
        window = str(output.extracted_answer)[:QA_CHAR_LIMIT].lower()
        return 1.0 if any(a.lower() in window for a in _aliases(instance)) else 0.0


@dataclass(frozen=True, slots=True)
class TrexExactMatchScorer(Scorer):
    """1.0 iff the reference appears within the first ``TREX_TOKEN_LIMIT`` content tokens."""

    name: str = "trex_em"

    def score(self, instance: Instance, output: LMOutput) -> float:
        if output.extracted_answer is None:
            return 0.0
        content = " ".join(re.findall(r"\S+", str(output.extracted_answer))[:TREX_TOKEN_LIMIT]).lower()
        return 1.0 if any(a.lower() in content for a in _aliases(instance)) else 0.0


def _format_query(question: str) -> str:
    return f"Question: {question}\nAnswer:"


@register("triviaqa")
class TriviaQA(Task):
    data_source = DataSource(
        path="mandarjoshi/trivia_qa", subset="rc.nocontext", split="validation"
    )
    split = Split.VALIDATION
    metrics = (AccuracyMetric(scorer=AliasExactMatchScorer),)
    primary_metric = AccuracyMetric(scorer=AliasExactMatchScorer)
    sampling_params = SamplingParams(
        max_tokens=32, temperature=0.0, stop_sequences=("\n", "Question:", "Q:")
    )
    num_fewshot = 0
    fewshot_split = "train"

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached()

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        question = str(doc.get("question", "")).strip()
        answer = doc.get("answer", {}) or {}
        aliases = [
            str(a)
            for a in (answer.get("aliases") or answer.get("normalized_aliases") or [])
            if str(a).strip()
        ]
        if not aliases and answer.get("value"):
            aliases = [str(answer["value"])]
        if not question or not aliases:
            return None
        return Instance(
            question=_format_query(question),
            gold_answer=aliases[0],
            metadata={
                "id": doc.get("question_id", index),
                "index": index,
                "aliases": aliases,
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        parts = [f"{ex.question} {ex.gold_answer}" for ex in self.get_fewshot()]
        parts.append(instance.question)
        return LMRequest(request_type=RequestType.COMPLETION, prompt="\n\n".join(parts))

    def extract_answer(self, output: LMOutput) -> str:
        return output.text.strip()


register_variant("triviaqa", "gen")
register_variant("triviaqa", "5shot", num_fewshot=5, fewshot_split="train")
