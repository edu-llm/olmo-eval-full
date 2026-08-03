"""PopQA long-tail parametric fact-recall (generative EM), for the SmolLM2-135M base/split A/B.

Same protocol as TriviaQA (greedy, alias-EM within the first 100 output chars; see
``triviaqa.py`` / ``colmlm/eval_fact_recall.py``), restricted to the long-tail subset used by
Co-LMLM (arXiv:2607.07707): entities with fewer than 100 monthly Wikipedia page views
(``s_pop < 100``), where a memorization-vs-externalization gap is clearest.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from olmo_eval.common.metrics import AccuracyMetric
from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, SamplingParams, Split
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import Task, register, register_variant
from olmo_eval.evals.tasks.triviaqa import AliasExactMatchScorer, _format_query

_LONG_TAIL_MAX_POPULARITY = 100  # keep entities with < 100 monthly Wikipedia page views


def _parse_answers(raw: Any) -> list[str]:
    """PopQA stores ``possible_answers`` as a JSON-encoded list (occasionally already a list)."""
    if isinstance(raw, (list, tuple)):
        values = list(raw)
    elif isinstance(raw, str) and raw.strip():
        try:
            values = json.loads(raw)
        except json.JSONDecodeError:
            values = [raw]
    else:
        values = []
    return [str(a) for a in values if str(a).strip()]


@register("popqa")
class PopQA(Task):
    data_source = DataSource(path="akariasai/PopQA", split="test")
    split = Split.TEST
    metrics = (AccuracyMetric(scorer=AliasExactMatchScorer),)
    primary_metric = AccuracyMetric(scorer=AliasExactMatchScorer)
    sampling_params = SamplingParams(
        max_tokens=32, temperature=0.0, stop_sequences=("\n", "Question:", "Q:")
    )
    num_fewshot = 0
    fewshot_split = "test"

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached()

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        # Long-tail subset only: skip popular entities (>= 100 monthly page views).
        popularity = doc.get("s_pop")
        if popularity is not None and float(popularity) >= _LONG_TAIL_MAX_POPULARITY:
            return None

        question = str(doc.get("question", "")).strip()
        aliases = _parse_answers(doc.get("possible_answers"))
        if not question or not aliases:
            return None
        return Instance(
            question=_format_query(question),
            gold_answer=aliases[0],
            metadata={
                "id": doc.get("id", index),
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


register_variant("popqa", "gen")
