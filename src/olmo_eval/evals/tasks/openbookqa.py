"""OpenBookQA (4-way multiple-choice science QA), OLMES-style RC/MC/BPB variants.

Fills the one reasoning-suite gap for the SmolLM2-135M base/split A/B: the control archive's
core reasoning set is HellaSwag / PIQA / OpenBookQA, and only OpenBookQA was missing here.
Modeled on ``csqa.py`` / ``arc.py`` (question + fixed choices + a letter ``answerKey``).

The ``olmo3base`` variant is 5-shot rank-classification (per-char normalized accuracy), matching
the other reasoning tasks' ``olmo3base``. Few-shot examples are sampled from the ``train`` split
(there is no OLMES fixed-fewshot constant for OpenBookQA in this repo yet); add a
``constants/openbookqa.py`` fixed set + ``fewshot_source`` later for exact OLMES fidelity.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from olmo_eval.common.formatters import MultipleChoiceFormatter
from olmo_eval.common.metrics import (
    BPBMetricInstanceAvg,
    LogprobMCAccuracyMetric,
    LogprobPerCharMCAccuracyMetric,
)
from olmo_eval.common.types import Instance, LMRequest, RequestType, SamplingParams, Split
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import Task, register, register_variant
from olmo_eval.evals.tasks.common.format_helpers import format_mc as _format_mc
from olmo_eval.evals.tasks.common.format_helpers import format_rc as _format_rc

# OpenBookQA's answerKey is normally a letter (A-D); some mirrors use 1-4.
_NUM_TO_LETTER = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}


@register("openbookqa")
class OpenBookQA(Task):
    data_source = DataSource(path="allenai/openbookqa", subset="main", split="test")
    split = Split.TEST
    metrics = (LogprobMCAccuracyMetric(),)
    num_fewshot = 0
    fewshot_split = "train"
    sampling_params = SamplingParams(temperature=0.0)

    @property
    def instances(self) -> Iterator[Instance]:
        split = (
            self.config.data_source.split
            if isinstance(self.config.data_source, DataSource)
            else None
        )
        yield from self._load_instances_cached(split=split)

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        # OpenBookQA calls the stem "question_stem"; keep "question" as a fallback.
        question = doc.get("question_stem") or doc.get("question", "")
        if not question:
            return None

        choices_data = doc.get("choices", {})
        choices = choices_data.get("text", [])
        if not choices:
            return None

        answer_key = str(doc.get("answerKey", ""))
        letter = _NUM_TO_LETTER.get(answer_key, answer_key)
        gold_idx = ord(letter) - ord("A") if letter else 0
        gold_text = choices[gold_idx] if 0 <= gold_idx < len(choices) else ""

        return Instance(
            question=question,
            choices=tuple(choices),
            gold_answer=letter,
            metadata={
                "id": doc.get("id", f"openbookqa_{index}"),
                "index": index,
                "dataset": "openbookqa",
                "gold_idx": gold_idx,
                "gold_text": gold_text,
                "num_choices": len(choices),
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        fewshot = self.get_fewshot()
        is_mc = isinstance(self.config.formatter, MultipleChoiceFormatter)

        parts: list[str] = []
        for ex in fewshot:
            if is_mc:
                # MC continuation is the option letter.
                letter = ex.metadata.get("mc_answer") or ex.gold_answer or ""
                parts.append(_format_mc(ex.question, ex.choices or (), letter))
            else:
                # RC continuation is the answer *text* (gold_answer holds the letter, so prefer text).
                answer = ex.metadata.get("gold_text", "") or ex.gold_answer or ""
                parts.append(_format_rc(ex.question, answer))

        if is_mc:
            parts.append(_format_mc(instance.question, instance.choices or ()))
            continuations = tuple(
                f" {chr(ord('A') + i)}" for i in range(len(instance.choices or ()))
            )
        else:
            parts.append(_format_rc(instance.question))
            continuations = tuple(f" {c}" for c in (instance.choices or ()))

        prompt = "\n\n".join(parts)
        return LMRequest(
            request_type=RequestType.LOGLIKELIHOOD,
            prompt=prompt,
            continuations=continuations,
        )


register_variant("openbookqa", "rc", metrics=(LogprobPerCharMCAccuracyMetric(),))
register_variant("openbookqa", "mc", formatter=MultipleChoiceFormatter())
register_variant(
    "openbookqa", "bpb", metrics=(BPBMetricInstanceAvg(),), primary_metric=BPBMetricInstanceAvg()
)
register_variant(
    "openbookqa",
    "olmo3base",
    num_fewshot=5,
    split=Split.TEST,
    metrics=(LogprobPerCharMCAccuracyMetric(),),
)
register_variant("openbookqa", "full")
