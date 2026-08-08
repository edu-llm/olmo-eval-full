from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any

from olmo_eval.common.formatters import MultipleChoiceFormatter
from olmo_eval.common.metrics import (
    BPBMetricInstanceAvg,
    LogprobMCAccuracyMetric,
    LogprobUncondMCAccuracyMetric,
)
from olmo_eval.common.types import Instance, LMRequest, RequestType, SamplingParams, Split
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import Task, register, register_variant
from olmo_eval.evals.tasks.common.format_helpers import format_mc as _format_mc
from olmo_eval.evals.tasks.common.format_helpers import format_rc as _format_rc

# The MathQA `options` field is a single string such as
# "a ) 38 , b ) 27.675 , c ) 30 , d ) 76 , e ) 65". Each option is introduced by a
# letter label followed by " )". A value is everything up to the next ", <letter> )"
# marker (or the end), so a value that itself contains a comma is not split.
_OPTION_RE = re.compile(r"([a-eA-E])\s*\)\s*(.*?)(?=\s*,\s*[a-eA-E]\s*\)|\s*$)")


def _parse_options(options: str) -> dict[str, str]:
    """Parse the MathQA options string into a {label: value} map (labels lower-cased)."""
    parsed: dict[str, str] = {}
    for label, value in _OPTION_RE.findall(options.strip()):
        cleaned = value.strip().rstrip(",").strip()
        if cleaned:
            parsed[label.lower()] = cleaned
    return parsed


@register("mathqa")
class MathQA(Task):
    """MathQA: 5-way multiple-choice math word problems, scored by log-likelihood."""

    data_source = DataSource(path="allenai/math_qa", split="test", revision="refs/convert/parquet")
    split = Split.TEST
    metrics = (LogprobMCAccuracyMetric(),)
    num_fewshot = 0
    fewshot_split = "train"
    sampling_params = SamplingParams(temperature=0.0)

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached()

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        question = str(doc.get("Problem", "")).strip()
        options_str = str(doc.get("options", ""))
        correct = str(doc.get("correct", "")).strip().lower()
        if not question or not options_str or not correct:
            return None

        by_label = _parse_options(options_str)
        labels = sorted(by_label)
        choices = [by_label[label] for label in labels]
        if len(choices) < 2 or correct not in by_label:
            return None

        gold_idx = labels.index(correct)

        return Instance(
            question=question,
            choices=tuple(choices),
            gold_answer=correct.upper(),
            metadata={
                "id": doc.get("id", f"mathqa_{index}"),
                "index": index,
                "dataset": "mathqa",
                "gold_idx": gold_idx,
                "gold_text": choices[gold_idx],
                "num_choices": len(choices),
            },
        )

    def _uses_uncond_metric(self) -> bool:
        return any(isinstance(m, LogprobUncondMCAccuracyMetric) for m in self.config.metrics)

    def format_request(self, instance: Instance) -> LMRequest:
        fewshot = self.get_fewshot()
        is_mc = isinstance(self.config.formatter, MultipleChoiceFormatter)

        parts: list[str] = []
        for ex in fewshot:
            if is_mc:
                answer = ex.metadata.get("mc_answer", ex.gold_answer or "")
                parts.append(_format_mc(ex.question, ex.choices or (), answer))
            else:
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

        if not is_mc and self._uses_uncond_metric():
            uncond_prompt = "Answer:"
            num_choices = len(continuations)
            all_continuations = continuations + continuations
            all_cont_prompts = tuple([prompt] * num_choices + [uncond_prompt] * num_choices)
            return LMRequest(
                request_type=RequestType.LOGLIKELIHOOD,
                prompt=prompt,
                continuations=all_continuations,
                continuation_prompts=all_cont_prompts,
            )

        return LMRequest(
            request_type=RequestType.LOGLIKELIHOOD,
            prompt=prompt,
            continuations=continuations,
        )


register_variant("mathqa", "mc", formatter=MultipleChoiceFormatter())
register_variant("mathqa", "rc", metrics=(LogprobUncondMCAccuracyMetric(),))
register_variant(
    "mathqa", "bpb", metrics=(BPBMetricInstanceAvg(),), primary_metric=BPBMetricInstanceAvg()
)
