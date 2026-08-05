from __future__ import annotations

import random
from collections.abc import Iterator
from typing import Any

from olmo_eval.common.formatters import MultipleChoiceFormatter
from olmo_eval.common.metrics import BPBMetricInstanceAvg, LogprobPerCharMCAccuracyMetric
from olmo_eval.common.types import Instance, LMRequest, RequestType, SamplingParams, Split
from olmo_eval.data import DataLoader, DataSource
from olmo_eval.evals.tasks.common import Task, register, register_variant
from olmo_eval.evals.tasks.common.format_helpers import (
    format_mc as _format_mc,
)
from olmo_eval.evals.tasks.common.format_helpers import (
    format_rc as _format_rc,
)
from olmo_eval.evals.tasks.constants.socialiqa import SOCIALIQA_FIXED_FEWSHOT


@register("socialiqa")
class SocialIQA(Task):
    data_source = DataSource(
        path="social_i_qa", split="validation", revision="refs/convert/parquet"
    )
    split = Split.VALIDATION
    metrics = (LogprobPerCharMCAccuracyMetric(),)
    num_fewshot = 0
    fewshot_split = "train"
    sampling_params = SamplingParams(temperature=0.0)

    #: Whether a limit samples the union of every split instead of the scored one.
    #:
    #: oe-eval-internal's limited SocialIQA variants declare split="all", so a limit
    #: there samples test -> validation -> train (for SocialIQA, validation -> train).
    #: We default to False because that union is not the evaluation split, and a
    #: sample of it is neither a subset of validation nor a superset: a limited run
    #: would then be incomparable to the full run it is standing in for, which is the
    #: one thing a cheap trial run has to avoid being.
    #:
    #: What that gives up is fidelity to another harness's *limited* runs, not to any
    #: published figure -- a subsample was never comparable to one. Reproducing a
    #: published OLMES number means running this task unlimited over its own split.
    #: Set this True on the class only to diff against oe-eval-internal's limited
    #: output instance by instance; it is deliberately not a TaskConfig field,
    #: because nothing here sets it and a config knob would advertise the union as an
    #: ordinary choice rather than the compatibility shim it is.
    #:
    #: The ``xlarge`` and ``mc_olmo3base`` variants below look like they opt out by
    #: declaring ``train+validation`` on their data source, but they do not:
    #: ``TaskConfig.get_data_source`` rewrites the split on every call, so that
    #: declaration has never reached the loader and their union came from this branch
    #: alone. With the default they score validation, all 1,954 rows of it, since
    #: that is fewer than their limit of 10,000. Neither is registered in either
    #: skill's benchmarks.json, so nothing here runs them; set this True to get the
    #: union back if you ever need to.
    limit_reads_all_splits = False

    @property
    def instances(self) -> Iterator[Instance]:
        if self._instances_cache is None:
            self._instances_cache = self._load_socialiqa_instances()
        yield from self._instances_cache

    def _load_socialiqa_instances(self) -> list[Instance]:
        loader = DataLoader()
        instances: list[Instance] = []

        splits = (
            ["validation", "train"]
            if self.config.limit and self.limit_reads_all_splits
            else [self.config.split.value]
        )

        index = 0
        for split in splits:
            source = self.config.get_data_source(split=split)
            for doc in loader.load(source):
                inst = self.process_doc(doc, index)
                if inst is not None:
                    instances.append(inst)
                    index += 1

        # Match oe-eval-internal's random subsampling: random.Random(1234).sample(docs, limit)
        if self.config.limit and len(instances) > self.config.limit:
            instances = random.Random(1234).sample(instances, self.config.limit)

        return instances

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        context = doc.get("context", "")
        question_text = doc.get("question", "")
        if not context or not question_text:
            return None

        question = f"{context} {question_text}"
        choices = (doc.get("answerA", ""), doc.get("answerB", ""), doc.get("answerC", ""))
        label = int(doc.get("label", "1")) - 1
        gold_text = choices[label] if 0 <= label < len(choices) else ""

        return Instance(
            question=question,
            choices=choices,
            gold_answer=str(label),
            metadata={
                "id": index,
                "index": index,
                "dataset": "socialiqa",
                "gold_idx": label,
                "gold_text": gold_text,
            },
        )

    def _build_fewshot(self) -> list[Instance]:
        if self.config.fewshot_source == "olmes_socialiqa_fixed":
            return self._build_fixed_fewshot()
        return super()._build_fewshot()

    def _build_fixed_fewshot(self) -> list[Instance]:
        instances = []
        for doc in SOCIALIQA_FIXED_FEWSHOT:
            question = f"{doc['context']} {doc['question']}"
            choices = (doc["answerA"], doc["answerB"], doc["answerC"])
            label = int(doc["label"]) - 1
            gold_text = choices[label] if 0 <= label < len(choices) else ""
            letter = chr(ord("A") + label)

            instances.append(
                Instance(
                    question=question,
                    choices=choices,
                    gold_answer=gold_text,
                    metadata={
                        "gold_idx": label,
                        "gold_text": gold_text,
                        "mc_answer": letter,
                    },
                )
            )
        # Take the first num_fewshot examples (no random sampling) to match
        # oe-eval-internal's fewshot_source behavior which uses [:k].
        if self.config.num_fewshot and self.config.num_fewshot < len(instances):
            instances = instances[: self.config.num_fewshot]
        return instances

    def format_request(self, instance: Instance) -> LMRequest:
        fewshot = self.get_fewshot()
        is_mc = self.config.formatter is not None

        parts: list[str] = []
        for ex in fewshot:
            if is_mc:
                answer = ex.metadata.get("mc_answer", "")
                parts.append(_format_mc(ex.question, ex.choices or (), answer))
            else:
                answer = ex.gold_answer or ex.metadata.get("gold_text", "")
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


register_variant("socialiqa", "rc")
register_variant("socialiqa", "mc", formatter=MultipleChoiceFormatter())
register_variant(
    "socialiqa",
    "olmo3base",
    num_fewshot=5,
    fewshot_source="olmes_socialiqa_fixed",
)
register_variant(
    "socialiqa",
    "mc_olmo3base",
    formatter=MultipleChoiceFormatter(),
    data_source=DataSource(
        path="social_i_qa", split="train+validation", revision="refs/convert/parquet"
    ),
    num_fewshot=5,
    limit=10000,
    fewshot_source="olmes_socialiqa_fixed",
)
register_variant(
    "socialiqa",
    "xlarge",
    data_source=DataSource(
        path="social_i_qa", split="train+validation", revision="refs/convert/parquet"
    ),
    num_fewshot=5,
    limit=10000,
    fewshot_source="olmes_socialiqa_fixed",
)
register_variant("socialiqa", "bpb", metrics=(BPBMetricInstanceAvg(),))
register_variant("socialiqa", "olmes", num_fewshot=5, fewshot_source="olmes_socialiqa_fixed")
register_variant("socialiqa", "full")
