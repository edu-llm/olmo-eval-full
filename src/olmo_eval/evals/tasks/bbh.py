"""BIG-Bench Hard as Open LLM Leaderboard v2 runs it.

``SaylorTwift/bbh`` holds one subset per BIG-Bench Hard task, each a free-form
``(input, target)`` pair. Leaderboard v2 reformulates 24 of them as multiple choice by
attaching a hardcoded choice set per subtask and ranking those strings under
``acc_norm``; the three genuinely generative subtasks -- ``dyck_languages``,
``multistep_arithmetic_two`` and ``word_sorting`` -- have no closed answer set to rank
and are excluded upstream, so they are absent here too.

Tasks (24 total): ``bbh_boolean_expressions`` through ``bbh_web_of_lies``, one per
subtask in :data:`~olmo_eval.evals.tasks.constants.bbh.BBH_CHOICES`. Sizes are 250 items
each except ``causal_judgement`` (187), ``snarks`` (178) and ``penguins_in_a_table``
(146), which are those subtasks' full upstream size rather than a truncation.

One registered task per subtask, rather than one task concatenating all 24, because a
calibrated bank spanning them is joined by ``<subtask>|<position>`` rebuilt from each
subtask's own ``enumerate(task.instances)``. Enumeration order is therefore part of the
contract: instances are yielded in dataset order, unsorted and unfiltered, so the k-th
instance is the k-th item of that subtask's test split.

Convention is lm-evaluation-harness ``leaderboard_bbh`` (``lm_eval/tasks/leaderboard/
bbh_mc``), which is what the bank was calibrated under; Research has no BBH task to match
against. Three-shot from fixed exemplars drawn in order, ``Q: {input}\\nA:`` per turn, a
one-line description ahead of the whole prompt, and ``acc_norm`` over the subtask's
choice set. See :mod:`olmo_eval.evals.tasks.constants.bbh` for why the choice sets are
transcribed rather than inferred.

The description runs straight into the first exemplar with no separator between them.
That reads like an oversight and is not one to correct: it is what
``ConfigurableTask.fewshot_context`` produces for a non-chat prompt, so the leading
``...expression.Q: not ( ( not not True ) ) is`` is the literal string every calibrated
difficulty here was estimated against.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from olmo_eval.common.metrics import LogprobPerCharMCAccuracyMetric
from olmo_eval.common.types import Instance, LMRequest, RequestType, SamplingParams, Split
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import Task, register_subtasks
from olmo_eval.evals.tasks.constants.bbh import (
    BBH_CHOICES,
    BBH_DESCRIPTIONS,
    BBH_FIXED_FEWSHOT,
)

log = logging.getLogger(__name__)

_DATASET = "SaylorTwift/bbh"


class BBHTask(Task):
    """Base class for the BIG-Bench Hard subtasks."""

    subtask: str = ""
    data_source = DataSource(path=_DATASET, split="test")
    split = Split.TEST
    metrics = (LogprobPerCharMCAccuracyMetric(),)
    num_fewshot = 3
    sampling_params = SamplingParams(temperature=0.0)

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached()

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        choices = BBH_CHOICES[self.subtask]
        target = doc["target"]

        # A target outside the choice set is unrankable, and upstream scores it wrong
        # for every model. Keep the instance so later positions do not shift, and let
        # the missing gold_idx mark it unanswerable.
        gold_idx = choices.index(target) if target in choices else None
        if gold_idx is None:
            log.warning(
                "bbh_%s item %d: target %r is not in the subtask's choice set, so no "
                "model can be scored correct on it.",
                self.subtask,
                index,
                target,
            )

        return Instance(
            question=doc["input"],
            choices=choices,
            gold_answer=target,
            metadata={
                "index": index,
                "dataset": "bbh",
                "subtask": self.subtask,
                "gold_idx": gold_idx,
                "gold_text": target,
            },
        )

    def _build_fewshot(self) -> list[Instance]:
        exemplars = BBH_FIXED_FEWSHOT[self.subtask][: self.config.num_fewshot]
        return [
            Instance(
                question=example["input"],
                gold_answer=example["target"],
                metadata={"gold_text": example["target"]},
            )
            for example in exemplars
        ]

    def format_request(self, instance: Instance) -> LMRequest:
        parts = [
            f"Q: {example.question}\nA: {example.gold_answer}" for example in self.get_fewshot()
        ]
        parts.append(f"Q: {instance.question}\nA:")
        return LMRequest(
            request_type=RequestType.LOGLIKELIHOOD,
            prompt=BBH_DESCRIPTIONS[self.subtask] + "\n\n".join(parts),
            continuations=tuple(f" {choice}" for choice in instance.choices or ()),
        )


register_subtasks(
    base_class=BBHTask,
    subtasks=list(BBH_CHOICES),
    task_prefix="bbh",
    data_source=_DATASET,
    subtask_attr="subtask",
)
