"""MuSR (Multistep Soft Reasoning) as Open LLM Leaderboard v2 runs it.

``TAUR-Lab/MuSR`` poses three kinds of commonsense reasoning problem, each a long
generated narrative followed by one question over it. The three are HuggingFace
*splits* rather than subset configurations -- ``murder_mysteries`` (250 items, 2
choices), ``object_placements`` (256, 2 to 5 choices) and ``team_allocation`` (250, 3
choices) -- which is why the split name travels on the task class instead of coming
from ``config.split``.

Tasks (3 total)::

    musr_murder_mysteries
    musr_object_placements
    musr_team_allocation

One registered task per subtask, rather than one task concatenating all three, because
a calibrated bank spanning them is joined by ``<subtask>|<position>`` rebuilt from each
subtask's own ``enumerate(task.instances)``. Enumeration order is therefore part of the
contract: instances are yielded in dataset order, unsorted and unfiltered, so the k-th
instance is the k-th item of that split.

Convention is lm-evaluation-harness ``leaderboard_musr`` (``lm_eval/tasks/leaderboard/
musr``), which is what the bank was calibrated under; Research has no MuSR task to match
against. Zero-shot, ``acc_norm`` over the choice texts, and a prompt of narrative,
question, then the choices numbered from one, then an ``Answer:`` cue. What is *ranked*
is the choice text and not its number, so the numbering is presentation only.

The narrative is the item. It is several thousand characters of the story the question
is answered from, and a MuSR question shown without it is not merely harder but
unanswerable -- the model would be guessing among names it has never seen. Graded that
way a model looks uniformly weak rather than broken, so the damage surfaces as a
depressed ability estimate rather than as an error.
"""

from __future__ import annotations

import ast
import logging
from collections.abc import Iterator
from typing import Any

from olmo_eval.common.metrics import LogprobPerCharMCAccuracyMetric
from olmo_eval.common.types import Instance, LMRequest, RequestType, SamplingParams, Split
from olmo_eval.data import DataLoader, DataSource
from olmo_eval.evals.tasks.common import Task, register

log = logging.getLogger(__name__)

_DATASET = "TAUR-Lab/MuSR"

#: Subtask name, which is also its HuggingFace split.
_SUBTASKS = ("murder_mysteries", "object_placements", "team_allocation")


def _format_musr(stem: str, choices: tuple[str, ...], answer: str | None = None) -> str:
    """Lay out one MuSR prompt, matching ``leaderboard_musr``'s ``doc_to_text``.

    The blank line before ``Answer:`` is not a typo: upstream builds the numbered block
    with a trailing newline and then joins it to the cue with another, and the extra
    break is inside the string every calibrated difficulty was estimated against.
    """
    numbered = "".join(f"{i + 1} - {choice}\n" for i, choice in enumerate(choices))
    prompt = f"{stem}\n\n{numbered}\nAnswer:"
    if answer:
        prompt += f" {answer}"
    return prompt


class MuSRTask(Task):
    """Base class for the MuSR subtasks."""

    subtask: str = ""
    data_source = DataSource(path=_DATASET)
    split = Split.TEST
    metrics = (LogprobPerCharMCAccuracyMetric(),)
    num_fewshot = 0
    sampling_params = SamplingParams(temperature=0.0)

    @property
    def instances(self) -> Iterator[Instance]:
        if self._instances_cache is None:
            loader = DataLoader()
            source = self.config.get_data_source(split=self.subtask)
            self._instances_cache = [
                instance
                for index, doc in enumerate(loader.load(source))
                if (instance := self.process_doc(doc, index)) is not None
            ]
        yield from self._instances_cache

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        # Upstream stores the choice list as the repr of a Python list, not as a
        # sequence, so it has to be parsed rather than read.
        choices = tuple(ast.literal_eval(doc["choices"]))
        gold_text = doc["answer_choice"]

        # Upstream resolves the gold by looking the answer text up in the choices;
        # answer_index is cross-checked rather than trusted because the two disagreeing
        # would mean the dataset moved, which is exactly what invalidates a positional
        # join. No choice set repeats a string, so the lookup is unambiguous. An item
        # whose gold cannot be resolved keeps its position and is scored incorrect for
        # everyone, matching upstream, rather than being dropped and shifting the join.
        gold_idx = choices.index(gold_text) if gold_text in choices else None
        if gold_idx != doc["answer_index"]:
            log.warning(
                "musr_%s item %d: answer_choice %r resolves to index %r, not the stated "
                "answer_index %r.",
                self.subtask,
                index,
                gold_text,
                gold_idx,
                doc["answer_index"],
            )
            gold_idx = None

        return Instance(
            question=f"{doc['narrative']}\n\n{doc['question']}",
            choices=choices,
            gold_answer=gold_text,
            metadata={
                "index": index,
                "dataset": "musr",
                "subtask": self.subtask,
                "gold_idx": gold_idx,
                "gold_text": gold_text,
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        parts = [
            _format_musr(example.question, example.choices or (), example.gold_answer)
            for example in self.get_fewshot()
        ]
        parts.append(_format_musr(instance.question, instance.choices or ()))
        return LMRequest(
            request_type=RequestType.LOGLIKELIHOOD,
            prompt="\n\n".join(parts),
            continuations=tuple(f" {choice}" for choice in instance.choices or ()),
        )


for _subtask in _SUBTASKS:
    _class_name = f"MuSR{_subtask.title().replace('_', '')}"
    _cls = type(
        _class_name,
        (MuSRTask,),
        {
            "__module__": __name__,
            "__qualname__": _class_name,
            "subtask": _subtask,
        },
    )
    globals()[_class_name] = _cls
    register(f"musr_{_subtask}")(_cls)
