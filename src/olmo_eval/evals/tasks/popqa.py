from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from olmo_eval.common.metrics import AccuracyMetric
from olmo_eval.common.scorers import ContainmentScorer, SQuADExactMatchScorer
from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, SamplingParams, Split
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import Task, register, register_variant
from olmo_eval.evals.tasks.constants.popqa import POPQA_FIXED_FEWSHOT


def _format_query(question: str) -> str:
    """PopQA's published prompt format.

    Mallen et al. (2023) section 4.1: "We use a simple template 'Q: A:' to format
    all of our questions for generative prediction." They note that more
    elaborate instructions did not help and risk overfitting to a given model, so
    this stays deliberately bare -- and unlike `naturalqs`, which uses
    "Question:/Answer:", it is not ours to restyle without losing comparability.
    """
    return f"Q: {question}\nA:"


def _parse_answers(raw: Any) -> list[str]:
    """PopQA's answer list, which arrives as a JSON string.

    The dataset is distributed as CSV, so ``possible_answers`` is the *text* of a
    JSON array rather than an array. Tolerate an already-parsed list too, in case
    a future export types the column properly.
    """
    if isinstance(raw, (list, tuple)):
        values = list(raw)
    elif isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return [raw.strip()] if raw.strip() else []
        values = parsed if isinstance(parsed, list) else [parsed]
    else:
        return []
    return [str(v).strip() for v in values if str(v).strip()]


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@register("popqa")
class PopQA(Task):
    """Entity-centric factual recall, templated from Wikidata triples.

    Follows the setup in Mallen et al. 2023, "When Not to Trust Language Models"
    (arXiv:2212.10511), which introduced the dataset: closed-book QA, the bare
    "Q: A:" template, 15-shot for open models, and accuracy by substring
    containment. Deviating from any of those makes the numbers incomparable to
    published PopQA results, which is the main reason to run this benchmark
    rather than one of the fact proxies.

    Containment is primary because it is the paper's metric. Exact match is
    reported beside it, not from the paper, because containment alone is
    inflated by two things -- alias substrings and models that hedge by listing
    candidates -- and the gap between the two is what exposes them. No F1: the
    answer is a single entity, so partial token overlap is not a meaningful
    quantity.

    PopQA's reason to exist is the popularity of the entity being asked about, so
    each instance carries its Wikipedia pageview counts through to the saved
    predictions. See ``instance_attributes`` below.
    """

    data_source = DataSource(path="akariasai/PopQA", split="test")
    # The only split PopQA publishes, and it is fully labeled, so the whole
    # thing is scored -- there is no held-out portion to leave out.
    split = Split.TEST
    metrics = (
        AccuracyMetric(scorer=ContainmentScorer),
        AccuracyMetric(scorer=SQuADExactMatchScorer),
    )
    primary_metric = AccuracyMetric(scorer=ContainmentScorer)
    sampling_params = SamplingParams(
        max_tokens=50,
        temperature=0.0,
        # "Q:" ends the answer by starting the next demonstration; "A:" is not a
        # stop, since it immediately precedes the text being generated.
        stop_sequences=("Q:", "\n\n"),
    )
    # 15-shot to match the paper's setting for open models. Mallen et al. use
    # zero-shot only for GPT-3, and only to hold down API cost.
    num_fewshot = 15
    _fewshot_source_name = "popqa_fixed"

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached()

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        question = str(doc.get("question", "")).strip()
        answers = _parse_answers(doc.get("possible_answers"))
        if not question or not answers:
            return None

        # Carried straight from the dataset, never produced by the model under
        # evaluation. s_pop is the subject entity's monthly Wikipedia pageviews
        # and is the axis PopQA is designed to be read along; o_pop is the same
        # for the answer entity. Kept under `instance_attributes` so the saved
        # predictions carry them beside each score and accuracy can be broken
        # down by popularity afterwards.
        attributes = {
            "s_pop": _as_int(doc.get("s_pop")),
            "o_pop": _as_int(doc.get("o_pop")),
            "prop": doc.get("prop"),
            "subj": doc.get("subj"),
        }

        return Instance(
            question=_format_query(question),
            gold_answer=answers[0],
            metadata={
                "id": doc.get("id", index),
                "index": index,
                "all_answers": answers,
                "instance_attributes": {k: v for k, v in attributes.items() if v is not None},
            },
        )

    def _build_fewshot(self) -> list[Instance]:
        # PopQA has no train split, so there is nothing to sample demonstrations
        # from without scoring an instance the model was just shown.
        return self._build_fixed_fewshot()

    def _build_fixed_fewshot(self) -> list[Instance]:
        instances = []
        for index, doc in enumerate(POPQA_FIXED_FEWSHOT):
            answers = [str(a) for a in doc["answer"]]
            instances.append(
                Instance(
                    question=_format_query(str(doc["question"])),
                    gold_answer=answers[0],
                    metadata={
                        "id": f"popqa_fixed_{index}",
                        "all_answers": answers,
                    },
                )
            )

        if self.config.num_fewshot and self.config.num_fewshot < len(instances):
            instances = instances[: self.config.num_fewshot]
        return instances

    def format_request(self, instance: Instance) -> LMRequest:
        parts: list[str] = []
        for ex in self.get_fewshot():
            parts.append(f"{ex.question} {ex.gold_answer}")
        parts.append(instance.question)
        prompt = "\n\n".join(parts)
        return LMRequest(request_type=RequestType.COMPLETION, prompt=prompt)

    def extract_answer(self, output: LMOutput) -> str:
        return output.text.strip()


register_variant("popqa", "gen")

# The paper's other setting: it prompts GPT-3 zero-shot purely to hold down API
# cost, not because zero-shot is preferable. Useful for reproducing that arm, and
# for measuring how much of exact match is prompt format rather than knowledge --
# with nothing demonstrating a bare-entity answer, exact match drops while
# containment barely moves.
register_variant("popqa", "zeroshot", num_fewshot=0)
