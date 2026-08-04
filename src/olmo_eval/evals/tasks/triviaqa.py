from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from olmo_eval.common.metrics import AccuracyMetric
from olmo_eval.common.scorers import SQuADExactMatchScorer, WindowedContainmentScorer
from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, SamplingParams, Split
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import Task, register, register_variant

# Co-LMLM's append_answer_stub.py, whose docstring gives the reason for it: "a
# base LM reads the question followed by an answer cue like 'The answer is' and
# emits the answer instead of continuing / echoing more questions."
_ANSWER_STUB = "The answer is"


def _format_query(question: str) -> str:
    return f"{question}\n{_ANSWER_STUB}"


def _answer_aliases(answer: Any) -> list[str]:
    """Reference answers, as Co-LMLM assembles them.

    Their loader takes the raw ``value`` and ``aliases`` fields::

        possible_answers = [answer["value"]] + list(answer.get("aliases", []))

    Deliberately not ``normalized_value`` / ``normalized_aliases``, which are
    the same strings lowercased and stripped of articles. The scorer lowercases
    anyway, and adding them would only duplicate references.
    """
    if not isinstance(answer, dict):
        return []
    values: list[str] = []
    primary = answer.get("value")
    if primary is not None:
        values.append(str(primary))
    aliases = answer.get("aliases") or []
    if isinstance(aliases, (list, tuple)):
        values.extend(str(a) for a in aliases)
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        v = v.strip()
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


@register("triviaqa")
class TriviaQA(Task):
    """Closed-book TriviaQA, mirroring Co-LMLM (arXiv:2607.07707).

    TriviaQA ships evidence documents and is titled a reading-comprehension
    dataset, so the config choice is what decides whether this measures recall
    or comprehension. ``rc.nocontext`` is the same 17,944 questions as ``rc``
    with the evidence stripped, which is what Co-LMLM loads and the only
    variant that probes parametric knowledge:

        ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")

    Zero-shot with a bare answer cue, matching their pipeline, which has no
    few-shot machinery at all. Note the consequence: nothing demonstrates the
    expected answer *shape*, so a model that knows the fact may still answer in
    a sentence. That is precisely what their containment metric absorbs and
    what exact match beside it exposes.

    One caveat on comparability, since it is easy to miss. ``rc`` is the
    reading-comprehension subset, filtered so evidence documents actually
    contain the answer. Stripping the evidence does not undo that filtering, so
    the question set still skews toward what was answerable from retrieved
    text. TriviaQA's authors suggest ``unfiltered`` for open-domain use; this
    task follows the paper instead, because matching it is the point.
    """

    data_source = DataSource(
        path="mandarjoshi/trivia_qa", subset="rc.nocontext", split="validation"
    )
    split = Split.VALIDATION
    metrics = (
        AccuracyMetric(scorer=WindowedContainmentScorer),
        AccuracyMetric(scorer=SQuADExactMatchScorer),
    )
    primary_metric = AccuracyMetric(scorer=WindowedContainmentScorer)
    # "All models use greedy decoding with a maximum of 32 tokens" (A.6). No
    # stop sequences: their pipeline sets none and relies on the token cap plus
    # the scorer's 100-character window to bound the answer.
    sampling_params = SamplingParams(max_tokens=32, temperature=0.0)
    num_fewshot = 0

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached()

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        question = str(doc.get("question", "")).strip()
        answers = _answer_aliases(doc.get("answer"))
        if not question or not answers:
            return None

        return Instance(
            question=_format_query(question),
            gold_answer=answers[0],
            metadata={
                "id": doc.get("question_id", index),
                "index": index,
                "all_answers": answers,
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        # No demonstrations and no assembly: the prompt is the question plus the
        # cue, exactly what their prepare script writes out.
        return LMRequest(request_type=RequestType.COMPLETION, prompt=instance.question)

    def extract_answer(self, output: LMOutput) -> str:
        return output.text.strip()


register_variant("triviaqa", "gen")
