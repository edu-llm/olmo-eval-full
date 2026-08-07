"""Leaderboard MATH-Hard: the Open LLM Leaderboard v2 math slice.

Dataset: ``DigitalLearningGmbH/MATH-lighteval`` per subject, **test** split,
filtered to ``level == "Level 5"`` -- the drop-in replacement lm-evaluation-harness
uses for ``leaderboard_math_hard`` after ``lighteval/MATH-Hard`` was removed. The
seven subjects are concatenated into one task; each item's ``metadata["id"]`` is
``"{subtask}|{doc_id}"`` where ``doc_id`` is the enumeration index within that
subject's Level-5 order, matching lm-eval's per-subtask ids so a calibrated ATLAS
bank keyed on those composite ids joins directly.

Prompt and scoring follow the Minerva/leaderboard convention (4-shot fixed
examples, ``Problem:\\n...\\n\\nSolution:``; correctness via sympy answer
equivalence over the ``\\boxed{}`` answer). Exact parity with lm-eval's own
``math_verify`` extraction is approximate.

**Do not substitute ``minerva_math_<subject>`` as the item source for this bank.**
The two look interchangeable -- same seven subjects, names that line up
(``algebra`` against ``algebra_hard``) -- and they are not.
:class:`~olmo_eval.evals.tasks.minerva_math.MinervaMathTask` enumerates the entire
``EleutherAI/hendrycks_math`` test split for a subject, every difficulty level, and
sets ``metadata["id"]`` to the position in that unfiltered split. The bank's
``question_id`` is a position in the Level-5-filtered subsequence of a different
dataset, so position 0 in one is not position 0 in the other.

Today that substitution fails safely because the math dataset spec keys the join on
the composite ``item_id``, which ``minerva_math_<subject>`` never emits. It stops
failing safely the moment anyone "makes the join work" by switching the spec to
``bridge_kind="atlas"``: the join would then key on the bare ``question_id``, match
close to 100% of the bank against close to entirely wrong problems, clear the 90%
overlap floor without a warning, and vendor a bank whose difficulty parameters
belong to different questions. Nothing downstream can detect that -- EAP treats
those parameters as fixed truth -- so the run reports a confident theta measured
against noise.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from olmo_eval.common.metrics import AccuracyMetric
from olmo_eval.common.scorers import MinervaMathScorer
from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, SamplingParams, Split
from olmo_eval.data import DataLoader, DataSource
from olmo_eval.evals.extract import MathExtractor
from olmo_eval.evals.tasks.common import Task, register
from olmo_eval.evals.tasks.constants.minerva_math import MINERVA_MATH_FIXED_FEWSHOT

# OpenLM subtask name -> MATH-lighteval subject subset.
_SUBTASK_TO_SUBSET: dict[str, str] = {
    "algebra_hard": "algebra",
    "counting_and_prob_hard": "counting_and_probability",
    "geometry_hard": "geometry",
    "intermediate_algebra_hard": "intermediate_algebra",
    "num_theory_hard": "number_theory",
    "prealgebra_hard": "prealgebra",
    "precalculus_hard": "precalculus",
}

_DATASET = "DigitalLearningGmbH/MATH-lighteval"
_HARD_LEVEL = "Level 5"


@register("leaderboard_math")
class LeaderboardMath(Task):
    """Concatenated Level-5 MATH across the seven leaderboard subtasks."""

    data_source = DataSource(path=_DATASET, split="test")
    split = Split.TEST
    fewshot_split = "train"
    metrics = (AccuracyMetric(scorer=MinervaMathScorer),)
    num_fewshot = 4
    sampling_params = SamplingParams(
        max_tokens=1024, temperature=0, stop_sequences=("Problem:", "\n\n")
    )
    dependencies = ["antlr4-python3-runtime>=4.11,<4.12"]

    @property
    def instances(self) -> Iterator[Instance]:
        if self._instances_cache is None:
            self._instances_cache = self._load_all_subtasks()
        yield from self._instances_cache

    def _load_all_subtasks(self) -> list[Instance]:
        loader = DataLoader()
        out: list[Instance] = []
        for subtask, subset in _SUBTASK_TO_SUBSET.items():
            source = DataSource(path=_DATASET, subset=subset, split="test")
            doc_id = 0
            for doc in loader.load(source):
                if doc.get("level") != _HARD_LEVEL:
                    continue
                instance = self._process_doc(doc, subtask, doc_id)
                if instance is not None:
                    out.append(instance)
                doc_id += 1
        return out

    def _process_doc(self, doc: dict[str, Any], subtask: str, doc_id: int) -> Instance | None:
        solution_text = doc["solution"]
        extracted = MathExtractor.extract_answer(solution_text)
        return Instance(
            question=doc["problem"],
            gold_answer=extracted[0] if extracted else None,
            metadata={
                "id": f"{subtask}|{doc_id}",
                "subtask": subtask,
                "level": doc.get("level"),
                "type": doc.get("type"),
                "solution_text": solution_text,
                "all_gold_answers": extracted if extracted else [],
            },
        )

    def _build_fewshot(self) -> list[Instance]:
        instances: list[Instance] = []
        for doc in MINERVA_MATH_FIXED_FEWSHOT:
            solution_text = doc["solution"]
            extracted = MathExtractor.extract_answer(solution_text)
            instances.append(
                Instance(
                    question=doc["problem"],
                    gold_answer=extracted[0] if extracted else None,
                    metadata={
                        "solution_text": solution_text,
                        "all_gold_answers": extracted if extracted else [],
                    },
                )
            )
        return instances

    def format_request(self, instance: Instance) -> LMRequest:
        parts: list[str] = []
        for ex in self.get_fewshot():
            text = f"Problem:\n{ex.question}\n\nSolution:"
            solution = ex.metadata.get("solution_text", ex.gold_answer or "")
            if solution:
                text += " " + str(solution)
            parts.append(text)
        parts.append(f"Problem:\n{instance.question}\n\nSolution:")
        return LMRequest(
            request_type=RequestType.COMPLETION,
            prompt="\n\n".join(parts),
        )

    def extract_answer(self, output: LMOutput) -> str | None:
        answers = MathExtractor.extract_answer(output.text)
        output.metadata["all_extracted_answers"] = answers if answers else []
        return answers[0] if answers else None
