"""IFBench / IFEval metrics.

All four metrics share :class:`IFEvalScorer`, which writes per-instruction
strict and loose pass lists to ``output.metadata["ifeval"]``. Each metric
aggregates that side-band data along the prompt or instruction axis.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from olmo_eval.common.metrics.base import Metric
from olmo_eval.common.scorers import IFEvalScorer, Scorer
from olmo_eval.common.types import Response


def _iter_results(responses: Sequence[Response], key: str) -> list[list[bool]]:
    out: list[list[bool]] = []
    for response in responses:
        if not response.outputs:
            out.append([])
            continue
        meta = response.outputs[0].metadata or {}
        ifeval = meta.get("ifeval") or {}
        out.append(list(ifeval.get(key, [])))
    return out


def _prompt_level(results: list[list[bool]]) -> float:
    if not results:
        return 0.0
    correct = sum(1 for r in results if r and all(r))
    return correct / len(results)


def _instruction_level(results: list[list[bool]]) -> float:
    total = sum(len(r) for r in results)
    if total == 0:
        return 0.0
    correct = sum(1 for r in results for v in r if v)
    return correct / total


def _response_results(response: Response, key: str) -> list[bool]:
    if not response.outputs:
        return []
    meta = response.outputs[0].metadata or {}
    return list((meta.get("ifeval") or {}).get(key, []))


def _prompt_level_instance(response: Response, key: str) -> float | None:
    results = _response_results(response, key)
    if not results:
        return None
    return 1.0 if all(results) else 0.0


def _instruction_level_instance(response: Response, key: str) -> float | None:
    results = _response_results(response, key)
    if not results:
        return None
    return sum(1.0 for v in results if v) / len(results)


@dataclass(frozen=True, slots=True)
class IFEvalPromptStrictAccuracy(Metric):
    """Fraction of prompts where every instruction passes under strict scoring."""

    name: str = "prompt_level_strict_acc"
    scorer: type[Scorer] = IFEvalScorer

    def compute(self, responses: Sequence[Response]) -> float:
        return _prompt_level(_iter_results(responses, "strict"))

    def compute_instance(self, response: Response) -> float | None:
        return _prompt_level_instance(response, "strict")

    def supports_pairwise_scorer_fallback(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class IFEvalPromptLooseAccuracy(Metric):
    """Fraction of prompts where every instruction passes under loose scoring."""

    name: str = "prompt_level_loose_acc"
    scorer: type[Scorer] = IFEvalScorer

    def compute(self, responses: Sequence[Response]) -> float:
        return _prompt_level(_iter_results(responses, "loose"))

    def compute_instance(self, response: Response) -> float | None:
        return _prompt_level_instance(response, "loose")

    def supports_pairwise_scorer_fallback(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class IFEvalInstStrictAccuracy(Metric):
    """Fraction of individual instructions passing under strict scoring."""

    name: str = "inst_level_strict_acc"
    scorer: type[Scorer] = IFEvalScorer

    def compute(self, responses: Sequence[Response]) -> float:
        return _instruction_level(_iter_results(responses, "strict"))

    def compute_instance(self, response: Response) -> float | None:
        return _instruction_level_instance(response, "strict")

    def supports_pairwise_scorer_fallback(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class IFEvalInstLooseAccuracy(Metric):
    """Fraction of individual instructions passing under loose scoring."""

    name: str = "inst_level_loose_acc"
    scorer: type[Scorer] = IFEvalScorer

    def compute(self, responses: Sequence[Response]) -> float:
        return _instruction_level(_iter_results(responses, "loose"))

    def compute_instance(self, response: Response) -> float | None:
        return _instruction_level_instance(response, "loose")

    def supports_pairwise_scorer_fallback(self) -> bool:
        return False
