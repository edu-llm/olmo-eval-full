"""Deterministic scoring backend for IFEval (plan Step 5).

IFEval criteria are not graded by an LLM judge -- each ships a machine-checkable
constraint. `IFEvalVerifier` implements the same `JudgeClient` protocol the engine expects
(`name`, `prompt_version`, `seed`, `evaluate`), so it drops into `run_evaluation` in place
of `OpenAICompatibleJudge`. Because an IFEval run is single-axis / one-benchmark, the whole
run uses this backend and `judge_results.jsonl` picks up its identity automatically.

For each criterion it reads `rubric.verifier == {"instruction_id", "kwargs"}` (written by
scripts/ingest_ifeval.py), looks up the vendored checker, reproduces the official grading
path (`build_description(**kwargs)` then `check_following(response)`), and returns a
`JudgeVerdict`. No LLM, no network -- grading is exact and reproducible.
"""
from __future__ import annotations

import random

from .schemas import JudgeVerdict, Rubric, Scenario
from .verifiers.ifeval import instructions_registry

PROMPT_VERSION = "ifeval-v1"

_REGISTRY = instructions_registry.INSTRUCTION_DICT


class IFEvalVerifier:
    """JudgeClient-compatible deterministic verifier for IFEval rubrics."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.prompt_version = PROMPT_VERSION

    @property
    def name(self) -> str:
        return "ifeval-verifier"

    def evaluate(self, scenario: Scenario, rubric: Rubric, response: str) -> JudgeVerdict:
        spec = rubric.verifier
        if not isinstance(spec, dict) or "instruction_id" not in spec:
            return JudgeVerdict(
                verdict="fail",
                rationale="rubric has no verifier spec; IFEvalVerifier requires one",
                unscorable_reason="missing_verifier_spec",
                raw_output="",
            )

        instruction_id = spec["instruction_id"]
        kwargs = spec.get("kwargs") or {}
        cls = _REGISTRY.get(instruction_id)
        if cls is None:
            return JudgeVerdict(
                verdict="fail",
                rationale=f"unknown IFEval instruction_id {instruction_id!r}",
                unscorable_reason="unknown_instruction_id",
                raw_output="",
            )

        try:
            checker = cls(instruction_id)
            # A few checkers draw random defaults for absent kwargs; the ingester supplies
            # the needed ones, so this is deterministic. Seed anyway for stability.
            random.seed(self.seed)
            checker.build_description(**kwargs)
            passed = bool(checker.check_following(response or ""))
        except Exception as e:  # noqa: BLE001 - a checker crash is a scoring failure, not a raise
            return JudgeVerdict(
                verdict="fail",
                rationale=f"verifier raised {type(e).__name__}: {e}",
                unscorable_reason="verifier_error",
                raw_output="",
            )

        return JudgeVerdict(
            verdict="pass" if passed else "fail",
            evidence=rubric.criterion,
            rationale=f"deterministic check {instruction_id}: {'satisfied' if passed else 'violated'}",
            unscorable_reason=None,
            raw_output=f"check_following={passed}",
        )
