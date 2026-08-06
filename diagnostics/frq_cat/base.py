"""Stable interface contract for FRQ CAT diagnostic styles.

This module is the frozen contract every free-response Computerized Adaptive
Testing (CAT) style implements. Styles live under ``styles/<name>/`` and
self-register with the registry; the generic engine in
:mod:`diagnostics.frq_cat.common.cat_loop` drives each style purely through the
:class:`FrqCatStyle` interface defined here.

FRQ grading is two-stage and both stages are shared (not per style): the
checkpoint under test generates a free-response answer per scenario
(:class:`RespGenModel`), then a frozen LLM-as-a-judge grades that answer against
each rubric criterion (:class:`Judge`). Each *criterion* is the IRT item: it
carries the fitted parameters and its pass/fail verdict is the graded response
the ability estimator consumes.

The dataclasses below are the shared vocabulary passed between the engine, the
shared utilities, and each style. They are intentionally minimal and
model-agnostic so unidimensional and multidimensional (MIRT) styles share one
contract: abilities and discriminations are scalars for unidimensional models and
tuples for multidimensional ones.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Protocol, runtime_checkable

#: Ability is a scalar for unidimensional IRT and a vector for MIRT.
Ability = float | tuple[float, ...]
#: Discrimination is a scalar for 1PL/2PL/3PL and a vector for multidimensional IRT.
Discrimination = float | tuple[float, ...]


@dataclass(frozen=True, slots=True)
class Scenario:
    """A single free-response tutoring scenario the checkpoint answers.

    ``messages`` is the chat-format prompt sent to the tutor endpoint; ``prompt``
    is the plain-text question kept for provenance and non-chat backends.
    """

    scenario_id: str
    prompt: str
    messages: tuple[dict[str, str], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Criterion:
    """One binary rubric criterion — the IRT item of an FRQ bank.

    ``scenario_id`` links the criterion to the scenario whose response it grades.
    ``q_modeled`` is the q-matrix row keyed by modeled skill name (a single
    ``{"ability": 1}`` entry for unidimensional styles, multiple entries for MIRT).
    """

    criterion_id: str
    scenario_id: str
    text: str
    primary_skill: str = "ability"
    q_modeled: Mapping[str, int] = field(default_factory=lambda: {"ability": 1})
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FrqBank:
    """A prepared FRQ bank: scenarios plus the criteria that grade them."""

    name: str
    scenarios: dict[str, Scenario]
    criteria: tuple[Criterion, ...]

    def __len__(self) -> int:
        return len(self.criteria)

    def get(self, criterion_id: str) -> Criterion:
        """Return the criterion with ``criterion_id`` or raise ``KeyError``."""
        for criterion in self.criteria:
            if criterion.criterion_id == criterion_id:
                return criterion
        raise KeyError(criterion_id)

    def get_scenario(self, scenario_id: str) -> Scenario:
        """Return the scenario with ``scenario_id`` or raise ``KeyError``."""
        return self.scenarios[scenario_id]


@dataclass(frozen=True, slots=True)
class IRTItemParams:
    """IRT parameters for one criterion.

    Supports 1PL/2PL/3PL and multidimensional (MIRT) models. ``discrimination``
    is a scalar for unidimensional models and a tuple for MIRT; ``guessing`` is
    zero for 1PL/2PL.
    """

    item_id: str
    difficulty: float
    discrimination: Discrimination = 1.0
    guessing: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class IRTBank:
    """IRT parameters for a bank, keyed by criterion id.

    ``dimensions`` is 1 for unidimensional models and greater than 1 for MIRT.
    """

    params: dict[str, IRTItemParams]
    dimensions: int = 1

    def __len__(self) -> int:
        return len(self.params)

    def get(self, item_id: str) -> IRTItemParams:
        """Return the parameters for ``item_id`` or raise ``KeyError``."""
        return self.params[item_id]


@dataclass(frozen=True, slots=True)
class AbilityEstimate:
    """A point estimate of ability with its standard error.

    Both fields are scalars for unidimensional models and tuples for MIRT.
    """

    theta: Ability
    standard_error: Ability
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class JudgeVerdict:
    """The frozen judge's graded outcome for one criterion."""

    criterion_id: str
    passed: bool
    rationale: str = ""
    evidence: str = ""
    unscorable_reason: str = ""
    raw_output: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CriterionResponse:
    """A graded response to one administered criterion (the IRT observation)."""

    criterion_id: str
    correct: bool
    verdict: JudgeVerdict | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CATState:
    """Mutable state threaded through a single adaptive test session.

    The generic engine owns this object: it appends responses, refreshes
    ``ability`` after each criterion, and consults the style's stopping rule. The
    stopping parameters supplied by the runner (``se_threshold``, ``max_items``)
    live here so a style's stopping rule can read them without a shared config.
    """

    benchmark: str
    se_threshold: float | None = None
    max_items: int | None = None
    administered: list[CriterionResponse] = field(default_factory=list)
    ability: AbilityEstimate | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def step(self) -> int:
        """Number of criteria administered so far."""
        return len(self.administered)

    @property
    def administered_ids(self) -> set[str]:
        """Ids of criteria already administered this session."""
        return {response.criterion_id for response in self.administered}


@dataclass(frozen=True, slots=True)
class CATReport:
    """The serializable outcome of a completed FRQ CAT session."""

    cat_style: str
    benchmark: str
    ability: AbilityEstimate
    num_items_administered: int
    responses: tuple[CriterionResponse, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the report."""
        return {
            "cat_style": self.cat_style,
            "benchmark": self.benchmark,
            "ability": {
                "theta": self.ability.theta,
                "standard_error": self.ability.standard_error,
                "metadata": self.ability.metadata,
            },
            "num_items_administered": self.num_items_administered,
            "responses": [
                {
                    "criterion_id": response.criterion_id,
                    "correct": response.correct,
                    "verdict": (
                        None
                        if response.verdict is None
                        else {
                            "passed": response.verdict.passed,
                            "rationale": response.verdict.rationale,
                            "evidence": response.verdict.evidence,
                            "unscorable_reason": response.verdict.unscorable_reason,
                        }
                    ),
                    "metadata": response.metadata,
                }
                for response in self.responses
            ],
            "metadata": self.metadata,
        }


@runtime_checkable
class RespGenModel(Protocol):
    """The checkpoint under test, serving free-response generations.

    Implementations generate one deterministic response per scenario. See
    :mod:`diagnostics.frq_cat.common.respgen` for the served-endpoint client.
    """

    def generate(self, scenarios: Sequence[Scenario]) -> dict[str, str]:
        """Return ``{scenario_id: response_text}`` for ``scenarios``."""
        ...


@runtime_checkable
class Judge(Protocol):
    """The frozen LLM-as-a-judge that grades a response per criterion.

    The judge is shared and calibrated (defined once in
    :mod:`diagnostics.frq_cat.common.judge`); styles select it via config and
    never fork it.
    """

    def evaluate(
        self, scenario: Scenario, criterion: Criterion, response_text: str
    ) -> JudgeVerdict:
        """Grade ``response_text`` for ``criterion`` and return a verdict."""
        ...


class FrqCatStyle(ABC):
    """The contract every FRQ CAT style implements.

    A style supplies the genuinely style-specific pieces (its IRT model and its
    criterion-selection policy) while reusing the shared utilities for the rest.
    The generic engine calls these methods in order: download the bank, load IRT
    parameters, then repeatedly select a criterion, have it generated + judged,
    estimate ability, and test the stopping rule before producing a report.

    Concrete styles must be constructible with no required arguments; per-style
    configuration is loaded by the style itself (conventionally from
    ``styles/<name>/config.yaml`` and the graduated bank under ``styles/<name>/bank/``).
    """

    #: Registry name, set by the ``@register(...)`` decorator.
    name: ClassVar[str]

    @abstractmethod
    def download_bank(self, benchmark: str, *, dest: Path | None = None) -> FrqBank:
        """Download and prepare the scenarios + criteria for ``benchmark``."""
        ...

    @abstractmethod
    def load_irt_params(self, source: str | Path) -> IRTBank:
        """Load IRT criterion parameters for the bank from ``source``."""
        ...

    @abstractmethod
    def estimate_ability(
        self,
        bank: IRTBank,
        responses: Sequence[CriterionResponse],
        *,
        previous: AbilityEstimate | None = None,
    ) -> AbilityEstimate:
        """Estimate ability (theta and standard error) from graded responses.

        This is the IRT-model-specific update the generic engine performs after
        each administered criterion; ``previous`` is the prior estimate, if any.
        """
        ...

    @abstractmethod
    def select_next_item(self, bank: IRTBank, state: CATState) -> str | None:
        """Choose the next criterion id to administer, or ``None`` when none remain."""
        ...

    @abstractmethod
    def stopping_rule(self, state: CATState) -> bool:
        """Return ``True`` when the session should stop (for example on SE)."""
        ...

    @abstractmethod
    def report(self, state: CATState) -> CATReport:
        """Summarize a completed session into a serializable report."""
        ...
