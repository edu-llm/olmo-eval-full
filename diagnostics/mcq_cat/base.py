"""Stable interface contract for MCQ CAT diagnostic styles.

This module is the frozen contract every Computerized Adaptive Testing (CAT)
style implements. Styles live under ``styles/<name>/`` and self-register with the
registry; the generic engine in :mod:`diagnostics.mcq_cat.common.cat_loop` drives
each style purely through the :class:`CatStyle` interface defined here.

The dataclasses below are the shared vocabulary passed between the engine, the
shared utilities, and each style. They are intentionally minimal and
model-agnostic so unidimensional and multidimensional (MIRT) styles share one
contract: abilities and discriminations are scalars for unidimensional models and
tuples for multidimensional ones.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Protocol, runtime_checkable

#: Ability is a scalar for unidimensional IRT and a vector for MIRT.
Ability = float | tuple[float, ...]
#: Discrimination is a scalar for 1PL/2PL/3PL and a vector for multidimensional IRT.
Discrimination = float | tuple[float, ...]


@dataclass(frozen=True, slots=True)
class BenchmarkItem:
    """A single multiple-choice item drawn from a benchmark."""

    item_id: str
    question: str
    choices: tuple[str, ...]
    gold_index: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BenchmarkBank:
    """A prepared collection of MCQ items for one benchmark."""

    name: str
    items: tuple[BenchmarkItem, ...]

    def __len__(self) -> int:
        return len(self.items)

    def get(self, item_id: str) -> BenchmarkItem:
        """Return the item with ``item_id`` or raise ``KeyError``."""
        for item in self.items:
            if item.item_id == item_id:
                return item
        raise KeyError(item_id)


@dataclass(frozen=True, slots=True)
class IRTItemParams:
    """IRT parameters for one item.

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
    """IRT parameters for a benchmark, keyed by item id.

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
class ItemResponse:
    """The model's graded response to one administered item."""

    item_id: str
    chosen_index: int
    correct: bool
    choice_logprobs: tuple[float, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CATState:
    """Mutable state threaded through a single adaptive test session.

    The generic engine owns this object: it appends responses, refreshes
    ``ability`` after each item, and consults the style's stopping rule. The
    stopping parameters supplied by the runner (``se_threshold``, ``max_items``)
    live here so a style's stopping rule can read them without a shared config.
    """

    benchmark: str
    se_threshold: float | None = None
    max_items: int | None = None
    administered: list[ItemResponse] = field(default_factory=list)
    ability: AbilityEstimate | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def step(self) -> int:
        """Number of items administered so far."""
        return len(self.administered)

    @property
    def administered_ids(self) -> set[str]:
        """Ids of items already administered this session."""
        return {response.item_id for response in self.administered}


@dataclass(frozen=True, slots=True)
class CATReport:
    """The serializable outcome of a completed CAT session."""

    cat_style: str
    benchmark: str
    ability: AbilityEstimate
    num_items_administered: int
    responses: tuple[ItemResponse, ...]
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
                    "item_id": response.item_id,
                    "chosen_index": response.chosen_index,
                    "correct": response.correct,
                    "choice_logprobs": list(response.choice_logprobs),
                    "metadata": response.metadata,
                }
                for response in self.responses
            ],
            "metadata": self.metadata,
        }


@runtime_checkable
class ScoringModel(Protocol):
    """A checkpoint-backed model that grades items and returns one response each.

    How an item is graded depends on its modality, and this protocol deliberately
    does not say which: :mod:`diagnostics.mcq_cat.common.inference` scores every
    choice by continuation log-likelihood and takes the argmax, while
    :mod:`diagnostics.mcq_cat.common.generative` samples a completion and matches an
    extracted answer. Both reduce to the same binary outcome, which is the only thing
    the IRT layer reads, so the CAT engine drives either one unchanged.
    :mod:`diagnostics.mcq_cat.common.grading` maps a modality to its grader.
    """

    def score_items(self, items: Sequence[BenchmarkItem]) -> list[ItemResponse]:
        """Grade ``items`` and return one response per item, in input order."""
        ...


class CatStyle(ABC):
    """The contract every MCQ CAT style implements.

    A style supplies the genuinely style-specific pieces (its IRT model and its
    item-selection policy) while reusing the shared utilities for the rest. The
    generic engine calls these methods in order: download the benchmark, load IRT
    parameters, then repeatedly select, score, estimate ability, and test the
    stopping rule before producing a report.

    Concrete styles must be constructible with no required arguments; per-style
    configuration is loaded by the style itself (conventionally from
    ``styles/<name>/config.yaml``).
    """

    #: Registry name, set by the ``@register(...)`` decorator.
    name: ClassVar[str]

    @abstractmethod
    def download_benchmark(self, benchmark: str, *, dest: Path | None = None) -> BenchmarkBank:
        """Download and prepare the MCQ items for ``benchmark``."""
        ...

    @abstractmethod
    def load_irt_params(self, source: str | Path) -> IRTBank:
        """Load IRT item parameters for the benchmark from ``source``."""
        ...

    @abstractmethod
    def score(self, model: ScoringModel, items: Sequence[BenchmarkItem]) -> list[ItemResponse]:
        """Grade ``items`` with ``model``, which already matches the bank's modality."""
        ...

    @abstractmethod
    def estimate_ability(
        self,
        bank: IRTBank,
        responses: Sequence[ItemResponse],
        *,
        previous: AbilityEstimate | None = None,
    ) -> AbilityEstimate:
        """Estimate ability (theta and standard error) from graded responses.

        This is the IRT-model-specific update the generic engine performs after
        each administered item; ``previous`` is the prior estimate, if any.
        """
        ...

    @abstractmethod
    def select_next_item(self, bank: IRTBank, state: CATState) -> str | None:
        """Choose the next item id to administer, or ``None`` when none remain."""
        ...

    @abstractmethod
    def stopping_rule(self, state: CATState) -> bool:
        """Return ``True`` when the session should stop (for example on SE)."""
        ...

    @abstractmethod
    def report(self, state: CATState) -> CATReport:
        """Summarize a completed session into a serializable report."""
        ...
