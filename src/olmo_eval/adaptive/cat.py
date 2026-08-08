"""Fisher-information computerized adaptive testing (CAT) + p-IRT accuracy.

:class:`CatSession` is the single source of truth for the adaptive loop: it
selects the next item by maximum Fisher information at the current ability
estimate and re-estimates ability (EAP) after each observed response. Both the
offline entry point (responses looked up from a precomputed table) and the
online entry point (responses fetched live from an inference provider) drive the
same session, so their selection and stopping behavior cannot diverge.
"""

from __future__ import annotations

from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from olmo_eval.adaptive.bank import ItemBank
from olmo_eval.adaptive.irt import eap_theta_se, fisher_info, prob

DEFAULT_SE_STOP = 0.3
DEFAULT_MIN_ITEMS = 8
DEFAULT_MAX_ITEMS = 200


class Responder(Protocol):
    """Returns the binary score (1 correct / 0 wrong) for a ``question_id``."""

    def __call__(self, question_id: str) -> int: ...


class AsyncResponder(Protocol):
    """Async responder that fetches a live 0/1 score for a ``question_id``."""

    def __call__(self, question_id: str) -> Awaitable[int]: ...


class TableResponder:
    """Offline responder backed by a precomputed ``question_id -> 0/1`` table."""

    def __init__(self, scores: dict[str, int]) -> None:
        self._scores = scores

    def __call__(self, question_id: str) -> int:
        return int(self._scores[question_id])


@dataclass
class CatResult:
    """Outcome of an adaptive session."""

    theta: float
    se: float
    order: list[int]
    scores: list[int]
    selected_question_ids: list[str]
    pirt_accuracy: float
    n_items: int
    bank_version: str


class CatSession:
    """Stateful Fisher-information CAT over a fixed item bank.

    Usage (driver supplies responses however it likes)::

        s = CatSession(bank)
        while (i := s.next_item()) is not None:
            s.record(i, get_score(bank.question_ids[i]))
        result = s.result()
    """

    def __init__(
        self,
        bank: ItemBank,
        *,
        se_stop: float = DEFAULT_SE_STOP,
        min_items: int = DEFAULT_MIN_ITEMS,
        max_items: int = DEFAULT_MAX_ITEMS,
    ) -> None:
        self.bank = bank
        self.se_stop = se_stop
        self.min_items = min_items
        self.max_items = min(max_items, len(bank))
        self._used = np.zeros(len(bank), dtype=bool)
        self.theta: float = 0.0
        self.se: float = 1.0
        self.order: list[int] = []
        self.scores: list[int] = []

    def _stop(self) -> bool:
        return len(self.order) >= self.min_items and self.se <= self.se_stop

    def next_item(self) -> int | None:
        """Index of the next item to administer, or ``None`` to stop."""
        if len(self.order) >= self.max_items or self._stop():
            return None
        info = fisher_info(self.theta, self.bank.a, self.bank.b, self.bank.c)
        info = np.where(self._used, -1.0, info)
        return int(np.argmax(info))

    def record(self, idx: int, score: int) -> None:
        """Record the observed score for item ``idx`` and re-estimate ability."""
        if self._used[idx]:
            raise ValueError(f"item {idx} already administered")
        self._used[idx] = True
        self.order.append(idx)
        self.scores.append(int(score))
        sel = np.asarray(self.order)
        self.theta, self.se = eap_theta_se(
            np.asarray(self.scores, dtype=float),
            self.bank.a[sel],
            self.bank.b[sel],
            self.bank.c[sel],
        )

    def result(self) -> CatResult:
        return CatResult(
            theta=self.theta,
            se=self.se,
            order=list(self.order),
            scores=list(self.scores),
            selected_question_ids=[self.bank.question_ids[i] for i in self.order],
            pirt_accuracy=pirt_accuracy(self.bank, self.order, self.scores, self.theta),
            n_items=len(self.order),
            bank_version=self.bank.version,
        )


def pirt_accuracy(bank: ItemBank, order: list[int], scores: list[int], theta: float) -> float:
    """ATLAS p-IRT: blend observed subset accuracy with IRT-predicted accuracy.

    The administered items contribute their observed 0/1; the unseen items
    contribute their model-predicted success probability at the final ``theta``.
    Blended by the fraction of items actually observed.
    """
    n = len(bank)
    if n == 0:
        return 0.0
    subset = set(order)
    avg_obs = float(np.mean(scores)) if scores else 0.0
    unobs = [i for i in range(n) if i not in subset]
    if unobs:
        idx = np.asarray(unobs)
        avg_pred = float(prob(theta, bank.a[idx], bank.b[idx], bank.c[idx]).mean())
    else:
        avg_pred = avg_obs
    w_obs = len(order) / n
    return w_obs * avg_obs + (1 - w_obs) * avg_pred


def run_cat(
    bank: ItemBank,
    responder: Responder,
    *,
    se_stop: float = DEFAULT_SE_STOP,
    min_items: int = DEFAULT_MIN_ITEMS,
    max_items: int = DEFAULT_MAX_ITEMS,
) -> CatResult:
    """Drive a full adaptive session with a synchronous responder (offline)."""
    session = CatSession(bank, se_stop=se_stop, min_items=min_items, max_items=max_items)
    while (idx := session.next_item()) is not None:
        score = responder(bank.question_ids[idx])
        session.record(idx, score)
    return session.result()


async def run_cat_async(
    bank: ItemBank,
    responder: AsyncResponder,
    *,
    se_stop: float = DEFAULT_SE_STOP,
    min_items: int = DEFAULT_MIN_ITEMS,
    max_items: int = DEFAULT_MAX_ITEMS,
) -> CatResult:
    """Drive a full adaptive session with an async responder (online).

    Only the items the loop selects are ever passed to ``responder`` (e.g. a
    live inference provider), which is what makes online CAT cheaper than a full
    benchmark run. Selection and stopping are identical to :func:`run_cat`.
    """
    session = CatSession(bank, se_stop=se_stop, min_items=min_items, max_items=max_items)
    while (idx := session.next_item()) is not None:
        score = await responder(bank.question_ids[idx])
        session.record(idx, score)
    return session.result()
