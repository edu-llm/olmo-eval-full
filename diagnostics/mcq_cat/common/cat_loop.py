"""The generic Computerized Adaptive Testing engine.

This engine is style-agnostic: it drives any :class:`~diagnostics.mcq_cat.base.CatStyle`
through the standard CAT loop (select an item, administer/score it, update the
ability estimate, then test the stopping rule) and returns the style's report.
All model- and selection-specific behavior lives behind the ``base.py`` interface,
so this module never needs to change when a style is added.
"""

from __future__ import annotations

import logging

from ..base import (
    BenchmarkBank,
    CATReport,
    CATState,
    CatStyle,
    IRTBank,
    ScoringModel,
)

log = logging.getLogger("mcq_cat.cat_loop")


def run_cat(
    style: CatStyle,
    *,
    bank: BenchmarkBank,
    irt_bank: IRTBank,
    model: ScoringModel,
    se_threshold: float | None = None,
    max_items: int | None = None,
) -> CATReport:
    """Run one adaptive test session and return the style's report.

    Args:
        style: The resolved CAT style supplying the IRT model and selector.
        bank: The prepared benchmark items.
        irt_bank: IRT parameters for the benchmark items.
        model: The checkpoint-backed log-likelihood MCQ scorer.
        se_threshold: Standard-error stop threshold, passed to the style's rule.
        max_items: Hard cap on administered items, enforced by the engine.

    Returns:
        The completed :class:`CATReport` produced by ``style.report``.
    """
    state = CATState(
        benchmark=bank.name,
        se_threshold=se_threshold,
        max_items=max_items,
    )

    while True:
        if max_items is not None and state.step >= max_items:
            log.info("Stopping: reached max_items=%d", max_items)
            break

        next_id = style.select_next_item(irt_bank, state)
        if next_id is None:
            log.info("Stopping: no further items to administer")
            break

        item = bank.get(next_id)
        responses = style.score(model, [item])
        state.administered.extend(responses)

        state.ability = style.estimate_ability(irt_bank, state.administered, previous=state.ability)
        log.info(
            "Administered %s (step %d): theta=%s se=%s",
            next_id,
            state.step,
            state.ability.theta,
            state.ability.standard_error,
        )

        if style.stopping_rule(state):
            log.info("Stopping: style stopping rule satisfied at step %d", state.step)
            break

    return style.report(state)
