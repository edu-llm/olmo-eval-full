"""The generic Computerized Adaptive Testing engine.

This engine is style-agnostic: it drives any :class:`~diagnostics.mcq_cat.base.CatStyle`
through the standard CAT loop (select an item, administer/score it, update the
ability estimate, then test the stopping rule) and returns the style's report.
All model- and selection-specific behavior lives behind the ``base.py`` interface,
so this module never needs to change when a style is added.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from ..base import (
    BenchmarkBank,
    CATReport,
    CATState,
    CatStyle,
    IRTBank,
    ScoringModel,
)

log = logging.getLogger("mcq_cat.cat_loop")

#: Signature of the streaming hook :func:`run_full_bank` calls as items accumulate.
#: It receives the live :class:`CATState` and the total number of items the run will
#: administer, and is expected to persist a snapshot. It never influences the run.
ProgressCallback = Callable[[CATState, int], None]


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


def run_full_bank(
    style: CatStyle,
    *,
    bank: BenchmarkBank,
    irt_bank: IRTBank,
    model: ScoringModel,
    batch_size: int = 16,
    progress_callback: ProgressCallback | None = None,
    progress_every: int = 100,
) -> CATReport:
    """Administer every calibrated item in the bank once and return the style's report.

    This is the non-adaptive counterpart to :func:`run_cat`. Where the CAT loop selects
    an informative subset and stops on precision, this administers the whole bank in its
    canonical order, so ``observed_accuracy`` in the report is the model's accuracy over
    every calibrated item rather than over a selected subset. Ability is still estimated
    by the style, refit after each batch, so the report's theta is the batch estimate over
    the complete response set. The style's selector and stopping rule are not consulted.

    Only items that carry IRT parameters are administered: ability estimation indexes the
    calibrated arrays by item id, so an item with a stem but no parameters has nowhere to
    land and is skipped exactly as it is for selection. ``max_items`` on the state is set
    to that count so the report records the full-bank length rather than the style's
    pinned CAT cap, which also marks the run as not comparable with a capped CAT session.

    Args:
        style: The resolved CAT style; its scorer, estimator and report are used.
        bank: The prepared benchmark items.
        irt_bank: IRT parameters; items absent from it are skipped.
        model: The checkpoint-backed scorer.
        batch_size: How many items are scored per forward batch and per ability refit.
        progress_callback: Optional hook invoked with ``(state, total)`` as batches land,
            for streaming a partial artifact. It cannot change what the run administers.
        progress_every: Fire ``progress_callback`` each time this many new items have been
            administered since the last fire (and once at the end). Throttles streaming
            writes so a long run does not upload after every small batch.

    Returns:
        The completed :class:`CATReport` produced by ``style.report``.
    """
    usable = [item for item in bank.items if item.item_id in irt_bank.params]
    total = len(usable)
    skipped = len(bank.items) - total
    if skipped:
        log.warning(
            "Full-bank: %d of %d items lack IRT parameters and are skipped.",
            skipped,
            len(bank.items),
        )
    if total == 0:
        raise ValueError(
            f"Full-bank run of {bank.name!r} has no items with IRT parameters to "
            f"administer. items.jsonl and params.json disagree."
        )

    state = CATState(benchmark=bank.name, se_threshold=None, max_items=total)
    log.info("Full-bank: administering all %d calibrated items in batches of %d", total, batch_size)

    last_fired = 0
    for start in range(0, total, batch_size):
        batch = usable[start : start + batch_size]
        state.administered.extend(style.score(model, batch))
        state.ability = style.estimate_ability(irt_bank, state.administered, previous=state.ability)
        log.info(
            "Full-bank progress %d/%d: theta=%s se=%s",
            state.step,
            total,
            state.ability.theta,
            state.ability.standard_error,
        )
        if progress_callback is not None and state.step - last_fired >= progress_every:
            progress_callback(state, total)
            last_fired = state.step

    if progress_callback is not None and state.step != last_fired:
        progress_callback(state, total)

    return style.report(state)
