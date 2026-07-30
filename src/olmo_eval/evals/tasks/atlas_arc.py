"""ATLAS adaptive-testing task (offline / Phase 1).

Runs the standard ARC-Challenge log-likelihood eval over the full item set, then
simulates an ATLAS Fisher-information CAT over the recorded per-item correctness
to report an ability estimate (theta), its standard error, the number of items a
live adaptive test would have used, and p-IRT reconstructed accuracy.

This does not save inference (the whole benchmark is still run); it proves the
bank, id-alignment, and CAT math end to end through the normal olmo-eval run /
Beaker path. The online variant that only queries the selected items lives in
``olmo_eval/evals/external/benchmarks/atlas_arc`` (Phase 2) and shares the same
``olmo_eval.adaptive`` core.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence

from olmo_eval.adaptive import TableResponder, load_bank, run_cat
from olmo_eval.adaptive.cat import DEFAULT_MAX_ITEMS, DEFAULT_MIN_ITEMS, DEFAULT_SE_STOP
from olmo_eval.common.types import Response
from olmo_eval.evals.tasks.arc import ARCChallenge
from olmo_eval.evals.tasks.common import register

logger = logging.getLogger(__name__)


@register("atlas_arc_challenge")
class AtlasARCChallenge(ARCChallenge):
    """ARC-Challenge with an appended ATLAS adaptive-testing report.

    Inherits ARC-Challenge instances, formatting, and scoring unchanged; the
    native ARC ``question_id`` is preserved in ``Instance.metadata["id"]`` and
    used to join responses to the 3PL bank.

    Per-item correctness comes from the task's configured primary metric, so
    scoring variants (for example ``LogprobUncondMCAccuracyMetric`` for PMI
    normalization) flow through to the CAT. The bank must have been calibrated
    on responses scored the same way, or item parameters will not transfer --
    see ``AdaptiveTesting/docs/02_atlas_and_adaptive_testing.md`` section 2.
    """

    # ATLAS knobs (bank directory defaults to the vendored ARC bank / env var).
    atlas_bank_dir: str | None = None
    atlas_se_stop: float = DEFAULT_SE_STOP
    atlas_min_items: int = DEFAULT_MIN_ITEMS
    atlas_max_items: int = DEFAULT_MAX_ITEMS

    def compute_metrics(self, responses: Sequence[Response]) -> dict[str, dict[str, float]]:
        result = super().compute_metrics(responses)

        instance_metric = self.config.get_primary_metric()
        if instance_metric is None:
            logger.warning(
                "no resolvable primary metric; skipping adaptive report. Set "
                "TaskConfig.primary_metric when the task defines several metrics."
            )
            return result

        scores: dict[str, int] = {}
        non_binary = 0
        for response in responses:
            qid = response.instance.metadata.get("id")
            if qid is None:
                continue
            score = instance_metric.compute_instance(response)
            if score is None:
                continue
            # IRT models a binary correct/incorrect cell; a continuous metric
            # (bpb, perplexity) would silently truncate into a meaningless code.
            if score not in (0.0, 1.0):
                non_binary += 1
                continue
            scores[qid] = int(score)

        if non_binary:
            logger.warning(
                "%s produced %d non-binary per-item scores; ATLAS needs 0/1 "
                "correctness, skipping adaptive report",
                type(instance_metric).__name__,
                non_binary,
            )
            return result

        bank_dir = self.atlas_bank_dir or os.environ.get("OLMO_EVAL_ATLAS_BANK_DIR")
        try:
            bank = load_bank(bank_dir)
        except FileNotFoundError as exc:
            logger.warning("ATLAS bank unavailable, skipping adaptive report: %s", exc)
            return result

        bank = bank.restrict(set(scores))
        if len(bank) == 0:
            logger.warning(
                "no overlap between ATLAS ARC bank and %d scored items; skipping adaptive report",
                len(scores),
            )
            return result

        cat = run_cat(
            bank,
            TableResponder(scores),
            se_stop=self.atlas_se_stop,
            min_items=self.atlas_min_items,
            max_items=self.atlas_max_items,
        )
        logger.info(
            "ATLAS CAT: theta=%.4f se=%.4f n_items=%d pirt_acc=%.4f bank=%s",
            cat.theta,
            cat.se,
            cat.n_items,
            cat.pirt_accuracy,
            cat.bank_version,
        )

        result["atlas_theta"] = {"atlas": cat.theta}
        result["atlas_se"] = {"atlas": cat.se}
        result["atlas_n_items"] = {"atlas": float(cat.n_items)}
        result["atlas_pirt_accuracy"] = {"atlas": cat.pirt_accuracy}
        # Record bank provenance as a float-valued map (version string as key).
        result["atlas_bank_version"] = {cat.bank_version: 1.0}
        return result
