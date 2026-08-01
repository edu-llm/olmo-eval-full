"""Config-driven ATLAS offline adaptive-testing tasks (Phase 1, multi-benchmark).

:class:`AtlasAdaptiveMixin` appends an ATLAS Fisher-information CAT report to any
olmo-eval task: the task runs unchanged, then ``compute_metrics`` derives
per-item 0/1 from the configured primary metric, joins to the calibrated 3PL
bank on ``metadata["id"]``, runs the CAT, and appends ``atlas_theta / atlas_se /
atlas_n_items / atlas_pirt_accuracy`` (+ ``atlas_bank_version``).

Because the per-item cell comes from the task's own primary metric, the mixin is
scoring-agnostic: MCQ tasks feed it a loglik-argmax accuracy (0/1) and gsm8k
feeds it a generative exact-match accuracy (0/1). No generative branch is needed
here -- the runner scores responses before ``compute_metrics``, so the metric's
``compute_instance`` already returns exact-match 0/1 for gsm8k.

One ``atlas_<name>`` task is registered per benchmark in
:func:`~olmo_eval.adaptive.benchmarks.cat_benchmarks` by combining the mixin with
that benchmark's base task. ARC-Challenge keeps its dedicated module
(:mod:`olmo_eval.evals.tasks.atlas_arc`) so ``atlas_arc_challenge`` and the
``AtlasARCChallenge`` symbol stay stable.

Banks are supplied out of band (synced from S3 to local disk at run time); when
a benchmark's bank directory/files are absent, ``load_bank`` raises
``FileNotFoundError`` and the task degrades to base metrics with a warning.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import ClassVar

from olmo_eval.adaptive import TableResponder, load_bank, run_cat
from olmo_eval.adaptive.benchmarks import AtlasBenchmark, bank_dir_for, cat_benchmarks
from olmo_eval.adaptive.cat import DEFAULT_MAX_ITEMS, DEFAULT_MIN_ITEMS, DEFAULT_SE_STOP
from olmo_eval.common.types import Response
from olmo_eval.evals.tasks.arc import ARCChallenge
from olmo_eval.evals.tasks.common import Task, register
from olmo_eval.evals.tasks.csqa import CommonsenseQA
from olmo_eval.evals.tasks.gsm8k import GSM8K
from olmo_eval.evals.tasks.hellaswag import HellaSwag
from olmo_eval.evals.tasks.ifeval import IFEval
from olmo_eval.evals.tasks.leaderboard_math import LeaderboardMath
from olmo_eval.evals.tasks.piqa import PiQA
from olmo_eval.evals.tasks.winogrande import Winogrande

logger = logging.getLogger(__name__)


class AtlasAdaptiveMixin(Task):
    """Mixin appending an ATLAS adaptive-testing report to an olmo-eval task.

    Combine with an olmo-eval :class:`Task` subclass, mixin first in the bases
    (e.g. ``class Foo(AtlasAdaptiveMixin, HellaSwag)``). ``compute_metrics``
    calls the base implementation, then runs the CAT over per-item correctness.

    Per-item correctness comes from the task's configured primary metric, so the
    same code path serves both MCQ (loglik-argmax accuracy) and generative
    (gsm8k exact-match accuracy) tasks: both resolve to a 0/1 cell via the
    metric's ``compute_instance``. The bank must have been calibrated on
    responses scored the same way, or item parameters will not transfer -- see
    ``AdaptiveTesting/docs/02_atlas_and_adaptive_testing.md`` section 2. Two
    guards skip the adaptive report instead of feeding a bad IRT cell: an
    unresolvable primary metric, or a metric producing non-binary per-item
    values (bpb, perplexity).
    """

    #: The benchmark this task adapts; set by subclasses / the factory below.
    atlas_benchmark: ClassVar[AtlasBenchmark]

    # ATLAS knobs (bank directory defaults to the benchmark's vendored subdir).
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
            raw_id = response.instance.metadata.get("id")
            if raw_id is None:
                continue
            # The bank keys items by string question_id, while several base tasks
            # emit an int id (native `ind` or a positional index). Normalize both
            # sides to str so the join is stable; the bridge for positional-id
            # benchmarks (winogrande/piqa) must be built against olmo-eval's split
            # ordering, otherwise there is simply no overlap (skip, not misalign).
            qid = str(raw_id)
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

        try:
            bank = load_bank(bank_dir_for(self.atlas_benchmark, self.atlas_bank_dir))
        except FileNotFoundError as exc:
            logger.warning(
                "ATLAS %s bank unavailable, skipping adaptive report: %s",
                self.atlas_benchmark.name,
                exc,
            )
            return result

        bank = bank.restrict(set(scores))
        if len(bank) == 0:
            logger.warning(
                "no overlap between ATLAS %s bank and %d scored items; skipping adaptive report%s",
                self.atlas_benchmark.name,
                len(scores),
                " (positional-id benchmark: bridge ordering may not match)"
                if self.atlas_benchmark.positional_id
                else "",
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
            "ATLAS CAT [%s]: theta=%.4f se=%.4f n_items=%d pirt_acc=%.4f bank=%s",
            self.atlas_benchmark.name,
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


# Base task class + dynamic class name for each MCQ benchmark. ARC is handled in
# atlas_arc.py to preserve its explicit class and registration.
_BASE_CLASSES: dict[str, type[Task]] = {
    "arc_challenge": ARCChallenge,
    "hellaswag": HellaSwag,
    "winogrande": Winogrande,
    "csqa": CommonsenseQA,
    "piqa": PiQA,
    "gsm8k": GSM8K,
    "ifeval": IFEval,
    "math": LeaderboardMath,
}
_CLASS_NAMES: dict[str, str] = {
    "hellaswag": "AtlasHellaSwag",
    "winogrande": "AtlasWinogrande",
    "csqa": "AtlasCommonsenseQA",
    "piqa": "AtlasPiQA",
    "gsm8k": "AtlasGSM8K",
    "ifeval": "AtlasIFEval",
    "math": "AtlasLeaderboardMath",
}


def make_atlas_task(benchmark: AtlasBenchmark) -> type[Task]:
    """Build an ``atlas_<name>`` task class from the mixin + the base task."""
    base = _BASE_CLASSES[benchmark.name]
    class_name = _CLASS_NAMES.get(benchmark.name, f"Atlas_{benchmark.name}")
    return type(
        class_name,
        (AtlasAdaptiveMixin, base),
        {
            "atlas_benchmark": benchmark,
            "__module__": __name__,
            "__qualname__": class_name,
            "__doc__": f"{base.__name__} with an appended ATLAS adaptive-testing report.",
        },
    )


def _register_atlas_tasks() -> None:
    for benchmark in cat_benchmarks():
        if benchmark.name == "arc_challenge":
            continue  # registered in atlas_arc.py to keep AtlasARCChallenge stable
        register(benchmark.offline_task_name)(make_atlas_task(benchmark))


_register_atlas_tasks()
