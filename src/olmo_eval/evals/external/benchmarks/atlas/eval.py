"""Config-driven online ATLAS adaptive-testing external evals (Phase 2).

:class:`AtlasExternalEval` runs an ATLAS Fisher-information CAT that queries the
model live, item by item: at each step it picks the highest-information unused
item at the current ability estimate, sends exactly that one log-likelihood
request to the provider, updates the estimate, and stops once the standard error
crosses a threshold. Only the selected items (~tens, not the full benchmark) are
ever sent to the model.

Selection, ability estimation, and stopping all come from the shared
``olmo_eval.adaptive`` core, so this cannot diverge from the offline
``atlas_<name>`` tasks. Item formatting and correctness reuse the benchmark's
real olmo-eval task for scoring parity.

Per-item correctness dispatches on ``AtlasBenchmark.scoring``:

- ``mcq_loglik`` -- score all choice continuations by log-likelihood and take
  the argmax vs ``gold_idx`` (arc/hellaswag/winogrande/csqa/piqa). Unchanged.
- ``generative`` -- send one generation request, then reuse the base task's
  answer extraction + primary-metric scoring to get an exact-match 0/1 (gsm8k).

One eval is registered per benchmark in
:func:`~olmo_eval.adaptive.benchmarks.cat_benchmarks`; ARC keeps the historical
``atlas_arc`` name via :mod:`olmo_eval.evals.external.benchmarks.atlas_arc`.
Banks are synced from S3 to local disk at run time; a missing bank yields a
clear error result.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from olmo_eval.adaptive import CatSession, load_bank
from olmo_eval.adaptive.benchmarks import (
    SCORING_GENERATIVE,
    AtlasBenchmark,
    bank_dir_for,
    cat_benchmarks,
)
from olmo_eval.adaptive.cat import DEFAULT_MAX_ITEMS, DEFAULT_MIN_ITEMS, DEFAULT_SE_STOP
from olmo_eval.common.types import Instance, Response
from olmo_eval.evals.external.base import ExternalEval
from olmo_eval.evals.external.result import ExternalEvalResult

if TYPE_CHECKING:
    from olmo_eval.evals.tasks.common import Task
    from olmo_eval.inference.base import InferenceProvider

logger = logging.getLogger(__name__)


class AtlasExternalEval(ExternalEval):
    """Online CAT over one MCQ benchmark using its calibrated ATLAS 3PL bank."""

    def __init__(self, benchmark: AtlasBenchmark) -> None:
        self.benchmark = benchmark

    @property
    def name(self) -> str:
        return self.benchmark.online_eval_name

    @property
    def description(self) -> str:
        return (
            f"Online ATLAS 3PL adaptive test over {self.benchmark.base_task} "
            "(item-by-item, early stop)."
        )

    @property
    def timeout_seconds(self) -> float:
        return 3600.0

    @property
    def arguments(self) -> dict[str, tuple[str, Any | None]]:
        return {
            "se_stop": ("Stop once the ability standard error drops to this.", DEFAULT_SE_STOP),
            "min_items": ("Minimum items to administer before stopping.", DEFAULT_MIN_ITEMS),
            "max_items": ("Maximum items to administer.", DEFAULT_MAX_ITEMS),
            "bank_path": (
                "Directory with the ATLAS bank CSVs (defaults to the benchmark's "
                "vendored subdir / $OLMO_EVAL_ATLAS_BANK_DIR for arc).",
                None,
            ),
            "task": ("olmo-eval task providing items/formatting.", self.benchmark.base_task),
            "seed": ("Seed recorded for provenance (selection is deterministic).", 0),
        }

    def _load_items(self, args: dict[str, Any]) -> tuple[Task, dict[str, Instance]]:
        """Load the benchmark task and index its instances by ``question_id``.

        Split out so tests can inject a tiny synthetic item set without touching
        the network. Ids are stringified to match the bank's string keys.
        """
        from olmo_eval.evals.tasks.common import get_task

        task = get_task(str(args.get("task") or self.benchmark.base_task))
        items: dict[str, Instance] = {}
        for instance in task.instances:
            qid = instance.metadata.get("id")
            if qid is not None:
                items[str(qid)] = instance
        return task, items

    async def _score_item(self, task: Task, provider: InferenceProvider, instance: Instance) -> int:
        """Query the provider for one item and return 1 (correct) / 0 (wrong).

        Dispatches on ``self.benchmark.scoring`` so MCQ benchmarks keep the exact
        loglik-argmax path while gsm8k uses generation + exact match. Both reuse
        the task's own ``format_request``/scoring so the online score matches how
        ``olmo-eval`` would score the item in a full run.
        """
        if self.benchmark.scoring == SCORING_GENERATIVE:
            return await self._score_item_generative(task, provider, instance)
        return await self._score_item_mcq(task, provider, instance)

    async def _score_item_mcq(
        self, task: Task, provider: InferenceProvider, instance: Instance
    ) -> int:
        """MCQ correctness: log-likelihood argmax over choice continuations."""
        request = task.format_request(instance)
        outputs = await provider.alogprobs([request])
        continuations = outputs[0]
        sums: list[float] = []
        for output in continuations:
            token_lps = [t["logprob"] for t in (output.logprobs or []) if "logprob" in t]
            sums.append(sum(token_lps) if token_lps else float("-inf"))
        gold_idx = instance.metadata.get("gold_idx", 0)
        return 1 if sums.index(max(sums)) == gold_idx else 0

    async def _score_item_generative(
        self, task: Task, provider: InferenceProvider, instance: Instance
    ) -> int:
        """Generative correctness: generate, then reuse the base task's scoring.

        Calls the provider's generation API (not ``alogprobs``), builds a
        one-output :class:`Response`, and runs the task's own
        ``score_responses`` (answer extraction + configured scorers). The
        primary metric's ``compute_instance`` then yields exact-match 0/1 -- the
        same value the offline ``atlas_gsm8k`` task derives -- so we never
        reimplement gsm8k answer parsing here.
        """
        request = task.format_request(instance)
        outputs = await provider.agenerate([request], task.get_sampling_params(instance))
        samples = outputs[0] if outputs else []
        if not samples:
            return 0
        response = Response(instance=instance, request=request, outputs=list(samples))
        await task.score_responses([response])
        metric = task.config.get_primary_metric()
        value = metric.compute_instance(response) if metric is not None else None
        return 1 if value == 1.0 else 0

    async def execute(
        self,
        provider: InferenceProvider,
        args: dict[str, Any],
        output_dir: str | None = None,
        container_runtime: str = "podman",
    ) -> ExternalEvalResult:
        start = time.time()
        se_stop = float(args.get("se_stop", DEFAULT_SE_STOP))
        min_items = int(args.get("min_items", DEFAULT_MIN_ITEMS))
        max_items = int(args.get("max_items", DEFAULT_MAX_ITEMS))
        bank_path = args.get("bank_path")

        try:
            bank = load_bank(bank_dir_for(self.benchmark, bank_path))
        except FileNotFoundError as exc:
            return self._error_result(f"ATLAS bank unavailable: {exc}", start)

        try:
            task, items = self._load_items(args)
        except Exception as exc:  # noqa: BLE001 - surface any loading failure as a result
            return self._error_result(f"failed to load {self.benchmark.name} items: {exc}", start)

        bank = bank.restrict(set(items))
        if len(bank) == 0:
            return self._error_result(
                f"no overlap between the ATLAS {self.benchmark.name} bank and the loaded items",
                start,
            )

        session = CatSession(bank, se_stop=se_stop, min_items=min_items, max_items=max_items)
        while (idx := session.next_item()) is not None:
            qid = bank.question_ids[idx]
            score = await self._score_item(task, provider, items[qid])
            session.record(idx, score)

        result = session.result()
        logger.info(
            "[%s] online CAT queried %d/%d items: theta=%.4f se=%.4f pirt=%.4f",
            self.name,
            result.n_items,
            len(bank),
            result.theta,
            result.se,
            result.pirt_accuracy,
        )

        eval_result = ExternalEvalResult(
            name=self.name,
            metrics={
                "pirt_accuracy": result.pirt_accuracy,
                "theta": result.theta,
                "se": result.se,
                "n_items": float(result.n_items),
                "bank_size": float(len(bank)),
            },
            metadata={
                "benchmark": self.benchmark.name,
                "bank_version": result.bank_version,
                "selected_question_ids": result.selected_question_ids,
                "se_stop": se_stop,
                "min_items": min_items,
                "max_items": max_items,
                "seed": args.get("seed", 0),
            },
            predictions=[
                {"question_id": q, "score": s}
                for q, s in zip(result.selected_question_ids, result.scores, strict=True)
            ],
            duration_seconds=time.time() - start,
        )
        if output_dir:
            self._save_results(eval_result, output_dir)
        return eval_result


def make_atlas_external_evals() -> list[AtlasExternalEval]:
    """Online evals for every wired benchmark except ARC (registered separately).

    Covers the MCQ set plus generative gsm8k; ARC is registered by the
    ``atlas_arc`` package to preserve its historical name.
    """
    return [AtlasExternalEval(b) for b in cat_benchmarks() if b.name != "arc_challenge"]
