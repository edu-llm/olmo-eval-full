"""Online ATLAS adaptive-testing external evaluation (Phase 2).

Runs an ATLAS Fisher-information CAT that queries the model *live*, item by
item: at each step it picks the highest-information unused item at the current
ability estimate, sends exactly that one ARC-Challenge log-likelihood request to
the provider, updates the estimate, and stops once the standard error crosses a
threshold. Only the selected items (~tens, not the full benchmark) are ever sent
to the model, which is what saves inference time versus the offline task.

Selection, ability estimation, and stopping all come from the shared
``olmo_eval.adaptive`` core, so this cannot diverge from the offline
``atlas_arc_challenge`` task. Item formatting and correctness reuse the real
``arc_challenge`` task for calibration parity.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from olmo_eval.adaptive import CatSession, load_bank
from olmo_eval.adaptive.cat import DEFAULT_MAX_ITEMS, DEFAULT_MIN_ITEMS, DEFAULT_SE_STOP
from olmo_eval.common.types import Instance
from olmo_eval.evals.external.base import ExternalEval
from olmo_eval.evals.external.result import ExternalEvalResult

if TYPE_CHECKING:
    from olmo_eval.evals.tasks.common import Task
    from olmo_eval.inference.base import InferenceProvider

logger = logging.getLogger(__name__)

DEFAULT_TASK = "arc_challenge"


class AtlasArcExternalEval(ExternalEval):
    """Online CAT over ARC-Challenge using a calibrated ATLAS 3PL bank."""

    @property
    def name(self) -> str:
        return "atlas_arc"

    @property
    def description(self) -> str:
        return "Online ATLAS 3PL adaptive test over ARC-Challenge (item-by-item, early stop)."

    @property
    def timeout_seconds(self) -> float:
        return 3600.0

    @property
    def arguments(self) -> dict[str, tuple[str, Any | None]]:
        return {
            "se_stop": ("Stop once the ability standard error drops to this.", DEFAULT_SE_STOP),
            "min_items": ("Minimum items to administer before stopping.", DEFAULT_MIN_ITEMS),
            "max_items": ("Maximum items to administer.", DEFAULT_MAX_ITEMS),
            "bank_path": ("Directory with the ATLAS bank CSVs (defaults to vendored ARC).", None),
            "task": ("olmo-eval task providing ARC items/formatting.", DEFAULT_TASK),
            "seed": ("Seed recorded for provenance (selection is deterministic).", 0),
        }

    def _load_items(self, args: dict[str, Any]) -> tuple[Task, dict[str, Instance]]:
        """Load the ARC task and index its instances by native ``question_id``.

        Split out so tests can inject a tiny synthetic item set without touching
        the network.
        """
        from olmo_eval.evals.tasks.common import get_task

        task = get_task(str(args.get("task", DEFAULT_TASK)))
        items: dict[str, Instance] = {}
        for instance in task.instances:
            qid = instance.metadata.get("id")
            if qid is not None:
                items[qid] = instance
        return task, items

    async def _score_item(self, task: Task, provider: InferenceProvider, instance: Instance) -> int:
        """Query the provider for one item and return 1 (correct) / 0 (wrong).

        Uses the task's own ``format_request`` and log-likelihood argmax so the
        online score matches how ``olmo-eval`` would score the item in a full run.
        """
        request = task.format_request(instance)
        outputs = await provider.alogprobs([request])
        continuations = outputs[0]
        sums: list[float] = []
        for output in continuations:
            token_lps = [t["logprob"] for t in (output.logprobs or []) if "logprob" in t]
            sums.append(sum(token_lps) if token_lps else float("-inf"))
        gold_idx = instance.metadata.get("gold_idx", 0)
        return 1 if sums.index(max(sums)) == gold_idx else 0

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
            bank = load_bank(bank_path)
        except FileNotFoundError as exc:
            return self._error_result(f"ATLAS bank unavailable: {exc}", start)

        try:
            task, items = self._load_items(args)
        except Exception as exc:  # noqa: BLE001 - surface any loading failure as a result
            return self._error_result(f"failed to load ARC items: {exc}", start)

        bank = bank.restrict(set(items))
        if len(bank) == 0:
            return self._error_result(
                "no overlap between the ATLAS bank and the loaded ARC items", start
            )

        session = CatSession(bank, se_stop=se_stop, min_items=min_items, max_items=max_items)
        while (idx := session.next_item()) is not None:
            qid = bank.question_ids[idx]
            score = await self._score_item(task, provider, items[qid])
            session.record(idx, score)

        result = session.result()
        logger.info(
            "[atlas_arc] online CAT queried %d/%d items: theta=%.4f se=%.4f pirt=%.4f",
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
