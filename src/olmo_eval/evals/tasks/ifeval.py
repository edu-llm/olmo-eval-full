"""IFEval: the Open LLM Leaderboard v2 instruction-following slice.

Dataset: ``wis-k/instruction-following-eval`` (541 prompts) -- the same content
lm-evaluation-harness serves as ``leaderboard_ifeval``. Each prompt carries a
list of verifiable instruction IDs and per-instruction kwargs; verifiers come
from the vendored registry used by :class:`IFEvalScorer`.

This mirrors the leaderboard config (chat format, zero-temperature generation,
prompt-level strict accuracy as primary). ``metadata["id"]`` is the enumeration
index in the split's native order, matching lm-eval's ``doc_id`` so a calibrated
ATLAS bank keyed on that id joins directly.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from olmo_eval.common.metrics import (
    IFEvalInstLooseAccuracy,
    IFEvalInstStrictAccuracy,
    IFEvalPromptLooseAccuracy,
    IFEvalPromptStrictAccuracy,
)
from olmo_eval.common.types import (
    Instance,
    LMOutput,
    LMRequest,
    RequestType,
    SamplingParams,
    Split,
)
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import Task, register

_PRIMARY_METRIC = IFEvalPromptStrictAccuracy()


@register("ifeval")
class IFEval(Task):
    data_source = DataSource(path="wis-k/instruction-following-eval", split="train")
    split = Split.TRAIN
    metrics = (
        IFEvalPromptStrictAccuracy(),
        IFEvalPromptLooseAccuracy(),
        IFEvalInstStrictAccuracy(),
        IFEvalInstLooseAccuracy(),
    )
    primary_metric = _PRIMARY_METRIC
    sampling_params = SamplingParams(
        max_tokens=1280,
        temperature=0.0,
        do_sample=False,
    )

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached()

    @property
    def request_type(self) -> RequestType:
        return RequestType.CHAT

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        prompt = doc["prompt"]
        instruction_id_list = list(doc.get("instruction_id_list") or [])
        raw_kwargs = doc.get("kwargs") or []
        kwargs_list = [{k: v for k, v in (kw or {}).items() if v is not None} for kw in raw_kwargs]
        return Instance(
            question=prompt,
            gold_answer=None,
            metadata={
                # Positional doc_id in native split order (matches lm-eval /
                # the ATLAS bank id-bridge). The dataset's own ``key`` is sparse
                # and is kept separately for reference only.
                "id": index,
                "key": doc.get("key", index),
                "prompt": prompt,
                "instruction_id_list": instruction_id_list,
                "kwargs": kwargs_list,
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        return LMRequest(
            request_type=RequestType.CHAT,
            messages=({"role": "user", "content": instance.question},),
        )

    def extract_answer(self, output: LMOutput) -> str:
        return output.text
