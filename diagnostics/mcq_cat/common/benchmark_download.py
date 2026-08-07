"""Download and prepare MCQ benchmark items.

The stable path loads items from a JSONL source (local path or ``s3://`` URI)
using a small, documented schema, returning a :class:`BenchmarkBank`. Building a
bank directly from an ``olmo_eval`` task definition is left as a marked extension
point: the eval suite already defines MCQ tasks and log-likelihood scoring under
``src/olmo_eval/evals/tasks``, and a style branch can wire that in when the task
selection is settled.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ..base import BenchmarkBank, BenchmarkItem
from . import s3_io

log = logging.getLogger("mcq_cat.benchmark_download")


def _item_from_record(record: dict, index: int) -> BenchmarkItem:
    """Build a :class:`BenchmarkItem` from one JSONL record.

    Recognized keys: ``id`` (optional; defaults to the row index), ``question``,
    ``choices`` (list of strings), ``gold_index`` (int) or ``answer`` (index or
    matching choice text), and an optional ``metadata`` mapping.
    """
    item_id = str(record.get("id", index))
    question = record["question"]
    choices = tuple(record["choices"])

    if "gold_index" in record:
        gold_index = int(record["gold_index"])
    elif "answer" in record:
        answer = record["answer"]
        gold_index = int(answer) if isinstance(answer, int) else list(choices).index(answer)
    else:
        raise ValueError(f"Record {item_id} has neither 'gold_index' nor 'answer'")

    metadata = dict(record.get("metadata", {}))
    return BenchmarkItem(
        item_id=item_id,
        question=question,
        choices=choices,
        gold_index=gold_index,
        metadata=metadata,
    )


def load_items_from_jsonl(source: str | Path, *, name: str | None = None) -> BenchmarkBank:
    """Load MCQ items from a JSONL source (local path or ``s3://`` URI).

    Read through :func:`~diagnostics.mcq_cat.common.s3_io.read_text` on both branches, so
    a local bank is decoded as UTF-8 rather than in the platform's locale encoding. The
    two disagree wherever a stem is not ASCII, and the failure is a substituted character
    in the prompt rather than an error.
    """
    source_str = str(source)
    text = s3_io.read_text(source_str)

    items: list[BenchmarkItem] = []
    for index, line in enumerate(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        items.append(_item_from_record(json.loads(line), index))

    if not items:
        raise ValueError(f"No items found in benchmark source: {source_str}")

    bank_name = name or Path(source_str).stem
    log.info("Loaded %d items for benchmark %r from %s", len(items), bank_name, source_str)
    return BenchmarkBank(name=bank_name, items=tuple(items))


def download_benchmark(benchmark: str, *, dest: Path | None = None) -> BenchmarkBank:
    """Prepare a :class:`BenchmarkBank` for ``benchmark``.

    ``benchmark`` may be a local path or ``s3://`` URI to a JSONL item file. A
    bare benchmark name (resolved against the ``olmo_eval`` task registry) is an
    intentional extension point, mirrored on the inference side.
    """
    if s3_io.is_s3_uri(benchmark) or Path(benchmark).exists():
        return load_items_from_jsonl(benchmark)
    return from_olmo_eval_task(benchmark)


def from_olmo_eval_task(task_name: str) -> BenchmarkBank:
    """Build a bank from an ``olmo_eval`` MCQ task (extension point).

    The eval suite defines MCQ tasks (for example ARC, MMLU) and their
    log-likelihood formatting under ``src/olmo_eval/evals/tasks``. A style branch
    should instantiate the task's config, load its instances, and map each to a
    :class:`BenchmarkItem`. Left unimplemented here so the interface stays stable
    until the benchmark-selection decisions in the plan are made.
    """
    raise NotImplementedError(
        "Building a benchmark bank from an olmo_eval task is an integration point. "
        "Provide the task name and mapping, then map olmo_eval Instances "
        "(src/olmo_eval/evals/tasks) to BenchmarkItem. For now pass a JSONL path or s3:// URI."
    )
