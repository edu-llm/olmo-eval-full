"""Open-ended generation: produce and durably store raw model responses.

Responses are persisted to ``<model>.responses.jsonl`` *before* judging so a
judge failure never loses generations.
"""

from __future__ import annotations

from common import Question, open_responses_path
from config import WriterConfig
from engine import Engine, GenParams
from models_registry import ModelSpec
from results_writer import JsonlResultWriter, existing_ids


def generate_open(
    engine: Engine,
    spec: ModelSpec,
    benchmark: str,
    questions: list[Question],
    gen: GenParams,
    writer_cfg: WriterConfig,
) -> int:
    """Generate + persist responses for one (benchmark, model); returns rows written."""
    out_path = open_responses_path(benchmark, spec.id)
    done = existing_ids(out_path)
    todo = [q for q in questions if q.qid not in done]
    if not todo:
        return 0

    prompts = [q.prompt for q in todo]
    responses = engine.generate(prompts, gen)

    written = 0
    with JsonlResultWriter(
        out_path,
        fsync_every_rows=writer_cfg.fsync_every_rows,
        fsync_every_seconds=writer_cfg.fsync_every_seconds,
    ) as w:
        for q, resp in zip(todo, responses, strict=True):
            w.write_row(
                {
                    "question_id": q.qid,
                    "model": spec.id,
                    "benchmark": benchmark,
                    "prompt": q.prompt,
                    "response": resp,
                    "reference": q.reference or "",
                    "meta": q.meta,
                }
            )
            written += 1
    return written
