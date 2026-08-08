"""MCQ scoring via log-likelihood ranking (single method for all models).

For each question we score every option continuation given the shared context
and pick the argmax. Results stream to a per-(benchmark, model) CSV with the
required ``correct``/``wrong`` column.
"""

from __future__ import annotations

from common import Question, letter, mcq_output_path
from config import WriterConfig
from engine import Engine
from models_registry import ModelSpec
from results_writer import CsvResultWriter, existing_ids

MCQ_FIELDS = ["question_id", "model", "benchmark", "predicted", "gold", "result", "scoring_method"]


def score_mcq(
    engine: Engine,
    spec: ModelSpec,
    benchmark: str,
    questions: list[Question],
    writer_cfg: WriterConfig,
) -> int:
    """Score all questions for one (benchmark, model); returns rows written."""
    out_path = mcq_output_path(benchmark, spec.id)
    done = existing_ids(out_path)
    todo = [q for q in questions if q.qid not in done]
    if not todo:
        return 0

    # Flatten every (context, continuation) pair for one batched engine call.
    contexts: list[str] = []
    continuations: list[str] = []
    spans: list[tuple[int, int]] = []  # (start, count) into the flat arrays per question
    for q in todo:
        ctx = engine.format_prompt(q.prompt)
        start = len(contexts)
        for opt in q.options:
            contexts.append(ctx)
            continuations.append(opt)
        spans.append((start, len(q.options)))

    scores = engine.loglikelihood(contexts, continuations)

    written = 0
    with CsvResultWriter(
        out_path,
        MCQ_FIELDS,
        fsync_every_rows=writer_cfg.fsync_every_rows,
        fsync_every_seconds=writer_cfg.fsync_every_seconds,
    ) as w:
        for q, (start, count) in zip(todo, spans, strict=True):
            opt_scores = scores[start : start + count]
            pred = max(range(count), key=lambda i: opt_scores[i])
            correct = pred == q.gold_index
            w.write_row(
                {
                    "question_id": q.qid,
                    "model": spec.id,
                    "benchmark": benchmark,
                    "predicted": letter(pred),
                    "gold": letter(q.gold_index),
                    "result": "correct" if correct else "wrong",
                    "scoring_method": "loglikelihood",
                }
            )
            written += 1
    return written
