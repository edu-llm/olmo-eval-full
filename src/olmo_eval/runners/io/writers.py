"""Functions for writing predictions and requests to JSONL files."""

from __future__ import annotations

import json
import os

from olmo_eval.common.logging import get_logger
from olmo_eval.runners.common.types import PREDICTIONS_SUFFIX, REQUESTS_SUFFIX
from olmo_eval.runners.processing.utils import sanitize_spec_for_filename

logger = get_logger("runners.writers")

# What a crash leaves behind. A task's rows are appended here as they are scored and
# the file is removed once the complete PREDICTIONS_SUFFIX file exists, so its presence
# means "this task was interrupted" and its absence means nothing was lost.
PARTIAL_PREDICTIONS_SUFFIX = "-predictions.partial.jsonl"


def _predictions_target(
    output_dir: str,
    spec: str,
    model_name: str,
    task_hash: str | None,
    suffix: str,
) -> tuple[str, str]:
    """Directory and full path for a predictions file.

    Shared so the partial file lands beside the complete one under the same name;
    two copies of this arithmetic would let them drift apart and strand a partial
    file that nothing goes on to delete.
    """
    pred_dir = os.path.join(output_dir, "predictions", sanitize_spec_for_filename(model_name))
    base_name = sanitize_spec_for_filename(spec)
    if task_hash:
        base_name = f"{base_name}_{task_hash[-6:]}"
    return pred_dir, os.path.join(pred_dir, base_name + suffix)


def write_predictions_jsonl(
    output_dir: str,
    spec: str,
    predictions: list[dict],
    model_name: str,
    task_hash: str | None = None,
) -> None:
    """Write per-instance predictions to JSONL.

    Args:
        output_dir: Base output directory
        spec: Task specification string (used for filename)
        predictions: List of prediction dicts to write
        model_name: Model name or alias (used for subdirectory)
        task_hash: Optional task config hash (last 6 chars added to filename)
    """
    pred_dir, filepath = _predictions_target(
        output_dir, spec, model_name, task_hash, PREDICTIONS_SUFFIX
    )
    os.makedirs(pred_dir, exist_ok=True)

    with open(filepath, "w") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")

    logger.info(f"Saved {len(predictions)} predictions: {spec}")


def append_partial_predictions_jsonl(
    output_dir: str,
    spec: str,
    predictions: list[dict],
    model_name: str,
    task_hash: str | None = None,
) -> None:
    """Append scored rows to the partial file, for a run that may not finish.

    Opened per call rather than held open, because the caller batches rows and a
    handle kept across an interrupted run buys nothing: the rows already written are
    what survives either way.
    """
    if not predictions:
        return

    pred_dir, filepath = _predictions_target(
        output_dir, spec, model_name, task_hash, PARTIAL_PREDICTIONS_SUFFIX
    )
    os.makedirs(pred_dir, exist_ok=True)

    with open(filepath, "a") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")
        f.flush()


def discard_partial_predictions(
    output_dir: str,
    spec: str,
    model_name: str,
    task_hash: str | None = None,
) -> None:
    """Drop the partial file now that the complete one supersedes it."""
    _, filepath = _predictions_target(
        output_dir, spec, model_name, task_hash, PARTIAL_PREDICTIONS_SUFFIX
    )
    try:
        os.remove(filepath)
    except FileNotFoundError:
        pass
    except OSError as error:  # noqa: BLE001 - a stale partial is untidy, not fatal
        logger.warning(f"Could not remove partial predictions for {spec}: {error}")


def write_requests_jsonl(
    output_dir: str,
    spec: str,
    requests: list[dict],
    model_name: str,
    task_hash: str | None = None,
) -> None:
    """Write per-instance requests to JSONL (oe-eval compatible format).

    This file shows exactly what the model saw during evaluation, useful for
    debugging and comparison with oe-eval outputs.

    Args:
        output_dir: Base output directory
        spec: Task specification string (used for filename)
        requests: List of request dicts to write
        model_name: Model name or alias (used for subdirectory)
        task_hash: Optional task config hash (last 6 chars added to filename)
    """
    req_dir = os.path.join(output_dir, "requests", sanitize_spec_for_filename(model_name))
    os.makedirs(req_dir, exist_ok=True)

    # Build filename with optional hash suffix
    base_name = sanitize_spec_for_filename(spec)
    if task_hash:
        base_name = f"{base_name}_{task_hash[-6:]}"
    filename = base_name + REQUESTS_SUFFIX
    filepath = os.path.join(req_dir, filename)

    with open(filepath, "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")

    logger.info(f"Saved {len(requests)} requests: {spec}")
