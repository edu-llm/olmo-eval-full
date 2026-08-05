"""I/O operations for evaluation runners."""

from olmo_eval.runners.io.builders import build_predictions, build_requests
from olmo_eval.runners.io.formatting import (
    build_s3_prefix,
    get_model_display_name,
    sanitize_model_name,
)
from olmo_eval.runners.io.storage import save_results, upload_to_s3
from olmo_eval.runners.io.writers import (
    PARTIAL_PREDICTIONS_SUFFIX,
    append_partial_predictions_jsonl,
    discard_partial_predictions,
    write_predictions_jsonl,
    write_requests_jsonl,
)

__all__ = [
    "PARTIAL_PREDICTIONS_SUFFIX",
    "append_partial_predictions_jsonl",
    "build_predictions",
    "build_requests",
    "build_s3_prefix",
    "discard_partial_predictions",
    "get_model_display_name",
    "sanitize_model_name",
    "save_results",
    "upload_to_s3",
    "write_predictions_jsonl",
    "write_requests_jsonl",
]
