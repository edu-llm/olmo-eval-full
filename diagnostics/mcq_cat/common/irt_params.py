"""Load IRT item parameters.

Supports the common unidimensional shapes (1PL/2PL/3PL) and multidimensional
(MIRT) discrimination vectors. Parameters are read from a JSON source (local path
or ``s3://`` URI) into an :class:`IRTBank` keyed by item id.

Accepted per-item keys (aliases in parentheses):

- ``item_id`` (``id``): item identifier.
- ``difficulty`` (``b``): scalar difficulty.
- ``discrimination`` (``a``): scalar for unidimensional, list for MIRT. Defaults
  to ``1.0`` (a 1PL/Rasch item).
- ``guessing`` (``c``): lower asymptote, defaults to ``0.0`` (1PL/2PL).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ..base import Discrimination, IRTBank, IRTItemParams
from . import s3_io

log = logging.getLogger("mcq_cat.irt_params")


def _first(record: dict, *keys: str, default: Any = None) -> Any:
    """Return the first present value among ``keys`` (supports aliases)."""
    for key in keys:
        if key in record:
            return record[key]
    return default


def _parse_discrimination(value: Any) -> tuple[Discrimination, int]:
    """Normalize a discrimination value to ``(parsed, dimensions)``."""
    if isinstance(value, (list, tuple)):
        vector = tuple(float(component) for component in value)
        return vector, len(vector)
    return float(value), 1


def _params_from_record(record: dict, index: int) -> tuple[IRTItemParams, int]:
    """Build :class:`IRTItemParams` from one record, returning its dimensionality."""
    item_id = str(_first(record, "item_id", "id", default=index))
    difficulty = float(_first(record, "difficulty", "b", default=0.0))
    discrimination, dimensions = _parse_discrimination(
        _first(record, "discrimination", "a", default=1.0)
    )
    guessing = float(_first(record, "guessing", "c", default=0.0))
    metadata = dict(record.get("metadata", {}))
    params = IRTItemParams(
        item_id=item_id,
        difficulty=difficulty,
        discrimination=discrimination,
        guessing=guessing,
        metadata=metadata,
    )
    return params, dimensions


def load_irt_params(source: str | Path) -> IRTBank:
    """Load IRT parameters from a JSON source (local path or ``s3://`` URI).

    The JSON payload may be a list of item records or a mapping of item id to
    record. All items must share the same discrimination dimensionality.
    """
    source_str = str(source)
    text = s3_io.read_text(source_str) if s3_io.is_s3_uri(source_str) else Path(source).read_text()
    payload = json.loads(text)

    if isinstance(payload, dict):
        records = [{"item_id": key, **value} for key, value in payload.items()]
    else:
        records = list(payload)

    params: dict[str, IRTItemParams] = {}
    dimensions = 1
    for index, record in enumerate(records):
        item_params, item_dimensions = _params_from_record(record, index)
        if index == 0:
            dimensions = item_dimensions
        elif item_dimensions != dimensions:
            raise ValueError(
                f"Inconsistent IRT dimensionality: item {item_params.item_id!r} has "
                f"{item_dimensions} dimensions, expected {dimensions}"
            )
        params[item_params.item_id] = item_params

    if not params:
        raise ValueError(f"No IRT parameters found in source: {source_str}")

    log.info("Loaded IRT params for %d items (%dD) from %s", len(params), dimensions, source_str)
    return IRTBank(params=params, dimensions=dimensions)
