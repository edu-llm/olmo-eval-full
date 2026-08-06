"""Load IRT criterion parameters for an FRQ bank.

Supports the common unidimensional shapes (1PL/2PL/3PL) and multidimensional
(MIRT) discrimination vectors, matching the graduated bank schema described in
``Plan/frq_cat_diagnostics/README.md``. Parameters are read from a JSON or JSONL
source (local path or ``s3://`` URI) into an :class:`IRTBank` keyed by criterion
id. This is the frozen schema each style's graduated bank conforms to at port time.

Accepted per-item keys (aliases in parentheses):

- ``criterion_id`` (``item_id``, ``id``): criterion identifier.
- ``difficulty`` (``b``): scalar difficulty.
- ``discrimination`` (``a``): scalar for unidimensional; a list, or a mapping
  keyed by modeled skill name (``{"ability": 1.2}``), for MIRT. Defaults to
  ``1.0`` (a 1PL/Rasch item).
- ``guessing`` (``c``): lower asymptote, defaults to ``0.0`` (1PL/2PL).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ..base import Discrimination, IRTBank, IRTItemParams
from . import s3_io

log = logging.getLogger("frq_cat.irt_params")


def _first(record: dict, *keys: str, default: Any = None) -> Any:
    """Return the first present value among ``keys`` (supports aliases)."""
    for key in keys:
        if key in record:
            return record[key]
    return default


def _parse_discrimination(value: Any) -> tuple[Discrimination, int]:
    """Normalize a discrimination value to ``(parsed, dimensions)``.

    Accepts a scalar (unidimensional), a list/tuple (MIRT), or a mapping keyed by
    modeled skill name (MIRT, the graduated q-matrix form). Mapping order is the
    sorted skill name so the vector is deterministic across records.
    """
    if isinstance(value, dict):
        vector = tuple(float(value[skill]) for skill in sorted(value))
        return vector, len(vector)
    if isinstance(value, (list, tuple)):
        vector = tuple(float(component) for component in value)
        return vector, len(vector)
    return float(value), 1


def _params_from_record(record: dict, index: int) -> tuple[IRTItemParams, int]:
    """Build :class:`IRTItemParams` from one record, returning its dimensionality."""
    item_id = str(_first(record, "criterion_id", "item_id", "id", default=index))
    difficulty = float(_first(record, "difficulty", "b", default=0.0))
    discrimination, dimensions = _parse_discrimination(
        _first(record, "discrimination", "a", default=1.0)
    )
    guessing = float(_first(record, "guessing", "c", default=0.0))
    metadata = dict(record.get("metadata", {}))
    if "q_modeled" in record:
        metadata.setdefault("q_modeled", record["q_modeled"])
    params = IRTItemParams(
        item_id=item_id,
        difficulty=difficulty,
        discrimination=discrimination,
        guessing=guessing,
        metadata=metadata,
    )
    return params, dimensions


def _load_records(source_str: str) -> list[dict]:
    """Read a JSON list/mapping or a JSONL file into a list of records."""
    text = (
        s3_io.read_text(source_str) if s3_io.is_s3_uri(source_str) else Path(source_str).read_text()
    )
    stripped = text.lstrip()
    if stripped.startswith("["):
        return list(json.loads(text))
    if stripped.startswith("{") and "\n" not in stripped.rstrip():
        payload = json.loads(text)
        return [{"criterion_id": key, **value} for key, value in payload.items()]
    # Otherwise treat as JSONL (one record per non-blank line).
    records: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def load_irt_params(source: str | Path) -> IRTBank:
    """Load IRT parameters from a JSON/JSONL source (local path or ``s3://`` URI).

    The payload may be a list of records, a mapping of criterion id to record, or
    a JSONL file. All items must share the same discrimination dimensionality.
    """
    source_str = str(source)
    records = _load_records(source_str)

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

    log.info("Loaded IRT params for %d criteria (%dD) from %s", len(params), dimensions, source_str)
    return IRTBank(params=params, dimensions=dimensions)
