"""Load an FRQ scenario + criterion bank into the frozen schema.

Reads the graduated payload a style ships (``bank/scenarios.jsonl`` and the fitted
``bank/params.jsonl``) into an :class:`FrqBank`. Scenarios carry the tutor prompt;
criteria are the IRT items, each linked to its scenario and its q-matrix row. A
``bank_sha256`` helper stamps provenance, matching the FRQ calibration packages.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from ..base import Criterion, FrqBank, Scenario
from . import s3_io

log = logging.getLogger("frq_cat.bank_loader")


def bank_sha256(source: str | Path) -> str:
    """Return the SHA-256 of a bank file (local path or ``s3://`` URI)."""
    source_str = str(source)
    if s3_io.is_s3_uri(source_str):
        data = s3_io.read_text(source_str).encode("utf-8")
    else:
        data = Path(source_str).read_bytes()
    return hashlib.sha256(data).hexdigest()


def _read_records(source: str | Path) -> list[dict]:
    """Read a JSON list or a JSONL file into a list of records."""
    source_str = str(source)
    text = (
        s3_io.read_text(source_str) if s3_io.is_s3_uri(source_str) else Path(source_str).read_text()
    )
    stripped = text.lstrip()
    if stripped.startswith("["):
        return list(json.loads(text))
    records: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def load_scenarios(source: str | Path) -> dict[str, Scenario]:
    """Load scenarios keyed by ``scenario_id`` from a JSON/JSONL source."""
    scenarios: dict[str, Scenario] = {}
    for record in _read_records(source):
        scenario_id = str(record.get("scenario_id") or record.get("id"))
        messages = tuple(record.get("messages", ()) or ())
        scenarios[scenario_id] = Scenario(
            scenario_id=scenario_id,
            prompt=record.get("prompt", ""),
            messages=messages,
            metadata=record.get("metadata", {}),
        )
    if not scenarios:
        raise ValueError(f"No scenarios found in source: {source}")
    return scenarios


def load_criteria(source: str | Path) -> tuple[Criterion, ...]:
    """Load criteria (the IRT items) from a JSON/JSONL source."""
    criteria: list[Criterion] = []
    for record in _read_records(source):
        criterion_id = str(record.get("criterion_id") or record.get("id"))
        criteria.append(
            Criterion(
                criterion_id=criterion_id,
                scenario_id=str(record.get("scenario_id", "")),
                text=record.get("criterion") or record.get("text", ""),
                primary_skill=record.get("primary_skill", "ability"),
                q_modeled=dict(record.get("q_modeled", {"ability": 1})),
                metadata=record.get("metadata", {}),
            )
        )
    if not criteria:
        raise ValueError(f"No criteria found in source: {source}")
    return tuple(criteria)


def load_bank(name: str, scenarios_source: str | Path, criteria_source: str | Path) -> FrqBank:
    """Load a complete :class:`FrqBank` from its scenario and criterion sources."""
    scenarios = load_scenarios(scenarios_source)
    criteria = load_criteria(criteria_source)
    log.info("Loaded FRQ bank %r: %d scenarios, %d criteria", name, len(scenarios), len(criteria))
    return FrqBank(name=name, scenarios=scenarios, criteria=criteria)
