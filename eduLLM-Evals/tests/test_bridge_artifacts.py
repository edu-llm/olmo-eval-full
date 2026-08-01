"""Regression checks for the curated Bridge v8 bank and its audit ledgers."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / "data" / "Bridge"


def _json(name: str):
    return json.loads((BRIDGE / name).read_text(encoding="utf-8"))


def _jsonl(name: str) -> list[dict]:
    return [
        json.loads(line)
        for line in (BRIDGE / name).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_bridge_v8_partition_and_schema_links() -> None:
    scenarios = _json("scenarios.json")
    rubrics = _json("rubrics.json")
    dropped = _json("dropped.json")

    assert scenarios == _jsonl("scenarios.jsonl")
    assert rubrics == _jsonl("rubrics.jsonl")
    assert dropped == _jsonl("dropped.jsonl")

    assert len(scenarios) == 250
    assert len(dropped) == 450
    assert len(scenarios) + len(dropped) == 700
    assert len(rubrics) == 4_795
    assert len({row["source_id"] for row in scenarios}) == 162
    assert {row["version"] for row in scenarios} == {"8.0"}
    assert {row["version"] for row in rubrics} == {"8.0"}

    scenario_ids = [row["scenario_id"] for row in scenarios]
    rubric_ids = [row["criterion_id"] for row in rubrics]
    assert len(scenario_ids) == len(set(scenario_ids))
    assert len(rubric_ids) == len(set(rubric_ids))
    assert set(scenario_ids).isdisjoint(
        row["scenario_id"] for row in dropped if row["scenario_id"] is not None
    )
    assert {cid for row in scenarios for cid in row["criterion_ids"]} == set(rubric_ids)
    assert {row["scenario_id"] for row in rubrics} == set(scenario_ids)

    assert Counter(row["reason"] for row in dropped) == {
        "missing_problem_statement": 245,
        "error_not_diagnosable_from_text": 86,
        "failed_adversarial_verification": 15,
        "no_clear_mistake": 56,
        "no_error_present": 18,
        "not_gradeable": 10,
        "not_mathematics": 18,
        "empty_student_turn": 2,
    }


def test_bridge_suspect_references_are_quarantined() -> None:
    scenarios = _json("scenarios.json")
    quarantine = _json("reference_suspect.json")
    flagged = {row["scenario_id"]: row for row in quarantine["flagged"]}
    kept_flagged = [row for row in scenarios if row["reference_suspect"]]

    assert quarantine["count"] == len(flagged) == 59
    assert Counter(row["severity"] for row in quarantine["flagged"]) == {
        "fatal": 37,
        "serious": 22,
    }
    assert len(kept_flagged) == 54
    assert Counter(flagged[row["scenario_id"]]["severity"] for row in kept_flagged) == {
        "fatal": 33,
        "serious": 21,
    }
    assert sum(len(row["criterion_ids"]) for row in kept_flagged) == 1_034

    for row in quarantine["flagged"]:
        text = row["withheld_reference_solution"]
        assert text
        assert row["source_id"]
        assert row["native_split"] in {"train", "validation", "test"}
        assert hashlib.sha256(text.encode("utf-8")).hexdigest() == row[
            "withheld_reference_sha256"
        ]

    for scenario in scenarios:
        assert isinstance(scenario["reference_suspect"], bool)
        if scenario["reference_suspect"]:
            assert scenario["scenario_id"] in flagged
            assert scenario["reference_solution"] == ""
        else:
            assert scenario["reference_solution"]


def test_bridge_blind_decision_reproduces_restorations() -> None:
    restored_doc = _json("restored.json")
    decision = _json("blind_readjudication.json")
    candidates = decision["candidates"]

    assert restored_doc["count"] == len(restored_doc["restored"]) == 78
    assert decision["candidate_count"] == len(candidates) == 133
    assert decision["upheld_count"] == 84
    assert decision["overturned_count"] == 49
    assert decision["restored_count"] == 78
    assert decision["already_kept_control_count"] == 6

    assert all(
        (row["yes_votes"] >= 2) == (row["decision"] == "upheld")
        for row in candidates
    )
    restored_from_votes = {
        row["scenario_id"]
        for row in candidates
        if row["origin"] == "audited_exclusion" and row["decision"] == "upheld"
    }
    assert restored_from_votes == {
        row["scenario_id"] for row in restored_doc["restored"]
    }
    assert Counter(
        row["yes_votes"] for row in candidates
        if row["scenario_id"] in restored_from_votes
    ) == {3: 74, 2: 4}
