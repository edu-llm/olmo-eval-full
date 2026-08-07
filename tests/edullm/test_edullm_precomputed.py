"""Strict file-contract tests for externally generated tutor responses."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from olmo_eval.edullm.precomputed import (
    PrecomputedTutorResponses,
    load_precomputed_tutor_responses,
)


def _load(path: Path, payload: bytes) -> PrecomputedTutorResponses:
    path.write_bytes(payload)
    return load_precomputed_tutor_responses(
        path,
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        expected_scenario_ids=("s1",),
    )


def test_loader_accepts_exact_utf8_jsonl_and_preserves_metadata(tmp_path: Path) -> None:
    payload = (
        b'{"scenario_id":"s1","response":"Tutor answer","metadata":{"finish_reason":"stop"}}\n'
    )

    loaded = _load(tmp_path / "responses.jsonl", payload)

    assert loaded.row_count == 1
    assert loaded.blank_count == 0
    assert loaded.responses["s1"].response == "Tutor answer"
    assert loaded.responses["s1"].metadata == {"finish_reason": "stop"}


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"\xff\n", "valid UTF-8"),
        (b'{"scenario_id":"s1","response":"ok"}\n\n', "must not be blank"),
        (
            b'{"scenario_id":"s1","scenario_id":"s1","response":"ok"}\n',
            "duplicate JSON key",
        ),
        (b'{"scenario_id":"s1","response":"ok","metadata":[]}\n', "must be an object"),
        (b'{"scenario_id":"s1"}\n', "fields differ"),
        (b'["s1","ok"]\n', "must be a JSON object"),
        (b'{"scenario_id":"s1","response":"ok","metadata":{"x":NaN}}\n', "invalid JSON"),
    ],
)
def test_loader_rejects_noncanonical_or_malformed_input(
    tmp_path: Path,
    payload: bytes,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _load(tmp_path / "responses.jsonl", payload)


def test_loader_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        load_precomputed_tutor_responses(
            tmp_path / "missing.jsonl",
            expected_sha256="0" * 64,
            expected_scenario_ids=("s1",),
        )
