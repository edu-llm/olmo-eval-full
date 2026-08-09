"""Strict file-contract tests for externally generated tutor responses."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
from typing import Any, cast

import pytest

import olmo_eval.edullm.precomputed as precomputed_module
from olmo_eval.edullm.precomputed import (
    PRECOMPUTED_RESPONSE_BATCH_SCHEMA_VERSION,
    PrecomputedTutorResponseBatch,
    PrecomputedTutorResponses,
    load_precomputed_tutor_response_batch,
    load_precomputed_tutor_responses,
)


def _publication_journal(path: Path, destination: str) -> None:
    transaction_id = "a" * 32
    entries = []
    for name in (destination, "report.json"):
        entries.append(
            {
                "destination": name,
                "staged": f".{name}.{transaction_id}.stage",
                "backup": f".{name}.{transaction_id}.backup",
                "original_existed": False,
                "original_sha256": None,
                "new_sha256": "b" * 64,
            }
        )
    path.write_text(
        json.dumps(
            {
                "schema_version": precomputed_module.RESPONSE_BATCH_PUBLICATION_SCHEMA_VERSION,
                "transaction_id": transaction_id,
                "state": "prepared",
                "entries": entries,
            }
        ),
        encoding="utf-8",
    )


def _load(path: Path, payload: bytes) -> PrecomputedTutorResponses:
    path.write_bytes(payload)
    return load_precomputed_tutor_responses(
        path,
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        expected_scenario_ids=("s1",),
    )


def _load_batch(
    path: Path,
    payload: bytes,
    *,
    expected_scenario_ids: tuple[str, ...] = ("s1",),
) -> PrecomputedTutorResponseBatch:
    path.write_bytes(payload)
    return load_precomputed_tutor_response_batch(
        path,
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        expected_scenario_ids=expected_scenario_ids,
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


def test_batch_loader_accepts_multiple_models_in_deterministic_order(
    tmp_path: Path,
) -> None:
    payload = (
        '{"model_id":"z-model","model_family":"zeta","model_revision":"r2",'
        '"scenario_id":"s2","response":"   "}\n'
        '{"model_id":"a-model","model_family":"alpha","model_revision":"r1",'
        '"scenario_id":"s2","response":"Café",'
        '"metadata":{"finish_reason":"stop","tokens":[1,2]}}\n'
        '{"model_id":"z-model","model_family":"zeta","model_revision":"r2",'
        '"scenario_id":"s1","response":"z answer"}\n'
        '{"model_id":"a-model","model_family":"alpha","model_revision":"r1",'
        '"scenario_id":"s1","response":""}\n'
    ).encode()
    path = tmp_path / "batch.jsonl"

    loaded = _load_batch(path, payload, expected_scenario_ids=("s1", "s2"))

    assert PRECOMPUTED_RESPONSE_BATCH_SCHEMA_VERSION == (
        "edullm-precomputed-tutor-response-batch-v1"
    )
    assert loaded.path == path.resolve()
    assert loaded.sha256 == hashlib.sha256(payload).hexdigest()
    assert loaded.model_count == 2
    assert loaded.row_count == 4
    assert loaded.blank_count == 2
    assert tuple(model.model_id for model in loaded.models) == ("a-model", "z-model")
    assert tuple(loaded.models_by_id) == ("a-model", "z-model")

    alpha = loaded.models_by_id["a-model"]
    assert alpha.model_family == "alpha"
    assert alpha.model_revision == "r1"
    assert alpha.row_count == 2
    assert alpha.blank_count == 1
    assert tuple(alpha.responses) == ("s1", "s2")
    assert alpha.responses["s2"].response == "Café"
    assert alpha.responses["s2"].metadata == {
        "finish_reason": "stop",
        "tokens": (1, 2),
    }
    normalized_row = {
        "model_id": "a-model",
        "model_family": "alpha",
        "model_revision": "r1",
        "scenario_id": "s2",
        "response": "Café",
        "metadata": {"finish_reason": "stop", "tokens": [1, 2]},
    }
    canonical_row = json.dumps(
        normalized_row,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    assert alpha.responses["s2"].source_row_sha256 == hashlib.sha256(canonical_row).hexdigest()
    with pytest.raises(TypeError):
        cast(Any, alpha.responses)["s3"] = alpha.responses["s2"]
    with pytest.raises(TypeError):
        cast(Any, loaded.models_by_id)["other"] = alpha
    with pytest.raises(TypeError):
        cast(Any, alpha.responses["s2"].metadata)["new"] = "value"
    with pytest.raises(AttributeError):
        cast(Any, alpha.responses["s2"].metadata["tokens"]).append(3)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"\xff\n", "valid UTF-8"),
        (
            b'{"model_id":"m","model_family":"f","model_revision":"r",'
            b'"scenario_id":"s1","response":"ok"}\n\n',
            "must not be blank",
        ),
        (
            b'{"model_id":"m","model_id":"m","model_family":"f",'
            b'"model_revision":"r","scenario_id":"s1","response":"ok"}\n',
            "duplicate JSON key",
        ),
        (b'["m","s1","ok"]\n', "must be a JSON object"),
        (
            b'{"model_id":"m","model_family":"f","model_revision":"r","scenario_id":"s1"}\n',
            "fields differ",
        ),
        (
            b'{"model_id":"m","model_family":"f","model_revision":"r",'
            b'"scenario_id":"s1","response":"ok","extra":1}\n',
            "fields differ",
        ),
        (
            b'{"model_id":1,"model_family":"f","model_revision":"r",'
            b'"scenario_id":"s1","response":"ok"}\n',
            "model_id must be a non-empty, unpadded string",
        ),
        (
            b'{"model_id":"m","model_family":false,"model_revision":"r",'
            b'"scenario_id":"s1","response":"ok"}\n',
            "model_family must be a non-empty, unpadded string",
        ),
        (
            b'{"model_id":"m","model_family":"f","model_revision":" r ",'
            b'"scenario_id":"s1","response":"ok"}\n',
            "model_revision must be a non-empty, unpadded string",
        ),
        (
            b'{"model_id":"m","model_family":"f","model_revision":"r",'
            b'"scenario_id":null,"response":"ok"}\n',
            "scenario_id must be a non-empty, unpadded string",
        ),
        (
            b'{"model_id":"m","model_family":"f","model_revision":"r",'
            b'"scenario_id":"s1","response":[]}\n',
            "response must be a string",
        ),
        (
            b'{"model_id":"m","model_family":"f","model_revision":"r",'
            b'"scenario_id":"s1","response":"ok","metadata":[]}\n',
            "metadata must be an object",
        ),
        (
            b'{"model_id":"m","model_family":"f","model_revision":"r",'
            b'"scenario_id":"s1","response":"ok","metadata":{"x":NaN}}\n',
            "invalid JSON",
        ),
        (
            b'{"model_id":"m","model_family":"f","model_revision":"r",'
            b'"scenario_id":"s1","response":"ok","metadata":{"x":1e999}}\n',
            "non-finite number",
        ),
    ],
)
def test_batch_loader_rejects_malformed_rows(
    tmp_path: Path,
    payload: bytes,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _load_batch(tmp_path / "batch.jsonl", payload)


def test_batch_loader_requires_consistent_identity_fields(tmp_path: Path) -> None:
    payload = (
        b'{"model_id":"m","model_family":"family","model_revision":"r1",'
        b'"scenario_id":"s1","response":"one"}\n'
        b'{"model_id":"m","model_family":"family","model_revision":"r2",'
        b'"scenario_id":"s2","response":"two"}\n'
    )

    with pytest.raises(ValueError, match="inconsistent identity fields for model_id 'm'"):
        _load_batch(
            tmp_path / "batch.jsonl",
            payload,
            expected_scenario_ids=("s1", "s2"),
        )


def test_batch_loader_rejects_duplicate_model_scenario_pair(tmp_path: Path) -> None:
    payload = (
        b'{"model_id":"m","model_family":"family","model_revision":"r1",'
        b'"scenario_id":"s1","response":"one"}\n'
        b'{"model_id":"m","model_family":"family","model_revision":"r1",'
        b'"scenario_id":"s1","response":"two"}\n'
    )

    with pytest.raises(ValueError, match="duplicate precomputed tutor response batch row"):
        _load_batch(tmp_path / "batch.jsonl", payload)


@pytest.mark.parametrize(
    ("expected_scenario_ids", "message"),
    [
        (("s1", "s2"), r"model_id 'm2'; missing=\['s2'\], extra=\[\]"),
        (("s1",), r"model_id 'm1'; missing=\[\], extra=\['s2'\]"),
    ],
)
def test_batch_loader_requires_exact_fitted_bank_coverage_per_model(
    tmp_path: Path,
    expected_scenario_ids: tuple[str, ...],
    message: str,
) -> None:
    payload = (
        b'{"model_id":"m1","model_family":"family","model_revision":"r1",'
        b'"scenario_id":"s1","response":"one"}\n'
        b'{"model_id":"m1","model_family":"family","model_revision":"r1",'
        b'"scenario_id":"s2","response":"two"}\n'
        b'{"model_id":"m2","model_family":"family","model_revision":"r2",'
        b'"scenario_id":"s1","response":"other"}\n'
    )

    with pytest.raises(ValueError, match=message):
        _load_batch(
            tmp_path / "batch.jsonl",
            payload,
            expected_scenario_ids=expected_scenario_ids,
        )


def test_batch_loader_requires_file_sha_and_at_least_one_model(tmp_path: Path) -> None:
    payload = (
        b'{"model_id":"m","model_family":"family","model_revision":"r1",'
        b'"scenario_id":"s1","response":""}\n'
    )
    path = tmp_path / "batch.jsonl"
    path.write_bytes(payload)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_precomputed_tutor_response_batch(
            path,
            expected_sha256="0" * 64,
            expected_scenario_ids=("s1",),
        )

    empty = tmp_path / "empty.jsonl"
    empty.write_bytes(b"")
    with pytest.raises(ValueError, match="must contain at least one model"):
        load_precomputed_tutor_response_batch(
            empty,
            expected_sha256=hashlib.sha256(b"").hexdigest(),
            expected_scenario_ids=("s1",),
        )


def test_batch_loader_rejects_duplicate_expected_scenario_ids(tmp_path: Path) -> None:
    payload = (
        b'{"model_id":"m","model_family":"family","model_revision":"r1",'
        b'"scenario_id":"s1","response":""}\n'
    )

    with pytest.raises(ValueError, match="scenario IDs must be unique"):
        _load_batch(
            tmp_path / "batch.jsonl",
            payload,
            expected_scenario_ids=("s1", "s1"),
        )


@pytest.mark.parametrize("alias_kind", ["symlink", "hardlink"])
def test_batch_loader_refuses_same_directory_alias_of_incomplete_publication(
    tmp_path: Path,
    alias_kind: str,
) -> None:
    payload = (
        b'{"model_id":"m","model_family":"family","model_revision":"r1",'
        b'"scenario_id":"s1","response":"answer"}\n'
    )
    source = tmp_path / "batch.jsonl"
    source.write_bytes(payload)
    journal = tmp_path / precomputed_module.RESPONSE_BATCH_PUBLICATION_JOURNAL_NAME
    _publication_journal(journal, source.name)
    alias = tmp_path / "alias.jsonl"
    if alias_kind == "symlink":
        alias.symlink_to(source.name)
    else:
        os.link(source, alias)

    with pytest.raises(ValueError, match="incomplete paired publication"):
        load_precomputed_tutor_response_batch(
            alias,
            expected_sha256=hashlib.sha256(payload).hexdigest(),
            expected_scenario_ids=("s1",),
        )


def test_batch_loader_refuses_cross_directory_hardlink_alias(tmp_path: Path) -> None:
    payload = (
        b'{"model_id":"m","model_family":"family","model_revision":"r1",'
        b'"scenario_id":"s1","response":"answer"}\n'
    )
    published = tmp_path / "published"
    published.mkdir()
    source = published / "batch.jsonl"
    source.write_bytes(payload)
    _publication_journal(
        published / precomputed_module.RESPONSE_BATCH_PUBLICATION_JOURNAL_NAME,
        source.name,
    )
    aliases = tmp_path / "aliases"
    aliases.mkdir()
    alias = aliases / "batch-alias.jsonl"
    os.link(source, alias)

    with pytest.raises(ValueError, match="hard-link aliases"):
        load_precomputed_tutor_response_batch(
            alias,
            expected_sha256=hashlib.sha256(payload).hexdigest(),
            expected_scenario_ids=("s1",),
        )


def test_batch_loader_refuses_active_or_unsafe_publication_lock(tmp_path: Path) -> None:
    payload = (
        b'{"model_id":"m","model_family":"family","model_revision":"r1",'
        b'"scenario_id":"s1","response":"answer"}\n'
    )
    source = tmp_path / "batch.jsonl"
    source.write_bytes(payload)
    lock = tmp_path / precomputed_module.RESPONSE_BATCH_PUBLICATION_LOCK_NAME
    descriptor = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="publication is active"):
            load_precomputed_tutor_response_batch(
                source,
                expected_sha256=hashlib.sha256(payload).hexdigest(),
                expected_scenario_ids=("s1",),
            )
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)

    lock.unlink()
    os.mkfifo(lock)
    with pytest.raises(ValueError, match="publication lock is unsafe"):
        load_precomputed_tutor_response_batch(
            source,
            expected_sha256=hashlib.sha256(payload).hexdigest(),
            expected_scenario_ids=("s1",),
        )


def test_batch_loader_fails_closed_on_empty_publication_journal(tmp_path: Path) -> None:
    payload = (
        b'{"model_id":"m","model_family":"family","model_revision":"r1",'
        b'"scenario_id":"s1","response":"answer"}\n'
    )
    source = tmp_path / "batch.jsonl"
    source.write_bytes(payload)
    journal = tmp_path / precomputed_module.RESPONSE_BATCH_PUBLICATION_JOURNAL_NAME
    journal.write_text(
        json.dumps(
            {
                "schema_version": precomputed_module.RESPONSE_BATCH_PUBLICATION_SCHEMA_VERSION,
                "transaction_id": "a" * 32,
                "state": "prepared",
                "entries": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exactly two entries"):
        load_precomputed_tutor_response_batch(
            source,
            expected_sha256=hashlib.sha256(payload).hexdigest(),
            expected_scenario_ids=("s1",),
        )


def test_batch_loader_fails_closed_on_malformed_two_entry_journal(tmp_path: Path) -> None:
    payload = (
        b'{"model_id":"m","model_family":"family","model_revision":"r1",'
        b'"scenario_id":"s1","response":"answer"}\n'
    )
    source = tmp_path / "batch.jsonl"
    source.write_bytes(payload)
    journal = tmp_path / precomputed_module.RESPONSE_BATCH_PUBLICATION_JOURNAL_NAME
    _publication_journal(journal, source.name)
    value = json.loads(journal.read_text(encoding="utf-8"))
    value["entries"][0]["staged"] = ".unbound.stage"
    journal.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="temporary name is invalid"):
        load_precomputed_tutor_response_batch(
            source,
            expected_sha256=hashlib.sha256(payload).hexdigest(),
            expected_scenario_ids=("s1",),
        )


def test_batch_loader_does_not_create_lock_in_read_only_dataset(tmp_path: Path) -> None:
    payload = (
        b'{"model_id":"m","model_family":"family","model_revision":"r1",'
        b'"scenario_id":"s1","response":"answer"}\n'
    )
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    source = dataset / "batch.jsonl"
    source.write_bytes(payload)
    dataset.chmod(0o555)
    try:
        loaded = load_precomputed_tutor_response_batch(
            source,
            expected_sha256=hashlib.sha256(payload).hexdigest(),
            expected_scenario_ids=("s1",),
        )
    finally:
        dataset.chmod(0o755)

    assert loaded.sha256 == hashlib.sha256(payload).hexdigest()
    assert not (dataset / precomputed_module.RESPONSE_BATCH_PUBLICATION_LOCK_NAME).exists()
