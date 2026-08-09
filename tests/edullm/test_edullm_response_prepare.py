"""Tests for strict, deterministic response-batch preparation."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import olmo_eval.edullm.precomputed as precomputed_module
import olmo_eval.edullm.response_prepare as prepare_module
from olmo_eval.edullm.precomputed import load_precomputed_tutor_response_batch
from olmo_eval.edullm.response_prepare import (
    ResponsePreparationError,
    build_precomputed_tutor_response_batch,
    check_precomputed_tutor_response_batch_write,
    load_fitted_scenario_roster,
    load_response_batch_source_manifest,
    write_precomputed_tutor_response_batch,
)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> bytes:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return payload


def _manifest(path: Path, rows: list[dict[str, Any]]) -> Path:
    _write_jsonl(path, rows)
    return path


def _roster(path: Path, scenario_ids: tuple[str, ...] = ("s1", "s2")) -> Path:
    _write_jsonl(path, [{"scenario_id": scenario_id} for scenario_id in scenario_ids])
    return path


def _single_source(
    path: Path,
    *,
    model_id: str,
    family: str,
    revision: str,
    sha256: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "format": "single-jsonl-v1",
        "path": path.name,
        "model_id": model_id,
        "model_family": family,
        "model_revision": revision,
    }
    if sha256 is not None:
        row["sha256"] = sha256
    return row


def _legacy_row(
    *,
    model_id: str,
    revision: str,
    scenario_id: str,
    output: str,
    finish_reason: str = "stop",
    rendered_prompt: str = "prompt",
    issue: int | None = None,
    issue_description: str | None = None,
) -> dict[str, Any]:
    issue_value = int(finish_reason == "error") if issue is None else issue
    return {
        "Benchmark": "TutorBench",
        "Scenario": scenario_id,
        "Model": model_id,
        "Model Revision": revision,
        "Chat Template Applied": 1,
        "Rendered Prompt": rendered_prompt,
        "Generation Params": {"temperature": 0.0},
        "Max Model Len": 4096,
        "Prompt Tokens": 10,
        "Output Tokens": len(output),
        "Finish Reason": finish_reason,
        "Truncated": 0,
        "Latency (s)": 1.5,
        "Output": output,
        "Issue": issue_value,
        "Issue Description": (
            issue_description
            if issue_description is not None
            else "generation failed"
            if issue_value
            else "N/A"
        ),
    }


def test_build_merges_formats_deterministically_and_writer_self_validates(
    tmp_path: Path,
) -> None:
    scenarios = _roster(tmp_path / "scenarios.jsonl", ("s2", "s1"))
    single = tmp_path / "a.jsonl"
    _write_jsonl(
        single,
        [
            {"scenario_id": "s1", "response": "   "},
            {"scenario_id": "s2", "response": "Café", "metadata": {"tokens": [1]}},
        ],
    )
    legacy = tmp_path / "z.jsonl"
    _write_jsonl(
        legacy,
        [
            _legacy_row(model_id="z/model", revision="z-rev", scenario_id="s1", output="Z1"),
            _legacy_row(model_id="z/model", revision="z-rev", scenario_id="s2", output="Z2"),
        ],
    )
    source_manifest = _manifest(
        tmp_path / "sources.jsonl",
        [
            {
                "format": "tutorbench-output-jsonl-v1",
                "path": legacy.name,
                "model_id": "z/model",
                "model_family": "zeta",
                "model_revision": "z-rev",
            },
            _single_source(
                single,
                model_id="a/model",
                family="alpha",
                revision="a-rev",
            ),
        ],
    )

    prepared = build_precomputed_tutor_response_batch(
        load_response_batch_source_manifest(source_manifest),
        load_fitted_scenario_roster(scenarios),
    )

    assert prepared.model_count == 2
    assert prepared.row_count == 4
    assert prepared.blank_count == 1
    assert prepared.synthesized_blank_count == 0
    assert prepared.sha256 == hashlib.sha256(prepared.payload).hexdigest()
    rows = [json.loads(line) for line in prepared.payload.splitlines()]
    assert [(row["model_id"], row["scenario_id"]) for row in rows] == [
        ("a/model", "s2"),
        ("a/model", "s1"),
        ("z/model", "s2"),
        ("z/model", "s1"),
    ]
    assert rows[1]["response"] == "   "
    assert rows[2]["metadata"]["source_format"] == "tutorbench-output-jsonl-v1"
    assert rows[2]["metadata"]["finish_reason"] == "stop"
    assert rows[2]["metadata"]["rendered_prompt"] == "prompt"
    assert rows[2]["metadata"]["rendered_prompt_sha256"] == hashlib.sha256(b"prompt").hexdigest()

    output = tmp_path / "out/batch.jsonl"
    report_path = tmp_path / "out/report.json"
    written = write_precomputed_tutor_response_batch(prepared, output, report_path)
    loaded = load_precomputed_tutor_response_batch(
        output,
        expected_sha256=prepared.sha256,
        expected_scenario_ids=("s2", "s1"),
    )

    assert loaded.model_count == 2
    assert written.output_path == output.resolve()
    assert written.report["output"]["path"] == str(output.resolve())
    assert json.loads(report_path.read_text())["validation"]["status"] == "passed"


def test_build_accepts_canonical_batch_source(tmp_path: Path) -> None:
    scenarios = _roster(tmp_path / "scenarios.jsonl")
    source = tmp_path / "batch-input.jsonl"
    source_payload = _write_jsonl(
        source,
        [
            {
                "model_id": "m",
                "model_family": "family",
                "model_revision": "revision",
                "scenario_id": "s2",
                "response": "two",
            },
            {
                "model_id": "m",
                "model_family": "family",
                "model_revision": "revision",
                "scenario_id": "s1",
                "response": "one",
            },
        ],
    )
    manifest = _manifest(
        tmp_path / "sources.jsonl",
        [
            {
                "format": "batch-jsonl-v1",
                "path": source.name,
                "sha256": hashlib.sha256(source_payload).hexdigest(),
            }
        ],
    )

    prepared = build_precomputed_tutor_response_batch(
        load_response_batch_source_manifest(manifest),
        load_fitted_scenario_roster(scenarios),
    )

    assert prepared.model_count == 1
    assert [json.loads(line)["scenario_id"] for line in prepared.payload.splitlines()] == [
        "s1",
        "s2",
    ]


def test_missing_rows_error_by_default_and_can_be_explicitly_blank(
    tmp_path: Path,
) -> None:
    scenarios = _roster(tmp_path / "scenarios.jsonl")
    source = tmp_path / "empty.jsonl"
    source.write_bytes(b"")
    manifest = _manifest(
        tmp_path / "sources.jsonl",
        [_single_source(source, model_id="m", family="family", revision="revision")],
    )
    loaded_manifest = load_response_batch_source_manifest(manifest)
    roster = load_fitted_scenario_roster(scenarios)

    with pytest.raises(ResponsePreparationError, match="exactly cover.*missing"):
        build_precomputed_tutor_response_batch(loaded_manifest, roster)

    prepared = build_precomputed_tutor_response_batch(
        loaded_manifest,
        roster,
        missing="blank",
    )

    assert prepared.blank_count == 2
    assert prepared.synthesized_blank_count == 2
    rows = [json.loads(line) for line in prepared.payload.splitlines()]
    assert all(row["response"] == "" for row in rows)
    assert all(row["metadata"] == {"preparation_status": "missing_source_row"} for row in rows)
    output = tmp_path / "would-be.jsonl"
    report = tmp_path / "would-be.report.json"
    checked = check_precomputed_tutor_response_batch_write(prepared, output, report)
    assert checked["output"]["path"] == str(output.resolve())
    assert not output.exists()
    assert not report.exists()


def test_build_rejects_duplicate_pairs_identity_conflicts_and_extra_scenarios(
    tmp_path: Path,
) -> None:
    scenarios = _roster(tmp_path / "scenarios.jsonl", ("s1",))
    source = tmp_path / "duplicates.jsonl"
    _write_jsonl(
        source,
        [
            {"scenario_id": "s1", "response": "one"},
            {"scenario_id": "s1", "response": "two"},
        ],
    )
    duplicate_manifest = _manifest(
        tmp_path / "duplicate-sources.jsonl",
        [_single_source(source, model_id="m", family="family", revision="revision")],
    )
    with pytest.raises(ResponsePreparationError, match="duplicate response"):
        build_precomputed_tutor_response_batch(
            load_response_batch_source_manifest(duplicate_manifest),
            load_fitted_scenario_roster(scenarios),
        )

    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_bytes(b"")
    second.write_bytes(b"")
    conflict_manifest = _manifest(
        tmp_path / "conflict-sources.jsonl",
        [
            _single_source(first, model_id="m", family="one", revision="revision"),
            _single_source(second, model_id="m", family="two", revision="revision"),
        ],
    )
    with pytest.raises(ResponsePreparationError, match="conflicts with identity"):
        build_precomputed_tutor_response_batch(
            load_response_batch_source_manifest(conflict_manifest),
            load_fitted_scenario_roster(scenarios),
            missing="blank",
        )

    extra = tmp_path / "extra.jsonl"
    _write_jsonl(extra, [{"scenario_id": "outside", "response": "answer"}])
    extra_manifest = _manifest(
        tmp_path / "extra-sources.jsonl",
        [_single_source(extra, model_id="m", family="family", revision="revision")],
    )
    with pytest.raises(ResponsePreparationError, match="outside the fitted bank"):
        build_precomputed_tutor_response_batch(
            load_response_batch_source_manifest(extra_manifest),
            load_fitted_scenario_roster(scenarios),
        )


def test_manifest_and_source_hashes_are_strict(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    _write_jsonl(source, [{"scenario_id": "s1", "response": "answer"}])
    null_hash_manifest = _manifest(
        tmp_path / "null-hash.jsonl",
        [
            {
                **_single_source(source, model_id="m", family="f", revision="r"),
                "sha256": None,
            }
        ],
    )
    with pytest.raises(ResponsePreparationError, match="lowercase 64-character"):
        load_response_batch_source_manifest(null_hash_manifest)

    wrong_hash_manifest = _manifest(
        tmp_path / "wrong-hash.jsonl",
        [
            _single_source(
                source,
                model_id="m",
                family="f",
                revision="r",
                sha256="0" * 64,
            )
        ],
    )
    with pytest.raises(ResponsePreparationError, match="SHA-256 mismatch"):
        build_precomputed_tutor_response_batch(
            load_response_batch_source_manifest(wrong_hash_manifest),
            load_fitted_scenario_roster(_roster(tmp_path / "scenarios.jsonl", ("s1",))),
        )


def test_writer_refuses_existing_and_source_collisions(tmp_path: Path) -> None:
    scenarios = _roster(tmp_path / "scenarios.jsonl", ("s1",))
    source = tmp_path / "source.jsonl"
    _write_jsonl(source, [{"scenario_id": "s1", "response": "answer"}])
    manifest_path = _manifest(
        tmp_path / "sources.jsonl",
        [_single_source(source, model_id="m", family="family", revision="revision")],
    )
    prepared = build_precomputed_tutor_response_batch(
        load_response_batch_source_manifest(manifest_path),
        load_fitted_scenario_roster(scenarios),
    )
    output = tmp_path / "output.jsonl"
    report = tmp_path / "report.json"
    output.write_text("existing")

    with pytest.raises(ResponsePreparationError, match="output already exists"):
        write_precomputed_tutor_response_batch(prepared, output, report)
    replaced = write_precomputed_tutor_response_batch(
        prepared,
        output,
        report,
        overwrite=True,
    )
    assert output.read_bytes() == prepared.payload
    assert replaced.sha256 == prepared.sha256
    with pytest.raises(ResponsePreparationError, match="source input"):
        write_precomputed_tutor_response_batch(
            prepared,
            source,
            report,
            overwrite=True,
        )
    assert source.read_text().endswith("\n")


def test_writer_leaves_no_output_when_staged_contract_validation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenarios = _roster(tmp_path / "scenarios.jsonl", ("s1",))
    source = tmp_path / "source.jsonl"
    _write_jsonl(source, [{"scenario_id": "s1", "response": "answer"}])
    manifest_path = _manifest(
        tmp_path / "sources.jsonl",
        [_single_source(source, model_id="m", family="family", revision="revision")],
    )
    prepared = build_precomputed_tutor_response_batch(
        load_response_batch_source_manifest(manifest_path),
        load_fitted_scenario_roster(scenarios),
    )
    output = tmp_path / "nested/output.jsonl"
    report = tmp_path / "nested/report.json"

    def fail_validation(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise ValueError("forced validation failure")

    monkeypatch.setattr(prepare_module, "load_precomputed_tutor_response_batch", fail_validation)

    with pytest.raises(ResponsePreparationError, match="final contract validation"):
        write_precomputed_tutor_response_batch(prepared, output, report)
    assert not output.exists()
    assert not report.exists()
    assert [path.name for path in output.parent.glob(".*")] == [
        ".edullm-response-batch-publish.lock"
    ]


@pytest.mark.parametrize("separator", ["\u0085", "\u2028", "\u2029"])
def test_unicode_line_separators_remain_inside_json_strings(
    tmp_path: Path,
    separator: str,
) -> None:
    scenarios = _roster(tmp_path / "scenarios.jsonl", ("s1",))
    source = tmp_path / "source.jsonl"
    _write_jsonl(source, [{"scenario_id": "s1", "response": f"left{separator}right"}])
    manifest_path = _manifest(
        tmp_path / "sources.jsonl",
        [_single_source(source, model_id="m", family="family", revision="revision")],
    )
    prepared = build_precomputed_tutor_response_batch(
        load_response_batch_source_manifest(manifest_path),
        load_fitted_scenario_roster(scenarios),
    )

    output = tmp_path / "out/batch.jsonl"
    report = tmp_path / "out/report.json"
    write_precomputed_tutor_response_batch(prepared, output, report)
    loaded = load_precomputed_tutor_response_batch(
        output,
        expected_sha256=prepared.sha256,
        expected_scenario_ids=("s1",),
    )

    assert loaded.models_by_id["m"].responses["s1"].response == f"left{separator}right"


def test_legacy_prompt_is_hash_bound_and_flagged_issue_rows_fail_closed(
    tmp_path: Path,
) -> None:
    scenarios = _roster(tmp_path / "scenarios.jsonl", ("s1",))
    legacy = tmp_path / "legacy.jsonl"

    def prepare_current() -> Any:
        manifest_path = _manifest(
            tmp_path / "sources.jsonl",
            [
                {
                    "format": "tutorbench-output-jsonl-v1",
                    "path": legacy.name,
                    "model_id": "m",
                    "model_family": "family",
                    "model_revision": "r",
                }
            ],
        )
        return build_precomputed_tutor_response_batch(
            load_response_batch_source_manifest(manifest_path),
            load_fitted_scenario_roster(scenarios),
        )

    def prepare_prompt(prompt: str) -> Any:
        _write_jsonl(
            legacy,
            [
                _legacy_row(
                    model_id="m",
                    revision="r",
                    scenario_id="s1",
                    output="answer",
                    rendered_prompt=prompt,
                )
            ],
        )
        return prepare_current()

    first = prepare_prompt("first request")
    second = prepare_prompt("second request")
    assert first.sha256 != second.sha256

    _write_jsonl(
        legacy,
        [
            _legacy_row(
                model_id="m",
                revision="r",
                scenario_id="s1",
                output="answer",
                issue=1,
                issue_description="generation anomaly",
            )
        ],
    )
    with pytest.raises(ResponsePreparationError, match="nonblank Output for a flagged issue"):
        prepare_current()


def test_writer_requires_colocated_pair_and_rejects_case_alias_source(
    tmp_path: Path,
) -> None:
    scenarios = _roster(tmp_path / "scenarios.jsonl", ("s1",))
    source = tmp_path / "Responses.JSONL"
    _write_jsonl(source, [{"scenario_id": "s1", "response": "answer"}])
    manifest_path = _manifest(
        tmp_path / "sources.jsonl",
        [_single_source(source, model_id="m", family="family", revision="revision")],
    )
    prepared = build_precomputed_tutor_response_batch(
        load_response_batch_source_manifest(manifest_path),
        load_fitted_scenario_roster(scenarios),
    )

    with pytest.raises(ResponsePreparationError, match="same directory"):
        write_precomputed_tutor_response_batch(
            prepared,
            tmp_path / "one/output.jsonl",
            tmp_path / "two/report.json",
        )

    alias = tmp_path / "responses.jsonl"
    if alias.exists() and os.path.samefile(source, alias):
        with pytest.raises(ResponsePreparationError, match="source input"):
            write_precomputed_tutor_response_batch(
                prepared,
                alias,
                tmp_path / "report.json",
                overwrite=True,
            )
        assert source.read_bytes()


def test_writer_rolls_back_both_files_when_second_install_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenarios = _roster(tmp_path / "scenarios.jsonl", ("s1",))
    source = tmp_path / "source.jsonl"
    _write_jsonl(source, [{"scenario_id": "s1", "response": "new"}])
    manifest_path = _manifest(
        tmp_path / "sources.jsonl",
        [_single_source(source, model_id="m", family="family", revision="revision")],
    )
    prepared = build_precomputed_tutor_response_batch(
        load_response_batch_source_manifest(manifest_path),
        load_fitted_scenario_roster(scenarios),
    )
    output = tmp_path / "published/output.jsonl"
    report = tmp_path / "published/report.json"
    output.parent.mkdir()
    output.write_bytes(b"old output")
    report.write_bytes(b"old report")
    real_replace = os.replace
    failed = False

    def fail_second_install(source_path: str | Path, destination_path: str | Path) -> None:
        nonlocal failed
        if not failed and str(source_path).endswith(".stage") and Path(destination_path) == report:
            failed = True
            raise OSError("forced second-install failure")
        real_replace(source_path, destination_path)

    monkeypatch.setattr(prepare_module.os, "replace", fail_second_install)

    with pytest.raises(ResponsePreparationError, match="could not publish"):
        write_precomputed_tutor_response_batch(prepared, output, report, overwrite=True)

    assert output.read_bytes() == b"old output"
    assert report.read_bytes() == b"old report"
    assert not (output.parent / ".edullm-response-batch-publish.transaction.json").exists()


def test_writer_recovers_interrupted_partial_publication_on_next_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenarios = _roster(tmp_path / "scenarios.jsonl", ("s1",))
    source = tmp_path / "source.jsonl"
    _write_jsonl(source, [{"scenario_id": "s1", "response": "answer"}])
    manifest_path = _manifest(
        tmp_path / "sources.jsonl",
        [_single_source(source, model_id="m", family="family", revision="revision")],
    )
    prepared = build_precomputed_tutor_response_batch(
        load_response_batch_source_manifest(manifest_path),
        load_fitted_scenario_roster(scenarios),
    )
    output = tmp_path / "published/output.jsonl"
    report = tmp_path / "published/report.json"
    real_link = os.link
    link_calls = 0

    class SimulatedProcessCrash(BaseException):
        pass

    def crash_on_second_link(source_path: str | Path, destination_path: str | Path) -> None:
        nonlocal link_calls
        link_calls += 1
        if link_calls == 2:
            raise SimulatedProcessCrash
        real_link(source_path, destination_path)

    with monkeypatch.context() as patch_context:
        patch_context.setattr(prepare_module.os, "link", crash_on_second_link)
        with pytest.raises(SimulatedProcessCrash):
            write_precomputed_tutor_response_batch(prepared, output, report)

    journal = output.parent / ".edullm-response-batch-publish.transaction.json"
    assert output.is_file()
    assert not report.exists()
    assert journal.is_file()
    with pytest.raises(ValueError, match="incomplete paired publication"):
        load_precomputed_tutor_response_batch(
            output,
            expected_sha256=prepared.sha256,
            expected_scenario_ids=("s1",),
        )

    write_precomputed_tutor_response_batch(prepared, output, report)

    assert output.read_bytes() == prepared.payload
    assert json.loads(report.read_text())["output"]["sha256"] == prepared.sha256
    assert not journal.exists()


def test_reader_lock_prevents_publication_during_snapshot_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenarios = _roster(tmp_path / "scenarios.jsonl", ("s1",))
    source = tmp_path / "source.jsonl"
    _write_jsonl(source, [{"scenario_id": "s1", "response": "answer"}])
    manifest_path = _manifest(
        tmp_path / "sources.jsonl",
        [_single_source(source, model_id="m", family="family", revision="revision")],
    )
    prepared = build_precomputed_tutor_response_batch(
        load_response_batch_source_manifest(manifest_path),
        load_fitted_scenario_roster(scenarios),
    )
    output = tmp_path / "published/output.jsonl"
    report = tmp_path / "published/report.json"
    write_precomputed_tutor_response_batch(prepared, output, report)
    real_refusal_check = precomputed_module._refuse_incomplete_publication
    attempted_publication = False

    def attempt_publication_while_reader_is_locked(path: Path) -> None:
        nonlocal attempted_publication
        attempted_publication = True
        with pytest.raises(ResponsePreparationError, match="publication is active"):
            write_precomputed_tutor_response_batch(
                prepared,
                output,
                report,
                overwrite=True,
            )
        real_refusal_check(path)

    monkeypatch.setattr(
        precomputed_module,
        "_refuse_incomplete_publication",
        attempt_publication_while_reader_is_locked,
    )

    loaded = load_precomputed_tutor_response_batch(
        output,
        expected_sha256=prepared.sha256,
        expected_scenario_ids=("s1",),
    )

    assert attempted_publication
    assert loaded.sha256 == prepared.sha256


def test_reader_retries_when_publication_lock_appears_during_optimistic_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenarios = _roster(tmp_path / "scenarios.jsonl", ("s1",))

    def prepare(directory: Path, response: str) -> Any:
        source = directory / "source.jsonl"
        _write_jsonl(source, [{"scenario_id": "s1", "response": response}])
        manifest_path = _manifest(
            directory / "sources.jsonl",
            [_single_source(source, model_id="m", family="family", revision="revision")],
        )
        return build_precomputed_tutor_response_batch(
            load_response_batch_source_manifest(manifest_path),
            load_fitted_scenario_roster(scenarios),
        )

    old = prepare(tmp_path / "old", "old")
    new = prepare(tmp_path / "new", "new")
    output = tmp_path / "published/output.jsonl"
    report = tmp_path / "published/report.json"
    output.parent.mkdir()
    output.write_bytes(old.payload)
    report.write_bytes(b"old report")
    real_read_bytes = Path.read_bytes
    publication_ran = False

    def publish_after_snapshot_read(path: Path) -> bytes:
        nonlocal publication_ran
        payload = real_read_bytes(path)
        if path == output and not publication_ran:
            publication_ran = True
            write_precomputed_tutor_response_batch(new, output, report, overwrite=True)
        return payload

    monkeypatch.setattr(Path, "read_bytes", publish_after_snapshot_read)

    loaded = load_precomputed_tutor_response_batch(
        output,
        expected_sha256=new.sha256,
        expected_scenario_ids=("s1",),
    )

    assert publication_ran
    assert loaded.sha256 == new.sha256
    assert loaded.models_by_id["m"].responses["s1"].response == "new"


def test_writer_rejects_report_copied_from_another_prepared_batch(tmp_path: Path) -> None:
    scenarios = _roster(tmp_path / "scenarios.jsonl", ("s1",))

    def prepare(directory: Path) -> Any:
        source = directory / "source.jsonl"
        _write_jsonl(source, [{"scenario_id": "s1", "response": "same payload"}])
        manifest_path = _manifest(
            directory / "sources.jsonl",
            [_single_source(source, model_id="m", family="family", revision="revision")],
        )
        return build_precomputed_tutor_response_batch(
            load_response_batch_source_manifest(manifest_path),
            load_fitted_scenario_roster(scenarios),
        )

    first = prepare(tmp_path / "first")
    second = prepare(tmp_path / "second")
    assert first.payload == second.payload
    mismatched = replace(
        first,
        report=second.report,
        report_sha256=second.report_sha256,
    )

    with pytest.raises(ResponsePreparationError, match="authoritative source receipt"):
        write_precomputed_tutor_response_batch(
            mismatched,
            tmp_path / "out/batch.jsonl",
            tmp_path / "out/report.json",
        )


def test_recovery_never_deletes_unrelated_post_crash_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenarios = _roster(tmp_path / "scenarios.jsonl", ("s1",))
    source = tmp_path / "source.jsonl"
    _write_jsonl(source, [{"scenario_id": "s1", "response": "new"}])
    manifest_path = _manifest(
        tmp_path / "sources.jsonl",
        [_single_source(source, model_id="m", family="family", revision="revision")],
    )
    prepared = build_precomputed_tutor_response_batch(
        load_response_batch_source_manifest(manifest_path),
        load_fitted_scenario_roster(scenarios),
    )
    output = tmp_path / "published/output.jsonl"
    report = tmp_path / "published/report.json"
    output.parent.mkdir()
    output.write_bytes(b"old output")
    report.write_bytes(b"old report")
    real_replace = os.replace

    class SimulatedProcessCrash(BaseException):
        pass

    def crash_during_report_install(source_path: str | Path, destination_path: str | Path) -> None:
        if str(source_path).endswith(".stage") and Path(destination_path) == report:
            raise SimulatedProcessCrash
        real_replace(source_path, destination_path)

    with monkeypatch.context() as patch_context:
        patch_context.setattr(prepare_module.os, "replace", crash_during_report_install)
        with pytest.raises(SimulatedProcessCrash):
            write_precomputed_tutor_response_batch(prepared, output, report, overwrite=True)

    output.write_bytes(b"independent recovery data")

    with pytest.raises(ResponsePreparationError, match="unrelated post-crash recovery data"):
        write_precomputed_tutor_response_batch(prepared, output, report, overwrite=True)

    assert output.read_bytes() == b"independent recovery data"
    assert (output.parent / ".edullm-response-batch-publish.transaction.json").is_file()


def test_recovery_preserves_destination_swapped_after_digest_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenarios = _roster(tmp_path / "scenarios.jsonl", ("s1",))
    source = tmp_path / "source.jsonl"
    _write_jsonl(source, [{"scenario_id": "s1", "response": "new"}])
    manifest_path = _manifest(
        tmp_path / "sources.jsonl",
        [_single_source(source, model_id="m", family="family", revision="revision")],
    )
    prepared = build_precomputed_tutor_response_batch(
        load_response_batch_source_manifest(manifest_path),
        load_fitted_scenario_roster(scenarios),
    )
    output = tmp_path / "published/output.jsonl"
    report = tmp_path / "published/report.json"
    output.parent.mkdir()
    output.write_bytes(b"old output")
    report.write_bytes(b"old report")
    real_replace = os.replace

    class SimulatedProcessCrash(BaseException):
        pass

    def crash_during_report_install(source_path: str | Path, destination_path: str | Path) -> None:
        if str(source_path).endswith(".stage") and Path(destination_path) == report:
            raise SimulatedProcessCrash
        real_replace(source_path, destination_path)

    with monkeypatch.context() as patch_context:
        patch_context.setattr(prepare_module.os, "replace", crash_during_report_install)
        with pytest.raises(SimulatedProcessCrash):
            write_precomputed_tutor_response_batch(prepared, output, report, overwrite=True)

    swapped = False

    def swap_at_capture(source_path: str | Path, destination_path: str | Path) -> None:
        nonlocal swapped
        if (
            not swapped
            and Path(source_path) == output
            and str(destination_path).endswith(".rollback")
        ):
            swapped = True
            output.write_bytes(b"independent concurrent data")
        real_replace(source_path, destination_path)

    with monkeypatch.context() as patch_context:
        patch_context.setattr(prepare_module.os, "replace", swap_at_capture)
        with pytest.raises(ResponsePreparationError, match="changed after rollback validation"):
            write_precomputed_tutor_response_batch(prepared, output, report, overwrite=True)

    assert swapped
    assert output.read_bytes() == b"independent concurrent data"
    assert not list(output.parent.glob("*.rollback"))
    assert (output.parent / ".edullm-response-batch-publish.transaction.json").is_file()
