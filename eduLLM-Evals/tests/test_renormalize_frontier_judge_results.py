from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT_PATH = ROOT / "scripts" / "renormalize_frontier_judge_results.py"
SPEC = importlib.util.spec_from_file_location(
    "renormalize_frontier_judge_results", SCRIPT_PATH
)
assert SPEC and SPEC.loader
renormalizer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = renormalizer
SPEC.loader.exec_module(renormalizer)
frontier = renormalizer.frontier


ACCEPTED_RAW = (
    '{"verdict":"pass","rationale":"All parts are satisfied.",'
    '"evidence":"The candidate explicitly gives the requested result."}'
)
RECOVERABLE_RAW = (
    "I checked the response against each part of the criterion.\n\n"
    '{"verdict":"fail","rationale":"A required part is missing.",'
    '"evidence":"NONE"}'
)
FIRST_VALID_RAW = (
    '{"verdict":"pass","rationale":"The requirement is met.",'
    '"evidence":"The candidate states the result."}'
)
SECOND_VALID_RAW = (
    '{"verdict":"fail","rationale":"The requirement is missing.",'
    '"evidence":"NONE"}'
)
AMBIGUOUS_RAW = f"{FIRST_VALID_RAW}\n{SECOND_VALID_RAW}"
MALFORMED_RAW = (
    '{"verdict":"pass","rationale":"Looks correct.",'
    '"evidence":"Candidate response","extra":"not allowed"}'
)


def _base_row(
    *,
    judge: str,
    configuration: dict,
    configuration_hash: str,
    frozen_hash: str,
    wave: str,
    case_id: str,
) -> dict:
    spec = frontier.FRONTIER_JUDGES[judge]
    variant, replicate = frontier.WAVES[wave]
    return {
        "case_id": case_id,
        "response_id": f"response-{case_id}",
        "scenario_id": f"scenario-{case_id}",
        "criterion_id": f"criterion-{case_id}",
        "candidate_family": "google" if spec.family != "google" else "openai",
        "judge_name": judge,
        "judge_family": spec.family,
        "judge_model": configuration["model_id"],
        "judge_revision": configuration["revision"],
        "served_model": configuration["model_id"],
        "resolved_provider_model": f"resolved/{judge}",
        "provider": spec.provider,
        "backend": "truefoundry",
        "base_url": frontier.TRUEFOUNDRY_BASE_URL,
        "api_surface": "chat.completions",
        "request_token_param": "max_completion_tokens",
        "request_temperature": None,
        "checkpoint_provenance": "truefoundry_gateway_configured_model",
        "checkpoint_verified_by_runner": False,
        "adapter": frontier.ADAPTER,
        "prompt_version": frontier.base.PROMPT_VERSION,
        "evidence_policy_version": frontier.base.EVIDENCE_POLICY_VERSION,
        "prompt_variant": variant,
        "replicate_id": replicate,
        "normalization_version": renormalizer.SOURCE_NORMALIZATION_VERSION,
        "routing_version": frontier.ROUTING_VERSION,
        "configuration_hash": configuration_hash,
        "frozen_configuration_hash": frozen_hash,
        "input_hash": f"input-{case_id}",
        "prompt_hash": f"prompt-{variant}-{case_id}",
        "attempt": 1,
        "provider_attempts": 1,
        "raw_response": {"id": f"raw-{case_id}"},
        "usage": {"total_tokens": 10},
        "latency_ms": 1.5,
        "created_at": "2026-07-26T00:00:00+00:00",
    }


def _write_wave(root: Path, judge: str, wave: str) -> tuple[Path, Path]:
    spec = frontier.FRONTIER_JUDGES[judge]
    variant, replicate = frontier.WAVES[wave]
    configuration = {
        "judge_name": judge,
        "judge_family": spec.family,
        "provider": spec.provider,
        "backend": "truefoundry",
        "base_url": frontier.TRUEFOUNDRY_BASE_URL,
        "model_id": frontier.TRUEFOUNDRY_DEFAULT_MODELS[judge],
        "model_id_source": "built_in_default",
        "model_id_env": spec.model_env,
        "revision": None,
        "api_surface": "chat.completions",
        "api_key_envs": [frontier.TRUEFOUNDRY_API_KEY_ENV],
        "adapter": frontier.ADAPTER,
        "prompt_version": frontier.base.PROMPT_VERSION,
        "evidence_policy_version": frontier.base.EVIDENCE_POLICY_VERSION,
        "normalization_version": renormalizer.SOURCE_NORMALIZATION_VERSION,
        "routing_version": frontier.ROUTING_VERSION,
        "prompt_variant": variant,
        "replicate_id": replicate,
        "runner_sha256": frontier.base.file_sha256(frontier.__file__),
        "generation": {
            "max_output_tokens": 4096,
            "token_param": "max_completion_tokens",
            "reasoning_level": None,
            "temperature": None,
        },
    }
    configuration_hash = frontier.base.stable_hash(configuration)
    frozen_hash = frontier.base.frozen_configuration_hash(configuration)

    rows: list[dict] = []
    cases = [
        (
            "accepted",
            {
                "verdict": "pass",
                "native_score": 1,
                "rationale": "All parts are satisfied.",
                "evidence": "The candidate explicitly gives the requested result.",
                "status": "ok",
                "error": None,
                "raw_output": ACCEPTED_RAW,
            },
        ),
        (
            "recoverable",
            {
                "verdict": "no_decision",
                "native_score": None,
                "rationale": "",
                "evidence": "",
                "status": "parse_error",
                "error": "expected one top-level JSON object",
                "raw_output": RECOVERABLE_RAW,
            },
        ),
        (
            "ambiguous",
            {
                "verdict": "no_decision",
                "native_score": None,
                "rationale": "",
                "evidence": "",
                "status": "parse_error",
                "error": "Extra data",
                "raw_output": AMBIGUOUS_RAW,
            },
        ),
        (
            "malformed",
            {
                "verdict": "no_decision",
                "native_score": None,
                "rationale": "",
                "evidence": "",
                "status": "parse_error",
                "error": "judgment fields differ",
                "raw_output": MALFORMED_RAW,
            },
        ),
        (
            "generation_error",
            {
                "verdict": "no_decision",
                "native_score": None,
                "rationale": "",
                "evidence": "",
                "status": "generation_error",
                "error": "provider timed out",
                # V3 may recover any non-ok attempt with a complete safe object.
                "raw_output": FIRST_VALID_RAW,
            },
        ),
    ]
    for case_id, decision_fields in cases:
        row = _base_row(
            judge=judge,
            configuration=configuration,
            configuration_hash=configuration_hash,
            frozen_hash=frozen_hash,
            wave=wave,
            case_id=case_id,
        )
        row.update(decision_fields)
        rows.append(row)

    output = root / judge / wave / f"{wave}.jsonl"
    renormalizer.write_jsonl(output, rows)
    manifest = {
        "status": "complete_with_errors",
        "started_at": "2026-07-26T00:00:00+00:00",
        "completed_at": "2026-07-26T00:01:00+00:00",
        "output_file": str(output),
        "judge_name": judge,
        "judge_family": spec.family,
        "backend": "truefoundry",
        "wave": wave,
        "prompt_variant": variant,
        "replicate_id": replicate,
        "eligible_case_count": len(rows),
        "own_family_excluded_count": 0,
        "configuration": configuration,
        "configuration_hash": configuration_hash,
        "frozen_configuration_hash": frozen_hash,
        "usable_decisions": 1,
        "no_decision_rows": 4,
        "resolved_provider_models": [f"resolved/{judge}"],
        "missing_resolved_model_rows": 0,
        "model_provenance_consistent": True,
    }
    manifest_path = output.with_suffix(".manifest.json")
    renormalizer.write_json(manifest_path, manifest)
    return output, manifest_path


def _write_tree(root: Path) -> list[tuple[Path, Path]]:
    return [
        _write_wave(root, judge, wave)
        for judge in frontier.FRONTIER_JUDGES
        for wave in frontier.WAVES
    ]


def _archived_attempt(primary: dict, *, attempt: int, raw_output: str) -> dict:
    row = dict(primary)
    row.update(
        {
            "attempt": attempt,
            "verdict": "no_decision",
            "native_score": None,
            "rationale": "",
            "evidence": "",
            "status": "parse_error",
            "error": "archived parser rejection",
            "raw_output": raw_output,
            "retry_archived_at": f"2026-07-26T00:00:0{attempt}+00:00",
        }
    )
    return row


@pytest.mark.parametrize(
    ("raw", "action", "verdict"),
    [
        (RECOVERABLE_RAW, "recovered_embedded_json", "fail"),
        (
            f"prefix {FIRST_VALID_RAW} harmless trailing prose",
            "unresolved_nonterminal",
            None,
        ),
        (
            f"Explanation first.\n```json\n{FIRST_VALID_RAW}\n```",
            "recovered_embedded_json",
            "pass",
        ),
        (AMBIGUOUS_RAW, "unresolved_ambiguous", None),
        (
            f'{FIRST_VALID_RAW}\n{{"verdict":"fail"',
            "unresolved_ambiguous",
            None,
        ),
        (
            f'{{"result":{FIRST_VALID_RAW}}}',
            "unresolved_ambiguous",
            None,
        ),
        (f"[{FIRST_VALID_RAW}]", "unresolved_ambiguous", None),
        (f"[{FIRST_VALID_RAW}", "unresolved_nonterminal", None),
        (f'{{"result":{FIRST_VALID_RAW}', "unresolved_nonterminal", None),
        (
            '{"verdict":"pass","verdict":"fail",'
            '"rationale":"missing","evidence":"NONE"}',
            "unresolved_no_valid_object",
            None,
        ),
        (MALFORMED_RAW, "unresolved_no_valid_object", None),
        (
            '{"verdict":"pass","rationale":"met","evidence":"NONE"}',
            "unresolved_no_valid_object",
            None,
        ),
        ("analysis with {math} but no JSON", "unresolved_no_valid_object", None),
    ],
)
def test_embedded_recovery_is_strict_and_unambiguous(
    raw: str, action: str, verdict: str | None
) -> None:
    result = renormalizer.recover_embedded_judgment(raw)
    assert result.action == action
    assert (result.parsed.verdict if result.parsed else None) == verdict


@pytest.mark.parametrize(
    ("raw", "action", "method", "verdict"),
    [
        (
            r'{"verdict":"pass","rationale":"Uses \(x\) notation",'
            r'"evidence":"Shows \alpha explicitly"}',
            "recovered_illegal_escape_json",
            "illegal_escape_repair",
            "pass",
        ),
        (
            r'{"verdict":"fail","rationale":"Bad \u12G4 token",'
            r'"evidence":"NONE"}',
            "recovered_illegal_escape_json",
            "illegal_escape_repair",
            "fail",
        ),
        (
            r'{"verdict":"pass","rationale":"Uses \(x\)",'
            r'"evidence":"quote"',
            "unresolved_no_valid_object",
            None,
            None,
        ),
        (
            r'{"verdict":"pass","rationale":"Uses \(x\)",'
            r'"evidence":"quote"} trailing',
            "unresolved_nonterminal",
            None,
            None,
        ),
        (
            r'{"verdict":"pass","verdict":"fail",'
            r'"rationale":"Bad \q","evidence":"NONE"}',
            "unresolved_no_valid_object",
            None,
            None,
        ),
        (
            r'{"verdict":"pass","rationale":"student says "x" now",'
            r'"evidence":"quote"}',
            "unresolved_no_valid_object",
            None,
            None,
        ),
        (
            r'[{"verdict":"pass","rationale":"Uses \q",'
            r'"evidence":"quote"}',
            "unresolved_nonterminal",
            None,
            None,
        ),
        (
            r'{"verdict":"pass","rationale":"Uses \q",'
            r'"evidence":"NONE"}',
            "unresolved_no_valid_object",
            None,
            None,
        ),
    ],
)
def test_illegal_escape_repair_is_narrow_and_schema_gated(
    raw: str,
    action: str,
    method: str | None,
    verdict: str | None,
) -> None:
    result = renormalizer.evaluate_attempt(raw)
    assert result.action == action
    assert result.method == method
    assert (result.parsed.verdict if result.parsed else None) == verdict
    if method == "illegal_escape_repair":
        assert result.illegal_escape_offsets
        assert result.repaired_sha256


def test_escape_transform_does_not_touch_prose_or_legal_escapes() -> None:
    source = (
        r'outside\q {"value":"inside\q","newline":"line\n",'
        r'"unicode":"\u0041"}'
    )
    repaired, offsets = renormalizer._repair_illegal_json_string_escapes(source)
    assert repaired == (
        r'outside\q {"value":"inside\\q","newline":"line\n",'
        r'"unicode":"\u0041"}'
    )
    assert len(offsets) == 1
    assert offsets[0] == source.index(r"\q", source.index("inside"))


def test_escape_repair_rejects_multiple_repairable_objects() -> None:
    first = (
        r'{"verdict":"pass","rationale":"Uses \q",'
        r'"evidence":"quote"}'
    )
    second = (
        r'{"verdict":"fail","rationale":"Uses \z",'
        r'"evidence":"NONE"}'
    )
    result = renormalizer.evaluate_attempt(f"{first}\n{second}")
    assert result.action == "unresolved_ambiguous"
    assert result.parsed is None


def test_tree_recovery_is_copy_on_write_uniform_and_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    source_files = _write_tree(source_root)
    source_hashes = {
        path: frontier.base.file_sha256(path)
        for pair in source_files
        for path in pair
    }
    source_raw = {
        (path, row["case_id"]): row["raw_output"]
        for path, _ in source_files
        for row in renormalizer.load_jsonl(path)
    }

    def forbidden(*_args, **_kwargs):
        raise AssertionError("offline postprocessor tried to access an API path")

    monkeypatch.setattr(frontier, "_load_dotenv", forbidden)
    monkeypatch.setattr(frontier, "build_generator", forbidden)
    output_root = tmp_path / "derived"
    summary = renormalizer.renormalize_tree(source_root, output_root)

    assert summary["wave_count"] == 18
    assert summary["case_rows"] == 90
    assert summary["normalization_actions"] == {
        "recovered_from_primary": 36,
        "unchanged_accepted": 18,
        "unresolved_ambiguous": 18,
        "unresolved_no_valid_object": 18,
    }
    assert all(
        frontier.base.file_sha256(path) == digest
        for path, digest in source_hashes.items()
    )

    frozen_by_judge: dict[str, set[str]] = {}
    for source_path, source_manifest_path in source_files:
        relative = source_path.relative_to(source_root)
        output_path = output_root / relative
        rows = renormalizer.load_jsonl(output_path)
        by_id = {row["case_id"]: row for row in rows}
        assert by_id["accepted"]["verdict"] == "pass"
        assert by_id["accepted"]["status"] == "ok"
        assert by_id["recoverable"]["verdict"] == "fail"
        assert by_id["recoverable"]["status"] == "ok"
        assert by_id["recoverable"]["error"] is None
        assert by_id["ambiguous"]["verdict"] == "no_decision"
        assert by_id["ambiguous"]["status"] == "parse_error"
        assert by_id["malformed"]["verdict"] == "no_decision"
        assert by_id["generation_error"]["status"] == "ok"
        assert by_id["generation_error"]["verdict"] == "pass"
        assert {row["normalization_version"] for row in rows} == {
            renormalizer.NORMALIZATION_VERSION
        }
        assert all(
            row["raw_output"] == source_raw[(source_path, row["case_id"])]
            for row in rows
        )

        output_manifest = renormalizer.load_json(
            output_path.with_suffix(".manifest.json")
        )
        configuration = output_manifest["configuration"]
        assert configuration["normalization_base_dependency_sha256"] == (
            frontier.base.file_sha256(frontier.base.__file__)
        )
        assert frontier.base.stable_hash(configuration) == output_manifest[
            "configuration_hash"
        ]
        assert (
            frontier.base.frozen_configuration_hash(configuration)
            == output_manifest["frozen_configuration_hash"]
        )
        assert output_manifest["output_sha256"] == frontier.base.file_sha256(
            output_path
        )
        assert output_manifest["source"]["output_sha256"] == source_hashes[
            source_path
        ]
        assert output_manifest["source"]["manifest_sha256"] == source_hashes[
            source_manifest_path
        ]
        assert output_manifest["normalization_version"] == (
            renormalizer.NORMALIZATION_VERSION
        )
        judge = output_manifest["judge_name"]
        frozen_by_judge.setdefault(judge, set()).add(
            output_manifest["frozen_configuration_hash"]
        )
        for row in rows:
            assert row["configuration_hash"] == output_manifest[
                "configuration_hash"
            ]
            assert row["frozen_configuration_hash"] == output_manifest[
                "frozen_configuration_hash"
            ]
            audit = row["normalization_recovery"]
            assert audit["source_output_sha256"] == source_hashes[source_path]
            assert audit["source_manifest_sha256"] == source_hashes[
                source_manifest_path
            ]

    assert all(len(values) == 1 for values in frozen_by_judge.values())
    summary_on_disk = renormalizer.load_json(
        output_root / "renormalization_summary.json"
    )
    assert summary_on_disk["normalization_version"] == (
        renormalizer.NORMALIZATION_VERSION
    )
    for wave in summary_on_disk["waves"]:
        manifest_path = Path(wave["output_file"]).with_suffix(".manifest.json")
        assert wave["output_manifest_sha256"] == frontier.base.file_sha256(
            manifest_path
        )


def test_retry_history_selects_earliest_safe_attempt_and_preserves_primary_raw(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    _write_tree(source_root)
    wave = "canonical_r1"
    source_path = source_root / "gpt-5.5" / wave / f"{wave}.jsonl"
    rows = renormalizer.load_jsonl(source_path)
    by_id = {row["case_id"]: row for row in rows}

    recoverable_primary = by_id["recoverable"]
    recoverable_primary["attempt"] = 3
    recoverable_primary["raw_output"] = MALFORMED_RAW
    accepted_primary = by_id["accepted"]
    accepted_primary["attempt"] = 2
    renormalizer.write_jsonl(source_path, rows)

    pass_with_prose = f"Reasoning first.\n{FIRST_VALID_RAW}"
    retry_rows = [
        _archived_attempt(
            recoverable_primary,
            attempt=2,
            raw_output=pass_with_prose,
        ),
        _archived_attempt(
            accepted_primary,
            attempt=1,
            raw_output=SECOND_VALID_RAW,
        ),
        _archived_attempt(
            recoverable_primary,
            attempt=1,
            raw_output=RECOVERABLE_RAW,
        ),
    ]
    retry_path = source_path.with_name(f"{wave}.retry_history.jsonl")
    renormalizer.write_jsonl(retry_path, retry_rows)
    retry_hash = frontier.base.file_sha256(retry_path)

    output_root = tmp_path / "derived"
    renormalizer.renormalize_tree(source_root, output_root)
    output_path = output_root / "gpt-5.5" / wave / f"{wave}.jsonl"
    output_rows = {
        row["case_id"]: row for row in renormalizer.load_jsonl(output_path)
    }

    recovered = output_rows["recoverable"]
    assert recovered["verdict"] == "fail"
    assert recovered["status"] == "ok"
    assert recovered["raw_output"] == MALFORMED_RAW
    audit = recovered["normalization_recovery"]
    assert audit["action"] == "recovered_from_retry_history"
    assert audit["method"] == "embedded_terminal_json"
    assert audit["selected_source"]["attempt"] == 1
    assert audit["selected_source"]["artifact_kind"] == "retry_history"
    assert audit["selected_source"]["artifact_sha256"] == retry_hash
    assert [item["attempt"] for item in audit["considered_attempts"]] == [1, 2, 3]
    assert all(item["row_sha256"] for item in audit["considered_attempts"])
    assert all(item["raw_output_sha256"] for item in audit["considered_attempts"])
    assert audit["recoverable_attempt_verdicts"] == ["fail", "pass"]
    assert audit["recoverable_attempts_disagree"] is True

    accepted = output_rows["accepted"]
    assert accepted["verdict"] == "pass"
    assert accepted["normalization_recovery"]["action"] == "unchanged_accepted"
    assert [
        item["attempt"]
        for item in accepted["normalization_recovery"]["considered_attempts"]
    ] == [2]

    manifest = renormalizer.load_json(output_path.with_suffix(".manifest.json"))
    assert manifest["attempt_verdict_disagreement_rows"] == 1
    assert manifest["source"]["retry_history"] == {
        "file": str(retry_path),
        "sha256": retry_hash,
        "row_count": 3,
        "case_count": 2,
    }


def test_primary_illegal_escape_recovery_records_explicit_provenance(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    _write_tree(source_root)
    wave = "canonical_r1"
    source_path = source_root / "gemini-3.6-flash" / wave / f"{wave}.jsonl"
    rows = renormalizer.load_jsonl(source_path)
    primary = next(row for row in rows if row["case_id"] == "malformed")
    invalid_escape_raw = (
        r'{"verdict":"fail","rationale":"Uses \(x\) incorrectly",'
        r'"evidence":"NONE"}'
    )
    primary["raw_output"] = invalid_escape_raw
    primary["error"] = "Invalid \\escape"
    renormalizer.write_jsonl(source_path, rows)

    output_root = tmp_path / "derived"
    renormalizer.renormalize_tree(source_root, output_root)
    output_path = (
        output_root / "gemini-3.6-flash" / wave / f"{wave}.jsonl"
    )
    output = next(
        row
        for row in renormalizer.load_jsonl(output_path)
        if row["case_id"] == "malformed"
    )
    assert output["verdict"] == "fail"
    assert output["status"] == "ok"
    assert output["raw_output"] == invalid_escape_raw
    audit = output["normalization_recovery"]
    assert audit["action"] == "recovered_from_primary"
    assert audit["method"] == "illegal_escape_repair"
    assert audit["illegal_escape_repair_version"] == (
        renormalizer.ILLEGAL_ESCAPE_REPAIR_VERSION
    )
    assert audit["illegal_escape_offsets"]
    assert audit["selected_sha256"] == renormalizer.text_sha256(
        invalid_escape_raw
    )
    assert audit["repaired_sha256"]
    assert audit["selected_source"]["artifact_kind"] == "primary"


def test_retry_history_identity_mismatch_aborts_atomically(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    _write_tree(source_root)
    wave = "canonical_r1"
    source_path = source_root / "gpt-5.5" / wave / f"{wave}.jsonl"
    rows = renormalizer.load_jsonl(source_path)
    primary = next(row for row in rows if row["case_id"] == "recoverable")
    primary["attempt"] = 2
    renormalizer.write_jsonl(source_path, rows)
    archived = _archived_attempt(primary, attempt=1, raw_output=RECOVERABLE_RAW)
    archived["prompt_hash"] = "different-prompt"
    retry_path = source_path.with_name(f"{wave}.retry_history.jsonl")
    renormalizer.write_jsonl(retry_path, [archived])

    output_root = tmp_path / "derived"
    with pytest.raises(ValueError, match="retry identity field 'prompt_hash'"):
        renormalizer.renormalize_tree(source_root, output_root)
    assert not output_root.exists()


def test_retry_history_rejects_duplicate_or_noncontiguous_attempts(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    _write_tree(source_root)
    wave = "canonical_r1"
    source_path = source_root / "gpt-5.5" / wave / f"{wave}.jsonl"
    rows = renormalizer.load_jsonl(source_path)
    primary = next(row for row in rows if row["case_id"] == "recoverable")
    primary["attempt"] = 3
    renormalizer.write_jsonl(source_path, rows)
    retry_path = source_path.with_name(f"{wave}.retry_history.jsonl")
    renormalizer.write_jsonl(
        retry_path,
        [
            _archived_attempt(primary, attempt=1, raw_output=RECOVERABLE_RAW),
            _archived_attempt(primary, attempt=1, raw_output=RECOVERABLE_RAW),
        ],
    )

    output_root = tmp_path / "derived"
    with pytest.raises(ValueError, match="duplicate archived attempt"):
        renormalizer.renormalize_tree(source_root, output_root)
    assert not output_root.exists()


def test_incomplete_source_aborts_without_output(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_files = _write_tree(source_root)
    _, manifest_path = source_files[0]
    manifest = renormalizer.load_json(manifest_path)
    manifest["status"] = "starting"
    renormalizer.write_json(manifest_path, manifest)
    output_root = tmp_path / "derived"

    with pytest.raises(ValueError, match="not complete"):
        renormalizer.renormalize_tree(source_root, output_root)
    assert not output_root.exists()


def test_accepted_source_row_is_never_reinterpreted(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_files = _write_tree(source_root)
    output_path, _ = source_files[0]
    rows = renormalizer.load_jsonl(output_path)
    rows[0]["raw_output"] = SECOND_VALID_RAW
    # Keep the accepted row's original pass fields to create a mismatch.
    renormalizer.write_jsonl(output_path, rows)
    derived_root = tmp_path / "derived"

    with pytest.raises(ValueError, match="does not match the exact parser"):
        renormalizer.renormalize_tree(source_root, derived_root)
    assert not derived_root.exists()


def test_source_change_during_read_aborts_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    source_files = _write_tree(source_root)
    changing_path, _ = source_files[0]
    original_load = renormalizer.load_jsonl
    changed = False

    def load_then_change(path: Path) -> list[dict]:
        nonlocal changed
        rows = original_load(path)
        if path == changing_path and not changed:
            changed = True
            path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        return rows

    monkeypatch.setattr(renormalizer, "load_jsonl", load_then_change)
    derived_root = tmp_path / "derived"
    with pytest.raises(ValueError, match="changed while it was read"):
        renormalizer.renormalize_tree(source_root, derived_root)
    assert not derived_root.exists()


def test_cross_wave_resolved_model_drift_is_rejected(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    _write_tree(source_root)
    wave = "canonical_r2"
    output = source_root / "gpt-5.5" / wave / f"{wave}.jsonl"
    rows = renormalizer.load_jsonl(output)
    for row in rows:
        row["resolved_provider_model"] = "resolved/different-gpt"
    renormalizer.write_jsonl(output, rows)
    manifest_path = output.with_suffix(".manifest.json")
    manifest = renormalizer.load_json(manifest_path)
    manifest["resolved_provider_models"] = ["resolved/different-gpt"]
    renormalizer.write_json(manifest_path, manifest)

    derived_root = tmp_path / "derived"
    with pytest.raises(ValueError, match="differs across waves"):
        renormalizer.renormalize_tree(source_root, derived_root)
    assert not derived_root.exists()


def test_requires_new_complete_output_location(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    _write_tree(source_root)
    existing = tmp_path / "already-there"
    existing.mkdir()

    with pytest.raises(FileExistsError):
        renormalizer.renormalize_tree(source_root, existing)
    with pytest.raises(ValueError, match="outside"):
        renormalizer.renormalize_tree(source_root, source_root / "derived")
