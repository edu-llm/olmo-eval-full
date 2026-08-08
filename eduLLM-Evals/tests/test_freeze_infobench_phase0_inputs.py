from __future__ import annotations

import importlib.util
import json
import sys
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "freeze_infobench_phase0_inputs.py"
SPEC = importlib.util.spec_from_file_location("freeze_infobench_phase0_inputs", SCRIPT)
assert SPEC and SPEC.loader
freeze = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = freeze
SPEC.loader.exec_module(freeze)


PROVENANCE = {
    "judge_name": "qwen",
    "judge_model": "Qwen/test",
    "judge_revision": "revision",
    "adapter": "generic-binary",
    "prompt_version": "judge-validation-v3",
    "normalization_version": "judge-normalization-v3",
    "evidence_policy_version": "criterion-evidence-gate-v1",
    "prompt_variant": "canonical",
    "replicate_id": "r1",
}


def _jsonl(rows: list[dict]) -> bytes:
    return b"".join((json.dumps(row) + "\n").encode() for row in rows)


def _write_sources(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    calibration = tmp_path / "calibration.zip"
    judge_manifest = {
        "judge_expected": PROVENANCE,
        "models": ["org/model-a", "org/model-b"],
        "n_criteria": 2,
    }
    verdict_rows = [
        {
            "model": "org/model-a",
            "criterion_id": "ifb_0000_c01",
            "y": 0,
            "verdict": "fail",
            "source": "auto_fail",
        },
        {
            "model": "org/model-a",
            "criterion_id": "ifb_0000_c02",
            "y": 1,
            "verdict": "pass",
            "source": "ingest",
            **PROVENANCE,
        },
        {
            "model": "org/model-b",
            "criterion_id": "ifb_0000_c01",
            "y": 0,
            "verdict": "fail",
            "source": "ingest",
            **PROVENANCE,
        },
        {
            "model": "org/model-b",
            "criterion_id": "ifb_0000_c02",
            "y": None,
            "verdict": "no_decision",
            "source": "ingest_no_decision",
            **PROVENANCE,
        },
    ]
    matrix = (
        "model,ifb_0000_c01,ifb_0000_c02\n"
        "org/model-a,0,1\n"
        "org/model-b,0,\n"
    ).encode()
    with zipfile.ZipFile(calibration, "w") as archive:
        archive.writestr("InFoBench/response_matrix.csv", matrix)
        archive.writestr("InFoBench/manifest.json", json.dumps(judge_manifest))
        archive.writestr("InFoBench/verdicts.jsonl", _jsonl(verdict_rows))

    responses = tmp_path / "responses.zip"
    with zipfile.ZipFile(responses, "w") as archive:
        for suffix, model in (("model-a", "org/model-a"), ("model-b", "org/model-b")):
            rows = [
                {
                    "Benchmark": "InFoBench",
                    "Scenario": f"ifb_{index:04d}",
                    "Model": model,
                    "Output": "answer",
                    "Finish Reason": "stop",
                }
                for index in range(2)
            ]
            archive.writestr(
                "full200_results/Outputs/open/infobench/"
                f"org__{suffix}.responses.jsonl",
                _jsonl(rows),
            )

    fit_manifest = tmp_path / "fit_manifest.json"
    fit_manifest.write_text(
        json.dumps(
            {
                "block": {
                    "n_criteria_in": 2,
                    "n_items_fit": 1,
                    "dropped_all_fail": 1,
                    "dropped_all_pass": 0,
                    "dropped_missing_qrow": 0,
                    "dropped_all_fail_criteria": ["ifb_0000_c01"],
                }
            }
        ),
        encoding="utf-8",
    )
    export_manifest = tmp_path / "export_manifest.json"
    export_manifest.write_text(
        json.dumps(
            {
                "counts": {
                    "source_criteria": 2,
                    "calibration_rows": 1,
                    "exported_criteria": 1,
                    "excluded_criteria": 1,
                },
                "exclusion_reason_counts": {"unfitted": 1},
                "excluded_criteria": [
                    {"criterion_id": "ifb_0000_c01", "reasons": ["unfitted"]}
                ],
            }
        ),
        encoding="utf-8",
    )
    return responses, calibration, fit_manifest, export_manifest


def _pins(responses: Path, calibration: Path) -> object:
    with zipfile.ZipFile(calibration) as archive:
        return freeze.Pins(
            response_archive_sha256=freeze.sha256_file(responses),
            calibration_archive_sha256=freeze.sha256_file(calibration),
            matrix_sha256=freeze.sha256_zip_member(
                archive, "InFoBench/response_matrix.csv"
            ),
            judge_manifest_sha256=freeze.sha256_zip_member(
                archive, "InFoBench/manifest.json"
            ),
            verdicts_sha256=freeze.sha256_zip_member(
                archive, "InFoBench/verdicts.jsonl"
            ),
        )


def _build(tmp_path: Path, output_name: str) -> dict:
    responses, calibration, fit_manifest, export_manifest = _write_sources(tmp_path)
    return freeze.build_bundle(
        response_archive_path=responses,
        calibration_archive_path=calibration,
        fit_manifest_path=fit_manifest,
        export_manifest_path=export_manifest,
        output_dir=tmp_path / output_name,
        pins=_pins(responses, calibration),
        expectations=freeze.Expectations(
            models=2, criteria=2, scenarios_per_model=2, missing_cells=1
        ),
        storage_uri="local-only://test/infobench_phase0_inputs.tar.gz",
    )


def test_bundle_is_deterministic_and_self_verifying(tmp_path: Path) -> None:
    first = _build(tmp_path, "first")
    second = freeze.build_bundle(
        response_archive_path=tmp_path / "responses.zip",
        calibration_archive_path=tmp_path / "calibration.zip",
        fit_manifest_path=tmp_path / "fit_manifest.json",
        export_manifest_path=tmp_path / "export_manifest.json",
        output_dir=tmp_path / "second",
        pins=_pins(tmp_path / "responses.zip", tmp_path / "calibration.zip"),
        expectations=freeze.Expectations(
            models=2, criteria=2, scenarios_per_model=2, missing_cells=1
        ),
        storage_uri="local-only://test/infobench_phase0_inputs.tar.gz",
    )
    assert first["archive"]["sha256"] == second["archive"]["sha256"]

    archive = tmp_path / "first" / freeze.BUNDLE_NAME
    verified = freeze.verify_archive(
        archive, expected_archive_sha256=first["archive"]["sha256"]
    )
    assert verified["members"] == 10
    assert verified["manifest"]["cohort"]["models"] == 2
    assert verified["manifest"]["criterion_count_flow"]["reconciled"] is True

    with tarfile.open(archive, "r:gz") as bundle:
        names = bundle.getnames()
        assert names == sorted(names)
        assert all(member.mtime == 0 for member in bundle.getmembers())


def test_count_flow_mismatch_blocks_bundle(tmp_path: Path) -> None:
    responses, calibration, fit_manifest, export_manifest = _write_sources(tmp_path)
    value = json.loads(export_manifest.read_text(encoding="utf-8"))
    value["counts"]["exported_criteria"] = 0
    export_manifest.write_text(json.dumps(value), encoding="utf-8")

    try:
        freeze.build_bundle(
            response_archive_path=responses,
            calibration_archive_path=calibration,
            fit_manifest_path=fit_manifest,
            export_manifest_path=export_manifest,
            output_dir=tmp_path / "out",
            pins=_pins(responses, calibration),
            expectations=freeze.Expectations(
                models=2, criteria=2, scenarios_per_model=2, missing_cells=1
            ),
        )
    except freeze.BundleError as exc:
        assert "fitted-to-export" in str(exc)
    else:
        raise AssertionError("irreconcilable criterion counts were accepted")
