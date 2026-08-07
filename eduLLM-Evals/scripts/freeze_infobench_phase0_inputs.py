#!/usr/bin/env python3
"""Freeze the exact InFoBench calibration inputs into a deterministic archive.

This is a data-freezing utility, not a calibration runner.  It combines the
already-normalized Qwen verdict table and response matrix with the 52 original
InFoBench tutor-response shards.  Before writing anything, it checks that all
four views of the cohort (matrix, verdicts, judge manifest, and response shards)
name the same models and that every matrix cell agrees with its normalized
verdict row.

The resulting ``.tar.gz`` is content-addressed and reproducible: file order,
tar metadata, and gzip timestamps are fixed.  The archive carries its own
payload checksums and provenance.  No file is uploaded by this script; absent
an explicit external URI, provenance marks the artifact as local-only.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import platform
import re
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO

ROOT = Path(__file__).resolve().parents[1]

BUNDLE_ROOT = "infobench_phase0_inputs"
BUNDLE_NAME = f"{BUNDLE_ROOT}.tar.gz"
RESPONSE_MEMBER_RE = re.compile(
    r"^full200_results/Outputs/open/infobench/[^/]+\.responses\.jsonl$"
)

DEFAULT_RESPONSE_ARCHIVE_SHA256 = (
    "1a535a451d03e5eb0e6de3f2fd6b8bbc4dfd38e941a0d761e0d5d168e06fbabd"
)
DEFAULT_CALIBRATION_ARCHIVE_SHA256 = (
    "45011c275fbaafd1d7ef5350bc90170bd5719b3a8a1858bd90a01b0c820dfeb1"
)
DEFAULT_MATRIX_SHA256 = (
    "087948fcaa884cde6660df1fb4964072ec60fe8aee398e293ed5f74db8f3f27c"
)
DEFAULT_JUDGE_MANIFEST_SHA256 = (
    "d2f19c55b8ebb8f3ebe0c641485e0b694ce861f2740835da760b7b51b40698e7"
)
DEFAULT_VERDICTS_SHA256 = (
    "03be745b0b9addc118dddef2636cc4ee2028a8665628fbcee5af38031a1807f6"
)
HISTORICAL_PIPELINE_COMMIT = "6d1be445461dde85719c87c0eb85e45872ae8b03"
HISTORICAL_REPORT_COMMIT = "c321919f8b6ba5b79e74e1a06f01d9bbdc16deb3"

CALIBRATION_MEMBERS = {
    "response_matrix": "InFoBench/response_matrix.csv",
    "judge_manifest": "InFoBench/manifest.json",
    "normalized_verdicts": "InFoBench/verdicts.jsonl",
}


class BundleError(RuntimeError):
    """Raised when the frozen bundle would be incomplete or inconsistent."""


@dataclass(frozen=True)
class Pins:
    response_archive_sha256: str
    calibration_archive_sha256: str
    matrix_sha256: str
    judge_manifest_sha256: str
    verdicts_sha256: str


@dataclass(frozen=True)
class Expectations:
    models: int = 52
    criteria: int = 2_250
    scenarios_per_model: int = 500
    missing_cells: int = 27


@dataclass(frozen=True)
class ZipPayload:
    archive: str
    member: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_zip_member(archive: zipfile.ZipFile, member: str) -> str:
    digest = hashlib.sha256()
    with archive.open(member) as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json_member(archive: zipfile.ZipFile, member: str) -> dict:
    try:
        with archive.open(member) as handle:
            value = json.load(handle)
    except KeyError as exc:
        raise BundleError(f"archive is missing required member {member!r}") from exc
    except json.JSONDecodeError as exc:
        raise BundleError(f"{member}: invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise BundleError(f"{member}: expected one JSON object")
    return value


def _check_hash(label: str, observed: str, expected: str) -> None:
    if observed != expected:
        raise BundleError(
            f"{label} SHA-256 mismatch: observed {observed}, expected {expected}"
        )


def _inventory_matrix(
    archive: zipfile.ZipFile,
    expectations: Expectations,
) -> tuple[list[str], list[str], dict[tuple[str, str], int | None], Counter]:
    member = CALIBRATION_MEMBERS["response_matrix"]
    try:
        raw = archive.open(member)
    except KeyError as exc:
        raise BundleError(f"archive is missing required member {member!r}") from exc
    with raw, io.TextIOWrapper(raw, encoding="utf-8", newline="") as text:
        reader = csv.reader(text)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise BundleError("response matrix is empty") from exc
        if not header or header[0] != "model":
            raise BundleError("response matrix first column must be 'model'")
        criteria = header[1:]
        if len(criteria) != len(set(criteria)):
            raise BundleError("response matrix contains duplicate criterion columns")
        if len(criteria) != expectations.criteria:
            raise BundleError(
                f"response matrix has {len(criteria)} criteria; "
                f"expected {expectations.criteria}"
            )

        models: list[str] = []
        cells: dict[tuple[str, str], int | None] = {}
        values: Counter = Counter()
        for line_number, row in enumerate(reader, 2):
            if len(row) != len(header):
                raise BundleError(
                    f"response matrix line {line_number} has {len(row)} columns; "
                    f"expected {len(header)}"
                )
            model = row[0].strip()
            if not model:
                raise BundleError(f"response matrix line {line_number} has blank model")
            if model in models:
                raise BundleError(f"response matrix repeats model {model!r}")
            models.append(model)
            for criterion_id, raw_value in zip(criteria, row[1:], strict=True):
                if raw_value == "":
                    value = None
                elif raw_value in {"0", "1"}:
                    value = int(raw_value)
                else:
                    raise BundleError(
                        f"response matrix has invalid value {raw_value!r} for "
                        f"{model}/{criterion_id}"
                    )
                cells[(model, criterion_id)] = value
                values["missing" if value is None else str(value)] += 1

    if len(models) != expectations.models:
        raise BundleError(
            f"response matrix has {len(models)} models; expected {expectations.models}"
        )
    if values["missing"] != expectations.missing_cells:
        raise BundleError(
            f"response matrix has {values['missing']} missing cells; "
            f"expected {expectations.missing_cells}"
        )
    return models, criteria, cells, values


def _expected_judge_provenance(manifest: dict) -> dict[str, object]:
    expected = manifest.get("judge_expected")
    if not isinstance(expected, dict):
        raise BundleError("judge manifest has no object-valued judge_expected")
    required = (
        "judge_name",
        "judge_model",
        "judge_revision",
        "adapter",
        "prompt_version",
        "normalization_version",
        "evidence_policy_version",
        "prompt_variant",
        "replicate_id",
    )
    missing = [field for field in required if not expected.get(field)]
    if missing:
        raise BundleError(f"judge manifest has blank provenance fields: {missing}")
    return {field: expected[field] for field in required}


def _inventory_verdicts(
    archive: zipfile.ZipFile,
    matrix_cells: dict[tuple[str, str], int | None],
    expected_provenance: dict[str, object],
) -> tuple[Counter, set[str], set[str]]:
    member = CALIBRATION_MEMBERS["normalized_verdicts"]
    counts: Counter = Counter()
    seen: set[tuple[str, str]] = set()
    models: set[str] = set()
    criteria: set[str] = set()
    try:
        raw = archive.open(member)
    except KeyError as exc:
        raise BundleError(f"archive is missing required member {member!r}") from exc

    with raw:
        for line_number, line in enumerate(raw, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BundleError(f"{member}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise BundleError(f"{member}:{line_number}: expected JSON object")
            model = str(row.get("model") or "").strip()
            criterion_id = str(row.get("criterion_id") or "").strip()
            key = (model, criterion_id)
            if not model or not criterion_id:
                raise BundleError(f"{member}:{line_number}: blank model or criterion_id")
            if key in seen:
                raise BundleError(f"{member}:{line_number}: duplicate cell {key}")
            if key not in matrix_cells:
                raise BundleError(f"{member}:{line_number}: cell is absent from matrix: {key}")
            seen.add(key)
            models.add(model)
            criteria.add(criterion_id)

            y = row.get("y")
            verdict = row.get("verdict")
            source = row.get("source")
            if y not in (0, 1, None):
                raise BundleError(f"{member}:{line_number}: invalid y={y!r}")
            expected_verdict = {0: "fail", 1: "pass", None: "no_decision"}[y]
            if verdict != expected_verdict:
                raise BundleError(
                    f"{member}:{line_number}: verdict={verdict!r} disagrees with y={y!r}"
                )
            if matrix_cells[key] != y:
                raise BundleError(
                    f"{member}:{line_number}: verdict y={y!r} disagrees with matrix "
                    f"value {matrix_cells[key]!r} for {key}"
                )
            if source != "auto_fail":
                for field, expected in expected_provenance.items():
                    if row.get(field) != expected:
                        raise BundleError(
                            f"{member}:{line_number}: {field}={row.get(field)!r}; "
                            f"expected {expected!r}"
                        )
            counts[(str(source), str(verdict))] += 1

    if seen != set(matrix_cells):
        raise BundleError(
            f"normalized verdict table covers {len(seen)} of {len(matrix_cells)} cells"
        )
    return counts, models, criteria


def _inventory_responses(
    archive: zipfile.ZipFile,
    expectations: Expectations,
) -> tuple[list[dict], set[str]]:
    members = sorted(name for name in archive.namelist() if RESPONSE_MEMBER_RE.fullmatch(name))
    if len(members) != expectations.models:
        raise BundleError(
            f"response archive has {len(members)} InFoBench shards; "
            f"expected {expectations.models}"
        )

    inventory: list[dict] = []
    all_models: set[str] = set()
    target_names: set[str] = set()
    for member in members:
        models: set[str] = set()
        scenarios: set[str] = set()
        rows = 0
        digest = hashlib.sha256()
        with archive.open(member) as handle:
            for line_number, line in enumerate(handle, 1):
                digest.update(line)
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise BundleError(f"{member}:{line_number}: invalid JSON: {exc}") from exc
                if not isinstance(row, dict):
                    raise BundleError(f"{member}:{line_number}: expected JSON object")
                if row.get("Benchmark") != "InFoBench":
                    raise BundleError(
                        f"{member}:{line_number}: Benchmark={row.get('Benchmark')!r}"
                    )
                model = str(row.get("Model") or "").strip()
                scenario = str(row.get("Scenario") or "").strip()
                if not model or not scenario:
                    raise BundleError(f"{member}:{line_number}: blank Model or Scenario")
                models.add(model)
                if scenario in scenarios:
                    raise BundleError(f"{member}: duplicate scenario {scenario!r}")
                scenarios.add(scenario)
                rows += 1
        if len(models) != 1:
            raise BundleError(f"{member}: expected one model, observed {sorted(models)}")
        if rows != expectations.scenarios_per_model:
            raise BundleError(
                f"{member}: has {rows} response rows; "
                f"expected {expectations.scenarios_per_model}"
            )
        model = next(iter(models))
        if model in all_models:
            raise BundleError(f"response archive repeats model {model!r}")
        all_models.add(model)
        target_name = PurePosixPath(member).name
        if target_name in target_names:
            raise BundleError(f"response shard basename collision: {target_name!r}")
        target_names.add(target_name)
        inventory.append(
            {
                "model": model,
                "rows": rows,
                "source_member": member,
                "bundle_path": f"inputs/responses/{target_name}",
                "sha256": digest.hexdigest(),
            }
        )
    return inventory, all_models


def _load_json_file(path: Path, label: str) -> dict:
    if not path.is_file():
        raise BundleError(f"{label} not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BundleError(f"{label} is invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise BundleError(f"{label} must contain one JSON object")
    return value


def _criterion_count_flow(fit_manifest_path: Path, export_manifest_path: Path) -> dict:
    fit = _load_json_file(fit_manifest_path, "fit manifest")
    export = _load_json_file(export_manifest_path, "export manifest")
    block = fit.get("block")
    counts = export.get("counts")
    reasons = export.get("exclusion_reason_counts")
    excluded = export.get("excluded_criteria")
    if not all(isinstance(value, dict) for value in (block, counts, reasons)):
        raise BundleError("fit/export manifests are missing count dictionaries")
    if not isinstance(excluded, list):
        raise BundleError("export manifest is missing excluded_criteria list")

    source = int(block["n_criteria_in"])
    fitted = int(block["n_items_fit"])
    all_fail = int(block["dropped_all_fail"])
    all_pass = int(block["dropped_all_pass"])
    missing_qrow = int(block["dropped_missing_qrow"])
    exported = int(counts["exported_criteria"])
    nonpositive = int(reasons.get("nonpositive_a", 0))
    unfitted = int(reasons.get("unfitted", 0))

    if source - all_fail - all_pass - missing_qrow != fitted:
        raise BundleError("source-to-fit criterion counts do not reconcile")
    if int(counts["source_criteria"]) != source:
        raise BundleError("fit and export manifests disagree on source criterion count")
    if int(counts["calibration_rows"]) != fitted:
        raise BundleError("fit and export manifests disagree on fitted criterion count")
    if unfitted != source - fitted:
        raise BundleError("export unfitted count does not equal source minus fitted")
    if fitted - nonpositive != exported:
        raise BundleError("fitted-to-export criterion counts do not reconcile")

    excluded_by_reason: dict[str, set[str]] = {}
    for position, row in enumerate(excluded, 1):
        if not isinstance(row, dict):
            raise BundleError(f"excluded criterion {position} is not an object")
        criterion_id = str(row.get("criterion_id") or "").strip()
        row_reasons = row.get("reasons")
        if not criterion_id or not isinstance(row_reasons, list):
            raise BundleError(f"excluded criterion {position} is malformed")
        for reason in row_reasons:
            excluded_by_reason.setdefault(str(reason), set()).add(criterion_id)
    all_fail_ids = set(block.get("dropped_all_fail_criteria") or [])
    if excluded_by_reason.get("unfitted", set()) != all_fail_ids:
        raise BundleError("unfitted export IDs differ from all-fail fit exclusions")
    if len(excluded_by_reason.get("nonpositive_a", set())) != nonpositive:
        raise BundleError("nonpositive_a exclusion IDs do not match the reported count")
    if len({str(row.get("criterion_id")) for row in excluded}) != int(
        counts["excluded_criteria"]
    ):
        raise BundleError("unique excluded criterion IDs do not match export count")

    return {
        "schema_version": "infobench-criterion-count-flow-v1",
        "source_criteria": source,
        "fitted_criteria": fitted,
        "exported_criteria": exported,
        "source_to_fit": {
            "excluded_all_fail": all_fail,
            "excluded_all_pass": all_pass,
            "excluded_missing_qrow": missing_qrow,
            "equation": f"{source} - {all_fail} - {all_pass} - {missing_qrow} = {fitted}",
        },
        "fit_to_export": {
            "excluded_nonpositive_discrimination": nonpositive,
            "equation": f"{fitted} - {nonpositive} = {exported}",
        },
        "total_excluded_from_export": source - exported,
        "total_equation": f"{source} - {unfitted} - {nonpositive} = {exported}",
        "reconciled": True,
        "evidence": {
            "fit_manifest": {
                "repo_path": _repo_relative(fit_manifest_path),
                "sha256": sha256_file(fit_manifest_path),
            },
            "export_manifest": {
                "repo_path": _repo_relative(export_manifest_path),
                "sha256": sha256_file(export_manifest_path),
            },
        },
    }


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def _git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _tar_info(path: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(path)
    info.size = size
    info.mode = 0o444
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    return info


def _add_stream(tar: tarfile.TarFile, path: str, size: int, handle: BinaryIO) -> None:
    tar.addfile(_tar_info(path, size), handle)


def _write_deterministic_archive(
    destination: Path,
    calibration_archive: zipfile.ZipFile,
    response_archive: zipfile.ZipFile,
    zip_payloads: dict[str, ZipPayload],
    generated_payloads: dict[str, bytes],
) -> None:
    with destination.open("wb") as raw_out:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw_out,
            compresslevel=9,
            mtime=0,
        ) as compressed:
            with tarfile.open(
                fileobj=compressed,
                mode="w",
                format=tarfile.USTAR_FORMAT,
            ) as tar:
                for path in sorted(set(zip_payloads) | set(generated_payloads)):
                    full_path = f"{BUNDLE_ROOT}/{path}"
                    if path in generated_payloads:
                        data = generated_payloads[path]
                        _add_stream(tar, full_path, len(data), io.BytesIO(data))
                        continue
                    source = zip_payloads[path]
                    archive = (
                        calibration_archive
                        if source.archive == "calibration"
                        else response_archive
                    )
                    info = archive.getinfo(source.member)
                    with archive.open(source.member) as handle:
                        _add_stream(tar, full_path, info.file_size, handle)


def _storage_record(uri: str | None, output_dir: Path) -> dict:
    if uri:
        return {
            "status": "externally_addressed_not_uploaded_by_this_script",
            "uri": uri,
            "fresh_clone_retrievable": not uri.startswith("local-only://"),
        }
    try:
        relative = (output_dir / BUNDLE_NAME).resolve().relative_to(ROOT.resolve())
        local_path = relative.as_posix()
    except ValueError:
        local_path = (output_dir / BUNDLE_NAME).name
    return {
        "status": "local_only_not_published",
        "uri": f"local-only://{local_path}",
        "fresh_clone_retrievable": False,
        "action_required": "publish the archive to immutable shared storage and replace this URI",
    }


def build_bundle(
    *,
    response_archive_path: Path,
    calibration_archive_path: Path,
    fit_manifest_path: Path,
    export_manifest_path: Path,
    output_dir: Path,
    pins: Pins,
    expectations: Expectations = Expectations(),
    storage_uri: str | None = None,
    provenance_out: Path | None = None,
    force: bool = False,
) -> dict:
    """Validate sources and write a deterministic bundle plus external manifest."""
    for label, path in (
        ("response archive", response_archive_path),
        ("calibration archive", calibration_archive_path),
    ):
        if not path.is_file():
            raise BundleError(f"{label} not found: {path}")
    response_archive_hash = sha256_file(response_archive_path)
    calibration_archive_hash = sha256_file(calibration_archive_path)
    _check_hash(
        "response archive", response_archive_hash, pins.response_archive_sha256
    )
    _check_hash(
        "calibration archive",
        calibration_archive_hash,
        pins.calibration_archive_sha256,
    )

    archive_path = output_dir / BUNDLE_NAME
    external_manifest_path = output_dir / "artifact_manifest.json"
    checksum_path = output_dir / f"{BUNDLE_NAME}.sha256"
    if not force:
        existing = [
            path
            for path in (archive_path, external_manifest_path, checksum_path)
            if path.exists()
        ]
        if existing:
            raise BundleError(
                "refusing to overwrite existing outputs without --force: "
                + ", ".join(str(path) for path in existing)
            )
    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(calibration_archive_path) as calibration_zip, zipfile.ZipFile(
        response_archive_path
    ) as response_zip:
        matrix_hash = sha256_zip_member(
            calibration_zip, CALIBRATION_MEMBERS["response_matrix"]
        )
        judge_manifest_hash = sha256_zip_member(
            calibration_zip, CALIBRATION_MEMBERS["judge_manifest"]
        )
        verdicts_hash = sha256_zip_member(
            calibration_zip, CALIBRATION_MEMBERS["normalized_verdicts"]
        )
        _check_hash("response matrix", matrix_hash, pins.matrix_sha256)
        _check_hash("judge manifest", judge_manifest_hash, pins.judge_manifest_sha256)
        _check_hash("normalized verdicts", verdicts_hash, pins.verdicts_sha256)

        judge_manifest = _read_json_member(
            calibration_zip, CALIBRATION_MEMBERS["judge_manifest"]
        )
        expected_provenance = _expected_judge_provenance(judge_manifest)
        models, criteria, matrix_cells, matrix_values = _inventory_matrix(
            calibration_zip, expectations
        )
        verdict_counts, verdict_models, verdict_criteria = _inventory_verdicts(
            calibration_zip, matrix_cells, expected_provenance
        )
        response_inventory, response_models = _inventory_responses(
            response_zip, expectations
        )

        manifest_models = set(judge_manifest.get("models") or [])
        model_set = set(models)
        if not (
            model_set == verdict_models == response_models == manifest_models
        ):
            raise BundleError(
                "model roster differs across matrix, verdicts, responses, or judge manifest"
            )
        if set(criteria) != verdict_criteria:
            raise BundleError("criterion roster differs between matrix and verdicts")
        if int(judge_manifest.get("n_criteria", -1)) != expectations.criteria:
            raise BundleError("judge manifest criterion count disagrees with expectations")

        count_flow = _criterion_count_flow(fit_manifest_path, export_manifest_path)
        if count_flow["source_criteria"] != expectations.criteria:
            raise BundleError("criterion count flow disagrees with the frozen matrix")

        zip_payloads: dict[str, ZipPayload] = {
            "inputs/response_matrix.csv": ZipPayload(
                "calibration", CALIBRATION_MEMBERS["response_matrix"]
            ),
            "inputs/judge_manifest.json": ZipPayload(
                "calibration", CALIBRATION_MEMBERS["judge_manifest"]
            ),
            "inputs/normalized_verdicts.jsonl": ZipPayload(
                "calibration", CALIBRATION_MEMBERS["normalized_verdicts"]
            ),
        }
        payload_hashes = {
            "inputs/response_matrix.csv": matrix_hash,
            "inputs/judge_manifest.json": judge_manifest_hash,
            "inputs/normalized_verdicts.jsonl": verdicts_hash,
        }
        for row in response_inventory:
            path = row["bundle_path"]
            zip_payloads[path] = ZipPayload("responses", row["source_member"])
            payload_hashes[path] = row["sha256"]

        storage = _storage_record(storage_uri, output_dir)
        internal_manifest = {
            "schema_version": "infobench-phase0-input-bundle-v1",
            "benchmark": "InFoBench",
            "purpose": "immutable inputs for calibration remediation; does not contain a new fit",
            "storage": storage,
            "source_archives": {
                "tutor_responses": {
                    "filename": response_archive_path.name,
                    "sha256": response_archive_hash,
                    "selected_members": len(response_inventory),
                },
                "normalized_calibration_inputs": {
                    "filename": calibration_archive_path.name,
                    "sha256": calibration_archive_hash,
                    "selected_members": len(CALIBRATION_MEMBERS),
                },
            },
            "cohort": {
                "models": len(models),
                "scenarios_per_model": expectations.scenarios_per_model,
                "criteria": len(criteria),
                "matrix_cells": len(matrix_cells),
                "matrix_value_counts": dict(sorted(matrix_values.items())),
                "normalized_verdict_counts": {
                    f"{source}/{verdict}": count
                    for (source, verdict), count in sorted(verdict_counts.items())
                },
                "response_shards": response_inventory,
            },
            "judge_provenance": expected_provenance,
            "payload": {
                path: {"sha256": digest}
                for path, digest in sorted(payload_hashes.items())
            },
            "criterion_count_flow": count_flow,
            "build_provenance": {
                "historical_source_commits": {
                    "calibration_pipeline": HISTORICAL_PIPELINE_COMMIT,
                    "curated_report": HISTORICAL_REPORT_COMMIT,
                },
                "repo_commit": _git_value("rev-parse", "HEAD"),
                "repo_commit_note": (
                    "base commit at freeze time; the builder is independently "
                    "pinned by its SHA-256"
                ),
                "repo_branch": _git_value("branch", "--show-current"),
                "builder": {
                    "repo_path": _repo_relative(Path(__file__)),
                    "sha256": sha256_file(Path(__file__)),
                },
                "environment": {
                    "python": platform.python_version(),
                    "implementation": platform.python_implementation(),
                    "platform": platform.platform(),
                    "pyproject": {
                        "repo_path": "pyproject.toml",
                        "sha256": sha256_file(ROOT / "pyproject.toml"),
                    },
                    "lockfile": {
                        "repo_path": "uv.lock",
                        "sha256": sha256_file(ROOT / "uv.lock"),
                    },
                },
                "determinism": {
                    "tar_format": "ustar",
                    "member_mtime": 0,
                    "member_uid_gid": 0,
                    "member_mode": "0444",
                    "gzip_mtime": 0,
                    "gzip_level": 9,
                },
                "rebuild_command": [
                    ".venv/bin/python",
                    "scripts/freeze_infobench_phase0_inputs.py",
                    "build",
                    "--response-archive",
                    response_archive_path.name,
                    "--calibration-archive",
                    calibration_archive_path.name,
                    "--output-dir",
                    "<output-dir>",
                ],
            },
        }
        manifest_bytes = _canonical_json(internal_manifest)
        count_flow_bytes = _canonical_json(count_flow)
        generated_payloads = {
            "metadata/ARTIFACT_MANIFEST.json": manifest_bytes,
            "metadata/CRITERION_COUNT_FLOW.json": count_flow_bytes,
            "metadata/evidence/calibration_mirt_manifest_strict.json": (
                fit_manifest_path.read_bytes()
            ),
            "metadata/evidence/export_manifest.json": export_manifest_path.read_bytes(),
        }
        all_checksummed = dict(payload_hashes)
        all_checksummed["metadata/ARTIFACT_MANIFEST.json"] = hashlib.sha256(
            manifest_bytes
        ).hexdigest()
        all_checksummed["metadata/CRITERION_COUNT_FLOW.json"] = hashlib.sha256(
            count_flow_bytes
        ).hexdigest()
        all_checksummed["metadata/evidence/calibration_mirt_manifest_strict.json"] = (
            sha256_file(fit_manifest_path)
        )
        all_checksummed["metadata/evidence/export_manifest.json"] = sha256_file(
            export_manifest_path
        )
        checksums = "".join(
            f"{digest}  {path}\n" for path, digest in sorted(all_checksummed.items())
        ).encode("utf-8")
        generated_payloads["metadata/SHA256SUMS.txt"] = checksums

        with tempfile.NamedTemporaryFile(
            prefix=f".{BUNDLE_ROOT}.",
            suffix=".tmp",
            dir=output_dir,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        try:
            _write_deterministic_archive(
                temporary_path,
                calibration_zip,
                response_zip,
                zip_payloads,
                generated_payloads,
            )
            os.replace(temporary_path, archive_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    archive_hash = sha256_file(archive_path)
    external_manifest = {
        **internal_manifest,
        "archive": {
            "filename": BUNDLE_NAME,
            "sha256": archive_hash,
            "bytes": archive_path.stat().st_size,
            "members": len(zip_payloads) + len(generated_payloads),
        },
    }
    external_bytes = _canonical_json(external_manifest)
    _atomic_write(external_manifest_path, external_bytes)
    _atomic_write(checksum_path, f"{archive_hash}  {BUNDLE_NAME}\n".encode("utf-8"))
    if provenance_out is not None:
        _atomic_write(provenance_out, external_bytes)
    verify_archive(archive_path, expected_archive_sha256=archive_hash)
    return external_manifest


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    ) as temporary:
        temporary.write(data)
        temporary_path = Path(temporary.name)
    try:
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def verify_archive(path: Path, expected_archive_sha256: str | None = None) -> dict:
    """Verify archive hash, normalized metadata, and every internal checksum."""
    if not path.is_file():
        raise BundleError(f"bundle archive not found: {path}")
    observed_archive_hash = sha256_file(path)
    if expected_archive_sha256:
        _check_hash("bundle archive", observed_archive_hash, expected_archive_sha256)

    observed: dict[str, str] = {}
    manifest: dict | None = None
    checksum_text: str | None = None
    expected_prefix = f"{BUNDLE_ROOT}/"
    with tarfile.open(path, mode="r:gz") as archive:
        for member in archive:
            if not member.isfile():
                raise BundleError(f"bundle contains non-file member {member.name!r}")
            if not member.name.startswith(expected_prefix):
                raise BundleError(f"bundle member is outside root: {member.name!r}")
            if (
                member.mtime != 0
                or member.uid != 0
                or member.gid != 0
                or member.mode != 0o444
            ):
                raise BundleError(f"bundle member metadata is not normalized: {member.name}")
            relative = member.name[len(expected_prefix) :]
            relative_path = PurePosixPath(relative)
            if (
                not relative
                or relative_path.is_absolute()
                or ".." in relative_path.parts
            ):
                raise BundleError(f"bundle member has unsafe path: {member.name!r}")
            if relative in observed:
                raise BundleError(f"bundle repeats member {member.name!r}")
            extracted = archive.extractfile(member)
            if extracted is None:
                raise BundleError(f"cannot read bundle member {member.name}")
            digest = hashlib.sha256()
            chunks: list[bytes] | None = (
                []
                if relative
                in {
                    "metadata/ARTIFACT_MANIFEST.json",
                    "metadata/SHA256SUMS.txt",
                }
                else None
            )
            for block in iter(lambda: extracted.read(1024 * 1024), b""):
                digest.update(block)
                if chunks is not None:
                    chunks.append(block)
            observed[relative] = digest.hexdigest()
            if relative == "metadata/ARTIFACT_MANIFEST.json":
                try:
                    manifest = json.loads(b"".join(chunks or []).decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise BundleError("bundle artifact manifest is invalid JSON") from exc
            elif relative == "metadata/SHA256SUMS.txt":
                checksum_text = b"".join(chunks or []).decode("utf-8")

    if manifest is None or checksum_text is None:
        raise BundleError("bundle is missing its manifest or checksum table")
    expected: dict[str, str] = {}
    for line_number, line in enumerate(checksum_text.splitlines(), 1):
        parts = line.split("  ", 1)
        if len(parts) != 2 or not re.fullmatch(r"[0-9a-f]{64}", parts[0]):
            raise BundleError(f"SHA256SUMS line {line_number} is malformed")
        digest, relative = parts
        if relative in expected:
            raise BundleError(f"SHA256SUMS repeats {relative!r}")
        expected[relative] = digest
    observed_without_table = {
        relative: digest
        for relative, digest in observed.items()
        if relative != "metadata/SHA256SUMS.txt"
    }
    if observed_without_table != expected:
        missing = sorted(set(expected) - set(observed_without_table))
        extra = sorted(set(observed_without_table) - set(expected))
        bad = sorted(
            key
            for key in set(expected) & set(observed_without_table)
            if expected[key] != observed_without_table[key]
        )
        raise BundleError(
            f"bundle checksum mismatch: missing={missing}, extra={extra}, changed={bad}"
        )
    return {
        "archive_sha256": observed_archive_hash,
        "members": len(observed),
        "manifest": manifest,
    }


def _pins_from_args(args: argparse.Namespace) -> Pins:
    return Pins(
        response_archive_sha256=args.expected_response_archive_sha256,
        calibration_archive_sha256=args.expected_calibration_archive_sha256,
        matrix_sha256=args.expected_matrix_sha256,
        judge_manifest_sha256=args.expected_judge_manifest_sha256,
        verdicts_sha256=args.expected_verdicts_sha256,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="validate and freeze input artifacts")
    build.add_argument("--response-archive", type=Path, required=True)
    build.add_argument("--calibration-archive", type=Path, required=True)
    build.add_argument(
        "--fit-manifest",
        type=Path,
        default=ROOT
        / "runs/calibration/InFoBench_full_20260804/fit/calibration_mirt_manifest_strict.json",
    )
    build.add_argument(
        "--export-manifest",
        type=Path,
        default=ROOT
        / "reports/infobench_calibration_20260804/selected_bank/export_manifest.json",
    )
    build.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "runs/calibration/InFoBench_remediation_phase0",
    )
    build.add_argument("--provenance-out", type=Path)
    build.add_argument("--storage-uri")
    build.add_argument("--force", action="store_true")
    build.add_argument(
        "--expected-response-archive-sha256",
        default=DEFAULT_RESPONSE_ARCHIVE_SHA256,
    )
    build.add_argument(
        "--expected-calibration-archive-sha256",
        default=DEFAULT_CALIBRATION_ARCHIVE_SHA256,
    )
    build.add_argument("--expected-matrix-sha256", default=DEFAULT_MATRIX_SHA256)
    build.add_argument(
        "--expected-judge-manifest-sha256",
        default=DEFAULT_JUDGE_MANIFEST_SHA256,
    )
    build.add_argument("--expected-verdicts-sha256", default=DEFAULT_VERDICTS_SHA256)

    verify = subparsers.add_parser("verify", help="verify a previously frozen archive")
    verify.add_argument("--archive", type=Path, required=True)
    verify.add_argument("--expected-archive-sha256")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "verify":
            result = verify_archive(args.archive, args.expected_archive_sha256)
        else:
            result = build_bundle(
                response_archive_path=args.response_archive,
                calibration_archive_path=args.calibration_archive,
                fit_manifest_path=args.fit_manifest,
                export_manifest_path=args.export_manifest,
                output_dir=args.output_dir,
                pins=_pins_from_args(args),
                storage_uri=args.storage_uri,
                provenance_out=args.provenance_out,
                force=args.force,
            )
    except (BundleError, OSError, zipfile.BadZipFile, tarfile.TarError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.command == "verify":
        summary = {
            "archive_sha256": result["archive_sha256"],
            "members": result["members"],
            "verified": True,
        }
    else:
        summary = {
            "archive": result["archive"],
            "storage": result["storage"],
            "criterion_count_flow": result["criterion_count_flow"],
            "verified": True,
        }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
