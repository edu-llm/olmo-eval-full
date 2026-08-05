#!/usr/bin/env python3
"""Export a fitted-only CAT bank from a calibration CSV.

This exporter is deliberately stricter than ``export_calibrated_bank.py``:

* it supports any ordered modeled-skill structure;
* it never copies synthetic/placeholder item parameters into the output;
* criteria absent from the calibration CSV are explicitly excluded as unfitted;
* criteria with nonpositive or extreme active loadings are excluded (or make the
  command fail, according to an explicit policy);
* scenario ``criterion_ids`` are filtered to the exact exported criterion set and
  scenarios with no fitted criteria are dropped; and
* the source rubric/scenario files are read-only and hash-checked after export.

The calibration CSV is expected to have ``criterion_id``, ``b``, and one
``a_<modeled_skill>`` column per modeled skill.  With no ``--dimensions`` flag,
the source Q-matrix identity structure is used. Collapsed or renamed structures
use the same canonical syntax as the calibration/k-fold tools::

    --dimensions 'correctness=content+diagnosis,scaffolding=scaffolding'

The modeled Q value is the logical OR of the listed source Q dimensions.  Every
source Q dimension that is positive anywhere in the bank must be covered by the
mapping, preventing a skill from being silently discarded.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tutor_cat.skill_structure import SkillStructure, parse_dimension_spec  # noqa: E402

CALIBRATION_SOURCE = "calibrated-m2pl-fitted-only"
DEFAULT_EXTREME_A = 6.0
DEFAULT_OFF_Q_TOLERANCE = 1e-10


class ExportError(RuntimeError):
    """Raised when a safe fitted-only export cannot be produced."""


@dataclass(frozen=True)
class ExportConfig:
    calibration_csv: Path
    rubrics: Path
    scenarios: Path
    out_rubrics: Path
    out_scenarios: Path
    out_manifest: Path
    calibration_manifest: Path
    dimensions: str | None = None
    structure_name: str = "export"
    source_q_field: str = "q_mapping"
    calibration_method: str = "confirmatory-m2pl-mml-em"
    nonpositive_policy: str = "exclude"
    extreme_policy: str = "exclude"
    invalid_policy: str = "error"
    extreme_a: float = DEFAULT_EXTREME_A
    off_q_tolerance: float = DEFAULT_OFF_Q_TOLERANCE
    force: bool = False


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _read_jsonl(path: Path, label: str) -> list[dict]:
    if not path.is_file():
        raise ExportError(f"{label} file not found: {path}")
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ExportError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ExportError(f"{path}:{line_number}: expected one JSON object")
            rows.append(value)
    if not rows:
        raise ExportError(f"{label} file contains no records: {path}")
    return rows


def _unique_index(rows: Iterable[dict], key: str, label: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for position, row in enumerate(rows, 1):
        value = str(row.get(key) or "").strip()
        if not value:
            raise ExportError(f"{label} record {position} has blank {key}")
        if value in out:
            raise ExportError(f"{label} has duplicate {key} {value!r}")
        out[value] = row
    return out


def _load_calibration_csv(path: Path) -> tuple[list[str], dict[str, dict[str, str]]]:
    if not path.is_file():
        raise ExportError(f"calibration CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        if "criterion_id" not in fields or "b" not in fields:
            raise ExportError("calibration CSV must contain criterion_id and b columns")
        a_columns = [field for field in fields if field.startswith("a_") and len(field) > 2]
        if not a_columns:
            raise ExportError("calibration CSV has no a_<skill> discrimination columns")
        modeled_skills = [field[2:] for field in a_columns]
        if len(modeled_skills) != len(set(modeled_skills)):
            raise ExportError("calibration CSV has duplicate modeled discrimination columns")

        rows: dict[str, dict[str, str]] = {}
        for line_number, row in enumerate(reader, 2):
            cid = str(row.get("criterion_id") or "").strip()
            if not cid:
                raise ExportError(f"{path}:{line_number}: blank criterion_id")
            if cid in rows:
                raise ExportError(f"{path}:{line_number}: duplicate criterion_id {cid!r}")
            rows[cid] = row
    if not rows:
        raise ExportError(f"calibration CSV contains no fitted rows: {path}")
    return modeled_skills, rows


def _parse_qmap(value: object, cid: str, field: str) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ExportError(f"criterion {cid} has no object-valued {field}")
    out: dict[str, int] = {}
    for key, raw in value.items():
        if raw not in (0, 1, "0", "1", False, True):
            raise ExportError(
                f"criterion {cid} has non-binary {field}[{key!r}]={raw!r}"
            )
        out[str(key)] = int(raw)
    return out


def _float(row: dict[str, str], field: str, cid: str) -> float:
    raw = row.get(field)
    try:
        return float(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise ExportError(
            f"calibration row {cid} has non-numeric {field}={raw!r}"
        ) from exc


def _validated_latent_correlation(value: object, n_dims: int) -> list[list[float]] | None:
    """Validate and normalize a calibration manifest's latent correlation."""
    if value is None:
        return None
    try:
        matrix = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ExportError("calibration latent_correlation is not numeric") from exc
    if matrix.shape != (n_dims, n_dims):
        raise ExportError(
            "calibration latent_correlation shape does not match skills_order: "
            f"got {matrix.shape}, expected {(n_dims, n_dims)}"
        )
    if not np.isfinite(matrix).all():
        raise ExportError("calibration latent_correlation contains nonfinite values")
    if not np.allclose(matrix, matrix.T, atol=1e-6):
        raise ExportError("calibration latent_correlation is not symmetric")
    if not np.allclose(np.diag(matrix), 1.0, atol=1e-6):
        raise ExportError("calibration latent_correlation diagonal is not one")
    if float(np.linalg.eigvalsh(matrix).min()) < -1e-6:
        raise ExportError("calibration latent_correlation is not positive semidefinite")
    return matrix.tolist()


def _flag_tokens(row: dict[str, str]) -> list[str]:
    return sorted(
        {
            token.strip()
            for token in str(row.get("flags") or "").split("|")
            if token.strip()
        }
    )


def _n_persons(row: dict[str, str], cid: str) -> int | None:
    raw = str(row.get("n_persons") or "").strip()
    if not raw:
        return None
    try:
        value = int(float(raw))
    except ValueError as exc:
        raise ExportError(f"calibration row {cid} has invalid n_persons={raw!r}") from exc
    if value < 0:
        raise ExportError(f"calibration row {cid} has negative n_persons={value}")
    return value


def _serialize_jsonl(rows: Iterable[dict]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows
    ).encode("utf-8")


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_name)
        raise


def _validate_paths(config: ExportConfig) -> None:
    inputs = [config.calibration_csv, config.rubrics, config.scenarios]
    inputs.append(config.calibration_manifest)
    outputs = [config.out_rubrics, config.out_scenarios, config.out_manifest]
    resolved_inputs = {path.resolve() for path in inputs}
    resolved_outputs = [path.resolve() for path in outputs]
    if len(resolved_outputs) != len(set(resolved_outputs)):
        raise ExportError("rubric, scenario, and manifest outputs must be different files")
    collisions = [path for path in outputs if path.resolve() in resolved_inputs]
    if collisions:
        raise ExportError(f"refusing to overwrite an input file: {collisions[0]}")
    existing = [path for path in outputs if path.exists()]
    if existing and not config.force:
        raise ExportError(
            f"output already exists: {existing[0]} (choose a new path or pass --force)"
        )


def export_fitted_bank(config: ExportConfig) -> dict:
    """Build and write a safe fitted-only bank; return its manifest."""
    for policy_name, policy in (
        ("nonpositive", config.nonpositive_policy),
        ("extreme", config.extreme_policy),
        ("invalid", config.invalid_policy),
    ):
        if policy not in {"exclude", "error"}:
            raise ExportError(f"{policy_name} policy must be 'exclude' or 'error'")
    if not math.isfinite(config.extreme_a) or config.extreme_a <= 0:
        raise ExportError("extreme_a must be a finite positive number")
    if not math.isfinite(config.off_q_tolerance) or config.off_q_tolerance < 0:
        raise ExportError("off_q_tolerance must be finite and nonnegative")

    _validate_paths(config)
    input_hashes = {
        "calibration_csv": _sha256_file(config.calibration_csv),
        "rubrics": _sha256_file(config.rubrics),
        "scenarios": _sha256_file(config.scenarios),
    }
    if not config.calibration_manifest.is_file():
        raise ExportError(f"calibration manifest not found: {config.calibration_manifest}")
    input_hashes["calibration_manifest"] = _sha256_file(config.calibration_manifest)

    rubric_rows = _read_jsonl(config.rubrics, "rubric bank")
    scenario_rows = _read_jsonl(config.scenarios, "scenario bank")
    rubrics_by_id = _unique_index(rubric_rows, "criterion_id", "rubric bank")
    scenarios_by_id = _unique_index(scenario_rows, "scenario_id", "scenario bank")
    csv_skills, calibration_rows = _load_calibration_csv(config.calibration_csv)

    unknown_calibration = sorted(set(calibration_rows) - set(rubrics_by_id))
    if unknown_calibration:
        raise ExportError(
            f"calibration CSV has {len(unknown_calibration)} criterion(s) absent from the "
            f"source bank; first={unknown_calibration[:10]}"
        )

    # Validate source scenario/rubric linkage before filtering it.
    referenced: set[str] = set()
    for sid, scenario in scenarios_by_id.items():
        raw_ids = scenario.get("criterion_ids")
        if not isinstance(raw_ids, list):
            raise ExportError(f"scenario {sid} has no list-valued criterion_ids")
        ids = [str(cid) for cid in raw_ids]
        if len(ids) != len(set(ids)):
            raise ExportError(f"scenario {sid} contains duplicate criterion_ids")
        for cid in ids:
            rubric = rubrics_by_id.get(cid)
            if rubric is None:
                raise ExportError(f"scenario {sid} references unknown criterion {cid}")
            if str(rubric.get("scenario_id") or "") != sid:
                raise ExportError(
                    f"criterion {cid} belongs to {rubric.get('scenario_id')!r}, not scenario {sid}"
                )
            if cid in referenced:
                raise ExportError(f"criterion {cid} is referenced by more than one scenario")
            referenced.add(cid)
    orphaned = sorted(set(rubrics_by_id) - referenced)
    if orphaned:
        raise ExportError(
            f"source bank has {len(orphaned)} rubric(s) not referenced by scenarios; "
            f"first={orphaned[:10]}"
        )

    source_q_by_id: dict[str, dict[str, int]] = {
        cid: _parse_qmap(row.get(config.source_q_field), cid, config.source_q_field)
        for cid, row in rubrics_by_id.items()
    }
    positive_source_skills = {
        skill
        for qmap in source_q_by_id.values()
        for skill, value in qmap.items()
        if value == 1
    }
    source_skill_order: list[str] = []
    seen_source: set[str] = set()
    for source in rubric_rows:
        for skill in source_q_by_id[str(source["criterion_id"])]:
            if skill in positive_source_skills and skill not in seen_source:
                seen_source.add(skill)
                source_skill_order.append(skill)
    if not source_skill_order:
        raise ExportError(f"{config.source_q_field} has no positive source skills")
    try:
        structure = (
            parse_dimension_spec(
                config.dimensions, source_skill_order, name=config.structure_name
            )
            if config.dimensions
            else SkillStructure.identity(source_skill_order, name=config.structure_name)
        )
    except ValueError as exc:
        raise ExportError(f"invalid skill structure: {exc}") from exc
    skill_order = list(structure.labels)
    if csv_skills != skill_order:
        raise ExportError(
            "calibration CSV a_<skill> columns must exactly match the modeled skill "
            f"structure order: csv={csv_skills}, structure={skill_order}"
        )
    q_source_matrix = np.array(
        [
            [source_q_by_id[str(row["criterion_id"])].get(skill, 0) for skill in source_skill_order]
            for row in rubric_rows
        ],
        dtype=int,
    )
    try:
        q_modeled_matrix = structure.transform_q(q_source_matrix)
    except ValueError as exc:
        raise ExportError(f"could not transform source Q-matrix: {exc}") from exc
    q_modeled_by_id = {
        str(row["criterion_id"]): {
            skill: int(q_modeled_matrix[index, k]) for k, skill in enumerate(skill_order)
        }
        for index, row in enumerate(rubric_rows)
    }

    try:
        raw_manifest = json.loads(config.calibration_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExportError(f"invalid calibration manifest: {exc}") from exc
    if not isinstance(raw_manifest, dict):
        raise ExportError("calibration manifest must contain one JSON object")
    calibration_manifest_summary = {
        key: raw_manifest.get(key)
        for key in (
            "method", "skills_order", "grid_nodes_per_dim", "grid_total_nodes",
            "ridge", "estimate_latent_corr", "n_items_fit", "n_persons_fit",
            "latent_correlation",
        )
        if key in raw_manifest
    }
    manifest_skills = raw_manifest.get("skills_order")
    if not isinstance(manifest_skills, list) or list(manifest_skills) != skill_order:
        raise ExportError(
            "calibration manifest skills_order does not match the requested export "
            f"structure: manifest={manifest_skills}, export={skill_order}"
        )
    manifest_n_items = raw_manifest.get("n_items_fit")
    if manifest_n_items is None or int(manifest_n_items) != len(calibration_rows):
        raise ExportError(
            "calibration manifest n_items_fit does not match calibration CSV rows: "
            f"manifest={manifest_n_items}, csv={len(calibration_rows)}"
        )
    latent_correlation = _validated_latent_correlation(
        raw_manifest.get("latent_correlation"), len(skill_order)
    )
    if latent_correlation is not None:
        latent_correlation_source = "estimated_in_calibration_manifest"
    elif raw_manifest.get("estimate_latent_corr") is False:
        latent_correlation = np.eye(len(skill_order), dtype=float).tolist()
        latent_correlation_source = "fixed_identity_in_calibration"
    else:
        raise ExportError(
            "calibration manifest has no latent_correlation and does not declare a "
            "fixed-identity fit; refusing a CAT bank that could silently change the prior"
        )

    exported: list[dict] = []
    excluded: list[dict] = []
    fatal: list[str] = []
    for source in rubric_rows:
        cid = str(source["criterion_id"])
        fit = calibration_rows.get(cid)
        if fit is None:
            excluded.append({"criterion_id": cid, "reasons": ["unfitted"]})
            continue

        q_modeled = q_modeled_by_id[cid]
        reasons: list[str] = []
        if not any(q_modeled.values()):
            reasons.append("invalid_all_zero_modeled_q")
        if source.get("exclude_from_fit") is True:
            reasons.append("invalid_source_exclude_from_fit")

        try:
            b = _float(fit, "b", cid)
            avec = {skill: _float(fit, f"a_{skill}", cid) for skill in skill_order}
            n_persons = _n_persons(fit, cid)
        except ExportError:
            reasons.append("invalid_non_numeric_parameter")
            b = float("nan")
            avec = {skill: float("nan") for skill in skill_order}
            n_persons = None

        if not math.isfinite(b) or any(not math.isfinite(value) for value in avec.values()):
            reasons.append("invalid_nonfinite_parameter")
        for skill in skill_order:
            if (
                q_modeled[skill] == 0
                and math.isfinite(avec[skill])
                and abs(avec[skill]) > config.off_q_tolerance
            ):
                reasons.append(f"invalid_off_q_nonzero:{skill}")

        active = [avec[skill] for skill in skill_order if q_modeled[skill] == 1]
        flags = _flag_tokens(fit)
        if any(math.isfinite(value) and value <= 0 for value in active) or "nonpositive_a" in flags:
            reasons.append("nonpositive_a")
        if (
            any(math.isfinite(value) and abs(value) > config.extreme_a for value in active)
            or "extreme_a" in flags
        ):
            reasons.append("extreme_a")
        reasons = list(dict.fromkeys(reasons))

        invalid = [reason for reason in reasons if reason.startswith("invalid_")]
        nonpositive = "nonpositive_a" in reasons
        extreme = "extreme_a" in reasons
        if invalid and config.invalid_policy == "error":
            fatal.append(f"{cid}: {', '.join(invalid)}")
        if nonpositive and config.nonpositive_policy == "error":
            fatal.append(f"{cid}: nonpositive active discrimination")
        if extreme and config.extreme_policy == "error":
            fatal.append(f"{cid}: active discrimination exceeds {config.extreme_a}")
        if reasons:
            excluded.append({"criterion_id": cid, "reasons": reasons})
            continue

        record = copy.deepcopy(source)
        # Remove every old parameter-bearing field before adding calibrated values.
        record.pop("difficulty", None)
        record.pop("discrimination", None)
        record.pop("irt_params", None)
        record.pop("calibration_version", None)
        record["q_modeled"] = q_modeled
        record["discrimination"] = {
            skill: (float(avec[skill]) if q_modeled[skill] == 1 else 0.0)
            for skill in skill_order
        }
        record["difficulty"] = float(b)
        record["calibration_version"] = CALIBRATION_SOURCE
        record["irt_params"] = {
            "source": CALIBRATION_SOURCE,
            "method": config.calibration_method,
            "calibrated": True,
            "fitted": True,
            "synthetic": False,
            "skills_order": skill_order,
            "modeled_skills": skill_order,
            "skill_structure": structure.as_dict(),
            "latent_correlation": latent_correlation,
            "latent_correlation_source": latent_correlation_source,
            "n_persons": n_persons,
            "flags": flags,
            "calibration_csv_sha256": input_hashes["calibration_csv"],
            "provenance": {
                "calibration_manifest": str(config.calibration_manifest),
                "calibration_manifest_sha256": input_hashes["calibration_manifest"],
                "skills_order": skill_order,
                "latent_correlation": latent_correlation,
                "latent_correlation_source": latent_correlation_source,
            },
        }
        exported.append(record)

    if fatal:
        raise ExportError(
            f"safe export blocked by {len(fatal)} error(s); first: " + "; ".join(fatal[:10])
        )
    if not exported:
        raise ExportError("no criteria remain after fitted/parameter safety filtering")

    exported_ids = {str(row["criterion_id"]) for row in exported}
    output_scenarios: list[dict] = []
    dropped_scenarios: list[str] = []
    for source in scenario_rows:
        sid = str(source["scenario_id"])
        kept_ids = [str(cid) for cid in source["criterion_ids"] if str(cid) in exported_ids]
        if not kept_ids:
            dropped_scenarios.append(sid)
            continue
        scenario = copy.deepcopy(source)
        scenario["criterion_ids"] = kept_ids
        output_scenarios.append(scenario)

    output_referenced = {
        str(cid) for scenario in output_scenarios for cid in scenario["criterion_ids"]
    }
    if output_referenced != exported_ids:
        raise ExportError("internal error: exported rubric and scenario criterion sets differ")

    # Final invariant: every output item has finite calibrated parameters, positive
    # active loadings, zero inactive loadings, and no synthetic provenance.
    per_skill_items = {skill: 0 for skill in skill_order}
    per_skill_single_load_anchors = {skill: 0 for skill in skill_order}
    for record in exported:
        if (record.get("irt_params") or {}).get("synthetic") is not False:
            raise ExportError(f"internal error: synthetic provenance at {record['criterion_id']}")
        q = record["q_modeled"]
        a = record["discrimination"]
        if not math.isfinite(float(record["difficulty"])):
            raise ExportError(f"internal error: nonfinite b at {record['criterion_id']}")
        active_count = sum(int(q[skill]) for skill in skill_order)
        for skill in skill_order:
            value = float(a[skill])
            if q[skill] == 1:
                if not (math.isfinite(value) and 0 < value <= config.extreme_a):
                    raise ExportError(
                        f"internal error: unsafe active loading at {record['criterion_id']}"
                    )
                per_skill_items[skill] += 1
                if active_count == 1:
                    per_skill_single_load_anchors[skill] += 1
            elif abs(value) > config.off_q_tolerance:
                raise ExportError(
                    f"internal error: nonzero inactive loading at {record['criterion_id']}"
                )

    exclusion_reason_counts = Counter(
        reason for item in excluded for reason in item["reasons"]
    )
    rubric_bytes = _serialize_jsonl(exported)
    scenario_bytes = _serialize_jsonl(output_scenarios)
    output_hashes = {
        "rubrics": _sha256_bytes(rubric_bytes),
        "scenarios": _sha256_bytes(scenario_bytes),
    }
    manifest = {
        "generated_at": _utcnow(),
        "schema_version": "fitted-bank-export-v1",
        "purpose": (
            "Fitted-only calibrated CAT bank. No synthetic or placeholder item "
            "parameters are present."
        ),
        "skills_order": skill_order,
        "skill_structure": structure.as_dict(),
        "latent_correlation": latent_correlation,
        "latent_correlation_source": latent_correlation_source,
        "source_q_field": config.source_q_field,
        "policies": {
            "unfitted": "exclude",
            "nonpositive": config.nonpositive_policy,
            "extreme": config.extreme_policy,
            "invalid": config.invalid_policy,
            "extreme_a_threshold": config.extreme_a,
            "off_q_tolerance": config.off_q_tolerance,
            "empty_scenarios": "drop",
        },
        "counts": {
            "source_criteria": len(rubric_rows),
            "calibration_rows": len(calibration_rows),
            "exported_criteria": len(exported),
            "excluded_criteria": len(excluded),
            "source_scenarios": len(scenario_rows),
            "exported_scenarios": len(output_scenarios),
            "dropped_empty_scenarios": len(dropped_scenarios),
        },
        "per_skill_items": per_skill_items,
        "per_skill_single_load_anchors": per_skill_single_load_anchors,
        "exclusion_reason_counts": dict(sorted(exclusion_reason_counts.items())),
        "excluded_criteria": excluded,
        "dropped_scenario_ids": dropped_scenarios,
        "calibration_manifest_summary": calibration_manifest_summary,
        "inputs": {
            "calibration_csv": str(config.calibration_csv),
            "rubrics": str(config.rubrics),
            "scenarios": str(config.scenarios),
            "calibration_manifest": str(config.calibration_manifest),
            "sha256": input_hashes,
        },
        "outputs": {
            "rubrics": str(config.out_rubrics),
            "scenarios": str(config.out_scenarios),
            "manifest": str(config.out_manifest),
            "sha256": output_hashes,
        },
        "invariants": {
            "source_files_unchanged": True,
            "fitted_only": True,
            "contains_synthetic_parameters": False,
            "scenario_criterion_ids_exactly_match_exported_rubrics": True,
            "all_active_loadings_positive_and_within_threshold": True,
            "all_inactive_loadings_zero": True,
        },
        "provenance": {
            "script": "scripts/export_fitted_bank.py",
            "calibration_method": config.calibration_method,
        },
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")

    _atomic_write(config.out_rubrics, rubric_bytes)
    _atomic_write(config.out_scenarios, scenario_bytes)
    _atomic_write(config.out_manifest, manifest_bytes)

    # Inputs must remain byte-identical. This catches accidental path aliasing or
    # future refactors that mutate the source bank while exporting.
    post_hashes = {
        "calibration_csv": _sha256_file(config.calibration_csv),
        "rubrics": _sha256_file(config.rubrics),
        "scenarios": _sha256_file(config.scenarios),
    }
    post_hashes["calibration_manifest"] = _sha256_file(config.calibration_manifest)
    if post_hashes != input_hashes:
        raise ExportError("an input file changed during export; outputs must not be trusted")
    return manifest


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--calibration-csv", type=Path, required=True)
    parser.add_argument("--rubrics", type=Path, required=True)
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument("--out-rubrics", type=Path, required=True)
    parser.add_argument("--out-scenarios", type=Path, required=True)
    parser.add_argument("--out-manifest", type=Path, required=True)
    parser.add_argument(
        "--dimensions",
        default=None,
        metavar="LABEL=SOURCE[+SOURCE...],...",
        help=(
            "modeled skill partition in canonical syntax. Default: identity over "
            "the positive source Q skills."
        ),
    )
    parser.add_argument("--structure-name", default="export")
    parser.add_argument("--source-q-field", default="q_mapping")
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument(
        "--calibration-method", default="confirmatory-m2pl-mml-em"
    )
    parser.add_argument(
        "--nonpositive-policy", choices=("exclude", "error"), default="exclude"
    )
    parser.add_argument(
        "--extreme-policy", choices=("exclude", "error"), default="exclude"
    )
    parser.add_argument(
        "--invalid-policy", choices=("exclude", "error"), default="error"
    )
    parser.add_argument("--extreme-a", type=float, default=DEFAULT_EXTREME_A)
    parser.add_argument(
        "--off-q-tolerance", type=float, default=DEFAULT_OFF_Q_TOLERANCE
    )
    parser.add_argument(
        "--force", action="store_true", help="replace existing output files"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    config = ExportConfig(
        calibration_csv=args.calibration_csv,
        rubrics=args.rubrics,
        scenarios=args.scenarios,
        out_rubrics=args.out_rubrics,
        out_scenarios=args.out_scenarios,
        out_manifest=args.out_manifest,
        dimensions=args.dimensions,
        structure_name=args.structure_name,
        source_q_field=args.source_q_field,
        calibration_manifest=args.calibration_manifest,
        calibration_method=args.calibration_method,
        nonpositive_policy=args.nonpositive_policy,
        extreme_policy=args.extreme_policy,
        invalid_policy=args.invalid_policy,
        extreme_a=args.extreme_a,
        off_q_tolerance=args.off_q_tolerance,
        force=args.force,
    )
    try:
        manifest = export_fitted_bank(config)
    except ExportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    counts = manifest["counts"]
    print(
        f"exported {counts['exported_criteria']} fitted criteria across "
        f"{counts['exported_scenarios']} scenarios"
    )
    print(f"excluded {counts['excluded_criteria']} criteria: {manifest['exclusion_reason_counts']}")
    print(f"rubrics  -> {config.out_rubrics}")
    print(f"scenarios-> {config.out_scenarios}")
    print(f"manifest -> {config.out_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
