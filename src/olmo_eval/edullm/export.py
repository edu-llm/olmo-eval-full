"""Safe, fitted-only EduLLM bank export.

This module is a reusable library port of
``origin/frq/infobench:eduLLM-Evals/scripts/export_fitted_bank.py`` at commit
``b4ea2e8e4b1da75067dd5e38ea5e83270776a07f`` (Git blob
``fd6a7d47ca2036bec73594be763cdf069b71063e``).

Unlike the source CLI, the primary API consumes an in-memory
:class:`~olmo_eval.edullm.calibration.CalibrationFitResult`.  It never reads item
parameters from the source rubric records, so synthetic or placeholder values cannot
leak into a fitted bank.  File output is a separate, explicitly requested operation.

The source exporter accepted positive-*semidefinite* latent correlations even though
the runtime requires a positive-*definite* prior.  This port resolves that defect at
the boundary: an export is built only when its correlation will be accepted by
``bank.load_fitted_bank``.  A one-dimensional identity matrix is positive definite and
remains valid.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, cast

import numpy as np

from .bank import FITTED_BANK_SCHEMA_VERSION
from .calibration import CalibrationFitResult
from .skill_structure import SkillStructure

SOURCE_BRANCH = "origin/frq/infobench"
SOURCE_COMMIT = "b4ea2e8e4b1da75067dd5e38ea5e83270776a07f"
SOURCE_BLOB = "fd6a7d47ca2036bec73594be763cdf069b71063e"
RUNTIME_MAX_OFF_Q_TOLERANCE = 1e-10

PolicyAction = Literal["exclude", "error"]


class ExportError(RuntimeError):
    """Raised when a fitted bank cannot be exported without weakening its contract."""


@dataclass(frozen=True, slots=True)
class ExportPolicy:
    """Explicit safety choices for fitted-item export.

    No scientific thresholds are silently selected.  ``unfitted`` applies to source
    criteria absent from ``item_ids``.  ``invalid`` covers non-finite parameters,
    nonzero inactive loadings, all-zero modeled Q rows, and source records explicitly
    marked ``exclude_from_fit``.
    """

    unfitted: PolicyAction
    nonpositive: PolicyAction
    extreme: PolicyAction
    invalid: PolicyAction
    extreme_a_threshold: float
    off_q_tolerance: float
    require_converged: bool

    def __post_init__(self) -> None:
        for name in ("unfitted", "nonpositive", "extreme", "invalid"):
            if getattr(self, name) not in ("exclude", "error"):
                raise ExportError(f"{name} policy must be 'exclude' or 'error'")
        if not math.isfinite(self.extreme_a_threshold) or self.extreme_a_threshold <= 0:
            raise ExportError("extreme_a_threshold must be finite and strictly positive")
        if (
            not math.isfinite(self.off_q_tolerance)
            or self.off_q_tolerance < 0
            or self.off_q_tolerance > RUNTIME_MAX_OFF_Q_TOLERANCE
        ):
            raise ExportError(
                "off_q_tolerance must be finite, nonnegative, and no greater than "
                f"the runtime limit {RUNTIME_MAX_OFF_Q_TOLERANCE}"
            )
        if not isinstance(self.require_converged, bool):
            raise ExportError("require_converged must be a boolean")

    def as_dict(self) -> dict[str, Any]:
        return {
            "unfitted": self.unfitted,
            "nonpositive": self.nonpositive,
            "extreme": self.extreme,
            "invalid": self.invalid,
            "extreme_a_threshold": float(self.extreme_a_threshold),
            "off_q_tolerance": float(self.off_q_tolerance),
            "empty_scenarios": "drop",
            "require_converged": self.require_converged,
        }


@dataclass(frozen=True, slots=True)
class SourceBankProvenance:
    """Hashes and identifiers for the source records used by an in-memory export.

    The required hashes preserve the identities of the original bank artifacts even
    when mappings were loaded by another layer.  ``additional_sha256`` can hold, for
    example, the response matrix or calibration-input hash.  Local source paths may be
    supplied solely as a write guard; the writer will never replace them.
    """

    rubrics_identifier: str
    rubrics_sha256: str
    scenarios_identifier: str
    scenarios_sha256: str
    additional_sha256: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    protected_input_paths: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        if not self.rubrics_identifier.strip() or not self.scenarios_identifier.strip():
            raise ExportError("source rubric and scenario identifiers must be non-empty")
        _validate_sha256(self.rubrics_sha256, "rubrics_sha256")
        _validate_sha256(self.scenarios_sha256, "scenarios_sha256")
        extra = {str(name): str(value).lower() for name, value in self.additional_sha256.items()}
        if any(not name.strip() for name in extra):
            raise ExportError("additional source-hash names must be non-empty")
        for name, value in extra.items():
            _validate_sha256(value, f"additional_sha256[{name!r}]")
        metadata = _plain_json(self.metadata, "source provenance metadata")
        inferred_paths = tuple(
            path
            for path in (
                _local_identifier_path(self.rubrics_identifier),
                _local_identifier_path(self.scenarios_identifier),
            )
            if path is not None
        )
        paths = inferred_paths + tuple(Path(path) for path in self.protected_input_paths)
        if len({_resolved(path) for path in paths}) != len(paths):
            raise ExportError("protected_input_paths contains duplicates")
        object.__setattr__(self, "rubrics_sha256", self.rubrics_sha256.strip().lower())
        object.__setattr__(self, "scenarios_sha256", self.scenarios_sha256.strip().lower())
        object.__setattr__(self, "additional_sha256", MappingProxyType(extra))
        object.__setattr__(self, "metadata", _freeze_json(metadata))
        object.__setattr__(self, "protected_input_paths", paths)

    def as_dict(self) -> dict[str, Any]:
        return {
            "rubrics": {
                "identifier": self.rubrics_identifier,
                "sha256": self.rubrics_sha256,
            },
            "scenarios": {
                "identifier": self.scenarios_identifier,
                "sha256": self.scenarios_sha256,
            },
            "additional_sha256": dict(self.additional_sha256),
            "metadata": _thaw_json(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class FittedBankExport:
    """Immutable in-memory fitted-bank artifacts ready for an explicit write."""

    rubric_records: tuple[Mapping[str, Any], ...]
    scenario_records: tuple[Mapping[str, Any], ...]
    manifest: Mapping[str, Any]
    rubrics_jsonl: bytes
    scenarios_jsonl: bytes
    protected_input_paths: tuple[Path, ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class WrittenFittedBank:
    """Paths and finalized manifest returned by the atomic writer."""

    rubrics_path: Path
    scenarios_path: Path
    manifest_path: Path
    manifest: Mapping[str, Any]


def build_fitted_bank_export(
    source_rubrics: Sequence[Mapping[str, Any]],
    source_scenarios: Sequence[Mapping[str, Any]],
    fit: CalibrationFitResult,
    item_ids: Sequence[str],
    skill_structure: SkillStructure,
    *,
    source_provenance: SourceBankProvenance,
    calibration_version: str,
    calibration_method: str,
    policy: ExportPolicy,
    source_q_field: str,
) -> FittedBankExport:
    """Build a fitted-only bank without changing or writing any source input.

    ``item_ids`` defines the row order of ``fit.loadings`` and ``fit.difficulties``.
    Output records retain source-bank order, with every old parameter-bearing field
    removed before the fitted values are installed.
    """

    version = calibration_version.strip()
    method = calibration_method.strip()
    q_field = source_q_field.strip()
    if not version or not method or not q_field:
        raise ExportError(
            "calibration_version, calibration_method, and source_q_field must be non-empty"
        )
    if policy.require_converged and not fit.converged:
        raise ExportError("calibration did not converge under a require_converged policy")

    rubric_rows = _copy_records(source_rubrics, "source rubric bank")
    scenario_rows = _copy_records(source_scenarios, "source scenario bank")
    rubrics_by_id = _unique_index(rubric_rows, "criterion_id", "source rubric bank")
    scenarios_by_id = _unique_index(scenario_rows, "scenario_id", "source scenario bank")
    _validate_source_links(rubrics_by_id, scenarios_by_id)

    fit_ids = tuple(str(item_id).strip() for item_id in item_ids)
    if not fit_ids or any(not item_id for item_id in fit_ids):
        raise ExportError("item_ids must contain non-empty criterion IDs")
    if len(fit_ids) != len(set(fit_ids)):
        raise ExportError("item_ids contains duplicate criterion IDs")
    unknown_fit_ids = sorted(set(fit_ids) - set(rubrics_by_id))
    if unknown_fit_ids:
        raise ExportError(
            f"fit contains {len(unknown_fit_ids)} criterion(s) absent from the source bank; "
            f"first={unknown_fit_ids[:10]}"
        )

    skill_order = tuple(skill_structure.labels)
    source_skill_order = tuple(skill_structure.source_skills)
    _validate_fit(fit, fit_ids, skill_structure)
    correlation = _positive_definite_correlation(fit.latent_correlation, len(skill_order))
    fit_sha = _fit_sha256(fit, fit_ids)

    source_q = np.empty((len(rubric_rows), len(source_skill_order)), dtype=np.int8)
    for row_index, row in enumerate(rubric_rows):
        cid = str(row["criterion_id"])
        qmap = _parse_qmap(row.get(q_field), cid, q_field, source_skill_order)
        source_q[row_index] = [qmap[skill] for skill in source_skill_order]
    try:
        modeled_q = skill_structure.transform_q(source_q)
    except ValueError as exc:
        raise ExportError(f"could not transform source Q-matrix: {exc}") from exc

    q_by_id = {str(row["criterion_id"]): modeled_q[index] for index, row in enumerate(rubric_rows)}
    fit_index = {cid: index for index, cid in enumerate(fit_ids)}
    exported: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    fatal: list[str] = []

    for source in rubric_rows:
        cid = str(source["criterion_id"])
        row_index = fit_index.get(cid)
        if row_index is None:
            _record_exclusion(
                cid,
                ["unfitted"],
                policy.unfitted,
                excluded=excluded,
                fatal=fatal,
            )
            continue

        q = q_by_id[cid]
        a = np.asarray(fit.loadings[row_index], dtype=float)
        b = float(fit.difficulties[row_index])
        reasons: list[str] = []
        if not np.any(q):
            reasons.append("invalid_all_zero_modeled_q")
        if source.get("exclude_from_fit") is True:
            reasons.append("invalid_source_exclude_from_fit")
        if not math.isfinite(b) or not np.isfinite(a).all():
            reasons.append("invalid_nonfinite_parameter")
        if np.any(np.isfinite(a[q == 0]) & (np.abs(a[q == 0]) > policy.off_q_tolerance)):
            reasons.append("invalid_off_q_nonzero")
        active = a[q == 1]
        if np.any(np.isfinite(active) & (active <= 0)):
            reasons.append("nonpositive_a")
        if np.any(np.isfinite(active) & (np.abs(active) > policy.extreme_a_threshold)):
            reasons.append("extreme_a")
        reasons = list(dict.fromkeys(reasons))

        actions: list[PolicyAction] = []
        if any(reason.startswith("invalid_") for reason in reasons):
            actions.append(policy.invalid)
        if "nonpositive_a" in reasons:
            actions.append(policy.nonpositive)
        if "extreme_a" in reasons:
            actions.append(policy.extreme)
        if reasons:
            action: PolicyAction = "error" if "error" in actions else "exclude"
            _record_exclusion(
                cid,
                reasons,
                action,
                excluded=excluded,
                fatal=fatal,
            )
            continue

        qmap = {skill: int(q[index]) for index, skill in enumerate(skill_order)}
        avec = {
            skill: float(a[index]) if q[index] == 1 else 0.0
            for index, skill in enumerate(skill_order)
        }
        record = copy.deepcopy(source)
        for stale_field in (
            "difficulty",
            "discrimination",
            "irt_params",
            "calibration_version",
            "q_modeled",
        ):
            record.pop(stale_field, None)
        record["scoring_type"] = "binary"
        record["q_modeled"] = qmap
        record["discrimination"] = avec
        record["difficulty"] = b
        record["calibration_version"] = version
        record["irt_params"] = {
            "source": version,
            "method": method,
            "calibrated": True,
            "fitted": True,
            "synthetic": False,
            "skills_order": list(skill_order),
            "modeled_skills": list(skill_order),
            "skill_structure": skill_structure.as_dict(),
            "latent_correlation": correlation.tolist(),
            "latent_correlation_source": "calibration_fit_result",
            "n_persons": fit.provenance.n_persons_fitted,
            "calibration_fit_sha256": fit_sha,
            "provenance": {
                "calibration_source_branch": fit.provenance.source_branch,
                "calibration_source_commit": fit.provenance.source_commit,
                "calibration_source_blob": fit.provenance.source_blob,
                "calibration_specification_sha256": fit.provenance.specification_sha256,
                "skills_order": list(skill_order),
                "latent_correlation": correlation.tolist(),
                "source_bank_sha256": {
                    "rubrics": source_provenance.rubrics_sha256,
                    "scenarios": source_provenance.scenarios_sha256,
                },
            },
        }
        exported.append(record)

    if fatal:
        raise ExportError(
            f"safe export blocked by {len(fatal)} policy error(s); first: " + "; ".join(fatal[:10])
        )
    if not exported:
        raise ExportError("no criteria remain after fitted-parameter safety filtering")

    exported_ids = {str(record["criterion_id"]) for record in exported}
    output_scenarios: list[dict[str, Any]] = []
    dropped_scenarios: list[str] = []
    for source in scenario_rows:
        sid = str(source["scenario_id"])
        kept = [str(cid) for cid in source["criterion_ids"] if str(cid) in exported_ids]
        if not kept:
            dropped_scenarios.append(sid)
            continue
        scenario = copy.deepcopy(source)
        scenario["criterion_ids"] = kept
        output_scenarios.append(scenario)

    referenced = {str(cid) for scenario in output_scenarios for cid in scenario["criterion_ids"]}
    if referenced != exported_ids:
        raise ExportError("internal error: exported rubric and scenario criterion sets differ")

    per_skill_items, single_load_anchors = _validate_exported_records(
        exported,
        skill_order,
        policy,
    )
    rubric_bytes = _serialize_jsonl(exported)
    scenario_bytes = _serialize_jsonl(output_scenarios)
    output_hashes = {
        "rubrics": _sha256_bytes(rubric_bytes),
        "scenarios": _sha256_bytes(scenario_bytes),
    }
    exclusion_counts = Counter(reason for item in excluded for reason in item["reasons"])
    source_record_hashes = {
        "rubrics_canonical_jsonl": _sha256_bytes(_serialize_jsonl(rubric_rows)),
        "scenarios_canonical_jsonl": _sha256_bytes(_serialize_jsonl(scenario_rows)),
    }
    manifest: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "schema_version": FITTED_BANK_SCHEMA_VERSION,
        "purpose": (
            "Fitted-only calibrated CAT bank. No synthetic or placeholder item "
            "parameters are present."
        ),
        "calibration_version": version,
        "calibration_method": method,
        "skills_order": list(skill_order),
        "skill_structure": skill_structure.as_dict(),
        "latent_correlation": correlation.tolist(),
        "latent_correlation_source": "calibration_fit_result",
        "source_q_field": q_field,
        "policies": policy.as_dict(),
        "counts": {
            "source_criteria": len(rubric_rows),
            "calibration_rows": len(fit_ids),
            "exported_criteria": len(exported),
            "excluded_criteria": len(excluded),
            "source_scenarios": len(scenario_rows),
            "exported_scenarios": len(output_scenarios),
            "dropped_empty_scenarios": len(dropped_scenarios),
        },
        "per_skill_items": per_skill_items,
        "per_skill_single_load_anchors": single_load_anchors,
        "exclusion_reason_counts": dict(sorted(exclusion_counts.items())),
        "excluded_criteria": excluded,
        "dropped_scenario_ids": dropped_scenarios,
        "calibration_fit": {
            "sha256": fit_sha,
            "converged": fit.converged,
            "termination_reason": fit.termination_reason,
            "returned_iterate_status": fit.returned_iterate_status,
            "iterations": fit.iterations,
            "marginal_log_likelihood": fit.marginal_log_likelihood,
            "penalized_objective": fit.penalized_objective,
            "n_parameters": fit.n_parameters,
            "n_free_loadings": fit.n_free_loadings,
            "n_fixed_loadings": fit.n_fixed_loadings,
            "item_ids": list(fit_ids),
            "provenance": fit.provenance.as_dict(),
        },
        "inputs": {
            "source_bank": source_provenance.as_dict(),
            "canonical_record_sha256": source_record_hashes,
        },
        "outputs": {
            "rubrics": "<in-memory>",
            "scenarios": "<in-memory>",
            "manifest": "<in-memory>",
            "sha256": output_hashes,
        },
        "invariants": {
            "source_files_unchanged": True,
            "source_records_unchanged": True,
            "fitted_only": True,
            "contains_synthetic_parameters": False,
            "scenario_criterion_ids_exactly_match_exported_rubrics": True,
            "all_active_loadings_positive_and_within_threshold": True,
            "all_inactive_loadings_zero": True,
            "latent_correlation_positive_definite": True,
        },
        "provenance": {
            "module": "olmo_eval.edullm.export",
            "source_branch": SOURCE_BRANCH,
            "source_commit": SOURCE_COMMIT,
            "source_blob": SOURCE_BLOB,
        },
    }
    # Validate that every emitted value is strict JSON before returning the artifacts.
    _serialize_manifest(manifest)
    return FittedBankExport(
        rubric_records=tuple(_freeze_json(record) for record in exported),
        scenario_records=tuple(_freeze_json(record) for record in output_scenarios),
        manifest=_freeze_json(manifest),
        rubrics_jsonl=rubric_bytes,
        scenarios_jsonl=scenario_bytes,
        protected_input_paths=source_provenance.protected_input_paths,
    )


def write_fitted_bank_export(
    export: FittedBankExport,
    rubrics_path: str | Path,
    scenarios_path: str | Path,
    manifest_path: str | Path,
    *,
    overwrite: bool = False,
    protected_input_paths: Sequence[str | Path] = (),
) -> WrittenFittedBank:
    """Write all artifacts via same-directory temporary files, manifest last.

    Each replacement is atomic.  The manifest is installed last, so an interrupted
    multi-file replacement fails closed through its content hashes.  Source paths
    carried by ``SourceBankProvenance`` and additional paths passed here are never
    valid output targets.
    """

    outputs = (Path(rubrics_path), Path(scenarios_path), Path(manifest_path))
    resolved_outputs = tuple(_resolved(path) for path in outputs)
    if len(set(resolved_outputs)) != len(outputs):
        raise ExportError("rubric, scenario, and manifest outputs must be different files")
    protected = tuple(export.protected_input_paths) + tuple(
        Path(path) for path in protected_input_paths
    )
    collisions = set(resolved_outputs) & {_resolved(path) for path in protected}
    if collisions:
        raise ExportError(f"refusing to overwrite a source input: {sorted(collisions)[0]}")
    if not overwrite:
        existing = [path for path in outputs if path.exists()]
        if existing:
            raise ExportError(f"output already exists: {existing[0]}")

    manifest = _thaw_json(export.manifest)
    manifest["outputs"].update(
        {
            "rubrics": str(outputs[0]),
            "scenarios": str(outputs[1]),
            "manifest": str(outputs[2]),
        }
    )
    manifest["outputs"]["sha256"] = {
        "rubrics": _sha256_bytes(export.rubrics_jsonl),
        "scenarios": _sha256_bytes(export.scenarios_jsonl),
    }
    manifest_bytes = _serialize_manifest(manifest)
    payloads = (export.rubrics_jsonl, export.scenarios_jsonl, manifest_bytes)

    staged: list[tuple[str, Path]] = []
    try:
        for path, payload in zip(outputs, payloads, strict=True):
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
            except Exception:
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(temporary)
                raise
            staged.append((temporary, path))
        # The manifest is deliberately last: stale/mixed data cannot validate against it.
        for temporary, path in staged:
            os.replace(temporary, path)
    finally:
        for temporary, _ in staged:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temporary)

    return WrittenFittedBank(
        rubrics_path=outputs[0],
        scenarios_path=outputs[1],
        manifest_path=outputs[2],
        manifest=_freeze_json(manifest),
    )


def _copy_records(records: Sequence[Mapping[str, Any]], label: str) -> list[dict[str, Any]]:
    if not records:
        raise ExportError(f"{label} contains no records")
    copied: list[dict[str, Any]] = []
    for position, record in enumerate(records, 1):
        if not isinstance(record, Mapping):
            raise ExportError(f"{label} record {position} must be an object")
        plain = _plain_json(record, f"{label} record {position}")
        if not isinstance(plain, dict):  # pragma: no cover - guarded by Mapping above
            raise ExportError(f"{label} record {position} must be an object")
        copied.append(plain)
    return copied


def _unique_index(
    records: Sequence[dict[str, Any]], key: str, label: str
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for position, record in enumerate(records, 1):
        value = str(record.get(key) or "").strip()
        if not value:
            raise ExportError(f"{label} record {position} has blank {key}")
        if value in indexed:
            raise ExportError(f"{label} has duplicate {key} {value!r}")
        record[key] = value
        indexed[value] = record
    return indexed


def _validate_source_links(
    rubrics: Mapping[str, dict[str, Any]],
    scenarios: Mapping[str, dict[str, Any]],
) -> None:
    referenced: set[str] = set()
    for sid, scenario in scenarios.items():
        raw_ids = scenario.get("criterion_ids")
        if not isinstance(raw_ids, list):
            raise ExportError(f"scenario {sid} has no list-valued criterion_ids")
        ids = [str(cid).strip() for cid in raw_ids]
        if any(not cid for cid in ids) or len(ids) != len(set(ids)):
            raise ExportError(f"scenario {sid} contains blank or duplicate criterion_ids")
        scenario["criterion_ids"] = ids
        for cid in ids:
            rubric = rubrics.get(cid)
            if rubric is None:
                raise ExportError(f"scenario {sid} references unknown criterion {cid}")
            if str(rubric.get("scenario_id") or "").strip() != sid:
                raise ExportError(
                    f"criterion {cid} belongs to {rubric.get('scenario_id')!r}, not {sid!r}"
                )
            if not str(rubric.get("criterion") or "").strip():
                raise ExportError(f"criterion {cid} has blank criterion text")
            scoring_type = str(rubric.get("scoring_type") or "binary")
            if scoring_type != "binary":
                raise ExportError(f"criterion {cid} has unsupported scoring_type {scoring_type!r}")
            if cid in referenced:
                raise ExportError(f"criterion {cid} is referenced by more than one scenario")
            referenced.add(cid)
    orphaned = sorted(set(rubrics) - referenced)
    if orphaned:
        raise ExportError(
            f"source bank has {len(orphaned)} unreferenced rubric(s); first={orphaned[:10]}"
        )


def _parse_qmap(
    value: object,
    cid: str,
    field_name: str,
    source_skills: tuple[str, ...],
) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise ExportError(f"criterion {cid} has no object-valued {field_name}")
    q_mapping = cast(Mapping[str, Any], value)
    if set(q_mapping) != set(source_skills):
        raise ExportError(
            f"criterion {cid} {field_name} keys must exactly equal the skill structure's "
            "source_skills"
        )
    parsed: dict[str, int] = {}
    for skill in source_skills:
        raw = q_mapping[skill]
        if raw not in (0, 1, "0", "1", False, True):
            raise ExportError(f"criterion {cid} has non-binary {field_name}[{skill!r}]={raw!r}")
        parsed[skill] = int(raw)
    if not any(parsed.values()):
        raise ExportError(f"criterion {cid} has an all-zero source Q row")
    return parsed


def _validate_fit(
    fit: CalibrationFitResult,
    fit_ids: tuple[str, ...],
    skill_structure: SkillStructure,
) -> None:
    n_items = len(fit_ids)
    n_dims = skill_structure.n_dims
    if np.asarray(fit.loadings).shape != (n_items, n_dims):
        raise ExportError(
            f"fit.loadings shape {np.asarray(fit.loadings).shape} != {(n_items, n_dims)}"
        )
    if np.asarray(fit.difficulties).shape != (n_items,):
        raise ExportError(
            f"fit.difficulties shape {np.asarray(fit.difficulties).shape} != {(n_items,)}"
        )
    if fit.provenance.n_items != n_items:
        raise ExportError(
            f"calibration provenance n_items={fit.provenance.n_items} != item_ids={n_items}"
        )
    declared_structure = _plain_json(
        fit.provenance.skill_structure,
        "calibration provenance skill_structure",
    )
    if declared_structure != skill_structure.as_dict():
        raise ExportError("calibration provenance skill structure does not match export structure")


def _positive_definite_correlation(value: object, n_dims: int) -> np.ndarray:
    try:
        correlation = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ExportError("calibration latent_correlation is not numeric") from exc
    if correlation.shape != (n_dims, n_dims):
        raise ExportError(
            f"calibration latent_correlation shape {correlation.shape} != {(n_dims, n_dims)}"
        )
    if not np.isfinite(correlation).all():
        raise ExportError("calibration latent_correlation must be finite")
    if not np.allclose(correlation, correlation.T, atol=1e-9, rtol=0.0):
        raise ExportError("calibration latent_correlation must be symmetric")
    if not np.allclose(np.diag(correlation), 1.0, atol=1e-6, rtol=0.0):
        raise ExportError("calibration latent_correlation must have unit diagonal")
    if float(np.linalg.eigvalsh(correlation).min()) <= 1e-10:
        raise ExportError("calibration latent_correlation must be positive definite")
    return correlation.copy()


def _record_exclusion(
    cid: str,
    reasons: list[str],
    action: PolicyAction,
    *,
    excluded: list[dict[str, Any]],
    fatal: list[str],
) -> None:
    excluded.append({"criterion_id": cid, "reasons": reasons})
    if action == "error":
        fatal.append(f"{cid}: {', '.join(reasons)}")


def _validate_exported_records(
    records: Sequence[Mapping[str, Any]],
    skill_order: tuple[str, ...],
    policy: ExportPolicy,
) -> tuple[dict[str, int], dict[str, int]]:
    per_skill = {skill: 0 for skill in skill_order}
    anchors = {skill: 0 for skill in skill_order}
    for record in records:
        cid = str(record["criterion_id"])
        irt = record.get("irt_params")
        if not isinstance(irt, Mapping) or irt.get("synthetic") is not False:
            raise ExportError(f"internal error: synthetic or missing provenance at {cid}")
        q = record["q_modeled"]
        a = record["discrimination"]
        if not isinstance(q, Mapping) or not isinstance(a, Mapping):
            raise ExportError(f"internal error: missing modeled parameters at {cid}")
        if set(q) != set(skill_order) or set(a) != set(skill_order):
            raise ExportError(f"internal error: parameter axes differ at {cid}")
        if not math.isfinite(float(record["difficulty"])):
            raise ExportError(f"internal error: nonfinite difficulty at {cid}")
        active_count = sum(int(q[skill]) for skill in skill_order)
        if active_count == 0:
            raise ExportError(f"internal error: all-zero modeled Q at {cid}")
        for skill in skill_order:
            value = float(a[skill])
            if int(q[skill]) == 1:
                if not (math.isfinite(value) and 0 < value <= policy.extreme_a_threshold):
                    raise ExportError(f"internal error: unsafe active loading at {cid}")
                per_skill[skill] += 1
                if active_count == 1:
                    anchors[skill] += 1
            elif value != 0.0:
                raise ExportError(f"internal error: inactive loading was not normalized at {cid}")
    return per_skill, anchors


def _fit_sha256(fit: CalibrationFitResult, item_ids: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(item_ids, separators=(",", ":")).encode())
    for name in ("loadings", "difficulties", "latent_correlation"):
        array = np.ascontiguousarray(np.asarray(getattr(fit, name), dtype="<f8"))
        digest.update(name.encode())
        digest.update(json.dumps(array.shape, separators=(",", ":")).encode())
        digest.update(array.tobytes())
    digest.update(
        json.dumps(
            fit.provenance.as_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    )
    return digest.hexdigest()


def _plain_json(value: Any, label: str) -> Any:
    try:
        return json.loads(json.dumps(_thaw_json(value), ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ExportError(f"{label} must contain only finite JSON values: {exc}") from exc


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_thaw_json(item) for item in value]
    return value


def _serialize_jsonl(records: Sequence[Mapping[str, Any]]) -> bytes:
    try:
        return "".join(
            json.dumps(
                record,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
            for record in records
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ExportError(f"records cannot be encoded as strict JSONL: {exc}") from exc


def _serialize_manifest(manifest: Mapping[str, Any]) -> bytes:
    try:
        return (json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode(
            "utf-8"
        )
    except (TypeError, ValueError) as exc:
        raise ExportError(f"manifest cannot be encoded as strict JSON: {exc}") from exc


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _validate_sha256(value: str, label: str) -> None:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ExportError(f"{label} must be a 64-character hexadecimal SHA-256")


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _local_identifier_path(identifier: str) -> Path | None:
    """Treat plain paths and file URLs as protected; leave other URIs untouched."""

    normalized = identifier.strip()
    if normalized.startswith("file://"):
        return Path(normalized.removeprefix("file://"))
    if "://" in normalized:
        return None
    return Path(normalized)
