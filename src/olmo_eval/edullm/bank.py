"""Strict, dimension-generic runtime contract for calibrated EduLLM banks.

This module is a library extraction of the fitted-bank and runtime-schema behavior
developed on ``origin/frq/infobench``.  Its primary sources are
``scripts/scenario_cat_lib.py`` (blob ``20bb1f61``), ``tutor_cat/schemas.py``
(``b6b8ed1f``), ``tutor_cat/dataio.py`` (``992a5aa4``), and the fitted-only exporter
(``fd6a7d47``).

The runtime intentionally fails closed.  It never invents item parameters, an identity
latent correlation, or a latent-skill order, and it never returns the valid subset of a
partly invalid bank.  A supplied export manifest is treated as an integrity contract:
its schema, counts, invariants, skill order, correlation, and file hashes must all match
the files being loaded.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import numpy as np

FITTED_BANK_SCHEMA_VERSION = "fitted-bank-export-v1"
_CORRELATION_ATOL = 1e-8
_INACTIVE_LOADING_ATOL = 1e-10


class BankValidationError(RuntimeError):
    """Raised when a bank cannot satisfy the calibrated runtime contract."""


@dataclass(frozen=True, slots=True)
class Scenario:
    """One scenario and its ordered, non-empty set of calibrated criteria."""

    scenario_id: str
    prompt: str
    criterion_ids: tuple[str, ...]
    use_case: str = ""
    subject: str = ""
    grade_band: str = ""
    modality: str = "text"
    conversation_context: tuple[Mapping[str, str], ...] = ()
    reference_solution: str = ""
    source: str = ""
    split: str = ""
    version: str = "1.0"
    benchmark: str = ""
    system_prompt: str = ""

    def __post_init__(self) -> None:
        if not self.scenario_id.strip():
            raise BankValidationError("scenario_id must be non-empty")
        if not self.criterion_ids:
            raise BankValidationError(f"scenario {self.scenario_id}: no criterion_ids")
        if any(not criterion_id.strip() for criterion_id in self.criterion_ids):
            raise BankValidationError(
                f"scenario {self.scenario_id}: criterion_ids must be non-empty"
            )
        if len(self.criterion_ids) != len(set(self.criterion_ids)):
            raise BankValidationError(f"scenario {self.scenario_id}: duplicate criterion_ids")

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> Scenario:
        """Parse structural fields; cross-record validation happens in the loader."""

        scenario_id = _nonempty_string(record.get("scenario_id"), "scenario_id")
        raw_criterion_ids = record.get("criterion_ids")
        if not isinstance(raw_criterion_ids, list):
            raise BankValidationError(f"scenario {scenario_id}: criterion_ids must be a list")
        criterion_ids = tuple(
            _nonempty_string(value, f"scenario {scenario_id} criterion_id")
            for value in raw_criterion_ids
        )
        if not criterion_ids:
            raise BankValidationError(f"scenario {scenario_id}: no criterion_ids")
        if len(criterion_ids) != len(set(criterion_ids)):
            raise BankValidationError(f"scenario {scenario_id}: duplicate criterion_ids")

        raw_context = record.get("conversation_context") or []
        if not isinstance(raw_context, list) or any(
            not isinstance(turn, dict) for turn in raw_context
        ):
            raise BankValidationError(
                f"scenario {scenario_id}: conversation_context must be a list of objects"
            )

        return cls(
            scenario_id=scenario_id,
            prompt=str(record.get("prompt") or ""),
            criterion_ids=criterion_ids,
            use_case=str(record.get("use_case") or ""),
            subject=str(record.get("subject") or ""),
            grade_band=str(record.get("grade_band") or ""),
            modality=str(record.get("modality") or "text"),
            conversation_context=tuple(
                MappingProxyType({str(k): str(v) for k, v in turn.items()}) for turn in raw_context
            ),
            reference_solution=str(record.get("reference_solution") or ""),
            source=str(record.get("source") or ""),
            split=str(record.get("split") or ""),
            version=str(record.get("version") or "1.0"),
            benchmark=str(record.get("benchmark") or ""),
            system_prompt=str(record.get("system_prompt") or ""),
        )


@dataclass(frozen=True, slots=True)
class Rubric:
    """One calibrated binary item on the bank's explicit latent-skill axis."""

    criterion_id: str
    scenario_id: str
    criterion: str
    q: np.ndarray  # shape (K,), binary and nonzero
    a: np.ndarray  # shape (K,), positive on-Q and exactly zero off-Q
    b: float
    calibration_version: str
    primary_skill: str = ""
    scoring_type: str = "binary"
    criticality: str = "standard"
    objectivity: str = ""
    explicitness: str = ""
    q_rationale: str = ""
    source: str = ""
    status: str = "approved"
    version: str = "1.0"
    verifier: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.criterion_id.strip() or not self.scenario_id.strip():
            raise BankValidationError("rubric criterion_id and scenario_id must be non-empty")
        if not self.criterion.strip():
            raise BankValidationError(f"{self.criterion_id}: criterion must be non-empty")
        if not self.calibration_version.strip():
            raise BankValidationError(f"{self.criterion_id}: calibration_version must be non-empty")
        if self.scoring_type != "binary":
            raise BankValidationError(f"{self.criterion_id}: scoring_type must be 'binary'")
        raw_q = np.asarray(self.q)
        a = np.asarray(self.a, dtype=float).copy()
        if raw_q.ndim != 1 or a.shape != raw_q.shape:
            raise BankValidationError(
                f"{self.criterion_id}: q and a must be aligned one-dimensional vectors"
            )
        if (
            raw_q.size == 0
            or not all(value in (0, 1, False, True) for value in raw_q.tolist())
            or int(raw_q.sum()) == 0
        ):
            raise BankValidationError(
                f"{self.criterion_id}: q must be a non-empty, binary, nonzero vector"
            )
        q = raw_q.astype(np.int8, copy=True)
        if not np.isfinite(a).all() or not math.isfinite(float(self.b)):
            raise BankValidationError(f"{self.criterion_id}: fitted parameters must be finite")
        if np.any(a[q == 1] <= 0):
            raise BankValidationError(
                f"{self.criterion_id}: active discrimination values must be positive"
            )
        if np.any(np.abs(a[q == 0]) > _INACTIVE_LOADING_ATOL):
            raise BankValidationError(
                f"{self.criterion_id}: inactive discrimination values must be zero"
            )
        q.setflags(write=False)
        a.setflags(write=False)
        object.__setattr__(self, "q", q)
        object.__setattr__(self, "a", a)


@dataclass(frozen=True, slots=True)
class FittedBank:
    """Fully validated runtime bank in one immutable latent-dimension order.

    ``Q``, ``A``, and ``b`` are aligned to ``criterion_ids``.  ``A`` contains the
    effective Q-masked loading, so all runtime probability and information calculations
    use exactly the same parameterization: ``sigmoid(A @ theta - b)``.
    """

    scenarios: Mapping[str, Scenario]
    rubrics: Mapping[str, Rubric]
    skills: tuple[str, ...]
    latent_correlation: np.ndarray
    records: tuple[Mapping[str, Any], ...]
    source_path: str
    scenarios_path: str
    manifest_path: str | None = None
    manifest_sha256: str | None = None
    schema_version: str | None = None
    criterion_ids: tuple[str, ...] = field(init=False)
    scenario_ids: tuple[str, ...] = field(init=False)
    Q: np.ndarray = field(init=False, repr=False)
    A: np.ndarray = field(init=False, repr=False)
    b: np.ndarray = field(init=False, repr=False)
    _index: Mapping[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        axis = _normalize_skills(self.skills, "bank skills")
        scenario_map = MappingProxyType(dict(self.scenarios))
        rubric_map = MappingProxyType(dict(self.rubrics))
        if not scenario_map or not rubric_map:
            raise BankValidationError("fitted bank must contain scenarios and rubrics")
        for scenario_id, scenario in scenario_map.items():
            if scenario_id != scenario.scenario_id:
                raise BankValidationError(
                    f"scenario mapping key {scenario_id!r} != record id {scenario.scenario_id!r}"
                )
        for criterion_id, rubric in rubric_map.items():
            if criterion_id != rubric.criterion_id:
                raise BankValidationError(
                    f"rubric mapping key {criterion_id!r} != record id {rubric.criterion_id!r}"
                )
            if rubric.q.shape != (len(axis),) or rubric.a.shape != (len(axis),):
                raise BankValidationError(
                    f"{criterion_id}: parameter dimensions do not match bank skills"
                )
        _validate_links(scenario_map, rubric_map)
        criterion_ids = tuple(rubric_map)
        scenario_ids = tuple(rubric_map[cid].scenario_id for cid in criterion_ids)
        q = np.stack([rubric_map[cid].q for cid in criterion_ids]).astype(np.int8)
        a = np.stack([rubric_map[cid].a for cid in criterion_ids]).astype(float)
        b = np.asarray([rubric_map[cid].b for cid in criterion_ids], dtype=float)
        corr = _validate_correlation(self.latent_correlation, len(axis), "bank")
        for array in (q, a, b, corr):
            array.setflags(write=False)

        object.__setattr__(self, "scenarios", scenario_map)
        object.__setattr__(self, "rubrics", rubric_map)
        object.__setattr__(self, "skills", axis)
        object.__setattr__(self, "criterion_ids", criterion_ids)
        object.__setattr__(self, "scenario_ids", scenario_ids)
        object.__setattr__(self, "Q", q)
        object.__setattr__(self, "A", a)
        object.__setattr__(self, "b", b)
        object.__setattr__(self, "latent_correlation", corr)
        object.__setattr__(
            self,
            "_index",
            MappingProxyType({cid: index for index, cid in enumerate(criterion_ids)}),
        )

    @property
    def dims(self) -> tuple[str, ...]:
        """Compatibility name used by the original InFoBench offline studies."""

        return self.skills

    @property
    def n_dims(self) -> int:
        return len(self.skills)

    @property
    def n_items(self) -> int:
        return len(self.criterion_ids)

    @property
    def index(self) -> Mapping[str, int]:
        return self._index

    def rubrics_for(self, scenario_id: str) -> list[Rubric]:
        """Return criteria in the scenario's recorded administration order."""

        try:
            scenario = self.scenarios[scenario_id]
        except KeyError:
            raise KeyError(f"unknown scenario_id {scenario_id!r}") from None
        return [self.rubrics[cid] for cid in scenario.criterion_ids]

    def scenario_groups(self) -> dict[str, list[int]]:
        groups: dict[str, list[int]] = {}
        for index, scenario_id in enumerate(self.scenario_ids):
            groups.setdefault(scenario_id, []).append(index)
        return groups


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for provenance and manifest validation."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_fitted_bank(
    rubrics_path: str | Path,
    scenarios_path: str | Path,
    *,
    manifest_path: str | Path | None = None,
    skills: Sequence[str] | None = None,
    latent_correlation: Sequence[Sequence[float]] | np.ndarray | None = None,
) -> FittedBank:
    """Load a fitted-only bank or reject the complete bundle.

    Parameters are accepted only when every item explicitly declares calibrated,
    fitted, non-synthetic provenance and a non-empty calibration version.  ``skills``
    and ``latent_correlation`` may be supplied by a trusted caller, but must agree with
    every corresponding declaration in the bank and manifest.  They are never used to
    paper over a contradictory declaration.
    """

    rubric_file = Path(rubrics_path)
    scenario_file = Path(scenarios_path)
    rubric_records = _read_jsonl(rubric_file, "rubric bank")
    scenario_records = _read_jsonl(scenario_file, "scenario bank")
    manifest = _read_manifest(Path(manifest_path)) if manifest_path is not None else None

    requested_skills = _normalize_skills(skills, "skills") if skills is not None else None
    manifest_skills = (
        _normalize_skills(manifest.get("skills_order"), "manifest skills_order")
        if manifest is not None
        else None
    )
    record_skill_orders = _record_skill_orders(rubric_records)
    declared_orders = [
        order
        for order in (requested_skills, manifest_skills, *record_skill_orders)
        if order is not None
    ]
    if not declared_orders:
        raise BankValidationError(
            "fitted bank has no explicit skills_order; latent dimensions cannot be inferred"
        )
    axis = declared_orders[0]
    if any(order != axis for order in declared_orders[1:]):
        raise BankValidationError(
            f"inconsistent modeled skill order declarations: {declared_orders!r}"
        )

    if manifest is not None:
        _validate_manifest_files(
            manifest,
            rubric_file,
            scenario_file,
            expected_skills=axis,
            rubric_count=len(rubric_records),
            scenario_count=len(scenario_records),
        )

    scenarios = _parse_scenarios(scenario_records)
    rubrics = _parse_rubrics(rubric_records, axis)
    _validate_links(scenarios, rubrics)
    calibration_versions = {rubric.calibration_version for rubric in rubrics.values()}
    if len(calibration_versions) != 1:
        raise BankValidationError(
            f"fitted-bank records mix calibration_version values: {sorted(calibration_versions)!r}"
        )

    correlation_candidates: list[tuple[str, np.ndarray]] = []
    if latent_correlation is not None:
        correlation_candidates.append(
            (
                "caller latent_correlation",
                _validate_correlation(latent_correlation, len(axis), "caller"),
            )
        )
    if manifest is not None and manifest.get("latent_correlation") is not None:
        correlation_candidates.append(
            (
                "manifest latent_correlation",
                _validate_correlation(manifest["latent_correlation"], len(axis), "manifest"),
            )
        )
    correlation_candidates.extend(_record_correlations(rubric_records, len(axis)))
    if not correlation_candidates:
        raise BankValidationError(
            "fitted bank has no explicit latent_correlation; refusing an implicit identity prior"
        )
    correlation = correlation_candidates[0][1]
    disagreeing = [
        label
        for label, candidate in correlation_candidates[1:]
        if not np.allclose(correlation, candidate, atol=_CORRELATION_ATOL, rtol=0.0)
    ]
    if disagreeing:
        raise BankValidationError(
            "latent_correlation declarations disagree: " + ", ".join(disagreeing)
        )

    if manifest is not None:
        _validate_manifest_invariants(manifest, rubrics)

    frozen_records = tuple(_freeze_mapping(record) for record in rubric_records)
    manifest_file = Path(manifest_path) if manifest_path is not None else None
    return FittedBank(
        scenarios=scenarios,
        rubrics=rubrics,
        skills=axis,
        latent_correlation=correlation,
        records=frozen_records,
        source_path=str(rubric_file),
        scenarios_path=str(scenario_file),
        manifest_path=str(manifest_file) if manifest_file is not None else None,
        manifest_sha256=sha256_file(manifest_file) if manifest_file is not None else None,
        schema_version=str(manifest.get("schema_version")) if manifest is not None else None,
    )


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise BankValidationError(f"{label} not found: {path}")
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise BankValidationError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
                if not isinstance(value, dict):
                    raise BankValidationError(f"{path}:{line_number}: expected one JSON object")
                rows.append(value)
    except OSError as exc:
        raise BankValidationError(f"could not read {label} {path}: {exc}") from exc
    if not rows:
        raise BankValidationError(f"{label} contains no records: {path}")
    return rows


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise BankValidationError(f"fitted-bank manifest not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BankValidationError(f"invalid fitted-bank manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BankValidationError("fitted-bank manifest must contain one JSON object")
    if value.get("schema_version") != FITTED_BANK_SCHEMA_VERSION:
        raise BankValidationError(
            f"unsupported fitted-bank manifest schema_version: {value.get('schema_version')!r}"
        )
    return value


def _normalize_skills(value: object, label: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise BankValidationError(f"{label} must be a non-empty ordered sequence")
    axis = tuple(str(skill).strip() for skill in value)
    if not axis or any(not skill for skill in axis) or len(axis) != len(set(axis)):
        raise BankValidationError(f"{label} must contain unique non-empty names: {axis!r}")
    return axis


def _record_skill_orders(
    records: Sequence[Mapping[str, Any]],
) -> list[tuple[str, ...] | None]:
    orders: list[tuple[str, ...] | None] = []
    for position, record in enumerate(records, 1):
        irt = record.get("irt_params")
        if not isinstance(irt, dict):
            raise BankValidationError(f"rubric record {position}: missing object irt_params")
        value = irt.get("skills_order")
        if value is None:
            value = irt.get("modeled_skills")
        if value is None:
            raise BankValidationError(f"rubric record {position}: missing irt_params.skills_order")
        orders.append(_normalize_skills(value, f"rubric record {position} skills_order"))
    return orders


def _parse_scenarios(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Scenario]:
    scenarios: dict[str, Scenario] = {}
    for record in records:
        scenario = Scenario.from_record(record)
        if scenario.scenario_id in scenarios:
            raise BankValidationError(f"duplicate scenario_id {scenario.scenario_id!r}")
        scenarios[scenario.scenario_id] = scenario
    return scenarios


def _parse_rubrics(
    records: Sequence[Mapping[str, Any]], axis: tuple[str, ...]
) -> dict[str, Rubric]:
    rubrics: dict[str, Rubric] = {}
    for position, record in enumerate(records, 1):
        cid = _nonempty_string(record.get("criterion_id"), f"rubric record {position} criterion_id")
        if cid in rubrics:
            raise BankValidationError(f"duplicate criterion_id {cid!r}")
        sid = _nonempty_string(record.get("scenario_id"), f"{cid} scenario_id")
        criterion = _nonempty_string(record.get("criterion"), f"{cid} criterion")
        scoring_type = str(record.get("scoring_type") or "binary")
        if scoring_type != "binary":
            raise BankValidationError(
                f"{cid}: unsupported scoring_type {scoring_type!r}; expected 'binary'"
            )

        irt = record.get("irt_params")
        if not isinstance(irt, dict):
            raise BankValidationError(f"{cid}: missing object irt_params")
        if irt.get("calibrated") is not True or irt.get("fitted") is not True:
            raise BankValidationError(f"{cid}: parameters must be explicitly calibrated and fitted")
        if irt.get("synthetic") is not False:
            raise BankValidationError(f"{cid}: parameters must explicitly declare synthetic=false")
        calibration_version = str(record.get("calibration_version") or "").strip()
        if not calibration_version:
            raise BankValidationError(f"{cid}: missing calibration_version")
        if not str(irt.get("source") or "").strip():
            raise BankValidationError(f"{cid}: missing irt_params.source provenance")

        qmap = record.get("q_modeled")
        disc = record.get("discrimination")
        if not isinstance(qmap, dict) or not isinstance(disc, dict):
            raise BankValidationError(f"{cid}: q_modeled and discrimination must be objects")
        if set(qmap) != set(axis) or set(disc) != set(axis):
            raise BankValidationError(
                f"{cid}: q_modeled/discrimination keys must exactly equal skills_order"
            )
        try:
            q = np.asarray([_binary_value(qmap[skill], cid, skill) for skill in axis])
            a = np.asarray([float(disc[skill]) for skill in axis], dtype=float)
            b = float(record["difficulty"])
        except (KeyError, TypeError, ValueError) as exc:
            raise BankValidationError(f"{cid}: invalid fitted parameters: {exc}") from exc
        if int(q.sum()) == 0:
            raise BankValidationError(f"{cid}: q row must map to at least one skill")
        if not np.isfinite(a).all() or not math.isfinite(b):
            raise BankValidationError(f"{cid}: fitted parameters must be finite")
        if np.any(a[q == 1] <= 0):
            raise BankValidationError(f"{cid}: active discrimination values must be positive")
        if np.any(np.abs(a[q == 0]) > _INACTIVE_LOADING_ATOL):
            raise BankValidationError(f"{cid}: inactive discrimination values must be zero")

        raw_verifier = record.get("verifier")
        if raw_verifier is not None and not isinstance(raw_verifier, dict):
            raise BankValidationError(f"{cid}: verifier must be an object or null")
        rubrics[cid] = Rubric(
            criterion_id=cid,
            scenario_id=sid,
            criterion=criterion,
            q=q,
            a=a,
            b=b,
            calibration_version=calibration_version,
            primary_skill=str(record.get("primary_skill") or ""),
            scoring_type=scoring_type,
            criticality=str(record.get("criticality") or "standard"),
            objectivity=str(record.get("objectivity") or ""),
            explicitness=str(record.get("explicitness") or ""),
            q_rationale=str(record.get("q_rationale") or ""),
            source=str(record.get("source") or ""),
            status=str(record.get("status") or "approved"),
            version=str(record.get("version") or "1.0"),
            verifier=MappingProxyType(dict(raw_verifier)) if raw_verifier is not None else None,
        )
    return rubrics


def _validate_links(scenarios: Mapping[str, Scenario], rubrics: Mapping[str, Rubric]) -> None:
    referenced: set[str] = set()
    for sid, scenario in scenarios.items():
        for cid in scenario.criterion_ids:
            rubric = rubrics.get(cid)
            if rubric is None:
                raise BankValidationError(
                    f"scenario {sid}: criterion_id {cid!r} has no fitted rubric"
                )
            if rubric.scenario_id != sid:
                raise BankValidationError(
                    f"criterion {cid} belongs to {rubric.scenario_id!r}, not {sid!r}"
                )
            if cid in referenced:
                raise BankValidationError(
                    f"criterion {cid!r} is referenced by more than one scenario"
                )
            referenced.add(cid)
    orphaned = sorted(set(rubrics) - referenced)
    if orphaned:
        raise BankValidationError(
            f"{len(orphaned)} fitted rubric(s) are not referenced by scenarios; "
            f"first={orphaned[:10]}"
        )


def _record_correlations(
    records: Sequence[Mapping[str, Any]], n_dims: int
) -> list[tuple[str, np.ndarray]]:
    found: list[tuple[str, np.ndarray]] = []
    for position, record in enumerate(records, 1):
        cid = str(record.get("criterion_id") or position)
        irt = record.get("irt_params") or {}
        value = irt.get("latent_correlation")
        if value is None:
            provenance = irt.get("provenance") or {}
            if not isinstance(provenance, dict):
                raise BankValidationError(f"{cid}: irt_params.provenance must be an object")
            value = provenance.get("latent_correlation")
        if value is None:
            continue
        found.append(
            (
                f"criterion {cid}",
                _validate_correlation(value, n_dims, f"criterion {cid}"),
            )
        )
    return found


def _validate_correlation(value: object, n_dims: int, label: str) -> np.ndarray:
    try:
        correlation = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise BankValidationError(f"{label} latent_correlation is not numeric") from exc
    if correlation.shape != (n_dims, n_dims):
        raise BankValidationError(
            f"{label} latent_correlation shape {correlation.shape} != {(n_dims, n_dims)}"
        )
    if not np.isfinite(correlation).all():
        raise BankValidationError(f"{label} latent_correlation must be finite")
    if not np.allclose(correlation, correlation.T, atol=1e-9, rtol=0.0):
        raise BankValidationError(f"{label} latent_correlation must be symmetric")
    if not np.allclose(np.diag(correlation), 1.0, atol=1e-6, rtol=0.0):
        raise BankValidationError(f"{label} latent_correlation must have unit diagonal")
    if float(np.linalg.eigvalsh(correlation).min()) <= 1e-10:
        raise BankValidationError(f"{label} latent_correlation must be positive definite")
    return correlation.copy()


def _validate_manifest_files(
    manifest: Mapping[str, Any],
    rubric_file: Path,
    scenario_file: Path,
    *,
    expected_skills: tuple[str, ...],
    rubric_count: int,
    scenario_count: int,
) -> None:
    outputs = manifest.get("outputs")
    hashes = outputs.get("sha256") if isinstance(outputs, dict) else None
    if not isinstance(hashes, dict):
        raise BankValidationError("manifest outputs.sha256 must be an object")
    for key, path in (("rubrics", rubric_file), ("scenarios", scenario_file)):
        expected = str(hashes.get(key) or "").strip().lower()
        if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
            raise BankValidationError(f"manifest has no valid SHA-256 for {key}")
        actual = sha256_file(path)
        if actual != expected:
            raise BankValidationError(
                f"manifest SHA-256 mismatch for {key}: expected {expected}, got {actual}"
            )

    counts = manifest.get("counts")
    if not isinstance(counts, dict):
        raise BankValidationError("manifest counts must be an object")
    expected_counts = {
        "exported_criteria": rubric_count,
        "exported_scenarios": scenario_count,
    }
    for key, actual in expected_counts.items():
        try:
            declared = int(counts[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise BankValidationError(f"manifest has invalid count {key}") from exc
        if declared != actual:
            raise BankValidationError(
                f"manifest {key}={declared} does not match loaded count {actual}"
            )
    if tuple(manifest.get("skills_order") or ()) != expected_skills:
        raise BankValidationError("manifest skills_order does not match runtime skill axis")


def _validate_manifest_invariants(
    manifest: Mapping[str, Any], rubrics: Mapping[str, Rubric]
) -> None:
    invariants = manifest.get("invariants")
    if not isinstance(invariants, dict):
        raise BankValidationError("manifest invariants must be an object")
    required_true = (
        "fitted_only",
        "scenario_criterion_ids_exactly_match_exported_rubrics",
        "all_active_loadings_positive_and_within_threshold",
        "all_inactive_loadings_zero",
    )
    for key in required_true:
        if invariants.get(key) is not True:
            raise BankValidationError(f"manifest does not guarantee invariant {key!r}")
    if invariants.get("contains_synthetic_parameters") is not False:
        raise BankValidationError("manifest does not guarantee contains_synthetic_parameters=false")

    policies = manifest.get("policies")
    if not isinstance(policies, dict):
        raise BankValidationError("manifest policies must be an object")
    try:
        extreme_threshold = float(policies["extreme_a_threshold"])
        off_q_tolerance = float(policies["off_q_tolerance"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BankValidationError(
            "manifest must declare numeric extreme_a_threshold and off_q_tolerance"
        ) from exc
    if not math.isfinite(extreme_threshold) or extreme_threshold <= 0:
        raise BankValidationError("manifest extreme_a_threshold must be finite and positive")
    if (
        not math.isfinite(off_q_tolerance)
        or off_q_tolerance < 0
        or off_q_tolerance > _INACTIVE_LOADING_ATOL
    ):
        raise BankValidationError(
            "manifest off_q_tolerance must be finite, nonnegative, and compatible "
            "with the runtime contract"
        )
    for rubric in rubrics.values():
        if np.any(rubric.a[rubric.q == 1] > extreme_threshold):
            raise BankValidationError(
                f"{rubric.criterion_id}: active discrimination exceeds manifest "
                f"threshold {extreme_threshold}"
            )


def _binary_value(value: object, cid: str, skill: str) -> int:
    if value not in (0, 1, "0", "1", False, True):
        raise BankValidationError(f"{cid}: q_modeled[{skill!r}] must be binary, got {value!r}")
    return int(cast(int | float | str | bool, value))


def _nonempty_string(value: object, label: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise BankValidationError(f"{label} must be non-empty")
    return normalized


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Recursively freeze provenance records so runtime state cannot mutate them."""

    return MappingProxyType({str(key): _freeze_value(item) for key, item in value.items()})


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    return value
