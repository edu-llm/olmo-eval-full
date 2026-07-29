"""Ingest the calibration judge-verdicts CSV into a staging response matrix.

RUN WHEN MAPPING ARRIVES (pick the ONE source the teammate provides):
    # (0) FULLY DE-BLINDED verdicts (BEST: tutor_model in every row, no mapping needed):
    ..\\.venv\\Scripts\\python.exe scripts/ingest_calibration_csv.py --jsonl run_data.jsonl
    # (a) private de-blinding index (PREFERRED):
    ..\\.venv\\Scripts\\python.exe scripts/ingest_calibration_csv.py \\
        --csv "run_data (1).csv" --case-index calibration_prepared_v2/private/case_index.jsonl
    # (b) merged output dir (reuses its built matrix if present, else its private index):
    ..\\.venv\\Scripts\\python.exe scripts/ingest_calibration_csv.py \\
        --csv "run_data (1).csv" --merged-dir calibration_merged_v2
    # (c) raw blinding secret (reproduces response_ids over the response archive):
    ..\\.venv\\Scripts\\python.exe scripts/ingest_calibration_csv.py \\
        --csv "run_data (1).csv" --id-key-file calibration_id_key.txt \\
        --responses tutorbench-responses-v2/responses    # or: --id-key <64-hex>

This converts one judge-verdicts CSV (blinded ``response_<24hex>`` HMAC ids) into
``staging/response_matrix.csv`` / ``.npy`` / ``_manifest.json`` in the EXACT format
consumed unchanged by ``scripts/calibrate_partial.py`` and ``scripts/calibrate_mirt.py``
(``pd.read_csv(index_col="model")`` -> rows = models, cols = curated criterion_ids,
cells in {0, 1, empty->NaN}). Cell rules: pass->1, fail->0, no_decision->NaN,
absent-from-CSV->NaN; case-index missing->NaN and auto_fail->0 override the CSV,
matching ``run_calibration_judging.py`` merge semantics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CSV = ROOT / "run_data (1).csv"
DEFAULT_CURATED = ROOT / "data" / "TutorBench" / "curated" / "rubrics_qmatrix_curated.jsonl"
DEFAULT_COHORT_POLICY = (
    ROOT
    / "handoff_extract"
    / "qwen_calibration_handoff_32k_20260728"
    / "calibration_cohort_policy.json"
)
DEFAULT_RESPONSES = ROOT / "tutorbench-responses-v2" / "responses"
DEFAULT_STAGING = ROOT / "staging"

# Frozen contract totals from RUN_INSTRUCTIONS.md / calibration_cohort_policy.json.
CONTRACT_CELL_COUNT = 506_760
CONTRACT_JUDGE_CASE_COUNT = 505_615
CONTRACT_MISSING_CELL_COUNT = 1_145

_RESPONSE_ID_PREFIX = "response_"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


def _sha256_file(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def response_id_for(id_key: bytes, model: str, scenario_id: str) -> str:
    """Reproduce a blinded response id (verbatim recipe from the prepare step)."""
    payload = f"response\x1f{model}\x1f{scenario_id}".encode()
    return _RESPONSE_ID_PREFIX + hmac.new(id_key, payload, hashlib.sha256).hexdigest()[:24]


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if line:
                yield line_number, json.loads(line)


def _coerce_id_key(hex_value: str) -> bytes:
    """Match run_calibration_judging.py: the id key is the hex string's UTF-8 bytes."""
    return hex_value.strip().encode("utf-8")


# ---------------------------------------------------------------------------
# curated criteria (matrix columns) + cohort policy
# ---------------------------------------------------------------------------


def load_curated(path: Path) -> tuple[list[str], dict[str, str]]:
    """Nonoptional curated criterion_ids (bank file order) + criterion->scenario map.

    Matches the ``optional is not True`` selection in run_calibration_judging.py
    (6,180 of the 6,845 curated rows for this study). The column order is the file
    order; ``criterion_scenario`` maps each column to its scenario_id.
    """
    columns: list[str] = []
    criterion_scenario: dict[str, str] = {}
    seen: set[str] = set()
    for _, record in _read_jsonl(path):
        if record.get("optional") is True:
            continue
        cid = str(record.get("criterion_id") or "").strip()
        if not cid or cid in seen:
            continue
        seen.add(cid)
        columns.append(cid)
        criterion_scenario[cid] = str(record.get("scenario_id") or "").strip()
    return columns, criterion_scenario


def load_curated_columns(path: Path) -> list[str]:
    """Nonoptional curated criterion_ids, in bank file order = the matrix columns."""
    return load_curated(path)[0]


def load_cohort_models(path: Path) -> list[str]:
    """The common-person calibration cohort = cohort policy ``included_models``."""
    with path.open("r", encoding="utf-8") as handle:
        policy = json.load(handle)
    return [str(m) for m in policy.get("included_models", [])]


# ---------------------------------------------------------------------------
# mapping sources
# ---------------------------------------------------------------------------


@dataclass
class ModelMapping:
    """Resolved response_id -> model mapping plus optional per-cell policy."""

    response_to_model: dict[str, str]
    model_order: list[str]
    source_desc: str
    # (model, criterion_id) -> "missing" | "auto_fail"; empty for path (c).
    cell_policy: dict[tuple[str, str], str] = field(default_factory=dict)
    reused_matrix: pd.DataFrame | None = None


def _ordered_unique(values) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def mapping_from_case_index(path: Path) -> ModelMapping:
    """Path (a): read the private de-blinding index."""
    response_to_model: dict[str, str] = {}
    model_first_pos: dict[str, int] = {}
    cell_policy: dict[tuple[str, str], str] = {}
    for fallback_pos, (_, row) in enumerate(_read_jsonl(path)):
        response_id = str(row["response_id"])
        model = str(row["model"])
        response_to_model[response_id] = model
        pos = int(row.get("position", fallback_pos))
        model_first_pos.setdefault(model, pos)
        criterion_id = str(row.get("criterion_id") or "")
        if criterion_id:
            if bool(row.get("missing")):
                cell_policy[(model, criterion_id)] = "missing"
            elif bool(row.get("auto_fail")):
                cell_policy[(model, criterion_id)] = "auto_fail"
    model_order = sorted(model_first_pos, key=lambda m: model_first_pos[m])
    return ModelMapping(
        response_to_model=response_to_model,
        model_order=model_order,
        source_desc=f"case-index:{path}",
        cell_policy=cell_policy,
    )


def mapping_from_hmac(id_key: bytes, responses_dir: Path) -> ModelMapping:
    """Path (c): reproduce response_ids over every (Model, Scenario) on disk."""
    response_to_model: dict[str, str] = {}
    models_seen: list[str] = []
    files = sorted(responses_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no response JSONL files under {responses_dir}")
    for file_path in files:
        for _, record in _read_jsonl(file_path):
            model = str(record.get("Model") or "")
            scenario_id = str(record.get("Scenario") or "")
            if not model or not scenario_id:
                continue
            rid = response_id_for(id_key, model, scenario_id)
            response_to_model[rid] = model
            models_seen.append(model)
    return ModelMapping(
        response_to_model=response_to_model,
        model_order=_ordered_unique(models_seen),
        source_desc=f"id-key-hmac:{responses_dir}",
    )


def _find_merged_matrix(merged_dir: Path) -> Path | None:
    candidate = merged_dir / "response_matrix.csv"
    return candidate if candidate.is_file() else None


def _find_merged_case_index(merged_dir: Path) -> Path | None:
    for candidate in (
        merged_dir / "private" / "case_index.jsonl",
        merged_dir / "case_index.jsonl",
    ):
        if candidate.is_file():
            return candidate
    return None


def mapping_from_merged_dir(merged_dir: Path) -> ModelMapping:
    """Path (b): reuse a merged matrix if present, else fall back to its index."""
    matrix_path = _find_merged_matrix(merged_dir)
    if matrix_path is not None:
        reused = pd.read_csv(matrix_path, index_col="model")
        reused = reused.apply(pd.to_numeric, errors="coerce")
        return ModelMapping(
            response_to_model={},
            model_order=[str(m) for m in reused.index],
            source_desc=f"merged-dir-matrix:{matrix_path}",
            reused_matrix=reused,
        )
    index_path = _find_merged_case_index(merged_dir)
    if index_path is not None:
        mapping = mapping_from_case_index(index_path)
        mapping.source_desc = f"merged-dir-index:{index_path}"
        return mapping
    raise FileNotFoundError(
        f"{merged_dir}: no response_matrix.csv and no private/case_index.jsonl to reuse"
    )


# ---------------------------------------------------------------------------
# CSV verdicts
# ---------------------------------------------------------------------------


_VERDICT_TO_CELL = {"pass": 1.0, "fail": 0.0, "no_decision": np.nan}


@dataclass
class CsvVerdicts:
    frame: pd.DataFrame
    total_rows: int
    verdict_counts: dict[str, int]
    n_conflicts: int


def load_csv_verdicts(path: Path) -> CsvVerdicts:
    """Load the judge-verdicts CSV, collapsing to one final decision per case_id."""
    usecols = {
        "response_id",
        "criterion_id",
        "scenario_id",
        "verdict",
        "decision_source",
        "status",
        "attempt",
    }
    df = pd.read_csv(path, usecols=lambda c: c in usecols, dtype=str, keep_default_na=False)
    total_rows = int(len(df))
    df["verdict"] = df["verdict"].str.strip()
    verdict_counts = {str(k): int(v) for k, v in df["verdict"].value_counts().to_dict().items()}

    if "attempt" in df.columns:
        df["_attempt"] = pd.to_numeric(df["attempt"], errors="coerce").fillna(0)
    else:
        df["_attempt"] = 0
    # Definitive (pass/fail) beats no_decision; then highest attempt wins.
    df["_definitive"] = df["verdict"].isin(["pass", "fail"]).astype(int)
    df = df.sort_values(["_definitive", "_attempt"])

    n_conflicts = 0
    definitive = df[df["_definitive"] == 1]
    if len(definitive):
        distinct = definitive.groupby(["response_id", "criterion_id"])["verdict"].nunique()
        n_conflicts = int((distinct > 1).sum())

    deduped = df.groupby(["response_id", "criterion_id"], sort=False).tail(1).copy()
    deduped["cell"] = deduped["verdict"].map(_VERDICT_TO_CELL)
    cols = [
        c
        for c in (
            "response_id",
            "criterion_id",
            "scenario_id",
            "verdict",
            "cell",
            "decision_source",
        )
        if c in deduped.columns
    ]
    return CsvVerdicts(
        frame=deduped[cols].reset_index(drop=True),
        total_rows=total_rows,
        verdict_counts=verdict_counts,
        n_conflicts=n_conflicts,
    )


# ---------------------------------------------------------------------------
# matrix assembly
# ---------------------------------------------------------------------------


@dataclass
class IngestResult:
    matrix: pd.DataFrame
    unresolved_response_ids: list[str]
    off_bank_criteria: set[str]
    csv_distinct_response_ids: int
    csv_resolved_response_ids: int
    no_decision_cells: int
    policy_missing_cells: int
    policy_auto_fail_cells: int
    applied_csv_cells: int


def build_matrix(
    columns: list[str],
    mapping: ModelMapping,
    verdicts: CsvVerdicts,
    *,
    cohort_only: bool,
    cohort_models: list[str],
) -> IngestResult:
    """Assemble the model x criterion matrix following the frozen cell rules."""
    col_set = set(columns)

    models = list(mapping.model_order)
    if cohort_only:
        cohort_set = set(cohort_models)
        models = [m for m in models if m in cohort_set]

    matrix = pd.DataFrame(
        np.nan, index=pd.Index(models, name="model"), columns=columns, dtype=float
    )

    # Reused merged matrix (path b): restrict to curated columns + selected rows.
    if mapping.reused_matrix is not None:
        src = mapping.reused_matrix
        shared_cols = [c for c in columns if c in src.columns]
        shared_rows = [m for m in models if m in src.index]
        if shared_cols and shared_rows:
            matrix.loc[shared_rows, shared_cols] = src.loc[shared_rows, shared_cols].to_numpy()
        off_bank = {str(c) for c in src.columns if c not in col_set}
        applied = int(matrix.notna().to_numpy().sum())
        return IngestResult(
            matrix=matrix,
            unresolved_response_ids=[],
            off_bank_criteria=off_bank,
            csv_distinct_response_ids=0,
            csv_resolved_response_ids=0,
            no_decision_cells=0,
            policy_missing_cells=0,
            policy_auto_fail_cells=0,
            applied_csv_cells=applied,
        )

    model_set = set(models)
    unresolved: set[str] = set()
    off_bank: set[str] = set()
    no_decision_cells = 0
    applied = 0

    resolved_rids: set[str] = set()
    distinct_rids: set[str] = set()
    for row in verdicts.frame.itertuples(index=False):
        response_id = row.response_id
        criterion_id = row.criterion_id
        distinct_rids.add(response_id)
        model = mapping.response_to_model.get(response_id)
        if model is None:
            unresolved.add(response_id)
            continue
        resolved_rids.add(response_id)
        if criterion_id not in col_set:
            off_bank.add(criterion_id)
            continue
        if model not in model_set:
            continue
        cell = row.cell
        if pd.isna(cell):
            no_decision_cells += 1
            # leave NaN: absent and no_decision are indistinguishable in the matrix
            continue
        matrix.at[model, criterion_id] = float(cell)
        applied += 1

    # Case-index policy overrides the CSV (matches merge: missing->NaN, auto_fail->0).
    policy_missing = 0
    policy_auto_fail = 0
    for (model, criterion_id), policy in mapping.cell_policy.items():
        if criterion_id not in col_set or model not in model_set:
            continue
        if policy == "auto_fail":
            matrix.at[model, criterion_id] = 0.0
            policy_auto_fail += 1
        else:  # missing
            matrix.at[model, criterion_id] = np.nan
            policy_missing += 1

    return IngestResult(
        matrix=matrix,
        unresolved_response_ids=sorted(unresolved),
        off_bank_criteria=off_bank,
        csv_distinct_response_ids=len(distinct_rids),
        csv_resolved_response_ids=len(resolved_rids),
        no_decision_cells=no_decision_cells,
        policy_missing_cells=policy_missing,
        policy_auto_fail_cells=policy_auto_fail,
        applied_csv_cells=applied,
    )


# ---------------------------------------------------------------------------
# de-blinded JSONL ingest (streaming; tutor_model is the row key, no mapping)
# ---------------------------------------------------------------------------


def _order_models(present: set[str], cohort_models: list[str], cohort_only: bool) -> list[str]:
    """Cohort order first (for present models), then any extras, sorted."""
    cohort_set = set(cohort_models)
    ordered = [m for m in cohort_models if m in present]
    ordered += sorted(m for m in present if m not in cohort_set)
    if cohort_only:
        ordered = [m for m in ordered if m in cohort_set]
    return ordered


def ingest_jsonl(
    path: Path,
    columns: list[str],
    *,
    cohort_only: bool,
    cohort_models: list[str],
) -> tuple[CsvVerdicts, IngestResult, set[str]]:
    """Stream a fully de-blinded verdicts JSONL (``tutor_model`` per row) -> matrix.

    Reads line-by-line (never loads the whole file). ``tutor_model`` is the row key,
    so no mapping artifact is needed. Applies the frozen cell rules: pass->1, fail->0,
    no_decision->NaN, absent->NaN. When a (model, criterion) appears more than once a
    definitive pass/fail beats no_decision and the last definitive verdict wins.
    """
    col_set = set(columns)
    verdict_counter: Counter[str] = Counter()
    total_rows = 0
    distinct_rids: set[str] = set()
    present_models: set[str] = set()
    present_scenarios: set[str] = set()
    off_bank: set[str] = set()
    nd_records: list[dict[str, str]] = []
    cell_verdict: dict[tuple[str, str], str] = {}
    n_conflicts = 0

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            total_rows += 1
            model = str(row.get("tutor_model") or "")
            criterion_id = str(row.get("criterion_id") or "")
            verdict = str(row.get("verdict") or "").strip()
            scenario_id = str(row.get("scenario_id") or "")
            response_id = str(row.get("response_id") or "")
            decision_source = row.get("decision_source")

            verdict_counter[verdict] += 1
            if response_id:
                distinct_rids.add(response_id)
            if model:
                present_models.add(model)
            if scenario_id:
                present_scenarios.add(scenario_id)

            if criterion_id not in col_set:
                off_bank.add(criterion_id)
                continue
            if verdict == "no_decision":
                nd_records.append(
                    {
                        "verdict": verdict,
                        "scenario_id": scenario_id,
                        "criterion_id": criterion_id,
                        "decision_source": str(decision_source or ""),
                    }
                )
            key = (model, criterion_id)
            prev = cell_verdict.get(key)
            cur_def = verdict in ("pass", "fail")
            if prev is None:
                cell_verdict[key] = verdict
            else:
                prev_def = prev in ("pass", "fail")
                if cur_def and prev_def:
                    if prev != verdict:
                        n_conflicts += 1
                    cell_verdict[key] = verdict  # last definitive wins
                elif cur_def and not prev_def:
                    cell_verdict[key] = verdict

    models = _order_models(present_models, cohort_models, cohort_only)
    model_set = set(models)
    matrix = pd.DataFrame(
        np.nan, index=pd.Index(models, name="model"), columns=columns, dtype=float
    )

    no_decision_cells = 0
    applied = 0
    for (model, criterion_id), verdict in cell_verdict.items():
        if model not in model_set:
            continue
        cell = _VERDICT_TO_CELL.get(verdict, np.nan)
        if pd.isna(cell):
            if verdict == "no_decision":
                no_decision_cells += 1
            continue
        matrix.at[model, criterion_id] = float(cell)
        applied += 1

    nd_frame = pd.DataFrame(
        nd_records, columns=["verdict", "scenario_id", "criterion_id", "decision_source"]
    )
    verdicts = CsvVerdicts(
        frame=nd_frame,
        total_rows=total_rows,
        verdict_counts={str(k): int(v) for k, v in verdict_counter.items()},
        n_conflicts=n_conflicts,
    )
    result = IngestResult(
        matrix=matrix,
        unresolved_response_ids=[],
        off_bank_criteria=off_bank,
        csv_distinct_response_ids=len(distinct_rids),
        csv_resolved_response_ids=len(distinct_rids),
        no_decision_cells=no_decision_cells,
        policy_missing_cells=0,
        policy_auto_fail_cells=0,
        applied_csv_cells=applied,
    )
    return verdicts, result, present_scenarios


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------


def write_matrix_csv(matrix: pd.DataFrame, path: Path) -> None:
    """Write in the EXACT format the calibrate_* fitters expect.

    Header ``["model", *criterion_ids]``; blank cell for NaN; integer 0/1 otherwise
    (csv.writer, so it matches run_calibration_judging.py::cmd_merge byte-for-byte).
    """
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", *matrix.columns])
        for model, series in matrix.iterrows():
            values: list[object] = []
            for value in series:
                if pd.isna(value):
                    values.append("")
                else:
                    values.append(int(round(value)))
            writer.writerow([model, *values])


def write_matrix_npy(matrix: pd.DataFrame, path: Path) -> None:
    """float32 with NaN holes, matching the merge ``response_matrix.npy``."""
    np.save(path, matrix.to_numpy(dtype=np.float32))


def _counts(matrix: pd.DataFrame) -> dict[str, int]:
    arr = matrix.to_numpy()
    n_pass = int(np.nansum(arr == 1.0))
    n_fail = int(np.nansum(arr == 0.0))
    n_missing = int(np.isnan(arr).sum())
    return {"pass": n_pass, "fail": n_fail, "missing_or_nan": n_missing}


def build_matrix_report(
    matrix: pd.DataFrame,
    columns: list[str],
    *,
    criterion_scenario: dict[str, str],
    present_scenarios: set[str],
    responses_dir: Path,
    empty_sample_size: int = 40,
) -> dict:
    """Coverage diagnostics: empty columns, per-model counts, scenarios, file match."""
    arr = matrix.to_numpy()
    n_models, n_criteria = matrix.shape
    total_cells = int(n_models * n_criteria)
    observed_mask = ~np.isnan(arr)
    observed = int(observed_mask.sum())
    per_col = observed_mask.sum(axis=0)
    per_row = observed_mask.sum(axis=1)

    empty_cols = [columns[i] for i in range(n_criteria) if per_col[i] == 0]
    per_model = sorted(
        ({"model": m, "observed_cells": int(per_row[i])} for i, m in enumerate(matrix.index)),
        key=lambda entry: entry["observed_cells"],
    )

    curated_scenarios = {s for s in criterion_scenario.values() if s}
    absent_scenarios = sorted(curated_scenarios - present_scenarios)

    files = {p.stem for p in responses_dir.glob("*.jsonl")} if responses_dir.is_dir() else set()
    unmatched_models = [m for m in matrix.index if m.replace("/", "_") not in files]

    return {
        "fill_rate": round(observed / total_cells, 6) if total_cells else 0.0,
        "observed_cells": observed,
        "total_cells": total_cells,
        "empty_column_count": len(empty_cols),
        "empty_columns_sample": empty_cols[:empty_sample_size],
        "curated_scenario_count": len(curated_scenarios),
        "present_scenario_count": len(present_scenarios & curated_scenarios),
        "absent_scenario_count": len(absent_scenarios),
        "absent_scenarios_sample": absent_scenarios[:empty_sample_size],
        "per_model_cell_counts": per_model,
        "least_covered_models": per_model[:10],
        "responses_dir_files_found": len(files),
        "tutor_model_unmatched_v2_files": unmatched_models,
    }


def build_manifest(
    *,
    source_path: Path,
    input_mode: str,
    source_desc: str,
    curated_path: Path,
    cohort_policy_path: Path,
    result: IngestResult,
    verdicts: CsvVerdicts,
    cohort_models: list[str],
    cohort_only: bool,
    row_order: str,
    matrix_report: dict | None = None,
) -> dict:
    matrix = result.matrix
    n_models, n_criteria = matrix.shape
    counts = _counts(matrix)
    cohort_set = set(cohort_models)
    mapped_in_cohort = [m for m in matrix.index if m in cohort_set]
    coverage = (
        result.csv_resolved_response_ids / result.csv_distinct_response_ids
        if result.csv_distinct_response_ids
        else None
    )
    return {
        "generated_at": _utcnow(),
        "input_mode": input_mode,
        "source_file": {"path": str(source_path), "sha256": _sha256_file(source_path)},
        "curated_bank": {"path": str(curated_path), "n_criteria": int(n_criteria)},
        "cohort_policy": {"path": str(cohort_policy_path)},
        "mapping_source": source_desc,
        "column_order": "curated bank file order, nonoptional criteria only",
        "row_order": row_order,
        "matrix_report": matrix_report,
        "n_models": int(n_models),
        "n_criteria": int(n_criteria),
        "cell_count": int(n_models * n_criteria),
        "pass_count": counts["pass"],
        "fail_count": counts["fail"],
        "no_decision_cells": result.no_decision_cells,
        "missing_or_nan_cells": counts["missing_or_nan"],
        "policy_missing_cells": result.policy_missing_cells,
        "policy_auto_fail_cells": result.policy_auto_fail_cells,
        "applied_csv_cells": result.applied_csv_cells,
        "csv_verdict_counts": verdicts.verdict_counts,
        "csv_total_rows": verdicts.total_rows,
        "csv_conflicting_case_ids": verdicts.n_conflicts,
        "csv_distinct_response_ids": result.csv_distinct_response_ids,
        "csv_resolved_response_ids": result.csv_resolved_response_ids,
        "response_id_coverage": coverage,
        "unresolved_response_id_count": len(result.unresolved_response_ids),
        "off_bank_criterion_count": len(result.off_bank_criteria),
        "contract": {
            "expected_cell_count": CONTRACT_CELL_COUNT,
            "expected_judge_case_count": CONTRACT_JUDGE_CASE_COUNT,
            "expected_missing_cell_count": CONTRACT_MISSING_CELL_COUNT,
            "source_total_rows": verdicts.total_rows,
            "source_rows_vs_expected_judged_gap": CONTRACT_JUDGE_CASE_COUNT - verdicts.total_rows,
            "gap_note": (
                "Source row count is ~20k short of the 505,615 expected judge cases; "
                "the shortfall corresponds to entire criterion columns with zero rows "
                "(absent criteria x models). Absent cells remain NaN. Reported, not hidden."
            ),
        },
        "cohort": {
            "cohort_model_count": len(cohort_models),
            "cohort_models": cohort_models,
            "cohort_only_applied": bool(cohort_only),
            "mapped_models_in_cohort": len(mapped_in_cohort),
            "mapped_models_out_of_cohort": [m for m in matrix.index if m not in cohort_set],
        },
        "assumptions": {
            "pass": "1",
            "fail": "0",
            "no_decision": "NaN (absent/no_decision indistinguishable in the matrix)",
            "absent_from_csv": "NaN",
            "case_index_missing": "NaN (blank + frozen over-length; overrides CSV)",
            "case_index_auto_fail": "0 (real fail; overrides CSV; 0 for this study)",
        },
    }


def build_audit(
    *,
    source_path: Path,
    source_desc: str,
    result: IngestResult,
    verdicts: CsvVerdicts,
    matrix_report: dict | None = None,
) -> dict:
    return {
        "generated_at": _utcnow(),
        "source_file": str(source_path),
        "mapping_source": source_desc,
        "matrix_report": matrix_report,
        "row_count_reconciliation": {
            "source_total_rows": verdicts.total_rows,
            "expected_judge_case_count": CONTRACT_JUDGE_CASE_COUNT,
            "gap": CONTRACT_JUDGE_CASE_COUNT - verdicts.total_rows,
            "expected_cell_count": CONTRACT_CELL_COUNT,
            "expected_missing_cell_count": CONTRACT_MISSING_CELL_COUNT,
        },
        "verdict_counts": verdicts.verdict_counts,
        "conflicting_case_ids": verdicts.n_conflicts,
        "no_decision": {
            "note": "model_verdict is NaN on no_decision rows; these cells are NaN.",
            "no_decision_by_scenario": _no_decision_breakdown(verdicts.frame, "scenario_id"),
            "no_decision_by_criterion": _no_decision_breakdown(verdicts.frame, "criterion_id"),
            "no_decision_by_decision_source": _no_decision_breakdown(
                verdicts.frame, "decision_source"
            ),
        },
        "unresolved_response_ids": result.unresolved_response_ids[:1000],
        "unresolved_response_id_count": len(result.unresolved_response_ids),
        "off_bank_criteria_sample": sorted(result.off_bank_criteria)[:50],
        "off_bank_criterion_count": len(result.off_bank_criteria),
    }


def _no_decision_breakdown(frame: pd.DataFrame, key: str) -> dict[str, int]:
    if key not in frame.columns:
        return {}
    nd = frame[frame["verdict"] == "no_decision"]
    if not len(nd):
        return {}
    counts = nd[key].value_counts()
    return {str(k): int(v) for k, v in counts.head(50).to_dict().items()}


def write_audit_md(audit: dict, path: Path) -> None:
    rc = audit["row_count_reconciliation"]
    lines = [
        "# Calibration verdict ingest audit",
        "",
        f"Generated: {audit['generated_at']}",
        f"Source file: `{audit['source_file']}`",
        f"Mapping source: `{audit['mapping_source']}`",
        "",
        "## Row-count reconciliation vs contract",
        "",
        f"- Source rows: {rc['source_total_rows']:,}",
        f"- Expected judge cases: {rc['expected_judge_case_count']:,}",
        f"- Gap (expected - source): {rc['gap']:,}",
        f"- Expected cells: {rc['expected_cell_count']:,}",
        f"- Expected missing cells: {rc['expected_missing_cell_count']:,}",
        "",
        "## Verdict counts",
        "",
    ]
    for verdict, count in sorted(audit["verdict_counts"].items()):
        lines.append(f"- {verdict}: {count:,}")
    lines += [
        "",
        f"Conflicting case_ids (pass vs fail): {audit['conflicting_case_ids']:,}",
    ]

    report = audit.get("matrix_report")
    if report:
        lines += [
            "",
            "## Matrix coverage",
            "",
            f"- Shape: {report['total_cells'] // max(len(report['per_model_cell_counts']), 1)} "
            f"criteria x {len(report['per_model_cell_counts'])} models "
            f"(total cells {report['total_cells']:,})",
            f"- Observed cells: {report['observed_cells']:,}",
            f"- Fill rate: {report['fill_rate'] * 100:.2f}%",
            f"- All-empty (all-NaN) criterion columns: {report['empty_column_count']:,}",
            f"- Scenarios present / curated: {report['present_scenario_count']:,} / "
            f"{report['curated_scenario_count']:,} "
            f"(absent: {report['absent_scenario_count']:,})",
            f"- tutor_model values unmatched to a v2 file: "
            f"{len(report['tutor_model_unmatched_v2_files'])}",
            "",
            "Sample all-empty criterion columns:",
            "",
        ]
        for cid in report["empty_columns_sample"]:
            lines.append(f"- {cid}")
        lines += ["", "Least-covered models (observed cells):", ""]
        for entry in report["least_covered_models"]:
            lines.append(f"- {entry['model']}: {entry['observed_cells']:,}")
        if report["tutor_model_unmatched_v2_files"]:
            lines += ["", "Unmatched tutor_model values:", ""]
            for model in report["tutor_model_unmatched_v2_files"]:
                lines.append(f"- {model}")

    lines += [
        "",
        "## no_decision distribution",
        "",
        "model_verdict is NaN on no_decision rows; those cells are NaN in the matrix.",
        "",
        "By decision_source:",
        "",
    ]
    for key, count in audit["no_decision"]["no_decision_by_decision_source"].items():
        lines.append(f"- {key}: {count:,}")
    lines += ["", "Top no_decision scenarios:", ""]
    for key, count in list(audit["no_decision"]["no_decision_by_scenario"].items())[:15]:
        lines.append(f"- {key}: {count:,}")
    lines += [
        "",
        "## Unresolved / off-bank",
        "",
        f"- Unresolved response_ids: {audit['unresolved_response_id_count']:,}",
        f"- Off-bank criterion_ids in source: {audit['off_bank_criterion_count']:,}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def resolve_mapping(args: argparse.Namespace) -> ModelMapping:
    if args.case_index is not None:
        return mapping_from_case_index(args.case_index)
    if args.merged_dir is not None:
        return mapping_from_merged_dir(args.merged_dir)
    if args.id_key is not None:
        id_key = _coerce_id_key(args.id_key)
    else:
        id_key = _coerce_id_key(args.id_key_file.read_text(encoding="utf-8"))
    return mapping_from_hmac(id_key, args.responses)


def run(args: argparse.Namespace) -> int:
    staging: Path = args.out_dir
    staging.mkdir(parents=True, exist_ok=True)

    columns, criterion_scenario = load_curated(args.curated)
    cohort_models = load_cohort_models(args.cohort_policy)

    if args.jsonl is not None:
        source_path = args.jsonl
        input_mode = "jsonl-deblinded"
        source_desc = f"jsonl-deblinded:{args.jsonl}"
        row_order = "cohort included_models order (present models), then extras"
        verdicts, result, present_scenarios = ingest_jsonl(
            args.jsonl,
            columns,
            cohort_only=args.cohort_only,
            cohort_models=cohort_models,
        )
    else:
        source_path = args.csv
        input_mode = "csv"
        mapping = resolve_mapping(args)
        source_desc = mapping.source_desc
        row_order = (
            "merged-matrix order"
            if mapping.reused_matrix is not None
            else "case-index position order (= cohort included_models order)"
        )
        verdicts = load_csv_verdicts(args.csv)
        result = build_matrix(
            columns,
            mapping,
            verdicts,
            cohort_only=args.cohort_only,
            cohort_models=cohort_models,
        )
        present_scenarios = (
            set(verdicts.frame["scenario_id"]) if "scenario_id" in verdicts.frame.columns else set()
        )

    matrix_report = build_matrix_report(
        result.matrix,
        columns,
        criterion_scenario=criterion_scenario,
        present_scenarios=present_scenarios,
        responses_dir=args.responses,
    )

    matrix_csv = staging / "response_matrix.csv"
    matrix_npy = staging / "response_matrix.npy"
    manifest_path = staging / "response_matrix_manifest.json"
    audit_json = staging / "ingest_audit.json"
    audit_md = staging / "ingest_audit.md"

    write_matrix_csv(result.matrix, matrix_csv)
    write_matrix_npy(result.matrix, matrix_npy)

    manifest = build_manifest(
        source_path=source_path,
        input_mode=input_mode,
        source_desc=source_desc,
        curated_path=args.curated,
        cohort_policy_path=args.cohort_policy,
        result=result,
        verdicts=verdicts,
        cohort_models=cohort_models,
        cohort_only=args.cohort_only,
        row_order=row_order,
        matrix_report=matrix_report,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    audit = build_audit(
        source_path=source_path,
        source_desc=source_desc,
        result=result,
        verdicts=verdicts,
        matrix_report=matrix_report,
    )
    audit_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    write_audit_md(audit, audit_md)

    n_models, n_criteria = result.matrix.shape
    print(f"mapping source     : {source_desc}")
    print(f"matrix             : {n_models} models x {n_criteria} criteria")
    print(
        f"cells pass/fail/NaN: {manifest['pass_count']} / {manifest['fail_count']} / "
        f"{manifest['missing_or_nan_cells']}"
    )
    print(
        f"no_decision cells  : {result.no_decision_cells}  "
        f"policy missing/auto_fail: {result.policy_missing_cells}/{result.policy_auto_fail_cells}"
    )
    print(
        f"fill rate          : {matrix_report['fill_rate'] * 100:.2f}%  "
        f"empty columns: {matrix_report['empty_column_count']}  "
        f"scenarios present/curated: {matrix_report['present_scenario_count']}/"
        f"{matrix_report['curated_scenario_count']}"
    )
    if result.unresolved_response_ids:
        print(f"unresolved rids    : {len(result.unresolved_response_ids)}")
    if matrix_report["tutor_model_unmatched_v2_files"]:
        print(f"UNMATCHED models   : {matrix_report['tutor_model_unmatched_v2_files']}")
    print(
        f"contract gap       : source rows {verdicts.total_rows} vs expected judged "
        f"{CONTRACT_JUDGE_CASE_COUNT} (gap {CONTRACT_JUDGE_CASE_COUNT - verdicts.total_rows})"
    )
    for out_path in (matrix_csv, matrix_npy, manifest_path, audit_json, audit_md):
        print(f"wrote {out_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="judge-verdicts CSV.")
    p.add_argument(
        "--curated",
        type=Path,
        default=DEFAULT_CURATED,
        help="curated rubric bank JSONL (defines matrix columns).",
    )
    p.add_argument(
        "--cohort-policy",
        type=Path,
        default=DEFAULT_COHORT_POLICY,
        help="calibration cohort policy JSON.",
    )
    p.add_argument(
        "--responses",
        type=Path,
        default=DEFAULT_RESPONSES,
        help="tutorbench responses dir (path (c) only).",
    )
    p.add_argument(
        "--out-dir", type=Path, default=DEFAULT_STAGING, help="staging output directory."
    )
    p.add_argument(
        "--cohort-only", action="store_true", help="subset matrix rows to the calibration cohort."
    )

    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--jsonl",
        type=Path,
        help="(0) fully de-blinded verdicts JSONL (tutor_model per row; no mapping needed).",
    )
    source.add_argument("--case-index", type=Path, help="(a) private case_index.jsonl (PREFERRED).")
    source.add_argument(
        "--merged-dir", type=Path, help="(b) calibration_merged_v2/ dir (reuse matrix or index)."
    )
    source.add_argument(
        "--id-key", type=str, help="(c) blinding secret hex (reproduce response_ids)."
    )
    source.add_argument(
        "--id-key-file", type=Path, help="(c) file holding the blinding secret hex."
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if (args.id_key is not None or args.id_key_file is not None) and args.responses is None:
        print("ERROR: --responses is required with --id-key/--id-key-file", file=sys.stderr)
        return 2
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
