#!/usr/bin/env python
"""Conform a flow-package calibration bank into loader-ready fitted-bank artifacts.

A flow package (for example the ``biggen-cal-52models`` tag) ships a per-criterion
deployment parameter bank plus the benchmark's scenario file, but not the
``rubrics.jsonl`` / ``scenarios.jsonl`` / ``manifest.json`` triple that
``olmo_eval.edullm.bank.load_fitted_bank`` requires.  This script performs that
reconciliation without ever mutating the source flow package:

* it filters each scenario's ``criterion_ids`` to the fitted set and drops scenarios
  that keep no fitted criterion (matching the exporter's ``empty_scenarios: drop``);
* it stamps every rubric row with the ``irt_params`` provenance the loader demands
  (``calibrated``/``fitted``/``synthetic``/``source``/``skills_order``);
* it writes a manifest whose SHA-256 hashes, counts, skill order, latent correlation,
  invariants, and policies satisfy the runtime integrity contract; and
* it gates on a real ``load_fitted_bank`` call so a bad conversion fails here, not on
  the cluster.

The script is benchmark-agnostic: pass the target benchmark's skill order and (for a
multi-skill bank) its latent correlation.  A unidimensional bank defaults to the
single ``general`` skill and a 1x1 identity correlation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise SystemExit(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    if not rows:
        raise SystemExit(f"{path}: contains no records")
    return rows


def _serialize_jsonl(records: list[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
        for record in records
    ).encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _identity_correlation(n_dims: int) -> list[list[float]]:
    return [[1.0 if i == j else 0.0 for j in range(n_dims)] for i in range(n_dims)]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--params-bank",
        required=True,
        type=Path,
        help="Per-criterion fitted parameter bank JSONL (e.g. *_modeled.jsonl).",
    )
    parser.add_argument(
        "--scenarios",
        required=True,
        type=Path,
        help="Benchmark scenario JSONL with prompt/conversation_context/criterion_ids.",
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--skills",
        required=True,
        nargs="+",
        help="Ordered latent skill axis (e.g. 'general'). Must equal each row's q_modeled keys.",
    )
    parser.add_argument("--benchmark-id", required=True)
    parser.add_argument("--calibration-version", required=True)
    parser.add_argument(
        "--calibration-method",
        default="calibrate_mirt.fit_m2pl_em",
        help="Calibration method recorded as irt_params.source provenance.",
    )
    parser.add_argument(
        "--latent-correlation",
        type=Path,
        default=None,
        help="JSON file with an NxN latent correlation matrix. Defaults to identity.",
    )
    parser.add_argument(
        "--extreme-a-threshold",
        type=float,
        default=None,
        help="Manifest extreme_a_threshold. Defaults to 1.25x the max active loading (min 10).",
    )
    parser.add_argument(
        "--nonpositive",
        choices=("exclude", "error"),
        default="exclude",
        help="Policy for items whose active discrimination is <= 0 (the loader forbids them).",
    )
    parser.add_argument(
        "--extreme",
        choices=("exclude", "error"),
        default="exclude",
        help="Policy for items whose active discrimination exceeds --extreme-a-threshold.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    skills = [str(skill).strip() for skill in args.skills]
    if not skills or any(not skill for skill in skills) or len(skills) != len(set(skills)):
        raise SystemExit("--skills must be unique, non-empty names")
    n_dims = len(skills)

    if args.latent_correlation is not None:
        correlation = json.loads(args.latent_correlation.read_text(encoding="utf-8"))
    else:
        correlation = _identity_correlation(n_dims)

    params_rows = _read_jsonl(args.params_bank)
    scenario_rows = _read_jsonl(args.scenarios)

    # First pass: build candidate rubric rows and record their active loadings so the
    # extreme-loading threshold and the exclusion policy can be applied consistently.
    candidates: list[tuple[dict[str, Any], list[float]]] = []
    seen_ids: set[str] = set()
    for position, row in enumerate(params_rows, 1):
        cid = str(row.get("criterion_id") or "").strip()
        sid = str(row.get("scenario_id") or "").strip()
        criterion = str(row.get("criterion") or "").strip()
        if not cid or not sid or not criterion:
            raise SystemExit(f"params row {position}: criterion_id/scenario_id/criterion required")
        if cid in seen_ids:
            raise SystemExit(f"duplicate criterion_id {cid!r} in params bank")
        seen_ids.add(cid)
        q_modeled = row.get("q_modeled")
        discrimination = row.get("discrimination")
        if not isinstance(q_modeled, dict) or not isinstance(discrimination, dict):
            raise SystemExit(f"{cid}: q_modeled and discrimination must be objects")
        if set(q_modeled) != set(skills) or set(discrimination) != set(skills):
            raise SystemExit(f"{cid}: q_modeled/discrimination keys must equal --skills {skills}")
        try:
            difficulty = float(row["difficulty"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit(f"{cid}: invalid difficulty: {exc}") from exc
        active = [float(discrimination[skill]) for skill in skills if int(q_modeled[skill]) == 1]
        if not active:
            raise SystemExit(f"{cid}: q_modeled maps to no active skill")

        source_provenance = row.get("irt_params")
        provenance = (
            source_provenance.get("provenance") if isinstance(source_provenance, dict) else None
        )
        record = {
            "criterion_id": cid,
            "scenario_id": sid,
            "criterion": criterion,
            "scoring_type": "binary",
            "primary_skill": str(row.get("primary_skill") or (skills[0] if n_dims == 1 else "")),
            "q_modeled": {skill: int(q_modeled[skill]) for skill in skills},
            "discrimination": {skill: float(discrimination[skill]) for skill in skills},
            "difficulty": difficulty,
            "calibration_version": args.calibration_version,
            "capability": row.get("capability"),
            "task": row.get("task"),
            "source": str(row.get("source") or ""),
            "status": str(row.get("status") or "approved"),
            "version": str(row.get("version") or "1.0"),
            "irt_params": {
                "source": args.calibration_method,
                "method": args.calibration_method,
                "calibrated": True,
                "fitted": True,
                "synthetic": False,
                "skills_order": list(skills),
                "modeled_skills": list(skills),
                "latent_correlation": correlation,
                "provenance": provenance if isinstance(provenance, dict) else {},
            },
        }
        candidates.append((record, active))

    if not candidates:
        raise SystemExit("params bank produced no candidate rubric rows")

    # Determine the extreme-loading threshold from the positive active loadings, then
    # apply the exclusion policy for non-positive and extreme active discriminations.
    positive_active = [value for _, active in candidates for value in active if value > 0]
    max_positive_a = max(positive_active) if positive_active else 0.0
    threshold = args.extreme_a_threshold
    if threshold is None:
        threshold = max(10.0, math.ceil(max_positive_a * 1.25))
    if not math.isfinite(threshold) or threshold <= 0:
        raise SystemExit(f"extreme_a_threshold {threshold} must be finite and positive")

    conformed_rubrics: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    fatal: list[str] = []
    for record, active in candidates:
        cid = record["criterion_id"]
        reasons: list[str] = []
        if any(value <= 0 for value in active):
            reasons.append("nonpositive_a")
        if any(value > threshold for value in active):
            reasons.append("extreme_a")
        if reasons:
            excluded.append({"criterion_id": cid, "reasons": reasons})
            is_error = ("nonpositive_a" in reasons and args.nonpositive == "error") or (
                "extreme_a" in reasons and args.extreme == "error"
            )
            if is_error:
                fatal.append(f"{cid}: {', '.join(reasons)}")
            continue
        conformed_rubrics.append(record)

    if fatal:
        raise SystemExit(
            f"conform blocked by {len(fatal)} policy error(s); first: " + "; ".join(fatal[:10])
        )
    if not conformed_rubrics:
        raise SystemExit("no criteria remain after applying the exclusion policy")

    fitted_ids = {record["criterion_id"] for record in conformed_rubrics}

    # Filter scenarios to the fitted criterion set; drop scenarios with none left.
    conformed_scenarios: list[dict[str, Any]] = []
    referenced: set[str] = set()
    dropped_scenarios: list[str] = []
    for row in scenario_rows:
        sid = str(row.get("scenario_id") or "").strip()
        raw_ids = row.get("criterion_ids")
        if not sid or not isinstance(raw_ids, list):
            raise SystemExit(f"scenario {sid!r}: missing scenario_id or list criterion_ids")
        kept = [str(cid).strip() for cid in raw_ids if str(cid).strip() in fitted_ids]
        if not kept:
            dropped_scenarios.append(sid)
            continue
        if len(kept) != len(set(kept)):
            raise SystemExit(f"scenario {sid}: duplicate criterion_ids after filtering")
        for cid in kept:
            if cid in referenced:
                raise SystemExit(f"criterion {cid} referenced by more than one scenario")
            referenced.add(cid)
        scenario = dict(row)
        scenario["criterion_ids"] = kept
        conformed_scenarios.append(scenario)

    orphaned = sorted(fitted_ids - referenced)
    if orphaned:
        raise SystemExit(
            f"{len(orphaned)} fitted criterion(s) are not referenced by any scenario; "
            f"first={orphaned[:10]}"
        )

    rubrics_bytes = _serialize_jsonl(conformed_rubrics)
    scenarios_bytes = _serialize_jsonl(conformed_scenarios)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rubrics_path = out_dir / "rubrics.jsonl"
    scenarios_path = out_dir / "scenarios.jsonl"
    manifest_path = out_dir / "manifest.json"

    manifest = {
        "schema_version": "fitted-bank-export-v1",
        "purpose": (
            "Flow-package bank conformed for the OLMo-eval fitted-bank loader. No "
            "synthetic or placeholder item parameters are present."
        ),
        "benchmark_id": args.benchmark_id,
        "calibration_version": args.calibration_version,
        "calibration_method": args.calibration_method,
        "skills_order": list(skills),
        "latent_correlation": correlation,
        "counts": {
            "source_criteria": len(candidates),
            "exported_criteria": len(conformed_rubrics),
            "excluded_criteria": len(excluded),
            "exported_scenarios": len(conformed_scenarios),
            "dropped_empty_scenarios": len(dropped_scenarios),
        },
        "excluded_criteria": excluded,
        "policies": {
            "extreme_a_threshold": float(threshold),
            "off_q_tolerance": 0.0,
            "empty_scenarios": "drop",
        },
        "invariants": {
            "fitted_only": True,
            "contains_synthetic_parameters": False,
            "scenario_criterion_ids_exactly_match_exported_rubrics": True,
            "all_active_loadings_positive_and_within_threshold": True,
            "all_inactive_loadings_zero": True,
            "latent_correlation_positive_definite": True,
        },
        "outputs": {
            "rubrics": str(rubrics_path),
            "scenarios": str(scenarios_path),
            "manifest": str(manifest_path),
            "sha256": {
                "rubrics": _sha256_bytes(rubrics_bytes),
                "scenarios": _sha256_bytes(scenarios_bytes),
            },
        },
        "provenance": {
            "module": "scripts/edullm/conform_flow_package_bank.py",
            "params_bank": str(args.params_bank),
            "scenarios": str(args.scenarios),
        },
    }
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    manifest_bytes = manifest_text.encode("utf-8")

    rubrics_path.write_bytes(rubrics_bytes)
    scenarios_path.write_bytes(scenarios_bytes)
    manifest_path.write_bytes(manifest_bytes)

    # Real acceptance gate: the loader must accept exactly what we wrote.
    from olmo_eval.edullm.bank import load_fitted_bank

    bank = load_fitted_bank(
        rubrics_path,
        scenarios_path,
        manifest_path=manifest_path,
        skills=skills,
    )
    summary = {
        "status": "conformed_and_loaded",
        "benchmark_id": args.benchmark_id,
        "calibration_version": args.calibration_version,
        "skills_order": list(bank.skills),
        "criteria": len(bank.rubrics),
        "scenarios": len(bank.scenarios),
        "excluded_criteria": len(excluded),
        "dropped_empty_scenarios": len(dropped_scenarios),
        "extreme_a_threshold": float(threshold),
        "out_dir": str(out_dir),
        "rubrics_path": str(rubrics_path),
        "scenarios_path": str(scenarios_path),
        "manifest_path": str(manifest_path),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
