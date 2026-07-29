"""Audit the teammate's judge verdict JSONL shards BEFORE ingestion.

This is a read-only reporting pass over the raw verdict rows emitted by
``aws_judge_handoff/scripts/run_judge_validation.py`` (subcommand ``run``). It
answers the questions we need before building the MIRT response matrix:

  * How many cells were judged, decided (pass/fail), and VOIDED (no_decision)?
  * What CAUSED the voids? The frozen run applies a consistency gate that voids a
    cell to ``no_decision`` when the two verdict signals -- the NATIVE free-text
    verdict (pass/fail parsed from the judge's prose) and the single-token P/F
    verdict -- disagree, or when the generation was truncated
    (``finish_reason == "length"``). We bucket:
        - native=pass / P/F=fail        (the dominant disagreement direction)
        - native=fail / P/F=pass        (the rare reverse)
        - truncation (finish_reason)    (output ran to the token cap)
        - truncation AND disagreement   (both fire on the same cell)
  * The ``finish_reason`` distribution, and per-shard / per-model no_decision
    counts (per-model requires either a ``model`` field on the rows or a
    ``cases_index.jsonl`` to de-blind ``case_id`` -> model).
  * Which CRITERIA SET the verdicts belong to: the curated bank
    (``data/curated/rubrics_qmatrix_curated.jsonl``) vs the reorganized final
    bank (``data/<benchmark>/rubrics_qmatrix_final.jsonl``), reported as a match
    percentage for each plus any unknown criterion_ids.

Outputs ``staging/audit_judge_verdicts.json`` (full report) and
``staging/audit_judge_verdicts.csv`` (flat summary).

IMPORTANT -- schema assumptions (confirm against the REAL shard files):
The canonical runner in this repo emits ``case_id, criterion_id, verdict,
native_score, status, error, raw_output`` and does NOT (in this checkout) carry
explicit ``native_verdict`` / ``pf_verdict`` / ``finish_reason`` / ``shard_id``
fields -- those come from the evolved two-signal consistency gate used for the
real run. All field names are therefore CONFIGURABLE via flags; the defaults
below are the best guess. If a signal field is absent, the audit reports the
buckets it can and clearly flags the ones it cannot compute rather than guessing.
When ``native_verdict`` is absent it falls back to ``native_score`` (1->pass,
0->fail). Pure standard library. Read-only. Deterministic.
"""

from __future__ import annotations

import argparse
import csv
import glob as globmod
import json
import sys
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_STAGING = ROOT / "staging"
DEFAULT_CURATED = ROOT / "data" / "curated" / "rubrics_qmatrix_curated.jsonl"
DEFAULT_OUT_JSON = DEFAULT_STAGING / "audit_judge_verdicts.json"
DEFAULT_OUT_CSV = DEFAULT_STAGING / "audit_judge_verdicts.csv"

PASS = "pass"
FAIL = "fail"


def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def expand_inputs(patterns: list[str]) -> list[Path]:
    """Resolve files, directories (``*.jsonl`` inside), and globs to a sorted list."""
    out: list[Path] = []
    for raw in patterns:
        p = Path(raw)
        if p.is_dir():
            out.extend(sorted(p.glob("*.jsonl")))
        elif any(ch in raw for ch in "*?["):
            out.extend(sorted(Path(m) for m in globmod.glob(raw, recursive=True)))
        else:
            out.append(p)
    # de-dup, keep order
    seen: set = set()
    uniq: list[Path] = []
    for p in out:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    return uniq


def find_final_rubrics(explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit
    matches = sorted(ROOT.glob("data/**/rubrics_qmatrix_final.jsonl"))
    return matches[0] if matches else None


def load_criterion_ids(path: Path | None, field: str) -> set:
    ids: set = set()
    if path is None or not path.is_file():
        return ids
    for row in iter_jsonl(path):
        cid = row.get(field)
        if cid:
            ids.add(str(cid))
    return ids


def load_case_to_model(index_path: Path | None, case_field: str) -> dict:
    mapping: dict[str, str] = {}
    if index_path is None or not index_path.is_file():
        return mapping
    for row in iter_jsonl(index_path):
        cid = str(row.get(case_field, "")).strip()
        model = row.get("model")
        if cid and model is not None:
            mapping[cid] = str(model)
    return mapping


def to_passfail(value: Any) -> str | None:
    """Normalize a native/PF signal to 'pass'/'fail'/None (handles str + int)."""
    if value is None:
        return None
    if isinstance(value, bool):
        return PASS if value else FAIL
    if isinstance(value, (int, float)):
        return PASS if int(value) == 1 else FAIL
    s = str(value).strip().lower()
    if s in (PASS, "yes", "true", "1"):
        return PASS
    if s in (FAIL, "no", "false", "0"):
        return FAIL
    return None


def shard_of(row: dict, path: Path, shard_field: str) -> str:
    val = row.get(shard_field)
    if val:
        return str(val)
    return path.stem


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("verdicts", nargs="+",
                    help="verdict JSONL file(s), directory(ies) of *.jsonl, or glob(s)")
    ap.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    ap.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    # configurable field names (defaults = best guess at the real schema)
    ap.add_argument("--case-field", default="case_id")
    ap.add_argument("--criterion-field", default="criterion_id")
    ap.add_argument("--verdict-field", default="verdict")
    ap.add_argument("--status-field", default="status")
    ap.add_argument("--native-field", default="native_verdict",
                    help="free-text-derived native pass/fail signal")
    ap.add_argument("--native-score-field", default="native_score",
                    help="fallback int native signal (1=pass,0=fail) when --native-field absent")
    ap.add_argument("--pf-field", default="pf_verdict",
                    help="single-token P/F verdict signal")
    ap.add_argument("--finish-reason-field", default="finish_reason")
    ap.add_argument("--model-field", default="model")
    ap.add_argument("--shard-field", default="shard_id",
                    help="shard id field; falls back to the verdict file stem")
    ap.add_argument("--truncation-value", default="length",
                    help="finish_reason value that marks a truncation void")
    ap.add_argument("--no-decision-values", nargs="+", default=["no_decision"],
                    help="verdict values that count as a void")
    ap.add_argument("--cases-index", type=Path, default=None,
                    help="staging/cases_index.jsonl to de-blind case_id->model "
                         "when rows carry no model field")
    ap.add_argument("--curated-rubrics", type=Path, default=DEFAULT_CURATED)
    ap.add_argument("--final-rubrics", type=Path, default=None,
                    help="default: first data/**/rubrics_qmatrix_final.jsonl found")
    return ap


def audit(args: argparse.Namespace) -> dict:
    paths = expand_inputs(args.verdicts)
    missing = [str(p) for p in paths if not p.is_file()]
    if missing or not paths:
        raise FileNotFoundError(f"verdict file(s) not found: {missing or args.verdicts}")

    nd_values = {v.strip().lower() for v in args.no_decision_values}
    trunc_val = str(args.truncation_value).strip().lower()
    case_to_model = load_case_to_model(args.cases_index, args.case_field)

    total = 0
    decided = 0
    no_decision = 0
    finish_all: Counter = Counter()
    finish_void: Counter = Counter()
    per_shard_total: Counter = Counter()
    per_shard_void: Counter = Counter()
    per_model_void: Counter = Counter()
    bucket = {
        "native_pass_pf_fail": 0,
        "native_fail_pf_pass": 0,
        "truncation": 0,
        "truncation_and_disagreement": 0,
    }
    verdict_criteria: set = set()
    native_present = False
    native_score_present = False
    pf_present = False
    finish_present = False
    model_resolvable = False
    void_unclassified = 0

    for path in paths:
        for row in iter_jsonl(path):
            total += 1
            shard = shard_of(row, path, args.shard_field)
            per_shard_total[shard] += 1

            cid = row.get(args.criterion_field)
            if cid:
                verdict_criteria.add(str(cid))

            finish_raw = row.get(args.finish_reason_field)
            if finish_raw is not None:
                finish_present = True
                finish_all[str(finish_raw).strip().lower()] += 1
            finish = str(finish_raw).strip().lower() if finish_raw is not None else None

            verdict = str(row.get(args.verdict_field, "")).strip().lower()
            status = str(row.get(args.status_field, "") or "").strip().lower()

            is_void = (verdict in nd_values) or (status not in ("", "ok"))
            if verdict in (PASS, FAIL) and status in ("", "ok"):
                is_void = False

            if not is_void:
                decided += 1
                continue

            # ---- this cell is a VOID (no_decision) ----
            no_decision += 1
            per_shard_void[shard] += 1
            if finish is not None:
                finish_void[finish] += 1

            # per-model
            model = row.get(args.model_field)
            if model is None and case_to_model:
                model = case_to_model.get(str(row.get(args.case_field, "")).strip())
            if model is not None:
                model_resolvable = True
                per_model_void[str(model)] += 1

            # signals
            native_raw = row.get(args.native_field)
            if native_raw is not None:
                native_present = True
            native = to_passfail(native_raw)
            if native is None:
                score_raw = row.get(args.native_score_field)
                if score_raw is not None:
                    native_score_present = True
                    native = to_passfail(score_raw)
            pf_raw = row.get(args.pf_field)
            if pf_raw is not None:
                pf_present = True
            pf = to_passfail(pf_raw)

            is_trunc = finish == trunc_val
            disagree_pf_fail = native == PASS and pf == FAIL
            disagree_pf_pass = native == FAIL and pf == PASS
            is_disagreement = disagree_pf_fail or disagree_pf_pass

            if is_trunc:
                bucket["truncation"] += 1
            if disagree_pf_fail:
                bucket["native_pass_pf_fail"] += 1
            if disagree_pf_pass:
                bucket["native_fail_pf_pass"] += 1
            if is_trunc and is_disagreement:
                bucket["truncation_and_disagreement"] += 1
            if not is_trunc and not is_disagreement:
                void_unclassified += 1

    # ---- criteria-set match ----
    curated_ids = load_criterion_ids(args.curated_rubrics, args.criterion_field)
    final_path = find_final_rubrics(args.final_rubrics)
    final_ids = load_criterion_ids(final_path, args.criterion_field)

    def match_stats(bank_ids: set) -> dict:
        if not verdict_criteria or not bank_ids:
            return {"n_bank": len(bank_ids), "n_matched": 0, "match_pct": 0.0}
        matched = len(verdict_criteria & bank_ids)
        return {
            "n_bank": len(bank_ids),
            "n_matched": matched,
            "match_pct": round(100.0 * matched / len(verdict_criteria), 4),
        }

    curated_m = match_stats(curated_ids)
    final_m = match_stats(final_ids)
    unknown_ids = sorted(verdict_criteria - curated_ids - final_ids)
    if final_m["match_pct"] >= curated_m["match_pct"] and final_ids:
        best = "final"
    elif curated_ids:
        best = "curated"
    else:
        best = "unknown"

    pct = round(100.0 * no_decision / total, 4) if total else 0.0
    report = {
        "inputs": {
            "verdict_files": [str(p) for p in paths],
            "n_files": len(paths),
            "cases_index": str(args.cases_index) if args.cases_index else None,
        },
        "field_names": {
            "case": args.case_field, "criterion": args.criterion_field,
            "verdict": args.verdict_field, "status": args.status_field,
            "native": args.native_field, "native_score": args.native_score_field,
            "pf": args.pf_field, "finish_reason": args.finish_reason_field,
            "model": args.model_field, "shard": args.shard_field,
            "truncation_value": trunc_val,
            "no_decision_values": sorted(nd_values),
        },
        "signal_availability": {
            "native_verdict_field_present": native_present,
            "native_score_fallback_present": native_score_present,
            "pf_verdict_field_present": pf_present,
            "finish_reason_field_present": finish_present,
            "model_resolvable": model_resolvable,
        },
        "totals": {
            "total_cells": total,
            "decided": decided,
            "no_decision": no_decision,
            "no_decision_pct": pct,
        },
        "no_decision_cause_buckets": bucket,
        "no_decision_disagreement_total": (
            bucket["native_pass_pf_fail"] + bucket["native_fail_pf_pass"]
        ),
        "no_decision_unclassified": void_unclassified,
        "finish_reason_distribution_all": dict(finish_all.most_common()),
        "finish_reason_distribution_no_decision": dict(finish_void.most_common()),
        "per_shard": {
            s: {"total": per_shard_total[s], "no_decision": per_shard_void.get(s, 0)}
            for s in sorted(per_shard_total)
        },
        "per_model_no_decision": dict(per_model_void.most_common()),
        "criteria_set_match": {
            "n_verdict_criteria": len(verdict_criteria),
            "curated": {"path": str(args.curated_rubrics), **curated_m},
            "final": {"path": str(final_path) if final_path else None, **final_m},
            "best_match": best,
            "n_unknown_ids": len(unknown_ids),
            "unknown_ids_sample": unknown_ids[:50],
        },
        "warnings": _warnings(native_present, native_score_present, pf_present,
                              finish_present, model_resolvable, void_unclassified),
    }
    return report


def _warnings(native, native_score, pf, finish, model, unclassified) -> list[str]:
    w: list[str] = []
    if not (native or native_score):
        w.append("No native verdict signal found (--native-field/--native-score-field); "
                 "disagreement buckets cannot be computed.")
    if not pf:
        w.append("No P/F verdict signal found (--pf-field); disagreement buckets "
                 "cannot be computed.")
    if not finish:
        w.append("No finish_reason field found (--finish-reason-field); truncation "
                 "bucket cannot be computed.")
    if not model:
        w.append("No per-model signal (no model field and no usable --cases-index); "
                 "per-model no_decision counts are empty.")
    if unclassified:
        w.append(f"{unclassified} void cell(s) matched neither truncation nor a "
                 "native/PF disagreement (check field mappings against real data).")
    return w


def write_csv(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        t = report["totals"]
        for k in ("total_cells", "decided", "no_decision", "no_decision_pct"):
            w.writerow([k, t[k]])
        for k, v in report["no_decision_cause_buckets"].items():
            w.writerow([f"bucket.{k}", v])
        w.writerow(["no_decision_disagreement_total",
                    report["no_decision_disagreement_total"]])
        w.writerow(["no_decision_unclassified", report["no_decision_unclassified"]])
        for reason, n in report["finish_reason_distribution_all"].items():
            w.writerow([f"finish_reason.{reason}", n])
        cm = report["criteria_set_match"]
        w.writerow(["criteria.best_match", cm["best_match"]])
        w.writerow(["criteria.curated_match_pct", cm["curated"]["match_pct"]])
        w.writerow(["criteria.final_match_pct", cm["final"]["match_pct"]])
        w.writerow(["criteria.n_unknown_ids", cm["n_unknown_ids"]])
        for shard, d in report["per_shard"].items():
            w.writerow([f"shard.{shard}.no_decision", d["no_decision"]])


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = audit(args)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    write_csv(args.out_csv, report)

    t = report["totals"]
    print("=" * 72)
    print("audit-judge-verdicts")
    print("=" * 72)
    print(f"files            : {report['inputs']['n_files']}")
    print(f"total cells      : {t['total_cells']}")
    print(f"decided          : {t['decided']}")
    print(f"no_decision      : {t['no_decision']} ({t['no_decision_pct']}%)")
    b = report["no_decision_cause_buckets"]
    print("cause buckets    :")
    print(f"  native=pass/PF=fail        : {b['native_pass_pf_fail']}")
    print(f"  native=fail/PF=pass        : {b['native_fail_pf_pass']}")
    print(f"  truncation (finish=length) : {b['truncation']}")
    print(f"  truncation AND disagreement: {b['truncation_and_disagreement']}")
    cm = report["criteria_set_match"]
    print(f"criteria match   : best={cm['best_match']} "
          f"(final {cm['final']['match_pct']}%, curated {cm['curated']['match_pct']}%, "
          f"unknown {cm['n_unknown_ids']})")
    for warn in report["warnings"]:
        print(f"WARNING: {warn}", file=sys.stderr)
    print("-" * 72)
    print(f"wrote JSON -> {args.out_json}")
    print(f"wrote CSV  -> {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
