"""Export a fitted-only rubric bank whose parameters match the calibration CSV exactly.

The published ``rubrics_qmatrix_calibrated_{2,3}skill.jsonl`` banks cover all 6,845
criteria: fitted items carry calibrated parameters, and the remainder fall back to
pre-calibration synthetic values. They also keep the legacy ``{content, diagnosis,
scaffolding}`` discrimination slots, whose meaning differs per skill-set, and they floor
negative loadings at zero. Those three properties together make the files easy to
misread.

This exporter emits a companion bank that removes the ambiguity:

* **fitted items only** -- every record comes from the calibration CSV, so there are no
  synthetic fallbacks mixed in;
* **true skill names** -- ``discrimination`` is keyed by the modeled skills, and
  ``q_modeled`` restates the Q-matrix in the same space (``q_mapping`` is preserved
  unchanged for provenance);
* **CSV-exact values** -- negative loadings are preserved rather than floored, so the
  file agrees with what the CAT harness actually reads.

Usage
-----
    python scripts/export_calibrated_bank.py --skills 2
    python scripts/export_calibrated_bank.py --skills 3
    python scripts/export_calibrated_bank.py --skills 2 --verify-only
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# a_* column in the calibration CSV -> modeled skill name.
SKILLSET = {
    2: {
        "dims": ("correctness", "scaffolding"),
        "csv": ROOT / "staging" / "calibration_mirt_full2skill.csv",
        "a_cols": ("a_correctness", "a_scaffolding"),
        "base": ROOT / "data" / "TutorBench" / "rubrics_qmatrix_calibrated_2skill.jsonl",
        "out": ROOT / "data" / "TutorBench" / "rubrics_qmatrix_calibrated_2skill_fitted.jsonl",
        # correctness is the content+diagnosis collapse; scaffolding passes through.
        "q_source": None,
        "q_note": "correctness = content OR diagnosis (collapse); scaffolding as-is",
    },
    3: {
        "dims": ("correctness", "scaffolding", "presentation"),
        "csv": ROOT / "staging" / "run6_presentation" / "calibration_mirt.csv",
        "a_cols": ("a_content", "a_diagnosis", "a_scaffolding"),
        "base": ROOT / "data" / "TutorBench" / "rubrics_qmatrix_calibrated_3skill.jsonl",
        "out": ROOT / "data" / "TutorBench" / "rubrics_qmatrix_calibrated_3skill_fitted.jsonl",
        # Run 6 fitted the repurposed Q; read it back rather than re-deriving it.
        "q_source": ROOT / "data" / "TutorBench" / "experimental"
        / "rubrics_qmatrix_collapse_presentation.jsonl",
        "q_note": "Run 6 slot repurposing: content->correctness, diagnosis->scaffolding, "
        "scaffolding->presentation",
    },
}

LEGACY_SLOTS = ("content", "diagnosis", "scaffolding")


def read_jsonl(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = json.loads(line)
                out[str(rec["criterion_id"])] = rec
    return out


def read_csv_bank(path: Path) -> dict[str, dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return {str(r["criterion_id"]): r for r in csv.DictReader(fh)}


def modeled_q(skills: int, legacy_q: dict, repurposed_q: dict | None) -> dict[str, int]:
    """Restate the Q-matrix row in the modeled skill space."""
    if skills == 2:
        return {
            "correctness": int(bool(legacy_q.get("content", 0)) or bool(legacy_q.get("diagnosis", 0))),
            "scaffolding": int(bool(legacy_q.get("scaffolding", 0))),
        }
    q = repurposed_q if repurposed_q is not None else legacy_q
    return {
        "correctness": int(bool(q.get("content", 0))),
        "scaffolding": int(bool(q.get("diagnosis", 0))),
        "presentation": int(bool(q.get("scaffolding", 0))),
    }


def build_records(skills: int, csv_path: Path | None = None) -> tuple[list[dict], dict]:
    cfg = SKILLSET[skills]
    dims = cfg["dims"]
    base = read_jsonl(cfg["base"])
    bank = read_csv_bank(csv_path or cfg["csv"])
    qsrc = read_jsonl(cfg["q_source"]) if cfg["q_source"] else {}

    provenance = (base[next(iter(base))].get("irt_params") or {}).get("provenance", {})

    missing_meta = [cid for cid in bank if cid not in base]
    missing_q = [cid for cid in bank if cfg["q_source"] and cid not in qsrc]

    records: list[dict] = []
    n_negative = 0
    for cid, row in bank.items():
        src = base.get(cid, {})
        rec = {k: v for k, v in src.items() if k not in ("difficulty", "discrimination", "irt_params")}

        disc = {}
        for col, dim in zip(cfg["a_cols"], dims, strict=True):
            val = float(row[col])
            if val < 0:
                n_negative += 1
            disc[dim] = val

        rec["difficulty"] = float(row["b"])
        rec["discrimination"] = disc
        rec["q_modeled"] = modeled_q(
            skills, src.get("q_mapping") or {}, (qsrc.get(cid) or {}).get("q_mapping")
        )
        flags = [f for f in (row.get("flags") or "").split("|") if f]
        rec["irt_params"] = {
            "source": f"calibrated-m2pl-{skills}skill-pilot",
            "method": "confirmatory-m2pl-mml-em",
            "calibrated": True,
            "fitted": True,
            "modeled_skills": list(dims),
            "n_persons": int(row["n_persons"]),
            "flags": flags,
            "version": "1.0",
            "provenance": {
                **provenance,
                "exported_by": "scripts/export_calibrated_bank.py",
                "csv_exact": True,
                "negative_loadings": "preserved (NOT floored to 0, unlike the full bank)",
                "q_modeled_derivation": cfg["q_note"],
                "legacy_slot_note": (
                    "`q_mapping` is the untouched legacy {content,diagnosis,scaffolding} row; "
                    "`q_modeled` and `discrimination` use the modeled skill names."
                ),
            },
        }
        records.append(rec)

    stats = {
        "n_records": len(records),
        "n_csv": len(bank),
        "n_negative_loadings": n_negative,
        "n_missing_metadata": len(missing_meta),
        "n_missing_q_source": len(missing_q),
        "n_extreme_a": sum(1 for r in records if "extreme_a" in r["irt_params"]["flags"]),
    }
    return records, stats


def verify(skills: int, records: list[dict], csv_path: Path | None = None) -> list[str]:
    """Re-read the CSV and assert every emitted value matches it exactly."""
    cfg = SKILLSET[skills]
    bank = read_csv_bank(csv_path or cfg["csv"])
    problems: list[str] = []
    if len(records) != len(bank):
        problems.append(f"record count {len(records)} != CSV rows {len(bank)}")
    for rec in records:
        cid = rec["criterion_id"]
        row = bank.get(cid)
        if row is None:
            problems.append(f"{cid}: not in CSV")
            continue
        for col, dim in zip(cfg["a_cols"], cfg["dims"], strict=True):
            if abs(float(row[col]) - float(rec["discrimination"][dim])) > 1e-9:
                problems.append(f"{cid}: {dim} {rec['discrimination'][dim]} != CSV {row[col]}")
        if abs(float(row["b"]) - float(rec["difficulty"])) > 1e-9:
            problems.append(f"{cid}: b {rec['difficulty']} != CSV {row['b']}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skills", type=int, choices=(2, 3), required=True)
    ap.add_argument("--csv", type=Path, default=None,
                    help="override the calibration CSV (e.g. an alternate refit).")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--verify-only", action="store_true",
                    help="build and check against the CSV without writing.")
    args = ap.parse_args()

    cfg = SKILLSET[args.skills]
    csv_path = args.csv or cfg["csv"]
    out_path = args.out or cfg["out"]

    records, stats = build_records(args.skills, csv_path)
    problems = verify(args.skills, records, csv_path)

    print(f"skills           : {args.skills} {list(cfg['dims'])}")
    print(f"csv              : {csv_path}")
    print(f"records          : {stats['n_records']} (CSV rows {stats['n_csv']})")
    print(f"negative loadings: {stats['n_negative_loadings']} (preserved)")
    print(f"extreme_a flagged: {stats['n_extreme_a']}")
    if stats["n_missing_metadata"]:
        print(f"WARNING: {stats['n_missing_metadata']} CSV ids had no metadata in the base bank")
    if stats["n_missing_q_source"]:
        print(f"WARNING: {stats['n_missing_q_source']} CSV ids missing from the Q source")

    if problems:
        print(f"\nVERIFY FAILED: {len(problems)} problem(s)")
        for p in problems[:10]:
            print("  ", p)
        return 1
    print("verify           : OK (every value matches the CSV exactly)")

    if args.verify_only:
        return 0

    out_path = out_path if out_path.is_absolute() else (ROOT / out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote            : {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
