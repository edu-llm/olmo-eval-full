"""Re-express each EduBench rubric criterion's q_mapping over the 4 skill groups
defined in manual_groups_qmat.csv, instead of the original 9 task-type slugs.

Each criterion instantiates one of the 12 EduBench metrics (the abbreviation in its
first line, e.g. "Criteria: ... (IFTC)."). manual_groups_qmat.csv gives, per metric,
a 0/1 over the 4 groups (academic assistance, emotional assistance,
checking-student-work, content generation). This script sets every criterion's
q_mapping to its metric's row from that CSV, so the rubric bank's q-matrix matches
the grouped q-matrix exactly.

Only `q_mapping` (value replaced, key position preserved) and `q_rationale` (updated
to name the CSV) change. No key is added/removed; all other fields, including the
null `difficulty`/`discrimination`, are untouched. Idempotent: re-running detects the
already-grouped axis and reproduces the same result. --write to apply; default dry run.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UNMERGED = ROOT / "data" / "EduBench" / "augmented_qmat" / "unmerged"
GROUPS_CSV = UNMERGED / "manual_groups_qmat.csv"
RUBRICS_JSONL = UNMERGED / "rubrics.jsonl"
RUBRICS_JSON = RUBRICS_JSONL.with_suffix(".json")
Q_RATIONALE = "manual_groups_qmat.csv (EduBench Table 7 distilled to 4 skill groups)"

TITLE_RE = re.compile(r"^Criteria: (?P<title>.+?) \((?P<abbr>[A-Z]+)\)\.$")


def load_groups(path: Path) -> tuple[list[str], dict[str, dict[str, int]]]:
    """(group order, {metric_abbr: {group: 0/1}}) from manual_groups_qmat.csv."""
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    groups = rows[0][1:]  # drop the "metric" header cell
    table = {r[0]: {g: int(v) for g, v in zip(groups, r[1:])} for r in rows[1:]}
    return groups, table


def metric_abbr(criterion: str) -> str | None:
    m = TITLE_RE.match(criterion.split("\n", 1)[0])
    return m.group("abbr") if m else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true", help="apply changes (default: dry run)")
    args = ap.parse_args()

    groups, table = load_groups(GROUPS_CSV)
    print(f"groups ({len(groups)}): {groups}")
    print(f"metrics in CSV ({len(table)}): {sorted(table)}\n")

    records = [json.loads(l) for l in RUBRICS_JSONL.open(encoding="utf-8")]

    unknown_metric, mismatch, changed, already = [], 0, 0, 0
    seen_metrics: set[str] = set()
    for r in records:
        abbr = metric_abbr(r["criterion"])
        if abbr is None or abbr not in table:
            unknown_metric.append((r["criterion_id"], abbr))
            continue
        seen_metrics.add(abbr)
        want = table[abbr]  # {group: 0/1} in CSV order
        cur = r.get("q_mapping")
        if cur == want:
            already += 1
        else:
            changed += 1
        if args.write:
            r["q_mapping"] = dict(want)
            r["q_rationale"] = Q_RATIONALE

    print(f"rubric criteria: {len(records)}")
    print(f"  metrics covered: {len(seen_metrics)}/{len(table)}"
          + (f"  MISSING {sorted(set(table) - seen_metrics)}" if set(table) - seen_metrics else ""))
    print(f"  would change: {changed}   already grouped: {already}")
    print(f"  unknown/unparseable metric: {len(unknown_metric)}"
          + (f" -> {unknown_metric[:10]}" if unknown_metric else ""))

    # Verify every metric's produced q_mapping equals its CSV row (spot the 12 distinct rows).
    produced = {}
    for r in records:
        abbr = metric_abbr(r["criterion"])
        if abbr in table:
            produced.setdefault(abbr, table[abbr])
    print("\nper-metric grouped q_mapping (what each criterion will carry):")
    for abbr in sorted(produced):
        vec = "".join(str(produced[abbr][g]) for g in groups)
        print(f"  {abbr:<5} {vec}  {produced[abbr]}")

    if not args.write:
        print("\n--dry-run: nothing written (pass --write to apply)")
        return 0
    if unknown_metric:
        raise SystemExit("refusing to write: some criteria have an unknown metric (see above)")

    with RUBRICS_JSONL.open("w", encoding="utf-8", newline="\n") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with RUBRICS_JSON.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(f"\nwrote {RUBRICS_JSONL.name} and {RUBRICS_JSON.name} ({len(records)} criteria)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
