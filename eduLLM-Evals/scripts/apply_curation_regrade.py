#!/usr/bin/env python3
"""Propagate the curation-driven criteria split/rewording into the official
human-grade packets (grader_packets/grader_0N.csv AND grader_0N.md).

Curation renumbered/split/merged the rubric criteria for three scenarios
(tb_0003, tb_0336, tb_0340). The NEW numbering lives in
data/curated/rubrics_qmatrix_curated.jsonl and is authoritative. This script
rebuilds ONLY those three scenarios inside the affected packets, replacing the
criterion rows/blocks with the curated criteria (new ids / text / primary_skill
/ criticality) and filling grade+notes from the deterministic mapping and the
explicit derived-grade tables below.

The transform is idempotent: a scenario already carrying the curated criterion
ids + text is left untouched.

Usage:
    python scripts/apply_curation_regrade.py            # apply changes
    python scripts/apply_curation_regrade.py --dry-run  # report only, no writes
"""
from __future__ import annotations

import csv
import re
import shutil
import sys
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
PACKETS = ROOT / "grader_packets"
CURATED = ROOT / "data" / "curated" / "rubrics_qmatrix_curated.jsonl"

AFFECTED_SCENARIOS = ("tb_0003", "tb_0336", "tb_0340")

CSV_FIELDS = [
    "assignment_id", "scenario_id", "anonymous_tutor", "use_case", "subject",
    "criterion_id", "primary_skill", "criticality", "criterion", "grade", "notes",
]

# scenario -> packet stems that contain that scenario
PACKETS_BY_SCENARIO = {
    "tb_0003": ["grader_01", "grader_03", "grader_04"],
    "tb_0336": ["grader_01", "grader_02", "grader_04"],
    "tb_0340": ["grader_01", "grader_03", "grader_05"],
}

# Deterministic old->new mapping (values are OLD short criterion ids that port
# grade+notes 1:1 into the NEW short id key).
PORT_MAP = {
    "tb_0003": {
        "c01": "c01", "c02": "c02", "c03": "c03", "c04": "c04",
        "c05": "c05", "c06": "c06", "c07": "c07", "c08": "c08",
        "c12": "c10",
    },
    "tb_0336": {
        "c01": "c01", "c02": "c02",
        "c05": "c04", "c06": "c05", "c07": "c06", "c08": "c07", "c09": "c08",
    },
    "tb_0340": {
        "c01": "c01", "c02": "c02", "c03": "c03", "c04": "c04",
        "c05": "c05", "c06": "c06", "c07": "c07",
        "c10": "c09", "c11": "c10",
    },
}

# Split children (new short ids) whose grades come from the derived tables.
SPLIT_CHILDREN = {
    "tb_0003": ["c09", "c10", "c11"],
    "tb_0336": ["c03", "c04"],
    "tb_0340": ["c08", "c09"],
}

# Merged style/persona consolidation: new short id <- P only if BOTH old shorts
# were P; notes = concatenation of the two old notes (blanks dropped).
MERGED = {
    "tb_0336": {"new": "c10", "old": ["c09", "c10"]},
}

# Derived split-child notes.
N_0003 = "derived from original compound tb_0003_c09 human grade during curation regrade"
N_0336 = "derived from original compound tb_0336_c03 human grade during curation regrade (no tutor engaged part c)"
N_0340_C08 = "derived from original compound tb_0340_c08 during curation regrade (validate; reworded criterion)"
N_0340_C09 = "derived; redirect/scaffolding move (prompt vs tell)"

# Explicit derived grades keyed [stem][scenario][new_short] = (grade, notes).
DERIVED = {
    "grader_01": {
        "tb_0003": {"c09": ("P", N_0003), "c10": ("P", N_0003), "c11": ("P", N_0003)},
        "tb_0336": {"c03": ("F", N_0336), "c04": ("F", N_0336)},
        "tb_0340": {"c08": ("F", N_0340_C08), "c09": ("F", N_0340_C09)},
    },
    "grader_02": {
        "tb_0336": {"c03": ("F", N_0336), "c04": ("F", N_0336)},
    },
    "grader_03": {
        "tb_0003": {"c09": ("F", N_0003), "c10": ("F", N_0003), "c11": ("P", N_0003)},
        "tb_0340": {"c08": ("P", N_0340_C08), "c09": ("F", N_0340_C09)},
    },
    "grader_04": {
        "tb_0003": {"c09": ("P", N_0003), "c10": ("P", N_0003), "c11": ("P", N_0003)},
        "tb_0336": {"c03": ("F", N_0336), "c04": ("F", N_0336)},
    },
    "grader_05": {
        "tb_0340": {"c08": ("F", N_0340_C08), "c09": ("F", N_0340_C09)},
    },
}

BLANK_NOTE = "____"


def short_id(cid: str) -> str:
    return cid.rsplit("_", 1)[1]


def is_blank(value: str) -> bool:
    return value == "" or set(value) <= {"_"}


def load_curated() -> dict:
    """scenario_id -> ordered list of curated criterion dicts (by c-number)."""
    out: dict[str, list[dict]] = {s: [] for s in AFFECTED_SCENARIOS}
    with open(CURATED, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # Skip malformed lines elsewhere in the corpus; the affected
                # scenarios parse cleanly.
                continue
            sid = rec.get("scenario_id")
            if sid in out:
                out[sid].append(rec)
    for sid, recs in out.items():
        recs.sort(key=lambda r: short_id(r["criterion_id"]))
    return out


def scenario_is_migrated(scenario: str, curated: list[dict], old_by_id: dict) -> bool:
    """True if the packet already carries every curated criterion id + text."""
    for rec in curated:
        cid = rec["criterion_id"]
        old = old_by_id.get(cid)
        if old is None or old["criterion"] != rec["criterion"]:
            return False
    return True


def resolve_grade_notes(stem, scenario, new_short, curated_cid, old_by_id):
    """Return (grade, notes) for a curated criterion."""
    derived = DERIVED.get(stem, {}).get(scenario, {})
    if new_short in derived:
        return derived[new_short]

    merged = MERGED.get(scenario)
    if merged and new_short == merged["new"]:
        olds = [old_by_id.get(f"{scenario}_{s}") for s in merged["old"]]
        grades = [(o["grade"] if o else "").upper() for o in olds]
        grade = "P" if all(g == "P" for g in grades) else "F"
        parts = []
        for o in olds:
            n = o["notes"] if o else ""
            if not is_blank(n) and n not in parts:
                parts.append(n)
        notes = "; ".join(parts) if parts else BLANK_NOTE
        return grade, notes

    port = PORT_MAP.get(scenario, {})
    if new_short in port:
        old_full = f"{scenario}_{port[new_short]}"
        old = old_by_id.get(old_full) or old_by_id.get(curated_cid)
        grade = (old["grade"] if old else "").upper()
        notes = old["notes"] if old else ""
        if is_blank(notes):
            notes = BLANK_NOTE
        return grade, notes

    raise ValueError(f"No mapping for {scenario} {new_short} in {stem}")


def build_scenario_rows(stem, scenario, template, curated, old_by_id):
    """Build the replacement CSV rows for a scenario (curated ascending order)."""
    rows = []
    for rec in curated:
        cid = rec["criterion_id"]
        ns = short_id(cid)
        grade, notes = resolve_grade_notes(stem, scenario, ns, cid, old_by_id)
        rows.append({
            "assignment_id": template["assignment_id"],
            "scenario_id": template["scenario_id"],
            "anonymous_tutor": template["anonymous_tutor"],
            "use_case": template["use_case"],
            "subject": template["subject"],
            "criterion_id": cid,
            "primary_skill": rec.get("primary_skill") or "",
            "criticality": rec.get("criticality") or "",
            "criterion": rec["criterion"],
            "grade": grade,
            "notes": notes,
        })
    return rows


def render_md_block(row) -> str:
    return (
        f"#### {row['criterion_id']}\n\n"
        f"- Criterion: {row['criterion']}\n"
        f"- Primary skill: `{row['primary_skill']}`\n"
        f"- Criticality: `{row['criticality']}`\n"
        f"- Grade (P/F): {row['grade']}\n"
        f"- Notes: {row['notes']}\n"
    )


def render_md_section(rows) -> str:
    out = "### Criteria To Grade\n\n"
    for row in rows:
        out += render_md_block(row) + "\n"
    return out


CRITERIA_SECTION = re.compile(r"(?ms)^### Criteria To Grade\n.*?(?=^## |\Z)")
FIRST_CID = re.compile(r"(?m)^####\s+(\S+)\s*$")


def process_packet(stem, curated_by_scenario, dry_run):
    csv_path = PACKETS / f"{stem}.csv"
    md_path = PACKETS / f"{stem}.md"
    scenarios = [s for s in PACKETS_BY_SCENARIO if stem in PACKETS_BY_SCENARIO[s]]

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        orig_rows = list(reader)
    old_by_id = {r["criterion_id"]: r for r in orig_rows}

    # Decide which scenarios actually need processing (skip already-migrated).
    to_process = {}
    summary = []
    for scenario in scenarios:
        curated = curated_by_scenario[scenario]
        if scenario_is_migrated(scenario, curated, old_by_id):
            summary.append(f"    {scenario}: already migrated -> skipped")
            continue
        template = next(r for r in orig_rows if r["scenario_id"] == scenario)
        new_rows = build_scenario_rows(stem, scenario, template, curated, old_by_id)
        to_process[scenario] = new_rows

    if not to_process:
        return summary

    # ---- rebuild CSV rows (replace scenario block in place) ----
    rebuilt = []
    inserted = set()
    for row in orig_rows:
        sc = row["scenario_id"]
        if sc in to_process:
            if sc not in inserted:
                rebuilt.extend(to_process[sc])
                inserted.add(sc)
            continue
        rebuilt.append(row)

    # ---- rebuild MD sections ----
    md_text = md_path.read_text(encoding="utf-8")

    def repl(m):
        block = m.group(0)
        fm = FIRST_CID.search(block)
        if not fm:
            return block
        scenario = fm.group(1).rsplit("_", 1)[0]
        if scenario in to_process:
            return render_md_section(to_process[scenario])
        return block

    new_md_text = CRITERIA_SECTION.sub(repl, md_text)

    # ---- summary of what changed ----
    for scenario, new_rows in to_process.items():
        ported = sorted(PORT_MAP.get(scenario, {}).keys())
        split = SPLIT_CHILDREN.get(scenario, [])
        merged = MERGED.get(scenario, {}).get("new")
        grades = {short_id(r["criterion_id"]): r["grade"] for r in new_rows}
        blanks = [k for k, v in grades.items() if v not in ("P", "F")]
        summary.append(
            f"    {scenario}: ported={ported} split_added={split} "
            f"merged={[merged] if merged else []} | grades={grades}"
            + (f"  !! BLANK/BAD: {blanks}" if blanks else "")
        )

    if not dry_run:
        # refresh backups first
        shutil.copyfile(csv_path, csv_path.with_suffix(".csv.bak"))
        shutil.copyfile(md_path, md_path.with_suffix(".md.bak"))
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=fieldnames, lineterminator="\n",
                quoting=csv.QUOTE_MINIMAL,
            )
            writer.writeheader()
            writer.writerows(rebuilt)
        md_path.write_text(new_md_text, encoding="utf-8")

    return summary


def main(argv):
    dry_run = "--dry-run" in argv
    curated_by_scenario = load_curated()
    for sid in AFFECTED_SCENARIOS:
        if not curated_by_scenario[sid]:
            print(f"ERROR: no curated criteria found for {sid}", file=sys.stderr)
            return 1

    stems = sorted({s for lst in PACKETS_BY_SCENARIO.values() for s in lst})
    print(f"{'DRY RUN: ' if dry_run else ''}apply curation regrade")
    for stem in stems:
        print(f"  {stem}:")
        for line in process_packet(stem, curated_by_scenario, dry_run):
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
