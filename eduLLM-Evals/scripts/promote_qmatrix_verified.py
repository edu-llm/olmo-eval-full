"""Promote verifier-resolved Q-matrix labels into the active ``q_mapping``.

``verify_qmatrix.py`` writes each record with a ``verification`` block whose
``final_q_mapping`` holds the resolved label (majority by default). Downstream
IRT/engine code reads the top-level ``q_mapping``, which still holds the *generator's*
original label -- so the verified result is inert until it is promoted here.

This script copies ``verification.final_q_mapping`` into the top-level ``q_mapping`` and
makes ``primary_skill`` consistent with it. By default the output is stripped down to the
tracked TutorBench rubric schema: the ``verification`` block and the promote-only fields
(``q_mapping_generator``, ``primary_skill_generator``, ``q_mapping_source``) are dropped.
The full verifier votes remain in ``rubrics_qmatrix_verified.jsonl``; pass
``--keep-provenance`` to retain them inline.

Usage::

    python scripts/promote_qmatrix_verified.py \
        --input data/TutorEval/rubrics_qmatrix_verified.jsonl        # dry run (summary only)
    python scripts/promote_qmatrix_verified.py \
        --input data/TutorEval/rubrics_qmatrix_verified.jsonl --write # write *_final.jsonl (+ .json)

The skill axis is inferred from each record's ``q_mapping`` keys, so this works for the
TutorEval 2-skill axis or the TutorBench 4-skill axis without changes.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
        f.write("\n")


def resolve_primary(new_map: dict, skills: list[str], old_primary, verifier_primaries: dict):
    """Pick a primary_skill consistent with the promoted q_mapping.

    Prefer the generator's prior primary if it survived; else a verifier's primary if it
    survived; else the first surviving skill in canonical order; None if all-zero.
    """
    surviving = [s for s in skills if int(new_map.get(s, 0)) == 1]
    if not surviving:
        return None
    if old_primary in surviving:
        return old_primary
    for ps in (verifier_primaries or {}).values():
        if ps in surviving:
            return ps
    return surviving[0]


# Verify/promote-only provenance fields, dropped by default so the finalized bank matches
# the tracked TutorBench rubric schema. The full verifier votes are preserved separately in
# rubrics_qmatrix_verified.jsonl; pass --keep-provenance to retain these inline.
PROVENANCE_FIELDS = (
    "verification",
    "q_mapping_generator",
    "primary_skill_generator",
    "q_mapping_source",
)


def promote(records: list[dict], keep_provenance: bool = False) -> tuple[list[dict], dict]:
    out: list[dict] = []
    stats = {
        "total": len(records),
        "promoted": 0,
        "no_verification": 0,
        "no_generator_label": 0,
        "resolution": collections.Counter(),
        "cells_changed": collections.Counter(),
        "primary_changed": 0,
        "load_before": collections.Counter(),
        "load_after": collections.Counter(),
        "allzero_before": 0,
        "allzero_after": 0,
    }

    for rec in records:
        r = dict(rec)
        gen_map = r.get("q_mapping")
        block = r.get("verification") or {}
        final_map = block.get("final_q_mapping")

        if not keep_provenance:
            r.pop("verification", None)

        if gen_map is None:
            stats["no_generator_label"] += 1
            if keep_provenance:
                r["q_mapping_source"] = "generator_failed"
            out.append(r)
            continue

        skills = list(gen_map.keys())
        for s in skills:
            stats["load_before"][s] += int(gen_map[s])
        if not any(int(gen_map[s]) for s in skills):
            stats["allzero_before"] += 1

        if final_map is None:
            # No verifier-resolved label (skipped item): keep the generator label as-is.
            stats["no_verification"] += 1
            if keep_provenance:
                r["q_mapping_source"] = "generator_only"
            for s in skills:
                stats["load_after"][s] += int(gen_map[s])
            if not any(int(gen_map[s]) for s in skills):
                stats["allzero_after"] += 1
            out.append(r)
            continue

        resolution = block.get("resolution", "unknown")
        stats["resolution"][resolution] += 1
        stats["promoted"] += 1

        new_map = {s: int(final_map.get(s, 0)) for s in skills}
        for s in skills:
            if new_map[s] != int(gen_map[s]):
                stats["cells_changed"][s] += 1
            stats["load_after"][s] += new_map[s]
        if not any(new_map[s] for s in skills):
            stats["allzero_after"] += 1

        old_primary = r.get("primary_skill")
        new_primary = resolve_primary(
            new_map, skills, old_primary, block.get("primary_skills")
        )
        if new_primary != old_primary:
            stats["primary_changed"] += 1

        r["q_mapping"] = new_map
        r["primary_skill"] = new_primary
        if keep_provenance:
            r["q_mapping_generator"] = gen_map
            r["primary_skill_generator"] = old_primary
            r["q_mapping_source"] = f"verify:{resolution}"
        out.append(r)

    return out, stats


def print_summary(stats: dict) -> None:
    print(f"records:              {stats['total']}")
    print(f"promoted:             {stats['promoted']}")
    print(f"kept (no verifier):   {stats['no_verification']}")
    print(f"kept (gen failed):    {stats['no_generator_label']}")
    if stats["resolution"]:
        print("resolution breakdown: " +
              ", ".join(f"{k}={v}" for k, v in sorted(stats["resolution"].items())))
    print(f"primary_skill changed: {stats['primary_changed']}")
    print("\nskill load (1s) before -> after promotion:")
    skills = sorted(set(stats["load_before"]) | set(stats["load_after"]))
    for s in skills:
        b, a = stats["load_before"][s], stats["load_after"][s]
        chg = stats["cells_changed"][s]
        print(f"  {s:26s} {b:4d} -> {a:4d}   ({chg} cells changed)")
    print(f"  {'all-zero criteria':26s} {stats['allzero_before']:4d} -> {stats['allzero_after']:4d}")


def default_out(input_path: Path) -> Path:
    name = input_path.name.replace("_verified", "_final")
    if name == input_path.name:  # input didn't contain _verified
        name = input_path.stem + "_final" + input_path.suffix
    return input_path.with_name(name)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True,
                   help="Verified Q-matrix JSONL from verify_qmatrix.py.")
    p.add_argument("--out", type=Path, default=None,
                   help="Output JSONL (default: input with _verified -> _final). "
                        "A pretty-printed .json twin is written alongside.")
    p.add_argument("--write", action="store_true",
                   help="Write output. Without this flag, only the summary is printed.")
    p.add_argument("--keep-provenance", action="store_true",
                   help="Retain the verifier 'verification' block and q_mapping_generator/"
                        "primary_skill_generator/q_mapping_source fields. Off by default so the "
                        "output matches the tracked TutorBench rubric schema.")
    args = p.parse_args()

    if not args.input.exists():
        p.error(f"input not found: {args.input}")

    records = read_jsonl(args.input)
    promoted, stats = promote(records, keep_provenance=args.keep_provenance)
    print_summary(stats)

    if not args.write:
        print("\n(dry run -- pass --write to persist)")
        return

    out_jsonl = args.out or default_out(args.input)
    out_json = out_jsonl.with_suffix(".json")
    write_jsonl(out_jsonl, promoted)
    write_json(out_json, promoted)
    print(f"\nwrote {len(promoted)} records -> {out_jsonl}")
    print(f"wrote pretty twin       -> {out_json}")


if __name__ == "__main__":
    main()
