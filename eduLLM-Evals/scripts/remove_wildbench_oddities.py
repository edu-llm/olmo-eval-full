"""Remove malformed/unanswerable WildBench scenarios, then re-index contiguously.

Audit trail for the oddity pass over data/WildBench/. Whole scenarios are
removed (and all their rubric criteria), then the survivors are re-numbered
WB_0000..WB_N with no gaps. scenario_id therefore no longer equals the source
parquet row index after this runs -- source_id remains the join key back to
allenai/WildBench.

Removed (22 scenarios):
  image-input references (model/judge can't see the image), 10:
    WB_0085 WB_0277 WB_0364 WB_0404 WB_0526 WB_0543 WB_0552 WB_0629 WB_0782 WB_0940
  truncated-context orphans (single-turn prompt opens mid-conversation), 3:
    WB_0544 WB_0772 WB_1023
  no clear question (single-turn pure content dump, no instruction), 2:
    WB_0627 WB_0750
  non-answer reference_solution (refusal / bare ack), 2:
    WB_0559 WB_0973
  non-English prompt, 4:
    WB_0229 WB_0298 WB_0426 WB_0499
  encoding artifact (U+FFFD), 1:
    WB_0564

Not touched: duplicate-criteria scenarios (inherited from WildBench's GPT-4T +
Claude-3-Opus checklist merge), constant metadata fields, non-question criteria.

NOTE: irt_logs/manifest.json describes the pre-removal IRT generation run and is
left as-is; its n_records / distribution stats are now stale.

Run:
    python scripts/remove_wildbench_oddities.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "WildBench"
SCEN = DATA / "scenarios.jsonl"
RUBR = DATA / "rubrics.jsonl"

LINE_SEPS = {0x2028: "\\u2028", 0x2029: "\\u2029"}

REMOVE = {
    "WB_0085", "WB_0277", "WB_0364", "WB_0404", "WB_0526", "WB_0543", "WB_0552",
    "WB_0629", "WB_0782", "WB_0940",           # image-input
    "WB_0544", "WB_0772", "WB_1023",           # orphans
    "WB_0627", "WB_0750",                        # no clear question
    "WB_0559", "WB_0973",                        # non-answer
    "WB_0229", "WB_0298", "WB_0426", "WB_0499",  # non-English
    "WB_0564",                                    # encoding artifact
}


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def write_jsonl(p: Path, rows: list[dict]) -> None:
    with p.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False).translate(LINE_SEPS) + "\n")


def write_json(p: Path, rows: list[dict]) -> None:
    with p.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def suffix(cid: str) -> str:
    return cid.rsplit("_", 1)[1]  # 'WB_0085_c01' -> 'c01'


def main() -> None:
    scen = read_jsonl(SCEN)
    rubr = read_jsonl(RUBR)
    n0_s, n0_r = len(scen), len(rubr)

    present = {s["scenario_id"] for s in scen}
    missing = REMOVE - present
    if missing:
        raise SystemExit(f"IDs not present, aborting: {sorted(missing)}")

    kept = [s for s in scen if s["scenario_id"] not in REMOVE]
    # survivors keep file order (already ascending); assign contiguous new ids
    remap = {s["scenario_id"]: f"WB_{i:04d}" for i, s in enumerate(kept)}

    for s in kept:
        old = s["scenario_id"]
        new = remap[old]
        s["scenario_id"] = new
        s["criterion_ids"] = [f"{new}_{suffix(c)}" for c in s["criterion_ids"]]

    new_rubr = []
    for r in rubr:
        old = r["scenario_id"]
        if old in REMOVE:
            continue
        new = remap[old]
        r["scenario_id"] = new
        r["criterion_id"] = f"{new}_{suffix(r['criterion_id'])}"
        new_rubr.append(r)

    # ---- validate before writing ----
    errs = []
    sids = [s["scenario_id"] for s in kept]
    if sids != [f"WB_{i:04d}" for i in range(len(kept))]:
        errs.append("scenario_ids not contiguous WB_0000..WB_N")
    declared = {c for s in kept for c in s["criterion_ids"]}
    actual = {r["criterion_id"] for r in new_rubr}
    if declared != actual:
        errs.append(f"criterion mismatch: {len(declared - actual)} missing, {len(actual - declared)} extra")
    if {r["scenario_id"] for r in new_rubr} - set(sids):
        errs.append("rubric references unknown scenario")
    if len(actual) != len(new_rubr):
        errs.append("duplicate criterion_id after remap")
    for s in kept:
        exp = [f"{s['scenario_id']}_c{i:02d}" for i in range(1, len(s["criterion_ids"]) + 1)]
        if s["criterion_ids"] != exp:
            errs.append(f"{s['scenario_id']}: criterion suffixes not contiguous c01..")
            break
    if errs:
        raise SystemExit("VALIDATION FAILED:\n  - " + "\n  - ".join(errs))

    write_jsonl(SCEN, kept)
    write_jsonl(RUBR, new_rubr)
    write_json(DATA / "scenarios.json", kept)
    write_json(DATA / "rubrics.json", new_rubr)

    print(f"scenarios: {n0_s} -> {len(kept)}  (removed {n0_s - len(kept)})")
    print(f"criteria : {n0_r} -> {len(new_rubr)}  (removed {n0_r - len(new_rubr)})")
    print("validation passed; wrote scenarios.jsonl/.json and rubrics.jsonl/.json")


if __name__ == "__main__":
    main()
