"""Restructure EduBench ES (mental_health) scenarios into a multi-turn form.

EduBench packs the whole ES task into one `prompt` string:
    Anxiety Level: <level>
    Dialogue with Student: <multi-turn Agent/Student exchange>
    <instruction, e.g. "Based on the student's emotional state ... in JSON format.">

This script splits that into the schema's existing fields, without adding/removing
any field or changing key order:
  * `conversation_context` (list[dict[str,str]] role/content turns) <- the dialogue
  * `prompt` <- ONLY the trailing instruction (the "Anxiety Level" line and the
    dialogue block are removed from the prompt, since the dialogue now lives in
    conversation_context).

Everything is derived from the raw en_data/ES.jsonl record (source of truth), so a
re-run is idempotent and reproducible: each scenario's current prompt must equal
either the raw full prompt (fresh) or the derived instruction (already processed),
which both catches by-order-join misalignment and makes the strip safe to repeat.

Roles map to the tutor_cat vocabulary (respgen/prompts._ROLE_MAP): Student -> "student",
Agent/AI/Intelligent Agent/Assistant/IA -> "tutor". Run with --write to apply,
otherwise it's a dry run.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_ES = ROOT / "data" / "EduBench" / "en_data" / "ES.jsonl"
SCEN_JSONL = ROOT / "data" / "EduBench" / "augmented_qmat" / "unmerged" / "scenarios.jsonl"
SCEN_JSON = SCEN_JSONL.with_suffix(".json")
DIALOGUE_FIELD = "Dialogue with Student"

# Plain-text (no-bracket) dialogues delimit turns by a role label at line start.
ROLE_ANCHOR = re.compile(
    r"(?im)^[ \t]*(intelligent agent|assistant|student|agent|teacher|tutor|system|ai|ia)[ \t]*:[ \t]*"
)


def norm_role(label: str) -> str:
    return "student" if "student" in label.strip().lower() else "tutor"


def parse_dialogue(d: str) -> list[dict[str, str]]:
    """Parse an ES dialogue string into [{role, content}, ...]. Handles the three
    observed shapes: list-of-dicts, list-of-strings, and plain newline-delimited."""
    d = d.strip()
    turns: list[dict[str, str]] = []

    lit = None
    if d[:1] in "[{":
        try:
            lit = ast.literal_eval(d)
        except (ValueError, SyntaxError):
            lit = None

    if isinstance(lit, dict) and lit:
        for role_label, content in lit.items():
            turns.append({"role": norm_role(str(role_label)), "content": str(content).strip()})
        return turns

    if isinstance(lit, list) and lit:
        for item in lit:
            if isinstance(item, dict):
                for role_label, content in item.items():
                    turns.append({"role": norm_role(str(role_label)),
                                  "content": str(content).strip()})
            elif isinstance(item, str):
                role_label, _, content = item.partition(":")
                if not content and ":" not in item:
                    role_label, content = "agent", item
                turns.append({"role": norm_role(role_label), "content": content.strip()})
        return turns

    matches = list(ROLE_ANCHOR.finditer(d))
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(d)
        turns.append({"role": norm_role(m.group(1)), "content": d[start:end].strip()})
    return turns


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true", help="apply changes (default: dry run)")
    args = ap.parse_args()

    raw = []
    with RAW_ES.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            info = rec.get("information", {}) or {}
            raw.append((str(info.get(DIALOGUE_FIELD, "")), str(rec.get("prompt", ""))))

    records = [json.loads(l) for l in SCEN_JSONL.open(encoding="utf-8")]
    es_idx = [i for i, r in enumerate(records) if r.get("use_case") == "mental_health"]

    if len(es_idx) != len(raw):
        raise SystemExit(f"count mismatch: {len(es_idx)} ES scenarios vs "
                         f"{len(raw)} raw ES rows -- refusing to align by order")

    empties, no_dialogue_in_raw, empty_instr, misaligned = [], [], [], []
    turn_total, n_fresh, n_stripped = 0, 0, 0
    role_counter = {"student": 0, "tutor": 0}
    for k, i in enumerate(es_idx):
        rec = records[i]
        dialogue, raw_prompt = raw[k]
        d = dialogue.strip()
        turns = parse_dialogue(dialogue)
        if not turns:
            empties.append(rec["scenario_id"])
        # Instruction = everything after the dialogue block in the raw prompt.
        if d and d in raw_prompt:
            instruction = raw_prompt.split(d, 1)[1].strip()
        else:
            no_dialogue_in_raw.append(rec["scenario_id"])
            instruction = ""
        if not instruction:
            empty_instr.append(rec["scenario_id"])
        # Per-record alignment + idempotency: current prompt must be the raw full
        # prompt (fresh) or already the derived instruction (re-run).
        cur = rec.get("prompt", "")
        if cur == raw_prompt:
            n_fresh += 1
        elif cur == instruction:
            n_stripped += 1
        else:
            misaligned.append(rec["scenario_id"])
        turn_total += len(turns)
        for t in turns:
            role_counter[t["role"]] = role_counter.get(t["role"], 0) + 1
        if args.write:
            rec["conversation_context"] = turns
            rec["prompt"] = instruction

    print(f"ES scenarios: {len(es_idx)}   (fresh={n_fresh}, already-stripped={n_stripped})")
    print(f"  turns parsed: {turn_total} (avg {turn_total / max(1, len(es_idx)):.1f}/scenario)")
    print(f"  role turns: {role_counter}")
    print(f"  ZERO parsed turns: {len(empties)}" + (f" -> {empties[:10]}" if empties else ""))
    print(f"  dialogue not in raw prompt: {len(no_dialogue_in_raw)}"
          + (f" -> {no_dialogue_in_raw[:10]}" if no_dialogue_in_raw else ""))
    print(f"  empty instruction after strip: {len(empty_instr)}"
          + (f" -> {empty_instr[:10]}" if empty_instr else ""))
    print(f"  alignment MISMATCH (prompt != raw and != instruction): {len(misaligned)}"
          + (f" -> {misaligned[:10]}" if misaligned else ""))

    ex_d, ex_p = raw[0]
    print(f"\nexample {records[es_idx[0]]['scenario_id']}:")
    print(f"  new prompt: {ex_p.split(ex_d.strip(), 1)[1].strip()!r}")
    print(f"  context (first 3 turns):")
    for t in parse_dialogue(ex_d)[:3]:
        print(f"    [{t['role']}] {t['content'][:100]}")

    if not args.write:
        print("\n--dry-run: nothing written (pass --write to apply)")
        return 0

    if empties or no_dialogue_in_raw or empty_instr or misaligned:
        raise SystemExit("refusing to write: unparsed / misaligned / empty-instruction ES rows (see above)")

    with SCEN_JSONL.open("w", encoding="utf-8", newline="\n") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with SCEN_JSON.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(f"\nwrote {SCEN_JSONL.name} and {SCEN_JSON.name} "
          f"({len(es_idx)} ES scenarios populated)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
