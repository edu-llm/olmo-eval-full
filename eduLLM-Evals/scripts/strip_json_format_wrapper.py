"""Strip the trailing "respond in JSON" format wrapper from EduBench unmerged prompts.

Removes ONLY the JSON-output format instruction from each scenario `prompt` -- the task
description and deliverable wording (the quoted key names like "Corrected Answer",
"Score", "Teaching Materials", ...) are left intact. For Q&A and IP the bare JSON key
STUB (`"Answer":,` / `"Reasoning Provided":,`) is also removed, since it is a JSON schema
echo that sits after the task already stated the deliverable; IP's redundant `..` is
collapsed. The resulting prompt is normalized to end with a single period.

Nothing else changes: key order and every other field (incl. ES `conversation_context`)
are preserved. Verified beforehand that "JSON" appears ONLY in this trailing clause, so no
content mention is at risk. Idempotent (a re-run matches nothing). --write to apply;
default is a dry run. Rewrites scenarios.jsonl and its pretty .json twin in the ingester's
format.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

U = Path(__file__).resolve().parents[1] / "data" / "EduBench" / "augmented_qmat" / "unmerged"
SCEN_JSONL = U / "scenarios.jsonl"
SCEN_JSON = SCEN_JSONL.with_suffix(".json")

# Per-use_case end-anchored JSON-wrapper pattern. Each removes the "in JSON format" clause
# (and, for Q&A/IP, the preceding key stub; for IP, the redundant '..').
WRAPPER = {
    "answering_questions": r'\s*"Answer"\s*:\s*,\s*in JSON format\s*\.?\s*$',
    "error_correction": r'\s*,\s*in JSON format\s*\.?\s*$',
    "idea_provision": r'\s*\.*\s*"Reasoning Provided"\s*:\s*,\s*return in JSON format\s*\.?\s*$',
    "learning_support": r'\s*,\s*returning the result in JSON format\s*\.?\s*$',
    "mental_health": r'\s*,\s*in JSON format\s*\.?\s*$',
    "question_generation": r'\s+in JSON format\s*\.?\s*$',
    "grading": r'\s*,\s*in JSON format\s*\.?\s*$',
    "material_generation": r'\s*,\s*in JSON format\s*\.?\s*$',
    "personalized_content_creation": r'\s*,\s*returned in JSON format\s*\.?\s*$',
}
WRAPPER = {uc: re.compile(p, re.IGNORECASE) for uc, p in WRAPPER.items()}
DANGLING = re.compile(r'"[A-Za-z][A-Za-z &]*"\s*:\s*,')


def strip_wrapper(use_case: str, prompt: str) -> str:
    """Remove the JSON wrapper for this use_case and normalize the trailing period."""
    pat = WRAPPER.get(use_case)
    if pat is None:
        return prompt
    out = pat.sub("", prompt).rstrip()
    if out and out[-1] != ".":
        out += "."
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true", help="apply changes (default: dry run)")
    args = ap.parse_args()

    records = [json.loads(l) for l in SCEN_JSONL.open(encoding="utf-8")]

    changed_by_uc: dict[str, int] = {}
    examples: dict[str, tuple[str, str, str]] = {}
    for r in records:
        uc = r.get("use_case", "")
        old = r.get("prompt", "")
        new = strip_wrapper(uc, old)
        if new != old:
            changed_by_uc[uc] = changed_by_uc.get(uc, 0) + 1
            examples.setdefault(uc, (r["scenario_id"], old, new))
        if args.write:
            r["prompt"] = new

    total_changed = sum(changed_by_uc.values())
    print(f"scenarios: {len(records)}   prompts changed: {total_changed}")
    print("\nchanged by use_case:")
    for uc in sorted(changed_by_uc):
        print(f"  {uc:<32} {changed_by_uc[uc]}")

    print("\nbefore -> after (one example per changed use_case):")
    for uc in sorted(examples):
        sid, old, new = examples[uc]
        print(f"\n  [{uc} | {sid}]")
        print(f"    BEFORE: ...{old[-110:]!r}")
        print(f"    AFTER : ...{new[-110:]!r}")

    # Post-conditions on the transformed prompts (whether or not we write).
    transformed = [strip_wrapper(r.get("use_case", ""), r.get("prompt", "")) for r in records] \
        if not args.write else [r["prompt"] for r in records]
    with_json = [r["scenario_id"] for r, t in zip(records, transformed) if "json" in t.lower()]
    dangling = [r["scenario_id"] for r, t in zip(records, transformed) if DANGLING.search(t)]
    empty = [r["scenario_id"] for r, t in zip(records, transformed) if not t.strip()]
    no_period = [r["scenario_id"] for r, t in zip(records, transformed) if t.rstrip() and not t.rstrip().endswith(".")]
    print("\npost-conditions:")
    print(f"  prompts still containing 'json': {len(with_json)}" + (f" -> {with_json[:10]}" if with_json else ""))
    print(f"  prompts with dangling '\"key\":,': {len(dangling)}" + (f" -> {dangling[:10]}" if dangling else ""))
    print(f"  empty prompts: {len(empty)}")
    print(f"  prompts not ending with '.': {len(no_period)}" + (f" -> {no_period[:10]}" if no_period else ""))

    if not args.write:
        print("\n--dry-run: nothing written (pass --write to apply)")
        return 0

    if with_json or dangling or empty or no_period:
        raise SystemExit("refusing to write: post-condition failed (see above)")

    with SCEN_JSONL.open("w", encoding="utf-8", newline="\n") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with SCEN_JSON.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(f"\nwrote {SCEN_JSONL.name} and {SCEN_JSON.name} ({total_changed} prompts stripped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
