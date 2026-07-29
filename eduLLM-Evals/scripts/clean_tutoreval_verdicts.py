"""Surgically clean the ~2% of TutorEval bullets that are broken as standalone
binary criteria, keeping every other bullet atomic (experimental).

The only class that genuinely breaks atomic yes/no grading is the **bare answer
verdict** -- ``yes`` / ``This is false.`` / ``the answer is Yes`` -- which carries
no gradeable content on its own ("does the response satisfy 'yes'?" is
meaningless). Every other short bullet (``explain the steps``, ``Sign is
positive``, ``Answer is $2B$``, dataset names) is a fine standalone criterion and
is left untouched.

Fix: when a scenario has a bare-verdict bullet AND a substantive sibling, remove
the verdict row and prepend the verdict onto the first substantive sibling so it
becomes one well-formed criterion ("This is false. The student is confused...").
A lone verdict with no sibling is left as-is (nothing to merge) and reported.

Originals are never modified; output goes to ``*_clean.{jsonl,json}``. Skill/IRT
fields stay null (skeleton), same as the per-bullet ingest.

Run:
    python scripts/clean_tutoreval_verdicts.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "TutorEval"

SCENARIO_KEYS = [
    "scenario_id", "source_id", "use_case", "subject", "subset", "grade_band",
    "modality", "prompt", "conversation_context", "reference_solution",
    "book_condition", "misleading_question", "answer_in_chapter",
    "criterion_ids", "source", "split", "version",
]

# A bare verdict = a pure yes/no/true/false judgment with no other content.
# Matched after lowercasing and stripping trailing punctuation.
_VERDICT_RE = re.compile(
    r"^(the )?answer is (yes|no)$"
    r"|^(this is )?(true|false|correct|incorrect|wrong|right|not wrong)$"
    r"|^(yes|no)(, it would)?$",
    re.I,
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def is_bare_verdict(text: str) -> bool:
    t = re.sub(r"[.,!]+$", "", (text or "").strip())
    return bool(_VERDICT_RE.match(t))


def normalize_verdict(text: str) -> str:
    """Render a verdict as a clean leading sentence: capitalized, single period."""
    t = re.sub(r"[.,!]+$", "", text.strip())
    t = t[:1].upper() + t[1:]
    return t + "."


def clean(scenarios: list[dict], rubrics: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    by_scenario: dict[str, list[dict]] = {}
    for r in rubrics:
        by_scenario.setdefault(r["scenario_id"], []).append(r)

    new_rubrics: list[dict] = []
    new_scenarios: list[dict] = []
    merged_log: list[dict] = []
    sid_to_ids: dict[str, list[str]] = {}

    for s in scenarios:
        sid = s["scenario_id"]
        rows = list(by_scenario.get(sid, []))
        verdicts = [r for r in rows if is_bare_verdict(r["criterion"])]
        siblings = [r for r in rows if not is_bare_verdict(r["criterion"])]

        # Merge only when there is a substantive sibling to attach the verdict to.
        if verdicts and siblings:
            prefix = " ".join(normalize_verdict(v["criterion"]) for v in verdicts)
            target = siblings[0]
            merged_log.append({"scenario_id": sid,
                               "verdicts": [v["criterion"] for v in verdicts],
                               "merged_into": target["criterion"]})
            target = {**target, "criterion": f"{prefix} {target['criterion']}"}
            kept = [target] + siblings[1:]
        else:
            kept = rows  # nothing to merge (lone verdict, or no verdict)

        # Renumber criterion ids contiguously within the scenario.
        ids = [f"{sid}_c{i:02d}" for i in range(1, len(kept) + 1)]
        sid_to_ids[sid] = ids
        for cid, r in zip(ids, kept):
            new_rubrics.append({**r, "criterion_id": cid})

        new_scenarios.append({**{k: s[k] for k in SCENARIO_KEYS},
                              "criterion_ids": ids})
        new_scenarios[-1] = {k: new_scenarios[-1][k] for k in SCENARIO_KEYS}

    return new_scenarios, new_rubrics, merged_log


def write(name: str, records: list[dict]) -> None:
    jsonl_path = OUT_DIR / f"{name}.jsonl"
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    json_path = OUT_DIR / f"{name}.json"
    with json_path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(f"wrote {jsonl_path.relative_to(ROOT)} + .json  ({len(records)} rows)")


def main() -> None:
    scenarios = read_jsonl(OUT_DIR / "scenarios.jsonl")
    rubrics = read_jsonl(OUT_DIR / "rubrics.jsonl")
    new_scenarios, new_rubrics, merged = clean(scenarios, rubrics)

    write("scenarios_clean", new_scenarios)
    write("rubrics_clean", new_rubrics)

    lone = [r["criterion"] for r in new_rubrics
            if is_bare_verdict(r["criterion"])]
    print(f"\n{len(rubrics)} -> {len(new_rubrics)} criteria "
          f"({len(merged)} verdicts merged into a sibling)")
    print(f"lone verdicts left as-is (no sibling to merge): {len(lone)} {lone}")
    print("\n--- example merges ---")
    for m in merged[:6]:
        print(f"  [{m['scenario_id']}] {m['verdicts']} + {m['merged_into'][:70]!r}")
    for sid in ("te_0261", "te_0165"):
        r = next(r for r in new_rubrics if r["scenario_id"] == sid)
        print(f"\n  {sid} c01 now: {r['criterion'][:130]!r}")


if __name__ == "__main__":
    main()
