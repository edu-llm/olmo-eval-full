"""Emit a review sheet for the reframed criteria that need a human decision.

Compares ``rubrics_clean.jsonl`` (before) to ``rubrics_reframed.jsonl`` (after)
and writes only the questionable reframes -- the mixed-voice ones (a verdict or
imperative welded onto an "Identifies ..." clause) and the identify-only ones
(goal/need/context observations turned into mandated tutor actions). The clean
confusion->"points out" reframes are omitted (already signed off).

Output ``reframe_review.tsv`` has one row per criterion with a ``decision``
column left blank for the reviewer to fill in (keep / fix / revert).

Run:
    python scripts/build_reframe_review.py
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "TutorEval"

# Leading OR embedded imperative cues (broader than the first scan -- catches
# "Do not ...", "Summarise it", "use only ...", etc.).
IMPER = re.compile(
    r"\b(Give|Explain|Show|Remind|Note|State|Point out|Consider|Provide|Describe|"
    r"Compute|Derive|Mention|Discuss|Use|Find|Summari[sz]e|Do not|Don't|Avoid|"
    r"Ask|Guide|Encourage|Clarify|Ensure|Make sure)\b"
)
VERDICT = re.compile(r"^(This is (true|false|correct|incorrect|wrong)|Yes|No|The answer is)\b")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def sents(text: str) -> list[str]:
    return re.split(r"(?<=[.!?])\s+", text.strip())


def bucket(before: str, after: str) -> str | None:
    """Classify a reframe into a review bucket, or None to skip it."""
    ss = sents(after)
    points_out = "Identifies and points out" in after
    identifies = "Identifies that the student" in after

    # Mixed voice: an imperative/verdict sentence sits alongside a reframed clause.
    other = [s for s in ss if not s.startswith("Identifies")]
    mixed = ("Identifies" in after) and any(IMPER.search(s) or VERDICT.match(s) for s in other)
    if mixed:
        return "C_mixed_voice"

    # Identify-only reframes are the manufactured-criterion risk; split by flavor.
    if identifies:
        low = before.lower()
        if re.search(r"\b(asking|requests?|requesting|refers?|referring|wants? to|"
                     r"aware|indicates|relating)\b", low):
            return "B1_scene_setting"
        if re.search(r"\bcorrect(ly)?\b", low):
            return "B2_student_correct"
        if re.search(r"misunderst|mistook|misread|misspell|confu|bug|typo|did not|"
                     r"not clear|added .* instead|expanded in", low):
            return "B3_error_observation"
        return "B4_other_identify"

    # Pure confusion->points-out (single voice): already approved, skip.
    if points_out:
        return None
    return "Z_uncategorized"


def main() -> None:
    before = {r["criterion_id"]: r["criterion"] for r in read_jsonl(OUT_DIR / "rubrics_clean.jsonl")}
    after = {r["criterion_id"]: r["criterion"] for r in read_jsonl(OUT_DIR / "rubrics_reframed.jsonl")}

    rows = []
    for cid, aft in after.items():
        bef = before.get(cid, "")
        if bef == aft:
            continue
        b = bucket(bef, aft)
        if b is None:
            continue
        rows.append((b, cid, bef, aft))

    rows.sort(key=lambda r: (r[0], r[1]))

    out = OUT_DIR / "reframe_review.tsv"
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["bucket", "criterion_id", "decision", "before", "after"])
        for b, cid, bef, aft in rows:
            w.writerow([b, cid, "", bef, aft])

    from collections import Counter
    counts = Counter(b for b, *_ in rows)
    print(f"wrote {out.relative_to(ROOT)}  ({len(rows)} rows for review)")
    for b, n in sorted(counts.items()):
        print(f"  {b}: {n}")


if __name__ == "__main__":
    main()
