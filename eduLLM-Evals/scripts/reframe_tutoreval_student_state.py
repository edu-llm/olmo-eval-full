"""Reframe TutorEval criteria phrased as student-state *observations* into
tutor-response *actions* (experimental).

~8% of the bullets read "The student is confused about X" -- a description of the
student, not something the tutor's response does, so "does the response satisfy
[a fact about the student]?" is ill-posed for the judge. This rewrites them into
the criterion voice the dataset already uses ("The response ... identifies ...").

Two templates, chosen per sentence:
- confusion / error states ("is confused", "has confused", "is wrong")
    -> "Identifies and points out to the student that they <...>"   (diagnosis +
       communicate; pronoun/verb converted to third-person "they")
- other states ("wants to understand X", "needs help with Y")
    -> "Identifies that the student <...>"                          (identify only)

Applied sentence-by-sentence so a merged-verdict criterion ("This is false. The
student is confused ...") gets its student-state clause reframed too. Deterministic,
zero LLM, reversible. Reads ``rubrics_clean.jsonl`` (verdict-merged) and writes
``*_reframed.{jsonl,json}``; skill/IRT fields stay null.

Run:
    python scripts/reframe_tutoreval_student_state.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "TutorEval"

# Third-person-singular -> "they" verb agreement for the point-out template.
AGREE = {
    "is": "are", "was": "were", "has": "have", "does": "do",
    "thinks": "think", "believes": "believe", "assumes": "assume",
    "confuses": "confuse", "conflates": "conflate", "misunderstands": "misunderstand",
    "makes": "make", "gets": "get", "mixes": "mix", "treats": "treat",
}
CONFUSION_RE = re.compile(
    r"confus|wrong|mistak|incorrect|misunderstand|conflat|\berror|mixe?[ds]|\bnot correct",
    re.I,
)
LEAD_RE = re.compile(r"^(the students?)\s+(.*)$", re.I)
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def they_form(first_word: str) -> str:
    """Convert a leading third-person-singular verb to its 'they' form."""
    low = first_word.lower()
    if low in AGREE:
        return AGREE[low]
    # Regular present-tense 3sg ("prefers" -> "prefer"); leave past/modal alone.
    if low.endswith("s") and not low.endswith("ss") and len(low) > 2:
        return first_word[:-1]
    return first_word


def reframe_sentence(sent: str) -> str:
    m = LEAD_RE.match(sent.strip())
    if not m:
        return sent
    rest = m.group(2)
    if CONFUSION_RE.search(rest):
        first, _, tail = rest.partition(" ")
        conv = they_form(first)
        clause = f"they {conv}" + (f" {tail}" if tail else "")
        return f"Identifies and points out to the student that {clause}"
    return f"Identifies that the student {rest}"


def reframe_criterion(text: str) -> tuple[str, bool]:
    sents = SENT_SPLIT.split(text.strip())
    new = [reframe_sentence(s) for s in sents]
    changed = new != sents
    return " ".join(new), changed


def main() -> None:
    scenarios = read_jsonl(OUT_DIR / "scenarios_clean.jsonl")
    rubrics = read_jsonl(OUT_DIR / "rubrics_clean.jsonl")

    out, examples, n_changed = [], [], 0
    for r in rubrics:
        new_text, changed = reframe_criterion(r["criterion"])
        if changed:
            n_changed += 1
            if len(examples) < 20:
                examples.append((r["criterion"], new_text))
        out.append({**r, "criterion": new_text})

    def write(name: str, records: list[dict]) -> None:
        (OUT_DIR / f"{name}.jsonl").write_text(
            "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in records),
            encoding="utf-8")
        (OUT_DIR / f"{name}.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote data/TutorEval/{name}.jsonl + .json  ({len(records)} rows)")

    write("rubrics_reframed", out)
    write("scenarios_reframed", scenarios)  # unchanged; paired for a clean set

    print(f"\nreframed {n_changed} criteria (of {len(rubrics)})")
    print("\n--- before -> after ---")
    for before, after in examples:
        print(f"\n  - {before[:110]!r}")
        print(f"  + {after[:130]!r}")


if __name__ == "__main__":
    main()
