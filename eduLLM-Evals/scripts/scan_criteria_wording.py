"""Deterministic wording scanner for the TutorEval criterion bank.

READ-ONLY. Never mutates ``data/TutorEval/rubrics_final.jsonl``. It flags every
criterion whose *wording* is likely to trip the binary LLM judge, which sees only
the scenario prompt + conversation context + tutor response + one criterion string
and grades pass/fail. A good criterion is therefore a self-contained, positively
framed, checkable requirement ON THE TUTOR'S RESPONSE.

Buckets (each a heuristic; a criterion may trip several):

- ``not_a_requirement``   -- a bare declarative fact or observation, not framed as a
  requirement on the response (does not lead with a 3rd-person action verb such as
  "States/Explains/Identifies/Confirms/Does not ...", nor with an imperative verb).
- ``compound_multipart``  -- bundles two or more independently gradeable requirements
  (multiple requirement sentences, conjoined actions, enumerations).
- ``hedging``             -- optional / non-committal language (may, might, could,
  not necessarily, ideally, often, ...), so pass/fail is not decidable.
- ``observational_verdict`` -- a verdict about correctness ("Yes, this is correct",
  "It is O(n)") rather than a requirement on the response.
- ``vague_qualifier``     -- leans on unobservable quality words (good, well, clearly,
  properly, appropriately, ...).
- ``under_specified``     -- <= 4 words; too terse to grade.
- ``over_specified``      -- >= 45 words; over-specified single criterion.
- ``negation_double``     -- double / stacked negation that inverts meaning.

Outputs (under ``data/TutorEval/``, both SEPARATE from the tracked ``*_final`` bank):
- ``wording_flags.tsv``   -- one row per flagged criterion:
  ``criterion_id, scenario_id, buckets (semicolon-joined), n_buckets, criterion``.
- ``wording_flags.summary.json`` -- per-bucket counts + totals.

Usage:
    python scripts/scan_criteria_wording.py
    python scripts/scan_criteria_wording.py --rubrics data/TutorEval/rubrics_final.jsonl \
        --out-tsv data/TutorEval/wording_flags.tsv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUBRICS = ROOT / "data" / "TutorEval" / "rubrics_final.jsonl"
DEFAULT_OUT_TSV = ROOT / "data" / "TutorEval" / "wording_flags.tsv"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

# 3rd-person-singular action verbs that frame a requirement ON the response, i.e.
# the style-guide-approved leads ("States that ...", "Identifies that ...").
REQUIREMENT_LEADS = {
    "states",
    "explains",
    "identifies",
    "confirms",
    "provides",
    "acknowledges",
    "notes",
    "recognizes",
    "describes",
    "shows",
    "corrects",
    "guides",
    "asks",
    "reminds",
    "suggests",
    "affirms",
    "warns",
    "focuses",
    "mentions",
    "clarifies",
    "encourages",
    "ensures",
    "avoids",
    "gives",
    "informs",
    "prompts",
    "helps",
    "emphasizes",
    "emphasises",
    "distinguishes",
    "derives",
    "computes",
    "uses",
    "highlights",
    "addresses",
    "offers",
    "presents",
    "demonstrates",
    "verifies",
    "checks",
    "reiterates",
    "restates",
    "summarizes",
    "summarises",
    "relates",
    "connects",
    "compares",
    "defines",
    "illustrates",
    "interprets",
    "justifies",
    "outlines",
    "reviews",
    "solves",
    "walks",
    "points",
    "reframes",
    "explores",
    "elaborates",
    "reveals",
    "answers",
    "responds",
    "recommends",
    "proposes",
    "requests",
    "leads",
    "steers",
    "redirects",
    "reinforces",
    "validates",
}

# Base-form (imperative) action verbs: these are still requirements, just in the
# wrong voice, so they are NOT counted as "not a requirement".
IMPERATIVE_LEADS = {
    "state",
    "explain",
    "identify",
    "confirm",
    "provide",
    "acknowledge",
    "note",
    "recognize",
    "recognise",
    "describe",
    "show",
    "correct",
    "guide",
    "ask",
    "remind",
    "suggest",
    "affirm",
    "warn",
    "focus",
    "mention",
    "clarify",
    "encourage",
    "ensure",
    "avoid",
    "give",
    "inform",
    "prompt",
    "help",
    "emphasize",
    "emphasise",
    "distinguish",
    "derive",
    "compute",
    "use",
    "highlight",
    "address",
    "offer",
    "present",
    "demonstrate",
    "verify",
    "check",
    "reiterate",
    "restate",
    "summarize",
    "summarise",
    "relate",
    "connect",
    "compare",
    "define",
    "illustrate",
    "interpret",
    "justify",
    "outline",
    "review",
    "solve",
    "walk",
    "point",
    "consider",
    "find",
    "discuss",
    "do",
    "keep",
    "make",
    "start",
    "begin",
    "let",
    "try",
    "answer",
    "respond",
    "elaborate",
}

# Requirement-framed multi-word leads (checked on the lowercased text prefix).
REQUIREMENT_PHRASE_LEADS = (
    "does not",
    "doesn't",
    "do not",
    "don't",
    "the response",
    "the answer",
    "response provides",
    "answer provides",
    "the tutor",
)


def first_word(text: str) -> str:
    m = re.match(r"[^A-Za-z]*([A-Za-z']+)", text)
    return m.group(1).lower() if m else ""


def words(text: str) -> list[str]:
    return [w for w in re.split(r"\s+", text.strip()) if w]


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------

_HEDGE = re.compile(
    r"\b(may|might|could|not necessarily|ideally|often|possibly|perhaps|"
    r"if possible|where appropriate|as appropriate|as needed|preferably|"
    r"optionally|tends? to|would ideally)\b",
    re.I,
)

_VAGUE = re.compile(
    r"\b(good|well|clearly|properly|appropriately|nicely|adequately|effectively|"
    r"thoroughly|neatly|elegantly|reasonably|succinctly|gracefully|cleanly)\b",
    re.I,
)

_VERDICT = re.compile(
    r"^\s*(yes\b|no\b|correct\b|incorrect\b|true\b|false\b|indeed\b|"
    r"this is (?:correct|incorrect|true|false|right|wrong)|"
    r"that is (?:correct|incorrect|true|false|right|wrong)|"
    r"the answer is\b|it is\b|the student is (?:correct|right|wrong|incorrect))",
    re.I,
)

_CONJ_ACTION = re.compile(
    r"(,\s*then\b|\bfollowed by\b|\bas well as\b|\bconcluding with\b|"
    r"\band then\b|\bthen (?:computes?|compute|finds?|solves?|determines?|"
    r"explains?|states?|identifies)\b)",
    re.I,
)
_ENUM = re.compile(r"(\b1\.\s|\bstep\s*\d|\bfirst\b.*\bthen\b|\(i\)|\(ii\)|\(a\).*\(b\))", re.I)

# Two distinct action verbs joined by "and" (rough compound signal).
_TWO_ACTIONS = re.compile(
    r"\b(identif\w*|correct\w*|explain\w*|provid\w*|calculat\w*|comput\w*|show\w*|"
    r"state\w*|acknowledg\w*|point\w*|describe\w*|derive\w*|prompt\w*|guide\w*|"
    r"clarif\w*|summar\w*|note\w*|recogniz\w*|address\w*)\b[^.]*\band\b[^.]*"
    r"\b(identif\w*|correct\w*|explain\w*|provid\w*|calculat\w*|comput\w*|show\w*|"
    r"state\w*|acknowledg\w*|point\w*|describe\w*|derive\w*|prompt\w*|guide\w*|"
    r"clarif\w*|summar\w*|note\w*|recogniz\w*|address\w*)\b",
    re.I,
)

_NEG = re.compile(r"\b(not|never|no|without|cannot|neither|nor)\b|n't", re.I)
_DOUBLE_NEG = re.compile(
    r"\bnot\s+un\w+|\bnot\b[^.]*\b(without|neither|nor|never|no one|nothing)\b|"
    r"\bcannot\b[^.]*\bnot\b|\bnever\b[^.]*\bnot\b",
    re.I,
)


def sentence_count(text: str) -> int:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(\"'])", text.strip())
    return len([p for p in parts if p.strip()])


def is_math_or_code(text: str) -> bool:
    """Semicolons inside math/code are not clause boundaries."""
    return bool(re.search(r"[${}\\=^_]|def |lambda |print\(|\bO\(", text))


def looks_like_requirement(text: str) -> bool:
    low = text.strip().lower()
    if any(low.startswith(p) for p in REQUIREMENT_PHRASE_LEADS):
        return True
    return first_word(text) in REQUIREMENT_LEADS


def looks_like_imperative(text: str) -> bool:
    return first_word(text) in IMPERATIVE_LEADS


def classify(text: str) -> list[str]:
    """Return the sorted list of buckets a criterion trips (possibly empty)."""
    buckets: list[str] = []
    wc = len(words(text))

    # 1. not-a-requirement: neither 3rd-person requirement nor imperative lead.
    if not looks_like_requirement(text) and not looks_like_imperative(text):
        buckets.append("not_a_requirement")

    # 2. compound / multi-part.
    compound = (
        sentence_count(text) >= 2
        or bool(_CONJ_ACTION.search(text))
        or bool(_ENUM.search(text))
        or bool(_TWO_ACTIONS.search(text))
        or (";" in text and not is_math_or_code(text))
    )
    if compound:
        buckets.append("compound_multipart")

    # 3. hedging.
    if _HEDGE.search(text):
        buckets.append("hedging")

    # 4. observational verdict.
    if _VERDICT.search(text):
        buckets.append("observational_verdict")

    # 5. vague qualifier.
    if _VAGUE.search(text):
        buckets.append("vague_qualifier")

    # 6/7. length.
    if wc <= 4:
        buckets.append("under_specified")
    if wc >= 45:
        buckets.append("over_specified")

    # 8. double negation.
    if len(_NEG.findall(text)) >= 2 and _DOUBLE_NEG.search(text):
        buckets.append("negation_double")

    return sorted(buckets)


BUCKET_ORDER = [
    "not_a_requirement",
    "compound_multipart",
    "hedging",
    "observational_verdict",
    "vague_qualifier",
    "under_specified",
    "over_specified",
    "negation_double",
]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--rubrics",
        type=Path,
        default=DEFAULT_RUBRICS,
        help="Input rubric JSONL (default: %(default)s).",
    )
    ap.add_argument(
        "--out-tsv",
        type=Path,
        default=DEFAULT_OUT_TSV,
        help="Flag-sheet TSV (default: %(default)s). A .summary.json twin is written alongside.",
    )
    args = ap.parse_args()

    rubrics = read_jsonl(args.rubrics)

    flagged: list[tuple[str, str, str, int, str]] = []
    bucket_counts: Counter = Counter()
    n_by_nbuckets: Counter = Counter()
    for r in rubrics:
        text = (r.get("criterion") or "").strip()
        buckets = classify(text)
        if not buckets:
            continue
        for b in buckets:
            bucket_counts[b] += 1
        n_by_nbuckets[len(buckets)] += 1
        flagged.append(
            (
                r.get("criterion_id", ""),
                r.get("scenario_id", ""),
                ";".join(buckets),
                len(buckets),
                text.replace("\t", " ").replace("\n", " ").strip(),
            )
        )

    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_tsv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["criterion_id", "scenario_id", "buckets", "n_buckets", "criterion"])
        for row in flagged:
            w.writerow(row)

    summary = {
        "rubrics_path": str(args.rubrics),
        "n_criteria": len(rubrics),
        "n_flagged": len(flagged),
        "flagged_pct": round(100 * len(flagged) / len(rubrics), 1) if rubrics else 0.0,
        "by_bucket": {b: bucket_counts.get(b, 0) for b in BUCKET_ORDER},
        "by_n_buckets": {str(k): n_by_nbuckets[k] for k in sorted(n_by_nbuckets)},
    }
    summary_path = args.out_tsv.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Criteria scanned: {len(rubrics)}")
    print(f"Flagged (>=1 bucket): {len(flagged)} ({summary['flagged_pct']}%)\n")
    print("Per-bucket counts:")
    for b in BUCKET_ORDER:
        print(f"  {b:22s} {bucket_counts.get(b, 0):5d}")
    print("\nBy number of buckets tripped:")
    for k in sorted(n_by_nbuckets):
        print(f"  {k} bucket(s): {n_by_nbuckets[k]}")
    print(f"\nFlag sheet  -> {args.out_tsv}")
    print(f"Summary JSON -> {summary_path}")


if __name__ == "__main__":
    main()
