"""Rubric-quality audit: quantify three human-flagged failure modes in the
TutorBench-derived criterion bank.

READ-ONLY. Never mutates data/rubrics_qmatrix_final.jsonl. Emits a report under
staging/ (audit_rubric_quality.csv + .json) and prints a summary.

The three flagged issues (from human grader packet review):

1. NON-ATOMIC ("bundled")  -- one criterion smuggles in multiple independently
   gradeable requirements, so a single P/F cannot honestly represent it. Detected
   via: multiple requirement verbs (must/should), enumerated sub-steps
   ("first...then", numbered lists), conjunction of distinct actions
   (" and ", ", then ", "as well as", "followed by", "concluding with"), or
   multiple sentence-level clauses each carrying a demand.

2. IRRELEVANT / MIS-SCOPED  -- the criterion grades something that is not clearly
   responsive to the *follow-up turn actually being evaluated*. Two proxies:
   (a) "restatement" criteria that demand the model re-emit a formula / the final
       numeric answer already delivered in an earlier turn ("must include the
       formula", "must provide the answer for ..."), and
   (b) pure surface/format/persona criteria with no skill loading
       (q_mapping all-zero AND text matches formatting/second-person/tone).

3. RIGID / VERBATIM  -- criterion is phrased so a correct-but-differently-worded
   response fails: heavy verbatim anchoring ("i.e." + long quoted target,
   "must say ...", "must state that ... is incorrect", exact-string demands),
   or a very long single criterion (proxy for over-specification).

Every flag is a HEURISTIC. The CSV dumps per-criterion flags for hand audit.

Usage:
    python scripts/audit_rubric_quality.py
    python scripts/audit_rubric_quality.py --rubrics data/rubrics_qmatrix_final.jsonl --out-dir staging
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def load_rubrics(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# --- detectors -------------------------------------------------------------

REQ_VERB = re.compile(r"\b(must|should|needs? to|has to)\b", re.I)
CONJ_ACTION = re.compile(
    r"(,\s*then\b|\bfollowed by\b|\bas well as\b|\bconcluding with\b|\bconclude with\b|"
    r"\bbeginning with\b|\bthen (?:computing|compute|find|solve|determine)\b|\band then\b)",
    re.I,
)
ENUM = re.compile(r"(\b1\.\s|\bstep\s*\d|\bfirst\b.*\bthen\b|\(i\)|\(ii\)|\(a\)\s.*\(b\)\s)", re.I)


def count_sentences(text: str) -> int:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(])", text.strip())
    return len([p for p in parts if p.strip()])


def is_non_atomic(text: str) -> tuple[bool, list[str]]:
    reasons = []
    if len(REQ_VERB.findall(text)) >= 2:
        reasons.append("multi_requirement_verb")
    if CONJ_ACTION.search(text):
        reasons.append("conjoined_actions")
    if ENUM.search(text):
        reasons.append("enumerated_substeps")
    # "and" joining two verb-like demands (rough)
    if re.search(r"\b(identify|correct|explain|provide|calculate|compute|show|state)\b[^.]*\band\b[^.]*\b(identify|correct|explain|provide|calculate|compute|show|state)\b", text, re.I):
        reasons.append("two_action_verbs_and")
    # multiple graded errors bundled
    if re.search(r"\b(\d+)\s+errors?\b", text, re.I) or text.lower().count("incorrect") >= 2:
        reasons.append("multiple_errors_bundled")
    return (len(reasons) > 0, reasons)


RESTATE = re.compile(
    r"(must|should)\b[^.]*\b(include|provide|state|give)\b[^.]*\b(the formula|the answer|the correct answer|the value|the equation)\b",
    re.I,
)


def is_irrelevant(text: str, q: dict) -> tuple[bool, list[str]]:
    reasons = []
    if RESTATE.search(text):
        reasons.append("restatement_of_formula_or_answer")
    all_zero = (q.get("content", 0) == 0 and q.get("diagnosis", 0) == 0 and q.get("scaffolding", 0) == 0)
    if all_zero and re.search(r"\b(second person|first person|perspective of|use of \"?you|markdown|latex|headings?|bold|bullet|format)\b", text, re.I):
        reasons.append("surface_format_or_persona_only")
    return (len(reasons) > 0, reasons)


RIGID_QUOTE = re.compile(r"\bi\.e\.|\be\.g\.\b", re.I)
RIGID_SAY = re.compile(r"(must|should)\b[^.]*\b(say|state|include the (?:phrase|sentence|wording))\b", re.I)
RIGID_INCORRECT = re.compile(r"\bidentify that\b[^.]*\bis incorrect\b", re.I)


def is_rigid(text: str) -> tuple[bool, list[str]]:
    reasons = []
    if RIGID_SAY.search(text):
        reasons.append("demands_specific_wording")
    if RIGID_INCORRECT.search(text):
        reasons.append("verbatim_incorrectness_label")
    # long i.e./e.g. anchored target
    m = re.search(r"\bi\.e\.\s*(.+)$", text, re.I)
    if m and len(m.group(1)) > 60:
        reasons.append("long_ie_anchor")
    if len(text) > 320:
        reasons.append("very_long_criterion")
    return (len(reasons) > 0, reasons)


# --- additional approved patterns (curation_v1) ----------------------------

# Tightened: require an explicit ERROR-detection clause AND an explicit
# CORRECTION-provision clause co-occurring (not just the word "correctly").
_DETECT = re.compile(
    r"(is (?:incorrect|wrong)|incorrectly|made (?:an?|the) (?:error|mistake)|"
    r"student'?s (?:error|mistake)|the (?:error|mistake) (?:is|was|lies))",
    re.I,
)
_FIX = re.compile(
    r"(the correct (?:answer|value|formula|expression|approach|version)|"
    r"should (?:instead )?be|corrected version|the correction is|rewrite it as)",
    re.I,
)


def is_identify_correct(text: str) -> tuple[bool, list[str]]:
    """One criterion bundling an explicit 'detect the error' AND 'give the fix'.
    Excludes the already-atomic 'correct the error' half of a detect/correct pair."""
    if text.strip().lower().startswith(("the response must correct", "the response should correct")):
        return (False, [])
    if _DETECT.search(text) and _FIX.search(text):
        return (True, ["identify_and_correct_bundled"])
    return (False, [])


REF_LEAK = re.compile(r"\b(golden response|reference solution|model answer|provided solution|ideal response)\b", re.I)


def is_ref_leak(text: str) -> tuple[bool, list[str]]:
    if REF_LEAK.search(text):
        return (True, ["reference_solution_leak"])
    return (False, [])


ABSOLUTE_FAIL = re.compile(
    r"(leads? to failure|causes? failure|results? in failure|constitutes? (?:a )?failure|"
    r"automatic(?:ally)? fail|"
    r"any (?:missing|omitted|omission|undefined|deviation|switch)[^.]{0,60}fail)",
    re.I,
)


def is_absolute_fail(text: str) -> tuple[bool, list[str]]:
    if ABSOLUTE_FAIL.search(text):
        return (True, ["absolute_failure_language"])
    return (False, [])


_WANTS_ANSWER = re.compile(
    r"(must|should)\b[^.]*\b(provide|state|give|include|report|arrive at|calculate|compute)\b[^.]*\b(answer|value|result)\b",
    re.I,
)
_TOLERANCE = re.compile(
    r"(approx|\u2248|\u007e|about|roughly|round|significant figure|sig ?fig|tolerance|equivalent|\+/-|\u00b1|or so)",
    re.I,
)


def is_numeric_no_tolerance(text: str, skill: str | None) -> tuple[bool, list[str]]:
    # diagnosis criteria that merely reference a student's wrong number are not
    # numeric-answer demands; withhold ("must not compute") is the opposite.
    if skill == "diagnosis":
        return (False, [])
    if re.search(r"\b(must not|should not|shouldn't|do not|don't)\b", text, re.I):
        return (False, [])
    if _WANTS_ANSWER.search(text) and re.search(r"\d", text) and not _TOLERANCE.search(text):
        return (True, ["numeric_no_tolerance"])
    return (False, [])


PRESCRIBED = re.compile(
    r"(similar to:\s*[\"\u201c']|use the phrase|the exact phrase|word[- ]for[- ]word|\bverbatim\b|"
    r"exactly as follows|say something like|phrased? (?:as|like)\s*[\"\u201c'])",
    re.I,
)


def is_prescribed_wording(text: str) -> tuple[bool, list[str]]:
    if PRESCRIBED.search(text):
        return (True, ["prescribed_wording"])
    return (False, [])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rubrics", default="data/rubrics_qmatrix_final.jsonl")
    ap.add_argument("--out-dir", default="staging")
    args = ap.parse_args()

    rubrics = load_rubrics(Path(args.rubrics))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_rows = []
    tallies = Counter()
    reason_tallies = Counter()
    flagged_any = 0

    for r in rubrics:
        text = r.get("criterion", "") or ""
        q = r.get("q_mapping", {}) or {}
        skill = r.get("primary_skill")
        results = {
            "non_atomic": is_non_atomic(text),
            "irrelevant_misscoped": is_irrelevant(text, q),
            "rigid_verbatim": is_rigid(text),
            "identify_correct": is_identify_correct(text),
            "reference_leak": is_ref_leak(text),
            "absolute_failure": is_absolute_fail(text),
            "numeric_no_tolerance": is_numeric_no_tolerance(text, skill),
            "prescribed_wording": is_prescribed_wording(text),
        }
        row = {
            "criterion_id": r.get("criterion_id"),
            "scenario_id": r.get("scenario_id"),
            "subject": r.get("subject"),
            "primary_skill": r.get("primary_skill"),
            "criticality": r.get("criticality"),
        }
        all_reasons: list[str] = []
        any_flag = False
        for key, (flag, rs) in results.items():
            row[key] = int(flag)
            if flag:
                tallies[key] += 1
                any_flag = True
                all_reasons.extend(rs)
        if any_flag:
            flagged_any += 1
        for x in all_reasons:
            reason_tallies[x] += 1
        row["reasons"] = "|".join(all_reasons)
        row["char_len"] = len(text)
        row["criterion"] = text.replace("\n", " ").strip()
        per_rows.append(row)

    n = len(rubrics)
    # write CSV
    import csv
    csv_path = out_dir / "audit_rubric_quality.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_rows[0].keys()))
        w.writeheader()
        w.writerows(per_rows)

    summary = {
        "n_criteria": n,
        "flagged_any": flagged_any,
        "flagged_any_pct": round(100 * flagged_any / n, 1),
        "by_issue": {k: {"count": v, "pct": round(100 * v / n, 1)} for k, v in tallies.items()},
        "by_reason": dict(reason_tallies.most_common()),
    }
    json_path = out_dir / "audit_rubric_quality.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Criteria audited: {n}")
    print(f"Flagged on >=1 issue: {flagged_any} ({summary['flagged_any_pct']}%)\n")
    for k, v in tallies.most_common():
        print(f"  {k:24s} {v:5d}  ({round(100*v/n,1)}%)")
    print("\nReason breakdown:")
    for k, v in reason_tallies.most_common():
        print(f"  {k:34s} {v:5d}")
    print(f"\nPer-criterion CSV -> {csv_path}")
    print(f"Summary JSON      -> {json_path}")


if __name__ == "__main__":
    main()
