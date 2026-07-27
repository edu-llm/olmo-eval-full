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
        na, na_r = is_non_atomic(text)
        ir, ir_r = is_irrelevant(text, q)
        rg, rg_r = is_rigid(text)
        if na:
            tallies["non_atomic"] += 1
        if ir:
            tallies["irrelevant_misscoped"] += 1
        if rg:
            tallies["rigid_verbatim"] += 1
        if na or ir or rg:
            flagged_any += 1
        for x in na_r + ir_r + rg_r:
            reason_tallies[x] += 1
        per_rows.append({
            "criterion_id": r.get("criterion_id"),
            "scenario_id": r.get("scenario_id"),
            "subject": r.get("subject"),
            "primary_skill": r.get("primary_skill"),
            "criticality": r.get("criticality"),
            "non_atomic": int(na),
            "irrelevant_misscoped": int(ir),
            "rigid_verbatim": int(rg),
            "reasons": "|".join(na_r + ir_r + rg_r),
            "char_len": len(text),
            "criterion": text.replace("\n", " ").strip(),
        })

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
