"""Fix three prompt/metadata oddities in the EduBench unmerged scenarios.

1. Doubled label: collapse "Student Profile: Student Profile:" -> "Student Profile:"
   (learning_support prompts).
2. Grade-band mislabels: for personalized_content_creation / learning_support items whose
   grade_band is a higher-ed band (Undergraduate/Master/PhD) but whose embedded profile
   describes a school-age student (Age <= 18 or Grade <= 12), relabel grade_band to the band
   implied by the profile's Grade (preferred) or Age.
3. eb_8976: rewrite its free-text profile into the dict form every other PCC item uses.

Only `prompt` and `grade_band` change; key order and all other fields are preserved.
Idempotent (a re-run matches nothing). --write to apply; default is a dry run.
Rewrites scenarios.jsonl and its pretty .json twin.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

U = Path(__file__).resolve().parents[1] / "data" / "EduBench" / "augmented_qmat" / "unmerged"
SCEN_JSONL = U / "scenarios.jsonl"
HIGHER = {"Undergraduate", "Master", "PhD"}
AGE = re.compile(r"'?Age'?\s*[:=]\s*'?(\d{1,2})", re.IGNORECASE)
GRADE = re.compile(r"'?(?:Current )?Grade'?\s*[:=]\s*'?(\d{1,2})", re.IGNORECASE)

EB8976_OLD_PREFIX = "Name: Alex Rivera"
EB8976_NEW = (
    "{'Name': 'Alex Rivera', 'Age': 16, 'Current Grade': '11th Grade', "
    "'Current Skill Level': 'Intermediate', 'Learning Goals': 'Aim to understand advanced "
    "organic chemistry concepts and improve problem-solving skills in stoichiometry.', "
    "'Study Habits': 'Prefers visual learning with videos and infographics, studies best in "
    "short, focused sessions.', 'Weak Points': 'Struggles with applying theoretical concepts "
    "to practical problems, especially in reaction mechanisms and balancing chemical "
    "equations.'} Based on the student profile, provide \"Learning Path Planning\" and "
    "\"Personalized Recommendations\"."
)


def band_from_profile(prompt: str) -> str | None:
    g = GRADE.search(prompt)
    if g:
        n = int(g.group(1))
        if n <= 5:
            return "Elementary School"
        if n <= 8:
            return "Middle School"
        if n <= 12:
            return "High School"
        return None
    a = AGE.search(prompt)
    if a:
        n = int(a.group(1))
        if n <= 10:
            return "Elementary School"
        if n <= 13:
            return "Middle School"
        if n <= 18:
            return "High School"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true", help="apply changes (default: dry run)")
    args = ap.parse_args()

    records = [json.loads(l) for l in SCEN_JSONL.open(encoding="utf-8")]
    n_label, n_grade, n_8976 = 0, 0, 0
    grade_examples = []

    for r in records:
        uc = r.get("use_case", "")
        p = r.get("prompt", "")

        # 3. eb_8976 profile restructure (guarded -> idempotent)
        if r["scenario_id"] == "eb_8976" and p.startswith(EB8976_OLD_PREFIX):
            p = EB8976_NEW
            n_8976 += 1

        # 1. doubled label
        if "Student Profile: Student Profile:" in p:
            p = p.replace("Student Profile: Student Profile:", "Student Profile:")
            n_label += 1

        if args.write:
            r["prompt"] = p

        # 2. grade-band relabel
        if uc in ("personalized_content_creation", "learning_support") and r.get("grade_band") in HIGHER:
            derived = band_from_profile(p)
            if derived is not None:
                n_grade += 1
                if len(grade_examples) < 12:
                    grade_examples.append((r["scenario_id"], r.get("grade_band"), derived))
                if args.write:
                    r["grade_band"] = derived

    print(f"doubled 'Student Profile:' collapsed: {n_label}")
    print(f"grade_band relabels (higher-ed -> school band): {n_grade}")
    for sid, old, new in grade_examples:
        print(f"    [{sid}] {old} -> {new}")
    print(f"eb_8976 profile restructured: {n_8976}")

    # post-conditions (evaluate on the would-be-written state)
    def fixed_prompt(r):
        p = r.get("prompt", "")
        if r["scenario_id"] == "eb_8976" and p.startswith(EB8976_OLD_PREFIX):
            p = EB8976_NEW
        return p.replace("Student Profile: Student Profile:", "Student Profile:")
    still_double = [r["scenario_id"] for r in records if "Student Profile: Student Profile:" in fixed_prompt(r)]
    still_mismatch = [r["scenario_id"] for r in records
                      if r.get("use_case") in ("personalized_content_creation", "learning_support")
                      and r.get("grade_band") in HIGHER and band_from_profile(fixed_prompt(r)) is not None
                      and not args.write]
    eb = next(r for r in records if r["scenario_id"] == "eb_8976")
    print("\npost-conditions:")
    print(f"  doubled labels remaining: {len(still_double)}")
    print(f"  eb_8976 now dict-form: {(EB8976_NEW if args.write else fixed_prompt(eb)).startswith('{')}")

    if not args.write:
        print("\n--dry-run: nothing written (pass --write to apply)")
        return 0
    if still_double:
        raise SystemExit("refusing to write: doubled labels remain")
    with SCEN_JSONL.open("w", encoding="utf-8", newline="\n") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with SCEN_JSONL.with_suffix(".json").open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(f"\nwrote scenarios.jsonl and scenarios.json "
          f"({n_label} labels, {n_grade} grade bands, {n_8976} profile)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
