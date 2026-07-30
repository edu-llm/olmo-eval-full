"""Stratified 10K subset of the cleaned Dr. SCI scenario/rubric JSONL.

For each split we sample an EQUAL number of scenarios per subject, uniformly at
random within each subject (fixed seed for reproducibility), then carry along the
matching rubric entries.

Quotas:
  verifiable  : 1,250 per subject x 8 subjects = 10,000
  open-ended  : 1,429 per subject x 7 subjects = 10,003  (no `math` subject)

Reads (data/dr-sci/):
  scenario_verifiable.jsonl / scenario_openended.jsonl
  rubrics_verifiable.jsonl  / rubrics_openended.jsonl
Writes (data/dr-sci/):
  scenario_verifiable_10k.jsonl / scenario_openended_10k.jsonl
  rubrics_verifiable_10k.jsonl  / rubrics_openended_10k.jsonl

Original dsv_/dso_ scenario_ids are preserved so provenance and criterion_id
links stay intact. Nothing overwrites the existing full files.
"""
import json
import os
import random
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
DR_SCI = os.path.normpath(os.path.join(HERE, "..", "data", "dr-sci"))
SEED = 42

SPLITS = {
    "verifiable": {"quota": 1250, "has_math": True},
    "open-ended": {"quota": 1429, "has_math": False},
}
SCEN = {"verifiable": "scenario_verifiable.jsonl", "open-ended": "scenario_openended.jsonl"}
RUB = {"verifiable": "rubrics_verifiable.jsonl", "open-ended": "rubrics_openended.jsonl"}
SCEN_OUT = {"verifiable": "scenario_verifiable_10k.jsonl", "open-ended": "scenario_openended_10k.jsonl"}
RUB_OUT = {"verifiable": "rubrics_verifiable_10k.jsonl", "open-ended": "rubrics_openended_10k.jsonl"}


def index_by_subject(path):
    by_subj = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            by_subj.setdefault(r["subject"], []).append(r["scenario_id"])
    return by_subj


def sample_split(kind):
    quota = SPLITS[kind]["quota"]
    scen_path = os.path.join(DR_SCI, SCEN[kind])
    by_subj = index_by_subject(scen_path)

    # Feasibility + expected-subject checks.
    print(f"\n[{kind}] available per subject: "
          + ", ".join(f"{s}={len(ids)}" for s, ids in sorted(by_subj.items())))
    if ("math" in by_subj) != SPLITS[kind]["has_math"]:
        raise SystemExit(f"[{kind}] unexpected subject set (math present={'math' in by_subj})")
    for subj, ids in by_subj.items():
        if len(ids) < quota:
            raise SystemExit(f"[{kind}] subject {subj!r} has {len(ids)} < quota {quota}")

    rng = random.Random(SEED)
    selected = set()
    for subj in sorted(by_subj):
        selected.update(rng.sample(by_subj[subj], quota))

    # Pass 2: write sampled scenario lines verbatim (preserve original ids).
    scen_out = os.path.join(DR_SCI, SCEN_OUT[kind])
    n = 0
    with open(scen_path, encoding="utf-8") as fin, open(scen_out, "w", encoding="utf-8") as fout:
        for line in fin:
            sid = json.loads(line)["scenario_id"]
            if sid in selected:
                fout.write(line if line.endswith("\n") else line + "\n")
                n += 1
    print(f"[{kind}] wrote {n} scenarios -> {SCEN_OUT[kind]} (expected {quota * len(by_subj)})")

    # Filter rubrics to sampled scenarios.
    rub_path = os.path.join(DR_SCI, RUB[kind])
    rub_out = os.path.join(DR_SCI, RUB_OUT[kind])
    m = 0
    with open(rub_path, encoding="utf-8") as fin, open(rub_out, "w", encoding="utf-8") as fout:
        for line in fin:
            if json.loads(line)["scenario_id"] in selected:
                fout.write(line if line.endswith("\n") else line + "\n")
                m += 1
    print(f"[{kind}] wrote {m} rubric criteria -> {RUB_OUT[kind]}")
    return selected


if __name__ == "__main__":
    for kind in SPLITS:
        sample_split(kind)
    print("\nDONE")
