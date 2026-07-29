"""Build an EXPERIMENTAL rubric-bank Q-matrix variant for a candidate 3-skill
structure ``[correctness, scaffolding, <candidate>]`` -- a read-only calibration
probe. GENERALIZATION of ``scripts/build_motivation_qmatrix.py`` (the motivation run
is just this script with ``--candidate motivation --hit-col hit_affect_motivation``).

This NEVER touches ``data/curated/rubrics_qmatrix_curated.jsonl``. It emits a COPY
under ``data/experimental/`` whose every record is byte-identical to the curated
bank EXCEPT its ``q_mapping``, which is replaced by the candidate mapping.

Candidate skill definition (per criterion)
-------------------------------------------
* correctness = content OR diagnosis  (logical OR of the two curated q bits)
* scaffolding = the curated scaffolding bit (unchanged)
* <candidate>  = 1 iff the heuristic classifier column ``--hit-col`` == 1, else 0.

The candidate tag comes from ``scripts/classify_orphan_criteria.py`` -- a MULTI-LABEL
keyword hit (a criterion counts even when it ALSO loads correctness/scaffolding). It
is a HEURISTIC keyword tag -- fine for a candidate test, needs a hand audit before
anything is committed. Report the exact ``--hit-col`` used.

Slot-repurposing trick (why the unmodified fitter accepts this)
---------------------------------------------------------------
``scripts/calibrate_mirt.py`` reads each Q row as
``[q_mapping.get(s, 0) for s in SKILLS]`` with ``SKILLS = (content, diagnosis,
scaffolding)`` and is otherwise agnostic to what those names MEAN. So we pack the
candidate skills into the three existing slots:

    q_mapping["content"]     <-  correctness
    q_mapping["diagnosis"]   <-  scaffolding
    q_mapping["scaffolding"] <-  <candidate>

The fitter then estimates a 3-dim confirmatory M2PL whose latent dims (in SKILLS
order) are ACTUALLY [correctness, scaffolding, <candidate>]. The output columns
``a_content / a_diagnosis / a_scaffolding`` therefore mean
``a_correctness / a_scaffolding / a_<candidate>``. No source edit is required.

Usage
-----
    python scripts/build_candidate_qmatrix.py --candidate metacognition \
        --hit-col hit_metacognitive
    python scripts/build_candidate_qmatrix.py --candidate communication \
        --hit-col hit_communication_clarity
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CURATED = ROOT / "data" / "curated" / "rubrics_qmatrix_curated.jsonl"
DEFAULT_LABELS = ROOT / "staging" / "run2_orphan" / "orphan_criteria.csv"
DEFAULT_MATRIX = ROOT / "staging" / "response_matrix.csv"

CURATED_NAME = "rubrics_qmatrix_curated.jsonl"

SLOT_KEYS = ("content", "diagnosis", "scaffolding")  # what the fitter reads
TRUEY = {"1", "1.0", "True", "true"}


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"not found: {path}")
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_candidate_ids(labels_csv: Path, hit_col: str) -> set[str]:
    """criterion_ids the classifier tagged for the candidate (``hit_col``==1)."""
    if not labels_csv.is_file():
        raise FileNotFoundError(f"classifier labels CSV not found: {labels_csv}")
    ids: set[str] = set()
    with labels_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if hit_col not in (reader.fieldnames or []):
            raise ValueError(
                f"{labels_csv} lacks '{hit_col}' column; columns={reader.fieldnames}"
            )
        for row in reader:
            if str(row.get(hit_col, "0")).strip() in TRUEY:
                cid = row.get("criterion_id")
                if cid:
                    ids.add(cid)
    return ids


def load_matrix_columns(matrix_csv: Path) -> set[str] | None:
    if not matrix_csv.is_file():
        return None
    with matrix_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, [])
    return set(header[1:]) if header else set()


def build(records: list[dict], candidate_ids: set[str]) -> tuple[list[dict], dict]:
    out: list[dict] = []
    n_correct = n_scaff = n_cand = n_allzero = 0
    n_cand_rescued = 0  # candidate=1 on an otherwise all-zero (correctness=0,scaff=0) row
    for rec in records:
        rec = dict(rec)  # shallow copy; we only replace q_mapping
        qm = rec.get("q_mapping") or {}
        content = int(qm.get("content", 0))
        diagnosis = int(qm.get("diagnosis", 0))
        scaffolding = int(qm.get("scaffolding", 0))

        correctness = int(bool(content or diagnosis))
        candidate = int(rec.get("criterion_id") in candidate_ids)

        rec["q_mapping"] = {
            "content": correctness,      # slot repurposed -> correctness
            "diagnosis": scaffolding,    # slot repurposed -> scaffolding
            "scaffolding": candidate,    # slot repurposed -> candidate
        }
        n_correct += correctness
        n_scaff += scaffolding
        n_cand += candidate
        if correctness == 0 and scaffolding == 0 and candidate == 0:
            n_allzero += 1
        if candidate == 1 and correctness == 0 and scaffolding == 0:
            n_cand_rescued += 1
        out.append(rec)

    stats = {
        "n_records": len(records),
        "correctness_1": n_correct,
        "scaffolding_1": n_scaff,
        "candidate_1": n_cand,
        "all_zero_q": n_allzero,
        "candidate_only_rescued": n_cand_rescued,
    }
    return out, stats


def subset_stats(records: list[dict], cols: set[str] | None) -> dict | None:
    if cols is None:
        return None
    n_correct = n_scaff = n_cand = n_allzero = n = n_rescued = 0
    for rec in records:
        if rec.get("criterion_id") not in cols:
            continue
        n += 1
        qm = rec["q_mapping"]
        c, s, m = qm["content"], qm["diagnosis"], qm["scaffolding"]
        n_correct += c
        n_scaff += s
        n_cand += m
        if not (c or s or m):
            n_allzero += 1
        if m == 1 and c == 0 and s == 0:
            n_rescued += 1
    return {
        "n_matrix_columns": n,
        "correctness_1": n_correct,
        "scaffolding_1": n_scaff,
        "candidate_1": n_cand,
        "all_zero_q": n_allzero,
        "candidate_only_rescued": n_rescued,
    }


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--candidate", required=True,
                   help="candidate axis name (e.g. metacognition, communication).")
    p.add_argument("--hit-col", required=True,
                   help="classifier CSV column used as the candidate tag "
                        "(e.g. hit_metacognitive, hit_communication_clarity).")
    p.add_argument("--curated", type=Path, default=DEFAULT_CURATED)
    p.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    p.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    out = args.out or (
        ROOT / "data" / "experimental"
        / f"rubrics_qmatrix_collapse_{args.candidate}.jsonl"
    )

    if out.name == CURATED_NAME or out.resolve() == args.curated.resolve():
        print("ERROR: refusing to overwrite the curated bank.", file=sys.stderr)
        return 2

    records = read_jsonl(args.curated)
    candidate_ids = load_candidate_ids(args.labels, args.hit_col)
    cols = load_matrix_columns(args.matrix)

    out_records, stats = build(records, candidate_ids)
    sub = subset_stats(out_records, cols)

    write_jsonl(out, out_records)

    print("=" * 72)
    print(f"EXPERIMENTAL Q-matrix variant: [correctness, scaffolding, {args.candidate}]")
    print(f"  slot repurposing: content<-correctness, diagnosis<-scaffolding, "
          f"scaffolding<-{args.candidate}")
    print("=" * 72)
    print(f"{args.candidate} tag rule : classifier {args.hit_col}==1 "
          f"({len(candidate_ids)} tagged criterion_ids in labels CSV)")
    print(f"curated records     : {stats['n_records']}")
    print(f"  correctness=1     : {stats['correctness_1']}")
    print(f"  scaffolding=1     : {stats['scaffolding_1']}")
    print(f"  {args.candidate}=1 : {stats['candidate_1']}")
    print(f"  all-zero Q        : {stats['all_zero_q']}")
    print(f"  {args.candidate} rescues (candidate=1 & correctness=0 & scaffolding=0): "
          f"{stats['candidate_only_rescued']}")
    if sub is not None:
        print(f"\nrestricted to the {sub['n_matrix_columns']} response-matrix columns "
              "(non-optional criteria actually fitted):")
        print(f"  correctness=1     : {sub['correctness_1']}")
        print(f"  scaffolding=1     : {sub['scaffolding_1']}")
        print(f"  {args.candidate}=1 : {sub['candidate_1']}")
        print(f"  all-zero Q        : {sub['all_zero_q']}")
        print(f"  {args.candidate} rescues (over matrix cols): "
              f"{sub['candidate_only_rescued']}")
    print(f"\nwrote experimental bank -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
