"""Build an EXPERIMENTAL rubric-bank Q-matrix variant for the candidate 3-skill
structure ``[correctness, scaffolding, motivation]`` -- a read-only calibration probe.

This NEVER touches ``data/TutorBench/curated/rubrics_qmatrix_curated.jsonl``. It emits a COPY
under ``data/TutorBench/experimental/`` whose every record is byte-identical to the curated
bank EXCEPT its ``q_mapping``, which is replaced by the candidate mapping.

Candidate skill definition (per criterion)
-------------------------------------------
* correctness = content OR diagnosis  (logical OR of the two curated q bits)
* scaffolding = the curated scaffolding bit (unchanged)
* motivation  = 1 iff the affect/motivation text classifier tagged this criterion,
                else 0.

The motivation tag (PRIMARY RULE): a criterion is motivation=1 iff the heuristic
classifier ``scripts/classify_orphan_criteria.py`` fired its affect bank on the
criterion text, i.e. the per-item CSV column ``hit_affect_motivation == 1``. This
is the MULTI-LABEL affect hit, so a criterion counts as motivation even when it
ALSO loads correctness or scaffolding (matching the task's "even if it already
loads content/scaffolding" clause). It is a HEURISTIC keyword tag -- fine for a
candidate test, needs a hand audit before anything is committed.

Slot-repurposing trick (why the unmodified fitter accepts this)
---------------------------------------------------------------
``scripts/calibrate_mirt.py`` reads each Q row as
``[q_mapping.get(s, 0) for s in SKILLS]`` with ``SKILLS = (content, diagnosis,
scaffolding)`` and is otherwise agnostic to what those names MEAN. So we pack the
candidate skills into the three existing slots:

    q_mapping["content"]     <-  correctness
    q_mapping["diagnosis"]   <-  scaffolding
    q_mapping["scaffolding"] <-  motivation

The fitter then estimates a 3-dim confirmatory M2PL whose latent dims (in SKILLS
order) are ACTUALLY [correctness, scaffolding, motivation]. The output columns
``a_content / a_diagnosis / a_scaffolding`` therefore mean
``a_correctness / a_scaffolding / a_motivation`` respectively. No source edit to
the fitter is required.

Usage
-----
    python scripts/build_motivation_qmatrix.py
    python scripts/build_motivation_qmatrix.py \
        --curated data/TutorBench/curated/rubrics_qmatrix_curated.jsonl \
        --labels  staging/run2_orphan/orphan_criteria.csv \
        --matrix  staging/response_matrix.csv \
        --out     data/TutorBench/experimental/rubrics_qmatrix_collapse_motivation.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CURATED = ROOT / "data" / "TutorBench" / "curated" / "rubrics_qmatrix_curated.jsonl"
DEFAULT_LABELS = ROOT / "staging" / "run2_orphan" / "orphan_criteria.csv"
DEFAULT_MATRIX = ROOT / "staging" / "response_matrix.csv"
DEFAULT_OUT = ROOT / "data" / "TutorBench" / "experimental" / "rubrics_qmatrix_collapse_motivation.jsonl"

CURATED_NAME = "rubrics_qmatrix_curated.jsonl"

# Slot order = tutor_cat.SKILLS. The candidate skills are packed into these slots.
CANDIDATE_ORDER = ("correctness", "scaffolding", "motivation")
SLOT_KEYS = ("content", "diagnosis", "scaffolding")  # what the fitter reads


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


def load_motivation_ids(labels_csv: Path) -> set[str]:
    """criterion_ids the classifier tagged affect_motivation (hit_affect_motivation==1)."""
    if not labels_csv.is_file():
        raise FileNotFoundError(f"classifier labels CSV not found: {labels_csv}")
    ids: set[str] = set()
    with labels_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "hit_affect_motivation" not in (reader.fieldnames or []):
            raise ValueError(
                f"{labels_csv} lacks 'hit_affect_motivation' column; "
                f"columns={reader.fieldnames}"
            )
        for row in reader:
            if str(row.get("hit_affect_motivation", "0")).strip() in {"1", "1.0", "True", "true"}:
                cid = row.get("criterion_id")
                if cid:
                    ids.add(cid)
    return ids


def load_matrix_columns(matrix_csv: Path) -> set[str] | None:
    """Non-optional criterion_ids that are actually columns in the response matrix."""
    if not matrix_csv.is_file():
        return None
    with matrix_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, [])
    # first column is the model/person index label; the rest are criterion_ids
    return set(header[1:]) if header else set()


def build(records: list[dict], motivation_ids: set[str]) -> tuple[list[dict], dict]:
    out: list[dict] = []
    n_correct = n_scaff = n_motiv = n_allzero = 0
    n_motiv_rescued = 0  # motivation=1 on an otherwise all-zero (correctness=0,scaffolding=0) row
    for rec in records:
        rec = dict(rec)  # shallow copy; we only replace q_mapping
        qm = rec.get("q_mapping") or {}
        content = int(qm.get("content", 0))
        diagnosis = int(qm.get("diagnosis", 0))
        scaffolding = int(qm.get("scaffolding", 0))

        correctness = int(bool(content or diagnosis))
        motivation = int(rec.get("criterion_id") in motivation_ids)

        rec["q_mapping"] = {
            "content": correctness,     # slot repurposed -> correctness
            "diagnosis": scaffolding,   # slot repurposed -> scaffolding
            "scaffolding": motivation,  # slot repurposed -> motivation
        }
        n_correct += correctness
        n_scaff += scaffolding
        n_motiv += motivation
        if correctness == 0 and scaffolding == 0 and motivation == 0:
            n_allzero += 1
        if motivation == 1 and correctness == 0 and scaffolding == 0:
            n_motiv_rescued += 1
        out.append(rec)

    stats = {
        "n_records": len(records),
        "correctness_1": n_correct,
        "scaffolding_1": n_scaff,
        "motivation_1": n_motiv,
        "all_zero_q": n_allzero,
        "motivation_only_rescued": n_motiv_rescued,
    }
    return out, stats


def subset_stats(records: list[dict], cols: set[str] | None) -> dict | None:
    if cols is None:
        return None
    n_correct = n_scaff = n_motiv = n_allzero = n = 0
    for rec in records:
        if rec.get("criterion_id") not in cols:
            continue
        n += 1
        qm = rec["q_mapping"]
        c, s, m = qm["content"], qm["diagnosis"], qm["scaffolding"]
        n_correct += c
        n_scaff += s
        n_motiv += m
        if not (c or s or m):
            n_allzero += 1
    return {
        "n_matrix_columns": n,
        "correctness_1": n_correct,
        "scaffolding_1": n_scaff,
        "motivation_1": n_motiv,
        "all_zero_q": n_allzero,
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
    p.add_argument("--curated", type=Path, default=DEFAULT_CURATED)
    p.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    p.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    if args.out.name == CURATED_NAME or args.out.resolve() == args.curated.resolve():
        print("ERROR: refusing to overwrite the curated bank.", file=sys.stderr)
        return 2

    records = read_jsonl(args.curated)
    motivation_ids = load_motivation_ids(args.labels)
    cols = load_matrix_columns(args.matrix)

    out_records, stats = build(records, motivation_ids)
    sub = subset_stats(out_records, cols)

    write_jsonl(args.out, out_records)

    print("=" * 72)
    print("EXPERIMENTAL Q-matrix variant: [correctness, scaffolding, motivation]")
    print("  slot repurposing: content<-correctness, diagnosis<-scaffolding, "
          "scaffolding<-motivation")
    print("=" * 72)
    print(f"motivation tag rule : classifier hit_affect_motivation==1 "
          f"({len(motivation_ids)} tagged criterion_ids in labels CSV)")
    print(f"curated records     : {stats['n_records']}")
    print(f"  correctness=1     : {stats['correctness_1']}")
    print(f"  scaffolding=1     : {stats['scaffolding_1']}")
    print(f"  motivation=1      : {stats['motivation_1']}")
    print(f"  all-zero Q        : {stats['all_zero_q']}")
    print(f"  motivation rescues (motivation=1 & correctness=0 & scaffolding=0): "
          f"{stats['motivation_only_rescued']}")
    if sub is not None:
        print(f"\nrestricted to the {sub['n_matrix_columns']} response-matrix columns "
              "(non-optional criteria actually fitted):")
        print(f"  correctness=1     : {sub['correctness_1']}")
        print(f"  scaffolding=1     : {sub['scaffolding_1']}")
        print(f"  motivation=1      : {sub['motivation_1']}")
        print(f"  all-zero Q        : {sub['all_zero_q']}")
    print(f"\nwrote experimental bank -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
