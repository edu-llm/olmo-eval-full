"""Generate ATLAS ``atlas_idx_to_question_id.csv`` bridges from olmo-eval tasks.

ATLAS item parameters are keyed by 1-based position ``X<k>`` in the original HF
split. This enumerates the corresponding olmo-eval task in order and writes
``atlas_idx=k -> question_id=str(metadata["id"])`` for the k-th instance, which is
exactly what ``bank.load_bank`` joins against.

Only benchmarks whose olmo-eval task enumerates the *full* HF split in raw order
are safe here (verified: the bank's max X index equals the split size). Run from
the repo root:

    uv run python AdaptiveTesting/Inputs/ATLAS/scripts/build_atlas_idx_bridge.py \
        --benchmark hellaswag winogrande gsm8k
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from olmo_eval.adaptive.benchmarks import cat_benchmarks
from olmo_eval.evals.tasks.common import get_task

REPO_ROOT = Path(__file__).resolve().parents[4]
ATLAS_DIR = REPO_ROOT / "AdaptiveTesting" / "Inputs" / "ATLAS"


def _bank_subdir(benchmark: str) -> str:
    for b in cat_benchmarks():
        if b.name == benchmark:
            return b.bank_subdir
    raise SystemExit(f"unknown ATLAS benchmark: {benchmark}")


def _base_task(benchmark: str) -> str:
    for b in cat_benchmarks():
        if b.name == benchmark:
            return b.base_task
    raise SystemExit(f"unknown ATLAS benchmark: {benchmark}")


def _expected_bank_size(bank_dir: Path) -> int | None:
    params = bank_dir / "irt_item_parameters_combined.csv"
    if not params.is_file():
        return None
    max_x = 0
    with open(params, newline="") as fh:
        for row in csv.DictReader(fh):
            k = int(str(row["X"]).lstrip("X"))
            max_x = max(max_x, k)
    return max_x


def build_bridge(benchmark: str, *, overwrite: bool) -> None:
    subdir = _bank_subdir(benchmark)
    bank_dir = ATLAS_DIR / subdir
    if not bank_dir.is_dir():
        raise SystemExit(f"bank dir not found: {bank_dir}")

    out_path = bank_dir / "atlas_idx_to_question_id.csv"
    if out_path.exists() and not overwrite:
        raise SystemExit(f"{out_path} exists; pass --overwrite to replace")

    task = get_task(_base_task(benchmark))
    instances = list(task.instances)
    n = len(instances)

    max_x = _expected_bank_size(bank_dir)
    if max_x is not None and max_x != n:
        raise SystemExit(
            f"{benchmark}: task enumerated {n} instances but bank max X index is "
            f"{max_x}. Positional bridge would be misaligned; aborting."
        )

    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["atlas_idx", "question_id"])
        for k, inst in enumerate(instances, start=1):
            qid = inst.metadata.get("id")
            if qid is None:
                raise SystemExit(f"{benchmark}: instance {k} has no metadata['id']")
            writer.writerow([k, str(qid)])

    print(f"{benchmark}: wrote {n} rows -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", nargs="+", required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    for b in args.benchmark:
        build_bridge(b, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
