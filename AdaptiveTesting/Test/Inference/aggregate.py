"""Roll per-question result files into a summary table (derived, non-authoritative).

Writes ``Outputs/_summary/summary.csv`` with one row per (benchmark, model).
The per-question files remain the source of truth; this can be regenerated at
any time.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from common import MCQ_OUT_DIR, OPEN_OUT_DIR, SUMMARY_DIR, ensure_dirs

SUMMARY_FIELDS = [
    "benchmark",
    "model",
    "type",
    "n",
    "n_correct_or_pass",
    "score",
    "metric",
]


def _summarize_csv(path: Path, result_col: str, positive: str) -> tuple[int, int]:
    n = 0
    hits = 0
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if result_col not in row:
                continue
            n += 1
            if row[result_col] == positive:
                hits += 1
    return n, hits


def aggregate() -> Path:
    ensure_dirs(SUMMARY_DIR)
    out = SUMMARY_DIR / "summary.csv"
    rows: list[dict] = []

    for bench_dir in sorted(MCQ_OUT_DIR.glob("*")):
        if not bench_dir.is_dir():
            continue
        for f in sorted(bench_dir.glob("*.csv")):
            n, hits = _summarize_csv(f, "result", "correct")
            model = _model_from_file(f)
            rows.append(
                {
                    "benchmark": bench_dir.name,
                    "model": model,
                    "type": "mcq",
                    "n": n,
                    "n_correct_or_pass": hits,
                    "score": round(hits / n, 4) if n else "",
                    "metric": "accuracy",
                }
            )

    for bench_dir in sorted(OPEN_OUT_DIR.glob("*")):
        if not bench_dir.is_dir():
            continue
        for f in sorted(bench_dir.glob("*.judged.csv")):
            n, hits = _summarize_csv(f, "result", "pass")
            model = _model_from_file(f, judged=True)
            rows.append(
                {
                    "benchmark": bench_dir.name,
                    "model": model,
                    "type": "open",
                    "n": n,
                    "n_correct_or_pass": hits,
                    "score": round(hits / n, 4) if n else "",
                    "metric": "pass_rate",
                }
            )

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return out


def _model_from_file(path: Path, judged: bool = False) -> str:
    name = path.name.replace(".judged.csv", "") if judged else path.name.replace(".csv", "")
    return name.replace("__", "/")


def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args()
    out = aggregate()
    print(f"wrote {out}")


if __name__ == "__main__":
    _main()
