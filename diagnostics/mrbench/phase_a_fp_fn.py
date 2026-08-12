"""CLI: per-class FP/FN decision-rate reliability for the cached Phase-A judge.

FREE and offline. It re-analyses the already-cached temp-0 judge outputs (no
judge, API, network or GPU calls), pairing them to gold exactly the way
``judge_run.py --metrics`` does, and reports for every dimension and ordinal
class the false-positive rate (fall-out), false-negative rate (miss rate),
precision, recall and one-vs-rest F1 with both a case-level and a
conversation-id-clustered 95% bootstrap CI.

    # readable table + write fp_fn_metrics.json into the run dir
    uv run python -m diagnostics.mrbench.phase_a_fp_fn

    # faster smoke pass with fewer resamples
    uv run python -m diagnostics.mrbench.phase_a_fp_fn --bootstrap 200

The machine-readable ``fp_fn_metrics.json`` is written next to (never over) the
existing ``metrics.json`` in the run directory.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

from .gold_damr import load_conversations
from .judge_cache import JudgeCache
from .metrics import build_gold_index
from .reference import DESIRED_LABELS, DIMENSION_HEADERS, DIMENSIONS
from .reliability import DimensionReliability, compute_reliability

DEFAULT_CACHE = Path(__file__).parent / "runs" / "judge_cache.jsonl"
DEFAULT_RUN_DIR = Path(__file__).parent / "runs" / "phase_a_haiku_4_5_2026-08-09"

# BEA-2025 recorded secondary number to cross-check against (PHASE_A_VALIDATION.md).
RECORDED_MACRO_F1_MEAN = 0.48


def _fmt_rate(v: float) -> str:
    return " nan" if math.isnan(v) else f"{v:.3f}"


def _fmt_ci(ci: tuple[float, float]) -> str:
    lo, hi = ci
    if math.isnan(lo) or math.isnan(hi):
        return "[  nan,   nan]"
    return f"[{lo:.3f}, {hi:.3f}]"


def _ci_width(ci: tuple[float, float]) -> float:
    lo, hi = ci
    if math.isnan(lo) or math.isnan(hi):
        return math.nan
    return hi - lo


def print_report(results: dict[str, DimensionReliability]) -> None:
    print("=" * 100)
    print("MRBench Phase-A — per-class one-vs-rest FP/FN decision-rate reliability")
    print("(offline re-analysis of cached Claude Haiku 4.5 temp-0 judgments; no API calls)")
    print("=" * 100)

    for dim in DIMENSIONS:
        res = results[dim]
        desired_label = DESIRED_LABELS[dim]
        print(
            f"\n{DIMENSION_HEADERS[dim]:>6}  {dim}  "
            f"(n_pairs={res.n_pairs}, conversations={res.n_conversations}, "
            f"macro-F1={res.macro_f1:.4f}, desired='{desired_label}')"
        )
        header = (
            f"  {'class (label)':<34}{'supp':>6}{'FP_rate':>9}"
            f"{'FP CI(case)':>16}{'FP CI(clust)':>16}"
            f"{'FN_rate':>9}{'FN CI(case)':>16}{'FN CI(clust)':>16}"
            f"{'prec':>7}{'rec':>7}{'F1':>7}{'F1 CI(clust)':>16}"
        )
        print(header)
        for cr in res.classes:
            tag = "*" if cr.is_desired else " "
            sparse = " (sparse)" if cr.sparse else ""
            cls_label = f"{tag}{cr.cls}:{cr.label}"[:34]
            print(
                f"  {cls_label:<34}{cr.n_gold_pos:>6}"
                f"{_fmt_rate(cr.fp_rate):>9}{_fmt_ci(cr.fp_rate_ci_case):>16}"
                f"{_fmt_ci(cr.fp_rate_ci_clustered):>16}"
                f"{_fmt_rate(cr.fn_rate):>9}{_fmt_ci(cr.fn_rate_ci_case):>16}"
                f"{_fmt_ci(cr.fn_rate_ci_clustered):>16}"
                f"{_fmt_rate(cr.precision):>7}{_fmt_rate(cr.recall):>7}"
                f"{_fmt_rate(cr.f1):>7}{_fmt_ci(cr.f1_ci_clustered):>16}{sparse}"
            )


def print_crosscheck(results: dict[str, DimensionReliability]) -> float:
    present = [r.macro_f1 for r in results.values() if not math.isnan(r.macro_f1)]
    mean_mf1 = sum(present) / len(present) if present else math.nan
    print("\n" + "-" * 100)
    print("Macro-F1 cross-check (should match metrics.macro_f1 / PHASE_A_VALIDATION.md)")
    print(f"  recomputed per-dimension macro-F1 mean = {mean_mf1:.4f}")
    print(f"  recorded (PHASE_A_VALIDATION.md)        = {RECORDED_MACRO_F1_MEAN:.2f}")
    delta = abs(mean_mf1 - RECORDED_MACRO_F1_MEAN)
    verdict = "MATCH" if delta <= 0.01 else "MISMATCH"
    print(f"  |delta| = {delta:.4f}  ->  {verdict}")
    return mean_mf1


def print_clustering_effect(results: dict[str, DimensionReliability]) -> None:
    print("\n" + "-" * 100)
    print("Clustered vs case-level CI width (mean over classes) — does conversation")
    print("clustering widen the interval?  ratio > 1 means clustering is wider.")
    print(
        f"  {'dimension':<28}{'FP case':>9}{'FP clust':>10}{'ratio':>8}"
        f"{'FN case':>9}{'FN clust':>10}{'ratio':>8}"
    )
    for dim in DIMENSIONS:
        res = results[dim]
        fp_case = _mean_width([c.fp_rate_ci_case for c in res.classes])
        fp_clu = _mean_width([c.fp_rate_ci_clustered for c in res.classes])
        fn_case = _mean_width([c.fn_rate_ci_case for c in res.classes])
        fn_clu = _mean_width([c.fn_rate_ci_clustered for c in res.classes])
        fp_ratio = fp_clu / fp_case if fp_case else math.nan
        fn_ratio = fn_clu / fn_case if fn_case else math.nan
        print(
            f"  {dim:<28}{fp_case:>9.3f}{fp_clu:>10.3f}{fp_ratio:>8.2f}"
            f"{fn_case:>9.3f}{fn_clu:>10.3f}{fn_ratio:>8.2f}"
        )


def _mean_width(cis: list[tuple[float, float]]) -> float:
    widths = [_ci_width(ci) for ci in cis]
    widths = [w for w in widths if not math.isnan(w)]
    return sum(widths) / len(widths) if widths else math.nan


def to_json(results: dict[str, DimensionReliability], mean_macro_f1: float) -> dict:
    return {
        "description": (
            "Per-class one-vs-rest FP/FN decision-rate reliability for the cached "
            "Phase-A Claude Haiku 4.5 judge (offline re-analysis; no API calls). "
            "Rates are fractions in [0, 1]. CIs are 95% percentile bootstrap; "
            "'case' resamples cells, 'clustered' resamples whole conversation_ids."
        ),
        "classes": [1, 2, 3],
        "macro_f1_mean": mean_macro_f1,
        "per_dimension": {
            dim: {
                "n_pairs": res.n_pairs,
                "n_conversations": res.n_conversations,
                "macro_f1": res.macro_f1,
                "desired_label": DESIRED_LABELS[dim],
                "classes": [asdict(cr) for cr in res.classes],
            }
            for dim, res in results.items()
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Offline per-class FP/FN decision-rate reliability for the cached judge."
    )
    parser.add_argument("--data-path", default=None, help="Path to MRBench_V1.json.")
    parser.add_argument("--cache", default=str(DEFAULT_CACHE), help="JSONL judge cache path.")
    parser.add_argument(
        "--run-dir",
        default=str(DEFAULT_RUN_DIR),
        help="Directory to write fp_fn_metrics.json into (never overwrites metrics.json).",
    )
    parser.add_argument(
        "--out", default=None, help="Explicit output JSON path (overrides run-dir)."
    )
    parser.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap resamples.")
    parser.add_argument("--seed", type=int, default=42, help="Bootstrap RNG seed.")
    parser.add_argument("--no-write", action="store_true", help="Print only; do not write JSON.")
    args = parser.parse_args(argv)

    cache_path = Path(args.cache)
    if not cache_path.exists():
        print(f"No judge cache at {cache_path}.")
        print("STOP: cannot find cached Phase-A judgments; refusing to trigger new judging.")
        return 1

    conversations = load_conversations(args.data_path)
    records = JudgeCache(cache_path).records()
    gold_index = build_gold_index(conversations)
    n_paired = sum(
        1
        for rec in records
        if rec.ok
        and rec.score is not None
        and (rec.conversation_id, rec.tutor, rec.dimension) in gold_index
    )
    print(f"Loaded {len(records)} cached records; {n_paired} pair to gold.")

    results = compute_reliability(records, gold_index, n_boot=args.bootstrap, seed=args.seed)

    print_report(results)
    mean_mf1 = print_crosscheck(results)
    print_clustering_effect(results)

    if not args.no_write:
        out_path = Path(args.out) if args.out else Path(args.run_dir) / "fp_fn_metrics.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(to_json(results, mean_mf1), fh, indent=2)
            fh.write("\n")
        print(f"\nWrote machine-readable metrics to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
