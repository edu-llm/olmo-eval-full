"""Standalone Prometheus judging pass over already-generated open responses.

The parallel sweep generates open-ended responses with judging disabled, then
this runs once with a single resident judge. Keeping one judge for the whole
fleet (instead of one per worker) is what keeps the co-located workers within a
single GPU's memory.

Judging is resumable: rows already present in a ``judged.csv`` are skipped.

Examples:
  python judge_all.py --benchmarks frq
  python judge_all.py --benchmarks tutorbench,bridge
"""

from __future__ import annotations

import argparse
import sys
import time

from common import OPEN_OUT_DIR, bootstrap_env
from config import InferenceConfig, JudgeConfig
from datasets_registry import BENCHMARKS, OPEN_BENCHMARKS
from judge_prometheus import Judge


def resolve(arg: str) -> list[str]:
    if arg.strip() in {"all", "open", "frq"}:
        return list(OPEN_BENCHMARKS)
    out = []
    for tok in arg.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok not in BENCHMARKS:
            raise SystemExit(f"unknown benchmark: {tok}")
        out.append(tok)
    return out


def pending(benchmark: str) -> list[tuple[str, str]]:
    """Return (model_id, path) for every responses file of a benchmark."""
    bench_dir = OPEN_OUT_DIR / benchmark
    if not bench_dir.is_dir():
        return []
    found: list[tuple[str, str]] = []
    for path in sorted(bench_dir.glob("*.responses.jsonl")):
        slug = path.name[: -len(".responses.jsonl")]
        found.append((slug.replace("__", "/"), str(path)))
    return found


def main(argv: list[str] | None = None) -> int:
    # Must precede the first HTTPS connection (TLS trust store + .env HF_TOKEN).
    bootstrap_env()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmarks", default="open", help="open | all | comma list")
    ap.add_argument("--judge-config", default=None)
    ap.add_argument("--inference-config", default=None)
    ap.add_argument("--backend", choices=["vllm", "hf", "mock"], default=None)
    args = ap.parse_args(argv)

    from pathlib import Path

    icfg = InferenceConfig.load(Path(args.inference_config) if args.inference_config else None)
    jcfg = JudgeConfig.load(Path(args.judge_config) if args.judge_config else None)

    benchmarks = resolve(args.benchmarks)
    work = {b: pending(b) for b in benchmarks}
    total = sum(len(v) for v in work.values())
    if not total:
        print("nothing to judge (no responses files found)", flush=True)
        return 0

    print(f"judging {total} response files across {len(benchmarks)} benchmarks", flush=True)
    judge = Judge(jcfg, backend=args.backend)
    try:
        for benchmark in benchmarks:
            items = work[benchmark]
            if not items:
                continue
            print(f"== judge {benchmark} ({len(items)} models) ==", flush=True)
            for model_id, _ in items:
                t0 = time.monotonic()
                try:
                    n = judge.judge_file(
                        benchmark, model_id, BENCHMARKS[benchmark].rubric_key, icfg.writer
                    )
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"  [skip] {benchmark} x {model_id}: {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    continue
                dt = time.monotonic() - t0
                print(f"  [judged] {benchmark:12s} x {model_id:40s} +{n:5d} ({dt:.1f}s)", flush=True)
    finally:
        judge.close()
    print("judging done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
