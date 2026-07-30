#!/usr/bin/env python3
"""Parallel prefetch + normalize all CPU-sweep benchmarks into local JSONL caches."""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

# Allow running from this folder or repo root.
HERE = os.path.dirname(os.path.abspath(__file__))
INF = os.path.dirname(HERE)
if INF not in sys.path:
    sys.path.insert(0, INF)

from datasets_registry import CPU_SWEEP_BENCHMARKS, load_benchmark  # noqa: E402


def _one(name: str, max_samples: int | None, seed: int) -> tuple[str, int | str]:
    try:
        qs = load_benchmark(name, max_samples, seed, use_cache=True)
        return name, len(qs)
    except Exception as exc:  # noqa: BLE001
        return name, f"{type(exc).__name__}: {exc}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--workers", type=int, default=min(8, len(CPU_SWEEP_BENCHMARKS)))
    ap.add_argument(
        "--benchmarks",
        default=",".join(CPU_SWEEP_BENCHMARKS),
        help="comma list (default: full cpu_sweep suite)",
    )
    args = ap.parse_args()
    names = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    print(f"prefetching {len(names)} benchmarks with {args.workers} workers …", flush=True)
    ok = fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(_one, n, args.max_samples, args.seed): n for n in names
        }
        for fut in as_completed(futs):
            name, result = fut.result()
            if isinstance(result, int):
                print(f"  OK  {name:14s}  n={result}", flush=True)
                ok += 1
            else:
                print(f"  FAIL {name:14s}  {result}", flush=True)
                fail += 1
    print(f"done. ok={ok} fail={fail}", flush=True)
    if fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
