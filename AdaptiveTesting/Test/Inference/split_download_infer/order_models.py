#!/usr/bin/env python3
"""Zig-zag processing order for the roster: small -> medium -> large, repeat.

Sorting the roster by size then round-robin'ing three size tiers means that any
early stop still leaves a representative spread across small / medium / large
models rather than only the smallest N. Both the CPU downloaders and the GPU
queue workers consume this same deterministic order:

  * downloaders take ``order[shard_index::num_shards]`` so weights land in S3
    roughly in zig-zag order;
  * GPU workers always claim the lowest-index model that is ready, so inference
    proceeds in zig-zag order regardless of exactly when each download finishes.

    python order_models.py                       # idx<TAB>id<TAB>cache_dir<TAB>params_b
    python order_models.py --field id            # just ids, in order
    python order_models.py --shard-index 0 --num-shards 16 --field id
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
INF = HERE.parent
if str(INF) not in sys.path:
    sys.path.insert(0, str(INF))

from models_registry import ModelSpec, load_models  # noqa: E402


def cache_dir_name(repo_id: str) -> str:
    return "models--" + repo_id.replace("/", "--")


def zigzag(specs: list[ModelSpec]) -> list[ModelSpec]:
    """Sort by size, split into 3 contiguous tiers, then round-robin them."""
    ordered = sorted(specs, key=lambda s: (s.params_b, s.id))
    n = len(ordered)
    if n == 0:
        return []
    t = (n + 2) // 3  # ceil(n/3)
    tiers = [ordered[0:t], ordered[t : 2 * t], ordered[2 * t :]]
    out: list[ModelSpec] = []
    for k in range(t):
        for tier in tiers:
            if k < len(tier):
                out.append(tier[k])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-yaml", default=None, help="roster path (default: registry MODELS_YAML)")
    ap.add_argument("--shard-index", type=int, default=None)
    ap.add_argument("--num-shards", type=int, default=None)
    ap.add_argument(
        "--field",
        choices=["row", "id", "cache_dir"],
        default="row",
        help="row = idx<TAB>id<TAB>cache_dir<TAB>params_b (default)",
    )
    ap.add_argument("--count", action="store_true", help="print the model count and exit")
    args = ap.parse_args()

    roster = Path(args.models_yaml) if args.models_yaml else None
    order = zigzag(load_models(roster))

    if args.count:
        print(len(order))
        return 0

    indexed = list(enumerate(order))
    if args.shard_index is not None and args.num_shards:
        indexed = [(i, s) for (i, s) in indexed if i % args.num_shards == args.shard_index]

    for i, s in indexed:
        if args.field == "id":
            print(s.id)
        elif args.field == "cache_dir":
            print(cache_dir_name(s.id))
        else:
            print(f"{i}\t{s.id}\t{cache_dir_name(s.id)}\t{s.params_b}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
