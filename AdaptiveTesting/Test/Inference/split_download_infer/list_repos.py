#!/usr/bin/env python3
"""Print the repo ids (or HF-cache dir names) for a roster / shard.

Used by ``pull_cache_from_s3.sh`` to build ``aws s3 sync --include`` filters so a
GPU worker downloads only its shard's weights instead of the whole cache.

    python list_repos.py --shard-index 0 --num-shards 8            # repo ids
    python list_repos.py --shard-index 0 --num-shards 8 --cache-dirs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
INF = HERE.parent
if str(INF) not in sys.path:
    sys.path.insert(0, str(INF))

from models_registry import ModelSpec, load_models, select_models  # noqa: E402


def cache_dir_name(repo_id: str) -> str:
    """HF hub cache folder for a repo: ``models--org--name``."""
    return "models--" + repo_id.replace("/", "--")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-yaml", default=None)
    ap.add_argument("--models", default=None)
    ap.add_argument("--max-params-b", type=float, default=None)
    ap.add_argument("--shard-index", type=int, default=None)
    ap.add_argument("--num-shards", type=int, default=None)
    ap.add_argument("--cache-dirs", action="store_true", help="print models--org--name folders")
    ap.add_argument("--include-tokenizers", action="store_true", help="also emit borrowed tokenizer repos")
    args = ap.parse_args()

    roster = Path(args.models_yaml) if args.models_yaml else None
    specs = select_models(
        load_models(roster),
        only=args.models.split(",") if args.models else None,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    if args.max_params_b is not None:
        specs = [s for s in specs if s.params_b <= args.max_params_b]

    ids: list[str] = []
    for s in specs:
        ids.append(s.id)
        if args.include_tokenizers and s.tokenizer_id and s.tokenizer_id != s.id:
            ids.append(s.tokenizer_id)

    seen: set[str] = set()
    for repo_id in ids:
        if repo_id in seen:
            continue
        seen.add(repo_id)
        print(cache_dir_name(repo_id) if args.cache_dirs else repo_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
