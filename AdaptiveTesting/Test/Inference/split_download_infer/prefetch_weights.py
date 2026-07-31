#!/usr/bin/env python3
"""Download model weights for the roster into the HF cache (no GPU required).

Part of the split-cost workflow: run this on a cheap CPU box, sync ``HF_HOME``
to S3, then let GPU workers pull the warm cache and run inference. Nothing here
imports ``torch`` or ``vllm`` - only ``huggingface_hub`` - so it runs on any
machine with network + disk.

The download uses the *standard* hub cache layout
(``$HF_HOME/hub/models--org--name/snapshots/<sha>/``) so that vLLM later
resolves each model by its repo id straight from the synced cache.

    python prefetch_weights.py --models-yaml ../../../Inputs/Models/models.yaml
    python prefetch_weights.py --max-params-b 1.5 --workers 4
    python prefetch_weights.py --shard-index 0 --num-shards 8   # split the roster
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Reuse the real registry so "what to download" never drifts from the sweep.
HERE = Path(__file__).resolve().parent
INF = HERE.parent
if str(INF) not in sys.path:
    sys.path.insert(0, str(INF))

from models_registry import ModelSpec, load_models, select_models  # noqa: E402

# Framework-specific artifacts vLLM never reads; skipping them saves cache/S3.
DEFAULT_IGNORE = ["*.h5", "*.msgpack", "*.onnx", "*.onnx_data", "*.tflite", "*.pb"]


def _targets(spec: ModelSpec, safetensors_only: bool) -> list[tuple[str, str | None, list[str] | None]]:
    """Repos to fetch for one model: the weights, plus a borrowed tokenizer."""
    ignore = list(DEFAULT_IGNORE)
    if safetensors_only:
        ignore.append("*.bin")
    out: list[tuple[str, str | None, list[str] | None]] = [(spec.id, spec.revision, ignore)]
    if spec.tokenizer_id and spec.tokenizer_id != spec.id:
        # Only tokenizer/config files are needed from a borrowed tokenizer repo.
        out.append((spec.tokenizer_id, None, ["*.bin", "*.safetensors", *DEFAULT_IGNORE]))
    return out


def _hub_dir() -> Path:
    home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    return Path(home) / "hub"


def _cache_dir_name(repo_id: str) -> str:
    return "models--" + repo_id.replace("/", "--")


def _push_and_free(repo_id: str, s3_prefix: str, region: str, delete_after: bool) -> str:
    """Sync one repo's cache folder to S3, then optionally delete it locally.

    Keeps peak disk flat and persists each model the moment it finishes, so a
    crash never loses completed downloads.
    """
    repo_dir = _hub_dir() / _cache_dir_name(repo_id)
    if not repo_dir.is_dir():
        return "pushed?nolocaldir"
    dest = f"{s3_prefix.rstrip('/')}/hub/{repo_dir.name}/"
    subprocess.run(
        ["aws", "s3", "sync", str(repo_dir), dest, "--region", region, "--only-show-errors"],
        check=True,
    )
    if delete_after:
        shutil.rmtree(repo_dir, ignore_errors=True)
    return "pushed+freed" if delete_after else "pushed"


def _download(repo_id: str, revision: str | None, ignore: list[str] | None,
              token: str | None, retries: int, s3_prefix: str | None,
              region: str, delete_after: bool) -> tuple[str, str, str]:
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            snapshot_download(
                repo_id=repo_id,
                revision=revision,
                token=token,
                ignore_patterns=ignore,
            )
            detail = "ok"
            if s3_prefix:
                # Push+free is itself retried by the outer loop on failure.
                detail = _push_and_free(repo_id, s3_prefix, region, delete_after)
            return repo_id, "ok", detail
        except (GatedRepoError, RepositoryNotFoundError) as exc:
            # Not retryable: missing access / typo. Report and move on.
            return repo_id, "skip", f"{type(exc).__name__}: {exc}"
        except Exception as exc:  # noqa: BLE001  (network/timeouts are retryable)
            last = exc
            if attempt < retries:
                time.sleep(min(2 ** attempt, 30))
    return repo_id, "fail", repr(last)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-yaml", default=None, help="roster path (default: registry MODELS_YAML)")
    ap.add_argument("--models", default=None, help="comma list of HF ids / trailing names to limit to")
    ap.add_argument("--max-params-b", type=float, default=None, help="only models <= this size (B)")
    ap.add_argument("--shard-index", type=int, default=None)
    ap.add_argument("--num-shards", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4, help="repo-level parallelism (hf_transfer parallelizes within)")
    ap.add_argument("--retries", type=int, default=4, help="retries per repo on transient errors")
    ap.add_argument("--safetensors-only", action="store_true", help="skip *.bin when safetensors exist")
    ap.add_argument("--manifest", default=None, help="write JSON result manifest here")
    ap.add_argument("--s3-prefix", default=None,
                    help="stream each model to this S3 cache prefix as soon as it finishes")
    ap.add_argument("--region", default=os.environ.get("AWS_REGION", "us-east-1"))
    ap.add_argument("--delete-after", action="store_true",
                    help="with --s3-prefix: rm each model locally after it is pushed (flat disk)")
    ap.add_argument("--dry-run", action="store_true", help="list what would be fetched, then exit")
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

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    hf_home = os.environ.get("HF_HOME", "(default ~/.cache/huggingface)")
    print(f"prefetch_weights: {len(specs)} model(s) -> HF_HOME={hf_home}", flush=True)
    if token:
        print(f"  HF token present ({len(token)} chars) - gated repos enabled", flush=True)
    else:
        print("  WARNING: no HF_TOKEN in env - gated models will be skipped", flush=True)

    jobs: list[tuple[str, str | None, list[str] | None]] = []
    seen: set[str] = set()
    for s in specs:
        for repo_id, rev, ignore in _targets(s, args.safetensors_only):
            key = f"{repo_id}@{rev or 'main'}"
            if key in seen:
                continue
            seen.add(key)
            jobs.append((repo_id, rev, ignore))

    if args.dry_run:
        for repo_id, rev, _ in jobs:
            print(f"  would fetch {repo_id} (rev={rev or 'main'})")
        print(f"total repos: {len(jobs)}")
        return 0

    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run bootstrap_downloader.sh first.", file=sys.stderr)
        return 2

    results: list[dict[str, str]] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futs = {
            pool.submit(_download, repo_id, rev, ignore, token, args.retries,
                        args.s3_prefix, args.region, args.delete_after): repo_id
            for repo_id, rev, ignore in jobs
        }
        done = 0
        for fut in as_completed(futs):
            repo_id, status, detail = fut.result()
            done += 1
            tag = {"ok": "OK  ", "skip": "SKIP", "fail": "FAIL"}.get(status, status)
            print(f"[{done}/{len(jobs)}] {tag} {repo_id}  {detail if status != 'ok' else ''}".rstrip(), flush=True)
            results.append({"repo_id": repo_id, "status": status, "detail": detail})

    ok = sum(r["status"] == "ok" for r in results)
    skipped = sum(r["status"] == "skip" for r in results)
    failed = sum(r["status"] == "fail" for r in results)
    dt = time.time() - t0
    print(f"\ndone in {dt:.0f}s: ok={ok} skip={skipped} fail={failed}", flush=True)

    if args.manifest:
        Path(args.manifest).write_text(
            json.dumps({"ok": ok, "skip": skipped, "fail": failed, "results": results}, indent=2)
        )
        print(f"manifest -> {args.manifest}", flush=True)

    # Fail the process only on hard failures, not on gated/missing skips.
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
