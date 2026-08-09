#!/usr/bin/env python3
"""CPU-only Hugging Face cache pre-warmer.

Reads a models manifest (the same `models:`/`id` YAML the respgen pipeline uses)
and downloads each model's weights/tokenizer/config into the standard HF cache
(``$HF_HOME/hub``) IN PARALLEL, so a cheap CPU box can fill a cache that GPU
boxes later restore with ``aws s3 sync`` and load offline — the GPU never spends
GPU time on a cold download.

Design constraints:
  * No torch / vllm / transformers import. Only ``huggingface_hub`` + ``pyyaml``.
    Everything here runs on CPU compute.
  * Safetensors-preferred: when a repo ships a ``*.safetensors`` variant, the
    ``*.bin``/``*.pth``/``*.h5``/``*.msgpack``/ONNX/GGUF duplicates are skipped so
    we move only the bytes the backends actually load. Repos that ship ONLY
    ``*.bin`` (gpt2/pythia/bloom/opt vintages) still download their ``*.bin``.
  * Idempotent: ``snapshot_download`` is a no-op for files already present, so
    re-running (or resuming after a restore-from-S3) only fills gaps.

The S3 upload itself is done by the shell wrapper (``prewarm_hf_cache.sh``) with
``aws s3 sync`` under the box's instance IAM role — this module never touches AWS.
"""

from __future__ import annotations

# hf_transfer must be requested before huggingface_hub is imported for it to take
# effect; set it defensively here too (the shell wrapper also exports it).
import os

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

import argparse
import concurrent.futures as cf
import json
import sys
import time
from pathlib import Path

import yaml

# Non-weight extras that no respgen backend loads; always skipped. Weight-format
# selection (.safetensors vs .bin) is decided per-repo below, not here.
_ALWAYS_IGNORE = [
    "*.gguf",
    "*.onnx",
    "onnx/*",
    "*.onnx_data",
    "*.tflite",
    "*.ot",
    "*.mlmodel",
    "coreml/*",
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.pdf",
    "*.md",
    "*.h5",  # TF weights
    "tf_model.h5",
    "*.msgpack",  # Flax weights
    "flax_model.msgpack",
]

# Dropped ONLY when a safetensors variant is present in the same repo.
_BIN_WHEN_SAFETENSORS = [
    "*.bin",
    "*.bin.index.json",
    "*.pth",
    "*.pt",
    "*.ckpt",
]


def read_manifest_ids(path: str | Path) -> list[str]:
    """Extract model ids from a respgen-style manifest (``models:`` list).

    Kept deliberately minimal (only the ``id`` field) so the pre-warmer stays a
    standalone CPU tool with no dependency on the tutor_cat package/[gen] extra.
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    ids: list[str] = []
    seen: set[str] = set()
    for entry in raw.get("models") or []:
        if isinstance(entry, str):
            mid = entry
        elif isinstance(entry, dict) and "id" in entry:
            mid = str(entry["id"])
        else:
            raise ValueError(f"{path}: model entry missing 'id': {entry!r}")
        if mid not in seen:
            seen.add(mid)
            ids.append(mid)
    if not ids:
        raise ValueError(f"{path}: no models listed")
    return ids


def _patterns_for(repo_id: str, token: str | None) -> tuple[list[str] | None, list[str]]:
    """Return (allow_patterns, ignore_patterns) for one repo.

    Lists the repo's files once (a cheap metadata call) to decide whether a
    safetensors variant exists; if it does, the redundant .bin/.pth/TF/Flax
    weights are added to the ignore set. Falls back to a safe ignore set if the
    listing fails (e.g. transient API error) so a download still proceeds.
    """
    from huggingface_hub import list_repo_files
    from huggingface_hub.utils import HfHubHTTPError

    ignore = list(_ALWAYS_IGNORE)
    try:
        files = list_repo_files(repo_id, token=token)
    except (HfHubHTTPError, OSError):
        return None, ignore
    if any(f.endswith(".safetensors") for f in files):
        ignore.extend(_BIN_WHEN_SAFETENSORS)
    return None, ignore


def _dir_bytes(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            pass
    return total


def prewarm_one(repo_id: str, token: str | None) -> dict:
    """Download one repo's snapshot into the HF cache. Returns a result dict."""
    from huggingface_hub import snapshot_download

    t0 = time.time()
    _, ignore = _patterns_for(repo_id, token)
    try:
        local = snapshot_download(
            repo_id=repo_id,
            token=token,
            ignore_patterns=ignore,
            # Standard hub cache layout (HF_HOME/hub) so a plain restore + offline
            # load finds it by repo id; no local_dir flattening.
        )
    except Exception as exc:  # noqa: BLE001 - report, don't abort the whole pool
        return {
            "id": repo_id,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "seconds": round(time.time() - t0, 1),
        }
    elapsed = time.time() - t0
    size = _dir_bytes(Path(local))
    mbps = (size / 1e6 / elapsed) if elapsed > 0 else 0.0
    return {
        "id": repo_id,
        "ok": True,
        "path": local,
        "bytes": size,
        "seconds": round(elapsed, 1),
        "MB_per_s": round(mbps, 1),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--manifest", default="models_small_l4/all.yaml", help="models YAML (reads each `id`)"
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("PREWARM_WORKERS", "8")),
        help="how many models to download concurrently",
    )
    ap.add_argument("--summary", default=None, help="write a JSON run summary to this path")
    args = ap.parse_args(argv)

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None
    ids = read_manifest_ids(args.manifest)

    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    hf_transfer = os.environ.get("HF_HUB_ENABLE_HF_TRANSFER")
    print(f"== prewarm {len(ids)} models from {args.manifest} ==")
    print(f"   HF_HOME={hf_home}  workers={args.workers}  hf_transfer={hf_transfer}")
    print(f"   token={'set' if token else 'UNSET (ungated repos only)'}")
    sys.stdout.flush()

    t0 = time.time()
    results: list[dict] = []
    with cf.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futs = {pool.submit(prewarm_one, mid, token): mid for mid in ids}
        for fut in cf.as_completed(futs):
            r = fut.result()
            results.append(r)
            if r["ok"]:
                print(
                    f"[ok]   {r['id']:<45} {r['bytes'] / 1e6:8.1f} MB  "
                    f"{r['seconds']:6.1f}s  {r['MB_per_s']:6.1f} MB/s"
                )
            else:
                print(f"[FAIL] {r['id']:<45} {r['error']}")
            sys.stdout.flush()

    wall = time.time() - t0
    ok = [r for r in results if r["ok"]]
    bad = [r for r in results if not r["ok"]]
    total_bytes = sum(r["bytes"] for r in ok)
    print("== summary ==")
    print(
        f"   ok={len(ok)}/{len(ids)}  failed={len(bad)}  "
        f"total={total_bytes / 1e9:.2f} GB  wall={wall / 60:.1f} min"
    )
    if wall > 0:
        print(f"   aggregate download throughput: {total_bytes/1e6/wall:.1f} MB/s")
    if bad:
        print("   FAILED: " + ", ".join(r["id"] for r in bad))

    if args.summary:
        Path(args.summary).write_text(
            json.dumps(
                {
                    "manifest": args.manifest,
                    "hf_home": hf_home,
                    "workers": args.workers,
                    "wall_seconds": round(wall, 1),
                    "total_bytes": total_bytes,
                    "results": sorted(results, key=lambda r: r["id"]),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"   wrote summary -> {args.summary}")

    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
