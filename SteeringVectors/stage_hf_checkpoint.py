#!/usr/bin/env python3
"""Download a HuggingFace model snapshot and upload it to S3 for GPU jobs.

GPU batch roles read ``teams/<team>/runs/*`` but cannot pull from the Hub at
job start for multi‑tens‑of‑GB checkpoints. Run this as a CPU platform job first.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import steering_common as sc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [stage_hf_checkpoint] %(levelname)s: %(message)s",
)
log = logging.getLogger("stage_hf_checkpoint")

DEFAULT_DEST = (
    "s3://sbsandbox-intern-edullm-outputs/teams/eval-inference/runs/qwen30b-thinking-staged/final/"
)


def run_stage(
    model_id: str,
    dest: str,
    *,
    revision: str | None = None,
    aws_region: str = "us-east-1",
    s3_endpoint: str | None = None,
    manifest_out: str | None = None,
) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    with tempfile.TemporaryDirectory(prefix="hf-stage-") as tmp:
        tmp_path = Path(tmp)
        log.info("Downloading %s from HuggingFace Hub to %s", model_id, tmp_path)
        local_dir = snapshot_download(
            repo_id=model_id,
            revision=revision,
            local_dir=str(tmp_path),
            local_dir_use_symlinks=False,
        )
        upload = sc.upload_local_directory(
            Path(local_dir), dest, region=aws_region, endpoint=s3_endpoint
        )

    summary: dict[str, Any] = {
        "model_id": model_id,
        "revision": revision,
        "dest": upload["dest"],
        "files_uploaded": upload["files_uploaded"],
        "bytes_uploaded": upload["bytes_uploaded"],
        "success": True,
        "timestamp": datetime.now(UTC).isoformat(),
    }

    if manifest_out:
        body = json.dumps(summary, indent=2).encode("utf-8")
        if sc.is_s3_uri(manifest_out):
            bucket, key = sc.parse_s3_uri(manifest_out)
            client = sc.s3_client(aws_region, s3_endpoint)
            client.put_object(
                Bucket=bucket,
                Key=key,
                Body=body,
                ContentType="application/json",
            )
            summary["manifest_uri"] = manifest_out
        else:
            out = Path(manifest_out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(body)
            summary["manifest_path"] = str(out)

    log.info("Staged %s -> %s (%d files)", model_id, dest, upload["files_uploaded"])
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download a HuggingFace model and upload to teams/.../runs/ on S3."
    )
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen3-30B-A3B-Thinking-2507",
        help="HuggingFace repo id",
    )
    parser.add_argument(
        "--dest",
        default=DEFAULT_DEST,
        help="Destination S3 prefix (must be under teams/.../runs/)",
    )
    parser.add_argument("--revision", help="Optional HF revision / commit sha")
    parser.add_argument("--manifest-out", help="Write staging manifest locally or to s3://")
    parser.add_argument("--aws-region", default="us-east-1")
    parser.add_argument("--s3-endpoint-url")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    plan = {
        "model_id": args.model_id,
        "dest": args.dest,
        "revision": args.revision,
        "manifest_out": args.manifest_out,
    }
    if args.dry_run:
        log.info("[dry-run] %s", json.dumps(plan, indent=2))
        return 0

    try:
        run_stage(
            args.model_id,
            args.dest,
            revision=args.revision,
            aws_region=args.aws_region,
            s3_endpoint=args.s3_endpoint_url,
            manifest_out=args.manifest_out,
        )
    except Exception as exc:  # noqa: BLE001
        log.error("HF staging failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
