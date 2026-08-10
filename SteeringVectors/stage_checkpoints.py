#!/usr/bin/env python3
"""Stage OLMo-core checkpoints from edullm-checkpoints into the outputs bucket.

GPU workload roles can read ``teams/<team>/runs/*`` but not ``edullm-checkpoints``.
Run this as a CPU platform job before any steering-vector GPU smoke.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import steering_common as sc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [stage_checkpoints] %(levelname)s: %(message)s",
)
log = logging.getLogger("stage_checkpoints")


def run_stage(
    source: str,
    dest: str,
    *,
    aws_region: str = "us-east-1",
    s3_endpoint: str | None = None,
    manifest_out: str | None = None,
) -> dict[str, Any]:
    summary = sc.copy_s3_prefix(source, dest, aws_region, s3_endpoint)
    summary.update(
        {
            "success": True,
            "timestamp": datetime.now(UTC).isoformat(),
            "staged_uri": summary["dest"],
        }
    )

    if manifest_out:
        if sc.is_s3_uri(manifest_out):
            bucket, key = sc.parse_s3_uri(manifest_out)
            client = sc.s3_client(aws_region, s3_endpoint)
            client.put_object(
                Bucket=bucket,
                Key=key,
                Body=json.dumps(summary, indent=2).encode("utf-8"),
                ContentType="application/json",
            )
            summary["manifest_uri"] = manifest_out
        else:
            out = Path(manifest_out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(summary, indent=2))
            summary["manifest_path"] = str(out)

    log.info("Staging complete: %d objects -> %s", summary["objects_copied"], dest)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Copy checkpoint objects from edullm-checkpoints to teams/.../runs/."
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Source checkpoint prefix (e.g. s3://edullm-checkpoints/.../step1315/)",
    )
    parser.add_argument(
        "--dest",
        required=True,
        help="Destination prefix under outputs bucket (teams/.../runs/...)",
    )
    parser.add_argument(
        "--manifest-out",
        help="Write staging manifest locally or to s3:// (optional)",
    )
    parser.add_argument("--aws-region", default="us-east-1")
    parser.add_argument("--s3-endpoint-url")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    plan = {
        "source": args.source,
        "dest": args.dest,
        "manifest_out": args.manifest_out,
    }
    if args.dry_run:
        log.info("[dry-run] %s", json.dumps(plan, indent=2))
        return 0

    try:
        run_stage(
            args.source,
            args.dest,
            aws_region=args.aws_region,
            s3_endpoint=args.s3_endpoint_url,
            manifest_out=args.manifest_out,
        )
    except Exception as exc:  # noqa: BLE001
        log.error("staging failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
