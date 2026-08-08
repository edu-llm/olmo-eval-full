"""Submit the sweep as an AWS Batch array job (one array index per model shard).

Usage:
  python aws/submit_batch.py --queue my-gpu-queue --job-def adaptive-inference \\
      --num-shards 100 --benchmarks all

Requires boto3 + AWS credentials. This only *submits*; the container entrypoint
(entrypoint.sh) does the work and S3 sync.
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queue", required=True, help="Batch job queue name/ARN")
    ap.add_argument("--job-def", required=True, help="Batch job definition name/ARN")
    ap.add_argument("--num-shards", type=int, default=100, help="array size = #model shards")
    ap.add_argument("--benchmarks", default="all")
    ap.add_argument("--backend", default="vllm")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--name", default="adaptive-inference-sweep")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    env = [
        {"name": "NUM_SHARDS", "value": str(args.num_shards)},
        {"name": "BENCHMARKS", "value": args.benchmarks},
        {"name": "BACKEND", "value": args.backend},
    ]
    if args.max_samples is not None:
        env.append({"name": "MAX_SAMPLES", "value": str(args.max_samples)})

    request = {
        "jobName": args.name,
        "jobQueue": args.queue,
        "jobDefinition": args.job_def,
        "arrayProperties": {"size": args.num_shards},
        "containerOverrides": {"environment": env},
    }

    if args.dry_run:
        import json

        print(json.dumps(request, indent=2))
        return 0

    import boto3

    client = boto3.client("batch")
    resp = client.submit_job(**request)
    print(f"submitted {resp['jobName']} -> {resp['jobId']} (array size {args.num_shards})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
