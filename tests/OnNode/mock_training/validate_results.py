#!/usr/bin/env python3
"""Validate the S3 outputs produced by the mock training run.

Lists everything the run wrote under the results prefix, then checks that every
step produced its own ``results.jsonl`` + ``manifest.json``, that each manifest
reports success with the expected prompt count, and that generations differ
across steps (proof that inference actually ran on each distinct checkpoint).

Configuration is read from the same environment as ``checkpoint_infer.py``.
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any


def _s3_client():
    import boto3

    kwargs: dict[str, Any] = {"region_name": os.environ.get("AWS_REGION", "us-east-1")}
    endpoint = os.environ.get("S3_ENDPOINT_URL")
    if endpoint:
        kwargs["endpoint_url"] = endpoint
    return boto3.client("s3", **kwargs)


def _list_keys(client, bucket: str, prefix: str) -> list[str]:
    keys: list[str] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        keys.extend(obj["Key"] for obj in page.get("Contents", []))
    return sorted(keys)


def _get(client, bucket: str, key: str) -> str:
    return client.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")


def main() -> int:
    bucket = os.environ["RESULTS_BUCKET"]
    run = os.environ["RUN_NAME"]
    prefix = os.environ.get("RESULTS_PREFIX", "checkpoint-infer")
    expected_prompts = int(os.environ.get("NUM_PROMPTS", "8"))
    run_prefix = f"{prefix}/{run}/"

    client = _s3_client()

    ckpt_keys = _list_keys(client, bucket, "checkpoints/")
    result_keys = _list_keys(client, bucket, run_prefix)

    print("=" * 72)
    print(f"Checkpoint objects under s3://{bucket}/checkpoints/ : {len(ckpt_keys)}")
    print(f"Result objects under s3://{bucket}/{run_prefix} :")
    for key in result_keys:
        print(f"  s3://{bucket}/{key}")
    print("=" * 72)

    steps = sorted({int(m.group(1)) for k in result_keys if (m := re.search(r"/step(\d+)/", k))})
    if not steps:
        print("FAIL: no step results found", file=sys.stderr)
        return 1

    ok = True
    for step in steps:
        base = f"{prefix}/{run}/step{step}"
        step_keys = {k.split("/")[-1] for k in result_keys if f"/step{step}/" in k}
        for required in ("results.jsonl", "manifest.json"):
            if required not in step_keys:
                print(f"FAIL: step {step} missing {required}", file=sys.stderr)
                ok = False

        manifest = json.loads(_get(client, bucket, f"{base}/manifest.json"))
        if manifest.get("success") is not True:
            print(f"FAIL: step {step} manifest.success != true", file=sys.stderr)
            ok = False
        if manifest.get("num_prompts") != expected_prompts:
            print(
                f"FAIL: step {step} num_prompts={manifest.get('num_prompts')} "
                f"(expected {expected_prompts})",
                file=sys.stderr,
            )
            ok = False

        rows = [
            json.loads(line)
            for line in _get(client, bucket, f"{base}/results.jsonl").splitlines()
            if line
        ]
        if len(rows) != expected_prompts:
            print(
                f"FAIL: step {step} has {len(rows)} rows (expected {expected_prompts})",
                file=sys.stderr,
            )
            ok = False
        print(
            f"step {step}: manifest success={manifest.get('success')} "
            f"num_prompts={manifest.get('num_prompts')} rows={len(rows)}"
        )

    distinct_outputs = {}
    for step in steps:
        base = f"{prefix}/{run}/step{step}"
        rows = [
            json.loads(line)
            for line in _get(client, bucket, f"{base}/results.jsonl").splitlines()
            if line
        ]
        distinct_outputs[step] = tuple(r["output"] for r in rows)

    sample_steps = [steps[0], steps[-1]] if len(steps) > 1 else steps
    for step in sample_steps:
        base = f"{prefix}/{run}/step{step}"
        print("-" * 72)
        print(f"sample manifest.json (step {step}):")
        print(_get(client, bucket, f"{base}/manifest.json"))
        print(f"sample results.jsonl rows (step {step}, first 2):")
        rows = [line for line in _get(client, bucket, f"{base}/results.jsonl").splitlines() if line]
        for line in rows[:2]:
            print(line)
    print("=" * 72)

    unique = len(set(distinct_outputs.values()))
    if unique != len(steps):
        print(
            f"FAIL: expected {len(steps)} distinct output sets, found {unique} "
            "(inference did not vary across checkpoints)",
            file=sys.stderr,
        )
        ok = False
    else:
        print(f"OK: all {len(steps)} steps produced distinct output sets")

    print("=" * 72)
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
