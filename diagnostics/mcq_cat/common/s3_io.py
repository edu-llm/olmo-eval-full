"""S3 read/write helpers for MCQ CAT diagnostics.

These mirror the patterns in ``tests/OnNode/checkpoint_infer.py`` (``parse_s3_uri``,
a lazily constructed client, prefix download, and object upload) so the two flows
handle S3 the same way. Kept dependency-light: ``boto3`` is imported lazily so the
package imports without the optional ``s3`` extra installed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger("mcq_cat.s3_io")

S3_SCHEME = "s3://"


def is_s3_uri(uri: str) -> bool:
    """Return ``True`` if ``uri`` is an ``s3://`` URI."""
    return uri.startswith(S3_SCHEME)


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Split ``s3://bucket/key/prefix`` into ``(bucket, key_prefix)``."""
    if not is_s3_uri(uri):
        raise ValueError(f"Not an S3 URI: {uri}")
    without_scheme = uri[len(S3_SCHEME) :]
    bucket, _, key = without_scheme.partition("/")
    return bucket, key.rstrip("/")


def s3_client(region: str = "us-east-1", endpoint_url: str | None = None) -> Any:
    """Construct a boto3 S3 client. ``boto3`` is imported lazily."""
    import boto3

    kwargs: dict[str, Any] = {"region_name": region}
    if endpoint_url:
        kwargs["endpoint_url"] = endpoint_url
    return boto3.client("s3", **kwargs)


def download_prefix(
    uri: str,
    dest: Path,
    *,
    region: str = "us-east-1",
    endpoint_url: str | None = None,
) -> Path:
    """Download every object under an S3 prefix into ``dest`` and return it."""
    bucket, prefix = parse_s3_uri(uri)
    client = s3_client(region=region, endpoint_url=endpoint_url)
    dest.mkdir(parents=True, exist_ok=True)

    paginator = client.get_paginator("list_objects_v2")
    downloaded = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            relative = key[len(prefix) :].lstrip("/")
            local_path = dest / relative
            local_path.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, key, str(local_path))
            downloaded += 1

    if downloaded == 0:
        raise RuntimeError(f"No objects found under {uri}")
    log.info("Downloaded %d objects from %s to %s", downloaded, uri, dest)
    return dest


def read_text(uri: str, *, region: str = "us-east-1", endpoint_url: str | None = None) -> str:
    """Read a single object's body as UTF-8 text from an ``s3://`` URI or a path.

    UTF-8 on both branches, and named on the local one rather than left to the platform.
    ``read_text()`` without an encoding uses the locale's, which is cp1252 on Windows,
    and the artifacts this reads -- item stems, IRT parameters -- are written UTF-8. A
    mis-decode there is not a crash on most bytes: it substitutes a different stem, and
    the run scores a question the bank holds no difficulty for.
    """
    if not is_s3_uri(uri):
        return Path(uri).read_text(encoding="utf-8")
    bucket, key = parse_s3_uri(uri)
    client = s3_client(region=region, endpoint_url=endpoint_url)
    obj = client.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read().decode("utf-8")


def upload_files(
    base_uri: str,
    files: dict[str, str],
    *,
    region: str = "us-east-1",
    endpoint_url: str | None = None,
) -> str:
    """Upload ``{name: content}`` under ``base_uri`` and return the base URI."""
    bucket, prefix = parse_s3_uri(base_uri)
    client = s3_client(region=region, endpoint_url=endpoint_url)
    for name, content in files.items():
        key = f"{prefix}/{name}" if prefix else name
        content_type = "application/json" if name.endswith(".json") else "application/x-ndjson"
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=content.encode("utf-8"),
            ContentType=content_type,
        )
    log.info("Uploaded %d files to %s", len(files), base_uri)
    return base_uri


def upload_json(
    base_uri: str,
    name: str,
    payload: dict[str, Any],
    *,
    region: str = "us-east-1",
    endpoint_url: str | None = None,
) -> str:
    """Upload ``payload`` as a JSON object named ``name`` under ``base_uri``."""
    return upload_files(
        base_uri,
        {name: json.dumps(payload, indent=2)},
        region=region,
        endpoint_url=endpoint_url,
    )


def resolve_checkpoint(
    checkpoint: str,
    dest: Path,
    *,
    region: str = "us-east-1",
    endpoint_url: str | None = None,
) -> Path:
    """Return a local checkpoint directory, downloading it first if it is on S3."""
    if is_s3_uri(checkpoint):
        return download_prefix(checkpoint, dest, region=region, endpoint_url=endpoint_url)
    local = Path(checkpoint)
    if not local.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint}")
    return local
