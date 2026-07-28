"""Optional in-process S3 upload (boto3), used to ship each finished shard off
the box. No-op when --s3-uri is unset.

Credentials come from the instance's IAM role — never from code or the repo.
This keeps the security constraint intact: the HF token lives in .env (gitignored)
and is never pushed; AWS access is the box's role, not a checked-in key.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """'s3://bucket/prefix' -> ('bucket', 'prefix'). Raises on a non-s3 uri."""
    p = urlparse(uri)
    if p.scheme != "s3" or not p.netloc:
        raise ValueError(f"not an s3:// uri: {uri!r}")
    return p.netloc, p.path.lstrip("/")


def maybe_upload(
    local_path: str | Path, s3_uri: str | None, rel_path: str | Path | None = None
) -> str | None:
    """Upload local_path under s3_uri's prefix. Returns the destination s3 uri,
    or None when s3_uri is falsy (upload disabled).

    `rel_path` sets the key suffix so the S3 layout can mirror a nested local
    tree — e.g. ``IFEval/Qwen.jsonl`` — instead of a bare filename. Without it,
    per-benchmark shards named ``<model>.jsonl`` would collide under one prefix
    (TutorBench and IFEval both writing ``Qwen.jsonl``). Defaults to the bare
    filename (the historical flat layout)."""
    if not s3_uri:
        return None
    import boto3  # lazy: only needed on the box, part of the [gen] extra

    bucket, prefix = parse_s3_uri(s3_uri)
    name = str(rel_path) if rel_path else Path(local_path).name
    name = name.replace("\\", "/")  # S3 keys use forward slashes on any OS
    key = f"{prefix.rstrip('/')}/{name}" if prefix else name
    boto3.client("s3").upload_file(str(local_path), bucket, key)
    return f"s3://{bucket}/{key}"
