"""Durable, resumable-friendly result sink for a uni_frq run.

The shared runner writes exactly one object, once, after the CAT session returns. A crash
mid-session, or a transient S3 fault after a long GPU run, therefore discards everything.
This sink instead:

- writes every artifact to a local run directory first, so results survive an S3 outage;
- mirrors to S3 with adaptive boto3 retries and a ``head_object`` size check, so a
  "success" log means the object is really readable;
- namespaces by checkpoint and run id, so two checkpoints cannot overwrite or falsely
  "skip" each other. The slug carries a digest of the full checkpoint id, because
  truncating a long path to its tail collides for checkpoints that share a suffix;
- writes ``_SUCCESS`` last, and remotely *before* locally, so the local marker never
  claims a completeness the remote copy does not have.

``boto3`` is imported lazily; a purely local run needs no AWS dependency at all.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

from ...common.s3_io import is_s3_uri, parse_s3_uri

log = logging.getLogger("uni_frq.sink")

SUCCESS_MARKER = "_SUCCESS"


def slugify(value: str, *, fallback: str = "checkpoint") -> str:
    """Return a collision-resistant, traversal-safe path segment for an id.

    A readable head plus a digest of the *full* input. The digest matters: checkpoint
    URIs often differ only in a long prefix, and a plain tail-truncation maps them to the
    same directory, which silently overwrites one run's results with another's.
    """
    raw = (value or "").strip()
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", raw.strip("/ ")).strip("-.")
    head = cleaned[-48:].strip("-.") or fallback
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]
    return f"{head}-{digest}"


def _safe_name(name: str) -> str:
    """Reject anything that is not a bare filename (no separators, no ``..``)."""
    if not name or name in {".", ".."} or name != Path(name).name:
        raise ValueError(f"artifact name must be a bare filename, got {name!r}")
    return name


def _normalize_uri(uri: str) -> str:
    """Collapse duplicate slashes in the path part so S3 keys never contain ``//``."""
    scheme, sep, rest = uri.partition("://")
    if not sep:
        return uri
    return f"{scheme}://{re.sub(r'/+', '/', rest).rstrip('/')}"


class ResultSink:
    """Collects run artifacts locally and mirrors them to a namespaced destination."""

    def __init__(
        self,
        out_uri: str,
        *,
        run_id: str,
        checkpoint_slug: str,
        local_dir: Path,
        region: str = "us-east-1",
        endpoint_url: str | None = None,
        namespace: bool = False,
    ) -> None:
        self.run_id = slugify(run_id, fallback="run")
        self.region = region
        self.endpoint_url = endpoint_url
        self._remote_base: str | None = None
        self._verify = True
        # Flat by default, because the platform already hands each job its own
        # per-run prefix and the MCQ flow writes cat_report.json at its root; nesting
        # there would hide the report from whatever collects it. Namespacing is for
        # self-managed destinations where several runs share one prefix.
        relative = f"{slugify(checkpoint_slug)}/{self.run_id}" if namespace else ""

        if is_s3_uri(out_uri):
            base = _normalize_uri(out_uri)
            self._remote_base = f"{base}/{relative}" if relative else base
            self.local_dir = local_dir / relative if relative else local_dir
        else:
            root = Path(out_uri).expanduser()
            self.local_dir = root / relative if relative else root
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self._pending: set[str] = set()

    @property
    def base_uri(self) -> str:
        """Where the caller should look for results."""
        return self._remote_base or str(self.local_dir)

    # ---- local writes -------------------------------------------------------------
    def append(self, name: str, record: dict[str, Any]) -> None:
        """Append one JSON record to a local NDJSON artifact."""
        with (self.local_dir / _safe_name(name)).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._pending.add(name)

    def write_json(self, name: str, payload: dict[str, Any]) -> None:
        """Write (or overwrite) a local JSON artifact."""
        path = self.local_dir / _safe_name(name)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._pending.add(name)

    # ---- remote mirror ------------------------------------------------------------
    def _client(self) -> Any:
        import boto3
        from botocore.config import Config

        return boto3.client(
            "s3",
            region_name=self.region,
            endpoint_url=self.endpoint_url,
            config=Config(retries={"max_attempts": 5, "mode": "adaptive"}),
        )

    def _upload(self, client: Any, bucket: str, prefix: str, name: str) -> None:
        """Upload one artifact and verify the stored size matches what we sent."""
        body = (self.local_dir / name).read_bytes()
        key = f"{prefix}/{name}" if prefix else name
        content_type = "application/json" if name.endswith(".json") else "text/plain"
        client.put_object(Bucket=bucket, Key=key, Body=body, ContentType=content_type)
        if not self._verify:
            return
        try:
            stored = client.head_object(Bucket=bucket, Key=key)["ContentLength"]
        except Exception as exc:  # noqa: BLE001
            # A write-only role (PutObject without GetObject) would otherwise abort the
            # whole run on the first artifact. Losing the check is far better than
            # losing the run, so warn once and stop verifying.
            self._verify = False
            log.warning("cannot verify uploads (%s); continuing without the size check", exc)
            return
        if stored != len(body):
            raise RuntimeError(f"s3 verify failed for {key}: sent {len(body)}, stored {stored}")

    def sync(self) -> None:
        """Mirror any locally-changed artifacts to S3. No-op for a local destination.

        On failure ``_pending`` is left intact so the next call retries; the local copy is
        authoritative either way.
        """
        if self._remote_base is None or not self._pending:
            self._pending.clear()
            return
        bucket, prefix = parse_s3_uri(self._remote_base)
        client = self._client()
        for name in sorted(self._pending):
            self._upload(client, bucket, prefix, name)
        log.info("synced %d artifact(s) to %s", len(self._pending), self._remote_base)
        self._pending.clear()

    def finalize(self, *, ok: bool) -> str:
        """Flush everything, then write the completion marker. Returns the base URI.

        The remote marker goes up before the local one, so a local ``_SUCCESS`` always
        implies the remote copy is complete too.
        """
        self.sync()
        if not ok:
            return self.base_uri
        marker = self.local_dir / SUCCESS_MARKER
        marker.write_text("", encoding="utf-8")
        if self._remote_base is not None:
            bucket, prefix = parse_s3_uri(self._remote_base)
            try:
                self._upload(self._client(), bucket, prefix, SUCCESS_MARKER)
            except Exception:
                marker.unlink(missing_ok=True)  # do not claim success the remote lacks
                raise
        return self.base_uri
