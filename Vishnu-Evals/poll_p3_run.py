"""Watch a sharded p3 evaluation sweep from the progress files it writes to S3.

One line per shard, which is the density you want when eight processes are
running and the question is whether all eight are alive and agreeing with each
other. Reads only the small marker objects, never the partial results, so
polling every few seconds costs nothing.

The three columns that matter early, because the sweep skipped its smoke test:
``nll`` should sit far below 11.93 (uniform over Qwen's vocabulary) or the
evaluated model is corrupt; ``gen`` and ``cap%`` answer whether generations
terminate on EOS or run to the token ceiling, which is the difference between a
half-hour sweep and a four-hour one; ``age`` going stale is the hang signal.

    python poll_p3_run.py --s3-out s3://bucket/teams/eval/vishnu-p3 --watch 60
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone

SHARD_KEY = re.compile(r"(?P<arm>[^/]+)/shard(?P<shard>\d+)/(?P<marker>_IN_PROGRESS|_READY|_FAILED)$")
TERMINAL = ("_FAILED", "_READY")
UNIFORM_NLL = 11.9312


def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise SystemExit(f"--s3-out must start with s3://, got {uri!r}")
    bucket, _, prefix = uri[len("s3://") :].partition("/")
    if not bucket:
        raise SystemExit(f"--s3-out has no bucket: {uri!r}")
    return bucket, prefix.strip("/")


def _age(timestamp: str | None) -> str:
    if not timestamp:
        return "?"
    try:
        then = datetime.fromisoformat(timestamp)
    except ValueError:
        return "?"
    seconds = int((datetime.now(timezone.utc) - then).total_seconds())
    if seconds < 90:
        return f"{seconds}s"
    if seconds < 5400:
        return f"{seconds // 60}m"
    return f"{seconds / 3600:.1f}h"


def _number(value, spec: str, width: int) -> str:
    return format(value, spec).rjust(width) if isinstance(value, (int, float)) else "-".rjust(width)


def collect(client, bucket: str, prefix: str) -> list[dict]:
    """Newest marker per shard, preferring a terminal one over _IN_PROGRESS."""
    markers: dict[tuple[str, int], set[str]] = {}
    paginator = client.get_paginator("list_objects_v2")
    scope = f"{prefix}/" if prefix else ""
    for page in paginator.paginate(Bucket=bucket, Prefix=scope):
        for entry in page.get("Contents", []):
            match = SHARD_KEY.search(entry["Key"])
            if match:
                shard = (match["arm"], int(match["shard"]))
                markers.setdefault(shard, set()).add(match["marker"])

    rows = []
    for (arm, shard), names in sorted(markers.items()):
        marker = next((name for name in TERMINAL if name in names), "_IN_PROGRESS")
        key = "/".join(filter(None, (prefix, arm, f"shard{shard}", marker)))
        try:
            body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
            state = json.loads(body)
        except Exception as error:  # noqa: BLE001 - a poller must never crash the watch
            rows.append({"arm": arm, "shard": shard, "marker": "ERROR", "error": str(error)})
            continue
        latest = state.get("latest") or (state.get("cell_summaries") or [{}])[-1]
        rows.append(
            {
                "arm": arm,
                "shard": shard,
                "marker": marker,
                "done": state.get("cells_completed"),
                "total": state.get("cells_total"),
                "rows": state.get("rows_scored"),
                "updated": state.get("updated_at") or state.get("finished_at"),
                "nll": latest.get("target_token_micro_nll_per_token"),
                "gen": latest.get("generated_tokens_mean"),
                "cap": latest.get("generation_cap_hit_fraction"),
                "cell": f"{latest.get('family', '?')}/{latest.get('condition', '?')}",
                "canaries": state.get("nll_canary_tripped_cells") or [],
            }
        )
    return rows


def render(rows: list[dict]) -> str:
    if not rows:
        return "no shards have reported yet"
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = [
        f"[{stamp}]  {len(rows)} shard(s)",
        f"{'arm':<6} {'sh':>2} {'state':<12} {'cells':>7} {'rows':>6} "
        f"{'age':>5} {'nll':>7} {'gen':>6} {'cap%':>5}  cell",
    ]
    for row in rows:
        if row["marker"] == "ERROR":
            lines.append(f"{row['arm']:<6} {row['shard']:>2} unreadable   {row['error'][:60]}")
            continue
        state = {"_READY": "done", "_FAILED": "FAILED", "_IN_PROGRESS": "running"}[row["marker"]]
        cells = (
            f"{row['done']}/{row['total']}"
            if row["done"] is not None and row["total"] is not None
            else "-"
        )
        flag = "  <-- NLL CANARY" if row["canaries"] else ""
        lines.append(
            f"{row['arm']:<6} {row['shard']:>2} {state:<12} {cells:>7} "
            f"{_number(row['rows'], ',', 6)} {_age(row['updated']):>5} "
            f"{_number(row['nll'], '.3f', 7)} {_number(row['gen'], '.0f', 6)} "
            f"{_number(None if row['cap'] is None else row['cap'] * 100, '.0f', 5)}  "
            f"{row['cell']}{flag}"
        )

    running = [row for row in rows if row.get("marker") == "_IN_PROGRESS"]
    failed = [row for row in rows if row.get("marker") == "_FAILED"]
    tripped = sorted({row["arm"] for row in rows if row.get("canaries")})
    if failed:
        lines.append(f"\n{len(failed)} shard(s) FAILED: " + ", ".join(
            f"{row['arm']}/shard{row['shard']}" for row in failed
        ))
    if tripped:
        lines.append(
            f"\nNLL CANARY TRIPPED on {', '.join(tripped)}. Micro NLL is near chance "
            f"({UNIFORM_NLL:.2f} nats); the evaluated model may be corrupted."
        )
    stale = [row for row in running if _age(row["updated"]).endswith(("m", "h"))]
    if stale:
        lines.append(
            f"\n{len(stale)} running shard(s) have not synced in over 90s; "
            "check for a hang if this persists past a single cell."
        )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--s3-out", required=True, help="the prefix passed to run_eval.py --s3-out")
    ap.add_argument("--watch", type=int, default=0, help="repoll every N seconds; 0 polls once")
    args = ap.parse_args()

    try:
        import boto3
    except ImportError:
        raise SystemExit("poll_p3_run.py needs boto3: pip install boto3") from None

    bucket, prefix = parse_s3_uri(args.s3_out)
    client = boto3.client("s3")
    while True:
        try:
            print(render(collect(client, bucket, prefix)), flush=True)
        except Exception as error:  # noqa: BLE001 - keep watching through transient failures
            print(f"poll failed, retrying: {error}", flush=True)
        if args.watch <= 0:
            return
        try:
            time.sleep(args.watch)
        except KeyboardInterrupt:
            return
        print(file=sys.stderr)


if __name__ == "__main__":
    main()
