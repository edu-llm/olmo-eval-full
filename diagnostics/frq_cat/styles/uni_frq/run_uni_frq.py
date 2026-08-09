"""FRQ CAT pipeline for uni_frq: tutor answers, judge grades each criterion, results to S3.

    python -m diagnostics.frq_cat.styles.uni_frq.run_uni_frq \
        --checkpoint s3://bucket/ckpts/step_1000 \
        --tutor-endpoint http://TUTOR:8000/v1 --served-model tutor \
        --judge-config diagnostics/frq_cat/styles/uni_frq/judge_frontier.yaml \
        --judge-endpoint https://api.openai.com/v1 \
        --out s3://bucket/frq_cat

This is a style-local hardening of the shared ``frq_cat.runner`` and keeps every shared
file untouched. It reuses the shared bank loaders, IRT schema, judge prompt and this
style's IRT model, and differs in the ways that matter for an unattended AWS run:
results are namespaced per checkpoint and run, written incrementally and verified, the
judge can authenticate to a hosted frontier model, one bad scenario no longer ends the
session, and an unexpected crash still flushes and reports where the partial results are.
The operating point comes from ``config.yaml`` unless overridden.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve()
for _up in _HERE.parents:  # make `diagnostics.` importable from any CWD
    if (_up / "diagnostics" / "frq_cat").is_dir():
        if str(_up) not in sys.path:
            sys.path.insert(0, str(_up))
        break

from diagnostics.frq_cat.common import bank_loader  # noqa: E402
from diagnostics.frq_cat.common.judge import load_spec  # noqa: E402
from diagnostics.frq_cat.common.s3_io import is_s3_uri  # noqa: E402
from diagnostics.frq_cat.styles.uni_frq.judge_client import ResilientJudge, redact  # noqa: E402
from diagnostics.frq_cat.styles.uni_frq.respgen_client import (  # noqa: E402
    ResilientRespGen,
    RespGenConfig,
)
from diagnostics.frq_cat.styles.uni_frq.result_sink import ResultSink, slugify  # noqa: E402
from diagnostics.frq_cat.styles.uni_frq.session import run_session  # noqa: E402
from diagnostics.frq_cat.styles.uni_frq.style import UniFrqStyle  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [uni_frq] %(levelname)s: %(message)s")
log = logging.getLogger("uni_frq.run")

_STYLE_DIR = _HERE.parent
_SECRETISH = re.compile(r"key|secret|token|password", re.IGNORECASE)


def _num(value: Any, default: float) -> float:
    """Coerce a config value, falling back when it is null or not a number."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _redact_map(data: dict[str, Any]) -> dict[str, Any]:
    """Redact anything secret-shaped before provenance is persisted."""
    out: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, str):
            named_secret = _SECRETISH.search(key) and not key.lower().endswith("_env")
            out[key] = "[redacted]" if named_secret else redact(value)
        else:
            out[key] = value
    return out


def build_parser(defaults: dict) -> argparse.ArgumentParser:
    """Build the CLI, defaulting the operating point to the style's config.yaml."""
    judge_default = defaults.get("judge_config") or "judge_frozen.yaml"
    p = argparse.ArgumentParser(prog="run_uni_frq", description=__doc__)
    p.add_argument("--checkpoint", required=True, help="Checkpoint id/URI (provenance).")
    p.add_argument("--checkpoint-kind", default="hf", choices=["hf", "olmo_core"])
    p.add_argument("--tutor-endpoint", help="Served /v1 base URL of the checkpoint under test.")
    p.add_argument("--served-model", default="tutor")
    p.add_argument("--tutor-max-tokens", type=int, default=1024)
    p.add_argument(
        "--judge-config",
        default=str(_STYLE_DIR / str(judge_default)),
        help="Judge selection YAML (judge_frozen.yaml or judge_frontier.yaml).",
    )
    p.add_argument("--judge-endpoint", help="Overrides `metadata.endpoint` in the judge config.")
    p.add_argument(
        "--judge-api-key-env",
        help="Env var NAME holding the judge API key; overrides `metadata.api_key_env`.",
    )
    p.add_argument("--out", required=True, help="Destination s3:// URI or local directory.")
    p.add_argument("--run-id", default="")
    p.add_argument("--se-threshold", type=float, default=_num(defaults.get("se_threshold"), 0.3))
    p.add_argument("--max-items", type=int, default=int(_num(defaults.get("max_items"), 40)))
    p.add_argument("--aws-region", default=os.environ.get("AWS_REGION", "us-east-1"))
    p.add_argument("--s3-endpoint-url", default=None)
    p.add_argument("--local-mirror", default="", help="Local staging dir for S3 destinations.")
    p.add_argument(
        "--namespace-run",
        action="store_true",
        help="Nest results under <checkpoint>/<run_id>/. Off by default so results land at "
        "--out itself, which is what the platform's per-run prefix expects.",
    )
    p.add_argument("--dry-run", action="store_true", help="Resolve and print the plan; no calls.")
    return p


def _resolve_judge_config(raw: str) -> Path:
    """Accept an absolute path, a CWD-relative path, or a name inside the style dir."""
    candidate = Path(raw).expanduser()
    if candidate.exists():
        return candidate
    fallback = _STYLE_DIR / candidate.name
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"judge config not found: {raw}")


def main(argv: list[str] | None = None) -> int:
    """Run the pipeline; returns 0 when at least one criterion was scored."""
    style = UniFrqStyle()
    args = build_parser(style.config).parse_args(argv)

    run_id = (
        slugify(args.run_id, fallback="run")
        if args.run_id
        else (f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}")
    )
    spec = load_spec(_resolve_judge_config(args.judge_config))
    metadata = dict(spec.metadata or {})
    judge_endpoint = args.judge_endpoint or str(metadata.get("endpoint", "") or "")
    api_key_env = args.judge_api_key_env or str(metadata.get("api_key_env", "") or "")

    if args.dry_run:
        log.info(
            "[dry-run] checkpoint=%s tutor=%s judge=%s(%s) out=%s run=%s se<%.3f max_items=%d",
            args.checkpoint,
            args.tutor_endpoint,
            spec.model_id,
            judge_endpoint or "<unset>",
            args.out,
            run_id,
            args.se_threshold,
            args.max_items,
        )
        if not args.tutor_endpoint or not judge_endpoint:
            log.error("[dry-run] missing tutor and/or judge endpoint; a real run would fail")
            return 2
        return 0

    if not args.tutor_endpoint or not judge_endpoint:
        log.error("need --tutor-endpoint and a judge endpoint (--judge-endpoint or config)")
        return 2

    judge = ResilientJudge(spec, endpoint=judge_endpoint, api_key_env=api_key_env)
    respgen = ResilientRespGen(
        RespGenConfig(
            endpoint=args.tutor_endpoint,
            served_model=args.served_model,
            max_tokens=args.tutor_max_tokens,
        )
    )
    if args.local_mirror:
        local_mirror = Path(args.local_mirror)
    elif is_s3_uri(args.out):
        local_mirror = Path(tempfile.mkdtemp(prefix="uni_frq-"))
    else:
        local_mirror = Path(args.out)  # local destination stages in place
    sink = ResultSink(
        args.out,
        run_id=run_id,
        checkpoint_slug=args.checkpoint,  # ResultSink slugifies; do not pre-slug it
        local_dir=local_mirror,
        region=args.aws_region,
        endpoint_url=args.s3_endpoint_url,
        namespace=args.namespace_run,
    )
    # Always say where the durable local copy is, so a failed upload is recoverable.
    log.info("results: %s | local copy: %s", sink.base_uri, sink.local_dir)

    bank = style.download_bank("")
    irt_bank = style.load_irt_params("")
    provenance = judge.provenance
    provenance["metadata"] = _redact_map(provenance.get("metadata", {}))

    manifest = {
        "run_id": run_id,
        "started_at": datetime.now(UTC).isoformat(),
        "cat_style": style.name,
        "style_config": style.config,
        "checkpoint": args.checkpoint,
        "checkpoint_kind": args.checkpoint_kind,
        "tutor": {
            "endpoint": args.tutor_endpoint,
            "served_model": args.served_model,
            "max_tokens": args.tutor_max_tokens,
            "temperature": 0.0,
            "seed": 0,
        },
        "judge": provenance,
        "bank": {
            "scenarios": len(bank.scenarios),
            "criteria": len(bank.criteria),
            "params_sha256": bank_loader.bank_sha256(_STYLE_DIR / "bank" / "params.jsonl"),
            "scenarios_sha256": bank_loader.bank_sha256(_STYLE_DIR / "bank" / "scenarios.jsonl"),
        },
        "stopping": {"se_threshold": args.se_threshold, "max_items": args.max_items},
        "destination": sink.base_uri,
        "local_dir": str(sink.local_dir),
    }
    # Provenance is written before any model call, so a crash still leaves a record.
    sink.write_json("manifest.json", manifest)
    sink.sync()

    try:
        summary = run_session(
            style,
            bank=bank,
            irt_bank=irt_bank,
            respgen=respgen,
            judge=judge,
            sink=sink,
            se_threshold=args.se_threshold,
            max_items=args.max_items,
        )
    except BaseException as exc:  # noqa: BLE001 - flush before anything propagates
        log.error("run failed (%s: %s); flushing partial results", type(exc).__name__, exc)
        manifest["failed_at"] = datetime.now(UTC).isoformat()
        manifest["error"] = f"{type(exc).__name__}: {exc}"[:500]
        sink.write_json("manifest.json", manifest)
        sink.finalize(ok=False)  # no _SUCCESS: the run is incomplete
        log.error("partial results kept at %s (local: %s)", sink.base_uri, sink.local_dir)
        raise

    report = summary.pop("report")
    report_dict = report.to_dict()
    report_dict["run"] = {
        "run_id": run_id,
        "checkpoint": args.checkpoint,
        "checkpoint_kind": args.checkpoint_kind,
        "judge": provenance,
        "finished_at": datetime.now(UTC).isoformat(),
        **summary,
    }
    sink.write_json("cat_report.json", report_dict)
    manifest["finished_at"] = report_dict["run"]["finished_at"]
    manifest["outcome"] = {k: v for k, v in summary.items() if k != "timing"}
    sink.write_json("manifest.json", manifest)

    scored = report.num_items_administered > 0
    base = sink.finalize(ok=scored)
    log.info(
        "%s | scored=%d skipped_scenarios=%d abstained=%d -> %s (local: %s)",
        "DONE" if scored else "NO SCORED CRITERIA",
        report.num_items_administered,
        summary["scenarios_skipped"],
        summary["criteria_abstained"],
        base,
        sink.local_dir,
    )
    print(
        json.dumps(
            {"base_uri": base, "local_dir": str(sink.local_dir), "run_id": run_id, **summary},
            indent=2,
        )
    )
    return 0 if scored else 1


if __name__ == "__main__":
    raise SystemExit(main())
