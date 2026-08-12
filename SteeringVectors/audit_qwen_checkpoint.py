#!/usr/bin/env python3
"""Pre-flight audit for Qwen3-30B-A3B-Thinking steering jobs."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import steering_common as sc
from checkpoint_manifest import load_manifest, load_manifest_raw, plan_dict

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("audit_qwen")


def audit_manifest(path: Path, *, check_s3: bool = False) -> dict[str, Any]:
    raw = load_manifest_raw(path)
    findings: list[dict[str, Any]] = []
    ok = True

    for role in ("early", "chinchilla", "final"):
        uri = raw["checkpoints"][role]
        if "REPLACE_WITH_YOUR_STAGED_URI" in uri:
            findings.append(
                {"level": "error", "check": f"checkpoint.{role}", "detail": "placeholder URI"}
            )
            ok = False
        if not uri.startswith("s3://"):
            findings.append(
                {"level": "error", "check": f"checkpoint.{role}", "detail": "expected s3:// URI"}
            )
            ok = False

    family = raw.get("model_family")
    if family != "qwen3_moe_thinking":
        findings.append(
            {
                "level": "warn",
                "check": "model_family",
                "detail": f"expected qwen3_moe_thinking, got {family!r}",
            }
        )

    load = raw.get("load") or {}
    if load.get("device_map") != "auto":
        findings.append(
            {
                "level": "warn",
                "check": "load.device_map",
                "detail": "MoE 30B typically needs device_map=auto on gpu-8xl40s",
            }
        )
    if load.get("enable_thinking") is not False:
        findings.append(
            {
                "level": "warn",
                "check": "load.enable_thinking",
                "detail": "eval generate() should disable thinking; set false in manifest",
            }
        )

    layers = raw.get("steering_layers") or []
    if layers and any(layer >= 48 for layer in layers):
        findings.append(
            {
                "level": "error",
                "check": "steering_layers",
                "detail": "Qwen3-30B has 48 layers (0-47)",
            }
        )
        ok = False

    plan = load_manifest(path, dry_run=True)
    report = {"ok": ok, "manifest": str(path), "plan": plan_dict(plan), "findings": findings}

    if check_s3 and ok:
        uri = raw["checkpoints"]["final"].rstrip("/") + "/"
        if sc.is_s3_uri(uri):
            try:
                config = sc.fetch_run_config(uri)
                detected = sc.detect_model_family(config)
                report["final_config"] = {
                    "model_type": config.get("model_type"),
                    "architectures": config.get("architectures"),
                    "num_hidden_layers": config.get("num_hidden_layers"),
                    "detected_family": detected,
                }
                if detected not in ("qwen3_moe_thinking", "qwen3_moe", "qwen3"):
                    findings.append(
                        {
                            "level": "warn",
                            "check": "config.family",
                            "detail": f"detected {detected} from final config.json",
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                findings.append({"level": "error", "check": "s3.config", "detail": str(exc)})
                report["ok"] = False

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Qwen steering manifest before submit.")
    parser.add_argument(
        "--manifest",
        default=str(Path(__file__).resolve().parent / "checkpoints_qwen30b-thinking.json"),
    )
    parser.add_argument("--check-s3", action="store_true", help="Fetch final config.json from S3")
    args = parser.parse_args()

    report = audit_manifest(Path(args.manifest), check_s3=args.check_s3)
    log.info("%s", json.dumps(report, indent=2))
    for item in report["findings"]:
        log.log(
            logging.ERROR if item["level"] == "error" else logging.WARNING,
            "[%s] %s: %s",
            item["level"],
            item["check"],
            item["detail"],
        )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
