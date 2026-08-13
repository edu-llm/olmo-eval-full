#!/usr/bin/env python3
"""Merge a staged PEFT LoRA adapter into its base model and upload full HF weights."""

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
    format="%(asctime)s [merge_peft_to_hf] %(levelname)s: %(message)s",
)
log = logging.getLogger("merge_peft_to_hf")

DEFAULT_ADAPTER = (
    "s3://sbsandbox-intern-edullm-outputs/teams/eval-inference/runs/"
    "qwen30b-thinking-staged/final/"
)
DEFAULT_DEST = (
    "s3://sbsandbox-intern-edullm-outputs/teams/eval-inference/runs/"
    "qwen30b-thinking-staged/merged/final/"
)


def merge_adapter(
    adapter_uri: str,
    dest_uri: str,
    *,
    dtype: str = "bfloat16",
    device_map: str = "auto",
    attn_implementation: str | None = "sdpa",
    aws_region: str = "us-east-1",
    s3_endpoint: str | None = None,
    manifest_out: str | None = None,
) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    with tempfile.TemporaryDirectory(prefix="merge-peft-") as tmp:
        tmp_path = Path(tmp)
        adapter_dir = sc.materialize_checkpoint(
            adapter_uri, tmp_path / "adapter", aws_region, s3_endpoint
        )
        if not sc.is_peft_adapter(adapter_dir):
            raise SystemExit(f"Not a PEFT adapter directory: {adapter_uri}")

        adapter_cfg = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
        base_id = adapter_cfg.get("base_model_name_or_path")
        if not base_id:
            raise SystemExit("adapter_config.json missing base_model_name_or_path")

        out_dir = tmp_path / "merged"
        load_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "device_map": device_map,
            "torch_dtype": getattr(torch, dtype),
        }
        if attn_implementation:
            load_kwargs["attn_implementation"] = attn_implementation

        log.info("Loading base %s and merging adapter from %s", base_id, adapter_uri)
        tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(base_id, **load_kwargs)
        model = PeftModel.from_pretrained(base, str(adapter_dir))
        merged = model.merge_and_unload()
        merged.save_pretrained(str(out_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(out_dir))

        upload = sc.upload_local_directory(out_dir, dest_uri, aws_region, s3_endpoint)

    summary: dict[str, Any] = {
        "adapter_uri": adapter_uri,
        "base_model": base_id,
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

    log.info("Merged adapter -> %s (%.2f GiB)", dest_uri, upload["bytes_uploaded"] / (1024**3))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge PEFT LoRA into base HF weights and upload to S3.")
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER, help="Staged LoRA prefix")
    parser.add_argument("--dest", default=DEFAULT_DEST, help="Destination S3 prefix for merged HF tree")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--manifest-out")
    parser.add_argument("--aws-region", default="us-east-1")
    parser.add_argument("--s3-endpoint-url")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    plan = {"adapter": args.adapter, "dest": args.dest, "dtype": args.dtype}
    if args.dry_run:
        log.info("[dry-run] %s", json.dumps(plan, indent=2))
        return 0

    try:
        merge_adapter(
            args.adapter,
            args.dest,
            dtype=args.dtype,
            device_map=args.device_map,
            attn_implementation=args.attn_implementation,
            aws_region=args.aws_region,
            s3_endpoint=args.s3_endpoint_url,
            manifest_out=args.manifest_out,
        )
    except Exception as exc:  # noqa: BLE001
        log.error("merge failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
