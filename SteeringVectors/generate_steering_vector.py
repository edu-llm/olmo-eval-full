#!/usr/bin/env python3
"""Build steering vectors from one staged checkpoint (GPU smoke path).

Supports OLMo-core and HuggingFace checkpoints (e.g. Qwen3-30B-A3B-Thinking).
Read load options from ``--manifest`` or pass flags directly.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import steering_common as sc
from checkpoint_manifest import load_manifest, load_manifest_raw, manifest_checkpoint_uri, plan_dict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [generate_steering_vector] %(levelname)s: %(message)s",
)
log = logging.getLogger("generate_steering_vector")


def _serialize_vector(vec) -> bytes:
    import torch

    buf = BytesIO()
    torch.save(vec.cpu(), buf)
    return buf.getvalue()


def _resolve_layers(layers: list[int] | None, manifest_layers: list[int] | None) -> list[int]:
    if layers:
        return layers
    if manifest_layers:
        return list(manifest_layers)
    return [6]


def run(
    checkpoint: str,
    dataset: str,
    layers: list[int],
    *,
    max_statements: int = 200,
    train_ratio: float = 0.5,
    seed: int = 1234,
    load_options: sc.LoadModelOptions | None = None,
    results_s3: str | None = None,
    run_name: str = "steering-vector-smoke",
    aws_region: str = "us-east-1",
    s3_endpoint: str | None = None,
) -> dict[str, Any]:
    if dataset not in sc.LABELED_DATASETS:
        raise SystemExit(
            f"Unknown dataset {dataset!r}; choose from {', '.join(sc.LABELED_DATASETS)}."
        )

    opts = load_options or sc.LoadModelOptions(seed=seed)
    opts.seed = seed
    step = sc.parse_step(checkpoint)
    with tempfile.TemporaryDirectory(prefix="steer-vec-") as tmp:
        tmp_path = Path(tmp)
        raw = sc.materialize_checkpoint(checkpoint, tmp_path / "raw", aws_region, s3_endpoint)
        hf = sc.ensure_hf_checkpoint(raw, tmp_path / "hf")
        tokenizer, model = sc.load_model(hf, options=opts)

        layer_list = sc.in_range_layers(model, layers)
        statements, labels = sc.load_probing_statements(dataset)
        vectors = sc.build_steering_vectors(
            tokenizer,
            model,
            statements,
            labels,
            layer_list,
            sc.model_input_device(model, opts.device),
            train_ratio,
            max_statements,
        )

    vector_meta: dict[str, Any] = {}
    uploaded: list[str] = []
    for layer, vec in vectors.items():
        norm = float(vec.norm().item())
        vector_meta[str(layer)] = {"norm": norm, "shape": list(vec.shape)}
        if results_s3:
            body = _serialize_vector(vec)
            uri = sc.upload_binary(
                results_s3,
                run_name,
                step,
                f"{dataset}_layer{layer}.pt",
                body,
                region=aws_region,
                endpoint=s3_endpoint,
            )
            uploaded.append(uri)

    payload = {
        "checkpoint": checkpoint,
        "step": step,
        "dataset": dataset,
        "layers": layer_list,
        "max_statements": max_statements,
        "train_ratio": train_ratio,
        "load": {
            "dtype": opts.dtype,
            "device_map": opts.device_map,
            "model_family": opts.model_family,
            "enable_thinking": opts.enable_thinking,
        },
        "vectors": vector_meta,
        "uploaded": uploaded,
        "run_name": run_name,
        "success": True,
        "timestamp": datetime.now(UTC).isoformat(),
    }

    out_dir = Path("outputs") / run_name / f"step{step}"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "steering_vectors.json"
    manifest_path.write_text(json.dumps(payload, indent=2))
    log.info("Wrote %s", manifest_path)

    if results_s3:
        sc.upload_files(
            results_s3,
            run_name,
            step,
            {"steering_vectors.json": json.dumps(payload, indent=2)},
            aws_region,
            s3_endpoint,
        )

    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate steering vectors from one checkpoint.")
    parser.add_argument("checkpoint", nargs="?", help="Staged checkpoint S3 URI or local dir")
    parser.add_argument("--checkpoint", dest="checkpoint_flag", help="Checkpoint URI override")
    parser.add_argument(
        "--manifest",
        help="JSON manifest; uses checkpoints.final and load block when --checkpoint omitted",
    )
    parser.add_argument(
        "--dataset",
        default="stereoset",
        choices=sorted(sc.LABELED_DATASETS),
    )
    parser.add_argument("--layers", type=int, nargs="+")
    parser.add_argument("--max-statements", type=int, default=200)
    parser.add_argument("--train-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        default=None,
        choices=["auto", "bfloat16", "float16", "float32"],
    )
    parser.add_argument("--device-map", help="HF device_map, e.g. auto for MoE models")
    parser.add_argument("--attn-implementation", help="e.g. sdpa or flash_attention_2")
    parser.add_argument("--results-s3", help="Upload prefix, e.g. $EDULLM_OUTPUT_PREFIX")
    parser.add_argument("--run-name")
    parser.add_argument("--aws-region", default="us-east-1")
    parser.add_argument("--s3-endpoint-url")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest) if args.manifest else None
    raw_manifest = load_manifest_raw(manifest_path) if manifest_path else {}
    checkpoint = args.checkpoint_flag or args.checkpoint
    if not checkpoint and manifest_path:
        if args.dry_run:
            checkpoint = raw_manifest["checkpoints"]["final"].rstrip("/") + "/"
        else:
            checkpoint = manifest_checkpoint_uri(raw_manifest, "final")

    if not checkpoint:
        parser.error("Pass --checkpoint, a positional URI, or --manifest with checkpoints.final.")

    load_opts = sc.load_options_from_manifest(raw_manifest, args.device) if raw_manifest else None
    if load_opts is None:
        load_opts = sc.LoadModelOptions(device=args.device)
    if args.dtype:
        load_opts.dtype = args.dtype
    if args.device_map:
        load_opts.device_map = args.device_map
    if args.attn_implementation:
        load_opts.attn_implementation = args.attn_implementation

    manifest_layers = raw_manifest.get("steering_layers")
    layers = _resolve_layers(args.layers, manifest_layers)
    run_name = args.run_name or raw_manifest.get("run_name") or "steering-vector-smoke"
    max_statements = args.max_statements
    if raw_manifest.get("max_statements") and args.max_statements == 200:
        max_statements = int(raw_manifest["max_statements"])

    plan = {
        "checkpoint": checkpoint,
        "dataset": args.dataset,
        "layers": layers,
        "max_statements": max_statements,
        "results_s3": args.results_s3,
        "run_name": run_name,
        "load": {
            "dtype": load_opts.dtype,
            "device_map": load_opts.device_map,
            "model_family": load_opts.model_family,
        },
    }
    if manifest_path and args.dry_run:
        plan["manifest_plan"] = plan_dict(load_manifest(manifest_path, dry_run=True))
    if args.dry_run:
        log.info("[dry-run] %s", json.dumps(plan, indent=2, default=str))
        return 0

    try:
        run(
            checkpoint,
            args.dataset,
            layers,
            max_statements=max_statements,
            train_ratio=args.train_ratio,
            seed=args.seed,
            load_options=load_opts,
            results_s3=args.results_s3,
            run_name=run_name,
            aws_region=args.aws_region,
            s3_endpoint=args.s3_endpoint_url,
        )
    except Exception as exc:  # noqa: BLE001
        log.error("vector generation failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
