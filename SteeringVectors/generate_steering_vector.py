#!/usr/bin/env python3
"""Build steering vectors from one staged checkpoint (GPU smoke path).

Materializes the checkpoint, converts OLMo-core to HF if needed, extracts
mean-difference steering vectors for the requested dataset/layers, and uploads
``.pt`` tensors plus a JSON manifest to S3.
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


def run(
    checkpoint: str,
    dataset: str,
    layers: list[int],
    *,
    max_statements: int = 200,
    train_ratio: float = 0.5,
    seed: int = 1234,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    results_s3: str | None = None,
    run_name: str = "steering-vector-smoke",
    aws_region: str = "us-east-1",
    s3_endpoint: str | None = None,
) -> dict[str, Any]:
    if dataset not in sc.LABELED_DATASETS:
        raise SystemExit(
            f"Unknown dataset {dataset!r}; choose from {', '.join(sc.LABELED_DATASETS)}."
        )

    step = sc.parse_step(checkpoint)
    with tempfile.TemporaryDirectory(prefix="steer-vec-") as tmp:
        tmp_path = Path(tmp)
        raw = sc.materialize_checkpoint(checkpoint, tmp_path / "raw", aws_region, s3_endpoint)
        hf = sc.ensure_hf_checkpoint(raw, tmp_path / "hf")
        tokenizer, model = sc.load_model(hf, device, seed)
        if dtype != "auto":
            import torch

            model = model.to(dtype=getattr(torch, dtype))

        layer_list = sc.in_range_layers(model, layers)
        statements, labels = sc.load_probing_statements(dataset)
        vectors = sc.build_steering_vectors(
            tokenizer,
            model,
            statements,
            labels,
            layer_list,
            device,
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
        "dtype": dtype,
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
    parser.add_argument(
        "--checkpoint",
        dest="checkpoint_flag",
        help="Staged checkpoint S3 URI (alternative to positional)",
    )
    parser.add_argument(
        "--dataset",
        default="stereoset",
        choices=sorted(sc.LABELED_DATASETS),
    )
    parser.add_argument("--layers", type=int, nargs="+", default=[6])
    parser.add_argument("--max-statements", type=int, default=200)
    parser.add_argument("--train-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["auto", "bfloat16", "float16", "float32"],
        help="Model dtype (bfloat16 required in platform command for the guard)",
    )
    parser.add_argument("--results-s3", help="Upload prefix, e.g. $EDULLM_OUTPUT_PREFIX")
    parser.add_argument("--run-name", default="steering-vector-smoke")
    parser.add_argument("--aws-region", default="us-east-1")
    parser.add_argument("--s3-endpoint-url")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    checkpoint = args.checkpoint_flag or args.checkpoint
    if not checkpoint:
        parser.error("Pass a checkpoint URI as a positional argument or --checkpoint.")

    plan = {
        "checkpoint": checkpoint,
        "dataset": args.dataset,
        "layers": args.layers,
        "max_statements": args.max_statements,
        "results_s3": args.results_s3,
        "run_name": args.run_name,
        "dtype": args.dtype,
    }
    if args.dry_run:
        log.info("[dry-run] %s", json.dumps(plan, indent=2))
        return 0

    try:
        run(
            checkpoint,
            args.dataset,
            args.layers,
            max_statements=args.max_statements,
            train_ratio=args.train_ratio,
            seed=args.seed,
            device=args.device,
            dtype=args.dtype,
            results_s3=args.results_s3,
            run_name=args.run_name,
            aws_region=args.aws_region,
            s3_endpoint=args.s3_endpoint_url,
        )
    except Exception as exc:  # noqa: BLE001
        log.error("vector generation failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
