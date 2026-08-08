#!/usr/bin/env python3
"""On-node checkpoint inference TEST (Flow 1).

Standalone script a training team invokes once per checkpoint. It:

  1. Takes a checkpoint S3 URI (a different one every call).
  2. Downloads/loads that checkpoint on the local GPU(s).
  3. Runs a few random inferences (prompts sampled with a fixed seed).
  4. Uploads the outputs to a deterministic S3 location.

It does NOT depend on the olmo_eval package - only a model backend + boto3.

See Plan/flow1_training_checkpoint/README.md for the design.

Usage:
    python checkpoint_infer.py s3://bucket/checkpoints/owner/run/step1000/

Configuration is read from environment variables (see checkpoint_infer.env.example).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [checkpoint_infer] %(levelname)s: %(message)s",
)
log = logging.getLogger("checkpoint_infer")

DEFAULT_PROMPTS_FILE = Path(__file__).parent / "prompts.jsonl"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """Runtime configuration, sourced from env vars with sensible defaults."""

    results_bucket: str
    run_name: str
    results_prefix: str = "checkpoint-infer"
    checkpoint_kind: str = "hf"  # "hf" or "olmo_core"
    prompts_file: Path = DEFAULT_PROMPTS_FILE
    num_prompts: int = 8
    max_new_tokens: int = 64
    temperature: float = 0.0
    seed: int = 1234
    num_gpus: int = 1
    aws_region: str = "us-east-1"
    s3_endpoint_url: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> Config:
        missing = [k for k in ("RESULTS_BUCKET", "RUN_NAME") if not os.environ.get(k)]
        if missing:
            raise SystemExit(
                f"Missing required environment variables: {', '.join(missing)}. "
                "See checkpoint_infer.env.example."
            )

        prompts_file = os.environ.get("PROMPTS_FILE")
        return cls(
            results_bucket=os.environ["RESULTS_BUCKET"],
            run_name=os.environ["RUN_NAME"],
            results_prefix=os.environ.get("RESULTS_PREFIX", "checkpoint-infer"),
            checkpoint_kind=os.environ.get("CHECKPOINT_KIND", "hf").lower(),
            prompts_file=Path(prompts_file) if prompts_file else DEFAULT_PROMPTS_FILE,
            num_prompts=int(os.environ.get("NUM_PROMPTS", "8")),
            max_new_tokens=int(os.environ.get("MAX_NEW_TOKENS", "64")),
            temperature=float(os.environ.get("TEMPERATURE", "0.0")),
            seed=int(os.environ.get("SEED", "1234")),
            num_gpus=int(os.environ.get("NUM_GPUS", "1")),
            aws_region=os.environ.get("AWS_REGION", "us-east-1"),
            s3_endpoint_url=os.environ.get("S3_ENDPOINT_URL"),
            extra={"olmo_core_package": os.environ.get("OLMO_CORE_PACKAGE")},
        )


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------
def _s3_client(cfg: Config):
    import boto3

    kwargs: dict[str, Any] = {"region_name": cfg.aws_region}
    if cfg.s3_endpoint_url:
        kwargs["endpoint_url"] = cfg.s3_endpoint_url
    return boto3.client("s3", **kwargs)


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Split ``s3://bucket/key/prefix`` into ``(bucket, key_prefix)``."""
    if not uri.startswith("s3://"):
        raise ValueError(f"Not an S3 URI: {uri}")
    without_scheme = uri[len("s3://") :]
    bucket, _, key = without_scheme.partition("/")
    return bucket, key.rstrip("/")


def parse_step(uri: str) -> str:
    """Extract the training step from a checkpoint URI (e.g. ``.../step1000-hf/``)."""
    match = re.search(r"step(\d+)", uri)
    return match.group(1) if match else "unknown"


def download_checkpoint(cfg: Config, uri: str, dest: Path) -> Path:
    """Download every object under the checkpoint prefix into ``dest``."""
    bucket, prefix = parse_s3_uri(uri)
    client = _s3_client(cfg)
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
    log.info("Downloaded %d checkpoint files to %s", downloaded, dest)
    return dest


def upload_results(cfg: Config, step: str, files: dict[str, str]) -> str:
    """Upload result files to s3://{bucket}/{prefix}/{run}/step{N}/ and return the base URI."""
    client = _s3_client(cfg)
    base_key = f"{cfg.results_prefix}/{cfg.run_name}/step{step}"
    for name, content in files.items():
        client.put_object(
            Bucket=cfg.results_bucket,
            Key=f"{base_key}/{name}",
            Body=content.encode("utf-8"),
            ContentType="application/json" if name.endswith(".json") else "application/x-ndjson",
        )
    base_uri = f"s3://{cfg.results_bucket}/{base_key}"
    log.info("Uploaded results to %s", base_uri)
    return base_uri


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
def load_prompts(cfg: Config) -> list[dict[str, Any]]:
    """Load the prompt pool and randomly sample ``num_prompts`` with a fixed seed."""
    if not cfg.prompts_file.exists():
        raise SystemExit(f"Prompts file not found: {cfg.prompts_file}")

    pool: list[dict[str, Any]] = []
    for line in cfg.prompts_file.read_text().splitlines():
        line = line.strip()
        if line:
            pool.append(json.loads(line))

    if not pool:
        raise SystemExit(f"Prompt pool is empty: {cfg.prompts_file}")

    rng = random.Random(cfg.seed)
    k = min(cfg.num_prompts, len(pool))
    return rng.sample(pool, k)


# ---------------------------------------------------------------------------
# Model loading + generation (backend-specific)
# ---------------------------------------------------------------------------
def load_model(cfg: Config, local_dir: Path):
    """Load the model + tokenizer for the configured checkpoint kind.

    Returns a callable ``generate(prompt: str) -> str``.
    """
    if cfg.checkpoint_kind == "hf":
        return _load_hf(cfg, local_dir)
    if cfg.checkpoint_kind == "olmo_core":
        return _load_olmo_core(cfg, local_dir)
    raise SystemExit(
        f"Unknown CHECKPOINT_KIND: {cfg.checkpoint_kind} (expected 'hf' or 'olmo_core')"
    )


def _load_hf(cfg: Config, local_dir: Path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    set_seed(cfg.seed)
    tokenizer = AutoTokenizer.from_pretrained(str(local_dir))
    model = AutoModelForCausalLM.from_pretrained(
        str(local_dir),
        torch_dtype="auto",
        device_map="auto" if cfg.num_gpus >= 1 else None,
    )
    model.eval()

    def generate(prompt: str) -> str:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=cfg.temperature > 0.0,
                temperature=cfg.temperature if cfg.temperature > 0.0 else None,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    return generate


def _load_olmo_core(cfg: Config, local_dir: Path):
    """Load a raw OLMo-core checkpoint using the trainer's own library.

    TODO(training-team): reconstruct the model + tokenizer from the run config and
    load the checkpoint weights. This depends on the run's config layout and is left
    as an integration point (see open questions in the plan).
    """
    raise NotImplementedError(
        "olmo_core loading is a training-env integration point. Provide the run config "
        "and checkpoint layout, then implement _load_olmo_core. See the plan's open questions."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(cfg: Config, checkpoint_uri: str) -> str:
    step = parse_step(checkpoint_uri)
    log.info("Checkpoint %s (step %s), kind=%s", checkpoint_uri, step, cfg.checkpoint_kind)

    prompts = load_prompts(cfg)
    log.info("Sampled %d prompts (seed=%d)", len(prompts), cfg.seed)

    with tempfile.TemporaryDirectory(prefix=f"ckpt-infer-step{step}-") as tmp:
        local_dir = download_checkpoint(cfg, checkpoint_uri, Path(tmp))

        load_start = time.time()
        generate = load_model(cfg, local_dir)
        load_seconds = time.time() - load_start
        log.info("Model loaded in %.1fs", load_seconds)

        rows: list[str] = []
        for item in prompts:
            prompt = item["prompt"]
            t0 = time.time()
            output = generate(prompt)
            rows.append(
                json.dumps(
                    {
                        "checkpoint": checkpoint_uri,
                        "step": step,
                        "run": cfg.run_name,
                        "prompt_id": item.get("id"),
                        "prompt": prompt,
                        "output": output,
                        "gen_params": {
                            "max_new_tokens": cfg.max_new_tokens,
                            "temperature": cfg.temperature,
                        },
                        "timestamp": datetime.now(UTC).isoformat(),
                        "wall_time_s": round(time.time() - t0, 3),
                    }
                )
            )

        manifest = json.dumps(
            {
                "checkpoint": checkpoint_uri,
                "step": step,
                "run": cfg.run_name,
                "checkpoint_kind": cfg.checkpoint_kind,
                "num_prompts": len(prompts),
                "model_load_seconds": round(load_seconds, 3),
                "success": True,
                "timestamp": datetime.now(UTC).isoformat(),
            },
            indent=2,
        )

    return upload_results(
        cfg,
        step,
        {"results.jsonl": "\n".join(rows) + "\n", "manifest.json": manifest},
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="On-node checkpoint inference TEST (Flow 1)")
    parser.add_argument("checkpoint_uri", help="Checkpoint S3 URI, e.g. s3://bucket/.../step1000/")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config + prompts and print the plan without loading the model",
    )
    args = parser.parse_args()

    cfg = Config.from_env()

    if args.dry_run:
        step = parse_step(args.checkpoint_uri)
        prompts = load_prompts(cfg)
        log.info(
            "[dry-run] would run %d prompts on %s and upload to s3://%s/%s/%s/step%s/",
            len(prompts),
            args.checkpoint_uri,
            cfg.results_bucket,
            cfg.results_prefix,
            cfg.run_name,
            step,
        )
        return 0

    try:
        base_uri = run(cfg, args.checkpoint_uri)
    except Exception as exc:  # noqa: BLE001 - surface a clear failure for the training team
        log.error("checkpoint_infer failed: %s", exc)
        return 1

    log.info("Done. Results at %s", base_uri)
    return 0


if __name__ == "__main__":
    sys.exit(main())
