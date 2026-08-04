#!/usr/bin/env python3
"""Mock training run for validating Flow 1 (on-node checkpoint inference).

Simulates a training loop that periodically writes a checkpoint to S3 and, right
after each checkpoint is written, fires the standalone ``checkpoint_infer.py``
hook (backgrounded) so inference runs on that checkpoint without blocking the
loop. It is deliberately tiny and CPU-only: a small GPT-2 model with a byte-level
tokenizer is built locally (no HuggingFace Hub download), perturbed per step so
each checkpoint is distinct, saved in HuggingFace format, and uploaded under the
AI2 layout ``s3://{bucket}/checkpoints/{owner}/{run}/step{N}/``.

See ``Plan/flow1_training_checkpoint/README.md`` (Validation) and
``tests/OnNode/skill/SKILL.md`` (Step 5) for the flow this exercises.

Configuration comes from the same environment variables ``checkpoint_infer.py``
reads (``RESULTS_BUCKET``, ``RUN_NAME``, ``S3_ENDPOINT_URL``, ...) plus a few that
only shape the mock loop (see ``from_env``).
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [mock_trainer] %(levelname)s: %(message)s",
)
log = logging.getLogger("mock_trainer")

CHECKPOINT_INFER = Path(__file__).resolve().parents[1] / "checkpoint_infer.py"


@dataclass
class MockConfig:
    """Settings for the mock loop, sourced from env vars."""

    results_bucket: str
    run_name: str
    owner: str = "mockowner"
    num_steps: int = 3
    step_size: int = 100
    aws_region: str = "us-east-1"
    s3_endpoint_url: str | None = None
    blocking: bool = False
    perturb_scale: float = 0.05
    model_seed: int = 7

    @classmethod
    def from_env(cls) -> MockConfig:
        missing = [k for k in ("RESULTS_BUCKET", "RUN_NAME") if not os.environ.get(k)]
        if missing:
            raise SystemExit(
                f"Missing required environment variables: {', '.join(missing)}. "
                "See tests/OnNode/checkpoint_infer.env.example."
            )
        return cls(
            results_bucket=os.environ["RESULTS_BUCKET"],
            run_name=os.environ["RUN_NAME"],
            owner=os.environ.get("OWNER", "mockowner"),
            num_steps=int(os.environ.get("MOCK_NUM_STEPS", "3")),
            step_size=int(os.environ.get("MOCK_STEP_SIZE", "100")),
            aws_region=os.environ.get("AWS_REGION", "us-east-1"),
            s3_endpoint_url=os.environ.get("S3_ENDPOINT_URL"),
            blocking=os.environ.get("BLOCKING", "false").lower() == "true",
            perturb_scale=float(os.environ.get("MOCK_PERTURB_SCALE", "0.05")),
            model_seed=int(os.environ.get("MOCK_MODEL_SEED", "7")),
        )


def _s3_client(cfg: MockConfig):
    import boto3

    kwargs: dict[str, Any] = {"region_name": cfg.aws_region}
    if cfg.s3_endpoint_url:
        kwargs["endpoint_url"] = cfg.s3_endpoint_url
    return boto3.client("s3", **kwargs)


def build_base_model():
    """Construct a tiny GPT-2 model and byte-level tokenizer entirely offline."""
    from tokenizers import Tokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    alphabet = sorted(ByteLevel.alphabet())
    vocab = {ch: i for i, ch in enumerate(alphabet)}
    eos = "<|endoftext|>"
    vocab[eos] = len(vocab)

    backend = Tokenizer(BPE(vocab=vocab, merges=[]))
    backend.pre_tokenizer = ByteLevel(add_prefix_space=False)
    backend.decoder = ByteLevelDecoder()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend)
    tokenizer.add_special_tokens({"eos_token": eos, "bos_token": eos, "pad_token": eos})

    config = GPT2Config(
        vocab_size=len(vocab),
        n_positions=256,
        n_embd=32,
        n_layer=2,
        n_head=2,
        bos_token_id=vocab[eos],
        eos_token_id=vocab[eos],
    )
    model = GPT2LMHeadModel(config)
    return model, tokenizer


def perturb_and_save(base_state, model, tokenizer, step: int, scale: float, dest: Path):
    """Reset the model to ``base_state`` plus step-seeded noise, then save it."""
    import torch

    generator = torch.Generator().manual_seed(step)
    new_state = {}
    for name, tensor in base_state.items():
        noise = torch.empty_like(tensor).normal_(mean=0.0, std=scale, generator=generator)
        new_state[name] = tensor + noise
    model.load_state_dict(new_state)
    dest.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(dest)
    tokenizer.save_pretrained(dest)


def upload_checkpoint(cfg: MockConfig, local_dir: Path, step: int) -> str:
    """Upload a local checkpoint dir to S3 and return its ``s3://.../stepN/`` URI."""
    client = _s3_client(cfg)
    base_key = f"checkpoints/{cfg.owner}/{cfg.run_name}/step{step}"
    for path in sorted(local_dir.rglob("*")):
        if path.is_file():
            rel = path.relative_to(local_dir).as_posix()
            client.upload_file(str(path), cfg.results_bucket, f"{base_key}/{rel}")
    uri = f"s3://{cfg.results_bucket}/{base_key}/"
    log.info("step %d: uploaded checkpoint to %s", step, uri)
    return uri


def fire_hook(checkpoint_uri: str, blocking: bool) -> subprocess.Popen | None:
    """Fire the Flow 1 hook exactly as the skill/plan prescribe.

    Equivalent to ``python tests/OnNode/checkpoint_infer.py "$CKPT" &`` — the
    checkpoint URI is the only per-call argument; all other config is inherited
    from the environment.
    """
    cmd = [sys.executable, str(CHECKPOINT_INFER), checkpoint_uri]
    log.info(
        "firing hook: python %s %s %s", CHECKPOINT_INFER, checkpoint_uri, "" if blocking else "&"
    )
    if blocking:
        subprocess.run(cmd, check=False)
        return None
    return subprocess.Popen(cmd)


def main() -> int:
    cfg = MockConfig.from_env()
    log.info(
        "mock training: run=%s bucket=%s steps=%d endpoint=%s blocking=%s",
        cfg.run_name,
        cfg.results_bucket,
        cfg.num_steps,
        cfg.s3_endpoint_url or "(real AWS)",
        cfg.blocking,
    )

    model, tokenizer = build_base_model()
    base_state = {name: tensor.clone() for name, tensor in model.state_dict().items()}

    procs: list[tuple[int, subprocess.Popen]] = []
    with tempfile.TemporaryDirectory(prefix="mock-train-") as workdir:
        for i in range(cfg.num_steps):
            step = (i + 1) * cfg.step_size
            log.info("=== training step %d ===", step)
            # Emulate a little training compute before the checkpoint boundary.
            time.sleep(0.2)

            ckpt_dir = Path(workdir) / f"step{step}"
            perturb_and_save(base_state, model, tokenizer, step, cfg.perturb_scale, ckpt_dir)
            checkpoint_uri = upload_checkpoint(cfg, ckpt_dir, step)

            fired_at = time.time()
            proc = fire_hook(checkpoint_uri, cfg.blocking)
            handoff = time.time() - fired_at
            if proc is not None:
                procs.append((step, proc))
                log.info(
                    "step %d: hook backgrounded in %.3fs; trainer continues immediately",
                    step,
                    handoff,
                )

    if procs:
        log.info("training loop done; waiting for %d background inference job(s)", len(procs))
        failures = 0
        for step, proc in procs:
            code = proc.wait()
            log.info("step %d: inference exited with code %d", step, code)
            failures += int(code != 0)
        if failures:
            log.error("%d inference job(s) failed", failures)
            return 1

    log.info("mock training run complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
