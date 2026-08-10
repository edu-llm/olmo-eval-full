#!/usr/bin/env python3
"""Linear-probing dynamics across pre-training checkpoints.

Pillar 1 of the TracingLLM (arXiv 2402.19465) small-scale replication. For each
pre-training checkpoint and each trustworthiness dimension, it feeds the vendored
statements through the model, captures the last-token hidden state at every
decoder layer, and fits a binary linear probe (4:1 split) per layer, reporting
test accuracy -- reproducing ``vendor/TracingLLM/src/train_probes.py``.

Following the paper (Section 2.2), the probe is a logistic-regression classifier
on standardized last-token activations. It is implemented here in torch (L2-
regularized logistic regression via L-BFGS) so the platform image needs no
scikit-learn.

Heavy dependencies (torch, transformers, boto3, olmo_core) are imported lazily so
``--dry-run`` validates the plan with none of them installed. Launch as an eduLLM
platform job (see ``.edullm/run.yaml``); it is not a laptop-side AWS entrypoint.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import steering_common as sc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [probe_dynamics] %(levelname)s: %(message)s",
)
log = logging.getLogger("probe_dynamics")


@dataclass
class Config:
    """A probing-dynamics sweep over several checkpoints of one pre-training run."""

    checkpoints: list[str]
    datasets: list[str] = field(default_factory=lambda: list(sc.LABELED_DATASETS))
    layers: list[int] | None = None  # None = every decoder layer
    max_statements: int = 1000  # balanced cap per dataset (0 = use all)
    test_ratio: float = 0.2  # 4:1 train/test split (paper Section 2.2)
    seed: int = 1234
    device: str = "cuda:0"
    results_s3: str | None = None
    run_name: str = "probe-dynamics"
    aws_region: str = "us-east-1"
    s3_endpoint_url: str | None = None

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> Config:
        def env(name: str, default: str | None = None) -> str | None:
            return os.environ.get(name, default)

        def _strs(value: str | None) -> list[str] | None:
            return [x for x in re.split(r"[,\s]+", value.strip()) if x] if value else None

        def _ints(value: str | None) -> list[int] | None:
            return [int(x) for x in re.split(r"[,\s]+", value.strip()) if x] if value else None

        checkpoints = args.checkpoints or _strs(env("CHECKPOINTS_S3"))
        if not checkpoints:
            raise SystemExit(
                "No checkpoints given. Pass one or more S3 URIs / local dirs or set CHECKPOINTS_S3."
            )
        datasets = args.datasets or _strs(env("DATASETS")) or list(sc.LABELED_DATASETS)
        layers = args.layers or _ints(env("LAYERS"))

        return cls(
            checkpoints=checkpoints,
            datasets=datasets,
            layers=layers,
            max_statements=int(args.max_statements or env("MAX_STATEMENTS", "1000")),
            test_ratio=float(args.test_ratio or env("TEST_RATIO", "0.2")),
            seed=int(args.seed or env("SEED", "1234")),
            device=args.device or env("DEVICE", "cuda:0"),
            results_s3=args.results_s3 or env("RESULTS_S3"),
            run_name=args.run_name or env("RUN_NAME", "probe-dynamics"),
            aws_region=env("AWS_REGION", "us-east-1"),
            s3_endpoint_url=env("S3_ENDPOINT_URL"),
        )

    def validate(self) -> None:
        for name in self.datasets:
            if name not in sc.LABELED_DATASETS:
                raise SystemExit(
                    f"Unknown dataset {name!r}; choose from {', '.join(sc.LABELED_DATASETS)}."
                )
            if not sc.dataset_path(name).exists():
                raise SystemExit(f"Vendored dataset not found: {sc.dataset_path(name)}")


def cap_balanced(statements: list[str], labels: list[int], cap: int) -> tuple[list[str], list[int]]:
    """Take up to ``cap`` items keeping the two classes balanced and order stable."""
    if not cap or cap >= len(statements):
        return statements, labels
    per_class = cap // 2
    kept_s: list[str] = []
    kept_y: list[int] = []
    counts = {0: 0, 1: 0}
    for statement, label in zip(statements, labels, strict=True):
        if label in counts and counts[label] < per_class:
            kept_s.append(statement)
            kept_y.append(label)
            counts[label] += 1
        if len(kept_s) >= per_class * 2:
            break
    return kept_s, kept_y


def train_probe(acts, labels, test_ratio: float, seed: int, device: str) -> dict[str, float]:
    """L2-regularized logistic regression on standardized activations, 4:1 split.

    Returns test accuracy plus split sizes.
    """
    import torch
    import torch.nn.functional as F

    y = torch.tensor(labels, dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(acts.shape[0], generator=generator)
    acts, y = acts[perm], y[perm]

    n_test = max(1, int(acts.shape[0] * test_ratio))
    x_test, y_test = acts[:n_test], y[:n_test]
    x_train, y_train = acts[n_test:], y[n_test:]

    # Standardize by train statistics (center + scale, no leakage).
    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, keepdim=True) + 1e-6
    x_train = ((x_train - mean) / std).to(device)
    x_test = ((x_test - mean) / std).to(device)
    y_train = y_train.to(device)

    dim = acts.shape[1]
    weight = torch.zeros(dim, device=device, requires_grad=True)
    bias = torch.zeros(1, device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS([weight, bias], max_iter=100, line_search_fn="strong_wolfe")
    inv_reg = 1.0  # analogue of sklearn LogisticRegression C=1.0

    def closure():
        optimizer.zero_grad()
        logits = x_train @ weight + bias
        loss = F.binary_cross_entropy_with_logits(logits, y_train)
        loss = loss + (1.0 / (2.0 * inv_reg * y_train.shape[0])) * (weight @ weight)
        loss.backward()
        return loss

    optimizer.step(closure)

    with torch.no_grad():
        pred = ((x_test @ weight + bias) > 0).long().cpu()
    accuracy = (pred == y_test.long()).float().mean().item()
    return {
        "accuracy": accuracy,
        "n_train": int(y_train.shape[0]),
        "n_test": int(y_test.shape[0]),
    }


def probe_checkpoint(cfg: Config, uri: str, tmp: Path) -> list[dict[str, Any]]:
    step = sc.parse_step(uri)
    log.info("Probing checkpoint step %s: %s", step, uri)
    raw = sc.materialize_checkpoint(uri, tmp / "raw", cfg.aws_region, cfg.s3_endpoint_url)
    ckpt_dir = sc.ensure_hf_checkpoint(raw, tmp / "hf")
    tokenizer, model = sc.load_model(ckpt_dir, cfg.device, cfg.seed)

    num_layers = len(sc.decoder_layers(model))
    layers = cfg.layers if cfg.layers else list(range(num_layers))
    layers = sc.in_range_layers(model, layers)

    rows: list[dict[str, Any]] = []
    for dataset in cfg.datasets:
        statements, labels = sc.load_probing_statements(dataset)
        statements, labels = cap_balanced(statements, labels, cfg.max_statements)
        log.info(
            "[step %s] %s: %d statements, %d layers",
            step,
            dataset,
            len(statements),
            len(layers),
        )
        acts = sc.collect_activations(tokenizer, model, statements, layers, cfg.device)
        for layer in layers:
            result = train_probe(acts[layer], labels, cfg.test_ratio, cfg.seed, cfg.device)
            rows.append({"step": step, "dataset": dataset, "layer": layer, **result})
            log.info(
                "[step %s] %s layer=%d acc=%.4f (train=%d/test=%d)",
                step,
                dataset,
                layer,
                result["accuracy"],
                result["n_train"],
                result["n_test"],
            )
    return rows


def run(cfg: Config) -> dict[str, Any]:
    all_rows: list[dict[str, Any]] = []
    for uri in cfg.checkpoints:
        with tempfile.TemporaryDirectory(prefix="probe-") as tmp:
            all_rows.extend(probe_checkpoint(cfg, uri, Path(tmp)))

    metrics = {
        "run": cfg.run_name,
        "checkpoints": cfg.checkpoints,
        "steps": [sc.parse_step(uri) for uri in cfg.checkpoints],
        "datasets": cfg.datasets,
        "max_statements": cfg.max_statements,
        "test_ratio": cfg.test_ratio,
        "results": all_rows,
        "seed": cfg.seed,
        "success": True,
        "timestamp": datetime.now(UTC).isoformat(),
    }

    files = {"probe_accuracy.json": json.dumps(metrics, indent=2)}
    out_dir = Path("outputs") / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (out_dir / name).write_text(content)
    log.info("Wrote results to %s", out_dir)

    if cfg.results_s3:
        metrics["s3_location"] = sc.upload_files(
            cfg.results_s3, cfg.run_name, "all", files, cfg.aws_region, cfg.s3_endpoint_url
        )
    return metrics


def _plan(cfg: Config) -> dict[str, Any]:
    per_dataset = {}
    for dataset in cfg.datasets:
        statements, _ = sc.load_probing_statements(dataset)
        capped = min(cfg.max_statements, len(statements)) if cfg.max_statements else len(statements)
        per_dataset[dataset] = {"available": len(statements), "used": capped}
    return {
        "checkpoints": cfg.checkpoints,
        "steps": [sc.parse_step(uri) for uri in cfg.checkpoints],
        "datasets": per_dataset,
        "layers": cfg.layers or "all",
        "test_ratio": cfg.test_ratio,
        "results_s3": cfg.results_s3,
        "would_upload_to": (
            f"{cfg.results_s3.rstrip('/')}/{cfg.run_name}/stepall/" if cfg.results_s3 else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Linear-probing dynamics across pre-training checkpoints."
    )
    parser.add_argument("checkpoints", nargs="*", help="Checkpoint S3 URIs / local dirs (per step)")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(sc.LABELED_DATASETS),
        help="Trustworthiness datasets to probe (default: all five)",
    )
    parser.add_argument("--layers", type=int, nargs="+", help="Decoder layers (default: all)")
    parser.add_argument("--max-statements", type=int, help="Balanced cap per dataset (0 = all)")
    parser.add_argument("--test-ratio", type=float, help="Held-out test fraction (default 0.2)")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--results-s3", help="S3 prefix for outputs")
    parser.add_argument("--run-name")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config + datasets and print the plan without torch, S3, or network",
    )
    args = parser.parse_args()

    cfg = Config.from_args(args)
    cfg.validate()

    if args.dry_run:
        log.info("[dry-run] plan:\n%s", json.dumps(_plan(cfg), indent=2))
        return 0

    try:
        metrics = run(cfg)
    except Exception as exc:  # noqa: BLE001 - surface a clear failure for the platform run
        log.error("probe dynamics failed: %s", exc)
        return 1

    log.info("Done. %d checkpoints, %d probe rows", len(cfg.checkpoints), len(metrics["results"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
