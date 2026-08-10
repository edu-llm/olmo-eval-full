#!/usr/bin/env python3
"""Lite HSIC mutual-information probe (TracingLLM paper Section 4) on three checkpoints."""

from __future__ import annotations

from typing import Any

import steering_common as sc


def _hsic(x, y) -> float:
    """Normalized HSIC between activation matrix x and binary labels y."""
    import torch

    x = torch.as_tensor(x, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.float32).reshape(-1, 1)
    n = x.shape[0]
    if n < 4:
        return 0.0
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    kx = x @ x.T
    ky = y @ y.T
    h = torch.eye(n) - torch.full((n, n), 1.0 / n)
    hsic = torch.trace(kx @ h @ ky @ h) / ((n - 1) ** 2)
    var = torch.sqrt(torch.trace(kx @ h @ kx @ h) * torch.trace(ky @ h @ ky @ h)) + 1e-8
    return float((hsic / var).clip(0, 1))


def mi_for_checkpoint(
    tokenizer,
    model,
    dataset: str,
    layers: list[int],
    device: str,
    max_statements: int,
) -> list[dict[str, Any]]:
    statements, labels = sc.load_probing_statements(dataset)
    if max_statements and max_statements < len(statements):
        statements, labels = statements[:max_statements], labels[:max_statements]
    acts = sc.collect_activations(tokenizer, model, statements, layers, device)
    rows = []
    for layer in layers:
        hsic = _hsic(acts[layer].numpy(), labels)
        rows.append({"dataset": dataset, "layer": layer, "hsic": hsic})
    return rows


def middle_layers(model, count: int = 5) -> list[int]:
    n = len(sc.decoder_layers(model))
    mid = n // 2
    half = count // 2
    return list(range(max(0, mid - half), min(n, mid - half + count)))


def run_mi_sweep(
    checkpoints: dict[str, tuple[Any, Any]],
    *,
    datasets: list[str] | None = None,
    device: str,
    max_statements: int,
) -> list[dict[str, Any]]:
    datasets = datasets or list(sc.LABELED_DATASETS)
    rows: list[dict[str, Any]] = []
    for role, (tokenizer, model) in checkpoints.items():
        layers = middle_layers(model)
        for dataset in datasets:
            for row in mi_for_checkpoint(tokenizer, model, dataset, layers, device, max_statements):
                rows.append({"checkpoint_role": role, **row})
    return rows
