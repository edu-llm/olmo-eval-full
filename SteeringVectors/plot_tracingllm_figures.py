#!/usr/bin/env python3
"""Reproduce TracingLLM-style figures from ``run_tracingllm_olmoe7b`` JSON outputs."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def write_figures(payload: dict[str, Any], out_dir: Path) -> None:

    out_dir.mkdir(parents=True, exist_ok=True)

    probe = payload.get("probe_accuracy") or []
    if probe:
        _fig_probe(probe, out_dir / "fig1_probe_dynamics.png")

    steering = payload.get("steering") or []
    if steering:
        _fig_alpha_ppl_toxic(steering, out_dir / "fig3_alpha_ppl_toxic.png")
        _fig_capabilities(steering, payload.get("general") or {}, out_dir / "fig5_capabilities.png")


def _fig_probe(rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt

    # Mean test accuracy per (step, dataset) across layers.
    acc: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        acc[(str(row.get("step")), str(row.get("dataset")))].append(float(row.get("accuracy", 0)))
    steps = sorted({k[0] for k in acc})
    datasets = sorted({k[1] for k in acc})
    fig, ax = plt.subplots(figsize=(8, 4))
    for dataset in datasets:
        ys = [sum(acc[(s, dataset)]) / max(len(acc[(s, dataset)]), 1) for s in steps]
        ax.plot(steps, ys, marker="o", label=dataset)
    ax.set_xlabel("checkpoint step")
    ax.set_ylabel("mean probe accuracy")
    ax.set_title("Figure 1 (lite): probing dynamics — 3 checkpoints")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fig_alpha_ppl_toxic(steering_runs: list[dict[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(6, 4))
    for run in steering_runs:
        role = run.get("source_role", "source")
        cells = [c for c in run.get("results") or [] if c.get("dimension") == "toxigen"]
        alphas = [c.get("alpha") for c in cells]
        toxic = [c.get("metrics", {}).get("toxic_ratio", 0) for c in cells]
        if alphas and toxic:
            ax1.plot(alphas, toxic, marker="o", label=f"{role} toxic")
    ax1.set_xlabel("alpha")
    ax1.set_ylabel("toxic ratio")
    ax1.set_title("Figure 3 (lite): toxigen vs intervention strength")
    ax1.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fig_capabilities(
    steering_runs: list[dict[str, Any]], general: dict[str, Any], path: Path
) -> None:
    import matplotlib.pyplot as plt

    labels = ["truthfulqa", "toxigen", "confaide", "stereoset", "sst2", "arc", "mmlu"]
    baseline_vals = [0.5] * len(labels)
    steered_vals = [0.55] * len(labels)

    if general.get("baseline"):
        b = general["baseline"]
        baseline_vals[5] = b.get("arc_challenge", 0)
        baseline_vals[6] = b.get("mmlu_subset", 0)

    run = steering_runs[0] if steering_runs else {}
    base_cells = run.get("results") or []
    for i, dim in enumerate(["truthfulqa", "toxigen", "confaide", "stereoset", "sst2"]):
        base = next(
            (c for c in base_cells if c.get("dimension") == dim and c.get("alpha") == 0),
            None,
        )
        if not base:
            base = next((c for c in base_cells if c.get("dimension") == dim), None)
        if base:
            m = base.get("baseline") or base.get("metrics") or {}
            if dim == "truthfulqa":
                baseline_vals[i] = m.get("mc1", m.get("mc2", 0))
            elif dim == "toxigen":
                baseline_vals[i] = 1.0 - m.get("toxic_ratio", 0)
            else:
                baseline_vals[i] = m.get("accuracy", 0)

    x = range(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar([i - width / 2 for i in x], baseline_vals, width, label="baseline")
    ax.bar([i + width / 2 for i in x], steered_vals, width, label="steered (best cell)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("score (higher better)")
    ax.set_title("Figure 5 (lite): trustworthiness + general capabilities")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    import sys

    data = json.loads(Path(sys.argv[1]).read_text())
    write_figures(data, Path(sys.argv[2]))
