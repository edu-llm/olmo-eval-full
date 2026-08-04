#!/usr/bin/env python
"""Render the parameter-distribution figure from the live models_200.yaml roster.

Reads AdaptiveTesting/Inputs/Models/models_200.yaml (156 checkpoints) and writes
models_200_param_dist.png next to this script. Data is read as-is; this only
plots it.
"""
import os
from collections import Counter

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "../../../.."))
YAML = os.path.join(REPO, "AdaptiveTesting/Inputs/Models/models_200.yaml")
OUT_PNG = os.path.join(HERE, "models_200_param_dist.png")

os.environ.setdefault("MPLCONFIGDIR", os.path.join(HERE, ".mplcache"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    models = yaml.safe_load(open(YAML))["models"]
    params = np.array([m["params_b"] for m in models], float)
    orgs = [m["id"].split("/")[0] for m in models]

    edges = [0.2, 1, 2, 3, 4, 5, 6, 7.001]
    labels = ["0.2-1", "1-2", "2-3", "3-4", "4-5", "5-6", "6-7"]
    counts = np.histogram(params, bins=edges)[0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    ax = axes[0]
    ax.bar(labels, counts, color="#1f77b4", edgecolor="black")
    ax.set_xlabel("parameter count (billions)")
    ax.set_ylabel("number of models")
    ax.set_title("Sizes cluster at 1-2B (42) and 7B (52)")
    for i, c in enumerate(counts):
        ax.text(i, c + 0.5, str(int(c)), ha="center", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    ax2 = axes[1]
    top = Counter(orgs).most_common(15)
    ax2.barh([t[0] for t in reversed(top)], [t[1] for t in reversed(top)],
             color="#ff7f0e", edgecolor="black")
    ax2.set_xlabel("number of models")
    ax2.set_title("Top 15 organizations")
    ax2.grid(True, axis="x", alpha=0.3)

    fig.suptitle(
        f"156-model roster: {params.min():.2f}-{params.max():.1f}B, "
        f"mean {params.mean():.1f}B, median {np.median(params):.1f}B, "
        f"{len(set(orgs))} organizations",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130)
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
