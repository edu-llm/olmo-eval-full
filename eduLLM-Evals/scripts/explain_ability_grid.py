"""Visual: what the "ability grid" is.

Calibration never assigns a model a single ability. It evaluates every model against a
fixed set of candidate ability values -- the quadrature grid -- and carries a probability
weight for each. This script draws that grid straight out of the real calibration helpers
(``scripts/calibrate_mirt.build_grid`` / ``base_log_weights`` / ``prior_log_weights``):

  panel 1: the 7 candidate values for a single skill, with their prior plausibility
  panel 2: the 49-point lattice those become for two skills
  panel 3: what changes at 5 / 7 / 21 nodes, and where the outermost node sits

Read-only with respect to the fitter: imports it, never modifies it.

Usage
-----
    python scripts/explain_ability_grid.py
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cm = _load("calibrate_mirt", ROOT / "scripts" / "calibrate_mirt.py")


def grid_and_weights(n_dims: int, nodes: int):
    g = cm.build_grid(n_dims, nodes)
    base = cm.base_log_weights(n_dims, nodes)
    lw = cm.prior_log_weights(g, base, np.eye(n_dims))
    return g, np.exp(lw)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path,
                   default=ROOT / "regenerated_figures" / "twopl_explainer")
    p.add_argument("--nodes", type=int, default=7)
    args = p.parse_args()

    g1, w1 = grid_and_weights(1, args.nodes)
    x1 = g1[:, 0]
    g2, w2 = grid_and_weights(2, args.nodes)

    print(f"{args.nodes} nodes, 1 skill: positions {np.round(x1, 4)}")
    print(f"  prior weights {np.round(w1, 6)}  (sum {w1.sum():.6f})")
    print(f"{args.nodes} nodes, 2 skills: {g2.shape[0]} lattice points, "
          f"weights sum {w2.sum():.6f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))

    # --- panel 1: one skill -------------------------------------------------
    ax = axes[0]
    xs = np.linspace(-4.6, 4.6, 400)
    ax.plot(xs, np.exp(-0.5 * xs**2) / np.sqrt(2 * np.pi), color="0.75", lw=1.4,
            label="assumed ability distribution")
    ax.vlines(x1, 0, w1, color="crimson", lw=2.2, zorder=3)
    ax.plot(x1, w1, "o", color="crimson", ms=7, zorder=4,
            label=f"the {args.nodes} candidate values")
    for xi, wi in zip(x1, w1):
        ax.annotate(f"{xi:+.2f}", (xi, wi), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7.5)
    ax.set_title(f"1 skill: {args.nodes} candidate ability values\n"
                 "height = how plausible each is before seeing data", fontsize=10)
    ax.set_xlabel("ability")
    ax.set_ylabel("prior weight")
    ax.set_ylim(0, max(w1.max(), 0.42) * 1.28)
    ax.legend(fontsize=7.5, loc="upper left")

    # --- panel 2: two skills ------------------------------------------------
    ax = axes[1]
    sc = ax.scatter(g2[:, 0], g2[:, 1], s=8 + 2600 * w2, c=w2,
                    cmap="viridis", edgecolor="0.3", linewidth=0.4, zorder=3)
    ax.set_title(f"2 skills: every combination, {g2.shape[0]} lattice points\n"
                 "marker size and colour = prior weight", fontsize=10)
    ax.set_xlabel("candidate correctness ability")
    ax.set_ylabel("candidate scaffolding ability")
    ax.axhline(0, color="0.85", lw=0.8, zorder=1)
    ax.axvline(0, color="0.85", lw=0.8, zorder=1)
    ax.set_aspect("equal")
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03).set_label("prior weight", fontsize=8)

    # --- panel 3: resolution ------------------------------------------------
    ax = axes[2]
    for row, n in enumerate([5, args.nodes, 21]):
        gn, _ = grid_and_weights(1, n)
        xn = gn[:, 0]
        ax.plot(xn, np.full_like(xn, row), "o", ms=5.5, color="crimson")
        ax.plot([xn.min(), xn.max()], [row, row], color="0.8", lw=1, zorder=1)
        gaps = np.diff(xn)
        ax.annotate(f"{n} nodes   outermost {xn.max():+.2f}   "
                    f"spacing {gaps.min():.2f}-{gaps.max():.2f}",
                    (0, row), textcoords="offset points", xytext=(0, 13),
                    ha="center", fontsize=8)
    ax.set_yticks([])
    ax.set_ylim(-0.6, 2.75)
    ax.set_xlabel("ability")
    ax.set_title("Resolution is capped by the grid\n"
                 "no estimate can land past the outermost node", fontsize=10)

    fig.tight_layout()
    out = args.out_dir / "ability_grid_explainer.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
