#!/usr/bin/env python
"""Render item characteristic curves (ICCs) for the unidimensional IRT appendix.

Writes irt_icc.png next to this script. The left panel varies difficulty b at a
fixed discrimination a; the right panel varies discrimination a at a fixed
difficulty b. Nothing is read from disk; the curves are the 2PL logistic
function evaluated on a theta grid.
"""

import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PNG = os.path.join(HERE, "irt_icc.png")

# MPLCONFIGDIR must be set before matplotlib is imported so the font cache lands
# in a local, writable .mplcache directory.
os.environ.setdefault("MPLCONFIGDIR", os.path.join(HERE, ".mplcache"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def icc(theta, a, b):
    """2PL probability of a correct response."""
    return 1.0 / (1.0 + np.exp(-a * (theta - b)))


def main():
    theta = np.linspace(-4, 4, 400)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    ax = axes[0]
    a_fixed = 1.0
    for b, color in zip([-1.5, 0.0, 1.5], ["#1f77b4", "#2ca02c", "#d62728"], strict=True):
        ax.plot(theta, icc(theta, a_fixed, b), color=color, lw=2, label=f"b = {b:+.1f}")
        ax.plot(b, 0.5, marker="o", color=color, ms=6)
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.set_title(f"Difficulty b shifts the curve (a = {a_fixed:.1f})")
    ax.set_xlabel(r"latent ability $\theta$")
    ax.set_ylabel(r"P(correct $\mid \theta$)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right", frameon=False)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    b_fixed = 0.0
    for a, color in zip([0.4, 1.0, 2.5], ["#1f77b4", "#2ca02c", "#d62728"], strict=True):
        ax.plot(theta, icc(theta, a, b_fixed), color=color, lw=2, label=f"a = {a:.1f}")
    ax.axvline(b_fixed, color="gray", ls=":", lw=1)
    ax.plot(b_fixed, 0.5, marker="o", color="black", ms=6)
    ax.set_title(f"Discrimination a sets the slope (b = {b_fixed:.1f})")
    ax.set_xlabel(r"latent ability $\theta$")
    ax.set_ylabel(r"P(correct $\mid \theta$)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right", frameon=False)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
