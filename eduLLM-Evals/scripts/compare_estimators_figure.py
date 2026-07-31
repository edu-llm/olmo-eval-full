"""Composite one figure per item bank comparing three CAT final-ability estimators.

What this composites
--------------------
``scripts/regen_cat_figures.py`` was run three times per bank with *identical* item
selection, varying only ``--final-estimator``:

* ``online`` -- the production sequential one-step Newton/Laplace update per item.
* ``batch``  -- discard the accumulated theta, re-estimate by EAP over the administered
  items (order-invariant).
* ``mwle``   -- EAP as initialisation, then multidimensional Warm Weighted Likelihood
  Estimation, which also drops the prior's inward pull at the tails.

Each run wrote a ``cat_per_model.csv`` holding, per latent dimension ``<dim>``, the
full-bank reference ability ``theta_full_<dim>`` (EAP over all ~3400-4200 calibrated
items) and the 100-item CAT estimate ``theta_cat_<dim>``. This script reads those six
CSVs and lays the recovery scatters out as a grid -- **rows = latent dimensions, columns
= estimators** (Online Gaussian, Batch EAP, Batch EAP + MWLE, left to right).

Every panel in a row shares one set of x/y limits and is drawn square, so the dashed
``y = x`` line sits at 45 degrees and the three columns are visually comparable: points
hugging ``y = x`` mean faithful recovery, while a shallow cloud means the ability scale
has been shrunk toward the prior mean. Each panel is annotated with its Pearson r, the
best-fit slope, and the mean signed gap over the 12 lowest-ability models, where the
prior's inward pull bites hardest.

Read-only by design
-------------------
This script only *reads* the analysis outputs under ``regenerated_figures/``. It never
re-runs a CAT, never touches an existing figure, CSV or ``metrics.json``, and writes
exclusively into ``--out-dir`` (default ``regenerated_figures/estimator_comparison``).

Outputs (``--out-dir``)
-----------------------
* ``comparison_2_skills.png``  -- 2 rows (correctness, scaffolding) x 3 estimators
* ``comparison_3_skills.png``  -- 3 rows (content, diagnosis, scaffolding) x 3 estimators

Usage
-----
    python scripts/compare_estimators_figure.py
    python scripts/compare_estimators_figure.py --out-dir /tmp/cmp --dpi 160
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

FIG_ROOT = Path("regenerated_figures")

# (column label, path of the per-model CSV relative to FIG_ROOT, "<bank>" is substituted)
ESTIMATORS: tuple[tuple[str, str], ...] = (
    ("Online Gaussian", "<bank>/cat_per_model.csv"),
    ("Batch EAP", "batched_EAP/<bank>/cat_per_model.csv"),
    ("Batch EAP + MWLE", "batchedEAP_MWLE/<bank>/cat_per_model.csv"),
)

BANKS: tuple[str, ...] = ("2_skills", "3_skills")

# Models at the bottom of the ability range, where prior shrinkage is most visible.
LOW_N = 12

COLUMN_COLORS = ("#3b6ea5", "#c2703d", "#3f8f60")


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def resolve(path: Path) -> Path:
    """Interpret relative paths against the repo root, so cwd does not matter."""
    return path if path.is_absolute() else (ROOT / path).resolve()


def load_bank(bank: str) -> dict[str, pd.DataFrame]:
    """The three per-model tables for one bank, keyed by estimator label."""
    frames: dict[str, pd.DataFrame] = {}
    for label, template in ESTIMATORS:
        path = resolve(FIG_ROOT / template.replace("<bank>", bank))
        if not path.exists():
            raise SystemExit(f"missing input: {path}")
        frames[label] = pd.read_csv(path)
        print(f"  {label:<18} {path.relative_to(ROOT)}  ({len(frames[label])} models)")
    return frames


def dimensions(df: pd.DataFrame) -> list[str]:
    """Latent dimensions present, read off the ``theta_full_*`` columns rather than
    hardcoded, keeping the CSV's own column order."""
    dims = []
    for col in df.columns:
        m = re.fullmatch(r"theta_full_(.+)", col)
        if m and f"theta_cat_{m.group(1)}" in df.columns:
            dims.append(m.group(1))
    if not dims:
        raise SystemExit("no theta_full_<dim>/theta_cat_<dim> column pairs found")
    return dims


# ---------------------------------------------------------------------------
# per-panel statistics
# ---------------------------------------------------------------------------


def panel_stats(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Pearson r, best-fit slope, and the mean signed CAT-minus-full gap over the
    ``LOW_N`` models with the smallest reference ability."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    low = np.argsort(x, kind="stable")[:LOW_N]
    return {
        "n": int(x.size),
        "r": float(np.corrcoef(x, y)[0, 1]),
        "slope": float(np.polyfit(x, y, 1)[0]),
        "low_gap": float(np.mean(y[low] - x[low])),
    }


def shared_limits(series: list[tuple[np.ndarray, np.ndarray]], margin: float = 0.3
                  ) -> tuple[float, float]:
    """One square window covering every estimator for a dimension, so the three panels
    in a row can be compared by eye."""
    vals = np.concatenate([np.concatenate([x, y]) for x, y in series])
    vals = vals[np.isfinite(vals)]
    return float(vals.min() - margin), float(vals.max() + margin)


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------


def build_figure(bank: str, frames: dict[str, pd.DataFrame], out_path: Path,
                 dpi: int) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [label for label, _ in ESTIMATORS]
    dims = dimensions(frames[labels[0]])
    n_rows, n_cols = len(dims), len(labels)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.35 * n_cols, 3.65 * n_rows),
                             squeeze=False, constrained_layout=True)
    rows: list[dict] = []

    for i, dim in enumerate(dims):
        pairs = [(frames[lab][f"theta_full_{dim}"].to_numpy(float),
                  frames[lab][f"theta_cat_{dim}"].to_numpy(float)) for lab in labels]
        lo, hi = shared_limits(pairs)

        for j, (label, (x, y)) in enumerate(zip(labels, pairs)):
            st = panel_stats(x, y)
            rows.append({"bank": bank, "dim": dim, "estimator": label, **st})

            ax = axes[i][j]
            ax.plot([lo, hi], [lo, hi], ls="--", color="gray", lw=1, zorder=1)
            ax.scatter(x, y, s=22, alpha=0.75, color=COLUMN_COLORS[j % len(COLUMN_COLORS)],
                       edgecolor="k", linewidth=0.3, zorder=2)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal", adjustable="box")
            ax.tick_params(labelsize=8)
            ax.text(0.04, 0.96,
                    f"r = {st['r']:.3f}\nslope = {st['slope']:.3f}\n"
                    f"low-{LOW_N} gap = {st['low_gap']:+.3f}",
                    transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
                    bbox=dict(boxstyle="round,pad=0.32", facecolor="white",
                              edgecolor="0.75", alpha=0.88), zorder=3)

            if i == 0:
                ax.set_title(label, fontsize=11.5, fontweight="bold")
            if i == n_rows - 1:
                ax.set_xlabel("full-bank EAP ability", fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{dim}\nCAT ability", fontsize=9.5, fontweight="bold")

    n_models = len(frames[labels[0]])
    fig.suptitle(
        f"TutorBench CAT ability recovery vs full-bank EAP -- {bank.replace('_', '-')} bank\n"
        f"CAT estimate (y) against full-bank reference ability (x), {n_models} models per "
        f"panel; dashed line is y = x.\nSlope 1.0 on y = x is perfect recovery; "
        f"slope < 1 means the ability scale is compressed toward zero.",
        fontsize=11.5)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return rows


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, default=Path("regenerated_figures/estimator_comparison"),
                   help="where the comparison PNGs are written (created if absent).")
    p.add_argument("--dpi", type=int, default=130, help="raster resolution (default 130).")
    args = p.parse_args()

    out_dir = resolve(args.out_dir)
    all_rows: list[dict] = []

    for bank in BANKS:
        print("=" * 72)
        print(f"estimator comparison :: bank={bank}")
        print("=" * 72)
        frames = load_bank(bank)

        # The reference axis is the same full-bank EAP in all three runs; a drift here
        # would mean the columns are not measuring recovery against a common truth.
        labels = [label for label, _ in ESTIMATORS]
        for dim in dimensions(frames[labels[0]]):
            ref = np.stack([frames[lab][f"theta_full_{dim}"].to_numpy(float)
                            for lab in labels])
            drift = float(np.nanmax(np.abs(ref - ref[0])))
            if drift > 1e-6:
                print(f"  WARNING: theta_full_{dim} differs across estimators "
                      f"(max |diff| = {drift:.3g})")

        out_path = out_dir / f"comparison_{bank}.png"
        rows = build_figure(bank, frames, out_path, args.dpi)
        all_rows.extend(rows)
        print(f"  wrote {out_path}")

    print()
    print(f"{'bank':<9} {'dimension':<13} {'estimator':<18} {'slope':>7} {'r':>7} "
          f"{'low-' + str(LOW_N) + ' gap':>11}")
    for r in all_rows:
        print(f"{r['bank']:<9} {r['dim']:<13} {r['estimator']:<18} "
              f"{r['slope']:>7.3f} {r['r']:>7.3f} {r['low_gap']:>+11.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
