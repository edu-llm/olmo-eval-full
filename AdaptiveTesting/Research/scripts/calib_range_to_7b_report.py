#!/usr/bin/env python3
"""Aggregate calib_range_to_7b results.csv into a matched-N r table, a comparison
figure (r for Config A vs B per benchmark), and a README with the one-line answer.

Plain style: title states the finding, short axis labels with units, no fabricated
numbers (everything is computed from results.csv)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
OUT_ROOT = REPO / "AdaptiveTesting/Research/01_MCQ_ATLAS/data/calib_range_to_7b"

CFG_LABEL = {"A": "A [0,5)B calib", "B": "B [3,6.5)B calib"}


def agg(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["benchmark", "config", "band", "se_target"], as_index=False).agg(
        r_mean=("corr", "mean"), r_sd=("corr", "std"),
        mae_mean=("mae", "mean"), items_mean=("mean_items", "mean"),
        n_calib=("n_calib", "first"), n_test=("n_test", "first"),
        n_seeds=("seed", "nunique"),
    )
    g["r_sd"] = g["r_sd"].fillna(0.0)
    return g


def _mpl():
    os.environ.setdefault("MPLCONFIGDIR", str(OUT_ROOT / ".mplcache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _finding(sub: pd.DataFrame) -> str:
    """Data-driven one-line title stating which pool recovers 7B accuracy better."""
    diffs = []
    for b in sorted(sub["benchmark"].unique()):
        a = sub[(sub["benchmark"] == b) & (sub["config"] == "A")]
        bb = sub[(sub["benchmark"] == b) & (sub["config"] == "B")]
        if len(a) and len(bb):
            diffs.append(float(bb["r_mean"].iloc[0]) - float(a["r_mean"].iloc[0]))
    if not diffs:
        return "Calibration range vs 7B-model recovery"
    avg = float(np.mean(diffs))
    if avg > 0.005:
        return f"In-range 3-7B calibration recovers 7B accuracy better (mean r gain {avg:+.02f})"
    if avg < -0.005:
        return f"0-5B calibration recovers 7B accuracy better (mean r gain {-avg:+.02f})"
    return "0-5B and 3-7B calibration recover 7B accuracy about equally"


def make_figure(g: pd.DataFrame, se: float) -> Path:
    sub = g[g["se_target"] == se].copy()
    benches = sorted(sub["benchmark"].unique())
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    x = np.arange(len(benches))
    w = 0.38
    for k, cfg in enumerate(["A", "B"]):
        rs, sds = [], []
        for b in benches:
            row = sub[(sub["benchmark"] == b) & (sub["config"] == cfg)]
            rs.append(float(row["r_mean"].iloc[0]) if len(row) else np.nan)
            sds.append(float(row["r_sd"].iloc[0]) if len(row) else 0.0)
        bars = ax.bar(x + (k - 0.5) * w, rs, w, yerr=sds, capsize=3,
                      label=CFG_LABEL[cfg], color=["#4c78a8", "#e45756"][k])
        for xi, ri in zip(x + (k - 0.5) * w, rs):
            if np.isfinite(ri):
                ax.text(xi, ri + 0.01, f"{ri:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(benches)
    ax.set_ylabel("Pearson r (pred vs actual acc)")
    ax.set_xlabel("benchmark")
    ax.set_ylim(0, 1.05)
    n_calib = int(sub["n_calib"].iloc[0])
    n_test = int(sub["n_test"].iloc[0])
    ax.set_title(f"{_finding(sub)} (SE<={se:g})\n"
                 f"matched N={n_calib} calib models, {n_test} held-out 7B test models")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    png = OUT_ROOT / "calib_range_to_7b_r_A_vs_B.png"
    fig.savefig(png, dpi=140)
    plt.close(fig)
    return png


def write_readme(g: pd.DataFrame, meta: pd.DataFrame) -> Path:
    lines: list[str] = []
    lines.append("# Calibration parameter range vs 7B-model recovery\n")
    lines.append(
        "Fixed TEST set = 7B models (params_b >= 6.5). Two calibration pools, matched N, "
        "same test set. Config A calibrates on [0,5)B (upward extrapolation); Config B on "
        "[3,7]B with the >=6.5 test models removed so its band is [3,6.5)B "
        "(in-range / interpolation). p-IRT + Fisher CAT recover full-benchmark accuracy.\n")

    lines.append("## Pool sizes (per benchmark, models present in cleaned matrix)\n")
    lines.append("| benchmark | 7B test (>=6.5) | pool A [0,5) | pool B [3,6.5) | matched N |")
    lines.append("|---|---|---|---|---|")
    for _, r in meta.iterrows():
        lines.append(
            f"| {r['bench']} | {int(r['n_test'])} | {int(r['n_pool_a'])} | "
            f"{int(r['n_pool_b'])} | {int(r['n_match'])} |")
    lines.append("")

    for se in sorted(g["se_target"].unique()):
        sub = g[g["se_target"] == se]
        lines.append(f"## Matched-N Pearson r (SE<={se:g})\n")
        lines.append("| benchmark | A r (mean+/-sd) | B r (mean+/-sd) | B - A | A MAE | B MAE |")
        lines.append("|---|---|---|---|---|---|")
        for b in sorted(sub["benchmark"].unique()):
            a = sub[(sub["benchmark"] == b) & (sub["config"] == "A")]
            bb = sub[(sub["benchmark"] == b) & (sub["config"] == "B")]
            if not len(a) or not len(bb):
                continue
            ar, asd = a["r_mean"].iloc[0], a["r_sd"].iloc[0]
            br, bsd = bb["r_mean"].iloc[0], bb["r_sd"].iloc[0]
            lines.append(
                f"| {b} | {ar:.3f}+/-{asd:.3f} | {br:.3f}+/-{bsd:.3f} | "
                f"{br - ar:+.3f} | {a['mae_mean'].iloc[0]:.3f} | {bb['mae_mean'].iloc[0]:.3f} |")
        lines.append("")

    # one-line answer from ifeval+math at SE0.3
    se = 0.3
    sub = g[g["se_target"] == se]
    diffs = []
    for b in ["ifeval", "math"]:
        a = sub[(sub["benchmark"] == b) & (sub["config"] == "A")]
        bb = sub[(sub["benchmark"] == b) & (sub["config"] == "B")]
        if len(a) and len(bb):
            diffs.append(br_diff := float(bb["r_mean"].iloc[0] - a["r_mean"].iloc[0]))
    lines.append("## Answer\n")
    if diffs:
        avg = float(np.mean(diffs))
        direction = "higher" if avg > 0 else "lower"
        lines.append(
            f"Calibrating on 3-7B vs 0-5B, tested on 7B models: at SE<=0.3, in-range "
            f"calibration (B) gives Pearson r that is on average {avg:+.3f} "
            f"({direction}) than upward extrapolation (A) across ifeval and math "
            f"(per-benchmark B-A: {', '.join(f'{d:+.3f}' for d in diffs)}). "
            f"See table above for exact per-benchmark values.\n")
    return _write(lines)


def _write(lines: list[str]) -> Path:
    path = OUT_ROOT / "README.md"
    path.write_text("\n".join(lines))
    return path


def main() -> None:
    df = pd.read_csv(OUT_ROOT / "results.csv")
    g = agg(df)
    g.to_csv(OUT_ROOT / "results_summary.csv", index=False)
    meta = _meta(df)
    png = make_figure(g, 0.3)
    readme = write_readme(g, meta)
    print(f"wrote {OUT_ROOT/'results_summary.csv'}")
    print(f"wrote {png}")
    print(f"wrote {readme}")
    print(g[g.se_target == 0.3].to_string(index=False))


def _meta(df: pd.DataFrame) -> pd.DataFrame:
    import calib_range_to_7b as exp
    params = exp.load_params()
    rows = []
    for b, gb in df.groupby("benchmark"):
        mat = exp.sw.build_master(b)
        test, pool_a, pool_b = exp.select_pools(list(mat.index), params)
        rows.append({
            "bench": b, "n_test": len(test),
            "n_pool_a": len(pool_a), "n_pool_b": len(pool_b),
            "n_match": int(gb["n_calib"].iloc[0]),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
