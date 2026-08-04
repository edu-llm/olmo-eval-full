#!/usr/bin/env python3
"""Full-pool ATLAS ARC: random train-size sweep + reverse-engineering oracle greedy selection.

This mirrors the OpenLM `oracle_subset_selection.py` experiment, but on the BALANCED,
full-range ATLAS ARC calibration pool (0.5B-59B, Gaussian-sampled over ability, ~3.7k
models -> ~844 items; this is the pool behind ATLAS's published ARC r~0.83) instead of
the OpenLM per-question data. We deliberately do NOT use the size-skewed 0.5-7B pool
(which tops out near r~0.59).

Unlike the sibling `atlas_arc_balanced_sweep.py` (which capped the pool at 500), here the
reference IS the entire ~3.7k-model train pool: the full-pool calibration is fit on all of
it, random subsets are drawn from it, and greedy candidates come from it. So random N=500
is a genuine 500-of-3.7k subset with seed-to-seed variance, not the whole pool.

Two questions, both at calibration-pool steps of 50:
  (a) random train-size sweep  -- how the NUMBER of calibration models affects held-out
      accuracy recovery (r vs N, 5 seeds, mean +/- sd);
  (b) reverse-engineering oracle greedy subset selection -- calibrate the whole pool,
      read each model's fitted latent ability (EAP theta), then greedily add the model
      that most improves held-out r (oracle_greedy). This uses post-hoc information, so it
      is an UPPER BOUND, not a deployable rule.

We reuse the validated calibration + CAT machinery unchanged:
  * openlm_trainsize_sweep.py: prepare_step / fit_and_link (chunked R mirt 3PL +
    mean-sigma linking with polarity flip + a>0 item filter) and diagnose (SE<=0.3
    EAP/Fisher adaptive CAT, MIN_ITEMS=8).
  * oracle_subset_selection.py: draw_pool, fit_subset, full_pool_calibration,
    latent_groups, select_random, oracle_greedy_build.
The ONLY change is the data source: we load the ARC train matrix as the model x item
frame and pass it where build_master's output is used, and we keep the pre-split 417 ARC
test models fixed as the held-out set for every strategy and N.

Pool: --pool-cap is set very large (100000) so draw_pool returns the WHOLE ~3.7k train
pool. That full pool IS "the full pool" for the reference r.

Outputs -> AdaptiveTesting/Research/01_MCQ_ATLAS/data/atlas_arc_fullpool_sweep/:
  results.csv            strategy, N, seed, se_target, r, mae, items, n_bank_items,
                         n_pool, n_test, calib_s
  latent_groups_arc.csv  group id, theta lo/hi bound, size, mean theta
  selected_subsets.csv   strategy, N, model, latent_group, theta (greedy picks per N)
  oracle_curve_arc.png   r vs N: random band, oracle_greedy, refs
  latent_axis_arc.png    selected subset positions along the latent-ability axis
  README.md              full-pool ref r, random r-vs-N curve, greedy-vs-random verdict,
                         contrast with the skewed 0.5-7B pool (~0.59) and ATLAS ~0.83

Usage (CPU-local, uv):
  uv run python AdaptiveTesting/Research/scripts/atlas_arc_fullpool_sweep.py --phase core \
      --pool-cap 100000 --n-list 50,100,150,200,250,300,350,400,450,500 \
      --seeds 0,1,2,3,4 --strategies random,oracle_greedy --jobs 12 --workers 1
  uv run python AdaptiveTesting/Research/scripts/atlas_arc_fullpool_sweep.py --phase greedy \
      --pool-cap 100000 --greedy-max-n 500 --greedy-batch 16 --jobs 12 --workers 1
  uv run python AdaptiveTesting/Research/scripts/atlas_arc_fullpool_sweep.py --phase plot
  uv run python AdaptiveTesting/Research/scripts/atlas_arc_fullpool_sweep.py --timing-probe \
      --pool-cap 100000 --jobs 12 --workers 1
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import openlm_trainsize_sweep as sweep  # noqa: E402  validated 3PL calibration + CAT
import oracle_subset_selection as oss  # noqa: E402  validated selection strategies

REPO = SCRIPTS.parents[2]
ATLAS = REPO / "AdaptiveTesting/Inputs/ATLAS"
TRAIN_CSV = ATLAS / "data/gaussian_sampled_arc_response_matrix_train.csv"
TEST_CSV = ATLAS / "data/gaussian_sampled_arc_response_matrix_test.csv"
OUT_ROOT = REPO / "AdaptiveTesting/Research/01_MCQ_ATLAS/data/atlas_arc_fullpool_sweep"
BENCH = "arc"  # single tag used for the reused per-bench work dirs / prints

# Context reference lines for the curve figure (see task + sibling experiments).
SKEWED_0P5_7B_R = 0.59  # size-skewed 0.5-7B ARC pool topped near here
ATLAS_FULL_ARC_R = 0.83  # ATLAS published full ARC p-IRT recovery

FIELDS = [
    "strategy",
    "N",
    "seed",
    "se_target",
    "r",
    "mae",
    "items",
    "n_bank_items",
    "n_pool",
    "n_test",
    "calib_s",
]
COLORS = {"random": "#7f7f7f", "oracle_stratified": "#1f77b4", "oracle_greedy": "#d62728"}
LABELS = {
    "random": "random (mean +/- sd)",
    "oracle_stratified": "oracle: latent-group stratified",
    "oracle_greedy": "oracle: greedy on held-out r",
}


# ---------------------------------------------------------------------------
# Data: load the balanced ARC matrices (model x item), keep the 417 test fixed
# ---------------------------------------------------------------------------
def load_arc_matrix(path: Path) -> pd.DataFrame:
    """Model x item 0/1 frame. Drops a trailing avg_score column if present."""
    df = pd.read_csv(path)
    df = df.rename(columns={df.columns[0]: "model"}).set_index("model")
    for extra in ("avg_score", "score", "accuracy"):
        if extra in df.columns:
            df = df.drop(columns=[extra])
    df.columns = [int(c) for c in df.columns]
    df = df[~df.index.duplicated(keep="first")]
    return df.astype(int)


def load_data(args) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Combined mat (train pool + fixed test) with a shared item set, plus the model lists.

    prepare_step slices mat.loc[train_models] and mat.loc[test_models, kept], so mat must
    contain every train-pool and test model on one shared column set with a unique index.
    """
    train_df = load_arc_matrix(TRAIN_CSV)
    test_df = load_arc_matrix(TEST_CSV)
    common = sorted(set(train_df.columns) & set(test_df.columns))
    if len(common) != train_df.shape[1] or len(common) != test_df.shape[1]:
        print(
            f"[arc] item columns differ: train={train_df.shape[1]} test={test_df.shape[1]} "
            f"common={len(common)} (using common)",
            flush=True,
        )
    train_df, test_df = train_df[common], test_df[common]

    overlap = set(train_df.index) & set(test_df.index)
    if overlap:
        print(
            f"[arc] {len(overlap)} models in both train+test; dropping them from train pool",
            flush=True,
        )
        train_df = train_df.drop(index=list(overlap))

    train_pool = list(train_df.index)
    test_models = list(test_df.index)
    if args.n_test and args.n_test < len(test_models):
        rng = np.random.default_rng(args.test_seed)
        idx = sorted(rng.choice(len(test_models), size=args.n_test, replace=False))
        test_models = [test_models[i] for i in idx]
        test_df = test_df.loc[test_models]

    mat = pd.concat([train_df, test_df])
    print(
        f"[arc] train_pool={len(train_pool)} models  test={len(test_models)} models  "
        f"items={len(common)}",
        flush=True,
    )
    return mat, train_pool, test_models


# ---------------------------------------------------------------------------
# Results CSV (merge/incremental so partial progress survives)
# ---------------------------------------------------------------------------
def _key(r: dict) -> tuple:
    return (str(r["strategy"]), int(r["N"]), int(r["seed"]), float(r["se_target"]))


def write_results(rows: list[dict]) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    path = OUT_ROOT / "results.csv"
    existing = pd.read_csv(path).to_dict("records") if path.exists() else []
    merged = {_key(r): r for r in existing}
    for r in rows:
        merged[_key(r)] = r
    out = sorted(
        merged.values(),
        key=lambda r: (str(r["strategy"]), float(r["se_target"]), int(r["N"]), int(r["seed"])),
    )
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in out:
            w.writerow({k: r.get(k) for k in FIELDS})


def _row(strategy, n, seed, se, m, n_pool, n_test, calib_s) -> dict:
    return {
        "strategy": strategy,
        "N": n,
        "seed": seed,
        "se_target": se,
        "r": m["corr"],
        "mae": m["mae"],
        "items": m["mean_items"],
        "n_bank_items": m["n_bank_items"],
        "n_pool": n_pool,
        "n_test": n_test,
        "calib_s": calib_s,
    }


def append_subsets(strategy: str, n: int, subset: list[str], theta_by_model, group_of) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    path = OUT_ROOT / "selected_subsets.csv"
    fields = ["strategy", "N", "model", "latent_group", "theta"]
    existing = pd.read_csv(path).to_dict("records") if path.exists() else []
    keep = [r for r in existing if not (r["strategy"] == strategy and int(r["N"]) == n)]
    for m in sorted(subset, key=lambda m: theta_by_model.get(m, 0.0)):
        keep.append(
            {
                "strategy": strategy,
                "N": n,
                "model": m,
                "latent_group": group_of.get(m, ""),
                "theta": round(theta_by_model.get(m, float("nan")), 4),
            }
        )
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(keep)


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------
def phase_core(args, mat, pool, test_models, grid, ses, fit_r, link_r) -> None:
    """full_pool reference (whole ~3.7k pool) + random (5 seeds) over the N grid.

    oracle_stratified is only run if explicitly requested in --strategies; the default
    fullpool config uses random,oracle_greedy (greedy is built in phase_greedy)."""
    se0 = ses[0]
    rows: list[dict] = []

    full_metrics, theta_by_model, full_s = oss.full_pool_calibration(
        args, BENCH, mat, test_models, pool, fit_r, link_r, ses
    )
    rows.append(
        _row("full_pool", len(pool), 0, se0, full_metrics[se0], len(pool), len(test_models), full_s)
    )
    write_results(rows)

    group_of, bounds = oss.latent_groups(theta_by_model, args.groups)
    _write_groups(bounds)

    tasks: list[tuple[str, int, int]] = []
    for n in grid:
        if "oracle_stratified" in args.strategies:
            tasks.append(("oracle_stratified", 0, n))
        if "random" in args.strategies:
            for seed in args.seeds:
                tasks.append(("random", seed, n))

    def run_task(strat: str, seed: int, n: int):
        sd = _work(args) / f"{strat}_s{seed}_n{n}"
        if strat == "random":
            subset = oss.select_random(pool, n, seed)
        else:
            subset = oss.select_oracle_stratified(theta_by_model, group_of, pool, args.groups, n)
        t0 = time.time()
        m = oss.fit_subset(args, mat, test_models, sd, fit_r, link_r, subset, ses)
        return _row(
            strat, n, seed, se0, m[se0], len(pool), len(test_models), round(time.time() - t0, 1)
        ), subset

    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(run_task, s, sd, n): (s, sd, n) for (s, sd, n) in tasks}
        for fut in as_completed(futs):
            s, sd, n = futs[fut]
            try:
                row, subset = fut.result()
                rows.append(row)
                if s in ("oracle_stratified",):
                    append_subsets(s, n, subset, theta_by_model, group_of)
                print(
                    f"[arc] {s:<18} seed={sd} N={n:<4} bank={row['n_bank_items']:<4} "
                    f"SE{se0} r={row['r']} items={row['items']} ({row['calib_s']}s)",
                    flush=True,
                )
            except Exception as exc:  # keep sweeping past a bad fit
                print(f"[arc] {s} seed={sd} N={n} FAILED: {exc}", flush=True)
                rows.append(
                    _row(
                        s,
                        n,
                        sd,
                        se0,
                        {
                            "corr": float("nan"),
                            "mae": float("nan"),
                            "mean_items": float("nan"),
                            "n_bank_items": 0,
                        },
                        len(pool),
                        len(test_models),
                        0.0,
                    )
                )
            write_results(rows)
    print(f"[arc] core phase done -> {OUT_ROOT / 'results.csv'}", flush=True)


def phase_greedy(args, mat, pool, test_models, grid, ses, fit_r, link_r) -> None:
    """Forward greedy maximizing held-out r, seeded at the theta extremes (cap N)."""
    theta_by_model, group_of = oss._reload_theta_groups(BENCH, args)
    if not theta_by_model:
        raise RuntimeError("full_pool calibration missing; run --phase core first")
    print(
        f"[arc] building oracle_greedy up to N={args.greedy_max_n} "
        f"(batch={args.greedy_batch}, jobs={args.jobs}); upper bound, peeks at held-out r",
        flush=True,
    )
    grows, greedy_subsets = oss.oracle_greedy_build(
        args, BENCH, mat, test_models, pool, theta_by_model, fit_r, link_r, ses, grid
    )
    rows = [
        _row(
            r["strategy"],
            r["N"],
            r["seed"],
            r["se_target"],
            {
                "corr": r["r"],
                "mae": r["mae"],
                "mean_items": r["items"],
                "n_bank_items": r["n_bank_items"],
            },
            r["n_pool"],
            r["n_test"],
            r["calib_s"],
        )
        for r in grows
    ]
    write_results(rows)
    for n, subset in greedy_subsets.items():
        append_subsets("oracle_greedy", n, subset, theta_by_model, group_of)
    print(f"[arc] greedy phase done -> {OUT_ROOT / 'results.csv'}", flush=True)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _work(args) -> Path:
    return Path(args.work_root) / BENCH


def _write_groups(bounds: list[dict]) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUT_ROOT / "latent_groups_arc.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["group", "theta_lo", "theta_hi", "size", "mean_theta"])
        w.writeheader()
        w.writerows(bounds)


def timing_probe(args, mat, pool, test_models, ses, fit_r, link_r) -> None:
    """Measure per-fit wall time for a small and the full calibration to size the greedy scope."""
    se0 = ses[0]
    for n in (100, len(pool)):
        subset = oss.select_random(pool, n, 0)
        sd = _work(args) / f"_probe_n{n}"
        t0 = time.time()
        m = oss.fit_subset(args, mat, test_models, sd, fit_r, link_r, subset, ses)
        dt = time.time() - t0
        print(
            f"[probe] N={n:<4} bank={m[se0]['n_bank_items']:<4} r={m[se0]['corr']} "
            f"items={m[se0]['mean_items']} mae={m[se0]['mae']}  wall={dt:.1f}s "
            f"(jobs={args.jobs} workers={args.workers})",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _mpl():
    os.environ.setdefault("MPLCONFIGDIR", str(OUT_ROOT / ".mplcache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_curve(df: pd.DataFrame, se: float, ref_r: float, n_pool: int, path: Path) -> None:
    plt = _mpl()
    gb = df[(df["se_target"] == se) & np.isfinite(df["r"])]
    fig, ax = plt.subplots(figsize=(8.4, 5.4))

    rnd = gb[gb["strategy"] == "random"].groupby("N")["r"].agg(["mean", "std"]).reset_index()
    rnd = rnd.sort_values("N")
    if len(rnd):
        ax.plot(
            rnd["N"], rnd["mean"], "o-", color=COLORS["random"], label=LABELS["random"], zorder=3
        )
        sd = rnd["std"].fillna(0.0)
        ax.fill_between(
            rnd["N"],
            rnd["mean"] - sd,
            rnd["mean"] + sd,
            color=COLORS["random"],
            alpha=0.2,
            zorder=1,
        )
    for strat, mk in (("oracle_stratified", "s-"), ("oracle_greedy", "^-")):
        ds = gb[gb["strategy"] == strat].groupby("N")["r"].mean().reset_index().sort_values("N")
        if len(ds):
            ax.plot(ds["N"], ds["r"], mk, color=COLORS[strat], label=LABELS[strat], zorder=4)

    ax.axhline(
        ref_r,
        color="black",
        ls="-",
        lw=1.3,
        label=f"full-pool reference r={ref_r:.3f} (N={n_pool})",
    )
    ax.axhline(
        ATLAS_FULL_ARC_R,
        color="#2ca02c",
        ls=":",
        lw=1.1,
        label=f"ATLAS full ARC ~{ATLAS_FULL_ARC_R:.2f}",
    )
    ax.axhline(
        SKEWED_0P5_7B_R,
        color="#9467bd",
        ls=":",
        lw=1.1,
        label=f"skewed 0.5-7B pool topped ~{SKEWED_0P5_7B_R:.2f}",
    )
    ax.set_xlabel("number of calibration models (N)")
    ax.set_ylabel("Pearson r (p-IRT predicted vs actual accuracy)")
    ax.set_title(
        f"Full-pool ATLAS ARC: held-out r vs # calibration models (SE<={se:g})\n"
        f"reference = full {n_pool}-model train pool; N grid 50-500 step 50; "
        f"{int(df['n_test'].dropna().iloc[0])} fixed test models",
        fontsize=11,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_latent_axis(theta_by_model, group_of, bounds, subsets_df, path: Path, k: int) -> None:
    plt = _mpl()
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(9.4, 4.0))
    models = list(theta_by_model)
    theta = np.array([theta_by_model[m] for m in models], float)
    gid = np.array([group_of[m] for m in models], int)
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, k))
    for g in range(k):
        sel = theta[gid == g]
        ax.scatter(
            sel,
            rng.uniform(-0.06, 0.06, len(sel)),
            s=12,
            color=cmap[g],
            alpha=0.35,
            label=f"group {g} (n={len(sel)})",
        )
    for row in bounds[:-1]:
        ax.axvline(row["theta_hi"], color="gray", ls=":", lw=0.8)

    lanes = {"oracle_stratified": 0.32, "oracle_greedy": 0.52}
    markers = {"oracle_stratified": "v", "oracle_greedy": "D"}
    for strat, y in lanes.items():
        sd = subsets_df[subsets_df["strategy"] == strat] if len(subsets_df) else subsets_df
        if not len(sd):
            continue
        n_pick = int(sd["N"].min())  # smallest recorded oracle subset = the interesting one
        sd = sd[sd["N"] == n_pick]
        ax.scatter(
            sd["theta"],
            np.full(len(sd), y),
            s=70,
            marker=markers[strat],
            color="none",
            edgecolors="black",
            linewidths=0.9,
            zorder=5,
        )
        ax.scatter(
            sd["theta"],
            np.full(len(sd), y),
            s=30,
            marker=markers[strat],
            c=[cmap[int(g)] for g in sd["latent_group"]],
            zorder=6,
        )
        ax.text(
            theta.min(), y + 0.07, f"{LABELS[strat].split(':')[0]} subset (N={n_pick})", fontsize=8
        )

    ax.set_yticks([])
    ax.set_ylim(-0.2, 0.75)
    ax.set_xlabel("oracle latent ability theta (full-pool EAP)")
    ax.set_title(
        "Balanced ATLAS ARC: selected subset positions along the latent-ability axis", fontsize=11
    )
    ax.legend(loc="lower right", fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Analysis + README
# ---------------------------------------------------------------------------
def _curve(df: pd.DataFrame, strat: str, se: float) -> pd.DataFrame:
    d = df[(df["strategy"] == strat) & (df["se_target"] == se) & np.isfinite(df["r"])]
    return d.groupby("N")["r"].agg(["mean", "std"]).reset_index().sort_values("N")


def _plateau_note(rnd: pd.DataFrame) -> str:
    """Does the random mean curve keep rising or plateau? Report where the per-step gain dies."""
    if len(rnd) < 2:
        return "insufficient random points to assess a plateau."
    xs = rnd["N"].to_numpy(float)
    ys = rnd["mean"].to_numpy(float)
    deltas = np.diff(ys)
    steps = xs[1:]
    incs = np.diff(xs)
    parts = [f"+{int(incs[i])}@N={int(steps[i])}:{deltas[i]:+.3f}" for i in range(len(deltas))]
    plateau_n = None
    for i in range(len(deltas)):
        if abs(deltas[i]) < 0.01:
            plateau_n = int(steps[i])
            break
    tail = (
        f"random r plateaus (per-step gain < 0.01) by N~{plateau_n}."
        if plateau_n
        else "random r is still rising at the top of the grid (no clear plateau)."
    )
    return "per-step gains [" + ", ".join(parts) + "]. " + tail


def write_readme(df: pd.DataFrame, se: float, args) -> None:
    full = df[(df["strategy"] == "full_pool") & (df["se_target"] == se)]
    ref_r = float(full["r"].iloc[0]) if len(full) else float("nan")
    n_pool = int(full["n_pool"].iloc[0]) if len(full) else args.pool_cap
    n_test = int(full["n_test"].iloc[0]) if len(full) else 0
    rnd = _curve(df, "random", se)
    greedy = _curve(df, "oracle_greedy", se)
    n_seeds = len(args.seeds)

    def tbl(cv: pd.DataFrame, with_sd: bool) -> list[str]:
        out = []
        for _, r in cv.iterrows():
            if with_sd and not np.isnan(r["std"]):
                out.append(f"| {int(r['N'])} | {r['mean']:.4f} +/- {r['std']:.4f} |")
            else:
                out.append(f"| {int(r['N'])} | {r['mean']:.4f} |")
        return out

    L = [
        "# Full-pool ATLAS ARC: random train-size sweep + reverse-engineering oracle greedy",
        "",
        "## What this is",
        "",
        "Two questions on the BALANCED, full-range ATLAS ARC calibration pool (0.5B-59B,",
        "Gaussian-sampled over ability; the pool behind ATLAS's published ARC r~0.83), at",
        "calibration-pool steps of 50, using the ENTIRE ~3.7k-model train pool as the",
        "reference (no 500-model cap):",
        "",
        "1. **random train-size sweep** -- how the NUMBER of calibration models affects",
        f"   held-out accuracy recovery (r vs N, {n_seeds} seeds, mean +/- sd).",
        "2. **reverse-engineering oracle greedy subset selection** -- calibrate the whole",
        "   pool, read each model's fitted latent ability (EAP theta), then greedily add the",
        "   model that most improves held-out r (`oracle_greedy`). This uses post-hoc",
        "   information, so it is an UPPER BOUND, not a deployable rule.",
        "",
        "This deliberately uses the balanced pool and NOT the size-skewed 0.5-7B ARC pool",
        f"(~95% 7B), which topped out near r~{SKEWED_0P5_7B_R:.2f}.",
        "",
        "## Method (reused, unchanged)",
        "",
        "Same validated 3PL pipeline as `openlm_trainsize_sweep.py` (chunked R mirt 3PL,",
        "mean-sigma linking with polarity flip, a>0 item filter) + `oracle_subset_selection.py`",
        "selection strategies + SE<=0.3 EAP/Fisher adaptive CAT (MIN_ITEMS=8). Only the data",
        "source changes: the ARC train matrix is the model x item frame, and the pre-split",
        f"{n_test} ARC test models are the FIXED held-out set for every strategy and N.",
        "",
        f"The reference is the ENTIRE {n_pool}-model train pool (no cap): the full-pool",
        "calibration is fit on all of it, random subsets are drawn from it, and greedy",
        f"candidates come from it. Latent groups are {args.groups} quantile bins of the",
        f"full-pool EAP theta (`latent_groups_arc.csv`). Reported at SE<={se:g}.",
        "",
        "## Full-pool reference",
        "",
        f"- Pool used: {n_pool} models (the whole train pool). "
        f"**Full-pool reference r = {ref_r:.4f}** (SE<={se:g}), "
        f"MAE = {float(full['mae'].iloc[0]):.4f}, bank = {int(full['n_bank_items'].iloc[0])} items."
        if len(full)
        else "- (full_pool row missing)",
        f"- Contrast: ATLAS full ARC recovers r~{ATLAS_FULL_ARC_R:.2f}; the size-skewed 0.5-7B "
        f"pool topped near r~{SKEWED_0P5_7B_R:.2f}.",
        "",
        "## (a) random correlation-vs-N",
        "",
        f"| N | r (mean +/- sd over {n_seeds} seeds) |",
        "|---|---|",
        *tbl(rnd, True),
        "",
        f"Shape: {_plateau_note(rnd)}",
        "",
        "## (b) reverse-engineering oracle greedy",
        "",
        "### oracle_greedy (forward greedy on held-out r)",
        "",
        "| N | r |",
        "|---|---|",
        *tbl(greedy, False),
        "",
        "## Takeaway",
        "",
        _verdict(ref_r, rnd, greedy, n_pool),
        "",
        "## Outputs",
        "",
        "- `results.csv`: strategy, N, seed, se_target, r, mae, items, n_bank_items, n_pool, "
        "n_test, calib_s.",
        "- `latent_groups_arc.csv`: group id, theta bounds, size, mean theta.",
        "- `selected_subsets.csv`: strategy, N, model, latent_group, theta (greedy picks per N).",
        "- `oracle_curve_arc.png`: r vs N (random band, greedy line, reference lines).",
        "- `latent_axis_arc.png`: selected subset positions along the latent-ability axis.",
        "",
        "## Caveats",
        "",
        "- Oracle/upper-bound only: `oracle_greedy` peeks at the held-out accuracies to pick",
        "  each next model, so it is NOT a deployable rule.",
        "- 3PL on very small model subsets is unstable (many items go constant / a<=0 and are",
        "  dropped, shrinking the bank); treat small-N points as noisy. See `n_bank_items`.",
        f"- Random subsets are genuine {n_seeds}-seed draws from the full {n_pool}-model pool, so",
        "  N=500 has seed-to-seed variance (it is a 500-of-pool subset, not the whole pool).",
    ]
    (OUT_ROOT / "README.md").write_text("\n".join(str(x) for x in L))


def _verdict(ref_r, rnd, greedy, n_pool) -> str:
    if np.isnan(ref_r) or not len(rnd):
        return "Results incomplete."
    top_rand = rnd["mean"].iloc[-1]
    parts = [
        f"On the balanced full-range ARC pool the full {n_pool}-model calibration recovers "
        f"held-out accuracy at r={ref_r:.3f}, versus ATLAS's published ~{ATLAS_FULL_ARC_R:.2f} "
        f"and the size-skewed 0.5-7B pool's ~{SKEWED_0P5_7B_R:.2f}. So a balanced full-range "
        "pool recovers much better than the skewed 0.5-7B pool."
    ]
    lo = rnd["mean"].iloc[0]
    parts.append(
        f"Random sweep: r rises from {lo:.3f} at N={int(rnd['N'].iloc[0])} to {top_rand:.3f} at "
        f"N={int(rnd['N'].iloc[-1])}."
    )

    # does greedy reach the reference plateau with fewer models than random?
    def first_within(cv, tol):
        hit = cv[cv["mean"] >= ref_r - tol]
        return int(hit["N"].iloc[0]) if len(hit) else None

    if len(greedy):
        n02 = first_within(greedy, 0.02)
        rn02 = first_within(rnd, 0.02)
        if n02 and rn02:
            parts.append(
                f"oracle_greedy reaches within 0.02 of the reference at N={n02} "
                f"(random needs N~{rn02})."
            )
        elif n02:
            parts.append(
                f"oracle_greedy reaches within 0.02 of the reference at N={n02} "
                "(random does not within the tested grid)."
            )
    return " ".join(parts)


def phase_plot(args) -> None:
    df = pd.read_csv(OUT_ROOT / "results.csv")
    se = args.se
    full = df[(df["strategy"] == "full_pool") & (df["se_target"] == se)]
    ref_r = float(full["r"].iloc[0]) if len(full) else float("nan")
    n_pool = int(full["n_pool"].iloc[0]) if len(full) else args.pool_cap
    plot_curve(df, se, ref_r, n_pool, OUT_ROOT / "oracle_curve_arc.png")

    gpath = OUT_ROOT / "latent_groups_arc.csv"
    theta_by_model, group_of = oss._reload_theta_groups(BENCH, args)
    if theta_by_model and gpath.exists():
        bounds = pd.read_csv(gpath).to_dict("records")
        spath = OUT_ROOT / "selected_subsets.csv"
        subsets_df = (
            pd.read_csv(spath)
            if spath.exists()
            else pd.DataFrame(columns=["strategy", "N", "model", "latent_group", "theta"])
        )
        plot_latent_axis(
            theta_by_model,
            group_of,
            bounds,
            subsets_df,
            OUT_ROOT / "latent_axis_arc.png",
            args.groups,
        )
    write_readme(df, se, args)
    print(f"[arc] plots + README -> {OUT_ROOT}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--phase", default="all", choices=["all", "core", "greedy", "plot"])
    p.add_argument("--timing-probe", action="store_true", dest="timing_probe")
    p.add_argument(
        "--n-list", default="50,100,150,200,250,300,350,400,450,500", dest="n_list"
    )
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--strategies", default="random,oracle_greedy")
    p.add_argument("--se", type=float, default=0.3)
    p.add_argument("--groups", type=int, default=5)
    p.add_argument("--pool-cap", type=int, default=100000, dest="pool_cap")
    p.add_argument("--pool-seed", type=int, default=13, dest="pool_seed")
    p.add_argument(
        "--n-test", type=int, default=0, dest="n_test", help="cap the fixed test set (0 = all 417)"
    )
    p.add_argument("--test-seed", type=int, default=7, dest="test_seed")
    p.add_argument("--chunk-size", type=int, default=100, dest="chunk_size")
    p.add_argument("--ncycles", type=int, default=500)
    p.add_argument("--jobs", type=int, default=12, help="parallel calibration tasks")
    p.add_argument("--workers", type=int, default=1, help="R chunk-fit workers per task")
    p.add_argument("--greedy-max-n", type=int, default=500, dest="greedy_max_n")
    p.add_argument(
        "--greedy-batch",
        type=int,
        default=12,
        dest="greedy_batch",
        help="candidates evaluated per greedy step (0 = whole pool)",
    )
    p.add_argument("--work-root", default=str(OUT_ROOT / "_work"), dest="work_root")
    a = p.parse_args()
    a.ns = [int(x) for x in a.n_list.split(",")]
    a.seeds = [int(x) for x in a.seeds.split(",")]
    a.strategies = a.strategies.split(",")
    return a


def main() -> None:
    args = parse()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    ses = [args.se]

    if args.phase == "plot":
        phase_plot(args)
        return

    fit_r, link_r = sweep.write_helper_scripts(_work(args))
    mat, train_pool, test_models = load_data(args)
    pool = oss.draw_pool(train_pool, args.pool_cap, args.pool_seed)
    grid = [n for n in args.ns if n <= len(pool)]
    print(
        f"[arc] POOL_USED={len(pool)}  N grid={grid}  strategies={args.strategies}  "
        f"jobs={args.jobs} workers={args.workers}",
        flush=True,
    )

    if args.timing_probe:
        timing_probe(args, mat, pool, test_models, ses, fit_r, link_r)
        return

    if args.phase in ("all", "core"):
        phase_core(args, mat, pool, test_models, grid, ses, fit_r, link_r)
    if args.phase in ("all", "greedy") and "oracle_greedy" in args.strategies:
        phase_greedy(args, mat, pool, test_models, grid, ses, fit_r, link_r)
    if args.phase == "all":
        phase_plot(args)


if __name__ == "__main__":
    main()
