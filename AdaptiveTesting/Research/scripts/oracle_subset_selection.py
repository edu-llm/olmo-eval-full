#!/usr/bin/env python3
"""Oracle (upper-bound) calibration-model subset selection for OpenLM ATLAS testing.

Research question: what is the THEORETICAL-BEST small subset of calibration models
that keeps roughly the same held-out correlation as the full pool? We answer it by
REVERSE-ENGINEERING the selection from the final calibration: calibrate on the full
pool, read each model's fitted latent ability (EAP theta) from that bank, partition
models into latent-ability groups, and then select representatives that span the
ability space. This uses information not available a priori (the final fitted
abilities), so every "oracle" number here is an UPPER BOUND / theoretical floor that
practical a-priori strategies (random, and the diversity strategies in the sibling
experiment data/model_diversity_selection/) can be compared against. It is NOT a
deployable selection rule.

Pipeline: we reuse the validated OpenLM 3PL pipeline in `openlm_trainsize_sweep.py`
(chunked R `mirt` 3PL fit, mean-sigma chunk linking with polarity flip, a>0 item
filter, EAP/Fisher-information p-IRT CAT accuracy recovery). The ONLY thing changed is
HOW the N calibration models are chosen. The held-out test set is the same fixed
seed-7 10% split for every strategy and every N, so all curves are comparable.

Scope (kept small for CPU-local speed):
  - Initial calibration pool is capped at --pool-cap (default 300) models, drawn once
    with a fixed seed from the cleaned train pool (after holding out the test set).
    This capped pool IS "the full pool" for this experiment: oracle latent abilities
    are fit on it and the full-pool reference correlation uses all of it.
  - Sweep N in coarse steps (default 25,50,100,150,200,250).
  - Strategies: random (3 seeds, mean +/- sd), oracle_stratified (latent-group
    spread), oracle_greedy (forward greedy maximizing held-out Pearson r; the
    strongest oracle, restricted to the smallest few N for tractability).
  - SE stopping target 0.3 only.

Outputs go to AdaptiveTesting/Research/01_MCQ_ATLAS/data/oracle_subset_selection/:
  results.csv                 benchmark, strategy, N, seed, se_target, r, mae, items, ...
  latent_groups_<bench>.csv   group id, theta lo/hi bound, size, mean theta
  best_subset_<bench>.csv     chosen models + latent group + theta for the smallest N
                              that holds correlation within tolerance
  oracle_curve_<bench>.png    r vs N: random band, oracle lines, full-pool reference
  latent_axis_<bench>.png     selected subset positions along the latent-ability axis
  README.md                   method + upper-bound caveat + min-N-to-hold vs random

Usage:
  uv run python AdaptiveTesting/Research/scripts/oracle_subset_selection.py \
      --benches ifeval,math --n-list 25,50,100,150,200,250 --seeds 0,1,2 \
      --pool-cap 300 --se-list 0.3 --groups 5 --jobs 6 --workers 2 \
      --greedy-max-n 100 --greedy-batch 20
  uv run python AdaptiveTesting/Research/scripts/oracle_subset_selection.py --plot-only
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import openlm_trainsize_sweep as sweep
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
REPO = SCRIPTS.parents[2]
OUT_ROOT = REPO / "AdaptiveTesting/Research/01_MCQ_ATLAS/data/oracle_subset_selection"

STRATEGIES = ("random", "oracle_stratified", "oracle_greedy")
FIELDS = [
    "benchmark",
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


# ---------------------------------------------------------------------------
# Pool + fit helpers (all calibration goes through the validated sweep pipeline)
# ---------------------------------------------------------------------------
def _work_root(args, bench: str) -> Path:
    root = Path(args.work_root) if args.work_root else OUT_ROOT / "_work"
    return root / bench


def draw_pool(train_pool: list[str], cap: int, seed: int) -> list[str]:
    """Draw a fixed-seed capped calibration pool from the cleaned train pool."""
    if cap <= 0 or cap >= len(train_pool):
        return list(train_pool)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(train_pool), size=cap, replace=False)
    return [train_pool[i] for i in sorted(idx)]


def fit_subset(args, mat, test_models, step_dir: Path, fit_r, link_r, subset, ses):
    """Calibrate on `subset`, run the held-out CAT, return sweep.diagnose metrics."""
    ends, _ = sweep.prepare_step(mat, subset, test_models, step_dir, args.chunk_size)
    sweep.fit_and_link(step_dir, ends, args.ncycles, args.workers, fit_r, link_r)
    return sweep.diagnose(step_dir, ses)


def _row(bench, strategy, n, seed, se, m, n_pool, n_test, calib_s) -> dict:
    return {
        "benchmark": bench,
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


# ---------------------------------------------------------------------------
# Step 1: full-pool calibration -> per-model latent ability (EAP theta) + reference r
# ---------------------------------------------------------------------------
def full_pool_calibration(args, bench, mat, test_models, pool, fit_r, link_r, ses):
    """Calibrate on the whole (capped) pool. Return (full_metrics, theta_by_model, s).

    theta_by_model is each pool model's EAP theta computed on the final linked bank
    (all a>0 items), i.e. its ORACLE latent ability used to define groups.
    """
    step_dir = _work_root(args, bench) / "full_pool"
    t0 = time.time()
    metrics = fit_subset(args, mat, test_models, step_dir, fit_r, link_r, pool, ses)
    calib_s = round(time.time() - t0, 1)
    train = sweep.load_matrix(step_dir / "data" / "response_matrix_train.csv")
    idxs, a, b, c = sweep.load_bank(step_dir / "calibration", list(train.columns))
    theta_by_model: dict[str, float] = {}
    for mid in train.index:
        resp = train.loc[mid, idxs].to_numpy(float)
        theta, _ = sweep.eap_se(resp, a, b, c)
        theta_by_model[mid] = float(theta)
    print(
        f"[{bench}] full pool: N={len(pool)} bank={len(idxs)} items "
        f"SE{ses[0]} r={metrics[ses[0]]['corr']} items={metrics[ses[0]]['mean_items']} "
        f"({calib_s}s)",
        flush=True,
    )
    return metrics, theta_by_model, calib_s


# ---------------------------------------------------------------------------
# Latent-ability groups (quantile bins of the full-pool theta)
# ---------------------------------------------------------------------------
def latent_groups(theta_by_model: dict[str, float], k: int):
    """Partition pool models into k quantile bins of oracle theta.

    Returns (group_of_model, bounds) where bounds has k rows: (lo, hi, size, mean).
    """
    models = list(theta_by_model)
    theta = np.array([theta_by_model[m] for m in models], float)
    edges = np.quantile(theta, np.linspace(0.0, 1.0, k + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    gid = np.clip(np.digitize(theta, edges[1:-1], right=False), 0, k - 1)
    group_of = {m: int(g) for m, g in zip(models, gid, strict=False)}
    bounds = []
    for g in range(k):
        sel = theta[gid == g]
        bounds.append(
            {
                "group": g,
                "theta_lo": round(float(sel.min()), 4) if len(sel) else float("nan"),
                "theta_hi": round(float(sel.max()), 4) if len(sel) else float("nan"),
                "size": int((gid == g).sum()),
                "mean_theta": round(float(sel.mean()), 4) if len(sel) else float("nan"),
            }
        )
    return group_of, bounds


# ---------------------------------------------------------------------------
# Selection strategies
# ---------------------------------------------------------------------------
def select_random(pool: list[str], n: int, seed: int) -> list[str]:
    rng = np.random.default_rng(1000 + seed)
    return [pool[i] for i in rng.choice(len(pool), size=n, replace=False)]


def select_oracle_stratified(theta_by_model, group_of, pool, k, n) -> list[str]:
    """Spread N models across the k latent groups (equal per group; remainder to the
    largest groups) and, within each group, pick models nearest to evenly spaced theta
    targets so the subset spans the ability axis inside every group too."""
    by_group: dict[int, list[str]] = {g: [] for g in range(k)}
    for m in pool:
        by_group[group_of[m]].append(m)
    sizes = {g: len(by_group[g]) for g in range(k)}
    alloc = {g: n // k for g in range(k)}
    rem = n - sum(alloc.values())
    for g in sorted(range(k), key=lambda g: sizes[g], reverse=True):
        if rem <= 0:
            break
        if alloc[g] < sizes[g]:
            alloc[g] += 1
            rem -= 1
    chosen: list[str] = []
    for g in range(k):
        members = sorted(by_group[g], key=lambda m: theta_by_model[m])
        take = min(alloc[g], len(members))
        if take <= 0:
            continue
        tvals = np.array([theta_by_model[m] for m in members], float)
        targets = np.quantile(tvals, np.linspace(0.0, 1.0, take)) if take > 1 else [tvals.mean()]
        used: set[str] = set()
        for q in targets:
            for j in np.argsort(np.abs(tvals - q)):
                if members[j] not in used:
                    used.add(members[j])
                    chosen.append(members[j])
                    break
    if len(chosen) < n:  # tiny groups can leave us short; top up by theta order
        remaining = [m for m in pool if m not in set(chosen)]
        remaining.sort(key=lambda m: theta_by_model[m])
        chosen += remaining[: n - len(chosen)]
    return chosen[:n]


# ---------------------------------------------------------------------------
# Oracle greedy: forward selection maximizing held-out Pearson r (upper bound)
# ---------------------------------------------------------------------------
def oracle_greedy_build(
    args, bench, mat, test_models, pool, theta_by_model, fit_r, link_r, ses, grid
):
    """Greedily add the pool model that most improves held-out Pearson r. Seeded with
    the two theta extremes so the initial bank is non-degenerate. Builds up to
    --greedy-max-n and records metrics at the requested grid sizes. This peeks at the
    held-out set, so it is an explicit ORACLE upper bound, not a usable rule."""
    se0 = ses[0]
    work = _work_root(args, bench)
    gtmp = work / "greedy_tmp"
    gkeep = work / "greedy_keep"
    max_n = min(args.greedy_max_n, len(pool))
    targets = sorted(nn for nn in grid if nn <= max_n)
    if not targets:
        return [], {}
    rng = np.random.default_rng(4242)

    thetas = np.array([theta_by_model[m] for m in pool], float)
    selected = [pool[int(np.argmin(thetas))], pool[int(np.argmax(thetas))]]
    rows: list[dict] = []
    subset_at: dict[int, list[str]] = {}

    def eval_subset(subset, tag):
        sd = gtmp / tag
        try:
            m = fit_subset(args, mat, test_models, sd, fit_r, link_r, subset, ses)
            r = m[se0]["corr"]
            return float(r) if r == r else float("-inf")
        except Exception:
            return float("-inf")
        finally:
            shutil.rmtree(sd, ignore_errors=True)

    while len(selected) < max_n:
        pend = [m for m in pool if m not in set(selected)]
        if args.greedy_batch and len(pend) > args.greedy_batch:
            cand = [pend[i] for i in rng.choice(len(pend), size=args.greedy_batch, replace=False)]
        else:
            cand = pend
        results: dict[str, float] = {}
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            futs = {
                ex.submit(eval_subset, selected + [c], f"s{len(selected)}_{i}"): c
                for i, c in enumerate(cand)
            }
            for fut in as_completed(futs):
                results[futs[fut]] = fut.result()
        selected.append(max(results, key=lambda c: results[c]))
        size = len(selected)
        if size in targets:
            subset_at[size] = list(selected)
            sd = gkeep / f"n{size}"
            m = fit_subset(args, mat, test_models, sd, fit_r, link_r, selected, ses)
            rows.append(
                _row(bench, "oracle_greedy", size, 0, se0, m[se0], len(pool), len(test_models), 0.0)
            )
            print(
                f"[{bench}] oracle_greedy N={size:<4} bank={m[se0]['n_bank_items']:<4} "
                f"SE{se0} r={m[se0]['corr']} items={m[se0]['mean_items']}",
                flush=True,
            )
            shutil.rmtree(sd, ignore_errors=True)
    return rows, subset_at


# ---------------------------------------------------------------------------
# Per-benchmark driver
# ---------------------------------------------------------------------------
def run_bench(args, bench: str, ses: list[float]) -> None:
    work = _work_root(args, bench)
    fit_r, link_r = sweep.write_helper_scripts(work)
    print(f"[{bench}] building master matrix...", flush=True)
    mat = sweep.build_master(bench)
    test_models, train_pool = sweep.split_models(mat, args.seed, args.test_frac)
    pool = draw_pool(train_pool, args.pool_cap, args.pool_seed)
    grid = [n for n in args.ns if n <= len(pool)]
    print(
        f"[{bench}] models={len(mat)} items={mat.shape[1]} n_test={len(test_models)} "
        f"cleaned_train_pool={len(train_pool)} POOL_USED={len(pool)} N grid={grid}",
        flush=True,
    )

    se0 = ses[0]
    rows: list[dict] = []

    full_metrics, theta_by_model, full_s = full_pool_calibration(
        args, bench, mat, test_models, pool, fit_r, link_r, ses
    )
    rows.append(
        _row(
            bench,
            "full_pool",
            len(pool),
            0,
            se0,
            full_metrics[se0],
            len(pool),
            len(test_models),
            full_s,
        )
    )

    group_of, bounds = latent_groups(theta_by_model, args.groups)
    _write_groups(bench, bounds)

    tasks: list[tuple[str, int, int]] = []
    for n in grid:
        if "oracle_stratified" in args.strategies:
            tasks.append(("oracle_stratified", 0, n))
        if "random" in args.strategies:
            for seed in args.seeds:
                tasks.append(("random", seed, n))

    def run_task(strat, seed, n):
        sd = work / f"{strat}_s{seed}_n{n}"
        if strat == "random":
            subset = select_random(pool, n, seed)
        else:
            subset = select_oracle_stratified(theta_by_model, group_of, pool, args.groups, n)
        t0 = time.time()
        m = fit_subset(args, mat, test_models, sd, fit_r, link_r, subset, ses)
        return (
            _row(
                bench,
                strat,
                n,
                seed,
                se0,
                m[se0],
                len(pool),
                len(test_models),
                round(time.time() - t0, 1),
            ),
            subset,
        )

    stratified_subsets: dict[int, list[str]] = {}
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(run_task, s, sd, n): (s, sd, n) for (s, sd, n) in tasks}
        for fut in as_completed(futs):
            s, sd, n = futs[fut]
            try:
                row, subset = fut.result()
                rows.append(row)
                if s == "oracle_stratified":
                    stratified_subsets[n] = subset
                print(
                    f"[{bench}] {s:<18} seed={sd} N={n:<4} bank={row['n_bank_items']:<4} "
                    f"SE{se0} r={row['r']} items={row['items']} ({row['calib_s']}s)",
                    flush=True,
                )
            except Exception as exc:
                print(f"[{bench}] {s} seed={sd} N={n} FAILED: {exc}", flush=True)
        _write_results(rows)

    greedy_subsets: dict[int, list[str]] = {}
    if "oracle_greedy" in args.strategies:
        print(f"[{bench}] building oracle_greedy (held-out r, upper bound)...", flush=True)
        grows, greedy_subsets = oracle_greedy_build(
            args, bench, mat, test_models, pool, theta_by_model, fit_r, link_r, ses, grid
        )
        rows.extend(grows)
    _write_results(rows)

    subsets_by_strat = {"oracle_stratified": stratified_subsets, "oracle_greedy": greedy_subsets}
    _write_best_subset(
        bench,
        se0,
        full_metrics[se0]["corr"],
        args.tol_primary,
        theta_by_model,
        group_of,
        subsets_by_strat,
    )


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def _min_n_hold(df_strat: pd.DataFrame, ref_r: float, tol: float):
    """Smallest N whose mean r >= ref_r - tol, else None."""
    g = df_strat.groupby("N")["r"].mean().reset_index().sort_values("N")
    hit = g[g["r"] >= ref_r - tol]
    return int(hit["N"].iloc[0]) if len(hit) else None


def _interp_random_n(df_rand: pd.DataFrame, ref_r: float, tol: float):
    """N at which the random mean curve first reaches ref_r - tol (linear interp)."""
    g = df_rand.groupby("N")["r"].mean().reset_index().sort_values("N")
    xs, ys = g["N"].to_numpy(float), g["r"].to_numpy(float)
    thr = ref_r - tol
    for i in range(len(xs)):
        if ys[i] >= thr:
            if i == 0:
                return float(xs[0])
            x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
            return float(x1) if y1 == y0 else float(x0 + (thr - y0) * (x1 - x0) / (y1 - y0))
    return None


def analyze(df: pd.DataFrame, se: float, benches: list[str], tols: list[float]) -> dict:
    out: dict[str, dict] = {}
    for bench in benches:
        gb = df[(df["benchmark"] == bench) & (df["se_target"] == se) & np.isfinite(df["r"])]
        full = gb[gb["strategy"] == "full_pool"]
        if not len(full):
            continue
        ref_r = float(full["r"].iloc[0])
        info: dict = {"ref_r": round(ref_r, 4), "n_pool": int(full["n_pool"].iloc[0]), "tols": {}}
        for tol in tols:
            entry: dict = {"random_n": None, "strategies": {}}
            rnd = gb[gb["strategy"] == "random"]
            if len(rnd):
                rn = _interp_random_n(rnd, ref_r, tol)
                entry["random_n"] = round(rn, 1) if rn else None
            for strat in ("oracle_stratified", "oracle_greedy"):
                ds = gb[gb["strategy"] == strat]
                if not len(ds):
                    continue
                nhit = _min_n_hold(ds, ref_r, tol)
                rn = entry["random_n"]
                entry["strategies"][strat] = {
                    "min_n": nhit,
                    "r_at_min_n": (
                        round(float(ds[ds["N"] == nhit]["r"].mean()), 4) if nhit else None
                    ),
                    "savings_vs_random": (round(rn / nhit, 2) if (nhit and rn) else None),
                }
            info["tols"][tol] = entry
        out[bench] = info
    return out


# ---------------------------------------------------------------------------
# Outputs: CSVs
# ---------------------------------------------------------------------------
def _write_results(rows: list[dict]) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    path = OUT_ROOT / "results.csv"
    existing = pd.read_csv(path).to_dict("records") if path.exists() else []

    def key(r):
        return (r["benchmark"], r["strategy"], int(r["N"]), int(r["seed"]), float(r["se_target"]))

    merged = {key(r): r for r in existing}
    for r in rows:
        merged[key(r)] = r
    out = sorted(
        merged.values(),
        key=lambda r: (
            r["benchmark"],
            r["strategy"],
            float(r["se_target"]),
            int(r["N"]),
            int(r["seed"]),
        ),
    )
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in out:
            w.writerow({k: r.get(k) for k in FIELDS})


def _write_groups(bench: str, bounds: list[dict]) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    path = OUT_ROOT / f"latent_groups_{bench}.csv"
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["group", "theta_lo", "theta_hi", "size", "mean_theta"])
        w.writeheader()
        w.writerows(bounds)


def _write_best_subset(bench, se, ref_r, tol, theta_by_model, group_of, subsets_by_strat) -> None:
    """Write the theoretical-best subset: smallest N (any oracle strategy) whose held-out
    r is within tol of the full-pool reference. Prefer oracle_greedy, else stratified."""
    df = pd.read_csv(OUT_ROOT / "results.csv")
    gb = df[(df["benchmark"] == bench) & (df["se_target"] == se) & np.isfinite(df["r"])]
    best = None  # (N, strategy, r)
    for strat in ("oracle_greedy", "oracle_stratified"):
        ds = gb[gb["strategy"] == strat].sort_values("N")
        hit = ds[ds["r"] >= ref_r - tol]
        if len(hit):
            n = int(hit["N"].iloc[0])
            if best is None or n < best[0]:
                best = (n, strat, float(hit["r"].iloc[0]))
    if best is None:
        return
    n, strat, r = best
    subset = subsets_by_strat.get(strat, {}).get(n)
    if not subset:
        return
    rows = [
        {
            "model": m,
            "latent_group": group_of.get(m, ""),
            "theta": round(theta_by_model.get(m, float("nan")), 4),
        }
        for m in sorted(subset, key=lambda m: theta_by_model.get(m, 0.0))
    ]
    path = OUT_ROOT / f"best_subset_{bench}.csv"
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "latent_group", "theta"])
        w.writeheader()
        w.writerows(rows)
    print(
        f"[{bench}] best subset: {strat} N={n} r={r:.4f} (ref {ref_r:.4f}, tol {tol}) "
        f"-> {path.name}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Outputs: figures
# ---------------------------------------------------------------------------
def _mpl():
    os.environ.setdefault("MPLCONFIGDIR", str(OUT_ROOT / ".mplcache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


COLORS = {"random": "#7f7f7f", "oracle_stratified": "#1f77b4", "oracle_greedy": "#d62728"}
LABELS = {
    "random": "random (mean +/- sd)",
    "oracle_stratified": "oracle: latent-group stratified",
    "oracle_greedy": "oracle: greedy on held-out r",
}


def _plot_curve(plt, df, bench, se, info, path, tol_primary):
    gb = df[(df["benchmark"] == bench) & (df["se_target"] == se) & np.isfinite(df["r"])]
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
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
    for strat in ("oracle_stratified", "oracle_greedy"):
        ds = gb[gb["strategy"] == strat].groupby("N")["r"].mean().reset_index().sort_values("N")
        if not len(ds):
            continue
        ax.plot(
            ds["N"],
            ds["r"],
            "s-" if strat == "oracle_stratified" else "^-",
            color=COLORS[strat],
            label=LABELS[strat],
            zorder=4,
        )
    ref_r = info["ref_r"]
    ax.axhline(
        ref_r,
        color="black",
        ls="-",
        lw=1.2,
        label=f"full-pool reference r={ref_r:.3f} (N={info['n_pool']})",
    )
    ax.axhline(
        ref_r - tol_primary, color="black", ls="--", lw=0.9, label=f"reference - {tol_primary:g}"
    )
    entry = info["tols"].get(tol_primary, {})
    for strat in ("oracle_stratified", "oracle_greedy"):
        s = entry.get("strategies", {}).get(strat, {})
        if s.get("min_n"):
            ax.scatter(
                [s["min_n"]],
                [s["r_at_min_n"]],
                s=190,
                facecolors="none",
                edgecolors=COLORS[strat],
                linewidths=2.4,
                zorder=5,
            )
    ax.set_xlabel("number of calibration models (N)")
    ax.set_ylabel("Pearson r (p-IRT predicted vs actual accuracy)")
    ax.set_title(_headline(bench, info, tol_primary), fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_latent_axis(plt, bench, theta_by_model, group_of, bounds, best_subset_path, path, k):
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(9.0, 3.6))
    models = list(theta_by_model)
    theta = np.array([theta_by_model[m] for m in models], float)
    gid = np.array([group_of[m] for m in models], int)
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, k))
    for g in range(k):
        sel = theta[gid == g]
        ax.scatter(
            sel,
            rng.uniform(-0.06, 0.06, len(sel)),
            s=14,
            color=cmap[g],
            alpha=0.35,
            label=f"group {g} (n={len(sel)})",
        )
    for row in bounds[:-1]:
        ax.axvline(row["theta_hi"], color="gray", ls=":", lw=0.8)
    if Path(best_subset_path).exists():
        bs = pd.read_csv(best_subset_path)
        for _, r in bs.iterrows():
            ax.scatter(
                [r["theta"]],
                [0.35],
                s=90,
                marker="v",
                color=cmap[int(r["latent_group"])],
                edgecolors="black",
                linewidths=0.6,
                zorder=5,
            )
        ax.text(
            theta.min(), 0.5, f"selected subset (N={len(bs)}), colored by latent group", fontsize=9
        )
    ax.set_yticks([])
    ax.set_ylim(-0.25, 0.65)
    ax.set_xlabel("oracle latent ability theta (full-pool EAP)")
    ax.set_title(f"{bench}: selected subset positions along the latent-ability axis", fontsize=11)
    ax.legend(loc="lower right", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Outputs: README
# ---------------------------------------------------------------------------
def _headline(bench: str, info: dict, tol: float) -> str:
    entry = info["tols"].get(tol, {})
    best = None
    for strat in ("oracle_greedy", "oracle_stratified"):
        s = entry.get("strategies", {}).get(strat, {})
        if s.get("min_n") and (best is None or s["min_n"] < best[1]):
            best = (strat, s["min_n"], s.get("savings_vs_random"))
    if best and best[2]:
        return (
            f"{bench}: oracle holds r within {tol:g} of the {info['n_pool']}-model pool "
            f"at N={best[1]} vs random N~{entry.get('random_n')} ({best[2]:g}x fewer)"
        )
    if best:
        return (
            f"{bench}: oracle holds r within {tol:g} of the {info['n_pool']}-model pool "
            f"at N={best[1]}"
        )
    return f"{bench}: oracle subset vs random (SE<=0.3), {info['n_pool']}-model pool"


def _fmt_strat(name: str) -> str:
    return {"oracle_stratified": "oracle-stratified", "oracle_greedy": "oracle-greedy"}[name]


def write_readme(df, analysis, se, benches, args, skipped_note):
    L = [
        "# Oracle (upper-bound) calibration-model subset selection",
        "",
        "## What this is (read first)",
        "",
        "This experiment finds the THEORETICAL-BEST small subset of calibration models",
        "that keeps roughly the same held-out correlation as the full pool, by",
        "REVERSE-ENGINEERING the choice from the final calibration. We calibrate on the",
        "whole pool, read each model's fitted latent ability (EAP theta) from that bank,",
        "split models into latent-ability groups, and select representatives that span",
        "the ability axis. The greedy variant goes further and directly maximizes the",
        "held-out correlation.",
        "",
        "**These are ORACLE numbers / an UPPER BOUND.** They use information not",
        "available before calibration (the final fitted abilities, and for greedy the",
        "held-out accuracies themselves). They are NOT a deployable selection rule; they",
        "set the theoretical floor that practical a-priori strategies (random, and the",
        "diversity strategies in `data/model_diversity_selection/`) are compared against.",
        "",
        "## Method",
        "",
        "Same validated OpenLM 3PL pipeline as `openlm_trainsize_sweep.py` (chunked R",
        "mirt 3PL, mean-sigma linking with polarity flip, a>0 item filter, EAP/Fisher",
        "p-IRT CAT). Only the choice of the N calibration models changes. The held-out",
        f"test set is the same fixed seed-{args.seed} {int(args.test_frac * 100)}% split for every",
        f"strategy and N, so all curves are comparable. Reported at SE<={se:g}.",
        "",
        f"Initial calibration pool is capped at {args.pool_cap} models, drawn once with a",
        "fixed seed from the cleaned train pool (after holding out the test set). That",
        'capped pool IS "the full pool" here: oracle abilities are fit on it and the',
        "full-pool reference correlation uses all of it. Latent groups are",
        f"{args.groups} quantile bins of the full-pool EAP theta (latent_groups_<bench>.csv).",
        "",
        "Strategies: `random` (mean +/- sd over seeds), `oracle_stratified` (spread N",
        "models evenly across the latent groups and across theta within each group),",
        "`oracle_greedy` (forward greedy that adds the model most improving held-out",
        "Pearson r, seeded at the two theta extremes; the strongest oracle).",
        "",
        "## Headline: minimum N to hold correlation",
        "",
    ]
    for bench in benches:
        info = analysis.get(bench)
        if not info:
            continue
        L.append(f"### {bench}")
        L.append("")
        L.append(
            f"- Pool used: {info['n_pool']} models. Full-pool reference "
            f"r = {info['ref_r']:.4f} (SE<={se:g})."
        )
        for tol in args.tols:
            entry = info["tols"].get(tol, {})
            L.append(f"- Within {tol:g} r of the reference (r >= {info['ref_r'] - tol:.4f}):")
            rn = entry.get("random_n")
            L.append(
                f"    - random needs N ~ {rn:g}."
                if rn
                else "    - random never reaches it on the tested grid."
            )
            for strat in ("oracle_stratified", "oracle_greedy"):
                s = entry.get("strategies", {}).get(strat)
                if not s:
                    continue
                if s["min_n"]:
                    sv = (
                        f", {s['savings_vs_random']:g}x fewer than random"
                        if s["savings_vs_random"]
                        else ""
                    )
                    L.append(
                        f"    - {_fmt_strat(strat)}: min N = {s['min_n']} "
                        f"(r = {s['r_at_min_n']:.4f}){sv}."
                    )
                else:
                    L.append(
                        f"    - {_fmt_strat(strat)}: did not hold within this tol on the grid."
                    )
        L.append("")
        L.append(
            f"See `best_subset_{bench}.csv` for the theoretical-best subset "
            "(models + latent group + theta) at the smallest holding N."
        )
        L.append("")
    L += ["## Verdict", "", _verdict(analysis, benches, args.tol_primary), ""]
    if skipped_note:
        L += ["## Skipped for time", "", skipped_note, ""]
    L += [
        "## Outputs",
        "",
        "- `results.csv`: benchmark, strategy (full_pool/random/oracle_stratified/"
        "oracle_greedy), N, seed, se_target, r, mae, items, n_bank_items, n_pool, n_test.",
        "- `latent_groups_<bench>.csv`: group id, theta bounds, size, mean theta.",
        "- `best_subset_<bench>.csv`: chosen models + latent group + theta at the "
        "smallest holding N.",
        "- `oracle_curve_<bench>.png`: r vs N, random band, oracle lines, reference line.",
        "- `latent_axis_<bench>.png`: selected subset positions along the latent-ability axis.",
        "",
        "## Caveats",
        "",
        "- Oracle/upper-bound only (uses final fitted abilities; greedy peeks at held-out r).",
        "- Random savings factors use linear interpolation of the random mean curve to the",
        "  tolerance threshold; treat single small-N points as noisy.",
        "- oracle_stratified is deterministic (one curve, no band).",
        "",
    ]
    (OUT_ROOT / "README.md").write_text("\n".join(L))


def _verdict(analysis, benches, tol) -> str:
    parts = []
    for bench in benches:
        info = analysis.get(bench)
        if not info:
            continue
        entry = info["tols"].get(tol, {})
        best = None
        for strat in ("oracle_greedy", "oracle_stratified"):
            s = entry.get("strategies", {}).get(strat, {})
            if s.get("min_n") and (best is None or s["min_n"] < best[0]):
                best = (s["min_n"], strat, s.get("savings_vs_random"))
        if best:
            sv = f", ~{best[2]:g}x fewer than random" if best[2] else ""
            parts.append(f"{bench}: oracle holds within {tol:g} r at N={best[0]}{sv}")
    if not parts:
        return "No oracle subset held within the primary tolerance on the tested grid."
    return (
        "An oracle subset that spans the latent-ability space keeps the full-pool "
        "correlation with far fewer calibration models ("
        + "; ".join(parts)
        + "). This is an upper bound; practical strategies fall between this and random."
    )


def make_outputs(df, se, benches, args, skipped_note) -> dict:
    analysis = analyze(df, se, benches, args.tols)
    plt = _mpl()
    for bench in benches:
        if bench not in analysis:
            continue
        _plot_curve(
            plt,
            df,
            bench,
            se,
            analysis[bench],
            OUT_ROOT / f"oracle_curve_{bench}.png",
            args.tol_primary,
        )
        gpath = OUT_ROOT / f"latent_groups_{bench}.csv"
        if gpath.exists():
            bounds = pd.read_csv(gpath).to_dict("records")
            theta_by_model, group_of = _reload_theta_groups(bench, args)
            if theta_by_model:
                _plot_latent_axis(
                    plt,
                    bench,
                    theta_by_model,
                    group_of,
                    bounds,
                    OUT_ROOT / f"best_subset_{bench}.csv",
                    OUT_ROOT / f"latent_axis_{bench}.png",
                    args.groups,
                )
    write_readme(df, analysis, se, benches, args, skipped_note)
    return analysis


def _reload_theta_groups(bench, args):
    """Recompute per-model theta + groups from the saved full_pool calibration (for
    plot-only reruns). Returns ({}, {}) if the work dir is gone."""
    step_dir = _work_root(args, bench) / "full_pool"
    calib = step_dir / "calibration"
    data = step_dir / "data" / "response_matrix_train.csv"
    if not (calib / "irt_item_parameters_combined.csv").exists() or not data.exists():
        return {}, {}
    train = sweep.load_matrix(data)
    idxs, a, b, c = sweep.load_bank(calib, list(train.columns))
    theta_by_model = {}
    for mid in train.index:
        theta, _ = sweep.eap_se(train.loc[mid, idxs].to_numpy(float), a, b, c)
        theta_by_model[mid] = float(theta)
    group_of, _ = latent_groups(theta_by_model, args.groups)
    return theta_by_model, group_of


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--benches", default="ifeval,math")
    p.add_argument("--n-list", default="25,50,100,150,200,250", dest="n_list")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--strategies", default="random,oracle_stratified,oracle_greedy")
    p.add_argument("--se-list", default="0.3", dest="se_list")
    p.add_argument("--groups", type=int, default=5, help="number of latent-ability groups")
    p.add_argument("--pool-cap", type=int, default=300, help="initial calibration pool cap")
    p.add_argument("--pool-seed", type=int, default=13, help="seed for drawing the capped pool")
    p.add_argument("--seed", type=int, default=7, help="split seed (held-out test set)")
    p.add_argument("--test-frac", type=float, default=0.10)
    p.add_argument("--chunk-size", type=int, default=100)
    p.add_argument("--ncycles", type=int, default=500)
    p.add_argument("--jobs", type=int, default=6, help="parallel calibration tasks")
    p.add_argument("--workers", type=int, default=2, help="R chunk-fit workers per task")
    p.add_argument(
        "--greedy-max-n",
        type=int,
        default=100,
        dest="greedy_max_n",
        help="build oracle_greedy up to this N (grid points <= it get greedy rows)",
    )
    p.add_argument(
        "--greedy-batch",
        type=int,
        default=20,
        dest="greedy_batch",
        help="candidates evaluated per greedy step (0 = whole pool)",
    )
    p.add_argument("--tols", default="0.02,0.05", help="hold-correlation tolerances")
    p.add_argument("--work-root", default="")
    p.add_argument(
        "--plot-only",
        action="store_true",
        help="rebuild figures + README from existing results.csv",
    )
    a = p.parse_args()

    a.ns = [int(x) for x in a.n_list.split(",")]
    a.seeds = [int(x) for x in a.seeds.split(",")]
    a.strategies = a.strategies.split(",")
    a.tols = [float(x) for x in a.tols.split(",")]
    a.tol_primary = a.tols[0]
    benches = a.benches.split(",")
    ses = [float(s) for s in a.se_list.split(",")]

    skipped_note = ""
    if "oracle_greedy" not in a.strategies:
        skipped_note = (
            "oracle_greedy (held-out greedy) was skipped to save time; only "
            "oracle_stratified and random were run."
        )

    if a.plot_only:
        df = pd.read_csv(OUT_ROOT / "results.csv")
        analysis = make_outputs(df, ses[0], benches, a, skipped_note)
        _print_summary(analysis, benches, a.tols)
        print(f"[plot-only] wrote figures + README to {OUT_ROOT}")
        return

    for bench in benches:
        run_bench(a, bench, ses)

    df = pd.read_csv(OUT_ROOT / "results.csv")
    analysis = make_outputs(df, ses[0], benches, a, skipped_note)
    print(f"\nDONE -> {OUT_ROOT}")
    _print_summary(analysis, benches, a.tols)


def _print_summary(analysis: dict, benches: list[str], tols: list[float]) -> None:
    print("\n=== oracle min-N-to-hold summary (SE<=0.3) ===")
    for bench in benches:
        info = analysis.get(bench)
        if not info:
            continue
        print(f"[{bench}] pool={info['n_pool']} full-pool reference r={info['ref_r']}")
        for tol in tols:
            entry = info["tols"].get(tol, {})
            print(
                f"  within {tol} r (>= {info['ref_r'] - tol:.4f}): random N~{entry.get('random_n')}"
            )
            for strat, s in entry.get("strategies", {}).items():
                print(
                    f"    {strat}: min N={s['min_n']} r={s['r_at_min_n']} "
                    f"savings={s['savings_vs_random']}x"
                )


if __name__ == "__main__":
    main()
