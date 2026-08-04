#!/usr/bin/env python3
"""Parameter-band extrapolation experiment for ATLAS-style 3PL adaptive testing on
OpenLM per-question response data.

Research question: does an IRT (3PL) item bank calibrated on models from ONE
parameter range extrapolate to models in a DIFFERENT, non-overlapping range? We split
the ~1,100 OpenLM `status==ok` models by `params_b` into two non-overlapping bands with
a deliberate GAP in the middle (no MIDDLE-band models used anywhere):

  LOW  = params_b <= --low-max   (default 2.0B)
  HIGH = params_b >= --high-min  (default 3.5B)
  MIDDLE = (low_max, high_min)   -> EXCLUDED from calibration and test

For each benchmark and seed we hold out `--n-test` models from EACH band, then match
the calibration pool size across bands (subsample the larger band down to the smaller
band's available N) so the ONLY difference between the LOW and HIGH banks is the
parameter range, not the pool size. Four conditions are evaluated:

  LOW->HIGH  (extrapolation): bank=calib_LOW,  test=heldout_HIGH
  HIGH->LOW  (extrapolation): bank=calib_HIGH, test=heldout_LOW
  LOW->LOW   (control)      : bank=calib_LOW,  test=heldout_LOW
  HIGH->HIGH (control)      : bank=calib_HIGH, test=heldout_HIGH

Each cross condition shares its held-out test set with its in-range control
(LOW->HIGH vs HIGH->HIGH share heldout_HIGH; HIGH->LOW vs LOW->LOW share heldout_LOW),
so the extrapolation penalty is a clean bank-only comparison:

  penalty(LOW->HIGH) = r(HIGH->HIGH) - r(LOW->HIGH)   # transfer onto HIGH test set
  penalty(HIGH->LOW) = r(LOW->LOW)   - r(HIGH->LOW)   # transfer onto LOW test set

Calibration + CAT reuse the EXACT validated ATLAS pipeline from
`openlm_trainsize_sweep.py`: chunked 3PL fit in R `mirt`, mean-sigma chunk linking with
polarity flip, the CRITICAL a>0 positive-discrimination item filter, and EAP/Fisher-info
CAT with p-IRT accuracy recovery. Only the model-SELECTION logic changes (parameter-band
instead of random k-subset). Each (band, seed) bank is fitted ONCE and reused for both of
its test targets.

Outputs -> AdaptiveTesting/Research/01_MCQ_ATLAS/data/param_extrapolation/
  param_extrapolation_results.csv  (one row per benchmark/condition/SE/seed)
  <bench>_param_extrapolation.png  (grouped bar of r across the 4 conditions)
  param_extrapolation_combined.png (all benchmarks)

Usage:
  export PYTHONPATH=/Users/arhant/Documents/EDLM/olmo-eval-full/eduLLM-Evals
  uv run python param_extrapolation_sweep.py --bench ifeval --seeds 3 --workers 6
  uv run python param_extrapolation_sweep.py --bench math --seeds 3 --workers 8
  uv run python param_extrapolation_sweep.py --plot        # (re)build figures from CSV
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import openlm_trainsize_sweep as ts  # noqa: E402  (reuse validated ATLAS internals)

REPO = SCRIPTS.parents[2]
STATUS_CSV = REPO / "AdaptiveTesting/Research/05_Data_Availability/data/openlm_download_status.csv"
OUT_ROOT = REPO / "AdaptiveTesting/Research/01_MCQ_ATLAS/data/param_extrapolation"

CONDITIONS = ["LOW->HIGH", "HIGH->LOW", "LOW->LOW", "HIGH->HIGH"]
FIELDS = [
    "benchmark",
    "condition",
    "se_target",
    "seed",
    "corr",
    "mae",
    "mean_items",
    "median_items",
    "pct_items",
    "n_calib",
    "n_test",
    "n_bank_items",
    "low_max_b",
    "high_min_b",
]


# ---------------------------------------------------------------------------
# Model -> params band selection (the only new logic vs. the train-size sweep)
# ---------------------------------------------------------------------------
def load_param_map() -> dict[str, float]:
    df = pd.read_csv(STATUS_CSV)
    df = df[df["status"] == "ok"][["model", "params_b"]].dropna()
    return dict(zip(df["model"], df["params_b"].astype(float), strict=False))


def select_bands(
    mat: pd.DataFrame,
    pmap: dict[str, float],
    low_max: float,
    high_min: float,
    n_test: int,
    seed: int,
) -> dict:
    """Held-out test + matched-N calibration pools for the LOW and HIGH bands."""
    low = [m for m in mat.index if m in pmap and pmap[m] <= low_max]
    high = [m for m in mat.index if m in pmap and pmap[m] >= high_min]
    rng = np.random.default_rng(seed)
    low = [low[i] for i in rng.permutation(len(low))]
    high = [high[i] for i in rng.permutation(len(high))]
    test_low, pool_low = low[:n_test], low[n_test:]
    test_high, pool_high = high[:n_test], high[n_test:]
    n_calib = min(len(pool_low), len(pool_high))
    return {
        "test_low": test_low,
        "test_high": test_high,
        "calib_low": pool_low[:n_calib],
        "calib_high": pool_high[:n_calib],
        "n_calib": n_calib,
        "n_low_total": len(low),
        "n_high_total": len(high),
    }


# ---------------------------------------------------------------------------
# Fit one bank (calibrate on train models only) + diagnose against a test set,
# reusing the exact ATLAS fit/link/CAT internals from openlm_trainsize_sweep.
# ---------------------------------------------------------------------------
def fit_bank(
    mat: pd.DataFrame,
    train_models: list[str],
    bank_dir: Path,
    chunk_size: int,
    ncycles: int,
    workers: int,
    fit_r: Path,
    link_r: Path,
) -> list:
    train = mat.loc[train_models]
    train = train.loc[:, train.nunique() > 1]
    train = train.loc[train.nunique(axis=1) > 1]
    kept = list(train.columns)
    ts.write_atlas(train, kept, bank_dir / "data" / "response_matrix_train.csv")
    ends = ts.chunk_ends(len(kept), chunk_size)
    (bank_dir / "data").mkdir(parents=True, exist_ok=True)
    (bank_dir / "data" / "chunk_ends.txt").write_text(",".join(map(str, ends)) + "\n")
    ts.fit_and_link(bank_dir, ends, ncycles, workers, fit_r, link_r)
    return kept


def diagnose_bank(
    mat: pd.DataFrame,
    bank_dir: Path,
    kept: list,
    test_models: list[str],
    ses: list[float],
) -> dict:
    """p-IRT Fisher-info CAT of a fitted bank on a held-out test set (mirrors
    ts.diagnose but takes an in-memory test set with the bank's kept columns)."""
    test = mat.loc[test_models, kept].astype(float).copy()
    test.columns = list(range(1, len(kept) + 1))  # match write_atlas renaming
    idxs, a, b, c = ts.load_bank(bank_dir / "calibration", list(test.columns))
    n_bank = len(idxs)
    se_floor = min(min(ses), 0.15)
    traces, actual = {}, {}
    for mid in test.index:
        resp = test.loc[mid, idxs].to_numpy(float)
        actual[mid] = float(resp.mean())
        traces[mid] = ts.full_cat_traces(resp, a, b, c, se_floor=se_floor)
    out = {}
    for se in ses:
        n_items, preds, acts = [], [], []
        for mid, steps in traces.items():
            chosen = None
            for ni, s, pr in steps:
                if ni >= ts.MIN_ITEMS and s <= se:
                    chosen = (ni, pr)
                    break
            if chosen is None:
                chosen = (steps[-1][0], steps[-1][2])
            n_items.append(chosen[0])
            preds.append(chosen[1])
            acts.append(actual[mid])
        preds = np.asarray(preds)
        acts = np.asarray(acts)
        r = float(np.corrcoef(preds, acts)[0, 1]) if preds.std() > 0 else float("nan")
        out[se] = {
            "corr": round(r, 4),
            "mae": round(float(np.mean(np.abs(preds - acts))), 4),
            "mean_items": round(float(np.mean(n_items)), 2),
            "median_items": float(np.median(n_items)),
            "pct_items": (
                round(100 * float(np.mean(n_items)) / n_bank, 2) if n_bank else float("nan")
            ),
            "n_bank_items": n_bank,
        }
    return out


# ---------------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------------
def run_bench(args: argparse.Namespace) -> list[dict]:
    bench = args.bench
    work = (Path(args.work_root) if args.work_root else OUT_ROOT / "_work") / bench
    fit_r, link_r = ts.write_helper_scripts(work)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    ses = [float(s) for s in args.se_list.split(",")]

    print(f"[{bench}] building master matrix...", flush=True)
    mat = ts.build_master(bench)
    pmap = load_param_map()
    print(f"[{bench}] matrix models={len(mat)} items={mat.shape[1]}", flush=True)

    rows: list[dict] = []
    for seed in range(args.seeds):
        sel = select_bands(mat, pmap, args.low_max, args.high_min, args.n_test, seed)
        print(
            f"[{bench}] seed={seed} LOW_total={sel['n_low_total']} "
            f"HIGH_total={sel['n_high_total']} matched_calib_N={sel['n_calib']} "
            f"n_test={args.n_test}",
            flush=True,
        )
        if sel["n_calib"] < args.min_calib:
            print(
                f"[{bench}] seed={seed} SKIP: matched calib N={sel['n_calib']} "
                f"< min_calib={args.min_calib} (band too thin)",
                flush=True,
            )
            continue

        banks = {}  # band -> (bank_dir, kept)
        for band, train in [("LOW", sel["calib_low"]), ("HIGH", sel["calib_high"])]:
            bank_dir = work / f"seed{seed}" / f"bank_{band}"
            t0 = time.time()
            kept = fit_bank(
                mat, train, bank_dir, args.chunk_size, args.ncycles, args.workers, fit_r, link_r
            )
            banks[band] = (bank_dir, kept)
            print(
                f"[{bench}] seed={seed} fit bank_{band} "
                f"({len(train)} models, {len(kept)} items) {round(time.time() - t0, 1)}s",
                flush=True,
            )

        targets = {
            "LOW->HIGH": ("LOW", sel["test_high"]),
            "HIGH->LOW": ("HIGH", sel["test_low"]),
            "LOW->LOW": ("LOW", sel["test_low"]),
            "HIGH->HIGH": ("HIGH", sel["test_high"]),
        }
        for cond, (band, test_models) in targets.items():
            bank_dir, kept = banks[band]
            metrics = diagnose_bank(mat, bank_dir, kept, test_models, ses)
            for se in ses:
                m = metrics[se]
                rows.append(
                    {
                        "benchmark": bench,
                        "condition": cond,
                        "se_target": se,
                        "seed": seed,
                        "corr": m["corr"],
                        "mae": m["mae"],
                        "mean_items": m["mean_items"],
                        "median_items": m["median_items"],
                        "pct_items": m["pct_items"],
                        "n_calib": sel["n_calib"],
                        "n_test": len(test_models),
                        "n_bank_items": m["n_bank_items"],
                        "low_max_b": args.low_max,
                        "high_min_b": args.high_min,
                    }
                )
            main = metrics[ses[0]]
            print(
                f"[{bench}] seed={seed} {cond:<10} SE{ses[0]}: r={main['corr']} "
                f"MAE={main['mae']} items={main['mean_items']} "
                f"bank={main['n_bank_items']}",
                flush=True,
            )
        _merge_results(rows)  # incremental save

    _merge_results(rows)
    return rows


def _merge_results(new_rows: list[dict]) -> None:
    """Upsert new_rows into the shared results CSV keyed by (bench,cond,se,seed)."""
    path = OUT_ROOT / "param_extrapolation_results.csv"
    existing = []
    if path.exists():
        existing = pd.read_csv(path).to_dict("records")

    def key(r: dict) -> tuple:
        return (r["benchmark"], r["condition"], float(r["se_target"]), int(r["seed"]))

    merged = {key(r): r for r in existing}
    for r in new_rows:
        merged[key(r)] = r
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in sorted(merged.values(), key=key):
            w.writerow({k: r.get(k) for k in FIELDS})


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_all(primary_se: float = 0.3) -> None:
    path = OUT_ROOT / "param_extrapolation_results.csv"
    if not path.exists():
        print("no results CSV found")
        return
    df = pd.read_csv(path)
    df = df[(df["se_target"] == primary_se) & df["corr"].notna()]
    if df.empty:
        print(f"no rows at SE={primary_se}")
        return
    plt = ts._mpl()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    colors = {
        "LOW->LOW": "#1f77b4",
        "HIGH->HIGH": "#2ca02c",
        "LOW->HIGH": "#ff7f0e",
        "HIGH->LOW": "#d62728",
    }
    order = ["LOW->LOW", "HIGH->HIGH", "LOW->HIGH", "HIGH->LOW"]

    benches = sorted(df["benchmark"].unique())
    for bench in benches:
        g = df[df["benchmark"] == bench]
        _grouped_bar(
            plt,
            g,
            order,
            colors,
            f"OpenLM {bench} — param-band extrapolation (SE<={primary_se:g})",
            OUT_ROOT / f"{bench}_param_extrapolation.png",
        )

    # combined: benches on x, condition as grouped bars (mean over seeds)
    fig, ax = plt.subplots(figsize=(1.8 * len(benches) + 3, 5.2))
    width = 0.2
    x = np.arange(len(benches))
    for i, cond in enumerate(order):
        means, sds = [], []
        for bench in benches:
            vals = df[(df["benchmark"] == bench) & (df["condition"] == cond)]["corr"]
            means.append(vals.mean() if len(vals) else np.nan)
            sds.append(vals.std(ddof=0) if len(vals) > 1 else 0.0)
        ax.bar(
            x + (i - 1.5) * width, means, width, yerr=sds, capsize=3, color=colors[cond], label=cond
        )
    ax.set_xticks(x)
    ax.set_xticklabels(benches)
    ax.set_ylabel("Pearson r (p-IRT pred vs actual)")
    ax.set_title(f"Param-band extrapolation across benchmarks (SE<={primary_se:g})")
    ax.set_ylim(0, 1.02)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="calib->test")
    fig.tight_layout()
    png = OUT_ROOT / "param_extrapolation_combined.png"
    fig.savefig(png, dpi=140)
    plt.close(fig)
    print(f"wrote {png}")


def _grouped_bar(plt, g, order, colors, title, png) -> None:
    means = [g[g["condition"] == c]["corr"].mean() for c in order]
    sds = [
        g[g["condition"] == c]["corr"].std(ddof=0) if len(g[g["condition"] == c]) > 1 else 0.0
        for c in order
    ]
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    bars = ax.bar(order, means, yerr=sds, capsize=4, color=[colors[c] for c in order])
    for bar, m in zip(bars, means, strict=False):
        if np.isfinite(m):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                m + 0.01,
                f"{m:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    ax.set_ylabel("Pearson r (p-IRT pred vs actual)")
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(png, dpi=140)
    plt.close(fig)
    print(f"wrote {png}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--bench", help="OpenLM benchmark (ifeval/math/bbh/gpqa)")
    p.add_argument("--plot", action="store_true", help="(re)build figures from CSV")
    p.add_argument("--low-max", type=float, default=2.0, help="LOW band: params_b <= this")
    p.add_argument("--high-min", type=float, default=3.5, help="HIGH band: params_b >= this")
    p.add_argument("--n-test", type=int, default=30, help="held-out test models per band")
    p.add_argument("--min-calib", type=int, default=100, help="skip seed if matched N below")
    p.add_argument("--seeds", type=int, default=3, help="number of seeds (0..seeds-1)")
    p.add_argument("--chunk-size", type=int, default=100)
    p.add_argument("--ncycles", type=int, default=500)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--se-list", default="0.3,0.2")
    p.add_argument("--work-root", default="")
    args = p.parse_args()

    if args.plot:
        plot_all()
        return
    if not args.bench:
        p.error("--bench required (or use --plot)")
    run_bench(args)
    plot_all()


if __name__ == "__main__":
    main()
