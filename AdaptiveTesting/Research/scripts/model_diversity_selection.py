#!/usr/bin/env python3
"""Calibration-model SELECTION sweep: can smart/diverse model choice reach the same
held-out correlation with fewer calibration models than random sampling?

We reuse the validated OpenLM 3PL pipeline in `openlm_trainsize_sweep.py` (chunked R
`mirt` 3PL fit, mean-sigma chunk linking with polarity flip, a>0 item filter, EAP/Fisher
p-IRT CAT accuracy recovery). The ONLY thing changed here is HOW the N calibration models
are chosen from the train pool. The held-out test set is held FIXED (same seed-7 10% split
as the train-size sweep) across every strategy and every N, so all curves are comparable.

Selection strategies (each fits its own item bank, then runs the SAME held-out CAT):
  - random        : uniform k-subset of the train pool; several seeds -> mean +/- sd band.
  - ability_spread: pick N models that evenly cover the observed benchmark-accuracy range
                    (nearest models to N evenly spaced accuracy quantiles). Tests the
                    "ability spread pins item parameters" hypothesis directly. Deterministic.
  - diverse       : farthest-point (k-center greedy) on the model-by-item correctness matrix
                    using Hamming distance, to maximize behavioral coverage. Deterministic.

For each strategy x N we record Pearson r (p-IRT predicted vs actual full-benchmark
accuracy on the held-out models), MAE, and mean # CAT items, per SE stopping target.

Outputs go to AdaptiveTesting/Research/01_MCQ_ATLAS/data/model_diversity_selection/:
  results.csv                    (benchmark, strategy, N, seed, se_target, r, mae, items, n_bank)
  selection_curve_<bench>.png    (r vs N per strategy, random error band, plateau line + hits)
  selection_curve_combined.png   (all benchmarks)
  README.md                      (curves, "same r with X fewer models" numbers, verdict)

Usage:
  uv run python AdaptiveTesting/Research/scripts/model_diversity_selection.py \
      --benches ifeval,math --n-list 10,20,30,50,80,120 --seeds 0,1,2,3,4 \
      --se-list 0.3,0.2 --jobs 6 --workers 2
  uv run python AdaptiveTesting/Research/scripts/model_diversity_selection.py --plot-only
"""

from __future__ import annotations

import argparse
import csv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import openlm_trainsize_sweep as sweep
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
REPO = SCRIPTS.parents[2]
OUT_ROOT = REPO / "AdaptiveTesting/Research/01_MCQ_ATLAS/data/model_diversity_selection"

STRATEGIES = ("random", "ability_spread", "diverse")
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
    "n_test",
    "calib_s",
]


# ---------------------------------------------------------------------------
# Selection strategies (operate on the FIXED train pool only)
# ---------------------------------------------------------------------------
def select_random(pool: list[str], n: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    return [pool[i] for i in rng.choice(len(pool), size=n, replace=False)]


def select_ability_spread(mat: pd.DataFrame, pool: list[str], n: int) -> list[str]:
    """Pick N models nearest to N evenly spaced quantiles of per-model mean accuracy,
    so the chosen set spans the full benchmark-accuracy range as evenly as possible."""
    acc = mat.loc[pool].mean(axis=1)
    vals = acc.to_numpy(float)
    names = list(acc.index)
    targets = np.quantile(vals, np.linspace(0.0, 1.0, n))
    chosen: list[str] = []
    used: set[str] = set()
    for q in targets:
        for idx in np.argsort(np.abs(vals - q)):
            m = names[idx]
            if m not in used:
                used.add(m)
                chosen.append(m)
                break
    return chosen


def select_diverse(mat: pd.DataFrame, pool: list[str], n: int) -> list[str]:
    """Farthest-point / k-center greedy on the model-by-item correctness matrix (Hamming
    distance). Start from the medoid (closest to the mean response vector) for determinism,
    then repeatedly add the model farthest from the current selected set."""
    x = mat.loc[pool].to_numpy(float)
    center = x.mean(axis=0)
    first = int(np.argmin(np.abs(x - center).sum(axis=1)))
    chosen = [first]
    dist = np.abs(x - x[first]).sum(axis=1)
    while len(chosen) < n:
        nxt = int(np.argmax(dist))
        if nxt in chosen:
            break
        chosen.append(nxt)
        dist = np.minimum(dist, np.abs(x - x[nxt]).sum(axis=1))
    return [pool[i] for i in chosen]


def select(strategy: str, mat: pd.DataFrame, pool: list[str], n: int, seed: int) -> list[str]:
    if strategy == "random":
        return select_random(pool, n, seed)
    if strategy == "ability_spread":
        return select_ability_spread(mat, pool, n)
    if strategy == "diverse":
        return select_diverse(mat, pool, n)
    raise ValueError(f"unknown strategy {strategy}")


# ---------------------------------------------------------------------------
# One (strategy, seed, N) calibration + held-out diagnosis
# ---------------------------------------------------------------------------
def run_one(args, bench, mat, test_models, pool, fit_r, link_r, strategy, seed, n, ses):
    work = _work_root(args, bench)
    tag = f"{strategy}_s{seed}_n{n}"
    step_dir = work / tag
    t0 = time.time()
    train_models = select(strategy, mat, pool, n, seed)
    ends, _ = sweep.prepare_step(mat, train_models, test_models, step_dir, args.chunk_size)
    sweep.fit_and_link(step_dir, ends, args.ncycles, args.workers, fit_r, link_r)
    metrics = sweep.diagnose(step_dir, ses)
    calib_s = round(time.time() - t0, 1)
    rows = []
    for se in ses:
        m = metrics[se]
        rows.append(
            {
                "benchmark": bench,
                "strategy": strategy,
                "N": n,
                "seed": seed,
                "se_target": se,
                "r": m["corr"],
                "mae": m["mae"],
                "items": m["mean_items"],
                "n_bank_items": m["n_bank_items"],
                "n_test": len(test_models),
                "calib_s": calib_s,
            }
        )
    return rows


def _work_root(args, bench) -> Path:
    root = Path(args.work_root) if args.work_root else OUT_ROOT / "_work"
    return root / bench


def build_tasks(strategies, seeds, ns):
    """(strategy, seed, N). Deterministic strategies use a single seed (0)."""
    tasks = []
    for strat in strategies:
        strat_seeds = seeds if strat == "random" else [0]
        for seed in strat_seeds:
            for n in ns:
                tasks.append((strat, seed, n))
    return tasks


def run_bench(args, bench: str, ses: list[float]) -> list[dict]:
    work = _work_root(args, bench)
    fit_r, link_r = sweep.write_helper_scripts(work)
    print(f"[{bench}] building master matrix...", flush=True)
    mat = sweep.build_master(bench)
    test_models, train_pool = sweep.split_models(mat, args.seed, args.test_frac)
    ns = [n for n in args.ns if n <= len(train_pool)]
    print(
        f"[{bench}] models={len(mat)} items={mat.shape[1]} n_test={len(test_models)} "
        f"train_pool={len(train_pool)} N={ns}",
        flush=True,
    )

    strategies = [s for s in STRATEGIES if s in args.strategies]
    tasks = build_tasks(strategies, args.seeds, ns)
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool_ex:
        futs = {
            pool_ex.submit(
                run_one,
                args,
                bench,
                mat,
                test_models,
                train_pool,
                fit_r,
                link_r,
                strat,
                seed,
                n,
                ses,
            ): (strat, seed, n)
            for (strat, seed, n) in tasks
        }
        for fut in as_completed(futs):
            strat, seed, n = futs[fut]
            try:
                out = fut.result()
                rows.extend(out)
                main = out[0]
                print(
                    f"[{bench}] {strat:<14} seed={seed} N={n:<4} "
                    f"bank={main['n_bank_items']:<4} SE{ses[0]}: r={main['r']} "
                    f"items={main['items']} MAE={main['mae']} ({main['calib_s']}s)",
                    flush=True,
                )
            except Exception as exc:  # keep sweeping past a bad fit
                print(f"[{bench}] {strat} seed={seed} N={n} FAILED: {exc}", flush=True)
                for se in ses:
                    rows.append(
                        {
                            "benchmark": bench,
                            "strategy": strat,
                            "N": n,
                            "seed": seed,
                            "se_target": se,
                            "r": float("nan"),
                            "mae": float("nan"),
                            "items": float("nan"),
                            "n_bank_items": 0,
                            "n_test": len(test_models),
                            "calib_s": float("nan"),
                        }
                    )
            _write_results(rows)
    return rows


# ---------------------------------------------------------------------------
# Results IO
# ---------------------------------------------------------------------------
def _write_results(rows: list[dict]) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    path = OUT_ROOT / "results.csv"
    existing = []
    if path.exists():
        existing = pd.read_csv(path).to_dict("records")

    # merge on (benchmark, strategy, N, seed, se_target)
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


# ---------------------------------------------------------------------------
# Analysis: plateau + model savings
# ---------------------------------------------------------------------------
def _random_plateau(df_bench: pd.DataFrame):
    """Random mean r per N; plateau = mean r at the largest N, noise = its sd."""
    rnd = df_bench[df_bench["strategy"] == "random"]
    g = rnd.groupby("N")["r"].agg(["mean", "std", "count"]).reset_index()
    g = g.sort_values("N")
    plateau_n = int(g["N"].iloc[-1])
    plateau = float(g["mean"].iloc[-1])
    noise = float(g["std"].iloc[-1]) if np.isfinite(g["std"].iloc[-1]) else 0.0
    return g, plateau_n, plateau, noise


def _first_n_at(df_strat: pd.DataFrame, thresh: float):
    """Smallest N whose r >= thresh (mean over seeds), else None."""
    g = df_strat.groupby("N")["r"].mean().reset_index().sort_values("N")
    hit = g[g["r"] >= thresh]
    return int(hit["N"].iloc[0]) if len(hit) else None


def _interp_random_n(rand_g: pd.DataFrame, thresh: float):
    """N at which the random mean curve first crosses thresh (linear interp)."""
    xs = rand_g["N"].to_numpy(float)
    ys = rand_g["mean"].to_numpy(float)
    for i in range(len(xs)):
        if ys[i] >= thresh:
            if i == 0:
                return float(xs[0])
            x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
            if y1 == y0:
                return float(x1)
            return float(x0 + (thresh - y0) * (x1 - x0) / (y1 - y0))
    return None


def analyze(df: pd.DataFrame, se: float) -> dict:
    out = {}
    for bench, gb in df[df["se_target"] == se].groupby("benchmark"):
        gb = gb[np.isfinite(gb["r"])]
        rand_g, plateau_n, plateau, noise = _random_plateau(gb)
        thresh = plateau - noise  # "within noise" of random's plateau
        info = {
            "plateau_n": plateau_n,
            "plateau_r": round(plateau, 4),
            "noise_sd": round(noise, 4),
            "thresh": round(thresh, 4),
            "strategies": {},
        }
        rand_cross = _interp_random_n(rand_g, thresh)
        for strat in ("ability_spread", "diverse"):
            ds = gb[gb["strategy"] == strat]
            if not len(ds):
                continue
            n_hit = _first_n_at(ds, thresh)
            savings = round(rand_cross / n_hit, 2) if (n_hit and rand_cross) else None
            info["strategies"][strat] = {
                "n_hit": n_hit,
                "rand_cross_n": (round(rand_cross, 1) if rand_cross else None),
                "savings_factor": savings,
                "r_at_hit": (round(float(ds[ds["N"] == n_hit]["r"].mean()), 4) if n_hit else None),
            }
        out[bench] = info
    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
COLORS = {"random": "#7f7f7f", "ability_spread": "#1f77b4", "diverse": "#d62728"}
LABELS = {
    "random": "random (mean +/- sd)",
    "ability_spread": "ability spread",
    "diverse": "response diversity",
}


def _plot_bench(plt, df: pd.DataFrame, bench: str, se: float, info: dict, path: Path):
    gb = df[(df["benchmark"] == bench) & (df["se_target"] == se) & np.isfinite(df["r"])]
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    rnd = (
        gb[gb["strategy"] == "random"]
        .groupby("N")["r"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("N")
    )
    ax.plot(rnd["N"], rnd["mean"], "o-", color=COLORS["random"], label=LABELS["random"], zorder=3)
    sd = rnd["std"].fillna(0.0)
    ax.fill_between(
        rnd["N"], rnd["mean"] - sd, rnd["mean"] + sd, color=COLORS["random"], alpha=0.2, zorder=1
    )
    for strat in ("ability_spread", "diverse"):
        ds = gb[gb["strategy"] == strat].groupby("N")["r"].mean().reset_index()
        ds = ds.sort_values("N")
        if not len(ds):
            continue
        ax.plot(
            ds["N"],
            ds["r"],
            "s-" if strat == "ability_spread" else "^-",
            color=COLORS[strat],
            label=LABELS[strat],
            zorder=4,
        )
        s = info["strategies"].get(strat, {})
        if s.get("n_hit"):
            ax.scatter(
                [s["n_hit"]],
                [s["r_at_hit"]],
                s=180,
                facecolors="none",
                edgecolors=COLORS[strat],
                linewidths=2.2,
                zorder=5,
            )
    ax.axhline(
        info["plateau_r"],
        color="black",
        ls="--",
        lw=1.0,
        label=f"random plateau r={info['plateau_r']:.3f} (N={info['plateau_n']})",
    )
    ax.set_xlabel("number of calibration models (N)")
    ax.set_ylabel("Pearson r (p-IRT predicted vs actual accuracy)")
    verdict = _bench_headline(bench, info)
    ax.set_title(verdict, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _bench_headline(bench: str, info: dict) -> str:
    best = None
    for strat, s in info["strategies"].items():
        if s.get("savings_factor") and (best is None or s["savings_factor"] > best[1]):
            best = (strat, s["savings_factor"], s["n_hit"], s["rand_cross_n"])
    if best:
        return (
            f"{bench}: {LABELS[best[0]].split(' (')[0]} N={best[2]} matches "
            f"random N~{best[3]:g} ({best[1]:g}x fewer models, SE<=0.3)"
        )
    return f"{bench}: smart selection vs random (SE<=0.3)"


def _plot_combined(
    plt, df: pd.DataFrame, se: float, benches: list[str], analysis: dict, path: Path
):
    n = len(benches)
    fig, axes = plt.subplots(1, n, figsize=(6.4 * n, 5.2), squeeze=False)
    for ax, bench in zip(axes[0], benches, strict=False):
        info = analysis.get(bench)
        if info is None:
            continue
        gb = df[(df["benchmark"] == bench) & (df["se_target"] == se) & np.isfinite(df["r"])]
        rnd = (
            gb[gb["strategy"] == "random"]
            .groupby("N")["r"]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values("N")
        )
        ax.plot(rnd["N"], rnd["mean"], "o-", color=COLORS["random"], label=LABELS["random"])
        sd = rnd["std"].fillna(0.0)
        ax.fill_between(
            rnd["N"], rnd["mean"] - sd, rnd["mean"] + sd, color=COLORS["random"], alpha=0.2
        )
        for strat in ("ability_spread", "diverse"):
            ds = gb[gb["strategy"] == strat].groupby("N")["r"].mean().reset_index()
            ds = ds.sort_values("N")
            if not len(ds):
                continue
            ax.plot(
                ds["N"],
                ds["r"],
                "s-" if strat == "ability_spread" else "^-",
                color=COLORS[strat],
                label=LABELS[strat],
            )
            s = info["strategies"].get(strat, {})
            if s.get("n_hit"):
                ax.scatter(
                    [s["n_hit"]],
                    [s["r_at_hit"]],
                    s=160,
                    facecolors="none",
                    edgecolors=COLORS[strat],
                    linewidths=2.2,
                )
        ax.axhline(info["plateau_r"], color="black", ls="--", lw=1.0)
        ax.set_xlabel("number of calibration models (N)")
        ax.set_ylabel("Pearson r (p-IRT pred vs actual)")
        ax.set_title(_bench_headline(bench, info), fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# README
# ---------------------------------------------------------------------------
def write_readme(df: pd.DataFrame, analysis: dict, se: float, benches: list[str]):
    lines = [
        "# Model-selection sweep: fewer calibration models by choosing them well",
        "",
        "Question: can a smart/diverse choice of calibration models reach the same",
        "held-out correlation as random sampling, but with fewer models?",
        "",
        "Method: same validated OpenLM 3PL pipeline as the train-size sweep",
        "(`openlm_trainsize_sweep.py`: chunked R mirt 3PL, mean-sigma linking, a>0",
        "item filter, EAP/Fisher p-IRT CAT). Only the choice of the N calibration",
        "models changes. The held-out test set is the same fixed seed-7 10% split for",
        f"every strategy and N, so all curves are comparable. Reported at SE<={se:g}.",
        "",
        "Strategies: `random` (5 seeds, mean +/- sd), `ability_spread` (models",
        "covering the accuracy range evenly), `diverse` (farthest-point k-center on the",
        "model-by-item correctness matrix, Hamming distance).",
        "",
        "## Headline numbers",
        "",
    ]
    for bench in benches:
        info = analysis.get(bench)
        if info is None:
            continue
        lines.append(f"### {bench}")
        lines.append("")
        lines.append(
            f"- Random plateau: r = {info['plateau_r']:.3f} at N = "
            f"{info['plateau_n']} (seed sd = {info['noise_sd']:.3f}); "
            f"match threshold r >= {info['thresh']:.3f}."
        )
        for strat in ("ability_spread", "diverse"):
            s = info["strategies"].get(strat)
            if not s:
                continue
            if s["n_hit"] and s["savings_factor"]:
                lines.append(
                    f"- {LABELS[strat].split(' (')[0]}: matches the plateau at N = "
                    f"{s['n_hit']} (r = {s['r_at_hit']:.3f}); random needs "
                    f"N ~ {s['rand_cross_n']:g} for the same r -> "
                    f"{s['savings_factor']:g}x fewer models."
                )
            else:
                lines.append(
                    f"- {LABELS[strat].split(' (')[0]}: did not reach the random "
                    "plateau within the tested N grid."
                )
        lines.append("")
    lines += ["## Verdict", "", _overall_verdict(analysis, benches), "", "## Figures", ""]
    for bench in benches:
        lines.append(
            f"- `selection_curve_{bench}.png`: r vs N per strategy, random "
            "error band, plateau line, and markers where smart strategies hit it."
        )
    lines.append("- `selection_curve_combined.png`: all benchmarks side by side.")
    lines += [
        "",
        "## Data",
        "",
        "- `results.csv`: benchmark, strategy, N, seed, se_target, r, mae, items, n_bank_items.",
        "",
        "## Caveats",
        "",
        "- Small-N 3PL fits (N=10-30 respondents against ~500-1000 items) are",
        "  heavily over-parameterized; item banks shrink as constant items drop.",
        "  Treat the smallest-N points as noisy and read the trend, not single points.",
        "- Smart strategies are deterministic (one curve, no band). Random shows the",
        "  seed spread. Savings factors use linear interpolation of the random mean",
        "  curve to the match threshold.",
        "",
    ]
    (OUT_ROOT / "README.md").write_text("\n".join(lines))


def _overall_verdict(analysis: dict, benches: list[str]) -> str:
    wins = []
    for bench in benches:
        info = analysis.get(bench, {})
        best = None
        for strat, s in info.get("strategies", {}).items():
            if s.get("savings_factor") and (best is None or s["savings_factor"] > best[1]):
                best = (strat, s["savings_factor"])
        if best:
            wins.append((bench, best[0], best[1]))
    if not wins:
        return (
            "Within the tested grid no smart strategy clearly beat random; "
            "correlation is set more by N than by selection here."
        )
    parts = [f"{b}: {LABELS[s].split(' (')[0]} {f:g}x fewer" for b, s, f in wins]
    strat_names = {s for _, s, _ in wins}
    winner = (
        LABELS[list(strat_names)[0]].split(" (")[0]
        if len(strat_names) == 1
        else "diversity-aware selection"
    )
    return (
        f"Yes: choosing calibration models well reaches random's plateau "
        f"correlation with fewer models ({'; '.join(parts)}). "
        f"Best overall: {winner}."
    )


def make_outputs(df: pd.DataFrame, se: float, benches: list[str]) -> dict:
    analysis = analyze(df, se)
    plt = _mpl()
    for bench in benches:
        if bench in analysis:
            _plot_bench(
                plt, df, bench, se, analysis[bench], OUT_ROOT / f"selection_curve_{bench}.png"
            )
    present = [b for b in benches if b in analysis]
    if present:
        _plot_combined(plt, df, se, present, analysis, OUT_ROOT / "selection_curve_combined.png")
    write_readme(df, analysis, se, present)
    return analysis


def _mpl():
    import os

    os.environ.setdefault("MPLCONFIGDIR", str(OUT_ROOT / ".mplcache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--benches", default="ifeval,math")
    p.add_argument("--n-list", default="10,20,30,50,80,120", dest="n_list")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--strategies", default="random,ability_spread,diverse")
    p.add_argument("--se-list", default="0.3,0.2", dest="se_list")
    p.add_argument("--seed", type=int, default=7, help="split seed (held-out test set)")
    p.add_argument("--test-frac", type=float, default=0.10)
    p.add_argument("--chunk-size", type=int, default=100)
    p.add_argument("--ncycles", type=int, default=500)
    p.add_argument("--jobs", type=int, default=6, help="parallel calibration tasks")
    p.add_argument("--workers", type=int, default=2, help="R chunk-fit workers per task")
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
    benches = a.benches.split(",")
    ses = [float(s) for s in a.se_list.split(",")]

    if a.plot_only:
        df = pd.read_csv(OUT_ROOT / "results.csv")
        analysis = make_outputs(df, ses[0], benches)
        print(f"[plot-only] wrote figures + README to {OUT_ROOT}")
        _print_summary(analysis, benches)
        return

    for bench in benches:
        run_bench(a, bench, ses)

    df = pd.read_csv(OUT_ROOT / "results.csv")
    analysis = make_outputs(df, ses[0], benches)
    print(f"\nDONE -> {OUT_ROOT}")
    _print_summary(analysis, benches)


def _print_summary(analysis: dict, benches: list[str]) -> None:
    print("\n=== model-savings summary (SE<=0.3) ===")
    for bench in benches:
        info = analysis.get(bench)
        if not info:
            continue
        print(
            f"[{bench}] random plateau r={info['plateau_r']} at N={info['plateau_n']} "
            f"(thresh r>={info['thresh']})"
        )
        for strat, s in info["strategies"].items():
            if s.get("savings_factor"):
                print(
                    f"    {strat}: N={s['n_hit']} matches random N~{s['rand_cross_n']} "
                    f"-> {s['savings_factor']}x fewer"
                )
            else:
                print(f"    {strat}: no plateau match within tested N")


if __name__ == "__main__":
    main()
