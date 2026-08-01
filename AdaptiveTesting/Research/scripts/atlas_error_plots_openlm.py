#!/usr/bin/env python3
"""Replicate ATLAS's error analyses/figures on OpenLM benchmarks.

ATLAS (Inputs/ATLAS) reports its p-IRT accuracy recovery with one canonical
per-(benchmark, SE) figure -- a 2x2 panel produced by
``scripts/analysis/compare_pirt_actual.r``:

  (1) p-IRT predicted vs actual accuracy scatter (y=x, linear fit, Pearson r, RMSE)
  (2) error distribution histogram (error = p-IRT - actual; mean, MAE)
  (3) absolute error vs #items administered (subset size)
  (4) error vs actual accuracy

and two cross-benchmark error-vs-SE summaries encoded in
``summary_pirt_mae_sd_se.csv`` / ``summary_theta_mae_sd_se.csv`` /
``rmse_cat_summary.csv``:

  - accuracy MAE (and SD) as a function of the SE stopping threshold
  - mean #items administered as a function of the SE stopping threshold
    (ATLAS adaptive vs a random-item baseline; see the atlas_*_random dirs)
  - theta (ability) recovery error vs the SE threshold

This script reproduces all four plot types on the OpenLM per-question data using
the SAME published-ATLAS-style 3PL bank + Fisher-information p-IRT CAT already
calibrated in this repo (Experiments/openlm_atlas_3pl/<bench>/calibration and
Experiments/openlm_gpqa_atlas_3pl/calibration). It reuses those frozen banks +
held-out test matrices (single calibration, fixed split/seed, no shuffles) and
the CAT/EAP conventions from Research/scripts/se_sweep.py & mcq_diagnostic.py.

For each held-out model we run ONE adaptive CAT to a tight SE floor, storing the
per-step trace, then read off (#items, theta, p-IRT prediction, SE) at each SE
target -- one fit, many thresholds -> self-consistent curves. Ground-truth theta
is the full-bank EAP theta (analogous to ATLAS's full-test WLE theta). The random
baseline administers items in a fixed shuffled order and records #items to reach
each SE.

Usage:
  export PYTHONPATH=/Users/arhant/Documents/EDLM/olmo-eval-full/eduLLM-Evals
  uv run python AdaptiveTesting/Research/scripts/atlas_error_plots_openlm.py \
      --benches ifeval,gpqa,math,bbh,musr \
      --ses 0.1,0.2,0.3 --seed 7 \
      --out-dir AdaptiveTesting/Research/01_MCQ_ATLAS/data/atlas_replication \
      --fig-dir AdaptiveTesting/Research/01_MCQ_ATLAS/figures

Outputs (mirroring ATLAS column/file conventions):
  <out-dir>/<bench>/pirt_vs_actual_se_<SE>.csv   (model, pirt_accuracy, theta,
      n_subset_items, n_all_items, actual_accuracy, error, abs_error, squared_error)
  <out-dir>/<bench>/pirt_vs_actual_se_<SE>.png   (ATLAS 2x2 error figure)
  <out-dir>/<bench>/theta_recovery_se_<SE>.csv   (theta_cat vs theta_full)
  <out-dir>/<bench>/items_adaptive_vs_random.csv (per-model #items by SE)
  <out-dir>/summary_pirt_mae_sd_se.csv           (accuracy MAE/SD/SE x bench x SE)
  <out-dir>/summary_theta_mae_sd_se.csv          (theta MAE/SD/SE x bench x SE)
  <out-dir>/summary_items_adaptive_vs_random.csv (mean items adaptive vs random)
  <fig-dir>/atlasrep_mae_vs_se.png               (accuracy MAE vs SE, all benches)
  <fig-dir>/atlasrep_items_vs_se.png             (mean #items vs SE, adaptive vs random)
  <fig-dir>/atlasrep_theta_mae_vs_se.png         (theta MAE vs SE, all benches)
  <fig-dir>/atlasrep_<bench>_pirt_vs_actual_se_<SE>.png  (copy of per-bench figure)
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

csv.field_size_limit(sys.maxsize)

REPO = Path(__file__).resolve().parents[3]
EXP = REPO / "AdaptiveTesting/Experiments"

NODES = np.linspace(-4.0, 4.0, 81)
PRIORW = np.exp(-0.5 * NODES**2)
PRIORW /= PRIORW.sum()
MIN_ITEMS = 8

# bench -> (combined bank csv, held-out test matrix csv)
BENCH_PATHS = {
    "ifeval": (EXP / "openlm_atlas_3pl/ifeval/calibration/irt_item_parameters_combined.csv",
               EXP / "openlm_atlas_3pl/ifeval/data/response_matrix_test.csv"),
    "math": (EXP / "openlm_atlas_3pl/math/calibration/irt_item_parameters_combined.csv",
             EXP / "openlm_atlas_3pl/math/data/response_matrix_test.csv"),
    "bbh": (EXP / "openlm_atlas_3pl/bbh/calibration/irt_item_parameters_combined.csv",
            EXP / "openlm_atlas_3pl/bbh/data/response_matrix_test.csv"),
    "musr": (EXP / "openlm_atlas_3pl/musr/calibration/irt_item_parameters_combined.csv",
             EXP / "openlm_atlas_3pl/musr/data/response_matrix_test.csv"),
    "gpqa": (EXP / "openlm_gpqa_atlas_3pl/calibration/irt_item_parameters_combined.csv",
             EXP / "openlm_gpqa_atlas_3pl/data/gpqa_response_matrix_test.csv"),
}


def prob(theta, a, b, c):
    z = np.clip(a * (theta - b), -30, 30)
    return np.clip(c + (1 - c) / (1.0 + np.exp(-z)), 1e-6, 1 - 1e-6)


def eap_se(resp, a, b, c):
    z = np.clip(a[None, :] * (NODES[:, None] - b[None, :]), -30, 30)
    p = np.clip(c[None, :] + (1 - c[None, :]) / (1.0 + np.exp(-z)), 1e-6, 1 - 1e-6)
    ll = (resp[None, :] * np.log(p) + (1 - resp)[None, :] * np.log(1 - p)).sum(axis=1)
    w = np.exp(ll - ll.max()) * PRIORW
    w /= w.sum()
    m = float((NODES * w).sum())
    sd = float(np.sqrt(((NODES - m) ** 2 * w).sum()))
    return m, sd


def pirt_pred(resp_all, order, theta, a, b, c):
    """p-IRT accuracy: observed on administered items + IRT-predicted on the rest."""
    n = len(a)
    if not order:
        return float(prob(theta, a, b, c).mean())
    subset = set(order)
    avg_obs = float(resp_all[np.asarray(order)].mean())
    unobs = [i for i in range(n) if i not in subset]
    avg_pred = float(prob(theta, a[unobs], b[unobs], c[unobs]).mean()) if unobs else avg_obs
    w = len(order) / n
    return w * avg_obs + (1 - w) * avg_pred


def adaptive_trace(resp_all, a, b, c, floor_se, max_items):
    """One full Fisher-info CAT to a tight floor; per-step (n, se, theta, pred)."""
    n = len(a)
    used = np.zeros(n, bool)
    theta = 0.0
    order: list[int] = []
    steps = []
    for _ in range(min(max_items, n)):
        p = prob(theta, a, b, c)
        info = (a**2) * ((1 - p) / p) * ((p - c) / (1 - c + 1e-9)) ** 2
        info[used] = -1.0
        j = int(np.argmax(info))
        used[j] = True
        order.append(j)
        idx = np.asarray(order)
        theta, se = eap_se(resp_all[idx], a[idx], b[idx], c[idx])
        steps.append((len(order), se, theta, pirt_pred(resp_all, order, theta, a, b, c)))
        if se <= floor_se and len(order) >= MIN_ITEMS:
            break
    return steps


def random_trace_items(resp_all, a, b, c, rng, ses, max_items):
    """Random-item order CAT; #items to first reach each SE target."""
    n = len(a)
    perm = rng.permutation(n)
    order: list[int] = []
    reached = {se: None for se in ses}
    for step, j in enumerate(perm[:max_items], 1):
        order.append(int(j))
        idx = np.asarray(order)
        _, se = eap_se(resp_all[idx], a[idx], b[idx], c[idx])
        for target in ses:
            if reached[target] is None and step >= MIN_ITEMS and se <= target:
                reached[target] = step
    for target in ses:
        if reached[target] is None:
            reached[target] = min(max_items, n)
    return reached


def read_off(steps, se_target):
    """First trace step meeting SE<=target (>=MIN_ITEMS); else full-CAT endpoint."""
    for (ni, se, theta, pred) in steps:
        if ni >= MIN_ITEMS and se <= se_target:
            return ni, se, theta, pred
    ni, se, theta, pred = steps[-1]
    return ni, se, theta, pred


def load_bank(bank_csv: Path, mat_cols: list[int]):
    by_idx: dict[int, tuple[float, float, float]] = {}
    pdf = pd.read_csv(bank_csv)
    name_col = pdf.columns[0]
    for _, row in pdf.iterrows():
        key = str(row[name_col]).lstrip("X")
        try:
            idx = int(float(key))
        except ValueError:
            continue
        a = float(row["a1"])
        d = float(row["d"])
        g = float(row["g"]) if "g" in pdf.columns else 0.0
        # Drop non-positive discriminations (mirt calibration failures / polarity
        # flips). This matches the canonical gpqa loader
        # (Experiments/openlm_gpqa_atlas_3pl/diagnostic_validation.py); keeping
        # them collapses the CAT (e.g. gpqa has 613/1192 negative-a items).
        if not np.isfinite(a) or a <= 0:
            continue
        by_idx[idx] = (a, -d / a, float(np.clip(g, 0.0, 0.999)))
    idxs, a, b, c = [], [], [], []
    for idx in mat_cols:
        if idx not in by_idx:
            continue
        aa, bb, cc = by_idx[idx]
        idxs.append(idx)
        a.append(aa)
        b.append(bb)
        c.append(cc)
    return idxs, np.asarray(a), np.asarray(b), np.asarray(c)


def load_matrix(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={df.columns[0]: "model"}).set_index("model")
    df.columns = [int(c) for c in df.columns]
    return df.astype(float)


def panel_figure(df: pd.DataFrame, bench: str, se: float, out_png: Path) -> dict:
    """Reproduce ATLAS's 2x2 error figure (compare_pirt_actual.r)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    act = df["actual_accuracy"].to_numpy()
    pir = df["pirt_accuracy"].to_numpy()
    err = df["error"].to_numpy()
    abserr = df["abs_error"].to_numpy()
    nsub = df["n_subset_items"].to_numpy()
    r = float(np.corrcoef(pir, act)[0, 1])
    rmse = float(np.sqrt(np.mean((pir - act) ** 2)))
    mae = float(np.mean(abserr))
    me = float(np.mean(err))

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))

    ax = axes[0, 0]
    lo = min(act.min(), pir.min()) - 0.02
    hi = max(act.max(), pir.max()) + 0.02
    ax.scatter(act, pir, s=22, color="black", alpha=0.35)
    ax.plot([lo, hi], [lo, hi], "--", color="red", lw=2, label="Perfect prediction")
    slope, intercept = np.polyfit(act, pir, 1)
    xs = np.array([lo, hi])
    ax.plot(xs, slope * xs + intercept, "-", color="blue", lw=2, label="Linear fit")
    ax.set_xlabel("Actual Accuracy")
    ax.set_ylabel("p-IRT Accuracy")
    ax.set_title(f"p-IRT vs Actual Accuracy\n{bench} (SE={se:g})")
    ax.plot([], [], " ", label=f"r = {r:.4f}")
    ax.plot([], [], " ", label=f"RMSE = {rmse:.4f}")
    ax.legend(loc="upper left", frameon=False, fontsize=8)

    ax = axes[0, 1]
    ax.hist(err, bins=30, color="lightblue", edgecolor="white",
            range=(-0.3, 0.3))
    ax.axvline(0, color="red", lw=2, ls="--")
    ax.axvline(me, color="blue", lw=2)
    ax.set_xlabel("Error (p-IRT - Actual)")
    ax.set_ylabel("Frequency")
    ax.set_title("Error Distribution")
    ax.text(0.97, 0.95, f"Mean = {me:.5f}\nMAE = {mae:.5f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9)

    ax = axes[1, 0]
    ax.scatter(nsub, abserr, s=22, color="black", alpha=0.35)
    if len(df) > 10 and np.ptp(nsub) > 0:
        o = np.argsort(nsub)
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess

            sm = lowess(abserr, nsub, frac=0.6, return_sorted=True)
            ax.plot(sm[:, 0], sm[:, 1], color="blue", lw=2)
        except Exception:
            k = max(3, len(df) // 8)
            cs = np.convolve(abserr[o], np.ones(k) / k, mode="valid")
            ax.plot(nsub[o][k // 2: k // 2 + len(cs)], cs, color="blue", lw=2)
    ax.set_xlabel("Subset Size")
    ax.set_ylabel("Absolute Error")
    ax.set_title("Absolute Error vs Subset Size")

    ax = axes[1, 1]
    ax.scatter(act, err, s=22, color="black", alpha=0.35)
    ax.axhline(0, color="red", lw=2, ls="--")
    if len(df) > 10 and np.ptp(act) > 0:
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess

            sm = lowess(err, act, frac=0.6, return_sorted=True)
            ax.plot(sm[:, 0], sm[:, 1], color="blue", lw=2)
        except Exception:
            o = np.argsort(act)
            k = max(3, len(df) // 8)
            cs = np.convolve(err[o], np.ones(k) / k, mode="valid")
            ax.plot(act[o][k // 2: k // 2 + len(cs)], cs, color="blue", lw=2)
    ax.set_xlabel("Actual Accuracy")
    ax.set_ylabel("Error (p-IRT - Actual)")
    ax.set_title("Error vs Actual Accuracy")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    return {"r": r, "rmse": rmse, "mae": mae, "me": me}


def run_bench(bench: str, ses: list[float], seed: int, floor_se: float,
              max_items: int, out_dir: Path) -> dict:
    bank_csv, test_csv = BENCH_PATHS[bench]
    mat = load_matrix(test_csv)
    idxs, a, b, c = load_bank(bank_csv, list(mat.columns))
    n_all = len(idxs)
    models = list(mat.index)
    resp = {m: mat.loc[m, idxs].to_numpy(float) for m in models}
    actual = {m: float(resp[m].mean()) for m in models}
    # full-bank EAP theta = ground-truth ability (analog of ATLAS full-test WLE theta)
    theta_full = {m: eap_se(resp[m], a, b, c)[0] for m in models}
    print(f"[{bench}] bank={n_all} items, held-out={len(models)} models", flush=True)

    traces = {m: adaptive_trace(resp[m], a, b, c, floor_se, max_items) for m in models}
    rng = np.random.default_rng(seed)
    rand_items = {m: random_trace_items(resp[m], a, b, c, rng, ses, max_items)
                  for m in models}

    bdir = out_dir / bench
    bdir.mkdir(parents=True, exist_ok=True)
    per_se_stats: dict[float, dict] = {}
    theta_stats: dict[float, dict] = {}

    for se in ses:
        rows = []
        theta_rows = []
        for m in models:
            ni, se_reached, theta_cat, pred = read_off(traces[m], se)
            err = pred - actual[m]
            rows.append({
                "model": m, "pirt_accuracy": pred, "theta": theta_cat,
                "n_subset_items": ni, "n_all_items": n_all,
                "actual_accuracy": actual[m], "se_reached": se_reached,
                "error": err, "abs_error": abs(err), "squared_error": err * err,
            })
            theta_rows.append({
                "model": m, "theta_cat": theta_cat, "theta_full": theta_full[m],
                "n_subset_items": ni, "se_reached": se_reached,
                "theta_abs_error": abs(theta_cat - theta_full[m]),
            })
        df = pd.DataFrame(rows)
        df.to_csv(bdir / f"pirt_vs_actual_se_{se:g}.csv", index=False)
        tdf = pd.DataFrame(theta_rows)
        tdf.to_csv(bdir / f"theta_recovery_se_{se:g}.csv", index=False)

        fig_stats = panel_figure(df, bench, se, bdir / f"pirt_vs_actual_se_{se:g}.png")
        abserr = df["abs_error"].to_numpy()
        per_se_stats[se] = {
            "benchmark": bench, "se_target": se,
            "mae": float(np.mean(abserr)), "sd": float(np.std(abserr)),
            "se": float(np.std(abserr) / np.sqrt(len(abserr))),
            "n": len(abserr), "n_subset_items": float(df["n_subset_items"].mean()),
            "n_all_items": n_all, "r": fig_stats["r"], "rmse": fig_stats["rmse"],
        }
        te = tdf["theta_abs_error"].to_numpy()
        theta_stats[se] = {
            "benchmark": bench, "se_target": se,
            "mae": float(np.mean(te)), "sd": float(np.std(te)),
            "se": float(np.std(te) / np.sqrt(len(te))), "n": len(te),
            "n_subset_items": float(tdf["n_subset_items"].mean()), "n_all_items": n_all,
        }
        print(f"[{bench}] SE<={se:g}: acc-MAE={per_se_stats[se]['mae']:.4f} "
              f"r={fig_stats['r']:.3f} items={per_se_stats[se]['n_subset_items']:.1f} "
              f"theta-MAE={theta_stats[se]['mae']:.3f}", flush=True)

    # adaptive vs random #items per model
    item_rows = []
    for m in models:
        row = {"model": m}
        for se in ses:
            ni, _, _, _ = read_off(traces[m], se)
            row[f"adaptive_items_se{se:g}"] = ni
            row[f"random_items_se{se:g}"] = rand_items[m][se]
        item_rows.append(row)
    pd.DataFrame(item_rows).to_csv(bdir / "items_adaptive_vs_random.csv", index=False)

    items_summary = {}
    for se in ses:
        adap = np.array([read_off(traces[m], se)[0] for m in models], float)
        rnd = np.array([rand_items[m][se] for m in models], float)
        items_summary[se] = {
            "benchmark": bench, "se_target": se,
            "mean_adaptive_items": float(adap.mean()),
            "mean_random_items": float(rnd.mean()),
            "savings_ratio": float(rnd.mean() / max(adap.mean(), 1e-9)),
            "n_all_items": n_all,
        }
    return {"pirt": per_se_stats, "theta": theta_stats, "items": items_summary,
            "n_all": n_all, "n_models": len(models)}


def cross_figures(all_stats: dict, ses: list[float], fig_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    benches = list(all_stats.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(benches), 3)))

    # (2) accuracy MAE vs SE
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    for col, bench in zip(colors, benches):
        y = [all_stats[bench]["pirt"][se]["mae"] for se in ses]
        yerr = [all_stats[bench]["pirt"][se]["se"] for se in ses]
        ax.errorbar(ses, y, yerr=yerr, marker="o", color=col, capsize=3, label=bench)
    ax.set_xlabel("SE stopping threshold")
    ax.set_ylabel("Accuracy MAE (|p-IRT - actual|)")
    ax.set_title("OpenLM: p-IRT accuracy MAE vs SE stopping threshold\n"
                 "(ATLAS-style 3PL CAT, held-out models)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "atlasrep_mae_vs_se.png", dpi=140)
    plt.close(fig)

    # (3) mean #items vs SE, adaptive vs random
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    for col, bench in zip(colors, benches):
        adap = [all_stats[bench]["items"][se]["mean_adaptive_items"] for se in ses]
        rnd = [all_stats[bench]["items"][se]["mean_random_items"] for se in ses]
        ax.plot(ses, adap, marker="o", color=col, label=f"{bench} adaptive")
        ax.plot(ses, rnd, marker="s", ls="--", color=col, alpha=0.6,
                label=f"{bench} random")
    ax.set_xlabel("SE stopping threshold")
    ax.set_ylabel("mean # items administered")
    ax.set_title("OpenLM: mean #items to reach SE — adaptive (Fisher) vs random\n"
                 "(ATLAS-style random-item baseline)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(fig_dir / "atlasrep_items_vs_se.png", dpi=140)
    plt.close(fig)

    # (4) theta MAE vs SE
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    for col, bench in zip(colors, benches):
        y = [all_stats[bench]["theta"][se]["mae"] for se in ses]
        yerr = [all_stats[bench]["theta"][se]["se"] for se in ses]
        ax.errorbar(ses, y, yerr=yerr, marker="o", color=col, capsize=3, label=bench)
    ax.set_xlabel("SE stopping threshold")
    ax.set_ylabel("theta MAE (|CAT theta - full-bank theta|)")
    ax.set_title("OpenLM: ability (theta) recovery MAE vs SE stopping threshold")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "atlasrep_theta_mae_vs_se.png", dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benches", default="ifeval,gpqa,math,bbh,musr")
    ap.add_argument("--ses", default="0.1,0.2,0.3")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--floor-se", type=float, default=0.095,
                    help="tight SE floor for the single full CAT trace per model")
    ap.add_argument("--max-items", type=int, default=400)
    ap.add_argument("--out-dir", type=Path,
                    default=REPO / "AdaptiveTesting/Research/01_MCQ_ATLAS/data/atlas_replication")
    ap.add_argument("--fig-dir", type=Path,
                    default=REPO / "AdaptiveTesting/Research/01_MCQ_ATLAS/figures")
    args = ap.parse_args()

    os.environ.setdefault("MPLCONFIGDIR", str(args.out_dir / ".mplcache"))
    benches = [b.strip() for b in args.benches.split(",") if b.strip()]
    ses = sorted((float(s) for s in args.ses.split(",")), reverse=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_stats: dict[str, dict] = {}
    for bench in benches:
        if bench not in BENCH_PATHS:
            print(f"[{bench}] unknown bench, skipping", flush=True)
            continue
        all_stats[bench] = run_bench(bench, ses, args.seed, args.floor_se,
                                     args.max_items, args.out_dir)

    # cross-benchmark summary CSVs (mirror ATLAS summary_*_mae_sd_se.csv columns)
    pirt_rows, theta_rows, items_rows = [], [], []
    for bench, st in all_stats.items():
        for se in ses:
            pirt_rows.append(st["pirt"][se])
            theta_rows.append(st["theta"][se])
            items_rows.append(st["items"][se])
    pd.DataFrame(pirt_rows).to_csv(args.out_dir / "summary_pirt_mae_sd_se.csv", index=False)
    pd.DataFrame(theta_rows).to_csv(args.out_dir / "summary_theta_mae_sd_se.csv", index=False)
    pd.DataFrame(items_rows).to_csv(
        args.out_dir / "summary_items_adaptive_vs_random.csv", index=False)

    cross_figures(all_stats, ses, args.fig_dir)

    # copy per-bench panel figures into fig-dir with atlasrep_ prefix
    import shutil

    for bench in all_stats:
        for se in ses:
            src = args.out_dir / bench / f"pirt_vs_actual_se_{se:g}.png"
            if src.exists():
                shutil.copy(src, args.fig_dir / f"atlasrep_{bench}_pirt_vs_actual_se_{se:g}.png")

    print(f"\nwrote summaries + figures under {args.out_dir} and {args.fig_dir}", flush=True)


if __name__ == "__main__":
    main()
