#!/usr/bin/env python3
"""Experiment 3: 1PL vs 2PL vs 3PL on three small-pool skill benchmarks.

For each benchmark in {pedagogy, piqa, socialiqa} and each IRT model in
{1PL, 2PL, 3PL} this calibrates an item bank on a 40-model TRAIN split, runs an
SE-stopped EAP/Fisher-information CAT on the 12 held-out TEST models, and records
the Pearson correlation (and MAE) of the CAT-predicted vs actual full-bank
accuracy, the mean number of CAT items administered, and the kept (a>0) bank size.

Everything is held identical to the published 2PL diagnostic so the numbers are
comparable and reproducible:

* Split: ``load_benchmark(...).dropna`` then ``default_rng(7).permutation``,
  ``n_test = min(12, n_models // 3)``, ``test = sorted(order[:n_test])``,
  ``train = order[n_test:]`` -- byte-for-byte the split in ``mcq_diagnostic.py``.
* Item filter: ``filter_items`` on the TRAIN slice (ATLAS point-biserial rules).
* CAT: the c-aware EAP/Fisher CAT from ``se_sweep.py`` (``full_cat_traces`` +
  ``pred_meanprob``); at c=0 it reduces exactly to the 2PL/1PL CAT, so 2PL and
  1PL reproduce ``mcq_diagnostic.py`` / ``rasch_1pl`` to 4 decimals (validated).

Fitters (reused, not reimplemented):

* 1PL  -> girth ``rasch_mml``   (a fixed at 1)
* 2PL  -> girth ``twopl_mml``
* 3PL  -> self-contained MML-EM (Bock-Aitkin, Fisher scoring) in
  ``se_sweep_small_pool.fit_3pl_mml`` -- girth's ``threepl_mml`` is broken under
  scipy>=1.15. All three go through ``se_sweep_small_pool.fit_bank``.

A uniform usability filter (finite, a>0, 0<=c<1) is applied to every fitted bank
(a no-op for 1PL/2PL here; the safety net that matters for the thin-pool 3PL).

Usage:
  uv run python AdaptiveTesting/Research/01_MCQ_ATLAS/data/pedagogy_feasibility/\
pl_1_2_3_comparison/run_pl_comparison.py [--workers 3]
"""

from __future__ import annotations

# Keep BLAS modest so a few parallel fits do not oversubscribe the machine.
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]  # .../olmo-eval-full
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplcache"))
SCRIPTS = REPO / "AdaptiveTesting/Research/scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(REPO / "eduLLM-Evals"))

import se_sweep as S  # noqa: E402  CAT: prob/eap_se/full_cat_traces/pred_meanprob/NODES/MIN_ITEMS
import se_sweep_small_pool as SP  # noqa: E402  fit_bank / fit_3pl_mml (self-contained 3PL)
from tutor_cat.mcq_irt.matrix import filter_items, load_benchmark  # noqa: E402

MCQ_DIR = REPO / "AdaptiveTesting/Outputs/full200_results/Outputs/mcq"
BENCHES = ("pedagogy", "piqa", "socialiqa")
MODEL_TYPES = ("1PL", "2PL", "3PL")
N_TEST = 12
SEED = 7
SE_TARGETS = (0.3, 0.15)
COLORS = {"1PL": "#55A868", "2PL": "#DD8452", "3PL": "#4C72B0"}

# Published references for validation (source-of-truth CSVs in this repo).
REF_2PL_R = {  # mcq_diagnostic_summary.csv
    ("pedagogy", 0.3): 0.7163,
    ("pedagogy", 0.15): 0.8996,
    ("piqa", 0.3): 0.9373,
    ("piqa", 0.15): 0.9536,
    ("socialiqa", 0.3): 0.2713,
    ("socialiqa", 0.15): 0.4296,
}
REF_1PL_R = {  # rasch_1pl/pedagogy_1pl_vs_2pl_summary.csv
    ("pedagogy", 0.3): 0.8903,
    ("pedagogy", 0.15): 0.9407,
}


# --------------------------------------------------------------------------- #
# Data prep (identical split to mcq_diagnostic.py / rasch_1pl)                 #
# --------------------------------------------------------------------------- #
def prepare(bench: str):
    mat = load_benchmark(MCQ_DIR, bench).dropna(axis=0, how="any")
    models = list(mat.index)
    rng = np.random.default_rng(SEED)
    order = [models[i] for i in rng.permutation(len(models))]
    n_test = min(N_TEST, len(models) // 3)
    test = sorted(order[:n_test])
    train = order[n_test:]
    actual_full = {m: float(v) for m, v in mat.mean(axis=1).to_dict().items()}
    kept_mat, report = filter_items(mat.loc[train], benchmark=bench)
    return mat, train, test, actual_full, kept_mat, report.n_items_raw


# --------------------------------------------------------------------------- #
# One (benchmark x model type) cell                                           #
# --------------------------------------------------------------------------- #
def run_cell(job: tuple[str, str]) -> dict:
    bench, model_type = job
    t0 = time.time()
    mat, train, test, actual_full, kept_mat, n_items_raw = prepare(bench)
    n_filtered = kept_mat.shape[1]

    items, a, b, c = SP.fit_bank(kept_mat, model_type)
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    c = np.asarray(c, float)

    usable = np.isfinite(a) & (a > 0) & np.isfinite(b) & np.isfinite(c) & (c >= 0) & (c < 1.0)
    n_dropped = int(np.sum(~usable))
    items = [it for it, keep in zip(items, usable, strict=False) if keep]
    a, b, c = a[usable], b[usable], c[usable]
    n_bank = len(items)
    fit_secs = time.time() - t0

    # degeneracy fingerprints (bounds pinned == thin-sample instability)
    a_lo = int(np.sum(a <= 0.05)) if n_bank else 0
    a_hi = int(np.sum(a >= 5.9)) if n_bank else 0
    c_hi = int(np.sum(c >= 0.45)) if (n_bank and model_type == "3PL") else 0

    resp = {m: mat.loc[m, items].to_numpy(float) for m in test}
    traces = {m: S.full_cat_traces(resp[m], a, b, c, S.pred_meanprob) for m in test}

    rows, per_model = [], []
    for se in SE_TARGETS:
        n_items, preds, acts, n_reached = [], [], [], 0
        for m in test:
            chosen = None
            for ni, s, pr in traces[m]:
                if ni >= S.MIN_ITEMS and s <= se:
                    chosen = (ni, pr)
                    n_reached += 1
                    break
            if chosen is None:  # SE floor never met -> full CAT end
                chosen = (traces[m][-1][0], traces[m][-1][2])
            n_items.append(chosen[0])
            preds.append(chosen[1])
            acts.append(actual_full[m])
            per_model.append(
                {
                    "benchmark": bench,
                    "model_type": model_type,
                    "se_stop": se,
                    "model": m,
                    "actual_full": round(actual_full[m], 6),
                    "pred": round(chosen[1], 6),
                    "n_items": chosen[0],
                }
            )
        preds = np.asarray(preds)
        acts = np.asarray(acts)
        r = float(np.corrcoef(preds, acts)[0, 1]) if n_bank >= 2 else float("nan")
        mae = float(np.mean(np.abs(preds - acts)))
        rows.append(
            {
                "benchmark": bench,
                "model_type": model_type,
                "se_stop": se,
                "r": round(r, 4),
                "mae": round(mae, 4),
                "mean_items": round(float(np.mean(n_items)), 2),
                "kept_items": n_bank,
                "frac_reached_se": round(n_reached / len(test), 2),
            }
        )

    info = {
        "benchmark": bench,
        "model_type": model_type,
        "n_items_raw": n_items_raw,
        "n_items_filtered": n_filtered,
        "kept_items_a_gt0": n_bank,
        "n_dropped_unusable": n_dropped,
        "a_min": round(float(a.min()), 3) if n_bank else float("nan"),
        "a_max": round(float(a.max()), 3) if n_bank else float("nan"),
        "a_pinned_lo": a_lo,
        "a_pinned_hi": a_hi,
        "c_min": round(float(c.min()), 3) if n_bank else float("nan"),
        "c_max": round(float(c.max()), 3) if n_bank else float("nan"),
        "c_pinned_hi": c_hi,
        "fit_seconds": round(fit_secs, 1),
    }
    return {"rows": rows, "info": info, "per_model": per_model}


# --------------------------------------------------------------------------- #
# Figures                                                                      #
# --------------------------------------------------------------------------- #
def _grouped(ax, rows, se, key, ylabel, ylim=None, annotate="{:.2f}"):
    x = np.arange(len(BENCHES))
    w = 0.26
    for i, mt in enumerate(MODEL_TYPES):
        vals = []
        for bn in BENCHES:
            hit = [
                r
                for r in rows
                if r["benchmark"] == bn and r["model_type"] == mt and r["se_stop"] == se
            ]
            vals.append(hit[0][key] if hit else np.nan)
        xs = x + (i - 1) * w
        ax.bar(xs, vals, w, label=mt, color=COLORS[mt], edgecolor="black", linewidth=0.5)
        for xi, v in zip(xs, vals, strict=False):
            if np.isfinite(v):
                ax.annotate(
                    annotate.format(v),
                    (xi, v),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    xytext=(0, 1),
                    textcoords="offset points",
                )
    ax.set_xticks(x)
    ax.set_xticklabels(BENCHES)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(True, axis="y", alpha=0.3)


def fig_main(rows, out_path, n_test):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.4))
    _grouped(ax1, rows, 0.3, "r", "Pearson r (CAT-pred vs actual full acc)", ylim=(0, 1.0))
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_title("Held-out Pearson r at SE<=0.30")
    ax1.legend(title="IRT model", loc="upper right")

    _grouped(ax2, rows, 0.3, "mean_items", "mean CAT items administered", annotate="{:.0f}")
    ax2.set_title("Mean CAT items at SE<=0.30")

    fig.suptitle(
        f"1PL vs 2PL vs 3PL on small-pool skill benchmarks "
        f"(40-model calibration pool, {n_test} held-out models)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig_r_by_se(rows, out_path, n_test):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for ax, se in zip(axes, (0.3, 0.15), strict=False):
        _grouped(ax, rows, se, "r", "Pearson r", ylim=(0, 1.0))
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"SE<={se:g}")
    axes[0].legend(title="IRT model", loc="upper right")
    fig.suptitle(
        f"Held-out Pearson r by IRT model and SE target (40-model pool, {n_test} held-out models)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3, help="modest parallelism over the 9 cells")
    ap.add_argument("--out-dir", type=Path, default=HERE)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    jobs = [(bench, mt) for bench in BENCHES for mt in MODEL_TYPES]

    def _log(res):
        i = res["info"]
        se3 = next(r for r in res["rows"] if r["se_stop"] == 0.3)
        print(
            f"done {i['benchmark']:<10} {i['model_type']}  bank={i['kept_items_a_gt0']:<4} "
            f"a=[{i['a_min']},{i['a_max']}] cmax={i['c_max']}  "
            f"r@.3={se3['r']:.3f} items@.3={se3['mean_items']:.1f}  ({i['fit_seconds']}s)",
            flush=True,
        )

    def run_sequential():
        out = []
        for job in jobs:
            res = run_cell(job)
            out.append(res)
            _log(res)
        return out

    results = []
    if args.workers > 1:
        try:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                for res in ex.map(run_cell, jobs):
                    results.append(res)
                    _log(res)
        except (PermissionError, OSError) as e:
            print(f"process pool unavailable ({e}); running sequentially", flush=True)
            results = run_sequential()
    else:
        results = run_sequential()

    rows = [r for res in results for r in res["rows"]]
    infos = [res["info"] for res in results]
    per_model = [p for res in results for p in res["per_model"]]
    n_test = len(
        {
            p["model"]
            for p in per_model
            if p["benchmark"] == "pedagogy" and p["model_type"] == "2PL" and p["se_stop"] == 0.3
        }
    )

    # combined deliverable table (benchmark x model x SE)
    order_mt = {mt: i for i, mt in enumerate(MODEL_TYPES)}
    order_bn = {bn: i for i, bn in enumerate(BENCHES)}
    rows_sorted = sorted(
        rows, key=lambda r: (order_bn[r["benchmark"]], -r["se_stop"], order_mt[r["model_type"]])
    )
    combined = args.out_dir / "pl_1_2_3_comparison.csv"
    fields = [
        "benchmark",
        "model_type",
        "se_stop",
        "r",
        "mae",
        "mean_items",
        "kept_items",
        "frac_reached_se",
    ]
    with open(combined, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows_sorted)
    print("wrote", combined)

    # fit diagnostics
    diag = args.out_dir / "fit_diagnostics.csv"
    with open(diag, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(infos[0].keys()))
        w.writeheader()
        w.writerows(
            sorted(infos, key=lambda i: (order_bn[i["benchmark"]], order_mt[i["model_type"]]))
        )
    print("wrote", diag)

    # per-model predictions (audit trail)
    pm = args.out_dir / "per_model_predictions.csv"
    with open(pm, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_model[0].keys()))
        w.writeheader()
        w.writerows(
            sorted(
                per_model,
                key=lambda p: (
                    order_bn[p["benchmark"]],
                    -p["se_stop"],
                    order_mt[p["model_type"]],
                    p["model"],
                ),
            )
        )
    print("wrote", pm)

    # figures
    fig_main(rows, args.out_dir / "pl_comparison_se0.3.png", n_test)
    print("wrote", args.out_dir / "pl_comparison_se0.3.png")
    fig_r_by_se(rows, args.out_dir / "pl_comparison_r_by_se.png", n_test)
    print("wrote", args.out_dir / "pl_comparison_r_by_se.png")

    # ------------------------------------------------------------------ #
    # validation against published 2PL and pedagogy-1PL numbers          #
    # ------------------------------------------------------------------ #
    print("\n=== validation (recomputed vs published) ===")
    r_lookup = {(r["benchmark"], r["model_type"], r["se_stop"]): r["r"] for r in rows}
    ok = True
    for (bn, se), ref in sorted(REF_2PL_R.items()):
        got = r_lookup.get((bn, "2PL", se))
        match = got is not None and abs(got - ref) <= 1e-3
        ok &= match
        print(f"  2PL {bn:<10} SE<={se:<4}  got={got}  ref={ref}  {'OK' if match else 'MISMATCH'}")
    for (bn, se), ref in sorted(REF_1PL_R.items()):
        got = r_lookup.get((bn, "1PL", se))
        match = got is not None and abs(got - ref) <= 1e-3
        ok &= match
        print(f"  1PL {bn:<10} SE<={se:<4}  got={got}  ref={ref}  {'OK' if match else 'MISMATCH'}")
    print(f"=== validation {'PASSED' if ok else 'FAILED'} ===")

    # compact pretty table to stdout
    print("\nbenchmark   model  SE     r       MAE     items   kept  reached")
    for r in rows_sorted:
        print(
            f"{r['benchmark']:<10}  {r['model_type']:<4}  {r['se_stop']:<4}  "
            f"{r['r']:<6}  {r['mae']:<6}  {r['mean_items']:<6}  "
            f"{r['kept_items']:<4}  {r['frac_reached_se']}"
        )


if __name__ == "__main__":
    main()
