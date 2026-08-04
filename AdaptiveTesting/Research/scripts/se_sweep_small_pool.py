#!/usr/bin/env python3
"""Small-pool ARC SE sweep: how the size of the calibration pool changes the
CAT items/correlation tradeoff.

This is the full-pool ARC SE sweep (`se_sweep.py`, `--source local`) re-run at
several calibration-pool sizes. Everything downstream of calibration is held
fixed and reused from `se_sweep.py`: the same held-out test models (seed 7,
n_test 13), the same EAP/Fisher-information CAT, the same SE grid, and the same
mean-probability IRT predictor. The ONLY thing that changes across runs is how
many TRAIN models the item bank is calibrated on.

Pool sizes: the full train pool (baseline) plus fixed-seed random subsamples of
the train models. 1PL (girth `rasch_mml`) and 2PL (girth `twopl_mml`) reproduce
the published local ARC bank exactly at the full pool. girth's `threepl_mml` is
broken under scipy>=1.15 (its SLSQP path feeds arrays into `fminbound`), so 3PL
is fit here with a self-contained marginal-maximum-likelihood EM (Bock-Aitkin,
Fisher scoring) on the same quadrature. 3PL is expected to destabilise at the
smallest pools; that is measured and reported, not forced.

Usage:
  export PYTHONPATH=$REPO/eduLLM-Evals
  uv run python AdaptiveTesting/Research/scripts/se_sweep_small_pool.py \
    --mcq-dir $REPO/AdaptiveTesting/Inputs/Open/LLM-Judge/mcq \
    --out-dir $REPO/AdaptiveTesting/Research/01_MCQ_ATLAS/data/se_sweep_small_pool
"""
from __future__ import annotations

# Keep BLAS modest so a few parallel fits do not oversubscribe the machine.
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "eduLLM-Evals"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import se_sweep as S  # noqa: E402  (reuse prob/eap_se/full_cat_traces/pred_meanprob/NODES/PRIORW)
from tutor_cat.mcq_irt.matrix import filter_items, load_benchmark  # noqa: E402

NODES = S.NODES
PRIORW = S.PRIORW
MODEL_TYPES = ("1PL", "2PL", "3PL")


# --------------------------------------------------------------------------- #
# 3PL calibration (self-contained MML-EM; girth's threepl_mml is broken here)  #
# --------------------------------------------------------------------------- #
def fit_3pl_mml(
    R: np.ndarray,
    nodes: np.ndarray = NODES,
    priorw: np.ndarray = PRIORW,
    max_iter: int = 100,
    tol: float = 1e-3,
    a_bounds: tuple[float, float] = (1e-2, 6.0),
    b_bounds: tuple[float, float] = (-6.0, 6.0),
    c_bounds: tuple[float, float] = (0.0, 0.5),
    c_init: float = 0.2,
    inner_steps: int = 25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Marginal-maximum-likelihood 3PL via Bock-Aitkin EM with a Fisher-scoring
    M-step, on the fixed EAP quadrature. `R` is a (persons x items) 0/1 matrix.
    Returns (a, b, c). Bounds keep the thin-sample optimiser numerically sane;
    degeneracy shows up as parameters pinned at those bounds and as a collapsing
    held-out correlation (both surfaced by the caller)."""
    R = np.asarray(R, float)
    _, J = R.shape
    p = R.mean(axis=0)
    a = np.ones(J)
    b = np.clip(-np.log(np.clip(p, 0.03, 0.97) / (1 - np.clip(p, 0.03, 0.97))), -3.0, 3.0)
    c = np.full(J, c_init)
    w = priorw
    eps = 1e-6
    node_col = nodes[:, None]

    def probs(a, b, c):
        z = np.clip(a[None, :] * (node_col - b[None, :]), -30, 30)
        g = 1.0 / (1.0 + np.exp(-z))
        P = np.clip(c[None, :] + (1 - c[None, :]) * g, eps, 1 - eps)
        return P, g

    prev_a = a.copy()
    for _ in range(max_iter):
        # E-step: posterior over ability nodes per person, then expected counts.
        P, _g = probs(a, b, c)
        logP, log1mP = np.log(P), np.log(1 - P)
        LL = R @ logP.T + (1 - R) @ log1mP.T  # (persons, nodes)
        LL -= LL.max(axis=1, keepdims=True)
        post = np.exp(LL) * w[None, :]
        post /= post.sum(axis=1, keepdims=True)
        Nk = post.sum(axis=0)  # (nodes,)
        Rk = post.T @ R  # (nodes, items) expected #correct at each node

        # M-step: bounded Fisher scoring on (a, b, c) for all items at once.
        Nk_col = Nk[:, None]
        for _s in range(inner_steps):
            z = np.clip(a[None, :] * (node_col - b[None, :]), -30, 30)
            g = 1.0 / (1.0 + np.exp(-z))
            P = np.clip(c[None, :] + (1 - c[None, :]) * g, eps, 1 - eps)
            gg = g * (1 - g)
            one_c = 1 - c[None, :]
            Pa = one_c * gg * (node_col - b[None, :])
            Pb = -one_c * a[None, :] * gg
            Pc = 1 - g
            q = Rk / P - (Nk_col - Rk) / (1 - P)  # dLL/dP
            W = Nk_col / (P * (1 - P))  # expected information weight
            s = np.stack([(q * Pa).sum(0), (q * Pb).sum(0), (q * Pc).sum(0)], axis=1)
            F = np.empty((J, 3, 3))
            F[:, 0, 0] = (W * Pa * Pa).sum(0)
            F[:, 0, 1] = F[:, 1, 0] = (W * Pa * Pb).sum(0)
            F[:, 0, 2] = F[:, 2, 0] = (W * Pa * Pc).sum(0)
            F[:, 1, 1] = (W * Pb * Pb).sum(0)
            F[:, 1, 2] = F[:, 2, 1] = (W * Pb * Pc).sum(0)
            F[:, 2, 2] = (W * Pc * Pc).sum(0)
            ridge = 1e-6 * (np.trace(F, axis1=1, axis2=2) / 3.0 + 1e-6)
            F[:, 0, 0] += ridge
            F[:, 1, 1] += ridge
            F[:, 2, 2] += ridge
            try:
                delta = np.linalg.solve(F, s[:, :, None])[:, :, 0]
            except np.linalg.LinAlgError:
                delta = np.stack([np.linalg.lstsq(F[j], s[j], rcond=None)[0] for j in range(J)])
            # clamp per-step move to avoid overshoot, then project into bounds
            delta = np.clip(delta, -1.0, 1.0)
            a = np.clip(a + delta[:, 0], *a_bounds)
            b = np.clip(b + delta[:, 1], *b_bounds)
            c = np.clip(c + delta[:, 2], *c_bounds)
            if np.abs(delta).max() < 1e-4:
                break

        if np.abs(a - prev_a).max() < tol:
            break
        prev_a = a.copy()
    return a, b, c


def fit_bank(kept_df, model_type: str):
    """Return (items, a, b, c) for the requested IRT model on a kept (dense,
    filtered) model x item matrix."""
    items = list(kept_df.columns)
    if model_type == "1PL":
        from girth import rasch_mml

        data = kept_df.to_numpy(dtype=int).T
        est = rasch_mml(data)
        b = np.asarray(est["Difficulty"], float)
        a = np.ones_like(b)
        c = np.zeros_like(b)
    elif model_type == "2PL":
        from girth import twopl_mml

        data = kept_df.to_numpy(dtype=int).T
        est = twopl_mml(data)
        a = np.asarray(est["Discrimination"], float)
        b = np.asarray(est["Difficulty"], float)
        c = np.zeros_like(a)
    elif model_type == "3PL":
        a, b, c = fit_3pl_mml(kept_df.to_numpy(dtype=float))
    else:
        raise ValueError(model_type)
    return items, a, b, c


# --------------------------------------------------------------------------- #
# One (pool size x model type) cell                                            #
# --------------------------------------------------------------------------- #
def subsample_train(train: list[str], n_pool: int, seed: int) -> list[str]:
    if n_pool >= len(train):
        return list(train)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(train))[:n_pool]
    return [train[i] for i in idx]


def run_cell(args) -> dict:
    (mat, train, test, actual, model_type, n_pool, seed, ses) = args
    t0 = time.time()
    sub = subsample_train(train, n_pool, seed)
    kept, _rep = filter_items(mat.loc[sub], benchmark="arc_challenge")
    n_kept = kept.shape[1]
    items, a, b, c = fit_bank(kept, model_type)

    a = np.asarray(a, float)
    b = np.asarray(b, float)
    c = np.asarray(c, float)
    usable = (
        np.isfinite(a)
        & (a > 0)
        & np.isfinite(b)
        & np.isfinite(c)
        & (c >= 0)
        & (c < 1.0)
    )
    n_a_le0 = int(np.sum(~((a > 0) & np.isfinite(a))))
    items = [it for it, keep in zip(items, usable, strict=False) if keep]
    a, b, c = a[usable], b[usable], c[usable]
    n_bank = len(items)

    fit_secs = time.time() - t0

    # degeneracy fingerprints (bounds pinned = thin-sample instability)
    a_lo = int(np.sum(a <= 0.05))
    a_hi = int(np.sum(a >= 5.9))
    c_hi = int(np.sum(c >= 0.45)) if model_type == "3PL" else 0

    resp = {m: mat.loc[m, items].to_numpy(float) for m in test}
    traces = {m: S.full_cat_traces(resp[m], a, b, c, S.pred_meanprob) for m in test}

    rows = []
    for se in ses:
        n_items, preds, acts = [], [], []
        for m in test:
            chosen = None
            for (ni, s, pr) in traces[m]:
                if ni >= S.MIN_ITEMS and s <= se:
                    chosen = (ni, pr)
                    break
            if chosen is None:
                chosen = (traces[m][-1][0], traces[m][-1][2])
            n_items.append(chosen[0])
            preds.append(chosen[1])
            acts.append(actual[m])
        preds = np.array(preds)
        acts = np.array(acts)
        r = float(np.corrcoef(preds, acts)[0, 1]) if n_bank >= 2 else float("nan")
        mae = float(np.mean(np.abs(preds - acts)))
        rows.append(
            {
                "model_type": model_type,
                "n_pool": n_pool,
                "se_target": se,
                "mean_items": round(float(np.mean(n_items)), 2),
                "corr": round(r, 4),
                "mae": round(mae, 4),
                "n_bank_items": n_bank,
            }
        )

    info = {
        "model_type": model_type,
        "n_pool": n_pool,
        "n_kept_items": n_kept,
        "n_bank_items": n_bank,
        "n_dropped_a_le0": n_a_le0,
        "a_min": round(float(a.min()), 3) if n_bank else float("nan"),
        "a_max": round(float(a.max()), 3) if n_bank else float("nan"),
        "a_pinned_lo": a_lo,
        "a_pinned_hi": a_hi,
        "c_min": round(float(c.min()), 3) if n_bank else float("nan"),
        "c_max": round(float(c.max()), 3) if n_bank else float("nan"),
        "c_pinned_hi": c_hi,
        "fit_seconds": round(fit_secs, 1),
    }
    return {"rows": rows, "info": info}


# --------------------------------------------------------------------------- #
# Data prep (shared, deterministic)                                            #
# --------------------------------------------------------------------------- #
def prepare(mcq_dir: str, bench: str, n_test: int, seed: int):
    mat = load_benchmark(mcq_dir, bench).dropna(axis=0, how="any")
    models = list(mat.index)
    rng = np.random.default_rng(seed)
    order = [models[i] for i in rng.permutation(len(models))]
    test = sorted(order[:n_test])
    train = order[n_test:]
    actual = {m: float(v) for m, v in mat.mean(axis=1).to_dict().items()}
    actual = {m: actual[m] for m in test}
    return mat, train, test, actual


# --------------------------------------------------------------------------- #
# Figures                                                                      #
# --------------------------------------------------------------------------- #
COLORS = {"1PL": "#55A868", "2PL": "#DD8452", "3PL": "#4C72B0"}


def fig_dual_axis(rows, n_pool, out_path, n_test):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(9.5, 5.6))
    ax2 = ax1.twinx()
    for mt in MODEL_TYPES:
        r = sorted(
            [x for x in rows if x["model_type"] == mt and x["n_pool"] == n_pool],
            key=lambda x: x["se_target"],
        )
        if not r:
            continue
        se = [x["se_target"] for x in r]
        items = [x["mean_items"] for x in r]
        corr = [x["corr"] for x in r]
        color = COLORS[mt]
        ax1.plot(se, items, "o-", color=color, label=f"{mt} items")
        ax2.plot(se, corr, "s--", color=color, alpha=0.75, label=f"{mt} r")
    ax1.set_yscale("log")
    ax1.invert_xaxis()
    ax1.set_xlabel("SE target (tighter to the right)")
    ax1.set_ylabel("mean CAT items (log scale)")
    ax2.set_ylabel("Pearson r")
    ax2.set_ylim(0.0, 1.0)
    ax1.set_title(
        f"ARC (calibration pool N={n_pool}): CAT items and correlation vs SE target\n"
        f"solid = items (left), dashed = r (right); {n_test} held-out models"
    )
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8, ncol=3)
    ax1.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig_corr_vs_pool(rows, se_target, pool_sizes, full_n, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    for mt in MODEL_TYPES:
        xs, ys = [], []
        for n in pool_sizes:
            hit = [
                x
                for x in rows
                if x["model_type"] == mt
                and x["n_pool"] == n
                and abs(x["se_target"] - se_target) < 1e-9
            ]
            if hit:
                xs.append(n)
                ys.append(hit[0]["corr"])
        ax.plot(xs, ys, "o-", color=COLORS[mt], label=mt)
    ax.set_xlabel("calibration pool size (# train models)")
    ax.set_ylabel(f"Pearson r at SE={se_target}")
    ax.set_xticks(pool_sizes)
    labels = [f"{n}\n(full)" if n == full_n else str(n) for n in pool_sizes]
    ax.set_xticklabels(labels)
    ax.set_title(f"ARC: correlation at SE={se_target} vs calibration pool size")
    ax.grid(True, alpha=0.3)
    ax.legend(title="IRT model")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mcq-dir", required=True)
    ap.add_argument("--bench", default="arc_challenge")
    ap.add_argument("--n-test", type=int, default=13)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--pools", default="15,30,45", help="small-pool sizes; full pool added automatically")
    ap.add_argument("--se-list", default=",".join(str(s) for s in S.DEFAULT_SES))
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--selftest", action="store_true", help="fit 3PL on full+N=15 only")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(args.out_dir / ".mplcache"))
    ses = [float(s) for s in args.se_list.split(",")]

    mat, train, test, actual = prepare(args.mcq_dir, args.bench, args.n_test, args.seed)
    full_n = len(train)
    print(f"total models={len(mat.index)}  full train pool N={full_n}  held-out test={len(test)}")

    small_pools = sorted({int(x) for x in args.pools.split(",") if int(x) < full_n})
    pool_sizes = small_pools + [full_n]
    print(f"pool sizes = {pool_sizes} (full = {full_n})")

    if args.selftest:
        for n in (full_n, 15):
            out = run_cell((mat, train, test, actual, "3PL", n, args.seed, ses))
            print("SELFTEST 3PL", out["info"])
            for row in out["rows"]:
                if row["se_target"] in (0.5, 0.3, 0.12):
                    print("   ", row)
        return

    jobs = [
        (mat, train, test, actual, mt, n, args.seed, ses)
        for n in pool_sizes
        for mt in MODEL_TYPES
    ]
    def _log(res):
        i = res["info"]
        print(f"done {i['model_type']} N={i['n_pool']:<3} bank={i['n_bank_items']:<4} "
              f"a=[{i['a_min']},{i['a_max']}] cmax={i['c_max']} ({i['fit_seconds']}s)", flush=True)

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
            # sandboxed environments may forbid the semaphores a process pool needs
            print(f"process pool unavailable ({e}); running sequentially", flush=True)
            results = run_sequential()
    else:
        results = run_sequential()

    rows = [r for res in results for r in res["rows"]]
    infos = [res["info"] for res in results]

    # combined CSV (requested schema)
    combined = args.out_dir / "se_sweep_small_pool_combined.csv"
    fields = ["model_type", "n_pool", "se_target", "mean_items", "corr", "mae", "n_bank_items"]
    rows_sorted = sorted(rows, key=lambda x: (x["model_type"], -x["n_pool"], -x["se_target"]))
    with open(combined, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows_sorted)
    print("wrote", combined)

    # fit diagnostics CSV
    diag = args.out_dir / "fit_diagnostics.csv"
    dfields = list(infos[0].keys())
    with open(diag, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=dfields)
        w.writeheader()
        w.writerows(sorted(infos, key=lambda x: (x["model_type"], -x["n_pool"])))
    print("wrote", diag)

    # corr @ SE=0.3 pivot
    pivot = args.out_dir / "corr_at_se0.3_by_pool.csv"
    with open(pivot, "w", newline="") as fh:
        w = csv.writer(fh)
        header = [f"N={n}" + ("(full)" if n == full_n else "") for n in pool_sizes]
        w.writerow(["model_type", *header])
        for mt in MODEL_TYPES:
            line = [mt]
            for n in pool_sizes:
                hit = [x for x in rows if x["model_type"] == mt and x["n_pool"] == n
                       and abs(x["se_target"] - 0.3) < 1e-9]
                line.append(hit[0]["corr"] if hit else "")
            w.writerow(line)
    print("wrote", pivot)

    # figures
    n30 = 30 if 30 in pool_sizes else small_pools[0]
    fig_dual_axis(rows, n30, args.out_dir / f"se_sweep_small_pool_N{n30}.png", len(test))
    print("wrote", args.out_dir / f"se_sweep_small_pool_N{n30}.png")
    fig_corr_vs_pool(rows, 0.3, pool_sizes, full_n, args.out_dir / "corr_vs_pool_se0.3.png")
    print("wrote", args.out_dir / "corr_vs_pool_se0.3.png")


if __name__ == "__main__":
    main()
