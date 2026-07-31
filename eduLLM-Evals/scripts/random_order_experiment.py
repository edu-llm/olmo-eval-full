"""Does the ORDER of the administered items change the CAT result? 10-trial randomization.

Hypothesis under test
---------------------
The CAT ability update is a *sequential* one-step Newton/Laplace approximation: at each
item it linearises the 2PL at whatever theta currently is, folds the item in, and never
revisits it. That makes the estimator **path dependent** -- administering the SAME set of
items in a different order should land on a different final theta. A batch estimator
(EAP over the whole administered set at once) has no such dependence.

Design
------
1. Run the production CAT (max-Fisher-info selection, SE target 0.3) once per model to get
   that model's administered item SET ``S`` (~30-40 items). This set is held FIXED.
2. For each of ``--trials`` random seeds, shuffle ``S`` and replay the identical sequential
   update in that order, applying the same SE stopping rule. Record the final ability.
3. Compare across trials, and against the order-free batch EAP over the same ``S``.

Holding ``S`` fixed is what isolates order effects from selection effects: every trial sees
exactly the same evidence, only the sequence differs.

Outputs (``--out-dir``)
-----------------------
* ``per_trial_summary.csv``  -- slope / r / worst-12 gap / mean items, per trial
* ``per_model_spread.csv``   -- every model's ability in every trial + spread stats
* ``metrics.json``           -- aggregate spread, and the batch-EAP comparator
* ``figures/recovery_overlay_<dim>.png``     -- all trials overlaid on one scatter
* ``figures/theta_spread_vs_ability_<dim>.png`` -- per-model across-trial range vs ability
* ``figures/recovery_scatter_<dim>_trial01.png`` -- one trial, directly comparable to the
  baseline figure in regenerated_figures/2pl/

Usage
-----
    python scripts/random_order_experiment.py --out-dir random_order_experiment
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, log_expit, logsumexp

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("regen", ROOT / "scripts" / "regen_cat_figures.py")
regen = importlib.util.module_from_spec(_spec)
sys.modules["regen"] = regen
_spec.loader.exec_module(regen)

SE_TARGET = 0.3
MAX_ITEMS = 100
N_WORST = 12


def maxinfo_set(y, mask, A, b, se_target=SE_TARGET, max_items=MAX_ITEMS):
    """Production CAT: returns (final theta, administered item indices in pick order)."""
    theta = np.zeros(A.shape[1])
    U = np.eye(A.shape[1])
    remaining = list(np.where(mask)[0])
    order: list[int] = []
    while remaining and len(order) < max_items:
        rem = np.asarray(remaining)
        m = A[rem]
        p = expit(m @ theta - b[rem])
        info = p * (1.0 - p) * np.sum(m * m, axis=1)
        pick = int(rem[int(np.argmax(info))])
        theta, U = regen.mirt_update(theta, U, A[pick], float(b[pick]), int(y[pick]))
        remaining.remove(pick)
        order.append(pick)
        if float(np.max(np.sqrt(np.diag(U)))) < se_target:
            break
    return theta, order


def replay(y, seq, A, b, se_target=SE_TARGET):
    """Replay the sequential update over a given item ORDER, same stopping rule."""
    theta = np.zeros(A.shape[1])
    U = np.eye(A.shape[1])
    used = 0
    for idx in seq:
        theta, U = regen.mirt_update(theta, U, A[idx], float(b[idx]), int(y[idx]))
        used += 1
        if float(np.max(np.sqrt(np.diag(U)))) < se_target:
            break
    return theta, used


def batch_eap(y, idx, A, b, grid, log_prior):
    """Order-free EAP over the administered set (the comparator)."""
    Ai, bi, yi = A[idx], b[idx], y[idx]
    eta = Ai @ grid.T - bi[:, None]
    ll = yi @ log_expit(eta) + (1.0 - yi) @ log_expit(-eta)
    joint = ll + log_prior
    post = np.exp(joint - logsumexp(joint))
    return post @ grid


def figures(out_dir: Path, dim_names, full, trials, baseline, spread, worst):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    def _save(fig, name):
        p = fig_dir / name
        fig.tight_layout()
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths[name] = str(p)

    n_trials = trials.shape[0]
    cmap = plt.get_cmap("viridis")

    for k, dim in enumerate(dim_names):
        x = full[:, k]

        # --- all trials overlaid ---
        fig, ax = plt.subplots(figsize=(5.2, 5.0))
        for t in range(n_trials):
            ax.scatter(x, trials[t, :, k], s=14, alpha=0.45,
                       color=cmap(t / max(n_trials - 1, 1)), linewidth=0,
                       label=f"trial {t+1}" if t in (0, n_trials - 1) else None)
        ax.scatter(x, baseline[:, k], s=26, facecolor="none", edgecolor="crimson",
                   linewidth=0.9, label="max-info order (baseline)")
        lo = min(x.min(), trials[:, :, k].min()) - 0.3
        hi = max(x.max(), trials[:, :, k].max()) + 0.3
        ax.plot([lo, hi], [lo, hi], ls="--", color="gray", lw=1, label="y = x")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel(f"full-bank EAP ability ({dim})")
        ax.set_ylabel(f"CAT ability ({dim})")
        ax.set_title(f"{n_trials} random item orders, same item set\n{dim}")
        ax.legend(loc="upper left", fontsize=7)
        _save(fig, f"recovery_overlay_{dim}.png")

        # --- how much does order move each model? ---
        fig, ax = plt.subplots(figsize=(5.6, 4.2))
        ax.errorbar(x, spread["mean"][:, k],
                    yerr=[spread["mean"][:, k] - spread["min"][:, k],
                          spread["max"][:, k] - spread["mean"][:, k]],
                    fmt="o", ms=4, lw=0.8, capsize=2, alpha=0.8)
        ax.set_xlabel(f"full-bank EAP ability ({dim})")
        ax.set_ylabel(f"CAT ability across trials ({dim})")
        ax.set_title(f"Across-trial spread from item order alone\n"
                     f"{dim}: mean SD = {spread['sd'][:, k].mean():.3f}, "
                     f"max range = {(spread['max'][:,k]-spread['min'][:,k]).max():.2f}")
        _save(fig, f"theta_spread_vs_ability_{dim}.png")

        # --- one trial, directly comparable to the baseline figure ---
        r = float(np.corrcoef(x, trials[0, :, k])[0, 1])
        fig, ax = plt.subplots(figsize=(4.6, 4.4))
        ax.scatter(x, trials[0, :, k], s=28, alpha=0.75, edgecolor="k", linewidth=0.3)
        lo = min(x.min(), trials[0, :, k].min()) - 0.3
        hi = max(x.max(), trials[0, :, k].max()) + 0.3
        ax.plot([lo, hi], [lo, hi], ls="--", color="gray", lw=1, label="y = x")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel(f"full-bank EAP ability ({dim})")
        ax.set_ylabel(f"CAT ability ({dim})")
        ax.set_title(f"TutorBench CAT recovery: {dim} (random order, trial 1)\n"
                     f"Pearson r = {r:.3f} (n = {len(x)} models)")
        ax.legend(loc="upper left", fontsize=9)
        _save(fig, f"recovery_scatter_{dim}_trial01.png")

    return paths


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bank", type=Path,
                   default=ROOT / "data" / "TutorBench" / "rubrics_qmatrix_calibrated_2skill.jsonl")
    p.add_argument("--matrix", type=Path, default=None,
                   help="response matrix CSV; defaults to the one the bank's provenance names.")
    p.add_argument("--out-dir", type=Path, default=ROOT / "random_order_experiment")
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--seed", type=int, default=20260731)
    p.add_argument("--grid", type=int, default=201)
    p.add_argument("--range", type=float, default=8.0)
    args = p.parse_args()

    args.matrix = regen.resolve_matrix(args.bank, args.matrix)
    prov = regen.verify_matrix(args.bank, args.matrix)
    print()

    matrix = pd.read_csv(args.matrix, index_col=0)
    items, Af, bf = regen.load_bank(args.bank)
    items, A, b, Y, mask, dim_names = regen.align(
        matrix, items, Af, bf, regen.modeled_skill_names(args.bank))
    models = list(matrix.index)
    n_dims = A.shape[1]

    empty = ~mask.any(axis=0)
    print("=" * 78)
    print("ALIGNMENT")
    print("=" * 78)
    print(f"calibrated items in bank      : {len(items)}")
    print(f"  with NO responses (all-NaN) : {int(empty.sum())}  <- excluded automatically")
    print(f"  effectively usable          : {int((~empty).sum())}")
    print(f"models x observed cells       : {len(models)} x {int(mask.sum())}")
    print(f"dims                          : {dim_names}")

    grid, log_prior = regen.build_grid(n_dims, args.grid, "uniform", args.range)
    full = regen.eap_all_models(Y, mask, A, b, grid, log_prior, 4096)
    worst = np.argsort(full[:, dim_names.index(dim_names[0])])[:N_WORST]

    # Baseline production CAT -> fixes each model's administered SET.
    baseline = np.zeros((len(models), n_dims))
    sets: list[np.ndarray] = []
    for r in range(len(models)):
        th, order = maxinfo_set(Y[r], mask[r], A, b)
        baseline[r] = th
        sets.append(np.asarray(order, dtype=int))
    print(f"\nbaseline max-info CAT: mean administered set = "
          f"{np.mean([len(s) for s in sets]):.1f} items")

    # Order-free comparator over the same sets.
    batch = np.array([batch_eap(Y[r], sets[r], A, b, grid, log_prior)
                      for r in range(len(models))])

    # Randomised-order trials.
    rng = np.random.default_rng(args.seed)
    trials = np.zeros((args.trials, len(models), n_dims))
    used = np.zeros((args.trials, len(models)), dtype=int)
    for t in range(args.trials):
        for r in range(len(models)):
            seq = sets[r].copy()
            rng.shuffle(seq)
            trials[t, r], used[t, r] = replay(Y[r], seq, A, b)

    spread = {"mean": trials.mean(axis=0), "sd": trials.std(axis=0),
              "min": trials.min(axis=0), "max": trials.max(axis=0)}

    def stats(th):
        out = {}
        for k, d in enumerate(dim_names):
            out[f"slope_{d}"] = float(np.polyfit(full[:, k], th[:, k], 1)[0])
            out[f"r_{d}"] = float(np.corrcoef(full[:, k], th[:, k])[0, 1])
            out[f"gap_worst12_{d}"] = float(np.mean(th[worst, k] - full[worst, k]))
        return out

    rows = [{"trial": "baseline (max-info order)", "mean_items": float(np.mean([len(s) for s in sets])),
             **stats(baseline)}]
    for t in range(args.trials):
        rows.append({"trial": f"random order {t+1}", "mean_items": float(used[t].mean()),
                     **stats(trials[t])})
    rows.append({"trial": "batch EAP (order-free)", "mean_items": float(np.mean([len(s) for s in sets])),
                 **stats(batch)})
    summary = pd.DataFrame(rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_dir / "per_trial_summary.csv", index=False)

    per_model = {"model": models}
    for k, d in enumerate(dim_names):
        per_model[f"full_{d}"] = full[:, k]
        per_model[f"baseline_{d}"] = baseline[:, k]
        per_model[f"batch_{d}"] = batch[:, k]
        for t in range(args.trials):
            per_model[f"trial{t+1}_{d}"] = trials[t, :, k]
        per_model[f"sd_{d}"] = spread["sd"][:, k]
        per_model[f"range_{d}"] = spread["max"][:, k] - spread["min"][:, k]
    pd.DataFrame(per_model).to_csv(args.out_dir / "per_model_spread.csv", index=False)

    fig_paths = figures(args.out_dir, dim_names, full, trials, baseline, spread, worst)

    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hypothesis": "the sequential CAT update is order-dependent; batch EAP is not",
        "config": {"bank": str(args.bank), "matrix": str(args.matrix),
                   "trials": args.trials, "seed": args.seed,
                   "grid_nodes_per_dim": args.grid, "range": args.range,
                   "se_target": SE_TARGET},
        "provenance": prov,
        "alignment": {"calibrated_items": len(items),
                      "items_without_responses": int(empty.sum()),
                      "usable_items": int((~empty).sum()),
                      "observed_cells": int(mask.sum()),
                      "n_models": len(models)},
        "across_trial_spread": {
            d: {"mean_sd": float(spread["sd"][:, k].mean()),
                "median_sd": float(np.median(spread["sd"][:, k])),
                "max_sd": float(spread["sd"][:, k].max()),
                "mean_range": float((spread["max"][:, k] - spread["min"][:, k]).mean()),
                "max_range": float((spread["max"][:, k] - spread["min"][:, k]).max())}
            for k, d in enumerate(dim_names)},
        "per_trial": summary.to_dict(orient="records"),
        "figures": fig_paths,
    }
    with (args.out_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    print("\n" + "=" * 78)
    print("PER-TRIAL METRICS")
    print("=" * 78)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary.round(3).to_string(index=False))

    print("\n" + "=" * 78)
    print("HOW MUCH DOES ORDER ALONE MOVE THE ANSWER?")
    print("=" * 78)
    for k, d in enumerate(dim_names):
        s = spread["sd"][:, k]
        rng_ = spread["max"][:, k] - spread["min"][:, k]
        print(f"{d:12s}: mean SD={s.mean():.3f}  median SD={np.median(s):.3f}  "
              f"max SD={s.max():.3f}  mean range={rng_.mean():.3f}  max range={rng_.max():.3f}")

    print(f"\nwrote -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
