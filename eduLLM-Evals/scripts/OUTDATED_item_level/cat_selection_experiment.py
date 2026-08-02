"""A/B the CAT item-selection criterion: trace (current) vs D-optimality (PRD spec).

Background
----------
The PRD ("Implementation Strategy + Success Metrics") specifies choosing the next item to
maximise the DETERMINANT of the updated Fisher information matrix -- D-optimality. Every
current harness (``scripts/regen_cat_figures.run_cat_person``,
``scripts/offline_engine_driver`` via ``tutor_cat.selector``) instead maximises the TRACE
of the per-item information at a plug-in point estimate:

    trace criterion :  argmax_i  w_i * (m_i . m_i)          (m_i = q_i (x) a_i,  w_i = p_i(1-p_i))

which ignores the current posterior covariance U. The CAT-uncertainty audit flagged this as
a deviation from the spec.

D-optimality, made cheap
------------------------
Adding item i updates the information matrix by  I_new = U^{-1} + w_i * m_i m_i^T. By the
matrix-determinant lemma,

    det(I_new) = det(U^{-1}) * (1 + w_i * m_i^T U m_i),

so maximising det(I_new) is EXACTLY maximising  w_i * (m_i^T U m_i)  -- the same weight w_i
as the trace rule, but the quadratic form is taken in the U metric instead of the identity.
D-optimality therefore prefers items that shrink the CURRENTLY WIDEST posterior direction,
which the trace rule cannot see. This is the uncertainty-aware selection.

This harness runs the identical online M2PL update (``regen.mirt_update``) and the identical
three final estimators (online / batch EAP / MWLE) under BOTH selection rules, on the frozen
calibrated bank + response matrix, and compares:

  * test length to the SE target (fewer items = more efficient selection),
  * recovery of the full-bank EAP reference ability (slope / r / worst-12 gap).

It is standalone and READ-ONLY: it does not modify ``tutor_cat`` or any frozen artifact.

Usage
-----
    python scripts/cat_selection_experiment.py \
        --bank data/TutorBench/rubrics_qmatrix_calibrated_2skill.jsonl \
        --out-dir regenerated_figures/selection_experiment/2_skills
    python scripts/cat_selection_experiment.py \
        --bank data/TutorBench/rubrics_qmatrix_calibrated_3skill.jsonl \
        --grid 61 --out-dir regenerated_figures/selection_experiment/3_skills
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
from scipy.special import expit

# parents[2] because this file now lives in scripts/OUTDATED_item_level/ (see README there).
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location("regen", ROOT / "scripts" / "regen_cat_figures.py")
regen = importlib.util.module_from_spec(_spec)
sys.modules["regen"] = regen
_spec.loader.exec_module(regen)

SELECTIONS = ("trace", "dopt")
ESTIMATORS = {"online": "production online", "batch": "batch EAP", "mwle": "batch EAP + MWLE"}


def run_cat_person(y, mask, A, b, min_items, max_items, se_target, selection):
    """Max-info adaptive administration under the chosen selection rule.

    ``trace``: argmax_i w_i * (m_i . m_i)  -- current behaviour (U-blind).
    ``dopt`` : argmax_i w_i * (m_i^T U m_i) -- D-optimality via the determinant lemma.
    Everything else (the online M2PL update, stopping rule) is identical.
    """
    n_dims = A.shape[1]
    theta = np.zeros(n_dims)
    U = np.eye(n_dims)
    remaining = set(int(i) for i in np.where(mask)[0])
    n_admin = 0
    order: list[int] = []

    while remaining and n_admin < max_items:
        rem = np.fromiter(remaining, dtype=int)
        m = A[rem]
        p = expit(m @ theta - b[rem])
        w = p * (1.0 - p)
        if selection == "trace":
            info = w * np.sum(m * m, axis=1)
        else:  # dopt: quadratic form in the U metric
            mU = m @ U
            info = w * np.sum(mU * m, axis=1)
        pick = int(rem[int(np.argmax(info))])
        theta, U = regen.mirt_update(theta, U, A[pick], float(b[pick]), int(y[pick]))
        remaining.discard(pick)
        order.append(pick)
        n_admin += 1
        se = np.sqrt(np.diag(U))
        if n_admin >= min_items and float(np.max(se)) < se_target:
            break

    return theta, n_admin, np.sqrt(np.diag(U)), order


def recovery_stats(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    w = np.argsort(x)[:12]
    return {"slope": float(np.polyfit(x, y, 1)[0]),
            "r": float(np.corrcoef(x, y)[0, 1]),
            "gap_worst12": float(np.mean(y[w] - x[w])), "n": int(x.size)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--matrix", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--grid", type=int, default=61)
    p.add_argument("--range", type=float, default=6.0)
    p.add_argument("--se-target", type=float, default=0.3)
    p.add_argument("--min-items", type=int, default=1)
    p.add_argument("--max-items", type=int, default=100)
    p.add_argument("--mwle-ridge", type=float, default=1e-6)
    args = p.parse_args()

    args.matrix = regen.resolve_matrix(args.bank, args.matrix)
    prov = regen.verify_matrix(args.bank, args.matrix)
    matrix = pd.read_csv(args.matrix, index_col=0)
    items, Af, bf = regen.load_bank(args.bank)
    items, A, b, Y, mask, dim_names = regen.align(
        matrix, items, Af, bf, regen.modeled_skill_names(args.bank))
    models = list(matrix.index)
    n_dims = A.shape[1]
    print(f"bank={args.bank.name}  models={len(models)}  dims={dim_names}")

    grid, log_prior = regen.build_grid(n_dims, args.grid, "uniform", args.range)
    theta_full = regen.eap_all_models(Y, mask, A, b, grid, log_prior, 4096)

    rows = []
    for r, model in enumerate(models):
        rec = {"model": model}
        for d, dn in enumerate(dim_names):
            rec[f"theta_full_{dn}"] = float(theta_full[r, d])
        for sel in SELECTIONS:
            th_on, n_admin, se, order = run_cat_person(
                Y[r], mask[r], A, b, args.min_items, args.max_items, args.se_target, sel)
            idx = np.asarray(order, dtype=int)
            th_ba = regen.eap_subset(Y[r], idx, A, b, grid, log_prior) if idx.size else th_on.copy()
            th_mw, ok = (regen.mwle_subset(Y[r], idx, A, b, th_ba, ridge=args.mwle_ridge)
                         if idx.size else (th_ba.copy(), True))
            rec[f"{sel}_n_admin"] = int(n_admin)
            rec[f"{sel}_mwle_converged"] = bool(ok)
            for d, dn in enumerate(dim_names):
                rec[f"{sel}_online_{dn}"] = float(th_on[d])
                rec[f"{sel}_batch_{dn}"] = float(th_ba[d])
                rec[f"{sel}_mwle_{dn}"] = float(th_mw[d])
        rows.append(rec)

    df = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "cat_selection_per_model.csv", index=False)

    agg: dict = {}
    for sel in SELECTIONS:
        agg[sel] = {"length": {"mean": float(df[f"{sel}_n_admin"].mean()),
                               "median": float(df[f"{sel}_n_admin"].median()),
                               "max": int(df[f"{sel}_n_admin"].max())},
                    "recovery": {}}
        for est in ESTIMATORS:
            agg[sel]["recovery"][est] = {}
            for dn in dim_names:
                x = df[f"theta_full_{dn}"].to_numpy()
                y = df[f"{sel}_{est}_{dn}"].to_numpy()
                agg[sel]["recovery"][est][dn] = recovery_stats(x, y)

    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "A/B CAT item selection: trace (U-blind) vs D-optimality (U-aware)",
        "config": {"bank": str(args.bank), "matrix": str(args.matrix),
                   "grid": args.grid, "range": args.range, "se_target": args.se_target,
                   "min_items": args.min_items, "max_items": args.max_items},
        "provenance": prov, "dims": dim_names, "n_models": len(models),
        "results": agg,
    }
    with (args.out_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    _figures(df, dim_names, agg, args.out_dir / "figures")

    print("\n" + "=" * 92)
    print("SELECTION A/B  (reference = full-bank EAP)")
    print("=" * 92)
    for sel in SELECTIONS:
        L = agg[sel]["length"]
        print(f"\n[{sel}]  length mean={L['mean']:.1f} median={L['median']:.0f} max={L['max']}")
        for est in ESTIMATORS:
            line = "  ".join(
                f"{dn[:4]} r={agg[sel]['recovery'][est][dn]['r']:.3f} "
                f"m={agg[sel]['recovery'][est][dn]['slope']:.3f}" for dn in dim_names)
            print(f"    {est:7s}: {line}")
    print(f"\nwrote -> {args.out_dir}")
    return 0


def _figures(df, dim_names, agg, fig_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    # test-length comparison histogram
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist([df["trace_n_admin"], df["dopt_n_admin"]], bins=20,
            label=[f"trace (mean {df['trace_n_admin'].mean():.1f})",
                   f"D-opt (mean {df['dopt_n_admin'].mean():.1f})"], alpha=0.8)
    ax.set_xlabel("items administered to SE target")
    ax.set_ylabel("models")
    ax.set_title("CAT length: trace vs D-optimality selection")
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(fig_dir / "length_trace_vs_dopt.png", dpi=130)
    plt.close(fig)

    # recovery scatter overlay per dim, MWLE estimator (the adopted one)
    for dn in dim_names:
        x = df[f"theta_full_{dn}"].to_numpy()
        fig, ax = plt.subplots(figsize=(5, 4.8))
        for sel, color in (("trace", "#1f77b4"), ("dopt", "#d62728")):
            y = df[f"{sel}_mwle_{dn}"].to_numpy()
            s = agg[sel]["recovery"]["mwle"][dn]
            ax.scatter(x, y, s=22, alpha=0.65, color=color, edgecolor="k", linewidth=0.2,
                       label=f"{sel}: r={s['r']:.3f}, slope={s['slope']:.3f}")
        lo = min(np.nanmin(x), df[[f"trace_mwle_{dn}", f"dopt_mwle_{dn}"]].min().min()) - 0.3
        hi = max(np.nanmax(x), df[[f"trace_mwle_{dn}", f"dopt_mwle_{dn}"]].max().max()) + 0.3
        ax.plot([lo, hi], [lo, hi], ls="--", color="gray", lw=1, label="y = x")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel(f"full-bank EAP ({dn})")
        ax.set_ylabel(f"CAT MWLE ({dn})")
        ax.set_title(f"Recovery: trace vs D-opt selection ({dn})")
        ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout(); fig.savefig(fig_dir / f"recovery_trace_vs_dopt_{dn}.png", dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
