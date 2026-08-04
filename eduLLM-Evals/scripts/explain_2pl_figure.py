"""Visual explainer for the 2PL: what the S-curve is actually fitted to.

Combines BOTH TutorBench grading runs (the 82-model cohort and the newer 33-model
tb33 cohort, which are disjoint) into one 115-model pool, scores every model with the
frozen calibrated bank, and then draws -- for a handful of real criteria -- the thing
calibration is fitting:

    x = the model's ability (theta)
    y = did that model pass this criterion (0 / 1)
    curve = the fitted logistic  P(pass) = sigmoid(a * theta - b)

Only criteria that load on a SINGLE skill are drawn, so the curve is genuinely a
function of one ability axis and the picture is honest. Panels are chosen to contrast
the two knobs: `a` (steepness / discrimination) and `b` (left-right shift / difficulty),
including a negative-`a` item where the curve slopes the wrong way.

Binned empirical pass-rates are overlaid so you can see the curve tracking the data
rather than just asserting it.

Read-only: reuses scripts/scenario_cat_lib.py for the bank loader and the EAP scorer.

Usage
-----
    python scripts/explain_2pl_figure.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location("scat", ROOT / "scripts" / "scenario_cat_lib.py")
scat = importlib.util.module_from_spec(_spec)
sys.modules["scat"] = scat
_spec.loader.exec_module(scat)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bank", type=Path,
                   default=ROOT / "data" / "TutorBench" / "rubrics_qmatrix_calibrated_2skill_fitted.jsonl")
    p.add_argument("--matrix-82", type=Path,
                   default=ROOT / "staging" / "response_matrix_full_nonopt.csv")
    p.add_argument("--matrix-33", type=Path,
                   default=ROOT / "tutorbench_tb33_grading" / "response_matrix" / "response_matrix.csv")
    p.add_argument("--out-dir", type=Path,
                   default=ROOT / "regenerated_figures" / "twopl_explainer")
    p.add_argument("--grid", type=int, default=201)
    p.add_argument("--range", type=float, default=8.0)
    args = p.parse_args()

    # ---- combine the two cohorts on their shared item set --------------------
    m82 = pd.read_csv(args.matrix_82, index_col=0)
    m33 = pd.read_csv(args.matrix_33, index_col=0)
    records, dims, neg = scat.load_fitted_bank(args.bank, "keep")   # keep negatives: we plot one
    ids = [r["criterion_id"] for r in records]
    shared = [c for c in ids if c in m82.columns and c in m33.columns]

    overlap = set(m82.index) & set(m33.index)
    combined = pd.concat([m82.reindex(columns=shared), m33.reindex(columns=shared)], axis=0)

    print("=" * 84)
    print("TUTORBENCH MODEL POOL")
    print("=" * 84)
    print(f"  run 1 (82-model cohort) : {m82.shape[0]:3d} models x {m82.shape[1]} criteria")
    print(f"  run 2 (tb33 cohort)     : {m33.shape[0]:3d} models x {m33.shape[1]} criteria")
    print(f"  overlap between cohorts : {len(overlap)}")
    print(f"  COMBINED                : {combined.shape[0]:3d} models")
    print(f"  bank items usable in both: {len(shared)} of {len(ids)}")
    obs = combined.notna().to_numpy()
    print(f"  combined graded cells   : {int(obs.sum()):,} ({obs.mean()*100:.2f}% filled)")

    # ---- score all 115 models with the frozen bank ---------------------------
    keep = [r for r in records if r["criterion_id"] in set(shared)]
    A = np.array([[float(r["discrimination"][d] or 0.0) for d in dims] for r in keep])
    b = np.array([float(r["difficulty"]) for r in keep])
    order = [r["criterion_id"] for r in keep]
    sub = combined.reindex(columns=order)
    Yraw = sub.to_numpy(dtype=float)
    mask = ~np.isnan(Yraw)
    Y = np.nan_to_num(Yraw, nan=0.0)

    grid, log_prior = scat.build_grid(len(dims), args.grid, args.range)
    print(f"\nscoring {combined.shape[0]} models by full-bank EAP ...")
    theta = scat.eap_all_models(Y, mask, A, b, grid, log_prior, 4096)
    k = dims.index("correctness")
    th = theta[:, k]
    print(f"  correctness ability: min={th.min():.2f} median={np.median(th):.2f} max={th.max():.2f}")

    # ---- pick single-skill criteria that illustrate the two knobs ------------
    only_corr = (np.abs(A[:, k]) > 1e-9) & (np.abs(A[:, 1 - k]) < 1e-9)
    npass = np.nansum(Yraw, axis=0)
    nobs = mask.sum(axis=0)
    rate = np.divide(npass, np.maximum(nobs, 1))
    usable = only_corr & (nobs > 100) & (rate > 0.03) & (rate < 0.97)

    def pick(pred, key, largest=True):
        cand = np.where(usable & pred)[0]
        if cand.size == 0:
            cand = np.where(usable)[0]
        vals = key[cand]
        return int(cand[np.argmax(vals) if largest else np.argmin(vals)])

    panels = [
        ("steep - high discrimination", pick(A[:, k] > 0, A[:, k], True)),
        ("flat - low discrimination", pick(A[:, k] > 0.05, -A[:, k], True)),
        ("hard - shifted right", pick(A[:, k] > 0.3, b, True)),
    ]
    negs = np.where(only_corr & (A[:, k] < 0) & (nobs > 100))[0]
    if negs.size:
        panels.append(("NEGATIVE discrimination - curve slopes down",
                       int(negs[np.argmin(A[negs, k])])))

    # ---- draw ----------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.out_dir.mkdir(parents=True, exist_ok=True)
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.3 * n, 4.3), sharey=True)
    axes = np.atleast_1d(axes)
    rng = np.random.default_rng(0)
    xs = np.linspace(th.min() - 0.5, th.max() + 0.5, 300)

    for ax, (label, j) in zip(axes, panels):
        obs_i = mask[:, j]
        x = th[obs_i]
        y = Yraw[obs_i, j]
        ax.scatter(x, y + rng.uniform(-0.035, 0.035, size=y.shape), s=16, alpha=0.45,
                   edgecolor="none", label="one model (jittered)")
        ax.plot(xs, sigmoid(A[j, k] * xs - b[j]), lw=2.2, color="crimson",
                label="fitted 2PL curve")
        # binned empirical pass rate
        qs = np.quantile(x, np.linspace(0, 1, 9))
        cx, cy = [], []
        for lo, hi in zip(qs[:-1], qs[1:]):
            sel = (x >= lo) & (x <= hi)
            if sel.sum() >= 4:
                cx.append(x[sel].mean()); cy.append(y[sel].mean())
        ax.plot(cx, cy, "o", ms=7, color="black", mfc="none", mew=1.6,
                label="observed rate (binned)")
        ax.axhline(0.5, ls=":", lw=0.8, color="gray")
        ax.set_title(f"{label}\na = {A[j, k]:.2f},  b = {b[j]:.2f}", fontsize=10)
        ax.set_xlabel("model ability (correctness)")
        ax.set_ylim(-0.12, 1.12)
    axes[0].set_ylabel("passed this criterion")
    axes[0].legend(loc="center left", fontsize=8, framealpha=0.9)
    fig.suptitle(
        f"What the 2PL fits: one criterion, {combined.shape[0]} TutorBench models "
        f"(82-model run + tb33 run)\n"
        "'a' sets the steepness, 'b' slides the curve left/right",
        fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out = args.out_dir / "what_the_2pl_fits.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)

    meta = {"n_models_run1": int(m82.shape[0]), "n_models_tb33": int(m33.shape[0]),
            "n_models_combined": int(combined.shape[0]), "cohort_overlap": len(overlap),
            "bank": str(args.bank), "items_shared": len(shared),
            "panels": [{"label": l, "criterion_id": order[j],
                        "a_correctness": float(A[j, k]), "b": float(b[j]),
                        "n_observed": int(mask[:, j].sum()),
                        "pass_rate": float(rate[j])} for l, j in panels]}
    with (args.out_dir / "panels.json").open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print("\npanels drawn:")
    for l, j in panels:
        print(f"  {l:46s} {order[j]:14s} a={A[j,k]:7.3f} b={b[j]:7.3f} "
              f"pass_rate={rate[j]:.3f}")
    print(f"\nwrote -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
