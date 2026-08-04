"""Visual: how EM molds the default logistic curve into the fitted one.

Calibration starts every criterion from a crude default (discrimination = 1 wherever the
Q-matrix allows, difficulty from the raw pass rate) and then alternates E- and M-steps
until the curves stop moving. This script snapshots that progression by running the REAL
fitter (``scripts/calibrate_mirt.fit_m2pl_em``) at increasing iteration caps and drawing
the curve for one criterion at each stage, against the observed pass/fail data.

Honest caveat rendered on the figure: the horizontal positions are the FINAL calibrated
ability estimates. During fitting the abilities are not known -- each model is a
probability smear over the latent grid -- so this is the after-the-fact view of what the
procedure converged to, not a snapshot of what it "saw" at that iteration.

Read-only: imports the fitter, never modifies it.

Usage
-----
    python scripts/explain_calibration_progression.py
    python scripts/explain_calibration_progression.py --minimal

``--minimal`` keeps only the warm start and the converged fit, and drops every
panel-to-panel change indicator, writing to ``calibration_progression_simple.png``.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, log_expit, logsumexp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cm = _load("calibrate_mirt", ROOT / "scripts" / "calibrate_mirt.py")
cp = cm.cp

COLLAPSE = ["content", "diagnosis"]
LABELS = ["correctness", "scaffolding"]


def eap(Y, M, A, b, grid, log_prior):
    ym = np.where(M, Y, 0.0)
    nm = np.where(M, 1.0 - Y, 0.0)
    eta = A @ grid.T - b[:, None]
    ll = ym @ log_expit(eta) + nm @ log_expit(-eta)
    joint = ll + log_prior[None, :]
    post = np.exp(joint - logsumexp(joint, axis=1)[:, None])
    return post @ grid


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matrix", type=Path,
                   default=ROOT / "staging" / "response_matrix_full_nonopt.csv")
    p.add_argument("--rubrics", type=Path,
                   default=ROOT / "data" / "TutorBench" / "curated" / "rubrics_qmatrix_curated.jsonl")
    p.add_argument("--out-dir", type=Path,
                   default=ROOT / "regenerated_figures" / "twopl_explainer")
    p.add_argument("--grid", type=int, default=7)
    p.add_argument("--ridge", type=float, default=1e-2)
    p.add_argument("--snapshots", type=int, nargs="+", default=[1, 2, 5, 60])
    p.add_argument("--minimal", action="store_true",
                   help="draw only the warm start and the converged fit, with none of the "
                        "per-panel change annotations")
    args = p.parse_args()

    mat = cp.load_matrix(args.matrix)
    q_by = cm.load_q_matrix(args.rubrics)
    Y, M, Q3, items, _, diag = cm.prepare_block(mat, q_by)
    Q2, labels, _ = cm.collapse_q_matrix(Q3, COLLAPSE)
    print(f"block: {Y.shape[0]} models x {Y.shape[1]} criteria  (dims={labels})")

    # --- iteration 0: the warm start the fitter itself uses -------------------
    Mf = M.astype(float)
    obs = Mf.sum(axis=0)
    rate = np.where(obs > 0, np.where(M, Y, 0.0).sum(axis=0) / np.maximum(obs, 1), 0.5)
    rate = np.clip(rate, 1e-3, 1 - 1e-3)
    A0 = np.zeros_like(Q2, dtype=float)
    A0[Q2 == 1] = 1.0
    b0 = -np.log(rate / (1 - rate))

    stages = [("iteration 0\n(default warm start)", A0, b0)]
    fits = {}
    for it in args.snapshots:
        print(f"  running EM with max_iter={it} ...", flush=True)
        f = cm.fit_m2pl_em(Y, M, Q2, args.grid, estimate_corr=False,
                           ridge=args.ridge, max_iter=it, tol=1e-4)
        fits[it] = f
        tag = (f"iteration {f['n_iter']}\n(converged)" if f["converged"]
               else f"iteration {f['n_iter']}")
        stages.append((tag, f["A"], f["b"]))

    final = fits[args.snapshots[-1]]
    grid = cm.build_grid(2, 21)
    base = cm.base_log_weights(2, 21)
    log_prior = cm.prior_log_weights(grid, base, np.eye(2))
    theta = eap(Y, M, final["A"], final["b"], grid, log_prior)
    th = theta[:, 0]

    # --- pick a criterion that visibly moves ---------------------------------
    only_c = (np.abs(final["A"][:, 0]) > 1e-9) & (np.abs(final["A"][:, 1]) < 1e-9)
    nobs = M.sum(axis=0)
    ok = only_c & (nobs > 60) & (rate > 0.15) & (rate < 0.85)
    travel = np.abs(final["A"][:, 0] - A0[:, 0])
    j = int(np.where(ok)[0][np.argmax(travel[ok])]) if ok.any() else int(np.argmax(travel))
    print(f"\ncriterion drawn: {items[j]}  "
          f"a: {A0[j,0]:.2f} -> {final['A'][j,0]:.2f},  "
          f"b: {b0[j]:.2f} -> {final['b'][j]:.2f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.minimal:
        stages = [stages[0], stages[-1]]
    annotate = not args.minimal

    n = len(stages)
    fig, axes = plt.subplots(1, n,
                             figsize=((5.0 if args.minimal else 4.4) * n,
                                      4.8 if args.minimal else 5.2),
                             sharey=True, sharex=True)
    rng = np.random.default_rng(0)
    xs = np.linspace(th.min() - 0.6, th.max() + 0.6, 300)
    seen = M[:, j]
    x, y = th[seen], Y[seen, j]

    def curve(A_, b_, at):
        return expit(A_[j, 0] * at - b_[j])

    A_start, b_start = stages[0][1], stages[0][2]
    loads = Q2 == 1
    # zoom on the transition band, where the late iterations move too little to see
    centre = final["b"][j] / final["A"][j, 0]
    xz = np.linspace(centre - 1.0, centre + 1.0, 300)
    fit_label = "fitted curve" if args.minimal else "fitted curve at this stage"
    dot_label = "one model" if args.minimal else "one model (jittered)"

    for i, (ax, (tag, Aa, bb)) in enumerate(zip(axes, stages)):
        cur = curve(Aa, bb, xs)
        ax.scatter(x, y + rng.uniform(-0.035, 0.035, size=y.shape), s=16, alpha=0.4,
                   edgecolor="none", label=dot_label if i == 0 else None)
        # the default warm start, repeated on every panel as the reference to move away from
        ax.plot(xs, curve(A_start, b_start, xs), lw=1.5, ls="--", color="gray", zorder=1,
                label="default (iteration 0)" if i == 0 else None)

        if annotate:
            if i == 0:
                note = "starting point\n(no fitting yet)"
            else:
                Ap, bp = stages[i - 1][1], stages[i - 1][2]
                prv = curve(Ap, bp, xs)
                ax.fill_between(xs, prv, cur, color="darkorange", alpha=0.30, zorder=2)
                ax.plot(xs, prv, lw=1.1, color="darkorange", zorder=2)
                # the drawn criterion can be done moving while the rest of the bank is not
                bank = np.abs(Aa - Ap)[loads].mean()
                note = (f"vs previous panel\n"
                        f"$\\Delta a$ = {Aa[j,0]-Ap[j,0]:+.2f}   "
                        f"$\\Delta b$ = {bb[j]-bp[j]:+.2f}\n"
                        f"max curve shift = {np.abs(cur-prv).max():.3f}\n"
                        f"whole bank: mean |$\\Delta a$| = {bank:.3f}")

        ax.plot(xs, cur, lw=2.4, color="crimson", zorder=3,
                label=fit_label if i == 0 else None)

        if annotate:
            ax.text(0.03, 0.60, note, transform=ax.transAxes, ha="left", va="top",
                    fontsize=7.5,
                    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.8", alpha=0.9))

            axin = ax.inset_axes([0.58, 0.11, 0.38, 0.33], zorder=6)
            axin.patch.set_facecolor("white")
            axin.patch.set_alpha(1.0)
            for sp in axin.spines.values():
                sp.set_edgecolor("0.45")
                sp.set_linewidth(0.8)
            axin.plot(xz, curve(A_start, b_start, xz), lw=1.0, ls="--", color="gray")
            if i > 0:
                axin.fill_between(xz, curve(Ap, bp, xz), curve(Aa, bb, xz),
                                  color="darkorange", alpha=0.35)
                axin.plot(xz, curve(Ap, bp, xz), lw=1.0, color="darkorange")
            axin.plot(xz, curve(Aa, bb, xz), lw=1.8, color="crimson")
            axin.set_xlim(xz[0], xz[-1])
            axin.set_ylim(-0.03, 1.03)
            axin.set_yticks([0, 1])
            axin.set_xticks([round(centre - 1, 1), round(centre + 1, 1)])
            axin.tick_params(labelsize=6, length=2, pad=1)
            axin.set_title(f"zoom: ability {xz[0]:.1f} to {xz[-1]:.1f}", fontsize=6.5, pad=2)

        ax.set_title(f"{tag}\na = {Aa[j,0]:.2f},  b = {bb[j]:.2f}", fontsize=10)
        ax.set_xlabel("model ability (correctness)")
        ax.set_ylim(-0.12, 1.12)

    axes[0].set_ylabel("passed this criterion")
    handles, _ = axes[0].get_legend_handles_labels()
    if annotate:
        handles.append(Patch(facecolor="darkorange", alpha=0.30,
                             label="what moved since previous panel"))
    axes[0].legend(handles=handles, loc="upper left", fontsize=7, framealpha=0.92)
    # the minimal figure is captioned in the paper, so it carries no baked-in title
    if not args.minimal:
        fig.suptitle(
            f"EM molds the default curve onto the data - criterion {items[j]} "
            f"(the furthest-travelling item in the bank), {int(seen.sum())} models\n"
            "Orange = what changed since the previous panel. "
            "Ability positions are the final estimates; during fitting they are not yet known.",
            fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 1.0 if args.minimal else 0.90])
    out = args.out_dir / ("calibration_progression_simple.png" if args.minimal
                          else "calibration_progression.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
