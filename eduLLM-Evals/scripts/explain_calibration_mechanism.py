"""Visual: the MECHANISM by which the 2PL curve adjusts itself during calibration.

Four standalone figures, each answering one question about how the fit actually works.
Companion to ``scripts/explain_calibration_progression.py``, which shows *that* the curve
moves; this one shows *why* it moves where it does.

    A  mechanism_a_two_dials.png      the curve has exactly two degrees of freedom
    B  mechanism_b_ability_spread.png  during fitting a model is a distribution, not a point
    C  mechanism_c_the_tally.png       what the curve is fitted to: expected counts at nodes
    D  mechanism_d_pulls_cancel.png    fitting = driving the weighted residuals to zero

Everything is the REAL fitter on the REAL matrix: the same loader, the same
``content+diagnosis`` collapse, the same Bock-Aitkin EM (``calibrate_mirt.fit_m2pl_em``),
and the same criterion the sibling script selects.

Read-only: imports the fitter, never modifies it.

Usage
-----
    python scripts/explain_calibration_mechanism.py
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
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
AXIS = "model ability (correctness)"

CRIMSON = "crimson"
GREY = "gray"
BLUE = "#4c8fbe"
DARKBLUE = "#1f5f8b"
ORANGE = "darkorange"


def posterior_over_grid(Y, M, A, b, grid, log_prior):
    """The E-step posterior, person x grid node (the sibling's ``eap`` before the mean)."""
    ym = np.where(M, Y, 0.0)
    nm = np.where(M, 1.0 - Y, 0.0)
    eta = A @ grid.T - b[:, None]
    joint = ym @ log_expit(eta) + nm @ log_expit(-eta) + log_prior[None, :]
    return np.exp(joint - logsumexp(joint, axis=1)[:, None])


def marginalise(grid, values, dim=0):
    """Sum a per-node quantity onto one latent axis. Returns (axis nodes, summed values)."""
    nodes = np.unique(grid[:, dim])
    out = np.stack([values[..., grid[:, dim] == t].sum(axis=-1) for t in nodes], axis=-1)
    return nodes, out


def expected_counts(Y, M, posterior):
    """Bock-Aitkin M-step expected counts: passes and attempts per item x node."""
    r_jg = np.where(M, Y, 0.0).T @ posterior
    N_jg = M.astype(float).T @ posterior
    return r_jg, N_jg


def _panel_box(ax, x, y, text, fontsize=8.0, va="top", ha="left"):
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=fontsize,
            bbox=dict(boxstyle="round,pad=0.38", fc="white", ec="0.75", alpha=0.94))


# ---------------------------------------------------------------------------
# figure A - the two dials
# ---------------------------------------------------------------------------


def figure_a(out_dir, a_fit, b_fit, a_warm, b_warm, cid):
    import matplotlib.pyplot as plt

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(9.8, 5.2), sharey=True)
    xs = np.linspace(-4.2, 4.2, 400)

    a_vals = [0.5, 1.0, 3.0, round(float(a_fit), 2)]
    shades = ["#b9c9d6", "#7fa6c4", "#3f6f96", CRIMSON]
    for a, col in zip(a_vals, shades):
        lw = 2.6 if col == CRIMSON else 1.9
        tag = f"a = {a:g}" + (f"  (fitted, {cid})" if col == CRIMSON else "")
        axl.plot(xs, expit(a * xs - 0.0), lw=lw, color=col, label=tag)
    axl.axvline(0.0, ls=":", lw=0.9, color="0.55")
    axl.annotate("a = %.2f is nearly a STEP:\nbelow $\\theta$ = 0 almost nobody passes,\n"
                 "above it almost everybody does" % a_fit,
                 xy=(0.30, expit(a_fit * 0.30)), xytext=(1.00, 0.52),
                 fontsize=8.2, ha="left", va="center",
                 arrowprops=dict(arrowstyle="->", color="0.35", lw=1.2),
                 bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.75", alpha=0.94))
    axl.set_title("Dial 1: $a$ sets the STEEPNESS\n($b$ held at 0, so every curve crosses "
                  "50% at $\\theta$ = 0)", fontsize=10.5)
    axl.legend(loc="upper left", fontsize=8.2, framealpha=0.94)

    b_vals = [-2.0, 0.0, round(float(b_warm), 2), 3.0]
    shades_b = ["#b9c9d6", "#7fa6c4", CRIMSON, "#3f6f96"]
    for b, col in zip(b_vals, shades_b):
        lw = 2.6 if col == CRIMSON else 1.9
        tag = f"b = {b:g}" + ("  (warm start)" if col == CRIMSON else "")
        axr.plot(xs, expit(1.0 * xs - b), lw=lw, color=col, label=tag)
        axr.plot([b], [0.5], "o", ms=5.5, color=col, mec="white", mew=0.9, zorder=5)
    axr.set_title("Dial 2: $b$ slides the curve LEFT / RIGHT\n($a$ held at 1.00; dots mark "
                  "the 50% point at $\\theta$ = $b/a$)", fontsize=10.5)
    axr.legend(loc="upper left", fontsize=8.2, framealpha=0.94)

    for ax in (axl, axr):
        ax.set_xlabel(AXIS)
        ax.set_ylim(-0.05, 1.10)
        ax.set_xlim(xs[0], xs[-1])
        ax.grid(alpha=0.18, lw=0.6)
    axl.set_ylabel("P(pass this criterion)")

    fig.tight_layout(rect=[0, 0.115, 1, 1])
    fig.text(0.5, 0.058,
             "These two numbers are the WHOLE model:  "
             "$P(\\mathrm{pass}) = \\sigma(a\\,\\theta - b)$.\n"
             "Calibration has nothing else to adjust - every curve in the other three "
             "figures is one of these.",
             ha="center", va="center", fontsize=9.2,
             bbox=dict(boxstyle="round,pad=0.5", fc="#f7f7f7", ec="0.75"))
    out = out_dir / "mechanism_a_two_dials.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote -> {out}")
    return out


# ---------------------------------------------------------------------------
# figure B - a model is a distribution over the grid
# ---------------------------------------------------------------------------


def figure_b(out_dir, nodes, weights, models, picks, fine_nodes, n_items_per_model):
    import matplotlib.pyplot as plt

    eap = weights @ nodes
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 5.9), sharey=True, sharex=True)
    occupied = nodes[weights.sum(axis=0) > 1e-4]
    lo, hi = occupied.min() - 1.0, occupied.max() + 1.0

    for ax, (label, i) in zip(axes, picks):
        w = weights[i]
        sd = float(np.sqrt(max(w @ nodes**2 - (w @ nodes) ** 2, 0.0)))
        # dedicated strip below the axis so the rug cannot be mistaken for a stem
        ax.axhspan(-0.115, -0.018, color="#eef3f7", zorder=0)
        ax.plot(eap, np.full(eap.shape, -0.066), "|", ms=12, color="0.35", alpha=0.7,
                zorder=2, label=f"where all {len(eap)} models' EAPs fall")
        ax.vlines(nodes, 0, w, lw=3.2, color=BLUE, alpha=0.9, zorder=3)
        ax.plot(nodes, w, "o", ms=4.5, color=DARKBLUE, zorder=4,
                label="posterior weight at a grid node")

        ax.axvline(eap[i], ls="--", lw=1.4, color=CRIMSON, zorder=1)
        ax.plot([eap[i]], [1.045], marker="v", ms=12, color=CRIMSON, clip_on=False,
                zorder=6, label="EAP point estimate (a summary)")

        top = int(np.argmax(w))
        # keep the callout on the empty side of the panel
        left = nodes[top] > 0.5 * (lo + hi)
        ax.annotate(f"{w[top]*100:.0f}% of this model's weight\nsits on the single node "
                    f"$\\theta$ = {nodes[top]:.2f}",
                    xy=(nodes[top], w[top]), xytext=(0.04 if left else 0.96, 0.60),
                    textcoords="axes fraction", fontsize=8.0,
                    ha="left" if left else "right", va="top",
                    arrowprops=dict(arrowstyle="->", color="0.35", lw=1.0),
                    bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="0.8", alpha=0.95))
        ax.set_title(f"{label}  -  {models[i]}\nEAP = {eap[i]:+.2f},   posterior SD = {sd:.3f},"
                     f"   {int((w > 0.01).sum())} node(s) above 1%", fontsize=10)
        ax.set_xlabel(AXIS)
        ax.grid(axis="y", alpha=0.18, lw=0.6)

    axes[0].set_ylabel("posterior weight")
    axes[0].set_ylim(-0.12, 1.08)
    axes[0].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axes[0].set_xlim(lo, hi)
    axes[0].legend(loc="upper left", fontsize=8.0, framealpha=0.95)

    fig.tight_layout(rect=[0, 0.155, 1, 1])
    fig.text(0.5, 0.075,
             f"The fitter never places a model at one ability. Each model enters the M-step "
             f"as these weights over the {fine_nodes}-node-per-dimension Gauss-Hermite grid "
             f"({fine_nodes ** 2} nodes, summed onto the\ncorrectness axis; the axis is "
             f"trimmed to the occupied nodes). The red arrow is a read-out of that spread, "
             f"not an input to the fit.\n"
             f"Honest caveat: every model here answers about {n_items_per_model:,} criteria, "
             f"so the spread is narrow - most models land almost entirely on one node. The "
             f"machinery is the same; this much data just pins it down hard.",
             ha="center", va="center", fontsize=8.6,
             bbox=dict(boxstyle="round,pad=0.5", fc="#f7f7f7", ec="0.75"))
    out = out_dir / "mechanism_b_ability_spread.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote -> {out}")
    return out


# ---------------------------------------------------------------------------
# figure C - the tally the curve is actually fitted to
# ---------------------------------------------------------------------------


def figure_c(out_dir, th, r_g, N_g, a, b, cid, grid_n, pool_err, n_models):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    ratio = np.where(N_g > 1e-9, r_g / np.maximum(N_g, 1e-12), np.nan)
    p = expit(a * th - b)
    # headroom: the bars and the curve are squeezed into the lower ~55% so the
    # explanatory boxes never sit on top of the data
    head = 1.80

    fig, ax = plt.subplots(figsize=(10.4, 6.4))
    w = 0.44
    ax.bar(th - w / 2, N_g, width=w, color=BLUE, alpha=0.9, edgecolor="white")
    ax.bar(th + w / 2, r_g, width=w, color=DARKBLUE, edgecolor="white")
    for t, n, r in zip(th, N_g, r_g):
        ax.text(t - w / 2, n + 0.3, f"{n:.1f}", ha="center", va="bottom", fontsize=8.2,
                color=DARKBLUE)
        ax.text(t + w / 2, r + 0.3, f"{r:.1f}", ha="center", va="bottom", fontsize=8.2,
                color=DARKBLUE)
    ax.set_ylabel("expected number of models (E-step weight)")
    ax.set_xlabel(AXIS + "  -  the %d quadrature nodes the M-step actually sees" % len(th))
    ax.set_ylim(-0.05 * N_g.max(), N_g.max() * head)
    ax.set_yticks(np.arange(0, N_g.max() + 1, 5))
    ax.set_xticks(th)
    ax.set_xticklabels([f"{t:.2f}" for t in th])
    ax.grid(axis="y", alpha=0.18, lw=0.6)

    ax2 = ax.twinx()
    xs = np.linspace(th[0] - 0.7, th[-1] + 0.7, 400)
    ax2.plot(xs, expit(a * xs - b), lw=2.6, color=CRIMSON, zorder=3)
    ax2.plot(th, ratio, "o", ms=10, mfc="none", mec="black", mew=1.9, zorder=4)
    ax2.set_ylabel("P(pass)", color=CRIMSON)
    ax2.tick_params(axis="y", labelcolor=CRIMSON)
    ax2.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax2.set_ylim(-0.05, head)
    ax2.set_xlim(xs[0], xs[-1])

    gap = float(np.nanmax(np.abs(ratio - p)[N_g > 0.5]))
    k = int(np.nanargmin(np.abs(np.where(N_g > 0.5, ratio, np.nan) - 0.5)))
    ax2.annotate(f"tally says $r_{{jg}}/N_{{jg}}$ = {ratio[k]:.3f}\n"
                 f"curve says $p_{{jg}}$ = {p[k]:.3f}",
                 xy=(th[k], ratio[k]), xytext=(0.30, 0.74), textcoords="axes fraction",
                 fontsize=9.0, ha="center", va="center", zorder=7,
                 arrowprops=dict(arrowstyle="->", color="0.3", lw=1.2),
                 bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.75", alpha=0.95))

    handles = [
        Patch(facecolor=BLUE, alpha=0.9, label="$N_{jg}$   expected attempts at this node"),
        Patch(facecolor=DARKBLUE, label="$r_{jg}$   expected passes at this node"),
        Line2D([], [], color=CRIMSON, lw=2.6,
               label=f"fitted curve  $\\sigma({a:.2f}\\,\\theta - {b:.2f})$   (right axis)"),
        Line2D([], [], color="black", marker="o", ls="none", ms=10, mfc="none", mew=1.9,
               label="observed ratio  $r_{jg}/N_{jg}$   (right axis)"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=9.0, framealpha=0.95)

    _panel_box(ax, 0.015, 0.975,
               f"The 2PL is NOT fitted to {n_models} pass/fail dots.\n"
               f"It is fitted to these {len(th)} (attempts, passes) tallies: the E-step hands\n"
               f"the M-step $N_{{jg}}$ and $r_{{jg}}$, and the curve is pulled through the\n"
               f"ratio $r_{{jg}}/N_{{jg}}$ at every node. Largest gap here: {gap:.4f}",
               fontsize=9.0, va="top", ha="left")
    _panel_box(ax, 0.985, 0.505,
               f"Grid: the fit's own {grid_n} x {grid_n} Gauss-Hermite\n"
               f"grid ({grid_n ** 2} nodes), summed onto the\n"
               f"correctness axis. That is a projection -\n"
               f"but a LOSSLESS one here: {cid}\n"
               f"loads on correctness only, so pooling\n"
               f"leaves the M-step's objective unchanged\n"
               f"(refitting from the pooled counts\n"
               f"reproduces $a$ and $b$ to {pool_err:.0e}).",
               fontsize=8.4, va="top", ha="right")

    fig.tight_layout()
    out = out_dir / "mechanism_c_the_tally.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote -> {out}")
    return out


# ---------------------------------------------------------------------------
# figure D - the pulls cancel
# ---------------------------------------------------------------------------


def figure_d(out_dir, panels, ridge, cid):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 6.9), sharey=True, sharex=True)
    warm = panels[0]
    xs = np.linspace(warm["th"][0] - 0.7, warm["th"][-1] + 0.7, 400)
    nmax = max(pl["N"].max() for pl in panels)
    notes = []

    for k, (ax, pl) in enumerate(zip(axes, panels)):
        th, r_g, N_g, a, b = pl["th"], pl["r"], pl["N"], pl["a"], pl["b"]
        ratio = np.where(N_g > 1e-9, r_g / np.maximum(N_g, 1e-12), np.nan)
        p = expit(a * th - b)

        ax.plot(xs, expit(warm["a"] * xs - warm["b"]), lw=1.6, ls="--", color=GREY,
                zorder=1, label="default (iteration 0)")
        if k > 0:
            ax.plot(xs, expit(a * xs - b), lw=2.6, color=CRIMSON, zorder=3,
                    label="fitted curve")

        for t, rt, pp, n in zip(th, ratio, p, N_g):
            if not np.isfinite(rt):
                continue
            wgt = float(n) / nmax
            ax.plot([t], [pp], "o", ms=4.5, color=CRIMSON if k else "0.35", zorder=5)
            if abs(rt - pp) > 0.008:
                ax.annotate("", xy=(t, rt), xytext=(t, pp), zorder=4,
                            arrowprops=dict(arrowstyle="-|>", color=ORANGE,
                                            mutation_scale=16 + 6 * wgt,
                                            lw=0.8 + 4.2 * wgt, alpha=0.5 + 0.45 * wgt,
                                            shrinkA=0, shrinkB=0))
            ax.plot([t], [rt], "o", ms=4 + 12 * wgt, mfc="none", mec="black",
                    mew=1.5, zorder=6)

        tot = float((r_g - N_g * p).sum())
        tilt = float((th * (r_g - N_g * p)).sum())
        gap = float(np.nanmax(np.abs(ratio - p)[N_g > 0.5]))
        extra = (f"\nridge $\\times\\, a$ = {ridge:g} $\\times$ {a:.2f} = {ridge * a:+.4f}"
                 f"\n$\\Rightarrow$ the tilt matches the PENALISED optimum to "
                 f"{abs(tilt - ridge * a):.0e};\n     the small residual is the ridge, "
                 f"not a failure to converge.") if k else ""
        _panel_box(ax, 0.035, 0.965,
                   f"matching conditions (units: models)\n"
                   f"total       $\\sum_g (r_{{jg}} - N_{{jg}}\\,p_{{jg}})$ = {tot:+.4f}\n"
                   f"tilt   $\\sum_g \\theta_g (r_{{jg}} - N_{{jg}}\\,p_{{jg}})$ = {tilt:+.4f}"
                   + extra,
                   fontsize=8.6, va="top", ha="left")
        notes.append(
            "The TOTAL is already near zero - the default $b$ comes straight from the\n"
            "pass rate, so it matches the headcount by construction. What is badly\n"
            "wrong is the TILT: the curve sits far too low where the strong models\n"
            f"are and too high below them. Largest single pull: {gap:.2f} in probability."
            if not k else
            "Every pull has shrunk to inside its own marker. The curve reproduces the\n"
            "tally at every node at once, and no reweighting of the nodes can improve\n"
            "it - which is all that \"fitted\" means.\n"
            f"Largest remaining gap: {gap:.4f} in probability.")
        ax.set_title(f"{pl['tag']}\na = {a:.2f},  b = {b:.2f}   "
                     f"(counts from this iteration's own E-step)", fontsize=10.5)
        ax.set_xlabel(AXIS)
        ax.grid(alpha=0.18, lw=0.6)

    handles = [
        Line2D([], [], color=GREY, ls="--", lw=1.6, label="default curve (iteration 0)"),
        Line2D([], [], color=CRIMSON, lw=2.6, label="fitted curve"),
        Line2D([], [], color=CRIMSON, marker="o", ls="none", ms=4.5,
               label="curve's prediction $p_{jg}$ at a node"),
        Line2D([], [], color="black", marker="o", ls="none", ms=9, mfc="none", mew=1.5,
               label="observed ratio $r_{jg}/N_{jg}$  (size $\\propto N_{jg}$)"),
        Line2D([], [], color=ORANGE, lw=3.0,
               label="the pull on the curve  (width $\\propto N_{jg}$)"),
    ]
    axes[0].set_ylabel("P(pass this criterion)")
    axes[0].set_ylim(-0.10, 1.62)
    axes[0].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axes[0].set_xlim(xs[0], xs[-1])

    fig.tight_layout(rect=[0, 0.235, 1, 1])
    for x, note in zip((0.268, 0.762), notes):
        fig.text(x, 0.155, note, ha="center", va="center", fontsize=8.6,
                 bbox=dict(boxstyle="round,pad=0.45", fc="#f7f7f7", ec="0.75"))
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8.8,
               framealpha=0.95, bbox_to_anchor=(0.5, 0.004))
    out = out_dir / "mechanism_d_pulls_cancel.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote -> {out}")
    return out


# ---------------------------------------------------------------------------


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
    p.add_argument("--fine-grid", type=int, default=21,
                   help="nodes/dim for the ability-spread figure (the sibling's EAP grid).")
    p.add_argument("--ridge", type=float, default=1e-2)
    p.add_argument("--max-iter", type=int, default=60)
    p.add_argument("--criterion", type=str, default="tb_0242_c04",
                   help="criterion to draw; the sibling script's furthest-travelling item.")
    args = p.parse_args()

    mat = cp.load_matrix(args.matrix)
    q_by = cm.load_q_matrix(args.rubrics)
    Y, M, Q3, items, block, diag = cm.prepare_block(mat, q_by)
    Q2, labels, _ = cm.collapse_q_matrix(Q3, COLLAPSE)
    models = [str(m) for m in block.index]
    print(f"block: {Y.shape[0]} models x {Y.shape[1]} criteria  (dims={labels})")

    Mf = M.astype(float)
    obs = Mf.sum(axis=0)
    rate = np.where(obs > 0, np.where(M, Y, 0.0).sum(axis=0) / np.maximum(obs, 1), 0.5)
    rate = np.clip(rate, 1e-3, 1 - 1e-3)
    A0 = np.zeros_like(Q2, dtype=float)
    A0[Q2 == 1] = 1.0
    b0 = -np.log(rate / (1 - rate))

    print(f"running EM (grid={args.grid}/dim, ridge={args.ridge}, max_iter={args.max_iter}) ...",
          flush=True)
    fit = cm.fit_m2pl_em(Y, M, Q2, args.grid, estimate_corr=False, ridge=args.ridge,
                         max_iter=args.max_iter, tol=1e-4)
    print(f"  EM finished: {fit['n_iter']} iterations, converged={fit['converged']}")

    if args.criterion not in items:
        print(f"ERROR: criterion {args.criterion} is not in the fitted block.", file=sys.stderr)
        return 2
    j = items.index(args.criterion)
    if not (abs(fit["A"][j, 0]) > 1e-9 and abs(fit["A"][j, 1]) < 1e-9):
        print(f"ERROR: {args.criterion} does not load on correctness alone; the 1-D "
              "presentation in these figures would be dishonest.", file=sys.stderr)
        return 3
    n_seen = int(M[:, j].sum())
    print(f"criterion {items[j]}: {n_seen} models,  "
          f"a: {A0[j,0]:.2f} -> {fit['A'][j,0]:.2f},  b: {b0[j]:.2f} -> {fit['b'][j]:.2f}")

    # --- E-step on the FIT grid: what the M-step genuinely consumes -----------
    grid = cm.build_grid(2, args.grid)
    log_prior = cm.prior_log_weights(grid, cm.base_log_weights(2, args.grid), np.eye(2))
    panels = []
    for tag, A_, b_ in [("warm start (iteration 0)", A0, b0),
                        (f"converged (iteration {fit['n_iter']})", fit["A"], fit["b"])]:
        post = posterior_over_grid(Y, M, A_, b_, grid, log_prior)
        r_jg, N_jg = expected_counts(Y, M, post)
        th, r_g = marginalise(grid, r_jg[j])
        _, N_g = marginalise(grid, N_jg[j])
        panels.append({"tag": tag, "th": th, "r": r_g, "N": N_g,
                       "a": float(A_[j, 0]), "b": float(b_[j]),
                       "r_full": r_jg[j], "N_full": N_jg[j]})

    conv = panels[-1]
    # the pooling in figures C/D is only honest if the M-step cannot tell the difference
    X_full = np.hstack([grid[:, [0]], np.ones((grid.shape[0], 1))])
    X_pool = np.hstack([conv["th"][:, None], np.ones((conv["th"].size, 1))])
    beta_full = cm._fit_item(X_full, conv["r_full"], conv["N_full"], args.ridge)
    beta_pool = cm._fit_item(X_pool, conv["r"], conv["N"], args.ridge)
    pool_err = float(np.max(np.abs(beta_full - beta_pool)))
    print(f"pooling check: refit from 49 nodes vs 7 pooled nodes differ by {pool_err:.2e}")

    print("\nmatching conditions for " + items[j] + ":")
    for pl in panels:
        pv = expit(pl["a"] * pl["th"] - pl["b"])
        res = pl["r"] - pl["N"] * pv
        print(f"  {pl['tag']:32s} total = {res.sum():+.6f}   "
              f"tilt = {(pl['th'] * res).sum():+.6f}   "
              f"(ridge*a = {args.ridge * pl['a']:+.6f})")

    # --- fine grid: the per-model ability posterior ---------------------------
    fine = cm.build_grid(2, args.fine_grid)
    fine_lp = cm.prior_log_weights(fine, cm.base_log_weights(2, args.fine_grid), np.eye(2))
    fine_post = posterior_over_grid(Y, M, fit["A"], fit["b"], fine, fine_lp)
    nodes, weights = marginalise(fine, fine_post)
    eap = weights @ nodes
    sd = np.sqrt(np.maximum(weights @ nodes**2 - eap**2, 0.0))

    sharp = sd <= np.quantile(sd, 0.6)
    hi = int(np.argmax(np.where(sharp, eap, -np.inf)))
    lo = int(np.argmin(np.where(sharp, eap, np.inf)))
    wide = int(np.argmax(sd))
    picks = [("confidently high", hi), ("confidently low", lo), ("genuinely uncertain", wide)]
    if len({hi, lo, wide}) < 3:
        print("WARNING: fewer than 3 distinct models matched the sharpness contrast.")
    print("\nability-spread models:")
    for label, i in picks:
        print(f"  {label:20s} {models[i]:44s} EAP={eap[i]:+.3f}  SD={sd[i]:.3f}")
    print(f"  posterior SD across all {len(models)} models: median={np.median(sd):.3f}  "
          f"max={sd.max():.3f}")

    import matplotlib
    matplotlib.use("Agg")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    figure_a(args.out_dir, fit["A"][j, 0], fit["b"][j], A0[j, 0], b0[j], items[j])
    figure_b(args.out_dir, nodes, weights, models, picks, args.fine_grid,
             int(np.median(M.sum(axis=1))))
    figure_c(args.out_dir, conv["th"], conv["r"], conv["N"], conv["a"], conv["b"],
             items[j], args.grid, pool_err, n_seen)
    figure_d(args.out_dir, panels, args.ridge, items[j])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
