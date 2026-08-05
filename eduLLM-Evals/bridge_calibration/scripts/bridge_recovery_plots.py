"""Bridge OOS recovery plots (reproduces the two BiGGen-style figures).

Held-out 5-fold, unidimensional ("general" ability):
  * Plot 1 -- CAT MWLE theta (y) vs held-out full-bank EAP theta (x).
  * Plot 2 -- p-IRT predicted pass rate (x, TRAIN params @ OOS MWLE theta) vs actual
    held-out pass rate (y).

Per fold: refit item (a,b) on TRAIN models (calibrate_mirt EM, Q=1), then for each held-out
model compute the full-bank EAP reference, run an item-level Fisher-info CAT with an MWLE
(Warm) ability estimate (min items + SE target), and predict pass rate at that theta.
Applies Bridge's source_id grouping. Preliminary (small N); item-level unidim reproduction
of the scenario-level engine.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit, logsumexp, log_expit

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import calibrate_partial as cp   # noqa: E402
import calibrate_mirt as cm      # noqa: E402


# ---------- data prep (matches bridge_calibration_study) ----------
def _load_map(path, key, val):
    out = {}
    for rec in cp.read_jsonl(path):
        k = rec.get(key)
        if k is not None:
            v = rec.get(val)
            out[str(k)] = str(v) if v is not None else str(k)
    return out


def dedupe_by_source(columns, c2s, s2src):
    grp = {}
    for c in columns:
        grp.setdefault(s2src.get(c2s.get(c, c), c2s.get(c, c)), set()).add(c2s.get(c, c))
    keep = {sorted(v)[0] for v in grp.values()}
    return [c for c in columns if c2s.get(c, c) in keep]


# ---------- unidimensional estimators ----------
def eap_theta(Y, M, a, b, grid, logw):
    eta = a[:, None] * grid[None, :] - b[:, None]
    LL = np.where(M, Y, 0.0) @ log_expit(eta) + np.where(M, 1.0 - Y, 0.0) @ log_expit(-eta)
    post = np.exp((LL + logw[None, :]) - logsumexp(LL + logw[None, :], axis=1, keepdims=True))
    return post @ grid


def mwle_theta(y, a, b, x0=0.0, iters=60):
    """Warm's weighted-likelihood estimate for a 1-D 2PL over the given items."""
    th = float(x0)
    for _ in range(iters):
        P = expit(a * th - b); Q = 1.0 - P
        J = np.sum(a * a * P * Q) + 1e-9
        Jp = np.sum(a ** 3 * P * Q * (Q - P))
        S = np.sum(a * (y - P)) + Jp / (2.0 * J)
        step = S / J
        th += float(np.clip(step, -1.5, 1.5))
        th = float(np.clip(th, -6.0, 6.0))
        if abs(step) < 1e-4:
            break
    return th


def run_cat(y, a, b, min_items=8, se_target=0.12, max_items=300):
    """Item-level Fisher-info CAT; MWLE ability. Returns (theta, n_items)."""
    n = a.size
    used = np.zeros(n, bool)
    th = 0.0
    order = []
    for step in range(min(max_items, n)):
        P = expit(a * th - b); info = a * a * P * (1.0 - P)
        info[used] = -1.0
        j = int(np.argmax(info))
        used[j] = True; order.append(j)
        idx = np.array(order)
        th = mwle_theta(y[idx], a[idx], b[idx], x0=th)
        Pu = expit(a[idx] * th - b[idx])
        se = 1.0 / np.sqrt(np.sum(a[idx] ** 2 * Pu * (1.0 - Pu)) + 1e-9)
        if len(order) >= min_items and se <= se_target:
            break
    return th, len(order)


# ---------- stats + plotting ----------
def boot_ci(x, y, fn, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    m = len(x)
    for _ in range(n):
        i = rng.integers(0, m, m)
        vals.append(fn(x[i], y[i]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def _r(x, y):
    return float(np.corrcoef(x, y)[0, 1])


def _slope(x, y):
    return float(np.polyfit(x, y, 1)[0])


def scatter_panel(ax, x, y, xlabel, ylabel, title):
    ax.scatter(x, y, s=28, alpha=0.75, color="#4C86C6", edgecolor="k", linewidth=0.3)
    lo = float(min(x.min(), y.min())) - 0.3
    hi = float(max(x.max(), y.max())) + 0.3
    ax.plot([lo, hi], [lo, hi], ls="--", color="gray", lw=1, label="y = x")
    m, c = np.polyfit(x, y, 1)
    xs = np.linspace(lo, hi, 100)
    ax.plot(xs, m * xs + c, color="#E1812C", lw=2, label=f"OLS fit (slope={m:.3f})")
    # simple CI band via bootstrap of the line
    rng = np.random.default_rng(0); lines = []
    for _ in range(400):
        i = rng.integers(0, len(x), len(x))
        mm, cc = np.polyfit(x[i], y[i], 1); lines.append(mm * xs + cc)
    lines = np.array(lines)
    ax.fill_between(xs, np.percentile(lines, 2.5, 0), np.percentile(lines, 97.5, 0),
                    color="#E1812C", alpha=0.20)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=10)
    ax.legend(loc="upper left", fontsize=9)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matrix", type=Path, required=True)
    p.add_argument("--rubrics", type=Path, required=True)
    p.add_argument("--scenarios", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--benchmark", default="Bridge")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--grid", type=int, default=41)
    p.add_argument("--eap-grid", type=int, default=61)
    p.add_argument("--ridge", type=float, default=1e-2)
    p.add_argument("--se-target", type=float, default=0.12)
    p.add_argument("--min-items", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260729)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    mat = cp.load_matrix(args.matrix)
    c2s = _load_map(args.rubrics, "criterion_id", "scenario_id")
    s2src = _load_map(args.scenarios, "scenario_id", "source_id")
    mat = mat[dedupe_by_source(list(mat.columns), c2s, s2src)]
    sub, _ = cp.select_sparse(mat)
    kept, _af, _ap = cp.split_zero_variance(sub)
    sub = sub[kept]; sub = sub[sub.notna().any(axis=1)]
    models = list(sub.index)
    Yall = np.nan_to_num(sub.to_numpy(float), nan=0.0)
    Mall = sub.notna().to_numpy()
    n = len(models)
    print(f"{args.benchmark}: {sub.shape[1]} items x {n} models; {args.k}-fold OOS")

    egrid = cm.build_grid(1, args.eap_grid)[:, 0]
    elogw = cm.base_log_weights(1, args.eap_grid); elogw -= logsumexp(elogw)

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(n)
    folds = [order[i::args.k] for i in range(args.k)]

    rows = []
    for fi, te in enumerate(folds):
        tr = np.array([i for i in range(n) if i not in set(te.tolist())])
        # keep items with train variance
        Mtr = Mall[tr]; Ytr = Yall[tr]
        keep = [j for j in range(sub.shape[1])
                if Mtr[:, j].sum() >= 2 and 0 < Ytr[Mtr[:, j], j].sum() < Mtr[:, j].sum()]
        keep = np.array(keep)
        Q = np.ones((len(keep), 1), int)
        fit = cm.fit_m2pl_em(np.nan_to_num(Ytr[:, keep]), Mtr[:, keep], Q, args.grid,
                             ridge=args.ridge, max_iter=100, tol=1e-4)
        a = fit["A"][:, 0]; b = fit["b"]
        print(f"  fold {fi}: train={len(tr)} test={len(te)} items={len(keep)} "
              f"converged={fit['converged']}", flush=True)
        Yte = Yall[np.ix_(te, keep)]; Mte = Mall[np.ix_(te, keep)]
        theta_ref = eap_theta(Yte, Mte, a, b, egrid, elogw)
        for ti, mi in enumerate(te):
            y = Yte[ti]
            th_cat, n_items = run_cat(y, a, b, args.min_items, args.se_target)
            pred = float(np.mean(expit(a * th_cat - b)))          # p-IRT predicted pass rate
            actual = float(np.mean(np.where(Mte[ti], Yte[ti], np.nan)[~np.isnan(
                np.where(Mte[ti], Yte[ti], np.nan))]))            # observed held-out pass rate
            rows.append({"model": models[mi], "fold": fi,
                         "theta_ref": float(theta_ref[ti]), "theta_mwle": float(th_cat),
                         "n_items": n_items, "pred_pass": pred, "actual_pass": actual})

    with (args.out_dir / "oos_per_model.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    x1 = np.array([r["theta_ref"] for r in rows]); y1 = np.array([r["theta_mwle"] for r in rows])
    x2 = np.array([r["pred_pass"] for r in rows]); y2 = np.array([r["actual_pass"] for r in rows])
    r1 = _r(x1, y1); s1 = _slope(x1, y1)
    r1lo, r1hi = boot_ci(x1, y1, _r); s1lo, s1hi = boot_ci(x1, y1, _slope)
    r2 = _r(x2, y2); s2 = _slope(x2, y2)
    r2lo, r2hi = boot_ci(x2, y2, _r)
    passmae = float(np.mean(np.abs(x2 - y2))); thmae = float(np.mean(np.abs(x1 - y1)))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig1, ax = plt.subplots(figsize=(6.2, 6.0))
    scatter_panel(ax, x1, y1, "held-out full-bank EAP theta (general)",
                  "CAT MWLE theta (general)",
                  f"{args.benchmark} OOS recovery (unidim, MWLE, k={args.k}) @ SE={args.se_target}/floor {args.min_items}\n"
                  f"r={r1:.3f} [{r1lo:.3f},{r1hi:.3f}]  slope={s1:.3f} [{s1lo:.3f},{s1hi:.3f}]  n={n}")
    fig1.tight_layout(); fig1.savefig(args.out_dir / f"{args.benchmark}_oos_recovery_theta.png", dpi=150)

    fig2, ax = plt.subplots(figsize=(6.2, 6.0))
    scatter_panel(ax, x2, y2, "predicted pass rate (p-IRT, TRAIN params @ OOS MWLE theta)",
                  "actual held-out pass rate",
                  f"{args.benchmark} p-IRT pred vs actual - OOS (k={args.k}, held-out)\n"
                  f"r={r2:.3f}[{r2lo:.3f},{r2hi:.3f}]  passMAE={passmae:.4f}  "
                  f"theta-MAE={thmae:.3f}  slope={s2:.3f}  n={n}")
    fig2.tight_layout(); fig2.savefig(args.out_dir / f"{args.benchmark}_pirt_pass_calibration.png", dpi=150)

    (args.out_dir / "recovery_metrics.json").write_text(json.dumps(
        {"theta_recovery": {"r": r1, "r_ci": [r1lo, r1hi], "slope": s1, "slope_ci": [s1lo, s1hi],
                            "theta_mae": thmae},
         "pass_rate": {"r": r2, "r_ci": [r2lo, r2hi], "pass_mae": passmae, "slope": s2},
         "n_models": n, "k": args.k, "se_target": args.se_target, "min_items": args.min_items},
        indent=2), encoding="utf-8")

    print(f"\ntheta recovery : r={r1:.3f} [{r1lo:.3f},{r1hi:.3f}]  slope={s1:.3f}")
    print(f"pass-rate      : r={r2:.3f}  passMAE={passmae:.4f}  theta-MAE={thmae:.3f}")
    print(f"wrote -> {args.out_dir}/  ({args.benchmark}_oos_recovery_theta.png, "
          f"{args.benchmark}_pirt_pass_calibration.png, oos_per_model.csv, recovery_metrics.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
