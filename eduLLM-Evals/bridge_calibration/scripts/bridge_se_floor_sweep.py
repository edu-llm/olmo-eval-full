"""Bridge SE-target x floor sweep (like BiGGen's 06_floor_se_grid).

For a grid of (SE target, floor=min criteria) it reports held-out OOS recovery
(CAT-MWLE theta vs full-bank EAP theta) and efficiency (mean criteria administered,
convergence rate), then picks the "knee": the shortest test that keeps recovery near
its max at 100% convergence.

Efficiency: item params depend only on the fold, and the greedy Fisher-info item
ORDER doesn't depend on SE/floor (only the stop point does). So we fit each fold once,
run each held-out model's full CAT trajectory once, and read off every (SE, floor).
Unidimensional ("general"), MWLE, k=5 -- same recipe as BiGGen frq/biggen.
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


def _load_map(path, key, val):
    out = {}
    for rec in cp.read_jsonl(path):
        k = rec.get(key)
        if k is not None:
            v = rec.get(val)
            out[str(k)] = str(v) if v is not None else str(k)
    return out


def dedupe_by_source(cols, c2s, s2src):
    grp = {}
    for c in cols:
        grp.setdefault(s2src.get(c2s.get(c, c), c2s.get(c, c)), set()).add(c2s.get(c, c))
    keep = {sorted(v)[0] for v in grp.values()}
    return [c for c in cols if c2s.get(c, c) in keep]


def eap_theta(Y, M, a, b, grid, logw):
    eta = a[:, None] * grid[None, :] - b[:, None]
    LL = np.where(M, Y, 0.0) @ log_expit(eta) + np.where(M, 1.0 - Y, 0.0) @ log_expit(-eta)
    post = np.exp((LL + logw[None, :]) - logsumexp(LL + logw[None, :], axis=1, keepdims=True))
    return post @ grid


def mwle_theta(y, a, b, x0=0.0, iters=60):
    th = float(x0)
    for _ in range(iters):
        P = expit(a * th - b); Q = 1.0 - P
        J = np.sum(a * a * P * Q) + 1e-9
        Jp = np.sum(a ** 3 * P * Q * (Q - P))
        step = (np.sum(a * (y - P)) + Jp / (2.0 * J)) / J
        th += float(np.clip(step, -1.5, 1.5)); th = float(np.clip(th, -6.0, 6.0))
        if abs(step) < 1e-4:
            break
    return th


def cat_trajectory(y, a, b, cap=300, se_stop=0.09):
    """Greedy Fisher-info CAT; returns per-step (theta, SE) arrays (SE/floor-independent)."""
    n = a.size; used = np.zeros(n, bool); th = 0.0; ths = []; ses = []
    for _ in range(min(cap, n)):
        P = expit(a * th - b); info = a * a * P * (1.0 - P); info[used] = -1.0
        used[int(np.argmax(info))] = True
        idx = np.where(used)[0]
        th = mwle_theta(y[idx], a[idx], b[idx], th)
        Pu = expit(a[idx] * th - b[idx])
        se = 1.0 / np.sqrt(np.sum(a[idx] ** 2 * Pu * (1.0 - Pu)) + 1e-9)
        ths.append(th); ses.append(se)
        if se <= se_stop:
            break
    return np.array(ths), np.array(ses)


def read_off(ths, ses, se_target, floor):
    K = len(ths)
    for k in range(K):
        if (k + 1) >= floor and ses[k] <= se_target:
            return ths[k], k + 1, True
    return ths[-1], K, bool(ses[-1] <= se_target)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matrix", type=Path, required=True)
    p.add_argument("--rubrics", type=Path, required=True)
    p.add_argument("--scenarios", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--se-list", default="0.10,0.12,0.15,0.20,0.25,0.30")
    p.add_argument("--floor-list", default="0,4,8,12,16,20")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--grid", type=int, default=41)
    p.add_argument("--eap-grid", type=int, default=61)
    p.add_argument("--ridge", type=float, default=1e-2)
    p.add_argument("--cap", type=int, default=300)
    p.add_argument("--seed", type=int, default=20260729)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    se_list = [float(s) for s in args.se_list.split(",")]
    floor_list = [int(s) for s in args.floor_list.split(",")]
    se_stop = min(se_list) - 0.02

    mat = cp.load_matrix(args.matrix)
    c2s = _load_map(args.rubrics, "criterion_id", "scenario_id")
    s2src = _load_map(args.scenarios, "scenario_id", "source_id")
    mat = mat[dedupe_by_source(list(mat.columns), c2s, s2src)]
    sub, _ = cp.select_sparse(mat)
    kept, _af, _ap = cp.split_zero_variance(sub)
    sub = sub[kept]; sub = sub[sub.notna().any(axis=1)]
    Yall = np.nan_to_num(sub.to_numpy(float), nan=0.0); Mall = sub.notna().to_numpy()
    n = sub.shape[0]
    print(f"Bridge sweep: {sub.shape[1]} items x {n} models; SE={se_list} floor={floor_list}")

    egrid = cm.build_grid(1, args.eap_grid)[:, 0]
    elogw = cm.base_log_weights(1, args.eap_grid); elogw -= logsumexp(elogw)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(n)
    folds = [order[i::args.k] for i in range(args.k)]

    # cache per (model): theta_ref and full CAT trajectory (fold params)
    theta_ref = np.full(n, np.nan)
    traj = {}
    for fi, te in enumerate(folds):
        tr = np.array([i for i in range(n) if i not in set(te.tolist())])
        Mtr, Ytr = Mall[tr], Yall[tr]
        keep = [j for j in range(sub.shape[1])
                if Mtr[:, j].sum() >= 2 and 0 < Ytr[Mtr[:, j], j].sum() < Mtr[:, j].sum()]
        keep = np.array(keep)
        fit = cm.fit_m2pl_em(np.nan_to_num(Ytr[:, keep]), Mtr[:, keep], np.ones((len(keep), 1), int),
                             args.grid, ridge=args.ridge, max_iter=100, tol=1e-4)
        a, b = fit["A"][:, 0], fit["b"]
        Yte, Mte = Yall[np.ix_(te, keep)], Mall[np.ix_(te, keep)]
        tref = eap_theta(Yte, Mte, a, b, egrid, elogw)
        print(f"  fold {fi}: train={len(tr)} test={len(te)} items={len(keep)}", flush=True)
        for ti, mi in enumerate(te):
            theta_ref[mi] = tref[ti]
            traj[mi] = cat_trajectory(Yte[ti], a, b, cap=args.cap, se_stop=se_stop)

    # evaluate the grid
    rows = []
    for se in se_list:
        for fl in floor_list:
            th = np.empty(n); nit = np.empty(n); conv = np.empty(n)
            for mi in range(n):
                t, k, c = read_off(traj[mi][0], traj[mi][1], se, fl)
                th[mi], nit[mi], conv[mi] = t, k, c
            r = float(np.corrcoef(theta_ref, th)[0, 1])
            slope = float(np.polyfit(theta_ref, th, 1)[0])
            tmae = float(np.mean(np.abs(theta_ref - th)))
            rows.append({"se_target": se, "floor": fl, "r": round(r, 4), "slope": round(slope, 4),
                         "theta_mae": round(tmae, 4), "mean_items": round(float(nit.mean()), 1),
                         "max_items": int(nit.max()), "conv_rate": round(float(conv.mean()), 3)})

    with (args.out_dir / "sweep_results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # pick the knee: full convergence + r within 0.005 of best, then fewest mean_items
    # Convergence-first: only consider operating points that reach the SE target for
    # every model (conv_rate == 1.0). Among those, take r within 0.005 of the best and
    # pick the fewest criteria (the efficiency knee). Never fall back to a <100%-conv point.
    conv_rows = [x for x in rows if x["conv_rate"] >= 0.999]
    pool = conv_rows if conv_rows else rows
    rmax = max(x["r"] for x in pool)
    elig = [x for x in pool if x["r"] >= rmax - 0.005]
    best = min(elig, key=lambda x: x["mean_items"])
    (args.out_dir / "best.json").write_text(json.dumps(
        {"rule": "100% convergence + r within 0.005 of best, then fewest mean criteria",
         "best": best, "r_max": rmax}, indent=2))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    # r vs floor, one line per SE
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for se in se_list:
        xs = floor_list; ys = [next(x["r"] for x in rows if x["se_target"] == se and x["floor"] == fl) for fl in floor_list]
        ax.plot(xs, ys, marker="o", label=f"SE {se}")
    ax.set_xlabel("floor (min criteria)"); ax.set_ylabel("OOS recovery r"); ax.legend(fontsize=8)
    ax.set_title("Bridge OOS r vs floor, by SE target"); fig.tight_layout()
    fig.savefig(args.out_dir / "r_vs_floor_by_se.png", dpi=140); plt.close(fig)
    # efficiency frontier: r vs mean_items
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.scatter([x["mean_items"] for x in rows], [x["r"] for x in rows], s=28,
               c=[x["se_target"] for x in rows], cmap="viridis")
    ax.scatter([best["mean_items"]], [best["r"]], s=140, edgecolor="red", facecolor="none", linewidth=2,
               label=f"best: SE {best['se_target']}/floor {best['floor']}")
    ax.set_xlabel("mean criteria administered"); ax.set_ylabel("OOS recovery r")
    ax.set_title("Bridge efficiency frontier (color = SE target)"); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(args.out_dir / "efficiency_frontier.png", dpi=140); plt.close(fig)

    print("\nse_target floor    r   slope  theta_mae  mean_items  conv")
    for x in rows:
        print(f"  {x['se_target']:<8} {x['floor']:<5} {x['r']:.3f} {x['slope']:.3f}   "
              f"{x['theta_mae']:.3f}      {x['mean_items']:<6} {x['conv_rate']:.2f}")
    print(f"\nBEST: SE={best['se_target']} floor={best['floor']}  "
          f"r={best['r']} slope={best['slope']} mean_items={best['mean_items']} conv={best['conv_rate']}")
    print(f"wrote -> {args.out_dir}/ (sweep_results.csv, best.json, r_vs_floor_by_se.png, efficiency_frontier.png)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
