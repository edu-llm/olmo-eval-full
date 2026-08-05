"""Bridge robustness pass: estimator comparison (10), ridge/grid sensitivity (12),
and order/seed stability (11). All reuse the validated unidim estimators.

  * estimator  -- OOS recovery r for EAP vs MWLE vs MLE on the SAME CAT-administered
                  subset (does the ability point-estimator matter at this N?).
  * sensitivity -- full-fit a/theta stability + OOS r across ridge and fit-grid settings.
  * order/seed -- theta SD across seeded CATs (random start + random tie-break); also
                  notes trace vs D-optimal item selection (identical in 1D -- info is a
                  scalar -- so D-opt only differs under a multidim fit, which does not
                  identify at N=51).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit, log_expit, logsumexp

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import calibrate_partial as cp          # noqa: E402
import calibrate_mirt as cm             # noqa: E402
import bridge_recovery_plots as brp     # noqa: E402


def prep(args):
    mat = cp.load_matrix(args.matrix)
    c2s = brp._load_map(args.rubrics, "criterion_id", "scenario_id")
    s2src = brp._load_map(args.scenarios, "scenario_id", "source_id")
    mat = mat[brp.dedupe_by_source(list(mat.columns), c2s, s2src)]
    sub, _ = cp.select_sparse(mat)
    kept, _af, _ap = cp.split_zero_variance(sub)
    sub = sub[kept]; sub = sub[sub.notna().any(axis=1)]
    Y = np.nan_to_num(sub.to_numpy(float), nan=0.0); M = sub.notna().to_numpy()
    return list(sub.index), Y, M


def mle_theta(y, a, b, egrid):
    eta = a[:, None] * egrid[None, :] - b[:, None]
    ll = y @ log_expit(eta) + (1.0 - y) @ log_expit(-eta)
    return float(egrid[int(np.argmax(ll))])


def cat_order(y, a, b, min_items, se_target, max_items=300, seed=None):
    """Fisher-max CAT; returns administered index order. seed!=None randomizes start
    and breaks near-ties randomly (for order/seed stability)."""
    rng = np.random.default_rng(seed) if seed is not None else None
    n = a.size; used = np.zeros(n, bool); th = 0.0; order = []
    for _ in range(min(max_items, n)):
        P = expit(a * th - b); info = a * a * P * (1.0 - P); info[used] = -np.inf
        if rng is not None and not order:
            j = int(rng.integers(0, n))
        elif rng is not None:
            top = np.where(info >= info.max() - 1e-6)[0]
            j = int(rng.choice(top))
        else:
            j = int(np.argmax(info))
        used[j] = True; order.append(j)
        idx = np.array(order); th = brp.mwle_theta(y[idx], a[idx], b[idx], x0=th)
        Pu = expit(a[idx] * th - b[idx])
        se = 1.0 / np.sqrt(np.sum(a[idx] ** 2 * Pu * (1.0 - Pu)) + 1e-9)
        if len(order) >= min_items and se <= se_target:
            break
    return np.array(order)


def kfold(n, k, seed):
    order = np.random.default_rng(seed).permutation(n)
    return [order[i::k] for i in range(k)]


def _r(x, y):
    return float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else float("nan")


def exp_estimator(models, Y, M, args, egrid, elogw, out):
    folds = kfold(len(Y), args.k, args.seed); J = Y.shape[1]
    ref, e_eap, e_mwle, e_mle = [], [], [], []
    for te in folds:
        tr = np.array([i for i in range(len(Y)) if i not in set(te.tolist())])
        keep = np.array([j for j in range(J) if M[tr, j].sum() >= 2 and
                         0 < Y[tr][M[tr, j], j].sum() < M[tr, j].sum()])
        ft = cm.fit_m2pl_em(Y[np.ix_(tr, keep)], M[np.ix_(tr, keep)], np.ones((len(keep), 1), int),
                            args.grid, ridge=args.ridge, max_iter=100, tol=1e-4)
        a, b = ft["A"][:, 0], ft["b"]
        Yte, Mte = Y[np.ix_(te, keep)], M[np.ix_(te, keep)]
        rf = brp.eap_theta(Yte, Mte, a, b, egrid, elogw)
        for ti in range(len(te)):
            y = Yte[ti]; od = cat_order(y, a, b, args.min_items, args.se_target)
            ref.append(rf[ti])
            e_eap.append(brp.eap_theta(y[od][None, :], np.ones((1, len(od)), bool),
                                       a[od], b[od], egrid, elogw)[0])
            e_mwle.append(brp.mwle_theta(y[od], a[od], b[od]))
            e_mle.append(mle_theta(y[od], a[od], b[od], egrid))
    ref = np.array(ref)
    res = {"EAP": _r(ref, np.array(e_eap)), "MWLE": _r(ref, np.array(e_mwle)),
           "MLE": _r(ref, np.array(e_mle)), "n": len(ref)}
    (out / "estimator_comparison.json").write_text(json.dumps(res, indent=2))
    print(f"  estimator r: EAP={res['EAP']:.3f} MWLE={res['MWLE']:.3f} MLE={res['MLE']:.3f}")
    return res


def exp_sensitivity(models, Y, M, args, egrid, elogw, out):
    J = Y.shape[1]
    base = cm.fit_m2pl_em(Y, M, np.ones((J, 1), int), 41, ridge=1e-2, max_iter=150, tol=1e-4)
    a0, th0 = base["A"][:, 0], brp.eap_theta(Y, M, base["A"][:, 0], base["b"], egrid, elogw)
    rows = []
    for ridge in [1e-3, 1e-2, 1e-1]:
        for grid in [21, 41]:
            ft = cm.fit_m2pl_em(Y, M, np.ones((J, 1), int), grid, ridge=ridge, max_iter=150, tol=1e-4)
            a, th = ft["A"][:, 0], brp.eap_theta(Y, M, ft["A"][:, 0], ft["b"], egrid, elogw)
            # OOS r: held-out EAP (train params) vs baseline full-data theta
            oos = []
            for te in kfold(len(Y), args.k, args.seed):
                tr = np.array([i for i in range(len(Y)) if i not in set(te.tolist())])
                keep = np.array([j for j in range(J) if M[tr, j].sum() >= 2 and
                                 0 < Y[tr][M[tr, j], j].sum() < M[tr, j].sum()])
                f2 = cm.fit_m2pl_em(Y[np.ix_(tr, keep)], M[np.ix_(tr, keep)],
                                    np.ones((len(keep), 1), int), grid, ridge=ridge, max_iter=100, tol=1e-4)
                thh = brp.eap_theta(Y[np.ix_(te, keep)], M[np.ix_(te, keep)], f2["A"][:, 0], f2["b"], egrid, elogw)
                oos.append((th0[te], thh))
            oref = np.concatenate([o[0] for o in oos]); oest = np.concatenate([o[1] for o in oos])
            rows.append({"ridge": ridge, "grid": grid, "corr_a_vs_base": round(_r(a0, a), 4),
                         "corr_theta_vs_base": round(_r(th0, th), 4), "oos_r": round(_r(oref, oest), 4)})
            print(f"  ridge={ridge:<6} grid={grid}: corr_a={rows[-1]['corr_a_vs_base']:.3f} "
                  f"oos_r={rows[-1]['oos_r']:.3f}")
    with (out / "ridge_grid_sensitivity.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    return rows


def exp_order_seed(models, Y, M, args, egrid, out):
    J = Y.shape[1]
    ft = cm.fit_m2pl_em(Y, M, np.ones((J, 1), int), args.grid, ridge=args.ridge, max_iter=150, tol=1e-4)
    a, b = ft["A"][:, 0], ft["b"]
    sds = []
    for mi in range(len(Y)):
        obs = np.where(M[mi])[0]
        ths = []
        for s in range(args.n_seeds):
            od = cat_order(Y[mi][obs], a[obs], b[obs], args.min_items, args.se_target, seed=1000 + s)
            ths.append(brp.mwle_theta(Y[mi][obs][od], a[obs][od], b[obs][od]))
        sds.append(float(np.std(ths)))
    res = {"n_seeds": args.n_seeds, "mean_theta_sd_across_seeds": float(np.mean(sds)),
           "max_theta_sd_across_seeds": float(np.max(sds)),
           "note": ("theta SD across seeded CATs (random start + random tie-break) under the "
                    "SE-stopping rule; near-zero SD => ability is order-robust (the stop rule "
                    "administers enough items that the final estimate does not depend on order). "
                    "trace vs D-optimal selection coincide in 1D (Fisher info is scalar); "
                    "D-optimality only differs under a multidim fit, which does not identify at N=51.")}
    (out / "order_seed_stability.json").write_text(json.dumps(res, indent=2))
    print(f"  order/seed: mean theta SD across {args.n_seeds} seeds = {res['mean_theta_sd_across_seeds']:.4f}")
    return res


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matrix", type=Path, required=True)
    p.add_argument("--rubrics", type=Path, required=True)
    p.add_argument("--scenarios", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--grid", type=int, default=41)
    p.add_argument("--eap-grid", type=int, default=61)
    p.add_argument("--ridge", type=float, default=1e-2)
    p.add_argument("--se-target", type=float, default=0.15)
    p.add_argument("--min-items", type=int, default=20)
    p.add_argument("--n-seeds", type=int, default=10)
    p.add_argument("--seed", type=int, default=20260729)
    p.add_argument("--only", default="estimator,sensitivity,order",
                   help="comma list of sub-experiments to run.")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    models, Y, M = prep(args)
    egrid = cm.build_grid(1, args.eap_grid)[:, 0]
    elogw = cm.base_log_weights(1, args.eap_grid); elogw -= logsumexp(elogw)
    want = {s.strip() for s in args.only.split(",") if s.strip()}
    print(f"sensitivity: {Y.shape[1]} items x {len(Y)} models; running {sorted(want)}")
    if "estimator" in want:
        exp_estimator(models, Y, M, args, egrid, elogw, args.out_dir)
    if "sensitivity" in want:
        exp_sensitivity(models, Y, M, args, egrid, elogw, args.out_dir)
    if "order" in want:
        exp_order_seed(models, Y, M, args, egrid, args.out_dir)
    print(f"wrote -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
