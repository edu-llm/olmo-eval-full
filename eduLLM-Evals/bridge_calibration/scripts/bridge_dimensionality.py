"""Bridge dimensionality comparison (experiment 03_structures).

Confirmatory M2PL on Bridge's native 5-skill axis (diagnosis / strategy / math /
communication / affective) vs collapses down to 1D, using calibrate_mirt's EM engine.
Reports loglik / #params / AIC / BIC, the latent correlation matrix, and held-out
k-fold marginal log-loss per structure, then applies the 1-SE selection rule.

EXPLORATORY at N=51: a 5-dim confirmatory M2PL is not identifiable at ~50 persons, so
the expected (and documented) result is that higher-D collapses toward 1D -- latent
correlations near +/-1 and, per the Bridge pilot, the AFFECTIVE axis fails to identify
(its anchors have disc <= 0). This script is written to surface exactly that.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import log_expit, logsumexp

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


def load_qmap(rubrics_path):
    """criterion_id -> {skill: 0/1} and the ordered skill axis (first-seen order)."""
    qmap, order = {}, []
    for rec in cp.read_jsonl(rubrics_path):
        cid = rec.get("criterion_id"); qm = rec.get("q_mapping")
        if cid is None or not isinstance(qm, dict):
            continue
        qmap[str(cid)] = {k: int(v) for k, v in qm.items()}
        for k in qm:
            if k not in order:
                order.append(k)
    return qmap, order


def build_Q(items, qmap, skills, groups):
    """groups: list of lists of skill names; each group -> one latent dim (logical OR)."""
    cols = []
    for c in items:
        row = qmap.get(c, {})
        cols.append([1 if any(int(row.get(s, 0)) == 1 for s in g) else 0 for g in groups])
    return np.array(cols, dtype=int)


def marginal_logloss(Yte, Mte, A, b, grid, log_prior):
    """Mean per-cell held-out marginal log-loss under train params (lower = better)."""
    eta = A @ grid.T - b[:, None]                      # (items, nodes)
    logP, log1mP = log_expit(eta), log_expit(-eta)
    YM = np.where(Mte, Yte, 0.0); NM = np.where(Mte, 1.0 - Yte, 0.0)
    LL = YM @ logP + NM @ log1mP                       # (persons, nodes)
    person_ll = logsumexp(LL + log_prior[None, :], axis=1)   # marginal per person
    ncells = Mte.sum()
    return float(-person_ll.sum() / max(ncells, 1))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matrix", type=Path, required=True)
    p.add_argument("--rubrics", type=Path, required=True)
    p.add_argument("--scenarios", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--grid", type=int, default=5, help="quadrature nodes/dim (5 -> 5^D nodes).")
    p.add_argument("--ridge", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=20260729)
    args = p.parse_args()
    (args.out_dir / "figures").mkdir(parents=True, exist_ok=True)

    mat = cp.load_matrix(args.matrix)
    c2s = _load_map(args.rubrics, "criterion_id", "scenario_id")
    s2src = _load_map(args.scenarios, "scenario_id", "source_id")
    mat = mat[dedupe_by_source(list(mat.columns), c2s, s2src)]
    sub, _ = cp.select_sparse(mat)
    kept, _af, _ap = cp.split_zero_variance(sub)
    sub = sub[kept]; sub = sub[sub.notna().any(axis=1)]
    items = list(sub.columns)
    Y = np.nan_to_num(sub.to_numpy(float), nan=0.0); M = sub.notna().to_numpy()
    n = len(sub.index)
    qmap, skills = load_qmap(args.rubrics)
    print(f"dimensionality: {len(items)} items x {n} models; skills={skills}")

    structures = {
        "overall_1d": [skills],
        "correlation_2d": [["diagnosis", "strategy", "math"], ["communication", "affective"]],
        "correlation_3d": [["diagnosis", "strategy"], ["math"], ["communication", "affective"]],
        "merge_affective_communication_4d": [["diagnosis"], ["strategy"], ["math"], ["communication", "affective"]],
        "full_5d": [[s] for s in skills],
    }
    # keep only groups whose skills exist in the axis
    structures = {name: [[s for s in g if s in skills] for g in groups if any(s in skills for s in g)]
                  for name, groups in structures.items()}

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(n)
    folds = [order[i::args.k] for i in range(args.k)]
    n_obs = int(M.sum())

    rows = []
    for name, groups in structures.items():
        Q = build_Q(items, qmap, skills, groups)
        keepj = np.where(Q.sum(axis=1) > 0)[0]     # items loading on >=1 modeled dim
        Qk = Q[keepj]; Yk = Y[:, keepj]; Mk = M[:, keepj]
        ndim = Qk.shape[1]
        try:
            fit = cm.fit_m2pl_em(Yk, Mk, Qk, args.grid, estimate_corr=(ndim > 1),
                                 ridge=args.ridge, max_iter=200, tol=1e-4)
        except Exception as e:
            print(f"  {name}: fit FAILED ({e})"); continue
        aic, bic = cm.aic_bic(fit["loglik"], fit["n_params"], n_obs)
        # k-fold held-out marginal log-loss
        grid = cm.build_grid(ndim, args.grid); base = cm.base_log_weights(ndim, args.grid)
        lp = base - logsumexp(base)
        lls = []
        for te in folds:
            tr = np.array([i for i in range(n) if i not in set(te.tolist())])
            ft = cm.fit_m2pl_em(Yk[tr], Mk[tr], Qk, args.grid, estimate_corr=False,
                                ridge=args.ridge, max_iter=150, tol=1e-4)
            lls.append(marginal_logloss(Yk[te], Mk[te], ft["A"], ft["b"], grid, lp))
        ll_mean = float(np.mean(lls)); ll_se = float(np.std(lls, ddof=1) / np.sqrt(len(lls)))
        R = fit.get("R")
        max_offdiag = (float(np.max(np.abs(R - np.eye(ndim)))) if (R is not None and ndim > 1) else 0.0)
        rows.append({"structure": name, "n_dims": ndim, "n_items": len(keepj),
                     "loglik": round(fit["loglik"], 1), "n_params": fit["n_params"],
                     "aic": round(aic, 1), "bic": round(bic, 1),
                     "oos_logloss_mean": round(ll_mean, 4), "oos_logloss_se": round(ll_se, 4),
                     "max_latent_corr_offdiag": round(max_offdiag, 3), "converged": fit["converged"]})
        print(f"  {name} ({ndim}d): AIC={aic:.0f} BIC={bic:.0f} OOS_ll={ll_mean:.4f}"
              f"±{ll_se:.4f} max|corr|={max_offdiag:.2f}", flush=True)

    # 1-SE selection on OOS log-loss, then fewest dims
    best = min(rows, key=lambda r: r["oos_logloss_mean"])
    thresh = best["oos_logloss_mean"] + best["oos_logloss_se"]
    within = [r for r in rows if r["oos_logloss_mean"] <= thresh]
    selected = min(within, key=lambda r: r["n_dims"])

    import csv
    with (args.out_dir / "structure_comparison.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    (args.out_dir / "selection.json").write_text(json.dumps({
        "rule": "lowest held-out marginal log-loss, keep within 1 SE, then fewest dimensions",
        "selected": selected["structure"], "selected_n_dims": selected["n_dims"],
        "best_ll_structure": best["structure"], "one_se_threshold": round(thresh, 4),
        "note": ("EXPLORATORY at N=51: higher-D confirmatory M2PL is not identifiable; "
                 "large max|latent corr| indicates collapse toward 1D. Affective axis is "
                 "expected to fail to identify (pilot disc <= 0)."),
        "structures": rows}, indent=2))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    xs = [r["structure"] for r in rows]; ys = [r["oos_logloss_mean"] for r in rows]
    es = [r["oos_logloss_se"] for r in rows]
    ax.errorbar(range(len(xs)), ys, yerr=es, fmt="o", capsize=4)
    ax.axhline(thresh, ls="--", color="gray", label="1-SE threshold")
    ax.set_xticks(range(len(xs))); ax.set_xticklabels(xs, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("held-out marginal log-loss (lower=better)")
    ax.set_title(f"Bridge dimensionality (N={n}); selected: {selected['structure']}")
    ax.legend(); fig.tight_layout(); fig.savefig(args.out_dir / "figures" / "structure_cv.png", dpi=140)
    print(f"\nselected structure: {selected['structure']} ({selected['n_dims']}d). wrote -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
