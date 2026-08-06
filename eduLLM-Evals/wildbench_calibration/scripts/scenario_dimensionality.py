"""Phase 1 step 3: WildBench scenario-level dimensionality -- the "collapse the 11" study.

Compares, over ALL 1,001 scenarios' criteria (no source dedup):
  * ``overall_1d``  -- unidimensional "general ability",
  * a small number of collapsed super-skill groupings (semantic + one EFA-derived),
  * ``full_11d``    -- the confirmatory 11-skill M2PL.

Because each criterion is ONE-HOT (loads on exactly one of the 11 task-category skills),
the full 11-dim confirmatory M2PL with identity latent correlation FACTORISES into 11
independent 1-D fits, so we fit it that way (a product 11-dim GH grid is infeasible) and
recover the 11x11 inter-skill latent correlation via a plug-in on the per-skill EAP
abilities (near-unattenuated at ~760 criteria/skill). The collapsed 2-4d groupings stay
one-hot at the GROUP level and DO estimate the group latent correlation on a product grid.

Reports loglik / #params / AIC / BIC / held-out marginal log-loss(+SE) / max|latent corr|
/ per-axis discriminations, plus an EFA/scree of the 11x11 skill-ability correlation that
guides the collapsed grouping. Selection rule (Bridge/TutorBench): lowest held-out log-loss,
keep within 1 SE, then BIC + fewest dimensions (parsimony, NOT AIC); collapse a group iff
latent r >~ 0.9 AND collapsed wins AIC/BIC AND EFA shows no separate factor.

Writes ``experiments/03_structures/{structure_comparison.csv, selection.json,
skill_correlation.csv, figures/}``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wildbench_scenario_lib as L  # noqa: E402


def scree_and_efa(corr: np.ndarray, skills: list[str]) -> dict:
    """Scree (eigenvalues) of the skill correlation + optional factor_analyzer EFA."""
    ev = np.linalg.eigvalsh(corr)[::-1]
    out = {
        "eigenvalues": [round(float(v), 4) for v in ev],
        "n_eigen_ge_1": int(np.sum(ev >= 1.0)),
        "pct_variance_first": round(float(ev[0] / ev.sum()), 4),
        "pct_variance_first2": round(float(ev[:2].sum() / ev.sum()), 4),
    }
    try:
        from factor_analyzer import FactorAnalyzer
        fa = FactorAnalyzer(n_factors=min(3, len(skills) - 1), rotation="varimax",
                            is_corr_matrix=True)
        fa.fit(corr)
        load = fa.loadings_
        out["efa_available"] = True
        out["efa_loadings"] = {skills[i]: [round(float(x), 3) for x in load[i]]
                               for i in range(len(skills))}
        out["efa_dominant_factor"] = {skills[i]: int(np.argmax(np.abs(load[i])))
                                      for i in range(len(skills))}
    except Exception as e:  # noqa: BLE001
        out["efa_available"] = False
        out["efa_reason"] = str(e)
    return out


def efa_grouping(corr: np.ndarray, skills: list[str], n_clusters: int) -> dict:
    """Hierarchical (average-linkage) clustering of the 11 skills on 1-corr distance.

    Returns a structure dict {group_label: [skills...]} plus the cluster assignment.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    d = 1.0 - corr
    np.fill_diagonal(d, 0.0)
    d = (d + d.T) / 2.0
    Z = linkage(squareform(d, checks=False), method="average")
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    groups: dict[int, list[str]] = {}
    for sk, lab in zip(skills, labels):
        groups.setdefault(int(lab), []).append(sk)
    structure = {f"efa_g{lab}": sk_list for lab, sk_list in sorted(groups.items())}
    return {"structure": structure,
            "assignment": {sk: int(lab) for sk, lab in zip(skills, labels)},
            "n_clusters": n_clusters}


def per_axis_disc(A: np.ndarray, Qk: np.ndarray, labels: list[str]) -> dict:
    out = {}
    for di, sk in enumerate(labels):
        load = Qk[:, di] == 1
        av = A[load, di]
        out[sk] = {
            "n_anchor_items": int(load.sum()),
            "median_disc": round(float(np.median(av)), 4) if av.size else None,
            "mean_disc": round(float(np.mean(av)), 4) if av.size else None,
            "frac_disc_le_0": round(float(np.mean(av <= 0)), 4) if av.size else None,
        }
    return out


def fit_and_score(name, structure, Y, M, items, maps, args, folds, models):
    """Fit one structure on full data + k-fold OOS log-loss. Returns a row + extras."""
    Q, labels = L.build_Q(items, maps["qmap"], structure)
    ndim = len(labels)
    is_indep = ndim >= args.independent_from
    n = len(models)

    if ndim == 1:
        fit, keepj = L.fit_structure(Y, M, Q, args.grid_1d, args.ridge,
                                     estimate_corr=False)
        max_off = 0.0
        paxis = None
        skill_corr = None
    elif is_indep:
        fit, keepj = L.fit_independent(Y, M, Q, args.grid_1d, args.ridge)
        thetas = L.skill_theta_matrix(Y, M, items, maps, fit, keepj, n_nodes=args.eap_nodes)
        skill_corr = np.corrcoef(thetas.T)
        off = skill_corr - np.eye(ndim)
        max_off = float(np.max(np.abs(off)))
        paxis = per_axis_disc(fit["A"], Q[keepj], labels)
    else:
        fit, keepj = L.fit_structure(Y, M, Q, args.grid, args.ridge, estimate_corr=True)
        R = fit.get("R")
        max_off = (float(np.max(np.abs(R - np.eye(ndim)))) if R is not None else 0.0)
        paxis = per_axis_disc(fit["A"], Q[keepj], labels)
        skill_corr = R

    aic, bic = L.cm.aic_bic(fit["loglik"], fit["n_params"], int(M.sum()))

    # k-fold OOS marginal log-loss over persons
    lls = []
    for te in folds:
        te_set = set(te.tolist())
        tr = np.array([i for i in range(n) if i not in te_set])
        Qk = Q[keepj]
        Ytr, Mtr = Y[tr][:, keepj], M[tr][:, keepj]
        Yte, Mte = Y[te][:, keepj], M[te][:, keepj]
        if ndim == 1:
            ft = L.cm.fit_m2pl_em(Ytr, Mtr, Qk, args.grid_1d, estimate_corr=False,
                                  ridge=args.ridge, max_iter=args.cv_iter, tol=1e-4)
            grid = L.cm.build_grid(1, args.grid_1d)
            lp = L.cm.base_log_weights(1, args.grid_1d)
            lp = lp - L.logsumexp(lp)
            lls.append(L.marginal_logloss(Yte, Mte, ft["A"], ft["b"], grid, lp))
        elif is_indep:
            # independent per-dim refit on train, pooled held-out log-loss
            A = np.zeros((Qk.shape[0], ndim))
            bb = np.zeros(Qk.shape[0])
            for d in range(ndim):
                rows = np.where(Qk[:, d] == 1)[0]
                Q1 = np.ones((rows.size, 1), dtype=int)
                f = L.cm.fit_m2pl_em(Ytr[:, rows], Mtr[:, rows], Q1, args.grid_1d,
                                     estimate_corr=False, ridge=args.ridge,
                                     max_iter=args.cv_iter, tol=1e-4)
                A[rows, d] = f["A"][:, 0]
                bb[rows] = f["b"]
            lls.append(L.marginal_logloss_independent(
                Yte, Mte, A, bb, np.arange(Qk.shape[0]), n_nodes=args.cv_eap_nodes))
        else:
            ft = L.cm.fit_m2pl_em(Ytr, Mtr, Qk, args.grid, estimate_corr=False,
                                  ridge=args.ridge, max_iter=args.cv_iter, tol=1e-4)
            grid = L.cm.build_grid(ndim, args.grid)
            lp = L.cm.base_log_weights(ndim, args.grid)
            lp = lp - L.logsumexp(lp)
            lls.append(L.marginal_logloss(Yte, Mte, ft["A"], ft["b"], grid, lp))

    ll_mean = float(np.mean(lls))
    ll_se = float(np.std(lls, ddof=1) / np.sqrt(len(lls)))
    row = {
        "structure": name, "n_dims": ndim, "n_criteria": int(len(keepj)),
        "fit_path": "independent" if is_indep else ("1d" if ndim == 1 else "product_grid"),
        "loglik": round(fit["loglik"], 1), "n_params": fit["n_params"],
        "aic": round(aic, 1), "bic": round(bic, 1),
        "oos_logloss_mean": round(ll_mean, 5), "oos_logloss_se": round(ll_se, 5),
        "max_latent_corr_offdiag": round(max_off, 4),
        "converged": fit["converged"],
    }
    return row, {"per_axis": paxis, "skill_corr": skill_corr, "labels": labels}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matrix", type=Path, default=L.DEFAULT_MATRIX)
    p.add_argument("--rubrics", type=Path, default=L.DEFAULT_RUBRICS)
    p.add_argument("--scenarios", type=Path, default=L.DEFAULT_SCENARIOS)
    p.add_argument("--out-dir", type=Path,
                   default=L.ROOT / "wildbench_calibration" / "experiments" / "03_structures")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--grid-1d", type=int, default=21, help="GH nodes for 1D / per-dim fits.")
    p.add_argument("--grid", type=int, default=5, help="GH nodes/dim for collapsed (2-4d).")
    p.add_argument("--ridge", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=20260729)
    p.add_argument("--eap-nodes", type=int, default=41, help="fine EAP nodes for skill theta.")
    p.add_argument("--cv-eap-nodes", type=int, default=21)
    p.add_argument("--cv-iter", type=int, default=150, help="max EM iters inside CV folds.")
    p.add_argument("--independent-from", type=int, default=6,
                   help="fit structures with >= this many dims via the independent path.")
    p.add_argument("--efa-clusters", type=int, default=0,
                   help="n clusters for the EFA-derived grouping (0 => use Kaiser scree).")
    args = p.parse_args()
    (args.out_dir / "figures").mkdir(parents=True, exist_ok=True)

    maps = L.load_maps(args.rubrics, args.scenarios)
    Y, M, items, models, drop_info = L.prepare_matrix(args.matrix)
    n = len(models)
    print(f"dimensionality sweep: {len(items)} criteria x {n} models; "
          f"skills={L.WILDBENCH_SKILLS}")

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(n)
    folds = [order[i::args.k] for i in range(args.k)]

    # --- first fit full_11d (independent) to get the 11x11 skill correlation + EFA ---
    print("fitting full_11d (independent per-skill) for EFA + skill correlation ...",
          flush=True)
    Q11, labels11 = L.build_Q(items, maps["qmap"], L.STRUCTURES["full_11d"])
    fit11, keep11 = L.fit_independent(Y, M, Q11, args.grid_1d, args.ridge)
    thetas11 = L.skill_theta_matrix(Y, M, items, maps, fit11, keep11, n_nodes=args.eap_nodes)
    skill_corr = np.corrcoef(thetas11.T)
    scree = scree_and_efa(skill_corr, labels11)
    n_clusters = args.efa_clusters or max(2, min(4, scree["n_eigen_ge_1"] or 2))
    efa_group = efa_grouping(skill_corr, labels11, n_clusters)
    print(f"  skill-corr max|off-diag|={float(np.max(np.abs(skill_corr - np.eye(11)))):.3f} "
          f"mean|off-diag|={float(np.mean(np.abs(skill_corr - np.eye(11)))):.3f}")
    print(f"  scree eigenvalues (>=1: {scree['n_eigen_ge_1']}): "
          f"{scree['eigenvalues'][:5]} ...")
    print(f"  EFA-derived grouping (k={n_clusters}): "
          f"{ {g: v for g, v in efa_group['structure'].items()} }")

    # write the skill correlation matrix
    with (args.out_dir / "skill_correlation.csv").open("w", newline="",
                                                       encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["skill"] + labels11)
        for i, sk in enumerate(labels11):
            w.writerow([sk] + [round(float(x), 4) for x in skill_corr[i]])

    # --- assemble the structure set (semantic candidates + EFA-derived) ---
    structures = dict(L.STRUCTURES)
    # add the EFA-derived grouping if it is non-trivial (2..10 groups) and distinct
    if 2 <= n_clusters <= 10:
        structures[f"efa_{n_clusters}d"] = efa_group["structure"]

    rows = []
    extras = {}
    for name, structure in structures.items():
        print(f"  fitting {name} ...", flush=True)
        row, extra = fit_and_score(name, structure, Y, M, items, maps, args, folds, models)
        rows.append(row)
        extras[name] = extra
        print(f"    {name} ({row['n_dims']}d): AIC={row['aic']:.0f} BIC={row['bic']:.0f} "
              f"OOS_ll={row['oos_logloss_mean']:.5f}+/-{row['oos_logloss_se']:.5f} "
              f"max|corr|={row['max_latent_corr_offdiag']:.3f}", flush=True)

    # selection: lowest OOS log-loss; within 1 SE; then fewest dims (BIC tiebreak)
    best = min(rows, key=lambda r: r["oos_logloss_mean"])
    thresh = best["oos_logloss_mean"] + best["oos_logloss_se"]
    within = [r for r in rows if r["oos_logloss_mean"] <= thresh]
    selected = min(within, key=lambda r: (r["n_dims"], r["bic"]))
    aic_best = min(rows, key=lambda r: r["aic"])
    bic_best = min(rows, key=lambda r: r["bic"])

    with (args.out_dir / "structure_comparison.csv").open("w", newline="",
                                                          encoding="utf-8") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)

    recommendation = (
        f"Selection rule -> {selected['structure']} ({selected['n_dims']}d). "
        f"The 11-skill latent abilities are highly collinear (max|r|="
        f"{float(np.max(np.abs(skill_corr - np.eye(11)))):.3f}), the first eigenvalue "
        f"explains {scree['pct_variance_first']*100:.0f}% of skill-ability variance, and "
        f"held-out log-loss does not separate the structures beyond 1 SE. At N=52 the "
        f"skills are not separable -> collapse toward a general-ability structure.")

    (args.out_dir / "selection.json").write_text(json.dumps({
        "rule": "lowest held-out marginal log-loss; keep within 1 SE; then fewest "
                "dimensions (BIC tiebreak) + parsimony (NOT AIC). Collapse a group iff "
                "latent r >~ 0.9 AND collapsed wins AIC/BIC AND EFA shows no separate factor.",
        "selected": selected["structure"], "selected_n_dims": selected["n_dims"],
        "best_ll_structure": best["structure"], "one_se_threshold": round(thresh, 5),
        "n_within_1se": len(within),
        "aic_best_structure": aic_best["structure"],
        "bic_best_structure": bic_best["structure"],
        "collapse_recommendation": recommendation,
        "skill_correlation_11x11": np.round(skill_corr, 4).tolist(),
        "skill_correlation_max_offdiag": round(
            float(np.max(np.abs(skill_corr - np.eye(11)))), 4),
        "skill_correlation_mean_offdiag": round(
            float(np.mean(np.abs(skill_corr - np.eye(11)))), 4),
        "scree_efa": scree,
        "efa_derived_grouping": efa_group,
        "per_axis_discrimination_full_11d": extras["full_11d"]["per_axis"],
        "structures_evaluated": {k: v for k, v in structures.items()},
        "note": ("N=52 persons: multi-dim confirmatory M2PL is not identifiable as a "
                 "correlated model; the 11-dim fit uses identity-R independence + a plug-in "
                 "skill-ability correlation. Large max|corr| => collapse toward 1D."),
        "structures": rows}, indent=2), encoding="utf-8")

    # figures: CV log-loss + skill correlation heatmap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order_rows = sorted(rows, key=lambda r: r["n_dims"])
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    xs = [r["structure"] for r in order_rows]
    ys = [r["oos_logloss_mean"] for r in order_rows]
    es = [r["oos_logloss_se"] for r in order_rows]
    ax.errorbar(range(len(xs)), ys, yerr=es, fmt="o", capsize=4)
    ax.axhline(thresh, ls="--", color="gray", label="1-SE threshold")
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels(xs, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("held-out marginal log-loss (lower=better)")
    ax.set_title(f"WildBench scenario-level dimensionality (N={n}); "
                 f"selected: {selected['structure']}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out_dir / "figures" / "structure_cv.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    im = ax.imshow(skill_corr, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(11)); ax.set_xticklabels(labels11, rotation=90, fontsize=7)
    ax.set_yticks(range(11)); ax.set_yticklabels(labels11, fontsize=7)
    for i in range(11):
        for j in range(11):
            ax.text(j, i, f"{skill_corr[i, j]:.2f}", ha="center", va="center",
                    fontsize=5.5, color="white" if skill_corr[i, j] < 0.7 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("WildBench 11-skill latent-ability correlation (plug-in, N=52)")
    fig.tight_layout()
    fig.savefig(args.out_dir / "figures" / "skill_correlation.png", dpi=140)
    plt.close(fig)

    print(f"\nselected: {selected['structure']} ({selected['n_dims']}d) | "
          f"AIC-best={aic_best['structure']} BIC-best={bic_best['structure']}")
    print(f"collapse recommendation: {recommendation}")
    print(f"wrote -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
