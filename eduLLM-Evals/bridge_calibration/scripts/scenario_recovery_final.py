"""Phase 2b exp 05 (recovery-final) + exp 10 (estimator comparison).

Definitive OOS model-fold recovery at the LOCKED operating point (min_scenarios=12,
SE target 0.15). k=5 person folds (seed 20260729): refit 1D (a,b) on TRAIN models (drop
within-train zero-variance among CAT-pool criteria), run the REAL scenario engine at the
locked stop rule on TEST models, and score each estimator's ability against the fold's
FULL-BANK reference. The reference theta uses a FINE uniform EAP grid (continuous; avoids
the coarse-GH quantization bug fixed on the item-level side).

CAT administration + theta scoring use the masked CAT-pool bank (A3 + extreme_a excluded).

Emits:
  experiments/05_oos_recovery/{recovery_metrics.json, oos_per_model.csv, figures/*}
  experiments/10_estimator_comparison/estimator_comparison.json  (EAP vs MWLE vs MLE)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import bridge_scenario_lib as L  # noqa: E402
import bridge_eap_stop_lib as E  # noqa: E402

ROOT = L.ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import scripts.scenario_cat_lib as scat  # noqa: E402

cm = L.cm


def make_folds(models, k, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(models))
    folds = [[] for _ in range(k)]
    for pos, i in enumerate(idx):
        folds[pos % k].append(models[int(i)])
    return [sorted(f) for f in folds]


def mle_subset_1d(y, idx, a, b, theta0, bound=8.0, iters=50, tol=1e-8):
    """Plain 1D maximum-likelihood ability over the administered items (no penalty).

    Newton with step-halving; clamps to +/-bound (all-pass/all-fail admin sets diverge)."""
    ai, bi, yi = a[idx], b[idx], y[idx].astype(float)
    th = float(theta0)
    for _ in range(iters):
        eta = ai * th - bi
        p = expit(eta)
        g = float(np.sum(ai * (yi - p)))
        h = float(-np.sum(ai * ai * p * (1.0 - p)))
        if abs(h) < 1e-12:
            break
        step = g / h
        new = th - step
        new = max(-bound, min(bound, new))
        if abs(new - th) < tol:
            th = new
            break
        th = new
    return np.array([th])


def recovery_stats(x, y):
    band = scat.ols_ci_band(x, y, np.linspace(x.min(), x.max(), 40), B=2000, seed=0)
    w = np.argsort(x)[:12]
    return {"r": round(band["r"], 4), "r_lo": round(band["r_lo"], 4),
            "r_hi": round(band["r_hi"], 4), "slope": round(band["slope"], 4),
            "slope_lo": round(band["slope_lo"], 4), "slope_hi": round(band["slope_hi"], 4),
            "mae": round(float(np.mean(np.abs(y - x))), 4),
            "gap_worst12": round(float(np.mean(y[w] - x[w])), 4)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    base = ROOT / "bridge_calibration"
    p.add_argument("--bank", type=Path, default=base / "bridge_scenario_fitted_1d_catpool.jsonl")
    p.add_argument("--matrix", type=Path, default=ROOT / "bridgegrade" / "response_matrix.csv")
    p.add_argument("--scenarios", type=Path, default=ROOT / "data" / "Bridge" / "scenarios.jsonl")
    p.add_argument("--out05", type=Path, default=base / "experiments" / "05_oos_recovery")
    p.add_argument("--out10", type=Path, default=base / "experiments" / "10_estimator_comparison")
    p.add_argument("--tmp-dir", type=Path, default=base / "experiments" / "05_oos_recovery" / "_tmp")
    p.add_argument("--stop-rule", choices=["eap", "online"], default="eap",
                   help="of-record = eap (EAP posterior SD <= target); online kept for provenance.")
    p.add_argument("--l-max", type=int, default=60, help="forced admin cap for the EAP stop.")
    p.add_argument("--min-scenarios", type=int, default=12)
    p.add_argument("--max-se", type=float, default=0.12)
    p.add_argument("--max-scenarios", type=int, default=228)
    p.add_argument("--min-evals-per-skill", type=int, default=15)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260729)
    p.add_argument("--fit-grid", type=int, default=7)
    p.add_argument("--eap-grid", type=int, default=321, help="FINE uniform EAP nodes (continuous ref).")
    p.add_argument("--range", type=float, default=8.0)
    p.add_argument("--ridge", type=float, default=1e-2)
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()
    (args.out05 / "figures").mkdir(parents=True, exist_ok=True)
    args.out10.mkdir(parents=True, exist_ok=True)
    args.tmp_dir.mkdir(parents=True, exist_ok=True)

    records, dims, _ = scat.load_fitted_bank(args.bank, "clamp")
    assert len(dims) == 1
    d = dims[0]
    ids = [r["criterion_id"] for r in records]
    scen_of = {r["criterion_id"]: r["scenario_id"] for r in records}
    matrix = pd.read_csv(args.matrix, index_col=0)
    Yraw = matrix.reindex(columns=ids).to_numpy(float)
    Mall = ~np.isnan(Yraw)
    models = list(matrix.index)
    row_of = {m: i for i, m in enumerate(models)}
    Q_all = np.ones((len(ids), 1), dtype=int)
    egrid, elog = scat.build_grid(1, args.eap_grid, args.range)
    gg, lp = E.eap_grid(args.eap_grid, args.range)

    print("=" * 88)
    print(f"EXP 05 + 10: OOS recovery at LOCKED op-point (min_scen={args.min_scenarios}, "
          f"SE={args.max_se}); stop_rule={args.stop_rule}; "
          f"FINE EAP grid={args.eap_grid} over +/-{args.range}")
    print("=" * 88)

    folds = make_folds(models, args.k, args.seed)
    rows = []
    fold_len_scen, fold_len_crit = [], []
    for f in range(args.k):
        test = folds[f]
        train = [m for m in models if m not in set(test)]
        tr = [row_of[m] for m in train]
        Ytr, Mtr = Yraw[tr], Mall[tr]
        keep = [j for j in range(len(ids)) if Mtr[:, j].sum() >= 2
                and 0 < Ytr[Mtr[:, j], j].sum() < Mtr[:, j].sum()]
        keep = np.array(keep, dtype=int)
        fit = cm.fit_m2pl_em(np.nan_to_num(Ytr[:, keep], nan=0.0), Mtr[:, keep],
                             Q_all[keep], args.fit_grid, ridge=args.ridge, max_iter=200)
        Ak, bk = fit["A"], fit["b"]
        kept_ids = [ids[j] for j in keep]
        colk = {c: i for i, c in enumerate(kept_ids)}
        fold_bank = args.tmp_dir / f"fold{f}_bank.jsonl"
        with fold_bank.open("w", encoding="utf-8") as fh:
            for jj, cid in enumerate(kept_ids):
                fh.write(json.dumps({"criterion_id": cid, "scenario_id": scen_of[cid],
                                     "criterion": "", "discrimination": {d: float(Ak[jj, 0])},
                                     "q_modeled": {d: 1}, "difficulty": float(bk[jj])}) + "\n")
        subte = matrix.loc[test].reindex(columns=kept_ids)
        Yte = np.nan_to_num(subte.to_numpy(float), nan=0.0)
        Mte = ~np.isnan(subte.to_numpy(float))
        theta_ref = scat.eap_all_models(Yte, Mte, Ak, bk, egrid, elog)  # fine EAP (n,1)
        if args.stop_rule == "eap":
            row_of_te = {m: i for i, m in enumerate(test)}
            admin = E.administer_eap(test, fold_bank, args.matrix, args.scenarios, dims,
                                     Ak[:, 0], bk, colk, scen_of, Yte, row_of_te, gg, lp,
                                     seed=args.seed, floor=args.min_scenarios,
                                     target=args.max_se, l_max=args.l_max, workers=args.workers)
            res_by = {m: {"order": admin[m]["order"], "theta_online": [admin[m]["eap_mean"]],
                          "scenarios_administered": admin[m]["scenarios_administered"],
                          "criteria_administered": admin[m]["criteria_administered"]}
                      for m in test}
        else:
            spec = scat.RunSpec(seed=args.seed, top_n=5, max_se=args.max_se,
                                min_evals_per_skill=args.min_evals_per_skill,
                                min_scenarios=args.min_scenarios, max_scenarios=args.max_scenarios,
                                selection="trace", mode="cat",
                                runs_dir=str(args.tmp_dir / f"runs_f{f}"))
            res = scat.run_models(test, fold_bank, args.matrix, args.scenarios, "clamp",
                                  dims, spec, workers=args.workers)
            res_by = {r["model"]: r for r in res}
        print(f"  fold {f}: train={len(train)} test={len(test)} kept={len(kept_ids)}", flush=True)
        for ti, m in enumerate(test):
            r0 = res_by[m]
            y = Yte[ti]
            idx = np.array([colk[c] for c in r0["order"] if c in colk], dtype=int)
            th_on = np.asarray(r0["theta_online"], float)
            if idx.size:
                th_ba = scat.eap_subset(y, idx, Ak, bk, egrid, elog)
                th_mw, _ = scat.mwle_subset(y, idx, Ak, bk, th_ba)
                th_ml = mle_subset_1d(y, idx, Ak[:, 0], bk, th_ba[0])
            else:
                th_ba = th_on.copy(); th_mw = th_on.copy(); th_ml = th_on.copy()
            pred_pass = float(expit(Ak[:, 0] * th_mw[0] - bk).mean())
            obs_pass = float(y[Mte[ti]].mean())
            fold_len_scen.append(r0["scenarios_administered"])
            fold_len_crit.append(r0["criteria_administered"])
            rows.append({"model": m, "fold": f,
                         "scenarios_administered": r0["scenarios_administered"],
                         "criteria_administered": r0["criteria_administered"],
                         f"theta_ref_{d}": float(theta_ref[ti][0]),
                         f"theta_online_{d}": float(th_on[0]),
                         f"theta_batch_{d}": float(th_ba[0]),
                         f"theta_mwle_{d}": float(th_mw[0]),
                         f"theta_mle_{d}": float(th_ml[0]),
                         "obs_pass_rate": obs_pass, "pred_pass_rate": pred_pass})

    df = pd.DataFrame(rows).sort_values("model")
    df.to_csv(args.out05 / "oos_per_model.csv", index=False)

    ref = df[f"theta_ref_{d}"].to_numpy()
    est = {"online": df[f"theta_online_{d}"].to_numpy(),
           "batch_eap": df[f"theta_batch_{d}"].to_numpy(),
           "mwle": df[f"theta_mwle_{d}"].to_numpy(),
           "mle": df[f"theta_mle_{d}"].to_numpy()}
    rec_by_est = {e: recovery_stats(ref, v) for e, v in est.items()}
    pass_band = scat.ols_ci_band(df["obs_pass_rate"].to_numpy(), df["pred_pass_rate"].to_numpy(),
                                 np.linspace(df["obs_pass_rate"].min(), df["obs_pass_rate"].max(), 40))
    pass_mae = float(np.mean(np.abs(df["pred_pass_rate"] - df["obs_pass_rate"])))

    headline = rec_by_est["mwle"]
    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "definitive OOS model-fold recovery at the locked scenario operating point",
        "operating_point": {"min_scenarios": args.min_scenarios, "max_se": args.max_se,
                            "selection": "trace", "estimator_headline": "mwle",
                            "stop_rule": args.stop_rule},
        "cv": f"OOS k={args.k} MODEL folds, seed {args.seed}", "n_models": len(df),
        "eap_reference_grid": {"nodes": args.eap_grid, "range": args.range,
                               "type": "fine uniform (continuous)"},
        "mean_scenarios_administered": round(float(np.mean(fold_len_scen)), 2),
        "mean_criteria_administered": round(float(np.mean(fold_len_crit)), 1),
        "headline_recovery_mwle": headline,
        "pass_rate_recovery": {"r": round(pass_band["r"], 4),
                               "r_lo": round(pass_band["r_lo"], 4),
                               "r_hi": round(pass_band["r_hi"], 4),
                               "slope": round(pass_band["slope"], 4),
                               "pass_mae": round(pass_mae, 4)},
        "recovery_by_estimator": rec_by_est,
    }
    (args.out05 / "recovery_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # estimator comparison (exp 10): slope closest to 1, but prefer MWLE over MLE when they
    # tie within noise (MLE can diverge on degenerate/short administrations; MWLE is robust).
    slopes = {e: rec_by_est[e]["slope"] for e in est}
    closest = min(slopes, key=lambda e: abs(slopes[e] - 1.0))
    if closest == "mle" and abs(slopes["mle"] - slopes["mwle"]) <= 0.01:
        recommended = "mwle"
        note = (f"MLE has the nominally-closest slope ({slopes['mle']:.3f}) but MWLE "
                f"({slopes['mwle']:.3f}) is within {abs(slopes['mle']-slopes['mwle']):.3f} "
                "(noise); MWLE is chosen as the deployment estimator because plain MLE "
                "diverges on all-pass/all-fail administrations (Warm's penalty keeps MWLE "
                "finite), matching the item-level study's headline.")
    else:
        recommended = closest
        note = (f"Selected '{recommended}' (slope={slopes[recommended]:.3f}) as the slope "
                "closest to 1 (least ability-scale shrinkage).")
    est_cmp = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "estimator comparison on the identical locked-point administered sets",
        "operating_point": {"min_scenarios": args.min_scenarios, "max_se": args.max_se},
        "n_models": len(df),
        "estimators": {e: {"r": rec_by_est[e]["r"], "slope": rec_by_est[e]["slope"],
                           "mae": rec_by_est[e]["mae"]} for e in est},
        "slope_closest_to_1": closest,
        "recommended_estimator": recommended,
        "rationale": "Pick the estimator whose recovery slope is closest to 1. " + note,
    }
    args.out10.mkdir(parents=True, exist_ok=True)
    (args.out10 / "estimator_comparison.json").write_text(json.dumps(est_cmp, indent=2),
                                                          encoding="utf-8")

    _figures(df, d, headline, pass_band, args.out05 / "figures", args.min_scenarios, args.max_se)

    for sub in tuple(f"runs_f{f}" for f in range(args.k)):
        shutil.rmtree(args.tmp_dir / sub, ignore_errors=True)

    print("\n" + "=" * 88)
    print(f"HEADLINE (MWLE): r={headline['r']} [{headline['r_lo']},{headline['r_hi']}] "
          f"slope={headline['slope']} MAE={headline['mae']}")
    print(f"pass-rate: r={metrics['pass_rate_recovery']['r']} MAE={metrics['pass_rate_recovery']['pass_mae']}")
    print("estimators (r/slope):", {e: (rec_by_est[e]['r'], rec_by_est[e]['slope']) for e in est})
    print(f"recommended estimator: {recommended}")
    print(f"mean length: {metrics['mean_scenarios_administered']} scen / "
          f"{metrics['mean_criteria_administered']} crit")
    print(f"wrote -> {args.out05}  and  {args.out10}")
    return 0


def _figures(df, d, headline, pass_band, fig_dir, floor, se):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    x = df[f"theta_ref_{d}"].to_numpy(); y = df[f"theta_mwle_{d}"].to_numpy()
    xs = np.linspace(x.min(), x.max(), 60)
    band = scat.ols_ci_band(x, y, xs, B=2000, seed=0)
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    lo, hi = min(x.min(), y.min()) - 0.3, max(x.max(), y.max()) + 0.3
    ax.plot([lo, hi], [lo, hi], ls="--", color="gray", lw=1, label="y = x")
    ax.fill_between(xs, band["band_lo"], band["band_hi"], color="#d95f0e", alpha=0.18)
    ax.plot(xs, band["slope"] * xs + band["intercept"], color="#d95f0e", lw=1.5,
            label=f"OLS slope={band['slope']:.3f}")
    ax.scatter(x, y, s=30, alpha=0.8, edgecolor="k", linewidth=0.3)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("full-bank EAP reference theta"); ax.set_ylabel("CAT MWLE theta")
    ax.set_title(f"Bridge scenario OOS recovery (floor={floor}, SE={se})\n"
                 f"r={headline['r']} [{headline['r_lo']},{headline['r_hi']}], "
                 f"slope={headline['slope']} (n={len(df)})", fontsize=9)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(fig_dir / "recovery_theta_mwle.png", dpi=140); plt.close(fig)

    ox = df["obs_pass_rate"].to_numpy(); oy = df["pred_pass_rate"].to_numpy()
    oxs = np.linspace(ox.min(), ox.max(), 60)
    pb = scat.ols_ci_band(ox, oy, oxs, B=2000, seed=0)
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    lo, hi = min(ox.min(), oy.min()) - 0.03, max(ox.max(), oy.max()) + 0.03
    ax.plot([lo, hi], [lo, hi], ls="--", color="gray", lw=1, label="y = x")
    ax.fill_between(oxs, pb["band_lo"], pb["band_hi"], color="#2c7fb8", alpha=0.18)
    ax.plot(oxs, pb["slope"] * oxs + pb["intercept"], color="#2c7fb8", lw=1.5,
            label=f"OLS slope={pb['slope']:.3f}")
    ax.scatter(ox, oy, s=30, alpha=0.8, edgecolor="k", linewidth=0.3)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("observed pass rate (full bank)"); ax.set_ylabel("predicted pass rate (CAT MWLE theta)")
    ax.set_title(f"p-IRT pass calibration\nr={pb['r']:.3f}, MAE={np.mean(np.abs(oy-ox)):.3f}", fontsize=9)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(fig_dir / "pirt_pass_calibration.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
