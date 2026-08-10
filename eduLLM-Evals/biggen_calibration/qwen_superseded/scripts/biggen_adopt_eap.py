"""Phase B: ADOPT the EAP-posterior stop as BiGGen's of-record CAT rule (locked floor 8 / SE 0.12).

Mirrors the WildBench adoption (``frq/wildbench`` ``wildbench_calibration/scripts/*`` +
``experiments/archive_onlineSE/``). The bank is UNCHANGED (stop-rule change only, no re-fit);
the EAP stop is a post-hoc rule on the production engine's forced-long adaptive order:

  stop at the first scenario with n_scenarios >= 8 AND EAP posterior SD <= 0.12 (posterior over
  the fine 3201-node theta grid from the administered responses), else the forced cap (L=40).

This driver (a) archives the current online-SE of-record under ``experiments/archive_onlineSE/``,
then (b) regenerates every STOP-DEPENDENT of-record artifact under the EAP stop, matching each
BiGGen experiment's existing schema, re-bootstrapping SE_param on the EAP-administered sets:
  * exp-05 recovery (+ estimator comparison: online[=EAP mean]/batch/MWLE/MLE) + p-IRT pass cols
  * exp-04 efficiency (adaptive arm -> EAP OOS; random baseline is stop-independent, kept)
  * exp-07 deployed SE (leaderboard_se_components, se_per_model, se_post_vs_total, se_vs_floor,
    precision_reached, se_ability_vs_total) -- full-bank SE_param FLOOR + item cov are
    stop-independent and preserved
  * exp-09 p-IRT pass-rate MAE (reuses exp-05 EAP OOS per-model)
  * exp-10 order/seed stability (8 seeds, EAP stop)
  * exp-08 leaderboard: FULL-BANK theta (stop-independent, mirrors WildBench) + EAP SE_total bars
    + weakly_identified flag = the EAP-native deployed caps (~6 lowest-ability tiny base models)
  * promote the Phase-A EAP grid (exp-12) into exp-06/06b_operating_point of-record
  * mark exp-11 eap_stop_prototype ADOPTED

KEPT (stop-independent, NOT re-run): dimensionality diagnostic (03_structures), ridge
sensitivity (03_ridge), the full-bank SE_param floor + item cov, the full-bank leaderboard theta.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
from scipy.special import expit, logsumexp

_HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "scripts"), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import scripts.scenario_cat_lib as scat  # noqa: E402
import biggen_eap_stop_lib as E  # noqa: E402
import biggen_recovery_grid as rg  # noqa: E402 (compute_item_cov / bootstrap_theta)

_cm_spec = importlib.util.spec_from_file_location("calibrate_mirt", ROOT / "scripts" / "calibrate_mirt.py")
cm = importlib.util.module_from_spec(_cm_spec)
sys.modules["calibrate_mirt"] = cm
_cm_spec.loader.exec_module(cm)

BASE = ROOT / "biggen_calibration"
EXP = BASE / "experiments"
DIM = "general"
FLOOR = 8
SE = 0.12
GRID_MODES = {"eap_grid": 3201, "range": 8.0}


def mle_subset_1d(y, idx, a, b, theta0, bound=8.0, iters=50, tol=1e-8):
    """Plain 1D ML ability over administered items (no penalty); Newton + step-halving."""
    ai, bi, yi = a[idx], b[idx], y[idx].astype(float)
    th = float(theta0)
    for _ in range(iters):
        eta = ai * th - bi
        p = expit(eta)
        g = float(np.sum(ai * (yi - p)))
        h = float(-np.sum(ai * ai * p * (1.0 - p)))
        if abs(h) < 1e-12:
            break
        new = max(-bound, min(bound, th - g / h))
        if abs(new - th) < tol:
            th = new
            break
        th = new
    return float(th)


def recovery_stats(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    w = np.argsort(x)[:12]
    return {"slope": float(np.polyfit(x, y, 1)[0]), "r": float(np.corrcoef(x, y)[0, 1]),
            "gap_worst12": float(np.mean(y[w] - x[w])), "n": int(x.size)}


def boot_mae_ci(err, B=2000, seed=0, ci=0.95):
    err = np.asarray(err, float)
    rng = np.random.default_rng(seed)
    n = err.size
    maes = np.array([np.mean(err[rng.integers(0, n, n)]) for _ in range(B)])
    a = (1 - ci) / 2
    return float(np.percentile(maes, 100 * a)), float(np.percentile(maes, 100 * (1 - a)))


def loo_mae(err):
    err = np.asarray(err, float)
    n = err.size
    tot = err.sum()
    return float(np.mean([(tot - err[i]) / (n - 1) for i in range(n)]))


# ---------------------------------------------------------------------------
# shared setup: load bank, full-data forced-long EAP order, k folds
# ---------------------------------------------------------------------------

def load_shared(args):
    records, dims, _ = scat.load_fitted_bank(args.bank, "clamp")
    assert len(dims) == 1 and dims[0] == DIM
    ids, A, b = scat.assemble_arrays(records, dims)
    a1 = A[:, 0]
    scen_of = {r["criterion_id"]: r["scenario_id"] for r in records}
    matrix = pd.read_csv(args.matrix, index_col=0)
    Yraw = matrix.reindex(columns=ids).to_numpy(float)
    Mall = ~np.isnan(Yraw)
    Y = np.nan_to_num(Yraw, nan=0.0)
    models = list(matrix.index)
    row_of = {m: i for i, m in enumerate(models)}
    col = {c: i for i, c in enumerate(ids)}
    egrid, elog = scat.build_grid(1, args.eap_grid, args.range)
    gg, lp = E.eap_grid(args.eap_grid, args.range)
    return dict(records=records, dims=dims, ids=ids, A=A, a1=a1, b=b, scen_of=scen_of,
                matrix=matrix, Yraw=Yraw, Mall=Mall, Y=Y, models=models, row_of=row_of,
                col=col, egrid=egrid, elog=elog, gg=gg, lp=lp)


def forced_long_full(S, args):
    """Forced-long full-data adaptive order (seed) -> per-model EAP walk."""
    spec = scat.RunSpec(seed=args.seed, top_n=5, max_se=0.0, min_evals_per_skill=0,
                        min_scenarios=args.l_max, max_scenarios=args.l_max, selection="trace",
                        mode="cat", runs_dir=str(args.tmp_dir / "fullL"))
    res = scat.run_models(S["models"], args.bank, args.matrix, args.scenarios, "clamp",
                          S["dims"], spec, workers=args.workers)
    walks = {}
    for r in res:
        m = r["model"]
        walks[m] = E.eap_walk(r["order"], S["Y"][S["row_of"][m]], S["col"], S["scen_of"],
                              S["a1"], S["b"], S["gg"], S["lp"])
    shutil.rmtree(args.tmp_dir / "fullL", ignore_errors=True)
    return walks


def fit_folds(S, args):
    """Fit k person-folds once; forced-long EAP order per fold; full-bank fold reference."""
    folds = E.make_folds(S["models"], args.k, args.seed)
    Q_all = np.ones((len(S["ids"]), 1), dtype=int)
    fold_data = []
    for f in range(args.k):
        test = folds[f]
        train = [m for m in S["models"] if m not in set(test)]
        tr = [S["row_of"][m] for m in train]
        Ytr, Mtr = S["Yraw"][tr], S["Mall"][tr]
        keep = [j for j in range(len(S["ids"])) if Mtr[:, j].sum() >= 2
                and 0 < Ytr[Mtr[:, j], j].sum() < Mtr[:, j].sum()]
        keep = np.array(keep, dtype=int)
        fit = cm.fit_m2pl_em(np.nan_to_num(Ytr[:, keep], nan=0.0), Mtr[:, keep],
                             Q_all[keep], args.fit_grid, ridge=args.ridge, max_iter=200)
        Ak, bk = fit["A"], fit["b"]
        kept_ids = [S["ids"][j] for j in keep]
        colk = {c: i for i, c in enumerate(kept_ids)}
        fold_bank = args.tmp_dir / f"fold{f}_bank.jsonl"
        with fold_bank.open("w", encoding="utf-8") as fh:
            for jj, cid in enumerate(kept_ids):
                fh.write(json.dumps({"criterion_id": cid, "scenario_id": S["scen_of"][cid],
                                     "criterion": "", "discrimination": {DIM: float(Ak[jj, 0])},
                                     "q_modeled": {DIM: 1}, "difficulty": float(bk[jj])}) + "\n")
        subte = S["matrix"].loc[test].reindex(columns=kept_ids)
        Yte = np.nan_to_num(subte.to_numpy(float), nan=0.0)
        Mte = ~np.isnan(subte.to_numpy(float))
        theta_ref = scat.eap_all_models(Yte, Mte, Ak, bk, S["egrid"], S["elog"])
        row_of_te = {m: i for i, m in enumerate(test)}
        admin = E.administer_eap(test, fold_bank, args.matrix, args.scenarios, S["dims"],
                                 Ak[:, 0], bk, colk, S["scen_of"], Yte, row_of_te, S["gg"], S["lp"],
                                 seed=args.seed, floor=FLOOR, target=SE, l_max=args.l_max,
                                 workers=args.workers, runs_dir=str(args.tmp_dir / f"foldL{f}"))
        fold_data.append(dict(test=test, Ak=Ak, bk=bk, colk=colk, Yte=Yte, Mte=Mte,
                              theta_ref=theta_ref, admin=admin))
        print(f"  fold {f}: train={len(train)} test={len(test)} kept={len(kept_ids)}", flush=True)
    return fold_data


# ---------------------------------------------------------------------------
# archiving
# ---------------------------------------------------------------------------

def _copy(src: Path, dst: Path):
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)
    return True


def archive_online(args):
    arc = EXP / "archive_onlineSE"
    if arc.exists() and not args.force_archive:
        print(f"archive already exists at {arc} (skipping; use --force-archive to overwrite)")
        return
    arc.mkdir(parents=True, exist_ok=True)
    # stop-dependent of-record dirs (exclude the *_coarse provenance already inside 05/08)
    _copy(EXP / "04_efficiency_vs_random", arc / "04_efficiency_vs_random")
    for name in ("metrics.json", "oos_per_model.csv", "results.csv"):
        _copy(EXP / "05_oos_recovery" / name, arc / "05_oos_recovery" / name)
    _copy(EXP / "05_oos_recovery" / "figures" / "oos_recovery_general.png",
          arc / "05_oos_recovery" / "figures" / "oos_recovery_general.png")
    _copy(EXP / "05_oos_recovery" / "figures" / "oos_recovery_scatter_general.png",
          arc / "05_oos_recovery" / "figures" / "oos_recovery_scatter_general.png")
    for name in ("results.csv", "results_fine_se.csv", "results_tight_se.csv"):
        _copy(EXP / "06_floor_se_grid" / name, arc / "06_floor_se_grid" / name)
    _copy(EXP / "06_floor_se_grid" / "06b_operating_point", arc / "06b_operating_point")
    for name in ("leaderboard_se_components.csv", "se_per_model.csv", "metrics.json",
                 "results.csv", "se_vs_floor.csv", "se_by_operating_point.csv"):
        _copy(EXP / "07_parameter_uncertainty" / name, arc / "07_deployed_se" / name)
    _copy(EXP / "07_parameter_uncertainty" / "figures", arc / "07_deployed_se" / "figures")
    _copy(EXP / "09_pirt_mae", arc / "09_pirt_mae")
    _copy(EXP / "10_order_seed", arc / "10_order_seed")
    # exp-08 was CAT-administered theta (deployed, stop-dependent) -> archived; the of-record
    # leaderboard becomes FULL-BANK theta (stop-independent), mirroring WildBench.
    _copy(EXP / "08_leaderboard" / "results.csv", arc / "08_leaderboard_catadmin" / "results.csv")
    _copy(EXP / "08_leaderboard" / "figures" / "leaderboard_general.png",
          arc / "08_leaderboard_catadmin" / "figures" / "leaderboard_general.png")
    _copy(EXP / "08_leaderboard" / "of_record_provenance.json",
          arc / "08_leaderboard_catadmin" / "of_record_provenance.json")
    (arc / "README.txt").write_text(
        "Archived online-SE-stop of-record outputs (provenance), superseded by the EAP-posterior "
        "stop adoption (floor 8 / SE 0.12). See ../../README.md.\n\n"
        "Archived (stop-dependent): 04_efficiency_vs_random, 05_oos_recovery (metrics + "
        "oos_per_model + recovery figures; the *_coarse fine-grid provenance stays in exp-05), "
        "06_floor_se_grid sweeps + 06b_operating_point (online grid), 07_deployed_se (deployed "
        "SE components / se_vs_floor / se_by_operating_point + figures), 09_pirt_mae, "
        "10_order_seed, and 08_leaderboard_catadmin (the prior CAT-administered-theta "
        "leaderboard -- the new of-record exp-08 uses FULL-BANK theta, mirroring WildBench).\n\n"
        "NOT archived (stop-independent, unchanged): 03_structures (dimensionality), 03_ridge "
        "(ridge sensitivity), 01_min_scenarios / 02_se_target (online-stop decision-evidence "
        "sweeps, left as historical), the exp-07 full-bank SE_param FLOOR + item covariance, and "
        "the full-bank leaderboard theta.\n",
        encoding="utf-8")
    print(f"archived online-SE of-record -> {arc}")


# ---------------------------------------------------------------------------
# exp-05 recovery + estimator comparison (+ p-IRT pass columns)
# ---------------------------------------------------------------------------

def step_recovery(S, fold_data, args):
    d = DIM
    rows = []
    len_scen, len_crit = [], []
    for fd in fold_data:
        Ak, bk, colk = fd["Ak"], fd["bk"], fd["colk"]
        for ti, m in enumerate(fd["test"]):
            y = fd["Yte"][ti]
            a = fd["admin"][m]
            idx = a["idx"]
            th_on = a["eap_mean"]  # EAP posterior mean at the stop = deployed online point est
            if idx.size:
                th_ba = float(scat.eap_subset(y, idx, Ak, bk, S["egrid"], S["elog"])[0])
                th_mw = float(scat.mwle_subset(y, idx, Ak, bk, np.array([th_ba]))[0][0])
                th_ml = mle_subset_1d(y, idx, Ak[:, 0], bk, th_ba)
            else:
                th_ba = th_mw = th_ml = float(fd["theta_ref"][ti][0])
            pred_pass = float(expit(Ak[:, 0] * th_mw - bk).mean())
            obs_pass = float(y[fd["Mte"][ti]].mean())
            len_scen.append(a["scenarios_administered"]); len_crit.append(a["criteria_administered"])
            rows.append({"model": m, "fold": fold_data.index(fd),
                         "criteria_administered": a["criteria_administered"],
                         "scenarios_administered": a["scenarios_administered"],
                         "precision_reached": bool(a["precision_reached"]),
                         f"theta_ref_{d}": float(fd["theta_ref"][ti][0]),
                         f"theta_online_{d}": float(th_on), f"theta_batch_{d}": th_ba,
                         f"theta_mwle_{d}": th_mw, f"theta_mle_{d}": th_ml,
                         "obs_pass_rate": obs_pass, "pred_pass_rate": pred_pass})
    df = pd.DataFrame(rows).sort_values("model").reset_index(drop=True)
    out = EXP / "05_oos_recovery"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "oos_per_model.csv", index=False)

    ref = df[f"theta_ref_{d}"].to_numpy()
    ests = {"online": df[f"theta_online_{d}"].to_numpy(), "batch": df[f"theta_batch_{d}"].to_numpy(),
            "mwle": df[f"theta_mwle_{d}"].to_numpy(), "mle": df[f"theta_mle_{d}"].to_numpy()}
    agg = {e: {d: recovery_stats(ref, v)} for e, v in ests.items()}
    mwle_band = scat.ols_ci_band(ref, ests["mwle"], np.linspace(ref.min(), ref.max(), 40),
                                 B=2000, seed=0)
    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "scenario-level OOS k-fold estimator recovery (real engine, EAP-posterior stop)",
        "bank": str(args.bank), "matrix": str(args.matrix), "dims": [d],
        "k": args.k, "seed": args.seed, "n_models": len(df),
        "config": {"fit_grid": args.fit_grid, "ridge": args.ridge, "eap_grid": args.eap_grid,
                   "max_se": SE, "min_scenarios": FLOOR, "min_evals_per_skill": 0,
                   "max_scenarios": args.l_max, "stop_rule": "eap"},
        "mean_criteria_administered": float(np.mean(len_crit)),
        "mean_scenarios_administered": float(np.mean(len_scen)),
        "oos_recovery": agg,
        "headline_recovery_mwle": {"r": round(mwle_band["r"], 4), "r_lo": round(mwle_band["r_lo"], 4),
                                   "r_hi": round(mwle_band["r_hi"], 4),
                                   "slope": round(mwle_band["slope"], 4),
                                   "theta_mae": round(float(np.mean(np.abs(ests["mwle"] - ref))), 4)},
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    _fig_3panel(ref, ests, agg, d, out / "figures" / "oos_recovery_general.png")
    _fig_scatter(ref, ests["mwle"], mwle_band, len(df), out / "figures" / "oos_recovery_scatter_general.png")

    prov = json.loads((out / "of_record_provenance.json").read_text(encoding="utf-8")) \
        if (out / "of_record_provenance.json").is_file() else {}
    prov.update({"of_record": True, "stop_rule": "eap", "eap_grid": args.eap_grid,
                 "eap_range": args.range, "eap_prior": "standard-normal (uniform product grid)",
                 "locked_config": {"max_se": SE, "min_scenarios": FLOOR, "ridge": args.ridge,
                                   "k": args.k, "seed": args.seed, "estimator": "mwle"},
                 "recovery_headline_mwle": metrics["headline_recovery_mwle"],
                 "reason": "EAP-posterior stop adopted as of-record (floor 8 / SE 0.12); replaces "
                           "the online normal-approx stop. Bank UNCHANGED (no re-fit). "
                           "Prior online-SE recovery archived under experiments/archive_onlineSE/.",
                 "generated_by": "biggen_calibration/scripts/biggen_adopt_eap.py"})
    (out / "of_record_provenance.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
    print(f"  exp-05: MWLE r={metrics['headline_recovery_mwle']['r']} "
          f"slope={metrics['headline_recovery_mwle']['slope']} "
          f"MAE={metrics['headline_recovery_mwle']['theta_mae']} "
          f"len {metrics['mean_scenarios_administered']:.2f} scen")
    return df, metrics


def _fig_3panel(ref, ests, agg, d, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = ("online", "batch", "mwle")
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), sharex=True, sharey=True)
    labels = {"online": "online (EAP mean)", "batch": "batch EAP", "mwle": "MWLE"}
    for ax, e in zip(axes, order):
        y = ests[e]; s = agg[e][d]
        ax.scatter(ref, y, s=20, alpha=0.6, edgecolor="k", linewidth=0.2)
        lo = min(ref.min(), y.min()) - 0.3; hi = max(ref.max(), y.max()) + 0.3
        ax.plot([lo, hi], [lo, hi], ls="--", color="gray", lw=1)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel(f"reference EAP ({d})"); ax.set_ylabel(f"CAT {labels[e]} ({d})")
        ax.set_title(f"{labels[e]}: r={s['r']:.3f} slope={s['slope']:.3f}", fontsize=9)
    fig.suptitle(f"BiGGen scenario OOS estimator recovery @ LOCKED 8/0.12 (EAP stop): {d}", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def _fig_scatter(x, y, band, n, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = np.linspace(x.min(), x.max(), 60)
    b2 = scat.ols_ci_band(x, y, xs, B=2000, seed=0)
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    lo, hi = min(x.min(), y.min()) - 0.3, max(x.max(), y.max()) + 0.3
    ax.plot([lo, hi], [lo, hi], ls="--", color="gray", lw=1, label="y = x")
    ax.fill_between(xs, b2["band_lo"], b2["band_hi"], color="#d95f0e", alpha=0.18)
    ax.plot(xs, b2["slope"] * xs + b2["intercept"], color="#d95f0e", lw=1.5,
            label=f"OLS slope={b2['slope']:.3f}")
    ax.scatter(x, y, s=30, alpha=0.8, edgecolor="k", linewidth=0.3)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("full-bank EAP reference theta (fine grid)"); ax.set_ylabel("CAT MWLE theta")
    ax.set_title(f"BiGGen scenario OOS recovery @ LOCKED 8/0.12 (EAP stop, of-record)\n"
                 f"r={band['r']:.4f}, slope={band['slope']:.3f}, "
                 f"theta-MAE={np.mean(np.abs(y - x)):.3f} (n={n})", fontsize=9)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


# ---------------------------------------------------------------------------
# exp-09 p-IRT (reuse exp-05 EAP oos)
# ---------------------------------------------------------------------------

def step_pirt(df05, args):
    d = DIM
    per = pd.DataFrame({"model": df05["model"], "predicted_passrate": df05["pred_pass_rate"],
                        "actual_passrate": df05["obs_pass_rate"], "theta_oos": df05[f"theta_mwle_{d}"]})
    per = per.sort_values("model").reset_index(drop=True)
    out = EXP / "09_pirt_mae"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    per.to_csv(out / "oos_per_model_pirt.csv", index=False)

    actual = per["actual_passrate"].to_numpy(); predicted = per["predicted_passrate"].to_numpy()
    theta_oos = per["theta_oos"].to_numpy()
    theta_ref = df05.sort_values("model")[f"theta_ref_{d}"].to_numpy()
    band = scat.ols_ci_band(actual, predicted, np.linspace(actual.min(), actual.max(), 40))
    pass_err = np.abs(predicted - actual)
    mae_raw = float(np.mean(pass_err)); mae_lo, mae_hi = boot_mae_ci(pass_err)
    mae_loo = loo_mae(pass_err)
    theta_err = np.abs(theta_oos - theta_ref)
    theta_mae = float(np.mean(theta_err)); th_lo, th_hi = boot_mae_ci(theta_err)
    results = pd.DataFrame([{
        "eval": "OOS", "passrate_mae_raw": mae_raw, "passrate_mae_raw_lo": mae_lo,
        "passrate_mae_raw_hi": mae_hi, "passrate_mae_loo": mae_loo, "theta_mae": theta_mae,
        "theta_mae_lo": th_lo, "theta_mae_hi": th_hi, "r": band["r"], "r_lo": band["r_lo"],
        "r_hi": band["r_hi"], "slope": band["slope"], "slope_lo": band["slope_lo"],
        "slope_hi": band["slope_hi"], "n": len(per)}])
    results.to_csv(out / "results.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = np.linspace(actual.min(), actual.max(), 60)
    b2 = scat.ols_ci_band(actual, predicted, xs, B=2000, seed=0)
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    lo, hi = min(actual.min(), predicted.min()) - 0.03, max(actual.max(), predicted.max()) + 0.03
    ax.plot([lo, hi], [lo, hi], ls="--", color="gray", lw=1, label="y = x")
    ax.fill_between(xs, b2["band_lo"], b2["band_hi"], color="#2c7fb8", alpha=0.18)
    ax.plot(xs, b2["slope"] * xs + b2["intercept"], color="#2c7fb8", lw=1.5,
            label=f"OLS slope={b2['slope']:.3f}")
    ax.scatter(actual, predicted, s=30, alpha=0.8, edgecolor="k", linewidth=0.3)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("actual held-out pass rate"); ax.set_ylabel("p-IRT predicted pass rate (MWLE theta)")
    ax.set_title(f"BiGGen scenario p-IRT pass calibration @ 8/0.12 (EAP stop)\n"
                 f"r={band['r']:.3f} [{band['r_lo']:.3f},{band['r_hi']:.3f}], "
                 f"pass-MAE={mae_raw:.3f} (n={len(per)})", fontsize=9)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(out / "figures" / "pirt_pred_vs_actual.png", dpi=140); plt.close(fig)
    print(f"  exp-09: pass r={band['r']:.4f} MAE={mae_raw:.4f} theta-MAE={theta_mae:.4f}")


# ---------------------------------------------------------------------------
# exp-07 deployed SE (re-bootstrap SE_param on EAP admin sets) + derived artifacts
# ---------------------------------------------------------------------------

def step_deployed_se(S, full_walks, args):
    d = DIM
    out = EXP / "07_parameter_uncertainty"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    cov = rg.compute_item_cov(args.bank, args.matrix, args.fit_grid, args.ridge)
    # full-bank SE_param FLOOR (median item-discrimination SE) -- STOP-INDEPENDENT, preserved.
    # chol is the Cholesky of the [a, -b] covariance, so chol[0,0] = SD(a).
    item_sd = [float(ch[0, 0]) for ch in cov["chol"] if getattr(ch, "shape", (0,))[0]]
    median_item_param_se = float(np.median([s for s in item_sd if s > 0]))
    bgrid, blp = S["gg"], S["lp"]  # fine grid -> se_post consistent with the EAP stop SD
    rng = np.random.default_rng(args.pu_seed)

    def deployed_rows(floor):
        recs = []
        for m in S["models"]:
            step, reached = E.eap_stop_point(full_walks[m], floor, SE)
            idx = step["idx"]
            _, se_post, se_par = rg.bootstrap_theta(S["Y"][S["row_of"][m]], idx, cov["beta"],
                                                    cov["chol"], bgrid, blp, args.n_boot, rng)
            se_tot = float(np.sqrt(se_post ** 2 + se_par ** 2))
            recs.append({"model": m, "n_admin": int(idx.size), "n_scen": step["n_scen"],
                         "theta": step["eap_mean"], "se_post": float(se_post),
                         "se_param": float(se_par), "se_total": se_tot,
                         "bar_inflation": se_tot / se_post if se_post > 0 else np.nan,
                         "precision_reached": bool(reached)})
        return pd.DataFrame(recs)

    dep = deployed_rows(FLOOR)
    # leaderboard_se_components.csv (se_posterior naming) + se_per_model.csv (se_ability naming)
    lc = pd.DataFrame({"model": dep["model"], "n_admin": dep["n_admin"], f"theta_{d}": dep["theta"],
                       f"se_posterior_{d}": dep["se_post"], f"se_param_{d}": dep["se_param"],
                       f"se_total_{d}": dep["se_total"], f"bar_inflation_{d}": dep["bar_inflation"]})
    lc.to_csv(out / "leaderboard_se_components.csv", index=False)
    sp = lc.rename(columns={f"se_posterior_{d}": f"se_ability_{d}"})
    sp.to_csv(out / "se_per_model.csv", index=False)
    # se_post_vs_total.csv (WildBench-parity deployed regime)
    pv = pd.DataFrame({"regime": "deployed_cat_8_0.12", "model": dep["model"], "theta": dep["theta"],
                       "SE_posterior": dep["se_post"], "SE_total": dep["se_total"],
                       "n_admin_criteria": dep["n_admin"]})
    pv.to_csv(out / "se_post_vs_total.csv", index=False)

    metrics = {"generated_at": datetime.now(timezone.utc).isoformat(),
               "purpose": "scenario-level parameter-uncertainty inflation of ability SEs (EAP stop)",
               "bank": str(args.bank), "matrix": str(args.matrix), "dims": [d],
               "config": {"fit_nodes": args.fit_grid, "eap_grid": args.eap_grid,
                          "n_boot": args.n_boot, "max_se": SE, "min_scenarios": FLOOR,
                          "stop_rule": "eap"},
               "n_models": int(len(dep)),
               "median_item_param_se": {d: median_item_param_se},
               "se_components": {d: {
                   "median_se_posterior": float(dep["se_post"].median()),
                   "median_se_param": float(dep["se_param"].median()),
                   "median_se_total": float(dep["se_total"].median()),
                   "median_bar_inflation": float(dep["bar_inflation"].median()),
                   "max_bar_inflation": float(dep["bar_inflation"].max())}}}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame([{"skill": d, "mean_se_ability": float(dep["se_post"].mean()),
                   "mean_se_param": float(dep["se_param"].mean()),
                   "mean_se_total": float(dep["se_total"].mean()),
                   "median_bar_inflation": float(dep["bar_inflation"].median()),
                   "max_bar_inflation": float(dep["bar_inflation"].max()),
                   "n": int(len(dep))}]).to_csv(out / "results.csv", index=False)

    # se_vs_floor.csv (EAP stop; floors 8/10/12)
    svf = []
    for fl in (8, 10, 12):
        dfl = deployed_rows(fl)
        svf.append({"min_scenarios": fl, "mean_scenarios": round(float(dfl["n_scen"].mean()), 2),
                    "se_ability": round(float(dfl["se_post"].median()), 4),
                    "se_param": round(float(dfl["se_param"].median()), 4),
                    "se_total": round(float(dfl["se_total"].median()), 4)})
    pd.DataFrame(svf).to_csv(out / "se_vs_floor.csv", index=False)

    _fig_mean_total_se(dep, out / "figures" / "mean_total_se.png")
    _fig_se_ability_vs_total(dep, out / "figures" / "se_ability_vs_total.png")
    pd.DataFrame([
        {"metric": "SE_ability_only", "mean": float(dep["se_post"].mean()),
         "sd_across_models": float(dep["se_post"].std(ddof=1)),
         "median": float(dep["se_post"].median()), "n_models": len(dep), "se_target": SE,
         "regime": "deployed_cat_8_0.12", "source": "se_post_vs_total.csv"},
        {"metric": "SE_total", "mean": float(dep["se_total"].mean()),
         "sd_across_models": float(dep["se_total"].std(ddof=1)),
         "median": float(dep["se_total"].median()), "n_models": len(dep), "se_target": SE,
         "regime": "deployed_cat_8_0.12", "source": "se_post_vs_total.csv"},
    ]).to_csv(out / "se_ability_vs_total_summary.csv", index=False)
    _fig_se_vs_floor(pd.DataFrame(svf), out / "figures" / "se_vs_floor.png")
    print(f"  exp-07: SE_post median={dep['se_post'].median():.4f} mean={dep['se_post'].mean():.4f}"
          f"  SE_total median={dep['se_total'].median():.4f}  floor-SE_param={median_item_param_se:.4f}")
    return dep, metrics


def _fig_mean_total_se(dep, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    means = [dep["se_post"].mean(), dep["se_total"].mean()]
    sds = [dep["se_post"].std(ddof=1), dep["se_total"].std(ddof=1)]
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.bar([0, 1], means, 0.58, color=["#4d648d", "#c1666b"], yerr=sds, capsize=6,
           error_kw={"elinewidth": 1.6, "ecolor": "#333"})
    ax.axhline(SE, ls=":", color="k", lw=1.6)
    ax.text(1.45, SE + 0.004, "SE target=0.12", ha="right", va="bottom", fontsize=9)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["SE ability-only", "SE total (+calibration)"])
    ax.set_ylabel("SE (general-ability theta, 1-D MWLE)")
    ax.set_title("BiGGen general-ability SE: ability-only vs total (EAP stop, 8/0.12; N=52)", fontsize=10)
    for i, (mn, sd) in enumerate(zip(means, sds)):
        ax.annotate(f"{mn:.3f}\n(SD {sd:.3f})", (i, mn), ha="center", va="bottom", fontsize=9,
                    xytext=(0, 3), textcoords="offset points")
    ax.set_ylim(0, max(means[1] + sds[1], SE) * 1.25)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _fig_se_ability_vs_total(dep, path):
    _fig_mean_total_se(dep, path)


def _fig_se_vs_floor(svf, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(svf["min_scenarios"], svf["se_ability"], "o-", label="SE ability")
    ax.plot(svf["min_scenarios"], svf["se_param"], "s-", label="SE param")
    ax.plot(svf["min_scenarios"], svf["se_total"], "^-", label="SE total")
    ax.axhline(SE, ls=":", color="k", label="SE target 0.12")
    ax.set_xlabel("min_scenarios (floor)"); ax.set_ylabel("median deployed SE")
    ax.set_title("BiGGen deployed SE vs floor (EAP stop)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


# ---------------------------------------------------------------------------
# exp-04 efficiency (adaptive arm -> EAP OOS; random baseline kept)
# ---------------------------------------------------------------------------

def step_efficiency(df05, metrics05, args):
    out = EXP / "04_efficiency_vs_random"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    d = DIM
    ref = df05[f"theta_ref_{d}"].to_numpy(); mw = df05[f"theta_mwle_{d}"].to_numpy()
    rs = recovery_stats(ref, mw)
    # convergence = fraction of held-out models reaching SE 0.12 within the L_max cap
    if "precision_reached" in df05.columns:
        conv = 100.0 * float(df05["precision_reached"].astype(bool).mean())
    else:
        conv = 100.0 * float((df05["scenarios_administered"] < args.l_max).mean())
    adaptive = {"arm": "adaptive", "mean_scen": metrics05["mean_scenarios_administered"],
                "mean_crit": metrics05["mean_criteria_administered"], "conv_pct": round(conv, 1),
                "oos_r": rs["r"], "oos_slope": rs["slope"], "n": len(df05)}
    # random baseline is stop-independent (uncapped random administration) -> keep archived row
    arc = EXP / "archive_onlineSE" / "04_efficiency_vs_random" / "results.csv"
    random_row = None
    if arc.is_file():
        old = pd.read_csv(arc)
        rr = old[old["arm"] == "random"]
        if not rr.empty:
            random_row = rr.iloc[0].to_dict()
    rows = [adaptive] + ([random_row] if random_row else [])
    with (out / "results.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["arm", "mean_scen", "mean_crit", "conv_pct",
                                           "oos_r", "oos_slope", "n"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in ("arm", "mean_scen", "mean_crit", "conv_pct",
                                              "oos_r", "oos_slope", "n")})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    arms = [r["arm"] for r in rows]
    colors = ["#1b7837", "#762a83"][:len(arms)]
    ax1.bar(arms, [float(r["mean_scen"]) for r in rows], color=colors)
    ax1.set_ylabel("mean scenarios administered")
    ax1.set_title("Test length: adaptive (EAP stop) vs random")
    for i, r in enumerate(rows):
        ax1.text(i, float(r["mean_scen"]), f"{float(r['mean_scen']):.1f}", ha="center", va="bottom")
    ax2.bar(arms, [float(r["oos_r"]) for r in rows], color=colors)
    ax2.set_ylim(0.9, 1.0); ax2.set_ylabel("OOS recovery r (MWLE vs full-bank EAP)")
    ax2.set_title("OOS recovery r: adaptive vs random")
    for i, r in enumerate(rows):
        ax2.text(i, float(r["oos_r"]), f"{float(r['oos_r']):.3f}", ha="center", va="bottom")
    fig.suptitle("BiGGen efficiency: adaptive CAT (EAP stop, 8/0.12) vs random baseline", fontsize=12)
    fig.tight_layout(); fig.savefig(out / "figures" / "efficiency_adaptive_vs_random.png", dpi=140)
    plt.close(fig)
    print(f"  exp-04: adaptive r={rs['r']:.4f} slope={rs['slope']:.3f} "
          f"len {adaptive['mean_scen']:.2f} scen conv={adaptive['conv_pct']}%"
          + (f" | random kept (r={random_row['oos_r']:.3f}, {random_row['mean_scen']:.0f} scen)"
             if random_row else ""))


# ---------------------------------------------------------------------------
# exp-10 order/seed stability (8 seeds under EAP)
# ---------------------------------------------------------------------------

def step_order_seed(S, dep_df, args):
    out = EXP / "10_order_seed"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    seeds = [args.seed + s for s in range(args.n_seeds)]
    theta_by_seed = np.full((len(S["models"]), args.n_seeds), np.nan)
    for si, seed in enumerate(seeds):
        admin = E.administer_eap(S["models"], args.bank, args.matrix, args.scenarios, S["dims"],
                                 S["a1"], S["b"], S["col"], S["scen_of"], S["Y"], S["row_of"],
                                 S["gg"], S["lp"], seed=seed, floor=FLOOR, target=SE,
                                 l_max=args.l_max, workers=args.workers,
                                 runs_dir=str(args.tmp_dir / f"seed{seed}"))
        for mi, m in enumerate(S["models"]):
            idx = admin[m]["idx"]
            if idx.size:
                th0 = scat.eap_subset(S["Y"][S["row_of"][m]], idx, S["A"], S["b"], S["egrid"], S["elog"])
                thm, _ = scat.mwle_subset(S["Y"][S["row_of"][m]], idx, S["A"], S["b"], th0)
                theta_by_seed[mi, si] = thm[0]
        print(f"    order/seed: seed {seed} done", flush=True)
    theta_sd = np.nanstd(theta_by_seed, axis=1, ddof=1)
    theta_range = np.nanmax(theta_by_seed, axis=1) - np.nanmin(theta_by_seed, axis=1)
    se_total_med = float(dep_df["se_total"].median())
    row = {"n_seeds": args.n_seeds, "mean_sd": float(np.nanmean(theta_sd)),
           "median_sd": float(np.nanmedian(theta_sd)), "max_sd": float(np.nanmax(theta_sd)),
           "mean_range": float(np.nanmean(theta_range)), "median_range": float(np.nanmedian(theta_range)),
           "max_range": float(np.nanmax(theta_range)),
           "mean_sd_frac_setarget": float(np.nanmean(theta_sd) / SE),
           "mean_sd_frac_setotal": float(np.nanmean(theta_sd) / se_total_med)}
    pd.DataFrame([row]).to_csv(out / "results.csv", index=False)
    pd.DataFrame({"model": S["models"], "theta_sd": theta_sd, "theta_range": theta_range}).to_csv(
        out / "per_model_spread.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.hist(theta_sd[np.isfinite(theta_sd)], bins=16, color="#4d648d", alpha=0.85)
    ax.axvline(float(np.nanmean(theta_sd)), ls="--", color="k", label=f"mean SD={np.nanmean(theta_sd):.3f}")
    ax.axvline(SE, ls=":", color="#c1666b", label=f"SE target {SE}")
    ax.set_xlabel("across-seed theta SD"); ax.set_ylabel("models")
    ax.set_title(f"BiGGen order/seed stability (EAP stop, 8/0.12, {args.n_seeds} seeds)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "figures" / "seed_spread_general.png", dpi=140); plt.close(fig)
    print(f"  exp-10: theta SD across seeds mean={row['mean_sd']:.4f} median={row['median_sd']:.4f} "
          f"max={row['max_sd']:.4f} (frac SE_target {row['mean_sd_frac_setarget']:.3f})")


# ---------------------------------------------------------------------------
# exp-08 leaderboard (full-bank theta + EAP SE bars + weakly_identified)
# ---------------------------------------------------------------------------

def step_leaderboard(S, dep_df, args):
    d = DIM
    out = EXP / "08_leaderboard"
    out.mkdir(parents=True, exist_ok=True)
    theta_full = scat.eap_all_models(S["Y"], S["Mall"], S["A"], S["b"], S["egrid"], S["elog"])[:, 0]
    obs_pass = np.array([S["Y"][i][S["Mall"][i]].mean() for i in range(len(S["models"]))])
    dep = dep_df.set_index("model")
    weak = set(dep_df[~dep_df["precision_reached"]]["model"])  # EAP-native deployed caps
    rows = []
    for i, m in enumerate(S["models"]):
        rows.append({"model": m, "theta": float(theta_full[i]),
                     "se_ability": float(dep.loc[m, "se_post"]) if m in dep.index else np.nan,
                     "se_param": float(dep.loc[m, "se_param"]) if m in dep.index else np.nan,
                     "se_total": float(dep.loc[m, "se_total"]) if m in dep.index else np.nan,
                     "observed_pass_rate": float(obs_pass[i]),
                     "weakly_identified": bool(m in weak)})
    df = pd.DataFrame(rows).sort_values("theta", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    df.to_csv(out / "results.csv", index=False)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    # spearman vs archived CAT-admin ranking
    spearman = None
    arc = EXP / "archive_onlineSE" / "08_leaderboard_catadmin" / "results.csv"
    if arc.is_file():
        old = pd.read_csv(arc)[["model", "rank"]].rename(columns={"rank": "rank_catadmin"})
        mrg = df.merge(old, on="model")
        spearman = round(float(pd.Series(mrg["rank"]).corr(pd.Series(mrg["rank_catadmin"]),
                                                           method="spearman")), 5)
    meta = {"generated_at": datetime.now(timezone.utc).isoformat(),
            "purpose": "full-bank ability leaderboard with SE_total bars (NOT the locked CAT)",
            "scoring": "full modeled bank, 1D fine-EAP theta (stop-independent)",
            "se_bars": "SE_total = sqrt(SE_ability_posterior^2 + SE_param^2) from exp-07 (EAP deployed)",
            "n_models": len(df), "n_criteria_scored": len(S["ids"]),
            "theta_range": [float(df["theta"].min()), float(df["theta"].max())],
            "se_total_range": [float(df["se_total"].min()), float(df["se_total"].max())],
            "top5": df.head(5)[["model", "theta", "se_total"]].to_dict("records"),
            "bottom5": df.tail(5)[["model", "theta", "se_total"]].to_dict("records"),
            "weakly_identified_source": "EAP-native deployed caps at 8/0.12 (posterior SD>0.12 within L=40)",
            "weakly_identified_models": df[df["weakly_identified"]][
                ["model", "theta", "se_total"]].to_dict("records"),
            "n_weakly_identified": int(df["weakly_identified"].sum()),
            "spearman_vs_catadmin_ranking": spearman}
    (out / "metrics.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    prov = {"of_record": True, "leaderboard_theta": "full-bank fine-EAP (stop-independent)",
            "eap_grid": args.eap_grid, "eap_range": args.range,
            "se_bars_from": "exp-07 EAP-deployed leaderboard_se_components.csv",
            "weakly_identified": "EAP-native deployed caps at 8/0.12",
            "note": "Mirrors WildBench: leaderboard theta is FULL-BANK (does not use the CAT stop). "
                    "The prior CAT-administered-theta leaderboard is archived under "
                    "experiments/archive_onlineSE/08_leaderboard_catadmin/.",
            "spearman_vs_catadmin_ranking": spearman,
            "generated_by": "biggen_calibration/scripts/biggen_adopt_eap.py"}
    (out / "of_record_provenance.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")

    _fig_leaderboard(df, out / "figures" / "leaderboard_general.png")
    print(f"  exp-08: full-bank theta range [{df['theta'].min():.2f}, {df['theta'].max():.2f}]  "
          f"weakly_identified={int(df['weakly_identified'].sum())}  spearman_vs_catadmin={spearman}")
    return df


def _fig_leaderboard(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = df.sort_values("theta").reset_index(drop=True)
    ypos = np.arange(len(order)); wi = order["weakly_identified"].to_numpy()
    fig, ax = plt.subplots(figsize=(8.6, max(9, 0.22 * len(order))))
    ax.errorbar(order["theta"], ypos, xerr=1.96 * order["se_total"], fmt="none",
                ecolor="#c1666b", elinewidth=2.2, alpha=0.6, label="+/-1.96 SE_total")
    ax.errorbar(order["theta"], ypos, xerr=1.96 * order["se_ability"], fmt="none",
                ecolor="#4d648d", elinewidth=1.0, label="+/-1.96 SE_ability")
    ax.scatter(order["theta"][~wi], ypos[~wi], s=12, color="k", zorder=3)
    ax.scatter(order["theta"][wi], ypos[wi], s=55, marker="<", facecolor="none",
               edgecolor="#d95f0e", linewidth=1.4, zorder=4,
               label="weakly identified (theta is an upper bound)")
    labels = [f"{m}  (weakly id.)" if w else m for m, w in zip(order["model"], wi)]
    ax.set_yticks(ypos); ax.set_yticklabels(labels, fontsize=5)
    for tick, w in zip(ax.get_yticklabels(), wi):
        if w:
            tick.set_color("#d95f0e")
    ax.set_xlabel("full-bank ability (theta, 1-D)")
    ax.set_title("BiGGen general-ability leaderboard (full-bank theta, EAP-deployed SE_total bars)",
                 fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


# ---------------------------------------------------------------------------
# promote Phase-A EAP grid -> exp-06 / 06b_operating_point of-record
# ---------------------------------------------------------------------------

def step_promote_grid(args):
    src = EXP / "12_eap_stop_grid"
    dst = EXP / "06_floor_se_grid" / "06b_operating_point"
    if not src.is_file() and not src.exists():
        print("  (exp-12 grid not found; skipping grid promotion)")
        return
    dst.mkdir(parents=True, exist_ok=True)
    _copy(src / "recovery_grid.csv", dst / "recovery_grid.csv")
    _copy(src / "recommendation.json", dst / "recommendation.json")
    _copy(src / "figures", dst / "figures")
    (dst / "PROVENANCE.txt").write_text(
        "Of-record operating-point grid under the EAP-posterior stop (floor 8 / SE 0.12 locked). "
        "Promoted from the Phase-A study experiments/12_eap_stop_grid/ (biggen_eap_stop_grid.py "
        "--stop-se eap). The prior online-SE grid is archived under "
        "experiments/archive_onlineSE/06b_operating_point/.\n", encoding="utf-8")
    print(f"  promoted EAP grid -> {dst}")


def step_prototype_note(args):
    note = EXP / "11_eap_stop_prototype" / "NOTE.md"
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(
        "# EAP-posterior stop vs online-SE stop (ADOPTED as of-record)\n\n"
        "> **UPDATE - ADOPTED.** Following this Phase-A prototype (and the floor x SE grid in "
        "`experiments/12_eap_stop_grid/`), the EAP-posterior stop is now BiGGen's **of-record** "
        "CAT stop rule at the locked **floor 8 / SE 0.12**. The stop-dependent experiments "
        "(exp-04/05/07/09/10 + deployed SE + the exp-06/06b operating-point grid + the exp-08 "
        "leaderboard SE bars) were re-run under it; the prior online-SE outputs are archived under "
        "`experiments/archive_onlineSE/`. The fitted bank is UNCHANGED (stop-rule change only, no "
        "re-fit). See the top-level `README.md`. This file is kept for provenance of the original "
        "head-to-head that motivated the switch (online reached the 0.12 posterior-SD target for "
        "only 28.8% of models vs 88.5% under EAP; OOS recovery r 0.9715->0.9823, slope "
        "0.813->0.921).\n", encoding="utf-8")
    print(f"  marked prototype ADOPTED -> {note}")


# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bank", type=Path, default=ROOT / "staging" / "biggen_unidim_modeled.jsonl")
    p.add_argument("--matrix", type=Path, default=ROOT / "staging" / "biggen_response_matrix.csv")
    p.add_argument("--scenarios", type=Path, default=ROOT / "data" / "BiGGen" / "scenarios.jsonl")
    p.add_argument("--tmp-dir", type=Path, default=ROOT / "staging" / "_biggen_adopt_eap")
    p.add_argument("--eap-grid", type=int, default=GRID_MODES["eap_grid"])
    p.add_argument("--range", type=float, default=GRID_MODES["range"])
    p.add_argument("--fit-grid", type=int, default=7)
    p.add_argument("--ridge", type=float, default=1e-2)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260729)
    p.add_argument("--pu-seed", type=int, default=20260801)
    p.add_argument("--l-max", type=int, default=E.EAP_L_MAX)
    p.add_argument("--n-boot", type=int, default=150)
    p.add_argument("--n-seeds", type=int, default=8)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--force-archive", action="store_true")
    p.add_argument("--steps", type=str, default="all",
                   help="comma list: archive,recovery,pirt,deployed_se,efficiency,order_seed,"
                        "leaderboard,grid,note (default all).")
    args = p.parse_args()
    args.tmp_dir.mkdir(parents=True, exist_ok=True)
    steps = ({"archive", "recovery", "pirt", "deployed_se", "efficiency", "order_seed",
              "leaderboard", "grid", "note"} if args.steps == "all"
             else set(s.strip() for s in args.steps.split(",") if s.strip()))

    print("=" * 92)
    print("BiGGen: ADOPT EAP-posterior stop as of-record (floor 8 / SE 0.12). Bank UNCHANGED.")
    print("=" * 92)

    if "archive" in steps:
        archive_online(args)

    S = load_shared(args)
    print("forced-long full-data order + EAP walks ...", flush=True)
    full_walks = forced_long_full(S, args)
    need_folds = steps & {"recovery", "pirt", "efficiency"}
    fold_data = fit_folds(S, args) if need_folds else None

    df05 = metrics05 = None
    if "recovery" in steps:
        df05, metrics05 = step_recovery(S, fold_data, args)
    if "pirt" in steps:
        if df05 is None:
            df05 = pd.read_csv(EXP / "05_oos_recovery" / "oos_per_model.csv")
        step_pirt(df05, args)

    dep_df = None
    if "deployed_se" in steps:
        dep_df, _ = step_deployed_se(S, full_walks, args)
    if "efficiency" in steps:
        if df05 is None or metrics05 is None:
            df05 = pd.read_csv(EXP / "05_oos_recovery" / "oos_per_model.csv")
            metrics05 = json.loads((EXP / "05_oos_recovery" / "metrics.json").read_text())
        step_efficiency(df05, metrics05, args)
    if "order_seed" in steps:
        if dep_df is None:
            lc = pd.read_csv(EXP / "07_parameter_uncertainty" / "leaderboard_se_components.csv")
            dep_df = lc.rename(columns={f"se_total_{DIM}": "se_total"})
        step_order_seed(S, dep_df, args)
    if "leaderboard" in steps:
        if dep_df is None or "se_post" not in dep_df.columns:
            lc = pd.read_csv(EXP / "07_parameter_uncertainty" / "leaderboard_se_components.csv")
            pv = pd.read_csv(EXP / "07_parameter_uncertainty" / "se_post_vs_total.csv")
            dep_df = pd.DataFrame({"model": lc["model"],
                                   "se_post": lc[f"se_posterior_{DIM}"],
                                   "se_param": lc[f"se_param_{DIM}"],
                                   "se_total": lc[f"se_total_{DIM}"]})
            dep_df["precision_reached"] = dep_df["se_post"] <= SE
        step_leaderboard(S, dep_df, args)
    if "grid" in steps:
        step_promote_grid(args)
    if "note" in steps:
        step_prototype_note(args)

    shutil.rmtree(args.tmp_dir, ignore_errors=True)
    print("\nDONE. Review git status under biggen_calibration/ before committing (user commits).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
