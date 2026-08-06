"""PROTOTYPE (Phase A, exploratory -- NOT of-record): EAP-posterior-SD stop vs online-SE stop
for BiGGen at the CURRENT locked op-point (floor 8 / SE 0.12).

Mirrors the WildBench prototype (``wildbench_calibration/scripts/prototype_eap_stop.py``). It
changes ONLY the CAT stop rule -- it does NOT modify the of-record engine, refit the bank, or
overwrite any of-record experiment. The adaptive SCENARIO SELECTION is the production engine's
(unchanged): we run the engine to a long forced length to obtain each model's adaptive
administration ORDER, then apply the EAP-posterior stop rule POST-HOC on that order (stop at
the first scenario with n_scenarios >= 8 AND EAP posterior SD <= 0.12, else the forced cap).

Head-to-head with the ONLINE-SE stop (engine's own normal-approx SE, floor 8 / SE 0.12), run
here too at the same seed (20260729) and bank for an apples-to-apples comparison. SE_param is
computed fresh via the observed-information parametric bootstrap over each rule's administered
set (exp-07 machinery, reused from ``biggen_recovery_grid``). Writes everything to
``experiments/11_eap_stop_prototype/`` (prototype only; of-record 04/05/07/08/09/10 untouched).
"""

from __future__ import annotations

import argparse
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
from scipy.special import logsumexp

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

TARGET = E.EAP_TARGET
FLOOR = E.EAP_FLOOR


def run_online(models, bank, matrix, scenarios, dims, seed, min_evals, workers, runs_dir):
    spec = scat.RunSpec(seed=seed, top_n=5, max_se=TARGET, min_evals_per_skill=min_evals,
                        min_scenarios=FLOOR, max_scenarios=50, selection="trace",
                        mode="cat", runs_dir=runs_dir)
    return {r["model"]: r for r in scat.run_models(models, bank, matrix, scenarios,
                                                   "clamp", dims, spec, workers=workers)}


def run_forced(models, bank, matrix, scenarios, dims, seed, l_max, workers, runs_dir):
    spec = scat.RunSpec(seed=seed, top_n=5, max_se=0.0, min_evals_per_skill=0,
                        min_scenarios=l_max, max_scenarios=l_max, selection="trace",
                        mode="cat", runs_dir=runs_dir)
    return {r["model"]: r for r in scat.run_models(models, bank, matrix, scenarios,
                                                   "clamp", dims, spec, workers=workers)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    base = ROOT / "biggen_calibration"
    p.add_argument("--bank", type=Path, default=ROOT / "staging" / "biggen_unidim_modeled.jsonl")
    p.add_argument("--matrix", type=Path, default=ROOT / "staging" / "biggen_response_matrix.csv")
    p.add_argument("--scenarios", type=Path, default=ROOT / "data" / "BiGGen" / "scenarios.jsonl")
    p.add_argument("--out-dir", type=Path,
                   default=base / "experiments" / "11_eap_stop_prototype")
    p.add_argument("--tmp-dir", type=Path, default=ROOT / "staging" / "_biggen_eap_proto")
    p.add_argument("--dim", default="general")
    p.add_argument("--l-max", type=int, default=E.EAP_L_MAX, help="forced admin cap (scenarios).")
    p.add_argument("--eap-grid", type=int, default=3201, help="fine EAP nodes (of-record ref).")
    p.add_argument("--range", type=float, default=8.0)
    p.add_argument("--fit-grid", type=int, default=7)
    p.add_argument("--ridge", type=float, default=1e-2)
    p.add_argument("--se-param-grid", type=int, default=241,
                   help="EAP nodes for the SE_param bootstrap (coarser; resolves SE_param).")
    p.add_argument("--se-param-range", type=float, default=6.0)
    p.add_argument("--min-evals-per-skill", type=int, default=15)
    p.add_argument("--n-boot", type=int, default=150)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260729)
    p.add_argument("--pu-seed", type=int, default=20260801)
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()
    (args.out_dir / "figures").mkdir(parents=True, exist_ok=True)
    args.tmp_dir.mkdir(parents=True, exist_ok=True)
    d = args.dim

    records, dims, _ = scat.load_fitted_bank(args.bank, "clamp")
    assert len(dims) == 1, "BiGGen prototype is 1-D"
    ids, A, b = scat.assemble_arrays(records, dims)
    a = A[:, 0]
    scen_of = {r["criterion_id"]: r["scenario_id"] for r in records}
    matrix = pd.read_csv(args.matrix, index_col=0)
    Yraw = matrix.reindex(columns=ids).to_numpy(float)
    Mall = ~np.isnan(Yraw)
    Y = np.nan_to_num(Yraw, nan=0.0)
    models = list(matrix.index)
    row_of = {m: i for i, m in enumerate(models)}
    col = {c: i for i, c in enumerate(ids)}
    gg, lp = E.eap_grid(args.eap_grid, args.range)
    egrid, elog = scat.build_grid(1, args.eap_grid, args.range)

    print("=" * 88)
    print(f"BiGGen PROTOTYPE: EAP-posterior stop vs online-SE stop (N={len(models)}, floor {FLOOR}/SE {TARGET})")
    print("=" * 88)

    # --- SE_param machinery (full-data observed info) for SE_total honesty comparison ---
    print("computing full-data item covariance (observed info) ...", flush=True)
    cov = rg.compute_item_cov(args.bank, args.matrix, args.fit_grid, args.ridge)
    bgrid = np.linspace(-args.se_param_range, args.se_param_range, args.se_param_grid)
    blp = -0.5 * bgrid ** 2
    blp = blp - logsumexp(blp)
    rng_boot = np.random.default_rng(args.pu_seed)

    # --- full-data head-to-head at floor 8 / SE 0.12 ---
    print("running ONLINE-SE stop (floor 8 / SE 0.12) full-data ...", flush=True)
    resO = run_online(models, args.bank, args.matrix, args.scenarios, dims, args.seed,
                      args.min_evals_per_skill, args.workers, str(args.tmp_dir / "online"))
    print(f"running forced-long (L={args.l_max}) full-data for the EAP-stop order ...", flush=True)
    resE = run_forced(models, args.bank, args.matrix, args.scenarios, dims, args.seed,
                      args.l_max, args.workers, str(args.tmp_dir / "forced"))

    rows = []
    for m in models:
        i = row_of[m]
        yrow = Y[i]
        # ONLINE rule: EAP posterior SD over the online-administered set (honesty check)
        wO = E.eap_walk(resO[m]["order"], yrow, col, scen_of, a, b, gg, lp)
        seO = wO[-1]["eap_sd"] if wO else np.nan
        oidx = wO[-1]["idx"] if wO else np.array([], dtype=int)
        nO_scen = resO[m]["scenarios_administered"]
        nO_crit = len(oidx)
        # EAP rule: walk forced-long order, stop at EAP SD<=0.12 & >=8 scen
        wE = E.eap_walk(resE[m]["order"], yrow, col, scen_of, a, b, gg, lp)
        stepE, reachedE = E.eap_stop_point(wE, FLOOR, TARGET)
        seE = stepE["eap_sd"]
        nE_scen = stepE["n_scen"]
        nE_crit = stepE["n_crit"]
        # SE_param (bootstrap) over each rule's administered set
        _, _, spO = rg.bootstrap_theta(yrow, oidx, cov["beta"], cov["chol"], bgrid, blp,
                                       args.n_boot, rng_boot)
        _, _, spE = rg.bootstrap_theta(yrow, stepE["idx"], cov["beta"], cov["chol"], bgrid, blp,
                                       args.n_boot, rng_boot)
        rows.append({
            "model": m,
            "online_eap_mean": round(wO[-1]["eap_mean"], 4) if wO else np.nan,
            "online_scenarios": nO_scen, "online_criteria": nO_crit,
            "online_SE_ability": round(seO, 4), "online_reached": bool(seO <= TARGET),
            "online_engine_se": round(float(resO[m]["se_online"][0]), 4),
            "eap_scenarios": nE_scen, "eap_criteria": nE_crit,
            "eap_SE_ability": round(seE, 4), "eap_reached": bool(reachedE),
            "eap_hit_cap": bool(not reachedE),
            "online_SE_param": round(spO, 4) if np.isfinite(spO) else np.nan,
            "eap_SE_param": round(spE, 4) if np.isfinite(spE) else np.nan,
            "online_SE_total": round(float(np.sqrt(seO ** 2 + spO ** 2)), 4) if np.isfinite(spO) else np.nan,
            "eap_SE_total": round(float(np.sqrt(seE ** 2 + spE ** 2)), 4) if np.isfinite(spE) else np.nan,
        })
    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "per_model_comparison.csv", index=False)

    # --- OOS k=5 recovery: BOTH stop rules, same folds/seed (apples-to-apples) ---
    print("OOS k=5 recovery under ONLINE and EAP stop (same folds) ...", flush=True)
    folds = E.make_folds(models, args.k, args.seed)
    ref_all, cat_eap, cat_online = [], [], []
    lenE_scen, lenE_crit, lenO_scen, lenO_crit = [], [], [], []
    Q_all = np.ones((len(ids), 1), dtype=int)
    for f in range(args.k):
        test = folds[f]
        train = [mm for mm in models if mm not in set(test)]
        tr = [row_of[mm] for mm in train]
        Ytr, Mtr = Yraw[tr], Mall[tr]
        keep = [j for j in range(len(ids)) if Mtr[:, j].sum() >= 2
                and 0 < Ytr[Mtr[:, j], j].sum() < Mtr[:, j].sum()]
        keep = np.array(keep, dtype=int)
        fit = cm.fit_m2pl_em(np.nan_to_num(Ytr[:, keep], nan=0.0), Mtr[:, keep],
                             Q_all[keep], args.fit_grid, ridge=args.ridge, max_iter=200)
        Ak, bk = fit["A"], fit["b"]
        ak = Ak[:, 0]
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
        theta_ref = scat.eap_all_models(Yte, Mte, Ak, bk, egrid, elog)
        resEf = run_forced(test, fold_bank, args.matrix, args.scenarios, dims, args.seed,
                           args.l_max, args.workers, str(args.tmp_dir / "fold_forced"))
        resOf = run_online(test, fold_bank, args.matrix, args.scenarios, dims, args.seed,
                           args.min_evals_per_skill, args.workers, str(args.tmp_dir / "fold_online"))
        for ti, m in enumerate(test):
            y = Yte[ti]
            # EAP-stop theta
            wE = E.eap_walk(resEf[m]["order"], y, colk, scen_of, ak, bk, gg, lp)
            stepE, _ = E.eap_stop_point(wE, FLOOR, TARGET)
            idxE = stepE["idx"] if stepE is not None else np.array([], dtype=int)
            if idxE.size:
                th_ba = scat.eap_subset(y, idxE, Ak, bk, egrid, elog)
                thE, _ = scat.mwle_subset(y, idxE, Ak, bk, th_ba)
                thE = float(thE[0])
            else:
                thE = float(theta_ref[ti][0])
            # ONLINE-stop theta
            idxO = np.array([colk[c] for c in resOf[m]["order"] if c in colk], dtype=int)
            if idxO.size:
                th_ba = scat.eap_subset(y, idxO, Ak, bk, egrid, elog)
                thO, _ = scat.mwle_subset(y, idxO, Ak, bk, th_ba)
                thO = float(thO[0])
            else:
                thO = float(theta_ref[ti][0])
            ref_all.append(float(theta_ref[ti][0]))
            cat_eap.append(thE)
            cat_online.append(thO)
            lenE_scen.append(stepE["n_scen"] if stepE else 0)
            lenE_crit.append(stepE["n_crit"] if stepE else 0)
            lenO_scen.append(resOf[m]["scenarios_administered"])
            lenO_crit.append(idxO.size)

    def _rec(ref, cat, ls, lc):
        ref = np.array(ref); cat = np.array(cat)
        band = scat.ols_ci_band(ref, cat, np.linspace(ref.min(), ref.max(), 40), B=2000, seed=0)
        return {"r": round(band["r"], 4), "r_lo": round(band["r_lo"], 4),
                "r_hi": round(band["r_hi"], 4), "slope": round(band["slope"], 4),
                "theta_mae": round(float(np.mean(np.abs(cat - ref))), 4),
                "mean_scenarios": round(float(np.mean(ls)), 2),
                "mean_criteria": round(float(np.mean(lc)), 1)}

    recE = _rec(ref_all, cat_eap, lenE_scen, lenE_crit)
    recO = _rec(ref_all, cat_online, lenO_scen, lenO_crit)

    shutil.rmtree(args.tmp_dir, ignore_errors=True)

    n = len(df)
    comp = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "PROTOTYPE / Phase A exploratory -- NOT adopted. of-record 04/05/07/08/09/10 untouched.",
        "design": {
            "stop_rule_eap": "EAP posterior SD <= 0.12 AND n_scenarios >= 8, else forced cap",
            "stop_rule_online": "engine online normal-approx SE <= 0.12 AND floor 8",
            "selection": "unchanged (production engine max-info scenario selection)",
            "seed": args.seed, "l_max_cap": args.l_max, "eap_grid": args.eap_grid,
            "eap_range": args.range, "estimator": "mwle",
            "se_param": "fresh observed-info parametric bootstrap over each rule's admin set"},
        "n_models": n,
        "length_deployed_full_data": {
            "online": {"scenarios": E.dist(df["online_scenarios"]),
                       "criteria": E.dist(df["online_criteria"])},
            "eap": {"scenarios": E.dist(df["eap_scenarios"]),
                    "criteria": E.dist(df["eap_criteria"])}},
        "pct_reaching_eap_sd_le_0.12": {
            "online_rule": {"n": int(df["online_reached"].sum()),
                            "pct": round(100.0 * df["online_reached"].mean(), 1)},
            "eap_rule": {"n": int(df["eap_reached"].sum()),
                         "pct": round(100.0 * df["eap_reached"].mean(), 1),
                         "n_hit_cap": int(df["eap_hit_cap"].sum())}},
        "se_ability_distribution_deployed": {"online": E.dist(df["online_SE_ability"]),
                                             "eap": E.dist(df["eap_SE_ability"])},
        "se_total_distribution_deployed": {"online": E.dist(df["online_SE_total"].dropna()),
                                           "eap": E.dist(df["eap_SE_total"].dropna())},
        "oos_recovery": {"eap_rule": recE, "online_rule": recO},
        "models_that_still_cap_under_eap": df[df["eap_hit_cap"]]["model"].tolist(),
    }
    (args.out_dir / "comparison.json").write_text(json.dumps(comp, indent=2), encoding="utf-8")

    _figure(df, args.out_dir / "figures" / "eap_vs_online_stop.png")

    print("\n--- LENGTH (scenarios, deployed full-data) ---")
    print(f"online: median={comp['length_deployed_full_data']['online']['scenarios']['median']} "
          f"mean={comp['length_deployed_full_data']['online']['scenarios']['mean']}  |  "
          f"eap: median={comp['length_deployed_full_data']['eap']['scenarios']['median']} "
          f"mean={comp['length_deployed_full_data']['eap']['scenarios']['mean']}")
    print("--- % reaching EAP posterior SD <= 0.12 ---")
    print(f"online: {comp['pct_reaching_eap_sd_le_0.12']['online_rule']['n']}/{n} "
          f"({comp['pct_reaching_eap_sd_le_0.12']['online_rule']['pct']}%)  |  "
          f"eap: {comp['pct_reaching_eap_sd_le_0.12']['eap_rule']['n']}/{n} "
          f"({comp['pct_reaching_eap_sd_le_0.12']['eap_rule']['pct']}%, "
          f"cap={comp['pct_reaching_eap_sd_le_0.12']['eap_rule']['n_hit_cap']})")
    print("--- SE_ability median (deployed) ---")
    print(f"online={comp['se_ability_distribution_deployed']['online']['median']}  "
          f"eap={comp['se_ability_distribution_deployed']['eap']['median']}")
    print("--- OOS recovery ---")
    print(f"online: r={recO['r']} slope={recO['slope']} MAE={recO['theta_mae']} "
          f"(len {recO['mean_scenarios']} scen / {recO['mean_criteria']} crit)")
    print(f"eap   : r={recE['r']} slope={recE['slope']} MAE={recE['theta_mae']} "
          f"(len {recE['mean_scenarios']} scen / {recE['mean_criteria']} crit)")
    print(f"\nwrote -> {args.out_dir}")
    return 0


def _figure(df, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(df))
    dd = df.sort_values("online_SE_ability").reset_index(drop=True)
    ax1.scatter(x, dd["online_SE_ability"], s=14, color="#4d648d", label="online-SE stop")
    ax1.scatter(x, dd["eap_SE_ability"], s=14, color="#c1666b", marker="x", label="EAP stop")
    ax1.axhline(TARGET, ls=":", color="k", label="target 0.12")
    ax1.set_xlabel("models (sorted by online SE_ability)")
    ax1.set_ylabel("final SE_ability (EAP posterior SD)")
    ax1.set_title("SE_ability at stop: online vs EAP rule"); ax1.legend(fontsize=8)
    ax2.scatter(df["online_scenarios"], df["eap_scenarios"], s=16, alpha=0.8, color="#333")
    lim = max(df["eap_scenarios"].max(), df["online_scenarios"].max()) + 2
    ax2.plot([0, lim], [0, lim], ls="--", color="gray")
    ax2.set_xlabel("online-rule scenarios"); ax2.set_ylabel("EAP-rule scenarios")
    ax2.set_title("Test length: EAP vs online (above y=x => EAP longer)")
    fig.suptitle(f"PROTOTYPE: EAP-posterior stop vs online-SE stop (BiGGen, N={len(df)})", fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
