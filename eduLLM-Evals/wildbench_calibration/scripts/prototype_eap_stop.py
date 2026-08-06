"""PROTOTYPE (exploratory, do NOT adopt): EAP-posterior-SE stop rule vs online-SE stop rule.

This changes ONLY the CAT stop rule -- it does NOT modify the of-record engine, refit the
bank, or re-run the SE_param bootstrap. The adaptive SCENARIO SELECTION is the production
engine's (unchanged): we run the engine to a long forced length to obtain each model's
adaptive administration ORDER, then apply the EAP-posterior stop rule POST-HOC on that order:

  at each scenario boundary compute the EAP posterior SD over the administered items so far on
  the fine theta grid (posterior ∝ L(administered|theta)·N(0,1)); STOP at the first scenario
  with n_scenarios >= 8 AND posterior SD <= 0.12, else continue to the forced cap (= "hits cap").

Compared head-to-head (same seed 20260729, same bank) with the ONLINE-SE stop rule (engine's
own normal-approx SE), run here too for an apples-to-apples comparison. SE_param is REUSED from
the of-record deployed bootstrap (se_post_vs_total.csv) -- no re-bootstrap; conservative for the
(longer) EAP test. Writes everything to experiments/13_eap_stop_prototype/ (prototype only).
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
from scipy.special import log_expit, logsumexp

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import wildbench_scenario_lib as L  # noqa: E402

ROOT = L.ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import scripts.scenario_cat_lib as scat  # noqa: E402

cm = L.cm
TARGET = 0.12
FLOOR = 8


def make_folds(models, k, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(models))
    folds = [[] for _ in range(k)]
    for pos, i in enumerate(idx):
        folds[pos % k].append(models[int(i)])
    return [sorted(f) for f in folds]


def eap_walk(order, yrow, col, scen_of, a, b, gg, lp):
    """Per-scenario cumulative EAP (mean, SD) along an administration order.

    Returns list of dicts {n_scen, n_crit, eap_mean, eap_sd, idx (cumulative col indices)}.
    """
    G = gg.size
    ll_data = np.zeros(G)
    out = []
    cur = None
    n_scen = 0
    n_crit = 0
    cum_idx = []
    for cid in order:
        j = col.get(cid)
        if j is None:
            continue
        sid = scen_of.get(cid)
        if sid != cur:
            if cur is not None:
                post = np.exp(ll_data + lp - logsumexp(ll_data + lp))
                mean = float(post @ gg)
                sd = float(np.sqrt(max(post @ (gg ** 2) - mean ** 2, 0.0)))
                out.append({"n_scen": n_scen, "n_crit": n_crit, "eap_mean": mean,
                            "eap_sd": sd, "idx": np.array(cum_idx, dtype=int)})
            cur = sid
            n_scen += 1
        eta = a[j] * gg - b[j]
        ll_data = ll_data + yrow[j] * log_expit(eta) + (1.0 - yrow[j]) * log_expit(-eta)
        n_crit += 1
        cum_idx.append(j)
    if cur is not None:
        post = np.exp(ll_data + lp - logsumexp(ll_data + lp))
        mean = float(post @ gg)
        sd = float(np.sqrt(max(post @ (gg ** 2) - mean ** 2, 0.0)))
        out.append({"n_scen": n_scen, "n_crit": n_crit, "eap_mean": mean,
                    "eap_sd": sd, "idx": np.array(cum_idx, dtype=int)})
    return out


def eap_stop_point(walk, floor=FLOOR, target=TARGET):
    """First scenario step meeting (n_scen>=floor AND eap_sd<=target); else last (capped)."""
    for step in walk:
        if step["n_scen"] >= floor and step["eap_sd"] <= target:
            return step, True
    return (walk[-1], False) if walk else (None, False)


def run_forced(models, bank, matrix_path, scenarios_path, dims, seed, l_max, workers):
    spec = scat.RunSpec(seed=seed, top_n=5, max_se=0.0, min_evals_per_skill=0,
                        min_scenarios=l_max, max_scenarios=l_max, selection="trace",
                        mode="cat", runs_dir=str(Path(bank).parent / "_proto_runs"))
    return {r["model"]: r for r in scat.run_models(models, bank, matrix_path, scenarios_path,
                                                   "clamp", dims, spec, workers=workers)}


def run_online(models, bank, matrix_path, scenarios_path, dims, seed, workers):
    spec = scat.RunSpec(seed=seed, top_n=5, max_se=TARGET, min_evals_per_skill=10,
                        min_scenarios=FLOOR, max_scenarios=50, selection="trace",
                        mode="cat", runs_dir=str(Path(bank).parent / "_proto_runs_o"))
    return {r["model"]: r for r in scat.run_models(models, bank, matrix_path, scenarios_path,
                                                   "clamp", dims, spec, workers=workers)}


def dist(s):
    s = np.asarray(s, float)
    return {"median": round(float(np.median(s)), 4), "mean": round(float(np.mean(s)), 4),
            "q1": round(float(np.quantile(s, .25)), 4), "q3": round(float(np.quantile(s, .75)), 4)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    base = ROOT / "wildbench_calibration"
    p.add_argument("--bank", type=Path,
                   default=base / "wildbench_scenario_fitted_1d_catpool.jsonl")
    p.add_argument("--matrix", type=Path, default=L.DEFAULT_MATRIX)
    p.add_argument("--scenarios", type=Path, default=L.DEFAULT_SCENARIOS)
    p.add_argument("--se-components", type=Path,
                   default=base / "experiments" / "07_parameter_uncertainty" / "se_post_vs_total.csv")
    p.add_argument("--out-dir", type=Path, default=base / "experiments" / "13_eap_stop_prototype")
    p.add_argument("--tmp-dir", type=Path,
                   default=base / "experiments" / "13_eap_stop_prototype" / "_tmp")
    p.add_argument("--l-max", type=int, default=40, help="forced administration cap (scenarios).")
    p.add_argument("--eap-grid", type=int, default=321)
    p.add_argument("--range", type=float, default=8.0)
    p.add_argument("--fit-grid", type=int, default=7)
    p.add_argument("--ridge", type=float, default=1e-2)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260729)
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()
    (args.out_dir / "figures").mkdir(parents=True, exist_ok=True)
    args.tmp_dir.mkdir(parents=True, exist_ok=True)

    records, dims, _ = scat.load_fitted_bank(args.bank, "clamp")
    d = dims[0]
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
    gg = np.linspace(-args.range, args.range, args.eap_grid)
    lp = -0.5 * gg ** 2
    lp = lp - logsumexp(lp)
    egrid, elog = scat.build_grid(1, args.eap_grid, args.range)

    se_param_by = {}
    if args.se_components.is_file():
        sc = pd.read_csv(args.se_components)
        sc = sc[sc["regime"] == "deployed_cat_8_0.12"]
        se_param_by = dict(zip(sc["model"], sc["SE_param"]))

    print("=" * 84)
    print("PROTOTYPE: EAP-posterior stop vs online-SE stop (WildBench, 52 models)")
    print("=" * 84)

    # --- full-data: online-SE run (O) + forced-long run (E) at seed 20260729 ---
    print("running ONLINE-SE stop (floor 8 / SE 0.12) ...", flush=True)
    resO = run_online(models, args.bank, args.matrix, args.scenarios, dims, args.seed, args.workers)
    print(f"running forced-long (L_max={args.l_max}) for the EAP stop order ...", flush=True)
    resE = run_forced(models, args.bank, args.matrix, args.scenarios, dims, args.seed,
                      args.l_max, args.workers)

    rows = []
    for m in models:
        i = row_of[m]
        yrow = Y[i]
        # ONLINE rule: EAP SD over the online-administered set
        oidx = np.array([col[c] for c in resO[m]["order"] if c in col], dtype=int)
        wO = eap_walk(resO[m]["order"], yrow, col, scen_of, a, b, gg, lp)
        seO = wO[-1]["eap_sd"] if wO else np.nan
        nO_scen = resO[m]["scenarios_administered"]
        nO_crit = len(oidx)
        # EAP rule: walk forced-long order, stop at EAP SD<=0.12 & >=8 scen
        wE = eap_walk(resE[m]["order"], yrow, col, scen_of, a, b, gg, lp)
        stepE, reachedE = eap_stop_point(wE)
        seE = stepE["eap_sd"]
        nE_scen = stepE["n_scen"]
        nE_crit = stepE["n_crit"]
        sp = float(se_param_by.get(m, np.nan))
        rows.append({
            "model": m, "theta_online_rule_eap": round(wO[-1]["eap_mean"], 4) if wO else np.nan,
            "online_scenarios": nO_scen, "online_criteria": nO_crit,
            "online_SE_ability": round(seO, 4),
            "online_reached": bool(seO <= TARGET),
            "eap_scenarios": nE_scen, "eap_criteria": nE_crit,
            "eap_SE_ability": round(seE, 4), "eap_reached": bool(reachedE),
            "eap_hit_cap": bool(not reachedE),
            "SE_param_reused": round(sp, 4) if np.isfinite(sp) else np.nan,
            "online_SE_total": round(float(np.sqrt(seO ** 2 + sp ** 2)), 4) if np.isfinite(sp) else np.nan,
            "eap_SE_total": round(float(np.sqrt(seE ** 2 + sp ** 2)), 4) if np.isfinite(sp) else np.nan,
        })
    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "per_model_comparison.csv", index=False)

    # --- OOS k=5 recovery under the EAP stop (mirrors exp-05, EAP stop) ---
    print("OOS k=5 recovery under the EAP stop ...", flush=True)
    folds = make_folds(models, args.k, args.seed)
    ref_all, cat_all = [], []
    len_scen, len_crit = [], []
    for f in range(args.k):
        test = folds[f]
        train = [mm for mm in models if mm not in set(test)]
        tr = [row_of[mm] for mm in train]
        Ytr, Mtr = Yraw[tr], Mall[tr]
        keep = [j for j in range(len(ids)) if Mtr[:, j].sum() >= 2
                and 0 < Ytr[Mtr[:, j], j].sum() < Mtr[:, j].sum()]
        keep = np.array(keep, dtype=int)
        fit = cm.fit_m2pl_em(np.nan_to_num(Ytr[:, keep], nan=0.0), Mtr[:, keep],
                             np.ones((keep.size, 1), int), args.fit_grid, ridge=args.ridge,
                             max_iter=200)
        Ak, bk = fit["A"], fit["b"]
        kept_ids = [ids[j] for j in keep]
        colk = {c: i for i, c in enumerate(kept_ids)}
        ak = Ak[:, 0]
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
                           args.l_max, args.workers)
        for ti, m in enumerate(test):
            wE = eap_walk(resEf[m]["order"], Yte[ti], colk, scen_of, ak, bk, gg, lp)
            stepE, _ = eap_stop_point(wE)
            idx = stepE["idx"] if stepE is not None else np.array([], dtype=int)
            if idx.size:
                th_ba = scat.eap_subset(Yte[ti], idx, Ak, bk, egrid, elog)
                th_mw, _ = scat.mwle_subset(Yte[ti], idx, Ak, bk, th_ba)
                th = float(th_mw[0])
            else:
                th = float(theta_ref[ti][0])
            ref_all.append(float(theta_ref[ti][0]))
            cat_all.append(th)
            len_scen.append(stepE["n_scen"] if stepE else 0)
            len_crit.append(stepE["n_crit"] if stepE else 0)
    ref_arr = np.array(ref_all); cat_arr = np.array(cat_all)
    band = scat.ols_ci_band(ref_arr, cat_arr, np.linspace(ref_arr.min(), ref_arr.max(), 40),
                            B=2000, seed=0)
    rec = {"r": round(band["r"], 4), "r_lo": round(band["r_lo"], 4), "r_hi": round(band["r_hi"], 4),
           "slope": round(band["slope"], 4),
           "theta_mae": round(float(np.mean(np.abs(cat_arr - ref_arr))), 4),
           "mean_scenarios": round(float(np.mean(len_scen)), 2),
           "mean_criteria": round(float(np.mean(len_crit)), 1)}

    shutil.rmtree(args.tmp_dir, ignore_errors=True)
    shutil.rmtree(Path(args.bank).parent / "_proto_runs", ignore_errors=True)
    shutil.rmtree(Path(args.bank).parent / "_proto_runs_o", ignore_errors=True)

    # --- aggregate comparison ---
    n = len(df)
    comp = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "PROTOTYPE / exploratory -- NOT adopted. of-record exp-04/05/06/07/08 untouched.",
        "design": {"stop_rule_new": "EAP posterior SD <= 0.12 AND n_scenarios >= 8, else cap",
                   "stop_rule_online": "engine online normal-approx SE <= 0.12 AND floor 8",
                   "selection": "unchanged (production engine max-info scenario selection)",
                   "seed": args.seed, "l_max_cap": args.l_max, "eap_grid": args.eap_grid,
                   "se_param": "REUSED from of-record deployed bootstrap (no re-bootstrap)"},
        "n_models": n,
        "length": {
            "online": {"scenarios": dist(df["online_scenarios"]), "criteria": dist(df["online_criteria"])},
            "eap": {"scenarios": dist(df["eap_scenarios"]), "criteria": dist(df["eap_criteria"])},
        },
        "pct_reaching_eap_sd_le_0.12": {
            "online_rule": {"n": int(df["online_reached"].sum()),
                            "pct": round(100.0 * df["online_reached"].mean(), 1)},
            "eap_rule": {"n": int(df["eap_reached"].sum()),
                         "pct": round(100.0 * df["eap_reached"].mean(), 1),
                         "n_hit_cap": int(df["eap_hit_cap"].sum())},
        },
        "se_ability_distribution": {"online": dist(df["online_SE_ability"]),
                                    "eap": dist(df["eap_SE_ability"])},
        "se_total_distribution": {"online": dist(df["online_SE_total"].dropna()),
                                  "eap": dist(df["eap_SE_total"].dropna())},
        "oos_recovery_eap_rule": rec,
        "oos_recovery_online_of_record_exp05": {"r": 0.9567, "slope": 0.8899, "theta_mae": 0.4283,
                                                "mean_scenarios": 8.98, "mean_criteria": 86.2},
        "models_that_still_cap_under_eap": df[df["eap_hit_cap"]]["model"].tolist(),
    }
    (args.out_dir / "comparison.json").write_text(json.dumps(comp, indent=2), encoding="utf-8")

    _figure(df, args.out_dir / "figures" / "eap_vs_online_stop.png")

    print("\n--- LENGTH (scenarios) ---")
    print(f"online: median={comp['length']['online']['scenarios']['median']} "
          f"mean={comp['length']['online']['scenarios']['mean']}  |  "
          f"eap: median={comp['length']['eap']['scenarios']['median']} "
          f"mean={comp['length']['eap']['scenarios']['mean']}")
    print("--- %% reaching EAP SD<=0.12 ---")
    print(f"online: {comp['pct_reaching_eap_sd_le_0.12']['online_rule']['n']}/{n} "
          f"({comp['pct_reaching_eap_sd_le_0.12']['online_rule']['pct']}%)  |  "
          f"eap: {comp['pct_reaching_eap_sd_le_0.12']['eap_rule']['n']}/{n} "
          f"({comp['pct_reaching_eap_sd_le_0.12']['eap_rule']['pct']}%, "
          f"cap={comp['pct_reaching_eap_sd_le_0.12']['eap_rule']['n_hit_cap']})")
    print("--- SE_ability median ---")
    print(f"online={comp['se_ability_distribution']['online']['median']}  "
          f"eap={comp['se_ability_distribution']['eap']['median']}")
    print("--- OOS recovery (EAP rule) ---")
    print(f"r={rec['r']} slope={rec['slope']} MAE={rec['theta_mae']} "
          f"(len {rec['mean_scenarios']} scen / {rec['mean_criteria']} crit) "
          f"vs online-of-record r=0.9567/slope 0.890/MAE 0.428")
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
    ax1.set_xlabel("models (sorted by online SE_ability)"); ax1.set_ylabel("final SE_ability (EAP posterior SD)")
    ax1.set_title("SE_ability at stop: online vs EAP rule"); ax1.legend(fontsize=8)
    ax2.scatter(df["online_scenarios"], df["eap_scenarios"], s=16, alpha=0.8, color="#333")
    lim = max(df["eap_scenarios"].max(), df["online_scenarios"].max()) + 2
    ax2.plot([0, lim], [0, lim], ls="--", color="gray")
    ax2.set_xlabel("online-rule scenarios"); ax2.set_ylabel("EAP-rule scenarios")
    ax2.set_title("Test length: EAP vs online (above y=x => EAP longer)")
    fig.suptitle("PROTOTYPE: EAP-posterior stop vs online-SE stop (WildBench, N=52)", fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
