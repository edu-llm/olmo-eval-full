"""Phase A favorability check: EAP-posterior-SD stop vs online-SE stop at the CURRENT
locked Bridge operating point (min_scenarios=12 / SE 0.15).

Mirrors the WildBench ``prototype_eap_stop.py`` and BiGGen ``biggen_eap_stop_*`` studies but
at Bridge's locked point. It changes ONLY the CAT stop rule (no engine change, no bank refit).
For each of the 51 models it compares, head-to-head on the same seed (20260729) and bank:

  * ONLINE rule  -- the deployed engine stop (normal-approx online SE <= 0.15 AND floor 12),
  * EAP rule     -- post-hoc EAP posterior SD <= 0.15 AND floor 12 on a forced-long adaptive
                    order (production selection unchanged), else the forced cap.

Reports, for the user's adoption decision:
  * % of models reaching EAP posterior SD <= 0.15 under each rule,
  * test-length distributions (scenarios AND criteria),
  * final SE_ability (EAP posterior SD) and SE_total (op-point-specific SE_param) distributions,
  * OOS k=5 model-fold recovery (MWLE theta vs full-bank fine-EAP reference) under each rule.

Because a Bridge scenario is a HEAVY ~18-criterion testlet, ability SD after 12 scenarios is
already far below 0.15 (of-record ability_se_median ~0.047 at f12_se0.15), so the min_scenarios
floor binds and the online-vs-EAP gap is expected to be SMALL -- unlike WildBench/BiGGen
(~11-criterion testlets). This script quantifies that honestly.

SE_param uses the of-record observed-information bootstrap machinery
(``scenario_param_uncertainty``) on each rule's administered set. Writes everything to
``experiments/13_eap_stop_prototype/`` (a NEW prototype dir; NOT of-record). Does NOT overwrite
the online-SE of-record grid or any downstream file.
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
sys.path.insert(0, str(_HERE))
import bridge_scenario_lib as L  # noqa: E402
import bridge_eap_stop_lib as E  # noqa: E402

ROOT = L.ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import scripts.scenario_cat_lib as scat  # noqa: E402

cm = L.cm

# Bridge step-2 SE_param module by explicit path (filename collides with the reference
# scripts/scenario_param_uncertainty.py that bridge_scenario_lib puts on sys.path first).
_pu_spec = importlib.util.spec_from_file_location(
    "bridge_scenario_param_uncertainty", _HERE / "scenario_param_uncertainty.py")
PU = importlib.util.module_from_spec(_pu_spec)
sys.modules["bridge_scenario_param_uncertainty"] = PU
_pu_spec.loader.exec_module(PU)


def run_online(models, bank, matrix, scenarios, dims, seed, floor, target, max_scen,
               min_evals, workers, runs_dir):
    spec = scat.RunSpec(seed=seed, top_n=5, max_se=target, min_evals_per_skill=min_evals,
                        min_scenarios=floor, max_scenarios=max_scen, selection="trace",
                        mode="cat", runs_dir=runs_dir)
    return {r["model"]: r for r in scat.run_models(models, bank, matrix, scenarios,
                                                   "clamp", dims, spec, workers=workers)}


def run_forced(models, bank, matrix, scenarios, dims, seed, l_max, workers, runs_dir):
    spec = scat.RunSpec(seed=seed, top_n=5, max_se=0.0, min_evals_per_skill=0,
                        min_scenarios=l_max, max_scenarios=l_max, selection="trace",
                        mode="cat", runs_dir=runs_dir)
    return {r["model"]: r for r in scat.run_models(models, bank, matrix, scenarios,
                                                   "clamp", dims, spec, workers=workers)}


def recovery_stats(x, y):
    band = scat.ols_ci_band(x, y, np.linspace(x.min(), x.max(), 40), B=2000, seed=0)
    return {"r": round(band["r"], 4), "r_lo": round(band["r_lo"], 4),
            "r_hi": round(band["r_hi"], 4), "slope": round(band["slope"], 4),
            "theta_mae": round(float(np.mean(np.abs(y - x))), 4)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    base = ROOT / "bridge_calibration"
    p.add_argument("--bank", type=Path, default=base / "bridge_scenario_fitted_1d_catpool.jsonl")
    p.add_argument("--matrix", type=Path, default=ROOT / "bridgegrade" / "response_matrix.csv")
    p.add_argument("--scenarios", type=Path, default=ROOT / "data" / "Bridge" / "scenarios.jsonl")
    p.add_argument("--out-dir", type=Path, default=base / "experiments" / "13_eap_stop_prototype")
    p.add_argument("--tmp-dir", type=Path,
                   default=base / "experiments" / "13_eap_stop_prototype" / "_tmp")
    p.add_argument("--of-record-exp05", type=Path,
                   default=base / "experiments" / "05_oos_recovery" / "recovery_metrics.json")
    p.add_argument("--floor", type=int, default=12)
    p.add_argument("--target", type=float, default=0.15)
    p.add_argument("--online-max-scenarios", type=int, default=228,
                   help="online-rule cap (matches exp-05 of-record recovery run).")
    p.add_argument("--min-evals-per-skill", type=int, default=15)
    p.add_argument("--l-max", type=int, default=60, help="forced-long cap for the EAP order.")
    p.add_argument("--eap-grid", type=int, default=321, help="FINE EAP nodes (ref + stop walk).")
    p.add_argument("--range", type=float, default=8.0)
    p.add_argument("--fit-grid", type=int, default=7)
    p.add_argument("--se-param-grid", type=int, default=61)
    p.add_argument("--se-param-range", type=float, default=6.0)
    p.add_argument("--ridge", type=float, default=1e-2)
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260729)
    p.add_argument("--pu-seed", type=int, default=20260801)
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()
    (args.out_dir / "figures").mkdir(parents=True, exist_ok=True)
    args.tmp_dir.mkdir(parents=True, exist_ok=True)
    floor, target = args.floor, args.target

    records, dims, _ = scat.load_fitted_bank(args.bank, "clamp")
    assert len(dims) == 1, "Bridge is 1-D"
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
    gg, lp = E.eap_grid(args.eap_grid, args.range)
    egrid, elog = scat.build_grid(1, args.eap_grid, args.range)

    print("=" * 88)
    print(f"PROTOTYPE: EAP-posterior stop vs online-SE stop (Bridge, N={len(models)}) "
          f"at floor {floor} / SE {target}")
    print("=" * 88)

    # --- SE_param machinery (full-data observed info; op-point-specific admin sets) ---
    cov = PU.compute_item_cov(args.bank, args.matrix, args.fit_grid, args.ridge)
    bgrid = np.linspace(-args.se_param_range, args.se_param_range, args.se_param_grid)
    blp = -0.5 * bgrid ** 2
    blp = blp - logsumexp(blp)
    rng_boot = np.random.default_rng(args.pu_seed)

    # --- full-data: online-SE run (O) + forced-long run (E) ---
    print(f"running ONLINE-SE stop (floor {floor} / SE {target}) ...", flush=True)
    resO = run_online(models, args.bank, args.matrix, args.scenarios, dims, args.seed, floor,
                      target, args.online_max_scenarios, args.min_evals_per_skill, args.workers,
                      str(args.tmp_dir / "online_runs"))
    print(f"running forced-long (L_max={args.l_max}) for the EAP stop order ...", flush=True)
    resE = run_forced(models, args.bank, args.matrix, args.scenarios, dims, args.seed,
                      args.l_max, args.workers, str(args.tmp_dir / "forced_runs"))

    rows = []
    for m in models:
        yrow = Y[row_of[m]]
        # ONLINE rule: EAP SD over the online-administered set
        wO = E.eap_walk(resO[m]["order"], yrow, col, scen_of, a, b, gg, lp)
        oidx = np.array([col[c] for c in resO[m]["order"] if c in col], dtype=int)
        seO = wO[-1]["eap_sd"] if wO else np.nan
        nO_scen = resO[m]["scenarios_administered"]
        nO_crit = len(oidx)
        _, _, spO = PU.bootstrap_theta(yrow, oidx, cov["beta"], cov["chol"], bgrid, blp,
                                       args.n_boot, rng_boot)
        # EAP rule: walk forced-long order, stop at EAP SD<=target & >=floor scen
        wE = E.eap_walk(resE[m]["order"], yrow, col, scen_of, a, b, gg, lp)
        stepE, reachedE = E.eap_stop_point(wE, floor, target)
        seE = stepE["eap_sd"]
        eidx = stepE["idx"]
        nE_scen = stepE["n_scen"]
        nE_crit = stepE["n_crit"]
        _, _, spE = PU.bootstrap_theta(yrow, eidx, cov["beta"], cov["chol"], bgrid, blp,
                                       args.n_boot, rng_boot)
        rows.append({
            "model": m,
            "online_scenarios": nO_scen, "online_criteria": nO_crit,
            "online_SE_ability": round(float(seO), 4),
            "online_reached": bool(seO <= target),
            "online_SE_param": round(float(spO), 4) if np.isfinite(spO) else np.nan,
            "online_SE_total": round(float(np.sqrt(seO ** 2 + spO ** 2)), 4)
            if np.isfinite(spO) else np.nan,
            "eap_scenarios": nE_scen, "eap_criteria": nE_crit,
            "eap_SE_ability": round(float(seE), 4), "eap_reached": bool(reachedE),
            "eap_hit_cap": bool(not reachedE),
            "eap_SE_param": round(float(spE), 4) if np.isfinite(spE) else np.nan,
            "eap_SE_total": round(float(np.sqrt(seE ** 2 + spE ** 2)), 4)
            if np.isfinite(spE) else np.nan,
        })
    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "per_model_comparison.csv", index=False)

    # --- OOS k=5 recovery under BOTH rules (same folds; mirrors exp-05) ---
    print("OOS k=5 recovery under EAP stop AND online stop ...", flush=True)
    folds = E.make_folds(models, args.k, args.seed)
    ref_all = []
    on_all, ep_all = [], []
    on_len_scen, on_len_crit, ep_len_scen, ep_len_crit = [], [], [], []
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
        # per-fold: online-rule engine run + forced-long order
        resOf = run_online(test, fold_bank, args.matrix, args.scenarios, dims, args.seed, floor,
                           target, args.online_max_scenarios, args.min_evals_per_skill,
                           args.workers, str(args.tmp_dir / f"online_f{f}"))
        resEf = run_forced(test, fold_bank, args.matrix, args.scenarios, dims, args.seed,
                           args.l_max, args.workers, str(args.tmp_dir / f"forced_f{f}"))
        for ti, m in enumerate(test):
            yv = Yte[ti]
            ref_all.append(float(theta_ref[ti][0]))
            # online rule
            oidx = np.array([colk[c] for c in resOf[m]["order"] if c in colk], dtype=int)
            if oidx.size:
                th_ba = scat.eap_subset(yv, oidx, Ak, bk, egrid, elog)
                th_mw, _ = scat.mwle_subset(yv, oidx, Ak, bk, th_ba)
                on_all.append(float(th_mw[0]))
            else:
                on_all.append(float(theta_ref[ti][0]))
            on_len_scen.append(resOf[m]["scenarios_administered"])
            on_len_crit.append(oidx.size)
            # eap rule
            wE = E.eap_walk(resEf[m]["order"], yv, colk, scen_of, ak, bk, gg, lp)
            stepE, _ = E.eap_stop_point(wE, floor, target)
            eidx = stepE["idx"] if stepE is not None else np.array([], dtype=int)
            if eidx.size:
                th_ba = scat.eap_subset(yv, eidx, Ak, bk, egrid, elog)
                th_mw, _ = scat.mwle_subset(yv, eidx, Ak, bk, th_ba)
                ep_all.append(float(th_mw[0]))
            else:
                ep_all.append(float(theta_ref[ti][0]))
            ep_len_scen.append(stepE["n_scen"] if stepE else 0)
            ep_len_crit.append(stepE["n_crit"] if stepE else 0)

    ref_arr = np.array(ref_all)
    rec_online = recovery_stats(ref_arr, np.array(on_all))
    rec_online["mean_scenarios"] = round(float(np.mean(on_len_scen)), 2)
    rec_online["mean_criteria"] = round(float(np.mean(on_len_crit)), 1)
    rec_eap = recovery_stats(ref_arr, np.array(ep_all))
    rec_eap["mean_scenarios"] = round(float(np.mean(ep_len_scen)), 2)
    rec_eap["mean_criteria"] = round(float(np.mean(ep_len_crit)), 1)

    of_record = {}
    if args.of_record_exp05.is_file():
        j = json.loads(args.of_record_exp05.read_text(encoding="utf-8"))
        h = j.get("headline_recovery_mwle", {})
        of_record = {"r": h.get("r"), "slope": h.get("slope"), "theta_mae": h.get("mae"),
                     "mean_scenarios": j.get("mean_scenarios_administered"),
                     "mean_criteria": j.get("mean_criteria_administered")}

    n = len(df)
    comp = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "PHASE A PROTOTYPE / decision-support -- NOT adopted. of-record "
                  "exp-04/05/06/06b/07/08/09/10/11/12 untouched.",
        "operating_point": {"min_scenarios": floor, "se_target": target},
        "design": {
            "stop_rule_new": f"EAP posterior SD <= {target} AND n_scenarios >= {floor}, else cap",
            "stop_rule_online": f"engine online normal-approx SE <= {target} AND floor {floor}",
            "selection": "unchanged (production engine max-info scenario selection)",
            "estimator": "MWLE (of-record deployment estimator)",
            "reference": f"full-bank FINE-EAP ({args.eap_grid} nodes over +/-{args.range})",
            "seed": args.seed, "l_max_cap": args.l_max, "eap_grid": args.eap_grid,
            "se_param": "op-point-specific observed-info bootstrap over the administered set",
        },
        "n_models": n,
        "length": {
            "online": {"scenarios": E.dist(df["online_scenarios"]),
                       "criteria": E.dist(df["online_criteria"])},
            "eap": {"scenarios": E.dist(df["eap_scenarios"]),
                    "criteria": E.dist(df["eap_criteria"])},
        },
        "pct_reaching_eap_sd_le_target": {
            "target": target,
            "online_rule": {"n": int(df["online_reached"].sum()),
                            "pct": round(100.0 * df["online_reached"].mean(), 1)},
            "eap_rule": {"n": int(df["eap_reached"].sum()),
                         "pct": round(100.0 * df["eap_reached"].mean(), 1),
                         "n_hit_cap": int(df["eap_hit_cap"].sum())},
        },
        "se_ability_distribution": {"online": E.dist(df["online_SE_ability"]),
                                    "eap": E.dist(df["eap_SE_ability"])},
        "se_total_distribution": {"online": E.dist(df["online_SE_total"].dropna()),
                                  "eap": E.dist(df["eap_SE_total"].dropna())},
        "oos_recovery": {"online_rule": rec_online, "eap_rule": rec_eap,
                         "of_record_exp05_online": of_record},
        "models_that_cap_under_eap": df[df["eap_hit_cap"]]["model"].tolist(),
        "favorability_note": (
            "Bridge scenarios are heavy ~18-criterion testlets: ability SD after the floor of "
            f"{floor} scenarios is already far below {target}, so the min_scenarios floor binds "
            "and the online and EAP rules stop at nearly the same length. The honesty gain "
            "(models the online rule declares 'converged' at SE<=target but that the EAP "
            "posterior SD would not) is therefore expected to be SMALL relative to WildBench / "
            "BiGGen (~11-criterion testlets), where lighter testlets leave a larger online-vs-"
            "EAP gap. See pct_reaching_eap_sd_le_target and the SE_ability distributions."),
    }
    (args.out_dir / "comparison.json").write_text(json.dumps(comp, indent=2), encoding="utf-8")

    _figure(df, target, args.out_dir / "figures" / "eap_vs_online_stop.png", floor)

    shutil.rmtree(args.tmp_dir, ignore_errors=True)

    print("\n--- LENGTH (scenarios) ---")
    print(f"online: median={comp['length']['online']['scenarios']['median']} "
          f"mean={comp['length']['online']['scenarios']['mean']}  |  "
          f"eap: median={comp['length']['eap']['scenarios']['median']} "
          f"mean={comp['length']['eap']['scenarios']['mean']}")
    print(f"--- %% reaching EAP SD<={target} ---")
    print(f"online: {comp['pct_reaching_eap_sd_le_target']['online_rule']['n']}/{n} "
          f"({comp['pct_reaching_eap_sd_le_target']['online_rule']['pct']}%)  |  "
          f"eap: {comp['pct_reaching_eap_sd_le_target']['eap_rule']['n']}/{n} "
          f"({comp['pct_reaching_eap_sd_le_target']['eap_rule']['pct']}%, "
          f"cap={comp['pct_reaching_eap_sd_le_target']['eap_rule']['n_hit_cap']})")
    print("--- SE_ability median ---")
    print(f"online={comp['se_ability_distribution']['online']['median']}  "
          f"eap={comp['se_ability_distribution']['eap']['median']}")
    print("--- OOS recovery (MWLE vs fine-EAP ref) ---")
    print(f"online-rule: r={rec_online['r']} slope={rec_online['slope']} "
          f"MAE={rec_online['theta_mae']} (len {rec_online['mean_scenarios']} scen)")
    print(f"eap-rule   : r={rec_eap['r']} slope={rec_eap['slope']} "
          f"MAE={rec_eap['theta_mae']} (len {rec_eap['mean_scenarios']} scen)")
    print(f"of-record exp05 online: r={of_record.get('r')} slope={of_record.get('slope')} "
          f"MAE={of_record.get('theta_mae')}")
    print(f"\nwrote -> {args.out_dir}")
    return 0


def _figure(df, target, path: Path, floor):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(df))
    dd = df.sort_values("online_SE_ability").reset_index(drop=True)
    ax1.scatter(x, dd["online_SE_ability"], s=14, color="#4d648d", label="online-SE stop")
    ax1.scatter(x, dd["eap_SE_ability"], s=14, color="#c1666b", marker="x", label="EAP stop")
    ax1.axhline(target, ls=":", color="k", label=f"target {target}")
    ax1.set_xlabel("models (sorted by online SE_ability)")
    ax1.set_ylabel("final SE_ability (EAP posterior SD)")
    ax1.set_title("SE_ability at stop: online vs EAP rule"); ax1.legend(fontsize=8)
    ax2.scatter(df["online_scenarios"], df["eap_scenarios"], s=16, alpha=0.8, color="#333")
    lim = max(df["eap_scenarios"].max(), df["online_scenarios"].max()) + 2
    ax2.plot([0, lim], [0, lim], ls="--", color="gray")
    ax2.set_xlabel("online-rule scenarios"); ax2.set_ylabel("EAP-rule scenarios")
    ax2.set_title("Test length: EAP vs online (above y=x => EAP longer)")
    fig.suptitle(f"PROTOTYPE: EAP-posterior stop vs online-SE stop (Bridge, N={len(df)}, "
                 f"floor {floor} / SE {target})", fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
