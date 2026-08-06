"""EAP-posterior stop-rule prototype (multidimensional TutorBench / TutorEval).

Offline, no engine change: for each model we drive the REAL production engine
(scenario_cat_lib.run_one_model -> tutor_cat.engine, trace selection) to the scenario cap
with the early stop disabled, capturing the exact administration ORDER. Because scenario
selection does not depend on the stop rule, both stop rules are just different truncation
points along that one sequence. We replay the order through the real M2PL update
(tutor_cat.mirt.update) and, at each scenario boundary (>= min_scenarios), compare:

  * online SE   = sqrt(diag(U))                      (current normal-approx stop)
  * EAP SD      = per-skill marginal SD of the JOINT posterior over a dense product grid
                  (scenario_cat_lib.eap_subset_mean_var), i.e. the honest uncertainty

For each rule we record test length, ability at stop, and whether the HONEST EAP SD is
<= target. %-reaching-target under the online rule shows how often it stops optimistically;
under the EAP rule it shows the honest convergence rate. Recovery = corr(theta_at_stop,
full-bank dense-EAP reference) per skill. Locked op-point: min_scenarios=12, SE=0.30.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)
import scenario_cat_lib as scl          # noqa: E402
from tutor_cat import mirt              # noqa: E402


def trajectory(order, rubrics, Yrow, col, A, b, dims, sgrid, slp, scen_of):
    """Replay the administered order; return per-scenario-boundary metrics."""
    n = len(dims)
    theta, U = mirt.initial_state(None, None, n)
    idx, bnds, nscen = [], [], 0
    for i, c in enumerate(order):
        r = rubrics[c]
        y = int(Yrow[col[c]])
        theta, U, _ = mirt.update(theta, U, r.a, r.q, r.b, y)
        idx.append(col[c])
        last = i == len(order) - 1
        boundary = last or (scen_of[order[i + 1]] != scen_of[c])
        if boundary:
            nscen += 1
            _, var = scl.eap_subset_mean_var(Yrow, np.array(idx), A, b, sgrid, slp)
            bnds.append({"nscen": nscen, "ncrit": len(idx), "theta": theta.copy(),
                         "online_se": np.sqrt(np.diag(U)), "eap_sd": np.sqrt(var)})
    return bnds


def first_stop(bnds, key, target, min_scen):
    for bnd in bnds:
        if bnd["nscen"] >= min_scen and bool(np.all(bnd[key] <= target)):
            return bnd, True
    return bnds[-1], False   # capped


def recov(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    r = float(np.corrcoef(x, y)[0, 1]) if x.size > 2 else float("nan")
    s = float(np.polyfit(x, y, 1)[0]) if x.size > 2 else float("nan")
    return r, s, float(np.mean(np.abs(x - y)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--scenarios", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--min-scenarios", type=int, default=12)
    ap.add_argument("--se-target", type=float, default=0.30)
    ap.add_argument("--cap", type=int, default=50)
    ap.add_argument("--ref-nodes", type=int, default=41, help="dense reference grid nodes/dim")
    ap.add_argument("--stop-nodes", type=int, default=25, help="EAP-stop grid nodes/dim")
    ap.add_argument("--seed", type=int, default=20260729)
    ap.add_argument("--negative-policy", default="clamp")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    records, dims, stats = scl.load_fitted_bank(args.bank, args.negative_policy)
    rubrics = scl.build_rubrics(records, dims)
    scen_raw = scl.load_scenarios(args.scenarios)
    scen_of = {cid: r.scenario_id for cid, r in rubrics.items()}
    matrix = pd.read_csv(args.matrix, index_col=0)
    ids_all, A_all, b_all = scl.assemble_arrays(records, dims)
    keep = [i for i, c in enumerate(ids_all) if c in matrix.columns]
    ids = [ids_all[i] for i in keep]; A = A_all[keep]; b = b_all[keep]
    col = {c: i for i, c in enumerate(ids)}
    models = list(matrix.index)
    print(f"[{args.benchmark}] dims={dims} criteria={len(ids)} models={len(models)} "
          f"min_scen={args.min_scenarios} SE={args.se_target}")

    # dense full-bank reference ability
    rgrid, rlp = scl.build_grid(len(dims), args.ref_nodes)
    Y = matrix[ids].to_numpy(dtype=float); mask = ~np.isnan(Y)
    theta_ref = scl.eap_all_models(np.nan_to_num(Y), mask, A, b, rgrid, rlp)   # (models, dims)

    sgrid, slp = scl.build_grid(len(dims), args.stop_nodes)
    spec = scl.RunSpec(seed=args.seed, max_se=1e-9, min_evals_per_skill=0,
                       min_scenarios=0, max_scenarios=args.cap, selection="trace",
                       mode="cat", write_logs=False)

    rows = []
    for mi, model in enumerate(models):
        row = matrix.loc[model]
        rec = scl.run_one_model(model, rubrics, scen_raw, row, dims, spec)
        order = [c for c in rec["order"] if c in col]
        if not order:
            continue
        Yrow = np.nan_to_num(row.reindex(ids).to_numpy(dtype=float))
        bnds = trajectory(order, rubrics, Yrow, col, A, b, dims, sgrid, slp, scen_of)
        on, on_reached = first_stop(bnds, "online_se", args.se_target, args.min_scenarios)
        ep, ep_reached = first_stop(bnds, "eap_sd", args.se_target, args.min_scenarios)
        rows.append({
            "model": model,
            "len_online": on["nscen"], "len_eap": ep["nscen"],
            # honest EAP SD at each rule's stop, per skill
            **{f"honestSD_online_{d}": float(on["eap_sd"][k]) for k, d in enumerate(dims)},
            **{f"honestSD_eap_{d}": float(ep["eap_sd"][k]) for k, d in enumerate(dims)},
            "online_honest_ok": bool(np.all(on["eap_sd"] <= args.se_target)),
            "eap_reached": bool(ep_reached),
            **{f"theta_online_{d}": float(on["theta"][k]) for k, d in enumerate(dims)},
            **{f"theta_eap_{d}": float(ep["theta"][k]) for k, d in enumerate(dims)},
            **{f"theta_ref_{d}": float(theta_ref[mi][k]) for k, d in enumerate(dims)},
        })
    if not rows:
        print("no rows"); return 1

    with (args.out_dir / "per_model.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    def agg(rule):
        L = np.array([r[f"len_{rule}"] for r in rows])
        out = {"mean_len": float(L.mean()), "median_len": float(np.median(L))}
        for k, d in enumerate(dims):
            honest = np.array([r[f"honestSD_{rule}_{d}"] for r in rows])
            out[f"pct_reach_{d}"] = float(np.mean(honest <= args.se_target))
            rr, ss, mae = recov([r[f"theta_ref_{d}"] for r in rows],
                                 [r[f"theta_{rule}_{d}"] for r in rows])
            out[f"recovery_r_{d}"] = round(rr, 4); out[f"slope_{d}"] = round(ss, 4)
            out[f"theta_mae_{d}"] = round(mae, 4)
        # overall honest convergence: ALL skills <= target
        allok = np.array([all(r[f"honestSD_{rule}_{d}"] <= args.se_target for d in dims) for r in rows])
        out["pct_reach_all_skills"] = float(np.mean(allok))
        return out

    summary = {"benchmark": args.benchmark, "dims": dims, "n_models": len(rows),
               "min_scenarios": args.min_scenarios, "se_target": args.se_target,
               "cap": args.cap, "ref_nodes": args.ref_nodes, "stop_nodes": args.stop_nodes,
               "bank_stats": stats, "online": agg("online"), "eap": agg("eap")}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    o, e = summary["online"], summary["eap"]
    print(f"\n[{args.benchmark}]  online -> eap   (target SE {args.se_target}, floor {args.min_scenarios})")
    print(f"  median length : {o['median_len']:.0f} -> {e['median_len']:.0f}   "
          f"mean {o['mean_len']:.1f} -> {e['mean_len']:.1f}")
    print(f"  %% reach target (all skills): {o['pct_reach_all_skills']:.1%} -> {e['pct_reach_all_skills']:.1%}")
    for d in dims:
        print(f"  {d:24} %reach {o[f'pct_reach_{d}']:.1%}->{e[f'pct_reach_{d}']:.1%}  "
              f"recovery r {o[f'recovery_r_{d}']:.3f}->{e[f'recovery_r_{d}']:.3f}  "
              f"theta-MAE {o[f'theta_mae_{d}']:.3f}->{e[f'theta_mae_{d}']:.3f}")
    print(f"wrote -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
