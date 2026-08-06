"""Phase A: BiGGen recovery x (floor x SE) operating-point grid UNDER THE EAP-POSTERIOR STOP.

Mirrors the WildBench ``scenario_recovery_grid.py`` (EAP path): for a rectangular grid of
operating points (min_scenarios floor x SE target) it reports, per cell,

  * OOS k-fold (k=5) theta-recovery r (+bootstrap CI), slope, theta-MAE, and pass-rate recovery
    (held-out models, MWLE vs the full-bank FINE-EAP reference -- BiGGen's of-record reference);
  * DEPLOYED (full-data frozen-bank) mean/median test length + convergence + median SE_total,
    where SE_total = sqrt(SE_posterior^2 + SE_param^2), SE_param from the observed-information
    parametric bootstrap over the administered set (exp-07 machinery).

Stop rule is selectable via ``--stop-se {online,eap}`` (default ``online`` = the deployed rule,
kept until adoption). Under ``eap`` (this Phase A study) EVERY cell is derived POST-HOC from ONE
forced-long adaptive order per fold/full-data run: at each scenario boundary the EAP posterior
SD is integrated over the fine theta grid and the test stops at the first scenario with
n_scenarios >= floor AND posterior SD <= SE target, else the forced cap. The production engine +
scenario selection are UNCHANGED; only the stop differs.

Convention note (mirrors the BiGGen README 9.4-vs-13.6 reconciliation): ``recovery_r`` is the
OOS k-fold measure (fold-refit params); ``scenarios_mean`` / ``se_total_median`` are the DEPLOYED
frozen-bank measures. The two conventions differ by design and are NOT comparable across panels.

Writes to ``experiments/12_eap_stop_grid/`` (a NEW, clearly-labelled prototype dir). It does NOT
overwrite the online-SE of-record grid (exp-06 / 06b_operating_point) or any of-record file.
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


def op_points(floors, se_list):
    return [{"name": f"f{fl}_se{se:.2f}", "floor": fl, "se": se}
            for fl in floors for se in se_list]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    base = ROOT / "biggen_calibration"
    p.add_argument("--bank", type=Path, default=ROOT / "staging" / "biggen_unidim_modeled.jsonl")
    p.add_argument("--matrix", type=Path, default=ROOT / "staging" / "biggen_response_matrix.csv")
    p.add_argument("--scenarios", type=Path, default=ROOT / "data" / "BiGGen" / "scenarios.jsonl")
    p.add_argument("--out-dir", type=Path, default=base / "experiments" / "12_eap_stop_grid")
    p.add_argument("--tmp-dir", type=Path, default=ROOT / "staging" / "_biggen_eap_grid")
    p.add_argument("--dim", default="general")
    p.add_argument("--floors", type=str, default="6,8,10,12,15")
    p.add_argument("--se-list", type=str, default="0.10,0.11,0.12,0.13,0.14,0.15")
    p.add_argument("--stop-se", choices=["online", "eap"], default="online",
                   help="online = deployed rule (per-cell engine); eap = post-hoc EAP posterior "
                        "SD stop (one forced-long order, all cells derived). Phase A uses eap.")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260729)
    p.add_argument("--fit-grid", type=int, default=7)
    p.add_argument("--eap-grid", type=int, default=3201, help="fine EAP nodes (ref + stop walk).")
    p.add_argument("--range", type=float, default=8.0)
    p.add_argument("--se-param-grid", type=int, default=241,
                   help="EAP nodes for the SE_param bootstrap (coarser; resolves SE_param).")
    p.add_argument("--se-param-range", type=float, default=6.0)
    p.add_argument("--ridge", type=float, default=1e-2)
    p.add_argument("--l-max", type=int, default=E.EAP_L_MAX)
    p.add_argument("--max-scenarios", type=int, default=50)
    p.add_argument("--min-evals-per-skill", type=int, default=15)
    p.add_argument("--n-boot", type=int, default=150)
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--pu-seed", type=int, default=20260801)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--from-csv", action="store_true",
                   help="regenerate recommendation + figures from recovery_grid.csv (no re-run).")
    args = p.parse_args()
    (args.out_dir / "figures").mkdir(parents=True, exist_ok=True)
    args.tmp_dir.mkdir(parents=True, exist_ok=True)
    d = args.dim

    floors = [int(x) for x in args.floors.split(",") if x.strip()]
    se_list = [round(float(x), 2) for x in args.se_list.split(",") if x.strip()]
    pts = op_points(floors, se_list)

    if args.from_csv:
        rows = pd.read_csv(args.out_dir / "recovery_grid.csv").to_dict("records")
        _figures(rows, floors, se_list, args.out_dir / "figures", args.stop_se)
        _write_recommendation(rows, args)
        print("(--from-csv) regenerated recommendation + figures")
        return 0

    records, dims, _ = scat.load_fitted_bank(args.bank, "clamp")
    assert len(dims) == 1, "BiGGen grid is 1-D"
    ids = [r["criterion_id"] for r in records]
    scen_of = {r["criterion_id"]: r["scenario_id"] for r in records}
    matrix = pd.read_csv(args.matrix, index_col=0)
    Yraw = matrix.reindex(columns=ids).to_numpy(float)
    Mall = ~np.isnan(Yraw)
    models = list(matrix.index)
    row_of = {m: i for i, m in enumerate(models)}
    col_full = {c: i for i, c in enumerate(ids)}
    Q_all = np.ones((len(ids), 1), dtype=int)
    egrid, elog = scat.build_grid(1, args.eap_grid, args.range)
    gg, lp = E.eap_grid(args.eap_grid, args.range)

    print("=" * 92)
    print(f"BiGGen recovery x (floor x SE) grid | stop-se={args.stop_se} | "
          f"floors={floors} SE={se_list}")
    print("=" * 92)
    print(f"bank={args.bank.name} models={len(models)} admin_criteria={len(ids)} "
          f"eap_grid={args.eap_grid} k={args.k} seed={args.seed}")

    # --- SE_param machinery (full-data observed info) ---
    cov = rg.compute_item_cov(args.bank, args.matrix, args.fit_grid, args.ridge)
    bgrid = np.linspace(-args.se_param_range, args.se_param_range, args.se_param_grid)
    blp = -0.5 * bgrid ** 2
    blp = blp - logsumexp(blp)
    rng_boot = np.random.default_rng(args.pu_seed)

    # --- fit the k folds ONCE (op-point-independent) ---
    print("\nfitting the 5 CV folds once (reused across op points) ...", flush=True)
    folds = E.make_folds(models, args.k, args.seed)
    fold_data = []
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
        theta_ref = scat.eap_all_models(Yte, Mte, Ak, bk, egrid, elog)
        obs_pass = np.array([Yte[i][Mte[i]].mean() for i in range(len(test))])
        fold_data.append({"test": test, "bank": fold_bank, "kept_ids": kept_ids, "colk": colk,
                          "Ak": Ak, "bk": bk, "Yte": Yte, "theta_ref": theta_ref,
                          "obs_pass": obs_pass})
        print(f"  fold {f}: train={len(train)} test={len(test)} kept={len(kept_ids)}", flush=True)

    # --- EAP mode: precompute forced-long orders ONCE (full data + per fold) ---
    full_walks, fold_walks = {}, {}
    if args.stop_se == "eap":
        print(f"\nforced-long (L={args.l_max}) full-data order for post-hoc EAP cells ...", flush=True)
        specL = scat.RunSpec(seed=args.seed, top_n=args.top_n, max_se=0.0, min_evals_per_skill=0,
                             min_scenarios=args.l_max, max_scenarios=args.l_max,
                             selection="trace", mode="cat", runs_dir=str(args.tmp_dir / "fullL"))
        fullL = scat.run_models(models, args.bank, args.matrix, args.scenarios, "clamp", dims,
                                specL, workers=args.workers)
        a_full, b_full = cov["A"][:, 0], cov["b"]
        for r in fullL:
            m = r["model"]
            full_walks[m] = E.eap_walk(r["order"], Yraw[row_of[m]], col_full, scen_of,
                                       a_full, b_full, gg, lp)
        for f, fd in enumerate(fold_data):
            specLf = scat.RunSpec(seed=args.seed, top_n=args.top_n, max_se=0.0,
                                  min_evals_per_skill=0, min_scenarios=args.l_max,
                                  max_scenarios=args.l_max, selection="trace", mode="cat",
                                  runs_dir=str(args.tmp_dir / "foldL"))
            resL = scat.run_models(fd["test"], fd["bank"], args.matrix, args.scenarios, "clamp",
                                   dims, specLf, workers=args.workers)
            rb = {r["model"]: r for r in resL}
            fw = {}
            for ti, m in enumerate(fd["test"]):
                fw[m] = E.eap_walk(rb[m]["order"], fd["Yte"][ti], fd["colk"], scen_of,
                                   fd["Ak"][:, 0], fd["bk"], gg, lp)
            fold_walks[f] = fw
        print("  forced-long orders ready; deriving cells post-hoc.", flush=True)

    rows = []
    for pt in pts:
        floor, se = pt["floor"], pt["se"]
        # (A) DEPLOYED full-data admin: achieved ability SE + admin sets + length + SE_param
        if args.stop_se == "eap":
            scen_admin, crit_admin = [], []
            n_capped = n_conv = 0
            se_post_list, se_param_list, se_total_list = [], [], []
            for m in models:
                step, reached = E.eap_stop_point(full_walks[m], floor, se)
                scen_admin.append(step["n_scen"]); crit_admin.append(step["n_crit"])
                n_conv += int(reached); n_capped += int(not reached)
                _, se_post, se_param = rg.bootstrap_theta(
                    cov["Ymat"][row_of[m]], step["idx"], cov["beta"], cov["chol"],
                    bgrid, blp, args.n_boot, rng_boot)
                if np.isnan(se_post):
                    continue
                se_post_list.append(se_post); se_param_list.append(se_param)
                se_total_list.append(float(np.sqrt(se_post ** 2 + se_param ** 2)))
            scen_admin = np.array(scen_admin); crit_admin = np.array(crit_admin)
        else:
            spec = scat.RunSpec(seed=args.seed, top_n=args.top_n, max_se=se,
                                min_evals_per_skill=args.min_evals_per_skill,
                                min_scenarios=floor, max_scenarios=args.max_scenarios,
                                selection="trace", mode="cat",
                                runs_dir=str(args.tmp_dir / "full_runs"))
            full = scat.run_models(models, args.bank, args.matrix, args.scenarios,
                                   "clamp", dims, spec, workers=args.workers)
            scen_admin = np.array([r["scenarios_administered"] for r in full])
            crit_admin = np.array([r["criteria_administered"] for r in full])
            n_capped = sum(1 for r in full if r["stop_reason"] == "max_scenarios_reached")
            n_conv = sum(1 for r in full if r["precision_reached"])
            se_post_list, se_param_list, se_total_list = [], [], []
            for r0 in full:
                y = cov["Ymat"][row_of[r0["model"]]]
                idx = np.array([col_full[c] for c in r0["order"] if c in col_full], dtype=int)
                _, se_post, se_param = rg.bootstrap_theta(y, idx, cov["beta"], cov["chol"],
                                                          bgrid, blp, args.n_boot, rng_boot)
                if np.isnan(se_post):
                    continue
                se_post_list.append(se_post); se_param_list.append(se_param)
                se_total_list.append(float(np.sqrt(se_post ** 2 + se_param ** 2)))

        # (B) OOS recovery over folds: MWLE theta vs fine-EAP reference + pass-rate
        ref_all, cat_all, pred_pass_all, obs_pass_all = [], [], [], []
        for f, fd in enumerate(fold_data):
            Ak, bk, colk = fd["Ak"], fd["bk"], fd["colk"]
            if args.stop_se == "online":
                spec_f = scat.RunSpec(seed=args.seed, top_n=args.top_n, max_se=se,
                                      min_evals_per_skill=args.min_evals_per_skill,
                                      min_scenarios=floor, max_scenarios=args.max_scenarios,
                                      selection="trace", mode="cat",
                                      runs_dir=str(args.tmp_dir / "fold_runs"))
                res_by = {r["model"]: r for r in scat.run_models(
                    fd["test"], fd["bank"], args.matrix, args.scenarios, "clamp", dims,
                    spec_f, workers=args.workers)}
            for ti, m in enumerate(fd["test"]):
                y = fd["Yte"][ti]
                if args.stop_se == "eap":
                    step, _ = E.eap_stop_point(fold_walks[f][m], floor, se)
                    idx = step["idx"] if step is not None else np.array([], dtype=int)
                else:
                    idx = np.array([colk[c] for c in res_by[m]["order"] if c in colk], dtype=int)
                if idx.size == 0:
                    th_cat = fd["theta_ref"][ti].copy()
                else:
                    th_ba = scat.eap_subset(y, idx, Ak, bk, egrid, elog)
                    th_cat, _ = scat.mwle_subset(y, idx, Ak, bk, th_ba)
                ref_all.append(float(fd["theta_ref"][ti][0]))
                cat_all.append(float(th_cat[0]))
                pred_pass_all.append(float(expit(Ak[:, 0] * th_cat[0] - bk).mean()))
                obs_pass_all.append(float(fd["obs_pass"][ti]))

        ref_arr = np.array(ref_all); cat_arr = np.array(cat_all)
        band = scat.ols_ci_band(ref_arr, cat_arr, np.linspace(ref_arr.min(), ref_arr.max(), 40),
                                B=2000, seed=0)
        theta_mae = float(np.mean(np.abs(cat_arr - ref_arr)))
        op = np.array(obs_pass_all); pp = np.array(pred_pass_all)
        pass_r = float(np.corrcoef(op, pp)[0, 1])
        pass_mae = float(np.mean(np.abs(pp - op)))

        row = {
            "op_point": pt["name"], "min_scenarios": floor, "se_target": se,
            "scenarios_mean_deployed": round(float(scen_admin.mean()), 2),
            "scenarios_median_deployed": int(np.median(scen_admin)),
            "scenarios_max_deployed": int(scen_admin.max()),
            "criteria_mean_deployed": round(float(crit_admin.mean()), 1),
            "convergence_rate_deployed": round(n_conv / len(models), 3),
            "capped_frac_deployed": round(n_capped / len(models), 3),
            "recovery_r": round(band["r"], 4),
            "recovery_r_lo": round(band["r_lo"], 4),
            "recovery_r_hi": round(band["r_hi"], 4),
            "recovery_slope": round(band["slope"], 4),
            "theta_mae": round(theta_mae, 4),
            "pass_r": round(pass_r, 4),
            "pass_mae": round(pass_mae, 4),
            "ability_se_median_deployed": round(float(np.median(se_post_list)), 4),
            "se_param_median_deployed": round(float(np.median(se_param_list)), 4),
            "se_total_median_deployed": round(float(np.median(se_total_list)), 4),
        }
        rows.append(row)
        print(f"  {pt['name']:12s}: scen_dep={row['scenarios_mean_deployed']:5.1f} "
              f"cap={row['capped_frac_deployed']:.2f} r={row['recovery_r']:.3f} "
              f"MAE={row['theta_mae']:.3f} passR={row['pass_r']:.3f} "
              f"abilSE={row['ability_se_median_deployed']:.3f} "
              f"SEtot={row['se_total_median_deployed']:.3f}", flush=True)

    with (args.out_dir / "recovery_grid.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    _figures(rows, floors, se_list, args.out_dir / "figures", args.stop_se)
    _write_recommendation(rows, args)
    for sub_ in ("fullL", "foldL", "full_runs", "fold_runs"):
        shutil.rmtree(args.tmp_dir / sub_, ignore_errors=True)
    print(f"\nwrote -> {args.out_dir / 'recovery_grid.csv'}")
    print(f"wrote -> {args.out_dir / 'figures' / 'heatmaps_floor_x_se.png'}")
    return 0


def _write_recommendation(rows, args):
    df = pd.DataFrame(rows)
    R_TARGET = 0.95
    keys = ("op_point", "min_scenarios", "se_target", "scenarios_mean_deployed",
            "scenarios_max_deployed", "convergence_rate_deployed", "capped_frac_deployed",
            "recovery_r", "recovery_r_lo", "recovery_r_hi", "recovery_slope", "theta_mae",
            "pass_r", "ability_se_median_deployed", "se_param_median_deployed",
            "se_total_median_deployed")
    CONV_MIN = 0.90  # ~6-12% of the lowest-ability tiny base models inherently cap at tight SE
    deploy = df[(df["recovery_r"] >= R_TARGET) & (df["convergence_rate_deployed"] >= CONV_MIN)]
    deploy = deploy.sort_values(["scenarios_mean_deployed", "se_total_median_deployed"])
    rec = (deploy.iloc[0].to_dict() if not deploy.empty
           else df.sort_values("recovery_r").iloc[-1].to_dict())
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "Phase A EAP-stop grid -- for the user's op-point decision. NOT of-record.",
        "stop_rule": args.stop_se,
        "recovery_cv": f"OOS k={args.k} MODEL folds; seed {args.seed}; fine uniform-EAP reference "
                       f"({args.eap_grid} nodes over +/-{args.range}); MWLE theta.",
        "se_param_source": "op-point-specific observed-info bootstrap over the administered set; "
                           "SE_total = sqrt(ability_post^2 + SE_param^2).",
        "convention_note": "recovery_r is OOS k-fold (fold-refit params); scenarios_*/se_total_* "
                           "are DEPLOYED frozen-bank. Not comparable across the two conventions.",
        "r_target_for_deployment": R_TARGET,
        "convergence_floor_for_deployment": CONV_MIN,
        "grid_shape": {"floors": sorted(df["min_scenarios"].unique().tolist()),
                       "se_targets": sorted(df["se_target"].unique().tolist())},
        "recommended_provisional": {k: rec[k] for k in keys},
        "cells_reaching_r_target_and_converged": [{k: r[k] for k in keys}
                                                  for r in deploy.to_dict("records")],
        "recommendation_rationale": (
            "Among cells with OOS r>=0.95 AND deployed convergence>=99%, the shortest (fewest "
            "mean deployed scenarios, then smallest SE_total) is the most efficient. FINAL choice "
            "is the user's; this grid is decision-support only."),
    }
    (args.out_dir / "recommendation.json").write_text(json.dumps(summary, indent=2),
                                                      encoding="utf-8")
    print("\nrecommended (provisional):",
          json.dumps(summary["recommended_provisional"], indent=2))


def _pivot(df, value, floors, se_list):
    M = np.full((len(floors), len(se_list)), np.nan)
    fi = {f: i for i, f in enumerate(floors)}
    si = {s: j for j, s in enumerate(se_list)}
    for _, r in df.iterrows():
        M[fi[int(r["min_scenarios"])], si[round(float(r["se_target"]), 2)]] = r[value]
    return M


def _heatmap(ax, M, floors, se_list, title, cmap, fmt):
    im = ax.imshow(M, aspect="auto", cmap=cmap, origin="lower")
    ax.set_xticks(range(len(se_list))); ax.set_xticklabels([f"{s:.2f}" for s in se_list], fontsize=8)
    ax.set_yticks(range(len(floors))); ax.set_yticklabels([str(f) for f in floors], fontsize=8)
    ax.set_xlabel("SE target"); ax.set_ylabel("min_scenarios (floor)")
    ax.set_title(title, fontsize=10)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=6.5)
    return im


def _figures(rows, floors, se_list, fig_dir: Path, stop_se: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    floors = sorted(set(int(x) for x in floors))
    se_list = sorted(set(round(float(x), 2) for x in se_list))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    im0 = _heatmap(axes[0], _pivot(df, "recovery_r", floors, se_list), floors, se_list,
                   "OOS recovery r  (OOS k-fold, fold-refit params)", "viridis", "{:.3f}")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = _heatmap(axes[1], _pivot(df, "se_total_median_deployed", floors, se_list), floors,
                   se_list, "median SE_total  (DEPLOYED frozen-bank)", "magma_r", "{:.3f}")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    im2 = _heatmap(axes[2], _pivot(df, "scenarios_mean_deployed", floors, se_list), floors,
                   se_list, "mean scenarios administered  (DEPLOYED)", "cividis", "{:.1f}")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)
    fig.suptitle(f"BiGGen recovery x (floor x SE) operating-point grid  [stop-se = {stop_se}]",
                 fontsize=13)
    fig.text(0.5, 0.005,
             "REGIME SPLIT (per the README 9.4-vs-13.6 reconciliation): recovery r is the OOS "
             "k-fold measure (fold-refit params); SE_total and test length are the DEPLOYED "
             "frozen-bank measures. The two conventions differ by design and are NOT directly "
             "comparable across panels.",
             ha="center", va="bottom", fontsize=8, wrap=True)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(fig_dir / "heatmaps_floor_x_se.png", dpi=140)
    plt.close(fig)

    def line_family(value, ylabel, fname, hline=None, fmt_int=False):
        fig, ax = plt.subplots(figsize=(9, 5))
        for fl in floors:
            dd = df[df["min_scenarios"] == fl].sort_values("se_target")
            ax.plot(dd["se_target"], dd[value], "o-", label=f"floor={fl}", alpha=0.85)
        if hline is not None:
            ax.axhline(hline, ls="--", color="gray", alpha=0.7)
        ax.set_xlabel("SE target"); ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs SE target (line per floor)  [stop-se = {stop_se}]")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout(); fig.savefig(fig_dir / fname, dpi=140); plt.close(fig)

    line_family("recovery_r", "OOS recovery r", "recovery_vs_op_point.png", hline=0.95)
    line_family("se_total_median_deployed", "median SE_total (deployed)", "total_se_vs_op_point.png")
    line_family("scenarios_mean_deployed", "mean scenarios (deployed)", "length_vs_op_point.png")


if __name__ == "__main__":
    raise SystemExit(main())
