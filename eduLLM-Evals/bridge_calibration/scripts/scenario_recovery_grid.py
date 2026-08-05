"""Phase 2a steps 3-4: recovery x operating-point grid (for finalizing the op point).

For a grid of operating points -- floor=0 swept across SE targets {0.15..0.40} (length
controlled by the SE target) plus floors {12,15,20} (SE target moot; report achieved SE) --
this reports, per operating point:
  * OOS k-fold (k=5) theta-recovery r (+bootstrap CI), theta-MAE, slope,
  * pass-rate recovery r + pass-MAE (predicted P(pass|theta_cat) vs observed pass rate),
  * mean test length (scenarios AND criteria),
  * achieved ability SE (posterior), SE_param at that op point, and SE_total = ability (+) SE_param.

CAT administration + theta scoring use the MASKED CAT-pool bank (A3 + extreme_a excluded).
Recovery folds are over MODELS (persons), mirroring the reference
``scripts/scenario_kfold_estimator_cv.py`` (the leakage-safe source->fold SCENARIO grouping
is recorded in build_manifest.json for scenario-holdout analyses; the recovery measure here
is per-model, so source grouping does not change the model folds). SE_param at each op point
is the observed-info parametric bootstrap over that op point's administered set (step 2 math).

Fold parameter fits are OP-POINT-INDEPENDENT, so the 5 folds are fit ONCE and reused across
all operating points; only the engine's administered set changes per op point.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

import importlib.util

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import bridge_scenario_lib as L  # noqa: E402

# Load the Bridge step-2 module by explicit path (its filename collides with the reference
# scripts/scenario_param_uncertainty.py that bridge_scenario_lib puts on sys.path first).
_pu_spec = importlib.util.spec_from_file_location(
    "bridge_scenario_param_uncertainty", _HERE / "scenario_param_uncertainty.py")
PU = importlib.util.module_from_spec(_pu_spec)
sys.modules["bridge_scenario_param_uncertainty"] = PU
_pu_spec.loader.exec_module(PU)

ROOT = L.ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import scripts.scenario_cat_lib as scat  # noqa: E402


def make_folds(models, k, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(models))
    folds = [[] for _ in range(k)]
    for pos, i in enumerate(idx):
        folds[pos % k].append(models[int(i)])
    return [sorted(f) for f in folds]


def op_points(floors, se_list):
    """Full cross-product of floors x SE targets."""
    return [{"name": f"f{fl}_se{se:.2f}", "floor": fl, "se": se}
            for fl in floors for se in se_list]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    base = ROOT / "bridge_calibration"
    p.add_argument("--bank", type=Path, default=base / "bridge_scenario_fitted_1d_catpool.jsonl")
    p.add_argument("--matrix", type=Path, default=ROOT / "bridgegrade" / "response_matrix.csv")
    p.add_argument("--scenarios", type=Path, default=ROOT / "data" / "Bridge" / "scenarios.jsonl")
    p.add_argument("--out-dir", type=Path, default=base / "experiments" / "06b_operating_point")
    p.add_argument("--tmp-dir", type=Path, default=base / "experiments" / "06b_operating_point" / "_tmp")
    p.add_argument("--floors", type=str, default="0,4,6,8,12,15,20")
    p.add_argument("--se-list", type=str, default="0.08,0.10,0.12,0.15,0.20,0.25,0.30")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260729)
    p.add_argument("--fit-grid", type=int, default=7)
    p.add_argument("--eap-grid", type=int, default=61)
    p.add_argument("--range", type=float, default=6.0)
    p.add_argument("--ridge", type=float, default=1e-2)
    p.add_argument("--max-scenarios", type=int, default=60,
                   help="cap on test length (scenarios) so tight/unachievable SE targets "
                        "terminate; cells hitting it are reported as capped (SE not achieved).")
    p.add_argument("--min-evals-per-skill", type=int, default=15)
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--from-csv", action="store_true",
                   help="regenerate recommendation.json + figures from recovery_grid.csv "
                        "(no engine/CV re-run).")
    args = p.parse_args()
    (args.out_dir / "figures").mkdir(parents=True, exist_ok=True)
    args.tmp_dir.mkdir(parents=True, exist_ok=True)

    cm = L.cm
    records, dims, _ = scat.load_fitted_bank(args.bank, "clamp")
    assert len(dims) == 1, "Phase 2a locked to 1D"
    d = dims[0]
    ids = [r["criterion_id"] for r in records]
    scen_of = {r["criterion_id"]: r["scenario_id"] for r in records}
    matrix = pd.read_csv(args.matrix, index_col=0)
    sub = matrix.reindex(columns=ids)
    Yraw = sub.to_numpy(float)
    Mall = ~np.isnan(Yraw)
    models = list(matrix.index)
    row_of = {m: i for i, m in enumerate(models)}
    col_full = {c: i for i, c in enumerate(ids)}
    Q_all = np.ones((len(ids), 1), dtype=int)

    se_list = [float(x) for x in args.se_list.split(",") if x.strip()]
    floors = [int(x) for x in args.floors.split(",") if x.strip()]
    pts = op_points(floors, se_list)

    if args.from_csv:
        rows = pd.read_csv(args.out_dir / "recovery_grid.csv").to_dict("records")
        print("(--from-csv: reusing recovery_grid.csv)")
        _figures(rows, floors, se_list, d, args.out_dir / "figures")
        _write_recommendation(rows, models, args)
        print(f"regenerated recommendation + figures -> {args.out_dir}")
        return 0

    egrid, elog = scat.build_grid(1, args.eap_grid, args.range)

    print("=" * 90)
    print("STEP 3-4: recovery x operating-point grid")
    print("=" * 90)
    print(f"bank={args.bank.name} dims={dims} models={len(models)} admin_criteria={len(ids)}")
    print(f"operating points: {[pt['name'] for pt in pts]}")

    # --- SE_param machinery (full-data observed info; op-point-specific admin sets) ---
    covdata = PU.compute_item_cov(args.bank, args.matrix, args.fit_grid, args.ridge)
    bgrid = np.linspace(-args.range, args.range, args.eap_grid)
    blp = -0.5 * bgrid ** 2
    from scipy.special import logsumexp
    blp = blp - logsumexp(blp)
    rng_boot = np.random.default_rng(20260801)

    # --- fit the k folds ONCE (op-point-independent) ---
    print("\nfitting the 5 CV folds once (reused across op points) ...", flush=True)
    folds = make_folds(models, args.k, args.seed)
    fold_data = []
    for f in range(args.k):
        test = folds[f]
        train = [m for m in models if m not in set(test)]
        tr_idx = [row_of[m] for m in train]
        Ytr, Mtr = Yraw[tr_idx], Mall[tr_idx]
        keep = []
        for j in range(len(ids)):
            obs = Mtr[:, j]
            if obs.sum() >= 2:
                vals = Ytr[obs, j]
                if 0 < vals.sum() < obs.sum():
                    keep.append(j)
        keep = np.array(keep, dtype=int)
        Ytr_k = np.nan_to_num(Ytr[:, keep], nan=0.0)
        fit = cm.fit_m2pl_em(Ytr_k, Mtr[:, keep], Q_all[keep], args.fit_grid,
                             ridge=args.ridge, max_iter=200)
        Ak, bk = fit["A"], fit["b"]
        kept_ids = [ids[j] for j in keep]
        colk = {c: i for i, c in enumerate(kept_ids)}
        fold_bank = args.tmp_dir / f"fold{f}_bank.jsonl"
        with fold_bank.open("w", encoding="utf-8") as fh:
            for jj, cid in enumerate(kept_ids):
                fh.write(json.dumps({
                    "criterion_id": cid, "scenario_id": scen_of[cid], "criterion": "",
                    "discrimination": {d: float(Ak[jj, 0])},
                    "q_modeled": {d: 1}, "difficulty": float(bk[jj])}) + "\n")
        te_idx = [row_of[m] for m in test]
        subte = matrix.loc[test].reindex(columns=kept_ids)
        Yte = np.nan_to_num(subte.to_numpy(float), nan=0.0)
        Mte = ~np.isnan(subte.to_numpy(float))
        theta_ref = scat.eap_all_models(Yte, Mte, Ak, bk, egrid, elog)  # (n_test,1)
        obs_pass = np.array([Yte[i][Mte[i]].mean() for i in range(len(test))])
        fold_data.append({"test": test, "bank": fold_bank, "kept_ids": kept_ids,
                          "colk": colk, "Ak": Ak, "bk": bk, "Yte": Yte,
                          "theta_ref": theta_ref, "obs_pass": obs_pass})
        print(f"  fold {f}: train={len(train)} test={len(test)} kept={len(kept_ids)} "
              f"(loglik={fit['loglik']:.0f})", flush=True)

    # --- per operating point ---
    rows = []
    for pt in pts:
        floor, se = pt["floor"], pt["se"]
        # (A) full-data engine run: achieved ability SE + admin sets + length + SE_param
        spec = scat.RunSpec(seed=42, top_n=5, max_se=se,
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
            m = r0["model"]
            y = Yraw[row_of[m]]
            idx = np.array([col_full[c] for c in r0["order"] if c in col_full], dtype=int)
            _, se_post, se_param = PU.bootstrap_theta(
                y, idx, covdata["beta"], covdata["chol"], bgrid, blp,
                args.n_boot, rng_boot)
            if np.isfinite(se_post):
                se_post_list.append(se_post)
                se_param_list.append(se_param)
                se_total_list.append(float(np.sqrt(se_post ** 2 + se_param ** 2)))

        # (B) OOS k-fold recovery (reuse fold fits; engine per fold at this op point)
        ref_all, cat_all, obs_pass_all, pred_pass_all = [], [], [], []
        for f, fd in enumerate(fold_data):
            spec_f = scat.RunSpec(seed=args.seed, top_n=5, max_se=se,
                                  min_evals_per_skill=args.min_evals_per_skill,
                                  min_scenarios=floor, max_scenarios=args.max_scenarios,
                                  selection="trace", mode="cat",
                                  runs_dir=str(args.tmp_dir / f"runs_f{f}"))
            res = scat.run_models(fd["test"], fd["bank"], args.matrix, args.scenarios,
                                  "clamp", dims, spec_f, workers=args.workers)
            res_by = {r["model"]: r for r in res}
            Ak, bk, colk = fd["Ak"], fd["bk"], fd["colk"]
            for ti, m in enumerate(fd["test"]):
                r0 = res_by[m]
                y = fd["Yte"][ti]
                idx = np.array([colk[c] for c in r0["order"] if c in colk], dtype=int)
                if idx.size == 0:
                    th_cat = fd["theta_ref"][ti].copy()
                else:
                    th_ba = scat.eap_subset(y, idx, Ak, bk, egrid, elog)
                    th_cat, _ = scat.mwle_subset(y, idx, Ak, bk, th_ba)
                ref_all.append(float(fd["theta_ref"][ti][0]))
                cat_all.append(float(th_cat[0]))
                pred = float(expit(Ak[:, 0] * th_cat[0] - bk).mean())
                pred_pass_all.append(pred)
                obs_pass_all.append(float(fd["obs_pass"][ti]))

        ref_arr = np.array(ref_all); cat_arr = np.array(cat_all)
        xs = np.linspace(ref_arr.min(), ref_arr.max(), 40)
        band = scat.ols_ci_band(ref_arr, cat_arr, xs, B=2000, seed=0)
        theta_mae = float(np.mean(np.abs(cat_arr - ref_arr)))
        op = np.array(obs_pass_all); pp = np.array(pred_pass_all)
        pass_r = float(np.corrcoef(op, pp)[0, 1])
        pass_mae = float(np.mean(np.abs(pp - op)))

        row = {
            "op_point": pt["name"], "min_scenarios": floor, "se_target": se,
            "scenarios_mean": round(float(scen_admin.mean()), 2),
            "scenarios_median": int(np.median(scen_admin)),
            "criteria_mean": round(float(crit_admin.mean()), 1),
            "criteria_median": int(np.median(crit_admin)),
            "convergence_rate": round(n_conv / len(full), 3),
            "capped_frac": round(n_capped / len(full), 3),
            "recovery_r": round(band["r"], 4),
            "recovery_r_lo": round(band["r_lo"], 4),
            "recovery_r_hi": round(band["r_hi"], 4),
            "recovery_slope": round(band["slope"], 4),
            "theta_mae": round(theta_mae, 4),
            "pass_r": round(pass_r, 4),
            "pass_mae": round(pass_mae, 4),
            "ability_se_median": round(float(np.median(se_post_list)), 4),
            "se_param_median": round(float(np.median(se_param_list)), 4),
            "se_total_median": round(float(np.median(se_total_list)), 4),
        }
        rows.append(row)
        print(f"  {pt['name']:12s}: scen={row['scenarios_mean']:5.1f} crit={row['criteria_mean']:5.0f} "
              f"cap={row['capped_frac']:.2f} r={row['recovery_r']:.3f} MAE={row['theta_mae']:.3f} "
              f"passR={row['pass_r']:.3f} abilSE={row['ability_se_median']:.3f} "
              f"SEparam={row['se_param_median']:.3f} SEtot={row['se_total_median']:.3f}", flush=True)

    with (args.out_dir / "recovery_grid.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    _figures(rows, floors, se_list, d, args.out_dir / "figures")
    _write_recommendation(rows, models, args)
    # tidy temp run dirs (keep fold banks for provenance)
    for sub_ in ("full_runs",) + tuple(f"runs_f{f}" for f in range(args.k)):
        shutil.rmtree(args.tmp_dir / sub_, ignore_errors=True)
    print(f"wrote -> {args.out_dir}")
    return 0


def _write_recommendation(rows, models, args):
    """Efficiency-frontier recommendation across the full floor x SE grid.

    Deployment target r>=0.95. The recommendation is the SHORTEST test (min mean scenarios,
    then min SE_total) among all cells reaching r>=0.95 -- i.e. the most efficient operating
    point. We explicitly answer whether any low-floor + tight-SE cell beats floor=12 on
    scenarios at equal recovery. FINAL choice is the user's."""
    df = pd.DataFrame(rows)
    R_TARGET = 0.95
    keys = ("op_point", "min_scenarios", "se_target", "scenarios_mean", "criteria_mean",
            "capped_frac", "recovery_r", "recovery_r_lo", "recovery_r_hi", "theta_mae",
            "pass_r", "pass_mae", "ability_se_median", "se_param_median", "se_total_median")
    deploy = df[df["recovery_r"] >= R_TARGET].sort_values(
        ["scenarios_mean", "se_total_median"])
    if not deploy.empty:
        rec = deploy.iloc[0].to_dict()
    else:
        rec = df.sort_values("recovery_r").iloc[-1].to_dict()

    # efficiency-frontier question: shortest floor=12 cell vs shortest any-cell at r>=0.95
    f12 = df[(df["min_scenarios"] == 12) & (df["recovery_r"] >= R_TARGET)]
    f12_min_scen = float(f12["scenarios_mean"].min()) if not f12.empty else None
    low_floor = deploy[deploy["min_scenarios"] < 12] if not deploy.empty else deploy
    beats = (not low_floor.empty and f12_min_scen is not None
             and float(low_floor["scenarios_mean"].min()) < f12_min_scen)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "recovery_cv": "OOS k=5 MODEL folds (mirrors scenario_kfold_estimator_cv); "
                       "seed 20260729; source->fold scenario map retained in build_manifest.",
        "se_param_source": "op-point-specific observed-info bootstrap (step 2 math) over the "
                           "engine's administered set; SE_total = sqrt(ability_post^2 + SE_param^2).",
        "n_models": len(models), "k": args.k, "r_target_for_deployment": R_TARGET,
        "grid_shape": {"floors": sorted(df["min_scenarios"].unique().tolist()),
                       "se_targets": sorted(df["se_target"].unique().tolist())},
        "grid": rows,
        "recommended": {k: rec[k] for k in keys},
        "cells_reaching_r_target": [{k: r[k] for k in keys}
                                    for r in deploy.to_dict("records")],
        "efficiency_frontier_question": {
            "floor12_min_scenarios_at_r_target": f12_min_scen,
            "low_floor_beats_floor12": bool(beats),
            "note": ("Does a low-floor + tight-SE cell reach r>=0.95 at fewer mean scenarios "
                     "than floor=12? See 'low_floor_beats_floor12'. Because a Bridge scenario "
                     "is a heavy ~18-criterion testlet and ability SE floors near ~0.076, a "
                     "tight SE at a low floor tends to administer roughly the same number of "
                     "scenarios as a moderate floor -- the two levers largely coincide."),
        },
        "recommendation_rationale": (
            "Among all cells with OOS r>=0.95, choose the shortest (fewest mean scenarios, then "
            "smallest SE_total) = most efficient. FINAL choice is the user's."),
    }
    (args.out_dir / "recommendation.json").write_text(json.dumps(summary, indent=2),
                                                      encoding="utf-8")
    print("\nrecommended (provisional):", json.dumps(summary["recommended"], indent=2))
    print("low_floor beats floor12 on scenarios @ r>=0.95:", beats)


def _pivot(df, value, floors, se_list):
    import numpy as _np
    M = _np.full((len(floors), len(se_list)), _np.nan)
    fi = {f: i for i, f in enumerate(floors)}
    si = {s: j for j, s in enumerate(se_list)}
    for _, r in df.iterrows():
        M[fi[int(r["min_scenarios"])], si[round(float(r["se_target"]), 2)]] = r[value]
    return M


def _heatmap(ax, M, floors, se_list, title, cmap, fmt="{:.2f}"):
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


def _figures(rows, floors, se_list, d, fig_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    floors = sorted(set(int(x) for x in floors))
    se_list = sorted(set(round(float(x), 2) for x in se_list))

    # --- heatmaps over (floor x SE): recovery r, SE_total, mean scenarios ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    im0 = _heatmap(axes[0], _pivot(df, "recovery_r", floors, se_list), floors, se_list,
                   "OOS recovery r", "viridis", "{:.3f}")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = _heatmap(axes[1], _pivot(df, "se_total_median", floors, se_list), floors, se_list,
                   "median SE_total", "magma_r", "{:.3f}")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    im2 = _heatmap(axes[2], _pivot(df, "scenarios_mean", floors, se_list), floors, se_list,
                   "mean scenarios administered", "cividis", "{:.1f}")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)
    fig.suptitle("Recovery / SE_total / length over (floor x SE)", fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_dir / "heatmaps_floor_x_se.png", dpi=140); plt.close(fig)

    # --- line families: one line per floor, x = SE target ---
    def line_family(value, ylabel, fname, extra=None):
        fig, ax = plt.subplots(figsize=(9, 5))
        for fl in floors:
            dd = df[df["min_scenarios"] == fl].sort_values("se_target")
            ax.plot(dd["se_target"], dd[value], "o-", label=f"floor={fl}", alpha=0.85)
        if extra is not None:
            extra(ax)
        ax.set_xlabel("SE target"); ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs SE target (line per floor)")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout(); fig.savefig(fig_dir / fname, dpi=140); plt.close(fig)

    line_family("recovery_r", "OOS recovery r", "recovery_vs_op_point.png",
                extra=lambda ax: ax.axhline(0.95, ls="--", color="gray", alpha=0.7))
    line_family("se_total_median", "median SE_total", "total_se_vs_op_point.png")
    line_family("scenarios_mean", "mean scenarios administered", "length_vs_op_point.png")


if __name__ == "__main__":
    raise SystemExit(main())
