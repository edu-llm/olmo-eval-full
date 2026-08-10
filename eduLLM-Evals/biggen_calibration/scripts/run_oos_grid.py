"""Phase-2 Step B: BiGGen gemini-3 CURATED bank floor x SE out-of-sample operating-point grid.

LOCAL study (read-only w.r.t. production). Mirrors the validated InfoBench / WildBench OOS design
(reports/infobench_oos_grid_unidim/run_oos_grid.py) but on the BiGGen gemini-3 CURATED
unidimensional ``general`` scale, using the EXACT BiGGen PRODUCTION calibration config:

  * per-fold refit = calibrate_mirt.fit_m2pl_em (Bock-Aitkin MML-EM), 7 GH nodes, ridge 0.01,
    max_iter 200, clamp negative discriminations at engine load (production negative_policy)
  * k=5 MODEL folds, seed 20260729; within each fold the SAME pass-imbalance exclusion is
    re-applied on the TRAINING models only (observed pass_count<=3 OR fail_count<=3)
  * production scenario/testlet CAT engine (scenario_cat_lib.run_models, selection="trace")
    for the adaptive administration ORDER (one forced-long run per fold), then a POST-HOC dense
    EAP-posterior-SD stop on the of-record fine grid (3201 nodes over [-8,8]) with a plateau guard
  * reference theta = full fold-bank fine-EAP on fold params for held-out models; recovery
    estimator = MWLE at the stop subset (of-record CAT recovery estimator)

Reach is decided on SE_posterior (EAP posterior SD) <= target ONLY. SE_total is reported
separately as sqrt(SD^2 + SE_param^2) where SE_param is the SOUND observed-information parametric
bootstrap (biggen_recovery_grid; full-data curated bank, reused across folds).

Writes ONLY into reports/biggen_gemini3_recal/oos_grid/. Does NOT modify the production engine,
does NOT call any judge/tutor model, does NOT lock an operating point.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
from scipy.special import log_expit, logsumexp

HERE = Path(__file__).resolve().parent
EE = HERE.parents[2]  # eduLLM-Evals (reports/biggen_gemini3_recal/oos_grid -> parents[2])
for _p in (str(EE), str(EE / "scripts"), str(EE / "biggen_calibration" / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import scripts.scenario_cat_lib as scat  # noqa: E402
import biggen_eap_stop_lib as E  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cm = _load("calibrate_mirt", EE / "scripts" / "calibrate_mirt.py")
rg = _load("biggen_recovery_grid", EE / "biggen_calibration" / "scripts" / "biggen_recovery_grid.py")

# ----------------------------------------------------------------------------- config
DIM = "general"
JUDGE = "gemini-3-flash-preview"
MATRIX_PATH = EE / "api_judge_pilot" / "grading_biggen" / "response_matrix.csv"
SCEN_PATH = EE / "data" / "BiGGen" / "scenarios.jsonl"
CURATED_BANK = EE / "reports" / "biggen_gemini3_recal" / "bank" / "biggen_unidim_modeled_gemini3_curated.jsonl"

# production fit config (matches recalibrate_gemini3 / Step A)
FIT_GRID = 7
RIDGE = 1e-2
FIT_MAX_ITER = 200
IMBALANCE_MARGIN = 3          # per-fold: exclude train pass_count<=3 OR fail_count<=3

# of-record fine EAP reference + stop grid (3201 nodes over [-8,8])
EAP_NODES = 3201
EAP_RANGE = 8.0

# sound SE_param (observed-info parametric bootstrap, full-data curated bank, reused across folds)
SEP_FIT_GRID = 7
SEP_RIDGE = 1e-2
SEP_BOOT_GRID_N = 241
SEP_BOOT_RANGE = 6.0
SEP_N_BOOT = 150
SEP_SEED = 20260801

K = 5
SEED = 20260729
CAP = 50                      # forced administration cap (scenarios)
TOP_N = 5
PLATEAU_DELTA = 0.005
PLATEAU_W = 3

FLOORS = [6, 8, 10, 12, 15]
TARGETS = [0.10, 0.12, 0.15, 0.18, 0.20, 0.25]


# ----------------------------------------------------------------------------- helpers
def variant_cols(Y, M, margin):
    """Keep columns with >=2 obs AND observed pass_count>margin AND fail_count>margin."""
    passc = np.where(M, Y, 0.0).sum(0)
    cnt = M.sum(0)
    failc = cnt - passc
    return (cnt >= 2) & (passc > margin) & (failc > margin)


def make_folds(models, k, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(models))
    folds = [[] for _ in range(k)]
    for pos, i in enumerate(idx):
        folds[pos % k].append(models[int(i)])
    return [sorted(f) for f in folds]


def stop_point(walk, floor, target):
    """First scenario step with n_scen>=floor AND (SD<=target -> reach) else plateau; else cap.

    Returns (step_dict, reason) with reason in {reach, plateau, cap}. Reach is SE_post<=target
    ONLY; plateau/cap are reported separately and NOT counted as reach.
    """
    post_floor = [s for s in walk if s["n_scen"] >= floor]
    for i, step in enumerate(post_floor):
        if step["eap_sd"] <= target:
            return step, "reach"
        if i + 1 >= PLATEAU_W:
            window = [post_floor[j]["eap_sd"] for j in range(i - PLATEAU_W + 1, i + 1)]
            if (max(window) - min(window)) < PLATEAU_DELTA:
                return step, "plateau"
    return (walk[-1], "cap") if walk else (None, "cap")


def _dist(vals):
    """median/mean/sd/max/iqr for a 1-D array (nan-safe)."""
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return dict(median=np.nan, mean=np.nan, sd=np.nan, max=np.nan, iqr=np.nan)
    return dict(median=float(np.median(v)), mean=float(np.mean(v)),
                sd=float(np.std(v, ddof=1)) if v.size > 1 else 0.0,
                max=float(np.max(v)),
                iqr=float(np.quantile(v, 0.75) - np.quantile(v, 0.25)))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--cap", type=int, default=CAP)
    ap.add_argument("--from-csv", action="store_true",
                    help="regenerate figures + decision table from existing CSVs (no re-run).")
    args = ap.parse_args()
    (HERE / "figures").mkdir(parents=True, exist_ok=True)
    tmp = EE / "staging" / "_biggen_gemini3_oos_grid"
    tmp.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if args.from_csv:
        cell_all = pd.read_csv(HERE / "oos_per_cell_grid.csv")
        cell_excl = pd.read_csv(HERE / "oos_per_cell_grid_excl_weak.csv")
        per_model_df = pd.read_csv(HERE / "oos_per_model_per_cell.csv")
        summary = json.loads((HERE / "summary.json").read_text(encoding="utf-8"))
        _heatmaps(cell_all, HERE / "figures" / "oos_grid_heatmaps.png")
        _decision_table(cell_all, cell_excl, per_model_df, summary, HERE / "DECISION_TABLE.md")
        print("(--from-csv) regenerated figures + decision table")
        return

    # ---- load curated bank (defines administrable pool + scen mapping) ----
    records, dims, _ = scat.load_fitted_bank(CURATED_BANK, "clamp")
    assert dims == [DIM], dims
    curated_ids = [r["criterion_id"] for r in records]
    scen_of_all = {r["criterion_id"]: r["scenario_id"] for r in records}

    matrix = pd.read_csv(MATRIX_PATH, index_col=0)
    models = list(matrix.index)
    P = len(models)
    row_of = {m: i for i, m in enumerate(models)}
    # full matrix on curated columns
    Ycur = matrix.reindex(columns=curated_ids).to_numpy(float)
    Mcur = ~np.isnan(Ycur)
    Ycur = np.nan_to_num(Ycur, nan=0.0)
    # full matrix on ALL columns (for per-fold refit / exclusion on training models)
    all_ids = list(matrix.columns)
    Yall = np.nan_to_num(matrix.to_numpy(float), nan=0.0)
    Mall = ~np.isnan(matrix.to_numpy(float))
    Q_all = np.ones((len(all_ids), 1), dtype=int)
    print(f"[data] {P} models x {len(all_ids)} criteria (curated pool={len(curated_ids)}); "
          f"fill={Mall.mean():.4f}", flush=True)

    egrid, elog = scat.build_grid(1, EAP_NODES, EAP_RANGE)     # fine EAP reference
    gg, lp = E.eap_grid(EAP_NODES, EAP_RANGE)                  # fine stop-walk grid

    # ---- SE_param (sound observed-info bootstrap on FULL-DATA curated bank; reused) ----
    print("[se_param] full-data curated observed-information item covariances ...", flush=True)
    cov = rg.compute_item_cov(CURATED_BANK, MATRIX_PATH, SEP_FIT_GRID, SEP_RIDGE)
    sep_col = cov["col"]
    bgrid = np.linspace(-SEP_BOOT_RANGE, SEP_BOOT_RANGE, SEP_BOOT_GRID_N)
    blp = -0.5 * bgrid ** 2
    blp = blp - logsumexp(blp)
    rng_boot = np.random.default_rng(SEP_SEED)

    def se_param_at(model, admin_ids):
        y = cov["Ymat"][row_of[model]]
        idx = np.array([sep_col[c] for c in admin_ids if c in sep_col], dtype=int)
        if idx.size == 0:
            return float("nan")
        _, _, se_par = rg.bootstrap_theta(y, idx, cov["beta"], cov["chol"],
                                          bgrid, blp, SEP_N_BOOT, rng_boot, batch=8)
        return float(se_par)

    # SE_param at FULL curated admin (floor reference, per model)
    full_admin_sep = np.array([
        se_param_at(m, [c for j, c in enumerate(curated_ids) if Mcur[row_of[m], j]])
        for m in models])
    print(f"  SE_param full-admin floor: median={np.nanmedian(full_admin_sep):.4f} "
          f"mean={np.nanmean(full_admin_sep):.4f}", flush=True)

    # ---- per-fold refit (production config) + forced-long engine order + fine-EAP ref ----
    folds = make_folds(models, K, SEED)
    fold_of = {m: f for f, fl in enumerate(folds) for m in fl}
    per_model = {}
    fold_conv = {}
    for f in range(K):
        test = folds[f]
        train = [m for m in models if m not in set(test)]
        tr = [row_of[m] for m in train]
        Ytr, Mtr = Yall[tr], Mall[tr]
        keep = variant_cols(Ytr, Mtr, IMBALANCE_MARGIN)
        src = np.where(keep)[0]
        kept_ids = [all_ids[j] for j in src]
        fit = cm.fit_m2pl_em(Ytr[:, keep], Mtr[:, keep], Q_all[keep], FIT_GRID,
                             estimate_corr=False, ridge=RIDGE, max_iter=FIT_MAX_ITER)
        Ak = np.asarray(fit["A"], float)          # (n,1)
        bk = np.asarray(fit["b"], float).ravel()
        colk = {c: i for i, c in enumerate(kept_ids)}
        scen_of = {c: scen_of_all.get(c, c.rsplit("_c", 1)[0]) for c in kept_ids}
        fold_conv[f] = {"converged": bool(fit["converged"]), "n_iter": int(fit["n_iter"]),
                        "n_kept": len(kept_ids), "max_a": float(Ak[:, 0].max()),
                        "abs_b_gt6": int((np.abs(bk) > 6).sum())}

        # fold bank (fresh records; criticality non-null so the engine runs)
        fold_bank = tmp / f"fold{f}_bank.jsonl"
        with fold_bank.open("w", encoding="utf-8") as fh:
            for jj, cid in enumerate(kept_ids):
                fh.write(json.dumps({
                    "criterion_id": cid, "scenario_id": scen_of[cid], "criterion": "",
                    "primary_skill": "general", "criticality": "not_critical",
                    "scoring_type": "binary", "status": "approved",
                    "discrimination": {DIM: float(Ak[jj, 0])}, "q_modeled": {DIM: 1},
                    "difficulty": float(bk[jj]),
                    "irt_params": {"source": "calibrated-m2pl"},
                }) + "\n")

        subte = matrix.loc[test].reindex(columns=kept_ids)
        Yte = np.nan_to_num(subte.to_numpy(float), nan=0.0)
        Mte = ~np.isnan(subte.to_numpy(float))
        theta_ref = scat.eap_all_models(Yte, Mte, Ak, bk, egrid, elog)
        obs_pass = np.array([Yte[i][Mte[i]].mean() if Mte[i].any() else np.nan
                             for i in range(len(test))])

        # forced-long adaptive order for held-out models
        specL = scat.RunSpec(seed=SEED, top_n=TOP_N, max_se=0.0, min_evals_per_skill=0,
                             min_scenarios=args.cap, max_scenarios=args.cap,
                             selection="trace", mode="cat", runs_dir=str(tmp / f"fold{f}_runs"))
        resL = scat.run_models(test, fold_bank, MATRIX_PATH, SCEN_PATH, "clamp", dims,
                               specL, workers=args.workers)
        rb = {r["model"]: r for r in resL}
        for ti, m in enumerate(test):
            order_ids = list(rb[m]["order"])
            walk = E.eap_walk(order_ids, Yte[ti], colk, scen_of, Ak[:, 0], bk, gg, lp)
            cap_sd = walk[-1]["eap_sd"] if walk else float("nan")
            per_model[m] = {"fold": f, "walk": walk, "theta_ref": float(theta_ref[ti][0]),
                            "colk": colk, "kept_ids": kept_ids, "Ak": Ak, "bk": bk,
                            "y_local": Yte[ti], "obs_pass": float(obs_pass[ti]),
                            "cap_sd": float(cap_sd)}
        shutil.rmtree(tmp / f"fold{f}_runs", ignore_errors=True)
        print(f"[fold {f}] train={len(train)} test={len(test)} kept={len(kept_ids)} "
              f"max_a={fold_conv[f]['max_a']:.2f} |b|>6={fold_conv[f]['abs_b_gt6']} "
              f"conv={fold_conv[f]['converged']}", flush=True)

    # ---- weakly-identified tail: cap SD (floor-independent) > loosest grid target ----
    loosest = max(TARGETS)
    weak_models = sorted([m for m in models if per_model[m]["cap_sd"] > loosest],
                         key=lambda m: -per_model[m]["cap_sd"])
    cap_sd_tail = sorted(((m, per_model[m]["cap_sd"]) for m in models),
                         key=lambda kv: -kv[1])[:8]
    print(f"[weak] cap-SD > {loosest} (can't reach any grid target even at cap): {weak_models}",
          flush=True)

    # ---- grid loop ----
    sep_cache = {}
    per_model_rows = []
    cell_rows = []
    for floor in FLOORS:
        for target in TARGETS:
            recs = []
            for m in models:
                pm = per_model[m]
                walk = pm["walk"]
                step, reason = stop_point(walk, floor, target)
                if step is None:
                    continue
                idx = step["idx"]
                n_scen = step["n_scen"]
                sd = step["eap_sd"]
                Ak, bk = pm["Ak"], pm["bk"]
                y = pm["y_local"]
                if idx.size == 0:
                    theta_cat = pm["theta_ref"]
                else:
                    th_ba = scat.eap_subset(y, idx, Ak, bk, egrid, elog)
                    th_cat, conv = scat.mwle_subset(y, idx, Ak, bk, th_ba)
                    theta_cat = float(th_cat[0])
                key = (m, int(n_scen))
                if key not in sep_cache:
                    admin_ids = [pm["kept_ids"][j] for j in idx]
                    sep_cache[key] = se_param_at(m, admin_ids)
                se_param = sep_cache[key]
                se_total = float(np.sqrt(sd ** 2 + se_param ** 2)) if np.isfinite(se_param) else float("nan")
                rec = {
                    "floor": floor, "target": target, "model": m, "fold": pm["fold"],
                    "n_scen": int(n_scen), "n_crit": int(step["n_crit"]), "reason": reason,
                    "reach_se_post": bool(reason == "reach"),
                    "plateau": bool(reason == "plateau"), "hit_cap": bool(reason == "cap"),
                    "overran_floor": bool(n_scen > floor),
                    "eap_sd": sd, "se_param": se_param, "se_total": se_total,
                    "theta_ref": pm["theta_ref"], "theta_cat": theta_cat,
                    "weak": bool(m in weak_models),
                }
                recs.append(rec)
                per_model_rows.append(rec)
            df = pd.DataFrame(recs)
            cell_rows.append(_cell_summary(df, floor, target, "all"))
            cell_rows.append(_cell_summary(df[~df["weak"]], floor, target, "excl_weak"))
        print(f"[grid] floor={floor} done ({time.time()-t0:.0f}s)", flush=True)

    per_model_df = pd.DataFrame(per_model_rows)
    cell_df = pd.DataFrame(cell_rows)

    per_model_df.to_csv(HERE / "oos_per_model_per_cell.csv", index=False)
    cell_all = cell_df[cell_df["subset"] == "all"].drop(columns=["subset"]).reset_index(drop=True)
    cell_excl = cell_df[cell_df["subset"] == "excl_weak"].drop(columns=["subset"]).reset_index(drop=True)
    cell_all.to_csv(HERE / "oos_per_cell_grid.csv", index=False)
    cell_excl.to_csv(HERE / "oos_per_cell_grid_excl_weak.csv", index=False)

    summary = {
        "study": "BiGGen gemini-3 CURATED floor x SE OOS operating-point grid (unidim general)",
        "phase": "2B", "local_only": True, "op_point_locked": False, "engine_untouched": True,
        "judge": JUDGE,
        "data": {
            "matrix": str(MATRIX_PATH.relative_to(EE)),
            "matrix_sha256": cov_matrix_sha(),
            "curated_bank": str(CURATED_BANK.relative_to(EE)),
            "n_models": P, "n_criteria_all": len(all_ids),
            "n_criteria_curated_pool": len(curated_ids),
            "n_scenarios_file": _count_lines(SCEN_PATH), "fill": float(Mall.mean()),
        },
        "fit_config": {
            "family": "m2pl-mml-em (production)", "fit_grid_gh_nodes": FIT_GRID, "ridge": RIDGE,
            "max_iter": FIT_MAX_ITER, "negative_policy": "clamp",
            "per_fold_imbalance_exclusion": f"train pass_count<={IMBALANCE_MARGIN} OR "
                                            f"fail_count<={IMBALANCE_MARGIN}",
            "fold_convergence": fold_conv,
        },
        "eap_stop": {"nodes": EAP_NODES, "range": EAP_RANGE, "prior": "standard-normal",
                     "plateau_delta": PLATEAU_DELTA, "plateau_w": PLATEAU_W, "cap": args.cap,
                     "reach_rule": "SE_posterior (EAP posterior SD) <= target ONLY"},
        "oos_design": {"k": K, "seed": SEED, "per_fold_refit": True,
                       "reference": "full fold-bank fine-EAP on fold params (held-out recovery)",
                       "final_estimator": "MWLE at stop", "top_n": TOP_N, "selection": "trace"},
        "se_param_sound": {
            "method": "observed-information parametric bootstrap (biggen_recovery_grid)",
            "fit_grid": SEP_FIT_GRID, "ridge": SEP_RIDGE, "boot_grid_nodes": SEP_BOOT_GRID_N,
            "boot_range": SEP_BOOT_RANGE, "n_boot": SEP_N_BOOT, "seed": SEP_SEED,
            "note": "full-data curated bank, reused across folds; SE_total = sqrt(SD^2 + SE_param^2)",
            "full_admin_floor_median": float(np.nanmedian(full_admin_sep)),
            "full_admin_floor_mean": float(np.nanmean(full_admin_sep)),
        },
        "grid": {"floors": FLOORS, "targets": TARGETS},
        "weak_identified": {"rule": f"cap-SD (forced-long, floor-independent) > {loosest}",
                            "models": weak_models,
                            "cap_sd_tail": [{"model": m, "cap_sd": round(s, 4)} for m, s in cap_sd_tail]},
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _heatmaps(cell_all, HERE / "figures" / "oos_grid_heatmaps.png")
    _decision_table(cell_all, cell_excl, per_model_df, summary, HERE / "DECISION_TABLE.md")
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"[done] wrote outputs to {HERE} in {time.time()-t0:.1f}s", flush=True)


def cov_matrix_sha():
    import hashlib
    h = hashlib.sha256()
    with open(MATRIX_PATH, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _count_lines(p):
    return sum(1 for line in open(p, encoding="utf-8") if line.strip())


def _cell_summary(df, floor, target, subset):
    base = {"subset": subset, "floor": floor, "target": target, "n_models": int(len(df))}
    if len(df) == 0:
        return base
    x = df["theta_ref"].to_numpy(float)
    y = df["theta_cat"].to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) >= 3 and x.std() > 0 and y.std() > 0:
        r = float(np.corrcoef(x, y)[0, 1])
        slope = float(np.polyfit(x, y, 1)[0])
        mae = float(np.mean(np.abs(y - x)))
    else:
        r = slope = mae = float("nan")
    ln = _dist(df["n_scen"])
    sdp = _dist(df["eap_sd"])
    stt = _dist(df["se_total"])
    base.update({
        # test length distribution
        "median_scen": ln["median"], "mean_scen": ln["mean"], "sd_scen": ln["sd"],
        "max_scen": ln["max"], "iqr_scen": ln["iqr"],
        "n_overran_floor": int(df["overran_floor"].sum()),
        "n_at_floor": int((df["n_scen"] == floor).sum()),
        # SE_post (EAP posterior SD) distribution
        "median_sd": sdp["median"], "mean_sd": sdp["mean"], "sd_sd": sdp["sd"],
        "max_sd": sdp["max"], "iqr_sd": sdp["iqr"],
        # SE_total distribution
        "median_se_total": stt["median"], "mean_se_total": stt["mean"], "sd_se_total": stt["sd"],
        "max_se_total": stt["max"], "iqr_se_total": stt["iqr"],
        "median_se_param": float(np.nanmedian(df["se_param"])),
        # reach / plateau / cap
        "pct_reach": float(100.0 * df["reach_se_post"].mean()),
        "pct_plateau": float(100.0 * df["plateau"].mean()),
        "pct_cap": float(100.0 * df["hit_cap"].mean()),
        # OOS recovery
        "recovery_r": r, "recovery_slope": slope, "theta_mae": mae,
    })
    return base


def _heatmaps(cell_all, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("recovery_r", "OOS recovery r"),
              ("mean_se_total", "mean SE_total"),
              ("mean_scen", "mean length (scenarios)"),
              ("pct_reach", "%reach (SE_post<=target)")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, (col, title) in zip(axes.ravel(), panels):
        piv = cell_all.pivot(index="floor", columns="target", values=col)
        im = ax.imshow(piv.values, aspect="auto", origin="lower", cmap="viridis")
        ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels([f"{c:.2f}" for c in piv.columns])
        ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
        ax.set_xlabel("SE_ability target"); ax.set_ylabel("floor (min scenarios)")
        ax.set_title(title)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                if np.isfinite(v):
                    txt = f"{v:.0f}" if col in ("pct_reach",) else (
                        f"{v:.1f}" if col == "mean_scen" else f"{v:.3f}")
                    ax.text(j, i, txt, ha="center", va="center", color="w", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("BiGGen gemini-3 CURATED floor x SE OOS grid (unidim general, N=52) "
                 "— mean-based panels", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _decision_table(cell_all, cell_excl, per_model_df, summary, path):
    def fmt(v, d=3):
        return "NA" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{d}f}"

    def ms(mean, sd, d=1):
        if not np.isfinite(mean):
            return "NA"
        return f"{mean:.{d}f}±{sd:.{d}f}"

    sep = summary["se_param_sound"]
    dta = summary["data"]
    lines = [
        "# BiGGen gemini-3 CURATED operating-point decision table (unidim `general`)",
        "",
        "LOCAL study. **No operating point is locked here** — this table is decision-support for "
        "the user to choose from.",
        "",
        "## Configuration",
        "",
        f"- Judge: `{summary['judge']}`. Matrix: `{dta['matrix']}` "
        f"(sha256 `{dta['matrix_sha256'][:16]}…`), N={dta['n_models']} models × "
        f"{dta['n_criteria_all']} criteria (curated administrable pool "
        f"{dta['n_criteria_curated_pool']}) × {dta['n_scenarios_file']} scenarios, "
        f"fill={dta['fill']:.4f}.",
        f"- Curated bank: `{dta['curated_bank']}`.",
        f"- Fit: production M2PL MML-EM, {summary['fit_config']['fit_grid_gh_nodes']} GH nodes, "
        f"ridge {summary['fit_config']['ridge']}, clamp. Per-fold refit (k={summary['oos_design']['k']}, "
        f"seed {summary['oos_design']['seed']}); per-fold pass-imbalance exclusion "
        f"({summary['fit_config']['per_fold_imbalance_exclusion']}).",
        f"- Stop: dense {summary['eap_stop']['nodes']}-node EAP posterior-SD over ±{summary['eap_stop']['range']}, "
        f"reach = **SE_post ≤ target ONLY**; plateau δ={summary['eap_stop']['plateau_delta']}/"
        f"W={summary['eap_stop']['plateau_w']}; cap={summary['eap_stop']['cap']}. MWLE θ at stop.",
        f"- Reference θ = full fold-bank fine-EAP on fold params (held-out recovery).",
        "",
        "## Sound SE_param used for SE_total",
        "",
        f"- Method: {sep['method']} (full-data curated bank, reused across folds). "
        f"SE_total = √(SD² + SE_param²).",
        f"- Full-bank admin SE_param floor: median **{fmt(sep['full_admin_floor_median'],4)}** "
        f"(mean {fmt(sep['full_admin_floor_mean'],4)}).",
        "",
        "## Per-cell grid — ALL 52 models (mean±SD for length & SE)",
        "",
        "| floor | target | len mean±SD | len med | len max | #overran | %reach | %plat | %cap | "
        "SD_post mean±SD | SE_tot mean±SD | med SE_param | r | slope | θMAE |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in cell_all.sort_values(["floor", "target"]).iterrows():
        lines.append(
            f"| {int(r['floor'])} | {r['target']:.2f} | {ms(r['mean_scen'], r['sd_scen'])} | "
            f"{r['median_scen']:.0f} | {r['max_scen']:.0f} | {int(r['n_overran_floor'])} | "
            f"{r['pct_reach']:.0f} | {r['pct_plateau']:.0f} | {r['pct_cap']:.0f} | "
            f"{ms(r['mean_sd'], r['sd_sd'], 3)} | {ms(r['mean_se_total'], r['sd_se_total'], 3)} | "
            f"{fmt(r['median_se_param'])} | {fmt(r['recovery_r'])} | {fmt(r['recovery_slope'])} | "
            f"{fmt(r['theta_mae'])} |")

    wk = summary["weak_identified"]
    lines += ["", "## Per-cell grid — EXCLUDING weakly-identified tail (mean±SD)", "",
              f"Weak rule: {wk['rule']}. Weak set (n={len(wk['models'])}): {wk['models']}", "",
              "| floor | target | len mean±SD | len med | %reach | %cap | SE_tot mean±SD | r | slope | θMAE |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in cell_excl.sort_values(["floor", "target"]).iterrows():
        lines.append(
            f"| {int(r['floor'])} | {r['target']:.2f} | {ms(r['mean_scen'], r['sd_scen'])} | "
            f"{r['median_scen']:.0f} | {r['pct_reach']:.0f} | {r['pct_cap']:.0f} | "
            f"{ms(r['mean_se_total'], r['sd_se_total'], 3)} | {fmt(r['recovery_r'])} | "
            f"{fmt(r['recovery_slope'])} | {fmt(r['theta_mae'])} |")

    # where the floor bites (mean length == floor -> floor binds; overran small)
    lines += ["", "## Where the floor bites", "",
              "A cell is *floor-bound* when most models stop AT the floor (few overran): the SE "
              "target is already satisfied by the floor length, so tightening the floor — not the "
              "SE target — controls length.", "",
              "| floor | target | %overran (of 52) | mean len | note |",
              "|---:|---:|---:|---:|:--|"]
    for _, r in cell_all.sort_values(["floor", "target"]).iterrows():
        pct_over = 100.0 * r["n_overran_floor"] / max(r["n_models"], 1)
        note = "floor binds" if pct_over <= 25 else ("SE binds" if pct_over >= 75 else "mixed")
        lines.append(f"| {int(r['floor'])} | {r['target']:.2f} | {pct_over:.0f} | "
                     f"{r['mean_scen']:.1f} | {note} |")

    lines += ["", "## Weakly-identified tail (cap SD, forced-long)", "",
              "Models whose EAP posterior SD at the forced-long cap still exceeds the loosest grid "
              "target — they *cannot* reach any grid SE target even at cap:", "",
              "| model | cap SD |", "|:--|---:|"]
    for row in wk["cap_sd_tail"]:
        flag = " **(weak)**" if row["model"] in wk["models"] else ""
        lines.append(f"| {row['model']}{flag} | {row['cap_sd']:.3f} |")

    lines += ["", "## Notes", "",
              "- Reach is gated on **SE_post ≤ target only**; SE_total (incl. SE_param) is reported, "
              "not gated.",
              "- `recovery_r`/`slope`/`θMAE` are OOS k-fold (fold-refit params, MWLE vs fold "
              "full-bank fine-EAP reference); length & SE are the achieved stop values.",
              f"- Prior Qwen op-point was floor 8 / SE 0.12. This grid re-evaluates it under "
              f"gemini-3 on the curated bank; **the choice remains the user's**.",
              ""]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
