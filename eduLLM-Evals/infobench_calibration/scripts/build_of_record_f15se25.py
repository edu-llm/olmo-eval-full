"""Build the OF-RECORD figure set for the InfoBench unidim operating point
(floor=15, SE_ability target=0.25) on the canonical ``instruction_following`` scale.

STUDY / reporting only, LOCAL. The production engine (``scripts/scenario_cat_lib.py``,
``scripts/calibrate_mirt.py``) and the grid runner (``run_oos_grid.py``) are imported
read-only and are NOT modified. Nothing is committed. Mirrors the TutorEval unidim
``of_record_f15se22`` numbered-experiment set.

From-CSV figures (reuse the already-run grid at floor=15/target=0.25):
  05 OOS recovery  -- theta_cat (MWLE at stop) vs theta_ref (full fold-bank OOS reference).
  07 parameter_uncertainty -- SE_post vs SE_total grouped bars (mean+/-SD AND median+IQR),
     0.25 target line, SE_param gap; plus length_and_se_distribution.csv.
  08 leaderboard   -- ranked unidim theta @15/0.25 with SE_total error bars, weak greyed.
  09 p-IRT MAE     -- predicted vs actual pass-rate at CAT theta (fold params over observed bank).

Fresh k-fold (reuses the grid's per-fold refit + engine adaptive order):
  04 adaptive-vs-random efficiency -- SE_post-vs-#scenarios and recovery-r-vs-#scenarios,
     with scenario savings to reach median SE_post <= 0.25.

Usage
-----
    uv run python reports/infobench_oos_grid_unidim/build_of_record_f15se25.py --workers 6
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

HERE = Path(__file__).resolve().parent
EE = HERE.parents[1]  # eduLLM-Evals
if str(EE) not in sys.path:
    sys.path.insert(0, str(EE))

import scripts.scenario_cat_lib as scat  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


GRID = _load("run_oos_grid", HERE / "run_oos_grid.py")

FLOOR = 15
SE_TARGET = 0.25
TAG = "f15se25"
WEAK_MODELS = (
    "BEE-spoke-data/smol_llama-220M-GQA-fineweb_edu",
    "BEE-spoke-data/smol_llama-220M-openhermes",
    "ai-forever/mGPT",
    "allenai/OLMo-1B-hf",
)


# ---------------------------------------------------------------------------
# Small stats helpers
# ---------------------------------------------------------------------------

def _ols(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    r = float(np.corrcoef(x, y)[0, 1])
    slope, intercept = np.polyfit(x, y, 1)
    mae = float(np.mean(np.abs(y - x)))
    return r, float(slope), float(intercept), mae


def _msmm(a):
    a = np.asarray(a, float)
    return (float(np.mean(a)), float(np.std(a, ddof=0)), float(np.median(a)),
            float(np.max(a)))


def _iqr(a):
    a = np.asarray(a, float)
    return float(np.percentile(a, 25)), float(np.percentile(a, 75))


# ---------------------------------------------------------------------------
# 05 OOS recovery (from CSV)
# ---------------------------------------------------------------------------

def fig05_recovery(cell: pd.DataFrame, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    h = cell[~cell["weak"]]
    w = cell[cell["weak"]]
    xh, yh = h["theta_ref"].to_numpy(), h["theta_cat"].to_numpy()
    xw, yw = w["theta_ref"].to_numpy(), w["theta_cat"].to_numpy()

    r_h, s_h, c_h, mae_h = _ols(xh, yh)
    r_all, s_all, c_all, mae_all = _ols(cell["theta_ref"], cell["theta_cat"])

    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    lo = min(cell["theta_ref"].min(), cell["theta_cat"].min()) - 0.3
    hi = max(cell["theta_ref"].max(), cell["theta_cat"].max()) + 0.3
    ax.plot([lo, hi], [lo, hi], ls="--", color="gray", lw=1, label="y = x")
    xs = np.linspace(lo, hi, 60)
    ax.plot(xs, s_h * xs + c_h, color="#c1666b", lw=1.6,
            label=f"OLS (excl-weak) slope={s_h:.3f}")
    ax.scatter(xh, yh, s=28, alpha=0.75, edgecolor="k", linewidth=0.25, color="#4d648d",
               label=f"headline N={len(h)} (r={r_h:.3f}, MAE={mae_h:.3f})")
    if xw.size:
        ax.scatter(xw, yw, s=90, marker="X", color="red", edgecolor="k",
                   label=f"weak (excl from fit, n={len(w)})", zorder=5)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("full-bank OOS reference theta (\u00a78.3, fold params)")
    ax.set_ylabel("CAT MWLE theta at stop (ability)")
    ax.set_title(
        f"InfoBench unidim OOS recovery @ 15/0.25\n"
        f"all-52 r={r_all:.3f}, slope={s_all:.3f} (slope>1 = mild scale EXPANSION)",
        fontsize=10.5)
    ax.legend(fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / f"oos_recovery_ability_{TAG}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    cell.sort_values("theta_ref").to_csv(out_dir / f"oos_per_model_{TAG}.csv", index=False)
    metrics = {
        "op_point": {"floor": FLOOR, "se_target": SE_TARGET},
        "headline_excl_weak": {"n": int(len(h)), "r": round(r_h, 4), "slope": round(s_h, 4),
                               "intercept": round(c_h, 4), "theta_mae": round(mae_h, 4)},
        "all_52": {"n": int(len(cell)), "r": round(r_all, 4), "slope": round(s_all, 4),
                   "intercept": round(c_all, 4), "theta_mae": round(mae_all, 4)},
        "note": "slope>1 => mild scale EXPANSION (opposite of TutorEval's compression).",
    }
    (out_dir / "recovery_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


# ---------------------------------------------------------------------------
# 07 parameter uncertainty (from CSV) -- mean+/-SD AND median+IQR
# ---------------------------------------------------------------------------

def fig07_se_components(cell: pd.DataFrame, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    h = cell[~cell["weak"]]

    sd = h["eap_sd"].to_numpy(float)
    st = h["se_total"].to_numpy(float)
    sp = h["se_param"].to_numpy(float)

    mean_sd, sdev_sd, med_sd, _ = _msmm(sd)
    mean_st, sdev_st, med_st, _ = _msmm(st)
    q1_sd, q3_sd = _iqr(sd)
    q1_st, q3_st = _iqr(st)
    med_sp = float(np.nanmedian(sp))

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    labels = ["SE_post\n(EAP posterior SD)", "SE_total\n(\u221a(SD\u00b2+SE_param\u00b2))"]
    x = np.arange(len(labels))
    wbar = 0.34

    means = [mean_sd, mean_st]
    sds = [sdev_sd, sdev_st]
    meds = [med_sd, med_st]
    lo_iqr = [med_sd - q1_sd, med_st - q1_st]
    hi_iqr = [q3_sd - med_sd, q3_st - med_st]

    b1 = ax.bar(x - wbar / 2, means, wbar, yerr=sds, capsize=6,
                color=["#4d648d", "#c1666b"], edgecolor="k",
                error_kw={"elinewidth": 1.3}, label="mean \u00b1 SD")
    b2 = ax.bar(x + wbar / 2, meds, wbar, yerr=[lo_iqr, hi_iqr], capsize=6,
                color=["#9db0cc", "#e0a3a6"], edgecolor="k",
                error_kw={"elinewidth": 1.3}, label="median + IQR")
    ax.axhline(SE_TARGET, ls=":", color="black", lw=1.5, label=f"SE target = {SE_TARGET}")

    for b, m in zip(b1, means):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.006, f"{m:.3f}", ha="center", fontsize=9)
    for b, m in zip(b2, meds):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.006, f"{m:.3f}", ha="center", fontsize=9)

    ax.annotate(f"SE_param gap\n(sound, median = {med_sp:.3f})",
                xy=(1 - wbar / 2, (mean_sd + mean_st) / 2), xytext=(0.42, 0.135),
                fontsize=8.5, color="#7a3b3f",
                arrowprops={"arrowstyle": "->", "color": "#7a3b3f", "lw": 1.0})
    ax.plot([1 - wbar / 2, 1 - wbar / 2], [mean_sd, mean_st], color="#7a3b3f", lw=1.4, ls="--")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("standard error (ability)")
    ax.set_title(f"InfoBench unidim SE components @ 15/0.25 (headline N={len(h)})", fontsize=10.5)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / f"se_ability_vs_total_bars_{TAG}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    summary = pd.DataFrame([
        {"component": "SE_post", "mean": round(mean_sd, 4), "sd": round(sdev_sd, 4),
         "median": round(med_sd, 4), "q1": round(q1_sd, 4), "q3": round(q3_sd, 4)},
        {"component": "SE_total", "mean": round(mean_st, 4), "sd": round(sdev_st, 4),
         "median": round(med_st, 4), "q1": round(q1_st, 4), "q3": round(q3_st, 4)},
        {"component": "SE_param", "mean": round(float(np.nanmean(sp)), 4),
         "sd": round(float(np.nanstd(sp, ddof=0)), 4), "median": round(med_sp, 4),
         "q1": round(float(np.nanpercentile(sp, 25)), 4),
         "q3": round(float(np.nanpercentile(sp, 75)), 4)},
    ])
    summary.to_csv(out_dir / "se_ability_vs_total_summary.csv", index=False)

    # length_and_se_distribution.csv: per-cell mean/SD/median/max/#overran-floor
    dist_rows = []
    for sub_name, sub in (("all_52", cell), ("excl_weak", cell[~cell["weak"]])):
        for metric, col in (("length_scenarios", "n_scen"), ("SE_post", "eap_sd"),
                            ("SE_total", "se_total")):
            a = sub[col].to_numpy(float)
            mean_, sd_, med_, max_ = _msmm(a)
            n_over = int((sub["n_scen"].to_numpy() > FLOOR).sum()) if metric == "length_scenarios" else ""
            dist_rows.append({
                "subset": sub_name, "metric": metric, "n_models": int(len(sub)),
                "mean": round(mean_, 4), "sd": round(sd_, 4), "median": round(med_, 4),
                "max": round(max_, 4), "n_overran_floor": n_over,
            })
    pd.DataFrame(dist_rows).to_csv(out_dir / "length_and_se_distribution.csv", index=False)

    return {"mean_se_post": round(mean_sd, 4), "sd_se_post": round(sdev_sd, 4),
            "median_se_post": round(med_sd, 4),
            "mean_se_total": round(mean_st, 4), "sd_se_total": round(sdev_st, 4),
            "median_se_total": round(med_st, 4), "median_se_param": round(med_sp, 4)}


# ---------------------------------------------------------------------------
# 08 leaderboard (from CSV)
# ---------------------------------------------------------------------------

def fig08_leaderboard(cell: pd.DataFrame, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    lb = cell.sort_values("theta_cat", ascending=False).reset_index(drop=True)
    lb.insert(0, "rank", np.arange(1, len(lb) + 1))
    cols = ["rank", "model", "theta_cat", "eap_sd", "se_param", "se_total",
            "n_scen", "n_crit", "reason", "reach_se_post", "weak"]
    lb[cols].to_csv(out_dir / f"leaderboard_{TAG}.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.5, max(9, 0.22 * len(lb))))
    y = np.arange(len(lb))[::-1]
    colors = ["#bdbdbd" if wk else "#4d648d" for wk in lb["weak"]]
    ax.errorbar(lb["theta_cat"], y, xerr=lb["se_total"], fmt="o", ms=4,
                ecolor="#999999", elinewidth=1, capsize=2, linestyle="none",
                mfc="none", mec="none", zorder=1)
    ax.scatter(lb["theta_cat"], y, c=colors, s=26, edgecolor="k", linewidth=0.3, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([m + ("  (weak)" if wk else "") for m, wk in zip(lb["model"], lb["weak"])],
                       fontsize=6.0)
    ax.set_xlabel("CAT MWLE theta at stop (ability), error bars = SE_total")
    ax.set_title("InfoBench unidim leaderboard @ 15/0.25 (N=52; weak greyed)", fontsize=11)
    ax.grid(axis="x", ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / f"leaderboard_{TAG}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    top_h = lb[~lb["weak"]].iloc[0]
    bot_h = lb[~lb["weak"]].iloc[-1]
    return {"top_headline": {"model": top_h["model"], "theta": round(float(top_h["theta_cat"]), 3),
                             "se_total": round(float(top_h["se_total"]), 3)},
            "bottom_headline": {"model": bot_h["model"], "theta": round(float(bot_h["theta_cat"]), 3),
                                "se_total": round(float(bot_h["se_total"]), 3)}}


# ---------------------------------------------------------------------------
# Fold refits (parallel) + engine adaptive / matched-random walks (for 04 and 09)
# ---------------------------------------------------------------------------

def _random_order(kept_ids, scen_of, observed_mask, col, seed_model):
    rng = np.random.default_rng(seed_model)
    by_scen: dict[str, list[str]] = {}
    for cid in kept_ids:
        j = col[cid]
        if not observed_mask[j]:
            continue
        by_scen.setdefault(scen_of[cid], []).append(cid)
    scens = list(by_scen.keys())
    rng.shuffle(scens)
    order = []
    for s in scens:
        order.extend(by_scen[s])
    return order


def build_walks(args):
    """Reproduce the grid's per-fold refit + engine adaptive order, add a matched random arm.

    Returns (adaptive, random, pirt) structures keyed by model.
    """
    matrix = pd.read_csv(GRID.MATRIX_PATH, index_col=0).apply(pd.to_numeric, errors="coerce")
    ids = list(matrix.columns)
    models = list(matrix.index)
    P = len(models)
    Yall = np.nan_to_num(matrix.to_numpy(float), nan=0.0)
    Mall = np.isfinite(matrix.to_numpy(float))
    scen_of_all = {c: c.rsplit("_c", 1)[0] for c in ids}

    folds = GRID.make_folds(models, GRID.K, GRID.SEED)
    test_idx_by_fold = [[models.index(m) for m in fl] for fl in folds]

    print(f"[fit] refitting {GRID.K} folds with {args.workers} workers ...", flush=True)
    payloads = [(f, test_idx_by_fold[f]) for f in range(GRID.K)]
    fold_fits = {}
    with ProcessPoolExecutor(max_workers=min(args.workers, GRID.K),
                             initializer=GRID._init_fit,
                             initargs=(Yall, Mall, ids)) as ex:
        for res in ex.map(GRID._fit_fold, payloads):
            fold_fits[res["fold_id"]] = res
            print(f"  fold {res['fold_id']}: kept={res['n_kept']} "
                  f"converged={res['converged']}", flush=True)

    quad = scat.build_quadrature(1, GRID.EAP_NODES, np.eye(1), max_nodes=2000,
                                 method="normal_trapezoid", linear_bound=GRID.EAP_BOUND)
    gg = quad.grid[:, 0]
    lp = quad.log_prior
    scen_records = scat.load_scenario_records(GRID.SCEN_PATH)

    adaptive = {}
    random = {}
    pirt = {}
    for f in range(GRID.K):
        ff = fold_fits[f]
        kept_cols = np.array(ff["kept_cols"], dtype=int)
        kept_ids = [ids[c] for c in kept_cols]
        a_local = np.array(ff["a"], dtype=float)
        b_local = np.array(ff["b"], dtype=float)
        col = {c: i for i, c in enumerate(kept_ids)}
        scen_of = {c: scen_of_all[c] for c in kept_ids}
        A_local = a_local[:, None]
        records = [
            {"criterion_id": cid, "scenario_id": scen_of[cid], "criterion": cid,
             "primary_skill": "", "irt_params": {"source": "calibrated-m2pl", "calibrated": True}}
            for cid in kept_ids
        ]
        bank = scat.FittedBank(
            records=records, dims=(GRID.DIM,), criterion_ids=tuple(kept_ids),
            scenario_ids=tuple(scen_of[c] for c in kept_ids),
            Q=np.ones((len(kept_ids), 1), int), A=A_local, b=b_local,
            latent_correlation=np.eye(1), source_path=f"fold{f}",
        )
        spec = scat.RunSpec(seed=GRID.SEED, top_n=GRID.TOP_N, max_se=0.0, min_evals_per_skill=0,
                            min_scenarios=0, max_scenarios=args.cap, selection="trace",
                            mode="cat", stop_se_method="online")
        for m in folds[f]:
            row = matrix.loc[m]
            resp_vec = pd.to_numeric(row.reindex(kept_ids), errors="coerce").to_numpy(float)
            y_local = np.nan_to_num(resp_vec, nan=0.0)
            obs_mask = np.isfinite(resp_vec)
            ref = scat.batch_eap(resp_vec, A_local, b_local, quad)
            res = scat.run_recorded_model(m, row, bank, scen_records, quad, spec, mwle_ridge=1e-6)
            order_ids = list(res["criterion_order"])
            aw = GRID.eap_walk(order_ids, col, y_local, scen_of, a_local, b_local, gg, lp)
            seed_m = (GRID.SEED * 1000003 + models.index(m)) & 0x7FFFFFFF
            ro = _random_order(kept_ids, scen_of, obs_mask, col, seed_m)
            rw = GRID.eap_walk(ro, col, y_local, scen_of, a_local, b_local, gg, lp)
            theta_ref = float(ref.theta[0])
            adaptive[m] = (aw, theta_ref)
            random[m] = (rw, theta_ref)
            pirt[m] = {"a": a_local, "b": b_local, "y": y_local, "obs": obs_mask}
        print(f"[engine] fold {f}: walks for {len(folds[f])} held-out models", flush=True)
    return adaptive, random, pirt


# ---------------------------------------------------------------------------
# 09 p-IRT MAE (fold params over observed bank, theta_cat from CSV)
# ---------------------------------------------------------------------------

def fig09_pirt(cell: pd.DataFrame, pirt_params: dict, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, rr in cell.iterrows():
        m = rr["model"]
        pp = pirt_params.get(m)
        if pp is None:
            continue
        obs = pp["obs"]
        if obs.sum() == 0:
            continue
        theta = float(rr["theta_cat"])
        a = pp["a"][obs]
        b = pp["b"][obs]
        p_pred = expit(a * theta - b)
        pred_rate = float(np.mean(p_pred))
        actual_rate = float(np.mean(pp["y"][obs]))
        rows.append({"model": m, "theta_cat": theta, "n_obs_criteria": int(obs.sum()),
                     "pred_pass_rate": round(pred_rate, 4), "actual_pass_rate": round(actual_rate, 4),
                     "abs_err": round(abs(pred_rate - actual_rate), 4), "weak": bool(rr["weak"])})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"pirt_per_model_{TAG}.csv", index=False)

    h = df[~df["weak"]]
    mae_h = float(np.mean(h["abs_err"]))
    r_h = float(np.corrcoef(h["pred_pass_rate"], h["actual_pass_rate"])[0, 1])
    mae_all = float(np.mean(df["abs_err"]))
    r_all = float(np.corrcoef(df["pred_pass_rate"], df["actual_pass_rate"])[0, 1])

    # OLS best-fit (predicted ~ actual) on the headline set (excl weak).
    slope_h, intercept_h = np.polyfit(h["actual_pass_rate"], h["pred_pass_rate"], 1)
    slope_h = float(slope_h)
    intercept_h = float(intercept_h)

    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1, label="y = x (identity)")
    hh = df[~df["weak"]]
    ww = df[df["weak"]]
    ax.scatter(hh["actual_pass_rate"], hh["pred_pass_rate"], s=28, alpha=0.75,
               edgecolor="k", linewidth=0.25, color="#4d648d",
               label=f"headline N={len(hh)} (MAE={mae_h:.3f}, r={r_h:.3f})")
    if len(ww):
        ax.scatter(ww["actual_pass_rate"], ww["pred_pass_rate"], s=90, marker="X",
                   color="red", edgecolor="k", label=f"weak (n={len(ww)})", zorder=5)
    xs_fit = np.array([0.0, 1.0])
    sign = "+" if intercept_h >= 0 else "-"
    ax.plot(xs_fit, slope_h * xs_fit + intercept_h, ls="-", color="#c1440e", lw=1.6,
            zorder=4,
            label=(f"OLS fit: y = {slope_h:.3f}x {sign} {abs(intercept_h):.3f} "
                   f"(r={r_h:.3f})"))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("actual observed pass-rate (full observed fold bank)")
    ax.set_ylabel("predicted pass-rate at CAT theta (p-IRT)")
    ax.set_title("InfoBench unidim p-IRT: predicted vs actual pass-rate @ 15/0.25", fontsize=10.5)
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / f"pirt_pred_vs_actual_{TAG}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    metrics = {"op_point": {"floor": FLOOR, "se_target": SE_TARGET},
               "headline_excl_weak": {"n": int(len(h)), "pass_rate_mae": round(mae_h, 4),
                                      "r": round(r_h, 4),
                                      "ols_slope": round(slope_h, 4),
                                      "ols_intercept": round(intercept_h, 4)},
               "all_52": {"n": int(len(df)), "pass_rate_mae": round(mae_all, 4),
                          "r": round(r_all, 4)}}
    (out_dir / "pirt_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


# ---------------------------------------------------------------------------
# 04 adaptive vs random efficiency
# ---------------------------------------------------------------------------

def _median_curve(walks, key, max_n):
    out = np.full(max_n + 1, np.nan)
    for n in range(1, max_n + 1):
        vals = []
        for w in walks:
            step = next((s for s in w if s["n_scen"] == n), None)
            if step is not None:
                vals.append(step[key])
        if vals:
            out[n] = float(np.median(vals))
    return out


def _recovery_curve(walks_with_ref, max_n):
    out = np.full(max_n + 1, np.nan)
    for n in range(1, max_n + 1):
        xs, ys = [], []
        for w, ref in walks_with_ref:
            step = next((s for s in w if s["n_scen"] == n), None)
            if step is not None:
                xs.append(ref)
                ys.append(step["eap_mean"])
        if len(xs) >= 3:
            out[n] = float(np.corrcoef(xs, ys)[0, 1])
    return out


def fig04_efficiency(adaptive: dict, random: dict, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    adaptive_walks = [aw for aw, _ in adaptive.values()]
    random_walks = [rw for rw, _ in random.values()]
    adaptive_ref = list(adaptive.values())
    random_ref = list(random.values())

    max_n = max(max((w[-1]["n_scen"] for w in adaptive_walks), default=0),
                min(120, max((w[-1]["n_scen"] for w in random_walks), default=0)))
    max_n = int(max_n)
    ad_sd = _median_curve(adaptive_walks, "eap_sd", max_n)
    rd_sd = _median_curve(random_walks, "eap_sd", max_n)
    ad_r = _recovery_curve(adaptive_ref, max_n)
    rd_r = _recovery_curve(random_ref, max_n)

    def _first_cross(curve, target):
        for n in range(1, len(curve)):
            if np.isfinite(curve[n]) and curve[n] <= target:
                return n
        return None

    ad_reach = _first_cross(ad_sd, SE_TARGET)
    rd_reach = _first_cross(rd_sd, SE_TARGET)
    savings = (round(rd_reach / ad_reach, 2) if ad_reach and rd_reach else None)

    ns = np.arange(max_n + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.plot(ns[1:], ad_sd[1:], color="#2a6f4e", lw=1.8, label="adaptive (engine max-info)")
    ax.plot(ns[1:], rd_sd[1:], color="#b5651d", lw=1.8, ls="--", label="random order")
    ax.axhline(SE_TARGET, ls=":", color="black", lw=1.2, label=f"SE target = {SE_TARGET}")
    if ad_reach:
        ax.axvline(ad_reach, ls=":", color="#2a6f4e", lw=1)
    if rd_reach:
        ax.axvline(rd_reach, ls=":", color="#b5651d", lw=1)
    ax.set_xlabel("# scenarios administered")
    ax.set_ylabel("median SE_post (posterior SD)")
    ttl = f"adaptive reaches median SE<={SE_TARGET} at {ad_reach} vs random at {rd_reach} scenarios"
    if savings:
        ttl += f"  (~{savings}x fewer)"
    ax.set_title(ttl, fontsize=9)
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.plot(ns[1:], ad_r[1:], color="#2a6f4e", lw=1.8, label="adaptive")
    ax.plot(ns[1:], rd_r[1:], color="#b5651d", lw=1.8, ls="--", label="random order")
    ax.set_xlabel("# scenarios administered")
    ax.set_ylabel("OOS recovery r (EAP mean vs ref)")
    ax.set_title("recovery-r vs test length", fontsize=10)
    ax.legend(fontsize=9, loc="lower right")
    fig.suptitle("InfoBench unidim: adaptive vs random efficiency @ SE=0.25", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / f"adaptive_vs_random_efficiency_{TAG}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    curves = pd.DataFrame({"n_scenarios": ns[1:], "adaptive_median_se": ad_sd[1:],
                           "random_median_se": rd_sd[1:], "adaptive_recovery_r": ad_r[1:],
                           "random_recovery_r": rd_r[1:]})
    curves.to_csv(out_dir / "adaptive_vs_random_curves.csv", index=False)
    metrics = {"op_point": {"floor": FLOOR, "se_target": SE_TARGET},
               "n_held_out_traces": int(len(adaptive_walks)),
               "adaptive_reach_scenarios": ad_reach, "random_reach_scenarios": rd_reach,
               "scenario_savings_x": savings}
    (out_dir / "efficiency_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--per-model-cell", type=Path, default=HERE / "oos_per_model_per_cell.csv")
    p.add_argument("--out-dir", type=Path, default=HERE / "of_record_f15se25")
    p.add_argument("--cap", type=int, default=GRID.CAP)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--skip-fresh", action="store_true",
                   help="skip the fresh k-fold walks (04 + 09)")
    args = p.parse_args()

    exp = args.out_dir / "experiments"
    exp.mkdir(parents=True, exist_ok=True)

    pm = pd.read_csv(args.per_model_cell)
    pm["weak"] = pm["weak"].astype(bool)
    cell = pm[(pm["floor"] == FLOOR) & (pm["target"] == SE_TARGET)].copy()
    print(f"[cell] floor={FLOOR} target={SE_TARGET}: {len(cell)} models, "
          f"{int(cell['weak'].sum())} weak")

    rec = fig05_recovery(cell, exp / "05_oos_recovery")
    se = fig07_se_components(cell, exp / "07_parameter_uncertainty")
    lb = fig08_leaderboard(cell, exp / "08_leaderboard")
    print(f"[05] recovery all-52 r={rec['all_52']['r']} slope={rec['all_52']['slope']} "
          f"MAE={rec['all_52']['theta_mae']}; excl-weak r={rec['headline_excl_weak']['r']} "
          f"slope={rec['headline_excl_weak']['slope']}")
    print(f"[07] SE_post mean={se['mean_se_post']}+/-{se['sd_se_post']} "
          f"SE_total mean={se['mean_se_total']}+/-{se['sd_se_total']} "
          f"SE_param med={se['median_se_param']}")
    print(f"[08] top={lb['top_headline']['model']} theta={lb['top_headline']['theta']}; "
          f"bottom={lb['bottom_headline']['model']} theta={lb['bottom_headline']['theta']}")

    pirt = None
    eff = None
    if not args.skip_fresh:
        adaptive, random, pirt_params = build_walks(args)
        pirt = fig09_pirt(cell, pirt_params, exp / "09_pirt_mae")
        eff = fig04_efficiency(adaptive, random, exp / "04_efficiency_vs_random")
        print(f"[09] p-IRT MAE excl-weak={pirt['headline_excl_weak']['pass_rate_mae']} "
              f"r={pirt['headline_excl_weak']['r']}")
        print(f"[04] adaptive reach@{eff['adaptive_reach_scenarios']} vs "
              f"random@{eff['random_reach_scenarios']} (x{eff['scenario_savings_x']})")

    # ---- headline stats (all-52 + excl-weak) ----
    def _cell_headline(sub):
        len_m, len_sd, len_med, len_max = _msmm(sub["n_scen"])
        sp_m, sp_sd, sp_med, _ = _msmm(sub["eap_sd"])
        st_m, st_sd, st_med, _ = _msmm(sub["se_total"])
        r, slope, _c, mae = _ols(sub["theta_ref"], sub["theta_cat"])
        return {
            "n_models": int(len(sub)),
            "length_mean": round(len_m, 2), "length_sd": round(len_sd, 2),
            "length_median": round(len_med, 1), "length_max": round(len_max, 1),
            "n_overran_floor": int((sub["n_scen"].to_numpy() > FLOOR).sum()),
            "pct_reach_se_post": round(100.0 * float(sub["reach_se_post"].mean()), 1),
            "se_post_mean": round(sp_m, 4), "se_post_sd": round(sp_sd, 4), "se_post_median": round(sp_med, 4),
            "se_total_mean": round(st_m, 4), "se_total_sd": round(st_sd, 4), "se_total_median": round(st_med, 4),
            "recovery_r": round(r, 4), "recovery_slope": round(slope, 4), "theta_mae": round(mae, 4),
        }

    head_all = _cell_headline(cell)
    head_excl = _cell_headline(cell[~cell["weak"]])

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "STUDY / reporting only - LOCAL; production engine untouched; nothing committed; "
                  "infobench_calibration/ NOT created (user reviews first).",
        "benchmark": "InfoBench", "scale": "unidimensional (instruction_following ability)",
        "op_point": {"floor_min_scenarios": FLOOR, "se_ability_target": SE_TARGET,
                     "reach_rule": "SE_post (EAP posterior SD at stop) <= target ONLY",
                     "stop": "dense 801-node normal_trapezoid EAP posterior-SD; plateau delta=0.005/W=3; "
                             "cap 70; MWLE theta at stop",
                     "oos": "k=5, seed 20260729, per-fold refit (log-shrinkage-2PL lambda16, 401-node)",
                     "se_total": "sqrt(SE_post^2 + SE_param^2); SE_param = sound observed-info parametric "
                                 "bootstrap (CAT-admin median ~0.084), NOT the stale ~0.19"},
        "weak_models": list(WEAK_MODELS),
        "headline_all_52": head_all,
        "headline_excl_weak": head_excl,
        "recovery": rec,
        "se_components": se,
        "leaderboard": lb,
        "pirt": pirt,
        "efficiency": eff,
        "figures": {
            "05_oos_recovery": f"experiments/05_oos_recovery/oos_recovery_ability_{TAG}.png",
            "04_efficiency_vs_random": f"experiments/04_efficiency_vs_random/adaptive_vs_random_efficiency_{TAG}.png",
            "08_leaderboard": f"experiments/08_leaderboard/leaderboard_{TAG}.png",
            "09_pirt_mae": f"experiments/09_pirt_mae/pirt_pred_vs_actual_{TAG}.png",
            "07_parameter_uncertainty": f"experiments/07_parameter_uncertainty/se_ability_vs_total_bars_{TAG}.png",
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
