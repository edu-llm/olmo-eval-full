"""Build the OF-RECORD figure set for the LOCKED TutorEval unidim operating point
(floor=15, SE_ability target=0.22), and fix the %reach metric definition.

STUDY / reporting only, LOCAL. The production engine (``tutor_cat/``,
``scripts/scenario_cat_lib.py``) is imported/called read-only and is NOT modified. Nothing
is committed. Mirrors the TutorBench unidim ``of_record_f20se27`` numbered-experiment set.

Two jobs
--------
Job A -- corrected %reach:
  The grid's ``%reach`` wrongly required (SE_ability<=target AND SE_total<=0.30). The correct
  definition is SE_ability (EAP posterior SD at stop) <= target ONLY. Recompute the corrected
  %reach for every grid cell from ``oos_per_model_per_cell.csv`` (all-52 and excl-weak),
  keeping median SE_total as a SEPARATE reported precision number. Writes
  ``oos_per_cell_grid_reach_corrected.csv``.

Job B -- of-record figure set @ 15/0.22 into ``of_record_f15se22/experiments/``:
  05 OOS recovery scatter (theta_stop MWLE vs full-bank OOS reference theta) -- from CSV.
  07 SE-components bar (SE_ability vs SE_total, IQR, 0.22 target, SE_param gap) -- from CSV.
  08 leaderboard (ranked per-model theta @15/0.22, SE_total error bars, weak greyed) -- from CSV.
  09 p-IRT MAE (predicted pass-rate at CAT theta vs actual observed pass-rate) -- from CSV+bank+matrix.
  04 adaptive-vs-random efficiency (SE_ability & recovery-r vs #scenarios) -- FRESH k-fold run
     that reuses the grid machinery (engine forced adaptive trace + a matched random-order arm).

Usage
-----
    python reports/eap_oos_grid_tutoreval_unidim/build_of_record_f15se22.py --workers 6
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, log_expit, logsumexp

# reports/eap_oos_grid_tutoreval_unidim/ -> reports/ -> eduLLM-Evals/
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.scenario_cat_lib as scat  # noqa: E402  (read-only import of the shared engine lib)

HERE = Path(__file__).resolve().parent

FLOOR = 15
SE_TARGET = 0.22
SE_TOTAL_GATE = 0.30  # kept only as a SEPARATE reported precision number, NOT a reach gate
SE_PARAM_MEDIAN = 0.1166732479300126
CAP = 70
WEAK_MODELS = (
    "BEE-spoke-data/smol_llama-220M-GQA-fineweb_edu",
    "ai-forever/mGPT",
    "allenai/OLMo-1B-hf",
    "ibm-granite/granite-3.1-2b-instruct",
)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cm = _load("calibrate_mirt", ROOT / "scripts" / "calibrate_mirt.py")
GRID = _load("run_oos_grid_unidim", HERE / "run_oos_grid_unidim.py")


# ---------------------------------------------------------------------------
# Job A -- corrected %reach for the whole grid
# ---------------------------------------------------------------------------

def job_a_corrected_reach(pm: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """CORRECT %reach = SE_ability (posterior SD at stop) <= target ONLY.

    Emits a per-cell table with the corrected reach (all-52 + excl-weak), the OLD combined-gate
    reach for contrast, and median SE_total kept as a SEPARATE precision number.
    """
    rows = []
    for (floor, tgt), cell in pm.groupby(["floor", "se_target"]):
        excl = cell[~cell["weak"]]

        def _reach_corrected(sub):
            return round(100.0 * float(np.mean(sub["sd"].to_numpy() <= tgt)), 1)

        def _reach_combined(sub):
            sd = sub["sd"].to_numpy()
            st = sub["se_total"].to_numpy()
            return round(100.0 * float(np.mean((sd <= tgt) & (st <= SE_TOTAL_GATE))), 1)

        rows.append({
            "floor": int(floor), "se_target": float(tgt),
            "n_models_all": int(len(cell)),
            "pct_reach_corrected_all": _reach_corrected(cell),
            "pct_reach_combined_old_all": _reach_combined(cell),
            "median_se_ability_all": round(float(cell["sd"].median()), 4),
            "median_se_total_all": round(float(cell["se_total"].median()), 4),
            "n_models_excl_weak": int(len(excl)),
            "pct_reach_corrected_excl_weak": _reach_corrected(excl),
            "pct_reach_combined_old_excl_weak": _reach_combined(excl),
            "median_se_ability_excl_weak": round(float(excl["sd"].median()), 4),
            "median_se_total_excl_weak": round(float(excl["se_total"].median()), 4),
        })
    out = pd.DataFrame(rows).sort_values(["floor", "se_target"]).reset_index(drop=True)
    out.to_csv(out_path, index=False)
    return out


# ---------------------------------------------------------------------------
# Job B figures 05 / 07 / 08 / 09 (from CSV + bank/matrix)
# ---------------------------------------------------------------------------

def _ols(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    r = float(np.corrcoef(x, y)[0, 1])
    slope, intercept = np.polyfit(x, y, 1)
    mae = float(np.mean(np.abs(y - x)))
    return r, float(slope), float(intercept), mae


def fig05_recovery(cell: pd.DataFrame, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    h = cell[~cell["weak"]]
    w = cell[cell["weak"]]
    xh, yh = h["theta_ref"].to_numpy(), h["theta_mwle"].to_numpy()
    xw, yw = w["theta_ref"].to_numpy(), w["theta_mwle"].to_numpy()

    r_h, s_h, c_h, mae_h = _ols(xh, yh)
    r_all, s_all, c_all, mae_all = _ols(cell["theta_ref"], cell["theta_mwle"])

    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    lo = min(cell["theta_ref"].min(), cell["theta_mwle"].min()) - 0.3
    hi = max(cell["theta_ref"].max(), cell["theta_mwle"].max()) + 0.3
    ax.plot([lo, hi], [lo, hi], ls="--", color="gray", lw=1, label="y = x")
    xs = np.linspace(lo, hi, 60)
    ax.plot(xs, s_h * xs + c_h, color="#c1666b", lw=1.6, label=f"OLS (excl-weak) slope={s_h:.3f}")
    ax.scatter(xh, yh, s=28, alpha=0.75, edgecolor="k", linewidth=0.25, color="#4d648d",
               label=f"headline N={len(h)} (r={r_h:.3f}, MAE={mae_h:.3f})")
    if xw.size:
        ax.scatter(xw, yw, s=90, marker="X", color="red", edgecolor="k",
                   label=f"weak (excl from fit, n={len(w)})", zorder=5)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("full-bank OOS reference theta (ability)")
    ax.set_ylabel("CAT MWLE theta at stop (ability)")
    ax.set_title("TutorEval unidim OOS recovery @ floor=15, SE=0.22", fontsize=11)
    ax.legend(fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "oos_recovery_ability_f15se22.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    cell.sort_values("theta_ref").to_csv(out_dir / "oos_per_model_f15se22.csv", index=False)
    metrics = {
        "op_point": {"floor": FLOOR, "se_target": SE_TARGET},
        "headline_excl_weak": {"n": int(len(h)), "r": round(r_h, 4), "slope": round(s_h, 4),
                               "intercept": round(c_h, 4), "theta_mae": round(mae_h, 4)},
        "all_52": {"n": int(len(cell)), "r": round(r_all, 4), "slope": round(s_all, 4),
                   "intercept": round(c_all, 4), "theta_mae": round(mae_all, 4)},
    }
    (out_dir / "recovery_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def fig07_se_components(cell: pd.DataFrame, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    h = cell[~cell["weak"]]

    def _stats(a):
        a = np.asarray(a, float)
        return (float(np.median(a)), float(np.percentile(a, 25)), float(np.percentile(a, 75)))

    med_sd, q1_sd, q3_sd = _stats(h["sd"])
    med_st, q1_st, q3_st = _stats(h["se_total"])
    med_sp = float(np.median(h["se_param"]))

    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    labels = ["SE_ability\n(posterior SD)", "SE_total\n(sqrt(SD^2+SE_param^2))"]
    meds = [med_sd, med_st]
    lo_err = [med_sd - q1_sd, med_st - q1_st]
    hi_err = [q3_sd - med_sd, q3_st - med_st]
    bars = ax.bar(labels, meds, yerr=[lo_err, hi_err], capsize=6,
                  color=["#4d648d", "#c1666b"], edgecolor="k", width=0.55,
                  error_kw={"elinewidth": 1.3})
    ax.axhline(SE_TARGET, ls=":", color="black", lw=1.4, label=f"SE_ability target = {SE_TARGET}")
    for b, m in zip(bars, meds):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.006, f"{m:.3f}", ha="center", fontsize=10)
    ax.annotate(f"SE_param gap (median = {med_sp:.3f})",
                xy=(1, med_sd), xytext=(1.02, (med_sd + med_st) / 2 + 0.02),
                fontsize=8.5, color="#7a3b3f")
    ax.plot([1, 1], [med_sd, med_st], color="#7a3b3f", lw=1.2, ls="--")
    ax.set_ylabel("standard error (ability)")
    ax.set_title(f"TutorEval unidim SE components @ 15/0.22 (headline N={len(h)}, median+IQR)",
                 fontsize=10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "se_ability_vs_total_bars_f15se22.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    summary = pd.DataFrame([
        {"component": "SE_ability", "median": round(med_sd, 4), "q1": round(q1_sd, 4), "q3": round(q3_sd, 4)},
        {"component": "SE_total", "median": round(med_st, 4), "q1": round(q1_st, 4), "q3": round(q3_st, 4)},
        {"component": "SE_param", "median": round(med_sp, 4), "q1": round(float(np.percentile(h["se_param"], 25)), 4),
         "q3": round(float(np.percentile(h["se_param"], 75)), 4)},
    ])
    summary.to_csv(out_dir / "se_ability_vs_total_summary.csv", index=False)
    return {"median_se_ability": round(med_sd, 4), "median_se_total": round(med_st, 4),
            "median_se_param": round(med_sp, 4)}


def fig08_leaderboard(cell: pd.DataFrame, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    lb = cell.sort_values("theta_mwle", ascending=False).reset_index(drop=True)
    lb.insert(0, "rank", np.arange(1, len(lb) + 1))
    cols = ["rank", "model", "theta_mwle", "sd", "se_param", "se_total",
            "n_scenarios", "n_criteria", "stop_reason", "weak"]
    lb[cols].to_csv(out_dir / "leaderboard_f15se22.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.5, max(9, 0.22 * len(lb))))
    y = np.arange(len(lb))[::-1]
    colors = ["#bdbdbd" if wk else "#4d648d" for wk in lb["weak"]]
    ax.errorbar(lb["theta_mwle"], y, xerr=lb["se_total"], fmt="o", ms=4,
                ecolor="#999999", elinewidth=1, capsize=2, linestyle="none",
                mfc="none", mec="none", zorder=1)
    ax.scatter(lb["theta_mwle"], y, c=colors, s=26, edgecolor="k", linewidth=0.3, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([m + ("  (weak)" if wk else "") for m, wk in zip(lb["model"], lb["weak"])],
                       fontsize=6.0)
    ax.set_xlabel("CAT MWLE theta at stop (ability), error bars = SE_total")
    ax.set_title("TutorEval unidim leaderboard @ 15/0.22 (N=52; weak greyed)", fontsize=11)
    ax.grid(axis="x", ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / "leaderboard_f15se22.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    top_h = lb[~lb["weak"]].iloc[0]
    bot_h = lb[~lb["weak"]].iloc[-1]
    return {"top_headline": {"model": top_h["model"], "theta": round(float(top_h["theta_mwle"]), 3),
                             "se_total": round(float(top_h["se_total"]), 3)},
            "bottom_headline": {"model": bot_h["model"], "theta": round(float(bot_h["theta_mwle"]), 3)}}


def fig09_pirt(cell: pd.DataFrame, bank_path: Path, matrix_path: Path, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    records, dims, _ = scat.load_fitted_bank(bank_path, "clamp")
    ids = [r["criterion_id"] for r in records]
    a = np.array([float(r["discrimination"][dims[0]]) for r in records])
    b = np.array([float(r["difficulty"]) for r in records])
    matrix = pd.read_csv(matrix_path, index_col=0).reindex(columns=ids)
    Y = matrix.to_numpy(float)
    M = ~np.isnan(Y)
    row_of = {m: i for i, m in enumerate(matrix.index)}

    rows = []
    for _, rr in cell.iterrows():
        m = rr["model"]
        if m not in row_of:
            continue
        i = row_of[m]
        obs = M[i]
        if obs.sum() == 0:
            continue
        theta = float(rr["theta_mwle"])
        p_pred = expit(a[obs] * theta - b[obs])
        pred_rate = float(np.mean(p_pred))
        actual_rate = float(np.mean(Y[i, obs]))
        rows.append({"model": m, "theta_mwle": theta, "n_obs_criteria": int(obs.sum()),
                     "pred_pass_rate": round(pred_rate, 4), "actual_pass_rate": round(actual_rate, 4),
                     "abs_err": round(abs(pred_rate - actual_rate), 4), "weak": bool(rr["weak"])})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "pirt_per_model_f15se22.csv", index=False)

    h = df[~df["weak"]]
    mae_h = float(np.mean(h["abs_err"]))
    r_h = float(np.corrcoef(h["pred_pass_rate"], h["actual_pass_rate"])[0, 1])
    mae_all = float(np.mean(df["abs_err"]))
    r_all = float(np.corrcoef(df["pred_pass_rate"], df["actual_pass_rate"])[0, 1])

    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1, label="y = x")
    hh = df[~df["weak"]]; ww = df[df["weak"]]
    ax.scatter(hh["actual_pass_rate"], hh["pred_pass_rate"], s=28, alpha=0.75,
               edgecolor="k", linewidth=0.25, color="#4d648d",
               label=f"headline N={len(hh)} (MAE={mae_h:.3f}, r={r_h:.3f})")
    if len(ww):
        ax.scatter(ww["actual_pass_rate"], ww["pred_pass_rate"], s=90, marker="X",
                   color="red", edgecolor="k", label=f"weak (n={len(ww)})", zorder=5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("actual observed pass-rate (full observed bank)")
    ax.set_ylabel("predicted pass-rate at CAT theta (p-IRT)")
    ax.set_title("TutorEval unidim p-IRT: predicted vs actual pass-rate @ 15/0.22", fontsize=10.5)
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "pirt_pred_vs_actual_f15se22.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    metrics = {"op_point": {"floor": FLOOR, "se_target": SE_TARGET},
               "headline_excl_weak": {"n": int(len(h)), "pass_rate_mae": round(mae_h, 4), "r": round(r_h, 4)},
               "all_52": {"n": int(len(df)), "pass_rate_mae": round(mae_all, 4), "r": round(r_all, 4)}}
    (out_dir / "pirt_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


# ---------------------------------------------------------------------------
# Job B figure 04 -- FRESH k-fold adaptive-vs-random efficiency
# ---------------------------------------------------------------------------

def _random_order(kept_ids, scen_of, observed_mask, colk, seed_model):
    """Build a matched RANDOM administration order: shuffle the OBSERVED scenarios, then emit
    their criteria (bank order). Only criteria observed for this model are included, so the
    random arm is administered over the same information the engine had."""
    rng = np.random.default_rng(seed_model)
    by_scen: dict[str, list[str]] = {}
    for cid in kept_ids:
        j = colk[cid]
        if not observed_mask[j]:
            continue
        by_scen.setdefault(scen_of[cid], []).append(cid)
    scens = list(by_scen.keys())
    rng.shuffle(scens)
    order = []
    for s in scens:
        order.extend(by_scen[s])
    return order


def _median_curve(walks, key, max_n):
    """Median across models of ``key`` (eap_sd / eap_mean-derived) at each n_scen in 1..max_n."""
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
    """recovery-r vs #scenarios: at each n_scen correlate per-model EAP mean with theta_ref."""
    out = np.full(max_n + 1, np.nan)
    for n in range(1, max_n + 1):
        xs, ys = [], []
        for w, ref in walks_with_ref:
            step = next((s for s in w if s["n_scen"] == n), None)
            if step is not None:
                xs.append(ref); ys.append(step["eap_mean"])
        if len(xs) >= 3:
            out[n] = float(np.corrcoef(xs, ys)[0, 1])
    return out


def fig04_efficiency(args, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    records, dims, _ = scat.load_fitted_bank(args.bank, "clamp")
    if dims != ["ability"]:
        raise SystemExit(f"expected unidim bank, got {dims}")
    d = dims[0]
    ids = [r["criterion_id"] for r in records]
    scen_of = {r["criterion_id"]: r["scenario_id"] for r in records}
    crit_of = {r["criterion_id"]: r.get("criterion", "") for r in records}
    Q_all = np.array([[int(r["q_modeled"][dd]) for dd in dims] for r in records])

    matrix = pd.read_csv(args.matrix, index_col=0)
    sub = matrix.reindex(columns=ids)
    Yraw = sub.to_numpy(float)
    Mall = ~np.isnan(Yraw)
    models = list(matrix.index)
    row_of = {m: i for i, m in enumerate(models)}

    gg = np.linspace(-8.0, 8.0, 321)
    lp = -0.5 * gg ** 2
    lp = lp - logsumexp(lp)

    folds = GRID.make_folds(models, args.k, args.seed)
    adaptive_walks, random_walks = [], []
    adaptive_ref, random_ref = [], []
    args.tmp_dir.mkdir(parents=True, exist_ok=True)

    for f in range(args.k):
        test = folds[f]
        train = [m for m in models if m not in set(test)]
        tr_idx = [row_of[m] for m in train]
        Ytr, Mtr = Yraw[tr_idx], Mall[tr_idx]
        keep = [j for j in range(len(ids))
                if Mtr[:, j].sum() >= 2 and 0 < Ytr[Mtr[:, j], j].sum() < Mtr[:, j].sum()]
        keep = np.array(keep, dtype=int)
        fit = cm.fit_m2pl_em(np.nan_to_num(Ytr[:, keep], nan=0.0), Mtr[:, keep],
                             Q_all[keep], args.fit_grid, ridge=args.ridge, max_iter=args.max_iter)
        Ak, bk = fit["A"], fit["b"]
        ak = Ak[:, 0]
        kept_ids = [ids[j] for j in keep]
        colk = {c: i for i, c in enumerate(kept_ids)}
        print(f"  fold {f}: TRAIN={len(train)} TEST={len(test)} fit {len(kept_ids)} items "
              f"(loglik={fit['loglik']:.0f}, iters={fit['n_iter']})", flush=True)

        fold_bank = args.tmp_dir / f"fold{f}_bank.jsonl"
        with fold_bank.open("w", encoding="utf-8") as fh:
            for jj, cid in enumerate(kept_ids):
                fh.write(json.dumps({"criterion_id": cid, "scenario_id": scen_of[cid],
                                     "criterion": crit_of[cid],
                                     "discrimination": {d: float(Ak[jj, 0])},
                                     "q_modeled": {d: 1}, "difficulty": float(bk[jj])}) + "\n")

        subte = matrix.loc[test].reindex(columns=kept_ids)
        Yte = np.nan_to_num(subte.to_numpy(float), nan=0.0)
        Mte = ~np.isnan(subte.to_numpy(float))
        egrid, elog = scat.build_grid(1, 61, 6.0)
        theta_ref = scat.eap_all_models(Yte, Mte, Ak, bk, egrid, elog)

        res = GRID.run_forced(test, fold_bank, args.matrix, args.scenarios, dims, args.seed,
                              args.cap, args.workers, args.tmp_dir / f"runs_f{f}")
        for ti, m in enumerate(test):
            ref = float(theta_ref[ti][0])
            aw = GRID.eap_walk(res[m]["order"], Yte[ti], colk, scen_of, ak, bk, gg, lp)
            adaptive_walks.append(aw); adaptive_ref.append((aw, ref))
            seed_m = (args.seed * 1000003 + row_of[m]) & 0x7FFFFFFF
            ro = _random_order(kept_ids, scen_of, Mte[ti], colk, seed_m)
            rw = GRID.eap_walk(ro, Yte[ti], colk, scen_of, ak, bk, gg, lp)
            random_walks.append(rw); random_ref.append((rw, ref))
        shutil.rmtree(args.tmp_dir / f"runs_f{f}", ignore_errors=True)

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
    ax.set_xlabel("# scenarios administered"); ax.set_ylabel("median SE_ability (posterior SD)")
    ttl = f"adaptive reaches median SE<={SE_TARGET} at {ad_reach} vs random at {rd_reach} scenarios"
    if savings:
        ttl += f"  (~{savings}x fewer)"
    ax.set_title(ttl, fontsize=9)
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.plot(ns[1:], ad_r[1:], color="#2a6f4e", lw=1.8, label="adaptive")
    ax.plot(ns[1:], rd_r[1:], color="#b5651d", lw=1.8, ls="--", label="random order")
    ax.set_xlabel("# scenarios administered"); ax.set_ylabel("OOS recovery r (EAP mean vs ref)")
    ax.set_title("recovery-r vs test length", fontsize=10)
    ax.legend(fontsize=9, loc="lower right")
    fig.suptitle("TutorEval unidim: adaptive vs random efficiency @ SE=0.22", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "adaptive_vs_random_efficiency_f15se22.png", dpi=140, bbox_inches="tight")
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
    shutil.rmtree(args.tmp_dir, ignore_errors=True)
    return metrics


# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bank", type=Path,
                   default=ROOT / "data" / "TutorEval" / "rubrics_qmatrix_final_unidim_fitted.jsonl")
    p.add_argument("--matrix", type=Path, default=HERE / "_input" / "response_matrix_tutoreval_unidim.csv")
    p.add_argument("--scenarios", type=Path, default=ROOT / "data" / "TutorEval" / "scenarios_final.jsonl")
    p.add_argument("--per-model-cell", type=Path, default=HERE / "oos_per_model_per_cell.csv")
    p.add_argument("--out-dir", type=Path, default=HERE / "of_record_f15se22")
    p.add_argument("--tmp-dir", type=Path, default=HERE / "_tmp_ofrecord_f15se22")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260729)
    p.add_argument("--fit-grid", type=int, default=7)
    p.add_argument("--ridge", type=float, default=1e-2)
    p.add_argument("--max-iter", type=int, default=200)
    p.add_argument("--cap", type=int, default=CAP)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--skip-efficiency", action="store_true", help="skip the fresh k-fold (04) run")
    args = p.parse_args()

    exp = args.out_dir / "experiments"
    exp.mkdir(parents=True, exist_ok=True)

    pm = pd.read_csv(args.per_model_cell)
    pm["weak"] = pm["weak"].astype(bool)
    cell = pm[(pm["floor"] == FLOOR) & (pm["se_target"] == SE_TARGET)].copy()
    print(f"[cell] floor={FLOOR} SE={SE_TARGET}: {len(cell)} models, {int(cell['weak'].sum())} weak")

    # --- Job A ---
    corrected = job_a_corrected_reach(pm, HERE / "oos_per_cell_grid_reach_corrected.csv")
    lock = corrected[(corrected["floor"] == FLOOR) & (corrected["se_target"] == SE_TARGET)].iloc[0]
    print(f"[Job A] 15/0.22 corrected %reach: all-52={lock['pct_reach_corrected_all']}% "
          f"(old combined {lock['pct_reach_combined_old_all']}%), "
          f"excl-weak={lock['pct_reach_corrected_excl_weak']}% "
          f"(old {lock['pct_reach_combined_old_excl_weak']}%)")

    # --- Job B from-CSV figures ---
    rec = fig05_recovery(cell, exp / "05_oos_recovery")
    se = fig07_se_components(cell, exp / "07_parameter_uncertainty")
    lb = fig08_leaderboard(cell, exp / "08_leaderboard")
    pirt = fig09_pirt(cell, args.bank, args.matrix, exp / "09_pirt_mae")
    print(f"[05] recovery excl-weak r={rec['headline_excl_weak']['r']} "
          f"slope={rec['headline_excl_weak']['slope']} MAE={rec['headline_excl_weak']['theta_mae']}")
    print(f"[07] SE_ability={se['median_se_ability']} SE_total={se['median_se_total']} "
          f"SE_param={se['median_se_param']}")
    print(f"[08] top={lb['top_headline']['model']} theta={lb['top_headline']['theta']}; "
          f"bottom={lb['bottom_headline']['model']} theta={lb['bottom_headline']['theta']}")
    print(f"[09] p-IRT MAE={pirt['headline_excl_weak']['pass_rate_mae']} r={pirt['headline_excl_weak']['r']}")

    # --- Job B fresh efficiency (04) ---
    eff = None
    if not args.skip_efficiency:
        print("[04] fresh k-fold adaptive-vs-random efficiency ...")
        eff = fig04_efficiency(args, exp / "04_efficiency_vs_random")
        print(f"[04] adaptive reach@{eff['adaptive_reach_scenarios']} vs "
              f"random@{eff['random_reach_scenarios']} (x{eff['scenario_savings_x']})")

    # --- summary.json ---
    med_len = float(cell[~cell["weak"]]["n_scenarios"].median())
    med_len_all = float(cell["n_scenarios"].median())
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "STUDY / reporting only - LOCAL; production engine untouched; nothing committed.",
        "benchmark": "TutorEval", "scale": "unidimensional (single 'ability' axis)",
        "op_point": {"floor_min_scenarios": FLOOR, "se_ability_target": SE_TARGET,
                     "stop": "EAP-posterior dense-grid SD; plateau delta=0.005/W=3; cap 70; MWLE theta at stop",
                     "oos": "k=5, seed 20260729, per-fold refit (ridge 0.01, fit_grid 7, clamp)"},
        "n_models": int(len(cell)), "n_weak": int(cell["weak"].sum()),
        "weak_models": list(WEAK_MODELS),
        "headline_excl_weak": {
            "recovery_r": rec["headline_excl_weak"]["r"],
            "recovery_slope": rec["headline_excl_weak"]["slope"],
            "theta_mae": rec["headline_excl_weak"]["theta_mae"],
            "median_length_scenarios": med_len,
            "median_se_ability": se["median_se_ability"],
            "median_se_total": se["median_se_total"],
            "median_se_param": se["median_se_param"],
            "pct_reach_corrected": float(lock["pct_reach_corrected_excl_weak"]),
            "pirt_pass_rate_mae": pirt["headline_excl_weak"]["pass_rate_mae"],
            "pirt_r": pirt["headline_excl_weak"]["r"],
        },
        "all_52": {
            "recovery_r": rec["all_52"]["r"], "recovery_slope": rec["all_52"]["slope"],
            "theta_mae": rec["all_52"]["theta_mae"], "median_length_scenarios": med_len_all,
            "median_se_total": round(float(cell["se_total"].median()), 4),
            "pct_reach_corrected": float(lock["pct_reach_corrected_all"]),
        },
        "reach_definition_corrected": "SE_ability (EAP posterior SD at stop) <= target ONLY",
        "reach_old_combined_gate": "SE_ability <= target AND SE_total <= 0.30 (SUPERSEDED)",
        "pct_reach_delta_vs_old": {
            "all_52": round(float(lock["pct_reach_corrected_all"] - lock["pct_reach_combined_old_all"]), 1),
            "excl_weak": round(float(lock["pct_reach_corrected_excl_weak"]
                                     - lock["pct_reach_combined_old_excl_weak"]), 1)},
        "leaderboard": lb,
        "efficiency": eff,
        "figures": {
            "05_oos_recovery": "experiments/05_oos_recovery/oos_recovery_ability_f15se22.png",
            "07_parameter_uncertainty": "experiments/07_parameter_uncertainty/se_ability_vs_total_bars_f15se22.png",
            "08_leaderboard": "experiments/08_leaderboard/leaderboard_f15se22.png",
            "09_pirt_mae": "experiments/09_pirt_mae/pirt_pred_vs_actual_f15se22.png",
            "04_efficiency_vs_random": "experiments/04_efficiency_vs_random/adaptive_vs_random_efficiency_f15se22.png",
        },
        "job_a_corrected_grid_csv": "../oos_per_cell_grid_reach_corrected.csv",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
