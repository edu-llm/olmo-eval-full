"""OF-RECORD deliverable package for the TutorBench CORRECTNESS-ONLY UNIDIMENSIONAL
EAP-posterior CAT at the LOCKED operating point floor(min_scenarios)=20, SE_ability=0.27.

STUDY / reporting ONLY. LOCAL. Nothing is committed. The production engine
(``tutor_cat/``, ``scripts/scenario_cat_lib.py``) is READ / called as a library only and is
NOT modified. The ``tutorbench_calibration/`` package is NOT touched or repointed.

Mirrors the numbered-experiment figure set the 2-skill sibling bench ships, but for the
correctness-only unidim scale, so the numbers can be reviewed before making unidim canonical.

Figures produced (into ``of_record_f20se27/experiments/<NN>_.../``):
  05_oos_recovery          - theta_stop (MWLE) vs full-bank OOS reference theta scatter [from CSV]
  04_efficiency_vs_random  - adaptive vs random SE-vs-#scenarios and recovery-r-vs-#scenarios
                             [FRESH COMPUTE: reruns k-fold, adds a random-order arm]
  08_leaderboard           - ranked per-model unidim theta @20/0.27 with SE_total bars  [from CSV]
  09_pirt_mae              - predicted (p-IRT at CAT theta) vs actual pass-rate scatter
                             [FRESH COMPUTE: reruns k-fold refit banks + adaptive stop theta]
  07_parameter_uncertainty - SE_ability vs SE_total grouped bar at 20/0.27               [from CSV]
plus a top-level README.md index + summary.json.

The from-CSV figures reuse:
  oos_grid_full/oos_per_model_per_cell.csv          (filtered floor==20 & se_target==0.27)
  oos_grid_full/oos_per_cell_grid_excl.csv          (headline grid row)
  param_uncertainty/metrics.json                    (median SE_param offset)

The FRESH computes reuse the exact k-fold machinery from
``oos_grid_unidim_correctness_only`` (same fitter/config/stop/seed) and only add a
matched random-order (mode="baseline") arm + the p-IRT prediction.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.special import expit  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # eduLLM-Evals
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import scenario_cat_lib as scl  # noqa: E402
import eap_stop_grid_study as G  # noqa: E402
import oos_grid_unidim_correctness_only as base  # noqa: E402

FLOOR = 20
SE_TARGET = 0.27
DIM = "correctness"
CAP = base.CAP  # 70
PLATEAU_W = base.PLATEAU_W  # 3
PLATEAU_DELTA = base.PLATEAU_DELTA  # 0.005
HEADLINE_EXCLUDE = ("Qwen/Qwen1.5-1.8B",)  # salamandra-7b rehabilitated -> kept
BLUE = "#4d648d"
RED = "#c1666b"
GREY = "#b0b0b0"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _fit_stats(x: np.ndarray, y: np.ndarray) -> dict:
    r = float(np.corrcoef(x, y)[0, 1])
    slope, intercept = np.polyfit(x, y, 1)
    return {"r": r, "slope": float(slope), "intercept": float(intercept),
            "mae": float(np.mean(np.abs(y - x)))}


def _step_at(nscen: list[int], values: list[float], targets: np.ndarray) -> np.ndarray:
    """Value of a right-continuous step function at each target length (last boundary<=target).
    NaN where target precedes the first boundary."""
    ns = np.asarray(nscen, float)
    out = np.full(targets.shape, np.nan)
    for i, t in enumerate(targets):
        j = np.searchsorted(ns, t, side="right") - 1
        if j >= 0:
            out[i] = values[j]
    return out


# ---------------------------------------------------------------------------
# Fig 05 - OOS recovery scatter (from existing per-model CSV)
# ---------------------------------------------------------------------------
def fig_oos_recovery(cell: pd.DataFrame, grid_row: pd.Series, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    head = cell[~cell["is_headline_excluded"]]
    weak = cell[cell["is_headline_excluded"]]
    x = head[f"theta_ref_{DIM}"].to_numpy(float)
    y = head[f"theta_mwle_{DIM}"].to_numpy(float)
    st = _fit_stats(x, y)
    xw = weak[f"theta_ref_{DIM}"].to_numpy(float)
    yw = weak[f"theta_mwle_{DIM}"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(6.4, 6.2))
    lo = min(x.min(), y.min(), xw.min() if xw.size else x.min()) - 0.3
    hi = max(x.max(), y.max(), xw.max() if xw.size else x.max()) + 0.3
    ax.plot([lo, hi], [lo, hi], ls="--", color="gray", lw=1, label="y = x")
    xs = np.linspace(lo, hi, 60)
    ax.plot(xs, st["slope"] * xs + st["intercept"], color=RED, lw=1.6,
            label=f"OLS fit (slope={st['slope']:.3f})")
    ax.scatter(x, y, s=26, alpha=0.75, edgecolor="k", linewidth=0.25, color=BLUE,
               label=f"headline N={x.size} (r={st['r']:.3f}, theta-MAE={st['mae']:.3f})")
    for xi, yi, name in zip(xw, yw, weak["model"]):
        ax.scatter([xi], [yi], s=95, marker="X", color="red", edgecolor="k", zorder=5,
                   label=f"excl. (info-limited): {name.split('/')[-1]}")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("reference full-bank OOS EAP theta (correctness)")
    ax.set_ylabel("CAT MWLE theta at stop (correctness)")
    ax.set_title(f"OOS recovery @ floor={FLOOR}, SE_ability={SE_TARGET}\n"
                 f"correctness-only unidim (N=114 headline)", fontsize=11)
    ax.legend(fontsize=7.8, loc="upper left")
    fig.tight_layout()
    p = out_dir / "oos_recovery_correctness_f20se27.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)

    cell.to_csv(out_dir / "oos_per_model_f20se27.csv", index=False)
    metrics = {
        "operating_point": {"floor": FLOOR, "se_ability_target": SE_TARGET},
        "scale": "correctness-only unidimensional", "n_headline": int(x.size),
        "recomputed_from_per_model_cell": st,
        "grid_row_20_0.27": {
            "r": float(grid_row[f"r_{DIM}"]), "slope": float(grid_row[f"slope_{DIM}"]),
            "theta_mae": float(grid_row[f"theta_mae_{DIM}"]),
            "median_sd": float(grid_row[f"median_sd_{DIM}"]),
            "median_se_total": float(grid_row[f"median_se_total_{DIM}"]),
            "median_len": float(grid_row["median_len"]), "mean_len": float(grid_row["mean_len"]),
        },
        "weak_excluded": {row["model"]: {"theta_ref": float(row[f"theta_ref_{DIM}"]),
                                          "theta_mwle": float(row[f"theta_mwle_{DIM}"])}
                          for _, row in weak.iterrows()},
        "source": "oos_grid_full/oos_per_model_per_cell.csv (from CSV, no recompute)",
    }
    (out_dir / "recovery_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return {"fig": p, "stats": st, "grid": metrics["grid_row_20_0.27"]}


# ---------------------------------------------------------------------------
# Fig 08 - leaderboard (from existing per-model CSV)
# ---------------------------------------------------------------------------
def fig_leaderboard(cell: pd.DataFrame, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    lb = pd.DataFrame({
        "model": cell["model"].to_numpy(),
        "theta_stop_correctness": cell[f"theta_mwle_{DIM}"].to_numpy(float),
        "sd_correctness": cell[f"sd_stop_{DIM}"].to_numpy(float),
        "se_total_correctness": cell[f"se_total_{DIM}"].to_numpy(float),
        "length": cell["length"].to_numpy(int),
        "stop_reason": cell["stop_reason"].to_numpy(),
        "weak_calibrated": cell["is_headline_excluded"].to_numpy(bool),
    })
    lb["reached_se_ability"] = lb["sd_correctness"] <= SE_TARGET
    lb = lb.sort_values("theta_stop_correctness", ascending=False).reset_index(drop=True)
    lb.to_csv(out_dir / "leaderboard_f20se27.csv", index=False)

    n = len(lb)
    fig, ax = plt.subplots(figsize=(9, max(6, n * 0.16)))
    ypos = np.arange(n)[::-1]
    colors = [GREY if w else BLUE for w in lb["weak_calibrated"]]
    ax.errorbar(lb["theta_stop_correctness"], ypos, xerr=lb["se_total_correctness"], fmt="none",
                ecolor="#999999", elinewidth=0.8, capsize=1.6, zorder=1)
    ax.scatter(lb["theta_stop_correctness"], ypos, c=colors, s=20, zorder=2,
               edgecolor="k", linewidth=0.2)
    ax.set_yticks(ypos)
    ax.set_yticklabels([m.split("/")[-1] for m in lb["model"]], fontsize=4.8)
    ax.set_xlabel("CAT MWLE theta at stop (correctness), error bars = SE_total")
    ax.set_title(f"TutorBench correctness-only unidim CAT leaderboard @ floor={FLOOR}/SE={SE_TARGET}\n"
                 f"ranked; grey = weak-calibrated (excl. from headline, N=114)", fontsize=9)
    for yi, w in zip(ypos, lb["weak_calibrated"]):
        if w:
            ax.annotate("weak", (ax.get_xlim()[0], yi), fontsize=5, color="red",
                        va="center", ha="left")
    ax.margins(y=0.005)
    fig.tight_layout()
    p = out_dir / "leaderboard_correctness_f20se27.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"fig": p, "csv": out_dir / "leaderboard_f20se27.csv", "table": lb}


# ---------------------------------------------------------------------------
# Fig 07 - SE_ability vs SE_total bars (from existing per-model CSV + param_uncertainty)
# ---------------------------------------------------------------------------
def fig_se_components(cell: pd.DataFrame, se_param_median: float, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    head = cell[~cell["is_headline_excluded"]]
    se_ab = head[f"sd_stop_{DIM}"].to_numpy(float)
    se_tot = head[f"se_total_{DIM}"].to_numpy(float)
    all_ab = cell[f"sd_stop_{DIM}"].to_numpy(float)
    all_tot = cell[f"se_total_{DIM}"].to_numpy(float)

    def med_iqr(v):
        return (float(np.median(v)), float(np.percentile(v, 25)), float(np.percentile(v, 75)))

    m_ab, lo_ab, hi_ab = med_iqr(se_ab)
    m_tot, lo_tot, hi_tot = med_iqr(se_tot)

    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    xpos = np.array([0.0, 0.9])
    meds = [m_ab, m_tot]
    errs = [[m_ab - lo_ab, m_tot - lo_tot], [hi_ab - m_ab, hi_tot - m_tot]]
    bars = ax.bar(xpos, meds, width=0.6, color=[BLUE, RED], edgecolor="k", linewidth=0.6,
                  yerr=errs, capsize=5, error_kw={"elinewidth": 1.0, "ecolor": "#333333"})
    ax.axhline(SE_TARGET, ls=":", color="crimson", lw=1.4, label=f"SE target = {SE_TARGET}")
    ax.set_xticks(xpos)
    ax.set_xticklabels(["SE_ability\n(posterior SD at stop)", "SE_total\n(sqrt(SD^2+SE_param^2))"],
                       fontsize=9)
    ax.set_ylabel("standard error (correctness theta)")
    ax.set_title(f"SE_ability vs SE_total @ floor={FLOOR}/SE={SE_TARGET}\n"
                 f"correctness-only unidim (N=114 headline)", fontsize=10.5)
    for b, m in zip(bars, meds):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.006, f"{m:.3f}", ha="center", fontsize=9)
    # annotate the SE_param gap
    ax.annotate(f"SE_param gap\n(median SE_param = {se_param_median:.3f})",
                xy=(0.9, m_tot), xytext=(0.35, m_tot + 0.09),
                fontsize=8, ha="center",
                arrowprops=dict(arrowstyle="->", color="#333333", lw=0.9))
    ax.set_ylim(0, max(hi_tot, SE_TARGET) + 0.13)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    p = out_dir / "se_ability_vs_total_bars_f20se27.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)

    summ = pd.DataFrame({
        "skill": [DIM, DIM], "pool": ["headline_N114", "all_115"],
        "median_se_ability": [m_ab, float(np.median(all_ab))],
        "p25_se_ability": [lo_ab, float(np.percentile(all_ab, 25))],
        "p75_se_ability": [hi_ab, float(np.percentile(all_ab, 75))],
        "median_se_total": [m_tot, float(np.median(all_tot))],
        "p25_se_total": [lo_tot, float(np.percentile(all_tot, 25))],
        "p75_se_total": [hi_tot, float(np.percentile(all_tot, 75))],
        "median_se_param_offset": [se_param_median, se_param_median],
    })
    summ.to_csv(out_dir / "se_ability_vs_total_summary.csv", index=False)
    return {"fig": p, "median_se_ability": m_ab, "median_se_total": m_tot,
            "median_se_param": se_param_median}


# ---------------------------------------------------------------------------
# FRESH k-fold run: adaptive + matched random arm + fold banks (for 04 and 09)
# ---------------------------------------------------------------------------
def run_kfold(args) -> dict:
    from concurrent.futures import ProcessPoolExecutor

    records, dims, _ = scl.load_fitted_bank(args.bank, args.negative_policy)
    ids = [r["criterion_id"] for r in records]
    scen_of = {r["criterion_id"]: r["scenario_id"] for r in records}
    crit_of = {r["criterion_id"]: r.get("criterion", "") for r in records}
    Q_all = np.array([[int(r["q_modeled"][d]) for d in dims] for r in records])
    matrix = pd.read_csv(args.matrix, index_col=0)
    Yraw = matrix.reindex(columns=ids).to_numpy(float)
    Mall = ~np.isnan(Yraw)
    models = list(matrix.index)
    row_of = {m: i for i, m in enumerate(models)}
    folds = base.make_folds(models, args.k, args.seed)

    adaptive_spec = scl.RunSpec(seed=args.seed, max_se=1e-9, min_evals_per_skill=0,
                                min_scenarios=0, max_scenarios=1_000_000, selection="trace",
                                mode="cat", write_logs=False,
                                runs_dir=str(args.tmp_dir / "engine_runs_adaptive"))
    random_spec = scl.RunSpec(seed=args.seed, max_se=1e-9, min_evals_per_skill=0,
                              min_scenarios=0, max_scenarios=1_000_000, selection="trace",
                              mode="baseline", write_logs=False,
                              runs_dir=str(args.tmp_dir / "engine_runs_random"))

    adaptive, random_, fold_bank_of_model = [], [], {}
    for f in range(args.k):
        test = folds[f]
        train = [m for m in models if m not in set(folds[f])]
        fold_bank = args.tmp_dir / f"fold{f}_bank.jsonl"
        diag = base.refit_fold(train, models, row_of, Yraw, Mall, ids, Q_all, scen_of, crit_of,
                               dims, fold_bank, args.fit_grid, args.ridge, args.max_iter)
        for m in test:
            fold_bank_of_model[m] = fold_bank
        print(f"  fold {f}: TRAIN={len(train)} TEST={len(test)} fit {diag['n_items']} items "
              f"(conv={diag['converged']})", flush=True)
        common = (str(fold_bank), str(args.matrix), str(args.scenarios), args.negative_policy,
                  args.stop_nodes, args.cap)
        for spec, sink, tag in ((adaptive_spec, adaptive, "adaptive"),
                                (random_spec, random_, "random")):
            initargs = common + (asdict(spec),)
            with ProcessPoolExecutor(max_workers=args.workers, initializer=base._init_worker,
                                     initargs=initargs) as ex:
                for tr in ex.map(base._run_model, test):
                    if tr is not None:
                        tr["fold"] = f
                        sink.append(tr)
            print(f"    -> {tag}: {len(test)} traces done", flush=True)
    return {"adaptive": adaptive, "random": random_, "dims": dims,
            "fold_bank_of_model": fold_bank_of_model, "matrix": matrix,
            "negative_policy": args.negative_policy, "folds": folds}


# ---------------------------------------------------------------------------
# Fig 04 - adaptive vs random efficiency (fresh)
# ---------------------------------------------------------------------------
def fig_efficiency(kf: dict, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    excl = set(HEADLINE_EXCLUDE)
    targets = np.arange(1, CAP + 1)

    def arm_curves(traces):
        sd_mat, x_ref, y_at = [], [], []
        for tr in traces:
            if tr["model"] in excl:
                continue
            sds = [float(s[0]) for s in tr["sd"]]
            mws = [float(m[0]) for m in tr["mwle"]]
            sd_mat.append(_step_at(tr["nscen"], sds, targets))
            y_at.append(_step_at(tr["nscen"], mws, targets))
            x_ref.append(float(tr["theta_ref"][0]))
        sd_mat = np.vstack(sd_mat)
        y_at = np.vstack(y_at)
        x_ref = np.asarray(x_ref, float)
        median_sd = np.nanmedian(sd_mat, axis=0)
        r_curve = np.full(targets.shape, np.nan)
        for j in range(targets.size):
            yj = y_at[:, j]
            ok = np.isfinite(yj)
            if ok.sum() >= 3:
                r_curve[j] = float(np.corrcoef(x_ref[ok], yj[ok])[0, 1])
        return median_sd, r_curve

    ad_sd, ad_r = arm_curves(kf["adaptive"])
    rd_sd, rd_r = arm_curves(kf["random"])

    def first_cross(median_sd):
        below = np.where(median_sd <= SE_TARGET)[0]
        return int(targets[below[0]]) if below.size else None

    ad_reach = first_cross(ad_sd)
    rd_reach = first_cross(rd_sd)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 5.2))
    axA.plot(targets, ad_sd, color=BLUE, lw=2, label="adaptive (CAT)")
    axA.plot(targets, rd_sd, color=RED, lw=2, ls="--", label="random order")
    axA.axhline(SE_TARGET, ls=":", color="crimson", lw=1.3, label=f"SE target = {SE_TARGET}")
    if ad_reach:
        axA.axvline(ad_reach, ls=":", color=BLUE, lw=1)
    if rd_reach:
        axA.axvline(rd_reach, ls=":", color=RED, lw=1)
    axA.set_xlabel("# scenarios administered")
    axA.set_ylabel("median SE_ability (posterior SD)")
    axA.set_title("(a) SE_ability vs #scenarios", fontsize=10.5)
    axA.legend(fontsize=8.5)
    axA.set_xlim(1, CAP)

    axB.plot(targets, ad_r, color=BLUE, lw=2, label="adaptive (CAT)")
    axB.plot(targets, rd_r, color=RED, lw=2, ls="--", label="random order")
    axB.set_xlabel("# scenarios administered")
    axB.set_ylabel("recovery r (theta_at vs full-bank OOS ref)")
    axB.set_title("(b) recovery-r vs #scenarios", fontsize=10.5)
    axB.legend(fontsize=8.5, loc="lower right")
    axB.set_xlim(1, CAP)

    reach_txt = (f"adaptive reaches median SE<={SE_TARGET} at "
                 f"{ad_reach if ad_reach else '>'+str(CAP)} scenarios; "
                 f"random at {rd_reach if rd_reach else '>'+str(CAP)}.")
    fig.suptitle(f"Adaptive vs random efficiency @ correctness-only unidim, floor={FLOOR}/SE={SE_TARGET} "
                 f"(N=114 headline)\n{reach_txt}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = out_dir / "adaptive_vs_random_efficiency_f20se27.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)

    curves = pd.DataFrame({"n_scenarios": targets, "adaptive_median_se": ad_sd,
                           "random_median_se": rd_sd, "adaptive_recovery_r": ad_r,
                           "random_recovery_r": rd_r})
    curves.to_csv(out_dir / "adaptive_vs_random_curves.csv", index=False)
    metrics = {
        "operating_point": {"floor": FLOOR, "se_ability_target": SE_TARGET},
        "n_headline": int(sum(1 for t in kf["adaptive"] if t["model"] not in excl)),
        "adaptive_scenarios_to_reach_se_le_0.27": ad_reach,
        "random_scenarios_to_reach_se_le_0.27": rd_reach,
        "savings_factor": (round(rd_reach / ad_reach, 2) if (ad_reach and rd_reach) else None),
        "adaptive_r_at_20": float(ad_r[19]), "random_r_at_20": float(rd_r[19]),
        "adaptive_median_se_at_20": float(ad_sd[19]), "random_median_se_at_20": float(rd_sd[19]),
        "adaptive_r_at_cap": float(ad_r[-1]), "random_r_at_cap": float(rd_r[-1]),
        "compute": "FRESH: k-fold refit rerun with a matched random-order (mode=baseline) arm",
    }
    (out_dir / "efficiency_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return {"fig": p, "metrics": metrics}


# ---------------------------------------------------------------------------
# Fig 09 - p-IRT pass-rate MAE (fresh)
# ---------------------------------------------------------------------------
def fig_pirt(kf: dict, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    excl = set(HEADLINE_EXCLUDE)
    dims = kf["dims"]
    matrix = kf["matrix"]

    # cache fold-bank arrays
    bank_cache: dict[Path, dict] = {}
    for p in set(kf["fold_bank_of_model"].values()):
        recs, _d, _ = scl.load_fitted_bank(p, kf["negative_policy"])
        bids, A, b = scl.assemble_arrays(recs, dims)
        bank_cache[p] = {"ids": bids, "A": A[:, 0], "b": b, "col": {c: i for i, c in enumerate(bids)}}

    rows = []
    for tr in kf["adaptive"]:
        m = tr["model"]
        i, _reason = G.resolve_stop(tr["sd"], tr["nscen"], FLOOR, SE_TARGET,
                                    PLATEAU_DELTA, PLATEAU_W, CAP)
        theta = float(tr["mwle"][i][0])
        bank = bank_cache[kf["fold_bank_of_model"][m]]
        mrow = matrix.loc[m]
        obs_ids = [c for c in bank["ids"] if c in mrow.index and not pd.isna(mrow[c])]
        if not obs_ids:
            continue
        idx = np.array([bank["col"][c] for c in obs_ids], int)
        A_i, b_i = bank["A"][idx], bank["b"][idx]
        y_i = mrow[obs_ids].to_numpy(float)
        p_pred = expit(A_i * theta - b_i)
        rows.append({"model": m, "theta_stop": theta, "n_criteria": int(y_i.size),
                     "pred_passrate": float(p_pred.mean()),
                     "actual_passrate": float(y_i.mean()),
                     "is_headline_excluded": m in excl})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "pirt_per_model_f20se27.csv", index=False)

    head = df[~df["is_headline_excluded"]]
    xa = head["actual_passrate"].to_numpy(float)
    yp = head["pred_passrate"].to_numpy(float)
    mae = float(np.mean(np.abs(yp - xa)))
    r = float(np.corrcoef(xa, yp)[0, 1])
    mae_all = float(np.mean(np.abs(df["pred_passrate"] - df["actual_passrate"])))

    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1, label="y = x")
    ax.scatter(xa, yp, s=28, alpha=0.75, edgecolor="k", linewidth=0.25, color=BLUE,
               label=f"headline N={xa.size} (MAE={mae:.3f}, r={r:.3f})")
    wk = df[df["is_headline_excluded"]]
    for _, row in wk.iterrows():
        ax.scatter([row["actual_passrate"]], [row["pred_passrate"]], s=95, marker="X",
                   color="red", edgecolor="k", zorder=5,
                   label=f"excl.: {row['model'].split('/')[-1]}")
    ax.set_xlabel("actual observed pass-rate")
    ax.set_ylabel("predicted pass-rate (p-IRT at CAT MWLE theta)")
    ax.set_title(f"p-IRT predicted vs actual pass-rate @ floor={FLOOR}/SE={SE_TARGET}\n"
                 f"correctness-only unidim (N=114 headline)", fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    pth = out_dir / "pirt_pred_vs_actual_f20se27.png"
    fig.savefig(pth, dpi=140, bbox_inches="tight")
    plt.close(fig)

    metrics = {
        "operating_point": {"floor": FLOOR, "se_ability_target": SE_TARGET},
        "n_headline": int(xa.size), "passrate_mae_headline": mae,
        "passrate_mae_all115": mae_all, "passrate_r_headline": r,
        "theta_source": "MWLE theta at 20/0.27 stop from FRESH adaptive trace",
        "prediction": "p = expit(A*theta - b) over each model's observed fold-bank criteria",
        "compute": "FRESH: reuses k-fold refit banks + adaptive-stop theta",
    }
    (out_dir / "pirt_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return {"fig": pth, "metrics": metrics}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bank", type=Path,
                    default=HERE / "rubrics_qmatrix_correctness_only_unidim_115_fitted.jsonl")
    ap.add_argument("--matrix", type=Path,
                    default=ROOT / "staging/response_matrix_full_nonopt_115.csv")
    ap.add_argument("--scenarios", type=Path, default=ROOT / "data/TutorBench/scenarios.jsonl")
    ap.add_argument("--per-model-cell", type=Path,
                    default=HERE / "oos_grid_full" / "oos_per_model_per_cell.csv")
    ap.add_argument("--grid-excl", type=Path,
                    default=HERE / "oos_grid_full" / "oos_per_cell_grid_excl.csv")
    ap.add_argument("--param-metrics", type=Path,
                    default=HERE / "param_uncertainty" / "metrics.json")
    ap.add_argument("--se-components", type=Path,
                    default=HERE / "param_uncertainty" / "leaderboard_se_components.csv")
    ap.add_argument("--out-dir", type=Path, default=HERE / "of_record_f20se27")
    ap.add_argument("--tmp-dir", type=Path,
                    default=ROOT / "staging" / "_eap_of_record_corr_only")
    ap.add_argument("--stop-nodes", type=int, default=321)
    ap.add_argument("--cap", type=int, default=CAP)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260729)
    ap.add_argument("--fit-grid", type=int, default=7)
    ap.add_argument("--ridge", type=float, default=1e-2)
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--negative-policy", default="clamp")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--skip-fresh", action="store_true",
                    help="only rebuild the from-CSV figures (05/07/08)")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.tmp_dir.mkdir(parents=True, exist_ok=True)
    exp = args.out_dir / "experiments"

    # ---- from-CSV inputs ----
    pm = pd.read_csv(args.per_model_cell)
    cell = pm[(pm["floor"] == FLOOR) & (pm["se_target"] == SE_TARGET)].copy()
    assert len(cell) == 115, f"expected 115 models in cell, got {len(cell)}"
    n_excl = int(cell["is_headline_excluded"].sum())
    assert n_excl == 1, f"expected 1 headline-excluded model, got {n_excl}"
    grid = pd.read_csv(args.grid_excl)
    grid_row = grid[(grid["floor"] == FLOOR) & (grid["se_target"] == SE_TARGET)].iloc[0]
    param_metrics = json.loads(args.param_metrics.read_text(encoding="utf-8"))
    se_param_median = float(param_metrics["se_components"][DIM]["median_se_param"])

    print("=== FROM-CSV figures (05 recovery, 08 leaderboard, 07 SE-components) ===")
    rec = fig_oos_recovery(cell, grid_row, exp / "05_oos_recovery")
    lb = fig_leaderboard(cell, exp / "08_leaderboard")
    se = fig_se_components(cell, se_param_median, exp / "07_parameter_uncertainty")
    print(f"  recovery: r={rec['stats']['r']:.4f} slope={rec['stats']['slope']:.4f} "
          f"theta_MAE={rec['stats']['mae']:.4f}")
    print(f"  SE bars: median SE_ability={se['median_se_ability']:.4f} "
          f"SE_total={se['median_se_total']:.4f} SE_param(offset)={se_param_median:.4f}")

    eff_m = pirt_m = None
    if not args.skip_fresh:
        print("=== FRESH k-fold run (04 efficiency + 09 p-IRT) ===")
        kf = run_kfold(args)
        eff = fig_efficiency(kf, exp / "04_efficiency_vs_random")
        pirt = fig_pirt(kf, exp / "09_pirt_mae")
        eff_m, pirt_m = eff["metrics"], pirt["metrics"]
        print(f"  efficiency: adaptive reach@{eff_m['adaptive_scenarios_to_reach_se_le_0.27']} "
              f"vs random@{eff_m['random_scenarios_to_reach_se_le_0.27']} scenarios "
              f"(x{eff_m['savings_factor']})")
        print(f"  p-IRT: pass-rate MAE={pirt_m['passrate_mae_headline']:.4f} "
              f"r={pirt_m['passrate_r_headline']:.4f}")

    # ---- top-level index + summary ----
    lb_tbl = lb["table"]
    head_tbl = lb_tbl[~lb_tbl["weak_calibrated"]]
    top = head_tbl.iloc[0]
    bot = head_tbl.iloc[-1]
    weak_rank = int(lb_tbl.index[lb_tbl["weak_calibrated"]][0]) + 1 if n_excl else None

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "STUDY / reporting only - LOCAL. Production engine + tutorbench_calibration/ "
                  "untouched. Nothing committed.",
        "deliverable": "OF-RECORD @ locked op-point floor=20 / SE_ability=0.27",
        "benchmark": "TutorBench", "scale": "correctness-only unidimensional", "dim": DIM,
        "operating_point": {"floor_min_scenarios": FLOOR, "se_ability_target": SE_TARGET,
                            "stop": "EAP-posterior 1-D marginal SD, dense 321-node grid; "
                                    "plateau delta=0.005/W=3; cap 70; MWLE theta at stop",
                            "oos": "k=5 refit-per-fold, seed 20260729"},
        "n_models": 115, "n_headline": 114,
        "headline_exclude": list(HEADLINE_EXCLUDE),
        "salamandra_7b": "rehabilitated under correctness-only unidim scale -> kept in headline",
        "headline_recovery": rec["stats"],
        "grid_row_20_0.27": rec["grid"],
        "se_components": {"median_se_ability": se["median_se_ability"],
                          "median_se_total": se["median_se_total"],
                          "median_se_param_offset": se_param_median},
        "leaderboard_top": {"model": top["model"], "theta": float(top["theta_stop_correctness"]),
                            "se_total": float(top["se_total_correctness"])},
        "leaderboard_bottom_headline": {"model": bot["model"],
                                        "theta": float(bot["theta_stop_correctness"]),
                                        "se_total": float(bot["se_total_correctness"])},
        "weak_model_rank": weak_rank,
        "efficiency_metrics": eff_m,
        "pirt_metrics": pirt_m,
        "compute_provenance": {
            "from_csv": ["05_oos_recovery", "07_parameter_uncertainty", "08_leaderboard"],
            "fresh_compute": ["04_efficiency_vs_random (random-order arm)",
                              "09_pirt_mae (predicted pass-rate)"] if not args.skip_fresh else [],
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def rel(p):
        return str(Path(p).relative_to(args.out_dir)).replace("\\", "/")

    eff_line = ("adaptive reaches median SE<=0.27 at "
                f"{eff_m['adaptive_scenarios_to_reach_se_le_0.27']} scenarios vs random at "
                f"{eff_m['random_scenarios_to_reach_se_le_0.27']} "
                f"(~{eff_m['savings_factor']}x fewer)." if eff_m else "TODO (run without --skip-fresh).")
    pirt_line = (f"pass-rate MAE = {pirt_m['passrate_mae_headline']:.3f} (r={pirt_m['passrate_r_headline']:.3f})."
                 if pirt_m else "TODO (run without --skip-fresh).")

    readme = f"""# TutorBench CORRECTNESS-ONLY UNIDIM CAT - OF-RECORD @ floor=20 / SE_ability=0.27

**correctness-only unidim, 20/0.27, N=114 headline.**

**Status:** STUDY / reporting only, LOCAL. Production engine (`tutor_cat/`,
`scripts/scenario_cat_lib.py`) and the `tutorbench_calibration/` package were NOT modified.
Nothing committed. This mirrors the 2-skill sibling numbered-experiment set for review before
making the unidim scale canonical.

Operating point: **min_scenarios (floor) = 20**, **SE_ability target = 0.27**, EAP-posterior
1-D marginal-SD stop (dense 321-node grid), plateau delta=0.005/W=3, cap 70, MWLE theta at stop.
OOS = k=5 refit-per-fold, seed 20260729. Headline **excludes only `Qwen/Qwen1.5-1.8B`**
(info-limited outlier); `salamandra-7b-instruct` is rehabilitated under this scale and kept.

## Figures

| # | figure | description | headline numbers | source |
|---|---|---|---|---|
| 05 | `{rel(rec['fig'])}` | OOS recovery: CAT MWLE theta at stop vs full-bank OOS reference theta, OLS fit + y=x | r={rec['stats']['r']:.3f}, slope={rec['stats']['slope']:.3f}, theta-MAE={rec['stats']['mae']:.3f} | from CSV |
| 04 | `{rel(exp / '04_efficiency_vs_random' / 'adaptive_vs_random_efficiency_f20se27.png')}` | adaptive vs random: (a) SE_ability vs #scenarios, (b) recovery-r vs #scenarios | {eff_line} | FRESH |
| 08 | `{rel(lb['fig'])}` | ranked per-model theta @20/0.27 with SE_total error bars, weak greyed | top {top['model'].split('/')[-1]} theta={top['theta_stop_correctness']:.2f}; bottom (headline) {bot['model'].split('/')[-1]} theta={bot['theta_stop_correctness']:.2f} | from CSV |
| 09 | `{rel(exp / '09_pirt_mae' / 'pirt_pred_vs_actual_f20se27.png')}` | p-IRT predicted pass-rate (at CAT theta) vs actual observed pass-rate | {pirt_line} | FRESH |
| 07 | `{rel(se['fig'])}` | SE_ability vs SE_total grouped bar, IQR whiskers, 0.27 target line, SE_param gap | SE_ability={se['median_se_ability']:.3f}, SE_total={se['median_se_total']:.3f}, SE_param={se_param_median:.3f} | from CSV |

## Headline numbers

- **OOS recovery (N=114):** r = **{rec['stats']['r']:.3f}**, slope = **{rec['stats']['slope']:.3f}**,
  theta-MAE = **{rec['stats']['mae']:.3f}** (grid row 20/0.27: r={rec['grid']['r']:.3f},
  slope={rec['grid']['slope']:.3f}, theta-MAE={rec['grid']['theta_mae']:.3f}).
- **Length:** median {rec['grid']['median_len']:.0f}, mean {rec['grid']['mean_len']:.2f} scenarios.
- **SE components (median, N=114):** SE_ability = **{se['median_se_ability']:.3f}**,
  SE_total = **{se['median_se_total']:.3f}**, fixed SE_param offset = **{se_param_median:.3f}**.
- **Adaptive vs random:** {eff_line}
- **p-IRT:** {pirt_line}
- **Leaderboard:** top headline model **{top['model']}** (theta={top['theta_stop_correctness']:.3f},
  SE_total={top['se_total_correctness']:.3f}); bottom headline **{bot['model']}**
  (theta={bot['theta_stop_correctness']:.3f}). `Qwen/Qwen1.5-1.8B` (weak, greyed) ranks {weak_rank}/115.

## Files

- `experiments/05_oos_recovery/` - scatter, `oos_per_model_f20se27.csv`, `recovery_metrics.json`.
- `experiments/04_efficiency_vs_random/` - efficiency fig, `adaptive_vs_random_curves.csv`, `efficiency_metrics.json`.
- `experiments/08_leaderboard/` - leaderboard fig + `leaderboard_f20se27.csv`.
- `experiments/09_pirt_mae/` - scatter, `pirt_per_model_f20se27.csv`, `pirt_metrics.json`.
- `experiments/07_parameter_uncertainty/` - SE bars + `se_ability_vs_total_summary.csv`.
- `summary.json` - machine-readable op-point / config / all headline numbers.

Source inputs (reused, not recomputed for 05/07/08): `../oos_grid_full/oos_per_model_per_cell.csv`
(row 20/0.27), `../oos_grid_full/oos_per_cell_grid_excl.csv`, `../param_uncertainty/metrics.json`.
Fresh computes (04/09) rerun the k-fold machinery from
`../oos_grid_unidim_correctness_only.py` (adaptive + matched random-order arm; p-IRT prediction).
"""
    (args.out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"\n[write] {args.out_dir}")
    print(f"  top headline: {top['model']} theta={top['theta_stop_correctness']:.3f}")
    print(f"  bottom headline: {bot['model']} theta={bot['theta_stop_correctness']:.3f}")
    print(f"  weak {HEADLINE_EXCLUDE[0]} rank {weak_rank}/115")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
