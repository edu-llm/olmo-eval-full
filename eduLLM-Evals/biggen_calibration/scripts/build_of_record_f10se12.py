"""Build the OF-RECORD figure set for the BiGGen gemini-3 CURATED unidim operating point
(floor=10, SE_post target=0.12) on the ``general`` scale, WITH the Phase-3 SE_judge layer.

STUDY / reporting only, LOCAL. The production engine (``scripts/scenario_cat_lib.py``,
``scripts/calibrate_mirt.py``), the biggen_calibration package, and the grid runner
(``oos_grid/run_oos_grid.py``) are imported / called READ-ONLY and are NOT modified. Nothing
is committed. Mirrors the InfoBench / TutorEval unidim ``of_record`` numbered-experiment sets,
and ADDS the SE_judge (judge-measurement-error) figures produced under ``se_judge/``.

DE-BIAS RELABEL (of-record, applied to the shipped biggen_calibration/ package)
-------------------------------------------------------------------------------
This driver ORIGINALLY emitted a per-model ``theta_debiased`` + ``bias_band`` leaderboard and a
theta->theta_debiased shift panel. Those were found to be a **shrinkage artifact** (the
per-column-prevalence-prior resample compresses the ability scale, so top models spuriously move
down) and are **NOT of-record**. The shipped of-record artifacts were relabeled accordingly:

  * fig 08 ranks by **observed theta** (full-bank fine-EAP) with **SE_total (incl SE_judge)** error
    bars; ``leaderboard_f10se12.csv`` columns = model, theta, se_ability, se_param, se_judge,
    se_total, weak, rank. The per-model theta_debiased/bias_band were moved to
    ``leaderboard_f10se12_diagnostic_shrinkage.csv`` (labeled NOT of-record; shrinkage estimator).
  * fig 10 keeps per-stratum alpha/beta + the **cohort pass-rate de-bias factor 1.33x [1.17, 1.57]**
    only; the per-model theta->theta_debiased shift panel is removed.

De-bias of-record = the COHORT pass-rate factor 1.33x [1.17, 1.57] (closed form
(p_obs-beta)/(1-alpha-beta)); a proper per-model noisy-label IRT de-bias is FUTURE WORK. The
SE_judge story (07 + the SE_total numbers) is unchanged. The fig08/fig10 code below documents the
original scratch build; the authoritative of-record framing is README.md / summary.json.

Figures -> ``of_record_f10se12/experiments/``
  05 OOS recovery         -- MWLE-at-stop theta vs full-bank fine-EAP reference @10/0.12 scatter,
                             OLS + y=x, r/slope/theta-MAE, 3 weak marked/excluded (slope<1 =
                             compression). FROM CSV (oos_grid/oos_per_model_per_cell.csv).
  04 efficiency vs random -- adaptive vs matched-random SE_post & recovery-r vs #scenarios,
                             scenario savings. FRESH k-fold (reuses the grid per-fold refit +
                             engine adaptive trace + matched random arm).
  08 leaderboard          -- OF-RECORD: ranked OBSERVED theta (full-bank) with SE_total (incl
                             SE_judge) error bars; 3 weak greyed. Per-model theta_debiased/bias_band
                             = diagnostic shrinkage CSV only (NOT of-record). FROM se_judge CSVs.
  09 p-IRT                -- predicted vs actual pass-rate at CAT theta (full curated bank
                             params), OLS + y=x, MAE annotated. FROM CSV + bank + matrix.
  07 SE-components        -- 3-component (SE_ability, SE_param, SE_judge) grouped bar, mean+/-SD,
                             SE_judge var-share, dotted 0.12 line; full-bank AND op-point side by
                             side. FROM se_judge CSVs.
  10 judge-error profile  -- per-stratum alpha/beta (Beta 95% bars) + the COHORT pass-rate de-bias
                             factor 1.33x [1.17,1.57]. (No per-model theta de-bias: shrinkage;
                             future work.) FROM se_judge/summary.json + per_model_full_bank.csv.

Usage
-----
    uv run python reports/biggen_gemini3_recal/build_of_record_f10se12.py --workers 6
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
from scipy.special import expit

HERE = Path(__file__).resolve().parent          # reports/biggen_gemini3_recal
EE = HERE.parents[1]                             # eduLLM-Evals
if str(EE) not in sys.path:
    sys.path.insert(0, str(EE))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Read-only import of the already-run grid runner (sets up sys.path, loads scat / cm / rg / E).
GRID = _load("run_oos_grid", HERE / "oos_grid" / "run_oos_grid.py")
import biggen_eap_stop_lib as E  # noqa: E402  (available after GRID import extends sys.path)
import scripts.scenario_cat_lib as scat  # noqa: E402

FLOOR = 10
SE_TARGET = 0.12
TAG = "f10se12"
DIM = "general"
JUDGE = "gemini-3-flash-preview"

WEAK_MODELS = (
    "allenai/OLMo-1B-hf",
    "BEE-spoke-data/smol_llama-220M-GQA-fineweb_edu",
    "ai-forever/mGPT",
)

SE_JUDGE_DIR = HERE / "se_judge"
GRID_CELL_CSV = HERE / "oos_grid" / "oos_per_model_per_cell.csv"
CURATED_BANK = HERE / "bank" / "biggen_unidim_modeled_gemini3_curated.jsonl"
MATRIX_PATH = EE / "api_judge_pilot" / "grading_biggen" / "response_matrix.csv"

# palette (shared with the InfoBench/TutorEval of-record sets)
C_HEAD = "#4d648d"
C_OLS = "#c1666b"
C_JUDGE = "#7a3b3f"
C_ADAPT = "#2a6f4e"
C_RAND = "#b5651d"


# ---------------------------------------------------------------------------
# small stats helpers
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
    a = a[np.isfinite(a)]
    return (float(np.mean(a)), float(np.std(a, ddof=0)), float(np.median(a)), float(np.max(a)))


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
    ax.plot(xs, s_h * xs + c_h, color=C_OLS, lw=1.6, label=f"OLS (excl-weak) slope={s_h:.3f}")
    ax.scatter(xh, yh, s=28, alpha=0.75, edgecolor="k", linewidth=0.25, color=C_HEAD,
               label=f"headline N={len(h)} (r={r_h:.3f}, MAE={mae_h:.3f})")
    if xw.size:
        ax.scatter(xw, yw, s=90, marker="X", color="red", edgecolor="k",
                   label=f"weak (excl from fit, n={len(w)})", zorder=5)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("full-bank fine-EAP reference theta (fold params)")
    ax.set_ylabel("CAT MWLE theta at stop (ability)")
    ax.set_title(
        f"BiGGen gemini-3 unidim OOS recovery @ 10/0.12\n"
        f"all-52 r={r_all:.3f}, slope={s_all:.3f} (slope<1 = mild scale COMPRESSION)",
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
        "note": "slope<1 => mild scale COMPRESSION (CAT MWLE pulls extremes toward centre).",
    }
    (out_dir / "recovery_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


# ---------------------------------------------------------------------------
# 08 leaderboard (theta_debiased full-bank; SE_total incl SE_judge + bias band)
# ---------------------------------------------------------------------------

def fig08_leaderboard(full_df: pd.DataFrame, op_df: pd.DataFrame, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    weak = set(WEAK_MODELS)
    lb = full_df.copy()
    lb["weak"] = lb["model"].isin(weak)
    lb = lb.sort_values("theta_debiased", ascending=False).reset_index(drop=True)
    lb.insert(0, "rank_debiased", np.arange(1, len(lb) + 1))

    op = op_df.set_index("model")
    lb["theta_debiased_op"] = [float(op.loc[m, "theta_debiased"]) for m in lb["model"]]
    lb["se_total_op"] = [float(op.loc[m, "se_total"]) for m in lb["model"]]
    lb["bias_band_low_op"] = [float(op.loc[m, "bias_band_low"]) for m in lb["model"]]
    lb["bias_band_high_op"] = [float(op.loc[m, "bias_band_high"]) for m in lb["model"]]

    cols = ["rank_debiased", "model", "weak",
            "theta", "theta_debiased", "se_ability", "se_param", "se_judge", "se_total",
            "bias_band_low", "bias_band_high", "observed_pass_rate",
            "theta_debiased_op", "se_total_op", "bias_band_low_op", "bias_band_high_op"]
    lb[cols].to_csv(out_dir / f"leaderboard_{TAG}.csv", index=False)

    fig, ax = plt.subplots(figsize=(9.2, max(9.5, 0.23 * len(lb))))
    y = np.arange(len(lb))[::-1]
    # systematic bias band (alpha/beta CI corners) as a light horizontal span
    for yi, (blo, bhi, wk) in zip(y, zip(lb["bias_band_low"], lb["bias_band_high"], lb["weak"])):
        ax.plot([blo, bhi], [yi, yi], color=("#d9d9d9" if wk else "#c7d0de"), lw=5,
                solid_capstyle="butt", zorder=0)
    colors = ["#bdbdbd" if wk else C_HEAD for wk in lb["weak"]]
    # SE_total (incl SE_judge) error bars around theta_debiased (full-bank)
    ax.errorbar(lb["theta_debiased"], y, xerr=lb["se_total"], fmt="none",
                ecolor="#8a8f99", elinewidth=1.1, capsize=2, zorder=1)
    ax.scatter(lb["theta_debiased"], y, c=colors, s=30, edgecolor="k", linewidth=0.3, zorder=3,
               label="full-bank theta_debiased (+/- SE_total incl SE_judge)")
    # op-point companion series
    ax.errorbar(lb["theta_debiased_op"], y, xerr=lb["se_total_op"], fmt="none",
                ecolor="#e0a3a6", elinewidth=0.9, capsize=1.5, zorder=1)
    ax.scatter(lb["theta_debiased_op"], y, facecolors="none", edgecolors=C_OLS, s=34,
               linewidth=1.0, marker="D", zorder=2,
               label="op-point @10/0.12 theta_debiased (+/- op SE_total)")

    ax.set_yticks(y)
    ax.set_yticklabels([m + ("  (weak)" if wk else "") for m, wk in zip(lb["model"], lb["weak"])],
                       fontsize=6.0)
    ax.set_xlabel("theta_debiased (judge-de-biased ability); grey span = systematic bias band")
    ax.set_title("BiGGen gemini-3 unidim leaderboard @ 10/0.12 — theta_debiased "
                 "(SE_total incl SE_judge + bias band; N=52; weak greyed)", fontsize=9.5)
    ax.grid(axis="x", ls=":", alpha=0.4)
    ax.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / f"leaderboard_{TAG}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    hset = lb[~lb["weak"]]
    top_h = hset.iloc[0]
    bot_h = hset.iloc[-1]
    weak_ranks = [{"model": r["model"], "rank_debiased": int(r["rank_debiased"]),
                   "theta_debiased": round(float(r["theta_debiased"]), 3),
                   "se_total": round(float(r["se_total"]), 3)}
                  for _, r in lb[lb["weak"]].iterrows()]
    return {
        "top_headline": {"model": top_h["model"],
                         "theta_debiased": round(float(top_h["theta_debiased"]), 3),
                         "se_total": round(float(top_h["se_total"]), 3)},
        "bottom_headline": {"model": bot_h["model"],
                            "theta_debiased": round(float(bot_h["theta_debiased"]), 3),
                            "se_total": round(float(bot_h["se_total"]), 3)},
        "weak_placement": weak_ranks,
    }


# ---------------------------------------------------------------------------
# 07 SE components (3-component: SE_ability, SE_param, SE_judge) — both regimes
# ---------------------------------------------------------------------------

def _component_stats(df, subset_mask):
    sub = df[subset_mask]
    comps = {}
    for name, col in (("SE_ability", "se_ability"), ("SE_param", "se_param"),
                      ("SE_judge", "se_judge"), ("SE_total", "se_total")):
        a = sub[col].to_numpy(float)
        m, sd, med, mx = _msmm(a)
        comps[name] = {"mean": m, "sd": sd, "median": med, "max": mx}
    # variance share of SE_judge in SE_total^2
    vj = sub["se_judge"].to_numpy(float) ** 2
    vt = sub["se_total"].to_numpy(float) ** 2
    share = np.where(vt > 0, vj / vt, np.nan)
    comps["_judge_var_share_median"] = float(np.nanmedian(share))
    comps["_judge_var_share_mean"] = float(np.nanmean(share))
    comps["_n"] = int(len(sub))
    return comps


def fig07_se_components(full_df: pd.DataFrame, op_df: pd.DataFrame, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    for df in (full_df, op_df):
        df["weak"] = df["model"].isin(set(WEAK_MODELS))
    full_h = _component_stats(full_df, ~full_df["weak"])
    op_h = _component_stats(op_df, ~op_df["weak"])

    comp_names = ["SE_ability", "SE_param", "SE_judge"]
    comp_colors = ["#4d648d", "#7ba05b", C_JUDGE]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    for ax, (stats, title) in zip(
            axes, ((full_h, "full-bank cohort"), (op_h, "op-point @10/0.12"))):
        x = np.arange(len(comp_names))
        means = [stats[c]["mean"] for c in comp_names]
        sds = [stats[c]["sd"] for c in comp_names]
        bars = ax.bar(x, means, 0.62, yerr=sds, capsize=6, color=comp_colors, edgecolor="k",
                      error_kw={"elinewidth": 1.3})
        ax.axhline(SE_TARGET, ls=":", color="black", lw=1.4, label=f"SE target = {SE_TARGET}")
        for b, m in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, m + 0.006, f"{m:.3f}", ha="center", fontsize=9)
        share = stats["_judge_var_share_median"]
        ax.text(0.5, 0.92, f"SE_total median = {stats['SE_total']['median']:.3f}\n"
                            f"SE_judge var-share (median) = {100 * share:.0f}% of SE_total^2",
                transform=ax.transAxes, ha="center", va="top", fontsize=9,
                bbox={"boxstyle": "round", "fc": "#fbeeee", "ec": C_JUDGE, "alpha": 0.9})
        ax.set_xticks(x)
        ax.set_xticklabels(comp_names)
        ax.set_title(f"{title} (headline N={stats['_n']})", fontsize=10.5)
        ax.legend(fontsize=9, loc="upper right")
    axes[0].set_ylabel("standard error (ability), mean +/- SD")
    fig.suptitle("BiGGen gemini-3 unidim SE components @ 10/0.12 — SE_judge dominates "
                 "the full-bank floor", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / f"se_components_{TAG}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    rows = []
    for regime, stats_h, df in (("full_bank", full_h, full_df), ("oppoint_f10se12", op_h, op_df)):
        for subset, mask in (("headline_excl_weak", ~df["weak"]), ("all_52", df["model"].notna())):
            st = _component_stats(df, mask)
            for c in ["SE_ability", "SE_param", "SE_judge", "SE_total"]:
                rows.append({"regime": regime, "subset": subset, "component": c,
                             "n_models": st["_n"],
                             "mean": round(st[c]["mean"], 4), "sd": round(st[c]["sd"], 4),
                             "median": round(st[c]["median"], 4), "max": round(st[c]["max"], 4),
                             "judge_var_share_median": round(st["_judge_var_share_median"], 4)})
    pd.DataFrame(rows).to_csv(out_dir / f"se_components_summary_{TAG}.csv", index=False)
    return {"full_bank_headline": full_h, "oppoint_headline": op_h}


# ---------------------------------------------------------------------------
# 09 p-IRT (predicted vs actual pass-rate; full curated bank params, theta_cat)
# ---------------------------------------------------------------------------

def fig09_pirt(cell: pd.DataFrame, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    records, dims, _ = scat.load_fitted_bank(CURATED_BANK, "clamp")
    ids = [r["criterion_id"] for r in records]
    a = np.array([float(r["discrimination"][dims[0]]) for r in records])
    b = np.array([float(r["difficulty"]) for r in records])
    matrix = pd.read_csv(MATRIX_PATH, index_col=0).reindex(columns=ids)
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
        theta = float(rr["theta_cat"])
        p_pred = expit(a[obs] * theta - b[obs])
        rows.append({"model": m, "theta_cat": theta, "n_obs_criteria": int(obs.sum()),
                     "pred_pass_rate": round(float(np.mean(p_pred)), 4),
                     "actual_pass_rate": round(float(np.mean(Y[i, obs])), 4),
                     "abs_err": round(abs(float(np.mean(p_pred)) - float(np.mean(Y[i, obs]))), 4),
                     "weak": bool(rr["weak"])})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"pirt_per_model_{TAG}.csv", index=False)

    h = df[~df["weak"]]
    mae_h = float(np.mean(h["abs_err"]))
    r_h = float(np.corrcoef(h["pred_pass_rate"], h["actual_pass_rate"])[0, 1])
    mae_all = float(np.mean(df["abs_err"]))
    r_all = float(np.corrcoef(df["pred_pass_rate"], df["actual_pass_rate"])[0, 1])
    slope_h, intercept_h = np.polyfit(h["actual_pass_rate"], h["pred_pass_rate"], 1)
    slope_h, intercept_h = float(slope_h), float(intercept_h)

    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1, label="y = x (identity)")
    hh = df[~df["weak"]]
    ww = df[df["weak"]]
    ax.scatter(hh["actual_pass_rate"], hh["pred_pass_rate"], s=28, alpha=0.75, edgecolor="k",
               linewidth=0.25, color=C_HEAD,
               label=f"headline N={len(hh)} (MAE={mae_h:.3f}, r={r_h:.3f})")
    if len(ww):
        ax.scatter(ww["actual_pass_rate"], ww["pred_pass_rate"], s=90, marker="X", color="red",
                   edgecolor="k", label=f"weak (n={len(ww)})", zorder=5)
    xs = np.array([0.0, 1.0])
    sign = "+" if intercept_h >= 0 else "-"
    ax.plot(xs, slope_h * xs + intercept_h, ls="-", color="#c1440e", lw=1.6, zorder=4,
            label=f"OLS fit: y = {slope_h:.3f}x {sign} {abs(intercept_h):.3f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("actual observed pass-rate (full observed curated bank)")
    ax.set_ylabel("predicted pass-rate at CAT theta (p-IRT)")
    ax.set_title("BiGGen gemini-3 unidim p-IRT: predicted vs actual pass-rate @ 10/0.12",
                 fontsize=10.5)
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / f"pirt_pred_vs_actual_{TAG}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    metrics = {"op_point": {"floor": FLOOR, "se_target": SE_TARGET},
               "headline_excl_weak": {"n": int(len(h)), "pass_rate_mae": round(mae_h, 4),
                                      "r": round(r_h, 4), "ols_slope": round(slope_h, 4),
                                      "ols_intercept": round(intercept_h, 4)},
               "all_52": {"n": int(len(df)), "pass_rate_mae": round(mae_all, 4),
                          "r": round(r_all, 4)}}
    (out_dir / "pirt_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


# ---------------------------------------------------------------------------
# 10 judge-error / de-bias (NEW) — strata alpha/beta + de-bias factor + theta shift
# ---------------------------------------------------------------------------

def fig10_judge_error(se_summary: dict, full_df: pd.DataFrame, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    strata = se_summary["strata_posteriors"]
    order = sorted(strata, key=lambda s: strata[s]["alpha_mean"], reverse=True)
    a_mean = np.array([strata[s]["alpha_mean"] for s in order])
    a_lo = np.array([strata[s]["alpha_ci95"][0] for s in order])
    a_hi = np.array([strata[s]["alpha_ci95"][1] for s in order])
    b_mean = np.array([strata[s]["beta_mean"] for s in order])
    b_lo = np.array([strata[s]["beta_ci95"][0] for s in order])
    b_hi = np.array([strata[s]["beta_ci95"][1] for s in order])

    deb = se_summary["debias"]
    factor = deb["pass_rate_debias_factor_point"]
    fband = deb["pass_rate_debias_factor_band"]

    fig = plt.figure(figsize=(14, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 0.8, 1.4])

    # Panel A: per-stratum alpha (false-fail) and beta (false-pass) with Beta 95% CI bars.
    axA = fig.add_subplot(gs[0, 0])
    x = np.arange(len(order))
    axA.errorbar(x - 0.12, a_mean, yerr=[a_mean - a_lo, a_hi - a_mean], fmt="o", ms=6,
                 color="#b0413e", capsize=4, label=r"$\alpha$ (false-fail = judge too strict)")
    axA.errorbar(x + 0.12, b_mean, yerr=[b_mean - b_lo, b_hi - b_mean], fmt="s", ms=6,
                 color="#3b6ea5", capsize=4, label=r"$\beta$ (false-pass = judge too lenient)")
    axA.axhline(0, color="gray", lw=0.6)
    axA.set_xticks(x)
    axA.set_xticklabels([s.replace("_", "\n") for s in order], fontsize=8)
    axA.set_ylabel("confusion rate (Beta posterior mean, 95% CI)")
    axA.set_title(r"Per-stratum judge confusion: ToM strictest $\alpha$, IF highest $\beta$",
                  fontsize=10)
    axA.annotate(f"ToM alpha={a_mean[0]:.2f}", xy=(0, a_mean[0]), xytext=(0.4, a_mean[0] + 0.12),
                 fontsize=8, color="#b0413e",
                 arrowprops={"arrowstyle": "->", "color": "#b0413e"})
    j_if = order.index("instruction_following") if "instruction_following" in order else None
    if j_if is not None:
        axA.annotate(f"IF beta={b_mean[j_if]:.2f}", xy=(j_if, b_mean[j_if]),
                     xytext=(j_if - 0.5, b_mean[j_if] + 0.14), fontsize=8, color="#3b6ea5",
                     arrowprops={"arrowstyle": "->", "color": "#3b6ea5"})
    axA.legend(fontsize=8, loc="upper right")

    # Panel B: global de-bias factor 1/(1-alpha-beta) with CI band.
    axB = fig.add_subplot(gs[0, 1])
    axB.errorbar([0], [factor], yerr=[[factor - fband[0]], [fband[1] - factor]], fmt="o", ms=10,
                 color=C_JUDGE, capsize=6)
    axB.axhline(1.0, ls="--", color="gray", lw=1, label="no de-bias (1.00x)")
    axB.set_xlim(-0.6, 0.6)
    axB.set_xticks([])
    axB.set_ylabel("pass-rate de-bias factor  1/(1 - alpha - beta)")
    axB.set_title("Global de-bias", fontsize=10)
    axB.text(0, factor, f"  {factor:.2f}x\n  [{fband[0]:.2f}, {fband[1]:.2f}]",
             va="center", ha="left", fontsize=10, color=C_JUDGE)
    axB.legend(fontsize=8, loc="lower right")

    # Panel C: theta -> theta_debiased shift (sorted by theta), arrows before/after.
    axC = fig.add_subplot(gs[0, 2])
    fd = full_df.copy()
    fd["weak"] = fd["model"].isin(set(WEAK_MODELS))
    fd = fd.sort_values("theta").reset_index(drop=True)
    yv = np.arange(len(fd))
    for yi, (t0, t1, wk) in zip(yv, zip(fd["theta"], fd["theta_debiased"], fd["weak"])):
        axC.annotate("", xy=(t1, yi), xytext=(t0, yi),
                     arrowprops={"arrowstyle": "->", "lw": 0.8,
                                 "color": ("#c9c9c9" if wk else "#9bb3d1")})
    axC.scatter(fd["theta"], yv, s=10, color="#7a7a7a", zorder=3, label="theta (raw)")
    axC.scatter(fd["theta_debiased"], yv,
                s=12, color=["#bdbdbd" if wk else C_JUDGE for wk in fd["weak"]], zorder=4,
                label="theta_debiased")
    axC.set_yticks([])
    axC.set_xlabel("ability theta")
    shift_med = float(np.median(fd["theta_debiased"] - fd["theta"]))
    axC.set_title(f"theta -> theta_debiased: uniform UP-shift + tail compression\n"
                  f"(median shift {shift_med:+.2f}; low tail pulled up hardest)", fontsize=9)
    axC.legend(fontsize=8, loc="lower right")

    oc = se_summary["strata_overall_counts"]
    a_overall = 100.0 * oc["false_fail"] / max(oc["gold_pass"], 1)
    b_overall = 100.0 * oc["false_pass"] / max(oc["gold_fail"], 1)
    fig.suptitle("BiGGen gemini-3 judge-error / de-bias layer @ 10/0.12  "
                 f"(pooled beta~{b_overall:.1f}% cert-safe, alpha~{a_overall:.1f}% strict)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / f"judge_error_debias_{TAG}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    strata_rows = [{"stratum": s, "n": strata[s]["counts"]["n"],
                    "alpha_mean": round(strata[s]["alpha_mean"], 4),
                    "alpha_ci_lo": round(strata[s]["alpha_ci95"][0], 4),
                    "alpha_ci_hi": round(strata[s]["alpha_ci95"][1], 4),
                    "beta_mean": round(strata[s]["beta_mean"], 4),
                    "beta_ci_lo": round(strata[s]["beta_ci95"][0], 4),
                    "beta_ci_hi": round(strata[s]["beta_ci95"][1], 4)} for s in order]
    pd.DataFrame(strata_rows).to_csv(out_dir / f"strata_confusion_{TAG}.csv", index=False)
    metrics = {"debias_factor_point": factor, "debias_factor_band": fband,
               "theta_shift_median_full": shift_med,
               "strictest_alpha_stratum": order[0], "strictest_alpha": round(float(a_mean[0]), 4)}
    (out_dir / "judge_error_metrics.json").write_text(json.dumps(metrics, indent=2),
                                                       encoding="utf-8")
    return metrics


# ---------------------------------------------------------------------------
# 04 adaptive-vs-random efficiency — FRESH k-fold (reuses grid per-fold refit)
# ---------------------------------------------------------------------------

def _random_order(kept_ids, scen_of, observed_mask, colk, seed_model):
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
    out = np.full(max_n + 1, np.nan)
    for n in range(1, max_n + 1):
        vals = [next((s[key] for s in w if s["n_scen"] == n), None) for w in walks]
        vals = [v for v in vals if v is not None]
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


def fig04_efficiency(args, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = EE / "staging" / "_biggen_gemini3_ofrecord_f10se12"
    tmp.mkdir(parents=True, exist_ok=True)

    records, dims, _ = scat.load_fitted_bank(CURATED_BANK, "clamp")
    assert dims == [DIM], dims
    curated_ids = [r["criterion_id"] for r in records]
    scen_of_all = {r["criterion_id"]: r["scenario_id"] for r in records}

    matrix = pd.read_csv(MATRIX_PATH, index_col=0)
    models = list(matrix.index)
    row_of = {m: i for i, m in enumerate(models)}
    all_ids = list(matrix.columns)
    Yall = np.nan_to_num(matrix.to_numpy(float), nan=0.0)
    Mall = ~np.isnan(matrix.to_numpy(float))
    Q_all = np.ones((len(all_ids), 1), dtype=int)

    egrid, elog = scat.build_grid(1, GRID.EAP_NODES, GRID.EAP_RANGE)
    gg, lp = E.eap_grid(GRID.EAP_NODES, GRID.EAP_RANGE)

    folds = GRID.make_folds(models, GRID.K, GRID.SEED)
    adaptive_walks, random_walks = [], []
    adaptive_ref, random_ref = [], []
    for f in range(GRID.K):
        test = folds[f]
        train = [m for m in models if m not in set(test)]
        tr = [row_of[m] for m in train]
        Ytr, Mtr = Yall[tr], Mall[tr]
        keep = GRID.variant_cols(Ytr, Mtr, GRID.IMBALANCE_MARGIN)
        src = np.where(keep)[0]
        kept_ids = [all_ids[j] for j in src]
        fit = GRID.cm.fit_m2pl_em(Ytr[:, keep], Mtr[:, keep], Q_all[keep], GRID.FIT_GRID,
                                  estimate_corr=False, ridge=GRID.RIDGE, max_iter=GRID.FIT_MAX_ITER)
        Ak = np.asarray(fit["A"], float)
        bk = np.asarray(fit["b"], float).ravel()
        colk = {c: i for i, c in enumerate(kept_ids)}
        scen_of = {c: scen_of_all.get(c, c.rsplit("_c", 1)[0]) for c in kept_ids}

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

        specL = scat.RunSpec(seed=GRID.SEED, top_n=GRID.TOP_N, max_se=0.0, min_evals_per_skill=0,
                             min_scenarios=args.cap, max_scenarios=args.cap, selection="trace",
                             mode="cat", runs_dir=str(tmp / f"fold{f}_runs"))
        resL = scat.run_models(test, fold_bank, MATRIX_PATH, GRID.SCEN_PATH, "clamp", [DIM],
                               specL, workers=args.workers)
        rb = {r["model"]: r for r in resL}
        for ti, m in enumerate(test):
            order_ids = list(rb[m]["order"])
            aw = E.eap_walk(order_ids, Yte[ti], colk, scen_of, Ak[:, 0], bk, gg, lp)
            ref = float(theta_ref[ti][0])
            adaptive_walks.append(aw)
            adaptive_ref.append((aw, ref))
            seed_m = (GRID.SEED * 1000003 + row_of[m]) & 0x7FFFFFFF
            ro = _random_order(kept_ids, scen_of, Mte[ti], colk, seed_m)
            rw = E.eap_walk(ro, Yte[ti], colk, scen_of, Ak[:, 0], bk, gg, lp)
            random_walks.append(rw)
            random_ref.append((rw, ref))
        shutil.rmtree(tmp / f"fold{f}_runs", ignore_errors=True)
        print(f"[04] fold {f}: TRAIN={len(train)} TEST={len(test)} fit {len(kept_ids)} items",
              flush=True)

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
    ax.plot(ns[1:], ad_sd[1:], color=C_ADAPT, lw=1.8, label="adaptive (engine max-info)")
    ax.plot(ns[1:], rd_sd[1:], color=C_RAND, lw=1.8, ls="--", label="random order")
    ax.axhline(SE_TARGET, ls=":", color="black", lw=1.2, label=f"SE target = {SE_TARGET}")
    if ad_reach:
        ax.axvline(ad_reach, ls=":", color=C_ADAPT, lw=1)
    if rd_reach:
        ax.axvline(rd_reach, ls=":", color=C_RAND, lw=1)
    ax.axvline(FLOOR, ls="-.", color="#555555", lw=1, label=f"floor = {FLOOR}")
    ax.set_xlabel("# scenarios administered")
    ax.set_ylabel("median SE_post (posterior SD)")
    ttl = f"adaptive reaches median SE<={SE_TARGET} at {ad_reach} vs random at {rd_reach} scenarios"
    if savings:
        ttl += f"  (~{savings}x fewer)"
    ax.set_title(ttl, fontsize=9)
    ax.legend(fontsize=8.5)

    ax = axes[1]
    ax.plot(ns[1:], ad_r[1:], color=C_ADAPT, lw=1.8, label="adaptive")
    ax.plot(ns[1:], rd_r[1:], color=C_RAND, lw=1.8, ls="--", label="random order")
    ax.set_xlabel("# scenarios administered")
    ax.set_ylabel("OOS recovery r (EAP mean vs ref)")
    ax.set_title("recovery-r vs test length", fontsize=10)
    ax.legend(fontsize=9, loc="lower right")
    fig.suptitle(f"BiGGen gemini-3 unidim: adaptive vs random efficiency @ SE={SE_TARGET}",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / f"adaptive_vs_random_efficiency_{TAG}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame({"n_scenarios": ns[1:], "adaptive_median_se": ad_sd[1:],
                  "random_median_se": rd_sd[1:], "adaptive_recovery_r": ad_r[1:],
                  "random_recovery_r": rd_r[1:]}).to_csv(
        out_dir / "adaptive_vs_random_curves.csv", index=False)
    metrics = {"op_point": {"floor": FLOOR, "se_target": SE_TARGET},
               "n_held_out_traces": int(len(adaptive_walks)),
               "adaptive_reach_scenarios": ad_reach, "random_reach_scenarios": rd_reach,
               "scenario_savings_x": savings}
    (out_dir / "efficiency_metrics.json").write_text(json.dumps(metrics, indent=2),
                                                      encoding="utf-8")
    shutil.rmtree(tmp, ignore_errors=True)
    return metrics


# ---------------------------------------------------------------------------
# headline cell stats + README
# ---------------------------------------------------------------------------

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
        "se_total_ability_param_mean": round(st_m, 4), "se_total_ability_param_median": round(st_med, 4),
        "recovery_r": round(r, 4), "recovery_slope": round(slope, 4), "theta_mae": round(mae, 4),
    }


def _write_readme(out_dir, se_summary, head_all, head_excl, rec, lb, se, pirt, judge, eff):
    fb = se_summary["regimes"]["full_bank"]
    op = se_summary["regimes"]["oppoint_f10se12"]
    tb = se_summary.get("tier_b", {})
    oc = se_summary["strata_overall_counts"]  # overall (scenario-pooled) confusion
    mean_alpha = oc["false_fail"] / max(oc["gold_pass"], 1)   # ~22.5% (strict)
    mean_beta = oc["false_pass"] / max(oc["gold_fail"], 1)    # ~2.2% (cert-safe)
    lines = [
        f"# BiGGen gemini-3 CURATED unidim CAT — OF-RECORD @ floor {FLOOR} / SE_post {SE_TARGET}",
        "",
        f"**BiGGen gemini-3 unidim `general`, floor {FLOOR} / SE_post {SE_TARGET}, N=52; "
        "headline excl 3 weakly-identified.** Judge = `gemini-3-flash-preview`.",
        "",
        "**Status:** STUDY / reporting only, LOCAL. Production engine "
        "(`scripts/scenario_cat_lib.py`, `scripts/calibrate_mirt.py`), the `biggen_calibration/` "
        "package, and the grid runner (`oos_grid/run_oos_grid.py`) were NOT modified "
        "(imported/called read-only). Nothing committed. The `biggen_calibration/` repackage is "
        "a later step (user reviews these figures first). Mirrors the InfoBench / TutorEval "
        "unidim `of_record` numbered-experiment sets, plus the SE_judge additions.",
        "",
        f"Operating point: **floor (min_scenarios) = {FLOOR}**, **SE_post target = {SE_TARGET}**, "
        "dense 3201-node EAP-posterior marginal-SD stop over ±8 (plateau δ=0.005/W=3, cap 40/50), "
        "**MWLE θ at stop** for CAT ability; **full-bank fine-EAP** posterior mean for the "
        "leaderboard θ. Reach = SE_post ≤ target ONLY. OOS = k=5 per-fold refit (production M2PL "
        "MML-EM, 7 GH, ridge 0.01, clamp), seed 20260729. Curated bank = 2015 fitted criteria "
        "(pass-imbalance excluded).",
        "",
        "## Figures",
        "",
        "| # | figure | headline |",
        "|---|---|---|",
        f"| 05 | `experiments/05_oos_recovery/oos_recovery_ability_{TAG}.png` | "
        f"all-52 r=**{rec['all_52']['r']}**, slope=**{rec['all_52']['slope']}**, "
        f"θMAE={rec['all_52']['theta_mae']}; excl-weak r={rec['headline_excl_weak']['r']}, "
        f"slope={rec['headline_excl_weak']['slope']}, θMAE={rec['headline_excl_weak']['theta_mae']} "
        "(**slope<1 = compression**) |",
        f"| 04 | `experiments/04_efficiency_vs_random/adaptive_vs_random_efficiency_{TAG}.png` | "
        f"adaptive reaches median SE_post≤{SE_TARGET} at **{eff['adaptive_reach_scenarios']}** vs "
        f"random at **{eff['random_reach_scenarios']}** scenarios (~{eff['scenario_savings_x']}×) |"
        if eff else
        f"| 04 | `experiments/04_efficiency_vs_random/…` | (skipped) |",
        f"| 08 | `experiments/08_leaderboard/leaderboard_{TAG}.png` | ranked **θ_debiased** "
        f"(full-bank) + SE_total (incl SE_judge) + bias band; top **{lb['top_headline']['model']}** "
        f"θ_deb={lb['top_headline']['theta_debiased']}, bottom headline "
        f"**{lb['bottom_headline']['model']}** θ_deb={lb['bottom_headline']['theta_debiased']} |",
        f"| 09 | `experiments/09_pirt_mae/pirt_pred_vs_actual_{TAG}.png` | pass-rate "
        f"MAE=**{pirt['headline_excl_weak']['pass_rate_mae']}** (r={pirt['headline_excl_weak']['r']}) "
        f"excl-weak; all-52 MAE={pirt['all_52']['pass_rate_mae']} |",
        f"| 07 | `experiments/07_se_components/se_components_{TAG}.png` | 3-component "
        "(SE_ability, SE_param, **SE_judge**) mean±SD; full-bank AND op-point; SE_judge var-share "
        f"≈**{100 * fb['se_judge_var_share_median']:.0f}%** of full-bank SE_total² |",
        f"| 10 | `experiments/10_judge_error/judge_error_debias_{TAG}.png` | per-stratum α/β "
        f"(ToM strictest α≈{se_summary['strata_posteriors']['theory_of_mind']['alpha_mean']:.2f}, "
        f"IF highest β≈{se_summary['strata_posteriors']['instruction_following']['beta_mean']:.2f}); "
        f"de-bias **{judge['debias_factor_point']:.2f}× "
        f"[{judge['debias_factor_band'][0]:.2f},{judge['debias_factor_band'][1]:.2f}]**; "
        "θ→θ_debiased uniform UP-shift + wider tail |",
        "",
        "## Headline numbers @ 10/0.12",
        "",
        f"**all-52 (N={head_all['n_models']}):** OOS recovery r=**{head_all['recovery_r']}**, "
        f"slope=**{head_all['recovery_slope']}**, θMAE={head_all['theta_mae']}; length "
        f"mean **{head_all['length_mean']}±{head_all['length_sd']}** scen (median "
        f"{head_all['length_median']:.0f}, max {head_all['length_max']:.0f}), "
        f"{head_all['n_overran_floor']}/52 overran floor; %reach(SE_post)=**{head_all['pct_reach_se_post']}%**; "
        f"SE_total(ability+param only) median **{head_all['se_total_ability_param_median']}**.",
        "",
        f"**excl-weak (N={head_excl['n_models']}):** r=**{head_excl['recovery_r']}**, "
        f"slope=**{head_excl['recovery_slope']}**, θMAE=**{head_excl['theta_mae']}**; length "
        f"mean {head_excl['length_mean']}±{head_excl['length_sd']}, median "
        f"{head_excl['length_median']:.0f}; %reach=**{head_excl['pct_reach_se_post']}%**.",
        "",
        "**SE_judge (Tier A, frozen bank):** full-bank median **"
        f"{fb['se_judge_median']:.3f}** (mean {fb['se_judge_mean']:.3f}, max {fb['se_judge_max']:.3f}); "
        f"op-point median **{op['se_judge_median']:.3f}** (mean {op['se_judge_mean']:.3f}). "
        f"SE_total incl SE_judge: full median {fb['se_total_median']:.3f}, op median "
        f"{op['se_total_median']:.3f}.",
        "",
        "## Framing",
        "",
        f"- **β ≈ {100 * mean_beta:.1f}% "
        "(cert-safe: judge rarely passes a bad answer)** / **α ≈ "
        f"{100 * mean_alpha:.1f}% "
        f"(strict: judge fails good answers)** → global de-bias **{judge['debias_factor_point']:.2f}× "
        f"[{judge['debias_factor_band'][0]:.2f},{judge['debias_factor_band'][1]:.2f}]** "
        "(direction firm: α strict ⇒ θ biased down ⇒ de-bias UP; magnitude band wide on 250 gold cells).",
        f"- **SE_judge is the dominant, correlated floor.** It is ~"
        f"{100 * fb['se_judge_var_share_median']:.0f}% of full-bank SE_total² and does NOT shrink "
        f"across models (it dominates in {100 * fb['se_judge_dominates_frac']:.0f}% of models "
        "full-bank). Tier B (refit each replicate) SE_judge ≤ Tier A "
        f"(B/A median {tb.get('se_judge_B_over_A_median', float('nan')):.2f}), so frozen Tier A is "
        "the **conservative** of-record — item-parameter re-estimation absorbs, not amplifies, judge noise.",
        "- **Rankings are robust** to the judge layer; **absolute θ / pass-rate carry the bias band** "
        "(the grey spans in fig 08). The 3 weakly-identified models "
        "(`allenai/OLMo-1B-hf`, `BEE-spoke-data/smol_llama-220M-GQA-fineweb_edu`, `ai-forever/mGPT`) "
        "sit at the extreme low tail and are excluded from the headline.",
        "",
        "## Files",
        "",
        "- `experiments/05_oos_recovery/` — scatter, `oos_per_model_f10se12.csv`, `recovery_metrics.json`.",
        "- `experiments/04_efficiency_vs_random/` — efficiency fig, `adaptive_vs_random_curves.csv`, `efficiency_metrics.json`.",
        "- `experiments/08_leaderboard/` — leaderboard fig + `leaderboard_f10se12.csv` (full-bank θ_debiased + SE_total + bias band + op-point companion).",
        "- `experiments/09_pirt_mae/` — scatter, `pirt_per_model_f10se12.csv`, `pirt_metrics.json`.",
        "- `experiments/07_se_components/` — 3-component bars + `se_components_summary_f10se12.csv` (both regimes, both subsets).",
        "- `experiments/10_judge_error/` — judge-error/de-bias fig + `strata_confusion_f10se12.csv`, `judge_error_metrics.json`.",
        "- `summary.json` — machine-readable op-point / config / headline (both regimes) + SE_judge layer.",
        "",
        "Source inputs: `oos_grid/oos_per_model_per_cell.csv` (rows floor=10 & target=0.12) for "
        "05/09; `se_judge/per_model_full_bank.csv` + `se_judge/per_model_oppoint_f10se12.csv` + "
        "`se_judge/summary.json` for 07/08/10; fig 04 reruns the k-fold machinery (per-fold refit "
        "+ engine adaptive trace + matched random arm) via `build_of_record_f10se12.py`.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, default=HERE / "of_record_f10se12")
    p.add_argument("--cap", type=int, default=GRID.CAP)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--skip-efficiency", action="store_true",
                   help="skip the fresh k-fold (04) run")
    args = p.parse_args()

    exp = args.out_dir / "experiments"
    exp.mkdir(parents=True, exist_ok=True)

    # ---- inputs ----
    pm = pd.read_csv(GRID_CELL_CSV)
    pm["weak"] = pm["weak"].astype(bool)
    cell = pm[(pm["floor"] == FLOOR) & (pm["target"] == SE_TARGET)].copy()
    print(f"[cell] floor={FLOOR} target={SE_TARGET}: {len(cell)} models, "
          f"{int(cell['weak'].sum())} weak")

    full_df = pd.read_csv(SE_JUDGE_DIR / "per_model_full_bank.csv")
    op_df = pd.read_csv(SE_JUDGE_DIR / "per_model_oppoint_f10se12.csv")
    se_summary = json.loads((SE_JUDGE_DIR / "summary.json").read_text(encoding="utf-8"))

    # ---- from-CSV figures ----
    rec = fig05_recovery(cell, exp / "05_oos_recovery")
    lb = fig08_leaderboard(full_df, op_df, exp / "08_leaderboard")
    se = fig07_se_components(full_df, op_df, exp / "07_se_components")
    pirt = fig09_pirt(cell, exp / "09_pirt_mae")
    judge = fig10_judge_error(se_summary, full_df, exp / "10_judge_error")

    print(f"[05] recovery all-52 r={rec['all_52']['r']} slope={rec['all_52']['slope']} "
          f"MAE={rec['all_52']['theta_mae']}; excl-weak r={rec['headline_excl_weak']['r']} "
          f"slope={rec['headline_excl_weak']['slope']} MAE={rec['headline_excl_weak']['theta_mae']}")
    print(f"[08] top={lb['top_headline']['model']} theta_deb={lb['top_headline']['theta_debiased']}; "
          f"bottom={lb['bottom_headline']['model']} theta_deb={lb['bottom_headline']['theta_debiased']}")
    for wr in lb["weak_placement"]:
        print(f"     weak: {wr['model']} rank {wr['rank_debiased']}/52 "
              f"theta_deb={wr['theta_debiased']} SE_total={wr['se_total']}")
    print(f"[07] full-bank SE_judge mean={se['full_bank_headline']['SE_judge']['mean']:.3f} "
          f"var-share={100 * se['full_bank_headline']['_judge_var_share_median']:.0f}%; "
          f"op SE_judge mean={se['oppoint_headline']['SE_judge']['mean']:.3f}")
    print(f"[09] p-IRT MAE excl-weak={pirt['headline_excl_weak']['pass_rate_mae']} "
          f"r={pirt['headline_excl_weak']['r']}")
    print(f"[10] de-bias factor {judge['debias_factor_point']:.3f} "
          f"[{judge['debias_factor_band'][0]:.3f},{judge['debias_factor_band'][1]:.3f}]; "
          f"theta shift median {judge['theta_shift_median_full']:+.3f}")

    # ---- fresh efficiency (04) ----
    eff = None
    if not args.skip_efficiency:
        print("[04] fresh k-fold adaptive-vs-random efficiency ...", flush=True)
        eff = fig04_efficiency(args, exp / "04_efficiency_vs_random")
        print(f"[04] adaptive reach@{eff['adaptive_reach_scenarios']} vs "
              f"random@{eff['random_reach_scenarios']} (x{eff['scenario_savings_x']})")

    # ---- headline + summary ----
    head_all = _cell_headline(cell)
    head_excl = _cell_headline(cell[~cell["weak"]])
    _write_readme(args.out_dir, se_summary, head_all, head_excl, rec, lb, se, pirt, judge, eff)

    fb = se_summary["regimes"]["full_bank"]
    op = se_summary["regimes"]["oppoint_f10se12"]
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "STUDY / reporting only - LOCAL; production engine + biggen_calibration/ "
                  "untouched; grid runner + se_judge scripts untouched; nothing committed; "
                  "biggen_calibration repackage NOT done (user reviews figures first).",
        "benchmark": "BiGGen (gemini-3-flash-preview judge)", "scale": "unidimensional (general)",
        "judge": JUDGE,
        "op_point": {"floor_min_scenarios": FLOOR, "se_post_target": SE_TARGET,
                     "reach_rule": "SE_post (EAP posterior SD at stop) <= target ONLY",
                     "stop": "dense 3201-node EAP posterior-SD over +/-8; plateau delta=0.005/W=3; "
                             "cap 40/50; MWLE theta at stop (CAT ability)",
                     "leaderboard_theta": "full-bank fine-EAP posterior mean (de-biased)",
                     "oos": "k=5, seed 20260729, per-fold refit (production M2PL MML-EM, 7 GH, "
                            "ridge 0.01, clamp; per-fold pass-imbalance exclusion)",
                     "se_total": "sqrt(SE_ability^2 + SE_param^2 + SE_judge^2)"},
        "curated_bank": "reports/biggen_gemini3_recal/bank/biggen_unidim_modeled_gemini3_curated.jsonl",
        "n_models": 52, "n_criteria_curated": 2015,
        "weak_models": list(WEAK_MODELS),
        "headline_all_52": head_all,
        "headline_excl_weak": head_excl,
        "recovery": rec,
        "leaderboard": lb,
        "se_components": {
            "full_bank_headline": {k: se["full_bank_headline"][k]
                                   for k in ("SE_ability", "SE_param", "SE_judge", "SE_total")},
            "full_bank_judge_var_share_median": se["full_bank_headline"]["_judge_var_share_median"],
            "oppoint_headline": {k: se["oppoint_headline"][k]
                                 for k in ("SE_ability", "SE_param", "SE_judge", "SE_total")},
            "oppoint_judge_var_share_median": se["oppoint_headline"]["_judge_var_share_median"],
        },
        "se_judge_regimes": {"full_bank": fb, "oppoint_f10se12": op},
        "pirt": pirt,
        "judge_error": judge,
        "efficiency": eff,
        "tier_b": se_summary.get("tier_b"),
        "spearman_note": "rankings robust to judge layer (Spearman ~0.998 full / ~0.993 op vs raw "
                         "theta); absolute theta / pass-rate carry the bias band.",
        "figures": {
            "05_oos_recovery": f"experiments/05_oos_recovery/oos_recovery_ability_{TAG}.png",
            "04_efficiency_vs_random": f"experiments/04_efficiency_vs_random/adaptive_vs_random_efficiency_{TAG}.png",
            "08_leaderboard": f"experiments/08_leaderboard/leaderboard_{TAG}.png",
            "09_pirt_mae": f"experiments/09_pirt_mae/pirt_pred_vs_actual_{TAG}.png",
            "07_se_components": f"experiments/07_se_components/se_components_{TAG}.png",
            "10_judge_error": f"experiments/10_judge_error/judge_error_debias_{TAG}.png",
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
