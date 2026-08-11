"""Build the OF-RECORD figure set for the TutorEval gemini-3 CURATED unidim operating point
(floor=15, SE_ability target=0.25) on the ``ability`` scale, WITH the Phase-3 SE_judge layer.

STUDY / reporting only, LOCAL. The production engine (``tutor_cat/``,
``scripts/scenario_cat_lib.py``, ``scripts/calibrate_mirt.py``), the ``tutoreval_calibration/``
package and the grid runner (``oppoint_grid/run_oppoint_grid_gemini3.py``) are imported / called
READ-ONLY and are NOT modified. Nothing is committed. Mirrors the BiGGen / InfoBench unidim
``of_record`` numbered-experiment sets, and ADDS the SE_judge (judge-measurement-error) figures.

De-bias / SE_judge publication convention = HYBRID (locked):
  * Op-point of-record leaderboard (08): publish **observed theta** (op-point MWLE-at-stop) with
    **SE_total error bars** (SE_total = sqrt(SE_ability^2 + SE_param^2 + SE_judge^2); SE_judge is
    the dominant floor), the 3 weak models greyed, and an annotated **1.83x cohort pass-rate
    de-bias caveat** (posterior CI 1.33-2.97x). NOT theta_debiased.
  * SECONDARY diagnostic (11): full-bank per-model theta_debiased BAND from the CLEAN full-bank
    regime (Spearman 0.993, gate passes, 0 violations). Clearly labeled secondary / NOT the
    of-record headline; per-model op-point theta_debiased is directional-only for the weak tail.

Figures -> ``<out-dir>/experiments/``
  05 OOS recovery         -- MWLE-at-stop theta vs full-bank reference @15/0.25 scatter,
                             OLS + y=x, r/slope/theta-MAE, 3 weak marked/excluded (slope<1 =
                             compression). FROM CSV (oppoint_grid/oos_per_model_per_cell.csv).
  04 efficiency vs random -- adaptive vs matched-random SE_ability & recovery-r vs #scenarios,
                             scenario savings. FRESH k-fold (reuses the grid per-fold refit +
                             engine adaptive trace + matched random arm).
  07 SE-components        -- 3-component (SE_ability, SE_param, SE_judge) grouped bar, mean+/-SD,
                             SE_judge var-share; full-bank AND op-point side by side.
  08 leaderboard (HYBRID) -- ranked OBSERVED op-point theta (MWLE-at-stop) with SE_total error
                             bars (ability/param/judge breakdown) + annotated cohort de-bias
                             caveat; 3 weak greyed. CSV + fig.
  09 p-IRT                -- predicted vs actual pass-rate at CAT theta (full curated bank
                             params), OLS + y=x, MAE annotated. FROM CSV + bank + matrix.
  10 judge-error / de-bias-- per-stratum alpha/beta (Beta 95% bars): conceptual_understanding
                             OWN alpha, quant+unknown pooled to overall alpha, beta=0; and the
                             cohort pass-rate de-bias factor 1.83x [1.33,2.97]. NO per-model
                             theta-shift plot (op-point per-model shift is not precise).
  11 debias band (SECOND) -- full-bank per-model theta_debiased band (clean regime); labeled
                             SECONDARY / NOT of-record headline.

Usage
-----
    uv run python reports/tutoreval_gemini3_recal/build_of_record_f15se25.py --workers 6
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
from scipy.special import expit, logsumexp

HERE = Path(__file__).resolve().parent          # reports/tutoreval_gemini3_recal
EE = HERE.parents[1]                             # eduLLM-Evals
if str(EE) not in sys.path:
    sys.path.insert(0, str(EE))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Read-only import of the canonical OOS-grid harness (fold machinery + eap_walk/stop_point).
GRID = _load("run_oos_grid_unidim",
             EE / "tutoreval_calibration" / "scripts" / "run_oos_grid_unidim.py")
import scripts.scenario_cat_lib as scat  # noqa: E402

FLOOR = 15
SE_TARGET = 0.25
TAG = "f15se25"
DIM = "ability"
JUDGE = "gemini-3-flash-preview"

WEAK_MODELS = (
    "BEE-spoke-data/smol_llama-220M-GQA-fineweb_edu",
    "ai-forever/mGPT",
    "allenai/OLMo-1B-hf",
)

SE_JUDGE_DIR = HERE / "se_judge"
GRID_CELL_CSV = HERE / "oppoint_grid" / "oos_per_model_per_cell.csv"
CURATED_BANK = HERE / "bank" / "rubrics_qmatrix_final_unidim_gemini3_fitted.jsonl"
MATRIX_PATH = EE / "api_judge_pilot" / "grading_tutoreval" / "response_matrix.csv"
SCEN_PATH = EE / "data" / "TutorEval" / "scenarios_final.jsonl"

# palette (shared with the BiGGen/InfoBench of-record sets)
C_HEAD = "#4d648d"
C_OLS = "#c1666b"
C_JUDGE = "#7a3b3f"
C_ADAPT = "#2a6f4e"
C_RAND = "#b5651d"

DENSE_NODES = 321
DENSE_RANGE = 8.0
REF_GRID = 61
REF_RANGE = 6.0
K = 5
SEED = 20260729
FIT_GRID = 7
RIDGE = 1e-2
FIT_MAX_ITER = 200


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
# 05 OOS recovery (from CSV) -- MWLE-at-stop theta vs full-bank reference
# ---------------------------------------------------------------------------

def fig05_recovery(cell: pd.DataFrame, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    cell = cell.rename(columns={"theta_mwle": "theta_cat"})
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
    ax.set_xlabel("full-bank EAP reference theta (fold params)")
    ax.set_ylabel("CAT MWLE theta at stop (ability)")
    ax.set_title(
        f"TutorEval gemini-3 unidim OOS recovery @ 15/0.25\n"
        f"excl-weak r={r_h:.3f}, slope={s_h:.3f} (slope<1 = mild scale COMPRESSION)",
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
# 08 leaderboard (HYBRID) -- OBSERVED op-point theta + SE_total (incl SE_judge) error bars
# ---------------------------------------------------------------------------

def fig08_leaderboard(op_df: pd.DataFrame, debias_factor, debias_band, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    weak = set(WEAK_MODELS)
    lb = op_df.copy().drop(columns=["rank"], errors="ignore")
    lb["weak"] = lb["model"].isin(weak)
    # observed op-point theta (MWLE-at-stop) is the 'theta' column of per_model_oppoint
    lb = lb.sort_values("theta", ascending=False).reset_index(drop=True)
    lb.insert(0, "rank", np.arange(1, len(lb) + 1))

    cols = ["rank", "model", "theta", "se_ability", "se_param", "se_judge", "se_total", "weak"]
    lb[cols].to_csv(out_dir / f"leaderboard_{TAG}.csv", index=False)

    fig, ax = plt.subplots(figsize=(9.2, max(9.5, 0.23 * len(lb))))
    y = np.arange(len(lb))[::-1]
    colors = ["#bdbdbd" if wk else C_HEAD for wk in lb["weak"]]
    # SE_total (incl SE_judge, the dominant floor) error bars around OBSERVED theta
    ax.errorbar(lb["theta"], y, xerr=lb["se_total"], fmt="none",
                ecolor="#8a8f99", elinewidth=1.1, capsize=2, zorder=1)
    ax.scatter(lb["theta"], y, c=colors, s=32, edgecolor="k", linewidth=0.3, zorder=3,
               label="observed op-point theta (MWLE-at-stop) +/- SE_total (incl SE_judge)")
    ax.set_yticks(y)
    ax.set_yticklabels([m + ("  (weak)" if wk else "") for m, wk in zip(lb["model"], lb["weak"])],
                       fontsize=6.0)
    ax.set_xlabel("observed ability theta @ 15/0.25 (op-point MWLE-at-stop; error bars = SE_total)")
    ax.set_title("TutorEval gemini-3 unidim OF-RECORD leaderboard @ 15/0.25 — observed theta "
                 "(SE_total incl SE_judge; N=52; weak greyed)", fontsize=9.5)
    ax.grid(axis="x", ls=":", alpha=0.4)
    ax.legend(fontsize=7.5, loc="lower right")
    # annotated cohort de-bias caveat (NOT applied per-model in the headline)
    ax.text(0.015, 0.015,
            f"Cohort de-bias caveat: judge (gemini-3) alpha≈0.45 strict, beta=0 ⇒ observed theta\n"
            f"is biased DOWN. Aggregate pass-rate de-bias factor 1/(1-alpha) = {debias_factor:.2f}× "
            f"[{debias_band[0]:.2f}, {debias_band[1]:.2f}] (cohort; posterior CI).\n"
            f"Direction firm (UP, no crossover); per-model correction is the SECONDARY full-bank "
            f"band (fig 11),\nNOT applied to these headline points.",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=7.2,
            bbox={"boxstyle": "round", "fc": "#fbeeee", "ec": C_JUDGE, "alpha": 0.92})
    fig.tight_layout()
    fig.savefig(out_dir / f"leaderboard_{TAG}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    hset = lb[~lb["weak"]]
    top_h = hset.iloc[0]
    bot_h = hset.iloc[-1]
    weak_rows = [{"model": r["model"], "rank": int(r["rank"]),
                  "theta": round(float(r["theta"]), 3), "se_total": round(float(r["se_total"]), 3)}
                 for _, r in lb[lb["weak"]].iterrows()]
    return {
        "convention": "HYBRID: observed op-point MWLE-at-stop theta + SE_total (incl SE_judge) "
                      "error bars + annotated 1.83x cohort de-bias caveat (NOT per-model debiased).",
        "top_headline": {"model": top_h["model"], "theta": round(float(top_h["theta"]), 3),
                         "se_total": round(float(top_h["se_total"]), 3)},
        "bottom_headline": {"model": bot_h["model"], "theta": round(float(bot_h["theta"]), 3),
                            "se_total": round(float(bot_h["se_total"]), 3)},
        "weak_placement": weak_rows,
        "cohort_debias_factor_point": debias_factor,
        "cohort_debias_factor_band": debias_band,
    }


# ---------------------------------------------------------------------------
# 07 SE components (3-component: SE_ability, SE_param, SE_judge) -- both regimes
# ---------------------------------------------------------------------------

def _component_stats(df, subset_mask):
    sub = df[subset_mask]
    comps = {}
    for name, col in (("SE_ability", "se_ability"), ("SE_param", "se_param"),
                      ("SE_judge", "se_judge"), ("SE_total", "se_total")):
        a = sub[col].to_numpy(float)
        m, sd, med, mx = _msmm(a)
        comps[name] = {"mean": m, "sd": sd, "median": med, "max": mx}
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
            axes, ((full_h, "full-bank cohort"), (op_h, "op-point @15/0.25"))):
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
    fig.suptitle("TutorEval gemini-3 unidim SE components @ 15/0.25 — SE_judge is the DOMINANT "
                 "floor", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / f"se_components_{TAG}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    rows = []
    for regime, df in (("full_bank", full_df), ("oppoint_f15se25", op_df)):
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
# 09 p-IRT (predicted vs actual pass-rate; full curated bank params, theta_mwle)
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
        theta = float(rr["theta_mwle"])
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
    ax.set_title("TutorEval gemini-3 unidim p-IRT: predicted vs actual pass-rate @ 15/0.25",
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
# 10 judge-error / de-bias -- strata alpha/beta + cohort de-bias factor (NO theta-shift panel)
# ---------------------------------------------------------------------------

def fig10_judge_error(se_summary: dict, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    cm = se_summary["confusion_model"]
    ap = cm["alpha_posteriors_beta_from_gold_jeffreys"]
    overall = cm["overall"]

    # Two strata bars: conceptual_understanding OWN alpha, and the pooled OVERALL alpha
    # (quantitative_procedural + unknown pool to overall). beta pinned to 0 (no false-pass).
    strata = [
        ("conceptual_understanding\n(own alpha)", ap["own_alpha_mean"], ap["own_alpha_ci95"],
         ap["own_counts"]["n"]),
        ("quant+unknown -> pooled\n(overall alpha)", ap["overall_alpha_mean"],
         ap["overall_alpha_ci95"], overall["n"]),
    ]

    gate = se_summary["de_bias_gate"]["pass_rate_debias_factor"]
    factor = gate["point_1_over_1_minus_alpha"]
    fband = gate["posterior_ci95"]

    fig = plt.figure(figsize=(12.5, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 0.9])

    # Panel A: per-stratum alpha (false-fail) + beta (=0) with Beta 95% CI bars.
    axA = fig.add_subplot(gs[0, 0])
    x = np.arange(len(strata))
    a_mean = np.array([s[1] for s in strata])
    a_lo = np.array([s[2][0] for s in strata])
    a_hi = np.array([s[2][1] for s in strata])
    axA.errorbar(x - 0.11, a_mean, yerr=[a_mean - a_lo, a_hi - a_mean], fmt="o", ms=8,
                 color="#b0413e", capsize=5, label=r"$\alpha$ (false-fail = judge too strict)")
    axA.errorbar(x + 0.11, np.zeros(len(strata)), yerr=0, fmt="s", ms=8,
                 color="#3b6ea5", capsize=5,
                 label=r"$\beta$ (false-pass = judge too lenient) = 0")
    axA.axhline(0, color="gray", lw=0.6)
    axA.set_xticks(x)
    axA.set_xticklabels([s[0] + f"\nn={s[3]}" for s in strata], fontsize=8.5)
    axA.set_ylim(-0.05, 0.85)
    axA.set_ylabel("confusion rate (Beta posterior mean, 95% CI)")
    axA.set_title(r"Per-stratum judge confusion (gemini-3): strict $\alpha\approx0.45$, "
                  r"$\beta=0$ (cert-safe)", fontsize=10.5)
    for xi, m in zip(x, a_mean):
        axA.annotate(f"alpha={m:.3f}", xy=(xi - 0.11, m), xytext=(xi - 0.05, m + 0.08),
                     fontsize=8, color="#b0413e")
    axA.legend(fontsize=8.5, loc="upper right")

    # Panel B: cohort de-bias factor 1/(1-alpha) with CI band (beta=0 => purely UP).
    axB = fig.add_subplot(gs[0, 1])
    axB.errorbar([0], [factor], yerr=[[factor - fband[0]], [fband[1] - factor]], fmt="o", ms=12,
                 color=C_JUDGE, capsize=7)
    axB.axhline(1.0, ls="--", color="gray", lw=1, label="no de-bias (1.00x)")
    axB.set_xlim(-0.6, 0.6)
    axB.set_xticks([])
    axB.set_ylabel("cohort pass-rate de-bias factor  1/(1 - alpha)")
    axB.set_title("Cohort de-bias (UP; beta=0)", fontsize=10.5)
    axB.text(0.04, factor, f"  {factor:.2f}x\n  [{fband[0]:.2f}, {fband[1]:.2f}]",
             va="center", ha="left", fontsize=11, color=C_JUDGE)
    axB.legend(fontsize=8.5, loc="lower right")

    oc = overall
    a_overall = 100.0 * oc["false_fail"] / max(oc["gold_pass"], 1)
    b_overall = 100.0 * oc["false_pass"] / max(oc["gold_fail"], 1)
    fig.suptitle("TutorEval gemini-3 judge-error / de-bias layer @ 15/0.25  "
                 f"(overall alpha~{a_overall:.0f}% strict, beta~{b_overall:.0f}% on "
                 f"{oc['n']} gold cells)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / f"judge_error_debias_{TAG}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    strata_rows = [{"stratum": s[0].split("\n")[0], "n": s[3],
                    "alpha_mean": round(s[1], 4), "alpha_ci_lo": round(s[2][0], 4),
                    "alpha_ci_hi": round(s[2][1], 4), "beta_mean": 0.0} for s in strata]
    pd.DataFrame(strata_rows).to_csv(out_dir / f"strata_confusion_{TAG}.csv", index=False)
    metrics = {"debias_factor_point": factor, "debias_factor_band": fband,
               "debias_posterior_median": gate["posterior_median"],
               "own_stratum": ap["own_stratum"], "own_alpha": round(ap["own_alpha_mean"], 4),
               "overall_alpha": round(ap["overall_alpha_mean"], 4), "beta": 0.0,
               "note": "beta=0 => de-bias is purely UP: p_true = p_obs/(1-alpha)."}
    (out_dir / "judge_error_metrics.json").write_text(json.dumps(metrics, indent=2),
                                                       encoding="utf-8")
    return metrics


# ---------------------------------------------------------------------------
# 11 SECONDARY -- full-bank per-model theta_debiased BAND (clean regime, NOT of-record headline)
# ---------------------------------------------------------------------------

def fig11_debias_band(full_df: pd.DataFrame, gate: dict, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    fd = full_df.copy()
    fd["weak"] = fd["model"].isin(set(WEAK_MODELS))
    fd = fd.sort_values("theta_debiased", ascending=False).reset_index(drop=True)
    fd.insert(0, "rank_debiased", np.arange(1, len(fd) + 1))

    cols = ["rank_debiased", "model", "weak", "theta", "theta_debiased",
            "bias_band_low", "bias_band_high", "se_ability", "se_param", "se_judge", "se_total",
            "observed_pass_rate"]
    fd[cols].to_csv(out_dir / f"debias_band_full_bank_{TAG}.csv", index=False)

    fig, ax = plt.subplots(figsize=(9.2, max(9.5, 0.23 * len(fd))))
    y = np.arange(len(fd))[::-1]
    for yi, (blo, bhi, wk) in zip(y, zip(fd["bias_band_low"], fd["bias_band_high"], fd["weak"])):
        ax.plot([blo, bhi], [yi, yi], color=("#d9d9d9" if wk else "#c7d0de"), lw=5,
                solid_capstyle="butt", zorder=0)
    colors = ["#bdbdbd" if wk else C_JUDGE for wk in fd["weak"]]
    ax.scatter(fd["theta_debiased"], y, c=colors, s=30, edgecolor="k", linewidth=0.3, zorder=3,
               label="full-bank theta_debiased (grey span = alpha-CI bias band)")
    ax.scatter(fd["theta"], y, facecolors="none", edgecolors="#7a7a7a", s=26, linewidth=0.9,
               marker="o", zorder=2, label="full-bank observed theta (raw)")
    ax.set_yticks(y)
    ax.set_yticklabels([m + ("  (weak)" if wk else "") for m, wk in zip(fd["model"], fd["weak"])],
                       fontsize=6.0)
    ax.set_xlabel("theta_debiased (full-bank; judge-de-biased ability); grey span = bias band")
    sp = gate["full_bank"]["spearman_obs_vs_debiased"]
    ax.set_title("SECONDARY DIAGNOSTIC — full-bank per-model theta_debiased band "
                 f"(clean regime, Spearman {sp:.3f}; 0 violations)\n"
                 "NOT the of-record headline; per-model op-point theta_debiased is "
                 "directional-only for the weak tail", fontsize=8.6)
    ax.grid(axis="x", ls=":", alpha=0.4)
    ax.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / f"debias_band_full_bank_{TAG}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    return {"regime": "full_bank", "label": "SECONDARY / NOT of-record headline",
            "spearman_obs_vs_debiased": sp,
            "n_violations": gate["full_bank"]["n_violations"],
            "shift_median": gate["full_bank"]["shift_median"],
            "note": "per-model op-point theta_debiased is directional-only (op-point Spearman "
                    f"{gate['oppoint']['spearman_obs_vs_debiased']:.3f}; weak-tail bands span "
                    "2-3 theta units)."}


# ---------------------------------------------------------------------------
# 04 adaptive-vs-random efficiency -- FRESH k-fold (reuses grid per-fold refit)
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
    tmp = EE / "staging" / "_tutoreval_gemini3_ofrecord_f15se25"
    tmp.mkdir(parents=True, exist_ok=True)

    records, dims, _ = scat.load_fitted_bank(CURATED_BANK, "clamp")
    assert dims == [DIM], dims
    all_ids = [r["criterion_id"] for r in records]
    scen_of_all = {r["criterion_id"]: r["scenario_id"] for r in records}
    crit_of_all = {r["criterion_id"]: r.get("criterion", "") for r in records}

    matrix = pd.read_csv(MATRIX_PATH, index_col=0).reindex(columns=all_ids)
    models = list(matrix.index)
    row_of = {m: i for i, m in enumerate(models)}
    Yall = np.nan_to_num(matrix.to_numpy(float), nan=0.0)
    Mall = ~np.isnan(matrix.to_numpy(float))
    Q_all = np.ones((len(all_ids), 1), dtype=int)

    egrid, elog = scat.build_grid(1, REF_GRID, REF_RANGE)
    gg = np.linspace(-DENSE_RANGE, DENSE_RANGE, DENSE_NODES)
    lp = -0.5 * gg ** 2
    lp = lp - logsumexp(lp)

    folds = GRID.make_folds(models, K, SEED)
    adaptive_walks, random_walks = [], []
    adaptive_ref, random_ref = [], []
    for f in range(K):
        test = folds[f]
        train = [m for m in models if m not in set(test)]
        tr = [row_of[m] for m in train]
        Ytr, Mtr = Yall[tr], Mall[tr]
        keep = np.array([j for j in range(len(all_ids))
                         if Mtr[:, j].sum() >= 2
                         and 0 < Ytr[Mtr[:, j], j].sum() < Mtr[:, j].sum()], dtype=int)
        kept_ids = [all_ids[j] for j in keep]
        fit = GRID.cm.fit_m2pl_em(Ytr[:, keep], Mtr[:, keep], Q_all[keep], FIT_GRID,
                                  ridge=RIDGE, max_iter=FIT_MAX_ITER)
        Ak = np.asarray(fit["A"], float)
        bk = np.asarray(fit["b"], float).ravel()
        colk = {c: i for i, c in enumerate(kept_ids)}
        scen_of = {c: scen_of_all[c] for c in kept_ids}

        fold_bank = tmp / f"fold{f}_bank.jsonl"
        with fold_bank.open("w", encoding="utf-8") as fh:
            for jj, cid in enumerate(kept_ids):
                fh.write(json.dumps({
                    "criterion_id": cid, "scenario_id": scen_of[cid],
                    "criterion": crit_of_all.get(cid, ""),
                    "discrimination": {DIM: float(Ak[jj, 0])}, "q_modeled": {DIM: 1},
                    "difficulty": float(bk[jj])}) + "\n")

        subte = matrix.loc[test].reindex(columns=kept_ids)
        Yte = np.nan_to_num(subte.to_numpy(float), nan=0.0)
        Mte = ~np.isnan(subte.to_numpy(float))
        theta_ref = scat.eap_all_models(Yte, Mte, Ak, bk, egrid, elog)

        rb = GRID.run_forced(test, fold_bank, MATRIX_PATH, SCEN_PATH, [DIM], SEED,
                             args.cap, args.workers, tmp / f"fold{f}_runs")
        for ti, m in enumerate(test):
            order_ids = list(rb[m]["order"])
            aw = GRID.eap_walk(order_ids, Yte[ti], colk, scen_of, Ak[:, 0], bk, gg, lp)
            ref = float(theta_ref[ti][0])
            adaptive_walks.append(aw)
            adaptive_ref.append((aw, ref))
            seed_m = (SEED * 1000003 + row_of[m]) & 0x7FFFFFFF
            ro = _random_order(kept_ids, scen_of, Mte[ti], colk, seed_m)
            rw = GRID.eap_walk(ro, Yte[ti], colk, scen_of, Ak[:, 0], bk, gg, lp)
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
    ax.set_ylabel("median SE_ability (posterior SD)")
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
    fig.suptitle(f"TutorEval gemini-3 unidim: adaptive vs random efficiency @ SE={SE_TARGET}",
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

def _cell_headline(cell, op_df):
    op = op_df.set_index("model")
    sub = cell.rename(columns={"theta_mwle": "theta_cat"})
    ls = np.array(sub["n_scenarios"], float)
    r, slope, _c, mae = _ols(sub["theta_ref"], sub["theta_cat"])
    reach = np.array([bool(s <= SE_TARGET) for s in sub["sd"]], float)
    st = np.array([float(op.loc[m, "se_total"]) if m in op.index else np.nan
                   for m in sub["model"]], float)
    len_m, len_sd, len_med, len_max = _msmm(ls)
    _, _, st_med, _ = _msmm(st)
    return {
        "n_models": int(len(sub)),
        "length_mean": round(len_m, 2), "length_sd": round(len_sd, 2),
        "length_median": round(len_med, 1), "length_max": round(len_max, 1),
        "n_overran_floor": int((ls > FLOOR).sum()),
        "pct_reach_se_ability": round(100.0 * float(reach.mean()), 1),
        "se_total_median": round(st_med, 4),
        "recovery_r": round(r, 4), "recovery_slope": round(slope, 4), "theta_mae": round(mae, 4),
    }


def _write_readme(out_dir, se_summary, head_all, head_excl, rec, lb, se, pirt, judge, band, eff):
    fb = se_summary["regimes"]["full_bank"]
    op = se_summary["regimes"]["oppoint_f15se25"]
    tb = se_summary.get("tier_b", {})
    lines = [
        f"# TutorEval gemini-3 CURATED unidim CAT — OF-RECORD @ floor {FLOOR} / SE_ability {SE_TARGET}",
        "",
        f"**TutorEval gemini-3 unidim `ability`, floor {FLOOR} / SE_ability {SE_TARGET}, N=52; "
        "headline excl 3 weakly-identified.** Judge = `gemini-3-flash-preview`.",
        "",
        "**Status:** STUDY / reporting only, LOCAL. Production engine (`tutor_cat/`, "
        "`scripts/scenario_cat_lib.py`, `scripts/calibrate_mirt.py`), the `tutoreval_calibration/` "
        "package, and the grid runner (`oppoint_grid/run_oppoint_grid_gemini3.py`) were NOT "
        "modified (imported/called read-only). Nothing committed. Calibrated on OBSERVED gemini-3 "
        "labels (Rule 0). Mirrors the BiGGen / InfoBench unidim `of_record` numbered-experiment "
        "sets, plus the SE_judge additions.",
        "",
        f"Operating point: **floor (min_scenarios) = {FLOOR}**, **SE_ability target = {SE_TARGET}**, "
        "dense 321-node EAP-posterior marginal-SD stop over ±8 (plateau δ=0.005/W=3, cap 70), "
        "**MWLE θ at stop** (deployed headline θ). Reach = SE_ability ≤ target ONLY. OOS = k=5 "
        "per-fold refit (production M2PL MML-EM, 7 GH, ridge 0.01, clamp), seed 20260729. Curated "
        "bank = 1244 fitted criteria (MINIMAL Rule-0; 542 all-fail columns auto-dropped).",
        "",
        "## De-bias convention = HYBRID",
        "",
        "- **Of-record leaderboard (08): OBSERVED θ + SE_total error bars** (SE_total = "
        "√(SE_ability² + SE_param² + SE_judge²); SE_judge is the dominant floor), 3 weak greyed, "
        f"plus an annotated **{judge['debias_factor_point']:.2f}× cohort pass-rate de-bias caveat** "
        f"(posterior CI {judge['debias_factor_band'][0]:.2f}–{judge['debias_factor_band'][1]:.2f}×). "
        "NOT per-model de-biased.",
        "- **SECONDARY (11): full-bank per-model θ_debiased BAND** from the CLEAN full-bank regime "
        f"(Spearman {se_summary['de_bias_gate']['full_bank']['spearman_obs_vs_debiased']:.3f}, gate "
        "passes, 0 violations). Directional-only for the weak tail (op-point Spearman "
        f"{se_summary['de_bias_gate']['oppoint']['spearman_obs_vs_debiased']:.3f}).",
        "",
        "## Figures",
        "",
        "| # | figure | headline |",
        "|---|---|---|",
        f"| 05 | `experiments/05_oos_recovery/oos_recovery_ability_{TAG}.png` | "
        f"excl-weak r=**{rec['headline_excl_weak']['r']}**, slope=**{rec['headline_excl_weak']['slope']}**, "
        f"θMAE={rec['headline_excl_weak']['theta_mae']}; all-52 r={rec['all_52']['r']}, "
        f"slope={rec['all_52']['slope']} (**slope<1 = compression**) |",
        f"| 04 | `experiments/04_efficiency_vs_random/adaptive_vs_random_efficiency_{TAG}.png` | "
        f"adaptive reaches median SE_ability≤{SE_TARGET} at **{eff['adaptive_reach_scenarios']}** "
        f"vs random at **{eff['random_reach_scenarios']}** scenarios (~{eff['scenario_savings_x']}×) |"
        if eff else
        f"| 04 | `experiments/04_efficiency_vs_random/…` | (skipped) |",
        f"| 07 | `experiments/07_se_components/se_components_{TAG}.png` | 3-component (SE_ability, "
        "SE_param, **SE_judge**) mean±SD; full-bank AND op-point; SE_judge var-share "
        f"≈**{100 * fb['se_judge_var_share_median']:.0f}%** full-bank / "
        f"**{100 * op['se_judge_var_share_median']:.0f}%** op-point |",
        f"| 08 | `experiments/08_leaderboard/leaderboard_{TAG}.png` | **OBSERVED** θ + SE_total "
        f"(incl SE_judge) + cohort de-bias caveat; top **{lb['top_headline']['model']}** "
        f"θ={lb['top_headline']['theta']}, bottom headline **{lb['bottom_headline']['model']}** "
        f"θ={lb['bottom_headline']['theta']} |",
        f"| 09 | `experiments/09_pirt_mae/pirt_pred_vs_actual_{TAG}.png` | pass-rate "
        f"MAE=**{pirt['headline_excl_weak']['pass_rate_mae']}** (r={pirt['headline_excl_weak']['r']}) "
        f"excl-weak; all-52 MAE={pirt['all_52']['pass_rate_mae']} |",
        f"| 10 | `experiments/10_judge_error/judge_error_debias_{TAG}.png` | conceptual_understanding "
        f"own α≈**{judge['own_alpha']}**, pooled overall α≈**{judge['overall_alpha']}**, β=0; cohort "
        f"de-bias **{judge['debias_factor_point']:.2f}× "
        f"[{judge['debias_factor_band'][0]:.2f},{judge['debias_factor_band'][1]:.2f}]** |",
        f"| 11 | `experiments/11_debias_band/debias_band_full_bank_{TAG}.png` | **SECONDARY** — "
        f"full-bank per-model θ_debiased band (Spearman {band['spearman_obs_vs_debiased']:.3f}, "
        "0 violations); NOT of-record headline |",
        "",
        "## Headline numbers @ 15/0.25",
        "",
        f"**excl-weak (N={head_excl['n_models']}):** OOS recovery r=**{head_excl['recovery_r']}**, "
        f"slope=**{head_excl['recovery_slope']}**, θMAE=**{head_excl['theta_mae']}**; length "
        f"mean **{head_excl['length_mean']}±{head_excl['length_sd']}** scen (median "
        f"{head_excl['length_median']:.0f}, max {head_excl['length_max']:.0f}); "
        f"%reach(SE_ability)=**{head_excl['pct_reach_se_ability']}%**; SE_total median "
        f"**{head_excl['se_total_median']}**.",
        "",
        f"**all-52 (N={head_all['n_models']}):** r=**{head_all['recovery_r']}**, "
        f"slope=**{head_all['recovery_slope']}**, θMAE={head_all['theta_mae']}; length mean "
        f"{head_all['length_mean']}±{head_all['length_sd']}, median {head_all['length_median']:.0f}; "
        f"%reach=**{head_all['pct_reach_se_ability']}%**.",
        "",
        "**SE_judge (Tier A, frozen bank) — the DOMINANT floor:** full-bank median "
        f"**{fb['se_judge_median']:.3f}** (mean {fb['se_judge_mean']:.3f}, max {fb['se_judge_max']:.3f}); "
        f"op-point median **{op['se_judge_median']:.3f}** (mean {op['se_judge_mean']:.3f}). "
        f"SE_total incl SE_judge: full median {fb['se_total_median']:.3f}, op median "
        f"**{op['se_total_median']:.3f}**.",
        "",
        "## Framing",
        "",
        f"- **Judge = gemini-3-flash-preview; α≈{judge['overall_alpha']} strict / β=0** (cert-safe: "
        "zero false-pass in 100 gold cells) → cohort pass-rate de-bias "
        f"**{judge['debias_factor_point']:.2f}× "
        f"[{judge['debias_factor_band'][0]:.2f},{judge['debias_factor_band'][1]:.2f}]** (UP, "
        "direction firm; magnitude band wide on 20 gold-PASS cells).",
        f"- **SE_judge is the DOMINANT, correlated floor** — op-point median {op['se_judge_median']:.3f}, "
        f"SE_total ≈ {op['se_total_median']:.3f}; ~{100 * fb['se_judge_var_share_median']:.0f}% of "
        f"full-bank / {100 * op['se_judge_var_share_median']:.0f}% of op-point SE_total², and it "
        "does NOT shrink with more models. Tier B (refit each replicate) SE_judge ≤ Tier A "
        f"(full-bank {tb.get('se_judge_full_B_median', float('nan')):.3f} vs "
        f"{tb.get('se_judge_full_A_median', float('nan')):.3f}), so frozen Tier A is the "
        "**conservative** of-record.",
        "- **Rankings are robust** to the judge layer (full-bank Spearman "
        f"{se_summary['de_bias_gate']['full_bank']['spearman_obs_vs_debiased']:.3f}); the de-bias is "
        "UP with **no shrinkage crossover**. **Absolute θ / pass-rate carry the cohort de-bias "
        "band.** The 3 weakly-identified models (`BEE-spoke-data/smol_llama-220M-GQA-fineweb_edu`, "
        "`ai-forever/mGPT`, `allenai/OLMo-1B-hf`) sit at the extreme low tail and are excluded from "
        "the headline.",
        "- **Precision lever:** tightening α needs a **judge-FAIL-enriched gold audit** (more "
        "true-passes gemini wrongly failed), not more cells — the α CI (hence the whole band) is "
        "wide because the gold is fail-heavy (20 gold-PASS cells).",
        "",
        "## Files",
        "",
        f"- `experiments/05_oos_recovery/` — scatter, `oos_per_model_{TAG}.csv`, `recovery_metrics.json`.",
        f"- `experiments/04_efficiency_vs_random/` — efficiency fig, `adaptive_vs_random_curves.csv`, "
        "`efficiency_metrics.json`.",
        f"- `experiments/07_se_components/` — 3-component bars + `se_components_summary_{TAG}.csv` "
        "(both regimes, both subsets).",
        f"- `experiments/08_leaderboard/` — OF-RECORD leaderboard fig + `leaderboard_{TAG}.csv` "
        "(rank, model, theta, se_ability, se_param, se_judge, se_total, weak).",
        f"- `experiments/09_pirt_mae/` — scatter, `pirt_per_model_{TAG}.csv`, `pirt_metrics.json`.",
        f"- `experiments/10_judge_error/` — judge-error/de-bias fig + `strata_confusion_{TAG}.csv`, "
        "`judge_error_metrics.json`.",
        f"- `experiments/11_debias_band/` — SECONDARY full-bank θ_debiased band fig + "
        f"`debias_band_full_bank_{TAG}.csv`.",
        "- `summary.json` — machine-readable op-point / config / headline (both regimes) + SE_judge "
        "layer.",
        "",
        "Source inputs: `oppoint_grid/oos_per_model_per_cell.csv` (rows floor=15 & target=0.25) for "
        "05/09; `se_judge/per_model_full_bank.csv` + `se_judge/per_model_oppoint_f15se25.csv` + "
        "`se_judge/summary.json` for 07/08/10/11; fig 04 reruns the k-fold machinery (per-fold "
        f"refit + engine adaptive trace + matched random arm) via `build_of_record_{TAG}.py`.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, default=HERE / f"of_record_{TAG}")
    p.add_argument("--cap", type=int, default=70)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--skip-efficiency", action="store_true",
                   help="skip the fresh k-fold (04) run")
    args = p.parse_args()

    exp = args.out_dir / "experiments"
    exp.mkdir(parents=True, exist_ok=True)

    # ---- inputs ----
    pm = pd.read_csv(GRID_CELL_CSV)
    pm["weak"] = pm["weak"].astype(bool)
    cell = pm[(pm["floor"] == FLOOR) & (pm["se_target"] == SE_TARGET)].copy()
    print(f"[cell] floor={FLOOR} target={SE_TARGET}: {len(cell)} models, "
          f"{int(cell['weak'].sum())} weak")

    full_df = pd.read_csv(SE_JUDGE_DIR / "per_model_full_bank.csv")
    op_df = pd.read_csv(SE_JUDGE_DIR / "per_model_oppoint_f15se25.csv")
    se_summary = json.loads((SE_JUDGE_DIR / "summary.json").read_text(encoding="utf-8"))
    gate = se_summary["de_bias_gate"]
    deb_factor = gate["pass_rate_debias_factor"]["point_1_over_1_minus_alpha"]
    deb_band = gate["pass_rate_debias_factor"]["posterior_ci95"]

    # ---- from-CSV figures ----
    rec = fig05_recovery(cell, exp / "05_oos_recovery")
    lb = fig08_leaderboard(op_df, deb_factor, deb_band, exp / "08_leaderboard")
    se = fig07_se_components(full_df, op_df, exp / "07_se_components")
    pirt = fig09_pirt(cell, exp / "09_pirt_mae")
    judge = fig10_judge_error(se_summary, exp / "10_judge_error")
    band = fig11_debias_band(full_df, gate, exp / "11_debias_band")

    print(f"[05] recovery excl-weak r={rec['headline_excl_weak']['r']} "
          f"slope={rec['headline_excl_weak']['slope']} MAE={rec['headline_excl_weak']['theta_mae']}; "
          f"all-52 r={rec['all_52']['r']} slope={rec['all_52']['slope']}")
    print(f"[08] top={lb['top_headline']['model']} theta={lb['top_headline']['theta']}; "
          f"bottom={lb['bottom_headline']['model']} theta={lb['bottom_headline']['theta']}")
    for wr in lb["weak_placement"]:
        print(f"     weak: {wr['model']} rank {wr['rank']}/52 "
              f"theta={wr['theta']} SE_total={wr['se_total']}")
    print(f"[07] full-bank SE_judge mean={se['full_bank_headline']['SE_judge']['mean']:.3f} "
          f"var-share={100 * se['full_bank_headline']['_judge_var_share_median']:.0f}%; "
          f"op SE_judge mean={se['oppoint_headline']['SE_judge']['mean']:.3f}")
    print(f"[09] p-IRT MAE excl-weak={pirt['headline_excl_weak']['pass_rate_mae']} "
          f"r={pirt['headline_excl_weak']['r']}")
    print(f"[10] de-bias factor {judge['debias_factor_point']:.3f} "
          f"[{judge['debias_factor_band'][0]:.3f},{judge['debias_factor_band'][1]:.3f}]; "
          f"own_alpha={judge['own_alpha']} overall_alpha={judge['overall_alpha']}")
    print(f"[11] SECONDARY full-bank band Spearman={band['spearman_obs_vs_debiased']:.3f} "
          f"violations={band['n_violations']}")

    # ---- fresh efficiency (04) ----
    eff = None
    if not args.skip_efficiency:
        print("[04] fresh k-fold adaptive-vs-random efficiency ...", flush=True)
        eff = fig04_efficiency(args, exp / "04_efficiency_vs_random")
        print(f"[04] adaptive reach@{eff['adaptive_reach_scenarios']} vs "
              f"random@{eff['random_reach_scenarios']} (x{eff['scenario_savings_x']})")

    # ---- headline + summary ----
    head_all = _cell_headline(cell, op_df)
    head_excl = _cell_headline(cell[~cell["weak"]], op_df)
    _write_readme(args.out_dir, se_summary, head_all, head_excl, rec, lb, se, pirt, judge, band, eff)

    fb = se_summary["regimes"]["full_bank"]
    op = se_summary["regimes"]["oppoint_f15se25"]
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "STUDY / reporting only - LOCAL; production engine + tutoreval_calibration/ "
                  "grid runner + se_judge scripts untouched; nothing committed; calibrated on "
                  "OBSERVED gemini-3 labels (Rule 0).",
        "benchmark": "TutorEval (gemini-3-flash-preview judge)", "scale": "unidimensional (ability)",
        "judge": JUDGE,
        "op_point": {"floor_min_scenarios": FLOOR, "se_ability_target": SE_TARGET,
                     "reach_rule": "SE_ability (EAP posterior SD at stop) <= target ONLY",
                     "stop": "dense 321-node EAP posterior-SD over +/-8; plateau delta=0.005/W=3; "
                             "cap 70; MWLE theta at stop (deployed headline theta)",
                     "leaderboard_theta": "observed op-point MWLE-at-stop (HYBRID convention)",
                     "oos": "k=5, seed 20260729, per-fold refit (production M2PL MML-EM, 7 GH, "
                            "ridge 0.01, clamp; within-fold zero-variance filter)",
                     "se_total": "sqrt(SE_ability^2 + SE_param^2 + SE_judge^2)"},
        "debias_convention": "HYBRID: of-record leaderboard = observed theta + SE_total + cohort "
                             "1.83x caveat; SECONDARY = full-bank per-model theta_debiased band; "
                             "op-point per-model theta_debiased = directional-only (weak tail).",
        "curated_bank": "reports/tutoreval_gemini3_recal/bank/"
                        "rubrics_qmatrix_final_unidim_gemini3_fitted.jsonl",
        "n_models": 52, "n_criteria_curated": 1244,
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
        "se_judge_regimes": {"full_bank": fb, "oppoint_f15se25": op},
        "pirt": pirt,
        "judge_error": judge,
        "debias_band_secondary": band,
        "efficiency": eff,
        "tier_b": se_summary.get("tier_b"),
        "de_bias_gate": {"full_bank": gate["full_bank"], "oppoint": gate["oppoint"],
                         "pass_rate_debias_factor": gate["pass_rate_debias_factor"],
                         "gate_passed": gate["gate_passed"]},
        "spearman_note": "rankings robust to judge layer (full-bank Spearman 0.993; op-point 0.778 "
                         "= weak-tail reshuffle only); observed theta / pass-rate carry the cohort "
                         "de-bias band.",
        "figures": {
            "05_oos_recovery": f"experiments/05_oos_recovery/oos_recovery_ability_{TAG}.png",
            "04_efficiency_vs_random": f"experiments/04_efficiency_vs_random/adaptive_vs_random_efficiency_{TAG}.png",
            "07_se_components": f"experiments/07_se_components/se_components_{TAG}.png",
            "08_leaderboard": f"experiments/08_leaderboard/leaderboard_{TAG}.png",
            "09_pirt_mae": f"experiments/09_pirt_mae/pirt_pred_vs_actual_{TAG}.png",
            "10_judge_error": f"experiments/10_judge_error/judge_error_debias_{TAG}.png",
            "11_debias_band": f"experiments/11_debias_band/debias_band_full_bank_{TAG}.png",
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
