"""MAE-style metrics + figures for the TutorBench free-response (FRQ) CAT results.

Parallels the MCQ/ATLAS benchmark sections (theta MAE, pass-rate MAE, adaptive-vs-random
efficiency, MAE-vs-SE curve, p-IRT predicted-vs-actual). MWLE ability estimates are the
primary numbers throughout. All inputs are already-computed per-model CSVs plus the fitted
banks and response matrices; nothing here reruns the engine (the random-baseline arm is
produced separately by ``offline_engine_driver.py --mode baseline``).

Outputs go under ``--out-dir`` (default
``regenerated_figures/scenario_level_115_min12/frq_mae``):
    figures/*.png
    frq_theta_mae.csv, frq_passrate_mae.csv, frq_adaptive_vs_random.csv,
    frq_mae_vs_se.csv, frq_mae_summary.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.scenario_cat_lib as scat  # noqa: E402

BASE = ROOT / "regenerated_figures" / "scenario_level_115_min12"

BANKS = {
    "2_skills": {
        "bank": ROOT / "data" / "TutorBench" / "rubrics_qmatrix_calibrated_2skill_115_fitted.jsonl",
        "matrix": ROOT / "staging" / "response_matrix_full_nonopt_115.csv",
        "dims": ["correctness", "scaffolding"],
    },
    "3_skills": {
        "bank": ROOT / "data" / "TutorBench" / "rubrics_qmatrix_calibrated_3skill_115_fitted.jsonl",
        "matrix": ROOT / "staging" / "response_matrix_full_115.csv",
        "dims": ["correctness", "scaffolding", "presentation"],
    },
}
SE_LEVELS = ["0.20", "0.25", "0.30", "0.35"]


def load_skill_items(bank_path: Path, dims: list[str]) -> dict[str, dict]:
    """Per skill: {criterion_id -> (a_skill, b)} for criteria that load on that skill.

    Uses the same clamp policy as the engine driver (negative loadings -> 0)."""
    out = {d: {} for d in dims}
    with bank_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            b = float(r["difficulty"])
            for d in dims:
                if int(r["q_modeled"][d]) == 1:
                    a = float(r["discrimination"][d] or 0.0)
                    out[d][r["criterion_id"]] = (max(a, 0.0), b)
    return out


def recovery(x: np.ndarray, y: np.ndarray) -> dict:
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    return {"r": float(np.corrcoef(x, y)[0, 1]),
            "slope": float(np.polyfit(x, y, 1)[0]),
            "mae": float(np.mean(np.abs(y - x))),
            "n": int(x.size)}


# metric statistics on aligned per-model arrays (x = reference, y = estimate)
def _mae(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.abs(y - x)))


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1])


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.polyfit(x, y, 1)[0])


def bootstrap_ci(func, arrays, B: int = 2000, seed: int = 0, ci: float = 0.95):
    """Nonparametric bootstrap CI over the MODELS (rows).

    Resample model indices with replacement, recompute ``func(*resampled_arrays)`` each
    draw, and return the (lo, hi) percentiles for the requested central interval. Uses a
    fixed-seed generator so the CIs are reproducible."""
    arrays = [np.asarray(a, dtype=float) for a in arrays]
    n = arrays[0].shape[0]
    rng = np.random.default_rng(seed)
    stats = np.empty(B)
    for t in range(B):
        idx = rng.integers(0, n, n)
        stats[t] = func(*[a[idx] for a in arrays])
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.percentile(stats, [100.0 * alpha, 100.0 * (1.0 - alpha)])
    return float(lo), float(hi)


def loo_affine_mae(pred: np.ndarray, actual: np.ndarray) -> float:
    """Leave-one-out affine recalibration MAE: for each model fit slope+intercept
    (actual ~ pred) on the OTHER models, apply to the held-out model, take |err|."""
    n = pred.size
    errs = np.empty(n)
    for i in range(n):
        m = np.arange(n) != i
        s, c = np.polyfit(pred[m], actual[m], 1)
        errs[i] = abs((s * pred[i] + c) - actual[i])
    return float(errs.mean())


def passrate_per_model(theta: pd.DataFrame, matrix: pd.DataFrame,
                       skill_items: dict[str, dict], dims: list[str]):
    """Return dict skill -> (pred[n], actual[n]) aligned to theta.index (models).

    pred_d[model]   = mean over skill-d criteria observed for the model of
                      sigmoid(a_d * theta_mwle_d - b)
    actual_d[model] = observed mean pass on those same criteria (full bank)."""
    res = {}
    models = list(theta.index)
    for d in dims:
        items = skill_items[d]
        cols = [c for c in items if c in matrix.columns]
        acol = np.array([items[c][0] for c in cols])
        bcol = np.array([items[c][1] for c in cols])
        M = matrix.loc[models, cols].to_numpy(dtype=float)  # (n_models, n_items) w/ NaN
        obs = ~np.isnan(M)
        th = theta[f"theta_mwle_{d}"].to_numpy()
        pred = np.full(len(models), np.nan)
        actual = np.full(len(models), np.nan)
        for i in range(len(models)):
            o = obs[i]
            if not o.any():
                continue
            p = expit(acol[o] * th[i] - bcol[o])
            pred[i] = float(p.mean())
            actual[i] = float(M[i, o].mean())
        res[d] = (pred, actual)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, default=BASE)
    ap.add_argument("--out-dir", type=Path, default=BASE / "frq_mae")
    ap.add_argument("--baseline-csv", type=Path,
                    default=BASE / "frq_mae" / "baseline" / "2_skills" / "cat_per_model.csv")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = args.out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    theta_rows = []       # theta-MAE (in-sample + OOS)
    passrate_rows = []     # pass-rate MAE (raw + LOO-recalibrated)
    mae_vs_se_rows = []    # theta-MAE (+ pass-rate MAE) vs se_target
    pirt_points = {}       # for the 2-skill scatter

    for bank_key, cfg in BANKS.items():
        dims = cfg["dims"]
        lb = pd.read_csv(args.base / "leaderboard" / bank_key / "cat_per_model.csv",
                         index_col="model")
        matrix = pd.read_csv(cfg["matrix"], index_col=0)
        skill_items = load_skill_items(cfg["bank"], dims)

        # ---- Deliverable 1: theta-MAE (MWLE) in-sample ----
        for d in dims:
            xf = lb[f"theta_full_{d}"].to_numpy()
            xm = lb[f"theta_mwle_{d}"].to_numpy()
            rec_in = recovery(xf, xm)
            row = {"bank": bank_key, "skill": d, "theta_mae_insample": rec_in["mae"],
                   "r_insample": rec_in["r"], "slope_insample": rec_in["slope"]}
            row["theta_mae_insample_lo"], row["theta_mae_insample_hi"] = \
                bootstrap_ci(_mae, (xf, xm))
            row["r_insample_lo"], row["r_insample_hi"] = bootstrap_ci(_corr, (xf, xm))
            row["slope_insample_lo"], row["slope_insample_hi"] = bootstrap_ci(_slope, (xf, xm))
            # OOS from k-fold per-model export (if present)
            oos_path = args.base / "kfold" / bank_key / "oos_per_model.csv"
            if oos_path.is_file():
                oos = pd.read_csv(oos_path)
                of = oos[f"theta_ref_{d}"].to_numpy()
                om = oos[f"theta_mwle_{d}"].to_numpy()
                rec_oos = recovery(of, om)
                row.update({"theta_mae_oos": rec_oos["mae"], "r_oos": rec_oos["r"],
                            "slope_oos": rec_oos["slope"]})
                row["theta_mae_oos_lo"], row["theta_mae_oos_hi"] = bootstrap_ci(_mae, (of, om))
                row["r_oos_lo"], row["r_oos_hi"] = bootstrap_ci(_corr, (of, om))
                row["slope_oos_lo"], row["slope_oos_hi"] = bootstrap_ci(_slope, (of, om))
            theta_rows.append(row)

        # ---- Deliverable 2: pass-rate MAE (raw + LOO recalibrated) ----
        pr = passrate_per_model(lb, matrix, skill_items, dims)
        for d in dims:
            pred, actual = pr[d]
            ok = np.isfinite(pred) & np.isfinite(actual)
            pred, actual = pred[ok], actual[ok]
            raw_mae = float(np.mean(np.abs(pred - actual)))
            cal_mae = loo_affine_mae(pred, actual)
            s, c = np.polyfit(pred, actual, 1)
            prow = {
                "bank": bank_key, "skill": d, "n": int(pred.size),
                "actual_mean": float(actual.mean()), "pred_mean": float(pred.mean()),
                "passrate_mae_raw": raw_mae, "passrate_mae_calibrated": cal_mae,
                "r": float(np.corrcoef(pred, actual)[0, 1]),
                "affine_slope": float(s), "affine_intercept": float(c)}
            # bootstrap CIs over models for the pred/actual pass-rate arrays
            prow["passrate_mae_raw_lo"], prow["passrate_mae_raw_hi"] = \
                bootstrap_ci(_mae, (pred, actual))
            prow["r_lo"], prow["r_hi"] = bootstrap_ci(_corr, (pred, actual))
            prow["affine_slope_lo"], prow["affine_slope_hi"] = bootstrap_ci(_slope, (pred, actual))
            passrate_rows.append(prow)
            if bank_key == "2_skills":
                pirt_points[d] = (pred, actual, raw_mae,
                                  float(np.corrcoef(pred, actual)[0, 1]))

        # ---- Deliverable 4: MAE-vs-SE (theta MWLE + pass-rate) ----
        for se in SE_LEVELS:
            se_csv = args.base / "se_sweep" / bank_key / f"se{se}" / "cat_per_model.csv"
            if not se_csv.is_file():
                continue
            sd = pd.read_csv(se_csv, index_col="model")
            prse = passrate_per_model(sd, matrix, skill_items, dims)
            for d in dims:
                xf = sd[f"theta_full_{d}"].to_numpy()
                xm = sd[f"theta_mwle_{d}"].to_numpy()
                tmae = float(np.mean(np.abs(xm - xf)))
                rec_se = recovery(xf, xm)
                r_lo, r_hi = bootstrap_ci(_corr, (xf, xm))
                ppred, pact = prse[d]
                ok = np.isfinite(ppred) & np.isfinite(pact)
                pmae = float(np.mean(np.abs(ppred[ok] - pact[ok])))
                mae_vs_se_rows.append({
                    "bank": bank_key, "se_target": float(se), "skill": d,
                    "theta_mae_mwle": tmae, "passrate_mae": pmae,
                    "recovery_r": rec_se["r"],
                    "recovery_r_lo": r_lo, "recovery_r_hi": r_hi,
                    "mean_criteria": float(sd["criteria_administered"].mean()),
                    "mean_scenarios": float(sd["scenarios_administered"].mean())})

    theta_df = pd.DataFrame(theta_rows)
    passrate_df = pd.DataFrame(passrate_rows)
    mae_se_df = pd.DataFrame(mae_vs_se_rows)

    # ================= Deliverable 3: adaptive vs random (2-skill) =============
    dims2 = BANKS["2_skills"]["dims"]
    adaptive = pd.read_csv(args.base / "leaderboard" / "2_skills" / "cat_per_model.csv",
                           index_col="model")
    avr_rows = []
    baseline = None
    if args.baseline_csv.is_file():
        baseline = pd.read_csv(args.baseline_csv, index_col="model")
        for arm_name, df in [("adaptive", adaptive), ("random", baseline)]:
            row = {"arm": arm_name,
                   "mean_scenarios": float(df["scenarios_administered"].mean()),
                   "mean_criteria": float(df["criteria_administered"].mean()),
                   "precision_reached": int(df["precision_reached"].sum()),
                   "n_models": int(len(df))}
            for d in dims2:
                xf = df[f"theta_full_{d}"].to_numpy()
                xm = df[f"theta_mwle_{d}"].to_numpy()
                rec = recovery(xf, xm)
                row[f"mwle_r_{d}"] = rec["r"]
                row[f"mwle_slope_{d}"] = rec["slope"]
                row[f"mwle_theta_mae_{d}"] = rec["mae"]
                row[f"mwle_r_{d}_lo"], row[f"mwle_r_{d}_hi"] = bootstrap_ci(_corr, (xf, xm))
                row[f"mwle_slope_{d}_lo"], row[f"mwle_slope_{d}_hi"] = \
                    bootstrap_ci(_slope, (xf, xm))
            avr_rows.append(row)
    avr_df = pd.DataFrame(avr_rows)

    # ============================ write CSVs ==================================
    args.out_dir.mkdir(parents=True, exist_ok=True)
    theta_df.to_csv(args.out_dir / "frq_theta_mae.csv", index=False)
    passrate_df.to_csv(args.out_dir / "frq_passrate_mae.csv", index=False)
    mae_se_df.to_csv(args.out_dir / "frq_mae_vs_se.csv", index=False)
    if not avr_df.empty:
        avr_df.to_csv(args.out_dir / "frq_adaptive_vs_random.csv", index=False)

    # tidy long summary
    summ = []
    for _, r in theta_df.iterrows():
        summ.append({"bank": r["bank"], "skill": r["skill"],
                     "metric": "theta_mae_mwle_insample", "value": r["theta_mae_insample"]})
        if "theta_mae_oos" in r and pd.notna(r.get("theta_mae_oos")):
            summ.append({"bank": r["bank"], "skill": r["skill"],
                         "metric": "theta_mae_mwle_oos", "value": r["theta_mae_oos"]})
    for _, r in passrate_df.iterrows():
        summ.append({"bank": r["bank"], "skill": r["skill"],
                     "metric": "passrate_mae_raw", "value": r["passrate_mae_raw"]})
        summ.append({"bank": r["bank"], "skill": r["skill"],
                     "metric": "passrate_mae_calibrated", "value": r["passrate_mae_calibrated"]})
    pd.DataFrame(summ).to_csv(args.out_dir / "frq_mae_summary.csv", index=False)

    # ============================ figures =====================================
    # D3: adaptive vs random -- top row: mean scenarios & criteria; bottom row (full
    # width): recovery r (arm x skill) vs the full-bank (all-items) EAP reference.
    if not avr_df.empty:
        fig, axd = plt.subplot_mosaic([["scen", "crit"], ["rec_r", "rec_slope"]],
                                      figsize=(11, 8))
        arms = avr_df["arm"].tolist()
        colors = ["#2b8cbe", "#d95f0e"]
        arm_color = {a: c for a, c in zip(arms, colors)}
        arm_label = {"adaptive": "adaptive (Fisher selection)",
                     "random": "random (random order)"}
        for key, col, title in [("scen", "mean_scenarios", "Mean scenarios administered"),
                                ("crit", "mean_criteria", "Mean criteria administered")]:
            ax = axd[key]
            vals = avr_df[col].tolist()
            bars = ax.bar(arms, vals, color=colors)
            ax.set_title(title, fontsize=11)
            ax.set_ylim(0, max(vals) * 1.18)
            for brc, v in zip(bars, vals):
                ax.text(brc.get_x() + brc.get_width() / 2, v, f"{v:.1f}",
                        ha="center", va="bottom", fontsize=10)

        x = np.arange(len(dims2))
        width = 0.38

        def grouped_metric(ax, metric):
            """Grouped arm x skill bars for ``metric`` with asymmetric bootstrap CI bars.

            Value labels sit ABOVE the upper CI whisker so they never collide with it."""
            for j, arm in enumerate(arms):
                arow = avr_df[avr_df["arm"] == arm].iloc[0]
                vals = [float(arow[f"{metric}_{d}"]) for d in dims2]
                his = list(vals)
                yerr = None
                if all(f"{metric}_{d}_lo" in avr_df.columns for d in dims2):
                    los = [float(arow[f"{metric}_{d}_lo"]) for d in dims2]
                    his = [float(arow[f"{metric}_{d}_hi"]) for d in dims2]
                    yerr = np.clip(np.array([[v - lo for v, lo in zip(vals, los)],
                                             [hi - v for v, hi in zip(vals, his)]]), 0.0, None)
                bars = ax.bar(x + (j - 0.5) * width, vals, width,
                              color=arm_color[arm], label=arm_label.get(arm, arm),
                              yerr=yerr, capsize=4,
                              error_kw={"ecolor": "black", "elinewidth": 1.0})
                for brc, v, hi in zip(bars, vals, his):
                    ax.text(brc.get_x() + brc.get_width() / 2, hi + 0.012, f"{v:.3f}",
                            ha="center", va="bottom", fontsize=8)
            ax.set_xticks(x)
            ax.set_xticklabels(dims2)

        # bottom-left: recovery r (arm x skill) vs the full-bank (all-items) EAP reference
        ax = axd["rec_r"]
        grouped_metric(ax, "mwle_r")
        ax.set_ylim(0, 1.12)  # small headroom for the value labels above the whiskers
        ax.set_ylabel("recovery r vs full-bank (all-items) EAP")
        ax.set_title("Recovery r (MWLE ~ full-bank EAP)", fontsize=11)
        # bottom-right: recovery SLOPE. Slope separates the arms where r cannot -- adaptive
        # correctness ~ 1.00 (well-calibrated spread) vs random-order ~ 1.16 (inflated spread);
        # r is ~0.97-0.98 for both, so only the slope exposes the calibration difference.
        ax = axd["rec_slope"]
        grouped_metric(ax, "mwle_slope")
        slope_hi = max(float(avr_df[avr_df["arm"] == arm][f"mwle_slope_{d}_hi"].iloc[0])
                       for arm in arms for d in dims2)
        ax.set_ylim(0, max(1.05, slope_hi) * 1.18)
        ax.axhline(1.0, ls="--", color="gray", lw=1)
        ax.text(0.015, 1.0, "ideal = 1", transform=ax.get_yaxis_transform(),
                va="bottom", ha="left", fontsize=7, color="gray")
        ax.set_ylabel("recovery slope (MWLE ~ full-bank EAP)")
        ax.set_title("Recovery slope (MWLE ~ full-bank EAP)", fontsize=11)
        # suptitle notes random is now run-to-convergence-or-exhaustion (not capped at 50)
        rand_row = avr_df[avr_df["arm"] == "random"]
        if not rand_row.empty:
            rc = int(rand_row["precision_reached"].iloc[0])
            rn = int(rand_row["n_models"].iloc[0])
            rs = float(rand_row["mean_scenarios"].iloc[0])
            if rc == rn:
                rand_note = (f"random run to convergence: all {rn} reached SE target "
                             f"(mean {rs:.1f} scenarios)")
            elif rc == 0:
                rand_note = "random never converges even at 662 scenarios (bank exhaustion)"
            else:
                rand_note = (f"random reached SE target for {rc}/{rn} "
                             f"(else bank exhaustion at 662)")
        else:
            rand_note = "random run to convergence or bank exhaustion (max 662)"
        fig.suptitle("FRQ CAT efficiency: adaptive (Fisher order) vs random-order "
                     "(same stopping rule, adaptive length)\n"
                     f"2-skill, se=0.30, min_scenarios=12 -- {rand_note}", fontsize=11)
        # single shared arm legend BELOW the bottom row so it never covers a panel's
        # bars / value labels (per-panel legends kept re-colliding with the left group).
        from matplotlib.patches import Patch
        handles = [Patch(color=arm_color[a], label=arm_label.get(a, a)) for a in arms]
        fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9,
                   framealpha=0.9, bbox_to_anchor=(0.5, 0.005))
        fig.tight_layout(rect=(0, 0.05, 1, 0.93))
        fig.savefig(fig_dir / "frq_adaptive_vs_random_efficiency.png", dpi=140)
        plt.close(fig)

    # D4: theta-MAE (MWLE) vs se_target, one line per skill; overlay pass-rate MAE
    for bank_key in BANKS:
        sub = mae_se_df[mae_se_df["bank"] == bank_key]
        if sub.empty:
            continue
        dims = BANKS[bank_key]["dims"]
        fig, ax = plt.subplots(figsize=(6.0, 4.4))
        ax2 = ax.twinx()
        for d in dims:
            s = sub[sub["skill"] == d].sort_values("se_target")
            ax.plot(s["se_target"], s["theta_mae_mwle"], marker="o",
                    label=f"theta-MAE {d}")
            ax2.plot(s["se_target"], s["passrate_mae"], marker="s", ls="--",
                     alpha=0.6, label=f"pass-rate MAE {d}")
        ax.set_xlabel("SE target")
        ax.set_ylabel("theta-MAE (MWLE, logits)")
        ax2.set_ylabel("pass-rate MAE (dashed)")
        ax.set_title(f"FRQ theta-MAE vs SE target ({bank_key.replace('_', '-')})")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        # place the legend below the axes so it never sits on the plotted lines
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper center",
                  bbox_to_anchor=(0.5, -0.12), ncol=2, framealpha=0.9)
        fig.tight_layout()
        fig.savefig(fig_dir / f"frq_theta_mae_vs_se_{bank_key}.png", dpi=140,
                    bbox_inches="tight")
        plt.close(fig)

    # D4b: recovery r (MWLE CAT vs full-bank theta) vs se_target, own figure per bank.
    # Higher SE target -> fewer items -> expect r to fall; shows fidelity retained.
    for bank_key in BANKS:
        sub = mae_se_df[mae_se_df["bank"] == bank_key]
        if sub.empty or "recovery_r" not in sub.columns:
            continue
        dims = BANKS[bank_key]["dims"]
        fig, ax = plt.subplots(figsize=(6.0, 4.4))
        for d in dims:
            s = sub[sub["skill"] == d].sort_values("se_target")
            line, = ax.plot(s["se_target"], s["recovery_r"], marker="o", label=d)
            if {"recovery_r_lo", "recovery_r_hi"}.issubset(s.columns):
                ax.fill_between(s["se_target"], s["recovery_r_lo"], s["recovery_r_hi"],
                                color=line.get_color(), alpha=0.2)
        ax.set_xlabel("SE target")
        ax.set_ylabel("recovery r (CAT MWLE vs full-bank theta)")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"FRQ recovery r vs SE target ({bank_key.replace('_', '-')})")
        ax.legend(fontsize=8, loc="lower left")
        fig.tight_layout()
        fig.savefig(fig_dir / f"frq_recovery_r_vs_se_{bank_key}.png", dpi=140)
        plt.close(fig)

    # D4c(A): recovery efficiency frontier -- recovery r vs mean scenarios administered.
    # Looser SE target -> shorter test -> lower-left; more scenarios buys higher fidelity
    # with diminishing returns. y-axis zoomed to the data so the decline is visible.
    for bank_key in BANKS:
        sub = mae_se_df[mae_se_df["bank"] == bank_key]
        if sub.empty or "recovery_r" not in sub.columns:
            continue
        dims = BANKS[bank_key]["dims"]
        fig, ax = plt.subplots(figsize=(7.0, 4.8))
        for d in dims:
            s = sub[sub["skill"] == d].sort_values("mean_scenarios")
            xr = s["mean_scenarios"].to_numpy()
            yr = s["recovery_r"].to_numpy()
            yerr = np.clip(np.array([yr - s["recovery_r_lo"].to_numpy(),
                                     s["recovery_r_hi"].to_numpy() - yr]), 0.0, None)
            line, = ax.plot(xr, yr, marker="o", label=d)
            ax.errorbar(xr, yr, yerr=yerr, fmt="none", ecolor=line.get_color(),
                        elinewidth=1.0, capsize=3, alpha=0.7)
        # test length (mean_scenarios) is shared across skills at a given SE target, so the
        # skills' markers stack at one x. Annotate each SE target ONCE, above the topmost
        # marker's whisker, so labels never overlap when two skill lines nearly coincide.
        for se_v, g in sub.groupby("se_target"):
            xx = float(g["mean_scenarios"].mean())
            ytop = float(g["recovery_r_hi"].max())
            ax.annotate(f"SE {se_v:.2f}", (xx, ytop), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8, color="dimgray",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                                  alpha=0.7))
        ylo = float(sub["recovery_r_lo"].min())
        yhi = float(sub["recovery_r_hi"].max())
        pad = max((yhi - ylo) * 0.10, 0.01)
        ax.set_ylim(ylo - pad, yhi + pad + 0.035)
        ax.set_xlabel("mean scenarios administered (test length)")
        ax.set_ylabel("recovery r vs full-bank (all-items) EAP")
        ax.set_title(f"FRQ recovery efficiency frontier ({bank_key.replace('_', '-')})\n"
                     "more scenarios buys higher fidelity (diminishing returns); labels = SE target",
                     fontsize=9)
        ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
        fig.tight_layout()
        fig.savefig(fig_dir / f"frq_recovery_frontier_{bank_key}.png", dpi=140)
        plt.close(fig)

    # D4c(B): twin-axis -- recovery r per skill (left, with CI band) + mean scenarios
    # (right, single dashed line; length does not depend on skill) vs SE target.
    for bank_key in BANKS:
        sub = mae_se_df[mae_se_df["bank"] == bank_key]
        if sub.empty or "recovery_r" not in sub.columns:
            continue
        dims = BANKS[bank_key]["dims"]
        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        ax2 = ax.twinx()
        for d in dims:
            s = sub[sub["skill"] == d].sort_values("se_target")
            line, = ax.plot(s["se_target"], s["recovery_r"], marker="o", label=f"r {d}")
            ax.fill_between(s["se_target"], s["recovery_r_lo"], s["recovery_r_hi"],
                            color=line.get_color(), alpha=0.2)
        # mean scenarios is shared across skills for a given se_target
        length = (sub.groupby("se_target")["mean_scenarios"].mean().reset_index()
                  .sort_values("se_target"))
        lh, = ax2.plot(length["se_target"], length["mean_scenarios"], marker="s", ls="--",
                       color="black", label="mean scenarios")
        ylo = float(sub["recovery_r_lo"].min())
        yhi = float(sub["recovery_r_hi"].max())
        pad = max((yhi - ylo) * 0.10, 0.01)
        ax.set_ylim(ylo - pad, yhi + pad)
        ax.set_xlabel("SE target")
        ax.set_ylabel("recovery r vs full-bank (all-items) EAP")
        ax2.set_ylabel("mean scenarios administered (dashed)")
        ax.set_title(f"FRQ recovery r & test length vs SE target ({bank_key.replace('_', '-')})",
                     fontsize=10)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper center",
                  bbox_to_anchor=(0.5, -0.12), ncol=len(dims) + 1, framealpha=0.9)
        fig.tight_layout()
        fig.savefig(fig_dir / f"frq_recovery_r_length_twin_{bank_key}.png", dpi=140,
                    bbox_inches="tight")
        plt.close(fig)

    # D5: p-IRT predicted vs actual pass-rate scatter (2-skill), correctness +
    # scaffolding side by side so the calibration contrast is read in one glance.
    panel_dims = [d for d in ("correctness", "scaffolding") if d in pirt_points]
    if panel_dims:
        fig, axes = plt.subplots(1, len(panel_dims), figsize=(4.6 * len(panel_dims), 4.6))
        if len(panel_dims) == 1:
            axes = [axes]
        for ax, d in zip(axes, panel_dims):
            pred, actual, mae, r = pirt_points[d]
            ax.scatter(actual, pred, s=28, alpha=0.75, edgecolor="k", linewidth=0.3)
            lo = min(pred.min(), actual.min()) - 0.02
            hi = max(pred.max(), actual.max()) + 0.02
            ax.plot([lo, hi], [lo, hi], ls="--", color="gray", lw=1, label="y = x")
            # OLS fit of the plotted axes (predicted ~ actual) + bootstrap CI band. Fitting in
            # the same direction as the axes keeps the drawn line and its annotated slope
            # consistent; the gap from y=x is the systematic bias. A slope well below 1
            # (scaffolding) means predicted pass rate barely tracks actual -- a flat calibration.
            ag = np.linspace(actual.min(), actual.max(), 60)
            fit = scat.ols_ci_band(actual, pred, ag, B=2000, seed=0)
            ax.plot(ag, fit["slope"] * ag + fit["intercept"], color="#d95f0e", lw=1.4,
                    label=f"OLS fit (slope = {fit['slope']:.3f})")
            ax.fill_between(ag, fit["band_lo"], fit["band_hi"], color="#d95f0e", alpha=0.18)
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel(f"actual full-bank pass rate ({d})")
            ax.set_ylabel(f"predicted pass rate (MWLE, {d})")
            ax.set_title(
                f"{d}\nr = {fit['r']:.3f} [{fit['r_lo']:.3f}, {fit['r_hi']:.3f}]\n"
                f"slope = {fit['slope']:.3f} [{fit['slope_lo']:.3f}, {fit['slope_hi']:.3f}], "
                f"MAE = {mae:.4f} (n = {pred.size})", fontsize=9)
            ax.legend(loc="best", fontsize=8, framealpha=0.9)
        fig.suptitle("FRQ p-IRT predicted vs actual pass rate (2-skill, MWLE)", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(fig_dir / "frq_pirt_vs_actual_2skill.png", dpi=140)
        plt.close(fig)

    # ============================ console ====================================
    print("=" * 88)
    print("theta-MAE (MWLE) per skill")
    print(theta_df.to_string(index=False))
    print("\npass-rate MAE (raw + LOO-recalibrated) per skill")
    print(passrate_df.to_string(index=False))
    if not avr_df.empty:
        print("\nadaptive vs random (2-skill)")
        print(avr_df.to_string(index=False))
    print(f"\nwrote -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
