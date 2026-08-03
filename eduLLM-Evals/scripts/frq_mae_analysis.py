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
            mae_in = float(np.mean(np.abs(lb[f"theta_mwle_{d}"] - lb[f"theta_full_{d}"])))
            rec_in = recovery(lb[f"theta_full_{d}"].to_numpy(), lb[f"theta_mwle_{d}"].to_numpy())
            row = {"bank": bank_key, "skill": d, "theta_mae_insample": mae_in,
                   "r_insample": rec_in["r"], "slope_insample": rec_in["slope"]}
            # OOS from k-fold per-model export (if present)
            oos_path = args.base / "kfold" / bank_key / "oos_per_model.csv"
            if oos_path.is_file():
                oos = pd.read_csv(oos_path)
                mae_oos = float(np.mean(np.abs(oos[f"theta_mwle_{d}"] - oos[f"theta_ref_{d}"])))
                rec_oos = recovery(oos[f"theta_ref_{d}"].to_numpy(),
                                   oos[f"theta_mwle_{d}"].to_numpy())
                row.update({"theta_mae_oos": mae_oos, "r_oos": rec_oos["r"],
                            "slope_oos": rec_oos["slope"]})
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
            passrate_rows.append({
                "bank": bank_key, "skill": d, "n": int(pred.size),
                "actual_mean": float(actual.mean()), "pred_mean": float(pred.mean()),
                "passrate_mae_raw": raw_mae, "passrate_mae_calibrated": cal_mae,
                "r": float(np.corrcoef(pred, actual)[0, 1]),
                "affine_slope": float(s), "affine_intercept": float(c)})
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
                tmae = float(np.mean(np.abs(sd[f"theta_mwle_{d}"] - sd[f"theta_full_{d}"])))
                ppred, pact = prse[d]
                ok = np.isfinite(ppred) & np.isfinite(pact)
                pmae = float(np.mean(np.abs(ppred[ok] - pact[ok])))
                mae_vs_se_rows.append({
                    "bank": bank_key, "se_target": float(se), "skill": d,
                    "theta_mae_mwle": tmae, "passrate_mae": pmae,
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
                rec = recovery(df[f"theta_full_{d}"].to_numpy(),
                               df[f"theta_mwle_{d}"].to_numpy())
                row[f"mwle_r_{d}"] = rec["r"]
                row[f"mwle_slope_{d}"] = rec["slope"]
                row[f"mwle_theta_mae_{d}"] = float(
                    np.mean(np.abs(df[f"theta_mwle_{d}"] - df[f"theta_full_{d}"])))
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
    # D3: adaptive vs random -- mean scenarios & criteria
    if not avr_df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
        arms = avr_df["arm"].tolist()
        colors = ["#2b8cbe", "#d95f0e"]
        for ax, col, title in [(axes[0], "mean_scenarios", "Mean scenarios administered"),
                               (axes[1], "mean_criteria", "Mean criteria administered")]:
            vals = avr_df[col].tolist()
            bars = ax.bar(arms, vals, color=colors)
            ax.set_title(title, fontsize=11)
            ax.set_ylim(0, max(vals) * 1.18)
            for brc, v in zip(bars, vals):
                ax.text(brc.get_x() + brc.get_width() / 2, v, f"{v:.1f}",
                        ha="center", va="bottom", fontsize=10)
        fig.suptitle("FRQ CAT efficiency: adaptive vs random (2-skill, se=0.30, min_scenarios=12)",
                     fontsize=11)
        fig.tight_layout()
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
        ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
        fig.tight_layout()
        fig.savefig(fig_dir / f"frq_theta_mae_vs_se_{bank_key}.png", dpi=140)
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
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel(f"actual full-bank pass rate ({d})")
            ax.set_ylabel(f"predicted pass rate (MWLE, {d})")
            ax.set_title(f"{d}\nr = {r:.3f}, MAE = {mae:.4f} (n = {pred.size})", fontsize=10)
            ax.legend(loc="upper left", fontsize=9)
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
