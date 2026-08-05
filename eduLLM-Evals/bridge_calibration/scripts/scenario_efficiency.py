"""Phase 2b exp 04: efficiency -- adaptive (CAT) vs random scenario selection.

Runs the REAL engine in ``mode=cat`` (adaptive, trace) and ``mode=baseline`` (seeded-random
scenario order) at a sweep of fixed test lengths L (min_scenarios=max_scenarios=L, forcing
exactly L scenarios), full data, masked CAT-pool bank. For each (mode, L) it scores MWLE
ability against the FULL-BANK fine-EAP reference (r, MAE) and reports the mean online SE.
Adaptive should recover ability better than random at equal length. Also reports the
locked-point (L=12) head-to-head.
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

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import bridge_scenario_lib as L  # noqa: E402

ROOT = L.ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import scripts.scenario_cat_lib as scat  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    base = ROOT / "bridge_calibration"
    p.add_argument("--bank", type=Path, default=base / "bridge_scenario_fitted_1d_catpool.jsonl")
    p.add_argument("--matrix", type=Path, default=ROOT / "bridgegrade" / "response_matrix.csv")
    p.add_argument("--scenarios", type=Path, default=ROOT / "data" / "Bridge" / "scenarios.jsonl")
    p.add_argument("--out-dir", type=Path, default=base / "experiments" / "04_efficiency_vs_random")
    p.add_argument("--tmp-dir", type=Path, default=base / "experiments" / "04_efficiency_vs_random" / "_tmp")
    p.add_argument("--lengths", type=str, default="1,2,3,4,6,8,10,12,15,20")
    p.add_argument("--locked-length", type=int, default=12)
    p.add_argument("--eap-grid", type=int, default=321)
    p.add_argument("--range", type=float, default=8.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()
    (args.out_dir / "figures").mkdir(parents=True, exist_ok=True)
    args.tmp_dir.mkdir(parents=True, exist_ok=True)

    records, dims, _ = scat.load_fitted_bank(args.bank, "clamp")
    d = dims[0]
    ids, A, b = scat.assemble_arrays(records, dims)
    matrix = pd.read_csv(args.matrix, index_col=0)
    Yraw = matrix.reindex(columns=ids).to_numpy(float)
    mask = ~np.isnan(Yraw)
    Y = np.nan_to_num(Yraw, nan=0.0)
    models = list(matrix.index)
    row_of = {m: i for i, m in enumerate(models)}
    col = {c: i for i, c in enumerate(ids)}
    egrid, elog = scat.build_grid(1, args.eap_grid, args.range)
    theta_ref = scat.eap_all_models(Y, mask, A, b, egrid, elog)  # (n,1) full-bank fine EAP

    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    print("=" * 84)
    print("EXP 04: CAT (adaptive) vs baseline (random) efficiency")
    print("=" * 84)

    rows = []
    curves = {"cat": {}, "baseline": {}}
    for mode in ("cat", "baseline"):
        for Ln in lengths:
            spec = scat.RunSpec(seed=args.seed, top_n=5, max_se=0.0,
                                min_evals_per_skill=0, min_scenarios=Ln,
                                max_scenarios=Ln, selection="trace", mode=mode,
                                runs_dir=str(args.tmp_dir / f"{mode}_L{Ln}"))
            res = scat.run_models(models, args.bank, args.matrix, args.scenarios,
                                  "clamp", dims, spec, workers=args.workers)
            cat_theta = np.full(len(models), np.nan)
            ses = []
            for r0 in res:
                r = row_of[r0["model"]]
                idx = np.array([col[c] for c in r0["order"] if c in col], dtype=int)
                if idx.size:
                    th0 = scat.eap_subset(Y[r], idx, A, b, egrid, elog)
                    thm, _ = scat.mwle_subset(Y[r], idx, A, b, th0)
                    cat_theta[r] = thm[0]
                else:
                    cat_theta[r] = 0.0
                ses.append(r0["se_online"][0])
            xr = theta_ref[:, 0]
            band = scat.ols_ci_band(xr, cat_theta, np.linspace(xr.min(), xr.max(), 30))
            mae = float(np.mean(np.abs(cat_theta - xr)))
            crit_mean = float(np.mean([r0["criteria_administered"] for r0 in res]))
            row = {"mode": mode, "scenarios": Ln, "criteria_mean": round(crit_mean, 1),
                   "recovery_r": round(band["r"], 4), "r_lo": round(band["r_lo"], 4),
                   "r_hi": round(band["r_hi"], 4), "recovery_slope": round(band["slope"], 4),
                   "theta_mae": round(mae, 4), "mean_online_se": round(float(np.mean(ses)), 4)}
            rows.append(row)
            curves[mode][Ln] = row
            print(f"  {mode:8s} L={Ln:2d}: r={row['recovery_r']:.3f} MAE={row['theta_mae']:.3f} "
                  f"crit={row['criteria_mean']:.0f} se={row['mean_online_se']:.3f}", flush=True)

    with (args.out_dir / "results.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    Ll = args.locked_length
    r_cat = curves["cat"][Ll]["recovery_r"]
    r_rnd = curves["baseline"][Ll]["recovery_r"]
    # scenarios each mode needs to match the OTHER's locked r (efficiency in length)
    def scen_to_reach(mode, target_r):
        for Ln in lengths:
            if curves[mode][Ln]["recovery_r"] >= target_r:
                return Ln
        return None
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "adaptive vs random scenario-selection efficiency",
        "n_models": len(models), "lengths": lengths,
        "locked_length": Ll,
        "locked_point_recovery": {"cat_r": r_cat, "random_r": r_rnd,
                                  "cat_minus_random_r": round(r_cat - r_rnd, 4)},
        "random_scenarios_to_match_cat_at_locked":
            scen_to_reach("baseline", r_cat),
        "adaptive_beats_random":
            bool(all(curves["cat"][Ln]["recovery_r"] >= curves["baseline"][Ln]["recovery_r"] - 1e-9
                     for Ln in lengths)),
        "mean_r_gap_over_lengths":
            round(float(np.mean([curves["cat"][Ln]["recovery_r"]
                                 - curves["baseline"][Ln]["recovery_r"] for Ln in lengths])), 4),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _figure(rows, lengths, args.out_dir / "figures", Ll)
    shutil.rmtree(args.tmp_dir, ignore_errors=True)

    print("\n" + "=" * 84)
    print(f"locked L={Ll}: CAT r={r_cat}  random r={r_rnd}  (gap {r_cat - r_rnd:+.3f})")
    print(f"adaptive>=random at every length: {summary['adaptive_beats_random']}; "
          f"mean r gap {summary['mean_r_gap_over_lengths']}")
    print(f"wrote -> {args.out_dir}")
    return 0


def _figure(rows, lengths, fig_dir, Ll):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    for mode, color in (("cat", "#1b7837"), ("baseline", "#762a83")):
        dd = df[df["mode"] == mode].sort_values("scenarios")
        ax1.plot(dd["scenarios"], dd["recovery_r"], "o-", color=color,
                 label=("adaptive (CAT)" if mode == "cat" else "random"))
        ax1.fill_between(dd["scenarios"], dd["r_lo"], dd["r_hi"], color=color, alpha=0.15)
        ax2.plot(dd["scenarios"], dd["theta_mae"], "o-", color=color,
                 label=("adaptive (CAT)" if mode == "cat" else "random"))
    for ax in (ax1, ax2):
        ax.axvline(Ll, ls=":", color="gray", alpha=0.7, label=f"locked L={Ll}")
    ax1.set_xlabel("scenarios administered"); ax1.set_ylabel("OOS recovery r")
    ax1.set_title("Recovery r vs test length"); ax1.legend(fontsize=8)
    ax2.set_xlabel("scenarios administered"); ax2.set_ylabel("theta-MAE")
    ax2.set_title("theta-MAE vs test length"); ax2.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(fig_dir / "efficiency_adaptive_vs_random.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
