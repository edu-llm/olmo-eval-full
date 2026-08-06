"""Phase 2b exp 09: standalone p-IRT pass-rate MAE (parity with BiGGen/Bridge 09_pirt_mae).

Breaks the p-IRT predicted-vs-actual content out of exp-05 into its own experiment. Reuses
the exp-05 OOS per-model data (locked op-point 8 scenarios / SE 0.12, model-fold k=5, fine
uniform-EAP reference): predicted pass rate = mean sigmoid(a*theta_oos - b) under TRAIN
params at the OOS MWLE theta; actual = held-out observed pass rate over the fold's CAT-pool
criteria.

Outputs (BiGGen schema):
  experiments/09_pirt_mae/results.csv
  experiments/09_pirt_mae/oos_per_model_pirt.csv
  experiments/09_pirt_mae/figures/pirt_pred_vs_actual.png
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import wildbench_scenario_lib as L  # noqa: E402

ROOT = L.ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import scripts.scenario_cat_lib as scat  # noqa: E402


def boot_mae_ci(err, B=2000, seed=0, ci=0.95):
    rng = np.random.default_rng(seed)
    n = err.size
    maes = np.array([np.mean(err[rng.integers(0, n, n)]) for _ in range(B)])
    a = (1 - ci) / 2
    return float(np.percentile(maes, 100 * a)), float(np.percentile(maes, 100 * (1 - a)))


def loo_mae(err):
    n = err.size
    tot = err.sum()
    return float(np.mean([(tot - err[i]) / (n - 1) for i in range(n)]))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    base = ROOT / "wildbench_calibration"
    p.add_argument("--oos-05", type=Path,
                   default=base / "experiments" / "05_oos_recovery" / "oos_per_model.csv")
    p.add_argument("--out-dir", type=Path, default=base / "experiments" / "09_pirt_mae")
    p.add_argument("--min-scenarios", type=int, default=8)
    p.add_argument("--max-se", type=float, default=0.12)
    p.add_argument("--dim", type=str, default="overall")
    args = p.parse_args()
    (args.out_dir / "figures").mkdir(parents=True, exist_ok=True)

    d = args.dim
    src = pd.read_csv(args.oos_05)
    per = pd.DataFrame({
        "model": src["model"],
        "predicted_passrate": src["pred_pass_rate"].astype(float),
        "actual_passrate": src["obs_pass_rate"].astype(float),
        "theta_oos": src[f"theta_mwle_{d}"].astype(float),
    })
    per["abs_error"] = (per["predicted_passrate"] - per["actual_passrate"]).abs()
    per = per.sort_values("model").reset_index(drop=True)
    per.to_csv(args.out_dir / "oos_per_model_pirt.csv", index=False)

    actual = per["actual_passrate"].to_numpy()
    predicted = per["predicted_passrate"].to_numpy()
    theta_oos = per["theta_oos"].to_numpy()
    theta_ref = src.sort_values("model")[f"theta_ref_{d}"].astype(float).to_numpy()

    xs = np.linspace(actual.min(), actual.max(), 40)
    band = scat.ols_ci_band(actual, predicted, xs, B=2000, seed=0)
    pass_err = np.abs(predicted - actual)
    mae_raw = float(np.mean(pass_err))
    mae_lo, mae_hi = boot_mae_ci(pass_err)
    mae_loo = loo_mae(pass_err)
    theta_err = np.abs(theta_oos - theta_ref)
    theta_mae = float(np.mean(theta_err))
    th_lo, th_hi = boot_mae_ci(theta_err)

    results = pd.DataFrame([{
        "eval": "OOS", "passrate_mae_raw": mae_raw, "passrate_mae_raw_lo": mae_lo,
        "passrate_mae_raw_hi": mae_hi, "passrate_mae_loo": mae_loo,
        "theta_mae": theta_mae, "theta_mae_lo": th_lo, "theta_mae_hi": th_hi,
        "r": band["r"], "r_lo": band["r_lo"], "r_hi": band["r_hi"],
        "slope": band["slope"], "slope_lo": band["slope_lo"], "slope_hi": band["slope_hi"],
        "n": len(per), "min_scenarios": args.min_scenarios, "max_se": args.max_se,
    }])
    results.to_csv(args.out_dir / "results.csv", index=False)

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "standalone p-IRT pass-rate MAE (parity with BiGGen/Bridge 09_pirt_mae)",
        "source": "reuses exp-05 OOS per-model p-IRT data (locked op-point, model-fold k=5)",
        "operating_point": {"min_scenarios": args.min_scenarios, "max_se": args.max_se},
        "pass_r": round(band["r"], 4), "pass_mae": round(mae_raw, 4),
        "pass_slope": round(band["slope"], 4), "theta_mae": round(theta_mae, 4),
        "n_models": len(per),
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    _figure(actual, predicted, band, mae_raw, len(per), args.min_scenarios, args.max_se,
            args.out_dir / "figures")

    print("=" * 84)
    print("EXP 09: p-IRT pass-rate MAE (standalone, BiGGen/Bridge parity)")
    print("=" * 84)
    print(f"pass-r={band['r']:.4f} [{band['r_lo']:.4f},{band['r_hi']:.4f}]  "
          f"pass-MAE={mae_raw:.4f} [{mae_lo:.4f},{mae_hi:.4f}]  slope={band['slope']:.4f}  "
          f"theta-MAE={theta_mae:.4f}  n={len(per)}")
    print(f"wrote -> {args.out_dir}")
    return 0


def _figure(actual, predicted, band, mae_raw, n, floor, se, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    xs = np.linspace(actual.min(), actual.max(), 60)
    b2 = scat.ols_ci_band(actual, predicted, xs, B=2000, seed=0)
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    lo, hi = min(actual.min(), predicted.min()) - 0.03, max(actual.max(), predicted.max()) + 0.03
    ax.plot([lo, hi], [lo, hi], ls="--", color="gray", lw=1, label="y = x")
    ax.fill_between(xs, b2["band_lo"], b2["band_hi"], color="#2c7fb8", alpha=0.18)
    ax.plot(xs, b2["slope"] * xs + b2["intercept"], color="#2c7fb8", lw=1.5,
            label=f"OLS slope={b2['slope']:.3f}")
    ax.scatter(actual, predicted, s=30, alpha=0.8, edgecolor="k", linewidth=0.3)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("actual held-out pass rate")
    ax.set_ylabel("p-IRT predicted pass rate (TRAIN params @ OOS theta)")
    ax.set_title(f"WildBench scenario p-IRT pass calibration (floor={floor}, SE={se})\n"
                 f"r={band['r']:.3f} [{band['r_lo']:.3f},{band['r_hi']:.3f}], "
                 f"pass-MAE={mae_raw:.3f} (n={n})", fontsize=9)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(fig_dir / "pirt_pred_vs_actual.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
