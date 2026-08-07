"""Phase 2b exp 08: full-bank ability leaderboard with honest SE_total bars.

Leaderboard scoring is the FULL administrable bank (all ~228 scenarios / 3961 non-excluded
criteria) per model -- NOT the 12-scenario CAT. Ability theta is the full-bank fine-EAP;
SE bars are SE_total = SE_ability (posterior) (+) SE_param (calibration noise), read from
exp 07 (observed-info bootstrap). Writes the ranked leaderboard + a bar figure, and a
top-level copy of model_leaderboard.csv.
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
    p.add_argument("--se-components", type=Path,
                   default=base / "experiments" / "07_parameter_uncertainty" / "leaderboard_se_components.csv")
    p.add_argument("--out-dir", type=Path, default=base / "experiments" / "08_leaderboard")
    p.add_argument("--top-level", type=Path, default=base / "model_leaderboard.csv")
    p.add_argument("--weakly-identified", type=Path,
                   default=base / "experiments" / "07_parameter_uncertainty" / "precision_reached.csv",
                   help="deployed-SE precision_reached.csv; weakly_identified = EAP-native cap.")
    p.add_argument("--eap-grid", type=int, default=321)
    p.add_argument("--range", type=float, default=8.0)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    records, dims, _ = scat.load_fitted_bank(args.bank, "clamp")
    d = dims[0]
    ids, A, b = scat.assemble_arrays(records, dims)
    matrix = pd.read_csv(args.matrix, index_col=0)
    Yraw = matrix.reindex(columns=ids).to_numpy(float)
    mask = ~np.isnan(Yraw)
    Y = np.nan_to_num(Yraw, nan=0.0)
    models = list(matrix.index)
    egrid, elog = scat.build_grid(1, args.eap_grid, args.range)
    theta = scat.eap_all_models(Y, mask, A, b, egrid, elog)[:, 0]
    obs_pass = np.array([Y[i][mask[i]].mean() for i in range(len(models))])

    se = pd.read_csv(args.se_components).set_index("model")
    rows = []
    for i, m in enumerate(models):
        se_ability = float(se.loc[m, f"se_posterior_{d}"]) if m in se.index else np.nan
        se_param = float(se.loc[m, f"se_param_{d}"]) if m in se.index else np.nan
        se_total = float(se.loc[m, f"se_total_{d}"]) if m in se.index else np.nan
        rows.append({"model": m, "theta": float(theta[i]),
                     "se_ability": se_ability, "se_param": se_param,
                     "se_total": se_total, "observed_pass_rate": float(obs_pass[i])})
    # weakly_identified flag = EAP-native cap under the deployed stop (never reaches
    # SE-ability <= 0.12 within the forced cap); theta is an UPPER BOUND for these models.
    weakly = {}
    if Path(args.weakly_identified).is_file():
        wdf = pd.read_csv(args.weakly_identified)
        if "weakly_identified" in wdf.columns:
            weakly = dict(zip(wdf["model"], wdf["weakly_identified"].astype(bool)))

    df = pd.DataFrame(rows).sort_values("theta", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    df["weakly_identified"] = df["model"].map(lambda m: bool(weakly.get(m, False)))
    df.to_csv(args.out_dir / "model_leaderboard.csv", index=False)
    df.to_csv(args.top_level, index=False)

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "full-bank ability leaderboard with SE_total bars (NOT the 12-scenario CAT)",
        "scoring": "full administrable bank (A3 + extreme_a excluded), 1D fine-EAP theta",
        "se_bars": "SE_total = sqrt(SE_ability_posterior^2 + SE_param^2), from exp 07",
        "n_models": len(df), "n_criteria_scored": len(ids),
        "theta_range": [float(df["theta"].min()), float(df["theta"].max())],
        "se_total_range": [float(df["se_total"].min()), float(df["se_total"].max())],
        "top5": df.head(5)[["model", "theta", "se_total"]].to_dict("records"),
        "bottom5": df.tail(5)[["model", "theta", "se_total"]].to_dict("records"),
        "weakly_identified_note": (
            "weakly_identified = EAP-native cap under the deployed stop @ 12/0.12 (never reaches "
            "SE-ability <= 0.12 within the forced cap); such models' full-bank theta is an UPPER "
            "BOUND, not a point estimate. Full-bank theta itself is stop-independent (unchanged "
            "by the adoption)."),
        "n_weakly_identified": int(df["weakly_identified"].sum()),
        "weakly_identified_models": df[df["weakly_identified"]][
            ["model", "theta", "se_total"]].to_dict("records"),
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = df.sort_values("theta").reset_index(drop=True)
    ypos = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(8, max(9, 0.22 * len(order))))
    ax.errorbar(order["theta"], ypos, xerr=1.96 * order["se_total"], fmt="none",
                ecolor="#c1666b", elinewidth=2.2, alpha=0.6, label="+/-1.96 SE_total")
    ax.errorbar(order["theta"], ypos, xerr=1.96 * order["se_ability"], fmt="none",
                ecolor="#4d648d", elinewidth=1.0, label="+/-1.96 SE_ability")
    ax.scatter(order["theta"], ypos, s=12, color="k", zorder=3)
    ax.set_yticks(ypos); ax.set_yticklabels(order["model"], fontsize=5)
    ax.set_xlabel("full-bank ability (theta, 1D)")
    ax.set_title("Bridge scenario-level leaderboard (full-bank theta, SE_total bars)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(args.out_dir / "leaderboard_general.png", dpi=140)
    plt.close(fig)

    print("=" * 84)
    print("EXP 08: full-bank leaderboard")
    print("=" * 84)
    print(f"theta range [{df['theta'].min():.2f}, {df['theta'].max():.2f}]  "
          f"SE_total range [{df['se_total'].min():.3f}, {df['se_total'].max():.3f}]")
    print("top5:", ", ".join(f"{r['model']} ({r['theta']:.2f})" for r in meta["top5"]))
    print("bottom5:", ", ".join(f"{r['model']} ({r['theta']:.2f})" for r in meta["bottom5"]))
    print(f"wrote -> {args.out_dir} and {args.top_level}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
