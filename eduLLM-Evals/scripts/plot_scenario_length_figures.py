"""Scenario-unit versions of the TutorEval CAT length / efficiency figures.

The scenario-level CAT experiments report test length in *criteria* administered by
default. Because a CAT selects whole scenarios (each scoring several criteria), the more
natural unit for "how long is the test" is *scenarios administered*. This is a thin,
read-only plotting driver: it re-expresses already-computed recovery numbers against a
scenarios x-axis. It does NOT run the engine and does NOT touch any existing figure.

Sources (all pre-computed):
  * Floor sweep (unidim): ``<sweep-dir>/min_evals_*/metrics.json`` with the
    ``mean_scenarios_administered`` field emitted by ``scenario_kfold_estimator_cv.py``.
  * Selection A/B (unidim & 2skill): ``selection/<bank>/per_model_{trace,dopt}.csv`` which
    already carry per-model ``scenarios`` and ``criteria`` columns.

Usage
-----
    python scripts/plot_scenario_length_figures.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCEN = ROOT / "regenerated_figures" / "scenario_level"

EST_STYLE = {
    "online": ("#1f77b4", "online"),
    "batch": ("#2ca02c", "batch"),
    "mwle": ("#d62728", "mwle (production)"),
}
FLOORS = (6, 8, 10, 12, 15)


def _load_sweep(sweep_dir: Path) -> pd.DataFrame:
    """One row per floor: mean length (criteria/scenarios) and OOS r/gap per estimator."""
    rows = []
    for floor in FLOORS:
        mpath = sweep_dir / f"min_evals_{floor}" / "metrics.json"
        d = json.loads(mpath.read_text())
        rec = d["oos_recovery"]
        dim = d["dims"][0]
        row = {
            "floor": floor,
            "mean_criteria": d["mean_criteria_administered"],
            "mean_scenarios": d["mean_scenarios_administered"],
        }
        for est in EST_STYLE:
            row[f"r_{est}"] = rec[est][dim]["r"]
            row[f"gap_{est}"] = rec[est][dim]["gap_worst12"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mean_scenarios").reset_index(drop=True)


def plot_sweep_scenarios(sweep_dir: Path, out_png: Path, out_csv: Path) -> pd.DataFrame:
    df = _load_sweep(sweep_dir)
    df.to_csv(out_csv, index=False)
    x = df["mean_scenarios"].to_numpy()

    fig, (ax_r, ax_g) = plt.subplots(1, 2, figsize=(11, 4.2))
    for est, (color, label) in EST_STYLE.items():
        lw = 2.4 if est == "mwle" else 1.6
        ax_r.plot(x, df[f"r_{est}"], "-o", color=color, lw=lw, label=label)
        ax_g.plot(x, df[f"gap_{est}"], "-o", color=color, lw=lw, label=label)
    for _, rr in df.iterrows():
        ax_r.annotate(
            str(int(rr.floor)),
            (rr.mean_scenarios, rr.r_mwle),
            textcoords="offset points",
            xytext=(4, -11),
            fontsize=8,
            color="#d62728",
        )
        ax_g.annotate(
            str(int(rr.floor)),
            (rr.mean_scenarios, rr.gap_mwle),
            textcoords="offset points",
            xytext=(4, 6),
            fontsize=8,
            color="#d62728",
        )
    ax_r.set_xlabel("mean scenarios administered (test length)")
    ax_r.set_ylabel("OOS recovery r")
    ax_r.set_title("Recovery r vs test length")
    ax_r.legend(loc="lower right", fontsize=9)
    ax_g.set_xlabel("mean scenarios administered (test length)")
    ax_g.set_ylabel("worst-12 gap (bias, lower=better)")
    ax_g.set_title("Worst-12 gap vs test length")
    ax_g.legend(loc="upper right", fontsize=9)
    fig.suptitle(
        "TutorEval unidim: CAT floor (min_evals_per_skill) sweep in "
        "SCENARIO units (labels = floor value, N=52)",
        fontsize=11,
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return df


def plot_selection_length(bank_dir: Path, out_png: Path) -> dict:
    trace = pd.read_csv(bank_dir / "per_model_trace.csv")
    dopt = pd.read_csv(bank_dir / "per_model_dopt.csv")
    tm, dm = trace.scenarios.mean(), dopt.scenarios.mean()
    hi = int(max(trace.scenarios.max(), dopt.scenarios.max()))
    lo = int(min(trace.scenarios.min(), dopt.scenarios.min()))
    bins = np.arange(lo - 0.5, hi + 1.5, 1.0)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(
        [trace.scenarios, dopt.scenarios],
        bins=bins,
        label=[f"trace (mean {tm:.1f})", f"D-opt (mean {dm:.1f})"],
        alpha=0.85,
    )
    ax.set_xlabel("scenarios administered")
    ax.set_ylabel("models")
    ax.set_title("Scenario-level CAT length: trace vs D-optimality")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return {"trace_scenarios_mean": float(tm), "dopt_scenarios_mean": float(dm)}


def plot_efficiency(sweep_df: pd.DataFrame, out_png: Path) -> None:
    """r vs scenarios administered: unidim floor sweep (line) + 2skill selection arms."""
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot(
        sweep_df["mean_scenarios"],
        sweep_df["r_mwle"],
        "-o",
        color="#d62728",
        lw=2.4,
        label="unidim (floor sweep, mwle)",
    )

    m2 = json.loads((SCEN / "selection" / "tutoreval_2skill" / "metrics.json").read_text())
    dims2 = m2["dims"]
    markers = {"conceptual_understanding": "s", "quantitative_procedural": "^"}
    colors = {"conceptual_understanding": "#1f77b4", "quantitative_procedural": "#9467bd"}
    for d in dims2:
        xs, ys = [], []
        for arm in ("trace", "dopt"):
            xs.append(m2["results"][arm]["length"]["scenarios_mean"])
            ys.append(m2["results"][arm]["recovery"]["theta_mwle"][d]["r"])
        order = np.argsort(xs)
        xs = np.array(xs)[order]
        ys = np.array(ys)[order]
        ax.plot(
            xs,
            ys,
            "--" + markers[d],
            color=colors[d],
            lw=1.6,
            label=f"2skill {d} (trace->dopt, mwle)",
        )

    ax.set_xlabel("mean scenarios administered")
    ax.set_ylabel("recovery r (mwle)")
    ax.set_title("CAT efficiency: recovery r vs scenarios administered")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--sweep-dir",
        type=Path,
        default=SCEN / "kfold_sweep" / "tutoreval_unidim" / "scenarios_rerun",
        help="dir holding min_evals_*/metrics.json with mean_scenarios_administered",
    )
    args = p.parse_args()

    sweep_png = SCEN / "kfold_sweep" / "tutoreval_unidim" / "sweep_summary_scenarios.png"
    sweep_csv = SCEN / "kfold_sweep" / "tutoreval_unidim" / "sweep_summary_scenarios.csv"
    sweep_df = plot_sweep_scenarios(args.sweep_dir, sweep_png, sweep_csv)
    cols = ["floor", "mean_criteria", "mean_scenarios"]
    print(f"sweep length (scenarios) by floor:\n{sweep_df[cols]}")
    print(f"wrote -> {sweep_png}")

    for bank in ("tutoreval_unidim", "tutoreval_2skill"):
        bank_dir = SCEN / "selection" / bank
        out = bank_dir / "figures" / "length_trace_vs_dopt_scenarios.png"
        stats = plot_selection_length(bank_dir, out)
        print(f"{bank} selection length (scenarios): {stats}")
        print(f"wrote -> {out}")

    eff_png = SCEN / "efficiency_r_vs_scenarios.png"
    plot_efficiency(sweep_df, eff_png)
    print(f"wrote -> {eff_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
