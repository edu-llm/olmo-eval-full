"""Regenerate the TutorEval unidim floor-sweep summary artifacts over all floors.

Thin, read-only-over-results plotting/summary driver. It reads the per-floor
``min_evals_<floor>/metrics.json`` recovery numbers already produced by
``scenario_kfold_estimator_cv.py`` and (re)writes the four sweep-summary artifacts:

  * ``sweep_results.md``            - criteria-axis table (one row per floor)
  * ``sweep_summary.png``           - recovery r / worst-12 gap vs mean CRITERIA
  * ``sweep_summary_scenarios.csv`` - per-floor length + recovery, scenario units
  * ``sweep_summary_scenarios.png`` - recovery r / worst-12 gap vs mean SCENARIOS

It does NOT run the engine and does NOT touch any per-floor directory. The
``mean_scenarios_administered`` field is taken from the per-floor metrics when
present, else from the sibling ``scenarios_rerun/min_evals_<floor>/metrics.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SWEEP = ROOT / "regenerated_figures" / "scenario_level" / "kfold_sweep" / "tutoreval_unidim"
FLOORS = (6, 8, 10, 12, 15, 18, 20)
ESTIMATORS = ("online", "batch", "mwle")
EST_STYLE = {
    "online": ("#1f77b4", "online"),
    "batch": ("#2ca02c", "batch"),
    "mwle": ("#d62728", "mwle (production)"),
}


def _load_floor(floor: int) -> dict:
    d = json.loads((SWEEP / f"min_evals_{floor}" / "metrics.json").read_text())
    dim = d["dims"][0]
    rec = d["oos_recovery"]
    mean_scen = d.get("mean_scenarios_administered")
    if mean_scen is None:
        rr = json.loads(
            (SWEEP / "scenarios_rerun" / f"min_evals_{floor}" / "metrics.json").read_text()
        )
        mean_scen = rr["mean_scenarios_administered"]
    row = {
        "floor": floor,
        "mean_criteria": d["mean_criteria_administered"],
        "mean_scenarios": mean_scen,
    }
    for est in ESTIMATORS:
        row[f"r_{est}"] = rec[est][dim]["r"]
        row[f"slope_{est}"] = rec[est][dim]["slope"]
        row[f"gap_{est}"] = rec[est][dim]["gap_worst12"]
    return row


def write_results_md(rows: list[dict], out_md: Path) -> None:
    header = (
        "| floor | mean_len | online_r | online_slope | online_gapW12 | batch_r | "
        "batch_slope | batch_gapW12 | mwle_r | mwle_slope | mwle_gapW12 |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    lines = [header]
    for r in rows:
        lines.append(
            f"| {r['floor']} | {r['mean_criteria']:.2f} | "
            f"{r['r_online']:.3f} | {r['slope_online']:.3f} | {r['gap_online']:.3f} | "
            f"{r['r_batch']:.3f} | {r['slope_batch']:.3f} | {r['gap_batch']:.3f} | "
            f"{r['r_mwle']:.3f} | {r['slope_mwle']:.3f} | {r['gap_mwle']:.3f} |\n"
        )
    out_md.write_text("".join(lines))


def write_scenarios_csv(rows: list[dict], out_csv: Path) -> None:
    ordered = sorted(rows, key=lambda r: r["mean_scenarios"])
    head = (
        "floor,mean_criteria,mean_scenarios,r_online,gap_online,r_batch,gap_batch,r_mwle,gap_mwle\n"
    )
    body = "".join(
        f"{r['floor']},{r['mean_criteria']},{r['mean_scenarios']},"
        f"{r['r_online']},{r['gap_online']},{r['r_batch']},{r['gap_batch']},"
        f"{r['r_mwle']},{r['gap_mwle']}\n"
        for r in ordered
    )
    out_csv.write_text(head + body)


def _plot(rows: list[dict], xkey: str, xlabel: str, suptitle: str, out_png: Path) -> None:
    rows = sorted(rows, key=lambda r: r[xkey])
    x = [r[xkey] for r in rows]
    fig, (ax_r, ax_g) = plt.subplots(1, 2, figsize=(11, 4.2))
    for est, (color, label) in EST_STYLE.items():
        lw = 2.4 if est == "mwle" else 1.6
        ax_r.plot(x, [r[f"r_{est}"] for r in rows], "-o", color=color, lw=lw, label=label)
        ax_g.plot(x, [r[f"gap_{est}"] for r in rows], "-o", color=color, lw=lw, label=label)
    for r in rows:
        ax_r.annotate(
            str(int(r["floor"])),
            (r[xkey], r["r_mwle"]),
            textcoords="offset points",
            xytext=(4, -11),
            fontsize=8,
            color="#d62728",
        )
        ax_g.annotate(
            str(int(r["floor"])),
            (r[xkey], r["gap_mwle"]),
            textcoords="offset points",
            xytext=(4, 6),
            fontsize=8,
            color="#d62728",
        )
    ax_r.set_xlabel(xlabel)
    ax_r.set_ylabel("OOS recovery r")
    ax_r.set_title("Recovery r vs test length")
    ax_r.legend(loc="lower right", fontsize=9)
    ax_g.set_xlabel(xlabel)
    ax_g.set_ylabel("worst-12 gap (bias, lower=better)")
    ax_g.set_title("Worst-12 gap vs test length")
    ax_g.legend(loc="upper right", fontsize=9)
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def main() -> int:
    rows = [_load_floor(f) for f in FLOORS]
    write_results_md(rows, SWEEP / "sweep_results.md")
    write_scenarios_csv(rows, SWEEP / "sweep_summary_scenarios.csv")
    _plot(
        rows,
        "mean_criteria",
        "mean criteria administered (test length)",
        "TutorEval unidim: CAT floor (min_evals_per_skill) sensitivity sweep "
        "(labels = floor value, N=52)",
        SWEEP / "sweep_summary.png",
    )
    _plot(
        rows,
        "mean_scenarios",
        "mean scenarios administered (test length)",
        "TutorEval unidim: CAT floor (min_evals_per_skill) sweep in "
        "SCENARIO units (labels = floor value, N=52)",
        SWEEP / "sweep_summary_scenarios.png",
    )
    for r in rows:
        print(
            f"floor {r['floor']:2d}: crit={r['mean_criteria']:.2f} "
            f"scen={r['mean_scenarios']:.2f} mwle_r={r['r_mwle']:.3f} "
            f"mwle_gap={r['gap_mwle']:.3f}"
        )
    print(f"wrote -> {SWEEP}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
