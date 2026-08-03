"""Build a TutorEval leaderboard that reports the ONLINE information-filter SE.

Companion to TutorEval's existing batch-EAP + parametric-bootstrap leaderboard
(``scripts/scenario_param_uncertainty.py`` ->
``regenerated_figures/scenario_level/param_uncertainty/tutoreval_{unidim,2skill}/``).
This analog reports the raw online SE the scenario CAT engine emits during a run,
``SE_k = sqrt(diag(U_k))`` from the recursive information update
``U^-1 <- U^-1 + p(1-p)(q(.)a)(q(.)a)^T`` with ``U0 = I`` -- i.e. WITHOUT the
parameter-uncertainty bootstrap. It mirrors the STYLE of TutorBench's online-SE
builder ``scripts/build_correctness_leaderboard.py`` (which uses ``final_se_*``).

No engine re-run: both quantities are reused from persisted seed=42 trace runs.

* Online SE + online ability point estimate come from the selection A/B run's
  ``per_model_trace.csv`` (``final_se_<skill>``, ``theta_online_<skill>``).
* Batch-EAP + bootstrap SE (for the direct comparison) come from the
  param-uncertainty ``leaderboard_se_components.csv`` (``se_posterior_*``,
  ``se_total_*``, ``theta_*``). The two files share administered sets: the
  batch ``theta_<skill>`` equals ``theta_batch_<skill>`` in the trace file.

Outputs (all NEW, under ``regenerated_figures/scenario_level/online_se_leaderboard/``):
* ``leaderboard_online_se_unidim.csv`` / ``leaderboard_online_se_2skill.csv``
* ``figures/leaderboard_bars_online_se_<skill>.png``
* ``comparison_online_vs_batch_eap.csv`` + ``metrics.json``
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEL = ROOT / "regenerated_figures" / "scenario_level" / "selection"
PU = ROOT / "regenerated_figures" / "scenario_level" / "param_uncertainty"
OUT_DIR = ROOT / "regenerated_figures" / "scenario_level" / "online_se_leaderboard"
FIG_DIR = OUT_DIR / "figures"

AXES = {
    "unidim": ("tutoreval_unidim", ["ability"]),
    "2skill": ("tutoreval_2skill", ["conceptual_understanding", "quantitative_procedural"]),
}


def build_axis(axis: str, run_dir: str, skills: list[str]) -> tuple[pd.DataFrame, list[dict]]:
    """Merge the online (trace) and batch-EAP+bootstrap tables for one axis.

    Returns the per-model leaderboard frame (sorted by the first skill's online
    ability, descending) and a list of per-skill SE comparison records."""
    trace = pd.read_csv(SEL / run_dir / "per_model_trace.csv")
    batch = pd.read_csv(PU / run_dir / "leaderboard_se_components.csv")

    lb = pd.DataFrame({"model": trace["model"]})
    for s in skills:
        lb[f"theta_{s}"] = trace[f"theta_online_{s}"].to_numpy(float)
        lb[f"se_online_{s}"] = trace[f"final_se_{s}"].to_numpy(float)
        lb[f"scorable_evals_{s}"] = trace[f"scorable_evals_{s}"].astype(int)
    lb["criteria_administered"] = trace["criteria"].astype(int)
    lb["scenarios_administered"] = trace["scenarios"].astype(int)

    lb = lb.sort_values(f"theta_{skills[0]}", ascending=False).reset_index(drop=True)
    lb.insert(0, "rank", np.arange(1, len(lb) + 1))

    # ---- comparison vs batch-EAP + bootstrap (same administered sets) --------
    b = batch.set_index("model")
    cmp_rows: list[dict] = []
    for s in skills:
        m = trace.set_index("model")
        # align on model so per-model ratios pair the right rows
        common = m.index.intersection(b.index)
        se_online = m.loc[common, f"final_se_{s}"].to_numpy(float)
        se_post = b.loc[common, f"se_posterior_{s}"].to_numpy(float)
        se_total = b.loc[common, f"se_total_{s}"].to_numpy(float)
        theta_online = m.loc[common, f"theta_online_{s}"].to_numpy(float)
        theta_batch = b.loc[common, f"theta_{s}"].to_numpy(float)
        cmp_rows.append(
            {
                "axis": axis,
                "skill": s,
                "n_models": int(len(common)),
                "mean_se_online": float(se_online.mean()),
                "mean_se_posterior_batch": float(se_post.mean()),
                "mean_se_total_batch": float(se_total.mean()),
                "median_se_online": float(np.median(se_online)),
                "median_se_posterior_batch": float(np.median(se_post)),
                "median_se_total_batch": float(np.median(se_total)),
                "ratio_mean_online_over_total": float(se_online.mean() / se_total.mean()),
                "ratio_mean_online_over_posterior": float(se_online.mean() / se_post.mean()),
                "mean_per_model_ratio_online_over_total": float(np.mean(se_online / se_total)),
                "mean_diff_total_minus_online": float(np.mean(se_total - se_online)),
                "theta_online_vs_batch_pearson": float(
                    np.corrcoef(theta_online, theta_batch)[0, 1]
                ),
            }
        )
    return lb, cmp_rows


def figure_axis(axis: str, run_dir: str, skills: list[str], lb: pd.DataFrame) -> list[Path]:
    """Leaderboard bar figure per skill: online SE bars over a faded batch
    SE_total bar so the narrowing (omitted parameter uncertainty) is visible.
    Mirrors ``scenario_param_uncertainty._figure`` layout."""
    batch = pd.read_csv(PU / run_dir / "leaderboard_se_components.csv").set_index("model")
    paths: list[Path] = []
    for s in skills:
        order = lb.sort_values(f"theta_{s}").reset_index(drop=True)
        ypos = np.arange(len(order))
        se_total = batch.reindex(order["model"])[f"se_total_{s}"].to_numpy(float)
        se_online = order[f"se_online_{s}"].to_numpy(float)
        theta = order[f"theta_{s}"].to_numpy(float)

        fig, ax = plt.subplots(figsize=(7.5, max(9, 0.22 * len(order))))
        ax.errorbar(
            theta, ypos, xerr=1.96 * se_total, fmt="none",
            ecolor="#c1666b", elinewidth=2.4, alpha=0.45,
            label="+/-1.96 SE_total (batch-EAP + bootstrap)",
        )
        ax.errorbar(
            theta, ypos, xerr=1.96 * se_online, fmt="none",
            ecolor="#4d648d", elinewidth=1.4,
            label="+/-1.96 SE_online (sqrt diag U)",
        )
        ax.scatter(theta, ypos, s=10, color="k", zorder=3)
        ax.set_yticks(ypos)
        ax.set_yticklabels(order["model"], fontsize=5)
        ax.set_xlabel(f"ability ({s})")
        ax.set_title(
            f"TutorEval online-SE leaderboard ({axis}): {s}\n"
            f"online SE = sqrt(diag(U)); faded bar = batch-EAP + bootstrap SE_total"
        )
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        out = FIG_DIR / f"leaderboard_bars_online_se_{s}.png"
        fig.savefig(out, dpi=130)
        plt.close(fig)
        paths.append(out)
    return paths


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    all_cmp: list[dict] = []
    summary: dict = {
        "generated_at": datetime.now(UTC).isoformat(),
        "purpose": "TutorEval leaderboard reporting the online information-filter SE "
        "(sqrt diag U), analog of TutorBench's online-SE leaderboard; complements the "
        "batch-EAP + bootstrap leaderboard.",
        "data_provenance": "reused persisted seed=42 trace runs (no engine re-run)",
        "sources": {
            "online_se": "regenerated_figures/scenario_level/selection/"
            "tutoreval_{unidim,2skill}/per_model_trace.csv (final_se_*, theta_online_*)",
            "batch_eap_bootstrap": "regenerated_figures/scenario_level/param_uncertainty/"
            "tutoreval_{unidim,2skill}/leaderboard_se_components.csv (se_posterior_*, se_total_*)",
        },
        "config": {"max_se": 0.30, "min_evals_per_skill": 15, "max_scenarios": 50,
                   "selection": "trace", "seed": 42, "eap_grid": 61},
        "axes": {},
    }

    for axis, (run_dir, skills) in AXES.items():
        lb, cmp_rows = build_axis(axis, run_dir, skills)
        csv_path = OUT_DIR / f"leaderboard_online_se_{axis}.csv"
        round_map = {c: 4 for c in lb.columns if c.startswith(("theta_", "se_online_"))}
        lb.round(round_map).to_csv(csv_path, index=False)

        fig_paths = figure_axis(axis, run_dir, skills, lb)
        all_cmp.extend(cmp_rows)
        summary["axes"][axis] = {
            "skills": skills,
            "n_models": int(len(lb)),
            "leaderboard_csv": str(csv_path.relative_to(ROOT)),
            "figures": [str(p.relative_to(ROOT)) for p in fig_paths],
            "se_comparison": cmp_rows,
        }

        print("=" * 88)
        print(f"AXIS {axis}  skills={skills}  models={len(lb)}")
        print(f"leaderboard -> {csv_path}")
        cols = (
            ["rank", "model"]
            + [f"theta_{s}" for s in skills]
            + [f"se_online_{s}" for s in skills]
        )
        print("\nTop 5:")
        print(lb.head(5)[cols].to_string(index=False))
        print("\nBottom 5:")
        print(lb.tail(5)[cols].to_string(index=False))
        print("\nonline SE vs batch-EAP + bootstrap SE:")
        for r in cmp_rows:
            print(
                f"  {r['skill']:26s} mean SE_online={r['mean_se_online']:.4f}  "
                f"SE_posterior={r['mean_se_posterior_batch']:.4f}  "
                f"SE_total={r['mean_se_total_batch']:.4f}  "
                f"| online/total={r['ratio_mean_online_over_total']:.3f}  "
                f"mean(total-online)={r['mean_diff_total_minus_online']:+.4f}"
            )
        print()

    cmp_df = pd.DataFrame(all_cmp)
    cmp_path = OUT_DIR / "comparison_online_vs_batch_eap.csv"
    cmp_df.to_csv(cmp_path, index=False)
    with (OUT_DIR / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("=" * 88)
    print(f"comparison csv -> {cmp_path}")
    print(f"metrics json   -> {OUT_DIR / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
