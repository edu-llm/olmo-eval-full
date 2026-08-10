"""Total ability SE for the 2-skill FRQ leaderboard = ability SE (+) calibration SE.

The leaderboard's per-model ``final_se_*`` is ability-only (conditional on the fitted item
parameters, hence overconfident). This script combines it with the item-parameter
(calibration) SE from the observed-information parametric bootstrap in
``scenario_param_uncertainty.py`` (``se_param_*``):

    SE_total = sqrt(SE_ability^2 + SE_param^2)

and writes a per-model long CSV plus two figures:

    frq_total_se_per_model.csv        model, skill, se_ability, se_param, se_total
    frq_mean_total_se_2skill.png      grouped bars: mean SE_ability vs mean SE_total (+/- SD)
    frq_se_shrinkage_2skill.png       mean posterior SE vs #scenarios administered (+/- SD)

The shrinkage figure re-runs the production engine with per-scenario logging on and reads
each model's ``steps.jsonl`` SE trace (no engine changes; ``scat.run_one_model`` is called
with ``cleanup=False`` so the logs survive).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.scenario_cat_lib as scat  # noqa: E402

BASE = ROOT / "regenerated_figures" / "scenario_level_115_min12"
DIMS = ["correctness", "scaffolding"]
SE_TARGET = 0.30
COLORS = {"ability": "#4d648d", "total": "#c1666b"}


# ---------------------------------------------------------------------------
# BiGGen gemini-3 SE_judge quadrature (Phase 3 extension)
# ---------------------------------------------------------------------------
#
# The 2-skill FRQ path above combines SE_ability with SE_param only. The BiGGen
# gemini-3 recalibration adds a THIRD, judge-measurement-error term (SE_judge, the
# Monte-Carlo SD of the of-record theta under the confusion-model resample). The
# same quadrature idea extends to any number of independent SE components:
#
#     SE_total = sqrt(SE_ability^2 + SE_param^2 + SE_judge^2)
#
# ``se_quadrature`` is the generic combiner (NaN treated as 0); ``biggen_se_judge_summary``
# refreshes SE_total on the driver's per-model CSVs and emits a compact JSON summary. The
# heavy resampling / engine calls live in the LOCAL scratch driver
# ``reports/biggen_gemini3_recal/scratch/se_judge_gemini3.py`` (this file only owns the
# quadrature + reporting, never the CAT engine).

BIGGEN_JUDGE = "gemini-3-flash-preview"


def se_quadrature(*components) -> np.ndarray:
    """SE_total = sqrt(sum of squares) over any number of SE component arrays (NaN -> 0)."""
    stacked = np.array([np.nan_to_num(np.asarray(c, dtype=float), nan=0.0) for c in components])
    return np.sqrt((stacked ** 2).sum(axis=0))


def _biggen_regime_stats(df: pd.DataFrame) -> dict:
    """Medians/maxes for one regime's per-model SE decomposition (SE_total recomputed)."""
    df = df.copy()
    df["se_total"] = se_quadrature(df["se_ability"], df["se_param"], df["se_judge"])
    theta_shift = (df["theta_debiased"] - df["theta"]) if "theta_debiased" in df else None
    out = {
        "n_models": int(len(df)),
        "se_ability_median": float(df["se_ability"].median()),
        "se_param_median": float(df["se_param"].median()),
        "se_judge_median": float(df["se_judge"].median()),
        "se_judge_mean": float(df["se_judge"].mean()),
        "se_judge_max": float(df["se_judge"].max()),
        "se_total_median": float(df["se_total"].median()),
        "se_total_mean": float(df["se_total"].mean()),
        # Share of SE_total^2 variance carried by the judge term (median across models).
        "se_judge_var_share_median": float(
            (df["se_judge"] ** 2 / df["se_total"] ** 2).median()),
        "se_judge_dominates_frac": float(
            (df["se_judge"] > np.sqrt(df["se_ability"] ** 2 + df["se_param"] ** 2)).mean()),
    }
    if theta_shift is not None:
        out["theta_debias_shift_median"] = float(theta_shift.median())
        out["theta_debias_shift_mean"] = float(theta_shift.mean())
    return out


def biggen_se_judge_summary(se_dir: Path) -> dict:
    """Recompute SE_total quadrature on the driver CSVs and return a compact summary.

    Reads ``per_model_full_bank.csv`` and ``per_model_oppoint_f10se12.csv``, writes the
    refreshed SE_total back in-place, and returns a dict comparing the two regimes plus the
    judge-term dominance diagnostics. Stamps judge = gemini-3-flash-preview.
    """
    out: dict = {"judge": BIGGEN_JUDGE, "regimes": {}}
    for tag, name in (("full_bank", "per_model_full_bank.csv"),
                      ("oppoint_f10se12", "per_model_oppoint_f10se12.csv")):
        path = se_dir / name
        if not path.is_file():
            continue
        df = pd.read_csv(path)
        df["se_total"] = se_quadrature(df["se_ability"], df["se_param"], df["se_judge"])
        df.to_csv(path, index=False)
        out["regimes"][tag] = _biggen_regime_stats(df)
    fb = out["regimes"].get("full_bank")
    op = out["regimes"].get("oppoint_f10se12")
    if fb and op:
        out["op_vs_full"] = {
            "se_judge_median_ratio": (op["se_judge_median"] / fb["se_judge_median"]
                                      if fb["se_judge_median"] > 0 else float("nan")),
            "se_judge_median_full": fb["se_judge_median"],
            "se_judge_median_op": op["se_judge_median"],
        }
    return out


def build_total_se_csv(out_dir: Path) -> pd.DataFrame:
    lb = pd.read_csv(BASE / "leaderboard" / "2_skills" / "cat_per_model.csv",
                     index_col="model")
    pu = pd.read_csv(BASE / "param_uncertainty" / "2_skills" / "leaderboard_se_components.csv",
                     index_col="model")
    rows = []
    for m in lb.index:
        if m not in pu.index:
            continue
        for d in DIMS:
            se_ab = float(lb.loc[m, f"final_se_{d}"])
            se_pa = float(pu.loc[m, f"se_param_{d}"])
            rows.append({"model": m, "skill": d, "se_ability": se_ab,
                         "se_param": se_pa,
                         "se_total": float(np.sqrt(se_ab**2 + se_pa**2))})
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "frq_total_se_per_model.csv", index=False)
    return df


def figure_mean_total_se(df: pd.DataFrame, fig_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(DIMS))
    width = 0.38
    ab_mean = [df[df.skill == d]["se_ability"].mean() for d in DIMS]
    ab_sd = [df[df.skill == d]["se_ability"].std() for d in DIMS]
    to_mean = [df[df.skill == d]["se_total"].mean() for d in DIMS]
    to_sd = [df[df.skill == d]["se_total"].std() for d in DIMS]

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    b1 = ax.bar(x - width / 2, ab_mean, width, yerr=ab_sd, capsize=4,
                color=COLORS["ability"], label="SE ability-only (params known)")
    b2 = ax.bar(x + width / 2, to_mean, width, yerr=to_sd, capsize=4,
                color=COLORS["total"], label="SE total (+ calibration uncertainty)")
    ax.axhline(SE_TARGET, ls="--", color="gray", lw=1, label=f"SE target = {SE_TARGET:.2f}")
    top = max([m + s for m, s in zip(ab_mean, ab_sd)] + [m + s for m, s in zip(to_mean, to_sd)])
    ax.set_ylim(0, top * 1.18)
    # value labels ABOVE the upper SD whisker so they never collide with it
    for bars, means, sds in [(b1, ab_mean, ab_sd), (b2, to_mean, to_sd)]:
        for brc, v, sd in zip(bars, means, sds):
            ax.text(brc.get_x() + brc.get_width() / 2, v + sd + 0.008, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(DIMS)
    ax.set_ylabel("mean ability SE across 115 models")
    ax.set_title("FRQ 2-skill ability SE: ability-only vs total\n"
                 "(error bars = SD across models)", fontsize=11)
    # legend below the axes so it never sits on the bars/whiskers
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.11),
              ncol=2, framealpha=0.9)
    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "frq_mean_total_se_2skill.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def se_trajectories(runs_dir: Path, workers: int) -> dict[int, dict[str, list[float]]]:
    """Run the engine with per-scenario logging; return {step -> {skill -> [se over models]}}."""
    bank = ROOT / "data" / "TutorBench" / "rubrics_qmatrix_calibrated_2skill_115_fitted.jsonl"
    matrix = ROOT / "staging" / "response_matrix_full_nonopt_115.csv"
    scen = ROOT / "data" / "TutorBench" / "scenarios.jsonl"
    records, dims, _ = scat.load_fitted_bank(bank, "clamp")
    rubrics = scat.build_rubrics(records, dims)
    scen_raw = scat.load_scenarios(scen)
    mat = pd.read_csv(matrix, index_col=0)
    spec = scat.RunSpec(seed=42, top_n=5, max_se=SE_TARGET, min_evals_per_skill=15,
                        min_scenarios=12, max_scenarios=50, selection="trace",
                        mode="cat", runs_dir=str(runs_dir), write_logs=True)
    if runs_dir.exists():
        shutil.rmtree(runs_dir, ignore_errors=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    by_step: dict[int, dict[str, list[float]]] = {}
    for m in mat.index:
        rec = scat.run_one_model(m, rubrics, scen_raw, mat.loc[m], dims, spec, cleanup=False)
        run_id = f"offline_{m.replace('/', '_')}_{spec.selection}_{spec.mode}_s{spec.seed}"
        steps_path = runs_dir / run_id / "steps.jsonl"
        if not steps_path.is_file():
            continue
        with steps_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                step = int(obj["step"])
                se = obj["se"]
                slot = by_step.setdefault(step, {d: [] for d in dims})
                for k, d in enumerate(dims):
                    slot[d].append(float(se[k]))
    shutil.rmtree(runs_dir, ignore_errors=True)
    return by_step


def figure_shrinkage(by_step: dict, fig_dir: Path, min_models: int = 5):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = sorted(by_step)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    palette = {"correctness": "#2b8cbe", "scaffolding": "#d95f0e"}
    for d in DIMS:
        xs, means, sds = [], [], []
        for s in steps:
            vals = np.array(by_step[s][d])
            if vals.size < min_models:
                continue
            xs.append(s)
            means.append(vals.mean())
            sds.append(vals.std())
        xs = np.array(xs); means = np.array(means); sds = np.array(sds)
        ax.plot(xs, means, color=palette[d], label=d)
        ax.fill_between(xs, means - sds, means + sds, color=palette[d], alpha=0.18)
    ax.axhline(SE_TARGET, ls="--", color="gray", lw=1, label=f"SE target = {SE_TARGET:.2f}")
    ax.set_xlabel("scenarios administered")
    ax.set_ylabel("mean posterior (ability) SE")
    ax.set_title("FRQ 2-skill SE shrinkage vs test length\n"
                 f"(shaded +/-1 SD; steps with >= {min_models} models)", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "frq_se_shrinkage_2skill.png", dpi=140)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=BASE / "frq_mae")
    ap.add_argument("--runs-dir", type=Path, default=ROOT / "staging" / "engine_runs_se_traj")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--skip-shrinkage", action="store_true")
    ap.add_argument("--biggen-se-judge-dir", type=Path, default=None,
                    help="BiGGen Phase-3 mode: recompute SE_total quadrature on the "
                         "gemini-3 SE_judge driver CSVs in this dir and print the summary.")
    args = ap.parse_args()

    if args.biggen_se_judge_dir is not None:
        summary = biggen_se_judge_summary(args.biggen_se_judge_dir)
        print(json.dumps(summary, indent=2))
        return 0

    fig_dir = args.out_dir / "figures"
    df = build_total_se_csv(args.out_dir)
    figure_mean_total_se(df, fig_dir)

    print("=" * 80)
    print("FRQ 2-skill ability-SE decomposition (mean across 115 models)")
    for d in DIMS:
        sub = df[df.skill == d]
        print(f"  {d:12s}: SE_ability={sub.se_ability.mean():.3f}  "
              f"SE_param={sub.se_param.mean():.3f}  SE_total={sub.se_total.mean():.3f}")

    if not args.skip_shrinkage:
        print("\nrunning engine with per-scenario logging for SE trajectories ...", flush=True)
        by_step = se_trajectories(args.runs_dir, args.workers)
        figure_shrinkage(by_step, fig_dir)
        print(f"  parsed {len(by_step)} distinct step positions")

    print(f"\nwrote -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
