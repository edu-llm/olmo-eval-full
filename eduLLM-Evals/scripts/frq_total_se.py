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
    args = ap.parse_args()

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
