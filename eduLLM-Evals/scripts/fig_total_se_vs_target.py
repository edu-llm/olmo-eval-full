"""Paper figure: total (parameter-inflated) SE vs se_target, with the parameter floor.

Replaces the obvious "convergence/length vs se_target" tradeoff with the informative one:
tightening the stopping target shrinks the posterior SE but NOT the ~se-independent
parameter-uncertainty floor, so the achieved *total* SE flattens toward that floor — tighter
targets buy test length, not real precision. A secondary line shows mean scenarios so the
length cost of chasing the floor is visible.

Reads the floor-12 se-sweep runs + the parameter-uncertainty run:
    regenerated_figures/scenario_level_115_min12/se_sweep/{2_skills,3_skills}/se*/{metrics.json,cat_per_model.csv}
    regenerated_figures/scenario_level_115_min12/param_uncertainty/{2_skills,3_skills}/metrics.json

Usage
-----
    python scripts/fig_total_se_vs_target.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _se_param(param_dir: Path, skill: str) -> float:
    m = json.load(open(param_dir / "metrics.json", encoding="utf-8"))
    return float(m["se_components"][skill]["median_se_param"])


def _se_param_ci(per_model_csv: Path, skill: str, b: int = 2000, seed: int = 0):
    """Median parameter-SE floor + a bootstrap 95% CI, resampling the per-model se_param
    values (only the median is stored in metrics.json). This single source of uncertainty
    is propagated to BOTH the floor line and the total-SE curve, so their bands are
    comonotonic (driven by the same se_param), not two independent error estimates."""
    d = pd.read_csv(per_model_csv)
    v = d.loc[d["skill"] == skill, "se_param"].to_numpy(float)
    v = v[np.isfinite(v)]
    med = float(np.median(v))
    rng = np.random.default_rng(seed)
    boots = np.median(v[rng.integers(0, v.size, (b, v.size))], axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return med, float(lo), float(hi)


def _bank_rows(sweep_dir: Path, bank: str, se_param: float, skill: str):
    rows = []
    for d in sorted((sweep_dir / bank).glob("se*")):
        mp, cm = d / "metrics.json", d / "cat_per_model.csv"
        if not (mp.exists() and cm.exists()):
            continue
        m = json.load(open(mp, encoding="utf-8"))
        se_t = float(m["config"]["max_se"])
        scen = float(m["scenarios_administered"]["mean"])
        se_post = float(pd.read_csv(cm)[f"final_se_{skill}"].median())
        total = float(np.sqrt(se_post**2 + se_param**2))
        rows.append((se_t, se_post, total, scen))
    return sorted(rows, key=lambda r: r[0])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    base = ROOT / "regenerated_figures" / "scenario_level_115_min12"
    p.add_argument("--sweep-dir", type=Path, default=base / "se_sweep")
    p.add_argument("--param-dir", type=Path, default=base / "param_uncertainty")
    p.add_argument("--per-model-se", type=Path,
                   default=base / "frq_mae" / "frq_total_se_per_model.csv",
                   help="per-model se_param source for the floor's bootstrap CI.")
    p.add_argument("--bank", default="2_skills",
                   help="which fit's se-sweep to use (default 2-skill; the SE-target "
                        "decision is introduced after 3-skill is ruled out).")
    p.add_argument("--skills", nargs="+", default=["correctness", "scaffolding"],
                   help="one panel per skill, all from the same bank, on a single figure.")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    bank = args.bank
    if args.out is None:
        tag = "2skill" if bank == "2_skills" else bank
        args.out = base / "figures" / f"total_se_vs_target_{tag}.png"

    if not (args.sweep_dir / bank).exists():
        raise SystemExit(f"no se_sweep dir for bank {bank} under {args.sweep_dir}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    skills = args.skills
    fig, axes = plt.subplots(1, len(skills), figsize=(5.4 * len(skills), 4.6), squeeze=False)
    for ax, skill in zip(axes[0], skills):
        se_param, sp_lo, sp_hi = _se_param_ci(args.per_model_se, skill)
        rows = _bank_rows(args.sweep_dir, bank, se_param, skill)
        se = [r[0] for r in rows]
        post = [r[1] for r in rows]
        total = [r[2] for r in rows]
        scen = [r[3] for r in rows]
        # the parameter-SE CI is a single source; push it through total = sqrt(post^2 +
        # se_param^2) so the total band and the floor band move together (comonotonic).
        total_lo = [float(np.sqrt(p**2 + sp_lo**2)) for p in post]
        total_hi = [float(np.sqrt(p**2 + sp_hi**2)) for p in post]

        ax.plot(se, total, "o-", color="#c1666b", lw=2, label="total SE (posterior ⊕ parameter)")
        ax.fill_between(se, total_lo, total_hi, color="#c1666b", alpha=0.18,
                        label="parameter-SE 95% CI")
        ax.plot(se, post, "^--", color="#4d648d", lw=1.2, alpha=0.8, label="posterior SE only")
        ax.axhline(se_param, ls=":", color="#555", lw=1.4,
                   label=f"parameter-SE floor ({se_param:.3f})")
        ax.axhspan(sp_lo, sp_hi, color="#555", alpha=0.10)  # same CI as the total band
        ax.set_xlabel("se_target")
        ax.set_ylabel(f"{skill} SE (logits)")
        ax.set_ylim(0, max(total_hi) * 1.15)
        ax.set_title(skill)
        ax2 = ax.twinx()
        ax2.plot(se, scen, "s--", color="#d95f0e", lw=1.4, label="mean scenarios")
        ax2.set_ylabel("mean scenarios administered", color="#d95f0e")
        ax2.tick_params(axis="y", labelcolor="#d95f0e")
        # combined legend in the empty lower-left so it never covers the SE curves
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=7.5, framealpha=0.9)

    fig.suptitle("Tighter targets buy mostly test length: total SE flattens toward the "
                 f"parameter floor ({bank.replace('_', '-')}, min_scenarios=12)", fontsize=11)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"wrote -> {args.out}")
    for skill in skills:
        sp, lo, hi = _se_param_ci(args.per_model_se, skill)
        print(f"\n{bank} [{skill}]  (SE_param floor = {sp:.3f} [{lo:.3f}, {hi:.3f}])")
        for se_t, post, total, scen in _bank_rows(args.sweep_dir, bank, sp, skill):
            print(f"  se={se_t:.2f}  postSE={post:.3f}  totalSE={total:.3f}  scen={scen:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
