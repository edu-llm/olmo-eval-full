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
    p.add_argument("--skill", default="correctness",
                   help="latent skill axis to plot (correctness / scaffolding / presentation).")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    skill = args.skill
    if args.out is None:
        args.out = base / "figures" / f"total_se_vs_target_{skill}.png"

    banks = [b for b in ("2_skills", "3_skills") if (args.sweep_dir / b).exists()]
    if not banks:
        raise SystemExit(f"no se_sweep bank dirs under {args.sweep_dir}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(banks), figsize=(5.4 * len(banks), 4.6), squeeze=False)
    for ax, bank in zip(axes[0], banks):
        se_param = _se_param(args.param_dir / bank, skill)
        rows = _bank_rows(args.sweep_dir, bank, se_param, skill)
        se = [r[0] for r in rows]
        post = [r[1] for r in rows]
        total = [r[2] for r in rows]
        scen = [r[3] for r in rows]

        ax.plot(se, total, "o-", color="#c1666b", lw=2, label="total SE (posterior ⊕ parameter)")
        ax.plot(se, post, "^--", color="#4d648d", lw=1.2, alpha=0.8, label="posterior SE only")
        ax.axhline(se_param, ls=":", color="#555", lw=1.4,
                   label=f"parameter-SE floor ({se_param:.3f})")
        ax.set_xlabel("se_target")
        ax.set_ylabel(f"{skill} SE (logits)")
        ax.set_ylim(0, max(total) * 1.15)
        ax.set_title(f"{bank.replace('_', '-')} (min_scenarios=12)")
        ax2 = ax.twinx()
        ax2.plot(se, scen, "s--", color="#d95f0e", lw=1.4, label="mean scenarios")
        ax2.set_ylabel("mean scenarios administered", color="#d95f0e")
        ax2.tick_params(axis="y", labelcolor="#d95f0e")
        # combined legend
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper center", fontsize=7.5)

    fig.suptitle(f"Tighter targets buy test length, not precision: total {skill} SE flattens to "
                 "the parameter floor", fontsize=11)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"wrote -> {args.out}")
    for bank in banks:
        sp = _se_param(args.param_dir / bank, skill)
        print(f"\n{bank} [{skill}]  (SE_param floor = {sp:.3f})")
        for se_t, post, total, scen in _bank_rows(args.sweep_dir, bank, sp, skill):
            print(f"  se={se_t:.2f}  postSE={post:.3f}  totalSE={total:.3f}  scen={scen:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
