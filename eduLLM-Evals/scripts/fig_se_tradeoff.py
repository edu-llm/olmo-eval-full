"""Paper figure: the se_target precision/length tradeoff (at the locked min_scenarios floor).

For each bank, plots how the stopping-precision target trades convergence and test length:
x = se_target; left axis = fraction of models reaching precision; right axis = mean scenarios
administered. Makes the "tighter target buys little once the SE_param floor dominates, but
collapses convergence" point visually.

Reads the per-run leaderboard metrics.json written by the se_target sweep at floor 12:
    regenerated_figures/scenario_level_115_min12/se_sweep/{2_skills,3_skills}/se{val}/metrics.json

Usage
-----
    python scripts/fig_se_tradeoff.py \
        --sweep-dir regenerated_figures/scenario_level_115_min12/se_sweep \
        --out regenerated_figures/scenario_level_115_min12/figures/se_tradeoff.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_bank(sweep_dir: Path, bank: str):
    """Glob every se* run dir and read the se target from its own metrics.json
    (robust to se0.2 vs se0.20 directory-naming)."""
    rows = []
    for mpath in sorted((sweep_dir / bank).glob("se*/metrics.json")):
        m = json.load(open(mpath, encoding="utf-8"))
        se = float(m["config"]["max_se"])
        n = m.get("n_models", 115)
        rows.append((se, m["precision_reached"] / n,
                     m["scenarios_administered"]["mean"]))
    return sorted(rows, key=lambda r: r[0])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep-dir", type=Path,
                   default=ROOT / "regenerated_figures" / "scenario_level_115_min12" / "se_sweep")
    p.add_argument("--out", type=Path,
                   default=ROOT / "regenerated_figures" / "scenario_level_115_min12"
                   / "figures" / "se_tradeoff.png")
    p.add_argument("--banks", nargs="+", default=["2_skills"],
                   help="banks to panel; default 2-skill only (the SE-target decision is "
                        "introduced after 3-skill is already ruled out).")
    args = p.parse_args()

    banks = [b for b in args.banks if (args.sweep_dir / b).exists()]
    if not banks:
        raise SystemExit(f"no se_sweep bank dirs under {args.sweep_dir}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(banks), figsize=(5.2 * len(banks), 4.6), squeeze=False)
    for ax, bank in zip(axes[0], banks):
        rows = _load_bank(args.sweep_dir, bank)
        se = [r[0] for r in rows]
        prec = [100 * r[1] for r in rows]
        scen = [r[2] for r in rows]
        ax.plot(se, prec, "o-", color="#2c7fb8", label="% reaching precision")
        ax.set_xlabel("se_target")
        ax.set_ylabel("% models converged", color="#2c7fb8")
        ax.set_ylim(0, 105)
        ax.tick_params(axis="y", labelcolor="#2c7fb8")
        ax2 = ax.twinx()
        ax2.plot(se, scen, "s--", color="#d95f0e", label="mean scenarios")
        ax2.set_ylabel("mean scenarios administered", color="#d95f0e")
        ax2.tick_params(axis="y", labelcolor="#d95f0e")
        ax.set_title(f"{bank.replace('_', '-')} (min_scenarios=12)")
    fig.suptitle("Stopping precision vs convergence and test length", fontsize=12)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"wrote -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
