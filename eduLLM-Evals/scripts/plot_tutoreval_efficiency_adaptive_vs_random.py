"""Adaptive-vs-random CAT efficiency curve for TutorEval (recovery r vs test length).

This mirrors the TutorBench adaptive-vs-random efficiency comparison
(``scripts/frq_mae_analysis.py`` Deliverable 3, whose ``random`` arm is produced by
``scripts/offline_engine_driver.py --mode baseline``). The definitions are taken verbatim
from that pipeline:

* **random selection** = the production engine's built-in non-adaptive baseline
  (``tutor_cat.engine.run_evaluation(..., mode="baseline")``), which fixes a *seeded-random
  scenario order* up front (``rng.permutation`` of the bank) and administers it in that order,
  reusing the exact same per-criterion M2PL updates as the adaptive arm.
* **adaptive selection** = the same engine in ``mode="cat"`` with ``selection="trace"`` (the
  report's headline rule): each step picks the scenario maximising per-criterion Fisher
  information for the widest-SE skill.
* **reference** = full-bank EAP ability over every graded criterion for the model
  (``scenario_cat_lib.eap_all_models``), the selection-independent target.
* **recovery r** = Pearson correlation between the full-bank reference ability and the MWLE
  ability estimated from the administered subset (MWLE is the report's headline estimator).

TutorBench's *figure* is a two-bar chart (adaptive vs random mean scenarios/criteria at a
fixed SE target). Here we generalise it into the efficiency **curve** the report references:
we sweep the test length by capping ``max_scenarios`` at a grid of scenario budgets while
disabling the precision stop (``max_se`` -> 0 so no arm halts early), so at each budget both
arms administer exactly that many scenarios. Because both the adaptive selection state and the
random permutation are prefix-consistent, one run per (arm, seed) at the largest budget yields
every shorter budget by truncation. The random arm (and, since ``top_n``>1 makes adaptive
selection mildly stochastic, the adaptive arm too) is averaged over several seeds with a
min-max band.

Nothing here reruns or modifies any tracked result; it drives the unmodified production engine
over the canonical response matrix. ProcessPool parallelism (``--workers``>1) must run
unsandboxed.

Usage
-----
    uv run --project .. python scripts/plot_tutoreval_efficiency_adaptive_vs_random.py \
        --workers 8
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "staging" / "_efficiency" / ".mplconfig"))

import sys  # noqa: E402

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scripts.scenario_cat_lib as scat  # noqa: E402

DATA = ROOT / "data" / "TutorEval"
STAGING = ROOT / "staging" / "_efficiency"
SCEN = ROOT / "regenerated_figures" / "scenario_level"

DEFAULT_BUDGETS = (2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50)
DEFAULT_SEEDS = tuple(20260729 + i for i in range(10))
ARMS = {"adaptive": "cat", "random": "baseline"}
ARM_COLOR = {"adaptive": "#1f77b4", "random": "#d95f0e"}


def recovery(x: np.ndarray, y: np.ndarray) -> dict:
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    return {
        "r": float(np.corrcoef(x, y)[0, 1]),
        "slope": float(np.polyfit(x, y, 1)[0]),
        "mae": float(np.mean(np.abs(y - x))),
        "n": int(x.size),
    }


def prefix_criteria_idx(
    order: list[str], cid2scen: dict[str, str], col: dict[str, int], budget: int
) -> np.ndarray:
    """Column indices for the criteria falling in the first ``budget`` administered scenarios.

    The engine grades a scenario's criteria consecutively, so counting distinct scenarios in
    administration order and stopping once ``budget`` are seen reproduces the exact criterion
    set the engine would have graded had it been capped at ``max_scenarios=budget``.
    """
    seen: set[str] = set()
    idxs: list[int] = []
    for cid in order:
        sid = cid2scen.get(cid)
        if sid not in seen:
            if len(seen) >= budget:
                break
            seen.add(sid)
        j = col.get(cid)
        if j is not None:
            idxs.append(j)
    return np.array(idxs, dtype=int)


def run_arm(
    mode: str,
    seed: int,
    models: list[str],
    bank: Path,
    matrix: Path,
    scenarios: Path,
    negative_policy: str,
    dims: list[str],
    selection: str,
    max_budget: int,
    workers: int,
) -> dict[str, list[str]]:
    """Run one (arm, seed) over all models; return {model: administered criterion order}.

    ``max_se`` is set to 0.0 so the precision stop never fires; every model therefore
    administers scenarios until the ``max_scenarios`` cap, giving a prefix from which any
    shorter budget is recovered by truncation.
    """
    spec = scat.RunSpec(
        seed=seed,
        top_n=5,
        max_se=0.0,
        min_evals_per_skill=0,
        min_scenarios=0,
        max_scenarios=max_budget,
        unmapped_criteria="judge",
        selection=selection,
        mode=mode,
        runs_dir=str(STAGING / "engine_runs"),
        write_logs=False,
    )
    results = scat.run_models(
        models, bank, matrix, scenarios, negative_policy, dims, spec, workers=workers
    )
    return {r["model"]: r["order"] for r in results}


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--bank", type=Path, default=DATA / "rubrics_qmatrix_final_unidim_fitted.jsonl")
    p.add_argument(
        "--matrix", type=Path, default=ROOT / "runs" / "judge" / "TutorEval" / "response_matrix.csv"
    )
    p.add_argument("--scenarios", type=Path, default=DATA / "scenarios_final.jsonl")
    p.add_argument("--selection", choices=("trace", "dopt"), default="trace")
    p.add_argument("--negative-policy", choices=("clamp", "keep", "drop"), default="clamp")
    p.add_argument("--budgets", type=int, nargs="+", default=list(DEFAULT_BUDGETS))
    p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--grid", type=int, default=61)
    p.add_argument("--range", type=float, default=8.0)
    p.add_argument("--out-png", type=Path, default=SCEN / "efficiency_r_vs_scenarios.png")
    p.add_argument("--out-csv", type=Path, default=STAGING / "efficiency_adaptive_vs_random.csv")
    p.add_argument(
        "--use-cache",
        action="store_true",
        help="skip the engine sweep and re-plot from an existing --out-csv.",
    )
    args = p.parse_args()

    budgets = sorted(set(args.budgets))
    max_budget = max(budgets)

    if args.use_cache and args.out_csv.is_file():
        long_df = pd.read_csv(args.out_csv)
    else:
        records, dims, _ = scat.load_fitted_bank(args.bank, args.negative_policy)
        ids, A, b = scat.assemble_arrays(records, dims)
        col = {c: i for i, c in enumerate(ids)}
        cid2scen = {r["criterion_id"]: r["scenario_id"] for r in records}

        matrix = pd.read_csv(args.matrix, index_col=0)
        models = list(matrix.index)
        sub = matrix.reindex(columns=ids)
        Yraw = sub.to_numpy(dtype=float)
        mask = ~np.isnan(Yraw)
        Y = np.nan_to_num(Yraw, nan=0.0)

        grid, log_prior = scat.build_grid(len(dims), args.grid, args.range)
        print(
            f"bank={args.bank.name} dims={dims} models={len(models)} "
            f"criteria={len(ids)} budgets={budgets} seeds={list(args.seeds)}"
        )
        print(f"full-bank EAP reference ({grid.shape[0]:,} nodes) ...")
        theta_full = scat.eap_all_models(Y, mask, A, b, grid, log_prior, 4096)
        ref = theta_full[:, 0]
        row_of = {m: i for i, m in enumerate(matrix.index)}

        rows = []
        for arm, mode in ARMS.items():
            for seed in args.seeds:
                print(
                    f"  running arm={arm:8s} seed={seed} "
                    f"(max_scenarios={max_budget}, workers={args.workers}) ..."
                )
                orders = run_arm(
                    mode,
                    seed,
                    models,
                    args.bank,
                    args.matrix,
                    args.scenarios,
                    args.negative_policy,
                    dims,
                    args.selection,
                    max_budget,
                    args.workers,
                )
                for budget in budgets:
                    est = np.full(len(models), np.nan)
                    for mi, m in enumerate(models):
                        idx = prefix_criteria_idx(orders[m], cid2scen, col, budget)
                        if idx.size == 0:
                            continue
                        r = row_of[m]
                        th_ba = scat.eap_subset(Y[r], idx, A, b, grid, log_prior)
                        th_mw, _ = scat.mwle_subset(Y[r], idx, A, b, th_ba)
                        est[mi] = th_mw[0]
                    rec = recovery(ref, est)
                    rows.append(
                        {
                            "arm": arm,
                            "seed": seed,
                            "budget": budget,
                            "r": rec["r"],
                            "slope": rec["slope"],
                            "mae": rec["mae"],
                            "n": rec["n"],
                        }
                    )
        long_df = pd.DataFrame(rows)
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        long_df.to_csv(args.out_csv, index=False)
        print(f"wrote -> {args.out_csv}")

    # -------- aggregate over seeds (mean + min/max band) and plot ------------
    agg = long_df.groupby(["arm", "budget"])["r"].agg(["mean", "min", "max"]).reset_index()
    n_models = int(long_df["n"].max())

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for arm in ARMS:
        a = agg[agg["arm"] == arm].sort_values("budget")
        color = ARM_COLOR[arm]
        ax.fill_between(a["budget"], a["min"], a["max"], color=color, alpha=0.15)
        ax.plot(a["budget"], a["mean"], "-o", color=color, lw=2.2, label=f"{arm} selection")
    ax.set_xlabel("scenarios administered (test length)")
    ax.set_ylabel("recovery r (MWLE vs full-bank ability)")
    ax.set_title(
        f"TutorEval CAT efficiency: adaptive vs random selection\n"
        f"(unidimensional, MWLE, N={n_models}; band = min-max over "
        f"{long_df['seed'].nunique()} seeds)",
        fontsize=10,
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=140)
    plt.close(fig)
    print(f"wrote -> {args.out_png}")

    # -------- console summary at representative lengths ----------------------
    print("\nrecovery r (mean over seeds) at representative test lengths:")
    piv = agg.pivot(index="budget", columns="arm", values="mean")
    for budget in budgets:
        if budget in piv.index:
            adp = piv.loc[budget, "adaptive"]
            rnd = piv.loc[budget, "random"]
            print(
                f"  {budget:3d} scenarios: adaptive r={adp:.3f}  random r={rnd:.3f}  "
                f"(gap {adp - rnd:+.3f})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
