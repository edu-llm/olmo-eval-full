"""SCENARIO-LEVEL path / seed dependence of the CAT ability estimate.

Scenario-level replacement for the item-level ``random_order_experiment.py``. An online
CAT's adaptive path is driven by the run seed (the engine picks among the top-n scenarios
with a seeded RNG). Different seeds therefore yield different administered scenario SETS and
orders — exactly the real-world variability of an online run. This measures how reproducible
each final estimator (online / batch EAP / MWLE) is across seeds.

Unlike the item-level version, we do NOT hold a fixed item set and shuffle order (an online
CAT never does that). We vary the run seed and let the real engine choose adaptively, which
is the honest scenario-level question: "run the same model through the CAT K times — how much
does its ability estimate move?"

Parallel across models within each seed via ``--workers``.

Usage
-----
    python scripts/scenario_order_experiment.py \
        --bank data/TutorBench/rubrics_qmatrix_calibrated_2skill_fitted.jsonl \
        --matrix staging/response_matrix_full_nonopt.csv \
        --out-dir regenerated_figures/scenario_level/order/2_skills --n-seeds 8 --workers 16
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.scenario_cat_lib as scat  # noqa: E402

ESTIMATORS = ("online", "batch", "mwle")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--matrix", type=Path, required=True)
    p.add_argument("--scenarios", type=Path,
                   default=ROOT / "data" / "TutorBench" / "scenarios.jsonl")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--runs-dir", type=Path, default=ROOT / "staging" / "engine_runs_ord")
    p.add_argument("--negative-policy", choices=("clamp", "keep", "drop"), default="clamp")
    p.add_argument("--n-seeds", type=int, default=8)
    p.add_argument("--base-seed", type=int, default=1000)
    p.add_argument("--selection", choices=("trace", "dopt"), default="trace")
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--max-se", type=float, default=0.30)
    p.add_argument("--min-evals-per-skill", type=int, default=15)
    p.add_argument("--max-scenarios", type=int, default=50)
    p.add_argument("--grid", type=int, default=61)
    p.add_argument("--range", type=float, default=6.0)
    p.add_argument("--workers", type=int, default=scat.default_workers())
    args = p.parse_args()

    records, dims, neg = scat.load_fitted_bank(args.bank, args.negative_policy)
    prov = scat.verify_provenance(args.bank, args.matrix)
    ids, A, b = scat.assemble_arrays(records, dims)
    matrix = pd.read_csv(args.matrix, index_col=0)
    sub = matrix.reindex(columns=ids)
    Y = np.nan_to_num(sub.to_numpy(float), nan=0.0)
    mask = ~np.isnan(sub.to_numpy(float))
    col = {c: i for i, c in enumerate(ids)}
    models = list(matrix.index)
    row_of = {m: i for i, m in enumerate(models)}
    n_models, n_dims = len(models), len(dims)
    print(f"bank={args.bank.name} dims={dims} models={n_models} seeds={args.n_seeds} "
          f"selection={args.selection} aligned={prov['aligned']} workers={args.workers}")

    grid, log_prior = scat.build_grid(n_dims, args.grid, args.range)
    theta_full = scat.eap_all_models(Y, mask, A, b, grid, log_prior)

    # theta[estimator] -> (n_seeds, n_models, n_dims)
    theta = {e: np.full((args.n_seeds, n_models, n_dims), np.nan) for e in ESTIMATORS}
    n_scen = np.full((args.n_seeds, n_models), np.nan)
    for s in range(args.n_seeds):
        seed = args.base_seed + s
        spec = scat.RunSpec(seed=seed, top_n=args.top_n, max_se=args.max_se,
                            min_evals_per_skill=args.min_evals_per_skill,
                            max_scenarios=args.max_scenarios, selection=args.selection,
                            runs_dir=str(args.runs_dir))
        results = scat.run_models(models, args.bank, args.matrix, args.scenarios,
                                  args.negative_policy, dims, spec, workers=args.workers)
        for rec0 in results:
            r = row_of[rec0["model"]]
            idx = np.array([col[c] for c in rec0["order"] if c in col], dtype=int)
            th_on = np.asarray(rec0["theta_online"], float)
            th_ba = scat.eap_subset(Y[r], idx, A, b, grid, log_prior) if idx.size else th_on.copy()
            th_mw, _ = (scat.mwle_subset(Y[r], idx, A, b, th_ba) if idx.size else (th_ba.copy(), True))
            theta["online"][s, r] = th_on
            theta["batch"][s, r] = th_ba
            theta["mwle"][s, r] = th_mw
            n_scen[s, r] = rec0["scenarios_administered"]
        print(f"  seed {seed}: mean scenarios={np.nanmean(n_scen[s]):.1f}", flush=True)

    # across-seed spread per estimator per dim
    spread = {}
    for e in ESTIMATORS:
        spread[e] = {}
        for k, d in enumerate(dims):
            sd = np.nanstd(theta[e][:, :, k], axis=0)          # per model
            rng = np.nanmax(theta[e][:, :, k], axis=0) - np.nanmin(theta[e][:, :, k], axis=0)
            spread[e][d] = {"mean_sd": float(np.nanmean(sd)), "median_sd": float(np.nanmedian(sd)),
                            "max_sd": float(np.nanmax(sd)), "mean_range": float(np.nanmean(rng)),
                            "max_range": float(np.nanmax(rng))}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # per-model spread table (online vs batch vs mwle) on the first dim + all dims
    rows = []
    for r, m in enumerate(models):
        rec = {"model": m, "mean_scenarios": float(np.nanmean(n_scen[:, r]))}
        for e in ESTIMATORS:
            for k, d in enumerate(dims):
                rec[f"{e}_{d}_sd"] = float(np.nanstd(theta[e][:, r, k]))
                rec[f"{e}_{d}_mean"] = float(np.nanmean(theta[e][:, r, k]))
                rec[f"full_{d}"] = float(theta_full[r, k])
        rows.append(rec)
    pd.DataFrame(rows).to_csv(args.out_dir / "per_model_spread.csv", index=False)

    metrics = {"generated_at": datetime.now(timezone.utc).isoformat(),
               "purpose": "scenario-level seed/path dependence of the CAT estimate (real engine)",
               "bank": str(args.bank), "matrix": str(args.matrix), "dims": dims,
               "provenance": prov, "n_models": n_models, "n_seeds": args.n_seeds,
               "selection": args.selection,
               "config": {"base_seed": args.base_seed, "max_se": args.max_se,
                          "min_evals_per_skill": args.min_evals_per_skill,
                          "max_scenarios": args.max_scenarios, "grid": args.grid},
               "across_seed_spread": spread}
    with (args.out_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    _figures(theta, theta_full, dims, spread, args.out_dir / "figures")

    print("\n" + "=" * 90)
    print("SCENARIO-LEVEL SEED DEPENDENCE (theta spread across seeds)")
    print("=" * 90)
    for e in ESTIMATORS:
        line = "  ".join(f"{d[:4]} meanSD={spread[e][d]['mean_sd']:.3f} "
                         f"maxRange={spread[e][d]['max_range']:.2f}" for d in dims)
        print(f"  {e:7s}: {line}")
    print(f"\nwrote -> {args.out_dir}")
    return 0


def _figures(theta, theta_full, dims, spread, fig_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    for k, d in enumerate(dims):
        # spread vs ability, online vs batch
        fig, ax = plt.subplots(figsize=(6.2, 4.4))
        x = theta_full[:, k]
        for e, color in (("online", "#d62728"), ("batch", "#1f77b4"), ("mwle", "#2ca02c")):
            sd = np.nanstd(theta[e][:, :, k], axis=0)
            ax.scatter(x, sd, s=20, alpha=0.7, color=color,
                       label=f"{e} (mean SD {spread[e][d]['mean_sd']:.3f})")
        ax.set_xlabel(f"full-bank EAP ability ({d})")
        ax.set_ylabel("across-seed SD of CAT estimate")
        ax.set_title(f"Scenario-level seed dependence by estimator ({d})")
        ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(fig_dir / f"seed_spread_{d}.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
