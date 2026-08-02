"""SCENARIO-LEVEL A/B of the CAT selection rule: trace (PRD) vs D-optimality.

This is the scenario-level replacement for the item-level ``cat_selection_experiment.py``.
It drives the REAL production engine (``tutor_cat.engine`` + ``selector.select_next``),
which selects whole scenarios and grades all their criteria. The only thing that varies
between the two arms is ``RunConfig.selection`` ("trace" vs "dopt"); the M2PL update,
stopping rule, and the full-bank reference ability are identical.

For each selection rule it reports, per bank (2- and 3-skill):
  * test length: scenarios administered and criteria administered,
  * recovery of the full-bank EAP reference by each final estimator (online/batch/MWLE):
    slope, Pearson r, worst-12 gap,
  * per-skill convergence: median final posterior SE and fraction of models reaching the
    precision target (this is where the item-level story about "presentation" gets
    corrected, since at the scenario level presentation rides along with ~every scenario).

Parallel across models with ``--workers``.

Usage
-----
    python scripts/scenario_selection_experiment.py \
        --bank data/TutorBench/rubrics_qmatrix_calibrated_2skill_fitted.jsonl \
        --matrix staging/response_matrix_full_nonopt.csv \
        --out-dir regenerated_figures/scenario_level/selection/2_skills --workers 16
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

SELECTIONS = ("trace", "dopt")
ESTIMATORS = {"theta_online": "production online", "theta_batch": "batch EAP",
              "theta_mwle": "batch EAP + MWLE"}


def recovery_stats(x: np.ndarray, y: np.ndarray) -> dict:
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    w = np.argsort(x)[:12]
    return {"slope": float(np.polyfit(x, y, 1)[0]), "r": float(np.corrcoef(x, y)[0, 1]),
            "gap_worst12": float(np.mean(y[w] - x[w])), "n": int(x.size)}


def run_arm(models, bank, matrix, matrix_path, scen_path, neg_policy, dims,
            A, b, grid, log_prior, col, Y, selection, workers, args):
    spec = scat.RunSpec(seed=args.seed, top_n=args.top_n, max_se=args.max_se,
                        min_evals_per_skill=args.min_evals_per_skill,
                        max_scenarios=args.max_scenarios, selection=selection,
                        runs_dir=str(args.runs_dir))
    results = scat.run_models(models, bank, matrix_path, scen_path, neg_policy,
                              dims, spec, workers=workers)
    rows = []
    for rec0 in results:
        m = rec0["model"]
        r = list(matrix.index).index(m)
        idx = np.array([col[c] for c in rec0["order"] if c in col], dtype=int)
        th_on = np.asarray(rec0["theta_online"], float)
        th_ba = scat.eap_subset(Y[r], idx, A, b, grid, log_prior) if idx.size else th_on.copy()
        th_mw, ok = (scat.mwle_subset(Y[r], idx, A, b, th_ba) if idx.size else (th_ba.copy(), True))
        row = {"model": m, "scenarios": rec0["scenarios_administered"],
               "criteria": rec0["criteria_administered"],
               "precision_reached": rec0["precision_reached"], "mwle_ok": ok}
        for k, d in enumerate(dims):
            row[f"theta_online_{d}"] = float(th_on[k])
            row[f"theta_batch_{d}"] = float(th_ba[k])
            row[f"theta_mwle_{d}"] = float(th_mw[k])
            row[f"final_se_{d}"] = float(rec0["se_online"][k])
            row[f"scorable_evals_{d}"] = int(rec0["scorable_evals"][k])
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--matrix", type=Path, required=True)
    p.add_argument("--scenarios", type=Path,
                   default=ROOT / "data" / "TutorBench" / "scenarios.jsonl")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--runs-dir", type=Path, default=ROOT / "staging" / "engine_runs_sel")
    p.add_argument("--negative-policy", choices=("clamp", "keep", "drop"), default="clamp")
    p.add_argument("--seed", type=int, default=42)
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
    print(f"bank={args.bank.name} dims={dims} models={len(models)} "
          f"aligned={prov['aligned']} workers={args.workers}")

    grid, log_prior = scat.build_grid(len(dims), args.grid, args.range)
    theta_full = scat.eap_all_models(Y, mask, A, b, grid, log_prior)
    row_of = {m: i for i, m in enumerate(matrix.index)}

    arms = {}
    for sel in SELECTIONS:
        print(f"\n=== selection = {sel} ===", flush=True)
        df = run_arm(models, args.bank, matrix, args.matrix, args.scenarios,
                     args.negative_policy, dims, A, b, grid, log_prior, col, Y, sel,
                     args.workers, args)
        rix = [row_of[m] for m in df.model]
        for k, d in enumerate(dims):
            df[f"theta_full_{d}"] = theta_full[rix, k]
        arms[sel] = df

    agg = {}
    for sel, df in arms.items():
        rec = {"length": {"scenarios_mean": float(df.scenarios.mean()),
                          "scenarios_median": float(df.scenarios.median()),
                          "criteria_mean": float(df.criteria.mean()),
                          "criteria_median": float(df.criteria.median())},
               "precision_reached": int(df.precision_reached.sum()),
               "recovery": {}, "per_skill": {}}
        for est in ESTIMATORS:
            rec["recovery"][est] = {}
            for d in dims:
                x = df[f"theta_full_{d}"].to_numpy(float)
                y = df[f"{est}_{d}"].to_numpy(float)
                rec["recovery"][est][d] = recovery_stats(x, y)
        for d in dims:
            rec["per_skill"][d] = {
                "median_final_se": float(df[f"final_se_{d}"].median()),
                "frac_se_over_target": float((df[f"final_se_{d}"] > args.max_se).mean()),
                "median_scorable_evals": float(df[f"scorable_evals_{d}"].median())}
        agg[sel] = rec

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for sel, df in arms.items():
        df.to_csv(args.out_dir / f"per_model_{sel}.csv", index=False)

    metrics = {"generated_at": datetime.now(timezone.utc).isoformat(),
               "purpose": "scenario-level trace vs D-optimality selection A/B (real engine)",
               "bank": str(args.bank), "matrix": str(args.matrix), "dims": dims,
               "provenance": prov,
               "config": {"seed": args.seed, "top_n": args.top_n, "max_se": args.max_se,
                          "min_evals_per_skill": args.min_evals_per_skill,
                          "max_scenarios": args.max_scenarios, "grid": args.grid,
                          "negative_policy": args.negative_policy},
               "n_models": len(models), "results": agg}
    with (args.out_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    _figures(arms, dims, agg, args.out_dir / "figures", args.max_se)

    print("\n" + "=" * 96)
    print("SCENARIO-LEVEL SELECTION A/B  (reference = full-bank EAP)")
    print("=" * 96)
    for sel in SELECTIONS:
        L = agg[sel]["length"]
        print(f"\n[{sel}] scenarios mean={L['scenarios_mean']:.1f} "
              f"criteria mean={L['criteria_mean']:.1f}  precision={agg[sel]['precision_reached']}/{len(models)}")
        for est in ESTIMATORS:
            line = "  ".join(f"{d[:4]} r={agg[sel]['recovery'][est][d]['r']:.3f} "
                             f"m={agg[sel]['recovery'][est][d]['slope']:.3f}" for d in dims)
            print(f"    {est:12s}: {line}")
        print("    per-skill final SE: " + "  ".join(
            f"{d[:4]}={agg[sel]['per_skill'][d]['median_final_se']:.3f}"
            f"(>{args.max_se}:{agg[sel]['per_skill'][d]['frac_se_over_target']:.0%})" for d in dims))
    print(f"\nwrote -> {args.out_dir}")
    return 0


def _figures(arms, dims, agg, fig_dir: Path, max_se: float):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    # length comparison
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist([arms["trace"].criteria, arms["dopt"].criteria], bins=18,
            label=[f"trace (mean {arms['trace'].criteria.mean():.0f})",
                   f"D-opt (mean {arms['dopt'].criteria.mean():.0f})"], alpha=0.85)
    ax.set_xlabel("criteria administered"); ax.set_ylabel("models")
    ax.set_title("Scenario-level CAT length: trace vs D-optimality")
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(fig_dir / "length_trace_vs_dopt.png", dpi=130); plt.close(fig)

    # recovery overlay per dim (MWLE)
    for d in dims:
        fig, ax = plt.subplots(figsize=(5, 4.8))
        for sel, color in (("trace", "#1f77b4"), ("dopt", "#d62728")):
            df = arms[sel]
            x = df[f"theta_full_{d}"].to_numpy(float)
            y = df[f"theta_mwle_{d}"].to_numpy(float)
            s = agg[sel]["recovery"]["theta_mwle"][d]
            ax.scatter(x, y, s=22, alpha=0.65, color=color, edgecolor="k", linewidth=0.2,
                       label=f"{sel}: r={s['r']:.3f}, slope={s['slope']:.3f}")
        allv = np.concatenate([arms[s][f"theta_full_{d}"].to_numpy(float) for s in arms] +
                              [arms[s][f"theta_mwle_{d}"].to_numpy(float) for s in arms])
        lo, hi = np.nanmin(allv) - 0.3, np.nanmax(allv) + 0.3
        ax.plot([lo, hi], [lo, hi], ls="--", color="gray", lw=1, label="y = x")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel(f"full-bank EAP ({d})"); ax.set_ylabel(f"CAT MWLE ({d})")
        ax.set_title(f"Scenario-level recovery: trace vs D-opt ({d})")
        ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout(); fig.savefig(fig_dir / f"recovery_trace_vs_dopt_{d}.png", dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
