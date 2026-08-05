"""Phase 2b exp 11: order / seed stability at the locked operating point.

Runs the REAL engine at the locked point (min_scenarios=12, SE 0.15) with the production
max-information start (top_n=5 seeded pick at theta=0 -- NOT a uniform-random first item)
across ``n_seeds`` master seeds. Each seed yields a different scenario order; we report the
per-model MWLE-theta SD across seeds and compare it to the SE target. Small SD => the
adaptive order does not materially move the ability estimate.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import bridge_scenario_lib as L  # noqa: E402

ROOT = L.ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import scripts.scenario_cat_lib as scat  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    base = ROOT / "bridge_calibration"
    p.add_argument("--bank", type=Path, default=base / "bridge_scenario_fitted_1d_catpool.jsonl")
    p.add_argument("--matrix", type=Path, default=ROOT / "bridgegrade" / "response_matrix.csv")
    p.add_argument("--scenarios", type=Path, default=ROOT / "data" / "Bridge" / "scenarios.jsonl")
    p.add_argument("--out-dir", type=Path, default=base / "experiments" / "11_order_seed")
    p.add_argument("--tmp-dir", type=Path, default=base / "experiments" / "11_order_seed" / "_tmp")
    p.add_argument("--n-seeds", type=int, default=8)
    p.add_argument("--min-scenarios", type=int, default=12)
    p.add_argument("--max-se", type=float, default=0.15)
    p.add_argument("--max-scenarios", type=int, default=228)
    p.add_argument("--eap-grid", type=int, default=321)
    p.add_argument("--range", type=float, default=8.0)
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.tmp_dir.mkdir(parents=True, exist_ok=True)

    records, dims, _ = scat.load_fitted_bank(args.bank, "clamp")
    d = dims[0]
    ids, A, b = scat.assemble_arrays(records, dims)
    matrix = pd.read_csv(args.matrix, index_col=0)
    Yraw = matrix.reindex(columns=ids).to_numpy(float)
    Y = np.nan_to_num(Yraw, nan=0.0)
    models = list(matrix.index)
    row_of = {m: i for i, m in enumerate(models)}
    col = {c: i for i, c in enumerate(ids)}
    egrid, elog = scat.build_grid(1, args.eap_grid, args.range)

    print("=" * 84)
    print(f"EXP 11: order/seed stability at locked point ({args.min_scenarios}/{args.max_se}), "
          f"{args.n_seeds} seeds")
    print("=" * 84)

    theta_by_seed = np.full((len(models), args.n_seeds), np.nan)
    scen_by_seed = np.full((len(models), args.n_seeds), np.nan)
    for s in range(args.n_seeds):
        seed = 1000 + s
        spec = scat.RunSpec(seed=seed, top_n=5, max_se=args.max_se,
                            min_evals_per_skill=15, min_scenarios=args.min_scenarios,
                            max_scenarios=args.max_scenarios, selection="trace", mode="cat",
                            runs_dir=str(args.tmp_dir / f"seed{seed}"))
        res = scat.run_models(models, args.bank, args.matrix, args.scenarios,
                              "clamp", dims, spec, workers=args.workers)
        for r0 in res:
            r = row_of[r0["model"]]
            idx = np.array([col[c] for c in r0["order"] if c in col], dtype=int)
            if idx.size:
                th0 = scat.eap_subset(Y[r], idx, A, b, egrid, elog)
                thm, _ = scat.mwle_subset(Y[r], idx, A, b, th0)
                theta_by_seed[r, s] = thm[0]
            scen_by_seed[r, s] = r0["scenarios_administered"]
        print(f"  seed {seed} done", flush=True)

    theta_sd = np.nanstd(theta_by_seed, axis=1, ddof=1)
    theta_range = np.nanmax(theta_by_seed, axis=1) - np.nanmin(theta_by_seed, axis=1)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "across-seed order stability of ability at the locked operating point",
        "operating_point": {"min_scenarios": args.min_scenarios, "max_se": args.max_se,
                            "start": "production max-info top5 seeded pick (not uniform-random)"},
        "n_models": len(models), "n_seeds": args.n_seeds,
        "theta_sd_across_seeds": {"mean": float(np.mean(theta_sd)),
                                  "median": float(np.median(theta_sd)),
                                  "max": float(np.max(theta_sd))},
        "theta_range_across_seeds": {"mean": float(np.mean(theta_range)),
                                     "max": float(np.max(theta_range))},
        "se_target": args.max_se,
        "mean_sd_over_se_target": float(np.mean(theta_sd) / args.max_se),
        "mean_scenarios_administered": float(np.nanmean(scen_by_seed)),
        "interpretation": ("mean across-seed theta SD as a fraction of the SE target; <<1 means "
                           "the adaptive order/seed contributes negligible variance relative to "
                           "measurement error."),
    }
    (args.out_dir / "order_seed_stability.json").write_text(json.dumps(summary, indent=2),
                                                            encoding="utf-8")
    pd.DataFrame({"model": models, "theta_sd": theta_sd, "theta_range": theta_range}).to_csv(
        args.out_dir / "per_model_spread.csv", index=False)
    shutil.rmtree(args.tmp_dir, ignore_errors=True)

    print(f"\ntheta SD across seeds: mean={np.mean(theta_sd):.4f} median={np.median(theta_sd):.4f} "
          f"max={np.max(theta_sd):.4f}  (SE target {args.max_se}; ratio {np.mean(theta_sd)/args.max_se:.3f})")
    print(f"wrote -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
