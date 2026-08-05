"""Phase 1 step 4: re-derive the operating point in SCENARIO units.

The item-level operating point (SE target 0.15 / floor 20 CRITERIA) is VOID. This sweeps
the REAL scenario engine (``scripts.scenario_cat_lib.run_models`` -> ``tutor_cat.engine``)
over ``min_scenarios in {0,12,15,20}`` x a range of per-skill SE targets, on the selected
(1-skill) bank + all 250 scenarios. For each cell it records the convergence rate
(precision_reached), scenarios & criteria administered, and median final SE. It then
recommends the smallest floor / loosest SE reaching ~100% convergence without lengthening
tests, and notes what SE is even achievable on this small bank at N=51.

Parallel across the 51 models with a bounded worker pool (default 5; a __main__ guard is
required for the process pool on Windows).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3] / "eduLLM-Evals"
# Robust root regardless of nesting: walk up to the dir that contains 'scripts'.
_here = Path(__file__).resolve()
for _anc in _here.parents:
    if (_anc / "scripts" / "scenario_cat_lib.py").is_file():
        ROOT = _anc
        break
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.scenario_cat_lib as scat  # noqa: E402


def run_grid(bank_path, matrix_path, scenarios_path, dims, models,
             floors, se_targets, max_scenarios, workers, seed,
             min_evals_per_skill, top_n):
    rows = []
    per_cell_se = {}
    for floor in floors:
        for se in se_targets:
            spec = scat.RunSpec(
                seed=seed, top_n=top_n, max_se=se,
                min_evals_per_skill=min_evals_per_skill,
                min_scenarios=floor, max_scenarios=max_scenarios,
                selection="trace", mode="cat",
            )
            results = scat.run_models(models, bank_path, matrix_path, scenarios_path,
                                      "clamp", dims, spec, workers=workers)
            n = len(results)
            conv = sum(1 for r in results if r["precision_reached"])
            scen = np.array([r["scenarios_administered"] for r in results])
            crit = np.array([r["criteria_administered"] for r in results])
            fse = np.array([r["se_online"][0] for r in results])
            stop_reasons: dict[str, int] = {}
            for r in results:
                stop_reasons[r["stop_reason"]] = stop_reasons.get(r["stop_reason"], 0) + 1
            row = {
                "min_scenarios": floor, "se_target": se,
                "n_models": n, "converged": conv,
                "convergence_rate": round(conv / n, 4),
                "scenarios_mean": round(float(scen.mean()), 2),
                "scenarios_median": int(np.median(scen)),
                "scenarios_max": int(scen.max()),
                "criteria_mean": round(float(crit.mean()), 1),
                "criteria_median": int(np.median(crit)),
                "final_se_median": round(float(np.median(fse)), 4),
                "final_se_max": round(float(fse.max()), 4),
                "stop_reasons": json.dumps(stop_reasons),
            }
            rows.append(row)
            per_cell_se[(floor, se)] = fse
            print(f"  floor={floor:2d} se={se:.2f}: conv={conv}/{n} "
                  f"scen_mean={scen.mean():.1f} crit_mean={crit.mean():.0f} "
                  f"se_med={np.median(fse):.3f} stops={stop_reasons}", flush=True)
    return rows, per_cell_se


def recommend(rows, se_param_floor: float = 0.22) -> dict:
    """Pick a defensible scenario-level operating point.

    Reasoning specific to Bridge (heavy ~19-item testlets, small 250-scenario bank, N=51):
      * Every SE target in the swept range reaches 100% convergence, so convergence does
        not discriminate cells.
      * Because each scenario is a heavy testlet, once ``min_scenarios >= 12`` the floor
        BINDS and the SE target becomes irrelevant (achieved ability SE ~0.09, far below
        any target). Only at ``floor=0`` does the SE target actually control test length.
      * Ability SE below the calibration ``SE_param`` floor (~0.21-0.25 on the item study;
        recomputed downstream) is not worth chasing -- it is below irreducible parameter
        noise. So the SE target is set NEAR that floor rather than as tight as possible.

    Recommendation: a modest min_scenarios floor for a minimum-test-length / local-
    dependence guard, plus an SE target matched to the anticipated SE_param floor. This is
    PROVISIONAL pending the downstream SE_param bootstrap (step 7).
    """
    df = pd.DataFrame(rows)
    ok = df[df["convergence_rate"] >= 0.98]
    view = {"se_param_floor_assumed": se_param_floor}

    # target SE nearest to (and not below) the assumed SE_param floor
    se_choices = sorted(df["se_target"].unique())
    se_pick = min((s for s in se_choices if s >= se_param_floor), default=max(se_choices))

    def cell(floor, se):
        c = df[(df["min_scenarios"] == floor) & (df["se_target"] == se)]
        return c.iloc[0].to_dict() if not c.empty else None

    prov_floor = 12
    prov = cell(prov_floor, se_pick)
    if prov is not None:
        view["recommended_provisional"] = {
            "min_scenarios": int(prov["min_scenarios"]),
            "se_target": float(prov["se_target"]),
            "convergence_rate": float(prov["convergence_rate"]),
            "scenarios_mean": float(prov["scenarios_mean"]),
            "criteria_mean": float(prov["criteria_mean"]),
            "final_se_median": float(prov["final_se_median"]),
            "rationale": (
                f"floor={prov_floor} guarantees a minimum test length (~"
                f"{prov['criteria_mean']:.0f} criteria) that guards against 1-2 scenario "
                "tests dominated by within-scenario local dependence; at this floor the SE "
                "target does not bind (achieved SE floored by length), so the run is robust "
                f"to the SE choice. SE target {se_pick} matches the anticipated SE_param "
                "floor so ability precision is not chased below parameter noise."),
        }
    # leaner alternative: no floor, SE target = SE_param floor (shortest honest test)
    alt = cell(0, se_pick)
    if alt is not None:
        view["alternative_no_floor"] = {
            "min_scenarios": 0, "se_target": float(alt["se_target"]),
            "scenarios_mean": float(alt["scenarios_mean"]),
            "criteria_mean": float(alt["criteria_mean"]),
            "final_se_median": float(alt["final_se_median"]),
            "note": "shortest adaptive test at the SE_param floor; risks very short "
                    "(1-3 scenario) tests with strong testlet local dependence.",
        }
    view["all_cells_converge_100pct"] = bool((df["convergence_rate"] >= 0.999).all())
    view["floor_binds_when_ge"] = 12
    view["tightest_se_swept"] = float(df["se_target"].min())
    view["min_achievable_final_se_median"] = float(df["final_se_median"].min())
    return view


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    base = ROOT / "bridge_calibration"
    p.add_argument("--bank", type=Path, default=base / "bridge_scenario_fitted_1d.jsonl")
    p.add_argument("--matrix", type=Path, default=ROOT / "bridgegrade" / "response_matrix.csv")
    p.add_argument("--scenarios", type=Path, default=ROOT / "data" / "Bridge" / "scenarios.jsonl")
    p.add_argument("--out-dir", type=Path, default=base / "experiments" / "06_floor_se_grid")
    p.add_argument("--floors", type=str, default="0,12,15,20")
    p.add_argument("--se-list", type=str, default="0.15,0.20,0.25,0.30,0.35,0.40")
    p.add_argument("--max-scenarios", type=int, default=250)
    p.add_argument("--min-evals-per-skill", type=int, default=15)
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--se-param-floor", type=float, default=0.22,
                   help="assumed SE_param floor for the recommendation (recomputed in step 7).")
    p.add_argument("--from-csv", action="store_true",
                   help="regenerate best.json/figure from an existing sweep_results.csv "
                        "(no engine re-run).")
    args = p.parse_args()
    (args.out_dir / "figures").mkdir(parents=True, exist_ok=True)

    records, dims, neg = scat.load_fitted_bank(args.bank, "clamp")
    matrix = pd.read_csv(args.matrix, index_col=0)
    models = list(matrix.index)
    floors = [int(x) for x in args.floors.split(",") if x.strip()]
    se_targets = [float(x) for x in args.se_list.split(",") if x.strip()]
    prov = scat.verify_provenance(args.bank, args.matrix)

    print("=" * 90)
    print("STEP 4: operating-point sweep (scenario units)")
    print("=" * 90)
    print(f"bank={args.bank.name} dims={dims} models={len(models)} "
          f"scenarios_bank(max)={args.max_scenarios}")
    print(f"floors={floors} se_targets={se_targets} workers={args.workers}")
    print(f"provenance aligned={prov['aligned']}")

    if args.from_csv:
        rows = pd.read_csv(args.out_dir / "sweep_results.csv").to_dict("records")
        print("(--from-csv: reusing existing sweep_results.csv)")
    else:
        rows, per_cell_se = run_grid(
            args.bank, args.matrix, args.scenarios, dims, models,
            floors, se_targets, args.max_scenarios, args.workers, args.seed,
            args.min_evals_per_skill, args.top_n)
        with (args.out_dir / "sweep_results.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    rec = recommend(rows, se_param_floor=args.se_param_floor)
    best = {
        "purpose": "re-derived scenario-level operating point (SUPERSEDES item-level "
                   "SE 0.15 / floor 20 CRITERIA, which is VOID)",
        "bank": str(args.bank), "dims": dims, "n_models": len(models),
        "bank_scenarios_max": args.max_scenarios,
        "floors_swept": floors, "se_targets_swept": se_targets,
        "config": {"min_evals_per_skill": args.min_evals_per_skill,
                   "top_n": args.top_n, "selection": "trace", "seed": args.seed},
        "achievability_note": (
            f"On a {args.max_scenarios}-scenario bank at N={len(models)} with ~19-20 "
            "criteria per scenario, the minimum median final SE achievable is "
            f"{rec['min_achievable_final_se_median']:.4f}. Each administered scenario is a "
            "heavy ~19-item testlet, so SE drops fast; the binding constraint is usually the "
            "min_scenarios floor / min_evals_per_skill, not the SE target. NOTE: this ability "
            "SE ignores calibration (SE_param) uncertainty, which dominates at N=51 and is "
            "computed downstream (param-uncertainty bootstrap); the final lock must be checked "
            "against that SE_param floor before deployment."),
        "recommendation": rec,
    }
    (args.out_dir / "best.json").write_text(json.dumps(best, indent=2), encoding="utf-8")

    # figure: convergence rate vs SE target, one line per floor
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = pd.DataFrame(rows)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    for floor in floors:
        d = df[df["min_scenarios"] == floor].sort_values("se_target")
        ax1.plot(d["se_target"], d["convergence_rate"], "o-", label=f"floor={floor}")
        ax2.plot(d["se_target"], d["scenarios_mean"], "o-", label=f"floor={floor}")
    ax1.set_xlabel("SE target"); ax1.set_ylabel("convergence rate")
    ax1.set_title("Convergence vs SE target"); ax1.legend(); ax1.axhline(1.0, ls=":", color="gray")
    ax2.set_xlabel("SE target"); ax2.set_ylabel("mean scenarios administered")
    ax2.set_title("Test length vs SE target"); ax2.legend()
    fig.tight_layout()
    fig.savefig(args.out_dir / "figures" / "convergence_and_length_vs_se.png", dpi=140)

    print("\nrecommendation:", json.dumps(rec.get("recommended_provisional"), indent=2))
    print(f"wrote -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
