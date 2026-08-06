"""Phase 1 steps 1-2: build the WildBench scenario-level bank + the core M2PL fit.

Step 1 (bank build + sanity gate): group WildBench criteria by scenario (testlet
structure), record counts, the criteria-per-scenario distribution, unique sources, the
criteria surviving zero-variance filtering, and the (leakage-safe) source->fold grouping.
Writes ``build_manifest.json``. SANITY-GATE: aborts before fitting if the shapes are
obviously wrong (not ~1001 scenarios, criteria don't map, etc).

Step 2 (core fit): fit the confirmatory 1-skill (overall) M2PL at the starting ridge
(1e-2) and emit the fitted bank in the scenario_cat_lib schema
(``wildbench_scenario_fitted_1d.jsonl``) + a fit manifest (loglik/AIC/BIC). The full
1D-vs-11D-vs-collapsed dimensionality comparison lives in ``scenario_dimensionality.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wildbench_scenario_lib as L  # noqa: E402


def sanity_gate(maps: dict, drop_info: dict, fold_info: dict) -> list[str]:
    """Return a list of fatal problems (empty => pass)."""
    problems: list[str] = []
    n_scen = len(set(maps["c2s"].values()))
    if not (960 <= n_scen <= 1040):
        problems.append(f"expected ~1001 scenarios, got {n_scen}")
    n_src = len(set(maps["s2src"].values()))
    if n_src < 900:
        problems.append(f"expected ~1001 unique sources, got {n_src}")
    if not (5000 <= drop_info["n_criteria_fit"] <= 11416):
        problems.append(f"{drop_info['n_criteria_fit']} criteria survive zero-variance "
                        "filtering (expected ~8358)")
    if drop_info["n_models"] < 40:
        problems.append(f"only {drop_info['n_models']} models (expected 52)")
    unmapped = [c for c in maps["c2s"].values() if c is None]
    if unmapped:
        problems.append(f"{len(unmapped)} criteria have no scenario_id")
    if fold_info["n_scenarios"] != n_scen:
        problems.append("fold grouping did not cover all scenarios")
    return problems


def criteria_per_scenario_dist(maps: dict, fitted_ids: set[str]) -> dict:
    raw = Counter()
    kept = Counter()
    for cid, sid in maps["c2s"].items():
        raw[sid] += 1
        if cid in fitted_ids:
            kept[sid] += 1
    raw_vals = np.array(list(raw.values()))
    kept_vals = np.array([v for v in kept.values() if v > 0])
    scen_all_dropped = [sid for sid in raw if kept.get(sid, 0) == 0]
    return {
        "raw": {"min": int(raw_vals.min()), "max": int(raw_vals.max()),
                "mean": round(float(raw_vals.mean()), 3),
                "median": int(np.median(raw_vals))},
        "after_zero_variance": {
            "min": int(kept_vals.min()), "max": int(kept_vals.max()),
            "mean": round(float(kept_vals.mean()), 3),
            "median": int(np.median(kept_vals)),
            "n_scenarios_with_fitted_criteria": int(len(kept_vals))},
        "n_scenarios_fully_dropped_all_zv": len(scen_all_dropped),
        "scenarios_fully_dropped": scen_all_dropped,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matrix", type=Path, default=L.DEFAULT_MATRIX)
    p.add_argument("--rubrics", type=Path, default=L.DEFAULT_RUBRICS)
    p.add_argument("--scenarios", type=Path, default=L.DEFAULT_SCENARIOS)
    p.add_argument("--out-dir", type=Path, default=L.ROOT / "wildbench_calibration")
    p.add_argument("--ridge", type=float, default=1e-2)
    p.add_argument("--grid-1d", type=int, default=21, help="uniform-EAP/GH nodes for 1D fit.")
    p.add_argument("--k-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260729)
    args = p.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("STEP 1: WildBench scenario-level bank build + sanity gate")
    print("=" * 90)
    maps = L.load_maps(args.rubrics, args.scenarios)
    Y, M, items, models, drop_info = L.prepare_matrix(args.matrix)
    fitted_ids = set(items)
    fold_info = L.source_fold_groups(maps, k=args.k_folds, seed=args.seed)

    print(f"scenarios (from scenarios) : {len(maps['s2src'])}")
    print(f"scenarios (from rubrics)   : {len(set(maps['c2s'].values()))}")
    print(f"unique sources             : {len(set(maps['s2src'].values()))} "
          f"(unique-per-scenario={fold_info['unique_source_per_scenario']})")
    print(f"models (persons)           : {len(models)}")
    print(f"criteria raw               : {drop_info['n_criteria_raw']}")
    print(f"criteria after zero-var     : {drop_info['n_criteria_fit']} "
          f"(dropped {drop_info['dropped_all_fail']} all-fail + "
          f"{drop_info['dropped_all_pass']} all-pass)")
    print(f"fill rate                  : {drop_info['fill_rate']:.5f}")
    print(f"source->fold sizes (scen.) : {fold_info['fold_sizes_scenarios']}")

    problems = sanity_gate(maps, drop_info, fold_info)
    dist = criteria_per_scenario_dist(maps, fitted_ids)
    print(f"criteria/scenario raw      : mean {dist['raw']['mean']} "
          f"[{dist['raw']['min']},{dist['raw']['max']}]")
    print(f"administrable scenarios     : {dist['after_zero_variance']['n_scenarios_with_fitted_criteria']} "
          f"({dist['n_scenarios_fully_dropped_all_zv']} all-ZV un-administrable)")

    build_manifest = {
        "generated_at": L.cp._utcnow(),
        "study": "wildbench scenario-level calibration (Phase 1 foundation)",
        "design": {
            "bank": "all scenarios (NO source dedup)",
            "source_id_use": "leakage-safe fold grouping only (unique-per-scenario here)",
            "administration": "testlet bundle (per-criterion IRT items grouped by scenario)",
            "skill_axis": L.WILDBENCH_SKILLS,
        },
        "inputs": {"matrix": str(args.matrix), "rubrics": str(args.rubrics),
                   "scenarios": str(args.scenarios),
                   "matrix_sha256": L.matrix_sha256(args.matrix)},
        "n_scenarios": len(set(maps["c2s"].values())),
        "n_unique_sources": len(set(maps["s2src"].values())),
        "n_models": len(models),
        "n_criteria_raw": drop_info["n_criteria_raw"],
        "n_criteria_after_zero_variance": drop_info["n_criteria_fit"],
        "dropped_all_fail": drop_info["dropped_all_fail"],
        "dropped_all_pass": drop_info["dropped_all_pass"],
        "fill_rate": drop_info["fill_rate"],
        "criteria_per_scenario": dist,
        "source_fold_grouping": {
            "k": fold_info["k"], "seed": fold_info["seed"],
            "n_sources": fold_info["n_sources"],
            "n_scenarios": fold_info["n_scenarios"],
            "unique_source_per_scenario": fold_info["unique_source_per_scenario"],
            "fold_sizes_scenarios": fold_info["fold_sizes_scenarios"],
            "scenario_to_fold": fold_info["scenario_to_fold"],
            "source_to_fold": fold_info["source_to_fold"],
        },
        "sanity_gate": {"passed": not problems, "problems": problems},
    }
    (out / "build_manifest.json").write_text(
        json.dumps(build_manifest, indent=2), encoding="utf-8")
    print(f"\nwrote -> {out / 'build_manifest.json'}")

    if problems:
        print("\n*** SANITY GATE FAILED ***")
        for pr in problems:
            print("  -", pr)
        print("Aborting before the fit. Fix the inputs and re-run.")
        return 3
    print("sanity gate: PASSED")

    print("\n" + "=" * 90)
    print("STEP 2: core scenario-level 1-skill M2PL fit (ridge=%g)" % args.ridge)
    print("=" * 90)
    n_obs = int(M.sum())

    Q1, labels1 = L.build_Q(items, maps["qmap"], L.STRUCTURES["overall_1d"])
    print(f"fitting 1-skill (overall) M2PL: grid={args.grid_1d} ...", flush=True)
    fit1, keep1 = L.fit_structure(Y, M, Q1, args.grid_1d, args.ridge, estimate_corr=False)
    aic1, bic1 = L.cm.aic_bic(fit1["loglik"], fit1["n_params"], n_obs)
    n1 = L.write_fitted_bank(out / "wildbench_scenario_fitted_1d.jsonl", items, keep1,
                             fit1["A"], fit1["b"], Q1, labels1, maps, args.matrix,
                             "overall_1d", args.ridge, args.grid_1d)
    a_vals = fit1["A"][:, 0]
    n_extreme = int(np.sum(~np.isfinite(a_vals)) + np.sum(np.abs(a_vals) > L.cm.EXTREME_A))
    print(f"  1d: loglik={fit1['loglik']:.1f} AIC={aic1:.1f} BIC={bic1:.1f} "
          f"items={n1} converged={fit1['converged']} n_extreme_a={n_extreme}")

    fit_manifest = {
        "generated_at": L.cp._utcnow(),
        "note": "core scenario-level 1-skill M2PL fit (testlet; per-criterion items "
                "grouped by scenario). Ridge 1e-2 starting point; re-swept later. Full "
                "dimensionality comparison in experiments/03_structures.",
        "ridge": args.ridge, "grid_1d": args.grid_1d,
        "n_models": len(models), "n_observed_cells": n_obs,
        "fits": {
            "overall_1d": {"n_dims": 1, "loglik": fit1["loglik"],
                           "n_params": fit1["n_params"], "aic": aic1, "bic": bic1,
                           "n_items": n1, "converged": fit1["converged"],
                           "n_iter": fit1["n_iter"], "n_extreme_a": n_extreme},
        },
        "fitted_banks": {"overall_1d": "wildbench_scenario_fitted_1d.jsonl"},
    }
    (out / "fit_manifest.json").write_text(json.dumps(fit_manifest, indent=2),
                                           encoding="utf-8")
    print(f"\nwrote -> {out / 'fit_manifest.json'}")
    print(f"wrote fitted bank -> wildbench_scenario_fitted_1d.jsonl ({n1} items)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
