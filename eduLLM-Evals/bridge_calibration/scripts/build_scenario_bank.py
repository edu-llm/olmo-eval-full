"""Phase 1 steps 1-2: build the scenario-level bank + the core M2PL fit.

Step 1 (bank build + sanity gate): group Bridge criteria by scenario (testlet
structure), record counts, the criteria-per-scenario distribution, unique sources, the
criteria surviving zero-variance filtering, and the leakage-safe source->fold grouping.
Writes ``build_manifest.json``. SANITY-GATE: aborts before fitting if the shapes are
obviously wrong (not ~250 scenarios, criteria don't map, etc).

Step 2 (core fit): fit the confirmatory M2PL at the native 5-skill Q AND the collapsed
1-skill Q, at the starting ridge (item-level default 1e-2). Emits fitted banks in the
scenario_cat_lib schema (``bridge_scenario_fitted_5d.jsonl`` /
``bridge_scenario_fitted_1d.jsonl``), a per-criterion item-parameter CSV, and a fit
manifest (loglik/AIC/BIC + latent correlation for the 5-skill fit).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bridge_scenario_lib as L  # noqa: E402


def sanity_gate(maps: dict, drop_info: dict, fold_info: dict) -> list[str]:
    """Return a list of fatal problems (empty => pass)."""
    problems: list[str] = []
    n_scen = len(set(maps["c2s"].values()))
    if not (240 <= n_scen <= 260):
        problems.append(f"expected ~250 scenarios, got {n_scen}")
    n_src = len(set(maps["s2src"].values()))
    if not (150 <= n_src <= 175):
        problems.append(f"expected ~162 unique sources, got {n_src}")
    if drop_info["n_criteria_fit"] < 3000:
        problems.append(f"only {drop_info['n_criteria_fit']} criteria survive "
                        "zero-variance filtering (expected ~4200)")
    # every fitted criterion must map to a known scenario
    unmapped = [c for c in maps["c2s"].values() if c is None]
    if unmapped:
        problems.append(f"{len(unmapped)} criteria have no scenario_id")
    if fold_info["n_scenarios"] != n_scen:
        problems.append("fold grouping did not cover all scenarios")
    return problems


def criteria_per_scenario_dist(maps: dict, fitted_ids: set[str]) -> dict:
    """Distribution of #criteria per scenario, both raw and after zero-var filtering."""
    from collections import Counter
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
        "after_zero_variance": {"min": int(kept_vals.min()), "max": int(kept_vals.max()),
                                "mean": round(float(kept_vals.mean()), 3),
                                "median": int(np.median(kept_vals)),
                                "n_scenarios_with_fitted_criteria": int(len(kept_vals))},
        "scenarios_fully_dropped": scen_all_dropped,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matrix", type=Path, default=L.DEFAULT_MATRIX)
    p.add_argument("--rubrics", type=Path, default=L.DEFAULT_RUBRICS)
    p.add_argument("--scenarios", type=Path, default=L.DEFAULT_SCENARIOS)
    p.add_argument("--out-dir", type=Path,
                   default=L.ROOT / "bridge_calibration")
    p.add_argument("--ridge", type=float, default=1e-2)
    p.add_argument("--grid", type=int, default=5,
                   help="GH nodes/dim for the 5-skill fit (5 -> 5^5 nodes).")
    p.add_argument("--grid-1d", type=int, default=7, help="GH nodes for the 1-skill fit.")
    p.add_argument("--k-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260729)
    args = p.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("STEP 1: scenario-level bank build + sanity gate")
    print("=" * 90)
    maps = L.load_maps(args.rubrics, args.scenarios)
    Y, M, items, models, drop_info = L.prepare_matrix(args.matrix)
    fitted_ids = set(items)
    fold_info = L.source_fold_groups(maps, k=args.k_folds, seed=args.seed)

    print(f"scenarios (from rubrics)   : {len(set(maps['c2s'].values()))}")
    print(f"unique sources             : {len(set(maps['s2src'].values()))}")
    print(f"models (persons)           : {len(models)}")
    print(f"criteria raw               : {drop_info['n_criteria_raw']}")
    print(f"criteria after zero-var     : {drop_info['n_criteria_fit']} "
          f"(dropped {drop_info['dropped_all_fail']} all-fail + "
          f"{drop_info['dropped_all_pass']} all-pass)")
    print(f"fill rate                  : {drop_info['fill_rate']:.4f}")
    print(f"source->fold sizes (scen.) : {fold_info['fold_sizes_scenarios']}")

    problems = sanity_gate(maps, drop_info, fold_info)
    dist = criteria_per_scenario_dist(maps, fitted_ids)

    build_manifest = {
        "generated_at": L._utcnow() if hasattr(L, "_utcnow") else L.cp._utcnow(),
        "study": "bridge scenario-level (Phase 1 foundation)",
        "design": {
            "bank": "all scenarios (NO source dedup)",
            "source_id_use": "leakage-safe fold grouping only",
            "administration": "testlet bundle (per-criterion IRT items grouped by scenario)",
            "skill_axis": L.BRIDGE_SKILLS,
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
    print("STEP 2: core scenario-level M2PL fit (ridge=%g)" % args.ridge)
    print("=" * 90)

    n_obs = int(M.sum())

    # 1-skill fit (collapsed) -----------------------------------------------
    Q1, labels1 = L.build_Q(items, maps["qmap"], L.STRUCTURES["overall_1d"])
    print(f"fitting 1-skill (overall) M2PL: grid={args.grid_1d} ...", flush=True)
    fit1, keep1 = L.fit_structure(Y, M, Q1, args.grid_1d, args.ridge,
                                  estimate_corr=False)
    aic1, bic1 = L.cm.aic_bic(fit1["loglik"], fit1["n_params"], n_obs)
    n1 = L.write_fitted_bank(out / "bridge_scenario_fitted_1d.jsonl", items, keep1,
                             fit1["A"], fit1["b"], Q1, labels1, maps, args.matrix,
                             "overall_1d", args.ridge, args.grid_1d)
    print(f"  1d: loglik={fit1['loglik']:.1f} AIC={aic1:.1f} BIC={bic1:.1f} "
          f"items={n1} converged={fit1['converged']}")

    # 5-skill fit (native, confirmatory) ------------------------------------
    Q5, labels5 = L.build_Q(items, maps["qmap"], L.STRUCTURES["full_5d"])
    print(f"fitting 5-skill (native) M2PL: grid={args.grid} ({args.grid**5} nodes) ...",
          flush=True)
    fit5, keep5 = L.fit_structure(Y, M, Q5, args.grid, args.ridge,
                                  estimate_corr=True)
    aic5, bic5 = L.cm.aic_bic(fit5["loglik"], fit5["n_params"], n_obs)
    n5 = L.write_fitted_bank(out / "bridge_scenario_fitted_5d.jsonl", items, keep5,
                             fit5["A"], fit5["b"], Q5, labels5, maps, args.matrix,
                             "full_5d", args.ridge, args.grid)
    R = fit5.get("R")
    max_corr = (float(np.max(np.abs(R - np.eye(5)))) if R is not None else None)
    print(f"  5d: loglik={fit5['loglik']:.1f} AIC={aic5:.1f} BIC={bic5:.1f} "
          f"items={n5} converged={fit5['converged']} max|latent corr|={max_corr}")

    # per-axis discrimination from the 5D fit
    per_axis = {}
    A5 = fit5["A"]
    Q5k = Q5[keep5]
    for di, sk in enumerate(labels5):
        load = Q5k[:, di] == 1
        av = A5[load, di]
        per_axis[sk] = {
            "n_anchor_items": int(load.sum()),
            "median_disc": round(float(np.median(av)), 4) if av.size else None,
            "mean_disc": round(float(np.mean(av)), 4) if av.size else None,
            "frac_disc_le_0": round(float(np.mean(av <= 0)), 4) if av.size else None,
        }

    # item-parameter CSV (5-skill loadings + b) -----------------------------
    csv_path = out / "item_params_5d.csv"
    with csv_path.open("w", encoding="utf-8") as fh:
        fh.write("criterion_id,scenario_id," + ",".join(f"a_{s}" for s in labels5)
                 + ",b,n_persons\n")
        kept_ids = [items[j] for j in keep5]
        nper = M[:, keep5].sum(axis=0).astype(int)
        for row, cid in enumerate(kept_ids):
            avals = ",".join(f"{A5[row, k]:.6f}" for k in range(len(labels5)))
            fh.write(f"{cid},{maps['c2s'].get(cid, cid)},{avals},"
                     f"{fit5['b'][row]:.6f},{int(nper[row])}\n")

    fit_manifest = {
        "generated_at": L.cp._utcnow(),
        "note": "core scenario-level M2PL fit (testlet; per-criterion items grouped by "
                "scenario). Ridge is the item-level default 1e-2; RE-SWEPT later.",
        "ridge": args.ridge,
        "grid_1d": args.grid_1d, "grid_5d": args.grid,
        "n_models": len(models), "n_observed_cells": n_obs,
        "fits": {
            "overall_1d": {"n_dims": 1, "loglik": fit1["loglik"],
                           "n_params": fit1["n_params"], "aic": aic1, "bic": bic1,
                           "n_items": n1, "converged": fit1["converged"],
                           "n_iter": fit1["n_iter"]},
            "full_5d": {"n_dims": 5, "loglik": fit5["loglik"],
                        "n_params": fit5["n_params"], "aic": aic5, "bic": bic5,
                        "n_items": n5, "converged": fit5["converged"],
                        "n_iter": fit5["n_iter"],
                        "max_latent_corr_offdiag": max_corr,
                        "latent_correlation": (np.round(R, 4).tolist()
                                               if R is not None else None),
                        "per_axis_discrimination": per_axis},
        },
        "fitted_banks": {"overall_1d": "bridge_scenario_fitted_1d.jsonl",
                         "full_5d": "bridge_scenario_fitted_5d.jsonl"},
    }
    (out / "fit_manifest.json").write_text(json.dumps(fit_manifest, indent=2),
                                           encoding="utf-8")
    print(f"\nwrote -> {csv_path}")
    print(f"wrote -> {out / 'fit_manifest.json'}")
    print(f"wrote fitted banks -> bridge_scenario_fitted_1d.jsonl ({n1} items), "
          f"bridge_scenario_fitted_5d.jsonl ({n5} items)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
