"""Phase A augmentation: TAIL-AWARE per-cell SE stats for the EAP-stop floor x SE grid.

The per-cell ``se_total_median`` in ``experiments/13_eap_stop_grid/recovery_grid.csv`` HIDES
the SE-target benefit: on Bridge's heavy ~18-criterion testlets the MEDIAN model is already
below any SE target once the min_scenarios floor is met, so tightening the SE target only helps
the TAIL (the few low-information / extreme-ability models). This script re-summarizes each grid
cell with tail-aware deployed statistics so the marginal benefit of the SE target is visible.

This is a RE-SUMMARIZE, not a new fit. The frozen bank
(``bridge_scenario_fitted_1d_catpool.jsonl``) is reused verbatim (no refit, no bank change).
It runs the DEPLOYED CAT on the full data (all 51 models):

  * EAP stop  -- ONE forced-long adaptive order (production selection, unchanged); every cell's
    per-model administered set is derived POST-HOC by the EAP posterior-SD stop on the fine
    theta grid (321 nodes over +/-8). Per-model deployed SE_ability = the fine-grid EAP
    posterior SD at the stop; SE_param = observed-information parametric bootstrap over the
    administered set (exp-07 machinery); SE_total = sqrt(SE_ability^2 + SE_param^2).
  * online stop (optional, floors 8/12 for side-by-side) -- the production engine run per cell;
    per-model deployed SE recomputed identically (fine-grid EAP posterior SD over the engine's
    administered set + the same SE_param bootstrap).

Per cell it emits, for BOTH SE_total and SE_ability: mean, SD, median, q3, max; the count of
models above 0.15 and above 0.20 SE_total; and mean/max deployed length (scenarios). ``recovery_r``
is carried over (read-only) from the respective of-record/grid CSVs for context (it is the OOS
k-fold measure; NOT recomputed here).

Writes ``experiments/13_eap_stop_grid/recovery_grid_tailstats.csv`` + ``TAILSTATS_NOTE.md``. Does
NOT modify of-record files or the existing grid CSVs.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
from scipy.special import logsumexp

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import bridge_scenario_lib as L  # noqa: E402
import bridge_eap_stop_lib as E  # noqa: E402

ROOT = L.ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import scripts.scenario_cat_lib as scat  # noqa: E402

_pu_spec = importlib.util.spec_from_file_location(
    "bridge_scenario_param_uncertainty", _HERE / "scenario_param_uncertainty.py")
PU = importlib.util.module_from_spec(_pu_spec)
sys.modules["bridge_scenario_param_uncertainty"] = PU
_pu_spec.loader.exec_module(PU)

THR = (0.15, 0.20)


def cellstats(arr, prefix):
    a = np.asarray(arr, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {f"{prefix}_{k}": np.nan for k in ("mean", "sd", "median", "q3", "max")}
    return {
        f"{prefix}_mean": round(float(a.mean()), 4),
        f"{prefix}_sd": round(float(a.std(ddof=1)) if a.size > 1 else 0.0, 4),
        f"{prefix}_median": round(float(np.median(a)), 4),
        f"{prefix}_q3": round(float(np.quantile(a, 0.75)), 4),
        f"{prefix}_max": round(float(a.max()), 4),
    }


def per_model_se(y, idx, cov, egrid, elog, bgrid, blp, n_boot, rng):
    """Deployed (SE_ability, SE_param, SE_total) for one model's administered set ``idx``.

    SE_ability = fine-grid EAP posterior SD at the frozen params (reliable; matches the stop
    rule). SE_param = observed-info parametric bootstrap SD over the same items.
    """
    idx = np.asarray(idx, dtype=int)
    if idx.size == 0:
        return np.nan, np.nan, np.nan
    _, var = scat.eap_subset_mean_var(y, idx, cov["A"], cov["b"], egrid, elog)
    se_ability = float(np.sqrt(max(float(var[0]), 0.0)))
    _, _, se_param = PU.bootstrap_theta(y, idx, cov["beta"], cov["chol"], bgrid, blp, n_boot, rng)
    if not np.isfinite(se_param):
        se_param = np.nan
    se_total = float(np.sqrt(se_ability ** 2 + se_param ** 2)) if np.isfinite(se_param) else np.nan
    return se_ability, se_param, se_total


def load_recovery_map(csv_path: Path, r_col: str = "recovery_r"):
    """(min_scenarios, round(se,2)) -> recovery_r from a grid CSV, if present."""
    out = {}
    if not csv_path.is_file():
        return out
    df = pd.read_csv(csv_path)
    if r_col not in df.columns:
        return out
    for _, r in df.iterrows():
        out[(int(r["min_scenarios"]), round(float(r["se_target"]), 2))] = float(r[r_col])
    return out


def summarize(stop_rule, floor, se, scen, se_ab, se_pa, se_to, n_models, rec_r, rec_src):
    scen = np.asarray(scen, float)
    se_to_arr = np.asarray(se_to, float)
    row = {
        "stop_rule": stop_rule, "op_point": f"f{floor}_se{se:.2f}",
        "min_scenarios": floor, "se_target": se, "n_models": n_models,
        "recovery_r": round(rec_r, 4) if rec_r is not None else np.nan,
        "recovery_r_source": rec_src,
        "scen_mean": round(float(np.mean(scen)), 2), "scen_max": int(np.max(scen)),
    }
    row.update(cellstats(se_to, "se_total"))
    row.update(cellstats(se_ab, "se_ability"))
    row["se_param_median"] = round(float(np.median(np.asarray(se_pa, float)
                                                   [np.isfinite(se_pa)])), 4)
    row["n_se_total_gt_0.15"] = int(np.sum(se_to_arr[np.isfinite(se_to_arr)] > 0.15))
    row["n_se_total_gt_0.20"] = int(np.sum(se_to_arr[np.isfinite(se_to_arr)] > 0.20))
    return row


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    base = ROOT / "bridge_calibration"
    p.add_argument("--bank", type=Path, default=base / "bridge_scenario_fitted_1d_catpool.jsonl")
    p.add_argument("--matrix", type=Path, default=ROOT / "bridgegrade" / "response_matrix.csv")
    p.add_argument("--scenarios", type=Path, default=ROOT / "data" / "Bridge" / "scenarios.jsonl")
    p.add_argument("--out-dir", type=Path, default=base / "experiments" / "13_eap_stop_grid")
    p.add_argument("--tmp-dir", type=Path,
                   default=base / "experiments" / "13_eap_stop_grid" / "_tail_tmp")
    p.add_argument("--eap-recovery-csv", type=Path,
                   default=base / "experiments" / "13_eap_stop_grid" / "recovery_grid.csv")
    p.add_argument("--online-recovery-csv", type=Path,
                   default=base / "experiments" / "06b_operating_point" / "recovery_grid.csv")
    p.add_argument("--floors", type=str, default="0,4,6,8,12,15,20")
    p.add_argument("--se-list", type=str, default="0.08,0.10,0.12,0.15,0.20,0.25,0.30")
    p.add_argument("--online-floors", type=str, default="8,12",
                   help="floors to also recompute under the ONLINE stop for side-by-side.")
    p.add_argument("--eap-grid", type=int, default=321)
    p.add_argument("--range", type=float, default=8.0)
    p.add_argument("--se-param-grid", type=int, default=61)
    p.add_argument("--se-param-range", type=float, default=6.0)
    p.add_argument("--fit-grid", type=int, default=7)
    p.add_argument("--ridge", type=float, default=1e-2)
    p.add_argument("--l-max", type=int, default=60)
    p.add_argument("--max-scenarios", type=int, default=60)
    p.add_argument("--min-evals-per-skill", type=int, default=15)
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260729)
    p.add_argument("--pu-seed", type=int, default=20260801)
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()
    args.tmp_dir.mkdir(parents=True, exist_ok=True)

    floors = [int(x) for x in args.floors.split(",") if x.strip()]
    se_list = [round(float(x), 2) for x in args.se_list.split(",") if x.strip()]
    online_floors = [int(x) for x in args.online_floors.split(",") if x.strip()]

    records, dims, _ = scat.load_fitted_bank(args.bank, "clamp")
    assert len(dims) == 1, "Bridge is 1-D"
    ids = [r["criterion_id"] for r in records]
    scen_of = {r["criterion_id"]: r["scenario_id"] for r in records}
    matrix = pd.read_csv(args.matrix, index_col=0)
    Yraw = matrix.reindex(columns=ids).to_numpy(float)
    models = list(matrix.index)
    row_of = {m: i for i, m in enumerate(models)}
    col_full = {c: i for i, c in enumerate(ids)}
    egrid, elog = scat.build_grid(1, args.eap_grid, args.range)
    gg, lp = E.eap_grid(args.eap_grid, args.range)

    cov = PU.compute_item_cov(args.bank, args.matrix, args.fit_grid, args.ridge)
    Yfull = cov["Ymat"]  # nan->0, aligned to ids / row_of
    bgrid = np.linspace(-args.se_param_range, args.se_param_range, args.se_param_grid)
    blp = -0.5 * bgrid ** 2
    blp = blp - logsumexp(blp)
    rng_boot = np.random.default_rng(args.pu_seed)

    rec_eap = load_recovery_map(args.eap_recovery_csv)
    rec_online = load_recovery_map(args.online_recovery_csv)

    print("=" * 90)
    print(f"TAIL-AWARE re-summarize | EAP floors={floors} SE={se_list} | "
          f"online floors={online_floors}")
    print("=" * 90)

    rows = []

    # --- EAP stop: one forced-long full-data order; all cells post-hoc ---
    print(f"\nforced-long (L={args.l_max}) full-data order (EAP stop) ...", flush=True)
    specL = scat.RunSpec(seed=args.seed, top_n=args.top_n, max_se=0.0, min_evals_per_skill=0,
                         min_scenarios=args.l_max, max_scenarios=args.l_max, selection="trace",
                         mode="cat", runs_dir=str(args.tmp_dir / "fullL"))
    fullL = scat.run_models(models, args.bank, args.matrix, args.scenarios, "clamp", dims,
                            specL, workers=args.workers)
    a_full, b_full = cov["A"][:, 0], cov["b"]
    walks = {r["model"]: E.eap_walk(r["order"], Yraw[row_of[r["model"]]], col_full, scen_of,
                                    a_full, b_full, gg, lp) for r in fullL}
    for floor in floors:
        for se in se_list:
            scen, se_ab, se_pa, se_to = [], [], [], []
            for m in models:
                step, _ = E.eap_stop_point(walks[m], floor, se)
                scen.append(step["n_scen"])
                sa, sp, st = per_model_se(Yfull[row_of[m]], step["idx"], cov, egrid, elog,
                                          bgrid, blp, args.n_boot, rng_boot)
                se_ab.append(sa); se_pa.append(sp); se_to.append(st)
            rr = rec_eap.get((floor, round(se, 2)))
            rows.append(summarize("eap", floor, se, scen, se_ab, se_pa, se_to, len(models),
                                  rr, "13_eap_stop_grid/recovery_grid.csv"))
            r0 = rows[-1]
            print(f"  eap    f{floor:2d} se{se:.2f}: scen {r0['scen_mean']:5.1f}(max {r0['scen_max']:2d}) "
                  f"SEtot mean={r0['se_total_mean']:.3f} sd={r0['se_total_sd']:.3f} "
                  f"max={r0['se_total_max']:.3f} >0.15={r0['n_se_total_gt_0.15']:2d} "
                  f">0.20={r0['n_se_total_gt_0.20']:2d}", flush=True)

    # --- online stop (side-by-side; floors 8/12): engine per cell ---
    for floor in online_floors:
        for se in se_list:
            spec = scat.RunSpec(seed=args.seed, top_n=args.top_n, max_se=se,
                                min_evals_per_skill=args.min_evals_per_skill,
                                min_scenarios=floor, max_scenarios=args.max_scenarios,
                                selection="trace", mode="cat",
                                runs_dir=str(args.tmp_dir / "online_runs"))
            res = scat.run_models(models, args.bank, args.matrix, args.scenarios, "clamp", dims,
                                  spec, workers=args.workers)
            scen, se_ab, se_pa, se_to = [], [], [], []
            for r0 in res:
                m = r0["model"]
                idx = np.array([col_full[c] for c in r0["order"] if c in col_full], dtype=int)
                scen.append(r0["scenarios_administered"])
                sa, sp, st = per_model_se(Yfull[row_of[m]], idx, cov, egrid, elog,
                                          bgrid, blp, args.n_boot, rng_boot)
                se_ab.append(sa); se_pa.append(sp); se_to.append(st)
            rr = rec_online.get((floor, round(se, 2)))
            rows.append(summarize("online", floor, se, scen, se_ab, se_pa, se_to, len(models),
                                  rr, "06b_operating_point/recovery_grid.csv"))
            r1 = rows[-1]
            print(f"  online f{floor:2d} se{se:.2f}: scen {r1['scen_mean']:5.1f}(max {r1['scen_max']:2d}) "
                  f"SEtot mean={r1['se_total_mean']:.3f} sd={r1['se_total_sd']:.3f} "
                  f"max={r1['se_total_max']:.3f} >0.15={r1['n_se_total_gt_0.15']:2d} "
                  f">0.20={r1['n_se_total_gt_0.20']:2d}", flush=True)

    fields = ["stop_rule", "op_point", "min_scenarios", "se_target", "n_models",
              "recovery_r", "recovery_r_source", "scen_mean", "scen_max",
              "se_total_mean", "se_total_sd", "se_total_median", "se_total_q3", "se_total_max",
              "se_ability_mean", "se_ability_sd", "se_ability_median", "se_ability_q3",
              "se_ability_max", "se_param_median", "n_se_total_gt_0.15", "n_se_total_gt_0.20"]
    out_csv = args.out_dir / "recovery_grid_tailstats.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    _write_note(rows, args.out_dir / "TAILSTATS_NOTE.md", se_list)
    shutil.rmtree(args.tmp_dir, ignore_errors=True)
    print(f"\nwrote -> {out_csv}")
    print(f"wrote -> {args.out_dir / 'TAILSTATS_NOTE.md'}")
    return 0


def _md_table(df, floor):
    hdr = ("| SE target | mean scen (max) | recovery r | SE_total mean | SD | median | max | "
           "#>0.15 | #>0.20 |")
    sep = "|---|---|---|---|---|---|---|---|---|"
    lines = [hdr, sep]
    for _, r in df[df["min_scenarios"] == floor].sort_values("se_target", ascending=False).iterrows():
        lines.append(
            f"| {r['se_target']:.2f} | {r['scen_mean']:.1f} ({int(r['scen_max'])}) | "
            f"{r['recovery_r'] if pd.notna(r['recovery_r']) else 'n/a'} | "
            f"{r['se_total_mean']:.3f} | {r['se_total_sd']:.3f} | {r['se_total_median']:.3f} | "
            f"{r['se_total_max']:.3f} | {int(r['n_se_total_gt_0.15'])} | "
            f"{int(r['n_se_total_gt_0.20'])} |")
    return "\n".join(lines)


def _write_note(rows, path: Path, se_list):
    df = pd.DataFrame(rows)
    eap = df[df["stop_rule"] == "eap"]
    online = df[df["stop_rule"] == "online"]
    parts = [
        "# Tail-aware per-cell SE stats (EAP-stop floor x SE grid)",
        "",
        "**Re-summarize only** -- frozen bank reused verbatim (no refit), deployed CAT on the "
        "full 51-model data. SE_ability = fine-grid EAP posterior SD (321 nodes / +/-8) at the "
        "stop; SE_param = observed-info bootstrap over the administered set; "
        "SE_total = sqrt(SE_ability^2 + SE_param^2). `recovery_r` is carried over (read-only) "
        "from the grid CSVs (OOS k-fold; not recomputed). No of-record file was modified.",
        "",
        "## Why the median hid the SE-target benefit",
        "On Bridge's heavy ~18-criterion testlets the MEDIAN model is already well below any SE "
        "target once the min_scenarios floor is met, so `se_total_median` is ~flat across SE "
        "targets. The SE target only helps the TAIL (low-information / extreme-ability models). "
        "The columns below expose that tail: as the SE target tightens, `se_total` mean / SD / "
        "max and the counts above 0.15 / 0.20 drop, at the cost of rising mean / max length.",
        "",
        "## EAP stop -- floor 12",
        _md_table(eap, 12),
        "",
        "## EAP stop -- floor 8",
        _md_table(eap, 8),
    ]
    if not online.empty:
        for fl in sorted(online["min_scenarios"].unique()):
            parts += ["", f"## Online stop -- floor {int(fl)} (side-by-side)", _md_table(online, int(fl))]
    parts += [
        "",
        "## Reading the tradeoff",
        "- `se_total_median` ~flat across SE targets => the median model does not benefit.",
        "- `se_total_mean`, `se_total_sd`, `se_total_max`, `#>0.15`, `#>0.20` FALL as SE tightens "
        "=> the tail is pulled in.",
        "- `scen_mean (max)` RISES as SE tightens => the cost is paid only by the tail (more "
        "scenarios for the few under-measured models); the floor-bound majority is unchanged.",
        "- Net: the SE target is a TAIL-precision guarantee, not a median-precision lever. Pick "
        "it by how many tail models you are willing to leave above a given SE_total, versus the "
        "extra length those models incur.",
    ]
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
