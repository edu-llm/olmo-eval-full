"""Floor x SE_ability OOS grid re-sweep for the TutorEval gemini-3 MINIMAL Rule-0 bank.

LOCAL / STUDY ONLY. The production engine (``tutor_cat/``, ``scripts/scenario_cat_lib.py``,
``scripts/calibrate_mirt.py``) and every canonical study script are imported/called READ-ONLY
and are NEVER modified. Nothing is committed. This script STOPS at the decision-table
checkpoint: it does NOT lock an op-point, does NOT run SE_judge, does NOT build of-record
figures.

Curation path (locked by the user)
----------------------------------
MINIMAL: keep the Rule-0 observed fit (auto zero-variance drop of the 542 all-fail columns ->
1244 fitted items). NO extra <=3/>=49 near-separation / pass-imbalance exclusion (the quantify
step showed the max-info CAT UNDER-selects near-sep/extreme items, administered share 0.14x of
bank share -- an exclusion would be largely inert for the administered test/recovery).

Bank re-used exactly as fit in Phase 1 (NOT refit differently):
  reports/tutoreval_gemini3_recal/bank/rubrics_qmatrix_final_unidim_gemini3_fitted.jsonl

Methodology (mirrors the of-record TutorEval unidim floor x SE OOS grid,
``tutoreval_calibration/scripts/run_oos_grid_unidim.py``):
  * k=5 model-fold OOS, seed 20260729, refit (a,b) per fold via calibrate_mirt.fit_m2pl_em
    (1 dim, ridge 0.01, fit_grid 7, within-fold zero-variance filtering). Q held fixed = ones.
  * ONE forced full-length trace per held-out model per fold (production max-info selection,
    engine cap 70), then OFFLINE-REPLAY every grid cell on that single trace.
  * Stop: EAP-posterior SD on a DENSE 1-D grid (321 nodes over [-8,8]); info-plateau
    delta=0.005 / W=3; cap 70; priority precision -> plateau -> cap -> bank_exhausted.
    Ability at stop = deployed MWLE over the administered items.
  * Grid: floors {10,12,15,20,25} x SE_ability {0.15,0.18,0.20,0.22,0.25,0.27,0.30,0.32}
    = 40 cells.
  * SE_param: observed-information PARAMETRIC bootstrap (the sound frq/bridge method), taken
    per model from the Phase-1 deployed table ``leaderboard_gemini3_f15se22.csv`` (SE_param
    column) as a fixed per-model offset -- NOT the stale nonparametric person-bootstrap; median
    fallback for any missing model. SE_total = sqrt(SE_ability^2 + SE_param^2).
  * %reach = SE_ability (EAP posterior SD at stop) <= target ONLY (the corrected suite
    definition). SE_total is reported SEPARATELY as a precision number, not a gate.
  * Lengths and SE components reported as MEAN +/- SD and MAX (medians hid the SE-target
    benefit in prior studies).
  * Weakly-identified models are IDENTIFIED from the run (best-achievable SE_total at the cap
    still > 0.30) -> an excl-weak grid variant is emitted alongside the all-52 grid.

Usage
-----
    uv run python eduLLM-Evals/reports/tutoreval_gemini3_recal/oppoint_grid/run_oppoint_grid_gemini3.py --workers 6
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

# reports/tutoreval_gemini3_recal/oppoint_grid/ -> .../tutoreval_gemini3_recal/ -> reports/ -> eduLLM-Evals/
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.scenario_cat_lib as scat  # noqa: E402  (read-only import of the shared engine lib)

HERE = Path(__file__).resolve().parent
RECAL = HERE.parent

SE_TOTAL_GATE = 0.30  # kept ONLY as a data-driven weak-model criterion, NOT a reach gate
CAP = 70
FLOORS = [10, 12, 15, 20, 25]
SE_TARGETS = [0.15, 0.18, 0.20, 0.22, 0.25, 0.27, 0.30, 0.32]
PLATEAU_DELTA = 0.005
PLATEAU_W = 3

# Qwen of-record baseline (tutoreval_calibration/summary.json) for the comparison row.
QWEN_BASELINE = {
    "headline_excl_weak": {"recovery_r": 0.95, "recovery_slope": 0.8009, "theta_mae": 0.3329,
                           "median_length_scenarios": 15.0, "median_se_ability": 0.1774,
                           "median_se_total": 0.2171, "pct_reach_corrected": 97.9},
    "all_52": {"recovery_r": 0.9702, "recovery_slope": 0.8131, "theta_mae": 0.3594,
               "median_length_scenarios": 15.0, "median_se_total": 0.2231,
               "pct_reach_corrected": 92.3},
    "weak_models": ["BEE-spoke-data/smol_llama-220M-GQA-fineweb_edu", "ai-forever/mGPT",
                    "allenai/OLMo-1B-hf", "ibm-granite/granite-3.1-2b-instruct"],
}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cm = _load("calibrate_mirt", ROOT / "scripts" / "calibrate_mirt.py")
# The canonical of-record OOS grid harness -- reuse its exact fold machinery read-only.
GRID = _load("run_oos_grid_unidim",
             ROOT / "tutoreval_calibration" / "scripts" / "run_oos_grid_unidim.py")


def _agg_stats(a):
    """mean / sd (ddof=1) / median / max / p90 over finite values."""
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"mean": float("nan"), "sd": float("nan"), "median": float("nan"),
                "max": float("nan"), "p90": float("nan")}
    return {"mean": float(np.mean(a)),
            "sd": float(np.std(a, ddof=1)) if a.size > 1 else 0.0,
            "median": float(np.median(a)), "max": float(np.max(a)),
            "p90": float(np.percentile(a, 90))}


def _cell_row(floor, tgt, rows):
    """Aggregate one grid cell from its per-model rows (a list of dicts)."""
    n = len(rows)
    ls = np.array([r["n_scenarios"] for r in rows], float)
    lc = np.array([r["n_criteria"] for r in rows], float)
    sd = np.array([r["sd"] for r in rows], float)
    sp = np.array([r["se_param"] for r in rows], float)
    st = np.array([r["se_total"] for r in rows], float)
    reach = np.array([1.0 if r["reach"] else 0.0 for r in rows])
    reasons = [r["stop_reason"] for r in rows]
    rc = GRID.recovery([r["theta_ref"] for r in rows], [r["theta_mwle"] for r in rows])

    L = _agg_stats(ls)
    LC = _agg_stats(lc)
    SA = _agg_stats(sd)
    SP = _agg_stats(sp)
    ST = _agg_stats(st)
    return {
        "floor": int(floor), "se_target": float(tgt), "n_models": int(n),
        "recovery_r": round(rc["r"], 4), "recovery_slope": round(rc["slope"], 4),
        "theta_mae": round(rc["theta_mae"], 4),
        "pct_reach": round(100.0 * float(reach.mean()), 1),
        # test length (scenarios) -- MEAN/SD/MAX (+median for reference)
        "len_scen_mean": round(L["mean"], 2), "len_scen_sd": round(L["sd"], 2),
        "len_scen_max": round(L["max"], 1), "len_scen_median": round(L["median"], 1),
        "len_crit_mean": round(LC["mean"], 2), "len_crit_sd": round(LC["sd"], 2),
        "len_crit_max": round(LC["max"], 1),
        # achieved SE_ability (posterior SD) -- MEAN/SD/MAX
        "se_ability_mean": round(SA["mean"], 4), "se_ability_sd": round(SA["sd"], 4),
        "se_ability_max": round(SA["max"], 4), "se_ability_median": round(SA["median"], 4),
        # SE_param (parametric bootstrap; fixed per-model offset) -- MEAN/SD/MAX
        "se_param_mean": round(SP["mean"], 4), "se_param_sd": round(SP["sd"], 4),
        "se_param_max": round(SP["max"], 4),
        # SE_total = sqrt(SE_ability^2 + SE_param^2) -- MEAN/SD/MAX
        "se_total_mean": round(ST["mean"], 4), "se_total_sd": round(ST["sd"], 4),
        "se_total_max": round(ST["max"], 4), "se_total_median": round(ST["median"], 4),
        # stop-reason mix
        "pct_precision": round(100.0 * np.mean([x == "precision" for x in reasons]), 1),
        "pct_plateau": round(100.0 * np.mean([x == "plateau" for x in reasons]), 1),
        "pct_cap": round(100.0 * np.mean([x == "cap" for x in reasons]), 1),
        "pct_bank_exhausted": round(100.0 * np.mean([x == "bank_exhausted" for x in reasons]), 1),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bank", type=Path,
                   default=RECAL / "bank" / "rubrics_qmatrix_final_unidim_gemini3_fitted.jsonl")
    p.add_argument("--matrix", type=Path,
                   default=ROOT / "api_judge_pilot" / "grading_tutoreval" / "response_matrix.csv")
    p.add_argument("--scenarios", type=Path,
                   default=ROOT / "data" / "TutorEval" / "scenarios_final.jsonl")
    p.add_argument("--se-components", type=Path,
                   default=RECAL / "leaderboard_gemini3_f15se22.csv",
                   help="Phase-1 deployed table w/ the PARAMETRIC-bootstrap SE_param per model.")
    p.add_argument("--out-dir", type=Path, default=HERE)
    p.add_argument("--tmp-dir", type=Path, default=HERE / "_tmp")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260729)
    p.add_argument("--fit-grid", type=int, default=7)
    p.add_argument("--ridge", type=float, default=1e-2)
    p.add_argument("--max-iter", type=int, default=200)
    p.add_argument("--ref-grid", type=int, default=61)
    p.add_argument("--ref-range", type=float, default=6.0)
    p.add_argument("--dense-grid", type=int, default=321)
    p.add_argument("--dense-range", type=float, default=8.0)
    p.add_argument("--cap", type=int, default=CAP)
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()

    if not args.matrix.is_file():
        sys.stderr.write(f"ERROR: gemini-3 response matrix not found: {args.matrix}\n")
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "figures").mkdir(parents=True, exist_ok=True)
    args.tmp_dir.mkdir(parents=True, exist_ok=True)

    # --- load the Phase-1 minimal Rule-0 bank (clamp negatives at consume) --------------------
    records, dims, bank_stats = scat.load_fitted_bank(args.bank, "clamp")
    if dims != ["ability"]:
        sys.stderr.write(f"ERROR: expected unidim bank dims=['ability'], got {dims}\n")
        return 2
    d = dims[0]
    ids = [r["criterion_id"] for r in records]
    scen_of = {r["criterion_id"]: r["scenario_id"] for r in records}
    crit_of = {r["criterion_id"]: r.get("criterion", "") for r in records}
    Q_all = np.array([[int(r["q_modeled"][dd]) for dd in dims] for r in records])

    prov = scat.verify_provenance(args.bank, args.matrix)

    matrix = pd.read_csv(args.matrix, index_col=0)
    sub = matrix.reindex(columns=ids)
    Yraw = sub.to_numpy(float)
    Mall = ~np.isnan(Yraw)
    models = list(matrix.index)
    row_of = {m: i for i, m in enumerate(models)}
    n_models = len(models)
    print(f"bank={args.bank.name} dims={dims} models={n_models} items={len(ids)} "
          f"k={args.k} seed={args.seed} matrix_aligned={prov['aligned']}")

    # --- SE_param per model: PARAMETRIC bootstrap (Phase-1 deployed), median fallback ---------
    se_param_by: dict[str, float] = {}
    if args.se_components.is_file():
        sc = pd.read_csv(args.se_components)
        col = "SE_param" if "SE_param" in sc.columns else "se_param"
        mcol = "model"
        se_param_by = {str(m): float(v) for m, v in zip(sc[mcol], sc[col]) if np.isfinite(v)}
    se_param_median = float(np.median(list(se_param_by.values()))) if se_param_by else 0.1013
    se_param = np.array([se_param_by.get(m, se_param_median) for m in models])
    print(f"SE_param source={'parametric bootstrap (Phase-1)' if se_param_by else 'median'} "
          f"median={se_param_median:.4f} n_have={len(se_param_by)}")

    # dense stop grid + reference EAP grid
    gg = np.linspace(-args.dense_range, args.dense_range, args.dense_grid)
    lp = -0.5 * gg ** 2
    lp = lp - logsumexp(lp)
    egrid, elog = scat.build_grid(1, args.ref_grid, args.ref_range)

    folds = GRID.make_folds(models, args.k, args.seed)

    # ------- ONE forced full-length trace per held-out model (per fold) -----------------------
    per_model: dict[str, dict] = {}
    for f in range(args.k):
        test = folds[f]
        train = [m for m in models if m not in set(test)]
        tr_idx = [row_of[m] for m in train]
        Ytr, Mtr = Yraw[tr_idx], Mall[tr_idx]
        keep = [j for j in range(len(ids))
                if Mtr[:, j].sum() >= 2 and 0 < Ytr[Mtr[:, j], j].sum() < Mtr[:, j].sum()]
        keep = np.array(keep, dtype=int)
        fit = cm.fit_m2pl_em(np.nan_to_num(Ytr[:, keep], nan=0.0), Mtr[:, keep],
                             Q_all[keep], args.fit_grid, ridge=args.ridge, max_iter=args.max_iter)
        Ak, bk = fit["A"], fit["b"]
        ak = Ak[:, 0]
        kept_ids = [ids[j] for j in keep]
        colk = {c: i for i, c in enumerate(kept_ids)}
        print(f"  fold {f}: TRAIN={len(train)} TEST={len(test)} fit {len(kept_ids)} items "
              f"(loglik={fit['loglik']:.0f}, iters={fit['n_iter']})", flush=True)

        fold_bank = args.tmp_dir / f"fold{f}_bank.jsonl"
        with fold_bank.open("w", encoding="utf-8") as fh:
            for jj, cid in enumerate(kept_ids):
                fh.write(json.dumps({"criterion_id": cid, "scenario_id": scen_of[cid],
                                     "criterion": crit_of[cid],
                                     "discrimination": {d: float(Ak[jj, 0])},
                                     "q_modeled": {d: 1}, "difficulty": float(bk[jj])}) + "\n")

        subte = matrix.loc[test].reindex(columns=kept_ids)
        Yte = np.nan_to_num(subte.to_numpy(float), nan=0.0)
        Mte = ~np.isnan(subte.to_numpy(float))
        theta_ref = scat.eap_all_models(Yte, Mte, Ak, bk, egrid, elog)

        res = GRID.run_forced(test, fold_bank, args.matrix, args.scenarios, dims, args.seed,
                              args.cap, args.workers, args.tmp_dir / f"runs_f{f}")
        for ti, m in enumerate(test):
            walk = GRID.eap_walk(res[m]["order"], Yte[ti], colk, scen_of, ak, bk, gg, lp)
            per_model[m] = {"fold": f, "walk": walk, "theta_ref": float(theta_ref[ti][0]),
                            "y": Yte[ti], "Ak": Ak, "bk": bk, "colk": colk}
        shutil.rmtree(args.tmp_dir / f"runs_f{f}", ignore_errors=True)

    # ------- identify weakly-identified models (data-driven) ----------------------------------
    weak = []
    best_se_total = {}
    for m in models:
        walk = per_model[m]["walk"]
        sp = float(se_param_by.get(m, se_param_median))
        if walk:
            best_se_total[m] = min(float(np.sqrt(s["eap_sd"] ** 2 + sp ** 2)) for s in walk)
        else:
            best_se_total[m] = float("nan")
        if not (best_se_total[m] <= SE_TOTAL_GATE):
            weak.append(m)

    # ------- OFFLINE replay: all cells on the single trace per model --------------------------
    per_model_cell_rows: list[dict] = []
    for floor in FLOORS:
        for tgt in SE_TARGETS:
            for m in models:
                info = per_model[m]
                walk = info["walk"]
                step, reason = GRID.stop_point(walk, floor, tgt, cap=args.cap)
                sp = float(se_param_by.get(m, se_param_median))
                if step is None or step["idx"].size == 0:
                    n_scen = n_crit = 0
                    sd = float("nan")
                    th_mw = info["theta_ref"]
                else:
                    n_scen = step["n_scen"]
                    n_crit = step["n_crit"]
                    sd = step["eap_sd"]
                    idx = step["idx"]
                    th_ba = scat.eap_subset(info["y"], idx, info["Ak"], info["bk"], egrid, elog)
                    th_mw_v, _ = scat.mwle_subset(info["y"], idx, info["Ak"], info["bk"], th_ba)
                    th_mw = float(th_mw_v[0])
                se_total = float(np.sqrt(sd ** 2 + sp ** 2)) if np.isfinite(sd) else float("nan")
                reach = bool(np.isfinite(sd) and sd <= tgt)  # CORRECTED: SE_ability only
                per_model_cell_rows.append({
                    "model": m, "floor": floor, "se_target": tgt,
                    "n_scenarios": n_scen, "n_criteria": n_crit, "stop_reason": reason,
                    "sd": sd, "se_param": sp, "se_total": se_total,
                    "theta_ref": info["theta_ref"], "theta_mwle": th_mw,
                    "reach": reach, "weak": m in weak})

    pm = pd.DataFrame(per_model_cell_rows)
    pm.to_csv(args.out_dir / "oos_per_model_per_cell.csv", index=False)

    # per-cell aggregates: all-52 and excl-weak
    def _grid(df):
        rows = []
        for floor in FLOORS:
            for tgt in SE_TARGETS:
                cell = df[(df.floor == floor) & (df.se_target == tgt)]
                if len(cell):
                    rows.append(_cell_row(floor, tgt, cell.to_dict("records")))
        return pd.DataFrame(rows)

    grid_all = _grid(pm)
    grid_all.to_csv(args.out_dir / "oos_per_cell_grid_all52.csv", index=False)
    grid_excl = _grid(pm[~pm.weak]) if weak else grid_all.copy()
    grid_excl.to_csv(args.out_dir / "oos_per_cell_grid_excl_weak.csv", index=False)

    # heatmaps (mean, not median): SE_total / test length / recovery r
    _heatmaps(grid_all, args.out_dir / "figures" / "oos_grid_heatmaps_all52.png",
              f"TutorEval gemini-3 minimal Rule-0 OOS grid (all {n_models} models; mean)")
    _heatmaps(grid_excl, args.out_dir / "figures" / "oos_grid_heatmaps_excl_weak.png",
              f"TutorEval gemini-3 minimal Rule-0 OOS grid (excl-weak N={n_models - len(weak)}; mean)")

    # combined decision-table CSV (all + excl in one file, tagged)
    ga = grid_all.copy(); ga.insert(0, "population", "all52")
    ge = grid_excl.copy(); ge.insert(0, "population", "excl_weak")
    dec = pd.concat([ga, ge], ignore_index=True)
    dec.to_csv(args.out_dir / "decision_table.csv", index=False)

    # reference row f15/se22 (all + excl) + Qwen comparison
    def _ref_row(g):
        r = g[(g.floor == 15) & (g.se_target == 0.22)]
        return r.iloc[0].to_dict() if len(r) else {}

    ref_all = _ref_row(grid_all)
    ref_excl = _ref_row(grid_excl)

    _decision_table_md(grid_all, grid_excl, weak, best_se_total, ref_all, ref_excl,
                       args.out_dir / "DECISION_TABLE.md", prov, n_models)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": ("LOCAL / STUDY ONLY -- production engine untouched; nothing committed/staged; "
                   "MINIMAL Rule-0 gemini-3 bank; STOP at decision-table checkpoint (no op-point "
                   "locked, no SE_judge, no of-record)."),
        "study": "OOS floor x SE EAP-posterior-stop grid, TutorEval gemini-3 MINIMAL Rule-0 bank",
        "curation": ("MINIMAL: Rule-0 observed fit (auto zero-variance drop of 542 all-fail cols "
                     "-> 1244 fitted items); NO extra near-sep / pass-imbalance exclusion."),
        "engine": ("read-only: tutor_cat + scripts/scenario_cat_lib (production); refit via "
                   "scripts/calibrate_mirt.fit_m2pl_em; fold machinery reused from "
                   "tutoreval_calibration/scripts/run_oos_grid_unidim.py"),
        "bank": str(args.bank), "matrix": str(args.matrix), "scenarios": str(args.scenarios),
        "matrix_provenance": prov, "bank_load_stats": bank_stats,
        "dims": dims, "n_models": n_models, "k": args.k, "seed": args.seed,
        "fit_config": {"fit_grid": args.fit_grid, "ridge": args.ridge, "max_iter": args.max_iter,
                       "negative_policy_at_load": "clamp"},
        "stop_rule": {"dense_grid_nodes": args.dense_grid, "dense_range": args.dense_range,
                      "plateau_delta": PLATEAU_DELTA, "plateau_w": PLATEAU_W, "cap": args.cap,
                      "priority": ["precision", "plateau", "cap", "bank_exhausted"],
                      "ability_at_stop": "MWLE"},
        "grid": {"floors": FLOORS, "se_targets": SE_TARGETS,
                 "n_cells": len(FLOORS) * len(SE_TARGETS)},
        "reach_definition": "SE_ability (EAP posterior SD at stop) <= target ONLY (corrected)",
        "se_param": {"method": "observed-information PARAMETRIC bootstrap (sound frq/bridge method)",
                     "source": str(args.se_components),
                     "usage": "fixed per-model offset (from Phase-1 deployed op-point table)",
                     "median": round(se_param_median, 4), "n_models_have": len(se_param_by)},
        "se_total": "sqrt(SE_ability^2 + SE_param^2) (SE_judge is Phase 3, NOT in this grid)",
        "weak_models": {"criterion": "best achievable SE_total at cap > 0.30 (data-driven)",
                        "n": len(weak), "models": weak,
                        "best_se_total_at_cap": {m: round(best_se_total[m], 4) for m in weak}},
        "reference_row_f15_se22": {"all52": ref_all, "excl_weak": ref_excl},
        "qwen_baseline": QWEN_BASELINE,
        "outputs": {
            "per_model_per_cell": "oos_per_model_per_cell.csv",
            "grid_all52": "oos_per_cell_grid_all52.csv",
            "grid_excl_weak": "oos_per_cell_grid_excl_weak.csv",
            "decision_table_csv": "decision_table.csv",
            "decision_table_md": "DECISION_TABLE.md",
            "heatmaps_all52": "figures/oos_grid_heatmaps_all52.png",
            "heatmaps_excl_weak": "figures/oos_grid_heatmaps_excl_weak.png",
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    shutil.rmtree(args.tmp_dir, ignore_errors=True)
    print(f"\nwrote -> {args.out_dir}")
    print(f"weakly-identified (SE_total>0.30 even at cap): {len(weak)} -> {weak}")
    if ref_excl:
        print(f"[f15/se22 excl-weak] r={ref_excl['recovery_r']} slope={ref_excl['recovery_slope']} "
              f"MAE={ref_excl['theta_mae']} reach={ref_excl['pct_reach']}% "
              f"len={ref_excl['len_scen_mean']}+/-{ref_excl['len_scen_sd']} "
              f"SE_total={ref_excl['se_total_mean']}")
    print(f"[Qwen baseline excl-weak] r=0.95 reach=97.9%")
    return 0


def _heatmaps(grid: pd.DataFrame, path: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    floors = sorted(grid["floor"].unique())
    tgts = sorted(grid["se_target"].unique())

    def mat(col):
        M = np.full((len(floors), len(tgts)), np.nan)
        for _, r in grid.iterrows():
            M[floors.index(r["floor"]), tgts.index(r["se_target"])] = r[col]
        return M

    panels = [("se_total_mean", "mean SE_total", "viridis_r", "%.3f"),
              ("len_scen_mean", "mean test length (scenarios)", "magma", "%.1f"),
              ("recovery_r", "OOS recovery r (ability)", "viridis", "%.3f")]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (col, ttl, cmap, fmt) in zip(axes.ravel(), panels):
        M = mat(col)
        im = ax.imshow(M, aspect="auto", cmap=cmap, origin="upper")
        ax.set_xticks(range(len(tgts)), [f"{t:g}" for t in tgts])
        ax.set_yticks(range(len(floors)), [str(f) for f in floors])
        ax.set_xlabel("SE_ability target"); ax.set_ylabel("floor (min scenarios)")
        ax.set_title(ttl, fontsize=11)
        for i in range(len(floors)):
            for j in range(len(tgts)):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, fmt % M[i, j], ha="center", va="center",
                            fontsize=7.5, color="w")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _decision_table_md(grid_all, grid_excl, weak, best_se_total, ref_all, ref_excl,
                       path: Path, prov, n_models):
    def _row(r):
        return (f"| {int(r['floor'])} | {r['se_target']:g} | "
                f"{r['len_scen_mean']:g}\u00b1{r['len_scen_sd']:g} (max {r['len_scen_max']:g}) | "
                f"{r['pct_reach']:g}% | {r['recovery_r']:.3f} | {r['recovery_slope']:.3f} | "
                f"{r['theta_mae']:.3f} | "
                f"{r['se_ability_mean']:.3f}\u00b1{r['se_ability_sd']:.3f} | "
                f"{r['se_param_mean']:.3f} | "
                f"{r['se_total_mean']:.3f}\u00b1{r['se_total_sd']:.3f} (max {r['se_total_max']:.3f}) | "
                f"{r['pct_cap']:g}% |")

    hdr = ("| floor | SE tgt | len scen (mean\u00b1sd) | %reach | rec r | slope | \u03b8-MAE | "
           "SE_abil (mean\u00b1sd) | SE_param | SE_total (mean\u00b1sd) | %cap |")
    sep = "|---|---|---|---|---|---|---|---|---|---|---|"

    lines = [
        "# TutorEval gemini-3 MINIMAL Rule-0 -- floor\u00d7SE OOS grid: DECISION TABLE",
        "",
        "**LOCAL / STUDY ONLY.** Engine untouched; nothing committed/staged. Op-point NOT locked "
        "(review checkpoint). SE_judge and of-record figures NOT run.",
        "",
        "- Curation: **MINIMAL Rule-0** (auto zero-variance drop of 542 all-fail cols "
        "\u2192 1244 fitted items; NO extra near-sep / pass-imbalance exclusion).",
        f"- Bank: `bank/rubrics_qmatrix_final_unidim_gemini3_fitted.jsonl` (single `ability`; "
        f"matrix aligned={prov['aligned']}).",
        f"- N models: {n_models}. Fit: 1-D 2PL, ridge 0.01, fit_grid 7. Stop: dense-grid EAP SD, "
        "plateau \u03b4=0.005/W=3, cap 70; ability at stop = MWLE. OOS: k=5 model-fold, seed 20260729.",
        "- **%reach = SE_ability (EAP posterior SD at stop) \u2264 target ONLY** (corrected suite "
        "definition). SE_total = \u221a(SE_ability\u00b2 + SE_param\u00b2) reported SEPARATELY (not a gate).",
        "- **SE_param** = observed-information **parametric bootstrap** (sound frq/bridge method), "
        "fixed per-model offset from the Phase-1 deployed table (NOT the stale person-bootstrap).",
        "- Lengths / SE components as **mean\u00b1sd (+max)** (medians hid the SE-target benefit).",
        f"- Weakly-identified (best SE_total at cap > 0.30): **{len(weak)}** \u2192 "
        f"`{', '.join(weak) if weak else 'none'}`.",
        "",
        "## f15 / se22 reference row vs Qwen of-record baseline",
        "",
        "| population | r | slope | \u03b8-MAE | %reach | len scen (mean\u00b1sd) | SE_total (mean) |",
        "|---|---|---|---|---|---|---|",
    ]
    if ref_excl:
        lines.append(f"| **gemini-3 excl-weak** | {ref_excl['recovery_r']:.3f} | "
                     f"{ref_excl['recovery_slope']:.3f} | {ref_excl['theta_mae']:.3f} | "
                     f"{ref_excl['pct_reach']:g}% | "
                     f"{ref_excl['len_scen_mean']:g}\u00b1{ref_excl['len_scen_sd']:g} | "
                     f"{ref_excl['se_total_mean']:.3f} |")
    if ref_all:
        lines.append(f"| gemini-3 all-{n_models} | {ref_all['recovery_r']:.3f} | "
                     f"{ref_all['recovery_slope']:.3f} | {ref_all['theta_mae']:.3f} | "
                     f"{ref_all['pct_reach']:g}% | "
                     f"{ref_all['len_scen_mean']:g}\u00b1{ref_all['len_scen_sd']:g} | "
                     f"{ref_all['se_total_mean']:.3f} |")
    qe = QWEN_BASELINE["headline_excl_weak"]
    qa = QWEN_BASELINE["all_52"]
    lines.append(f"| _Qwen baseline excl-weak (N=48)_ | {qe['recovery_r']:.3f} | "
                 f"{qe['recovery_slope']:.3f} | {qe['theta_mae']:.3f} | "
                 f"{qe['pct_reach_corrected']:g}% | ~{qe['median_length_scenarios']:g} (median) | "
                 f"{qe['median_se_total']:.3f} (median) |")
    lines.append(f"| _Qwen baseline all-52_ | {qa['recovery_r']:.3f} | {qa['recovery_slope']:.3f} | "
                 f"{qa['theta_mae']:.3f} | {qa['pct_reach_corrected']:g}% | "
                 f"~{qa['median_length_scenarios']:g} (median) | {qa['median_se_total']:.3f} (median) |")
    lines += ["",
              "> Qwen length/SE_total are reported as medians in the of-record package; recovery r, "
              "slope, \u03b8-MAE and %reach are directly comparable.", ""]

    lines += ["## Full grid \u2014 excl-weak (headline population)", "", hdr, sep]
    for _, r in grid_excl.sort_values(["floor", "se_target"]).iterrows():
        lines.append(_row(r))
    lines += ["", "## Full grid \u2014 all-52", "", hdr, sep]
    for _, r in grid_all.sort_values(["floor", "se_target"]).iterrows():
        lines.append(_row(r))

    lines += ["", "> Operating point is left OPEN. Lower SE targets + higher floors buy precision "
              "(lower SE_total, lower \u03b8-MAE) at the cost of test length and more plateau/cap stops. "
              "Pick per the length\u2194precision\u2194recovery tradeoff."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
