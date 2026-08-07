"""OUT-OF-SAMPLE floor x SE_ability GRID study for the EAP-posterior stop CAT (TutorBench, 2-skill).

STUDY-ONLY. Production engine untouched (``tutor_cat/engine.py``, ``selector.py``). Nothing
committed. This is the honest OOS counterpart of the in-sample op-point grid study
(``eap_stop_grid_study.py``): instead of picking a cell in-sample on 33 models and spot-checking,
we run the ENTIRE 35-cell (floor x SE) grid OUT-OF-SAMPLE on the full 115-model set with a proper
k-fold, refit-per-fold protocol, so the op-point can be SELECTED from OOS metrics.

Design (mirrors the validated WildBench 1-D OOS reference ``prototype_eap_stop.py`` + exp-05
k-fold, generalized to 2 skills; reuses ``scenario_cat_lib`` + ``calibrate_mirt`` + the grid
study's dense-marginal-SD stop / MWLE code):

  1. k=5 person/model folds, seed 20260729 (suite convention), on the FULL 115 models.
  2. For EACH fold: refit the 2-skill confirmatory M2PL on TRAIN models only, using the SAME
     fitter/config/ridge as the of-record fitted bank (``calibrate_mirt.fit_m2pl_em``,
     fit_grid=7, ridge=1e-2, max_iter=200), with within-fold zero-variance criterion filtering
     (>=2 TRAIN observations and both a pass and a fail among them). Q (which skill each criterion
     loads) is frozen from the bank's ``q_modeled``. Freeze the fold bank.
  3. For each HELD-OUT test model: load the fold bank with negative_policy=clamp (runtime requires
     a>=0) so reference, stop and engine all use the SAME clamped params; compute the full-bank
     dense-grid EAP reference theta (all observed items) and drive the REAL engine to FULL bank
     exhaustion once, recording the administered order.
  4. OFFLINE-replay ALL 35 (floor, SE) cells on that single full trace: at each scenario boundary
     compute the honest per-skill MARGINAL SD (marginal of the joint 2-D posterior) on the dense
     grid + the deployed MWLE theta; ``resolve_stop`` truncates the trace per cell
     (precision -> info_plateau(delta=0.005,W=3) -> cap(70) -> bank_exhausted).
  5. Aggregate OOS pairs (theta_ref, theta_at_stop) across all folds, per cell, per skill.

Headline metrics EXCLUDE the 2 weakly-calibrated models (``Qwen/Qwen1.5-1.8B``,
``BSC-LT/salamandra-7b-instruct``); their per-cell fate is reported separately. SE_total uses the
115-min12 per-model SE_param as a FIXED offset (NOT per-fold): SE_total=sqrt(SD^2 + SE_param^2).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import scenario_cat_lib as scl  # noqa: E402
import eap_stop_grid_study as G  # noqa: E402  (reuse dense_marginal_meanvar/boundaries/resolve_stop)
import calibrate_mirt as cm  # noqa: E402

# ---- NEW OOS op-point grid (floors now reach the binding range 20/25) ----------
FLOORS = [10, 12, 15, 20, 25]
SE_TARGETS = [0.15, 0.20, 0.22, 0.25, 0.27, 0.30, 0.32]
CAP = 70
PLATEAU_W = 3
PLATEAU_DELTA = 0.005
SE_TOTAL_THRESHOLDS = [0.25, 0.30]
WEAK_MODELS = ("Qwen/Qwen1.5-1.8B", "BSC-LT/salamandra-7b-instruct")


def make_folds(models, k, seed):
    """Person/model-level folds by round-robin over a seeded permutation (suite convention)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(models))
    folds = [[] for _ in range(k)]
    for pos, i in enumerate(idx):
        folds[pos % k].append(models[int(i)])
    return [sorted(f) for f in folds]


# ---------------------------------------------------------------------------
# per-(fold) worker: load the frozen fold bank (clamped), then for each held-out
# test model build the full-exhaustion trace + dense SD/MWLE per boundary + the
# full-bank EAP reference theta. Each model is independent.
# ---------------------------------------------------------------------------
_S: dict = {}


def _init_worker(fold_bank, matrix_path, scenarios_path, negative_policy, stop_nodes, cap,
                 spec_dict):
    scl._limit_blas_threads()
    records, dims, _ = scl.load_fitted_bank(Path(fold_bank), negative_policy)
    rubrics = scl.build_rubrics(records, dims)
    scen_raw = scl.load_scenarios(Path(scenarios_path))
    matrix = pd.read_csv(matrix_path, index_col=0)
    ids_all, A_all, b_all = scl.assemble_arrays(records, dims)
    keep = [i for i, c in enumerate(ids_all) if c in matrix.columns]
    ids = [ids_all[i] for i in keep]
    grid, log_prior = scl.build_grid(len(dims), stop_nodes)
    _S.update(dims=dims, rubrics=rubrics, scen_raw=scen_raw, matrix=matrix, ids=ids,
              A=A_all[keep], b=b_all[keep], col={c: i for i, c in enumerate(ids)},
              scen_of={cid: r.scenario_id for cid, r in rubrics.items()},
              grid=grid, log_prior=log_prior, cap=cap, spec=scl.RunSpec(**spec_dict))


def _run_model(model: str) -> dict | None:
    S = _S
    Yrow = np.nan_to_num(S["matrix"].loc[model].reindex(S["ids"]).to_numpy(dtype=float))
    obs = ~np.isnan(S["matrix"].loc[model].reindex(S["ids"]).to_numpy(dtype=float))
    obs_idx = np.where(obs)[0]
    # full-bank dense-grid EAP reference theta on the fold-refit (clamped) params. Use the
    # CHUNKED marginal estimator (identical math to eap_subset's mean, bounded memory so the
    # ~3.6k-item x 25921-node boundary does not blow up with many parallel workers).
    if obs_idx.size:
        theta_ref, _ = G.dense_marginal_meanvar(Yrow[obs_idx], S["A"][obs_idx], S["b"][obs_idx],
                                                S["grid"], S["log_prior"])
    else:
        theta_ref = np.zeros(len(S["dims"]))
    # drive the REAL engine to full bank exhaustion -> administration order
    rec = scl.run_one_model(model, S["rubrics"], S["scen_raw"], S["matrix"].loc[model],
                            S["dims"], S["spec"])
    bnds = G.boundaries_from_order(rec["order"], S["col"], S["scen_of"])
    if not bnds:
        return None
    total = bnds[-1][0]
    b_eval = min(S["cap"], total)
    nscen_series, sd_series, mwle_series = [], [], []
    for nscen, idx in bnds:
        if nscen > b_eval:
            break
        mean, var = G.dense_marginal_meanvar(Yrow[idx], S["A"][idx], S["b"][idx],
                                             S["grid"], S["log_prior"])
        theta_mwle, _ = scl.mwle_subset(Yrow, idx, S["A"], S["b"], mean)
        nscen_series.append(nscen)
        sd_series.append(np.sqrt(var))
        mwle_series.append(np.asarray(theta_mwle, float))
    return {"model": model, "total_scenarios": total, "theta_ref": np.asarray(theta_ref, float),
            "nscen": nscen_series, "sd": sd_series, "mwle": mwle_series}


def refit_fold(train, models, row_of, Yraw, Mall, ids, Q_all, scen_of, crit_of, dims,
               fold_bank_path, fit_grid, ridge, max_iter):
    """Refit the 2-skill M2PL on TRAIN only (within-fold zero-variance filtering) and write the
    frozen fold bank in the fitted-bank schema. Returns fit diagnostics."""
    tr_idx = [row_of[m] for m in train]
    Ytr, Mtr = Yraw[tr_idx], Mall[tr_idx]
    keep = []
    for j in range(len(ids)):
        obs = Mtr[:, j]
        if obs.sum() >= 2:
            vals = Ytr[obs, j]
            if 0 < vals.sum() < obs.sum():
                keep.append(j)
    keep = np.array(keep, dtype=int)
    Q_k = Q_all[keep]
    fit = cm.fit_m2pl_em(np.nan_to_num(Ytr[:, keep], nan=0.0), Mtr[:, keep], Q_k, fit_grid,
                         ridge=ridge, max_iter=max_iter)
    A_f, b_f = fit["A"], fit["b"]
    kept_ids = [ids[j] for j in keep]
    with fold_bank_path.open("w", encoding="utf-8") as fh:
        for jj, cid in enumerate(kept_ids):
            rec = {"criterion_id": cid, "scenario_id": scen_of[cid], "criterion": crit_of[cid],
                   "discrimination": {d: float(A_f[jj, kk]) for kk, d in enumerate(dims)},
                   "q_modeled": {d: int(Q_k[jj, kk]) for kk, d in enumerate(dims)},
                   "difficulty": float(b_f[jj])}
            fh.write(json.dumps(rec) + "\n")
    return {"n_items": len(kept_ids), "loglik": float(fit["loglik"]), "n_iter": int(fit["n_iter"]),
            "converged": bool(fit["converged"])}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bank", type=Path,
                    default=ROOT / "data/TutorBench/rubrics_qmatrix_calibrated_2skill_115_fitted.jsonl")
    ap.add_argument("--matrix", type=Path,
                    default=ROOT / "staging/response_matrix_full_nonopt_115.csv")
    ap.add_argument("--scenarios", type=Path, default=ROOT / "data/TutorBench/scenarios.jsonl")
    ap.add_argument("--se-components", type=Path,
                    default=ROOT / "regenerated_figures/scenario_level_115_min12/param_uncertainty/2_skills/leaderboard_se_components.csv")
    ap.add_argument("--out-dir", type=Path,
                    default=ROOT / "reports/eap_oos_grid_tutorbench_2skill")
    ap.add_argument("--tmp-dir", type=Path, default=ROOT / "staging" / "_eap_oos_grid")
    ap.add_argument("--stop-nodes", type=int, default=161, help="DENSE stop/reference grid nodes/dim")
    ap.add_argument("--cap", type=int, default=CAP)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260729)
    ap.add_argument("--fit-grid", type=int, default=7)
    ap.add_argument("--ridge", type=float, default=1e-2)
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--negative-policy", default="clamp")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit-test", type=int, default=0, help="debug: first N test models per fold")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "figures").mkdir(parents=True, exist_ok=True)
    args.tmp_dir.mkdir(parents=True, exist_ok=True)

    # bank structure (ids/scenario/criterion text/Q) + matrix
    records, dims, stats = scl.load_fitted_bank(args.bank, args.negative_policy)
    ids = [r["criterion_id"] for r in records]
    scen_of = {r["criterion_id"]: r["scenario_id"] for r in records}
    crit_of = {r["criterion_id"]: r.get("criterion", "") for r in records}
    Q_all = np.array([[int(r["q_modeled"][d]) for d in dims] for r in records])
    matrix = pd.read_csv(args.matrix, index_col=0)
    Yraw = matrix.reindex(columns=ids).to_numpy(float)
    Mall = ~np.isnan(Yraw)
    models = list(matrix.index)
    row_of = {m: i for i, m in enumerate(models)}
    folds = make_folds(models, args.k, args.seed)

    n_joint = args.stop_nodes ** len(dims)
    print(f"[data] bank={args.bank.name} matrix={args.matrix.name} dims={dims} "
          f"models={len(models)} items={len(ids)} k={args.k} seed={args.seed}")
    print(f"[grid] stop/ref {args.stop_nodes} nodes/dim ({n_joint} joint); cap={args.cap}; "
          f"fit_grid={args.fit_grid} ridge={args.ridge} neg_policy={args.negative_policy}")
    print(f"[folds] sizes={[len(f) for f in folds]}")

    # per-model SE_param (FIXED offset from the 115-min12 study; NOT per-fold)
    se_df = pd.read_csv(args.se_components, index_col=0)
    se_df = se_df[~se_df.index.duplicated(keep="first")]
    med_param = {d: float(np.median(se_df[f"se_param_{d}"])) for d in dims}
    se_param, n_param_fallback = {}, 0
    for m in models:
        if m in se_df.index:
            se_param[m] = {d: float(se_df.loc[m, f"se_param_{d}"]) for d in dims}
        else:
            se_param[m] = dict(med_param)
            n_param_fallback += 1

    # ---- k-fold: refit TRAIN, then full-exhaustion trace + dense SD/MWLE per TEST model ----
    from concurrent.futures import ProcessPoolExecutor

    spec = scl.RunSpec(seed=args.seed, max_se=1e-9, min_evals_per_skill=0, min_scenarios=0,
                       max_scenarios=1_000_000, selection="trace", mode="cat", write_logs=False,
                       runs_dir=str(args.tmp_dir / "engine_runs"))
    all_traces: list[dict] = []
    fold_diag = []
    for f in range(args.k):
        test = folds[f]
        if args.limit_test:
            test = test[: args.limit_test]
        train = [m for m in models if m not in set(folds[f])]
        fold_bank = args.tmp_dir / f"fold{f}_bank.jsonl"
        diag = refit_fold(train, models, row_of, Yraw, Mall, ids, Q_all, scen_of, crit_of,
                          dims, fold_bank, args.fit_grid, args.ridge, args.max_iter)
        fold_diag.append({"fold": f, "n_train": len(train), "n_test": len(test), **diag})
        print(f"  fold {f}: TRAIN={len(train)} TEST={len(test)} fit {diag['n_items']} items "
              f"(loglik={diag['loglik']:.0f} iters={diag['n_iter']} conv={diag['converged']})",
              flush=True)
        initargs = (str(fold_bank), str(args.matrix), str(args.scenarios), args.negative_policy,
                    args.stop_nodes, args.cap, asdict(spec))
        if args.workers and args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                                     initargs=initargs) as ex:
                for tr in ex.map(_run_model, test):
                    if tr is not None:
                        tr["fold"] = f
                        all_traces.append(tr)
        else:
            _init_worker(*initargs)
            for m in test:
                tr = _run_model(m)
                if tr is not None:
                    tr["fold"] = f
                    all_traces.append(tr)
        print(f"    -> {len(test)} test traces done", flush=True)

    print(f"[traces] total held-out model traces: {len(all_traces)}")

    # ---- offline replay of ALL 35 cells on every held-out trace ----
    per_cell_rows, per_model_cell_rows = [], []
    weak_set = set(WEAK_MODELS)
    for floor in FLOORS:
        for se_t in SE_TARGETS:
            # accumulate excl-2 headline pool + weak pool
            pool = {"x": {d: [] for d in dims}, "y": {d: [] for d in dims},
                    "sd": {d: [] for d in dims}, "stot": {d: [] for d in dims},
                    "len": [], "cap": [], "plat": []}
            for tr in all_traces:
                i, reason = G.resolve_stop(tr["sd"], tr["nscen"], floor, se_t,
                                           PLATEAU_DELTA, PLATEAU_W, args.cap)
                length = tr["nscen"][i]
                sd_stop = tr["sd"][i]
                mwle_stop = tr["mwle"][i]
                sp = se_param[tr["model"]]
                stot = {d: float(np.sqrt(sd_stop[k] ** 2 + sp[d] ** 2))
                        for k, d in enumerate(dims)}
                is_weak = tr["model"] in weak_set
                per_model_cell_rows.append({
                    "floor": floor, "se_target": se_t, "model": tr["model"], "fold": tr["fold"],
                    "is_weak_excluded": is_weak, "length": length, "stop_reason": reason,
                    **{f"theta_ref_{d}": float(tr["theta_ref"][k]) for k, d in enumerate(dims)},
                    **{f"theta_mwle_{d}": float(mwle_stop[k]) for k, d in enumerate(dims)},
                    **{f"sd_stop_{d}": float(sd_stop[k]) for k, d in enumerate(dims)},
                    **{f"se_total_{d}": stot[d] for d in dims},
                })
                if is_weak:
                    continue
                for k, d in enumerate(dims):
                    pool["x"][d].append(float(tr["theta_ref"][k]))
                    pool["y"][d].append(float(mwle_stop[k]))
                    pool["sd"][d].append(float(sd_stop[k]))
                    pool["stot"][d].append(stot[d])
                pool["len"].append(length)
                pool["cap"].append(reason == "cap")
                pool["plat"].append(reason == "info_plateau")
            L = np.array(pool["len"], float)
            SDm = {d: np.array(pool["sd"][d]) for d in dims}
            STm = {d: np.array(pool["stot"][d]) for d in dims}
            row = {"floor": floor, "se_target": se_t, "n_models_excl2": len(L),
                   "median_len": float(np.median(L)), "mean_len": float(L.mean()),
                   "pct_cap": float(np.mean(pool["cap"])),
                   "pct_info_plateau": float(np.mean(pool["plat"])),
                   "pct_reach_se_ability_all": float(np.mean(
                       np.all(np.stack([SDm[d] for d in dims], 1) <= se_t, axis=1)))}
            for d in dims:
                x = np.array(pool["x"][d]); y = np.array(pool["y"][d])
                row[f"r_{d}"] = float(np.corrcoef(x, y)[0, 1])
                row[f"slope_{d}"] = float(np.polyfit(x, y, 1)[0])
                row[f"theta_mae_{d}"] = float(np.mean(np.abs(y - x)))
                row[f"median_sd_{d}"] = float(np.median(SDm[d]))
                row[f"median_se_total_{d}"] = float(np.median(STm[d]))
                row[f"pct_reach_se_ability_{d}"] = float(np.mean(SDm[d] <= se_t))
                for thr in SE_TOTAL_THRESHOLDS:
                    row[f"pct_se_total_le_{thr}_{d}"] = float(np.mean(STm[d] <= thr))
            # all-skills SE_total reach at each threshold
            for thr in SE_TOTAL_THRESHOLDS:
                allok = np.all(np.stack([STm[d] for d in dims], 1) <= thr, axis=1)
                row[f"pct_reach_se_total_le_{thr}_all"] = float(np.mean(allok))
            per_cell_rows.append(row)

    # ---- write artifacts ----
    with (args.out_dir / "oos_per_cell_grid.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_cell_rows[0].keys()))
        w.writeheader(); w.writerows(per_cell_rows)
    with (args.out_dir / "oos_per_model_per_cell.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_model_cell_rows[0].keys()))
        w.writeheader(); w.writerows(per_model_cell_rows)

    # weak-model fate: per-cell length/stop_reason/sd/se_total for the 2 excluded models
    weak_rows = [r for r in per_model_cell_rows if r["is_weak_excluded"]]
    with (args.out_dir / "weak_models_per_cell.csv").open("w", newline="", encoding="utf-8") as fh:
        if weak_rows:
            w = csv.DictWriter(fh, fieldnames=list(weak_rows[0].keys()))
            w.writeheader(); w.writerows(weak_rows)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "STUDY ONLY - production engine untouched; nothing committed.",
        "benchmark": "TutorBench", "dims": dims, "protocol": "OOS k-fold refit-per-fold, offline 35-cell replay",
        "model_set": "full-115", "n_models": len(models),
        "matrix": str(args.matrix.relative_to(ROOT)), "bank": str(args.bank.relative_to(ROOT)),
        "k": args.k, "seed": args.seed, "fold_sizes": [len(f) for f in folds],
        "fit_config": {"fitter": "calibrate_mirt.fit_m2pl_em (confirmatory M2PL, Bock-Aitkin EM)",
                       "fit_grid_nodes_per_dim": args.fit_grid, "ridge": args.ridge,
                       "max_iter": args.max_iter, "negative_policy": args.negative_policy,
                       "within_fold_zero_variance_filter": True,
                       "q_source": "frozen bank q_modeled (confirmatory)"},
        "stop_rule": {"metric": "EAP-posterior honest per-skill MARGINAL SD (marginal of joint 2-D posterior)",
                      "stop_nodes_per_dim": args.stop_nodes, "grid_joint_nodes": int(n_joint),
                      "plateau_delta": PLATEAU_DELTA, "plateau_W": PLATEAU_W, "cap": args.cap,
                      "priority": "precision -> info_plateau -> cap -> bank_exhausted",
                      "estimator_at_stop": "MWLE (scenario_cat_lib.mwle_subset), start=dense-grid EAP mean"},
        "reference": "full-bank dense-grid EAP theta on fold-refit clamped params (all observed items)",
        "se_param_source": str(args.se_components.relative_to(ROOT)),
        "se_param_note": "115-min12 per-model SE_param used as a FIXED offset (NOT per-fold)",
        "se_param_median": med_param, "n_se_param_fallback": n_param_fallback,
        "se_total_thresholds": SE_TOTAL_THRESHOLDS,
        "excluded_weak_models": list(WEAK_MODELS),
        "floors": FLOORS, "se_targets": SE_TARGETS,
        "bank_stats": stats, "fold_diagnostics": fold_diag,
        "n_held_out_traces": len(all_traces),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _heatmaps(per_cell_rows, args.out_dir / "figures" / "oos_grid_heatmaps.png")
    _scatter_at(per_model_cell_rows, dims, 12, 0.27,
                args.out_dir / "figures" / "oos_recovery_scatter_f12se27.png")
    print(f"[write] {args.out_dir}")
    # console: quick view of the classic 0.27 column at each floor
    print("\nfloor  SE    len(med)  r_corr  r_scaff  medSEtot_c  %reach_SEa  %plat  %cap")
    for r in per_cell_rows:
        if r["se_target"] in (0.25, 0.27, 0.30):
            print(f"  {r['floor']:<4} {r['se_target']:<5} {r['median_len']:<8.0f} "
                  f"{r['r_correctness']:<7.3f} {r['r_scaffolding']:<8.3f} "
                  f"{r['median_se_total_correctness']:<11.3f} "
                  f"{r['pct_reach_se_ability_all']*100:<11.0f} "
                  f"{r['pct_info_plateau']*100:<6.0f} {r['pct_cap']*100:<4.0f}")
    return 0


def _heatmaps(rows, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    floors = sorted({r["floor"] for r in rows})
    ses = sorted({r["se_target"] for r in rows})
    by = {(r["floor"], r["se_target"]): r for r in rows}

    def grid_of(key):
        return np.array([[by[(fl, se)][key] for se in ses] for fl in floors], float)

    panels = [("r_correctness", "OOS recovery r (correctness)", "viridis", "{:.3f}"),
              ("median_se_total_correctness", "median SE_total (correctness)", "magma_r", "{:.3f}"),
              ("median_len", "median length (scenarios)", "cividis", "{:.0f}"),
              ("pct_reach_se_ability_all", "%reach SE_ability<=target (both skills)", "YlGnBu", "{:.0%}")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (key, title, cmap, fmt) in zip(axes.ravel(), panels):
        M = grid_of(key)
        im = ax.imshow(M, aspect="auto", cmap=cmap, origin="lower")
        ax.set_xticks(range(len(ses))); ax.set_xticklabels(ses)
        ax.set_yticks(range(len(floors))); ax.set_yticklabels(floors)
        ax.set_xlabel("SE_ability target"); ax.set_ylabel("min_scenarios floor")
        ax.set_title(title, fontsize=10)
        for ii in range(len(floors)):
            for jj in range(len(ses)):
                ax.text(jj, ii, fmt.format(M[ii, jj]), ha="center", va="center",
                        fontsize=7, color="w" if cmap in ("viridis", "cividis") else "k")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("TutorBench 2-skill EAP-posterior stop: OUT-OF-SAMPLE floor x SE grid (N=113 excl-2, k=5)",
                 fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


def _scatter_at(per_model_rows, dims, floor, se_t, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in per_model_rows if r["floor"] == floor and r["se_target"] == se_t]
    fig, axes = plt.subplots(1, len(dims), figsize=(5.2 * len(dims), 5))
    for ax, d in zip(np.atleast_1d(axes), dims):
        xh = np.array([r[f"theta_ref_{d}"] for r in rows if not r["is_weak_excluded"]])
        yh = np.array([r[f"theta_mwle_{d}"] for r in rows if not r["is_weak_excluded"]])
        xw = np.array([r[f"theta_ref_{d}"] for r in rows if r["is_weak_excluded"]])
        yw = np.array([r[f"theta_mwle_{d}"] for r in rows if r["is_weak_excluded"]])
        r = float(np.corrcoef(xh, yh)[0, 1]); s, c = np.polyfit(xh, yh, 1)
        mae = float(np.mean(np.abs(yh - xh)))
        lo = min(xh.min(), yh.min()) - 0.3; hi = max(xh.max(), yh.max()) + 0.3
        ax.plot([lo, hi], [lo, hi], ls="--", color="gray", lw=1)
        xs = np.linspace(lo, hi, 40); ax.plot(xs, s * xs + c, color="#c1666b", lw=1.5,
                                              label=f"OLS slope={s:.3f}")
        ax.scatter(xh, yh, s=22, alpha=0.7, edgecolor="k", linewidth=0.2, color="#4d648d",
                   label=f"models (r={r:.3f}, MAE={mae:.3f})")
        if xw.size:
            ax.scatter(xw, yw, s=70, marker="X", color="red", edgecolor="k",
                       label="weak (excluded)", zorder=5)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel(f"reference full-bank EAP theta ({d})")
        ax.set_ylabel(f"CAT MWLE theta at stop ({d})")
        ax.set_title(f"{d}: OOS recovery @ floor={floor}, SE={se_t}", fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("OOS recovery: reference vs MWLE-at-stop (fine-grid ref, k=5 pooled)", fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
