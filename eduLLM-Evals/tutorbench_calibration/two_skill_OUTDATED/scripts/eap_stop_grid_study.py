"""EAP-posterior stop-rule OP-POINT GRID study (TutorBench, 2-skill).

STUDY-ONLY re-do of the ``eap_stop_prototype`` op-point derivation, fixing three review
findings (see the report README for the full write-up):

  FIX 1 (ability estimator).  Ability at stop uses the DEPLOYED multidimensional **MWLE**
         (``scenario_cat_lib.mwle_subset``), started from the dense-grid EAP mean -- NOT the
         online Laplace theta the prototype reported.
  FIX 2 (integration grid).  The honest per-skill marginal posterior SD is integrated on a
         DENSE product grid (``--stop-nodes`` default 161/dim => spacing 0.075 over [-6,6],
         vs the prototype's 25/dim). ``dense_marginal_meanvar`` chunks over grid nodes so the
         full-bank boundary (thousands of criteria) stays in memory; it is verified equal to
         ``scenario_cat_lib.eap_subset_mean_var`` on a coarse grid.
  FIX 3 (op-point).  12/0.30 was inherited; here we sweep a 35-cell grid
         (floor x SE_ability target) and select against SE_total + length + %bank-limited.

Design (offline replay of ONE full-length trace / model, faithful to the prototype):
  * Drive the REAL engine (``scenario_cat_lib.run_one_model`` -> ``tutor_cat.engine``) once
    per model to FULL BANK EXHAUSTION (max_se=1e-9, min_scenarios=0, max_scenarios huge), so
    ``select_next`` returns None only when the bank is empty. Record the exact administered
    criterion order.
  * Because scenario selection is independent of the stop rule, EVERY grid cell + plateau +
    cap are just different truncation points along that one sequence. At each scenario
    boundary we compute the dense-grid per-skill marginal posterior SD and MWLE theta.

Stop priority at each boundary (past the floor):
  precision_reached (all skills' SD <= target) -> info_plateau (bank-limited) -> cap(70)
  -> bank_exhausted.

Info-plateau ("bank-limited"): past the floor, if for EVERY still-unconverged skill the
marginal SD dropped by < delta over the last W administered scenarios, stop (reason
"info_plateau"). Headline delta=0.005, W=3; sensitivity over delta in {0.003,0.005,0.010}.

SE_total_skill = sqrt(SD_at_stop_skill^2 + SE_param_skill^2); SE_param is the per-model
scenario-level parameter-uncertainty component from the 2-skill param_uncertainty study
(median fallback). This is the selection metric per the review.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import log_expit, logsumexp

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)
import scenario_cat_lib as scl  # noqa: E402

# ---- user-locked op-point grid -------------------------------------------------
FLOORS = [6, 8, 10, 12, 15]
SE_TARGETS = [0.15, 0.20, 0.22, 0.25, 0.27, 0.30, 0.32]
CAP = 70
PLATEAU_W = 3
PLATEAU_DELTAS = [0.003, 0.005, 0.010]
PLATEAU_HEADLINE = 0.005
SE_TOTAL_THRESHOLDS = [0.25, 0.30]


# ---------------------------------------------------------------------------
# dense marginal posterior SD (FIX 2) — chunked over grid nodes so memory stays
# bounded regardless of #administered criteria. Equal to scl.eap_subset_mean_var.
# ---------------------------------------------------------------------------
def dense_marginal_meanvar(yi, Ai, bi, grid, log_prior, chunk: int = 4096):
    """Per-skill marginal mean/var of the JOINT posterior over the administered set."""
    n_nodes = grid.shape[0]
    ll = np.empty(n_nodes, dtype=float)
    for s in range(0, n_nodes, chunk):
        eta = Ai @ grid[s:s + chunk].T - bi[:, None]
        ll[s:s + chunk] = yi @ log_expit(eta) + (1.0 - yi) @ log_expit(-eta)
    post = np.exp(ll + log_prior - logsumexp(ll + log_prior))
    mean = post @ grid
    var = post @ (grid**2) - mean**2
    return mean, np.clip(var, 0.0, None)


def boundaries_from_order(order, col, scen_of):
    """Return list of (nscen, idx_array) at each scenario boundary along the administered
    order. idx_array indexes the (ids, A, b) arrays for all criteria administered so far."""
    idx: list[int] = []
    out: list[tuple[int, np.ndarray]] = []
    nscen = 0
    kept = [c for c in order if c in col]
    for i, c in enumerate(kept):
        idx.append(col[c])
        last = i == len(kept) - 1
        boundary = last or (scen_of[kept[i + 1]] != scen_of[c])
        if boundary:
            nscen += 1
            out.append((nscen, np.array(idx, dtype=int)))
    return out


def resolve_stop(sd_series, nscen_series, floor, se_target, delta, w, cap):
    """Walk boundaries; return (stop_i, reason). sd_series[i] is per-skill SD at boundary i.
    Priority past the floor: precision -> info_plateau -> cap -> bank_exhausted."""
    n = len(nscen_series)
    for i in range(n):
        ns = nscen_series[i]
        if ns < floor:
            continue
        sd = sd_series[i]
        if bool(np.all(sd <= se_target)):
            return i, "precision_reached"
        # info-plateau: every still-unconverged skill dropped < delta over last W scenarios
        j = i - w
        if j >= 0:
            unconv = sd > se_target
            drop = sd_series[j] - sd
            if np.any(unconv) and bool(np.all(drop[unconv] < delta)):
                return i, "info_plateau"
        if ns >= cap:
            return i, "cap"
    return n - 1, "bank_exhausted"


def build_model_trace(rec, col, ids, A, b, Yrow, scen_of, grid, log_prior, cap):
    """Full offline evaluation for one model: dense SD + MWLE theta at every boundary up to
    min(cap, total), plus the asymptotic (full-bank) SD floor. Returns a dict."""
    order = rec["order"]
    bnds = boundaries_from_order(order, col, scen_of)
    if not bnds:
        return None
    total = bnds[-1][0]
    b_eval = min(cap, total)
    nscen_series: list[int] = []
    sd_series: list[np.ndarray] = []
    mwle_series: list[np.ndarray] = []
    for nscen, idx in bnds:
        if nscen > b_eval:
            break
        yi, Ai, bi = Yrow[idx], A[idx], b[idx]
        mean, var = dense_marginal_meanvar(yi, Ai, bi, grid, log_prior)
        theta_mwle, _ = scl.mwle_subset(Yrow, idx, A, b, mean)
        nscen_series.append(nscen)
        sd_series.append(np.sqrt(var))
        mwle_series.append(np.asarray(theta_mwle, float))
    # asymptotic SD floor at full bank exhaustion
    _, var_full = dense_marginal_meanvar(Yrow[bnds[-1][1]], A[bnds[-1][1]], b[bnds[-1][1]],
                                         grid, log_prior)
    return {
        "model": rec["model"],
        "total_scenarios": total,
        "nscen": nscen_series,
        "sd": sd_series,          # list of per-skill SD arrays
        "mwle": mwle_series,      # list of per-skill MWLE theta arrays
        "sd_floor": np.sqrt(var_full),
    }


# ---------------------------------------------------------------------------
# parallel per-model worker: run engine to exhaustion + offline dense SD/MWLE.
# Each model is independent; the worker builds the shared read-only state once.
# ---------------------------------------------------------------------------
_S: dict = {}


def _init_worker(bank_path, matrix_path, scenarios_path, negative_policy, stop_nodes,
                 cap, spec_dict):
    scl._limit_blas_threads()
    records, dims, _ = scl.load_fitted_bank(Path(bank_path), negative_policy)
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
    rec = scl.run_one_model(model, _S["rubrics"], _S["scen_raw"], _S["matrix"].loc[model],
                            _S["dims"], _S["spec"])
    Yrow = np.nan_to_num(_S["matrix"].loc[model].reindex(_S["ids"]).to_numpy(dtype=float))
    return build_model_trace(rec, _S["col"], _S["ids"], _S["A"], _S["b"], Yrow,
                             _S["scen_of"], _S["grid"], _S["log_prior"], _S["cap"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bank", type=Path,
                    default=ROOT / "data/TutorBench/rubrics_qmatrix_calibrated_2skill_fitted.jsonl")
    ap.add_argument("--matrix", type=Path,
                    default=ROOT / "tutorbench_tb33_grading/response_matrix/response_matrix.csv")
    ap.add_argument("--scenarios", type=Path, default=ROOT / "data/TutorBench/scenarios.jsonl")
    ap.add_argument("--se-components", type=Path,
                    default=ROOT / "regenerated_figures/scenario_level_115_min12/param_uncertainty/2_skills/leaderboard_se_components.csv")
    ap.add_argument("--out-dir", type=Path,
                    default=ROOT / "reports/eap_stop_grid_tutorbench_2skill")
    ap.add_argument("--stop-nodes", type=int, default=161, help="DENSE stop grid nodes/dim")
    ap.add_argument("--cap", type=int, default=CAP)
    ap.add_argument("--seed", type=int, default=20260729)
    ap.add_argument("--negative-policy", default="clamp")
    ap.add_argument("--workers", type=int, default=scl.default_workers())
    ap.add_argument("--limit", type=int, default=0, help="debug: first N models only")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Parent only needs bank stats, dims, the model list and per-model SE_param; the per-model
    # arrays/grid are (re)built inside each worker via _init_worker.
    records, dims, stats = scl.load_fitted_bank(args.bank, args.negative_policy)
    matrix = pd.read_csv(args.matrix, index_col=0)
    ids_all = scl.assemble_arrays(records, dims)[0]
    n_criteria = sum(1 for c in ids_all if c in matrix.columns)
    models = list(matrix.index)
    if args.limit:
        models = models[: args.limit]

    # per-model SE_param (FIX for SE_total selection metric)
    se_df = pd.read_csv(args.se_components, index_col=0)
    se_df = se_df[~se_df.index.duplicated(keep="first")]
    med_param = {d: float(np.median(se_df[f"se_param_{d}"])) for d in dims}
    se_param = {}
    n_param_median = 0
    for m in models:
        if m in se_df.index:
            se_param[m] = {d: float(se_df.loc[m, f"se_param_{d}"]) for d in dims}
        else:
            se_param[m] = dict(med_param)
            n_param_median += 1

    n_joint_nodes = args.stop_nodes ** len(dims)
    spacing = 12.0 / (args.stop_nodes - 1)
    print(f"[grid] {args.stop_nodes} nodes/dim over [-6,6] (spacing {spacing:.4f}); "
          f"{n_joint_nodes} joint nodes")
    print(f"[data] dims={dims} criteria(bank&matrix)={n_criteria} models={len(models)} "
          f"cap={args.cap}; SE_param median fallback used for {n_param_median} models")

    # --- run engine to FULL EXHAUSTION + offline dense SD/MWLE (parallel per model) ---
    from concurrent.futures import ProcessPoolExecutor
    from dataclasses import asdict

    spec = scl.RunSpec(seed=args.seed, max_se=1e-9, min_evals_per_skill=0, min_scenarios=0,
                       max_scenarios=1_000_000, selection="trace", mode="cat", write_logs=False)
    print(f"[run] full-exhaustion traces + dense SD/MWLE, workers={args.workers} ...")
    traces = []
    initargs = (str(args.bank), str(args.matrix), str(args.scenarios), args.negative_policy,
                args.stop_nodes, args.cap, asdict(spec))
    if args.workers and args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                                 initargs=initargs) as ex:
            for tr in ex.map(_run_model, models):
                if tr is not None:
                    traces.append(tr)
                    print(f"  {tr['model']:48} total_scen={tr['total_scenarios']:4d} "
                          f"floorSD=({tr['sd_floor'][0]:.3f},{tr['sd_floor'][1]:.3f})")
    else:
        _init_worker(*initargs)
        for m in models:
            tr = _run_model(m)
            if tr is not None:
                traces.append(tr)
                print(f"  {tr['model']:48} total_scen={tr['total_scenarios']:4d} "
                      f"floorSD=({tr['sd_floor'][0]:.3f},{tr['sd_floor'][1]:.3f})")

    # --- asymptotic SD floors summary ---
    floors_arr = np.array([tr["sd_floor"] for tr in traces])
    asym = {"per_model": {tr["model"]: {d: float(tr["sd_floor"][k]) for k, d in enumerate(dims)}
                          for tr in traces}}
    for k, d in enumerate(dims):
        col_k = floors_arr[:, k]
        asym[d] = {"min": float(col_k.min()), "median": float(np.median(col_k)),
                   "max": float(col_k.max()),
                   "n_above_0.30": int(np.sum(col_k > 0.30)),
                   "n_above_0.25": int(np.sum(col_k > 0.25))}

    # --- 35-cell grid (headline delta) + per-model-per-cell detail ---
    per_cell_rows = []
    per_model_cell_rows = []
    for floor in FLOORS:
        for se_t in SE_TARGETS:
            lengths, caphit, plat, reach = [], [], [], []
            setot = {d: [] for d in dims}
            for tr in traces:
                i, reason = resolve_stop(tr["sd"], tr["nscen"], floor, se_t,
                                         PLATEAU_HEADLINE, PLATEAU_W, args.cap)
                length = tr["nscen"][i]
                sd_stop = tr["sd"][i]
                mwle_stop = tr["mwle"][i]
                sp = se_param[tr["model"]]
                st = {d: float(np.sqrt(sd_stop[k] ** 2 + sp[d] ** 2))
                      for k, d in enumerate(dims)}
                lengths.append(length)
                caphit.append(reason == "cap")
                plat.append(reason == "info_plateau")
                reach.append(bool(np.all(sd_stop <= se_t)))
                for k, d in enumerate(dims):
                    setot[d].append(st[d])
                per_model_cell_rows.append({
                    "floor": floor, "se_target": se_t, "model": tr["model"],
                    "length": length, "stop_reason": reason,
                    **{f"sd_stop_{d}": float(sd_stop[k]) for k, d in enumerate(dims)},
                    **{f"theta_mwle_{d}": float(mwle_stop[k]) for k, d in enumerate(dims)},
                    **{f"se_total_{d}": st[d] for d in dims},
                })
            L = np.array(lengths, float)
            row = {"floor": floor, "se_target": se_t, "n_models": len(traces),
                   "median_len": float(np.median(L)), "mean_len": float(L.mean()),
                   "pct_cap_hit": float(np.mean(caphit)),
                   "pct_info_plateau": float(np.mean(plat)),
                   "pct_reach_all_skills": float(np.mean(reach))}
            for d in dims:
                arr = np.array(setot[d], float)
                row[f"median_se_total_{d}"] = float(np.median(arr))
                row[f"max_se_total_{d}"] = float(arr.max())
                for thr in SE_TOTAL_THRESHOLDS:
                    row[f"frac_se_total_le_{thr}_{d}"] = float(np.mean(arr <= thr))
            per_cell_rows.append(row)

    # --- plateau sensitivity: %plateau & median length averaged across cells, per delta ---
    plateau_sens = []
    for delta in PLATEAU_DELTAS:
        cell_plat, cell_len = [], []
        for floor in FLOORS:
            for se_t in SE_TARGETS:
                lengths, plat = [], []
                for tr in traces:
                    i, reason = resolve_stop(tr["sd"], tr["nscen"], floor, se_t,
                                             delta, PLATEAU_W, args.cap)
                    lengths.append(tr["nscen"][i])
                    plat.append(reason == "info_plateau")
                cell_plat.append(np.mean(plat))
                cell_len.append(np.median(lengths))
        plateau_sens.append({"delta": delta, "W": PLATEAU_W,
                             "mean_pct_info_plateau_across_cells": float(np.mean(cell_plat)),
                             "mean_median_len_across_cells": float(np.mean(cell_len))})

    # --- write artifacts ---
    with (args.out_dir / "per_cell_grid.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_cell_rows[0].keys()))
        w.writeheader()
        w.writerows(per_cell_rows)
    with (args.out_dir / "per_model_per_cell.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_model_cell_rows[0].keys()))
        w.writeheader()
        w.writerows(per_model_cell_rows)

    summary = {
        "benchmark": "TutorBench", "dims": dims, "n_models": len(traces),
        "stop_nodes_per_dim": args.stop_nodes, "grid_spacing": spacing,
        "grid_joint_nodes": int(n_joint_nodes), "cap": args.cap,
        "plateau": {"headline_delta": PLATEAU_HEADLINE, "W": PLATEAU_W,
                    "sensitivity": plateau_sens},
        "estimator_at_stop": "MWLE (scenario_cat_lib.mwle_subset), start=dense-grid EAP mean",
        "se_param_source": str(args.se_components.relative_to(ROOT)),
        "se_param_median": med_param, "n_se_param_median_fallback": n_param_median,
        "bank_stats": stats, "floors": FLOORS, "se_targets": SE_TARGETS,
        "asymptotic_sd_floor": asym,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    # asymptotic floors as its own small CSV
    with (args.out_dir / "asymptotic_sd_floor.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model"] + [f"sd_floor_{d}" for d in dims] + ["total_scenarios"])
        for tr in traces:
            w.writerow([tr["model"]] + [f"{tr['sd_floor'][k]:.6f}" for k in range(len(dims))]
                       + [tr["total_scenarios"]])
    print(f"[write] {args.out_dir}")
    print("[asym floor] correctness median "
          f"{asym['correctness']['median']:.3f} (max {asym['correctness']['max']:.3f}, "
          f"{asym['correctness']['n_above_0.30']} models >0.30); scaffolding median "
          f"{asym['scaffolding']['median']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
