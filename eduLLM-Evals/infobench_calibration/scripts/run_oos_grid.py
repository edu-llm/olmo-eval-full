"""LOCAL STUDY (read-only w.r.t. production): InfoBench floor x SE out-of-sample
operating-point grid on the canonical unidimensional ``instruction_following`` 2PL scale.

Mirrors the validated WildBench OOS design
(git 421e421:eduLLM-Evals/wildbench_calibration/scripts/prototype_eap_stop.py):
k=5 model-fold cross-validation with a per-fold refit of the unidim bank, the
frozen fold bank, a full-bank EAP reference theta on fold params for held-out
models (the section 8.3 held-out recovery), the production scenario/testlet CAT
engine for adaptive administration order, a dense 801-node normal-trapezoid
EAP posterior-SD stop with a plateau guard, and MWLE theta at the stop subset.

Reach is decided on SE_posterior (EAP posterior SD) <= target ONLY. SE_total is
reported separately as sqrt(SD^2 + SE_param^2) where SE_param is the SOUND
observed-information parametric bootstrap (reused, full-data), NOT the branch's
stale nonparametric person-resample-refit bootstrap.

This script does not modify the production engine, does not call any tutor/judge
model, and writes only into reports/infobench_oos_grid_unidim/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import log_expit, logsumexp

HERE = Path(__file__).resolve().parent
EE = HERE.parents[1]  # eduLLM-Evals
sys.path.insert(0, str(EE))
import scripts.calibrate_mirt as cm  # noqa: E402
import scripts.scenario_cat_lib as scat  # noqa: E402

# ----------------------------------------------------------------------------- config
DIM = "instruction_following"
MATRIX_PATH = EE / "runs" / "calibration" / "InFoBench_full_20260804" / "inputs" / "response_matrix.csv"
SCEN_PATH = EE / "data" / "InFoBench" / "scenarios.jsonl"

# of-record fit (configs/infobench_calibration_cat_2pl_only_v1.json, spec log_shrinkage_2pl_lambda16)
FIT_MODEL = cm.LOG_SHRINKAGE_2PL
LOG_A_SHRINK = 16.0            # lambda16 (log_a_prior_sd = 0.25 => 1/0.25^2 = 16)
FIT_GRID = 401                 # numerical_lock.fit_grid
FIT_QUAD = "normal_trapezoid"
FIT_BOUND = 8.0
FIT_MAX_ITER = 200
FIT_TOL = 1e-4
FIT_PARAM_TOL = 5e-5
FIT_PASSES = 2
EXPORT_MAX_A = 6.0             # export policy: drop nonpositive / extreme discriminations

# dense EAP stop grid (of-record eap_grid = 801 normal_trapezoid, bound 8)
EAP_NODES = 801
EAP_BOUND = 8.0

# sound SE_param bootstrap (recompute_se_param.py: observed-information parametric bootstrap)
SEP_FIT_NODES = 5             # GH nodes used by the validated recompute (free-2pl fit that
SEP_RIDGE = 1e-2              # reproduced the frozen matrix); SE_param is a separate uncertainty
SEP_EAP_NODES = 61           # component, computed on the full-data bank and reused across folds
SEP_EAP_RANGE = 6.0
SEP_B = 200                  # bootstrap draws
SEP_SEED = 0

K = 5
SEED = 20260729
CAP = 70                     # forced administration cap (scenarios); stated in outputs
TOP_N = 5
PLATEAU_DELTA = 0.005
PLATEAU_W = 3

FLOORS = [10, 12, 15, 20, 25]
TARGETS = [0.15, 0.20, 0.22, 0.25, 0.27, 0.30, 0.32]

WEAK_SUBSTR = ["smol_llama-220M", "mGPT", "OLMo-1B-hf"]


# ----------------------------------------------------------------------------- helpers
def variant_cols(Y, M):
    succ = np.where(M, Y, 0.0).sum(0)
    cnt = M.sum(0)
    return (cnt >= 2) & (succ > 0) & (succ < cnt)


def make_folds(models, k, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(models))
    folds = [[] for _ in range(k)]
    for pos, i in enumerate(idx):
        folds[pos % k].append(models[int(i)])
    return [sorted(f) for f in folds]


def eap_walk(order_ids, col, y_local, scen_of, a, b, gg, lp):
    """Per-scenario cumulative EAP (mean, SD) along an administration order.

    order_ids: administered criterion_ids in engine order.
    col: criterion_id -> local fold-bank index. y_local: fold-local response vector.
    Returns list of {n_scen, n_crit, eap_mean, eap_sd, idx(local cumulative indices)}.
    """
    G = gg.size
    ll = np.zeros(G)
    out = []
    cur = None
    n_scen = 0
    n_crit = 0
    cum = []
    for cid in order_ids:
        j = col.get(cid)
        if j is None:
            continue
        sid = scen_of.get(cid)
        if sid != cur:
            if cur is not None:
                post = np.exp(ll + lp - logsumexp(ll + lp))
                mean = float(post @ gg)
                sd = float(np.sqrt(max(post @ (gg ** 2) - mean ** 2, 0.0)))
                out.append({"n_scen": n_scen, "n_crit": n_crit, "eap_mean": mean,
                            "eap_sd": sd, "idx": np.array(cum, dtype=int)})
            cur = sid
            n_scen += 1
        eta = a[j] * gg - b[j]
        ll = ll + y_local[j] * log_expit(eta) + (1.0 - y_local[j]) * log_expit(-eta)
        n_crit += 1
        cum.append(j)
    if cur is not None:
        post = np.exp(ll + lp - logsumexp(ll + lp))
        mean = float(post @ gg)
        sd = float(np.sqrt(max(post @ (gg ** 2) - mean ** 2, 0.0)))
        out.append({"n_scen": n_scen, "n_crit": n_crit, "eap_mean": mean,
                    "eap_sd": sd, "idx": np.array(cum, dtype=int)})
    return out


def stop_point(walk, floor, target):
    """First scenario step with n_scen>=floor AND (SD<=target -> reach) else plateau; else cap.

    Returns (step_dict, reason) with reason in {reach, plateau, cap}.
    Plateau: post-floor window of the last PLATEAU_W SDs with range < PLATEAU_DELTA.
    """
    post_floor = [s for s in walk if s["n_scen"] >= floor]
    for i, step in enumerate(post_floor):
        if step["eap_sd"] <= target:
            return step, "reach"
        if i + 1 >= PLATEAU_W:
            window = [post_floor[k]["eap_sd"] for k in range(i - PLATEAU_W + 1, i + 1)]
            if (max(window) - min(window)) < PLATEAU_DELTA:
                return step, "plateau"
    return (walk[-1], "cap") if walk else (None, "cap")


# ----------------------------- per-fold refit worker (parallel) -----------------------------
_G = {}


def _init_fit(Yall, Mall, ids):
    _G["Y"] = Yall
    _G["M"] = Mall
    _G["ids"] = ids


def _fit_fold(payload):
    fold_id, test_idx = payload
    Yall, Mall, ids = _G["Y"], _G["M"], _G["ids"]
    P = Yall.shape[0]
    train = np.array([i for i in range(P) if i not in set(test_idx)], dtype=int)
    Ytr, Mtr = Yall[train], Mall[train]
    keep = variant_cols(Ytr, Mtr)
    src = np.where(keep)[0]
    Q = np.ones((int(keep.sum()), 1), dtype=int)
    fit = cm.fit_m2pl_em(
        Ytr[:, keep], Mtr[:, keep], Q, FIT_GRID,
        estimate_corr=False, ridge=0.0, max_iter=FIT_MAX_ITER, tol=FIT_TOL,
        calibration_model=FIT_MODEL, log_a_shrinkage=LOG_A_SHRINK,
        quadrature_method=FIT_QUAD, linear_bound=FIT_BOUND,
        convergence_mode="returned_iterate", parameter_tol=FIT_PARAM_TOL,
        consecutive_convergence_passes=FIT_PASSES,
    )
    a = np.asarray(fit["A"], float).ravel()
    b = np.asarray(fit["b"], float).ravel()
    exp_keep = (a > 0) & (a <= EXPORT_MAX_A)
    kept_cols = src[exp_keep]  # column indices into the full 2250-criterion matrix
    return {
        "fold_id": fold_id,
        "a": a[exp_keep].tolist(),
        "b": b[exp_keep].tolist(),
        "kept_cols": kept_cols.tolist(),
        "converged": bool(fit["converged"]),
        "n_iter": int(fit["n_iter"]),
        "n_kept": int(exp_keep.sum()),
    }


# ----------------------------- sound SE_param (full-data, observed information) --------------
def item_cov_1d_clean(Yv, Mv, a, b, fit_nodes, ridge):
    grid = cm.build_grid(1, fit_nodes)
    base = cm.base_log_weights(1, fit_nodes)
    log_prior = cm.prior_log_weights(grid, base, np.eye(1))
    g = grid[:, 0]
    YM = np.where(Mv, Yv, 0.0)
    NM = np.where(Mv, 1.0 - Yv, 0.0)
    Mf = Mv.astype(float)
    eta = a[:, None] * g[None, :] - b[:, None]
    lp = log_expit(eta)
    lq = log_expit(-eta)
    ll = YM @ lp + NM @ lq
    joint = ll + log_prior[None, :]
    post = np.exp(joint - logsumexp(joint, axis=1)[:, None])
    r_jg = YM.T @ post
    N_jg = Mf.T @ post
    ones = np.ones((g.size, 1))
    X = np.hstack([g[:, None], ones])
    chol = []
    for j in range(a.size):
        beta = np.array([a[j], -b[j]])
        _, _, H = cm._item_neg_loglik(beta, X, r_jg[j], N_jg[j], ridge)
        try:
            cov = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            cov = np.linalg.pinv(H)
        cov = (cov + cov.T) / 2.0
        w, V = np.linalg.eigh(cov)
        cov = (V * np.clip(w, 0.0, None)) @ V.T
        try:
            L = np.linalg.cholesky(cov + 1e-10 * np.eye(2))
        except np.linalg.LinAlgError:
            L = np.zeros((2, 2))
        chol.append(L)
    return chol


class SEParam:
    """Sound observed-information parametric bootstrap SE_param on the full-data bank.

    Perturb administered item params via the inverse observed information at the frozen
    fit, re-EAP, take the SD across draws. No person resampling, no refit.
    """

    def __init__(self, matrix_df):
        Yall = np.nan_to_num(matrix_df.to_numpy(float), nan=0.0)
        Mall = np.isfinite(matrix_df.to_numpy(float))
        self.models = list(matrix_df.index)
        self.row_of = {m: i for i, m in enumerate(self.models)}
        keep = variant_cols(Yall, Mall)
        Yv, Mv = Yall[:, keep], Mall[:, keep]
        col_ids = [matrix_df.columns[j] for j in np.where(keep)[0]]
        Q = np.ones((int(keep.sum()), 1), dtype=int)
        fit = cm.fit_m2pl_em(Yv, Mv, Q, SEP_FIT_NODES, estimate_corr=False,
                             ridge=SEP_RIDGE, max_iter=200, tol=1e-4)
        a = np.asarray(fit["A"], float).ravel()
        b = np.asarray(fit["b"], float).ravel()
        exp_keep = (a > 0) & (a <= EXPORT_MAX_A)
        self.a = a[exp_keep]
        self.b = b[exp_keep]
        self.Yv = Yv[:, exp_keep]
        self.Mv = Mv[:, exp_keep]
        kept_ids = [col_ids[j] for j in np.where(exp_keep)[0]]
        self.col = {c: i for i, c in enumerate(kept_ids)}
        self.chol = item_cov_1d_clean(self.Yv, self.Mv, self.a, self.b, SEP_FIT_NODES, SEP_RIDGE)
        self.chol = np.array(self.chol)  # (n,2,2)
        self.grid = np.linspace(-SEP_EAP_RANGE, SEP_EAP_RANGE, SEP_EAP_NODES)
        lp = -0.5 * self.grid ** 2
        self.lp = lp - logsumexp(lp)
        self.rng = np.random.default_rng(SEP_SEED)
        self.n_items = self.a.size
        self.converged = bool(fit["converged"])
        self.n_iter = int(fit["n_iter"])

    def _boot(self, model, ids, B=SEP_B):
        r = self.row_of[model]
        idx = np.array([self.col[c] for c in ids if c in self.col], dtype=int)
        if idx.size == 0:
            return float("nan"), 0
        y = self.Yv[r, idx]
        a0 = self.a[idx]
        b0 = self.b[idx]
        L = self.chol[idx]  # (k,2,2)
        g = self.grid
        z = self.rng.standard_normal((B, idx.size, 2))
        # draw[...,0] = a + L@z (component a), draw[...,1] = -b + ... ; beta=[a,-b]
        draw = np.einsum("kij,bkj->bki", L, z)
        a_d = a0[None, :] + draw[:, :, 0]
        negb_d = (-b0)[None, :] + draw[:, :, 1]
        # eta: (B,k,G)
        eta = a_d[:, :, None] * g[None, None, :] + negb_d[:, :, None]
        ll = np.einsum("k,bkg->bg", y, log_expit(eta)) + \
            np.einsum("k,bkg->bg", (1.0 - y), log_expit(-eta))
        joint = ll + self.lp[None, :]
        post = np.exp(joint - logsumexp(joint, axis=1)[:, None])
        mean = post @ g
        return float(np.std(mean, ddof=1)), int(idx.size)

    def full_admin(self, model):
        r = self.row_of[model]
        ids = [c for c, j in self.col.items() if self.Mv[r, j]]
        return self._boot(model, ids)[0]


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--cap", type=int, default=CAP)
    args = ap.parse_args()
    t0 = time.time()

    matrix = pd.read_csv(MATRIX_PATH, index_col=0).apply(pd.to_numeric, errors="coerce")
    ids = list(matrix.columns)
    models = list(matrix.index)
    P = len(models)
    Yall = np.nan_to_num(matrix.to_numpy(float), nan=0.0)
    Mall = np.isfinite(matrix.to_numpy(float))
    scen_of_all = {c: c.rsplit("_c", 1)[0] for c in ids}
    print(f"[data] {P} models x {len(ids)} criteria; fill={Mall.mean():.4f}", flush=True)

    folds = make_folds(models, K, SEED)
    fold_of = {m: f for f, fl in enumerate(folds) for m in fl}
    test_idx_by_fold = [[models.index(m) for m in fl] for fl in folds]

    # ---- parallel per-fold refits (the only expensive step) ----
    print(f"[fit] refitting {K} folds (log-shrinkage-2pl lambda16, {FIT_GRID}-node "
          f"normal_trapezoid) with {args.workers} workers ...", flush=True)
    payloads = [(f, test_idx_by_fold[f]) for f in range(K)]
    fold_fits = {}
    with ProcessPoolExecutor(max_workers=min(args.workers, K),
                             initializer=_init_fit,
                             initargs=(Yall, Mall, ids)) as ex:
        for res in ex.map(_fit_fold, payloads):
            fold_fits[res["fold_id"]] = res
            print(f"  fold {res['fold_id']}: kept={res['n_kept']} "
                  f"converged={res['converged']} n_iter={res['n_iter']}", flush=True)

    # ---- sound SE_param (full-data observed-information bootstrap; reused across folds) ----
    print("[se_param] fitting full-data bank and item covariances (sound method) ...", flush=True)
    sep = SEParam(matrix)
    full_admin_sep = {m: sep.full_admin(m) for m in models}
    fa = np.array(list(full_admin_sep.values()))
    print(f"  SE_param full-admin floor: median={np.nanmedian(fa):.4f} "
          f"mean={np.nanmean(fa):.4f} (n_items={sep.n_items}, "
          f"free-2pl converged={sep.converged})", flush=True)

    scen_records = scat.load_scenario_records(SCEN_PATH)
    quad = scat.build_quadrature(1, EAP_NODES, np.eye(1), max_nodes=2000,
                                 method="normal_trapezoid", linear_bound=EAP_BOUND)
    gg = quad.grid[:, 0]
    lp = quad.log_prior

    # ---- per-fold: build fold bank, engine forced-long, reference theta, per-model walk ----
    per_model = {}  # model -> dict(walk, theta_ref, order_ids, a_local, b_local, col, y_local, resp_vec)
    for f in range(K):
        ff = fold_fits[f]
        kept_cols = np.array(ff["kept_cols"], dtype=int)
        kept_ids = [ids[c] for c in kept_cols]
        a_local = np.array(ff["a"], dtype=float)
        b_local = np.array(ff["b"], dtype=float)
        col = {c: i for i, c in enumerate(kept_ids)}
        scen_of = {c: scen_of_all[c] for c in kept_ids}
        A_local = a_local[:, None]
        records = [
            {"criterion_id": cid, "scenario_id": scen_of[cid], "criterion": cid,
             "primary_skill": "", "irt_params": {"source": "calibrated-m2pl", "calibrated": True}}
            for cid in kept_ids
        ]
        bank = scat.FittedBank(
            records=records, dims=(DIM,), criterion_ids=tuple(kept_ids),
            scenario_ids=tuple(scen_of[c] for c in kept_ids),
            Q=np.ones((len(kept_ids), 1), int), A=A_local, b=b_local,
            latent_correlation=np.eye(1), source_path=f"fold{f}",
        )
        spec = scat.RunSpec(seed=SEED, top_n=TOP_N, max_se=0.0, min_evals_per_skill=0,
                            min_scenarios=0, max_scenarios=args.cap, selection="trace",
                            mode="cat", stop_se_method="online")
        for m in folds[f]:
            row = matrix.loc[m]
            resp_vec = pd.to_numeric(row.reindex(kept_ids), errors="coerce").to_numpy(float)
            y_local = np.nan_to_num(resp_vec, nan=0.0)
            # reference theta: full fold-bank EAP on fold params (section 8.3 held-out recovery)
            ref = scat.batch_eap(resp_vec, A_local, b_local, quad)
            res = scat.run_recorded_model(m, row, bank, scen_records, quad, spec, mwle_ridge=1e-6)
            order_ids = list(res["criterion_order"])
            walk = eap_walk(order_ids, col, y_local, scen_of, a_local, b_local, gg, lp)
            per_model[m] = {
                "fold": f, "walk": walk, "theta_ref": float(ref.theta[0]),
                "order_ids": order_ids, "col": col, "A_local": A_local,
                "b_local": b_local, "resp_vec": resp_vec,
                "kept_ids": kept_ids,
            }
        print(f"[engine] fold {f}: administered order for {len(folds[f])} held-out models", flush=True)

    # ---- grid loop ----
    weak_models = [m for m in models if any(s in m for s in WEAK_SUBSTR)]
    print(f"[grid] weak-identified audit set: {weak_models}", flush=True)
    sep_cache = {}  # (model, n_scen) -> SE_param

    per_model_rows = []
    cell_rows = []
    for floor in FLOORS:
        for target in TARGETS:
            recs = []
            for m in models:
                pm = per_model[m]
                walk = pm["walk"]
                step, reason = stop_point(walk, floor, target)
                if step is None:
                    continue
                idx = step["idx"]
                n_scen = step["n_scen"]
                n_crit = step["n_crit"]
                sd = step["eap_sd"]
                # MWLE theta at stop subset (fold params)
                mw = scat.mwle(pm["resp_vec"], pm["A_local"], pm["b_local"], idx,
                               theta0=np.array([step["eap_mean"]]), ridge=1e-6)
                theta_cat = float(mw.theta[0]) if mw.converged else float(step["eap_mean"])
                # sound SE_param at administered set (cache by model,n_scen)
                key = (m, n_scen)
                if key not in sep_cache:
                    admin_ids = [pm["kept_ids"][j] for j in idx]
                    sep_cache[key] = sep._boot(m, admin_ids)[0]
                se_param = sep_cache[key]
                se_total = float(np.sqrt(sd ** 2 + se_param ** 2)) if np.isfinite(se_param) else float("nan")
                rec = {
                    "floor": floor, "target": target, "model": m, "fold": pm["fold"],
                    "n_scen": n_scen, "n_crit": n_crit, "reason": reason,
                    "reach_se_post": bool(reason == "reach"),
                    "hit_cap": bool(reason == "cap"), "plateau": bool(reason == "plateau"),
                    "eap_sd": sd, "se_param": se_param, "se_total": se_total,
                    "theta_ref": pm["theta_ref"], "theta_cat": theta_cat,
                    "mwle_converged": bool(mw.converged),
                    "weak": bool(m in weak_models),
                }
                recs.append(rec)
                per_model_rows.append(rec)
            df = pd.DataFrame(recs)
            cell_rows.append(_cell_summary(df, floor, target, subset="all"))
            cell_rows.append(_cell_summary(df[~df["weak"]], floor, target, subset="excl_weak"))
        print(f"[grid] floor={floor} done", flush=True)

    per_model_df = pd.DataFrame(per_model_rows)
    cell_df = pd.DataFrame(cell_rows)

    # ---- write outputs ----
    per_model_df.to_csv(HERE / "oos_per_model_per_cell.csv", index=False)
    cell_all = cell_df[cell_df["subset"] == "all"].drop(columns=["subset"])
    cell_excl = cell_df[cell_df["subset"] == "excl_weak"].drop(columns=["subset"])
    cell_all.to_csv(HERE / "oos_per_cell_grid.csv", index=False)
    cell_excl.to_csv(HERE / "oos_per_cell_grid_excl_weak.csv", index=False)

    fold_conv = {f: {"converged": fold_fits[f]["converged"], "n_iter": fold_fits[f]["n_iter"],
                     "n_kept": fold_fits[f]["n_kept"]} for f in range(K)}
    cat_admin_sep = float(np.nanmedian(per_model_df["se_param"]))
    summary = {
        "study": "InfoBench floor x SE OOS operating-point grid (unidim instruction_following)",
        "local_only": True, "op_point_locked": False,
        "data": {
            "matrix": str(MATRIX_PATH.relative_to(EE)),
            "matrix_sha256_prefix": "087948fcaa884cde",
            "n_models": P, "n_criteria": len(ids), "n_scenarios": len(scen_records),
            "fill": float(Mall.mean()),
        },
        "fit_config": {
            "family": "log-shrinkage-2pl", "log_a_shrinkage": LOG_A_SHRINK,
            "log_a_prior_sd": 0.25, "fit_grid": FIT_GRID, "fit_quadrature": FIT_QUAD,
            "linear_bound": FIT_BOUND, "convergence_mode": "returned_iterate",
            "max_iter": FIT_MAX_ITER, "parameter_tol": FIT_PARAM_TOL,
            "consecutive_passes": FIT_PASSES, "export_max_a": EXPORT_MAX_A,
            "spec_id": "log_shrinkage_2pl_lambda16",
            "fold_convergence": fold_conv,
        },
        "eap_stop": {"nodes": EAP_NODES, "method": "normal_trapezoid", "bound": EAP_BOUND,
                     "plateau_delta": PLATEAU_DELTA, "plateau_w": PLATEAU_W, "cap": args.cap,
                     "reach_rule": "SE_posterior (EAP posterior SD) <= target ONLY"},
        "oos_design": {"k": K, "seed": SEED, "per_fold_refit": True,
                       "reference": "full fold-bank EAP on fold params (section 8.3 held-out recovery)",
                       "final_estimator": "MWLE at stop", "top_n": TOP_N,
                       "selection": "trace", "testlet_level": True},
        "se_param_sound": {
            "method": "observed-information parametric bootstrap (recompute_se_param.py)",
            "note": "reused full-data bootstrap (mirrors WildBench reuse of deployed bootstrap); "
                    "NOT the stale nonparametric person-resample-refit (~0.19).",
            "fit_nodes": SEP_FIT_NODES, "ridge": SEP_RIDGE, "eap_nodes": SEP_EAP_NODES,
            "bootstrap_draws": SEP_B, "n_items": sep.n_items,
            "full_admin_floor_median": float(np.nanmedian(fa)),
            "full_admin_floor_mean": float(np.nanmean(fa)),
            "full_admin_floor_q1": float(np.nanpercentile(fa, 25)),
            "full_admin_floor_q3": float(np.nanpercentile(fa, 75)),
            "cat_admin_median_over_all_cells": cat_admin_sep,
        },
        "grid": {"floors": FLOORS, "targets": TARGETS},
        "weak_identified": weak_models,
        "overall_oos_recovery_note": "see oos_per_cell_grid.csv (recovery_r/slope/theta_mae per cell)",
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _heatmaps(cell_all, HERE / "figures" / "oos_grid_heatmaps.png")
    _decision_table(cell_all, cell_excl, per_model_df, summary, HERE / "DECISION_TABLE.md")

    print(f"[done] wrote outputs to {HERE} in {time.time()-t0:.1f}s", flush=True)


def _cell_summary(df, floor, target, subset):
    if len(df) == 0:
        return {"subset": subset, "floor": floor, "target": target, "n_models": 0}
    x = df["theta_ref"].to_numpy(float)
    y = df["theta_cat"].to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) >= 3 and x.std() > 0 and y.std() > 0:
        r = float(np.corrcoef(x, y)[0, 1])
        slope = float(np.polyfit(x, y, 1)[0])
        mae = float(np.mean(np.abs(y - x)))
    else:
        r = slope = mae = float("nan")
    return {
        "subset": subset, "floor": floor, "target": target, "n_models": int(len(df)),
        "median_scen": float(df["n_scen"].median()), "mean_scen": float(df["n_scen"].mean()),
        "median_crit": float(df["n_crit"].median()),
        "pct_reach": float(100.0 * df["reach_se_post"].mean()),
        "pct_plateau": float(100.0 * df["plateau"].mean()),
        "pct_cap": float(100.0 * df["hit_cap"].mean()),
        "recovery_r": r, "recovery_slope": slope, "theta_mae": mae,
        "median_sd": float(df["eap_sd"].median()),
        "median_se_param": float(np.nanmedian(df["se_param"])),
        "median_se_total": float(np.nanmedian(df["se_total"])),
    }


def _heatmaps(cell_all, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("recovery_r", "OOS recovery r"), ("median_se_total", "median SE_total"),
              ("median_scen", "median length (scenarios)"), ("pct_reach", "%reach (SE_post<=target)")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, (col, title) in zip(axes.ravel(), panels):
        piv = cell_all.pivot(index="floor", columns="target", values=col)
        im = ax.imshow(piv.values, aspect="auto", origin="lower", cmap="viridis")
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels(piv.columns)
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels(piv.index)
        ax.set_xlabel("SE_ability target")
        ax.set_ylabel("floor (min scenarios)")
        ax.set_title(title)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}" if col != "median_scen" and col != "pct_reach"
                            else f"{v:.0f}", ha="center", va="center", color="w", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("InfoBench floor x SE OOS grid (unidim instruction_following, N=52)", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _decision_table(cell_all, cell_excl, per_model_df, summary, path):
    def fmt(v, d=3):
        return "NA" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{d}f}"

    sep = summary["se_param_sound"]
    lines = [
        "# InfoBench operating-point decision table (unidim `instruction_following`)",
        "",
        "LOCAL study. No operating point is locked here — this table is for the user to choose from.",
        "",
        "## Configuration",
        "",
        f"- Bank / matrix: `{summary['data']['matrix']}` (sha256 `087948fcaa884cde…`), "
        f"N={summary['data']['n_models']} models x {summary['data']['n_criteria']} criteria "
        f"x {summary['data']['n_scenarios']} scenarios, fill={summary['data']['fill']:.4f}.",
        f"- Fit: log-shrinkage-2PL λ16 (prior_sd 0.25), {summary['fit_config']['fit_grid']}-node "
        f"normal_trapezoid, returned_iterate. Per-fold refit (k={summary['oos_design']['k']}, "
        f"seed {summary['oos_design']['seed']}).",
        f"- Stop: dense {summary['eap_stop']['nodes']}-node normal_trapezoid EAP posterior-SD, "
        f"reach = **SE_post ≤ target ONLY**; plateau δ={summary['eap_stop']['plateau_delta']}/"
        f"W={summary['eap_stop']['plateau_w']}; cap={summary['eap_stop']['cap']}. MWLE θ at stop.",
        f"- Reference θ = full fold-bank EAP on fold params (§8.3 held-out recovery).",
        "",
        "## Sound SE_param used for SE_total",
        "",
        f"- Method: {sep['method']} (reused full-data; **not** the stale nonparametric ~0.19).",
        f"- **Full-bank admin floor**: median **{fmt(sep['full_admin_floor_median'],4)}** "
        f"(mean {fmt(sep['full_admin_floor_mean'],4)}, "
        f"IQR [{fmt(sep['full_admin_floor_q1'],4)}, {fmt(sep['full_admin_floor_q3'],4)}]).",
        f"- **CAT-admin median across all grid cells**: **{fmt(sep['cat_admin_median_over_all_cells'],4)}** "
        f"(per-model, at each model's administered set). SE_total = √(SD² + SE_param²).",
        "",
        "## Per-cell grid (all 52 models)",
        "",
        "| floor | target | med len | %reach | med SD | med SE_param | med SE_total | r | slope | θMAE | %plateau | %cap |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in cell_all.sort_values(["floor", "target"]).iterrows():
        lines.append(
            f"| {int(r['floor'])} | {r['target']:.2f} | {r['median_scen']:.0f} | "
            f"{r['pct_reach']:.0f} | {fmt(r['median_sd'])} | {fmt(r['median_se_param'])} | "
            f"{fmt(r['median_se_total'])} | {fmt(r['recovery_r'])} | {fmt(r['recovery_slope'])} | "
            f"{fmt(r['theta_mae'])} | {r['pct_plateau']:.0f} | {r['pct_cap']:.0f} |"
        )
    lines += ["", "## Per-cell grid (excluding weakly-identified tail)", "",
              f"Weak set: {summary['weak_identified']}", "",
              "| floor | target | med len | %reach | med SD | med SE_total | r | slope | θMAE | %cap |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in cell_excl.sort_values(["floor", "target"]).iterrows():
        lines.append(
            f"| {int(r['floor'])} | {r['target']:.2f} | {r['median_scen']:.0f} | "
            f"{r['pct_reach']:.0f} | {fmt(r['median_sd'])} | {fmt(r['median_se_total'])} | "
            f"{fmt(r['recovery_r'])} | {fmt(r['recovery_slope'])} | {fmt(r['theta_mae'])} | "
            f"{r['pct_cap']:.0f} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
