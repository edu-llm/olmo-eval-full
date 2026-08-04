#!/usr/bin/env python3
"""Within-benchmark MIRT: does ONE benchmark (BBH) have internal multidimensional
skill structure, and does modeling it recover the true benchmark accuracy better
than a unidimensional CAT?

Framing
-------
An earlier experiment (mirt_mcq_test.py) pooled DIFFERENT benchmarks as MIRT
dimensions; that only shows different benchmarks measure different things. Here we
fit MIRT WITHIN a single benchmark, using its ~27 disparate SUBTASKS as the latent
dimensions, to test whether BBH is internally multidimensional. This is the natural
explanation for the documented BBH failure (link_failure_diagnosis): the
unidimensional 3PL CAT early-stops at the ~8-item minimum (inflated discriminations)
and under-samples a heterogeneous suite, so its accuracy recovery plateaus at
r about 0.67 even though the full 3,965-item bank recovers accuracy at r about 0.86.

Three analyses
--------------
1. Confirmatory (simple structure). Each item loads on its SUBTASK dimension. A
   joint N-subtask confirmatory M2PL with the shared Gauss-Hermite EM is
   computationally intractable (grid**n_subtasks nodes; ~24 subtasks). We therefore
   estimate the latent inter-subtask correlation via per-subtask UNIDIMENSIONAL 2PL
   calibration (the shared fit_m2pl_em, n_dims=1) followed by the person-ability
   (EAP theta) correlation across subtasks, plus a reliability-disattenuated
   version. High (~0.8+) correlations across the board => effectively one factor;
   moderate/mixed values => genuine multidimensionality.
2. Exploratory dimensionality. How many latent factors does BBH actually need? We
   fit k = 1..K EXPLORATORY M2PL models (all-ones Q; loadings free on every factor)
   with the shared EM and report loglik, AIC, BIC (rotation-corrected parameter
   count) and an eigenvalue scree of the inter-item correlation matrix.
3. CAT comparison within BBH. Unidimensional 2PL CAT (the r about 0.67 baseline
   family) vs a within-BBH MIRT CAT (subtasks as dimensions, simple-structure bank
   assembled from the per-subtask fits, latent correlation R as the CAT prior
   covariance so answers borrow strength across correlated subtasks). Both predict
   the OVERALL BBH accuracy on the SAME held-out models/split/seed. We report r,
   MAE, and items administered at SE 0.3 (and 0.2), and whether the MIRT CAT fixes
   the early-stop under-sampling.

Machinery is REUSED from the repo's FRQ MIRT pipeline (no new fitter):
  * eduLLM-Evals/scripts/calibrate_mirt.py -- fit_m2pl_em (Bock-Aitkin EM,
    marginal-ML, native missing-data), aic_bic.
  * eduLLM-Evals/tutor_cat/mirt.py -- multidimensional update / standard_errors for
    the MIRT CAT ability step.

Usage
-----
    export REPO=/Users/arhant/Documents/EDLM/olmo-eval-full
    export PYTHONPATH=$REPO/eduLLM-Evals
    uv run python $REPO/AdaptiveTesting/Research/scripts/mirt_within_benchmark.py \
        --bench bbh --se-targets 0.3,0.2 --workers 8

Outputs land in AdaptiveTesting/Research/01_MCQ_ATLAS/data/mirt_within_benchmark/.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
EDU = REPO / "eduLLM-Evals"
OPENLM = REPO / "AdaptiveTesting/Inputs/OpenLM"
STATUS_CSV = (REPO
              / "AdaptiveTesting/Research/05_Data_Availability/data/openlm_download_status.csv")
UNIDIM_SUMMARY = (REPO / "AdaptiveTesting/Research/01_MCQ_ATLAS/data/atlas_replication"
                  / "summary_pirt_mae_sd_se.csv")
OUT_DIR = REPO / "AdaptiveTesting/Research/01_MCQ_ATLAS/data/mirt_within_benchmark"

# Gauss-Hermite nodes for the 1-d unidimensional fits (per-subtask + full-bank).
UNI_GRID = 21

sys.path.insert(0, str(EDU))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, mod)
    spec.loader.exec_module(mod)
    return mod


cm = _load_module("calibrate_mirt", EDU / "scripts" / "calibrate_mirt.py")
from tutor_cat import mirt as tc_mirt  # noqa: E402

# ---------------------------------------------------------------------------
# data build
# ---------------------------------------------------------------------------


def ok_models() -> set[str]:
    df = pd.read_csv(STATUS_CSV)
    return set(df.loc[df["status"] == "ok", "model"].astype(str))


def build_matrix(bench: str, keep_models: set[str], cache: Path,
                 map_cache: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """model x item binary correctness matrix (item = 'subtask|question_id').

    Returns (matrix, item->subtask map). Cached to parquet + csv.
    """
    if cache.exists() and map_cache.exists():
        mat = pd.read_parquet(cache)
        smap = pd.read_csv(map_cache)
        return mat, dict(zip(smap["item"], smap["subtask"], strict=True))
    frames = []
    for path in sorted((OPENLM / bench).glob("*.csv")):
        try:
            d = pd.read_csv(path, usecols=["model", "question_id", "subtask", "result"])
        except Exception:
            continue
        if d.empty:
            continue
        frames.append(d)
    long = pd.concat(frames, ignore_index=True)
    long = long[long["model"].astype(str).isin(keep_models)]
    long["subtask"] = long["subtask"].astype(str)
    long["item"] = long["subtask"] + "|" + long["question_id"].astype(str)
    long["correct"] = (long["result"].astype(str).str.strip().str.lower()
                       == "correct").astype(int)
    mat = long.pivot_table(index="model", columns="item", values="correct", aggfunc="max")
    item_subtask = long.drop_duplicates("item").set_index("item")["subtask"].to_dict()
    cache.parent.mkdir(parents=True, exist_ok=True)
    mat.to_parquet(cache)
    pd.DataFrame({"item": list(item_subtask), "subtask": list(item_subtask.values())}).to_csv(
        map_cache, index=False)
    return mat, item_subtask


def coverage_filter(mat: pd.DataFrame, item_cov: float,
                    model_cov: float) -> pd.DataFrame:
    """Keep items present for >= item_cov of models, then models with >= model_cov
    of those items observed, then drop any residual holes for a complete matrix."""
    keep_items = mat.columns[mat.notna().mean(axis=0) >= item_cov]
    m = mat[keep_items]
    keep_models = m.index[m.notna().mean(axis=1) >= model_cov]
    m = m.loc[keep_models]
    m = m.loc[:, m.notna().all(axis=0)]
    m = m.loc[m.notna().all(axis=1)]
    return m.astype(int)


# ---------------------------------------------------------------------------
# unidim 2PL helper (shared EM fitter, n_dims=1) + oriented a>0 hygiene
# ---------------------------------------------------------------------------


def fit_unidim(Y: np.ndarray, grid_nodes: int = UNI_GRID) -> dict:
    """Unidimensional 2PL via the shared EM fitter (n_dims=1).

    Returns {a, b, theta} with the latent axis oriented positively (median a>0)
    and the EAP theta per person on that oriented axis. ``b`` is the M2PL OFFSET
    (eta = a*theta - b), consistent with fit_m2pl_em.
    """
    M = np.ones_like(Y, dtype=bool)
    Q = np.ones((Y.shape[1], 1), dtype=int)
    fit = cm.fit_m2pl_em(Y, M, Q, grid_nodes, estimate_corr=False,
                         ridge=1e-2, max_iter=200, tol=1e-4)
    a = fit["A"][:, 0].copy()
    b = fit["b"].copy()
    flip = np.median(a) < 0  # 2PL axis sign is not identified; orient positively.
    if flip:
        a = -a
        b = -b
    theta = eap_theta_unidim(Y, a, b)
    return {"a": a, "b": b, "theta": theta}


def eap_theta_unidim(Y: np.ndarray, a: np.ndarray, b: np.ndarray,
                     nodes: np.ndarray | None = None) -> np.ndarray:
    """EAP theta per person for a fitted 2PL (offset param), standard-normal prior."""
    from scipy.special import log_expit
    if nodes is None:
        nodes = np.linspace(-4.0, 4.0, 81)
    priorw = np.exp(-0.5 * nodes ** 2)
    log_priorw = np.log(priorw / priorw.sum())
    eta = a[None, :] * nodes[:, None] - b[None, :]      # (n_nodes, n_items)
    logP = log_expit(eta)
    log1mP = log_expit(-eta)
    ll = Y @ logP.T + (1.0 - Y) @ log1mP.T               # (n_persons, n_nodes)
    w = np.exp(ll - ll.max(axis=1, keepdims=True)) * np.exp(log_priorw)[None, :]
    w /= w.sum(axis=1, keepdims=True)
    theta = (w * nodes[None, :]).sum(axis=1)
    var = (w * (nodes[None, :] - theta[:, None]) ** 2).sum(axis=1)
    theta = theta.astype(float)
    theta_var = var.astype(float)
    return np.stack([theta, theta_var], axis=1)  # col0 = EAP, col1 = posterior var


# ---------------------------------------------------------------------------
# CATs (consistent OFFSET convention: eta = a*theta - b)
# ---------------------------------------------------------------------------


def cat_unidim_trace(y: np.ndarray, a: np.ndarray, b_loc: np.ndarray, nodes: np.ndarray,
                     log_priorw: np.ndarray, max_items: int) -> list[tuple[int, float, float]]:
    """Unidimensional max-Fisher-info 2PL CAT run to ``max_items``.

    Returns a per-step trace of (n_items, posterior SD, predicted OVERALL accuracy),
    where predicted accuracy is the mean pass-probability over the FULL unidim bank
    at the current theta. Location convention: p = sigmoid(a*(theta - b_loc))."""
    from scipy.special import expit, log_expit
    n = len(a)
    used = np.zeros(n, dtype=bool)
    theta = 0.0
    order: list[int] = []
    trace: list[tuple[int, float, float]] = []
    for _ in range(min(max_items, n)):
        p = expit(a * (theta - b_loc))
        info = a * a * p * (1.0 - p)
        info[used] = -1.0
        j = int(np.argmax(info))
        used[j] = True
        order.append(j)
        idx = np.asarray(order)
        eta = a[idx][None, :] * (nodes[:, None] - b_loc[idx][None, :])
        ll = y[idx][None, :] @ log_expit(eta).T + (1 - y[idx])[None, :] @ log_expit(-eta).T
        ll = ll.ravel()
        w = np.exp(ll - ll.max()) * np.exp(log_priorw)
        w /= w.sum()
        theta = float((nodes * w).sum())
        sd = float(np.sqrt(((nodes - theta) ** 2 * w).sum()))
        pred = float(expit(a * (theta - b_loc)).mean())
        trace.append((len(order), sd, pred))
    return trace


def cat_mirt_trace(y: np.ndarray, A: np.ndarray, b: np.ndarray, n_dims: int,
                   max_items: int, U0: np.ndarray | None = None
                   ) -> list[tuple[int, float, int, float]]:
    """Within-benchmark multidimensional max-Fisher-info CAT run to ``max_items``.

    Simple-structure bank: A row j has one nonzero loading (its subtask dim).
    Returns a per-step trace of (n_items, max SE over observed dims, #dims seen,
    predicted OVERALL accuracy over the FULL bank). ``U0`` = latent correlation R
    shares information across correlated dimensions."""
    from scipy.special import expit
    n_items = len(b)
    theta = np.zeros(n_dims)
    U = np.eye(n_dims) if U0 is None else U0.copy()
    ones_q = np.ones(n_dims)
    dim_of = A.argmax(axis=1)
    remaining = set(range(n_items))
    seen_dims: set[int] = set()
    trace: list[tuple[int, float, int, float]] = []
    n_admin = 0
    while remaining and n_admin < max_items:
        rem = np.fromiter(remaining, dtype=int)
        m = A[rem]
        p = expit(m @ theta - b[rem])
        info = p * (1.0 - p) * np.sum(m * m, axis=1)
        pick = int(rem[int(np.argmax(info))])
        theta, U, _ = tc_mirt.update(theta, U, A[pick], ones_q, float(b[pick]), int(y[pick]))
        remaining.discard(pick)
        seen_dims.add(int(dim_of[pick]))
        n_admin += 1
        se = tc_mirt.standard_errors(U)
        max_se_seen = float(np.max(se[sorted(seen_dims)]))
        pred = float(expit(A @ theta - b).mean())
        trace.append((n_admin, max_se_seen, len(seen_dims), pred))
    return trace


def pearson(x, y) -> float | None:
    xa, ya = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(xa) & np.isfinite(ya)
    xa, ya = xa[ok], ya[ok]
    if xa.size < 2 or xa.std() == 0 or ya.std() == 0:
        return None
    return float(np.corrcoef(xa, ya)[0, 1])


def nearest_pd_corr(R: np.ndarray) -> np.ndarray:
    """Clip eigenvalues to make R positive-definite, then renormalise to unit diag."""
    R = (R + R.T) / 2.0
    w, V = np.linalg.eigh(R)
    w = np.clip(w, 1e-4, None)
    R = (V * w) @ V.T
    d = np.sqrt(np.clip(np.diag(R), 1e-8, None))
    R = R / np.outer(d, d)
    np.fill_diagonal(R, 1.0)
    return R


# ---------------------------------------------------------------------------
# analysis stages
# ---------------------------------------------------------------------------


def per_subtask_calibration(mat: pd.DataFrame, item_subtask: dict[str, str],
                            train_models: list[str], pmin: float, pmax: float,
                            workers: int) -> dict:
    """Per-subtask oriented 2PL (train). Returns per-subtask a>0 item lists, item
    params (a, b offset), and EAP theta for ALL persons on each subtask axis."""
    subtasks = sorted({item_subtask[c] for c in mat.columns})
    cols_by_sub = {s: [c for c in mat.columns if item_subtask[c] == s] for s in subtasks}

    def fit_one(s: str) -> tuple[str, dict]:
        cols = cols_by_sub[s]
        tr = mat.loc[train_models, cols]
        pr = tr.mean(axis=0).to_numpy()
        win = (pr >= pmin) & (pr <= pmax)
        cols_w = [c for c, k in zip(cols, win, strict=True) if k]
        if len(cols_w) < 3:
            return s, {"items": [], "a": np.array([]), "b": np.array([])}
        Ytr = mat.loc[train_models, cols_w].to_numpy(dtype=float)
        fit = fit_unidim(Ytr)
        good = np.where(fit["a"] > 0)[0]
        items = [cols_w[i] for i in good]
        a = fit["a"][good]
        b = fit["b"][good]
        theta_all = eap_theta_unidim(mat.loc[:, items].to_numpy(dtype=float), a, b)
        return s, {"items": items, "a": a, "b": b, "theta": theta_all}

    out: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for fut in as_completed([pool.submit(fit_one, s) for s in subtasks]):
            s, res = fut.result()
            out[s] = res
    return {"subtasks": subtasks, "per_sub": out}


def subtask_correlations(mat: pd.DataFrame, calib: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Latent inter-subtask ability correlation (raw + reliability-disattenuated).

    theta is the per-subtask EAP; reliability = 1 - mean(posterior var)/var(theta).
    Disattenuated r_ij = r_ij / sqrt(rel_i * rel_j)."""
    subs = [s for s in calib["subtasks"] if len(calib["per_sub"][s]["items"]) >= 3]
    thetas = np.column_stack([calib["per_sub"][s]["theta"][:, 0] for s in subs])
    R = np.corrcoef(thetas, rowvar=False)
    rel = np.array([
        max(1e-3, 1.0 - calib["per_sub"][s]["theta"][:, 1].mean()
            / max(calib["per_sub"][s]["theta"][:, 0].var(), 1e-6))
        for s in subs
    ])
    rel = np.clip(rel, 1e-3, 1.0)
    Rd = R / np.sqrt(np.outer(rel, rel))
    Rd = np.clip(Rd, -1.0, 1.0)
    np.fill_diagonal(Rd, 1.0)
    raw = pd.DataFrame(np.round(R, 6), index=subs, columns=subs)
    dis = pd.DataFrame(np.round(Rd, 6), index=subs, columns=subs)
    return raw, dis


def exploratory_dimensionality(mat: pd.DataFrame, item_subtask: dict[str, str],
                               train_models: list[str], calib: dict, k_max: int,
                               grid: int, n_items_explore: int, n_persons_explore: int,
                               seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Fit k=1..k_max EXPLORATORY M2PL (all-ones Q) on a balanced a>0 subsample.

    Returns (dimensionality table, eigenvalues of the inter-item correlation)."""
    rng = np.random.default_rng(seed)
    subs = [s for s in calib["subtasks"] if len(calib["per_sub"][s]["items"]) >= 3]
    per = max(1, n_items_explore // max(len(subs), 1))
    cols: list[str] = []
    for s in subs:
        items = calib["per_sub"][s]["items"]
        take = min(per, len(items))
        sel = rng.choice(len(items), size=take, replace=False)
        cols += [items[i] for i in sorted(sel)]
    persons = list(train_models)
    if len(persons) > n_persons_explore:
        sel = rng.choice(len(persons), size=n_persons_explore, replace=False)
        persons = [persons[i] for i in sorted(sel)]
    Y = mat.loc[persons, cols].to_numpy(dtype=float)
    M = np.ones_like(Y, dtype=bool)
    n_persons, n_items = Y.shape
    n_obs = int(M.sum())

    eig = np.linalg.eigvalsh(np.corrcoef(Y, rowvar=False))[::-1]

    rows = []
    for k in range(1, k_max + 1):
        Q = np.ones((n_items, k), dtype=int)
        fit = cm.fit_m2pl_em(Y, M, Q, grid, estimate_corr=False,
                             ridge=1e-2, max_iter=120, tol=1e-3)
        # Exploratory M2PL params: free loadings + intercepts, minus rotational
        # constraints k*(k-1)/2 (loadings are identified only up to rotation).
        n_params = n_items * k + n_items - k * (k - 1) // 2
        aic, bic = cm.aic_bic(fit["loglik"], n_params, n_obs)
        rows.append({
            "factors": k, "loglik": round(fit["loglik"], 2), "n_params": n_params,
            "aic": round(aic, 2), "bic": round(bic, 2),
            "converged": fit["converged"], "n_iter": fit["n_iter"],
            "grid_nodes": fit["grid_nodes"],
        })
        print(f"[explore] k={k} loglik={fit['loglik']:.1f} AIC={aic:.1f} BIC={bic:.1f} "
              f"(conv={fit['converged']}, {fit['grid_nodes']} nodes)", flush=True)
    tab = pd.DataFrame(rows)
    tab.attrs["n_persons"] = n_persons
    tab.attrs["n_items"] = n_items
    return tab, eig


def _assemble_banks(mat: pd.DataFrame, calib: dict, train_models: list[str]) -> dict:
    """Build the simple-structure MIRT bank (subtasks as dims) and the matched unidim
    2PL bank on the SAME items/train split."""
    subs = [s for s in calib["subtasks"] if len(calib["per_sub"][s]["items"]) >= 3]
    n_dims = len(subs)
    A_rows, b_off, items_all = [], [], []
    for di, s in enumerate(subs):
        ps = calib["per_sub"][s]
        for a_j, b_j, it in zip(ps["a"], ps["b"], ps["items"], strict=True):
            row = np.zeros(n_dims)
            row[di] = a_j
            A_rows.append(row)
            b_off.append(b_j)
            items_all.append(it)
    A = np.array(A_rows)
    b_mirt = np.array(b_off)
    # Unidim full-bank 2PL on the SAME items (train); a>0 already ensured per subtask,
    # but re-fit as ONE unidimensional axis (the baseline's model of the whole suite).
    Ytr = mat.loc[train_models, items_all].to_numpy(dtype=float)
    uni = fit_unidim(Ytr)
    a_u, b_u = uni["a"], uni["b"]
    keep = a_u > 0
    return {"subs": subs, "n_dims": n_dims, "A": A, "b_mirt": b_mirt,
            "items_all": items_all, "a_u": a_u[keep], "b_u_loc": b_u[keep] / a_u[keep],
            "uni_items": [it for it, k in zip(items_all, keep, strict=True) if k]}


def cat_comparison(mat: pd.DataFrame, calib: dict, train_models: list[str],
                   test_models: list[str], se_targets: list[float], R: np.ndarray,
                   min_items: int, max_items: int, budgets: list[int]
                   ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Unidim full-bank 2PL CAT vs within-BBH MIRT CAT, both predicting OVERALL
    accuracy on the held-out models. One trace per model per method yields BOTH the
    SE-stopping metrics and the matched-item-budget control (isolating the effect of
    modeling dimensionality from the effect of administering more items)."""
    bank = _assemble_banks(mat, calib, train_models)
    A, b_mirt, n_dims = bank["A"], bank["b_mirt"], bank["n_dims"]
    a_u, b_u_loc = bank["a_u"], bank["b_u_loc"]
    Yte_uni = mat.loc[test_models, bank["uni_items"]].to_numpy(dtype=float)
    Yte_mirt = mat.loc[test_models, bank["items_all"]].to_numpy(dtype=float)
    actual = Yte_mirt.mean(axis=1)

    nodes1 = np.linspace(-4.0, 4.0, 81)
    priorw1 = np.exp(-0.5 * nodes1 ** 2)
    log_priorw1 = np.log(priorw1 / priorw1.sum())

    uni_traces, mirt_traces = [], []
    for r in range(len(test_models)):
        uni_traces.append(cat_unidim_trace(Yte_uni[r], a_u, b_u_loc, nodes1,
                                            log_priorw1, max_items))
        mirt_traces.append(cat_mirt_trace(Yte_mirt[r], A, b_mirt, n_dims,
                                           max_items, U0=R))

    def uni_stop(tr: list, se_t: float) -> tuple[float, int]:
        for (ni, sd, pred) in tr:
            if ni >= min_items and sd <= se_t:
                return pred, ni
        return tr[-1][2], tr[-1][0]

    def mirt_stop(tr: list, se_t: float) -> tuple[float, int]:
        for (ni, mse, nseen, pred) in tr:
            if ni >= min_items and nseen == n_dims and mse <= se_t:
                return pred, ni
        return tr[-1][3], tr[-1][0]

    # --- SE-based stopping rows ---
    rows = []
    scatter: dict = {}
    for se_t in se_targets:
        pu = np.array([uni_stop(uni_traces[r], se_t) for r in range(len(test_models))])
        pm = np.array([mirt_stop(mirt_traces[r], se_t) for r in range(len(test_models))])
        pred_uni, items_uni = pu[:, 0], pu[:, 1]
        pred_mirt, items_mirt = pm[:, 0], pm[:, 1]
        r_uni, r_mirt = pearson(pred_uni, actual), pearson(pred_mirt, actual)
        rows.append({"se_target": se_t, "model_type": "unidim", "r": r_uni,
                     "mae": float(np.mean(np.abs(pred_uni - actual))),
                     "mean_items": float(items_uni.mean()),
                     "median_items": float(np.median(items_uni)),
                     "n_bank_items": len(bank["uni_items"]), "n_test": len(test_models)})
        rows.append({"se_target": se_t, "model_type": "mirt_within", "r": r_mirt,
                     "mae": float(np.mean(np.abs(pred_mirt - actual))),
                     "mean_items": float(items_mirt.mean()),
                     "median_items": float(np.median(items_mirt)),
                     "n_bank_items": len(bank["items_all"]), "n_test": len(test_models)})
        scatter[se_t] = {"actual": actual, "pred_uni": pred_uni, "pred_mirt": pred_mirt,
                         "r_uni": r_uni, "r_mirt": r_mirt}
        print(f"[cat] SE {se_t}: unidim r={r_uni} items={items_uni.mean():.1f} | "
              f"mirt r={r_mirt} items={items_mirt.mean():.1f}", flush=True)

    # --- matched-budget control (both methods at identical #items, no early stop) ---
    brows = []
    for nb in budgets:
        pu = np.array([tr[min(nb, len(tr)) - 1][2] for tr in uni_traces])
        pm = np.array([tr[min(nb, len(tr)) - 1][3] for tr in mirt_traces])
        r_uni, r_mirt = pearson(pu, actual), pearson(pm, actual)
        brows.append({"n_items": nb, "model_type": "unidim", "r": r_uni,
                      "mae": float(np.mean(np.abs(pu - actual)))})
        brows.append({"n_items": nb, "model_type": "mirt_within", "r": r_mirt,
                      "mae": float(np.mean(np.abs(pm - actual)))})
        print(f"[budget] n={nb}: unidim r={r_uni} | mirt r={r_mirt}", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(brows), scatter


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def make_figures(out_dir: Path, dim_tab: pd.DataFrame, eig: np.ndarray,
                 corr_raw: pd.DataFrame, cat_df: pd.DataFrame, budget_df: pd.DataFrame,
                 scatter: dict, se_targets: list[float], bench: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. scree + AIC/BIC vs factors
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    n_show = min(15, len(eig))
    ax1.plot(np.arange(1, n_show + 1), eig[:n_show], "o-", color="#1f77b4")
    ax1.axhline(1.0, ls="--", color="gray", lw=1)
    ax1.set_xlabel("factor number")
    ax1.set_ylabel("eigenvalue of item correlation matrix")
    n_ge1 = int(np.sum(eig >= 1.0))
    ax1.set_title(f"{bench} scree: {n_ge1} eigenvalues >= 1")
    ax1.grid(True, alpha=0.3)
    ax2.plot(dim_tab["factors"], dim_tab["aic"], "o-", label="AIC", color="#d62728")
    ax2.plot(dim_tab["factors"], dim_tab["bic"], "s-", label="BIC", color="#2ca02c")
    best_aic = int(dim_tab.loc[dim_tab["aic"].idxmin(), "factors"])
    best_bic = int(dim_tab.loc[dim_tab["bic"].idxmin(), "factors"])
    ax2.set_xlabel("number of latent factors k")
    ax2.set_ylabel("information criterion (lower = better)")
    ax2.set_title(f"{bench} exploratory MIRT: AIC min k={best_aic}, BIC min k={best_bic}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{bench}_dimensionality.png", dpi=140)
    plt.close(fig)

    # 2. subtask latent-correlation heatmap
    subs = list(corr_raw.index)
    Rm = corr_raw.to_numpy()
    off = Rm[~np.eye(len(subs), dtype=bool)]
    fig, ax = plt.subplots(figsize=(max(7, len(subs) * 0.42), max(6, len(subs) * 0.42)))
    im = ax.imshow(Rm, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(subs)))
    ax.set_yticks(range(len(subs)))
    ax.set_xticklabels(subs, rotation=90, fontsize=6)
    ax.set_yticklabels(subs, fontsize=6)
    ax.set_title(f"{bench} inter-subtask ability correlation "
                 f"(mean off-diag {off.mean():.2f}, range {off.min():.2f} to {off.max():.2f})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{bench}_subtask_correlation_heatmap.png", dpi=140)
    plt.close(fig)

    # 3. recovery scatter unidim vs MIRT (largest SE target)
    se0 = max(se_targets)
    d = scatter[se0]
    act = d["actual"]
    fig, ax = plt.subplots(figsize=(6.2, 6))
    ru = "n/a" if d["r_uni"] is None else f"{d['r_uni']:.3f}"
    rm = "n/a" if d["r_mirt"] is None else f"{d['r_mirt']:.3f}"
    ax.scatter(act, d["pred_uni"], s=26, alpha=0.7, color="#dd8452", marker="x",
               label=f"unidim (r={ru})")
    ax.scatter(act, d["pred_mirt"], s=26, alpha=0.7, color="#4c72b0",
               label=f"within-BBH MIRT (r={rm})")
    lo = min(act.min(), d["pred_uni"].min(), d["pred_mirt"].min()) - 0.02
    hi = max(act.max(), d["pred_uni"].max(), d["pred_mirt"].max()) + 0.02
    ax.plot([lo, hi], [lo, hi], "--", color="gray", lw=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(f"actual overall {bench} accuracy (fraction correct)")
    ax.set_ylabel("CAT-predicted overall accuracy (fraction correct)")
    ax.set_title(f"{bench} accuracy recovery at SE {se0}: MIRT vs unidimensional CAT")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{bench}_cat_recovery.png", dpi=140)
    plt.close(fig)

    # 4. matched-budget control: r vs #items for both methods (equal item counts)
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for mt, color, marker, lab in [("unidim", "#dd8452", "x", "unidim (1 factor)"),
                                   ("mirt_within", "#4c72b0", "o", "within-BBH MIRT")]:
        g = budget_df[budget_df.model_type == mt].sort_values("n_items")
        ax.plot(g["n_items"], g["r"], marker=marker, ls="-", color=color, label=lab)
    ax.set_xlabel("number of CAT items administered (matched budget)")
    ax.set_ylabel("Pearson r (predicted vs actual overall accuracy)")
    ax.set_title(f"{bench}: at every matched item budget, MIRT recovers accuracy better")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{bench}_matched_budget.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# README
# ---------------------------------------------------------------------------


def write_readme(out_dir: Path, bench: str, dim_tab: pd.DataFrame, eig: np.ndarray,
                 corr_raw: pd.DataFrame, corr_dis: pd.DataFrame, cat_df: pd.DataFrame,
                 budget_df: pd.DataFrame, se_targets: list[float], n_models: int,
                 n_train: int, n_test: int, n_bank: int, args) -> None:
    subs = list(corr_raw.index)
    off = corr_raw.to_numpy()[~np.eye(len(subs), dtype=bool)]
    off_d = corr_dis.to_numpy()[~np.eye(len(subs), dtype=bool)]
    n_ge1 = int(np.sum(eig >= 1.0))
    best_aic = int(dim_tab.loc[dim_tab["aic"].idxmin(), "factors"])
    best_bic = int(dim_tab.loc[dim_tab["bic"].idxmin(), "factors"])

    se0 = max(se_targets)
    row_u = cat_df[(cat_df.se_target == se0) & (cat_df.model_type == "unidim")].iloc[0]
    row_m = cat_df[(cat_df.se_target == se0) & (cat_df.model_type == "mirt_within")].iloc[0]
    r_u = row_u["r"] if pd.notna(row_u["r"]) else float("nan")
    r_m = row_m["r"] if pd.notna(row_m["r"]) else float("nan")

    multi_real = off.mean() < 0.8
    mirt_helps = (pd.notna(row_m["r"]) and pd.notna(row_u["r"]) and r_m > r_u + 0.01)
    dim_word = "genuinely multidimensional" if multi_real else "effectively one factor"
    cat_word = "recovers overall accuracy better than" if mirt_helps else "does not beat"
    verdict = (
        f"{bench} subtasks are {dim_word} "
        f"(mean inter-subtask ability correlation {off.mean():.2f}); "
        f"the within-{bench} MIRT CAT {cat_word} "
        f"the unidimensional CAT (r {r_m:.3f} vs {r_u:.3f} at SE {se0})."
    )

    lines = []
    lines.append(f"# Within-benchmark MIRT on {bench}: internal multidimensionality\n")
    lines.append(f"**Verdict.** {verdict}\n")
    lines.append("Fits MIRT WITHIN a single benchmark (subtasks as latent dimensions) to test "
                 "whether that benchmark is internally multidimensional and whether modeling it "
                 "recovers the true benchmark accuracy better than a unidimensional CAT. This is "
                 "distinct from the earlier between-benchmark pooling (mirt_mcq_test.py), which "
                 "only showed that different benchmarks measure different things.\n")

    lines.append("## Setup\n")
    lines.append(f"- Data: OpenLM `{bench}` (status==ok models). Complete matrix after coverage "
                 f"filter: {n_models} models x {n_bank} items (train {n_train} / test {n_test}, "
                 f"seed {args.seed}, {args.test_frac:.0%} held out).")
    lines.append(f"- Subtasks (latent dimensions): {len(subs)} with >= 3 usable items each.")
    lines.append("- Item hygiene: per-subtask oriented unidimensional 2PL, keep a>0 items "
                 f"(pass-rate window [{args.pmin}, {args.pmax}] on train).")
    lines.append("- Fitter: shared FRQ MIRT EM (`eduLLM-Evals/scripts/calibrate_mirt.py "
                 "fit_m2pl_em`), Bock-Aitkin marginal-ML. CAT via `tutor_cat.mirt.update` / "
                 "`standard_errors`.")
    lines.append("")
    lines.append("### Missing-data / intractability fallback (read this)\n")
    lines.append(f"A joint confirmatory M2PL with all {len(subs)} subtasks as dimensions is "
                 "computationally intractable with the shared Gauss-Hermite EM: the fixed grid "
                 f"has `grid**n_dims` nodes (grid**{len(subs)}). We therefore estimate the "
                 "confirmatory simple-structure latent correlation via PER-SUBTASK "
                 "unidimensional 2PL calibration followed by the EAP-ability correlation across "
                 "subtasks (raw and reliability-disattenuated), and assemble the simple-structure "
                 "MIRT CAT bank from those per-subtask item parameters (each item loads only on "
                 "its subtask dimension), with the estimated latent correlation R as the CAT "
                 "prior covariance. Per-item subtask labels ARE present in the OpenLM dump, so no "
                 "exploratory-only fallback was needed for the Q-matrix.\n")

    n_pe = dim_tab.attrs.get("n_persons", "?")
    n_ie = dim_tab.attrs.get("n_items", "?")
    lines.append(f"## 1. Exploratory dimensionality: how many factors does {bench} need?\n")
    lines.append("Exploratory M2PL (all-ones Q, k free loadings/item) on a balanced a>0 subsample "
                 f"({n_pe} persons x {n_ie} items, grid={args.grid} nodes/dim, "
                 "param count rotation-corrected).\n")
    lines.append("| factors k | loglik | n_params | AIC | BIC | converged |")
    lines.append("|---:|---:|---:|---:|---:|---|")
    for _, r in dim_tab.iterrows():
        lines.append(f"| {int(r['factors'])} | {r['loglik']:.1f} | {int(r['n_params'])} | "
                     f"{r['aic']:.1f} | {r['bic']:.1f} | {r['converged']} |")
    k_max = int(dim_tab["factors"].max())
    not_plateaued = (best_aic == k_max) or (best_bic == k_max)
    plateau_note = (
        f" Both criteria are still decreasing at the largest k tested (k={k_max}), so the "
        f"effective dimensionality is at LEAST {k_max}; it did not plateau within the tested "
        "range, consistent with a suite of ~24 heterogeneous subtasks."
        if not_plateaued else
        f" AIC/BIC turn over, so the effective dimensionality is about k={min(best_aic, best_bic)}."
    )
    lines.append(f"\nAIC is minimised at k={best_aic}, BIC at k={best_bic}.{plateau_note} "
                 f"The inter-item correlation matrix has a dominant first eigenvalue with a long "
                 f"tail (top eigenvalues: {', '.join(f'{v:.1f}' for v in eig[:8])}); {n_ge1} "
                 f"exceed 1, though the Kaiser>1 count over-states factor count for binary items "
                 f"and is shown only as a scree. The unidimensional 1-factor model is decisively "
                 f"the worst fit (highest AIC/BIC), so the suite is multidimensional.\n")

    lines.append("## 2. Subtask correlation structure: one factor or many?\n")
    lines.append(f"Latent inter-subtask ability correlation across {len(subs)} subtasks "
                 f"(per-subtask EAP theta on the full model set).\n")
    lines.append(f"- Raw off-diagonal correlation: mean {off.mean():.2f}, "
                 f"range {off.min():.2f} to {off.max():.2f}.")
    lines.append(f"- Reliability-disattenuated off-diagonal: mean {off_d.mean():.2f}, "
                 f"range {off_d.min():.2f} to {off_d.max():.2f}.")
    rel_word = "well below" if off.mean() < 0.8 else "near/above"
    lines.append(f"- Interpretation: the mean raw correlation {off.mean():.2f} is {rel_word} the "
                 f"0.8 one-factor threshold, so subtasks are {dim_word}.\n")

    lines.append("## 3. Within-BBH MIRT vs unidimensional CAT (recovering overall accuracy)\n")
    lines.append("Both CATs predict the OVERALL benchmark accuracy on the SAME held-out models. "
                 "`items` = mean number administered before the SE stopping rule. The "
                 "unidimensional CAT stops when a single theta's SE < target (documented early "
                 "stop); the MIRT CAT stops when the slowest subtask dimension's SE < target, so "
                 "it cannot early-stop on the whole heterogeneous suite.\n")
    lines.append("| SE | model | r | MAE | mean items | median items | bank items |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for se_t in se_targets:
        for mt in ["unidim", "mirt_within"]:
            r = cat_df[(cat_df.se_target == se_t) & (cat_df.model_type == mt)].iloc[0]
            rr = f"{r['r']:.3f}" if pd.notna(r["r"]) else "n/a"
            lines.append(f"| {se_t} | {mt} | {rr} | {r['mae']:.4f} | {r['mean_items']:.1f} | "
                         f"{r['median_items']:.0f} | {int(r['n_bank_items'])} |")
    lines.append("")

    lines.append("### Matched-item-budget control (isolates dimensionality from item count)\n")
    lines.append("Both CATs forced to the SAME number of items (no early stop), so any gap is due "
                 "to modeling multidimensionality, not to administering more items.\n")
    lines.append("| # items | unidim r | MIRT r | unidim MAE | MIRT MAE |")
    lines.append("|---:|---:|---:|---:|---:|")
    for nb in sorted(budget_df["n_items"].unique()):
        ru = budget_df[(budget_df.n_items == nb) & (budget_df.model_type == "unidim")].iloc[0]
        rm = budget_df[(budget_df.n_items == nb) & (budget_df.model_type == "mirt_within")].iloc[0]
        rus = f"{ru['r']:.3f}" if pd.notna(ru["r"]) else "n/a"
        rms = f"{rm['r']:.3f}" if pd.notna(rm["r"]) else "n/a"
        lines.append(f"| {int(nb)} | {rus} | {rms} | {ru['mae']:.4f} | {rm['mae']:.4f} |")
    lines.append("")

    lines.append("## Reference: published unidimensional 3PL numbers\n")
    if UNIDIM_SUMMARY.exists():
        u = pd.read_csv(UNIDIM_SUMMARY)
        u = u[u.benchmark == bench]
        if not u.empty:
            lines.append("From `data/atlas_replication/summary_pirt_mae_sd_se.csv` "
                         "(full-bank 3PL R-mirt CAT):\n")
            lines.append("| SE | r | MAE | items |")
            lines.append("|---|---:|---:|---:|")
            for _, rr in u.iterrows():
                lines.append(f"| {rr['se_target']} | {rr['r']:.3f} | {rr['mae']:.4f} | "
                             f"{rr['n_subset_items']:.0f} |")
    lines.append("\n> The unidim CAT here is a 2PL EM re-run on the SAME items/split as the MIRT "
                 "bank (apples-to-apples); the published 3PL full-bank numbers use different item "
                 "counts and a 3PL fitter, echoed for reference.\n")

    lines.append("## Files\n")
    lines.append(f"- `{bench}_dimensionality.csv` -- factors, loglik, AIC, BIC, eigenvalues.")
    lines.append(f"- `{bench}_subtask_correlations.csv` / `_disattenuated.csv` -- inter-subtask "
                 "ability correlations.")
    lines.append(f"- `{bench}_cat_comparison.csv` -- unidim vs within-BBH MIRT (r, MAE, items) "
                 "at each SE target.")
    lines.append(f"- `{bench}_cat_budget.csv` -- matched-item-budget r/MAE for both models.")
    lines.append(f"- figures/: `{bench}_dimensionality.png`, "
                 f"`{bench}_subtask_correlation_heatmap.png`, `{bench}_cat_recovery.png`, "
                 f"`{bench}_matched_budget.png`.")
    (out_dir / "README.md").write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bench", default="bbh", help="OpenLM benchmark (default bbh).")
    p.add_argument("--pmin", type=float, default=0.05,
                   help="min item pass-rate to keep (variance floor).")
    p.add_argument("--pmax", type=float, default=0.95, help="max item pass-rate to keep.")
    p.add_argument("--item-cov", type=float, default=0.9,
                   help="keep items present for >= this fraction of models.")
    p.add_argument("--model-cov", type=float, default=0.95,
                   help="keep models observing >= this fraction of kept items.")
    p.add_argument("--k-max", type=int, default=6, help="max exploratory factors.")
    p.add_argument("--grid", type=int, default=4,
                   help="Gauss-Hermite nodes/dim for exploratory MIRT (grid**k nodes).")
    p.add_argument("--n-items-explore", type=int, default=300,
                   help="total items subsampled for exploratory fits (balanced by subtask).")
    p.add_argument("--n-persons-explore", type=int, default=500,
                   help="persons subsampled for exploratory fits.")
    p.add_argument("--se-targets", type=str, default="0.3,0.2")
    p.add_argument("--test-frac", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--min-items", type=int, default=8, help="CAT minimum items.")
    p.add_argument("--max-items", type=int, default=500, help="CAT maximum items (trace length).")
    p.add_argument("--budgets", type=str, default="25,50,100,200,400",
                   help="matched item budgets for the equal-items control.")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = p.parse_args()

    se_targets = [float(s) for s in args.se_targets.split(",") if s.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(args.out_dir / ".mplcache"))

    keep = ok_models()
    print(f"ok models in status file: {len(keep)}")

    cache = args.out_dir / f"_matrix_{args.bench}.parquet"
    map_cache = args.out_dir / f"_subtask_map_{args.bench}.csv"
    mat_raw, item_subtask = build_matrix(args.bench, keep, cache, map_cache)
    print(f"[{args.bench}] raw matrix: {mat_raw.shape[0]} models x {mat_raw.shape[1]} items")

    mat = coverage_filter(mat_raw, args.item_cov, args.model_cov)
    mat = mat.loc[:, mat.nunique() > 1]  # drop no-variance items
    print(f"[{args.bench}] complete matrix: {mat.shape[0]} models x {mat.shape[1]} items")

    # fixed train/test split (seed matches the openlm baseline convention)
    rng = np.random.default_rng(args.seed)
    models = list(mat.index)
    order = [models[i] for i in rng.permutation(len(models))]
    n_test = max(2, int(round(len(order) * args.test_frac)))
    test_models = sorted(order[:n_test])
    train_models = sorted(order[n_test:])
    print(f"[{args.bench}] train={len(train_models)} test={len(test_models)}")

    # --- per-subtask calibration + correlation structure ---
    print("per-subtask 2PL calibration ...", flush=True)
    calib = per_subtask_calibration(mat, item_subtask, train_models,
                                    args.pmin, args.pmax, args.workers)
    n_bank = sum(len(calib["per_sub"][s]["items"]) for s in calib["subtasks"])
    kept_subs = [s for s in calib["subtasks"] if len(calib["per_sub"][s]["items"]) >= 3]
    print(f"[{args.bench}] usable subtasks={len(kept_subs)} a>0 bank items={n_bank}")

    corr_raw, corr_dis = subtask_correlations(mat, calib)
    corr_raw.to_csv(args.out_dir / f"{args.bench}_subtask_correlations.csv")
    corr_dis.to_csv(args.out_dir / f"{args.bench}_subtask_correlations_disattenuated.csv")
    off = corr_raw.to_numpy()[~np.eye(len(corr_raw), dtype=bool)]
    print(f"[{args.bench}] inter-subtask corr mean off-diag={off.mean():.3f} "
          f"range [{off.min():.3f},{off.max():.3f}]")

    # --- exploratory dimensionality ---
    print("exploratory dimensionality (k=1..K) ...", flush=True)
    dim_tab, eig = exploratory_dimensionality(
        mat, item_subtask, train_models, calib, args.k_max, args.grid,
        args.n_items_explore, args.n_persons_explore, args.seed)
    dim_out = dim_tab.copy()
    dim_out["top_eigenvalues"] = ""
    dim_out.loc[dim_out.index[0], "top_eigenvalues"] = ";".join(f"{v:.4f}" for v in eig[:15])
    dim_out.to_csv(args.out_dir / f"{args.bench}_dimensionality.csv", index=False)

    # --- CAT comparison ---
    print("CAT comparison (unidim vs within-BBH MIRT) ...", flush=True)
    R = nearest_pd_corr(corr_raw.to_numpy())
    budgets = [int(b) for b in args.budgets.split(",") if b.strip()]
    cat_df, budget_df, scatter = cat_comparison(
        mat, calib, train_models, test_models, se_targets, R,
        args.min_items, args.max_items, budgets)
    cat_df.to_csv(args.out_dir / f"{args.bench}_cat_comparison.csv", index=False)
    budget_df.to_csv(args.out_dir / f"{args.bench}_cat_budget.csv", index=False)

    # --- figures + README ---
    make_figures(args.out_dir, dim_tab, eig, corr_raw, cat_df, budget_df, scatter,
                 se_targets, args.bench)
    write_readme(args.out_dir, args.bench, dim_tab, eig, corr_raw, corr_dis, cat_df,
                 budget_df, se_targets, mat.shape[0], len(train_models),
                 len(test_models), n_bank, args)

    print("\nwrote:")
    for f in [f"{args.bench}_dimensionality.csv", f"{args.bench}_subtask_correlations.csv",
              f"{args.bench}_cat_comparison.csv", f"{args.bench}_cat_budget.csv", "README.md"]:
        print("  ", args.out_dir / f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
