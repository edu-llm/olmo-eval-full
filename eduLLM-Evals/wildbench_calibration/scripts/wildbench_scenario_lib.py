"""Shared helpers for the SCENARIO-LEVEL WildBench calibration study.

This mirrors the scenario-level Bridge study (``bridge_calibration/scripts/
bridge_scenario_lib.py``) but for WildBench: 1,001 scenarios / 11,416 criteria /
52 models, with an 11-way task-category skill axis. As in Bridge, a scenario is
administered as a testlet BUNDLE of its per-criterion IRT items (grouped by
scenario). The per-criterion M2PL *fit* reuses ``calibrate_mirt.fit_m2pl_em``
verbatim; only the Q-matrix assembly + fitted-bank emission are WildBench-specific.

Locked design (mirrors Bridge):
  * Bank = ALL 1,001 scenarios (NO source dedup). ``source_id`` is used ONLY for
    leakage-safe fold grouping. NOTE: in WildBench every scenario carries a UNIQUE
    source_id, so source-grouping is a no-op (each scenario is its own source) --
    recorded for parity + auditability.
  * Criterion-level zero-variance (all-fail / all-pass) filtering applies within
    every fit and every fold, exactly as Bridge did.

Dimensionality note (the 11-skill question):
  Each criterion is ONE-HOT mapped to exactly one of the 11 task-category skills.
  A full 11-dim confirmatory M2PL over a product Gauss-Hermite grid is infeasible
  (nodes**11). BUT because the loadings are one-hot, the 11-dim model with a fixed
  (identity) latent correlation FACTORISES into 11 independent 1-D fits -- so we fit
  it exactly that way (:func:`fit_independent`) and recover the inter-skill latent
  correlation via a plug-in on the per-skill EAP abilities (:func:`skill_theta_matrix`),
  which is essentially unattenuated here (~760 criteria/skill pin each ability).
  Small collapsed groupings (2-5 super-skills) stay one-hot at the GROUP level and
  are cheap on a product grid, so those DO estimate the group latent correlation.
"""

from __future__ import annotations

import hashlib
import json
import os

# Pin BLAS threads BEFORE numpy imports downstream (OpenBLAS oversubscription guard).
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
from pathlib import Path

import numpy as np
from scipy.special import log_expit, logsumexp

# eduLLM-Evals root (…/wildbench_calibration/scripts/this_file -> parents[2]).
ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import calibrate_partial as cp  # noqa: E402  (load_matrix / select_sparse / split_zero_variance / jsonl IO)
import calibrate_mirt as cm  # noqa: E402  (fit_m2pl_em / build_grid / base_log_weights / aic_bic)

# ---------------------------------------------------------------------------
# canonical WildBench inputs + skill axis
# ---------------------------------------------------------------------------

DEFAULT_MATRIX = ROOT / "WildBenchGrade" / "response_matrix.csv"
DEFAULT_RUBRICS = ROOT / "data" / "WildBench" / "rubrics.jsonl"
DEFAULT_SCENARIOS = ROOT / "data" / "WildBench" / "scenarios.jsonl"

# WildBench's 11 task-category skills (fixed alphabetical order used everywhere).
WILDBENCH_SKILLS = [
    "advice_seeking", "brainstorming", "coding_debugging", "creative_writing",
    "data_analysis", "editing", "information_seeking", "math", "planning",
    "reasoning", "role_playing",
]

# Candidate dimensionality structures. Each maps a modeled dim label -> the native
# skills OR'd into it. ``overall_1d`` and ``full_11d`` are the extremes; the middle
# candidates are semantic super-skill groupings (an EFA-derived grouping is added at
# run time by the dimensionality driver). Because each criterion is one-hot, every
# collapsed group is itself one-hot at the group level.
GEN = ["creative_writing", "role_playing", "brainstorming"]
TECH = ["coding_debugging", "math", "data_analysis", "reasoning"]
INFO = ["information_seeking", "advice_seeking", "editing", "planning"]

STRUCTURES: dict[str, dict[str, list[str]]] = {
    "overall_1d": {"overall": list(WILDBENCH_SKILLS)},
    "gen_vs_rest_2d": {
        "generative": GEN,
        "analytic_assist": TECH + INFO,
    },
    "gen_tech_info_3d": {
        "generative": GEN,
        "technical": TECH,
        "info_assist": INFO,
    },
    "gen_tech_info_editcreate_4d": {
        "generative": GEN,
        "technical": TECH,
        "info_seeking": ["information_seeking", "advice_seeking"],
        "structuring": ["editing", "planning"],
    },
    "full_11d": {s: [s] for s in WILDBENCH_SKILLS},
}


# ---------------------------------------------------------------------------
# maps
# ---------------------------------------------------------------------------


def load_maps(rubrics_path: Path, scenarios_path: Path) -> dict:
    """Load the criterion/scenario/source maps + per-criterion metadata."""
    c2s: dict[str, str] = {}
    qmap: dict[str, dict[str, int]] = {}
    crit_text: dict[str, str] = {}
    primary: dict[str, str] = {}
    criticality: dict[str, str] = {}
    for rec in cp.read_jsonl(rubrics_path):
        cid = rec.get("criterion_id")
        if cid is None:
            continue
        c2s[cid] = rec.get("scenario_id", cid)
        qm = rec.get("q_mapping") or {}
        qmap[cid] = {k: int(v) for k, v in qm.items()}
        crit_text[cid] = rec.get("criterion", "")
        primary[cid] = rec.get("primary_skill", "")
        criticality[cid] = rec.get("criticality", "standard")

    s2src: dict[str, str] = {}
    scen_criterion_ids: dict[str, list[str]] = {}
    use_case: dict[str, str] = {}
    for rec in cp.read_jsonl(scenarios_path):
        sid = rec.get("scenario_id")
        if sid is None:
            continue
        s2src[sid] = str(rec.get("source_id", sid))
        scen_criterion_ids[sid] = list(rec.get("criterion_ids", []))
        use_case[sid] = rec.get("use_case", "")
    return {
        "c2s": c2s, "qmap": qmap, "crit_text": crit_text, "primary": primary,
        "criticality": criticality, "s2src": s2src,
        "scen_criterion_ids": scen_criterion_ids, "use_case": use_case,
    }


# ---------------------------------------------------------------------------
# matrix prep + zero-variance filtering (identical policy to Bridge/item study)
# ---------------------------------------------------------------------------


def prepare_matrix(matrix_path: Path):
    """Load matrix, drop empty rows/cols + zero-variance criteria (over observed cells).

    Returns (Y, M, items, models, drop_info). NO source dedup.
    """
    mat = cp.load_matrix(matrix_path)
    n_cols_raw = mat.shape[1]
    sub, _ = cp.select_sparse(mat)
    kept, all_fail, all_pass = cp.split_zero_variance(sub)
    sub = sub[kept]
    sub = sub[sub.notna().any(axis=1)]
    items = list(sub.columns)
    models = list(sub.index)
    Y = np.nan_to_num(sub.to_numpy(dtype=float), nan=0.0)
    M = sub.notna().to_numpy()
    drop_info = {
        "n_criteria_raw": int(n_cols_raw),
        "n_models": len(models),
        "dropped_all_fail": len(all_fail),
        "dropped_all_pass": len(all_pass),
        "dropped_zero_variance_total": len(all_fail) + len(all_pass),
        "dropped_all_fail_ids": list(all_fail),
        "dropped_all_pass_ids": list(all_pass),
        "n_criteria_fit": len(items),
        "fill_rate": float(M.mean()),
    }
    return Y, M, items, models, drop_info


def build_Q(items: list[str], qmap: dict[str, dict[str, int]],
            structure: dict[str, list[str]]) -> tuple[np.ndarray, list[str]]:
    """Confirmatory Q for a modeled structure: q_dc = 1 iff criterion c loads on any
    native skill folded into modeled dim d (logical OR). Returns (Q, dim_labels)."""
    labels = list(structure.keys())
    rows = []
    for c in items:
        row = qmap.get(c, {})
        rows.append([1 if any(int(row.get(s, 0)) == 1 for s in structure[d]) else 0
                     for d in labels])
    return np.array(rows, dtype=int), labels


# ---------------------------------------------------------------------------
# fit paths: product-grid (small dims) OR independent per-dim (one-hot large dims)
# ---------------------------------------------------------------------------


def fit_structure(Y, M, Q, grid, ridge, estimate_corr, max_iter=200, tol=1e-4):
    """Product-grid confirmatory M2PL for one (small) structure. estimate_corr gives R."""
    keepj = np.where(Q.sum(axis=1) > 0)[0]
    Qk = Q[keepj]
    fit = cm.fit_m2pl_em(Y[:, keepj], M[:, keepj], Qk, grid,
                         estimate_corr=estimate_corr, ridge=ridge,
                         max_iter=max_iter, tol=tol)
    return fit, keepj


def fit_independent(Y, M, Q, grid, ridge, max_iter=200, tol=1e-4):
    """Fit a one-hot confirmatory M2PL as INDEPENDENT per-dimension 1-D fits.

    Valid iff every kept item loads on exactly ONE modeled dim (one-hot Q at the
    modeled level). The n_dims-dim model with identity latent correlation then
    factorises across dims; we fit each dim's items as a 1-D M2PL and stitch the
    loadings back into an (n_items, n_dims) matrix (0 off the one-hot column).

    Returns a fit dict with the same keys as ``fit_m2pl_em`` plus ``per_dim`` (the
    per-dim sub-fits) and ``R`` = None (correlation recovered downstream via a plug-in
    on the per-skill EAP abilities). loglik / n_params are the exact block-diagonal
    (R=I) totals, so AIC/BIC are directly comparable to the other structures.
    """
    keepj = np.where(Q.sum(axis=1) > 0)[0]
    Qk = Q[keepj]
    if not np.all(Qk.sum(axis=1) == 1):
        raise ValueError("fit_independent requires one-hot modeled Q (>=1 item loads on "
                         "multiple dims).")
    n_items, n_dims = Qk.shape
    A = np.zeros((n_items, n_dims))
    b = np.zeros(n_items)
    loglik = 0.0
    n_iter = 0
    converged = True
    per_dim = {}
    for d in range(n_dims):
        rows = np.where(Qk[:, d] == 1)[0]
        cols = keepj[rows]
        Q1 = np.ones((rows.size, 1), dtype=int)
        f = cm.fit_m2pl_em(Y[:, cols], M[:, cols], Q1, grid, estimate_corr=False,
                           ridge=ridge, max_iter=max_iter, tol=tol)
        A[rows, d] = f["A"][:, 0]
        b[rows] = f["b"]
        loglik += f["loglik"]
        n_iter = max(n_iter, f["n_iter"])
        converged = converged and f["converged"]
        per_dim[d] = {"n_items": int(rows.size), "loglik": float(f["loglik"]),
                      "converged": bool(f["converged"])}
    n_free_load = int(Qk.sum())
    n_params = n_free_load + n_items  # R fixed to I (no correlation params)
    return {
        "A": A, "b": b, "R": None, "loglik": float(loglik), "n_params": int(n_params),
        "n_free_loadings": n_free_load, "n_iter": n_iter, "converged": converged,
        "n_dims": n_dims, "grid_nodes": int(grid), "per_dim": per_dim,
    }, keepj


def eap_theta_1d(Y, M, A1, b1, n_nodes=41, half=6.0):
    """Per-model 1-D EAP ability over a fine uniform grid for a single skill's items.

    ``A1`` (n_items,) discriminations, ``b1`` (n_items,) difficulties, ``Y``/``M`` the
    responses/mask restricted to that skill's items. Returns theta (n_models,).
    """
    axis = np.linspace(-half, half, n_nodes)
    logphi = -0.5 * axis ** 2
    logphi = logphi - logsumexp(logphi)
    eta = A1[:, None] * axis[None, :] - b1[:, None]        # (n_items, n_nodes)
    logP, log1mP = log_expit(eta), log_expit(-eta)
    YM = np.where(M, Y, 0.0)
    NM = np.where(M, 1.0 - Y, 0.0)
    LL = YM @ logP + NM @ log1mP                            # (n_models, n_nodes)
    joint = LL + logphi[None, :]
    post = np.exp(joint - logsumexp(joint, axis=1)[:, None])
    return post @ axis


def skill_theta_matrix(Y, M, items, maps, fit_indep, keepj, n_nodes=41):
    """(n_models, n_dims) EAP-ability matrix from an independent per-dim fit.

    Column d = each model's 1-D EAP ability on dim d's items. Used to estimate the
    inter-skill latent correlation as a plug-in (near-unattenuated at this bank size).
    """
    A = fit_indep["A"]
    b = fit_indep["b"]
    n_dims = A.shape[1]
    thetas = np.full((Y.shape[0], n_dims), np.nan)
    for d in range(n_dims):
        rows = np.where(A[:, d] != 0)[0]
        # rows are indices into keepj-space; map to matrix columns
        cols = keepj[rows]
        if rows.size == 0:
            continue
        thetas[:, d] = eap_theta_1d(Y[:, cols], M[:, cols], A[rows, d], b[rows], n_nodes)
    return thetas


# ---------------------------------------------------------------------------
# fitted-bank emission (schema consumed by scripts.scenario_cat_lib)
# ---------------------------------------------------------------------------


def matrix_sha256(matrix_path: Path) -> str:
    return hashlib.sha256(Path(matrix_path).read_bytes()).hexdigest()


def write_fitted_bank(out_path: Path, items: list[str], keepj: np.ndarray,
                      A: np.ndarray, b: np.ndarray, Q: np.ndarray,
                      dim_labels: list[str], maps: dict, matrix_path: Path,
                      structure_name: str, ridge: float, grid: int) -> int:
    """Emit a ``*_fitted.jsonl`` bank in the schema scenario_cat_lib expects."""
    provenance = {
        "source": "calibrated-m2pl-scenario",
        "method": "confirmatory-m2pl-mml-em",
        "structure": structure_name,
        "dims": list(dim_labels),
        "ridge": ridge,
        "grid_nodes_per_dim": grid,
        "provenance": {
            "matrix_csv": str(matrix_path),
            "matrix_sha256": matrix_sha256(matrix_path),
        },
        "calibrated_at": cp._utcnow(),
        "version": "1.0",
    }
    kept_ids = [items[j] for j in keepj]
    n = 0
    with Path(out_path).open("w", encoding="utf-8") as fh:
        for row, cid in enumerate(kept_ids):
            avec = A[row]
            if not (np.all(np.isfinite(avec)) and np.isfinite(b[row])):
                continue
            rec = {
                "criterion_id": cid,
                "scenario_id": maps["c2s"].get(cid, cid),
                "criterion": maps["crit_text"].get(cid, ""),
                "primary_skill": maps["primary"].get(cid, ""),
                "criticality": maps["criticality"].get(cid, "standard"),
                "scoring_type": "binary",
                "discrimination": {d: round(float(avec[k]), 6)
                                   for k, d in enumerate(dim_labels)},
                "q_modeled": {d: int(Q[keepj[row], k])
                              for k, d in enumerate(dim_labels)},
                "difficulty": round(float(b[row]), 6),
                "irt_params": provenance,
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


# ---------------------------------------------------------------------------
# leakage-safe source -> fold grouping (unique sources => scenario partition)
# ---------------------------------------------------------------------------


def source_fold_groups(maps: dict, k: int = 5, seed: int = 20260729) -> dict:
    """Assign every scenario to one of ``k`` folds, grouping by source_id.

    Greedy balanced bin-packing of sources (largest first) into the smallest fold.
    In WildBench each source has exactly one scenario, so this reduces to a balanced
    scenario partition -- recorded for parity with Bridge.
    """
    src_to_scens: dict[str, list[str]] = {}
    for sid, src in maps["s2src"].items():
        src_to_scens.setdefault(src, []).append(sid)

    rng = np.random.default_rng(seed)
    order = sorted(src_to_scens, key=lambda s: (-len(src_to_scens[s]), s))
    # deterministic shuffle so equal-size sources pack unbiased but reproducibly
    order = list(np.array(order)[rng.permutation(len(order))]) if order else order
    order = sorted(order, key=lambda s: -len(src_to_scens[s]))
    fold_load = np.zeros(k, dtype=int)
    scen_fold: dict[str, int] = {}
    src_fold: dict[str, int] = {}
    for src in order:
        f = int(np.argmin(fold_load))
        src_fold[src] = f
        for sid in src_to_scens[src]:
            scen_fold[sid] = f
        fold_load[f] += len(src_to_scens[src])
    return {
        "k": k, "seed": seed,
        "n_sources": len(src_to_scens),
        "n_scenarios": len(scen_fold),
        "unique_source_per_scenario": all(len(v) == 1 for v in src_to_scens.values()),
        "fold_sizes_scenarios": [int(x) for x in fold_load],
        "scenario_to_fold": scen_fold,
        "source_to_fold": src_fold,
    }


# ---------------------------------------------------------------------------
# held-out marginal log-loss (person k-fold) -- mirrors Bridge
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# CAT-pool exclusion mask (extreme_a) + masked-bank writer -- Phase 2a
# ---------------------------------------------------------------------------
#
# WildBench has NO affective safety-gate (Bridge's A3), so the CAT-pool exclusion is
# extreme_a ONLY: criteria whose fitted 1D discrimination is non-finite or |a| > EXTREME_A
# are excluded from CAT ADMINISTRATION + ability (theta) SCORING, but KEPT in the
# calibration fit and reporting. PROVISIONAL: this set may shift when ridge is swept in
# Phase 2b (a larger ridge drains extreme_a).


def build_exclusion_mask_extreme_a(fitted_1d_bank: Path) -> dict:
    """Fixed CAT-pool exclusion mask from the locked 1D fit: extreme_a criteria only."""
    extreme_ids: list[str] = []
    a_by: dict[str, float] = {}
    for line in Path(fitted_1d_bank).open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        disc = r["discrimination"]
        a = float(next(iter(disc.values())) or 0.0)
        cid = r["criterion_id"]
        a_by[cid] = a
        if (not np.isfinite(a)) or abs(a) > cm.EXTREME_A:
            extreme_ids.append(cid)
    return {
        "extreme_a_threshold": cm.EXTREME_A,
        "n_fitted": len(a_by),
        "n_extreme_a": len(extreme_ids),
        "n_excluded_in_fitted": len(extreme_ids),
        "excluded_ids": sorted(extreme_ids),
        "extreme_a_ids": sorted(extreme_ids),
    }


def write_masked_bank(in_bank: Path, out_bank: Path, exclude_ids: set[str]) -> int:
    """Copy a fitted-bank JSONL, omitting the excluded criteria (the CAT pool)."""
    exclude = set(exclude_ids)
    n = 0
    with Path(in_bank).open(encoding="utf-8") as fin, \
            Path(out_bank).open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["criterion_id"] in exclude:
                continue
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


# ---------------------------------------------------------------------------
# held-out marginal log-loss (person k-fold) -- mirrors Bridge
# ---------------------------------------------------------------------------


def marginal_logloss(Yte, Mte, A, b, grid, log_prior) -> float:
    """Mean per-cell held-out marginal log-loss under train params (lower=better).

    ``grid`` is (n_nodes, n_dims); ``log_prior`` (n_nodes,). Works for any dim count.
    """
    eta = A @ grid.T - b[:, None]
    logP, log1mP = log_expit(eta), log_expit(-eta)
    YM = np.where(Mte, Yte, 0.0)
    NM = np.where(Mte, 1.0 - Yte, 0.0)
    LL = YM @ logP + NM @ log1mP
    person_ll = logsumexp(LL + log_prior[None, :], axis=1)
    ncells = Mte.sum()
    return float(-person_ll.sum() / max(ncells, 1))


# ---------------------------------------------------------------------------
# EAP-posterior CAT stop rule (of-record) -- Phase 2b adoption
# ---------------------------------------------------------------------------
#
# The of-record WildBench CAT stop is the EAP POSTERIOR SD over the administered items on a
# fine theta grid (not the engine's online normal-approx SE). Because the adaptive SCENARIO
# SELECTION is independent of the stop target, we obtain each model's adaptive order by running
# the production engine to a forced length (``EAP_L_MAX``) and then apply the EAP stop POST-HOC:
# stop at the first scenario with n_scenarios >= floor AND EAP posterior SD <= target, else cap.
# This keeps the production engine + selection UNMODIFIED.

EAP_L_MAX = 40      # forced administration cap (scenarios) to source the adaptive order
EAP_TARGET = 0.12   # of-record SE target (posterior SD)
EAP_FLOOR = 8       # of-record min_scenarios floor


def eap_grid(nodes: int = 321, half: float = 8.0):
    """Fine uniform theta grid + standard-normal log-prior (normalised)."""
    gg = np.linspace(-half, half, nodes)
    lp = -0.5 * gg ** 2
    lp = lp - logsumexp(lp)
    return gg, lp


def eap_walk(order, yrow, col, scen_of, a, b, gg, lp):
    """Per-scenario cumulative EAP (mean, SD) along an administration order.

    ``order`` = criterion_ids in administration order; ``yrow`` = the model's responses
    (aligned to ``col``); ``a``/``b`` = 1-D discriminations/difficulties (aligned to ``col``).
    Returns a list of dicts {n_scen, n_crit, eap_mean, eap_sd, idx} recorded at each scenario
    boundary (idx = cumulative col indices administered so far).
    """
    ll_data = np.zeros(gg.size)
    out = []
    cur = None
    n_scen = 0
    n_crit = 0
    cum = []
    for cid in order:
        j = col.get(cid)
        if j is None:
            continue
        sid = scen_of.get(cid)
        if sid != cur:
            if cur is not None:
                post = np.exp(ll_data + lp - logsumexp(ll_data + lp))
                mean = float(post @ gg)
                sd = float(np.sqrt(max(post @ (gg ** 2) - mean ** 2, 0.0)))
                out.append({"n_scen": n_scen, "n_crit": n_crit, "eap_mean": mean,
                            "eap_sd": sd, "idx": np.array(cum, dtype=int)})
            cur = sid
            n_scen += 1
        eta = a[j] * gg - b[j]
        ll_data = ll_data + yrow[j] * log_expit(eta) + (1.0 - yrow[j]) * log_expit(-eta)
        n_crit += 1
        cum.append(j)
    if cur is not None:
        post = np.exp(ll_data + lp - logsumexp(ll_data + lp))
        mean = float(post @ gg)
        sd = float(np.sqrt(max(post @ (gg ** 2) - mean ** 2, 0.0)))
        out.append({"n_scen": n_scen, "n_crit": n_crit, "eap_mean": mean,
                    "eap_sd": sd, "idx": np.array(cum, dtype=int)})
    return out


def eap_stop_point(walk, floor=EAP_FLOOR, target=EAP_TARGET):
    """First scenario step meeting (n_scen>=floor AND eap_sd<=target); else last (capped)."""
    for step in walk:
        if step["n_scen"] >= floor and step["eap_sd"] <= target:
            return step, True
    return (walk[-1], False) if walk else (None, False)


def administer_eap(models, bank_path, matrix_path, scenarios_path, dims, a, b, col, scen_of,
                   Yfull, row_of, gg, lp, seed=20260729, floor=EAP_FLOOR, target=EAP_TARGET,
                   l_max=EAP_L_MAX, workers=6, runs_dir=None):
    """Administer the EAP-stop CAT for ``models``: one forced-long engine run + post-hoc stop.

    Returns {model: {order, scenarios_administered, criteria_administered, eap_sd, eap_mean,
    idx, hit_cap, precision_reached}}. ``a``/``b``/``col``/``scen_of``/``Yfull``/``row_of`` are
    aligned to the SAME bank as ``bank_path`` (the caller supplies them for the fold or full
    data). Selection is the production engine's; only the stop differs.
    """
    import scripts.scenario_cat_lib as scat  # lazy (heavy tutor_cat import)
    import shutil
    rd = runs_dir or str(Path(bank_path).parent / "_eap_runs")
    spec = scat.RunSpec(seed=seed, top_n=5, max_se=0.0, min_evals_per_skill=0,
                        min_scenarios=l_max, max_scenarios=l_max, selection="trace",
                        mode="cat", runs_dir=rd)
    res = scat.run_models(models, bank_path, matrix_path, scenarios_path, "clamp", dims,
                          spec, workers=workers)
    out = {}
    for r in res:
        m = r["model"]
        yrow = Yfull[row_of[m]]
        walk = eap_walk(r["order"], yrow, col, scen_of, a, b, gg, lp)
        step, reached = eap_stop_point(walk, floor, target)
        if step is None:
            out[m] = {"order": [], "scenarios_administered": 0, "criteria_administered": 0,
                      "eap_sd": float("nan"), "eap_mean": float("nan"),
                      "idx": np.array([], dtype=int), "hit_cap": True,
                      "precision_reached": False}
            continue
        k = step["n_crit"]
        out[m] = {"order": [c for c in r["order"] if c in col][:k],
                  "scenarios_administered": step["n_scen"],
                  "criteria_administered": step["n_crit"],
                  "eap_sd": step["eap_sd"], "eap_mean": step["eap_mean"],
                  "idx": step["idx"], "hit_cap": bool(not reached),
                  "precision_reached": bool(reached)}
    shutil.rmtree(rd, ignore_errors=True)
    return out


def marginal_logloss_independent(Yte, Mte, A, b, keepj, n_nodes=21, half=6.0) -> float:
    """Held-out per-cell marginal log-loss for an INDEPENDENT (one-hot, R=I) fit.

    Each cell is scored under its own dim's 1-D standard-normal prior; because dims are
    independent the total is the pooled per-cell mean over all dims' cells. Comparable
    to :func:`marginal_logloss` (same per-cell metric, standard-normal prior per axis).
    """
    axis = np.linspace(-half, half, n_nodes)
    logphi = -0.5 * axis ** 2
    logphi = logphi - logsumexp(logphi)
    n_dims = A.shape[1]
    tot_ll = 0.0
    tot_cells = 0
    for d in range(n_dims):
        rows = np.where(A[:, d] != 0)[0]
        if rows.size == 0:
            continue
        cols = keepj[rows]
        Ad = A[rows, d]
        bd = b[rows]
        Yd = Yte[:, cols]
        Md = Mte[:, cols]
        eta = Ad[:, None] * axis[None, :] - bd[:, None]     # (n_items_d, n_nodes)
        logP, log1mP = log_expit(eta), log_expit(-eta)
        YM = np.where(Md, Yd, 0.0)
        NM = np.where(Md, 1.0 - Yd, 0.0)
        LL = YM @ logP + NM @ log1mP                        # (n_models, n_nodes)
        person_ll = logsumexp(LL + logphi[None, :], axis=1)
        tot_ll += float(person_ll.sum())
        tot_cells += int(Md.sum())
    return float(-tot_ll / max(tot_cells, 1))
