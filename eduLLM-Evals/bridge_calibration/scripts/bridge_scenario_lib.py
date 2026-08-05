"""Shared helpers for the SCENARIO-LEVEL Bridge calibration study.

This is the scenario-level sibling of the item-level ``bridge_calibration/`` study.
Where the item-level study fit and administered ONE criterion at a time, this study
drives the REAL TutorBench scenario engine (``tutor_cat.engine.run_evaluation`` via
``scripts.scenario_cat_lib``), which administers a whole SCENARIO (all its criteria,
graded together as a testlet) per CAT step. Nothing here reimplements CAT selection or
the M2PL update -- those stay in ``tutor_cat`` / ``scripts.scenario_cat_lib``. The
per-criterion M2PL *fit* reuses ``calibrate_mirt.fit_m2pl_em`` verbatim; only the
Q-matrix assembly (Bridge's native 5-skill axis + collapses) and the fitted-bank
emission are Bridge-specific.

Locked design (per the rerun plan + user spec):
  * Bank = ALL 250 scenarios (NO source dedup). ``source_id`` is used ONLY for
    leakage-safe fold grouping (scenarios sharing a source go to the same fold).
  * A scenario is a testlet bundle of its per-criterion IRT items; per-criterion
    (a, b, q) drive the multidimensional ability/SE update. No polytomous collapse.
  * Criterion-level zero-variance (all-fail / all-pass) filtering applies within every
    fit and every fold, exactly as the item-level study did.
"""

from __future__ import annotations

import hashlib
import json
import os

# Pin BLAS threads BEFORE numpy imports downstream to avoid OpenBLAS oversubscription
# / memory crashes on this box (seen in prior heavy fits).
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import log_expit, logsumexp

# eduLLM-Evals root (…/bridge_calibration/scripts/this_file -> parents[2]).
ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import calibrate_partial as cp  # noqa: E402  (load_matrix / select_sparse / split_zero_variance / jsonl IO)
import calibrate_mirt as cm  # noqa: E402  (fit_m2pl_em / build_grid / base_log_weights / aic_bic)

# ---------------------------------------------------------------------------
# canonical Bridge inputs + skill axis
# ---------------------------------------------------------------------------

DEFAULT_MATRIX = ROOT / "bridgegrade" / "response_matrix.csv"
DEFAULT_RUBRICS = ROOT / "data" / "Bridge" / "rubrics.jsonl"
DEFAULT_SCENARIOS = ROOT / "data" / "Bridge" / "scenarios.jsonl"

# Bridge's native 5-skill axis (first-seen order in rubrics.jsonl q_mapping).
BRIDGE_SKILLS = ["diagnosis", "strategy", "math", "communication", "affective"]

# Dimensionality structures to sweep (1..5 latent skills). Each entry maps a modeled
# dimension label -> the native skills OR'd into it. Mirrors the item-level
# bridge_dimensionality.py grouping choices so the 1..5 comparison is apples-to-apples.
STRUCTURES: dict[str, dict[str, list[str]]] = {
    "overall_1d": {"overall": list(BRIDGE_SKILLS)},
    "cognitive_vs_relational_2d": {
        "cognitive": ["diagnosis", "strategy", "math"],
        "relational": ["communication", "affective"],
    },
    "correlation_3d": {
        "diag_strat": ["diagnosis", "strategy"],
        "math": ["math"],
        "comm_affect": ["communication", "affective"],
    },
    "merge_affect_comm_4d": {
        "diagnosis": ["diagnosis"],
        "strategy": ["strategy"],
        "math": ["math"],
        "comm_affect": ["communication", "affective"],
    },
    "full_5d": {s: [s] for s in BRIDGE_SKILLS},
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
    for rec in cp.read_jsonl(scenarios_path):
        sid = rec.get("scenario_id")
        if sid is None:
            continue
        s2src[sid] = str(rec.get("source_id", sid))
        scen_criterion_ids[sid] = list(rec.get("criterion_ids", []))
    return {
        "c2s": c2s, "qmap": qmap, "crit_text": crit_text, "primary": primary,
        "criticality": criticality, "s2src": s2src,
        "scen_criterion_ids": scen_criterion_ids,
    }


# ---------------------------------------------------------------------------
# matrix prep + zero-variance filtering (identical policy to the item study)
# ---------------------------------------------------------------------------


def prepare_matrix(matrix_path: Path):
    """Load matrix, drop empty rows/cols + zero-variance criteria (over observed cells).

    Returns (Y, M, items, models, drop_info). NO source dedup -- the bank is all 250
    scenarios; source_id is only a fold-grouping key (see :func:`source_fold_groups`).
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
# fit + fitted-bank emission (schema consumed by scripts.scenario_cat_lib)
# ---------------------------------------------------------------------------


def fit_structure(Y, M, Q, grid, ridge, estimate_corr, max_iter=200, tol=1e-4):
    """Fit a confirmatory M2PL for one structure over the items that load on >=1 dim."""
    keepj = np.where(Q.sum(axis=1) > 0)[0]
    Qk = Q[keepj]
    fit = cm.fit_m2pl_em(Y[:, keepj], M[:, keepj], Qk, grid,
                         estimate_corr=estimate_corr, ridge=ridge,
                         max_iter=max_iter, tol=tol)
    return fit, keepj


def matrix_sha256(matrix_path: Path) -> str:
    return hashlib.sha256(Path(matrix_path).read_bytes()).hexdigest()


def write_fitted_bank(out_path: Path, items: list[str], keepj: np.ndarray,
                      A: np.ndarray, b: np.ndarray, Q: np.ndarray,
                      dim_labels: list[str], maps: dict, matrix_path: Path,
                      structure_name: str, ridge: float, grid: int) -> int:
    """Emit a ``*_fitted.jsonl`` bank in the schema scenario_cat_lib expects.

    One record per fitted criterion: modeled-skill-keyed ``discrimination`` +
    ``q_modeled``, scalar ``difficulty``, scenario_id, and an
    ``irt_params.provenance.matrix_sha256`` cross-checkable against the matrix.
    """
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
# leakage-safe source -> fold grouping (scenarios sharing a source share a fold)
# ---------------------------------------------------------------------------


def source_fold_groups(maps: dict, k: int = 5, seed: int = 20260729) -> dict:
    """Assign every scenario to one of ``k`` folds, grouping by source_id.

    Both scenarios sharing a source_id always land in the same fold (leakage-safe).
    Greedy balanced bin-packing of sources (largest first) into the currently
    smallest fold. Returns a dict with the scenario->fold map + fold sizes.
    """
    src_to_scens: dict[str, list[str]] = {}
    for sid, src in maps["s2src"].items():
        src_to_scens.setdefault(src, []).append(sid)

    rng = np.random.default_rng(seed)
    sources = sorted(src_to_scens, key=lambda s: (-len(src_to_scens[s]), s))
    # light shuffle within equal-size groups for reproducible-but-unbiased packing
    fold_load = np.zeros(k, dtype=int)
    scen_fold: dict[str, int] = {}
    src_fold: dict[str, int] = {}
    for src in sources:
        f = int(np.argmin(fold_load))
        src_fold[src] = f
        for sid in src_to_scens[src]:
            scen_fold[sid] = f
        fold_load[f] += len(src_to_scens[src])
    _ = rng  # reserved for future randomized tie-breaks; deterministic for now
    return {
        "k": k, "seed": seed,
        "n_sources": len(src_to_scens),
        "n_scenarios": len(scen_fold),
        "fold_sizes_scenarios": [int(x) for x in fold_load],
        "scenario_to_fold": scen_fold,
        "source_to_fold": src_fold,
    }


# ---------------------------------------------------------------------------
# CAT-pool exclusion mask (A3 safety-gate + extreme_a) -- Phase 2a
# ---------------------------------------------------------------------------
#
# Locked decision: A3 (criterion_code "A3", the per-scenario affective safety-gate in the
# c15 slot) and the extreme_a criteria (|a| > cm.EXTREME_A or non-finite in the locked 1D
# fit) are excluded from CAT ADMINISTRATION and ability (theta) SCORING, but KEPT in the
# calibration fit and in reporting. The mask is a fixed set of criterion_ids applied
# uniformly everywhere the CAT pool is used (sweep, SE_param admin sets, CV fold banks).


def load_criterion_codes(rubrics_path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for rec in cp.read_jsonl(rubrics_path):
        cid = rec.get("criterion_id")
        if cid is not None:
            out[cid] = rec.get("criterion_code")
    return out


def build_exclusion_mask(rubrics_path: Path, fitted_1d_bank: Path,
                         a3_code: str = "A3") -> dict:
    """Fixed CAT-pool exclusion mask from the locked 1D fit.

    Excludes (a) every criterion whose criterion_code == ``a3_code`` (the affective
    safety gate) and (b) every criterion whose fitted 1D discrimination is non-finite or
    |a| > cm.EXTREME_A. Returns the excluded id set + a breakdown for documentation.
    """
    codes = load_criterion_codes(rubrics_path)
    a3_ids = {cid for cid, code in codes.items() if code == a3_code}

    extreme_ids: set[str] = set()
    a_by: dict[str, float] = {}
    for line in Path(fitted_1d_bank).open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        # 1D bank has a single modeled dim; take whichever key it uses.
        disc = r["discrimination"]
        a = float(next(iter(disc.values())) or 0.0)
        cid = r["criterion_id"]
        a_by[cid] = a
        if (not np.isfinite(a)) or abs(a) > cm.EXTREME_A:
            extreme_ids.add(cid)

    fitted_ids = set(a_by)
    excluded = (a3_ids | extreme_ids) & fitted_ids
    return {
        "a3_code": a3_code,
        "extreme_a_threshold": cm.EXTREME_A,
        "n_a3_total": len(a3_ids),
        "n_a3_in_fitted": len(a3_ids & fitted_ids),
        "n_extreme_a": len(extreme_ids),
        "n_excluded_in_fitted": len(excluded),
        "excluded_ids": sorted(excluded),
        "a3_ids": sorted(a3_ids),
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
# held-out marginal log-loss (person k-fold) -- mirrors bridge_dimensionality.py
# ---------------------------------------------------------------------------


def marginal_logloss(Yte, Mte, A, b, grid, log_prior) -> float:
    """Mean per-cell held-out marginal log-loss under train params (lower=better)."""
    eta = A @ grid.T - b[:, None]
    logP, log1mP = log_expit(eta), log_expit(-eta)
    YM = np.where(Mte, Yte, 0.0)
    NM = np.where(Mte, 1.0 - Yte, 0.0)
    LL = YM @ logP + NM @ log1mP
    person_ll = logsumexp(LL + log_prior[None, :], axis=1)
    ncells = Mte.sum()
    return float(-person_ll.sum() / max(ncells, 1))
