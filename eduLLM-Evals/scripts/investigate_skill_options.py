"""Directional investigation of two candidate skill-definition changes for the
tutor-skill IRT model, run on the ONLY graded data available now: the 1/4 judge
run over the OLD (pre-curation) criteria (``staging/response_matrix.csv``), whose
columns are keyed by OLD ``criterion_id`` and therefore mapped by the OLD Q-matrix
``data/rubrics_qmatrix_final.jsonl`` (NOT the curated file).

Two hypotheses
--------------
H1  Should the normalized "presentation" criterion (added once per scenario in
    curation) become its own latent SKILL DIMENSION -- possibly REPLACING
    "diagnosis"?
H2  If "content" were reframed as "explanation" (explanation quality, not raw
    correctness), would "diagnosis" become a more standalone/separable dimension?

The curated one-per-scenario presentation criterion was NOT graded in this run.
BUT the OLD run DID grade the granular format/persona/structure orphan criteria
that curation later consolidated into "presentation". So we build a PRESENTATION
PROXY from the OLD all-zero-Q style/format orphans (organization/markdown/LaTeX +
persona/second-person), EXCLUDING affect_motivation (a different candidate skill).

*** EVERYTHING HERE IS DIRECTIONAL. ***
N is ~28 usable models (a biased small-model subsample) on a ~1/4-filled matrix.
Any dimensionality/factor/latent-correlation result is a directional read, not a
decision. AIC/BIC at this N are power artifacts and are deliberately NOT used to
adjudicate. The definitive test is the FULL curated-criteria matrix (see the
pre-registration section of the memo). This script never edits IRT params, the
Q-matrix, or any rubric/data file; it only writes reports under ``staging/``.

Outputs
-------
* ``staging/skill_options_report.json`` -- full structured results.
* ``staging/skill_options_report.csv``  -- per-model group pass-rate table.
* ``staging/skill_options_corr.csv``    -- group-score correlation matrix.

Usage
-----
    python scripts/investigate_skill_options.py
    python scripts/investigate_skill_options.py --grid 5 --min-items 3
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_module(name: str, rel: str):
    path = ROOT / rel
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, mod)
    spec.loader.exec_module(mod)
    return mod


cp = _load_module("calibrate_partial", "scripts/calibrate_partial.py")
mirt = _load_module("calibrate_mirt", "scripts/calibrate_mirt.py")

SKILLS = ("content", "diagnosis", "scaffolding")

DEFAULT_MATRIX = ROOT / "staging" / "response_matrix.csv"
DEFAULT_RUBRICS = ROOT / "data" / "TutorBench" / "rubrics_qmatrix_final.jsonl"
DEFAULT_ORPHANS = ROOT / "staging" / "orphan_criteria.csv"
DEFAULT_OUT_DIR = ROOT / "staging"

# Persona / second-person / conversational-delivery patterns. These sit alongside
# the orphan classifier's ``organization_structure`` bucket (markdown/LaTeX/layout)
# to form the presentation proxy. Kept narrow and, like the org bucket, only used
# on all-zero-Q items with NO affect_motivation hit.
_PERSONA_PATTERNS = [
    r"second[-\s]person",
    r"\bpersona\b",
    r"conversational",
    r"address(?:es|ing)?\s+the\s+student\s+(?:directly|by\s+name)",
    r"speak\w*\s+directly\s+to\s+the\s+student",
    r"refer\w*\s+to\s+the\s+student\s+as\s+\"?you\"?",
    r"uses?\s+(?:the\s+)?second[-\s]person",
    r"\bgreet\w*\b",
]
_PERSONA_RE = re.compile("|".join(_PERSONA_PATTERNS), re.IGNORECASE)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# item sets
# ---------------------------------------------------------------------------


def load_q_and_primary(rubrics_path: Path):
    """criterion_id -> (q3 tuple, primary_skill) from the OLD frozen rubric bank."""
    q_by: dict[str, tuple[int, int, int]] = {}
    prim_by: dict[str, str] = {}
    for rec in cp.read_jsonl(rubrics_path):
        cid = rec.get("criterion_id")
        qm = rec.get("q_mapping")
        if cid is None or not isinstance(qm, dict):
            continue
        q_by[cid] = (int(qm.get("content", 0)), int(qm.get("diagnosis", 0)),
                     int(qm.get("scaffolding", 0)))
        prim_by[cid] = rec.get("primary_skill")
    return q_by, prim_by


def build_item_sets(orphans: pd.DataFrame, q_by, prim_by) -> dict:
    """Skill/primary sets from the Q-matrix + the PRESENTATION proxy from orphans."""
    sets: dict[str, set[str]] = {
        "content_loaded": set(), "diagnosis_loaded": set(), "scaffolding_loaded": set(),
        "content_only": set(), "diagnosis_only": set(), "scaffolding_only": set(),
        "content_diagnosis": set(),
        "content_primary": set(), "diagnosis_primary": set(),
    }
    for cid, (c, d, s) in q_by.items():
        if c:
            sets["content_loaded"].add(cid)
        if d:
            sets["diagnosis_loaded"].add(cid)
        if s:
            sets["scaffolding_loaded"].add(cid)
        if c and not d and not s:
            sets["content_only"].add(cid)
        if d and not c and not s:
            sets["diagnosis_only"].add(cid)
        if s and not c and not d:
            sets["scaffolding_only"].add(cid)
        if c and d and not s:
            sets["content_diagnosis"].add(cid)
        if prim_by.get(cid) == "content":
            sets["content_primary"].add(cid)
        elif prim_by.get(cid) == "diagnosis":
            sets["diagnosis_primary"].add(cid)

    # PRESENTATION proxy: all-zero-Q style/format orphans =
    #   organization_structure (markdown/LaTeX/layout) OR persona/second-person,
    #   with NO affect_motivation hit (affect is a DIFFERENT candidate skill).
    pres_org: set[str] = set()
    pres_persona: set[str] = set()
    affect_all_zero: set[str] = set()
    for _, r in orphans.iterrows():
        cid = r["criterion_id"]
        if r.get("q_pattern") != "all_zero":
            continue
        affect = int(r.get("hit_affect_motivation", 0) or 0)
        org = int(r.get("hit_organization_structure", 0) or 0)
        persona = int(bool(_PERSONA_RE.search(str(r.get("criterion", "")))))
        if affect:
            affect_all_zero.add(cid)
            continue
        if org:
            pres_org.add(cid)
        if persona:
            pres_persona.add(cid)
    sets["presentation_org_only"] = set(pres_org)
    sets["presentation_persona_only"] = set(pres_persona)
    sets["presentation_proxy"] = pres_org | pres_persona
    sets["affect_all_zero"] = affect_all_zero
    return sets


# ---------------------------------------------------------------------------
# correlation primitives (mirror analyze_collinearity conventions)
# ---------------------------------------------------------------------------


def phi_tet(x: np.ndarray, y: np.ndarray, min_overlap: int):
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < min_overlap:
        return np.nan, np.nan, n
    xi, yi = x[mask], y[mask]
    if xi.std() == 0 or yi.std() == 0:
        return np.nan, np.nan, n
    phi = float(np.corrcoef(xi, yi)[0, 1])
    a = float(np.sum((xi == 1) & (yi == 1))) + 0.5
    b = float(np.sum((xi == 1) & (yi == 0))) + 0.5
    c = float(np.sum((xi == 0) & (yi == 1))) + 0.5
    dd = float(np.sum((xi == 0) & (yi == 0))) + 0.5
    orat = (a * dd) / (b * c)
    r_tet = float(np.cos(np.pi / (1.0 + orat ** 0.75)))
    return phi, r_tet, n


def mean_pair_corr(mat: pd.DataFrame, cols_a, cols_b, min_overlap, max_pairs,
                   rng, same=False) -> dict:
    arr = {c: mat[c].to_numpy(dtype=float)
           for c in set(cols_a) | set(cols_b) if c in mat.columns}
    a = [c for c in cols_a if c in arr]
    b = [c for c in cols_b if c in arr]
    pairs = []
    if same:
        for i in range(len(a)):
            for j in range(i + 1, len(a)):
                pairs.append((a[i], a[j]))
    else:
        for ca in a:
            for cb in b:
                pairs.append((ca, cb))
    n_all = len(pairs)
    if n_all > max_pairs:
        idx = rng.choice(n_all, size=max_pairs, replace=False)
        pairs = [pairs[k] for k in idx]
    phis, tets, ns = [], [], []
    for ca, cb in pairs:
        phi, tet, n = phi_tet(arr[ca], arr[cb], min_overlap)
        if np.isfinite(phi):
            phis.append(phi)
            ns.append(n)
        if np.isfinite(tet):
            tets.append(tet)
    return {
        "n_pairs_total": n_all,
        "n_pairs_evaluated": len(pairs),
        "n_pairs_usable": len(phis),
        "mean_phi": round(float(np.mean(phis)), 4) if phis else None,
        "median_phi": round(float(np.median(phis)), 4) if phis else None,
        "mean_tetrachoric_approx": round(float(np.mean(tets)), 4) if tets else None,
        "median_overlap_n": int(np.median(ns)) if ns else 0,
    }


def per_model_scores(mat: pd.DataFrame, sets: dict, group_names, min_items: int):
    """model x group pass-rate frame (NaN where a model has < min_items in group)."""
    data = {}
    for g in group_names:
        cols = [c for c in sets[g] if c in mat.columns]
        if not cols:
            data[g] = pd.Series(np.nan, index=mat.index)
            continue
        block = mat[cols]
        rates = []
        for model in mat.index:
            row = block.loc[model]
            rates.append(float(row.mean(skipna=True))
                         if int(row.notna().sum()) >= min_items else np.nan)
        data[g] = pd.Series(rates, index=mat.index)
    return pd.DataFrame(data)


def pairwise_corr(frame: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """Pairwise-complete correlation (keeps N up at tiny sample sizes)."""
    return frame.corr(method=method, min_periods=3)


# ---------------------------------------------------------------------------
# lightweight EFA (pure numpy: PCA on correlation matrix + varimax)
# ---------------------------------------------------------------------------


def varimax(loadings: np.ndarray, gamma: float = 1.0, q: int = 50, tol: float = 1e-6):
    p, k = loadings.shape
    if k < 2:
        return loadings
    R = np.eye(k)
    d = 0.0
    for _ in range(q):
        Lr = loadings @ R
        u, s, vt = np.linalg.svd(
            loadings.T @ (Lr ** 3 - (gamma / p) * Lr @ np.diag(np.diag(Lr.T @ Lr)))
        )
        R = u @ vt
        d_old = d
        d = float(np.sum(s))
        if d_old != 0 and d / d_old < 1 + tol:
            break
    return loadings @ R


def efa_from_corr(corr: pd.DataFrame, n_factors: int) -> dict:
    """PCA-based factor loadings (varimax) from a correlation matrix.

    At N~=28 this is a DIRECTIONAL structure read only. Returns eigenvalues,
    varimax-rotated loadings, and per-variable dominant factor.
    """
    C = corr.to_numpy(dtype=float)
    C = np.nan_to_num(C, nan=0.0)
    np.fill_diagonal(C, 1.0)
    vals, vecs = np.linalg.eigh(C)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    k = min(n_factors, (vals > 0).sum())
    load = vecs[:, :k] * np.sqrt(np.clip(vals[:k], 0, None))
    load_r = varimax(load) if k >= 2 else load
    labels = list(corr.columns)
    loadings = {labels[i]: [round(float(v), 3) for v in load_r[i]]
                for i in range(len(labels))}
    dominant = {labels[i]: int(np.argmax(np.abs(load_r[i]))) for i in range(len(labels))}
    return {
        "n_factors": int(k),
        "eigenvalues": [round(float(v), 4) for v in vals[: max(6, n_factors + 2)]],
        "n_eigen_ge_1": int(np.sum(vals >= 1.0)),
        "loadings_varimax": loadings,
        "dominant_factor": dominant,
    }


# ---------------------------------------------------------------------------
# confirmatory M2PL with a custom Q (adds a presentation dim)
# ---------------------------------------------------------------------------


def build_YMQ(mat: pd.DataFrame, qrow_of):
    """Build (Y, M, Q, items) from the sparse/variant block using a custom Q row fn.

    ``qrow_of(cid) -> list[int] | None``. Items whose Q row is all-zero (or None)
    are dropped (unfittable under confirmatory masking). Holes are preserved.
    """
    sub, _ = cp.select_sparse(mat)
    kept, _af, _ap = cp.split_zero_variance(sub)
    sub = sub[kept]
    sub = sub[sub.notna().any(axis=1)]
    rows, items = [], []
    for c in sub.columns:
        q = qrow_of(c)
        if q is None or int(sum(q)) == 0:
            continue
        rows.append(q)
        items.append(c)
    if not rows:
        return None
    Q = np.array(rows, dtype=int)
    sub = sub[items]
    sub = sub[sub.notna().any(axis=1)]
    Y = np.nan_to_num(sub.to_numpy(dtype=float), nan=0.0)
    M = sub.notna().to_numpy()
    return Y, M, Q, items, int(sub.shape[0])


def fit_latent_corr(mat, qrow_of, labels, grid, max_iter):
    built = build_YMQ(mat, qrow_of)
    if built is None:
        return {"available": False, "reason": "no fittable items"}
    Y, M, Q, items, n_persons = built
    res = mirt.fit_m2pl_em(Y, M, Q, grid, estimate_corr=True,
                           ridge=1e-3, max_iter=max_iter, tol=1e-4)
    R = np.asarray(res["R"])
    corr = {}
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            corr[f"{labels[i]}__{labels[j]}"] = round(float(R[i, j]), 4)
    return {
        "available": True,
        "labels": labels,
        "n_items_fit": len(items),
        "n_persons_fit": n_persons,
        "q_pattern_counts": {
            "".join(map(str, row)): int((Q == row).all(axis=1).sum())
            for row in np.unique(Q, axis=0)
        },
        "latent_correlation_matrix": np.round(R, 4).tolist(),
        "offdiag": corr,
        "converged": bool(res["converged"]),
        "n_iter": int(res["n_iter"]),
        "loglik": round(float(res["loglik"]), 2),
        "n_params": int(res["n_params"]),
    }


# ---------------------------------------------------------------------------
# analyses
# ---------------------------------------------------------------------------


def variance_reliability(mat, sets, min_items, min_overlap, max_pairs, rng) -> dict:
    """H1.1 -- presentation-proxy spread across models + internal coherence."""
    out = {}
    for g in ("presentation_proxy", "presentation_org_only",
              "presentation_persona_only", "affect_all_zero"):
        cols = [c for c in sets[g] if c in mat.columns]
        block = mat[cols] if cols else None
        rates = []
        if block is not None:
            for model in mat.index:
                row = block.loc[model]
                if int(row.notna().sum()) >= min_items:
                    rates.append(float(row.mean(skipna=True)))
        arr = np.array(rates, dtype=float)
        rec = {
            "n_items_total": len(sets[g]),
            "n_items_gradeable": len(cols),
            "n_models_used": len(rates),
            "mean_passrate": round(float(arr.mean()), 4) if len(arr) else None,
            "std_passrate": round(float(arr.std(ddof=1)), 4) if len(arr) > 1 else None,
            "min_passrate": round(float(arr.min()), 4) if len(arr) else None,
            "max_passrate": round(float(arr.max()), 4) if len(arr) else None,
        }
        if len(arr) > 1:
            m = rec["mean_passrate"]
            rec["spread_flag"] = ("near_floor" if m <= 0.1 else
                                  "near_ceiling" if m >= 0.9 else "has_spread")
            if rec["std_passrate"] is not None and rec["std_passrate"] < 0.1:
                rec["spread_flag"] += "|low_variance"
        out[g] = rec

    coherence = {}
    for g in ("presentation_proxy", "presentation_org_only"):
        cols = list(sets[g])
        coherence[g] = mean_pair_corr(mat, cols, cols, min_overlap, max_pairs,
                                      rng, same=True)
    # reference internal coherence for the existing skills
    for g in ("content_primary", "diagnosis_primary", "scaffolding_loaded"):
        cols = list(sets[g])
        coherence[g] = mean_pair_corr(mat, cols, cols, min_overlap, max_pairs,
                                      rng, same=True)
    return {"per_model_spread": out, "internal_coherence_inter_item": coherence}


def separability(mat, sets, min_items, min_overlap, max_pairs, rng) -> dict:
    """H1.2/H1.3 -- is presentation a factor distinct from the 3 skills?"""
    groups = ["content_primary", "diagnosis_primary", "scaffolding_loaded",
              "presentation_proxy"]
    scores = per_model_scores(mat, sets, groups, min_items)
    pear = pairwise_corr(scores, "pearson")
    spear = pairwise_corr(scores, "spearman")
    n_by_pair = {}
    for i, a in enumerate(groups):
        for b in groups[i + 1:]:
            n = int((scores[a].notna() & scores[b].notna()).sum())
            n_by_pair[f"{a}__{b}"] = n

    cross = {}
    for other in ("content_primary", "diagnosis_primary", "scaffolding_loaded"):
        cross[f"presentation_x_{other}"] = mean_pair_corr(
            mat, list(sets["presentation_proxy"]), list(sets[other]),
            min_overlap, max_pairs, rng)

    # EFA on the group-score correlation matrix (listwise for a stable matrix).
    listwise = scores.dropna()
    efa = {}
    if listwise.shape[0] >= 4:
        cmat = listwise.corr(method="pearson")
        for nf in (2, 3):
            efa[f"{nf}_factor"] = efa_from_corr(cmat, nf)
        efa["n_models_listwise"] = int(listwise.shape[0])
    else:
        efa = {"available": False, "reason": f"only {listwise.shape[0]} listwise models"}

    return {
        "group_score_pearson": _corr_to_dict(pear),
        "group_score_spearman": _corr_to_dict(spear),
        "group_score_pair_n": n_by_pair,
        "presentation_cross_item_corr": cross,
        "group_score_efa": efa,
        "_scores_frame": scores,
    }


def h2_diagnosis_content(mat, sets, min_items, min_overlap, max_pairs, rng, grid,
                         max_iter) -> dict:
    """H2 -- unconstrained factor structure & content<->diagnosis separability."""
    groups = ["content_only", "content_diagnosis", "diagnosis_only",
              "scaffolding_loaded", "presentation_proxy"]
    scores = per_model_scores(mat, sets, groups, min_items)
    listwise = scores.dropna()
    efa = {}
    if listwise.shape[0] >= 4:
        cmat = listwise.corr(method="pearson")
        for nf in (2, 3, 4):
            efa[f"{nf}_factor"] = efa_from_corr(cmat, nf)
        efa["n_models_listwise"] = int(listwise.shape[0])
        efa["group_score_pearson"] = _corr_to_dict(cmat)
    else:
        efa = {"available": False, "reason": f"only {listwise.shape[0]} listwise models"}

    # Item-level cross corr: do diagnosis-primary items track content-primary items?
    cross = {
        "content_primary_x_diagnosis_primary": mean_pair_corr(
            mat, list(sets["content_primary"]), list(sets["diagnosis_primary"]),
            min_overlap, max_pairs, rng),
        "within_content_primary": mean_pair_corr(
            mat, list(sets["content_primary"]), list(sets["content_primary"]),
            min_overlap, max_pairs, rng, same=True),
        "within_diagnosis_primary": mean_pair_corr(
            mat, list(sets["diagnosis_primary"]), list(sets["diagnosis_primary"]),
            min_overlap, max_pairs, rng, same=True),
        "diagnosis_only_x_content_only": mean_pair_corr(
            mat, list(sets["diagnosis_only"]), list(sets["content_only"]),
            min_overlap, max_pairs, rng),
    }

    # Per-model diagnosis-only vs content pass-rate correlation (robust read).
    dc = per_model_scores(mat, sets, ["content_only", "diagnosis_only"], min_items)
    dc_lw = dc.dropna()
    diag_vs_content = {
        "n_models": int(dc_lw.shape[0]),
        "pearson_r": (round(float(dc_lw["content_only"].corr(dc_lw["diagnosis_only"])), 4)
                      if dc_lw.shape[0] >= 3 else None),
    }
    return {
        "group_score_efa": efa,
        "item_cross_corr": cross,
        "diagnosis_only_vs_content_only_permodel": diag_vs_content,
    }


def _corr_to_dict(corr: pd.DataFrame) -> dict:
    out = {}
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            v = corr.loc[a, b]
            out[f"{a}__{b}"] = round(float(v), 4) if pd.notna(v) else None
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    p.add_argument("--rubrics", type=Path, default=DEFAULT_RUBRICS)
    p.add_argument("--orphans", type=Path, default=DEFAULT_ORPHANS)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--min-items", type=int, default=3,
                   help="min observed items in a group for a model to contribute.")
    p.add_argument("--min-overlap", type=int, default=10,
                   help="min common respondents for an item-pair correlation.")
    p.add_argument("--max-pairs", type=int, default=40000,
                   help="cap on item pairs evaluated per correlation cell.")
    p.add_argument("--grid", type=int, default=5,
                   help="Gauss-Hermite nodes/dim for the directional M2PL latent-corr.")
    p.add_argument("--max-iter", type=int, default=80, help="EM max iterations.")
    p.add_argument("--seed", type=int, default=20260728)
    p.add_argument("--skip-mirt", action="store_true",
                   help="skip the (slow) confirmatory M2PL latent-corr fits.")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    mat = cp.load_matrix(args.matrix)
    q_by, prim_by = load_q_and_primary(args.rubrics)
    orphans = pd.read_csv(args.orphans)
    sets = build_item_sets(orphans, q_by, prim_by)

    print("=" * 74)
    print("SKILL-DEFINITION OPTIONS INVESTIGATION (DIRECTIONAL, N tiny)")
    print("=" * 74)
    print(f"matrix: {mat.shape[0]} models x {mat.shape[1]} criteria "
          f"(fill {mat.notna().to_numpy().mean():.3f})")
    print("presentation proxy items: "
          f"total={len(sets['presentation_proxy'])} "
          f"(org={len(sets['presentation_org_only'])}, "
          f"persona={len(sets['presentation_persona_only'])}); "
          f"gradeable={len([c for c in sets['presentation_proxy'] if c in mat.columns])}")

    h1_var = variance_reliability(mat, sets, args.min_items, args.min_overlap,
                                  args.max_pairs, rng)
    h1_sep = separability(mat, sets, args.min_items, args.min_overlap,
                          args.max_pairs, rng)
    scores_frame = h1_sep.pop("_scores_frame")
    h2 = h2_diagnosis_content(mat, sets, args.min_items, args.min_overlap,
                              args.max_pairs, rng, args.grid, args.max_iter)

    # Confirmatory M2PL latent correlations (DIRECTIONAL; not identifiable at N~=28).
    mirt_block = {"note": "skipped"}
    if not args.skip_mirt:
        pres = sets["presentation_proxy"]

        def q_current(cid):
            q = q_by.get(cid)
            return list(q) if q else None

        def q_full4(cid):
            q = q_by.get(cid, (0, 0, 0))
            return [q[0], q[1], q[2], 1 if cid in pres else 0]

        def q_alt(cid):  # {content+diagnosis, scaffolding, presentation}
            q = q_by.get(cid, (0, 0, 0))
            return [1 if (q[0] or q[1]) else 0, q[2], 1 if cid in pres else 0]

        print("\nfitting DIRECTIONAL confirmatory M2PL latent correlations ...")
        mirt_block = {
            "CAVEAT": ("N~=28 -> multidim M2PL NOT identifiable; these latent "
                       "correlations are a directional structure read ONLY."),
            "current_3d_content_diagnosis_scaffolding": fit_latent_corr(
                mat, q_current, list(SKILLS), args.grid, args.max_iter),
            "full_4d_add_presentation": fit_latent_corr(
                mat, q_full4, [*SKILLS, "presentation"], args.grid, args.max_iter),
            "alt_3d_replace_diagnosis": fit_latent_corr(
                mat, q_alt, ["content+diagnosis", "scaffolding", "presentation"],
                args.grid, args.max_iter),
        }

    report = {
        "generated_at": _utcnow(),
        "headline_caveat": (
            "DIRECTIONAL ONLY. Graded data = 1/4 judge run on OLD pre-curation "
            "criteria (staging/response_matrix.csv), mapped by OLD Q-matrix "
            "data/rubrics_qmatrix_final.jsonl. N~=28 biased small-model subsample. "
            "The curated one-per-scenario presentation criterion was NOT graded; "
            "presentation is a PROXY from OLD all-zero-Q org/format + persona "
            "orphans (affect excluded). No AIC/BIC adjudication. Definitive test = "
            "full curated-criteria matrix (see memo pre-registration)."
        ),
        "matrix": {
            "n_models": int(mat.shape[0]),
            "n_criteria": int(mat.shape[1]),
            "fill_rate": round(float(mat.notna().to_numpy().mean()), 4),
        },
        "item_set_sizes": {k: len(v) for k, v in sets.items()},
        "item_set_gradeable": {
            k: len([c for c in v if c in mat.columns]) for k, v in sets.items()
        },
        "H1_presentation": {
            "reliability_variance": h1_var,
            "separability": h1_sep,
        },
        "H2_content_explanation_diagnosis": h2,
        "confirmatory_mirt_latent_corr_directional": mirt_block,
        "params": {
            "min_items": args.min_items, "min_overlap": args.min_overlap,
            "max_pairs": args.max_pairs, "grid": args.grid, "max_iter": args.max_iter,
            "seed": args.seed,
        },
        "provenance": {"script": "scripts/investigate_skill_options.py",
                       "argv": sys.argv[1:]},
    }

    json_path = args.out_dir / "skill_options_report.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    scores_path = args.out_dir / "skill_options_report.csv"
    scores_frame.round(4).to_csv(scores_path)

    corr_path = args.out_dir / "skill_options_corr.csv"
    scores_frame.corr(method="pearson", min_periods=3).round(4).to_csv(corr_path)

    # console summary
    print("\n--- H1 presentation spread (per model) ---")
    for g, r in h1_var["per_model_spread"].items():
        print(f"  {g:26s} n_models={r['n_models_used']:>2}  "
              f"mean={r['mean_passrate']}  std={r['std_passrate']}  "
              f"flag={r.get('spread_flag')}")
    print("\n--- H1 separability (group-score pearson) ---")
    for k, v in h1_sep["group_score_pearson"].items():
        print(f"  {k:52s} r={v}")
    print("\n--- H1 presentation x skill (item-level mean phi/tet) ---")
    for k, v in h1_sep["presentation_cross_item_corr"].items():
        print(f"  {k:40s} phi={v['mean_phi']} tet~={v['mean_tetrachoric_approx']} "
              f"(usable={v['n_pairs_usable']})")
    print("\n--- H2 item cross corr ---")
    for k, v in h2["item_cross_corr"].items():
        print(f"  {k:40s} phi={v['mean_phi']} tet~={v['mean_tetrachoric_approx']} "
              f"(usable={v['n_pairs_usable']})")
    if not args.skip_mirt:
        print("\n--- directional M2PL latent correlations ---")
        for model_name, blk in mirt_block.items():
            if isinstance(blk, dict) and blk.get("available"):
                print(f"  [{model_name}] n_items={blk['n_items_fit']} "
                      f"n_persons={blk['n_persons_fit']} conv={blk['converged']}")
                for pair, v in blk["offdiag"].items():
                    print(f"      {pair:40s} r={v}")

    print(f"\nwrote {json_path}")
    print(f"wrote {scores_path}")
    print(f"wrote {corr_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
