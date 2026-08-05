"""Confirmatory MULTIDIMENSIONAL 2PL (M2PL) calibration from a response matrix
and an explicitly ordered Q-matrix skill axis.

Context
-------
This is the *deferred* multidimensional fit promised by
``scripts/calibrate_partial.py`` (which does the UNIDIMENSIONAL, preliminary 2PL).
It closes the audit gaps GAP-5/6/7: it estimates a genuinely multidimensional item
bank (per-skill discrimination loadings + a scalar difficulty) under the
CONFIRMATORY constraint that each item may only load on the skills its Q-matrix row
marks (``q_k == 1``); every ``q_k == 0`` loading is held at EXACTLY zero.

The fitter is PURE numpy/scipy (no torch / JAX / rpy2 / girth). It is portable to
Python 3.14 where those wheels are risky. The only optional dependency is
``factor_analyzer`` for the ``--efa`` scree diagnostic; it is guarded and skipped
cleanly when absent.

Model
-----
For person ``i`` with a K-dimensional latent ability ``theta_i ~ MVN(0, R)`` and
item ``j`` with loading vector ``a_j`` (a K-vector masked by the Q row) and scalar difficulty
``b_j``::

    P(y_ij = 1 | theta_i) = sigmoid( a_j . theta_i - b_j )

This reduces to the ordinary 2PL ``sigmoid(a (theta - b_loc))`` when the item loads
on a single dimension (with the offset ``b = a * b_loc``). ``b`` here is the natural
M2PL scalar *offset* difficulty (a.k.a. ``-d`` intercept), reported consistently for
both the uni and multi fits so their log-likelihoods are directly comparable.

Estimation
----------
Marginal maximum likelihood by the Bock--Aitkin EM algorithm over a FIXED
Gauss--Hermite quadrature grid (``--grid`` nodes per dimension; total
``grid**K`` nodes -- cost grows exponentially with the number of skills):

* E-step: posterior over the latent grid per person, using ONLY the observed
  (non-NaN) cells for that person -> holes are handled natively (marginalised).
* M-step: per item, a masked weighted-logistic Newton solve for the free loadings
  (only ``q_k == 1`` dims) plus the scalar difficulty, using the E-step expected
  counts at each grid node.

The latent correlation ``R`` starts at the identity. ``--estimate-latent-corr``
re-estimates the KxK latent *correlation* matrix from the posterior second moments
each EM iteration (variances fixed to 1 for identifiability; the loadings carry the
scale). The off-diagonals of ``R`` are the "are the 3 skills distinguishable?"
evidence: near +/-1 correlations mean the model is collapsing toward
unidimensionality.

Uni-vs-multi comparison
-----------------------
We ALSO fit the UNIDIMENSIONAL 2PL baseline. To keep the log-likelihood / AIC / BIC
comparison apples-to-apples (same marginal-ML likelihood, same native missing-data
handling) the baseline is the SAME EM machinery run with ``n_dims = 1`` and a single
free loading per item. When ``girth`` happens to be importable we ALSO run
``tutor_cat.mcq_irt.calibrate.fit_2pl`` as an auxiliary cross-check and report the
rank agreement of its ``a`` / ``b`` with our internal unidim fit (girth cannot
consume holes, so it is a dense-block cross-check only, never the primary baseline).

Identifiability WARNING
-----------------------
A multidimensional confirmatory M2PL needs a healthy person sample to be identified.
The script prints and records a
WARNING whenever the fitted person count is below ``--min-persons-identifiable``
(default 150). It still RUNS on tiny data (and on the synthetic tests) so the
machinery can be exercised now; just do not trust tiny-N numbers.

Outputs
-------
* ``staging/calibration_mirt.csv``     -- per fitted criterion: one ``a_<skill>``
  column for each configured skill (0 where q=0), b, n_persons, flags.
* ``staging/calibration_mirt_manifest.json`` -- method, grid, fit dimensions,
  dropped zero-variance items, loglik/AIC/BIC for uni & multi, latent corr, and the
  identifiability warning.
* ``--write-params`` -> a NEW file ``data/rubrics_qmatrix_mirt.jsonl`` with the
  masked K-vector ``discrimination``, scalar ``difficulty``, and
  ``irt_params.source = "calibrated-m2pl"`` + provenance. The frozen
  ``data/rubrics_qmatrix_final.jsonl`` is NEVER overwritten.

Usage
-----
    # coverage report only (no fit):
    python scripts/calibrate_mirt.py --report-only

    # TutorBench's historical 3-skill fit (7 nodes/dim = 343 nodes):
    python scripts/calibrate_mirt.py --estimate-latent-corr

    # InFoBench's 5-skill fit (3 nodes/dim = 243 nodes):
    python scripts/calibrate_mirt.py \
      --matrix runs/judge/InFoBench/response_matrix.csv \
      --rubrics data/InFoBench/rubrics.jsonl \
      --skills content,format,number,style,linguistic --grid 3

    # Compare a reduced two-dimensional InFoBench structure without rewriting Q:
    python scripts/calibrate_mirt.py \
      --matrix runs/judge/InFoBench/response_matrix.csv \
      --rubrics data/InFoBench/rubrics.jsonl \
      --source-skills content,format,number,style,linguistic \
      --dimensions semantic=content+style,constraints=format+number+linguistic \
      --structure-name infobench_2d --grid 5

    # smaller/faster grid + optional EFA scree:
    python scripts/calibrate_mirt.py --grid 5 --efa

    # also write a calibrated rubric COPY (new file, never overwrites final):
    python scripts/calibrate_mirt.py --estimate-latent-corr --write-params

    # skill-collapse decision instrument: merge content+diagnosis into ONE latent
    # dim (fit-time, in-memory Q transform ONLY) and get a 3-way uni/collapsed/full
    # comparison in ONE run (loglik, #params, AIC, BIC + winners):
    python scripts/calibrate_mirt.py --collapse content,diagnosis --estimate-latent-corr

    # show the plan without fitting:
    python scripts/calibrate_mirt.py --dry-run

Skill-collapse decision instrument
----------------------------------
The historical TutorBench content<->diagnosis collinearity question is answered by
three numbers read side by side. ``--collapse content,diagnosis`` fits the collapsed 2-dim model
ALONGSIDE the full 3-dim model (both share the unidimensional baseline), so a
SINGLE run reports unidimensional vs collapsed-2-dim vs full-3-dim (loglik, k,
AIC, BIC) and names the AIC/BIC winner. The collapse is a pure in-memory Q-matrix
transform (logical OR of the merged skills' columns per item); it NEVER edits
``data/rubrics_qmatrix_final.jsonl`` or the skill definitions. The full 3-dim run
(``--collapse`` OFF) still drives the CSV / --write-params output and yields the
3x3 latent correlation whose content<->diagnosis off-diagonal is the key
diagnostic.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, log_expit, logsumexp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tutor_cat import SKILLS as PACKAGE_SKILLS  # noqa: E402
from tutor_cat.skill_structure import SkillStructure, parse_dimension_spec  # noqa: E402

# Reuse the sibling script's IO / coverage / zero-variance helpers verbatim so the
# two calibrators stay behaviourally identical (same loader, same coverage report,
# same zero-variance definition, same manifest/provenance conventions).
_CP_PATH = ROOT / "scripts" / "calibrate_partial.py"
_spec = importlib.util.spec_from_file_location("calibrate_partial", _CP_PATH)
assert _spec and _spec.loader
cp = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("calibrate_partial", cp)
_spec.loader.exec_module(cp)

# Preserve the historical TutorBench axis by default. The CLI's explicit
# ``--skills`` value may replace these process-local globals for another bank.
SKILLS = tuple(PACKAGE_SKILLS)
N_SKILLS = len(SKILLS)

DEFAULT_MATRIX = ROOT / "staging" / "response_matrix.csv"
DEFAULT_RUBRICS = ROOT / "data" / "rubrics_qmatrix_final.jsonl"
DEFAULT_OUT_DIR = ROOT / "staging"
DEFAULT_MIRT_RUBRICS = ROOT / "data" / "rubrics_qmatrix_mirt.jsonl"

CALIBRATION_CSV_NAME = "calibration_mirt.csv"
CALIBRATION_MANIFEST_NAME = "calibration_mirt_manifest.json"

CALIBRATION_SOURCE = "calibrated-m2pl"

# Loadings above this are treated as unstable (thin-sample / near-separation).
EXTREME_A = 6.0

# The tensor-product quadrature cost is exponential in the number of skills. This
# guard keeps an omitted ``--grid`` on a 5-skill bank from silently requesting the
# historical 7**5 = 16,807-node fit. It can be overridden deliberately.
DEFAULT_MAX_GRID_NODES = 5_000
IGNORED_Q_KEYS = {"adaptation"}  # historical, intentionally outside tutor_cat.SKILLS


class CalibrationError(RuntimeError):
    pass


def configure_skills(value: str | None) -> tuple[str, ...]:
    """Set the ordered Q-matrix axis for this calibration process only."""
    global SKILLS, N_SKILLS
    if value is None:
        skills = tuple(PACKAGE_SKILLS)
    else:
        skills = tuple(part.strip() for part in value.split(",") if part.strip())
    if not skills:
        raise ValueError("--skills must contain at least one non-empty skill name")
    if len(set(skills)) != len(skills):
        raise ValueError(f"--skills contains duplicates: {list(skills)}")
    SKILLS = skills
    N_SKILLS = len(skills)
    return SKILLS


def load_matrix_strict(path: Path) -> pd.DataFrame:
    """Load a binary response matrix without silently coercing bad cells to holes."""
    if not path.is_file():
        raise FileNotFoundError(f"response matrix not found: {path}")
    try:
        with path.open(encoding="utf-8", newline="") as f:
            header = next(csv.reader(f), None)
    except OSError as e:
        raise FileNotFoundError(f"could not read response matrix {path}: {e}") from e
    if not header or header[0] != "model":
        raise CalibrationError("response matrix must begin with a 'model' column")
    duplicates = sorted(name for name, count in Counter(header).items() if count > 1)
    if duplicates:
        raise CalibrationError(f"response matrix has duplicate column names: {duplicates[:10]}")

    try:
        raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception as e:
        raise CalibrationError(f"could not parse response matrix CSV: {e}") from e
    if raw.empty:
        raise CalibrationError("response matrix has no model rows")
    models = raw["model"].astype(str).str.strip()
    if (models == "").any():
        raise CalibrationError("response matrix contains a blank model id")
    duplicate_models = sorted(models[models.duplicated(keep=False)].unique().tolist())
    if duplicate_models:
        raise CalibrationError(f"response matrix has duplicate model rows: {duplicate_models[:10]}")

    values = raw.drop(columns=["model"])
    allowed = {"", "0", "1", "0.0", "1.0"}
    bad: list[str] = []
    for col in values.columns:
        stripped = values[col].astype(str).str.strip()
        invalid = stripped[~stripped.isin(allowed)]
        for row_idx, value in invalid.head(max(0, 10 - len(bad))).items():
            bad.append(f"model={models.iloc[row_idx]!r}, criterion={col!r}, value={value!r}")
        if len(bad) >= 10:
            break
    if bad:
        raise CalibrationError(
            "response matrix contains non-binary, non-empty cells; examples: " + "; ".join(bad)
        )

    numeric = values.replace({"": np.nan, "0": 0.0, "1": 1.0, "0.0": 0.0, "1.0": 1.0})
    numeric = numeric.astype(float)
    numeric.index = models.tolist()
    numeric.index.name = "model"
    return numeric


def validate_matrix_bank_alignment(
    matrix: pd.DataFrame, q_by: dict[str, np.ndarray], require_complete: bool
) -> dict:
    """Check criterion identity across the response matrix and active Q bank."""
    matrix_cols = list(matrix.columns)
    matrix_set = set(matrix_cols)
    bank_set = set(q_by)
    unknown = [cid for cid in matrix_cols if cid not in bank_set]
    missing = [cid for cid in q_by if cid not in matrix_set]
    if unknown:
        raise CalibrationError(
            f"matrix has {len(unknown)} criterion column(s) absent from the active Q bank; "
            f"first: {unknown[:10]}"
        )
    if require_complete and missing:
        raise CalibrationError(
            f"matrix is missing {len(missing)} active bank criterion column(s); "
            f"first: {missing[:10]}"
        )
    return {
        "n_matrix_criteria": len(matrix_cols),
        "n_active_bank_criteria": len(q_by),
        "n_bank_criteria_missing_from_matrix": len(missing),
        "bank_criteria_missing_from_matrix": missing,
        "complete_bank_alignment": not missing,
    }


# ---------------------------------------------------------------------------
# Q-matrix
# ---------------------------------------------------------------------------


def load_q_matrix(
    rubrics_path: Path,
    structure: SkillStructure | None = None,
) -> dict[str, np.ndarray]:
    """Map criterion_id to a 0/1 Q row in configured modeled-skill order.

    With ``structure=None``, the configured ``SKILLS`` must be literal keys in the
    source bank (the historical behavior).  A ``SkillStructure`` instead OR-merges
    its source-skill columns into one or more modeled dimensions in memory.  The
    rubric file is never rewritten.
    """
    if not rubrics_path.is_file():
        raise FileNotFoundError(f"rubric bank not found: {rubrics_path}")
    q_by: dict[str, np.ndarray] = {}
    seen_keys: set[str] = set()
    positive_unconfigured: set[str] = set()
    for rec in cp.read_jsonl(rubrics_path):
        cid = rec.get("criterion_id")
        qmap = rec.get("q_mapping")
        if cid is None or not isinstance(qmap, dict):
            continue
        # Fit-exclusion mask (Gate B): criteria flagged ``exclude_from_fit`` stay IN
        # the bank but are held OUT of every M2PL fit (reverse-behaving / degenerate
        # items whose fitted loading is a numeric artifact, not real signal). Dropping
        # them here keeps them out of the assembled Q so downstream selection excludes
        # them; honoured by all future fits (incl. the 200-run) without touching math.
        if rec.get("exclude_from_fit"):
            continue
        if cid in q_by:
            raise CalibrationError(f"duplicate criterion_id in rubric bank: {cid}")
        expected_source = structure.source_skills if structure is not None else SKILLS
        missing = [skill for skill in expected_source if skill not in qmap]
        if missing:
            raise CalibrationError(
                f"criterion {cid} is missing configured q_mapping key(s) {missing}."
            )
        parsed_q: dict[str, int] = {}
        for key, value in qmap.items():
            if value not in (0, 1, "0", "1", False, True):
                raise CalibrationError(
                    f"criterion {cid} has non-binary q_mapping[{key!r}]={value!r}; "
                    "Q values must be 0 or 1."
                )
            parsed_q[str(key)] = int(value)
        seen_keys.update(str(key) for key in qmap)
        for key, value in parsed_q.items():
            if key not in expected_source and key not in IGNORED_Q_KEYS and value != 0:
                positive_unconfigured.add(str(key))
        source_row = np.array([parsed_q[s] for s in expected_source], dtype=int)
        q_by[cid] = (
            structure.transform_q(source_row[None, :])[0]
            if structure is not None
            else source_row
        )
    if positive_unconfigured:
        raise CalibrationError(
            "rubric bank has positively mapped Q dimensions that are not configured: "
            f"{sorted(positive_unconfigured)}. Pass the complete native axis with "
            "--skills, or with --source-skills when using --dimensions."
        )
    expected_source = structure.source_skills if structure is not None else SKILLS
    absent = [skill for skill in expected_source if skill not in seen_keys]
    if absent:
        raise CalibrationError(
            f"configured skill(s) absent from every q_mapping: {absent}; "
            f"source axis is {list(expected_source)}."
        )
    return q_by


def align_q_rows(
    columns: list[str], q_by: dict[str, np.ndarray]
) -> tuple[np.ndarray, list[str], list[str]]:
    """Build the Q matrix aligned to ``columns``.

    Returns (Q, aligned_columns, missing). Criteria absent from the bank, or whose
    Q row is all-zero (no skill assigned -> unfittable under confirmatory masking),
    are reported in ``missing`` and excluded from the fit.
    """
    rows: list[np.ndarray] = []
    aligned: list[str] = []
    missing: list[str] = []
    for c in columns:
        q = q_by.get(c)
        if q is None or int(q.sum()) == 0:
            missing.append(c)
            continue
        rows.append(q)
        aligned.append(c)
    Q = np.array(rows, dtype=int) if rows else np.zeros((0, N_SKILLS), dtype=int)
    return Q, aligned, missing


def collapse_q_matrix(
    Q: np.ndarray, collapse_skills, skills: list[str] | None = None
) -> tuple[np.ndarray, list[str], dict]:
    """Merge the named skills' Q columns into ONE combined latent dimension.

    This is a FIT-TIME, in-memory transform ONLY -- it never touches the rubric
    bank or the skill definitions on disk. The merged column is the logical OR of
    the 1s across the merged skills for each item (an item that loaded on ANY of
    the merged skills loads on the combined dimension). The latent dimension count
    is reduced by ``k - 1`` where ``k`` = number of skills merged.

    Column order in the returned matrix preserves the original ``skills`` order,
    with the merged skills replaced by a single combined column placed at the
    position of the FIRST merged skill (so a ``content,diagnosis`` collapse of the
    ``(content, diagnosis, scaffolding)`` layout yields columns
    ``[content+diagnosis, scaffolding]``).

    Returns (Q_collapsed, collapsed_labels, info). ``info`` records the merged
    skills, their original indices, the new labels, and the collapsed dim count.
    """
    if skills is None:
        skills = list(SKILLS)
    skills = list(skills)
    name_to_idx = {s: i for i, s in enumerate(skills)}

    requested = [s.strip() for s in collapse_skills if str(s).strip()]
    unknown = [s for s in requested if s not in name_to_idx]
    if unknown:
        raise CalibrationError(
            f"--collapse names unknown skill(s) {unknown}; valid skills are {skills}."
        )
    merge_idx = sorted({name_to_idx[s] for s in requested})
    if len(merge_idx) < 2:
        raise CalibrationError(
            "--collapse needs >= 2 distinct skills to merge "
            f"(got {requested!r} -> {[skills[i] for i in merge_idx]})."
        )

    merge_set = set(merge_idx)
    first = min(merge_idx)
    new_cols: list[np.ndarray] = []
    labels: list[str] = []
    for i, s in enumerate(skills):
        if i in merge_set:
            if i == first:
                merged = (Q[:, merge_idx].sum(axis=1) > 0).astype(int)
                new_cols.append(merged)
                labels.append("+".join(skills[m] for m in merge_idx))
            # subsequent merged columns are folded into the combined column above
        else:
            new_cols.append(Q[:, i].astype(int))
            labels.append(s)

    Q_collapsed = (
        np.stack(new_cols, axis=1)
        if new_cols
        else np.zeros((Q.shape[0], 0), dtype=int)
    )

    info = {
        "merged_skills": [skills[m] for m in merge_idx],
        "merged_indices": merge_idx,
        "collapsed_labels": labels,
        "n_dims": int(Q_collapsed.shape[1]),
    }
    return Q_collapsed, labels, info


# ---------------------------------------------------------------------------
# Gauss-Hermite quadrature grid
# ---------------------------------------------------------------------------


def build_grid(n_dims: int, nodes_per_dim: int) -> np.ndarray:
    """Fixed Gauss-Hermite grid nodes for a standard-normal latent (per dim).

    Returns an (n_nodes, n_dims) array of node coordinates. The companion
    standard-normal quadrature weights are obtained from :func:`base_log_weights`.
    Physicists' GH integrates ``int e^{-x^2} g(x) dx``; the change of variables
    ``theta = sqrt(2) x`` maps it to ``int phi(theta) g(theta) dtheta`` (standard
    normal), so node positions are ``sqrt(2) * x`` and weights ``w / sqrt(pi)``.
    """
    x, _ = np.polynomial.hermite_e.hermegauss(nodes_per_dim)  # probabilists' GH
    # hermegauss integrates int e^{-x^2/2} g(x) dx ~ sum w g(x); nodes are already
    # on the standard-normal scale. Grid = cartesian product over dims.
    mesh = np.meshgrid(*([x] * n_dims), indexing="ij")
    grid = np.stack([m.reshape(-1) for m in mesh], axis=1)
    return grid.astype(float)


def _base_1d_weights(nodes_per_dim: int) -> tuple[np.ndarray, np.ndarray]:
    x, w = np.polynomial.hermite_e.hermegauss(nodes_per_dim)
    w = w / np.sqrt(2.0 * np.pi)  # normalise to a standard-normal density weight
    return x, w


def base_log_weights(n_dims: int, nodes_per_dim: int) -> np.ndarray:
    """log of the product standard-normal quadrature weights over the grid."""
    _, w = _base_1d_weights(nodes_per_dim)
    logw = np.log(w)
    mesh = np.meshgrid(*([logw] * n_dims), indexing="ij")
    return np.sum(np.stack([m.reshape(-1) for m in mesh], axis=1), axis=1)


def prior_log_weights(grid: np.ndarray, base_logw: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Log prior weight of each grid node under MVN(0, R), normalised to sum 1.

    The fixed grid + ``base_logw`` approximate ``int g(theta) prod phi(theta_k)``.
    To integrate against MVN(0, R) we reweight each node by the density ratio
    ``N(node; 0, R) / prod phi(node_k)`` (in log space, this drops the per-dim
    normalisers and leaves ``-0.5 theta' (R^-1 - I) theta - 0.5 log|R|``).
    """
    n_dims = grid.shape[1]
    R = np.asarray(R, dtype=float)
    if R.shape != (n_dims, n_dims) or not np.allclose(R, R.T, atol=1e-10):
        raise CalibrationError("latent correlation matrix has the wrong shape or is not symmetric")
    if np.allclose(R, np.eye(n_dims)):
        logw = base_logw
    else:
        if float(np.min(np.linalg.eigvalsh(R))) <= 0:
            raise CalibrationError("latent correlation matrix is not positive definite")
        Rinv = np.linalg.inv(R)
        sign, logdet = np.linalg.slogdet(R)
        if sign <= 0 or not np.isfinite(logdet):
            raise CalibrationError("latent correlation matrix has a non-positive determinant")
        quad = np.einsum("gi,ij,gj->g", grid, (Rinv - np.eye(n_dims)), grid)
        logw = base_logw - 0.5 * quad - 0.5 * logdet
    logw = logw - logsumexp(logw)
    return logw


# ---------------------------------------------------------------------------
# M2PL EM
# ---------------------------------------------------------------------------


def _project_correlation(matrix: np.ndarray, eigen_floor: float = 1e-6) -> np.ndarray:
    """Return a symmetric positive-definite correlation matrix.

    Posterior moment updates can become nearly singular when latent dimensions are
    highly correlated. Eigenvalue flooring followed by diagonal normalization keeps
    the next quadrature reweighting numerically defined without changing the unit-
    variance identification convention.
    """
    sym = (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T) / 2.0
    vals, vecs = np.linalg.eigh(sym)
    vals = np.maximum(vals, eigen_floor)
    pd = (vecs * vals) @ vecs.T
    scale = np.sqrt(np.clip(np.diag(pd), eigen_floor, None))
    corr = pd / np.outer(scale, scale)
    corr = (corr + corr.T) / 2.0
    np.fill_diagonal(corr, 1.0)
    # Normalization can reintroduce a tiny numerical negative eigenvalue. One more
    # projection is cheap at the small K used here.
    vals, vecs = np.linalg.eigh(corr)
    if float(vals.min()) <= 0:
        vals = np.maximum(vals, eigen_floor)
        corr = (vecs * vals) @ vecs.T
        scale = np.sqrt(np.clip(np.diag(corr), eigen_floor, None))
        corr = corr / np.outer(scale, scale)
        corr = (corr + corr.T) / 2.0
        np.fill_diagonal(corr, 1.0)
    return corr


def _item_neg_loglik(beta, X, r, N, ridge):
    """Weighted-logistic negative loglik + gradient + Hessian for one item.

    ``X`` is (n_nodes, p) with a trailing constant column (the -b intercept).
    ``r`` = expected successes per node, ``N`` = expected count per node. ``ridge``
    penalises the loadings (not the intercept) for numerical stability.
    """
    eta = X @ beta
    logp = log_expit(eta)
    log1mp = log_expit(-eta)
    nll = -(np.dot(r, logp) + np.dot(N - r, log1mp))
    p = expit(eta)
    grad = X.T @ (N * p - r)
    W = N * p * (1.0 - p)
    H = (X * W[:, None]).T @ X
    if ridge > 0:
        pen = np.ones(beta.shape[0])
        pen[-1] = 0.0  # do not penalise the intercept
        nll = nll + 0.5 * ridge * float(np.dot(pen * beta, beta))
        grad = grad + ridge * (pen * beta)
        H = H + ridge * np.diag(pen)
    return nll, grad, H


def _fit_item(X, r, N, ridge, max_newton=50, tol=1e-8):
    """Newton-Raphson MLE for one item's [free loadings..., intercept]."""
    p = X.shape[1]
    beta = np.zeros(p)
    # Warm start the intercept from the pooled pass rate at this item.
    tot = float(N.sum())
    if tot > 0:
        pbar = float(r.sum()) / tot
        pbar = min(max(pbar, 1e-3), 1 - 1e-3)
        beta[-1] = np.log(pbar / (1.0 - pbar))
    nll, grad, H = _item_neg_loglik(beta, X, r, N, ridge)
    for _ in range(max_newton):
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, grad, rcond=None)[0]
        # Step-halving line search to guarantee descent.
        alpha = 1.0
        for _ in range(30):
            cand = beta - alpha * step
            new_nll, new_grad, new_H = _item_neg_loglik(cand, X, r, N, ridge)
            if np.isfinite(new_nll) and new_nll <= nll + 1e-12:
                break
            alpha *= 0.5
        else:
            break
        if abs(nll - new_nll) < tol and np.max(np.abs(beta - cand)) < tol:
            beta, nll, grad, H = cand, new_nll, new_grad, new_H
            break
        beta, nll, grad, H = cand, new_nll, new_grad, new_H
    return beta


def fit_m2pl_em(
    Y: np.ndarray,
    M: np.ndarray,
    Q: np.ndarray,
    nodes_per_dim: int,
    estimate_corr: bool = False,
    ridge: float = 1e-3,
    max_iter: int = 200,
    tol: float = 1e-4,
) -> dict:
    """Confirmatory M2PL by Bock-Aitkin EM.

    ``Y`` (n_persons, n_items) 0/1 (values under holes ignored), ``M`` the observed
    mask (bool), ``Q`` (n_items, n_dims) 0/1 confirmatory mask. Returns a dict with
    fitted ``A`` (n_items, n_dims; 0 exactly off-mask), ``b`` (n_items,), ``R``, the
    marginal ``loglik``, ``n_params``, ``n_iter`` and ``converged``.
    """
    Y = np.asarray(Y, dtype=float)
    M = np.asarray(M, dtype=bool)
    Q_raw = np.asarray(Q)
    if Y.ndim != 2 or M.shape != Y.shape:
        raise CalibrationError(f"Y and M must be same-shape 2-D arrays; got {Y.shape}, {M.shape}")
    if Q_raw.ndim != 2 or Q_raw.shape[0] != Y.shape[1] or Q_raw.shape[1] < 1:
        raise CalibrationError(
            f"Q must have shape (n_items, n_dims>=1); got {Q_raw.shape} for Y {Y.shape}"
        )
    if nodes_per_dim < 2:
        raise CalibrationError("nodes_per_dim must be at least 2")
    if not np.isin(Q_raw, (0, 1)).all():
        raise CalibrationError("Q must contain only 0/1 values")
    Q = Q_raw.astype(int)
    if np.any(Q.sum(axis=0) == 0):
        empty = np.where(Q.sum(axis=0) == 0)[0].tolist()
        raise CalibrationError(f"Q has latent dimension(s) with no loading items: {empty}")
    if np.any(Q.sum(axis=1) == 0):
        raise CalibrationError("Q has item row(s) with no configured loading")
    observed_values = Y[M]
    if observed_values.size and not np.isin(observed_values, (0.0, 1.0)).all():
        raise CalibrationError("observed Y values must be binary 0/1")

    n_persons, n_items = Y.shape
    n_dims = Q.shape[1]
    grid = build_grid(n_dims, nodes_per_dim)
    base_logw = base_log_weights(n_dims, nodes_per_dim)
    n_nodes = grid.shape[0]

    YM = np.where(M, Y, 0.0)          # observed successes
    NM = np.where(M, 1.0 - Y, 0.0)    # observed failures
    Mf = M.astype(float)

    A = np.zeros((n_items, n_dims))
    A[Q == 1] = 1.0                   # free loadings start at 1
    # Difficulty warm start from item pass rates (offset scale).
    obs_per_item = Mf.sum(axis=0)
    pass_rate = np.where(obs_per_item > 0, YM.sum(axis=0) / np.maximum(obs_per_item, 1), 0.5)
    pass_rate = np.clip(pass_rate, 1e-3, 1 - 1e-3)
    b = -np.log(pass_rate / (1.0 - pass_rate))
    R = np.eye(n_dims)

    # Cache one design matrix per distinct Q pattern rather than per item. InFoBench
    # has 2,250 items but at most 31 non-zero patterns on its five-skill axis.
    free_dims = [np.where(Q[j] == 1)[0] for j in range(n_items)]
    ones_col = np.ones((n_nodes, 1))
    designs = {
        tuple(fd.tolist()): np.hstack([grid[:, fd], ones_col])
        for fd in free_dims
    }

    prev_ll = -np.inf
    converged = False
    n_iter = 0
    posterior = None
    for it in range(max_iter):
        n_iter = it + 1
        log_prior = prior_log_weights(grid, base_logw, R)
        # E-step: per-person loglik over nodes, then posterior.
        eta = A @ grid.T - b[:, None]              # (n_items, n_nodes)
        logP = log_expit(eta)
        log1mP = log_expit(-eta)
        LL = YM @ logP + NM @ log1mP               # (n_persons, n_nodes)
        joint = LL + log_prior[None, :]
        person_ll = logsumexp(joint, axis=1)       # (n_persons,)
        marg_ll = float(person_ll.sum())
        posterior = np.exp(joint - person_ll[:, None])  # (n_persons, n_nodes)

        # M-step: expected counts per item x node.
        r_jg = YM.T @ posterior                    # (n_items, n_nodes)
        N_jg = Mf.T @ posterior                    # (n_items, n_nodes)
        for j in range(n_items):
            beta = _fit_item(designs[tuple(free_dims[j].tolist())], r_jg[j], N_jg[j], ridge)
            A[j] = 0.0
            A[j, free_dims[j]] = beta[:-1]
            b[j] = -beta[-1]

        if estimate_corr and n_dims > 1:
            # Posterior second moment -> covariance -> correlation (var fixed to 1).
            w = posterior.sum(axis=0)              # (n_nodes,)
            Sigma = (grid.T * w) @ grid / n_persons
            d = np.sqrt(np.clip(np.diag(Sigma), 1e-8, None))
            R = _project_correlation(Sigma / np.outer(d, d))

        if abs(marg_ll - prev_ll) < tol:
            converged = True
            prev_ll = marg_ll
            break
        prev_ll = marg_ll

    A[Q == 0] = 0.0  # enforce the confirmatory mask exactly

    # Recompute the marginal loglik with the FINAL params (the in-loop value lags
    # one M-step behind), so it matches the returned A/b for exact AIC/BIC.
    log_prior = prior_log_weights(grid, base_logw, R)
    eta = A @ grid.T - b[:, None]
    LL = YM @ log_expit(eta) + NM @ log_expit(-eta)
    final_ll = float(logsumexp(LL + log_prior[None, :], axis=1).sum())

    n_free_load = int(Q.sum())
    n_params = n_free_load + n_items
    if estimate_corr and n_dims > 1:
        n_params += n_dims * (n_dims - 1) // 2

    return {
        "A": A,
        "b": b,
        "R": R,
        "loglik": final_ll,
        "n_params": int(n_params),
        "n_free_loadings": n_free_load,
        "n_iter": n_iter,
        "converged": converged,
        "n_dims": n_dims,
        "grid_nodes": int(n_nodes),
    }


def aic_bic(loglik: float, n_params: int, n_obs: int) -> tuple[float, float]:
    aic = 2 * n_params - 2 * loglik
    bic = n_params * np.log(max(n_obs, 1)) - 2 * loglik
    return float(aic), float(bic)


def json_finite(value):
    """Recursively replace NaN/Inf numpy or Python values with JSON null."""
    if isinstance(value, dict):
        return {str(key): json_finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_finite(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_finite(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


# ---------------------------------------------------------------------------
# optional girth cross-check + EFA
# ---------------------------------------------------------------------------


def girth_crosscheck(block: pd.DataFrame, internal_items, internal_a, internal_b) -> dict:
    """Optional dense-block girth 2PL cross-check (rank agreement only). Guarded."""
    try:
        from tutor_cat.mcq_irt.calibrate import fit_2pl
    except Exception as e:  # pragma: no cover
        return {"available": False, "reason": f"import failed: {e}"}
    try:
        calib = fit_2pl(block, method="girth")
    except Exception as e:  # pragma: no cover - girth absent or degenerate block
        return {"available": False, "reason": str(e)}
    ga = calib.a_by_item()
    gb = calib.b_by_item()
    shared = [it for it in internal_items if it in ga]
    if len(shared) < 3:
        return {"available": True, "n_shared": len(shared)}
    idx = {it: k for k, it in enumerate(internal_items)}
    ia = pd.Series([internal_a[idx[it]] for it in shared])
    ib = pd.Series([internal_b[idx[it]] for it in shared])
    ca = pd.Series([ga[it] for it in shared])
    cb = pd.Series([gb[it] for it in shared])
    return {
        "available": True,
        "n_shared": len(shared),
        "a_rank_corr": float(ia.rank().corr(ca.rank())),
        "b_rank_corr": float(ib.rank().corr(cb.rank())),
    }


def run_efa(block: np.ndarray, n_factors: int | None = None) -> dict:
    """Optional EFA / scree on the variant-item block. Skips cleanly if absent."""
    n_factors = N_SKILLS if n_factors is None else int(n_factors)
    try:
        from factor_analyzer import FactorAnalyzer
    except Exception:
        return {"available": False, "reason": "factor_analyzer not installed"}
    try:
        fa = FactorAnalyzer(n_factors=n_factors, rotation="varimax")
        fa.fit(block)
        ev, _ = fa.get_eigenvalues()
        return {
            "available": True,
            "eigenvalues": [float(v) for v in ev[: max(n_factors + 3, 6)]],
            "n_factors_ge_1": int(np.sum(np.asarray(ev) >= 1.0)),
        }
    except Exception as e:  # pragma: no cover
        return {"available": False, "reason": str(e)}


# ---------------------------------------------------------------------------
# selection (holes handled natively) + fit driver
# ---------------------------------------------------------------------------


def prepare_block(mat: pd.DataFrame, q_by: dict[str, np.ndarray]) -> tuple:
    """Drop empty rows/cols, zero-variance items, and Q-less items; align Q rows.

    Returns (Y, M, Q, items, block_df, diag). Holes stay in-place (marginalised by the E-step);
    only fully-empty rows and unfittable columns are removed.
    """
    sub, sel_diag = cp.select_sparse(mat)
    kept, all_fail, all_pass = cp.split_zero_variance(sub)
    sub = sub[kept]
    sub = sub[sub.notna().any(axis=1)]

    Q, items, missing_q = align_q_rows(list(sub.columns), q_by)
    sub = sub[items]
    sub = sub[sub.notna().any(axis=1)]

    Y = np.nan_to_num(sub.to_numpy(dtype=float), nan=0.0)
    M = sub.notna().to_numpy()

    per_skill_items = {
        skill: int(Q[:, k].sum()) if Q.size else 0 for k, skill in enumerate(SKILLS)
    }
    per_skill_single_load_anchors = {
        skill: int(np.sum((Q[:, k] == 1) & (Q.sum(axis=1) == 1))) if Q.size else 0
        for k, skill in enumerate(SKILLS)
    }
    empty_skills = [skill for skill, count in per_skill_items.items() if count == 0]
    if empty_skills:
        raise CalibrationError(
            "no surviving variant items load on configured skill(s) "
            f"{empty_skills}; those dimensions cannot be calibrated."
        )

    diag = {
        **sel_diag,
        "dropped_all_fail": len(all_fail),
        "dropped_all_pass": len(all_pass),
        "dropped_zero_variance_total": len(all_fail) + len(all_pass),
        "dropped_all_fail_criteria": all_fail,
        "dropped_all_pass_criteria": all_pass,
        "dropped_missing_qrow": len(missing_q),
        "dropped_missing_qrow_criteria": missing_q,
        "n_items_fit": int(sub.shape[1]),
        "n_persons_fit": int(sub.shape[0]),
        "q_pattern_counts": _q_pattern_counts(Q),
        "per_skill_items_fit": per_skill_items,
        "per_skill_single_load_anchors": per_skill_single_load_anchors,
    }
    return Y, M, Q, items, sub, diag


def _q_pattern_counts(Q: np.ndarray) -> dict:
    counts: dict[str, int] = {}
    for row in Q:
        key = "".join(str(int(v)) for v in row)
        counts[key] = counts.get(key, 0) + 1
    return counts


def per_item_n_persons(M: np.ndarray) -> np.ndarray:
    return M.sum(axis=0).astype(int)


def item_flags(
    A: np.ndarray, Q: np.ndarray, n_persons_item: np.ndarray, min_ident: int
) -> list[str]:
    flags = []
    for j in range(A.shape[0]):
        f = []
        free_vals = A[j][Q[j] == 1]
        if np.any(free_vals <= 0):
            f.append("nonpositive_a")
        if np.any(~np.isfinite(free_vals)) or np.any(np.abs(free_vals) > EXTREME_A):
            f.append("extreme_a")
        if n_persons_item[j] < min_ident:
            f.append("low_n")
        flags.append("|".join(f))
    return flags


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------


def build_frame(items, A, b, n_persons_item, flags) -> pd.DataFrame:
    data = {"criterion_id": items}
    for k, skill in enumerate(SKILLS):
        data[f"a_{skill}"] = np.round(A[:, k], 6)
    data["b"] = np.round(b, 6)
    data["n_persons"] = n_persons_item
    data["flags"] = flags
    cols = ["criterion_id"] + [f"a_{s}" for s in SKILLS] + ["b", "n_persons", "flags"]
    return pd.DataFrame(data)[cols]


def write_csv(frame: pd.DataFrame, out_dir: Path) -> Path:
    path = out_dir / CALIBRATION_CSV_NAME
    frame.to_csv(path, index=False)
    return path


def _collapse_manifest(
    collapsed, collapse_info, multi, bic_n, multi_aic, multi_bic, uni, uni_aic, uni_bic
) -> dict | None:
    """The collapsed-model block for the manifest (a clean 3-way comparison).

    ``None`` when ``--collapse`` was not given, so the default (non-collapsed) run
    is byte-for-byte unchanged except for this explicit null field.
    """
    if collapsed is None or collapse_info is None:
        return None
    coll_aic, coll_bic = aic_bic(collapsed["loglik"], collapsed["n_params"], bic_n)
    return {
        "requested": True,
        "merged_skills": collapse_info["merged_skills"],
        "collapsed_labels": collapse_info.get("labels", collapse_info["collapsed_labels"]),
        "n_dims": collapse_info["n_dims"],
        "collapsed_model": {
            "n_dims": collapsed["n_dims"],
            "loglik": collapsed["loglik"],
            "n_params": collapsed["n_params"],
            "aic": coll_aic,
            "bic": coll_bic,
        },
        "three_way": {
            "unidim": {"n_dims": 1, "loglik": uni["loglik"],
                       "n_params": uni["n_params"], "aic": uni_aic, "bic": uni_bic},
            "collapsed": {"n_dims": collapsed["n_dims"], "loglik": collapsed["loglik"],
                          "n_params": collapsed["n_params"], "aic": coll_aic, "bic": coll_bic},
            "full": {"n_dims": multi["n_dims"], "loglik": multi["loglik"],
                     "n_params": multi["n_params"], "aic": multi_aic, "bic": multi_bic},
        },
        "collapsed_beats_full_aic": bool(coll_aic < multi_aic),
        "collapsed_beats_full_bic": bool(coll_bic < multi_bic),
        "delta_aic_full_minus_collapsed": float(multi_aic - coll_aic),
        "delta_bic_full_minus_collapsed": float(multi_bic - coll_bic),
        "aic_winner": min(
            {"unidim": uni_aic, "collapsed": coll_aic, "full": multi_aic}.items(),
            key=lambda kv: kv[1],
        )[0],
        "bic_winner": min(
            {"unidim": uni_bic, "collapsed": coll_bic, "full": multi_bic}.items(),
            key=lambda kv: kv[1],
        )[0],
        "collapsed_latent_correlation": (
            np.round(np.asarray(collapsed["R"]), 6).tolist()
            if collapsed.get("R") is not None
            and collapsed["n_dims"] > 1
            and not np.allclose(collapsed["R"], np.eye(collapsed["n_dims"]))
            else None
        ),
    }


def write_manifest(
    out_dir: Path,
    args: argparse.Namespace,
    diag: dict,
    multi: dict,
    uni: dict,
    n_obs: int,
    latent_corr,
    crosscheck: dict,
    efa: dict,
    matrix_prov: dict,
    identifiable: bool,
    collapsed: dict | None = None,
    collapse_info: dict | None = None,
) -> Path:
    path = out_dir / CALIBRATION_MANIFEST_NAME
    bic_n = int(diag["n_persons_fit"])
    multi_aic, multi_bic = aic_bic(multi["loglik"], multi["n_params"], bic_n)
    uni_aic, uni_bic = aic_bic(uni["loglik"], uni["n_params"], bic_n)
    _, multi_bic_cells = aic_bic(multi["loglik"], multi["n_params"], n_obs)
    _, uni_bic_cells = aic_bic(uni["loglik"], uni["n_params"], n_obs)
    warning = None
    if not identifiable:
        warning = (
            f"n_persons_fit={diag['n_persons_fit']} < "
            f"--min-persons-identifiable={args.min_persons_identifiable}: the "
            f"{N_SKILLS}-dim "
            "confirmatory M2PL is below the configured heuristic person-count "
            "threshold. Treat parameters as provisional and check held-out stability."
        )
    manifest = {
        "generated_at": cp._utcnow(),
        "note": (
            "CONFIRMATORY MULTIDIMENSIONAL 2PL (M2PL) fit, pure numpy/scipy, "
            "Bock-Aitkin EM over a Gauss-Hermite grid. Loadings masked by the "
            f"{N_SKILLS}-skill Q-matrix (a_k free iff q_k==1, else exactly 0)."
        ),
        "method": "confirmatory-m2pl-mml-em",
        "skills_order": list(SKILLS),
        "skill_structure": diag.get("skill_structure"),
        "grid_nodes_per_dim": args.grid,
        "grid_total_nodes": multi["grid_nodes"],
        "ridge": args.ridge,
        "estimate_latent_corr": args.estimate_latent_corr,
        "em": {
            "max_iter": args.max_iter,
            "tol": args.tol,
            "multi_n_iter": multi["n_iter"],
            "multi_converged": multi["converged"],
            "uni_n_iter": uni["n_iter"],
            "uni_converged": uni["converged"],
        },
        "block": diag,
        "n_items_fit": diag["n_items_fit"],
        "n_persons_fit": diag["n_persons_fit"],
        "n_observed_cells": n_obs,
        "dropped_zero_variance": diag["dropped_zero_variance_total"],
        "dropped_all_fail": diag["dropped_all_fail"],
        "dropped_all_pass": diag["dropped_all_pass"],
        "dropped_missing_qrow": diag["dropped_missing_qrow"],
        "comparison": {
            "multi": {
                "n_dims": multi["n_dims"],
                "loglik": multi["loglik"],
                "n_params": multi["n_params"],
                "aic": multi_aic,
                "bic": multi_bic,
            },
            "uni": {
                "n_dims": 1,
                "loglik": uni["loglik"],
                "n_params": uni["n_params"],
                "aic": uni_aic,
                "bic": uni_bic,
            },
            "multi_beats_uni_aic": bool(multi_aic < uni_aic),
            "multi_beats_uni_bic": bool(multi_bic < uni_bic),
            "delta_aic_uni_minus_multi": float(uni_aic - multi_aic),
            "delta_bic_uni_minus_multi": float(uni_bic - multi_bic),
            "bic_sample_size_convention": "n_persons_fit",
            "bic_sensitivity_observed_cells": {
                "sample_size": n_obs,
                "multi_bic": multi_bic_cells,
                "uni_bic": uni_bic_cells,
                "multi_beats_uni": bool(multi_bic_cells < uni_bic_cells),
            },
        },
        "collapse": _collapse_manifest(collapsed, collapse_info, multi, bic_n,
                                       multi_aic, multi_bic, uni, uni_aic, uni_bic),
        "latent_correlation": (
            np.round(np.asarray(latent_corr), 6).tolist() if latent_corr is not None else None
        ),
        "girth_crosscheck": crosscheck,
        "efa": efa,
        "identifiability": {
            "min_persons_identifiable": args.min_persons_identifiable,
            "threshold_kind": "heuristic person-count warning, not a proof of identifiability",
            "meets_person_count_threshold": identifiable,
            # Backward-compatible key retained for older report readers.
            "identifiable": identifiable,
            "warning": warning,
        },
        "matrix": matrix_prov,
        "provenance": {"script": "scripts/calibrate_mirt.py", "argv": sys.argv[1:]},
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_finite(manifest), f, indent=2, allow_nan=False)
    return path


def write_mirt_rubrics(
    rubrics_path: Path,
    out_path: Path,
    items,
    A,
    b,
    args: argparse.Namespace,
    matrix_prov: dict,
    latent_corr,
) -> tuple[Path, int]:
    """Write a COPY of the rubric bank with the fitted masked K-vector + scalar b.

    Fitted criteria get ``discrimination`` = the masked K-vector (0 where q=0),
    ``difficulty`` = b, and ``irt_params.source = 'calibrated-m2pl'`` + provenance.
    Non-fitted criteria keep all synthetic values. Never mutates the input; refuses
    to write over ``rubrics_qmatrix_final.jsonl`` or the input path.
    """
    if out_path.resolve() == rubrics_path.resolve():
        raise CalibrationError("refusing to overwrite the input rubric bank in place.")
    if out_path.name == DEFAULT_RUBRICS.name:
        raise CalibrationError(
            "refusing to write to the frozen rubrics_qmatrix_final.jsonl; "
            "choose a different --out-rubrics."
        )
    records = cp.read_jsonl(rubrics_path)
    a_by = {it: A[i] for i, it in enumerate(items)}
    b_by = {it: float(b[i]) for i, it in enumerate(items)}

    provenance = {
        "source": CALIBRATION_SOURCE,
        "method": "confirmatory-m2pl-mml-em",
        "skills_order": list(SKILLS),
        "n_dimensions": N_SKILLS,
        "grid_nodes_per_dim": args.grid,
        "estimate_latent_corr": args.estimate_latent_corr,
        "ridge": args.ridge,
        "latent_correlation": (
            np.round(np.asarray(latent_corr), 6).tolist() if latent_corr is not None else None
        ),
        "matrix": matrix_prov,
        "calibrated_at": cp._utcnow(),
        "version": "1.0",
    }

    n_updated = 0
    for rec in records:
        cid = rec.get("criterion_id")
        if cid in a_by:
            avec = a_by[cid]
            if np.all(np.isfinite(avec)) and np.isfinite(b_by[cid]):
                rec["discrimination"] = {
                    s: round(float(avec[k]), 4) for k, s in enumerate(SKILLS)
                }
                rec["difficulty"] = round(b_by[cid], 4)
                rec["irt_params"] = dict(provenance)
                n_updated += 1
    cp.write_jsonl(out_path, records)
    return out_path, n_updated


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX,
                   help=f"response matrix CSV (default: {DEFAULT_MATRIX}).")
    p.add_argument("--rubrics", type=Path, default=DEFAULT_RUBRICS,
                   help=f"rubric bank JSONL for the Q-matrix (default: {DEFAULT_RUBRICS}).")
    p.add_argument(
        "--require-complete-bank",
        action="store_true",
        help="fail unless the matrix contains every active criterion in --rubrics.",
    )
    p.add_argument(
        "--skills",
        default=None,
        metavar="SKILL,SKILL[,...]",
        help=(
            "ordered Q-matrix dimensions. Defaults to the package axis "
            f"{list(PACKAGE_SKILLS)}. InFoBench uses "
            "content,format,number,style,linguistic."
        ),
    )
    p.add_argument(
        "--source-skills",
        default=None,
        metavar="SKILL,SKILL[,...]",
        help=(
            "ordered native Q-matrix axis when --dimensions defines a reduced "
            "latent structure. Example: content,format,number,style,linguistic."
        ),
    )
    p.add_argument(
        "--dimensions",
        default=None,
        metavar="LABEL=SKILL[+SKILL],...",
        help=(
            "validated partition of --source-skills used to compare reduced latent "
            "structures. Q columns in each group are OR-merged in memory; the source "
            "bank is not changed. Example: semantic=content+style,constraints="
            "format+number+linguistic."
        ),
    )
    p.add_argument(
        "--structure-name",
        default="custom",
        help="provenance label recorded for --dimensions (default: custom).",
    )
    p.add_argument("--grid", type=int, default=7,
                   help="Gauss-Hermite nodes per latent dimension (default 7; "
                        "total nodes = grid ** number_of_skills).")
    p.add_argument(
        "--max-grid-nodes",
        type=int,
        default=DEFAULT_MAX_GRID_NODES,
        help=(
            "safety cap on grid ** number_of_skills (default 5000); "
            "use --allow-large-grid to override deliberately."
        ),
    )
    p.add_argument(
        "--allow-large-grid",
        action="store_true",
        help="allow a quadrature grid above --max-grid-nodes (may be slow or memory-heavy).",
    )
    p.add_argument("--estimate-latent-corr", action="store_true",
                   help="estimate the KxK latent correlation from the posterior "
                        "(default: fixed identity).")
    p.add_argument("--collapse", type=str, default=None, metavar="SKILL,SKILL[,...]",
                   help="comma-separated skill names to MERGE into a single latent "
                        "dimension (fit-time, in-memory Q-matrix transform ONLY; the "
                        "rubric bank is never touched). E.g. --collapse content,diagnosis "
                        "fits a collapsed model (content+diagnosis merged via "
                        "logical-OR of the Q columns) ALONGSIDE the full K-dimensional "
                        "model for a uni/collapsed/full comparison.")
    p.add_argument("--ridge", type=float, default=1e-2,
                   help="L2 ridge on loadings in the M-step for stability (default 1e-2; "
                        "adopted 2026-07-31 per the Scaffolding Hygiene audit plateau — "
                        "best OOS scaffolding+correctness recovery, drains extreme-a).")
    p.add_argument("--max-iter", type=int, default=200, help="max EM iterations.")
    p.add_argument("--tol", type=float, default=1e-4,
                   help="EM convergence tol on marginal loglik (default 1e-4).")
    p.add_argument("--min-persons-identifiable", type=int, default=150,
                   help="warn if persons < this (M2PL identifiability; default 150).")
    p.add_argument("--efa", action="store_true",
                   help="optional EFA/scree diagnostic (needs factor_analyzer; guarded).")
    p.add_argument("--report-only", action="store_true",
                   help="emit the coverage report and exit (no fit).")
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan (block shape, grid) and exit without fitting.")
    p.add_argument("--write-params", action="store_true",
                   help="write a calibrated COPY only when the fit converged and every "
                        "active item has a finite positive/stable fit (OFF by default).")
    p.add_argument(
        "--out-rubrics",
        type=Path,
        default=DEFAULT_MIRT_RUBRICS,
        help=f"output rubric JSONL for --write-params (default: {DEFAULT_MIRT_RUBRICS}).",
    )
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                   help=f"directory for calibration outputs (default: {DEFAULT_OUT_DIR}).")
    args = p.parse_args()

    structure: SkillStructure | None = None
    try:
        if args.dimensions:
            if not args.source_skills:
                raise ValueError("--dimensions requires --source-skills")
            if args.skills:
                raise ValueError(
                    "use --source-skills plus --dimensions for a reduced structure; "
                    "do not also pass --skills"
                )
            source_axis = tuple(
                skill.strip() for skill in args.source_skills.split(",") if skill.strip()
            )
            structure = parse_dimension_spec(
                args.dimensions,
                source_axis,
                name=args.structure_name,
            )
            configure_skills(",".join(structure.labels))
        else:
            if args.source_skills:
                raise ValueError("--source-skills is only meaningful with --dimensions")
            configure_skills(args.skills)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    if structure is not None and args.collapse:
        print(
            "ERROR: --collapse and --dimensions are alternative structure transforms; "
            "use one or the other.",
            file=sys.stderr,
        )
        return 2
    if structure is not None and args.write_params:
        print(
            "ERROR: --write-params is disabled for transformed structures. Fit first, "
            "then use scripts/export_fitted_bank.py so unfitted/unstable items and "
            "scenario links are handled explicitly.",
            file=sys.stderr,
        )
        return 2

    if args.grid < 2:
        print("ERROR: --grid must be at least 2.", file=sys.stderr)
        return 2
    if args.max_iter < 1:
        print("ERROR: --max-iter must be positive.", file=sys.stderr)
        return 2
    if args.tol <= 0:
        print("ERROR: --tol must be positive.", file=sys.stderr)
        return 2
    if args.ridge < 0:
        print("ERROR: --ridge cannot be negative.", file=sys.stderr)
        return 2
    if args.min_persons_identifiable < 1:
        print("ERROR: --min-persons-identifiable must be positive.", file=sys.stderr)
        return 2
    if args.max_grid_nodes < 1:
        print("ERROR: --max-grid-nodes must be positive.", file=sys.stderr)
        return 2
    grid_total = args.grid ** N_SKILLS
    if grid_total > args.max_grid_nodes and not args.allow_large_grid and not args.report_only:
        print(
            f"ERROR: requested {args.grid}^{N_SKILLS} = {grid_total:,} quadrature nodes, "
            f"above the safety cap {args.max_grid_nodes:,}. Lower --grid "
            "(InFoBench should start with --grid 3), raise --max-grid-nodes, or pass "
            "--allow-large-grid after checking memory/runtime.",
            file=sys.stderr,
        )
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)

    collapse_skills: list[str] | None = None
    if args.collapse:
        collapse_skills = [s.strip() for s in args.collapse.split(",") if s.strip()]
        unknown = [s for s in collapse_skills if s not in SKILLS]
        if unknown:
            print(f"ERROR: --collapse names unknown skill(s) {unknown}; "
                  f"valid skills are {list(SKILLS)}.", file=sys.stderr)
            return 2
        if len(set(collapse_skills)) < 2:
            print("ERROR: --collapse needs >= 2 distinct skills to merge.",
                  file=sys.stderr)
            return 2

    try:
        mat = load_matrix_strict(args.matrix)
    except (FileNotFoundError, CalibrationError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    cov = cp.compute_coverage(mat)
    cp.print_coverage(cov)
    cov_csv, cov_json = cp.write_coverage_reports(cov, args.out_dir)
    print(f"\nwrote coverage report -> {cov_csv}")
    print(f"wrote coverage summary -> {cov_json}")

    if args.report_only:
        return 0

    try:
        q_by = load_q_matrix(args.rubrics, structure=structure)
    except (FileNotFoundError, CalibrationError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    try:
        alignment = validate_matrix_bank_alignment(mat, q_by, args.require_complete_bank)
    except CalibrationError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 3

    try:
        Y, M, Q, items, block_df, diag = prepare_block(mat, q_by)
    except CalibrationError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 3
    diag["matrix_bank_alignment"] = alignment
    diag["skill_structure"] = (
        structure.as_dict()
        if structure is not None
        else SkillStructure.identity(SKILLS, name="identity").as_dict()
    )

    n_items = diag["n_items_fit"]
    n_persons = diag["n_persons_fit"]
    print("\n" + "=" * 72)
    print("confirmatory M2PL calibration plan")
    print("=" * 72)
    print(f"fitting {n_items} items x {n_persons} persons, {args.grid} nodes/dim "
          f"({args.grid ** N_SKILLS} grid nodes)")
    print(f"dropped zero-variance : {diag['dropped_zero_variance_total']} "
          f"({diag['dropped_all_fail']} all-fail, {diag['dropped_all_pass']} all-pass)")
    print(f"dropped missing Q-row : {diag['dropped_missing_qrow']}")
    print(f"Q pattern counts      : {diag['q_pattern_counts']}")
    print(f"items per skill       : {diag['per_skill_items_fit']}")
    print(f"single-load anchors   : {diag['per_skill_single_load_anchors']}")
    print(f"bank alignment        : {alignment['n_matrix_criteria']}/"
          f"{alignment['n_active_bank_criteria']} criteria present")

    identifiable = n_persons >= args.min_persons_identifiable
    if not identifiable:
        print(f"WARNING: n_persons={n_persons} < {args.min_persons_identifiable}; the "
              f"{N_SKILLS}-dim M2PL is below the configured heuristic person-count "
              "threshold. Treat the fit as provisional and validate out of sample.")

    if args.dry_run:
        print("\n--dry-run: not fitting. Re-run without --dry-run to calibrate.")
        return 0

    if n_items < 1 or n_persons < 2:
        print("\nERROR: nothing fittable after selection (need >=1 item, >=2 persons).",
              file=sys.stderr)
        return 3

    n_obs = int(M.sum())

    print("\nfitting UNIDIMENSIONAL 2PL baseline (EM, 1 dim) ...")
    Q_uni = np.ones((n_items, 1), dtype=int)
    uni = fit_m2pl_em(Y, M, Q_uni, args.grid, estimate_corr=False,
                      ridge=args.ridge, max_iter=args.max_iter, tol=args.tol)

    print(f"fitting CONFIRMATORY M2PL ({N_SKILLS} dims) ...")
    multi = fit_m2pl_em(Y, M, Q, args.grid, estimate_corr=args.estimate_latent_corr,
                        ridge=args.ridge, max_iter=args.max_iter, tol=args.tol)

    # Optional collapsed model: a FIT-TIME, in-memory Q-matrix transform that merges
    # the requested skills into ONE latent dimension. Runs on the SAME data + all the
    # same machinery (mask/quadrature/latent-corr/AIC/BIC) as the full model.
    collapsed = None
    collapse_info = None
    if collapse_skills is not None:
        Q_collapsed, collapsed_labels, collapse_info = collapse_q_matrix(Q, collapse_skills)
        print(f"fitting COLLAPSED M2PL ({collapse_info['n_dims']} dims; "
              f"merged {'+'.join(collapse_info['merged_skills'])}) ...")
        collapsed = fit_m2pl_em(Y, M, Q_collapsed, args.grid,
                                estimate_corr=args.estimate_latent_corr,
                                ridge=args.ridge, max_iter=args.max_iter, tol=args.tol)
        collapse_info["labels"] = collapsed_labels

    A, b = multi["A"], multi["b"]
    latent_corr = multi["R"] if args.estimate_latent_corr else None

    n_persons_item = per_item_n_persons(M)
    flags = item_flags(A, Q, n_persons_item, args.min_persons_identifiable)

    bic_n = n_persons
    multi_aic, multi_bic = aic_bic(multi["loglik"], multi["n_params"], bic_n)
    uni_aic, uni_bic = aic_bic(uni["loglik"], uni["n_params"], bic_n)

    print("\n" + "=" * 72)
    print("uni vs multi" if collapsed is None else "uni vs collapsed vs multi")
    print("=" * 72)
    print(f"unidim    : loglik={uni['loglik']:.2f}  k={uni['n_params']}  "
          f"AIC={uni_aic:.2f}  BIC={uni_bic:.2f}")
    if collapsed is not None:
        coll_aic, coll_bic = aic_bic(collapsed["loglik"], collapsed["n_params"], bic_n)
        label = "+".join(collapse_info["merged_skills"])
        print(f"collapsed : loglik={collapsed['loglik']:.2f}  k={collapsed['n_params']}  "
              f"AIC={coll_aic:.2f}  BIC={coll_bic:.2f}  "
              f"({collapse_info['n_dims']}-dim, merged {label})")
    print(f"multi     : loglik={multi['loglik']:.2f}  k={multi['n_params']}  "
          f"AIC={multi_aic:.2f}  BIC={multi_bic:.2f}  ({multi['n_dims']}-dim, full)")
    print(f"BIC sample size: {bic_n} tutor models (cell-count BIC retained in manifest only)")
    print(f"multi beats uni : AIC={multi_aic < uni_aic}  BIC={multi_bic < uni_bic}")
    if collapsed is not None:
        coll_aic, coll_bic = aic_bic(collapsed["loglik"], collapsed["n_params"], bic_n)
        # Winner by each criterion across the 3 candidate models.
        full_label = f"full-{N_SKILLS}-dim"
        cand_aic = {"unidim": uni_aic, "collapsed": coll_aic, full_label: multi_aic}
        cand_bic = {"unidim": uni_bic, "collapsed": coll_bic, full_label: multi_bic}
        best_aic = min(cand_aic, key=cand_aic.get)
        best_bic = min(cand_bic, key=cand_bic.get)
        print(f"collapsed beats full : AIC={coll_aic < multi_aic}  "
              f"BIC={coll_bic < multi_bic}  "
              f"(delta_AIC full-collapsed={multi_aic - coll_aic:+.2f}, "
              f"delta_BIC={multi_bic - coll_bic:+.2f})")
        print(f"AIC winner : {best_aic}   BIC winner : {best_bic}")
    if latent_corr is not None:
        print(f"latent correlation ({', '.join(SKILLS)}):")
        for row in np.round(latent_corr, 3):
            print("   ", row.tolist())

    crosscheck = girth_crosscheck(block_df.dropna(axis=0, how="any"), items,
                                  uni["A"][:, 0], uni["b"])
    efa = run_efa(np.nan_to_num(block_df.to_numpy(dtype=float)), N_SKILLS) if args.efa else \
        {"available": False, "reason": "not requested (pass --efa)"}
    if args.efa:
        if efa.get("available"):
            print(f"EFA eigenvalues : {efa['eigenvalues']} "
                  f"({efa['n_factors_ge_1']} >= 1)")
        else:
            print(f"EFA skipped     : {efa['reason']}")

    frame = build_frame(items, A, b, n_persons_item, flags)
    csv_path = write_csv(frame, args.out_dir)
    matrix_prov = cp._matrix_manifest_prov(args.matrix)
    manifest_path = write_manifest(
        args.out_dir, args, diag, multi, uni, n_obs, latent_corr, crosscheck, efa,
        matrix_prov, identifiable, collapsed=collapsed, collapse_info=collapse_info,
    )
    print(f"\nwrote calibration CSV -> {csv_path}")
    print(f"wrote calibration manifest -> {manifest_path}")

    if args.write_params:
        unfitted_active = sorted(set(q_by) - set(items))
        unstable_items = [
            item for item, flag in zip(items, flags, strict=True)
            if "nonpositive_a" in flag or "extreme_a" in flag
        ]
        blockers: list[str] = []
        if not multi["converged"]:
            blockers.append("the multidimensional EM fit did not converge")
        if unfitted_active:
            blockers.append(
                f"{len(unfitted_active)} active bank item(s) were not fitted "
                "(for example all-pass/all-fail or absent from the matrix)"
            )
        if unstable_items:
            blockers.append(
                f"{len(unstable_items)} fitted item(s) have nonpositive/nonfinite/extreme loadings"
            )
        if blockers:
            print(
                "\nERROR: refusing --write-params because " + "; ".join(blockers) + ".",
                file=sys.stderr,
            )
            print(
                "Write the calibration CSV/manifest first, review exclusions, then build a "
                "fitted-only CAT bank rather than mixing calibrated and synthetic values.",
                file=sys.stderr,
            )
            return 4
        try:
            out_path, n_updated = write_mirt_rubrics(
                args.rubrics, args.out_rubrics, items, A, b, args, matrix_prov, latent_corr
            )
        except (CalibrationError, FileNotFoundError) as e:
            print(f"\nERROR: --write-params failed: {e}", file=sys.stderr)
            return 4
        print(f"\nwrote calibrated rubric copy -> {out_path} "
              f"({n_updated} criteria updated; rest kept synthetic)")
        print(f"(input {args.rubrics} left untouched)")

    if not identifiable:
        print(f"\nreminder: {N_SKILLS}-dim fit is below the configured person-count "
              "warning threshold; report it as provisional.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
