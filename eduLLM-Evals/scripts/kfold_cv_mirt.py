"""Person-level k-fold validation of an arbitrary confirmatory M2PL structure.

The response matrix rows are tutor models. For each fold, this script refits item
parameters on the training models, freezes them, EAP-scores held-out models, and
predicts responses from a disjoint set of held-out-person scenarios. It also
reports clearly labeled posterior-reconstruction metrics and cross-fold
item-parameter stability. It never writes parameters back to the rubric bank.

The source Q-matrix and modeled structure are separate. For example, InFoBench has
five source skills, while a candidate structure may merge them into three modeled
dimensions. Modeled dimensions must partition the source axis, and merged Q columns
are formed by logical OR.

With no structure flags, the historical TutorBench behavior is preserved:
``content+diagnosis`` becomes ``correctness`` and ``scaffolding`` remains separate.
Passing ``--skills`` without ``--dimensions`` uses the identity structure. A custom
partition uses ``--dimensions label=skill+skill,label=skill``; every source skill
must appear exactly once.

Examples
--------
    # Historical TutorBench two-dimensional collapse:
    python scripts/kfold_cv_mirt.py

    # InFoBench's native five dimensions:
    python scripts/kfold_cv_mirt.py \
      --matrix runs/calibration/.../response_matrix.csv \
      --rubrics data/InFoBench/rubrics.jsonl \
      --skills content,format,number,style,linguistic --grid 5

    # A pre-specified InFoBench three-dimensional candidate:
    python scripts/kfold_cv_mirt.py \
      --skills content,format,number,style,linguistic \
      --dimensions content_style=content+style,format=format,number_linguistic=number+linguistic

Important metric distinction
----------------------------
Primary metrics estimate held-out-person theta on scoring scenarios and evaluate
disjoint scenarios, so evaluation responses do not leak into theta. For continuity,
the output also includes ``reconstruction_*`` metrics that estimate theta and
reconstruct the same cells; those secondary metrics must not drive model selection.
Scenario-level CAT theta recovery remains a separate study.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, log_expit, logsumexp
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, mod)
    spec.loader.exec_module(mod)
    return mod


# Reuse the EXACT fitter internals (EM/MML, quadrature, collapse, block prep,
# zero-variance dropping) from the calibration script -- never reimplemented here.
cm = _load_module("calibrate_mirt", ROOT / "scripts" / "calibrate_mirt.py")
cp = cm.cp  # calibrate_partial (loader / select_sparse / split_zero_variance)
from tutor_cat.skill_structure import SkillStructure, parse_dimension_spec  # noqa: E402

# Historical defaults retained when neither --skills nor --dimensions is passed.
COLLAPSE_SKILLS = ["content", "diagnosis"]
DIM_LABELS = ["correctness", "scaffolding"]

DEFAULT_MATRIX = ROOT / "staging" / "response_matrix.csv"
DEFAULT_RUBRICS = ROOT / "data" / "TutorBench" / "curated" / "rubrics_qmatrix_curated.jsonl"
DEFAULT_OUT_DIR = ROOT / "staging" / "kfold"

EPS = 1e-12
DEFAULT_MAX_GRID_NODES = 5_000


# ---------------------------------------------------------------------------
# fold construction
# ---------------------------------------------------------------------------


def make_folds(models: list[str], k: int, seed: int) -> list[list[str]]:
    """Partition ``models`` into ``k`` roughly-equal folds by a seeded shuffle."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(models))
    folds: list[list[str]] = [[] for _ in range(k)]
    for pos, i in enumerate(idx):
        folds[pos % k].append(models[int(i)])
    return [sorted(f) for f in folds]


def make_stratified_folds(matrix: pd.DataFrame, k: int, seed: int) -> list[list[str]]:
    """Balance folds over observed pass rate without using any judge metadata.

    Models are ordered by pass rate, split into adjacent ability blocks of size
    ``k``, and one member of each block is assigned to each fold using a seeded
    permutation. This prevents a small fold from accidentally containing only very
    weak or very strong models while remaining deterministic.
    """
    rates = matrix.mean(axis=1, skipna=True)
    if rates.isna().any():
        bad = rates.index[rates.isna()].tolist()
        raise ValueError(f"cannot stratify fully-empty model rows: {bad[:10]}")
    ordered = sorted(matrix.index, key=lambda model: (float(rates.loc[model]), str(model)))
    rng = np.random.default_rng(seed)
    folds: list[list[str]] = [[] for _ in range(k)]
    for start in range(0, len(ordered), k):
        block = ordered[start : start + k]
        destinations = rng.permutation(k)[: len(block)]
        for model, fold_index in zip(block, destinations, strict=True):
            folds[int(fold_index)].append(str(model))
    return [sorted(fold) for fold in folds]


def build_structure(
    source_skills: tuple[str, ...],
    dimensions: str | None,
    name: str | None,
    *,
    historical_default: bool = True,
) -> SkillStructure:
    """Resolve CLI structure flags while preserving the historical default."""
    structure_name = name or "custom"
    if dimensions:
        return parse_dimension_spec(dimensions, source_skills, name=structure_name)

    historical = ("content", "diagnosis", "scaffolding")
    if historical_default and source_skills == historical:
        return SkillStructure.from_groups(
            name or "tutorbench_correctness_scaffolding",
            source_skills,
            [
                ("correctness", ("content", "diagnosis")),
                ("scaffolding", ("scaffolding",)),
            ],
        )
    return SkillStructure.identity(source_skills, name=name)


# ---------------------------------------------------------------------------
# fit / score
# ---------------------------------------------------------------------------


def fit_structure(
    train_df: pd.DataFrame,
    q_by: dict,
    args: argparse.Namespace,
    structure: SkillStructure,
) -> dict:
    """Fit ``structure`` on the training models using the shared M2PL fitter."""
    Y, M, q_source, items, _block_df, diag = cm.prepare_block(train_df, q_by)
    q_modeled = structure.transform_q(q_source)
    fit = cm.fit_m2pl_em(
        Y, M, q_modeled, args.grid,
        estimate_corr=args.estimate_latent_corr,
        ridge=args.ridge, max_iter=args.max_iter, tol=args.tol,
        calibration_model=getattr(
            args, "calibration_model", cm.DEFAULT_CALIBRATION_MODEL
        ),
        log_a_shrinkage=getattr(
            args, "log_a_shrinkage", cm.DEFAULT_LOG_A_SHRINKAGE
        ),
        quadrature_method=getattr(
            args, "quadrature_method", cm.DEFAULT_CALIBRATION_QUADRATURE
        ),
        linear_bound=getattr(
            args, "linear_bound", cm.DEFAULT_QUADRATURE_LINEAR_BOUND
        ),
        convergence_mode=getattr(
            args, "convergence_mode", cm.LEGACY_PRE_MSTEP_CONVERGENCE
        ),
        parameter_tol=getattr(args, "parameter_tol", None),
        consecutive_convergence_passes=getattr(
            args, "consecutive_convergence_passes", 1
        ),
        initial_A=getattr(args, "initial_A", None),
        initial_b=getattr(args, "initial_b", None),
    )
    modeled_pattern_counts: dict[str, int] = {}
    for row in q_modeled:
        key = "".join(str(int(value)) for value in row)
        modeled_pattern_counts[key] = modeled_pattern_counts.get(key, 0) + 1
    diag = {
        **diag,
        "modeled_structure": structure.as_dict(),
        "modeled_q_pattern_counts": modeled_pattern_counts,
        "modeled_items_per_dimension": {
            label: int(q_modeled[:, index].sum())
            for index, label in enumerate(structure.labels)
        },
        "modeled_single_load_anchors": {
            label: int(np.sum((q_modeled[:, index] == 1) & (q_modeled.sum(axis=1) == 1)))
            for index, label in enumerate(structure.labels)
        },
    }
    result = {
        "items": items,
        "A": fit["A"],
        "b": fit["b"],
        "R": fit["R"],
        "dim_labels": list(structure.labels),
        # Backward-compatible key for older report readers.
        "collapsed_labels": list(structure.labels),
        "loglik": fit["loglik"],
        "n_params": fit["n_params"],
        "n_iter": fit["n_iter"],
        "converged": fit["converged"],
        "calibration_specification": fit["calibration_specification"],
        "diag": diag,
    }
    for key in (
        "quadrature_method",
        "quadrature_linear_bound",
        "convergence_mode",
        "penalized_objective",
        "convergence_diagnostics",
    ):
        if key in fit:
            result[key] = fit[key]
    return result


def fit_2skill(
    train_df: pd.DataFrame, q_by: dict, args: argparse.Namespace
) -> dict:
    """Backward-compatible historical TutorBench two-skill wrapper."""
    source = ("content", "diagnosis", "scaffolding")
    return fit_structure(train_df, q_by, args, build_structure(source, None, None))


def eap_predict(
    score_df: pd.DataFrame,
    items: list[str],
    A: np.ndarray,
    b: np.ndarray,
    R: np.ndarray,
    grid: np.ndarray,
    log_prior: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """EAP-score each person in ``score_df`` on the FIXED item params, then predict.

    Holding (A, b) fixed, each person's ability is the posterior mean (EAP) over
    the quadrature grid computed from ONLY their observed responses on ``items``
    (holes are marginalised, exactly as the fitter's E-step does). We then predict
    ``P(pass) = sigmoid(a_j . theta_hat - b_j)`` for every observed cell.

    Returns (y_obs, p_obs, theta, obs_mask): flattened observed labels + predicted
    probabilities over all (person, fitted-item) observed cells, the per-person EAP
    thetas, and the (n_persons, n_items) observation mask.
    """
    sub = score_df.reindex(columns=items)
    Yraw = sub.to_numpy(dtype=float)
    Mobs = ~np.isnan(Yraw)
    Y = np.nan_to_num(Yraw, nan=0.0)
    YM = np.where(Mobs, Y, 0.0)
    NM = np.where(Mobs, 1.0 - Y, 0.0)

    eta = A @ grid.T - b[:, None]          # (n_items, n_nodes)
    logP = log_expit(eta)
    log1mP = log_expit(-eta)

    LL = YM @ logP + NM @ log1mP           # (n_persons, n_nodes)
    joint = LL + log_prior[None, :]
    person_ll = logsumexp(joint, axis=1)
    post = np.exp(joint - person_ll[:, None])   # (n_persons, n_nodes)
    theta = post @ grid                    # (n_persons, n_dims)

    P = expit(theta @ A.T - b[None, :])    # (n_persons, n_items)

    y_obs = Y[Mobs]
    p_obs = P[Mobs]
    return y_obs, p_obs, theta, Mobs


def load_item_scenarios(rubrics_path: Path) -> dict[str, str]:
    """Load criterion-to-scenario provenance for leakage-free test-cell splits."""
    mapping: dict[str, str] = {}
    for record in cp.read_jsonl(rubrics_path):
        criterion_id = record.get("criterion_id")
        scenario_id = record.get("scenario_id")
        if criterion_id is None:
            continue
        if scenario_id is None:
            # A criterion without scenario metadata is its own independent group.
            scenario_id = str(criterion_id)
        if criterion_id in mapping:
            raise ValueError(f"duplicate criterion_id in rubric bank: {criterion_id}")
        mapping[str(criterion_id)] = str(scenario_id)
    return mapping


def split_evaluation_scenarios(
    item_to_scenario: dict[str, str], fraction: float, seed: int
) -> set[str]:
    """Select a deterministic, scenario-disjoint evaluation subset."""
    if not 0.0 < fraction < 1.0:
        raise ValueError("evaluation scenario fraction must be strictly between 0 and 1")
    scenarios = sorted(set(item_to_scenario.values()))
    if len(scenarios) < 2:
        raise ValueError("need at least two scenarios for a disjoint scoring/evaluation split")
    rng = np.random.default_rng(seed)
    shuffled = [scenarios[int(index)] for index in rng.permutation(len(scenarios))]
    n_evaluation = min(len(scenarios) - 1, max(1, int(round(fraction * len(scenarios)))))
    return set(shuffled[:n_evaluation])


def eap_predict_disjoint(
    score_df: pd.DataFrame,
    items: list[str],
    A: np.ndarray,
    b: np.ndarray,
    grid: np.ndarray,
    log_prior: np.ndarray,
    item_to_scenario: dict[str, str],
    evaluation_scenarios: set[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate theta on scoring scenarios and predict disjoint evaluation scenarios.

    The item parameters were fitted only on training people. For each held-out
    person, no response from an evaluation scenario contributes to their theta.
    Thus the returned probabilities are genuine held-out-cell predictions rather
    than posterior reconstructions of the cells used to estimate theta.
    """
    scenario_ids = [item_to_scenario.get(item, item) for item in items]
    evaluation_index = np.array(
        [index for index, scenario in enumerate(scenario_ids) if scenario in evaluation_scenarios],
        dtype=int,
    )
    scoring_index = np.array(
        [
            index
            for index, scenario in enumerate(scenario_ids)
            if scenario not in evaluation_scenarios
        ],
        dtype=int,
    )
    if scoring_index.size == 0 or evaluation_index.size == 0:
        raise ValueError(
            "the fitted item set does not contain both scoring and evaluation scenarios"
        )

    sub = score_df.reindex(columns=items)
    raw = sub.to_numpy(dtype=float)
    scoring_raw = raw[:, scoring_index]
    scoring_observed = ~np.isnan(scoring_raw)
    scoring_y = np.nan_to_num(scoring_raw, nan=0.0)
    successes = np.where(scoring_observed, scoring_y, 0.0)
    failures = np.where(scoring_observed, 1.0 - scoring_y, 0.0)

    scoring_a = A[scoring_index]
    scoring_b = b[scoring_index]
    eta = scoring_a @ grid.T - scoring_b[:, None]
    log_likelihood = successes @ log_expit(eta) + failures @ log_expit(-eta)
    joint = log_likelihood + log_prior[None, :]
    posterior = np.exp(joint - logsumexp(joint, axis=1)[:, None])
    theta = posterior @ grid

    evaluation_raw = raw[:, evaluation_index]
    evaluation_observed = ~np.isnan(evaluation_raw)
    evaluation_y = np.nan_to_num(evaluation_raw, nan=0.0)
    probabilities = expit(theta @ A[evaluation_index].T - b[evaluation_index][None, :])
    return (
        evaluation_y[evaluation_observed],
        probabilities[evaluation_observed],
        theta,
        evaluation_observed,
    )


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y)
    p = np.asarray(p)
    n1 = float(y.sum())
    n0 = float(len(y) - n1)
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(p)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    if y.size == 0 or p.size != y.size:
        raise ValueError("metrics require equally sized, non-empty labels and probabilities")
    logloss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    acc = float(np.mean((p >= 0.5).astype(float) == y))
    brier = float(np.mean((p - y) ** 2))
    return {
        "n_cells": int(len(y)),
        "pos_rate": float(y.mean()),
        "log_loss": logloss,
        "accuracy": acc,
        "auc": _auc(y, p),
        "brier": brier,
    }


def _spread(values: list[float]) -> dict:
    arr = np.asarray([v for v in values if v == v], dtype=float)  # drop NaN
    if len(arr) == 0:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def select_calibration_spec_one_se(
    candidates: list[dict],
    simplicity_order: list[str],
) -> dict:
    """Apply a predeclared one-standard-error rule to inner-CV candidates.

    Each candidate must contain ``mean_log_loss``, ``family_cluster_se``,
    ``eligible``, and a complete ``calibration_specification`` with a ``cache_key``.
    The cutoff is the best mean loss plus *that best candidate's* family-clustered
    standard error. The first within-cutoff cache key in ``simplicity_order`` wins.
    No outer-fold outcome is accepted by this helper, which keeps model selection
    isolated from outer CAT performance.
    """
    if not candidates:
        raise ValueError("one-SE selection requires at least one candidate")
    eligible = [row for row in candidates if bool(row.get("eligible", False))]
    if not eligible:
        raise ValueError("one-SE selection has no eligible calibration specification")
    by_key: dict[str, dict] = {}
    for row in candidates:
        spec = row.get("calibration_specification")
        if not isinstance(spec, dict) or not isinstance(spec.get("cache_key"), str):
            raise ValueError("every candidate needs a complete calibration_specification")
        key = spec["cache_key"]
        if key in by_key:
            raise ValueError(f"duplicate calibration specification cache key: {key}")
        by_key[key] = row
    if len(set(simplicity_order)) != len(simplicity_order):
        raise ValueError("simplicity_order contains duplicate cache keys")
    eligible_keys = {row["calibration_specification"]["cache_key"] for row in eligible}
    if not eligible_keys.issubset(set(simplicity_order)):
        missing = sorted(eligible_keys - set(simplicity_order))
        raise ValueError(f"simplicity_order is missing eligible specification(s): {missing}")

    for row in eligible:
        mean = row.get("mean_log_loss")
        se = row.get("family_cluster_se")
        if not isinstance(mean, (int, float)) or not np.isfinite(mean):
            raise ValueError("eligible candidate mean_log_loss must be finite")
        if not isinstance(se, (int, float)) or not np.isfinite(se) or se < 0:
            raise ValueError("eligible candidate family_cluster_se must be finite and non-negative")
    best = min(eligible, key=lambda row: float(row["mean_log_loss"]))
    cutoff = float(best["mean_log_loss"]) + float(best["family_cluster_se"])
    within = {
        row["calibration_specification"]["cache_key"]
        for row in eligible
        if float(row["mean_log_loss"]) <= cutoff + 1e-15
    }
    selected_key = next((key for key in simplicity_order if key in within), None)
    if selected_key is None:  # guarded by the subset validation above
        raise ValueError("no within-one-SE candidate appears in simplicity_order")
    return {
        "selected": by_key[selected_key],
        "selected_cache_key": selected_key,
        "empirical_best_cache_key": best["calibration_specification"]["cache_key"],
        "one_se_cutoff": cutoff,
        "within_one_se_cache_keys": [key for key in simplicity_order if key in within],
        "selection_basis": "inner-family-clustered-disjoint-log-loss",
    }


# ---------------------------------------------------------------------------
# item-param cross-fold stability
# ---------------------------------------------------------------------------


def item_param_tables(fold_fits: list[dict]) -> dict[str, pd.DataFrame]:
    """Wide per-parameter tables: rows = criterion_id, cols = fold_<f>."""
    if not fold_fits:
        return {}
    labels = list(fold_fits[0]["dim_labels"])
    if any(list(fit["dim_labels"]) != labels for fit in fold_fits):
        raise ValueError("all fold fits must use the same ordered dimension labels")
    tables: dict[str, dict] = {f"a_{label}": {} for label in labels}
    tables["b"] = {}
    for f, fit in enumerate(fold_fits):
        A, b, items = fit["A"], fit["b"], fit["items"]
        for index, label in enumerate(labels):
            tables[f"a_{label}"][f"fold_{f}"] = pd.Series(A[:, index], index=items)
        tables["b"][f"fold_{f}"] = pd.Series(b, index=items)
    return {name: pd.DataFrame(cols) for name, cols in tables.items()}


def item_params_frame(fit: dict) -> pd.DataFrame:
    """Long item table with one discrimination column per modeled dimension."""
    data: dict[str, object] = {"criterion_id": fit["items"]}
    for index, label in enumerate(fit["dim_labels"]):
        data[f"a_{label}"] = np.round(fit["A"][:, index], 6)
    data["b"] = np.round(fit["b"], 6)
    return pd.DataFrame(data)


def cross_fold_stability(fold_fits: list[dict]) -> tuple[dict, dict[str, pd.DataFrame]]:
    """Pairwise cross-fold Pearson correlations of each item parameter.

    Only items fit in BOTH folds of a pair contribute to that pair's correlation.
    Returns (summary, wide_tables). ``summary`` reports, per parameter, the pairwise
    correlation matrix, the median off-diagonal correlation, and the median number
    of shared items per pair.
    """
    tables = item_param_tables(fold_fits)
    k = len(fold_fits)
    summary: dict[str, dict] = {}
    for name, df in tables.items():
        pair_corrs: list[float] = []
        pair_ns: list[int] = []
        mat = np.full((k, k), np.nan)
        for i in range(k):
            for j in range(k):
                if i == j:
                    mat[i, j] = 1.0
                    continue
                a = df[f"fold_{i}"]
                b = df[f"fold_{j}"]
                common = a.notna() & b.notna()
                n = int(common.sum())
                if n >= 3:
                    va = a[common].to_numpy()
                    vb = b[common].to_numpy()
                    if va.std() > 0 and vb.std() > 0:
                        c = float(np.corrcoef(va, vb)[0, 1])
                    else:
                        c = float("nan")
                else:
                    c = float("nan")
                mat[i, j] = c
                if i < j:
                    pair_corrs.append(c)
                    pair_ns.append(n)
        valid = [c for c in pair_corrs if c == c]
        summary[name] = {
            "pairwise_matrix": [[None if v != v else round(float(v), 4) for v in row]
                                for row in mat],
            "median_pairwise_corr": (float(np.median(valid)) if valid else None),
            "mean_pairwise_corr": (float(np.mean(valid)) if valid else None),
            "n_fold_pairs": len(pair_corrs),
            "median_shared_items_per_pair": (int(np.median(pair_ns)) if pair_ns else 0),
        }
    return summary, tables


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


def run(args: argparse.Namespace) -> int:
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    source_skills = cm.configure_skills(getattr(args, "skills", None))
    structure = build_structure(
        tuple(source_skills),
        getattr(args, "dimensions", None),
        getattr(args, "structure_name", None),
        historical_default=getattr(args, "skills", None) is None,
    )
    if args.k < 2:
        raise ValueError("--k must be at least 2")
    if args.grid < 2:
        raise ValueError("--grid must be at least 2")
    if args.ridge < 0:
        raise ValueError("--ridge cannot be negative")
    calibration_model = getattr(args, "calibration_model", cm.DEFAULT_CALIBRATION_MODEL)
    log_a_shrinkage = getattr(
        args, "log_a_shrinkage", cm.DEFAULT_LOG_A_SHRINKAGE
    )
    calibration_spec = cm.calibration_specification(
        calibration_model,
        ridge=args.ridge,
        log_a_shrinkage=log_a_shrinkage,
    )
    if args.max_iter < 1:
        raise ValueError("--max-iter must be positive")
    if args.tol <= 0:
        raise ValueError("--tol must be positive")
    grid_total = args.grid ** structure.n_dims
    max_grid_nodes = getattr(args, "max_grid_nodes", DEFAULT_MAX_GRID_NODES)
    if max_grid_nodes < 1:
        raise ValueError("--max-grid-nodes must be positive")
    if grid_total > max_grid_nodes and not getattr(args, "allow_large_grid", False):
        raise ValueError(
            f"requested {args.grid}^{structure.n_dims} = {grid_total:,} quadrature nodes, "
            f"above --max-grid-nodes={max_grid_nodes:,}"
        )

    mat = cm.load_matrix_strict(args.matrix)
    q_by = cm.load_q_matrix(args.rubrics)
    alignment = cm.validate_matrix_bank_alignment(
        mat, q_by, getattr(args, "require_complete_bank", False)
    )
    models = list(mat.index)
    n_models = len(models)
    if args.k > n_models:
        raise ValueError(f"--k={args.k} exceeds the {n_models} available models")
    print(f"loaded matrix: {n_models} models x {mat.shape[1]} criteria")
    print(f"Q-matrix source: {args.rubrics} ({len(q_by)} criteria with q_mapping)")
    print(f"modeled structure: {structure.name} -> {structure.groups}")
    print(f"calibration specification: {calibration_spec['family']} "
          f"({calibration_spec['cache_key']})")

    # Shared quadrature grid (same GH nodes/weights as the fitter).
    grid = cm.build_grid(structure.n_dims, args.grid)
    base_logw = cm.base_log_weights(structure.n_dims, args.grid)

    item_to_scenario = load_item_scenarios(args.rubrics)
    if getattr(args, "evaluation_split_seed", None) is None:
        args.evaluation_split_seed = args.seed
    evaluation_scenarios = split_evaluation_scenarios(
        item_to_scenario,
        getattr(args, "evaluation_scenario_fraction", 0.20),
        args.evaluation_split_seed,
    )

    fold_strategy = getattr(args, "fold_strategy", "stratified")
    folds = (
        make_stratified_folds(mat, args.k, args.seed)
        if fold_strategy == "stratified"
        else make_folds(models, args.k, args.seed)
    )
    fold_of = {m: f for f, fold in enumerate(folds) for m in fold}
    print(f"\nk={args.k} {fold_strategy} folds (seed={args.seed}): "
          f"sizes {[len(f) for f in folds]}")
    print(f"leakage-free evaluation scenarios: {len(evaluation_scenarios)}")

    # ---- per-fold fit + held-out scoring ----
    fold_fits: list[dict] = []
    per_fold_metrics: list[dict] = []
    all_y: list[np.ndarray] = []
    all_p: list[np.ndarray] = []
    all_y_reconstruction: list[np.ndarray] = []
    all_p_reconstruction: list[np.ndarray] = []

    for f in range(args.k):
        test_models = folds[f]
        train_models = [m for m in models if m not in set(test_models)]
        print(f"\n=== fold {f}: TRAIN={len(train_models)} TEST={len(test_models)} ===")
        print(f"  fitting {structure.n_dims}-skill M2PL on TRAIN ...", flush=True)
        fit = fit_structure(mat.loc[train_models], q_by, args, structure)
        fold_fits.append(fit)
        print(f"  fitted {len(fit['items'])} items "
              f"(loglik={fit['loglik']:.1f}, iters={fit['n_iter']}, "
              f"converged={fit['converged']})")

        log_prior = cm.prior_log_weights(grid, base_logw, fit["R"])
        y_reconstruction, p_reconstruction, _theta_all, _ = eap_predict(
            mat.loc[test_models], fit["items"], fit["A"], fit["b"], fit["R"],
            grid, log_prior,
        )
        y, p, _theta_split, _ = eap_predict_disjoint(
            mat.loc[test_models], fit["items"], fit["A"], fit["b"], grid,
            log_prior, item_to_scenario, evaluation_scenarios,
        )
        m = metrics(y, p)
        reconstruction = metrics(y_reconstruction, p_reconstruction)
        for metric_name, value in reconstruction.items():
            m[f"reconstruction_{metric_name}"] = value
        m["fold"] = f
        m["n_train"] = len(train_models)
        m["n_test"] = len(test_models)
        m["n_items_fit"] = len(fit["items"])
        m["fit_converged"] = bool(fit["converged"])
        per_fold_metrics.append(m)
        all_y.append(y)
        all_p.append(p)
        all_y_reconstruction.append(y_reconstruction)
        all_p_reconstruction.append(p_reconstruction)
        print(f"  OOS disjoint: cells={m['n_cells']} logloss={m['log_loss']:.4f} "
              f"acc={m['accuracy']:.4f} auc={m['auc']:.4f} brier={m['brier']:.4f}")

        # per-fold item params
        item_params_frame(fit).to_csv(out_dir / f"fold_{f}_item_params.csv", index=False)

    # pooled OOS (all held-out cells across folds)
    y_all = np.concatenate(all_y)
    p_all = np.concatenate(all_p)
    pooled = metrics(y_all, p_all)
    pooled_reconstruction = metrics(
        np.concatenate(all_y_reconstruction), np.concatenate(all_p_reconstruction)
    )
    print(f"\n=== POOLED OOS DISJOINT: cells={pooled['n_cells']} "
          f"logloss={pooled['log_loss']:.4f} acc={pooled['accuracy']:.4f} "
          f"auc={pooled['auc']:.4f} brier={pooled['brier']:.4f} ===")

    # ---- in-sample baseline ----
    print(f"\n=== in-sample baseline: fit and score all {n_models} ===", flush=True)
    full_fit = fit_structure(mat, q_by, args, structure)
    log_prior_full = cm.prior_log_weights(grid, base_logw, full_fit["R"])
    yi_reconstruction, pi_reconstruction, _theta_full, _ = eap_predict(
        mat, full_fit["items"], full_fit["A"], full_fit["b"], full_fit["R"],
        grid, log_prior_full,
    )
    yi, pi, _theta_full_split, _ = eap_predict_disjoint(
        mat, full_fit["items"], full_fit["A"], full_fit["b"], grid,
        log_prior_full, item_to_scenario, evaluation_scenarios,
    )
    insample = metrics(yi, pi)
    insample_reconstruction = metrics(yi_reconstruction, pi_reconstruction)
    print(f"  in-sample: cells={insample['n_cells']} logloss={insample['log_loss']:.4f} "
          f"acc={insample['accuracy']:.4f} auc={insample['auc']:.4f} "
          f"brier={insample['brier']:.4f}")

    item_params_frame(full_fit).to_csv(
        out_dir / "insample_baseline_item_params.csv", index=False
    )

    # ---- item-param cross-fold stability ----
    print("\n=== item-param cross-fold stability ===")
    stability, wide_tables = cross_fold_stability(fold_fits)
    for name, s in stability.items():
        print(f"  {name}: median pairwise corr = {s['median_pairwise_corr']} "
              f"(median shared items/pair = {s['median_shared_items_per_pair']})")

    # Also correlate each fold against the all-model reference parameters.
    full_series = {
        f"a_{label}": pd.Series(full_fit["A"][:, index], index=full_fit["items"])
        for index, label in enumerate(structure.labels)
    }
    full_series["b"] = pd.Series(full_fit["b"], index=full_fit["items"])
    fold_vs_full: dict[str, list] = {}
    for name, df in wide_tables.items():
        ref = full_series[name]
        corrs = []
        for f in range(args.k):
            col = df[f"fold_{f}"]
            common = col.notna() & ref.notna()
            if int(common.sum()) >= 3 and col[common].std() > 0 and ref[common].std() > 0:
                corrs.append(float(np.corrcoef(col[common], ref[common])[0, 1]))
            else:
                corrs.append(float("nan"))
        fold_vs_full[name] = corrs
        stability[name]["median_fold_vs_full_corr"] = (
            float(np.nanmedian(corrs)) if any(c == c for c in corrs) else None
        )

    # ---- write outputs ----
    _write_outputs(
        out_dir, args, structure, alignment, folds, fold_of, per_fold_metrics,
        pooled, pooled_reconstruction, insample, insample_reconstruction,
        stability, wide_tables, fold_vs_full, full_fit, mat, models,
        evaluation_scenarios,
    )

    print(f"\nall outputs written under {out_dir}")
    return 0


def _write_outputs(
    out_dir,
    args,
    structure,
    alignment,
    folds,
    fold_of,
    per_fold_metrics,
    pooled,
    pooled_reconstruction,
    insample,
    insample_reconstruction,
    stability,
    wide_tables,
    fold_vs_full,
    full_fit,
    mat,
    models,
    evaluation_scenarios,
):
    # fold assignments
    with (out_dir / "fold_assignments.json").open("w", encoding="utf-8") as fh:
        json.dump({"k": args.k, "seed": args.seed,
                   "strategy": getattr(args, "fold_strategy", "stratified"),
                   "folds": {str(f): folds[f] for f in range(args.k)},
                   "model_to_fold": fold_of}, fh, indent=2)
    pd.DataFrame({"model": models, "fold": [fold_of[m] for m in models],
                  "observed_pass_rate": [float(mat.loc[m].mean(skipna=True)) for m in models]}) \
        .sort_values(["fold", "model"]).to_csv(out_dir / "fold_assignments.csv", index=False)

    # per-fold metrics csv
    pf_cols = ["fold", "n_train", "n_test", "n_items_fit", "fit_converged",
               "n_cells", "pos_rate", "log_loss", "accuracy", "auc", "brier",
               "reconstruction_n_cells", "reconstruction_pos_rate",
               "reconstruction_log_loss", "reconstruction_accuracy",
               "reconstruction_auc", "reconstruction_brier"]
    pd.DataFrame(per_fold_metrics)[pf_cols].to_csv(
        out_dir / "metrics_per_fold.csv", index=False)

    # aggregate metrics json (pooled OOS, per-fold spread, in-sample gap)
    spread = {m: _spread([pf[m] for pf in per_fold_metrics])
              for m in ["log_loss", "accuracy", "auc", "brier"]}
    gap = {m: (pooled[m] - insample[m]) for m in ["log_loss", "accuracy", "auc", "brier"]}
    aggregate = {
        "generated_at": _utcnow(),
        "k": args.k, "grid": args.grid, "seed": args.seed,
        "estimate_latent_corr": args.estimate_latent_corr,
        "calibration_specification": full_fit["calibration_specification"],
        "metric_kind": "heldout_person_disjoint_scenario_prediction",
        "pooled_oos": pooled,
        "pooled_oos_reconstruction": pooled_reconstruction,
        "per_fold_spread": spread,
        "in_sample_baseline": insample,
        "in_sample_reconstruction": insample_reconstruction,
        "oos_minus_insample_gap": gap,
        "note": (
            "Primary OOS metrics estimate held-out-person theta on scoring scenarios "
            "and evaluate disjoint scenarios. Reconstruction metrics use the same "
            "held-out-person cells to estimate theta and reconstruct them; do not use "
            "reconstruction metrics alone for model selection. Gap = primary pooled "
            "OOS metric - primary in-sample metric."
        ),
    }
    with (out_dir / "metrics_aggregate.json").open("w", encoding="utf-8") as fh:
        json.dump(aggregate, fh, indent=2)

    # in-sample baseline json
    with (out_dir / "insample_baseline.json").open("w", encoding="utf-8") as fh:
        json.dump({"generated_at": _utcnow(), "metrics": insample,
                   "reconstruction_metrics": insample_reconstruction,
                   "n_items_fit": len(full_fit["items"]),
                   "loglik": full_fit["loglik"],
                   "latent_correlation": np.round(np.asarray(full_fit["R"]), 6).tolist()},
                  fh, indent=2)

    # stability tables
    with (out_dir / "item_param_stability.json").open("w", encoding="utf-8") as fh:
        json.dump({"generated_at": _utcnow(),
                   "stability": stability,
                   "fold_vs_full_corr": fold_vs_full}, fh, indent=2)
    for parameter, table in wide_tables.items():
        table.index.name = "criterion_id"
        table.to_csv(out_dir / f"item_param_{parameter}_by_fold.csv")
    # long-format stability summary csv
    rows = []
    for name, s in stability.items():
        rows.append({
            "parameter": name,
            "median_pairwise_corr": s["median_pairwise_corr"],
            "mean_pairwise_corr": s["mean_pairwise_corr"],
            "median_fold_vs_full_corr": s.get("median_fold_vs_full_corr"),
            "n_fold_pairs": s["n_fold_pairs"],
            "median_shared_items_per_pair": s["median_shared_items_per_pair"],
        })
    pd.DataFrame(rows).to_csv(out_dir / "item_param_stability.csv", index=False)

    # full summary
    with (out_dir / "kfold_summary.json").open("w", encoding="utf-8") as fh:
        json.dump({
            "generated_at": _utcnow(),
            "provenance": {"script": "scripts/kfold_cv_mirt.py", "argv": sys.argv[1:],
                           "matrix": str(args.matrix), "rubrics": str(args.rubrics)},
            "config": {"k": args.k, "grid": args.grid, "seed": args.seed,
                       "ridge": args.ridge, "max_iter": args.max_iter, "tol": args.tol,
                       "calibration_specification": full_fit[
                           "calibration_specification"
                       ],
                       "estimate_latent_corr": args.estimate_latent_corr,
                       "fold_strategy": getattr(args, "fold_strategy", "stratified"),
                       "evaluation_scenario_fraction": getattr(
                           args, "evaluation_scenario_fraction", 0.20
                       ),
                       "evaluation_split_seed": getattr(
                           args, "evaluation_split_seed", args.seed
                       ),
                       "structure": structure.as_dict()},
            "matrix_bank_alignment": alignment,
            "n_models": len(models),
            "fold_sizes": [len(f) for f in folds],
            "n_evaluation_scenarios": len(evaluation_scenarios),
            "evaluation_scenarios": sorted(evaluation_scenarios),
            "per_fold_metrics": per_fold_metrics,
            "metric_kind": "heldout_person_disjoint_scenario_prediction",
            "pooled_oos": pooled,
            "pooled_oos_reconstruction": pooled_reconstruction,
            "per_fold_spread": spread,
            "in_sample_baseline": insample,
            "in_sample_reconstruction": insample_reconstruction,
            "oos_minus_insample_gap": gap,
            "full_fit": {
                "n_items": len(full_fit["items"]),
                "loglik": full_fit["loglik"],
                "n_params": full_fit["n_params"],
                "n_iter": full_fit["n_iter"],
                "converged": full_fit["converged"],
                "calibration_specification": full_fit[
                    "calibration_specification"
                ],
                "diagnostics": full_fit["diag"],
            },
            "item_param_stability": stability,
        }, fh, indent=2)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX,
                   help=f"response matrix CSV (default: {DEFAULT_MATRIX}).")
    p.add_argument("--rubrics", type=Path, default=DEFAULT_RUBRICS,
                   help=f"rubric bank JSONL for the Q-matrix (default: {DEFAULT_RUBRICS}).")
    p.add_argument(
        "--skills",
        default=None,
        metavar="SKILL,SKILL[,...]",
        help=(
            "ordered source Q-matrix skills. Omit for the historical package axis; "
            "InFoBench uses content,format,number,style,linguistic."
        ),
    )
    p.add_argument(
        "--dimensions",
        default=None,
        metavar="LABEL=SKILL[+SKILL],...",
        help=(
            "modeled skill partition. Every source skill must appear exactly once. "
            "If --skills is explicit and this is omitted, the identity structure is used."
        ),
    )
    p.add_argument(
        "--structure-name",
        default=None,
        help="stable name recorded with the modeled skill structure.",
    )
    p.add_argument(
        "--require-complete-bank",
        action="store_true",
        help="fail unless the matrix contains every active rubric-bank criterion.",
    )
    p.add_argument("--k", type=int, default=5, help="number of person folds (default 5).")
    p.add_argument("--grid", type=int, default=7,
                   help="Gauss-Hermite nodes per latent dimension (default 7).")
    p.add_argument(
        "--max-grid-nodes",
        type=int,
        default=DEFAULT_MAX_GRID_NODES,
        help="safety cap on grid ** modeled_dimensions (default 5000).",
    )
    p.add_argument(
        "--allow-large-grid",
        action="store_true",
        help="deliberately override --max-grid-nodes.",
    )
    p.add_argument("--seed", type=int, default=20260729, help="shuffle seed for folds.")
    p.add_argument(
        "--fold-strategy",
        choices=("stratified", "random"),
        default="stratified",
        help="balance folds over model pass rate or use the legacy random shuffle.",
    )
    p.add_argument(
        "--evaluation-scenario-fraction",
        type=float,
        default=0.20,
        help=(
            "fraction of scenarios held out from theta scoring and used only for "
            "leakage-free held-out-cell metrics (default 0.20)."
        ),
    )
    p.add_argument(
        "--evaluation-split-seed",
        type=int,
        default=None,
        help="scenario scoring/evaluation split seed (default: --seed).",
    )
    p.add_argument("--estimate-latent-corr", action="store_true",
                   help="re-estimate the latent correlation each EM iteration (default off).")
    p.add_argument("--ridge", type=float, default=1e-3,
                   help="L2 ridge on free-2pl loadings (default 1e-3; matches Run 1).")
    p.add_argument(
        "--calibration-model",
        choices=cm.CALIBRATION_MODELS,
        default=cm.DEFAULT_CALIBRATION_MODEL,
        help=(
            "item discrimination family: historical free-2pl (default), "
            "fixed-a=1 1pl, or strictly-positive log-shrinkage-2pl."
        ),
    )
    p.add_argument(
        "--log-a-shrinkage",
        type=float,
        default=cm.DEFAULT_LOG_A_SHRINKAGE,
        help=(
            "log(a) penalty precision for log-shrinkage-2pl; equivalent prior "
            "SD is 1/sqrt(value)."
        ),
    )
    p.add_argument("--max-iter", type=int, default=200, help="max EM iterations.")
    p.add_argument("--tol", type=float, default=1e-4, help="EM convergence tol on loglik.")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                   help=f"output directory (default: {DEFAULT_OUT_DIR}).")
    return p


def main() -> int:
    try:
        return run(build_argparser().parse_args())
    except (ValueError, FileNotFoundError, cm.CalibrationError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
