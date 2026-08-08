"""Focused synthetic tests for the v2 calibration-family interface."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


cm = _load("calibrate_mirt_family_tests", ROOT / "scripts" / "calibrate_mirt.py")
kf = _load("kfold_cv_mirt_family_tests", ROOT / "scripts" / "kfold_cv_mirt.py")


def _simulate_1d(seed: int = 91, n_people: int = 180, n_items: int = 8):
    rng = np.random.default_rng(seed)
    theta = rng.standard_normal(n_people)
    difficulty = np.linspace(-1.0, 1.0, n_items)
    probabilities = 1.0 / (1.0 + np.exp(-(theta[:, None] - difficulty[None, :])))
    y = (rng.random(probabilities.shape) < probabilities).astype(float)
    mask = np.ones_like(y, dtype=bool)
    q = np.ones((n_items, 1), dtype=int)
    return y, mask, q, difficulty


def test_canonical_specs_are_complete_and_cache_distinct() -> None:
    free_a = cm.calibration_specification(cm.FREE_2PL, ridge=0.001)
    free_b = cm.calibration_specification(cm.FREE_2PL, ridge=0.1)
    one = cm.calibration_specification(cm.ONE_PL, ridge=999.0)
    shrink_1 = cm.calibration_specification(cm.LOG_SHRINKAGE_2PL, log_a_shrinkage=4.0)
    shrink_2 = cm.calibration_specification(cm.LOG_SHRINKAGE_2PL, log_a_shrinkage=16.0)

    assert free_a["regularization"]["ridge"] == 0.001
    assert free_a["cache_key"] != free_b["cache_key"]
    assert one["discrimination"]["fixed_a"] == 1.0
    assert one["regularization"]["ridge"] is None
    assert shrink_1["discrimination"]["parameterization"] == "a=exp(log_a)"
    assert shrink_1["regularization"]["log_a_prior_sd"] == pytest.approx(0.5)
    assert shrink_1["cache_key"] != shrink_2["cache_key"]


def test_1pl_fixes_active_loadings_and_counts_only_difficulties() -> None:
    y, mask, q, difficulty = _simulate_1d()
    fit = cm.fit_m2pl_em(
        y,
        mask,
        q,
        nodes_per_dim=7,
        max_iter=80,
        tol=1e-5,
        calibration_model=cm.ONE_PL,
    )
    assert np.array_equal(fit["A"], np.ones_like(fit["A"]))
    assert fit["n_free_loadings"] == 0
    assert fit["n_fixed_loadings"] == q.size
    assert fit["n_params"] == q.shape[0]
    assert np.all(np.isfinite(fit["b"]))
    assert np.corrcoef(fit["b"], difficulty)[0, 1] > 0.85
    assert fit["calibration_specification"]["family"] == cm.ONE_PL


def test_1pl_parameter_count_includes_estimated_latent_correlations() -> None:
    rng = np.random.default_rng(10)
    q = np.array([[1, 0], [0, 1], [1, 1], [1, 0], [0, 1]], dtype=int)
    y = rng.integers(0, 2, size=(60, len(q))).astype(float)
    fit = cm.fit_m2pl_em(
        y,
        np.ones_like(y, dtype=bool),
        q,
        nodes_per_dim=3,
        estimate_corr=True,
        max_iter=2,
        calibration_model=cm.ONE_PL,
    )
    # Five item difficulties plus one free off-diagonal latent correlation.
    assert fit["n_params"] == 6
    assert np.array_equal(fit["A"], q.astype(float))


def test_log_shrinkage_is_positive_and_stronger_penalty_moves_a_toward_one() -> None:
    grid = np.linspace(-2.5, 2.5, 31)[:, None]
    counts = np.full(len(grid), 80.0)
    successes = counts * (1.0 / (1.0 + np.exp(-(3.0 * grid[:, 0] - 0.25))))

    weak, _ = cm._fit_item_log_shrinkage(
        grid, successes, counts, strength=0.01, max_iter=500
    )
    strong, _ = cm._fit_item_log_shrinkage(
        grid, successes, counts, strength=1000.0, max_iter=500
    )
    assert weak[0] > 0 and strong[0] > 0
    assert abs(strong[0] - 1.0) < abs(weak[0] - 1.0)

    y, mask, q, _ = _simulate_1d(seed=33, n_people=100, n_items=5)
    fitted = cm.fit_m2pl_em(
        y,
        mask,
        q,
        nodes_per_dim=5,
        max_iter=15,
        calibration_model=cm.LOG_SHRINKAGE_2PL,
        log_a_shrinkage=4.0,
    )
    assert np.all(fitted["A"][q == 1] > 0)
    assert fitted["n_params"] == 2 * q.shape[0]
    assert fitted["calibration_specification"]["regularization"][
        "log_a_prior_sd"
    ] == pytest.approx(0.5)


def test_default_fit_is_exactly_the_explicit_legacy_free_2pl_path() -> None:
    y, mask, q, _ = _simulate_1d(seed=7, n_people=90, n_items=5)
    kwargs = dict(nodes_per_dim=5, ridge=0.01, max_iter=10, tol=1e-5)
    legacy = cm.fit_m2pl_em(y, mask, q, **kwargs)
    explicit = cm.fit_m2pl_em(
        y, mask, q, calibration_model=cm.FREE_2PL, **kwargs
    )
    assert np.array_equal(legacy["A"], explicit["A"])
    assert np.array_equal(legacy["b"], explicit["b"])
    assert legacy["loglik"] == explicit["loglik"]
    assert legacy["n_params"] == explicit["n_params"] == 2 * q.shape[0]
    assert legacy["n_iter"] == explicit["n_iter"]
    assert legacy["calibration_specification"] == explicit["calibration_specification"]


def test_manifest_records_the_exact_fit_specification(tmp_path: Path) -> None:
    spec = cm.calibration_specification(
        cm.LOG_SHRINKAGE_2PL, ridge=0.77, log_a_shrinkage=9.0
    )
    base_fit = {
        "n_dims": 1,
        "loglik": -20.0,
        "n_params": 6,
        "grid_nodes": 5,
        "n_iter": 3,
        "converged": True,
        "calibration_specification": spec,
    }
    args = SimpleNamespace(
        grid=5,
        ridge=0.77,
        calibration_model=cm.LOG_SHRINKAGE_2PL,
        log_a_shrinkage=9.0,
        estimate_latent_corr=False,
        max_iter=10,
        tol=1e-4,
        min_persons_identifiable=5,
    )
    diag = {
        "n_items_fit": 3,
        "n_persons_fit": 10,
        "dropped_zero_variance_total": 0,
        "dropped_all_fail": 0,
        "dropped_all_pass": 0,
        "dropped_missing_qrow": 0,
    }
    path = cm.write_manifest(
        tmp_path,
        args,
        diag,
        dict(base_fit),
        dict(base_fit),
        n_obs=30,
        latent_corr=None,
        crosscheck={},
        efa={},
        matrix_prov={},
        identifiable=True,
    )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["method"] == "confirmatory-positive-log-shrinkage-m2pl-mml-em"
    assert manifest["calibration_specification"] == spec
    assert manifest["comparison"]["multi"]["calibration_specification"] == spec
    assert manifest["calibration_specification"]["regularization"]["ridge"] is None


def test_fit_structure_propagates_complete_specification() -> None:
    y, _, q, _ = _simulate_1d(seed=12, n_people=50, n_items=5)
    items = [f"item_{index}" for index in range(q.shape[0])]
    import pandas as pd

    frame = pd.DataFrame(y, columns=items, index=[f"m{index}" for index in range(len(y))])
    q_by = {item: q[index] for index, item in enumerate(items)}
    structure = kf.SkillStructure.identity(("content",), name="one_dim")
    old_skills = tuple(kf.cm.SKILLS)
    try:
        kf.cm.configure_skills("content")
        args = SimpleNamespace(
            grid=41,
            estimate_latent_corr=False,
            ridge=0.01,
            max_iter=5,
            tol=1e-4,
            calibration_model=cm.LOG_SHRINKAGE_2PL,
            log_a_shrinkage=9.0,
            quadrature_method=cm.NORMAL_TRAPEZOID_QUADRATURE,
            linear_bound=7.0,
            convergence_mode=cm.RETURNED_ITERATE_CONVERGENCE,
            parameter_tol=1e-3,
        )
        fit = kf.fit_structure(frame, q_by, args, structure)
    finally:
        kf.cm.configure_skills(",".join(old_skills))
    assert fit["calibration_specification"]["family"] == cm.LOG_SHRINKAGE_2PL
    assert fit["calibration_specification"]["regularization"][
        "log_a_prior_sd"
    ] == pytest.approx(1 / 3)
    assert fit["quadrature_method"] == cm.NORMAL_TRAPEZOID_QUADRATURE
    assert fit["quadrature_linear_bound"] == 7.0
    assert fit["convergence_mode"] == cm.RETURNED_ITERATE_CONVERGENCE
    assert "convergence_diagnostics" in fit


def test_one_se_helper_selects_predeclared_simplest_eligible_spec() -> None:
    one = cm.calibration_specification(cm.ONE_PL)
    shrink = cm.calibration_specification(cm.LOG_SHRINKAGE_2PL, log_a_shrinkage=9.0)
    free = cm.calibration_specification(cm.FREE_2PL, ridge=0.01)
    candidates = [
        {
            "calibration_specification": one,
            "mean_log_loss": 0.414,
            "family_cluster_se": 0.004,
            "eligible": True,
        },
        {
            "calibration_specification": shrink,
            "mean_log_loss": 0.410,
            "family_cluster_se": 0.006,
            "eligible": True,
        },
        {
            "calibration_specification": free,
            "mean_log_loss": 0.405,
            "family_cluster_se": 0.010,
            "eligible": True,
        },
    ]
    selected = kf.select_calibration_spec_one_se(
        candidates,
        [one["cache_key"], shrink["cache_key"], free["cache_key"]],
    )
    # Best + its SE = .415, so all three are tied and the predeclared simplest wins.
    assert selected["empirical_best_cache_key"] == free["cache_key"]
    assert selected["selected_cache_key"] == one["cache_key"]
    assert selected["one_se_cutoff"] == pytest.approx(0.415)
