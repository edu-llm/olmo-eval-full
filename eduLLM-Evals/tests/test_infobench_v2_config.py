from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import calibrate_mirt as cm  # noqa: E402

CONFIG = ROOT / "configs" / "infobench_calibration_cat_v2.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_v2_inputs_and_repeated_splits_are_hash_pinned() -> None:
    config = _load()
    assert config["schema_version"] == "infobench-calibration-cat-v2-v1"
    assert config["status"] == "frozen_pre_run"
    baseline = config["baseline"]
    for path_key, hash_key in (
        ("response_matrix", "response_matrix_sha256"),
        ("judge_manifest", "judge_manifest_sha256"),
        ("rubrics", "rubrics_sha256"),
        ("scenarios", "scenarios_sha256"),
    ):
        assert _sha256(ROOT / baseline[path_key]) == baseline[hash_key]

    cv = config["cross_validation"]
    for path_key, hash_key in (
        ("split_config", "split_config_sha256"),
        ("split_manifest", "split_manifest_sha256"),
        ("model_assignments", "model_assignments_sha256"),
        ("scenario_assignments", "scenario_assignments_sha256"),
    ):
        assert _sha256(ROOT / cv[path_key]) == cv[hash_key]
    assert cv["repetitions"] == 5
    assert cv["outer_folds_per_repetition"] == 5
    assert cv["inner_folds_per_outer_panel"] == 4
    assert cv["treat_model_repeat_rows_as_independent"] is False


def test_v2_exact_calibration_specifications_match_the_fitter() -> None:
    configured = _load()["calibration_specifications"]
    expected = [
        ("1pl_fixed_a1", cm.ONE_PL, 0.0, 1.0),
        ("log_shrinkage_2pl_lambda16", cm.LOG_SHRINKAGE_2PL, 0.0, 16.0),
        ("log_shrinkage_2pl_lambda4", cm.LOG_SHRINKAGE_2PL, 0.0, 4.0),
        ("free_2pl_ridge_0p1", cm.FREE_2PL, 0.1, 1.0),
        ("free_2pl_ridge_0p01", cm.FREE_2PL, 0.01, 1.0),
        ("free_2pl_ridge_0p001", cm.FREE_2PL, 0.001, 1.0),
    ]
    assert [row["simplicity_rank"] for row in configured] == list(range(6))
    assert len({row["spec_id"] for row in configured}) == 6
    for row, (spec_id, family, ridge, shrinkage) in zip(configured, expected, strict=True):
        canonical = cm.calibration_specification(
            family,
            ridge=ridge,
            log_a_shrinkage=shrinkage,
        )
        assert row["spec_id"] == spec_id
        assert row["family"] == family
        assert row["canonical_cache_key"] == canonical["cache_key"]
        assert row["log_a_prior_sd"] == canonical["regularization"]["log_a_prior_sd"]


def test_v2_has_one_primary_policy_and_no_cat_grid_search() -> None:
    config = _load()
    policies = config["cat_policies"]
    assert policies["primary"] == {
        "policy_id": "primary_floor15_se0p20_trace",
        "role": "primary",
        "minimum_scenarios": 15,
        "conditional_se_target": 0.2,
        "selector": "trace",
    }
    assert {
        (row["minimum_scenarios"], row["conditional_se_target"], row["selector"])
        for row in policies["sensitivities"]
    } == {(12, 0.2, "trace"), (15, 0.2, "dopt")}
    assert policies["sensitivities_can_replace_primary"] is False
    assert "cat_candidates" not in config
    assert config["selection_gates"]["apply_to_primary_only"] is True
    assert config["selection_gates"]["allow_fallback_if_primary_fails"] is False


def test_v2_frozen_product_and_uncertainty_decisions_are_exact() -> None:
    config = _load()
    gates = config["selection_gates"]
    assert gates["minimum_scenario_reduction_vs_random"] == 0.5
    assert gates["require_paired_family_bootstrap_ci_favors_cat"] is True
    uncertainty = config["uncertainty"]
    assert uncertainty["absolute_p90_total_se_maximum"] == 0.5
    assert uncertainty["parameter_bootstrap_replicates"] == 100
    assert uncertainty["minimum_valid_parameter_bootstrap_rate"] == 0.9
    assert len(uncertainty["order_seeds"]) == 20
    assert len(set(uncertainty["order_seeds"])) == 20


def test_v2_interpretation_remains_conditional_and_internal() -> None:
    limitations = _load()["limitations"]
    assert limitations["infobench_human_judge_audit_performed"] is False
    assert limitations["additional_tutor_models_added"] is False
    assert limitations["same_cohort_reused_after_v1"] is True
    assert limitations["independent_confirmation"] is False
    assert limitations["unseen_family_generalization_claim_allowed"] is False
