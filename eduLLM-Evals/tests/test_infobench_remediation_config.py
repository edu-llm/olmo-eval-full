from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "infobench_calibration_remediation_v1.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_preregistered_inputs_match_the_frozen_baseline() -> None:
    config = _load()
    baseline = config["baseline"]
    for path_key, hash_key in (
        ("response_matrix", "response_matrix_sha256"),
        ("judge_manifest", "judge_manifest_sha256"),
    ):
        path = ROOT / baseline[path_key]
        assert path.is_file()
        assert _sha256(path) == baseline[hash_key]

    assert baseline["models"] == 52
    assert baseline["source_scenarios"] == 500
    assert baseline["source_criteria"] == 2250
    assert baseline["fitted_criteria"] == 2105
    assert baseline["exported_criteria"] == 2096
    assert (
        baseline["source_criteria"]
        - baseline["unfitted_all_fail_criteria"]
        - baseline["excluded_nonpositive_discrimination"]
        == baseline["exported_criteria"]
    )

    phase0 = config["phase0_artifact"]
    assert _sha256(ROOT / phase0["local_archive"]) == phase0["archive_sha256"]
    provenance = json.loads((ROOT / phase0["provenance_manifest"]).read_text(encoding="utf-8"))
    assert provenance["archive"]["sha256"] == phase0["archive_sha256"]
    assert phase0["storage_status"] == "local_only_not_published"
    assert phase0["stable_storage_uri"] is None


def test_dense_grid_and_nested_cv_decisions_are_frozen() -> None:
    config = _load()
    dense = config["dense_grid"]
    assert dense["structure_screen_fit_grid"] == 5
    assert dense["fit_grid_candidates"] == [5, 7, 15, 25, 41]
    assert dense["eap_grid_candidates"] == [21, 41, 81]
    heldout = dense["fit_grid_stability"]["heldout_metric_stability"]
    assert heldout == {
        "rule": "absolute_pooled_shift_or_paired_fold_se",
        "metrics": ["log_loss", "brier"],
        "maximum_absolute_pooled_shift": 0.005,
        "paired_fold_se_multiplier": 2.0,
        "require_every_metric": True,
    }

    cv = config["cross_validation"]
    for path_key, hash_key in (
        ("fold_manifest", "fold_manifest_sha256"),
        ("model_assignments", "model_assignments_sha256"),
        ("scenario_assignments", "scenario_assignments_sha256"),
    ):
        assert _sha256(ROOT / cv[path_key]) == cv[hash_key]
    assert cv["outer_folds"] == 5
    assert cv["inner_folds"] == 4
    assert cv["group_related_model_families"] is True
    assert cv["administration_scenario_fraction"] == 0.8
    assert cv["evaluation_scenario_fraction"] == 0.2


def test_cat_grid_is_the_full_48_way_factorial_with_no_fallback() -> None:
    config = _load()
    candidates = config["cat_candidates"]
    product = list(
        itertools.product(
            candidates["minimum_scenarios"],
            candidates["conditional_se_targets"],
            candidates["selectors"],
        )
    )
    assert len(product) == 48
    assert len(set(product)) == 48
    assert candidates["run_full_factorial"] is True
    assert config["selection_gates"]["allow_fallback_if_none_pass"] is False
    assert len(config["uncertainty"]["order_seeds"]) >= 20


def test_release_gates_prevent_premature_final_claims() -> None:
    gates = _load()["release_gates"]
    assert gates["judge_audit_must_pass"] is True
    assert gates["dense_grid_must_be_stable"] is True
    assert gates["nested_oos_gates_must_pass"] is True
    assert gates["total_uncertainty_gate_must_pass"] is True
    assert gates["order_stability_gate_must_pass"] is True
    assert gates["external_frozen_bank_validation"] == "pending_new_model_cohort"
