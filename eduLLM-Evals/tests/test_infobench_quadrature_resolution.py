from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import nested_scenario_cat_cv as nested

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "run_infobench_quadrature_resolution.py"
SPEC = importlib.util.spec_from_file_location("infobench_quadrature_resolution_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
resolution = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = resolution
SPEC.loader.exec_module(resolution)


def _gates(first: bool, second: bool, endpoint: bool):
    return {
        "401_vs_801": {"passed": first},
        "801_vs_1601": {"passed": second},
        "401_vs_1601": {"passed": endpoint},
    }


def _cells(offset: float = 0.0) -> pd.DataFrame:
    rows = []
    for index, family in enumerate(("a", "a", "b", "b", "c", "c", "d", "d")):
        label = index % 2
        base = 0.75 if label else 0.25
        rows.append(
            {
                "outer_fold": index % 2,
                "model": f"model-{index}",
                "family": family,
                "criterion_id": f"criterion-{index}",
                "label": label,
                "probability": base + offset,
            }
        )
    return pd.DataFrame(rows)


def test_context_verifies_frozen_parent_and_inputs() -> None:
    context = resolution.load_context()
    assert context.raw["fit_grid"] == 61
    assert context.base.fit_grids == (41, 61, 81)
    assert context.base.ridge_candidates == (0.001, 0.01, 0.1)
    assert context.parent_run.is_dir()
    assert context.dense_parent_run.is_dir()


def test_primary_lock_requires_endpoint_for_401_and_allows_801_fallback() -> None:
    assert resolution.select_primary_nodes(_gates(True, True, True)) == 401
    assert resolution.select_primary_nodes(_gates(True, True, False)) is None
    assert resolution.select_primary_nodes(_gates(False, True, False)) == 801
    assert resolution.select_primary_nodes(_gates(False, False, True)) is None
    assert resolution.select_primary_nodes(_gates(True, False, True)) is None


def test_primary_theta_shift_uses_worst_of_full_and_administration_support() -> None:
    merged = pd.DataFrame(
        {
            "theta_full_lower": [0.0, 1.0],
            "theta_full_upper": [0.001, 1.001],
            "theta_heldout_lower": [0.0, 1.0],
            "theta_heldout_upper": [0.2, 1.1],
        }
    )
    result = resolution._theta_shift_diagnostics(merged)
    assert result["theta_shift_by_support"]["full"][
        "maximum_absolute_theta_shift"
    ] == pytest.approx(0.001)
    assert result["theta_shift_by_support"]["administration_only"][
        "maximum_absolute_theta_shift"
    ] == pytest.approx(0.2)
    assert result["maximum_absolute_theta_shift"] == pytest.approx(0.2)


def test_node_pileup_gate_checks_candidate_grid_itself() -> None:
    assert resolution._node_pileup_pass(
        {"unique_theta_rounded_6": 6, "largest_rounded_value_fraction": 0.49},
        5,
    )
    assert not resolution._node_pileup_pass(
        {"unique_theta_rounded_6": 5, "largest_rounded_value_fraction": 0.49},
        5,
    )
    assert not resolution._node_pileup_pass(
        {"unique_theta_rounded_6": 6, "largest_rounded_value_fraction": 0.5},
        5,
    )


def test_method_specific_score_paths_do_not_collide(tmp_path: Path) -> None:
    trapezoid = resolution.score_dir(
        tmp_path,
        method="normal_trapezoid",
        nodes=1601,
        bound=8.0,
        fit_grid=61,
        ridge=0.1,
    )
    bound_ten = resolution.score_dir(
        tmp_path,
        method="normal_trapezoid",
        nodes=2001,
        bound=10.0,
        fit_grid=61,
        ridge=0.1,
    )
    scipy_gh = resolution.score_dir(
        tmp_path,
        method="gauss_hermite_scipy",
        nodes=1601,
        bound=None,
        fit_grid=61,
        ridge=0.1,
    )
    assert len({trapezoid, bound_ten, scipy_gh}) == 3
    assert "method_normal_trapezoid" in str(trapezoid)
    assert "bound_10p0" in str(bound_ten)
    assert "method_gauss_hermite_scipy" in str(scipy_gh)


def test_array_hash_is_float64_c_order_raw_bytes() -> None:
    value = np.asarray([[1, 2], [3, 4]], dtype=np.int16).T
    canonical = np.ascontiguousarray(value, dtype=np.float64)
    expected = hashlib.sha256(canonical.tobytes(order="C")).hexdigest()
    assert resolution.array_sha256(value) == expected


def test_common_cell_equivalence_uses_labels_and_family_clusters() -> None:
    result, common = resolution.common_cell_equivalence(
        _cells(),
        _cells(offset=0.0001),
        margin=0.005,
        replicates=300,
        seed=7,
    )
    assert result["passed"] is True
    assert result["common_cell_labels_identical"] is True
    assert result["cluster_unit"] == "model_family"
    assert len(common) == 8
    assert result["metrics"]["log_loss"]["ci_wholly_within_equivalence_bounds"] is True


def test_common_cell_equivalence_rejects_label_mismatch() -> None:
    upper = _cells()
    upper.loc[0, "label"] = 1
    with pytest.raises(resolution.QuadratureResolutionError, match="labels differ"):
        resolution.common_cell_equivalence(_cells(), upper, margin=0.005, replicates=20, seed=3)


def test_portable_artifact_path_recovers_after_repo_move(tmp_path: Path) -> None:
    parent = tmp_path / "runs" / "calibration" / "parent-study"
    recorded = Path("/old/machine/repo/runs/calibration/parent-study/fit_cache/grid/fit.npz")
    assert resolution.portable_artifact_relative(recorded, parent) == Path("fit_cache/grid/fit.npz")


def test_grid_lock_contains_native_and_quadrature_identity_fields() -> None:
    common = {"passed": True, "locked_fit_grid": 61}
    primary = {"passed": True, "locked_eap_grid": 401}
    payload = resolution.build_grid_lock_payload(
        locked_nodes=401,
        score_manifest={
            "requested_node_count": 401,
            "effective_node_count": 401,
            "quadrature_method": "normal_trapezoid",
            "linear_bound": 8.0,
            "quadrature_axis_sha256": "a" * 64,
            "quadrature_log_prior_sha256": "b" * 64,
        },
        common_gate=common,
        primary_gate=primary,
        bound_gate={"passed": True},
        cross_family_gate={"passed": True},
    )
    assert payload["fit_grid"] == 61
    assert payload["eap_grid"] == 401
    assert payload["quadrature_method"] == "normal_trapezoid"
    assert payload["linear_bound"] == 8.0
    assert payload["effective_node_count"] == 401
    assert payload["quadrature_axis_sha256"] == "a" * 64
    assert payload["quadrature_log_prior_sha256"] == "b" * 64
    assert payload["ridge_scheduled_only_after_this_lock"] is True


def test_native_lock_is_consumable_by_nested_runner(tmp_path: Path) -> None:
    quadrature = resolution._quadrature(
        resolution.load_context(), "normal_trapezoid", 401, 8.0
    )
    payload = resolution.build_grid_lock_payload(
        locked_nodes=401,
        score_manifest={
            "requested_node_count": 401,
            "effective_node_count": len(quadrature.grid),
            "quadrature_method": "normal_trapezoid",
            "linear_bound": 8.0,
            "quadrature_axis_sha256": resolution.array_sha256(quadrature.grid[:, 0]),
            "quadrature_log_prior_sha256": resolution.array_sha256(
                quadrature.log_prior
            ),
        },
        common_gate={
            "passed": True,
            "require_every_comparison": True,
            "locked_fit_grid": 61,
            "comparisons": {
                "61_vs_81": {
                    "passed": True,
                    "common_cell_keys_identical_after_explicit_intersection": True,
                    "common_cell_labels_identical": True,
                }
            },
        },
        primary_gate={
            "passed": True,
            "locked_eap_grid": 401,
            "comparisons": {"401_vs_801": {"passed": True}},
        },
        bound_gate={
            "passed": True,
            "model_keys_identical": True,
            "posterior_tail_mass_passed": True,
        },
        cross_family_gate={"passed": True, "model_keys_identical": True},
    )
    dense_dir = tmp_path / "study" / "dense_grid"
    dense_dir.mkdir(parents=True)
    lock_path = dense_dir / "grid_lock.json"
    resolution._write_json(lock_path, payload)
    resolution._write_json(
        tmp_path / "study" / "study_manifest.json",
        {
            "status": "phase2_complete",
            "result": {"fit_grid": 61, "eap_grid": 401, "grid_lock": payload},
        },
    )
    args = nested.build_argparser().parse_args(
        ["--dense-grid-manifest", str(lock_path)]
    )
    settings = nested.resolve_dense_settings(args)
    assert settings.fit_grid == 61
    assert settings.eap_grid == 401
    assert settings.quadrature_method == "normal_trapezoid"
    assert settings.effective_node_count == 401


def test_grid_lock_refuses_any_failed_numerical_gate() -> None:
    with pytest.raises(resolution.QuadratureResolutionError, match="did not pass"):
        resolution.build_grid_lock_payload(
            locked_nodes=401,
            score_manifest={
                "requested_node_count": 401,
                "effective_node_count": 401,
                "quadrature_method": "normal_trapezoid",
                "linear_bound": 8.0,
                "quadrature_axis_sha256": "a" * 64,
                "quadrature_log_prior_sha256": "b" * 64,
            },
            common_gate={"passed": False},
            primary_gate={"passed": True},
            bound_gate={"passed": True},
            cross_family_gate={"passed": True},
        )


def test_parent_paths_are_protected_even_for_worker_invocation() -> None:
    context = resolution.load_context()
    with pytest.raises(resolution.QuadratureResolutionError, match="historical parent"):
        resolution._assert_safe_output(context, context.parent_run / "worker-output")


def test_grid61_ridge_point1_refit_is_rejected_before_worker_launch(tmp_path: Path) -> None:
    with pytest.raises(resolution.QuadratureResolutionError, match="never ridge 0.1"):
        resolution.main(
            [
                "--out-dir",
                str(tmp_path / "new-study"),
                "--worker",
                "fit-ridge",
                "--fit-grid",
                "61",
                "--ridge",
                "0.1",
            ]
        )


def test_post_lock_ridge_worker_is_rejected_before_verified_grid_lock(
    tmp_path: Path,
) -> None:
    context = resolution.load_context()
    out_dir = tmp_path / "quadrature-study"
    resolution.ResolutionOrchestrator(
        context, out_dir, resume=False, plan_only=False
    ).prepare()
    with pytest.raises(
        resolution.QuadratureResolutionError,
        match="requires a verified passed grid_lock",
    ):
        resolution.main(
            [
                "--out-dir",
                str(out_dir),
                "--worker",
                "fit-ridge",
                "--fit-grid",
                "61",
                "--ridge",
                "0.01",
            ]
        )


def test_post_lock_worker_verifies_lock_stage_hash(tmp_path: Path) -> None:
    context = resolution.load_context()
    out_dir = tmp_path / "quadrature-study"
    lock_path = out_dir / "dense_grid" / "grid_lock.json"
    payload = resolution.build_grid_lock_payload(
        locked_nodes=401,
        score_manifest={
            "requested_node_count": 401,
            "effective_node_count": 401,
            "quadrature_method": "normal_trapezoid",
            "linear_bound": 8.0,
            "quadrature_axis_sha256": "a" * 64,
            "quadrature_log_prior_sha256": "b" * 64,
        },
        common_gate={"passed": True, "locked_fit_grid": 61},
        primary_gate={"passed": True, "locked_eap_grid": 401},
        bound_gate={"passed": True},
        cross_family_gate={"passed": True},
    )
    resolution._write_json(lock_path, payload)
    relative = "dense_grid/grid_lock.json"
    resolution._write_json(
        out_dir / "stage_markers" / "emit_native_grid_lock.json",
        {
            "stage": "emit_native_grid_lock",
            "kind": "deterministic_analysis",
            "command": ["internal-analysis", "emit_native_grid_lock"],
            "outputs": [relative],
            "output_sha256": {relative: resolution._sha256(lock_path)},
        },
    )
    assert resolution._verify_passed_grid_lock(context, out_dir) == payload


def test_prepare_records_all_code_and_input_hashes_and_terminal_resume_fails(
    tmp_path: Path,
) -> None:
    context = resolution.load_context()
    out_dir = tmp_path / "quadrature-study"
    orchestrator = resolution.ResolutionOrchestrator(
        context, out_dir, resume=False, plan_only=False
    )
    orchestrator.prepare()
    manifest_path = out_dir / "study_manifest.json"
    manifest = resolution._read_json(manifest_path)
    for key in (
        "runner_sha256",
        "scenario_cat_lib_sha256",
        "nested_runner_sha256",
        "config_sha256",
        "split_manifest_sha256",
        "parent_fit_imports_sha256",
    ):
        assert len(manifest[key]) == 64
    assert set(manifest["inputs"]) == {"matrix", "rubrics", "scenarios"}
    assert manifest["fit_grid_61_ridge_0p1_refit"] is False
    assert all(
        not Path(record["source_path_relative_to_run"]).is_absolute()
        and not Path(record["destination_path_relative_to_study"]).is_absolute()
        for record in resolution._read_json(out_dir / "parent_fit_imports.json")["artifacts"]
    )
    manifest["status"] = "quadrature_resolution_complete"
    resolution._write_json(manifest_path, manifest)
    resumed = resolution.ResolutionOrchestrator(context, out_dir, resume=True, plan_only=False)
    with pytest.raises(resolution.QuadratureResolutionError, match="terminal study"):
        resumed.prepare()
