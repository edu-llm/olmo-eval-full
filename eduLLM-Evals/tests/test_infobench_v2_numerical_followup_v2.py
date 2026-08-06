"""Regression tests for the corrected append-only numerical follow-up v2."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_infobench_v2_numerical_followup_v2.py"
CONFIG = ROOT / "configs" / "infobench_v2_numerical_followup_v2.json"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "infobench_v2_numerical_followup_v2", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


followup = _load_module()


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    lower = pd.DataFrame(
        {
            "outer_fold": [0, 1],
            "model": ["m1", "m2"],
            "family": ["family_a", "family_b"],
            "criterion_id": ["c1", "c2"],
            "label": [1, 0],
            "probability": [0.8, 0.2],
        }
    )
    upper = lower.copy()
    upper["probability"] = [0.801, 0.199]
    return lower, upper


@pytest.mark.parametrize("lower_grid,upper_grid", [(81, 101), (101, 121)])
def test_dynamic_common_cell_labels_match_actual_grids(
    lower_grid: int, upper_grid: int
) -> None:
    lower, upper = _frames()
    gate, cells = followup.dynamic_common_cell_equivalence(
        lower,
        upper,
        lower_grid=lower_grid,
        upper_grid=upper_grid,
        log_loss_margin=0.005,
        brier_margin=0.005,
        replicates=20,
        seed=17,
    )
    expected_probability = [
        f"probability_grid_{lower_grid}",
        f"probability_grid_{upper_grid}",
    ]
    expected_delta = [
        f"log_loss_delta_grid_{upper_grid}_minus_grid_{lower_grid}",
        f"brier_delta_grid_{upper_grid}_minus_grid_{lower_grid}",
    ]
    assert gate["comparison"] == f"grid_{lower_grid}_to_grid_{upper_grid}"
    assert gate["lower_grid"] == lower_grid
    assert gate["upper_grid"] == upper_grid
    assert gate["csv_probability_columns"] == expected_probability
    assert gate["csv_delta_columns"] == expected_delta
    assert all(column in cells.columns for column in expected_probability + expected_delta)
    for metric in ("log_loss", "brier"):
        directional = f"pooled_shift_grid_{upper_grid}_minus_grid_{lower_grid}"
        assert directional in gate["metrics"][metric]
    serialized = json.dumps({"gate": gate, "columns": list(cells.columns)})
    assert "grid61" not in serialized
    assert "_61" not in serialized


def test_overlay_cites_and_hash_verifies_the_preserved_v1_abort() -> None:
    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    followup._validate_overlay(raw)
    evidence = followup._verify_superseded_attempt(raw)
    assert evidence["aborted_marker_sha256"] == raw["superseded_attempt"][
        "aborted_marker_sha256"
    ]
    assert evidence["output_file_count"] == 295
    assert raw["superseded_attempt"]["reuse_policy"].startswith("No fit")


def test_effective_config_has_new_leaf_and_dynamic_cell_contract() -> None:
    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    effective = followup._materialize_effective_config(raw)
    followup._validate_effective_config(effective)
    assert effective["schema_version"] == followup.CONFIG_SCHEMA
    assert effective["output_dir"].endswith("InFoBench_v2_numerical_followup_v2")
    fit = effective["fit_grid_followup"]
    assert "heldout_common_cell_equivalence" not in fit
    assert "heldout_dynamic_common_cell_equivalence" in fit
    assert fit["heldout_dynamic_common_cell_equivalence"]["bootstrap_seed"] == 20260805


@pytest.mark.parametrize(
    "mutation",
    [
        ("cells", "bootstrap_seed", 1),
        ("cells", "bootstrap_replicates", 1999),
        ("runtime", "fit_max_iter", 201),
        ("runtime", "fit_tolerance", 0.001),
        ("eap", "run_for_every_exact_specification", False),
        ("eap", "run_only_after_common_fit_grid_lock", False),
        ("fit", "always_run_both_fit_comparisons", False),
        ("fit", "second_comparison", "81_vs_121"),
        ("fit", "lock_rule", {"otherwise": 121}),
        ("eap", "lock_rule", {"otherwise": {"eap_grid": 1001}}),
    ],
)
def test_material_preregistration_fields_fail_closed(mutation: tuple[str, str, object]) -> None:
    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    effective = followup._materialize_effective_config(raw)
    section, key, value = mutation
    if section == "cells":
        effective["fit_grid_followup"][
            "heldout_dynamic_common_cell_equivalence"
        ][key] = value
    elif section == "runtime":
        effective["runtime"][key] = value
    elif section == "eap":
        effective["eap_bound_followup"][key] = value
    else:
        effective["fit_grid_followup"][key] = value
    with pytest.raises(followup.FollowupV2Error):
        followup._validate_effective_config(effective)


def test_frozen_contract_itself_fails_closed() -> None:
    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    changed = copy.deepcopy(raw)
    changed["frozen_contract"]["runtime"]["fit_max_iter"] = 201
    with pytest.raises(followup.FollowupV2Error, match="material preregistration"):
        followup._validate_overlay(changed)


def test_full_parent_exact_spec_contract_is_compared_not_ids_only() -> None:
    canonical = {"cache_key": "k", "family": "1pl", "ridge": 0.0}
    child = SimpleNamespace(
        spec_id="same",
        family="1pl",
        ridge=None,
        log_a_shrinkage=None,
        canonical=canonical,
    )
    parent = SimpleNamespace(**child.__dict__)
    assert followup._require_full_parent_spec_match([child], [parent])[0][
        "canonical_specification"
    ] == canonical
    changed_family = SimpleNamespace(**{**parent.__dict__, "family": "free-2pl"})
    with pytest.raises(followup.FollowupV2Error, match="exact specifications"):
        followup._require_full_parent_spec_match([child], [changed_family])
    changed_regularization = SimpleNamespace(**{**parent.__dict__, "ridge": 0.1})
    with pytest.raises(followup.FollowupV2Error, match="exact specifications"):
        followup._require_full_parent_spec_match([child], [changed_regularization])
    changed_canonical = SimpleNamespace(
        **{**parent.__dict__, "canonical": {**canonical, "cache_key": "other"}}
    )
    with pytest.raises(followup.FollowupV2Error, match="exact specifications"):
        followup._require_full_parent_spec_match([child], [changed_canonical])


def test_context_signature_freezes_full_contract_and_no_v1_cache_reuse() -> None:
    context = followup.load_context(CONFIG)
    signature = context.study_signature
    assert len(signature["exact_specifications"]) == 6
    assert all(
        {
            "spec_id",
            "family",
            "ridge",
            "log_a_shrinkage",
            "canonical_specification",
        }
        == set(record)
        for record in signature["exact_specifications"]
    )
    assert signature["comparison_schedule"] == [[81, 101], [101, 121]]
    assert signature["aborted_v1_artifacts_reused"] is False
    schedule = followup.runtime_schedule(context)
    assert schedule["required_new_grid101_fits"] == 36
    assert schedule["required_new_grid121_fits"] == 36
    assert schedule["aborted_v1_fit_artifacts_reused"] == 0


def test_cli_has_resume_but_no_destructive_fresh_option() -> None:
    parser = followup.build_parser()
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert "--resume" in option_strings
    assert "--fresh" not in option_strings
    assert "rmtree" not in SCRIPT.read_text(encoding="utf-8")


def test_resume_rejects_terminal_or_mismatched_checkpoint_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = followup.load_context(CONFIG)
    temporary_root = tmp_path / "repo"
    output = temporary_root / "runs" / "calibration" / followup.OUTPUT_LEAF
    isolated = replace(context, out_dir=output)
    monkeypatch.setattr(followup, "ROOT", temporary_root)
    followup._prepare(isolated, resume=False)
    followup._prepare(isolated, resume=True)

    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir()
    bad = checkpoint_dir / "unknown.json"
    bad.write_text("{}\n", encoding="utf-8")
    with pytest.raises(followup.FollowupV2Error, match="unexpected checkpoint"):
        followup._prepare(isolated, resume=True)
    bad.unlink()

    manifest_path = output / "study_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "blocked_numerical_followup"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(followup.FollowupV2Error, match="terminal evidence"):
        followup._prepare(isolated, resume=True)


def test_write_once_evidence_never_replaces_different_content(tmp_path: Path) -> None:
    path = tmp_path / "decision.json"
    followup._write_json_once(path, {"passed": False})
    followup._write_json_once(path, {"passed": False})
    with pytest.raises(followup.FollowupV2Error, match="append-only JSON"):
        followup._write_json_once(path, {"passed": True})
