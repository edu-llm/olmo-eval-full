from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "run_infobench_eap_grid_followup.py"
SPEC = importlib.util.spec_from_file_location("infobench_eap_followup_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
followup = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = followup
SPEC.loader.exec_module(followup)
PARENT_MANIFEST = (
    ROOT
    / "runs"
    / "calibration"
    / "InFoBench_remediation_v1_dense_followup"
    / "study_manifest.json"
)


def _gates(first: bool, second: bool):
    return {
        "81_vs_161": {"passed": first},
        "161_vs_321": {"passed": second},
    }


@pytest.mark.skipif(
    not PARENT_MANIFEST.is_file(),
    reason="integration check requires the separately frozen dense-grid follow-up",
)
def test_eap_followup_preserves_parent_fit_and_thresholds() -> None:
    context, raw, parent, fit_gate = followup.load_context()
    assert context.fit_grids == (61,)
    assert context.eap_grids == (81, 161, 321)
    assert context.ridge_candidates == (0.001, 0.01, 0.1)
    assert parent.is_dir()
    assert fit_gate["passed"] is True
    assert fit_gate["locked_fit_grid"] == 61
    assert raw["threshold_change_policy"].startswith("No threshold may be relaxed")


def test_eap_lock_requires_the_frozen_successive_pattern() -> None:
    assert followup.select_locked_eap_grid(_gates(True, True)) == 81
    assert followup.select_locked_eap_grid(_gates(False, True)) == 161
    assert followup.select_locked_eap_grid(_gates(True, False)) is None
    assert followup.select_locked_eap_grid(_gates(False, False)) is None
