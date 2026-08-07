from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "run_infobench_dense_grid_followup.py"
SPEC = importlib.util.spec_from_file_location("infobench_dense_grid_followup_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
followup = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = followup
SPEC.loader.exec_module(followup)
PARENT_MANIFEST = (
    ROOT / "runs" / "calibration" / "InFoBench_remediation_v1" / "study_manifest.json"
)


def _gates(first: bool, second: bool):
    return {
        "41_vs_61": {"passed": first},
        "61_vs_81": {"passed": second},
    }


@pytest.mark.skipif(
    not PARENT_MANIFEST.is_file(),
    reason="integration check requires the separately frozen Phase-2 run artifact",
)
def test_followup_context_preserves_frozen_thresholds() -> None:
    context, raw, parent = followup.load_context()
    assert context.fit_grids == (41, 61, 81)
    assert context.eap_grids == (21, 41, 81)
    assert context.ridge_candidates == (0.001, 0.01, 0.1)
    assert parent.is_dir()
    assert raw["threshold_change_policy"].startswith("No threshold may be relaxed")


def test_fit_grid_lock_requires_the_frozen_successive_pattern() -> None:
    assert followup.select_locked_fit_grid(_gates(True, True)) == 41
    assert followup.select_locked_fit_grid(_gates(False, True)) == 61
    assert followup.select_locked_fit_grid(_gates(True, False)) is None
    assert followup.select_locked_fit_grid(_gates(False, False)) is None
