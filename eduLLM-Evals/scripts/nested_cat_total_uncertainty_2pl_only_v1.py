"""Versioned Phase-4 adapter for the InFoBench 2PL-only-v1 study.

The numerical calculations are the already-tested V2/V3 Phase-4 engine.  This
module changes only the study identity, Phase-3 producer, canonical output
leaves, and provenance inventory.  It never accepts or writes the completed
V3 leaves.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import calibrate_mirt as cm  # noqa: E402
from scripts import nested_cat_total_uncertainty_v2 as engine  # noqa: E402
from scripts import nested_cat_total_uncertainty_v3 as v3  # noqa: E402
from scripts import nested_scenario_cat_cv as phase3_v1  # noqa: E402
from scripts import nested_scenario_cat_cv_2pl_only_v1 as phase3  # noqa: E402
from scripts import scenario_cat_lib as scat  # noqa: E402

SCRIPT_SCHEMA = "infobench-nested-cat-total-uncertainty-2pl-only-v1"
CHECKPOINT_SCHEMA = "infobench-2pl-only-v1-phase4-checkpoint-v1"
BOOTSTRAP_CACHE_SCHEMA = "infobench-2pl-only-v1-phase4-bootstrap-fit-v1"
DECISION_SCHEMA = "infobench-2pl-only-v1-phase4-decision-v1"

DEFAULT_CONFIG = ROOT / "configs" / "infobench_calibration_cat_2pl_only_v1.json"
DEFAULT_PHASE3 = ROOT / "runs" / "calibration" / "InFoBench_2pl_only_v1" / "phase3"
DEFAULT_OUTPUT = ROOT / "runs" / "calibration" / "InFoBench_2pl_only_v1" / "phase4"
DEFAULT_NUMERICAL_FOLLOWUP_CONFIG = phase3.DEFAULT_NUMERICAL_FOLLOWUP_CONFIG
DEFAULT_NUMERICAL_LOCK = phase3.DEFAULT_NUMERICAL_LOCK

EXPECTED_REPEATS = phase3.EXPECTED_REPEATS
EXPECTED_OUTER_FOLDS = phase3.EXPECTED_OUTER_FOLDS
EXPECTED_MODELS = phase3.EXPECTED_MODELS
EXPECTED_PANELS = EXPECTED_REPEATS * EXPECTED_OUTER_FOLDS
PRIMARY_POLICY_ID = phase3.PRIMARY_POLICY_ID
MAXIMUM_INNER_GRADIENT = v3.MAXIMUM_INNER_GRADIENT
INFOBENCH_SKILLS = v3.INFOBENCH_SKILLS
FINAL_OUTPUTS = engine.FINAL_OUTPUTS

CODE_DEPENDENCIES = (
    ROOT / "scripts" / "nested_cat_total_uncertainty_2pl_only_v1.py",
    ROOT / "scripts" / "nested_cat_total_uncertainty_v3.py",
    ROOT / "scripts" / "nested_cat_total_uncertainty_v2.py",
    ROOT / "scripts" / "nested_scenario_cat_cv_2pl_only_v1.py",
    ROOT / "scripts" / "nested_scenario_cat_cv.py",
    ROOT / "scripts" / "calibrate_partial.py",
    ROOT / "scripts" / "calibrate_mirt.py",
    ROOT / "scripts" / "kfold_cv_mirt.py",
    ROOT / "scripts" / "scenario_cat_lib.py",
    ROOT / "scripts" / "scenario_kfold_estimator_cv.py",
    ROOT / "scripts" / "run_infobench_v2_numerical_followup_v4.py",
    ROOT / "tutor_cat" / "mirt.py",
    ROOT / "tutor_cat" / "engine.py",
    ROOT / "tutor_cat" / "selector.py",
    ROOT / "tutor_cat" / "dataio.py",
    ROOT / "tutor_cat" / "schemas.py",
    ROOT / "tutor_cat" / "skill_structure.py",
    ROOT / "tutor_cat" / "__init__.py",
)


class TwoPLPhase4Error(v3.V3Phase4Error):
    """A 2PL-only-v1 authorization, provenance, or completeness check failed."""


def validate_two_pl_inner_contract(
    decision: Mapping[str, Any], selected: Mapping[str, Any]
) -> None:
    """Require 25 complete two-spec panels and exactly 8 inner blocks each."""

    inner = decision.get("inner_selection_fail_closed") or {}
    expected_per_panel = phase3.EXPECTED_SPEC_FOLD_BLOCKS_PER_PANEL
    expected_specs = list(phase3.EXPECTED_PHASE3_SPEC_IDS)
    if (
        type(inner.get("expected_spec_fold_combinations_per_panel")) is not int
        or inner.get("expected_spec_fold_combinations_per_panel") != expected_per_panel
    ):
        raise TwoPLPhase4Error("Phase-3 decision does not freeze 8 spec-fold blocks per panel")
    panels = selected.get("panels")
    if not isinstance(panels, list) or len(panels) != phase3.EXPECTED_PANELS:
        raise TwoPLPhase4Error("Phase-3 selected-spec handoff does not contain 25 panels")
    observed_panels: set[tuple[int, int]] = set()
    valid_total = 0
    for raw in panels:
        if not isinstance(raw, Mapping):
            raise TwoPLPhase4Error("Phase-3 selected-spec handoff contains a malformed panel")
        repeat = raw.get("repeat")
        outer_fold = raw.get("outer_fold")
        if type(repeat) is not int or type(outer_fold) is not int:
            raise TwoPLPhase4Error("Phase-3 panel coordinates are not exact integers")
        panel_key = (repeat, outer_fold)
        if (
            panel_key in observed_panels
            or repeat not in range(phase3.EXPECTED_REPEATS)
            or outer_fold not in range(phase3.EXPECTED_OUTER_FOLDS)
        ):
            raise TwoPLPhase4Error("Phase-3 panel coordinates are not the exact 5x5 grid")
        observed_panels.add(panel_key)
        if str(raw.get("selected_spec_id") or "") not in expected_specs:
            raise TwoPLPhase4Error("Phase-3 panel selected a non-2PL specification")
        inner_selection = raw.get("inner_selection")
        if not isinstance(inner_selection, Mapping):
            raise TwoPLPhase4Error("Phase-3 panel lacks its inner-selection audit")
        audit = inner_selection.get("complete_inner_evidence_audit")
        if not isinstance(audit, Mapping):
            raise TwoPLPhase4Error("Phase-3 panel lacks its complete inner-evidence audit")
        combinations = audit.get("combination_audits")
        if (
            audit.get("passed") is not True
            or audit.get("require_all_specs_every_inner_fold") is not True
            or audit.get("survivor_selection_allowed") is not False
            or audit.get("expected_inner_folds") != list(range(phase3.EXPECTED_INNER_FOLDS))
            or audit.get("expected_spec_ids") != expected_specs
            or audit.get("expected_spec_fold_combinations") != expected_per_panel
            or audit.get("valid_spec_fold_combinations") != expected_per_panel
            or audit.get("failed_checks") != []
            or not isinstance(combinations, list)
            or len(combinations) != expected_per_panel
        ):
            raise TwoPLPhase4Error("Phase-3 panel lacks its complete 4-fold x 2-spec audit")
        if not all(isinstance(row, Mapping) for row in combinations):
            raise TwoPLPhase4Error("Phase-3 panel contains a malformed spec-fold audit")
        observed_blocks = {
            (row.get("inner_fold"), str(row.get("spec_id") or "")) for row in combinations
        }
        expected_blocks = {
            (inner_fold, spec_id)
            for inner_fold in range(phase3.EXPECTED_INNER_FOLDS)
            for spec_id in expected_specs
        }
        if observed_blocks != expected_blocks or not all(
            row.get("passed") is True for row in combinations
        ):
            raise TwoPLPhase4Error("Phase-3 panel spec-fold audit roster is incomplete")
        valid_total += int(audit["valid_spec_fold_combinations"])
    expected_panels = {
        (repeat, outer_fold)
        for repeat in range(phase3.EXPECTED_REPEATS)
        for outer_fold in range(phase3.EXPECTED_OUTER_FOLDS)
    }
    if observed_panels != expected_panels:
        raise TwoPLPhase4Error("Phase-3 selected-spec handoff is not the exact 5x5 grid")
    if valid_total != phase3.EXPECTED_TOTAL_SPEC_FOLD_BLOCKS:
        raise TwoPLPhase4Error("Phase-3 inner evidence does not total exactly 200 blocks")


_V3_BINDING_NAMES = {
    "phase3": phase3,
    "SCRIPT_SCHEMA": SCRIPT_SCHEMA,
    "CHECKPOINT_SCHEMA": CHECKPOINT_SCHEMA,
    "BOOTSTRAP_CACHE_SCHEMA": BOOTSTRAP_CACHE_SCHEMA,
    "DECISION_SCHEMA": DECISION_SCHEMA,
    "DEFAULT_CONFIG": DEFAULT_CONFIG,
    "DEFAULT_PHASE3": DEFAULT_PHASE3,
    "DEFAULT_OUTPUT": DEFAULT_OUTPUT,
    "DEFAULT_NUMERICAL_FOLLOWUP_CONFIG": DEFAULT_NUMERICAL_FOLLOWUP_CONFIG,
    "DEFAULT_NUMERICAL_LOCK": DEFAULT_NUMERICAL_LOCK,
    "EXPECTED_REPEATS": EXPECTED_REPEATS,
    "EXPECTED_OUTER_FOLDS": EXPECTED_OUTER_FOLDS,
    "EXPECTED_MODELS": EXPECTED_MODELS,
    "EXPECTED_PANELS": EXPECTED_PANELS,
    "PRIMARY_POLICY_ID": PRIMARY_POLICY_ID,
    "CODE_DEPENDENCIES": CODE_DEPENDENCIES,
    "V3Phase4Error": TwoPLPhase4Error,
}


@contextmanager
def _two_pl_bindings() -> Iterator[None]:
    """Temporarily bind the immutable V3 adapter to this versioned study."""

    previous = {name: getattr(v3, name) for name in _V3_BINDING_NAMES}
    for name, value in _V3_BINDING_NAMES.items():
        setattr(v3, name, value)
    try:
        yield
    finally:
        for name, value in previous.items():
            setattr(v3, name, value)


def validate_phase3_authorization(
    decision: Mapping[str, Any],
    selected: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    with _two_pl_bindings():
        v3.validate_phase3_authorization(decision, selected, manifest)
    validate_two_pl_inner_contract(decision, selected)


def require_phase3_numerical_profile(
    manifest: Mapping[str, Any], profile: Mapping[str, Any]
) -> None:
    with _two_pl_bindings():
        v3.require_phase3_numerical_profile(manifest, profile)


def validate_phase3_artifact_inventory(
    phase3_dir: Path, manifest: Mapping[str, Any]
) -> dict[str, str]:
    with _two_pl_bindings():
        return v3.validate_phase3_artifact_inventory(phase3_dir, manifest)


def validate_phase3_cross_links(**kwargs: Any) -> None:
    with _two_pl_bindings():
        v3.validate_phase3_cross_links(**kwargs)


def validate_frozen_phase4_design(config: Mapping[str, Any]) -> None:
    with _two_pl_bindings():
        v3.validate_frozen_phase4_design(config)


def dense_fit_validity(fit: Mapping[str, Any], spec: phase3.CalibrationSpec) -> dict[str, Any]:
    with _two_pl_bindings():
        return v3.dense_fit_validity(fit, spec)


class TwoPLPhase4Runner(v3.V3Phase4Runner):
    """Run the shared Phase-4 engine under the 2PL-only-v1 bindings."""

    def __init__(self, args: argparse.Namespace):
        with _two_pl_bindings():
            super().__init__(args)
        validate_two_pl_inner_contract(self.phase3_decision, self.selected)

    def run(self) -> int:
        with _two_pl_bindings():
            return super().run()

    def plan_payload(self) -> dict[str, Any]:
        with _two_pl_bindings():
            return super().plan_payload()

    def _base_manifest(self, status: str) -> dict[str, Any]:
        with _two_pl_bindings():
            payload = super()._base_manifest(status)
        payload.update(
            {
                "script": "scripts/nested_cat_total_uncertainty_2pl_only_v1.py",
                "versioned_adapter": "scripts/nested_cat_total_uncertainty_v3.py",
                "phase3_candidate_spec_ids": list(phase3.EXPECTED_PHASE3_SPEC_IDS),
                "v4_numerical_spec_ids": list(phase3.EXPECTED_V4_SPEC_IDS),
                "parent_v3_artifacts_read_or_written": False,
            }
        )
        return payload


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phase3-dir", type=Path, default=DEFAULT_PHASE3)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--numerical-followup-config",
        type=Path,
        default=DEFAULT_NUMERICAL_FOLLOWUP_CONFIG,
    )
    parser.add_argument("--numerical-lock", type=Path, default=DEFAULT_NUMERICAL_LOCK)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    try:
        with _two_pl_bindings():
            return TwoPLPhase4Runner(args).run()
    except (
        TwoPLPhase4Error,
        v3.V3Phase4Error,
        engine.V2Phase4Error,
        phase3.V3Phase3Error,
        phase3_v1.NestedCVError,
        scat.OfflineStudyError,
        cm.CalibrationError,
        FileNotFoundError,
        OSError,
        ValueError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
