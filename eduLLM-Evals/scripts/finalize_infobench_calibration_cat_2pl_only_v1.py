"""Authorized all-52 final fit/replay for InFoBench 2PL-only-v1.

This versioned adapter reuses the audited V3 final-fit implementation while
pinning it to the 2PL-only Phase-3/Phase-4 schemas and output namespace.  It is
fail-closed and cannot read or write the completed V3 study leaves.
"""

from __future__ import annotations

import argparse
import copy
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from scripts import calibrate_mirt as cm  # noqa: E402
from scripts import finalize_infobench_calibration_cat_v3 as v3  # noqa: E402
from scripts import nested_cat_total_uncertainty_2pl_only_v1 as phase4  # noqa: E402
from scripts import nested_cat_total_uncertainty_v2 as phase4_engine  # noqa: E402
from scripts import nested_cat_total_uncertainty_v3 as phase4_v3  # noqa: E402
from scripts import nested_scenario_cat_cv as phase3_v1  # noqa: E402
from scripts import nested_scenario_cat_cv_2pl_only_v1 as phase3  # noqa: E402
from scripts import scenario_cat_lib as scat  # noqa: E402

SCRIPT_SCHEMA = "infobench-final-fit-replay-2pl-only-v1"
FIT_MANIFEST_SCHEMA = "infobench-final-all52-fit-2pl-only-v1"
FIT_TRANSACTION_SCHEMA = "infobench-final-all52-fit-transaction-2pl-only-v1"
CHECKPOINT_SCHEMA = "infobench-final-replay-checkpoint-2pl-only-v1"
EXPORT_MANIFEST_SCHEMA = "infobench-final-bank-export-2pl-only-v1"
DECISION_SCHEMA = "infobench-final-fit-decision-2pl-only-v1"

EXPECTED_MODELS = phase3.EXPECTED_MODELS
EXPECTED_FAMILIES = phase3.EXPECTED_FAMILIES
EXPECTED_PANELS = phase3.EXPECTED_PANELS
EXPECTED_REPEATS = phase3.EXPECTED_REPEATS
PRIMARY_POLICY_ID = phase3.PRIMARY_POLICY_ID
FINAL_BANK_SOURCE = "calibrated-infobench-2pl-only-v1-final-all52"
CANONICAL_PHASE3_DIR = ROOT / "runs" / "calibration" / "InFoBench_2pl_only_v1" / "phase3"
CANONICAL_PHASE4_DIR = ROOT / "runs" / "calibration" / "InFoBench_2pl_only_v1" / "phase4"
CANONICAL_FINAL_OUTPUT = ROOT / "runs" / "calibration" / "InFoBench_2pl_only_v1" / "final_fit"
DEFAULT_CONFIG = phase3.DEFAULT_CONFIG
DEFAULT_PHASE3 = CANONICAL_PHASE3_DIR
DEFAULT_PHASE4 = CANONICAL_PHASE4_DIR
DEFAULT_OUTPUT = CANONICAL_FINAL_OUTPUT
DEFAULT_NUMERICAL_FOLLOWUP_CONFIG = phase3.DEFAULT_NUMERICAL_FOLLOWUP_CONFIG
DEFAULT_NUMERICAL_LOCK = phase3.DEFAULT_NUMERICAL_LOCK

CODE_DEPENDENCIES = (
    ROOT / "scripts" / "finalize_infobench_calibration_cat_2pl_only_v1.py",
    ROOT / "scripts" / "finalize_infobench_calibration_cat_v3.py",
    ROOT / "scripts" / "nested_cat_total_uncertainty_2pl_only_v1.py",
    ROOT / "scripts" / "nested_cat_total_uncertainty_v3.py",
    ROOT / "scripts" / "nested_cat_total_uncertainty_v2.py",
    ROOT / "scripts" / "nested_scenario_cat_cv_2pl_only_v1.py",
    ROOT / "scripts" / "nested_scenario_cat_cv.py",
    ROOT / "scripts" / "calibrate_mirt.py",
    ROOT / "scripts" / "kfold_cv_mirt.py",
    ROOT / "scripts" / "scenario_cat_lib.py",
    ROOT / "scripts" / "scenario_kfold_estimator_cv.py",
    ROOT / "tutor_cat" / "dataio.py",
    ROOT / "tutor_cat" / "engine.py",
    ROOT / "tutor_cat" / "mirt.py",
    ROOT / "tutor_cat" / "schemas.py",
    ROOT / "tutor_cat" / "selector.py",
)
REQUIRED_CODE_DEPENDENCIES = frozenset(CODE_DEPENDENCIES)
FINAL_OUTPUTS = v3.FINAL_OUTPUTS
FINAL_FIT_BUNDLE = v3.FINAL_FIT_BUNDLE

_ORIGINAL_SELECT_SPEC = v3.select_authorized_final_spec


def _validate_versioned_output(
    configured: str | Path,
    requested: str | Path,
    *,
    root: Path = ROOT,
    canonical: Path = CANONICAL_FINAL_OUTPUT,
) -> Path:
    """Pin output to the sole 2PL-only-v1 leaf without path aliasing."""

    canonical_lexical = v3._lexical_absolute(canonical)
    root_lexical = v3._lexical_absolute(root)
    required_parent = root_lexical / "runs" / "calibration" / "InFoBench_2pl_only_v1"
    if canonical_lexical != required_parent / "final_fit":
        raise v3.FinalFitError("internal 2PL-only canonical final-output definition changed")

    configured_path = Path(configured)
    if not configured_path.is_absolute():
        configured_path = root_lexical / configured_path
    requested_path = Path(requested)
    if not requested_path.is_absolute():
        requested_path = Path.cwd() / requested_path
    for label, path in (("configured", configured_path), ("requested", requested_path)):
        lexical = v3._lexical_absolute(path)
        if lexical != canonical_lexical or path.resolve(strict=False) != canonical_lexical:
            raise v3.FinalFitError(
                f"{label} final output is not the canonical 2PL-only-v1 final_fit leaf"
            )
        v3._assert_no_symlink_components(lexical, root=root_lexical)
    if canonical_lexical.exists() and not canonical_lexical.is_dir():
        raise v3.FinalFitError("canonical final output exists but is not a directory")
    return canonical_lexical


def _select_two_pl_spec(
    config: Mapping[str, Any],
    selected: Mapping[str, Any],
    phase3_decision: Mapping[str, Any],
    phase4_decision: Mapping[str, Any],
    phase3_manifest: Mapping[str, Any],
) -> phase3.CalibrationSpec:
    phase4.validate_two_pl_inner_contract(phase3_decision, selected)
    # The audited V3 validator has one historical literal (12 blocks/panel).
    # Validate the real 2PL contract first, then adapt only that literal in an
    # in-memory copy so all remaining authorization checks are reused.
    adapted = copy.deepcopy(dict(phase3_decision))
    adapted["inner_selection_fail_closed"]["expected_spec_fold_combinations_per_panel"] = 12
    with _two_pl_bindings():
        return _ORIGINAL_SELECT_SPEC(config, selected, adapted, phase4_decision, phase3_manifest)


_BINDINGS = {
    "phase3": phase3,
    "phase4": phase4,
    "SCRIPT_SCHEMA": SCRIPT_SCHEMA,
    "FIT_MANIFEST_SCHEMA": FIT_MANIFEST_SCHEMA,
    "FIT_TRANSACTION_SCHEMA": FIT_TRANSACTION_SCHEMA,
    "CHECKPOINT_SCHEMA": CHECKPOINT_SCHEMA,
    "EXPORT_MANIFEST_SCHEMA": EXPORT_MANIFEST_SCHEMA,
    "DECISION_SCHEMA": DECISION_SCHEMA,
    "EXPECTED_MODELS": EXPECTED_MODELS,
    "EXPECTED_FAMILIES": EXPECTED_FAMILIES,
    "EXPECTED_PANELS": EXPECTED_PANELS,
    "EXPECTED_REPEATS": EXPECTED_REPEATS,
    "PRIMARY_POLICY_ID": PRIMARY_POLICY_ID,
    "FINAL_BANK_SOURCE": FINAL_BANK_SOURCE,
    "CANONICAL_PHASE3_DIR": CANONICAL_PHASE3_DIR,
    "CANONICAL_PHASE4_DIR": CANONICAL_PHASE4_DIR,
    "CANONICAL_FINAL_OUTPUT": CANONICAL_FINAL_OUTPUT,
    "DEFAULT_CONFIG": DEFAULT_CONFIG,
    "DEFAULT_PHASE3": DEFAULT_PHASE3,
    "DEFAULT_PHASE4": DEFAULT_PHASE4,
    "DEFAULT_OUTPUT": DEFAULT_OUTPUT,
    "DEFAULT_NUMERICAL_FOLLOWUP_CONFIG": DEFAULT_NUMERICAL_FOLLOWUP_CONFIG,
    "DEFAULT_NUMERICAL_LOCK": DEFAULT_NUMERICAL_LOCK,
    "CODE_DEPENDENCIES": CODE_DEPENDENCIES,
    "REQUIRED_CODE_DEPENDENCIES": REQUIRED_CODE_DEPENDENCIES,
    "validate_canonical_output_leaf": _validate_versioned_output,
    "select_authorized_final_spec": _select_two_pl_spec,
}


@contextmanager
def _two_pl_bindings() -> Iterator[None]:
    previous = {name: getattr(v3, name) for name in _BINDINGS}
    for name, value in _BINDINGS.items():
        setattr(v3, name, value)
    try:
        yield
    finally:
        for name, value in previous.items():
            setattr(v3, name, value)


class FinalFitRunner(v3.FinalFitRunner):
    """Strict versioned finalizer using the unchanged audited implementation."""

    def __init__(self, args: argparse.Namespace):
        with _two_pl_bindings():
            super().__init__(args)

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
                "script": "scripts/finalize_infobench_calibration_cat_2pl_only_v1.py",
                "versioned_implementation": "scripts/finalize_infobench_calibration_cat_v3.py",
                "phase3_candidate_spec_ids": list(phase3.EXPECTED_PHASE3_SPEC_IDS),
                "v4_numerical_spec_ids": list(phase3.EXPECTED_V4_SPEC_IDS),
                "parent_v3_artifacts_read_or_written": False,
            }
        )
        return payload

    def _write_summary_markdown(
        self,
        decision: Mapping[str, Any],
        deployment: Mapping[str, Any],
        paired: Mapping[str, Any],
    ) -> None:
        deployment_attempts = (
            f"{deployment['n_attempts_observed']}/{deployment['n_attempts_expected']}"
        )
        paired_attempts = f"{paired['n_attempts_observed']}/{paired['n_attempts_expected']}"
        text = f"""# InFoBench 2PL-only-v1 final all-52 fit and replay

**Status:** {decision["status"]}

The exact 2PL calibration specification authorized by repeated Phase 3 and
passing Phase 4 was **`{self.final_spec.spec_id}`**. It was refit once on all 52
frozen tutor models using the V4-locked 401-node calibration grid and 801-node
scoring grid. The primary CAT policy remains floor 15, conditional SE 0.20, and
trace selection.

## Replay coverage

- Deployment replay: {deployment_attempts} recorded attempts.
- Paired 20-seed CAT/random panel: {paired_attempts} recorded attempts.
- All deployment CAT scores valid: {decision["leaderboard"]["all_52_deployment_scores_valid"]}.

## Interpretation

This final-bank replay is a same-cohort operational diagnostic, not independent
or unseen-family evidence. Cross-fitted Phase-3 results remain the internal
generalization proxy. All conclusions are conditional on frozen Qwen labels
that were not independently human-validated on InFoBench. This driver emits no
leaderboard.
"""
        self._write_or_verify_bytes(self.output_dir / "FINAL_FIT_SUMMARY.md", text.encode("utf-8"))


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phase3-dir", type=Path, default=DEFAULT_PHASE3)
    parser.add_argument("--phase4-dir", type=Path, default=DEFAULT_PHASE4)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--numerical-followup-config",
        type=Path,
        default=DEFAULT_NUMERICAL_FOLLOWUP_CONFIG,
    )
    parser.add_argument("--numerical-lock", type=Path, default=DEFAULT_NUMERICAL_LOCK)
    parser.add_argument("--skills", default=",".join(phase3.DEFAULT_SKILLS))
    parser.add_argument("--dimensions", default=phase3.DEFAULT_DIMENSIONS)
    parser.add_argument("--structure-name", default="infobench_overall_1d_v3")
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--max-scenarios", type=int, default=50)
    parser.add_argument("--minimum-scored-criteria", type=int, default=15)
    parser.add_argument("--mwle-ridge", type=float, default=1e-6)
    parser.add_argument("--metric-bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--negative-policy", choices=("error", "drop", "keep"), default="drop")
    parser.add_argument("--max-grid-nodes", type=int, default=50_000)
    parser.add_argument(
        "--require-complete-bank", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    try:
        if args.resume and args.plan_only:
            raise v3.FinalFitError("--resume and --plan-only cannot be combined")
        if args.max_grid_nodes < 801:
            raise v3.FinalFitError("max-grid-nodes cannot hold the locked 801-node grid")
        if args.metric_bootstrap_replicates < 100:
            raise v3.FinalFitError("family bootstrap requires at least 100 replicates")
        with _two_pl_bindings():
            return FinalFitRunner(args).run()
    except (
        v3.FinalFitError,
        phase4.TwoPLPhase4Error,
        phase4_v3.V3Phase4Error,
        phase4_engine.V2Phase4Error,
        phase3.V3Phase3Error,
        phase3_v1.NestedCVError,
        scat.OfflineStudyError,
        cm.CalibrationError,
        FileNotFoundError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
        np.linalg.LinAlgError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    finally:
        cm.configure_skills(None)


if __name__ == "__main__":
    raise SystemExit(main())
