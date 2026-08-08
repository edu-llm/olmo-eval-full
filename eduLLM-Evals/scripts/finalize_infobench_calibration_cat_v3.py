"""Authorized all-52 final fit and same-cohort replay for InFoBench V3.

This is the terminal *artifact-production* stage of the frozen InFoBench V3
study.  It may run only after:

* the V4 dense-fitter numerical lock passes;
* V3 Phase 3 passes every repeated nested-CV gate and names one stable modal
  exact calibration specification; and
* V3 Phase 4 passes every uncertainty/order gate and explicitly authorizes a
  final fit.

The driver then fits that one already-authorized specification on all 52
recorded tutors, exports the fitted-only bank, and performs two clearly
separated same-cohort diagnostics:

1. one deployment replay at the frozen master seed; and
2. paired CAT-versus-random replays at the 20 already-frozen order seeds.

The final-bank replays are operational diagnostics, not out-of-sample recovery
evidence.  The cross-fitted Phase-3 rows remain the headline internal evidence.
No leaderboard is emitted by this driver.  A downstream exploratory leaderboard
is allowed only when the terminal manifest says all 52 deployment scores are
present and valid, and it must retain the same-cohort/Qwen limitations.

``--plan-only`` validates the complete authorization/provenance chain and is
read-only.  Existing output is append-only: a fresh run requires an empty,
versioned leaf, and ``--resume`` accepts only byte-verified checkpoints bearing
the exact study signature.  This module never calls a tutor model or LLM judge.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import calibrate_mirt as cm  # noqa: E402
from scripts import kfold_cv_mirt as cell_cv  # noqa: E402
from scripts import nested_cat_total_uncertainty_v2 as phase4_engine  # noqa: E402
from scripts import nested_cat_total_uncertainty_v3 as phase4  # noqa: E402
from scripts import nested_scenario_cat_cv as phase3_v1  # noqa: E402
from scripts import nested_scenario_cat_cv_v3 as phase3  # noqa: E402
from scripts import scenario_cat_lib as scat  # noqa: E402
from scripts import scenario_kfold_estimator_cv as scenario_cv  # noqa: E402

SCRIPT_SCHEMA = "infobench-final-fit-replay-v3-v1"
FIT_MANIFEST_SCHEMA = "infobench-final-all52-fit-v3-v1"
FIT_TRANSACTION_SCHEMA = "infobench-final-all52-fit-transaction-v3-v1"
CHECKPOINT_SCHEMA = "infobench-final-replay-checkpoint-v3-v1"
EXPORT_MANIFEST_SCHEMA = "infobench-final-bank-export-v3-v1"
DECISION_SCHEMA = "infobench-final-fit-decision-v3-v1"

EXPECTED_MODELS = 52
EXPECTED_FAMILIES = 22
EXPECTED_PANELS = 25
EXPECTED_REPEATS = 5
PRIMARY_POLICY_ID = phase3.PRIMARY_POLICY_ID
FINAL_BANK_SOURCE = "calibrated-infobench-v3-final-all52"
CANONICAL_PHASE3_DIR = ROOT / "runs" / "calibration" / "InFoBench_v3" / "phase3"
CANONICAL_PHASE4_DIR = ROOT / "runs" / "calibration" / "InFoBench_v3" / "phase4"
CANONICAL_FINAL_OUTPUT = (
    ROOT / "runs" / "calibration" / "InFoBench_v3" / "final_fit"
)
FINAL_FIT_BUNDLE = "final_fit_bundle"
EXACT_SKILLS = ",".join(phase3.DEFAULT_SKILLS)
EXACT_DIMENSIONS = phase3.DEFAULT_DIMENSIONS
EXACT_STRUCTURE_NAME = "infobench_overall_1d_v3"
EXACT_MAX_GRID_NODES = 50_000

DEFAULT_CONFIG = phase3.DEFAULT_CONFIG
DEFAULT_PHASE3 = CANONICAL_PHASE3_DIR
DEFAULT_PHASE4 = CANONICAL_PHASE4_DIR
DEFAULT_OUTPUT = CANONICAL_FINAL_OUTPUT
DEFAULT_NUMERICAL_FOLLOWUP_CONFIG = phase3.DEFAULT_NUMERICAL_FOLLOWUP_CONFIG
DEFAULT_NUMERICAL_LOCK = phase3.DEFAULT_NUMERICAL_LOCK

# These are terminal artifacts only.  Checkpoints and the fit cache are hashed
# separately so an interrupted exact study can resume without rewriting them.
FINAL_OUTPUTS = (
    f"{FINAL_FIT_BUNDLE}/final_fit_arrays.npz",
    f"{FINAL_FIT_BUNDLE}/final_fit_manifest.json",
    f"{FINAL_FIT_BUNDLE}/TRANSACTION.json",
    "final_item_parameters.csv",
    "final_rubrics_fitted.jsonl",
    "final_scenarios_fitted.jsonl",
    "final_bank_export_manifest.json",
    "deployment_replay_per_model.csv",
    "paired_seed_replays.csv",
    "paired_seed_per_model_aggregate.csv",
    "final_replay_summary.json",
    "cat_length_evidence.csv",
    "final_fit_decision.json",
    "FINAL_FIT_SUMMARY.md",
)

CODE_DEPENDENCIES = (
    ROOT / "scripts" / "finalize_infobench_calibration_cat_v3.py",
    ROOT / "scripts" / "nested_cat_total_uncertainty_v3.py",
    ROOT / "scripts" / "nested_cat_total_uncertainty_v2.py",
    ROOT / "scripts" / "nested_scenario_cat_cv_v3.py",
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


class FinalFitError(RuntimeError):
    """A final-fit authorization, provenance, or completeness check failed."""


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        _json_ready(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FinalFitError(f"could not read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise FinalFitError(f"expected one JSON object in {path}")
    return value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(_json_ready(value), indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_fsynced_json(path: Path, value: Any) -> None:
    payload = (
        json.dumps(_json_ready(value), indent=2, sort_keys=True, ensure_ascii=False)
        + "\n"
    ).encode("utf-8")
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _tree_hash(path: Path) -> str:
    if not path.is_dir():
        return _canonical_hash({})
    return _canonical_hash(
        {
            str(item.relative_to(path)): _sha256(item)
            for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
        }
    )


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _environment_provenance() -> dict[str, Any]:
    payload = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "dependencies": phase3_v1._dependency_versions(),
    }
    return {**payload, "canonical_sha256": _canonical_hash(payload)}


def _code_hashes() -> dict[str, str]:
    if frozenset(CODE_DEPENDENCIES) != REQUIRED_CODE_DEPENDENCIES or len(
        CODE_DEPENDENCIES
    ) != len(REQUIRED_CODE_DEPENDENCIES):
        raise FinalFitError("direct code-dependency inventory is not exact")
    hashes: dict[str, str] = {}
    for path in CODE_DEPENDENCIES:
        if not path.is_file():
            raise FinalFitError(f"required code dependency is missing: {path}")
        hashes[_display_path(path)] = _sha256(path)
    return hashes


def validate_recorded_code_inventory(
    label: str, provenance: Mapping[str, Any]
) -> dict[str, str]:
    """Require the upstream code files to remain byte-identical."""

    files = provenance.get("files")
    if not isinstance(files, Mapping) or not files:
        raise FinalFitError(f"{label} has no recorded code-file inventory")
    normalized = {str(name): str(digest) for name, digest in files.items()}
    recorded_canonical = str(provenance.get("canonical_sha256") or "")
    if recorded_canonical != _canonical_hash(normalized):
        raise FinalFitError(f"{label} code inventory canonical hash is invalid")
    for name, digest in normalized.items():
        path = _repo_path(name)
        if not path.is_file() or _sha256(path) != digest:
            raise FinalFitError(f"{label} code dependency changed: {name}")
    return dict(sorted(normalized.items()))


def validate_frozen_scientific_contract(config: Mapping[str, Any]) -> None:
    """Assert every prospectively frozen Phase-3 scientific gate exactly."""

    gates = config.get("selection_gates")
    if not isinstance(gates, Mapping):
        raise FinalFitError("selection_gates must be a frozen mapping")
    exact_numbers = {
        "minimum_replay_success_rate": 0.99,
        "minimum_mwle_convergence_rate": 0.95,
        "minimum_nominal_precision_rate": 0.95,
        "minimum_nominal_precision_lower_95_ci": 0.90,
        "minimum_recovery_correlation_lower_95_ci": 0.85,
        "minimum_recovery_slope": 0.90,
        "maximum_recovery_slope": 1.10,
        "maximum_disjoint_pass_rate_mae": 0.07,
        "maximum_absolute_disjoint_pass_rate_bias": 0.03,
        "minimum_scenario_reduction_vs_random": 0.50,
        "minimum_valid_parameter_bootstrap_rate": 0.90,
    }
    for field, expected in exact_numbers.items():
        observed = gates.get(field)
        if (
            not isinstance(observed, (int, float, np.integer, np.floating))
            or isinstance(observed, (bool, np.bool_))
            or not math.isfinite(float(observed))
            or not _same_number(observed, expected)
        ):
            raise FinalFitError(f"frozen Phase-3 scientific gate changed: {field}")
    exact_booleans = {
        "apply_to_primary_only": True,
        "allow_fallback_if_primary_fails": False,
        "require_paired_family_bootstrap_ci_favors_cat": True,
        "require_every_outer_panel": True,
        "require_every_repetition_pooled_gate": True,
        "treat_260_model_repeat_rows_as_independent": False,
    }
    for field, expected in exact_booleans.items():
        if gates.get(field) is not expected:
            raise FinalFitError(f"frozen Phase-3 scientific gate changed: {field}")
    expected_panel_applied = [
        "replay_success_rate",
        "mwle_convergence_rate",
        "nominal_precision_rate",
        "recovery_slope",
        "disjoint_pass_rate_mae",
        "absolute_disjoint_pass_rate_bias",
        "scenario_reduction_vs_random",
    ]
    expected_panel_diagnostic = [
        "nominal_precision_lower_95_ci",
        "recovery_correlation_lower_95_ci",
        "paired_ci_favors_cat",
    ]
    expected_repetition = [
        "replay_success_rate",
        "mwle_convergence_rate",
        "nominal_precision_rate",
        "nominal_precision_lower_95_ci",
        "recovery_correlation_lower_95_ci",
        "recovery_slope",
        "disjoint_pass_rate_mae",
        "absolute_disjoint_pass_rate_bias",
        "scenario_reduction_vs_random",
        "paired_ci_favors_cat",
    ]
    exact_lists = {
        "outer_panel_applied_gate_names": expected_panel_applied,
        "outer_panel_diagnostic_gate_names": expected_panel_diagnostic,
        "repetition_applied_gate_names": expected_repetition,
    }
    for field, expected in exact_lists.items():
        if gates.get(field) != expected:
            raise FinalFitError(f"frozen Phase-3 gate inventory changed: {field}")
    if gates.get("repetition_expected_unique_oof_tutors") != EXPECTED_MODELS:
        raise FinalFitError("Phase-3 repetition denominator changed")

    selection = config.get("calibration_selection") or {}
    modal_threshold = selection.get(
        "minimum_exact_spec_modal_fraction_across_outer_panels"
    )
    if (
        selection.get("allow_fallback") is not False
        or selection.get("require_all_specs_every_inner_fold") is not True
        or selection.get("survivor_selection_allowed") is not False
        or not isinstance(modal_threshold, (int, float, np.integer, np.floating))
        or isinstance(modal_threshold, (bool, np.bool_))
        or not _same_number(modal_threshold, 0.80)
    ):
        raise FinalFitError("frozen calibration-selection fail-closed contract changed")
    cat = config.get("cat_policies") or {}
    if cat.get("sensitivities_can_replace_primary") is not False:
        raise FinalFitError("sensitivity policy promotion became possible")
    uncertainty = config.get("uncertainty") or {}
    bootstrap_rate = uncertainty.get("minimum_valid_parameter_bootstrap_rate")
    if (
        not isinstance(bootstrap_rate, (int, float, np.integer, np.floating))
        or isinstance(bootstrap_rate, (bool, np.bool_))
        or not _same_number(bootstrap_rate, 0.90)
    ):
        raise FinalFitError("Phase-4 valid parameter-bootstrap gate changed")


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _assert_no_symlink_components(path: Path, *, root: Path) -> None:
    lexical_path = _lexical_absolute(path)
    lexical_root = _lexical_absolute(root)
    try:
        relative = lexical_path.relative_to(lexical_root)
    except ValueError as error:
        raise FinalFitError("final output is outside the repository root") from error
    current = lexical_root
    if current.exists() and current.is_symlink():
        raise FinalFitError("repository root is a symlink")
    for part in relative.parts:
        current = current / part
        if current.exists() and current.is_symlink():
            raise FinalFitError(f"final output traverses a symlink: {current}")


def _assert_source_has_no_symlink_components(path: Path) -> None:
    lexical = _lexical_absolute(path)
    try:
        lexical.relative_to(_lexical_absolute(ROOT))
    except ValueError:
        current = Path(lexical.anchor)
        for part in lexical.parts[1:]:
            current = current / part
            if current.exists() and current.is_symlink():
                raise FinalFitError(
                    f"provenance source traverses a symlink: {current}"
                ) from None
    else:
        _assert_no_symlink_components(lexical, root=ROOT)


def validate_canonical_output_leaf(
    configured: str | Path,
    requested: str | Path,
    *,
    root: Path = ROOT,
    canonical: Path = CANONICAL_FINAL_OUTPUT,
) -> Path:
    """Pin final output to the sole V3 leaf without symlink/path traversal."""

    canonical_lexical = _lexical_absolute(canonical)
    root_lexical = _lexical_absolute(root)
    required_parent = root_lexical / "runs" / "calibration" / "InFoBench_v3"
    if canonical_lexical != required_parent / "final_fit":
        raise FinalFitError("internal canonical final-output definition changed")

    configured_path = Path(configured)
    if not configured_path.is_absolute():
        configured_path = root_lexical / configured_path
    requested_path = Path(requested)
    if not requested_path.is_absolute():
        requested_path = Path.cwd() / requested_path
    for label, path in (("configured", configured_path), ("requested", requested_path)):
        lexical = _lexical_absolute(path)
        if lexical != canonical_lexical or path.resolve(strict=False) != canonical_lexical:
            raise FinalFitError(
                f"{label} final output is not the canonical V3 final_fit leaf"
            )
        _assert_no_symlink_components(lexical, root=root_lexical)
    if canonical_lexical.exists() and not canonical_lexical.is_dir():
        raise FinalFitError("canonical final output exists but is not a directory")
    return canonical_lexical


def validate_exact_runtime_axis(args: argparse.Namespace) -> dict[str, Any]:
    """Freeze the complete InFoBench latent/runtime axis used by final fitting."""

    expected = {
        "skills": EXACT_SKILLS,
        "dimensions": EXACT_DIMENSIONS,
        "structure_name": EXACT_STRUCTURE_NAME,
        "require_complete_bank": True,
        "max_grid_nodes": EXACT_MAX_GRID_NODES,
    }
    observed = {field: getattr(args, field, None) for field in expected}
    if observed != expected:
        changed = [field for field in expected if observed[field] != expected[field]]
        raise FinalFitError(f"final InFoBench runtime axis changed: {changed}")
    return {
        "source_skills_order": list(phase3.DEFAULT_SKILLS),
        "source_skills_csv": EXACT_SKILLS,
        "dimensions": EXACT_DIMENSIONS,
        "structure_name": EXACT_STRUCTURE_NAME,
        "require_complete_bank": True,
        "max_grid_nodes": EXACT_MAX_GRID_NODES,
    }


@dataclass(frozen=True)
class CapturedProvenance:
    """Immutable path/hash snapshot that must be reverified at every boundary."""

    entries: dict[str, dict[str, str]]
    canonical_sha256: str

    @classmethod
    def capture(cls, paths: Mapping[str, Path]) -> CapturedProvenance:
        if not paths:
            raise FinalFitError("cannot capture an empty provenance inventory")
        entries: dict[str, dict[str, str]] = {}
        seen_paths: dict[Path, str] = {}
        for name, raw_path in sorted(paths.items()):
            lexical = _lexical_absolute(raw_path)
            _assert_source_has_no_symlink_components(lexical)
            if not lexical.is_file():
                raise FinalFitError(f"provenance source is missing or symlinked: {name}")
            path = lexical.resolve()
            prior = seen_paths.get(path)
            if prior is not None:
                raise FinalFitError(f"provenance aliases one file as {prior!r} and {name!r}")
            seen_paths[path] = str(name)
            entries[str(name)] = {
                "path": _display_path(path),
                "sha256": _sha256(path),
            }
        return cls(entries=entries, canonical_sha256=_canonical_hash(entries))

    def sha256(self, name: str) -> str:
        try:
            return self.entries[name]["sha256"]
        except KeyError as error:
            raise FinalFitError(f"uncaptured provenance key: {name}") from error

    def reverify(self, stage: str) -> None:
        if self.canonical_sha256 != _canonical_hash(self.entries):
            raise FinalFitError("captured provenance inventory mutated in memory")
        for name, raw in self.entries.items():
            stored = Path(raw["path"])
            path = stored if stored.is_absolute() else ROOT / stored
            path = _lexical_absolute(path)
            _assert_source_has_no_symlink_components(path)
            if not path.is_file() or _sha256(path) != raw["sha256"]:
                raise FinalFitError(f"captured provenance changed before {stage}: {name}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "files": copy.deepcopy(self.entries),
            "canonical_sha256": self.canonical_sha256,
        }


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def _is_exact_bool(value: Any) -> bool:
    return isinstance(value, (bool, np.bool_))


def _same_number(left: Any, right: float) -> bool:
    try:
        return math.isclose(float(left), right, rel_tol=0, abs_tol=0)
    except (TypeError, ValueError):
        return False


def _is_canonical_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_phase4_artifact_inventory(
    phase4_dir: Path, manifest: Mapping[str, Any]
) -> dict[str, str]:
    """Hash-check the complete passing Phase-4 handoff."""

    directory = phase4_dir.resolve()
    if manifest.get("schema_version") != phase4.SCRIPT_SCHEMA:
        raise FinalFitError("Phase-4 manifest is not the V3 schema")
    if manifest.get("status") != "phase4_complete_pass":
        raise FinalFitError("Phase 4 is not a completed pass")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping) or set(map(str, outputs)) != set(
        phase4_engine.FINAL_OUTPUTS
    ):
        raise FinalFitError("Phase-4 output inventory is incomplete or has extras")

    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file() or _read_json(manifest_path) != dict(manifest):
        raise FinalFitError("Phase-4 manifest changed after loading")
    inventory = {"manifest.json": _sha256(manifest_path)}
    for name in phase4_engine.FINAL_OUTPUTS:
        raw = outputs.get(name)
        path = directory / name
        if not isinstance(raw, Mapping) or not path.is_file():
            raise FinalFitError(f"required Phase-4 artifact is missing: {name}")
        recorded_path = str(raw.get("path") or "")
        if recorded_path not in {name, _display_path(path), str(path.resolve())}:
            raise FinalFitError(f"Phase-4 artifact path changed: {name}")
        observed = _sha256(path)
        if str(raw.get("sha256") or "") != observed:
            raise FinalFitError(f"Phase-4 artifact failed hash validation: {name}")
        inventory[name] = observed
    if manifest.get("checkpoint_tree_sha256") != _tree_hash(
        directory / "phase4_checkpoints"
    ):
        raise FinalFitError("Phase-4 checkpoint tree failed hash validation")
    if manifest.get("bootstrap_cache_tree_sha256") != _tree_hash(
        directory / "phase4_cache"
    ):
        raise FinalFitError("Phase-4 bootstrap cache failed hash validation")
    return dict(sorted(inventory.items()))


def validate_phase4_decision(
    decision: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    phase4_dir: Path,
    phase3_manifest: Mapping[str, Any],
    config_path: Path,
    phase3_manifest_path: Path,
    phase3_decision_path: Path,
    selected_path: Path,
) -> pd.DataFrame:
    """Require every frozen Phase-4 gate—not just a permissive status flag."""

    required = {
        "schema_version": phase4.DECISION_SCHEMA,
        "status": "pass",
        "phase4_pass": True,
        "all_five_repetitions_pass": True,
        "final_fit_authorized": True,
        "sensitivity_policies_considered": False,
        "model_repeat_rows_treated_as_independent": False,
    }
    for field, expected in required.items():
        observed = decision.get(field)
        if isinstance(expected, bool):
            matches = _is_exact_bool(observed) and bool(observed) is expected
        else:
            matches = observed == expected and isinstance(observed, type(expected))
        if not matches:
            raise FinalFitError(f"Phase-4 decision did not authorize final fit: {field}")
    if not isinstance(decision.get("failed_repeats"), list) or decision.get(
        "failed_repeats"
    ):
        raise FinalFitError("Phase-4 decision contains failed repetitions")
    if decision.get("phase3_study_signature") != phase3_manifest.get("study_signature"):
        raise FinalFitError("Phase-4 decision refers to a different Phase-3 study")
    if decision.get("study_signature") != manifest.get("study_signature"):
        raise FinalFitError("Phase-4 decision and manifest study signatures differ")
    if not _is_canonical_sha256(decision.get("study_signature")):
        raise FinalFitError("Phase-4 study signature is not a canonical SHA-256")
    expected_primary = {
        "policy_id": PRIMARY_POLICY_ID,
        "role": "primary",
        "minimum_scenarios": 15,
        "conditional_se_target": 0.2,
        "selector": "trace",
    }
    observed_primary = decision.get("primary_policy")
    if (
        observed_primary != expected_primary
        or not isinstance(observed_primary, Mapping)
        or not _is_exact_int(observed_primary.get("minimum_scenarios"))
        or isinstance(observed_primary.get("conditional_se_target"), bool)
        or not _same_number(observed_primary.get("conditional_se_target"), 0.2)
    ):
        raise FinalFitError("Phase-4 decision changed the frozen primary policy")
    phase3_inputs_raw = phase3_manifest.get("inputs") or {}
    phase3_inputs = {
        name: str((phase3_inputs_raw.get(name) or {}).get("sha256") or "")
        for name in ("response_matrix", "rubrics", "scenarios", "judge_manifest")
    }
    if not all(phase3_inputs.values()) or decision.get("input_hashes") != phase3_inputs:
        raise FinalFitError("Phase-4 decision changed the frozen input hashes")
    expected_hashes = {
        "config_sha256": _sha256(config_path),
        "phase3_manifest_sha256": _sha256(phase3_manifest_path),
        "selected_calibration_specs_sha256": _sha256(selected_path),
        "phase3_decision_sha256": _sha256(phase3_decision_path),
    }
    # Historical engine field name is pre_outer_selection_lock_sha256; it is
    # validated by the Phase-4 runner and retained in its manifest.  The fields
    # below directly bind every source consumed again by this terminal stage.
    for field, expected in expected_hashes.items():
        if decision.get(field) != expected:
            raise FinalFitError(f"Phase-4 decision provenance changed {field}")
    if (manifest.get("decision") or {}) != dict(decision):
        raise FinalFitError("Phase-4 manifest and decision file disagree")

    gate_raw = decision.get("gate_table")
    if not isinstance(gate_raw, Mapping):
        raise FinalFitError("Phase-4 decision has no gate-table provenance")
    gate_path = phase4_dir / str(gate_raw.get("path") or "")
    if gate_path.resolve() != (phase4_dir / "repetition_gate_results.csv").resolve():
        raise FinalFitError("Phase-4 gate table path is not canonical")
    if not gate_path.is_file() or _sha256(gate_path) != gate_raw.get("sha256"):
        raise FinalFitError("Phase-4 gate table failed hash validation")
    gates = pd.read_csv(gate_path)
    if (
        len(gates) != EXPECTED_REPEATS
        or "repeat" not in gates
        or not gates["repeat"].map(_is_exact_int).all()
        or set(map(int, gates["repeat"])) != set(range(EXPECTED_REPEATS))
    ):
        raise FinalFitError("Phase-4 gate table does not contain repetitions 0..4")
    required_true = (
        "complete_total_se_support",
        "complete_order_support",
        "every_model_valid_draw_gate",
        "p90_total_se_gate",
        "every_order_seed_present_and_valid",
        "order_stability_gate",
        "repetition_phase4_pass",
    )
    for field in required_true:
        if (
            field not in gates
            or not gates[field].map(_is_exact_bool).all()
            or not gates[field].map(bool).all()
        ):
            raise FinalFitError(f"Phase-4 gate table failed {field}")
    numeric_contract = {
        "n_models_expected": (lambda value: int(value) == EXPECTED_MODELS),
        "n_total_se_rows": (lambda value: int(value) == EXPECTED_MODELS),
        "n_order_rows": (lambda value: int(value) == EXPECTED_MODELS),
        "minimum_valid_draw_rate_required": (
            lambda value: _same_number(value, 0.9)
        ),
        "minimum_valid_draw_rate_observed": (
            lambda value: _finite_number(value) and 0.9 <= float(value) <= 1.0
        ),
        "maximum_p90_cat_total_se": (lambda value: _same_number(value, 0.5)),
        "p90_cat_total_se": (
            lambda value: _finite_number(value) and 0.0 <= float(value) <= 0.5
        ),
        "maximum_median_order_path_sd": (
            lambda value: _same_number(value, 0.2)
        ),
        "median_order_path_sd": (
            lambda value: _finite_number(value) and 0.0 <= float(value) <= 0.2
        ),
    }
    for field, predicate in numeric_contract.items():
        if field not in gates:
            raise FinalFitError(f"Phase-4 gate table lacks {field}")
        try:
            passed = gates[field].map(predicate).all()
        except (TypeError, ValueError, OverflowError):
            passed = False
        if not passed:
            raise FinalFitError(f"Phase-4 numeric gate contract failed {field}")
    for field in ("n_models_expected", "n_total_se_rows", "n_order_rows"):
        if not gates[field].map(_is_exact_int).all():
            raise FinalFitError(f"Phase-4 count is not an exact integer: {field}")
    return gates.sort_values("repeat").reset_index(drop=True)


def select_authorized_final_spec(
    config: Mapping[str, Any],
    selected: Mapping[str, Any],
    phase3_decision: Mapping[str, Any],
    phase4_decision: Mapping[str, Any],
    phase3_manifest: Mapping[str, Any],
) -> phase3.CalibrationSpec:
    """Return the unique stable modal exact spec, or fail without a fallback."""

    specs = phase3.load_calibration_specs(config)
    by_id = {spec.spec_id: spec for spec in specs}
    phase3_signature = phase3_manifest.get("study_signature")
    if (
        phase3_manifest.get("schema_version") != phase3.SCRIPT_SCHEMA
        or phase3_manifest.get("status") != "phase3_complete"
        or not _is_canonical_sha256(phase3_signature)
        or phase3_decision.get("study_signature") != phase3_signature
    ):
        raise FinalFitError("Phase-3 manifest/decision authorization is invalid")
    if selected.get("schema_version") != phase3.SELECTED_SPECS_SCHEMA:
        raise FinalFitError("selected-calibration artifact has the wrong schema")
    if selected.get("study_signature") != phase3_manifest.get("study_signature"):
        raise FinalFitError("selected-calibration artifact has the wrong study signature")
    if selected.get("sensitivity_promotion_allowed") is not False:
        raise FinalFitError("selected-calibration artifact permits sensitivity promotion")
    primary = selected.get("primary_policy") or {}
    expected_primary = asdict(phase3.load_policies(config)[0])
    if primary != expected_primary:
        raise FinalFitError("selected-calibration artifact changed the primary policy")

    panels = selected.get("panels")
    if not isinstance(panels, list) or len(panels) != EXPECTED_PANELS:
        raise FinalFitError("selected-calibration artifact lacks all 25 panels")
    keys: set[tuple[int, int]] = set()
    panel_specs: dict[tuple[int, int], str] = {}
    for raw in panels:
        if not isinstance(raw, Mapping):
            raise FinalFitError("selected-calibration panel is malformed")
        if not _is_exact_int(raw.get("repeat")) or not _is_exact_int(
            raw.get("outer_fold")
        ):
            raise FinalFitError("selected-calibration panel coordinates are not integers")
        key = (int(raw["repeat"]), int(raw["outer_fold"]))
        if key in keys or key[0] not in range(5) or key[1] not in range(5):
            raise FinalFitError("selected-calibration panel keys are not the 5x5 grid")
        keys.add(key)
        spec_id = str(raw.get("selected_spec_id") or "")
        spec = by_id.get(spec_id)
        if (
            spec is None
            or raw.get("selected_calibration_specification") != spec.canonical
            or raw.get("status") != "evaluation_complete"
            or ((raw.get("inner_selection") or {}).get("complete_inner_evidence_audit") or {}).get(
                "passed"
            )
            is not True
        ):
            raise FinalFitError(f"panel {key} is not a complete authorized selection")
        panel_specs[key] = spec_id
    if keys != {(repeat, fold) for repeat in range(5) for fold in range(5)}:
        raise FinalFitError("selected-calibration panels do not cover the 5x5 grid")

    counts = Counter(panel_specs.values())
    maximum = max(counts.values(), default=0)
    modes = sorted(spec_id for spec_id, count in counts.items() if count == maximum)
    if len(modes) != 1:
        raise FinalFitError("calibration specification has no unique modal choice")
    modal = modes[0]
    modal_fraction = maximum / EXPECTED_PANELS
    stability = phase3_decision.get("calibration_stability") or {}
    inner = phase3_decision.get("inner_selection_fail_closed") or {}
    coverage = phase3_decision.get("coverage") or {}
    primary_decision = phase3_decision.get("primary_policy") or {}
    sensitivities = phase3_decision.get("sensitivities") or {}
    coverage_by_repetition = coverage.get("by_repetition")
    exact_coverage = isinstance(coverage_by_repetition, Mapping) and set(
        map(str, coverage_by_repetition)
    ) == {str(repeat) for repeat in range(EXPECTED_REPEATS)}
    if exact_coverage:
        for repeat in range(EXPECTED_REPEATS):
            raw = coverage_by_repetition[str(repeat)]
            if (
                not isinstance(raw, Mapping)
                or not _is_exact_int(raw.get("n_primary_rows"))
                or int(raw["n_primary_rows"]) != EXPECTED_MODELS
                or not _is_exact_int(raw.get("n_unique_models"))
                or int(raw["n_unique_models"]) != EXPECTED_MODELS
                or raw.get("all_52_models_exactly_once") is not True
            ):
                exact_coverage = False
                break
    if (
        phase3_decision.get("schema_version") != phase3.DECISION_SCHEMA
        or phase3_decision.get("status") != "pass"
        or phase3_decision.get("phase3_pass") is not True
        or phase3_decision.get("phase4_authorized") is not True
        or phase3_decision.get("failed_conditions") != []
        or inner.get("require_all_specs_every_inner_fold") is not True
        or not _is_exact_int(inner.get("expected_spec_fold_combinations_per_panel"))
        or int(inner["expected_spec_fold_combinations_per_panel"]) != 12
        or inner.get("survivor_selection_allowed") is not False
        or inner.get("all_panel_selections_locked_before_outer_outcomes") is not True
        or inner.get("all_25_panels_complete") is not True
        or coverage.get("all_25_panels_evaluated") is not True
        or coverage.get("all_52_models_each_repetition") is not True
        or coverage.get("model_repeat_rows_treated_as_independent") is not False
        or not exact_coverage
        or stability.get("passed") is not True
        or stability.get("unique_modal_spec_id") != modal
        or stability.get("tied_modal_spec_ids") != []
        or not _is_exact_int(stability.get("modal_count"))
        or int(stability.get("modal_count", -1)) != maximum
        or not _same_number(stability.get("modal_fraction_all_25_panels"), modal_fraction)
        or modal_fraction < 0.80
        or not _same_number(stability.get("required_fraction"), 0.80)
        or primary_decision.get("policy") != expected_primary
        or primary_decision.get("all_outer_panels_pass") is not True
        or primary_decision.get("all_repetitions_pass") is not True
        or primary_decision.get("failed_panel_ids") != []
        or primary_decision.get("failed_repetitions") != []
        or sensitivities.get("diagnostic_only") is not True
        or sensitivities.get("can_promote") is not False
        or sensitivities.get("affected_phase3_decision") is not False
    ):
        raise FinalFitError("Phase-3 calibration specification is not stably authorized")

    phase4_panels = phase4_decision.get("panel_selected_specifications")
    if not isinstance(phase4_panels, list) or len(phase4_panels) != EXPECTED_PANELS:
        raise FinalFitError("Phase-4 decision lacks its 25 selected specifications")
    observed_phase4: dict[tuple[int, int], str] = {}
    for raw in phase4_panels:
        if (
            not isinstance(raw, Mapping)
            or not _is_exact_int(raw.get("repeat"))
            or not _is_exact_int(raw.get("outer_fold"))
        ):
            raise FinalFitError("Phase-4 panel coordinates are not exact integers")
        key = (int(raw["repeat"]), int(raw["outer_fold"]))
        spec_id = str(raw.get("spec_id") or "")
        spec = by_id.get(spec_id)
        if key in observed_phase4 or spec is None:
            raise FinalFitError("Phase-4 panel specification inventory is malformed")
        if raw.get("calibration_specification") != spec.canonical:
            raise FinalFitError("Phase-4 exact calibration specification changed")
        observed_phase4[key] = spec_id
    if observed_phase4 != panel_specs:
        raise FinalFitError("Phase 3 and Phase 4 disagree on panel specifications")
    return by_id[modal]


def _fit_validity_or_raise(
    fit: Mapping[str, Any], spec: phase3.CalibrationSpec, *, context: str
) -> dict[str, Any]:
    validity = phase4.dense_fit_validity(fit, spec)
    if validity.get("fit_valid") is not True:
        failed = sorted(
            key
            for key, value in validity.items()
            if key not in {"maximum_observed_inner_gradient", "maximum_allowed_inner_gradient"}
            and value is False
        )
        raise FinalFitError(f"{context} violates the V4 dense-fit contract: {failed}")
    return validity


def _is_exact_int(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def validate_final_fit_contract(
    fit: Mapping[str, Any],
    spec: phase3.CalibrationSpec,
    expected_items: Sequence[str],
) -> dict[str, Any]:
    """Validate final-fit arrays, roster, likelihood, parameter count, and trace."""

    items = tuple(map(str, fit.get("items") or ()))
    expected = tuple(map(str, expected_items))
    if items != expected or len(items) != len(set(items)):
        raise FinalFitError("final fit item roster/order differs from prepared all-52 roster")
    n_items = len(expected)
    A = np.asarray(fit.get("A"), dtype=float)
    b = np.asarray(fit.get("b"), dtype=float)
    R = np.asarray(fit.get("R"), dtype=float)
    if A.shape != (n_items, 1) or b.shape != (n_items,) or R.shape != (1, 1):
        raise FinalFitError("final fit array shapes do not match item roster/overall-1D")
    if not all(np.all(np.isfinite(array)) for array in (A, b, R)):
        raise FinalFitError("final fit arrays contain non-finite values")
    if not np.array_equal(R, np.eye(1)):
        raise FinalFitError("final fit latent correlation must be exact 1x1 identity")
    if list(fit.get("dim_labels") or ()) != ["instruction_following"]:
        raise FinalFitError("final fit dimension labels changed")
    if fit.get("converged") is not True:
        raise FinalFitError("final fit did not report exact converged=True")
    if not _is_exact_int(fit.get("grid_nodes")) or int(fit["grid_nodes"]) != 401:
        raise FinalFitError("final fit did not use the locked 401-node grid")
    if fit.get("quadrature_method") != cm.NORMAL_TRAPEZOID_QUADRATURE:
        raise FinalFitError("final fit quadrature method changed")
    if not _finite_number(fit.get("quadrature_linear_bound")) or not _same_number(
        fit["quadrature_linear_bound"], 8.0
    ):
        raise FinalFitError("final fit quadrature bound changed")
    if fit.get("convergence_mode") != cm.RETURNED_ITERATE_CONVERGENCE:
        raise FinalFitError("final fit convergence mode changed")
    if fit.get("calibration_specification") != spec.canonical:
        raise FinalFitError("final fit calibration specification changed")
    if not math.isfinite(float(fit.get("loglik", math.nan))):
        raise FinalFitError("final fit marginal log likelihood is non-finite")
    if not math.isfinite(float(fit.get("penalized_objective", math.nan))):
        raise FinalFitError("final fit penalized objective is non-finite")
    expected_n_params = n_items if spec.family == cm.ONE_PL else 2 * n_items
    if not _is_exact_int(fit.get("n_params")) or int(fit["n_params"]) != expected_n_params:
        raise FinalFitError("final fit reports an incorrect parameter count")
    if not _is_exact_int(fit.get("n_iter")):
        raise FinalFitError("final fit iteration count is not an exact integer")
    n_iter = int(fit["n_iter"])
    if n_iter < 1 or n_iter > 1500:
        raise FinalFitError("final fit iteration count is outside 1..1500")

    diagnostics = fit.get("convergence_diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise FinalFitError("final fit lacks convergence diagnostics")
    trace = diagnostics.get("trace")
    if not isinstance(trace, list) or len(trace) != n_iter:
        raise FinalFitError("final fit optimizer trace length does not equal n_iter")
    expected_iterations = list(range(1, n_iter + 1))
    expected_optimizer_method = {
        cm.ONE_PL: "vectorized_1d_1pl_newton",
        cm.LOG_SHRINKAGE_2PL: "vectorized_1d_log_shrinkage_newton",
        cm.FREE_2PL: "per_item_warm_started",
    }[spec.family]
    observed_iterations: list[int] = []
    maximum_gradient = 0.0
    for position, row in enumerate(trace, 1):
        if not isinstance(row, Mapping) or not _is_exact_int(row.get("iteration")):
            raise FinalFitError(f"final fit trace row {position} has invalid iteration")
        observed_iterations.append(int(row["iteration"]))
        optimizer = row.get("mstep_optimizer")
        if not isinstance(optimizer, Mapping):
            raise FinalFitError(f"final fit trace row {position} lacks optimizer audit")
        if optimizer.get("method") != expected_optimizer_method:
            raise FinalFitError(
                f"final fit trace row {position} has invalid optimizer method"
            )
        for field, expected_count in (
            ("n_items", n_items),
            ("n_converged", n_items),
            ("n_failed", 0),
        ):
            if not _is_exact_int(optimizer.get(field)) or int(optimizer[field]) != expected_count:
                raise FinalFitError(
                    f"final fit trace row {position} has invalid optimizer {field}"
                )
        gradient = float(optimizer.get("max_abs_gradient", math.nan))
        if (
            not math.isfinite(gradient)
            or gradient < 0
            or gradient > phase4.MAXIMUM_INNER_GRADIENT
        ):
            raise FinalFitError(f"final fit trace row {position} has invalid gradient")
        maximum_gradient = max(maximum_gradient, gradient)
        for field in ("marginal_loglik", "penalized_objective", "objective_change"):
            if not math.isfinite(float(row.get(field, math.nan))):
                raise FinalFitError(f"final fit trace row {position} has non-finite {field}")
    if observed_iterations != expected_iterations:
        raise FinalFitError("final fit optimizer iteration sequence is not contiguous")
    if not _same_number(fit["loglik"], trace[-1]["marginal_loglik"]):
        raise FinalFitError(
            "final fit loglik disagrees with the final optimizer trace row"
        )
    if not _same_number(
        fit["penalized_objective"], trace[-1]["penalized_objective"]
    ):
        raise FinalFitError(
            "final fit penalized_objective disagrees with the final optimizer trace row"
        )
    dense_validity = _fit_validity_or_raise(fit, spec, context="all-52 final fit")
    return {
        **dense_validity,
        "expected_item_count": n_items,
        "item_roster_sha256": _canonical_hash(expected),
        "expected_n_params": expected_n_params,
        "trace_length": len(trace),
        "maximum_observed_inner_gradient": maximum_gradient,
        "array_shapes_valid": True,
        "latent_correlation_exact_identity": True,
        "loglik_finite": True,
    }


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and math.isfinite(float(value))
    )


def _parse_scenario_order(value: Any, *, context: str) -> list[str]:
    if not isinstance(value, str):
        raise FinalFitError(f"{context} scenario order is not serialized JSON")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise FinalFitError(f"{context} scenario order is invalid JSON") from error
    if (
        not isinstance(parsed, list)
        or any(not isinstance(item, str) or not item for item in parsed)
        or len(parsed) != len(set(parsed))
    ):
        raise FinalFitError(f"{context} scenario order is not a unique string list")
    return parsed


def validate_replay_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_model_to_family: Mapping[str, str],
    expected_seeds: Sequence[int],
    expected_scope: str,
    expected_policy: phase3.Policy,
    expected_spec_id: str,
    expected_cache_key: str,
    administration_scenario_ids: Sequence[str],
    maximum_scenarios: int,
) -> list[dict[str, Any]]:
    """Validate replay rows as scientific records, not merely as present keys."""

    model_to_family = {
        str(model): str(family) for model, family in expected_model_to_family.items()
    }
    seeds = tuple(int(seed) for seed in expected_seeds)
    if not model_to_family or not seeds or len(seeds) != len(set(seeds)):
        raise FinalFitError("invalid expected replay model/seed grid")
    expected_keys = {
        (model, seed) for model in model_to_family for seed in seeds
    }
    allowed_scenarios = set(map(str, administration_scenario_ids))
    if not allowed_scenarios:
        raise FinalFitError("replay validation has no administration scenarios")
    expected_candidate_id = expected_policy.as_candidate().candidate_id
    expected_rows: list[dict[str, Any]] = []
    observed_keys: list[tuple[str, int]] = []

    for position, source in enumerate(rows, 1):
        if not isinstance(source, Mapping):
            raise FinalFitError(f"replay row {position} is not a mapping")
        row = dict(source)
        model = row.get("model")
        seed = row.get("replay_seed")
        if not isinstance(model, str) or model not in model_to_family:
            raise FinalFitError(f"replay row {position} has an unexpected model")
        if not _is_exact_int(seed) or int(seed) not in seeds:
            raise FinalFitError(f"replay row {position} has an unexpected seed")
        observed_keys.append((model, int(seed)))

        exact_values = {
            "model_family": model_to_family[model],
            "candidate_id": expected_candidate_id,
            "policy_id": expected_policy.policy_id,
            "policy_role": expected_policy.role,
            "replay_scope": expected_scope,
            "spec_id": expected_spec_id,
            "fit_cache_key": expected_cache_key,
            "final_fit_cache_key": expected_cache_key,
            "selector": expected_policy.selector,
        }
        for field, expected in exact_values.items():
            if row.get(field) != expected or not isinstance(row.get(field), str):
                raise FinalFitError(f"replay row {position} changed {field}")
        if (
            not _is_exact_int(row.get("minimum_scenarios"))
            or int(row["minimum_scenarios"]) != expected_policy.minimum_scenarios
        ):
            raise FinalFitError(f"replay row {position} changed minimum_scenarios")
        if (
            not _finite_number(row.get("conditional_se_target"))
            or not _same_number(
                row["conditional_se_target"], expected_policy.conditional_se_target
            )
        ):
            raise FinalFitError(f"replay row {position} changed conditional_se_target")
        if row.get("same_cohort_final_bank") is not True:
            raise FinalFitError(f"replay row {position} changed same-cohort scope")
        if row.get("out_of_sample") is not False:
            raise FinalFitError(f"replay row {position} changed out-of-sample scope")

        status = row.get("status")
        if status not in {"ok", "replay_error"} or not isinstance(status, str):
            raise FinalFitError(f"replay row {position} has an invalid status")
        error_text = row.get("error")
        if not isinstance(error_text, str):
            raise FinalFitError(f"replay row {position} has a non-string error")

        arm_successes: list[bool] = []
        for arm in ("cat", "baseline"):
            context = f"replay row {position}/{arm}"
            replay_success = row.get(f"{arm}_replay_success")
            precision_reached = row.get(f"{arm}_precision_reached")
            converged = row.get(f"{arm}_mwle_converged")
            if not all(
                _is_exact_bool(value)
                for value in (replay_success, precision_reached, converged)
            ):
                raise FinalFitError(f"{context} flags are not exact booleans")
            replay_success = bool(replay_success)
            converged = bool(converged)
            arm_successes.append(replay_success)
            order = _parse_scenario_order(
                row.get(f"{arm}_scenario_order"), context=context
            )
            scenarios = row.get(f"{arm}_scenarios_administered")
            criteria = row.get(f"{arm}_criteria_administered")
            theta = row.get(f"theta_{arm}_mwle")
            if replay_success:
                if (
                    not _is_exact_int(scenarios)
                    or not 1 <= int(scenarios) <= int(maximum_scenarios)
                    or len(order) != int(scenarios)
                    or not set(order) <= allowed_scenarios
                ):
                    raise FinalFitError(f"{context} has invalid scenario administration")
                if not _is_exact_int(criteria) or int(criteria) < 1:
                    raise FinalFitError(f"{context} has invalid criterion count")
                if not isinstance(row.get(f"{arm}_stop_reason"), str) or not row.get(
                    f"{arm}_stop_reason"
                ):
                    raise FinalFitError(f"{context} has no stop reason")
            elif scenarios is not None or criteria is not None or order:
                raise FinalFitError(f"{context} failure contains administration results")
            if not replay_success and (bool(precision_reached) or converged):
                raise FinalFitError(f"{context} failure contains successful flags")
            if converged:
                if not replay_success:
                    raise FinalFitError(f"{context} has an invalid converged theta")
                if status == "ok" and not _finite_number(theta):
                    raise FinalFitError(f"{context} has an invalid converged theta")
                if status == "replay_error" and theta is not None:
                    raise FinalFitError(f"{context} error exposes a discarded theta")
            elif theta is not None:
                raise FinalFitError(f"{context} has theta without MWLE convergence")

            n_cells = row.get(f"{arm}_eval_n_cells")
            correct = row.get(f"{arm}_eval_correct_count")
            if (
                not _is_exact_int(n_cells)
                or int(n_cells) < 0
                or not _is_exact_int(correct)
                or not 0 <= int(correct) <= int(n_cells)
            ):
                raise FinalFitError(f"{context} has invalid evaluation counts")
            log_loss = row.get(f"{arm}_eval_log_loss_sum")
            brier = row.get(f"{arm}_eval_brier_sum")
            if (
                not _finite_number(log_loss)
                or float(log_loss) < 0
                or not _finite_number(brier)
                or float(brier) < 0
            ):
                raise FinalFitError(f"{context} has invalid evaluation losses")
            for field in ("observed_pass_rate", "predicted_pass_rate"):
                rate = row.get(f"{arm}_eval_{field}")
                if int(n_cells) == 0:
                    if rate is not None:
                        raise FinalFitError(f"{context} has a rate with zero cells")
                elif not _finite_number(rate) or not 0 <= float(rate) <= 1:
                    raise FinalFitError(f"{context} has invalid {field}")
            if status == "replay_error" and (
                int(n_cells) != 0
                or int(correct) != 0
                or not _same_number(log_loss, 0.0)
                or not _same_number(brier, 0.0)
                or row.get(f"{arm}_eval_observed_pass_rate") is not None
                or row.get(f"{arm}_eval_predicted_pass_rate") is not None
            ):
                raise FinalFitError(
                    f"{context} error contains nonempty evaluation statistics"
                )
            if float(brier) > float(n_cells):
                raise FinalFitError(f"{context} has invalid evaluation losses")

        if status == "ok":
            if error_text or arm_successes != [True, True]:
                raise FinalFitError(f"replay row {position} status contradicts arm results")
            if not _finite_number(row.get("theta_reference")):
                raise FinalFitError(f"replay row {position} has no finite reference theta")
            if row.get("theta_reference_scope") != "all_observed_administration_pool_items":
                raise FinalFitError(f"replay row {position} changed reference scope")
        else:
            if not error_text or all(arm_successes):
                raise FinalFitError(f"replay row {position} error status is contradictory")
            if row.get("theta_reference") is not None:
                raise FinalFitError(f"replay row {position} error contains reference theta")
        expected_rows.append(row)

    if (
        len(observed_keys) != len(expected_keys)
        or len(observed_keys) != len(set(observed_keys))
        or set(observed_keys) != expected_keys
    ):
        raise FinalFitError("replay rows do not cover the exact model/seed grid")
    return expected_rows


def replay_coverage(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_models: Sequence[str],
    expected_seeds: Sequence[int],
) -> dict[str, Any]:
    """Audit attempt coverage separately from observed replay outcomes."""

    expected = {
        (str(model), int(seed)) for model in expected_models for seed in expected_seeds
    }
    observed = [(str(row.get("model")), int(row.get("replay_seed", -1))) for row in rows]
    exact = len(observed) == len(expected) and len(set(observed)) == len(observed) and set(
        observed
    ) == expected
    successes = [
        row
        for row in rows
        if row.get("status") == "ok"
        and row.get("cat_replay_success") is True
        and row.get("baseline_replay_success") is True
    ]
    valid_cat_scores = [
        row
        for row in rows
        if row.get("status") == "ok"
        and row.get("cat_replay_success") is True
        and row.get("cat_mwle_converged") is True
        and _finite_number(row.get("theta_cat_mwle"))
    ]
    all_paired_successful = exact and len(successes) == len(expected)
    all_cat_scores_valid = exact and len(valid_cat_scores) == len(expected)
    replay_errors = [row for row in rows if row.get("status") != "ok"]
    return {
        "attempt_grid_complete": exact,
        "n_attempts_expected": len(expected),
        "n_attempts_observed": len(observed),
        "n_unique_attempt_keys": len(set(observed)),
        "n_status_ok_attempts": len(rows) - len(replay_errors),
        "n_replay_error_attempts": len(replay_errors),
        "n_paired_replay_successes": len(successes),
        "paired_replay_success_rate": len(successes) / len(expected) if expected else 0.0,
        "all_paired_replays_successful": all_paired_successful,
        "n_valid_cat_scores": len(valid_cat_scores),
        "all_cat_scores_valid": all_cat_scores_valid,
        "paired_evidence_available": all_paired_successful and all_cat_scores_valid,
    }


def scientific_replay_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Select successful paired attempts eligible for scientific aggregation."""

    eligible: list[dict[str, Any]] = []
    for row in rows:
        if (
            row.get("status") != "ok"
            or row.get("cat_replay_success") is not True
            or row.get("baseline_replay_success") is not True
        ):
            continue
        cat_length = row.get("cat_scenarios_administered")
        baseline_length = row.get("baseline_scenarios_administered")
        if (
            not _finite_number(cat_length)
            or float(cat_length) <= 0
            or not _finite_number(baseline_length)
            or float(baseline_length) <= 0
        ):
            continue
        eligible.append(dict(row))
    return eligible


def aggregate_paired_lengths(
    rows: Sequence[Mapping[str, Any]],
    model_to_family: Mapping[str, str],
    *,
    seed: int,
    bootstrap_replicates: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Aggregate seeds within tutor first, then infer over tutor families."""

    frame = pd.DataFrame(rows)
    required = {
        "model",
        "status",
        "replay_seed",
        "cat_replay_success",
        "baseline_replay_success",
        "cat_scenarios_administered",
        "baseline_scenarios_administered",
    }
    if not required <= set(frame):
        raise FinalFitError("paired replay table lacks required length fields")
    valid = frame[
        frame["status"].astype(str).eq("ok")
        & frame["cat_replay_success"].map(_as_bool)
        & frame["baseline_replay_success"].map(_as_bool)
    ].copy()
    valid["cat_scenarios_administered"] = pd.to_numeric(
        valid["cat_scenarios_administered"], errors="coerce"
    )
    valid["baseline_scenarios_administered"] = pd.to_numeric(
        valid["baseline_scenarios_administered"], errors="coerce"
    )
    valid = valid.dropna(
        subset=["cat_scenarios_administered", "baseline_scenarios_administered"]
    )
    valid = valid[
        np.isfinite(valid["cat_scenarios_administered"].to_numpy(float))
        & np.isfinite(valid["baseline_scenarios_administered"].to_numpy(float))
        & (valid["cat_scenarios_administered"] > 0)
        & (valid["baseline_scenarios_administered"] > 0)
    ]
    aggregates: list[dict[str, Any]] = []
    all_models = sorted(map(str, model_to_family))
    for model in all_models:
        subset = valid[valid["model"].astype(str) == model]
        cat = subset["cat_scenarios_administered"].to_numpy(float)
        baseline = subset["baseline_scenarios_administered"].to_numpy(float)
        aggregates.append(
            {
                "model": model,
                "model_family": str(model_to_family[model]),
                "n_valid_seed_pairs": int(len(subset)),
                "mean_cat_scenarios": float(cat.mean()) if cat.size else None,
                "mean_random_scenarios": float(baseline.mean()) if baseline.size else None,
                "mean_paired_scenarios_saved": (
                    float(np.mean(baseline - cat)) if cat.size else None
                ),
            }
        )
    per_model = pd.DataFrame(aggregates).sort_values("model").reset_index(drop=True)
    complete = per_model[
        per_model["mean_cat_scenarios"].notna()
        & per_model["mean_random_scenarios"].notna()
    ]

    def statistic(sample: pd.DataFrame) -> tuple[float, float]:
        cat_mean = float(sample["mean_cat_scenarios"].mean())
        random_mean = float(sample["mean_random_scenarios"].mean())
        reduction = 1.0 - cat_mean / random_mean if random_mean > 0 else math.nan
        return reduction, random_mean - cat_mean

    reduction, saved = statistic(complete) if len(complete) else (math.nan, math.nan)
    complete_models = set(complete["model"].astype(str))
    families = sorted(
        {
            str(family)
            for model, family in model_to_family.items()
            if str(model) in complete_models
        }
    )
    members = {
        family: sorted(
            str(model)
            for model, value in model_to_family.items()
            if str(value) == family and str(model) in complete_models
        )
        for family in families
    }
    indexed = complete.set_index(complete["model"].astype(str), drop=False)
    rng = np.random.default_rng(seed)
    reductions: list[float] = []
    savings: list[float] = []
    if len(families) >= 2 and bootstrap_replicates >= 2:
        for _ in range(bootstrap_replicates):
            choices = rng.integers(0, len(families), size=len(families))
            models = [
                model
                for choice in choices
                for model in members[families[int(choice)]]
                if model in indexed.index
            ]
            if not models:
                continue
            sample = indexed.loc[models]
            sample_reduction, sample_saved = statistic(sample)
            if math.isfinite(sample_reduction):
                reductions.append(sample_reduction)
            if math.isfinite(sample_saved):
                savings.append(sample_saved)

    def interval(values: Sequence[float]) -> tuple[float | None, float | None]:
        if not values:
            return None, None
        array = np.asarray(values, dtype=float)
        return float(np.percentile(array, 2.5)), float(np.percentile(array, 97.5))

    reduction_ci = interval(reductions)
    saved_ci = interval(savings)
    summary = {
        "n_attempt_rows": int(len(frame)),
        "n_replay_error_rows": int((frame["status"].astype(str) != "ok").sum()),
        "n_scientific_length_rows": int(len(valid)),
        "scientific_row_filter": (
            "status == 'ok' and both paired replays succeeded with finite positive lengths"
        ),
        "n_unique_models_expected": len(all_models),
        "n_unique_models_with_valid_pair": int(len(complete)),
        "n_families": len(families),
        "seed_rows_are_independent": False,
        "aggregation_order": "average seeds within tutor, then bootstrap tutor families",
        "bootstrap_unit": "tutor_family",
        "bootstrap_replicates": int(bootstrap_replicates),
        "mean_cat_scenarios": (
            float(complete["mean_cat_scenarios"].mean()) if len(complete) else None
        ),
        "mean_random_scenarios": (
            float(complete["mean_random_scenarios"].mean()) if len(complete) else None
        ),
        "scenario_reduction_vs_random": reduction if math.isfinite(reduction) else None,
        "scenario_reduction_lower_95_ci": reduction_ci[0],
        "scenario_reduction_upper_95_ci": reduction_ci[1],
        "mean_paired_scenarios_saved": saved if math.isfinite(saved) else None,
        "mean_paired_scenarios_saved_lower_95_ci": saved_ci[0],
        "mean_paired_scenarios_saved_upper_95_ci": saved_ci[1],
    }
    return per_model, summary


def validate_export_roundtrip(
    rubric_path: Path,
    scenario_path: Path,
    *,
    expected_bank: scat.FittedBank,
    expected_crosswalk: Mapping[str, str],
    expected_scenario_ids: Sequence[str],
) -> dict[str, Any]:
    """Reload exported files and prove their item parameters/crosswalk are exact."""

    try:
        loaded = scat.load_fitted_bank(
            rubric_path,
            skills=expected_bank.dims,
            negative_policy="error",
            require_calibrated=True,
        )
        scenario_rows = cm.cp.read_jsonl(scenario_path)
    except (OSError, ValueError, KeyError, scat.OfflineStudyError) as error:
        raise FinalFitError(f"could not reload exported final bank: {error}") from error

    if loaded.dims != expected_bank.dims:
        raise FinalFitError("export round-trip changed skill order")
    if loaded.criterion_ids != expected_bank.criterion_ids:
        raise FinalFitError("export round-trip changed criterion order")
    if loaded.scenario_ids != expected_bank.scenario_ids:
        raise FinalFitError("export round-trip changed rubric scenario IDs")
    for label, observed, expected in (
        ("Q", loaded.Q, expected_bank.Q),
        ("A", loaded.A, expected_bank.A),
        ("b", loaded.b, expected_bank.b),
        ("R", loaded.latent_correlation, expected_bank.latent_correlation),
    ):
        if not np.array_equal(np.asarray(observed), np.asarray(expected)):
            raise FinalFitError(f"export round-trip changed {label} parameters")

    observed_scenario_ids: list[str] = []
    observed_crosswalk: dict[str, str] = {}
    criterion_occurrences: list[str] = []
    for position, row in enumerate(scenario_rows, 1):
        if not isinstance(row, Mapping):
            raise FinalFitError(f"exported scenario row {position} is malformed")
        scenario_id = row.get("scenario_id")
        criterion_ids = row.get("criterion_ids")
        if (
            not isinstance(scenario_id, str)
            or not scenario_id
            or scenario_id in observed_scenario_ids
            or not isinstance(criterion_ids, list)
            or not criterion_ids
            or any(not isinstance(cid, str) or not cid for cid in criterion_ids)
        ):
            raise FinalFitError(f"exported scenario row {position} is invalid")
        observed_scenario_ids.append(scenario_id)
        for criterion_id in criterion_ids:
            if criterion_id in observed_crosswalk:
                raise FinalFitError(
                    f"exported criterion appears more than once: {criterion_id}"
                )
            observed_crosswalk[criterion_id] = scenario_id
            criterion_occurrences.append(criterion_id)

    normalized_expected_crosswalk = {
        str(criterion): str(scenario)
        for criterion, scenario in expected_crosswalk.items()
    }
    if tuple(observed_scenario_ids) != tuple(map(str, expected_scenario_ids)):
        raise FinalFitError("export round-trip changed scenario order")
    if observed_crosswalk != normalized_expected_crosswalk:
        raise FinalFitError("export round-trip changed criterion/scenario crosswalk")
    if len(criterion_occurrences) != expected_bank.n_items or set(
        observed_crosswalk
    ) != set(expected_bank.criterion_ids):
        raise FinalFitError("export round-trip criterion support is incomplete")

    return {
        "status": "passed",
        "exact_roundtrip": True,
        "n_items": expected_bank.n_items,
        "n_scenarios": len(observed_scenario_ids),
        "skills_order": list(expected_bank.dims),
        "criterion_order_sha256": _canonical_hash(expected_bank.criterion_ids),
        "scenario_order_sha256": _canonical_hash(observed_scenario_ids),
        "criterion_scenario_crosswalk_sha256": _canonical_hash(
            observed_crosswalk
        ),
        "Q_sha256": _canonical_hash(expected_bank.Q),
        "A_sha256": _canonical_hash(expected_bank.A),
        "b_sha256": _canonical_hash(expected_bank.b),
        "R_sha256": _canonical_hash(expected_bank.latent_correlation),
        "rubrics_sha256": _sha256(rubric_path),
        "scenarios_sha256": _sha256(scenario_path),
    }


class FinalFitRunner:
    """Strict terminal orchestrator for the frozen V3 study."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.runtime_axis = validate_exact_runtime_axis(args)
        self.config_path = args.config.resolve()
        self.config = _read_json(self.config_path)
        try:
            phase3._check_frozen_config(self.config)
            phase4.validate_frozen_phase4_design(self.config)
        except (phase3.V3Phase3Error, phase4.V3Phase4Error) as error:
            raise FinalFitError(str(error)) from error
        validate_frozen_scientific_contract(self.config)

        configured = self.config.get("outputs") or {}
        self.output_dir = validate_canonical_output_leaf(
            str(configured.get("final_fit") or ""), args.out_dir
        )
        self.phase3_dir = self._validate_upstream_leaf(
            "phase3", configured.get("phase3"), args.phase3_dir, CANONICAL_PHASE3_DIR
        )
        self.phase4_dir = self._validate_upstream_leaf(
            "phase4", configured.get("phase4"), args.phase4_dir, CANONICAL_PHASE4_DIR
        )
        if configured.get("never_overwrite_v1_or_v2") is not True:
            raise FinalFitError("historical V1/V2 output protection is not frozen")

        self.specs = phase3.load_calibration_specs(self.config)
        self.policies = phase3.load_policies(self.config)
        self.primary_policy = self.policies[0]
        if asdict(self.primary_policy) != {
            "policy_id": PRIMARY_POLICY_ID,
            "role": "primary",
            "minimum_scenarios": 15,
            "conditional_se_target": 0.2,
            "selector": "trace",
        }:
            raise FinalFitError("final stage is not frozen to floor15/SE.20/trace")

        try:
            self.numerical = phase3._validate_numerical_lock(
                self.config,
                self.specs,
                base_config_path=self.config_path,
                lock_path=args.numerical_lock.resolve(),
                followup_config_path=args.numerical_followup_config.resolve(),
                allow_pending=False,
                enforce_configured_paths=True,
            )
        except phase3.V3Phase3Error as error:
            raise FinalFitError(str(error)) from error
        self._validate_numerical_profile()

        self.phase3_manifest_path = self.phase3_dir / "manifest.json"
        self.phase3_decision_path = self.phase3_dir / "phase3_decision.json"
        self.selected_path = self.phase3_dir / "selected_calibration_specs.json"
        self.phase3_manifest = _read_json(self.phase3_manifest_path)
        self.phase3_decision = _read_json(self.phase3_decision_path)
        self.selected = _read_json(self.selected_path)
        try:
            phase4.validate_phase3_authorization(
                self.phase3_decision, self.selected, self.phase3_manifest
            )
            phase4.require_phase3_numerical_profile(
                self.phase3_manifest, self.numerical
            )
            self.phase3_artifacts = phase4.validate_phase3_artifact_inventory(
                self.phase3_dir, self.phase3_manifest
            )
        except (phase4.V3Phase4Error, phase4_engine.V2Phase4Error) as error:
            raise FinalFitError(str(error)) from error
        self.phase3_code_inventory = validate_recorded_code_inventory(
            "Phase 3", self.phase3_manifest.get("code_provenance") or {}
        )

        self.phase4_manifest_path = self.phase4_dir / "manifest.json"
        self.phase4_decision_path = self.phase4_dir / "phase4_decision.json"
        self.phase4_manifest = _read_json(self.phase4_manifest_path)
        self.phase4_decision = _read_json(self.phase4_decision_path)
        self.phase4_artifacts = validate_phase4_artifact_inventory(
            self.phase4_dir, self.phase4_manifest
        )
        self.phase4_code_inventory = validate_recorded_code_inventory(
            "Phase 4", self.phase4_manifest.get("code_provenance") or {}
        )
        self.phase4_gate_table = validate_phase4_decision(
            self.phase4_decision,
            self.phase4_manifest,
            phase4_dir=self.phase4_dir,
            phase3_manifest=self.phase3_manifest,
            config_path=self.config_path,
            phase3_manifest_path=self.phase3_manifest_path,
            phase3_decision_path=self.phase3_decision_path,
            selected_path=self.selected_path,
        )
        self.final_spec = select_authorized_final_spec(
            self.config,
            self.selected,
            self.phase3_decision,
            self.phase4_decision,
            self.phase3_manifest,
        )

        self._capture_provenance_sources()
        self._reload_authorization_under_capture()
        self._load_frozen_inputs()
        self._validate_runtime()
        self.code_hashes = {
            _display_path(path): self._captured_hash_path(path)
            for path in CODE_DEPENDENCIES
        }
        self.environment = _environment_provenance()
        self.git_commit = _git_commit()
        self.study_signature = self._study_signature()
        self._already_complete = False
        self._resume_manifest_snapshot: dict[str, Any] | None = None
        self._reverify_sources("end of initialization")

    @staticmethod
    def _validate_upstream_leaf(
        label: str,
        configured: Any,
        requested: Path,
        canonical: Path,
    ) -> Path:
        canonical_lexical = _lexical_absolute(canonical)
        configured_path = Path(str(configured or ""))
        if not configured_path.is_absolute():
            configured_path = ROOT / configured_path
        requested_path = requested if requested.is_absolute() else Path.cwd() / requested
        for source, path in (("configured", configured_path), ("requested", requested_path)):
            if (
                _lexical_absolute(path) != canonical_lexical
                or path.resolve(strict=False) != canonical_lexical
            ):
                raise FinalFitError(
                    f"{source} {label} path is not the canonical V3 {label} leaf"
                )
            _assert_no_symlink_components(path, root=ROOT)
        return canonical_lexical

    def _capture_provenance_sources(self) -> None:
        """Capture every frozen physical source once, with logical aliases."""

        paths: dict[str, Path] = {}
        aliases: dict[str, str] = {}
        physical_to_key: dict[Path, str] = {}
        expected_by_key: dict[str, str] = {}

        def add(logical: str, raw_path: Path, expected: str | None = None) -> None:
            if logical in aliases:
                raise FinalFitError(f"duplicate provenance logical key: {logical}")
            lexical = _lexical_absolute(raw_path)
            _assert_source_has_no_symlink_components(lexical)
            physical = lexical.resolve(strict=False)
            key = physical_to_key.get(physical)
            if key is None:
                key = logical
                physical_to_key[physical] = key
                paths[key] = lexical
            aliases[logical] = key
            if expected is not None:
                normalized = str(expected)
                prior = expected_by_key.get(key)
                if prior is not None and prior != normalized:
                    raise FinalFitError(
                        f"recorded provenance hashes disagree for {logical}"
                    )
                expected_by_key[key] = normalized

        add("config", self.config_path)
        add("numerical_followup_config", self.args.numerical_followup_config.resolve())
        numerical_lock_path = self.args.numerical_lock.resolve()
        add("numerical_lock", numerical_lock_path)
        add("numerical_lock_companion", numerical_lock_path.with_suffix(".sha256"))
        add(
            "numerical_followup_decision",
            numerical_lock_path.parent / "numerical_followup_decision.json",
        )
        add(
            "numerical_study_manifest",
            numerical_lock_path.parent / "study_manifest.json",
        )
        add("phase3_manifest", self.phase3_manifest_path)
        add("phase3_decision", self.phase3_decision_path)
        add("selected_calibration_specs", self.selected_path)
        add("phase4_manifest", self.phase4_manifest_path)
        add("phase4_decision", self.phase4_decision_path)

        for name in ("response_matrix", "rubrics", "scenarios", "judge_manifest"):
            path, expected = phase3._input_entry(self.config, name)
            add(f"input:{name}", path, expected)
        cv = self.config.get("cross_validation") or {}
        add(
            "split_manifest",
            _repo_path(str(cv.get("split_manifest") or "")),
            str(cv.get("split_manifest_sha256") or ""),
        )

        for name, digest in self.phase3_artifacts.items():
            add(f"phase3_artifact:{name}", self.phase3_dir / name, digest)
        for name, digest in self.phase4_artifacts.items():
            add(f"phase4_artifact:{name}", self.phase4_dir / name, digest)
        for name, digest in self.phase3_code_inventory.items():
            add(f"phase3_code:{name}", _repo_path(name), digest)
        for name, digest in self.phase4_code_inventory.items():
            add(f"phase4_code:{name}", _repo_path(name), digest)

        initial_direct_hashes = _code_hashes()
        for path in CODE_DEPENDENCIES:
            display = _display_path(path)
            add(f"direct_code:{display}", path, initial_direct_hashes[display])

        lock = _read_json(numerical_lock_path)
        evidence_hashes = lock.get("evidence_sha256") or {}
        evidence_paths = lock.get("evidence_paths") or {}
        if not isinstance(evidence_hashes, Mapping) or not isinstance(
            evidence_paths, Mapping
        ):
            raise FinalFitError("numerical lock lacks an evidence inventory")
        for raw_key, raw_digest in evidence_hashes.items():
            key_text = str(raw_key)
            relative = evidence_paths.get(key_text)
            if not isinstance(relative, str) or not relative:
                raise FinalFitError(f"numerical evidence path is missing: {key_text}")
            candidate = Path(relative)
            if candidate.is_absolute():
                evidence_path = candidate.resolve()
            elif relative.startswith("runs/"):
                evidence_path = _repo_path(relative)
            else:
                evidence_path = (numerical_lock_path.parent / candidate).resolve()
            try:
                evidence_path.relative_to(numerical_lock_path.parent.resolve())
            except ValueError as error:
                raise FinalFitError(
                    f"numerical evidence escapes the V4 leaf: {relative}"
                ) from error
            add(f"numerical_evidence:{key_text}", evidence_path, str(raw_digest))

        for prefix, directory in (
            ("numerical_artifact", numerical_lock_path.parent),
            ("phase3_file", self.phase3_dir),
            ("phase4_file", self.phase4_dir),
        ):
            for path in sorted(directory.rglob("*")):
                if path.is_file() or path.is_symlink():
                    add(f"{prefix}:{path.relative_to(directory)}", path)

        self.provenance = CapturedProvenance.capture(paths)
        self.provenance_keys = aliases
        self._provenance_path_keys = {
            str(_lexical_absolute(path)): key for key, path in paths.items()
        }
        for key, expected in expected_by_key.items():
            if self.provenance.sha256(key) != expected:
                raise FinalFitError(f"captured provenance differs from producer hash: {key}")
        self.provenance.reverify("post-capture authorization reload")

    def _captured_hash(self, logical: str) -> str:
        try:
            key = self.provenance_keys[logical]
        except KeyError as error:
            raise FinalFitError(f"uncaptured logical provenance source: {logical}") from error
        return self.provenance.sha256(key)

    def _captured_hash_path(self, path: Path) -> str:
        try:
            key = self._provenance_path_keys[str(_lexical_absolute(path))]
        except KeyError as error:
            raise FinalFitError(f"uncaptured physical provenance source: {path}") from error
        return self.provenance.sha256(key)

    def _reverify_sources(self, stage: str) -> None:
        self.provenance.reverify(stage)

    def _reload_authorization_under_capture(self) -> None:
        """Re-read all authorization objects after the immutable snapshot exists."""

        self._reverify_sources("authorization reload")
        config = _read_json(self.config_path)
        phase3_manifest = _read_json(self.phase3_manifest_path)
        phase3_decision = _read_json(self.phase3_decision_path)
        selected = _read_json(self.selected_path)
        phase4_manifest = _read_json(self.phase4_manifest_path)
        phase4_decision = _read_json(self.phase4_decision_path)
        stale = {
            "config": config != self.config,
            "phase3_manifest": phase3_manifest != self.phase3_manifest,
            "phase3_decision": phase3_decision != self.phase3_decision,
            "selected_calibration_specs": selected != self.selected,
            "phase4_manifest": phase4_manifest != self.phase4_manifest,
            "phase4_decision": phase4_decision != self.phase4_decision,
        }
        changed = sorted(name for name, value in stale.items() if value)
        if changed:
            raise FinalFitError(
                "authorization source changed between discovery and capture: "
                + ", ".join(changed)
            )
        try:
            phase3._check_frozen_config(config)
            phase4.validate_frozen_phase4_design(config)
            numerical = phase3._validate_numerical_lock(
                config,
                self.specs,
                base_config_path=self.config_path,
                lock_path=self.args.numerical_lock.resolve(),
                followup_config_path=self.args.numerical_followup_config.resolve(),
                allow_pending=False,
                enforce_configured_paths=True,
            )
            phase4.validate_phase3_authorization(
                phase3_decision, selected, phase3_manifest
            )
            phase4.require_phase3_numerical_profile(phase3_manifest, numerical)
            phase3_artifacts = phase4.validate_phase3_artifact_inventory(
                self.phase3_dir, phase3_manifest
            )
            phase4_artifacts = validate_phase4_artifact_inventory(
                self.phase4_dir, phase4_manifest
            )
        except (phase3.V3Phase3Error, phase4.V3Phase4Error, phase4_engine.V2Phase4Error) as error:
            raise FinalFitError(str(error)) from error
        if numerical != self.numerical:
            raise FinalFitError("numerical authorization changed during provenance capture")
        if phase3_artifacts != self.phase3_artifacts:
            raise FinalFitError("Phase-3 artifact inventory changed during capture")
        if phase4_artifacts != self.phase4_artifacts:
            raise FinalFitError("Phase-4 artifact inventory changed during capture")
        phase3_code = validate_recorded_code_inventory(
            "Phase 3", phase3_manifest.get("code_provenance") or {}
        )
        phase4_code = validate_recorded_code_inventory(
            "Phase 4", phase4_manifest.get("code_provenance") or {}
        )
        if phase3_code != self.phase3_code_inventory or phase4_code != self.phase4_code_inventory:
            raise FinalFitError("upstream code inventory changed during capture")
        gate_table = validate_phase4_decision(
            phase4_decision,
            phase4_manifest,
            phase4_dir=self.phase4_dir,
            phase3_manifest=phase3_manifest,
            config_path=self.config_path,
            phase3_manifest_path=self.phase3_manifest_path,
            phase3_decision_path=self.phase3_decision_path,
            selected_path=self.selected_path,
        )
        final_spec = select_authorized_final_spec(
            config,
            selected,
            phase3_decision,
            phase4_decision,
            phase3_manifest,
        )
        if final_spec != self.final_spec:
            raise FinalFitError("authorized final specification changed during capture")
        self.config = config
        self.numerical = numerical
        self.phase3_manifest = phase3_manifest
        self.phase3_decision = phase3_decision
        self.selected = selected
        self.phase4_manifest = phase4_manifest
        self.phase4_decision = phase4_decision
        self.phase3_artifacts = phase3_artifacts
        self.phase4_artifacts = phase4_artifacts
        self.phase4_gate_table = gate_table
        self._validate_numerical_profile()
        self._reverify_sources("completed authorization reload")

    def _validate_numerical_profile(self) -> None:
        expected = {
            "status": "passed",
            "fit_grid": 401,
            "fit_quadrature_method": cm.NORMAL_TRAPEZOID_QUADRATURE,
            "fit_linear_bound": 8.0,
            "fit_convergence_mode": cm.RETURNED_ITERATE_CONVERGENCE,
            "fit_max_iter": 1500,
            "fit_objective_tolerance": 1e-4,
            "fit_parameter_tolerance": 5e-5,
            "fit_consecutive_convergence_passes": 2,
            "eap_grid": 801,
            "quadrature_method": cm.NORMAL_TRAPEZOID_QUADRATURE,
            "linear_bound": 8.0,
        }
        for field, value in expected.items():
            observed = self.numerical.get(field)
            matches = (
                _same_number(observed, float(value))
                if isinstance(value, float)
                else observed == value
            )
            if not matches:
                raise FinalFitError(f"V4 numerical profile changed {field}")
        eligible = list(map(str, self.numerical.get("eligible_spec_ids") or []))
        if eligible != [spec.spec_id for spec in self.specs]:
            raise FinalFitError("V4 numerical lock eligibility changed")

    def _load_frozen_inputs(self) -> None:
        self._reverify_sources("frozen input load")
        paths: dict[str, Path] = {}
        hashes: dict[str, str] = {}
        for name in ("response_matrix", "rubrics", "scenarios", "judge_manifest"):
            try:
                path, expected = phase3._input_entry(self.config, name)
            except phase3.V3Phase3Error as error:
                raise FinalFitError(str(error)) from error
            if not path.is_file() or self._captured_hash_path(path) != expected:
                raise FinalFitError(f"frozen input failed hash validation: {name}")
            paths[name] = path
            hashes[name] = expected
        self.input_paths = paths
        self.input_hashes = hashes
        self.matrix = cm.load_matrix_strict(paths["response_matrix"])
        if len(self.matrix) != EXPECTED_MODELS:
            raise FinalFitError("final fit requires exactly all 52 frozen tutors")
        self.scenario_records = scat.load_scenario_records(paths["scenarios"])

        cv = self.config.get("cross_validation") or {}
        self.split_path = _repo_path(str(cv.get("split_manifest") or ""))
        expected_split = str(cv.get("split_manifest_sha256") or "")
        if (
            not self.split_path.is_file()
            or self._captured_hash_path(self.split_path) != expected_split
        ):
            raise FinalFitError("frozen split manifest failed hash validation")
        split = _read_json(self.split_path)
        try:
            self.split_audit = phase3.audit_repeated_split_manifest(
                split,
                expected_models=self.matrix.index,
                expected_scenarios=self.scenario_records,
            )
        except phase3.V3Phase3Error as error:
            raise FinalFitError(str(error)) from error
        if len(set(self.split_audit["model_to_family"].values())) != EXPECTED_FAMILIES:
            raise FinalFitError("final cohort must retain all 22 frozen tutor families")

        source_skills = cm.configure_skills(self.args.skills)
        if tuple(source_skills) != tuple(phase3.DEFAULT_SKILLS):
            raise FinalFitError("InFoBench source-skill order changed")
        self.structure = cell_cv.build_structure(
            tuple(source_skills), self.args.dimensions, self.args.structure_name
        )
        if self.structure.n_dims != 1 or tuple(self.structure.labels) != (
            "instruction_following",
        ):
            raise FinalFitError("final fit must remain one-dimensional instruction-following")
        expected_structure = cell_cv.build_structure(
            tuple(phase3.DEFAULT_SKILLS), EXACT_DIMENSIONS, EXACT_STRUCTURE_NAME
        )
        if self.structure.as_dict() != expected_structure.as_dict():
            raise FinalFitError("final InFoBench SkillStructure changed")
        self.structure_contract = self.structure.as_dict()
        self.structure_contract_sha256 = _canonical_hash(self.structure_contract)
        self.q_by = cm.load_q_matrix(paths["rubrics"])
        self.matrix_bank_alignment = cm.validate_matrix_bank_alignment(
            self.matrix, self.q_by, self.args.require_complete_bank
        )
        if self.matrix_bank_alignment.get("complete_bank_alignment") is not True:
            raise FinalFitError("final fit requires complete response/Q-bank alignment")
        self.source_records = scenario_cv.source_records_by_id(paths["rubrics"])
        prepared = cm.prepare_block(self.matrix, self.q_by)
        _Y, _M, q_source, prepared_items, _block, prepared_diag = prepared
        q_modeled = self.structure.transform_q(q_source)
        if q_modeled.shape != (len(prepared_items), 1) or not np.array_equal(
            q_modeled, np.ones_like(q_modeled)
        ):
            raise FinalFitError("prepared final Q-matrix is not exact overall-1D support")
        self.expected_fit_items = tuple(map(str, prepared_items))
        if (
            not self.expected_fit_items
            or len(self.expected_fit_items) != len(set(self.expected_fit_items))
        ):
            raise FinalFitError("prepared final item roster is empty or duplicated")
        self.expected_fit_items_sha256 = _canonical_hash(self.expected_fit_items)
        self.prepared_fit_diagnostics = prepared_diag
        self.expected_crosswalk = {
            criterion_id: str(self.source_records[criterion_id]["scenario_id"])
            for criterion_id in self.expected_fit_items
        }
        if len(self.expected_crosswalk) != len(self.expected_fit_items):
            raise FinalFitError("prepared criterion/scenario crosswalk is incomplete")
        self.expected_crosswalk_sha256 = _canonical_hash(self.expected_crosswalk)
        self._reverify_sources("completed frozen input load")

    def _validate_runtime(self) -> None:
        runtime = self.config.get("runtime") or {}
        cat = self.config.get("cat_policies") or {}
        uncertainty = self.config.get("uncertainty") or {}
        expected = {
            "seed": int(runtime.get("master_seed", -1)),
            "top_n": int(cat.get("top_n", -1)),
            "max_scenarios": int(cat.get("maximum_adaptive_scenarios", -1)),
            "minimum_scored_criteria": int(cat.get("minimum_scored_criteria", -1)),
            "mwle_ridge": float(runtime.get("mwle_ridge", math.nan)),
            "negative_policy": str(runtime.get("negative_loading_policy") or ""),
            "metric_bootstrap_replicates": int(
                runtime.get("metric_family_bootstrap_replicates", -1)
            ),
        }
        for field, value in expected.items():
            observed = getattr(self.args, field)
            matches = (
                _same_number(observed, value)
                if isinstance(value, float)
                else observed == value
            )
            if not matches:
                raise FinalFitError(f"runtime argument changed frozen {field}")
        try:
            self.paired_seeds = phase4_engine.validate_order_seeds(
                uncertainty.get("order_seeds") or []
            )
        except phase4_engine.V2Phase4Error as error:
            raise FinalFitError(str(error)) from error
        self.deployment_seed = int(runtime["master_seed"])

    def _study_signature(self) -> str:
        return _canonical_hash(
            {
                "schema_version": SCRIPT_SCHEMA,
                "captured_provenance_sha256": self.provenance.canonical_sha256,
                "config_sha256": self._captured_hash("config"),
                "phase3_artifacts": self.phase3_artifacts,
                "phase4_artifacts": self.phase4_artifacts,
                "phase3_code_inventory": self.phase3_code_inventory,
                "phase4_code_inventory": self.phase4_code_inventory,
                "numerical_lock": self.numerical,
                "exact_final_specification": self.final_spec.canonical,
                "primary_policy": asdict(self.primary_policy),
                "all_52_model_ids": sorted(map(str, self.matrix.index)),
                "model_to_family": self.split_audit["model_to_family"],
                "deployment_seed": self.deployment_seed,
                "paired_replay_seeds": list(self.paired_seeds),
                "runtime": {
                    "top_n": self.args.top_n,
                    "max_scenarios": self.args.max_scenarios,
                    "minimum_scored_criteria": self.args.minimum_scored_criteria,
                    "mwle_ridge": self.args.mwle_ridge,
                    "negative_policy": self.args.negative_policy,
                    "metric_bootstrap_replicates": self.args.metric_bootstrap_replicates,
                },
                "inputs": self.input_hashes,
                "split_sha256": self._captured_hash("split_manifest"),
                "runtime_axis": self.runtime_axis,
                "skill_structure": self.structure_contract,
                "skill_structure_sha256": self.structure_contract_sha256,
                "prepared_item_roster_sha256": self.expected_fit_items_sha256,
                "prepared_crosswalk_sha256": self.expected_crosswalk_sha256,
                "code_sha256": _canonical_hash(self.code_hashes),
                "environment_sha256": self.environment["canonical_sha256"],
            }
        )

    def _base_manifest(self, status: str) -> dict[str, Any]:
        return {
            "schema_version": SCRIPT_SCHEMA,
            "status": status,
            "created_at": _utcnow(),
            "study_signature": self.study_signature,
            "script": "scripts/finalize_infobench_calibration_cat_v3.py",
            "command": sys.argv,
            "git_commit": self.git_commit,
            "authorization": {
                "v4_numerical_lock_status": "passed",
                "phase3_status": "pass",
                "phase4_status": "pass",
                "final_fit_authorized": True,
                "fallback_allowed": False,
                "sensitivity_policy_promotion_allowed": False,
            },
            "inputs": {
                "config": {
                    "path": _display_path(self.config_path),
                    "sha256": self._captured_hash("config"),
                },
                "phase3_manifest": {
                    "path": _display_path(self.phase3_manifest_path),
                    "sha256": self._captured_hash("phase3_manifest"),
                },
                "phase3_decision": {
                    "path": _display_path(self.phase3_decision_path),
                    "sha256": self._captured_hash("phase3_decision"),
                },
                "selected_calibration_specs": {
                    "path": _display_path(self.selected_path),
                    "sha256": self._captured_hash("selected_calibration_specs"),
                },
                "phase4_manifest": {
                    "path": _display_path(self.phase4_manifest_path),
                    "sha256": self._captured_hash("phase4_manifest"),
                },
                "phase4_decision": {
                    "path": _display_path(self.phase4_decision_path),
                    "sha256": self._captured_hash("phase4_decision"),
                },
                "split_manifest": {
                    "path": _display_path(self.split_path),
                    "sha256": self._captured_hash("split_manifest"),
                },
                **{
                    name: {"path": _display_path(path), "sha256": self.input_hashes[name]}
                    for name, path in self.input_paths.items()
                },
            },
            "phase3_artifact_hashes": self.phase3_artifacts,
            "phase4_artifact_hashes": self.phase4_artifacts,
            "phase3_code_inventory": self.phase3_code_inventory,
            "phase4_code_inventory": self.phase4_code_inventory,
            "captured_provenance": self.provenance.as_dict(),
            "captured_provenance_aliases": dict(sorted(self.provenance_keys.items())),
            "v4_numerical_authorization": self.numerical,
            "exact_final_spec_id": self.final_spec.spec_id,
            "exact_final_calibration_specification": self.final_spec.canonical,
            "fit_population": {
                "n_tutors": EXPECTED_MODELS,
                "n_tutor_families": EXPECTED_FAMILIES,
                "model_ids": sorted(map(str, self.matrix.index)),
                "uses_all_frozen_tutors": True,
                "prepared_item_count": len(self.expected_fit_items),
                "prepared_item_roster_sha256": self.expected_fit_items_sha256,
                "criterion_scenario_crosswalk_sha256": self.expected_crosswalk_sha256,
            },
            "runtime_axis": self.runtime_axis,
            "skill_structure": self.structure_contract,
            "skill_structure_sha256": self.structure_contract_sha256,
            "primary_policy": asdict(self.primary_policy),
            "fit_profile": {
                "quadrature_method": cm.NORMAL_TRAPEZOID_QUADRATURE,
                "linear_bound": 8.0,
                "nodes": 401,
                "convergence_mode": cm.RETURNED_ITERATE_CONVERGENCE,
                "max_iter": 1500,
                "objective_tolerance": 1e-4,
                "parameter_tolerance": 5e-5,
                "consecutive_convergence_passes": 2,
                "maximum_inner_gradient": phase4.MAXIMUM_INNER_GRADIENT,
            },
            "scoring_profile": {
                "quadrature_method": cm.NORMAL_TRAPEZOID_QUADRATURE,
                "linear_bound": 8.0,
                "nodes": 801,
                "estimator": "mwle",
            },
            "replay_design": {
                "deployment_seed": self.deployment_seed,
                "paired_cat_random_seeds": list(self.paired_seeds),
                "paired_seed_derivation": (
                    "reuse the 20 prospectively frozen Phase-4 order seeds; no new "
                    "response-dependent seed selection"
                ),
                "same_seed_used_for_cat_and_random_within_pair": True,
                "seed_rows_treated_as_independent": False,
                "aggregate_seeds_within_tutor_before_family_bootstrap": True,
            },
            "code_provenance": {
                "files": self.code_hashes,
                "canonical_sha256": _canonical_hash(self.code_hashes),
            },
            "environment": self.environment,
            "limitations": [
                "final-bank replay is same-cohort operational evidence, not OOS recovery",
                "the 52-tutor cohort is internal development, not independent confirmation",
                "results are conditional on frozen Qwen labels not human-validated on InFoBench",
                "no unseen-family generalization claim is authorized",
            ],
        }

    def plan_payload(self) -> dict[str, Any]:
        self._reverify_sources("plan-only output")
        return {
            **self._base_manifest("plan_only"),
            "read_only": True,
            "selected_stable_modal_spec_id": self.final_spec.spec_id,
            "planned_fresh_final_fits": 1,
            "planned_deployment_model_replays": EXPECTED_MODELS,
            "planned_paired_seed_model_replays": EXPECTED_MODELS * len(self.paired_seeds),
            "each_model_replay_contains_cat_and_random_arms": True,
            "planned_outputs": list(FINAL_OUTPUTS),
            "leaderboard_generated_by_this_driver": False,
            "output_directory": _display_path(self.output_dir),
        }

    def _revalidate_output_leaf(self) -> None:
        configured = self.config.get("outputs") or {}
        self.output_dir = validate_canonical_output_leaf(
            str(configured.get("final_fit") or ""), self.output_dir
        )

    def _prepare_output(self) -> None:
        self._reverify_sources("output preparation")
        self._revalidate_output_leaf()
        manifest_path = self.output_dir / "manifest.json"
        if manifest_path.is_symlink():
            raise FinalFitError("final-fit top-level manifest is a symlink")
        if self.args.resume:
            if not manifest_path.is_file():
                raise FinalFitError("--resume requires an existing final-fit manifest")
            manifest = _read_json(manifest_path)
            if (
                manifest.get("schema_version") != SCRIPT_SCHEMA
                or manifest.get("study_signature") != self.study_signature
            ):
                raise FinalFitError("resume target is not this exact final-fit study")
            status = str(manifest.get("status") or "")
            if status == "final_fit_complete":
                self._reject_orphan_fit_staging()
                self._validate_fit_transaction()
                if self._load_final_fit() is None:
                    raise FinalFitError("completed run has no valid final-fit bundle")
                outputs = manifest.get("outputs")
                if not isinstance(outputs, Mapping) or set(map(str, outputs)) != set(
                    FINAL_OUTPUTS
                ):
                    raise FinalFitError("completed final output inventory is not exact")
                for name in FINAL_OUTPUTS:
                    raw = outputs[name]
                    path = self.output_dir / name
                    self._assert_output_artifact_path(path)
                    if (
                        not isinstance(raw, Mapping)
                        or raw.get("path") != name
                        or not path.is_file()
                        or raw.get("sha256") != _sha256(path)
                    ):
                        raise FinalFitError(f"completed final artifact failed hash: {name}")
                checkpoint_dir = self.output_dir / "checkpoints"
                history_dir = self.output_dir / "attempt_history"
                self._assert_output_artifact_path(checkpoint_dir)
                self._assert_output_artifact_path(history_dir)
                if manifest.get("checkpoint_tree_sha256") != _tree_hash(checkpoint_dir):
                    raise FinalFitError("completed final checkpoint tree changed")
                if manifest.get("attempt_history_tree_sha256") != _tree_hash(history_dir):
                    raise FinalFitError("completed final attempt history changed")
                decision_path = self.output_dir / "final_fit_decision.json"
                self._assert_output_artifact_path(decision_path)
                if manifest.get("decision") != _read_json(decision_path):
                    raise FinalFitError(
                        "completed final manifest and decision artifact disagree"
                    )
                self._already_complete = True
                return
            if status not in {"final_fit_running", "final_fit_aborted"}:
                raise FinalFitError(f"final-fit manifest is not resumable: {status}")
            self._resume_manifest_snapshot = manifest
        elif self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise FinalFitError(
                f"output directory is not empty: {self.output_dir}; use --resume for "
                "the exact interrupted study or choose a new versioned config/output leaf"
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._revalidate_output_leaf()
        self._reject_orphan_fit_staging()
        for legacy in ("final_fit_arrays.npz", "final_fit_manifest.json"):
            if (self.output_dir / legacy).exists():
                raise FinalFitError(f"legacy root-level final-fit cache is forbidden: {legacy}")
        self._reverify_sources("completed output preparation")

    def _fit_manifest_path(self) -> Path:
        return self.output_dir / FINAL_FIT_BUNDLE / "final_fit_manifest.json"

    def _fit_arrays_path(self) -> Path:
        return self.output_dir / FINAL_FIT_BUNDLE / "final_fit_arrays.npz"

    def _fit_transaction_path(self) -> Path:
        return self.output_dir / FINAL_FIT_BUNDLE / "TRANSACTION.json"

    def _orphan_fit_staging_dirs(self) -> list[Path]:
        if not self.output_dir.is_dir():
            return []
        return sorted(
            path
            for path in self.output_dir.iterdir()
            if path.name.startswith(f".{FINAL_FIT_BUNDLE}.staging-")
        )

    def _reject_orphan_fit_staging(self) -> None:
        orphaned = self._orphan_fit_staging_dirs()
        if orphaned:
            raise FinalFitError(
                "orphan final-fit staging directory is preserved for manual audit: "
                + ", ".join(map(str, orphaned))
            )

    def _validate_fit_transaction(self) -> dict[str, Any]:
        bundle = self.output_dir / FINAL_FIT_BUNDLE
        if bundle.is_symlink() or not bundle.is_dir():
            raise FinalFitError("final-fit bundle is missing, symlinked, or not a directory")
        expected_names = {
            "final_fit_arrays.npz",
            "final_fit_manifest.json",
            "TRANSACTION.json",
        }
        observed = {path.name for path in bundle.iterdir()}
        if observed != expected_names or any(
            path.is_symlink() or not path.is_file() for path in bundle.iterdir()
        ):
            raise FinalFitError("final-fit bundle file inventory is not exact")
        transaction = _read_json(self._fit_transaction_path())
        content_hash = str(transaction.pop("transaction_content_sha256", ""))
        if not content_hash or content_hash != _canonical_hash(transaction):
            raise FinalFitError("final-fit transaction marker content hash changed")
        expected = {
            "schema_version": FIT_TRANSACTION_SCHEMA,
            "status": "committed",
            "bundle_name": FINAL_FIT_BUNDLE,
            "study_signature": self.study_signature,
            "cache_key": self._fit_cache_key(),
        }
        for field, value in expected.items():
            if transaction.get(field) != value:
                raise FinalFitError(f"final-fit transaction changed {field}")
        files = transaction.get("files")
        if not isinstance(files, Mapping) or set(map(str, files)) != {
            "final_fit_arrays.npz",
            "final_fit_manifest.json",
        }:
            raise FinalFitError("final-fit transaction file inventory is invalid")
        for name in ("final_fit_arrays.npz", "final_fit_manifest.json"):
            raw = files[name]
            path = bundle / name
            if (
                not isinstance(raw, Mapping)
                or raw.get("path") != name
                or raw.get("sha256") != _sha256(path)
            ):
                raise FinalFitError(f"final-fit transaction failed hash: {name}")
        return transaction

    def _fit_bundle_file_hash(self, name: str) -> str:
        transaction = self._validate_fit_transaction()
        try:
            return str(transaction["files"][name]["sha256"])
        except (KeyError, TypeError) as error:
            raise FinalFitError(f"final-fit transaction lacks {name}") from error

    def _fit_cache_key(self) -> str:
        return _canonical_hash(
            {
                "schema_version": FIT_MANIFEST_SCHEMA,
                "study_signature": self.study_signature,
                "training_model_ids": sorted(map(str, self.matrix.index)),
                "exact_specification": self.final_spec.canonical,
                "fit_profile": self.numerical,
                "input_hashes": self.input_hashes,
                "captured_provenance_sha256": self.provenance.canonical_sha256,
                "runtime_axis": self.runtime_axis,
                "skill_structure_sha256": self.structure_contract_sha256,
                "prepared_item_roster_sha256": self.expected_fit_items_sha256,
                "prepared_crosswalk_sha256": self.expected_crosswalk_sha256,
                "code_sha256": _canonical_hash(self.code_hashes),
                "environment_sha256": self.environment["canonical_sha256"],
            }
        )

    def _assert_output_artifact_path(self, path: Path) -> None:
        self._revalidate_output_leaf()
        lexical = _lexical_absolute(path)
        try:
            lexical.relative_to(self.output_dir)
        except ValueError as error:
            raise FinalFitError(f"output artifact escapes canonical leaf: {path}") from error
        _assert_no_symlink_components(lexical, root=ROOT)
        if lexical.exists() and lexical.is_symlink():
            raise FinalFitError(f"output artifact is a symlink: {path}")

    def _write_or_verify_bytes(self, path: Path, payload: bytes) -> None:
        """Write a new artifact, or byte-verify it during an exact resume."""

        self._assert_output_artifact_path(path)
        if path.exists():
            if not self.args.resume:
                raise FinalFitError(f"refusing to overwrite existing artifact: {path}")
            if not path.is_file() or path.read_bytes() != payload:
                raise FinalFitError(f"resume artifact is not byte-identical: {path}")
            return
        _atomic_bytes(path, payload)

    def _write_or_verify_json(self, path: Path, value: Any) -> None:
        payload = (
            json.dumps(
                _json_ready(value), indent=2, sort_keys=True, ensure_ascii=False
            )
            + "\n"
        ).encode("utf-8")
        self._write_or_verify_bytes(path, payload)

    def _write_or_verify_csv(self, path: Path, frame: pd.DataFrame) -> None:
        self._write_or_verify_bytes(path, frame.to_csv(index=False).encode("utf-8"))

    def _write_or_verify_jsonl(
        self, path: Path, rows: Iterable[Mapping[str, Any]]
    ) -> None:
        payload = "".join(
            json.dumps(
                _json_ready(dict(row)),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
            for row in rows
        ).encode("utf-8")
        self._write_or_verify_bytes(path, payload)

    def _load_final_fit(self) -> dict[str, Any] | None:
        self._reverify_sources("final-fit cache load")
        self._reject_orphan_fit_staging()
        manifest_path = self._fit_manifest_path()
        arrays_path = self._fit_arrays_path()
        if not self.args.resume:
            return None
        bundle = self.output_dir / FINAL_FIT_BUNDLE
        if not bundle.exists():
            return None
        self._validate_fit_transaction()
        manifest = _read_json(manifest_path)
        content_hash = str(manifest.pop("manifest_content_sha256", ""))
        if not content_hash or content_hash != _canonical_hash(manifest):
            raise FinalFitError("final-fit manifest content hash changed")
        expected = {
            "schema_version": FIT_MANIFEST_SCHEMA,
            "study_signature": self.study_signature,
            "cache_key": self._fit_cache_key(),
            "role": "final_all52_fit",
            "training_model_ids": sorted(map(str, self.matrix.index)),
            "n_training_models": EXPECTED_MODELS,
            "prepared_item_roster_sha256": self.expected_fit_items_sha256,
            "prepared_crosswalk_sha256": self.expected_crosswalk_sha256,
            "items": list(self.expected_fit_items),
            "dim_labels": ["instruction_following"],
            "spec_id": self.final_spec.spec_id,
            "calibration_specification": self.final_spec.canonical,
            "input_hashes": self.input_hashes,
            "config_sha256": self._captured_hash("config"),
            "phase3_decision_sha256": self._captured_hash("phase3_decision"),
            "phase4_decision_sha256": self._captured_hash("phase4_decision"),
            "captured_provenance_sha256": self.provenance.canonical_sha256,
            "runtime_axis": self.runtime_axis,
            "skill_structure": self.structure_contract,
            "skill_structure_sha256": self.structure_contract_sha256,
            "numerical_lock": self.numerical,
            "code_sha256": _canonical_hash(self.code_hashes),
            "environment_sha256": self.environment["canonical_sha256"],
        }
        for field, value in expected.items():
            if manifest.get(field) != value:
                raise FinalFitError(f"final-fit cache changed {field}")
        if not _is_exact_int(manifest.get("n_training_models")):
            raise FinalFitError("final-fit training-model count is not an exact integer")
        if manifest.get("arrays_sha256") != _sha256(arrays_path):
            raise FinalFitError("final-fit arrays failed hash validation")
        try:
            with np.load(arrays_path, allow_pickle=False) as arrays:
                if set(arrays.files) != {"A", "b", "R"}:
                    raise FinalFitError("final-fit NPZ key inventory is not exact")
                fit = {
                    "items": manifest.get("items"),
                    "A": np.asarray(arrays["A"], dtype=float),
                    "b": np.asarray(arrays["b"], dtype=float),
                    "R": np.asarray(arrays["R"], dtype=float),
                    "dim_labels": manifest.get("dim_labels"),
                    "collapsed_labels": manifest.get("dim_labels"),
                    "loglik": manifest.get("loglik"),
                    "n_params": manifest.get("n_params"),
                    "n_iter": manifest.get("n_iter"),
                    "converged": manifest.get("converged"),
                    "grid_nodes": manifest.get("grid_nodes"),
                    "quadrature_method": manifest.get("quadrature_method"),
                    "quadrature_linear_bound": manifest.get(
                        "quadrature_linear_bound"
                    ),
                    "convergence_mode": manifest.get("convergence_mode"),
                    "penalized_objective": manifest.get("penalized_objective"),
                    "convergence_diagnostics": manifest.get(
                        "convergence_diagnostics"
                    ),
                    "calibration_specification": manifest.get(
                        "calibration_specification"
                    ),
                    "diag": manifest.get("diagnostics") or {},
                }
        except FinalFitError:
            raise
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise FinalFitError(f"could not load final-fit NPZ: {error}") from error
        validity = validate_final_fit_contract(
            fit, self.final_spec, self.expected_fit_items
        )
        if manifest.get("dense_fit_validity") != _json_ready(validity):
            raise FinalFitError("final-fit manifest validity audit changed")
        self._reverify_sources("completed final-fit cache load")
        return fit

    def _save_final_fit(
        self, fit: Mapping[str, Any], validity: Mapping[str, Any], bank_policy: Mapping[str, Any]
    ) -> None:
        self._reverify_sources("final-fit bundle publication")
        self._revalidate_output_leaf()
        self._reject_orphan_fit_staging()
        bundle_path = self.output_dir / FINAL_FIT_BUNDLE
        if bundle_path.exists():
            raise FinalFitError("refusing to overwrite a published final-fit bundle")
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{FINAL_FIT_BUNDLE}.staging-", dir=self.output_dir
            )
        )
        arrays_path = staging / "final_fit_arrays.npz"
        with arrays_path.open("xb") as handle:
            np.savez_compressed(
                handle,
                A=np.asarray(fit["A"], dtype=float),
                b=np.asarray(fit["b"], dtype=float),
                R=np.asarray(fit["R"], dtype=float),
            )
            handle.flush()
            os.fsync(handle.fileno())
        manifest = {
            "schema_version": FIT_MANIFEST_SCHEMA,
            "study_signature": self.study_signature,
            "cache_key": self._fit_cache_key(),
            "role": "final_all52_fit",
            "training_model_ids": sorted(map(str, self.matrix.index)),
            "n_training_models": EXPECTED_MODELS,
            "prepared_item_roster_sha256": self.expected_fit_items_sha256,
            "prepared_crosswalk_sha256": self.expected_crosswalk_sha256,
            "spec_id": self.final_spec.spec_id,
            "calibration_specification": self.final_spec.canonical,
            "input_hashes": self.input_hashes,
            "config_sha256": self._captured_hash("config"),
            "phase3_decision_sha256": self._captured_hash("phase3_decision"),
            "phase4_decision_sha256": self._captured_hash("phase4_decision"),
            "captured_provenance_sha256": self.provenance.canonical_sha256,
            "runtime_axis": self.runtime_axis,
            "skill_structure": self.structure_contract,
            "skill_structure_sha256": self.structure_contract_sha256,
            "numerical_lock": self.numerical,
            "code_sha256": _canonical_hash(self.code_hashes),
            "environment_sha256": self.environment["canonical_sha256"],
            "items": list(map(str, fit["items"])),
            "dim_labels": list(map(str, fit["dim_labels"])),
            "loglik": fit["loglik"],
            "n_params": fit["n_params"],
            "n_iter": fit["n_iter"],
            "converged": fit["converged"],
            "grid_nodes": fit.get("grid_nodes"),
            "quadrature_method": fit.get("quadrature_method"),
            "quadrature_linear_bound": fit.get("quadrature_linear_bound"),
            "convergence_mode": fit.get("convergence_mode"),
            "penalized_objective": fit.get("penalized_objective"),
            "convergence_diagnostics": fit.get("convergence_diagnostics"),
            "diagnostics": fit.get("diag") or {},
            "dense_fit_validity": dict(validity),
            "bank_policy": dict(bank_policy),
            "arrays_sha256": _sha256(arrays_path),
        }
        manifest["manifest_content_sha256"] = _canonical_hash(manifest)
        manifest_path = staging / "final_fit_manifest.json"
        _write_fsynced_json(manifest_path, manifest)
        transaction = {
            "schema_version": FIT_TRANSACTION_SCHEMA,
            "status": "committed",
            "bundle_name": FINAL_FIT_BUNDLE,
            "study_signature": self.study_signature,
            "cache_key": self._fit_cache_key(),
            "files": {
                "final_fit_arrays.npz": {
                    "path": "final_fit_arrays.npz",
                    "sha256": _sha256(arrays_path),
                },
                "final_fit_manifest.json": {
                    "path": "final_fit_manifest.json",
                    "sha256": _sha256(manifest_path),
                },
            },
        }
        transaction["transaction_content_sha256"] = _canonical_hash(transaction)
        _write_fsynced_json(staging / "TRANSACTION.json", transaction)
        _fsync_directory(staging)
        staging.replace(bundle_path)
        _fsync_directory(self.output_dir)
        self._validate_fit_transaction()
        self._reverify_sources("completed final-fit bundle publication")

    def _fit_all52(self) -> tuple[dict[str, Any], scat.FittedBank, scat.Quadrature, dict[str, Any]]:
        self._reverify_sources("before all-52 fit")
        fit = self._load_final_fit()
        loaded = fit is not None
        if not loaded:
            fit_args = argparse.Namespace(
                grid=401,
                estimate_latent_corr=False,
                ridge=0.0 if self.final_spec.ridge is None else float(self.final_spec.ridge),
                log_a_shrinkage=(
                    cm.DEFAULT_LOG_A_SHRINKAGE
                    if self.final_spec.log_a_shrinkage is None
                    else float(self.final_spec.log_a_shrinkage)
                ),
                calibration_model=self.final_spec.family,
                max_iter=1500,
                tol=1e-4,
                quadrature_method=cm.NORMAL_TRAPEZOID_QUADRATURE,
                linear_bound=8.0,
                convergence_mode=cm.RETURNED_ITERATE_CONVERGENCE,
                parameter_tol=5e-5,
                consecutive_convergence_passes=2,
            )
            fit = phase3._fit_structure_dense(
                self.matrix.loc[sorted(map(str, self.matrix.index))],
                self.q_by,
                fit_args,
                self.structure,
            )
            self._reverify_sources("after all-52 fit")
        validity = validate_final_fit_contract(
            fit, self.final_spec, self.expected_fit_items
        )
        bank, bank_policy = scenario_cv.build_fold_bank(
            fit,
            self.structure,
            self.source_records,
            negative_policy=self.args.negative_policy,
        )
        if (
            bank.n_items != len(fit["items"])
            or bank.dropped_negative_items
            or int(bank_policy.get("n_nonpositive_items", -1)) != 0
        ):
            raise FinalFitError(
                "positive final specification unexpectedly lost fitted items at export"
            )
        if not loaded:
            self._save_final_fit(fit, validity, bank_policy)
        self._validate_fit_transaction()
        quadrature = scat.build_quadrature(
            1,
            801,
            bank.latent_correlation,
            max_nodes=max(self.args.max_grid_nodes, 801),
            method=cm.NORMAL_TRAPEZOID_QUADRATURE,
            linear_bound=8.0,
        )
        if quadrature.nodes_per_dim != 801 or quadrature.method != cm.NORMAL_TRAPEZOID_QUADRATURE:
            raise FinalFitError("final replay did not construct the locked EAP-801 grid")
        return fit, bank, quadrature, dict(bank_policy)

    def _export_bank(
        self, fit: Mapping[str, Any], bank: scat.FittedBank, bank_policy: Mapping[str, Any]
    ) -> dict[str, Any]:
        self._reverify_sources("final-bank export")
        transaction = self._validate_fit_transaction()
        fit_manifest_hash = str(
            transaction["files"]["final_fit_manifest.json"]["sha256"]
        )
        arrays_hash = str(transaction["files"]["final_fit_arrays.npz"]["sha256"])
        records: list[dict[str, Any]] = []
        for source in bank.records:
            record = copy.deepcopy(source)
            irt = dict(record.get("irt_params") or {})
            irt.update(
                {
                    "source": FINAL_BANK_SOURCE,
                    "method": self.final_spec.family,
                    "calibrated": True,
                    "fitted": True,
                    "synthetic": False,
                    "skills_order": list(bank.dims),
                    "latent_correlation": bank.latent_correlation.tolist(),
                    "n_persons": EXPECTED_MODELS,
                    "calibration_specification": self.final_spec.canonical,
                    "provenance": {
                        "study_signature": self.study_signature,
                        "final_fit_manifest": _display_path(self._fit_manifest_path()),
                        "final_fit_manifest_sha256": fit_manifest_hash,
                        "final_fit_arrays_sha256": arrays_hash,
                        "phase3_decision_sha256": self._captured_hash(
                            "phase3_decision"
                        ),
                        "phase4_decision_sha256": self._captured_hash(
                            "phase4_decision"
                        ),
                        "response_matrix_sha256": self.input_hashes["response_matrix"],
                        "rubrics_sha256": self.input_hashes["rubrics"],
                        "scenarios_sha256": self.input_hashes["scenarios"],
                        "judge_manifest_sha256": self.input_hashes["judge_manifest"],
                        "training_model_ids_sha256": _canonical_hash(
                            sorted(map(str, self.matrix.index))
                        ),
                        "conditional_on_frozen_qwen_labels": True,
                    },
                }
            )
            record["irt_params"] = irt
            record["calibration_version"] = FINAL_BANK_SOURCE
            records.append(record)
        rubric_path = self.output_dir / "final_rubrics_fitted.jsonl"
        self._write_or_verify_jsonl(rubric_path, records)

        source_scenarios = cm.cp.read_jsonl(self.input_paths["scenarios"])
        fitted_ids = set(map(str, bank.criterion_ids))
        scenarios: list[dict[str, Any]] = []
        referenced: list[str] = []
        for source in source_scenarios:
            kept = [str(cid) for cid in source.get("criterion_ids") or [] if str(cid) in fitted_ids]
            if not kept:
                continue
            scenario = copy.deepcopy(source)
            scenario["criterion_ids"] = kept
            scenarios.append(scenario)
            referenced.extend(kept)
        if len(referenced) != len(set(referenced)) or set(referenced) != fitted_ids:
            raise FinalFitError("exported scenario/rubric criterion support differs")
        scenario_path = self.output_dir / "final_scenarios_fitted.jsonl"
        self._write_or_verify_jsonl(scenario_path, scenarios)

        item_frame = scenario_cv.fold_item_frame(bank)
        item_frame["spec_id"] = self.final_spec.spec_id
        item_frame["n_training_models"] = EXPECTED_MODELS
        item_frame["study_signature"] = self.study_signature
        self._write_or_verify_csv(
            self.output_dir / "final_item_parameters.csv", item_frame
        )

        roundtrip = validate_export_roundtrip(
            rubric_path,
            scenario_path,
            expected_bank=bank,
            expected_crosswalk=self.expected_crosswalk,
            expected_scenario_ids=[
                str(row["scenario_id"])
                for row in source_scenarios
                if any(str(cid) in fitted_ids for cid in row.get("criterion_ids") or [])
            ],
        )

        manifest = {
            "schema_version": EXPORT_MANIFEST_SCHEMA,
            "status": "complete",
            "study_signature": self.study_signature,
            "source": FINAL_BANK_SOURCE,
            "spec_id": self.final_spec.spec_id,
            "calibration_specification": self.final_spec.canonical,
            "n_training_models": EXPECTED_MODELS,
            "n_training_families": EXPECTED_FAMILIES,
            "n_fitted_items": bank.n_items,
            "n_exported_items": len(records),
            "n_exported_scenarios": len(scenarios),
            "skills_order": list(bank.dims),
            "latent_correlation": bank.latent_correlation.tolist(),
            "bank_policy": dict(bank_policy),
            "roundtrip_validation": roundtrip,
            "fit": {
                "manifest": _display_path(self._fit_manifest_path()),
                "manifest_sha256": fit_manifest_hash,
                "arrays_sha256": arrays_hash,
                "dense_fit_valid": True,
            },
            "outputs": {
                "rubrics": {
                    "path": _display_path(rubric_path),
                    "sha256": _sha256(rubric_path),
                },
                "scenarios": {
                    "path": _display_path(scenario_path),
                    "sha256": _sha256(scenario_path),
                },
                "item_parameters": {
                    "path": _display_path(self.output_dir / "final_item_parameters.csv"),
                    "sha256": _sha256(self.output_dir / "final_item_parameters.csv"),
                },
            },
            "source_inputs": self.input_hashes,
            "limitations": [
                "parameters were fitted on the same 52 tutors used for deployment replay",
                "criterion outcomes are frozen Qwen labels without an InFoBench human audit",
            ],
        }
        self._write_or_verify_json(
            self.output_dir / "final_bank_export_manifest.json", manifest
        )
        self._reverify_sources("completed final-bank export")
        return manifest

    def _checkpoint_path(self, label: str) -> Path:
        return self.output_dir / "checkpoints" / f"{label}.json"

    def _validated_replay_rows(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        seed: int,
        scope: str,
    ) -> list[dict[str, Any]]:
        return validate_replay_rows(
            rows,
            expected_model_to_family=self.split_audit["model_to_family"],
            expected_seeds=[seed],
            expected_scope=scope,
            expected_policy=self.primary_policy,
            expected_spec_id=self.final_spec.spec_id,
            expected_cache_key=self._fit_cache_key(),
            administration_scenario_ids=self.split_audit[
                "administration_scenario_ids"
            ],
            maximum_scenarios=self.args.max_scenarios,
        )

    def _load_replay_checkpoint(
        self, label: str, *, seed: int, scope: str
    ) -> list[dict[str, Any]] | None:
        self._reverify_sources(f"replay checkpoint load {label}")
        path = self._checkpoint_path(label)
        self._assert_output_artifact_path(path)
        if not (self.args.resume and path.is_file()):
            return None
        payload = _read_json(path)
        content_hash = str(payload.pop("checkpoint_content_sha256", ""))
        if not content_hash or content_hash != _canonical_hash(payload):
            raise FinalFitError(f"replay checkpoint content changed: {label}")
        expected = {
            "schema_version": CHECKPOINT_SCHEMA,
            "study_signature": self.study_signature,
            "scope": scope,
            "seed": int(seed),
            "expected_models": sorted(map(str, self.matrix.index)),
            "final_fit_manifest_sha256": self._fit_bundle_file_hash(
                "final_fit_manifest.json"
            ),
            "primary_policy": asdict(self.primary_policy),
        }
        for field, value in expected.items():
            if payload.get(field) != value:
                raise FinalFitError(f"replay checkpoint changed {field}: {label}")
        rows = payload.get("rows")
        if not isinstance(rows, list):
            raise FinalFitError(f"replay checkpoint has no rows: {label}")
        validated = self._validated_replay_rows(rows, seed=seed, scope=scope)
        coverage = replay_coverage(
            validated,
            expected_models=list(map(str, self.matrix.index)),
            expected_seeds=[seed],
        )
        if coverage["attempt_grid_complete"] is not True:
            raise FinalFitError(f"replay checkpoint coverage is incomplete: {label}")
        self._reverify_sources(f"completed replay checkpoint load {label}")
        return validated

    def _save_replay_checkpoint(
        self, label: str, *, seed: int, scope: str, rows: Sequence[Mapping[str, Any]]
    ) -> None:
        self._reverify_sources(f"replay checkpoint save {label}")
        self._assert_output_artifact_path(self._checkpoint_path(label))
        validated = self._validated_replay_rows(rows, seed=seed, scope=scope)
        payload = {
            "schema_version": CHECKPOINT_SCHEMA,
            "study_signature": self.study_signature,
            "scope": scope,
            "seed": int(seed),
            "expected_models": sorted(map(str, self.matrix.index)),
            "final_fit_manifest_sha256": self._fit_bundle_file_hash(
                "final_fit_manifest.json"
            ),
            "primary_policy": asdict(self.primary_policy),
            "rows": validated,
        }
        payload["checkpoint_content_sha256"] = _canonical_hash(payload)
        _atomic_json(self._checkpoint_path(label), payload)
        self._reverify_sources(f"completed replay checkpoint save {label}")

    def _run_seed(
        self,
        *,
        seed: int,
        scope: str,
        bank: scat.FittedBank,
        quadrature: scat.Quadrature,
    ) -> list[dict[str, Any]]:
        self._reverify_sources(f"before replay seed {seed}/{scope}")
        label = "deployment_master_seed" if scope == "deployment" else f"paired_seed_{seed}"
        cached = self._load_replay_checkpoint(label, seed=seed, scope=scope)
        if cached is not None:
            return cached
        administration_bank = phase3_v1.subset_fitted_bank(
            bank, self.split_audit["administration_scenario_ids"]
        )
        evaluation_indices = phase3_v1.evaluation_item_indices(
            bank, self.split_audit["evaluation_scenario_ids"]
        )
        rows: list[dict[str, Any]] = []
        model_to_family = self.split_audit["model_to_family"]
        for model in sorted(map(str, self.matrix.index)):
            row = phase3_v1.evaluate_model_pair(
                model=model,
                row=self.matrix.loc[model],
                full_bank=bank,
                administration_bank=administration_bank,
                evaluation_indices=evaluation_indices,
                scenario_records=self.scenario_records,
                quadrature=quadrature,
                candidate=self.primary_policy.as_candidate(),
                seed=int(seed),
                top_n=self.args.top_n,
                maximum_scenarios=self.args.max_scenarios,
                minimum_scored_criteria=self.args.minimum_scored_criteria,
                mwle_ridge=self.args.mwle_ridge,
            )
            row.update(
                {
                    "policy_id": PRIMARY_POLICY_ID,
                    "policy_role": "primary",
                    "replay_scope": scope,
                    "replay_seed": int(seed),
                    "model_family": model_to_family[model],
                    "spec_id": self.final_spec.spec_id,
                    "fit_cache_key": self._fit_cache_key(),
                    "final_fit_cache_key": self._fit_cache_key(),
                    "same_cohort_final_bank": True,
                    "out_of_sample": False,
                }
            )
            rows.append(row)
        self._reverify_sources(f"after replay seed {seed}/{scope}")
        rows = self._validated_replay_rows(rows, seed=seed, scope=scope)
        coverage = replay_coverage(
            rows,
            expected_models=list(map(str, self.matrix.index)),
            expected_seeds=[seed],
        )
        if coverage["attempt_grid_complete"] is not True:
            raise FinalFitError(f"seed {seed} replay did not record all 52 attempts")
        self._save_replay_checkpoint(label, seed=seed, scope=scope, rows=rows)
        return rows

    def _cross_fitted_length_evidence(self) -> dict[str, Any]:
        self._reverify_sources("cross-fitted evidence aggregation")
        path = self.phase3_dir / "outer_oof_per_model.csv"
        frame = pd.read_csv(path)
        primary = frame[frame["policy_id"].astype(str) == PRIMARY_POLICY_ID].copy()
        primary["repeat"] = pd.to_numeric(primary["repeat"], errors="coerce")
        keys = list(zip(primary["model"].astype(str), primary["repeat"], strict=False))
        expected = {
            (model, repeat)
            for model in map(str, self.matrix.index)
            for repeat in range(EXPECTED_REPEATS)
        }
        if len(keys) != len(expected) or len(set(keys)) != len(keys) or set(keys) != expected:
            raise FinalFitError("Phase-3 primary OOF rows do not cover 52 tutors x 5 repeats")
        normalized = primary.to_dict(orient="records")
        for row in normalized:
            row["replay_seed"] = int(row["repeat"])
        _per_model, length_summary = aggregate_paired_lengths(
            normalized,
            self.split_audit["model_to_family"],
            seed=self.deployment_seed,
            bootstrap_replicates=self.args.metric_bootstrap_replicates,
        )
        return {
            "evidence_type": "cross_fitted_proxy",
            "status": "available",
            "out_of_sample_within_each_repeat": True,
            "same_cohort_development": True,
            "n_unique_models": length_summary["n_unique_models_with_valid_pair"],
            "n_model_repeat_rows": int(len(primary)),
            "model_repeat_rows_treated_as_independent": False,
            "n_families": EXPECTED_FAMILIES,
            "mean_cat_scenarios": length_summary["mean_cat_scenarios"],
            "mean_random_scenarios": length_summary["mean_random_scenarios"],
            "scenario_reduction_vs_random": length_summary[
                "scenario_reduction_vs_random"
            ],
            "scenario_reduction_lower_95_ci": length_summary[
                "scenario_reduction_lower_95_ci"
            ],
            "scenario_reduction_upper_95_ci": length_summary[
                "scenario_reduction_upper_95_ci"
            ],
            "bootstrap_unit": "tutor_family_after_per_tutor_repeat_aggregation",
            "source_path": _display_path(path),
            "source_sha256": self._captured_hash_path(path),
            "interpretation": "headline internal OOF length/recovery proxy",
        }

    def _write_summary_markdown(
        self,
        decision: Mapping[str, Any],
        deployment: Mapping[str, Any],
        paired: Mapping[str, Any],
    ) -> None:
        text = f"""# InFoBench V3 final all-52 fit and replay

**Status:** {decision['status']}

The exact calibration specification authorized by repeated Phase 3 and passing
Phase 4 was **`{self.final_spec.spec_id}`**. It was refit once on all 52 frozen
tutor models using 401-node normal-trapezoid calibration and scored with the
frozen 801-node grid. The deployed primary policy remains floor 15, conditional
SE 0.20, and trace selection.

## Replay coverage

- Deployment replay: {deployment['n_attempts_observed']}/
  {deployment['n_attempts_expected']} recorded attempts.
- Paired 20-seed CAT/random panel: {paired['n_attempts_observed']}/
  {paired['n_attempts_expected']} recorded attempts.
- All deployment CAT scores valid: {decision['leaderboard']['all_52_deployment_scores_valid']}.

## Interpretation

The final-bank replay is a **same-cohort operational diagnostic** because the
same 52 tutors contributed to the fitted item bank. It is not the headline
out-of-sample recovery estimate. Cross-fitted Phase-3 results remain the honest
internal proxy for generalization. No unseen-family or independent-confirmation
claim is authorized.

All results are conditional on the frozen Qwen labels. Qwen was not independently
validated against humans on InFoBench. This driver emits no leaderboard; a later
exploratory same-cohort leaderboard is permitted only when the decision file says
all 52 deployment scores are valid and the above limitations remain visible.
"""
        self._write_or_verify_bytes(
            self.output_dir / "FINAL_FIT_SUMMARY.md", text.encode("utf-8")
        )

    def run(self) -> int:
        self._reverify_sources("run entry")
        if self.args.plan_only:
            print(json.dumps(_json_ready(self.plan_payload()), indent=2, sort_keys=True))
            return 0
        self._prepare_output()
        if self._already_complete:
            return 0
        if self._resume_manifest_snapshot is not None:
            snapshot_hash = _canonical_hash(self._resume_manifest_snapshot)
            snapshot_status = str(
                self._resume_manifest_snapshot.get("status") or "unknown"
            )
            history_path = (
                self.output_dir
                / "attempt_history"
                / f"{snapshot_status}__{snapshot_hash[:16]}.json"
            )
            self._write_or_verify_json(
                history_path, self._resume_manifest_snapshot
            )
        running = self._base_manifest("final_fit_running")
        self._assert_output_artifact_path(self.output_dir / "manifest.json")
        _atomic_json(self.output_dir / "manifest.json", running)
        try:
            fit, bank, quadrature, bank_policy = self._fit_all52()
            export = self._export_bank(fit, bank, bank_policy)

            deployment_rows = self._run_seed(
                seed=self.deployment_seed,
                scope="deployment",
                bank=bank,
                quadrature=quadrature,
            )
            deployment_frame = pd.DataFrame(deployment_rows).sort_values("model")
            self._write_or_verify_csv(
                self.output_dir / "deployment_replay_per_model.csv", deployment_frame
            )

            paired_rows: list[dict[str, Any]] = []
            for seed in self.paired_seeds:
                paired_rows.extend(
                    self._run_seed(
                        seed=seed,
                        scope="paired_cat_random",
                        bank=bank,
                        quadrature=quadrature,
                    )
                )
            paired_frame = pd.DataFrame(paired_rows).sort_values(
                ["replay_seed", "model"], kind="stable"
            )
            self._write_or_verify_csv(
                self.output_dir / "paired_seed_replays.csv", paired_frame
            )

            deployment_coverage = replay_coverage(
                deployment_rows,
                expected_models=list(map(str, self.matrix.index)),
                expected_seeds=[self.deployment_seed],
            )
            paired_coverage = replay_coverage(
                paired_rows,
                expected_models=list(map(str, self.matrix.index)),
                expected_seeds=list(self.paired_seeds),
            )
            self._reverify_sources("replay aggregation")
            paired_per_model, paired_summary = aggregate_paired_lengths(
                paired_rows,
                self.split_audit["model_to_family"],
                seed=self.deployment_seed,
                bootstrap_replicates=self.args.metric_bootstrap_replicates,
            )
            paired_eligibility_fields = {
                "n_attempt_rows",
                "n_replay_error_rows",
                "n_scientific_length_rows",
                "scientific_row_filter",
            }
            paired_eligibility = {
                key: paired_summary[key] for key in paired_eligibility_fields
            }
            paired_length_summary = {
                key: value
                for key, value in paired_summary.items()
                if key not in paired_eligibility_fields
            }
            self._write_or_verify_csv(
                self.output_dir / "paired_seed_per_model_aggregate.csv", paired_per_model
            )

            deployment_scientific_rows = scientific_replay_rows(deployment_rows)
            deployment_metrics = phase3.aggregate_policy_rows_clustered(
                deployment_scientific_rows,
                self.split_audit["model_to_family"],
                seed=self.deployment_seed,
                replicates=self.args.metric_bootstrap_replicates,
                label="same_cohort_final_bank_deployment",
            )
            deployment_metrics.pop("replay_success_rate", None)
            deployment_eligibility = {
                "scientific_row_filter": (
                    "status == 'ok' and both paired replays succeeded with finite "
                    "positive lengths; convergence/finite-theta rules then apply "
                    "inside each scientific metric"
                ),
                "n_attempt_rows": len(deployment_rows),
                "n_scientific_rows": len(deployment_scientific_rows),
                "n_replay_error_rows": sum(
                    row.get("status") != "ok" for row in deployment_rows
                ),
            }
            replay_summary = {
                "schema_version": SCRIPT_SCHEMA,
                "status": "complete",
                "study_signature": self.study_signature,
                "evidence_scope": "same_cohort_final_bank_operational_diagnostic",
                "out_of_sample": False,
                "deployment": {
                    "seed": self.deployment_seed,
                    "coverage": deployment_coverage,
                    "scientific_eligibility": deployment_eligibility,
                    "metrics": deployment_metrics,
                },
                "paired_cat_random": {
                    "status": (
                        "available"
                        if paired_coverage["paired_evidence_available"]
                        else "incomplete"
                    ),
                    "seeds": list(self.paired_seeds),
                    "coverage": paired_coverage,
                    "scientific_eligibility": paired_eligibility,
                    "length_summary": paired_length_summary,
                    "seed_rows_treated_as_independent": False,
                    "bootstrap_unit": "tutor_family_after_per_tutor_seed_aggregation",
                },
                "limitations": running["limitations"],
            }
            self._write_or_verify_json(
                self.output_dir / "final_replay_summary.json", replay_summary
            )

            cross_fitted = self._cross_fitted_length_evidence()
            paired_available = bool(paired_coverage["paired_evidence_available"])
            same_cohort = {
                "evidence_type": "same_cohort_final_bank",
                "status": "available" if paired_available else "incomplete",
                "out_of_sample_within_each_repeat": False,
                "same_cohort_development": True,
                "n_unique_models": paired_summary[
                    "n_unique_models_with_valid_pair"
                ],
                "n_model_repeat_rows": len(paired_rows),
                "model_repeat_rows_treated_as_independent": False,
                "n_families": EXPECTED_FAMILIES,
                "mean_cat_scenarios": paired_summary["mean_cat_scenarios"],
                "mean_random_scenarios": paired_summary["mean_random_scenarios"],
                "scenario_reduction_vs_random": paired_summary[
                    "scenario_reduction_vs_random"
                ],
                "scenario_reduction_lower_95_ci": paired_summary[
                    "scenario_reduction_lower_95_ci"
                ],
                "scenario_reduction_upper_95_ci": paired_summary[
                    "scenario_reduction_upper_95_ci"
                ],
                "bootstrap_unit": "tutor_family_after_per_tutor_seed_aggregation",
                "source_path": _display_path(
                    self.output_dir / "paired_seed_per_model_aggregate.csv"
                ),
                "source_sha256": _sha256(
                    self.output_dir / "paired_seed_per_model_aggregate.csv"
                ),
                "interpretation": (
                    "operational/deployment length diagnostic only"
                    if paired_available
                    else "incomplete operational diagnostic; one or more paired replays failed"
                ),
            }
            external = {
                "evidence_type": "external_frozen_bank",
                "status": "pending_new_untouched_cohort",
                "out_of_sample_within_each_repeat": None,
                "same_cohort_development": False,
                "n_unique_models": 0,
                "n_model_repeat_rows": 0,
                "model_repeat_rows_treated_as_independent": False,
                "n_families": 0,
                "mean_cat_scenarios": None,
                "mean_random_scenarios": None,
                "scenario_reduction_vs_random": None,
                "scenario_reduction_lower_95_ci": None,
                "scenario_reduction_upper_95_ci": None,
                "bootstrap_unit": None,
                "source_path": None,
                "source_sha256": None,
                "interpretation": "not available; no external cohort was collected",
            }
            self._write_or_verify_csv(
                self.output_dir / "cat_length_evidence.csv",
                pd.DataFrame([same_cohort, cross_fitted, external]),
            )

            self._reverify_sources("final decision aggregation")
            artifact_complete = bool(
                export.get("status") == "complete"
                and deployment_coverage["attempt_grid_complete"]
                and paired_coverage["attempt_grid_complete"]
            )
            all_52_scores_valid = bool(
                deployment_coverage["all_cat_scores_valid"]
                and deployment_coverage["n_valid_cat_scores"] == EXPECTED_MODELS
            )
            decision = {
                "schema_version": DECISION_SCHEMA,
                "status": "complete" if artifact_complete else "incomplete",
                "artifact_complete": artifact_complete,
                "upstream_gates_passed": True,
                "study_signature": self.study_signature,
                "spec_id": self.final_spec.spec_id,
                "calibration_specification": self.final_spec.canonical,
                "primary_policy": asdict(self.primary_policy),
                "fit": {
                    "all_52_models_used": True,
                    "dense_fit_contract_passed": True,
                    "n_fitted_items": bank.n_items,
                    "manifest_sha256": self._fit_bundle_file_hash(
                        "final_fit_manifest.json"
                    ),
                },
                "deployment_replay": deployment_coverage,
                "paired_seed_replay": paired_coverage,
                "replay_failures_are_reported_not_used_to_reselect_policy": True,
                "leaderboard": {
                    "generated_by_this_driver": False,
                    "all_52_deployment_scores_valid": all_52_scores_valid,
                    "downstream_exploratory_same_cohort_artifact_allowed": bool(
                        artifact_complete and all_52_scores_valid
                    ),
                    "generalization_or_external_leaderboard_claim_allowed": False,
                },
                "headline_generalization_source": _display_path(
                    self.phase3_dir / "outer_oof_per_model.csv"
                ),
                "final_bank_replay_role": "same_cohort_operational_diagnostic_only",
                "limitations": running["limitations"],
            }
            self._write_or_verify_json(
                self.output_dir / "final_fit_decision.json", decision
            )
            self._write_summary_markdown(
                decision, deployment_coverage, paired_coverage
            )
            if not artifact_complete:
                raise FinalFitError("final artifact/replay attempt coverage is incomplete")
        except Exception as error:
            failed = {
                **running,
                "status": "final_fit_aborted",
                "aborted_at": _utcnow(),
                "error": f"{type(error).__name__}: {error}",
                "leaderboard": {
                    "generated": False,
                    "authorized": False,
                    "reason": "terminal final-fit stage did not complete",
                },
            }
            self._assert_output_artifact_path(self.output_dir / "manifest.json")
            _atomic_json(self.output_dir / "manifest.json", failed)
            raise

        self._reverify_sources("terminal manifest publication")
        self._validate_fit_transaction()
        manifest = {
            **running,
            "status": "final_fit_complete",
            "completed_at": _utcnow(),
            "decision": decision,
            "outputs": {
                name: {"path": name, "sha256": _sha256(self.output_dir / name)}
                for name in FINAL_OUTPUTS
            },
            "checkpoint_tree_sha256": _tree_hash(self.output_dir / "checkpoints"),
            "attempt_history_tree_sha256": _tree_hash(
                self.output_dir / "attempt_history"
            ),
            "leaderboard_generated": False,
            "leaderboard_downstream_allowed": decision["leaderboard"][
                "downstream_exploratory_same_cohort_artifact_allowed"
            ],
        }
        self._assert_output_artifact_path(self.output_dir / "manifest.json")
        _atomic_json(self.output_dir / "manifest.json", manifest)
        return 0


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
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
        "--require-complete-bank",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    try:
        if args.resume and args.plan_only:
            raise FinalFitError("--resume and --plan-only cannot be combined")
        if args.max_grid_nodes < 801:
            raise FinalFitError("max-grid-nodes cannot hold the locked 801-node grid")
        if args.metric_bootstrap_replicates < 100:
            raise FinalFitError("family bootstrap requires at least 100 replicates")
        return FinalFitRunner(args).run()
    except (
        FinalFitError,
        phase4.V3Phase4Error,
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
