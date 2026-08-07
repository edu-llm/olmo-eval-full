"""Phase-4 fold-specific uncertainty and order stability for nested scenario CAT.

This command consumes a *completed* ``nested_scenario_cat_cv.py`` study.  For
each outer fold it uses the ridge and exact length-tied finalist panel selected
without the outer-test models, then:

* consumes the inner-only Phase-3 finalist shortlist for every outer fold;
* resamples only that fold's outer-training models;
* refits item parameters once per fold/bootstrap draw and reuses that fit across
  every finalist in the fold;
* reruns every finalist/outer-test CAT path under every bootstrap bank;
* combines conditional MWLE variance and across-bank theta variance with the
  law of total variance; and
* independently reruns the locked outer-fold bank under at least 20 CAT seeds.

No tutor or judge model is called.  ``--plan-only`` is read-only and validates
the prerequisite schema.  ``--fresh`` removes only Phase-4-owned outputs and
caches; it never deletes or rewrites Phase-3 artifacts.  ``--resume`` validates
the study signature and reuses atomic per-replicate/per-seed checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import shutil
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tutor_cat.skill_structure import SkillStructure  # noqa: E402

from scripts import kfold_cv_mirt as cell_cv  # noqa: E402
from scripts import nested_scenario_cat_cv as nested  # noqa: E402
from scripts import scenario_cat_lib as scat  # noqa: E402
from scripts import scenario_kfold_estimator_cv as scenario_cv  # noqa: E402
from scripts import scenario_order_experiment as order_exp  # noqa: E402
from scripts import scenario_param_uncertainty as param_uncertainty  # noqa: E402

cm = cell_cv.cm
SCRIPT_SCHEMA = "infobench-nested-cat-phase4-v2"
NESTED_SCHEMA = nested.SCRIPT_SCHEMA
FINALIST_SCHEMA = "infobench-phase3-finalist-shortlist-v2"
DEFAULT_NESTED_DIR = ROOT / "runs" / "calibration" / "InFoBench_remediation_v1" / "nested_cat_cv"
FINAL_OUTPUTS = (
    "outer_total_se.csv",
    "outer_order_stability.csv",
    "phase4_pareto.csv",
    "phase4_policy.json",
    "phase4_summary.json",
    "phase4_manifest.json",
)
OWNED_PATHS = FINAL_OUTPUTS + ("phase4_cache", "phase4_checkpoints")


class Phase4Error(RuntimeError):
    """Raised when Phase-4 provenance, leakage, or output invariants fail."""


class RecoverableNumericalFailure(RuntimeError):
    """A preregistered numerical draw failure that may be retained as an invalid draw."""


@dataclass(frozen=True)
class FoldPanel:
    outer_fold: int
    training_model_ids: tuple[str, ...]
    test_model_ids: tuple[str, ...]
    selected_ridge: float
    candidates: tuple[nested.Candidate, ...]
    fit_cache_key: str

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(candidate.candidate_id for candidate in self.candidates)


@dataclass
class Prerequisites:
    nested_dir: Path
    nested_manifest_path: Path
    nested_manifest: dict[str, Any]
    fold_assignments_path: Path
    fold_assignments: dict[str, Any]
    finalists_path: Path
    finalists_payload: dict[str, Any]
    inner_results_path: Path
    inner_results: pd.DataFrame
    outer_rows_path: Path
    outer_rows: pd.DataFrame
    config_path: Path
    config: dict[str, Any]
    matrix_path: Path
    matrix: pd.DataFrame
    rubrics_path: Path
    scenarios_path: Path
    scenario_records: dict[str, dict[str, Any]]
    administration_scenario_ids: tuple[str, ...]
    structure: SkillStructure
    q_by: dict[str, Any]
    source_records: dict[str, dict[str, Any]]
    fit_grid: int
    eap_grid: int
    quadrature_method: str
    linear_bound: float
    quadrature_axis_sha256: str | None
    quadrature_log_prior_sha256: str | None
    effective_node_count: int | None
    panels: tuple[FoldPanel, ...]
    input_hashes: dict[str, str]


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
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
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    return value


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        _json_ready(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Phase4Error(f"could not read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise Phase4Error(f"expected a JSON object in {path}")
    return value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(_json_ready(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _tree_inventory(path: Path) -> dict[str, str]:
    if not path.is_dir():
        raise Phase4Error(f"completed artifact directory is missing: {path}")
    return {
        str(item.relative_to(path)): _sha256(item)
        for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    }


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


def _phase4_code_dependency_hashes() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        Path(nested.__file__).resolve(),
        Path(scat.__file__).resolve(),
        Path(cell_cv.__file__).resolve(),
        Path(scenario_cv.__file__).resolve(),
        Path(order_exp.__file__).resolve(),
        Path(param_uncertainty.__file__).resolve(),
    )
    return {
        str(path.relative_to(ROOT)): _sha256(path)
        for path in sorted(set(paths), key=lambda item: str(item))
    }


def _phase4_dependency_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": importlib.metadata.version("numpy"),
        "pandas": importlib.metadata.version("pandas"),
        "scipy": importlib.metadata.version("scipy"),
    }


def _validate_nested_provenance(manifest: Mapping[str, Any]) -> None:
    signature = str(manifest.get("study_signature") or "")
    if len(signature) != 64:
        raise Phase4Error("nested manifest lacks a canonical study signature")
    code = manifest.get("code_provenance")
    if not isinstance(code, Mapping) or not isinstance(code.get("files"), Mapping):
        raise Phase4Error("nested manifest lacks code provenance")
    recorded_files = {str(key): str(value) for key, value in code["files"].items()}
    if code.get("canonical_sha256") != _canonical_hash(recorded_files):
        raise Phase4Error("nested manifest code-provenance signature does not reproduce")
    current_files = nested._code_dependency_hashes()
    if recorded_files != current_files:
        raise Phase4Error("nested Phase-3 code dependencies changed after the completed run")
    environment = manifest.get("environment")
    if not isinstance(environment, Mapping):
        raise Phase4Error("nested manifest lacks dependency provenance")
    current_versions = nested._dependency_versions()
    if any(environment.get(key) != value for key, value in current_versions.items()):
        raise Phase4Error("nested Phase-3 dependency versions changed after the completed run")
    if environment.get("canonical_dependency_versions_sha256") != _canonical_hash(
        current_versions
    ):
        raise Phase4Error("nested dependency-version signature does not reproduce")


def _structure_from_manifest(value: Mapping[str, Any]) -> SkillStructure:
    try:
        dimensions = [
            (str(item["label"]), tuple(map(str, item["members"]))) for item in value["dimensions"]
        ]
        return SkillStructure.from_groups(
            str(value["name"]), tuple(map(str, value["source_skills"])), dimensions
        )
    except (KeyError, TypeError, ValueError) as error:
        raise Phase4Error(f"invalid nested-CV skill structure: {error}") from error


def _verify_manifest_output(nested_dir: Path, manifest: Mapping[str, Any], name: str) -> Path:
    path = nested_dir / name
    if not path.is_file():
        raise Phase4Error(f"completed nested-CV prerequisite is missing {path}")
    output = (manifest.get("outputs") or {}).get(name)
    if not isinstance(output, Mapping) or not output.get("sha256"):
        raise Phase4Error(f"nested manifest does not provenance {name}")
    actual = _sha256(path)
    if actual != str(output["sha256"]):
        raise Phase4Error(
            f"nested output hash mismatch for {name}: expected {output['sha256']}, got {actual}"
        )
    return path


def _manifest_input_path(manifest: Mapping[str, Any], name: str) -> tuple[Path, str]:
    item = (manifest.get("inputs") or {}).get(name)
    if not isinstance(item, Mapping):
        raise Phase4Error(f"nested manifest is missing input provenance for {name}")
    path = _repo_path(str(item.get("path") or "")).resolve()
    expected = str(item.get("sha256") or "")
    if not path.is_file() or not expected:
        raise Phase4Error(f"nested input {name} is missing or un-hashed: {path}")
    actual = _sha256(path)
    if actual != expected:
        raise Phase4Error(
            f"nested input hash mismatch for {name}: expected {expected}, got {actual}"
        )
    return path, actual


def _candidate_lookup(inner: pd.DataFrame, outer_fold: int, candidate_id: str) -> nested.Candidate:
    required = {
        "outer_fold",
        "candidate_id",
        "minimum_scenarios",
        "conditional_se_target",
        "selector",
    }
    missing = sorted(required - set(inner.columns))
    if missing:
        raise Phase4Error(f"inner_candidate_results.csv lacks columns {missing}")
    rows = inner[
        (pd.to_numeric(inner["outer_fold"], errors="coerce") == outer_fold)
        & (inner["candidate_id"].astype(str) == candidate_id)
    ]
    if len(rows) != 1:
        raise Phase4Error(
            f"outer fold {outer_fold} has {len(rows)} rows for selected candidate {candidate_id}"
        )
    row = rows.iloc[0]
    candidate = nested.Candidate(
        minimum_scenarios=int(row["minimum_scenarios"]),
        conditional_se_target=float(row["conditional_se_target"]),
        selector=str(row["selector"]),
    )
    if candidate.candidate_id != candidate_id:
        raise Phase4Error(
            f"selected candidate fields do not reproduce ID {candidate_id}: "
            f"{candidate.candidate_id}"
        )
    return candidate


def _exact_numeric_equal(left: Any, right: Any) -> bool:
    """Compare frozen handoff numbers without inventing a tie tolerance."""

    try:
        return float(left) == float(right)
    except (TypeError, ValueError):
        return False


def _candidate_from_finalist(
    inner: pd.DataFrame,
    *,
    outer_fold: int,
    finalist: Mapping[str, Any],
    fit_cache_key: str,
) -> nested.Candidate:
    candidate_id = str(finalist.get("candidate_id") or "")
    if not candidate_id:
        raise Phase4Error(f"outer fold {outer_fold} has a finalist without a candidate_id")
    candidate = _candidate_lookup(inner, outer_fold, candidate_id)
    expected = {
        "minimum_scenarios": candidate.minimum_scenarios,
        "conditional_se_target": candidate.conditional_se_target,
        "selector": candidate.selector,
        "outer_training_fit_cache_key": fit_cache_key,
        "all_gates_pass": True,
        "phase3_length_filter_status": "retained_through_p90_and_mean",
        "deferred_selection_fields": ["lowest_total_uncertainty", "prefer_trace"],
    }
    for key, value in expected.items():
        if finalist.get(key) != value:
            raise Phase4Error(
                f"outer fold {outer_fold} finalist {candidate_id} has invalid {key}: "
                f"expected {value!r}, got {finalist.get(key)!r}"
            )
    inner_row = inner[
        (pd.to_numeric(inner["outer_fold"], errors="coerce") == outer_fold)
        & (inner["candidate_id"].astype(str) == candidate_id)
    ].iloc[0]
    if not bool(inner_row.get("all_gates_pass")):
        raise Phase4Error(
            f"outer fold {outer_fold} finalist {candidate_id} did not pass every Phase-3 gate"
        )
    for key in ("p90_scenario_count", "mean_scenario_count"):
        if not _exact_numeric_equal(finalist.get(key), inner_row.get(key)):
            raise Phase4Error(
                f"outer fold {outer_fold} finalist {candidate_id} changed frozen {key}"
            )
    return candidate


def assert_fit_boundary(
    training_model_ids: Sequence[str],
    test_model_ids: Sequence[str],
    sampled_model_ids: Sequence[str] | None = None,
) -> None:
    """Fail closed if a held-out model can enter an item-parameter fit."""

    training = set(map(str, training_model_ids))
    test = set(map(str, test_model_ids))
    if not training or not test:
        raise Phase4Error("outer training and test sets must both be non-empty")
    overlap = sorted(training & test)
    if overlap:
        raise Phase4Error(f"outer train/test leakage: {overlap[:10]}")
    if sampled_model_ids is not None:
        sampled = list(map(str, sampled_model_ids))
        leaked = sorted(set(sampled) & test)
        foreign = sorted(set(sampled) - training)
        if leaked:
            raise Phase4Error(f"outer-test IDs entered bootstrap fitting: {leaked[:10]}")
        if foreign:
            raise Phase4Error(
                f"bootstrap contains IDs outside its outer-training set: {foreign[:10]}"
            )
        if len(sampled) != len(training):
            raise Phase4Error("bootstrap must draw exactly one outer-training-set-sized sample")


def draw_outer_training_sample(
    training_model_ids: Sequence[str], test_model_ids: Sequence[str], *, seed: int
) -> tuple[str, ...]:
    """Draw one deterministic row bootstrap from outer-training IDs only."""

    training = tuple(sorted(map(str, training_model_ids)))
    test = tuple(sorted(map(str, test_model_ids)))
    assert_fit_boundary(training, test)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(training), size=len(training))
    sample = tuple(training[int(index)] for index in indices)
    assert_fit_boundary(training, test, sample)
    return sample


def _replicate_seed(master_seed: int, outer_fold: int, replicate: int) -> int:
    digest = hashlib.sha256(
        f"{master_seed}:outer={outer_fold}:replicate={replicate}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def validate_order_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(int(seed) for seed in seeds)
    if len(normalized) < 20:
        raise Phase4Error("order stability requires at least 20 deterministic seeds")
    if len(set(normalized)) != len(normalized):
        raise Phase4Error("order stability seeds must be unique")
    return normalized


def assert_balanced_panel_support(
    frame: pd.DataFrame,
    *,
    panels: Sequence[FoldPanel],
    index_column: str,
    index_values: Sequence[int],
) -> None:
    """Require one row on identical fold/model/index support for every candidate."""

    required = {"outer_fold", "candidate_id", "model", index_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise Phase4Error(f"panel rows lack support columns {missing}")
    expected = {
        (panel.outer_fold, candidate.candidate_id, model, int(index_value))
        for panel in panels
        for candidate in panel.candidates
        for model in panel.test_model_ids
        for index_value in index_values
    }
    actual = [
        (int(fold), str(candidate), str(model), int(index_value))
        for fold, candidate, model, index_value in zip(
            frame["outer_fold"],
            frame["candidate_id"],
            frame["model"],
            frame[index_column],
            strict=True,
        )
    ]
    if len(actual) != len(expected) or set(actual) != expected:
        raise Phase4Error(
            f"{index_column} rows do not provide unique, balanced "
            "candidate×fold×model support"
        )


def load_prerequisites(nested_dir: Path) -> Prerequisites:
    nested_dir = nested_dir.resolve()
    manifest_path = nested_dir / "manifest.json"
    if not manifest_path.is_file():
        raise Phase4Error(f"nested-CV prerequisite is absent: {manifest_path}. Run Phase 3 first.")
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != NESTED_SCHEMA:
        raise Phase4Error(
            f"expected nested schema {NESTED_SCHEMA!r}, got {manifest.get('schema_version')!r}"
        )
    if manifest.get("status") != "phase3_finalists_complete":
        raise Phase4Error(
            "nested-CV prerequisite status is "
            f"{manifest.get('status')!r}, not 'phase3_finalists_complete'"
        )
    if (manifest.get("cross_validation") or {}).get("outer_test_used_for_selection") is not False:
        raise Phase4Error("nested prerequisite does not prove outer-test isolation")
    _validate_nested_provenance(manifest)
    handoff_manifest = manifest.get("phase3_finalist_handoff")
    expected_manifest_handoff = {
        "artifact": "outer_fold_finalists.json",
        "artifact_schema_version": FINALIST_SCHEMA,
        "configured_selection_order": [
            "lowest_p90_scenario_count",
            "lowest_mean_scenario_count",
            "lowest_total_uncertainty",
            "prefer_trace",
        ],
        "applied_in_phase3": [
            "absolute_gates",
            "lowest_p90_scenario_count",
            "lowest_mean_scenario_count",
        ],
        "deferred_to_phase4": ["lowest_total_uncertainty", "prefer_trace"],
        "length_tie_policy": "exact_numeric_equality",
        "outer_evaluation_panel": (
            "each fold-local exact-tie finalist set, evaluated only on that "
            "fold's outer-test models"
        ),
        "cross_fold_candidate_id_comparison_authorized": False,
        "phase3_declares_winner": False,
        "final_policy_freeze_authorized": False,
    }
    if handoff_manifest != expected_manifest_handoff:
        raise Phase4Error("nested manifest does not carry the frozen fold-local handoff")
    if (manifest.get("ridge_selection") or {}).get("headline_eligible") is not True:
        raise Phase4Error("nested prerequisite used a debug ridge/numerical override")

    fold_path = _verify_manifest_output(nested_dir, manifest, "fold_assignments.json")
    finalists_path = _verify_manifest_output(nested_dir, manifest, "outer_fold_finalists.json")
    inner_path = _verify_manifest_output(nested_dir, manifest, "inner_candidate_results.csv")
    outer_path = _verify_manifest_output(nested_dir, manifest, "outer_oof_per_model.csv")
    folds = _read_json(fold_path)
    finalists_payload = _read_json(finalists_path)
    if folds.get("schema_version") != NESTED_SCHEMA:
        raise Phase4Error("fold_assignments.json has the wrong schema")
    if folds.get("leakage_audit") != "passed":
        raise Phase4Error("fold assignments do not contain a passed leakage audit")
    if finalists_payload.get("schema_version") != NESTED_SCHEMA:
        raise Phase4Error("outer_fold_finalists.json has the wrong nested schema")
    if finalists_payload.get("artifact_schema_version") != FINALIST_SCHEMA:
        raise Phase4Error("outer_fold_finalists.json has an unsupported artifact schema")
    expected_handoff = {
        "selection_source": "inner-fold metrics only",
        "outer_outcomes_used_for_shortlisting": False,
        "allow_fallback_if_none_pass": False,
        "configured_selection_order": [
            "lowest_p90_scenario_count",
            "lowest_mean_scenario_count",
            "lowest_total_uncertainty",
            "prefer_trace",
        ],
        "applied_in_phase3": [
            "absolute_gates",
            "lowest_p90_scenario_count",
            "lowest_mean_scenario_count",
        ],
        "deferred_to_phase4": ["lowest_total_uncertainty", "prefer_trace"],
        "length_tie_policy": "exact_numeric_equality",
        "phase3_declares_winner": False,
        "final_policy_freeze_authorized": False,
        "outer_evaluation_support": (
            "each fold-local finalist set on only that fold's outer-test models"
        ),
        "cross_fold_candidate_id_comparison_authorized": False,
    }
    for key, value in expected_handoff.items():
        if finalists_payload.get(key) != value:
            raise Phase4Error(
                f"outer finalist handoff has invalid {key}: expected {value!r}, "
                f"got {finalists_payload.get(key)!r}"
            )

    inner = pd.read_csv(inner_path)
    outer_rows = pd.read_csv(outer_path)
    required_outer_columns = {
        "outer_fold",
        "model",
        "candidate_id",
        "selected_ridge",
        "fit_cache_key",
        "cat_scenarios_administered",
        "theta_reference",
        "theta_cat_mwle",
    }
    missing_outer_columns = sorted(required_outer_columns - set(outer_rows.columns))
    if missing_outer_columns:
        raise Phase4Error(f"outer_oof_per_model.csv lacks required columns {missing_outer_columns}")
    config_path, config_hash = _manifest_input_path(manifest, "config")
    matrix_path, matrix_hash = _manifest_input_path(manifest, "response_matrix")
    rubrics_path, rubrics_hash = _manifest_input_path(manifest, "rubrics")
    scenarios_path, scenarios_hash = _manifest_input_path(manifest, "scenarios")
    config = _read_json(config_path)
    matrix = cm.load_matrix_strict(matrix_path)
    scenario_records = scat.load_scenario_records(scenarios_path)
    structure = _structure_from_manifest(manifest.get("structure") or {})
    if structure.n_dims != 1:
        raise Phase4Error("Phase-4 remediation is locked to the selected 1D structure")
    cm.configure_skills(",".join(structure.source_skills))
    q_by = cm.load_q_matrix(rubrics_path)
    cm.validate_matrix_bank_alignment(matrix, q_by, False)
    source_records = scenario_cv.source_records_by_id(rubrics_path)

    dense = manifest.get("dense_grid_lock") or {}
    try:
        fit_grid = int(dense["fit_grid"])
        eap_grid = int(dense["eap_grid"])
        quadrature_method = str(dense.get("quadrature_method", "gauss_hermite"))
        linear_bound = float(dense.get("linear_bound", 8.0))
        quadrature_axis_sha256 = (
            str(dense["quadrature_axis_sha256"])
            if dense.get("quadrature_axis_sha256")
            else None
        )
        quadrature_log_prior_sha256 = (
            str(dense["quadrature_log_prior_sha256"])
            if dense.get("quadrature_log_prior_sha256")
            else None
        )
        effective_node_count = (
            int(dense["effective_node_count"])
            if dense.get("effective_node_count") is not None
            else None
        )
    except (KeyError, TypeError, ValueError) as error:
        raise Phase4Error(f"nested manifest lacks locked dense grids: {error}") from error
    if fit_grid < 2 or eap_grid < 2:
        raise Phase4Error("nested manifest contains invalid dense-grid settings")
    if quadrature_method not in {"gauss_hermite", "normal_trapezoid"}:
        raise Phase4Error("nested manifest contains an invalid quadrature method")
    if not math.isfinite(linear_bound) or linear_bound <= 0:
        raise Phase4Error("nested manifest contains an invalid linear quadrature bound")
    if quadrature_method != "gauss_hermite":
        if not quadrature_axis_sha256 or not quadrature_log_prior_sha256:
            raise Phase4Error("nested manifest lacks locked quadrature hashes")
        locked_quadrature = scat.build_quadrature(
            1,
            eap_grid,
            np.eye(1),
            max_nodes=max(eap_grid, 50_000),
            method=quadrature_method,
            linear_bound=linear_bound,
        )
        if nested._array_sha256(locked_quadrature.grid[:, 0]) != quadrature_axis_sha256:
            raise Phase4Error("nested quadrature axis hash does not reproduce")
        if nested._array_sha256(locked_quadrature.log_prior) != quadrature_log_prior_sha256:
            raise Phase4Error("nested quadrature prior hash does not reproduce")
        if effective_node_count is not None and len(locked_quadrature.grid) != effective_node_count:
            raise Phase4Error("nested quadrature effective node count does not reproduce")

    fold_rows = folds.get("outer_folds") or []
    finalist_folds = finalists_payload.get("folds") or []
    if len(fold_rows) != 5 or len(finalist_folds) != 5:
        raise Phase4Error("Phase 4 requires exactly five completed outer folds")
    fold_by_id = {int(row["outer_fold"]): row for row in fold_rows}
    finalists_by_id = {int(row["outer_fold"]): row for row in finalist_folds}
    if len(fold_by_id) != 5 or len(finalists_by_id) != 5:
        raise Phase4Error("outer fold IDs must be unique")
    if set(fold_by_id) != set(range(5)) or set(finalists_by_id) != set(range(5)):
        raise Phase4Error("outer folds/finalist panels must be numbered exactly 0..4")

    all_test: set[str] = set()
    resolved_panels: list[FoldPanel] = []
    for fold in range(5):
        assignment = fold_by_id[fold]
        handoff = finalists_by_id[fold]
        train_ids = tuple(sorted(map(str, assignment.get("train_model_ids") or [])))
        test_ids = tuple(sorted(map(str, assignment.get("test_model_ids") or [])))
        assert_fit_boundary(train_ids, test_ids)
        if set(train_ids) | set(test_ids) != set(map(str, matrix.index)):
            raise Phase4Error(f"outer fold {fold} does not partition the response matrix")
        if all_test & set(test_ids):
            raise Phase4Error(f"outer fold {fold} repeats an outer-test model")
        all_test.update(test_ids)
        if handoff.get("shortlist_status") != "finalists_ready":
            raise Phase4Error(f"outer fold {fold} has no Phase-3 finalist panel")
        fold_policy = {
            "configured_selection_order": expected_handoff["configured_selection_order"],
            "applied_in_phase3": expected_handoff["applied_in_phase3"],
            "deferred_to_phase4": expected_handoff["deferred_to_phase4"],
            "phase3_declares_winner": False,
            "final_policy_freeze_authorized": False,
        }
        for key, value in fold_policy.items():
            if handoff.get(key) != value:
                raise Phase4Error(f"outer fold {fold} changed frozen handoff field {key}")
        if handoff.get("ridge_selected_before_outer_scoring") is not True:
            raise Phase4Error(f"outer fold {fold} ridge was not frozen before outer scoring")
        if handoff.get("outer_outcomes_available_at_shortlisting") is not False:
            raise Phase4Error(f"outer fold {fold} does not prove inner-only shortlisting")
        for key in ("ridge_evidence_sha256", "inner_metrics_sha256"):
            if len(str(handoff.get(key) or "")) != 64:
                raise Phase4Error(f"outer fold {fold} lacks {key} provenance")
        ridge = float(handoff["selected_ridge"])
        fit_cache_key = str(handoff.get("outer_training_fit_cache_key") or "")
        if not fit_cache_key:
            raise Phase4Error(f"outer fold {fold} finalist handoff lacks its fit cache key")
        finalist_rows = handoff.get("finalists") or []
        finalist_ids = list(map(str, handoff.get("finalist_candidate_ids") or []))
        if not finalist_rows or len(finalist_rows) != int(handoff.get("n_finalists", -1)):
            raise Phase4Error(f"outer fold {fold} has an invalid finalist count")
        row_ids = [str(row.get("candidate_id") or "") for row in finalist_rows]
        if finalist_ids != sorted(finalist_ids) or row_ids != finalist_ids:
            raise Phase4Error(
                f"outer fold {fold} finalist IDs must be unique, lexical, and row-aligned"
            )
        if len(set(finalist_ids)) != len(finalist_ids):
            raise Phase4Error(f"outer fold {fold} repeats a finalist candidate")
        if list(map(str, handoff.get("outer_evaluation_candidate_ids") or [])) != finalist_ids:
            raise Phase4Error(f"outer fold {fold} changed its local evaluation panel")
        if handoff.get("outer_evaluation_support") != (
            "fold_local_finalists_on_corresponding_outer_test_models_only"
        ):
            raise Phase4Error(f"outer fold {fold} lacks fold-local support provenance")
        local_candidates = tuple(
            _candidate_from_finalist(
                inner,
                outer_fold=fold,
                finalist=finalist,
                fit_cache_key=fit_cache_key,
            )
            for finalist in finalist_rows
        )
        if tuple(candidate.candidate_id for candidate in local_candidates) != tuple(finalist_ids):
            raise Phase4Error(f"outer fold {fold} local finalist records are not row-aligned")
        candidates = local_candidates
        fold_oof = outer_rows[pd.to_numeric(outer_rows.get("outer_fold"), errors="coerce") == fold]
        expected_pairs = {
            (candidate_id, model) for candidate_id in finalist_ids for model in test_ids
        }
        actual_pairs = list(
            zip(fold_oof["candidate_id"].astype(str), fold_oof["model"].astype(str), strict=True)
        )
        if len(actual_pairs) != len(expected_pairs) or set(actual_pairs) != expected_pairs:
            raise Phase4Error(
                f"outer fold {fold} OOF rows do not provide one row per finalist/test-model pair"
            )
        if "selected_ridge" not in fold_oof or not np.allclose(
            pd.to_numeric(fold_oof["selected_ridge"], errors="coerce"), ridge
        ):
            raise Phase4Error(f"outer fold {fold} OOF rows use a different ridge")
        fit_keys = set(fold_oof["fit_cache_key"].dropna().astype(str))
        if fit_keys != {fit_cache_key}:
            raise Phase4Error(f"outer fold {fold} finalists do not share the frozen fit cache")
        resolved_panels.append(
            FoldPanel(
                outer_fold=fold,
                training_model_ids=train_ids,
                test_model_ids=test_ids,
                selected_ridge=ridge,
                candidates=candidates,
                fit_cache_key=fit_cache_key,
            )
        )
    if all_test != set(map(str, matrix.index)):
        raise Phase4Error("outer-test folds do not cover every response-matrix model exactly once")

    scenario_split = folds.get("scenario_split") or {}
    administration = tuple(
        sorted(map(str, scenario_split.get("administration_scenario_ids") or []))
    )
    evaluation = set(map(str, scenario_split.get("evaluation_scenario_ids") or []))
    if not administration or set(administration) & evaluation:
        raise Phase4Error("invalid administration/evaluation scenario split")
    if set(administration) | evaluation != set(scenario_records):
        raise Phase4Error("scenario split does not cover the frozen scenario records")

    return Prerequisites(
        nested_dir=nested_dir,
        nested_manifest_path=manifest_path,
        nested_manifest=manifest,
        fold_assignments_path=fold_path,
        fold_assignments=folds,
        finalists_path=finalists_path,
        finalists_payload=finalists_payload,
        inner_results_path=inner_path,
        inner_results=inner,
        outer_rows_path=outer_path,
        outer_rows=outer_rows,
        config_path=config_path,
        config=config,
        matrix_path=matrix_path,
        matrix=matrix,
        rubrics_path=rubrics_path,
        scenarios_path=scenarios_path,
        scenario_records=scenario_records,
        administration_scenario_ids=administration,
        structure=structure,
        q_by=q_by,
        source_records=source_records,
        fit_grid=fit_grid,
        eap_grid=eap_grid,
        quadrature_method=quadrature_method,
        linear_bound=linear_bound,
        quadrature_axis_sha256=quadrature_axis_sha256,
        quadrature_log_prior_sha256=quadrature_log_prior_sha256,
        effective_node_count=effective_node_count,
        panels=tuple(resolved_panels),
        input_hashes={
            "config": config_hash,
            "response_matrix": matrix_hash,
            "rubrics": rubrics_hash,
            "scenarios": scenarios_hash,
        },
    )


def law_total_variance(
    draw_rows: pd.DataFrame,
    *,
    model_ids: Sequence[str],
    dimensions: Sequence[str],
    n_boot: int,
    minimum_valid_rate: float,
) -> pd.DataFrame:
    """Calculate E[conditional variance] + Var[conditional mean]."""

    output: list[dict[str, Any]] = []
    for model in sorted(map(str, model_ids)):
        model_rows = draw_rows[draw_rows["model"].astype(str) == model]
        for dim in dimensions:
            theta_source = model_rows.get(
                f"theta_mwle_{dim}", pd.Series(np.nan, index=model_rows.index)
            )
            se_source = model_rows.get(f"se_mwle_{dim}", pd.Series(np.nan, index=model_rows.index))
            theta = pd.to_numeric(theta_source, errors="coerce")
            se = pd.to_numeric(se_source, errors="coerce")
            converged = model_rows.get(
                "mwle_converged", pd.Series(False, index=model_rows.index)
            ).astype(bool)
            status_ok = (
                model_rows.get("status", pd.Series("", index=model_rows.index)).astype(str).eq("ok")
            )
            valid = status_ok & converged & theta.notna() & se.notna() & (se >= 0)
            theta_values = theta[valid].to_numpy(float)
            se_values = se[valid].to_numpy(float)
            conditional_variance = float(np.mean(np.square(se_values))) if se_values.size else None
            parameter_variance = (
                float(np.var(theta_values, ddof=1)) if theta_values.size >= 2 else None
            )
            total_variance = (
                conditional_variance + parameter_variance
                if conditional_variance is not None and parameter_variance is not None
                else None
            )
            valid_rate = theta_values.size / n_boot if n_boot else 0.0
            output.append(
                {
                    "model": model,
                    "dimension": str(dim),
                    "n_boot_requested": int(n_boot),
                    "n_valid_draws": int(theta_values.size),
                    "n_invalid_draws": int(n_boot - theta_values.size),
                    "valid_draw_rate": float(valid_rate),
                    "mean_theta": (float(theta_values.mean()) if theta_values.size else None),
                    "mean_conditional_variance": conditional_variance,
                    "parameter_variance": parameter_variance,
                    "total_variance": total_variance,
                    "se_ability": (
                        math.sqrt(conditional_variance)
                        if conditional_variance is not None
                        else None
                    ),
                    "se_param": (
                        math.sqrt(parameter_variance) if parameter_variance is not None else None
                    ),
                    "se_total": (math.sqrt(total_variance) if total_variance is not None else None),
                    "valid_draw_gate": bool(valid_rate >= minimum_valid_rate),
                }
            )
    return pd.DataFrame(output).sort_values(["model", "dimension"]).reset_index(drop=True)


def replay_model(
    *,
    model: str,
    row: pd.Series,
    bank: scat.FittedBank,
    scenario_records: Mapping[str, dict[str, Any]],
    quadrature: scat.Quadrature,
    candidate: nested.Candidate,
    seed: int,
    top_n: int,
    max_scenarios: int,
    minimum_scored_criteria: int,
    mwle_ridge: float,
) -> dict[str, Any]:
    """Rerun a CAT path under the supplied (possibly bootstrap) bank."""

    spec = scat.RunSpec(
        seed=int(seed),
        top_n=int(top_n),
        max_se=float(candidate.conditional_se_target),
        min_evals_per_skill=int(minimum_scored_criteria),
        min_scenarios=int(candidate.minimum_scenarios),
        max_scenarios=int(max_scenarios),
        selection=str(candidate.selector),
        mode="cat",
    )
    result = scat.run_recorded_model(
        str(model),
        row,
        bank,
        scenario_records,
        quadrature,
        spec,
        mwle_ridge=float(mwle_ridge),
    )
    if not set(result["scenario_order"]) <= set(bank.scenario_ids):
        raise Phase4Error(f"{model}: CAT replay left the administration bank")
    return result


def _load_fit_arrays(
    cache_dir: Path,
    *,
    expected_cache_key: str,
    expected_training_ids: Sequence[str],
    expected_fit_grid: int,
    expected_ridge: float,
) -> dict[str, Any]:
    manifest_path = cache_dir / "fit_manifest.json"
    arrays_path = cache_dir / "fit_arrays.npz"
    if not manifest_path.is_file() or not arrays_path.is_file():
        raise Phase4Error(f"completed nested fit cache is missing under {cache_dir}")
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != NESTED_SCHEMA:
        raise Phase4Error(f"outer fit-cache schema mismatch in {cache_dir}")
    if manifest.get("cache_key") != expected_cache_key:
        raise Phase4Error(f"outer fit-cache key mismatch in {cache_dir}")
    if manifest.get("training_model_ids") != sorted(map(str, expected_training_ids)):
        raise Phase4Error(f"outer fit-cache training IDs mismatch in {cache_dir}")
    if manifest.get("fit_grid") != int(expected_fit_grid) or not _exact_numeric_equal(
        manifest.get("ridge"), expected_ridge
    ):
        raise Phase4Error(f"outer fit-cache numerical settings mismatch in {cache_dir}")
    if manifest.get("arrays_sha256") != _sha256(arrays_path):
        raise Phase4Error(f"outer fit-cache array hash mismatch in {cache_dir}")
    with np.load(arrays_path, allow_pickle=False) as arrays:
        fit = {
            "items": list(map(str, manifest["items"])),
            "A": np.asarray(arrays["A"], dtype=float),
            "b": np.asarray(arrays["b"], dtype=float),
            "R": np.asarray(arrays["R"], dtype=float),
            "dim_labels": list(map(str, manifest["dim_labels"])),
            "loglik": float(manifest["loglik"]),
            "n_params": int(manifest["n_params"]),
            "n_iter": int(manifest["n_iter"]),
            "converged": bool(manifest["converged"]),
            "diag": manifest.get("diagnostics") or {},
        }
    _validate_fit_payload(fit, context=str(cache_dir))
    return fit


def _validate_fit_payload(fit: Mapping[str, Any], *, context: str) -> None:
    items = list(map(str, fit.get("items") or []))
    dims = list(map(str, fit.get("dim_labels") or []))
    if not items or len(items) != len(set(items)) or not dims or len(dims) != len(set(dims)):
        raise Phase4Error(f"fit cache has empty or duplicate item/dimension IDs: {context}")
    A = np.asarray(fit.get("A"), dtype=float)
    b = np.asarray(fit.get("b"), dtype=float)
    R = np.asarray(fit.get("R"), dtype=float)
    if A.shape != (len(items), len(dims)) or b.shape != (len(items),):
        raise Phase4Error(f"fit cache item-array shapes do not match provenance: {context}")
    if R.shape != (len(dims), len(dims)):
        raise Phase4Error(f"fit cache latent-correlation shape is invalid: {context}")
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)) or not np.all(np.isfinite(R)):
        raise Phase4Error(f"fit cache contains non-finite arrays: {context}")


def _bootstrap_cache_key(
    prereq: Prerequisites,
    panel: FoldPanel,
    replicate: int,
    sampled_ids: Sequence[str],
    args: argparse.Namespace,
) -> str:
    return _canonical_hash(
        {
            "schema": SCRIPT_SCHEMA,
            "nested_manifest_sha256": _sha256(prereq.nested_manifest_path),
            "outer_fold": panel.outer_fold,
            "replicate": replicate,
            "sampled_model_ids": list(sampled_ids),
            "outer_training_model_ids": list(panel.training_model_ids),
            "outer_test_model_ids": list(panel.test_model_ids),
            "fit_grid": prereq.fit_grid,
            "ridge": panel.selected_ridge,
            "max_iter": args.max_iter,
            "tol": args.tol,
            "estimate_latent_corr": args.estimate_latent_corr,
            "negative_policy": args.negative_policy,
            "structure": prereq.structure.as_dict(),
        }
    )


def _save_bootstrap_fit(
    cache_dir: Path,
    *,
    fit: Mapping[str, Any],
    cache_key: str,
    panel: FoldPanel,
    replicate: int,
    sampled_ids: Sequence[str],
    policy: Mapping[str, Any],
    fit_grid: int,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = cache_dir / "fit_arrays.npz"
    temporary = arrays_path.with_name(arrays_path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            A=np.asarray(fit["A"], dtype=float),
            b=np.asarray(fit["b"], dtype=float),
            R=np.asarray(fit["R"], dtype=float),
        )
    temporary.replace(arrays_path)
    _atomic_json(
        cache_dir / "fit_manifest.json",
        {
            "schema_version": SCRIPT_SCHEMA,
            "cache_key": cache_key,
            "outer_fold": panel.outer_fold,
            "replicate": replicate,
            "outer_training_model_ids": list(panel.training_model_ids),
            "outer_test_model_ids_forbidden": list(panel.test_model_ids),
            "sampled_model_ids": list(sampled_ids),
            "sampled_model_counts": dict(sorted(Counter(sampled_ids).items())),
            "sample_signature": param_uncertainty._sample_signature(
                np.asarray(
                    [panel.training_model_ids.index(model) for model in sampled_ids],
                    dtype=np.int64,
                )
            ),
            "fit_grid": fit_grid,
            "ridge": panel.selected_ridge,
            "items": list(map(str, fit["items"])),
            "dim_labels": list(map(str, fit["dim_labels"])),
            "loglik": fit["loglik"],
            "n_params": fit["n_params"],
            "n_iter": fit["n_iter"],
            "converged": fit["converged"],
            "diagnostics": fit.get("diag") or {},
            "bank_policy": policy,
            "arrays_sha256": _sha256(arrays_path),
        },
    )


def _load_bootstrap_fit(
    cache_dir: Path,
    *,
    expected_key: str,
    panel: FoldPanel,
    replicate: int,
    sampled_ids: Sequence[str],
    fit_grid: int,
) -> dict[str, Any] | None:
    manifest_path = cache_dir / "fit_manifest.json"
    arrays_path = cache_dir / "fit_arrays.npz"
    if not manifest_path.exists() and not arrays_path.exists():
        return None
    if not manifest_path.is_file() or not arrays_path.is_file():
        raise Phase4Error(f"partial bootstrap fit cache under {cache_dir}")
    manifest = _read_json(manifest_path)
    expected = {
        "schema_version": SCRIPT_SCHEMA,
        "cache_key": expected_key,
        "outer_fold": panel.outer_fold,
        "replicate": replicate,
        "outer_training_model_ids": list(panel.training_model_ids),
        "outer_test_model_ids_forbidden": list(panel.test_model_ids),
        "sampled_model_ids": list(sampled_ids),
        "fit_grid": fit_grid,
        "ridge": panel.selected_ridge,
        "sampled_model_counts": dict(sorted(Counter(sampled_ids).items())),
        "sample_signature": param_uncertainty._sample_signature(
            np.asarray(
                [panel.training_model_ids.index(model) for model in sampled_ids],
                dtype=np.int64,
            )
        ),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise Phase4Error(f"bootstrap cache mismatch for {key}: {cache_dir}")
    assert_fit_boundary(
        panel.training_model_ids, panel.test_model_ids, manifest["sampled_model_ids"]
    )
    if manifest.get("arrays_sha256") != _sha256(arrays_path):
        raise Phase4Error(f"bootstrap cache array hash mismatch: {cache_dir}")
    with np.load(arrays_path, allow_pickle=False) as arrays:
        fit = {
            "items": list(map(str, manifest["items"])),
            "A": np.asarray(arrays["A"], dtype=float),
            "b": np.asarray(arrays["b"], dtype=float),
            "R": np.asarray(arrays["R"], dtype=float),
            "dim_labels": list(map(str, manifest["dim_labels"])),
            "loglik": float(manifest["loglik"]),
            "n_params": int(manifest["n_params"]),
            "n_iter": int(manifest["n_iter"]),
            "converged": bool(manifest["converged"]),
            "diag": manifest.get("diagnostics") or {},
        }
    _validate_fit_payload(fit, context=str(cache_dir))
    return fit


class Phase4Runner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.prereq = load_prerequisites(args.nested_dir)
        self._validate_locked_runtime()
        self._base_banks: dict[int, tuple[scat.FittedBank, scat.Quadrature]] = {}
        self._preflight_outer_caches_and_grid()
        uncertainty = self.prereq.config.get("uncertainty") or {}
        self.configured_n_boot = int(uncertainty.get("parameter_bootstrap_replicates", 0))
        self.n_boot = int(args.n_boot) if args.n_boot is not None else self.configured_n_boot
        if self.n_boot < 2:
            raise Phase4Error("parameter uncertainty requires at least two bootstraps")
        self.configured_order_seeds = validate_order_seeds(
            list(uncertainty.get("order_seeds") or [])
        )
        raw_seeds = (
            [int(part.strip()) for part in args.order_seeds.split(",") if part.strip()]
            if args.order_seeds
            else list(self.configured_order_seeds)
        )
        self.order_seeds = validate_order_seeds(raw_seeds)
        if uncertainty.get("parameter_bootstrap_training_models_only") is not True:
            raise Phase4Error("config does not require training-only parameter bootstraps")
        if uncertainty.get("rerun_cat_path_per_bootstrap") is not True:
            raise Phase4Error("config does not require CAT path reruns per bootstrap bank")
        if uncertainty.get("total_variance_method") != "law_of_total_variance":
            raise Phase4Error("config does not lock the law-of-total-variance method")
        gates = self.prereq.config.get("selection_gates") or {}
        self.minimum_valid_rate = float(gates.get("minimum_valid_parameter_bootstrap_rate", 0.9))
        if not 0 < self.minimum_valid_rate <= 1:
            raise Phase4Error("invalid minimum valid-bootstrap rate")
        self.code_hashes = _phase4_code_dependency_hashes()
        self.dependency_versions = _phase4_dependency_versions()
        self.phase4_signature = self._signature()

    def _preflight_outer_caches_and_grid(self) -> None:
        """Read and validate every frozen outer fit, including during --plan-only."""

        required_nodes = self.prereq.effective_node_count or self.prereq.eap_grid
        if int(self.args.max_grid_nodes) < int(required_nodes):
            raise Phase4Error(
                f"--max-grid-nodes={self.args.max_grid_nodes} is below the locked "
                f"effective node count {required_nodes}"
            )
        for panel in self.prereq.panels:
            bank, quadrature = self._load_outer_base_bank(panel)
            if len(quadrature.grid) != int(required_nodes):
                raise Phase4Error(
                    f"outer fold {panel.outer_fold} quadrature has {len(quadrature.grid)} "
                    f"nodes, expected the locked {required_nodes}"
                )
            self._base_banks[panel.outer_fold] = (bank, quadrature)

    def _validate_locked_runtime(self) -> None:
        """Keep Phase-4 replays on the exact Phase-3 CAT/optimizer runtime."""

        runtime = self.prereq.nested_manifest.get("runtime")
        if not isinstance(runtime, Mapping):
            raise Phase4Error("nested manifest lacks its locked runtime configuration")
        expected = {
            "seed": int(self.args.cat_seed),
            "top_n": int(self.args.top_n),
            "maximum_adaptive_scenarios": int(self.args.max_scenarios),
            "minimum_scored_criteria": int(self.args.minimum_scored_criteria),
            "max_iter": int(self.args.max_iter),
            "negative_policy": str(self.args.negative_policy),
            "estimate_latent_corr": bool(self.args.estimate_latent_corr),
            "allow_unconverged_fit": bool(self.args.allow_unconverged_fit),
        }
        for key, supplied in expected.items():
            if runtime.get(key) != supplied:
                raise Phase4Error(
                    f"Phase-4 {key}={supplied!r} differs from locked Phase-3 "
                    f"value {runtime.get(key)!r}"
                )
        for key, supplied in (
            ("tol", float(self.args.tol)),
            ("mwle_ridge", float(self.args.mwle_ridge)),
        ):
            try:
                locked = float(runtime[key])
            except (KeyError, TypeError, ValueError) as error:
                raise Phase4Error(f"nested manifest lacks numeric runtime {key}") from error
            if not math.isclose(locked, supplied, rel_tol=0.0, abs_tol=1e-15):
                raise Phase4Error(
                    f"Phase-4 {key}={supplied!r} differs from locked Phase-3 value {locked!r}"
                )

    def _signature(self) -> str:
        return _canonical_hash(
            {
                "schema": SCRIPT_SCHEMA,
                "code_dependency_hashes": self.code_hashes,
                "dependency_versions": self.dependency_versions,
                "nested_manifest_sha256": _sha256(self.prereq.nested_manifest_path),
                "fold_assignments_sha256": _sha256(self.prereq.fold_assignments_path),
                "finalists_sha256": _sha256(self.prereq.finalists_path),
                "outer_rows_sha256": _sha256(self.prereq.outer_rows_path),
                "fit_grid": self.prereq.fit_grid,
                "eap_grid": self.prereq.eap_grid,
                "quadrature_method": self.prereq.quadrature_method,
                "linear_bound": self.prereq.linear_bound,
                "quadrature_axis_sha256": self.prereq.quadrature_axis_sha256,
                "quadrature_log_prior_sha256": self.prereq.quadrature_log_prior_sha256,
                "effective_node_count": self.prereq.effective_node_count,
                "finalist_panels": [
                    {
                        "outer_fold": panel.outer_fold,
                        "ridge": panel.selected_ridge,
                        "fit_cache_key": panel.fit_cache_key,
                        "candidates": [asdict(candidate) for candidate in panel.candidates],
                    }
                    for panel in self.prereq.panels
                ],
                "n_boot": self.n_boot,
                "bootstrap_seed": self.args.bootstrap_seed,
                "order_seeds": self.order_seeds,
                "runtime": {
                    "top_n": self.args.top_n,
                    "max_scenarios": self.args.max_scenarios,
                    "minimum_scored_criteria": self.args.minimum_scored_criteria,
                    "mwle_ridge": self.args.mwle_ridge,
                    "max_iter": self.args.max_iter,
                    "tol": self.args.tol,
                    "negative_policy": self.args.negative_policy,
                    "estimate_latent_corr": self.args.estimate_latent_corr,
                    "allow_unconverged_fit": self.args.allow_unconverged_fit,
                },
            }
        )

    @property
    def nested_dir(self) -> Path:
        return self.prereq.nested_dir

    def plan_payload(self) -> dict[str, Any]:
        return {
            "schema_version": SCRIPT_SCHEMA,
            "status": "plan_validated",
            "read_only": True,
            "nested_dir": str(self.nested_dir),
            "nested_manifest_sha256": _sha256(self.prereq.nested_manifest_path),
            "phase4_signature": self.phase4_signature,
            "fit_grid": self.prereq.fit_grid,
            "eap_grid": self.prereq.eap_grid,
            "quadrature_method": self.prereq.quadrature_method,
            "linear_bound": self.prereq.linear_bound,
            "quadrature_axis_sha256": self.prereq.quadrature_axis_sha256,
            "quadrature_log_prior_sha256": self.prereq.quadrature_log_prior_sha256,
            "effective_node_count": self.prereq.effective_node_count,
            "locked_grid_within_max_grid_nodes": True,
            "max_grid_nodes": self.args.max_grid_nodes,
            "outer_fit_cache_preflight": {
                "status": "passed",
                "folds_verified": sorted(self._base_banks),
            },
            "outer_folds": len(self.prereq.panels),
            "outer_training_sizes": {
                str(panel.outer_fold): len(panel.training_model_ids)
                for panel in self.prereq.panels
            },
            "outer_test_sizes": {
                str(panel.outer_fold): len(panel.test_model_ids) for panel in self.prereq.panels
            },
            "finalists_per_fold": {
                str(panel.outer_fold): len(panel.candidates) for panel in self.prereq.panels
            },
            "parameter_bootstrap_replicates_per_fold": self.n_boot,
            "preregistered_parameter_bootstrap_replicates": self.configured_n_boot,
            "headline_eligible": self.n_boot == self.configured_n_boot,
            "order_seed_panel_preregistered": self.order_seeds == self.configured_order_seeds,
            "order_seeds": list(self.order_seeds),
            "estimated_item_fits": len(self.prereq.panels) * self.n_boot,
            "estimated_bootstrap_cat_replays": sum(
                len(panel.test_model_ids) * len(panel.candidates) * self.n_boot
                for panel in self.prereq.panels
            ),
            "estimated_order_cat_replays": sum(
                len(panel.test_model_ids) * len(panel.candidates) * len(self.order_seeds)
                for panel in self.prereq.panels
            ),
            "outer_test_in_fit": False,
            "outputs": list(FINAL_OUTPUTS),
        }

    def _owned_existing(self) -> list[Path]:
        return [self.nested_dir / name for name in OWNED_PATHS if (self.nested_dir / name).exists()]

    def _prepare_output(self) -> None:
        if self.args.fresh:
            for path in self._owned_existing():
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
        manifest_path = self.nested_dir / "phase4_manifest.json"
        if self.args.resume:
            if not manifest_path.is_file():
                raise Phase4Error("--resume requires an existing phase4_manifest.json")
            manifest = _read_json(manifest_path)
            if manifest.get("phase4_signature") != self.phase4_signature:
                raise Phase4Error("resume manifest does not match this Phase-4 study")
            if manifest.get("status") == "complete":
                self._verify_completed_outputs(manifest)
                return
        elif self._owned_existing():
            raise Phase4Error(
                "Phase-4-owned outputs already exist; use --resume or --fresh. "
                "Phase-3 files are never deleted."
            )

    def _verify_completed_outputs(self, manifest: Mapping[str, Any]) -> None:
        outputs = manifest.get("outputs")
        if not isinstance(outputs, Mapping):
            raise Phase4Error("complete Phase-4 manifest lacks output hashes")
        for name in FINAL_OUTPUTS:
            path = self.nested_dir / name
            if not path.is_file():
                raise Phase4Error(f"complete Phase-4 manifest is missing {name}")
            if name == "phase4_manifest.json":
                continue
            item = outputs.get(name)
            if not isinstance(item, Mapping) or item.get("sha256") != _sha256(path):
                raise Phase4Error(f"complete Phase-4 output hash mismatch for {name}")
        inventories = manifest.get("completed_artifact_inventories")
        if not isinstance(inventories, Mapping):
            raise Phase4Error("complete Phase-4 manifest lacks intermediate inventories")
        for name, relative in (
            ("phase4_checkpoints", "phase4_checkpoints"),
            ("phase4_cache", "phase4_cache"),
        ):
            item = inventories.get(name)
            if not isinstance(item, Mapping):
                raise Phase4Error(f"complete Phase-4 manifest lacks {name} inventory")
            inventory = _tree_inventory(self.nested_dir / relative)
            if int(item.get("file_count", -1)) != len(inventory):
                raise Phase4Error(f"complete Phase-4 {name} file count changed")
            if item.get("inventory_sha256") != _canonical_hash(inventory):
                raise Phase4Error(f"complete Phase-4 {name} inventory hash changed")

    def _base_manifest(self, status: str) -> dict[str, Any]:
        return {
            "schema_version": SCRIPT_SCHEMA,
            "status": status,
            "phase4_signature": self.phase4_signature,
            "generated_at": _utcnow(),
            "script": "scripts/nested_cat_total_uncertainty.py",
            "command": sys.argv,
            "git_commit": _git_commit(),
            "nested_prerequisite": {
                "path": str(self.prereq.nested_manifest_path),
                "sha256": _sha256(self.prereq.nested_manifest_path),
                "schema_version": NESTED_SCHEMA,
                "status": "phase3_finalists_complete",
            },
            "inputs": {
                name: {"path": str(path), "sha256": self.prereq.input_hashes[name]}
                for name, path in (
                    ("config", self.prereq.config_path),
                    ("response_matrix", self.prereq.matrix_path),
                    ("rubrics", self.prereq.rubrics_path),
                    ("scenarios", self.prereq.scenarios_path),
                )
            },
            "locked_numerical_settings": {
                "fit_grid": self.prereq.fit_grid,
                "eap_grid": self.prereq.eap_grid,
                "quadrature_method": self.prereq.quadrature_method,
                "linear_bound": self.prereq.linear_bound,
                "quadrature_axis_sha256": self.prereq.quadrature_axis_sha256,
                "quadrature_log_prior_sha256": self.prereq.quadrature_log_prior_sha256,
                "effective_node_count": self.prereq.effective_node_count,
                "source": self.prereq.nested_manifest.get("dense_grid_lock"),
            },
            "outer_fold_finalist_panels": [
                {
                    "outer_fold": panel.outer_fold,
                    "training_model_ids": list(panel.training_model_ids),
                    "test_model_ids": list(panel.test_model_ids),
                    "selected_ridge": panel.selected_ridge,
                    "candidate_ids": list(panel.candidate_ids),
                    "candidates": [asdict(candidate) for candidate in panel.candidates],
                    "base_fit_cache_key": panel.fit_cache_key,
                }
                for panel in self.prereq.panels
            ],
            "uncertainty": {
                "method": "outer-training-only nonparametric model bootstrap",
                "replicates_per_fold": self.n_boot,
                "preregistered_replicates_per_fold": self.configured_n_boot,
                "headline_eligible": self.n_boot == self.configured_n_boot,
                "bootstrap_seed": self.args.bootstrap_seed,
                "cat_path_rerun_under_every_bootstrap_bank": True,
                "law_of_total_variance": (
                    "total_variance = mean(conditional_mwle_se^2) + "
                    "sample_variance(bootstrap_mwle_theta)"
                ),
                "minimum_valid_draw_rate": self.minimum_valid_rate,
            },
            "order_stability": {
                "seeds": list(self.order_seeds),
                "n_seeds": len(self.order_seeds),
                "preregistered_seeds": list(self.configured_order_seeds),
                "headline_eligible": self.order_seeds == self.configured_order_seeds,
                "reported_separately_from_total_se": True,
            },
            "leakage_guards": {
                "outer_test_ids_forbidden_from_bootstrap_fit": True,
                "bootstrap_sample_size_equals_outer_training_size": True,
                "administration_scenarios_only_in_cat_path": True,
                "outer_finalists_frozen_before_phase4": True,
                "one_bootstrap_fit_reused_across_each_fold_finalist_panel": True,
            },
            "policy": {
                "selection_order": [
                    "lowest_p90_scenario_count",
                    "lowest_mean_scenario_count",
                    "lowest_p90_se_total",
                    "prefer_trace",
                    "candidate_id_lexical",
                ],
                "length_tie_policy": "exact_numeric_equality",
                "total_uncertainty_tiebreak_metric": "p90_se_total",
                "distinct_total_se_tolerance": None,
                "final_policy_freeze_allowed": False,
                "provisional_choice_uses_outer_outcomes": True,
                "not_an_unbiased_nested_performance_estimate": True,
                "missing_tolerance_reason": (
                    "The frozen remediation config does not define a distinct "
                    "preregistered total-SE acceptance tolerance."
                ),
            },
            "environment": {
                "platform": platform.platform(),
                **self.dependency_versions,
                "canonical_dependency_versions_sha256": _canonical_hash(
                    self.dependency_versions
                ),
            },
            "code_provenance": {
                "files": self.code_hashes,
                "canonical_sha256": _canonical_hash(self.code_hashes),
            },
        }

    def _fit_bootstrap(
        self, panel: FoldPanel, replicate: int, sampled_ids: Sequence[str]
    ) -> tuple[scat.FittedBank, scat.Quadrature, str]:
        assert_fit_boundary(panel.training_model_ids, panel.test_model_ids, sampled_ids)
        cache_key = _bootstrap_cache_key(self.prereq, panel, replicate, sampled_ids, self.args)
        cache_dir = (
            self.nested_dir
            / "phase4_cache"
            / "bootstrap_fits"
            / f"outer_{panel.outer_fold}"
            / f"replicate_{replicate:04d}"
        )
        fit = _load_bootstrap_fit(
            cache_dir,
            expected_key=cache_key,
            panel=panel,
            replicate=replicate,
            sampled_ids=sampled_ids,
            fit_grid=self.prereq.fit_grid,
        )
        if fit is None:
            fit_args = argparse.Namespace(
                grid=self.prereq.fit_grid,
                estimate_latent_corr=self.args.estimate_latent_corr,
                ridge=panel.selected_ridge,
                max_iter=self.args.max_iter,
                tol=self.args.tol,
            )
            sampled_matrix = self.prereq.matrix.loc[list(sampled_ids)]
            if set(map(str, sampled_matrix.index)) & set(panel.test_model_ids):
                raise Phase4Error("outer-test ID reached the calibration DataFrame")
            fit = cell_cv.fit_structure(
                sampled_matrix, self.prereq.q_by, fit_args, self.prereq.structure
            )
            if not fit["converged"] and not self.args.allow_unconverged_fit:
                raise RecoverableNumericalFailure("bootstrap M2PL fit did not converge")
            bank, policy = scenario_cv.build_fold_bank(
                fit,
                self.prereq.structure,
                self.prereq.source_records,
                negative_policy=self.args.negative_policy,
            )
            _save_bootstrap_fit(
                cache_dir,
                fit=fit,
                cache_key=cache_key,
                panel=panel,
                replicate=replicate,
                sampled_ids=sampled_ids,
                policy=policy,
                fit_grid=self.prereq.fit_grid,
            )
        else:
            bank, _policy = scenario_cv.build_fold_bank(
                fit,
                self.prereq.structure,
                self.prereq.source_records,
                negative_policy=self.args.negative_policy,
            )
        administration_bank = nested.subset_fitted_bank(
            bank, self.prereq.administration_scenario_ids
        )
        quadrature = scat.build_quadrature(
            administration_bank.n_dims,
            self.prereq.eap_grid,
            administration_bank.latent_correlation,
            max_nodes=self.args.max_grid_nodes,
            method=self.prereq.quadrature_method,
            linear_bound=self.prereq.linear_bound,
        )
        return administration_bank, quadrature, cache_key

    def _bootstrap_checkpoint(self, panel: FoldPanel, replicate: int) -> Path:
        return (
            self.nested_dir
            / "phase4_checkpoints"
            / "bootstrap"
            / f"outer_{panel.outer_fold}_replicate_{replicate:04d}.json"
        )

    def _order_checkpoint(self, panel: FoldPanel, seed: int) -> Path:
        return (
            self.nested_dir
            / "phase4_checkpoints"
            / "order"
            / f"outer_{panel.outer_fold}_seed_{seed}.json"
        )

    def _checkpoint_rows(
        self,
        path: Path,
        *,
        kind: str,
        panel: FoldPanel,
        index_name: str,
        index_value: int,
        expected_extra: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]] | None:
        if not (self.args.resume and path.is_file()):
            return None
        payload = _read_json(path)
        expected = {
            "schema_version": SCRIPT_SCHEMA,
            "phase4_signature": self.phase4_signature,
            "kind": kind,
            "outer_fold": panel.outer_fold,
            "candidate_ids": list(panel.candidate_ids),
            index_name: index_value,
            **dict(expected_extra or {}),
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise Phase4Error(f"checkpoint mismatch for {key}: {path}")
        if payload.get("test_model_ids") != list(panel.test_model_ids):
            raise Phase4Error(f"checkpoint test IDs mismatch: {path}")
        rows = payload.get("rows")
        if not isinstance(rows, list):
            raise Phase4Error(f"checkpoint does not contain rows: {path}")
        if payload.get("row_count") != len(rows):
            raise Phase4Error(f"checkpoint row count mismatch: {path}")
        if payload.get("rows_sha256") != _canonical_hash(rows):
            raise Phase4Error(f"checkpoint row hash mismatch: {path}")
        validated = [dict(row) for row in rows]
        self._validate_checkpoint_rows(
            validated,
            path=path,
            kind=kind,
            panel=panel,
            index_name=index_name,
            index_value=index_value,
            expected_row_values={
                key: expected_extra[key]
                for key in (
                    "sample_seed",
                    "sample_signature",
                    "bootstrap_fit_cache_key",
                    "base_fit_cache_key",
                )
                if expected_extra is not None and key in expected_extra
            },
        )
        return validated

    @staticmethod
    def _validate_checkpoint_rows(
        rows: Sequence[Mapping[str, Any]],
        *,
        path: Path,
        kind: str,
        panel: FoldPanel,
        index_name: str,
        index_value: int,
        expected_row_values: Mapping[str, Any] | None = None,
    ) -> None:
        expected_pairs = {
            (candidate.candidate_id, model)
            for candidate in panel.candidates
            for model in panel.test_model_ids
        }
        actual_pairs: list[tuple[str, str]] = []
        candidate_by_id = {
            candidate.candidate_id: candidate for candidate in panel.candidates
        }
        allowed_statuses = (
            {"ok", "fit_numerical_error", "replay_numerical_error"}
            if kind == "parameter_bootstrap_cat_path"
            else {"ok", "replay_numerical_error"}
        )
        for row in rows:
            candidate_id = str(row.get("candidate_id") or "")
            model = str(row.get("model") or "")
            actual_pairs.append((candidate_id, model))
            candidate = candidate_by_id.get(candidate_id)
            if candidate is None:
                raise Phase4Error(f"checkpoint contains a foreign candidate: {path}")
            expected = {
                "outer_fold": panel.outer_fold,
                index_name: index_value,
                "selected_ridge": panel.selected_ridge,
                "minimum_scenarios": candidate.minimum_scenarios,
                "conditional_se_target": candidate.conditional_se_target,
                "selector": candidate.selector,
                **dict(expected_row_values or {}),
            }
            for key, value in expected.items():
                if row.get(key) != value:
                    raise Phase4Error(f"checkpoint row mismatch for {key}: {path}")
            if str(row.get("status") or "") not in allowed_statuses:
                raise Phase4Error(f"checkpoint contains an unsupported row status: {path}")
            if row.get("status") == "ok" and "mwle_converged" not in row:
                raise Phase4Error(f"successful checkpoint row lacks MWLE provenance: {path}")
        if len(actual_pairs) != len(expected_pairs) or set(actual_pairs) != expected_pairs:
            raise Phase4Error(
                f"checkpoint must contain exactly one row per finalist/test-model pair: {path}"
            )
        fit_failures = [row.get("status") == "fit_numerical_error" for row in rows]
        if any(fit_failures) and not all(fit_failures):
            raise Phase4Error(f"bootstrap fit failure must apply to the entire panel: {path}")

    def _write_checkpoint(
        self,
        path: Path,
        *,
        kind: str,
        panel: FoldPanel,
        index_name: str,
        index_value: int,
        rows: Sequence[Mapping[str, Any]],
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        normalized_rows = [dict(row) for row in rows]
        self._validate_checkpoint_rows(
            normalized_rows,
            path=path,
            kind=kind,
            panel=panel,
            index_name=index_name,
            index_value=index_value,
            expected_row_values={
                key: extra[key]
                for key in (
                    "sample_seed",
                    "sample_signature",
                    "bootstrap_fit_cache_key",
                    "base_fit_cache_key",
                )
                if extra is not None and key in extra
            },
        )
        _atomic_json(
            path,
            {
                "schema_version": SCRIPT_SCHEMA,
                "phase4_signature": self.phase4_signature,
                "kind": kind,
                "outer_fold": panel.outer_fold,
                "candidate_ids": list(panel.candidate_ids),
                "test_model_ids": list(panel.test_model_ids),
                index_name: index_value,
                **dict(extra or {}),
                "row_count": len(normalized_rows),
                "rows_sha256": _canonical_hash(normalized_rows),
                "rows": normalized_rows,
            },
        )

    def _run_bootstraps(self) -> pd.DataFrame:
        all_rows: list[dict[str, Any]] = []
        for panel in self.prereq.panels:
            for replicate in range(self.n_boot):
                sample_seed = _replicate_seed(
                    self.args.bootstrap_seed, panel.outer_fold, replicate
                )
                sample = draw_outer_training_sample(
                    panel.training_model_ids, panel.test_model_ids, seed=sample_seed
                )
                sample_signature = _canonical_hash(list(sample))
                cache_key = _bootstrap_cache_key(
                    self.prereq, panel, replicate, sample, self.args
                )
                checkpoint = self._bootstrap_checkpoint(panel, replicate)
                checkpoint_provenance = {
                    "sample_seed": sample_seed,
                    "sampled_model_ids": list(sample),
                    "sample_signature": sample_signature,
                    "bootstrap_fit_cache_key": cache_key,
                }
                rows = self._checkpoint_rows(
                    checkpoint,
                    kind="parameter_bootstrap_cat_path",
                    panel=panel,
                    index_name="replicate",
                    index_value=replicate,
                    expected_extra=checkpoint_provenance,
                )
                if rows is not None:
                    if any(row.get("status") != "fit_numerical_error" for row in rows):
                        cache_dir = (
                            self.nested_dir
                            / "phase4_cache"
                            / "bootstrap_fits"
                            / f"outer_{panel.outer_fold}"
                            / f"replicate_{replicate:04d}"
                        )
                        cached_fit = _load_bootstrap_fit(
                            cache_dir,
                            expected_key=cache_key,
                            panel=panel,
                            replicate=replicate,
                            sampled_ids=sample,
                            fit_grid=self.prereq.fit_grid,
                        )
                        if cached_fit is None:
                            raise Phase4Error(
                                f"successful checkpoint lacks its bootstrap fit: {checkpoint}"
                            )
                    all_rows.extend(rows)
                    continue
                try:
                    bank, quadrature, loaded_cache_key = self._fit_bootstrap(
                        panel, replicate, sample
                    )
                except (
                    RecoverableNumericalFailure,
                    np.linalg.LinAlgError,
                    FloatingPointError,
                ) as error:
                    rows = [
                        {
                            "outer_fold": panel.outer_fold,
                            "replicate": replicate,
                            "sample_seed": sample_seed,
                            "sample_signature": sample_signature,
                            "model": model,
                            "candidate_id": candidate.candidate_id,
                            "selected_ridge": panel.selected_ridge,
                            "minimum_scenarios": candidate.minimum_scenarios,
                            "conditional_se_target": candidate.conditional_se_target,
                            "selector": candidate.selector,
                            "bootstrap_fit_cache_key": cache_key,
                            "status": "fit_numerical_error",
                            "error": f"{type(error).__name__}: {error}",
                            "mwle_converged": False,
                        }
                        for candidate in panel.candidates
                        for model in panel.test_model_ids
                    ]
                else:
                    if loaded_cache_key != cache_key:
                        raise Phase4Error("bootstrap fit returned a different cache key")
                    rows = []
                    for candidate in panel.candidates:
                        for model in panel.test_model_ids:
                            try:
                                result = replay_model(
                                    model=model,
                                    row=self.prereq.matrix.loc[model],
                                    bank=bank,
                                    scenario_records=self.prereq.scenario_records,
                                    quadrature=quadrature,
                                    candidate=candidate,
                                    seed=self.args.cat_seed,
                                    top_n=self.args.top_n,
                                    max_scenarios=self.args.max_scenarios,
                                    minimum_scored_criteria=self.args.minimum_scored_criteria,
                                    mwle_ridge=self.args.mwle_ridge,
                                )
                                flat = scat.flatten_result(result, bank.dims)
                                row = {"status": "ok", "error": "", **flat}
                            except (np.linalg.LinAlgError, FloatingPointError) as error:
                                row = {
                                    "model": model,
                                    "status": "replay_numerical_error",
                                    "error": f"{type(error).__name__}: {error}",
                                    "mwle_converged": False,
                                    "scenario_order": "[]",
                                }
                            row.update(
                                {
                                    "outer_fold": panel.outer_fold,
                                    "replicate": replicate,
                                    "sample_seed": sample_seed,
                                    "sample_signature": sample_signature,
                                    "candidate_id": candidate.candidate_id,
                                    "selected_ridge": panel.selected_ridge,
                                    "minimum_scenarios": candidate.minimum_scenarios,
                                    "conditional_se_target": candidate.conditional_se_target,
                                    "selector": candidate.selector,
                                    "bootstrap_fit_cache_key": cache_key,
                                }
                            )
                            rows.append(row)
                self._write_checkpoint(
                    checkpoint,
                    kind="parameter_bootstrap_cat_path",
                    panel=panel,
                    index_name="replicate",
                    index_value=replicate,
                    rows=rows,
                    extra=checkpoint_provenance,
                )
                all_rows.extend(rows)
        frame = pd.DataFrame(all_rows)
        return frame.sort_values(
            ["outer_fold", "replicate", "candidate_id", "model"]
        ).reset_index(drop=True)

    def _load_outer_base_bank(
        self, panel: FoldPanel
    ) -> tuple[scat.FittedBank, scat.Quadrature]:
        cache_dir = (
            self.nested_dir
            / "fit_cache"
            / f"outer_{panel.outer_fold}"
            / "outer_train"
            / f"ridge_{nested._ridge_slug(panel.selected_ridge)}"
        )
        fit = _load_fit_arrays(
            cache_dir,
            expected_cache_key=panel.fit_cache_key,
            expected_training_ids=panel.training_model_ids,
            expected_fit_grid=self.prereq.fit_grid,
            expected_ridge=panel.selected_ridge,
        )
        if not fit["converged"] and not self.args.allow_unconverged_fit:
            raise Phase4Error(f"outer fold {panel.outer_fold} base fit did not converge")
        bank, _policy = scenario_cv.build_fold_bank(
            fit,
            self.prereq.structure,
            self.prereq.source_records,
            negative_policy=self.args.negative_policy,
        )
        administration_bank = nested.subset_fitted_bank(
            bank, self.prereq.administration_scenario_ids
        )
        quadrature = scat.build_quadrature(
            administration_bank.n_dims,
            self.prereq.eap_grid,
            administration_bank.latent_correlation,
            max_nodes=self.args.max_grid_nodes,
            method=self.prereq.quadrature_method,
            linear_bound=self.prereq.linear_bound,
        )
        return administration_bank, quadrature

    def _outer_base_bank(self, panel: FoldPanel) -> tuple[scat.FittedBank, scat.Quadrature]:
        try:
            return self._base_banks[panel.outer_fold]
        except KeyError as error:
            raise Phase4Error(
                f"outer fold {panel.outer_fold} was not preflighted"
            ) from error

    def _run_order_seeds(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for panel in self.prereq.panels:
            bank, quadrature = self._outer_base_bank(panel)
            for seed in self.order_seeds:
                checkpoint = self._order_checkpoint(panel, seed)
                seed_rows = self._checkpoint_rows(
                    checkpoint,
                    kind="order_path_seed",
                    panel=panel,
                    index_name="seed",
                    index_value=seed,
                    expected_extra={"base_fit_cache_key": panel.fit_cache_key},
                )
                if seed_rows is None:
                    seed_rows = []
                    for candidate in panel.candidates:
                        for model in panel.test_model_ids:
                            try:
                                result = replay_model(
                                    model=model,
                                    row=self.prereq.matrix.loc[model],
                                    bank=bank,
                                    scenario_records=self.prereq.scenario_records,
                                    quadrature=quadrature,
                                    candidate=candidate,
                                    seed=seed,
                                    top_n=self.args.top_n,
                                    max_scenarios=self.args.max_scenarios,
                                    minimum_scored_criteria=self.args.minimum_scored_criteria,
                                    mwle_ridge=self.args.mwle_ridge,
                                )
                                flat = scat.flatten_result(result, bank.dims)
                                row = {"status": "ok", "error": "", **flat}
                            except (np.linalg.LinAlgError, FloatingPointError) as error:
                                row = {
                                    "model": model,
                                    "seed": seed,
                                    "status": "replay_numerical_error",
                                    "error": f"{type(error).__name__}: {error}",
                                    "mwle_converged": False,
                                    "scenario_order": "[]",
                                }
                            row.update(
                                {
                                    "seed": seed,
                                    "outer_fold": panel.outer_fold,
                                    "candidate_id": candidate.candidate_id,
                                    "selected_ridge": panel.selected_ridge,
                                    "base_fit_cache_key": panel.fit_cache_key,
                                    "minimum_scenarios": candidate.minimum_scenarios,
                                    "conditional_se_target": candidate.conditional_se_target,
                                    "selector": candidate.selector,
                                }
                            )
                            seed_rows.append(row)
                    self._write_checkpoint(
                        checkpoint,
                        kind="order_path_seed",
                        panel=panel,
                        index_name="seed",
                        index_value=seed,
                        rows=seed_rows,
                        extra={"base_fit_cache_key": panel.fit_cache_key},
                    )
                rows.extend(seed_rows)
        return (
            pd.DataFrame(rows)
            .sort_values(["outer_fold", "candidate_id", "model", "seed"])
            .reset_index(drop=True)
        )

    def _total_se_frame(self, draws: pd.DataFrame) -> pd.DataFrame:
        pieces: list[pd.DataFrame] = []
        for panel in self.prereq.panels:
            fold_rows = draws[
                pd.to_numeric(draws["outer_fold"], errors="coerce") == panel.outer_fold
            ]
            for candidate in panel.candidates:
                subset = fold_rows[
                    fold_rows["candidate_id"].astype(str) == candidate.candidate_id
                ]
                frame = law_total_variance(
                    subset,
                    model_ids=panel.test_model_ids,
                    dimensions=self.prereq.structure.labels,
                    n_boot=self.n_boot,
                    minimum_valid_rate=self.minimum_valid_rate,
                )
                frame.insert(0, "outer_fold", panel.outer_fold)
                frame.insert(2, "candidate_id", candidate.candidate_id)
                frame["selected_ridge"] = panel.selected_ridge
                frame["minimum_scenarios"] = candidate.minimum_scenarios
                frame["conditional_se_target"] = candidate.conditional_se_target
                frame["selector"] = candidate.selector
                frame["se_total_le_conditional_target_diagnostic"] = (
                    pd.to_numeric(frame["se_total"], errors="coerce")
                    <= candidate.conditional_se_target
                )
                pieces.append(frame)
        return (
            pd.concat(pieces, ignore_index=True)
            .sort_values(["outer_fold", "candidate_id", "model", "dimension"])
            .reset_index(drop=True)
        )

    def _order_frame(self, run_rows: pd.DataFrame) -> pd.DataFrame:
        pieces: list[pd.DataFrame] = []
        dim = self.prereq.structure.labels[0]
        for panel in self.prereq.panels:
            fold_rows = run_rows[
                pd.to_numeric(run_rows["outer_fold"], errors="coerce") == panel.outer_fold
            ]
            for candidate in panel.candidates:
                subset = fold_rows[
                    fold_rows["candidate_id"].astype(str) == candidate.candidate_id
                ]
                spread = order_exp.per_model_spread(subset, self.prereq.structure.labels)
                spread.insert(0, "outer_fold", panel.outer_fold)
                spread.insert(2, "candidate_id", candidate.candidate_id)
                spread["selected_ridge"] = panel.selected_ridge
                spread["minimum_scenarios"] = candidate.minimum_scenarios
                spread["conditional_se_target"] = candidate.conditional_se_target
                spread["selector"] = candidate.selector
                spread["valid_seed_rate"] = pd.to_numeric(
                    spread["mwle_valid_runs"], errors="coerce"
                ) / len(self.order_seeds)
                spread["all_seed_attempts_present"] = (
                    pd.to_numeric(spread["n_seed_runs"], errors="coerce")
                    == len(self.order_seeds)
                )
                spread["all_seed_runs_valid"] = (
                    pd.to_numeric(spread["mwle_valid_runs"], errors="coerce")
                    == len(self.order_seeds)
                )
                spread["order_path_sd"] = spread[f"mwle_{dim}_sd"]
                spread["order_path_range"] = spread[f"mwle_{dim}_range"]
                spread["order_path_gate"] = (
                    spread["all_seed_attempts_present"].astype(bool)
                    & spread["all_seed_runs_valid"].astype(bool)
                    & (
                        pd.to_numeric(spread["order_path_sd"], errors="coerce")
                        <= candidate.conditional_se_target
                    )
                )
                pieces.append(spread)
        return (
            pd.concat(pieces, ignore_index=True)
            .sort_values(["outer_fold", "candidate_id", "model"])
            .reset_index(drop=True)
        )

    @staticmethod
    def _finite_summary(values: pd.Series) -> dict[str, float | None]:
        numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
        return {
            "mean": float(numeric.mean()) if numeric.size else None,
            "median": float(np.median(numeric)) if numeric.size else None,
            "p90": float(np.percentile(numeric, 90)) if numeric.size else None,
            "max": float(numeric.max()) if numeric.size else None,
        }

    def _pareto_frame(self, total_se: pd.DataFrame, order: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        outer = self.prereq.outer_rows.copy()
        handoff_by_fold = {
            int(row["outer_fold"]): row
            for row in (self.prereq.finalists_payload.get("folds") or [])
        }
        for panel in self.prereq.panels:
            finalist_by_id = {
                str(row["candidate_id"]): row
                for row in (handoff_by_fold[panel.outer_fold].get("finalists") or [])
            }
            for candidate in panel.candidates:
                candidate_id = candidate.candidate_id
                se_rows = total_se[
                    (pd.to_numeric(total_se["outer_fold"], errors="coerce") == panel.outer_fold)
                    & (total_se["candidate_id"].astype(str) == candidate_id)
                ]
                order_rows = order[
                    (pd.to_numeric(order["outer_fold"], errors="coerce") == panel.outer_fold)
                    & (order["candidate_id"].astype(str) == candidate_id)
                ]
                oof = outer[
                    (pd.to_numeric(outer["outer_fold"], errors="coerce") == panel.outer_fold)
                    & (outer["candidate_id"].astype(str) == candidate_id)
                ]
                if (
                    set(se_rows["model"].astype(str)) != set(panel.test_model_ids)
                    or set(order_rows["model"].astype(str)) != set(panel.test_model_ids)
                    or set(oof["model"].astype(str)) != set(panel.test_model_ids)
                ):
                    raise Phase4Error(
                        f"outer fold {panel.outer_fold} candidate {candidate_id} lacks "
                        "identical local model support"
                    )
                recovery = scenario_cv.recovery_stats(
                    pd.to_numeric(oof.get("theta_reference"), errors="coerce"),
                    pd.to_numeric(oof.get("theta_cat_mwle"), errors="coerce"),
                )
                se_stats = self._finite_summary(se_rows["se_total"])
                order_stats = self._finite_summary(order_rows["order_path_sd"])
                length = pd.to_numeric(oof.get("cat_scenarios_administered"), errors="coerce")
                finalist = finalist_by_id[candidate_id]
                rows.append(
                    {
                        "outer_fold": panel.outer_fold,
                        "candidate_id": candidate_id,
                        "minimum_scenarios": candidate.minimum_scenarios,
                        "conditional_se_target": candidate.conditional_se_target,
                        "selector": candidate.selector,
                        "phase3_inner_p90_scenario_count": finalist["p90_scenario_count"],
                        "phase3_inner_mean_scenario_count": finalist["mean_scenario_count"],
                        "n_outer_test_models": len(panel.test_model_ids),
                        "oof_mean_scenario_count": (
                            float(length.mean()) if length.notna().any() else None
                        ),
                        "oof_p90_scenario_count": (
                            float(np.percentile(length.dropna(), 90))
                            if length.notna().any()
                            else None
                        ),
                        "mean_se_total": se_stats["mean"],
                        "median_se_total": se_stats["median"],
                        "p90_se_total": se_stats["p90"],
                        "max_se_total": se_stats["max"],
                        "minimum_valid_draw_rate": float(se_rows["valid_draw_rate"].min()),
                        "valid_bootstrap_gate_all_models": bool(
                            se_rows["valid_draw_gate"].all()
                        ),
                        "proportion_se_total_le_conditional_target_diagnostic": float(
                            se_rows[
                                "se_total_le_conditional_target_diagnostic"
                            ].astype(bool).mean()
                        ),
                        "median_order_path_sd": order_stats["median"],
                        "p90_order_path_sd": order_stats["p90"],
                        "max_order_path_sd": order_stats["max"],
                        "all_order_seed_runs_valid": bool(
                            order_rows["all_seed_runs_valid"].astype(bool).all()
                        ),
                        "order_path_gate": bool(
                            order_rows["all_seed_attempts_present"].astype(bool).all()
                            and order_rows["all_seed_runs_valid"].astype(bool).all()
                            and order_stats["median"] is not None
                            and order_stats["median"] <= candidate.conditional_se_target
                        ),
                        "order_path_median_le_conditional_target": bool(
                            order_stats["median"] is not None
                            and order_stats["median"] <= candidate.conditional_se_target
                        ),
                        "recovery_n": recovery["n"],
                        "recovery_correlation": recovery["r"],
                        "recovery_slope": recovery["slope"],
                        "recovery_mae": recovery["mae"],
                        "comparison_scope": (
                            "fold-local finalists on identical outer models, "
                            "bootstrap draws, and order seeds"
                        ),
                    }
                )
        frame = pd.DataFrame(rows).sort_values(
            ["outer_fold", "candidate_id"]
        ).reset_index(drop=True)
        efficient: list[bool] = []
        for index, row in frame.iterrows():
            values = np.asarray(
                [row["oof_p90_scenario_count"], row["p90_se_total"], row["median_order_path_sd"]],
                dtype=float,
            )
            if not np.all(np.isfinite(values)):
                efficient.append(False)
                continue
            dominated = False
            for other_index, other in frame.iterrows():
                if other_index == index or int(other["outer_fold"]) != int(row["outer_fold"]):
                    continue
                other_values = np.asarray(
                    [
                        other["oof_p90_scenario_count"],
                        other["p90_se_total"],
                        other["median_order_path_sd"],
                    ],
                    dtype=float,
                )
                if (
                    np.all(np.isfinite(other_values))
                    and np.all(other_values <= values)
                    and np.any(other_values < values)
                ):
                    dominated = True
                    break
            efficient.append(not dominated)
        frame["pareto_efficient"] = efficient
        return frame

    def _policy_payload(
        self, pareto: pd.DataFrame, total_se: pd.DataFrame, order: pd.DataFrame
    ) -> dict[str, Any]:
        required = {
            "outer_fold",
            "candidate_id",
            "phase3_inner_p90_scenario_count",
            "phase3_inner_mean_scenario_count",
            "p90_se_total",
            "selector",
            "valid_bootstrap_gate_all_models",
            "order_path_gate",
        }
        missing = sorted(required - set(pareto.columns))
        if missing:
            raise Phase4Error(f"Phase-4 policy table lacks columns {missing}")
        candidates = pareto.copy()
        candidates["technical_gates_pass"] = (
            pd.to_numeric(candidates["p90_se_total"], errors="coerce").notna()
            & candidates["valid_bootstrap_gate_all_models"].astype(bool)
            & candidates["order_path_gate"].astype(bool)
        )
        fold_choices: list[dict[str, Any]] = []
        for panel in self.prereq.panels:
            fold_rows = candidates[
                pd.to_numeric(candidates["outer_fold"], errors="coerce")
                == panel.outer_fold
            ].copy()
            if set(fold_rows["candidate_id"].astype(str)) != set(panel.candidate_ids):
                raise Phase4Error(f"outer fold {panel.outer_fold} policy rows changed finalists")
            p90_lengths = pd.to_numeric(
                fold_rows["phase3_inner_p90_scenario_count"], errors="coerce"
            )
            mean_lengths = pd.to_numeric(
                fold_rows["phase3_inner_mean_scenario_count"], errors="coerce"
            )
            if p90_lengths.isna().any() or mean_lengths.isna().any():
                raise Phase4Error(f"outer fold {panel.outer_fold} lacks Phase-3 length evidence")
            if p90_lengths.nunique() != 1 or mean_lengths.nunique() != 1:
                raise Phase4Error(
                    f"outer fold {panel.outer_fold} finalists are not an exact Phase-3 length tie"
                )
            eligible = fold_rows[fold_rows["technical_gates_pass"]].copy()
            ranked = sorted(
                fold_rows.to_dict(orient="records"),
                key=lambda row: (
                    0 if bool(row["technical_gates_pass"]) else 1,
                    (
                        float(row["p90_se_total"])
                        if row.get("p90_se_total") is not None
                        and math.isfinite(float(row["p90_se_total"]))
                        else math.inf
                    ),
                    0 if str(row["selector"]) == "trace" else 1,
                    str(row["candidate_id"]),
                ),
            )
            provisional_ranking = [
                {
                    "rank": index,
                    "candidate_id": str(row["candidate_id"]),
                    "technical_gates_pass": bool(row["technical_gates_pass"]),
                    "p90_se_total": row.get("p90_se_total"),
                    "selector": str(row["selector"]),
                }
                for index, row in enumerate(ranked, start=1)
            ]
            selection_trace: list[dict[str, Any]] = [
                {
                    "step": "phase3_exact_length_tie_already_applied",
                    "p90_scenario_count": float(p90_lengths.iloc[0]),
                    "mean_scenario_count": float(mean_lengths.iloc[0]),
                    "remaining_candidate_ids": list(panel.candidate_ids),
                }
            ]
            if not eligible.empty:
                minimum_p90_total_se = float(eligible["p90_se_total"].min())
                eligible = eligible[
                    pd.to_numeric(eligible["p90_se_total"], errors="coerce")
                    == minimum_p90_total_se
                ]
                selection_trace.append(
                    {
                        "step": "lowest_p90_se_total",
                        "exact_value": minimum_p90_total_se,
                        "remaining_candidate_ids": sorted(
                            eligible["candidate_id"].astype(str)
                        ),
                    }
                )
                if (eligible["selector"].astype(str) == "trace").any():
                    eligible = eligible[eligible["selector"].astype(str) == "trace"]
                selection_trace.append(
                    {
                        "step": "prefer_trace",
                        "remaining_candidate_ids": sorted(
                            eligible["candidate_id"].astype(str)
                        ),
                    }
                )
                eligible = eligible.sort_values("candidate_id")
            leader = str(eligible.iloc[0]["candidate_id"]) if not eligible.empty else None
            fold_choices.append(
                {
                    "outer_fold": panel.outer_fold,
                    "status": (
                        "provisional_choice_not_frozen"
                        if leader is not None
                        else "no_candidate_passed_phase4_technical_gates"
                    ),
                    "provisional_candidate_id": leader,
                    "provisional_choice_uses_outer_outcomes": True,
                    "not_for_unbiased_fold_performance_estimation": True,
                    "local_finalist_candidate_ids": list(panel.candidate_ids),
                    "provisional_ranking": provisional_ranking,
                    "selection_trace": selection_trace,
                }
            )

        chosen_pairs = {
            (int(row["outer_fold"]), str(row["provisional_candidate_id"]))
            for row in fold_choices
            if row["provisional_candidate_id"] is not None
        }
        all_folds_have_choice = len(chosen_pairs) == len(self.prereq.panels)
        aggregate: dict[str, Any] | None = None
        if all_folds_have_choice:
            def chosen_mask(frame: pd.DataFrame) -> pd.Series:
                return pd.Series(
                    [
                        (int(fold), str(candidate)) in chosen_pairs
                        for fold, candidate in zip(
                            frame["outer_fold"], frame["candidate_id"], strict=True
                        )
                    ],
                    index=frame.index,
                    dtype=bool,
                )

            oof = self.prereq.outer_rows[chosen_mask(self.prereq.outer_rows)].copy()
            selected_total = total_se[chosen_mask(total_se)].copy()
            selected_order = order[chosen_mask(order)].copy()
            selected_summaries = candidates[chosen_mask(candidates)].copy()
            expected_models = set(map(str, self.prereq.matrix.index))
            if (
                len(oof) != len(expected_models)
                or set(oof["model"].astype(str)) != expected_models
                or len(selected_total) != len(expected_models) * self.prereq.structure.n_dims
                or len(selected_order) != len(expected_models)
            ):
                raise Phase4Error(
                    "provisional per-fold choices do not aggregate to one nested OOF row per model"
                )
            recovery = scenario_cv.recovery_stats(
                pd.to_numeric(oof["theta_reference"], errors="coerce"),
                pd.to_numeric(oof["theta_cat_mwle"], errors="coerce"),
            )
            lengths = pd.to_numeric(oof["cat_scenarios_administered"], errors="coerce")
            aggregate = {
                "scope": (
                    "descriptive, selection-conditioned aggregate after choosing each "
                    "fold leader on that same fold's outer total-uncertainty outcomes"
                ),
                "provisional_choice_uses_outer_outcomes": True,
                "not_an_unbiased_nested_performance_estimate": True,
                "n_models": len(expected_models),
                "mean_scenario_count": (
                    float(lengths.mean()) if lengths.notna().any() else None
                ),
                "p90_scenario_count": (
                    float(np.percentile(lengths.dropna(), 90))
                    if lengths.notna().any()
                    else None
                ),
                "total_se": self._finite_summary(selected_total["se_total"]),
                "order_path_sd": self._finite_summary(selected_order["order_path_sd"]),
                "recovery": recovery,
                "all_valid_bootstrap_gates_pass": bool(
                    selected_total["valid_draw_gate"].astype(bool).all()
                ),
                "all_order_path_gates_pass": bool(
                    selected_summaries["order_path_gate"].astype(bool).all()
                ),
            }

        additional_blockers = []
        if not all_folds_have_choice:
            additional_blockers.append("one_or_more_folds_failed_phase4_technical_gates")
        return {
            "schema_version": SCRIPT_SCHEMA,
            "generated_at": _utcnow(),
            "status": "blocked_missing_preregistered_total_se_tolerance",
            "final_policy": None,
            "provisional_fold_choices_not_frozen": fold_choices,
            "provisional_selection_conditioned_outer_aggregate": aggregate,
            "provisional_choice_uses_outer_outcomes": True,
            "not_an_unbiased_nested_performance_estimate": True,
            "additional_blockers": additional_blockers,
            "selection_order": [
                "lowest_p90_scenario_count",
                "lowest_mean_scenario_count",
                "lowest_p90_se_total",
                "prefer_trace",
                "candidate_id_lexical",
            ],
            "length_tie_policy": "exact_numeric_equality",
            "total_uncertainty_tiebreak_metric": "p90_se_total",
            "distinct_total_se_tolerance": None,
            "final_policy_freeze_allowed": False,
            "missing_tolerance_reason": (
                "The frozen remediation config does not define a distinct preregistered "
                "total-SE acceptance tolerance; the conditional online stopping target "
                "cannot be silently reused as that gate."
            ),
            "comparison_scope": (
                "candidate comparisons are fold-local only; candidate IDs are never "
                "compared across unequal outer-fold subsets"
            ),
            "phase5_note": (
                "A production configuration must be selected later using the all-52 "
                "inner-CV procedure; this Phase-4 provisional ranking cannot estimate "
                "its own performance without selection conditioning."
            ),
            "candidate_metrics": candidates.to_dict(orient="records"),
        }

    def _summary(
        self,
        draws: pd.DataFrame,
        seed_runs: pd.DataFrame,
        total_se: pd.DataFrame,
        order: pd.DataFrame,
        pareto: pd.DataFrame,
        policy: Mapping[str, Any],
    ) -> dict[str, Any]:
        se_stats = self._finite_summary(total_se["se_total"])
        order_stats = self._finite_summary(order["order_path_sd"])
        return {
            "schema_version": SCRIPT_SCHEMA,
            "generated_at": _utcnow(),
            "scope": (
                "fold-local finalist panels on identical support within each fold; "
                "the provisional post-choice aggregate is selection-conditioned and descriptive"
            ),
            "provisional_choice_uses_outer_outcomes": True,
            "not_an_unbiased_nested_performance_estimate": True,
            "n_outer_folds": len(self.prereq.panels),
            "n_outer_test_models": int(total_se["model"].nunique()),
            "n_local_finalists_by_fold": {
                str(panel.outer_fold): len(panel.candidates) for panel in self.prereq.panels
            },
            "parameter_bootstrap_replicates_per_fold": self.n_boot,
            "order_seed_count": len(self.order_seeds),
            "bootstrap_draw_rows": int(len(draws)),
            "order_seed_run_rows": int(len(seed_runs)),
            "total_se": {
                **se_stats,
                "minimum_valid_draw_rate": float(total_se["valid_draw_rate"].min()),
                "all_models_pass_valid_draw_gate": bool(total_se["valid_draw_gate"].all()),
                "proportion_at_or_below_conditional_target_diagnostic": float(
                    total_se["se_total_le_conditional_target_diagnostic"].astype(bool).mean()
                ),
                "method": "law of total variance",
                "distinct_acceptance_tolerance": None,
            },
            "order_path_sd": {
                **order_stats,
                "all_models_have_every_seed_attempt": bool(
                    order["all_seed_attempts_present"].astype(bool).all()
                ),
                "all_seed_runs_valid": bool(order["all_seed_runs_valid"].astype(bool).all()),
                "all_candidate_order_gates_pass": bool(
                    pareto["order_path_gate"].astype(bool).all()
                ),
                "reported_separately_from_total_se": True,
            },
            "pareto_candidates": pareto.to_dict(orient="records"),
            "policy": dict(policy),
            "policy_gate_status": {
                "valid_parameter_bootstrap": bool(total_se["valid_draw_gate"].all()),
                "order_stability": bool(pareto["order_path_gate"].astype(bool).all()),
                "total_uncertainty": (
                    "blocked: remediation config does not define a distinct "
                    "preregistered total-SE acceptance tolerance"
                ),
                "automatic_policy_freeze": False,
            },
            "total_uncertainty_tiebreak_metric": "p90_se_total",
            "distinct_total_se_tolerance": None,
            "final_policy_freeze_allowed": False,
            "missing_tolerance_reason": policy["missing_tolerance_reason"],
        }

    def run(self) -> int:
        if self.args.plan_only:
            print(json.dumps(self.plan_payload(), indent=2, sort_keys=True))
            return 0
        self._prepare_output()
        existing = self.nested_dir / "phase4_manifest.json"
        if self.args.resume and existing.is_file():
            current = _read_json(existing)
            if current.get("status") == "complete":
                print(f"Phase-4 study already complete -> {self.nested_dir}")
                return 0
        _atomic_json(self.nested_dir / "phase4_manifest.json", self._base_manifest("running"))
        (self.nested_dir / "phase4_cache").mkdir(parents=True, exist_ok=True)
        (self.nested_dir / "phase4_checkpoints").mkdir(parents=True, exist_ok=True)
        draws = self._run_bootstraps()
        seed_runs = self._run_order_seeds()
        assert_balanced_panel_support(
            draws,
            panels=self.prereq.panels,
            index_column="replicate",
            index_values=range(self.n_boot),
        )
        assert_balanced_panel_support(
            seed_runs,
            panels=self.prereq.panels,
            index_column="seed",
            index_values=self.order_seeds,
        )
        total_se = self._total_se_frame(draws)
        order = self._order_frame(seed_runs)
        pareto = self._pareto_frame(total_se, order)
        policy = self._policy_payload(pareto, total_se, order)
        summary = self._summary(draws, seed_runs, total_se, order, pareto, policy)

        _atomic_csv(self.nested_dir / "outer_total_se.csv", total_se)
        _atomic_csv(self.nested_dir / "outer_order_stability.csv", order)
        _atomic_csv(self.nested_dir / "phase4_pareto.csv", pareto)
        _atomic_json(self.nested_dir / "phase4_policy.json", policy)
        _atomic_json(self.nested_dir / "phase4_summary.json", summary)
        manifest = self._base_manifest("complete")
        manifest["completed_at"] = _utcnow()
        manifest["outputs"] = {
            name: {
                "path": str(self.nested_dir / name),
                "sha256": _sha256(self.nested_dir / name),
            }
            for name in FINAL_OUTPUTS
            if name != "phase4_manifest.json"
        }
        manifest["intermediate_artifacts"] = {
            "bootstrap_draws": "phase4_checkpoints/bootstrap/*.json",
            "order_seed_runs": "phase4_checkpoints/order/*.json",
            "bootstrap_fit_cache": "phase4_cache/bootstrap_fits/",
        }
        manifest["completed_artifact_inventories"] = {}
        for name in ("phase4_checkpoints", "phase4_cache"):
            inventory = _tree_inventory(self.nested_dir / name)
            manifest["completed_artifact_inventories"][name] = {
                "path": name,
                "file_count": len(inventory),
                "inventory_sha256": _canonical_hash(inventory),
            }
        manifest["policy_status"] = policy["status"]
        manifest["total_uncertainty_tiebreak_metric"] = "p90_se_total"
        manifest["distinct_total_se_tolerance"] = None
        manifest["final_policy_freeze_allowed"] = False
        manifest["provisional_choice_uses_outer_outcomes"] = True
        manifest["not_an_unbiased_nested_performance_estimate"] = True
        manifest["missing_tolerance_reason"] = policy["missing_tolerance_reason"]
        _atomic_json(self.nested_dir / "phase4_manifest.json", manifest)
        print(f"completed Phase-4 total uncertainty and order stability -> {self.nested_dir}")
        return 0


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--nested-dir", type=Path, default=DEFAULT_NESTED_DIR)
    parser.add_argument("--n-boot", type=int, default=None)
    parser.add_argument("--bootstrap-seed", type=int, default=20260804)
    parser.add_argument(
        "--order-seeds",
        default=None,
        help="comma-separated override; at least 20 unique seeds are required",
    )
    parser.add_argument("--cat-seed", type=int, default=20260729)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--max-scenarios", type=int, default=50)
    parser.add_argument("--minimum-scored-criteria", type=int, default=15)
    parser.add_argument("--mwle-ridge", type=float, default=1e-6)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--max-grid-nodes", type=int, default=50_000)
    parser.add_argument("--negative-policy", choices=("error", "drop", "keep"), default="drop")
    parser.add_argument(
        "--estimate-latent-corr", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--allow-unconverged-fit", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--fresh", action="store_true")
    return parser


def _validate_runtime(args: argparse.Namespace) -> None:
    if args.plan_only and (args.resume or args.fresh):
        raise Phase4Error("--plan-only cannot be combined with --resume or --fresh")
    if args.n_boot is not None and args.n_boot < 2:
        raise Phase4Error("--n-boot must be at least 2")
    if args.top_n < 1 or args.max_scenarios < 1 or args.minimum_scored_criteria < 0:
        raise Phase4Error("invalid CAT runtime limits")
    if args.mwle_ridge <= 0 or args.max_iter < 1 or args.tol <= 0:
        raise Phase4Error("invalid calibration/MWLE optimizer settings")
    if args.max_grid_nodes < 2:
        raise Phase4Error("--max-grid-nodes must be at least 2")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    try:
        _validate_runtime(args)
        return Phase4Runner(args).run()
    except (
        FileNotFoundError,
        KeyError,
        OSError,
        ValueError,
        Phase4Error,
        nested.NestedCVError,
        scat.OfflineStudyError,
        cm.CalibrationError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    finally:
        cm.configure_skills(None)


if __name__ == "__main__":
    raise SystemExit(main())
