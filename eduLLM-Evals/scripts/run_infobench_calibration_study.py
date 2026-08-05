#!/usr/bin/env python3
"""Run the complete InFoBench calibration playbook on the full response matrix.

This is an orchestrator, not a second IRT implementation.  It calls the audited
single-purpose scripts in the order required by
``plans+prds/Calibration Playbook - Cross-Benchmark.md``:

1. full-data core M2PL fits for predeclared latent-skill structures;
2. leakage-free person-level k-fold validation for every structure;
3. deterministic structure selection (one-standard-error rule, then parsimony);
4. grid/ridge sensitivity on the selected structure;
5. a fitted-only CAT-bank export (no synthetic fallback);
6. out-of-sample EAP/MWLE recovery;
7. scenario-floor, SE-target, selector, order, and CAT-vs-random studies;
8. parameter uncertainty / total SE and one consolidated report.

The default command runs every configured full-data stage.  It does not run a
subset, pilot, or smoke study.  ``--plan-only`` is a read-only command preview.
Every stage writes its exact argv and a completion marker, so ``--resume`` can
continue a long study without silently accepting partial outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tutor_cat.skill_structure import SkillStructure, parse_dimension_spec  # noqa: E402

DEFAULT_CONFIG = ROOT / "configs" / "infobench_calibration_study.json"


class StudyError(RuntimeError):
    pass


@dataclass(frozen=True)
class Candidate:
    name: str
    dimensions: str
    status: str
    structure: SkillStructure


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StudyError(f"could not read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise StudyError(f"expected one JSON object in {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _matrix_dimensions(path: Path) -> tuple[int, int]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        model_rows = sum(1 for row in reader if row)
    if not header or header[0] != "model":
        raise StudyError("response matrix must begin with a model column")
    return model_rows, len(header) - 1


def _slug_number(value: float | int) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def load_and_validate_config(path: Path) -> tuple[dict[str, Any], list[Candidate]]:
    config = _read_json(path)
    if config.get("schema_version") != "calibration-study-v1":
        raise StudyError("config schema_version must be calibration-study-v1")
    if config.get("benchmark") != "InFoBench":
        raise StudyError("this runner requires benchmark=InFoBench")

    inputs = config.get("inputs") or {}
    for key in ("matrix", "rubrics", "scenarios", "judge_manifest"):
        if key not in inputs:
            raise StudyError(f"config inputs is missing {key!r}")
        resolved = _resolve(inputs[key])
        if not resolved.is_file():
            raise StudyError(f"configured {key} does not exist: {resolved}")

    expectations = config.get("input_expectations") or {}
    if expectations:
        matrix_path = _resolve(inputs["matrix"])
        model_rows, n_criteria = _matrix_dimensions(matrix_path)
        if model_rows != int(expectations["models"]):
            raise StudyError(
                f"response matrix has {model_rows} models, expected {expectations['models']}"
            )
        if n_criteria != int(expectations["criteria"]):
            raise StudyError(
                f"response matrix has {n_criteria} criteria, expected "
                f"{expectations['criteria']}"
            )
        scenario_rows = sum(
            1 for line in _resolve(inputs["scenarios"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if scenario_rows != int(expectations["scenarios"]):
            raise StudyError(
                f"scenario bank has {scenario_rows} rows, expected {expectations['scenarios']}"
            )
        judge = _read_json(_resolve(inputs["judge_manifest"]))
        observed = judge.get("judge_expected") or {}
        required_judge = {
            "judge_name": expectations.get("judge_name"),
            "judge_model": expectations.get("judge_model"),
            "adapter": expectations.get("judge_adapter"),
        }
        for key, expected in required_judge.items():
            if expected is not None and observed.get(key) != expected:
                raise StudyError(
                    f"judge manifest {key}={observed.get(key)!r}, expected {expected!r}"
                )
        policy = ((judge.get("counts") or {}).get("this_run") or {}).get(
            "no_decision_policy"
        )
        expected_policy = expectations.get("no_decision_policy")
        if expected_policy is not None and policy != expected_policy:
            raise StudyError(
                f"judge no_decision_policy={policy!r}, expected {expected_policy!r}"
            )

    source_skills = tuple(str(value).strip() for value in config.get("source_skills", ()))
    if not source_skills or len(set(source_skills)) != len(source_skills):
        raise StudyError("source_skills must be a non-empty unique ordered list")

    candidates: list[Candidate] = []
    names: set[str] = set()
    for raw in config.get("structures") or []:
        name = str(raw.get("name") or "").strip()
        dimensions = str(raw.get("dimensions") or "").strip()
        if not name or name in names:
            raise StudyError(f"blank or duplicate structure name: {name!r}")
        try:
            structure = parse_dimension_spec(dimensions, source_skills, name=name)
        except ValueError as exc:
            raise StudyError(f"invalid structure {name}: {exc}") from exc
        names.add(name)
        candidates.append(
            Candidate(name, dimensions, str(raw.get("status") or "unspecified"), structure)
        )
    if len(candidates) < 2:
        raise StudyError("at least two latent structures are required for comparison")

    fit = config.get("fit") or {}
    if int(fit.get("grid", 0)) < 2 or float(fit.get("ridge", -1)) < 0:
        raise StudyError("fit.grid must be >=2 and fit.ridge must be nonnegative")
    kfold = config.get("kfold") or {}
    if int(kfold.get("folds", 0)) < 2:
        raise StudyError("kfold.folds must be >=2")
    fraction = float(kfold.get("evaluation_scenario_fraction", 0))
    if not 0 < fraction < 1:
        raise StudyError("kfold.evaluation_scenario_fraction must lie strictly between 0 and 1")

    return config, candidates


class Orchestrator:
    def __init__(
        self,
        config_path: Path,
        config: dict[str, Any],
        candidates: list[Candidate],
        out_dir: Path,
        *,
        resume: bool,
        plan_only: bool,
    ) -> None:
        self.config_path = config_path
        self.config = config
        self.candidates = candidates
        self.out_dir = out_dir
        self.resume = resume
        self.plan_only = plan_only
        self.python = Path(sys.executable)
        self.commands: list[dict[str, Any]] = []

        self.inputs = {key: _resolve(value) for key, value in config["inputs"].items()}
        self.source_skills = tuple(config["source_skills"])
        self.fit = config["fit"]
        self.kfold = config["kfold"]

    def prepare(self) -> None:
        if self.out_dir.exists() and not self.resume and not self.plan_only:
            raise StudyError(
                f"output directory already exists: {self.out_dir}; choose a fresh path "
                "or pass --resume"
            )
        if not self.plan_only:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.config_path, self.out_dir / "study_config.json")
            model_rows, n_criteria = _matrix_dimensions(self.inputs["matrix"])
            provenance = {
                "generated_at": _utcnow(),
                "runner": "scripts/run_infobench_calibration_study.py",
                "runner_argv": sys.argv[1:],
                "python": str(self.python),
                "inputs": {
                    key: {"path": str(path), "sha256": _sha256(path)}
                    for key, path in self.inputs.items()
                },
                "source_skills": list(self.source_skills),
                "full_data_only": True,
                "pilot_or_subset": False,
            }
            _write_json(self.out_dir / "study_manifest.json", provenance)
            _write_json(
                self.out_dir / "playbook_coverage.json",
                {
                    "source": "plans+prds/Calibration Playbook - Cross-Benchmark.md",
                    "stages": {
                        "response_matrix_and_policy": {
                            "status": "reused_and_hash_verified",
                            "artifact": str(self.inputs["judge_manifest"]),
                        },
                        "native_q_matrix": {
                            "status": "preserved",
                            "artifact": str(self.inputs["rubrics"]),
                            "note": "Reduced structures are in-memory OR partitions only.",
                        },
                        "core_m2pl_and_dimensionality": "scheduled",
                        "person_kfold_cell_prediction": "scheduled_leakage_free",
                        "oos_theta_recovery_eap_mwle": "scheduled",
                        "grid_and_ridge_sensitivity": "scheduled",
                        "min_scenarios_and_se_sweeps": "scheduled",
                        "trace_vs_dopt": "scheduled",
                        "order_seed_dependence": "scheduled",
                        "parameter_uncertainty_and_total_se": "scheduled",
                        "cat_vs_seeded_random": "scheduled",
                        "model_bootstrap_confidence_intervals": "scheduled",
                        "fitted_bank_safety_export": {
                            "status": "scheduled",
                            "policy": (
                                "Exclude unfitted, nonpositive-loading, and extreme-loading "
                                "criteria; never substitute synthetic parameters."
                            ),
                        },
                        "predeclared_reverse_item_exclusions": {
                            "status": "none_in_current_source_bank",
                            "note": (
                                "calibrate_mirt honors exclude_from_fit when present. The "
                                "current InFoBench bank has no such flags, so unstable items "
                                "are removed at the fitted-bank export gate rather than "
                                "retroactively changing the frozen source Q-matrix."
                            ),
                        },
                        "efa": {
                            "status": "not_used_for_selection",
                            "reason": (
                                f"With {model_rows} persons and {n_criteria:,} binary "
                                "criteria, an "
                                "item-level EFA is underidentified; reduced structures are "
                                "instead decided by disjoint-scenario OOS performance."
                            ),
                        },
                        "within_scenario_local_dependence": {
                            "status": "known_limitation_not_modeled_as_testlets",
                            "mitigation": (
                                "Primary validation holds out whole scenarios, but the core "
                                "criterion-level M2PL still assumes conditional independence."
                            ),
                        },
                        "infobench_specific_human_judge_audit": {
                            "status": (self.config.get("external_validation") or {}).get(
                                "infobench_specific_human_judge_audit", "unspecified"
                            ),
                            "note": (self.config.get("external_validation") or {}).get("note"),
                        },
                    },
                },
            )

    def run_command(self, stage: str, command: list[str], outputs: Iterable[Path]) -> None:
        stage_dir = self.out_dir / "stage_markers"
        marker = stage_dir / f"{stage}.json"
        expected = [Path(path) for path in outputs]
        entry = {"stage": stage, "command": command, "outputs": [str(path) for path in expected]}
        self.commands.append(entry)
        if self.plan_only:
            print(" ".join(command))
            return
        if self.resume and marker.is_file() and all(path.exists() for path in expected):
            previous = _read_json(marker)
            if previous.get("command") != command:
                raise StudyError(
                    f"resume marker command differs for {stage}; use a fresh output directory"
                )
            print(f"[resume] {stage}")
            return

        stage_dir.mkdir(parents=True, exist_ok=True)
        log_path = stage_dir / f"{stage}.log"
        print(f"\n[{stage}]", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            log.write("COMMAND\n" + json.dumps(command) + "\n\nOUTPUT\n")
            proc = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="")
                log.write(line)
            returncode = proc.wait()
        if returncode != 0:
            raise StudyError(f"stage {stage} failed with exit code {returncode}; see {log_path}")
        missing = [str(path) for path in expected if not path.exists()]
        if missing:
            raise StudyError(f"stage {stage} succeeded but expected outputs are missing: {missing}")
        _write_json(
            marker,
            {
                **entry,
                "completed_at": _utcnow(),
                "output_sha256": {str(path): _sha256(path) for path in expected if path.is_file()},
            },
        )

    def fit_command(
        self, candidate: Candidate, out_dir: Path, grid: int, ridge: float
    ) -> list[str]:
        command = [
            str(self.python),
            "scripts/calibrate_mirt.py",
            "--matrix", str(self.inputs["matrix"]),
            "--rubrics", str(self.inputs["rubrics"]),
            "--source-skills", ",".join(self.source_skills),
            "--dimensions", candidate.dimensions,
            "--structure-name", candidate.name,
            "--require-complete-bank",
            "--grid", str(grid),
            "--max-grid-nodes", str(self.fit.get("max_grid_nodes", 5000)),
            "--ridge", str(ridge),
            "--max-iter", str(self.fit.get("max_iterations", 200)),
            "--tol", str(self.fit.get("tolerance", 1e-4)),
            "--min-persons-identifiable", str(self.fit.get("minimum_person_warning", 150)),
            "--out-dir", str(out_dir),
        ]
        if self.fit.get("estimate_latent_correlation", True):
            command.append("--estimate-latent-corr")
        return command

    def kfold_command(
        self, candidate: Candidate, out_dir: Path, grid: int, ridge: float
    ) -> list[str]:
        command = [
            str(self.python),
            "scripts/kfold_cv_mirt.py",
            "--matrix", str(self.inputs["matrix"]),
            "--rubrics", str(self.inputs["rubrics"]),
            "--skills", ",".join(self.source_skills),
            "--dimensions", candidate.dimensions,
            "--structure-name", candidate.name,
            "--require-complete-bank",
            "--k", str(self.kfold["folds"]),
            "--grid", str(grid),
            "--max-grid-nodes", str(self.fit.get("max_grid_nodes", 5000)),
            "--ridge", str(ridge),
            "--max-iter", str(self.fit.get("max_iterations", 200)),
            "--tol", str(self.fit.get("tolerance", 1e-4)),
            "--seed", str(self.kfold["seed"]),
            "--fold-strategy", str(self.kfold.get("strategy", "stratified")),
            "--evaluation-scenario-fraction",
            str(self.kfold.get("evaluation_scenario_fraction", 0.2)),
            "--evaluation-split-seed",
            str(self.kfold.get("evaluation_split_seed", self.kfold["seed"])),
            "--out-dir", str(out_dir),
        ]
        if self.fit.get("estimate_latent_correlation", True):
            command.append("--estimate-latent-corr")
        return command

    def run_structure_studies(self) -> None:
        grid = int(self.fit["grid"])
        ridge = float(self.fit["ridge"])
        for candidate in self.candidates:
            base = self.out_dir / "structures" / candidate.name
            fit_dir = base / "fit"
            kfold_dir = base / "kfold"
            self.run_command(
                f"fit_{candidate.name}",
                self.fit_command(candidate, fit_dir, grid, ridge),
                [fit_dir / "calibration_mirt.csv", fit_dir / "calibration_mirt_manifest.json"],
            )
            self.run_command(
                f"kfold_{candidate.name}",
                self.kfold_command(candidate, kfold_dir, grid, ridge),
                [
                    kfold_dir / "metrics_aggregate.json",
                    kfold_dir / "metrics_per_fold.csv",
                    kfold_dir / "item_param_stability.json",
                    kfold_dir / "kfold_summary.json",
                ],
            )

    def collect_and_select_structure(self) -> Candidate | None:
        if self.plan_only:
            return None
        rows: list[dict[str, Any]] = []
        for candidate in self.candidates:
            base = self.out_dir / "structures" / candidate.name
            manifest = _read_json(base / "fit" / "calibration_mirt_manifest.json")
            kmetrics = _read_json(base / "kfold" / "metrics_aggregate.json")
            stability = _read_json(base / "kfold" / "item_param_stability.json")["stability"]
            per_fold_path = base / "kfold" / "metrics_per_fold.csv"
            with per_fold_path.open(newline="", encoding="utf-8") as handle:
                fold_rows = list(csv.DictReader(handle))
            fold_losses = [float(row["log_loss"]) for row in fold_rows]
            all_fold_fits_converged = bool(fold_rows) and all(
                str(row["fit_converged"]).strip().lower() in {"true", "1"}
                for row in fold_rows
            )
            mean_loss = sum(fold_losses) / len(fold_losses)
            variance = (
                sum((value - mean_loss) ** 2 for value in fold_losses)
                / max(1, len(fold_losses) - 1)
            )
            se_loss = math.sqrt(variance) / math.sqrt(len(fold_losses))
            stable_values = [
                float(value["median_pairwise_corr"])
                for value in stability.values()
                if value.get("median_pairwise_corr") is not None
                and math.isfinite(float(value["median_pairwise_corr"]))
            ]
            comparison = manifest["comparison"]["multi"]
            pooled = kmetrics["pooled_oos"]
            rows.append(
                {
                    "structure": candidate.name,
                    "status": candidate.status,
                    "n_dims": candidate.structure.n_dims,
                    "fit_converged": bool(manifest["em"]["multi_converged"]),
                    "all_fold_fits_converged": all_fold_fits_converged,
                    "loglik": comparison["loglik"],
                    "n_params": comparison["n_params"],
                    "aic": comparison["aic"],
                    "bic": comparison["bic"],
                    "oos_log_loss": pooled["log_loss"],
                    "oos_log_loss_fold_mean": mean_loss,
                    "oos_log_loss_fold_se": se_loss,
                    "oos_brier": pooled["brier"],
                    "oos_accuracy": pooled["accuracy"],
                    "oos_auc": pooled["auc"],
                    "median_parameter_stability": (
                        sum(stable_values) / len(stable_values) if stable_values else None
                    ),
                }
            )
        comparison_path = self.out_dir / "structure_comparison.csv"
        with comparison_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        eligible = [
            row
            for row in rows
            if row["fit_converged"] and row["all_fold_fits_converged"]
        ]
        if not eligible:
            raise StudyError("no candidate structure has a converged core fit and all folds")
        override = (self.config.get("structure_selection") or {}).get(
            "selected_structure_override"
        )
        if override:
            selected_row = next((row for row in eligible if row["structure"] == override), None)
            if selected_row is None:
                raise StudyError(
                    "selected_structure_override is absent or lacks a converged "
                    f"core/all-fold fit: {override}"
                )
            threshold = None
            qualifying = [selected_row["structure"]]
        else:
            best = min(eligible, key=lambda row: row["oos_log_loss_fold_mean"])
            threshold = best["oos_log_loss_fold_mean"] + best["oos_log_loss_fold_se"]
            within_one_se = [
                row for row in eligible if row["oos_log_loss_fold_mean"] <= threshold
            ]
            selected_row = min(
                within_one_se,
                key=lambda row: (row["n_dims"], row["oos_log_loss_fold_mean"]),
            )
            qualifying = [row["structure"] for row in within_one_se]
        selected = next(c for c in self.candidates if c.name == selected_row["structure"])
        _write_json(
            self.out_dir / "structure_selection.json",
            {
                "generated_at": _utcnow(),
                "rule": (self.config.get("structure_selection") or {}).get("rule"),
                "primary_metric": "heldout-person, disjoint-scenario OOS log loss",
                "selected_structure": selected.name,
                "selected_dimensions": selected.structure.as_dict(),
                "one_standard_error_threshold": threshold,
                "structures_within_threshold": qualifying,
                "override_used": bool(override),
                "note": (
                    "The five native InFoBench tags remain unchanged. This selects only "
                    "the latent ability structure used for calibration and CAT."
                ),
            },
        )
        print(f"\nselected latent structure: {selected.name} ({selected.structure.n_dims}D)")
        return selected

    def run_sensitivity(self, selected: Candidate) -> None:
        base_grid = int(self.fit["grid"])
        base_ridge = float(self.fit["ridge"])
        sensitivity = self.config.get("sensitivity") or {}
        for grid in sensitivity.get("grid_values", [base_grid]):
            for ridge in sensitivity.get("ridge_values", [base_ridge]):
                grid = int(grid)
                ridge = float(ridge)
                if grid == base_grid and math.isclose(ridge, base_ridge):
                    continue
                if grid ** selected.structure.n_dims > int(self.fit.get("max_grid_nodes", 5000)):
                    continue
                label = f"grid{grid}_ridge{_slug_number(ridge)}"
                base = self.out_dir / "sensitivity" / label
                self.run_command(
                    f"sensitivity_fit_{label}",
                    self.fit_command(selected, base / "fit", grid, ridge),
                    [
                        base / "fit" / "calibration_mirt.csv",
                        base / "fit" / "calibration_mirt_manifest.json",
                    ],
                )
                self.run_command(
                    f"sensitivity_kfold_{label}",
                    self.kfold_command(selected, base / "kfold", grid, ridge),
                    [
                        base / "kfold" / "metrics_aggregate.json",
                        base / "kfold" / "metrics_per_fold.csv",
                        base / "kfold" / "item_param_stability.json",
                        base / "kfold" / "kfold_summary.json",
                    ],
                )

    def select_sensitivity(self, selected: Candidate) -> tuple[Path, int, float]:
        """Select ridge at the fixed production grid; other grids stay diagnostics."""
        sensitivity = self.config.get("sensitivity") or {}
        grid = int(sensitivity.get("production_grid", self.fit["grid"]))
        base_grid = int(self.fit["grid"])
        base_ridge = float(self.fit["ridge"])
        rows: list[dict[str, Any]] = []
        for ridge_value in sensitivity.get("ridge_values", [base_ridge]):
            ridge = float(ridge_value)
            if grid == base_grid and math.isclose(ridge, base_ridge):
                base = self.out_dir / "structures" / selected.name
            else:
                label = f"grid{grid}_ridge{_slug_number(ridge)}"
                base = self.out_dir / "sensitivity" / label
            kfold_dir = base / "kfold"
            fit_dir = base / "fit"
            if not (kfold_dir / "metrics_per_fold.csv").is_file():
                raise StudyError(f"missing ridge-sensitivity fold metrics: {kfold_dir}")
            with (kfold_dir / "metrics_per_fold.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                fold_rows = list(csv.DictReader(handle))
            manifest = _read_json(fit_dir / "calibration_mirt_manifest.json")
            core_converged = bool(manifest["em"]["multi_converged"])
            folds_converged = bool(fold_rows) and all(
                str(row["fit_converged"]).strip().lower() in {"true", "1"}
                for row in fold_rows
            )
            losses = [float(row["log_loss"]) for row in fold_rows]
            mean = sum(losses) / len(losses)
            variance = sum((value - mean) ** 2 for value in losses) / max(1, len(losses) - 1)
            rows.append(
                {
                    "grid": grid,
                    "ridge": ridge,
                    "cv_log_loss_mean": mean,
                    "cv_log_loss_se": math.sqrt(variance) / math.sqrt(len(losses)),
                    "core_converged": core_converged,
                    "all_fold_fits_converged": folds_converged,
                    "fit_dir": str(fit_dir),
                }
            )
        valid = [
            row
            for row in rows
            if row["core_converged"] and row["all_fold_fits_converged"]
        ]
        if not valid:
            raise StudyError(
                "no ridge candidate has a converged core fit and all fold fits"
            )
        best = min(valid, key=lambda row: row["cv_log_loss_mean"])
        boundary = best["cv_log_loss_mean"] + best["cv_log_loss_se"]
        eligible = [row for row in valid if row["cv_log_loss_mean"] <= boundary]
        chosen = max(eligible, key=lambda row: row["ridge"])
        for row in rows:
            row["one_se_eligible"] = row in eligible
            row["selected"] = row is chosen
        with (self.out_dir / "ridge_sensitivity_selection.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        _write_json(
            self.out_dir / "sensitivity_selection.json",
            {
                "rule": sensitivity.get("ridge_selection_rule"),
                "production_grid": grid,
                "best_mean_ridge": best["ridge"],
                "one_standard_error_boundary": boundary,
                "eligible_ridges": [row["ridge"] for row in eligible],
                "selected_ridge": chosen["ridge"],
                "selected_fit_dir": chosen["fit_dir"],
                "grid_sweep_role": "numerical sensitivity only; it does not tune on OOS data",
            },
        )
        return Path(chosen["fit_dir"]), grid, float(chosen["ridge"])

    def export_selected_bank(
        self, selected: Candidate, fit_dir: Path
    ) -> tuple[Path, Path]:
        bank_dir = self.out_dir / "selected_bank"
        rubrics = bank_dir / "rubrics_fitted_only.jsonl"
        scenarios = bank_dir / "scenarios_fitted_only.jsonl"
        manifest = bank_dir / "export_manifest.json"
        export = self.config.get("export") or {}
        command = [
            str(self.python),
            "scripts/export_fitted_bank.py",
            "--calibration-csv", str(fit_dir / "calibration_mirt.csv"),
            "--calibration-manifest", str(fit_dir / "calibration_mirt_manifest.json"),
            "--rubrics", str(self.inputs["rubrics"]),
            "--scenarios", str(self.inputs["scenarios"]),
            "--dimensions", selected.dimensions,
            "--structure-name", selected.name,
            "--out-rubrics", str(rubrics),
            "--out-scenarios", str(scenarios),
            "--out-manifest", str(manifest),
            "--nonpositive-policy", str(export.get("nonpositive_discrimination_policy", "exclude")),
            "--extreme-policy", str(export.get("extreme_discrimination_policy", "exclude")),
            "--extreme-a", str(export.get("extreme_discrimination_threshold", 6.0)),
        ]
        self.run_command("export_selected_bank", command, [rubrics, scenarios, manifest])
        return rubrics, scenarios

    def run_scenario_kfold(
        self,
        selected: Candidate,
        *,
        fit_grid: int,
        ridge: float,
        min_scenarios: int,
        max_se: float,
        selection: str,
    ) -> None:
        estimator = self.config.get("estimator") or {}
        out_dir = self.out_dir / "estimator_oos_kfold"
        command = [
            str(self.python),
            "scripts/scenario_kfold_estimator_cv.py",
            "--matrix", str(self.inputs["matrix"]),
            "--rubrics", str(self.inputs["rubrics"]),
            "--scenarios", str(self.inputs["scenarios"]),
            "--skills", ",".join(self.source_skills),
            "--dimensions", selected.dimensions,
            "--structure-name", selected.name,
            "--require-complete-bank",
            "--k", str(self.kfold["folds"]),
            "--seed", str(self.kfold["seed"]),
            "--fold-strategy", str(self.kfold.get("strategy", "stratified")),
            "--fit-grid", str(fit_grid),
            "--ridge", str(ridge),
            "--max-iter", str(self.fit.get("max_iterations", 200)),
            "--tol", str(self.fit.get("tolerance", 1e-4)),
            "--estimate-latent-corr",
            "--eap-grid", str(estimator.get("quadrature_grid", 5)),
            "--max-grid-nodes", str(estimator.get("max_grid_nodes", 50000)),
            "--negative-policy", "drop",
            "--selection", selection,
            "--top-n", str((self.config.get("cat") or {}).get("top_n", 5)),
            "--max-se", str(max_se),
            "--min-evals-per-skill",
            str((self.config.get("cat") or {}).get("min_evals_per_skill", 15)),
            "--min-scenarios", str(min_scenarios),
            "--max-scenarios",
            str((self.config.get("cat") or {}).get("adaptive_max_scenarios", 50)),
            "--mwle-ridge", str(estimator.get("mwle_ridge", 1e-6)),
            "--out-dir", str(out_dir),
        ]
        self.run_command(
            "scenario_kfold_estimators",
            command,
            [
                out_dir / "oos_per_model.csv",
                out_dir / "metrics_aggregate.json",
                out_dir / "pass_rate_calibration.csv",
                out_dir / "manifest.json",
            ],
        )

    def replay_command(
        self,
        bank: Path,
        scenarios: Path,
        out_dir: Path,
        *,
        min_scenarios: int,
        max_se: float,
        selection: str,
        mode: str = "cat",
        seed: int | None = None,
    ) -> list[str]:
        cat = self.config.get("cat") or {}
        estimator = self.config.get("estimator") or {}
        return [
            str(self.python),
            "scripts/offline_engine_driver.py",
            "--bank", str(bank),
            "--matrix", str(self.inputs["matrix"]),
            "--scenarios", str(scenarios),
            "--out-dir", str(out_dir),
            "--mode", mode,
            "--selection", selection,
            "--seed", str(cat.get("seed", 42) if seed is None else seed),
            "--top-n", str(cat.get("top_n", 5)),
            "--max-se", str(max_se),
            "--min-evals-per-skill", str(cat.get("min_evals_per_skill", 15)),
            "--min-scenarios", str(min_scenarios),
            "--max-scenarios", str(cat.get("adaptive_max_scenarios", 50)),
            "--baseline-max-scenarios", str(cat.get("random_max_scenarios", 0)),
            "--grid", str(estimator.get("quadrature_grid", 5)),
            "--max-grid-nodes", str(estimator.get("max_grid_nodes", 50000)),
            "--mwle-ridge", str(estimator.get("mwle_ridge", 1e-6)),
            "--negative-policy", "error",
        ]

    def run_cat_studies(self, bank: Path, scenarios: Path) -> None:
        cat = self.config.get("cat") or {}
        reference_se = 0.30
        for floor in cat.get("min_scenarios_values", [0, 12, 15, 20]):
            label = f"floor_{floor}"
            out = self.out_dir / "cat_sweeps" / "min_scenarios" / label
            self.run_command(
                f"cat_{label}",
                self.replay_command(
                    bank, scenarios, out, min_scenarios=int(floor),
                    max_se=reference_se, selection="trace",
                ),
                [out / "per_model.csv", out / "metrics.json"],
            )
        selected_floor = self._preselect_cat_floor()
        for target in cat.get("se_target_values", [0.2, 0.25, 0.3, 0.35]):
            label = f"se_{_slug_number(target)}"
            out = self.out_dir / "cat_sweeps" / "se_target" / label
            self.run_command(
                f"cat_{label}",
                self.replay_command(
                    bank, scenarios, out, min_scenarios=selected_floor,
                    max_se=float(target), selection="trace",
                ),
                [out / "per_model.csv", out / "metrics.json"],
            )
        selected_se = self._preselect_cat_se_target()
        for selector in cat.get("selection_rules", ["trace", "dopt"]):
            out = self.out_dir / "cat_sweeps" / "selection" / str(selector)
            self.run_command(
                f"cat_selection_{selector}",
                self.replay_command(
                    bank, scenarios, out, min_scenarios=selected_floor,
                    max_se=selected_se, selection=str(selector),
                ),
                [out / "per_model.csv", out / "metrics.json"],
            )

    @staticmethod
    def _cat_mode_summary(path: Path) -> dict[str, Any]:
        payload = _read_json(path / "metrics.json")
        try:
            return payload["modes"]["cat"]
        except (KeyError, TypeError) as exc:
            raise StudyError(f"CAT metrics are missing modes.cat: {path}") from exc

    @staticmethod
    def _macro_mwle_metric(
        summary: dict[str, Any], metric: str
    ) -> float | None:
        recovery = ((summary.get("recovery_vs_full_bank_eap") or {}).get("mwle") or {})
        values = [
            float(row[metric])
            for row in recovery.values()
            if row.get(metric) is not None and math.isfinite(float(row[metric]))
        ]
        return sum(values) / len(values) if values else None

    @staticmethod
    def _quality_eligible_cat_rows(
        rows: list[dict[str, Any]],
        *,
        required_precision: float,
        minimum_mwle_convergence: float,
        maximum_recovery_drop: float,
        maximum_slope_error_increase: float,
    ) -> list[dict[str, Any]]:
        """Retain CAT settings with adequate coverage and near-best recovery."""
        primary = [
            row
            for row in rows
            if row["precision_rate"] >= required_precision
            and row["mwle_convergence_rate"] >= minimum_mwle_convergence
        ]
        if not primary:
            return []
        finite_r = [row["mwle_recovery_r"] for row in primary if row["mwle_recovery_r"] is not None]
        if finite_r:
            best_r = max(finite_r)
            primary = [
                row
                for row in primary
                if row["mwle_recovery_r"] is not None
                and row["mwle_recovery_r"] >= best_r - maximum_recovery_drop
            ]
        finite_slope_error = [
            abs(row["mwle_recovery_slope"] - 1.0)
            for row in primary
            if row["mwle_recovery_slope"] is not None
        ]
        if finite_slope_error:
            best_error = min(finite_slope_error)
            primary = [
                row
                for row in primary
                if row["mwle_recovery_slope"] is not None
                and abs(row["mwle_recovery_slope"] - 1.0)
                <= best_error + maximum_slope_error_increase
            ]
        return primary

    def _cat_policy_kwargs(self) -> dict[str, float]:
        cat = self.config.get("cat") or {}
        return {
            "required_precision": float(cat.get("minimum_precision_rate", 0.95)),
            "minimum_mwle_convergence": float(
                cat.get("minimum_mwle_convergence_rate", 0.95)
            ),
            "maximum_recovery_drop": float(cat.get("maximum_recovery_drop", 0.01)),
            "maximum_slope_error_increase": float(
                cat.get("maximum_slope_error_increase", 0.05)
            ),
        }

    def _cat_candidate_row(
        self, value: Any, summary: dict[str, Any]
    ) -> dict[str, Any]:
        n_models = int(summary["n_models"])
        return {
            "value": value,
            "precision_rate": float(summary["precision_rate"]),
            "mean_scenarios": float(summary["scenarios"]["mean"]),
            "mwle_convergence_rate": (
                float(summary["mwle_converged"]) / n_models if n_models else 0.0
            ),
            "mwle_recovery_r": self._macro_mwle_metric(summary, "r"),
            "mwle_recovery_slope": self._macro_mwle_metric(summary, "slope"),
        }

    def _preselect_cat_floor(self) -> int:
        cat = self.config.get("cat") or {}
        rows = [
            self._cat_candidate_row(
                int(floor),
                self._cat_mode_summary(
                    self.out_dir
                    / "cat_sweeps"
                    / "min_scenarios"
                    / f"floor_{floor}"
                ),
            )
            for floor in cat.get("min_scenarios_values", [0, 12, 15, 20])
        ]
        eligible = self._quality_eligible_cat_rows(rows, **self._cat_policy_kwargs())
        chosen = (
            min(eligible, key=lambda row: (row["mean_scenarios"], row["value"]))
            if eligible
            else max(
                rows,
                key=lambda row: (
                    row["precision_rate"],
                    row["mwle_convergence_rate"],
                    -row["mean_scenarios"],
                ),
            )
        )
        return int(chosen["value"])

    def _preselect_cat_se_target(self) -> float:
        cat = self.config.get("cat") or {}
        rows = []
        for target in cat.get("se_target_values", [0.2, 0.25, 0.3, 0.35]):
            label = f"se_{_slug_number(target)}"
            rows.append(
                self._cat_candidate_row(
                    float(target),
                    self._cat_mode_summary(
                        self.out_dir / "cat_sweeps" / "se_target" / label
                    ),
                )
            )
        eligible = self._quality_eligible_cat_rows(rows, **self._cat_policy_kwargs())
        chosen = (
            min(eligible, key=lambda row: row["value"])
            if eligible
            else max(rows, key=lambda row: (row["precision_rate"], -row["value"]))
        )
        return float(chosen["value"])

    def select_cat_configuration(self) -> tuple[int, float, str]:
        cat = self.config.get("cat") or {}
        required_precision = float(cat.get("minimum_precision_rate", 0.95))
        minimum_mwle_convergence = float(
            cat.get("minimum_mwle_convergence_rate", 0.95)
        )
        maximum_recovery_drop = float(cat.get("maximum_recovery_drop", 0.01))
        maximum_slope_error_increase = float(
            cat.get("maximum_slope_error_increase", 0.05)
        )

        def candidate(value: Any, summary: dict[str, Any]) -> dict[str, Any]:
            n_models = int(summary["n_models"])
            return {
                "value": value,
                "precision_rate": float(summary["precision_rate"]),
                "mean_scenarios": float(summary["scenarios"]["mean"]),
                "mwle_convergence_rate": (
                    float(summary["mwle_converged"]) / n_models if n_models else 0.0
                ),
                "mwle_recovery_r": self._macro_mwle_metric(summary, "r"),
                "mwle_recovery_slope": self._macro_mwle_metric(summary, "slope"),
            }

        def quality_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return self._quality_eligible_cat_rows(
                rows,
                required_precision=required_precision,
                minimum_mwle_convergence=minimum_mwle_convergence,
                maximum_recovery_drop=maximum_recovery_drop,
                maximum_slope_error_increase=maximum_slope_error_increase,
            )

        floor_rows: list[dict[str, Any]] = []
        for floor in cat.get("min_scenarios_values", [0, 12, 15, 20]):
            path = self.out_dir / "cat_sweeps" / "min_scenarios" / f"floor_{floor}"
            summary = self._cat_mode_summary(path)
            floor_rows.append(candidate(int(floor), summary))
        eligible_floors = quality_rows(floor_rows)
        selected_floor_row = (
            min(eligible_floors, key=lambda row: (row["mean_scenarios"], row["value"]))
            if eligible_floors
            else max(
                floor_rows,
                key=lambda row: (
                    row["precision_rate"],
                    row["mwle_convergence_rate"],
                    -row["mean_scenarios"],
                ),
            )
        )

        se_rows: list[dict[str, Any]] = []
        for target in cat.get("se_target_values", [0.2, 0.25, 0.3, 0.35]):
            label = f"se_{_slug_number(target)}"
            summary = self._cat_mode_summary(
                self.out_dir / "cat_sweeps" / "se_target" / label
            )
            se_rows.append(candidate(float(target), summary))
        eligible_se = quality_rows(se_rows)
        selected_se_row = (
            min(eligible_se, key=lambda row: row["value"])
            if eligible_se
            else max(se_rows, key=lambda row: (row["precision_rate"], -row["value"]))
        )

        selector_rows: list[dict[str, Any]] = []
        for selector in cat.get("selection_rules", ["trace", "dopt"]):
            summary = self._cat_mode_summary(
                self.out_dir / "cat_sweeps" / "selection" / str(selector)
            )
            selector_rows.append(candidate(str(selector), summary))
        trace = next((row for row in selector_rows if row["value"] == "trace"), None)
        selected_selector_row = trace or selector_rows[0]
        dopt = next((row for row in selector_rows if row["value"] == "dopt"), None)
        if trace is not None and dopt is not None and trace["mean_scenarios"] > 0:
            reduction = (trace["mean_scenarios"] - dopt["mean_scenarios"]) / trace[
                "mean_scenarios"
            ]
            trace_r = trace["mwle_recovery_r"]
            dopt_r = dopt["mwle_recovery_r"]
            recovery_drop = (
                float(trace_r) - float(dopt_r)
                if trace_r is not None and dopt_r is not None
                else float("inf")
            )
            if (
                reduction >= float(cat.get("selector_minimum_length_reduction", 0.05))
                and recovery_drop <= float(cat.get("selector_maximum_recovery_drop", 0.01))
                and dopt["precision_rate"] >= required_precision
                and dopt["mwle_convergence_rate"] >= minimum_mwle_convergence
                and (
                    dopt["mwle_recovery_slope"] is not None
                    and trace["mwle_recovery_slope"] is not None
                    and abs(dopt["mwle_recovery_slope"] - 1.0)
                    <= abs(trace["mwle_recovery_slope"] - 1.0)
                    + maximum_slope_error_increase
                )
            ):
                selected_selector_row = dopt

        for kind, rows, selected in (
            ("min_scenarios", floor_rows, selected_floor_row),
            ("se_target", se_rows, selected_se_row),
            ("selection", selector_rows, selected_selector_row),
        ):
            for row in rows:
                row["kind"] = kind
                row["selected"] = row is selected
        all_rows = floor_rows + se_rows + selector_rows
        with (self.out_dir / "cat_configuration_candidates.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
            writer.writeheader()
            writer.writerows(all_rows)
        result = {
            "rule": {
                "minimum_precision_rate": required_precision,
                "minimum_mwle_convergence_rate": minimum_mwle_convergence,
                "maximum_recovery_drop": maximum_recovery_drop,
                "maximum_slope_error_increase": maximum_slope_error_increase,
                "floor": (
                    "shortest mean test among settings meeting coverage, MWLE "
                    "convergence, near-best r, and near-best slope"
                ),
                "se_target": (
                    "strictest (smallest) target meeting the same recovery and "
                    "coverage gates"
                ),
                "selection": (
                    "keep trace unless D-opt shortens mean test length by the configured "
                    "minimum without a material MWLE-recovery loss"
                ),
            },
            "selected_min_scenarios": selected_floor_row["value"],
            "selected_se_target": selected_se_row["value"],
            "selected_selection_rule": selected_selector_row["value"],
        }
        _write_json(self.out_dir / "cat_configuration_selection.json", result)
        return (
            int(selected_floor_row["value"]),
            float(selected_se_row["value"]),
            str(selected_selector_row["value"]),
        )

    def run_final_cat_comparison(
        self,
        bank: Path,
        scenarios: Path,
        *,
        min_scenarios: int,
        max_se: float,
        selection: str,
    ) -> None:
        comparison = self.out_dir / "cat_vs_random"
        self.run_command(
            "cat_vs_random",
            self.replay_command(
                bank, scenarios, comparison, min_scenarios=min_scenarios,
                max_se=max_se, selection=selection, mode="both",
            ),
            [
                comparison / "per_model.csv",
                comparison / "metrics.json",
                comparison / "paired_cat_vs_random.csv",
            ],
        )

        # Keep a one-row-per-model adaptive artifact for the consolidated paired
        # bootstrap report.  The combined run above intentionally has two rows per
        # model (CAT and random), so it cannot be consumed as a single CAT sweep.
        selected_cat = self.out_dir / "cat_sweeps" / "selected_configuration"
        self.run_command(
            "selected_cat_configuration",
            self.replay_command(
                bank, scenarios, selected_cat, min_scenarios=min_scenarios,
                max_se=max_se, selection=selection, mode="cat",
            ),
            [selected_cat / "per_model.csv", selected_cat / "metrics.json"],
        )

        random_only = self.out_dir / "cat_sweeps" / "random_baseline"
        self.run_command(
            "random_baseline",
            self.replay_command(
                bank, scenarios, random_only, min_scenarios=min_scenarios,
                max_se=max_se, selection=selection, mode="baseline",
            ),
            [random_only / "per_model.csv", random_only / "metrics.json"],
        )

    def validate_final_cat_configuration(self, selection: str) -> None:
        final_summary = self._cat_mode_summary(
            self.out_dir / "cat_sweeps" / "selected_configuration"
        )
        selector_summary = self._cat_mode_summary(
            self.out_dir / "cat_sweeps" / "selection" / selection
        )
        final_row = self._cat_candidate_row(selection, final_summary)
        selector_row = self._cat_candidate_row(selection, selector_summary)
        if (
            final_row["mwle_recovery_r"] is None
            or final_row["mwle_recovery_slope"] is None
            or selector_row["mwle_recovery_r"] is None
            or selector_row["mwle_recovery_slope"] is None
        ):
            raise StudyError("final CAT configuration has no estimable MWLE recovery")
        eligible = self._quality_eligible_cat_rows(
            [final_row], **self._cat_policy_kwargs()
        )
        if not eligible:
            raise StudyError(
                "the jointly frozen CAT configuration failed the configured "
                "precision or MWLE-convergence gate"
            )
        compared_fields = (
            "precision_rate",
            "mean_scenarios",
            "mwle_convergence_rate",
            "mwle_recovery_r",
            "mwle_recovery_slope",
        )
        mismatches = [
            field
            for field in compared_fields
            if not math.isclose(
                float(final_row[field]),
                float(selector_row[field]),
                rel_tol=1e-10,
                abs_tol=1e-10,
            )
        ]
        if mismatches:
            raise StudyError(
                "final CAT replay disagrees with its sequential selector sweep for "
                f"fields {mismatches}"
            )
        _write_json(
            self.out_dir / "cat_configuration_validation.json",
            {
                "status": "validated",
                "selection": selection,
                "policy": self._cat_policy_kwargs(),
                "metrics": final_row,
                "matched_selector_sweep": True,
            },
        )

    def run_order_and_uncertainty(
        self,
        bank: Path,
        scenarios: Path,
        *,
        fit_grid: int,
        ridge: float,
        min_scenarios: int,
        max_se: float,
        selection: str,
    ) -> None:
        cat = self.config.get("cat") or {}
        estimator = self.config.get("estimator") or {}
        uncertainty = self.config.get("uncertainty") or {}
        order_seeds = [int(value) for value in cat.get("order_seeds", [])]
        if not order_seeds:
            raise StudyError("cat.order_seeds must contain at least one seed")
        expected_seeds = list(range(order_seeds[0], order_seeds[0] + len(order_seeds)))
        if order_seeds != expected_seeds:
            raise StudyError(
                "scenario_order_experiment currently requires consecutive order_seeds"
            )
        order_dir = self.out_dir / "order_stability"
        order_command = [
            str(self.python), "scripts/scenario_order_experiment.py",
            "--bank", str(bank), "--matrix", str(self.inputs["matrix"]),
            "--scenarios", str(scenarios), "--out-dir", str(order_dir),
            "--n-seeds", str(len(order_seeds)), "--base-seed", str(order_seeds[0]),
            "--selection", selection, "--top-n", str(cat.get("top_n", 5)),
            "--max-se", str(max_se), "--min-evals-per-skill",
            str(cat.get("min_evals_per_skill", 15)), "--min-scenarios",
            str(min_scenarios),
            "--max-scenarios", str(cat.get("adaptive_max_scenarios", 50)),
            "--grid", str(estimator.get("quadrature_grid", 5)),
            "--max-grid-nodes", str(estimator.get("max_grid_nodes", 50000)),
            "--mwle-ridge", str(estimator.get("mwle_ridge", 1e-6)),
        ]
        self.run_command(
            "order_stability", order_command,
            [
                order_dir / "seed_runs.csv",
                order_dir / "per_model_spread.csv",
                order_dir / "metrics.json",
            ],
        )

        targets = [float(value) for value in cat.get("se_target_values", [0.2, 0.25, 0.3, 0.35])]
        param_dir = self.out_dir / "parameter_uncertainty"
        param_command = [
            str(self.python), "scripts/scenario_param_uncertainty.py",
            "--bank", str(bank), "--matrix", str(self.inputs["matrix"]),
            "--scenarios", str(scenarios), "--out-dir", str(param_dir),
            "--n-boot", str(uncertainty.get("parameter_bootstrap_replicates", 100)),
            "--seed", str(uncertainty.get("seed", 0)),
            "--cat-seed", str(cat.get("seed", 42)),
            "--fit-grid", str(fit_grid), "--ridge", str(ridge),
            "--max-iter", str(self.fit.get("max_iterations", 200)),
            "--tol", str(self.fit.get("tolerance", 1e-4)),
            "--eap-grid", str(estimator.get("quadrature_grid", 5)),
            "--max-grid-nodes", str(estimator.get("max_grid_nodes", 50000)),
            "--mwle-ridge", str(estimator.get("mwle_ridge", 1e-6)),
            "--selection", selection, "--top-n", str(cat.get("top_n", 5)),
            "--max-se", str(max_se), "--se-targets",
            ",".join(str(value) for value in targets),
            "--min-evals-per-skill", str(cat.get("min_evals_per_skill", 15)),
            "--min-scenarios", str(min_scenarios),
            "--max-scenarios", str(cat.get("adaptive_max_scenarios", 50)),
        ]
        self.run_command(
            "parameter_uncertainty", param_command,
            [
                param_dir / "ability_se_components.csv",
                param_dir / "bootstrap_replicates.csv",
                param_dir / "total_se_vs_target.csv",
                param_dir / "metrics.json",
            ],
        )

    def write_aggregation_spec(self, selected: Candidate) -> Path:
        structures = [
            {
                "name": candidate.name,
                "n_dims": candidate.structure.n_dims,
                "core_manifest": str(
                    self.out_dir / "structures" / candidate.name / "fit"
                    / "calibration_mirt_manifest.json"
                ),
                "kfold_dir": str(
                    self.out_dir / "structures" / candidate.name / "kfold"
                ),
            }
            for candidate in self.candidates
        ]
        dims = list(selected.structure.labels)
        cat_specs: list[dict[str, Any]] = []

        def add_cat(
            name: str,
            path: Path,
            *,
            group: str = "diagnostic_sweeps",
            baseline: bool = False,
            **metadata: Any,
        ) -> None:
            cat_specs.append(
                {
                    "name": name,
                    "csv": str(path / "per_model.csv"),
                    "group": group,
                    "baseline": baseline,
                    "reference_prefix": "theta_full_eap_",
                    "estimate_prefixes": ["theta_online_", "theta_eap_", "theta_mwle_"],
                    "skills": dims,
                    "metadata": metadata,
                }
            )

        cat = self.config.get("cat") or {}
        for floor in cat.get("min_scenarios_values", [0, 12, 15, 20]):
            add_cat(
                f"floor_{floor}",
                self.out_dir / "cat_sweeps" / "min_scenarios" / f"floor_{floor}",
                sweep="min_scenarios", value=floor,
            )
        for target in cat.get("se_target_values", [0.2, 0.25, 0.3, 0.35]):
            label = f"se_{_slug_number(target)}"
            add_cat(
                label,
                self.out_dir / "cat_sweeps" / "se_target" / label,
                sweep="se_target", value=target,
            )
        for selector in cat.get("selection_rules", ["trace", "dopt"]):
            add_cat(
                f"selection_{selector}",
                self.out_dir / "cat_sweeps" / "selection" / str(selector),
                sweep="selection", value=selector,
            )
        cat_selection = _read_json(self.out_dir / "cat_configuration_selection.json")
        selected_metadata = {
            "min_scenarios": cat_selection["selected_min_scenarios"],
            "max_se": cat_selection["selected_se_target"],
            "selection": cat_selection["selected_selection_rule"],
        }
        add_cat(
            "selected_configuration",
            self.out_dir / "cat_sweeps" / "selected_configuration",
            group="final_cat_vs_random",
            sweep="final",
            **selected_metadata,
        )
        add_cat(
            "random_baseline",
            self.out_dir / "cat_sweeps" / "random_baseline",
            group="final_cat_vs_random",
            baseline=True,
            sweep="baseline",
            value="seeded_random",
            **selected_metadata,
        )
        sensitivity_config = self.config.get("sensitivity") or {}
        base_grid = int(self.fit["grid"])
        base_ridge = float(self.fit["ridge"])
        sensitivity_specs: list[dict[str, Any]] = [
            {
                "name": f"grid{base_grid}_ridge{_slug_number(base_ridge)}",
                "dir": str(self.out_dir / "structures" / selected.name),
                "grid": base_grid,
                "ridge": base_ridge,
            }
        ]
        for grid_value in sensitivity_config.get("grid_values", [base_grid]):
            for ridge_value in sensitivity_config.get("ridge_values", [base_ridge]):
                grid = int(grid_value)
                ridge = float(ridge_value)
                if grid == base_grid and math.isclose(ridge, base_ridge):
                    continue
                if grid ** selected.structure.n_dims > int(
                    self.fit.get("max_grid_nodes", 5000)
                ):
                    continue
                label = f"grid{grid}_ridge{_slug_number(ridge)}"
                sensitivity_specs.append(
                    {
                        "name": label,
                        "dir": str(self.out_dir / "sensitivity" / label),
                        "grid": grid,
                        "ridge": ridge,
                    }
                )
        spec = self.out_dir / "aggregation_spec.json"
        _write_json(
            spec,
            {
                "structures": structures,
                "cat_sweeps": cat_specs,
                "estimator_oos_kfold": {
                    "dir": str(self.out_dir / "estimator_oos_kfold")
                },
                "order_stability": {"dir": str(self.out_dir / "order_stability")},
                "parameter_uncertainty": {
                    "name": "multi_target",
                    "dir": str(self.out_dir / "parameter_uncertainty"),
                },
                "sensitivity": sensitivity_specs,
            },
        )
        return spec

    def write_commands(self) -> None:
        if not self.plan_only:
            _write_json(self.out_dir / "commands.json", self.commands)

    def finalize_playbook_coverage(self) -> None:
        path = self.out_dir / "playbook_coverage.json"
        payload = _read_json(path)
        for key, value in list(payload["stages"].items()):
            if isinstance(value, str) and value.startswith("scheduled"):
                payload["stages"][key] = "completed"
            elif (
                isinstance(value, dict)
                and isinstance(value.get("status"), str)
                and value["status"].startswith("scheduled")
            ):
                value["status"] = "completed"
        payload["completed_at"] = _utcnow()
        _write_json(path, payload)

    def run(self) -> None:
        self.prepare()
        self.run_structure_studies()
        selected = self.collect_and_select_structure()
        if self.plan_only:
            print("\nPlan stops here because downstream commands depend on the selected structure.")
            return
        assert selected is not None
        self.run_sensitivity(selected)
        selected_fit_dir, selected_grid, selected_ridge = self.select_sensitivity(selected)
        bank, scenarios = self.export_selected_bank(selected, selected_fit_dir)
        self.run_cat_studies(bank, scenarios)
        selected_floor, selected_se, selected_selector = self.select_cat_configuration()
        self.run_scenario_kfold(
            selected,
            fit_grid=selected_grid,
            ridge=selected_ridge,
            min_scenarios=selected_floor,
            max_se=selected_se,
            selection=selected_selector,
        )
        self.run_final_cat_comparison(
            bank,
            scenarios,
            min_scenarios=selected_floor,
            max_se=selected_se,
            selection=selected_selector,
        )
        self.validate_final_cat_configuration(selected_selector)
        self.run_order_and_uncertainty(
            bank,
            scenarios,
            fit_grid=selected_grid,
            ridge=selected_ridge,
            min_scenarios=selected_floor,
            max_se=selected_se,
            selection=selected_selector,
        )
        self.write_commands()
        # The aggregator owns final tables, bootstrap CIs, and the plain-language report.
        aggregation_spec = self.write_aggregation_spec(selected)
        aggregate = [
            str(self.python), "scripts/aggregate_calibration_study.py",
            "--spec", str(aggregation_spec),
            "--out-dir", str(self.out_dir),
            "--primary-metric", "log_loss",
            "--bootstrap-b", str((self.config.get("uncertainty") or {}).get(
                "metric_bootstrap_replicates", 2000
            )),
            "--bootstrap-seed", str((self.config.get("uncertainty") or {}).get("seed", 0)),
        ]
        override = (self.config.get("structure_selection") or {}).get(
            "selected_structure_override"
        )
        if override:
            aggregate.extend(["--override-structure", str(override)])
        self.run_command(
            "aggregate_report", aggregate,
            [
                self.out_dir / "CALIBRATION_STUDY_SUMMARY.md",
                self.out_dir / "study_results.json",
                self.out_dir / "selection.json",
                self.out_dir / "cat_sweep_metrics.csv",
            ],
        )
        self.write_commands()
        self.finalize_playbook_coverage()
        print(f"\nComplete InFoBench calibration study -> {self.out_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="fresh output directory (default: timestamped under runs/calibration)",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="validate inputs/config and print commands; does not fit a subset or write outputs",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = _resolve(args.config)
    try:
        config, candidates = load_and_validate_config(config_path)
        if args.out_dir is None:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            out_dir = ROOT / "runs" / "calibration" / f"InFoBench_playbook_{stamp}"
        else:
            out_dir = _resolve(args.out_dir)
        Orchestrator(
            config_path, config, candidates, out_dir,
            resume=args.resume, plan_only=args.plan_only,
        ).run()
        return 0
    except (StudyError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
