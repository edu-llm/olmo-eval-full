#!/usr/bin/env python3
"""Run the preregistered dense-grid numerical follow-up for InFoBench.

The original Phase-2 study is an immutable failed result: grid 25 versus 41
missed the discrimination-parameter rank-stability gate.  This runner verifies
that parent result, copies (without refitting) its completed grid-41 artifacts,
fits grids 61 and 81, and applies the unchanged gates recorded in
``configs/infobench_dense_grid_followup_v1.json``.

Both successive comparisons are always run.  Grid 41 is locked only when both
41-vs-61 and 61-vs-81 pass.  If only the latter passes, grid 61 is locked.  EAP
stability and ridge sensitivity are unreachable until a fit grid is locked.
"""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import importlib.util
import json
import math
import platform
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CONFIG = ROOT / "configs" / "infobench_dense_grid_followup_v1.json"


def _load_dense_module():
    path = ROOT / "scripts" / "run_infobench_dense_grid_study.py"
    spec = importlib.util.spec_from_file_location("infobench_dense_grid_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load dense-grid base runner from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("infobench_dense_grid_base", module)
    spec.loader.exec_module(module)
    return module


dense = _load_dense_module()
FollowupError = dense.DenseGridError


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    return dense._read_json(path)


def _sha256(path: Path) -> str:
    return dense._sha256(path)


def _validate_followup(raw: Mapping[str, Any], config_path: Path) -> None:
    if raw.get("schema_version") != "infobench-dense-grid-followup-v1":
        raise FollowupError("unexpected dense-grid follow-up schema")
    if raw.get("status") != "preregistered_before_followup_results":
        raise FollowupError("follow-up configuration is not preregistered")
    if raw.get("benchmark") != "InFoBench":
        raise FollowupError("follow-up runner requires benchmark=InFoBench")
    if tuple(map(int, raw.get("fit_grid_candidates") or [])) != (41, 61, 81):
        raise FollowupError("fit-grid follow-up must remain 41,61,81")
    if tuple(map(str, raw.get("successive_fit_comparisons") or [])) != (
        "41_vs_61",
        "61_vs_81",
    ):
        raise FollowupError("successive fit-grid comparisons must remain frozen")
    if tuple(map(int, raw.get("eap_grid_candidates") or [])) != (21, 41, 81):
        raise FollowupError("EAP-grid candidates must remain 21,41,81")
    if tuple(map(float, raw.get("ridge_candidates_after_grid_lock") or [])) != (
        0.001,
        0.01,
        0.1,
    ):
        raise FollowupError("ridge candidates differ from the frozen follow-up")
    if not math.isclose(float(raw.get("initial_ridge", -1)), 0.1):
        raise FollowupError("initial ridge must remain 0.1")
    if "No threshold may be relaxed" not in str(raw.get("threshold_change_policy")):
        raise FollowupError("follow-up must explicitly prohibit threshold relaxation")
    if not config_path.is_file():
        raise FollowupError(f"follow-up configuration is missing: {config_path}")


def load_context(config_path: Path = DEFAULT_CONFIG):
    config_path = config_path.resolve()
    raw = _read_json(config_path)
    _validate_followup(raw, config_path)
    parent = raw["parent_study"]
    parent_config = _resolve(parent["config"])
    parent_run = _resolve(parent["run_dir"])
    parent_manifest = _resolve(parent["study_manifest"])
    parent_gate = _resolve(parent["failed_fit_gate"])
    for path, expected in (
        (parent_manifest, str(parent["study_manifest_sha256"])),
        (parent_gate, str(parent["failed_fit_gate_sha256"])),
    ):
        if not path.is_file() or _sha256(path) != expected:
            raise FollowupError(f"immutable parent artifact hash mismatch: {path}")
    manifest = _read_json(parent_manifest)
    gate = _read_json(parent_gate)
    if manifest.get("status") != parent.get("required_status"):
        raise FollowupError("parent dense-grid study is not the frozen failed run")
    if gate.get("comparison") != "25_vs_41" or gate.get("passed") is not False:
        raise FollowupError("parent 25-vs-41 gate is not the expected failed result")
    if gate.get("minimum_a_spearman_observed", 1.0) >= gate.get(
        "minimum_item_parameter_spearman_required", 0.0
    ):
        raise FollowupError("parent result no longer records the discrimination failure")

    base = dense.load_context(parent_config)
    parent_fit_gate = base.config["dense_grid"]["fit_grid_stability"]
    followup_fit_gate = raw["fit_grid_lock_rule"]
    for key in (
        "require_all_fits_converged",
        "minimum_item_parameter_spearman",
        "minimum_exportability_agreement",
        "heldout_metric_stability",
    ):
        if followup_fit_gate.get(key) != parent_fit_gate.get(key):
            raise FollowupError(f"follow-up changed frozen fit-gate field {key}")
    if raw["eap_grid_stability"] != base.config["dense_grid"]["eap_grid_stability"]:
        raise FollowupError("follow-up changed a frozen EAP-grid threshold")

    merged = copy.deepcopy(base.config)
    merged["dense_grid"]["fit_grid_candidates"] = list(raw["fit_grid_candidates"])
    merged["dense_grid"]["eap_grid_candidates"] = list(raw["eap_grid_candidates"])
    merged["dense_grid"]["initial_ridge"] = float(raw["initial_ridge"])
    merged["dense_grid"]["ridge_candidates_after_grid_lock"] = list(
        raw["ridge_candidates_after_grid_lock"]
    )
    merged["dense_grid"]["fit_grid_stability"] = {
        "comparison": "41_vs_61",
        **copy.deepcopy(raw["fit_grid_lock_rule"]),
    }
    merged["dense_grid"]["eap_grid_stability"] = copy.deepcopy(
        raw["eap_grid_stability"]
    )
    merged["immutable_decisions"]["new_output_dir"] = str(raw["output_dir"])
    merged["dense_grid_followup"] = copy.deepcopy(raw)
    context = replace(
        base,
        config_path=config_path,
        config=merged,
        fit_grids=(41, 61, 81),
        eap_grids=(21, 41, 81),
        ridge_candidates=(0.001, 0.01, 0.1),
        initial_ridge=0.1,
        default_out_dir=_resolve(raw["output_dir"]),
    )
    return context, raw, parent_run


def select_locked_fit_grid(gates: Mapping[str, Mapping[str, Any]]) -> int | None:
    """Apply the frozen two-comparison lock rule without a fallback."""

    first = bool(gates["41_vs_61"]["passed"])
    second = bool(gates["61_vs_81"]["passed"])
    if first and second:
        return 41
    if not first and second:
        return 61
    return None


def _context_for_pair(context, comparison: str):
    config = copy.deepcopy(context.config)
    config["dense_grid"]["fit_grid_stability"]["comparison"] = comparison
    return replace(context, config=config)


def summarize_fit_followup(context, out_dir: Path) -> dict[str, Any]:
    gates: dict[str, dict[str, Any]] = {}
    pair_frames: list[pd.DataFrame] = []
    for comparison in ("41_vs_61", "61_vs_81"):
        pair_context = _context_for_pair(context, comparison)
        gate = dense.summarize_fit_grids(pair_context, out_dir)
        pair = pd.read_csv(out_dir / "dense_grid" / "fit_grid_pairwise.csv")
        pair["comparison"] = comparison
        pair_frames.append(pair)
        gates[comparison] = gate
        dense._write_json(
            out_dir / "dense_grid" / f"fit_grid_stability_{comparison}.json", gate
        )
        dense._write_csv(
            out_dir / "dense_grid" / f"fit_grid_pairwise_{comparison}.csv", pair
        )
    dense._write_csv(
        out_dir / "dense_grid" / "fit_grid_pairwise.csv",
        pd.concat(pair_frames, ignore_index=True),
    )
    locked = select_locked_fit_grid(gates)
    decision = {
        "comparisons": gates,
        "selection_rule": context.config["dense_grid_followup"]["fit_grid_lock_rule"][
            "description"
        ],
        "thresholds_changed_from_parent": False,
        "passed": locked is not None,
        "locked_fit_grid": locked,
    }
    dense._write_json(out_dir / "dense_grid" / "fit_grid_stability.json", decision)
    return decision


def _worker_command(
    context,
    out_dir: Path,
    worker: str,
    fit_grid: int,
    ridge: float,
    eap_grid: int | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--config",
        str(context.config_path),
        "--out-dir",
        str(out_dir),
        "--worker",
        worker,
        "--fit-grid",
        str(fit_grid),
        "--ridge",
        str(ridge),
    ]
    if eap_grid is not None:
        command.extend(
            [
                "--eap-grid",
                str(eap_grid),
                "--fit-cache-dir",
                str(dense._fit_cache_dir(out_dir, fit_grid, ridge)),
            ]
        )
    return command


def _import_parent_stage(parent_run: Path, out_dir: Path, marker_name: str) -> list[dict[str, str]]:
    marker_path = parent_run / "stage_markers" / marker_name
    marker = _read_json(marker_path)
    records: list[dict[str, str]] = []
    hashes = marker.get("output_sha256") or {}
    if not hashes:
        raise FollowupError(f"parent stage marker has no output hashes: {marker_path}")
    for source_text, expected in hashes.items():
        source = Path(source_text).resolve()
        try:
            relative = source.relative_to(parent_run.resolve())
        except ValueError as exc:
            raise FollowupError(f"parent marker references an out-of-run path: {source}") from exc
        if not source.is_file() or _sha256(source) != expected:
            raise FollowupError(f"parent stage output hash mismatch: {source}")
        destination = out_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if _sha256(destination) != expected:
            raise FollowupError(f"copied parent artifact hash mismatch: {destination}")
        records.append(
            {
                "source": str(source),
                "source_sha256": expected,
                "destination": str(destination),
                "destination_sha256": _sha256(destination),
            }
        )
    records.append(
        {
            "source": str(marker_path),
            "source_sha256": _sha256(marker_path),
            "destination": "provenance_only_not_copied",
            "destination_sha256": "not_applicable",
        }
    )
    return records


class FollowupOrchestrator(dense.Orchestrator):
    def __init__(self, context, raw: dict[str, Any], parent_run: Path, out_dir: Path, **kwargs):
        super().__init__(context, out_dir, **kwargs)
        self.raw = raw
        self.parent_run = parent_run

    def prepare(self) -> None:
        self._guard_output()
        if self.plan_only:
            return
        runner = Path(__file__).resolve()
        if not self.resume:
            self.out_dir.mkdir(parents=True)
            shutil.copy2(self.context.config_path, self.out_dir / "followup_config.json")
            shutil.copy2(self.context.split_path, self.out_dir / "frozen_splits.json")
            versions: dict[str, str | None] = {}
            for package in ("numpy", "pandas", "scipy"):
                try:
                    versions[package] = importlib.metadata.version(package)
                except importlib.metadata.PackageNotFoundError:
                    versions[package] = None
            dense._write_json(
                self.out_dir / "study_manifest.json",
                {
                    "generated_at": dense._utcnow(),
                    "status": "running",
                    "runner": str(runner),
                    "runner_sha256": _sha256(runner),
                    "config": str(self.context.config_path),
                    "config_sha256": _sha256(self.context.config_path),
                    "split_manifest": str(self.context.split_path),
                    "split_manifest_sha256": _sha256(self.context.split_path),
                    "parent_run": str(self.parent_run),
                    "parent_failed_gate_sha256": self.raw["parent_study"][
                        "failed_fit_gate_sha256"
                    ],
                    "git_commit": dense._git_value("rev-parse", "HEAD"),
                    "git_branch": dense._git_value("branch", "--show-current"),
                    "git_dirty": bool(dense._git_value("status", "--porcelain")),
                    "python": sys.version,
                    "platform": platform.platform(),
                    "packages": versions,
                    "fit_grid_candidates": list(self.context.fit_grids),
                    "eap_grid_candidates": list(self.context.eap_grids),
                    "thresholds_changed_from_parent": False,
                    "grid_41_refit": False,
                    "judge_calls": False,
                    "cat_configuration_selection": False,
                    "historical_outputs_modified": False,
                },
            )
            imported = []
            imported.extend(
                _import_parent_stage(
                    self.parent_run,
                    self.out_dir,
                    "fit_grid_041_ridge_0p1.json",
                )
            )
            imported.extend(
                _import_parent_stage(
                    self.parent_run,
                    self.out_dir,
                    "score_fit_041_eap_081_ridge_0p1.json",
                )
            )
            dense._write_json(
                self.out_dir / "imported_parent_grid_041.json",
                {
                    "imported_at": dense._utcnow(),
                    "policy": self.raw["parent_study"]["reuse_policy"],
                    "artifacts": imported,
                },
            )
        else:
            manifest = _read_json(self.out_dir / "study_manifest.json")
            for key, expected in (
                ("runner_sha256", _sha256(runner)),
                ("config_sha256", _sha256(self.context.config_path)),
                ("split_manifest_sha256", _sha256(self.context.split_path)),
            ):
                if manifest.get(key) != expected:
                    raise FollowupError(
                        f"resume {key} differs; preserve this run and use a fresh output"
                    )
            imported = _read_json(self.out_dir / "imported_parent_grid_041.json")
            for record in imported.get("artifacts") or []:
                if record.get("destination") == "provenance_only_not_copied":
                    continue
                destination = Path(record["destination"])
                if not destination.is_file() or _sha256(destination) != record.get(
                    "destination_sha256"
                ):
                    raise FollowupError(f"imported grid-41 artifact changed: {destination}")

    def _fit_stage(self, grid: int, ridge: float) -> None:
        cache = dense._fit_cache_dir(self.out_dir, grid, ridge)
        scopes = ["full"] + [
            f"outer_{int(record['outer_fold'])}"
            for record in self.context.splits["outer_folds"]
        ]
        outputs = [cache / "fit_grid_manifest.json"]
        for scope in scopes:
            outputs.extend(
                [
                    cache / scope / "fit.npz",
                    cache / scope / "fit_manifest.json",
                    cache / scope / "item_params.csv",
                ]
            )
        self.run_stage(
            f"fit_grid_{grid:03d}_ridge_{dense._slug_number(ridge)}",
            _worker_command(self.context, self.out_dir, "fit-grid", grid, ridge),
            outputs,
        )

    def _score_stage(self, fit_grid: int, eap_grid: int, ridge: float) -> None:
        score = dense._score_dir(self.out_dir, fit_grid, eap_grid, ridge)
        self.run_stage(
            "score_fit_"
            f"{fit_grid:03d}_eap_{eap_grid:03d}_ridge_{dense._slug_number(ridge)}",
            _worker_command(
                self.context,
                self.out_dir,
                "score-grid",
                fit_grid,
                ridge,
                eap_grid,
            ),
            [
                score / "per_model.csv",
                score / "oos_metrics.csv",
                score / "recovery.csv",
                score / "pass_rate.csv",
                score / "score_manifest.json",
            ],
        )

    def run(self) -> int:
        self.prepare()
        if self.plan_only:
            for grid in (61, 81):
                print(json.dumps({"stage": "fit", "command": _worker_command(
                    self.context, self.out_dir, "fit-grid", grid, self.context.initial_ridge
                )}))
                print(json.dumps({"stage": "score_eap_81", "command": _worker_command(
                    self.context, self.out_dir, "score-grid", grid,
                    self.context.initial_ridge, 81
                )}))
            print(json.dumps({
                "conditional": "fit-grid gate must pass",
                "then": "EAP 21/41/81 stability followed by ridge 0.001/0.01/0.1",
            }))
            return 0

        for grid in (61, 81):
            self._fit_stage(grid, self.context.initial_ridge)
            self._score_stage(grid, 81, self.context.initial_ridge)
        fit_gate = summarize_fit_followup(self.context, self.out_dir)
        if not fit_gate["passed"]:
            self._finish("blocked_fit_grid_stability", {"fit_gate": fit_gate})
            raise FollowupError(
                "41/61/81 fit-grid follow-up failed; no EAP lock or ridge run was scheduled"
            )
        locked_fit = int(fit_gate["locked_fit_grid"])
        for eap_grid in self.context.eap_grids:
            score_path = dense._score_dir(
                self.out_dir, locked_fit, eap_grid, self.context.initial_ridge
            ) / "score_manifest.json"
            if not score_path.is_file():
                self._score_stage(locked_fit, eap_grid, self.context.initial_ridge)
        eap_gate = dense.summarize_eap_grids(self.context, self.out_dir, locked_fit)
        if not eap_gate["passed"]:
            self._finish(
                "blocked_eap_grid_stability",
                {"fit_gate": fit_gate, "eap_gate": eap_gate},
            )
            raise FollowupError(
                "41-vs-81 EAP-grid stability failed; no ridge run was scheduled"
            )
        locked_eap = int(eap_gate["locked_eap_grid"])
        dense._write_json(
            self.out_dir / "dense_grid" / "grid_lock.json",
            {
                "locked_at": dense._utcnow(),
                "fit_grid": locked_fit,
                "eap_grid": locked_eap,
                "initial_ridge": self.context.initial_ridge,
                "fit_gate": fit_gate,
                "eap_gate": eap_gate,
                "thresholds_changed_from_parent": False,
                "ridge_scheduled_only_after_this_lock": True,
            },
        )
        for ridge in self.context.ridge_candidates:
            if math.isclose(ridge, self.context.initial_ridge):
                continue
            self._fit_stage(locked_fit, ridge)
            self._score_stage(locked_fit, locked_eap, ridge)
        dense.summarize_ridges(self.context, self.out_dir, locked_fit, locked_eap)
        self._finish(
            "phase2_complete",
            {
                "fit_grid": locked_fit,
                "eap_grid": locked_eap,
                "ridge_candidates_completed": list(self.context.ridge_candidates),
                "production_ridge_selection": "deferred_to_nested_cv",
            },
        )
        return 0


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker", choices=("fit-grid", "score-grid"), default=None)
    parser.add_argument("--fit-grid", type=int, default=None)
    parser.add_argument("--eap-grid", type=int, default=None)
    parser.add_argument("--ridge", type=float, default=None)
    parser.add_argument("--fit-cache-dir", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.plan_only and args.resume:
        raise FollowupError("--plan-only and --resume are mutually exclusive")
    context, raw, parent_run = load_context(args.config)
    out_dir = (args.out_dir or context.default_out_dir).resolve()
    if args.worker:
        if args.plan_only or args.resume:
            raise FollowupError("internal workers do not accept --plan-only/--resume")
        if args.fit_grid is None or args.ridge is None:
            raise FollowupError("internal worker requires --fit-grid and --ridge")
        dense._assert_not_historical(context, out_dir)
        if args.worker == "fit-grid":
            if args.eap_grid is not None or args.fit_cache_dir is not None:
                raise FollowupError("fit worker cannot receive EAP/cache arguments")
            return dense.worker_fit_grid(context, out_dir, args.fit_grid, args.ridge)
        if args.eap_grid is None or args.fit_cache_dir is None:
            raise FollowupError("score worker requires --eap-grid and --fit-cache-dir")
        return dense.worker_score_grid(
            context,
            out_dir,
            args.fit_cache_dir,
            args.fit_grid,
            args.eap_grid,
            args.ridge,
        )
    if any(
        value is not None
        for value in (args.fit_grid, args.eap_grid, args.ridge, args.fit_cache_dir)
    ):
        raise FollowupError("worker-only arguments cannot be used without --worker")
    orchestrator = FollowupOrchestrator(
        context,
        raw,
        parent_run,
        out_dir,
        resume=args.resume,
        plan_only=args.plan_only,
    )
    return orchestrator.run()


if __name__ == "__main__":
    raise SystemExit(main())
