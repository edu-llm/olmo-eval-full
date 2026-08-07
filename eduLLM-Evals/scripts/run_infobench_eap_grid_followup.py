#!/usr/bin/env python3
"""Run the append-only dense-EAP follow-up for InFoBench remediation.

The parent follow-up locked fit grid 61 but failed its 41-vs-81 EAP-theta
stability gate.  This runner verifies that immutable result, imports the grid-61
fit and EAP-81 score without recomputation, scores EAP grids 161 and 321, and
applies the original thresholds unchanged.  Ridge work remains unreachable
until an EAP grid is honestly locked.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CONFIG = ROOT / "configs" / "infobench_eap_grid_followup_v1.json"


def _load_followup_module():
    path = ROOT / "scripts" / "run_infobench_dense_grid_followup.py"
    spec = importlib.util.spec_from_file_location("infobench_dense_followup_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load dense-grid follow-up from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("infobench_dense_followup_base", module)
    spec.loader.exec_module(module)
    return module


fit_followup = _load_followup_module()
dense = fit_followup.dense
FollowupError = dense.DenseGridError


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    return dense._read_json(path)


def _sha256(path: Path) -> str:
    return dense._sha256(path)


def _validate_config(raw: Mapping[str, Any]) -> None:
    if raw.get("schema_version") != "infobench-eap-grid-followup-v1":
        raise FollowupError("unexpected EAP follow-up schema")
    if raw.get("status") != "preregistered_before_followup_results":
        raise FollowupError("EAP follow-up is not preregistered")
    if raw.get("benchmark") != "InFoBench":
        raise FollowupError("EAP follow-up requires benchmark=InFoBench")
    if int(raw.get("fit_grid", -1)) != 61:
        raise FollowupError("the previously locked fit grid must remain 61")
    if tuple(map(int, raw.get("eap_grid_candidates") or [])) != (81, 161, 321):
        raise FollowupError("EAP candidates must remain 81,161,321")
    if tuple(map(str, raw.get("successive_eap_comparisons") or [])) != (
        "81_vs_161",
        "161_vs_321",
    ):
        raise FollowupError("EAP comparisons differ from the frozen follow-up")
    if tuple(map(float, raw.get("ridge_candidates_after_grid_lock") or [])) != (
        0.001,
        0.01,
        0.1,
    ):
        raise FollowupError("ridge candidates differ from the frozen follow-up")
    if not math.isclose(float(raw.get("initial_ridge", -1)), 0.1):
        raise FollowupError("initial ridge must remain 0.1")
    if "No threshold may be relaxed" not in str(raw.get("threshold_change_policy")):
        raise FollowupError("EAP follow-up must prohibit threshold relaxation")


def load_context(config_path: Path = DEFAULT_CONFIG):
    config_path = config_path.resolve()
    raw = _read_json(config_path)
    _validate_config(raw)
    parent = raw["parent_study"]
    parent_run = _resolve(parent["run_dir"])
    paths = (
        (_resolve(parent["study_manifest"]), str(parent["study_manifest_sha256"])),
        (_resolve(parent["fit_gate"]), str(parent["fit_gate_sha256"])),
        (_resolve(parent["failed_eap_gate"]), str(parent["failed_eap_gate_sha256"])),
    )
    for path, expected in paths:
        if not path.is_file() or _sha256(path) != expected:
            raise FollowupError(f"immutable parent artifact hash mismatch: {path}")
    parent_manifest = _read_json(paths[0][0])
    fit_gate = _read_json(paths[1][0])
    failed_eap = _read_json(paths[2][0])
    if parent_manifest.get("status") != parent.get("required_status"):
        raise FollowupError("parent study is not the frozen EAP-blocked run")
    if not (
        fit_gate.get("passed") is True
        and int(fit_gate.get("locked_fit_grid", -1)) == int(parent["locked_fit_grid"])
    ):
        raise FollowupError("parent fit-grid lock is absent or inconsistent")
    if failed_eap.get("passed") is not False or failed_eap.get("comparison") != "41_vs_81":
        raise FollowupError("parent EAP gate is not the expected failed result")
    if not (
        float(failed_eap["median_absolute_theta_shift"])
        > float(failed_eap["thresholds"]["maximum_median_absolute_theta_shift"])
        and float(failed_eap["p95_absolute_theta_shift"])
        > float(failed_eap["thresholds"]["maximum_p95_absolute_theta_shift"])
    ):
        raise FollowupError("parent result no longer records the theta-shift failure")

    parent_context, _parent_raw, _original_run = fit_followup.load_context(
        _resolve(parent["config"])
    )
    inherited_thresholds = copy.deepcopy(
        parent_context.config["dense_grid"]["eap_grid_stability"]
    )
    inherited_thresholds.pop("comparison", None)
    configured_thresholds = copy.deepcopy(raw["eap_grid_lock_rule"])
    configured_thresholds.pop("description", None)
    if configured_thresholds != inherited_thresholds:
        raise FollowupError("EAP follow-up changed a frozen stability threshold")

    merged = copy.deepcopy(parent_context.config)
    merged["dense_grid"]["fit_grid_candidates"] = [61]
    merged["dense_grid"]["eap_grid_candidates"] = list(raw["eap_grid_candidates"])
    merged["dense_grid"]["initial_ridge"] = float(raw["initial_ridge"])
    merged["dense_grid"]["ridge_candidates_after_grid_lock"] = list(
        raw["ridge_candidates_after_grid_lock"]
    )
    merged["dense_grid"]["eap_grid_stability"] = {
        "comparison": "81_vs_161",
        **configured_thresholds,
    }
    merged["immutable_decisions"]["new_output_dir"] = str(raw["output_dir"])
    merged["eap_grid_followup"] = copy.deepcopy(raw)
    context = replace(
        parent_context,
        config_path=config_path,
        config=merged,
        fit_grids=(61,),
        eap_grids=(81, 161, 321),
        ridge_candidates=(0.001, 0.01, 0.1),
        initial_ridge=0.1,
        default_out_dir=_resolve(raw["output_dir"]),
    )
    return context, raw, parent_run, fit_gate


def select_locked_eap_grid(gates: Mapping[str, Mapping[str, Any]]) -> int | None:
    first = bool(gates["81_vs_161"]["passed"])
    second = bool(gates["161_vs_321"]["passed"])
    if first and second:
        return 81
    if not first and second:
        return 161
    return None


def _context_for_pair(context, comparison: str):
    config = copy.deepcopy(context.config)
    config["dense_grid"]["eap_grid_stability"]["comparison"] = comparison
    return replace(context, config=config)


def summarize_eap_followup(context, out_dir: Path) -> dict[str, Any]:
    gates: dict[str, dict[str, Any]] = {}
    for comparison in ("81_vs_161", "161_vs_321"):
        gate = dense.summarize_eap_grids(
            _context_for_pair(context, comparison), out_dir, locked_fit_grid=61
        )
        gates[comparison] = gate
        dense._write_json(
            out_dir / "dense_grid" / f"eap_grid_stability_{comparison}.json", gate
        )
    locked = select_locked_eap_grid(gates)
    decision = {
        "comparisons": gates,
        "selection_rule": context.config["eap_grid_followup"]["eap_grid_lock_rule"][
            "description"
        ],
        "thresholds_changed_from_parent": False,
        "passed": locked is not None,
        "locked_eap_grid": locked,
    }
    dense._write_json(out_dir / "dense_grid" / "eap_grid_stability.json", decision)
    return decision


def _worker_command(
    context,
    out_dir: Path,
    worker: str,
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
        "61",
        "--ridge",
        str(ridge),
    ]
    if eap_grid is not None:
        command.extend(
            [
                "--eap-grid",
                str(eap_grid),
                "--fit-cache-dir",
                str(dense._fit_cache_dir(out_dir, 61, ridge)),
            ]
        )
    return command


class EAPFollowupOrchestrator(dense.Orchestrator):
    def __init__(
        self,
        context,
        raw: dict[str, Any],
        parent_run: Path,
        parent_fit_gate: dict[str, Any],
        out_dir: Path,
        **kwargs,
    ):
        super().__init__(context, out_dir, **kwargs)
        self.raw = raw
        self.parent_run = parent_run
        self.parent_fit_gate = parent_fit_gate

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
                    "parent_failed_eap_gate_sha256": self.raw["parent_study"][
                        "failed_eap_gate_sha256"
                    ],
                    "git_commit": dense._git_value("rev-parse", "HEAD"),
                    "git_branch": dense._git_value("branch", "--show-current"),
                    "git_dirty": bool(dense._git_value("status", "--porcelain")),
                    "python": sys.version,
                    "platform": platform.platform(),
                    "packages": versions,
                    "fit_grid": 61,
                    "eap_grid_candidates": list(self.context.eap_grids),
                    "thresholds_changed_from_parent": False,
                    "fit_grid_61_refit": False,
                    "eap_grid_81_rescored": False,
                    "judge_calls": False,
                    "cat_configuration_selection": False,
                    "historical_outputs_modified": False,
                },
            )
            imported = []
            imported.extend(
                fit_followup._import_parent_stage(
                    self.parent_run,
                    self.out_dir,
                    "fit_grid_061_ridge_0p1.json",
                )
            )
            imported.extend(
                fit_followup._import_parent_stage(
                    self.parent_run,
                    self.out_dir,
                    "score_fit_061_eap_081_ridge_0p1.json",
                )
            )
            dense._write_json(
                self.out_dir / "imported_parent_grid_061.json",
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
            imported = _read_json(self.out_dir / "imported_parent_grid_061.json")
            for record in imported.get("artifacts") or []:
                if record.get("destination") == "provenance_only_not_copied":
                    continue
                destination = Path(record["destination"])
                if not destination.is_file() or _sha256(destination) != record.get(
                    "destination_sha256"
                ):
                    raise FollowupError(f"imported parent artifact changed: {destination}")

    def _fit_stage(self, ridge: float) -> None:
        cache = dense._fit_cache_dir(self.out_dir, 61, ridge)
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
            f"fit_grid_061_ridge_{dense._slug_number(ridge)}",
            _worker_command(self.context, self.out_dir, "fit-grid", ridge),
            outputs,
        )

    def _score_stage(self, eap_grid: int, ridge: float) -> None:
        score = dense._score_dir(self.out_dir, 61, eap_grid, ridge)
        self.run_stage(
            f"score_fit_061_eap_{eap_grid:03d}_ridge_{dense._slug_number(ridge)}",
            _worker_command(self.context, self.out_dir, "score-grid", ridge, eap_grid),
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
            for grid in (161, 321):
                print(
                    json.dumps(
                        {
                            "stage": f"score_eap_{grid}",
                            "command": _worker_command(
                                self.context,
                                self.out_dir,
                                "score-grid",
                                self.context.initial_ridge,
                                grid,
                            ),
                        }
                    )
                )
            print(
                json.dumps(
                    {
                        "conditional": "EAP-grid gate must pass",
                        "then": "ridge 0.001/0.01/0.1 sensitivity",
                    }
                )
            )
            return 0
        for grid in (161, 321):
            self._score_stage(grid, self.context.initial_ridge)
        eap_gate = summarize_eap_followup(self.context, self.out_dir)
        if not eap_gate["passed"]:
            self._finish(
                "blocked_eap_grid_stability",
                {"fit_gate": self.parent_fit_gate, "eap_gate": eap_gate},
            )
            raise FollowupError(
                "81/161/321 EAP-grid follow-up failed; no ridge run was scheduled"
            )
        locked_eap = int(eap_gate["locked_eap_grid"])
        dense._write_json(
            self.out_dir / "dense_grid" / "grid_lock.json",
            {
                "locked_at": dense._utcnow(),
                "fit_grid": 61,
                "eap_grid": locked_eap,
                "initial_ridge": self.context.initial_ridge,
                "fit_gate": self.parent_fit_gate,
                "eap_gate": eap_gate,
                "thresholds_changed_from_parent": False,
                "ridge_scheduled_only_after_this_lock": True,
            },
        )
        for ridge in self.context.ridge_candidates:
            if math.isclose(ridge, self.context.initial_ridge):
                continue
            self._fit_stage(ridge)
            self._score_stage(locked_eap, ridge)
        dense.summarize_ridges(self.context, self.out_dir, 61, locked_eap)
        self._finish(
            "phase2_complete",
            {
                "fit_grid": 61,
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
    context, raw, parent_run, fit_gate = load_context(args.config)
    out_dir = (args.out_dir or context.default_out_dir).resolve()
    if args.worker:
        if args.plan_only or args.resume:
            raise FollowupError("internal workers do not accept --plan-only/--resume")
        if args.fit_grid != 61 or args.ridge is None:
            raise FollowupError("worker requires the locked fit grid 61 and a ridge")
        dense._assert_not_historical(context, out_dir)
        if args.worker == "fit-grid":
            if args.eap_grid is not None or args.fit_cache_dir is not None:
                raise FollowupError("fit worker cannot receive EAP/cache arguments")
            return dense.worker_fit_grid(context, out_dir, 61, args.ridge)
        if args.eap_grid is None or args.fit_cache_dir is None:
            raise FollowupError("score worker requires --eap-grid and --fit-cache-dir")
        return dense.worker_score_grid(
            context,
            out_dir,
            args.fit_cache_dir,
            61,
            args.eap_grid,
            args.ridge,
        )
    if any(
        value is not None
        for value in (args.fit_grid, args.eap_grid, args.ridge, args.fit_cache_dir)
    ):
        raise FollowupError("worker-only arguments cannot be used without --worker")
    orchestrator = EAPFollowupOrchestrator(
        context,
        raw,
        parent_run,
        fit_gate,
        out_dir,
        resume=args.resume,
        plan_only=args.plan_only,
    )
    return orchestrator.run()


if __name__ == "__main__":
    raise SystemExit(main())
