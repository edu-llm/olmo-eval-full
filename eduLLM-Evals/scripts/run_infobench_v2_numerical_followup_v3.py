#!/usr/bin/env python3
"""Frozen append-only v3 rerun of the InFoBench v2 numerical follow-up.

V1 remains an aborted historical attempt. V2 remains a terminal blocked but
non-promotable run because its runner changed while its process was active.
V3 inherits the unchanged scientific contract and current audited
orchestration, writes only to a new leaf, recomputes all 72 grid-101/grid-121
fits, and reuses no v1 or v2 fit/checkpoint artifact. No CAT or calibration
model selection is performed.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import calibrate_mirt as cm  # noqa: E402
from scripts import run_infobench_v2_numerical_followup_v2 as v2  # noqa: E402

DEFAULT_CONFIG = ROOT / "configs" / "infobench_v2_numerical_followup_v3.json"
CONFIG_SCHEMA = "infobench-v2-numerical-followup-v3"
RUNNER_SCHEMA = "infobench-v2-numerical-followup-run-v3"
CHECKPOINT_SCHEMA = "infobench-v2-numerical-followup-checkpoint-v3"
LOCK_SCHEMA = "infobench-v2-numerical-followup-lock-v3"
OUTPUT_LEAF = "InFoBench_v2_numerical_followup_v3"
FROZEN_CONTRACT_SHA256 = v2.FROZEN_CONTRACT_SHA256

CODE_DEPENDENCIES = (
    Path(__file__).resolve(),
    ROOT / "scripts" / "run_infobench_v2_numerical_followup_v2.py",
    ROOT / "scripts" / "run_infobench_v2_numerical_followup.py",
    ROOT / "scripts" / "run_infobench_v2_numerical_checks.py",
    ROOT / "scripts" / "calibrate_mirt.py",
    ROOT / "scripts" / "kfold_cv_mirt.py",
    ROOT / "scripts" / "scenario_cat_lib.py",
    ROOT / "tutor_cat" / "mirt.py",
)

EXPECTED_ORCHESTRATION_CORRECTIONS = {
    **v2.EXPECTED_CORRECTIONS,
    "top_level_decisions_carry_v3_schema": True,
    "stored_fit_decision_provenance_is_validated": True,
}


class FollowupV3Error(v2.FollowupV2Error):
    """V3 frozen provenance, code-freeze, or append-only contract failed."""


def _resolve(value: str | Path) -> Path:
    return v2._resolve(value)


def _display(path: Path) -> str:
    return v2._display(path)


def _sha256(path: Path) -> str:
    return v2._sha256(path)


def _read_json(path: Path) -> dict[str, Any]:
    return v2._read_json(path)


def _atomic_json(path: Path, value: Any) -> None:
    v2._atomic_json(path, value)


def _canonical_sha256(value: Any) -> str:
    return v2._canonical_sha256(value)


def _require_hash(path: Path, expected: str, label: str) -> None:
    v2._require_hash(path, expected, label)


def _validate_config(raw: Mapping[str, Any]) -> None:
    expected = {
        "schema_version": CONFIG_SCHEMA,
        "status": "preregistered_before_followup_results",
        "benchmark": "InFoBench",
        "output_dir": f"runs/calibration/{OUTPUT_LEAF}",
        "lock_path": f"runs/calibration/{OUTPUT_LEAF}/numerical_followup_lock.json",
    }
    for key, value in expected.items():
        if raw.get(key) != value:
            raise FollowupV3Error(f"v3 config field changed: {key}")
    source = raw.get("scientific_design_source") or {}
    if source.get("config") != "configs/infobench_v2_numerical_followup_v2.json":
        raise FollowupV3Error("v3 scientific design source changed")
    if source.get("frozen_contract_sha256") != FROZEN_CONTRACT_SHA256:
        raise FollowupV3Error("v3 scientific contract hash changed")
    orchestration = raw.get("orchestration_source") or {}
    if orchestration.get("runner") != "scripts/run_infobench_v2_numerical_followup_v2.py":
        raise FollowupV3Error("v3 orchestration source changed")
    if orchestration.get("required_corrections") != EXPECTED_ORCHESTRATION_CORRECTIONS:
        raise FollowupV3Error("v3 orchestration-correction contract changed")
    history = raw.get("historical_evidence") or {}
    reuse = history.get("reuse_policy") or {}
    if reuse != {
        "v1_fit_or_checkpoint_artifacts_reused": 0,
        "v2_fit_or_checkpoint_artifacts_reused": 0,
        "required_new_grid101_fits": 36,
        "required_new_grid121_fits": 36,
        "required_total_new_fits": 72,
        "parent_grid81_imports": 36,
    }:
        raise FollowupV3Error("v3 zero-reuse/new-fit contract changed")
    freeze = raw.get("code_freeze") or {}
    if (
        freeze.get("status") != "declared_before_v3_results"
        or freeze.get("runner") != "scripts/run_infobench_v2_numerical_followup_v3.py"
    ):
        raise FollowupV3Error("v3 code-freeze declaration changed")
    runner_hash = freeze.get("runner_sha256")
    if not isinstance(runner_hash, str) or len(runner_hash) != 64:
        raise FollowupV3Error("v3 frozen runner SHA-256 is not finalized")
    if not str(raw.get("threshold_change_policy", "")).startswith("No scientific"):
        raise FollowupV3Error("v3 unchanged-threshold policy is missing")


def _verify_v1_history(raw: Mapping[str, Any], source_config: Mapping[str, Any]) -> None:
    observed = v2._verify_superseded_attempt(source_config)
    frozen = raw["historical_evidence"]["v1_aborted"]
    expected = {
        "aborted_marker_sha256": frozen["aborted_marker_sha256"],
        "output_tree_sha256": frozen["output_tree_sha256"],
        "output_file_count": frozen["output_file_count"],
    }
    for key, value in expected.items():
        if observed.get(key) != value:
            raise FollowupV3Error(f"v1 historical evidence changed: {key}")
    marker = _read_json(_resolve(frozen["aborted_marker"]))
    if marker.get("status") != frozen["required_status"] or frozen.get("promotable") is not False:
        raise FollowupV3Error("v1 aborted disposition changed")


def _verify_v2_history(raw: Mapping[str, Any]) -> dict[str, Any]:
    frozen = raw["historical_evidence"]["v2_terminal"]
    run_dir = _resolve(frozen["run_dir"])
    count, tree_hash = v2._tree_sha256(run_dir)
    if count != int(frozen["output_file_count"]) or tree_hash != frozen[
        "output_tree_sha256"
    ]:
        raise FollowupV3Error("terminal v2 output tree changed")
    paths = {
        "manifest": _resolve(frozen["manifest"]),
        "decision": _resolve(frozen["decision"]),
        "fit_decision": _resolve(frozen["fit_decision"]),
        "eap_decision": _resolve(frozen["eap_decision"]),
        "launched_runner": _resolve(frozen["launched_runner_snapshot"]),
    }
    expected_hashes = {
        "manifest": frozen["manifest_sha256"],
        "decision": frozen["decision_sha256"],
        "fit_decision": frozen["fit_decision_sha256"],
        "eap_decision": frozen["eap_decision_sha256"],
        "launched_runner": frozen["launched_runner_sha256"],
    }
    for name, path in paths.items():
        _require_hash(path, str(expected_hashes[name]), f"terminal v2 {name}")
    manifest = _read_json(paths["manifest"])
    decision = _read_json(paths["decision"])
    if (
        manifest.get("status") != frozen["required_manifest_status"]
        or manifest.get("study_signature_sha256")
        != frozen["required_study_signature_sha256"]
        or decision.get("status") != frozen["required_decision_status"]
        or decision.get("passed") is not frozen["required_decision_passed"]
    ):
        raise FollowupV3Error("terminal v2 disposition changed")
    recorded_runner = (manifest.get("study_signature") or {}).get("code_sha256", {}).get(
        "scripts/run_infobench_v2_numerical_followup_v2.py"
    )
    if recorded_runner != frozen["launched_runner_sha256"]:
        raise FollowupV3Error("terminal v2 launched-runner provenance changed")
    current_runner_path = _resolve(raw["orchestration_source"]["runner"])
    if _sha256(current_runner_path) != frozen["current_runner_sha256"]:
        raise FollowupV3Error("audited current v2 orchestration source changed")
    if recorded_runner == frozen["current_runner_sha256"]:
        raise FollowupV3Error("v2 provenance mismatch is no longer explicitly represented")
    lock = run_dir / "numerical_followup_lock.json"
    lock_hash = run_dir / "numerical_followup_lock.sha256"
    if frozen.get("required_lock_absent") is not True or lock.exists() or lock_hash.exists():
        raise FollowupV3Error("terminal v2 unexpectedly has a promotable lock")
    if frozen.get("promotable") is not False:
        raise FollowupV3Error("terminal v2 must remain non-promotable")
    return {
        "manifest_sha256": frozen["manifest_sha256"],
        "decision_sha256": frozen["decision_sha256"],
        "fit_decision_sha256": frozen["fit_decision_sha256"],
        "eap_decision_sha256": frozen["eap_decision_sha256"],
        "launched_runner_sha256": frozen["launched_runner_sha256"],
        "current_runner_sha256": frozen["current_runner_sha256"],
        "output_tree_sha256": tree_hash,
        "output_file_count": count,
        "promotable": False,
    }


def load_context(
    config_path: Path = DEFAULT_CONFIG, out_dir: Path | None = None
) -> v2.v1.Context:
    config_path = config_path.resolve()
    raw = _read_json(config_path)
    _validate_config(raw)
    freeze = raw["code_freeze"]
    _require_hash(Path(__file__).resolve(), freeze["runner_sha256"], "frozen v3 runner")
    source = raw["scientific_design_source"]
    source_path = _resolve(source["config"])
    _require_hash(source_path, source["config_sha256"], "v3 scientific design source")
    orchestration = raw["orchestration_source"]
    _require_hash(
        _resolve(orchestration["runner"]),
        orchestration["runner_sha256"],
        "v3 orchestration source",
    )
    source_raw = _read_json(source_path)
    v2._validate_overlay(source_raw)
    source_effective = v2._materialize_effective_config(source_raw)
    v2._validate_effective_config(source_effective)
    if _canonical_sha256(source_raw["frozen_contract"]) != FROZEN_CONTRACT_SHA256:
        raise FollowupV3Error("source scientific contract no longer matches v3")
    _verify_v1_history(raw, source_effective)
    v2_history = _verify_v2_history(raw)

    base = v2.load_context(source_path)
    effective = copy.deepcopy(base.config)
    effective.update(
        {
            "schema_version": CONFIG_SCHEMA,
            "status": raw["status"],
            "purpose": raw["purpose"],
            "scientific_design_source": copy.deepcopy(raw["scientific_design_source"]),
            "orchestration_source": copy.deepcopy(raw["orchestration_source"]),
            "historical_evidence": copy.deepcopy(raw["historical_evidence"]),
            "code_freeze": copy.deepcopy(raw["code_freeze"]),
            "output_dir": raw["output_dir"],
            "lock_path": raw["lock_path"],
            "threshold_change_policy": raw["threshold_change_policy"],
            "selection_policy": raw["selection_policy"],
        }
    )
    configured_out = _resolve(effective["output_dir"])
    chosen_out = configured_out if out_dir is None else out_dir.resolve()
    if chosen_out != configured_out:
        raise FollowupV3Error("--out-dir must equal the preregistered v3 output leaf")
    if _resolve(effective["lock_path"]) != chosen_out / "numerical_followup_lock.json":
        raise FollowupV3Error("v3 lock path escapes the output leaf")

    provisional = v2.v1.Context(
        **{
            **base.__dict__,
            "config_path": config_path,
            "config": effective,
            "out_dir": chosen_out,
            "study_signature": {},
        }
    )
    environment = v2.v1.parent_runner._environment()
    parent_signature = base.study_signature
    signature_payload = {
        "schema_version": RUNNER_SCHEMA,
        "config_sha256": _sha256(config_path),
        "effective_config_sha256": _canonical_sha256(effective),
        "frozen_contract_sha256": FROZEN_CONTRACT_SHA256,
        "historical_v1_sha256": {
            "aborted_marker_sha256": raw["historical_evidence"]["v1_aborted"][
                "aborted_marker_sha256"
            ],
            "output_tree_sha256": raw["historical_evidence"]["v1_aborted"][
                "output_tree_sha256"
            ],
        },
        "historical_v2_sha256": v2_history,
        "parent_manifest_sha256": parent_signature["parent_manifest_sha256"],
        "parent_decision_sha256": parent_signature["parent_decision_sha256"],
        "parent_checkpoint_sha256": parent_signature["parent_checkpoint_sha256"],
        "input_sha256": parent_signature["input_sha256"],
        "code_sha256": {_display(path): _sha256(path) for path in CODE_DEPENDENCIES},
        "environment_sha256": environment["canonical_sha256"],
        "exact_specifications": parent_signature["exact_specifications"],
        "fit_grids": list(v2.FIT_GRIDS),
        "comparison_schedule": [list(pair) for pair in v2.COMPARISON_SCHEDULE],
        "eap_profiles": v2.v1.EAP_PROFILES,
        "v1_fit_or_checkpoint_artifacts_reused": 0,
        "v2_fit_or_checkpoint_artifacts_reused": 0,
        "required_total_new_fits": 72,
    }
    signature = {
        **signature_payload,
        "environment": environment,
        "canonical_sha256": _canonical_sha256(signature_payload),
    }
    return v2.v1.Context(**{**provisional.__dict__, "study_signature": signature})


def runtime_schedule(context: v2.v1.Context) -> dict[str, Any]:
    schedule = v2.runtime_schedule(context)
    schedule.update(
        {
            "append_only_output_leaf": _display(context.out_dir),
            "historical_v1_artifacts_reused": 0,
            "historical_v2_artifacts_reused": 0,
            "required_total_new_fits": 72,
            "code_freeze_runner_sha256": context.config["code_freeze"][
                "runner_sha256"
            ],
            "fresh_mode_available": False,
        }
    )
    return schedule


def _assert_safe_output(context: v2.v1.Context) -> None:
    calibration_root = (ROOT / "runs" / "calibration").resolve()
    output = context.out_dir.resolve()
    if output == calibration_root or calibration_root not in output.parents:
        raise FollowupV3Error("v3 output must be a leaf below runs/calibration")
    if output.name != OUTPUT_LEAF:
        raise FollowupV3Error("refusing a non-preregistered v3 output leaf")
    protected = {
        context.parent_run.resolve(),
        _resolve(context.config["superseded_attempt"]["run_dir"]),
        _resolve(context.config["historical_evidence"]["v2_terminal"]["run_dir"]),
    }
    if output in protected:
        raise FollowupV3Error("parent, v1, and v2 outputs are immutable")


def _prepare(context: v2.v1.Context, *, resume: bool, fresh: bool = False) -> None:
    _assert_safe_output(context)
    _verify_v1_history(context.config, context.config)
    _verify_v2_history(context.config)
    if fresh:
        raise FollowupV3Error("destructive --fresh mode does not exist in append-only v3")
    manifest_path = context.out_dir / "study_manifest.json"
    if resume:
        if not manifest_path.is_file():
            raise FollowupV3Error("--resume requires an existing v3 manifest")
        manifest = _read_json(manifest_path)
        if manifest.get("schema_version") != RUNNER_SCHEMA:
            raise FollowupV3Error("resume manifest schema differs")
        if manifest.get("study_signature_sha256") != context.study_signature[
            "canonical_sha256"
        ]:
            raise FollowupV3Error("resume signature differs from frozen v3 inputs/code")
        if manifest.get("status") != "running":
            raise FollowupV3Error(
                f"terminal v3 evidence is immutable: {manifest.get('status')!r}"
            )
        if (context.out_dir / "ABORTED.json").exists():
            raise FollowupV3Error("aborted v3 evidence cannot be resumed")
        v2._validate_existing_checkpoints(context)
        return
    if context.out_dir.exists():
        raise FollowupV3Error("v3 output exists; use --resume only for a running study")
    context.out_dir.mkdir(parents=True)
    _atomic_json(
        manifest_path,
        {
            "schema_version": RUNNER_SCHEMA,
            "status": "running",
            "started_at": v2.v1.parent_runner._utcnow(),
            "study_signature_sha256": context.study_signature["canonical_sha256"],
            "study_signature": context.study_signature,
            "runtime_schedule": runtime_schedule(context),
            "historical_evidence": context.config["historical_evidence"],
            "parent_failure_preserved": True,
            "v1_abort_preserved": True,
            "v2_terminal_preserved": True,
            "historical_fit_or_checkpoint_artifacts_reused": 0,
            "cat_results_inspected": False,
            "selection_performed": False,
        },
    )


def _finalize(
    context: v2.v1.Context,
    fit_decision: Mapping[str, Any],
    first_gates: Mapping[str, Mapping[str, Any]],
    second_gates: Mapping[str, Mapping[str, Any]],
    eap_gates: Mapping[str, Mapping[str, Any]],
    eap_decision: Mapping[str, Any],
) -> dict[str, Any]:
    passed = fit_decision.get("passed") is True and eap_decision.get("passed") is True
    effective_profile = (
        {
            "fit_grid": int(fit_decision["locked_common_fit_grid"]),
            **dict(eap_decision["locked_common_eap_profile"]),
        }
        if passed
        else None
    )
    evidence_hashes: dict[str, str] = {}
    for comparison, gates in (
        ("81_vs_101", first_gates),
        ("101_vs_121", second_gates),
    ):
        for spec_id in gates:
            path = (
                context.out_dir
                / "fit_comparisons"
                / comparison
                / spec_id
                / "fit_pair_gate.json"
            )
            evidence_hashes[f"fit/{comparison}/{spec_id}"] = _sha256(path)
    for spec_id in eap_gates:
        path = context.out_dir / "eap_bound" / spec_id / "eap_bound_gate.json"
        evidence_hashes[f"eap/{spec_id}"] = _sha256(path)
    history = context.config["historical_evidence"]
    decision = {
        "schema_version": RUNNER_SCHEMA,
        "status": "complete_pass" if passed else "blocked_numerical_followup",
        "passed": passed,
        "parent_failure_preserved": True,
        "v1_abort_preserved": True,
        "v2_terminal_preserved": True,
        "historical_fit_or_checkpoint_artifacts_reused": 0,
        "historical_v1_sha256": {
            "aborted_marker_sha256": history["v1_aborted"]["aborted_marker_sha256"],
            "output_tree_sha256": history["v1_aborted"]["output_tree_sha256"],
        },
        "historical_v2_sha256": {
            "manifest_sha256": history["v2_terminal"]["manifest_sha256"],
            "decision_sha256": history["v2_terminal"]["decision_sha256"],
            "launched_runner_sha256": history["v2_terminal"][
                "launched_runner_sha256"
            ],
            "output_tree_sha256": history["v2_terminal"]["output_tree_sha256"],
        },
        "audit_disclosure": context.config["audit_disclosure"],
        "parent_manifest_sha256": context.config["parent_study"][
            "study_manifest_sha256"
        ],
        "parent_decision_sha256": context.config["parent_study"]["decision_sha256"],
        "fit_grid_decision": dict(fit_decision),
        "eap_profile_decision": dict(eap_decision),
        "fit_comparison_all_six_passed": dict(
            fit_decision.get("fit_comparison_all_six_passed") or {}
        ),
        "effective_global_profile": effective_profile,
        "evidence_sha256": evidence_hashes,
        "passed_spec_ids": [spec.spec_id for spec in context.specs] if passed else [],
        "calibration_specification_selected": None,
        "selection_performed": False,
        "cat_results_inspected": False,
        "phase3_ready": False,
        "phase3_blocker": (
            "Phase 3 must explicitly consume and hash this v3 lock/profile."
            if passed
            else "V3 numerical follow-up did not produce a common locked profile."
        ),
    }
    decision_path = context.out_dir / "numerical_followup_decision.json"
    v2._write_json_once(decision_path, decision)
    lock_path = context.out_dir / "numerical_followup_lock.json"
    sha_path = context.out_dir / "numerical_followup_lock.sha256"
    if passed:
        lock = {
            "schema_version": LOCK_SCHEMA,
            "status": "complete_pass",
            "passed_spec_ids": [spec.spec_id for spec in context.specs],
            "common_fit_grid": fit_decision["locked_common_fit_grid"],
            **dict(eap_decision["locked_common_eap_profile"]),
            "fit_comparison_all_six_passed": dict(
                fit_decision["fit_comparison_all_six_passed"]
            ),
            "effective_global_profile": effective_profile,
            "all_six_exact_specifications_share_profile": True,
            "config_sha256": _sha256(context.config_path),
            "study_signature_sha256": context.study_signature["canonical_sha256"],
            "parent_manifest_sha256": context.config["parent_study"][
                "study_manifest_sha256"
            ],
            "parent_decision_sha256": context.config["parent_study"][
                "decision_sha256"
            ],
            "historical_v1_aborted_marker_sha256": history["v1_aborted"][
                "aborted_marker_sha256"
            ],
            "historical_v1_output_tree_sha256": history["v1_aborted"][
                "output_tree_sha256"
            ],
            "historical_v2_manifest_sha256": history["v2_terminal"][
                "manifest_sha256"
            ],
            "historical_v2_decision_sha256": history["v2_terminal"][
                "decision_sha256"
            ],
            "historical_v2_output_tree_sha256": history["v2_terminal"][
                "output_tree_sha256"
            ],
            "historical_fit_or_checkpoint_artifacts_reused": 0,
            "followup_decision_sha256": _sha256(decision_path),
            "evidence_sha256": evidence_hashes,
            "calibration_specification_selected": None,
            "selection_performed": False,
            "cat_results_inspected": False,
            "historical_parent_result": "preserved_blocked_numerical_equivalence",
            "historical_v1_attempt": "preserved_aborted_before_fit_comparison",
            "historical_v2_result": "preserved_blocked_non_promotable_provenance_mismatch",
            "audit_disclosure": context.config["audit_disclosure"],
        }
        v2._write_json_once(lock_path, lock)
        v2._write_text_once(
            sha_path, f"{_sha256(lock_path)}  numerical_followup_lock.json\n"
        )
    elif lock_path.exists() or sha_path.exists():
        raise FollowupV3Error("failed v3 follow-up must not retain a lock")
    manifest_path = context.out_dir / "study_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "running":
        raise FollowupV3Error("terminal v3 manifest evidence cannot be replaced")
    manifest.update(
        {
            "status": decision["status"],
            "finished_at": v2.v1.parent_runner._utcnow(),
            "decision_sha256": _sha256(decision_path),
            "lock_sha256": _sha256(lock_path) if lock_path.is_file() else None,
            "parent_failure_preserved": True,
            "v1_abort_preserved": True,
            "v2_terminal_preserved": True,
            "historical_fit_or_checkpoint_artifacts_reused": 0,
            "selection_performed": False,
            "cat_results_inspected": False,
        }
    )
    _atomic_json(manifest_path, manifest)
    return decision


def _install_runtime_contract() -> None:
    v2.CONFIG_SCHEMA = CONFIG_SCHEMA
    v2.RUNNER_SCHEMA = RUNNER_SCHEMA
    v2.CHECKPOINT_SCHEMA = CHECKPOINT_SCHEMA
    v2.LOCK_SCHEMA = LOCK_SCHEMA
    v2.OUTPUT_LEAF = OUTPUT_LEAF
    v2.DEFAULT_CONFIG = DEFAULT_CONFIG
    v2.CODE_DEPENDENCIES = CODE_DEPENDENCIES
    v2._prepare = _prepare
    v2._finalize = _finalize


def run(context: v2.v1.Context, *, resume: bool) -> dict[str, Any]:
    _install_runtime_contract()
    return v2.run(context, resume=resume)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        context = load_context(args.config, args.out_dir)
        if args.plan_only:
            if args.resume:
                raise FollowupV3Error("--plan-only cannot combine with --resume")
            print(json.dumps(runtime_schedule(context), indent=2, sort_keys=True))
            return 0
        decision = run(context, resume=args.resume)
        print(json.dumps(decision, indent=2, sort_keys=True))
        return 0 if decision["passed"] else 2
    except (
        FollowupV3Error,
        v2.FollowupV2Error,
        v2.v1.FollowupError,
        v2.v1.parent_runner.NumericalCheckError,
        cm.CalibrationError,
        ValueError,
        OSError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
