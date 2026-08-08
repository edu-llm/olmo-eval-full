#!/usr/bin/env python
"""Generate a strict ``olmo-eval run-modes`` config for a live two-machine CAT run.

The generated YAML wires a local candidate tutor (``vllm_server`` on the eval job's
GPU) to a remote frozen Qwen judge (``vllm_server`` reached over ``base_url`` on a
separate machine).  Every frozen judge value is imported from the package
(``olmo_eval.edullm.judge`` / ``olmo_eval.edullm.mode``) so the config can never drift
from the code the runner validates against.  The bank block points at conformed
artifacts (see ``conform_flow_package_bank.py``) and the CAT block is filled from the
target benchmark's locked, per-benchmark operating point -- for biggen the values are
the ``FLOW_PACKAGE.md`` locked point (SE 0.12, floor 8, ridge 0.01, MWLE, EAP stop).

Nothing here is benchmark-hardcoded except the biggen CAT defaults; pass a different
benchmark's manifest and operating point to target another bank.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from olmo_eval.edullm.judge import (
    ADAPTER_VERSION,
    FAILURE_PROBABILITY_THRESHOLD,
    PROMPT_VERSION,
    QWEN_FAIL_TOKEN_IDS,
    QWEN_JUDGE_MODEL,
    QWEN_JUDGE_REVISION,
    QWEN_PASS_TOKEN_IDS,
)
from olmo_eval.edullm.mode import ATOMIC_REQUIREMENT_POLICY
from olmo_eval.edullm.mode import MODE_CONFIG_SCHEMA_VERSION as ADAPTIVE_SCHEMA_VERSION
from olmo_eval.runners.mode_config import MODE_CONFIG_SCHEMA_VERSION as TOP_SCHEMA_VERSION

# Provenance of the calibration-time grader recorded in judge_frozen.yaml. The
# integration judge below reuses the same Qwen weights + revision but a different
# prompt/adapter lineage; see the metadata.judge_policy_provenance block.
_CALIBRATION_JUDGE = {"prompt_version": "judge-validation-v3", "adapter": "generic-binary"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bank-manifest",
        required=True,
        type=Path,
        help="manifest.json produced by conform_flow_package_bank.py.",
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", required=True, help="New/empty results directory.")
    parser.add_argument("--out-config", required=True, type=Path)

    # Remote judge (machine A).
    parser.add_argument(
        "--judge-base-url",
        required=True,
        help="OpenAI-compatible base URL of the remote frozen Qwen server, e.g. "
        "http://JUDGE_HOST:8000/v1.",
    )

    # Local candidate tutor (machine B / the eval job).
    parser.add_argument("--tutor-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--tutor-revision", default="main")
    parser.add_argument("--tutor-model-family", default="qwen")
    parser.add_argument("--tutor-max-tokens", type=int, default=2048)

    # Per-benchmark locked CAT operating point (biggen defaults).
    parser.add_argument("--max-se", type=float, default=0.12)
    parser.add_argument("--min-scenarios", type=int, default=8)
    parser.add_argument("--min-evals-per-skill", type=int, default=0)
    parser.add_argument(
        "--max-scenarios",
        type=int,
        default=12,
        help="Smoke cap. Raise for a full run (biggen deploys ~19 scenarios median).",
    )
    parser.add_argument("--mwle-ridge", type=float, default=0.01)
    parser.add_argument("--stop-se-method", default="eap")
    parser.add_argument("--selection", default="trace")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--policy-id", default=None)
    parser.add_argument(
        "--op-point-source",
        default="FLOW_PACKAGE.md locked operating point",
        help="Human-readable provenance for the CAT operating point.",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip the local load_mode_config parse/validation of the emitted config.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    manifest = json.loads(args.bank_manifest.read_text(encoding="utf-8"))
    bank_dir = args.bank_manifest.resolve().parent
    benchmark_id = str(manifest["benchmark_id"])
    calibration_version = str(manifest["calibration_version"])
    skills_order = list(manifest["skills_order"])
    rubrics_path = bank_dir / "rubrics.jsonl"
    scenarios_path = bank_dir / "scenarios.jsonl"
    manifest_path = args.bank_manifest.resolve()
    for path in (rubrics_path, scenarios_path, manifest_path):
        if not path.is_file():
            raise SystemExit(f"bank artifact missing next to manifest: {path}")

    policy_id = args.policy_id or (
        f"{benchmark_id}-{args.stop_se_method}-se{args.max_se}-floor{args.min_scenarios}"
    )

    judge_kwargs = {
        "language_model_only": True,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.9,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    config: dict[str, Any] = {
        "schema_version": TOP_SCHEMA_VERSION,
        "run_id": args.run_id,
        "output_dir": args.output_dir,
        "continue_on_mode_failure": False,
        "metadata": {
            "benchmark": benchmark_id,
            "bank_release": calibration_version,
            "topology": "two_machine_live_cat",
            "operating_point": {
                "source": args.op_point_source,
                "max_se": args.max_se,
                "min_scenarios": args.min_scenarios,
                "mwle_ridge": args.mwle_ridge,
                "estimator": "mwle",
                "stop_se_method": args.stop_se_method,
                "selection": args.selection,
                "top_n": args.top_n,
                "seed": args.seed,
                "note": "Operating point is per-benchmark and locked; do not reuse across banks.",
            },
            "judge_policy_provenance": {
                "assumption": (
                    "The integration frozen judge is treated as equivalent to the bank's "
                    "calibration-time grader. Same Qwen weights and revision; the prompt and "
                    "adapter lineage differ."
                ),
                "status": "unverified - no document asserts equivalence",
                "calibration_time_judge": _CALIBRATION_JUDGE,
                "integration_judge": {
                    "prompt_version": PROMPT_VERSION,
                    "adapter_version": ADAPTER_VERSION,
                },
                "action": "revisit if smoke pass-rates look anomalous",
            },
        },
        "harness": {
            "name": f"edullm-two-machine-{benchmark_id}",
            "provider": {
                "kind": "vllm_server",
                "model": args.tutor_model,
                "revision": args.tutor_revision,
                "dtype": "bfloat16",
                "max_model_len": 32768,
                "num_instances": 1,
                "kwargs": {"tensor_parallel_size": 1, "gpu_memory_utilization": 0.9},
            },
            "auxiliary_providers": {
                "judge": {
                    "kind": "vllm_server",
                    "model": QWEN_JUDGE_MODEL,
                    "revision": QWEN_JUDGE_REVISION,
                    "tokenizer": QWEN_JUDGE_MODEL,
                    "base_url": args.judge_base_url,
                    "dtype": "bfloat16",
                    "max_model_len": 32768,
                    "trust_remote_code": False,
                    "num_instances": 1,
                    "kwargs": judge_kwargs,
                },
            },
        },
        "modes": [
            {
                "name": "edullm_adaptive",
                "config": {
                    "schema_version": ADAPTIVE_SCHEMA_VERSION,
                    "bank": {
                        "rubrics_path": str(rubrics_path),
                        "scenarios_path": str(scenarios_path),
                        "manifest_path": str(manifest_path),
                        "skills_order": skills_order,
                        "benchmark_id": benchmark_id,
                        "calibration_version": calibration_version,
                        "policy_id": policy_id,
                        "scientific_status": "experimental",
                    },
                    "tutor": {
                        "expected_model": args.tutor_model,
                        "model_family": args.tutor_model_family,
                        "model_provenance": {
                            "source": "olmo_eval_harness_provider",
                            "revision": args.tutor_revision,
                        },
                        "generation": {
                            "max_tokens": args.tutor_max_tokens,
                            "temperature": 0.0,
                            "top_p": 1.0,
                            "top_k": None,
                            "stop_sequences": None,
                            "num_samples": 1,
                            "do_sample": False,
                        },
                    },
                    "judge": {
                        "provider": "judge",
                        "model": QWEN_JUDGE_MODEL,
                        "model_family": "qwen",
                        "revision": QWEN_JUDGE_REVISION,
                        "prompt_version": PROMPT_VERSION,
                        "adapter_version": ADAPTER_VERSION,
                        "enable_thinking": False,
                        "language_model_only": True,
                        "pass_token_ids": list(QWEN_PASS_TOKEN_IDS),
                        "fail_token_ids": list(QWEN_FAIL_TOKEN_IDS),
                        "failure_probability_threshold": FAILURE_PROBABILITY_THRESHOLD,
                        "atomic_requirement_policy": ATOMIC_REQUIREMENT_POLICY,
                    },
                    "quadrature": {
                        "nodes_per_dim": 21,
                        "max_nodes": 200000,
                        "method": "gauss_hermite",
                        "linear_bound": 8.0,
                    },
                    "cat": {
                        "max_se": args.max_se,
                        "min_evals_per_skill": args.min_evals_per_skill,
                        "min_scenarios": args.min_scenarios,
                        "max_scenarios": args.max_scenarios,
                        "seed": args.seed,
                        "top_n": args.top_n,
                        "selection": args.selection,
                        "stop_se_method": args.stop_se_method,
                        "theta_init": None,
                        "covariance_init_diag": None,
                        "mwle_ridge": args.mwle_ridge,
                    },
                },
            }
        ],
    }

    args.out_config.parent.mkdir(parents=True, exist_ok=True)
    args.out_config.write_text(
        yaml.safe_dump(config, sort_keys=False, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )

    result: dict[str, Any] = {
        "status": "generated",
        "out_config": str(args.out_config),
        "benchmark_id": benchmark_id,
        "judge_base_url": args.judge_base_url,
        "tutor_model": args.tutor_model,
        "operating_point": {
            "max_se": args.max_se,
            "min_scenarios": args.min_scenarios,
            "max_scenarios": args.max_scenarios,
            "mwle_ridge": args.mwle_ridge,
            "stop_se_method": args.stop_se_method,
        },
    }

    if not args.no_validate:
        # Structure/frozen-judge/provider parse validation. This does NOT contact the
        # remote judge (/version) or load the bank; those happen at run time and in the
        # orchestrator preflight once the judge server is reachable.
        from olmo_eval.runners.mode_config import load_mode_config

        parsed = load_mode_config(args.out_config)
        result["validated"] = "load_mode_config_ok"
        result["parsed_modes"] = list(parsed.mode_names)

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
