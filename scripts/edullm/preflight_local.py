#!/usr/bin/env python
"""Local, no-GPU/no-network preflight for a generated two-machine run config.

``olmo-eval run-modes --check`` performs the full read-only preflight, but two of its
checks cannot run on a laptop: local GPU detection for the candidate tutor, and the
live ``{base_url}/version`` probe of the remote judge. This helper exercises every
other part of ``ModeRunOrchestrator.preflight`` -- output-dir validation, provider
resolution, required-auxiliary wiring, and the frozen adaptive/judge config checks --
by supplying an empty GPU set and a stub judge runtime checker.

It proves the config is structurally runnable and the two-machine wiring resolves; run
the real ``olmo-eval run-modes --config ... --check`` on the platform once the judge
server is reachable to close the deferred version probe.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from olmo_eval.edullm.judge import QWEN_JUDGE_MODEL, QWEN_JUDGE_REVISION
from olmo_eval.inference.providers.config import ProviderConfig
from olmo_eval.runners.mode_config import load_mode_config
from olmo_eval.runners.mode_orchestrator import ModeRunOrchestrator


def _stub_judge_runtime(provider: ProviderConfig) -> Mapping[str, Any]:
    """Validate the remote-judge wiring without contacting the endpoint."""
    if not provider.base_url:
        raise SystemExit(
            "judge provider has no base_url; a two-machine live run requires an external "
            "vllm_server endpoint for the frozen judge"
        )
    if provider.model != QWEN_JUDGE_MODEL or provider.revision != QWEN_JUDGE_REVISION:
        raise SystemExit("judge provider model/revision are not the frozen Qwen judge")
    return {
        "deployment": "external_endpoint",
        "base_url": provider.base_url,
        "version": "deferred: run olmo-eval run-modes --check against the live judge",
        "explicit_token_logprobs": "required_by_frozen_request_contract",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    config = load_mode_config(args.config)
    orchestrator = ModeRunOrchestrator(
        config,
        runtime_checker=_stub_judge_runtime,
        available_gpu_ids=(),
    )
    preflight = orchestrator.preflight()
    print(
        json.dumps(
            {
                "status": "local_preflight_passed",
                "run_id": config.run_id,
                "selected_modes": list(preflight.selected_modes),
                "provider_names": list(preflight.provider_names),
                "required_auxiliary_providers": list(preflight.required_auxiliary_providers),
                "judge_runtime": dict(preflight.judge_runtime or {}),
                "deferred": [
                    "candidate GPU detection (needs the eval job's GPU)",
                    "live judge /version probe (needs the judge server)",
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
