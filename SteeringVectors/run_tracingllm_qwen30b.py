#!/usr/bin/env python3
"""TracingLLM protocol on Qwen3-30B-A3B-Thinking (three HF checkpoints)."""

from __future__ import annotations

from pathlib import Path

import run_tracingllm_olmoe7b as job


def main() -> int:
    default_manifest = Path(__file__).resolve().parent / "checkpoints_qwen30b-thinking.json"
    argv = job.sys.argv
    if "--manifest" not in argv:
        argv = [*argv[:1], "--manifest", str(default_manifest), *argv[1:]]
        job.sys.argv = argv
    return job.main()


if __name__ == "__main__":
    raise SystemExit(main())
