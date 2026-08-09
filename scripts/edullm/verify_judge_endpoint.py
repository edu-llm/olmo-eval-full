#!/usr/bin/env python
"""Verify a remote frozen-judge endpoint the way the runner will use it.

This reproduces the two checks that matter for the live two-machine CAT run:

1. ``ModeRunOrchestrator._validate_external_qwen_runtime`` requires ``{base_url}/version``
   to report vLLM ``0.26.0``; and
2. the frozen judge request contract needs explicit ``logprob_token_ids`` support, which
   this script probes with a one-token completion carrying the frozen P/F token IDs.

Standard library only, so it runs on the judge machine without installing olmo-eval.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

REQUIRED_VLLM_VERSION = "0.26.0"
# Mirror of QWEN_PASS_TOKEN_IDS / QWEN_FAIL_TOKEN_IDS in src/olmo_eval/edullm/judge.py.
DEFAULT_PASS_TOKEN_IDS = [47, 387, 9729]
DEFAULT_FAIL_TOKEN_IDS = [37, 426, 12362]


def _version_url(base_url: str) -> str:
    trimmed = base_url.rstrip("/")
    if trimmed.endswith("/v1"):
        trimmed = trimmed[: -len("/v1")]
    return f"{trimmed}/version"


def _get_json(url: str, timeout: float) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def _post_json(url: str, payload: dict, timeout: float) -> tuple[int, dict | str]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(  # noqa: S310
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="e.g. http://JUDGE_HOST:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument(
        "--skip-logprob-probe",
        action="store_true",
        help="Only check /version (do not send a completion).",
    )
    args = parser.parse_args()

    result: dict = {"base_url": args.base_url, "checks": {}}
    ok = True

    version_url = _version_url(args.base_url)
    try:
        payload = _get_json(version_url, args.timeout)
        version = payload.get("version")
        version_ok = version == REQUIRED_VLLM_VERSION
        result["checks"]["version"] = {
            "url": version_url,
            "reported": version,
            "required": REQUIRED_VLLM_VERSION,
            "ok": version_ok,
        }
        ok = ok and version_ok
    except (urllib.error.URLError, ValueError, OSError) as exc:
        result["checks"]["version"] = {"url": version_url, "ok": False, "error": str(exc)}
        ok = False

    if not args.skip_logprob_probe:
        completions_url = f"{args.base_url.rstrip('/')}/completions"
        probe = {
            "model": args.model,
            "prompt": "P",
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": len(DEFAULT_PASS_TOKEN_IDS),
            "logprob_token_ids": DEFAULT_PASS_TOKEN_IDS + DEFAULT_FAIL_TOKEN_IDS,
        }
        status, body = _post_json(completions_url, probe, args.timeout)
        logprob_ok = status == 200
        result["checks"]["logprob_token_ids"] = {
            "url": completions_url,
            "status": status,
            "ok": logprob_ok,
            "detail": body if not logprob_ok else "accepted explicit logprob_token_ids",
        }
        ok = ok and logprob_ok

    result["status"] = "ok" if ok else "failed"
    print(json.dumps(result, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
