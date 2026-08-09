#!/usr/bin/env python
"""Diagnose why a gateway model produces unscorable judge outputs.

Grades the 261 human cases with the chosen adapter, but captures the RAW model text,
finish_reason, and token usage for every call, then classifies each unscorable output
(truncated / fenced / thinking-prefixed / empty / other) and dumps samples. This tells us
whether Gemini's unscorable rate is truncation (raise max_tokens / shorten fields),
markdown fences (strip them), reasoning tokens (disable thinking), or something else.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any

import httpx

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_api_judge_pilot import ADAPTERS, parse_env, parse_verdict  # noqa: E402


def _load_cases(path: Path, limit: int | None) -> list[dict[str, Any]]:
    cases = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    return cases[:limit] if limit else cases


def _classify(content: str, finish_reason: str, completion_tokens: int | None, max_tokens: int) -> str:
    if not content or not content.strip():
        return "empty"
    verdict, reason = parse_verdict(content)
    if reason is None:
        return "ok"
    truncated = finish_reason.lower() in {"length", "max_tokens"} or (
        completion_tokens is not None and completion_tokens >= max_tokens - 2
    )
    if truncated:
        return "unscorable_truncated"
    if "```" in content:
        return "unscorable_fenced"
    # substantial non-JSON text before the first brace suggests reasoning/preamble
    brace = content.find("{")
    if brace > 40 or brace == -1:
        return "unscorable_prose_or_thinking"
    return "unscorable_other"


async def _one(client, base_url, api_key, model, case, adapter, max_tokens, sem):
    payload = {
        "model": model,
        "messages": ADAPTERS[adapter](case),
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    async with sem:
        try:
            r = await client.post(f"{base_url}/chat/completions", json=payload, headers=headers)
            r.raise_for_status()
            data = r.json()
            choice = data["choices"][0]
            content = choice.get("message", {}).get("content") or ""
            finish = str(choice.get("finish_reason") or "")
            usage = data.get("usage") or {}
            ctoks = usage.get("completion_tokens")
            return {
                "case_id": case["case_id"],
                "finish_reason": finish,
                "completion_tokens": ctoks,
                "klass": _classify(content, finish, ctoks, max_tokens),
                "content": content,
            }
        except Exception as exc:  # noqa: BLE001
            return {"case_id": case["case_id"], "finish_reason": "ERROR", "completion_tokens": None,
                    "klass": "request_error", "content": f"{type(exc).__name__}: {exc}"}


async def _amain() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases", type=Path, default=Path(
        "AdaptiveTesting/Inputs/Open/LLM-Judge/aws_judge_handoff_extracted/"
        "aws_judge_handoff/inputs/judge_cases.blinded.jsonl"))
    p.add_argument("--env", type=Path, default=Path("eduLLM-Evals/.env"))
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", choices=sorted(ADAPTERS), default="generic-binary-strict")
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", type=Path, default=Path("eduLLM-Evals/api_judge_pilot/diagnose"))
    args = p.parse_args()

    env = parse_env(args.env)
    base_url = env.get("MODEL_API_BASE", "").rstrip("/")
    api_key = env.get("MODEL_API_KEY", "")
    cases = _load_cases(args.cases, args.limit)

    sem = asyncio.Semaphore(args.concurrency)
    start = time.perf_counter()
    async with httpx.AsyncClient(timeout=180.0, limits=httpx.Limits(max_connections=args.concurrency + 8)) as client:
        results = await asyncio.gather(*[
            _one(client, base_url, api_key, args.model, c, args.adapter, args.max_tokens, sem)
            for c in cases
        ])
    wall = time.perf_counter() - start

    counts: dict[str, int] = {}
    ctoks_unscorable: list[int] = []
    for r in results:
        counts[r["klass"]] = counts.get(r["klass"], 0) + 1
        if r["klass"].startswith("unscorable") and r["completion_tokens"]:
            ctoks_unscorable.append(r["completion_tokens"])

    args.out.mkdir(parents=True, exist_ok=True)
    tag = args.model.replace("/", "__") + f".{args.adapter}.mt{args.max_tokens}"
    (args.out / f"{tag}.raw.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results), encoding="utf-8")

    print(f"model={args.model} adapter={args.adapter} max_tokens={args.max_tokens} "
          f"n={len(results)} wall={wall:.1f}s")
    print("class breakdown: " + json.dumps(dict(sorted(counts.items())), indent=2))
    if ctoks_unscorable:
        ctoks_unscorable.sort()
        print(f"unscorable completion_tokens: min={ctoks_unscorable[0]} "
              f"median={ctoks_unscorable[len(ctoks_unscorable)//2]} max={ctoks_unscorable[-1]} "
              f"(max_tokens={args.max_tokens})")
    # dump a few samples per unscorable class
    shown: dict[str, int] = {}
    for r in results:
        k = r["klass"]
        if k.startswith("unscorable") and shown.get(k, 0) < 2:
            shown[k] = shown.get(k, 0) + 1
            body = r["content"]
            head = body[:300].replace("\n", "\\n").encode("ascii", "replace").decode("ascii")
            tail = body[-200:].replace("\n", "\\n").encode("ascii", "replace").decode("ascii")
            print(f"\n[{k}] {r['case_id']} finish={r['finish_reason']} ctoks={r['completion_tokens']}")
            print(f"  HEAD: {head}")
            print(f"  TAIL: {tail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_amain()))
