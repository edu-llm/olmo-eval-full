#!/usr/bin/env python
"""Regrade a benchmark's tutor responses with the frontier API judge (resumable).

Benchmark-agnostic: give it a responses directory (Full200Run open/<benchmark>/*.responses.jsonl),
the benchmark's rubrics.jsonl + scenarios.jsonl, and it builds one judge case per
(model, scenario, criterion), applies the stage_judge_inputs auto-fail policy
(generation-error / empty / missing -> fail WITHOUT a judge call; length-truncated ->
graded), grades the rest with the strict binary prompt + JSON mode, and writes verdicts
incrementally so a long run is restart-safe.

Reports throughput, token volume, pass rate, unscorable counts, and (optionally) a light
sanity agreement vs a Qwen verdict matrix -- plus a cost/time extrapolation to a full run.

Reuses the prompt/parse/env helpers from run_api_judge_pilot.py (same directory).
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_api_judge_pilot import ADAPTERS, parse_env, parse_verdict  # noqa: E402

ERROR_FINISH = "error"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _norm_model_name(file_stem: str) -> str:
    # "01-ai__Yi-1.5-6B" -> "01-ai/Yi-1.5-6B"
    return file_stem.replace(".responses", "").replace("__", "/")


def build_cases(
    responses_dir: Path,
    rubrics_path: Path,
    scenarios_path: Path,
    model_limit: int | None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rubrics = _load_jsonl(rubrics_path)
    scenarios = {str(s["scenario_id"]): s for s in _load_jsonl(scenarios_path)}
    # criteria grouped by scenario, deterministic order
    criteria_by_scenario: dict[str, list[dict[str, Any]]] = {}
    for r in rubrics:
        criteria_by_scenario.setdefault(str(r["scenario_id"]), []).append(r)

    response_files = sorted(responses_dir.glob("*.responses.jsonl"))
    if model_limit is not None:
        response_files = response_files[:model_limit]

    cases: list[dict[str, Any]] = []
    counts = {"auto_fail_error": 0, "auto_fail_empty": 0, "auto_fail_missing": 0, "gradable": 0}
    for rf in response_files:
        rows = _load_jsonl(rf)
        model = str(rows[0].get("Model") or _norm_model_name(rf.stem)) if rows else _norm_model_name(rf.stem)
        by_scenario = {str(row.get("Scenario")): row for row in rows}
        for sid, crits in criteria_by_scenario.items():
            scenario = scenarios.get(sid, {})
            row = by_scenario.get(sid)
            output = (row or {}).get("Output")
            finish = str((row or {}).get("Finish Reason") or "")
            if row is None:
                auto_fail, reason = 1, "missing_response"
                counts["auto_fail_missing"] += 1
            elif finish == ERROR_FINISH:
                auto_fail, reason = 1, "finish_reason_error"
                counts["auto_fail_error"] += 1
            elif not str(output or "").strip():
                auto_fail, reason = 1, "empty_output"
                counts["auto_fail_empty"] += 1
            else:
                auto_fail, reason = 0, ""
            for r in crits:
                if auto_fail == 0:
                    counts["gradable"] += 1
                cases.append(
                    {
                        "model": model,
                        "scenario_id": sid,
                        "criterion_id": str(r["criterion_id"]),
                        "criterion": str(r.get("criterion") or ""),
                        "scenario_prompt": scenario.get("prompt", ""),
                        "conversation_context": scenario.get("conversation_context") or [],
                        "reference_solution": scenario.get("reference_solution") or "",
                        "expected_evidence": r.get("expected_evidence") or [],
                        "candidate_response": output or "",
                        "auto_fail": auto_fail,
                        "auto_fail_reason": reason,
                    }
                )
    return cases, counts


def _load_done(verdicts_path: Path) -> set[tuple[str, str]]:
    done: set[tuple[str, str]] = set()
    if verdicts_path.is_file():
        for line in verdicts_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done.add((row["model"], row["criterion_id"]))
    return done


async def _grade_one(
    client: httpx.AsyncClient, base_url: str, api_key: str, model: str,
    case: dict[str, Any], sem: asyncio.Semaphore, adapter: str, max_tokens: int,
    max_retries: int, out_lock: asyncio.Lock, out_handle: Any,
) -> dict[str, Any]:
    key = (case["model"], case["criterion_id"])
    if case["auto_fail"] == 1:
        record = {
            "model": case["model"], "criterion_id": case["criterion_id"],
            "verdict": "fail", "status": "auto_fail", "reason": case["auto_fail_reason"],
            "latency_s": 0.0, "prompt_tokens": 0, "completion_tokens": 0,
        }
        async with out_lock:
            out_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_handle.flush()
        return record

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
        start = time.perf_counter()
        last_error = None
        for attempt in range(max_retries):
            try:
                resp = await client.post(f"{base_url}/chat/completions", json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                text = data["choices"][0]["message"]["content"] or ""
                verdict, unscorable = parse_verdict(text)
                usage = data.get("usage") or {}
                record = {
                    "model": case["model"], "criterion_id": case["criterion_id"],
                    "verdict": verdict, "status": "ok", "reason": unscorable or "",
                    "latency_s": round(time.perf_counter() - start, 3),
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                }
                async with out_lock:
                    out_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_handle.flush()
                return record
            except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                await asyncio.sleep(1.0 * (attempt + 1))
        record = {
            "model": case["model"], "criterion_id": case["criterion_id"],
            "verdict": "fail", "status": "error", "reason": "request_failed",
            "latency_s": round(time.perf_counter() - start, 3),
            "prompt_tokens": None, "completion_tokens": None, "error": last_error,
        }
        async with out_lock:
            out_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_handle.flush()
        return record


async def _amain() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses-dir", type=Path, required=True)
    parser.add_argument("--rubrics", type=Path, required=True)
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument("--env", type=Path, default=Path("eduLLM-Evals/.env"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", default="claude-group/claude-sonnet-4-6")
    parser.add_argument("--adapter", choices=sorted(ADAPTERS), default="generic-binary-strict")
    parser.add_argument("--concurrency", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--model-limit", type=int, default=1, help="Grade the first N model files.")
    parser.add_argument("--qwen-matrix", type=Path, default=None, help="Optional Qwen matrix for sanity agreement.")
    parser.add_argument(
        "--extrapolate-criteria",
        type=int,
        default=None,
        help="Total criteria for a full-run cost/time extrapolation (e.g. biggen 2678 + tutoreval 1793 = 4471).",
    )
    parser.add_argument("--extrapolate-models", type=int, default=100)
    args = parser.parse_args()

    env = parse_env(args.env)
    base_url = env.get("MODEL_API_BASE", "").rstrip("/")
    api_key = env.get("MODEL_API_KEY", "")
    if not base_url:
        raise SystemExit(f"MODEL_API_BASE not found in {args.env}")

    cases, counts = build_cases(args.responses_dir, args.rubrics, args.scenarios, args.model_limit)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    verdicts_path = args.out_dir / "verdicts.jsonl"
    done = _load_done(verdicts_path)
    todo = [c for c in cases if (c["model"], c["criterion_id"]) not in done]
    print(f"cases={len(cases)} already_done={len(done)} todo={len(todo)} auto_fail_counts={counts}")

    sem = asyncio.Semaphore(args.concurrency)
    out_lock = asyncio.Lock()
    limits = httpx.Limits(max_connections=args.concurrency + 8, max_keepalive_connections=args.concurrency + 8)
    start = time.perf_counter()
    with verdicts_path.open("a", encoding="utf-8") as out_handle:
        async with httpx.AsyncClient(timeout=args.timeout, limits=limits) as client:
            tasks = [
                _grade_one(client, base_url, api_key, args.model, case, sem,
                           args.adapter, args.max_tokens, args.max_retries, out_lock, out_handle)
                for case in todo
            ]
            results = await asyncio.gather(*tasks)
    wall = time.perf_counter() - start

    # --- metrics over this run's graded cells ---
    judged = [r for r in results if r["status"] == "ok"]
    auto_failed = [r for r in results if r["status"] == "auto_fail"]
    errors = [r for r in results if r["status"] == "error"]
    passes = sum(1 for r in results if r["verdict"] == "pass")
    unscorable = sum(1 for r in judged if r.get("reason"))
    prompt_tokens = sum(r["prompt_tokens"] or 0 for r in judged)
    completion_tokens = sum(r["completion_tokens"] or 0 for r in judged)
    n_judge_calls = len(judged) + len(errors)

    summary: dict[str, Any] = {
        "model": args.model,
        "adapter": args.adapter,
        "concurrency": args.concurrency,
        "cells_this_run": len(results),
        "judge_calls": n_judge_calls,
        "auto_failed": len(auto_failed),
        "errors": len(errors),
        "unscorable": unscorable,
        "pass_rate_all_cells": round(passes / len(results), 4) if results else 0.0,
        "throughput": {
            "wall_s": round(wall, 1),
            "judge_calls_per_s": round(n_judge_calls / wall, 2) if wall else 0.0,
            "mean_prompt_tokens": round(prompt_tokens / max(1, len(judged)), 1),
            "mean_completion_tokens": round(completion_tokens / max(1, len(judged)), 1),
            "total_prompt_tokens": prompt_tokens,
            "total_completion_tokens": completion_tokens,
        },
    }

    if args.qwen_matrix and args.qwen_matrix.is_file():
        gold: dict[tuple[str, str], int] = {}
        with args.qwen_matrix.open(encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader)
            crit_cols = header[1:]
            for row in reader:
                m = row[0]
                for cid, val in zip(crit_cols, row[1:], strict=False):
                    if val in ("0", "1"):
                        gold[(m, cid)] = int(val)
        matched = [(r, gold[(r["model"], r["criterion_id"])]) for r in judged
                   if (r["model"], r["criterion_id"]) in gold and not r.get("reason")]
        if matched:
            agree = sum(1 for r, g in matched if (r["verdict"] == "pass") == (g == 1))
            summary["qwen_sanity"] = {
                "compared_cells": len(matched),
                "agreement": round(agree / len(matched), 4),
            }

    if args.extrapolate_criteria:
        total_calls = args.extrapolate_criteria * args.extrapolate_models
        rate = summary["throughput"]["judge_calls_per_s"] or 1.0
        summary["extrapolation"] = {
            "assumes_models": args.extrapolate_models,
            "assumes_criteria": args.extrapolate_criteria,
            "total_judge_calls": total_calls,
            "est_wall_hours_at_this_rate": round(total_calls / rate / 3600, 2),
            "est_prompt_tokens": total_calls * summary["throughput"]["mean_prompt_tokens"],
            "est_completion_tokens": total_calls * summary["throughput"]["mean_completion_tokens"],
        }

    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    # sample verdicts to skim quality
    print("\n=== sample verdicts (first 5 judged) ===")
    for r in judged[:5]:
        print(f"{r['criterion_id']}: {r['verdict']}  (reason={r.get('reason') or '-'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_amain()))
