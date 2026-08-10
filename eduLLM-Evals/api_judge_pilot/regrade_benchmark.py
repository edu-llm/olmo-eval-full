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
import hashlib
import inspect
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_api_judge_pilot import ADAPTERS, parse_env, parse_verdict  # noqa: E402

ERROR_FINISH = "error"
SCHEMA_VERSION = "frontier-regrade-v2"
PROMPT_VERSION_DEFAULT = "generic-binary-strict-v1"
TERMINAL_STATUSES = {"ok", "auto_fail"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"JSONL file does not exist: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Expected a JSON object in {path}:{line_number}")
        rows.append(row)
    return rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _combined_response_hash(response_files: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in response_files:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


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
    scenario_rows = _load_jsonl(scenarios_path)
    if not rubrics:
        raise ValueError(f"Rubric bank is empty: {rubrics_path}")
    if not scenario_rows:
        raise ValueError(f"Scenario bank is empty: {scenarios_path}")

    scenarios: dict[str, dict[str, Any]] = {}
    for scenario in scenario_rows:
        sid = str(scenario.get("scenario_id") or "")
        if not sid:
            raise ValueError("Scenario row is missing scenario_id")
        if sid in scenarios:
            raise ValueError(f"Duplicate scenario_id in bank: {sid}")
        scenarios[sid] = scenario

    # criteria grouped by scenario, deterministic order
    criteria_by_scenario: dict[str, list[dict[str, Any]]] = {}
    criterion_ids: set[str] = set()
    for r in rubrics:
        cid = str(r.get("criterion_id") or "")
        sid = str(r.get("scenario_id") or "")
        if not cid or not sid:
            raise ValueError("Rubric row is missing criterion_id or scenario_id")
        if cid in criterion_ids:
            raise ValueError(f"Duplicate criterion_id in bank: {cid}")
        if sid not in scenarios:
            raise ValueError(f"Criterion {cid} references unknown scenario {sid}")
        criterion_ids.add(cid)
        criteria_by_scenario.setdefault(sid, []).append(r)

    if not responses_dir.is_dir():
        raise ValueError(f"Responses directory does not exist: {responses_dir}")
    response_files = sorted(responses_dir.glob("*.responses.jsonl"))
    if not response_files:
        raise ValueError(f"No *.responses.jsonl files found in {responses_dir}")
    if model_limit is not None:
        if model_limit < 1:
            raise ValueError("--model-limit must be at least 1")
        response_files = response_files[:model_limit]

    cases: list[dict[str, Any]] = []
    counts = {
        "response_files": len(response_files),
        "models": 0,
        "scenarios": len(scenarios),
        "criteria": len(criterion_ids),
        "auto_fail_error": 0,
        "auto_fail_empty": 0,
        "auto_fail_missing": 0,
        "gradable": 0,
    }
    seen_models: set[str] = set()
    for rf in response_files:
        rows = _load_jsonl(rf)
        if not rows:
            raise ValueError(f"Response file is empty: {rf}")
        row_models = {str(row.get("Model")) for row in rows if row.get("Model")}
        if len(row_models) > 1:
            raise ValueError(f"Inconsistent Model values within {rf}: {sorted(row_models)}")
        model = next(iter(row_models), _norm_model_name(rf.stem))
        if model in seen_models:
            raise ValueError(f"Duplicate tutor model across response files: {model}")
        seen_models.add(model)
        counts["models"] += 1
        by_scenario: dict[str, dict[str, Any]] = {}
        for row in rows:
            sid = str(row.get("Scenario") or "")
            if not sid:
                raise ValueError(f"Response row in {rf} is missing Scenario")
            if sid in by_scenario:
                raise ValueError(f"Duplicate Scenario {sid} in {rf}")
            by_scenario[sid] = row
        unknown_scenarios = set(by_scenario) - set(scenarios)
        if unknown_scenarios:
            raise ValueError(
                f"Response file {rf} contains {len(unknown_scenarios)} unknown scenarios"
            )
        for sid, crits in criteria_by_scenario.items():
            scenario = scenarios[sid]
            row = by_scenario.get(sid)
            output = (row or {}).get("Output")
            finish = str((row or {}).get("Finish Reason") or "")
            if row is None:
                auto_fail, reason = 1, "missing_response"
                counts["auto_fail_missing"] += len(crits)
            elif finish == ERROR_FINISH:
                auto_fail, reason = 1, "finish_reason_error"
                counts["auto_fail_error"] += len(crits)
            elif not str(output or "").strip():
                auto_fail, reason = 1, "empty_output"
                counts["auto_fail_empty"] += len(crits)
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
    expected_case_count = counts["models"] * counts["criteria"]
    if len(cases) != expected_case_count:
        raise ValueError(
            f"Case matrix is incomplete: built {len(cases)}, expected {expected_case_count}"
        )
    return cases, counts


def _record_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row["model"]),
        str(row["scenario_id"]),
        str(row["criterion_id"]),
    )


def _load_terminal_records(verdicts_path: Path) -> set[tuple[str, str, str]]:
    records: set[tuple[str, str, str]] = set()
    if not verdicts_path.is_file():
        return records
    lines = verdicts_path.read_text(encoding="utf-8").splitlines()
    nonempty_indexes = [index for index, line in enumerate(lines) if line.strip()]
    last_nonempty = nonempty_indexes[-1] if nonempty_indexes else -1
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            if index == last_nonempty:
                raise ValueError(
                    f"Truncated final line in {verdicts_path}; preserve and repair it before resume"
                ) from exc
            raise ValueError(f"Invalid JSON at {verdicts_path}:{index + 1}") from exc
        status = str(row.get("status") or "")
        if status not in TERMINAL_STATUSES:
            raise ValueError(
                f"Non-terminal status {status!r} found in canonical verdict file; "
                "errors must be written to errors.jsonl"
            )
        key = _record_key(row)
        if key in records:
            raise ValueError(f"Duplicate terminal verdict in {verdicts_path}: {key}")
        records.add(key)
    return records


def _resolve_gateway(env: dict[str, str]) -> tuple[str, str, str, str]:
    base_key = "MODEL_API_BASE" if env.get("MODEL_API_BASE") else "TFY_BASE_URL"
    api_key_name = "MODEL_API_KEY" if env.get("MODEL_API_KEY") else "TFY_API_KEY"
    base_url = env.get(base_key, "").rstrip("/")
    api_key = env.get(api_key_name, "")
    return base_url, api_key, base_key, api_key_name


def _prompt_template_hash(adapter: str) -> str:
    sentinel = {
        "scenario_prompt": "SENTINEL INSTRUCTION",
        "conversation_context": [{"role": "student", "content": "SENTINEL CONTEXT"}],
        "reference_solution": "SENTINEL REFERENCE",
        "expected_evidence": ["SENTINEL EXPECTED EVIDENCE"],
        "candidate_response": "SENTINEL RESPONSE",
        "criterion": "SENTINEL CRITERION",
    }
    rendered = json.dumps(
        ADAPTERS[adapter](sentinel), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return _sha256_text(rendered)


def _config_fingerprint(config: dict[str, Any]) -> str:
    return _sha256_text(
        json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def _prepare_manifest(
    args: argparse.Namespace,
    effective_max_tokens: int,
    counts: dict[str, int],
    response_files: list[Path],
    credential_source: tuple[str, str],
) -> dict[str, Any]:
    frozen_config: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "judge_name": args.judge_name,
        "judge_model": args.model,
        "adapter": args.adapter,
        "prompt_version": args.prompt_version,
        "prompt_template_sha256": _prompt_template_hash(args.adapter),
        "temperature": 0.0,
        "max_tokens": effective_max_tokens,
        "response_format": {"type": "json_object"},
        "parser_version": "json-verdict-regex-recovery-fail-closed-v1",
        "model_limit": args.model_limit,
        "inputs": {
            "responses_dir": str(args.responses_dir.resolve()),
            "responses_sha256": _combined_response_hash(response_files),
            "rubrics": str(args.rubrics.resolve()),
            "rubrics_sha256": _sha256_file(args.rubrics),
            "scenarios": str(args.scenarios.resolve()),
            "scenarios_sha256": _sha256_file(args.scenarios),
        },
        "expected": {
            "models": counts["models"],
            "scenarios": counts["scenarios"],
            "criteria": counts["criteria"],
            "cells": counts["models"] * counts["criteria"],
            "gradable": counts["gradable"],
            "auto_fail": (
                counts["auto_fail_error"]
                + counts["auto_fail_empty"]
                + counts["auto_fail_missing"]
            ),
        },
        "selection_provenance": {
            "source_branch": "origin/frq-lab",
            "source_commit": "de6cac1eabd91dd3a1f6536f75b206348b7419a9",
            "checkpoint": "tutoreval_gemini3_v1_2026-08-09",
        },
        "runtime_sha256": {
            "regrade_benchmark.py": _sha256_file(Path(__file__).resolve()),
            "run_api_judge_pilot.py": _sha256_file(
                Path(__file__).resolve().with_name("run_api_judge_pilot.py")
            ),
        },
    }
    return {
        "frozen_config": frozen_config,
        "config_fingerprint": _config_fingerprint(frozen_config),
        "git_commit": _git_commit(),
        "credential_source": {
            "base_url_variable": credential_source[0],
            "api_key_variable": credential_source[1],
        },
        "created_at": _utc_now(),
        "status": "starting",
    }


def _validate_or_write_manifest(manifest_path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("config_fingerprint") != manifest["config_fingerprint"]:
            raise ValueError(
                f"Output directory contains a different frozen configuration: {manifest_path}"
            )
        return existing
    _atomic_json(manifest_path, manifest)
    return manifest


async def _grade_one(
    client: httpx.AsyncClient, base_url: str, api_key: str, model: str,
    case: dict[str, Any], sem: asyncio.Semaphore, adapter: str, max_tokens: int,
    max_retries: int, out_lock: asyncio.Lock, out_handle: Any, error_handle: Any,
    judge_name: str, prompt_version: str, config_fingerprint: str,
) -> dict[str, Any]:
    base_record = {
        "schema_version": SCHEMA_VERSION,
        "model": case["model"],
        "scenario_id": case["scenario_id"],
        "criterion_id": case["criterion_id"],
        "judge_name": judge_name,
        "judge_model": model,
        "adapter": adapter,
        "prompt_version": prompt_version,
        "config_fingerprint": config_fingerprint,
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    if case["auto_fail"] == 1:
        record = {
            **base_record,
            "verdict": "fail", "status": "auto_fail", "reason": case["auto_fail_reason"],
            "latency_s": 0.0, "prompt_tokens": 0, "completion_tokens": 0,
            "raw_output": "", "prompt_sha256": None, "recorded_at": _utc_now(),
        }
        async with out_lock:
            out_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_handle.flush()
        return record

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    async with sem:
        # Build the potentially large prompt only after acquiring the semaphore.
        # Otherwise a 90k-cell run materializes every prompt in memory at once.
        messages = ADAPTERS[adapter](case)
        prompt_serialized = json.dumps(
            messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
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
                    **base_record,
                    "verdict": verdict, "status": "ok", "reason": unscorable or "",
                    "latency_s": round(time.perf_counter() - start, 3),
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "raw_output": text,
                    "prompt_sha256": _sha256_text(prompt_serialized),
                    "finish_reason": data["choices"][0].get("finish_reason"),
                    "request_id": resp.headers.get("x-request-id"),
                    "provider_model": data.get("model"),
                    "attempts": attempt + 1,
                    "recorded_at": _utc_now(),
                }
                async with out_lock:
                    out_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_handle.flush()
                return record
            except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
                    retry_after = exc.response.headers.get("retry-after")
                    try:
                        delay = min(60.0, max(1.0, float(retry_after or 0)))
                    except ValueError:
                        delay = min(30.0, 2.0 ** attempt)
                else:
                    delay = min(30.0, 2.0 ** attempt)
                if attempt + 1 < max_retries:
                    await asyncio.sleep(delay)
        record = {
            **base_record,
            "verdict": None, "status": "error", "reason": "request_failed",
            "latency_s": round(time.perf_counter() - start, 3),
            "prompt_tokens": None, "completion_tokens": None, "error": last_error,
            "raw_output": "", "prompt_sha256": _sha256_text(prompt_serialized),
            "attempts": max_retries, "recorded_at": _utc_now(),
        }
        async with out_lock:
            error_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            error_handle.flush()
        return record


def _summarize_verdicts(
    verdicts_path: Path,
    expected_keys: set[tuple[str, str, str]],
    *,
    wall_s: float,
    new_cells: int,
    new_judge_calls: int,
    errors_this_invocation: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    seen: set[tuple[str, str, str]] = set()
    status_counts = {"ok": 0, "auto_fail": 0}
    passes = 0
    unscorable = 0
    prompt_tokens = 0
    completion_tokens = 0
    tokenized_judgments = 0
    samples: list[dict[str, Any]] = []
    with verdicts_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = _record_key(row)
            if key in seen:
                raise ValueError(f"Duplicate verdict at {verdicts_path}:{line_number}: {key}")
            seen.add(key)
            status = str(row.get("status") or "")
            if status not in status_counts:
                raise ValueError(f"Unexpected status in canonical verdicts: {status!r}")
            status_counts[status] += 1
            passes += int(row.get("verdict") == "pass")
            if status == "ok":
                unscorable += int(bool(row.get("reason")))
                if row.get("prompt_tokens") is not None:
                    prompt_tokens += int(row["prompt_tokens"])
                    completion_tokens += int(row.get("completion_tokens") or 0)
                    tokenized_judgments += 1
                if len(samples) < 5:
                    samples.append(
                        {
                            "model": row["model"],
                            "criterion_id": row["criterion_id"],
                            "verdict": row["verdict"],
                            "reason": row.get("reason") or "",
                        }
                    )

    extra = seen - expected_keys
    missing = expected_keys - seen
    if extra:
        raise ValueError(f"Canonical verdict file has {len(extra)} unexpected cells")
    total = len(seen)
    judge_calls_total = status_counts["ok"]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "expected_cells": len(expected_keys),
        "cells_total": total,
        "cells_new_this_invocation": new_cells,
        "coverage": round(total / len(expected_keys), 6) if expected_keys else 0.0,
        "missing_cells": len(missing),
        "judge_calls_total": judge_calls_total,
        "judge_calls_this_invocation": new_judge_calls,
        "auto_failed_total": status_counts["auto_fail"],
        "errors_this_invocation": errors_this_invocation,
        "unscorable_total": unscorable,
        "pass_rate_all_cells": round(passes / total, 6) if total else 0.0,
        "throughput_this_invocation": {
            "wall_s": round(wall_s, 1),
            "judge_calls_per_s": round(new_judge_calls / wall_s, 2) if wall_s else 0.0,
        },
        "tokens_total": {
            "prompt": prompt_tokens,
            "completion": completion_tokens,
            "judgments_with_usage": tokenized_judgments,
            "mean_prompt": round(prompt_tokens / max(1, tokenized_judgments), 1),
            "mean_completion": round(completion_tokens / max(1, tokenized_judgments), 1),
        },
        "completed": not missing and errors_this_invocation == 0,
        "updated_at": _utc_now(),
    }
    return summary, samples


async def _amain() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses-dir", type=Path, required=True)
    parser.add_argument("--rubrics", type=Path, required=True)
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument("--env", type=Path, default=Path("eduLLM-Evals/.env"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", default="claude-group/claude-sonnet-4-6")
    parser.add_argument("--judge-name", default="frontier-api-judge")
    parser.add_argument("--adapter", choices=sorted(ADAPTERS), default="generic-binary-strict")
    parser.add_argument("--prompt-version", default=PROMPT_VERSION_DEFAULT)
    parser.add_argument("--concurrency", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--model-limit", type=int, default=1, help="Grade the first N model files.")
    parser.add_argument("--expected-models", type=int, default=None)
    parser.add_argument("--expected-scenarios", type=int, default=None)
    parser.add_argument("--expected-criteria", type=int, default=None)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--allow-unauthenticated", action="store_true")
    parser.add_argument("--qwen-matrix", type=Path, default=None, help="Optional Qwen matrix for sanity agreement.")
    parser.add_argument(
        "--extrapolate-criteria",
        type=int,
        default=None,
        help="Total criteria for a full-run cost/time extrapolation (e.g. biggen 2678 + tutoreval 1793 = 4471).",
    )
    parser.add_argument("--extrapolate-models", type=int, default=100)
    args = parser.parse_args()

    if args.concurrency < 1 or args.batch_size < 1 or args.max_retries < 1:
        raise SystemExit("concurrency, batch-size, and max-retries must all be positive")

    cases, counts = build_cases(args.responses_dir, args.rubrics, args.scenarios, args.model_limit)
    for label, expected, actual in (
        ("models", args.expected_models, counts["models"]),
        ("scenarios", args.expected_scenarios, counts["scenarios"]),
        ("criteria", args.expected_criteria, counts["criteria"]),
    ):
        if expected is not None and expected != actual:
            raise SystemExit(f"Expected {expected} {label}, found {actual}")

    # Safety: gemini-3 is verbose (long internal reasoning) and truncates at low caps,
    # which fail-closes cells on truncation rather than judgment. Never grade it below 6144.
    effective_max_tokens = args.max_tokens
    if "gemini-3" in args.model and effective_max_tokens < 6144:
        print(f"NOTE: bumping --max-tokens {args.max_tokens} -> 6144 for {args.model} "
              "(verbose; 4096 still truncated ~2.4% of full-biggen cells -> fake fails; see FINDINGS)")
        effective_max_tokens = 6144

    response_files = sorted(args.responses_dir.glob("*.responses.jsonl"))
    if args.model_limit is not None:
        response_files = response_files[: args.model_limit]
    expected_keys = {
        (c["model"], c["scenario_id"], c["criterion_id"]) for c in cases
    }
    preflight = {
        "judge_model": args.model,
        "adapter": args.adapter,
        "prompt_version": args.prompt_version,
        "max_tokens": effective_max_tokens,
        "counts": counts,
        "expected_cells": len(expected_keys),
        "prompt_template_sha256": _prompt_template_hash(args.adapter),
    }
    print(json.dumps({"preflight": preflight}, indent=2), flush=True)
    if args.preflight_only:
        return 0

    env = parse_env(args.env)
    base_url, api_key, base_key, api_key_name = _resolve_gateway(env)
    if not base_url:
        raise SystemExit(
            f"Neither MODEL_API_BASE nor TFY_BASE_URL was found in {args.env}"
        )
    if not api_key and not args.allow_unauthenticated:
        raise SystemExit(
            f"Neither MODEL_API_KEY nor TFY_API_KEY was found in {args.env}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    verdicts_path = args.out_dir / "verdicts.jsonl"
    errors_path = args.out_dir / "errors.jsonl"
    manifest_path = args.out_dir / "manifest.json"
    manifest = _prepare_manifest(
        args,
        effective_max_tokens,
        counts,
        response_files,
        (base_key, api_key_name),
    )
    manifest = _validate_or_write_manifest(manifest_path, manifest)

    done = _load_terminal_records(verdicts_path)
    extra_existing = done - expected_keys
    if extra_existing:
        raise SystemExit(
            f"Existing verdicts contain {len(extra_existing)} cells outside this input matrix"
        )
    todo = [
        c
        for c in cases
        if (c["model"], c["scenario_id"], c["criterion_id"]) not in done
    ]
    print(
        f"cases={len(cases)} already_done={len(done)} todo={len(todo)} "
        f"max_tokens={effective_max_tokens} auto_fail_counts={counts}",
        flush=True,
    )

    sem = asyncio.Semaphore(args.concurrency)
    out_lock = asyncio.Lock()
    limits = httpx.Limits(max_connections=args.concurrency + 8, max_keepalive_connections=args.concurrency + 8)
    start = time.perf_counter()
    new_cells = 0
    new_judge_calls = 0
    errors_this_invocation = 0
    with (
        verdicts_path.open("a", encoding="utf-8") as out_handle,
        errors_path.open("a", encoding="utf-8") as error_handle,
    ):
        async with httpx.AsyncClient(timeout=args.timeout, limits=limits) as client:
            for offset in range(0, len(todo), args.batch_size):
                batch = todo[offset : offset + args.batch_size]
                batch_results = await asyncio.gather(
                    *(
                        _grade_one(
                            client,
                            base_url,
                            api_key,
                            args.model,
                            case,
                            sem,
                            args.adapter,
                            effective_max_tokens,
                            args.max_retries,
                            out_lock,
                            out_handle,
                            error_handle,
                            args.judge_name,
                            args.prompt_version,
                            manifest["config_fingerprint"],
                        )
                        for case in batch
                    )
                )
                new_cells += sum(
                    1 for result in batch_results if result["status"] in TERMINAL_STATUSES
                )
                new_judge_calls += sum(
                    1 for result in batch_results if result["status"] != "auto_fail"
                )
                errors_this_invocation += sum(
                    1 for result in batch_results if result["status"] == "error"
                )
                out_handle.flush()
                error_handle.flush()
                os.fsync(out_handle.fileno())
                os.fsync(error_handle.fileno())
                completed = min(offset + len(batch), len(todo))
                print(
                    f"progress={completed}/{len(todo)} new_terminal={new_cells} "
                    f"errors={errors_this_invocation}",
                    flush=True,
                )
    wall = time.perf_counter() - start

    summary, samples = _summarize_verdicts(
        verdicts_path,
        expected_keys,
        wall_s=wall,
        new_cells=new_cells,
        new_judge_calls=new_judge_calls,
        errors_this_invocation=errors_this_invocation,
    )
    summary = {
        **summary,
        "model": args.model,
        "adapter": args.adapter,
        "judge_name": args.judge_name,
        "prompt_version": args.prompt_version,
        "max_tokens": effective_max_tokens,
        "concurrency": args.concurrency,
        "batch_size": args.batch_size,
        "config_fingerprint": manifest["config_fingerprint"],
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
        compared = 0
        agree = 0
        with verdicts_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                gold_key = (str(row["model"]), str(row["criterion_id"]))
                if row["status"] == "ok" and not row.get("reason") and gold_key in gold:
                    compared += 1
                    agree += int((row["verdict"] == "pass") == (gold[gold_key] == 1))
        if compared:
            summary["qwen_sanity"] = {
                "compared_cells": compared,
                "agreement": round(agree / compared, 4),
            }

    if args.extrapolate_criteria:
        total_calls = args.extrapolate_criteria * args.extrapolate_models
        rate = summary["throughput_this_invocation"]["judge_calls_per_s"] or 1.0
        summary["extrapolation"] = {
            "assumes_models": args.extrapolate_models,
            "assumes_criteria": args.extrapolate_criteria,
            "total_judge_calls": total_calls,
            "est_wall_hours_at_this_rate": round(total_calls / rate / 3600, 2),
            "est_prompt_tokens": total_calls * summary["tokens_total"]["mean_prompt"],
            "est_completion_tokens": total_calls * summary["tokens_total"]["mean_completion"],
        }

    _atomic_json(args.out_dir / "summary.json", summary)
    manifest["status"] = "complete" if summary["completed"] else "incomplete"
    manifest["last_updated_at"] = _utc_now()
    manifest["summary_sha256"] = _sha256_file(args.out_dir / "summary.json")
    manifest["summary"] = {
        "expected_cells": summary["expected_cells"],
        "cells_total": summary["cells_total"],
        "missing_cells": summary["missing_cells"],
        "errors_this_invocation": summary["errors_this_invocation"],
    }
    _atomic_json(manifest_path, manifest)
    print(json.dumps(summary, indent=2))

    # sample verdicts to skim quality
    print("\n=== sample verdicts (first 5 judged) ===")
    for r in samples:
        print(f"{r['criterion_id']}: {r['verdict']}  (reason={r.get('reason') or '-'})")
    return 0 if summary["completed"] else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_amain()))
