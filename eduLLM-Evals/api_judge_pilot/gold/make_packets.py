#!/usr/bin/env python
"""Generate dual-proposer gold-labeling packets for the sampled biggen cells.

For each sampled case, two INDEPENDENT proposers (opus-4.8 + gpt-5.5) each produce a
careful, structured PASS/FAIL judgment with evidence. Proposers are blinded: they see
only the scenario + criterion + response, never any judge verdict or the Qwen reference.

Concordance rule (per the reviewer's spec): if both proposers agree, that label is the
`recommendation`; if they disagree, `recommendation` is null and both options are shown.
The human adjudicates every case; `adjudicated_label` starts blank.

The gold-labeler prompt is deliberately MORE thorough than the fast judge prompt (decompose
the criterion, cite evidence for and against, decide only on the response's own content) so
the gold is higher quality than the judge under test, not a replica of it.

Resumable: rows already in packets.jsonl are skipped. Run with --limit for a smoke test.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_api_judge_pilot import parse_env  # noqa: E402

# opus-4.8 and sonnet-5 are catalog-listed but denied by the gateway's Bedrock backend
# (marketplace subscription not enabled). opus-5 is the strongest available independent
# proposer (newer than the opus-4-6 judge candidate), substituted for opus-4.8.
PROPOSERS = {
    "opus-5": "claude-group/claude-opus-5",
    "gpt-5.5": "openai-group/gpt-5.5",
}

_SYSTEM = (
    "You are an expert grader constructing a GOLD-STANDARD label for a single evaluation "
    "criterion. Your label will be reviewed by a human and used as ground truth to test "
    "automated judges, so be more careful and thorough than a fast judge. Evaluate only "
    "the one criterion given, using only evidence present in the candidate response itself."
)

_POLICY = (
    "Decision policy:\n"
    "1. Apply ONLY this one criterion. Ignore general quality or unrelated correct content.\n"
    "2. If the criterion has multiple required parts, EVERY part must be satisfied by the "
    "response's own content. Missing, partial, vague, merely implied, incorrect, or "
    "contradicted parts => FAIL.\n"
    "3. The task, context, and any reference tell you what to look for; they cannot supply "
    "content on the response's behalf.\n"
    "4. Grade on meaning, not exact wording: equivalent phrasing, synonyms, mathematically "
    "equivalent notation/work, reordering, and rounding are acceptable when actually "
    "expressed and correct, unless the criterion demands an exact form.\n"
    '5. For "must not"/prohibition criteria, PASS iff the response avoids the prohibited '
    "content.\n"
    "6. When genuinely uncertain, prefer FAIL and lower your confidence."
)

_OUTPUT = (
    "Respond with ONLY a JSON object:\n"
    '{"required_parts": ["<each required part of the criterion>"],\n'
    ' "evidence_for": "<quote/point to response text satisfying parts, or empty>",\n'
    ' "evidence_against": "<what is missing/partial/contradicted, or empty>",\n'
    ' "label": "pass" | "fail",\n'
    ' "confidence": <0.0-1.0>,\n'
    ' "rationale": "<=2 sentences"}'
)


def render_case(case: dict[str, Any]) -> str:
    parts = [f"# Task / student request\n{case.get('scenario_prompt', '')}".rstrip()]
    ctx = case.get("conversation_context") or []
    if ctx:
        turns = "\n".join(f"[{t.get('role', '?')}] {t.get('content', '')}" for t in ctx)
        parts.append(f"# Prior conversation\n{turns}")
    ref = str(case.get("reference_solution") or "").strip()
    if ref:
        parts.append(f"# Reference / background (cannot supply missing response content)\n{ref}")
    parts.append(f"# Criterion to evaluate (the ONLY thing being graded)\n{case.get('criterion', '')}")
    parts.append(f"# Candidate response to grade\n{case.get('candidate_response', '')}")
    return "\n\n".join(parts)


def build_payload(model_slug: str, case: dict[str, Any], max_tokens: int) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": f"{_SYSTEM}\n\n{_POLICY}\n\n{_OUTPUT}"},
        {"role": "user", "content": render_case(case)},
    ]
    payload: dict[str, Any] = {
        "model": model_slug,
        "messages": messages,
        "response_format": {"type": "json_object"},
    }
    # GPT-5 series: reasoning models want max_completion_tokens and only default temperature.
    if model_slug.startswith("openai-group/gpt-5"):
        payload["max_completion_tokens"] = max_tokens
    else:
        payload["max_tokens"] = max_tokens
        payload["temperature"] = 0.0
    return payload


def parse_proposal(text: str) -> dict[str, Any]:
    raw = text or ""
    start, end = raw.find("{"), raw.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(raw[start : end + 1])
            label = str(obj.get("label", "")).strip().lower()
            if label in ("pass", "fail"):
                return {
                    "label": label,
                    "confidence": obj.get("confidence"),
                    "required_parts": obj.get("required_parts"),
                    "evidence_for": obj.get("evidence_for"),
                    "evidence_against": obj.get("evidence_against"),
                    "rationale": obj.get("rationale"),
                    "status": "ok",
                }
        except (ValueError, TypeError):
            pass
    return {"label": None, "status": "parse_error", "raw": raw[:800]}


async def call_proposer(
    client: httpx.AsyncClient, base_url: str, api_key: str, name: str, slug: str,
    case: dict[str, Any], max_tokens: int, max_retries: int, sem: asyncio.Semaphore,
) -> dict[str, Any]:
    payload = build_payload(slug, case, max_tokens)
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    async with sem:
        last_error = None
        for attempt in range(max_retries):
            try:
                resp = await client.post(f"{base_url}/chat/completions", json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                text = data["choices"][0]["message"]["content"] or ""
                out = parse_proposal(text)
                out["proposer"] = name
                out["model_slug"] = slug
                return out
            except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                body = getattr(getattr(exc, "response", None), "text", "")
                if body:
                    last_error += f" | {body[:300]}"
                await asyncio.sleep(1.5 * (attempt + 1))
        return {"proposer": name, "model_slug": slug, "label": None,
                "status": "error", "error": last_error}


async def process_case(client, base_url, api_key, case, max_tokens, max_retries, sem,
                       out_lock, out_handle) -> dict[str, Any]:
    proposals = await asyncio.gather(*[
        call_proposer(client, base_url, api_key, name, slug, case, max_tokens, max_retries, sem)
        for name, slug in PROPOSERS.items()
    ])
    by_name = {p["proposer"]: p for p in proposals}
    labels = [by_name[n].get("label") for n in PROPOSERS]
    both_ok = all(lbl in ("pass", "fail") for lbl in labels)
    concordant = both_ok and labels[0] == labels[1]
    recommendation = labels[0] if concordant else None
    packet = {
        "gold_case_id": case["gold_case_id"],
        "model": case["model"],
        "scenario_id": case["scenario_id"],
        "criterion_id": case["criterion_id"],
        "stratum": case.get("stratum", case.get("capability")),
        "criticality": case.get("criticality"),
        "task": case.get("task", ""),
        "criterion": case["criterion"],
        "scenario_prompt": case["scenario_prompt"],
        "conversation_context": case["conversation_context"],
        "reference_solution": case["reference_solution"],
        "candidate_response": case["candidate_response"],
        "response_truncated": case.get("response_truncated", False),
        "proposals": by_name,
        "concordant": concordant,
        "recommendation": recommendation,
        "adjudicated_label": None,
        "adjudicator_notes": "",
    }
    async with out_lock:
        out_handle.write(json.dumps(packet, ensure_ascii=False) + "\n")
        out_handle.flush()
    return packet


async def amain() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--env", type=Path, default=Path("eduLLM-Evals/.env"))
    ap.add_argument("--max-tokens", type=int, default=1400)
    ap.add_argument("--max-retries", type=int, default=4)
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--limit", type=int, default=None, help="Smoke test: only first N cases.")
    args = ap.parse_args()

    env = parse_env(args.env)
    base_url = env.get("MODEL_API_BASE", "").rstrip("/")
    api_key = env.get("MODEL_API_KEY", "")
    if not base_url:
        raise SystemExit(f"MODEL_API_BASE not found in {args.env}")

    cases = [json.loads(l) for l in args.sample.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        cases = cases[: args.limit]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.out.is_file():
        for l in args.out.read_text(encoding="utf-8").splitlines():
            if l.strip():
                done.add(json.loads(l)["gold_case_id"])
    todo = [c for c in cases if c["gold_case_id"] not in done]
    print(f"cases={len(cases)} done={len(done)} todo={len(todo)} proposers={list(PROPOSERS)}")

    sem = asyncio.Semaphore(args.concurrency)
    out_lock = asyncio.Lock()
    start = time.perf_counter()
    limits = httpx.Limits(max_connections=args.concurrency * 2 + 4)
    with args.out.open("a", encoding="utf-8") as out_handle:
        async with httpx.AsyncClient(timeout=args.timeout, limits=limits) as client:
            results = await asyncio.gather(*[
                process_case(client, base_url, api_key, c, args.max_tokens,
                             args.max_retries, sem, out_lock, out_handle)
                for c in todo
            ])
    wall = time.perf_counter() - start

    concordant = sum(1 for r in results if r["concordant"])
    parse_issues = sum(
        1 for r in results for p in r["proposals"].values() if p.get("status") != "ok"
    )
    rec_pass = sum(1 for r in results if r["recommendation"] == "pass")
    rec_fail = sum(1 for r in results if r["recommendation"] == "fail")
    print(json.dumps({
        "processed": len(results),
        "wall_s": round(wall, 1),
        "concordant": concordant,
        "discordant_needs_review": len(results) - concordant,
        "recommendation_pass": rec_pass,
        "recommendation_fail": rec_fail,
        "proposer_call_issues": parse_issues,
    }, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(amain()))
