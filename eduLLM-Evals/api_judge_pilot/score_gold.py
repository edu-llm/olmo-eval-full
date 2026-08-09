#!/usr/bin/env python
"""Score judge candidates against a stratified human-gold set, full metric picture.

Grades each gold cell with each candidate (chosen adapter, JSON mode, temp 0) and reports,
per candidate: false-pass rate (overall / per-capability / population-weighted), accuracy,
false-fail, balanced accuracy, per-class F1 + macro-F1, MCC, coverage/unscorable, plus
test-retest flip rate (self-consistency across replicates) and prompt-flip rate (verdict
changes under benign prompt perturbations). Also reports the frozen Qwen baseline for free
from `_qwen_verdict`. Optimize on the full set, not just false-pass.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_api_judge_pilot import ADAPTERS, parse_env, parse_verdict  # noqa: E402

DEFAULT_MODELS = [
    "claude-group/claude-sonnet-4-6",
    "claude-group/claude-opus-4-6",
    "gemini-group/gemini-2.5-flash",
    "gemini-group/gemini-3-flash-preview",
    "openai-group/gpt-4.1",
]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def _perturb(messages: list[dict[str, str]], variant: str) -> list[dict[str, str]]:
    msgs = [dict(m) for m in messages]
    content = msgs[-1]["content"]
    if variant == "whitespace":
        content = "\n" + content.replace("\n\n", "\n \n\n") + "\n  "
    elif variant == "politeness":
        content = (
            "Please read the following carefully and evaluate it objectively and impartially.\n\n"
            + content
        )
    msgs[-1]["content"] = content
    return msgs


def _metrics(pairs: list[tuple[str, str, str]], pop_share: dict[str, float]) -> dict[str, Any]:
    tp = sum(1 for v, g, _ in pairs if v == "pass" and g == "pass")
    tn = sum(1 for v, g, _ in pairs if v == "fail" and g == "fail")
    fp = sum(1 for v, g, _ in pairs if v == "pass" and g == "fail")
    fn = sum(1 for v, g, _ in pairs if v == "fail" and g == "pass")
    n = len(pairs)
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    prec_pass = tp / (tp + fp) if (tp + fp) else 0.0
    f1_pass = 2 * prec_pass * tpr / (prec_pass + tpr) if (prec_pass + tpr) else 0.0
    prec_fail = tn / (tn + fn) if (tn + fn) else 0.0
    f1_fail = 2 * prec_fail * tnr / (prec_fail + tnr) if (prec_fail + tnr) else 0.0
    mcc_den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / mcc_den) if mcc_den else 0.0

    caps: dict[str, list[tuple[str, str, str]]] = {}
    for p in pairs:
        caps.setdefault(p[2], []).append(p)
    per_cap_fp: dict[str, Any] = {}
    wnum = wden = 0.0
    for cap, cp in caps.items():
        cfp = sum(1 for v, g, _ in cp if v == "pass" and g == "fail")
        cfails = sum(1 for _, g, _ in cp if g == "fail")
        per_cap_fp[cap] = {"fp": cfp, "gold_fails": cfails, "fp_rate": _rate(cfp, cfails)}
        if cfails and cap in pop_share:
            wnum += pop_share[cap] * (cfp / cfails)
            wden += pop_share[cap]
    return {
        "n": n,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "accuracy": _rate(tp + tn, n),
        "false_pass_rate": _rate(fp, fp + tn),
        "false_pass_rate_weighted": round(wnum / wden, 4) if wden else None,
        "false_fail_rate": _rate(fn, fn + tp),
        "balanced_accuracy": round((tpr + tnr) / 2, 4),
        "f1_pass": round(f1_pass, 4),
        "f1_fail": round(f1_fail, 4),
        "macro_f1": round((f1_pass + f1_fail) / 2, 4),
        "mcc": round(mcc, 4),
        "per_capability_fp": per_cap_fp,
    }


async def _grade(client, base_url, api_key, model, cases, adapter, max_tokens, concurrency, variant=None):
    sem = asyncio.Semaphore(concurrency)

    async def one(case):
        messages = ADAPTERS[adapter](case)
        if variant:
            messages = _perturb(messages, variant)
        payload = {"model": model, "messages": messages, "temperature": 0.0,
                   "max_tokens": max_tokens, "response_format": {"type": "json_object"}}
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        async with sem:
            for attempt in range(4):
                try:
                    r = await client.post(f"{base_url}/chat/completions", json=payload, headers=headers)
                    r.raise_for_status()
                    text = r.json()["choices"][0]["message"]["content"] or ""
                    verdict, unscorable = parse_verdict(text)
                    return case["gold_case_id"], verdict, bool(unscorable)
                except (httpx.HTTPError, KeyError, IndexError, ValueError):
                    await asyncio.sleep(1.0 * (attempt + 1))
            return case["gold_case_id"], "fail", True

    rows = await asyncio.gather(*[one(c) for c in cases])
    return {cid: (v, u) for cid, v, u in rows}


def _flip_rate(a: dict[str, tuple[str, bool]], b: dict[str, tuple[str, bool]]) -> float | None:
    keys = [k for k in a if k in b]
    if not keys:
        return None
    return round(sum(1 for k in keys if a[k][0] != b[k][0]) / len(keys), 4)


async def _amain() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gold", type=Path, required=True)
    p.add_argument("--sample", type=Path, required=True)
    p.add_argument("--sample-manifest", type=Path, required=True)
    p.add_argument("--env", type=Path, default=Path("eduLLM-Evals/.env"))
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--adapter", choices=sorted(ADAPTERS), default="generic-binary-strict")
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--gemini3-max-tokens", type=int, default=4096)
    p.add_argument("--concurrency", type=int, default=48)
    p.add_argument("--replicates", type=int, default=2, help="Runs per model for test-retest flip.")
    p.add_argument("--variants", nargs="*", default=["whitespace", "politeness"],
                   help="Benign prompt perturbations for prompt-flip; empty to skip.")
    args = p.parse_args()

    env = parse_env(args.env)
    base_url = env.get("MODEL_API_BASE", "").rstrip("/")
    api_key = env.get("MODEL_API_KEY", "")

    gold = {g["gold_case_id"]: g for g in _load_jsonl(args.gold) if g.get("gold_label")}
    sample = {s["gold_case_id"]: s for s in _load_jsonl(args.sample)}
    manifest = json.loads(args.sample_manifest.read_text(encoding="utf-8"))
    pop = manifest.get("capability_population") or manifest.get("stratum_population") or {}
    total_pop = sum(pop.values()) or 1
    pop_share = {k: v / total_pop for k, v in pop.items()}
    cases = [sample[cid] for cid in gold if cid in sample]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"gold={len(gold)} cases={len(cases)} adapter={args.adapter} "
          f"replicates={args.replicates} variants={args.variants}")

    def _stratum(g):
        return g.get("capability") or g.get("stratum") or "?"

    def _pairs(vmap):
        return [(vmap[cid][0], gold[cid]["gold_label"], _stratum(gold[cid]))
                for cid in gold if cid in vmap]

    summaries = {}
    qwen_cells = {cid for cid in gold if gold[cid].get("_qwen_verdict") is not None}
    if qwen_cells:
        qwen = {cid: ("pass" if int(gold[cid]["_qwen_verdict"]) == 1 else "fail", False)
                for cid in qwen_cells}
        summaries["_qwen_baseline"] = _metrics(_pairs(qwen), pop_share)
    else:
        print("(no _qwen_verdict in gold -> skipping Qwen baseline)")

    async with httpx.AsyncClient(timeout=180.0,
                                 limits=httpx.Limits(max_connections=args.concurrency + 8)) as client:
        for model in args.models:
            mt = args.gemini3_max_tokens if "gemini-3" in model else args.max_tokens
            start = time.perf_counter()
            reps = []
            for _ in range(max(1, args.replicates)):
                reps.append(await _grade(client, base_url, api_key, model, cases, args.adapter, mt, args.concurrency))
            m = _metrics(_pairs(reps[0]), pop_share)
            m["unscorable"] = sum(1 for cid in reps[0] if reps[0][cid][1])
            m["max_tokens"] = mt
            m["test_retest_flip"] = _flip_rate(reps[0], reps[1]) if len(reps) > 1 else None
            prompt_flips = {}
            for variant in (args.variants or []):
                vmap = await _grade(client, base_url, api_key, model, cases, args.adapter, mt, args.concurrency, variant)
                prompt_flips[variant] = _flip_rate(reps[0], vmap)
            m["prompt_flip"] = prompt_flips
            m["prompt_flip_worst"] = max([v for v in prompt_flips.values() if v is not None], default=None)
            m["wall_s"] = round(time.perf_counter() - start, 1)
            summaries[model] = m
            safe = model.replace("/", "__")
            (args.out_dir / f"{safe}.gold_verdicts.jsonl").write_text(
                "".join(json.dumps({"gold_case_id": cid, "verdict": reps[0][cid][0]}, ensure_ascii=False) + "\n"
                        for cid in reps[0]), encoding="utf-8")

    (args.out_dir / f"summary.{args.adapter}.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    hdr = (f"{'model':38} {'FP%':>5} {'FPwt%':>6} {'macroF1':>8} {'MCC':>6} {'acc':>5} "
           f"{'FF%':>5} {'retest%':>8} {'pflip%':>7} {'unsc':>5}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for name, m in summaries.items():
        print(f"{name[:38]:38} {(m['false_pass_rate'] or 0)*100:>5.1f} "
              f"{(m.get('false_pass_rate_weighted') or 0)*100:>6.1f} {m['macro_f1']*100:>8.1f} "
              f"{m['mcc']:>6.3f} {(m['accuracy'] or 0)*100:>5.1f} {(m['false_fail_rate'] or 0)*100:>5.1f} "
              f"{(m.get('test_retest_flip') or 0)*100:>8.1f} {(m.get('prompt_flip_worst') or 0)*100:>7.1f} "
              f"{m.get('unscorable', 0):>5}")
    print(f"\nWrote {args.out_dir / f'summary.{args.adapter}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_amain()))
