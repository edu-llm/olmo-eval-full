#!/usr/bin/env python
"""Draw a representative core sample of gradable biggen cells for gold labeling.

A "cell" is one (model, scenario, criterion). We sample only *gradable* cells
(auto_fail==0: the response exists, finish!=error, output non-empty), because auto-fail
cells never reach the judge and are trivially correct. The sample is stratified by the
criterion's `capability` (biggen's skill axis; `criticality`/`primary_skill` are null for
biggen) with proportional allocation, drawn with a fixed seed for reproducibility.

Output: sample.jsonl (one materialized case per sampled cell, with response text +
scenario/criterion context + the Qwen reference verdict for later analysis) and
sample_manifest.json (population size, per-capability strata, seed, allocation) so the
downstream false-pass rate is a defensible probability-sample estimate.

Pass 1 indexes gradable (model, scenario) cheaply; pass 2 materializes response text only
for the sampled cells, so we never hold all response bodies in memory.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ERROR_FINISH = "error"


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _norm_model_name(file_stem: str) -> str:
    return file_stem.replace(".responses", "").replace("__", "/")


def load_rubrics(path: Path, stratify_field: str) -> dict[str, dict[str, Any]]:
    """criterion_id -> {scenario_id, criterion, stratum, criticality, task}."""
    out: dict[str, dict[str, Any]] = {}
    for r in _iter_jsonl(path):
        cid = str(r["criterion_id"])
        out[cid] = {
            "scenario_id": str(r["scenario_id"]),
            "criterion": str(r.get("criterion") or ""),
            "stratum": (r.get(stratify_field) or "unknown"),
            "criticality": r.get("criticality"),
            "task": (r.get("task") or ""),
            "expected_evidence": r.get("expected_evidence") or [],
        }
    return out


def load_scenarios(path: Path) -> dict[str, dict[str, Any]]:
    return {str(s["scenario_id"]): s for s in _iter_jsonl(path)}


def load_qwen_matrix(path: Path | None) -> dict[tuple[str, str], int]:
    gold: dict[tuple[str, str], int] = {}
    if not path or not path.is_file():
        return gold
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        crit_cols = header[1:]
        for row in reader:
            m = row[0]
            for cid, val in zip(crit_cols, row[1:], strict=False):
                if val in ("0", "1"):
                    gold[(m, cid)] = int(val)
    return gold


def index_gradable_scenarios(responses_dir: Path) -> dict[str, set[str]]:
    """model -> set of scenario_ids with a gradable response (streamed, text discarded)."""
    model_scen: dict[str, set[str]] = {}
    for rf in sorted(responses_dir.glob("*.responses.jsonl")):
        model = None
        gradable: set[str] = set()
        for row in _iter_jsonl(rf):
            model = model or str(row.get("Model") or _norm_model_name(rf.stem))
            sid = str(row.get("Scenario"))
            finish = str(row.get("Finish Reason") or "")
            output = str(row.get("Output") or "").strip()
            if finish != ERROR_FINISH and output:
                gradable.add(sid)
        model_scen[model or _norm_model_name(rf.stem)] = gradable
    return model_scen


def materialize_responses(
    responses_dir: Path, needed: set[tuple[str, str]]
) -> dict[tuple[str, str], dict[str, Any]]:
    """(model, scenario) -> {output, finish, truncated} for just the sampled cells."""
    by_model: dict[str, set[str]] = defaultdict(set)
    for m, s in needed:
        by_model[m].add(s)
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for rf in sorted(responses_dir.glob("*.responses.jsonl")):
        stem_model = _norm_model_name(rf.stem)
        # a file's model is authoritative from its rows; match either form
        for row in _iter_jsonl(rf):
            model = str(row.get("Model") or stem_model)
            if model not in by_model and stem_model not in by_model:
                break
            key_model = model if model in by_model else stem_model
            sid = str(row.get("Scenario"))
            if sid in by_model.get(key_model, set()):
                out[(key_model, sid)] = {
                    "output": str(row.get("Output") or ""),
                    "finish": str(row.get("Finish Reason") or ""),
                    "truncated": bool(row.get("Truncated") or False),
                }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--responses-dir", type=Path, required=True)
    ap.add_argument("--rubrics", type=Path, required=True)
    ap.add_argument("--scenarios", type=Path, required=True)
    ap.add_argument("--qwen-matrix", type=Path, default=None)
    ap.add_argument("--n", type=int, default=100, help="Target sample size.")
    ap.add_argument("--seed", type=int, default=20260809)
    ap.add_argument("--stratify-field", default="capability",
                    help="rubric field to stratify by (biggen: capability; tutoreval: primary_skill)")
    ap.add_argument("--benchmark", default="biggen")
    ap.add_argument("--extend", action="store_true",
                    help="Extend an existing out-dir/sample.jsonl to --target-n, drawing only NEW distinct cells.")
    ap.add_argument("--target-n", type=int, default=None, help="Total sample size when --extend.")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    rubrics = load_rubrics(args.rubrics, args.stratify_field)
    scenarios = load_scenarios(args.scenarios)
    qwen = load_qwen_matrix(args.qwen_matrix)

    crit_by_scenario: dict[str, list[str]] = defaultdict(list)
    for cid, r in rubrics.items():
        crit_by_scenario[r["scenario_id"]].append(cid)

    # Pass 1: enumerate gradable cells as (model, scenario, criterion) with capability.
    model_scen = index_gradable_scenarios(args.responses_dir)
    population: list[tuple[str, str, str, str]] = []  # (model, scenario, criterion, capability)
    for model, scen_set in model_scen.items():
        for sid in scen_set:
            for cid in crit_by_scenario.get(sid, ()):
                population.append((model, sid, cid, rubrics[cid]["stratum"]))

    if not population:
        raise SystemExit("no gradable cells found; check responses-dir / rubrics")
    total = len(population)

    # Extend mode: keep the existing sample + its adjudications; draw only NEW distinct cells.
    existing_records: list[dict[str, Any]] = []
    start_index = 0
    if args.extend:
        sample_path_existing = args.out_dir / "sample.jsonl"
        if not sample_path_existing.is_file():
            raise SystemExit(f"--extend requires an existing {sample_path_existing}")
        existing_records = [json.loads(l) for l in
                            sample_path_existing.read_text(encoding="utf-8").splitlines() if l.strip()]
        start_index = len(existing_records)
        if args.target_n is None or args.target_n <= start_index:
            raise SystemExit(f"--target-n must exceed existing sample size {start_index}")
        exclude = {(r["model"], r["scenario_id"], r["criterion_id"]) for r in existing_records}
        population = [c for c in population if (c[0], c[1], c[2]) not in exclude]
        n = min(args.target_n - start_index, len(population))
    else:
        n = min(args.n, total)

    # Stratified proportional allocation by stratum (largest-remainder), seeded.
    rng = random.Random(args.seed + start_index)
    by_cap: dict[str, list[tuple[str, str, str, str]]] = defaultdict(list)
    for cell in population:
        by_cap[cell[3]].append(cell)
    pool_total = len(population)
    # largest-remainder allocation so strata sum to exactly n
    raw = {cap: len(cells) * n / pool_total for cap, cells in by_cap.items()}
    alloc = {cap: int(v) for cap, v in raw.items()}
    remainder = n - sum(alloc.values())
    for cap, _ in sorted(raw.items(), key=lambda kv: kv[1] - int(kv[1]), reverse=True)[:remainder]:
        alloc[cap] += 1

    sampled: list[tuple[str, str, str, str]] = []
    for cap, cells in by_cap.items():
        k = min(alloc.get(cap, 0), len(cells))
        sampled.extend(rng.sample(cells, k))
    rng.shuffle(sampled)

    # Pass 2: materialize response text for sampled cells only.
    needed = {(m, s) for (m, s, _c, _cap) in sampled}
    resp = materialize_responses(args.responses_dir, needed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sample_path = args.out_dir / "sample.jsonl"
    with sample_path.open("w", encoding="utf-8") as out:
        for rec in existing_records:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        for j, (model, sid, cid, cap) in enumerate(sampled):
            i = start_index + j
            r = rubrics[cid]
            scen = scenarios.get(sid, {})
            rr = resp.get((model, sid), {})
            rec = {
                "gold_case_id": f"{args.benchmark}__{i:04d}",
                "benchmark": args.benchmark,
                "model": model,
                "scenario_id": sid,
                "criterion_id": cid,
                "stratum": cap,
                "stratum_field": args.stratify_field,
                "criticality": r["criticality"],
                "task": r["task"],
                "criterion": r["criterion"],
                "expected_evidence": r["expected_evidence"],
                "scenario_prompt": scen.get("prompt", ""),
                "conversation_context": scen.get("conversation_context") or [],
                "reference_solution": scen.get("reference_solution") or "",
                "candidate_response": rr.get("output", ""),
                "response_truncated": rr.get("truncated", False),
                # reference only (NOT shown to proposers or during blinded adjudication):
                "_qwen_verdict": qwen.get((model, cid)),
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    total_size = len(existing_records) + len(sampled)
    manifest = {
        "benchmark": args.benchmark,
        "seed": args.seed,
        "extended": bool(args.extend),
        "existing_kept": len(existing_records),
        "new_drawn": len(sampled),
        "population_gradable_cells": total,
        "distinct_models": len(model_scen),
        "sample_size": total_size,
        "stratify_by": args.stratify_field,
        "stratum_population": dict(Counter(c[3] for c in population)),
        "stratum_allocation": alloc,
        "stratum_sampled_new": dict(Counter(c[3] for c in sampled)),
        "sampling": "stratified proportional (largest-remainder), simple random within stratum",
        "inputs": {
            "responses_dir": str(args.responses_dir),
            "rubrics": str(args.rubrics),
            "scenarios": str(args.scenarios),
            "qwen_matrix": str(args.qwen_matrix) if args.qwen_matrix else None,
        },
    }
    (args.out_dir / "sample_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: manifest[k] for k in (
        "population_gradable_cells", "distinct_models", "sample_size",
        "existing_kept", "new_drawn", "stratum_sampled_new")}, indent=2))
    print(f"wrote {sample_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
