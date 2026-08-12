"""Local join -> judge -> DAMR for MRBench Phase B (generate-on-cluster split).

The recommended Phase-B shape (see ``Plan/mrbench/PHASE_B_EDULLM_PLAN.md``) is to
*generate on the cluster* with ``MRBENCH_GENERATE_ONLY=1`` (no judge key, no
network) and *judge locally*. This script is the local half: it takes the
predictions JSONL that the generate-only run wrote, joins each ``final_output``
back to ``MRBench_V1.json`` by ``conversation_id`` (= the prediction's
``native_id``), runs the already-validated Figure-6 judge by REUSING the existing
modules (``judge_prompt`` / ``judge_client`` / ``parse`` / ``judge_cache``), and
computes **DAMR only**.

There is deliberately no AC and no macro-F1 here: those are judge-vs-human-gold
metrics and belong to Phase A. A model-under-test's fresh responses have no human
gold, so DAMR (the desired-annotation match rate of the judge's own labels) is
the Phase-B deliverable.

Safe by default: with no flags it performs a **dry run** — it reports the join,
prints a few sample judge prompts, and prints a cost estimate, making ZERO
network calls. A live run needs ``--live`` *and* a judge API key *and* a real
model id in the environment (the same ``OPENAI_API_KEY`` / ``MRBENCH_JUDGE_MODEL``
/ base-URL knobs the Phase-A harness reads; nothing is hardcoded). Judge results
are cached idempotently, so re-runs skip completed
``(conversation_id, model, dimension)`` calls.

    # dry run: join + sample prompts + cost estimate (no network)
    uv run python -m diagnostics.mrbench.phase_b_judge --predictions <preds.jsonl>

    # compute DAMR from an existing cache (no network)
    uv run python -m diagnostics.mrbench.phase_b_judge --predictions <preds.jsonl> --metrics

    # live judge run (gated; needs key + model id in env)
    uv run python -m diagnostics.mrbench.phase_b_judge --predictions <preds.jsonl> --live
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .cost import (
    CHARS_PER_TOKEN,
    DEFAULT_OUTPUT_TOKENS_PER_CALL,
    RATE_INPUT_PER_MTOK,
    RATE_OUTPUT_PER_MTOK,
)
from .gold_damr import load_conversations
from .judge_cache import JudgeCache, JudgeRecord
from .judge_client import JudgeConfig, make_client, score_messages
from .judge_prompt import build_messages, reference_guided_default
from .metrics import (
    bootstrap_ci_rate,
    collect_dimension_desired_indicators,
    compute_judge_damr,
)
from .parse import aggregate_samples, parse_result
from .reference import DESIRED_LABELS, DIMENSIONS

DEFAULT_MODEL_LABEL = "model_under_test"
DEFAULT_CACHE = Path(__file__).parent / "runs" / "phase_b_judge_cache.jsonl"


# --------------------------------------------------------------------------- #
# Join: predictions JSONL -> conversation context
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class JoinedItem:
    """One generation joined to its MRBench conversation context."""

    conversation_id: str
    answer: str
    history: str
    solution: str


def _prediction_answer(record: dict) -> str:
    """The tutor text a prediction record carries (final_output, else output[0])."""
    final_output = record.get("final_output")
    if isinstance(final_output, str) and final_output:
        return final_output
    outputs = record.get("model_output") or []
    if outputs and isinstance(outputs[0], dict):
        text = outputs[0].get("text")
        if isinstance(text, str):
            return text
    return ""


def load_predictions(path: str | Path) -> list[dict]:
    """Read the predictions JSONL produced by a generate-only run."""
    pred_path = Path(path)
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions artifact not found: {pred_path}")
    records: list[dict] = []
    with pred_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_conversation_index(conversations: list[dict]) -> dict[str, dict]:
    """Map ``conversation_id`` -> conversation for the local join."""
    return {str(conv.get("conversation_id")): conv for conv in conversations}


@dataclass
class JoinResult:
    items: list[JoinedItem]
    matched: int
    unmatched_ids: list[str]


def join_predictions(
    predictions: list[dict],
    index: dict[str, dict],
    limit: int | None = None,
) -> JoinResult:
    """Join predictions to conversation context by ``native_id`` == ``conversation_id``."""
    items: list[JoinedItem] = []
    unmatched: list[str] = []
    for record in predictions:
        native_id = record.get("native_id")
        cid = str(native_id) if native_id is not None else ""
        conv = index.get(cid)
        if conv is None:
            unmatched.append(cid)
            continue
        items.append(
            JoinedItem(
                conversation_id=cid,
                answer=_prediction_answer(record),
                history=conv.get("conversation_history", ""),
                solution=conv.get("Ground_Truth_Solution", ""),
            )
        )
        if limit is not None and len(items) >= limit:
            break
    return JoinResult(items=items, matched=len(items), unmatched_ids=unmatched)


# --------------------------------------------------------------------------- #
# Dry run + cost (no network)
# --------------------------------------------------------------------------- #
def print_dry_run(join: JoinResult, samples: int, reference_guided: bool) -> None:
    print("=" * 90)
    label = "reference-guided (OPT-IN)" if reference_guided else "byte-faithful Figure-6"
    print(f"DRY RUN — join predictions -> {label} judge prompt (NO network calls)")
    print("=" * 90)
    print(f"  joined generations               : {join.matched}")
    print(f"  unmatched predictions            : {len(join.unmatched_ids)}")
    if join.unmatched_ids:
        preview = ", ".join(join.unmatched_ids[:5])
        print(f"    (first unmatched native_ids)   : {preview}")
    pairs = itertools.islice(((item, dim) for item in join.items for dim in DIMENSIONS), samples)
    for i, (item, dim) in enumerate(pairs, start=1):
        messages = build_messages(
            item.history,
            item.answer,
            dim,
            reference_solution=item.solution,
            reference_guided=reference_guided,
        )
        print(f"\n----- sample {i}: conversation={item.conversation_id} dimension={dim} -----")
        for msg in messages:
            print(f"\n[{msg['role'].upper()} MESSAGE]")
            print(msg["content"])
    print("\n" + "=" * 90)


def print_cost(
    join: JoinResult,
    output_tokens: int,
    reference_guided: bool,
    judge_samples: int,
) -> None:
    """DAMR-only judge cost, sized from the real joined prompts. Reuses cost.py rates."""
    judge_samples = max(1, judge_samples)
    total_input_chars = 0
    for item in join.items:
        for dim in DIMENSIONS:
            messages = build_messages(
                item.history,
                item.answer,
                dim,
                reference_solution=item.solution,
                reference_guided=reference_guided,
            )
            total_input_chars += sum(len(m["content"]) for m in messages)
    total_input_chars *= judge_samples
    n_calls = join.matched * len(DIMENSIONS) * judge_samples
    input_tokens = round(total_input_chars / CHARS_PER_TOKEN)
    output_total = n_calls * output_tokens
    input_cost = input_tokens / 1_000_000 * RATE_INPUT_PER_MTOK
    output_cost = output_total / 1_000_000 * RATE_OUTPUT_PER_MTOK

    print("\nCOST ESTIMATE — local judge over the joined generations (DAMR only)")
    print("=" * 90)
    print(f"  self-consistency k (samples)     : {judge_samples}")
    print(f"  judge calls (joined x 8 x k)     : {n_calls:,}")
    print(f"  measured input chars (total)     : {total_input_chars:,}")
    print(f"  chars/token heuristic            : {CHARS_PER_TOKEN}")
    print(f"  est. input tokens                : {input_tokens:,}")
    print(f"  assumed output tokens / call     : {output_tokens}")
    print(
        f"  assumed rates                    : "
        f"${RATE_INPUT_PER_MTOK}/MTok in, ${RATE_OUTPUT_PER_MTOK}/MTok out"
    )
    print(f"  input cost                       : ${input_cost:,.2f}")
    print(f"  output cost                      : ${output_cost:,.2f}")
    print(f"  TOTAL (point estimate)           : ${input_cost + output_cost:,.2f}")
    print("=" * 90)


# --------------------------------------------------------------------------- #
# Live run (gated) — reuses the exact Phase-A judge path
# --------------------------------------------------------------------------- #
def run_live(
    join: JoinResult,
    config: JudgeConfig,
    model_label: str,
    cache_path: Path,
    concurrency: int,
    reference_guided: bool,
    k: int,
) -> int:
    missing = config.missing_for_live()
    if missing:
        print("LIVE RUN REFUSED — missing configuration:")
        for item in missing:
            print(f"  - {item}")
        print(f"\n  base_url = {config.base_url}")
        print("  (set the env vars and pass --live again)")
        return 2

    cache = JudgeCache(cache_path)
    tasks = [(item, dim) for item in join.items for dim in DIMENSIONS]
    todo = [t for t in tasks if not cache.is_done((t[0].conversation_id, model_label, t[1]))]
    k = max(1, k)
    print(f"Live judge: {len(tasks)} calls, {len(tasks) - len(todo)} cached, {len(todo)} to do.")
    print(f"  gateway={config.base_url}  model={config.model}  concurrency={concurrency}")
    print(f"  model_under_test={model_label}  reference_guided={reference_guided}  k={k}")

    client = make_client(config)

    def _do(task: tuple[JoinedItem, str]) -> JudgeRecord:
        item, dim = task
        messages = build_messages(
            item.history,
            item.answer,
            dim,
            reference_solution=item.solution,
            reference_guided=reference_guided,
        )
        if k == 1:
            parsed = parse_result(score_messages(client, config, messages), dim)
            samples = None
        else:
            raws = [score_messages(client, config, messages) for _ in range(k)]
            parsed, per_sample = aggregate_samples(raws, dim)
            samples = [s.raw for s in per_sample]
        return JudgeRecord(
            conversation_id=item.conversation_id,
            tutor=model_label,
            dimension=dim,
            score=parsed.score,
            label=parsed.label,
            feedback=parsed.feedback,
            ok=parsed.ok,
            used_fallback=parsed.used_fallback,
            raw=parsed.raw,
            model=config.model,
            samples=samples,
        )

    done = 0
    failures = 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_do, t): t for t in todo}
        for fut in as_completed(futures):
            try:
                rec = fut.result()
                cache.put(rec)
                done += 1
                if not rec.ok:
                    failures += 1
            except Exception as exc:  # noqa: BLE001 - surface, keep going
                failures += 1
                print(f"  ! call failed: {exc}")
    print(f"Live judge complete: {done} new results, {failures} failures. Cache: {cache_path}")
    return 0


# --------------------------------------------------------------------------- #
# DAMR (only) from the cache
# --------------------------------------------------------------------------- #
def _pct(v: float) -> str:
    return " nan" if math.isnan(v) else f"{v:6.2f}"


def compute_and_print_damr(
    cache_path: Path,
    model_label: str,
    bootstrap: int = 0,
    out_path: Path | None = None,
) -> int:
    if not cache_path.exists():
        print(f"No judge cache at {cache_path}; run --live first.")
        return 1
    cache = JudgeCache(cache_path)
    records = [r for r in cache.records() if r.tutor == model_label]
    if not records:
        print(f"No cached judge records for model {model_label!r} in {cache_path}.")
        return 1

    damr = compute_judge_damr(records)
    cells = damr.cells.get(model_label, {})
    per_dim = {dim: cells.get(dim, math.nan) for dim in DIMENSIONS}
    present = [v for v in per_dim.values() if not math.isnan(v)]
    aggregate = sum(present) / len(present) if present else math.nan

    indicators = collect_dimension_desired_indicators(records)
    ci_by_dim: dict[str, tuple[float, float]] = {}
    if bootstrap > 0:
        ci_by_dim = {
            dim: bootstrap_ci_rate(indicators[dim], n_boot=bootstrap) for dim in DIMENSIONS
        }

    print("\nDAMR (judge-derived) — MRBench Phase B, DAMR only (no AC / macro-F1)")
    print(f"model under test: {model_label}   responses judged: {damr.totals.get(model_label, 0)}")
    print("=" * 90)
    header = "dimension".ljust(28) + "DAMR%".rjust(9)
    if bootstrap > 0:
        header += "DAMR% 95% CI".rjust(22)
    print(header)
    for dim in DIMENSIONS:
        line = dim.ljust(28) + _pct(per_dim[dim]).rjust(9)
        if bootstrap > 0:
            lo, hi = ci_by_dim[dim]
            line += f"[{lo:.1f}, {hi:.1f}]".rjust(22)
        print(line)
    print("-" * 90)
    print("aggregate damr".ljust(28) + _pct(aggregate).rjust(9))
    print("=" * 90)

    result: dict[str, Any] = {
        "model": model_label,
        "responses_judged": damr.totals.get(model_label, 0),
        "aggregate_damr": aggregate,
        "per_dimension_damr": per_dim,
        "desired_labels": {dim: DESIRED_LABELS[dim] for dim in DIMENSIONS},
    }
    if bootstrap > 0:
        result["per_dimension_damr_ci95"] = {dim: list(ci_by_dim[dim]) for dim in DIMENSIONS}
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Wrote DAMR result to {out_path}")
    return 0


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=("MRBench Phase-B local join -> judge -> DAMR. Safe by default (dry run)."),
    )
    parser.add_argument(
        "--predictions", required=True, help="Predictions JSONL from a generate-only run."
    )
    parser.add_argument("--data-path", default=None, help="Path to MRBench_V1.json.")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL_LABEL, help="Label for the model under test (cache key)."
    )
    parser.add_argument("--cache", default=str(DEFAULT_CACHE), help="JSONL judge cache path.")
    parser.add_argument("--concurrency", type=int, default=8, help="Live-run worker threads.")
    parser.add_argument("--limit", type=int, default=None, help="Cap joined generations judged.")
    parser.add_argument("--samples", type=int, default=2, help="Dry-run prompts to print.")
    parser.add_argument("--output-tokens", type=int, default=DEFAULT_OUTPUT_TOKENS_PER_CALL)
    parser.add_argument("--live", action="store_true", help="Perform the gated live judge run.")
    parser.add_argument(
        "--metrics", action="store_true", help="Compute DAMR from the cache (no network)."
    )
    parser.add_argument(
        "--reference-guided",
        action="store_true",
        help="OPT-IN: inject Ground_Truth_Solution for correctness-linked dims (off-protocol).",
    )
    parser.add_argument(
        "--k", type=int, default=1, help="OPT-IN self-consistency: judge samples/call (default 1)."
    )
    parser.add_argument(
        "--bootstrap", type=int, default=0, help="Bootstrap resamples for DAMR CIs (0=off)."
    )
    parser.add_argument("--out", default=None, help="Optional path to write the DAMR result JSON.")
    args = parser.parse_args(argv)

    conversations = load_conversations(args.data_path)
    index = build_conversation_index(conversations)
    predictions = load_predictions(args.predictions)
    join = join_predictions(predictions, index, limit=args.limit)
    cache_path = Path(args.cache)
    out_path = Path(args.out) if args.out else None
    reference_guided = args.reference_guided or reference_guided_default()

    if join.matched == 0:
        print("No predictions joined to any conversation (native_id != conversation_id?).")
        print(f"  predictions read: {len(predictions)}  unmatched: {len(join.unmatched_ids)}")
        return 1

    if args.metrics and not args.live:
        return compute_and_print_damr(
            cache_path, args.model, bootstrap=args.bootstrap, out_path=out_path
        )

    if args.live:
        rc = run_live(
            join,
            JudgeConfig.from_env(),
            args.model,
            cache_path,
            args.concurrency,
            reference_guided=reference_guided,
            k=args.k,
        )
        if rc == 0:
            rc = compute_and_print_damr(
                cache_path, args.model, bootstrap=args.bootstrap, out_path=out_path
            )
        return rc

    # Default: dry run (join + sample prompts + cost estimate), zero network.
    print_dry_run(join, args.samples, reference_guided=reference_guided)
    print_cost(join, args.output_tokens, reference_guided=reference_guided, judge_samples=args.k)
    cfg = JudgeConfig.from_env()
    print("\nLive-run readiness")
    missing = cfg.missing_for_live()
    if missing:
        print("  NOT ready — still need:")
        for item in missing:
            print(f"    - {item}")
    else:
        print("  Ready: pass --live to execute.")
    print(f"  base_url = {cfg.base_url}")
    print(f"  model    = {cfg.model}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
