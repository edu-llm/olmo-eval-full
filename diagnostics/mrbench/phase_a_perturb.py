"""CLI: P2 judge-robustness under benign, meaning-preserving perturbations.

Re-judges a stratified subsample of the Phase-A tutor responses under each benign
variant (:mod:`perturbations`) and measures how much the judge's decisions and the
Phase-A conclusions move (:mod:`robustness`). Faithful to the harness that produced
the validated Phase-A run: it reuses the cached canonical judgments, the Figure-6
prompt, the parser and the AC/DAMR maths — nothing here is re-derived.

Safe by default. With no ``--live`` flag it performs a **dry run + cost estimate**
only, making ZERO network calls:

    # dry run: subsample plan + per-variant judge-call count + Haiku cost estimate
    uv run python -m diagnostics.mrbench.phase_a_perturb

    # recompute robustness metrics from already-written per-variant caches
    uv run python -m diagnostics.mrbench.phase_a_perturb --metrics

    # LIVE (gated): judge the subsample under each variant, then report robustness
    uv run python -m diagnostics.mrbench.phase_a_perturb --live

Each variant writes to its own cache file (``<run-dir>/perturb/<variant>.jsonl``)
so a perturbation run never collides with the canonical Phase-A cache. A live run
additionally requires a TrueFoundry key and model id in the environment (the same
gate as ``judge_run.py``).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from .cost import (
    CHARS_PER_TOKEN,
    CHARS_PER_TOKEN_HIGH,
    CHARS_PER_TOKEN_LOW,
    DEFAULT_OUTPUT_TOKENS_PER_CALL,
)
from .gold_damr import load_conversations
from .judge_cache import CacheKey, JudgeCache, JudgeRecord
from .judge_client import JudgeConfig, make_client, score_messages
from .judge_prompt import build_messages
from .metrics import build_gold_index
from .parse import parse_result
from .perturbations import VARIANTS, apply_variant, assert_meaning_preserved
from .reference import DIMENSIONS, PAPER_TO_DATA_KEY
from .robustness import RobustnessReport, compute_robustness

DEFAULT_CACHE = Path(__file__).parent / "runs" / "judge_cache.jsonl"
DEFAULT_RUN_DIR = Path(__file__).parent / "runs" / "phase_a_haiku_4_5_2026-08-09"
DEFAULT_SUBSAMPLE = 1200

# Claude Haiku 4.5 pricing (USD per million tokens), per PHASE_A_VALIDATION.md.
HAIKU_RATE_INPUT_PER_MTOK = 1.0
HAIKU_RATE_OUTPUT_PER_MTOK = 5.0


# --------------------------------------------------------------------------- #
# Response lookup + subsampling
# --------------------------------------------------------------------------- #
def build_response_lookup(conversations: list[dict]) -> dict[tuple[str, str], tuple[str, str, str]]:
    """Map ``(conversation_id, paper_tutor) -> (history, response, solution)``."""
    data_key_to_paper = {data: paper for paper, data in PAPER_TO_DATA_KEY.items()}
    lookup: dict[tuple[str, str], tuple[str, str, str]] = {}
    for conv in conversations:
        cid = str(conv.get("conversation_id"))
        history = conv.get("conversation_history", "")
        solution = conv.get("Ground_Truth_Solution", "")
        for data_key, info in conv.get("anno_llm_responses", {}).items():
            paper = data_key_to_paper.get(data_key)
            if paper is None:
                continue
            lookup[(cid, paper)] = (history, info.get("response", ""), solution)
    return lookup


def build_source_map(conversations: list[dict]) -> dict[str, str]:
    """Map ``conversation_id -> source`` (e.g. ``MathDial`` / ``Bridge``)."""
    return {str(c.get("conversation_id")): c.get("Data", "Unknown") for c in conversations}


def eligible_keys(
    records: list[JudgeRecord], gold_index: dict, response_lookup: dict
) -> list[CacheKey]:
    """Canonical cells that are ok, paired to gold, and have a reconstructable prompt."""
    keys: list[CacheKey] = []
    for rec in records:
        if not (rec.ok and rec.score is not None):
            continue
        key = rec.key
        if key not in gold_index:
            continue
        if (rec.conversation_id, rec.tutor) not in response_lookup:
            continue
        keys.append(key)
    return keys


def stratified_subsample(
    keys: list[CacheKey], source_map: dict[str, str], n_target: int, seed: int
) -> list[CacheKey]:
    """Proportional stratified subsample of cell keys.

    Strata are ``(dimension, tutor, source)``. Within each stratum the keys are
    deterministically shuffled (fixed seed) and a proportional share is taken so
    the subsample keeps the canonical dimension x tutor x source mix. Returns all
    keys when ``n_target`` meets or exceeds the universe size.
    """
    if n_target >= len(keys) or n_target <= 0:
        return sorted(keys)

    strata: dict[tuple[str, str, str], list[CacheKey]] = defaultdict(list)
    for key in keys:
        cid, tutor, dim = key
        strata[(dim, tutor, source_map.get(cid, "Unknown"))].append(key)

    fraction = n_target / len(keys)
    rng = random.Random(seed)
    selected: list[CacheKey] = []
    for stratum in sorted(strata):
        bucket = sorted(strata[stratum])
        rng.shuffle(bucket)
        take = max(1, round(fraction * len(bucket)))
        selected.extend(bucket[:take])
    return sorted(selected)


# --------------------------------------------------------------------------- #
# Plan + cost (dry run, no network)
# --------------------------------------------------------------------------- #
def messages_for(
    key: CacheKey, variant: str, response_lookup: dict[tuple[str, str], tuple[str, str, str]]
) -> list[dict[str, str]]:
    """Canonical Figure-6 messages for a cell, transformed by ``variant``."""
    cid, tutor, dim = key
    history, response, _solution = response_lookup[(cid, tutor)]
    canonical_messages = build_messages(history, response, dim)
    return apply_variant(variant, canonical_messages)


def print_plan_and_cost(
    subsample: list[CacheKey],
    variants: list[str],
    response_lookup: dict,
    output_tokens: int,
) -> None:
    print("=" * 90)
    print("P2 PERTURBATION-ROBUSTNESS PLAN (dry run — NO network calls)")
    print("=" * 90)
    print(f"  subsample size (cells)           : {len(subsample):,}")
    print(f"  variants ({len(variants)})                     : {', '.join(variants)}")
    total_calls = len(subsample) * len(variants)
    print(f"  judge calls (subsample x variants): {total_calls:,}")

    total_input_chars = 0
    per_variant_chars: dict[str, int] = {}
    for variant in variants:
        vchars = 0
        for key in subsample:
            msgs = messages_for(key, variant, response_lookup)
            vchars += sum(len(m["content"]) for m in msgs)
        per_variant_chars[variant] = vchars
        total_input_chars += vchars

    input_tokens = round(total_input_chars / CHARS_PER_TOKEN)
    total_output_tokens = total_calls * output_tokens
    input_cost = input_tokens / 1_000_000 * HAIKU_RATE_INPUT_PER_MTOK
    output_cost = total_output_tokens / 1_000_000 * HAIKU_RATE_OUTPUT_PER_MTOK

    def _dollar(in_tok: int, out_tok: int) -> float:
        return (
            in_tok / 1_000_000 * HAIKU_RATE_INPUT_PER_MTOK
            + out_tok / 1_000_000 * HAIKU_RATE_OUTPUT_PER_MTOK
        )

    low = _dollar(round(total_input_chars / CHARS_PER_TOKEN_LOW), total_output_tokens)
    high = _dollar(round(total_input_chars / CHARS_PER_TOKEN_HIGH), total_output_tokens)

    print(f"  measured input chars (total)     : {total_input_chars:,}")
    print(f"  chars/token heuristic            : {CHARS_PER_TOKEN} (band 3.5-4.5)")
    print(f"  est. input tokens                : {input_tokens:,}")
    print(f"  assumed output tokens / call     : {output_tokens}")
    print(f"  est. output tokens (total)       : {total_output_tokens:,}")
    print(
        f"  assumed rates                    : "
        f"${HAIKU_RATE_INPUT_PER_MTOK}/MTok in, "
        f"${HAIKU_RATE_OUTPUT_PER_MTOK}/MTok out (Claude Haiku 4.5)"
    )
    print(f"  input cost                       : ${input_cost:,.2f}")
    print(f"  output cost                      : ${output_cost:,.2f}")
    print(f"  TOTAL (point estimate)           : ${input_cost + output_cost:,.2f}")
    print(f"  TOTAL (range, token band)        : ${low:,.2f} - ${high:,.2f}")
    print("=" * 90)


# --------------------------------------------------------------------------- #
# Live run (gated) — one cache file per variant
# --------------------------------------------------------------------------- #
def variant_cache_path(run_dir: Path, variant: str) -> Path:
    return run_dir / "perturb" / f"{variant}.jsonl"


def run_live(
    subsample: list[CacheKey],
    variants: list[str],
    response_lookup: dict,
    config: JudgeConfig,
    run_dir: Path,
    concurrency: int,
) -> int:
    missing = config.missing_for_live()
    if missing:
        print("LIVE RUN REFUSED — missing configuration:")
        for item in missing:
            print(f"  - {item}")
        print(f"\n  base_url = {config.base_url}")
        print("  (set the env vars and pass --live again)")
        return 2

    client = make_client(config)
    for variant in variants:
        cache = JudgeCache(variant_cache_path(run_dir, variant))
        todo = [k for k in subsample if not cache.is_done(k)]
        print(
            f"\n[{variant}] {len(subsample)} cells, "
            f"{len(subsample) - len(todo)} cached, {len(todo)} to do."
        )

        def _do(key: CacheKey, variant: str = variant) -> JudgeRecord:
            cid, tutor, dim = key
            messages = messages_for(key, variant, response_lookup)
            # Never spend on a prompt that silently dropped meaning-bearing content.
            _history, response, _sol = response_lookup[(cid, tutor)]
            assert_meaning_preserved(messages, response=response, dimension=dim)
            parsed = parse_result(score_messages(client, config, messages), dim)
            return JudgeRecord(
                conversation_id=cid,
                tutor=tutor,
                dimension=dim,
                score=parsed.score,
                label=parsed.label,
                feedback=parsed.feedback,
                ok=parsed.ok,
                used_fallback=parsed.used_fallback,
                raw=parsed.raw,
                model=config.model,
                samples=None,
            )

        done = failures = 0
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(_do, k): k for k in todo}
            for fut in as_completed(futures):
                try:
                    rec = fut.result()
                    cache.put(rec)
                    done += 1
                    if not rec.ok:
                        failures += 1
                except Exception as exc:  # noqa: BLE001 - surface, keep going
                    failures += 1
                    print(f"  ! task failed: {exc}")
                if done % 100 == 0 and done:
                    print(f"  ... {done}/{len(todo)} completed")
        print(f"[{variant}] complete: {done} new, {failures} failures.")
    return 0


# --------------------------------------------------------------------------- #
# Metrics (from per-variant caches vs canonical)
# --------------------------------------------------------------------------- #
def load_variant_records(variants: list[str], run_dir: Path) -> dict[str, list[JudgeRecord]]:
    out: dict[str, list[JudgeRecord]] = {}
    for variant in variants:
        path = variant_cache_path(run_dir, variant)
        if path.exists():
            out[variant] = JudgeCache(path).records()
    return out


def _fmt(v: float) -> str:
    return " nan" if math.isnan(v) else f"{v:+.4f}"


def _fmt_rate(v: float) -> str:
    return " nan" if math.isnan(v) else f"{v:.4f}"


def print_report(report: RobustnessReport) -> None:
    print("\n" + "=" * 100)
    print("P2 ROBUSTNESS — benign perturbation vs canonical Phase-A judge")
    print("=" * 100)
    for variant, vr in report.variants.items():
        print(f"\n### variant: {variant}")
        print(
            f"  {'dimension':<26}{'desiredFlip':>12}{'rawFlip':>10}"
            f"{'AC_canon':>10}{'AC_var':>10}{'dAC':>9}{'dDAMR':>9}{'gate':>10}"
        )
        for dim in DIMENSIONS:
            dr = vr.per_dimension[dim]
            gate = (
                "excluded"
                if not dr.eligible_for_gate
                else ("CHANGED" if dr.gate_verdict_changed else "same")
            )
            print(
                f"  {dim:<26}{_fmt_rate(dr.desired_flip_rate):>12}"
                f"{_fmt_rate(dr.raw_flip_rate):>10}"
                f"{_fmt(dr.ac_canonical):>10}{_fmt(dr.ac_variant):>10}"
                f"{_fmt(dr.delta_ac):>9}{_fmt(dr.delta_damr):>9}{gate:>10}"
            )
        print(
            f"  {'-- overall':<26}{_fmt_rate(vr.overall_desired_flip_rate):>12}"
            f"{_fmt_rate(vr.overall_raw_flip_rate):>10}"
            f"{'':>10}{'':>10}{_fmt(vr.max_abs_delta_ac):>9}"
            f"{_fmt(vr.max_abs_delta_damr):>9}"
            f"{('CHANGED' if vr.any_gate_verdict_changed else 'same'):>10}"
        )

    print("\n" + "-" * 100)
    print("AGGREGATE GATES (from PHASE_A_FP_FN.md)")
    g = report.gates
    print(
        f"  worst desired-decision flip rate : {_fmt_rate(report.worst_desired_flip_rate)}  "
        f"(gate <= {g['desired_flip_max']})"
    )
    print(
        f"  worst raw 3-class flip rate      : {_fmt_rate(report.worst_raw_flip_rate)}  "
        f"(diagnostic <= {g['raw_flip_max']})"
    )
    print(
        f"  max |dAC| per dimension          : {_fmt_rate(report.max_abs_delta_ac)}  "
        f"(gate <= {g['delta_ac_max']})"
    )
    print(
        f"  max |dDAMR| per dimension        : {_fmt_rate(report.max_abs_delta_damr)}  "
        f"(gate <= {g['delta_damr_max']})"
    )
    print(f"  any AC-gate verdict changed      : {report.any_gate_verdict_changed}")
    print(f"\n  ROBUST (verdict-stable & flip<=0.10 & |dAC|<=0.05): {report.robust}")
    print("=" * 100)


def report_to_json(report: RobustnessReport, subsample_size: int, seed: int) -> dict:
    return {
        "description": (
            "P2 perturbation-robustness of the Phase-A Claude Haiku 4.5 judge under "
            "benign, meaning-preserving prompt variants, vs the canonical judgments. "
            "Flip rates and DAMR are fractions in [0, 1]; dAC is on the Pearson scale."
        ),
        "subsample_size": subsample_size,
        "seed": seed,
        "gates": report.gates,
        "aggregate": {
            "worst_desired_flip_rate": report.worst_desired_flip_rate,
            "worst_raw_flip_rate": report.worst_raw_flip_rate,
            "max_abs_delta_ac": report.max_abs_delta_ac,
            "max_abs_delta_damr": report.max_abs_delta_damr,
            "any_gate_verdict_changed": report.any_gate_verdict_changed,
            "robust": report.robust,
        },
        "variants": {
            variant: {
                "overall_desired_flip_rate": vr.overall_desired_flip_rate,
                "overall_raw_flip_rate": vr.overall_raw_flip_rate,
                "max_abs_delta_ac": vr.max_abs_delta_ac,
                "max_abs_delta_damr": vr.max_abs_delta_damr,
                "any_gate_verdict_changed": vr.any_gate_verdict_changed,
                "per_dimension": {dim: asdict(dr) for dim, dr in vr.per_dimension.items()},
            }
            for variant, vr in report.variants.items()
        },
    }


def compute_and_report(
    variants: list[str],
    canonical_records: list[JudgeRecord],
    gold_index: dict,
    run_dir: Path,
    subsample_size: int,
    seed: int,
    bootstrap: int,
    out_path: Path | None,
    no_write: bool,
) -> int:
    variant_records = load_variant_records(variants, run_dir)
    if not variant_records:
        print("No per-variant caches found under", run_dir / "perturb")
        print("Run a live perturbation pass first (--live), or check --run-dir.")
        return 1
    report = compute_robustness(
        canonical_records,
        variant_records,
        gold_index,
        n_boot=bootstrap,
        flip_boot=bootstrap,
        seed=seed,
    )
    print_report(report)
    if not no_write:
        target = out_path or (run_dir / "perturb_metrics.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as fh:
            json.dump(report_to_json(report, subsample_size, seed), fh, indent=2)
            fh.write("\n")
        print(f"\nWrote machine-readable metrics to {target}")
    return 0


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="P2 judge robustness under benign perturbations. Safe by default."
    )
    parser.add_argument("--data-path", default=None, help="Path to MRBench_V1.json.")
    parser.add_argument("--cache", default=str(DEFAULT_CACHE), help="Canonical judge cache path.")
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR), help="Run dir for outputs.")
    parser.add_argument(
        "--subsample", type=int, default=DEFAULT_SUBSAMPLE, help="Target subsample cell count."
    )
    parser.add_argument("--seed", type=int, default=42, help="Subsample + bootstrap RNG seed.")
    parser.add_argument(
        "--variants",
        nargs="*",
        default=list(VARIANTS),
        help=f"Benign variants to run (subset of {list(VARIANTS)}).",
    )
    parser.add_argument("--output-tokens", type=int, default=DEFAULT_OUTPUT_TOKENS_PER_CALL)
    parser.add_argument(
        "--live", action="store_true", help="Perform the gated live perturbation run."
    )
    parser.add_argument(
        "--metrics", action="store_true", help="Compute metrics from variant caches."
    )
    parser.add_argument("--concurrency", type=int, default=8, help="Live-run worker threads.")
    parser.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap resamples for CIs.")
    parser.add_argument("--out", default=None, help="Explicit perturb_metrics.json path.")
    parser.add_argument(
        "--no-write", action="store_true", help="Do not write perturb_metrics.json."
    )
    args = parser.parse_args(argv)

    unknown = [v for v in args.variants if v not in VARIANTS]
    if unknown:
        print(f"Unknown variant(s): {unknown}; known variants are {list(VARIANTS)}")
        return 2

    cache_path = Path(args.cache)
    if not cache_path.exists():
        print(f"No canonical judge cache at {cache_path}.")
        print("STOP: cannot find cached Phase-A judgments; refusing to trigger new judging.")
        return 1

    run_dir = Path(args.run_dir)
    conversations = load_conversations(args.data_path)
    canonical_records = JudgeCache(cache_path).records()
    gold_index = build_gold_index(conversations)
    response_lookup = build_response_lookup(conversations)
    source_map = build_source_map(conversations)

    universe = eligible_keys(canonical_records, gold_index, response_lookup)
    subsample = stratified_subsample(universe, source_map, args.subsample, args.seed)
    print(
        f"Loaded {len(canonical_records)} canonical records; "
        f"{len(universe)} eligible cells; subsample = {len(subsample)} "
        f"(target {args.subsample}, seed {args.seed})."
    )

    out_path = Path(args.out) if args.out else None

    if args.metrics and not args.live:
        return compute_and_report(
            args.variants,
            canonical_records,
            gold_index,
            run_dir,
            len(subsample),
            args.seed,
            args.bootstrap,
            out_path,
            args.no_write,
        )

    if args.live:
        rc = run_live(
            subsample,
            args.variants,
            response_lookup,
            JudgeConfig.from_env(),
            run_dir,
            args.concurrency,
        )
        if rc != 0:
            return rc
        return compute_and_report(
            args.variants,
            canonical_records,
            gold_index,
            run_dir,
            len(subsample),
            args.seed,
            args.bootstrap,
            out_path,
            args.no_write,
        )

    # Default: dry-run plan + Haiku cost estimate, zero network.
    print_plan_and_cost(subsample, args.variants, response_lookup, args.output_tokens)
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
    print("\n(No API calls were made. Re-run with --live once teammates approve the cost.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
