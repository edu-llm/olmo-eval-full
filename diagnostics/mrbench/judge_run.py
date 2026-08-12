"""CLI for the Milestone-1 Claude Haiku 4.5 judge (TrueFoundry gateway).

Safe by default: with no flags it performs a **dry run** — it builds and prints
the exact Figure-6 prompt for a few responses and prints the full-run cost
estimate, making ZERO network calls. A live run requires ``--live`` *and* a
TrueFoundry API key in the environment *and* a real model id.

    # dry run: print sample prompts + cost estimate (no network)
    uv run python -m diagnostics.mrbench.judge_run

    # offline self-test of parser + metrics on fabricated judge outputs
    uv run python -m diagnostics.mrbench.judge_run --validate

    # compute AC + judge-DAMR from an existing cache (no network)
    uv run python -m diagnostics.mrbench.judge_run --metrics

    # live run (gated; needs TFY token + model id in env)
    uv run python -m diagnostics.mrbench.judge_run --live --limit 8
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .cost import estimate_cost, estimate_phase_b_cost
from .gold_damr import load_conversations
from .judge_cache import JudgeCache, JudgeRecord
from .judge_client import JudgeConfig, make_client, score_messages
from .judge_prompt import build_messages
from .metrics import (
    bootstrap_ci_pearson,
    bootstrap_ci_rate,
    build_gold_index,
    collect_dimension_desired_indicators,
    collect_dimension_pairs,
    compare_ac_to_baselines,
    compute_ac,
    compute_judge_damr,
    compute_macro_f1,
)
from .parse import aggregate_samples, parse_result
from .reference import DIMENSION_HEADERS, DIMENSIONS, PAPER_TO_DATA_KEY, PAPER_TUTOR_ORDER

DEFAULT_CACHE = Path(__file__).parent / "runs" / "judge_cache.jsonl"


# --------------------------------------------------------------------------- #
# Task enumeration
# --------------------------------------------------------------------------- #
def enumerate_tasks(conversations: list[dict], limit: int | None = None):
    """Yield (conversation_id, paper_tutor, data_key, dimension, history, response, solution)."""
    data_key_to_paper = {data: paper for paper, data in PAPER_TO_DATA_KEY.items()}
    emitted = 0
    for conv in conversations:
        cid = str(conv.get("conversation_id"))
        history = conv.get("conversation_history", "")
        solution = conv.get("Ground_Truth_Solution", "")
        for data_key, info in conv.get("anno_llm_responses", {}).items():
            paper = data_key_to_paper.get(data_key)
            if paper is None:
                continue
            response = info.get("response", "")
            for dim in DIMENSIONS:
                yield (cid, paper, data_key, dim, history, response, solution)
                emitted += 1
                if limit is not None and emitted >= limit:
                    return


# --------------------------------------------------------------------------- #
# Dry run
# --------------------------------------------------------------------------- #
def print_dry_run(conversations: list[dict], samples: int, reference_guided: bool = False) -> None:
    print("=" * 90)
    label = "reference-guided (OPT-IN)" if reference_guided else "byte-faithful Figure-6"
    print(f"DRY RUN — {label} prompt (NO network calls)")
    print("=" * 90)
    sampled = itertools.islice(enumerate_tasks(conversations), samples)
    for i, (cid, paper, _dk, dim, history, response, solution) in enumerate(sampled, start=1):
        messages = build_messages(
            history,
            response,
            dim,
            reference_solution=solution,
            reference_guided=reference_guided,
        )
        print(f"\n----- sample {i}: conversation={cid} tutor={paper} dimension={dim} -----")
        for msg in messages:
            print(f"\n[{msg['role'].upper()} MESSAGE]")
            print(msg["content"])
    print("\n" + "=" * 90)


def print_cost(conversations: list[dict], output_tokens: int, judge_samples: int = 1) -> None:
    est = estimate_cost(
        conversations, output_tokens_per_call=output_tokens, judge_samples=judge_samples
    )
    print("COST ESTIMATE — full live run (assumptions stated)")
    print("=" * 90)
    print(f"  self-consistency k (samples)     : {est.judge_samples}")
    print(f"  judge calls (responses x 8 x k)  : {est.n_calls:,}")
    print(f"  measured input chars (total)     : {est.total_input_chars:,}")
    print(f"  avg input chars / call           : {est.avg_input_chars:,.0f}")
    print(f"  chars/token heuristic            : {est.chars_per_token} (band 3.5-4.5)")
    print(f"  est. input tokens                : {est.input_tokens:,}")
    print(f"  assumed output tokens / call     : {est.output_tokens_per_call}")
    print(f"  est. output tokens (total)       : {est.total_output_tokens:,}")
    print(
        f"  assumed rates                    : "
        f"${est.rate_input_per_mtok}/MTok in, ${est.rate_output_per_mtok}/MTok out "
        "(Claude Haiku 4.5)"
    )
    print(f"  input cost                       : ${est.input_cost:,.2f}")
    print(f"  output cost                      : ${est.output_cost:,.2f}")
    print(f"  TOTAL (point estimate)           : ${est.total_cost:,.2f}")
    print(
        f"  TOTAL (range, token band)        : "
        f"${est.total_cost_low:,.2f} - ${est.total_cost_high:,.2f}"
    )
    print("=" * 90)


def print_phase_b_cost(
    conversations: list[dict],
    gen_rate_in: float,
    gen_rate_out: float,
    judge_rate_in: float,
    judge_rate_out: float,
    judge_samples: int = 1,
) -> None:
    est = estimate_phase_b_cost(
        conversations,
        gen_rate_in=gen_rate_in,
        gen_rate_out=gen_rate_out,
        judge_rate_in=judge_rate_in,
        judge_rate_out=judge_rate_out,
        judge_samples=judge_samples,
    )
    print("\nPHASE B COST ESTIMATE — score ONE model-under-test (judge-agnostic)")
    print("=" * 90)
    print(f"  conversations                    : {est.n_conversations:,}")
    print(f"  self-consistency k (samples)     : {est.judge_samples}")
    print(f"  generation calls                 : {est.gen_calls:,}")
    print(f"  judge calls (gen x 8 x k)        : {est.judge_calls:,}")
    print(f"  assumed response chars           : {est.assumed_response_chars:,.0f}")
    print(
        f"  gen tokens (in/out)              : {est.gen_input_tokens:,} / {est.gen_output_tokens:,}"
    )
    print(
        f"  judge tokens (in/out)            : "
        f"{est.judge_input_tokens:,} / {est.judge_output_tokens:,}"
    )
    print(f"  gen rates (in/out $/MTok)        : {est.gen_rate_in}/{est.gen_rate_out}")
    print(f"  judge rates (in/out $/MTok)      : {est.judge_rate_in}/{est.judge_rate_out}")
    print(f"  generation cost                  : ${est.gen_cost:,.2f}")
    print(f"  judge cost                       : ${est.judge_cost:,.2f}")
    print(f"  TOTAL (point estimate)           : ${est.total_cost:,.2f}")
    print("  (rates are parameters; pass --gen-rate-* / --judge-rate-* to override)")
    print("=" * 90)


# --------------------------------------------------------------------------- #
# Live run (gated)
# --------------------------------------------------------------------------- #
def run_live(
    conversations: list[dict],
    config: JudgeConfig,
    cache_path: Path,
    concurrency: int,
    limit: int | None,
    reference_guided: bool = False,
    k: int = 1,
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
    tasks = list(enumerate_tasks(conversations, limit=limit))
    todo = [t for t in tasks if not cache.is_done((t[0], t[1], t[3]))]
    k = max(1, k)
    print(f"Live run: {len(tasks)} tasks, {len(tasks) - len(todo)} cached, {len(todo)} to do.")
    print(f"  gateway={config.base_url}  model={config.model}  concurrency={concurrency}")
    print(f"  reference_guided={reference_guided}  self_consistency_k={k}")

    client = make_client(config)

    def _do(task) -> JudgeRecord:
        cid, paper, _dk, dim, history, response, solution = task
        messages = build_messages(
            history,
            response,
            dim,
            reference_solution=solution,
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
            conversation_id=cid,
            tutor=paper,
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
                print(f"  ! task failed: {exc}")
            if done % 100 == 0:
                print(f"  ... {done}/{len(todo)} completed")
    print(f"Live run complete: {done} new results, {failures} failures. Cache: {cache_path}")
    return 0


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _fmt(v: float) -> str:
    return " nan " if math.isnan(v) else f"{v:+.2f}"


def _pct(v: float) -> str:
    return (" nan " if math.isnan(v) else f"{v:6.2f}").rjust(8)


def print_metrics(
    conversations: list[dict],
    cache_path: Path,
    bootstrap: int = 0,
    macro_f1: bool = False,
) -> int:
    if not cache_path.exists():
        print(f"No cache at {cache_path}; run --live first")
        print("(or --validate for an offline self-test).")
        return 1
    cache = JudgeCache(cache_path)
    records = cache.records()
    gold_index = build_gold_index(conversations)

    ac = compute_ac(records, gold_index)
    damr = compute_judge_damr(records)

    header = "tutor".ljust(16) + "".join(DIMENSION_HEADERS[d].rjust(8) for d in DIMENSIONS)

    print("\nAC (Pearson judge-vs-gold) — per dimension, overall")
    print("dimension".ljust(28) + "AC".rjust(8) + "n_pairs".rjust(10))
    for dim in DIMENSIONS:
        line = dim.ljust(28) + _fmt(ac.per_dimension[dim]).rjust(8)
        line += str(ac.n_pairs_per_dimension[dim]).rjust(10)
        print(line)

    if bootstrap > 0:
        pairs = collect_dimension_pairs(records, gold_index)
        indicators = collect_dimension_desired_indicators(records)
        print(f"\n95% bootstrap CIs ({bootstrap} resamples) — per dimension, overall")
        print("dimension".ljust(28) + "AC 95% CI".rjust(20) + "DAMR% 95% CI".rjust(22))
        for dim in DIMENSIONS:
            lo, hi = bootstrap_ci_pearson(pairs[dim], n_boot=bootstrap)
            dlo, dhi = bootstrap_ci_rate(indicators[dim], n_boot=bootstrap)
            ci_ac = f"[{_fmt(lo).strip()}, {_fmt(hi).strip()}]"
            ci_damr = f"[{dlo:.1f}, {dhi:.1f}]"
            print(dim.ljust(28) + ci_ac.rjust(20) + ci_damr.rjust(22))

    print("\nAC per tutor x dimension (comparable to paper Tables 5/6)")
    print(header)
    for tutor in PAPER_TUTOR_ORDER:
        row = ac.per_tutor_dimension.get(tutor)
        if row is None:
            continue
        print(tutor.ljust(16) + "".join(_fmt(row[d]).rjust(8) for d in DIMENSIONS))

    print("\nJudge-derived DAMR (%) per tutor x dimension")
    print(header)
    for tutor in PAPER_TUTOR_ORDER:
        cells = damr.cells.get(tutor)
        if cells is None:
            continue
        vals = "".join(_pct(cells[d]) for d in DIMENSIONS)
        print(tutor.ljust(16) + vals)

    print("\nAC diff vs baselines (judge_AC - baseline_AC); '*' = we correlate better")
    cmp = compare_ac_to_baselines(ac)
    for name, per_tutor in cmp.items():
        print(f"\n  vs {name}:")
        print("  " + header)
        for tutor in PAPER_TUTOR_ORDER:
            diffs = per_tutor.get(tutor)
            if not diffs:
                continue
            cells = "".join(
                (f"{diffs[d]:+.2f}" + ("*" if diffs[d] > 0 else "")).rjust(8)
                for d in DIMENSIONS
                if d in diffs
            )
            print("  " + tutor.ljust(16) + cells)

    if macro_f1:
        mf1 = compute_macro_f1(records, gold_index)
        present = [v for v in mf1.values() if not math.isnan(v)]
        mean_mf1 = sum(present) / len(present) if present else math.nan
        print("\n[SECONDARY] 3-class macro-F1 (judge vs gold) — BEA-2025 comparable")
        print("(reporting only; primary metric stays DAMR, judge-reliability stays Pearson AC)")
        print("dimension".ljust(28) + "macroF1".rjust(10))
        for dim in DIMENSIONS:
            print(dim.ljust(28) + _pct(mf1[dim]).strip().rjust(10))
        print("mean".ljust(28) + _pct(mean_mf1).strip().rjust(10))

    print("\nSelf-bias caveat: N/A. Claude Haiku 4.5 (the judge) is not one of the 9 MRBench")
    print("tutors, so no row above is a self-authored subset. The 'Sonnet' row is a judged")
    print("tutor in the data, unrelated to the judge model.")
    return 0


# --------------------------------------------------------------------------- #
# Offline validation (fabricated outputs; no network)
# --------------------------------------------------------------------------- #
def run_validation() -> int:
    from .metrics import pearson

    checks: list[tuple[str, bool]] = []

    # --- parser ---
    p = parse_result("Feedback: Correctly found it. [RESULT] 1", "Mistake_Identification")
    checks.append(
        ("parse [RESULT] 1 -> score 1, label Yes", p.ok and p.score == 1 and p.label == "Yes")
    )

    p = parse_result("Feedback: Reveals it. [RESULT] 1", "Revealing_of_the_Answer")
    checks.append(
        (
            "revealing score 1 -> 'Yes (and the answer is correct)'",
            p.label == "Yes (and the answer is correct)",
        )
    )

    p = parse_result("Feedback: Neutral tone. [RESULT] 2", "Tutor_Tone")
    checks.append(("tone score 2 -> Neutral", p.ok and p.label == "Neutral"))

    p = parse_result("The score should be 3", "Coherence")
    checks.append(("fallback parse of trailing 3", p.ok and p.score == 3 and p.used_fallback))

    p = parse_result("no numeric score present", "Coherence")
    checks.append(("unparseable -> ok=False", (not p.ok) and p.score is None))

    # --- Pearson ---
    checks.append(("pearson perfect +1", abs(pearson([1, 2, 3], [1, 2, 3]) - 1.0) < 1e-9))
    checks.append(("pearson perfect -1", abs(pearson([1, 2, 3], [3, 2, 1]) + 1.0) < 1e-9))
    checks.append(("pearson constant -> nan", math.isnan(pearson([2, 2, 2], [1, 2, 3]))))

    # --- metrics on fabricated records ---
    # Two tutors, one dimension varied to have signal.
    gold_index = {
        ("c1", "GPT-4", "Mistake_Identification"): 1,
        ("c2", "GPT-4", "Mistake_Identification"): 2,
        ("c3", "GPT-4", "Mistake_Identification"): 3,
    }

    def rec(cid, score, label, ok=True):
        return JudgeRecord(
            cid, "GPT-4", "Mistake_Identification", score, label, "fb", ok, False, "raw", "m"
        )

    perfect = [rec("c1", 1, "Yes"), rec("c2", 2, "To some extent"), rec("c3", 3, "No")]
    ac = compute_ac(perfect, gold_index)
    checks.append(
        ("AC perfect match -> +1.0", abs(ac.per_dimension["Mistake_Identification"] - 1.0) < 1e-9)
    )

    anti = [rec("c1", 3, "No"), rec("c2", 2, "To some extent"), rec("c3", 1, "Yes")]
    ac2 = compute_ac(anti, gold_index)
    checks.append(
        (
            "AC anti-correlated -> -1.0",
            abs(ac2.per_dimension["Mistake_Identification"] + 1.0) < 1e-9,
        )
    )

    # judge-DAMR: 2 of 3 judged 'Yes' (desired) -> 66.67%
    damr = compute_judge_damr([rec("c1", 1, "Yes"), rec("c2", 1, "Yes"), rec("c3", 3, "No")])
    got = damr.cells["GPT-4"]["Mistake_Identification"]
    checks.append(("judge-DAMR 2/3 desired -> 66.67%", abs(got - (200.0 / 3.0)) < 1e-6))

    # parse-failure excluded from AC pairs
    ac3 = compute_ac([rec("c1", None, None, ok=False)], gold_index)
    checks.append(
        (
            "parse failures excluded from AC",
            ac3.n_pairs_per_dimension["Mistake_Identification"] == 0,
        )
    )

    # --- macro-F1 (SECONDARY) ---
    from .metrics import compute_macro_f1, macro_f1

    # perfect predictions over all 3 classes -> 1.0
    perfect_pairs = [(1, 1), (2, 2), (3, 3), (1, 1), (2, 2), (3, 3)]
    checks.append(("macro-F1 perfect -> 1.0", abs(macro_f1(perfect_pairs) - 1.0) < 1e-9))

    # degenerate majority-guesser: always predicts class 1; classes 2 and 3 get F1 0.
    # gold: 4x class1, 1x class2, 1x class3 -> only class-1 F1 is nonzero.
    guesser = [(1, 1), (1, 1), (1, 1), (1, 1), (1, 2), (1, 3)]
    f1_c1 = (2 * (4 / 6) * 1.0) / ((4 / 6) + 1.0)  # precision 4/6, recall 1.0
    checks.append(("macro-F1 majority-guesser << 1", abs(macro_f1(guesser) - f1_c1 / 3.0) < 1e-9))

    # rare middle class recovered perfectly still averages over all 3 classes.
    rare = [(2, 2), (1, 1), (1, 1), (1, 1), (3, 3), (1, 1)]
    checks.append(("macro-F1 rare-class perfect -> 1.0", abs(macro_f1(rare) - 1.0) < 1e-9))

    # compute_macro_f1 wires per-dimension pairs from records/gold.
    mf1 = compute_macro_f1(perfect, gold_index)
    checks.append(
        (
            "compute_macro_f1 perfect records -> 1.0",
            abs(mf1["Mistake_Identification"] - 1.0) < 1e-9,
        )
    )

    # --- reference-guided prompt construction (OPT-IN, default OFF) ---
    from .judge_prompt import REFERENCE_GUIDED_DIMENSIONS, build_messages

    hist, resp, sol = "H", "R", "x=42"
    base = build_messages(hist, resp, "Mistake_Identification")
    off = build_messages(
        hist, resp, "Mistake_Identification", reference_solution=sol, reference_guided=False
    )
    checks.append(("reference-guided OFF is byte-identical", base == off))

    on_corr = build_messages(
        hist, resp, "Mistake_Identification", reference_solution=sol, reference_guided=True
    )
    checks.append(
        (
            "reference-guided ON injects solution for correctness dim",
            sol in on_corr[1]["content"] and on_corr != base,
        )
    )

    on_style = build_messages(
        hist, resp, "Tutor_Tone", reference_solution=sol, reference_guided=True
    )
    style_base = build_messages(hist, resp, "Tutor_Tone")
    checks.append(
        (
            "reference-guided ON leaves non-correctness dim byte-identical",
            on_style == style_base and sol not in on_style[1]["content"],
        )
    )
    expected_ref_dims = frozenset(
        {
            "Mistake_Identification",
            "Mistake_Location",
            "Providing_Guidance",
            "Revealing_of_the_Answer",
        }
    )
    checks.append(
        (
            "reference-guided dims = {MistakeId, MistakeLoc, Guidance, Reveal}",
            expected_ref_dims == REFERENCE_GUIDED_DIMENSIONS,
        )
    )

    # --- self-consistency (k-sample majority vote) ---
    from .parse import aggregate_samples, majority_vote

    checks.append(("majority vote 2v1", majority_vote([1, 1, 2]) == 1))
    checks.append(("majority vote tie -> smallest", majority_vote([3, 1, 3, 1]) == 1))
    checks.append(("majority vote all-None -> None", majority_vote([None, None]) is None))

    raws3 = [
        "Feedback: a [RESULT] 1",
        "Feedback: b [RESULT] 2",
        "Feedback: c [RESULT] 1",
    ]
    combined, per_sample = aggregate_samples(raws3, "Mistake_Identification")
    checks.append(
        (
            "k=3 majority -> score 1, 3 samples recorded",
            combined.ok and combined.score == 1 and len(per_sample) == 3,
        )
    )
    single = aggregate_samples(["Feedback: a [RESULT] 2"], "Mistake_Identification")[0]
    direct = parse_result("Feedback: a [RESULT] 2", "Mistake_Identification")
    checks.append(
        (
            "k=1 aggregate reproduces parse_result",
            single.score == direct.score and single.label == direct.label and single.ok,
        )
    )

    print("OFFLINE VALIDATION (fabricated judge outputs, no network)")
    print("=" * 70)
    all_ok = True
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        all_ok = all_ok and ok
    print("=" * 70)
    print("RESULT:", "ALL PASS" if all_ok else "FAILURES PRESENT")
    return 0 if all_ok else 1


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "MRBench Milestone-1 Claude Haiku 4.5 judge (TrueFoundry gateway). Safe by default."
        ),
    )
    parser.add_argument("--data-path", default=None, help="Path to MRBench_V1.json.")
    parser.add_argument("--samples", type=int, default=3, help="Dry-run prompts to print.")
    parser.add_argument("--output-tokens", type=int, default=40, help="Assumed output tokens/call.")
    parser.add_argument("--live", action="store_true", help="Perform the gated live run.")
    parser.add_argument("--limit", type=int, default=None, help="Cap number of judge calls.")
    parser.add_argument("--concurrency", type=int, default=8, help="Live-run worker threads.")
    parser.add_argument("--cache", default=str(DEFAULT_CACHE), help="JSONL cache path.")
    parser.add_argument(
        "--metrics", action="store_true", help="Compute AC + judge-DAMR from cache."
    )
    parser.add_argument("--validate", action="store_true", help="Offline parser/metrics self-test.")
    parser.add_argument("--phase-b", action="store_true", help="Also print Phase B cost estimate.")
    parser.add_argument(
        "--bootstrap", type=int, default=0, help="Bootstrap resamples for CIs (0=off)."
    )
    parser.add_argument("--gen-rate-in", type=float, default=3.0, help="Phase B gen $/MTok input.")
    parser.add_argument(
        "--gen-rate-out", type=float, default=15.0, help="Phase B gen $/MTok output."
    )
    parser.add_argument("--judge-rate-in", type=float, default=1.0, help="Phase B judge $/MTok in.")
    parser.add_argument(
        "--judge-rate-out", type=float, default=5.0, help="Phase B judge $/MTok out."
    )
    parser.add_argument(
        "--reference-guided",
        action="store_true",
        help="OPT-IN: inject Ground_Truth_Solution for correctness-linked dims (off-protocol).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=1,
        help="OPT-IN self-consistency: judge samples per call, majority-voted (default 1).",
    )
    parser.add_argument(
        "--macro-f1",
        action="store_true",
        help="With --metrics, also print SECONDARY 3-class macro-F1 (judge vs gold).",
    )
    args = parser.parse_args(argv)

    if args.validate:
        return run_validation()

    conversations = load_conversations(args.data_path)
    cache_path = Path(args.cache)

    if args.live:
        rc = run_live(
            conversations,
            JudgeConfig.from_env(),
            cache_path,
            args.concurrency,
            args.limit,
            reference_guided=args.reference_guided,
            k=args.k,
        )
        if args.metrics and rc == 0:
            rc = print_metrics(
                conversations, cache_path, bootstrap=args.bootstrap, macro_f1=args.macro_f1
            )
        return rc

    if args.metrics:
        return print_metrics(
            conversations, cache_path, bootstrap=args.bootstrap, macro_f1=args.macro_f1
        )

    # Default: dry run + cost estimate, zero network.
    print_dry_run(conversations, args.samples, reference_guided=args.reference_guided)
    print_cost(conversations, args.output_tokens, judge_samples=args.k)
    if args.phase_b:
        print_phase_b_cost(
            conversations,
            args.gen_rate_in,
            args.gen_rate_out,
            args.judge_rate_in,
            args.judge_rate_out,
            judge_samples=args.k,
        )
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
