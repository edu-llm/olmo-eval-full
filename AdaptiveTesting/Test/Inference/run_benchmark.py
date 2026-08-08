"""Unified pre-calibration driver: run models across MCQ + FRQ benchmarks.

One script covers both item types:
  * **MCQ** -> log-likelihood ranking over option continuations, scored
    immediately to ``correct``/``wrong`` (``mcq_scoring``).
  * **FRQ** -> faithful tutor prompts (per-benchmark/use_case system turn +
    conversation history) and free-form generation stored in the Model Output
    schema (``frq_generate``); grading happens later via ``judge_all.py``.

Key properties:
  * each model is loaded **once** (resident) - never reloaded per benchmark;
  * benchmark-first fleet ordering is achieved by sharding (1 model per worker
    => every worker marches benchmarks in the same order) and a per-benchmark
    completion barrier marker; ``--resident-all`` reproduces the exact
    benchmark-outer / model-inner ordering inside a single co-located process;
  * results stream to one durable file per (benchmark, model) with item-level
    resume. FRQ resume is validity-aware: rows whose generation failed are
    regenerated rather than counted as done, and a pair is only marked complete
    once every scenario has a valid row.

Examples:
  # local smoke test, no GPU, no network
  python run_benchmark.py --benchmarks synth_mcq,synth_open \\
      --models Qwen/Qwen2.5-0.5B --backend mock --max-samples 5

  # everything (3 MCQ + 6 FRQ)
  python run_benchmark.py --benchmarks all

  # one AWS worker handling shard 3 of 100
  python run_benchmark.py --benchmarks all --shard-index 3 --num-shards 100
"""

from __future__ import annotations

import argparse
import sys
import time

from common import (
    MANIFEST_DIR,
    BenchType,
    benchmark_done_path,
    bootstrap_env,
    ensure_dirs,
    is_pair_done,
    mark_pair_done,
)
from config import InferenceConfig
from datasets_registry import (
    ALL_BENCHMARKS,
    BENCHMARKS,
    CPU_SWEEP_BENCHMARKS,
    MCQ_BENCHMARKS,
    OPEN_BENCHMARKS,
    load_items,
    register_frq_benchmark,
)
from engine import Engine, build_engine, probe_backend
from frq_generate import generate_frq
from frq_scenarios import FRQBankNotFound, load_banks_yaml
from mcq_scoring import score_mcq
from models_registry import ModelSpec, load_models, select_models


def register_extra_banks(paths: list[str]) -> list[str]:
    """Register FRQ banks declared in ``--frq-banks`` YAML files.

    Must run before :func:`resolve_benchmarks`, which rejects unknown names.
    """
    keys: list[str] = []
    for path in paths:
        for key in load_banks_yaml(path):
            register_frq_benchmark(key)
            keys.append(key)
    return keys


def resolve_benchmarks(arg: str) -> list[str]:
    tokens = [t.strip() for t in arg.split(",") if t.strip()]
    out: list[str] = []
    for t in tokens:
        if t == "all":
            out.extend(ALL_BENCHMARKS)
        elif t == "mcq":
            out.extend(MCQ_BENCHMARKS)
        elif t in {"open", "frq"}:
            out.extend(OPEN_BENCHMARKS)
        elif t in {"cpu_sweep", "cpu-sweep"}:
            out.extend(CPU_SWEEP_BENCHMARKS)
        elif t in BENCHMARKS:
            out.append(t)
        else:
            raise SystemExit(f"unknown benchmark: {t}")
    # de-dup, preserve order
    seen: set[str] = set()
    return [b for b in out if not (b in seen or seen.add(b))]


def make_engine(spec: ModelSpec, cfg: InferenceConfig) -> Engine:
    """Load a model with retries + smoke test + vLLM->hf fallback (see
    :func:`engine.build_engine`)."""
    return build_engine(spec, cfg.backend, cfg.vllm)


def _log_engine(
    spec: ModelSpec, cfg: InferenceConfig, engine: Engine, degraded: list[str]
) -> str:
    """Print the backend the model actually runs on, loudly when it degraded from
    vLLM to transformers, and return that backend. A model routed to hf by
    capability (``hf_fallback``) is not a degradation."""
    routed = probe_backend(spec, cfg.backend)
    if engine.backend != routed:
        degraded.append(spec.id)
        print(
            f"  [engine] *** DEGRADED: {spec.id} requested {routed}, running on "
            f"{engine.backend} *** (max_model_len={spec.max_model_len})",
            flush=True,
        )
    else:
        print(
            f"  [engine] {spec.id} backend={engine.backend} "
            f"max_model_len={spec.max_model_len}",
            flush=True,
        )
    return engine.backend


def _frq_overrides(cfg: InferenceConfig, benchmark: str) -> dict:
    """Decoding overrides for FRQ generation from inference.yaml (global
    `generation` + per-benchmark `overrides`). Manifest values win where unset.

    The config's generic `max_tokens` is deliberately NOT read as the FRQ budget.
    It is an MCQ-era 512 that silently out-ranked the manifest's respgen-parity
    `max_new_tokens: 4096` and truncated every long-form answer. FRQ therefore
    keeps the manifest default unless a config explicitly sets `max_new_tokens`
    (globally under `generation:` or per benchmark under `overrides:`), which
    remains a deliberate, visible override.
    """
    eff = cfg.for_benchmark(benchmark)
    keys = ("temperature", "top_p", "seed", "repetition_penalty", "max_new_tokens")
    return {k: eff[k] for k in keys if k in eff}


class SlowModelError(RuntimeError):
    """Raised when a probe of N items exceeds the wall-clock budget."""


def _probe_speed(
    engine: Engine,
    spec: ModelSpec,
    benchmark: str,
    items: list,
    cfg: InferenceConfig,
    probe_n: int,
    max_seconds: float,
) -> None:
    """Score/generate ``probe_n`` items; skip the model if wall time exceeds budget."""
    if probe_n <= 0 or max_seconds <= 0 or not items:
        return
    sample = items[: min(probe_n, len(items))]
    t0 = time.monotonic()
    if BENCHMARKS[benchmark].type == BenchType.MCQ:
        score_mcq(engine, spec, benchmark, sample, cfg.writer)
    else:
        generate_frq(
            engine, spec, benchmark, sample, cfg.writer,
            overrides=_frq_overrides(cfg, benchmark),
        )
    dt = time.monotonic() - t0
    if dt > max_seconds:
        raise SlowModelError(
            f"probe of {len(sample)} {benchmark} items took {dt:.1f}s "
            f"> budget {max_seconds:.1f}s"
        )


def run_pair(
    engine: Engine,
    spec: ModelSpec,
    benchmark: str,
    cfg: InferenceConfig,
    max_samples: int | None,
    use_cache: bool,
    probe_n: int = 0,
    probe_max_seconds: float = 0.0,
) -> None:
    if is_pair_done(benchmark, spec.id):
        return
    items = load_items(benchmark, max_samples, cfg.sample_seed, use_cache=use_cache)
    if probe_n > 0 and probe_max_seconds > 0:
        _probe_speed(engine, spec, benchmark, items, cfg, probe_n, probe_max_seconds)
    t0 = time.monotonic()
    if BENCHMARKS[benchmark].type == BenchType.MCQ:
        n = score_mcq(engine, spec, benchmark, items, cfg.writer)
        complete = True
        kind = "mcq"
    else:
        res = generate_frq(
            engine, spec, benchmark, items, cfg.writer,
            overrides=_frq_overrides(cfg, benchmark),
        )
        n, complete, kind = res.written, res.complete, "frq"
    # Only bank a .done marker once every item has a usable row; otherwise the
    # pair must stay resumable so failed cells regenerate on the next run.
    if complete:
        mark_pair_done(benchmark, spec.id, len(items))
    dt = time.monotonic() - t0
    flag = "" if complete else "  [incomplete - will retry]"
    print(
        f"  [{kind}] {benchmark:14s} x {spec.id:40s} +{n:5d} new  ({dt:.1f}s){flag}",
        flush=True,
    )


def _all_pairs_done(benchmark: str, specs: list[ModelSpec]) -> bool:
    return all(is_pair_done(benchmark, s.id) for s in specs)


def _print_backend_summary(backends: dict[str, str], degraded: list[str]) -> None:
    """End-of-run roll-up: which models ran where. Degradation to transformers is
    listed explicitly - those responses are valid but were produced on a slower
    path, which matters when comparing throughput or debugging a partial sweep."""
    if not backends:
        return
    counts: dict[str, int] = {}
    for b in backends.values():
        counts[b] = counts.get(b, 0) + 1
    print(
        "backends: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())),
        flush=True,
    )
    if degraded:
        print(
            f"ALERT: {len(degraded)} model(s) degraded vllm -> hf: {', '.join(sorted(degraded))}",
            flush=True,
        )
    failed = sorted(m for m, b in backends.items() if b == "failed")
    if failed:
        print(f"ALERT: {len(failed)} model(s) failed to load: {', '.join(failed)}", flush=True)


def run(args: argparse.Namespace) -> None:
    cfg = InferenceConfig.load(args.inference_config)
    if args.backend:
        cfg.backend = args.backend
    max_samples = args.max_samples if args.max_samples is not None else cfg.max_samples
    use_cache = not args.no_cache

    from pathlib import Path

    models_yaml = Path(args.models_yaml) if args.models_yaml else None
    specs = select_models(
        load_models(models_yaml),
        only=args.models.split(",") if args.models else None,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    if args.limit_models:
        specs = specs[: args.limit_models]
    benchmarks = resolve_benchmarks(args.benchmarks)
    if not specs:
        raise SystemExit("no models selected")

    probe_n = int(args.probe_questions)
    probe_max = float(args.probe_max_seconds)

    # Pre-flight the item sources once so an unreachable source (gated, moved,
    # schema drift) cannot abort a sweep that spans days. A missing FRQ bank is a
    # local-data problem, not a flaky remote, so it is surfaced as a loud ALERT.
    usable: list[str] = []
    missing_banks: list[str] = []
    for b in benchmarks:
        try:
            load_items(b, max_samples, cfg.sample_seed, use_cache=use_cache)
        except FRQBankNotFound as exc:
            print(f"\n*** ALERT: FRQ dataset not found locally ***\n  {exc}\n", flush=True)
            missing_banks.append(b)
            continue
        except Exception as exc:  # noqa: BLE001
            print(f"[drop benchmark] {b}: {type(exc).__name__}: {exc}", flush=True)
            continue
        usable.append(b)
    benchmarks = usable
    if not benchmarks:
        raise SystemExit("no benchmarks could be loaded")

    ensure_dirs(MANIFEST_DIR)
    print(
        f"models={len(specs)} benchmarks={len(benchmarks)} backend={cfg.backend} "
        f"max_samples={max_samples} "
        f"order={'resident-all' if args.resident_all else 'model-outer'} "
        f"probe={probe_n}q/{probe_max:.0f}s",
        flush=True,
    )
    if missing_banks:
        print(f"ALERT: skipped (missing FRQ banks): {', '.join(missing_banks)}", flush=True)

    # backend each model actually ran on (vllm | hf | mock | failed), so a silent
    # degradation to transformers is visible in the run's final summary.
    backends: dict[str, str] = {}
    degraded: list[str] = []

    if args.resident_all:
        # co-located: hold every shard model resident, iterate benchmark-outer.
        engines: dict[str, Engine] = {}
        for spec in specs:
            try:
                engines[spec.id] = make_engine(spec, cfg)
                backends[spec.id] = _log_engine(spec, cfg, engines[spec.id], degraded)
            except Exception as exc:  # noqa: BLE001 - one bad model must not kill the run
                backends[spec.id] = "failed"
                print(f"[skip model] {spec.id}: {type(exc).__name__}: {exc}", flush=True)
        specs = [s for s in specs if s.id in engines]
        if not specs:
            raise SystemExit("no models could be loaded")
        try:
            for benchmark in benchmarks:
                print(f"== benchmark {benchmark} ==", flush=True)
                for spec in specs:
                    try:
                        run_pair(
                            engines[spec.id],
                            spec,
                            benchmark,
                            cfg,
                            max_samples,
                            use_cache,
                            probe_n=probe_n,
                            probe_max_seconds=probe_max,
                        )
                    except SlowModelError as exc:
                        print(f"[skip slow model] {spec.id}: {exc}", flush=True)
                    except Exception as exc:  # noqa: BLE001
                        print(
                            f"  [skip] {benchmark} x {spec.id}: {type(exc).__name__}: {exc}",
                            flush=True,
                        )
                if _all_pairs_done(benchmark, specs):
                    benchmark_done_path(benchmark).write_text("done\n")
        finally:
            for e in engines.values():
                e.close()
    else:
        # efficient default: one model resident at a time, all its benchmarks.
        for spec in specs:
            print(f"== model {spec.id} ==", flush=True)
            try:
                engine = make_engine(spec, cfg)
            except Exception as exc:  # noqa: BLE001
                backends[spec.id] = "failed"
                print(f"[skip model] {spec.id}: {type(exc).__name__}: {exc}", flush=True)
                continue
            backends[spec.id] = _log_engine(spec, cfg, engine, degraded)
            try:
                for benchmark in benchmarks:
                    try:
                        run_pair(
                            engine,
                            spec,
                            benchmark,
                            cfg,
                            max_samples,
                            use_cache,
                            probe_n=probe_n,
                            probe_max_seconds=probe_max,
                        )
                    except SlowModelError as exc:
                        print(f"[skip slow model] {spec.id}: {exc}", flush=True)
                        break  # abandon remaining benches for this model
                    except Exception as exc:  # noqa: BLE001
                        print(
                            f"  [skip] {benchmark} x {spec.id}: {type(exc).__name__}: {exc}",
                            flush=True,
                        )
            finally:
                engine.close()
        for benchmark in benchmarks:
            if _all_pairs_done(benchmark, specs):
                benchmark_done_path(benchmark).write_text("done\n")

    _print_backend_summary(backends, degraded)
    print("done. (FRQ responses are unjudged - run judge_all.py next)", flush=True)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--benchmarks",
        default="all",
        help="all | mcq | frq (= open) | cpu_sweep | comma list",
    )
    ap.add_argument("--models", help="comma list of HF ids or trailing names")
    ap.add_argument(
        "--models-yaml",
        default=None,
        help="path to models yaml (default: Inputs/Models/models.yaml)",
    )
    ap.add_argument(
        "--frq-banks",
        action="append",
        default=None,
        metavar="PATH",
        help="YAML file registering extra FRQ scenario banks (repeatable); "
        "see configs/frq_banks.yaml",
    )
    ap.add_argument("--shard-index", type=int, default=None)
    ap.add_argument("--num-shards", type=int, default=None)
    ap.add_argument("--limit-models", type=int, default=None)
    ap.add_argument("--backend", choices=["vllm", "hf", "mock"], help="override config backend")
    # Judging is now a separate stage (judge_all.py) so a judge failure can never
    # lose generations. These are accepted-but-ignored for script compatibility.
    ap.add_argument("--judge-backend", choices=["vllm", "hf", "mock"], help=argparse.SUPPRESS)
    ap.add_argument("--no-judge", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--max-samples", type=int, default=None, help="override per-benchmark cap")
    ap.add_argument("--no-cache", action="store_true", help="ignore normalized dataset cache")
    ap.add_argument(
        "--probe-questions",
        type=int,
        default=0,
        help="if >0, time the first N questions before full run; skip model if over budget",
    )
    ap.add_argument(
        "--probe-max-seconds",
        type=float,
        default=0.0,
        help="wall-clock budget for --probe-questions (e.g. 600 for 100Q / 10min)",
    )
    ap.add_argument(
        "--resident-all",
        action="store_true",
        help="co-locate all shard models; benchmark-outer order",
    )
    ap.add_argument("--inference-config", default=None)
    ap.add_argument("--judge-config", default=None)
    ap.add_argument("--list", action="store_true", help="list benchmarks and exit")
    return ap


def main(argv: list[str] | None = None) -> None:
    # Must precede the first HTTPS connection (TLS trust store + .env HF_TOKEN).
    bootstrap_env()
    args = build_parser().parse_args(argv)
    # Before --list and before resolve_benchmarks, so extra banks are both
    # listable and nameable.
    if args.frq_banks:
        print(f"registered FRQ bank(s): {', '.join(register_extra_banks(args.frq_banks))}")
    if args.list:
        print("MCQ (HuggingFace)      :", ", ".join(MCQ_BENCHMARKS))
        print("FRQ (local banks)      :", ", ".join(OPEN_BENCHMARKS))
        print("Full pre-calibration   :", ", ".join(CPU_SWEEP_BENCHMARKS))
        return
    from pathlib import Path

    if args.judge_backend or args.no_judge:
        print(
            "note: judging is a separate stage now; --judge-backend/--no-judge are "
            "ignored. Run judge_all.py after generation.",
            file=sys.stderr,
        )
    if args.inference_config:
        args.inference_config = Path(args.inference_config)
    run(args)


if __name__ == "__main__":
    sys.exit(main())
