"""Sweep driver: run 0-7B models across MCQ + open-ended benchmarks.

Key properties:
  * each model is loaded **once** (resident) - never reloaded per benchmark;
  * benchmark-first fleet ordering is achieved by sharding (1 model per worker
    => every worker marches benchmarks in the same order) and a per-benchmark
    completion barrier marker; ``--resident-all`` reproduces the exact
    benchmark-outer / model-inner ordering inside a single co-located process;
  * results stream to one durable file per (benchmark, model) with question-level
    resume (see results_writer / README 4.4).

Examples:
  # local smoke test, no GPU, synthetic-free (uses real loaders unless --backend mock + tiny cap)
  python run_benchmark.py --benchmarks arc_easy,squad_v2 --models Qwen/Qwen2.5-0.5B \\
      --backend mock --max-samples 5

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
    ensure_dirs,
    is_pair_done,
    mark_pair_done,
)
from config import InferenceConfig, JudgeConfig
from datasets_registry import (
    ALL_BENCHMARKS,
    BENCHMARKS,
    MCQ_BENCHMARKS,
    OPEN_BENCHMARKS,
    load_benchmark,
)
from engine import Engine, GenParams, probe_backend
from judge_prometheus import Judge
from mcq_scoring import score_mcq
from models_registry import ModelSpec, load_models, select_models
from open_generate import generate_open


def resolve_benchmarks(arg: str) -> list[str]:
    tokens = [t.strip() for t in arg.split(",") if t.strip()]
    out: list[str] = []
    for t in tokens:
        if t == "all":
            out.extend(ALL_BENCHMARKS)
        elif t == "mcq":
            out.extend(MCQ_BENCHMARKS)
        elif t == "open":
            out.extend(OPEN_BENCHMARKS)
        elif t in BENCHMARKS:
            out.append(t)
        else:
            raise SystemExit(f"unknown benchmark: {t}")
    # de-dup, preserve order
    seen: set[str] = set()
    return [b for b in out if not (b in seen or seen.add(b))]


def make_engine(spec: ModelSpec, cfg: InferenceConfig) -> Engine:
    backend = probe_backend(spec, cfg.backend)
    return Engine(spec, backend=backend, vllm_cfg=cfg.vllm)


def gen_params_for(cfg: InferenceConfig, benchmark: str) -> GenParams:
    eff = cfg.for_benchmark(benchmark)
    return GenParams(
        temperature=eff.get("temperature", 0.0),
        top_p=eff.get("top_p", 1.0),
        max_tokens=eff.get("max_tokens", 512),
        seed=eff.get("seed", 1234),
    )


def run_pair(
    engine: Engine,
    spec: ModelSpec,
    benchmark: str,
    cfg: InferenceConfig,
    judge: Judge | None,
    max_samples: int | None,
    use_cache: bool,
) -> None:
    if is_pair_done(benchmark, spec.id):
        return
    bspec = BENCHMARKS[benchmark]
    questions = load_benchmark(benchmark, max_samples, cfg.sample_seed, use_cache=use_cache)
    t0 = time.monotonic()
    if bspec.type == BenchType.MCQ:
        n = score_mcq(engine, spec, benchmark, questions, cfg.writer)
        kind = "mcq"
    else:
        gen = gen_params_for(cfg, benchmark)
        n = generate_open(engine, spec, benchmark, questions, gen, cfg.writer)
        if judge is not None:
            judge.judge_file(benchmark, spec.id, bspec.rubric_key, cfg.writer)
        kind = "open"
    mark_pair_done(benchmark, spec.id, len(questions))
    dt = time.monotonic() - t0
    print(f"  [{kind}] {benchmark:14s} x {spec.id:40s} +{n:5d} new  ({dt:.1f}s)", flush=True)


def _all_pairs_done(benchmark: str, specs: list[ModelSpec]) -> bool:
    return all(is_pair_done(benchmark, s.id) for s in specs)


def run(args: argparse.Namespace) -> None:
    cfg = InferenceConfig.load(args.inference_config)
    if args.backend:
        cfg.backend = args.backend
    max_samples = args.max_samples if args.max_samples is not None else cfg.max_samples
    use_cache = not args.no_cache

    specs = select_models(
        load_models(),
        only=args.models.split(",") if args.models else None,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    if args.limit_models:
        specs = specs[: args.limit_models]
    benchmarks = resolve_benchmarks(args.benchmarks)
    if not specs:
        raise SystemExit("no models selected")

    # Pre-flight the datasets once so an unreachable source (gated, moved,
    # schema drift) cannot abort a sweep that spans days.
    usable: list[str] = []
    for b in benchmarks:
        try:
            load_benchmark(b, max_samples, cfg.sample_seed, use_cache=use_cache)
        except Exception as exc:  # noqa: BLE001
            print(f"[drop benchmark] {b}: {type(exc).__name__}: {exc}", flush=True)
            continue
        usable.append(b)
    benchmarks = usable
    if not benchmarks:
        raise SystemExit("no benchmarks could be loaded")

    need_judge = any(BENCHMARKS[b].type == BenchType.OPEN for b in benchmarks) and not args.no_judge
    judge = None
    if need_judge:
        jcfg = JudgeConfig.load(args.judge_config)
        judge = Judge(jcfg, backend=args.judge_backend or (args.backend if args.backend else None))

    ensure_dirs(MANIFEST_DIR)
    print(
        f"models={len(specs)} benchmarks={len(benchmarks)} backend={cfg.backend} "
        f"max_samples={max_samples} judge={'on' if judge else 'off'} "
        f"order={'resident-all' if args.resident_all else 'model-outer'}",
        flush=True,
    )

    if args.resident_all:
        # co-located: hold every shard model resident, iterate benchmark-outer.
        engines = {s.id: make_engine(s, cfg) for s in specs}
        try:
            for benchmark in benchmarks:
                print(f"== benchmark {benchmark} ==", flush=True)
                for spec in specs:
                    run_pair(engines[spec.id], spec, benchmark, cfg, judge, max_samples, use_cache)
                if _all_pairs_done(benchmark, specs):
                    benchmark_done_path(benchmark).write_text("done\n")
        finally:
            for e in engines.values():
                e.close()
    else:
        # efficient default: one model resident at a time, all its benchmarks.
        for spec in specs:
            print(f"== model {spec.id} (backend routed) ==", flush=True)
            try:
                engine = make_engine(spec, cfg)
            except Exception as exc:  # noqa: BLE001
                print(f"[skip model] {spec.id}: {type(exc).__name__}: {exc}", flush=True)
                continue
            try:
                for benchmark in benchmarks:
                    try:
                        run_pair(engine, spec, benchmark, cfg, judge, max_samples, use_cache)
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

    if judge is not None:
        judge.close()
    print("done.", flush=True)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--benchmarks", default="all", help="all | mcq | open | comma list")
    ap.add_argument("--models", help="comma list of HF ids or trailing names")
    ap.add_argument("--shard-index", type=int, default=None)
    ap.add_argument("--num-shards", type=int, default=None)
    ap.add_argument("--limit-models", type=int, default=None)
    ap.add_argument("--backend", choices=["vllm", "hf", "mock"], help="override config backend")
    ap.add_argument(
        "--judge-backend", choices=["vllm", "hf", "mock"], help="override judge backend"
    )
    ap.add_argument(
        "--no-judge", action="store_true", help="generate open responses but skip judging"
    )
    ap.add_argument("--max-samples", type=int, default=None, help="override per-benchmark cap")
    ap.add_argument("--no-cache", action="store_true", help="ignore normalized dataset cache")
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
    args = build_parser().parse_args(argv)
    if args.list:
        print("MCQ :", ", ".join(MCQ_BENCHMARKS))
        print("Open:", ", ".join(OPEN_BENCHMARKS))
        return
    from pathlib import Path

    if args.inference_config:
        args.inference_config = Path(args.inference_config)
    if args.judge_config:
        args.judge_config = Path(args.judge_config)
    run(args)


if __name__ == "__main__":
    sys.exit(main())
