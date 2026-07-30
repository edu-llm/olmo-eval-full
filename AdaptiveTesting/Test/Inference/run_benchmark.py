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
  python run_benchmark.py --benchmarks openbookqa,squad_v2 --models Qwen/Qwen2.5-0.5B \\
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
    CPU_SWEEP_BENCHMARKS,
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


class SlowModelError(RuntimeError):
    """Raised when a probe of N questions exceeds the wall-clock budget."""


def _probe_speed(
    engine: Engine,
    spec: ModelSpec,
    benchmark: str,
    questions: list,
    cfg: InferenceConfig,
    probe_n: int,
    max_seconds: float,
) -> None:
    """Score/generate ``probe_n`` items; skip the model if wall time exceeds budget."""
    if probe_n <= 0 or max_seconds <= 0 or not questions:
        return
    sample = questions[: min(probe_n, len(questions))]
    t0 = time.monotonic()
    bspec = BENCHMARKS[benchmark]
    if bspec.type == BenchType.MCQ:
        score_mcq(engine, spec, benchmark, sample, cfg.writer)
    else:
        gen = gen_params_for(cfg, benchmark)
        generate_open(engine, spec, benchmark, sample, gen, cfg.writer)
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
    judge: Judge | None,
    max_samples: int | None,
    use_cache: bool,
    probe_n: int = 0,
    probe_max_seconds: float = 0.0,
) -> None:
    if is_pair_done(benchmark, spec.id):
        return
    bspec = BENCHMARKS[benchmark]
    questions = load_benchmark(benchmark, max_samples, cfg.sample_seed, use_cache=use_cache)
    if probe_n > 0 and probe_max_seconds > 0:
        _probe_speed(engine, spec, benchmark, questions, cfg, probe_n, probe_max_seconds)
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
        f"order={'resident-all' if args.resident_all else 'model-outer'} "
        f"probe={probe_n}q/{probe_max:.0f}s",
        flush=True,
    )

    if args.resident_all:
        # co-located: hold every shard model resident, iterate benchmark-outer.
        engines = {s.id: make_engine(s, cfg) for s in specs}
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
                            judge,
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
            print(f"== model {spec.id} (backend routed) ==", flush=True)
            try:
                engine = make_engine(spec, cfg)
            except Exception as exc:  # noqa: BLE001
                print(f"[skip model] {spec.id}: {type(exc).__name__}: {exc}", flush=True)
                continue
            try:
                for benchmark in benchmarks:
                    try:
                        run_pair(
                            engine,
                            spec,
                            benchmark,
                            cfg,
                            judge,
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

    if judge is not None:
        judge.close()
    print("done.", flush=True)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--benchmarks",
        default="all",
        help="all | mcq | open | cpu_sweep | comma list",
    )
    ap.add_argument("--models", help="comma list of HF ids or trailing names")
    ap.add_argument(
        "--models-yaml",
        default=None,
        help="path to models yaml (default: Inputs/Models/models.yaml)",
    )
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
    args = build_parser().parse_args(argv)
    if args.list:
        print("MCQ :", ", ".join(MCQ_BENCHMARKS))
        print("Open:", ", ".join(OPEN_BENCHMARKS))
        print("CPU :", ", ".join(CPU_SWEEP_BENCHMARKS))
        return
    from pathlib import Path

    if args.inference_config:
        args.inference_config = Path(args.inference_config)
    if args.judge_config:
        args.judge_config = Path(args.judge_config)
    run(args)


if __name__ == "__main__":
    sys.exit(main())
