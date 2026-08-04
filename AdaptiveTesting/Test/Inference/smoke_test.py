"""End-to-end smoke test for the MCQ + FRQ pre-calibration pipeline.

Picks 5 random models from the roster and runs each through the REAL pipeline -
the same ``make_engine`` / ``score_mcq`` / ``generate_frq`` the full sweep calls -
over 5 MCQ questions and 5 FRQ scenarios. Reports model outputs, errors found in
the responses, and a timing breakdown (download / load / inference / cumulative).

Why the indirection through ``run_benchmark``: the engine construction and the
FRQ override resolution are imported from the driver rather than reimplemented,
so this test fails when the driver breaks instead of passing against a private
copy of the logic.

Isolation: every output path is redirected to ``Outputs/_smoke/<run-id>/``. This
never writes into the real ``Outputs/{mcq,open}`` trees and never creates the
``.done`` markers that would make a later real sweep skip those pairs.

Read SMOKE_TEST.md before running on a new machine.

    python smoke_test.py --dry-run              # preflight + plan, no models
    python smoke_test.py --synthetic --backend mock   # offline, no weights
    python smoke_test.py --max-params-b 1.5     # real weights, small download
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Trust store + .env must be installed before anything opens an HTTPS connection.
from common import bootstrap_env

bootstrap_env()

import common  # noqa: E402  (deliberately after bootstrap_env)
from common import EDULLM_ROOT  # noqa: E402
from config import InferenceConfig  # noqa: E402
from datasets_registry import MCQ_BENCHMARKS, OPEN_BENCHMARKS, load_items  # noqa: E402
from engine import probe_backend  # noqa: E402
from frq_generate import generate_frq  # noqa: E402
from frq_scenarios import FRQ_BANKS, FRQBankNotFound, bank_path  # noqa: E402
from mcq_scoring import score_mcq  # noqa: E402
from models_registry import (  # noqa: E402
    estimate_weight_gb,
    known_context_window,
    load_models,
    select_models,
)
from run_benchmark import _frq_overrides, make_engine, register_extra_banks  # noqa: E402

PREVIEW = 240  # chars of model output shown inline; full text goes to the JSON
PROMPT_PREVIEW = 400


# ---------------------------------------------------------------------------
# report structures
# ---------------------------------------------------------------------------


@dataclass
class Stage:
    """One timed step. ``note`` carries cache / degradation detail."""

    name: str
    seconds: float = 0.0
    ok: bool = True
    note: str = ""
    error: str = ""


@dataclass
class ModelReport:
    model_id: str
    params_b: float
    declared_backend: str
    effective_backend: str = ""
    degraded: bool = False
    static_window: int | None = None
    resolved_window: int | None = None
    stages: list[Stage] = field(default_factory=list)
    mcq_rows: int = 0
    mcq_correct: int = 0
    frq_rows: int = 0
    frq_issues: int = 0
    frq_truncated: int = 0
    frq_hit_budget: int = 0
    frq_empty: int = 0
    outputs: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    cumulative_seconds: float = 0.0  # wall clock at this model's completion

    @property
    def total_seconds(self) -> float:
        return sum(s.seconds for s in self.stages)

    def stage_seconds(self, *prefixes: str) -> float:
        return sum(s.seconds for s in self.stages if s.name.startswith(prefixes))


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------


def _pkg_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def preflight(backend: str, need_frq: bool, need_hf_datasets: bool) -> list[tuple[str, bool, str]]:
    """Environment checks. A False here aborts the run (unless --dry-run)."""
    checks: list[tuple[str, bool, str]] = []
    weights = backend != "mock"

    py = sys.version_info
    checks.append(
        ("python >= 3.11", py >= (3, 11), f"{py.major}.{py.minor}.{py.micro} / {platform.system()}")
    )

    for pkg, required in (
        ("pyyaml", True),
        ("datasets", need_hf_datasets),
        ("transformers", weights),
        ("torch", weights),
        ("huggingface-hub", weights),
        ("vllm", False),
        ("truststore", False),
        ("python-dotenv", False),
    ):
        ver = _pkg_version(pkg)
        detail = ver or ("MISSING" + ("" if required else "  (not needed for this run)"))
        checks.append((f"package {pkg}", bool(ver) or not required, detail))

    gpu, ok_gpu = "not checked (mock backend)", True
    if weights:
        try:
            import torch

            if torch.cuda.is_available():
                names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
                gpu = f"{len(names)} device(s): {', '.join(names)}"
            else:
                # vLLM requires CUDA; transformers can limp along on CPU.
                gpu = "no CUDA device"
                ok_gpu = backend == "hf"
        except Exception as exc:  # noqa: BLE001
            gpu, ok_gpu = f"torch import failed: {exc!r}", False
    checks.append(("CUDA availability", ok_gpu, gpu))

    # FRQ items come from the sibling eduLLM-Evals repo, not HuggingFace.
    if need_frq:
        checks.append(("eduLLM-Evals root", EDULLM_ROOT.is_dir(), str(EDULLM_ROOT)))
        missing = [k for k in FRQ_BANKS if not bank_path(k).exists()]
        checks.append(
            (
                "FRQ scenario banks",
                not missing,
                f"all {len(FRQ_BANKS)} present" if not missing else f"MISSING: {', '.join(missing)}",
            )
        )

    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    checks.append(
        ("HF_TOKEN", True, f"set ({len(tok)} chars)" if tok else "not set - gated repos will 401")
    )

    if weights:
        try:
            from huggingface_hub.constants import HF_HUB_CACHE
            import shutil

            free = shutil.disk_usage(_existing_ancestor(Path(HF_HUB_CACHE))).free / 1e9
            checks.append(("disk free (HF cache)", free > 20, f"{free:.0f} GB at {HF_HUB_CACHE}"))
        except Exception as exc:  # noqa: BLE001
            checks.append(("disk free (HF cache)", True, f"unknown: {exc!r}"))
    return checks


def _existing_ancestor(p: Path) -> Path:
    while not p.exists() and p != p.parent:
        p = p.parent
    return p


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------


def pick_models(specs: list, n: int, seed: int, max_params_b: float | None, only: str | None):
    if only:
        chosen = select_models(specs, only=[o.strip() for o in only.split(",") if o.strip()])
        if not chosen:
            raise SystemExit(
                f"--models matched nothing in the roster: {only}\n"
                f"  ids must appear in the roster verbatim (or by trailing name); "
                f"list them with: python models_registry.py"
            )
        return chosen
    pool = [s for s in specs if max_params_b is None or s.params_b <= max_params_b]
    if not pool:
        raise SystemExit(f"no models in the roster with params_b <= {max_params_b}")
    return random.Random(seed).sample(pool, min(n, len(pool)))


def pick_items(
    names: list[str], n: int, seed: int, cap: int, use_cache: bool
) -> tuple[list[tuple[str, list]], list[str]]:
    """Spread ``n`` items round-robin across ``names``, so several benchmarks are
    exercised rather than n items from one. Returns (per-benchmark items, notes)."""
    rng = random.Random(seed)
    loaded: dict[str, list] = {}
    notes: list[str] = []
    if n <= 0:
        return [], notes
    for name in names:
        try:
            items = load_items(name, cap, seed, use_cache=use_cache)
            if items:
                loaded[name] = items
            else:
                notes.append(f"{name}: loaded 0 items")
        except FRQBankNotFound as exc:
            notes.append(f"ALERT {name}: FRQ bank missing -> {exc}")
        except Exception as exc:  # noqa: BLE001
            notes.append(f"{name}: dropped, {type(exc).__name__}: {exc}")
    if not loaded:
        return [], notes

    order = sorted(loaded)
    remaining = {k: rng.sample(range(len(v)), len(v)) for k, v in loaded.items()}
    picked: dict[str, list] = {k: [] for k in order}
    total = 0
    while total < n and any(remaining[k] for k in order):
        for name in order:
            if total >= n:
                break
            if remaining[name]:
                picked[name].append(loaded[name][remaining[name].pop()])
                total += 1
    return [(k, v) for k, v in picked.items() if v], notes


# ---------------------------------------------------------------------------
# timing helpers
# ---------------------------------------------------------------------------


def _hf_cache_bytes() -> int:
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        root = Path(HF_HUB_CACHE)
        if not root.exists():
            return 0
        return sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
    except Exception:
        return 0


def time_download(model_id: str, revision: str | None) -> Stage:
    """Pre-fetch weights so download time is separable from load time. A cached
    model returns in seconds and is reported as cached."""
    st = Stage("download")
    before = _hf_cache_bytes()
    t0 = time.monotonic()
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            model_id,
            revision=revision,
            # Never pull duplicate weight formats; the loaders want safetensors/bin.
            ignore_patterns=["*.h5", "*.msgpack", "*.onnx*", "*.tflite", "*.gguf"],
        )
        st.seconds = time.monotonic() - t0
        delta = max(0, _hf_cache_bytes() - before)
        st.note = (
            f"already cached (verified in {st.seconds:.1f}s)"
            if delta < 1_000_000
            else f"downloaded {delta / 1e9:.2f} GB"
        )
    except Exception as exc:  # noqa: BLE001
        st.seconds = time.monotonic() - t0
        st.ok = False
        st.error = f"{type(exc).__name__}: {exc}"
    return st


# ---------------------------------------------------------------------------
# per-model run
# ---------------------------------------------------------------------------


def run_one_model(
    spec,
    mcq_sets: list[tuple[str, list]],
    frq_sets: list[tuple[str, list]],
    cfg: InferenceConfig,
    args: argparse.Namespace,
) -> ModelReport:
    rep = ModelReport(model_id=spec.id, params_b=spec.params_b, declared_backend=spec.backend)
    rep.static_window = known_context_window(spec.id)

    if cfg.backend != "mock" and not args.skip_download_timing:
        st = time_download(spec.id, spec.revision)
        rep.stages.append(st)
        if not st.ok:
            rep.errors.append(f"download failed: {st.error}")
            return rep

    # --- load: capability routing, retries, smoke generation, vLLM -> hf fallback
    st = Stage("load")
    t0 = time.monotonic()
    try:
        engine = make_engine(spec, cfg)
    except Exception as exc:  # noqa: BLE001
        st.seconds = time.monotonic() - t0
        st.ok = False
        st.error = f"{type(exc).__name__}: {exc}"
        rep.stages.append(st)
        rep.errors.append(f"load failed: {st.error}")
        return rep
    st.seconds = time.monotonic() - t0
    rep.stages.append(st)

    # build_engine resolved the window onto the spec; compare with what routing wanted.
    rep.resolved_window = spec.max_model_len
    rep.effective_backend = engine.backend
    rep.degraded = engine.backend != probe_backend(spec, cfg.backend)
    st.note = f"backend={engine.backend}, max_model_len={spec.max_model_len}" + (
        "   *** DEGRADED ***" if rep.degraded else ""
    )
    if rep.degraded:
        rep.errors.append(
            f"degraded: requested {probe_backend(spec, cfg.backend)}, ran on {engine.backend}"
        )

    try:
        for name, items in mcq_sets:
            _run_mcq(engine, spec, name, items, cfg, rep)
        for name, items in frq_sets:
            _run_frq(engine, spec, name, items, cfg, args, rep)
    finally:
        st = Stage("close")
        t0 = time.monotonic()
        try:
            engine.close()
        except Exception as exc:  # noqa: BLE001
            st.ok = False
            st.error = repr(exc)
        st.seconds = time.monotonic() - t0
        rep.stages.append(st)
    return rep


def _run_mcq(engine, spec, name: str, items: list, cfg: InferenceConfig, rep: ModelReport) -> None:
    st = Stage(f"mcq:{name}")
    t0 = time.monotonic()
    try:
        n = score_mcq(engine, spec, name, items, cfg.writer)
        st.seconds = time.monotonic() - t0
        rep.mcq_rows += n
        st.note = f"{n} question(s) scored"
        for row in _read_mcq(name, spec.id):
            if row.get("result") == "correct":
                rep.mcq_correct += 1
            rep.outputs.append(
                {
                    "type": "mcq",
                    "benchmark": name,
                    "item": row.get("question_id"),
                    "predicted": row.get("predicted"),
                    "gold": row.get("gold"),
                    "result": row.get("result"),
                    "scoring_method": row.get("scoring_method"),
                }
            )
    except Exception as exc:  # noqa: BLE001 - one benchmark must not kill the model
        st.seconds = time.monotonic() - t0
        st.ok = False
        st.error = f"{type(exc).__name__}: {exc}"
        rep.errors.append(f"mcq {name}: {st.error}")
    rep.stages.append(st)


def _run_frq(
    engine, spec, name: str, items: list, cfg: InferenceConfig,
    args: argparse.Namespace, rep: ModelReport,
) -> None:
    st = Stage(f"frq:{name}")
    overrides = _frq_overrides(cfg, name)
    if args.max_new_tokens > 0:
        overrides["max_new_tokens"] = args.max_new_tokens
    t0 = time.monotonic()
    try:
        res = generate_frq(engine, spec, name, items, cfg.writer, overrides=overrides)
        st.seconds = time.monotonic() - t0
        rep.frq_rows += res.written
        st.note = f"{res.written} generated, complete={res.complete}, status={res.status}"
    except Exception as exc:  # noqa: BLE001
        st.seconds = time.monotonic() - t0
        st.ok = False
        st.error = f"{type(exc).__name__}: {exc}"
        rep.errors.append(f"frq {name}: {st.error}")
    rep.stages.append(st)

    # Response-level problems: Issue cells, empty text, truncated prompt, budget hit.
    for row in _read_frq(name, spec.id):
        out = row.get("Output") or ""
        if row.get("Issue"):
            rep.frq_issues += 1
            rep.errors.append(f"frq {name} {row.get('Scenario')}: {row.get('Issue Description')}")
        elif not out.strip():
            rep.frq_empty += 1
            rep.errors.append(f"frq {name} {row.get('Scenario')}: empty output (Issue=0)")
        if row.get("Truncated"):
            rep.frq_truncated += 1
        if row.get("Finish Reason") == "length":
            rep.frq_hit_budget += 1
        gp = row.get("Generation Params") or {}
        rep.outputs.append(
            {
                "type": "frq",
                "benchmark": row.get("Benchmark"),
                "item": row.get("Scenario"),
                "chat_template_applied": row.get("Chat Template Applied"),
                "max_model_len": row.get("Max Model Len"),
                "prompt_tokens": row.get("Prompt Tokens"),
                "output_tokens": row.get("Output Tokens"),
                "max_new_tokens": gp.get("max_new_tokens"),
                "finish_reason": row.get("Finish Reason"),
                "truncated": row.get("Truncated"),
                "latency_s": row.get("Latency (s)"),
                "issue": row.get("Issue"),
                "issue_description": row.get("Issue Description"),
                "rendered_prompt": (row.get("Rendered Prompt") or "")[:2000],
                "output": out,
            }
        )


def _read_mcq(benchmark: str, model_id: str) -> list[dict]:
    p = common.mcq_output_path(benchmark, model_id)
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_frq(benchmark: str, model_id: str) -> list[dict]:
    p = common.open_responses_path(benchmark, model_id)
    if not p.exists():
        return []
    rows: list[dict] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def _flat(text: str, limit: int) -> str:
    return " ".join((text or "").split())[:limit]


def print_report(
    reports: list[ModelReport], wall: float, item_load_seconds: float,
    run_dir: Path, notes: list[str],
) -> None:
    bar = "=" * 78
    print(f"\n{bar}\nSMOKE TEST RESULTS\n{bar}")

    for r in reports:
        status = "OK" if not r.errors else f"{len(r.errors)} PROBLEM(S)"
        print(f"\n{'-' * 78}\n{r.model_id}   {r.params_b}B   [{status}]")
        print(
            f"  backend  declared={r.declared_backend}  effective={r.effective_backend or '-'}"
            + ("   *** DEGRADED vllm -> hf ***" if r.degraded else "")
        )
        print(f"  context  resolved={r.resolved_window}  static_table={r.static_window}")
        print("  timings")
        for s in r.stages:
            flag = " " if s.ok else "!"
            print(f"   {flag} {s.name:20} {s.seconds:9.2f}s   {s.note}".rstrip())
            if s.error:
                print(f"       error: {s.error}")
        print(
            f"  model total {r.total_seconds:.2f}s   |   cumulative at finish "
            f"{r.cumulative_seconds:.2f}s"
        )
        print(
            f"  results  mcq {r.mcq_correct}/{r.mcq_rows} correct   |   "
            f"frq {r.frq_rows} rows, {r.frq_issues} issue, {r.frq_empty} empty, "
            f"{r.frq_truncated} prompt-truncated, {r.frq_hit_budget} hit-budget"
        )

        first_frq = True
        for o in r.outputs:
            if o["type"] == "mcq":
                print(
                    f"    [mcq  {o['benchmark']}/{o['item']}] "
                    f"predicted={o['predicted']} gold={o['gold']} -> {o['result']}"
                )
                continue
            print(
                f"    [frq  {o['benchmark']}/{o['item']}] chat_template={o['chat_template_applied']} "
                f"prompt_tok={o['prompt_tokens']}/{o['max_model_len']} "
                f"out_tok={o['output_tokens']}/{o['max_new_tokens']} "
                f"finish={o['finish_reason']}"
                + ("  PROMPT-TRUNCATED" if o["truncated"] else "")
                + (f"  ISSUE: {o['issue_description']}" if o["issue"] else "")
            )
            if first_frq:  # one rendered prompt per model, to eyeball prompt fidelity
                print(f"      prompt> {_flat(o['rendered_prompt'], PROMPT_PREVIEW)}")
                first_frq = False
            print(f"      output> {_flat(o['output'], PREVIEW) or '(EMPTY)'}")

    print(f"\n{bar}\nSUMMARY\n{bar}")
    clean = [r for r in reports if not r.errors]
    print(f"models clean            : {len(clean)}/{len(reports)}")
    print(f"benchmark item load     : {item_load_seconds:9.2f}s  (before the models; not in the total below)")
    print(f"download total          : {sum(r.stage_seconds('download') for r in reports):9.2f}s")
    print(f"load total              : {sum(r.stage_seconds('load') for r in reports):9.2f}s")
    print(f"inference total         : {sum(r.stage_seconds('mcq:', 'frq:') for r in reports):9.2f}s")
    print(f"teardown total          : {sum(r.stage_seconds('close') for r in reports):9.2f}s")
    print(f"WALL CLOCK (cumulative) : {wall:9.2f}s  ({wall / 60:.1f} min)")

    slowest = sorted(reports, key=lambda r: -r.total_seconds)[:3]
    if slowest:
        print(
            "slowest models          : "
            + ", ".join(f"{r.model_id} ({r.total_seconds:.0f}s)" for r in slowest)
        )
    degraded = [r.model_id for r in reports if r.degraded]
    failed = [r.model_id for r in reports if any(not s.ok for s in r.stages)]
    responses = [r for r in reports if r.frq_issues or r.frq_empty]
    if degraded:
        print(f"ALERT degraded to hf    : {', '.join(degraded)}")
    if failed:
        print(f"ALERT stage failures    : {', '.join(failed)}")
    if responses:
        print(
            "ALERT bad responses     : "
            + ", ".join(f"{r.model_id} ({r.frq_issues} issue, {r.frq_empty} empty)" for r in responses)
        )
    if not (degraded or failed or responses):
        print("no alerts.")
    for n in notes:
        print(f"note: {n}")
    print(f"\nartifacts: {run_dir}")


def write_json_report(
    path: Path, reports: list[ModelReport], wall: float, item_load_seconds: float,
    args: argparse.Namespace, cfg: InferenceConfig, roster: Path,
    checks: list[tuple[str, bool, str]], notes: list[str],
) -> None:
    payload = {
        "run_id": path.parent.name,
        "host": {"platform": platform.platform(), "python": sys.version.split()[0]},
        "backend": cfg.backend,
        "roster": str(roster),
        "seed": args.seed,
        "max_new_tokens_override": args.max_new_tokens or None,
        "wall_seconds": round(wall, 2),
        "preflight": [{"check": c, "ok": o, "detail": d} for c, o, d in checks],
        "notes": notes,
        "totals": {
            "item_load_s": round(item_load_seconds, 2),
            "download_s": round(sum(r.stage_seconds("download") for r in reports), 2),
            "load_s": round(sum(r.stage_seconds("load") for r in reports), 2),
            "inference_s": round(sum(r.stage_seconds("mcq:", "frq:") for r in reports), 2),
            "models_clean": sum(1 for r in reports if not r.errors),
            "models_total": len(reports),
        },
        "models": [
            {
                "model_id": r.model_id,
                "params_b": r.params_b,
                "declared_backend": r.declared_backend,
                "effective_backend": r.effective_backend,
                "degraded": r.degraded,
                "resolved_max_model_len": r.resolved_window,
                "static_table_window": r.static_window,
                "mcq_rows": r.mcq_rows,
                "mcq_correct": r.mcq_correct,
                "frq_rows": r.frq_rows,
                "frq_issues": r.frq_issues,
                "frq_empty": r.frq_empty,
                "frq_truncated": r.frq_truncated,
                "frq_hit_budget": r.frq_hit_budget,
                "total_seconds": round(r.total_seconds, 2),
                "cumulative_seconds": round(r.cumulative_seconds, 2),
                "stages": [
                    {
                        "name": s.name,
                        "seconds": round(s.seconds, 3),
                        "ok": s.ok,
                        "note": s.note,
                        "error": s.error,
                    }
                    for s in r.stages
                ],
                "outputs": r.outputs,
                "errors": r.errors,
            }
            for r in reports
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--num-models", type=int, default=5, help="how many models to sample (5)")
    ap.add_argument("--num-mcq", type=int, default=5, help="MCQ questions total, spread over banks")
    ap.add_argument("--num-frq", type=int, default=5, help="FRQ scenarios total, spread over banks")
    ap.add_argument("--models", default=None,
                    help="comma list of ids (or trailing names) to pin instead of sampling")
    ap.add_argument("--max-params-b", type=float, default=None,
                    help="only sample models at or below this size; keeps the download small")
    ap.add_argument("--seed", type=int, default=0,
                    help="selection seed - same seed picks the same models and items")
    ap.add_argument("--backend", choices=["vllm", "hf", "mock"], default=None,
                    help="default: inference.yaml (vllm). 'mock' needs no weights and no GPU")
    ap.add_argument("--models-yaml", default=None,
                    help="roster (default: Inputs/Models/models_200.yaml, else models.yaml)")
    ap.add_argument("--frq-banks", action="append", default=None, metavar="PATH",
                    help="YAML file registering extra FRQ scenario banks (repeatable)")
    ap.add_argument("--max-new-tokens", type=int, default=256,
                    help="cap the FRQ generation budget so the test finishes quickly; "
                         "0 = use the manifest/config value (4096)")
    ap.add_argument("--item-cap", type=int, default=200,
                    help="items loaded per benchmark before sampling from them")
    ap.add_argument("--synthetic", action="store_true",
                    help="use the offline synth_mcq/synth_open banks (no dataset download)")
    ap.add_argument("--skip-download-timing", action="store_true",
                    help="let the engine fetch weights instead of pre-fetching them separately")
    ap.add_argument("--no-cache", action="store_true", help="ignore the normalized MCQ cache")
    ap.add_argument("--dry-run", action="store_true", help="preflight + plan, then exit")
    ap.add_argument("--inference-config", default=None)
    return ap


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    args = build_parser().parse_args(argv)
    cfg = InferenceConfig.load(Path(args.inference_config) if args.inference_config else None)
    if args.backend:
        cfg.backend = args.backend
    extra_banks = register_extra_banks(args.frq_banks) if args.frq_banks else []

    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = common.OUTPUTS_DIR / "_smoke" / run_id
    # Redirect every output path. common.*_path() read these module globals at call
    # time, so this keeps the real Outputs/ trees and .done markers untouched.
    common.MCQ_OUT_DIR = run_dir / "mcq"
    common.OPEN_OUT_DIR = run_dir / "open"
    common.MANIFEST_DIR = run_dir / "_manifests"
    common.ensure_dirs(common.MCQ_OUT_DIR, common.OPEN_OUT_DIR, common.MANIFEST_DIR)

    mcq_names = ["synth_mcq"] if args.synthetic else MCQ_BENCHMARKS
    frq_names = ["synth_open"] if args.synthetic else OPEN_BENCHMARKS

    print(f"smoke test {run_id}   backend={cfg.backend}   seed={args.seed}")
    print(f"output dir: {run_dir}")
    if extra_banks:
        print(f"registered FRQ bank(s): {', '.join(extra_banks)}")
    print()
    print("PREFLIGHT")
    checks = preflight(
        cfg.backend,
        need_frq=bool(args.num_frq) and not args.synthetic,
        need_hf_datasets=bool(args.num_mcq) and not args.synthetic,
    )
    for name, ok, detail in checks:
        print(f"  [{'ok  ' if ok else 'FAIL'}] {name:24} {detail}")
    failed_checks = [name for name, ok, _ in checks if not ok]
    if failed_checks:
        print(f"\npreflight FAILED: {', '.join(failed_checks)}")
        print("SMOKE_TEST.md documents how to satisfy each requirement.")
        if not args.dry_run:
            return 1

    roster = Path(args.models_yaml) if args.models_yaml else None
    if roster is None:
        preferred = common.INPUTS_DIR / "Models" / "models_200.yaml"
        roster = preferred if preferred.exists() else common.MODELS_YAML
    specs = load_models(roster)
    chosen = pick_models(specs, args.num_models, args.seed, args.max_params_b, args.models)

    # First run downloads the MCQ datasets from HuggingFace, which can take a
    # while; announce it so the wait doesn't look like a hang.
    use_cache = not args.no_cache
    print("\nloading benchmark items (MCQ datasets download on first use) ...", flush=True)
    t_items = time.monotonic()
    mcq_sets, n1 = pick_items(mcq_names, args.num_mcq, args.seed, args.item_cap, use_cache)
    frq_sets, n2 = pick_items(frq_names, args.num_frq, args.seed, args.item_cap, use_cache)
    item_load_seconds = time.monotonic() - t_items
    notes = n1 + n2

    print(f"\nPLAN   roster={roster.name} ({len(specs)} models)")
    print(f"  items loaded in {item_load_seconds:.1f}s")
    for s in chosen:
        print(
            f"  model  {s.id:50} {s.params_b:>6}B  backend={s.backend:11} "
            f"chat={int(s.apply_chat_template)} gated={int(s.gated)}"
        )
    print(f"  MCQ    {sum(len(v) for _, v in mcq_sets)} items from {[k for k, _ in mcq_sets]}")
    print(f"  FRQ    {sum(len(v) for _, v in frq_sets)} items from {[k for k, _ in frq_sets]}")
    if cfg.backend != "mock":
        print(
            f"  weights: ~{sum(estimate_weight_gb(s) for s in chosen):.0f} GB to download "
            f"if nothing is cached"
        )
    if args.max_new_tokens:
        print(f"  FRQ generation budget capped at {args.max_new_tokens} tokens/item")
    for n in notes:
        print(f"  note: {n}")

    if args.dry_run:
        print("\n--dry-run: stopping before any model load.")
        return 1 if failed_checks else 0
    if not mcq_sets and not frq_sets:
        print("\nnothing to run: no benchmark could be loaded (see notes above)")
        return 1

    wall0 = time.monotonic()
    reports: list[ModelReport] = []
    for i, spec in enumerate(chosen, 1):
        print(f"\n[{i}/{len(chosen)}] {spec.id}", flush=True)
        try:
            rep = run_one_model(spec, mcq_sets, frq_sets, cfg, args)
        except Exception:  # noqa: BLE001 - never let one model kill the run
            rep = ModelReport(spec.id, spec.params_b, spec.backend)
            rep.errors.append("unhandled: " + traceback.format_exc(limit=4))
        rep.cumulative_seconds = time.monotonic() - wall0
        reports.append(rep)
        print(
            f"    -> {rep.total_seconds:.1f}s, {len(rep.errors)} problem(s), "
            f"cumulative {rep.cumulative_seconds:.1f}s",
            flush=True,
        )
    wall = time.monotonic() - wall0

    print_report(reports, wall, item_load_seconds, run_dir, notes)
    report_path = run_dir / "smoke_report.json"
    write_json_report(
        report_path, reports, wall, item_load_seconds, args, cfg, roster, checks, notes
    )
    print(f"json report: {report_path}")
    print("Send that file (and the console log) back with the results.")

    return 1 if any(r.errors for r in reports) else 0


if __name__ == "__main__":
    sys.exit(main())
