"""Run frontier (hosted API) models over the eval benchmarks and save judge-ready
responses.

This is the API-model counterpart to the open-weight fleet (`tutor-cat generate`,
which loads HuggingFace checkpoints on GPU). It calls the TrueFoundry
OpenAI-compatible gateway for closed models (Claude / GPT / Gemini) and writes one
row per (model, scenario) in the EXACT same Model Output shard format
(`runs/responses/<Benchmark>/<model>.jsonl`), so the existing judge / Q-matrix /
IRT stages consume the results with no changes.

Design (mirrors the fleet so responses stay comparable):
  * Prompts come from ``tutor_cat.respgen.prompts.build_chat_messages`` — the same
    per-benchmark system prompts the open-weight fleet uses.
  * Rows come from ``tutor_cat.respgen.records.build_record`` / ``error_record``.
  * Sharding + resume come from ``tutor_cat.respgen.shard``.
  * Benchmarks are selected from ``benchmarks.yaml``; scenarios via
    ``tutor_cat.respgen.runner.load_scenarios``.

Credentials: ``MODEL_API_KEY`` + ``MODEL_API_BASE`` from ``eduLLM-Evals/.env``
(same vars ``scripts/prepare_human_judge_validation.py`` uses). ``MODEL_API_BASE``
is normalized to the gateway's OpenAI inference path when only the host root is
given.

Examples
--------
    # Cost/token estimate, no API calls:
    python scripts/run_frontier_models.py --benchmarks TutorBench,TutorEval --dry-run

    # One cheap model, one scenario, live smoke test:
    python scripts/run_frontier_models.py --benchmarks TutorBench \
        --models gemini-group/gemini-3.5-flash --limit 1

    # Full requested set (6 models) over two benchmarks:
    python scripts/run_frontier_models.py --benchmarks TutorBench,TutorEval
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tutor_cat.respgen import prompts as P  # noqa: E402
from tutor_cat.respgen import records as R  # noqa: E402
from tutor_cat.respgen.benchmarks import load_benchmarks, select_benchmarks  # noqa: E402
from tutor_cat.respgen.runner import load_scenarios  # noqa: E402
from tutor_cat.respgen.shard import (  # noqa: E402
    ShardWriter,
    rewrite_shard,
    scan_shard,
    shard_path,
)

# The six frontier slugs requested for the comparison run. Override with --models.
DEFAULT_MODELS = [
    "claude-group/claude-opus-4-8",
    "claude-group/claude-sonnet-4-6",
    "gemini-group/gemini-3.1-pro",
    "gemini-group/gemini-3.5-flash",
    "openai-group/gpt-5.5",
    "openai-group/gpt-5.6-terra",
]

# The gateway root (MODEL_API_BASE in .env) plus the OpenAI-compatible inference
# path. The working scripts (audit_bridge_*, config.yaml) use the full path; the
# committed .env only has the host, so we append this when it is missing.
DEFAULT_INFERENCE_SUFFIX = "/api/llm/api/inference/openai"

# Public list-price PROXIES ($ per 1M tokens, input, output). The internal gateway
# may bill differently; these are only for a ballpark cost estimate. Unknown slugs
# fall back to DEFAULT_PRICE and are flagged in the summary.
PRICES: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-group/claude-opus-4-6": (15.0, 75.0),
    "claude-group/claude-opus-4-7": (15.0, 75.0),
    "claude-group/claude-opus-4-8": (15.0, 75.0),
    "claude-group/claude-opus-5": (15.0, 75.0),
    "claude-group/claude-sonnet-4-5-20250929": (3.0, 15.0),
    "claude-group/claude-sonnet-4-6": (3.0, 15.0),
    "claude-group/claude-sonnet-5": (3.0, 15.0),
    "claude-group/claude-haiku-4-5": (1.0, 5.0),
    "claude-group/claude-haiku-4-5-20251001": (1.0, 5.0),
    # Google
    "gemini-group/gemini-2.5-pro": (1.25, 10.0),
    "gemini-group/gemini-3.1-pro": (1.25, 10.0),
    "gemini-group/gemini-2.5-flash": (0.30, 2.50),
    "gemini-group/gemini-3.5-flash": (0.30, 2.50),
    "gemini-group/gemini-3.6-flash": (0.30, 2.50),
    "gemini-group/gemini-3.5-flash-lite": (0.10, 0.40),
    # OpenAI
    "openai-group/gpt-4o": (2.5, 10.0),
    "openai-group/gpt-4.1": (2.0, 8.0),
    "openai-group/gpt-4.1-nano": (0.10, 0.40),
    "openai-group/gpt-5": (1.25, 10.0),
    "openai-group/gpt-5-mini": (0.25, 2.0),
    "openai-group/gpt-5.2": (1.25, 10.0),
    "openai-group/gpt-5.4": (1.25, 10.0),
    "openai-group/gpt-5.4-mini": (0.25, 2.0),
    "openai-group/gpt-5.4-nano": (0.10, 0.40),
    "openai-group/gpt-5.5": (1.25, 10.0),
    "openai-group/gpt-5.6-luna": (1.25, 10.0),
    "openai-group/gpt-5.6-sol": (1.25, 10.0),
    "openai-group/gpt-5.6-terra": (1.25, 10.0),
}
DEFAULT_PRICE = (1.25, 10.0)


# ---------------------------------------------------------------------------
# env + endpoint
# ---------------------------------------------------------------------------


def _load_env() -> None:
    """Load eduLLM-Evals/.env so MODEL_API_KEY / MODEL_API_BASE are available."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(ROOT / ".env")


def normalize_base_url(base: str) -> str:
    """Turn the gateway host root into the OpenAI-compatible inference endpoint.

    ``.env`` ships ``MODEL_API_BASE=https://tfy.promptlens.trilogy.com`` (host
    only), while the OpenAI SDK needs the full ``.../api/llm/api/inference/openai``
    path (as used in config.yaml and the audit_bridge_* scripts). If the base
    already points at an inference/openai path we leave it untouched.
    """
    base = base.rstrip("/")
    low = base.lower()
    if "inference" in low or low.endswith("/openai") or low.endswith("/v1"):
        return base
    return base + DEFAULT_INFERENCE_SUFFIX


def estimate_tokens(text: str) -> int:
    """Cheap chars/4 token estimate for --dry-run (no tokenizer/API needed)."""
    return max(1, len(text) // 4)


def cost_usd(model: str, in_tok: int, out_tok: int) -> float:
    p_in, p_out = PRICES.get(model, DEFAULT_PRICE)
    return in_tok / 1_000_000 * p_in + out_tok / 1_000_000 * p_out


# ---------------------------------------------------------------------------
# gateway client
# ---------------------------------------------------------------------------


class CallResult:
    """One completion outcome: text + usage + finish metadata, or an error."""

    __slots__ = (
        "text",
        "prompt_tokens",
        "output_tokens",
        "finish_reason",
        "latency_s",
        "error",
    )

    def __init__(
        self,
        text: str = "",
        prompt_tokens: int = 0,
        output_tokens: int = 0,
        finish_reason: str = "",
        latency_s: float | None = None,
        error: str = "",
    ):
        self.text = text
        self.prompt_tokens = prompt_tokens
        self.output_tokens = output_tokens
        self.finish_reason = finish_reason
        self.latency_s = latency_s
        self.error = error


class GatewayClient:
    """Thin OpenAI-compatible client with retry + backoff.

    Mirrors ``tutor_cat.tutors.OpenAITutor``: uses ``max_completion_tokens`` and,
    on provider push-back, drops ``temperature`` (reasoning models reject an
    explicit temperature) or falls back to ``max_tokens``.
    """

    def __init__(self, base_url: str, api_key: str, timeout: float = 180.0):
        from openai import OpenAI  # lazy so --dry-run needs no SDK

        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    def complete(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float | None,
        max_tokens: int | None,
        max_retries: int = 4,
    ) -> CallResult:
        kwargs: dict = {"model": model, "messages": messages}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_completion_tokens"] = max_tokens

        last_err: Exception | None = None
        for attempt in range(max_retries):
            start = time.monotonic()
            try:
                completion = self._client.chat.completions.create(**kwargs)
                latency = time.monotonic() - start
                choice = completion.choices[0]
                usage = getattr(completion, "usage", None)
                return CallResult(
                    text=choice.message.content or "",
                    prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                    output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                    finish_reason=choice.finish_reason or "",
                    latency_s=latency,
                )
            except Exception as e:  # provider SDK exception types vary
                last_err = e
                msg = str(e)
                # Adapt the request to common gateway quirks, then retry.
                if "temperature" in msg and "temperature" in kwargs:
                    kwargs.pop("temperature")
                    continue
                if "max_completion_tokens" in msg and "max_completion_tokens" in kwargs:
                    kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")
                    continue
                time.sleep(2.0 * (attempt + 1))
        return CallResult(error=f"{type(last_err).__name__}: {last_err}")


# ---------------------------------------------------------------------------
# per-scenario work
# ---------------------------------------------------------------------------


def _rendered_prompt(messages: list[dict[str, str]]) -> str:
    """Store the exact message list sent to the API as the row's Rendered Prompt
    (API models apply their own chat template server-side, so there is no single
    flattened string; the JSON message list is the faithful, inspectable record)."""
    return json.dumps(messages, ensure_ascii=False)


def _record_for(
    client: GatewayClient,
    model: str,
    scenario,
    benchmark: str,
    temperature: float | None,
    max_tokens: int,
) -> dict:
    """Call the gateway for one scenario and return a Model Output row."""
    messages = P.build_chat_messages(scenario)
    rendered = _rendered_prompt(messages)
    gen_params = R.generation_params(
        temperature=temperature if temperature is not None else 0.0,
        top_p=1.0,
        max_new_tokens=max_tokens,
        repetition_penalty=None,  # N/A for hosted APIs; kept for schema parity
        seed=0,
    )
    res = client.complete(model, messages, temperature, max_tokens)
    if res.error:
        return R.error_record(
            scenario_id=scenario.scenario_id,
            model_id=model,
            rendered_prompt=rendered,
            gen_params=gen_params,
            description=res.error,
            benchmark=benchmark,
        )
    return R.build_record(
        scenario_id=scenario.scenario_id,
        model_id=model,
        model_revision="",
        chat_template_applied=True,  # applied server-side by the gateway
        rendered_prompt=rendered,
        gen_params=gen_params,
        max_model_len=None,
        prompt_tokens=res.prompt_tokens,
        output_tokens=res.output_tokens,
        finish_reason=res.finish_reason or "stop",
        truncated=(res.finish_reason == "length"),
        latency_s=res.latency_s,
        output=res.text,
        issue=not res.text.strip(),
        issue_description="empty output" if not res.text.strip() else "",
        benchmark=benchmark,
    )


# ---------------------------------------------------------------------------
# per (model, benchmark) run
# ---------------------------------------------------------------------------


def run_model_benchmark(
    client: GatewayClient,
    model: str,
    benchmark: str,
    scenarios: list,
    out_dir: Path,
    temperature: float | None,
    max_tokens: int,
    concurrency: int,
    resume: bool,
) -> dict:
    """Generate + write one shard for (model, benchmark). Returns usage totals."""
    path = shard_path(out_dir, model, benchmark)

    done: set[str] = set()
    if resume:
        done, valid_rows, had_invalid = scan_shard(path)
        if had_invalid:
            # Compact out Issue/corrupted/duplicate rows so they regenerate clean.
            rewrite_shard(path, valid_rows)
    todo = [s for s in scenarios if s.scenario_id not in done]

    totals = {
        "model": model,
        "benchmark": benchmark,
        "n_total": len(scenarios),
        "n_skipped": len(scenarios) - len(todo),
        "n_done": 0,
        "n_issues": 0,
        "prompt_tokens": 0,
        "output_tokens": 0,
    }
    print(
        f"  [{benchmark}] {model}: {len(todo)} to run "
        f"({totals['n_skipped']} already done)",
        flush=True,
    )
    if not todo:
        return totals

    writer = ShardWriter(path, truncate=not resume)
    try:
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            futures = {
                pool.submit(
                    _record_for,
                    client,
                    model,
                    s,
                    benchmark,
                    temperature,
                    max_tokens,
                ): s
                for s in todo
            }
            for fut in as_completed(futures):
                row = fut.result()
                writer.write(row)
                totals["n_done"] += 1
                totals["prompt_tokens"] += int(row.get("Prompt Tokens", 0) or 0)
                totals["output_tokens"] += int(row.get("Output Tokens", 0) or 0)
                if row.get("Issue", 0) == 1:
                    totals["n_issues"] += 1
                if totals["n_done"] % 25 == 0:
                    print(
                        f"    ... {totals['n_done']}/{len(todo)} "
                        f"({totals['n_issues']} issues)",
                        flush=True,
                    )
    finally:
        writer.close()
    return totals


# ---------------------------------------------------------------------------
# dry run (estimate only)
# ---------------------------------------------------------------------------


def dry_run(
    models: list[str],
    benchmarks: list[tuple[str, list]],
    max_tokens: int,
    assumed_output_tokens: int,
) -> None:
    """Estimate token volume + proxy cost per (model, benchmark) without calling
    the API. Input tokens are chars/4 over the rendered messages; output tokens
    are the smaller of --assumed-output-tokens and --max-tokens."""
    out_per_call = min(assumed_output_tokens, max_tokens)

    # Input tokens are model-independent (same prompts), so compute once.
    per_bench_in: dict[str, int] = {}
    per_bench_n: dict[str, int] = {}
    for name, scenarios in benchmarks:
        tot = 0
        for s in scenarios:
            text = "".join(m["content"] for m in P.build_chat_messages(s))
            tot += estimate_tokens(text)
        per_bench_in[name] = tot
        per_bench_n[name] = len(scenarios)

    print("\n" + "=" * 78)
    print("DRY RUN — token + proxy-cost estimate (no API calls)")
    print("  input tokens = chars/4 over rendered prompts")
    print(f"  output tokens assumed = {out_per_call}/call")
    print("  $ = PUBLIC LIST-PRICE PROXY per model family; gateway may differ")
    print("=" * 78)

    grand = 0.0
    for model in models:
        priced = model in PRICES
        model_in = model_out = 0
        model_cost = 0.0
        print(f"\n{model}{'' if priced else '  [no price -> default proxy]'}")
        for name, _ in benchmarks:
            n = per_bench_n[name]
            in_tok = per_bench_in[name]
            out_tok = n * out_per_call
            c = cost_usd(model, in_tok, out_tok)
            model_in += in_tok
            model_out += out_tok
            model_cost += c
            print(
                f"    {name:<12} {n:>5} calls  "
                f"in={in_tok/1e6:5.2f}M  out~{out_tok/1e6:5.2f}M  ~${c:7.2f}"
            )
        print(
            f"    {'TOTAL':<12} {'':>5}        "
            f"in={model_in/1e6:5.2f}M  out~{model_out/1e6:5.2f}M  ~${model_cost:7.2f}"
        )
        grand += model_cost

    print("\n" + "-" * 78)
    print(f"GRAND TOTAL (all {len(models)} models): ~${grand:,.2f}")
    print(
        "NOTE: reasoning-heavy models can emit 2-3x the assumed output tokens, "
        "raising output-side cost proportionally."
    )
    print("-" * 78)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--models",
        type=str,
        default=None,
        help="comma-separated gateway model slugs (default: the 6 frontier slugs).",
    )
    p.add_argument(
        "--benchmarks",
        type=str,
        default=None,
        help="comma-separated benchmark names from benchmarks.yaml "
        "(default: every enabled benchmark). WildBench is disabled by default; "
        "name it explicitly to include it.",
    )
    p.add_argument(
        "--benchmarks-yaml",
        type=Path,
        default=ROOT / "benchmarks.yaml",
    )
    p.add_argument("--out-dir", type=Path, default=ROOT / "runs" / "responses")
    p.add_argument("--limit", type=int, default=None, help="cap scenarios PER benchmark (smoke).")
    p.add_argument("--max-tokens", type=int, default=4096, help="max completion tokens per call.")
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="decoding temperature (dropped automatically if a model rejects it).",
    )
    p.add_argument("--concurrency", type=int, default=8, help="in-flight requests per model.")
    p.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="override the gateway base URL (else MODEL_API_BASE, normalized).",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="overwrite shards instead of skipping already-generated scenarios.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="estimate tokens + proxy cost without calling the API or writing shards.",
    )
    p.add_argument(
        "--assumed-output-tokens",
        type=int,
        default=800,
        help="assumed output tokens/call for --dry-run cost estimation.",
    )
    args = p.parse_args()

    _load_env()

    models = (
        [m.strip() for m in args.models.split(",") if m.strip()]
        if args.models
        else list(DEFAULT_MODELS)
    )

    only = (
        [b.strip() for b in args.benchmarks.split(",") if b.strip()]
        if args.benchmarks
        else None
    )
    specs = select_benchmarks(load_benchmarks(args.benchmarks_yaml), only=only)

    # Load each benchmark's scenarios once (shared across models).
    benchmarks: list[tuple[str, list]] = []
    for spec in specs:
        scen = load_scenarios(
            ROOT / spec.scenarios, limit=args.limit, benchmark=spec.name
        )
        benchmarks.append((spec.name, scen))

    print("=" * 78)
    print("FRONTIER MODEL RESPONSE RUNNER")
    print("=" * 78)
    print(f"models     : {len(models)} -> {', '.join(models)}")
    print(
        "benchmarks : "
        + ", ".join(f"{n} ({len(s)})" for n, s in benchmarks)
    )
    print(f"out-dir    : {args.out_dir}")
    print(
        f"decoding   : temperature={args.temperature} max_tokens={args.max_tokens} "
        f"concurrency={args.concurrency} resume={not args.no_resume}"
    )

    if args.dry_run:
        dry_run(models, benchmarks, args.max_tokens, args.assumed_output_tokens)
        return 0

    base_url = normalize_base_url(args.base_url or os.environ.get("MODEL_API_BASE", ""))
    api_key = os.environ.get("MODEL_API_KEY", "")
    if not base_url:
        raise SystemExit("MODEL_API_BASE is not set (and no --base-url given).")
    if not api_key:
        raise SystemExit("MODEL_API_KEY is not set in the environment / .env.")
    print(f"gateway    : {base_url}")

    client = GatewayClient(base_url, api_key)

    all_totals: list[dict] = []
    for model in models:
        print(f"\n[model] {model}")
        for name, scen in benchmarks:
            totals = run_model_benchmark(
                client=client,
                model=model,
                benchmark=name,
                scenarios=scen,
                out_dir=args.out_dir,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                concurrency=args.concurrency,
                resume=not args.no_resume,
            )
            all_totals.append(totals)

    # ---- summary + actual-usage cost --------------------------------------
    print("\n" + "=" * 78)
    print("SUMMARY (actual API usage; $ = proxy list-price)")
    print("=" * 78)
    grand_cost = 0.0
    by_model: dict[str, dict] = {}
    for t in all_totals:
        m = by_model.setdefault(
            t["model"], {"in": 0, "out": 0, "issues": 0, "done": 0, "skipped": 0}
        )
        m["in"] += t["prompt_tokens"]
        m["out"] += t["output_tokens"]
        m["issues"] += t["n_issues"]
        m["done"] += t["n_done"]
        m["skipped"] += t["n_skipped"]
    for model, m in by_model.items():
        c = cost_usd(model, m["in"], m["out"])
        grand_cost += c
        print(
            f"{model:<38} done={m['done']:>5} skip={m['skipped']:>5} "
            f"issues={m['issues']:>4}  in={m['in']/1e6:5.2f}M out={m['out']/1e6:5.2f}M "
            f"~${c:8.2f}"
        )
    print("-" * 78)
    print(f"GRAND TOTAL (this run): ~${grand_cost:,.2f}")
    print(f"responses under: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
