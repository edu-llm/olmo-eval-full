"""FRQ (free-response) generation: render faithful tutor prompts, generate, and
durably store the Model Output schema.

Ported from eduLLM-Evals ``respgen/runner.py`` and adapted to this repo's
resident-engine model. What it preserves from respgen, and why:

  * **Prompt fidelity** - the full multi-turn message list (per-benchmark or
    per-use_case system turn + conversation history, coalesced to alternate) via
    :mod:`frq_prompts`, with a flat ``Student:``/``Tutor:`` fallback for base
    models with no chat template.
  * **Per-item generation budget** - the prompt is fit to
    ``max_model_len - MIN_GEN`` keeping the TAIL (the student's latest turn), then
    each item decodes with ``min(max_new_tokens, max_model_len - prompt_tokens)``,
    never below MIN_GEN. A single global ``max_tokens`` would either truncate long
    prompts to nothing or leave no room to answer.
  * **Failure isolation** - a generation error writes an Issue row for every
    outstanding scenario, so the (model, scenario) matrix is never left with holes.
  * **Validity-aware resume** - see :mod:`frq_shard`; Issue rows regenerate.

Responses are persisted here and never judged inline: a judge failure can't lose
generations (run ``judge_all.py`` afterwards).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import frq_prompts as P
import frq_records as R
from common import open_responses_path
from config import WriterConfig
from engine import Engine, GenParams
from frq_shard import rewrite_shard, scan_shard
from models_registry import ModelSpec
from results_writer import JsonlResultWriter

# Always leave at least this many tokens for the model to answer. The prompt is
# truncated to (max_model_len - MIN_GEN) first; generation then takes whatever
# context is left.
MIN_GEN = 256


@dataclass
class FRQResult:
    written: int
    complete: bool  # every scenario now has a valid row (safe to mark pair done)
    status: str = "ok"


def _resolve_revision(model_id: str, override: str | None) -> str:
    """Pin the exact commit for provenance. Empty string if the Hub is
    unreachable (recorded as-is; the default branch is then used at load)."""
    if override:
        return override
    try:
        from huggingface_hub import HfApi  # lazy

        return HfApi().model_info(model_id).sha or ""
    except Exception:
        return ""


def _render_prompt(engine: Engine, spec: ModelSpec, scenario) -> tuple[str, bool]:
    """(rendered prompt, chat_template_applied). Apply the chat template only if
    the model is chat-tuned AND its tokenizer actually ships one; otherwise fall
    back to the flat base rendering."""
    messages = P.build_chat_messages(scenario)
    text, applied = engine.render_chat(messages, getattr(spec, "enable_thinking", None))
    if applied:
        return text, True
    return P.render_base_prompt(scenario), False


def _fit_prompt_and_budget(
    text: str, tokenizer, max_model_len: int, max_new_tokens: int
) -> tuple[str, int, bool, int]:
    """Fit the prompt into the context window and size this item's generation
    budget. Returns (text, prompt_tokens, truncated, gen_budget).

    Without a tokenizer (mock backend) tokens are estimated at ~4 chars each and
    the same window math is applied, so a smoke test exercises fitting and
    budgeting rather than silently bypassing them.
    """
    prompt_cap = max(1, max_model_len - MIN_GEN)
    if tokenizer is None:
        prompt_tokens = max(1, len(text) // 4)
        truncated = prompt_tokens > prompt_cap
        if truncated:
            text = text[-prompt_cap * 4 :]
            prompt_tokens = max(1, len(text) // 4)
    else:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        truncated = len(ids) > prompt_cap
        if truncated:
            ids = ids[-prompt_cap:]  # keep the most recent turn (the student's latest prompt)
            text = tokenizer.decode(ids, skip_special_tokens=False)
        prompt_tokens = len(ids)
    gen_budget = min(max_new_tokens, max(MIN_GEN, max_model_len - prompt_tokens))
    gen_budget = max(1, min(gen_budget, max_model_len - 1))
    return text, prompt_tokens, truncated, gen_budget


def _count_output_tokens(tokenizer, text: str) -> int:
    if tokenizer is None:
        return len(text.split())
    try:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])
    except Exception:
        return len(text.split())


def generate_frq(
    engine: Engine,
    spec: ModelSpec,
    benchmark_key: str,
    scenarios: list,
    writer_cfg: WriterConfig,
    *,
    resume: bool = True,
    overrides: dict[str, Any] | None = None,
) -> FRQResult:
    """Generate + persist FRQ responses for one (benchmark, model) pair.

    Decoding defaults come from the model manifest (respgen parity); ``overrides``
    (inference.yaml ``generation`` + per-benchmark ``overrides``) can adjust
    max_new_tokens / temperature / top_p / seed / repetition_penalty.
    """
    ov = overrides or {}

    def _eff(name: str, default):
        val = ov.get(name)
        if val is None:
            val = getattr(spec, name, None)
        return default if val is None else val
    out_path = open_responses_path(benchmark_key, spec.id)
    label = getattr(scenarios[0], "benchmark", "") if scenarios else ""
    label = label or R.BENCHMARK

    # Validity-aware resume: only scenarios with a valid row count as done, and
    # any Issue / corrupted / duplicate rows are compacted out first so the shard
    # ends with exactly one clean row per scenario.
    done: set[str] = set()
    if resume:
        done, valid_rows, had_invalid = scan_shard(out_path)
        if had_invalid:
            rewrite_shard(out_path, valid_rows)
    else:
        rewrite_shard(out_path, [])

    todo = [s for s in scenarios if s.scenario_id not in done]
    if not todo:
        return FRQResult(written=0, complete=True, status="already_complete")

    # The engine was built with spec.max_model_len; the cap can only lower the
    # window we fit prompts into (never raise it past what the engine allocated).
    max_model_len = min(
        int(getattr(spec, "max_model_len", 4096) or 4096),
        int(getattr(spec, "max_model_len_cap", 32768) or 32768),
    )
    max_new_tokens = int(_eff("max_new_tokens", 4096))
    revision = _resolve_revision(spec.id, getattr(spec, "revision", None))
    tokenizer = engine.tokenizer

    # --- render + fit every prompt, then batch-generate ---
    rendered: list[str] = []
    applied: list[bool] = []
    prompt_tokens: list[int] = []
    truncated: list[bool] = []
    budgets: list[int] = []
    for s in todo:
        text, ct = _render_prompt(engine, spec, s)
        text, ptok, trunc, gbud = _fit_prompt_and_budget(
            text, tokenizer, max_model_len, max_new_tokens
        )
        rendered.append(text)
        applied.append(ct)
        prompt_tokens.append(ptok)
        truncated.append(trunc)
        budgets.append(gbud)

    params = GenParams(
        temperature=float(_eff("temperature", 0.0)),
        top_p=float(_eff("top_p", 1.0)),
        max_tokens=max_new_tokens,
        seed=int(_eff("seed", 0)),
        repetition_penalty=float(_eff("repetition_penalty", 1.1)),
    )

    t0 = time.monotonic()
    gen_error: str | None = None
    outputs: list[str] | None
    try:
        outputs = engine.generate(
            rendered, params, pre_rendered=True, max_tokens_per_prompt=budgets
        )
    except Exception as exc:  # noqa: BLE001 - isolate to Issue cells, never lose the pair
        outputs = None
        gen_error = repr(exc)
    elapsed = time.monotonic() - t0
    avg_latency = round(elapsed / len(todo), 4) if todo else None

    written = 0
    with JsonlResultWriter(
        out_path,
        fsync_every_rows=writer_cfg.fsync_every_rows,
        fsync_every_seconds=writer_cfg.fsync_every_seconds,
    ) as w:
        for i, s in enumerate(todo):
            gp = R.generation_params(
                params.temperature, params.top_p, budgets[i],
                params.repetition_penalty, params.seed,
            )
            if outputs is None:
                rec = R.error_record(
                    scenario_id=s.scenario_id,
                    model_id=spec.id,
                    model_revision=revision,
                    rendered_prompt=rendered[i],
                    gen_params=gp,
                    max_model_len=max_model_len,
                    description=f"generation failed: {gen_error}",
                    benchmark=label,
                )
            else:
                text = outputs[i]
                n_out = _count_output_tokens(tokenizer, text)
                rec = R.build_record(
                    scenario_id=s.scenario_id,
                    model_id=spec.id,
                    model_revision=revision,
                    chat_template_applied=applied[i],
                    rendered_prompt=rendered[i],
                    gen_params=gp,
                    max_model_len=max_model_len,
                    prompt_tokens=prompt_tokens[i],
                    output_tokens=n_out,
                    finish_reason="length" if n_out >= budgets[i] else "stop",
                    truncated=truncated[i],
                    latency_s=avg_latency,
                    output=text,
                    benchmark=label,
                )
            w.write_row(rec)
            written += 1

    return FRQResult(
        written=written,
        complete=outputs is not None,
        status="ok" if outputs is not None else "generation_failed",
    )


def dry_run(spec: ModelSpec, scenarios: list, n: int = 3) -> str:
    """Show the message list per benchmark WITHOUT loading a model, so the
    per-benchmark system-prompt selection (e.g. InFoBench omits the system turn)
    is visible at a glance."""
    lines: list[str] = []
    shown: dict[str, int] = {}
    for s in scenarios:
        b = getattr(s, "benchmark", "") or R.BENCHMARK
        if shown.get(b, 0) >= n:
            continue
        shown[b] = shown.get(b, 0) + 1
        msgs = P.build_chat_messages(s)
        roles = [m["role"] for m in msgs]
        lines.append(f"\n-- [{b}] {s.scenario_id} ({s.use_case or '-'}) roles={roles}")
        for m in msgs:
            preview = " ".join(m["content"].split())[:200]
            lines.append(f"   [{m['role']}] {preview}")
    return "\n".join(lines)
