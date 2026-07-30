"""The FRQ "Model Output" record (ported from eduLLM-Evals ``respgen/records.py``).

One row per (model, scenario). Field names are the PRD's exact Title-Case keys
and are the contract the downstream judge / Q-matrix / IRT stages read, so they
must not drift.

Note what is deliberately absent: there is no predicted choice, no logprob and
no correctness column. An FRQ item's correctness is not computable at generation
time - it is decided later by the judge - so this record stores the raw generated
text plus everything needed to audit how it was produced.

Pure module (no torch/vllm) so the schema is testable anywhere.
"""

from __future__ import annotations

from typing import Any

BENCHMARK = "TutorBench"

# The nested "Generation Params" block. temperature/top_p/seed pin decoding to a
# deterministic, construct-valid setting; repetition_penalty (kept low + uniform)
# still changes the argmax at temperature 0, so it is recorded per row.
def generation_params(
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    repetition_penalty: float,
    seed: int,
) -> dict[str, Any]:
    return {
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": repetition_penalty,
        "seed": seed,
    }


def build_record(
    *,
    scenario_id: str,
    model_id: str,
    model_revision: str | None,
    chat_template_applied: bool,
    rendered_prompt: str,
    gen_params: dict[str, Any],
    max_model_len: int | None,
    prompt_tokens: int,
    output_tokens: int,
    finish_reason: str,
    truncated: bool,
    latency_s: float | None,
    output: str,
    issue: bool = False,
    issue_description: str = "",
    benchmark: str = BENCHMARK,
) -> dict[str, Any]:
    """Assemble one Model Output row with the exact PRD Title-Case keys."""
    if not issue and not issue_description:
        issue_description = "N/A"  # schema: "<desc> if there was an issue, N/A otherwise"
    return {
        "Benchmark": benchmark or BENCHMARK,
        "Scenario": scenario_id,
        "Model": model_id,
        "Model Revision": model_revision or "",
        "Chat Template Applied": int(bool(chat_template_applied)),
        "Rendered Prompt": rendered_prompt,
        "Generation Params": gen_params,
        "Max Model Len": max_model_len,
        "Prompt Tokens": prompt_tokens,
        "Output Tokens": output_tokens,
        "Finish Reason": finish_reason,
        "Truncated": int(bool(truncated)),
        "Latency (s)": latency_s,
        "Output": output,
        "Issue": int(bool(issue)),
        "Issue Description": issue_description,
    }


def error_record(
    *,
    scenario_id: str,
    model_id: str,
    model_revision: str | None = "",
    rendered_prompt: str = "",
    gen_params: dict[str, Any] | None = None,
    max_model_len: int | None = None,
    description: str = "",
    benchmark: str = BENCHMARK,
) -> dict[str, Any]:
    """A failure cell (load error, generation error) so the matrix still has an
    entry for this (model, scenario). Issue=1, Finish Reason='error', empty Output.

    These rows are intentionally NOT treated as done on resume (see frq_shard),
    so a transient failure regenerates on the next run.
    """
    return build_record(
        scenario_id=scenario_id,
        model_id=model_id,
        model_revision=model_revision,
        chat_template_applied=False,
        rendered_prompt=rendered_prompt,
        gen_params=gen_params or {},
        max_model_len=max_model_len,
        prompt_tokens=0,
        output_tokens=0,
        finish_reason="error",
        truncated=False,
        latency_s=None,
        output="",
        issue=True,
        issue_description=description,
        benchmark=benchmark,
    )
