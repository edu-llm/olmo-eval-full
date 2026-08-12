#!/usr/bin/env python3
"""Load the three-checkpoint manifest and resolve early / chinchilla / final roles."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import steering_common as sc


@dataclass
class ThreeCheckpointPlan:
    run_name: str
    early_uri: str
    chinchilla_uri: str
    final_uri: str
    early_step: int
    chinchilla_step: int
    final_step: int
    tokens_per_step: int
    chinchilla_target_tokens: int
    max_statements: int
    eval_limit: int
    toxigen_limit: int
    mmlu_subjects: int
    mmlu_limit_per_subject: int
    ppl_limit: int
    steering_layers: list[int] | None
    steering_alphas: list[float]
    model_family: str | None = None
    load_options: sc.LoadModelOptions | None = None

    @property
    def probing_uris(self) -> list[str]:
        return [self.early_uri, self.chinchilla_uri, self.final_uri]

    @property
    def steering_sources(self) -> dict[str, str]:
        return {"early": self.early_uri, "chinchilla": self.chinchilla_uri}


def _step_from_uri(uri: str) -> int | None:
    match = re.search(r"step(\d+)", uri)
    return int(match.group(1)) if match else None


def _training_schedule(config: dict[str, Any]) -> dict[str, Any]:
    loader = config.get("data_loader") or {}
    batch = int(loader.get("global_batch_size") or 0)
    if not batch:
        raise SystemExit("config.json missing data_loader.global_batch_size")
    trainer = config.get("trainer") or {}
    callbacks = trainer.get("callbacks") or {}
    ckpt_cb = callbacks.get("checkpointer") or {}
    fixed = [int(x) for x in ckpt_cb.get("fixed_steps") or []]
    max_dur = trainer.get("max_duration") or {}
    max_tokens = int(max_dur.get("value") or 0)
    final_step = int(max_tokens / batch) if max_tokens else (max(fixed) if fixed else 0)
    model = config.get("model") or {}
    d_model = int(model.get("d_model") or 0)
    n_layers = int(model.get("n_layers") or 0)
    non_emb = d_model * d_model * n_layers * 12 if d_model and n_layers else 7_000_000_000
    return {
        "tokens_per_step": batch,
        "fixed_steps": sorted(fixed),
        "final_step": final_step,
        "max_tokens": max_tokens,
        "non_embedding_params_est": non_emb,
    }


def _hf_schedule_fallback(raw: dict[str, Any], final_uri: str) -> dict[str, Any]:
    non_emb = int(raw.get("non_embedding_params") or 29_900_000_000)
    final_step = int(raw.get("final_step") or _step_from_uri(final_uri) or 500_000)
    return {
        "tokens_per_step": int(raw.get("tokens_per_step") or 4_194_304),
        "fixed_steps": [],
        "final_step": final_step,
        "non_embedding_params_est": non_emb,
    }


def _resolve_schedule(
    raw: dict[str, Any],
    final_uri: str,
    *,
    aws_region: str,
    s3_endpoint: str | None,
    dry_run: bool,
) -> dict[str, Any]:
    if dry_run:
        return {
            "tokens_per_step": 4_194_304,
            "fixed_steps": [10_000, 200_000, 500_000],
            "final_step": int(raw.get("final_step") or _step_from_uri(final_uri) or 500_000),
            "non_embedding_params_est": int(raw.get("non_embedding_params") or 29_900_000_000),
        }

    family = str(raw.get("model_family") or "")
    if family and not family.startswith("olmo"):
        return _hf_schedule_fallback(raw, final_uri)

    config = sc.fetch_run_config(final_uri, aws_region, s3_endpoint)
    if isinstance(config.get("data_loader"), dict):
        return _training_schedule(config)
    if raw.get("final_step"):
        return _hf_schedule_fallback(raw, final_uri)
    return _training_schedule(config)


def _pick_chinchilla_step(
    fixed: list[int], final_step: int, tokens_per_step: int, target_tokens: int
) -> int:
    candidates = fixed or [max(1, final_step // 2)]
    return min(candidates, key=lambda s: abs(s * tokens_per_step - target_tokens))


def load_manifest_raw(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def manifest_checkpoint_uri(raw: dict[str, Any], role: str = "final") -> str:
    uri = raw["checkpoints"][role].rstrip("/") + "/"
    if "REPLACE_WITH_YOUR_STAGED_URI" in uri:
        raise SystemExit(
            f"Manifest checkpoint {role!r} is still a placeholder. "
            f"Edit {raw.get('run_name', 'manifest')} with your staged URI."
        )
    return uri


def load_manifest(
    path: Path,
    *,
    aws_region: str = "us-east-1",
    s3_endpoint: str | None = None,
    dry_run: bool = False,
) -> ThreeCheckpointPlan:
    raw = load_manifest_raw(path)
    ckpts = raw["checkpoints"]
    final_uri = ckpts["final"].rstrip("/") + "/"

    schedule = _resolve_schedule(
        raw, final_uri, aws_region=aws_region, s3_endpoint=s3_endpoint, dry_run=dry_run
    )

    tokens_per_step = schedule["tokens_per_step"]
    final_step = raw.get("final_step") or _step_from_uri(final_uri) or schedule["final_step"]
    if not final_step:
        raise SystemExit("Could not determine final_step; set it in the manifest.")

    mult = float(raw.get("chinchilla_multiplier") or 20)
    chinchilla_tokens = raw.get("chinchilla_tokens")
    if chinchilla_tokens is None:
        chinchilla_tokens = int(mult * schedule["non_embedding_params_est"])

    fixed = schedule["fixed_steps"]
    early_step = raw.get("early_step") or (fixed[0] if fixed else max(1, final_step // 20))
    chinchilla_step = raw.get("chinchilla_step") or _pick_chinchilla_step(
        fixed, final_step, tokens_per_step, int(chinchilla_tokens)
    )

    early_uri = ckpts.get("early", "").rstrip("/") + "/" if ckpts.get("early") else final_uri
    chin_uri = ckpts.get("chinchilla", "").rstrip("/") + "/"
    chinchilla_uri = chin_uri if ckpts.get("chinchilla") else final_uri

    return ThreeCheckpointPlan(
        run_name=str(raw.get("run_name") or "tracingllm"),
        early_uri=early_uri,
        chinchilla_uri=chinchilla_uri,
        final_uri=final_uri,
        early_step=int(early_step),
        chinchilla_step=int(chinchilla_step),
        final_step=int(final_step),
        tokens_per_step=tokens_per_step,
        chinchilla_target_tokens=int(chinchilla_tokens),
        max_statements=int(raw.get("max_statements") or 1000),
        eval_limit=int(raw.get("eval_limit") or 0),
        toxigen_limit=int(raw.get("toxigen_limit") or 300),
        mmlu_subjects=int(raw.get("mmlu_subjects") or 8),
        mmlu_limit_per_subject=int(raw.get("mmlu_limit_per_subject") or 50),
        ppl_limit=int(raw.get("ppl_limit") or 200),
        steering_layers=raw.get("steering_layers"),
        steering_alphas=[float(x) for x in raw.get("steering_alphas") or [-4, -2, -1, 1, 2, 4]],
        model_family=raw.get("model_family"),
        load_options=sc.load_options_from_manifest(raw),
    )


def plan_dict(plan: ThreeCheckpointPlan) -> dict[str, Any]:
    return {
        "run_name": plan.run_name,
        "early": {"uri": plan.early_uri, "step": plan.early_step},
        "chinchilla": {
            "uri": plan.chinchilla_uri,
            "step": plan.chinchilla_step,
            "target_tokens": plan.chinchilla_target_tokens,
        },
        "final": {"uri": plan.final_uri, "step": plan.final_step},
        "tokens_per_step": plan.tokens_per_step,
        "steering_sources": plan.steering_sources,
        "steering_target": plan.final_uri,
        "model_family": plan.model_family,
        "load": {
            "dtype": plan.load_options.dtype if plan.load_options else "auto",
            "device_map": plan.load_options.device_map if plan.load_options else None,
        },
    }
