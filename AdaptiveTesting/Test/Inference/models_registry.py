"""Model manifest loading, flag derivation, and sharding."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from common import MODELS_YAML

# Orgs whose weights require accepted licenses + HF_TOKEN.
GATED_ORGS = {"meta-llama", "google", "mistralai"}

# Substrings that indicate an instruction/chat-tuned checkpoint.
CHAT_MARKERS = ("instruct", "-it", "chat", "zephyr", "vicuna", "sft")

# Architectures the pinned vLLM build may not support -> HF fallback.
HF_FALLBACK_MARKERS = ("mamba", "openelm", "gemma-3")


@dataclass
class ModelSpec:
    id: str
    params_b: float
    dtype: str = "bfloat16"
    trust_remote_code: bool = True
    max_model_len: int = 4096
    tp: int = 1
    apply_chat_template: bool = False
    scoring_method: str = "loglikelihood"
    gated: bool = False
    backend: str = "vllm"  # vllm | hf_fallback

    @property
    def slug(self) -> str:
        return self.id.replace("/", "__")

    @property
    def org(self) -> str:
        return self.id.split("/")[0] if "/" in self.id else ""


def _derive_chat(model_id: str) -> bool:
    low = model_id.lower()
    return any(m in low for m in CHAT_MARKERS)


def _derive_gated(model_id: str) -> bool:
    return model_id.split("/")[0] in GATED_ORGS


def _derive_backend(model_id: str) -> str:
    low = model_id.lower()
    return "hf_fallback" if any(m in low for m in HF_FALLBACK_MARKERS) else "vllm"


def _coalesce(entry: dict[str, Any], defaults: dict[str, Any], key: str, fallback: Any) -> Any:
    if key in entry and entry[key] is not None:
        return entry[key]
    if key in defaults and defaults[key] is not None:
        return defaults[key]
    return fallback


def load_models(path: Path | None = None) -> list[ModelSpec]:
    path = path or MODELS_YAML
    with open(path) as f:
        raw = yaml.safe_load(f)
    defaults = raw.get("defaults", {}) or {}
    specs: list[ModelSpec] = []
    for entry in raw["models"]:
        mid = entry["id"]
        chat = entry.get("apply_chat_template")
        if chat is None:
            chat = defaults.get("apply_chat_template")
        if chat is None:
            chat = _derive_chat(mid)
        spec = ModelSpec(
            id=mid,
            params_b=float(entry["params_b"]),
            dtype=_coalesce(entry, defaults, "dtype", "bfloat16"),
            trust_remote_code=_coalesce(entry, defaults, "trust_remote_code", True),
            max_model_len=_coalesce(entry, defaults, "max_model_len", 4096),
            tp=_coalesce(entry, defaults, "tp", 1),
            apply_chat_template=bool(chat),
            scoring_method=_coalesce(entry, defaults, "scoring_method", "loglikelihood"),
            gated=entry.get("gated", _derive_gated(mid)),
            backend=entry.get("backend", _derive_backend(mid)),
        )
        specs.append(spec)
    return specs


def select_models(
    specs: list[ModelSpec],
    only: list[str] | None = None,
    shard_index: int | None = None,
    num_shards: int | None = None,
) -> list[ModelSpec]:
    """Filter by explicit ids (full or trailing name) then apply round-robin sharding."""
    if only:
        wanted = {o.lower() for o in only}
        specs = [
            s for s in specs if s.id.lower() in wanted or s.id.split("/")[-1].lower() in wanted
        ]
    if shard_index is not None and num_shards:
        specs = [s for i, s in enumerate(specs) if i % num_shards == shard_index]
    return specs


def estimate_weight_gb(spec: ModelSpec) -> float:
    """Approx bf16 weight footprint in GB (2 bytes/param)."""
    return spec.params_b * 2.0


def _main() -> None:
    ap = argparse.ArgumentParser(description="Inspect the model manifest.")
    ap.add_argument("--count", action="store_true", help="print number of models")
    ap.add_argument("--gated", action="store_true", help="list gated models")
    ap.add_argument("--chat", action="store_true", help="list chat-templated models")
    ap.add_argument("--fallback", action="store_true", help="list hf_fallback models")
    args = ap.parse_args()
    specs = load_models()
    if args.count:
        print(f"{len(specs)} models")
    if args.gated:
        for s in specs:
            if s.gated:
                print(s.id)
    if args.chat:
        for s in specs:
            if s.apply_chat_template:
                print(s.id)
    if args.fallback:
        for s in specs:
            if s.backend == "hf_fallback":
                print(s.id)
    if not any([args.count, args.gated, args.chat, args.fallback]):
        for s in specs:
            print(
                f"{s.id:48s} {s.params_b:>5}B chat={int(s.apply_chat_template)} "
                f"gated={int(s.gated)} backend={s.backend}"
            )


if __name__ == "__main__":
    _main()
