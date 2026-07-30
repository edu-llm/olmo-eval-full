"""Model manifest loading, flag derivation, context-window resolution, sharding."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from common import MODELS_YAML

# Orgs whose weights require accepted licenses + HF_TOKEN.
GATED_ORGS = {"meta-llama", "google", "mistralai"}

# Substrings that indicate an instruction/chat-tuned checkpoint. Matched against
# the lowercased id, so capitalization variants ("-Instruct", "Text-Instruct")
# are covered. Being liberal here is safe: `Engine.render_chat` returns
# applied=False when the tokenizer ships no chat_template, so a false positive
# degrades to the flat Student:/Tutor: rendering rather than erroring.
# The alignment-suffix and family entries below were added after a roster audit
# found ~17 instruct/aligned models silently rendering as base models.
CHAT_MARKERS = (
    "instruct", "-it", "chat", "zephyr", "vicuna", "sft",
    # alignment / post-training suffixes
    "dpo", "orpo", "rlhf", "kto", "distill", "reasoning",
    # instruct-tuned families whose ids carry no generic marker
    "hermes", "alpaca", "dolly", "lamini", "magicoder", "kullm", "gpt-jt",
)

# Architectures the pinned vLLM build may not support -> HF fallback.
HF_FALLBACK_MARKERS = ("mamba", "openelm", "gemma-3")


@dataclass
class ModelSpec:
    id: str
    params_b: float
    dtype: str = "bfloat16"
    trust_remote_code: bool = True
    # None => resolve at load time to min(max_model_len_cap, the model's own
    # declared window). An explicit value is a hard request, still clamped by
    # what the checkpoint actually supports. See resolve_max_model_len.
    max_model_len: int | None = None
    tp: int = 1
    apply_chat_template: bool = False
    scoring_method: str = "loglikelihood"
    gated: bool = False
    backend: str = "vllm"  # vllm | hf_fallback

    # --- FRQ generation knobs (respgen parity) -----------------------------
    # MCQ scoring ignores these; FRQ generation records them per output row.
    # `max_model_len` remains authoritative for the engine; the cap is the
    # ceiling the auto-resolved window is clamped to.
    max_model_len_cap: int = 32768
    max_new_tokens: int = 4096
    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: float = 1.1
    seed: int = 0
    enable_thinking: bool | None = None  # Qwen3: False suppresses <think> traces
    tokenizer_id: str | None = None      # borrow another repo's tokenizer (OpenELM -> Llama-2)
    revision: str | None = None          # pin a commit SHA; None => resolve at load

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


# ---------------------------------------------------------------------------
# Context-window resolution
# ---------------------------------------------------------------------------

# Used when a model's real window is unknown (config unreadable AND no table
# entry): small enough to load anywhere, never a gamble.
DEFAULT_MAX_MODEL_LEN = 4096

# config.json keys that state a positional limit, most standard first.
# "max_context_length" is OpenELM's key.
_WINDOW_KEYS = (
    "max_position_embeddings",
    "n_positions",
    "max_seq_len",
    "seq_length",
    "max_context_length",
)

# Offline fallback: declared windows for the manifest's families, used only when
# config.json can't be read (no network / SSL failure / mock runs). Values are
# deliberately CONSERVATIVE - never above what the checkpoint supports - since an
# over-estimate is a hard vLLM init failure. Matched as substrings of the
# lowercased id, first match wins, so specific entries come before general ones.
# ORDER MATTERS: first substring match wins, so a specific entry must precede any
# more general one that would also match it (e.g. "rugpt3" before "gpt2", because
# rugpt3large_based_on_gpt2 is really 2048; "vicuna-7b-v1.5" before "vicuna-7b").
KNOWN_CONTEXT_WINDOWS: tuple[tuple[str, int], ...] = (
    # --- specific-before-general guards ---
    ("rugpt3", 2048),          # ..._based_on_gpt2, but 2048 not 1024
    ("lamini-gpt", 1024),      # GPT-2 based
    ("gpt3-finnish", 2048),
    ("gpt-sw3", 2048),
    ("gpt-jt", 2048),
    ("gpt-j", 2048),
    ("gpt-neox", 2048),
    ("gpt-neo-", 2048),
    ("gpt2", 1024),
    # --- 1024-position models ---
    ("biogpt", 1024),
    ("biomedlm", 1024),
    # --- 2048-position models ---
    ("pythia", 2048),
    ("cerebras-gpt", 2048),
    ("facebook/opt-", 2048),
    ("bloom", 2048),
    ("mpt-7b", 2048),
    ("redpajama-incite", 2048),
    ("llm360/amber", 2048),
    ("tinyllama", 2048),
    ("openelm", 2048),
    ("falcon-7b", 2048),
    ("phi-1", 2048),           # covers phi-1 and phi-1_5
    ("phi-2", 2048),
    ("xglm", 2048),
    ("codegen", 2048),
    ("polyglot-ko", 2048),
    ("open-calm", 2048),
    ("opencalm", 2048),
    ("open_llama", 2048),
    ("openllama", 2048),
    ("dolly-v2", 2048),
    ("santacoder", 2048),
    ("replit-code", 2048),
    ("kogpt", 2048),
    ("mobilellm", 2048),
    ("camel-5b", 2048),
    ("palmyra", 2048),
    ("aquila", 2048),
    ("mgpt", 2048),
    ("japanese-large-lm", 2048),
    ("japanese-gpt", 2048),
    # --- families where a v1/v2 split changes the window ---
    ("smollm3", 32768),
    ("smollm2", 8192),
    ("smollm-", 2048),         # SmolLM v1 is 2048, NOT v2's 8192
    ("vicuna-7b-v1.5", 4096),
    ("vicuna-7b", 2048),       # v1.1
    ("olmo-2", 4096),
    ("olmo-1b", 2048),
    ("olmo-7b", 2048),
    # --- 4k+ models ---
    ("phi-3-mini-4k", 4096),
    ("phi-3.5-mini", 32768),
    ("qwen2.5", 32768),
    ("qwen2-", 32768),
    ("qwen3", 32768),
    ("llama-3.2", 32768),
    ("gemma-2-", 8192),
    ("gemma-3-", 32768),
    ("mistral-7b", 32768),
    ("zephyr-7b", 32768),
    ("stablelm", 4096),
    ("falcon3", 8192),
    ("granite-3.1", 32768),
    ("granite-3.3", 32768),
    ("granite-3.0", 4096),
    ("danube", 4096),
    ("yi-6b", 4096),
    ("internlm2", 32768),
    ("minicpm", 4096),
    ("deepseek", 4096),
    ("nemotron-mini", 4096),
    ("mamba", 4096),  # SSM: no positional limit, keep it modest
)


def _window_from_config(cfg: dict[str, Any]) -> int | None:
    for key in _WINDOW_KEYS:
        val = cfg.get(key)
        if val:
            return int(val)
    text_cfg = cfg.get("text_config")  # multimodal configs nest the LM here
    if isinstance(text_cfg, dict):
        return _window_from_config(text_cfg)
    return None


def _fetch_hub_config(model_id: str) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download  # lazy

    path = hf_hub_download(model_id, "config.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# Latched once the Hub is proven unreachable (no network, TLS interception without
# the corporate CA, offline box). Without this every model re-attempts the read and
# huggingface_hub retries ~5x per call, so a 200-model roster burns a long time
# failing silently and then resolves everything from the static table anyway.
_HUB_UNREACHABLE = False

# Error text that means "the transport is broken", i.e. no id will ever resolve -
# as opposed to a per-model 404/401, which says nothing about the next model.
# NOTE: huggingface_hub >=1.x swallows the TLS detail - after its internal retries
# it raises a bare `RuntimeError: Cannot send a request, as the client has been
# closed`, and the certificate text only reaches stderr via its retry logger. So
# that phrasing must be matched too, or the latch never fires and a 200-model
# roster retry-storms in silence (which is exactly what happened).
_TRANSPORT_ERRORS = (
    "certificate_verify_failed",
    "sslerror",
    "ssl:",
    "client has been closed",
    "max retries exceeded",
    "connectionerror",
    "connecterror",
    "failed to establish a new connection",
    "name or service not known",
    "temporary failure in name resolution",
    "offline mode",
)


def _exception_text(exc: BaseException) -> str:
    """The exception and its whole cause/context chain, lowercased. The useful
    signal is often on a wrapped cause rather than the exception that escaped."""
    parts: list[str] = []
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        parts.append(f"{type(cur).__name__}: {cur}")
        cur = cur.__cause__ or cur.__context__
    return " | ".join(parts).lower()


def _note_hub_failure(model_id: str, exc: Exception) -> None:
    """Latch + warn ONCE when the failure is a transport problem rather than a
    per-model error, so the remaining models skip the Hub instead of re-failing."""
    global _HUB_UNREACHABLE
    if _HUB_UNREACHABLE:
        return
    text = _exception_text(exc)
    if not any(marker in text for marker in _TRANSPORT_ERRORS):
        return
    _HUB_UNREACHABLE = True
    print(
        f"warning: cannot reach the HuggingFace Hub while resolving context windows "
        f"({model_id}: {type(exc).__name__}). Falling back to the static "
        f"KNOWN_CONTEXT_WINDOWS table for all models; a model missing from that table "
        f"resolves to DEFAULT_MAX_MODEL_LEN={DEFAULT_MAX_MODEL_LEN}.",
        file=sys.stderr,
        flush=True,
    )


def known_context_window(model_id: str) -> int | None:
    low = model_id.lower()
    for marker, window in KNOWN_CONTEXT_WINDOWS:
        if marker in low:
            return window
    return None


def declared_context_window(
    model_id: str, fetch_config: Callable[[str], dict[str, Any]] | None = None
) -> int | None:
    """The model's own declared window from config.json, else the static table,
    else None. The Hub read is the authority - the table only covers offline.

    Skips the Hub entirely once a transport failure has been latched, so an
    unreachable Hub costs one warning instead of 200 silent retry storms.
    """
    fetch = fetch_config or _fetch_hub_config
    found = None
    if not (_HUB_UNREACHABLE and fetch_config is None):
        try:
            found = _window_from_config(fetch(model_id))
        except Exception as exc:  # noqa: BLE001 - fall back to the table below
            if fetch_config is None:
                _note_hub_failure(model_id, exc)
    return found or known_context_window(model_id)


def resolve_max_model_len(
    spec: ModelSpec,
    *,
    offline: bool = False,
    fetch_config: Callable[[str], dict[str, Any]] | None = None,
) -> int:
    """The context window to actually run this model with.

    ``min(request, declared, cap)``: the manifest states a request (or None =>
    use the cap), and it is clamped by what the checkpoint declares, so a big
    default window never exceeds e.g. GPT-2's 1024 positions. FRQ needs this to
    be generous - TutorEval prompts reach ~10k tokens, and anything above the
    window is truncated away before the model ever sees it.

    ``offline`` skips the Hub read and uses the static table only.
    """
    declared = known_context_window(spec.id) if offline else declared_context_window(
        spec.id, fetch_config
    )
    cap = int(spec.max_model_len_cap or 32768)
    requested = int(spec.max_model_len) if spec.max_model_len else cap
    if declared is not None:
        return max(1, min(requested, declared, cap))
    # Nothing authoritative: honor an explicit request, otherwise stay small.
    return requested if spec.max_model_len else DEFAULT_MAX_MODEL_LEN


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
            max_model_len=_coalesce(entry, defaults, "max_model_len", None),
            tp=_coalesce(entry, defaults, "tp", 1),
            apply_chat_template=bool(chat),
            scoring_method=_coalesce(entry, defaults, "scoring_method", "loglikelihood"),
            gated=entry.get("gated", _derive_gated(mid)),
            backend=entry.get("backend", _derive_backend(mid)),
            max_model_len_cap=int(_coalesce(entry, defaults, "max_model_len_cap", 32768)),
            max_new_tokens=int(_coalesce(entry, defaults, "max_new_tokens", 4096)),
            temperature=float(_coalesce(entry, defaults, "temperature", 0.0)),
            top_p=float(_coalesce(entry, defaults, "top_p", 1.0)),
            repetition_penalty=float(_coalesce(entry, defaults, "repetition_penalty", 1.1)),
            seed=int(_coalesce(entry, defaults, "seed", 0)),
            enable_thinking=_coalesce(entry, defaults, "enable_thinking", None),
            tokenizer_id=_coalesce(entry, defaults, "tokenizer_id", None),
            revision=_coalesce(entry, defaults, "revision", None),
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
