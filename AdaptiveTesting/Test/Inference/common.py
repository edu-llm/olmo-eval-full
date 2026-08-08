"""Shared types and path helpers for the AdaptiveTesting inference sweep."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# .../AdaptiveTesting/Test/Inference/common.py -> AdaptiveTesting/
INFERENCE_DIR = Path(__file__).resolve().parent
ADAPTIVE_ROOT = INFERENCE_DIR.parent.parent

# FRQ items come from the sibling eduLLM-Evals repo's scenario banks (they carry
# use_case / conversation_context / native system_prompt, which HF loaders drop).
# Override with EDULLM_EVALS_ROOT when the two repos aren't side by side.
EDULLM_ROOT = Path(
    os.environ.get("EDULLM_EVALS_ROOT") or (ADAPTIVE_ROOT.parent / "eduLLM-Evals")
).resolve()

INPUTS_DIR = ADAPTIVE_ROOT / "Inputs"
MODELS_YAML = INPUTS_DIR / "Models" / "models.yaml"
MCQ_BENCH_DIR = INPUTS_DIR / "MCQ" / "Benchmarks"
OPEN_BENCH_DIR = INPUTS_DIR / "Open" / "Benchmarks"
RUBRIC_DIR = INPUTS_DIR / "Open" / "LLM-Judge"

OUTPUTS_DIR = ADAPTIVE_ROOT / "Outputs"
MCQ_OUT_DIR = OUTPUTS_DIR / "mcq"
OPEN_OUT_DIR = OUTPUTS_DIR / "open"
MANIFEST_DIR = OUTPUTS_DIR / "_manifests"
SUMMARY_DIR = OUTPUTS_DIR / "_summary"

CONFIG_DIR = INFERENCE_DIR / "configs"

# ---------------------------------------------------------------------------
# Environment bootstrap
# ---------------------------------------------------------------------------

# aws/put_hf_secret.sh already reads HF_TOKEN from AdaptiveTesting/.env, so that
# is the canonical location; a .env beside the entry points wins if both exist.
_ENV_FILES = (INFERENCE_DIR / ".env", ADAPTIVE_ROOT / ".env")

_bootstrapped = False


def bootstrap_env() -> None:
    """Install the OS trust store and load ``.env``. Call first in ``main()``.

    Two failures this prevents:

    * corporate networks TLS-intercept with a company root CA that Python's
      bundled certifi list doesn't know, so every Hub read dies with
      ``CERTIFICATE_VERIFY_FAILED`` - context windows silently degrade to
      ``models_registry.KNOWN_CONTEXT_WINDOWS`` and weight downloads fail;
    * an ``HF_TOKEN`` that lives only in ``.env`` is never seen, so gated repos
      401 even with a valid key.

    Ordering matters: ``inject_into_ssl()`` patches ``ssl.SSLContext``, which
    only affects sockets opened afterwards. Every ``huggingface_hub`` / ``vllm``
    / ``datasets`` import in this package is lazy (inside functions), so calling
    this at the top of ``main()`` still precedes the first HTTPS connection.

    Both packages are optional; a missing one degrades to prior behavior.
    """
    global _bootstrapped
    if _bootstrapped:
        return
    _bootstrapped = True

    try:
        import truststore

        truststore.inject_into_ssl()
    except ImportError:
        pass

    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    # override=False (the default) throughout: an already-exported value - the
    # Secrets Manager token in aws/run_parallel.sh - must beat a stale local file.
    load_dotenv()
    for path in _ENV_FILES:
        if path.is_file():
            load_dotenv(path)
    # huggingface_hub < 0.19 and some downstream libs only read the legacy name.
    token = os.environ.get("HF_TOKEN")
    if token and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token


class BenchType(StrEnum):
    MCQ = "mcq"
    OPEN = "open"


def slugify(model_id: str) -> str:
    """HF id -> filesystem-safe slug (``org/name`` -> ``org__name``)."""
    return model_id.replace("/", "__")


# ---------------------------------------------------------------------------
# Unified question record (output of every dataset loader)
# ---------------------------------------------------------------------------


@dataclass
class Question:
    """One normalized evaluation item.

    MCQ items carry ``options`` + ``gold_index``. Open-ended items carry
    ``reference`` (may be empty) and are graded by the judge.
    """

    qid: str
    prompt: str
    options: list[str] | None = None
    gold_index: int | None = None
    reference: str | None = None
    meta: dict = field(default_factory=dict)

    @property
    def is_mcq(self) -> bool:
        return self.options is not None and self.gold_index is not None

    @property
    def gold_letter(self) -> str | None:
        if self.gold_index is None:
            return None
        return chr(ord("A") + self.gold_index)


def letter(index: int) -> str:
    return chr(ord("A") + index)


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def mcq_output_path(benchmark: str, model_id: str) -> Path:
    return MCQ_OUT_DIR / benchmark / f"{slugify(model_id)}.csv"


def open_responses_path(benchmark: str, model_id: str) -> Path:
    return OPEN_OUT_DIR / benchmark / f"{slugify(model_id)}.responses.jsonl"


def open_judged_path(benchmark: str, model_id: str) -> Path:
    return OPEN_OUT_DIR / benchmark / f"{slugify(model_id)}.judged.csv"


def done_marker_path(benchmark: str, model_id: str) -> Path:
    return MANIFEST_DIR / f"{benchmark}__{slugify(model_id)}.done"


def benchmark_done_path(benchmark: str) -> Path:
    return MANIFEST_DIR / f"BENCHMARK__{benchmark}.done"


def is_pair_done(benchmark: str, model_id: str) -> bool:
    return done_marker_path(benchmark, model_id).exists()


def mark_pair_done(benchmark: str, model_id: str, n_items: int) -> None:
    ensure_dirs(MANIFEST_DIR)
    path = done_marker_path(benchmark, model_id)
    path.write_text(f"n_items={n_items}\n")


def env_flag(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}
