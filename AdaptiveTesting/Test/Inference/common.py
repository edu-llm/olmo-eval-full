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
