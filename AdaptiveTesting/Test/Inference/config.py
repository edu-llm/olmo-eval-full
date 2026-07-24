"""Config loading for the inference sweep (inference.yaml / judge.yaml)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from common import CONFIG_DIR, RUBRIC_DIR


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


@dataclass
class WriterConfig:
    fsync_every_rows: int = 50
    fsync_every_seconds: float = 5.0


@dataclass
class InferenceConfig:
    max_samples: int | None = 2000
    sample_seed: int = 1234
    backend: str = "vllm"
    vllm: dict[str, Any] = field(default_factory=dict)
    generation: dict[str, Any] = field(default_factory=dict)
    writer: WriterConfig = field(default_factory=WriterConfig)
    overrides: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | None = None) -> InferenceConfig:
        path = path or (CONFIG_DIR / "inference.yaml")
        raw = _load_yaml(path)
        writer = WriterConfig(**(raw.get("writer") or {}))
        return cls(
            max_samples=raw.get("max_samples", 2000),
            sample_seed=raw.get("sample_seed", 1234),
            backend=raw.get("backend", "vllm"),
            vllm=raw.get("vllm") or {},
            generation=raw.get("generation") or {},
            writer=writer,
            overrides=raw.get("overrides") or {},
        )

    def for_benchmark(self, name: str) -> dict[str, Any]:
        """Return effective knobs for a benchmark (global + per-bench override)."""
        eff: dict[str, Any] = {
            "max_samples": self.max_samples,
            "sample_seed": self.sample_seed,
            **{k: v for k, v in self.generation.items()},
        }
        eff.update(self.overrides.get(name, {}))
        return eff


@dataclass
class JudgeConfig:
    model: str = "prometheus-eval/prometheus-7b-v2.0"
    backend: str = "vllm"
    scale_min: int = 1
    scale_max: int = 5
    pass_threshold: int = 4
    max_tokens: int = 512
    temperature: float = 0.0
    rubric_dir: Path = RUBRIC_DIR
    default_rubric: str = "default.rubric.txt"

    @classmethod
    def load(cls, path: Path | None = None) -> JudgeConfig:
        path = path or (CONFIG_DIR / "judge.yaml")
        raw = _load_yaml(path)
        rubric_dir = raw.get("rubric_dir")
        if rubric_dir:
            rd = Path(rubric_dir)
            if not rd.is_absolute():
                rd = (path.parent / rd).resolve()
        else:
            rd = RUBRIC_DIR
        return cls(
            model=raw.get("model", cls.model),
            backend=raw.get("backend", "vllm"),
            scale_min=raw.get("scale_min", 1),
            scale_max=raw.get("scale_max", 5),
            pass_threshold=raw.get("pass_threshold", 4),
            max_tokens=raw.get("max_tokens", 512),
            temperature=raw.get("temperature", 0.0),
            rubric_dir=rd,
            default_rubric=raw.get("default_rubric", "default.rubric.txt"),
        )
