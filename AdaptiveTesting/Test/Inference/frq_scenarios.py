"""FRQ item source: the eduLLM-Evals ``scenarios.jsonl`` banks.

FRQ items are NOT loaded from HuggingFace here. The banks under
``eduLLM-Evals/data/<Benchmark>/scenarios.jsonl`` are the only source that
carries the fields faithful prompt construction needs - ``use_case``,
``conversation_context`` and (for BiGGen) a native per-instance
``system_prompt`` - all of which a flat HF loader throws away.

If a bank is missing this module raises :class:`FRQBankNotFound` with the exact
expected path, and the driver surfaces it as a loud ALERT rather than silently
skipping the benchmark.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from common import EDULLM_ROOT


class FRQBankNotFound(FileNotFoundError):
    """An FRQ benchmark's scenarios.jsonl is not present locally."""


@dataclass
class Scenario:
    """One FRQ item, mirroring the eduLLM-Evals Scenario schema (text modality).

    ``benchmark`` is the canonical label (e.g. "TutorBench", "BiGGen"); it drives
    the per-benchmark system-prompt selection and the ``Benchmark`` value written
    into every output row.
    """

    scenario_id: str
    prompt: str
    use_case: str = ""
    subject: str = ""
    grade_band: str = ""
    modality: str = "text"
    conversation_context: list[dict[str, str]] = field(default_factory=list)
    reference_solution: str = ""
    criterion_ids: list[str] = field(default_factory=list)
    system_prompt: str = ""
    benchmark: str = ""

    @classmethod
    def from_json(cls, obj: dict[str, Any]) -> "Scenario":
        return cls(
            scenario_id=obj["scenario_id"],
            prompt=obj["prompt"],
            use_case=obj.get("use_case", "") or "",
            subject=obj.get("subject", "") or "",
            grade_band=obj.get("grade_band", "") or "",
            modality=obj.get("modality", "text") or "text",
            conversation_context=obj.get("conversation_context") or [],
            reference_solution=obj.get("reference_solution", "") or "",
            criterion_ids=list(obj.get("criterion_ids") or []),
            system_prompt=obj.get("system_prompt") or "",
            benchmark=obj.get("benchmark", "") or "",
        )

    # `qid` keeps the FRQ path interchangeable with the MCQ Question in driver code.
    @property
    def qid(self) -> str:
        return self.scenario_id


@dataclass
class FRQBank:
    """Registry entry: CLI key -> canonical label + bank path + judge rubric."""

    key: str
    label: str
    path: str  # relative to the eduLLM-Evals repo root
    rubric_key: str = "default"


# The active FRQ set. EduBench and IFEval are intentionally excluded; their
# prompt rules live on in frq_prompts._NO_SYSTEM_BENCHMARKS should they return.
FRQ_BANKS: dict[str, FRQBank] = {
    "tutorbench": FRQBank("tutorbench", "TutorBench", "data/TutorBench/scenarios.jsonl", "tutorbench"),
    "tutoreval": FRQBank("tutoreval", "TutorEval", "data/TutorEval/scenarios.jsonl", "tutoreval"),
    "bridge": FRQBank("bridge", "Bridge", "data/Bridge/scenarios.jsonl", "default"),
    "biggen": FRQBank("biggen", "BiGGen", "data/BiGGen/scenarios.jsonl", "default"),
    "infobench": FRQBank("infobench", "InFoBench", "data/InFoBench/scenarios.jsonl", "default"),
    "wildbench": FRQBank("wildbench", "WildBench", "data/WildBench/scenarios.jsonl", "default"),
}


def bank_path(key: str) -> Path:
    if key not in FRQ_BANKS:
        raise KeyError(f"unknown FRQ benchmark {key!r}; known: {sorted(FRQ_BANKS)}")
    return EDULLM_ROOT / FRQ_BANKS[key].path


def _cap(items: list[Scenario], max_samples: int | None, seed: int) -> list[Scenario]:
    """Deterministic subsample, matching datasets_registry._cap so --max-samples
    behaves identically for MCQ and FRQ."""
    if max_samples is None or len(items) <= max_samples:
        return items
    rng = random.Random(seed)
    idx = list(range(len(items)))
    rng.shuffle(idx)
    idx = sorted(idx[:max_samples])
    return [items[i] for i in idx]


def load_frq_scenarios(
    key: str, max_samples: int | None = None, seed: int = 1234
) -> list[Scenario]:
    """Load one FRQ bank as Scenarios (text modality only), stamping the canonical
    benchmark label on each. The label is authoritative and overrides any
    ``benchmark`` already present in the row."""
    spec = FRQ_BANKS[key] if key in FRQ_BANKS else None
    if spec is None:
        raise KeyError(f"unknown FRQ benchmark {key!r}; known: {sorted(FRQ_BANKS)}")
    path = EDULLM_ROOT / spec.path
    if not path.exists():
        raise FRQBankNotFound(
            f"FRQ bank for '{key}' ({spec.label}) not found at: {path}\n"
            f"        (eduLLM-Evals root resolved to: {EDULLM_ROOT})\n"
            f"        Set EDULLM_EVALS_ROOT to override the repo location."
        )
    out: list[Scenario] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if (obj.get("modality") or "text") != "text":
                continue
            scn = Scenario.from_json(obj)
            scn.benchmark = spec.label
            out.append(scn)
    return _cap(out, max_samples, seed)
