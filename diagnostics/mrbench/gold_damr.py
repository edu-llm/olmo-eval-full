"""Parse MRBench V1 and compute gold-label DAMR per tutor and dimension.

DAMR (Desired Annotation Match Rate) for a (tutor, dimension) cell is the
percentage of that tutor's responses whose *gold* human label equals the desired
label for that dimension (see ``reference.DESIRED_LABELS``). This module performs
pure data parsing and arithmetic: no judge, no API calls, no network.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .reference import DESIRED_LABELS, DIMENSIONS, PAPER_TO_DATA_KEY

# MRBench_V1.json ships alongside this module under ``data/``.
DEFAULT_DATA_PATH = Path(__file__).parent / "data" / "MRBench_V1.json"

DOWNLOAD_HINT = (
    "MRBench V1 not found. Download it from the official repo (CC BY-SA 4.0):\n"
    "  curl -sSL "
    "https://raw.githubusercontent.com/kaushal0494/UnifyingAITutorEvaluation/"
    "main/MRBench/MRBench_V1.json \\\n"
    f"    -o {DEFAULT_DATA_PATH}"
)


@dataclass(frozen=True)
class Cell:
    """One (tutor, dimension) DAMR result."""

    matches: int
    total: int

    @property
    def rate(self) -> float:
        """DAMR as a percentage in [0, 100]; 0.0 when the tutor has no responses."""
        return 100.0 * self.matches / self.total if self.total else 0.0


@dataclass(frozen=True)
class DamrTable:
    """Computed DAMR for every paper tutor across all eight dimensions."""

    cells: dict[str, dict[str, Cell]]  # paper_tutor -> dimension -> Cell
    response_totals: dict[str, int]  # paper_tutor -> #responses scored
    missing_tutors: list[str]  # paper tutors whose data key was absent
    n_conversations: int
    n_responses: int
    data_source_counts: dict[str, int]  # e.g. {"MathDial": 139, "Bridge": 53}


def load_conversations(path: str | Path | None = None) -> list[dict]:
    """Load the MRBench V1 conversation list from disk."""
    data_path = Path(path) if path is not None else DEFAULT_DATA_PATH
    if not data_path.exists():
        raise FileNotFoundError(DOWNLOAD_HINT)
    with data_path.open(encoding="utf-8") as fh:
        conversations = json.load(fh)
    if not isinstance(conversations, list):
        raise ValueError(
            f"Expected a JSON list of conversations, got {type(conversations).__name__}."
        )
    return conversations


def _resolve_label(annotation: dict, dimension: str) -> str | None:
    """Return the annotation value for ``dimension``, tolerating key-casing drift.

    The V1 file stores the eighth dimension under the lowercase key
    ``"humanlikeness"`` even though the documented schema and Table 3 use
    ``"Humanlikeness"``. Exact match is tried first, then a case-insensitive
    fallback, so no other dimension is affected.
    """
    if dimension in annotation:
        return annotation[dimension]
    lowered = dimension.lower()
    for key, value in annotation.items():
        if key.lower() == lowered:
            return value
    return None


def compute_damr(conversations: list[dict]) -> DamrTable:
    """Compute per-(tutor, dimension) DAMR from parsed MRBench conversations."""
    data_key_to_paper = {data: paper for paper, data in PAPER_TO_DATA_KEY.items()}

    matches: dict[str, dict[str, int]] = {
        paper: dict.fromkeys(DIMENSIONS, 0) for paper in PAPER_TO_DATA_KEY
    }
    totals: dict[str, int] = dict.fromkeys(PAPER_TO_DATA_KEY, 0)
    source_counts: dict[str, int] = {}
    n_responses = 0

    for conv in conversations:
        source = conv.get("Data", "Unknown")
        source_counts[source] = source_counts.get(source, 0) + 1
        for data_key, info in conv.get("anno_llm_responses", {}).items():
            paper = data_key_to_paper.get(data_key)
            if paper is None:
                # Unknown key: surface later rather than silently absorbing it.
                continue
            annotation = info.get("annotation", {})
            totals[paper] += 1
            n_responses += 1
            for dimension in DIMENSIONS:
                if _resolve_label(annotation, dimension) == DESIRED_LABELS[dimension]:
                    matches[paper][dimension] += 1

    cells = {
        paper: {dim: Cell(matches[paper][dim], totals[paper]) for dim in DIMENSIONS}
        for paper in PAPER_TO_DATA_KEY
    }
    missing = [paper for paper, total in totals.items() if total == 0]
    return DamrTable(
        cells=cells,
        response_totals=totals,
        missing_tutors=missing,
        n_conversations=len(conversations),
        n_responses=n_responses,
        data_source_counts=source_counts,
    )
