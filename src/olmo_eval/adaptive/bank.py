"""Calibrated 3PL item bank loading for ATLAS adaptive testing.

An :class:`ItemBank` pairs each benchmark ``question_id`` with its calibrated
3PL parameters ``(a, b, c)``. ATLAS publishes mirt-style parameters keyed by a
numeric item index (``X1``, ``X2``, ...); a per-benchmark
``atlas_idx_to_question_id.csv`` bridges those indices to native benchmark
``question_id``s (e.g. ARC's ``Mercury_7175875``), which is exactly the id
``olmo-eval`` carries in ``Instance.metadata["id"]``.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# repo_root/src/olmo_eval/adaptive/bank.py -> parents[3] == repo root
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_ARC_DIR = _REPO_ROOT / "AdaptiveTesting" / "Inputs" / "ATLAS" / "arc"
_ENV_BANK_DIR = "OLMO_EVAL_ATLAS_BANK_DIR"


@dataclass
class ItemBank:
    """A calibrated 3PL item bank aligned to benchmark ``question_id``s."""

    question_ids: list[str]
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    version: str

    def __post_init__(self) -> None:
        self._pos: dict[str, int] = {q: i for i, q in enumerate(self.question_ids)}

    def __len__(self) -> int:
        return len(self.question_ids)

    def position(self, question_id: str) -> int | None:
        """Bank column index for a ``question_id`` (or ``None`` if absent)."""
        return self._pos.get(question_id)

    def has(self, question_id: str) -> bool:
        return question_id in self._pos

    def restrict(self, allowed: set[str]) -> ItemBank:
        """Return a sub-bank of items whose ``question_id`` is in ``allowed``.

        Used to align a full-benchmark run (which produces responses for every
        item) to the calibrated subset the bank actually covers.
        """
        keep = [i for i, q in enumerate(self.question_ids) if q in allowed]
        idx = np.asarray(keep, dtype=int)
        return ItemBank(
            question_ids=[self.question_ids[i] for i in keep],
            a=self.a[idx],
            b=self.b[idx],
            c=self.c[idx],
            version=f"{self.version}:restrict{len(keep)}",
        )


def _resolve_bank_dir(bank_dir: str | os.PathLike[str] | None) -> Path:
    if bank_dir is not None:
        return Path(bank_dir)
    env = os.environ.get(_ENV_BANK_DIR)
    if env:
        return Path(env)
    return _DEFAULT_ARC_DIR


def load_bank(
    bank_dir: str | os.PathLike[str] | None = None,
    *,
    params_file: str = "irt_item_parameters_combined.csv",
    idx_map_file: str = "atlas_idx_to_question_id.csv",
) -> ItemBank:
    """Load a 3PL bank from an ATLAS benchmark directory.

    Args:
        bank_dir: Directory holding the params + idx-map CSVs. Defaults to the
            vendored ARC bank (or ``$OLMO_EVAL_ATLAS_BANK_DIR``).
        params_file: mirt-style params CSV (columns ``X, a1, d, g, u``).
        idx_map_file: index->question_id bridge (columns ``atlas_idx, question_id``).

    Returns:
        An :class:`ItemBank` with items whose discrimination is positive,
        finite, and present in the idx map.
    """
    root = _resolve_bank_dir(bank_dir)
    params_path = root / params_file
    idx_map_path = root / idx_map_file
    idx_map = _load_idx_map(idx_map_path)

    qids: list[str] = []
    a_list: list[float] = []
    b_list: list[float] = []
    c_list: list[float] = []
    with open(params_path, newline="") as fh:
        for row in csv.DictReader(fh):
            k = int(str(row["X"]).lstrip("X"))
            a = float(row["a1"])
            d = float(row["d"])
            g = float(row["g"])
            if a <= 0 or not np.isfinite(a) or k not in idx_map:
                continue
            qids.append(idx_map[k])
            a_list.append(a)
            b_list.append(-d / a)
            c_list.append(float(np.clip(g, 0.0, 0.999)))

    version = f"{root.name}:{params_file}:n{len(qids)}"
    return ItemBank(
        question_ids=qids,
        a=np.asarray(a_list, dtype=float),
        b=np.asarray(b_list, dtype=float),
        c=np.asarray(c_list, dtype=float),
        version=version,
    )


def _load_idx_map(idx_map_path: Path) -> dict[int, str]:
    if not idx_map_path.is_file():
        raise FileNotFoundError(
            f"missing ATLAS index->question_id map: {idx_map_path}. "
            "Point bank_dir at a directory containing atlas_idx_to_question_id.csv."
        )
    out: dict[int, str] = {}
    with open(idx_map_path, newline="") as fh:
        for row in csv.DictReader(fh):
            qid = (row.get("question_id") or "").strip()
            if qid:
                out[int(row["atlas_idx"])] = qid
    return out
