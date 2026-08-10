"""Phase-A metrics: AC (judge-vs-gold Pearson) and judge-derived DAMR.

  * AC  — per-dimension Pearson correlation between the judge score (1-3) and the
          gold score (1-3), computed overall and per tutor (comparable to the
          paper's Tables 5/6). This is the judge-reliability number.
  * judge-DAMR — the M0 desired-label rate, but using the judge's labels instead
          of gold, so tutor quality can be read off the judge directly.

Both operate on cached :class:`~diagnostics.mrbench.judge_cache.JudgeRecord`s plus
the parsed MRBench conversations (for gold). Pure arithmetic; no network.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field

from .baselines import BASELINES
from .judge_cache import JudgeRecord
from .judge_prompt import GOLD_LABEL_TO_SCORE
from .reference import DESIRED_LABELS, DIMENSIONS, PAPER_TO_DATA_KEY

GoldKey = tuple[str, str, str]  # (conversation_id, paper_tutor, dimension)


def pearson(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation; NaN if fewer than 2 pairs or either side is constant."""
    n = len(xs)
    if n < 2 or len(ys) != n:
        return math.nan
    mx = sum(xs) / n
    my = sum(ys) / n
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    num = sum(a * b for a, b in zip(dx, dy, strict=True))
    den = math.sqrt(sum(a * a for a in dx) * sum(b * b for b in dy))
    if den == 0.0:
        return math.nan
    return num / den


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return math.nan
    idx = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


def bootstrap_ci_pearson(
    pairs: Sequence[tuple[float, float]], n_boot: int = 1000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI for Pearson AC over (judge, gold) pairs.

    Cheap and non-invasive: pure resampling of an already-computed pair list.
    """
    n = len(pairs)
    if n < 2:
        return (math.nan, math.nan)
    rng = random.Random(seed)
    stats: list[float] = []
    for _ in range(n_boot):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        r = pearson([p[0] for p in sample], [p[1] for p in sample])
        if not math.isnan(r):
            stats.append(r)
    stats.sort()
    return (_percentile(stats, alpha / 2), _percentile(stats, 1 - alpha / 2))


def bootstrap_ci_rate(
    indicators: Sequence[int], n_boot: int = 1000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI (in %) for a match rate over 0/1 indicators."""
    n = len(indicators)
    if n == 0:
        return (math.nan, math.nan)
    rng = random.Random(seed)
    stats: list[float] = []
    for _ in range(n_boot):
        s = sum(indicators[rng.randrange(n)] for _ in range(n))
        stats.append(100.0 * s / n)
    stats.sort()
    return (_percentile(stats, alpha / 2), _percentile(stats, 1 - alpha / 2))


def _resolve_gold_label(annotation: dict, dimension: str) -> str | None:
    if dimension in annotation:
        return annotation[dimension]
    lowered = dimension.lower()
    for key, value in annotation.items():
        if key.lower() == lowered:
            return value
    return None


def build_gold_index(conversations: list[dict]) -> dict[GoldKey, int]:
    """Map (conversation_id, paper_tutor, dimension) -> gold ordinal score (1-3)."""
    data_key_to_paper = {data: paper for paper, data in PAPER_TO_DATA_KEY.items()}
    index: dict[GoldKey, int] = {}
    for conv in conversations:
        cid = str(conv.get("conversation_id"))
        for data_key, info in conv.get("anno_llm_responses", {}).items():
            paper = data_key_to_paper.get(data_key)
            if paper is None:
                continue
            annotation = info.get("annotation", {})
            for dim in DIMENSIONS:
                label = _resolve_gold_label(annotation, dim)
                score = GOLD_LABEL_TO_SCORE[dim].get(label) if label is not None else None
                if score is not None:
                    index[(cid, paper, dim)] = score
    return index


@dataclass
class AcResult:
    per_dimension: dict[str, float] = field(default_factory=dict)
    per_tutor_dimension: dict[str, dict[str, float]] = field(default_factory=dict)
    n_pairs_per_dimension: dict[str, int] = field(default_factory=dict)


def compute_ac(records: list[JudgeRecord], gold_index: dict[GoldKey, int]) -> AcResult:
    """AC per dimension (overall) and per (tutor, dimension)."""
    dim_pairs: dict[str, list[tuple[int, int]]] = {d: [] for d in DIMENSIONS}
    tut_dim_pairs: dict[str, dict[str, list[tuple[int, int]]]] = {}

    for rec in records:
        if not rec.ok or rec.score is None:
            continue
        gold = gold_index.get((rec.conversation_id, rec.tutor, rec.dimension))
        if gold is None:
            continue
        dim_pairs[rec.dimension].append((rec.score, gold))
        tut_dim_pairs.setdefault(rec.tutor, {d: [] for d in DIMENSIONS})
        tut_dim_pairs[rec.tutor][rec.dimension].append((rec.score, gold))

    result = AcResult()
    for dim in DIMENSIONS:
        pairs = dim_pairs[dim]
        result.per_dimension[dim] = pearson([p[0] for p in pairs], [p[1] for p in pairs])
        result.n_pairs_per_dimension[dim] = len(pairs)
    for tutor, per_dim in tut_dim_pairs.items():
        result.per_tutor_dimension[tutor] = {
            dim: pearson([p[0] for p in per_dim[dim]], [p[1] for p in per_dim[dim]])
            for dim in DIMENSIONS
        }
    return result


def collect_dimension_pairs(
    records: list[JudgeRecord], gold_index: dict[GoldKey, int]
) -> dict[str, list[tuple[int, int]]]:
    """Per-dimension (judge_score, gold_score) pairs, for bootstrap CIs on AC."""
    pairs: dict[str, list[tuple[int, int]]] = {d: [] for d in DIMENSIONS}
    for rec in records:
        if not rec.ok or rec.score is None:
            continue
        gold = gold_index.get((rec.conversation_id, rec.tutor, rec.dimension))
        if gold is not None:
            pairs[rec.dimension].append((rec.score, gold))
    return pairs


def collect_dimension_desired_indicators(records: list[JudgeRecord]) -> dict[str, list[int]]:
    """Per-dimension 0/1 indicators of judge-label == desired, for DAMR CIs."""
    ind: dict[str, list[int]] = {d: [] for d in DIMENSIONS}
    for rec in records:
        if not rec.ok or rec.label is None:
            continue
        ind[rec.dimension].append(1 if rec.label == DESIRED_LABELS[rec.dimension] else 0)
    return ind


# --------------------------------------------------------------------------- #
# SECONDARY metric: per-dimension 3-class macro-F1 (BEA-2025 comparable).
#
# This is a *secondary* reporting number only; it does NOT replace DAMR (tutor
# quality) or Pearson AC (the paper's judge-reliability number). It scores the
# judge's 1-3 predictions against the human gold 1-3 labels, macro-averaged over
# the three ordinal classes so the under-represented middle class ("To some
# extent") is weighted equally — this is what penalises a majority-class guesser.
# --------------------------------------------------------------------------- #
MACRO_F1_CLASSES: tuple[int, ...] = (1, 2, 3)


def f1_for_class(pairs: Sequence[tuple[int, int]], cls: int) -> float:
    """One-vs-rest F1 for a single class over (pred, gold) pairs; 0.0 if undefined."""
    tp = sum(1 for pred, gold in pairs if pred == cls and gold == cls)
    fp = sum(1 for pred, gold in pairs if pred == cls and gold != cls)
    fn = sum(1 for pred, gold in pairs if pred != cls and gold == cls)
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def macro_f1(pairs: Sequence[tuple[int, int]], classes: Sequence[int] = MACRO_F1_CLASSES) -> float:
    """Unweighted mean of per-class F1 over ``classes``; NaN if there are no pairs.

    A class that is present in gold but never predicted contributes F1 0.0, so a
    degenerate majority-class predictor scores far below 1.0 (exactly the BEA
    intent). ``classes`` are fixed at (1, 2, 3) to match the Figure-6 rubric.
    """
    if not pairs:
        return math.nan
    return sum(f1_for_class(pairs, c) for c in classes) / len(classes)


def compute_macro_f1(
    records: list[JudgeRecord],
    gold_index: dict[GoldKey, int],
    classes: Sequence[int] = MACRO_F1_CLASSES,
) -> dict[str, float]:
    """Per-dimension 3-class macro-F1 of judge vs gold (SECONDARY reporting)."""
    pairs = collect_dimension_pairs(records, gold_index)
    return {dim: macro_f1(pairs[dim], classes) for dim in DIMENSIONS}


@dataclass
class JudgeDamrResult:
    cells: dict[str, dict[str, float]] = field(default_factory=dict)
    totals: dict[str, int] = field(default_factory=dict)


def compute_judge_damr(records: list[JudgeRecord]) -> JudgeDamrResult:
    """Judge-derived DAMR: % of a tutor's responses whose judge label is desired."""
    matches: dict[str, dict[str, int]] = {}
    counted: dict[str, set[str]] = {}  # tutor -> conversation ids seen

    # Per (tutor, dim) totals, tracked independently in case some dims fail to parse.
    dim_totals: dict[str, dict[str, int]] = {}
    for rec in records:
        if not rec.ok or rec.label is None:
            continue
        matches.setdefault(rec.tutor, {d: 0 for d in DIMENSIONS})
        dim_totals.setdefault(rec.tutor, {d: 0 for d in DIMENSIONS})
        dim_totals[rec.tutor][rec.dimension] += 1
        if rec.label == DESIRED_LABELS[rec.dimension]:
            matches[rec.tutor][rec.dimension] += 1
        counted.setdefault(rec.tutor, set()).add(rec.conversation_id)

    result = JudgeDamrResult()
    for tutor in matches:
        result.cells[tutor] = {
            dim: (
                100.0 * matches[tutor][dim] / dim_totals[tutor][dim]
                if dim_totals[tutor][dim]
                else math.nan
            )
            for dim in DIMENSIONS
        }
        result.totals[tutor] = len(counted.get(tutor, set()))
    return result


def compare_ac_to_baselines(ac: AcResult) -> dict[str, dict[str, dict[str, float]]]:
    """Per-tutor, per-dimension diff of our judge AC vs each paper baseline.

    Returns ``{baseline_name: {tutor: {dimension: judge_ac - baseline_ac}}}``.
    Cells with a NaN judge AC are skipped.
    """
    out: dict[str, dict[str, dict[str, float]]] = {}
    for name, table in BASELINES.items():
        out[name] = {}
        for tutor, ref_row in table.items():
            judge_row = ac.per_tutor_dimension.get(tutor)
            if judge_row is None:
                continue
            diffs: dict[str, float] = {}
            for i, dim in enumerate(DIMENSIONS):
                jv = judge_row.get(dim, math.nan)
                if not math.isnan(jv):
                    diffs[dim] = jv - ref_row[i]
            out[name][tutor] = diffs
    return out
