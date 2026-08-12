"""Per-class one-vs-rest decision-rate reliability for the MRBench judge.

This is a FREE, offline re-analysis of already-cached judge outputs. It adds a
classification / decision-rate lens on top of the Pearson-AC judge validation:
for each dimension and each ordinal class it treats ``label == class`` as the
positive and reports the false-positive rate (fall-out), false-negative rate
(miss rate), precision, recall and one-vs-rest F1, each with a 95% confidence
interval.

Two bootstrap flavours are provided so the effect of within-dialogue correlation
is visible:

  * a plain case-level percentile bootstrap that resamples individual judged
    cells (reusing :func:`metrics.bootstrap_ci_rate` for the two rates), and
  * a ``conversation_id``-clustered percentile bootstrap that resamples whole
    conversations, since cells from the same dialogue (across tutors and
    dimensions) are correlated and a case-level interval understates the
    uncertainty.

Alignment of judge to gold is identical to :mod:`metrics`: a judged cell is
paired to gold via ``(conversation_id, tutor, dimension)`` and only successfully
parsed records with a matching gold score contribute. The per-dimension macro-F1
(mean of the per-class F1s) is therefore a direct cross-check against
:func:`metrics.compute_macro_f1`.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field

from .judge_cache import JudgeRecord
from .judge_prompt import GOLD_LABEL_TO_SCORE, SCORE_TO_LABEL
from .metrics import GoldKey, bootstrap_ci_rate, f1_for_class
from .reference import DESIRED_LABELS, DIMENSIONS

# The ordinal classes shared by every Figure-6 rubric (1-3). One-vs-rest is run
# over these so the per-dimension macro-F1 lines up with ``metrics.macro_f1``.
CLASSES: tuple[int, ...] = (1, 2, 3)

# A judged cell tagged with the conversation it came from, so the clustered
# bootstrap can resample whole dialogues.
ClusteredPair = tuple[int, int, str]  # (judge_score, gold_score, conversation_id)


def collect_dimension_pairs_clustered(
    records: list[JudgeRecord], gold_index: dict[GoldKey, int]
) -> dict[str, list[ClusteredPair]]:
    """Per-dimension ``(judge, gold, conversation_id)`` triples.

    Mirrors :func:`metrics.collect_dimension_pairs` exactly (same skip-if-unparsed
    and same gold lookup) but keeps the conversation id so a clustered bootstrap
    can group cells by dialogue.
    """
    pairs: dict[str, list[ClusteredPair]] = {d: [] for d in DIMENSIONS}
    for rec in records:
        if not rec.ok or rec.score is None:
            continue
        gold = gold_index.get((rec.conversation_id, rec.tutor, rec.dimension))
        if gold is not None:
            pairs[rec.dimension].append((rec.score, gold, rec.conversation_id))
    return pairs


def _confusion(pairs: Sequence[tuple[int, int]], cls: int) -> tuple[int, int, int, int]:
    """One-vs-rest ``(tp, fp, fn, tn)`` for ``cls`` over ``(judge, gold)`` pairs."""
    tp = fp = fn = tn = 0
    for pred, gold in pairs:
        pred_pos = pred == cls
        gold_pos = gold == cls
        if pred_pos and gold_pos:
            tp += 1
        elif pred_pos and not gold_pos:
            fp += 1
        elif not pred_pos and gold_pos:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def _fp_rate(tp: int, fp: int, fn: int, tn: int) -> float:
    """Fall-out: FP / (gold != c) = FP / (FP + TN); NaN if no negatives."""
    neg = fp + tn
    return fp / neg if neg else math.nan


def _fn_rate(tp: int, fp: int, fn: int, tn: int) -> float:
    """Miss rate: FN / (gold == c) = FN / (TP + FN) = 1 - recall; NaN if no positives."""
    pos = tp + fn
    return fn / pos if pos else math.nan


def _precision(tp: int, fp: int, fn: int, tn: int) -> float:
    denom = tp + fp
    return tp / denom if denom else math.nan


def _recall(tp: int, fp: int, fn: int, tn: int) -> float:
    denom = tp + fn
    return tp / denom if denom else math.nan


def _f1_from_confusion(tp: int, fp: int, fn: int, tn: int) -> float:
    """F1 matching :func:`metrics.f1_for_class` (0.0 when there are no true positives)."""
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    if prec + rec == 0.0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return math.nan
    idx = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


# --------------------------------------------------------------------------- #
# Bootstraps
# --------------------------------------------------------------------------- #
def _stat_for_class(pairs: Sequence[tuple[int, int]], cls: int, stat: str) -> float:
    tp, fp, fn, tn = _confusion(pairs, cls)
    if stat == "fp_rate":
        return _fp_rate(tp, fp, fn, tn)
    if stat == "fn_rate":
        return _fn_rate(tp, fp, fn, tn)
    if stat == "f1":
        return _f1_from_confusion(tp, fp, fn, tn)
    raise ValueError(f"unknown stat {stat!r}")


def bootstrap_ci_case(
    pairs: Sequence[tuple[int, int]],
    cls: int,
    stat: str,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Case-level percentile CI: resample individual judged cells with replacement."""
    n = len(pairs)
    if n == 0:
        return (math.nan, math.nan)
    rng = random.Random(seed)
    stats: list[float] = []
    for _ in range(n_boot):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        v = _stat_for_class(sample, cls, stat)
        if not math.isnan(v):
            stats.append(v)
    stats.sort()
    return (_percentile(stats, alpha / 2), _percentile(stats, 1 - alpha / 2))


def _group_by_conversation(
    pairs: Sequence[ClusteredPair],
) -> tuple[list[str], dict[str, list[tuple[int, int]]]]:
    groups: dict[str, list[tuple[int, int]]] = {}
    for judge, gold, cid in pairs:
        groups.setdefault(cid, []).append((judge, gold))
    return list(groups), groups


def bootstrap_ci_clustered_multi(
    pairs: Sequence[ClusteredPair],
    classes: Sequence[int] = CLASSES,
    stats: Sequence[str] = ("fp_rate", "fn_rate", "f1"),
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[int, dict[str, tuple[float, float]]]:
    """Conversation-clustered percentile CIs for every (class, stat) in one pass.

    Whole conversations are resampled with replacement (the block bootstrap for a
    two-level design), then each statistic is recomputed on the pooled cells. All
    requested statistics for all classes are read off the *same* resample so the
    intervals are internally consistent and cheap to produce.
    """
    cids, groups = _group_by_conversation(pairs)
    n_clusters = len(cids)
    result: dict[int, dict[str, list[float]]] = {c: {s: [] for s in stats} for c in classes}
    if n_clusters == 0:
        return {c: {s: (math.nan, math.nan) for s in stats} for c in classes}

    rng = random.Random(seed)
    for _ in range(n_boot):
        resampled: list[tuple[int, int]] = []
        for _ in range(n_clusters):
            resampled.extend(groups[cids[rng.randrange(n_clusters)]])
        for c in classes:
            tp, fp, fn, tn = _confusion(resampled, c)
            for s in stats:
                if s == "fp_rate":
                    v = _fp_rate(tp, fp, fn, tn)
                elif s == "fn_rate":
                    v = _fn_rate(tp, fp, fn, tn)
                else:
                    v = _f1_from_confusion(tp, fp, fn, tn)
                if not math.isnan(v):
                    result[c][s].append(v)

    out: dict[int, dict[str, tuple[float, float]]] = {}
    for c in classes:
        out[c] = {}
        for s in stats:
            vals = sorted(result[c][s])
            out[c][s] = (_percentile(vals, alpha / 2), _percentile(vals, 1 - alpha / 2))
    return out


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #
@dataclass
class ClassReliability:
    dimension: str
    cls: int
    label: str
    is_desired: bool
    n_gold_pos: int  # gold == c  (denominator of FN rate)
    n_gold_neg: int  # gold != c  (denominator of FP rate)
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float
    fp_rate: float
    fn_rate: float
    fp_rate_ci_case: tuple[float, float]
    fp_rate_ci_clustered: tuple[float, float]
    fn_rate_ci_case: tuple[float, float]
    fn_rate_ci_clustered: tuple[float, float]
    f1_ci_case: tuple[float, float]
    f1_ci_clustered: tuple[float, float]
    sparse: bool


@dataclass
class DimensionReliability:
    dimension: str
    n_pairs: int
    n_conversations: int
    macro_f1: float
    classes: list[ClassReliability] = field(default_factory=list)


def _desired_score(dimension: str) -> int | None:
    """Ordinal score of the DAMR-desired label for a dimension (never hard-coded)."""
    desired_label = DESIRED_LABELS[dimension]
    return GOLD_LABEL_TO_SCORE[dimension].get(desired_label)


def compute_dimension_reliability(
    dimension: str,
    clustered_pairs: Sequence[ClusteredPair],
    classes: Sequence[int] = CLASSES,
    n_boot: int = 2000,
    seed: int = 42,
    sparse_threshold: int = 30,
) -> DimensionReliability:
    """Per-class one-vs-rest decision-rate metrics + CIs for one dimension."""
    flat: list[tuple[int, int]] = [(j, g) for j, g, _ in clustered_pairs]
    n_conversations = len({cid for _, _, cid in clustered_pairs})
    desired = _desired_score(dimension)

    clustered_cis = bootstrap_ci_clustered_multi(
        clustered_pairs, classes=classes, n_boot=n_boot, seed=seed
    )

    class_results: list[ClassReliability] = []
    f1s: list[float] = []
    for c in classes:
        tp, fp, fn, tn = _confusion(flat, c)
        n_gold_pos = tp + fn
        n_gold_neg = fp + tn
        f1 = f1_for_class(flat, c)  # reuse metrics' definition for the cross-check
        f1s.append(f1)

        # Case-level CIs: reuse metrics.bootstrap_ci_rate for the two rates.
        # FP rate is a mean of (judge == c) over the gold != c subpopulation;
        # FN rate is a mean of (judge != c) over the gold == c subpopulation.
        fp_indicators = [1 if j == c else 0 for j, g in flat if g != c]
        fn_indicators = [1 if j != c else 0 for j, g in flat if g == c]
        fp_lo, fp_hi = bootstrap_ci_rate(fp_indicators, n_boot=n_boot, seed=seed)
        fn_lo, fn_hi = bootstrap_ci_rate(fn_indicators, n_boot=n_boot, seed=seed)
        f1_case = bootstrap_ci_case(flat, c, "f1", n_boot=n_boot, seed=seed)

        class_results.append(
            ClassReliability(
                dimension=dimension,
                cls=c,
                label=SCORE_TO_LABEL[dimension][c],
                is_desired=(desired == c),
                n_gold_pos=n_gold_pos,
                n_gold_neg=n_gold_neg,
                tp=tp,
                fp=fp,
                fn=fn,
                tn=tn,
                precision=_precision(tp, fp, fn, tn),
                recall=_recall(tp, fp, fn, tn),
                f1=f1,
                fp_rate=_fp_rate(tp, fp, fn, tn),
                fn_rate=_fn_rate(tp, fp, fn, tn),
                # bootstrap_ci_rate returns percentages; convert to fractions.
                fp_rate_ci_case=(fp_lo / 100.0, fp_hi / 100.0),
                fp_rate_ci_clustered=clustered_cis[c]["fp_rate"],
                fn_rate_ci_case=(fn_lo / 100.0, fn_hi / 100.0),
                fn_rate_ci_clustered=clustered_cis[c]["fn_rate"],
                f1_ci_case=f1_case,
                f1_ci_clustered=clustered_cis[c]["f1"],
                sparse=(n_gold_pos < sparse_threshold),
            )
        )

    macro = sum(f1s) / len(f1s) if f1s else math.nan
    return DimensionReliability(
        dimension=dimension,
        n_pairs=len(flat),
        n_conversations=n_conversations,
        macro_f1=macro,
        classes=class_results,
    )


def compute_reliability(
    records: list[JudgeRecord],
    gold_index: dict[GoldKey, int],
    classes: Sequence[int] = CLASSES,
    n_boot: int = 2000,
    seed: int = 42,
    sparse_threshold: int = 30,
) -> dict[str, DimensionReliability]:
    """Per-dimension per-class FP/FN decision-rate reliability over all dimensions."""
    clustered = collect_dimension_pairs_clustered(records, gold_index)
    return {
        dim: compute_dimension_reliability(
            dim,
            clustered[dim],
            classes=classes,
            n_boot=n_boot,
            seed=seed,
            sparse_threshold=sparse_threshold,
        )
        for dim in DIMENSIONS
    }
