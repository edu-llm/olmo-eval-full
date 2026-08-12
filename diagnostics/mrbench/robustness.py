"""Perturbation-robustness metrics for the MRBench judge (P2).

Given the canonical (Phase-A) judge records and, per benign variant, a second set
of judge records over the *same* cells, this module measures how much the judge's
decisions and the Phase-A conclusions move under a meaning-preserving reword. It
implements the gates proposed in ``Plan/mrbench/PHASE_A_FP_FN.md``:

  * **desired-decision flip rate** — on the binarised desired-vs-rest decision
    (positive = the DAMR-desired label), per dimension and worst-variant; the
    primary gate is worst variant ``<= 0.10``.
  * **raw 3-class label-flip rate** — diagnostic only, loosely bounded ``<= 0.15``.
  * **|ΔAC|** per dimension (variant minus canonical Pearson AC, over the paired
    cells, reusing :mod:`metrics`); gate ``<= 0.05``.
  * **AC-gate-verdict-unchanged** — recompute the ``AC >= 0.30`` and CI-lower ``> 0``
    pass/fail on the same subsample for canonical and each variant; the per-cell
    verdict of every gate-eligible dimension (Humanlikeness excluded) must not
    change under any variant.
  * **|ΔDAMR|** per dimension (judge-desired rate as a fraction); gate ``<= 0.05``.

The aggregate ``robust`` verdict follows the doc: every variant preserves the
AC-gate verdict **and** worst desired-decision flip ``<= 0.10`` **and**
max ``|ΔAC| <= 0.05``.

Everything here is pure arithmetic over already-collected records; no network.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field

from .judge_cache import CacheKey, JudgeRecord
from .metrics import GoldKey, bootstrap_ci_pearson, pearson
from .reference import DESIRED_LABELS, DIMENSIONS

# Gate constants (from PHASE_A_FP_FN.md "Proposed gates for P2/P3").
DESIRED_FLIP_MAX = 0.10
RAW_FLIP_MAX = 0.15
DELTA_AC_MAX = 0.05
DELTA_DAMR_MAX = 0.05
# AC-gate (PHASE_A_VALIDATION.md): pass if AC >= 0.30 and 95% CI lower bound > 0.
AC_GATE_MIN = 0.30
# Humanlikeness is reported but excluded from the AC gate.
GATE_INELIGIBLE = frozenset({"Humanlikeness"})


def index_ok_records(records: Sequence[JudgeRecord]) -> dict[CacheKey, JudgeRecord]:
    """Index successfully-parsed records by ``(conversation_id, tutor, dimension)``.

    Last write wins if a key appears twice, matching :class:`JudgeCache`'s in-memory
    index semantics.
    """
    idx: dict[CacheKey, JudgeRecord] = {}
    for rec in records:
        if rec.ok and rec.score is not None and rec.label is not None:
            idx[rec.key] = rec
    return idx


@dataclass(frozen=True)
class PairedCell:
    """One cell judged both canonically and under a variant, aligned to gold."""

    conversation_id: str
    tutor: str
    dimension: str
    canon_score: int
    var_score: int
    canon_label: str
    var_label: str
    gold: int | None


def pair_cells(
    canonical_idx: dict[CacheKey, JudgeRecord],
    variant_idx: dict[CacheKey, JudgeRecord],
    gold_index: dict[GoldKey, int],
    dimension: str,
) -> list[PairedCell]:
    """Cells present-and-ok in both the canonical and variant sets, for one dimension."""
    paired: list[PairedCell] = []
    for key, var_rec in variant_idx.items():
        if key[2] != dimension:
            continue
        canon_rec = canonical_idx.get(key)
        if canon_rec is None:
            continue
        # index_ok_records guarantees these are non-None; narrow for the type checker.
        canon_score, var_score = canon_rec.score, var_rec.score
        canon_label, var_label = canon_rec.label, var_rec.label
        if canon_score is None or var_score is None or canon_label is None or var_label is None:
            continue
        paired.append(
            PairedCell(
                conversation_id=key[0],
                tutor=key[1],
                dimension=dimension,
                canon_score=canon_score,
                var_score=var_score,
                canon_label=canon_label,
                var_label=var_label,
                gold=gold_index.get(key),
            )
        )
    return paired


# --------------------------------------------------------------------------- #
# Flip rates (with an optional conversation-clustered bootstrap CI)
# --------------------------------------------------------------------------- #
def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return math.nan
    idx = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


def _rate(indicators: Sequence[int]) -> float:
    return sum(indicators) / len(indicators) if indicators else math.nan


def clustered_ci_rate(
    indicators_by_conversation: dict[str, list[int]],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Conversation-clustered percentile CI for a 0/1 rate.

    Mirrors the block-bootstrap design of :mod:`reliability` (resample whole
    ``conversation_id``s with replacement, then pool their cells and recompute the
    rate), so within-dialogue correlation is not treated away.
    """
    cids = list(indicators_by_conversation)
    n_clusters = len(cids)
    if n_clusters == 0:
        return (math.nan, math.nan)
    rng = random.Random(seed)
    stats: list[float] = []
    for _ in range(n_boot):
        pooled: list[int] = []
        for _ in range(n_clusters):
            pooled.extend(indicators_by_conversation[cids[rng.randrange(n_clusters)]])
        if pooled:
            stats.append(_rate(pooled))
    stats.sort()
    return (_percentile(stats, alpha / 2), _percentile(stats, 1 - alpha / 2))


def _desired_indicator(label: str, dimension: str) -> int:
    return 1 if label == DESIRED_LABELS[dimension] else 0


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #
@dataclass
class DimensionRobustness:
    dimension: str
    variant: str
    eligible_for_gate: bool
    n_paired: int
    n_conversations: int
    n_ac_pairs: int
    desired_flip_rate: float
    desired_flip_ci_clustered: tuple[float, float]
    raw_flip_rate: float
    raw_flip_ci_clustered: tuple[float, float]
    ac_canonical: float
    ac_variant: float
    delta_ac: float
    ac_ci_lower_canonical: float
    ac_ci_lower_variant: float
    gate_pass_canonical: bool
    gate_pass_variant: bool
    gate_verdict_changed: bool
    damr_canonical: float
    damr_variant: float
    delta_damr: float


@dataclass
class VariantRobustness:
    variant: str
    per_dimension: dict[str, DimensionRobustness] = field(default_factory=dict)
    overall_desired_flip_rate: float = math.nan
    overall_raw_flip_rate: float = math.nan
    max_abs_delta_ac: float = math.nan
    max_abs_delta_damr: float = math.nan
    any_gate_verdict_changed: bool = False


@dataclass
class RobustnessReport:
    variants: dict[str, VariantRobustness] = field(default_factory=dict)
    worst_desired_flip_rate: float = math.nan
    worst_raw_flip_rate: float = math.nan
    max_abs_delta_ac: float = math.nan
    max_abs_delta_damr: float = math.nan
    any_gate_verdict_changed: bool = False
    robust: bool = False
    gates: dict[str, float] = field(default_factory=dict)


def _ac_and_gate(
    pairs: Sequence[tuple[int, int]], n_boot: int, seed: int
) -> tuple[float, float, bool]:
    """Return ``(ac, ci_lower, gate_pass)`` for (judge, gold) pairs.

    Gate pass = ``ac >= AC_GATE_MIN`` and the 95% bootstrap CI lower bound ``> 0``.
    """
    ac = pearson([p[0] for p in pairs], [p[1] for p in pairs])
    lo, _hi = bootstrap_ci_pearson(pairs, n_boot=n_boot, seed=seed)
    gate = (not math.isnan(ac)) and ac >= AC_GATE_MIN and (not math.isnan(lo)) and lo > 0.0
    return ac, lo, gate


def compute_dimension_robustness(
    dimension: str,
    variant: str,
    paired: Sequence[PairedCell],
    *,
    n_boot: int = 1000,
    flip_boot: int = 1000,
    seed: int = 42,
) -> DimensionRobustness:
    """All P2 metrics for one (dimension, variant) over the paired cells."""
    desired_ind = [
        1
        if _desired_indicator(c.canon_label, dimension)
        != _desired_indicator(c.var_label, dimension)
        else 0
        for c in paired
    ]
    raw_ind = [1 if c.canon_score != c.var_score else 0 for c in paired]

    desired_by_conv: dict[str, list[int]] = {}
    raw_by_conv: dict[str, list[int]] = {}
    for c, d, r in zip(paired, desired_ind, raw_ind, strict=True):
        desired_by_conv.setdefault(c.conversation_id, []).append(d)
        raw_by_conv.setdefault(c.conversation_id, []).append(r)

    ac_pairs_canon = [(c.canon_score, c.gold) for c in paired if c.gold is not None]
    ac_pairs_var = [(c.var_score, c.gold) for c in paired if c.gold is not None]
    ac_canon, lo_canon, gate_canon = _ac_and_gate(ac_pairs_canon, n_boot, seed)
    ac_var, lo_var, gate_var = _ac_and_gate(ac_pairs_var, n_boot, seed)
    delta_ac = ac_var - ac_canon if not (math.isnan(ac_var) or math.isnan(ac_canon)) else math.nan

    damr_canon = _rate([_desired_indicator(c.canon_label, dimension) for c in paired])
    damr_var = _rate([_desired_indicator(c.var_label, dimension) for c in paired])
    delta_damr = (
        damr_var - damr_canon if not (math.isnan(damr_var) or math.isnan(damr_canon)) else math.nan
    )

    eligible = dimension not in GATE_INELIGIBLE
    verdict_changed = eligible and (gate_canon != gate_var)

    return DimensionRobustness(
        dimension=dimension,
        variant=variant,
        eligible_for_gate=eligible,
        n_paired=len(paired),
        n_conversations=len(desired_by_conv),
        n_ac_pairs=len(ac_pairs_canon),
        desired_flip_rate=_rate(desired_ind),
        desired_flip_ci_clustered=clustered_ci_rate(desired_by_conv, n_boot=flip_boot, seed=seed),
        raw_flip_rate=_rate(raw_ind),
        raw_flip_ci_clustered=clustered_ci_rate(raw_by_conv, n_boot=flip_boot, seed=seed),
        ac_canonical=ac_canon,
        ac_variant=ac_var,
        delta_ac=delta_ac,
        ac_ci_lower_canonical=lo_canon,
        ac_ci_lower_variant=lo_var,
        gate_pass_canonical=gate_canon,
        gate_pass_variant=gate_var,
        gate_verdict_changed=verdict_changed,
        damr_canonical=damr_canon,
        damr_variant=damr_var,
        delta_damr=delta_damr,
    )


def _abs_nanmax(values: Sequence[float]) -> float:
    present = [abs(v) for v in values if not math.isnan(v)]
    return max(present) if present else math.nan


def compute_variant_robustness(
    variant: str,
    canonical_idx: dict[CacheKey, JudgeRecord],
    variant_idx: dict[CacheKey, JudgeRecord],
    gold_index: dict[GoldKey, int],
    *,
    n_boot: int = 1000,
    flip_boot: int = 1000,
    seed: int = 42,
) -> VariantRobustness:
    """Per-dimension + pooled P2 metrics for a single variant."""
    result = VariantRobustness(variant=variant)
    pooled_desired: list[int] = []
    pooled_raw: list[int] = []
    for dim in DIMENSIONS:
        paired = pair_cells(canonical_idx, variant_idx, gold_index, dim)
        dr = compute_dimension_robustness(
            dim, variant, paired, n_boot=n_boot, flip_boot=flip_boot, seed=seed
        )
        result.per_dimension[dim] = dr
        pooled_desired.extend(
            1
            if _desired_indicator(c.canon_label, dim) != _desired_indicator(c.var_label, dim)
            else 0
            for c in paired
        )
        pooled_raw.extend(1 if c.canon_score != c.var_score else 0 for c in paired)

    result.overall_desired_flip_rate = _rate(pooled_desired)
    result.overall_raw_flip_rate = _rate(pooled_raw)
    result.max_abs_delta_ac = _abs_nanmax([dr.delta_ac for dr in result.per_dimension.values()])
    result.max_abs_delta_damr = _abs_nanmax([dr.delta_damr for dr in result.per_dimension.values()])
    result.any_gate_verdict_changed = any(
        dr.gate_verdict_changed for dr in result.per_dimension.values()
    )
    return result


def compute_robustness(
    canonical_records: Sequence[JudgeRecord],
    variant_records: dict[str, list[JudgeRecord]],
    gold_index: dict[GoldKey, int],
    *,
    n_boot: int = 1000,
    flip_boot: int = 1000,
    seed: int = 42,
) -> RobustnessReport:
    """Full P2 robustness report across every variant vs the canonical judge."""
    canonical_idx = index_ok_records(canonical_records)
    report = RobustnessReport(
        gates={
            "desired_flip_max": DESIRED_FLIP_MAX,
            "raw_flip_max": RAW_FLIP_MAX,
            "delta_ac_max": DELTA_AC_MAX,
            "delta_damr_max": DELTA_DAMR_MAX,
            "ac_gate_min": AC_GATE_MIN,
        }
    )
    for variant, records in variant_records.items():
        variant_idx = index_ok_records(records)
        report.variants[variant] = compute_variant_robustness(
            variant,
            canonical_idx,
            variant_idx,
            gold_index,
            n_boot=n_boot,
            flip_boot=flip_boot,
            seed=seed,
        )

    if report.variants:
        report.worst_desired_flip_rate = _abs_nanmax(
            [v.overall_desired_flip_rate for v in report.variants.values()]
        )
        report.worst_raw_flip_rate = _abs_nanmax(
            [v.overall_raw_flip_rate for v in report.variants.values()]
        )
        report.max_abs_delta_ac = _abs_nanmax(
            [v.max_abs_delta_ac for v in report.variants.values()]
        )
        report.max_abs_delta_damr = _abs_nanmax(
            [v.max_abs_delta_damr for v in report.variants.values()]
        )
        report.any_gate_verdict_changed = any(
            v.any_gate_verdict_changed for v in report.variants.values()
        )

    report.robust = (
        (not report.any_gate_verdict_changed)
        and (
            not math.isnan(report.worst_desired_flip_rate)
            and report.worst_desired_flip_rate <= DESIRED_FLIP_MAX
        )
        and (not math.isnan(report.max_abs_delta_ac) and report.max_abs_delta_ac <= DELTA_AC_MAX)
    )
    return report
