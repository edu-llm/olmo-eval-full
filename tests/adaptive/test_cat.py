"""Unit tests for the shared adaptive-testing core (olmo_eval.adaptive)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from olmo_eval.adaptive import (
    CatSession,
    ItemBank,
    TableResponder,
    fisher_info,
    load_bank,
    pirt_accuracy,
    prob,
    run_cat,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ARC_BANK_DIR = _REPO_ROOT / "AdaptiveTesting" / "Inputs" / "ATLAS" / "arc"
_ARC_MCQ_DIR = (
    _REPO_ROOT / "AdaptiveTesting" / "Inputs" / "Open" / "LLM-Judge" / "mcq" / "arc_challenge"
)
_REF_RESULTS = (
    _REPO_ROOT
    / "AdaptiveTesting"
    / "Experiments"
    / "atlas_transfer_published"
    / "results"
    / "atlas_arc_heldout_se0.3.csv"
)


def _toy_bank(n: int = 40, seed: int = 0) -> ItemBank:
    rng = np.random.default_rng(seed)
    return ItemBank(
        question_ids=[f"q{i}" for i in range(n)],
        a=rng.uniform(0.5, 2.5, n),
        b=rng.uniform(-2.0, 2.0, n),
        c=rng.uniform(0.0, 0.3, n),
        version="toy",
    )


def test_prob_is_monotonic_and_bounded() -> None:
    a = np.array([1.5]), np.array([0.0]), np.array([0.2])
    lo = prob(-3.0, *a)[0]
    hi = prob(3.0, *a)[0]
    assert 0.2 <= lo < hi < 1.0


def test_fisher_info_nonnegative() -> None:
    bank = _toy_bank()
    info = fisher_info(0.0, bank.a, bank.b, bank.c)
    assert np.all(info >= 0.0)


def test_cat_never_repeats_and_respects_max_items() -> None:
    bank = _toy_bank(n=30)
    # High-ability responder (always correct) will not hit an SE stop quickly,
    # so max_items governs termination.
    result = run_cat(
        bank,
        TableResponder({q: 1 for q in bank.question_ids}),
        se_stop=0.0,
        min_items=5,
        max_items=12,
    )
    assert result.n_items == 12
    assert len(set(result.order)) == 12  # no item administered twice


def test_cat_stops_on_se_threshold() -> None:
    bank = _toy_bank(n=200)
    result = run_cat(
        bank,
        TableResponder({q: (i % 2) for i, q in enumerate(bank.question_ids)}),
        se_stop=0.3,
        min_items=8,
        max_items=200,
    )
    assert result.n_items >= 8
    assert result.se <= 0.3 or result.n_items == 200


def test_pirt_accuracy_between_zero_and_one() -> None:
    bank = _toy_bank()
    session = CatSession(bank, se_stop=0.3, min_items=8, max_items=20)
    while (idx := session.next_item()) is not None:
        session.record(idx, 1 if idx % 2 else 0)
    acc = pirt_accuracy(bank, session.order, session.scores, session.theta)
    assert 0.0 <= acc <= 1.0


@pytest.mark.skipif(
    not _ARC_BANK_DIR.exists() or not _REF_RESULTS.exists(),
    reason="vendored ATLAS ARC bank / reference results not present",
)
def test_matches_reference_atlas_numbers() -> None:
    """Reproduce a recorded held-out row from the teammate's ATLAS diagnostic.

    Validates that the lifted core produces identical theta / p-IRT / item count
    to atlas_transfer_published/results/atlas_arc_heldout_se0.3.csv.
    """
    model = "allenai/OLMo-2-1124-7B"
    resp_csv = _ARC_MCQ_DIR / f"{model.replace('/', '__')}.csv"
    if not resp_csv.exists():
        pytest.skip(f"missing responses for {model}")

    bank = load_bank(_ARC_BANK_DIR)
    scores: dict[str, int] = {}
    with open(resp_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            scores[row["question_id"]] = 1 if row["result"].strip().lower() == "correct" else 0
    if not all(q in scores for q in bank.question_ids):
        pytest.skip("response file does not cover the full bank")

    result = run_cat(bank, TableResponder(scores), se_stop=0.3, min_items=8, max_items=200)

    ref = _load_ref_row(model)
    assert result.n_items == int(ref["n_items"])
    assert result.theta == pytest.approx(float(ref["theta"]), abs=1e-4)
    assert result.se == pytest.approx(float(ref["se"]), abs=1e-4)
    assert result.pirt_accuracy == pytest.approx(float(ref["pirt_pred"]), abs=1e-4)


@pytest.mark.skipif(
    not _ARC_BANK_DIR.exists() or not _ARC_MCQ_DIR.exists(),
    reason="vendored ATLAS ARC bank / responses not present",
)
def test_online_matches_offline_and_queries_small_subset() -> None:
    """Online CAT == offline CAT given identical responses, using only a subset.

    Both entry points drive the same CatSession, so with the same per-item
    scores they must produce identical theta/order. The item count must be far
    below the full bank -- that gap is the inference saving of online CAT.
    """
    import asyncio

    from olmo_eval.adaptive import run_cat_async

    model = "allenai/OLMo-2-1124-7B"
    resp_csv = _ARC_MCQ_DIR / f"{model.replace('/', '__')}.csv"
    if not resp_csv.exists():
        pytest.skip(f"missing responses for {model}")

    bank = load_bank(_ARC_BANK_DIR)
    scores: dict[str, int] = {}
    with open(resp_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            scores[row["question_id"]] = 1 if row["result"].strip().lower() == "correct" else 0
    if not all(q in scores for q in bank.question_ids):
        pytest.skip("response file does not cover the full bank")

    table = TableResponder(scores)
    offline = run_cat(bank, table, se_stop=0.3, min_items=8, max_items=200)

    async def aresponder(qid: str) -> int:
        return table(qid)

    online = asyncio.run(run_cat_async(bank, aresponder, se_stop=0.3, min_items=8, max_items=200))

    assert online.order == offline.order
    assert online.theta == pytest.approx(offline.theta, abs=1e-12)
    assert online.n_items == offline.n_items
    assert online.n_items < len(bank) / 5  # only a small slice of the bank is used


def _load_ref_row(model: str) -> dict[str, str]:
    with open(_REF_RESULTS, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["model"] == model:
                return row
    raise AssertionError(f"{model} not in reference results")
