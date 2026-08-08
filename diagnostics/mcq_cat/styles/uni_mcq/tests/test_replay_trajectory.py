"""The offline trajectory replay: that it is exact, and that it says so when it is not.

``cat_report.json`` records only the final ability estimate, and both requested figures
are functions of the intermediate ones. :mod:`..scripts.replay_trajectory` recovers them
by re-running EAP over prefixes of the response sequence the session actually
administered. The whole construction rests on one property -- that the replay is the
same estimator over the same inputs, and therefore reproduces the recorded endpoint
exactly -- so most of what is tested below is the guard that checks it rather than the
arithmetic it guards.

The distinction the mismatch tests are drawing is not "roughly right versus wrong". A
standard-error curve falls smoothly whether or not it belongs to the run named in its
title, and a theta trajectory converges either way, so a reconstruction that is subtly
wrong produces a figure indistinguishable from a correct one. Nothing downstream can
detect it. The endpoint check is the only place it can be caught, and a test that only
proved the happy path would leave that check free to be silently disabled.

**One thing the endpoint check cannot do, and the tests here have to.** EAP's likelihood
is a product over responses, so the final estimate is invariant to their order: a replay
that shuffled the sequence would land on exactly the recorded endpoint and draw an
entirely fictional path to it. :meth:`TestTheReplayIsTheEstimatorOverEachPrefix` is what
pins the ordering, by recomputing every prefix independently.

Torch-free and matplotlib-free by construction. The script imports numpy and the frozen
loaders, and reaches for pyplot only inside a function that returns ``None`` when it is
absent -- which, since ``matplotlib`` lives in the optional ``analysis`` extra nothing
syncs, is the case on every machine this currently runs on.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ..irt import eap_theta_se, mwle_theta_se
from ..scripts import replay_trajectory as replay_mod
from ..scripts.replay_trajectory import (
    DEFAULT_TOLERANCE,
    PRIOR_SE,
    PRIOR_THETA,
    SINGLE_RUN_NOTE,
    ReplayError,
    first_crossing,
)

#: Where the committed banks live, spelled out rather than imported so this module does
#: not depend on the resolver to find out whether a bank exists.
CALIBRATED_DATASETS = Path(__file__).resolve().parents[5] / "calibrated_datasets"

#: A six-item bank spanning easy to hard, with one 2PL-shaped item (guessing 0). Wide
#: enough in ``b`` that a wrong parameter row moves theta by far more than any tolerance
#: under discussion, which is what makes the mismatch tests mean something.
BANK = [
    {"item_id": "it_0", "difficulty": -1.8, "discrimination": 1.3, "guessing": 0.20},
    {"item_id": "it_1", "difficulty": -0.9, "discrimination": 1.9, "guessing": 0.25},
    {"item_id": "it_2", "difficulty": -0.1, "discrimination": 2.4, "guessing": 0.00},
    {"item_id": "it_3", "difficulty": 0.6, "discrimination": 1.6, "guessing": 0.20},
    {"item_id": "it_4", "difficulty": 1.4, "discrimination": 1.1, "guessing": 0.25},
    {"item_id": "it_5", "difficulty": 2.1, "discrimination": 0.9, "guessing": 0.20},
]

#: An administered sequence and its outcomes, in order.
ORDER = ["it_2", "it_3", "it_1", "it_4", "it_0", "it_5"]
OUTCOMES = [True, True, False, True, True, False]


def bank_arrays(order: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(a, b, c)`` for ``order``, built independently of the script under test."""
    by_id = {row["item_id"]: row for row in BANK}
    return (
        np.asarray([by_id[i]["discrimination"] for i in order], dtype=float),
        np.asarray([by_id[i]["difficulty"] for i in order], dtype=float),
        np.asarray([by_id[i]["guessing"] for i in order], dtype=float),
    )


def write_bank(root: Path, *, bank: list[dict[str, Any]] | None = None) -> Path:
    """Write a ``params.json`` in the vendored shape and return its directory."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "params.json").write_text(
        json.dumps(BANK if bank is None else bank, indent=2), encoding="utf-8"
    )
    return root


def bank_sha256(root: Path) -> str:
    """The digest ``vendor_bank`` would have recorded for this ``params.json``."""
    text = (root / "params.json").read_text(encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_report(
    root: Path,
    *,
    order: list[str] | None = None,
    outcomes: list[bool] | None = None,
    se_threshold: float = 0.3,
    min_items: int = 8,
    benchmark: str = "toy_bank",
    checkpoint: str = "s3://bucket/run/step_1000",
    provenance: dict[str, Any] | None = None,
    name: str = "cat_report.json",
    **overrides: Any,
) -> Path:
    """Write a ``cat_report.json`` whose recorded ability is the genuine EAP endpoint.

    The final theta and standard error are produced by calling :func:`eap_theta_se` here
    rather than being written by hand, so a passing replay means the script reproduced a
    number the estimator really returns for this pattern -- not that two constants in
    this file happen to agree.
    """
    order = ORDER if order is None else order
    outcomes = OUTCOMES if outcomes is None else outcomes
    if order:
        a, b, c = bank_arrays(order)
        resp = np.asarray([1.0 if ok else 0.0 for ok in outcomes], dtype=float)
        theta, se = eap_theta_se(resp, a, b, c)
    else:
        # What ``estimate_ability`` short-circuits to on no responses, which is not what
        # the grid would return -- see TestThePriorPoint.
        theta, se = PRIOR_THETA, PRIOR_SE

    metadata: dict[str, Any] = {
        "theta": theta,
        "standard_error": se,
        "bank_size": len(BANK),
        "stop_reason": "precision_reached",
        "selected_item_ids": list(order),
        "cat_settings": {
            "se_threshold": se_threshold,
            "min_items": min_items,
            "max_items": 40,
            "max_items_pinned_by_style": 40,
            "max_items_is_pinned_value": True,
        },
    }
    if provenance is not None:
        metadata["bank_provenance"] = provenance

    report: dict[str, Any] = {
        "cat_style": "uni_mcq",
        "benchmark": benchmark,
        "ability": {"theta": theta, "standard_error": se, "metadata": {}},
        "num_items_administered": len(order),
        "responses": [
            {
                "item_id": item_id,
                "chosen_index": 0 if ok else 1,
                "correct": ok,
                "choice_logprobs": [],
                "metadata": {},
            }
            for item_id, ok in zip(order, outcomes, strict=True)
        ],
        "metadata": metadata,
        "run": {"cat_style": "uni_mcq", "checkpoint": checkpoint},
    }
    for key, value in overrides.items():
        report[key] = value

    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def replay_one(report_path: Path, bank_dir: Path, **kwargs: Any) -> replay_mod.ReplayedRun:
    """Replay one report against ``bank_dir``, with the script's own defaults."""
    options: dict[str, Any] = {
        "bank_dir": bank_dir,
        "tolerance": DEFAULT_TOLERANCE,
        "se_threshold": None,
    }
    options.update(kwargs)
    return replay_mod.replay_report(report_path, **options)


@pytest.fixture
def bank_dir(tmp_path: Path) -> Path:
    return write_bank(tmp_path / "bank")


@pytest.fixture
def report(tmp_path: Path) -> Path:
    return make_report(tmp_path / "results")


class TestTheReplayReproducesTheReport:
    """The property everything else rests on: the endpoint lands where it was recorded."""

    def test_the_replayed_final_estimate_matches_the_recorded_one(
        self, report: Path, bank_dir: Path
    ) -> None:
        run = replay_one(report, bank_dir)

        assert run.theta_delta <= DEFAULT_TOLERANCE
        assert run.se_delta <= DEFAULT_TOLERANCE

    def test_it_matches_exactly_and_not_merely_within_tolerance(
        self, report: Path, bank_dir: Path
    ) -> None:
        """The tolerance exists for a different NumPy build, not for this one.

        Same interpreter, same arrays, same function: the two are the same float. If
        this ever starts failing while the test above passes, the replay has stopped
        being the identical computation and the tolerance is hiding it.
        """
        run = replay_one(report, bank_dir)

        assert run.final.theta == run.recorded_theta
        assert run.final.se == run.recorded_se

    def test_the_trajectory_starts_at_the_prior(self, report: Path, bank_dir: Path) -> None:
        """``estimate_ability`` returns (0.0, 1.0) on no responses, which is where the
        standard-error curve has to start for the fall from the prior to be visible."""
        first = replay_one(report, bank_dir).trajectory[0]

        assert (first.item_index, first.item_id, first.correct) == (0, None, None)
        assert (first.theta, first.se) == (PRIOR_THETA, PRIOR_SE)

    def test_there_is_one_point_per_item_plus_the_prior(self, report: Path, bank_dir: Path) -> None:
        run = replay_one(report, bank_dir)

        assert len(run.trajectory) == len(ORDER) + 1
        assert run.n_items == len(ORDER)

    def test_a_session_that_administered_nothing_is_the_prior_alone(
        self, tmp_path: Path, bank_dir: Path
    ) -> None:
        path = make_report(tmp_path / "empty", order=[], outcomes=[])

        run = replay_one(path, bank_dir)

        assert run.n_items == 0
        assert (run.final.theta, run.final.se) == (PRIOR_THETA, PRIOR_SE)


class TestThePriorPoint:
    """Step 0 is the style's declared prior, and that is not the grid's own answer.

    ``estimate_ability`` short-circuits an empty response list to exactly ``(0.0, 1.0)``
    (``../style.py:491-492``) rather than calling the estimator, and the two do not
    agree: EAP over no responses returns the standard-normal prior as discretized on 81
    nodes truncated to [-4, 4], whose standard deviation is a little under 1. The
    difference is around 4e-4 -- far too small to see on a figure and far too large to
    pass the 1e-9 endpoint check, which is exactly the situation in which a replay that
    picked the other convention would fail on a run that administered nothing.
    """

    def test_the_estimator_and_the_short_circuit_disagree(self) -> None:
        empty = np.asarray([], dtype=float)

        theta, se = eap_theta_se(empty, empty, empty, empty)

        assert theta == pytest.approx(PRIOR_THETA, abs=1e-9)
        assert se != pytest.approx(PRIOR_SE, abs=1e-9)
        assert se == pytest.approx(PRIOR_SE, abs=1e-3)

    def test_the_replay_uses_the_short_circuit(self, report: Path, bank_dir: Path) -> None:
        """So the trajectory's first point is the value a report would actually carry."""
        first = replay_one(report, bank_dir).trajectory[0]

        assert (first.theta, first.se) == (0.0, 1.0)


class TestTheReplayIsTheEstimatorOverEachPrefix:
    """Every intermediate point, recomputed independently.

    The endpoint check cannot see this: EAP is a product over responses and therefore
    order-invariant, so a replay that shuffled the sequence, or that conditioned step
    ``k`` on ``k + 1`` responses, would still land on the recorded final estimate. These
    are the assertions that pin the threading itself.
    """

    def test_each_point_is_eap_over_exactly_that_prefix(self, report: Path, bank_dir: Path) -> None:
        a, b, c = bank_arrays(ORDER)
        resp = np.asarray([1.0 if ok else 0.0 for ok in OUTCOMES], dtype=float)

        trajectory = replay_one(report, bank_dir).trajectory

        for k in range(1, len(ORDER) + 1):
            theta, se = eap_theta_se(resp[:k], a[:k], b[:k], c[:k])
            assert trajectory[k].theta == theta, f"theta drifted at prefix {k}"
            assert trajectory[k].se == se, f"se drifted at prefix {k}"

    def test_each_point_names_the_item_it_conditioned_on(
        self, report: Path, bank_dir: Path
    ) -> None:
        trajectory = replay_one(report, bank_dir).trajectory

        assert [point.item_id for point in trajectory[1:]] == ORDER
        assert [point.correct for point in trajectory[1:]] == OUTCOMES

    def test_a_permuted_bank_moves_the_intermediate_points(
        self, tmp_path: Path, bank_dir: Path
    ) -> None:
        """Proof that prefix alignment is doing work rather than being unfalsifiable.

        The same six responses in a different order reach the same endpoint -- which is
        why the endpoint check cannot police the order -- and take a visibly different
        route there.
        """
        forward = make_report(tmp_path / "fwd")
        backward = make_report(
            tmp_path / "rev", order=list(reversed(ORDER)), outcomes=list(reversed(OUTCOMES))
        )

        one = replay_one(forward, bank_dir)
        other = replay_one(backward, bank_dir)

        assert one.final.theta == pytest.approx(other.final.theta, abs=1e-12)
        midpoint = len(ORDER) // 2
        assert one.trajectory[midpoint].theta != pytest.approx(
            other.trajectory[midpoint].theta, abs=1e-3
        )


class TestAMismatchIsRefused:
    """The verification fires, and says what disagreed by how much."""

    def test_a_moved_theta_is_caught(self, tmp_path: Path, bank_dir: Path) -> None:
        path = make_report(tmp_path / "bad")
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["ability"]["theta"] += 1e-6
        path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ReplayError, match="does not reproduce the recorded final"):
            replay_one(path, bank_dir)

    def test_a_moved_standard_error_is_caught(self, tmp_path: Path, bank_dir: Path) -> None:
        """Checked separately because theta and SE fail independently: a bank whose
        discriminations moved but whose difficulties did not shifts the second and
        barely touches the first."""
        path = make_report(tmp_path / "bad")
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["ability"]["standard_error"] += 1e-6
        path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ReplayError, match="does not reproduce the recorded final"):
            replay_one(path, bank_dir)

    def test_the_message_names_both_numbers_and_the_tolerance(
        self, tmp_path: Path, bank_dir: Path
    ) -> None:
        """A failure a reader cannot act on gets worked around rather than fixed."""
        path = make_report(tmp_path / "bad")
        payload = json.loads(path.read_text(encoding="utf-8"))
        recorded = payload["ability"]["theta"]
        payload["ability"]["theta"] = recorded + 0.25
        path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ReplayError) as caught:
            replay_one(path, bank_dir)

        message = str(caught.value)
        assert repr(recorded + 0.25) in message
        assert "tolerance" in message
        assert "re-vendored" in message

    def test_a_bank_whose_parameters_moved_is_caught(self, tmp_path: Path, report: Path) -> None:
        """The failure this guard actually exists for. Re-vendoring is routine, and a
        replay against the new bank draws a plausible curve for a session that never
        saw those parameters."""
        moved = [dict(row) for row in BANK]
        moved[2]["difficulty"] += 0.4
        elsewhere = write_bank(tmp_path / "moved", bank=moved)

        with pytest.raises(ReplayError, match="does not reproduce the recorded final"):
            replay_one(report, elsewhere)

    def test_the_tolerance_is_the_boundary_that_is_applied(
        self, tmp_path: Path, bank_dir: Path
    ) -> None:
        """Both sides of it, so neither a tolerance that is ignored nor one that has
        been widened to always pass can survive."""
        path = make_report(tmp_path / "edge")
        payload = json.loads(path.read_text(encoding="utf-8"))
        recorded = payload["ability"]["theta"]

        payload["ability"]["theta"] = recorded + DEFAULT_TOLERANCE / 2
        path.write_text(json.dumps(payload), encoding="utf-8")
        assert replay_one(path, bank_dir).theta_delta <= DEFAULT_TOLERANCE

        payload["ability"]["theta"] = recorded + DEFAULT_TOLERANCE * 10
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ReplayError):
            replay_one(path, bank_dir)

    def test_the_default_tolerance_is_far_below_a_real_defect(self) -> None:
        """One swapped parameter row against the tolerance meant to catch it. The gap is
        the justification for 1e-9 reduced to a number."""
        a, b, c = bank_arrays(ORDER)
        resp = np.asarray([1.0 if ok else 0.0 for ok in OUTCOMES], dtype=float)
        honest, _ = eap_theta_se(resp, a, b, c)

        swapped_b = b.copy()
        swapped_b[0], swapped_b[1] = swapped_b[1], swapped_b[0]
        wrong, _ = eap_theta_se(resp, a, swapped_b, c)

        assert abs(honest - wrong) > 1e4 * DEFAULT_TOLERANCE


class TestTheReportHasToDescribeOneSession:
    """Internal disagreements, caught before they become an alignment nobody checked."""

    def test_selected_item_ids_disagreeing_with_the_responses_is_refused(
        self, tmp_path: Path, bank_dir: Path
    ) -> None:
        path = make_report(tmp_path / "split")
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["metadata"]["selected_item_ids"] = list(reversed(ORDER))
        path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ReplayError, match="two of them"):
            replay_one(path, bank_dir)

    def test_a_miscounted_session_is_refused(self, tmp_path: Path, bank_dir: Path) -> None:
        path = make_report(tmp_path / "miscount")
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["num_items_administered"] = len(ORDER) + 1
        path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ReplayError, match="does not describe one session"):
            replay_one(path, bank_dir)

    def test_an_item_the_bank_does_not_hold_is_refused(
        self, tmp_path: Path, bank_dir: Path
    ) -> None:
        """Named rather than skipped, because an item silently dropped from a prefix
        shifts every estimate after it and the endpoint check would call the result a
        mismatch without saying which item went missing."""
        path = make_report(tmp_path / "stranger")
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["responses"].append(
            {"item_id": "it_99", "chosen_index": 0, "correct": True, "metadata": {}}
        )
        payload["num_items_administered"] = len(payload["responses"])
        payload["metadata"]["selected_item_ids"] = [row["item_id"] for row in payload["responses"]]
        path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ReplayError, match="it_99"):
            replay_one(path, bank_dir)

    def test_something_that_is_not_a_report_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "cat_report.json"
        path.write_text(json.dumps({"tasks": [], "step": 1000}), encoding="utf-8")

        with pytest.raises(ReplayError, match="does not look like a cat_report.json"):
            replay_mod.load_report(path)


class TestTheBankDigest:
    """Phase 2's pin, read forward-compatibly: honoured where present, warned where not."""

    def test_a_matching_digest_replays(self, tmp_path: Path, bank_dir: Path) -> None:
        path = make_report(
            tmp_path / "pinned",
            provenance={"sha256": {"params.json": bank_sha256(bank_dir)}},
        )

        run = replay_one(path, bank_dir)

        assert run.bank_hash_note.startswith("pinned")

    def test_a_bare_string_digest_is_also_honoured(self, tmp_path: Path, bank_dir: Path) -> None:
        path = make_report(tmp_path / "pinned", provenance={"sha256": bank_sha256(bank_dir)})

        assert replay_one(path, bank_dir).bank_hash_note.startswith("pinned")

    def test_a_digest_that_does_not_match_is_refused(self, tmp_path: Path, bank_dir: Path) -> None:
        """Refused before the replay rather than after, because a re-vendoring that
        preserved the endpoint would otherwise pass verification and plot."""
        path = make_report(tmp_path / "stale", provenance={"sha256": {"params.json": "0" * 64}})

        with pytest.raises(ReplayError, match="Bank mismatch"):
            replay_one(path, bank_dir)

    def test_an_unpinned_report_warns_and_replays(
        self, report: Path, bank_dir: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Every report written before phase 2 is in this state, so it cannot be an
        error -- but an unpinned replay is not a checked one and has to say so."""
        with caplog.at_level("WARNING"):
            run = replay_one(report, bank_dir)

        assert run.bank_hash_note == "unpinned"
        assert "sha256" in caplog.text

    def test_the_digest_is_over_normalized_text_not_file_bytes(self, bank_dir: Path) -> None:
        """``vendor_bank`` hashes the string it is about to write while ``write_text``
        translates newlines on the way to disk, so on Windows the two differ and hashing
        bytes would report every committed bank as tampered with."""
        params = bank_dir / "params.json"
        params.write_bytes(params.read_text(encoding="utf-8").replace("\n", "\r\n").encode())

        assert replay_mod.bank_digest(params) == bank_sha256(bank_dir)


class TestNCross:
    """The item at which SE first reached the target, which is not the test length."""

    def test_it_is_the_first_index_at_or_under_the_threshold(self) -> None:
        trajectory = trace([1.0, 0.62, 0.44, 0.31, 0.29, 0.27, 0.26])

        assert first_crossing(trajectory, 0.3) == 4

    def test_the_floor_makes_it_differ_from_the_test_length(
        self, tmp_path: Path, bank_dir: Path
    ) -> None:
        """The case ``CAT_METRICS.md`` says is large on bbh and currently invisible: a
        posterior that collapsed early and a session that kept going to its floor.
        ``stop_reason`` reads ``precision_reached`` either way, so the distance between
        the two numbers is the only record that the floor, not precision, set the
        length."""
        path = make_report(tmp_path / "floored", se_threshold=0.75, min_items=6)

        run = replay_one(path, bank_dir)

        assert run.n_cross is not None
        assert run.n_cross < run.n_items
        assert run.se_threshold == 0.75

    def test_it_equals_the_test_length_when_precision_stopped_the_run(self) -> None:
        trajectory = trace([1.0, 0.7, 0.5, 0.38, 0.28])

        assert first_crossing(trajectory, 0.3) == 4
        assert len(trajectory) - 1 == 4

    def test_it_is_none_when_the_target_is_never_reached(
        self, tmp_path: Path, bank_dir: Path
    ) -> None:
        path = make_report(tmp_path / "never", se_threshold=0.001)

        assert replay_one(path, bank_dir).n_cross is None

    def test_it_is_none_when_the_run_recorded_no_threshold(
        self, tmp_path: Path, bank_dir: Path
    ) -> None:
        path = make_report(tmp_path / "unset")
        payload = json.loads(path.read_text(encoding="utf-8"))
        del payload["metadata"]["cat_settings"]["se_threshold"]
        path.write_text(json.dumps(payload), encoding="utf-8")

        run = replay_one(path, bank_dir)

        assert run.se_threshold is None
        assert run.n_cross is None

    def test_the_prior_point_is_never_the_crossing(self) -> None:
        """It is not an administered item. A threshold at or above the prior's SE of 1
        would otherwise report that precision was reached before any question was asked."""
        assert first_crossing(trace([1.0, 0.9, 0.8]), 1.0) == 1

    def test_an_override_measures_a_different_target(self, tmp_path: Path, bank_dir: Path) -> None:
        """The threshold is a property of the run, but asking where a *different* one
        would have stopped it is the cheapest version of a stopping-rule sweep."""
        path = make_report(tmp_path / "override", se_threshold=0.001)

        run = replay_one(path, bank_dir, se_threshold=0.9)

        assert run.n_cross is not None


def trace(errors: list[float]) -> list[replay_mod.TrajectoryPoint]:
    """A trajectory carrying only the standard errors, for the crossing tests."""
    return [
        replay_mod.TrajectoryPoint(
            item_index=index,
            item_id=None if index == 0 else f"it_{index}",
            correct=None if index == 0 else True,
            theta=0.0,
            se=value,
        )
        for index, value in enumerate(errors)
    ]


def make_mwle_report(root: Path, **kwargs: Any) -> tuple[Path, float, float]:
    """A report as a run launched under ``batch_eap+mwle`` would have written it.

    ``ability`` carries Warm's estimate, ``metadata.theta_batch`` carries the EAP one.
    Both are computed here from the estimators themselves rather than written by hand,
    so the fixture is a report those functions really produce.

    Returns ``(path, theta_batch, theta_mwle)``.
    """
    a, b, c = bank_arrays(ORDER)
    resp = np.asarray([1.0 if ok else 0.0 for ok in OUTCOMES], dtype=float)
    theta_batch, se_batch = eap_theta_se(resp, a, b, c)
    mwle = mwle_theta_se(resp, a, b, c, theta0=theta_batch)
    assert mwle.converged, mwle.note

    path = make_report(root, **kwargs)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["ability"] = {
        "theta": mwle.theta,
        "standard_error": mwle.standard_error,
        "metadata": {},
    }
    payload["metadata"].update(
        {
            "theta": mwle.theta,
            "standard_error": mwle.standard_error,
            "ability_estimator": "batch_eap+mwle",
            "ability_estimator_reported": "batch_eap+mwle",
            "theta_online": theta_batch,
            "se_online": se_batch,
            "theta_batch": theta_batch,
            "se_batch": se_batch,
            "theta_mwle": mwle.theta,
            "se_mwle": mwle.standard_error,
            "mwle_ok": True,
        }
    )
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path, theta_batch, mwle.theta


class TestTheOfflineMwleRefit:
    """What MWLE would have said about a run that was reported under EAP.

    The point of doing this here rather than by re-running: every report already on disk
    predates ``--ability-estimator``, and MWLE is a pure function of the responses and
    the item parameters, both of which the report and the bank already hold. So the
    comparison costs nothing -- no checkpoint, no GPU, no network -- and it is the only
    way to price the toggle before paying for a run under it.
    """

    def test_it_reports_the_estimator_the_run_used(self, report: Path, bank_dir: Path) -> None:
        """A report written before the flag existed carries no estimator field, and the
        only estimator that existed then was EAP."""
        assert replay_one(report, bank_dir).reported_estimator == "batch_eap"

    def test_the_refit_is_the_estimator_over_the_same_inputs(
        self, report: Path, bank_dir: Path
    ) -> None:
        a, b, c = bank_arrays(ORDER)
        resp = np.asarray([1.0 if ok else 0.0 for ok in OUTCOMES], dtype=float)
        run = replay_one(report, bank_dir)

        expected = mwle_theta_se(resp, a, b, c, theta0=run.theta_batch)

        assert run.mwle.converged
        assert run.mwle.theta == expected.theta

    def test_theta_batch_is_the_verified_endpoint(self, report: Path, bank_dir: Path) -> None:
        run = replay_one(report, bank_dir)

        assert run.theta_batch == run.final.theta
        assert run.theta_batch == run.recorded_theta

    def test_a_session_that_administered_nothing_has_no_mwle_estimate(
        self, tmp_path: Path, bank_dir: Path
    ) -> None:
        run = replay_one(make_report(tmp_path / "empty", order=[], outcomes=[]), bank_dir)

        assert run.mwle.converged is False
        assert "nothing was administered" in run.mwle.note

    def test_the_refit_is_seeded_at_the_verified_endpoint(
        self, report: Path, bank_dir: Path
    ) -> None:
        """Not at the report's ``ability``, which under MWLE is Warm's estimate and not
        an EAP one. Seeding cannot move the answer, but seeding from an unverified number
        would mean the refit described a session the replay had not checked."""
        run = replay_one(report, bank_dir)
        a, b, c = bank_arrays(ORDER)
        resp = np.asarray([1.0 if ok else 0.0 for ok in OUTCOMES], dtype=float)

        assert run.mwle.theta == pytest.approx(
            mwle_theta_se(resp, a, b, c, theta0=-3.0).theta, abs=1e-9
        )


class TestReplayingAnMwleRun:
    """A report whose headline number is Warm's, not EAP's.

    Verification has to compare like with like. EAP over prefixes converges to the EAP
    endpoint, so checking it against a published MWLE theta would reject every such run
    as a bad reconstruction -- the reports record the EAP endpoint separately for exactly
    this reason.
    """

    def test_it_verifies_against_the_recorded_batch_eap(
        self, tmp_path: Path, bank_dir: Path
    ) -> None:
        path, theta_batch, theta_mwle = make_mwle_report(tmp_path / "mwle")

        run = replay_one(path, bank_dir)

        assert theta_mwle != pytest.approx(theta_batch, abs=1e-6)
        assert run.recorded_theta == theta_batch
        assert run.theta_delta == 0.0
        assert run.reported_estimator == "batch_eap+mwle"

    def test_the_recorded_mwle_is_checked_against_the_refit(
        self, tmp_path: Path, bank_dir: Path
    ) -> None:
        path, _, theta_mwle = make_mwle_report(tmp_path / "mwle")

        run = replay_one(path, bank_dir)

        assert run.mwle.theta == theta_mwle
        assert run.mwle_delta == 0.0

    def test_a_recorded_mwle_that_does_not_reproduce_is_refused(
        self, tmp_path: Path, bank_dir: Path
    ) -> None:
        """The same guard :func:`verify_replay` applies to the EAP path, applied to the
        estimator that actually produced this run's published number."""
        path, _, theta_mwle = make_mwle_report(tmp_path / "mwle")
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["metadata"]["theta_mwle"] = theta_mwle + 0.25
        path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ReplayError, match="does not reproduce the recorded one"):
            replay_one(path, bank_dir)

    def test_a_fallback_report_is_not_cross_checked(self, tmp_path: Path, bank_dir: Path) -> None:
        """``mwle_ok: false`` means ``theta_mwle`` holds the EAP value the run fell back
        to, so there is no MWLE estimate in it for the refit to disagree with."""
        path, theta_batch, _ = make_mwle_report(tmp_path / "fallback")
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["ability"] = {
            "theta": theta_batch,
            "standard_error": payload["metadata"]["se_batch"],
            "metadata": {},
        }
        payload["metadata"]["mwle_ok"] = False
        payload["metadata"]["theta_mwle"] = theta_batch
        payload["metadata"]["ability_estimator_reported"] = "batch_eap"
        path.write_text(json.dumps(payload), encoding="utf-8")

        run = replay_one(path, bank_dir)

        assert run.mwle_delta is None
        assert run.mwle.converged is True
        assert run.reported_estimator == "batch_eap"

    def test_a_moved_batch_eap_is_still_caught(self, tmp_path: Path, bank_dir: Path) -> None:
        """Preferring the metadata pair must not weaken the endpoint check."""
        path, _, _ = make_mwle_report(tmp_path / "mwle")
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["metadata"]["theta_batch"] += 1e-6
        path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ReplayError, match="does not reproduce the recorded final"):
            replay_one(path, bank_dir)


class TestTheCsv:
    """The primary artifact. Figures are optional; this is not."""

    def test_it_has_the_documented_columns(self, tmp_path: Path, bank_dir: Path) -> None:
        rows = read_csv(run_cli(tmp_path, bank_dir))

        assert list(rows[0]) == list(replay_mod.CSV_COLUMNS)

    def test_there_is_one_row_per_item_plus_the_prior(self, tmp_path: Path, bank_dir: Path) -> None:
        rows = read_csv(run_cli(tmp_path, bank_dir))

        assert [int(row["item_index"]) for row in rows] == list(range(len(ORDER) + 1))
        assert [row["item_id"] for row in rows[1:]] == ORDER
        assert [row["correct"] for row in rows[1:]] == ["1" if ok else "0" for ok in OUTCOMES]

    def test_the_prior_row_names_no_item(self, tmp_path: Path, bank_dir: Path) -> None:
        first = read_csv(run_cli(tmp_path, bank_dir))[0]

        assert (first["item_id"], first["correct"]) == ("", "")
        assert (float(first["theta"]), float(first["se"])) == (PRIOR_THETA, PRIOR_SE)

    def test_theta_and_se_round_trip_exactly(self, tmp_path: Path, bank_dir: Path) -> None:
        """The figures are drawn from these rows, so a value rounded on the way out
        would mean the plotted series and the verified one were not the same series."""
        rows = read_csv(run_cli(tmp_path, bank_dir))
        expected = replay_one(tmp_path / "results" / "cat_report.json", bank_dir)

        assert [float(row["theta"]) for row in rows] == [
            point.theta for point in expected.trajectory
        ]
        assert [float(row["se"]) for row in rows] == [point.se for point in expected.trajectory]

    def test_the_run_level_columns_are_carried_on_every_row(
        self, tmp_path: Path, bank_dir: Path
    ) -> None:
        rows = read_csv(run_cli(tmp_path, bank_dir))

        assert {row["n_items_administered"] for row in rows} == {str(len(ORDER))}
        assert {row["se_threshold"] for row in rows} == {"0.3"}
        assert {row["benchmark"] for row in rows} == {"toy_bank"}
        assert {row["checkpoint"] for row in rows} == {"s3://bucket/run/step_1000"}

    def test_the_mwle_refit_is_carried_on_every_row(self, tmp_path: Path, bank_dir: Path) -> None:
        """One number per run, denormalized like ``n_cross``: there is no per-step MWLE
        to plot, and a second file to join against would mean nobody reads it."""
        rows = read_csv(run_cli(tmp_path, bank_dir))
        expected = replay_one(tmp_path / "results" / "cat_report.json", bank_dir)

        assert {row["mwle_ok"] for row in rows} == {"1"}
        assert {float(row["theta_mwle"]) for row in rows} == {expected.mwle.theta}

    def test_a_run_with_no_mwle_estimate_leaves_the_column_empty(
        self, tmp_path: Path, bank_dir: Path
    ) -> None:
        """Empty rather than the EAP value it fell back to, which would read as an MWLE
        number that happened to agree."""
        rows = read_csv(run_cli(tmp_path, bank_dir, order=[], outcomes=[]))

        assert {row["theta_mwle"] for row in rows} == {""}
        assert {row["mwle_ok"] for row in rows} == {"0"}


class TestSeveralReports:
    """Enough to overlay a checkpoint sweep later, without aggregating one now."""

    def test_two_reports_land_in_one_table(self, tmp_path: Path, bank_dir: Path) -> None:
        first = make_report(tmp_path / "a", checkpoint="s3://b/step_1000")
        second = make_report(
            tmp_path / "b", checkpoint="s3://b/step_2000", order=ORDER[:4], outcomes=OUTCOMES[:4]
        )

        code = replay_mod.main(
            [
                str(first),
                str(second),
                "--bank-dir",
                str(bank_dir),
                "--out-dir",
                str(tmp_path / "out"),
            ]
        )
        rows = read_csv(tmp_path / "out" / "cat_trajectory.csv")

        assert code == 0
        assert {row["checkpoint"] for row in rows} == {"s3://b/step_1000", "s3://b/step_2000"}
        assert len(rows) == (len(ORDER) + 1) + 5

    def test_a_directory_is_walked_for_reports(self, tmp_path: Path, bank_dir: Path) -> None:
        make_report(tmp_path / "sweep" / "step_1000")
        make_report(tmp_path / "sweep" / "step_2000")

        code = replay_mod.main(
            [
                str(tmp_path / "sweep"),
                "--bank-dir",
                str(bank_dir),
                "--out-dir",
                str(tmp_path / "out"),
            ]
        )

        assert code == 0
        assert len(read_csv(tmp_path / "out" / "cat_trajectory.csv")) == 2 * (len(ORDER) + 1)

    def test_a_directory_holding_no_report_is_an_error(
        self, tmp_path: Path, bank_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (tmp_path / "empty").mkdir()

        code = replay_mod.main(
            [str(tmp_path / "empty"), "--bank-dir", str(bank_dir), "--out-dir", str(tmp_path)]
        )

        assert code == 1
        assert "No cat_report.json" in capsys.readouterr().err

    def test_one_bad_report_does_not_take_the_good_ones_with_it(
        self, tmp_path: Path, bank_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A sweep should name every failure in one pass. The exit status is still
        non-zero, because a partial result that looks complete is the whole hazard."""
        good = make_report(tmp_path / "good")
        bad = make_report(tmp_path / "bad")
        payload = json.loads(bad.read_text(encoding="utf-8"))
        payload["ability"]["theta"] += 0.5
        bad.write_text(json.dumps(payload), encoding="utf-8")

        code = replay_mod.main(
            [str(good), str(bad), "--bank-dir", str(bank_dir), "--out-dir", str(tmp_path / "o")]
        )
        printed = capsys.readouterr().out

        assert code == 1
        assert "1 report(s) FAILED and were not plotted" in printed
        assert len(read_csv(tmp_path / "o" / "cat_trajectory.csv")) == len(ORDER) + 1

    def test_a_report_that_fails_contributes_no_rows(self, tmp_path: Path, bank_dir: Path) -> None:
        bad = make_report(tmp_path / "bad")
        payload = json.loads(bad.read_text(encoding="utf-8"))
        payload["ability"]["standard_error"] += 0.5
        bad.write_text(json.dumps(payload), encoding="utf-8")

        code = replay_mod.main(
            [str(bad), "--bank-dir", str(bank_dir), "--out-dir", str(tmp_path / "o")]
        )

        assert code == 1
        assert not (tmp_path / "o" / "cat_trajectory.csv").exists()


class TestWithoutMatplotlib:
    """Graceful degradation, which is the normal case rather than the edge case."""

    def test_the_figure_import_returns_none_rather_than_raising(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``None`` in ``sys.modules`` is the sentinel that makes an import raise, so
        this holds whether or not the machine running it happens to have matplotlib."""
        monkeypatch.setitem(sys.modules, "matplotlib", None)

        assert replay_mod.pyplot() is None

    def test_the_csv_is_still_written_and_the_run_succeeds(
        self, tmp_path: Path, bank_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(replay_mod, "pyplot", lambda: None)
        report = make_report(tmp_path / "results")

        code = replay_mod.main(
            [str(report), "--bank-dir", str(bank_dir), "--out-dir", str(tmp_path / "out")]
        )

        assert code == 0
        assert read_csv(tmp_path / "out" / "cat_trajectory.csv")
        assert not list((tmp_path / "out").glob("*.png"))

    def test_the_absence_is_reported_rather_than_passed_over(
        self,
        tmp_path: Path,
        bank_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(replay_mod, "pyplot", lambda: None)
        report = make_report(tmp_path / "results")

        replay_mod.main(
            [str(report), "--bank-dir", str(bank_dir), "--out-dir", str(tmp_path / "out")]
        )

        assert "matplotlib is not installed" in capsys.readouterr().out

    def test_render_figures_draws_nothing_without_it(
        self, tmp_path: Path, bank_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(replay_mod, "pyplot", lambda: None)
        run = replay_one(make_report(tmp_path / "r"), bank_dir)

        assert replay_mod.render_figures(run, replay_mod.rows_for(run), tmp_path / "out") == []

    def test_no_figure_is_emitted_for_a_run_that_failed_verification(
        self, tmp_path: Path, bank_dir: Path
    ) -> None:
        """Stated separately from the CSV because it is the requirement: a figure whose
        trajectory does not end where the report says is worse than no figure."""
        monkey = FakePyplot()
        bad = make_report(tmp_path / "bad")
        payload = json.loads(bad.read_text(encoding="utf-8"))
        payload["ability"]["theta"] += 0.5
        bad.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(replay_mod, "pyplot", lambda: monkey)
            code = replay_mod.main(
                [str(bad), "--bank-dir", str(bank_dir), "--out-dir", str(tmp_path / "o")]
            )

        assert code == 1
        assert monkey.saved == []


class FakeAxes:
    """Records the calls a figure makes, so the plotting path can be tested unplotted."""

    def __init__(self) -> None:
        self.title = ""
        self.labels: dict[str, str] = {}

    def set_title(self, text: str, **_: Any) -> None:
        self.title = text

    def set_xlabel(self, text: str, **_: Any) -> None:
        self.labels["x"] = text

    def set_ylabel(self, text: str, **_: Any) -> None:
        self.labels["y"] = text

    def __getattr__(self, _name: str) -> Any:
        return lambda *args, **kwargs: None


class FakeFigure:
    def __init__(self, owner: FakePyplot, axes: FakeAxes) -> None:
        self.owner = owner
        self.axes_object = axes

    def savefig(self, path: Path, **kwargs: Any) -> None:
        self.owner.saved.append((Path(path), self.axes_object.title, kwargs))

    def tight_layout(self, **_: Any) -> None:
        return None


class FakePyplot:
    """The narrowest stand-in for pyplot this script uses.

    Matplotlib is not installed and the tests may not require it, but "renders two
    figures with distinguishable titles" is a requirement that has to be checked
    somewhere. A stub checks the part that is ours -- which files, which titles, which
    axis labels -- without pretending to check that matplotlib draws.
    """

    def __init__(self) -> None:
        self.saved: list[tuple[Path, str, dict[str, Any]]] = []
        self.axes: list[FakeAxes] = []

    def subplots(self, **_: Any) -> tuple[FakeFigure, FakeAxes]:
        axes = FakeAxes()
        self.axes.append(axes)
        return FakeFigure(self, axes), axes

    def close(self, _figure: Any) -> None:
        return None


class TestTheFigures:
    """What is drawn when matplotlib is there, checked through a stub."""

    def test_two_figures_are_written_per_run(
        self, tmp_path: Path, bank_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakePyplot()
        monkeypatch.setattr(replay_mod, "pyplot", lambda: fake)
        run = replay_one(make_report(tmp_path / "r"), bank_dir)

        written = replay_mod.render_figures(run, replay_mod.rows_for(run), tmp_path / "out")

        assert [path.name for path in written] == [
            "toy_bank_step_1000_se_curve.png",
            "toy_bank_step_1000_theta_trajectory.png",
        ]
        assert all(kwargs["dpi"] == 150 for _, _, kwargs in fake.saved)

    def test_both_titles_say_the_figure_is_one_run(self, tmp_path: Path, bank_dir: Path) -> None:
        """``CAT_METRICS.md`` notes the published ``se_reduction_curve.png`` is a mean
        over 115 models. Ours is one trace for one checkpoint, and a caption that does
        not say so invites the two to be read as the same figure."""
        run = replay_one(make_report(tmp_path / "r"), bank_dir)

        se_title, theta_title = replay_mod.figure_titles(run)

        assert SINGLE_RUN_NOTE in se_title
        assert SINGLE_RUN_NOTE in theta_title
        assert se_title != theta_title
        for title in (se_title, theta_title):
            assert "toy_bank" in title
            assert "step_1000" in title
            assert f"{len(ORDER):,} of {len(BANK):,} items" in title

    def test_the_axes_are_labelled_the_way_the_precedent_labels_them(
        self, tmp_path: Path, bank_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakePyplot()
        monkeypatch.setattr(replay_mod, "pyplot", lambda: fake)
        run = replay_one(make_report(tmp_path / "r"), bank_dir)

        replay_mod.render_figures(run, replay_mod.rows_for(run), tmp_path / "out")

        assert [axes.labels["x"] for axes in fake.axes] == [
            "items administered",
            "items administered",
        ]
        assert fake.axes[0].labels["y"] == "posterior SE (logits)"

    def test_two_runs_of_one_dataset_do_not_overwrite_each_other(
        self, tmp_path: Path, bank_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(replay_mod, "pyplot", lambda: FakePyplot())
        first = replay_one(make_report(tmp_path / "a", checkpoint="ckpt"), bank_dir)
        second = replay_one(make_report(tmp_path / "b", checkpoint="ckpt"), bank_dir)
        taken: set[str] = set()

        one = replay_mod.render_figures(
            first, replay_mod.rows_for(first), tmp_path / "out", taken=taken
        )
        other = replay_mod.render_figures(
            second, replay_mod.rows_for(second), tmp_path / "out", taken=taken
        )

        assert {path.name for path in one}.isdisjoint({path.name for path in other})


class TestAgainstACommittedBank:
    """One pass through the real resolver and the real loader, on a real ``params.json``.

    Everything above replays a bank this file wrote. This replays the 650-item ARC bank
    that is actually committed, resolved by benchmark name the way the script resolves it
    with no ``--bank-dir``, which is the only place the allowlist, the manifest and the
    JSON on disk are exercised together.
    """

    @pytest.fixture
    def arc_items(self) -> list[str]:
        params = CALIBRATED_DATASETS / "arc_challenge" / "params.json"
        if not params.is_file():
            pytest.skip("arc_challenge has not been vendored")
        records = json.loads(params.read_text(encoding="utf-8"))
        return [str(record["item_id"]) for record in records[:12]]

    def test_a_report_naming_a_real_benchmark_resolves_and_replays(
        self, tmp_path: Path, arc_items: list[str]
    ) -> None:
        from ....common.irt_params import load_irt_params

        params = CALIBRATED_DATASETS / "arc_challenge" / "params.json"
        bank = load_irt_params(params)
        outcomes = [index % 3 != 2 for index, _ in enumerate(arc_items)]
        resp = np.asarray([1.0 if ok else 0.0 for ok in outcomes], dtype=float)
        rows = [bank.params[item_id] for item_id in arc_items]
        theta, se = eap_theta_se(
            resp,
            np.asarray([float(row.discrimination) for row in rows]),
            np.asarray([float(row.difficulty) for row in rows]),
            np.asarray([float(row.guessing) for row in rows]),
        )

        path = tmp_path / "cat_report.json"
        path.write_text(
            json.dumps(
                {
                    "cat_style": "uni_mcq",
                    "benchmark": "arc_challenge",
                    "ability": {"theta": theta, "standard_error": se, "metadata": {}},
                    "num_items_administered": len(arc_items),
                    "responses": [
                        {"item_id": item_id, "chosen_index": 0, "correct": ok, "metadata": {}}
                        for item_id, ok in zip(arc_items, outcomes, strict=True)
                    ],
                    "metadata": {
                        "bank_size": len(bank),
                        "selected_item_ids": arc_items,
                        "cat_settings": {"se_threshold": 0.3, "min_items": 8},
                    },
                }
            ),
            encoding="utf-8",
        )

        run = replay_mod.replay_report(
            path, bank_dir=None, tolerance=DEFAULT_TOLERANCE, se_threshold=None
        )

        assert run.params_path == params
        assert run.final.theta == theta
        assert run.final.se == se
        assert run.n_items == len(arc_items)


def run_cli(tmp_path: Path, bank_dir: Path, **kwargs: Any) -> Path:
    """Write one report, run the CLI over it, and return the CSV it produced."""
    report = make_report(tmp_path / "results", **kwargs)
    out_dir = tmp_path / "out"
    code = replay_mod.main(
        [str(report), "--bank-dir", str(bank_dir), "--out-dir", str(out_dir), "--no-figures"]
    )
    assert code == 0, "the CLI refused a report these tests expect it to accept"
    return out_dir / "cat_trajectory.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
