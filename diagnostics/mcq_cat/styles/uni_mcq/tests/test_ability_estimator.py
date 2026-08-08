"""The reported ability estimator: the MWLE math, the toggle, and the fallback.

Three separate claims are under test here and they fail for different reasons, so they
are tested separately.

**The math.** MWLE maximizes ``ln L(theta) + 1/2 ln I(theta)``, and the solver finds the
root of that objective's derivative. Both halves of the derivative are written out by
hand in :mod:`..irt`, so :class:`TestTheSlopesAreTheDerivatives` checks them against
finite differences of the functions they claim to differentiate, and
:class:`TestTheSolverFindsTheMaximum` checks the root the solver returns against an
independent grid search over the objective itself. A sign error in either slope produces
a perfectly plausible theta a few tenths off, which nothing downstream would notice.

**The point of the estimator.** EAP's standard-normal prior pulls every estimate toward
zero, hardest where the response pattern is most extreme. MWLE drops that prior. So the
property worth pinning is not that the two estimators differ but that they differ *in
that direction*: :class:`TestMwleRemovesPriorShrinkage` requires ``|theta_mwle|`` to
exceed ``|theta_batch|`` on patterns from simulees the prior is compressing. An
implementation that dropped the prior and then reintroduced shrinkage some other way
would pass a mere inequality and fail this.

**The seam.** ``--ability-estimator`` moves the headline number and must not move
anything else. :class:`TestSelectionIsUnchanged` pins that both modes administer the
same items in the same order, which is what makes two runs of one checkpoint comparable
at all, and :class:`TestTheFallbackIsLoud` pins that a non-converged MWLE is reported as
the EAP estimate it fell back to rather than published as MWLE.

Torch-free: everything here is numpy, the frozen loaders and a simulated scorer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from .... import runner as runner_mod
from ....base import AbilityEstimate, CATState, ItemResponse
from ....common import cat_loop
from .. import irt as irt_mod
from .. import resolve
from .. import style as style_mod
from ..irt import (
    ESTIMATOR_BATCH_EAP,
    ESTIMATOR_BATCH_EAP_MWLE,
    ESTIMATORS,
    MWLE_BOUND,
    MWLE_RIDGE,
    MwleResult,
    eap_theta_se,
    fisher_info,
    information_slope,
    likelihood_slope,
    mwle_theta_se,
    penalized_slope,
    prob,
)
from ..style import UniMcqStyle
from .conftest import SimScorer, write_bank

#: A mixed 3PL fixture: a large guessing floor, a 2PL-shaped item, and a hard one. The
#: same shape ``test_irt_parity`` uses, so a failure here and a failure there point at
#: the same parameters.
A = np.array([1.5, 0.8, 2.2])
B = np.array([-0.5, 0.0, 1.0])
C = np.array([0.2, 0.0, 0.25])


def objective(
    resp: np.ndarray,
    theta: float,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    *,
    ridge: float = MWLE_RIDGE,
) -> float:
    """``ln L(theta) + 1/2 ln(I(theta) + ridge)``, written independently of the solver.

    The solver never evaluates this -- it works on the derivative -- so a grid search
    over it is a genuinely independent check on where the maximum is.
    """
    p = prob(theta, a, b, c)
    log_likelihood = float((resp * np.log(p) + (1 - resp) * np.log(1 - p)).sum())
    info = float(fisher_info(theta, a, b, c).sum()) + ridge
    return log_likelihood + 0.5 * float(np.log(info))


def two_pl_bank(n: int = 40, *, discrimination: float = 1.4) -> list[dict[str, Any]]:
    """A 2PL ladder spanning the difficulty range, with no guessing floor.

    ``c = 0`` on purpose. A guessing floor puts a hard ceiling on how far a low-ability
    estimate can fall, which is exactly the room the shrinkage tests need to measure in.
    """
    return [
        {
            "item_id": f"lad_{i}",
            "difficulty": -3.0 + 6.0 * i / (n - 1),
            "discrimination": discrimination,
            "guessing": 0.0,
        }
        for i in range(n)
    ]


def write_ladder(
    root: Path, params: list[dict[str, Any]], *, dataset: str = "arc_challenge"
) -> Path:
    """Write a vendored-shaped bank from ``params`` and return its directory."""
    bank_dir = root / dataset
    bank_dir.mkdir(parents=True, exist_ok=True)
    (bank_dir / "params.json").write_text(json.dumps(params), encoding="utf-8")
    (bank_dir / "items.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "id": row["item_id"],
                    "question": f"Q{index}",
                    "choices": ["a", "b", "c", "d"],
                    "gold_index": index % 4,
                }
            )
            + "\n"
            for index, row in enumerate(params)
        ),
        encoding="utf-8",
    )
    (bank_dir / "manifest.json").write_text(
        json.dumps({"fit_family": "2pl", "items": len(params)}), encoding="utf-8"
    )
    return bank_dir


def arrays(params: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(a, b, c)`` for a params list, in its own order."""
    return (
        np.asarray([row["discrimination"] for row in params], dtype=float),
        np.asarray([row["difficulty"] for row in params], dtype=float),
        np.asarray([row["guessing"] for row in params], dtype=float),
    )


@pytest.fixture
def ladder_style(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """A style over a 40-item 2PL ladder registered as ``arc_challenge``."""
    params = two_pl_bank()
    write_ladder(tmp_path, params)
    monkeypatch.setattr(resolve, "CALIBRATED_DATASETS", tmp_path)

    def build(estimator: str = ESTIMATOR_BATCH_EAP) -> UniMcqStyle:
        style = UniMcqStyle()
        style.set_ability_estimator(estimator)
        return style

    return build, params


def run_ladder(build, params: list[dict[str, Any]], *, estimator: str, true_theta: float):
    """Run the real engine over the ladder at ``true_theta`` and return the report."""
    style = build(estimator)
    bank = style.download_benchmark("arc_challenge")
    irt_bank = style.load_irt_params("arc_challenge")
    lookup = {
        row["item_id"]: (row["discrimination"], row["difficulty"], row["guessing"])
        for row in params
    }
    return cat_loop.run_cat(
        style,
        bank=bank,
        irt_bank=irt_bank,
        model=SimScorer(true_theta, lookup, seed=11),
        se_threshold=0.3,
        max_items=40,
    )


class TestTheSlopesAreTheDerivatives:
    """Both halves of Warm's score equation, against finite differences.

    Hand-differentiated 3PL algebra is where this implementation is most likely to be
    quietly wrong, and a wrong slope still converges -- to the wrong ability.
    """

    resp = np.array([1.0, 0.0, 1.0])
    step = 1e-6

    @pytest.mark.parametrize("theta", [-2.5, -1.0, -0.3, 0.0, 0.6, 1.8, 3.0])
    def test_likelihood_slope_differentiates_the_log_likelihood(self, theta: float) -> None:
        def log_likelihood(x: float) -> float:
            p = prob(x, A, B, C)
            return float((self.resp * np.log(p) + (1 - self.resp) * np.log(1 - p)).sum())

        finite = (log_likelihood(theta + self.step) - log_likelihood(theta - self.step)) / (
            2 * self.step
        )
        assert likelihood_slope(self.resp, theta, A, B, C) == pytest.approx(finite, abs=1e-6)

    @pytest.mark.parametrize("theta", [-2.5, -1.0, -0.3, 0.0, 0.6, 1.8, 3.0])
    def test_information_slope_differentiates_fisher_info(self, theta: float) -> None:
        def info(x: float) -> float:
            return float(fisher_info(x, A, B, C).sum())

        finite = (info(theta + self.step) - info(theta - self.step)) / (2 * self.step)
        assert information_slope(theta, A, B, C) == pytest.approx(finite, abs=1e-6)

    @pytest.mark.parametrize("theta", [-2.5, -1.0, -0.3, 0.0, 0.6, 1.8, 3.0])
    def test_penalized_slope_differentiates_the_objective(self, theta: float) -> None:
        finite = (
            objective(self.resp, theta + self.step, A, B, C)
            - objective(self.resp, theta - self.step, A, B, C)
        ) / (2 * self.step)
        assert penalized_slope(self.resp, theta, A, B, C) == pytest.approx(finite, abs=1e-6)

    def test_the_two_pl_reduction_of_the_likelihood_slope(self) -> None:
        """At ``c = 0`` the score is ``sum a_i (y_i - P_i)``, which is the textbook form.

        ``rtol`` and not ``abs``, at the same 1e-8 ``test_irt_parity`` uses for the
        Fisher reduction and for the same reason: both functions divide by ``1 - c +
        1e-9`` rather than by ``1 - c``, so the reduction is exact to that guard and no
        further.
        """
        zero_c = np.zeros(3)
        for theta in (-1.5, 0.0, 1.5):
            p = prob(theta, A, B, zero_c)
            assert likelihood_slope(self.resp, theta, A, B, zero_c) == pytest.approx(
                float((A * (self.resp - p)).sum()), rel=1e-8
            )

    def test_the_two_pl_reduction_of_the_information_slope(self) -> None:
        """At ``c = 0`` it is ``a^3 P (1 - P)(1 - 2P)`` -- the ``w (1 - 2p)`` weighting
        the multidimensional reference implementation used, in one dimension."""
        zero_c = np.zeros(3)
        for theta in (-1.5, 0.0, 1.5):
            p = prob(theta, A, B, zero_c)
            assert information_slope(theta, A, B, zero_c) == pytest.approx(
                float(((A**3) * p * (1 - p) * (1 - 2 * p)).sum()), rel=1e-8
            )


class TestTheSolverFindsTheMaximum:
    """The returned theta against a grid search over the objective the solver never sees."""

    @pytest.mark.parametrize(
        "pattern",
        [(1, 1, 1), (0, 0, 0), (1, 0, 1), (0, 1, 0), (1, 1, 0)],
        ids=["all-correct", "all-wrong", "mixed", "inverted", "two-of-three"],
    )
    def test_it_lands_on_the_grid_maximum(self, pattern: tuple[int, ...]) -> None:
        resp = np.asarray(pattern, dtype=float)
        theta0, _ = eap_theta_se(resp, A, B, C)
        result = mwle_theta_se(resp, A, B, C, theta0=theta0)

        assert result.converged, result.note
        grid = np.linspace(-MWLE_BOUND, MWLE_BOUND, 240_001)
        best = float(grid[int(np.argmax([objective(resp, float(t), A, B, C) for t in grid]))])
        assert result.theta == pytest.approx(best, abs=1e-3)

    def test_the_returned_theta_is_a_root_of_the_penalized_slope(self) -> None:
        resp = np.array([1.0, 0.0, 1.0])
        theta0, _ = eap_theta_se(resp, A, B, C)
        result = mwle_theta_se(resp, A, B, C, theta0=theta0)

        assert abs(penalized_slope(resp, result.theta, A, B, C)) < 1e-8

    def test_the_answer_does_not_depend_on_the_seed_it_started_from(self) -> None:
        """``theta0`` picks which side of the root the bracket opens on and nothing else.

        This is what makes "batch EAP + MWLE" a name for the procedure rather than a
        claim that the EAP value contributes evidence to the answer.
        """
        resp = np.array([1.0, 0.0, 1.0])
        answers = [mwle_theta_se(resp, A, B, C, theta0=seed).theta for seed in (-6.0, 0.0, 3.5)]

        assert answers[0] == pytest.approx(answers[1], abs=1e-9)
        assert answers[1] == pytest.approx(answers[2], abs=1e-9)

    def test_it_is_finite_on_an_all_wrong_pattern_where_plain_mle_diverges(self) -> None:
        """The information penalty is the only thing pinning this one: the likelihood
        alone keeps improving all the way to minus infinity."""
        resp = np.zeros(3)
        result = mwle_theta_se(resp, A, B, C, theta0=-1.0)

        assert result.converged
        assert np.isfinite(result.theta)
        assert -MWLE_BOUND < result.theta < 0.0

    def test_it_is_order_invariant(self) -> None:
        """The likelihood is a product over items, so a permutation cannot move it."""
        resp = np.array([1.0, 0.0, 1.0])
        order = np.array([2, 0, 1])
        forward = mwle_theta_se(resp, A, B, C, theta0=0.0)
        shuffled = mwle_theta_se(resp[order], A[order], B[order], C[order], theta0=0.0)

        assert forward.theta == pytest.approx(shuffled.theta, abs=1e-9)


class TestMwleRemovesPriorShrinkage:
    """The whole point: dropping the prior moves the estimate away from zero.

    Checked on both tails, because a prior pulls up as well as down and an
    implementation that only fixed one would still pass a low-ability test.
    """

    ladder = two_pl_bank(24)

    def estimates(self, pattern: np.ndarray) -> tuple[float, float]:
        a, b, c = arrays(self.ladder)
        theta_batch, _ = eap_theta_se(pattern, a, b, c)
        result = mwle_theta_se(pattern, a, b, c, theta0=theta_batch)
        assert result.converged, result.note
        return theta_batch, result.theta

    def test_a_low_ability_simulee_is_placed_further_below_zero(self) -> None:
        """Wrong on all but the two easiest items: a genuinely weak taker, whom EAP
        cannot place as low as the responses warrant because the prior is holding it up."""
        pattern = np.zeros(len(self.ladder))
        pattern[:2] = 1.0

        theta_batch, theta_mwle = self.estimates(pattern)

        assert theta_batch < 0.0
        assert theta_mwle < theta_batch
        assert abs(theta_mwle) > abs(theta_batch)

    def test_a_high_ability_simulee_is_placed_further_above_zero(self) -> None:
        pattern = np.ones(len(self.ladder))
        pattern[-2:] = 0.0

        theta_batch, theta_mwle = self.estimates(pattern)

        assert theta_batch > 0.0
        assert theta_mwle > theta_batch
        assert abs(theta_mwle) > abs(theta_batch)

    @pytest.mark.parametrize("correct", [1, 2, 3, 4, 5, 6])
    def test_the_gap_holds_across_the_whole_low_tail(self, correct: int) -> None:
        """Not one lucky pattern. Every weak simulee is compressed and every one is
        decompressed, so the direction is a property of the estimator."""
        pattern = np.zeros(len(self.ladder))
        pattern[:correct] = 1.0

        theta_batch, theta_mwle = self.estimates(pattern)

        assert abs(theta_mwle) > abs(theta_batch)

    def test_the_shrinkage_removed_grows_toward_the_tail(self) -> None:
        """The prior's pull is strongest where the pattern is most extreme, so the
        correction it needs is too. A constant offset would pass the tests above."""
        gaps = []
        for correct in (1, 4, 8):
            pattern = np.zeros(len(self.ladder))
            pattern[:correct] = 1.0
            theta_batch, theta_mwle = self.estimates(pattern)
            gaps.append(abs(theta_mwle) - abs(theta_batch))

        assert gaps[0] > gaps[1] > gaps[2] > 0.0

    def test_a_middling_simulee_is_barely_moved(self) -> None:
        """At the prior mean there is no inward pull to remove, so there should be
        almost nothing to correct -- which is what makes the tail numbers meaningful."""
        pattern = np.zeros(len(self.ladder))
        pattern[::2] = 1.0

        theta_batch, theta_mwle = self.estimates(pattern)

        assert abs(theta_mwle - theta_batch) < 0.15


class TestTheSolverRefusesRatherThanGuessing:
    """``converged=False`` in each documented case, with ``theta0`` handed back."""

    def test_an_empty_administered_set_is_not_an_estimate(self) -> None:
        result = mwle_theta_se(np.array([]), A, B, C, theta0=0.25)

        assert result.converged is False
        assert result.theta == 0.25
        assert "no items" in result.note

    def test_a_bound_too_tight_to_contain_the_root_is_refused(self) -> None:
        """The bound is a backstop, not an answer. A solution pinned against it means
        the information penalty never turned the objective over, and clamping to the
        edge would publish the edge of the search as a measured ability."""
        resp = np.zeros(3)
        result = mwle_theta_se(resp, A, B, C, theta0=0.0, bound=0.01)

        assert result.converged is False
        assert result.theta == 0.0
        assert "does not change sign" in result.note

    def test_an_unreachable_tolerance_exhausts_the_budget_and_says_so(self) -> None:
        resp = np.array([1.0, 0.0, 1.0])
        result = mwle_theta_se(resp, A, B, C, theta0=0.0, tolerance=0.0)

        assert result.converged is False
        assert "bisection did not reach" in result.note

    def test_a_refusal_hands_back_the_seed_unchanged(self) -> None:
        """Which is why ``converged`` has to be read: the theta looks like an answer."""
        result = mwle_theta_se(np.zeros(3), A, B, C, theta0=-0.75, bound=0.01)

        assert result.theta == -0.75
        assert result.note


class TestTheReportedEstimatorToggle:
    """What the flag changes, and what it must not."""

    def test_the_default_is_batch_eap(self) -> None:
        assert UniMcqStyle().ability_estimator == ESTIMATOR_BATCH_EAP

    def test_an_unknown_estimator_is_refused(self) -> None:
        with pytest.raises(ValueError, match="Unknown ability estimator"):
            UniMcqStyle().set_ability_estimator("wle")

    def test_batch_eap_reproduces_the_final_online_estimate_exactly(self, ladder_style) -> None:
        """EAP conditions on the whole pattern each time, so the last online estimate is
        already the batch one. Bitwise, not approximately: same function, same inputs. If
        this ever fails, ``estimate_ability`` has started accumulating and the trajectory
        replay's core assumption has gone with it."""
        build, params = ladder_style
        report = run_ladder(build, params, estimator=ESTIMATOR_BATCH_EAP, true_theta=-1.2)
        meta = report.metadata

        assert meta["theta_batch"] == meta["theta_online"]
        assert meta["se_batch"] == meta["se_online"]
        assert meta["theta"] == meta["theta_online"]
        assert report.ability.theta == meta["theta_online"]

    def test_theta_online_is_read_off_the_session_not_recomputed(self, ladder_style) -> None:
        """The two agree on every real run, which is why this has to be forced apart.

        ``theta_online`` is the estimate the *selector* saw -- the last value the engine
        put in ``state.ability``. ``theta_batch`` is this style's own refit of the same
        responses. Recomputing the first from the second would make them agree by
        construction and the report would lose the only evidence that the online
        estimator never accumulated. Here the state is built by hand with an ability the
        responses do not imply, so a recomputed field is visible.
        """
        build, _ = ladder_style
        style = build(ESTIMATOR_BATCH_EAP)
        style.download_benchmark("arc_challenge")
        style.load_irt_params("arc_challenge")

        state = CATState(benchmark="arc_challenge", se_threshold=0.3)
        state.administered = [
            ItemResponse(item_id="lad_2", chosen_index=0, correct=False),
            ItemResponse(item_id="lad_5", chosen_index=0, correct=False),
        ]
        state.ability = AbilityEstimate(theta=2.5, standard_error=0.4)

        meta = style.report(state).metadata

        assert meta["theta_online"] == 2.5
        assert meta["se_online"] == 0.4
        assert meta["theta_batch"] != pytest.approx(2.5, abs=0.5)
        assert meta["theta"] == meta["theta_batch"]

    def test_the_default_report_carries_no_mwle_number(self, ladder_style) -> None:
        build, params = ladder_style
        meta = run_ladder(build, params, estimator=ESTIMATOR_BATCH_EAP, true_theta=-1.2).metadata

        assert "theta_mwle" not in meta
        assert "mwle_ok" not in meta
        assert meta["ability_estimator_reported"] == ESTIMATOR_BATCH_EAP

    def test_enabling_mwle_changes_the_reported_theta(self, ladder_style) -> None:
        build, params = ladder_style
        default = run_ladder(build, params, estimator=ESTIMATOR_BATCH_EAP, true_theta=-1.2).metadata
        enabled = run_ladder(
            build, params, estimator=ESTIMATOR_BATCH_EAP_MWLE, true_theta=-1.2
        ).metadata

        assert enabled["mwle_ok"] is True
        assert enabled["theta"] == enabled["theta_mwle"]
        assert enabled["theta"] != default["theta"]
        assert abs(enabled["theta"]) > abs(enabled["theta_batch"])

    def test_both_thetas_are_recorded_under_either_setting(self, ladder_style) -> None:
        """So a run reported one way is still comparable with a run reported the other."""
        build, params = ladder_style
        for estimator in ESTIMATORS:
            meta = run_ladder(build, params, estimator=estimator, true_theta=-1.2).metadata

            assert "theta_online" in meta
            assert "theta_batch" in meta
            assert meta["ability_estimator"] == estimator

    def test_the_two_settings_agree_about_the_eap_estimate(self, ladder_style) -> None:
        build, params = ladder_style
        default = run_ladder(build, params, estimator=ESTIMATOR_BATCH_EAP, true_theta=-1.2).metadata
        enabled = run_ladder(
            build, params, estimator=ESTIMATOR_BATCH_EAP_MWLE, true_theta=-1.2
        ).metadata

        assert enabled["theta_batch"] == default["theta_batch"]
        assert enabled["theta_online"] == default["theta_online"]

    def test_the_report_is_json_serializable_under_mwle(self, ladder_style) -> None:
        """``se_mwle`` can be non-finite, and ``json.dumps`` would emit a bare ``NaN``
        that no strict parser reads back."""
        build, params = ladder_style
        report = run_ladder(build, params, estimator=ESTIMATOR_BATCH_EAP_MWLE, true_theta=-1.2)
        payload = json.loads(json.dumps(report.to_dict()))

        assert payload["metadata"]["theta_mwle"] == report.metadata["theta_mwle"]


class TestSelectionIsUnchanged:
    """The online estimate drives selection in both modes, so both administer the same test."""

    def test_the_same_items_arrive_in_the_same_order(self, ladder_style) -> None:
        build, params = ladder_style
        default = run_ladder(build, params, estimator=ESTIMATOR_BATCH_EAP, true_theta=-1.2)
        enabled = run_ladder(build, params, estimator=ESTIMATOR_BATCH_EAP_MWLE, true_theta=-1.2)

        assert [r.item_id for r in enabled.responses] == [r.item_id for r in default.responses]
        assert [r.correct for r in enabled.responses] == [r.correct for r in default.responses]

    def test_the_test_length_and_stop_reason_are_unchanged(self, ladder_style) -> None:
        """The stopping rule reads the online standard error, which MWLE never touches."""
        build, params = ladder_style
        default = run_ladder(build, params, estimator=ESTIMATOR_BATCH_EAP, true_theta=-1.2)
        enabled = run_ladder(build, params, estimator=ESTIMATOR_BATCH_EAP_MWLE, true_theta=-1.2)

        assert enabled.num_items_administered == default.num_items_administered
        assert enabled.metadata["stop_reason"] == default.metadata["stop_reason"]

    def test_estimate_ability_ignores_the_setting(self, ladder_style) -> None:
        """Directly, rather than through the engine: this is the method the selector's
        theta comes from, and it must return EAP whatever the report will publish."""
        build, _ = ladder_style
        responses = [
            ItemResponse(item_id="lad_3", chosen_index=0, correct=True),
            ItemResponse(item_id="lad_9", chosen_index=0, correct=False),
        ]
        both = []
        for estimator in ESTIMATORS:
            style = build(estimator)
            style.download_benchmark("arc_challenge")
            irt_bank = style.load_irt_params("arc_challenge")
            both.append(style.estimate_ability(irt_bank, responses))

        assert both[0].theta == both[1].theta
        assert both[0].standard_error == both[1].standard_error


class TestTheFallbackIsLoud:
    """A non-converged MWLE must be visible in the artifact, not only in the log."""

    def stub(self, monkeypatch: pytest.MonkeyPatch, note: str = "the bracket ran out") -> None:
        """Make the solver refuse, whatever it is handed."""
        monkeypatch.setattr(
            style_mod,
            "mwle_theta_se",
            lambda resp, a, b, c, **kwargs: MwleResult(float(kwargs["theta0"]), 0.5, False, note),
        )

    def test_it_reports_the_batch_eap_estimate(
        self, ladder_style, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        build, params = ladder_style
        self.stub(monkeypatch)

        meta = run_ladder(
            build, params, estimator=ESTIMATOR_BATCH_EAP_MWLE, true_theta=-1.2
        ).metadata

        assert meta["mwle_ok"] is False
        assert meta["theta"] == meta["theta_batch"]
        assert meta["standard_error"] == meta["se_batch"]

    def test_the_report_says_which_estimator_it_actually_used(
        self, ladder_style, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``ability_estimator`` is what was asked for and ``ability_estimator_reported``
        is what was published. A run that recorded only the request would claim an MWLE
        theta it does not carry."""
        build, params = ladder_style
        self.stub(monkeypatch)

        meta = run_ladder(
            build, params, estimator=ESTIMATOR_BATCH_EAP_MWLE, true_theta=-1.2
        ).metadata

        assert meta["ability_estimator"] == ESTIMATOR_BATCH_EAP_MWLE
        assert meta["ability_estimator_reported"] == ESTIMATOR_BATCH_EAP

    def test_the_reason_is_carried_in_the_report(
        self, ladder_style, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        build, params = ladder_style
        self.stub(monkeypatch, note="the bracket ran out")

        meta = run_ladder(
            build, params, estimator=ESTIMATOR_BATCH_EAP_MWLE, true_theta=-1.2
        ).metadata

        assert "the bracket ran out" in meta["mwle_fallback"]
        assert "batch EAP" in meta["mwle_fallback"]

    def test_it_is_logged_at_error(
        self, ladder_style, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        build, params = ladder_style
        self.stub(monkeypatch)

        with caplog.at_level("ERROR", logger="mcq_cat.uni_mcq"):
            run_ladder(build, params, estimator=ESTIMATOR_BATCH_EAP_MWLE, true_theta=-1.2)

        assert "MWLE did not converge" in caplog.text

    def test_a_converged_run_carries_no_fallback_key(self, ladder_style) -> None:
        build, params = ladder_style
        meta = run_ladder(
            build, params, estimator=ESTIMATOR_BATCH_EAP_MWLE, true_theta=-1.2
        ).metadata

        assert "mwle_fallback" not in meta

    def test_a_session_that_administered_nothing_falls_back(self, ladder_style) -> None:
        """There is no likelihood to maximize, so there is no MWLE estimate to publish."""
        build, _ = ladder_style
        style = build(ESTIMATOR_BATCH_EAP_MWLE)
        style.download_benchmark("arc_challenge")
        style.load_irt_params("arc_challenge")

        meta = style.report(CATState(benchmark="arc_challenge")).metadata

        assert meta["mwle_ok"] is False
        assert meta["theta"] == 0.0
        assert meta["se_mwle"] is None


class TestTheRunnerFlag:
    """The CLI seam: the vocabulary, the default, and the refusal."""

    def test_the_runner_and_the_style_agree_on_the_vocabulary(self) -> None:
        """``runner.py`` spells these out rather than importing them from one style, so
        the two lists have to be pinned equal somewhere."""
        assert runner_mod.ABILITY_ESTIMATORS == ESTIMATORS
        assert runner_mod.DEFAULT_ABILITY_ESTIMATOR == ESTIMATOR_BATCH_EAP

    def test_the_parser_default_is_batch_eap(self) -> None:
        args = runner_mod.build_parser().parse_args([])

        assert args.ability_estimator == ESTIMATOR_BATCH_EAP

    def test_the_parser_accepts_both_values(self) -> None:
        for estimator in ESTIMATORS:
            args = runner_mod.build_parser().parse_args(["--ability-estimator", estimator])
            assert args.ability_estimator == estimator

    def test_the_parser_rejects_anything_else(self) -> None:
        with pytest.raises(SystemExit):
            runner_mod.build_parser().parse_args(["--ability-estimator", "mle"])

    def test_it_reaches_the_style(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        write_bank(tmp_path, dataset="arc_challenge")
        monkeypatch.setattr(resolve, "CALIBRATED_DATASETS", tmp_path)
        style = UniMcqStyle()

        assert runner_mod._apply_ability_estimator(style, ESTIMATOR_BATCH_EAP_MWLE) is True
        assert style.ability_estimator == ESTIMATOR_BATCH_EAP_MWLE

    def test_a_style_without_the_seam_is_fine_at_the_default(self) -> None:
        class Plain:
            name = "plain"

        assert runner_mod._apply_ability_estimator(Plain(), ESTIMATOR_BATCH_EAP) is True

    def test_a_style_without_the_seam_refuses_a_non_default(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Accepting it silently would publish a shrunk theta out of a run whose command
        line asked for an unshrunk one, and nothing in the report would say so."""

        class Plain:
            name = "plain"

        with caplog.at_level("ERROR", logger="mcq_cat.runner"):
            allowed = runner_mod._apply_ability_estimator(Plain(), ESTIMATOR_BATCH_EAP_MWLE)

        assert allowed is False
        assert "does not implement one" in caplog.text


class TestTheConstantsAreTheRecoveredOnes:
    """The ridge and the bound come from the implementation this was ported from.

    Both are load-bearing and neither is derivable from anything else in this file, so
    they are pinned where a change has to be deliberate.
    """

    def test_the_ridge_is_the_reference_default(self) -> None:
        assert MWLE_RIDGE == 1e-6

    def test_the_bound_is_the_reference_default(self) -> None:
        assert MWLE_BOUND == 12.0

    def test_the_estimator_names_are_the_two_documented_ones(self) -> None:
        assert ESTIMATORS == ("batch_eap", "batch_eap+mwle")

    def test_the_ridge_is_what_survives_an_item_that_discriminates_nothing(self) -> None:
        """The case it is actually for: information of exactly zero.

        ``1/2 ln I`` is ``-inf`` there and ``1/sqrt(I)`` divides by zero, and a bank may
        carry ``a = 0`` on an item the fit could not separate. With the ridge both stay
        finite and the solver reports non-convergence instead of raising.
        """
        flat_a = np.array([0.0])
        flat_b = np.array([0.0])
        flat_c = np.array([0.0])
        assert float(fisher_info(0.0, flat_a, flat_b, flat_c).sum()) == 0.0

        assert np.isfinite(float(np.log(0.0 + MWLE_RIDGE)))
        result = mwle_theta_se(np.array([1.0]), flat_a, flat_b, flat_c, theta0=0.0)
        assert result.converged is False
        assert "not identified" in result.note

    def test_on_a_real_bank_the_epsilon_clip_floors_information_above_the_ridge(self) -> None:
        """So the ridge cannot move a converged estimate, which is the intent.

        ``prob`` clips its output into ``[1e-6, 1 - 1e-6]``, which already floors each
        item's Fisher information near ``a^2 * 1e-6``. On any bank whose discriminations
        sum to more than about one that floor is the binding one and the ridge is inert
        -- a guard that costs nothing and is only reached in the degenerate case above.
        Worth pinning because the opposite reading, that the ridge is what tames the
        tails, would make 1e-6 look like a tuning knob.
        """
        a, b, c = arrays(two_pl_bank(24, discrimination=2.5))

        assert float(fisher_info(MWLE_BOUND, a, b, c).sum()) > MWLE_RIDGE
        assert float(fisher_info(-MWLE_BOUND, a, b, c).sum()) > MWLE_RIDGE

    def test_irt_imports_nothing_but_numpy(self) -> None:
        """The runtime image installs ``.[hf,s3]`` and neither extra carries scipy, and
        ``replay_trajectory`` has to run this math on a laptop with no model and no GPU.
        The reference implementation reached for ``scipy.optimize.minimize``; this one
        may not. Read off the import statements rather than the text, so the prose
        explaining that constraint cannot trip the check that enforces it.
        """
        import ast

        tree = ast.parse(Path(irt_mod.__file__).read_text(encoding="utf-8"))
        roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])

        assert roots == {"__future__", "dataclasses", "numpy"}


class TestTheEstimateAgreesWithTheEngine:
    """One end-to-end pass, so the pieces are known to compose outside a unit test."""

    def test_a_low_ability_checkpoint_is_placed_lower_by_mwle(self, ladder_style) -> None:
        build, params = ladder_style
        report = run_ladder(build, params, estimator=ESTIMATOR_BATCH_EAP_MWLE, true_theta=-1.8)
        meta = report.metadata

        assert meta["mwle_ok"] is True
        assert meta["theta_mwle"] < meta["theta_batch"] < 0.0
        assert report.ability.theta == meta["theta_mwle"]
        assert report.ability.standard_error == meta["se_mwle"]

    def test_the_pirt_prediction_uses_the_reported_theta(self, ladder_style) -> None:
        """Predicted accuracy is evaluated at an ability, and publishing one ability
        beside a prediction made at a different one would be two answers to one question."""
        from ..pirt import pirt_accuracy

        build, params = ladder_style
        style = build(ESTIMATOR_BATCH_EAP_MWLE)
        bank = style.download_benchmark("arc_challenge")
        irt_bank = style.load_irt_params("arc_challenge")
        lookup = {
            row["item_id"]: (row["discrimination"], row["difficulty"], row["guessing"])
            for row in params
        }
        report = cat_loop.run_cat(
            style,
            bank=bank,
            irt_bank=irt_bank,
            model=SimScorer(-1.8, lookup, seed=11),
            se_threshold=0.3,
            max_items=40,
        )

        a, b, c = style._arrays()
        order = [style._index[r.item_id] for r in report.responses]
        scores = [1 if r.correct else 0 for r in report.responses]
        expected = pirt_accuracy(a, b, c, order, scores, report.metadata["theta_mwle"])

        assert report.metadata["pirt_accuracy"] == pytest.approx(expected, abs=1e-12)


class TestAbilityEstimateStaysAScalar:
    """``AbilityEstimate`` is shared with future MIRT styles and takes tuples too."""

    def test_the_reported_estimate_is_two_floats(self, ladder_style) -> None:
        build, params = ladder_style
        report = run_ladder(build, params, estimator=ESTIMATOR_BATCH_EAP_MWLE, true_theta=-1.2)

        assert isinstance(report.ability, AbilityEstimate)
        assert isinstance(report.ability.theta, float)
        assert isinstance(report.ability.standard_error, float)
