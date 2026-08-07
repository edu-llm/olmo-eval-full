"""Pin the ported IRT math against the Research implementation.

The math in :mod:`..irt` and :mod:`..pirt` is a *copy* of
``src/olmo_eval/adaptive/{irt,cat}.py`` at commit ``6998270`` on the ``Research``
branch, because this branch has nothing to import. Copies drift. These reference values
were produced by executing that upstream code directly, so a change here that moves any
number will fail loudly instead of silently invalidating comparison with every
published ATLAS CAT result.
"""

from __future__ import annotations

import numpy as np
import pytest

from ..irt import THETA_NODES, eap_theta_se, fisher_info, prob
from ..pirt import observed_accuracy, pirt_accuracy

# A deliberately mixed fixture: one easy high-discrimination item with a large guessing
# floor, one 2PL-shaped item (c = 0), one hard item.
A = np.array([1.5, 0.8, 2.2])
B = np.array([-0.5, 0.0, 1.0])
C = np.array([0.2, 0.0, 0.25])
THETA = 0.3

# Produced by running olmo_eval.adaptive.irt / .cat at Research commit 6998270.
REF_PROB = [0.8148198267992142, 0.5597136492671929, 0.3824014560843375]
REF_FISHER = [0.30201684711360705, 0.1577179389429287, 0.24360998351732535]
REF_EAP_THETA = 0.27614879958398675
REF_EAP_SE = 0.9212654588581957
REF_PIRT_PARTIAL = 0.8532378830890643
REF_PIRT_FULL = 0.6666666666666666
REF_PIRT_EMPTY = 0.5856449773835816


def test_prob_matches_research() -> None:
    np.testing.assert_allclose(prob(THETA, A, B, C), REF_PROB, rtol=0, atol=1e-12)


def test_fisher_info_matches_research() -> None:
    np.testing.assert_allclose(fisher_info(THETA, A, B, C), REF_FISHER, rtol=0, atol=1e-12)


def test_eap_matches_research() -> None:
    theta, se = eap_theta_se(np.array([1.0, 0.0, 1.0]), A, B, C)
    assert theta == pytest.approx(REF_EAP_THETA, abs=1e-12)
    assert se == pytest.approx(REF_EAP_SE, abs=1e-12)


def test_quadrature_grid_is_the_atlas_grid() -> None:
    """81 nodes on [-4, 4]. The grid is part of the estimator, not a free parameter."""
    assert THETA_NODES.shape == (81,)
    assert THETA_NODES[0] == pytest.approx(-4.0)
    assert THETA_NODES[-1] == pytest.approx(4.0)


def test_pirt_blends_observed_and_predicted() -> None:
    """Partially observed: observed and predicted both contribute, weighted by coverage."""
    got = pirt_accuracy(A, B, C, [0, 2], [1, 1], THETA)
    assert got == pytest.approx(REF_PIRT_PARTIAL, abs=1e-12)


def test_pirt_with_full_coverage_is_observed_accuracy() -> None:
    """With every item administered there is nothing left to predict."""
    got = pirt_accuracy(A, B, C, [0, 1, 2], [1, 0, 1], THETA)
    assert got == pytest.approx(REF_PIRT_FULL, abs=1e-12)
    assert got == pytest.approx(2 / 3, abs=1e-12)


def test_pirt_with_no_coverage_is_pure_prediction() -> None:
    """With nothing administered the estimate is the model's mean over the bank."""
    got = pirt_accuracy(A, B, C, [], [], THETA)
    assert got == pytest.approx(REF_PIRT_EMPTY, abs=1e-12)
    assert got == pytest.approx(float(prob(THETA, A, B, C).mean()), abs=1e-12)


def test_pirt_on_empty_bank_is_zero() -> None:
    empty = np.array([])
    assert pirt_accuracy(empty, empty, empty, [], [], THETA) == 0.0


def test_observed_accuracy() -> None:
    assert observed_accuracy([1, 0, 1, 1]) == pytest.approx(0.75)
    assert observed_accuracy([]) == 0.0


class TestFitFamilyReduction:
    """2PL is the ``c = 0`` case of 3PL, exactly.

    This is why one implementation serves both families and there is no 2PL branch.
    If someone later adds one, these tests say it was unnecessary.
    """

    a = np.array([0.7, 1.3, 2.0])
    b = np.array([-1.0, 0.25, 1.5])
    zero_c = np.zeros(3)

    @pytest.mark.parametrize("theta", [-2.0, -0.5, 0.0, 0.8, 2.5])
    def test_prob_reduces_to_the_bare_sigmoid(self, theta: float) -> None:
        expected = 1.0 / (1.0 + np.exp(-self.a * (theta - self.b)))
        np.testing.assert_allclose(
            prob(theta, self.a, self.b, self.zero_c), expected, rtol=0, atol=1e-9
        )

    @pytest.mark.parametrize("theta", [-2.0, -0.5, 0.0, 0.8, 2.5])
    def test_fisher_info_reduces_to_two_pl_form(self, theta: float) -> None:
        """``a^2 (1-p)/p ((p-c)/(1-c))^2`` becomes ``a^2 p (1-p)`` at ``c = 0``.

        This is the 2PL Fisher formula Research uses in
        ``Research/scripts/mcq_diagnostic.py``.
        """
        p = prob(theta, self.a, self.b, self.zero_c)
        expected = (self.a**2) * p * (1 - p)
        np.testing.assert_allclose(
            fisher_info(theta, self.a, self.b, self.zero_c), expected, rtol=1e-8, atol=1e-12
        )

    def test_a_nonzero_guessing_floor_raises_the_probability(self) -> None:
        """Sanity check that ``c`` is doing something, so the reduction tests mean something."""
        low = prob(-2.0, self.a, self.b, self.zero_c)
        high = prob(-2.0, self.a, self.b, np.full(3, 0.25))
        assert np.all(high > low)
