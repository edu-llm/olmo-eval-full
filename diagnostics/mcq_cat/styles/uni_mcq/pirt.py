"""p-IRT predicted accuracy for the ``uni_mcq`` CAT style.

**Ported verbatim** from ``pirt_accuracy`` in ``src/olmo_eval/adaptive/cat.py`` at commit
``6998270f51af47ffc5623ced960461ef28557118`` on the ``Research`` branch. The arithmetic
is unchanged; only the signature is adapted, taking the bank's ``(a, b, c)`` arrays
directly instead of the ``ItemBank`` object that class does not exist on this branch.
``tests/test_irt_parity.py`` pins the result.

The blend answers "what would this model score on the whole bank?" after administering
only a handful of items. Administered items contribute their observed 0/1; the rest
contribute their IRT-predicted success probability at the final theta. The two are
weighted by how much of the bank was actually seen, so a longer test leans more on
observation and less on the model.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .irt import prob


def pirt_accuracy(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    order: Sequence[int],
    scores: Sequence[int],
    theta: float,
) -> float:
    """ATLAS p-IRT: blend observed subset accuracy with IRT-predicted accuracy.

    The administered items contribute their observed 0/1; the unseen items
    contribute their model-predicted success probability at the final ``theta``.
    Blended by the fraction of items actually observed.

    Args:
        a, b, c: 3PL parameters for **every** item in the bank.
        order: Bank indices of the administered items.
        scores: Observed 0/1 outcomes, aligned to ``order``.
        theta: The final ability estimate.

    Returns:
        Predicted accuracy over the bank. Note the denominator is the bank -- the
        calibrated item set -- and not the benchmark's full evaluation split, which is
        larger. See ``calibrated_datasets/README.md``.
    """
    n = len(a)
    if n == 0:
        return 0.0
    subset = set(int(i) for i in order)
    avg_obs = float(np.mean(scores)) if len(scores) else 0.0
    unobs = [i for i in range(n) if i not in subset]
    if unobs:
        idx = np.asarray(unobs)
        avg_pred = float(prob(theta, a[idx], b[idx], c[idx]).mean())
    else:
        avg_pred = avg_obs
    w_obs = len(order) / n
    return w_obs * avg_obs + (1 - w_obs) * avg_pred


def observed_accuracy(scores: Sequence[int]) -> float:
    """Plain accuracy over the administered items.

    Reported alongside :func:`pirt_accuracy` because it costs nothing and makes the
    predicted number checkable: on a short test the two should be in the same
    neighbourhood, and a large gap means the ability estimate is doing heavy lifting.
    """
    if not len(scores):
        return 0.0
    return float(np.mean(scores))
