"""Online ATLAS adaptive-testing external evaluation over ARC-Challenge (Phase 2).

Thin subclass pinning the generic :class:`AtlasExternalEval` to ARC-Challenge so
the historical ``atlas_arc`` eval name and the ``AtlasArcExternalEval`` symbol
stay stable. All logic lives in the shared base, which is driven by the
``olmo_eval.adaptive`` core and therefore cannot diverge from the offline
``atlas_arc_challenge`` task.
"""

from __future__ import annotations

from olmo_eval.adaptive.benchmarks import get_benchmark
from olmo_eval.evals.external.benchmarks.atlas.eval import AtlasExternalEval

#: Default olmo-eval task ARC's online CAT scores against (formatting parity).
DEFAULT_TASK = "arc_challenge"


class AtlasArcExternalEval(AtlasExternalEval):
    """Online CAT over ARC-Challenge using a calibrated ATLAS 3PL bank."""

    def __init__(self) -> None:
        super().__init__(get_benchmark("arc_challenge"))
