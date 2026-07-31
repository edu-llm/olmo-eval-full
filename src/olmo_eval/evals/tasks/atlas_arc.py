"""ATLAS adaptive-testing task for ARC-Challenge (offline / Phase 1).

Kept as a dedicated module so the ``atlas_arc_challenge`` registration and the
:class:`AtlasARCChallenge` symbol stay stable. The reusable CAT logic now lives
in :class:`olmo_eval.evals.tasks.atlas.AtlasAdaptiveMixin`, shared by every
``atlas_<benchmark>`` task and the online external evals so the offline and
online paths cannot diverge.
"""

from __future__ import annotations

from olmo_eval.adaptive.benchmarks import get_benchmark
from olmo_eval.evals.tasks.arc import ARCChallenge
from olmo_eval.evals.tasks.atlas import AtlasAdaptiveMixin
from olmo_eval.evals.tasks.common import register


@register("atlas_arc_challenge")
class AtlasARCChallenge(AtlasAdaptiveMixin, ARCChallenge):
    """ARC-Challenge with an appended ATLAS adaptive-testing report.

    Inherits ARC-Challenge instances, formatting, and scoring unchanged; the
    native ARC ``question_id`` is preserved in ``Instance.metadata["id"]`` and
    used to join responses to the 3PL bank.
    """

    atlas_benchmark = get_benchmark("arc_challenge")
