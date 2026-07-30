"""Online ATLAS adaptive-testing external evaluation over ARC-Challenge.

Registers ``atlas_arc``, which runs an item-by-item Fisher-information CAT
against a live inference provider, evaluating only the items the adaptive loop
selects. Shares the ``olmo_eval.adaptive`` core with the offline
``atlas_arc_challenge`` task.
"""

from olmo_eval.evals.external.benchmarks.atlas_arc.eval import AtlasArcExternalEval
from olmo_eval.evals.external.registry import register_external_eval

register_external_eval(AtlasArcExternalEval())
