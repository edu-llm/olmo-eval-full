"""Online ATLAS adaptive-testing external evals for the wired benchmark set.

Registers ``atlas_<name>`` for hellaswag/winogrande/csqa/piqa (MCQ loglik) and
gsm8k (generative exact-match). ARC keeps its own ``atlas_arc`` registration in
the sibling ``atlas_arc`` package. All share the ``olmo_eval.adaptive`` core
with the offline ``atlas_<name>`` tasks.
"""

from olmo_eval.evals.external.benchmarks.atlas.eval import make_atlas_external_evals
from olmo_eval.evals.external.registry import register_external_eval

for _eval in make_atlas_external_evals():
    register_external_eval(_eval)
