"""ATLAS adaptive-testing suite.

Groups the offline ``atlas_<name>`` tasks wired end to end
(arc/hellaswag/winogrande/csqa/piqa via MCQ loglik, plus gsm8k via generative
exact-match) so ``olmo-eval run -s atlas`` exercises the whole set. truthfulqa
(varies / no base task) is excluded until it has an olmo-eval base task.

Aggregation is ``NONE``: the headline per-task signal is theta/SE (and diagnostic
p-IRT), which are not comparable across benchmarks, so the suite collects
individual task results rather than averaging them. Banks are synced from S3 at
run time; a task whose bank is absent degrades to base metrics (no ``atlas_*``
keys), so the suite is safe to define over every wired benchmark.
"""

from olmo_eval.adaptive.benchmarks import cat_benchmarks
from olmo_eval.evals.suites.registry import AggregationStrategy, Suite, register

ATLAS = register(
    Suite(
        name="atlas",
        tasks=tuple(b.offline_task_name for b in cat_benchmarks()),
        aggregation=AggregationStrategy.NONE,
        description=(
            "ATLAS IRT adaptive-testing tasks (theta/SE/p-IRT) over the wired benchmark set."
        ),
    )
)
