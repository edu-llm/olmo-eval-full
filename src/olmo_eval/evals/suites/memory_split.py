"""Suites for the SmolLM2-135M memory-split (base vs split) A/B evaluation.

Reasoning first: the six control-archive reasoning tasks at 5-shot rank-classification
(per-char-normalized accuracy), each via its ``olmo3base`` variant. Fact-recall tasks
(TriviaQA / PopQA / T-REx) are added in a later pass as generative EM tasks; this file will
grow a ``memory_split_facts`` and a combined ``memory_split`` suite then.

Run against a checkpoint with the native OLMo-core provider (off-platform; the eduLLM image is
mock-only), e.g.::

    uv run olmo-eval run -m olmo_core --model <checkpoint-dir> -t memory_split_reasoning
"""

from olmo_eval.evals.suites.registry import AggregationStrategy, make_suite

make_suite(
    name="memory_split_reasoning",
    tasks=(
        "hellaswag:olmo3base",
        "piqa:olmo3base",
        "arc_easy:olmo3base",
        "csqa:olmo3base",
        "socialiqa:olmo3base",
        "openbookqa:olmo3base",
    ),
    aggregation=AggregationStrategy.AVERAGE,
    description=(
        "SmolLM2-135M base/split reasoning A/B: 5-shot rank-classification accuracy over "
        "HellaSwag, PIQA, ARC-Easy, CommonsenseQA, Social IQa, OpenBookQA."
    ),
)
