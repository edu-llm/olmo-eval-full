"""Suites for the SmolLM2-135M memory-split (base vs split) A/B evaluation.

Three suites:
  * ``memory_split_reasoning`` -- six control-archive reasoning tasks at 5-shot rank-classification
    (per-char-normalized accuracy), each via its ``olmo3base`` variant.
  * ``memory_split_facts``     -- parametric fact-recall EM (greedy, no retrieval): TriviaQA and
    long-tail PopQA. This is the headline base-vs-split metric (expect base >> split). T-REx is
    added once a data source for it is wired.
  * ``memory_split``           -- both of the above, collected together (no blended average, since
    reasoning accuracy and fact-recall EM are different scales).

Run against a checkpoint with the native OLMo-core provider (on-platform via submit-run, or
locally), e.g.::

    olmo-eval run -m <checkpoint-dir> -t memory_split -o provider.kind=olmo_core
"""

from olmo_eval.evals.suites.registry import AggregationStrategy, get_suite, make_suite

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

make_suite(
    name="memory_split_facts",
    tasks=(
        "triviaqa",
        "popqa",
    ),
    aggregation=AggregationStrategy.AVERAGE,
    description=(
        "SmolLM2-135M base/split parametric fact-recall (greedy, no retrieval): TriviaQA + "
        "long-tail PopQA, alias exact-match. Headline metric; expect base >> split."
    ),
)

make_suite(
    name="memory_split",
    tasks=(
        get_suite("memory_split_reasoning"),
        get_suite("memory_split_facts"),
    ),
    aggregation=AggregationStrategy.NONE,
    description=(
        "SmolLM2-135M base/split full A/B: reasoning (5-shot RC accuracy) + parametric "
        "fact-recall (EM). Collected without a blended average."
    ),
)
