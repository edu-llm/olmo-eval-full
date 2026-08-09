"""Pedagogy Benchmark: teacher-facing pedagogical knowledge, scored by ranking options.

``AI-for-Education/pedagogy-benchmark`` (MIT), config ``cdpk_main``, 920 questions drawn
from teacher certification and licensure exams.

This task carries a second obligation beyond evaluating a checkpoint: it is the
enumeration ``diagnostics/mcq_cat`` vendors the calibrated ``pedagogy`` item bank
against. An IRT difficulty describes an item *as it was presented* when the calibrating
sweep estimated it, so a bank whose task later renders the same rows differently is a
bank of numbers about prompts nobody sends. Every rendering choice below therefore
reproduces the 2026-08-01 loader (``load_pedagogy`` in the Research tree's
``datasets_registry.py``) rather than picking whatever this repo's other MCQ tasks
happen to do: the ``Question: ... / Answer:`` frame, the option text as the scored
continuation behind a single space, zero shots, and the per-token mean the calibrating
engine scored with. The one deliberate divergence is that the loader read the dataset's
mutable ``main`` branch and this task pins a revision; see :data:`PEDAGOGY_REVISION`.

Two properties of the upstream layout are easy to get wrong and expensive to get wrong
silently, so both are stated here and enforced in :meth:`Pedagogy.process_doc`. Neither
is visible in ``cdpk_main`` as it stands, which is the reason to write them down rather
than to leave them to be rediscovered:

* **The option columns are sparse and the option count is not fixed.** Options live in
  seven columns, ``answer_a`` .. ``answer_g``, and a question with four options still
  has all seven; the rest hold nulls. Every row of ``cdpk_main`` at the pinned revision
  fills exactly ``answer_a`` .. ``answer_d``, so nothing here is ragged today — but the
  schema is seven wide and the sibling ``cdpk_send`` config shares it, so code written
  around four choices is one config away from being wrong.
* **Gold is an index among the options that survived, not among the columns.** The
  dataset states its answer as a column letter, but the scored choice list holds only
  the non-null columns, so the two numberings diverge as soon as a null column sits
  before the keyed one: with ``answer_b`` empty, the answer ``"c"`` is choice 1 and not
  choice 2. At the pinned revision no row is keyed to a null column and the two
  numberings coincide everywhere, which makes resolving the letter with plain ``ord``
  arithmetic look correct and leaves nothing to notice when it stops being so.

The dataset is **gated**: an unauthenticated fetch returns HTTP 401. Loading needs a
Hugging Face token with access granted (``HF_TOKEN``, or a ``hf auth login`` session,
which the backend picks up). This bites at vendoring time only — a vendored bank is read
from local JSONL and never reaches this module.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

from olmo_eval.common.formatters import MultipleChoiceFormatter
from olmo_eval.common.metrics import BPBMetricInstanceAvg, LogprobPerTokenMCAccuracyMetric
from olmo_eval.common.types import Instance, LMRequest, RequestType, SamplingParams, Split
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import Task, register, register_variant
from olmo_eval.evals.tasks.common.format_helpers import (
    format_mc as _format_mc,
)
from olmo_eval.evals.tasks.common.format_helpers import (
    format_rc as _format_rc,
)

#: Commit the task reads, rather than the ``main`` the calibrating loader read.
#:
#: Pinned because this bank's item ids are positional in all but name: ``question_id``
#: is the row index, so an upstream reorder that renumbered the column would move every
#: item's difficulty onto a different question and change nothing a checksum could see.
#: ``main`` has not moved since 2025-06-24, before the calibration sweep, so the pinned
#: content is byte-for-byte what was calibrated and the pin costs nothing today; it
#: exists to keep that true.
PEDAGOGY_REVISION = "55133a11b8f3c05adc186c11d8e4e100219dd36d"

#: Option column suffixes, in presentation order: ``answer_a`` .. ``answer_g``.
OPTION_LETTERS = "abcdefg"

#: Values that mean "this option column is unused". The literal string ``"None"`` is
#: here because the calibrating loader treated it as absent, and no row of ``cdpk_main``
#: currently holds one: the entry defines which rows survive, so removing it as dead
#: weight would change the choice set the moment such a value appeared.
_UNUSED_OPTION = (None, "", "None")

#: Columns the stem may arrive under, in precedence order. The second is upstream
#: drift insurance carried over from the calibrating loader, not a live alternative.
_STEM_COLUMNS = ("question", "Question")

#: Columns the answer key may arrive under, in precedence order. As above.
_ANSWER_COLUMNS = ("correct_answer", "answer", "correct")


def _first_present(doc: dict[str, Any], columns: Sequence[str]) -> Any:
    """Return the first of ``columns`` present in ``doc`` with a non-empty value."""
    for column in columns:
        if column in doc and doc[column] not in (None, ""):
            return doc[column]
    return None


def _resolve_gold(answer: Any, choices: Sequence[str]) -> int | None:
    """Resolve a gold answer stated as a letter, an index, or the option text itself.

    The fallback path, reached only when the answer is not one of the letters whose
    column survived. That happens when the keyed column is null — a mis-keyed row — and
    it is exactly where letter-to-index arithmetic is least trustworthy, so the reading
    here is deliberately literal and returns ``None`` rather than guessing when nothing
    matches. A row with no resolvable gold is dropped: kept, it would score incorrect for
    every model and depress every ability estimate for a reason that is not the model's.
    """
    text = str(answer).strip()
    if len(text) == 1 and text.upper().isalpha():
        index = ord(text.upper()) - ord("A")
        return index if 0 <= index < len(choices) else None
    if text.isdigit():
        index = int(text)
        return index if 0 <= index < len(choices) else None
    for index, choice in enumerate(choices):
        if str(choice).strip() == text:
            return index
    return None


@register("pedagogy")
class Pedagogy(Task):
    """Rank the answer options of a pedagogical-knowledge question by log-likelihood.

    Metric note: the per-token mean, matching the engine that produced the responses the
    bank was fit from, which averaged over continuation tokens. The sibling MCQ tasks are
    not consistent about this — per-character, per-token and unnormalized all appear —
    and length normalization decides which option wins, so copying a neighbour here would
    re-rank items against their own difficulties.
    """

    data_source = DataSource(
        path="AI-for-Education/pedagogy-benchmark",
        subset="cdpk_main",
        split="train",
        revision=PEDAGOGY_REVISION,
    )
    split = Split.TRAIN
    metrics = (LogprobPerTokenMCAccuracyMetric(),)
    num_fewshot = 0
    fewshot_split = "train"
    sampling_params = SamplingParams(temperature=0.0)

    @property
    def instances(self) -> Iterator[Instance]:
        """Yield one instance per surviving row, in the dataset's own row order.

        No subsampling happens here, unlike the sibling tasks that merge splits and
        resample to reproduce an oe-eval ordering. This bank is a single split, and the
        vendoring join is positional against this enumeration, so a reordering step
        inside it would be a silent misalignment. ``config.limit`` is still honoured —
        the runner applies it to whatever a task yields.
        """
        yield from self._load_instances_cached()

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        """Build one instance, or ``None`` for a row that cannot be scored as an MCQ.

        Three rejections, all inherited from the calibrating loader so that a row it kept
        is a row this keeps: no stem, no answer key, or fewer than two surviving options
        (one option is not a choice). The gold index is then resolved against the
        *surviving* options — ``letters.index(key)`` below rather than
        ``ord(key) - ord('a')`` — for the reason the module docstring gives.

        Stem and options are stripped, because the scored span is ``" {choice}"`` and a
        choice carrying its own leading or trailing whitespace would be scored as a
        different string than the one the difficulties were estimated behind. Five
        options in this config need it.
        """
        choices: list[str] = []
        letters: list[str] = []
        for letter in OPTION_LETTERS:
            value = doc.get(f"answer_{letter}")
            if value in _UNUSED_OPTION:
                continue
            choices.append(str(value).strip())
            letters.append(letter)

        stem = _first_present(doc, _STEM_COLUMNS)
        answer = _first_present(doc, _ANSWER_COLUMNS)
        if stem is None or answer is None or len(choices) < 2:
            return None

        key = str(answer).strip().lower()
        gold_idx = letters.index(key) if key in letters else _resolve_gold(answer, choices)
        if gold_idx is None:
            return None

        question_id = doc.get("question_id")
        item_id = (
            f"pedagogy_{question_id}" if question_id not in (None, "") else f"pedagogy_{index:05d}"
        )

        return Instance(
            question=str(stem).strip(),
            choices=tuple(choices),
            # The option text, not the letter: it is what an RC-style few-shot exemplar
            # has to show after "Answer:", and it is the fallback the bank vendorer uses
            # when an instance carries no ``gold_idx``.
            gold_answer=choices[gold_idx],
            metadata={
                # The identifier the calibrated bank keys on, reproduced exactly. The
                # column is a dense 0..919 that happens to equal the row index, so the
                # anchor is only as strong as the revision pin above.
                "id": item_id,
                "index": index,
                "dataset": "pedagogy",
                "gold_idx": gold_idx,
                "gold_text": choices[gold_idx],
                "mc_answer": chr(ord("A") + gold_idx),
                "question_id": question_id,
                # The upstream column letters behind each choice, in choice order. Kept
                # because it is the only record of which columns were dropped, and the
                # gap between this and A/B/C ordering is where a mis-keyed item hides.
                "option_letters": tuple(letters),
                "num_choices": len(choices),
                "category": doc.get("category"),
                "subdomain": doc.get("pedagogical_subdomain"),
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        """Frame the stem and score one continuation per option.

        The default (``rc``) shape is the calibrated one: the standard ``Question: ... /
        Answer:`` frame from ``format_helpers``, which is ``question_answer`` in the CAT
        runner's ``MCQ_PROMPT_STYLES``, with each option scored behind a single leading
        space. The ``mc`` shape lists the options in the prompt and scores bare letters
        instead; with up to seven options those run A through G.
        """
        fewshot = self.get_fewshot()
        is_mc = self.config.formatter is not None

        parts: list[str] = []
        for ex in fewshot:
            if is_mc:
                answer = ex.metadata.get("mc_answer", "")
                parts.append(_format_mc(ex.question, ex.choices or (), answer))
            else:
                answer = ex.gold_answer or ex.metadata.get("gold_text", "")
                parts.append(_format_rc(ex.question, answer))

        if is_mc:
            parts.append(_format_mc(instance.question, instance.choices or ()))
            continuations = tuple(
                f" {chr(ord('A') + i)}" for i in range(len(instance.choices or ()))
            )
        else:
            parts.append(_format_rc(instance.question))
            continuations = tuple(f" {c}" for c in (instance.choices or ()))

        prompt = "\n\n".join(parts)
        return LMRequest(
            request_type=RequestType.LOGLIKELIHOOD,
            prompt=prompt,
            continuations=continuations,
        )


# No ``olmes`` / ``olmo3base`` variant and no ``constants/pedagogy.py``: those exist for
# the OLMES suite tasks, whose fixed few-shot exemplar lists have to be vendored verbatim
# to reproduce a published number. Pedagogy is not an OLMES task, was calibrated 0-shot,
# and its only split is ``train``, so there is no held-out pool to draw exemplars from
# and nothing to freeze.
register_variant("pedagogy", "rc")
register_variant("pedagogy", "mc", formatter=MultipleChoiceFormatter())
register_variant("pedagogy", "bpb", metrics=(BPBMetricInstanceAvg(),))
register_variant("pedagogy", "full")
