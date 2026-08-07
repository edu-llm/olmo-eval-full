"""Checkpoint load + batched log-likelihood MCQ scoring.

This mirrors the checkpoint-loading patterns in ``tests/OnNode/checkpoint_infer.py``:
the HuggingFace path is a real implementation and the raw ``olmo_core`` path is a
marked integration point that raises ``NotImplementedError`` with guidance.

Scoring follows the log-likelihood MCQ convention used across ``olmo_eval``
(``RequestType.LOGLIKELIHOOD`` in ``src/olmo_eval/evals/tasks``): for each item, score
one continuation per answer choice and pick the highest-scoring choice.

Nothing beyond that is shared, because the tasks these banks were calibrated behind do
not share it. Two things vary per dataset, both are data rather than code, and both are
recorded in every bank's manifest and checked against a run's resolved settings before a
checkpoint is fetched.

**The prompt.** Each task defines its own ``format_request``, and an item's calibrated
difficulty was estimated behind the one its task builds, so the layout is
:data:`MCQ_PROMPT_STYLES`, selected by :attr:`InferenceConfig.prompt_style` exactly as
the generative side selects a
:class:`~diagnostics.mcq_cat.common.generative.PromptTemplate`. Three shapes exist,
because the tasks use three:

- most benchmarks build **one prompt and vary the continuation**, which is
  :class:`SharedPromptStyle`;
- WinoGrande uses Trinh & Le (2018) partial evaluation, which **varies the prompt and
  shares the continuation**, which is :class:`BlankSubstitutionStyle`;
- MuSR lists the choices **inside** the prompt and then scores each of them as a
  continuation of it, which is :class:`NumberedChoicesStyle`.

All three reduce to a list of :class:`ScoredChoice` pairs, so the scorer itself does not
branch on which one it was handed.

**How a choice's score is formed** from that pair, which is
:data:`MCQ_SCORE_NORMALIZATIONS` selected by
:attr:`InferenceConfig.score_normalization`. ``arc_challenge``, ``hellaswag`` and
``winogrande`` declare ``LogprobMCAccuracyMetric``, the plain sum of the continuation's
token log-probabilities; MuSR declares ``LogprobPerCharMCAccuracyMetric``, the same sum
divided by the continuation's length, because Open LLM Leaderboard v2 reported MuSR as
``acc_norm`` and that is the binary its bank was fit on. The choice between them is not
a preference: they rank a choice set differently, so a bank scored under the other one
has every item's outcome decided by a rule its difficulty was not estimated under, and
EAP absorbs the whole difference into theta with an untouched standard error. A run
whose normalization disagrees with the manifest is refused rather than reported.

Where a task's ``:rc``, ``:mc`` or ``:bpb`` variant declares something else again
(``LogprobUncondMCAccuracyMetric`` on ``arc_challenge:rc``) it does not apply here: we
never select a variant, and only a task's default metric is reproduced.

This is one of the harness's two grading schemes. Generative benchmarks are graded by
:mod:`.generative`, and :mod:`.grading` picks between them by dataset modality. Heavy
dependencies (``torch``, ``transformers``) are imported lazily so this module imports
without a GPU stack installed.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..base import BenchmarkItem, ItemResponse, ScoringModel

log = logging.getLogger("mcq_cat.inference")

#: Shot count of a prompt style that frames the item's own stem and nothing else, which
#: is the default and what every style here but one carries.
#:
#: Still a structural fact rather than a setting: the MCQ scorer has no few-shot block to
#: prepend and no configuration key that could add one, so a style's count describes the
#: stems it is handed. What varies is whether those stems arrived with exemplars already
#: inside them, which is how BBH's per-subtask 3-shot prefix reaches a scorer that cannot
#: assemble one -- it is rendered by the task and frozen in at vendoring. Recording 0 for
#: such a bank would describe a prompt no model is ever shown, and the count is per style
#: rather than global so that a config edit swapping BBH onto a stem-framing style is
#: refused at startup against its manifest instead of silently dropping the prefix.
NUM_FEWSHOT = 0


@dataclass(frozen=True, slots=True)
class ScoredChoice:
    """One ``(prompt, continuation)`` pair, the unit a log-likelihood scorer consumes.

    A choice is a pair rather than a bare continuation because the two prompt shapes
    disagree about which half varies. Naming the pair is what lets both shapes reach the
    scorer through one signature, and it is also olmo-eval's own model: ``LMRequest``
    carries a ``continuation_prompts`` tuple alongside ``continuations`` precisely so a
    task can vary the prompt per choice.
    """

    prompt: str
    continuation: str


@runtime_checkable
class McqPromptStyle(Protocol):
    """How one benchmark presents an item to a log-likelihood scorer.

    The swappable part of MCQ scoring, and the counterpart of
    :class:`~diagnostics.mcq_cat.common.generative.PromptTemplate`. Implementations see
    only an item, never a model or a tokenizer, so a benchmark with a new presentation
    costs one class and no changes to scoring.

    Attributes:
        name: Recorded in :data:`MCQ_PROMPT_STYLES` and used in error messages.
        num_fewshot: How many worked examples the prompts this style builds put in front
            of the item. :data:`NUM_FEWSHOT` for a style that frames the stem alone, and
            non-zero only where the stems themselves were frozen with exemplars in them.
    """

    name: str
    num_fewshot: int

    def scored_choices(self, item: BenchmarkItem) -> tuple[ScoredChoice, ...]:
        """Return one ``(prompt, continuation)`` pair per answer choice, in order."""
        ...


@dataclass(frozen=True, slots=True)
class SharedPromptStyle:
    """One prompt for the item, one continuation per choice.

    The common shape: ``arc_challenge`` and ``hellaswag`` both build a single prompt and
    hand the scorer a continuation per choice. They differ only in the framing around
    the stem, which is what ``question_template`` holds.

    The empty template is a third case rather than a degenerate one, and BBH is why
    ``num_fewshot`` is a field here. Its stems are whole rendered prompts -- a subtask
    description, three exemplars and the item, frozen in at vendoring -- so the framing
    this class would add is already inside the stem and the only thing left to do is
    append the choice. The shape is the same; the shot count is not, and it has to travel
    with the style because a manifest recording 0 for those items would be describing a
    prompt no model is shown.

    Attributes:
        name: Registry name.
        question_template: The stem's framing. Formatted with ``question``; a stem
            containing braces is substituted in, not re-parsed.
        choice_prefix: Joins the prompt to a choice. A leading space, because every
            olmo-eval MCQ task builds its continuations as ``f" {choice}"`` and the
            space is inside the scored span, so dropping it changes the score.
        num_fewshot: Exemplars already present in the stems this style is given.
    """

    name: str
    question_template: str = "{question}"
    choice_prefix: str = " "
    num_fewshot: int = NUM_FEWSHOT

    def scored_choices(self, item: BenchmarkItem) -> tuple[ScoredChoice, ...]:
        """Frame the stem once and pair it with each prefixed choice."""
        prompt = self.question_template.format(question=item.question)
        return tuple(
            ScoredChoice(prompt=prompt, continuation=f"{self.choice_prefix}{choice}")
            for choice in item.choices
        )


@dataclass(frozen=True, slots=True)
class BlankSubstitutionStyle:
    """Trinh & Le (2018) partial evaluation: the choice goes in the prompt, not after it.

    WinoGrande's shape, and the reason :class:`ScoredChoice` is a pair. Each candidate is
    substituted into the sentence's blank, and what is scored is the text *after* the
    blank -- the same string for every choice::

        prompt A: "Sarah was a much better surgeon than Maria so Sarah"
        prompt B: "Sarah was a much better surgeon than Maria so Maria"
        both continuations: " always got the easier cases."

    This is ``_partial_context`` and ``_partial_target`` from the olmo-eval winogrande
    task, and getting it wrong is not a matter of degree. Scoring the two names as
    continuations of ``"{sentence}\\nAnswer:"`` asks which name is likelier to follow an
    answer cue in a sentence that already contains both, which is decided by recency and
    name frequency rather than by coreference. It also breaks the metric: under partial
    evaluation the continuations are byte-identical, so an unnormalized sum is
    length-invariant by construction, while option-as-continuation makes the shorter
    option win on token count alone -- a systematic push toward chance and below it.

    Attributes:
        name: Registry name.
        placeholder: The blank marker in the stem. WinoGrande's is a bare underscore,
            and it survives into the vendored items because the task emits the sentence
            unmodified.
        num_fewshot: Exemplars already present in the stems this style is given.
    """

    name: str
    placeholder: str = "_"
    num_fewshot: int = NUM_FEWSHOT

    def scored_choices(self, item: BenchmarkItem) -> tuple[ScoredChoice, ...]:
        """Substitute each choice into the blank and score the shared remainder.

        The blank is found with :func:`lone_blank_re` rather than by searching for the
        marker anywhere, because the first occurrence of ``_`` in a stem is not always a
        blank and cutting at one that is not produces no error. ARC's
        ``"When frozen carbon dioxide (CO_{2}) is heated, ..."`` splits inside the
        subscript, giving the prompt ``"When frozen carbon dioxide (CO"`` plus a choice
        and a continuation that is the rest of the question -- a well-formed pair scoring
        a question nobody asked. A run of ``___`` cuts after the first of the three and
        leaves ``"__"`` in the continuation.

        Refusing those is what lets :func:`check_prompt_style_fits` guard only the
        appending direction on the stated grounds that this one raises by itself. With a
        bare ``find`` that was true of stems with no underscore at all and false of the
        three ARC stems that have an incidental one.
        """
        sentence = item.question
        match = lone_blank_re(self.placeholder).search(sentence)
        if match is None:
            raise ValueError(
                f"Item {item.item_id!r} is scored by the {self.name!r} prompt style, "
                f"which splits the stem at a standalone {self.placeholder!r}, and its "
                f"stem has none -- the marker is either absent or occurs only inside a "
                f"word. The bank on disk was not produced from a task that emits the "
                f"blank; re-vendor it. Falling back to appending the choice, or to "
                f"cutting at an occurrence that is not the blank, would score a "
                f"different question than the one whose difficulty this bank holds."
            )
        target = " " + sentence[match.end() :].strip()
        return tuple(
            ScoredChoice(prompt=sentence[: match.start()] + choice, continuation=target)
            for choice in item.choices
        )


@dataclass(frozen=True, slots=True)
class NumberedChoicesStyle:
    """The choice list is inside the prompt, and each choice is also the continuation.

    MuSR's shape, and the third thing :class:`ScoredChoice` being a pair buys. A MuSR
    item is a multi-paragraph narrative followed by a question, and the candidate
    answers are entity names or assignments that mean nothing on their own -- "Alice"
    scored as a continuation of the narrative is a plausibility judgement about the
    story, not an answer to the question. ``leaderboard_musr``'s ``doc_to_text``
    therefore enumerates the options in the prompt before the answer cue, and the
    continuation is the option's *text* rather than its number, so the model is asked
    which of a set it has just been shown completes ``Answer:``.

    The blank line before the cue is not a typo and is not cosmetic: upstream builds the
    numbered block with a trailing newline and joins it to the cue with another, so the
    prompt every calibrated difficulty was estimated behind contains both. Reproducing
    the layout to the character is the whole point of having a style per benchmark.

    Attributes:
        name: Registry name.
        numbering_start: The first option's label. One rather than zero, which is what
            upstream writes; it is presentation only, since the number is never scored.
        num_fewshot: Exemplars already present in the stems this style is given.
    """

    name: str
    numbering_start: int = 1
    num_fewshot: int = NUM_FEWSHOT

    def scored_choices(self, item: BenchmarkItem) -> tuple[ScoredChoice, ...]:
        """List every choice in the prompt, then score each of them after the cue."""
        numbered = "".join(
            f"{index} - {choice}\n"
            for index, choice in enumerate(item.choices, start=self.numbering_start)
        )
        prompt = f"{item.question}\n\n{numbered}\nAnswer:"
        return tuple(
            ScoredChoice(prompt=prompt, continuation=f" {choice}") for choice in item.choices
        )


#: Prompt style name -> layout, selected by :attr:`InferenceConfig.prompt_style` and
#: overridden per dataset from the style's ``config.yaml``.
#:
#: Keyed by convention rather than by dataset wherever a convention is shared, unlike the
#: generative table. ``question_answer`` is olmo-eval's
#: ``evals/tasks/common/format_helpers.format_rc``, which that module exists to share
#: among "tasks that follow the standard Question: / Answer: template", and
#: ``bare_context`` is the shape of every task whose stem already ends mid-clause. Naming
#: the first of them ``arc_challenge`` would claim an ownership the upstream code
#: contradicts, and the default below has to name a convention rather than a dataset. The
#: two entries that do carry a benchmark's name carry it because the layout is that one
#: benchmark's and no other task here writes it.
MCQ_PROMPT_STYLES: dict[str, McqPromptStyle] = {
    "question_answer": SharedPromptStyle(
        name="question_answer",
        question_template="Question: {question}\nAnswer:",
    ),
    # HellaSwag's contexts end mid-clause by construction ("A man is sitting on a roof.
    # He"), so the continuation is the grammatical completion of the stem itself. An
    # answer cue inserted between the two breaks that on essentially every item.
    "bare_context": SharedPromptStyle(name="bare_context"),
    "blank_substitution": BlankSubstitutionStyle(name="blank_substitution"),
    # Named after a benchmark rather than a convention, because it is one benchmark's:
    # lm-evaluation-harness writes this block for leaderboard_musr and for nothing else
    # here, down to the blank line before the answer cue.
    "musr": NumberedChoicesStyle(name="musr"),
    # Also one benchmark's, and the only style here that adds nothing at all. BBH is
    # 3-shot behind a description that differs per *subtask*, which no per-dataset
    # setting can express, so the whole prefix is rendered by the task and frozen into
    # each stem at vendoring -- the same treatment gpqa gives its shuffled choice block.
    # What reaches run time is a finished prompt ending in the answer cue, and appending
    # the choice is all that is left. The count is 3 rather than 0 because that is what
    # the model reads, and it is what a manifest holds a run to.
    "bbh": SharedPromptStyle(name="bbh", num_fewshot=3),
}

#: Used by a dataset that names no style of its own.
#:
#: ``question_answer`` rather than a refusal, because it is olmo-eval's *own* default:
#: ``format_helpers`` is the module a task uses when it has no bespoke framing, so an
#: unlisted dataset most likely wants it. It is a defensible guess and not a safe one,
#: which is why :func:`check_prompt_style_fits` refuses the one case where guessing
#: wrong is silent, and why every dataset this style ships is listed explicitly in
#: ``config.yaml`` rather than left to fall through here.
DEFAULT_PROMPT_STYLE = "question_answer"


def lone_blank_re(placeholder: str) -> re.Pattern[str]:
    """The pattern matching ``placeholder`` where it stands alone as a blank.

    Bounded on both sides so an identifier-shaped underscore in ordinary prose
    (``fill_in``, ``__init__``, a LaTeX subscript, a run of ``___``) is not mistaken for
    a slot a candidate fills.

    Shared by the two places that have to agree on what a blank is:
    :func:`check_prompt_style_fits`, which decides whether a bank is cloze, and
    :meth:`BlankSubstitutionStyle.scored_choices`, which decides where to cut. They read
    the same stems and disagreeing would mean a bank the guard calls non-cloze is still
    split as though it were.
    """
    marker = re.escape(placeholder)
    return re.compile(rf"(?<![A-Za-z0-9_]){marker}(?![A-Za-z0-9_])")


#: A bare underscore standing alone, which is how the cloze bank here writes its slot.
_LONE_BLANK_RE = lone_blank_re("_")

#: The styles that leave the stem whole and put the choice after it, which is the shape
#: :func:`check_prompt_style_fits` refuses for a cloze bank. Listed rather than inferred
#: because the property that matters is not visible on the class: what makes a cloze
#: item unanswerable is the blank being left unfilled, and both of these leave it.
_APPENDING_STYLES = (SharedPromptStyle, NumberedChoicesStyle)

#: Fraction of a bank's stems that must carry a lone blank before the bank is taken to
#: be a cloze benchmark. Decisive rather than tuned: WinoGrande is at 100% and the other
#: two MCQ banks here are at 0.02% and 0%, one HellaSwag stem out of 4,840.
BLANK_BANK_FRACTION = 0.5


def get_mcq_prompt_style(prompt_style: str) -> McqPromptStyle:
    """Return the layout for ``prompt_style``, or raise naming what is available."""
    try:
        return MCQ_PROMPT_STYLES[prompt_style]
    except KeyError:
        raise ValueError(
            f"Unknown prompt_style {prompt_style!r}. Known styles: "
            f"{', '.join(sorted(MCQ_PROMPT_STYLES))}."
        ) from None


def check_prompt_style_fits(prompt_style: str, items: Sequence[BenchmarkItem]) -> None:
    """Refuse a cloze bank that is about to be scored by a shared-prompt style.

    The guard on :data:`DEFAULT_PROMPT_STYLE`, and the only fallback failure that would
    otherwise be invisible. Appending a choice to a sentence whose blank is still in it
    produces a well-formed prompt, a finite log-probability for every option and a
    perfectly healthy standard error; the answer is simply to a different question than
    the one the bank's difficulties describe, and EAP absorbs the whole gap into theta.
    Every other mismatch fails loudly on its own -- :class:`BlankSubstitutionStyle`
    raises on a stem with no standalone blank, an unknown style name raises in
    :func:`get_mcq_prompt_style` -- so this is the case that needs an explicit check.
    That first claim holds only because the substitution style looks for the same
    standalone marker this guard counts; matching a bare ``_`` anywhere would have it
    quietly cut a non-cloze stem at an incidental underscore instead.

    Deliberately a bank-level test rather than a per-item one. A lone underscore turns up
    in one HellaSwag stem out of 4,840, and refusing that item would fail a run over a
    typographic coincidence; a benchmark that is *actually* cloze has the marker in every
    stem. :data:`BLANK_BANK_FRACTION` sits between the two by a wide margin.
    """
    style = get_mcq_prompt_style(prompt_style)
    if not items or not isinstance(style, _APPENDING_STYLES):
        return
    blanks = sum(1 for item in items if _LONE_BLANK_RE.search(item.question))
    if blanks / len(items) < BLANK_BANK_FRACTION:
        return
    raise ValueError(
        f"{blanks} of {len(items)} stems in this bank carry a standalone blank, so it "
        f"is a fill-in-the-blank benchmark, and it is configured to be scored by the "
        f"{prompt_style!r} prompt style, which appends each choice after the stem. That "
        f"leaves the blank unfilled and asks which choice likeliest follows a sentence "
        f"that already contains all of them -- a question the bank's difficulties were "
        f"never estimated against, answered with a healthy-looking standard error. Set "
        f"this dataset's prompt_style to a substitution style (for example "
        f"'blank_substitution') in the style's config.yaml."
    )


#: Prompt style name -> the sentence a report adds about where its exemplars came from.
#:
#: Only a style whose stems arrive with exemplars already inside them needs one. For the
#: rest the count is the whole story and an extra sentence would be noise; here the count
#: alone is misleading in the other direction, since "3-shot" reads as a block this
#: harness assembled and the MCQ path has no way to assemble one.
FEWSHOT_NOTES: dict[str, str] = {
    "bbh": (
        "Those three exemplars and the subtask's one-line description were not assembled "
        "at run time -- nothing here can -- but rendered by the task and frozen into each "
        "item's stem at vendoring, so the prompt is the one the difficulties were "
        "estimated behind rather than a reconstruction of it."
    ),
}


def fewshot_count(prompt_style: str) -> int:
    """Return how many exemplars ``prompt_style``'s prompts put in front of the item.

    Read off the style rather than off :data:`NUM_FEWSHOT`, because a theta is a
    statement about what the model was shown and every MCQ report used to say 0-shot
    whichever style produced it. That was true while every stem was a bare question and
    is false of one whose stems are finished 3-shot prompts.
    """
    style = MCQ_PROMPT_STYLES.get(prompt_style)
    return NUM_FEWSHOT if style is None else style.num_fewshot


def fewshot_note(prompt_style: str) -> str:
    """Return the sentence explaining ``prompt_style``'s shot count, or none if it needs no
    explaining."""
    return FEWSHOT_NOTES.get(prompt_style, "")


@runtime_checkable
class ScoreNormalization(Protocol):
    """How one benchmark turns a continuation's summed log-probability into a score.

    The second swappable part of MCQ scoring, beside :class:`McqPromptStyle`, and kept
    separate from it because the two vary independently: MuSR shares ARC's
    "one prompt, one continuation per choice" shape and disagrees with it about how the
    resulting numbers are compared. Implementations see one continuation's total and
    that continuation's text, never a model or a tokenizer.

    Attributes:
        name: Recorded in :data:`MCQ_SCORE_NORMALIZATIONS`, written into every bank's
            manifest, and used in error messages.
    """

    name: str

    def score(self, total_logprob: float, continuation: str) -> float:
        """Return the number to rank this choice by."""
        ...


@dataclass(frozen=True, slots=True)
class UnnormalizedSum:
    """The plain sum, which is ``LogprobMCAccuracyMetric`` composed over ``LogprobScorer``.

    What ``arc_challenge``, ``hellaswag`` and ``winogrande`` declare, and the right rule
    for a benchmark whose choices are comparable in length or, as under WinoGrande's
    partial evaluation, byte-identical. It is also the only rule that leaves those three
    banks scored exactly as they were vendored, which is why it is the default rather
    than merely one of two options.
    """

    name: str

    def score(self, total_logprob: float, continuation: str) -> float:
        """Return the total unchanged."""
        return total_logprob


@dataclass(frozen=True, slots=True)
class PerCharacterMean:
    """The sum divided by the continuation's length, which is ``acc_norm``.

    ``LogprobPerCharMCAccuracyMetric``, the metric the MuSR task declares and the binary
    Open LLM Leaderboard v2 reported it under. Length normalization matters on a
    benchmark whose choices are whole clauses of very different lengths -- an
    unnormalized sum there is decided by token count on a large share of items, which is
    the same bias :class:`BlankSubstitutionStyle` exists to escape on WinoGrande.

    The divisor is the length of the *continuation*, leading space included, and the
    floor of 1 keeps an empty choice finite. That matches
    ``LogprobPerCharMCAccuracyMetric``, which divides by ``len(output.text)`` where the
    output text is the scored span, and it is one character longer than
    lm-evaluation-harness's divisor of ``len(choice)``. The difference is uniform across
    a choice set, so it can only reorder two choices whose per-character scores are
    already within about one part in their length, and matching the repo is the standing
    policy: ``arc_easy:rc`` and ``hellaswag:rc`` are scored under the same off-by-one, so
    a second per-character rule here would put two definitions of ``acc_norm`` in one
    checkout. Our ``acc_norm`` is therefore very close to, and not identical with,
    lm-eval's.
    """

    name: str

    def score(self, total_logprob: float, continuation: str) -> float:
        """Return the per-character mean, as the olmo-eval metric computes it."""
        return total_logprob / max(len(continuation), 1)


#: Normalization name -> rule, selected by :attr:`InferenceConfig.score_normalization`
#: and overridden per dataset from the style's ``config.yaml``.
#:
#: Keyed by what the rule does rather than by a benchmark, for :data:`MCQ_PROMPT_STYLES`'
#: reason: neither of these is any one benchmark's, and both name a metric olmo-eval
#: shares across many tasks. Which one a dataset takes is a fact about its calibration
#: and not a run-time preference, so it is recorded in the bank's manifest and a run
#: resolving the other one is refused before the checkpoint is fetched.
MCQ_SCORE_NORMALIZATIONS: dict[str, ScoreNormalization] = {
    "unnormalized_sum_of_continuation_logprobs": UnnormalizedSum(
        name="unnormalized_sum_of_continuation_logprobs"
    ),
    "continuation_logprob_per_character": PerCharacterMean(
        name="continuation_logprob_per_character"
    ),
}

#: Used by a dataset that names no normalization of its own.
#:
#: The unnormalized sum, and the string the four banks vendored before this was
#: configurable already record. Keeping the name as well as the behaviour is what makes
#: those manifests still resolve to exactly what they say: a renamed default would fail
#: every one of them at startup over a rule that had not changed.
DEFAULT_SCORE_NORMALIZATION = "unnormalized_sum_of_continuation_logprobs"


#: Normalization name -> the clause a report uses to say how its choices were ranked.
#:
#: The counterpart of ``generative.GRADER_NOTES``, and it exists for the same reason: two
#: banks graded under different rules are on different scales, and a report that
#: described only the modality would have every MCQ theta carrying the same sentence
#: whichever rule produced it. Naming the metric is what lets a reader check the claim
#: against the task.
NORMALIZATION_NOTES: dict[str, str] = {
    "unnormalized_sum_of_continuation_logprobs": (
        "summed over the continuation's tokens and not length-normalized, matching "
        "olmo-eval's LogprobMCAccuracyMetric"
    ),
    "continuation_logprob_per_character": (
        "summed over the continuation's tokens and divided by its length in characters, "
        "which is acc_norm -- olmo-eval's LogprobPerCharMCAccuracyMetric, whose divisor "
        "includes the continuation's leading space and is therefore one character longer "
        "than lm-evaluation-harness's"
    ),
}


def get_mcq_score_normalization(score_normalization: str) -> ScoreNormalization:
    """Return the rule for ``score_normalization``, or raise naming what is available."""
    try:
        return MCQ_SCORE_NORMALIZATIONS[score_normalization]
    except KeyError:
        raise ValueError(
            f"Unknown score_normalization {score_normalization!r}. Known normalizations: "
            f"{', '.join(sorted(MCQ_SCORE_NORMALIZATIONS))}."
        ) from None


def normalization_note(score_normalization: str) -> str:
    """Return the report clause for ``score_normalization``, or a neutral one."""
    return NORMALIZATION_NOTES.get(score_normalization, f"ranked by {score_normalization}")


@dataclass
class InferenceConfig:
    """Configuration for checkpoint loading and MCQ scoring.

    ``prompt_style`` and ``score_normalization`` are the benchmark-specific settings.
    Both are per dataset rather than global because the MCQ tasks here disagree about
    both -- about how an item is presented, and about how the resulting log-probabilities
    are compared -- and each bank was calibrated behind its own pair, so a style's
    ``config.yaml`` overrides them from a ``datasets`` map exactly as it does for the
    generative settings.
    """

    checkpoint_kind: str = "hf"  # "hf" or "olmo_core"
    batch_size: int = 16
    max_length: int | None = None
    seed: int = 1234
    device_map: str = "auto"
    prompt_style: str = DEFAULT_PROMPT_STYLE
    score_normalization: str = DEFAULT_SCORE_NORMALIZATION


def scored_choices(item: BenchmarkItem, config: InferenceConfig) -> tuple[ScoredChoice, ...]:
    """Return the ``(prompt, continuation)`` pairs to score for ``item``."""
    return get_mcq_prompt_style(config.prompt_style).scored_choices(item)


def continuation_token_count(prompt_tokens: int, full_tokens: int, max_length: int | None) -> int:
    """How many of a tokenized ``prompt + continuation`` belong to the continuation.

    Length arithmetic kept apart from the forward pass so it can be checked without a
    model, since it is the part that decides *which tokens are scored* and the forward
    pass only reads them.

    The count is taken from the untruncated pair. :attr:`InferenceConfig.max_length`
    drops tokens off the left, which is the prompt end and never the continuation, so
    the continuation's length does not change when it applies -- but measuring after
    the cut makes it look as though it had, and the score then covers only the last
    ``max_length - prompt_tokens`` tokens of the answer, or nothing at all once the cap
    sits below the prompt. Every choice would come back ``0.0``, the argmax would take
    the first one on every item, and the run would report an ability computed from a
    fixed answer with no sign in it that a length cap had done the choosing.

    A cap that cannot hold the continuation and at least one token of prompt is refused
    rather than clamped: there is no span left that scores the item, and the caller is
    asking for something a smaller value of a memory knob cannot deliver.

    Returns:
        The continuation's token count, or 0 when the tokenizer produced no extra tokens
        for it -- a continuation that merges entirely into the prompt's last token,
        which has no span of its own to score.
    """
    count = full_tokens - prompt_tokens
    if count <= 0:
        return 0
    if max_length is not None and full_tokens > max_length and max_length <= count:
        raise ValueError(
            f"max_length is {max_length} but this item's continuation alone is "
            f"{count} tokens, so left-truncating to the cap would cut into the answer "
            f"being scored and leave no prompt in front of it. max_length truncates "
            f"the prompt; it cannot be set below what the choice itself needs."
        )
    return count


def load_scoring_model(checkpoint_dir: Path, config: InferenceConfig) -> ScoringModel:
    """Load an MCQ log-likelihood scoring model for the configured checkpoint kind."""
    if config.checkpoint_kind == "hf":
        return _HFScoringModel(checkpoint_dir, config)
    if config.checkpoint_kind == "olmo_core":
        return _load_olmo_core(checkpoint_dir, config)
    raise ValueError(
        f"Unknown checkpoint_kind: {config.checkpoint_kind!r} (expected 'hf' or 'olmo_core')"
    )


class _HFScoringModel:
    """HuggingFace-backed batched log-likelihood MCQ scorer."""

    def __init__(self, checkpoint_dir: Path, config: InferenceConfig) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

        self._torch = torch
        self.config = config
        set_seed(config.seed)

        self.tokenizer: Any = AutoTokenizer.from_pretrained(str(checkpoint_dir))
        self.model: Any = AutoModelForCausalLM.from_pretrained(
            str(checkpoint_dir),
            torch_dtype="auto",
            device_map=config.device_map,
        )
        self.model.eval()

    def _continuation_logprob(self, prompt: str, continuation: str) -> float:
        """Sum the log-probabilities of ``continuation`` tokens given ``prompt``."""
        torch = self._torch
        prompt_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        full_ids = self.tokenizer(prompt + continuation, return_tensors="pt")["input_ids"]

        cont_len = continuation_token_count(
            prompt_ids.shape[1], full_ids.shape[1], self.config.max_length
        )
        if cont_len <= 0:
            return 0.0

        if self.config.max_length is not None and full_ids.shape[1] > self.config.max_length:
            full_ids = full_ids[:, -self.config.max_length :]

        full_ids = full_ids.to(self.model.device)

        with torch.no_grad():
            logits = self.model(full_ids).logits

        log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        targets = full_ids[:, 1:]
        token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        continuation_log_probs = token_log_probs[:, -cont_len:]
        return float(continuation_log_probs.sum().item())

    def score_items(self, items: Sequence[BenchmarkItem]) -> list[ItemResponse]:
        """Grade each item by scoring every choice's continuation log-likelihood.

        One forward pass per :class:`ScoredChoice`, which is one per answer choice
        whichever prompt shape produced them. A shared-prompt benchmark re-encodes the
        same prompt for every choice; a substitution benchmark re-encodes a different
        prompt each time and is cheaper for it, since a WinoGrande partial context is
        shorter than the sentence it was cut from.

        The normalization is applied here rather than inside ``_continuation_logprob``
        so that method stays a statement about the model -- the summed log-probability
        of a span, the same quantity olmo-eval's ``LogprobScorer`` returns -- and the
        benchmark's convention stays one lookup, in one place, alongside its prompt.
        ``choice_logprobs`` on the response therefore carries the numbers the argmax
        actually compared, which is what makes a report re-checkable.
        """
        normalization = get_mcq_score_normalization(self.config.score_normalization)
        responses: list[ItemResponse] = []
        for item in items:
            if not item.choices:
                raise ValueError(
                    f"Item {item.item_id!r} has no answer choices, so there is nothing "
                    f"to rank by log-likelihood. This is a generative item reaching the "
                    f"MCQ scorer; grade it with common.generative instead. "
                    f"common.grading.check_bank_modality catches this at the bank level, "
                    f"before a checkpoint is loaded."
                )
            choice_logprobs = tuple(
                normalization.score(
                    self._continuation_logprob(choice.prompt, choice.continuation),
                    choice.continuation,
                )
                for choice in scored_choices(item, self.config)
            )
            chosen_index = max(range(len(choice_logprobs)), key=lambda i: choice_logprobs[i])
            responses.append(
                ItemResponse(
                    item_id=item.item_id,
                    chosen_index=chosen_index,
                    correct=chosen_index == item.gold_index,
                    choice_logprobs=choice_logprobs,
                )
            )
        return responses


def _load_olmo_core(checkpoint_dir: Path, config: InferenceConfig) -> ScoringModel:
    """Load a raw OLMo-core checkpoint for scoring (integration point).

    Mirrors ``_load_olmo_core`` in ``tests/OnNode/checkpoint_infer.py``: reconstruct
    the model and tokenizer from the run config and load the checkpoint weights.
    This depends on the run's config layout and is left as an integration point.
    """
    raise NotImplementedError(
        "olmo_core scoring is a training-env integration point. Provide the run config "
        "and checkpoint layout, then implement _load_olmo_core (mirror the HF scorer's "
        "score_items). See tests/OnNode/checkpoint_infer.py and the plan's open questions."
    )
