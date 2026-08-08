"""Checkpoint load + sampled-completion grading for generative benchmarks.

This is the second of the harness's two grading schemes and is deliberately a module
of its own rather than a branch inside :mod:`.inference`. The two share nothing but
the :class:`~diagnostics.mcq_cat.base.ScoringModel` protocol: the MCQ path ranks a
fixed choice set by continuation log-likelihood and never generates a token, while
this path generates a completion and then decides whether the answer inside it is
right. Kept apart, neither can quietly acquire the other's behaviour, and a reader
asking "how is GSM8K graded" has one file to read.

Everything above the scorer is untouched. Fisher-information selection, EAP and p-IRT
only ever see a binary correct/incorrect, so a generative bank runs through the same
frozen CAT engine as an MCQ bank. :mod:`.grading` is where a dataset's modality picks
between the two.

Four small named pieces, smallest first:

- a **grader** turns a completion into a :class:`Verdict`. This is the swappable part:
  a benchmark whose answers are not numbers, or an LLM-as-a-judge, is one more entry in
  :data:`ANSWER_GRADERS` and touches no sampling code. There are two shapes of it,
  because there are two ways a benchmark can define a correct answer. Most compare the
  completion against a gold string and implement :class:`AnswerGrader`
  (:class:`LastNumberExactMatch`, :class:`MathLatexEquivalence`,
  :class:`GPQACoTLetter`). IFEval has no gold to compare against -- the prompt names
  verifier functions and the response either satisfies all of them or does not -- so it
  implements :class:`ItemGrader` and reads the item instead.
- a **prompt template** (:class:`PromptTemplate`) turns an item and a few-shot block
  into a prompt. Benchmarks differ here -- GSM8K asks ``Question:``/``Answer:`` and
  MATH asks ``Problem:``/``Solution:`` -- and the convention is part of what an item's
  difficulty was estimated under, so it belongs in configuration rather than in code.
- a **completer** (:class:`_HFCompleter`) turns a prompt into a completion. This is
  the backend-specific part, and the only part that needs a GPU. It is also where a
  chat-format benchmark's prompt is wrapped in the checkpoint's own chat template, and
  where a :data:`SYSTEM_PROMPTS` entry becomes a system turn, since that wrapping is a
  property of the checkpoint rather than of the benchmark.
- :class:`GenerativeScorer` composes them into a ``ScoringModel``.

Every one of those mirrors the corresponding olmo-eval task, except where the style's
``config.yaml`` argues its way off it in writing. GSM8K: comma separators stripped, the
last number in the completion taken, ``Question:`` and a blank line as stop sequences,
512 new tokens, greedy decoding. MATH: the Minerva ``Problem:``/``Solution:`` framing,
``\\boxed{}`` extraction and sympy equivalence, and -- the two departures --
``Problem:`` as the only stop sequence, matching lm-eval's ``leaderboard_math`` rather
than the task's added blank line, inside 2048 new tokens. IFEval: the prompt verbatim as
a completion, no few-shot block, no stop sequences, 1536 new tokens,
and every named instruction verified. GPQA: the expert-scientist system prompt asking
for step-by-step reasoning ending in ``ANSWER: X``, the question and its lettered
choices as the user turn, no stop sequences, 1024 new tokens, and the extracted letter
compared to the item's gold letter. That fidelity is the whole point. All four banks
were calibrated against Open LLM Leaderboard scoring, and an item's difficulty only
means anything under the grader it was estimated with. A looser grader makes every item
look easier than its calibrated ``b``, and because EAP treats ``b`` as fixed truth the
entire discrepancy is absorbed into theta -- a silent scale shift that the standard
error does not reveal.

Heavy dependencies (``torch``, ``transformers``) are imported lazily, exactly as
:mod:`.inference` does, so this module imports without a GPU stack installed.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..base import BenchmarkItem, ItemResponse, ScoringModel

log = logging.getLogger("mcq_cat.generative")

#: ``ItemResponse.chosen_index`` for a generative item. There is no choice set, so
#: there is no index; -1 says so rather than pointing at a choice that does not exist.
NO_CHOICE_INDEX = -1

#: Both copied from ``olmo_eval.evals.tasks.gsm8k``. Comma separators are removed
#: first so "1,234" reads as one number rather than as 1 followed by 234.
_NUMBER_RE = re.compile(r"[-+]?\d*\.\d+|[-+]?\d+")
_COMMA_IN_NUMBER_RE = re.compile(r"(\d),(\d)")


def extract_last_number(text: str) -> str | None:
    """Return the last number in ``text``, or ``None`` if it contains none.

    Mirrors ``_extract_last_number`` in the olmo-eval GSM8K task. Last rather than
    first because the convention the few-shot block teaches is to reason aloud and
    finish with "So the answer is N", so every intermediate step is a number too.
    """
    output = _COMMA_IN_NUMBER_RE.sub(r"\1\2", text)
    numbers = _NUMBER_RE.findall(output)
    return numbers[-1] if numbers else None


def clean_answer_text(text: str) -> str:
    """Normalize a gold answer the way the olmo-eval task does at load time.

    Mirrors ``_clean_short_answer``: the last number if there is one, otherwise the
    text unchanged. Vendored gold answers are already in this form, so this is
    idempotent; applying it anyway keeps the diagnostic correct if a future bank
    carries raw answer text instead.
    """
    return extract_last_number(text) or text


def _exact_match(predicted: str | None, gold: str) -> bool:
    """Compare as ``ExactMatchScorer`` does: strip surrounding space, ignore case."""
    if predicted is None:
        return False
    return predicted.strip().lower() == gold.strip().lower()


#: ``ItemResponse.metadata`` key marking a zero that is not the model's. Always present
#: on a generative response so a reader can tell "nothing was ungradable" from "this
#: report predates the distinction"; True only on a fabricated zero.
#:
#: Not to be confused with the manifest's ``ungradable_instances``, which counts
#: instances *vendoring* refused to emit. This one counts items that reached a
#: checkpoint and came back with no outcome.
UNGRADABLE_KEY = "ungradable"

#: Companion key carrying why the item could not be graded, present only when it could
#: not be. A bare count says a run is suspect; this says which of the two failures it
#: is, and the two want different fixes -- re-vendor the bank, or install the verifiers.
UNGRADABLE_REASON_KEY = "ungradable_reason"


@dataclass(frozen=True, slots=True)
class Verdict:
    """One graded completion: what was extracted, what was expected, and the outcome.

    ``gold`` is ``None`` for a benchmark that defines correctness without an expected
    answer, and ``detail`` is where such a grader puts whatever a reader would need to
    see why the verdict went the way it did -- for IFEval, which instruction the
    response broke. Without it the report would show a wrong answer with no completion
    fault visible anywhere in it, and re-grading from the report alone would be
    impossible.

    ``ungradable_reason`` separates the two ways a verdict can be ``correct=False``.
    Normally it is the model's answer being wrong, which is evidence about the model and
    is exactly what EAP should condition on. Set, it means the item carried nothing to
    decide against, so the zero is fabricated: still fed to EAP, because Research's
    online CAT scores such an item 0 and continues rather than abandoning a run that has
    already staged a checkpoint and booted a GPU, but recorded so that the fabrication
    is visible in the report instead of arriving as an unexplained low theta.
    """

    correct: bool
    extracted: str | None
    gold: str | None
    detail: Mapping[str, Any] = field(default_factory=dict)
    ungradable_reason: str | None = None


@runtime_checkable
class AnswerGrader(Protocol):
    """Decides whether a completion answers an item correctly.

    The swappable half of generative grading. Implementations see only text, never a
    model or a tokenizer, so a new answer format costs one class and no changes to
    sampling.
    """

    #: Recorded on every response so a report says which grader produced it.
    name: str

    def grade(self, completion: str, gold_answer: str) -> Verdict:
        """Grade ``completion`` against ``gold_answer``."""
        ...


@runtime_checkable
class ItemGrader(Protocol):
    """Decides correctness from the item itself rather than from a gold answer.

    The second grader shape, added for IFEval and kept separate from
    :class:`AnswerGrader` rather than folded into it. An IFEval item has
    ``gold_answer = None`` by construction: the prompt names verifier functions and
    per-instruction arguments, and a response is correct when every one of them passes.
    There is no string to compare against, so a grader that reduced the item to one
    would have to invent it.

    Two protocols with one method each, instead of one method taking both a gold and a
    metadata mapping, because the two are not alternatives within a single grader:
    :class:`LastNumberExactMatch` has no use for metadata and this grader has no use
    for a gold. Splitting them keeps each signature honest and leaves the numeric and
    ``math_latex`` graders exactly as they were.

    Attributes:
        name: Recorded on every response so a report says which grader produced it.
        required_metadata: Item metadata keys this grader decides on. Vendoring copies
            exactly these off the enumerated instance and refuses to emit an item
            missing any of them, and :func:`is_gradable` re-checks them before a
            checkpoint is loaded. Declaring them here rather than in the vendoring
            script keeps one definition of what the grader needs, so a grader cannot
            acquire a new input without the bank that feeds it carrying that input too.
    """

    name: str
    required_metadata: tuple[str, ...]

    def grade_item(self, completion: str, item: BenchmarkItem) -> Verdict:
        """Grade ``completion`` against whatever ``item`` says makes it correct."""
        ...


#: Either grader shape may sit in :data:`ANSWER_GRADERS`; the call site tells them apart.
Grader = AnswerGrader | ItemGrader


@dataclass(frozen=True, slots=True)
class LastNumberExactMatch:
    """Strict match on the last number in the completion.

    The Open LLM Leaderboard convention, which is the population the GSM8K bank was
    harvested from rather than a metric its release states; each bank's manifest records
    what is and is not known about its calibration. Strict deliberately: accepting
    "72 apples" where gold is "72", or falling back to substring containment, would mark
    items correct that the calibration counted wrong and shift theta upward against a
    bank that cannot shift with it.
    """

    name: str = "last_number_exact_match"

    def grade(self, completion: str, gold_answer: str) -> Verdict:
        """Extract the last number from ``completion`` and compare it to gold."""
        extracted = extract_last_number(completion)
        gold = clean_answer_text(gold_answer)
        return Verdict(correct=_exact_match(extracted, gold), extracted=extracted, gold=gold)


def _math_extract() -> Any:
    """Return ``olmo_eval.evals.extract.math``, the module a full MATH eval scores with.

    Imported rather than reimplemented, for the same reason the few-shot block is:
    ``MinervaMathScorer`` grades a full MATH eval by running this module's
    ``extract_math_answer`` and ``is_equiv`` over the same strings, and a second copy
    here would be a second definition of what counts as a correct answer.
    Reimplementing the comparison is worse still -- ``is_equiv`` tries sympy first and
    falls back to Hendrycks string normalization, and a diagnostic doing only one of
    the two would disagree with the calibration on a long tail of items in whichever
    direction its half was weaker.

    Lazy so this module still loads where ``olmo_eval`` is absent and only MCQ banks
    are being built.
    """
    try:
        from olmo_eval.evals.extract import math as math_extract
    except ImportError as exc:
        raise RuntimeError(
            "Symbolic math grading lives in olmo_eval "
            "(olmo_eval.evals.extract.math.extract_math_answer and is_equiv) and "
            "olmo_eval is not importable here. It is not vendored into the diagnostic "
            "on purpose: a local copy would drift from the scorer a full eval uses. "
            "Install the package, or do not run a math_latex bank -- there is no "
            "string-equality fallback, because one would silently mark every "
            "correctly-reformatted answer wrong and drive theta down against a bank "
            "that cannot move with it."
        ) from exc
    return math_extract


def has_final_answer(text: str) -> bool:
    """Whether ``text`` states an answer in either convention the MATH prompt teaches.

    The Minerva few-shot block ends every worked solution both ways -- a ``\\boxed{}``
    expression and a ``Final Answer: The final answer is $X$.`` line -- so a model that
    produced neither has not answered at all. Distinguishing that from a wrong answer
    matters because ``extract_math_answer`` never comes back empty: with nothing
    answer-shaped to find it normalizes the whole completion and returns that, which
    grades correctly but reads, in a report, as though the model said something it
    did not.
    """
    module = _math_extract()
    if module.last_boxed_only_string(text) is not None:
        return True
    return module.get_unnormalized_answer(text) != "[invalidanswer]"


@dataclass(frozen=True, slots=True)
class MathLatexEquivalence:
    """Symbolic equivalence against a LaTeX gold answer, as a full MATH eval scores it.

    Extraction and comparison are olmo-eval's own, composed the way
    ``MinervaMathScorer`` composes them: every candidate the extractor finds is tried
    against the gold, and any equivalence is a correct answer. Two properties follow
    that neither :class:`LastNumberExactMatch` nor plain string equality has, and the
    MATH bank needs both.

    Extraction is by ``\\boxed{}`` (or the Minerva ``Final Answer:`` line), not by the
    last number. Last-number extraction does not merely lose accuracy on a boxed
    expression, it stops reading the answer at all: it reduces both the completion and
    the gold to their final digit, so ``x^2`` and ``y^2`` both become ``2`` and the
    item is marked correct. Most MATH answers end in a digit, so most of the bank
    would be scored on that coincidence and look far easier than its difficulties say.

    Comparison is symbolic, and applies to numeric golds too. ``vendor_bank`` reads
    ``answer_type`` off the answer string, so a MATH gold of ``7`` classifies as
    ``"numeric"`` and one of ``\\frac{1}{2}`` as ``"text"``; grading the first half of
    a bank by exact match and the second half by equivalence would put two grading
    conventions inside one calibrated scale. Every MATH item is graded here, whatever
    its answer happens to look like.

    A completion with no ``\\boxed{}`` and no ``Final Answer:`` line is scored
    incorrect, with a warning, rather than raising. Refusing to score it would be the
    wrong call twice over. It is not a harness fault: failing to state an answer in
    the taught format is model behaviour, and it is behaviour the calibration already
    counted -- the leaderboard models this bank was fit on failed that way too, on the
    hardest items, and those failures are inside the difficulties EAP now treats as
    fixed. Dropping them here would leave a response set biased toward the items the
    model can format an answer for and return a theta that is too high with a healthy
    standard error. The warning is what carries the other reading: a run where most
    items warn is a prompt-convention failure, not a weak model, and that is visible
    in the log rather than only in an implausibly low theta.
    """

    name: str = "math_latex_equivalence"

    def grade(self, completion: str, gold_answer: str) -> Verdict:
        """Compare every extracted candidate against ``gold_answer`` symbolically."""
        module = _math_extract()
        gold = gold_answer.strip()
        candidates = [
            candidate for candidate in module.extract_math_answer(completion) if candidate.strip()
        ]
        if not has_final_answer(completion):
            log.warning(
                "No \\boxed{} answer and no 'Final Answer:' line in a completion being "
                "graded against gold %r; scoring it incorrect. Repeated across a run "
                "this is the prompt convention failing to take, not a weak checkpoint.",
                gold,
            )
        correct = any(module.is_equiv(candidate.strip(), gold) for candidate in candidates)
        return Verdict(correct=correct, extracted=candidates[0] if candidates else None, gold=gold)


def _ifeval_scoring() -> tuple[Any, Any, Any]:
    """Return olmo-eval's ``IFEvalScorer`` and the two types it grades over.

    Imported rather than reimplemented, for the reason :func:`_math_extract` is: this
    scorer *is* the definition of whether an IFEval response follows its instructions.
    It resolves each ``instruction_id`` against the IFBench verifier registry, fills in
    ``prompt_to_repeat`` for the verifiers that need the prompt back, and writes the
    per-instruction strict and loose pass lists to ``output.metadata["ifeval"]``. A
    local copy would be a second, drifting definition of a correct response, and a
    hand-rolled subset of the verifiers would silently pass the constraints it did not
    implement -- inflating theta against a bank that counted those failures.

    Lazy so this module still loads where ``olmo_eval`` is absent and only MCQ banks
    are being built.
    """
    try:
        from olmo_eval.common.scorers import IFEvalScorer
        from olmo_eval.common.types import Instance, LMOutput
    except ImportError as exc:
        raise RuntimeError(
            "IFEval constraint verification lives in olmo_eval "
            "(olmo_eval.common.scorers.IFEvalScorer) and olmo_eval is not importable "
            "here. It is not vendored into the diagnostic on purpose: a local copy "
            "would drift from the scorer a full eval uses. Install the package, or do "
            "not run an ifeval bank -- there is no fallback, because the only "
            "available one is to treat unverified constraints as satisfied, which "
            "marks every item correct and drives theta up against a bank that cannot "
            "move with it."
        ) from exc
    return IFEvalScorer, Instance, LMOutput


@dataclass(frozen=True, slots=True)
class IFEvalPromptStrict:
    """Every verifiable instruction in the prompt must pass, which is IFEval's metric.

    The IFEval bank was calibrated on ``prompt_level_strict_acc``: one binary per
    prompt, 1 only when every instruction the prompt carries is satisfied by the
    response exactly as written. All three of the alternatives the same verifier run
    also produces are wrong here. Instruction-level accuracy is a fraction rather than
    a binary and has no place in a 3PL bank. Loose accuracy retries each instruction
    against eight lightly edited variants of the response -- markdown stars stripped,
    first and last lines dropped -- and is uniformly the more forgiving of the two, so
    scoring against strict-calibrated difficulties with it would make every item look
    easier than its ``b`` and push theta up. ``IFEvalScorer.score``'s own return value
    *is* the loose number, which is why this reads the strict list out of the metadata
    the scorer writes and ignores what it returns.

    An item whose strict list comes back empty is a bank fault, not a wrong answer.
    Empty means nothing was verified, so there was never an outcome to record, and the
    0 that goes into the response pattern is the harness's rather than the model's.
    Vendoring refuses to emit such an item, so reaching this means the bank on disk was
    not produced by :mod:`~diagnostics.mcq_cat.styles.uni_mcq.scripts.vendor_bank`.
    It is nonetheless scored 0 and marked :data:`UNGRADABLE_KEY` rather than raised on,
    because one malformed row should not destroy a diagnostic that has already staged a
    checkpoint -- and because the marking is what keeps the difference legible
    afterwards, which raising and losing the whole report does not.

    An item the verifiers refuse outright is the same fault arriving one step earlier
    and is handled the same way. A ``kwargs`` list shorter than the instruction list and
    an ``instruction_id`` the registry does not hold both reach the scorer, which pairs
    the two strictly and indexes the registry directly, and the exception would
    otherwise leave the CAT loop and end the session with no report at all -- the very
    outcome the empty-strict case is written to avoid, for a bank that is wrong in the
    same way. A missing ``ifbench`` is not in that class and still raises: it makes every
    item ungradable rather than one, and installing it is the fix.
    """

    name: str = "ifeval_prompt_strict"
    required_metadata: tuple[str, ...] = ("instruction_id_list", "kwargs")

    def grade_item(self, completion: str, item: BenchmarkItem) -> Verdict:
        """Run every instruction verifier the item names and require all of them."""
        scorer_cls, instance_cls, output_cls = _ifeval_scoring()
        instruction_ids = list(item.metadata.get("instruction_id_list") or ())
        instance = instance_cls(
            question=item.question,
            gold_answer=None,
            metadata={
                # The scorer reads the prompt back for the verifiers that quote it
                # (prompt_to_repeat). The item's question is that prompt: the IFEval
                # task carries the prompt verbatim, with no template around it.
                "prompt": item.question,
                "instruction_id_list": instruction_ids,
                "kwargs": [dict(kw or {}) for kw in (item.metadata.get("kwargs") or ())],
            },
        )
        output = output_cls(text=completion)
        try:
            scorer_cls().score(instance, output)
        except ImportError as exc:
            raise RuntimeError(
                f"IFEval grading needs the IFBench verifier registry, which is a "
                f"declared dependency of this project (ifbench, a git URL) but is not "
                f"importable here ({exc}). Install it before running an ifeval bank."
            ) from exc
        except Exception as exc:  # noqa: BLE001 - a malformed item, not a broken harness
            reason = (
                f"the verifiers could not be run over this item's "
                f"{len(instruction_ids)} instruction(s) ({type(exc).__name__}: {exc}); "
                f"re-vendor the bank"
            )
            log.warning(
                "IFEval item %s could not be verified (%s: %s), so nothing about the "
                "response was checked; scoring it 0 and continuing. This is the bank "
                "rather than the checkpoint, and the report counts it separately from "
                "a wrong answer.",
                item.item_id,
                type(exc).__name__,
                exc,
            )
            return Verdict(
                correct=False,
                extracted=None,
                gold=None,
                detail={"instruction_id_list": instruction_ids, "strict": []},
                ungradable_reason=reason,
            )

        strict = list((output.metadata.get("ifeval") or {}).get("strict", ()))
        if not strict:
            reason = (
                f"the scorer verified none of the {len(instruction_ids)} instruction(s) "
                f"this item names, so no outcome was produced; re-vendor the bank"
            )
            log.warning(
                "IFEval item %s produced no instruction results, so nothing about the "
                "response was checked; scoring it 0 and continuing. This is the bank "
                "or the verifier registry rather than the checkpoint, and the report "
                "counts it separately from a wrong answer.",
                item.item_id,
            )
            return Verdict(
                correct=False,
                extracted=None,
                gold=None,
                detail={"instruction_id_list": instruction_ids, "strict": strict},
                ungradable_reason=reason,
            )
        return Verdict(
            correct=all(strict),
            extracted=None,
            gold=None,
            detail={"instruction_id_list": instruction_ids, "strict": strict},
        )


def _gpqa_scoring() -> tuple[Any, Any, Any, Any]:
    """Return olmo-eval's GPQA answer extractor, its scorer, and the two types it grades.

    Imported rather than reimplemented, for the reason :func:`_math_extract` and
    :func:`_ifeval_scoring` are. ``GPQATask.extract_answer`` is a three-stage cascade --
    the last ``ANSWER: X`` the response states, else the last parenthesized capital,
    else a bare trailing A-D -- and each stage past the first exists to catch a model
    that did not follow the system prompt. A local copy with only the first stage would
    score those responses as no answer at all, and the calibration counted them.

    The base ``GPQATask`` is constructed directly rather than fetched from the registry
    by name: extraction is defined once on it and does not vary across the twelve
    registered subsets, so naming one of them would suggest a distinction that does not
    exist. Construction touches no data; the dataset is only loaded by ``instances``.

    Lazy so this module still loads where ``olmo_eval`` is absent and only MCQ banks
    are being built.
    """
    try:
        from olmo_eval.common.scorers import MultipleChoiceScorer
        from olmo_eval.common.types import Instance, LMOutput
        from olmo_eval.evals.tasks.common.base import TaskConfig
        from olmo_eval.evals.tasks.gpqa import GPQATask
    except ImportError as exc:
        raise RuntimeError(
            "GPQA answer extraction lives in olmo_eval "
            "(olmo_eval.evals.tasks.gpqa.GPQATask.extract_answer) and olmo_eval is not "
            "importable here. It is not vendored into the diagnostic on purpose: a "
            "local copy would drift from the extractor a full eval uses. Install the "
            "package, or do not run a gpqa bank -- there is no fallback, because the "
            "only available one is to read the last capital letter in the response, "
            "which matches a chain of thought's stray initials as often as its answer."
        ) from exc
    return GPQATask(TaskConfig(name="gpqa")), MultipleChoiceScorer, Instance, LMOutput


@dataclass(frozen=True, slots=True)
class GPQACoTLetter:
    """The letter a chain of thought settles on, matched against the item's gold letter.

    GPQA's default is not a log-likelihood ranking. ``src/olmo_eval/evals/tasks/gpqa.py``
    registers each subset with an ``MCQAChatFormatter`` carrying an expert-scientist
    system prompt that asks the model to reason step by step and end with ``ANSWER: X``,
    a 1024-token greedy budget, and ``AccuracyMetric(scorer=MultipleChoiceScorer)``. The
    ``:mc`` and ``:bpb`` variants that would make it a log-likelihood benchmark exist and
    are not the default, and we never select a variant. Research never wired GPQA into
    its CAT at all -- ``adaptive/benchmarks.py`` on ``origin/Research`` registers nine
    benchmarks and GPQA is not among them -- so there is no CAT precedent to match and
    the task's own default is the convention.

    A gold-matched grader rather than an
    :class:`ItemGrader`, because that is what the task does: ``MultipleChoiceScorer``
    upper-cases ``instance.gold_answer`` and ``output.extracted_answer`` and compares
    them, both being letters. Nothing about the item beyond that letter takes part.

    The letter is only meaningful against a particular ordering of the choices, and
    GPQA reshuffles them per question from ``Random(f"{seed}:{index}")``. Vendoring
    therefore freezes both halves of that ordering into the item: the choice block goes
    into the stem in the order the shuffle produced, and ``metadata["gold_answer"]`` is
    the post-shuffle letter. Nothing re-derives either at run time, so a bank and the
    run that scores it cannot disagree about which choice ``B`` is.

    A response the extractor finds no letter in scores incorrect rather than raising, on
    the same terms as an unboxed MATH completion: failing to state an answer in the
    taught format is model behaviour, and it is behaviour inside the difficulties the
    bank holds.
    """

    name: str = "gpqa_cot_letter"

    def grade(self, completion: str, gold_answer: str) -> Verdict:
        """Extract the answer letter as the task does and score it as the task does."""
        task, scorer_cls, instance_cls, output_cls = _gpqa_scoring()
        output = output_cls(text=completion)
        output.extracted_answer = task.extract_answer(output)
        gold = gold_answer.strip().upper()
        instance = instance_cls(question="", gold_answer=gold)
        if output.extracted_answer is None:
            log.warning(
                "No answer letter found in a completion being graded against gold %r; "
                "scoring it incorrect. Repeated across a run this is the chat system "
                "prompt failing to take, not a weak checkpoint.",
                gold,
            )
        return Verdict(
            correct=bool(scorer_cls().score(instance, output)),
            extracted=output.extracted_answer,
            gold=gold,
        )


#: ``metadata["answer_type"]`` -> grader. GSM8K items carry ``"numeric"``; a MATH item
#: must carry ``"math_latex"``, an IFEval item ``"ifeval_strict"`` and a GPQA item
#: ``"gpqa_letter"``, all properties of the benchmark's notion of a correct answer
#: rather than of the individual answer. This table and :data:`FEWSHOT_SOURCES` are the
#: two places a second generative benchmark touches; an LLM-as-a-judge grader is one
#: more row here.
ANSWER_GRADERS: dict[str, Grader] = {
    "numeric": LastNumberExactMatch(),
    "math_latex": MathLatexEquivalence(),
    "ifeval_strict": IFEvalPromptStrict(),
    "gpqa_letter": GPQACoTLetter(),
}

#: Grader name -> the clause a report uses to say how an item was decided. Separate
#: from the sampling convention, which every generative bank shares, because this part
#: does not: last-number matching, symbolic equivalence and constraint verification are
#: three different definitions of correct, and a theta means nothing without knowing
#: which one produced it.
GRADER_NOTES: dict[str, str] = {
    "last_number_exact_match": (
        "strict-matched on the last number in the completion, mirroring olmo-eval's gsm8k task"
    ),
    "math_latex_equivalence": (
        "compared against the gold answer by olmo-eval's symbolic equivalence over the "
        "\\boxed{} or Final Answer: expression, mirroring MinervaMathScorer rather "
        "than the lm-eval math_verify extraction the bank was calibrated under; the "
        "two are close but not identical and can shift an item's difficulty"
    ),
    "ifeval_prompt_strict": (
        "checked against every verifiable instruction the prompt names, counting the "
        "item correct only when all of them pass, which is IFEval's "
        "prompt_level_strict_acc"
    ),
    "gpqa_cot_letter": (
        "read as the answer letter the chain of thought settles on, by olmo-eval's own "
        "three-stage GPQA extractor, and compared to the item's post-shuffle gold "
        "letter as MultipleChoiceScorer does -- the gpqa task's default, which is chat "
        "with chain of thought rather than the log-likelihood ranking its :mc variant "
        "would give"
    ),
}

#: Used when an item does not declare an ``answer_type``.
DEFAULT_ANSWER_TYPE = "numeric"


def get_answer_grader(answer_type: str) -> Grader:
    """Return the grader for ``answer_type``, or raise naming what is available."""
    try:
        return ANSWER_GRADERS[answer_type]
    except KeyError:
        raise ValueError(
            f"No answer grader for answer_type {answer_type!r}. Known types: "
            f"{', '.join(sorted(ANSWER_GRADERS))}. Add one to ANSWER_GRADERS in "
            f"diagnostics/mcq_cat/common/generative.py rather than loosening an "
            f"existing grader, which would change the scale of every bank using it."
        ) from None


def grader_note(grader_name: str) -> str:
    """Return the report clause for ``grader_name``, or a neutral one if unregistered."""
    return GRADER_NOTES.get(grader_name, f"graded by {grader_name}")


def gradable_gold(item: BenchmarkItem) -> str | None:
    """Return ``item``'s gold answer as the string a grader compares against, or ``None``.

    Blank counts as absent. A gold of ``""`` or of whitespace is not something any
    grader here can match: the numeric one compares an extracted number against it, the
    symbolic one asks sympy whether a candidate equals it, and the GPQA one upper-cases
    it and looks for that letter. All three come back false for every completion, so the
    item is marked wrong for every checkpoint and contributes a zero that is the bank's
    rather than the model's -- exactly the fabricated evidence :data:`UNGRADABLE_KEY`
    exists to keep out of a response pattern unannounced.

    The emptiness test is on the string form rather than on the value's own truthiness,
    because a legitimate gold of ``0`` is falsy and must not be refused; and it is a test
    at all rather than a plain ``is not None`` because that admitted the blank.
    """
    gold = item.metadata.get("gold_answer")
    if gold is None:
        return None
    text = str(gold)
    return text if text.strip() else None


def is_gradable(item: BenchmarkItem) -> bool:
    """Whether a generative item carries what its declared grader decides on.

    Two grader shapes mean two answers to "is this item complete". A gold-answer bank
    needs ``metadata["gold_answer"]``; an IFEval item has none and needs the constraint
    payload its grader declares in ``required_metadata`` instead. Asking the grader,
    rather than testing for a gold, is what lets :mod:`.grading`'s modality guard admit
    a verifier-scored bank without loosening into accepting an item that nothing can
    grade -- which would be scored incorrect for every model and read as a weak
    checkpoint.

    This is the *bank-level* half of that protection and the strict one: a bank with any
    such item is refused outright, before a checkpoint is staged, because the fault is
    in an artifact that can simply be re-vendored. The item-level half is
    :data:`UNGRADABLE_KEY`, which handles the one that gets past here -- a hand-edited
    bank, or an item whose payload only turns out to decide nothing once the verifiers
    run -- by scoring it 0 and recording that the 0 was not the model's.
    """
    try:
        grader = get_answer_grader(str(item.metadata.get("answer_type", DEFAULT_ANSWER_TYPE)))
    except ValueError:
        return False
    if isinstance(grader, ItemGrader):
        return all(item.metadata.get(key) for key in grader.required_metadata)
    return gradable_gold(item) is not None


def _gsm8k_fixed_fewshot() -> tuple[dict[str, str], ...]:
    """The eight fixed GSM8K examples the olmo-eval task prepends.

    Imported from ``olmo_eval`` rather than copied here: a second copy of a
    calibration-defining prompt is how the two drift apart without anyone noticing.
    The import is lazy so this module still loads in an environment that has no
    ``olmo_eval`` installed and is only building MCQ banks.
    """
    try:
        from olmo_eval.evals.tasks.constants.gsm_symbolic import GSM8K_FIXED_FEWSHOT
    except ImportError as exc:
        raise RuntimeError(
            "The gsm8k few-shot block lives in olmo_eval "
            "(olmo_eval.evals.tasks.constants.gsm_symbolic.GSM8K_FIXED_FEWSHOT) and "
            "olmo_eval is not importable here. It is not vendored into the diagnostic "
            "on purpose. Install the package, or set generative.num_fewshot: 0 in the "
            "style's config.yaml and accept that theta is then off the bank's "
            "calibrated scale."
        ) from exc
    return tuple(dict(example) for example in GSM8K_FIXED_FEWSHOT)


def _leaderboard_math_fixed_fewshot() -> tuple[dict[str, str], ...]:
    """The four fixed Minerva MATH examples the olmo-eval leaderboard_math task prepends.

    Imported for the same reason the GSM8K block is, and renamed for one more: the
    upstream constant calls the stem ``problem`` while every prompt template here keys
    it as ``question``. Renaming once, on the way out of the source, keeps each
    template able to render any few-shot block rather than needing to know which
    benchmark produced it.

    These four examples are also what makes the grader work at all. Each solution ends
    both with a ``\\boxed{}`` expression and with a ``Final Answer: The final answer is
    $X$.`` line, which are exactly the two forms :class:`MathLatexEquivalence` reads. A
    0-shot run teaches neither, so the model states its answer in prose and the grader
    finds nothing to compare.
    """
    try:
        from olmo_eval.evals.tasks.constants.minerva_math import MINERVA_MATH_FIXED_FEWSHOT
    except ImportError as exc:
        raise RuntimeError(
            "The leaderboard_math few-shot block lives in olmo_eval "
            "(olmo_eval.evals.tasks.constants.minerva_math.MINERVA_MATH_FIXED_FEWSHOT) "
            "and olmo_eval is not importable here. It is not vendored into the "
            "diagnostic on purpose. Install the package, or set "
            "generative.num_fewshot: 0 in the style's config.yaml and accept that "
            "theta is then off the bank's calibrated scale."
        ) from exc
    return tuple(
        {"question": example["problem"], "solution": example["solution"]}
        for example in MINERVA_MATH_FIXED_FEWSHOT
    )


#: Few-shot block name -> loader. Selected by ``GenerationConfig.fewshot_source``.
FEWSHOT_SOURCES: dict[str, Callable[[], tuple[dict[str, str], ...]]] = {
    "gsm8k": _gsm8k_fixed_fewshot,
    "leaderboard_math": _leaderboard_math_fixed_fewshot,
}


def _gpqa_system_prompt() -> str:
    """The expert-scientist system prompt the olmo-eval GPQA tasks send.

    Read off the registered task's own formatter rather than copied, for the reason the
    few-shot blocks are imported: this text is what teaches the ``ANSWER: X`` ending
    that :class:`GPQACoTLetter` extracts, so a divergent copy would leave the grader
    reading for a format the model was never asked to produce. It carries the whole
    instruction to reason step by step, which is the difference between GPQA's default
    and its ``:mc`` variant.

    ``get_task`` builds the task's config from class attributes and touches no data.
    ``gpqa_diamond`` names one of the three subsets our bank spans; all twelve
    registered GPQA tasks are constructed with the same formatter instance, and the
    prompt is asserted to be non-empty here rather than assumed, since an upstream
    change that dropped it would otherwise send a bare question and be graded as though
    it had not.
    """
    try:
        from olmo_eval.evals.tasks.common.registry import get_task
    except ImportError as exc:
        raise RuntimeError(
            "The gpqa system prompt lives in olmo_eval (the MCQAChatFormatter on the "
            "registered gpqa tasks) and olmo_eval is not importable here. It is not "
            "vendored into the diagnostic on purpose. Install the package, or do not "
            "run a gpqa bank -- without the prompt the model is not asked to reason or "
            "to end with 'ANSWER: X', and the grader would score its silence."
        ) from exc
    formatter = get_task("gpqa_diamond").config.formatter
    prompt = str(getattr(formatter, "system_prompt", "") or "")
    if not prompt:
        raise RuntimeError(
            "The registered gpqa_diamond task carries no system prompt, so the "
            "chain-of-thought instruction the grader depends on would not be sent. "
            "The task definition has changed; re-check the gpqa convention before "
            "running this bank."
        )
    return prompt


#: System prompt name -> loader. Selected by ``GenerationConfig.system_prompt_source``,
#: and read only by a ``chat_format`` bank, since a completion prompt has no system turn
#: to put one in.
#:
#: Separate from :data:`PROMPT_TEMPLATES` because the two answer different questions. A
#: template lays out the benchmark's content; a system prompt is a standing instruction
#: about how to answer, and for GPQA it is the instruction that produces the answer
#: format the grader reads. IFEval deliberately has no entry, and would not have one
#: even back when it was chat-format: its constraints are in the prompt text, and a
#: standing instruction beside them is one more constraint the verifiers were never
#: told about.
SYSTEM_PROMPTS: dict[str, Callable[[], str]] = {
    "gpqa": _gpqa_system_prompt,
}


def get_system_prompt(system_prompt_source: str) -> str:
    """Return the system prompt for ``system_prompt_source``, or raise naming the known ones."""
    try:
        loader = SYSTEM_PROMPTS[system_prompt_source]
    except KeyError:
        raise ValueError(
            f"Unknown system_prompt_source {system_prompt_source!r}. Known sources: "
            f"{', '.join(sorted(SYSTEM_PROMPTS))}."
        ) from None
    return loader()


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    """How one benchmark's prompt is laid out, few-shot block included.

    A benchmark's framing is not cosmetic. It is part of the convention its items were
    calibrated under, and it is also what teaches the model the answer form the grader
    extracts: GSM8K's examples end "So the answer is N" for the last-number extractor,
    MATH's end in ``\\boxed{}`` for the symbolic one. Hardcoding one benchmark's
    framing would force the next one to either edit shared code or run under a prompt
    its bank was never fit against, so the layout is data selected by
    ``GenerationConfig.prompt_style``.

    Attributes:
        name: Recorded in :data:`PROMPT_TEMPLATES` and used in error messages.
        question_template: The stem's framing. Formatted with ``question``; a stem
            containing braces is substituted in, not re-parsed, so LaTeX is safe.
        answer_prefix: Joins a few-shot example's stem to its worked answer. The live
            question is left without it, which is what cues the model to continue.
        fewshot_answer_key: Which key of a few-shot example holds the worked answer.
            Few-shot blocks are plain dicts loaded from :data:`FEWSHOT_SOURCES`, and
            each benchmark names it differently. ``None`` marks a benchmark that has no
            few-shot convention at all, and rendering an example under such a template
            raises rather than borrowing another benchmark's block.
        separator: Between examples, and before the live question.
    """

    name: str
    question_template: str
    fewshot_answer_key: str | None = None
    answer_prefix: str = " "
    separator: str = "\n\n"

    def render(self, question: str, examples: Sequence[Mapping[str, str]]) -> str:
        """Lay out ``examples`` followed by ``question``, which is left unanswered."""
        parts = [self._example(example) for example in examples]
        parts.append(self.question_template.format(question=question))
        return self.separator.join(parts)

    def _example(self, example: Mapping[str, str]) -> str:
        """Render one worked few-shot example, or say which pairing is inconsistent."""
        if self.fewshot_answer_key is None:
            raise ValueError(
                f"The {self.name!r} prompt template is 0-shot only, and num_fewshot is "
                f"not 0. Its benchmark was calibrated with no few-shot block and has no "
                f"convention for what a worked example would look like, so any block "
                f"reaching here belongs to another benchmark. Prepending one would put "
                f"the model behind a prompt its difficulties were never estimated "
                f"under, and EAP would absorb the whole difference into theta. Set "
                f"num_fewshot: 0 for this dataset in the style's config.yaml."
            )
        try:
            answer = example[self.fewshot_answer_key]
        except KeyError:
            raise ValueError(
                f"Few-shot example has keys {sorted(example)} but the {self.name!r} "
                f"prompt template reads its answer from {self.fewshot_answer_key!r}. "
                f"prompt_style and fewshot_source are set independently and these two "
                f"do not belong together; a prompt built from the wrong block would "
                f"teach the wrong answer format and be graded as though it had not."
            ) from None
        stem = self.question_template.format(question=example["question"])
        return stem + self.answer_prefix + answer


#: Prompt style name -> layout. Selected by ``GenerationConfig.prompt_style``, and
#: paired with a :data:`FEWSHOT_SOURCES` entry of the same name.
PROMPT_TEMPLATES: dict[str, PromptTemplate] = {
    "gsm8k": PromptTemplate(
        name="gsm8k",
        question_template="Question: {question}\nAnswer:",
        fewshot_answer_key="answer",
    ),
    "leaderboard_math": PromptTemplate(
        name="leaderboard_math",
        question_template="Problem:\n{question}\n\nSolution:",
        fewshot_answer_key="solution",
    ),
    # The IFEval prompt is the item, unframed: its instructions are addressed to the
    # model in the prompt text itself, and any wrapper -- a "Question:" cue, a worked
    # example, a stop sequence at a blank line -- is another constraint the verifiers
    # were never told about. Nothing is applied around it either: the bank is scored in
    # completion format, so what reaches the model is this string exactly, which is what
    # lm-eval's leaderboard_ifeval sends and what the leaderboard sent its pretrained
    # submissions.
    "ifeval": PromptTemplate(name="ifeval", question_template="{question}"),
    # Unframed for a different reason: a GPQA item's stem already *is* the user turn
    # MCQAChatFormatter builds, question and lettered choice block together, frozen at
    # vendoring time because the letters only mean anything against the shuffle that
    # produced them. The standing instruction to reason and end with "ANSWER: X" is a
    # system turn rather than part of the content, and is applied by the completer.
    "gpqa": PromptTemplate(name="gpqa", question_template="{question}"),
}


def get_prompt_template(prompt_style: str) -> PromptTemplate:
    """Return the layout for ``prompt_style``, or raise naming what is available."""
    try:
        return PROMPT_TEMPLATES[prompt_style]
    except KeyError:
        raise ValueError(
            f"Unknown prompt_style {prompt_style!r}. Known styles: "
            f"{', '.join(sorted(PROMPT_TEMPLATES))}."
        ) from None


@dataclass
class GenerationConfig:
    """Configuration for checkpoint loading and sampled-completion grading.

    Every sampling default is copied from the olmo-eval GSM8K task's
    ``SamplingParams``, so a diagnostic run and a full eval of the same checkpoint ask
    the model to do the same thing. A benchmark that needs something else -- MATH's
    ``Problem:``/``Solution:`` framing, its token budget, its stop sequences --
    overrides those defaults from the style's ``config.yaml`` rather than inheriting
    GSM8K's, since the defaults are one benchmark's convention and not a house style.
    """

    checkpoint_kind: str = "hf"  # "hf" or "olmo_core"
    max_new_tokens: int = 512
    temperature: float = 0.0
    stop_sequences: tuple[str, ...] = ("Question:", "\n\n")
    num_fewshot: int = 8
    fewshot_source: str = "gsm8k"
    prompt_style: str = "gsm8k"
    seed: int = 1234
    device_map: str = "auto"
    max_length: int | None = None
    #: Send the prompt as a single user turn through the checkpoint's chat template.
    #: A property of the benchmark, not of the checkpoint, and it decides which
    #: checkpoints can be scored at all: a ``chat_format`` bank refuses one with no
    #: template. False for the completion benchmarks, whose few-shot blocks already
    #: supply the framing, and false for ifeval, whose bank was harvested from a
    #: leaderboard that templated its chat submissions and not its pretrained ones --
    #: so both framings are in the calibration and the completion half is the one a base
    #: checkpoint can be measured against. True only for gpqa, which has no completion
    #: form anywhere: its system prompt is what teaches the answer format its grader
    #: extracts, and see :meth:`__post_init__` for why that cannot simply be pasted in.
    chat_format: bool = False
    #: Which entry of :data:`SYSTEM_PROMPTS` precedes the user turn, if any. A name
    #: rather than the text, so ``config.yaml`` stays a plain map of field names and the
    #: prompt itself keeps a single definition in ``olmo_eval``.
    system_prompt_source: str | None = None

    def __post_init__(self) -> None:
        self.stop_sequences = tuple(self.stop_sequences)
        if self.system_prompt_source is not None and not self.chat_format:
            raise ValueError(
                f"system_prompt_source is {self.system_prompt_source!r} but "
                f"chat_format is False, so there is no system turn to put it in. A "
                f"completion prompt would silently drop it, and for a benchmark whose "
                f"answer format that prompt teaches -- gpqa's 'end with ANSWER: X' -- "
                f"dropping it means grading the model against a format it was never "
                f"asked for. Set chat_format: true, or remove the source.\n"
                f"Prepending the text to the completion prompt is the third option and "
                f"is deliberately not offered. It reads like the way to score a base "
                f"checkpoint on gpqa, and it is not: no bank here was calibrated behind "
                f"it. The registered task sends a system turn through a chat template, "
                f"lm-evaluation-harness's leaderboard_gpqa sends no system prompt at all "
                f"and ranks log-likelihoods over the lettered options, and the harvest "
                f"these difficulties were fit from is the latter. EAP treats difficulty "
                f"as fixed, so a fourth framing moves theta by the whole of the "
                f"difference with the standard error still looking healthy. Score an "
                f"instruct checkpoint, or run a bank that has a completion form -- "
                f"ifeval does, and gsm8k and leaderboard_math are completion banks."
            )
        if self.temperature != 0.0:
            raise ValueError(
                f"temperature must be 0: this grader decodes greedily, so a nonzero "
                f"temperature ({self.temperature}) would be accepted and then ignored. "
                f"The bank was calibrated on single greedy completions; sampling would "
                f"make the same checkpoint score differently run to run and break "
                f"comparability with it."
            )
        if self.num_fewshot < 0:
            raise ValueError(f"num_fewshot must be >= 0, got {self.num_fewshot}")
        if any(not stop for stop in self.stop_sequences):
            raise ValueError(
                f"stop_sequences {list(self.stop_sequences)} contains an empty string. "
                f"An empty sequence is found at offset 0 of every completion, so "
                f"truncation would cut all of them to nothing and every item would be "
                f"graded on a blank response -- a whole run of fabricated zeroes that "
                f"reads as a checkpoint which answered nothing. A bank that wants no "
                f"truncation, as ifeval and gpqa do, sets stop_sequences: [] instead."
            )


def fewshot_examples(config: GenerationConfig) -> tuple[dict[str, str], ...]:
    """Return the few-shot block to prepend, honouring ``config.num_fewshot``.

    Prompting few-shot rather than 0-shot is a decision, not an inherited default. An
    item's calibrated ``b`` records how hard it was for the models ATLAS harvested,
    under whatever prompt they answered; EAP treats ``b`` as fixed truth, so any gap
    between that prompt and this one has nowhere to go except theta. A harder prompt
    yields a uniformly depressed ability estimate carrying a perfectly healthy
    standard error: confidently wrong, and invisible in the report. The examples also
    demonstrate the "So the answer is N" ending that the last-number extractor relies
    on, so 0-shot costs extraction accuracy on top of the scale shift.

    The default of 8 is the olmo-eval GSM8K task's, so a diagnostic run and a full
    eval of the same checkpoint ask the model the same thing. It is *not* known to
    match ATLAS. The vendored material records no shot count anywhere -- not in the
    release README, the fit scripts, or the response matrices -- and the harvested
    population is Open LLM Leaderboard v1, whose GSM8K convention is 5-shot. So 8 is
    the value that makes this diagnostic self-consistent, and agreement between our
    theta and a published ATLAS theta is unverified rather than expected. The value
    used is recorded on every response, and this stays configurable so a bank whose
    convention is established can be run under it.

    MATH is in the same position and is set to 4 for the same reason. Its calibration
    directory holds parameter CSVs, response matrices and figures and no record of how
    the responses were prompted, so the shot count it was fit under is unrecoverable
    from what was vendored. 4 is the olmo-eval ``leaderboard_math`` task's value, which
    keeps a diagnostic and a full eval in agreement, and it is also the count Open LLM
    Leaderboard v2 publishes for MATH-Hard; that the harvest came from that leaderboard
    makes the two likely to coincide, which is a better position than GSM8K's, but it
    is still an inference about the harvest rather than something the bank states.
    """
    if config.num_fewshot == 0:
        return ()
    try:
        loader = FEWSHOT_SOURCES[config.fewshot_source]
    except KeyError:
        raise ValueError(
            f"Unknown fewshot_source {config.fewshot_source!r}. Known sources: "
            f"{', '.join(sorted(FEWSHOT_SOURCES))}."
        ) from None
    return loader()[: config.num_fewshot]


def format_generative_prompt(item: BenchmarkItem, config: GenerationConfig) -> str:
    """Build the prompt for ``item`` under the configured style, few-shot block included.

    Each layout -- examples joined by a blank line, the live question last with a bare
    answer cue -- reproduces its olmo-eval task's ``format_request``. The join is also
    what the stop sequences are read against: it is the boundary the model has been
    shown between one question and the next. GSM8K stops on the blank line itself;
    MATH stops on the ``Problem:`` header that follows it, because a MATH solution may
    contain a blank line of its own and a GSM8K answer may not. See the argument on
    ``stop_sequences`` in the style's ``config.yaml``.
    """
    template = get_prompt_template(config.prompt_style)
    return template.render(item.question, fewshot_examples(config))


def truncate_at_stop(text: str, stop_sequences: Sequence[str]) -> str:
    """Cut ``text`` at the earliest stop sequence it contains.

    A local ``generate`` call has no server-side stop handling, so the sequences the
    task declares have to be applied here. This changes grades, not just tidiness: the
    GSM8K extractor takes the *last* number in the completion, so a model that runs on
    into a hallucinated next question would otherwise be graded on that question's
    answer.

    It cuts at the *earliest* match, and that cuts both ways. A sequence that can occur
    inside a legitimate answer deletes the rest of it, extracted answer included, and
    the completion is then graded as though the model had stopped there. Which
    sequences a bank declares is therefore a correctness question for that bank, argued
    per dataset in the style's ``config.yaml``.
    """
    cut = len(text)
    for stop in stop_sequences:
        found = text.find(stop)
        if found != -1:
            cut = min(cut, found)
    return text[:cut]


def apply_grader(grader: Grader, item: BenchmarkItem, completion: str) -> Verdict:
    """Grade ``completion`` for ``item`` with whichever grader shape ``grader`` is.

    The whole of the dispatch between the two shapes. An :class:`ItemGrader` decides
    from the item and is handed it; anything else compares against a gold answer, and an
    item with none cannot be graded at all. That second case is a vendoring fault, and
    ``check_bank_modality`` normally rejects the whole bank for it before a checkpoint is
    fetched, so an item reaching here with no gold means the guard was bypassed -- a
    hand-edited bank, or a scorer driven directly.

    It is scored 0 and marked rather than raised on. The zero is fabricated evidence
    about the model and must not pass for a wrong answer, which is what
    :data:`UNGRADABLE_KEY` records; but raising would abandon a run mid-flight and leave
    no report at all, which is the more expensive of the two failures and the less
    informative -- an aborted run says nothing about how many other items were affected.
    """
    if isinstance(grader, ItemGrader):
        return grader.grade_item(completion, item)
    gold = gradable_gold(item)
    if gold is None:
        log.warning(
            "Generative item %s carries no usable metadata['gold_answer'], so the %s "
            "grader had nothing to compare against; scoring it 0 and continuing. "
            "Re-vendor the bank -- the report counts this separately from a wrong "
            "answer.",
            item.item_id,
            grader.name,
        )
        return Verdict(
            correct=False,
            extracted=None,
            gold=None,
            ungradable_reason=(
                f"the item carries no usable metadata['gold_answer'] for "
                f"{grader.name!r} to compare against (it is "
                f"{item.metadata.get('gold_answer')!r}); re-vendor the bank"
            ),
        )
    return grader.grade(completion, gold)


def grade_completion(
    item: BenchmarkItem, completion: str, config: GenerationConfig
) -> ItemResponse:
    """Turn one raw completion into a graded :class:`ItemResponse`.

    Pure text in, response out, with no model involved, which is what makes the
    grading half of this module testable without a GPU.

    ``chosen_index`` is :data:`NO_CHOICE_INDEX` and ``choice_logprobs`` is empty
    because a generative item has no choice set; a fabricated index would make a
    generative response indistinguishable from an MCQ one in the report. The
    completion, the extracted answer and the gold answer are all carried in
    ``metadata`` so a finished run can be audited, or re-graded under a different
    grader, from its report alone. ``grader_detail`` appears only for a grader that
    produces one, so a bank that has always been gold-matched keeps the six keys its
    reports have always had.

    :data:`UNGRADABLE_KEY` is the exception to that and is always written, because its
    absence and its being False have to mean different things: a reader auditing a
    suspicious theta needs to distinguish a run where nothing was ungradable from one
    produced before the harness could tell.
    """
    grader = get_answer_grader(str(item.metadata.get("answer_type", DEFAULT_ANSWER_TYPE)))
    text = truncate_at_stop(completion, config.stop_sequences)
    verdict = apply_grader(grader, item, text)
    metadata: dict[str, Any] = {
        "modality": "generative",
        "grader": grader.name,
        "completion": text,
        "extracted_answer": verdict.extracted,
        "gold_answer": verdict.gold,
        "num_fewshot": config.num_fewshot,
        UNGRADABLE_KEY: verdict.ungradable_reason is not None,
    }
    if verdict.ungradable_reason is not None:
        metadata[UNGRADABLE_REASON_KEY] = verdict.ungradable_reason
    if verdict.detail:
        metadata["grader_detail"] = dict(verdict.detail)
    return ItemResponse(
        item_id=item.item_id,
        chosen_index=NO_CHOICE_INDEX,
        correct=verdict.correct,
        choice_logprobs=(),
        metadata=metadata,
    )


class GenerativeScorer:
    """A :class:`ScoringModel` that samples one completion per item and grades it.

    Sampling is injected as a plain ``prompt -> completion`` callable. That split is
    what lets the grading path be exercised offline, and it makes a different backend
    (vLLM, a hosted endpoint) a new completer rather than a new scorer.
    """

    def __init__(self, complete: Callable[[str], str], config: GenerationConfig) -> None:
        self.complete = complete
        self.config = config

    def score_items(self, items: Sequence[BenchmarkItem]) -> list[ItemResponse]:
        """Grade each item by sampling a greedy completion and extracting its answer."""
        responses: list[ItemResponse] = []
        for item in items:
            if item.choices:
                raise ValueError(
                    f"Item {item.item_id!r} has {len(item.choices)} answer choices, so "
                    f"it is an MCQ item reaching the generative grader. Sampling free "
                    f"text for it would ignore the choice set and grade against a gold "
                    f"answer it does not have. Grade it with common.inference instead."
                )
            prompt = format_generative_prompt(item, self.config)
            responses.append(grade_completion(item, self.complete(prompt), self.config))
        return responses


class _HFCompleter:
    """Greedy ``transformers`` generation, one prompt at a time."""

    def __init__(self, checkpoint_dir: Path, config: GenerationConfig) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

        self._torch = torch
        self.config = config
        set_seed(config.seed)

        self.tokenizer: Any = AutoTokenizer.from_pretrained(str(checkpoint_dir))
        if config.chat_format and not getattr(self.tokenizer, "chat_template", None):
            raise ValueError(
                f"{checkpoint_dir} defines no chat template, and this bank is scored "
                f"in chat format. The bank was calibrated on instruction-tuned models "
                f"answering a single user turn; sending the prompt raw instead would "
                f"still produce a completion and still be graded, and the whole gap "
                f"between a base continuation and an assistant reply would land in "
                f"theta with a healthy standard error beside it. Score a chat "
                f"checkpoint, or run a completion-format bank."
            )
        self.model: Any = AutoModelForCausalLM.from_pretrained(
            str(checkpoint_dir),
            torch_dtype="auto",
            device_map=config.device_map,
        )
        self.model.eval()

    def _render(self, prompt: str) -> str:
        """Wrap ``prompt`` in the checkpoint's chat template when the bank asks for chat.

        The chat turn belongs here rather than in :class:`PromptTemplate` because its
        text is the checkpoint's, not the benchmark's: two instruction-tuned models
        spell the same user turn with different special tokens, and the prompt a
        benchmark defines is the content inside it. A system turn is the same kind of
        thing one level up -- a standing instruction about how to answer rather than
        content -- so it is named in the config and its text loaded from olmo-eval here.
        """
        if not self.config.chat_format:
            return prompt
        messages: list[dict[str, str]] = []
        if self.config.system_prompt_source is not None:
            messages.append(
                {"role": "system", "content": get_system_prompt(self.config.system_prompt_source)}
            )
        messages.append({"role": "user", "content": prompt})
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def __call__(self, prompt: str) -> str:
        """Return the continuation of ``prompt``, with the prompt echo removed.

        ``do_sample=False`` is how ``transformers`` spells temperature 0; passing
        ``temperature=0`` to ``generate`` is rejected by the library.
        """
        torch = self._torch
        input_ids = self.tokenizer(self._render(prompt), return_tensors="pt")["input_ids"]
        if self.config.max_length is not None and input_ids.shape[1] > self.config.max_length:
            input_ids = input_ids[:, -self.config.max_length :]
        input_ids = input_ids.to(self.model.device)

        with torch.no_grad():
            generated = self.model.generate(
                input_ids,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
                pad_token_id=self._pad_token_id(),
            )
        return self.tokenizer.decode(generated[0][input_ids.shape[1] :], skip_special_tokens=True)

    def _pad_token_id(self) -> int | None:
        """Fall back to the EOS token when the tokenizer defines no pad token.

        Base checkpoints routinely have no pad token, and ``generate`` warns and pads
        with EOS anyway; naming it keeps the log clean.
        """
        pad = getattr(self.tokenizer, "pad_token_id", None)
        return pad if pad is not None else getattr(self.tokenizer, "eos_token_id", None)


#: Checkpoint kind -> the generative grading backend that reads it.
#:
#: The counterpart of ``inference.MCQ_SCORING_BACKENDS``, and separate from it on purpose:
#: the two modalities register independently, so a backend can exist for one and not the
#: other. That asymmetry is a real state rather than a hypothetical -- it is what the
#: native OLMo-core reader was in, and what a served backend would be in if it scored
#: log-probs before it generated.
GENERATIVE_BACKENDS: dict[str, Callable[[Path, GenerationConfig], ScoringModel]] = {
    "hf": lambda checkpoint_dir, config: GenerativeScorer(
        _HFCompleter(checkpoint_dir, config), config
    ),
}


def load_generative_model(checkpoint_dir: Path, config: GenerationConfig) -> ScoringModel:
    """Load a sampled-completion grading model for the configured checkpoint kind.

    The generative counterpart of ``inference.load_scoring_model``; :mod:`.grading`
    chooses between the two by dataset modality.
    """
    try:
        backend = GENERATIVE_BACKENDS[config.checkpoint_kind]
    except KeyError:
        raise ValueError(
            f"Unknown checkpoint_kind: {config.checkpoint_kind!r}. Registered generative "
            f"backends: {', '.join(sorted(GENERATIVE_BACKENDS))}."
        ) from None
    return backend(checkpoint_dir, config)


def _load_olmo_core(checkpoint_dir: Path, config: GenerationConfig) -> ScoringModel:
    """Load a raw OLMo-core checkpoint for generation (integration point).

    Deliberately not in :data:`GENERATIVE_BACKENDS`, so it is unreachable rather than
    half-wired, and the one place a reader is told why the two modalities disagree about
    ``olmo_core``. The MCQ half is no longer blocked --
    ``inference._OlmoCoreScoringModel`` rebuilds the model and resolves the tokenizer
    already, and this could reuse both. What it cannot reuse is the exemption that makes
    them work: that scorer skips ``_validate_token_ids`` and builds no
    ``GenerationConfig`` because a forward-only scorer never pads and never stops, and
    this checkpoint family writes ``pad_token_id == eos_token_id == 0``. Decoding needs a
    distinct EOS to stop on, so the blocker is real here and cannot be waved through the
    same way.
    """
    raise NotImplementedError(
        "olmo_core generation is a training-env integration point. The MCQ side reads "
        "this format natively (inference._OlmoCoreScoringModel); what is missing here is "
        "a prompt -> completion callable to hand to GenerativeScorer, and a stopping "
        "criterion for a checkpoint whose eos_token_id equals its pad_token_id."
    )
