"""Checkpoint load + sampled-completion grading for generative benchmarks.

This is the second of the harness's two grading schemes and is deliberately a module
of its own rather than a branch inside :mod:`.inference`. The two share no *grading*
beyond the :class:`~diagnostics.mcq_cat.base.ScoringModel` protocol: the MCQ path ranks
a fixed choice set by continuation log-likelihood and never generates a token, while
this path generates a completion and then decides whether the answer inside it is
right. Kept apart, neither can quietly acquire the other's behaviour, and a reader
asking "how is GSM8K graded" has one file to read.

They do share one *loader*, and only one. :class:`_OlmoCoreCompleter` builds its model
by instantiating ``inference._OlmoCoreScoringModel`` and taking the model, tokenizer and
device off it, because reading a raw OLMo-core checkpoint -- resolving its layout, its
tokenizer identifier, its device and its precision, and fabricating the pad id its
``GenerationConfig`` validator demands -- is a statement about the checkpoint format and
not about how a bank is graded. Two copies of it would be two places to fix the next
thing training changes, and the copy that rots is the one nobody has run. Nothing of
that class's *scoring* is used or reachable from here.

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
- :class:`GenerativeScorer` composes them into a ``ScoringModel``. It is also where a
  per-item generation budget is decided, because the item is in scope there and in no
  smaller piece; see :func:`item_token_budget` and :data:`BUDGET_AWARE_ATTR`.

Every one of those mirrors the corresponding olmo-eval task, except where the style's
``config.yaml`` argues its way off it in writing. GSM8K: comma separators stripped, the
last number in the completion taken, ``Question:`` and a blank line as stop sequences,
512 new tokens, greedy decoding. MATH: the Minerva ``Problem:``/``Solution:`` framing,
``\\boxed{}`` extraction and sympy equivalence, and -- the two departures --
``Problem:`` as the only stop sequence, matching lm-eval's ``leaderboard_math`` rather
than the task's added blank line, inside lm-eval's own 1024 new tokens. IFEval: the
prompt verbatim as a completion, no few-shot block, no stop sequences, and every named
instruction verified -- inside a budget derived per item from the length constraints that
item declares, which is the third departure and the only place a bank's token budget is
not a single number; see :func:`ifeval_token_budget`. GPQA: the expert-scientist system
prompt asking for step-by-step reasoning ending in ``ANSWER: X``, the question and its
lettered choices as the user turn, no stop sequences, 1024 new tokens, and the extracted
letter compared to the item's gold letter. That fidelity is the whole point. All four banks
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
import math
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..base import BenchmarkItem, ItemResponse, ScoringModel
from . import hf_config_patch, inference

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

#: The ``answer_type`` whose items carry their own generation budget; see
#: :func:`ifeval_token_budget`. Named rather than spelled inline because it is the
#: discriminator for the per-item budget as well as for the grader, and the two must
#: pick out the same bank.
IFEVAL_ANSWER_TYPE = "ifeval_strict"

#: There is deliberately no ``TOKENS_PER_WORD`` constant.
#:
#: Words are converted to tokens by the evaluated checkpoint's *own* tokenizer, measured
#: at run time by :meth:`GenerativeScorer._tokens_per_word`, and there is no code path
#: that converts without one: a scorer whose completer publishes no
#: :data:`TOKEN_COUNTER_ATTR` does not run the cascade at all and every item takes the
#: flat benchmark default. So a fixed multiplier has no job, and one was removed rather
#: than left unused.
#:
#: What the removed survey established, kept because it bounds how wrong the fallback
#: path can be: over the 480 English-response prompts, thirteen tokenizers spanning
#: WordPiece, SentencePiece-unigram, SentencePiece-BPE and byte-level BPE and vocabularies
#: from 30k to 250k gave aggregate ratios in a narrow band, 1.236 (OLMo-2-7B) to 1.376
#: (Llama-2-7b), with per-prompt p95s from 1.423 to 1.714. Vocabulary size did not predict
#: the ratio -- xlm-roberta at 250k tokenizes English no better than Llama-2 at 32k,
#: because its vocabulary is spent elsewhere. English prose therefore costs roughly 1.25
#: to 1.4 tokens per word on anything plausible, which is why the flat 1280 default is a
#: reasonable thing to fall back to and a poor thing to derive from.
#:
#: The measurement has to be a measurement because IFEval ships no reference answer
#: anywhere -- all 511 items have ``gold_index: -1``, empty ``choices``, and metadata
#: holding only ``answer_type``, ``modality``, ``instruction_id_list`` and ``kwargs``.
#: That is inherent to the benchmark rather than a vendoring gap: upstream
#: ``google/IFEval`` ships prompts and constraints and no gold response, because
#: correctness is the verifiers' verdict. There is no answer text to fit anything to, so
#: every constant below says what it was fit to instead.
#:
#: Two consequences worth stating rather than burying. The budget is now
#: **checkpoint-dependent**, so like the run-time EOS in :func:`eos_stop_sequences` it is
#: invisible to ``convention.check_runtime_convention``, which runs before the checkpoint
#: is fetched and can only check what the benchmark declares. And no model-specific value
#: may be written into ``config.yaml``; what that file pins is the ceiling and the
#: unit conversions, all of which are properties of the benchmark.

#: Tokens added on top of the converted word count, so that exceeding a limit is
#: *observable* rather than prevented.
#:
#: This is the reason the budget is not simply the converted count. A "less than 20
#: words" item graded at exactly 20 words of budget cannot distinguish a model that
#: obeyed the limit from one that would have run past it and was cut at the boundary by
#: the harness: both produce 20 words, and the verifier passes both. The extra tokens let
#: an over-running model visibly over-run and be graded as violating the constraint,
#: which is the measurement this bank exists to make.
#:
#: It does a second job for the other relation. For "at least N words", truncation at N
#: still satisfies *that* constraint -- the text really does reach N words -- but 6 of
#: the 30 "at least" items also carry a constraint on how the response *ends*
#: (``startend:end_checker``, ``startend:quotation``, ``detectable_content:postscript``,
#: ``combination:two_responses``), and a cut at exactly N destroys the ending and fails
#: that instruction instead. The headroom is what leaves room for it.
#:
#: 50 is the user-specified value and is not measured. It is comfortably above the
#: longest such ending in the bank (a postscript or a closing quotation is a handful of
#: tokens) and is small enough not to distort a short budget.
DETECTION_HEADROOM_TOKENS = 50

#: Words per sentence, for ``length_constraints:number_sentences`` (39 items).
#:
#: Measured over the 511 prompt texts split on sentence-final punctuation: mean 13.79,
#: median 12.67, p90 20.00, p95 24.00, max 62.00. 25 is p95 rounded up. Prompt text is
#: the proxy again, and here it is a conservative one in the direction that matters: an
#: IFEval prompt is imperative and clipped, while a model writing to a sentence quota
#: writes expository prose, so the true figure for a response is likelier to sit above
#: the prompt mean than below it -- which is why the quantile rather than the mean is
#: taken.
WORDS_PER_SENTENCE = 25

#: Sentences per paragraph, for the two ``num_paragraphs`` signals (39 items between
#: them). **Not measured** -- a paragraph boundary is not something the prompts exhibit
#: in the register the responses would, and there is no reference response to count.
#: 6 is picked at the conservative end: it makes a paragraph 150 words, which is a long
#: paragraph rather than a typical one, and the largest ``num_paragraphs`` in the bank
#: is 10, so the error this constant can cause is bounded and one-sided.
SENTENCES_PER_PARAGRAPH = 6

#: Words per paragraph. Derived rather than free, so the two constants above cannot
#: drift apart and a reader has one number to check instead of two.
WORDS_PER_PARAGRAPH = WORDS_PER_SENTENCE * SENTENCES_PER_PARAGRAPH

#: Words per bullet, for ``detectable_format:number_bullet_lists`` (31 items).
#: **Not measured**, for the reason above. Two sentences' worth is the conservative end:
#: a markdown bullet is usually a phrase or a single sentence, so this over-budgets
#: rather than under-budgets, and the largest ``num_bullets`` here is 10.
WORDS_PER_BULLET = 2 * WORDS_PER_SENTENCE

#: The budget for an item that declares no length signal *and* whose constraints survive
#: being cut short -- 84 of the 511. The only branch of the cascade that economises, and
#: therefore the only one that could produce a wrong grade if it is set too low.
#:
#: Two independent anchors, measured 2026-08-08, which is why it is not a round number:
#:
#: * A prior measurement of this repo put the typical demand of an item that declares no
#:   size near 52 tokens. 52 is a *median* and emphatically not a budget -- half of such
#:   answers run longer, and prose lengths are right-skewed -- so this is 4x it, 208, which
#:   for a right-skewed distribution lands around its p90 rather than its middle.
#: * The 84 items this branch actually governs have prompts of 12/37/69/215 tokens
#:   min/median/p90/max on SmolLM2. Three times the p90 prompt is 207.
#:
#: The two anchors agree to within one token, from different directions, and 208 is taken
#: as the larger. About 162 words at the 1.285 tokens per word measured here: a couple of
#: solid paragraphs, which is the right scale for "answer this, no length stated, in
#: lowercase" or "reply with one of three fixed phrases".
#:
#: Both failure modes were weighed and they are not symmetric: too small truncates a
#: compliant answer and fabricates a constraint violation, too large only spends decode
#: time. What makes 208 safe is not the multiple but the company it keeps -- every item
#: whose grade depends on the response *finishing* is routed away from this branch by
#: :data:`TRUNCATION_FRAGILE_IDS` and takes the ceiling instead.
UNCONSTRAINED_FLOOR_TOKENS = 208

#: There is deliberately no ``NON_ENGLISH_FLOOR_TOKENS``. An item demanding a non-English
#: response takes the ceiling, which under a 1280 cap is the whole benchmark default --
#: see the ``language:response_language`` branch of :func:`ifeval_token_budget`.

#: Instructions whose grade depends on the response not being cut short, so an item
#: carrying one is never economised: it takes the ceiling.
#:
#: Why this set exists. An item that declares no length signal has no *stated* size, but
#: that is not the same as having no size demand, and 188 of the 272 such items carry a
#: constraint that truncation breaks. Handing those the floor would cut the response and
#: fail a constraint the model was in the middle of satisfying -- the same failure class as
#: the MATH blank-line stop that deleted 40.9% of answers, which was treated as a
#: correctness bug rather than a tuning question. Detected from ``instruction_id_list``,
#: never by inspecting response text.
#:
#: **Ending-dependent** (83 no-signal items). The grader reads the *end* of the response,
#: so a cut tail is an automatic fail at any budget:
ENDING_DEPENDENT_IDS = {
    # Must end with a given phrase; the grader compares the final characters.
    "startend:end_checker",
    # The whole response must be wrapped in quotes, so the closing quote is last.
    "startend:quotation",
    # The postscript the grader looks for is by definition the last thing written.
    "detectable_content:postscript",
    # A cut anywhere leaves unbalanced braces, and the grader parses the whole response.
    "detectable_format:json_format",
}

#: **Count-dependent** (the remaining 105). Truncation lowers a count the grader is
#: checking. Less absolute than the ending cases -- the floor might well hold enough of
#: them -- but the direction is the same and the tie is broken toward correctness.
COUNT_DEPENDENT_IDS = {
    # N *highlighted* spans; a cut drops the last of them.
    "detectable_format:number_highlighted_sections",
    # N [placeholders]; likewise.
    "detectable_content:number_placeholders",
    # Every listed keyword must appear somewhere, and a cut can remove the last.
    "keywords:existence",
    # A keyword at least N times.
    "keywords:frequency",
    # A letter at least N times.
    "keywords:letter_frequency",
    # N all-caps words.
    "change_case:capital_word_frequency",
}

#: The union, which is what :func:`ifeval_token_budget` tests.
#:
#: **Anything absent from this set is being asserted truncation-tolerant, and that is a
#: claim rather than an absence of one.** The 84 items the floor governs carry only these
#: six instructions, and the assertion is made one by one:
#:
#: * ``keywords:forbidden_words`` (19) and ``punctuation:no_comma`` (18) are *negative*
#:   constraints. Truncation can only help satisfy them.
#: * ``change_case:english_lowercase`` (14) and ``change_case:english_capital`` (14) are
#:   global properties of the text that hold on any prefix of a compliant response.
#: * ``detectable_format:constrained_response`` (10) demands the response be exactly one of
#:   three fixed phrases, about 6 tokens. It is the shortest demand in the bank.
#: * ``detectable_format:title`` (14) is the weakest of the six claims and worth a reader's
#:   attention: a ``<<title>>`` may appear anywhere, and the assertion is that a model
#:   writes it at the top. If a model were found to append titles, this id belongs above.
#:
#: Two ids differ from the set the user proposed, both verified against the bank:
#:
#: * ``detectable_format:constrained_response`` was proposed and is excluded, per above.
#:   Including it would spend 1280 tokens on a 6-token answer for 10 items.
#: * ``detectable_format:multiple_sections`` was proposed and cannot apply: all 14 items
#:   carrying it declare a usable ``num_sections``, so it is a *length signal* handled by
#:   :func:`_declared_word_demand` and no item carrying it ever reaches this split.
TRUNCATION_FRAGILE_IDS = ENDING_DEPENDENT_IDS | COUNT_DEPENDENT_IDS

#: Named because two unrelated decisions turn on it: it doubles the budget in
#: :func:`ifeval_token_budget`, and it is the one instruction that suppresses the
#: leaked-EOS stop in :func:`eos_stop_sequences`.
TWO_RESPONSES_ID = "combination:two_responses"

#: Attribute a completer sets to publish the *text* of its checkpoint's end token.
#:
#: Read by :class:`GenerativeScorer` and handed to :func:`eos_stop_sequences`. It is
#: resolved from the live tokenizer and deliberately never written into ``config.yaml``
#: or a bank manifest: which string ends a generation is a property of the checkpoint,
#: while those two artifacts describe the benchmark. Recording it would make the bank's
#: convention checkpoint-specific and fail every run against a model that spells its end
#: token differently.
EOS_TEXT_ATTR = "eos_text"

#: Generation tokens the context window must be able to leave for an item to be worth
#: sending at all, used by :func:`fit_budget_to_context`.
#:
#: Taken from ``MIN_GEN`` in Research's ``tutor_cat/respgen/runner.py``, which is the
#: pipeline this clamp is modelled on, rather than invented here. It is only ever applied
#: as ``min(budget, MIN_GENERATION_TOKENS)``, so it cannot reject an item whose own budget
#: is smaller -- IFEval's shortest is 76 tokens, and that item is refused only if 76 will
#: not fit.
#:
#: The value it replaced there computed ``max(1, max_model_len - max_new_tokens)``, which
#: collapses to 1 whenever the context is no larger than the nominal budget -- true of
#: every model that branch ran at ``max_model_len <= 4096`` -- and then left-truncated the
#: prompt to its final token. The reserve has to key off the ACTUAL prompt length, which
#: is why :func:`fit_budget_to_context` takes one.
#:
#: 256 is also enough for a MATH solution at the median, which is the other bank the
#: clamp governs: the reference solutions run 255 tokens at the median and 329 at the
#: mean. It is not enough for a long one, which is the honest cost of running that bank
#: on a small-context model and is recorded per item.
MIN_GENERATION_TOKENS = 256

#: Attribute a completer sets to publish which tokenizer it resolved, for the report.
#:
#: Identity only -- nothing branches on it. It is recorded so a reader can tell from
#: ``cat_report.json`` alone which tokenizer produced that run's budgets, which matters
#: because the words-to-tokens ratio is a property of the tokenizer and two checkpoints
#: will not give an item the same budget.
TOKENIZER_ID_ATTR = "tokenizer_id"

#: Report key carrying how each item's generation budget was arrived at.
#:
#: Written onto every generative response so the style can aggregate a run-level block off
#: the responses rather than from a tally kept in parallel -- the same reason
#: ``_ungradable_block`` counts off them. A budget that was defaulted rather than computed
#: has to be visible in the report, because a run in the degraded mode looks entirely
#: normal otherwise: plausible theta, healthy standard error, nothing amiss.
BUDGET_KEY = "generation_budget"

#: ``BUDGET_KEY['source']`` values, named so the report and the tests agree on the spelling.
#:
#: ``BUDGET_FROM_CASCADE`` -- derived from the item's own declared length signals.
#: ``BUDGET_FROM_CEILING`` -- the bank's flat ``max_new_tokens``, either because the bank has
#: no per-item policy or because the item's cascade branch returns the ceiling.
#: ``BUDGET_FROM_CONTEXT`` -- lowered further by :func:`fit_budget_to_context`.
BUDGET_FROM_CASCADE = "cascade"
BUDGET_FROM_CEILING = "flat_ceiling"
BUDGET_FROM_CONTEXT = "context_clamped"

#: Attribute a completer sets to publish its checkpoint's context window in tokens.
#:
#: ``None`` when it cannot be determined, and then the clamp is skipped rather than
#: applied against a guessed window: a wrong context length would either refuse items that
#: would have generated fine or cap budgets for no reason, and both are worse than the
#: status quo of not checking.
CONTEXT_WINDOW_ATTR = "max_context_tokens"

#: Attribute a completer sets to publish ``text -> token count`` for its own tokenizer.
#:
#: The other half of :data:`EOS_TEXT_ATTR`, and what makes the per-item budget honest
#: rather than estimated: with it, a declared word count is converted by the tokenizer
#: that will actually do the generating. Without it there is no conversion at all and
#: :func:`item_token_budget` returns the flat benchmark default, because a words-to-tokens
#: ratio guessed for an unknown tokenizer is exactly the kind of quiet approximation this
#: whole budget exists to remove.
#:
#: Published by the completer rather than reached for by the scorer for the reason the
#: module docstring gives: the tokenizer is the backend's, and a scorer that imported one
#: would make a second backend a change to the scorer.
TOKEN_COUNTER_ATTR = "count_tokens"

_WORD_RE = re.compile(r"\S+")


def count_words(text: str) -> int:
    """Whitespace-delimited words, which is the unit IFEval's own verifiers count in.

    ``ifbench``'s ``NumberOfWords`` instruction tokenizes with a word-boundary regex, so
    this is deliberately the crude split rather than anything cleverer: the budget has to
    be denominated in the same unit as the constraint it is protecting. It is also why the
    ``language:response_language`` items cannot be converted at all -- for an unspaced
    script this returns 1 for a whole sentence.
    """
    return len(_WORD_RE.findall(text))


def _declared_word_demand(
    instruction_id: str, kwargs: Mapping[str, Any]
) -> int | None:
    """Words a single declared instruction obliges the response to produce, if any.

    ``None`` for an instruction that carries no length signal, which is most of them.
    Each branch converts to *words* rather than straight to tokens so that one measured
    words-to-tokens ratio applies to all of them, and so an item declaring several
    signals can be compared across them in one unit.
    """
    if instruction_id == "length_constraints:number_words":
        # The bank's own unit, so no conversion. ``relation`` is deliberately not read:
        # both relations need a budget that reaches the stated count, "at least" so a
        # compliant answer fits and "less than" so an over-running one can be seen to
        # over-run. See DETECTION_HEADROOM_TOKENS. Not reading it is also what makes the
        # duplicate case below come out right without a special branch.
        return _positive_int(kwargs.get("num_words"))
    if instruction_id == "length_constraints:number_sentences":
        sentences = _positive_int(kwargs.get("num_sentences"))
        return None if sentences is None else sentences * WORDS_PER_SENTENCE
    if instruction_id in (
        "length_constraints:number_paragraphs",
        # Carries num_paragraphs too, alongside the first word it pins. 12 items.
        "length_constraints:nth_paragraph_first_word",
    ):
        paragraphs = _positive_int(kwargs.get("num_paragraphs"))
        return None if paragraphs is None else paragraphs * WORDS_PER_PARAGRAPH
    if instruction_id == "detectable_format:number_bullet_lists":
        bullets = _positive_int(kwargs.get("num_bullets"))
        return None if bullets is None else bullets * WORDS_PER_BULLET
    if instruction_id == "detectable_format:multiple_sections":
        sections = _positive_int(kwargs.get("num_sections"))
        return None if sections is None else sections * WORDS_PER_PARAGRAPH
    return None


def _positive_int(value: Any) -> int | None:
    """``value`` as a positive int, or ``None`` if it is not one.

    A malformed ``kwargs`` entry must not be able to produce a budget of 0 -- that would
    generate nothing and grade the item on an empty response -- so anything that is not
    a usable count is dropped and the item falls through to the floor.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value > 0 else None


def ifeval_token_budget(
    item: BenchmarkItem,
    *,
    ceiling: int,
    tokens_per_word: float,
    count_tokens: Callable[[str], int],
    unconstrained: int = UNCONSTRAINED_FLOOR_TOKENS,
) -> int:
    """Tokens to allow ``item``, derived from the length signals it declares.

    A signal *cascade*, not a lookup: ``instruction_id_list`` is a list and items
    routinely declare several constraints at once, so every declared signal is converted
    and the largest demand wins. An item that is both "at least 300 words" and
    ``combination:two_responses`` needs about twice the 300-word budget, not the
    300-word budget.

    The cascade runs over all 511 items uniformly. An explicit
    ``length_constraints:number_words`` is one branch like any other and does not
    short-circuit the rest, because the bank does not let it: of the 44 items declaring a
    word count, 5 declare a second length signal as well -- 3 with
    ``detectable_format:number_bullet_lists`` and 2 with ``combination:two_responses`` --
    and for those last 2 a word-count-wins rule would hand an item asking for N words in
    each of two responses a single N-word budget and truncate the second. Across the bank
    308 items declare none of these signals, 194 declare one and 9 declare two.

    The three stages are different in kind and compose in this order:

    1. **The answer.** The largest per-signal word demand, converted at the item's
       words-to-tokens ratio, plus :data:`DETECTION_HEADROOM_TOKENS`. An item declaring no
       signal is split rather than given one number: if it carries a constraint truncation
       would break it takes ``ceiling``, and only if its constraints survive being cut
       short does it take ``unconstrained``. See :data:`TRUNCATION_FRAGILE_IDS` -- 188 of
       the 272 no-signal items are fragile, so the economising branch governs 84.
    2. **The echo**, added on top. ``combination:repeat_prompt`` (35 items) obliges the
       response to reproduce the request verbatim *and then answer it*, so its cost is
       additive rather than a maximum -- and an item whose only signal is the echo still
       needs a whole answer's worth of budget after it, which is why stage 1 falls back
       to ``unconstrained`` rather than to zero. Getting this wrong is not a rounding
       error: treating the echo as the item's total demand gives these 35 items 64 to 108
       tokens, enough for the echo and nothing else, and truncates every one of them.
    3. **The multiplier.** ``combination:two_responses`` (24 items) asks for two
       separated responses, so the whole of stages 1 and 2 doubles.

    Everything is then clamped to ``ceiling``, and the cascade therefore only ever
    *lowers* a budget. See :func:`item_token_budget` for what that costs.

    Args:
        item: The IFEval item, read for ``metadata['instruction_id_list']`` and the
            parallel ``metadata['kwargs']``.
        ceiling: Hard upper bound, the bank's configured ``max_new_tokens``. No derived
            budget may exceed it.
        tokens_per_word: Words-to-tokens ratio measured on the evaluated checkpoint's own
            tokenizer. There is no default: a caller without a tokenizer has no business
            running the cascade and :func:`item_token_budget` does not let it.
        count_tokens: That tokenizer's ``text -> token count``, used for the two things
            that can be counted exactly rather than converted -- the echoed prompt, and
            the uppercase ratio for an ALL-CAPS item.
        unconstrained: Budget for an item declaring no length signal.

    Returns:
        A positive token count, at most ``ceiling``.
    """
    instruction_ids = list(item.metadata.get("instruction_id_list") or ())
    raw_kwargs = list(item.metadata.get("kwargs") or ())
    # A list of pairs, never a dict keyed by instruction id. An id can repeat within one
    # item -- the bank holds 46 number_words constraints across 44 items -- and a dict
    # would keep the last and drop the rest without anything to show it had happened.
    #
    # Both duplicates are a bracketing pair, verified against the bank rather than
    # assumed: af6a27b556603a01 declares "at least 600" together with "less than 701",
    # and 1e657beb0dd15c56 declares "at least 100" together with "less than 121". Taking
    # the maximum over the pair follows the *upper* bound, which is the safe reading: the
    # model has to be able to reach the upper limit for the verifier's judgement of it to
    # mean anything, and a budget set at the lower bound would cut the response at 600
    # words and record a pass on "less than 701" that the harness produced rather than
    # the model. The lower bound needs nothing of its own, being already covered.
    #
    # Zipped rather than indexed for a second reason: a kwargs list shorter than the
    # instruction list is a malformed bank, which IFEvalPromptStrict already scores 0 and
    # marks ungradable. Budgeting must not be the thing that raises first, so a short
    # list simply contributes no demand.
    declared = list(
        zip(instruction_ids, [dict(kw or {}) for kw in raw_kwargs], strict=False)
    )
    signals = set(instruction_ids)

    # The non-English exemption, taken before anything is converted.
    #
    # Every conversion below is an English measurement, and a run-time tokenizer does not
    # rescue it: the ratio has to be measured on text in the item's *target* language and
    # no sample of that exists anywhere in the bank -- the prompt is English asking for a
    # Marathi response, so tokenizing the prompt measures the wrong language. Measured
    # 2026-08-08 across seven tokenizers on target-language samples, a non-English response
    # costs 2.0x (Portuguese) to 22.3x (Tamil) what an English one does per word, with Thai
    # at 158x because it is unspaced and ``count_words`` calls a whole sentence one word.
    # A budget converted at an English ratio would land 2 to 22 times too small.
    #
    # So these items bypass conversion and take the ceiling. Under a 1280 cap that is the
    # whole benchmark default, which is the clean answer: it is the most the cascade is
    # allowed to hand out, it is what these items would have got with no cascade at all,
    # and it needs no constant of its own. 31 items; 26 declare no length signal and would
    # otherwise drop to ``unconstrained``, and 5 declare one -- 914b4ebdd73c5cc1 (Marathi,
    # 3 paragraphs) is the concrete hazard, since it is the only one of the five whose
    # signal converts to a word count and it would otherwise be handed 450 English words'
    # worth of tokens for a Marathi answer.
    #
    # Detected from ``instruction_id_list`` rather than by sniffing the prompt text, which
    # would be a language guess on top of a ratio guess.
    if "language:response_language" in signals:
        return ceiling

    # Ending-dependent constraints take the ceiling whatever else the item declares, and
    # this is deliberately checked *before* any word count is read.
    #
    # A declared word count does not rescue these items, because a declared count is not an
    # upper bound on a compliant answer. 23 items carry both an ending-dependent constraint
    # and a length signal, and 6 of them state a *minimum* -- f0da3bf76abece71 is "at least
    # 400 words" wrapped in quotes, adc7b04618c125b6 "at least 800", 9baf1cb36bafc2bb "at
    # least 900" ending in a fixed phrase. Converting the minimum gives f0da3bf76abece71
    # 564 tokens; a compliant 600-word quoted answer is then cut and loses its closing
    # quote, and the grader records a failure the harness produced. Deriving from a lower
    # bound would be a regression against the flat cap these items had before, which is the
    # one thing this change must not do.
    #
    # "less than N" plus an ending constraint is the tolerable case -- there the count is a
    # genuine upper bound -- but it is not separated out, because IFEval's strict prompt
    # accuracy needs every constraint satisfied, so the only thing separating it would buy
    # is a different name for the same failure.
    if signals & ENDING_DEPENDENT_IDS:
        return ceiling

    # Uppercase is a different tokenization regime, not a rounding difference on the same
    # one, so an ALL-CAPS item is measured under ``str.upper`` rather than converted at the
    # as-written ratio. Measured on the checkpoint's own tokenizer because the penalty does
    # not transfer between them: in the earlier survey it ran from nothing at all (the
    # uncased WordPiece tokenizers lowercase their input) to 84% (Llama-2, 1.376 -> 2.531).
    # ``change_case:english_capital``, 25 items, 5 of which also declare a length signal.
    #
    # The other case and punctuation constraints in the bank were checked because they
    # looked like the same hazard and are not: ``change_case:english_lowercase`` and
    # ``punctuation:no_comma`` both leave the ratio within 1% of as-written, so neither
    # gets a branch.
    ratio = tokens_per_word
    if "change_case:english_capital" in signals:
        prompt_words = count_words(item.question)
        if prompt_words:
            ratio = max(ratio, count_tokens(item.question.upper()) / prompt_words)

    demands = [
        words
        for instruction_id, kwargs in declared
        if (words := _declared_word_demand(instruction_id, kwargs)) is not None
    ]
    if demands:
        budget = math.ceil(max(demands) * ratio) + DETECTION_HEADROOM_TOKENS
    elif signals & COUNT_DEPENDENT_IDS:
        # No stated size, but a count the grader checks that truncation would lower. The
        # floor is plausibly enough for most of these; the ceiling is taken because the
        # cost of being wrong is a wrong grade and the cost of being generous is decode
        # time. See TRUNCATION_FRAGILE_IDS.
        budget = ceiling
    else:
        budget = unconstrained

    # The echoed request, counted exactly rather than converted: the text is right here
    # and the tokenizer that will generate it is in hand, so there is nothing to estimate.
    # ``prompt_to_repeat`` is preferred over ``item.question`` because the kwarg is what
    # the verifier compares against, and the two can differ once a prompt template has
    # wrapped the question.
    for instruction_id, kwargs in declared:
        if instruction_id == "combination:repeat_prompt":
            echoed = str(kwargs.get("prompt_to_repeat") or item.question)
            budget += count_tokens(echoed)

    if TWO_RESPONSES_ID in signals:
        budget *= 2

    return min(budget, ceiling)


def item_token_budget(
    item: BenchmarkItem,
    config: GenerationConfig,
    *,
    tokens_per_word: float | None = None,
    count_tokens: Callable[[str], int] | None = None,
) -> int:
    """Tokens to allow ``item``, per item where the bank and the backend both support it.

    ``config.max_new_tokens`` is a **ceiling**, not a target. Every other generative bank
    gets it flat, and for ifeval the cascade may only lower an item below it.

    Two gates, and both are deliberate:

    * **The bank.** Dispatches on ``answer_type`` and not on anything looser, so the
      cascade reaches exactly the bank whose items carry the metadata it reads. A GSM8K or
      MATH item declares no ``instruction_id_list``, and a cascade over an absent field
      would hand it the unconstrained floor -- 512 tokens against MATH's derived 2048, cut
      mid-derivation on the longest 26 of its 1,183 items.
    * **The tokenizer.** No tokenizer, no cascade: every item takes the flat ceiling. A
      words-to-tokens ratio is the load-bearing step in every branch, and the only honest
      way to get one is to ask the tokenizer that will do the generating. Falling back to
      a fixed multiplier would put a guess underneath a budget whose entire purpose is to
      stop the harness from truncating a compliant answer and grading it as a violation.
      In a real run the tokenizer is always loaded, so this path is for tests and for a
      future backend that has not published one yet.

    The consequence of the ceiling, recorded because it is a real loss and not a rounding
    one: a handful of items declare constraints that convert to more than the ceiling and
    are truncated. On SmolLM2 those are 7 items wanting 1310 to 3367 tokens, led by
    ab5ca590f2d20f37 at 2500 declared words. They are cut at exactly the point the
    calibration population was cut -- upstream ``olmo_eval`` and lm-eval both generate
    IFEval at 1280 -- so the truncation is inherited from the harness that produced these
    difficulties rather than introduced here. That is the whole argument for a ceiling
    rather than a floor: theta is only interpretable on the scale the difficulties came
    from.
    """
    answer_type = str(item.metadata.get("answer_type", DEFAULT_ANSWER_TYPE))
    if answer_type != IFEVAL_ANSWER_TYPE:
        return config.max_new_tokens
    if tokens_per_word is None or count_tokens is None:
        return config.max_new_tokens
    return ifeval_token_budget(
        item,
        ceiling=config.max_new_tokens,
        tokens_per_word=tokens_per_word,
        count_tokens=count_tokens,
    )


def fit_budget_to_context(
    budget: int, prompt_tokens: int, context_window: int
) -> int | None:
    """``budget`` reduced to what the context window actually leaves, or ``None`` to refuse.

    The last of three stages, and the ordering is the thing to be clear about because a
    reader will ask which of them wins: :func:`ifeval_token_budget` derives a demand, the
    bank's ``max_new_tokens`` caps it, and this caps whatever survived that. Each stage can
    only lower, so the answer is simply whichever is smallest, and this one is last because
    it is the only hard constraint of the three -- the other two are conventions, while a
    prompt and its continuation physically have to fit in the window.

    **This is insurance rather than a fix for a live bug, and the distinction is worth
    keeping straight.** Measured over the bank, IFEval prompts run 51/81/387 tokens
    median/p90/max, the 35 ``combination:repeat_prompt`` items topping out near 112. Against
    a 2048-token window the longest prompt still leaves 1936, well clear of the 1280
    ceiling, so this is inert for this bank on any model with a context of 2048 or more and
    only begins to bind below roughly 1667. Nothing in the tree today trips it. It is here
    because the point of this work is a correct pipeline for future submitters rather than a
    score for the current checkpoint, and a small-context model is exactly the submitter
    that would otherwise be mangled quietly.

    The formula is Research's ``_fit_prompt_and_budget``, and its docstring records the bug
    worth not repeating: an earlier version computed ``max(1, max_model_len -
    max_new_tokens)``, which collapsed to 1 whenever the window was smaller than the
    nominal budget -- every model they ran at 4096 or below -- and then left-truncated the
    prompt to its final token. So the clamp is a function of the *measured* prompt length
    and never of the nominal budget, and it cannot collapse: what it returns is either at
    least ``min(budget, MIN_GENERATION_TOKENS)`` or nothing at all.

    **Where this deliberately departs from Research: it refuses instead of left-truncating.**
    Research keeps the tail of an over-long prompt, which is right for a chat transcript,
    where the tail is the student's latest turn and the front is stale history. For IFEval
    it would be silently destructive. The instruction being graded *is* the prompt, and the
    verifiers do not read the prompt -- they read ``metadata['kwargs']`` from the bank -- so
    dropping the front of it does not soften what is checked. It produces a model that was
    never shown the constraint it is about to be graded on, and a failure recorded against
    it is the harness's, not the model's. That is fabricated evidence, which is what
    :data:`UNGRADABLE_KEY` exists to keep out of a response pattern unannounced; the item is
    scored 0, marked, and surfaced in ``cat_report.json``'s ``ungradable`` block with its id
    and reason, where it reads as a configuration problem rather than as a wrong answer.

    Args:
        budget: Tokens the earlier stages settled on.
        prompt_tokens: The *rendered* prompt's real length, chat template and any fewshot
            block included, because that is what occupies the window.
        context_window: The checkpoint's context length in tokens.

    Returns:
        The budget to generate with, or ``None`` if the window cannot hold the prompt plus
        a usable generation, meaning the item must not be sent.
    """
    # Never more than min(budget, MIN_GENERATION_TOKENS), so an item whose own demand is
    # tiny is not refused for failing to leave room it never wanted.
    reserve = min(budget, MIN_GENERATION_TOKENS)
    remaining = context_window - prompt_tokens
    if remaining < reserve or remaining < 1:
        return None
    return max(1, min(budget, remaining))


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

    The end-of-text token this bank's exemplars are closed with is deliberately NOT added
    here, even though this function is the transformation boundary and adding it would be
    one line. This returns the same four examples whoever is being scored; the token is
    the scored checkpoint's. Appending it here would need this function to be handed a
    tokenizer, which would make a source of prompt data depend on a loaded model, and the
    result would no longer be cacheable or comparable across runs. It is appended by
    :meth:`PromptTemplate.render`, where the benchmark's declared convention
    (``exemplars_end_with_eos``) and the model's string meet for one render.
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


def resolve_eos_token(tokenizer: Any) -> tuple[str | None, int | None]:
    """The end-of-text token ``tokenizer`` uses, as ``(text, id)``.

    Read off the loaded checkpoint's own tokenizer, never written down anywhere. The
    token is a property of the model and not of the benchmark: this checkpoint family
    resolves ``HuggingFaceTB/SmolLM2-135M`` from its ``dataset.tokenizer.identifier``,
    which spells it ``<|endoftext|>`` at id 0, and the next submitter's may spell it
    ``</s>`` or ``<|im_end|>`` at some other id. A literal in ``config.yaml`` or in an
    exemplar constant would be that model's answer frozen into the benchmark's
    description of itself, and for every other model it would be ordinary text -- shown
    to the model as characters to imitate and matched by nothing.

    Both halves can be absent independently and neither is an error:

    - No ``eos_token_id``: nothing to stop on and nothing to teach. ``(None, None)``,
      and the caller falls back to ``max_new_tokens`` as the only bound.
    - An id but no text, or text that does not survive a round trip: stop on the id but
      teach nothing. Inserting a string the tokenizer reads as ordinary characters is
      strictly worse than inserting nothing, because it teaches the model to type them.

    The round trip is checked rather than assumed for the same reason ``num_fewshot`` is
    recorded rather than trusted. ``tokenizer.eos_token`` is a label; what decides
    whether the exemplars teach anything is whether that label encodes back to the id
    ``generate`` halts on, which holds when the tokenizer registers it as an added
    special token -- SmolLM2 does, ``normalized: false`` and ``special: true`` -- and
    fails when it does not.

    Wrapped in ``except Exception`` because this runs against whatever object a backend
    hands over, and a probe that cannot be performed is a reason to teach nothing rather
    than to abandon a run that would otherwise score fine.
    """
    token_id = getattr(tokenizer, "eos_token_id", None)
    if token_id is None:
        return None, None
    text = getattr(tokenizer, "eos_token", None)
    if not text:
        return None, token_id
    try:
        encoded = list(tokenizer(text, add_special_tokens=False)["input_ids"])
    except Exception:  # noqa: BLE001 - see the docstring's last paragraph
        return None, token_id
    if encoded != [token_id]:
        log.warning(
            "This checkpoint's tokenizer spells its end-of-text token %r but encodes "
            "that string to %s rather than to its own eos_token_id %s, so it is "
            "ordinary text here. Generation will still stop on the id; the few-shot "
            "exemplars will not be closed with the token, because doing so would teach "
            "the model to spell those characters instead of to stop.",
            text,
            encoded,
            token_id,
        )
        return None, token_id
    return text, token_id


def exemplar_eos_token(config: GenerationConfig, eos_token: str | None) -> str | None:
    """The end-of-text text this run should close its exemplars with, or ``None``.

    Three gates, and each rules out a case where appending would be meaningless rather
    than merely unnecessary. No token resolved from the checkpoint: nothing to append.
    ``num_fewshot`` of 0: no exemplar exists to append it to, and with nothing having
    demonstrated the token there is also no leaked spelling of it to catch. And a prompt
    style that does not declare the convention -- gsm8k, ifeval, gpqa -- keeps the
    prompt its own difficulties were estimated behind, since this is a deviation from
    the calibration harness and is argued per bank rather than applied house-wide.

    Note the asymmetry with :meth:`_HFCompleter._eos_kwargs`, which is deliberate. The
    stopping id is passed for *every* bank, because a checkpoint that cannot stop is
    off-convention for all of them and halting early changes no text. The exemplar
    suffix changes the prompt, so it goes only where it has been argued.
    """
    if not eos_token or config.num_fewshot == 0:
        return None
    if not get_prompt_template(config.prompt_style).exemplars_end_with_eos:
        return None
    return eos_token


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
        exemplars_end_with_eos: Whether each worked example is closed with the
            checkpoint's end-of-text token. A boolean and not the token, which is the
            whole point: the benchmark declares the convention and the model supplies
            the string, so nothing model-specific is frozen here. ``False`` everywhere
            but ``leaderboard_math``, where it is the one deviation from lm-eval's
            prompt and is argued in ``config.yaml`` beside ``stop_sequences``.
    """

    name: str
    question_template: str
    fewshot_answer_key: str | None = None
    answer_prefix: str = " "
    separator: str = "\n\n"
    exemplars_end_with_eos: bool = False

    def render(
        self,
        question: str,
        examples: Sequence[Mapping[str, str]],
        *,
        eos_token: str | None = None,
    ) -> str:
        """Lay out ``examples`` followed by ``question``, which is left unanswered.

        ``eos_token`` is the loaded checkpoint's end-of-text string, or ``None`` when
        there is no tokenizer to ask -- which is the case for every caller that builds a
        prompt offline. ``None`` renders exactly what this template rendered before the
        setting existed, so the degradation is the old behaviour rather than a new one.
        """
        suffix = eos_token if (self.exemplars_end_with_eos and eos_token) else ""
        parts = [self._example(example, suffix) for example in examples]
        parts.append(self.question_template.format(question=question))
        return self.separator.join(parts)

    def _example(self, example: Mapping[str, str], suffix: str = "") -> str:
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
        return stem + self.answer_prefix + answer + suffix


#: Prompt style name -> layout. Selected by ``GenerationConfig.prompt_style``, and
#: paired with a :data:`FEWSHOT_SOURCES` entry of the same name.
PROMPT_TEMPLATES: dict[str, PromptTemplate] = {
    "gsm8k": PromptTemplate(
        name="gsm8k",
        question_template="Question: {question}\nAnswer:",
        fewshot_answer_key="answer",
    ),
    # The one template that closes its exemplars with the checkpoint's end-of-text token,
    # which neither lm-eval's leaderboard_math nor the olmo-eval task does. The reason is
    # a property of what is being scored rather than of MATH: this checkpoint family
    # writes eos == pad == bos == 0 and shows no sign of having been taught to end a
    # document, so a generative item decoded the whole token budget however long ago it
    # finished. A base model has no instruction-following to appeal to, so demonstrating
    # the token is the only lever, and four exemplars are what there is to demonstrate in.
    #
    # ALL FOUR carry it, the last included, and that is the half worth arguing.
    #
    # For: the fourth exemplar abuts the live question, so it is the demonstration
    # recency weights most heavily, and ending it bare would show the model -- last of
    # all, immediately before being asked to answer -- a finished solution that is not
    # followed by the token. Uniformity also keeps a reduced shot count honest, since
    # fewshot_examples() slices this block from the front and a positional exception
    # would silently change what a 1-shot or 2-shot run teaches.
    #
    # Against, and this is the real objection: an end-of-text token followed by more
    # prompt text arguably demonstrates that the token does *not* end anything. It is
    # answered by the vocabulary rather than dismissed. SmolLM2 spells eos, bos and unk
    # with the same "<|endoftext|>" at id 0, so in pretraining that token is precisely
    # the document *separator* -- it ends one document and opens the next, with text on
    # both sides. "...correct.<|endoftext|>\n\nProblem:" is therefore the in-distribution
    # shape of a boundary for this model, not a contradiction of one. And the objection
    # applies equally to all four occurrences, since each is followed by the next
    # exemplar, so dropping the fourth pays the cost anyway and loses the demonstration
    # that adjacency makes most salient.
    "leaderboard_math": PromptTemplate(
        name="leaderboard_math",
        question_template="Problem:\n{question}\n\nSolution:",
        fewshot_answer_key="solution",
        exemplars_end_with_eos=True,
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
    #: Precision the *native* backend builds the model at, and the twin of
    #: :attr:`~diagnostics.mcq_cat.common.inference.InferenceConfig.dtype`. It sits on
    #: this config for the same reason ``checkpoint_kind`` and ``device_map`` do -- it
    #: describes how the checkpoint is loaded, not what the benchmark asks of it, and
    #: :func:`~diagnostics.mcq_cat.styles.uni_mcq.convention._generative_convention`
    #: therefore does not record it and no manifest changes shape.
    #:
    #: It exists because the MCQ half of this flag was inert for a whole run and nobody
    #: could tell: run_019fe277 reported ``bfloat16`` and scored in the checkpoint's
    #: float32, because ``--dtype`` reached ``prepare_checkpoint``, which converts
    #: nothing under ``--checkpoint-prep none``. The generative half had exactly the same
    #: gap the moment a native completer existed, so it is closed in the same commit
    #: rather than left to be discovered on a second card.
    #:
    #: Unread on the ``hf`` path, which loads with ``torch_dtype="auto"`` and takes
    #: whatever precision conversion wrote. The sentinel is the one the MCQ config and
    #: ``OlmoCoreProvider`` both use, and it means "no opinion": the kwarg is omitted
    #: rather than passed, because ``DType("auto")`` raises.
    dtype: str = inference.DTYPE_CHECKPOINT_DEFAULT
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


def fewshot_examples(
    config: GenerationConfig, num_fewshot: int | None = None
) -> tuple[dict[str, str], ...]:
    """Return the few-shot block to prepend, honouring ``config.num_fewshot``.

    ``num_fewshot`` overrides the configured count and exists for one caller,
    :class:`PromptFitter`, which walks the count down when the block does not fit the
    model's context window. It slices from the END, so a reduced count keeps the
    exemplars nearest the live question; see :meth:`PromptFitter.fit` for why.

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
    shots = config.num_fewshot if num_fewshot is None else num_fewshot
    if shots == 0:
        return ()
    try:
        loader = FEWSHOT_SOURCES[config.fewshot_source]
    except KeyError:
        raise ValueError(
            f"Unknown fewshot_source {config.fewshot_source!r}. Known sources: "
            f"{', '.join(sorted(FEWSHOT_SOURCES))}."
        ) from None
    block = loader()[: config.num_fewshot]
    return block[len(block) - shots :] if shots < len(block) else block


def format_generative_prompt(
    item: BenchmarkItem,
    config: GenerationConfig,
    *,
    eos_token: str | None = None,
    num_fewshot: int | None = None,
) -> str:
    """Build the prompt for ``item`` under the configured style, few-shot block included.

    ``eos_token`` is the loaded checkpoint's end-of-text string and defaults to ``None``,
    which renders the prompt this function rendered before the parameter existed. That
    default is load-bearing rather than convenient: most callers here have no tokenizer
    to ask, and a prompt built offline has to be the prompt built with one minus only the
    model-specific part.

    Each layout -- examples joined by a blank line, the live question last with a bare
    answer cue -- reproduces its olmo-eval task's ``format_request``. The join is also
    what the stop sequences are read against: it is the boundary the model has been
    shown between one question and the next. GSM8K stops on the blank line itself;
    MATH stops on the ``Problem:`` header that follows it, because a MATH solution may
    contain a blank line of its own and a GSM8K answer may not. See the argument on
    ``stop_sequences`` in the style's ``config.yaml``.
    """
    template = get_prompt_template(config.prompt_style)
    return template.render(
        item.question,
        fewshot_examples(config, num_fewshot),
        eos_token=exemplar_eos_token(config, eos_token),
    )


#: Where the context length is read from on a model config, most likely spelling first.
#:
#: ``max_position_embeddings`` is what ``hf_config_patch._llama_config`` emits for this
#: checkpoint family; ``n_positions`` is the GPT-2-era name, tried second so a
#: differently-shaped config still resolves instead of silently disabling the clamp.
#: Named because the absence of both is a real state -- skip the clamp and say so --
#: rather than a bug.
CONTEXT_LENGTH_ATTRS = ("max_position_embeddings", "n_positions")

#: ``ungradable_reason`` for an item the context window cannot hold a usable prompt for.
#:
#: Scored 0 like any other ungradable item, and it has to be distinguishable from a
#: grading failure, because the fix is different: this one is the checkpoint's context
#: window being too small for the bank, not a malformed row or a missing verifier.
#:
#: One stable string rather than a sentence carrying this item's figures, because
#: ``style._ungradable_block`` publishes ``reasons`` as a set: a per-item message would
#: make every refused item its own reason and turn that field into a transcript. The
#: figures are on each response's ``context_fit`` block and in the warning logged beside
#: it, so nothing is lost by keeping the reason itself groupable.
CONTEXT_OVERFLOW_REASON = "the prompt does not fit the checkpoint's context window"


@dataclass(frozen=True, slots=True)
class PromptFit:
    """One item's prompt, sized against the model's context window.

    Attributes:
        prompt: The rendered prompt, always whole exemplars and a whole question.
        num_fewshot: Exemplars actually used, which may be fewer than configured.
        prompt_tokens: Measured length, or ``None`` when nothing measured it.
        gen_budget: Tokens to generate, or ``None`` when nothing measured the window and
            the item's own ceiling is therefore the only bound.
        context_length: The window this was fitted to, or ``None`` if undeterminable.
        clamped: Whether ``gen_budget`` came out below the ceiling ``fit`` was given.
        configured_fewshot: What ``config.num_fewshot`` asked for, so a reader can see
            that a reduced ``num_fewshot`` is the ladder's doing and not the config's.
        ungradable_reason: Set when not even the floor prompt fits; see
            :data:`CONTEXT_OVERFLOW_REASON`.
    """

    prompt: str
    num_fewshot: int
    configured_fewshot: int
    prompt_tokens: int | None = None
    gen_budget: int | None = None
    context_length: int | None = None
    clamped: bool = False
    ungradable_reason: str | None = None

    @property
    def dropped_exemplars(self) -> int:
        """Exemplars the ladder gave up, for the report."""
        return max(0, self.configured_fewshot - self.num_fewshot)


@dataclass(frozen=True, slots=True)
class PromptFitter:
    """Fit each prompt into the checkpoint's context window without cutting an exemplar.

    Research's ``_fit_prompt_and_budget`` left-truncates and keeps the tail, which is
    right for a chat transcript where the latest turn is what matters. **It is destructive
    here and is deliberately not copied.** The few-shot block is what teaches the
    ``\\boxed{}`` expression and the ``Final Answer: The final answer is $X$.`` line, and
    those two forms are exactly what :class:`MathLatexEquivalence` reads;
    :func:`_leaderboard_math_fixed_fewshot` says as much. Eating the front of that block
    would leave a prompt of the right length that no longer teaches the answer format, so
    the grader would find nothing to compare, score the item 0, and every guard in the
    harness would still pass. That produces a theta rather than an error, which is the
    failure this whole change exists to remove.

    So the unit of reduction is a whole exemplar. The ladder tries the configured count,
    measures, and drops one exemplar at a time until :func:`fit_budget_to_context`
    accepts the prompt -- and stops at ONE, never zero, on a bank that asked for any. A
    0-shot MATH prompt teaches neither answer form, so an item that cannot fit even one
    exemplar is recorded :data:`CONTEXT_OVERFLOW_REASON` instead of being sent bare.
    Scoring 0 for a prompt the model was never taught to answer would be a fabricated
    zero attributed to its mathematics.

    **The ladder is layered on the clamp rather than duplicating it.** Deciding whether a
    prompt fits, and what it leaves to generate with, is one question and
    :func:`fit_budget_to_context` is the one place that answers it -- including the
    reserve that keeps a small demand from being refused for room it never wanted, and
    the refusal that replaces Research's left-truncation. What this class adds on top is
    the *reduction rule*: which prompt to offer the clamp next when it says no. A bank
    with no few-shot block has exactly one prompt to offer, so for ifeval this collapses
    to a single call and the clamp is all there is.

    Exemplars are dropped from the FRONT, keeping those nearest the live question.
    Recency dominates in-context learning, and the adjacent exemplar is doing the most
    work to establish the answer format -- which is the property that must survive, since
    losing it is the failure described above. The alternative reading, that the first
    exemplars set up the task and the last only reinforce it, would matter more if the
    four differed in kind; they do not, being four worked Level-5 solutions in one format.
    Measured with SmolLM2-135M they are 146, 118, 221 and 195 tokens closed with
    end-of-text and joined, so dropping from the front also happens to shed the cheapest
    exemplar first, which makes the ladder take more steps than dropping the largest
    would. That is accepted: the step order is chosen for what it preserves.

    Attributes:
        count_tokens: The loaded checkpoint's ``text -> token count``, used only to
            measure. The same callable :data:`TOKEN_COUNTER_ATTR` publishes, so the
            prompt is measured by the tokenizer that will generate from it.
        context_length: Its window, or ``None`` to skip clamping entirely.
        eos_token: Passed through to prompt rendering so the measured string is the
            string that will be sent.
    """

    count_tokens: Callable[[str], int]
    context_length: int | None
    eos_token: str | None = None

    def fit(
        self,
        item: BenchmarkItem,
        config: GenerationConfig,
        *,
        ceiling: int | None = None,
    ) -> PromptFit:
        """Size one item's prompt and generation budget.

        ``ceiling`` is the most this item may generate before the window is consulted:
        the bank's ``max_new_tokens`` for every bank but ifeval, and that item's own
        cascade budget for ifeval. It defaults to ``config.max_new_tokens`` so a caller
        with no per-item policy needs to know nothing about one.

        The order the ceiling and the window are applied in is the question a reader will
        ask, and it is fixed here: the ceiling is chosen first and the window can only
        lower it. That way the clamp never *grants* headroom the calibration did not
        have -- a 16k-context model still stops at the bank's cap -- and only a
        small-context one goes below it.
        """
        cap = config.max_new_tokens if ceiling is None else ceiling
        configured = config.num_fewshot
        render = lambda shots: format_generative_prompt(  # noqa: E731
            item, config, eos_token=self.eos_token, num_fewshot=shots
        )
        if self.context_length is None:
            return PromptFit(render(None), configured, configured)

        floor = 1 if configured >= 1 else 0
        for shots in range(configured, floor - 1, -1):
            prompt = render(shots)
            tokens = self._measure(prompt)
            if tokens is None:
                return PromptFit(prompt, shots, configured)
            budget = fit_budget_to_context(cap, tokens, self.context_length)
            if budget is not None:
                return PromptFit(
                    prompt,
                    shots,
                    configured,
                    prompt_tokens=tokens,
                    gen_budget=budget,
                    context_length=self.context_length,
                    clamped=budget < cap,
                )
        # Nothing was sent, so the shot count used is 0 rather than the floor that was
        # tried and refused, and every configured exemplar counts as dropped.
        return PromptFit(
            "",
            0,
            configured,
            prompt_tokens=self._measure(render(floor)),
            context_length=self.context_length,
            ungradable_reason=CONTEXT_OVERFLOW_REASON,
        )

    def _measure(self, prompt: str) -> int | None:
        """Prompt length in tokens, or ``None`` if this tokenizer cannot say."""
        try:
            return self.count_tokens(prompt)
        except Exception:  # noqa: BLE001 - a window we cannot measure is one we cannot use
            log.warning(
                "Could not tokenize a prompt to measure it against the %s-token context "
                "window, so the generation budget is left at the benchmark's cap and the "
                "prompt is sent unfitted.",
                self.context_length,
            )
            return None


def context_length_of(model: Any) -> int | None:
    """The model's context window, or ``None`` with a warning saying the clamp is off.

    Read from the loaded model's own config rather than configured, because it is a fact
    about the weights: a benchmark cannot know it and a hardcoded value would be wrong
    for the next checkpoint. ``None`` rather than a fallback guess, for the reason
    :func:`fit_budget_to_context` gives -- a wrong window is worse than no window,
    because it would either refuse items that would have generated fine or cap budgets
    for no reason.
    """
    config = getattr(model, "config", None)
    for attr in CONTEXT_LENGTH_ATTRS:
        length = getattr(config, attr, None)
        if isinstance(length, int) and not isinstance(length, bool) and length > 0:
            return length
    log.warning(
        "This checkpoint's config declares none of %s, so the context clamp is disabled "
        "for this run: the generation budget stays at the benchmark's cap and a "
        "prompt longer than the window will be truncated by the model rather than "
        "fitted by dropping whole exemplars. Every item is still bounded by "
        "max_new_tokens.",
        ", ".join(CONTEXT_LENGTH_ATTRS),
    )
    return None


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


def eos_stop_sequences(
    item: BenchmarkItem, config: GenerationConfig, eos_text: str | None
) -> tuple[str, ...]:
    """``config.stop_sequences`` plus the checkpoint's own end token, where that is safe.

    The one runtime stop list, for every generative bank. A configured list -- ifeval's
    empty one, MATH's ``["Problem:", "problem:"]`` -- is what the benchmark names, is what
    the manifest records and is what ``convention.check_runtime_convention`` compares
    against. The end token appended here is the *checkpoint's*, resolved from the live
    tokenizer, so it belongs to the list the running scorer uses and is written down
    nowhere: a literal in ``config.yaml`` would be one model's answer recorded as the
    benchmark's, and ordinary text to every other model. With no tokenizer or no EOS
    defined the list is returned unchanged.

    **What this buys, and what it does not.** The two mechanisms are not symmetric and it
    matters which is which. :func:`ifeval_token_budget` and ``generate(eos_token_id=...)``
    are the compute bounds: they decide how many tokens are actually generated. This buys
    no compute at all -- :func:`truncate_at_stop` runs over an already-finished completion,
    so every token it removes has been paid for -- and exists for one narrow correctness
    reason. A model can emit the literal characters of its end marker as *ordinary
    vocabulary tokens* rather than as the special token; those are not special, so
    ``skip_special_tokens=True`` in :meth:`_HFCompleter.__call__` does not remove them, and
    they land inside the graded span. Nor can this entry ever fire on an end token the
    model genuinely emits, for the same reason: the decode deletes it first.

    The leak is invited by two independent things, and the entry is added for both. On
    MATH the exemplars are closed with the token, so a model shown it four times can learn
    to write its characters; there the damage is *inflation* rather than corruption, since
    :class:`MathLatexEquivalence` scores ``any(is_equiv(candidate, gold))`` over every
    candidate the extractor finds and a run-on can only add candidates -- turning wrong
    answers right and never right answers wrong, which reads as ability the checkpoint
    does not have on a scale where theta is meant to be comparable.

    On ifeval nothing demonstrates the token and the damage runs the other way, because 93
    of the 511 items carry a constraint anchored at the *end* of the response and every one
    of them is decided by what the last characters are. ``startend:end_checker`` (26 items)
    does ``value.strip().strip('"').lower().endswith(phrase)``; ``startend:quotation`` (40)
    requires ``value.strip()`` to begin and end with ``"``; ``detectable_format:json_format``
    (17) and ``detectable_format:constrained_response`` (10) both parse the whole span. A
    trailing ``<|endoftext|>`` fails all four on a response that satisfied them, and
    cutting there restores the ending -- ``end_checker`` and ``quotation`` strip
    whitespace, so the cut leaves nothing behind that they object to.

    **Why it is conditional.** It is suppressed for the 24 items declaring
    :data:`TWO_RESPONSES_ID`, and those are the reason this is not a blanket stop.
    ``TwoResponsesChecker`` splits the response on ``******`` and requires exactly two
    non-empty parts. A model that leaks its end marker *between* the two responses -- the
    natural thing for a chat-tuned checkpoint treating each as a turn -- would have the
    second deleted by a cut at the first occurrence, leaving one part and failing the
    constraint. Suppressing costs nothing measurable: the untruncated text passes
    ``TwoResponsesChecker`` with the leaked literal sitting harmlessly inside one of the
    two parts, and **0 of those 24 items carry any of the four end-anchored constraints**
    above, so nothing is given up by not cutting them.

    Cutting at the last occurrence instead of the first was considered and rejected: it
    fixes the two-responses case only when the model leaks twice, and it weakens the
    end-anchored case, which is the one this exists for.

    The residual, stated plainly: for an item that is *not* two-responses, a leak strictly
    before content that would have satisfied a constraint loses that content. That is
    accepted because the ordering it needs -- end marker, then more prose, then a correct
    ending -- is the rarer pattern, while a marker emitted after the answer and followed
    by a hallucinated next turn is the common one, and is exactly what a first-occurrence
    cut is right for.
    """
    if not eos_text:
        return tuple(config.stop_sequences)
    if TWO_RESPONSES_ID in set(item.metadata.get("instruction_id_list") or ()):
        return tuple(config.stop_sequences)
    if eos_text in config.stop_sequences:
        return tuple(config.stop_sequences)
    return (*config.stop_sequences, eos_text)


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
    item: BenchmarkItem,
    completion: str,
    config: GenerationConfig,
    eos_text: str | None = None,
    *,
    ungradable_reason: str | None = None,
    budget: Mapping[str, Any] | None = None,
    fit: PromptFit | None = None,
) -> ItemResponse:
    """Turn one raw completion into a graded :class:`ItemResponse`.

    Pure text in, response out, with no model involved, which is what makes the
    grading half of this module testable without a GPU.

    ``eos_text`` is the running checkpoint's end-token text, when the completer published
    one, and reaches the graded span only through :func:`eos_stop_sequences` -- see there
    for what that does and, more importantly, what it does not. It is optional so that
    every existing caller -- and the whole of the offline grading path -- keeps working
    with no end token at all, which is also the state a run is in when the tokenizer
    defines none. It is the raw string rather than the exemplar suffix
    :func:`exemplar_eos_token` derives from it: whether a bank closes its exemplars with
    the token decides what the *prompt* says, and a leaked literal has to be cut out of
    the graded span either way.

    ``fit`` is the :class:`PromptFitter` outcome for this item, when one ran. It is what
    makes ``num_fewshot`` in the response the count actually used rather than the count
    configured, and it adds a ``context_fit`` block whenever the two differ, the budget
    was clamped, or the item was refused. Those facts have to be per item, not per
    session: a window that forces 2 exemplars on a long stem and allows 4 on a short one
    mixes prompts inside one ability estimate, and a single session-level field would
    average that away.

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

    ``ungradable_reason`` is for an item that was never sent to the model, which today means
    one the context clamp refused -- see :data:`CONTEXT_OVERFLOW_REASON`. The grader is then
    skipped rather than run over an empty string: several of the IFEval verifiers *pass* on
    empty input -- ``forbidden_words`` and ``no_comma`` are satisfied by having no text at
    all -- so grading a non-attempt would invent a partial score for it. It routes through
    this function anyway so the response carries the same metadata shape as every other,
    which is what lets the report's ``ungradable`` block list it beside the other kind.
    """
    grader = get_answer_grader(str(item.metadata.get("answer_type", DEFAULT_ANSWER_TYPE)))
    text = truncate_at_stop(completion, eos_stop_sequences(item, config, eos_text))
    if ungradable_reason is not None:
        verdict = Verdict(
            correct=False, extracted=None, gold=None, ungradable_reason=ungradable_reason
        )
    else:
        verdict = apply_grader(grader, item, text)
    metadata: dict[str, Any] = {
        "modality": "generative",
        "grader": grader.name,
        "completion": text,
        "extracted_answer": verdict.extracted,
        "gold_answer": verdict.gold,
        "num_fewshot": config.num_fewshot if fit is None else fit.num_fewshot,
        UNGRADABLE_KEY: verdict.ungradable_reason is not None,
    }
    if verdict.ungradable_reason is not None:
        metadata[UNGRADABLE_REASON_KEY] = verdict.ungradable_reason
    # Omitted when the clamp did nothing, so nine banks' reports keep their shape, and
    # written whenever it did something -- dropped an exemplar, lowered the budget, or
    # refused the item outright, which is the case whose figures the stable
    # CONTEXT_OVERFLOW_REASON deliberately does not carry.
    if fit is not None and (fit.dropped_exemplars or fit.clamped or fit.ungradable_reason):
        metadata["context_fit"] = {
            "context_length": fit.context_length,
            "prompt_tokens": fit.prompt_tokens,
            "gen_budget": fit.gen_budget,
            "max_new_tokens_configured": config.max_new_tokens,
            "num_fewshot_configured": fit.configured_fewshot,
            "exemplars_dropped": fit.dropped_exemplars,
        }
    if verdict.detail:
        metadata["grader_detail"] = dict(verdict.detail)
    # Omitted rather than written null for a caller that has no budget to report -- the
    # offline grading path, and every MCQ-shaped test lambda -- so a response's having the
    # key means a real budget decision was made for it.
    if budget is not None:
        metadata[BUDGET_KEY] = dict(budget)
    return ItemResponse(
        item_id=item.item_id,
        chosen_index=NO_CHOICE_INDEX,
        correct=verdict.correct,
        choice_logprobs=(),
        metadata=metadata,
    )


#: Attribute a completer sets to declare that it honours a per-item token budget.
#:
#: The completer protocol is ``Callable[[str], str]`` and stays that way. A per-item
#: budget needs a second argument, and the naive extension -- give
#: :meth:`_HFCompleter.__call__` an optional second parameter and always pass it -- does
#: not work, because the incompatibility is on the *caller's* side: a one-argument
#: callable is the norm here, not a legacy case. ``GENERATIVE_BACKENDS`` registers one
#: completer, and every other completer in the tree is a one-argument function
#: (``lambda _: ""`` in test_generative_grading, test_grading_dispatch,
#: test_symbolic_grading and test_runtime_guards), which a two-argument call breaks with
#: a ``TypeError`` raised from inside the scorer.
#:
#: So the extension is opt-in and explicit. A completer that can take a budget says so
#: with this attribute; :class:`GenerativeScorer` reads it once at construction and
#: calls the one-argument form otherwise. Signature introspection and a ``TypeError``
#: retry were both rejected: the first is implicit, and the second would swallow a
#: genuine ``TypeError`` from inside a model backend and silently re-run generation.
BUDGET_AWARE_ATTR = "accepts_token_budget"


class GenerativeScorer:
    """A :class:`ScoringModel` that samples one completion per item and grades it.

    Sampling is injected as a plain ``prompt -> completion`` callable. That split is
    what lets the grading path be exercised offline, and it makes a different backend
    (vLLM, a hosted endpoint) a new completer rather than a new scorer.

    A completer that declares :data:`BUDGET_AWARE_ATTR` is instead called
    ``(prompt, max_new_tokens)`` with the budget derived for that item. The item is in
    scope here and nowhere below, which is why the budget is computed at this level and
    passed down rather than looked up by the completer: the completer's job stays "turn
    this prompt into that much text", with no benchmark metadata in it.

    Everything checkpoint-specific flows the other way and is published by the completer
    rather than reached for here, because the tokenizer is the backend's and a scorer that
    imported one would make a second backend a change to the scorer. None of it belongs in
    :class:`GenerationConfig`, which is the benchmark's description of itself and is what
    ``convention.runtime_convention`` reads and the manifest guard compares against.

    - :data:`EOS_TEXT_ATTR` is the text of the checkpoint's end token, kept as
      ``eos_text`` and handed to grading, where :func:`eos_stop_sequences` decides per
      item whether to cut on it. ``eos_token`` is the same string once
      :func:`exemplar_eos_token` has decided whether *this bank* closes its exemplars
      with it, which is a different question with a different answer per bank, and is
      what reaches prompt rendering.
    - :data:`TOKEN_COUNTER_ATTR` is its tokenizer's token count, which does two jobs: it
      converts declared word counts for :func:`item_token_budget`, and it measures
      rendered prompts for the :class:`PromptFitter` this builds when the completer also
      publishes :data:`CONTEXT_WINDOW_ATTR`. Without it neither runs.

    The three constructor arguments are for what a completer cannot publish about itself.
    ``eos_token`` overrides the published text with one the backend has verified round
    trips (see :func:`resolve_eos_token`), because teaching a spelling the tokenizer reads
    as ordinary characters is worse than teaching nothing. ``fitter`` replaces the one
    built here, for a caller that wants a window the completer does not publish.
    ``checkpoint_facts`` is what :meth:`runtime_facts` reports and nothing branches on.
    All three default to the degraded state, so the offline tests -- which pass a bare
    lambda -- score exactly as this class scored before any of it existed.
    """

    def __init__(
        self,
        complete: Callable[..., str],
        config: GenerationConfig,
        *,
        eos_token: str | None = None,
        fitter: PromptFitter | None = None,
        checkpoint_facts: Mapping[str, Any] | None = None,
    ) -> None:
        self.complete = complete
        self.config = config
        self.checkpoint_facts = dict(checkpoint_facts or {})
        self._budget_aware = bool(getattr(complete, BUDGET_AWARE_ATTR, False))
        published_eos = getattr(complete, EOS_TEXT_ATTR, None)
        self.eos_text = str(published_eos) if published_eos else None
        self.eos_token = exemplar_eos_token(config, eos_token or self.eos_text)
        counter = getattr(complete, TOKEN_COUNTER_ATTR, None)
        self._count_tokens: Callable[[str], int] | None = (
            counter if callable(counter) else None
        )
        tokenizer_id = getattr(complete, TOKENIZER_ID_ATTR, None)
        self._tokenizer_id = str(tokenizer_id) if tokenizer_id else None
        context = getattr(complete, CONTEXT_WINDOW_ATTR, None)
        self._context_window = (
            int(context) if isinstance(context, int) and context > 0 else None
        )
        if self._context_window is None and self._count_tokens is not None:
            log.info(
                "The completer publishes no usable %s, so per-item budgets will not be "
                "clamped to a context window. Budgets are still bounded by "
                "max_new_tokens=%d; this only means the harness cannot tell whether a "
                "prompt plus its continuation fits the checkpoint's window.",
                CONTEXT_WINDOW_ATTR,
                config.max_new_tokens,
            )
        self.fitter = fitter if fitter is not None else self._published_fitter()
        self._fits: list[PromptFit] = []
        # Running totals over every item text this scorer has measured, for the reason
        # given in _tokens_per_word.
        self._corpus_tokens = 0
        self._corpus_words = 0

    def _published_fitter(self) -> PromptFitter | None:
        """A fitter over what the completer published, or ``None`` if it published no counter.

        Built here rather than by each backend so there is one place that turns a
        completer's published tokenizer and window into a fitted prompt. A completer that
        publishes a counter but no window still gets one: the ladder is then inert and the
        fitter only renders, which keeps a single code path through :meth:`_fit` instead of
        a second unfitted one.
        """
        if self._count_tokens is None:
            return None
        return PromptFitter(
            count_tokens=self._count_tokens,
            context_length=self._context_window,
            eos_token=self.eos_token,
        )

    def _tokens_per_word(self, text: str) -> float | None:
        """Words-to-tokens for the live tokenizer, or ``None`` if there is not one.

        Measured on item text and accumulated across the session, then used as
        ``max(this item's ratio, the running aggregate)``. Both halves of that are load
        bearing and the naive version is a truncation bug:

        * **Not the single item's ratio alone.** A prompt is a small sample and a short one
          is a bad estimator. Measured over this bank, per-item prompt ratios run 1.08 to
          2.10 against a corpus aggregate of 1.285 on SmolLM2, and taking each item's own
          ratio under-budgets 77 of them relative to the aggregate, by up to 185 tokens.
          The worst, ef9aa240cb4c0265, declares 750 words and would get 860 tokens where
          the corpus rate asks 1014 -- a compliant answer cut at about 670 words and graded
          as failing a length constraint it met. The same check across seven tokenizers put
          the count between 64 and 97 items, so it is not a quirk of one.
        * **Not the aggregate alone.** ``run_cat`` administers one item per call, so the
          corpus is a single prompt on the first item and grows from there. Taking the
          maximum means an item whose own text is unusually dense is budgeted at its own
          rate rather than at a session average that has not seen it yet, and the estimate
          only improves as the session runs.

        The maximum is the conservative direction on both counts, which is the whole
        criterion here: too small is a wrong grade, too large is decode time, and the
        1280 ceiling bounds how much decode time being wrong can cost.

        The residual, which cannot be measured away: this is fitted to prompt text, and
        IFEval ships no reference answer to fit to -- so it assumes a response tokenizes at
        about the rate a request does. :data:`DETECTION_HEADROOM_TOKENS` absorbs a little of
        that, and the ceiling bounds the rest.
        """
        if self._count_tokens is None:
            return None
        words = count_words(text)
        if not words:
            return None
        tokens = self._count_tokens(text)
        self._corpus_tokens += tokens
        self._corpus_words += words
        return max(tokens / words, self._corpus_tokens / self._corpus_words)

    def score_items(self, items: Sequence[BenchmarkItem]) -> list[ItemResponse]:
        """Grade each item by sampling a greedy completion and extracting its answer.

        Three stages per item, each of which can only lower what the one before settled
        on, so which of them wins is simply whichever is smallest:

        1. :func:`item_token_budget` derives the item's own ceiling from what it declares,
           for the one bank that declares anything, and hands every other item the bank's
           flat ``max_new_tokens``.
        2. :meth:`PromptFitter.fit` renders the prompt against that ceiling, dropping whole
           exemplars while the context window will not hold it, and lowers the budget to
           what the window leaves.
        3. If nothing fits, the item is refused rather than sent.
        """
        responses: list[ItemResponse] = []
        for item in items:
            if item.choices:
                raise ValueError(
                    f"Item {item.item_id!r} has {len(item.choices)} answer choices, so "
                    f"it is an MCQ item reaching the generative grader. Sampling free "
                    f"text for it would ignore the choice set and grade against a gold "
                    f"answer it does not have. Grade it with common.inference instead."
                )
            # Measured on item.question rather than on the rendered prompt: the ratio
            # wanted is the one for prose, and a rendered prompt carries fewshot
            # exemplars and chat scaffolding that tokenize at their own rate.
            ceiling = item_token_budget(
                item,
                self.config,
                tokens_per_word=self._tokens_per_word(item.question),
                count_tokens=self._count_tokens,
            )
            fit = self._fit(item, ceiling)
            self._fits.append(fit)
            if fit.ungradable_reason is not None:
                responses.append(self._refuse_for_context(item, fit, ceiling))
                continue
            budget = ceiling if fit.gen_budget is None else fit.gen_budget
            # Provenance, not value. An item the cascade deliberately left at the ceiling --
            # a fragile or non-English one -- is still a computed budget, and labelling it
            # "flat" would make it indistinguishable from the degraded mode, which is the
            # one distinction this record exists to draw. How many items ended up at the
            # ceiling is recoverable from the recorded ceiling and the token counts.
            source = BUDGET_FROM_CASCADE if self.cascade_active else BUDGET_FROM_CEILING
            if fit.clamped:
                source = BUDGET_FROM_CONTEXT
            responses.append(
                grade_completion(
                    item,
                    self._complete(fit.prompt, budget),
                    self.config,
                    self.eos_text,
                    fit=fit,
                    budget=self._budget_record(budget, source),
                )
            )
        return responses

    def _fit(self, item: BenchmarkItem, ceiling: int) -> PromptFit:
        """This item's prompt and budget, fitted to the context window if one is known."""
        if self.fitter is None:
            return PromptFit(
                format_generative_prompt(item, self.config, eos_token=self.eos_token),
                self.config.num_fewshot,
                self.config.num_fewshot,
            )
        return self.fitter.fit(item, self.config, ceiling=ceiling)

    def _complete(self, prompt: str, budget: int) -> str:
        """Sample, passing the budget only to a completer that declared it takes one.

        Gated on :data:`BUDGET_AWARE_ATTR` rather than on whether a budget was derived,
        because a budget is always derived now and the many injected completers that are a
        plain ``lambda prompt: ...`` would break on a second argument.
        """
        if not self._budget_aware:
            return self.complete(prompt)
        return self.complete(prompt, budget)

    def _refuse_for_context(
        self, item: BenchmarkItem, fit: PromptFit, ceiling: int
    ) -> ItemResponse:
        """Record an item the window cannot hold, without sampling it.

        Nothing is generated. Left-truncating the prompt to make it fit is what Research's
        pipeline does and is silently destructive for both banks here: on ifeval the
        instruction being graded *is* the prompt, and the verifiers read
        ``metadata['kwargs']`` rather than the prompt, so cutting its front does not soften
        what is checked -- it grades a model on a constraint it was never shown. On MATH
        the front of the prompt is the exemplar block that teaches the two answer forms the
        grader reads, so cutting into it leaves a correctly-sized prompt the grader can
        find nothing in.

        Either way the item is scored 0 and marked ungradable, which is what
        ``style._ungradable_block`` counts and reports with its ids and reasons -- so the
        cost lands in the report as a fabricated zero rather than in theta as the
        checkpoint having failed. It routes through :func:`grade_completion` so the
        response carries the same metadata shape as every other one, and the grader is
        skipped rather than run over the empty string.
        """
        log.warning(
            "Item %s was not sent: its prompt is %s tokens against a %s-token context "
            "window, which leaves too little room to generate an answer worth grading at "
            "the %d tokens this item was budgeted%s. Truncating the prompt to fit would "
            "have deleted part of what the grader depends on, so the item was scored 0 and "
            "marked ungradable instead. Score a checkpoint with a larger context window.",
            item.item_id,
            fit.prompt_tokens,
            fit.context_length,
            ceiling,
            (
                " -- and that is with the exemplar ladder already down to its one-shot "
                "floor, below which the prompt would teach the grader's answer format to "
                "nobody"
                if fit.configured_fewshot
                else ""
            ),
        )
        return grade_completion(
            item,
            "",
            self.config,
            self.eos_text,
            ungradable_reason=fit.ungradable_reason,
            fit=fit,
            budget=self._budget_record(0, BUDGET_FROM_CONTEXT),
        )

    def runtime_facts(self) -> dict[str, Any]:
        """What this run actually did about end-of-text and the context window.

        Reported rather than only logged, because the artifact is what gets compared across
        checkpoints and the difference recorded here is large: with a working
        ``eos_token_id`` a completion stops when the answer is done, and without one every
        item burns its full budget at the cacheless rate ``TORCH_TODOS.md`` measured. A
        reader has to be able to tell which run they are holding without the log.

        Session-level facts only. The per-item ones -- shots used, budget granted -- are on
        each response's ``context_fit`` and ``num_fewshot``, because on a small window they
        differ item by item, and a session-level summary would hide exactly the mixing that
        makes theta uninterpretable.
        """
        shots = sorted({fit.num_fewshot for fit in self._fits})
        clamp_on = self.fitter is not None and self.fitter.context_length is not None
        facts: dict[str, Any] = {
            **self.checkpoint_facts,
            "exemplars_end_with_eos": self.eos_token is not None,
            "context_clamp_active": clamp_on,
            "context_length": None if self.fitter is None else self.fitter.context_length,
            "num_fewshot_configured": self.config.num_fewshot,
            "num_fewshot_used": shots,
            "context_clamp_fired": any(f.clamped or f.dropped_exemplars for f in self._fits),
            "items_over_context": sum(1 for f in self._fits if f.ungradable_reason is not None),
        }
        if len(shots) > 1:
            facts["mixed_shot_alert"] = (
                f"Items in this session were prompted at {shots} exemplars, because the "
                f"{facts['context_length']}-token context window could not hold "
                f"{self.config.num_fewshot} for every stem. Every difficulty in this bank "
                f"was estimated behind a {self.config.num_fewshot}-shot prompt, so this "
                f"theta mixes scales inside one estimate and is NOT comparable with a run "
                f"that stayed at {self.config.num_fewshot}. Per-item counts are in each "
                f"response's num_fewshot."
            )
        return facts

    @property
    def cascade_active(self) -> bool:
        """Whether a live tokenizer was resolved, so per-item budgets are computed.

        ``False`` means every item takes the flat ``max_new_tokens``. That is a legitimate
        mode to exercise offline and an illegitimate one to reach in a real run, which is
        why :func:`load_generative_model` refuses it at the point a checkpoint is loaded
        rather than here. Keeping the refusal there and the capability flag here is what
        lets the tests drive the degraded path deliberately without an ``is_testing`` flag.
        """
        return self._count_tokens is not None

    def _probe_token_counter(self) -> int:
        """Call the published token counter once, so a broken one fails at load time.

        Used by :func:`require_live_tokenizer`. The probe text is ordinary prose because
        that is what the counter is asked for in earnest; a counter that cannot handle it is
        not one this can budget with.
        """
        assert self._count_tokens is not None
        return self._count_tokens("a probe of ordinary English prose")

    def _budget_record(self, budget: int, source: str) -> dict[str, Any]:
        """The per-item budget provenance recorded under :data:`BUDGET_KEY`."""
        return {
            "tokens": budget,
            "source": source,
            "cascade_active": self.cascade_active,
            "ceiling": self.config.max_new_tokens,
            "context_window": self._context_window,
            "tokenizer": self._tokenizer_id,
        }


def require_chat_template(tokenizer: Any, checkpoint_dir: Path, config: GenerationConfig) -> None:
    """Refuse a chat-format bank on a checkpoint whose tokenizer has no chat template.

    A completer's guard rather than the scorer's, because only a backend holds a
    tokenizer -- but it is the *same* guard for every backend, since what it refuses is a
    property of the loaded tokenizer and of the bank, and neither of those is
    backend-specific. Shared rather than copied for the reason the budget machinery is
    shared: a second copy is a second thing to keep true, and the one that rots is the one
    on the newer path, which is also the one nobody has run yet.
    """
    if not config.chat_format or getattr(tokenizer, "chat_template", None):
        return
    raise ValueError(
        f"{checkpoint_dir} defines no chat template, and this bank is scored "
        f"in chat format. The bank was calibrated on instruction-tuned models "
        f"answering a single user turn; sending the prompt raw instead would "
        f"still produce a completion and still be graded, and the whole gap "
        f"between a base continuation and an assistant reply would land in "
        f"theta with a healthy standard error beside it. Score a chat "
        f"checkpoint, or run a completion-format bank."
    )


def render_chat_prompt(prompt: str, tokenizer: Any, config: GenerationConfig) -> str:
    """Wrap ``prompt`` in the checkpoint's chat template when the bank asks for chat.

    The chat turn belongs to the completer rather than to :class:`PromptTemplate`
    because its text is the checkpoint's, not the benchmark's: two instruction-tuned
    models spell the same user turn with different special tokens, and the prompt a
    benchmark defines is the content inside it. A system turn is the same kind of thing
    one level up -- a standing instruction about how to answer rather than content -- so
    it is named in the config and its text loaded from olmo-eval here.

    Free rather than a method for the reason :func:`require_chat_template` is: the
    template is the tokenizer's and the decision is the bank's, so nothing in it belongs
    to one backend. Both callers pass an ordinary ``transformers`` tokenizer, the native
    path included -- ``olmo_core`` resolves its tokenizer through ``AutoTokenizer`` too.
    """
    if not config.chat_format:
        return prompt
    messages: list[dict[str, str]] = []
    if config.system_prompt_source is not None:
        messages.append(
            {"role": "system", "content": get_system_prompt(config.system_prompt_source)}
        )
    messages.append({"role": "user", "content": prompt})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


class _HFCompleter:
    """Greedy ``transformers`` generation, one prompt at a time."""

    #: See :data:`BUDGET_AWARE_ATTR`. Declared on the class so the scorer can read it off
    #: the instance without constructing anything.
    accepts_token_budget = True

    def __init__(self, checkpoint_dir: Path, config: GenerationConfig) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

        self._torch = torch
        self.config = config
        set_seed(config.seed)

        self.tokenizer: Any = self._load_tokenizer(AutoTokenizer, checkpoint_dir)
        require_chat_template(self.tokenizer, checkpoint_dir, config)
        self.model: Any = AutoModelForCausalLM.from_pretrained(
            str(checkpoint_dir),
            torch_dtype="auto",
            device_map=config.device_map,
        )
        self.model.eval()
        self.eos_token, self.eos_token_id = resolve_eos_token(self.tokenizer)
        self._require_end_of_text(checkpoint_dir)
        self._warn_if_tokenizer_outgrows_model()
        self.context_length = context_length_of(self.model)

    @staticmethod
    def _load_tokenizer(auto_tokenizer: Any, checkpoint_dir: Path) -> Any:
        """Load the checkpoint's tokenizer, or refuse the run explaining what that costs.

        A real evaluation always has a tokenizer -- inference is impossible without one --
        so a missing one is never a deployment mode to accommodate. It is a failure, and
        the likely one is named in the message: this conversion writes no tokenizer files
        of its own, resolving the identifier from the checkpoint's
        ``dataset.tokenizer.identifier`` instead, so a venue that can reach S3 but not
        huggingface.co produces exactly this. ``HF_CONVERSION.md`` flags that risk.

        Refusing rather than proceeding, because proceeding is the silent-wrong-answer
        shape: no end-of-text in the exemplars, no ``eos_token_id`` on ``generate``, no
        context clamp, every item burning its whole budget, and a report that looks
        entirely normal.

        One of three refusals about the tokenizer, and they are layers rather than copies
        of each other -- each catches a fault the other two cannot see. This one is *the
        tokenizer would not load*, and only this backend can raise it. Beside it,
        :meth:`_require_end_of_text` is *it loaded and declares no end-of-text id*, which
        is a different checkpoint defect and produces a usable tokenizer.
        :func:`require_live_tokenizer` is *the scorer did not end up with a working token
        counter*, which on this path is unreachable through either of the first two and
        exists to catch the plumbing between a backend and the scorer -- a fault no
        completer can detect about itself.
        """
        try:
            return auto_tokenizer.from_pretrained(str(checkpoint_dir))
        except Exception as exc:
            raise RuntimeError(
                f"No usable tokenizer could be loaded from {checkpoint_dir} ({exc}). "
                f"This is refused rather than worked around, because a generative bank "
                f"scored without a tokenizer would still produce a plausible report: the "
                f"few-shot exemplars would carry no end-of-text token, generate would be "
                f"given no eos_token_id and so would never halt, every item would burn "
                f"the full max_new_tokens, and a prompt longer than the context window "
                f"would be truncated into the exemplar block -- destroying the answer "
                f"format the grader reads -- with nothing in the report to say so. If the "
                f"checkpoint ships no tokenizer files, its config names one by identifier "
                f"and this venue must be able to fetch it; check network access to the "
                f"Hub, or stage the tokenizer beside the weights."
            ) from exc

    def _require_end_of_text(self, checkpoint_dir: Path) -> None:
        """Refuse a tokenizer that defines no end-of-text id, for the same reason.

        Without an id there is nothing for ``generate`` to halt on, so every item decodes
        its whole budget however long ago the answer finished. That is a cost failure and a
        correctness one -- the run-on text is graded -- and it is invisible in the report.
        A tokenizer whose id is present but whose *string* does not round-trip is a
        different, milder case: generation still halts, only the exemplars are left
        unclosed, and :func:`resolve_eos_token` warns and continues.
        """
        if self.eos_token_id is not None:
            return
        raise RuntimeError(
            f"The tokenizer at {checkpoint_dir} defines no eos_token_id, so generate has "
            f"nothing to halt on: every item would decode the full "
            f"{self.config.max_new_tokens} new tokens however early its answer finished, "
            f"the run-on text would be graded, and the report would look normal. This is "
            f"refused rather than accepted. Stage a tokenizer that declares its "
            f"end-of-text token, or fix the identifier the checkpoint config names."
        )

    def _warn_if_tokenizer_outgrows_model(self) -> None:
        """Say so if the tokenizer has more tokens than the model has embeddings.

        Cheap, and it catches the one mistake this area invites. The conversion writes no
        tokenizer files of its own, so the identifier is resolved from the checkpoint's
        ``dataset.tokenizer.identifier`` -- and a wrong resolution produces a plausible
        tokenizer rather than an error. Reading ``_resolve_tokenizer_id``'s fallback
        branch as its behaviour, this checkpoint was taken for an ``allenai/dolma2``
        model, 100,278 tokens against its actual 49,152, which this comparison would have
        contradicted immediately.

        Only the one direction is checked, because only it is unambiguous: a tokenizer
        that can emit ids the model has no row for is broken, whereas a model larger than
        its tokenizer is the ordinary result of padding the embedding to a multiple of
        128. A warning and not an error -- ``generate`` will fail loudly on its own if
        this matters, and a run that would otherwise score fine should not be stopped by
        a heuristic.
        """
        try:
            tokenizer_size = len(self.tokenizer)
            model_size = int(self.model.config.vocab_size)
        except Exception:  # noqa: BLE001 - a probe that cannot be made is not a finding
            return
        if tokenizer_size > model_size:
            log.warning(
                "This checkpoint's tokenizer has %d tokens but the model has embeddings "
                "for %d, so the tokenizer can produce ids the model cannot represent. "
                "The usual cause is a wrong tokenizer identifier in the checkpoint "
                "config rather than a corrupt checkpoint; the prompts and the scores "
                "that come out of this run are both suspect.",
                tokenizer_size,
                model_size,
            )

    def _render(self, prompt: str) -> str:
        """This checkpoint's chat framing for ``prompt``. See :func:`render_chat_prompt`."""
        return render_chat_prompt(prompt, self.tokenizer, self.config)

    def __call__(self, prompt: str, max_new_tokens: int | None = None) -> str:
        """Return the continuation of ``prompt``, with the prompt echo removed.

        ``max_new_tokens`` is the per-item budget :class:`GenerativeScorer` derives -- the
        item's own cascade ceiling, lowered to whatever room :class:`PromptFitter` found
        the prompt left in the window. It defaults to the configured flat value so this
        completer is still usable as a bare ``prompt -> completion`` callable.

        ``do_sample=False`` is how ``transformers`` spells temperature 0; passing
        ``temperature=0`` to ``generate`` is rejected by the library.

        ``skip_special_tokens=True`` on the decode is why a genuine end-of-text token can
        never be matched by a ``stop_sequences`` entry: it is removed from the text before
        :func:`truncate_at_stop` sees any of it. The completion is bounded by
        ``max_new_tokens`` and by ``**self._eos_kwargs()``, in that order of reliability.
        """
        torch = self._torch
        input_ids = self.tokenizer(self._render(prompt), return_tensors="pt")["input_ids"]
        if self.config.max_length is not None and input_ids.shape[1] > self.config.max_length:
            input_ids = input_ids[:, -self.config.max_length :]
        input_ids = input_ids.to(self.model.device)

        with torch.no_grad():
            generated = self.model.generate(
                input_ids,
                max_new_tokens=(
                    self.config.max_new_tokens if max_new_tokens is None else max_new_tokens
                ),
                do_sample=False,
                pad_token_id=self._pad_token_id(),
                **self._eos_kwargs(),
            )
        return self.tokenizer.decode(generated[0][input_ids.shape[1] :], skip_special_tokens=True)

    def _eos_kwargs(self) -> dict[str, int]:
        """``{"eos_token_id": ...}`` from the tokenizer, or nothing.

        The single change that makes a generative item able to stop early at all, and it
        applies to every bank rather than to MATH -- ``gsm8k``, ``ifeval`` and ``gpqa``
        each decode their whole budget without it.

        ``generate`` halts on ``eos_token_id`` and reads that from the model's
        ``GenerationConfig``, which ``PreTrainedModel.__init__`` builds from
        ``generation_config.json`` if one exists and from ``config.json`` otherwise. Our
        conversion writes neither value: ``hf_config_patch._llama_config`` emits
        ``eos_token_id=None`` because ``olmo_core.save_hf_model`` fills the ids only when
        handed a tokenizer object, which the call site does not do, and it writes no
        ``generation_config.json`` at all. The ids reach the output directory in
        ``tokenizer_config.json``, which ``generate`` does not read -- verified against
        ``transformers.generation.configuration_utils``, where the only filename is
        ``generation_config.json``. So the id has to be passed explicitly or generation has
        nothing to halt on, whatever the checkpoint declares about itself.

        ``is None`` rather than a truth test, which is the trap here. This family's
        end-of-text id is 0, ``bool(0)`` is ``False``, and ``eos_token_id or None``
        discards precisely the checkpoint this exists for.
        """
        if self.eos_token_id is None:
            return {}
        return {"eos_token_id": self.eos_token_id}

    def _pad_token_id(self) -> int | None:
        """Fall back to the EOS token when the tokenizer defines no pad token.

        Base checkpoints routinely have no pad token, and ``generate`` warns and pads
        with EOS anyway; naming it keeps the log clean.
        """
        pad = getattr(self.tokenizer, "pad_token_id", None)
        return pad if pad is not None else getattr(self.tokenizer, "eos_token_id", None)

    @property
    def eos_text(self) -> str | None:
        """The literal text of this checkpoint's end token, or ``None`` if it has none.

        See :data:`EOS_TEXT_ATTR` for what reads this and :func:`eos_stop_sequences` for
        what it is used for. Deliberately the tokenizer's label as written, and NOT the
        round-tripped :attr:`eos_token` that :func:`resolve_eos_token` verifies: the two
        answer different questions. Closing an exemplar with a string the tokenizer reads
        as ordinary characters would teach the model to type them, so that side demands the
        verified spelling. Cutting a leaked literal out of the graded span wants the label
        exactly because the model typed it as ordinary characters, so an unverified
        spelling is the case that most needs cutting.
        """
        token = getattr(self.tokenizer, "eos_token", None)
        return str(token) if token else None

    def count_tokens(self, text: str) -> int:
        """Tokens ``text`` costs in this checkpoint's tokenizer.

        See :data:`TOKEN_COUNTER_ATTR`. ``add_special_tokens=False`` because every caller
        is measuring the cost of *content* -- a words-to-tokens rate, the length of a
        prompt an item obliges the model to echo, and the length of a rendered prompt
        against the window. A BOS token counted into any of them would inflate the rate on
        short text and is not part of what the model has to produce.
        """
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    @property
    def tokenizer_id(self) -> str | None:
        """Which tokenizer this resolved, for the report. See :data:`TOKENIZER_ID_ATTR`."""
        name = getattr(self.tokenizer, "name_or_path", None)
        return str(name) if name else None

    @property
    def max_context_tokens(self) -> int | None:
        """This checkpoint's context window, or ``None`` if its config does not say.

        See :data:`CONTEXT_WINDOW_ATTR`. Resolved once at load by :func:`context_length_of`
        and republished here under the name the scorer reads, so there is one reader of the
        model config and one warning when it says nothing.
        """
        return self.context_length

    def checkpoint_facts(self) -> dict[str, Any]:
        """What was resolved off this checkpoint, for the report's ``generation_runtime``.

        ``tokenizer_identifier`` is what the loaded tokenizer says it came from. On this
        path that is the checkpoint directory, because a converted checkpoint that ships
        tokenizer files is scored from those files; the *upstream* identifier the
        conversion resolved -- ``dataset.tokenizer.identifier``, ``HuggingFaceTB/SmolLM2-135M``
        here -- is recorded by the conversion step and is not re-derivable from the output
        directory. Naming what was actually loaded is the claim this function can support.
        """
        return {
            "tokenizer_identifier": str(getattr(self.tokenizer, "name_or_path", "") or ""),
            "tokenizer_class": type(self.tokenizer).__name__,
            "tokenizer_vocab_size": self._tokenizer_size(),
            "eos_token": self.eos_token,
            "eos_token_id": self.eos_token_id,
            "eos_token_id_passed_to_generate": self._eos_kwargs() != {},
            "context_length_declared": self.context_length,
        }

    def _tokenizer_size(self) -> int | None:
        try:
            return len(self.tokenizer)
        except Exception:  # noqa: BLE001 - a probe that cannot be made is not a finding
            return None


def _load_hf(checkpoint_dir: Path, config: GenerationConfig) -> ScoringModel:
    """Grade a converted HF checkpoint by generating from it with ``transformers``.

    A function rather than the lambda this was, because two things now have to be handed
    across explicitly. The rest -- the end-token text, the tokenizer that measures a
    prompt, the context window to measure it against -- the completer publishes as
    attributes and :class:`GenerativeScorer` reads for itself, so a second backend
    supplies them the same way rather than needing a second loader that knows the order.

    What cannot be published is the *verified* end-of-text spelling, which is
    :func:`resolve_eos_token`'s answer rather than the tokenizer's label, and the
    checkpoint facts the report carries. The completer keeps the end-of-text *id*, which
    is what stops generation; the scorer takes the text, which closes the exemplars.
    """
    completer = _HFCompleter(checkpoint_dir, config)
    return GenerativeScorer(
        completer,
        config,
        eos_token=completer.eos_token,
        checkpoint_facts=completer.checkpoint_facts(),
    )


#: ``config.json`` path -> what that number means, for a raw OLMo-core checkpoint's
#: context window, most authoritative first. Used by :func:`olmo_core_context_length`.
#:
#: The ``model.*`` spellings are not retyped here: they are olmo-eval's own
#: ``olmo_core_utils._MAX_LENGTH_CONFIG_KEYS``, read off that module at resolution time so
#: the diagnostic and ``OlmoCoreProvider`` cannot end up disagreeing about the window one
#: checkpoint declares. Only the second entry is added, and it is the one this checkpoint
#: family actually writes -- ``hf_config_patch`` exists precisely because "the model object
#: does not know the sequence length it was trained at", so the architecture block is
#: silent and the dataset block is not.
OLMO_CORE_DATASET_SEQUENCE_LENGTH = ("dataset", "sequence_length")


def olmo_core_context_length(
    checkpoint_config: Mapping[str, Any] | None,
    *,
    checkpoint_dir: Path,
    model_keys: Sequence[str],
) -> tuple[int, str]:
    """This raw checkpoint's context window, and the name of what declared it.

    **Why this cannot be a constant.** On the converted path the window arrives as
    ``max_position_embeddings``, which the conversion writes from
    ``hf_config_patch.DEFAULT_MAX_POSITION_EMBEDDINGS`` -- 2048 for this family. There is
    no converted config here, so that constant is a number from a different code path
    with no claim on the directory being read, and hardcoding it would mean every future
    checkpoint silently inheriting one checkpoint's training length.

    **Why the tokenizer is not a fallback**, although ``_resolve_max_length`` uses it as
    one. A raw checkpoint ships no tokenizer and names one by identifier, so
    ``model_max_length`` here is a *third party's* number: this family resolves
    ``HuggingFaceTB/SmolLM2-135M``, whose tokenizer declares 8192 while the weights were
    trained at 2048. Taking it would leave the clamp nominally on and inert, four
    exemplars in front of every MATH stem, and a 2600-token prompt sent to a 2048-token
    model -- which is the exact silent failure the clamp exists to prevent, wearing the
    appearance of a resolved window. A wrong window is worse than no window, and a wrong
    window that is four times too large is worse than one that is too small.

    **Why an unreadable window is refused rather than run unclamped.** Running unclamped
    is what happens today when a completer publishes no
    :data:`CONTEXT_WINDOW_ATTR`, and on this bank it fabricates zeroes: MATH's long stems
    keep all four exemplars, the prompt overflows, and the model is truncated into the
    block that teaches ``\\boxed{}`` and ``Final Answer:`` -- so the grader finds nothing,
    scores 0, and the report reads as a checkpoint that cannot do mathematics. That costs
    a GPU hour and produces a number that is wrong in a direction nobody can see. This
    costs a load, names every place it looked, and is fixed either by a checkpoint that
    declares its length or by one environment variable. ``_HFCompleter`` warns instead
    of refusing in the same situation, and the asymmetry is real rather than an
    oversight: there the config is one *our own* conversion wrote and always populates,
    so its absence means a foreign checkpoint whose window may legitimately be
    unbounded; here the value comes from training and its absence means nobody knows.

    Returns:
        ``(tokens, source)``, where ``source`` names what supplied the number so the
        report can say it. Nothing branches on ``source``.

    Raises:
        RuntimeError: If no source declares one.
    """
    override = os.environ.get(hf_config_patch.MAX_POSITION_EMBEDDINGS_ENV)
    if override is not None:
        # Deliberately the variable the converted path already reads, rather than a
        # second one. It names one quantity -- this checkpoint family's context window --
        # and two spellings of it would let a native run and a converted run of the same
        # weights be clamped differently while both reports looked normal.
        # ``hf_config_patch.max_position_embeddings`` is NOT called, because it falls back
        # to its 2048 constant and this must not acquire a default by reusing a parser.
        try:
            length = int(override)
        except ValueError as exc:
            raise RuntimeError(
                f"{hf_config_patch.MAX_POSITION_EMBEDDINGS_ENV}={override!r} is not an "
                f"integer. On this path it is the context window every prompt is measured "
                f"against, so a bad value either refuses items that would have generated "
                f"fine or lets an overflowing prompt through."
            ) from exc
        if length <= 0:
            raise RuntimeError(
                f"{hf_config_patch.MAX_POSITION_EMBEDDINGS_ENV}={override!r} is not a "
                f"positive number of tokens, so no prompt could fit it and every item "
                f"would be refused."
            )
        return length, hf_config_patch.MAX_POSITION_EMBEDDINGS_ENV

    config = checkpoint_config or {}
    lookups = [("model", key) for key in model_keys]
    lookups.append(OLMO_CORE_DATASET_SEQUENCE_LENGTH)
    for block, key in lookups:
        section = config.get(block)
        if not isinstance(section, Mapping):
            continue
        value = section.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value, f"{block}.{key}"

    searched = ", ".join(f"{block}.{key}" for block, key in lookups)
    raise RuntimeError(
        f"{checkpoint_dir} declares no context window: none of {searched} is a positive "
        f"integer in its config.json (top-level keys: "
        f"{', '.join(sorted(str(key) for key in config)) or '<none>'}). This is refused "
        f"rather than run unclamped, because unclamped is not a degraded run on a "
        f"generative bank -- it is a wrong one that looks right. A prompt longer than the "
        f"window is truncated by the model into the few-shot block that teaches the "
        f"answer format, so the grader finds nothing to extract, the item scores 0, and "
        f"the report attributes that to the checkpoint. The tokenizer's model_max_length "
        f"is deliberately not used as a fallback: a raw checkpoint names its tokenizer by "
        f"identifier, so that number belongs to whatever model the identifier points at "
        f"and not to these weights. Either add the length to the checkpoint config, or "
        f"set {hf_config_patch.MAX_POSITION_EMBEDDINGS_ENV} to the sequence length this "
        f"run was trained at -- the same variable the conversion path reads."
    )


class _OlmoCoreCompleter:
    """Greedy OLMo-core generation from a raw training checkpoint, one prompt at a time.

    The generative twin of ``inference._OlmoCoreScoringModel``, and an *adapter* over it
    rather than a second loader. Everything about reading this checkpoint format --
    resolving the sharded layout, resolving the tokenizer through
    ``dataset.tokenizer.identifier``, choosing the device, threading ``dtype``, and
    fabricating the pad id that ``GenerationConfig.validate`` demands -- happens by
    constructing that class and taking its ``model``, ``tokenizer`` and ``device``. That
    is what probe ``run_019fe316-0e47`` did, so the combination is measured rather than
    assumed, and it means the next thing training changes about the format is fixed in
    one place. None of that class's scoring is reachable from here.

    What this adds is the four lines of decoding the probe established, and the
    *contract* the merged budget machinery reads off a completer. That contract is the
    real content of this class, because three of its five parts fail silently:

    - :data:`TOKEN_COUNTER_ATTR` (``count_tokens``) -- the only one whose absence is
      caught, by :func:`require_live_tokenizer`, which refuses the load.
    - :data:`BUDGET_AWARE_ATTR` (``accepts_token_budget``) -- without it every item
      quietly takes the flat ``max_new_tokens``, and on a checkpoint that never emits
      end-of-text that is the whole cap on every item.
    - :data:`CONTEXT_WINDOW_ATTR` (``max_context_tokens``) -- without it
      :meth:`PromptFitter.fit` returns unclamped and the exemplar ladder never runs. The
      dangerous one; see :func:`olmo_core_context_length`.
    - :data:`EOS_TEXT_ATTR` (``eos_text``) -- without it the leak-catching stop in
      :func:`eos_stop_sequences` is weakened.
    - :data:`TOKENIZER_ID_ATTR` and :meth:`checkpoint_facts` -- report only.

    **Two measured facts this is built around and does not try to fix.**

    ``use_cache=False`` on every call. OLMo-core's default torch attention backend
    refuses KV caching outright -- ``assert_supports_kv_cache`` raises the moment
    ``prepare_inference_cache`` runs -- and the flash backends that implement it need a
    binary wheel this image does not have. Run run_019fe2e6 died exactly there. Every
    step therefore re-reads the whole prefix, which is quadratic and slow and correct.
    ``TORCH_TODOS.md`` holds the measurement and what would lift it; note that the
    backend is fixed at ``from_checkpoint`` rather than chosen per call, so lifting it
    means a differently-built module and not a keyword here.

    This checkpoint **never emits its end-of-text token**: the probe decoded 192 tokens
    across three prompts and token 0 appeared in none of them. So the ``eos_token_id``
    the loader passed is inert for this model and every item pays its whole budget. That
    is a cost property of these weights, bounded by ``max_new_tokens`` and by the context
    clamp, and it is why the per-item budget is worth publishing rather than a nicety.
    """

    #: See :data:`BUDGET_AWARE_ATTR`. Declared on the class so the scorer can read it off
    #: the instance without constructing anything -- and so a test can assert it without
    #: a checkpoint, which is the only way this one gets asserted at all: a completer that
    #: dropped it would still load, still generate, and still produce a report.
    accepts_token_budget = True

    def __init__(self, checkpoint_dir: Path, config: GenerationConfig) -> None:
        self.config = config
        # The whole native loader, reused. `InferenceConfig` is built here rather than
        # threaded in because the caller has a GenerationConfig and the two agree on
        # exactly one field, `dtype`; passing more would mean an MCQ setting silently
        # steering a generative run.
        self._loaded = inference._OlmoCoreScoringModel(
            checkpoint_dir,
            inference.InferenceConfig(checkpoint_kind="olmo_core", dtype=config.dtype),
        )
        self._torch: Any = self._loaded._torch
        self.tokenizer: Any = self._loaded.tokenizer
        self.model: Any = self._loaded.model
        self.device: Any = self._loaded.device

        require_chat_template(self.tokenizer, checkpoint_dir, config)
        self._warn_if_vocab_sizes_disagree()
        self.eos_token, self.eos_token_id = resolve_eos_token(self.tokenizer)
        core_utils = inference._olmo_core_utils()
        self.context_length, self.context_length_source = olmo_core_context_length(
            self._loaded.checkpoint_config,
            checkpoint_dir=checkpoint_dir,
            model_keys=core_utils._MAX_LENGTH_CONFIG_KEYS,
        )
        log.info(
            "Generating from %s natively at a %d-token context window declared by %s, "
            "with KV caching off (see TORCH_TODOS.md).",
            checkpoint_dir,
            self.context_length,
            self.context_length_source,
        )

    def __call__(self, prompt: str, max_new_tokens: int | None = None) -> str:
        """Return the continuation of ``prompt``, with the prompt echo removed.

        ``max_new_tokens`` is the per-item budget :class:`GenerativeScorer` derives, and
        it defaults to the configured flat value so this is still usable as a bare
        ``prompt -> completion`` callable.

        **The prompt is in the return and has to be sliced off.** ``generate_batch``
        seeds its output with ``input_ids`` and concatenates onto it, so the first
        ``prompt_len`` positions are the prompt verbatim unless ``completions_only=True``
        is passed. Slicing is chosen over that flag because slicing is what the probe
        measured -- on all three banks the returned prefix compared equal to the input --
        while ``completions_only`` was read off the source and never exercised on these
        weights.

        Greedy decoding is spelled the way ``OlmoCoreProvider._build_generation_kwargs``
        spells it, and it is passed explicitly rather than left to the module's config.
        :meth:`GenerationConfig.__post_init__` refuses a nonzero temperature on the
        grounds that this grader decodes greedily, so a completer that inherited a
        sampling default would quietly break a promise the config enforces.
        """
        torch = self._torch
        input_ids = self.tokenizer(
            render_chat_prompt(prompt, self.tokenizer, self.config), return_tensors="pt"
        )["input_ids"]
        if self.config.max_length is not None and input_ids.shape[1] > self.config.max_length:
            input_ids = input_ids[:, -self.config.max_length :]
        input_ids = input_ids.to(self.device)
        prompt_tokens = input_ids.shape[1]

        with torch.no_grad():
            output = self.model.generate_batch(
                input_ids,
                max_new_tokens=(
                    self.config.max_new_tokens if max_new_tokens is None else max_new_tokens
                ),
                **self._decoding_kwargs(),
            )
        # `generate_batch` returns `(tokens, logprobs, ...)` when asked for logprobs and
        # the bare tokens otherwise. Read defensively exactly as the probe read it, since
        # the probe is the only thing that has seen the real return.
        generated = output[0] if isinstance(output, tuple) else output
        completion = generated[0][prompt_tokens:]
        return self.tokenizer.decode(completion.tolist(), skip_special_tokens=True)

    def _decoding_kwargs(self) -> dict[str, Any]:
        """Greedy decoding with the cache off, in the provider's own spelling.

        ``do_sample``/``temperature``/``top_k``/``top_p`` are lifted from
        ``OlmoCoreProvider._build_generation_kwargs`` at ``temperature == 0``, which is
        the only temperature :class:`GenerationConfig` permits.

        ``use_cache=False`` is the probe's fallback path and is not currently optional;
        see the class docstring. It is a keyword on the call rather than on the loaded
        module because ``generate_batch`` folds its kwargs onto the module's config with
        ``replace(**generation_kwargs)``, so turning caching back on later needs no
        reload -- while the attention backend that would make it work is fixed at
        ``from_checkpoint`` and does.
        """
        return {
            "do_sample": False,
            "temperature": 0.0,
            "top_k": -1,
            "top_p": 1.0,
            "use_cache": False,
        }

    @property
    def eos_text(self) -> str | None:
        """The literal text of this checkpoint's end token. See :data:`EOS_TEXT_ATTR`.

        The tokenizer's label as written, not the round-tripped spelling, for the reason
        :attr:`_HFCompleter.eos_text` gives: this one is used to cut a *leaked* literal
        out of the graded span, and a spelling the tokenizer does not recognize is
        exactly the one a model types as ordinary characters.
        """
        token = getattr(self.tokenizer, "eos_token", None)
        return str(token) if token else None

    def count_tokens(self, text: str) -> int:
        """Tokens ``text`` costs in this checkpoint's tokenizer. See :data:`TOKEN_COUNTER_ATTR`.

        ``add_special_tokens=False`` for the reason :meth:`_HFCompleter.count_tokens`
        gives, and it matters more here: ``inference.describe_tokenizer_defaults`` exists
        because this family's tokenizer is resolved from an identifier and may well
        prepend a BOS, which would inflate every words-to-tokens ratio measured on short
        text.
        """
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    @property
    def tokenizer_id(self) -> str | None:
        """Which tokenizer this resolved, for the report. See :data:`TOKENIZER_ID_ATTR`."""
        name = getattr(self.tokenizer, "name_or_path", None)
        return str(name) if name else None

    @property
    def max_context_tokens(self) -> int:
        """This checkpoint's context window. See :data:`CONTEXT_WINDOW_ATTR`.

        Never ``None`` on this backend, unlike the HuggingFace one:
        :func:`olmo_core_context_length` refuses the load instead of returning nothing,
        so a native generative run either has a window or does not start.
        """
        return self.context_length

    def checkpoint_facts(self) -> dict[str, Any]:
        """What was resolved off this checkpoint, for the report's ``generation_runtime``.

        ``context_length_source`` is the field that makes the number auditable, and it is
        the reason the resolver returns a name alongside a value: 2048 read out of a
        training config and 2048 supplied by an environment variable are the same clamp
        and very different claims.

        ``kv_cache`` is recorded because it is the largest single term in what a run
        costs and it is invisible in the theta. ``tokenization`` is the same record the
        MCQ path publishes under ``run.tokenization``, carried here because a generative
        scorer is not the object the runner reads that off.
        """
        return {
            "tokenizer_identifier": str(getattr(self.tokenizer, "name_or_path", "") or ""),
            "tokenizer_class": type(self.tokenizer).__name__,
            "tokenizer_vocab_size": self._tokenizer_size(),
            "eos_token": self.eos_token,
            "eos_token_id": self.eos_token_id,
            "eos_token_id_passed_to_generate": self.eos_token_id is not None,
            "context_length_declared": self.context_length,
            "context_length_source": self.context_length_source,
            "dtype_requested": self.config.dtype,
            "kv_cache": False,
            "tokenization": self._loaded.tokenizer_defaults.as_dict(),
        }

    def _tokenizer_size(self) -> int | None:
        try:
            return len(self.tokenizer)
        except Exception:  # noqa: BLE001 - a probe that cannot be made is not a finding
            return None

    def _model_vocab_size(self) -> int | None:
        """The embedding row count the checkpoint declares, or ``None`` if it does not.

        Read from the checkpoint's own config rather than off the built model, because
        the config is the artifact the tokenizer identifier was also read out of, so the
        two claims being compared come from the same place.
        """
        model = (self._loaded.checkpoint_config or {}).get("model")
        if not isinstance(model, dict):
            return None
        try:
            return int(model["vocab_size"])
        except (KeyError, TypeError, ValueError):
            return None

    def _warn_if_vocab_sizes_disagree(self) -> None:
        """Say so if the tokenizer and the checkpoint disagree about how many tokens exist.

        The native counterpart of :meth:`_HFCompleter._warn_if_tokenizer_outgrows_model`.
        It exists because this path had no such check at all: the completer recorded
        ``tokenizer_vocab_size`` in :meth:`checkpoint_facts` and compared it against
        nothing, so the one mistake this area invites was written into every report and
        caught by no one.

        That mistake is not hypothetical. The tokenizer is an identifier in the
        checkpoint config rather than files on disk, so a wrong resolution produces a
        plausible tokenizer instead of an error. Reading ``convert._resolve_tokenizer_id``'s
        fallback branch as its behaviour, this checkpoint was taken for an
        ``allenai/dolma2`` model -- 100,278 tokens against its actual 49,152 -- and this
        one comparison contradicts that immediately.

        Only the tokenizer-larger direction warns, for the reason the HF side gives: a
        tokenizer that can emit ids the model has no row for is broken, whereas a model
        larger than its tokenizer is the ordinary result of padding an embedding out to a
        multiple of 128. A warning rather than a refusal, because the refusals this class
        carries are for facts it can establish, and a heuristic should not stop a run that
        would otherwise score correctly.
        """
        tokenizer_size = self._tokenizer_size()
        model_size = self._model_vocab_size()
        if tokenizer_size is None or model_size is None:
            return
        if tokenizer_size > model_size:
            log.warning(
                "This checkpoint's tokenizer has %d tokens but its config declares "
                "vocab_size %d, so the tokenizer can produce ids the model has no "
                "embedding row for. The usual cause is a wrong "
                "dataset.tokenizer.identifier rather than a corrupt checkpoint, and both "
                "the prompts and the scores from this run are suspect.",
                tokenizer_size,
                model_size,
            )


def _load_olmo_core(checkpoint_dir: Path, config: GenerationConfig) -> ScoringModel:
    """Grade a raw OLMo-core checkpoint by generating from it, with no conversion step.

    The same two lines as :func:`_load_hf` over a different completer, which is the
    claim the per-modality registry makes: a backend is a completer plus a row in
    :data:`GENERATIVE_BACKENDS`, and everything between a prompt and a graded response
    is backend-agnostic and already written.

    Reached by ``--checkpoint-prep none --checkpoint-kind olmo_core``. Preparation has to
    be off, exactly as on the MCQ side: ``auto`` converts the sharded directory this
    reads and hands back an HF one, which ``_load_hf`` would then be the right loader
    for.

    **What this replaced.** The stub that used to be here refused, and the reason it gave
    was withdrawn by measurement rather than by argument. It held that decoding needs an
    end-of-text id distinct from the pad id, that this family writes both as 0, and that
    the exemption ``inference._OlmoCoreScoringModel`` takes therefore could not carry
    over. That confused two ids: ``GenerationConfig.validate`` rejects only ``pad ==
    eos``, so fabricating the *pad* leaves the real end-of-text at 0 and satisfies it,
    which is what that scorer already does and what probe ``run_019fe316-0e47`` decoded
    64 tokens for each of three prompts with. What was genuinely missing was the
    completer contract, and that is what :class:`_OlmoCoreCompleter` is.
    """
    completer = _OlmoCoreCompleter(checkpoint_dir, config)
    return GenerativeScorer(
        completer,
        config,
        eos_token=completer.eos_token,
        checkpoint_facts=completer.checkpoint_facts(),
    )


#: Checkpoint kind -> the generative grading backend that reads it.
#:
#: The counterpart of ``inference.MCQ_SCORING_BACKENDS``, and separate from it on purpose:
#: the two modalities register independently, so a backend can exist for one and not the
#: other. That asymmetry is a real state rather than a hypothetical -- it is what the
#: native OLMo-core reader was in until :func:`_load_olmo_core` was written, and what a
#: served backend would be in if it scored log-probs before it generated. The two tables
#: agreeing today is a fact about today, not a reason to merge them.
GENERATIVE_BACKENDS: dict[str, Callable[[Path, GenerationConfig], ScoringModel]] = {
    "hf": _load_hf,
    "olmo_core": _load_olmo_core,
}


def load_generative_model(checkpoint_dir: Path, config: GenerationConfig) -> ScoringModel:
    """Load a sampled-completion grading model for the configured checkpoint kind.

    The generative counterpart of ``inference.load_scoring_model``; :mod:`.grading`
    chooses between the two by dataset modality.

    This is also where a run without a usable tokenizer is refused, and the location is
    the design rather than convenience. See :func:`require_live_tokenizer`.
    """
    try:
        backend = GENERATIVE_BACKENDS[config.checkpoint_kind]
    except KeyError:
        raise ValueError(
            f"Unknown checkpoint_kind: {config.checkpoint_kind!r}. Registered generative "
            f"backends: {', '.join(sorted(GENERATIVE_BACKENDS))}."
        ) from None
    model = backend(checkpoint_dir, config)
    if isinstance(model, GenerativeScorer):
        require_live_tokenizer(model, checkpoint_dir)
    return model


def require_live_tokenizer(scorer: GenerativeScorer, checkpoint_dir: Path) -> None:
    """Refuse a loaded scorer that cannot count tokens.

    **Why this is a refusal and not a fallback.** A real evaluation always has a tokenizer,
    because inference is impossible without one, so a missing one is never a deployment
    shape to accommodate. It is one of two faults: the tokenizer genuinely could not be
    resolved -- the Hub unreachable for a checkpoint that names its tokenizer by identifier
    and ships no files -- or it was resolved and not threaded to the code that needs it.
    Continuing either way would hand every item the flat ``max_new_tokens`` while the report
    looked entirely normal: a plausible theta with a healthy standard error and nothing
    recording that the budgets were defaulted rather than computed. That is the
    silent-wrong-answer shape this whole budget exists to remove, so it fails here instead.

    **Why here and not in :meth:`GenerativeScorer.score_items`.** The guard belongs at the
    point a checkpoint is loaded, which is exactly the boundary between a real run and an
    offline one. Everything that grades text without a checkpoint -- the whole of the
    prompt-building and grading test surface, which passes one-argument lambdas -- never
    reaches here, so the degraded path stays reachable on purpose for the tests that
    exercise it and unreachable by accident in production. No ``is_testing`` flag, which
    would be a second thing to keep true.

    Checked by *probing* rather than by testing the attribute, because a completer that
    publishes a counter which raises is the same fault as one that publishes none, and the
    plumbing defect this is aimed at is more likely to look like the former.

    **Why this is not the same guard as** :meth:`_HFCompleter._load_tokenizer`, which also
    refuses a missing tokenizer. That one is a backend's statement about its own
    checkpoint: it fires when ``AutoTokenizer.from_pretrained`` raises, so on this backend
    it fires first and this function never sees that case. This one is the loader's
    statement about the *scorer it just built*, and the fault it exists for is the one no
    completer can detect about itself -- a tokenizer that loaded fine and did not reach
    the code that budgets with it, on this backend or on the next one, which may publish
    nothing at all. Deleting either would leave a real failure silent.
    """
    if scorer.cascade_active:
        try:
            scorer._probe_token_counter()
        except Exception as error:  # noqa: BLE001 - any failure is the same fault
            raise RuntimeError(
                f"The completer loaded for {checkpoint_dir} publishes a "
                f"{TOKEN_COUNTER_ATTR!r} that raised {error!r} when called. A generative "
                f"run needs a working tokenizer: the per-item generation budget converts "
                f"each item's declared word counts into tokens with it, and without one "
                f"every item would silently take the flat max_new_tokens="
                f"{scorer.config.max_new_tokens} instead, producing a normal-looking "
                f"report whose budgets were defaulted rather than computed. Fix the "
                f"tokenizer plumbing rather than running degraded."
            ) from error
        return
    raise RuntimeError(
        f"The completer loaded for {checkpoint_dir} publishes no {TOKEN_COUNTER_ATTR!r}, "
        f"so no tokenizer could be resolved for it. A generative run needs one: the "
        f"per-item generation budget converts each item's declared word counts into tokens "
        f"with the evaluated checkpoint's own tokenizer, and without it every item would "
        f"silently take the flat max_new_tokens={scorer.config.max_new_tokens} -- for "
        f"ifeval that means the whole bank at the ceiling, with nothing in the report to "
        f"say the budgets were defaulted rather than computed. Either the tokenizer could "
        f"not be resolved (a checkpoint naming it by Hub identifier while shipping no "
        f"tokenizer files, with the Hub unreachable) or it was resolved and not threaded "
        f"through to the completer. Both are faults to fix rather than to run past."
    )
