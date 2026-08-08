"""The one place a dataset's modality picks a grading scheme.

The frozen :class:`~diagnostics.mcq_cat.base.ScoringModel` protocol says nothing about
*how* an item is graded, and :meth:`CatStyle.score` receives an already-constructed
model, so the choice has to be made where the model is built: in the runner, before
the CAT loop starts. This module holds that choice as one table, :data:`GRADERS`, plus
the guard that stops a bank being run through the wrong entry.

The guard is the reason this module exists rather than a two-line lookup in the
runner. A generative item has ``choices == ()`` and ``gold_index == -1``; handed to
the log-likelihood scorer there is nothing to rank, so every item comes back
incorrect. EAP would then return a confident, very low theta from a run that looks
completely healthy -- the standard error shrinks as usual, p-IRT reports a plausible
accuracy, and the report is well formed. The reverse pairing is quieter still: an MCQ
item has no ``metadata["gold_answer"]`` to match a completion against. Both are caught
here, before a checkpoint is loaded, and the message names the dataset, the modality
and the grader so the mistake is obvious from a log line.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

from ..base import BenchmarkItem, ScoringModel
from . import generative, inference

log = logging.getLogger("mcq_cat.grading")

#: Items carry a choice set and are graded by continuation log-likelihood.
MCQ = "mcq"
#: Items carry a gold answer string and are graded by sampling and extraction.
GENERATIVE = "generative"


class ModalityMismatch(ValueError):
    """A bank is about to be graded by the scheme for the other modality."""


@dataclass
class GradingSettings:
    """Configuration for both schemes. Only the selected scheme's half is read.

    The runner fills this from its CLI flags once and does not need to know which
    half will be used.
    """

    mcq: inference.InferenceConfig = field(default_factory=inference.InferenceConfig)
    generation: generative.GenerationConfig = field(default_factory=generative.GenerationConfig)


@dataclass(frozen=True, slots=True)
class Grader:
    """One grading scheme: what makes an item correct, and how to load it."""

    modality: str
    summary: str
    load: Callable[[Path, GradingSettings], ScoringModel]


def _load_mcq(checkpoint_dir: Path, settings: GradingSettings) -> ScoringModel:
    """Construct the continuation log-likelihood scorer."""
    return inference.load_scoring_model(checkpoint_dir, settings.mcq)


def _load_generative(checkpoint_dir: Path, settings: GradingSettings) -> ScoringModel:
    """Construct the sampled-completion scorer."""
    return generative.load_generative_model(checkpoint_dir, settings.generation)


#: Modality -> grading scheme. This is the entire dispatch. A third modality means a
#: module beside ``inference.py`` and ``generative.py`` and one more row here; nothing
#: else in the harness branches on modality.
GRADERS: dict[str, Grader] = {
    MCQ: Grader(
        modality=MCQ,
        summary="continuation log-likelihood over the item's answer choices, argmax",
        load=_load_mcq,
    ),
    GENERATIVE: Grader(
        modality=GENERATIVE,
        summary=(
            "greedy sampled completion, decided by the grader the item's "
            "metadata['answer_type'] names: answer extraction and a strict match "
            "against metadata['gold_answer'] for a bank that has one, constraint "
            "verification against the item for a bank that does not"
        ),
        load=_load_generative,
    ),
}


@dataclass(frozen=True, slots=True)
class GradingRequest:
    """How one resolved bank must be graded.

    A style produces this and the runner consumes it. It exists because the runner is
    style-agnostic -- it knows nothing about banks, manifests or allowlists -- while
    modality is a property of the dataset the style resolved.

    ``generation`` carries the style's generative overrides (few-shot count, token
    budget, prompt style, stop sequences) and ``mcq`` its log-likelihood overrides
    (prompt style); each is ignored for the other modality. Both are the style's block
    verbatim, including any :data:`PER_DATASET_KEY` map, and
    :func:`resolve_dataset_overrides` narrows them to ``dataset`` -- so a style never has
    to know that benchmarks within a modality disagree about prompting.
    """

    dataset: str
    modality: str = MCQ
    generation: Mapping[str, Any] = field(default_factory=dict)
    mcq: Mapping[str, Any] = field(default_factory=dict)


def request_for(style: object, *, dataset: str) -> GradingRequest:
    """Ask ``style`` how its resolved bank must be graded.

    ``CatStyle`` is frozen and has no modality method, so this is an optional hook.
    A style that does not implement ``grading_request`` is taken to be MCQ, which is
    what every style written before generative support is.
    """
    hook = getattr(style, "grading_request", None)
    if hook is None:
        return GradingRequest(dataset=dataset)
    request = hook()
    if not isinstance(request, GradingRequest):
        raise TypeError(
            f"{type(style).__name__}.grading_request returned {type(request).__name__}, "
            f"expected a GradingRequest."
        )
    return request


def check_scoring_convention(
    style: object, request: GradingRequest, settings: GradingSettings
) -> None:
    """Ask ``style`` to confirm its bank was vendored under the settings about to be used.

    An optional hook on the same terms as :func:`request_for`: ``CatStyle`` is frozen,
    and a style that keeps no record of what its bank was calibrated against has
    nothing to check. Routed through here rather than called from the runner directly
    so the runner stays ignorant of banks and manifests, and so the hook receives the
    settings the scorer will actually be built from -- :func:`resolve_settings` is the
    same fold :func:`load_grader` performs, not a second reading of the style's config.
    That distinction is the point of the hook: a style comparing its own ``config.yaml``
    against its own manifest would agree with itself no matter what the caller passed.
    """
    hook = getattr(style, "check_scoring_convention", None)
    if hook is None:
        return
    hook(request, resolve_settings(request, settings))


def get_grader(modality: str) -> Grader:
    """Return the grading scheme for ``modality``, or raise naming the known ones."""
    try:
        return GRADERS[modality]
    except KeyError:
        raise ValueError(
            f"Unknown modality {modality!r}. Known modalities: {', '.join(sorted(GRADERS))}."
        ) from None


def _item_matches(modality: str, item: BenchmarkItem) -> bool:
    """Whether ``item``'s shape is the one ``modality`` implies.

    The generative half asks the item's own grader whether the item is complete rather
    than testing for a gold answer. Not every generative benchmark has one -- IFEval
    decides correctness by running verifiers named in the item's metadata -- and a
    fixed gold test would reject all 511 of its items as malformed while accepting an
    IFEval item stripped of its constraints.

    The MCQ half bounds the gold above as well as below, because the scorer compares it
    against an argmax over ``choices`` and a gold past the end simply never equals one.
    The item is then wrong for every checkpoint, and the zero it contributes is the
    bank's rather than the model's -- indistinguishable in the response pattern from an
    item the model genuinely failed, and pulling theta down with no sign of why.
    """
    if modality == GENERATIVE:
        return not item.choices and generative.is_gradable(item)
    return bool(item.choices) and 0 <= item.gold_index < len(item.choices)


_SHAPE_HINTS = {
    MCQ: (
        "An MCQ item needs a non-empty choices tuple and a gold_index into it; these "
        "look like generative items, which carry no choices and keep their answer in "
        "metadata['gold_answer']."
    ),
    GENERATIVE: (
        "A generative item needs an empty choices tuple and whatever the grader named "
        "by its metadata['answer_type'] decides on -- a metadata['gold_answer'] for the "
        "gold-matched graders, the constraint payload for a verifier-scored one. These "
        "either look like MCQ items or are missing that payload."
    ),
}


def check_bank_modality(request: GradingRequest, items: Sequence[BenchmarkItem]) -> None:
    """Verify the bank's items have the shape their declared modality implies.

    Called by the runner after the bank loads and before the checkpoint is fetched,
    so a mismatch costs nothing. It raises rather than warns because the failure it
    prevents is not a crash: it is a completed run that publishes a confident theta
    computed from grading that never had a chance of being right.
    """
    grader = get_grader(request.modality)
    offenders = [item for item in items if not _item_matches(request.modality, item)]
    if not offenders:
        log.info(
            "Grading %s as %s: %d items, %s",
            request.dataset,
            request.modality,
            len(items),
            grader.summary,
        )
        return

    sample = ", ".join(item.item_id for item in offenders[:5])
    raise ModalityMismatch(
        f"Dataset {request.dataset!r} declares modality {request.modality!r}, graded by "
        f"{grader.summary}, but {len(offenders)} of {len(items)} items are not shaped "
        f"that way (for example: {sample}). {_SHAPE_HINTS[request.modality]} Grading "
        f"them anyway would produce a well-formed report and a meaningless theta, so it "
        f"stops here."
    )


#: Reserved key inside a style's per-modality settings block, holding
#: ``dataset -> overrides``. Not a field of either config dataclass, and must never
#: become one: it is stripped before the remaining keys are validated against them.
PER_DATASET_KEY = "datasets"


def resolve_dataset_overrides(
    overrides: Mapping[str, Any], *, dataset: str, block: str = "Generation"
) -> dict[str, Any]:
    """Flatten one of a style's settings blocks for one dataset.

    Benchmarks within a modality do not share a prompting convention, and each bank was
    calibrated under its own. On the generative side GSM8K is 8-shot
    ``Question:``/``Answer:`` inside 512 tokens while MATH is 4-shot
    ``Problem:``/``Solution:`` inside 1024; on the MCQ side ARC frames the stem as
    ``Question:``/``Answer:``, HellaSwag frames it not at all, and WinoGrande substitutes
    the choice into the stem. A style holds one block per modality, so a block carries a
    :data:`PER_DATASET_KEY` map and the entry matching the resolved bank wins over the
    shared keys. Resolving here rather than in the style keeps ``config.yaml`` a plain
    map of config-dataclass field names, which is the schema the misspelling guards can
    actually check.

    A dataset with no entry gets the shared keys unchanged, which is what keeps every
    bank vendored before a second convention existed running exactly as before.

    Args:
        overrides: The style's block, verbatim.
        dataset: The resolved bank's name.
        block: How the block is named in error messages, so a misconfigured MCQ block
            does not report itself as a generation setting.
    """
    base = dict(overrides)
    per_dataset = base.pop(PER_DATASET_KEY, None) or {}
    if not isinstance(per_dataset, Mapping):
        raise ValueError(
            f"{block} setting {PER_DATASET_KEY!r} must be a mapping of dataset name "
            f"to overrides, got {type(per_dataset).__name__}."
        )
    specific = per_dataset.get(dataset) or {}
    if not isinstance(specific, Mapping):
        raise ValueError(
            f"{block} overrides for dataset {dataset!r} must be a mapping, got "
            f"{type(specific).__name__}."
        )
    if specific:
        log.info(
            "Applying %d dataset-specific %s setting(s) for %s: %s",
            len(specific),
            block.lower(),
            dataset,
            ", ".join(sorted(specific)),
        )
    return {**base, **dict(specific)}


def _reject_unknown(resolved: Mapping[str, Any], config_cls: type, *, label: str) -> None:
    """Raise unless every resolved key names a field of ``config_cls``.

    A misspelled key in ``config.yaml`` would otherwise leave the pinned default in
    place and score the run under a convention nobody asked for -- exactly the silent
    divergence from the calibration that this harness exists to make impossible.
    """
    known = {f.name for f in fields(config_cls)}
    unknown = sorted(set(resolved) - known)
    if unknown:
        raise ValueError(
            f"Unknown {label} settings {unknown}. {config_cls.__name__} accepts: "
            f"{', '.join(sorted(known))}, plus {PER_DATASET_KEY!r} holding per-dataset "
            f"overrides of those same fields."
        )


def _apply_generation_overrides(
    settings: GradingSettings, overrides: Mapping[str, Any], *, dataset: str
) -> GradingSettings:
    """Fold a style's generative settings onto the defaults, rejecting unknown keys."""
    resolved = resolve_dataset_overrides(overrides, dataset=dataset, block="Generation")
    if not resolved:
        return settings
    _reject_unknown(resolved, generative.GenerationConfig, label="generation")
    return replace(settings, generation=replace(settings.generation, **resolved))


def _apply_mcq_overrides(
    settings: GradingSettings, overrides: Mapping[str, Any], *, dataset: str
) -> GradingSettings:
    """Fold a style's MCQ settings onto the defaults, rejecting unknown keys.

    The log-likelihood half of the same mechanism, and it matters for the same reason:
    a dropped ``prompt_style`` leaves the default framing in place, which scores the
    item behind a prompt its difficulty was not estimated under and reports the
    difference as ability.
    """
    resolved = resolve_dataset_overrides(overrides, dataset=dataset, block="MCQ")
    if not resolved:
        return settings
    _reject_unknown(resolved, inference.InferenceConfig, label="mcq")
    return replace(settings, mcq=replace(settings.mcq, **resolved))


def resolve_settings(request: GradingRequest, settings: GradingSettings) -> GradingSettings:
    """Fold ``request``'s style blocks onto ``settings`` for its dataset.

    Both blocks are folded on unconditionally. Which of them carries anything is the
    style's decision -- it populates only the one matching its bank's modality -- so
    there is no modality switch here beside :data:`GRADERS`, and an empty block folds
    to a no-op.

    Factored out of :func:`load_grader` so the convention check and the scorer are
    built from one resolution rather than two. Folding is idempotent, so a caller that
    has already resolved may still pass the result to :func:`load_grader`.
    """
    resolved = _apply_generation_overrides(settings, request.generation, dataset=request.dataset)
    return _apply_mcq_overrides(resolved, request.mcq, dataset=request.dataset)


#: Modality -> the checkpoint formats that modality's grader can build a model from.
#:
#: Both entries read the same today, and the dict is still the right shape rather than a
#: shared tuple. A backend is registered per modality -- ``inference.py`` and
#: ``generative.py`` keep their own registries -- so the two can legitimately disagree,
#: and they have: the MCQ scorer gained a native OLMo-core loader while the generative
#: completer was still a stub. Collapsing this to one tuple makes that state unsayable,
#: and a single list naming the union would wave a generative run past the one check
#: standing in front of ``resolve_checkpoint``.
LOADABLE_CHECKPOINT_KINDS: dict[str, tuple[str, ...]] = {
    MCQ: ("hf",),
    GENERATIVE: ("hf",),
}

#: Modality -> the :class:`GradingSettings` field its grader is built from, and the
#: loader that would have to grow a branch for a format that field names.
_CHECKPOINT_SOURCES = {
    MCQ: ("mcq", "inference.load_scoring_model"),
    GENERATIVE: ("generation", "generative.load_generative_model"),
}


def check_checkpoint_kind(request: GradingRequest, settings: GradingSettings) -> None:
    """Refuse a checkpoint format this bank's grader cannot load, before one is staged.

    Called by the runner beside the bank and convention checks, and for the same reason
    they are called there rather than left to fail on their own: the failure is already
    certain from a flag, and the step after these is
    :func:`~diagnostics.mcq_cat.common.s3_io.resolve_checkpoint`, which pulls every
    object under an ``s3://`` prefix. Discovering an unloadable format inside the loader
    means discovering it after a multi-gigabyte download, and the message it gives is
    exactly as useful before the download as after it.

    Read off :func:`resolve_settings`, the fold :func:`load_grader` performs, so the
    format checked is the one a model would be built from rather than the one the caller
    passed before a style's overrides applied.

    Only the half matching ``request.modality`` is read, because only that half is ever
    loaded from. The runner fills both from one ``--checkpoint-kind``, so an MCQ bank
    carries a generation half naming a format nothing will ask the generative completer
    to load; checking it would refuse runs that are fine.

    Note that preparation runs *after* this check: a native OLMo-core directory converted
    by :mod:`~diagnostics.mcq_cat.common.convert` is loaded as ``hf``, so the kind named
    here describes the backend, not what is in the bucket.

    Raises:
        NotImplementedError: If the half this bank is graded from names a format outside
            its modality's entry in :data:`LOADABLE_CHECKPOINT_KINDS`. The same error the
            loader would raise, so a caller that skips this check is no worse informed,
            only later.
    """
    modality = get_grader(request.modality).modality
    label, loader = _CHECKPOINT_SOURCES[modality]
    loadable = LOADABLE_CHECKPOINT_KINDS[modality]
    kind = getattr(resolve_settings(request, settings), label).checkpoint_kind
    if kind not in loadable:
        raise NotImplementedError(
            f"checkpoint_kind {kind!r} on the {label} settings cannot be loaded by the "
            f"{modality} grader. Loadable formats: {', '.join(loadable)}. Raised here "
            f"rather than at model construction so the checkpoint is not staged first; "
            f"see {loader} for what implementing it needs."
        )


def load_grader(
    request: GradingRequest, checkpoint_dir: Path, settings: GradingSettings
) -> ScoringModel:
    """Construct the :class:`ScoringModel` for ``request``'s modality."""
    grader = get_grader(request.modality)
    resolved = resolve_settings(request, settings)
    log.info("Loading %s grader for %s: %s", grader.modality, request.dataset, grader.summary)
    return grader.load(checkpoint_dir, resolved)
