"""The scoring convention a bank was vendored under, recorded and then checked.

An item's calibrated difficulty is a statement about one presentation and one grading
rule. EAP treats that difficulty as fixed truth, so a run that grades under a different
convention has nowhere to put the difference except theta: the estimate moves by the
whole gap, the standard error is unaffected, p-IRT reports a plausible accuracy and the
report is well formed. There is nothing in the output to read it off. That is the
failure this module exists to make impossible.

It has two halves, and keeping them apart is the point.

The **run-time** convention is what this harness does: the modality, the prompt style,
the shot count, and the rule that turns a response into a binary. It is pinned in the
style's ``config.yaml``, written into every manifest by
:func:`manifest_block` at vendoring time, and compared against the settings a run
actually resolved by :func:`check_runtime_convention` before the checkpoint is fetched.
Both sides go through :func:`runtime_convention`, so the manifest records what the
harness will do rather than a second description of it that is free to drift.

The **calibration** convention is what the bank's parameters were fit against, and it
lives in :class:`~diagnostics.mcq_cat.styles.uni_mcq.datasets.CalibrationConvention`
because it is authored provenance rather than anything derivable. Most of it is
:data:`~diagnostics.mcq_cat.styles.uni_mcq.datasets.UNRECORDED`, which is the honest
answer and a useful one: it says a theta from this bank cannot be checked against a
published one. Nothing compares it to a run, because a difference between the two is
not a misconfiguration to be corrected -- it is a property of the bank, and the two
banks here that are known to differ (leaderboard_math's metric, arc_challenge's shot
count) are deliberate and documented in ``calibrated_datasets/DEVIATIONS.md``.

Which fields belong in the run-time half is decided by one question: does changing it
change whether an individual item is answered correctly? A prompt style, a shot count, a
stop sequence, a token budget and a prompt-length cap all do, so all of them are
recorded and a difference is an error. ``max_items``, ``se_threshold``, ``batch_size``,
``device_map``, ``checkpoint_kind`` and ``seed`` do not -- they decide how many items
are administered, how fast, and on what -- so they are not recorded and are free to
vary, with the report already noting when ``max_items`` departs from the pinned value.
The line matters in both directions: a guard that failed a run over a batch size is one
people learn to route around.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ...common import generative, grading, inference
from .datasets import DatasetSpec
from .resolve import MANIFEST_NAME, DatasetNotAvailable, ResolvedBank

log = logging.getLogger("mcq_cat.uni_mcq.convention")

#: The style's pinned settings. Read here rather than in ``style.py`` because every
#: caller that needs it is asking the same question -- what convention does this style
#: pin for this dataset -- and vendoring has to ask it without constructing a style.
CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

#: Manifest key holding the whole block.
CONVENTION_KEY = "scoring_convention"

#: Sub-blocks of :data:`CONVENTION_KEY`. Only ``runtime`` is checked at startup.
RUNTIME_KEY = "runtime"
CALIBRATION_KEY = "calibration"

#: Whether the block was written by the vendoring run or reconstructed afterwards by
#: ``scripts/migrate_manifests.py``. Not cosmetic: a vendored block records the config
#: that was in force when the artifacts were produced, while a migrated one is a claim
#: about the config as it stands now, since the six banks that predate this recorded
#: nothing to compare against.
RECORDED_BY_KEY = "recorded_by"

#: ``config.yaml``'s two per-modality settings blocks.
MCQ_BLOCK = "mcq"
GENERATIVE_BLOCK = "generative"


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Read ``config.yaml``, falling back to the pinned defaults."""
    try:
        import yaml
    except ImportError:
        log.warning("PyYAML unavailable; using pinned defaults for uni_mcq config.")
        return {}
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        log.warning("Could not read %s (%s); using pinned defaults.", path, exc)
        return {}
    return loaded if isinstance(loaded, dict) else {}


def grading_request(
    spec: DatasetSpec,
    *,
    modality: str,
    mcq: Mapping[str, Any],
    generation: Mapping[str, Any],
) -> grading.GradingRequest:
    """Build the request that declares how ``spec``'s bank must be graded.

    One builder for both callers. The style asks it at run time and vendoring asks it
    offline, and they have to agree exactly, because what vendoring writes into the
    manifest is what a run is then required to match. Two constructions of the same
    thing would be free to diverge, and the first sign of it would be a bank nobody can
    run or, worse, one that agrees with itself while scoring under the wrong prompt.

    The block for the other modality is dropped rather than passed through: an MCQ bank
    has no generative settings to honour, and a leftover block would be folded onto a
    config that ignores it and then recorded as though it had applied.
    """
    return grading.GradingRequest(
        dataset=spec.name,
        modality=modality,
        generation=generation if modality == grading.GENERATIVE else {},
        mcq=mcq if modality == grading.MCQ else {},
    )


def _mcq_convention(config: inference.InferenceConfig) -> dict[str, Any]:
    """The recorded convention for a log-likelihood bank.

    ``score_normalization`` was recorded here while it was still a constant of the
    scoring scheme, which is why the four banks that predate its being configurable
    already carry it and did not have to be re-vendored when MuSR arrived needing
    ``acc_norm``. That is the case for recording a fact rather than only the settings
    that happen to be settable today: the guard covering a new field costs nothing on
    the day it stops being fixed. ``num_fewshot`` is the second field to make that
    passage, and it made it without touching the four manifests, since every style they
    resolve frames the item's stem alone and still reports 0.

    It is read off the prompt style rather than off a constant because BBH's stems are
    finished 3-shot prompts, frozen at vendoring because a per-*subtask* description is
    not something a per-dataset setting can carry. Recording 0 for those items would
    describe a prompt no model is shown, and recording the style's own count turns the
    edit that would drop the prefix -- swapping BBH onto a stem-framing style -- into a
    startup refusal rather than a quietly 0-shot run against 3-shot difficulties.

    ``max_length`` is a settable field of both configs and is recorded for that reason
    rather than for its default, which is no limit. It reads as a memory knob and is
    not one: it truncates the prompt from the left, so a value low enough to reach the
    stem scores a different question than the one the bank holds a difficulty for. A
    frozen 3-shot stem is the case where that is likeliest to bite, being several times
    the length of anything else vendored here.
    """
    return {
        "modality": grading.MCQ,
        "prompt_style": config.prompt_style,
        "num_fewshot": inference.get_mcq_prompt_style(config.prompt_style).num_fewshot,
        "score_normalization": config.score_normalization,
        "max_length": config.max_length,
    }


#: Key recording the per-item budget policy, for the one bank that has one.
PER_ITEM_BUDGET_KEY = "per_item_token_budget"


def _per_item_budget_record(answer_type: str) -> dict[str, Any] | None:
    """The per-item generation budget policy, or ``None`` for a bank with none.

    ``max_new_tokens`` keeps its full meaning for this bank and does not narrow: it is the
    *ceiling*, the cascade may only lower an item below it, and it is what an item gets
    when no tokenizer is available to lower it with. So the recorded integer still bounds
    every generation the bank can produce, which is what the guard was protecting.

    What it stops covering is where inside that ceiling each item lands, and the constants
    that decide it are recorded here instead. The budget is a pure function of these, the
    item's own ``instruction_id_list``, and the checkpoint's tokenizer; the items are fixed
    by the manifest's ``sha256``, so pinning the constants pins everything about the policy
    that belongs to the benchmark. Change one and a run refuses until the bank is
    re-stamped.

    One term is deliberately *not* pinned, and is recorded as prose so its absence cannot
    read as an oversight. The words-to-tokens ratio is measured at run time from the
    evaluated checkpoint's own tokenizer, so it is a property of the model rather than the
    benchmark -- like the run-time EOS in
    :func:`~diagnostics.mcq_cat.common.generative.eos_stop_sequences`, and invisible to
    :func:`check_runtime_convention` for the same reason: that guard runs before the
    checkpoint is fetched and can only check what the benchmark declares. Pinning a ratio
    would mean re-stamping the bank for every model evaluated against it.

    Returned as ``None``, and then omitted, for every other bank rather than recorded as
    null. ``_differences`` reports a key the manifest lacks as loudly as one it
    disagrees with, so emitting it unconditionally would invalidate all eight sibling
    manifests and require them re-stamped for a field none of them has a policy for.
    """
    if answer_type != generative.IFEVAL_ANSWER_TYPE:
        return None
    return {
        "policy": "ifeval_length_signal_cascade",
        "tokens_per_word": "resolved at runtime from the checkpoint tokenizer",
        "detection_headroom_tokens": generative.DETECTION_HEADROOM_TOKENS,
        "words_per_sentence": generative.WORDS_PER_SENTENCE,
        "words_per_paragraph": generative.WORDS_PER_PARAGRAPH,
        "words_per_bullet": generative.WORDS_PER_BULLET,
        "unconstrained_floor_tokens": generative.UNCONSTRAINED_FLOOR_TOKENS,
        "truncation_fragile_ids": sorted(generative.TRUNCATION_FRAGILE_IDS),
    }


def _generative_convention(
    spec: DatasetSpec, config: generative.GenerationConfig
) -> dict[str, Any]:
    """The recorded convention for a sampled-completion bank.

    ``grader`` is resolved from ``spec.answer_type`` rather than read off the bank's
    items, which are stamped from the same field by the same vendoring run and so cannot
    disagree with it without the file having been hand-edited. The allowlist can and
    does move underneath a committed bank, and that is the case worth catching: a
    ``leaderboard_math`` retyped as ``numeric`` would grade LaTeX answers by their last
    digit and mark most of the bank correct.

    ``fewshot_source`` is recorded as ``None`` at 0 shots because that is what a run
    does with it -- no block is loaded and the field is never read -- so pinning the
    name a 0-shot dataset happens to inherit would fail runs over a setting with no
    effect on any score.

    ``max_new_tokens`` is the flat budget for every bank but ifeval, whose budget is
    per item; see :func:`_per_item_budget_record` for what that does to this block.
    """
    recorded = {
        "modality": grading.GENERATIVE,
        "prompt_style": config.prompt_style,
        "num_fewshot": config.num_fewshot,
        "fewshot_source": config.fewshot_source if config.num_fewshot else None,
        "chat_format": config.chat_format,
        "system_prompt_source": config.system_prompt_source,
        "stop_sequences": list(config.stop_sequences),
        "max_new_tokens": config.max_new_tokens,
        "max_length": config.max_length,
        "grader": generative.get_answer_grader(spec.answer_type).name,
    }
    per_item = _per_item_budget_record(spec.answer_type)
    if per_item is not None:
        recorded[PER_ITEM_BUDGET_KEY] = per_item
    return recorded


def runtime_convention(
    spec: DatasetSpec, settings: grading.GradingSettings, *, modality: str
) -> dict[str, Any]:
    """The convention ``settings`` scores ``spec``'s bank under.

    Read off the resolved settings the scorer is built from, never off ``config.yaml``
    directly, so what is recorded and what is checked are both the thing that actually
    happens.

    Args:
        spec: The dataset, for the fields that are properties of the benchmark rather
            than of the settings.
        settings: Already folded by
            :func:`~diagnostics.mcq_cat.common.grading.resolve_settings`.
        modality: The modality the bank is being graded as. Passed rather than taken
            from ``spec`` because the manifest is what settles it at run time, and a
            manifest disagreeing with the allowlist is refused elsewhere by
            ``UniMcqStyle.bank_modality`` rather than silently resolved here.
    """
    if modality == grading.GENERATIVE:
        return _generative_convention(spec, settings.generation)
    if modality == grading.MCQ:
        return _mcq_convention(settings.mcq)
    raise ValueError(
        f"{spec.name}: cannot describe the scoring convention of unknown modality "
        f"{modality!r}. Known modalities: {', '.join(sorted(grading.GRADERS))}."
    )


def configured_convention(spec: DatasetSpec, config: Mapping[str, Any]) -> dict[str, Any]:
    """The convention ``config`` pins for ``spec``, resolved as a run would resolve it.

    The offline entry point, used by vendoring and by the test that holds every
    committed manifest to the shipped ``config.yaml``. It goes the long way round --
    build the request, fold the overrides, read the resulting config object -- so that a
    per-dataset override, a rejected unknown key or a default that moves is reflected
    here for the same reason it would be reflected in a run.
    """
    request = grading_request(
        spec,
        modality=spec.modality,
        mcq=dict(config.get(MCQ_BLOCK) or {}),
        generation=dict(config.get(GENERATIVE_BLOCK) or {}),
    )
    resolved = grading.resolve_settings(request, grading.GradingSettings())
    return runtime_convention(spec, resolved, modality=spec.modality)


def manifest_block(
    spec: DatasetSpec, config: Mapping[str, Any], *, recorded_by: str
) -> dict[str, Any]:
    """The whole ``scoring_convention`` block for ``spec``'s manifest."""
    return {
        RECORDED_BY_KEY: recorded_by,
        RUNTIME_KEY: configured_convention(spec, config),
        CALIBRATION_KEY: spec.calibration.as_dict(),
    }


def _differences(recorded: Mapping[str, Any], actual: Mapping[str, Any]) -> list[str]:
    """Describe every way ``actual`` departs from ``recorded``, in field order.

    A field the manifest does not carry is reported as loudly as one it disagrees with.
    The recorded set is fixed by the modality, so a gap means the manifest was written
    against a different schema and the fields it does carry cannot be trusted to be the
    whole convention either.
    """
    lines = []
    for field, value in actual.items():
        if field not in recorded:
            lines.append(f"  {field}: the manifest records nothing, this run resolved {value!r}")
        elif recorded[field] != value:
            lines.append(
                f"  {field}: the manifest records {recorded[field]!r}, this run resolved {value!r}"
            )
    for field in recorded:
        if field not in actual:
            lines.append(
                f"  {field}: the manifest records {recorded[field]!r}, which this "
                f"harness no longer has a setting for"
            )
    return lines


def check_runtime_convention(
    resolved: ResolvedBank, settings: grading.GradingSettings, *, modality: str
) -> None:
    """Refuse to score a bank under a convention other than the one it records.

    An error and not a warning, on the same grounds as
    :func:`~diagnostics.mcq_cat.styles.uni_mcq.resolve.check_fit_family`: the convention
    is a property of how the bank was produced, not a run-time preference, and the run
    it would otherwise produce is not a degraded measurement but a confident wrong one.
    A warning here would scroll past in a GPU log and the number would be published.

    Raises:
        DatasetNotAvailable: If the manifest carries no convention block, or if the
            settings this run resolved disagree with the one it does carry.
    """
    spec = resolved.spec
    manifest_path = resolved.root / MANIFEST_NAME
    recorded = resolved.manifest.get(CONVENTION_KEY)

    if not isinstance(recorded, Mapping) or not isinstance(recorded.get(RUNTIME_KEY), Mapping):
        raise DatasetNotAvailable(
            f"The {spec.name} bank at {manifest_path} records no {CONVENTION_KEY!r} "
            f"block, so there is nothing to check this run's prompt style, shot count "
            f"and grader against. Banks vendored before the block existed are in this "
            f"state, and running one is not safe merely because it used to be: the "
            f"whole gap between the convention its difficulties were estimated behind "
            f"and whatever this run is configured for would land in theta with a "
            f"healthy standard error beside it. Add the block without touching the "
            f"parameters:\n"
            f"    python -m diagnostics.mcq_cat.styles.uni_mcq.scripts.migrate_manifests "
            f"--dataset {spec.name}"
        )

    actual = runtime_convention(spec, settings, modality=modality)
    differences = _differences(recorded[RUNTIME_KEY], actual)
    if not differences:
        log.info(
            "Scoring convention matches the %s manifest: %s",
            spec.name,
            ", ".join(f"{key}={value!r}" for key, value in actual.items()),
        )
        return

    raise DatasetNotAvailable(
        f"The {spec.name} bank was vendored to be scored under one convention and this "
        f"run is configured for another:\n" + "\n".join(differences) + "\n"
        f"An item's difficulty was estimated behind a particular presentation and "
        f"grading rule, and EAP treats it as fixed, so the difference does not degrade "
        f"the estimate -- it moves theta by the whole of itself and leaves the standard "
        f"error looking healthy. Either restore the recorded values in {CONFIG_PATH}, "
        f"or, if the new convention is the right one, re-vendor the bank under it so "
        f"the manifest and the run agree about what was measured:\n"
        f"    python -m diagnostics.mcq_cat.styles.uni_mcq.scripts.vendor_bank "
        f"--dataset {spec.name} --source-ref origin/Research"
    )
