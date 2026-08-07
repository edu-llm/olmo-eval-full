"""The ``uni_mcq`` style: unidimensional Fisher-information CAT over a vendored bank.

Implements the seven :class:`~diagnostics.mcq_cat.base.CatStyle` methods. The frozen
engine in ``common/cat_loop.py`` sequences them; every piece of IRT math lives here,
in :mod:`.irt` and :mod:`.pirt`, ported from the Research implementation.

What this style owns:

- resolving a dataset name to a vendored bank (:mod:`.resolve`)
- declaring how that bank must be graded, via :meth:`UniMcqStyle.grading_request`
- selecting the next item by maximum Fisher information
- re-estimating ability by EAP after each response
- the stopping rule
- assembling the report, including p-IRT predicted accuracy

What it delegates unchanged to the shared code: loading ``items.jsonl`` and
``params.json``, and grading against the checkpoint. Grading comes in two schemes --
continuation log-likelihood for MCQ banks, sampled completions for generative ones --
and this style picks between them only by naming its bank's modality; the table that
turns that name into a scorer lives in ``common/grading.py``. Both schemes take their
prompt convention per dataset from ``config.yaml``, since a benchmark's presentation is
part of what its difficulties were estimated under and no two of these benchmarks agree
on one.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ...base import (
    AbilityEstimate,
    BenchmarkBank,
    BenchmarkItem,
    CATReport,
    CATState,
    CatStyle,
    IRTBank,
    ItemResponse,
    ScoringModel,
)
from ...common import generative, grading, inference, s3_io
from ...common.benchmark_download import load_items_from_jsonl
from ...common.irt_params import load_irt_params
from . import convention
from . import resolve as resolve_mod
from .datasets import MODALITIES
from .irt import eap_theta_se, fisher_info
from .pirt import observed_accuracy, pirt_accuracy

log = logging.getLogger("mcq_cat.uni_mcq")

#: Fallbacks if config.yaml is unreadable. Kept identical to the committed values so a
#: parse failure cannot silently change the test length.
DEFAULT_SE_THRESHOLD = 0.3
DEFAULT_MIN_ITEMS = 8
DEFAULT_MAX_ITEMS = 40

#: How ``config.yaml``'s per-dataset CAT block is named in error messages.
CAT_BLOCK = "CAT"

#: The only CAT setting a dataset may override, so a misspelling raises here instead of
#: leaving the shared floor in place under a name that looks like it took effect.
#:
#: ``se_threshold`` and ``max_items`` are deliberately not in this list. Both already
#: have a CLI flag, both are recorded per run, and a report that departs from the pinned
#: cap says so; a second, quieter way to move them per dataset would make two runs of the
#: same bank incomparable with nothing in either report to show why.
PER_DATASET_CAT_KEYS = ("min_items",)

#: How each modality was actually graded, stamped into every report. A theta is not
#: interpretable without the scoring convention that produced it, and the two schemes
#: put it on different scales relative to their banks' calibrations.
#:
#: Both entries are templates rather than sentences, because in each case only part of
#: the convention is shared. Every MCQ bank is ranked by continuation log-likelihood and
#: they disagree about how those log-likelihoods are compared, so the ``{scoring}``
#: clause is filled from the normalization the run resolved, and they no longer agree on
#: the shot count either, so ``{shots}`` is filled from the resolved prompt style: BBH's
#: stems are finished 3-shot prompts and a note claiming 0-shot would describe something
#: the model was never shown. A style whose count needs explaining rather than merely
#: stating adds a sentence after the template instead of inside it, so the four banks
#: whose exemplar count is 0 keep the note they have always carried, word for word. Every
#: generative bank is sampled the same way and each defines a correct answer differently,
#: so ``{grading}`` is filled from the grader the run actually used -- read off the
#: responses, not off the allowlist, so a report can never describe a grader other than
#: the one that produced its theta.
SCORING_NOTES = {
    "mcq": (
        "Items are scored here by {shots}-shot continuation log-likelihood, {scoring}, and "
        "presented behind the benchmark's own prompt format -- recorded as prompt_style "
        "beside this note, because the MCQ tasks here disagree about presentation as "
        "well as about ranking. Where the bank was calibrated under different scoring, "
        "theta's absolute scale shifts; relative comparison across checkpoints is "
        "unaffected."
    ),
    "generative": (
        "Items are scored here by greedy sampled completion behind the benchmark's own "
        "fixed prompt, then {grading}. The shot count the bank was calibrated under is "
        "not recorded upstream, so agreement with a published ATLAS theta is "
        "unverified. Where the bank was calibrated under different scoring, theta's "
        "absolute scale shifts; relative comparison across checkpoints is unaffected."
    ),
}

#: Filled into the generative note when a session recorded no grader at all, which
#: means it administered nothing. Named rather than inlined so the empty-session
#: wording is not mistaken for a grader's own description.
NO_GRADER_NOTE = "graded by whichever grader each item's answer_type names"

#: Share of administered items that may come back ungradable before the report stops
#: describing the checkpoint and starts describing the harness.
#:
#: A handful of fabricated zeroes pulls theta down a little and is worth knowing about.
#: A fifth of them is a different claim entirely -- a missing ifbench, a bank vendored
#: against the wrong convention -- and the theta such a run reports is a floor produced
#: by the harness, with a standard error that shrinks exactly as it would on a real weak
#: checkpoint. One fifth rather than a half because that failure mode is rarely partial:
#: whatever breaks grading usually breaks it for every item, so a rate this high already
#: means something systematic, and the cost of saying so on a run that was merely
#: unlucky is a field in a report.
UNGRADABLE_ALERT_RATE = 0.2


def _scoring_note(
    modality: str,
    responses: Sequence[ItemResponse],
    *,
    score_normalization: str = "",
    prompt_style: str = "",
) -> str:
    """Return the scoring note for a finished session, naming the rule it was decided by.

    The two modalities fill their template from different places, because the varying
    part sits in different places. A generative bank's grader is a property of the items,
    so the note is completed from the grader names the responses carry; several would
    mean a bank mixing conventions inside one calibrated scale, which vendoring does not
    produce, and the note lists them all rather than picking one. An MCQ bank's
    normalization and shot count are properties of the resolved settings and leave no
    trace on a response, so both are passed in -- from the same resolution the scorer was
    built from, never from a second reading of ``config.yaml``, since a report that
    described a rule other than the one that ranked the choices is the failure this
    guards against.
    """
    if modality == grading.MCQ:
        note = SCORING_NOTES[modality].format(
            shots=inference.fewshot_count(prompt_style),
            scoring=inference.normalization_note(score_normalization),
        )
        detail = inference.fewshot_note(prompt_style)
        return f"{note} {detail}" if detail else note
    if modality != grading.GENERATIVE:
        return SCORING_NOTES[modality]
    used = sorted({str(r.metadata.get("grader")) for r in responses if r.metadata.get("grader")})
    clauses = [generative.grader_note(name) for name in used] or [NO_GRADER_NOTE]
    return SCORING_NOTES[modality].format(grading=", and ".join(clauses))


def dataset_min_items(config: Mapping[str, Any], dataset: str) -> int:
    """Return the item floor ``config`` pins for ``dataset``.

    Per dataset for the same reason ``prompt_style`` is, and resolved through the same
    :func:`~diagnostics.mcq_cat.common.grading.resolve_dataset_overrides` fold, so a
    dataset with no entry keeps the shared value and every bank vendored before this
    existed stops exactly where it always did.

    What makes a shared floor wrong is that it is a claim about the bank's
    discriminations rather than about the harness. Eight items reach SE <= 0.3 on a bank
    whose median ``a`` is around 1.4, and on BBH's -- median 3.99, inflated by a
    unidimensional fit over 24 unrelated subtasks -- the posterior collapses immediately
    and the session stops before it has touched a third of the suite. Raising the floor
    is not a fix for that: Fisher selection still takes whatever is most informative and
    may draw heavily from a few subtasks, and BBH's theta error does not fall with more
    items in any case. What it buys is that the shortfall is no longer arithmetic.

    Deliberately not part of the recorded scoring convention. ``convention.py`` draws
    that line at whether a setting changes an individual item's outcome, and this one
    changes only how many items there are, so a bank vendored under one floor and run
    under another is comparable in everything except test length -- which the report
    already carries.

    Raises:
        ValueError: If a dataset's entry names anything but :data:`PER_DATASET_CAT_KEYS`.
            A misspelling would otherwise leave the shared floor in place under a name
            that reads as though it had been applied.
    """
    block = {
        "min_items": config.get("min_items", DEFAULT_MIN_ITEMS),
        grading.PER_DATASET_KEY: config.get(grading.PER_DATASET_KEY) or {},
    }
    resolved = grading.resolve_dataset_overrides(block, dataset=dataset, block=CAT_BLOCK)
    unknown = sorted(set(resolved) - set(PER_DATASET_CAT_KEYS))
    if unknown:
        raise ValueError(
            f"Unknown per-dataset {CAT_BLOCK} settings {unknown} for {dataset!r}. Only "
            f"{', '.join(PER_DATASET_CAT_KEYS)} may be set per dataset; se_threshold and "
            f"max_items are run-level and have CLI flags."
        )
    return int(resolved["min_items"])


class UniMcqStyle(CatStyle):
    """Unidimensional (2PL/3PL) Fisher-information CAT over a calibrated bank."""

    def __init__(self) -> None:
        config = convention.load_config()
        self._config: dict[str, Any] = config
        self.se_threshold: float = float(config.get("se_threshold", DEFAULT_SE_THRESHOLD))
        self.max_items: int = int(config.get("max_items", DEFAULT_MAX_ITEMS))

        # The shared floor, which is what applies until a bank is resolved and says
        # otherwise. Settled per dataset in _adopt rather than re-derived on each
        # stopping-rule call, so a session cannot change its own test length halfway.
        self.min_items: int = dataset_min_items(config, "")

        # Passed through to GenerationConfig only for a generative bank, and to
        # InferenceConfig only for an MCQ one. Left as raw mappings so config.yaml has
        # exactly one schema to satisfy per modality, the dataclass's; common/grading.py
        # rejects any key that is not a field of it.
        self.generation_settings: dict[str, Any] = dict(config.get("generative") or {})
        self.mcq_settings: dict[str, Any] = dict(config.get("mcq") or {})

        self._resolved: resolve_mod.ResolvedBank | None = None
        self._items: dict[str, BenchmarkItem] = {}
        self._order: list[str] = []
        self._index: dict[str, int] = {}
        self._a: np.ndarray | None = None
        self._b: np.ndarray | None = None
        self._c: np.ndarray | None = None

    # -- loading -----------------------------------------------------------------

    def download_benchmark(self, benchmark: str, *, dest: Path | None = None) -> BenchmarkBank:
        """Resolve ``benchmark`` to a vendored bank and load its items.

        Nothing is downloaded: resolution is a local lookup and the items were
        enumerated once at vendoring time. ``dest`` is unused and accepted only to
        satisfy the interface.

        Raises:
            resolve.DatasetNotAvailable: If the dataset is unsupported or unvendored.
                Raised here, before the engine starts, and before the runner has spent
                anything on the checkpoint.
        """
        resolved = self._adopt(resolve_mod.resolve(benchmark))
        bank = load_items_from_jsonl(resolved.items_path, name=resolved.spec.name)
        self._items = {item.item_id: item for item in bank.items}
        return bank

    def _adopt(self, resolved: resolve_mod.ResolvedBank) -> resolve_mod.ResolvedBank:
        """Take ``resolved`` as this session's bank and settle what depends on which it is.

        One place rather than two assignments, because ``min_items`` is now a property of
        the dataset and both entry points that can resolve a bank -- the ordinary one and
        ``load_irt_params`` reached with a bare dataset name -- have to leave the style in
        the same state. A run that resolved through the second and kept the shared floor
        would stop early on a bank whose spec asks for more, and nothing in its report
        would distinguish it from one that stopped early on precision.
        """
        self._resolved = resolved
        self.min_items = dataset_min_items(self._config, resolved.spec.name)
        return resolved

    def load_irt_params(self, source: str | Path) -> IRTBank:
        """Load the bank's item parameters from ``source``.

        ``source`` is either a location -- a local path or an ``s3://`` URI -- or the
        dataset name the runner passes through from ``--benchmark``, which resolves to
        the vendored ``params.json``.
        """
        params_path = self._params_source(source)
        irt_bank = load_irt_params(params_path)
        if irt_bank.dimensions != 1:
            raise ValueError(
                f"uni_mcq is a unidimensional style but {params_path} declares "
                f"{irt_bank.dimensions} dimensions. Use a multidimensional style for a "
                f"MIRT bank."
            )
        if self._resolved is not None:
            resolve_mod.check_fit_family(self._resolved, None)
        self._build_arrays(irt_bank)
        return irt_bank

    def _params_source(self, source: str | Path) -> str | Path:
        """Return the parameter file ``source`` names, adopting the bank if it named one.

        A bank already resolved by :meth:`download_benchmark` is reused only when
        ``source`` is that bank's name. The reuse exists so the ordinary call, where the
        runner passes through the same name it just downloaded, does not resolve twice.
        Reusing it for *any* source that is not a readable local file made every other
        value of ``--irt-params`` silently mean the vendored parameters instead: a
        mistyped path, and an ``s3://`` URI, which the loader beneath this has always
        read and which this branch never reached. Neither failed -- the run finished and
        reported an ability estimated against parameters nobody asked for, which is the
        one outcome worth spending a branch to prevent. A name that is neither a file nor
        the resolved bank now goes back through the allowlist and fails there.
        """
        location = str(source)
        if s3_io.is_s3_uri(location):
            return location
        candidate = Path(source)
        if candidate.is_file():
            return candidate
        resolved = self._resolved
        if resolved is None or resolved.spec.name != location:
            resolved = resolve_mod.resolve(location)
        return self._adopt(resolved).params_path

    def _build_arrays(self, irt_bank: IRTBank) -> None:
        """Cache a canonical item order and its ``(a, b, c)`` arrays.

        The CAT works over parallel numpy arrays, so item ids need a stable index.
        Only items present in *both* the item file and the parameter file are usable;
        vendoring already guarantees the two agree, and this re-checks it cheaply.
        """
        order = [item_id for item_id in self._items if item_id in irt_bank.params]
        if not order:
            raise ValueError(
                "No item has both a stem and IRT parameters. items.jsonl and "
                "params.json disagree; re-run vendor_bank.py for this dataset."
            )
        missing = len(self._items) - len(order)
        if missing:
            log.warning("%d items have no IRT parameters and cannot be selected.", missing)

        self._order = order
        self._index = {item_id: i for i, item_id in enumerate(order)}
        params = [irt_bank.get(item_id) for item_id in order]
        self._a = np.asarray([float(p.discrimination) for p in params], dtype=float)
        self._b = np.asarray([float(p.difficulty) for p in params], dtype=float)
        self._c = np.asarray([float(p.guessing) for p in params], dtype=float)
        log.info("Bank ready: %d selectable items", len(order))

    # -- grading -----------------------------------------------------------------

    def grading_request(self) -> grading.GradingRequest:
        """Declare how the resolved bank must be graded.

        The runner consults this to build the scoring model. It is the style's whole
        contribution to grading: which modality the bank is, and what settings the config
        pins for it. Turning that into a scorer is ``common/grading.py``'s job, so adding
        a modality never touches this file.

        For an MCQ bank this is also the last point before a checkpoint is staged at
        which the resolved prompt style can be checked against the items it will be
        applied to, so the cloze guard runs here -- beside ``check_bank_modality``, and
        for the same reason it exists.

        Raises:
            RuntimeError: If no bank has been resolved yet. Modality is a property of
                the dataset, not of the style, so there is nothing to answer with.
            ValueError: If the bank's items and its configured prompt style disagree.
        """
        resolved = self._resolved
        if resolved is None:
            raise RuntimeError(
                "No bank has been resolved, so the grading modality is unknown. Call "
                "download_benchmark before grading_request."
            )
        modality = self.bank_modality()
        if modality == grading.MCQ:
            inference.check_prompt_style_fits(self.mcq_prompt_style(), list(self._items.values()))
            # Resolved for its side effect: an unknown normalization would otherwise
            # first be looked up inside score_items, after the checkpoint had been
            # staged and the GPU booted. Every other MCQ setting is settled before that
            # point and this one is no cheaper to discover late.
            inference.get_mcq_score_normalization(self.mcq_score_normalization())
        return convention.grading_request(
            resolved.spec,
            modality=modality,
            mcq=self.mcq_settings,
            generation=self.generation_settings,
        )

    def check_scoring_convention(
        self, request: grading.GradingRequest, settings: grading.GradingSettings
    ) -> None:
        """Refuse to run if the resolved settings are not what the bank records.

        The runner calls this beside ``check_bank_modality``, before the checkpoint is
        fetched, and it is the same class of guard: neither failure is a crash the
        engine would hit later, both are completed runs that publish a confident theta
        computed from grading that never had a chance of matching the bank's scale.

        ``settings`` is what the scorer will be built from, folded by the runner rather
        than re-read here. Re-reading ``config.yaml`` at this point would compare the
        style against itself and pass whatever the caller had actually configured.

        Raises:
            RuntimeError: If no bank has been resolved yet.
            resolve.DatasetNotAvailable: If the bank records no convention, or records
                one this run disagrees with.
        """
        resolved = self._resolved
        if resolved is None:
            raise RuntimeError(
                "No bank has been resolved, so there is no recorded scoring convention "
                "to check. Call download_benchmark before check_scoring_convention."
            )
        convention.check_runtime_convention(resolved, settings, modality=request.modality)

    def mcq_prompt_style(self) -> str:
        """Return the prompt style the resolved MCQ bank will be scored behind."""
        return str(self._mcq_setting("prompt_style", inference.DEFAULT_PROMPT_STYLE))

    def mcq_score_normalization(self) -> str:
        """Return the rule the resolved MCQ bank's choices will be ranked by.

        The prompt's counterpart, and reported beside it for the same reason: an MCQ
        theta is a statement about one presentation *and* one comparison rule, and the
        two banks here disagree about the second as well as the first.
        """
        return str(self._mcq_setting("score_normalization", inference.DEFAULT_SCORE_NORMALIZATION))

    def _mcq_setting(self, key: str, default: str) -> Any:
        """Return one resolved MCQ setting for the resolved bank.

        Resolved through ``common/grading.py`` rather than read straight out of the
        config, so the report names the settings the scorer was actually built with. A
        second reading of the same map here would be free to drift from it, and the
        thing that drifted would be the record of how theta was produced.
        """
        resolved = self._resolved
        if resolved is None:
            raise RuntimeError(
                f"No bank has been resolved, so there is no {key} to report. "
                f"Call download_benchmark first."
            )
        overrides = grading.resolve_dataset_overrides(
            self.mcq_settings, dataset=resolved.spec.name, block="MCQ"
        )
        return overrides.get(key, default)

    def bank_modality(self) -> str:
        """Return the resolved bank's modality, refusing to guess when records disagree.

        ``datasets.py`` records what the dataset is expected to be and the manifest
        records what was actually vendored. Vendoring writes both from the same run, so
        a disagreement means one of them is stale, and either choice would grade real
        items under an assumption nobody checked. Banks vendored before modality
        existed have no manifest key and fall back to the allowlist, which defaults to
        MCQ -- correct, since every such bank is one.
        """
        resolved = self._resolved
        if resolved is None:
            raise RuntimeError(
                "No bank has been resolved, so the grading modality is unknown. Call "
                "download_benchmark before bank_modality."
            )

        declared = resolved.spec.modality
        vendored = resolved.manifest.get("modality")
        if vendored is not None and vendored != declared:
            raise resolve_mod.DatasetNotAvailable(
                f"{resolved.spec.name} is listed as {declared!r} in datasets.py but the "
                f"manifest at {resolved.root / resolve_mod.MANIFEST_NAME} says "
                f"{vendored!r}. One of the two is stale; re-vendor the bank rather than "
                f"letting a guess pick the grader."
            )

        modality = vendored or declared
        if modality not in MODALITIES:
            raise resolve_mod.DatasetNotAvailable(
                f"{resolved.spec.name} declares an unknown modality {modality!r}. "
                f"Known modalities: {', '.join(MODALITIES)}."
            )
        return str(modality)

    # -- the CAT loop ------------------------------------------------------------

    def score(self, model: ScoringModel, items: Sequence[BenchmarkItem]) -> list[ItemResponse]:
        """Grade ``items`` with whichever scorer the runner built for this bank's modality."""
        return model.score_items(items)

    def estimate_ability(
        self,
        bank: IRTBank,
        responses: Sequence[ItemResponse],
        *,
        previous: AbilityEstimate | None = None,
    ) -> AbilityEstimate:
        """Re-estimate theta and its standard error by EAP over administered items.

        EAP is memoryless: it conditions on the full response pattern each time, so
        ``previous`` is not needed and is accepted only to satisfy the interface.
        """
        if not responses:
            return AbilityEstimate(theta=0.0, standard_error=1.0)

        a, b, c = self._arrays()
        selected = np.asarray([self._index[r.item_id] for r in responses], dtype=int)
        observed = np.asarray([1.0 if r.correct else 0.0 for r in responses], dtype=float)
        theta, se = eap_theta_se(observed, a[selected], b[selected], c[selected])
        return AbilityEstimate(theta=theta, standard_error=se)

    def select_next_item(self, bank: IRTBank, state: CATState) -> str | None:
        """Return the unused item with maximum Fisher information at the current theta.

        Returns ``None`` only when the bank is exhausted; the engine enforces
        ``max_items`` and calls :meth:`stopping_rule` after each response.
        """
        a, b, c = self._arrays()
        used = state.administered_ids
        if len(used) >= len(self._order):
            return None

        theta = float(state.ability.theta) if state.ability is not None else 0.0
        info = fisher_info(theta, a, b, c)
        if used:
            mask = np.asarray([self._index[item_id] for item_id in used], dtype=int)
            info = info.copy()
            info[mask] = -1.0
        return self._order[int(np.argmax(info))]

    def stopping_rule(self, state: CATState) -> bool:
        """Stop once enough items are in and the standard error is small enough.

        Both conditions must hold. Since the session starts at theta 0 with SE 1, the
        ``min_items`` floor always binds before the precision test can pass.
        """
        if state.ability is None:
            return False
        threshold = state.se_threshold if state.se_threshold is not None else self.se_threshold
        return state.step >= self.min_items and float(state.ability.standard_error) <= threshold

    # -- reporting ---------------------------------------------------------------

    def report(self, state: CATState) -> CATReport:
        """Assemble the report, including p-IRT predicted accuracy and provenance.

        A bank whose spec carries a :attr:`~.datasets.DatasetSpec.report_caveat` puts it
        here as well, under ``bank_caveat``. Everything else in this dict is a
        measurement, and the one thing a reader cannot get from a measurement is which of
        them this bank supports: BBH's predicted accuracy is usable and its theta is not,
        and a report that carried both as bare numbers would be read as offering both.
        Only a bank that has one gets the key, so no other report changes shape.
        """
        a, b, c = self._arrays()
        ability = state.ability or AbilityEstimate(theta=0.0, standard_error=1.0)
        theta = float(ability.theta)

        order = [self._index[r.item_id] for r in state.administered]
        scores = [1 if r.correct else 0 for r in state.administered]

        predicted = pirt_accuracy(a, b, c, order, scores, theta)
        observed = observed_accuracy(scores)

        threshold = state.se_threshold if state.se_threshold is not None else self.se_threshold
        max_items = state.max_items if state.max_items is not None else self.max_items
        stop_reason = self._stop_reason(state, threshold, max_items)

        # The caller's --max-items always wins, because an explicit flag should. But
        # ability estimates are not comparable across different caps, so a run that did
        # not use the pinned value says so in its own report rather than looking
        # interchangeable with one that did. The pin matches the runner and
        # run_checkpoint_diag.sh defaults, so an ordinary invocation agrees already.
        if max_items != self.max_items:
            log.warning(
                "Ran with max_items=%d but this style pins %d. Estimates are not "
                "comparable across caps; set MAX_ITEMS=%d for comparability.",
                max_items,
                self.max_items,
                self.max_items,
            )

        metadata: dict[str, Any] = {
            "fit_family": self._resolved.fit_family if self._resolved else "unknown",
            "theta": theta,
            "standard_error": float(ability.standard_error),
            "pirt_accuracy": predicted,
            "observed_accuracy": observed,
            "pirt_accuracy_denominator": (
                f"the {len(self._order)}-item calibrated bank, not the full evaluation "
                f"split -- predicted accuracy is over the calibrated subset"
            ),
            "n_items_administered": state.step,
            "bank_size": len(self._order),
            "stop_reason": stop_reason,
            "ungradable": self._ungradable_block(state),
            "selected_item_ids": [r.item_id for r in state.administered],
            "cat_settings": {
                "se_threshold": threshold,
                "min_items": self.min_items,
                "max_items": max_items,
                "max_items_pinned_by_style": self.max_items,
                "max_items_is_pinned_value": max_items == self.max_items,
            },
        }
        if self._resolved is not None:
            modality = self.bank_modality()
            is_mcq = modality == grading.MCQ
            normalization = self.mcq_score_normalization() if is_mcq else ""
            prompt_style = self.mcq_prompt_style() if is_mcq else ""
            metadata["bank_provenance"] = self._resolved.provenance()
            metadata["modality"] = modality
            metadata["scoring_note"] = _scoring_note(
                modality,
                state.administered,
                score_normalization=normalization,
                prompt_style=prompt_style,
            )
            if is_mcq:
                metadata["prompt_style"] = prompt_style
                metadata["score_normalization"] = normalization
            caveat = self._resolved.spec.report_caveat
            if caveat:
                metadata["bank_caveat"] = caveat

        return CATReport(
            cat_style=type(self).name,
            benchmark=state.benchmark,
            ability=ability,
            num_items_administered=state.step,
            responses=tuple(state.administered),
            metadata=metadata,
        )

    def _ungradable_block(self, state: CATState) -> dict[str, Any]:
        """Account for the administered items that produced no outcome of their own.

        An ungradable item is scored 0 and the session continues, which is what
        Research's online CAT does and is the right trade when the alternative is
        losing a whole checkpoint's diagnostic to one malformed row. The cost is that
        EAP is handed a wrong answer it will read as evidence about the model, so the
        run has to say how much of its response pattern was fabricated. Counted off the
        responses rather than tracked alongside them, for the reason
        :func:`_scoring_note` reads graders off them: a tally kept in parallel is free
        to disagree with the pattern theta was actually computed from.

        Above :data:`UNGRADABLE_ALERT_RATE` the block carries an ``alert``. The
        distinction it draws is the one a reader cannot make from theta: a systematic
        grading failure and a genuinely weak checkpoint both produce a low estimate
        with a healthy standard error, and only one of them is about the model. Both
        the count and the ids are recorded either way, so a run that stays under the
        threshold can still be audited item by item.
        """
        ungradable = [
            response
            for response in state.administered
            if response.metadata.get(generative.UNGRADABLE_KEY)
        ]
        rate = len(ungradable) / state.step if state.step else 0.0
        block: dict[str, Any] = {
            "count": len(ungradable),
            "rate": rate,
            "item_ids": [response.item_id for response in ungradable],
            "reasons": sorted(
                {
                    str(response.metadata.get(generative.UNGRADABLE_REASON_KEY))
                    for response in ungradable
                }
            ),
        }
        if not ungradable:
            return block

        if rate < UNGRADABLE_ALERT_RATE:
            log.warning(
                "%d of %d administered items could not be graded and were scored 0. "
                "Theta is that much lower than the response pattern warrants.",
                len(ungradable),
                state.step,
            )
            return block

        block["alert"] = (
            f"{len(ungradable)} of {state.step} administered items ({rate:.0%}) could "
            f"not be graded and were scored 0, so most of this response pattern is the "
            f"harness rather than the checkpoint. Read theta as a floor produced by "
            f"grading failing, not as a measurement: a missing dependency, a bank "
            f"vendored against a different convention or a prompt the verifiers were "
            f"not told about all look like this, and all of them shrink the standard "
            f"error exactly as a real weak checkpoint would."
        )
        log.error(
            "%d of %d administered items (%.0f%%) were ungradable. This run measured "
            "the harness, not the checkpoint; see the report's ungradable.alert.",
            len(ungradable),
            state.step,
            rate * 100,
        )
        return block

    def _stop_reason(self, state: CATState, threshold: float, max_items: int) -> str:
        """Classify why the session ended, for the report."""
        if state.step >= len(self._order):
            return "bank_exhausted"
        if state.step >= max_items:
            return "max_items_reached"
        if self.stopping_rule(state):
            return "precision_reached"
        return "stopped_early"

    def _arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the cached ``(a, b, c)`` arrays, or explain what was skipped."""
        if self._a is None or self._b is None or self._c is None:
            raise RuntimeError(
                "IRT parameters have not been loaded. Call load_irt_params before "
                "running the CAT loop."
            )
        return self._a, self._b, self._c
