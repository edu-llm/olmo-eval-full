"""The supported-dataset allowlist for the ``uni_mcq`` style.

A dataset is supported only when all four conditions hold:

1. Its calibrated bank came from **Route B** (Open LLM Leaderboard v2 harvest, fit
   locally under ``AdaptiveTesting/Experiments/openlm_*``) or **Route C** (published
   by the ATLAS release, vendored under ``AdaptiveTesting/Inputs/ATLAS/``). Route A,
   our own inference sweep over a 63-100 model roster, is excluded: it did serialize
   banks for four benchmarks under ``eduLLM-Evals/runs/mcq_irt*/``, but they are 2PL
   fits where this pipeline standardizes on 3PL, and their theta is anchored to that
   83-model pool rather than to the ATLAS population, so the two scales are not
   comparable.
2. A **grader exists for how the benchmark is answered**, recorded as
   :attr:`DatasetSpec.modality`. Multiple choice is graded by ranking continuation
   log-likelihoods; generative is graded by sampling a completion and matching an
   extracted answer. Anything needing a third grading convention is out.
3. A **task definition exists on this branch** under ``src/olmo_eval/evals/tasks``,
   so items can be enumerated at vendoring time.
4. The bank is **usable** -- it linked successfully and has a bridge from bank index to
   an id the task can be keyed by. For most datasets that id is the task's
   ``metadata["id"]``; for a bank spanning several tasks it is the composite
   ``<subtask>|<position>``, rebuilt by counting each subtask's enumeration, which is
   why those datasets carry the extra per-subtask guards.

Meeting all four is necessary and is not sufficient, which is what :attr:`
DatasetSpec.blocked` is for. Condition 4 asks whether the bridge *links*, and a bridge
that names the right items in the wrong order links perfectly: every row joins, overlap
reads 100%, the positional floor is cleared, the CAT converges and the standard error
collapses on schedule. Nothing in the pipeline can tell that apart from a correct join,
because the difference is not visible in any quantity it computes. Deciding it takes an
external statistic -- item p-value against the bank's implied ``b = -d/a1``, strongly
negative when the join is right and at zero when it is scrambled -- and on that test
three of the four ATLAS banks here once failed or could not be tested at all.

All three have since been repaired; see :data:`_REBUILT_BRIDGE`. What is worth carrying
forward is that the repair was accepted on the same external statistic that condemned
them and not on the guards, which passed throughout and were passing while the banks
were wrong. The evidence differs per dataset and each note says which it has, because
"the bridge was rebuilt" is not by itself a reason to believe a bank.

A sound join is in turn not the same as a useful measurement, which is what :attr:`
DatasetSpec.report_caveat` is for. bbh joins correctly, converges, and reports a theta
whose error does not fall however many items are administered, because a unidimensional
model over 24 unrelated subtasks has nothing to converge to. Its predicted accuracy is
worth reading and its theta is not, and no field of a report distinguishes the two, so
the distinction is carried into every one of them rather than left here.

Deliberate exclusions, recorded in :data:`EXCLUDED` so they are not re-litigated.

This module is pure data plus lookup helpers. It performs no I/O.

Where the harness's presentation or grading differs from the Research/olmo-eval
implementation of the same benchmark, the difference is recorded per dataset in
``calibrated_datasets/DEVIATIONS.md``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

#: Fit families this style can consume. 2PL is the ``c = 0`` case of 3PL, so one
#: implementation serves both; see ``irt.py`` for the reduction.
FIT_FAMILIES = ("2pl", "3pl")

#: The bridge conventions vendoring knows how to read, mapped to the column holding
#: the joinable key.
#:
#: ``atlas`` is the ATLAS release's ``atlas_idx_to_question_id.csv``, whose
#: ``question_id`` is the task's own ``metadata["id"]``. ``item_id_map`` is the
#: four-column map the local Route B fits write, whose ``question_id`` is positional
#: *within* a subtask and whose joinable key is therefore the composite ``item_id``;
#: gpqa and bbh still take it. ``content_hash`` is the kind rebuilt from the
#: leaderboard's own per-example records, keyed by :func:`content_item_id`.
#:
#: Seven of the nine banks are now content-keyed and the two that are not are a
#: deliberate stop rather than work outstanding. GPQA's three subsets are nested quality
#: filters over one pool, so 448 of its 546 distinct questions were calibrated under two
#: or three subtask labels, and BBH ships four literally duplicated items across two
#: subtasks, so on both banks a content hash names more than one calibrated row and
#: vendoring would drop the collisions as ambiguous. Their composite keys are nonetheless
#: no longer an assumption: the same recovery that re-keyed the other two confirmed,
#: against the leaderboard's own per-example records, that every one of gpqa's 1,192 and
#: bbh's 5,761 ``doc_id`` values names the question a fresh enumeration assumes. GPQA's
#: repeats are then removed by :attr:`DatasetSpec.duplicate_question_precedence`, which
#: is a separate decision from how the bridge keys them. See each spec's notes.
BRIDGE_KEY_COLUMNS = {
    "atlas": "question_id",
    "item_id_map": "item_id",
    "content_hash": "item_id",
}

#: A bridge whose joinable key is derived from the item's text rather than read off it.
#:
#: Used where no field of the dataset is both present and unique. HellaSwag is the case
#: that forced it: the validation split concatenates a ``zeroshot`` and an ``indomain``
#: sub-split whose native ``ind`` values each restart from zero, so 10,042 rows carry
#: only 9,609 distinct values and ``ind == 180`` names both "Sharpening knives" at
#: position 8 and "How to become a sports announcer" at position 3260. The other two
#: rebuilt banks have a unique field -- a bare split position -- and deliberately do not
#: use it, because a position is exactly the key whose silent drift this whole family of
#: bridges was rebuilt to escape.
CONTENT_HASH = "content_hash"

#: Hex digits of the SHA-256 kept as an item id.
#:
#: Sixty-four bits. The largest split keyed this way holds 10,042 items, which puts the
#: chance of any collision near 3e-12, while a full digest would quadruple the size of
#: three committed bridges and every ``params.json`` and ``items.jsonl`` built from
#: them. A collision would not corrupt silently in any case: two items sharing an id
#: are dropped as ambiguous by :func:`~.scripts.vendor_bank.drop_ambiguous_ids`.
CONTENT_HASH_DIGITS = 16

#: How a dataset's items are answered, and therefore how they must be graded.
#: ``mcq`` items carry choices and are graded by continuation log-likelihood;
#: ``generative`` items have no choices and are graded by sampling a completion and
#: matching an extracted answer. The IRT layer is identical either way -- it only ever
#: sees a binary correct/incorrect -- so this selects a grader, nothing more.
MODALITIES = ("mcq", "generative")

#: Written into a calibration field whose value is not recorded anywhere upstream.
#:
#: An explicit string rather than ``None`` or an omitted key, because those two read as
#: "not applicable" and this means "nobody knows". The difference decides what a reader
#: is entitled to conclude from a theta: an unrecorded prompt means agreement with a
#: published ability score is unverified, while a recorded one that we match means it is
#: expected. A plausible reconstruction in this slot would be worse than either, since
#: the manifest is the only record of provenance and whatever it says will be believed.
UNRECORDED = "unrecorded"


@dataclass(frozen=True, slots=True)
class CalibrationConvention:
    """How a bank's responses were prompted and scored when its parameters were fit.

    Deliberately separate from the convention a *run* uses, which lives in the style's
    ``config.yaml`` and is rebuilt into every manifest by
    :mod:`~diagnostics.mcq_cat.styles.uni_mcq.convention`. They are different claims and
    conflating them is what makes a silent scale shift possible: the run-time side is
    what this harness does and is checked at startup, while this side is what the
    difficulties EAP treats as fixed were estimated against, and for most of these banks
    it was never written down. Only where the two are known to *agree* is a theta
    comparable with a published one.

    Attributes:
        prompt_style: The presentation the calibrated responses were produced behind.
        num_fewshot: The shot count they were produced under. An int where a source
            records one, otherwise :data:`UNRECORDED`.
        metric: What counted as a correct response during calibration.
        note: Where the recorded facts come from, and what is inferred rather than
            stated. Kept short; the long form is in :attr:`DatasetSpec.notes`.
    """

    prompt_style: str = UNRECORDED
    num_fewshot: int | str = UNRECORDED
    metric: str = UNRECORDED
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        """The manifest form, with every unknown spelled out rather than omitted."""
        return asdict(self)


#: The default for a dataset whose calibration convention nothing upstream describes.
UNRECORDED_CALIBRATION = CalibrationConvention()


def content_item_id(question: str, choices: Iterable[str] = ()) -> str:
    """Return the :data:`CONTENT_HASH` joinable id for one item's presented text.

    The whole point of this key is that both sides of the join compute it from the
    item itself, so it has to be defined once and imported. ``build_leaderboard_bridge``
    calls it to decide what a bridge row names, and ``vendor_bank`` calls it to decide
    what a task instance answers to; two implementations that agreed today would be free
    to drift apart, and the failure would be a bridge that joins nothing rather than one
    that joins wrongly -- loud, but only after a bank had been rebuilt around it.

    Question and choices are hashed together and each field is terminated, not
    concatenated. HellaSwag needs both: 22 of its validation contexts recur with
    different endings, so the stem alone is not unique. Termination is what stops
    ``("ab", "c")`` and ``("a", "bc")`` from colliding, which on a benchmark whose
    choices are sentence fragments of the stem is not a hypothetical.

    The text is the *task's*, after whatever preprocessing the task applies. That makes
    the id a statement about the item this harness would administer rather than about
    the row upstream stored, which is the direction that matters: an edit to the task's
    presentation changes the id, the join drops those rows, and the overlap floor turns
    it into an abort instead of a bank of items scored behind a prompt they were not
    matched under.
    """
    digest = hashlib.sha256()
    digest.update(question.encode("utf-8"))
    digest.update(b"\x00")
    for choice in choices:
        digest.update(choice.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()[:CONTENT_HASH_DIGITS]


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """Provenance and join details for one supported dataset.

    Attributes:
        name: The dataset name a user passes, and the directory under
            ``calibrated_datasets/``.
        task: The ``olmo_eval`` task spec used to enumerate items at vendoring time.
            Empty for a dataset whose bank spans several tasks, which lists them in
            :attr:`subtasks` instead; read :attr:`task_names` rather than this field.
        subtasks: ``(subtask label, task spec)`` pairs for a bank calibrated across
            several tasks, ordered as the bridge names them. Only ``item_id_map``
            bridges need this: their joinable id is the composite
            ``<subtask>|<position>``, so the label has to match the bridge's
            ``subtask`` column exactly or the rebuilt keys join nothing. Empty for
            the single-task datasets, whose specs are unchanged.
        route: ``"B"`` (OpenLM v2 harvest) or ``"C"`` (ATLAS published).
        bank_dir: Directory on the source ref holding the calibration output.
        bridge_path: Path mapping bank index to a joinable id. On the source ref
            unless :attr:`bridge_in_repo` says otherwise.
        bridge_in_repo: True when :attr:`bridge_path` is relative to this checkout
            rather than to the source ref. Set for the bridges rebuilt from the
            leaderboard's own per-example records, which have no upstream to be read
            from: they are our artifacts, and the reason they are committed rather than
            regenerated is that regenerating one needs the Hub, an archived details
            file and the HuggingFace split all reachable at once, and a bank whose join
            can only be reproduced under those conditions is a bank whose join
            nobody will reproduce.
        bridge_kind: A key of :data:`BRIDGE_KEY_COLUMNS`. ``"atlas"`` for the
            two-or-three column ``atlas_idx_to_question_id.csv``; ``"item_id_map"``
            for the four-column map whose joinable key is the composite ``item_id``;
            :data:`CONTENT_HASH` for a bridge keyed by :func:`content_item_id`.
        fit_family: The family the parameters were **estimated under**. Authoritative
            at run time via the bank manifest, never overridden by a flag.
        expected_bank_rows: Calibrated rows upstream, used as a vendoring sanity check.
        positional_ids: True when the task's ``metadata["id"]`` is a position rather
            than a native dataset id. Read :attr:`needs_overlap_floor` rather than this
            field to decide whether the floor applies; this one is a fact about the id
            and is reported as such in the manifest.
        notes: Why this dataset behaves the way it does.
        modality: One of :data:`MODALITIES`. Selects the grading scheme; the CAT and
            IRT layers are unaffected because both schemes emit the same binary
            outcome.
        answer_type: For a generative dataset, which entry of
            ``common/generative.py``'s ``ANSWER_GRADERS`` decides its items, stamped
            into every vendored record. Declared per dataset rather than sniffed off
            each gold answer because it is a property of the benchmark: MATH golds are
            LaTeX and a gold that happens to read ``7`` is still graded symbolically,
            and IFEval has no gold at all. Ignored for an MCQ dataset.
        blocked: When set, the reason this dataset cannot be run today. The dataset
            stays listed because the bank exists upstream and the work is understood;
            the resolver surfaces this string instead of a bare "bank missing".

            Nothing sets it today, and the field stays because both situations that
            once did recur. One is a bank that cannot be *vendored* -- gpqa's items
            were behind a HuggingFace token scope until that scope was granted -- where
            nothing is on disk and any ladder order would fail anyway. The other is a
            bank already vendored whose join was later shown wrong, which is why
            :func:`~.resolve.resolve` consults this field before it looks for
            artifacts: ``params.json`` and ``items.jsonl`` are present and load
            cleanly, so a ladder that checked the disk first would run the dataset and
            report a confident theta computed from parameters attributed to the wrong
            questions. The three ATLAS banks were in exactly that state.
        calibration: What is known about the convention the parameters were estimated
            under. Copied into the bank's manifest at vendoring time and never checked
            against a run, because it describes the past rather than the present; the
            run-time half of that manifest block is the one with a startup guard.
        frozen_prompt: True when an item's stem is the whole prompt the task renders for
            it rather than the bare question, so run time appends the choice and adds
            nothing. Set for a benchmark whose presentation this style cannot express --
            BBH's 3-shot block is per *subtask*, and ``config.yaml`` is per dataset -- and
            it settles the prompt at vendoring instead. gpqa made the same trade with its
            shuffled choice block while it was generative and no longer needs to, since a
            choice list carries the ordering without freezing the prompt around it. The
            cost is that a re-render of the
            task no longer reaches the bank; the gain is that a run cannot disagree with
            the bank about what the exemplars were.
        report_caveat: What a reader of ``cat_report.json`` has to know before using the
            numbers beside it, or empty where the measurements speak for themselves.
            Reserved for the case where a bank supports one of its outputs and not the
            other, which nothing in a report reveals: predicted accuracy and theta are
            both printed as plain floats with a standard error, and a bank whose theta
            does not recover produces exactly the same shape as one whose does. Only bbh
            sets it today.
        choice_order_control: True when a committed
            ``bridges/<name>.choice_order.json`` records the option ordering this bank's
            ``gold_index`` indexes, and vendoring must refuse to write a choice list that
            departs from it. Set only for gpqa, which needs it because it is the one bank
            whose options were shuffled per question upstream *and* whose modality
            changed underneath that shuffle: its ordering was frozen inside the stems of
            a generative bank and had to be carried across rather than re-derived.

            An MCQ bank does not otherwise need one. The other five present their choices
            in the dataset's own order, so a fresh enumeration reproduces it or fails
            visibly, and a content-hashed id covers the texts as well. GPQA has neither
            property: its order comes from ``Random(f'{seed}:{index}')`` inside
            ``process_doc``, which would silently produce a different permutation if the
            seed or the enumeration index moved, and its composite key says nothing about
            content at all. A permutation there is invisible to every other guard --
            counts match, the join is total, the CAT converges -- so this is the only
            thing standing between a moved shuffle and a confident wrong theta.
        duplicate_question_precedence: Subtask labels ordered most-preferred first, for a
            bank whose subtasks are nested rather than disjoint and which therefore
            calibrated some questions more than once. Empty everywhere but gpqa, whose
            three subsets are quality filters over one pool of questions; see its notes
            and :func:`~.scripts.vendor_bank.drop_duplicate_questions`. Every subtask must
            be ranked, because an unranked one leaves a repeated question with no defined
            survivor.
    """

    name: str
    task: str
    route: str
    bank_dir: str
    bridge_path: str
    bridge_kind: str
    fit_family: str
    expected_bank_rows: int
    positional_ids: bool
    notes: str
    modality: str = "mcq"
    answer_type: str = "numeric"
    blocked: str | None = None
    bridge_in_repo: bool = False
    subtasks: tuple[tuple[str, str], ...] = ()
    calibration: CalibrationConvention = UNRECORDED_CALIBRATION
    frozen_prompt: bool = False
    report_caveat: str = ""
    choice_order_control: bool = False
    duplicate_question_precedence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Reject a spec whose declarations cannot all be honoured at vendoring time.

        Vendoring branches on which of task and subtasks is populated. A spec setting
        both would take one branch and quietly ignore the other declaration; a spec
        setting neither would ask the registry for the task named by the empty string
        and fail deep inside enumeration rather than here, where the mistake actually
        is.

        :attr:`frozen_prompt` fails more quietly still on a generative dataset, because
        that path already decides its own presentation in ``vendor_bank``'s
        ``presented_question`` and never consults the flag. The bank would be written
        with an unfrozen stem, the manifest would record the convention that path
        implies, every guard would pass, and the only sign would be the difference
        between the prompt the author asked for and the one the difficulties describe.

        :attr:`duplicate_question_precedence` is a total order over the subtasks or it is
        not an order at all. A label it omits has no rank, so a question repeated in that
        subtask would either survive alongside its twin or be dropped depending on which
        arm of the comparison ran first, and a label it names that no subtask declares is
        a spec that has gone stale against its own bridge.
        """
        if bool(self.task) == bool(self.subtasks):
            raise ValueError(
                f"{self.name}: set exactly one of task and subtasks. A single-task "
                f"dataset names its task; a bank spanning several names them in "
                f"subtasks and leaves task empty."
            )
        if self.choice_order_control and self.modality != "mcq":
            raise ValueError(
                f"{self.name}: choice_order_control governs the choice list an MCQ "
                f"record carries and this spec is {self.modality!r}, whose records carry "
                f"no choices at all. Vendoring would load the control, find nothing to "
                f"hold to it, and report a bank as order-checked that has no order in it."
            )
        if self.frozen_prompt and self.modality != "mcq":
            raise ValueError(
                f"{self.name}: frozen_prompt is an MCQ-only declaration and this spec is "
                f"{self.modality!r}. A generative bank's stem is settled by "
                f"vendor_bank.presented_question, which does not read this flag, so "
                f"setting it here would be ignored rather than applied."
            )
        ranked = tuple(sorted(self.duplicate_question_precedence))
        if ranked and ranked != tuple(sorted(label for label, _ in self.subtasks)):
            raise ValueError(
                f"{self.name}: duplicate_question_precedence "
                f"{list(self.duplicate_question_precedence)} must rank every subtask "
                f"exactly once, and this spec declares "
                f"{[label for label, _ in self.subtasks]}. A precedence that is not a "
                f"total order over the subtasks cannot decide which calibration of a "
                f"repeated question survives."
            )

    @property
    def params_csv(self) -> str:
        """Path to the linked item-parameter CSV on the source ref."""
        return f"{self.bank_dir}/irt_item_parameters_combined.csv"

    @property
    def is_multi_task(self) -> bool:
        """True when the bank spans several tasks and joins by composite id."""
        return bool(self.subtasks)

    @property
    def needs_overlap_floor(self) -> bool:
        """True when a shortfall in the task join is the only sign this bank would give.

        Two kinds of id have that property and they fail in opposite directions, which
        is why one flag covers both. A positional id names nothing about the item, so a
        shifted enumeration rekeys every row onto its neighbour's question and the join
        still succeeds -- the floor is the only quantity that moves. A
        :data:`CONTENT_HASH` id names the item and nothing else, so an edit to the task's
        presentation misses instead of mismatching, and the floor is what turns a quiet
        shortfall into an abort before a half-joined bank is written.

        A native dataset id needs neither. It survives reordering, and an upstream change
        that reassigned it would collapse the join to zero, which already aborts on the
        empty-intersection path.
        """
        return self.positional_ids or self.bridge_kind == CONTENT_HASH

    @property
    def task_names(self) -> tuple[str, ...]:
        """Every ``olmo_eval`` task this dataset enumerates, in bridge order."""
        if self.subtasks:
            return tuple(task for _, task in self.subtasks)
        return (self.task,)


_ATLAS = "AdaptiveTesting/Inputs/ATLAS"
_EXPERIMENTS = "AdaptiveTesting/Experiments"

#: Where the bridges rebuilt from leaderboard example order live, in this checkout.
_BRIDGES = "diagnostics/mcq_cat/styles/uni_mcq/bridges"

#: Shared by the three banks whose bridge was rebuilt from leaderboard example order.
#: Written once because the history is one history, and quoting it three times invites
#: the three copies to drift apart as each bank is re-examined.
#:
#: What was wrong. All three bridges were generated by
#: ``Inputs/ATLAS/scripts/build_atlas_idx_bridge.py``, which assumes ATLAS column *k* is
#: the *k*-th instance olmo-eval enumerates from the HuggingFace split. ARC is the
#: control that disproves it, being the only benchmark with both a bridge the release
#: shipped and the ability to reproduce what the generator would have written instead:
#: the two agree on 2 of 1,172 positions. Not an offset and not a different item set --
#: the same questions in a different order, because the release's indices are Open LLM
#: Leaderboard v1 example order. The generator's only validation was that the bank's
#: maximum index equals the split size, which a permutation satisfies. Nothing
#: downstream would have noticed either: a permuted bridge joins every row, reports 100%
#: overlap, selects items by a Fisher information computed from the wrong difficulties
#: and collapses the standard error on schedule, so theta is noise with a healthy
#: interval printed beside it.
#:
#: What replaced it. ``scripts/build_leaderboard_bridge.py`` reads the order off the v1
#: leaderboard's own per-model detail repos rather than inferring it, and the same run
#: on ARC reproduces the shipped bridge on 1,172 of 1,172 stems. The two positions where
#: the *ids* differ are ones where the shipped bridge is itself wrong: ATLAS resolved its
#: ids by matching stem text, ARC holds two questions whose stems recur under a second
#: id, and both collided onto the wrong twin, so the shipped file carries two ids twice
#: and never carries TIMSS_2003_8_pg47 or Mercury_7116183. A known-answer control that
#: reproduces the answer and finds two of its errors is the strongest evidence available
#: for the procedure, and it is the whole of the case for the one bank that cannot be
#: checked any other way.
#: BBH's 24 subtask labels, in the order its bridge names them, which is also the order
#: ``constants/bbh.BBH_CHOICES`` lists and the order the tasks are registered in.
#:
#: Written out rather than imported from the task module, which is the opposite of what
#: the count guards elsewhere do, because the two lists are the two sides of the join: if
#: a subtask were added to or renamed in the task module, importing it here would rebuild
#: the keys under the new name and report the change as a clean vendoring rather than as
#: the mismatch it is. The bridge is the authority for what this bank covers, and this
#: transcribes the bridge.
#:
#: The three subtasks absent from it are BIG-Bench Hard's genuinely generative ones --
#: dyck_languages, multistep_arithmetic_two, word_sorting -- which have no closed answer
#: set to rank. Open LLM Leaderboard v2 excludes them, so they were never harvested, never
#: calibrated, and are not registered as tasks either.
_BBH_SUBTASKS = (
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "disambiguation_qa",
    "formal_fallacies",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "logical_deduction_three_objects",
    "movie_recommendation",
    "navigate",
    "object_counting",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "sports_understanding",
    "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "web_of_lies",
)

_REBUILT_BRIDGE = (
    "Its bridge was rebuilt from Open LLM Leaderboard v1 example order by "
    "scripts/build_leaderboard_bridge.py and is committed under styles/uni_mcq/bridges/, "
    "replacing one generated by Inputs/ATLAS/scripts/build_atlas_idx_bridge.py on the "
    "assumption that ATLAS index k is the k-th instance olmo-eval enumerates. That "
    "assumption is disproved on ARC, the only benchmark carrying a bridge the release "
    "shipped: the generated and shipped bridges agree on 2 of 1,172 positions. The "
    "replacement is validated on the same benchmark, reproducing the shipped bridge on "
    "1,172 of 1,172 question stems, with its only two id-level differences being "
    "positions where the shipped bridge duplicated an id onto a repeated stem. "
)

#: Every dataset the ``uni_mcq`` style supports, keyed by user-facing name.
SUPPORTED: dict[str, DatasetSpec] = {
    "arc_challenge": DatasetSpec(
        name="arc_challenge",
        task="arc_challenge",
        route="C",
        bank_dir=f"{_ATLAS}/arc",
        bridge_path=f"{_ATLAS}/arc/atlas_idx_to_question_id.csv",
        bridge_kind="atlas",
        fit_family="3pl",
        expected_bank_rows=839,
        positional_ids=False,
        calibration=CalibrationConvention(
            num_fewshot=25,
            note=(
                "The one shot count recorded anywhere for these banks: the ATLAS "
                "release states its ARC-Challenge labels are 25-shot leaderboard "
                "scoring. It does not state the prompt those responses were produced "
                "behind, nor the metric they were scored by, so neither is claimed "
                "here. This harness scores 0-shot, matching Research's own CAT."
            ),
        ),
        notes=(
            "The one ATLAS bank whose join is confirmed sound, and the reason is that "
            "its bridge was shipped with the release rather than generated here. Every "
            "other ATLAS bridge in this file was produced by "
            "Inputs/ATLAS/scripts/build_atlas_idx_bridge.py against the assumption that "
            "ATLAS index k is the k-th instance olmo-eval enumerates from the "
            "HuggingFace split; this one records what the release itself used, which is "
            "Open LLM Leaderboard v1 harness_arc_challenge_25 example order. The two "
            "are not the same order and the difference is the whole difference: "
            "regenerating the bridge for this benchmark and diffing it against the "
            "shipped one gives agreement on 2 of 1,172 positions. Leaderboard example "
            "order is not inferred from that, it is checked directly -- the v1 details "
            "parquet for a leaderboard model (open-llm-leaderboard-old/details_*, config "
            "harness_arc_challenge_25) carries one row per example in evaluation order, "
            "and its row k stem equals this bridge's atlas_idx k+1 question on 1,172 of "
            "1,172 rows. "
            "It also verifies empirically, more strongly than any other bank here. Item "
            "p-value against the bank's implied b = -d/a1 is Spearman -0.857 through each "
            "of the two committed ATLAS response matrices independently, -0.542 through "
            "63 of our own models and -0.319 through the four Qwen2.5 checkpoints the "
            "other banks have; a positional bridge in its place collapses that to -0.017 "
            "and a one-position shift to -0.027. Both the ordering and the difficulties "
            "are therefore attested, and this is the bank to compare the others against. "
            "That second role is now load-bearing. Because this bridge is a known answer, "
            "it is what scripts/build_leaderboard_bridge.py is validated against before "
            "the same recovery is applied to the three banks that have no shipped bridge, "
            "and the run is committed at bridges/arc_challenge.control.json. The "
            "recovered order reproduces this file on 1,172 of 1,172 question stems, "
            "differing on the id at exactly two positions -- 331 and 738 -- where this "
            "file is the one in error: ATLAS resolved its ids by stem text, ARC holds two "
            "stems that recur under a second id, and both collided onto the wrong twin, "
            "which is why Mercury_406639 and Mercury_SC_LBS10597 each appear twice here "
            "and why TIMSS_2003_8_pg47 and Mercury_7116183 appear not at all. "
            "One of those two costs a real item and it is worth stating rather than "
            "rounding away. Mercury_SC_LBS10597 is claimed by atlas_idx 331 and 644, only "
            "331 is calibrated, and 331 is really TIMSS_2003_8_pg47 -- so the vendored "
            "item printed as Mercury_SC_LBS10597 carries its twin's difficulty. Both are "
            "ARC-Challenge questions reading 'Which is a chemical change?' with different "
            "choice sets. The Mercury_406639 collision costs nothing: of its two claimants "
            "only 249 is calibrated and 249 is genuinely Mercury_406639. So one item in "
            "650 is misattributed. "
            "This bank nonetheless keeps the shipped bridge rather than the reproduction. "
            "Every figure quoted above, and every comparison the other three banks are "
            "judged by, was measured against this file; replacing the one independently "
            "attested artifact here to recover 0.15% of its items would trade the "
            "reference away for nothing that moves a theta. "
            "ATLAS calibrated ARC-Challenge on 3,747 train models. Native string ids "
            "(Mercury_7175875), so the join is unambiguous. Its bridge uniquely carries "
            "a third 'question' column, which is stem-only and therefore not usable as "
            "an item source. Calibrated under 25-shot leaderboard scoring. Yields 650 "
            "usable items: 189 of the 839 rows have non-positive discrimination. That "
            "the release's unqualified 'ARC' is ARC-Challenge is settled at item-id "
            "level rather than inferred from counts: 763 of the 765 ids in Route A's "
            "own arc_challenge_items.csv appear in this bridge, the two absentees being "
            "TIMSS_2003_8_pg47 and Mercury_7116183, while none of the bridge's 1,172 "
            "ids appears anywhere in Route A's arc_easy_items.csv. Near-total "
            "containment one way, complete disjointness the other."
        ),
    ),
    "hellaswag": DatasetSpec(
        name="hellaswag",
        task="hellaswag",
        route="C",
        bank_dir=f"{_ATLAS}/hellaswag",
        bridge_path=f"{_BRIDGES}/hellaswag.csv",
        bridge_in_repo=True,
        bridge_kind=CONTENT_HASH,
        fit_family="3pl",
        expected_bank_rows=5600,
        positional_ids=False,
        calibration=CalibrationConvention(
            note=(
                "The ATLAS release records nothing about how HellaSwag's calibration "
                "responses were prompted or scored. The bare-context prompt and "
                "unnormalized sum used here are the olmo-eval task's, which is a "
                "choice about self-consistency and not evidence about the bank."
            ),
        ),
        notes=(
            "The best-attested bank here after ARC, and the one whose repair is easiest "
            "to believe because it was measured before and after. "
            + _REBUILT_BRIDGE
            + "Of its 10,042 positions only 3 coincide with split order, so the bridge "
            "it replaced was wrong almost everywhere. "
            "Empirically the new join holds up and the old one did not. The 1,984 items "
            "with committed per-question responses (Inputs/Open/LLM-Judge/mcq/hellaswag/, "
            "four Qwen2.5 checkpoints) give a p-value whose rank correlation against the "
            "bank's implied b = -d/a1 is -0.538 over the 910 rows that join, against a "
            "scrambled null of 0.00 with a standard deviation of 0.03. It was -0.036 on "
            "the superseded bridge. That is not merely negative but larger than ARC's own "
            "verified join reads on these same four checkpoints (-0.319), and it "
            "strengthens as the bank's weaker items are excluded: -0.563 over the 805 "
            "rows with a1 >= 1. It is also specific to this exact ordering rather than to "
            "any plausible one -- shifting the recovered order by a single position "
            "collapses it to +0.095, and by two to +0.040. "
            "Its join key is a content hash rather than the native 'ind', and here that "
            "is a correctness matter rather than a preference. The validation split "
            "concatenates the zeroshot and indomain sub-splits, whose ind values each "
            "restart, so 10,042 rows carry only 9,609 distinct values -- ind 180 is both "
            "'Sharpening knives' at position 8 and 'How to become a sports announcer' at "
            "position 3260. Keying on ind cost 204 items to the ambiguous-id drop, which "
            "is why this bank now yields 5,044 usable items against the 4,840 it yielded "
            "before: the only rows lost are the 556 with non-positive discrimination, and "
            "nothing at all is lost to the bridge or the join. "
            "The ATLAS release reports 3,467 models for HellaSwag, which is the combined "
            "train and test population rather than the calibration set; see the WinoGrande "
            "note for why that distinction matters. The 3PL fit's guessing parameter has a "
            "p95 of 0.834 on a four-choice benchmark where chance is 0.25, which is the "
            "third parameter absorbing misfit; a real 2PL refit would be the better bank "
            "if one were vendored, and that is now the largest remaining reservation about "
            "this bank rather than its join."
        ),
    ),
    "winogrande": DatasetSpec(
        name="winogrande",
        task="winogrande",
        route="C",
        bank_dir=f"{_ATLAS}/winogrande",
        bridge_path=f"{_BRIDGES}/winogrande.csv",
        bridge_in_repo=True,
        bridge_kind=CONTENT_HASH,
        fit_family="3pl",
        expected_bank_rows=1045,
        positional_ids=False,
        calibration=CalibrationConvention(
            note=(
                "The ATLAS release records nothing about how WinoGrande's calibration "
                "responses were prompted or scored. Trinh & Le partial evaluation is "
                "the olmo-eval task's formulation and the only one that measures "
                "coreference, which is why it is used here; the bank does not say "
                "whether it is the one its difficulties were estimated behind."
            ),
        ),
        notes=(
            "The weakest of the three repairs, and the one to read the reservations on "
            "before quoting a theta from it. "
            + _REBUILT_BRIDGE
            + "Of its 1,267 positions exactly 1 coincides with split order, and the "
            "bridge it replaced was a pure identity map -- atlas_idx - 1 equalled "
            "question_id on every row -- so the old join was wrong essentially "
            "everywhere and carried no information beyond the refuted assumption. "
            "The ordering is confirmed; the difficulties are corroborated only weakly. "
            "The four Qwen2.5 checkpoints in Inputs/Open/LLM-Judge/mcq/winogrande/ give "
            "a p-value correlating with the bank's implied b = -d/a1 at -0.177 over 865 "
            "rows, against a scrambled null of 0.00 with a standard deviation of 0.04. "
            "That is the right sign where the superseded bridge gave the wrong one "
            "(+0.037), roughly five standard deviations from the null, stable under "
            "leave-one-out across the four checkpoints (-0.147 to -0.181), and specific "
            "to this exact ordering -- a one-position shift in either direction collapses "
            "it to -0.017 and +0.017. What it is not is as large as it should be. "
            "Attenuating ARC's four-model -0.319 for the difference in how reliably each "
            "p-value ranks items (Spearman-Brown 0.616 here against 0.835 there) predicts "
            "about -0.27, so roughly a third of the expected signal is unaccounted for. "
            "Two things plausibly absorb it and neither is demonstrated: these four "
            "checkpoints score 0.537 to 0.603 on WinoGrande, so their p-value is close to "
            "a coin flip and its five attainable values are mostly noise; and the ATLAS "
            "calibration is a 5-shot leaderboard convention while this harness scores "
            "0-shot Trinh & Le partial evaluation, which reorders difficulty most on a "
            "benchmark where models are near chance. "
            "One confound recorded against the earlier finding is now ruled out rather "
            "than merely argued about. Our own WinoGrande harvest is keyed by split "
            "position, so it could in principle have been misordered itself, which would "
            "have made this test uninterpretable in either direction; the harvest's "
            "recorded gold letter agrees with the task's on 1,267 of 1,267 items, and on "
            "a two-choice benchmark a misordered harvest would sit near half that. The "
            "harvest is correctly keyed, so this measures the bank. "
            "The ATLAS release reports 4,680 models for WinoGrande. That figure is the "
            "combined train and test population, not the calibration set: for ARC the "
            "same table says 4,162 while the committed matrices hold 3,747 train and 417 "
            "test rows. No WinoGrande matrix is committed, so the calibration count is "
            "not known here and none is claimed. Bank max index, bridge rows and "
            "enumerated instances all agree at 1,267. 180 of the 1,045 calibrated rows "
            "have a non-positive discrimination and are dropped (X3 is the first, "
            "a1 = -0.25); nothing is lost to the bridge or the join, so 865 items survive "
            "at 100% overlap, the same count as before the repair and for the same "
            "reasons -- only which question each row describes has changed. Enumeration "
            "needs DataSource(path='allenai/winogrande'): the bare 'winogrande' repo id "
            "the task used to declare is no longer accepted by the Hub."
        ),
        blocked=(
            "withheld pending a stronger join check, not because anything is known to be "
            "wrong. The rebuilt bridge is the best evidence available for it, but the "
            "evidence tops out well short of HellaSwag's: the four-model harvest at "
            "Inputs/Open/LLM-Judge/mcq/winogrande carries no question text -- its columns "
            "are question_id, model, benchmark, predicted, gold, result, scoring_method -- "
            "so the join can only be checked through the recorded gold letter. That agrees "
            "on 1,267 of 1,267 items, which rules out any reordering that moves a gold "
            "across positions, but on a two-choice benchmark roughly half the items share "
            "each gold, so a permutation confined to same-gold items would pass unseen. "
            "The correlation is correspondingly weak at -0.177 against HellaSwag's -0.538, "
            "and its four checkpoints all score 0.537-0.603, near enough to chance that "
            "the p-value reliability is 0.540. Everything is retained and vendored: "
            "delete the blocked reason to restore it once a harvest carrying question "
            "text, or one over more than four models, makes the join measurable"
        ),
    ),
    "gsm8k": DatasetSpec(
        name="gsm8k",
        task="gsm8k",
        route="C",
        bank_dir=f"{_ATLAS}/gsm8k",
        bridge_path=f"{_BRIDGES}/gsm8k.csv",
        bridge_in_repo=True,
        bridge_kind=CONTENT_HASH,
        fit_family="3pl",
        expected_bank_rows=1306,
        positional_ids=False,
        modality="generative",
        calibration=CalibrationConvention(
            note=(
                "The ATLAS release records no prompt, shot count or metric for GSM8K. "
                "Its population is Open LLM Leaderboard v1, whose GSM8K convention is "
                "5-shot strict match, so the last-number strict match used here is "
                "likely the right family of grader and the 8-shot block this harness "
                "prepends is likely not the calibrated count -- but both of those are "
                "inferences about the harvest rather than anything the bank states, "
                "which is why all three fields above say so."
            ),
        ),
        notes=(
            "The one generative dataset here: items carry no choices, and a response "
            "is graded by extracting the final number from a sampled completion and "
            "exact-matching it against metadata['gold_answer']. Strict match is used "
            "because a more forgiving grader would score the same model higher against "
            "these parameters and the resulting theta would not be comparable to the "
            "published one; that the calibration used it is inferred from the harvested "
            "population being Open LLM Leaderboard v1 rather than recorded anywhere. "
            "The ATLAS "
            "release reports 4,195 calibration models over 1,307 items in 12 chunks; "
            "it does not publish the train/test split of that population, so no "
            "train-model count is claimed here. The task declares no split and so "
            "inherits the TaskConfig default of test, giving the 1,319 instances the "
            "bridge was built against. Bank max index, bridge rows and enumerated "
            "instances all agree at 1,319. 8 of the 1,306 calibrated rows have a "
            "non-positive discrimination and are dropped, leaving 1,298 items at 100% "
            "overlap -- the same count as before the bridge was rebuilt, since only "
            "which question each row describes has changed. "
            "Enumeration needs DataSource(path='openai/gsm8k'): the bare 'gsm8k' repo "
            "id the task used to declare is no longer accepted by the Hub. "
            + _REBUILT_BRIDGE
            + "Of its 1,319 positions exactly 2 coincide with split order, and the "
            "bridge it replaced was a pure identity map -- atlas_idx - 1 equalled "
            "question_id on every row -- so the old join was wrong essentially "
            "everywhere. "
            "This bank has no empirical verification of its own and cannot be given "
            "one, which is the single most important thing to know about it. No "
            "response matrix, actual_accuracy.csv, item-selection frequency or per-item "
            "statistic for gsm8k is committed on any of the 113 local and remote refs "
            "in this checkout, so the p-value correlation that reads -0.538 on the "
            "rebuilt HellaSwag bank, -0.177 on WinoGrande and -0.857 on ARC cannot be "
            "computed here at all. What stands behind this bridge is the recovery "
            "procedure and nothing else: the same script, reading the same pinned v1 "
            "details commit, reproduces ARC's shipped bridge on 1,172 of 1,172 stems, "
            "and matched gsm8k's 1,319 examples to the split as an exact bijection on "
            "question text with no fuzzy matching anywhere in it. That is a good reason "
            "to prefer this bridge to the one it replaced, which is refuted. It is not a "
            "measurement of this bank, and a theta from it rests on the ARC control "
            "rather than on anything observed about GSM8K."
        ),
        blocked=(
            "withheld because its join cannot be measured at all, which is a weaker "
            "position than any other bank here occupies. No response matrix, "
            "actual_accuracy.csv, item-selection frequency or per-item statistic for "
            "gsm8k exists on any of the 113 refs in this checkout, so the p-value "
            "correlation that reads -0.857 on ARC and -0.538 on the rebuilt HellaSwag "
            "cannot be computed for it even in principle. Its bridge rests entirely on "
            "the recovery procedure -- which does reproduce ARC's shipped bridge on "
            "1,172 of 1,172 stems and matched gsm8k's 1,319 examples as an exact "
            "bijection on question text -- and that is an argument from uniformity of "
            "the ATLAS pipeline, not an observation about this bank. Everything is "
            "retained and vendored: harvesting per-item gsm8k responses for even a "
            "handful of models would make it measurable, and is the one thing that "
            "would justify deleting this reason"
        ),
    ),
    "leaderboard_math": DatasetSpec(
        name="leaderboard_math",
        task="leaderboard_math",
        route="B",
        bank_dir=f"{_EXPERIMENTS}/openlm_atlas_3pl/math/calibration",
        bridge_path=f"{_BRIDGES}/leaderboard_math.csv",
        bridge_in_repo=True,
        bridge_kind=CONTENT_HASH,
        fit_family="3pl",
        expected_bank_rows=1206,
        positional_ids=False,
        modality="generative",
        answer_type="math_latex",
        calibration=CalibrationConvention(
            metric="lm-eval math_verify exact match",
            note=(
                "The metric is the one recorded fact and it is a deviation: the bank "
                "was fit on lm-eval's math_verify exact match while this harness grades "
                "with olmo-eval's Minerva equivalence, which Research's own "
                "benchmarks.py calls close but not identical and able to shift item "
                "difficulty. The calibration directory holds parameter CSVs, response "
                "matrices and figures and no record of prompting, so the 4-shot Minerva "
                "block used here is the olmo-eval task's value -- it coincides with the "
                "count Open LLM Leaderboard v2 publishes for MATH-Hard, which is where "
                "this bank's responses were harvested, but that is an inference about "
                "the harvest."
            ),
        ),
        notes=(
            "MATH-Hard: the Level-5 slice of the seven MATH subjects that Open LLM "
            "Leaderboard v2 scores, fit locally on that harvest. Ten-fold "
            "cross-validation over 901 models gives r = 0.918, slope 0.705, accuracy "
            "MAE 0.0210 with a 95% CI of [0.0191, 0.0230] and a mean bank of 1,182 "
            "(Research/01_MCQ_ATLAS/data/atlas_replication_cv/math/results.json). 23 of "
            "the 1,206 calibrated rows have a non-positive discrimination and are "
            "dropped, leaving 1,183, and all 1,183 match the task. "
            "Two things about it are worth knowing before reading a report. Its "
            "average CAT length upstream is 96 items to SE <= 0.3, roughly four times "
            "ifeval's and well over twice the max_items this style pins, so a real "
            "checkpoint is likely to stop on the cap with a standard error above 0.3; a simulated "
            "taker drawn from the bank's own 3PL converges at the 8-item floor "
            "instead, and the whole gap between those two numbers is real models "
            "misfitting the model. And the bank has a high floor: difficulties run "
            "from 0.85 at the 5th percentile to 2.78 at the median, so a checkpoint "
            "below theta -1 answers every item wrong however many are administered, "
            "and its reported theta is the prior pulled down rather than a "
            "measurement. On both counts the honest reading is the standard error and "
            "the stop reason, not the point estimate. "
            "This is the one bank in this file whose re-key changed which questions it "
            "holds, and it is the reason the exercise was worth doing. Its bridge was "
            "the composite item_id_map.csv, keyed '<subtask>|<doc_id>' like "
            "'algebra_hard|0', and the leaderboard_math task emits exactly that "
            "composite itself -- so the join read as self-describing and needed nothing "
            "rebuilt by counting. It was wrong anyway. The task numbers each subject's "
            "Level-5 subsequence of DigitalLearningGmbH/MATH-lighteval from zero, while "
            "lm-eval's doc_id counts the pre-filtered lighteval/MATH-Hard split it "
            "originally served, and the two are unrelated orderings of the same 1,324 "
            "problems. Reading the document each doc_id actually named off Open LLM "
            "Leaderboard v2's own per-model detail files matches all 1,324 as an exact "
            "bijection on problem text with no normalization, and exactly 8 of them sit "
            "at the position the composite key assumed. "
            "The consequence is not a count. The bank still holds 1,183 items and still "
            "joins at 100%, and only 5 of those 1,183 bank columns describe the question "
            "they described before: 123 problems have left the bank and 123 others have "
            "entered it, and the remaining 1,055 are the same problems moved onto "
            "different columns. Every guard passed before and after, which is the whole "
            "point -- a permuted bridge joins every row, clears the overlap floor, "
            "converges, and reports a confident theta measured against noise. Anything "
            "computed from this bank before 2026-08-07 was measuring that. "
            "scripts/check_bridge_alignment.py is unmoved at Spearman -0.731 across the "
            "change, and that is not a reassurance but a demonstration of what it does "
            "not attest: it pairs each bank column with its own response-matrix column, "
            "so both sides move together under any permutation of the join. "
            "Its bridge is now the content hash at styles/uni_mcq/bridges/"
            "leaderboard_math.csv, cross-checked against three further models spanning "
            "September 2024 to February 2025, all agreeing on all 1,324 documents. It is "
            "still a single-task dataset, and the enumeration is recovered from the "
            "task's own composite id rather than from the registry, which is what "
            "build_leaderboard_bridge's COMPOSITE_ID case exists for. Routing it through "
            "the multi-task path would abort, and correctly -- check_subtask_alignment "
            "demands each subtask's positions be a gapless 0..n-1 run, and these are "
            "sparse within a subject because calibration dropped items that failed to "
            "converge (geometry_hard has 115 rows spread over positions 0..131). Every "
            "subject's maximum bridge position is below the count the task enumerates "
            "for it (307 / 123 / 132 / 280 / 154 / 193 / 135, summing to 1,324), which "
            "is why the join is total. Because a hash names only the item, vendoring "
            "copies the bridge's subtask column onto each parameter record; without it "
            "this bank would be the only one that cannot attribute an item to a subject "
            "at all, the composite id having carried that until now. "
            "Do not reach for minerva_math_<subject> as an item source. It looks like "
            "the natural one -- the same seven subjects, algebra lining up with "
            "algebra_hard -- and it is not: it enumerates the entire unfiltered test "
            "split of EleutherAI/hendrycks_math and keys metadata['id'] by position in "
            "that split, while the harvest these parameters come from is the "
            "Level-5-filtered MATH-Hard slice. Under the content key that substitution "
            "now fails the way a substitution should: the two tasks render different "
            "problem sets, so the hashes miss and the overlap floor aborts rather than "
            "matching something. It was the switch to bridge_kind='atlas' that used to "
            "be the live hazard here -- keying on the bare question_id, unique only "
            "within a subject, collapsed the 1,206 rows onto 307 distinct ids and left "
            "46 survivors matching minerva positions 0..306 at full overlap -- and that "
            "route is closed now only because nothing reads a question_id from this "
            "spec any more. "
            "Parity caveat, and it is the real one here: the bank was calibrated under "
            "lm-eval's math_verify exact match, while items are graded here by "
            "olmo-eval's Minerva composition -- \\boxed{} or 'Final Answer:' "
            "extraction, then sympy equivalence with a Hendrycks string-normalization "
            "fallback. Research's own benchmarks.py calls the two 'close but not "
            "identical, which can shift item difficulty', and matching the local "
            "Research implementation is the standing policy, so Minerva it is; the "
            "residual disagreement lands in theta's absolute scale and not in "
            "checkpoint-to-checkpoint comparison. One environment quirk compounds it: "
            "minerva_is_equiv wraps its sympy comparison in a signal.SIGALRM timeout, "
            "which does not exist on Windows, so on a Windows box the symbolic half "
            "never runs and every comparison falls back to string normalization. A "
            "Windows run and a Linux run of the same checkpoint can therefore disagree "
            "on a tail of items, with Windows the stricter of the two. "
            "AdaptiveTesting/Inputs/ATLAS/math/ holds a byte-identical mirror of the "
            "parameter CSV beside an atlas_idx_to_question_id.csv whose question_id is "
            "already the composite; the Experiments path is used here because that is "
            "where the fit lives."
        ),
    ),
    "ifeval": DatasetSpec(
        name="ifeval",
        task="ifeval",
        route="B",
        bank_dir=f"{_EXPERIMENTS}/openlm_atlas_3pl/ifeval/calibration",
        bridge_path=f"{_BRIDGES}/ifeval.csv",
        bridge_in_repo=True,
        bridge_kind=CONTENT_HASH,
        fit_family="3pl",
        expected_bank_rows=535,
        positional_ids=False,
        modality="generative",
        answer_type="ifeval_strict",
        calibration=CalibrationConvention(
            prompt_style="the bare prompt; chat framing mixed across the harvest",
            metric="prompt_level_strict_acc",
            note=(
                "The one bank whose recorded metric this harness matches exactly: it "
                "was fit on prompt_level_strict_acc, which is what "
                "IFEvalPromptStrict computes. The loose variant the same verifier run "
                "also produces would make every item look easier than its difficulty. "
                "The prompt text is recorded too, and it is the bare instruction: "
                "lm-evaluation-harness's leaderboard_ifeval reads doc_to_text straight "
                "off the dataset's prompt field, which is forced rather than convenient "
                "-- an IFEval prompt states verifiable constraints about its own text, "
                "so any framing is one more constraint the verifiers were never told "
                "about. What is *not* uniform, and is the one thing to know before "
                "comparing a theta from this bank against a published number, is "
                "whether that prompt arrived inside a chat template. Open LLM "
                "Leaderboard v2 decided that per submission -- automatically on for "
                "chat models, off for pretrained ones -- so the 1,102-model harvest "
                "these difficulties were fit over contains both framings and no single "
                "run-time value matches all of it. Its own per-example records show the "
                "split on one document: for key 1000, meta-llama/Meta-Llama-3-8B was "
                "sent the prompt verbatim and microsoft/Phi-3-mini-4k-instruct was sent "
                "'<|user|>\\n...<|end|>\\n<|assistant|>\\n'. This harness scores in "
                "completion format, matching the pretrained half exactly, prompt text "
                "and generation kwargs alike; the gap against the chat-templated half "
                "is a scale shift with nowhere to go but theta, and chat models gain "
                "substantially from the template on this benchmark in particular. Read "
                "a theta from this bank as measured on the completion scale. Nothing "
                "records the shot count, and there is little room for it to differ."
            ),
        ),
        notes=(
            "The strongest bank here by a wide margin: ten-fold cross-validation over "
            "1,102 models gives r = 0.942, slope 0.944 and accuracy MAE 0.0425 with a "
            "95% CI of [0.0400, 0.0451], reaching SE <= 0.3 in 27 items on a mean bank "
            "of 513.5 "
            "(Research/01_MCQ_ATLAS/data/atlas_replication_cv/ifeval/results.json). 24 "
            "of the 535 calibrated rows have a non-positive discrimination and are "
            "dropped, leaving 511, and all 511 match the task. "
            "It is the one dataset with no gold answer anywhere in it. An IFEval prompt "
            "names verifiable instructions -- 'no commas', 'at least three highlighted "
            "sections', '300+ words' -- and correctness is those verifiers run against "
            "the response, so the item's metadata carries instruction_id_list and "
            "kwargs where a gsm8k item carries a number. The binary is prompt-level "
            "*strict* accuracy, every instruction passing on the response exactly as "
            "written, which is the metric the bank was calibrated under; the loose "
            "variant retries each instruction against eight lightly edited copies of "
            "the response and would make every item look easier than its difficulty. "
            "It is still an ordinary generative bank -- one sampled completion, one "
            "binary -- so it needs no third modality, only a grader that reads the item "
            "instead of a gold string. It is scored 0-shot in completion format, the "
            "prompt verbatim, 1,280 tokens, no stop sequences; a stop at a blank line "
            "would truncate a multi-paragraph response into a constraint violation. "
            "Completion format is a deviation from the olmo-eval task, which is "
            "RequestType.CHAT, and it is the one deviation this bank has. It was chat "
            "here too until the base-checkpoint path needed a generative bank it could "
            "run, and the reason it could change is that the calibration turned out not "
            "to settle the question: Open LLM Leaderboard v2 templated its chat "
            "submissions and not its pretrained ones, so both framings are in the "
            "1,102-model harvest and the completion half is what a base checkpoint can "
            "be measured against. It matches lm-evaluation-harness's leaderboard_ifeval "
            "exactly -- doc_to_text is the bare prompt field, until [], do_sample "
            "false, temperature 0, max_gen_toks 1280 -- and the leaderboard's own "
            "records confirm that is what a pretrained submission was sent. The scale "
            "shift against the templated half is in the calibration note and in "
            "DEVIATIONS.md. Grading needs the ifbench verifier registry, a declared "
            "dependency of this project installed from a git URL; vendoring does not. "
            "The package is not the whole prerequisite. Several verifiers -- the ones "
            "counting sentences or capitalised words, such as "
            "change_case:capital_word_frequency -- tokenize with NLTK and fetch "
            "punkt_tab and averaged_perceptron_tagger_eng the first time they run. "
            "Where that fetch cannot reach the network the failure is a LookupError "
            "raised mid-grade, on whichever item happens to select such a verifier, "
            "after the checkpoint has been staged; pre-fetch both corpora on a box that "
            "will run this bank. With them present all 511 items grade to a real "
            "per-instruction verdict, none ungradable. "
            "Its bridge is keyed by content hash, rebuilt from what the leaderboard "
            "recorded evaluating, and it is committed at styles/uni_mcq/bridges/"
            "ifeval.csv. What it replaced was "
            "Inputs/ATLAS/ifeval/atlas_idx_to_question_id.csv, whose question_id is a "
            "bare integer equal to the task's metadata['id'] -- the position in the "
            "split's native order, and therefore a statement about an enumeration rather "
            "than about a prompt. Its ids run 0..540 with six absent (115, 440, 453, "
            "455, 523, 528) because 541 prompts were harvested and 535 calibrated, and "
            "the re-keyed bridge keeps exactly that sparse run in its split_index column. "
            "The measurement came out in the old key's favour and that is worth stating "
            "plainly rather than dressing up. All 541 evaluated documents match the "
            "task's enumeration as an exact bijection with no normalization -- an IFEval "
            "prompt names verifiable instructions about its own text, so there is no "
            "normalization that would not also change what the verifiers check -- and "
            "all 541 sit at the position a fresh enumeration assumes. The bank is "
            "therefore unchanged at 511 items naming the same 511 prompts, confirmed "
            "position by position against the superseded file. What changed is the "
            "evidence and the failure mode: the join now rests on the leaderboard's "
            "record of what it evaluated, cross-checked against three further models "
            "spanning September 2024 to February 2025 and agreeing on all 541 documents, "
            "and a re-rendered prompt now misses rather than silently naming its "
            "neighbour's. That matters more here than on most banks, because the "
            "positional key this replaced had a live near-miss beside it. "
            "Not Experiments/openlm_atlas_3pl/ifeval/data/item_id_map.csv either: same "
            "535 items, but keyed as the composite 'ifeval|k', which this task never "
            "emits. "
            "eduLLM-Evals/data/IFEval/scenarios.jsonl is not an item source either, and "
            "is the dangerous one: its ife_NNNN ids were assigned after sorting the "
            "prompts by integer key, while the dataset's native order is lexicographic "
            "by the key's string form, so document 0 is key 1000 and is ife_0049 while "
            "ife_0000 is key 13. Both id spaces are dense 0..540 over the same 541 "
            "prompts, so a positional join finds ~100% overlap, clears the floor, and "
            "produces a well-formed bank of mismatched items. That near-miss is the "
            "clearest argument for the content key on this bank: under the old integer "
            "id the two orderings are indistinguishable to every guard in the pipeline, "
            "and under a hash of the prompt the wrong one joins nothing."
        ),
    ),
    "gpqa": DatasetSpec(
        name="gpqa",
        task="",
        subtasks=(
            ("diamond", "gpqa_diamond"),
            ("extended", "gpqa_extended"),
            ("main", "gpqa_main"),
        ),
        route="B",
        bank_dir=f"{_EXPERIMENTS}/openlm_gpqa_atlas_3pl/calibration",
        bridge_path=f"{_EXPERIMENTS}/openlm_gpqa_atlas_3pl/data/item_id_map.csv",
        bridge_kind="item_id_map",
        fit_family="3pl",
        expected_bank_rows=1192,
        positional_ids=True,
        modality="mcq",
        choice_order_control=True,
        duplicate_question_precedence=("extended", "main", "diamond"),
        calibration=CalibrationConvention(
            prompt_style="gpqa",
            num_fewshot=0,
            metric="acc_norm",
            note=(
                "The only bank here whose calibration convention is known in full and "
                "matched in full, and it is known because the harvest was identified "
                "rather than because anything upstream wrote it down -- the local fit "
                "records no prompt, shot count or metric. Each cell of the fit's own "
                "response matrix is the Open LLM Leaderboard v2 acc_norm outcome of "
                "lm-evaluation-harness's leaderboard_gpqa for that document, verified "
                "cell by cell against the leaderboard's per-example records for two of "
                "the fitted models over all 1,192 columns each with no disagreement. "
                "That task is output_type: multiple_choice, ranking '(A)'..'(D)' by "
                "log-likelihood at 0-shot under acc_norm with no system prompt for any "
                "submission, and the prompt recorded here is the arg_0 those records "
                "carry. This bank was vendored as generative until 2026-08-08 on the "
                "registered olmo-eval task's chat chain-of-thought default; that was "
                "never the convention the parameters were fit against."
            ),
        ),
        notes=(
            "Multiple choice on paper, multiple choice in the calibration, and "
            "generative in this repo until 2026-08-08, which is the first thing to know "
            "about it. The registered gpqa tasks default to an MCQAChatFormatter "
            "carrying an expert-scientist system prompt that asks for step-by-step "
            "reasoning ending in 'ANSWER: X', 1,024 greedy tokens, and "
            "AccuracyMetric(scorer=MultipleChoiceScorer) over the extracted letter, and "
            "that default was taken as the convention because Research never wired GPQA "
            "into its CAT -- adaptive/benchmarks.py on origin/Research registers "
            "arc_challenge, hellaswag, winogrande, csqa, piqa, gsm8k, ifeval, math and "
            "truthfulqa, and gpqa appears nowhere under src/olmo_eval/adaptive/. The "
            "absence of a precedent was read as licence to use the task's default; it "
            "was not, because the harvest is itself a record of the convention and is "
            "readable. Each cell of the fit's own gpqa_response_matrix_{train,test}.csv "
            "is the leaderboard v2 acc_norm outcome of leaderboard_gpqa for that "
            "document: for microsoft/Phi-3-mini-4k-instruct and 01-ai/Yi-1.5-6B-Chat, "
            "both fitted rows, the matrix agrees with the leaderboard's own per-example "
            "records on 1,192 of 1,192 columns each. So the difficulties were estimated "
            "behind a log-likelihood ranking of four option letters, and grading them by "
            "chain of thought was a convention mismatch EAP would have absorbed entirely "
            "into theta. It also made the bank unreachable: a chain of thought needs a "
            "chat template and a base checkpoint has none. "
            "The choice ordering is the hazard this change turns on and it is worth "
            "being exact about. GPQATask.process_doc reshuffles the choices per question "
            "from Random(f'{seed}:{index}'), so the gold index means nothing apart from "
            "one ordering, and the generative bank had frozen that ordering into each "
            "stem as a lettered block with the post-shuffle letter beside it. Rather "
            "than re-derive the order, vendoring now transfers it: "
            "scripts/freeze_choice_order.py parsed the 395 committed stems back into "
            "(question, choices) pairs -- requiring a unique decomposition and a "
            "byte-identical round trip for each -- into bridges/gpqa.choice_order.json, "
            "and check_choice_order holds every re-vendored item to that file before "
            "anything is written. Today's fresh enumeration and the ordering frozen at "
            "the old vendoring agree on all 395 items, on the choice texts in order, on "
            "the question, and on which option the gold names. "
            "That is a transfer with no ordering assumption in it, and it is still not "
            "enough on its own: both halves of it are this repo's, so a mapping shifted "
            "before the stems were written would be reproduced faithfully and every "
            "byte-equality test would pass. Two outside checks answer that, and the "
            "control carries the evidence for both so a test can re-run them offline. "
            "The first is about identity and reaches every item. Open LLM Leaderboard "
            "v2's per-example records carry GPQA's own Correct Answer and Incorrect "
            "Answer columns untouched, and our four options equal that set on 395 of 395 "
            "while our gold_index names the Correct Answer on 395 of 395. It "
            "discriminates: rotating the gold index breaks it on 390 of the 395, and the "
            "five survivors are questions whose source data repeats an option, where the "
            "two candidates are the same string; swapping two options breaks it on 188, "
            "which is every item whose gold sits at one of the two positions moved. "
            "The second is about behaviour and is the one that closes the loop. Those "
            "same records carry each option's log-likelihood under "
            "microsoft/Phi-3-mini-4k-instruct, one of the fitted models; replaying them "
            "through our choice list and our gold_index reproduces the leaderboard's own "
            "acc_norm on 388 of 388 replayable items, and equals the cell of the "
            "calibration response matrix that this item's difficulty was fit from on 388 "
            "of 388. Controls: 178 of 388 with the gold index rotated by one, 178 with "
            "the choices rotated, 276 with two swapped. The 7 items it cannot reach are "
            "the ones lm-evaluation-harness's own preprocessing empties -- it deletes "
            "bracketed spans, which leaves two options indistinguishable, and on "
            "extended|476 all four are bracketed and it showed the model four blank "
            "options -- so they are covered by the identity check and not by this one. "
            "And the gold lands on all four positions (103/87/110/95), so a wrong "
            "ordering would fail loudly rather than coincide. "
            "The index-vs-id hazard that would apply to a single-task bank does not "
            "arise here. These tasks emit metadata['index'] and no metadata['id'], so "
            "load_task_items's fallback would key them by a bare integer and find zero "
            "overlap against 'diamond|0' -- but a multi-task spec goes through "
            "load_multi_task_items, which keys by the composite position and ignores "
            "the instance's own id on purpose. "
            "Fit locally rather than published by ATLAS, on "
            "992 of the 1,102 Open LLM Leaderboard v2 models harvested, the other 110 "
            "being held out for validation. Its bridge is a four-column "
            "item_id_map.csv whose joinable key is the composite item_id (for example "
            "'diamond|0'), and that position is numbered within a subtask and restarts "
            "at zero, so the bank spans three registered tasks rather than one: "
            "gpqa_diamond at 198 bridge rows, gpqa_extended at 546 and gpqa_main at "
            "448, summing to the bridge's 1,192. It is the most fragile join here. "
            "Positions are rebuilt by counting each subtask's enumeration from "
            "scratch, so a task emitting one instance fewer than the bridge expects "
            "rekeys every later item onto its neighbour's question while the overlap "
            "still reads near-total; vendoring therefore checks each subtask's "
            "enumerated count against its bridge rows, and applies the overlap floor "
            "per subtask as well as globally, because a subtask holding a sixth of the "
            "bank can fall to half-matched with the dataset still clearing 90%. Of the "
            "1,192 calibrated rows 613 have a non-positive discrimination and are "
            "dropped -- X2 is merely the second row at a1 = -38.9, the most extreme "
            "being X16 at -60.3 -- with nothing lost to a non-finite a, a missing "
            "bridge entry or an ambiguous id, leaving 579 rows to enter the task join. "
            "A further 184 are dropped as duplicate_question, which is the other thing "
            "to know about this bank and is set out below: those 579 rows are only 395 "
            "distinct questions, because the three subsets are nested. The vendored "
            "bank is 395 items. "
            "Under half the bank surviving the a > 0 filter alone makes this a markedly "
            "weaker bank than the ATLAS ones, which keep 78% on ARC and 86% on "
            "HellaSwag. "
            "The upstream experiment README's held-out r = 0.737, MAE 0.070 and "
            "roughly 13 items to SE <= 0.3 are stale: it was written 2026-07-30 "
            "against a single 90/10 split and never revised. The authoritative figures "
            "are the ten-fold cross-validation at "
            "Research/01_MCQ_ATLAS/data/atlas_replication_cv/gpqa/results.json "
            "(commit c5213e9, 2026-08-04): r = 0.652, slope 1.255, accuracy MAE 0.0738 "
            "with a 95% CI of [0.0709, 0.0767], 13.7 items on average, mean bank size "
            "575.6. That design is strictly the stronger one -- every model gets one "
            "out-of-sample prediction against a bank recalibrated without it, n = "
            "1,102 against the single split's 110. Read the r for what it measures: it "
            "correlates predicted accuracy against actual accuracy computed over the "
            "filtered a > 0 subset, not over the whole benchmark, so on a bank that "
            "drops 51% of its items the prediction target itself moves with the "
            "filter. Measured against full-benchmark accuracy GPQA's theta recovery is "
            "r = 0.32 (Research/01_MCQ_ATLAS/data/link_failure_diagnosis/README.md), "
            "which is the figure to quote when the claim is about the benchmark rather "
            "than about the bank. Every one of those figures was computed over the "
            "post-filter 579 and none of them over the deduplicated 395, which is the "
            "right reading of them rather than a caveat: they describe upstream's bank "
            "and the deduplication is ours. The 579 is the count on the frozen bank, "
            "reproduced three independent ways and equal to fold 0 of that "
            "cross-validation, while 575.6 is the ten-fold mean; the two describe "
            "different runs rather than disagreeing. A real 2PL refit exists upstream "
            "at calibration_2pl/ and keeps 873 of the 1,192 rows against the 3PL's "
            "579. We stay on 3PL by policy and the 2PL fit is not vendored; the note "
            "is here because a bank that discards 51% of its items has an alternative "
            "worth knowing about. Vendoring it needs --bank-dir pointed at that "
            "directory as well as --fit-family 2pl, because bank_dir comes from this "
            "spec: the flag alone would relabel the 3PL fit rather than fetch the 2PL "
            "one, which is why it now refuses. "
            "It was blocked until 2026-08-07 on a HuggingFace token scope rather than "
            "on anything about the bank -- the three subtasks enumerate from the gated "
            "Idavidrein/gpqa, and a fine-grained token without the gated-repository "
            "scope returns 403 on every data file while HfApi().dataset_info still "
            "resolves the repo, which surfaces as a FileNotFoundError reading like a "
            "network fault. With the scope granted all three subsets enumerate at "
            "198 / 546 / 448 and the join is total: 579 of 579 filtered rows match, "
            "with each subtask at 100% against its own share, and nothing lost to a "
            "non-finite a, a missing bridge entry or an ambiguous id. "
            "Verified by scripts/check_bridge_alignment.py against the calibration's "
            "own response matrices (1,102 models over the 395 vendored items): item "
            "p-value against the bank's implied b = -d/a1 is Spearman -0.754, with a "
            "scrambled control of +0.002 +/- 0.047. The same test over the 579 rows "
            "before deduplication read -0.746 against -0.001 +/- 0.042, so removing the "
            "repeats sharpened the statistic on a third fewer items, and sharpened "
            "Pearson far more -- -0.26 to -0.39 -- because the copies deduplication "
            "removes are the ones carrying the implausible b magnitudes. "
            "Read what that attests and not more. "
            "For a Route B bank the matrix columns are the same positional index as the "
            "parameter rows, so the statistic confirms that these difficulties describe "
            "the responses they were fit from -- the calibration directory, the "
            "b = -d/a1 conversion and the a > 0 filter -- and says nothing about which "
            "question each column is. That part rests on the bridge's key being the "
            "self-describing composite and on the per-subtask overlap floor, which is "
            "why both are load-bearing here in a way they are not for a bank with a "
            "question-keyed harvest. It says nothing whatever about the order of the "
            "options *within* an item, either, and cannot: the p-value is read off the "
            "calibration's own matrix rather than by re-scoring anything, so a permuted "
            "choice list leaves the statistic exactly where it is. The option order is "
            "what the two outside checks above are for, and this is not one of them. "
            "That composite key is no longer an assumption, and this bank keeps it on "
            "purpose rather than for want of trying. Running the same recovery that "
            "re-keyed musr, ifeval and leaderboard_math -- reading the document each "
            "doc_id named off Open LLM Leaderboard v2's own per-model detail files -- "
            "matches all 1,192 evaluated documents to the three subsets as an exact "
            "bijection on question text, cross-checked against three further models "
            "spanning September 2024 to February 2025 and agreeing on every document, "
            "and every one of the 1,192 sits at the position a fresh enumeration "
            "assumes. So '<subtask>|<doc_id>' names the question the leaderboard "
            "evaluated, measured rather than assumed, which is the same standing "
            "leaderboard_math's composite turned out not to have. "
            "What blocks the re-key is the id, not the join, and the cause is the one "
            "fact that shapes everything else about this bank: GPQA's three subsets are "
            "not different content. Per the paper (arXiv 2311.12022 section 2.3 and "
            "Table 2) they are nested quality filters over a single pool of 546 "
            "questions -- extended is all 546, main is the 448 left after removing "
            "questions both experts missed and all three non-experts got, diamond is the "
            "198 where both experts were right and most non-experts wrong -- so "
            "diamond is inside main is inside extended and 198 + 448 + 546 is the "
            "bridge's 1,192 rather than a coincidence. Enumerating the three tasks "
            "confirms it exactly: 546 distinct questions over the 1,192 instances, 198 "
            "appearing three times, 250 twice and 98 once, with no repeat inside any one "
            "subset. "
            "That is what rules out the content key. content_item_id hashes the question "
            "with its choices and process_doc reshuffles those per subset from "
            "Random(f'{seed}:{index}'), so two labels for one question collide exactly "
            "when their shuffles coincide -- 32 of the 844 shared pairs, close to the "
            "1-in-24 a random permutation of four options predicts -- and both rows of a "
            "collision are dropped as ambiguous. A content-hashed gpqa would therefore "
            "lose calibrated rows to an accident of the shuffle, while keeping every pair "
            "whose shuffles happened to differ, which is the worst of both. The composite "
            "distinguishes the rows and the deduplication below removes the ones that "
            "should not have been separate rows, which is the division of labour that "
            "works. "
            "The repeats are removed at vendoring time, as duplicate_question, and the "
            "reason is stronger than redundancy. The CAT masks administered items by id "
            "and these copies have different ids, so one session could administer the "
            "same question two or three times, score it each time, and hand EAP each "
            "scoring as independent evidence about the checkpoint. The copies also "
            "disagree, systematically rather than randomly. atlas_idx was assigned over "
            "the lexicographically sorted composite ids, so diamond occupies indices "
            "1-198 and was fit almost entirely inside its own first two of the twelve "
            "~100-item chunks calibration ran in (irt_item_parameters_101.csv through "
            "1193.csv), separately from its extended and main twins several chunks away, "
            "with mean-sigma linking only approximately reconciling the chunk scales. "
            "The wild value is nearly always the diamond copy: diamond|91 reads b = "
            "182.58 where its extended twin reads 1.85, diamond|76 reads 43.99 against "
            "0.52 and 0.21, and diamond|100 reads a = 28.020 against its main twin's "
            "0.993 -- and since Fisher information scales with a squared, an inflated "
            "discrimination is preferentially selected. "
            "Precedence is extended, then main, then diamond: for each distinct question "
            "keep the extended calibration if it survived a > 0, else main, else "
            "diamond. That prefers the fit estimated over the widest chunk span and the "
            "largest response set, and it is a rule rather than a judgement per item. Of "
            "the 579 filtered rows only 395 are distinct questions -- 237 calibrated "
            "once, 132 twice, 26 three times -- so 184 are repeats and the bank is 395 "
            "items. The survivors are extended 252, main 119, diamond 24, and those 24 "
            "matter: they are questions whose extended and main calibrations both failed "
            "a > 0, so precedence keeps the diamond copy rather than losing the question. "
            "Deliberately a different rule from hellaswag's, where every claimant to a "
            "contested id was dropped. There, two different questions collided on one id "
            "and neither could be shown to own it. Here it is one question calibrated "
            "repeatedly, the copies are distinguishable by how they were fit, and "
            "dropping all of them would discard the question. "
            "This is nonetheless the weakest bank of the verified set, for the reason "
            "the r above records: most of the items are gone and theta recovery against "
            "full-benchmark accuracy is 0.32. "
            "It also has a floor, in leaderboard_math's sense though a milder one. "
            "Difficulty on the deduplicated bank runs from -0.22 at the 5th percentile "
            "to 1.31 at the median, so there is very little below theta 0 to measure a "
            "weak checkpoint against: a simulated taker at theta -1.5 runs to the "
            "40-item cap and stops at a standard error of 0.32, still above the 0.3 "
            "threshold, and at -1.0 needs 25 items, where the same taker at -0.5 or "
            "above converges at the 8-item floor. Read the standard error and the stop "
            "reason on a low estimate rather than the point estimate."
        ),
    ),
    "musr": DatasetSpec(
        name="musr",
        task="",
        subtasks=(
            ("murder_mysteries", "musr_murder_mysteries"),
            ("object_placements", "musr_object_placements"),
            ("team_allocation", "musr_team_allocation"),
        ),
        route="B",
        bank_dir=f"{_EXPERIMENTS}/openlm_atlas_3pl/musr/calibration",
        bridge_path=f"{_BRIDGES}/musr.csv",
        bridge_in_repo=True,
        bridge_kind=CONTENT_HASH,
        fit_family="3pl",
        expected_bank_rows=754,
        positional_ids=False,
        modality="mcq",
        calibration=CalibrationConvention(
            metric="acc_norm",
            note=(
                "The metric is recorded and this harness matches it: Open LLM "
                "Leaderboard v2 scored MuSR by acc_norm, the length-normalized "
                "log-likelihood ranking, and the bank was fit on that binary. Nothing "
                "upstream records the prompt or shot count, but there is little room "
                "for either to differ -- the harvest is leaderboard v2, whose MuSR "
                "configuration is lm-evaluation-harness's 0-shot leaderboard_musr, and "
                "that is the layout musr.py reproduces."
            ),
        ),
        notes=(
            "The only multiple-choice bank here that is not from ATLAS, and the only "
            "one scored by acc_norm rather than by an unnormalized sum. Both facts "
            "follow from the same thing: it was harvested from Open LLM Leaderboard v2 "
            "and fit locally, and v2 reported MuSR length-normalized. Scoring it by the "
            "sum the other three MCQ banks use would rank a choice set differently on a "
            "benchmark whose options are whole clauses, so score_normalization is a per "
            "dataset setting recorded in the manifest and checked at startup. "
            "Its bank spans three registered tasks rather than one: murder_mysteries at "
            "250 bridge rows, object_placements at 254 and team_allocation at 250, "
            "summing to the bridge's 754. The three are HuggingFace splits of "
            "TAUR-Lab/MuSR rather than subset configurations, which is why the split "
            "name travels on the task class. "
            "Its bridge is keyed by content hash and not by the composite "
            "'<subtask>|<doc_id>' the upstream item_id_map.csv uses, because doc_id is "
            "not a property of a question. It is lm-eval's enumeration position within a "
            "subtask's split at harvest time, so joining through it means re-deriving the "
            "key by counting an enumeration today and assuming the two agree -- the same "
            "class of assumption that proved false for the ATLAS banks, where a "
            "positional bridge disagreed with the shipped one on 1,170 of 1,172 "
            "positions. scripts/build_leaderboard_bridge.py instead reads which document "
            "each doc_id named off Open LLM Leaderboard v2's own per-model detail files, "
            "joins that through item_id_map.csv, and writes atlas_idx -> content hash to "
            "styles/uni_mcq/bridges/musr.csv. "
            "The measurement came out in the old assumption's favour, which is worth "
            "stating plainly rather than dressing up. All 756 evaluated documents match "
            "the task's enumeration as an exact bijection with no normalization, and all "
            "756 sit at the position a fresh enumeration assumes, so the re-keyed bridge "
            "names exactly the items the composite one did and the bank is unchanged at "
            "432. What changed is the evidence and the failure mode. The join now rests "
            "on the leaderboard's record of what it evaluated -- cross-checked against "
            "three further models spanning July 2024 to February 2025, all agreeing on "
            "all 756 documents -- rather than on doc_id being reproducible by counting; "
            "and a re-rendered or reordered split now misses instead of silently naming "
            "its neighbour's question, which the overlap floor turns into an abort. "
            "object_placements is the reason the per-subtask alignment guard tolerates a "
            "sparse bridge, and that tolerance is now load-bearing for gpqa and bbh "
            "rather than here. Its 254 rows spread over positions 0..255, with 136 and "
            "140 absent because those two items were dropped during calibration -- "
            "mid-run rather than at the tail, so the gap is real and not a truncation. "
            "Demanding a gapless 0..n-1 run refused the dataset before enumeration "
            "started, and comparing enumerated instances against bridge *rows* would have "
            "rejected the 256 instances the task correctly yields one step later. Under a "
            "content-hashed key neither question arises: the two dropped positions simply "
            "have no bank row, and nothing is numbered. The guard that does still apply "
            "is the per-subtask overlap floor, unchanged at 90% of each subtask's own "
            "bank rows, reading the subtask off the bridge's own column now that the id "
            "no longer carries it. "
            "Of the 754 calibrated rows 322 have a non-positive discrimination and are "
            "dropped, leaving 432, and all 432 match the task at 100% per subtask, with "
            "nothing lost to a non-finite a, a missing bridge entry or an ambiguous id. "
            "43% dropped puts it beside gpqa rather than beside the ATLAS banks. "
            "The r = -0.036 that once kept this dataset excluded was the pre-filter "
            "figure and was bank hygiene rather than a linking failure: "
            "link_failure_diagnosis/diagnosis_summary.csv records both sides directly, "
            "cat_r_prefilter_old = -0.036264 against cat_r_postfilter_new = 0.745955, "
            "and the ten-fold cross-validation gives r = 0.757. Vendoring applies "
            "exactly that a > 0 filter. "
            "Verified by scripts/check_bridge_alignment.py against the calibration's own "
            "response matrices (1,099 models over the 432 vendored items): item p-value "
            "against the bank's implied b = -d/a1 is Spearman -0.616, with a scrambled "
            "control of +0.001 +/- 0.049. Unambiguously the right sign, about thirteen "
            "standard deviations from the null, and the weakest of the four banks this "
            "test can be run on -- ifeval reads -0.888, gpqa -0.754 and "
            "leaderboard_math -0.731 through the same code path. The fit's own "
            "instability below is the readiest explanation and is not demonstrated to "
            "be the whole of it. As with the other Route B banks, the matrix columns "
            "are the same positional index as the parameter rows, so this attests the "
            "difficulties against the responses they were fit from and says nothing "
            "about which question each column is. That second half is what the content "
            "re-key supplies, and every Route B bank has it now: ifeval and "
            "leaderboard_math were re-keyed the same way, and gpqa and bbh keep their "
            "composite because their items collide under a content hash rather than "
            "because their joins are unverified -- the same recovery confirmed both, "
            "document by document. leaderboard_math is the one where it mattered, its "
            "composite having named the wrong question on all but 8 of 1,324 positions. "
            "The reservation to read before quoting a theta is the fit family. The 3PL "
            "fit is unstable across folds -- one collapses to r = 0.142, for a fold-r "
            "standard deviation of 0.202 -- while the 2PL refit is markedly steadier, "
            "pooling to 0.802 with a worst fold of 0.710 and a standard deviation of "
            "0.042. This pipeline standardizes on 3PL and that is what is vendored, so "
            "the instability is a property of the bank a reader has to carry rather than "
            "something the harness corrects."
        ),
    ),
    "bbh": DatasetSpec(
        name="bbh",
        task="",
        subtasks=tuple((label, f"bbh_{label}") for label in _BBH_SUBTASKS),
        route="B",
        bank_dir=f"{_EXPERIMENTS}/openlm_atlas_3pl/bbh/calibration",
        bridge_path=f"{_EXPERIMENTS}/openlm_atlas_3pl/bbh/data/item_id_map.csv",
        bridge_kind="item_id_map",
        fit_family="3pl",
        expected_bank_rows=5761,
        positional_ids=True,
        modality="mcq",
        frozen_prompt=True,
        calibration=CalibrationConvention(
            metric="acc_norm",
            note=(
                "The metric is recorded on the same footing as MuSR's: Open LLM "
                "Leaderboard v2 scored BBH by acc_norm, the length-normalized "
                "log-likelihood ranking, and this bank was fit on that binary. The "
                "3-shot block and per-subtask description frozen into these stems are "
                "lm-evaluation-harness's leaderboard_bbh configuration, which is where "
                "the harvest came from, so the two are very likely the same prompt -- "
                "but that is an inference about the harvest and the local calibration "
                "directory records nothing, so both fields stay unrecorded."
            ),
        ),
        report_caveat=(
            "Read this bank's predicted accuracy and treat its theta with heavy "
            "scepticism. Both are worth having and only one of them holds up. The full "
            "3,965-item bank recovers full-benchmark accuracy at r = 0.86, and an "
            "adaptive session's p-IRT predicted accuracy correlates at r = 0.674 with an "
            "MAE of 0.114 -- the weakest of the nine banks here and still a usable "
            "prediction. Theta is not: theta_mae is 0.684 at SE <= 0.3 and does not fall "
            "with precision (0.719 at SE <= 0.2, 0.711 at SE <= 0.1), which is what a "
            "unidimensional model applied to 24 unrelated subtasks looks like rather "
            "than a stopping rule that quits too early. No cross-validation was ever run "
            "for this bank; that r = 0.674 is from one 90/10 split over 110 held-out "
            "models, where every other Route B bank here has a ten-fold figure over "
            "about 1,100. min_items is 24 here against 8 everywhere else, so a session "
            "cannot stop before it has administered as many items as the bank has "
            "subtasks -- which makes coverage possible and does not make it happen."
        ),
        notes=(
            "The ninth bank and the least trustworthy, shipped with that written on it "
            "rather than left out. What it is good for is p-IRT predicted accuracy; what "
            "it is not good for is theta. Its report_caveat says so in every "
            "cat_report.json it produces, and those two figures are the summary of "
            "everything below. "
            "Fit locally over the Open LLM Leaderboard v2 harvest -- 1,102 models in the "
            "committed response matrices, 992 train and 110 test -- across 24 registered "
            "tasks joined by the "
            "composite '<subtask>|<position>' its item_id_map bridge uses. Twenty-one "
            "subtasks contribute a dense 0..249; causal_judgement contributes 187, snarks "
            "178 and penguins_in_a_table 146, and those are those subtasks' true full "
            "sizes upstream rather than truncations, which is why the per-subtask "
            "alignment guard finds no gaps anywhere. BIG-Bench Hard's three genuinely "
            "generative subtasks (dyck_languages, multistep_arithmetic_two, word_sorting) "
            "have no closed answer set to rank, are excluded by leaderboard v2, and are "
            "absent from the bank and from the task module alike. "
            "Of the 5,761 calibrated rows 1,796 have a non-positive discrimination and "
            "are dropped, leaving 3,965, and all 3,965 match the task at 100% per "
            "subtask, with nothing lost to a non-finite a, a missing bridge entry or an "
            "ambiguous id. 31% dropped puts it between the ATLAS banks and gpqa's 51%. "
            "Its prompt is frozen rather than reconstructed, and that is the one "
            "structural thing to understand about this bank. leaderboard_bbh is 3-shot "
            "behind a one-line description that differs per *subtask*, while this style's "
            "config.yaml is per dataset, so no setting here could express it. Rather than "
            "build run-time per-subtask few-shot machinery, vendoring calls each task's "
            "own format_request and writes the finished prompt -- description, three "
            "fixed exemplars, the item, the trailing 'A:' cue -- into the item's stem, "
            "exactly as gpqa freezes its shuffled choice block. Run time appends the "
            "choice and adds nothing, which is what prompt_style 'bbh' means, and the "
            "manifest records num_fewshot 3 rather than 0 because 3 is what the model "
            "reads. Three consequences worth stating: the scorer needs no few-shot "
            "support at all; the prompt is byte-identical to the one the difficulties "
            "were estimated against, because it came from the code that would render it "
            "in a full eval rather than from a reconstruction; and a config edit swapping "
            "this dataset onto a stem-framing style is refused at startup, since the "
            "recorded 3 would no longer match the resolved 0. "
            "Two upstream artifacts come with the prompt and both are preserved rather "
            "than tidied, because tidying either would change the string a difficulty "
            "describes. The description runs straight into the first exemplar with no "
            "separator ('...random Boolean expression.Q: not ( ( not not True ) ) is'); "
            "tests/evals/tasks/test_bbh.py guards that one at the renderer and "
            "tests/test_bbh_bank.py guards the 3,965 stems it produced. And the two sized "
            "families share their exemplars: logical_deduction_five_objects and "
            "_seven_objects both get the three-object examples, whose options stop at "
            "(C) while their own items run to (E) and (G), and the three "
            "tracking_shuffled_objects variants do the same. That is what "
            "lm-evaluation-harness's YAML says, so 24 subtasks render 20 distinct "
            "prefixes and completing the short ones would reprompt the 665 vendored "
            "items whose parameters describe the short version. "
            "min_items is 24 for this dataset and 8 for every other, and it is the only "
            "harness setting this bank moves. Its filtered discriminations are inflated "
            "-- median 3.99, the highest of any bank here against roughly 1.4 for gpqa "
            "and musr and 2.2 for math -- so the posterior standard error collapses at "
            "once and upstream's CAT reaches SE <= 0.3 in an average of 8.2 items, the "
            "fewest of the set. Eight items cannot touch a third of 24 unrelated "
            "subtasks, and 24 is the smallest floor under which touching all of them is "
            "arithmetically possible. It is also well above the 15.4 items upstream "
            "needed even at SE <= 0.1, so precision alone would never reach it, and well "
            "below the pinned max_items of 40, so a session that still needs items can "
            "still take them. Two limits belong beside that and neither is small. Raising "
            "the floor forces more items and does not guarantee subtask coverage: "
            "Fisher-information selection takes whatever is most informative at the "
            "current theta and may draw heavily from a few subtasks, and content "
            "balancing is implemented neither here nor in Research. And theta_mae stays "
            "flat at 0.684 across every stopping rule upstream tried, so more items do "
            "not fix the estimate -- that plateau is a unidimensional model applied to 24 "
            "unrelated skills, not a stopping-rule problem. Research's own "
            "within-benchmark MIRT run puts numbers on it: per-skill recovery lifts "
            "causal_judgement from unidimensional r = -0.014 to 0.999 and web_of_lies "
            "from 0.100 to 0.999, and the one-factor fit is rejected against every "
            "higher-factor alternative on AIC and BIC alike. "
            "Verified by scripts/check_bridge_alignment.py against the calibration's own "
            "response matrices (1,102 models, 992 train and 110 test, over the 3,965 "
            "vendored items): item p-value against the bank's implied b = -d/a1 is "
            "Spearman -0.629, with a scrambled control of -0.000 +/- 0.017. That is "
            "thirty-eight standard deviations from the null and the second weakest of the "
            "five banks this test can be run on, above musr's -0.616 and below "
            "leaderboard_math's -0.731, gpqa's -0.754 and ifeval's -0.888. It is also the "
            "bank that most needs the test to be Spearman rather than Pearson, which "
            "reads only -0.110 here: b is a ratio with a1 underneath it and these a1 run "
            "down to 0.0018, so the implied difficulties span -714 to +467 and a handful "
            "of well-ordered items with absurd magnitudes flatten any linear measure. "
            "Read what it attests and not more. For a Route B bank the matrix columns are "
            "the same positional index as the parameter rows, so the statistic confirms "
            "that these difficulties describe the responses they were fit from -- the "
            "calibration directory, the b = -d/a1 conversion and the a > 0 filter -- and "
            "says nothing about which question each column is. That part rests on the "
            "bridge's key being the self-describing composite and on the per-subtask "
            "overlap floor, which is why both are load-bearing here exactly as they are "
            "for gpqa. "
            "The composite is measured rather than assumed, and this bank keeps it "
            "deliberately. Reading the document each doc_id named off Open LLM "
            "Leaderboard v2's own per-model detail files puts all 5,761 evaluated "
            "documents at the position a fresh enumeration assumes, across all 24 "
            "subtasks, cross-checked against three further models spanning September "
            "2024 to February 2025 and agreeing on every document. What blocks the "
            "re-key is BIG-Bench Hard's own data: four items are literal duplicates "
            "within their subtask -- two in sports_understanding sharing an input and a "
            "target, and two in causal_judgement sharing an input under differing "
            "targets -- and since leaderboard v2 attaches one hardcoded choice set to "
            "every item of a subtask, a hash of question and choices cannot separate "
            "them. Widening the match to the target would resolve causal_judgement and "
            "not sports_understanding, and would not help either way, because the "
            "collision is in the id rather than in the match. The composite distinguishes "
            "them by position, which is the one thing that does. "
            "Its published figures are not the kind its siblings have, and that is worth "
            "its own sentence. There is no atlas_replication_cv/bbh directory on "
            "origin/Research: ifeval, math, gpqa and musr each carry a ten-fold "
            "cross-validation over roughly 1,100 models and BBH carries none, so its "
            "r = 0.674 and accuracy MAE 0.114 come from the single 90/10 split in "
            "Research/01_MCQ_ATLAS/data/atlas_replication/summary_pirt_mae_sd_se.csv over "
            "110 held-out models, which is the weaker design. What makes the bank worth "
            "shipping is measured on the full bank rather than on a session: theta over "
            "all 3,965 items recovers full-benchmark accuracy at r = 0.86, up from 0.69 "
            "before the a > 0 filter "
            "(Research/01_MCQ_ATLAS/data/link_failure_diagnosis/README.md). The signal is "
            "in the bank and an early-stopped session does not reach it."
        ),
    ),
}

#: Datasets deliberately not supported, with the reason. Surfaced in error messages
#: so a user asking for one gets the explanation rather than a bare "unknown".
EXCLUDED: dict[str, str] = {
    "math": (
        "not a name this style knows, rather than a bank it lacks: the MATH-Hard bank "
        "is supported and vendored as 'leaderboard_math', after the olmo-eval task its "
        "items are enumerated from. The name matters because 'math' is ambiguous in "
        "this checkout -- math500 is a registered task and a 500-item subset the bank "
        "was never calibrated against, and minerva_math_<subject> enumerates a "
        "different dataset on a different id scheme. Pass --benchmark leaderboard_math"
    ),
    "mmlu": "task exists but no calibrated bank was ever produced",
    "csqa": (
        "registered upstream with a bank_subdir that does not exist on disk; it passes "
        "name validation and then fails at run time"
    ),
    "piqa": (
        "registered upstream with a bank_subdir that does not exist on disk; it passes "
        "name validation and then fails at run time"
    ),
    "arc_easy": (
        "has a Route A bank of its own, and the disqualifier is that the fit is "
        "degenerate: every one of its 1,662 kept items has its discrimination pinned at "
        "the optimizer's 5.0 bound, with min, q25, median, q75 and max all 4.999993, so "
        "the bank carries no information about which item to administer next and its "
        "theta recovery is the worst of the four Route A banks at r = 0.611. Borrowing "
        "the ATLAS bank instead is no answer either -- ATLAS calibrated ARC-Challenge "
        "and the two splits share no items, so that id join would find zero overlap"
    ),
    "sciq": (
        "has a serialized 2PL bank, in both eduLLM-Evals/runs/mcq_irt/ and "
        "runs/mcq_irt_kfold/ on origin/Research, as sciq_items.csv with columns "
        "item, a, b, p_value, point_biserial. The k-fold run is the better of the two: "
        "82 models, 842 of 1,000 raw items kept, theta recovery r = 0.919; the older "
        "holdout run is where the 62-model figure comes from. It stays excluded for two "
        "other reasons. The fit is 2PL where this pipeline standardizes on 3PL, and its "
        "theta is anchored to our own 83-model pool rather than to the ATLAS "
        "population, so the scale is not comparable to the theta the supported banks "
        "report"
    ),
    "openbookqa": (
        "has a serialized 2PL bank, in both eduLLM-Evals/runs/mcq_irt/ and "
        "runs/mcq_irt_kfold/ on origin/Research, as openbookqa_items.csv with columns "
        "item, a, b, p_value, point_biserial. The k-fold run is the better of the two: "
        "83 models, 314 of 500 raw items kept, theta recovery r = 0.929; the older "
        "holdout run is where the 63-model figure comes from. It stays excluded on the "
        "same terms as sciq -- a 2PL fit where this pipeline standardizes on 3PL, and a "
        "theta anchored to our own 83-model pool rather than to the ATLAS population, "
        "so the scale is not comparable to the theta the supported banks report"
    ),
    "boolq": "Route A pilot grids only (4 models), no serialized bank",
    "educationq": "Route A pilot grids only (3 models), no serialized bank",
    "pedagogy": (
        "fit in memory via girth during the feasibility study but never serialized, so "
        "no bank file exists anywhere"
    ),
    "socialiqa": (
        "fit in memory via girth during the feasibility study but never serialized, so "
        "no bank file exists anywhere"
    ),
}


def supported_names() -> list[str]:
    """Return the sorted names of every supported dataset, blocked ones included."""
    return sorted(SUPPORTED)


def ready_names() -> list[str]:
    """Return the datasets that can actually be vendored and run today.

    Excludes any dataset carrying a :attr:`DatasetSpec.blocked` reason. Use this in
    user-facing messages so nobody is pointed at a dataset that cannot run.
    """
    return sorted(name for name, spec in SUPPORTED.items() if spec.blocked is None)


def get_spec(name: str) -> DatasetSpec:
    """Return the :class:`DatasetSpec` for ``name``.

    Raises:
        KeyError: If ``name`` is not supported. The message lists what is supported
            and, when the name is a known exclusion, why it is excluded.
    """
    spec = SUPPORTED.get(name)
    if spec is not None:
        return spec

    ready = ", ".join(ready_names())
    reason = EXCLUDED.get(name)
    if reason is not None:
        raise KeyError(
            f"Dataset {name!r} is not supported by the uni_mcq style: {reason}. "
            f"Ready datasets: {ready}."
        )
    raise KeyError(
        f"Unknown dataset {name!r}. The uni_mcq style supports datasets with a "
        f"unidimensional calibrated bank obtained from ATLAS (Route C) or Open LLM "
        f"Leaderboard v2 (Route B). Ready datasets: {ready}."
    )
