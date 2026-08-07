"""Rebuild a bank's index bridge from what the Open LLM Leaderboard actually evaluated.

Run offline by a developer, never on the eval box::

    python -m diagnostics.mcq_cat.styles.uni_mcq.scripts.build_leaderboard_bridge \\
        --dataset hellaswag
    python -m diagnostics.mcq_cat.styles.uni_mcq.scripts.build_leaderboard_bridge \\
        --dataset musr

Two generations of leaderboard sit behind the banks this style consumes, and they fail
the same way for different reasons. Both are repaired here, by the same move: stop
asserting which question a bank column describes and read it off the harness's own
per-model detail repos, which record one entry per example and carry enough of that
example to identify it. What is written is therefore a lookup rather than a
reconstruction, and its key is a content hash, so a re-rendered or reordered split
*misses* instead of silently naming its neighbour's question.

**v1, the Route C banks.** ATLAS calibrated over Open LLM Leaderboard v1 responses and
keyed the result by nothing but a column position, ``X<k>``. The superseded generator
``Inputs/ATLAS/scripts/build_atlas_idx_bridge.py`` assumed column *k* is the *k*-th
instance olmo-eval enumerates from the HuggingFace split; the leaderboard shuffled, so
the two orders are a permutation of each other, and on ARC -- the one benchmark with a
shipped bridge to compare against -- they agree on 2 of 1,172 positions. A permuted
bridge is the worst available failure: every row joins, overlap reads 100%, the CAT
converges, and the reported ability is noise. :data:`LEADERBOARD_ORDER` names the
pinned v1 details parquet per dataset, whose row *k* is ATLAS column *k + 1*, and
:data:`LEADERBOARD_RECIPES` rebuilds what the v1 harness wrote in that row so the two
can be matched.

**v2, the Route B banks.** These join through an ``item_id_map.csv`` whose key is the
composite ``<subtask>|<doc_id>``. That is a weaker version of the same assumption:
``doc_id`` is lm-eval's enumeration position within a subtask's split, so it identifies
a question only relative to an enumeration performed at harvest time, and rebuilding it
now by counting a fresh enumeration anchors nothing. Leaderboard v2's per-model detail
repos record the document each ``doc_id`` named, which turns the composite key into a
statement about a question. :data:`LEADERBOARD_V2` holds one entry per benchmark and is
the only thing that has to be written to bring a new one through.

Whichever generation a dataset comes from, nothing is written until the match is a
bijection over the whole split and the ids it emits are unique. A recipe that renders
the harness's text wrongly, or a split revised since the harvest, fails there rather
than producing a bridge that is right for most rows.
:func:`verify_against_shipped` adds a known-answer control for the one v1 dataset that
can have one, and :func:`cross_check_documents` adds the v2 equivalent -- the pinned
ordering compared against models evaluated months apart, since an ordering that moved
between leaderboard runs would not be an ordering at all.
"""

from __future__ import annotations

import argparse
import ast
import csv
import io
import json
import logging
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..datasets import CONTENT_HASH, content_item_id, get_spec

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [uni_mcq.bridge] %(levelname)s: %(message)s",
)
log = logging.getLogger("uni_mcq.bridge")

#: repo_root/diagnostics/mcq_cat/styles/uni_mcq/scripts/build_leaderboard_bridge.py
REPO_ROOT = Path(__file__).resolve().parents[5]

#: Where rebuilt bridges are committed, relative to the repo root.
BRIDGES_DIR = "diagnostics/mcq_cat/styles/uni_mcq/bridges"

#: How much of the stem each bridge row carries, for a reader auditing the file.
#:
#: Audit only, and never an item source: it is truncated, whitespace-collapsed, and
#: taken before the choices that make several HellaSwag stems distinct. The joinable
#: key is the hash beside it, which is checkable by recomputation rather than by eye.
STEM_CHARS = 96

#: Columns of a bridge rebuilt from v1 example order.
BRIDGE_COLUMNS = ("atlas_idx", "item_id", "split_index", "question")

#: Columns of a bridge rebuilt from v2 details, which name a subtask as well.
#:
#: ``subtask`` is not decoration. A Route B bank spans several registered tasks, and once
#: its rows are keyed by a content hash the id no longer says which task a row came from,
#: so vendoring's per-subtask overlap floor -- the guard that catches one subtask drifting
#: while the dataset as a whole stays above 90% -- would have nothing to group by. Taking
#: the label from the bridge is also the stronger reading of where a row belongs: it is
#: the calibration's own statement rather than something re-derived from an enumeration.
V2_BRIDGE_COLUMNS = ("atlas_idx", "item_id", "subtask", "split_index", "question")


@dataclass(frozen=True)
class LeaderboardSource:
    """One pinned v1 details file, and the harness config whose order it records.

    Attributes:
        repo: The v1 details dataset on the Hub. The v1 leaderboard's per-model repos
            were moved to the ``open-llm-leaderboard-old`` namespace when v2 took the
            original one, and the v2 repos that now live under ``open-llm-leaderboard``
            are a different benchmark suite in a different layout.
        revision: The commit the order was read at. Pinned so a rebuild reproduces the
            committed bridge rather than whatever the repo says later.
        path: The details file inside that repo.
        config: The harness task config the file records, as the leaderboard named it.
    """

    repo: str
    revision: str
    path: str
    config: str


#: The v1 detail repo each bridge's ordering is read from.
#:
#: One model supplies all four, which is deliberate: the order is a property of the
#: leaderboard's evaluation, not of the model evaluated, and drawing the four from one
#: pinned commit means one thing to re-verify rather than four. That it *is* a property
#: of the leaderboard is checked rather than assumed -- see the ``cross_checked`` list
#: in each committed provenance sidecar, six models spanning July 2023 to 2024 whose
#: example order is byte-identical on all four configs.
#:
#: This particular model is pinned because its ARC file predates a v1 schema change and
#: still carries native ARC ids in ``example`` where later repos carry the rendered
#: prompt. Nothing here needs those ids, but the control in
#: :func:`verify_against_shipped` does, and a control run against a different commit
#: from the bridges it validates is not a control.
LEADERBOARD_ORDER: dict[str, LeaderboardSource] = {
    "arc_challenge": LeaderboardSource(
        repo="open-llm-leaderboard-old/details_Corianas__Quokka_2.7b",
        revision="05ea88af3c1503f9949e054ebb930a13d532f207",
        path=(
            "2023-07-19T15:58:12.174583/"
            "details_harness|arc:challenge|25_2023-07-19T15:58:12.174583.parquet"
        ),
        config="harness_arc_challenge_25",
    ),
    "hellaswag": LeaderboardSource(
        repo="open-llm-leaderboard-old/details_Corianas__Quokka_2.7b",
        revision="05ea88af3c1503f9949e054ebb930a13d532f207",
        path=(
            "2023-07-19T15:58:12.174583/"
            "details_harness|hellaswag|10_2023-07-19T15:58:12.174583.parquet"
        ),
        config="harness_hellaswag_10",
    ),
    "winogrande": LeaderboardSource(
        repo="open-llm-leaderboard-old/details_Corianas__Quokka_2.7b",
        revision="05ea88af3c1503f9949e054ebb930a13d532f207",
        path=(
            "2023-09-18T03-05-58.053951/"
            "details_harness|winogrande|5_2023-09-18T03-05-58.053951.parquet"
        ),
        config="harness_winogrande_5",
    ),
    "gsm8k": LeaderboardSource(
        repo="open-llm-leaderboard-old/details_Corianas__Quokka_2.7b",
        revision="05ea88af3c1503f9949e054ebb930a13d532f207",
        path=(
            "2023-09-18T03-05-58.053951/details_harness|gsm8k|5_2023-09-18T03-05-58.053951.parquet"
        ),
        config="harness_gsm8k_5",
    ),
}

#: Models whose example order was compared against the pinned one before it was trusted.
CROSS_CHECKED = (
    "open-llm-leaderboard-old/details_mistralai__Mistral-7B-v0.1",
    "open-llm-leaderboard-old/details_microsoft__phi-2",
    "open-llm-leaderboard-old/details_google__gemma-7b",
    "open-llm-leaderboard-old/details_Qwen__Qwen1.5-7B",
    "open-llm-leaderboard-old/details_tiiuae__falcon-7b",
)


def hellaswag_preprocess(text: str) -> str:
    """Reproduce ``lm_eval.tasks.hellaswag.preprocess`` as the v1 leaderboard ran it.

    Deliberately not the olmo-eval task's ``_preprocess``, which is the same function
    with one substitution widened: olmo-eval strips an optional period before the
    ``[title]`` marker, so a context reading ``supplies. [title] Gather`` becomes
    ``supplies. Gather`` there and ``supplies.. Gather`` here. Roughly 8% of the
    validation split differs on that one character.

    Both renderings are correct for their own harness, and this one is the only one that
    can be matched against what the leaderboard stored. Using olmo-eval's here would
    silently strand those rows: the bridge would be built from the fraction that happened
    to agree and the count check would catch it, but only after the mistake had been
    made to look like a data problem.
    """
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text


def hellaswag_example(doc: dict[str, Any]) -> str:
    """Return the query the v1 harness wrote for one HellaSwag document."""
    context = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
    return hellaswag_preprocess(doc["activity_label"] + ": " + context)


@dataclass(frozen=True)
class LeaderboardRecipe:
    """How to identify one v1 example, from both sides of the match.

    Attributes:
        columns: The details-file columns that together identify an example. Read from
            the parquet and flattened into a tuple of strings, so a list column such as
            HellaSwag's ``choices`` contributes one element per entry.
        from_document: Rebuilds the same tuple from a HuggingFace document, reproducing
            what the v1 harness would have written.
    """

    columns: tuple[str, ...]
    from_document: Callable[[dict[str, Any]], tuple[str, ...]]


#: How to identify a v1 example, per dataset.
#:
#: The details file holds whatever the v1 harness thought identified an example, and it
#: varies: a native id where the dataset has one, the rendered question where it does
#: not. ARC is the first case, and having its native ids is what let the shipped ARC
#: bridge be reproduced item for item. The other three are the second, so their bridges
#: rest on matching text -- which is sound because the match is required to be a
#: bijection over the whole split rather than merely to find a candidate per row.
#:
#: HellaSwag needs its choices in the key and the requirement is not cosmetic. Twenty-two
#: of its validation contexts recur with different endings, so a bridge built on the stem
#: alone would have several documents competing for each of those rows and would pick by
#: whichever happened to be found first.
LEADERBOARD_RECIPES: dict[str, LeaderboardRecipe] = {
    "arc_challenge": LeaderboardRecipe(
        columns=("example",),
        from_document=lambda doc: (str(doc["id"]),),
    ),
    "hellaswag": LeaderboardRecipe(
        columns=("example", "choices"),
        from_document=lambda doc: (
            hellaswag_example(doc),
            *(hellaswag_preprocess(ending) for ending in doc["endings"]),
        ),
    ),
    "winogrande": LeaderboardRecipe(
        columns=("example",),
        from_document=lambda doc: (str(doc["sentence"]),),
    ),
    "gsm8k": LeaderboardRecipe(
        columns=("example",),
        from_document=lambda doc: (str(doc["question"]),),
    ),
}


@dataclass(frozen=True)
class LeaderboardV2Source:
    """The one v2 evaluation run every Route B bridge reads its documents from.

    Attributes:
        repo: A per-model details dataset in the current ``open-llm-leaderboard``
            namespace. Not the ``-old`` one, which holds v1 under a different layout.
            These are gated ``auto``, so an authenticated user has to have accepted
            the terms once per repo before any file in it can be read.
        revision: The commit the documents were read at, so a rebuild reproduces the
            committed bridge rather than whatever the repo says later.
        run: The ISO timestamp naming the evaluation whose files are used. One model
            can carry several runs, and mixing them would read one benchmark's order
            from one evaluation and another's from a different one.

    One model supplies every benchmark, deliberately: a leaderboard run evaluates all
    of them together, and drawing them from one pinned commit means one thing to
    re-verify rather than five. That the documents are a property of the *harness* and
    not of the model is checked rather than assumed, by
    :func:`cross_check_documents` against :data:`V2_CROSS_CHECKED`.
    """

    repo: str
    revision: str
    run: str


#: The v2 details run pinned for every Route B rebuild.
#:
#: This model is pinned rather than a more familiar one because it is in the harvest the
#: banks were fit from -- it appears in each experiment's ``split_models.csv`` -- and
#: because its run is the earliest of the four compared, which makes the cross-check span
#: the leaderboard's whole life rather than a few weeks of it.
V2_SOURCE = LeaderboardV2Source(
    repo="open-llm-leaderboard/microsoft__Phi-3-mini-4k-instruct-details",
    revision="427d64648094a1aea773aec93fe5e3b5ca069074",
    run="2024-07-18T11-04-02.101450",
)

#: Details repos whose ``doc_id -> document`` map is compared against the pinned one.
#:
#: July 2024 to February 2025, which is very nearly the leaderboard's whole run. The
#: comparison is what licenses reading an ordering off one arbitrary model: if two
#: evaluations months apart disagreed about which document ``doc_id`` 137 names, then
#: ``doc_id`` is not a property of the benchmark and no bridge can be built on it.
V2_CROSS_CHECKED = (
    "open-llm-leaderboard/Qwen__Qwen2.5-7B-Instruct-details",
    "open-llm-leaderboard/Danielbrdz__Barcenas-R1-Qwen-1.5b-details",
    "open-llm-leaderboard/DeepMount00__Qwen2-1.5B-Ita-details",
)

#: How a benchmark's subtasks are enumerated on the olmo-eval side.
#:
#: The three cases are not stylistic. ``PER_SUBTASK_TASK`` is a bank whose spec declares
#: one registered task per subtask. ``WHOLE_TASK`` is a single-task bank whose bridge
#: names one subtask. ``COMPOSITE_ID`` is the awkward middle -- leaderboard_math's task
#: concatenates seven subjects and emits ``<subject>|<position>`` as its own id -- where
#: the per-subtask enumeration has to be recovered from that id rather than from the
#: registry.
PER_SUBTASK_TASK = "per_subtask_task"
WHOLE_TASK = "whole_task"
COMPOSITE_ID = "composite_id"


@dataclass(frozen=True)
class LeaderboardV2Recipe:
    """Everything benchmark-specific about rebuilding one Route B bridge.

    This is the table a new benchmark is brought through by. Nothing outside it should
    need editing, which is the point: five banks share one recovery and differ only in
    where the question text lives and how their subtasks map onto registered tasks.

    Attributes:
        task_prefix: Joined to a bridge subtask label to give the leaderboard task name:
            ``leaderboard_musr_`` + ``murder_mysteries``. A benchmark the leaderboard
            evaluated as a single task splits its name across the two, ``leaderboard_``
            here and ``ifeval`` in :attr:`single_subtask`.
        index_map: Path on the source ref of the file mapping ``atlas_idx`` to a
            subtask and a ``doc_id``. Usually the local fit's ``item_id_map.csv``;
            ifeval is the exception, its bank keyed by the bare integer in
            ``Inputs/ATLAS/ifeval/atlas_idx_to_question_id.csv``.
        doc_id_column: The column of that file holding lm-eval's ``doc_id``.
        subtask_column: The column naming the subtask, empty for a bank whose bridge
            covers one.
        single_subtask: The label to attribute every row to when there is no
            :attr:`subtask_column`. Must be the label the leaderboard task is named
            after, since :attr:`task_prefix` is joined to it.
        doc_fields: The keys of the details record's ``doc`` that :attr:`from_record`
            reads. Audit only -- it is what a reader checks the recipe against, and
            what ``--inspect`` prints beside the schema it found.
        from_record: The identity of one evaluated example, built from the details
            record's ``doc``. This has to reproduce the *olmo-eval task's* rendering,
            not the leaderboard's prompt: the two differ, and the task's is the side
            the item id is computed from.
        from_instance: The same identity built from an olmo-eval instance. Kept
            separate from the item id so a benchmark whose id folds in a rendering
            decision -- gpqa's shuffled choice block -- can still be matched on the
            question it came from.
        enumeration: One of :data:`PER_SUBTASK_TASK`, :data:`WHOLE_TASK`,
            :data:`COMPOSITE_ID`.
        normalization: What :attr:`from_record` had to do to reach the task's
            rendering, copied verbatim into the provenance sidecar. Recorded in prose
            because it is the one part of a bridge a later reader cannot recover by
            re-running anything: a recipe that silently loosened a comparison and a
            recipe that reproduced a conversion produce the same passing build.
    """

    task_prefix: str
    index_map: str
    doc_fields: tuple[str, ...]
    from_record: Callable[[dict[str, Any]], tuple[str, ...]]
    from_instance: Callable[[Any], tuple[str, ...]]
    enumeration: str
    normalization: str
    doc_id_column: str = "question_id"
    subtask_column: str = "subtask"
    single_subtask: str = ""

    def __post_init__(self) -> None:
        """Reject a half-written entry at import rather than part-way through a build.

        Both mistakes are ones a new entry makes and neither fails where it was made. An
        index map with no subtask column and no label to fall back on attributes every
        calibrated row to the empty subtask, which then matches no declared task and
        reads as the spec being stale. An unknown enumeration is not discovered until
        after the details files have been read and cross-checked, which on bbh is several
        minutes of downloads before the typo surfaces.
        """
        if not self.subtask_column and not self.single_subtask:
            raise ValueError(
                f"{self.index_map}: an index map with no subtask column needs "
                f"single_subtask to name the one subtask its rows belong to."
            )
        if self.enumeration == WHOLE_TASK and not self.single_subtask:
            raise ValueError(
                f"{self.index_map}: enumeration {WHOLE_TASK} means one task covers the "
                f"whole bank, so single_subtask has to name the label its rows carry."
            )
        if self.enumeration not in (PER_SUBTASK_TASK, WHOLE_TASK, COMPOSITE_ID):
            raise ValueError(
                f"{self.index_map}: unknown enumeration {self.enumeration!r}, expected "
                f"one of {PER_SUBTASK_TASK}, {WHOLE_TASK}, {COMPOSITE_ID}."
            )


def question_and_choices(instance: Any) -> tuple[str, ...]:
    """Return the identity of an olmo-eval instance as its stem and its choice texts.

    The default for a multiple-choice benchmark, and it is the same content
    :func:`~..datasets.content_item_id` hashes. Choices belong in it because a stem can
    recur: HellaSwag has 22 contexts that do, and a bridge built on the stem alone would
    have several documents competing for one row and pick by whichever came first.
    """
    return (str(instance.question), *(str(choice) for choice in instance.choices or ()))


def musr_document(doc: dict[str, Any]) -> tuple[str, ...]:
    """Return the identity of one MuSR document, as the olmo-eval task renders it.

    The details record stores the raw dataset row rather than the prompt, so the two
    conversions the task performs have to be repeated here: the stem is the narrative
    and the question joined by a blank line, and ``choices`` is the *repr of a Python
    list* rather than a sequence, which upstream stores that way and both sides have to
    parse rather than read.

    Nothing else is normalized and nothing should be. The leaderboard's own rendering --
    the numbered choice block, the ``Answer:`` cue and the model's chat template -- lives
    in the record's ``arguments`` and is a different string from the one this bank's
    items are keyed by; matching against it would need a normalization step, and a
    normalization step is where a mismatch stops being visible.
    """
    return (
        f"{doc['narrative']}\n\n{doc['question']}",
        *(str(choice) for choice in ast.literal_eval(doc["choices"])),
    )


def gpqa_document(doc: dict[str, Any]) -> tuple[str, ...]:
    """Return the identity of one GPQA document, as the olmo-eval task renders it.

    The stem alone, and the reason is the shuffle. ``GPQATask.process_doc`` reorders the
    four options from ``Random(f"{seed}:{index}")``, a seed of the *task's* choosing, and
    the details record carries the leaderboard's own shuffle in ``choice1..4`` instead.
    The two orderings are unrelated, so a key including the choices would leave every
    document unmatched while looking like a data problem. What is matched is the one
    thing both sides agree on, which is the question.

    ``_clean_text`` is imported from the task rather than reproduced here, unlike
    :func:`hellaswag_preprocess`, and the difference is deliberate: there the two
    harnesses genuinely render differently and only lm-eval's version can be matched,
    while here the target *is* the task's rendering, so a second copy of it would be free
    to drift and would fail as an unmatched document long after the edit that caused it.
    """
    from olmo_eval.evals.tasks.gpqa import _clean_text

    return (_clean_text(str(doc["Question"])),)


def question_only(instance: Any) -> tuple[str, ...]:
    """Return the identity of an olmo-eval instance as its stem alone.

    For the two benchmarks whose choices are not a statement about which question an
    instance is. GPQA reshuffles its four options per question from the *task's* own
    seed, so the instance's choice order cannot be rebuilt from the raw doc and folding
    it in would leave every document unmatched. BBH attaches one hardcoded choice set to
    every item of a subtask, so the choices are constant within the split being matched
    and carry no information at all.

    Narrower than :func:`question_and_choices` and therefore checked rather than
    assumed: :func:`match_documents` refuses a split where two instances share an
    identity, so a benchmark whose stems recur would abort here instead of picking
    whichever instance came first. The item id is computed separately, from the matched
    instance's full text, so nothing about the key is loosened by this.
    """
    return (str(instance.question),)


#: How to rebuild each Route B bridge, and the only place a new benchmark is declared.
#:
#: An entry is added after ``--inspect`` has shown what the details records actually
#: hold, because the doc schema is the one thing here that cannot be reasoned out: it is
#: whatever lm-evaluation-harness handed the leaderboard, which for gpqa is 70-odd
#: annotation columns and for ifeval is a prompt beside its verifier arguments.
LEADERBOARD_V2: dict[str, LeaderboardV2Recipe] = {
    "ifeval": LeaderboardV2Recipe(
        task_prefix="leaderboard_",
        index_map="AdaptiveTesting/Inputs/ATLAS/ifeval/atlas_idx_to_question_id.csv",
        doc_fields=("prompt",),
        from_record=lambda doc: (str(doc["prompt"]),),
        from_instance=question_and_choices,
        enumeration=WHOLE_TASK,
        subtask_column="",
        single_subtask="ifeval",
        normalization=(
            "none. The details record's doc is the raw wis-k/instruction-following-eval "
            "row and the olmo-eval task passes its prompt through unaltered, so the two "
            "renderings are the same string. IFEval is the one benchmark where that is "
            "forced rather than convenient: a prompt names verifiable instructions about "
            "its own text -- no commas, 300+ words, a fixed number of highlighted "
            "sections -- so any normalization applied to match it would also be a change "
            "to what the verifiers were told to check."
        ),
    ),
    "leaderboard_math": LeaderboardV2Recipe(
        task_prefix="leaderboard_math_",
        index_map="AdaptiveTesting/Experiments/openlm_atlas_3pl/math/data/item_id_map.csv",
        doc_fields=("problem",),
        from_record=lambda doc: (str(doc["problem"]),),
        from_instance=question_and_choices,
        enumeration=COMPOSITE_ID,
        normalization=(
            "none. The details record's doc is the raw MATH-lighteval row and the "
            "olmo-eval task's stem is its problem field unaltered, so the two renderings "
            "are the same string; the 4-shot Minerva block and the Problem:/Solution: "
            "framing are added by format_request and are deliberately not matched. The "
            "one thing this entry depends on beyond the text is the enumeration: the "
            "task concatenates seven subjects and numbers each one's Level-5 subsequence "
            "from zero, emitting <subject>|<position> as its own id, so composite_id "
            "recovers the per-subject enumeration from that id rather than from the "
            "registry -- there is one registered task, not seven."
        ),
    ),
    "gpqa": LeaderboardV2Recipe(
        task_prefix="leaderboard_gpqa_",
        index_map="AdaptiveTesting/Experiments/openlm_gpqa_atlas_3pl/data/item_id_map.csv",
        doc_fields=("Question",),
        from_record=gpqa_document,
        from_instance=question_only,
        enumeration=PER_SUBTASK_TASK,
        normalization=(
            "the task's own _clean_text, imported rather than reimplemented: it replaces "
            "a [title] marker with '. ', collapses runs of two or more spaces, and "
            "strips. The details record's doc is the raw Idavidrein/gpqa row with 70-odd "
            "annotation columns beside the question, and the leaderboard's rendered "
            "prompt is not what is matched. The identity is the question alone, because "
            "GPQATask.process_doc reshuffles the four options from its own seed while "
            "the record carries the leaderboard's shuffle in choice1..4; the two "
            "orderings are unrelated and neither is recoverable from the other. That is "
            "not a loosened match -- the bijection is required over each subset's whole "
            "split, so a repeated stem would abort rather than pick a candidate -- and "
            "the item id is still computed from the matched instance's question and its "
            "own shuffled choices."
        ),
    ),
    "bbh": LeaderboardV2Recipe(
        task_prefix="leaderboard_bbh_",
        index_map="AdaptiveTesting/Experiments/openlm_atlas_3pl/bbh/data/item_id_map.csv",
        doc_fields=("input",),
        from_record=lambda doc: (str(doc["input"]),),
        from_instance=question_only,
        enumeration=PER_SUBTASK_TASK,
        normalization=(
            "none. The details record's doc is the raw SaylorTwift/bbh row and the "
            "olmo-eval task's question is its input field unaltered. Neither side of "
            "this match is the string the bank's stems hold: this dataset is "
            "frozen_prompt, so vendoring keeps the rendered 3-shot prompt -- the "
            "subtask's description, three fixed exemplars, the item and the trailing A: "
            "cue -- and matching that would compare a per-subtask framing against a bare "
            "item. The identity is therefore the instance's question rather than its "
            "stem. The choices are left out for a different reason: leaderboard v2 "
            "attaches one hardcoded choice set to every item of a subtask, so within the "
            "split being matched they are constant and distinguish nothing."
        ),
    ),
    "musr": LeaderboardV2Recipe(
        task_prefix="leaderboard_musr_",
        index_map="AdaptiveTesting/Experiments/openlm_atlas_3pl/musr/data/item_id_map.csv",
        doc_fields=("narrative", "question", "choices"),
        from_record=musr_document,
        from_instance=question_and_choices,
        enumeration=PER_SUBTASK_TASK,
        normalization=(
            "none. The details record's doc is the raw TAUR-Lab/MuSR row, so the two "
            "sides differ only by the conversions the olmo-eval task performs on it: "
            "narrative and question joined by a blank line, and the choices column parsed "
            "out of its Python-list repr. The leaderboard's rendered prompt -- the "
            "numbered choice block, the Answer: cue and the model's chat template -- is in "
            "the record's arguments field and is deliberately not what is matched, since "
            "it is per model and is not the string this bank's items are keyed by."
        ),
    ),
}


def use_system_trust_store() -> None:
    """Route TLS verification through the OS trust store, when available.

    Same reason as ``vendor_bank._use_system_trust_store``: behind a TLS-inspecting
    proxy the bundled CA list lacks the intercepting root and every Hub request fails
    with ``CERTIFICATE_VERIFY_FAILED``. Optional -- absent the package, verification
    proceeds with the default bundle.
    """
    try:
        import truststore
    except ImportError:
        return
    truststore.inject_into_ssl()


def read_leaderboard_order(
    source: LeaderboardSource, recipe: LeaderboardRecipe
) -> list[tuple[str, ...]]:
    """Return each row's identifying tuple, in the order the leaderboard evaluated.

    Read through :class:`~huggingface_hub.HfFileSystem` into memory rather than
    downloaded. The v1 details files are named after a harness config and an ISO
    timestamp -- ``details_harness|gsm8k|5_2023-09-18T03-05-58.053951.parquet``, inside a
    directory named ``2023-07-19T15:58:12.174583`` -- and neither the colons nor the
    pipes are legal in a Windows path, so ``hf_hub_download`` cannot place the file on
    disk at all. It fails while creating the cache directory, with a ``WinError 123``
    that names the syntax rather than the character.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    uri = f"datasets/{source.repo}@{source.revision}/{source.path}"
    with HfFileSystem().open(uri, "rb") as handle:
        table = pq.read_table(handle, columns=list(recipe.columns))
    return [flatten(row[column] for column in recipe.columns) for row in table.to_pylist()]


def flatten(values: Any) -> tuple[str, ...]:
    """Return one flat tuple of strings from a row's identifying columns."""
    parts: list[str] = []
    for value in values:
        if isinstance(value, (list, tuple)):
            parts.extend(str(item) for item in value)
        else:
            parts.append(str(value))
    return tuple(parts)


def task_documents(task: Any) -> list[dict[str, Any]]:
    """Return the raw documents of the split ``task`` enumerates, in enumeration order.

    Vendoring joins against ``task.instances``, but the leaderboard's ``example`` was
    written from the raw document, and for HellaSwag the two renderings differ. So the
    match runs on documents and the id is taken from the instance beside it, which makes
    :func:`check_documents_align` the load-bearing step rather than an assertion.
    """
    from olmo_eval.data import DataLoader

    source = task.config.get_data_source(split=task.config.split.value)
    return list(DataLoader().load(source))


def check_documents_align(dataset: str, docs: list[dict[str, Any]], instances: list[Any]) -> None:
    """Abort unless document *i* and instance *i* are the same item.

    The bridge is built by matching leaderboard text against ``docs`` and then reading
    the item id off ``instances``, so the two lists being the same items in the same
    order is the assumption the whole artifact rests on. It holds because all three
    tasks map their split one-to-one and filter nothing, and if one ever began dropping
    a malformed row the lists would shift past each other and every id from the drop
    onward would name its neighbour's question -- the exact failure this bridge exists
    to repair, reintroduced one layer down.

    Length alone would not catch that. Where a task exposes the split position it
    enumerated as ``metadata["index"]`` the pairing is checked row by row instead.
    """
    if len(docs) != len(instances):
        raise SystemExit(
            f"{dataset}: the task enumerated {len(instances)} instances from a split of "
            f"{len(docs)} documents. The bridge reads its match from the documents and "
            f"its ids from the instances, so it can only pair them when the task keeps "
            f"every row. Nothing was written."
        )
    for position, instance in enumerate(instances):
        index = (instance.metadata or {}).get("index")
        if index is not None and int(index) != position:
            raise SystemExit(
                f"{dataset}: instance {position} reports metadata['index'] {index}, so "
                f"the task is not enumerating its split in order and document "
                f"{position} is not the item this instance renders. Nothing was written."
            )


def match_order(
    dataset: str, examples: list[tuple[str, ...]], docs: list[dict[str, Any]]
) -> list[int]:
    """Return the split position of each leaderboard example, or abort.

    Required to be a bijection, and that requirement is doing the real work. Any
    permutation of the split satisfies "every ATLAS column found something", which is
    exactly how the superseded generator passed its own guard; what distinguishes a
    recovered order from an invented one is that every example matches a *distinct*
    document and every document is claimed. A recipe that rendered the leaderboard's
    text incorrectly, or a split that has been revised since 2023, fails here rather
    than producing a bridge that is right for most rows.
    """
    recipe = LEADERBOARD_RECIPES[dataset]
    keys = [recipe.from_document(doc) for doc in docs]

    repeated = [key for key, n in Counter(keys).items() if n > 1]
    if repeated:
        raise SystemExit(
            f"{dataset}: the v1 example identity built from {list(recipe.columns)} is not "
            f"unique over the split -- {len(repeated)} value(s) appear more than once, the "
            f"first being {repeated[0][0][:120]!r}. A leaderboard row matching it could name "
            f"either document, so the order cannot be recovered from these columns. Widen "
            f"the recipe. Nothing was written."
        )

    position_of = {key: position for position, key in enumerate(keys)}
    order: list[int] = []
    unmatched: list[tuple[str, ...]] = []
    for example in examples:
        position = position_of.get(example)
        if position is None:
            unmatched.append(example)
            continue
        order.append(position)

    if unmatched:
        raise SystemExit(
            f"{dataset}: {len(unmatched)} of {len(examples)} leaderboard examples match "
            f"no document in the split this task enumerates. The first is "
            f"{unmatched[0][0][:200]!r}. Either the recipe rebuilding the v1 example text "
            f"is wrong or the split has been revised; a bridge built from the rows that "
            f"did match would be a bridge over a different item set. Nothing was written."
        )
    if len(set(order)) != len(docs):
        duplicated = [position for position, n in Counter(order).items() if n > 1]
        raise SystemExit(
            f"{dataset}: the match is not a bijection -- {len(order)} examples claim "
            f"{len(set(order))} of {len(docs)} documents, with {len(duplicated)} claimed "
            f"more than once. Nothing was written."
        )
    return order


def stem(text: str) -> str:
    """Return the truncated single-line stem a bridge row carries for auditing."""
    collapsed = " ".join(text.split())
    return collapsed[:STEM_CHARS]


def build_rows(dataset: str, order: list[int], instances: list[Any]) -> list[dict[str, Any]]:
    """Return the bridge rows, aborting if two items hash to one id.

    Uniqueness is checked here rather than left to vendoring's ambiguous-id drop. That
    drop exists for a bridge we inherited and cannot fix; this is a bridge being written
    now, and a collision in it means :func:`~..datasets.content_item_id` is keying on too
    little of the item -- a defect to repair rather than a loss to absorb.
    """
    rows: list[dict[str, Any]] = []
    for atlas_idx, position in enumerate(order, start=1):
        instance = instances[position]
        choices = tuple(instance.choices or ())
        rows.append(
            {
                "atlas_idx": atlas_idx,
                "item_id": content_item_id(instance.question, choices),
                "split_index": position,
                "question": stem(instance.question),
            }
        )

    collisions = [
        item_id for item_id, n in Counter(row["item_id"] for row in rows).items() if n > 1
    ]
    if collisions:
        raise SystemExit(
            f"{dataset}: {len(collisions)} content hash(es) name more than one item, the "
            f"first being {collisions[0]}. Two distinct items sharing an id would both be "
            f"dropped as ambiguous at vendoring time, so widen the key rather than ship "
            f"this. Nothing was written."
        )
    return rows


def git_show(ref: str, path: str) -> str:
    """Return the contents of ``path`` at ``ref`` without checking anything out."""
    completed = subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(
            f"git show {ref}:{path} failed (exit {completed.returncode}).\n"
            f"{completed.stderr.strip()}"
        )
    return completed.stdout


def verify_against_shipped(
    rows: list[dict[str, Any]],
    order: list[int],
    instances: list[Any],
    *,
    source_ref: str,
    shipped_path: str,
) -> dict[str, Any]:
    """Compare a rebuilt bridge against one the ATLAS release shipped.

    This is the whole justification for the other three bridges. ARC-Challenge is the
    only benchmark ATLAS shipped a bridge for, so it is the only one where the recovered
    order can be checked against a known answer, and applying an unvalidated recovery to
    three banks that cannot be checked would repeat the mistake being repaired -- a
    different wrong bridge is not an improvement on a known wrong one.

    Agreement is reported twice because the two numbers answer different questions and
    only together are they honest. ``positions_agreeing`` resolves the shipped
    ``question_id`` to the instance it names and compares items, which is the strict
    reading. ``positions_agreeing_on_stem`` compares against the stem the shipped bridge
    carries in its third column, which is the reading that isolates the *ordering* --
    and the ordering is what the other three bridges borrow.

    The two numbers differ by exactly the two positions where the shipped bridge is
    itself wrong, so the gap is worth recording rather than rounding away. ATLAS resolved
    its ids by matching stem text; ARC-Challenge holds two questions whose stems are
    duplicated under a second native id; and both collided onto the wrong twin, which is
    why the shipped file contains two ids twice and never contains ``TIMSS_2003_8_pg47``
    or ``Mercury_7116183`` at all. Vendoring drops a duplicated id as ambiguous, so those
    positions were already unusable.
    """
    shipped = list(csv.DictReader(io.StringIO(git_show(source_ref, shipped_path))))
    by_native_id = {str((instance.metadata or {}).get("id")): instance for instance in instances}

    agreed = 0
    on_stem = 0
    disagreements: list[dict[str, str]] = []
    for row, position, shipped_row in zip(rows, order, shipped, strict=True):
        recovered = instances[position]
        native_id = shipped_row["question_id"].strip()
        named = by_native_id.get(native_id)

        if named is not None and row["item_id"] == content_item_id(
            named.question, tuple(named.choices or ())
        ):
            agreed += 1
        else:
            disagreements.append(
                {
                    "atlas_idx": str(row["atlas_idx"]),
                    "shipped_question_id": native_id,
                    "recovered_question_id": str((recovered.metadata or {}).get("id")),
                    "stems_identical": str(
                        stem(recovered.question) == stem(shipped_row["question"])
                    ),
                }
            )
        if stem(recovered.question) == stem(shipped_row["question"]):
            on_stem += 1

    duplicated = sorted(
        {
            native
            for native, n in Counter(r["question_id"].strip() for r in shipped).items()
            if n > 1
        }
    )
    result = {
        "shipped_bridge": f"{source_ref}:{shipped_path}",
        "positions_compared": len(shipped),
        "positions_agreeing": agreed,
        "agreement": round(agreed / len(shipped), 6) if shipped else 0.0,
        "positions_agreeing_on_stem": on_stem,
        "ids_duplicated_in_shipped_bridge": duplicated,
        "disagreements": disagreements,
    }
    log.info(
        "Control against %s: %d of %d positions agree on the item (%.4f%%), %d of %d on "
        "the stem. The shipped bridge duplicates %d id(s).",
        shipped_path,
        agreed,
        len(shipped),
        100 * agreed / len(shipped) if shipped else 0.0,
        on_stem,
        len(shipped),
        len(duplicated),
    )
    return result


def write_bridge(
    dataset: str,
    rows: list[dict[str, Any]],
    provenance: dict[str, Any],
    *,
    fieldnames: Sequence[str] = BRIDGE_COLUMNS,
) -> None:
    """Write the bridge CSV and its provenance sidecar."""
    out_dir = REPO_ROOT / BRIDGES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"{dataset}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)

    json_path = out_dir / f"{dataset}.json"
    json_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    log.info("Wrote %d rows to %s and provenance to %s", len(rows), csv_path, json_path)


def write_control(provenance: dict[str, Any]) -> None:
    """Write the ARC control record, and deliberately no bridge beside it.

    ARC-Challenge is rebuilt to be checked, never to be used. Its spec stays pointed at
    the bridge the ATLAS release shipped, which is the bank in this file whose join is
    independently attested -- item p-value against implied difficulty at Spearman -0.857
    -- and swapping a verified artifact for a reproduction of it buys two recovered items
    and risks the one bank the others are calibrated against. Emitting a CSV here anyway
    would leave a second ARC bridge in the directory with nothing to distinguish it from
    the three that are live.
    """
    out_dir = REPO_ROOT / BRIDGES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "arc_challenge.control.json"
    path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    log.info("Wrote the control record to %s; ARC keeps the bridge the release shipped.", path)


#: A per-model v2 sample file, whose name carries the leaderboard task and the run.
#:
#: Only ``.jsonl`` is matched. The early v2 repos also carry ``.json`` files from runs
#: superseded within weeks, and the harvest these banks were fit from read the ``.jsonl``
#: ones; a bridge built off a superseded run would be describing an evaluation nothing
#: else in this pipeline refers to.
V2_SAMPLE_FILE = re.compile(r"samples_(leaderboard_[a-z0-9_]+)_(\d{4}-\d{2}-\d{2}T[\d.-]+)\.jsonl$")


def list_details_files(repo: str, revision: str | None) -> list[str]:
    """Return every file in a v2 details repo at ``revision``."""
    from huggingface_hub import HfApi

    return list(HfApi().list_repo_files(repo, repo_type="dataset", revision=revision))


def details_path(repo: str, files: list[str], task: str, *, run: str = "") -> str:
    """Return the sample file for one leaderboard task, or abort saying what was found.

    Resolved from the repo listing rather than templated, because a details repo holds
    one directory per model and the run timestamp differs per repo, so the only way to
    name the file for a cross-checked model is to look. Where ``run`` is given the
    resolved file has to belong to it: a repo can hold several evaluations of the same
    model, and taking the newest of those for one benchmark and the newest for another
    would read two benchmarks' orders from two different evaluations.
    """
    matches = [
        path
        for path in files
        if (found := V2_SAMPLE_FILE.search(path.rsplit("/", 1)[-1])) is not None
        and found.group(1) == task
        and (not run or found.group(2) == run)
    ]
    if not matches:
        available = sorted(
            {
                found.group(1)
                for path in files
                if (found := V2_SAMPLE_FILE.search(path.rsplit("/", 1)[-1])) is not None
            }
        )
        raise SystemExit(
            f"{repo} has no samples file for leaderboard task {task!r}"
            + (f" in run {run}" if run else "")
            + f". It carries {len(available)} task(s), the first being {available[:4]}. "
            f"Either the task_prefix in LEADERBOARD_V2 is wrong or this run did not "
            f"evaluate the benchmark. Nothing was written."
        )
    return sorted(matches)[-1]


def read_details(repo: str, revision: str | None, path: str) -> dict[int, dict[str, Any]]:
    """Return ``doc_id -> document`` from one v2 sample file.

    Keyed by the record's explicit ``doc_id`` and never by its position in the file,
    which is the single most important thing about this format. lm-eval writes samples
    in completion order, not in document order -- MuSR's murder_mysteries file opens
    with doc_ids 0, 8, 16, 24 -- so reading row order as the enumeration would produce a
    scrambled bridge that passed every count and bijection check in this module.

    Read through :class:`~huggingface_hub.HfFileSystem` into memory rather than
    downloaded. A build reads every subtask of the pinned model and of each cross-checked
    one -- 120 files for bbh -- and wants none of it afterwards, so streaming keeps a
    rebuild from leaving a cache behind that dwarfs the artifact it produced.
    """
    from huggingface_hub import HfFileSystem

    uri = f"datasets/{repo}@{revision}/{path}" if revision else f"datasets/{repo}/{path}"
    with HfFileSystem().open(uri, "r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]

    documents: dict[int, dict[str, Any]] = {}
    for record in records:
        doc_id = record.get("doc_id")
        document = record.get("doc")
        if doc_id is None or not isinstance(document, dict):
            raise SystemExit(
                f"{repo}:{path} holds a record with no doc_id or no doc. This reader "
                f"depends on doc_id being an explicit field, because the file is not "
                f"written in document order and there is nothing else to key on. "
                f"Nothing was written."
            )
        documents[int(doc_id)] = document

    if len(documents) != len(records):
        raise SystemExit(
            f"{repo}:{path} has {len(records)} records over only {len(documents)} "
            f"distinct doc_ids, so a doc_id names more than one evaluated example and "
            f"cannot identify a question. Nothing was written."
        )
    return documents


def cross_check_documents(
    dataset: str,
    recipe: LeaderboardV2Recipe,
    labels: Sequence[str],
    pinned: dict[str, dict[int, tuple[str, ...]]],
) -> list[dict[str, Any]]:
    """Compare the pinned ``doc_id -> example`` map against other leaderboard runs.

    The whole justification for reading an ordering off one arbitrary model. ``doc_id``
    is a position in an enumeration lm-eval performed, and a position is only a usable
    key if the enumeration is the same one every time; that is very likely true and it
    is exactly the kind of very likely that produced the bridges this script replaces.
    Comparing runs months apart is the cheapest way to make it a measurement.

    What is compared is the identity tuple rather than the whole record, because the
    identity is what the bridge rests on. A details schema that gained an annotation
    column between runs would fail a byte comparison of the document and would change
    nothing about which question ``doc_id`` names.

    Any disagreement aborts. There is no sensible way to average two orderings, and a
    bridge built from the pinned one while a second run says otherwise would be a bridge
    whose own provenance record contradicts it.
    """
    results: list[dict[str, Any]] = []
    for repo in V2_CROSS_CHECKED:
        files = list_details_files(repo, None)
        compared = 0
        run = ""
        for label in labels:
            path = details_path(repo, files, recipe.task_prefix + label)
            found = V2_SAMPLE_FILE.search(path.rsplit("/", 1)[-1])
            run = found.group(2) if found else ""
            other = read_details(repo, None, path)
            mine = pinned[label]

            shared = sorted(set(mine) & set(other))
            if set(mine) != set(other):
                raise SystemExit(
                    f"{dataset}: {repo} evaluated {len(other)} documents for subtask "
                    f"{label!r} where the pinned run evaluated {len(mine)}, so the two "
                    f"runs do not agree on what the split is. Nothing was written."
                )
            for doc_id in shared:
                if recipe.from_record(other[doc_id]) != mine[doc_id]:
                    raise SystemExit(
                        f"{dataset}: {repo} and the pinned run disagree about which "
                        f"document subtask {label!r} doc_id {doc_id} names. doc_id is "
                        f"therefore not a property of the benchmark and no bridge can "
                        f"be keyed by it. Nothing was written."
                    )
            compared += len(shared)

        log.info("Cross-check: %s agrees on all %d documents (run %s)", repo, compared, run)
        results.append({"repo": repo, "run": run, "documents_agreeing": compared})
    return results


def subtask_instances(
    dataset: str, spec: Any, recipe: LeaderboardV2Recipe, labels: Sequence[str]
) -> dict[str, list[Any]]:
    """Return each subtask's olmo-eval instances, in the order the task enumerates them.

    Three shapes of bank need three answers and the difference is not cosmetic; see
    :data:`PER_SUBTASK_TASK`. What every branch has to produce is the *enumeration*, not
    a set: the item id is read off the instance a document matched, so an instance list
    that skipped or reordered a row would attach a difficulty to the wrong question at
    the one step this module cannot check by counting.

    The ``COMPOSITE_ID`` fall-through is the only branch that trusts something the task
    says about itself, and it is held to a dense ``0..n-1`` run per subtask because that
    id is a position: a gap in it means the task is not enumerating what the details
    file's ``doc_id`` counted, and grouping by it would silently renumber the rest.
    """
    from olmo_eval.evals.tasks.common.registry import get_task

    if recipe.enumeration == PER_SUBTASK_TASK:
        declared = dict(spec.subtasks)
        missing = [label for label in labels if label not in declared]
        if missing:
            raise SystemExit(
                f"{dataset}: the index map names subtask(s) {missing} that spec.subtasks "
                f"does not map to a registered task, so their bank rows would have no "
                f"enumeration to be matched against. Nothing was written."
            )
        return {label: list(get_task(declared[label]).instances) for label in labels}

    if recipe.enumeration == WHOLE_TASK:
        return {recipe.single_subtask: list(get_task(spec.task).instances)}

    grouped: dict[str, dict[int, Any]] = {}
    for instance in get_task(spec.task).instances:
        item_id = str((instance.metadata or {}).get("id", ""))
        label, separator, position = item_id.partition("|")
        if not separator or not position.isdigit():
            raise SystemExit(
                f"{dataset}: enumeration {COMPOSITE_ID} expects the task to emit ids like "
                f"'<subtask>|<position>' and it emitted {item_id!r}. Nothing was written."
            )
        grouped.setdefault(label, {})[int(position)] = instance

    enumerations: dict[str, list[Any]] = {}
    for label, by_position in grouped.items():
        positions = sorted(by_position)
        if positions != list(range(len(positions))):
            raise SystemExit(
                f"{dataset}: the task's own ids for subtask {label!r} are not a dense run "
                f"from zero -- they reach {positions[-1]} over {len(positions)} instances "
                f"-- so they are not the enumeration the details file's doc_id counted. "
                f"Nothing was written."
            )
        enumerations[label] = [by_position[position] for position in positions]
    return enumerations


def match_documents(
    dataset: str,
    label: str,
    recipe: LeaderboardV2Recipe,
    documents: dict[int, tuple[str, ...]],
    instances: Sequence[Any],
) -> dict[int, int]:
    """Return ``doc_id -> enumeration position`` for one subtask, or abort.

    Required to be a bijection, and that is where the work happens. "Every calibrated
    position found a question" is satisfied by any permutation, which is how the
    positional join this replaces passed its own guards; what distinguishes a recovered
    mapping from an invented one is that each evaluated document matches a *distinct*
    instance and every instance is claimed.

    Matching on text rather than loosening it to a prefix or a normalized form is
    deliberate. The two sides read the same HuggingFace rows through the same
    conversions, so an exact match is available; where it is not, the answer is to fix
    the recipe until the two renderings agree, because a match that tolerates a
    difference cannot tell a rendering difference from a different question.
    """
    keys = [recipe.from_instance(instance) for instance in instances]

    repeated = [key for key, n in Counter(keys).items() if n > 1]
    if repeated:
        raise SystemExit(
            f"{dataset}: the identity built from {list(recipe.doc_fields)} is not unique "
            f"over subtask {label!r} -- {len(repeated)} value(s) recur, the first being "
            f"{repeated[0][0][:120]!r}. A leaderboard document matching it could name "
            f"either instance, so the mapping cannot be recovered from these fields. "
            f"Widen the recipe. Nothing was written."
        )

    position_of = {key: position for position, key in enumerate(keys)}
    order: dict[int, int] = {}
    unmatched: list[int] = []
    for doc_id, key in sorted(documents.items()):
        position = position_of.get(key)
        if position is None:
            unmatched.append(doc_id)
            continue
        order[doc_id] = position

    if unmatched:
        example = documents[unmatched[0]][0]
        raise SystemExit(
            f"{dataset}: {len(unmatched)} of {len(documents)} documents the leaderboard "
            f"evaluated for subtask {label!r} match no instance the task enumerates. The "
            f"first is doc_id {unmatched[0]}, reading {example[:200]!r}. Either the "
            f"recipe renders the document differently from the task or the split has "
            f"been revised; a bridge over the rows that did match would be a bridge over "
            f"a different item set. Nothing was written."
        )
    if len(set(order.values())) != len(instances):
        raise SystemExit(
            f"{dataset}: subtask {label!r} is not a bijection -- {len(order)} documents "
            f"claim {len(set(order.values()))} of {len(instances)} instances. Nothing "
            f"was written."
        )
    return order


def read_index_map(
    dataset: str, recipe: LeaderboardV2Recipe, source_ref: str
) -> list[tuple[int, str, int]]:
    """Return ``(atlas_idx, subtask, doc_id)`` for every calibrated bank column.

    The upstream artifact this whole rebuild is re-keying, read rather than replaced:
    which bank column was fit from which harvested response is a fact about the
    calibration and only the fit knows it. What is being replaced is the *other* half,
    the claim that ``doc_id`` can be re-derived by counting an enumeration today.
    """
    rows = list(csv.DictReader(io.StringIO(git_show(source_ref, recipe.index_map))))
    if not rows:
        raise SystemExit(f"{dataset}: {source_ref}:{recipe.index_map} is empty.")

    needed = [recipe.doc_id_column] + ([recipe.subtask_column] if recipe.subtask_column else [])
    absent = [column for column in needed if column not in rows[0]]
    if absent:
        raise SystemExit(
            f"{dataset}: {recipe.index_map} has columns {list(rows[0])} and the recipe "
            f"asks for {absent}. Nothing was written."
        )

    entries = [
        (
            int(row["atlas_idx"]),
            (
                row[recipe.subtask_column].strip()
                if recipe.subtask_column
                else recipe.single_subtask
            ),
            int(row[recipe.doc_id_column]),
        )
        for row in rows
    ]
    indices = sorted(atlas_idx for atlas_idx, _, _ in entries)
    if indices != list(range(1, len(entries) + 1)):
        raise SystemExit(
            f"{dataset}: {recipe.index_map} does not address a dense 1..{len(entries)} "
            f"run of bank columns -- it runs to {indices[-1]} over {len(entries)} rows. "
            f"The bank addresses columns by number, so a gap is an unaddressable row and "
            f"vendoring's max-index guard would reject the result. Nothing was written."
        )
    return sorted(entries)


def build_v2_rows(
    dataset: str,
    entries: Sequence[tuple[int, str, int]],
    order: dict[str, dict[int, int]],
    instances: dict[str, list[Any]],
) -> list[dict[str, Any]]:
    """Return the bridge rows, aborting if a calibrated column has nothing to name.

    Every guard here is about the index map and the details run describing the same
    harvest. A calibrated column whose ``doc_id`` the leaderboard never evaluated means
    the two are from different evaluations, and there is no partial version of that
    worth shipping: the rows that did resolve would be a bridge over an item set nobody
    chose.
    """
    rows: list[dict[str, Any]] = []
    for atlas_idx, label, doc_id in entries:
        position = order.get(label, {}).get(doc_id)
        if position is None:
            raise SystemExit(
                f"{dataset}: bank column {atlas_idx} was calibrated from subtask "
                f"{label!r} doc_id {doc_id}, which the pinned leaderboard run never "
                f"evaluated. The index map and the details run describe different "
                f"evaluations. Nothing was written."
            )
        instance = instances[label][position]
        rows.append(
            {
                "atlas_idx": atlas_idx,
                "item_id": content_item_id(instance.question, tuple(instance.choices or ())),
                "subtask": label,
                "split_index": doc_id,
                "question": stem(instance.question),
            }
        )

    collisions = [
        item_id for item_id, n in Counter(row["item_id"] for row in rows).items() if n > 1
    ]
    if collisions:
        raise SystemExit(
            f"{dataset}: {len(collisions)} content hash(es) name more than one item, the "
            f"first being {collisions[0]}. Two distinct items sharing an id would both be "
            f"dropped as ambiguous at vendoring time, so widen the key rather than ship "
            f"this. Nothing was written."
        )
    return rows


def build_v2(dataset: str, *, source_ref: str, dry_run: bool) -> int:
    """Rebuild one Route B bridge from v2 details, and report what it found."""
    use_system_trust_store()

    spec = get_spec(dataset)
    recipe = LEADERBOARD_V2[dataset]
    entries = read_index_map(dataset, recipe, source_ref)
    labels = sorted({label for _, label, _ in entries})
    log.info(
        "Index map %s: %d calibrated columns over subtasks %s",
        recipe.index_map,
        len(entries),
        labels,
    )

    files = list_details_files(V2_SOURCE.repo, V2_SOURCE.revision)
    paths = {
        label: details_path(V2_SOURCE.repo, files, recipe.task_prefix + label, run=V2_SOURCE.run)
        for label in labels
    }
    documents = {
        label: {
            doc_id: recipe.from_record(document)
            for doc_id, document in read_details(V2_SOURCE.repo, V2_SOURCE.revision, path).items()
        }
        for label, path in paths.items()
    }
    log.info(
        "Read %d evaluated documents from %s run %s",
        sum(len(found) for found in documents.values()),
        V2_SOURCE.repo,
        V2_SOURCE.run,
    )

    cross_checked = cross_check_documents(dataset, recipe, labels, documents)

    instances = subtask_instances(dataset, spec, recipe, labels)
    order: dict[str, dict[int, int]] = {}
    identity = 0
    for label in labels:
        order[label] = match_documents(dataset, label, recipe, documents[label], instances[label])
        coincident = sum(1 for doc_id, position in order[label].items() if doc_id == position)
        identity += coincident
        log.info(
            "Subtask %r: matched %d documents to %d instances as a bijection, %d of them "
            "at the position a fresh enumeration would have assumed",
            label,
            len(order[label]),
            len(instances[label]),
            coincident,
        )

    rows = build_v2_rows(dataset, entries, order, instances)
    evaluated = sum(len(found) for found in documents.values())

    provenance: dict[str, Any] = {
        "dataset": dataset,
        "bridge_kind": CONTENT_HASH,
        "rows": len(rows),
        "ordering": {
            "source": "Open LLM Leaderboard v2 per-model detail repo",
            "harness_config": f"{recipe.task_prefix}*",
            "repo": V2_SOURCE.repo,
            "revision": V2_SOURCE.revision,
            "run": V2_SOURCE.run,
            "paths": paths,
            "doc_id": (
                "read from each record's explicit doc_id field, not from its position in "
                "the file: lm-eval writes samples in completion order, so row order and "
                "document order differ"
            ),
            "cross_checked": cross_checked,
        },
        "item_id": {
            "scheme": CONTENT_HASH,
            "derivation": (
                "sha256 of the olmo_eval instance's question and choices, each "
                "NUL-terminated, truncated to the first 16 hex digits; see "
                "datasets.content_item_id"
            ),
            "tasks": list(spec.task_names),
        },
        "matching": {
            "recipe": (
                "each evaluated document rebuilt from the details record as the olmo-eval "
                "task renders it, then matched as a bijection over each subtask's split"
            ),
            "doc_fields": list(recipe.doc_fields),
            "normalization": recipe.normalization,
            "enumeration": recipe.enumeration,
            "documents_evaluated": evaluated,
            "documents_matched": evaluated,
            "positions_agreeing_with_enumeration_order": identity,
        },
        "index_map": {
            "source_ref": source_ref,
            "path": recipe.index_map,
            "rows": len(entries),
            "subtask_rows": dict(sorted(Counter(label for _, label, _ in entries).items())),
        },
        "generated_at": datetime.now(UTC).isoformat(),
    }

    if dry_run:
        log.info("[dry-run] %s: %d rows ready. Nothing written.", dataset, len(rows))
        return 0

    write_bridge(dataset, rows, provenance, fieldnames=V2_BRIDGE_COLUMNS)
    return 0


def inspect_v2(dataset: str, details_task: str) -> int:
    """Print what a v2 details file holds, so a recipe is written against facts.

    The step before a benchmark gets a :data:`LEADERBOARD_V2` entry. What a record's
    ``doc`` contains is whatever lm-evaluation-harness passed the leaderboard and is not
    guessable -- gpqa's carries seventy-odd annotation columns beside the question,
    ifeval's carries verifier arguments, and the *prompt* the model saw is somewhere else
    again -- so this reports the schema, the doc_id coverage and whether the file is
    written in document order, which for every v2 file checked so far it is not.
    """
    use_system_trust_store()

    tasks = [details_task]
    if not details_task:
        recipe = LEADERBOARD_V2.get(dataset)
        if recipe is None:
            raise SystemExit(
                f"{dataset} has no LEADERBOARD_V2 entry, so there is nothing to say which "
                f"leaderboard tasks its subtasks correspond to. Inspect one by name "
                f"instead: --inspect --details-task leaderboard_<benchmark>_<subtask>."
            )
        entries = read_index_map(dataset, recipe, "origin/Research")
        tasks = [recipe.task_prefix + label for label in sorted({label for _, label, _ in entries})]

    files = list_details_files(V2_SOURCE.repo, V2_SOURCE.revision)
    for task in tasks:
        path = details_path(V2_SOURCE.repo, files, task, run=V2_SOURCE.run)
        documents = read_details(V2_SOURCE.repo, V2_SOURCE.revision, path)
        doc_ids = sorted(documents)
        first = documents[doc_ids[0]]
        print(
            json.dumps(
                {
                    "leaderboard_task": task,
                    "path": path,
                    "documents": len(documents),
                    "doc_id_range": [doc_ids[0], doc_ids[-1]],
                    "doc_id_is_dense": doc_ids == list(range(doc_ids[0], doc_ids[-1] + 1)),
                    "doc_keys": sorted(first),
                    "first_document": {
                        key: stem(str(value)) for key, value in sorted(first.items())
                    },
                },
                indent=2,
            )
        )
    return 0


def buildable() -> Iterator[str]:
    """The datasets a leaderboard-order bridge can be built for."""
    yield from LEADERBOARD_ORDER
    yield from LEADERBOARD_V2


def build(dataset: str, *, source_ref: str, dry_run: bool) -> int:
    """Rebuild one dataset's bridge and report what it found."""
    use_system_trust_store()
    from olmo_eval.evals.tasks.common.registry import get_task

    spec = get_spec(dataset)
    source = LEADERBOARD_ORDER[dataset]
    recipe = LEADERBOARD_RECIPES[dataset]

    examples = read_leaderboard_order(source, recipe)
    log.info("Read %d examples in %s order from %s", len(examples), source.config, source.repo)

    task = get_task(spec.task)
    instances = list(task.instances)
    docs = task_documents(task)
    check_documents_align(dataset, docs, instances)
    log.info(
        "Task %r enumerates %d instances over %d documents", spec.task, len(instances), len(docs)
    )

    if len(examples) != len(docs):
        raise SystemExit(
            f"{dataset}: the leaderboard evaluated {len(examples)} examples but this "
            f"split holds {len(docs)} documents. The bridge has to span the whole split "
            f"because ATLAS calibrated column indices up to its size. Nothing was written."
        )

    order = match_order(dataset, examples, docs)
    identity = sum(1 for atlas_idx, position in enumerate(order) if atlas_idx == position)
    log.info(
        "Matched every example to a distinct document. %d of %d positions coincide with "
        "split order, which is what the superseded generator assumed for all of them.",
        identity,
        len(order),
    )

    rows = build_rows(dataset, order, instances)

    provenance: dict[str, Any] = {
        "dataset": dataset,
        "bridge_kind": CONTENT_HASH,
        "rows": len(rows),
        "ordering": {
            "source": "Open LLM Leaderboard v1 per-model detail repo",
            "harness_config": source.config,
            "repo": source.repo,
            "revision": source.revision,
            "path": source.path,
            "cross_checked": list(CROSS_CHECKED),
        },
        "item_id": {
            "scheme": CONTENT_HASH,
            "derivation": (
                "sha256 of the olmo_eval instance's question and choices, each "
                "NUL-terminated, truncated to the first 16 hex digits; see "
                "datasets.content_item_id"
            ),
            "task": spec.task,
            "split": task.config.split.value,
            "data_source": str(task.config.get_data_source(split=task.config.split.value)),
        },
        "matching": {
            "recipe": (
                "each leaderboard example rebuilt from the HuggingFace document as the v1 "
                "harness rendered it, then matched as a bijection over the split"
            ),
            "details_columns": list(recipe.columns),
            "positions_agreeing_with_split_order": identity,
        },
        "generated_at": datetime.now(UTC).isoformat(),
    }

    control_only = dataset == "arc_challenge"
    if control_only:
        provenance["control"] = verify_against_shipped(
            rows, order, instances, source_ref=source_ref, shipped_path=spec.bridge_path
        )

    if dry_run:
        log.info("[dry-run] %s: %d rows ready. Nothing written.", dataset, len(rows))
        return 0

    if control_only:
        write_control(provenance)
    else:
        write_bridge(dataset, rows, provenance)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the bridge generator's argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m diagnostics.mcq_cat.styles.uni_mcq.scripts.build_leaderboard_bridge",
        description=(
            "Rebuild a bank's index bridge from Open LLM Leaderboard v1 example order "
            "or v2 per-example details."
        ),
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(buildable()),
        help=(
            "Dataset to rebuild. arc_challenge is the control: it keeps the bridge the "
            "ATLAS release shipped, and rebuilding it reports agreement against that "
            "rather than replacing it."
        ),
    )
    parser.add_argument(
        "--source-ref",
        default="origin/Research",
        help=(
            "Git ref holding the shipped ARC bridge the v1 control compares against, and "
            "the index map a v2 rebuild re-keys."
        ),
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help=(
            "Report what the pinned v2 details run holds for a benchmark rather than "
            "building anything: the file it resolved, the doc_id coverage, and the "
            "record's doc schema. Run this before adding a LEADERBOARD_V2 entry."
        ),
    )
    parser.add_argument(
        "--details-task",
        default="",
        help=(
            "Inspect one leaderboard task by name (leaderboard_bbh_snarks) instead of "
            "every subtask of a configured dataset. The way in for a benchmark that has "
            "no LEADERBOARD_V2 entry yet."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run every check and report counts without writing the bridge.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Rebuild one dataset's bridge, or inspect the details run it would read."""
    args = build_parser().parse_args(argv)

    if args.inspect:
        if not args.dataset and not args.details_task:
            raise SystemExit("--inspect needs either --dataset or --details-task.")
        return inspect_v2(args.dataset, args.details_task)
    if not args.dataset:
        raise SystemExit("--dataset is required unless --inspect names a --details-task.")
    if args.dataset in LEADERBOARD_V2:
        return build_v2(args.dataset, source_ref=args.source_ref, dry_run=args.dry_run)
    return build(args.dataset, source_ref=args.source_ref, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
