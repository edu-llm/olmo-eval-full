"""Vendor a calibrated unidimensional IRT bank into ``calibrated_datasets/``.

Run offline by a developer, never on the eval box::

    python -m diagnostics.mcq_cat.styles.uni_mcq.scripts.vendor_bank \\
        --dataset arc_challenge --source-ref origin/Research

Upstream calibration output is not directly usable: it is in mirt's ``(a1, d, g)``
parameterization, keyed by a positional column index, and carries no question text.
This script performs the three transformations that stand between it and a runnable
bank, exactly as ``src/olmo_eval/adaptive/bank.py`` on ``origin/Research`` does, so
run-time behavior matches the published research results:

1. Convert the parameters to ``(a, b, c)`` and drop unusable items.
2. Join the bank's positional index to a joinable id through the dataset's bridge, which
   is the task's ``metadata["id"]`` for most datasets and a content hash for the banks
   re-keyed from Open LLM Leaderboard details. Those are the reason a bridge may live in
   this repo rather than on the source ref; see :func:`read_bridge_text` and
   ``scripts/build_leaderboard_bridge.py``.
3. Enumerate the olmo-eval task to recover the item text -- choices and a gold index
   for an MCQ dataset, and for a generative one whatever its grader decides on, which
   is a gold answer string for most and a set of instruction constraints for IFEval --
   then intersect with the bank.

A dataset declaring ``frozen_prompt`` takes step 3 one further and keeps the whole
rendered prompt as its stem, not just the question. BBH is the case that needs it: its
3-shot block sits behind a description that differs per *subtask*, which no per-dataset
setting can express, so the alternative was run-time few-shot machinery that could
disagree with the bank about the exemplars. The task's rendering stops reaching the bank,
and in exchange a run cannot present the item any way but the one its difficulty was
estimated behind.

A dataset declaring ``choice_order_control`` adds a guard rather than a step, and it is
the one guard here whose absence would be silent. GPQA's options are shuffled per
question inside ``process_doc``, so its ``gold_index`` is meaningful against exactly one
permutation and a re-enumeration under a moved seed would produce a different one --
which every other check in this file would pass. :func:`check_choice_order` holds each
vendored choice list to the ordering committed in ``bridges/<dataset>.choice_order.json``
and refuses to write on any departure; see that function and
``scripts/freeze_choice_order.py``.

Step 3 has two forms, and a dataset takes exactly one. Most banks map onto a single
task and go through :func:`load_task_items`. A bank calibrated across several tasks
that upstream addressed as one -- GPQA's three subsets, MuSR's three splits -- declares
them in ``DatasetSpec.subtasks`` and goes through :func:`load_multi_task_items`, which
keys by the composite ``<subtask>|<position>`` its bridge uses and carries its own
per-subtask guards.

One multi-task bank needs a fourth step. GPQA's subtasks are nested rather than
disjoint, so upstream calibrated many of its questions two or three times over, and
:func:`drop_duplicate_questions` keeps one calibration each. It is declared per dataset
and runs for no other bank; see that function for what goes wrong without it.

The fit family is not a label the caller picks. ``--fit-family`` names how the
parameters were *estimated*, so asking for a family the spec does not record means
nothing without a ``--bank-dir`` that holds that fit, and the name is settled against
the parameters themselves before anything is written; see :func:`resolve_fit_family`
and :func:`check_parameter_family`.

Doing this once, offline, means a zero-overlap join is caught by a developer in
seconds rather than becoming a runtime failure after a checkpoint has been staged.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import math
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

from ....common import generative
from .. import convention
from ..datasets import (
    BRIDGE_KEY_COLUMNS,
    CONTENT_HASH,
    FIT_FAMILIES,
    MODALITIES,
    DatasetSpec,
    content_item_id,
    get_spec,
    supported_names,
)
from . import freeze_choice_order

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [uni_mcq.vendor] %(levelname)s: %(message)s",
)
log = logging.getLogger("uni_mcq.vendor")

#: repo_root/diagnostics/mcq_cat/styles/uni_mcq/scripts/vendor_bank.py
REPO_ROOT = Path(__file__).resolve().parents[5]
CALIBRATED_DATASETS = REPO_ROOT / "calibrated_datasets"

#: Below this fraction of the bank surviving the task join, a positional-id dataset
#: is assumed to have drifted rather than merely lost a few items.
DEFAULT_MIN_OVERLAP = 0.90

#: Stamped into the scoring-convention block so a reader can tell a block written
#: alongside the artifacts from one reconstructed afterwards by the migration script.
VENDOR_RECORDED_BY = "vendor_bank"


@dataclass
class DropCounts:
    """Why items were discarded, so the manifest can account for every upstream row."""

    non_positive_discrimination: int = 0
    non_finite_discrimination: int = 0
    not_in_bridge: int = 0
    not_in_task: int = 0
    ambiguous_item_id: int = 0
    duplicate_question: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "non_positive_discrimination": self.non_positive_discrimination,
            "non_finite_discrimination": self.non_finite_discrimination,
            "not_in_bridge": self.not_in_bridge,
            "not_in_task": self.not_in_task,
            "ambiguous_item_id": self.ambiguous_item_id,
            "duplicate_question": self.duplicate_question,
        }


@dataclass
class BankItem:
    """One calibrated item after conversion, before the task join.

    ``subtask`` is the bridge's own statement of where a row came from, empty for a
    single-task bank. Carried here rather than parsed back out of ``item_id`` because a
    content-hashed id says nothing about its subtask, and the per-subtask overlap floor
    is the guard that catches one task's enumeration drifting while the dataset as a
    whole stays comfortably above the global floor.
    """

    item_id: str
    atlas_idx: int
    discrimination: float
    difficulty: float
    guessing: float
    subtask: str = ""


@dataclass
class VendorResult:
    """Everything the manifest needs from one vendoring pass."""

    items: list[dict[str, Any]] = field(default_factory=list)
    params: list[dict[str, Any]] = field(default_factory=list)
    drops: DropCounts = field(default_factory=DropCounts)
    upstream_bank_rows: int = 0
    bridge_rows: int = 0
    ungradable_instances: int = 0
    #: Items held to a committed option ordering, 0 for a bank with no such control.
    #: Recorded in the manifest so a reader can tell a bank whose choice order was
    #: verified from one where the check simply did not apply.
    choice_order_checked: int = 0


def git_show(ref: str, path: str, *, repo: Path = REPO_ROOT) -> str:
    """Return the contents of ``path`` at ``ref`` without checking anything out."""
    completed = subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        cwd=repo,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(
            f"git show {ref}:{path} failed (exit {completed.returncode}).\n"
            f"{completed.stderr.strip()}\n"
            f"Fetch the ref first, for example: git fetch origin Research"
        )
    return completed.stdout


def _rows(text: str) -> list[dict[str, str]]:
    """Parse CSV text into a list of dict rows."""
    return list(csv.DictReader(io.StringIO(text)))


def _atlas_index(raw: str) -> int:
    """Parse an ``X{k}`` bank row label into its integer index."""
    return int(str(raw).strip().strip('"').lstrip("X"))


def drop_ambiguous_ids(items: list[BankItem], drops: DropCounts) -> list[BankItem]:
    """Remove every item whose joinable id is claimed by more than one bank row.

    Upstream bridges are not guaranteed to be injective. HellaSwag's maps 10,042
    ``atlas_idx`` values onto only 9,609 unique ``question_id`` values, so two
    separately calibrated items -- with different ``(a, b, c)`` -- can claim the same
    id. For example id ``180`` is claimed by ``atlas_idx`` 9 and 3261.

    Left alone this corrupts a run rather than failing it. ``load_irt_params`` builds a
    dict keyed by item id, so the later record silently overwrites the earlier one;
    ``BenchmarkBank.get`` returns only the first matching item; and the CAT can then
    score an item using parameters that were calibrated for a different one.

    There is no principled way to decide which row owns the id, so both are dropped.
    The cost is small -- 102 of 5,044 items on HellaSwag -- and the alternative is a
    confident but wrong ability estimate.
    """
    counts = Counter(item.item_id for item in items)
    ambiguous = {item_id for item_id, n in counts.items() if n > 1}
    if not ambiguous:
        return items

    kept = [item for item in items if item.item_id not in ambiguous]
    drops.ambiguous_item_id = len(items) - len(kept)
    log.warning(
        "Dropped %d items across %d ambiguous ids: the bridge maps several bank rows "
        "onto the same joinable id, so parameters cannot be attributed to an item. "
        "Examples: %s",
        drops.ambiguous_item_id,
        len(ambiguous),
        sorted(ambiguous)[:5],
    )
    return kept


def sibling_fit_dir(bank_dir: str, fit_family: str) -> str | None:
    """Return the conventionally named sibling directory holding ``fit_family``, if any.

    Upstream parks an alternative fit beside the recorded one under a suffixed name:
    GPQA's 2PL refit is ``calibration_2pl/`` next to the ``calibration/`` its spec
    points at. Used only so the override error can name a concrete path rather than
    describe one; nothing checks that the path exists, because whether a directory
    holds the fit it is named for is settled by :func:`check_parameter_family`.
    """
    head, _, tail = bank_dir.rpartition("/")
    if head and tail == "calibration":
        return f"{head}/calibration_{fit_family}"
    return None


def resolve_fit_family(
    spec: DatasetSpec, requested: str | None, bank_dir: str | None
) -> tuple[DatasetSpec, str]:
    """Return the spec to vendor from and the family to stamp, or abort.

    ``--fit-family`` names how the parameters were *estimated*. ``bank_dir`` comes from
    the spec and the flag never touched it, so on its own an override read the recorded
    calibration directory and wrote the requested name over what that directory actually
    held. Nothing downstream could notice: the manifest *is* the record of provenance,
    so ``resolve.check_fit_family`` -- which refuses at run time to score a bank as a
    family its manifest disagrees with -- would read the false name and be satisfied.
    Warning and proceeding made this the one place the mistake that is a correctness
    error everywhere else could be committed to disk.

    An override therefore has to arrive with a ``bank_dir`` holding the requested fit,
    and it has to be a *different* directory, since the recorded one holds the recorded
    family by definition. A redirect on its own is only a claim; it is checked against
    the parameters, not against the directory's name, by :func:`check_parameter_family`.
    """
    family = requested or spec.fit_family
    redirected = bank_dir is not None and bank_dir != spec.bank_dir

    if family != spec.fit_family and not redirected:
        sibling = sibling_fit_dir(spec.bank_dir, family)
        beside = f" Upstream keeps one beside it at {sibling}." if sibling else ""
        raise SystemExit(
            f"{spec.name}: --fit-family {family} disagrees with the {spec.fit_family} fit "
            f"datasets.py records, and --bank-dir was not redirected. The family describes "
            f"how the parameters were estimated, not how they should be labelled, so on its "
            f"own this flag would read the {spec.fit_family} parameters in {spec.bank_dir} "
            f"and stamp {family!r} on them -- after which the manifest is the only record "
            f"of provenance and it is wrong. Point --bank-dir at a directory holding a real "
            f"{family} fit.{beside} Nothing was written."
        )

    if bank_dir is not None:
        spec = replace(spec, bank_dir=bank_dir)
    return spec, family


def check_parameter_family(spec: DatasetSpec, fit_family: str, rows: list[dict[str, str]]) -> None:
    """Abort unless the calibrated rows were estimated under ``fit_family``.

    The directory a caller names is not evidence of what it holds, so the family being
    stamped is settled against the parameters. Each family is a constrained case of the
    next, and the constraint is what identifies it. 2PL is the ``c = 0`` case of 3PL,
    which makes the guessing column discriminating: every row of all five 3PL banks here
    carries a non-zero ``g``, while GPQA's 2PL refit keeps the column and leaves it
    identically zero across all 1,192 rows. 1PL is in turn the ``a = 1`` case of 2PL, so
    a bank whose discriminations are all exactly 1 was not fitted for discrimination at
    all. A fit that estimated ``g`` is a 3PL fit whatever it is called; one that estimated
    ``a`` is a 2PL fit; one that estimated neither is Rasch.

    Only rows the bank will actually use are read, meaning those with a positive ``a``.
    The locally fitted banks carry a zeroed row for every position ``filter_items``
    dropped, so that the index space stays complete for :func:`check_alignment`; those
    rows are discarded by :func:`load_bank` as non-positive discrimination and would
    otherwise make every Rasch bank look like a 2PL one on an ``a == 1`` test.

    This is what the override in :func:`resolve_fit_family` rests on. Without it,
    redirecting ``--bank-dir`` anywhere at all would be enough to stamp any family onto
    anything, and the redirect requirement would be paperwork rather than a guard.
    """
    with_guessing = [row for row in rows if float(row.get("g") or 0.0) != 0.0]
    scored = [row for row in rows if float(row.get("a1") or 0.0) > 0.0]
    unit_discrimination = scored and all(
        math.isclose(float(row["a1"]), 1.0, rel_tol=0.0, abs_tol=1e-9) for row in scored
    )

    if with_guessing:
        observed = "3pl"
    elif unit_discrimination:
        observed = "1pl"
    else:
        observed = "2pl"
    if observed == fit_family:
        return

    if observed == "3pl":
        raise SystemExit(
            f"{spec.name}: {spec.bank_dir} is being vendored as {fit_family}, but "
            f"{len(with_guessing)} of its {len(rows)} rows carry a non-zero guessing "
            f"parameter, so these are 3PL estimates. 2PL is the c = 0 case of 3PL: zeroing "
            f"a g that was estimated jointly with a and b is not a relabelling, it shifts "
            f"every ability estimate the bank produces. Vendor from a directory whose g "
            f"column is identically zero. Nothing was written."
        )
    if observed == "1pl":
        raise SystemExit(
            f"{spec.name}: {spec.bank_dir} is being vendored as {fit_family}, but every one "
            f"of its {len(scored)} scorable rows has a = 1, which is a Rasch fit. Calling it "
            f"{fit_family} would claim a discrimination was estimated per item when the "
            f"calibration sample went entirely into difficulty. Pass --fit-family 1pl, or "
            f"point --bank-dir at the {fit_family} calibration. Nothing was written."
        )
    raise SystemExit(
        f"{spec.name}: {spec.bank_dir} is being vendored as {fit_family}, but its "
        f"{len(scored)} scorable rows have g = 0 and a varying a, which is a 2PL fit. Pass "
        f"--fit-family 2pl so the manifest records the family these parameters were actually "
        f"estimated under, or point --bank-dir at the {fit_family} calibration. Nothing was "
        f"written."
    )


def read_bridge_text(spec: DatasetSpec, source_ref: str) -> str:
    """Return the bridge's CSV text from wherever that dataset's bridge lives.

    Most bridges are upstream artifacts and are read off the source ref, which keeps
    them versioned with the calibration they describe. The ones rebuilt from Open LLM
    Leaderboard details have no upstream: they are ours, and they are committed here
    rather than regenerated because regenerating one needs the Hub, an archived details
    file and the HuggingFace split all reachable at once. A join that can only be
    reproduced under those conditions is a join that will not be reproduced.
    """
    if not spec.bridge_in_repo:
        return git_show(source_ref, spec.bridge_path)

    path = REPO_ROOT / spec.bridge_path
    if not path.is_file():
        raise SystemExit(
            f"{spec.name}: bridge_in_repo is set but {path} does not exist. Rebuild it "
            f"with:\n    python -m diagnostics.mcq_cat.styles.uni_mcq.scripts."
            f"build_leaderboard_bridge --dataset {spec.name}"
        )
    return path.read_text(encoding="utf-8")


@dataclass(frozen=True)
class Bridge:
    """A dataset's bridge as the file states it, not as a join would reconstruct it.

    Attributes:
        ids: ``atlas_idx -> joinable item id``, the map the whole vendoring rests on.
        subtasks: ``atlas_idx -> subtask``, where the bridge names one. Present for a
            multi-task bank and empty otherwise, and the distinction matters because it
            decides whether the per-subtask overlap floor has anything to group by.
        rows: How many rows the file holds, which the max-index guard compares against.
    """

    ids: dict[int, str]
    subtasks: dict[int, str]
    rows: int


def load_bridge(spec: DatasetSpec, source_ref: str) -> Bridge:
    """Return the bridge's ``atlas_idx`` map, keyed on the column its kind names.

    Three conventions exist and :data:`~..datasets.BRIDGE_KEY_COLUMNS` names the column
    each keys on. ATLAS banks ship ``atlas_idx_to_question_id.csv``, whose
    ``question_id`` is the task's own ``metadata["id"]``. GPQA ships a four-column
    ``item_id_map.csv`` where ``question_id`` is positional *within* a subtask, so the
    joinable key is the composite ``item_id``. A :data:`~..datasets.CONTENT_HASH` bridge
    keys on an ``item_id`` derived from the item's text by
    :func:`~..datasets.content_item_id`, which is what the leaderboard-order rebuilds
    use.

    A ``subtask`` column is read where the file carries one and is not required, because
    an ``item_id_map`` bridge already names the subtask inside its composite id and the
    ATLAS bridges span a single task. It is what a content-hashed multi-task bridge has
    instead: hashing the item is what removes the position from the key, and it removes
    the subtask with it.
    """
    rows = _rows(read_bridge_text(spec, source_ref))
    if not rows:
        raise SystemExit(f"Bridge {spec.bridge_path} is empty.")

    key = BRIDGE_KEY_COLUMNS.get(spec.bridge_kind)
    if key is None:
        raise SystemExit(
            f"{spec.name}: unknown bridge_kind {spec.bridge_kind!r}, expected one of "
            f"{', '.join(sorted(BRIDGE_KEY_COLUMNS))}. Vendoring cannot guess which "
            f"column holds the joinable id."
        )
    if key not in rows[0]:
        raise SystemExit(
            f"Bridge {spec.bridge_path} has columns {list(rows[0])}, expected a {key!r} "
            f"column for bridge_kind={spec.bridge_kind!r}."
        )

    return Bridge(
        ids={int(row["atlas_idx"]): str(row[key]).strip() for row in rows},
        subtasks={
            int(row["atlas_idx"]): str(row["subtask"]).strip() for row in rows if "subtask" in row
        },
        rows=len(rows),
    )


def bridge_provenance(spec: DatasetSpec) -> dict[str, Any] | None:
    """Return the sidecar recording where an in-repo bridge's ordering came from.

    Copied into the bank's manifest so a vendored bank carries the answer to "which
    order is this, and how was it recovered" without a reader having to find the bridge
    directory. Absent for a bridge read off the source ref, whose provenance is the
    upstream commit the manifest already records.
    """
    if not spec.bridge_in_repo:
        return None
    sidecar = (REPO_ROOT / spec.bridge_path).with_suffix(".json")
    if not sidecar.is_file():
        raise SystemExit(
            f"{spec.name}: {sidecar} is missing. A rebuilt bridge without its provenance "
            f"record is an ordering nobody can re-derive, so the manifest would claim a "
            f"join it cannot account for. Nothing was written."
        )
    return json.loads(sidecar.read_text(encoding="utf-8"))


def load_bank(
    spec: DatasetSpec,
    source_ref: str,
    bridge: Bridge,
    drops: DropCounts,
    *,
    fit_family: str,
) -> tuple[list[BankItem], int]:
    """Convert the upstream parameter CSV into ``(a, b, c)`` items.

    Mirrors ``load_bank`` in ``src/olmo_eval/adaptive/bank.py``: ``a = a1``,
    ``b = -d / a1``, ``c = clip(g, 0, 0.999)``, dropping any item whose
    discrimination is non-positive or non-finite, or which has no bridge entry.

    The discrimination filter is not cosmetic. GPQA's ``X2`` has ``a1 = -38.9`` and
    WinoGrande's ``X3`` has ``a1 = -0.25``; a negative discrimination scores backwards,
    and since Fisher information scales with ``a`` squared such an item would actively
    corrupt selection if left in.

    ``fit_family`` is the name about to be stamped into the manifest, and this is the
    first point at which the parameters it describes are in hand, so it is checked
    against them here rather than taken on trust.
    """
    rows = _rows(git_show(source_ref, spec.params_csv))
    check_parameter_family(spec, fit_family, rows)
    items: list[BankItem] = []

    for row in rows:
        idx = _atlas_index(row["X"])
        a = float(row["a1"])
        d = float(row["d"])
        g = float(row.get("g") or 0.0)

        if not math.isfinite(a):
            drops.non_finite_discrimination += 1
            continue
        if a <= 0:
            drops.non_positive_discrimination += 1
            continue
        if idx not in bridge.ids:
            drops.not_in_bridge += 1
            continue

        items.append(
            BankItem(
                item_id=bridge.ids[idx],
                atlas_idx=idx,
                discrimination=a,
                difficulty=-d / a,
                guessing=min(max(g, 0.0), 0.999),
                subtask=bridge.subtasks.get(idx, ""),
            )
        )

    return items, len(rows)


def check_alignment(spec: DatasetSpec, source_ref: str, bridge_rows: int) -> None:
    """Abort unless the bank's max index equals the bridge row count.

    This is the guard from ``build_atlas_idx_bridge.py``. The bridge is a positional
    map over the task's enumeration, so if the two disagree the join is misaligned and
    would score the wrong items while still producing a confident-looking number.

    Note this compares the bank's *max index*, not its row count. Calibration drops
    items that fail to converge, leaving sparse indices, so fewer bank rows than bridge
    rows is expected and correct.
    """
    rows = _rows(git_show(source_ref, spec.params_csv))
    max_x = max(_atlas_index(row["X"]) for row in rows)
    if max_x != bridge_rows:
        raise SystemExit(
            f"{spec.name}: bank max index is {max_x} but the bridge has {bridge_rows} "
            f"rows. A positional join would be misaligned; aborting rather than "
            f"vendoring a bank that scores the wrong items."
        )
    log.info("Alignment guard passed: max bank index %d == bridge rows %d", max_x, bridge_rows)


def _use_system_trust_store() -> None:
    """Route TLS verification through the OS trust store, when available.

    Enumerating a task downloads its dataset from HuggingFace. On a machine behind a
    TLS-inspecting proxy the bundled CA list does not contain the intercepting root,
    and the download fails with ``CERTIFICATE_VERIFY_FAILED``. ``truststore`` uses the
    operating system's certificate store instead, which does contain it. Same approach
    as ``bootstrap_env()`` in the upstream inference sweep. Optional: absent the
    package, verification simply proceeds with the default bundle.
    """
    try:
        import truststore
    except ImportError:
        return
    truststore.inject_into_ssl()
    log.info("TLS verification routed through the OS trust store (truststore).")


def mcq_record(
    item_id: str, instance: Any, *, render_stem: Callable[[Any], str] | None = None
) -> dict[str, Any] | None:
    """Build the ``items.jsonl`` record for a multiple-choice instance.

    Returns ``None`` for an instance that cannot be graded by ranking choices: one
    with no choice set at all, or one whose gold choice cannot be identified. Most
    tasks state the gold as ``metadata["gold_idx"]``; the fallback matches
    ``gold_answer`` against the choice text, which is how a task that carries only the
    answer string expresses it. An item with no recoverable gold would be scored
    incorrect for every model, dragging its ability estimate down for a reason that
    has nothing to do with the model.

    ``render_stem`` supplies the stem for a ``frozen_prompt`` dataset and is
    :func:`frozen_stem` bound to the enumerating task. Without it the stem is the
    instance's own question, which is what every bank vendored before BBH holds and what
    the committed ones must keep holding: this record's keys, order and values are a
    cross-component contract with ``load_items_from_jsonl``.
    """
    choices = list(instance.choices or ())
    if not choices:
        return None

    metadata = instance.metadata or {}
    gold_index = metadata.get("gold_idx")
    if gold_index is None:
        gold_answer = instance.gold_answer
        gold_index = choices.index(gold_answer) if gold_answer in choices else None
    if gold_index is None:
        return None

    return {
        "id": item_id,
        "question": instance.question if render_stem is None else render_stem(instance),
        "choices": choices,
        "gold_index": int(gold_index),
    }


def frozen_stem(spec: DatasetSpec, task: Any, instance: Any) -> str:
    """Return the prompt ``task`` renders for ``instance``, checked before it is kept.

    Reproduced by calling the task rather than by rebuilding the layout here, which is
    the whole point: a second implementation of BBH's prompt would agree today, drift
    later, and the drift would be a bank of frozen prompts nothing else in the checkout
    produces. What is written is what a full eval sends.

    Three things are checked, because freezing a prompt puts the presentation out of
    reach of every later guard. The scored spans have to be the plain ``" {choice}"`` the
    run-time style will append, since only the prompt half is being frozen and a task
    scoring something else would leave the two halves describing different requests. No
    per-choice prompt may exist, because one frozen stem cannot carry one and a
    substitution layout quietly reduced to a shared prompt asks a different question. And
    the rendered prompt must contain the question and be more than it, which is what
    catches a few-shot block that has silently gone missing -- a bank of 0-shot stems
    scored against 3-shot difficulties, with nothing downstream able to tell.
    """
    request = task.format_request(instance)
    prompt = str(request.prompt)
    expected = tuple(f" {choice}" for choice in instance.choices or ())
    actual = tuple(request.continuations or ())

    if actual != expected:
        raise SystemExit(
            f"{spec.name}: the task scores continuations {actual[:3]} for an item whose "
            f"choices imply {expected[:3]}. Only the prompt is frozen here, so the scored "
            f"span has to be the choice behind a leading space or the two halves describe "
            f"different requests. Nothing was written."
        )
    if request.continuation_prompts:
        raise SystemExit(
            f"{spec.name}: the task varies the prompt per choice, which one frozen stem "
            f"cannot carry. A substitution layout collapsed onto a shared prompt scores a "
            f"different question than the bank holds a difficulty for. Nothing was written."
        )
    if instance.question not in prompt or prompt == instance.question:
        raise SystemExit(
            f"{spec.name}: the task rendered {prompt[:80]!r} for a question starting "
            f"{str(instance.question)[:40]!r}, which is not that question inside a "
            f"framing. This dataset keeps the rendered prompt as its stem precisely "
            f"because the framing is where the few-shot block lives, so a bare stem here "
            f"would be a 0-shot bank scored against difficulties estimated behind "
            f"exemplars. Nothing was written."
        )
    return prompt


def mcqa_chat_formatter() -> Any:
    """Return olmo-eval's ``MCQAChatFormatter``, or abort with a usable message.

    Used to lay out a multiple-choice question that is answered in prose. Imported
    rather than reproduced because the layout it writes is the one the letters in the
    gold answer refer to; a hand-rolled ``(A) ...`` block that differed in any way would
    still parse and would still be graded.
    """
    try:
        from olmo_eval.common.formatters import MCQAChatFormatter
    except ImportError as exc:  # pragma: no cover - environment problem, not logic
        raise SystemExit(
            f"Could not import olmo_eval's MCQAChatFormatter ({exc}), which lays out a "
            f"chain-of-thought multiple-choice prompt. Run this from the repo root with "
            f"the project installed."
        ) from exc
    return MCQAChatFormatter


def presented_question(instance: Any) -> str:
    """Return the text a generative item is presented as, choice block included.

    Most generative benchmarks present the stem alone. GPQA does not: its default is
    chain of thought over a multiple-choice question, so the lettered choices are part
    of the prompt and the answer is a letter into them. That makes the presentation
    something the bank has to carry rather than something run time can rebuild, for one
    specific reason -- ``GPQATask.process_doc`` reshuffles the choices per question from
    ``Random(f"{seed}:{index}")``, so the letter a gold answer names is only meaningful
    against one ordering. Freezing the block into the stem here freezes that ordering
    with it, and the gold letter beside it stays correct however the task's seed or
    enumeration later moves.

    The block is laid out by ``MCQAChatFormatter``, the formatter the gpqa tasks are
    registered with, so what is written is the user turn a full eval would send. The
    system turn is deliberately not folded in: it is a standing instruction rather than
    content, it is identical for every item, and at run time it comes from the
    checkpoint's own chat template through ``GenerationConfig.system_prompt_source``.
    """
    if not instance.choices:
        return str(instance.question)
    request = mcqa_chat_formatter()().format(instance)
    return str(request.messages[-1]["content"])


def check_choice_gold(item_id: str, instance: Any, gold_answer: str) -> None:
    """Abort unless a choices-carrying instance's gold letter indexes its own choices.

    The join between a shuffled choice block and the letter that answers it. The task
    derives both from the same shuffle, so they agree by construction today, and this
    checks that rather than trusting it: an upstream change that emitted the gold letter
    from the pre-shuffle order would leave the bank pointing at the wrong choice on
    roughly three items in four, every one of them well formed.
    """
    letters = [chr(ord("A") + i) for i in range(len(instance.choices))]
    if gold_answer not in letters:
        raise SystemExit(
            f"Item {item_id!r} has {len(instance.choices)} choices, so its gold answer "
            f"must be one of {letters}, and it is {gold_answer!r}. A gold that does not "
            f"index the choice block cannot be matched against an extracted letter, so "
            f"every response would score incorrect. Nothing was written."
        )
    gold_idx = (instance.metadata or {}).get("gold_idx")
    if gold_idx is None:
        return
    expected = chr(ord("A") + int(gold_idx))
    if expected != gold_answer:
        raise SystemExit(
            f"Item {item_id!r} states gold_answer {gold_answer!r} but gold_idx "
            f"{gold_idx}, which is choice {expected!r}. The two disagree about which "
            f"choice is correct, so the shuffled block and the letter that answers it "
            f"came from different orderings. Nothing was written."
        )


def check_choice_order(spec: DatasetSpec, items: Sequence[dict[str, Any]]) -> int:
    """Hold every vendored choice list to the ordering frozen before the bank moved.

    The guard for the one hazard nothing else in this pipeline can see. An MCQ item's
    options are shuffled per question upstream, ``gold_index`` indexes that shuffle and
    nothing else, and a choice list in the wrong order is not a broken bank -- it is a
    working one measuring the wrong thing. Every count matches, the join is total, the
    overlap floor is cleared, Fisher selection runs on the real difficulties, the CAT
    converges and the standard error collapses on schedule. The theta is noise.

    So the ordering is not re-derived and then sanity-checked; it is transferred and then
    *enforced*. ``bridges/<dataset>.choice_order.json`` holds the option texts and gold
    letter parsed out of the stems the bank shipped before this modality change, and this
    refuses to write anything that departs from them. Three separate departures are
    caught, because they fail in different directions and only one of them looks wrong:
    a choice list whose texts differ, a choice list holding the right texts in a
    different order, and a gold index naming a different option than the frozen letter
    did. The third is the one that would otherwise pass every test in the suite.

    The stem digest is checked too, and it is what makes the other three more than a
    comparison against a file this repo also wrote. It ties the question now being
    vendored to the *whole* stem the control was read out of: the new question and the
    frozen block have to reassemble that stem byte for byte, so a question that has
    drifted, a block that was parsed at the wrong boundary, or a control regenerated
    against a different bank all fail here rather than agreeing with themselves.

    Coverage is required in both directions. An item with no frozen ordering is one this
    guard cannot speak for, and a frozen ordering with no item means the bank moved
    underneath the control, which is exactly when it stops being evidence.

    Returns:
        How many items were checked, for the log and for the manifest.

    Raises:
        SystemExit: On any mismatch, any uncovered item and any unused control entry.
    """
    control = freeze_choice_order.load_control(spec.name)
    vendored = {str(item["id"]): item for item in items}

    uncovered = sorted(set(vendored) - set(control))
    unused = sorted(set(control) - set(vendored))
    if uncovered or unused:
        raise SystemExit(
            f"{spec.name}: the option-ordering control covers {len(control)} items and "
            f"this vendoring produced {len(vendored)}. "
            f"{len(uncovered)} items have no frozen ordering (for example "
            f"{uncovered[:3]}) and {len(unused)} frozen orderings have no item (for "
            f"example {unused[:3]}). The control is only evidence about the bank it was "
            f"read from, so a bank it does not cover item for item is one whose choice "
            f"order is unverified. Nothing was written."
        )

    for item_id, frozen in control.items():
        item = vendored[item_id]
        choices = list(item["choices"])
        gold_index = int(item["gold_index"])

        if choices != frozen.choices:
            same_set = sorted(choices) == sorted(frozen.choices)
            raise SystemExit(
                f"Item {item_id!r} was vendored with choices {choices!r} and the frozen "
                f"ordering is {frozen.choices!r}. "
                + (
                    "The options are the same and the order is not, which is the failure "
                    "this check exists for: gold_index indexes one permutation, so every "
                    "item would be scored against the wrong option while every count, "
                    "every join and every standard error stayed healthy. The task's "
                    "per-question shuffle has moved -- its seed, its enumeration index or "
                    "its choice construction. "
                    if same_set
                    else "The option texts themselves differ, so the task's preprocessing "
                    "has changed and these are not the items the bank holds difficulties "
                    "for. "
                )
                + "Nothing was written."
            )
        if gold_index != frozen.gold_index:
            raise SystemExit(
                f"Item {item_id!r} was vendored with gold_index {gold_index} "
                f"({choices[gold_index]!r}) and the frozen ordering names "
                f"{frozen.gold_letter} ({frozen.choices[frozen.gold_index]!r}). The "
                f"choice list agrees and the answer does not, which is the quietest way "
                f"this bank can be wrong: nothing downstream compares the two, so the "
                f"run would score every item against a distractor and report a confident "
                f"theta. Nothing was written."
            )

        stem = str(item["question"]) + freeze_choice_order.BLOCK_SEPARATOR
        stem += freeze_choice_order.render_block(frozen.choices)
        if freeze_choice_order.sha256(stem) != frozen.stem_sha256:
            raise SystemExit(
                f"Item {item_id!r} has a question that does not reassemble the stem its "
                f"frozen ordering was read out of. The choices match, so what has moved "
                f"is the question text or the boundary the control was parsed at, and "
                f"either way the ordering above is being transferred between two "
                f"different items. Nothing was written."
            )

    log.info(
        "Choice order: %d of %d %s items match the frozen ordering, texts, order and gold",
        len(control),
        len(vendored),
        spec.name,
    )
    return len(control)


def generative_record(item_id: str, instance: Any, *, answer_type: str) -> dict[str, Any] | None:
    """Build the ``items.jsonl`` record for a free-response instance.

    The empty ``choices`` and ``gold_index`` of ``-1`` are load-bearing, not padding.
    ``load_items_from_jsonl`` in the frozen common layer turns this record into a
    :class:`~diagnostics.mcq_cat.base.BenchmarkItem` verbatim, and that pair is what
    marks the item as not multiple choice at run time -- it is the signal that sends
    the item to the sampling grader instead of a log-likelihood ranking over a choice
    set that does not exist. They stay empty even for a benchmark whose questions do
    have choices, because the question there is still answered by generating prose: the
    choices are inside the stem, by :func:`presented_question`, and there is nothing for
    a log-likelihood ranking to rank.

    What makes a record complete depends on the grader ``answer_type`` names, and the
    grader is asked rather than assumed. A gold-matched bank needs the gold string. An
    :class:`~diagnostics.mcq_cat.common.generative.ItemGrader` bank has no gold at all
    -- an IFEval prompt is answered correctly when the verifiers it names pass, not
    when it matches an answer -- and needs the keys that grader declares in
    ``required_metadata`` instead. Requiring a gold unconditionally would skip all 541
    IFEval prompts as ungradable and abort vendoring on an empty join.

    Returns ``None`` when neither is satisfiable, because the item would then score
    incorrect for every model: a fixed 0 in the response pattern, read by EAP as
    evidence about the checkpoint.
    """
    grader = generative.get_answer_grader(answer_type)
    metadata: dict[str, Any] = {"answer_type": answer_type, "modality": "generative"}

    gold_answer = str(instance.gold_answer or "").strip()
    if gold_answer:
        metadata["gold_answer"] = gold_answer

    source = instance.metadata or {}
    required = grader.required_metadata if isinstance(grader, generative.ItemGrader) else ()
    for key in required:
        value = source.get(key)
        if not value:
            return None
        metadata[key] = value

    if not gold_answer and not required:
        return None
    if instance.choices and gold_answer:
        check_choice_gold(item_id, instance, gold_answer)

    return {
        "id": item_id,
        "question": presented_question(instance),
        "choices": [],
        "gold_index": -1,
        "metadata": metadata,
    }


def answer_type(gold_answer: str) -> str:
    """Classify a gold answer as ``"numeric"`` or ``"text"`` for the grader.

    Read off the answer itself, and no longer what vendoring uses: ``DatasetSpec``
    declares the answer type per dataset, because it is a property of the benchmark
    rather than of the individual answer. Kept because the distinction it draws is
    still the right one for a bank of bare numbers -- GSM8K's answers are uniformly
    numeric only because ``process_doc`` has already stripped the ``####`` marker and
    the comma separators, and labelling a prose answer numeric would have a grader
    parse a number out of it and compare that.
    """
    try:
        float(gold_answer)
    except ValueError:
        return "text"
    return "numeric"


def task_registry() -> Any:
    """Return ``olmo_eval``'s ``get_task``, or abort with a usable message."""
    _use_system_trust_store()
    try:
        from olmo_eval.evals.tasks.common.registry import get_task
    except ImportError as exc:  # pragma: no cover - environment problem, not logic
        raise SystemExit(
            f"Could not import the olmo_eval task registry ({exc}). Run this from the "
            f"repo root with the project installed, for example: uv run python -m ..."
        ) from exc
    return get_task


def record_builder(spec: DatasetSpec, task: Any) -> Any:
    """Return the record builder for ``spec.modality``, refusing an unknown one.

    The two shapes are deliberately different: an MCQ record carries the choice set
    the log-likelihood scorer ranks, a generative record carries whatever the grader
    named by ``spec.answer_type`` decides on. Guessing at a third convention would
    produce records no grader reads correctly, so an unrecognised modality aborts.

    This is also the one place the answer type is fixed, which is why it is bound here
    rather than sniffed per record. It is a property of the benchmark: every MATH gold
    is LaTeX to be compared symbolically, including the ones that happen to read as a
    bare number, and IFEval has no gold to read a type off at all. Resolving the grader
    now turns an unknown ``answer_type`` into an abort before the task downloads a
    dataset, instead of into one identical failure per enumerated instance.

    ``task`` is the enumeration these records are being built from, and it is a parameter
    rather than something :func:`frozen_stem` could look up because a multi-task bank has
    one per subtask. BBH's description and exemplars differ per subtask, so a builder
    bound to the wrong one would render every item behind another subtask's prefix --
    well-formed prompts, a total join, and 24 subtasks' difficulties attached to the
    wrong instructions.
    """
    if spec.modality not in MODALITIES:
        raise SystemExit(
            f"{spec.name}: unknown modality {spec.modality!r}, expected one of "
            f"{', '.join(MODALITIES)}. Vendoring cannot guess how the items are graded."
        )
    if spec.modality != "generative":
        if not spec.frozen_prompt:
            return mcq_record
        return partial(mcq_record, render_stem=partial(frozen_stem, spec, task))
    generative.get_answer_grader(spec.answer_type)
    return partial(generative_record, answer_type=spec.answer_type)


def task_item_key(spec: DatasetSpec) -> Any:
    """Return the function naming a task instance the way this dataset's bridge does.

    Most bridges name an instance by its own ``metadata["id"]``, falling back to the
    enumeration position for a task that emits none.

    A :data:`~..datasets.CONTENT_HASH` bridge names it by its text instead, and for
    HellaSwag that is not a preference. Its ``metadata["id"]`` is the native ``ind``,
    and the validation split concatenates two sub-splits whose ``ind`` values each
    restart, so 10,042 instances answer to only 9,609 ids: the id-keyed dict below
    silently keeps whichever instance was enumerated last for each collided id, and 204
    calibrated rows then have to be dropped as ambiguous because no row can be shown to
    own its id. Keying on the item's own text removes the collision rather than
    absorbing it.

    The question and choices used are the instance's own, matching what
    ``build_leaderboard_bridge`` hashed. Deliberately not :func:`presented_question`,
    whose GPQA branch folds a formatter's choice block into the stem: that is a
    rendering decision, and a bank should not be rekeyed by one.
    """
    if spec.bridge_kind != CONTENT_HASH:
        return lambda index, instance: str((instance.metadata or {}).get("id", index)).strip()
    return lambda index, instance: content_item_id(instance.question, tuple(instance.choices or ()))


def load_task_items(spec: DatasetSpec) -> tuple[dict[str, dict[str, Any]], int]:
    """Enumerate the olmo-eval task, keyed the way :func:`task_item_key` says.

    This is the single-task path, used by every dataset whose bridge maps onto one
    task's enumeration. A bank spanning several tasks goes through
    :func:`load_multi_task_items` instead, which keys by a composite id; the two are
    kept apart so it is obvious from the call site which join a dataset takes.

    This is the same source the live Research path uses at run time; we resolve it
    once, offline, and commit the result so the eval box needs no network.

    Returns the items alongside the number of instances skipped as ungradable. That
    count is deliberately kept out of :class:`DropCounts`, which accounts for upstream
    *bank* rows: a skipped instance already reappears there as ``not_in_task`` if the
    bank calibrated it, so folding the two together would double-count and break the
    manifest's accounting identity.
    """
    get_task = task_registry()
    key_of = task_item_key(spec)

    task = get_task(spec.task)
    build = record_builder(spec, task)
    items: dict[str, dict[str, Any]] = {}
    ungradable = 0

    for index, instance in enumerate(task.instances):
        item_id = key_of(index, instance)
        record = build(item_id, instance)
        if record is None:
            ungradable += 1
            continue
        items[item_id] = record

    if ungradable:
        log.warning(
            "Skipped %d of %d instances from task %r as ungradable under modality %r. "
            "Any calibrated bank row pointing at one of them is counted as not_in_task.",
            ungradable,
            len(items) + ungradable,
            spec.task,
            spec.modality,
        )
    log.info("Enumerated %d %s items from task %r", len(items), spec.modality, spec.task)
    return items, ungradable


def subtask_of(item_id: str) -> str:
    """Return the subtask half of a composite ``<subtask>|<position>`` item id."""
    return item_id.split("|", 1)[0]


def bank_subtask(item: BankItem) -> str:
    """Return the subtask a bank row belongs to, however its bridge says so.

    Both readings are the bridge's own statement and neither consults the task, which is
    the property that matters: the floor these labels feed exists to detect a task whose
    enumeration has moved, and a label derived from that enumeration would move with it.
    An ``item_id_map`` bridge names the subtask inside the composite id; a content-hashed
    bridge carries it in a column of its own, because a hash of the item names nothing
    but the item.
    """
    return item.subtask or subtask_of(item.item_id)


def subtask_row_counts(bridge: dict[int, str]) -> dict[str, int]:
    """Return ``subtask -> bridge rows``, read off the composite ids."""
    return dict(sorted(Counter(subtask_of(item_id) for item_id in bridge.values()).items()))


def check_subtask_coverage(spec: DatasetSpec, labels: Iterable[str]) -> None:
    """Abort unless the bridge's subtasks and the spec's declared tasks are the same set.

    Applies to every multi-task bank whatever its bridge keys on, because both failures
    are quiet in the same way. A subtask the bridge names but the spec never maps has
    nothing to enumerate it, so its whole share of the bank -- 198 of GPQA's 1,192 rows
    for diamond alone -- is written off as ``not_in_task``, which is indistinguishable
    from ordinary attrition. A subtask the spec declares that the bridge never names is a
    spec that has gone stale against the file it is joining through.
    """
    declared = {label for label, _ in spec.subtasks}
    named = set(labels)

    undeclared = sorted(named - declared)
    if undeclared:
        raise SystemExit(
            f"{spec.name}: the bridge names subtask(s) {undeclared} that no entry in "
            f"spec.subtasks maps to a task. Their bank rows would be dropped as "
            f"not_in_task rather than scored. Nothing was written."
        )
    absent = sorted(declared - named)
    if absent:
        raise SystemExit(
            f"{spec.name}: spec.subtasks declares {absent}, which the bridge never "
            f"names. The spec is stale relative to {spec.bridge_path}; nothing was "
            f"written."
        )


def check_subtask_alignment(spec: DatasetSpec, bridge: dict[int, str]) -> dict[str, int]:
    """Return each subtask's enumeration span, aborting unless the bridge is joinable.

    An ``item_id_map`` bridge encodes a position *within* a subtask, and vendoring
    rebuilds those positions by enumerating each subtask's task from zero. Two things
    therefore have to hold before enumeration is worth attempting.

    Every subtask the bridge names must have a task declared in ``spec.subtasks``.
    Without one, nothing rebuilds its keys and its entire share of the bank -- 198 of
    GPQA's 1,192 rows for diamond alone -- is written off as ``not_in_task``, which
    looks like ordinary attrition rather than a whole subtask going missing.

    Each subtask's positions must be distinct and must fall inside ``0..max``. What is
    returned is ``max + 1``, the length of the enumeration the bridge was built against,
    which :func:`load_multi_task_items` then holds the task to. That is the quantity the
    join actually depends on, and it is not the row count: calibration drops items that
    fail to converge, so a subtask can be sparse within its own span without anything
    being wrong. MuSR's ``object_placements`` is exactly that -- 254 rows spread over
    positions 0..255, two items lost in the fit -- and demanding a gapless ``0..n-1``
    run refused it before enumeration ever started, over an absence upstream is entitled
    to have.

    What the relaxation gives up is small and what it keeps is the part that matters. A
    duplicate position still aborts, since two bank rows claiming one question cannot
    both be right. A position beyond the task's enumeration still aborts, in
    :func:`load_multi_task_items`, because the span and the count are compared there. And
    a subtask whose enumeration has genuinely shifted is caught by
    :func:`check_subtask_overlap`, which is the real guard here: a rekeyed subtask
    matches far fewer of its own bank rows, and that floor is per subtask and unchanged.
    A gap in the bridge, by contrast, is invisible to all three because it is not an
    error -- the missing positions simply have no bank row to join.

    Only an ``item_id_map`` bridge reaches this. A content-hashed bridge over the same
    bank has no positions to align, which is the point of re-keying it: nothing is
    rebuilt by counting, so there is nothing for a counting guard to protect.
    """
    counts = subtask_row_counts(bridge)
    check_subtask_coverage(spec, counts)

    positions: dict[str, list[int]] = {}
    for item_id in bridge.values():
        label, _, position = item_id.partition("|")
        positions.setdefault(label, []).append(int(position))

    spans: dict[str, int] = {}
    for label, seen in sorted(positions.items()):
        duplicated = len(seen) - len(set(seen))
        if duplicated:
            repeated = sorted({p for p in seen if seen.count(p) > 1})
            raise SystemExit(
                f"{spec.name}: subtask {label!r} names {duplicated} position(s) more "
                f"than once (for example {repeated[:5]}), so several bank rows claim "
                f"the same question and no row can be shown to own it. Nothing was "
                f"written."
            )
        if min(seen) < 0:
            raise SystemExit(
                f"{spec.name}: subtask {label!r} has a negative bridge position "
                f"({min(seen)}). Positions are rebuilt by counting an enumeration from "
                f"zero, so this join cannot be reproduced. Nothing was written."
            )
        spans[label] = max(seen) + 1

    log.info(
        "Per-subtask alignment guard passed: %s bridge rows over spans %s",
        counts,
        spans,
    )
    return spans


def load_multi_task_items(
    spec: DatasetSpec,
    subtask_spans: dict[str, int],
    *,
    source_questions: dict[str, str] | None = None,
) -> tuple[dict[str, dict[str, Any]], int]:
    """Enumerate every subtask's task, keyed the way this dataset's bridge names items.

    The multi-task counterpart of :func:`load_task_items`, for a bank calibrated
    across several tasks that upstream addressed as one, and it keys by whichever of the
    two conventions the bridge uses.

    For an ``item_id_map`` bridge the key is the composite ``<subtask>|<position>``, with
    position restarting at zero per subtask; the instance's own ``metadata["id"]`` is
    deliberately ignored, since a native id would not join to a composite key and a task
    that happens to expose a positional one would only make the coincidence look
    load-bearing. ``subtask_spans`` is what :func:`check_subtask_alignment` returned --
    one past each subtask's highest bridge position -- and each subtask's enumeration is
    held to it. That is the check that matters here: a task emitting one instance fewer
    than the bridge's positions reach fails to join only its final position, so the
    overlap floor still passes while every item after the missing one is keyed to its
    neighbour's question. The span is compared rather than the row count because the two
    differ for a bank whose calibration dropped items -- MuSR's ``object_placements`` had
    254 rows over a 256-long enumeration -- and the span is what the positions were
    numbered against.

    For a :data:`~..datasets.CONTENT_HASH` bridge the key is the item's own text and
    ``subtask_spans`` is empty, because there are no positions to hold anything to. A
    shortened enumeration then loses exactly the items it dropped instead of rekeying
    everything after them, which is the whole reason to re-key a bank this way, and the
    overlap floor is what turns that loss into an abort.

    The record builder is rebuilt per subtask rather than once, because for a
    ``frozen_prompt`` dataset it carries the task whose prompt is being frozen and BBH's
    prefix differs per subtask. One builder for all 24 would render every item behind
    ``boolean_expressions``' description and exemplars.

    ``source_questions`` collects each kept item's own question text, as the dataset row
    states it and before any choice block or system prompt is rendered into its stem. It
    is what :func:`drop_duplicate_questions` groups a nested bank's repeats by, and it is
    taken here because this is the only point where the instance and the record are both
    in hand. Recovering it afterwards would mean parsing a rendered stem, which is not a
    tool worth owning: a regex splitting on the first option letter reports 2,957 phantom
    duplicates on BBH, whose frozen few-shot exemplars contain option letters of their
    own, while looking correct on GPQA.

    Returns the items and the number of instances skipped as ungradable, summed across
    subtasks, on the same terms as the single-task path.
    """
    get_task = task_registry()
    key_of = task_item_key(spec)
    by_content = spec.bridge_kind == CONTENT_HASH

    items: dict[str, dict[str, Any]] = {}
    ungradable = 0

    for label, task_name in spec.subtasks:
        try:
            task = get_task(task_name)
        except KeyError as exc:
            raise SystemExit(
                f"{spec.name}: subtask {label!r} maps to task {task_name!r}, which is "
                f"not registered on this branch ({exc}). Nothing was written."
            ) from exc

        build = record_builder(spec, task)
        enumerated = 0
        skipped = 0
        for position, instance in enumerate(task.instances):
            enumerated += 1
            item_id = key_of(position, instance) if by_content else f"{label}|{position}"
            if by_content and item_id in items:
                raise SystemExit(
                    f"{spec.name}: subtask {label!r} position {position} hashes to "
                    f"{item_id}, which another instance in this bank already answers to. "
                    f"Two items sharing a content id cannot both be scored, so widen the "
                    f"key rather than let one shadow the other. Nothing was written."
                )
            record = build(item_id, instance)
            if record is None:
                skipped += 1
                continue
            items[item_id] = record
            if source_questions is not None:
                source_questions[item_id] = str(instance.question)

        expected = subtask_spans.get(label)
        if expected is not None and enumerated != expected:
            raise SystemExit(
                f"{spec.name}: subtask {label!r} enumerated {enumerated} instances "
                f"from task {task_name!r} but the bridge's positions run to "
                f"{expected - 1}, so it was built against an enumeration of "
                f"{expected}. The composite id is a position within the subtask, so a "
                f"count mismatch rekeys every item past the discrepancy. Nothing was "
                f"written."
            )

        ungradable += skipped
        log.info(
            "Enumerated %d %s items from subtask %r (task %r), %d ungradable",
            enumerated - skipped,
            spec.modality,
            label,
            task_name,
            skipped,
        )

    if ungradable:
        log.warning(
            "Skipped %d of %d instances across %d subtasks as ungradable under "
            "modality %r. Any calibrated bank row pointing at one of them is counted "
            "as not_in_task.",
            ungradable,
            len(items) + ungradable,
            len(spec.subtasks),
            spec.modality,
        )
    log.info(
        "Enumerated %d %s items across %d subtasks", len(items), spec.modality, len(spec.subtasks)
    )
    return items, ungradable


def check_subtask_overlap(
    spec: DatasetSpec,
    bank_items: list[BankItem],
    matched_ids: Iterable[str],
    *,
    min_overlap: float,
) -> None:
    """Abort unless every subtask clears the overlap floor on its own.

    The global floor cannot see a single subtask drifting. These positions are
    numbered within a subtask, so one task's enumeration can shift while the others
    stay aligned, and a subtask holding a sixth of the bank -- diamond's share of
    GPQA -- can fall to half-matched with the dataset still above 90% overall. The
    global check passes, a sixth of the administered items are scored against the
    wrong questions, and the reported theta carries no sign of it.

    Which subtask a row belongs to is :func:`bank_subtask`'s answer and therefore the
    bridge's, for both kinds of key. A content-hashed bank cannot answer it from the id
    at all, and taking it from the enumeration instead would let a drifted subtask
    relabel its own rows and pass.
    """
    label_of = {item.item_id: bank_subtask(item) for item in bank_items}
    totals = Counter(label_of.values())
    kept = Counter(label_of[item_id] for item_id in matched_ids if item_id in label_of)

    for label in sorted(totals):
        overlap = kept[label] / totals[label]
        if overlap < min_overlap:
            raise SystemExit(
                f"{spec.name}: subtask {label!r} matched only {overlap:.1%} of its "
                f"{totals[label]} bank items, below the {min_overlap:.0%} floor, even "
                f"though the dataset as a whole may pass. Its enumeration has drifted "
                f"from what the bridge was built against. Nothing was written."
            )
        log.info(
            "Subtask %r: %d of %d bank items matched (%.1f%%)",
            label,
            kept[label],
            totals[label],
            overlap * 100,
        )


def drop_duplicate_questions(
    spec: DatasetSpec,
    items: list[BankItem],
    questions: dict[str, str],
    drops: DropCounts,
) -> list[BankItem]:
    """Keep one calibration per distinct question, where a bank's subtasks are nested.

    GPQA is the only bank here whose subtasks are not different content. Its three
    subsets are nested quality filters over one pool of 546 questions -- diamond (198)
    inside main (448) inside extended (546), per arXiv 2311.12022 section 2.3 -- which is
    why the three sum to the bridge's 1,192. Upstream calibrated that union as though the
    subsets were independent items, so every diamond question was fit three times and
    every main-only question twice, and 184 of the 579 rows that survive the ``a > 0``
    filter are repeat calibrations of a question another row already carries.

    At run time that is worse than redundancy. The CAT masks administered items by id and
    these copies have different ids, so one session can administer the same question two
    or three times, score it each time, and hand EAP each scoring as independent evidence
    about the checkpoint. The copies also disagree, systematically rather than randomly:
    ``atlas_idx`` was assigned over the lexicographically sorted composite ids, putting
    diamond at indices 1-198 and fitting it almost entirely inside its own first two of
    twelve ~100-item chunks, apart from its extended and main twins several chunks away,
    with mean-sigma linking only approximately reconciling the chunk scales. ``diamond|91``
    reads ``b = 182.58`` where its extended twin reads 1.85, and ``diamond|100`` reads
    ``a = 28.020`` against its main twin's 0.993 -- and since Fisher information scales
    with ``a`` squared, the inflated copy is the one selection prefers.

    ``spec.duplicate_question_precedence`` ranks the subtasks and the survivor is the
    earliest ranked tier a question still has a calibration in. Extended before main
    before diamond keeps the fit estimated over the widest chunk span, and it is a rule
    rather than a choice made per item.

    Deliberately not the resolution :func:`drop_ambiguous_ids` applies, and the difference
    is in the data rather than in taste. There, two *different* questions collided on one
    id and no row could be shown to own it, so every claimant went. Here it is one
    question calibrated repeatedly, the copies are distinguishable by how they were fit,
    and dropping all of them would discard the question along with them -- which on 24
    GPQA questions, whose extended and main copies both failed ``a > 0``, would mean
    losing the only calibration that exists.

    Two preconditions are checked rather than assumed. Every item must have a recorded
    question, because one missing key silently exempts that row from the rule instead of
    failing. And no two rows of one subtask may share a question, because that is a
    repeat *inside* a quality filter rather than across them: the precedence cannot rank
    two rows of one tier, so the survivor would fall out of bank order.
    """
    if not spec.duplicate_question_precedence:
        return items

    unrecorded = [item.item_id for item in items if item.item_id not in questions]
    if unrecorded:
        raise SystemExit(
            f"{spec.name}: {len(unrecorded)} of {len(items)} joined bank rows have no "
            f"recorded source question (for example {unrecorded[:3]}), so they cannot be "
            f"grouped with their repeats and would each survive as a distinct item. "
            f"Nothing was written."
        )

    order = " > ".join(spec.duplicate_question_precedence)
    rank = {label: position for position, label in enumerate(spec.duplicate_question_precedence)}
    groups: dict[str, list[BankItem]] = {}
    for item in items:
        groups.setdefault(questions[item.item_id], []).append(item)

    survivors: set[str] = set()
    for question, group in groups.items():
        labels = [bank_subtask(item) for item in group]
        repeated = sorted({label for label in labels if labels.count(label) > 1})
        if repeated:
            raise SystemExit(
                f"{spec.name}: subtask(s) {repeated} calibrate one question more than "
                f"once ({question[:60]!r}), which is a repeat inside a quality filter "
                f"rather than across them. The {order} precedence ranks tiers and cannot "
                f"choose between two rows of one tier, so the survivor would depend on "
                f"bank order. Nothing was written."
            )
        survivors.add(min(group, key=lambda item: rank[bank_subtask(item)]).item_id)

    kept = [item for item in items if item.item_id in survivors]
    drops.duplicate_question = len(items) - len(kept)
    log.info(
        "Deduplicated %d nested calibrations down to %d distinct questions by %s; "
        "survivors per subtask %s",
        len(items),
        len(kept),
        order,
        dict(sorted(Counter(bank_subtask(item) for item in kept).items())),
    )
    return kept


def subtask_item_counts(params: Iterable[dict[str, Any]]) -> dict[str, int]:
    """Return ``subtask -> vendored items`` for a bank whose records name a subtask.

    Recorded in the manifest because it is the distribution a reader would otherwise
    have to recount from ``params.json``, and on a deduplicated bank it is the thing that
    shows the precedence rule at work: GPQA's 579 rows are 252 / 228 / 99 across extended,
    main and diamond, and its 395 items are 252 / 119 / 24.
    """
    counts = Counter(
        str((record.get("metadata") or {})["subtask"])
        for record in params
        if "subtask" in (record.get("metadata") or {})
    )
    return dict(sorted(counts.items()))


def vendor(
    spec: DatasetSpec, source_ref: str, *, fit_family: str, min_overlap: float
) -> VendorResult:
    """Produce the params and items payloads for one dataset."""
    bridge = load_bridge(spec, source_ref)
    check_alignment(spec, source_ref, bridge.rows)

    subtask_spans: dict[str, int] = {}
    if spec.is_multi_task and spec.bridge_kind == CONTENT_HASH:
        check_subtask_coverage(spec, bridge.subtasks.values())
    elif spec.is_multi_task:
        subtask_spans = check_subtask_alignment(spec, bridge.ids)

    drops = DropCounts()
    bank_items, upstream_rows = load_bank(spec, source_ref, bridge, drops, fit_family=fit_family)
    bank_items = drop_ambiguous_ids(bank_items, drops)
    log.info(
        "Bank: %d upstream rows -> %d after conversion and filtering",
        upstream_rows,
        len(bank_items),
    )
    if upstream_rows != spec.expected_bank_rows:
        log.warning(
            "Upstream bank has %d rows but datasets.py expects %d. The bank may have "
            "been recalibrated; verify before committing.",
            upstream_rows,
            spec.expected_bank_rows,
        )

    source_questions: dict[str, str] = {}
    if spec.is_multi_task:
        task_items, ungradable = load_multi_task_items(
            spec, subtask_spans, source_questions=source_questions
        )
    else:
        task_items, ungradable = load_task_items(spec)

    matched: list[BankItem] = []
    for bank_item in bank_items:
        if bank_item.item_id not in task_items:
            drops.not_in_task += 1
            continue
        matched.append(bank_item)

    if not matched:
        raise SystemExit(
            f"{spec.name}: the bank and the task share zero item ids. The bridge maps to "
            f"ids like {next(iter(bridge.ids.values()), '?')!r} while the task emits ids "
            f"like {next(iter(task_items), '?')!r}. Nothing was written."
        )

    overlap = len(matched) / len(bank_items) if bank_items else 0.0
    log.info(
        "Task join: %d of %d bank items matched (%.1f%%)",
        len(matched),
        len(bank_items),
        overlap * 100,
    )
    if spec.is_multi_task:
        check_subtask_overlap(
            spec,
            bank_items,
            (bank_item.item_id for bank_item in matched),
            min_overlap=min_overlap,
        )
    if spec.needs_overlap_floor and overlap < min_overlap:
        drifted = (
            "This dataset joins by a content hash, so a low overlap means the task no "
            "longer renders the items the bridge was built against"
            if spec.bridge_kind == CONTENT_HASH
            else "This dataset joins by position, so a low overlap means the task's "
            "enumeration order has drifted from what the bridge was built against"
        )
        raise SystemExit(
            f"{spec.name}: only {overlap:.1%} of bank items matched the task, below the "
            f"{min_overlap:.0%} floor. {drifted}. Nothing was written."
        )

    # After the overlap guards rather than before them, because those measure whether the
    # bridge still names the questions it was built against and a bank row removed here
    # matched perfectly well. Deduplicating first would put a nested bank's least
    # preferred subtask below its own floor and abort on a healthy join.
    kept = drop_duplicate_questions(spec, matched, source_questions, drops)

    params: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    for bank_item in kept:
        # Recorded whenever the bridge names a subtask, rather than only for a
        # multi-task spec, because a content-hashed id no longer says which subtask an
        # item came from and leaderboard_math is the case where those two conditions
        # come apart: one registered task concatenates seven MATH subjects, so the spec
        # is single-task while the bank spans subjects the composite id used to carry.
        metadata: dict[str, Any] = {"atlas_idx": bank_item.atlas_idx}
        if bank_item.subtask or spec.is_multi_task:
            metadata["subtask"] = bank_subtask(bank_item)
        params.append(
            {
                "item_id": bank_item.item_id,
                "difficulty": bank_item.difficulty,
                "discrimination": bank_item.discrimination,
                "guessing": bank_item.guessing,
                "metadata": metadata,
            }
        )
        items.append(task_items[bank_item.item_id])

    # Last, and after deduplication rather than before it, because what has to be
    # verified is the choice list actually about to be written. Checking the pre-dedup
    # set would leave the surviving copy of a repeated question unexamined, which on this
    # bank is 395 of the 579 rows -- the whole of it.
    order_checked = check_choice_order(spec, items) if spec.choice_order_control else 0

    return VendorResult(
        items=items,
        params=params,
        drops=drops,
        upstream_bank_rows=upstream_rows,
        bridge_rows=bridge.rows,
        ungradable_instances=ungradable,
        choice_order_checked=order_checked,
    )


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_artifacts(
    spec: DatasetSpec,
    result: VendorResult,
    *,
    source_ref: str,
    source_commit: str,
    fit_family: str,
    out_dir: Path,
) -> None:
    """Write ``params.json``, ``items.jsonl`` and ``manifest.json``.

    The manifest's ``scoring_convention`` block is built from the style's ``config.yaml``
    through the same resolution a run performs, rather than described here. It is what
    :func:`~diagnostics.mcq_cat.styles.uni_mcq.convention.check_runtime_convention` then
    holds every run of this bank to, so a bank vendored while the config said one thing
    and run after it says another fails at startup instead of reporting the difference
    as ability.

    ``duplicate_question_precedence`` and ``subtask_items`` are ``null`` on every bank
    that does not deduplicate, and are recorded together because either alone invites the
    wrong reading. The rule says which calibration of a repeated question was preferred,
    and the distribution is what that rule left behind; a subtask count that has fallen
    is otherwise indistinguishable from a subtask that joined badly.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    params_text = json.dumps(result.params, indent=2) + "\n"
    items_text = "".join(json.dumps(item) + "\n" for item in result.items)

    (out_dir / "params.json").write_text(params_text, encoding="utf-8")
    (out_dir / "items.jsonl").write_text(items_text, encoding="utf-8")

    manifest = {
        "dataset": spec.name,
        "task": ", ".join(spec.task_names),
        "route": spec.route,
        "modality": spec.modality,
        "answer_type": spec.answer_type if spec.modality == "generative" else None,
        "fit_family": fit_family,
        "source_ref": source_ref,
        "source_commit": source_commit,
        "bank_dir": spec.bank_dir,
        "bridge_path": spec.bridge_path,
        "bridge_kind": spec.bridge_kind,
        "bridge_in_repo": spec.bridge_in_repo,
        "bridge_provenance": bridge_provenance(spec),
        "upstream_bank_rows": result.upstream_bank_rows,
        "bridge_rows": result.bridge_rows,
        "items": len(result.params),
        "subtask_items": subtask_item_counts(result.params) or None,
        "dropped": result.drops.as_dict(),
        "duplicate_question_precedence": list(spec.duplicate_question_precedence) or None,
        "ungradable_instances": result.ungradable_instances,
        "positional_ids": spec.positional_ids,
        "frozen_prompt": spec.frozen_prompt,
        # How many items had their option ordering held to a committed control, and
        # ``null`` where no control applies. Recorded rather than inferred from the
        # spec because it is a statement about this vendoring run: a bank whose
        # gold_index was verified against a frozen permutation and one where the
        # question never arose are indistinguishable on disk otherwise, and only one of
        # them can be believed about which option its difficulties describe.
        "choice_order_checked": result.choice_order_checked or None,
        convention.CONVENTION_KEY: convention.manifest_block(
            spec, convention.load_config(), recorded_by=VENDOR_RECORDED_BY
        ),
        # Recorded beside the notes rather than inside them because it is the one thing
        # a reader has to see before using the numbers, and the notes are long.
        "bank_caveat": spec.report_caveat or None,
        "notes": spec.notes,
        "sha256": {
            "params.json": _sha256(params_text),
            "items.jsonl": _sha256(items_text),
        },
        "generated_at": datetime.now(UTC).isoformat(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    log.info("Wrote %d items to %s", len(result.params), out_dir)


def build_parser() -> argparse.ArgumentParser:
    """Build the vendoring script's argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m diagnostics.mcq_cat.styles.uni_mcq.scripts.vendor_bank",
        description="Vendor a calibrated IRT bank into calibrated_datasets/.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help=f"Dataset to vendor. One of: {', '.join(supported_names())}.",
    )
    parser.add_argument(
        "--source-ref",
        default="origin/Research",
        help="Git ref holding the upstream calibration output (default: origin/Research).",
    )
    parser.add_argument(
        "--fit-family",
        default=None,
        choices=FIT_FAMILIES,
        help=(
            "The family the parameters were estimated under. Defaults to the dataset's "
            "recorded family; stamped into manifest.json and authoritative at run time. "
            "Overriding it requires --bank-dir, because this is not a relabelling."
        ),
    )
    parser.add_argument(
        "--bank-dir",
        default=None,
        help=(
            "Read the calibration output from this directory on the source ref instead of "
            "the one the spec records. Needed to vendor an alternative fit: GPQA's real 2PL "
            "refit is calibration_2pl/, beside the calibration/ its spec points at."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Override the output directory (default: calibrated_datasets/<dataset>).",
    )
    parser.add_argument(
        "--min-overlap",
        type=float,
        default=DEFAULT_MIN_OVERLAP,
        help=(
            "Minimum fraction of bank items that must match the task, enforced for "
            f"positional-id datasets (default: {DEFAULT_MIN_OVERLAP})."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run every check and report counts without writing artifacts.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Vendor one dataset."""
    args = build_parser().parse_args(argv)

    try:
        spec = get_spec(args.dataset)
    except KeyError as exc:
        log.error("%s", exc.args[0])
        return 2

    spec, fit_family = resolve_fit_family(spec, args.fit_family, args.bank_dir)
    if fit_family != spec.fit_family:
        log.info(
            "Vendoring %s as %s from %s, which datasets.py does not record as its bank; "
            "the parameters are checked against that family before anything is written.",
            spec.name,
            fit_family,
            spec.bank_dir,
        )

    source_commit = subprocess.run(
        ["git", "rev-parse", args.source_ref],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()

    result = vendor(spec, args.source_ref, fit_family=fit_family, min_overlap=args.min_overlap)

    if args.dry_run:
        log.info(
            "[dry-run] %s: %d items ready, drops=%s, ungradable_instances=%d. Nothing written.",
            spec.name,
            len(result.params),
            result.drops.as_dict(),
            result.ungradable_instances,
        )
        return 0

    out_dir = Path(args.out_dir) if args.out_dir else CALIBRATED_DATASETS / spec.name
    write_artifacts(
        spec,
        result,
        source_ref=args.source_ref,
        source_commit=source_commit,
        fit_family=fit_family,
        out_dir=out_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
