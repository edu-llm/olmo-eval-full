"""Guard the vendoring step against a bank row that is not the item it looks like.

Two failures live here and they need opposite resolutions, which is the thing to hold
on to while reading.

**One id, two questions.** An ``atlas_idx -> question_id`` bridge is not guaranteed to
be one-to-one. HellaSwag's maps 10,042 indices onto only 9,609 unique ids, so two
separately calibrated bank rows -- with different ``(a, b, c)`` -- can claim the same
joinable id. Left unhandled that corrupts a run rather than failing it, because every
downstream structure quietly picks a winner: ``load_irt_params`` builds a dict keyed by
item id so the later record overwrites the earlier one, and ``BenchmarkBank.get``
returns the first item whose id matches. The CAT would then score one item using
parameters calibrated for a different one, and still report a confident ability
estimate. No row can be shown to own the id, so every claimant is dropped.

**One question, two ids.** GPQA's three subsets are nested quality filters over one pool
rather than different content, so upstream calibrated many of its questions two or three
times under different composite ids. Nothing collides -- each id is honestly its own bank
row -- and that is what makes it worse: the CAT masks administered items by id, so a
session can administer one question three times and hand EAP each scoring as independent
evidence. Here the copies *are* distinguishable, by which subset's chunk of the
calibration they were fit in, so one is kept by a declared precedence rather than all
being dropped along with the question.

These tests pin both resolutions, pin that the second one runs for no bank that has not
declared it, and pin the invariant both exist to protect: params and items describe
exactly the same item set, one record each.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ..datasets import SUPPORTED, DatasetSpec, supported_names
from ..scripts.vendor_bank import (
    BankItem,
    DropCounts,
    drop_ambiguous_ids,
    drop_duplicate_questions,
    subtask_item_counts,
)

CALIBRATED_DATASETS = Path(__file__).resolve().parents[5] / "calibrated_datasets"


def _item(item_id: str, atlas_idx: int, a: float = 1.0) -> BankItem:
    return BankItem(
        item_id=item_id,
        atlas_idx=atlas_idx,
        discrimination=a,
        difficulty=0.0,
        guessing=0.0,
    )


def test_injective_bridge_is_left_alone() -> None:
    """A one-to-one bridge must pass through untouched, with nothing counted."""
    items = [_item("a", 1), _item("b", 2), _item("c", 3)]
    drops = DropCounts()

    kept = drop_ambiguous_ids(items, drops)

    assert [i.item_id for i in kept] == ["a", "b", "c"]
    assert drops.ambiguous_item_id == 0


def test_collision_drops_every_claimant_not_just_the_duplicate() -> None:
    """Both rows sharing an id are dropped, since neither can be shown to own it.

    Keeping the first would be arbitrary and would silently attach one row's
    parameters to the other row's item.
    """
    items = [
        _item("shared", 9, a=3.9),
        _item("unique", 10, a=1.1),
        _item("shared", 3261, a=0.4),
    ]
    drops = DropCounts()

    kept = drop_ambiguous_ids(items, drops)

    assert [i.item_id for i in kept] == ["unique"]
    assert drops.ambiguous_item_id == 2


def test_three_way_collision_is_fully_removed() -> None:
    """A id claimed three times contributes three drops, not one."""
    items = [_item("x", 1), _item("x", 2), _item("x", 3), _item("y", 4)]
    drops = DropCounts()

    kept = drop_ambiguous_ids(items, drops)

    assert [i.item_id for i in kept] == ["y"]
    assert drops.ambiguous_item_id == 3


def test_drop_counter_is_reported_in_the_manifest_schema() -> None:
    """The count must survive into ``as_dict`` so vendoring stays fully accounted."""
    drops = DropCounts()
    drop_ambiguous_ids([_item("d", 1), _item("d", 2)], drops)

    assert drops.as_dict()["ambiguous_item_id"] == 2


#: GPQA's precedence: the most-determined tier first.
PRECEDENCE = ("extended", "main", "diamond")


def nested_spec(precedence: tuple[str, ...] = PRECEDENCE) -> DatasetSpec:
    """A GPQA-shaped spec; only the subtasks and the precedence are read here."""
    return DatasetSpec(
        name="fake_nested",
        task="",
        subtasks=(
            ("diamond", "gpqa_diamond"),
            ("extended", "gpqa_extended"),
            ("main", "gpqa_main"),
        ),
        route="B",
        bank_dir="fake",
        bridge_path="fake/item_id_map.csv",
        bridge_kind="item_id_map",
        fit_family="3pl",
        expected_bank_rows=1,
        positional_ids=True,
        notes="fixture",
        duplicate_question_precedence=precedence,
    )


def flat_spec() -> DatasetSpec:
    """The same shape with no precedence declared, as MuSR's and BBH's specs are."""
    return DatasetSpec(
        name="fake_flat",
        task="",
        subtasks=(("murder_mysteries", "musr_murder_mysteries"),),
        route="B",
        bank_dir="fake",
        bridge_path="fake/item_id_map.csv",
        bridge_kind="item_id_map",
        fit_family="3pl",
        expected_bank_rows=1,
        positional_ids=True,
        notes="fixture",
    )


class TestNestedSubsetsAreDeduplicated:
    """One calibration per question, chosen by a rule rather than by bank order."""

    def test_the_most_preferred_tier_wins_a_three_way_repeat(self) -> None:
        items = [_item("diamond|76", 76), _item("extended|265", 465), _item("main|200", 950)]
        questions = dict.fromkeys((i.item_id for i in items), "how many neutrinos")
        drops = DropCounts()

        kept = drop_duplicate_questions(nested_spec(), items, questions, drops)

        assert [i.item_id for i in kept] == ["extended|265"]
        assert drops.as_dict()["duplicate_question"] == 2

    def test_main_wins_when_the_question_never_reached_extended(self) -> None:
        """Precedence is over the tiers a question survives in, not over all three."""
        items = [_item("diamond|4", 4), _item("main|9", 800)]
        questions = dict.fromkeys((i.item_id for i in items), "which orbital")
        drops = DropCounts()

        kept = drop_duplicate_questions(nested_spec(), items, questions, drops)

        assert [i.item_id for i in kept] == ["main|9"]

    def test_a_question_surviving_only_in_diamond_is_kept(self) -> None:
        """The case the rule must not silently drop, and 24 real GPQA questions are it.

        Diamond is least preferred, so a lone diamond row means the extended and main
        calibrations of that question both failed the ``a > 0`` filter. Dropping every
        row of a repeated question -- the resolution the ambiguous-id path takes -- would
        take the question with them.
        """
        items = [_item("diamond|3", 3), _item("extended|7", 300), _item("main|8", 900)]
        questions = {
            "diamond|3": "only in diamond",
            "extended|7": "somewhere else",
            "main|8": "somewhere else",
        }
        drops = DropCounts()

        kept = drop_duplicate_questions(nested_spec(), items, questions, drops)

        assert [i.item_id for i in kept] == ["diamond|3", "extended|7"]
        assert drops.as_dict()["duplicate_question"] == 1

    def test_distinct_questions_are_left_alone(self) -> None:
        items = [_item("extended|0", 200), _item("main|0", 750)]
        questions = {"extended|0": "first", "main|0": "second"}
        drops = DropCounts()

        kept = drop_duplicate_questions(nested_spec(), items, questions, drops)

        assert [i.item_id for i in kept] == ["extended|0", "main|0"]
        assert drops.as_dict()["duplicate_question"] == 0

    def test_bank_order_is_preserved(self) -> None:
        """``params.json`` stays in bank order, so a diff against it stays readable."""
        items = [
            _item("diamond|1", 1),
            _item("extended|1", 201),
            _item("extended|2", 202),
            _item("main|1", 751),
        ]
        questions = {
            "diamond|1": "shared",
            "extended|1": "shared",
            "extended|2": "own",
            "main|1": "own too",
        }

        kept = drop_duplicate_questions(nested_spec(), items, questions, DropCounts())

        assert [i.item_id for i in kept] == ["extended|1", "extended|2", "main|1"]

    def test_the_choice_of_survivor_does_not_follow_bank_order(self) -> None:
        """Reversing the input must not reverse the outcome, or the rule is decorative."""
        forward = [_item("diamond|5", 5), _item("main|5", 800)]
        questions = dict.fromkeys(("diamond|5", "main|5"), "same question")

        kept_forward = drop_duplicate_questions(nested_spec(), forward, questions, DropCounts())
        kept_reverse = drop_duplicate_questions(
            nested_spec(), list(reversed(forward)), questions, DropCounts()
        )

        assert [i.item_id for i in kept_forward] == ["main|5"]
        assert [i.item_id for i in kept_reverse] == ["main|5"]


class TestThePreconditions:
    """Checked rather than assumed, because both failures are quiet."""

    def test_a_row_with_no_recorded_question_aborts(self) -> None:
        """One missing key exempts that row from the rule instead of failing."""
        items = [_item("extended|0", 200), _item("main|0", 750)]

        with pytest.raises(SystemExit, match="no recorded source question"):
            drop_duplicate_questions(nested_spec(), items, {"extended|0": "q"}, DropCounts())

    def test_one_subtask_calibrating_a_question_twice_aborts(self) -> None:
        """A repeat inside a quality filter is not a repeat across them.

        The precedence ranks tiers, so it cannot choose between two rows of one tier and
        the survivor would fall out of bank order. GPQA has none of these -- all three
        subsets enumerate distinct questions -- and an upstream revision that introduced
        one should stop vendoring rather than be resolved arbitrarily.
        """
        items = [_item("extended|0", 200), _item("extended|9", 209)]
        questions = dict.fromkeys(("extended|0", "extended|9"), "same question")

        with pytest.raises(SystemExit, match="more than once"):
            drop_duplicate_questions(nested_spec(), items, questions, DropCounts())

    def test_a_precedence_that_leaves_a_subtask_unranked_is_refused(self) -> None:
        """Caught building the spec, since an unranked tier has no defined survivor."""
        with pytest.raises(ValueError, match="must rank every subtask exactly once"):
            nested_spec(("extended", "main"))

    def test_a_precedence_naming_an_unknown_subtask_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must rank every subtask exactly once"):
            nested_spec(("extended", "main", "diamond", "platinum"))


class TestEveryOtherBankIsUntouched:
    """The rule is declared per dataset and eight of the nine banks do not declare it."""

    def test_only_gpqa_declares_a_precedence(self) -> None:
        declaring = [name for name, spec in SUPPORTED.items() if spec.duplicate_question_precedence]

        assert declaring == ["gpqa"]

    def test_a_spec_without_one_passes_its_rows_straight_through(self) -> None:
        """Including the repeats a nested bank would lose, and the identity is the point.

        BBH ships four literally duplicated items inside two of its subtasks. They are
        not nested calibrations of one question, they are the benchmark's own data, and a
        rule applied bank-wide would delete them.
        """
        items = [_item("murder_mysteries|0", 1), _item("murder_mysteries|1", 2)]
        questions = dict.fromkeys((i.item_id for i in items), "identical text")
        drops = DropCounts()

        kept = drop_duplicate_questions(flat_spec(), items, questions, drops)

        assert kept is items
        assert drops.as_dict()["duplicate_question"] == 0

    def test_no_other_committed_bank_records_the_drop(self) -> None:
        """Read off the artifacts, so a bank re-vendored under a stray declaration fails.

        Absent keys pass: eight of these manifests predate the field entirely, and the
        claim being made is that no bank but gpqa lost items this way rather than that
        every manifest mentions it.
        """
        for dataset in VENDORED:
            root = CALIBRATED_DATASETS / dataset
            if dataset == "gpqa" or not root.exists():
                continue
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            assert manifest["dropped"].get("duplicate_question", 0) == 0, dataset
            assert manifest.get("duplicate_question_precedence") is None, dataset


def test_subtask_counts_are_read_off_the_records_that_name_one() -> None:
    """What the manifest reports as ``subtask_items``, and it must skip records with none."""
    params = [
        {"item_id": "extended|0", "metadata": {"atlas_idx": 200, "subtask": "extended"}},
        {"item_id": "main|0", "metadata": {"atlas_idx": 750, "subtask": "main"}},
        {"item_id": "extended|1", "metadata": {"atlas_idx": 201, "subtask": "extended"}},
        {"item_id": "Mercury_7175875", "metadata": {"atlas_idx": 1}},
    ]

    assert subtask_item_counts(params) == {"extended": 2, "main": 1}
    assert subtask_item_counts([]) == {}


#: Every bank committed under ``calibrated_datasets/``, MCQ and generative alike. Taken
#: from the allowlist rather than listed, because a bank added there and forgotten here
#: would be the one bank these invariants are not asserted against.
VENDORED = supported_names()


@pytest.mark.parametrize("dataset", VENDORED)
def test_vendored_bank_is_one_to_one(dataset: str) -> None:
    """Every committed bank has exactly one params record per item, and no duplicates.

    This is the invariant the dedup exists to protect. It is asserted against the real
    committed artifacts, so a future re-vendor that reintroduces collisions fails here.
    """
    root = CALIBRATED_DATASETS / dataset
    if not root.exists():
        pytest.skip(f"{dataset} has not been vendored")

    params = json.loads((root / "params.json").read_text(encoding="utf-8"))
    items = [
        json.loads(line)
        for line in (root / "items.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    param_ids = [p["item_id"] for p in params]
    item_ids = [i["id"] for i in items]

    assert len(param_ids) == len(set(param_ids)), f"{dataset}: duplicate ids in params.json"
    assert len(item_ids) == len(set(item_ids)), f"{dataset}: duplicate ids in items.jsonl"
    assert set(param_ids) == set(item_ids), f"{dataset}: params and items describe different sets"


@pytest.mark.parametrize("dataset", VENDORED)
def test_manifest_accounts_for_every_upstream_row(dataset: str) -> None:
    """Kept items plus every drop reason must equal the upstream row count.

    ``ungradable_instances`` is deliberately not part of this sum: it counts task
    instances skipped during enumeration, not calibrated bank rows.
    """
    root = CALIBRATED_DATASETS / dataset
    if not root.exists():
        pytest.skip(f"{dataset} has not been vendored")

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    accounted = manifest["items"] + sum(manifest["dropped"].values())

    assert accounted == manifest["upstream_bank_rows"], (
        f"{dataset}: {manifest['items']} kept + {sum(manifest['dropped'].values())} dropped "
        f"!= {manifest['upstream_bank_rows']} upstream rows"
    )
