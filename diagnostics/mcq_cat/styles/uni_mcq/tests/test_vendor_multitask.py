"""Pin the multi-task vendoring path and the per-subtask guards that make it safe.

GPQA's and MuSR's banks were each calibrated across three tasks that upstream addressed
as one, and their bridges key each item by the composite ``<subtask>|<position>`` -- a
position numbered *within* a subtask, restarting at zero. Rebuilding those keys means
enumerating three separate registered tasks and counting each from scratch.

Every failure this path can suffer is silent rather than loud. A subtask the bridge
names but the spec never maps loses its whole share of the bank to ``not_in_task``,
which is indistinguishable from ordinary attrition. A task emitting one instance fewer
than the bridge expects shifts every later position onto its neighbour's question while
the join still reports near-total overlap. And because the dataset-level overlap floor
averages across subtasks, one subtask can drift badly while the dataset as a whole
stays comfortably above it. In all three cases the CAT would administer items whose
parameters belong to different questions and report a theta with a plausible standard
error, so the guards below are the only thing standing between a rekeyed bank and a
confident wrong number.

The registry is stubbed through ``sys.modules``, as elsewhere in these tests, so
nothing here needs the network or ``PYTHONPATH``.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from typing import Any

import pytest

from ..datasets import CONTENT_HASH, SUPPORTED, DatasetSpec, content_item_id
from ..scripts.vendor_bank import (
    BankItem,
    check_subtask_alignment,
    check_subtask_coverage,
    check_subtask_overlap,
    load_multi_task_items,
    load_task_items,
    subtask_row_counts,
)

#: The datasets vendored through the original single-task path, which this branch must
#: leave exactly as they were.
SINGLE_TASK = ("arc_challenge", "hellaswag", "winogrande", "gsm8k")


@dataclass
class FakeInstance:
    """The subset of an ``olmo_eval`` ``Instance`` that vendoring actually reads."""

    question: str
    choices: tuple[str, ...] | None = ("alpha", "beta")
    gold_answer: str | None = None
    metadata: dict[str, Any] = field(default_factory=lambda: {"gold_idx": 0})


class FakeTask:
    """A task whose enumeration is fixed in the test, so nothing touches the network."""

    def __init__(self, instances: list[FakeInstance]) -> None:
        self.instances = instances


@pytest.fixture
def registry(monkeypatch):
    """Return a callable installing a stub registry over the given task enumerations.

    Enumerating the real GPQA subsets is not an option even offline: they sit behind a
    HuggingFace access request, so a test that reached for them could never run.
    """

    def install(tasks: dict[str, list[FakeInstance]]) -> None:
        module = types.ModuleType("olmo_eval.evals.tasks.common.registry")

        def get_task(name: str) -> FakeTask:
            if name not in tasks:
                raise KeyError(f"Unknown task '{name}'. Available: {', '.join(sorted(tasks))}")
            return FakeTask(tasks[name])

        module.get_task = get_task
        monkeypatch.setitem(sys.modules, "olmo_eval.evals.tasks.common.registry", module)

    return install


def multi_spec(subtasks: tuple[tuple[str, str], ...]) -> DatasetSpec:
    """A multi-task spec shaped like GPQA's; only the join fields are read here."""
    return DatasetSpec(
        name="fake_multi",
        task="",
        subtasks=subtasks,
        route="B",
        bank_dir="fake",
        bridge_path="fake/item_id_map.csv",
        bridge_kind="item_id_map",
        fit_family="3pl",
        expected_bank_rows=1,
        positional_ids=True,
        notes="fixture",
    )


def content_spec(subtasks: tuple[tuple[str, str], ...]) -> DatasetSpec:
    """A multi-task spec shaped like MuSR's after the re-key: subtasks, content hash."""
    return DatasetSpec(
        name="fake_content",
        task="",
        subtasks=subtasks,
        route="B",
        bank_dir="fake",
        bridge_path="fake/bridges/fake.csv",
        bridge_in_repo=True,
        bridge_kind=CONTENT_HASH,
        fit_family="3pl",
        expected_bank_rows=1,
        positional_ids=False,
        notes="fixture",
    )


def single_spec() -> DatasetSpec:
    """A single-task spec, unchanged in shape from before multi-task existed."""
    return DatasetSpec(
        name="fake_single",
        task="fake",
        route="C",
        bank_dir="fake",
        bridge_path="fake/atlas_idx_to_question_id.csv",
        bridge_kind="atlas",
        fit_family="3pl",
        expected_bank_rows=1,
        positional_ids=False,
        notes="fixture",
    )


def fake_bridge(counts: dict[str, int]) -> dict[int, str]:
    """An ``item_id_map`` bridge: composite ids, positions restarting per subtask."""
    bridge: dict[int, str] = {}
    atlas_idx = 1
    for label, rows in counts.items():
        for position in range(rows):
            bridge[atlas_idx] = f"{label}|{position}"
            atlas_idx += 1
    return bridge


def bank(item_ids: list[str]) -> list[BankItem]:
    """Converted bank rows carrying only the ids the join and the guards read."""
    return [
        BankItem(
            item_id=item_id,
            atlas_idx=index,
            discrimination=1.0,
            difficulty=0.0,
            guessing=0.0,
        )
        for index, item_id in enumerate(item_ids, start=1)
    ]


class TestCompositeKeys:
    def test_position_restarts_at_zero_for_each_subtask(self, registry) -> None:
        """The whole join rests on this: ``question_id`` is per-subtask, not global."""
        registry(
            {
                "gpqa_diamond": [FakeInstance(question="d0"), FakeInstance(question="d1")],
                "gpqa_main": [FakeInstance(question="m0")],
            }
        )
        spec = multi_spec((("diamond", "gpqa_diamond"), ("main", "gpqa_main")))

        items, ungradable = load_multi_task_items(spec, {"diamond": 2, "main": 1})

        assert list(items) == ["diamond|0", "diamond|1", "main|0"]
        assert items["main|0"]["question"] == "m0"
        assert ungradable == 0

    def test_the_instances_own_id_is_ignored(self, registry) -> None:
        """A native id cannot join to a composite key, so position always wins.

        GPQA exposes ``metadata["index"]`` rather than ``metadata["id"]``, so the
        single-task path's fallback would happen to produce the same numbers today.
        Depending on that coincidence would make the join break the moment the task
        starts emitting a real id.
        """
        registry(
            {
                "gpqa_diamond": [
                    FakeInstance(question="d0", metadata={"id": "recW3elXvOhGRuwMh", "gold_idx": 0})
                ]
            }
        )
        spec = multi_spec((("diamond", "gpqa_diamond"),))

        items, _ = load_multi_task_items(spec, {"diamond": 1})

        assert list(items) == ["diamond|0"]

    def test_an_ungradable_instance_does_not_shift_later_positions(self, registry) -> None:
        """Positions count the enumeration, not the survivors.

        Renumbering around a skipped instance would rekey everything after it, which is
        exactly the corruption the alignment guard exists to prevent.
        """
        registry(
            {
                "gpqa_main": [
                    FakeInstance(question="m0"),
                    FakeInstance(question="m1", choices=()),
                    FakeInstance(question="m2"),
                ]
            }
        )
        spec = multi_spec((("main", "gpqa_main"),))

        items, ungradable = load_multi_task_items(spec, {"main": 3})

        assert list(items) == ["main|0", "main|2"]
        assert ungradable == 1


class TestContentKeyedMultiTask:
    """The same three-task join, with the position taken out of the key.

    What changes is which failures are possible rather than how many guards there are. A
    short enumeration no longer rekeys everything after it, so there is no span to hold
    a task to; instead the missing items simply fail to join and the overlap floor turns
    that into an abort. What has to keep working is the part the composite id used to
    give away for free: which subtask a row belongs to.
    """

    def test_items_are_keyed_by_their_text_across_every_subtask(self, registry) -> None:
        registry(
            {
                "musr_murder_mysteries": [FakeInstance(question="who did it")],
                "musr_team_allocation": [FakeInstance(question="who does what")],
            }
        )
        spec = content_spec(
            (
                ("murder_mysteries", "musr_murder_mysteries"),
                ("team_allocation", "musr_team_allocation"),
            )
        )

        items, ungradable = load_multi_task_items(spec, {})

        assert set(items) == {
            content_item_id("who did it", ("alpha", "beta")),
            content_item_id("who does what", ("alpha", "beta")),
        }
        assert ungradable == 0

    def test_a_short_enumeration_is_a_miss_rather_than_a_rekey(self, registry) -> None:
        """The point of the re-key, stated as the behaviour it buys.

        Under the composite key an enumeration one instance short renamed every item
        after the shortfall onto its neighbour's question, which is why that path is held
        to a span. Here the dropped instance is simply absent and every other id is
        untouched, so the loss shows up in the overlap floor instead of in the theta.
        """
        registry({"musr_team_allocation": [FakeInstance(question="q0")]})
        spec = content_spec((("team_allocation", "musr_team_allocation"),))

        items, _ = load_multi_task_items(spec, {})

        assert list(items) == [content_item_id("q0", ("alpha", "beta"))]

    def test_two_subtasks_rendering_one_item_abort(self, registry) -> None:
        """A hash that names two items cannot be scored as either of them."""
        registry(
            {
                "musr_murder_mysteries": [FakeInstance(question="same")],
                "musr_team_allocation": [FakeInstance(question="same")],
            }
        )
        spec = content_spec(
            (
                ("murder_mysteries", "musr_murder_mysteries"),
                ("team_allocation", "musr_team_allocation"),
            )
        )

        with pytest.raises(SystemExit, match="already answers to"):
            load_multi_task_items(spec, {})

    def test_the_overlap_floor_groups_by_the_bridge_column(self) -> None:
        """A hashed id says nothing about its subtask, so the bridge has to.

        Without the label carried on the bank row this reduces to one bucket, the drifted
        subtask is averaged into the healthy one, and the guard passes while a third of
        the administered items are scored against the wrong questions.
        """
        bank_items = [
            BankItem(
                item_id=f"hash{i}",
                atlas_idx=i,
                discrimination=1.0,
                difficulty=0.0,
                guessing=0.0,
                subtask="murder_mysteries" if i < 10 else "team_allocation",
            )
            for i in range(100)
        ]
        matched = [f"hash{i}" for i in range(5, 100)]

        with pytest.raises(SystemExit, match="'murder_mysteries' matched only 50.0%"):
            check_subtask_overlap(
                content_spec((("murder_mysteries", "a"), ("team_allocation", "b"))),
                bank_items,
                matched,
                min_overlap=0.90,
            )

    def test_coverage_still_refuses_a_subtask_no_spec_maps(self) -> None:
        """The guard that survives the re-key unchanged, because its failure does too.

        A bridge subtask with no declared task loses its whole share of the bank to
        ``not_in_task`` whatever the key is, and that reads as ordinary attrition.
        """
        spec = content_spec((("murder_mysteries", "musr_murder_mysteries"),))

        with pytest.raises(SystemExit, match="team_allocation"):
            check_subtask_coverage(spec, ["murder_mysteries", "team_allocation"])

    def test_coverage_refuses_a_spec_the_bridge_never_names(self) -> None:
        spec = content_spec(
            (
                ("murder_mysteries", "musr_murder_mysteries"),
                ("team_allocation", "musr_team_allocation"),
            )
        )

        with pytest.raises(SystemExit, match="team_allocation"):
            check_subtask_coverage(spec, ["murder_mysteries"])


class TestBridgeAgainstSpec:
    def test_row_counts_are_read_off_the_composite_ids(self) -> None:
        counts = subtask_row_counts(fake_bridge({"diamond": 198, "extended": 546, "main": 448}))

        assert counts == {"diamond": 198, "extended": 546, "main": 448}

    def test_alignment_returns_the_counts_when_spec_and_bridge_agree(self) -> None:
        spec = multi_spec((("diamond", "gpqa_diamond"), ("main", "gpqa_main")))

        assert check_subtask_alignment(spec, fake_bridge({"diamond": 3, "main": 2})) == {
            "diamond": 3,
            "main": 2,
        }

    def test_bridge_subtask_with_no_declared_task_aborts(self) -> None:
        """Otherwise a whole subtask's bank rows vanish into ``not_in_task``."""
        spec = multi_spec((("diamond", "gpqa_diamond"),))

        with pytest.raises(SystemExit, match="extended"):
            check_subtask_alignment(spec, fake_bridge({"diamond": 3, "extended": 2}))

    def test_declared_subtask_absent_from_the_bridge_aborts(self) -> None:
        """A spec naming a subtask the bridge never mentions is stale, not harmless."""
        spec = multi_spec((("diamond", "gpqa_diamond"), ("main", "gpqa_main")))

        with pytest.raises(SystemExit, match="main"):
            check_subtask_alignment(spec, fake_bridge({"diamond": 3}))

    def test_a_gap_is_tolerated_and_reported_as_a_span(self) -> None:
        """Calibration drops items, so a sparse subtask is upstream's right, not an error.

        What the join is rebuilt against is the enumeration the positions were numbered
        in, which is one past the highest of them. MuSR's ``object_placements`` is this
        case on real data: 254 rows over a 256-long split.
        """
        spec = multi_spec((("diamond", "gpqa_diamond"),))
        bridge = {1: "diamond|0", 2: "diamond|1", 3: "diamond|7"}

        assert check_subtask_alignment(spec, bridge) == {"diamond": 8}

    def test_a_duplicated_position_still_aborts(self) -> None:
        """Two bank rows claiming one question cannot both be right, sparse or not."""
        spec = multi_spec((("diamond", "gpqa_diamond"),))
        bridge = {1: "diamond|0", 2: "diamond|1", 3: "diamond|1"}

        with pytest.raises(SystemExit, match="more than once"):
            check_subtask_alignment(spec, bridge)

    def test_a_negative_position_still_aborts(self) -> None:
        """Positions are rebuilt by counting from zero, so there is nothing below it."""
        spec = multi_spec((("diamond", "gpqa_diamond"),))
        bridge = {1: "diamond|-1", 2: "diamond|0"}

        with pytest.raises(SystemExit, match="negative"):
            check_subtask_alignment(spec, bridge)


class TestMissingRegisteredTask:
    def test_a_declared_task_the_registry_does_not_know_aborts(self, registry) -> None:
        """The bridge names the subtask, the spec maps it, and the task is not there.

        This is the failure GPQA would have hit had the registry exposed a single
        combined task: the name reads plausibly and only the registry knows better.
        """
        registry({"gpqa_diamond": [FakeInstance(question="d0")]})
        spec = multi_spec((("diamond", "gpqa_diamond"), ("extended", "gpqa_extended")))

        with pytest.raises(SystemExit, match="gpqa_extended"):
            load_multi_task_items(spec, {"diamond": 1, "extended": 1})

    def test_a_short_enumeration_aborts(self, registry) -> None:
        """One instance fewer than the bridge expects rekeys the rest of the subtask.

        The overlap floor cannot catch this on its own: only the final position fails
        to join, so the join still reports well above 90% while every item after the
        missing one carries its neighbour's parameters.
        """
        registry({"gpqa_main": [FakeInstance(question=f"m{i}") for i in range(447)]})
        spec = multi_spec((("main", "gpqa_main"),))

        with pytest.raises(SystemExit, match="447 instances"):
            load_multi_task_items(spec, {"main": 448})


class TestPerSubtaskOverlap:
    def test_a_drifted_subtask_fails_while_the_global_overlap_passes(self) -> None:
        """The case the global floor is blind to, and the reason this guard exists.

        Diamond holds a tenth of this bank and matches half of it. The dataset overall
        is at 95%, clearing the 90% floor, while a twentieth of the administered items
        would be scored against the wrong questions.
        """
        bank_items = bank([f"diamond|{i}" for i in range(10)] + [f"main|{i}" for i in range(90)])
        matched = [f"diamond|{i}" for i in range(5)] + [f"main|{i}" for i in range(90)]
        assert len(matched) / len(bank_items) == 0.95

        with pytest.raises(SystemExit, match="'diamond' matched only 50.0%"):
            check_subtask_overlap(
                multi_spec((("diamond", "gpqa_diamond"), ("main", "gpqa_main"))),
                bank_items,
                matched,
                min_overlap=0.90,
            )

    def test_every_subtask_above_the_floor_passes(self) -> None:
        bank_items = bank([f"diamond|{i}" for i in range(10)] + [f"main|{i}" for i in range(10)])
        matched = [item.item_id for item in bank_items]

        check_subtask_overlap(
            multi_spec((("diamond", "gpqa_diamond"), ("main", "gpqa_main"))),
            bank_items,
            matched,
            min_overlap=0.90,
        )

    def test_the_floor_is_not_relaxed_by_a_subtask_holding_few_items(self) -> None:
        """A small subtask is still scored, so it gets the same floor as a large one."""
        bank_items = bank(["diamond|0"] + [f"main|{i}" for i in range(500)])
        matched = [f"main|{i}" for i in range(500)]

        with pytest.raises(SystemExit, match="'diamond' matched only 0.0%"):
            check_subtask_overlap(
                multi_spec((("diamond", "gpqa_diamond"), ("main", "gpqa_main"))),
                bank_items,
                matched,
                min_overlap=0.90,
            )


class TestASparseBridgeJoins:
    """A subtask whose calibration dropped items, and what the tolerance does not cost.

    MuSR's ``object_placements`` has 254 bridge rows over positions 0..255, because two
    items failed to converge during the fit. Nothing about that is a fault: the two
    missing positions simply have no bank row, and the 254 that remain still name the
    questions the enumeration will produce. Demanding a gapless run refused the whole
    dataset over it, and comparing the enumeration against the bridge's *row* count
    refused it again one step later.

    What the two guards now compare is the span, one past the highest position, which is
    the length of the enumeration the bridge was numbered against. The tests below pair
    each relaxation with the misalignment it must still catch, because a guard that stops
    firing is indistinguishable in a passing suite from one that was never needed.
    """

    #: 254-of-256 in miniature: four rows over a six-long enumeration, gaps at 3 and 4.
    SPARSE = {1: "op|0", 2: "op|1", 3: "op|2", 4: "op|5"}
    SPAN = 6

    def spec(self) -> DatasetSpec:
        return multi_spec((("op", "musr_object_placements"),))

    def test_alignment_reports_the_span_rather_than_the_row_count(self) -> None:
        assert check_subtask_alignment(self.spec(), self.SPARSE) == {"op": self.SPAN}
        assert subtask_row_counts(self.SPARSE) == {"op": 4}

    def test_the_enumeration_is_held_to_the_span(self, registry) -> None:
        """The 256 instances the task yields must be accepted against 254 rows."""
        registry(
            {"musr_object_placements": [FakeInstance(question=f"o{i}") for i in range(self.SPAN)]}
        )
        spans = check_subtask_alignment(self.spec(), self.SPARSE)

        items, ungradable = load_multi_task_items(self.spec(), spans)

        assert list(items) == [f"op|{i}" for i in range(self.SPAN)]
        assert ungradable == 0

    def test_the_sparse_positions_are_the_ones_that_join(self, registry) -> None:
        """Every bridge row finds its question, and the gaps stay empty rather than shift."""
        registry(
            {"musr_object_placements": [FakeInstance(question=f"o{i}") for i in range(self.SPAN)]}
        )
        spans = check_subtask_alignment(self.spec(), self.SPARSE)
        items, _ = load_multi_task_items(self.spec(), spans)

        assert {items[item_id]["question"] for item_id in self.SPARSE.values()} == {
            "o0",
            "o1",
            "o2",
            "o5",
        }

    def test_a_short_enumeration_under_a_sparse_bridge_still_aborts(self, registry) -> None:
        """The misalignment the tolerance must not admit.

        A task one instance shorter than the span rekeys everything after the shortfall,
        and because the bridge is sparse the row count alone can no longer tell. Holding
        the enumeration to the span is what keeps this loud.
        """
        registry(
            {
                "musr_object_placements": [
                    FakeInstance(question=f"o{i}") for i in range(self.SPAN - 1)
                ]
            }
        )
        spans = check_subtask_alignment(self.spec(), self.SPARSE)

        with pytest.raises(SystemExit, match="positions run to 5"):
            load_multi_task_items(self.spec(), spans)

    def test_the_overlap_floor_is_unweakened_on_a_sparse_subtask(self) -> None:
        """The real guard, and it counts bank rows rather than bridge positions.

        A sparse bridge changes which positions exist; it does not change the rule that
        a subtask must match nearly all of the bank rows it does have. Here three of the
        four calibrated rows join, which is 75% and still refused.
        """
        bank_items = bank(["op|0", "op|1", "op|2", "op|5"])

        with pytest.raises(SystemExit, match="'op' matched only 75.0%"):
            check_subtask_overlap(
                self.spec(), bank_items, ["op|0", "op|1", "op|2"], min_overlap=0.90
            )

    def test_a_fully_joined_sparse_subtask_clears_it(self) -> None:
        bank_items = bank(["op|0", "op|1", "op|2", "op|5"])
        check_subtask_overlap(
            self.spec(), bank_items, [item.item_id for item in bank_items], min_overlap=0.90
        )


class TestSingleTaskPathIsUnchanged:
    def test_a_single_task_dataset_still_keys_by_its_own_id(self, registry) -> None:
        """No composite key, no position: the original join, byte for byte."""
        registry(
            {
                "fake": [
                    FakeInstance(
                        question="Which property is a chemical one?",
                        choices=("mass", "flammability"),
                        metadata={"id": "Mercury_7175875", "gold_idx": 1},
                    )
                ]
            }
        )

        items, ungradable = load_task_items(single_spec())

        assert items == {
            "Mercury_7175875": {
                "id": "Mercury_7175875",
                "question": "Which property is a chemical one?",
                "choices": ["mass", "flammability"],
                "gold_index": 1,
            }
        }
        assert ungradable == 0

    @pytest.mark.parametrize("dataset", SINGLE_TASK)
    def test_the_vendored_datasets_declare_no_subtasks(self, dataset: str) -> None:
        """Their committed banks were joined by the single-task path and stay valid."""
        spec = SUPPORTED[dataset]

        assert spec.subtasks == ()
        assert spec.is_multi_task is False
        assert spec.task_names == (spec.task,)

    def test_the_multi_task_datasets_are_the_three_that_span_subtasks(self) -> None:
        multi = sorted(name for name, spec in SUPPORTED.items() if spec.is_multi_task)

        assert multi == ["bbh", "gpqa", "musr"]

    def test_bbh_maps_each_of_its_twenty_four_bridge_subtasks_to_a_registered_task(self) -> None:
        """Uniform ``bbh_<label>``, which is why the pairs are generated rather than typed.

        gpqa's labels differ from its task names and musr's happen to match; bbh's match
        for all 24, so listing them twice would be 24 more chances to mistype one, and a
        mistyped label loses that subtask's whole share of the bank to not_in_task.
        """
        spec = SUPPORTED["bbh"]

        assert len(spec.subtasks) == 24
        assert all(task == f"bbh_{label}" for label, task in spec.subtasks)
        assert spec.subtasks[0] == ("boolean_expressions", "bbh_boolean_expressions")
        assert spec.subtasks[-1] == ("web_of_lies", "bbh_web_of_lies")

    def test_bbh_declares_none_of_the_three_generative_subtasks(self) -> None:
        """They have no closed answer set, so leaderboard v2 never harvested them.

        Naming one here would demand a task nothing registers and abort vendoring, which
        is the loud failure; the quiet one is the reverse, and ``check_subtask_alignment``
        covers it by refusing a bridge subtask no spec maps.
        """
        labels = {label for label, _ in SUPPORTED["bbh"].subtasks}

        assert labels.isdisjoint({"dyck_languages", "multistep_arithmetic_two", "word_sorting"})

    def test_gpqa_maps_each_bridge_subtask_to_its_registered_task(self) -> None:
        """The three names are what the registry exposes; there is no combined task."""
        assert SUPPORTED["gpqa"].subtasks == (
            ("diamond", "gpqa_diamond"),
            ("extended", "gpqa_extended"),
            ("main", "gpqa_main"),
        )
        assert SUPPORTED["gpqa"].task_names == ("gpqa_diamond", "gpqa_extended", "gpqa_main")

    def test_musr_maps_each_bridge_subtask_to_its_registered_task(self) -> None:
        """Its labels are HuggingFace split names, and the bridge spells them the same."""
        assert SUPPORTED["musr"].subtasks == (
            ("murder_mysteries", "musr_murder_mysteries"),
            ("object_placements", "musr_object_placements"),
            ("team_allocation", "musr_team_allocation"),
        )

    @pytest.mark.parametrize("dataset", ["gpqa", "bbh"])
    def test_a_positional_composite_join_demands_the_overlap_floor(self, dataset: str) -> None:
        """Rebuilt positions name nothing about the item, so a shift joins silently."""
        assert SUPPORTED[dataset].positional_ids
        assert SUPPORTED[dataset].needs_overlap_floor

    def test_musr_is_the_multi_task_bank_re_keyed_by_content(self) -> None:
        """It still spans three tasks and no longer joins by a position within them.

        The two halves have to move together. Dropping ``positional_ids`` without the
        content key would leave a positional join with its floor switched off; keeping
        the composite key while claiming a content hash would join nothing at all.
        """
        spec = SUPPORTED["musr"]

        assert spec.is_multi_task
        assert spec.bridge_kind == CONTENT_HASH
        assert spec.bridge_in_repo
        assert not spec.positional_ids
        assert spec.needs_overlap_floor


class TestSpecValidation:
    def test_a_spec_naming_neither_one_task_nor_several_is_rejected(self) -> None:
        """Left unchecked this asks the registry for the task named by the empty string."""
        with pytest.raises(ValueError, match="exactly one of task and subtasks"):
            DatasetSpec(
                name="fake",
                task="",
                route="C",
                bank_dir="fake",
                bridge_path="fake",
                bridge_kind="atlas",
                fit_family="3pl",
                expected_bank_rows=1,
                positional_ids=False,
                notes="fixture",
            )

    def test_a_spec_naming_both_is_rejected(self) -> None:
        """Vendoring would take one branch and silently ignore the other declaration."""
        with pytest.raises(ValueError, match="exactly one of task and subtasks"):
            DatasetSpec(
                name="fake",
                task="gpqa_main",
                subtasks=(("main", "gpqa_main"),),
                route="B",
                bank_dir="fake",
                bridge_path="fake",
                bridge_kind="item_id_map",
                fit_family="3pl",
                expected_bank_rows=1,
                positional_ids=True,
                notes="fixture",
            )
