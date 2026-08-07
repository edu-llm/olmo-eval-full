"""Pin the generative vendoring path and the record shapes it hands downstream.

A generative item is written with an empty ``choices`` and ``gold_index = -1``. That
pair is not filler: ``load_items_from_jsonl`` in the frozen common layer copies the
record into a :class:`~diagnostics.mcq_cat.base.BenchmarkItem` verbatim, and an item
with no choices is how the run-time grader knows to sample a completion rather than
rank continuation log-likelihoods over a choice set that does not exist. If either
value drifts, a GSM8K run would try to score an empty choice list and either crash or,
worse, score every item incorrect and report a confidently low ability.

The tests below therefore assert the record shape literally, on both branches: an MCQ
dataset must keep the exact keys the already-vendored banks were written with, since
changing them would silently invalidate artifacts nobody re-vendored.
"""

from __future__ import annotations

import json
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from ....base import BenchmarkItem
from ....common import generative
from .. import datasets, resolve
from ..datasets import DatasetSpec
from ..scripts.vendor_bank import (
    DropCounts,
    VendorResult,
    answer_type,
    load_task_items,
    write_artifacts,
)

CALIBRATED_DATASETS = Path(__file__).resolve().parents[5] / "calibrated_datasets"

#: Datasets this branch vendored, and the modality each was vendored under.
VENDORED = {
    "winogrande": "mcq",
    "gsm8k": "generative",
    "leaderboard_math": "generative",
    "ifeval": "generative",
}


@dataclass
class FakeInstance:
    """The subset of an ``olmo_eval`` ``Instance`` that vendoring actually reads."""

    question: str
    choices: tuple[str, ...] | None = None
    gold_answer: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class FakeTask:
    """A task whose enumeration is fixed in the test, so nothing touches the network."""

    def __init__(self, instances: list[FakeInstance]) -> None:
        self.instances = instances


def _spec(modality: str, answer_type: str = "numeric") -> DatasetSpec:
    """A minimal spec; only ``name``, ``task``, ``modality`` and ``answer_type`` are read."""
    return DatasetSpec(
        name="fake",
        task="fake",
        route="C",
        bank_dir="fake",
        bridge_path="fake/atlas_idx_to_question_id.csv",
        bridge_kind="atlas",
        fit_family="3pl",
        expected_bank_rows=1,
        positional_ids=False,
        notes="fixture",
        modality=modality,
        answer_type=answer_type,
    )


@pytest.fixture
def enumerate_task(monkeypatch):
    """Return a callable that enumerates ``instances`` under a given modality.

    Stubs the registry module rather than importing ``olmo_eval``, so these tests run
    in any environment and never download a dataset to check a record's shape.
    """

    def run(
        modality: str, instances: list[FakeInstance], answer_type: str = "numeric"
    ) -> tuple[dict, int]:
        registry = types.ModuleType("olmo_eval.evals.tasks.common.registry")
        registry.get_task = lambda name: FakeTask(instances)
        monkeypatch.setitem(sys.modules, "olmo_eval.evals.tasks.common.registry", registry)
        return load_task_items(_spec(modality, answer_type))

    return run


class TestRecordShapes:
    def test_generative_record_is_exactly_what_the_grader_reads(self, enumerate_task) -> None:
        """The literal record, keys and all. This is a cross-component contract."""
        instance = FakeInstance(
            question="Natalia sold clips to 48 of her friends...",
            gold_answer="72",
            metadata={"id": 0},
        )

        items, dropped = enumerate_task("generative", [instance])

        assert dropped == 0
        assert items["0"] == {
            "id": "0",
            "question": "Natalia sold clips to 48 of her friends...",
            "choices": [],
            "gold_index": -1,
            "metadata": {
                "gold_answer": "72",
                "answer_type": "numeric",
                "modality": "generative",
            },
        }

    def test_a_verifier_scored_record_carries_constraints_instead_of_a_gold(
        self, enumerate_task
    ) -> None:
        """IFEval's shape, literally. No gold_answer key at all, and no empty one.

        An empty string there would satisfy every ``metadata.get("gold_answer")`` test
        in the harness while matching nothing, so the absence has to be real.
        """
        instance = FakeInstance(
            question="Write a summary with no commas.",
            gold_answer=None,
            metadata={
                "id": 0,
                "instruction_id_list": ["punctuation:no_comma"],
                "kwargs": [{}],
                "prompt": "Write a summary with no commas.",
                "key": 1000,
            },
        )

        items, dropped = enumerate_task("generative", [instance], answer_type="ifeval_strict")

        assert dropped == 0
        assert items["0"] == {
            "id": "0",
            "question": "Write a summary with no commas.",
            "choices": [],
            "gold_index": -1,
            "metadata": {
                "answer_type": "ifeval_strict",
                "modality": "generative",
                "instruction_id_list": ["punctuation:no_comma"],
                "kwargs": [{}],
            },
        }

    def test_only_the_metadata_the_grader_declares_is_copied(self, enumerate_task) -> None:
        """Everything else the task carries stays out of the bank.

        ``key`` and ``prompt`` are on the instance and are not copied: ``prompt``
        duplicates the question, and ``key`` is the dataset's own sparse id, which is
        not the join key and would invite someone to treat it as one.
        """
        instance = FakeInstance(
            question="q",
            metadata={
                "id": 0,
                "instruction_id_list": ["punctuation:no_comma"],
                "kwargs": [{}],
                "prompt": "q",
                "key": 1000,
            },
        )

        items, _ = enumerate_task("generative", [instance], answer_type="ifeval_strict")

        assert set(items["0"]["metadata"]) == {
            "answer_type",
            "modality",
            "instruction_id_list",
            "kwargs",
        }

    def test_mcq_record_is_unchanged(self, enumerate_task) -> None:
        """No new keys and no reordering: the committed banks must stay valid.

        arc_challenge and hellaswag were vendored before the generative branch existed
        and are not re-vendored here, so any change to this shape would leave the two
        halves of ``calibrated_datasets/`` describing items differently.
        """
        instance = FakeInstance(
            question="Which property is a chemical one?",
            choices=("mass", "flammability"),
            metadata={"id": "Mercury_7175875", "gold_idx": 1},
        )

        items, dropped = enumerate_task("mcq", [instance])

        assert dropped == 0
        assert items["Mercury_7175875"] == {
            "id": "Mercury_7175875",
            "question": "Which property is a chemical one?",
            "choices": ["mass", "flammability"],
            "gold_index": 1,
        }

    def test_mcq_gold_falls_back_to_matching_the_answer_text(self, enumerate_task) -> None:
        """Tasks that carry only the answer string still yield a usable gold index."""
        instance = FakeInstance(
            question="Pick one.",
            choices=("alpha", "beta"),
            gold_answer="beta",
            metadata={"id": "7"},
        )

        items, _ = enumerate_task("mcq", [instance])

        assert items["7"]["gold_index"] == 1


class TestUngradableInstances:
    def test_no_choices_and_no_gold_answer_is_dropped_and_counted(self, enumerate_task) -> None:
        """Nothing to rank and nothing to match, so the item can never be answered."""
        instances = [
            FakeInstance(question="gradable", gold_answer="42", metadata={"id": 0}),
            FakeInstance(question="ungradable", metadata={"id": 1}),
        ]

        items, dropped = enumerate_task("generative", instances)

        assert list(items) == ["0"]
        assert dropped == 1

    def test_blank_gold_answer_does_not_count_as_an_answer(self, enumerate_task) -> None:
        """Whitespace would exact-match nothing, so it is a drop rather than a gold."""
        items, dropped = enumerate_task(
            "generative",
            [FakeInstance(question="q", gold_answer="   ", metadata={"id": 0})],
        )

        assert items == {}
        assert dropped == 1

    def test_mcq_instance_without_choices_is_dropped_and_counted(self, enumerate_task) -> None:
        """The MCQ branch skipped these silently before; the count makes it visible."""
        items, dropped = enumerate_task(
            "mcq",
            [FakeInstance(question="q", gold_answer="42", metadata={"id": 0})],
        )

        assert items == {}
        assert dropped == 1

    def test_unknown_modality_aborts(self, enumerate_task) -> None:
        """Guessing at a grading convention is worse than refusing to vendor."""
        with pytest.raises(SystemExit, match="unknown modality"):
            enumerate_task("essay", [FakeInstance(question="q", metadata={"id": 0})])

    def test_an_unknown_answer_type_aborts_before_enumeration(self, enumerate_task) -> None:
        """Named before the task is even asked for, so no dataset is downloaded first."""
        with pytest.raises(ValueError, match="No answer grader for answer_type"):
            enumerate_task(
                "generative",
                [FakeInstance(question="q", gold_answer="1", metadata={"id": 0})],
                answer_type="essay_rubric",
            )

    def test_a_verifier_item_missing_its_constraints_is_dropped(self, enumerate_task) -> None:
        """An empty instruction list can never produce an outcome, so it is not an item.

        Emitting it would put an item in the bank that the grader refuses at run time,
        turning a vendoring fault into a mid-session crash on whichever run happened to
        select it.
        """
        items, dropped = enumerate_task(
            "generative",
            [
                FakeInstance(
                    question="q", metadata={"id": 0, "instruction_id_list": [], "kwargs": []}
                )
            ],
            answer_type="ifeval_strict",
        )

        assert items == {}
        assert dropped == 1


class TestAnswerType:
    @pytest.mark.parametrize("gold", ["72", "-3", "0.5", "1e3"])
    def test_numbers_are_numeric(self, gold: str) -> None:
        assert answer_type(gold) == "numeric"

    @pytest.mark.parametrize("gold", ["Paris", "72 clips", ""])
    def test_everything_else_is_text(self, gold: str) -> None:
        """Mislabelling prose as numeric would have the grader compare parsed digits."""
        assert answer_type(gold) == "text"


class TestManifest:
    def test_manifest_records_the_modality(self, tmp_path: Path) -> None:
        """The manifest is how a run learns which grader the bank was built for."""
        spec = _spec("generative")
        result = VendorResult(
            items=[{"id": "0", "question": "q", "choices": [], "gold_index": -1}],
            params=[{"item_id": "0", "difficulty": 0.0, "discrimination": 1.0, "guessing": 0.0}],
            drops=DropCounts(),
            upstream_bank_rows=1,
            bridge_rows=1,
            ungradable_instances=3,
        )

        write_artifacts(
            spec,
            result,
            source_ref="test",
            source_commit="0" * 40,
            fit_family="3pl",
            out_dir=tmp_path,
        )
        manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))

        assert manifest["modality"] == "generative"
        assert manifest["ungradable_instances"] == 3

    def test_ungradable_instances_stay_out_of_the_drop_accounting(self, tmp_path: Path) -> None:
        """``dropped`` must keep summing to the upstream row count.

        Enumeration skips are not upstream bank rows. A skipped instance already shows
        up as ``not_in_task`` if the bank calibrated it, so counting it in ``dropped``
        too would double-count and break the identity a manifest is checked against.
        """
        result = VendorResult(
            items=[{"id": "0", "question": "q", "choices": [], "gold_index": -1}],
            params=[{"item_id": "0", "difficulty": 0.0, "discrimination": 1.0, "guessing": 0.0}],
            drops=DropCounts(not_in_task=2),
            upstream_bank_rows=3,
            bridge_rows=5,
            ungradable_instances=2,
        )

        write_artifacts(
            _spec("generative"),
            result,
            source_ref="test",
            source_commit="0" * 40,
            fit_family="3pl",
            out_dir=tmp_path,
        )
        manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))

        assert manifest["items"] + sum(manifest["dropped"].values()) == 3


@pytest.mark.parametrize("dataset", sorted(VENDORED))
class TestVendoredDatasets:
    """The datasets this branch vendored, asserted against the committed artifacts.

    Two of the four are blocked, and their artifacts are still committed and still
    checked here. The blocker is about the bridge attributing parameters to the wrong
    questions, which leaves the record *shape* -- empty choices, ``gold_index = -1``,
    the metadata a grader declares it needs -- exactly as load-bearing as before, and
    a shape allowed to rot while a bank sits out would fail for a second, unrelated
    reason on the day the bridge is rebuilt.
    """

    def test_is_supported(self, dataset: str) -> None:
        assert datasets.get_spec(dataset).name == dataset

    def test_spec_modality_matches_what_was_vendored(self, dataset: str) -> None:
        assert datasets.get_spec(dataset).modality == VENDORED[dataset]

    def test_readiness_tracks_the_blocker(self, dataset: str) -> None:
        """``ready_names()`` is what every user-facing message lists.

        It may not name a dataset the resolver is going to refuse, in either
        direction: pointing someone at a blocked bank wastes a run, and omitting a
        runnable one hides it.
        """
        spec = datasets.get_spec(dataset)
        assert (dataset in datasets.ready_names()) is (spec.blocked is None)

    def test_the_committed_manifest_describes_the_bank(self, dataset: str) -> None:
        """Read off disk rather than through the resolver, so a blocked bank is covered."""
        manifest = json.loads(
            (CALIBRATED_DATASETS / dataset / "manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["modality"] == VENDORED[dataset]
        assert manifest["items"] > 0

    def test_resolution_follows_the_blocker_and_not_the_disk(self, dataset: str) -> None:
        """Committed artifacts do not make a blocked bank runnable.

        Every dataset here has a ``params.json`` and an ``items.jsonl`` that parse and
        load, so their presence says nothing about whether each row's parameters belong
        to the question printed beside them. The resolver has to decide on the blocker.
        """
        spec = datasets.get_spec(dataset)
        if spec.blocked is not None:
            with pytest.raises(resolve.DatasetNotAvailable, match="is blocked"):
                resolve.resolve(dataset)
            return

        resolved = resolve.resolve(dataset)
        assert resolved.params_path.is_file()
        assert resolved.items_path.is_file()
        assert resolved.manifest["modality"] == VENDORED[dataset]
        assert resolved.manifest["items"] > 0

    def test_items_match_their_declared_modality(self, dataset: str) -> None:
        """A generative bank must carry no choices, an MCQ bank must carry them.

        Mixing the two would send items to the wrong grader one at a time, which shows
        up as a plausible-looking ability estimate rather than as an error. Whether a
        generative record is *complete* is asked of the grader its answer_type names,
        because two of the three generative banks answer that with a gold string and
        ifeval answers it with its constraint payload.
        """
        path = CALIBRATED_DATASETS / dataset / "items.jsonl"
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert records

        for record in records:
            if VENDORED[dataset] == "generative":
                assert record["choices"] == []
                assert record["gold_index"] == -1
                assert record["metadata"]["modality"] == "generative"
                item = BenchmarkItem(
                    item_id=record["id"],
                    question=record["question"],
                    choices=(),
                    gold_index=-1,
                    metadata=record["metadata"],
                )
                assert generative.is_gradable(item), record["id"]
            else:
                assert len(record["choices"]) >= 2
                assert 0 <= record["gold_index"] < len(record["choices"])
                assert "metadata" not in record

    def test_the_manifest_records_the_grader_the_bank_was_built_for(self, dataset: str) -> None:
        """Provenance has to name the grader, since theta's scale depends on which it was.

        Banks vendored before the answer type was declared per dataset carry no such
        key and are skipped rather than re-vendored, on the same terms as the modality
        key: the grader they ran under is the numeric default either way.
        """
        manifest = json.loads(
            (CALIBRATED_DATASETS / dataset / "manifest.json").read_text(encoding="utf-8")
        )
        if "answer_type" not in manifest:
            pytest.skip(f"{dataset} predates the manifest recording answer_type")
        expected = (
            datasets.get_spec(dataset).answer_type if VENDORED[dataset] == "generative" else None
        )
        assert manifest["answer_type"] == expected
