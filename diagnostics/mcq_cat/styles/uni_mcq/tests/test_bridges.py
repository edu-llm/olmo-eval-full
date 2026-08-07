"""The bridges rebuilt from what the Open LLM Leaderboard recorded evaluating.

A bridge decides which question each calibrated parameter row describes, and getting it
wrong is the one failure this pipeline cannot see: every row joins, overlap reads 100%,
the CAT converges and the reported ability is noise. Three of these were wrong for
exactly that reason and were rebuilt, so what is asserted here is less "the files parse"
than "the mistake has not come back".

Two generations are covered and they are asserted differently on purpose. The v1 three
recovered a *shuffled* order, so what has to be shown about them is that they are not the
identity map they replaced. The v2 rebuilds split on that question rather than answering
it uniformly -- musr and ifeval recovered orders that turned out to *be* the identity,
leaderboard_math recovered one that agrees with it on 8 of 1,324 positions -- so the
shared claims are about provenance and each bridge's own finding is pinned separately.

That split is the reason these numbers are asserted rather than merely logged. Whether a
positional join happened to be right is the whole result of re-keying a bank, and it is
the one thing a rebuilt bridge cannot be checked for by reading it.

Everything here reads committed artifacts and needs no network, so it runs in both the
``PYTHONPATH=src`` and bare configurations.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from .. import datasets
from ..datasets import CONTENT_HASH, content_item_id
from ..scripts import vendor_bank

BRIDGES = Path(__file__).resolve().parents[1] / "bridges"
CALIBRATED_DATASETS = Path(__file__).resolve().parents[5] / "calibrated_datasets"

#: Rows each v1 rebuild must span: the whole evaluation split, because ATLAS calibrated
#: column indices up to its size.
V1_EXPECTED_ROWS = {"hellaswag": 10042, "winogrande": 1267, "gsm8k": 1319}

#: Rows each v2 rebuild must span, which is the calibrated bank rather than the split.
#: A Route B fit drops items that fail to converge, so MuSR's 754 columns sit over 756
#: evaluated documents and the two absentees are not the bridge's to carry.
V2_EXPECTED_ROWS = {"ifeval": 535, "leaderboard_math": 1206, "musr": 754}

#: Documents the pinned leaderboard run evaluated, against how many of them sit at the
#: position a fresh enumeration assumes -- which is what the superseded composite key
#: assumed for all of them.
#:
#: The pair is the finding, per bank, and the reason it is pinned rather than logged.
#: Equal means the old positional join happened to be right and the bank came out
#: unchanged; far below means it was wrong and the bank changed. leaderboard_math is the
#: one that was wrong, and it was wrong almost everywhere: its task numbers each subject's
#: Level-5 subsequence of MATH-lighteval from zero, while lm-eval's doc_id counts the
#: pre-filtered MATH-Hard split, and the two orderings are unrelated.
V2_DOCUMENTS = {"ifeval": 541, "leaderboard_math": 1324, "musr": 756}
V2_POSITIONS_AGREEING = {"ifeval": 541, "leaderboard_math": 8, "musr": 756}

#: MuSR's bridge rows per subtask, and the enumeration each was numbered against. The
#: pair differs only for ``object_placements``, where calibration dropped two items.
V2_SUBTASK_ROWS = {"murder_mysteries": 250, "object_placements": 254, "team_allocation": 250}
V2_SUBTASK_SPANS = {"murder_mysteries": 250, "object_placements": 256, "team_allocation": 250}

#: leaderboard_math's calibrated rows per MATH subject, and the Level-5 split each was
#: numbered against. Every subject is sparse within its own span, because calibration
#: dropped the items that failed to converge.
MATH_SUBTASK_ROWS = {
    "algebra_hard": 303,
    "counting_and_prob_hard": 120,
    "geometry_hard": 115,
    "intermediate_algebra_hard": 223,
    "num_theory_hard": 151,
    "prealgebra_hard": 188,
    "precalculus_hard": 106,
}

#: The Level-5 split each MATH subject's positions were numbered against.
MATH_SUBTASK_SPANS = {
    "algebra_hard": 307,
    "counting_and_prob_hard": 123,
    "geometry_hard": 132,
    "intermediate_algebra_hard": 280,
    "num_theory_hard": 154,
    "prealgebra_hard": 193,
    "precalculus_hard": 135,
}

EXPECTED_ROWS = V1_EXPECTED_ROWS | V2_EXPECTED_ROWS

REBUILT = sorted(EXPECTED_ROWS)
V1_REBUILT = sorted(V1_EXPECTED_ROWS)
V2_REBUILT = sorted(V2_EXPECTED_ROWS)


def read_bridge(dataset: str) -> list[dict[str, str]]:
    with (BRIDGES / f"{dataset}.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.mark.parametrize("dataset", REBUILT)
class TestCommittedBridges:
    def test_it_has_the_row_count_its_bank_addresses(self, dataset: str) -> None:
        """A short bridge silently truncates the bank rather than failing.

        ``check_alignment`` compares the bank's max index against the bridge's row
        count, so a bridge missing its tail would abort there -- but only if the missing
        rows are the tail. Pinning the exact count catches the case where a rebuild
        dropped rows anywhere and happened to keep the last one.
        """
        assert len(read_bridge(dataset)) == EXPECTED_ROWS[dataset]

    def test_atlas_idx_is_a_dense_one_based_run(self, dataset: str) -> None:
        """The bank addresses columns by number, so a gap means an unaddressable row."""
        indices = sorted(int(row["atlas_idx"]) for row in read_bridge(dataset))
        assert indices == list(range(1, EXPECTED_ROWS[dataset] + 1))

    def test_every_item_id_is_claimed_once(self, dataset: str) -> None:
        """Two rows sharing an id are both dropped as ambiguous at vendoring time.

        Keying on a content hash is what made this true, and it is the whole reason
        HellaSwag recovered the 204 items its native-``ind`` bridge lost.
        """
        counts = Counter(row["item_id"] for row in read_bridge(dataset))
        assert [item_id for item_id, n in counts.items() if n > 1] == []

    def test_the_spec_points_at_it(self, dataset: str) -> None:
        """The bridge is only load-bearing if the spec actually reads this file."""
        spec = datasets.SUPPORTED[dataset]
        assert spec.bridge_in_repo
        assert spec.bridge_kind == CONTENT_HASH
        assert spec.bridge_path.endswith(f"bridges/{dataset}.csv")

    def test_the_vendored_bank_was_built_from_it(self, dataset: str) -> None:
        """Every id in the bank has to appear in the bridge, or the two have drifted.

        The bridge and the bank are committed separately, so nothing but this stops one
        being replaced without the other -- after which vendoring would still succeed on
        the next run and the artifacts on disk would describe a join nobody performed.
        """
        params_path = CALIBRATED_DATASETS / dataset / "params.json"
        if not params_path.is_file():
            pytest.skip(f"{dataset} has not been vendored")

        bank = json.loads(params_path.read_text(encoding="utf-8"))
        bridge_ids = {row["item_id"] for row in read_bridge(dataset)}
        assert {record["item_id"] for record in bank} <= bridge_ids


@pytest.mark.parametrize("dataset", V1_REBUILT)
class TestV1Bridges:
    """What the v1 rebuilds have to show that the v2 one does not: a recovered shuffle."""

    def test_it_is_not_the_positional_bridge_it_replaced(self, dataset: str) -> None:
        """The regression guard, and the reason this file exists.

        The superseded generator asserted that ATLAS column *k* describes split row
        *k - 1*. Reading the order off the v1 leaderboard instead shows the two agree on
        3 of 10,042 HellaSwag positions, 1 of 1,267 WinoGrande and 2 of 1,319 GSM8K --
        coincidences, not an offset. Anyone regenerating these from
        ``Inputs/ATLAS/scripts/build_atlas_idx_bridge.py`` would produce a file that
        passes every other assertion here and fails this one.
        """
        rows = read_bridge(dataset)
        coincident = sum(1 for row in rows if int(row["split_index"]) == int(row["atlas_idx"]) - 1)
        assert coincident < 0.01 * len(rows)

    def test_split_index_covers_the_split_exactly_once(self, dataset: str) -> None:
        """A bijection onto the split, which is what makes it a recovered order.

        Any permutation satisfies "every column found something" -- that is how the
        refuted bridge passed its own guard. What separates a recovery from an invention
        is that no split row is claimed twice and none is left out.
        """
        positions = sorted(int(row["split_index"]) for row in read_bridge(dataset))
        assert positions == list(range(V1_EXPECTED_ROWS[dataset]))

    def test_its_provenance_names_a_reproducible_ordering(self, dataset: str) -> None:
        """A committed bridge with no record of where its order came from is a claim.

        These are not regenerated on demand -- doing so needs the Hub, a 2023 details
        parquet and the HuggingFace split reachable at once -- so the sidecar is the only
        thing standing between a reader and having to take the ordering on faith.
        """
        provenance = json.loads((BRIDGES / f"{dataset}.json").read_text(encoding="utf-8"))
        ordering = provenance["ordering"]

        assert ordering["repo"].startswith("open-llm-leaderboard-old/")
        assert len(ordering["revision"]) == 40
        assert ordering["harness_config"].startswith("harness_")
        assert ordering["cross_checked"]
        assert provenance["rows"] == V1_EXPECTED_ROWS[dataset]
        assert provenance["bridge_kind"] == CONTENT_HASH
        assert provenance["item_id"]["task"] == datasets.SUPPORTED[dataset].task


def v2_provenance(dataset: str) -> dict:
    return json.loads((BRIDGES / f"{dataset}.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("dataset", V2_REBUILT)
class TestV2Bridges:
    """What every Route B re-key has to record, whatever its own finding turned out to be.

    Each replaced a ``<subtask>|<doc_id>`` composite, where ``doc_id`` is lm-eval's
    enumeration position at harvest time -- a key that has to be re-derived by counting
    rather than read off the item. The rebuild reads the document each ``doc_id`` named
    from leaderboard v2's own detail files, which is why the sidecar has to say which run
    was read and what agreed with it.
    """

    def test_it_names_the_v2_run_it_was_read_from(self, dataset: str) -> None:
        provenance = v2_provenance(dataset)
        ordering = provenance["ordering"]

        assert ordering["repo"].startswith("open-llm-leaderboard/")
        assert ordering["repo"].endswith("-details")
        assert len(ordering["revision"]) == 40
        assert ordering["run"]
        assert ordering["paths"]
        assert provenance["rows"] == V2_EXPECTED_ROWS[dataset]
        assert provenance["bridge_kind"] == CONTENT_HASH

    def test_the_ordering_is_attested_by_runs_from_other_periods(self, dataset: str) -> None:
        """One model's file is one evaluation, and an ordering has to outlive one.

        ``doc_id`` is only a usable key if every leaderboard run enumerated the split the
        same way. Agreement between runs months apart is the cheapest measurement of that
        and the only reason reading an arbitrary model's file is legitimate.
        """
        provenance = v2_provenance(dataset)
        cross_checked = provenance["ordering"]["cross_checked"]
        evaluated = provenance["matching"]["documents_evaluated"]

        assert len(cross_checked) >= 2
        assert {entry["documents_agreeing"] for entry in cross_checked} == {evaluated}
        assert len({entry["run"][:7] for entry in cross_checked}) >= 2

    def test_the_sidecar_says_where_doc_id_came_from(self, dataset: str) -> None:
        """Explicit field or row order are different evidence and read the same later.

        v2 sample files are written in completion order rather than document order, so a
        reader who assumed row order would have produced a scrambled bridge that passed
        every count here. Recording which one was used is what stops that being
        re-litigated from the artifact alone.
        """
        doc_id = v2_provenance(dataset)["ordering"]["doc_id"]

        assert "doc_id field" in doc_id
        assert "not from its position" in doc_id

    def test_the_match_needed_no_normalization_and_was_total(self, dataset: str) -> None:
        """A loosened comparison and a reproduced conversion look identical when passing.

        These details records hold the raw dataset row, so the two sides can be matched
        exactly; the sidecar says so, and a later rebuild that had to relax something
        would have to say that instead. gpqa is the reason this is asserted per bridge
        rather than assumed: its recipe does normalize, which is why the sidecar carries
        prose rather than a flag.
        """
        matching = v2_provenance(dataset)["matching"]

        assert matching["documents_matched"] == matching["documents_evaluated"]
        assert matching["normalization"].startswith("none")

    def test_the_sidecar_records_whether_the_old_positional_key_was_right(
        self, dataset: str
    ) -> None:
        """The finding, and the only reason to re-key a bank whose count does not move.

        Pinned per bank because the three answers differ and each is a claim about a
        committed artifact: musr and ifeval agree with the enumeration on every document,
        so their item sets are unchanged, while leaderboard_math agrees on 8 of 1,324 and
        its bank now names different questions. A rebuild that quietly moved one of these
        numbers would be reporting a different join under the same filename.
        """
        matching = v2_provenance(dataset)["matching"]

        agreeing = matching["positions_agreeing_with_enumeration_order"]

        assert matching["documents_evaluated"] == V2_DOCUMENTS[dataset]
        assert agreeing == V2_POSITIONS_AGREEING[dataset]


class TestLeaderboardMathBridge:
    """The one Route B re-key that changed which questions the bank holds.

    Its predecessor joined through ``<subject>|<doc_id>`` and the olmo-eval task emits
    exactly that composite, so the join looked self-describing and was not: the task
    numbers each subject's Level-5 subsequence of ``DigitalLearningGmbH/MATH-lighteval``
    from zero, while lm-eval's ``doc_id`` counts the pre-filtered ``lighteval/MATH-Hard``
    split. Same 1,324 problems, unrelated orderings, and every guard passed throughout.
    """

    DATASET = "leaderboard_math"

    def test_it_is_not_the_positional_bridge_it_replaced(self) -> None:
        """The regression guard: regenerating the composite key would fail here.

        ``split_index`` is the ``doc_id`` the calibration was keyed by, and the item id
        beside it is the question that ``doc_id`` actually named. Under the superseded
        join those two were the same thing, so a bridge whose rows still lined up would
        mean the recovery had been undone.
        """
        provenance = v2_provenance(self.DATASET)["matching"]

        assert provenance["positions_agreeing_with_enumeration_order"] < 0.01 * 1324

    def test_each_subject_contributes_the_rows_its_calibration_did(self) -> None:
        counts = Counter(row["subtask"] for row in read_bridge(self.DATASET))

        assert dict(sorted(counts.items())) == MATH_SUBTASK_ROWS

    def test_a_split_position_is_claimed_once_within_its_subject(self) -> None:
        """Positions restart per subject, so global uniqueness would pass trivially.

        Every subject is sparse within its own span here, unlike MuSR where only
        ``object_placements`` is, because this calibration dropped items that failed to
        converge across all seven.
        """
        seen: dict[str, list[int]] = {}
        for row in read_bridge(self.DATASET):
            seen.setdefault(row["subtask"], []).append(int(row["split_index"]))

        for subtask, positions in seen.items():
            assert len(set(positions)) == len(positions), subtask
            assert min(positions) >= 0, subtask
            assert max(positions) < MATH_SUBTASK_SPANS[subtask], subtask

    def test_the_bank_records_the_subject_the_hash_no_longer_carries(self) -> None:
        """A single-task spec over a bank that spans seven subjects.

        The composite id used to name the subject and a content hash names only the item,
        so vendoring copies the bridge's own column onto each parameter record. Without
        it nothing downstream can attribute a MATH item to a subject at all, and this is
        the one bank where that is not covered by the multi-task path.
        """
        params_path = CALIBRATED_DATASETS / self.DATASET / "params.json"
        if not params_path.is_file():
            pytest.skip("leaderboard_math has not been vendored")

        bridge = {row["item_id"]: row["subtask"] for row in read_bridge(self.DATASET)}
        bank = json.loads(params_path.read_text(encoding="utf-8"))

        assert bank
        for record in bank:
            assert record["metadata"]["subtask"] == bridge[record["item_id"]]


class TestIfevalBridge:
    """The single-task Route B bank, whose index map is the odd one out.

    It joins through ``Inputs/ATLAS/ifeval/atlas_idx_to_question_id.csv``, a two-column
    file whose ``question_id`` is a bare integer rather than a composite, so its recipe
    is the only one setting ``subtask_column=""`` and naming the label itself.
    """

    DATASET = "ifeval"

    def test_the_recovery_confirmed_the_key_it_replaced(self) -> None:
        """Reported rather than tuned to, and it is a result rather than a formality.

        All 541 evaluated documents sit at the position a fresh enumeration assumes, so
        the bare integer named the right prompt and the bank is unchanged at 511 items.
        The same recovery on leaderboard_math found the assumption wrong on 1,316 of
        1,324, which is why agreement is worth recording rather than assuming.
        """
        matching = v2_provenance(self.DATASET)["matching"]

        assert matching["positions_agreeing_with_enumeration_order"] == 541
        assert matching["enumeration"] == "whole_task"

    def test_every_row_carries_the_one_subtask_label(self) -> None:
        """The recipe supplies it, because the index map has no column to read it from."""
        assert {row["subtask"] for row in read_bridge(self.DATASET)} == {"ifeval"}

    def test_its_positions_are_the_sparse_run_the_harvest_left(self) -> None:
        """541 prompts were harvested and 535 calibrated, so six ids are simply absent.

        Pinned because a dense 0..534 here would mean the bridge had been renumbered onto
        its own row count, which joins perfectly and names the wrong prompt from the first
        gap onward.
        """
        positions = sorted(int(row["split_index"]) for row in read_bridge(self.DATASET))

        assert len(positions) == 535
        assert max(positions) == 540
        assert set(range(541)) - set(positions) == {115, 440, 453, 455, 523, 528}


class TestMusrBridge:
    """The pilot re-key, and the subtask claims that are its own."""

    DATASET = "musr"

    def test_each_subtask_contributes_the_rows_its_calibration_did(self) -> None:
        counts = Counter(row["subtask"] for row in read_bridge(self.DATASET))

        assert dict(sorted(counts.items())) == V2_SUBTASK_ROWS

    def test_a_split_position_is_claimed_once_within_its_subtask(self) -> None:
        """The bijection that survives a sparse bank, stated per subtask.

        Positions restart per subtask, so a global uniqueness check would pass trivially
        while two rows of one subtask claimed the same document. ``object_placements`` is
        the case that needs the span rather than the count: 254 rows over 0..255, with
        136 and 140 dropped in the fit.
        """
        seen: dict[str, list[int]] = {}
        for row in read_bridge(self.DATASET):
            seen.setdefault(row["subtask"], []).append(int(row["split_index"]))

        for subtask, positions in seen.items():
            assert len(set(positions)) == len(positions), subtask
            assert min(positions) == 0, subtask
            assert max(positions) < V2_SUBTASK_SPANS[subtask], subtask

    def test_the_spec_declares_every_subtask_the_bridge_names(self) -> None:
        """Without this a whole subtask's bank rows vanish into ``not_in_task``."""
        declared = {label for label, _ in datasets.SUPPORTED[self.DATASET].subtasks}

        assert {row["subtask"] for row in read_bridge(self.DATASET)} == declared


class TestArcControl:
    """The known-answer run that licenses the other three.

    ARC-Challenge is the only benchmark ATLAS shipped a bridge for, so it is the only
    place the recovery can be checked against an answer we already have. Rebuilding
    three unverifiable bridges on an unvalidated procedure would have swapped a
    known-wrong join for an unknown one, and for GSM8K this control is the only evidence
    there is or can be.
    """

    @pytest.fixture
    def control(self) -> dict:
        path = BRIDGES / "arc_challenge.control.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def test_the_recovered_order_reproduces_the_shipped_bridge(self, control: dict) -> None:
        result = control["control"]
        assert result["positions_agreeing_on_stem"] == result["positions_compared"]
        assert result["agreement"] > 0.99

    def test_the_disagreements_are_the_shipped_bridge_duplicating_an_id(
        self, control: dict
    ) -> None:
        """Both differences are positions where the shipped file is the one in error.

        ATLAS resolved its ids by matching stem text and ARC holds two stems that recur
        under a second id, so each collided onto the wrong twin. Recorded because the raw
        agreement rate reads as two unexplained misses otherwise, and because it is the
        evidence that the recovery is not merely as good as the shipped bridge.
        """
        result = control["control"]
        assert len(result["disagreements"]) == 2
        assert set(result["ids_duplicated_in_shipped_bridge"]) == {
            entry["shipped_question_id"] for entry in result["disagreements"]
        }
        assert all(entry["stems_identical"] == "True" for entry in result["disagreements"])

    def test_arc_keeps_the_bridge_the_release_shipped(self, control: dict) -> None:
        """The reproduction validates the procedure; it does not replace the reference.

        Every figure the other three banks are judged against was measured through the
        shipped bridge, so swapping it for a reproduction would trade the reference away.
        No ``arc_challenge.csv`` is written for the same reason: a second ARC bridge in
        that directory would be indistinguishable from the three that are live.
        """
        assert not (BRIDGES / "arc_challenge.csv").exists()
        assert not datasets.SUPPORTED["arc_challenge"].bridge_in_repo
        assert datasets.SUPPORTED["arc_challenge"].bridge_kind == "atlas"


class TestContentItemId:
    """The join key both sides of the bridge compute independently."""

    def test_it_is_stable(self) -> None:
        """The committed bridges are keyed by it, so a change to it orphans them all."""
        assert content_item_id("q", ("a", "b")) == content_item_id("q", ("a", "b"))
        assert len(content_item_id("q")) == datasets.CONTENT_HASH_DIGITS

    def test_choices_are_part_of_the_item(self) -> None:
        """22 HellaSwag contexts recur with different endings, so the stem is not enough."""
        assert content_item_id("q", ("a", "b")) != content_item_id("q", ("a", "c"))

    def test_choice_order_is_part_of_the_item(self) -> None:
        """A reordered choice block is a different presentation and scores differently."""
        assert content_item_id("q", ("a", "b")) != content_item_id("q", ("b", "a"))

    def test_fields_cannot_run_together(self) -> None:
        """Terminating each field is what stops a shifted boundary from colliding.

        On a benchmark whose choices are sentence fragments continuing the stem, a plain
        concatenation would let two genuinely different items hash alike.
        """
        assert content_item_id("ab", ("c",)) != content_item_id("a", ("bc",))
        assert content_item_id("q", ("a", "b")) != content_item_id("q", ("ab",))


class TestVendoringReadsThem:
    """The paths in ``vendor_bank`` that a bridge living in this repo introduced."""

    @pytest.fixture
    def spec(self) -> datasets.DatasetSpec:
        return datasets.SUPPORTED["winogrande"]

    def test_an_in_repo_bridge_is_read_off_disk(self, spec) -> None:
        """Reached without a source ref, since these bridges have no upstream at all."""
        text = vendor_bank.read_bridge_text(spec, source_ref="no-such-ref")
        assert text.splitlines()[0].startswith("atlas_idx,item_id")

    def test_a_missing_bridge_names_the_command_that_rebuilds_it(self) -> None:
        """Deleting one should not read as a corrupt spec."""
        spec = replace(datasets.SUPPORTED["winogrande"], bridge_path="bridges/absent.csv")
        with pytest.raises(SystemExit) as exc:
            vendor_bank.read_bridge_text(spec, source_ref="no-such-ref")
        assert "build_leaderboard_bridge" in str(exc.value)

    def test_provenance_is_required_beside_a_rebuilt_bridge(self) -> None:
        """A manifest claiming a recovered ordering has to be able to account for it."""
        spec = replace(datasets.SUPPORTED["winogrande"], bridge_path="bridges/absent.csv")
        with pytest.raises(SystemExit, match="provenance"):
            vendor_bank.bridge_provenance(spec)

    def test_an_upstream_bridge_carries_no_sidecar(self) -> None:
        """Its provenance is the upstream commit the manifest already records."""
        assert vendor_bank.bridge_provenance(datasets.SUPPORTED["arc_challenge"]) is None

    def test_an_unknown_bridge_kind_aborts_rather_than_guessing_a_column(self) -> None:
        spec = replace(datasets.SUPPORTED["winogrande"], bridge_kind="invented")
        with pytest.raises(SystemExit, match="unknown bridge_kind"):
            vendor_bank.load_bridge(spec, "no-such-ref")

    def test_a_content_hash_dataset_keys_instances_by_their_text(self) -> None:
        """The other half of the join, and it has to agree with the bridge builder.

        Two implementations of the key that agreed on the day they were written would be
        free to drift, so both call ``content_item_id``; this pins that the vendoring
        side does, and that it ignores ``metadata["id"]`` -- which for HellaSwag is the
        non-unique native ``ind`` the rebuild exists to stop using.
        """
        instance = SimpleNamespace(question="q", choices=("a", "b"), metadata={"id": "180"})
        key_of = vendor_bank.task_item_key(datasets.SUPPORTED["hellaswag"])
        assert key_of(0, instance) == content_item_id("q", ("a", "b"))

    def test_other_datasets_still_key_by_metadata_id(self) -> None:
        instance = SimpleNamespace(question="q", choices=("a",), metadata={"id": "Mercury_1"})
        key_of = vendor_bank.task_item_key(datasets.SUPPORTED["arc_challenge"])
        assert key_of(0, instance) == "Mercury_1"


class TestOverlapFloorApplies:
    def test_a_content_hash_join_is_guarded(self) -> None:
        """It is not positional, and it still needs the floor, for the opposite reason.

        A positional id survives an edit and breaks on a reorder; a content hash survives
        a reorder and breaks on an edit. Reading ``positional_ids`` directly would leave
        the second case unguarded, which is how a bank half-joined against a re-rendered
        task gets written without complaint.
        """
        for dataset in REBUILT:
            spec = datasets.SUPPORTED[dataset]
            assert not spec.positional_ids
            assert spec.needs_overlap_floor

    def test_a_native_id_join_needs_no_floor(self) -> None:
        """ARC joins on native strings, which cannot be silently rekeyed."""
        assert not datasets.SUPPORTED["arc_challenge"].needs_overlap_floor
