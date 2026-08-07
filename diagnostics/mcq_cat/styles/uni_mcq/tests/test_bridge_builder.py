"""The v2 bridge builder's guards, which are what a new benchmark will be resting on.

Four more Route B banks come through :data:`LEADERBOARD_V2` and each is one table entry
written by someone who has not read this module. What is pinned here is the behaviour
that entry depends on and cannot see: that a half-written entry fails at import rather
than after several minutes of downloads, that the pinned run is honoured rather than
quietly replaced by a newer one in the same repo, and that a match which is not a
bijection aborts instead of producing a bridge that is right for most rows.

Nothing here touches the Hub. The Hub-facing halves -- resolving a repo listing, reading a
JSONL, comparing runs -- are exercised by actually building MuSR's bridge, and the
artifact that produced is asserted in :mod:`.test_bridges`.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ..scripts.build_leaderboard_bridge import (
    COMPOSITE_ID,
    LEADERBOARD_V2,
    PER_SUBTASK_TASK,
    WHOLE_TASK,
    LeaderboardV2Recipe,
    build_v2_rows,
    details_path,
    match_documents,
    musr_document,
    question_and_choices,
)

MUSR = LEADERBOARD_V2["musr"]

#: A repo listing shaped like a details repo's: one directory per model, one file per
#: leaderboard task per run, and two runs of the same model on the same day.
FILES = [
    ".gitattributes",
    "m/results_2024-07-18T11-04-02.101450.json",
    "m/samples_leaderboard_musr_murder_mysteries_2024-07-18T11-04-02.101450.jsonl",
    "m/samples_leaderboard_musr_murder_mysteries_2024-09-19T16-21-01.446061.jsonl",
    "m/samples_leaderboard_musr_object_placements_2024-07-18T11-04-02.101450.jsonl",
]


def instance(question: str, choices: tuple[str, ...] = ("a", "b")) -> SimpleNamespace:
    """The subset of an ``olmo_eval`` ``Instance`` the builder reads."""
    return SimpleNamespace(question=question, choices=choices)


class TestRecipeValidation:
    """A new entry is configuration, and configuration that half-works is the hazard."""

    def kwargs(self, **overrides: object) -> dict[str, object]:
        base = {
            "task_prefix": "leaderboard_fake_",
            "index_map": "fake/item_id_map.csv",
            "doc_fields": ("prompt",),
            "from_record": lambda doc: (doc["prompt"],),
            "from_instance": question_and_choices,
            "enumeration": PER_SUBTASK_TASK,
            "normalization": "none",
        }
        return base | overrides

    def test_an_index_map_with_no_subtask_column_needs_a_label(self) -> None:
        """Otherwise every calibrated row is attributed to the empty subtask.

        That then matches no declared task and reads as a stale spec, which is a long way
        from where the mistake was made. IFEval is the entry that will hit this: its bank
        joins through a bridge whose only columns are ``atlas_idx`` and ``question_id``.
        """
        with pytest.raises(ValueError, match="single_subtask"):
            LeaderboardV2Recipe(**self.kwargs(subtask_column=""))

    def test_a_single_task_enumeration_needs_the_label_its_rows_carry(self) -> None:
        with pytest.raises(ValueError, match="single_subtask"):
            LeaderboardV2Recipe(**self.kwargs(enumeration=WHOLE_TASK))

    def test_an_unknown_enumeration_is_refused_at_import(self) -> None:
        """The alternative is discovering a typo after the cross-check has downloaded."""
        with pytest.raises(ValueError, match="unknown enumeration"):
            LeaderboardV2Recipe(**self.kwargs(enumeration="per_subtask"))

    def test_a_single_subtask_entry_is_accepted(self) -> None:
        recipe = LeaderboardV2Recipe(
            **self.kwargs(enumeration=WHOLE_TASK, subtask_column="", single_subtask="ifeval")
        )

        assert recipe.single_subtask == "ifeval"

    def test_the_shipped_entry_declares_a_normalization(self) -> None:
        """It goes into the sidecar verbatim and is the one claim nothing else records.

        A recipe that quietly loosened a comparison and one that reproduced a conversion
        both pass every guard here, so the difference only survives if it is written down.
        """
        assert MUSR.normalization
        assert MUSR.enumeration == PER_SUBTASK_TASK


class TestResolvingTheDetailsFile:
    def test_it_honours_the_pinned_run(self) -> None:
        """A repo can hold several evaluations of one model, months apart.

        Taking the newest file per task would read one benchmark's order from one
        evaluation and another's from a different one, which is a bridge assembled out of
        two harness versions.
        """
        path = details_path(
            "r", FILES, "leaderboard_musr_murder_mysteries", run="2024-07-18T11-04-02.101450"
        )

        assert path.endswith("murder_mysteries_2024-07-18T11-04-02.101450.jsonl")

    def test_it_takes_the_newest_when_no_run_is_pinned(self) -> None:
        """Cross-checked repos have their own run timestamps, so they resolve rather than pin."""
        path = details_path("r", FILES, "leaderboard_musr_murder_mysteries")

        assert path.endswith("murder_mysteries_2024-09-19T16-21-01.446061.jsonl")

    def test_a_task_absent_from_the_run_names_what_the_repo_does_hold(self) -> None:
        """The likely cause is a wrong ``task_prefix``, and the fix is in the listing."""
        with pytest.raises(SystemExit, match="leaderboard_musr_team_allocation"):
            details_path("r", FILES, "leaderboard_musr_team_allocation")

    def test_a_task_present_only_in_another_run_still_aborts(self) -> None:
        with pytest.raises(SystemExit, match="in run 2024-09-19"):
            details_path(
                "r", FILES, "leaderboard_musr_object_placements", run="2024-09-19T16-21-01.446061"
            )


class TestMusrDocumentIdentity:
    """The conversions the task performs, repeated on the details record's raw row."""

    DOC = {
        "narrative": "A long story.",
        "question": "Who did it?",
        "choices": "['Ana', 'Mackenzie']",
        "answer_choice": "Ana",
        "answer_index": 0,
    }

    def test_the_stem_is_the_narrative_and_question_joined_by_a_blank_line(self) -> None:
        """A MuSR question without its story is unanswerable rather than merely harder."""
        assert musr_document(self.DOC)[0] == "A long story.\n\nWho did it?"

    def test_the_choices_are_parsed_out_of_their_python_repr(self) -> None:
        """Upstream stores the column as a repr, so both sides parse rather than read."""
        assert musr_document(self.DOC)[1:] == ("Ana", "Mackenzie")

    def test_it_agrees_with_what_the_task_instance_yields(self) -> None:
        """The two sides of the match, and the whole point of them being separate.

        They are written independently and have to produce the same tuple; if they ever
        stop agreeing, the bijection guard is what turns that into an abort rather than a
        bridge built from the fraction that happened to line up.
        """
        rendered = instance("A long story.\n\nWho did it?", ("Ana", "Mackenzie"))

        assert musr_document(self.DOC) == question_and_choices(rendered)


class TestMatchingIsABijection:
    def test_it_returns_the_position_each_doc_id_names(self) -> None:
        instances = [instance("q0"), instance("q1"), instance("q2")]
        documents = {5: ("q2", "a", "b"), 6: ("q0", "a", "b"), 7: ("q1", "a", "b")}

        assert match_documents("fake", "s", MUSR, documents, instances) == {5: 2, 6: 0, 7: 1}

    def test_a_document_matching_nothing_aborts(self) -> None:
        """A bridge over the rows that did match is a bridge over a different item set."""
        documents = {0: ("q0", "a", "b"), 1: ("renamed", "a", "b")}

        with pytest.raises(SystemExit, match="match no instance"):
            match_documents("fake", "s", MUSR, documents, [instance("q0"), instance("q1")])

    def test_an_instance_no_document_claims_aborts(self) -> None:
        """Not every permutation is caught by "everything found something".

        Two documents finding two of three instances leaves one unclaimed, which means the
        split the task enumerates is not the split the leaderboard evaluated.
        """
        documents = {0: ("q0", "a", "b"), 1: ("q1", "a", "b")}
        instances = [instance("q0"), instance("q1"), instance("q2")]

        with pytest.raises(SystemExit, match="not a bijection"):
            match_documents("fake", "s", MUSR, documents, instances)

    def test_a_repeated_instance_identity_aborts_before_matching(self) -> None:
        """A document matching it could name either, so the mapping is not recoverable."""
        instances = [instance("same"), instance("same")]

        with pytest.raises(SystemExit, match="not unique"):
            match_documents(
                "fake", "s", MUSR, documents={0: ("same", "a", "b")}, instances=instances
            )


class TestBuildingTheRows:
    ORDER = {"mm": {0: 0, 1: 1}}
    INSTANCES = {"mm": [instance("q0"), instance("q1")]}

    def test_each_row_names_its_subtask_and_its_split_position(self) -> None:
        rows = build_v2_rows("fake", [(1, "mm", 1), (2, "mm", 0)], self.ORDER, self.INSTANCES)

        assert [(row["atlas_idx"], row["subtask"], row["split_index"]) for row in rows] == [
            (1, "mm", 1),
            (2, "mm", 0),
        ]
        assert rows[0]["question"] == "q1"

    def test_a_calibrated_column_the_run_never_evaluated_aborts(self) -> None:
        """The index map and the details run would then describe different evaluations.

        There is no partial version of that worth shipping: the rows that did resolve
        would be a bridge over an item set nobody chose.
        """
        with pytest.raises(SystemExit, match="never evaluated"):
            build_v2_rows("fake", [(1, "mm", 9)], self.ORDER, self.INSTANCES)

    def test_two_items_hashing_alike_aborts_rather_than_being_absorbed(self) -> None:
        """Vendoring would drop both as ambiguous, which is a defect to repair here."""
        order = {"mm": {0: 0, 1: 1}}
        instances = {"mm": [instance("same"), instance("same")]}

        with pytest.raises(SystemExit, match="name more than one item"):
            build_v2_rows("fake", [(1, "mm", 0), (2, "mm", 1)], order, instances)


def test_the_composite_enumeration_constant_is_reachable_from_the_table() -> None:
    """Named so a new entry can be written against the constant rather than a string."""
    assert {PER_SUBTASK_TASK, WHOLE_TASK, COMPOSITE_ID} == {
        "per_subtask_task",
        "whole_task",
        "composite_id",
    }
