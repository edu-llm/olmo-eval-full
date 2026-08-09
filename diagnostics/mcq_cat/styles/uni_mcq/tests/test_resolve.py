"""The allowlist and the three-branch ladder.

Every failure path must raise, not warn. These tests assert that, and that each message
says enough for a user to know what to do next.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from ....common import generative
from .. import datasets, resolve
from .conftest import unblock, write_bank


class TestAllowlist:
    def test_ready_datasets_are_vendorable(self) -> None:
        """Ready means no recorded blocker."""
        for name in datasets.ready_names():
            assert datasets.SUPPORTED[name].blocked is None

    def test_ready_is_a_subset_of_supported(self) -> None:
        assert set(datasets.ready_names()) <= set(datasets.supported_names())

    def test_every_spec_is_internally_consistent(self) -> None:
        for name, spec in datasets.SUPPORTED.items():
            assert spec.name == name
            assert spec.route in {"B", "C"}
            assert spec.fit_family in datasets.FIT_FAMILIES
            assert spec.modality in datasets.MODALITIES
            assert spec.bridge_kind in datasets.BRIDGE_KEY_COLUMNS
            assert spec.expected_bank_rows > 0
            assert spec.notes, f"{name} has no notes explaining its behavior"
            assert spec.params_csv.endswith("irt_item_parameters_combined.csv")

    def test_every_generative_spec_names_a_registered_grader(self) -> None:
        """An answer_type with no grader vendors a bank nothing can score."""
        for name, spec in datasets.SUPPORTED.items():
            if spec.modality == "generative":
                assert spec.answer_type in generative.ANSWER_GRADERS, name

    def test_every_route_b_and_c_dataset_is_present(self) -> None:
        assert set(datasets.supported_names()) == {
            "arc_challenge",
            "hellaswag",
            "winogrande",
            "gsm8k",
            "leaderboard_math",
            "ifeval",
            "gpqa",
            "musr",
            "bbh",
            "pedagogy",
            "piqa",
            "socialiqa",
        }

    def test_only_winogrande_gsm8k_and_ifeval_are_blocked_today(self) -> None:
        """Pinned in both directions, because both directions are deliberate acts.

        A fourth dataset acquiring a blocker changes what this branch will run, and one
        of these three losing its blocker ships a bank nobody re-examined; either should
        have a test to update rather than pass unremarked. This test earned that framing
        on 2026-08-08, when it was the thing that made blocking ``ifeval`` a visible
        decision instead of a silent one.

        The three are withheld for two different kinds of reason. ``winogrande`` and
        ``gsm8k`` are withheld over how well their rebuilt bridge can be *checked*, not
        over anything known to be wrong: winogrande's harvest carries no question text,
        so the join can only be read through a gold letter on a two-choice benchmark,
        and no per-item gsm8k harvest exists on this checkout at all.

        ``ifeval`` is different and worse. Something *is* known to be wrong with what a
        run of it would report: a model that recites its prompt passes 129 of the 511
        items, because an IFEval instruction is verifiable precisely by naming its own
        success token, so stating the constraint puts that token in the prompt. The
        resulting theta is high, tight, stops on precision, and carries no field saying
        anything is amiss. See its ``blocked`` reason and ``RESULT_CAVEATS.md``. It comes
        back when the echo-baseline guard exists, not when someone re-checks a join.
        """
        withheld = set(datasets.supported_names()) - set(datasets.ready_names())

        assert withheld == {"winogrande", "gsm8k", "ifeval"}

    def test_bbh_is_no_longer_excluded_and_its_stale_reason_is_gone(self) -> None:
        """The entry claimed no bbh task existed on any ref, which stopped being true.

        Worth its own test rather than a deletion, because an EXCLUDED reason is what a
        user is shown instead of the dataset and this one would have kept telling them
        the bank was unusable and unauthorable while both had been settled.
        """
        assert "bbh" not in datasets.EXCLUDED
        assert not any("bbh" in reason for reason in datasets.EXCLUDED.values())

    def test_supported_and_excluded_do_not_overlap(self) -> None:
        assert not set(datasets.SUPPORTED) & set(datasets.EXCLUDED)

    def test_known_exclusion_explains_itself(self) -> None:
        """A user asking for csqa should learn *why*, not just that it is unknown."""
        with pytest.raises(KeyError) as exc:
            datasets.get_spec("csqa")
        message = exc.value.args[0]
        assert "bank_subdir" in message
        assert "arc_challenge" in message

    def test_the_three_locally_fitted_banks_left_excluded_together(self) -> None:
        """Their exclusions named the gap that fitting them closed, so both must go.

        All three were excluded for the same reason and it was a true one: the feasibility
        study fit them with girth in memory and kept only the correlations, so no bank file
        existed anywhere to vendor. Serializing that fit is exactly what removed the
        obstacle, and a reason left behind would keep telling a user the bank does not
        exist while it sits in ``calibrated_datasets/``.
        """
        for name in ("pedagogy", "piqa", "socialiqa"):
            assert name not in datasets.EXCLUDED
        assert not any("never serialized" in reason for reason in datasets.EXCLUDED.values())

    def test_unknown_name_lists_what_is_ready(self) -> None:
        with pytest.raises(KeyError) as exc:
            datasets.get_spec("not_a_dataset")
        assert "arc_challenge" in exc.value.args[0]


class TestLadder:
    def test_branch_one_unknown_dataset_raises(self, monkeypatch, tmp_path: Path) -> None:
        monkeypatch.setattr(resolve, "CALIBRATED_DATASETS", tmp_path)
        with pytest.raises(resolve.DatasetNotAvailable, match="Unknown dataset"):
            resolve.resolve("nonsense")

    def test_branch_two_present_bank_resolves(self, monkeypatch, tmp_path: Path) -> None:
        write_bank(tmp_path, dataset="arc_challenge")
        monkeypatch.setattr(resolve, "CALIBRATED_DATASETS", tmp_path)

        resolved = resolve.resolve("arc_challenge")

        assert resolved.spec.name == "arc_challenge"
        assert resolved.params_path.is_file()
        assert resolved.items_path.is_file()
        assert resolved.fit_family == "3pl"
        assert resolved.provenance()["route"] == "C"

    def block(self, monkeypatch, dataset: str, reason: str) -> None:
        """Give ``dataset`` a blocker for one test.

        The situations that set one -- a bank that cannot be enumerated, a bank whose
        join was afterwards shown wrong or shown unmeasurable -- all recur, and the
        ladder's ordering around it is what stops the last from reporting a confident
        theta. Synthesized here rather than borrowed from whichever dataset happens to
        be out, so the coverage does not disappear the next time one is restored.
        """
        monkeypatch.setitem(
            datasets.SUPPORTED, dataset, replace(datasets.SUPPORTED[dataset], blocked=reason)
        )

    def test_blocked_dataset_reports_its_blocker(self, monkeypatch, tmp_path: Path) -> None:
        """The reason has to reach the user, since a bare "unavailable" invites a retry."""
        self.block(monkeypatch, "gpqa", "its items sit behind a HuggingFace token scope")
        monkeypatch.setattr(resolve, "CALIBRATED_DATASETS", tmp_path)

        with pytest.raises(resolve.DatasetNotAvailable) as exc:
            resolve.resolve("gpqa")
        message = exc.value.args[0]
        assert "HuggingFace token scope" in message
        assert "is blocked" in message

    def test_a_blocker_beats_a_bank_sitting_on_disk(self, monkeypatch, tmp_path: Path) -> None:
        """A blocker has to win against artifacts, not merely against their absence.

        The three ATLAS banks are the reason this ordering exists. While their bridge
        attributed parameters to the wrong questions their artifacts were committed and
        loaded cleanly, so a ladder that asked the filesystem first would have found
        them, resolved, and reported a theta at full overlap with a converging CAT and a
        healthy standard error -- no sign of a problem anywhere. All three are repaired,
        so the combination is recreated here instead of borrowed.
        """
        self.block(monkeypatch, "gpqa", "its join was afterwards shown wrong")
        write_bank(tmp_path, dataset="gpqa")
        monkeypatch.setattr(resolve, "CALIBRATED_DATASETS", tmp_path)

        with pytest.raises(resolve.DatasetNotAvailable) as exc:
            resolve.resolve("gpqa")
        assert "is blocked" in exc.value.args[0]

    @pytest.mark.parametrize("dataset", ["hellaswag", "winogrande", "gsm8k"])
    def test_the_repaired_atlas_banks_resolve(self, monkeypatch, dataset: str) -> None:
        """Against the real committed artifacts, with only the blocker lifted.

        These three were refused until their bridges were rebuilt from Open LLM
        Leaderboard v1 example order. Asserting they resolve pins the repair itself:
        reverting a bridge should not pass silently. Two are withheld again over how
        far that rebuild can be checked rather than over the artifacts, which are all
        retained -- so each lifts its own blocker for the length of this test, and the
        day one is restored it is restored onto a bank already known to resolve.
        """
        unblock(monkeypatch, dataset)

        resolved = resolve.resolve(dataset)
        assert resolved.manifest["items"] > 0
        assert resolved.manifest["bridge_kind"] == datasets.CONTENT_HASH

    @pytest.mark.parametrize("dataset", ["hellaswag", "winogrande", "gsm8k"])
    def test_each_rebuilt_bank_records_where_its_ordering_came_from(
        self, monkeypatch, dataset: str
    ) -> None:
        """A recovered ordering nobody can re-derive is an assertion, not provenance.

        These bridges are committed rather than regenerated, so the manifest is the only
        place a reader learns which leaderboard run the order was read off and at which
        commit. Without that the bank is back to being an ordering someone once believed.
        The two withheld banks are held to it with their blockers lifted, since that
        record is the evidence any decision to restore them would be made against.
        """
        unblock(monkeypatch, dataset)

        provenance = resolve.resolve(dataset).manifest["bridge_provenance"]
        ordering = provenance["ordering"]
        assert ordering["repo"].startswith("open-llm-leaderboard-old/")
        assert len(ordering["revision"]) == 40
        assert ordering["harness_config"].startswith("harness_")
        assert provenance["rows"] > 0

    def test_branch_three_cached_matrix_names_the_missing_work(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        """A cached response matrix should point at calibration, not at vendoring."""
        banks = tmp_path / "banks"
        banks.mkdir()
        matrices = tmp_path / "matrices"
        (matrices / "arc_challenge").mkdir(parents=True)
        (matrices / "arc_challenge" / "response_matrix.csv").write_text("m,1\n", encoding="utf-8")

        monkeypatch.setattr(resolve, "CALIBRATED_DATASETS", banks)
        monkeypatch.setattr(resolve, "RESPONSE_MATRICES", matrices)

        with pytest.raises(resolve.DatasetNotAvailable) as exc:
            resolve.resolve("arc_challenge")
        message = exc.value.args[0]
        assert "response_matrix.csv" in message
        assert "mirt" in message
        assert "not implemented on this branch" in message

    def test_branch_four_nothing_cached_points_at_the_vendoring_script(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(resolve, "CALIBRATED_DATASETS", tmp_path / "banks")
        monkeypatch.setattr(resolve, "RESPONSE_MATRICES", tmp_path / "matrices")

        with pytest.raises(resolve.DatasetNotAvailable) as exc:
            resolve.resolve("arc_challenge")
        assert "vendor_bank" in exc.value.args[0]

    def test_missing_manifest_still_resolves(self, monkeypatch, tmp_path: Path) -> None:
        """Provenance is desirable, not load-bearing."""
        bank = write_bank(tmp_path, dataset="arc_challenge")
        (bank / "manifest.json").unlink()
        monkeypatch.setattr(resolve, "CALIBRATED_DATASETS", tmp_path)

        resolved = resolve.resolve("arc_challenge")
        assert resolved.manifest == {}
        assert resolved.fit_family == "3pl"  # falls back to the spec
        assert resolved.provenance() == {}


class TestManifestAuthority:
    """The bank's manifest outranks any fit family supplied at run time."""

    def test_matching_family_is_accepted(self, monkeypatch, tmp_path: Path) -> None:
        write_bank(tmp_path, dataset="arc_challenge", fit_family="3pl")
        monkeypatch.setattr(resolve, "CALIBRATED_DATASETS", tmp_path)
        resolved = resolve.resolve("arc_challenge")
        assert resolve.check_fit_family(resolved, "3pl") == "3pl"

    def test_no_request_defers_to_the_manifest(self, monkeypatch, tmp_path: Path) -> None:
        write_bank(tmp_path, dataset="arc_challenge", fit_family="3pl")
        monkeypatch.setattr(resolve, "CALIBRATED_DATASETS", tmp_path)
        resolved = resolve.resolve("arc_challenge")
        assert resolve.check_fit_family(resolved, None) == "3pl"

    def test_conflicting_family_raises(self, monkeypatch, tmp_path: Path) -> None:
        """Scoring a 3PL-fit bank as 2PL is an error, not a preference."""
        write_bank(tmp_path, dataset="arc_challenge", fit_family="3pl")
        monkeypatch.setattr(resolve, "CALIBRATED_DATASETS", tmp_path)
        resolved = resolve.resolve("arc_challenge")

        with pytest.raises(resolve.DatasetNotAvailable) as exc:
            resolve.check_fit_family(resolved, "2pl")
        message = exc.value.args[0]
        assert "calibrated as '3pl'" in message
        assert "not a runtime choice" in message

    def test_manifest_overrides_the_spec(self, monkeypatch, tmp_path: Path) -> None:
        """If a 2PL bank is vendored for a dataset the allowlist calls 3PL, the bank wins."""
        write_bank(tmp_path, dataset="arc_challenge", fit_family="2pl")
        monkeypatch.setattr(resolve, "CALIBRATED_DATASETS", tmp_path)
        resolved = resolve.resolve("arc_challenge")
        assert resolved.spec.fit_family == "3pl"
        assert resolved.fit_family == "2pl"
