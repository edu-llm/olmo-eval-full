"""Pin the guard that stops a fit family being stamped onto parameters it does not fit.

``--fit-family`` names how the item parameters were *estimated*. It used to change
nothing but the string written into ``manifest.json``: ``bank_dir`` came from the spec,
so ``--fit-family 2pl`` against a dataset whose spec points at a 3PL calibration read
the 3PL parameters and labelled them 2PL, with a warning that a terminal scrolls past.

Nothing downstream can recover from that. The manifest is the record of provenance, so
``resolve.check_fit_family`` -- which refuses at run time to score a bank as a family
its manifest disagrees with, on the grounds that zeroing a ``g`` estimated jointly with
``a`` and ``b`` biases weak models upward by around 1.2 logits -- reads the false name
and is satisfied. The one place able to catch the mistake is the one place that made it.

Two things therefore have to hold, and these tests pin both. A family override is
refused unless ``--bank-dir`` is redirected somewhere that could hold that fit, so the
error can say what the caller actually needs to do. And the redirect is then checked
against the parameters rather than against the directory's name: 2PL is the ``c = 0``
case of 3PL, so a bank whose guessing column was estimated is a 3PL bank whatever the
directory is called, and one whose column is identically zero is a 2PL bank whatever
the flag says.
"""

from __future__ import annotations

import pytest

from ..datasets import SUPPORTED, DatasetSpec
from ..scripts import vendor_bank
from ..scripts.vendor_bank import (
    Bridge,
    DropCounts,
    build_parser,
    check_parameter_family,
    load_bank,
    resolve_fit_family,
    sibling_fit_dir,
)


def bridge(ids: dict[int, str]) -> Bridge:
    """A single-task bridge, which names no subtask because it spans one task."""
    return Bridge(ids=ids, subtasks={}, rows=len(ids))


#: GPQA's recorded 3PL calibration and the real 2PL refit that sits beside it. These are
#: the paths the failure message has to name, so they are spelled out rather than derived.
GPQA_3PL = "AdaptiveTesting/Experiments/openlm_gpqa_atlas_3pl/calibration"
GPQA_2PL = "AdaptiveTesting/Experiments/openlm_gpqa_atlas_3pl/calibration_2pl"


def spec(fit_family: str = "3pl", bank_dir: str = GPQA_3PL) -> DatasetSpec:
    """A single-task spec carrying only the fields the family guard reads."""
    return DatasetSpec(
        name="fake",
        task="fake",
        route="B",
        bank_dir=bank_dir,
        bridge_path=f"{bank_dir}/atlas_idx_to_question_id.csv",
        bridge_kind="atlas",
        fit_family=fit_family,
        expected_bank_rows=3,
        positional_ids=False,
        notes="fixture",
    )


def rows(*guessing: float) -> list[dict[str, str]]:
    """Upstream parameter rows, in the ``(X, a1, d, g, u)`` shape mirt writes."""
    return [
        {"X": f"X{i}", "a1": "1.0", "d": "0.0", "g": repr(g), "u": "1"}
        for i, g in enumerate(guessing, start=1)
    ]


class TestOverrideNeedsARedirect:
    def test_no_request_keeps_the_recorded_family_and_directory(self) -> None:
        resolved, family = resolve_fit_family(spec(), None, None)

        assert family == "3pl"
        assert resolved.bank_dir == GPQA_3PL

    def test_requesting_the_recorded_family_is_a_no_op(self) -> None:
        """Naming the family the spec already records asks for nothing unusual."""
        resolved, family = resolve_fit_family(spec(), "3pl", None)

        assert family == "3pl"
        assert resolved.bank_dir == GPQA_3PL

    def test_a_bare_family_override_is_an_error_not_a_warning(self) -> None:
        """The regression this guard exists for: 3PL parameters labelled 2PL."""
        with pytest.raises(SystemExit) as exc:
            resolve_fit_family(spec(), "2pl", None)

        message = str(exc.value)
        assert "--bank-dir was not redirected" in message
        assert "Nothing was written." in message

    def test_the_error_names_the_directory_that_would_be_mislabelled(self) -> None:
        with pytest.raises(SystemExit) as exc:
            resolve_fit_family(spec(), "2pl", None)

        assert GPQA_3PL in str(exc.value)

    def test_the_error_names_the_real_2pl_directory_to_use_instead(self) -> None:
        """A message saying only 'this is wrong' leaves the caller nowhere to go."""
        with pytest.raises(SystemExit) as exc:
            resolve_fit_family(spec(), "2pl", None)

        assert GPQA_2PL in str(exc.value)

    def test_redirecting_at_the_recorded_directory_is_still_an_error(self) -> None:
        """The recorded directory holds the recorded family, so this redirects nothing.

        Accepting it would turn the requirement into a formality a caller satisfies by
        pasting the path the error message just quoted back at it.
        """
        with pytest.raises(SystemExit, match="--bank-dir was not redirected"):
            resolve_fit_family(spec(), "2pl", GPQA_3PL)

    def test_a_genuine_redirect_is_accepted_and_moves_the_bank_dir(self) -> None:
        resolved, family = resolve_fit_family(spec(), "2pl", GPQA_2PL)

        assert family == "2pl"
        assert resolved.bank_dir == GPQA_2PL
        assert resolved.params_csv == f"{GPQA_2PL}/irt_item_parameters_combined.csv"

    def test_a_redirect_without_a_family_override_keeps_the_recorded_family(self) -> None:
        """Re-vendoring the same fit from a moved directory is not a family change."""
        resolved, family = resolve_fit_family(spec(), None, "elsewhere/calibration")

        assert family == "3pl"
        assert resolved.bank_dir == "elsewhere/calibration"

    def test_the_redirect_does_not_disturb_the_rest_of_the_spec(self) -> None:
        """Only ``bank_dir`` moves; the bridge, join and modality are unrelated to it."""
        original = spec()
        resolved, _ = resolve_fit_family(original, "2pl", GPQA_2PL)

        assert resolved.bridge_path == original.bridge_path
        assert resolved.bridge_kind == original.bridge_kind
        assert resolved.fit_family == original.fit_family
        assert resolved.modality == original.modality
        assert resolved.name == original.name

    @pytest.mark.parametrize("name", sorted(SUPPORTED))
    def test_every_shipped_dataset_vendors_unchanged_by_default(self, name: str) -> None:
        """The guard must be invisible to the ordinary invocation of the script."""
        resolved, family = resolve_fit_family(SUPPORTED[name], None, None)

        assert family == SUPPORTED[name].fit_family
        assert resolved is SUPPORTED[name]


class TestSiblingDirectory:
    def test_the_gpqa_layout_is_recognised(self) -> None:
        assert sibling_fit_dir(GPQA_3PL, "2pl") == GPQA_2PL

    def test_a_directory_not_named_calibration_suggests_nothing(self) -> None:
        """Better to say nothing than to invent a path that follows no convention."""
        assert sibling_fit_dir("AdaptiveTesting/Inputs/ATLAS/arc", "2pl") is None

    def test_a_bare_directory_name_suggests_nothing(self) -> None:
        assert sibling_fit_dir("calibration", "2pl") is None


class TestParametersDecideTheFamily:
    def test_estimated_guessing_labelled_2pl_aborts(self) -> None:
        """The loophole a redirect alone would open: any directory, any label."""
        with pytest.raises(SystemExit) as exc:
            check_parameter_family(spec(), "2pl", rows(0.21, 0.0, 0.34))

        message = str(exc.value)
        assert "2 of its 3 rows carry a non-zero guessing parameter" in message
        assert "Nothing was written." in message

    def test_a_single_estimated_row_is_enough_to_refuse(self) -> None:
        """A 2PL fit zeroes ``g`` for every item, so one exception disproves the claim."""
        with pytest.raises(SystemExit, match="1 of its 3 rows"):
            check_parameter_family(spec(), "2pl", rows(0.0, 0.0, 0.19))

    def test_an_all_zero_bank_labelled_3pl_aborts(self) -> None:
        """The mirror image, and just as wrong: a 2PL refit recorded as the 3PL fit."""
        with pytest.raises(SystemExit) as exc:
            check_parameter_family(spec(), "3pl", rows(0.0, 0.0, 0.0))

        message = str(exc.value)
        assert "every one of its 3 rows has g = 0" in message
        assert "--fit-family 2pl" in message

    def test_a_real_3pl_bank_stamped_3pl_passes(self) -> None:
        check_parameter_family(spec(), "3pl", rows(0.21, 0.03, 0.34))

    def test_a_real_2pl_bank_stamped_2pl_passes(self) -> None:
        check_parameter_family(spec(fit_family="2pl", bank_dir=GPQA_2PL), "2pl", rows(0.0, 0.0))

    def test_an_absent_guessing_column_reads_as_2pl(self) -> None:
        """``load_bank`` already treats a missing ``g`` as zero, and so must this."""
        without_g = [{"X": "X1", "a1": "1.0", "d": "0.0"}]

        check_parameter_family(spec(fit_family="2pl"), "2pl", without_g)
        with pytest.raises(SystemExit, match="has g = 0"):
            check_parameter_family(spec(), "3pl", without_g)

    def test_an_empty_guessing_cell_reads_as_2pl(self) -> None:
        blank = [{"X": "X1", "a1": "1.0", "d": "0.0", "g": ""}]

        check_parameter_family(spec(fit_family="2pl"), "2pl", blank)


class TestLoadBankEnforcesIt:
    """The check has to sit on the path that reads the parameters, not beside it."""

    def test_a_redirect_to_a_3pl_directory_labelled_2pl_still_aborts(self, monkeypatch) -> None:
        """Redirecting is a claim about a directory; this is what tests the claim.

        A caller who reads the override error and points ``--bank-dir`` at any other
        directory has satisfied the letter of it. Only the parameters settle whether
        that directory holds a 2PL fit.
        """
        csv_text = "X,a1,d,g,u\nX1,1.0,0.0,0.21,1\nX2,1.0,0.0,0.18,1\n"
        monkeypatch.setattr(vendor_bank, "git_show", lambda ref, path: csv_text)

        resolved, family = resolve_fit_family(spec(), "2pl", "somewhere/calibration_2pl")

        with pytest.raises(SystemExit, match="carry a non-zero guessing parameter"):
            load_bank(resolved, "origin/Research", bridge({}), DropCounts(), fit_family=family)

    def test_a_genuine_2pl_directory_loads(self, monkeypatch) -> None:
        csv_text = "X,a1,d,g,u\nX1,1.5,0.6,0,1\nX2,0.8,-0.4,0,1\n"
        monkeypatch.setattr(vendor_bank, "git_show", lambda ref, path: csv_text)

        resolved, family = resolve_fit_family(spec(), "2pl", GPQA_2PL)
        items, upstream = load_bank(
            resolved, "origin/Research", bridge({1: "a", 2: "b"}), DropCounts(), fit_family=family
        )

        assert upstream == 2
        assert [item.item_id for item in items] == ["a", "b"]
        assert all(item.guessing == 0.0 for item in items)

    def test_the_default_path_is_unaffected(self, monkeypatch) -> None:
        """An ordinary 3PL vendoring must load exactly as it did before the guard."""
        csv_text = "X,a1,d,g,u\nX1,1.5,0.6,0.25,1\n"
        monkeypatch.setattr(vendor_bank, "git_show", lambda ref, path: csv_text)

        resolved, family = resolve_fit_family(spec(), None, None)
        items, upstream = load_bank(
            resolved, "origin/Research", bridge({1: "a"}), DropCounts(), fit_family=family
        )

        assert upstream == 1
        assert items[0].guessing == 0.25


class TestCommandLine:
    def test_bank_dir_is_exposed_as_a_flag(self) -> None:
        """The override error tells the caller to pass this, so it has to exist."""
        args = build_parser().parse_args(
            ["--dataset", "gpqa", "--fit-family", "2pl", "--bank-dir", GPQA_2PL]
        )

        assert args.fit_family == "2pl"
        assert args.bank_dir == GPQA_2PL

    def test_bank_dir_defaults_to_the_spec(self) -> None:
        args = build_parser().parse_args(["--dataset", "gpqa"])

        assert args.bank_dir is None
        assert args.fit_family is None

    def test_the_shipped_gpqa_spec_points_the_caller_at_calibration_2pl(self) -> None:
        """Read off the real allowlist entry, so a moved bank_dir is caught here."""
        with pytest.raises(SystemExit) as exc:
            resolve_fit_family(SUPPORTED["gpqa"], "2pl", None)

        assert "calibration_2pl" in str(exc.value)
