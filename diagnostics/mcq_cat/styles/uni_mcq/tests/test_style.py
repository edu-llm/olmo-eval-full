"""The style: loading through the frozen loaders, selection, stopping, and reporting.

The CAT smoke test is the important one. It runs the real frozen engine against a
simulated scorer with a known true theta, so it checks the pieces actually compose:
selection narrows in, the ability estimate converges, and the stopping rule fires.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ....common import cat_loop, generative
from ....registry import available_styles, get_cat
from .. import resolve
from ..style import UNGRADABLE_ALERT_RATE, UniMcqStyle
from .conftest import SimScorer, write_bank


@pytest.fixture
def style(monkeypatch, tmp_path: Path) -> UniMcqStyle:
    """A style instance pointed at a toy bank registered as ``arc_challenge``."""
    write_bank(tmp_path, dataset="arc_challenge")
    monkeypatch.setattr(resolve, "CALIBRATED_DATASETS", tmp_path)
    return UniMcqStyle()


class TestRegistration:
    def test_style_is_discovered(self) -> None:
        assert "uni_mcq" in available_styles()

    def test_registry_returns_the_style(self) -> None:
        assert isinstance(get_cat("uni_mcq"), UniMcqStyle)

    def test_name_is_stamped_by_the_decorator(self) -> None:
        assert UniMcqStyle.name == "uni_mcq"

    def test_constructible_with_no_arguments(self) -> None:
        """The interface requires this; the runner instantiates via the registry."""
        assert UniMcqStyle().min_items > 0


class TestConfig:
    def test_pinned_values_are_loaded(self, style: UniMcqStyle) -> None:
        assert style.se_threshold == 0.3
        assert style.min_items == 8
        assert style.max_items == 40

    def test_config_carries_no_fit_family(self) -> None:
        """Fit family belongs to a bank, not to the style."""
        config_text = (Path(__file__).resolve().parents[1] / "config.yaml").read_text(
            encoding="utf-8"
        )
        active = [
            line
            for line in config_text.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        assert not any("fit_family" in line for line in active)


class TestLoading:
    def test_items_load_through_the_frozen_loader(self, style: UniMcqStyle) -> None:
        bank = style.download_benchmark("arc_challenge")
        assert bank.name == "arc_challenge"
        assert len(bank) == 5
        assert bank.get("toy_0").choices == ("alpha", "beta", "gamma", "delta")

    def test_params_load_and_align_with_items(self, style: UniMcqStyle) -> None:
        style.download_benchmark("arc_challenge")
        irt = style.load_irt_params("arc_challenge")
        assert len(irt) == 5
        assert irt.dimensions == 1
        assert set(style._index) == set(irt.params)

    def test_params_accept_an_explicit_path(self, style: UniMcqStyle, tmp_path: Path) -> None:
        style.download_benchmark("arc_challenge")
        irt = style.load_irt_params(tmp_path / "arc_challenge" / "params.json")
        assert len(irt) == 5

    def test_unresolvable_dataset_raises_before_any_work(self, style: UniMcqStyle) -> None:
        with pytest.raises(resolve.DatasetNotAvailable):
            style.download_benchmark("piqa")

    def test_using_arrays_before_loading_params_explains_itself(self, style: UniMcqStyle) -> None:
        with pytest.raises(RuntimeError, match="load_irt_params"):
            style._arrays()


class TestSelection:
    def test_picks_the_highest_information_item(self, style: UniMcqStyle) -> None:
        import numpy as np

        from ....base import CATState
        from ..irt import fisher_info

        style.download_benchmark("arc_challenge")
        irt = style.load_irt_params("arc_challenge")
        state = CATState(benchmark="arc_challenge")

        chosen = style.select_next_item(irt, state)

        a, b, c = style._arrays()
        assert chosen == style._order[int(np.argmax(fisher_info(0.0, a, b, c)))]

    def test_never_repeats_an_item(self, style: UniMcqStyle, toy_params) -> None:
        style.download_benchmark("arc_challenge")
        irt = style.load_irt_params("arc_challenge")
        scorer = SimScorer(0.5, toy_params)

        report = cat_loop.run_cat(
            style,
            bank=style.download_benchmark("arc_challenge"),
            irt_bank=irt,
            model=scorer,
            se_threshold=0.3,
            max_items=40,
        )
        administered = [r.item_id for r in report.responses]
        assert len(administered) == len(set(administered))

    def test_returns_none_when_the_bank_is_exhausted(self, style: UniMcqStyle) -> None:
        from ....base import CATState, ItemResponse

        style.download_benchmark("arc_challenge")
        irt = style.load_irt_params("arc_challenge")
        state = CATState(benchmark="arc_challenge")
        state.administered = [
            ItemResponse(item_id=item_id, chosen_index=0, correct=True) for item_id in style._order
        ]
        assert style.select_next_item(irt, state) is None


class TestStopping:
    def test_min_items_floor_binds_before_precision(self, style: UniMcqStyle) -> None:
        """Starting at SE 1.0, precision can never pass first -- min_items always does."""
        from ....base import AbilityEstimate, CATState, ItemResponse

        state = CATState(benchmark="arc_challenge", se_threshold=0.3)
        state.administered = [ItemResponse(item_id="toy_0", chosen_index=0, correct=True)]
        state.ability = AbilityEstimate(theta=0.0, standard_error=0.01)

        assert state.step < style.min_items
        assert style.stopping_rule(state) is False

    def test_stops_once_both_conditions_hold(self, style: UniMcqStyle) -> None:
        from ....base import AbilityEstimate, CATState, ItemResponse

        state = CATState(benchmark="arc_challenge", se_threshold=0.3)
        state.administered = [
            ItemResponse(item_id=f"toy_{i % 5}", chosen_index=0, correct=True) for i in range(8)
        ]
        state.ability = AbilityEstimate(theta=0.4, standard_error=0.25)
        assert style.stopping_rule(state) is True

    def test_does_not_stop_while_se_is_too_large(self, style: UniMcqStyle) -> None:
        from ....base import AbilityEstimate, CATState, ItemResponse

        state = CATState(benchmark="arc_challenge", se_threshold=0.3)
        state.administered = [
            ItemResponse(item_id=f"toy_{i % 5}", chosen_index=0, correct=True) for i in range(10)
        ]
        state.ability = AbilityEstimate(theta=0.4, standard_error=0.55)
        assert style.stopping_rule(state) is False

    def test_runner_threshold_overrides_the_config(self, style: UniMcqStyle) -> None:
        from ....base import AbilityEstimate, CATState, ItemResponse

        state = CATState(benchmark="arc_challenge", se_threshold=0.6)
        state.administered = [
            ItemResponse(item_id=f"toy_{i % 5}", chosen_index=0, correct=True) for i in range(8)
        ]
        state.ability = AbilityEstimate(theta=0.4, standard_error=0.55)
        assert style.stopping_rule(state) is True


class TestCatSmoke:
    """A full session through the real frozen engine against a simulated scorer."""

    def test_five_item_bank_runs_to_exhaustion(self, style: UniMcqStyle, toy_params) -> None:
        bank = style.download_benchmark("arc_challenge")
        irt = style.load_irt_params("arc_challenge")
        scorer = SimScorer(0.8, toy_params)

        report = cat_loop.run_cat(
            style, bank=bank, irt_bank=irt, model=scorer, se_threshold=0.3, max_items=40
        )

        # min_items is 8 but the toy bank holds 5, so the bank bounds the session.
        assert report.num_items_administered == 5
        assert report.metadata["stop_reason"] == "bank_exhausted"
        assert len(scorer.calls) == 5

    def test_max_items_caps_the_session(self, style: UniMcqStyle, toy_params) -> None:
        bank = style.download_benchmark("arc_challenge")
        irt = style.load_irt_params("arc_challenge")

        report = cat_loop.run_cat(
            style,
            bank=bank,
            irt_bank=irt,
            model=SimScorer(0.0, toy_params),
            se_threshold=0.001,
            max_items=3,
        )
        assert report.num_items_administered == 3
        assert report.metadata["stop_reason"] == "max_items_reached"

    def test_standard_error_decreases_as_items_accumulate(
        self, style: UniMcqStyle, toy_params
    ) -> None:
        """More evidence should mean less uncertainty than the 1.0 prior."""
        from ....base import CATState

        bank = style.download_benchmark("arc_challenge")
        irt = style.load_irt_params("arc_challenge")
        scorer = SimScorer(0.8, toy_params)

        state = CATState(benchmark="arc_challenge", se_threshold=0.3, max_items=5)
        errors = []
        for _ in range(5):
            item_id = style.select_next_item(irt, state)
            state.administered.extend(style.score(scorer, [bank.get(item_id)]))
            state.ability = style.estimate_ability(irt, state.administered)
            errors.append(state.ability.standard_error)

        assert errors[-1] < 1.0
        assert errors[-1] < errors[0]

    def test_ability_recovers_a_known_theta_on_a_larger_bank(self, monkeypatch, tmp_path) -> None:
        """With enough informative items the estimate should land near the truth.

        Five items cannot pin an ability, so this builds a 60-item bank spanning the
        difficulty range and checks recovery within a generous tolerance.
        """
        import json as _json

        import numpy as np

        params = []
        items = []
        for i in range(60):
            b = -3.0 + 6.0 * i / 59
            params.append(
                {
                    "item_id": f"big_{i}",
                    "difficulty": b,
                    "discrimination": 1.8,
                    "guessing": 0.0,
                }
            )
            items.append(
                {
                    "id": f"big_{i}",
                    "question": f"Q{i}",
                    "choices": ["a", "b", "c", "d"],
                    "gold_index": i % 4,
                }
            )
        bank_dir = tmp_path / "arc_challenge"
        bank_dir.mkdir(parents=True)
        (bank_dir / "params.json").write_text(_json.dumps(params), encoding="utf-8")
        (bank_dir / "items.jsonl").write_text(
            "".join(_json.dumps(it) + "\n" for it in items), encoding="utf-8"
        )
        (bank_dir / "manifest.json").write_text(
            _json.dumps({"fit_family": "2pl", "items": 60}), encoding="utf-8"
        )
        monkeypatch.setattr(resolve, "CALIBRATED_DATASETS", tmp_path)

        style = UniMcqStyle()
        bank = style.download_benchmark("arc_challenge")
        irt = style.load_irt_params("arc_challenge")
        lookup = {
            p["item_id"]: (p["discrimination"], p["difficulty"], p["guessing"]) for p in params
        }

        true_theta = 0.75
        report = cat_loop.run_cat(
            style,
            bank=bank,
            irt_bank=irt,
            model=SimScorer(true_theta, lookup, seed=7),
            se_threshold=0.3,
            max_items=40,
        )
        assert report.metadata["theta"] == pytest.approx(true_theta, abs=0.6)
        assert report.metadata["standard_error"] <= 0.3
        assert report.metadata["stop_reason"] == "precision_reached"
        assert np.isfinite(report.metadata["pirt_accuracy"])


class TestReport:
    def test_carries_every_documented_field(self, style: UniMcqStyle, toy_params) -> None:
        bank = style.download_benchmark("arc_challenge")
        irt = style.load_irt_params("arc_challenge")
        report = cat_loop.run_cat(
            style,
            bank=bank,
            irt_bank=irt,
            model=SimScorer(0.3, toy_params),
            se_threshold=0.3,
            max_items=style.max_items,
        )
        meta = report.metadata

        for key in (
            "theta",
            "standard_error",
            "pirt_accuracy",
            "observed_accuracy",
            "pirt_accuracy_denominator",
            "n_items_administered",
            "bank_size",
            "stop_reason",
            "fit_family",
            "cat_settings",
            "bank_provenance",
            "selected_item_ids",
            "scoring_note",
        ):
            assert key in meta, f"report is missing {key}"

        assert meta["cat_settings"] == {
            "se_threshold": 0.3,
            "min_items": 8,
            "max_items": 40,
            "max_items_pinned_by_style": 40,
            "max_items_is_pinned_value": True,
        }

    def test_flags_a_cap_that_is_not_the_pinned_value(self, style: UniMcqStyle, toy_params) -> None:
        """A caller-supplied cap that differs from the pin must be declared.

        The pin now matches the runner and run_checkpoint_diag.sh defaults, so the
        ordinary path agrees. This covers the case where someone overrides it anyway.
        200 is the value that would arrive from Research's library default,
        ``DEFAULT_MAX_ITEMS`` in ``adaptive/cat.py``, which none of its launchers use;
        estimates are not comparable across caps.
        """
        bank = style.download_benchmark("arc_challenge")
        irt = style.load_irt_params("arc_challenge")
        report = cat_loop.run_cat(
            style,
            bank=bank,
            irt_bank=irt,
            model=SimScorer(0.3, toy_params),
            se_threshold=0.3,
            max_items=200,
        )
        settings = report.metadata["cat_settings"]
        assert settings["max_items"] == 200
        assert settings["max_items_pinned_by_style"] == 40
        assert settings["max_items_is_pinned_value"] is False

    def test_states_the_pirt_denominator(self, style: UniMcqStyle, toy_params) -> None:
        """Predicted accuracy is over the calibrated bank, not the full split."""
        bank = style.download_benchmark("arc_challenge")
        irt = style.load_irt_params("arc_challenge")
        report = cat_loop.run_cat(
            style,
            bank=bank,
            irt_bank=irt,
            model=SimScorer(0.3, toy_params),
            se_threshold=0.3,
            max_items=40,
        )
        note = report.metadata["pirt_accuracy_denominator"]
        assert "calibrated" in note
        assert "not the full evaluation split" in note

    def test_accuracies_are_probabilities(self, style: UniMcqStyle, toy_params) -> None:
        bank = style.download_benchmark("arc_challenge")
        irt = style.load_irt_params("arc_challenge")
        report = cat_loop.run_cat(
            style,
            bank=bank,
            irt_bank=irt,
            model=SimScorer(0.3, toy_params),
            se_threshold=0.3,
            max_items=40,
        )
        assert 0.0 <= report.metadata["pirt_accuracy"] <= 1.0
        assert 0.0 <= report.metadata["observed_accuracy"] <= 1.0

    def test_is_json_serializable(self, style: UniMcqStyle, toy_params) -> None:
        """The runner writes this straight to S3."""
        bank = style.download_benchmark("arc_challenge")
        irt = style.load_irt_params("arc_challenge")
        report = cat_loop.run_cat(
            style,
            bank=bank,
            irt_bank=irt,
            model=SimScorer(0.3, toy_params),
            se_threshold=0.3,
            max_items=40,
        )
        payload = json.loads(json.dumps(report.to_dict()))
        assert payload["cat_style"] == "uni_mcq"
        assert payload["benchmark"] == "arc_challenge"
        assert payload["ability"]["theta"] == pytest.approx(report.metadata["theta"])


class TestUngradableAccounting:
    """An item scored 0 because nothing could grade it must not read as a wrong answer.

    The session continues past one, which is what Research's online CAT does, so the
    fabricated zeroes are inside the pattern theta was computed from. What the report
    owes a reader is the size of that contamination, and specifically whether it is
    large enough that the theta describes the harness instead of the checkpoint.
    """

    def state(self, style: UniMcqStyle, *, ungradable: int, total: int):
        """A finished session over the toy bank with ``ungradable`` fabricated zeroes."""
        from ....base import CATState, ItemResponse

        style.download_benchmark("arc_challenge")
        style.load_irt_params("arc_challenge")
        item_ids = (style._order * total)[:total]
        state = CATState(benchmark="arc_challenge", se_threshold=0.3, max_items=total)
        for position, item_id in enumerate(item_ids):
            bad = position < ungradable
            metadata = {generative.UNGRADABLE_KEY: bad}
            if bad:
                metadata[generative.UNGRADABLE_REASON_KEY] = "no gold answer"
            state.administered.append(
                ItemResponse(item_id=item_id, chosen_index=-1, correct=False, metadata=metadata)
            )
        return state

    def block(self, style: UniMcqStyle, *, ungradable: int, total: int) -> dict:
        """The report's ungradable accounting for such a session."""
        return style.report(self.state(style, ungradable=ungradable, total=total)).metadata[
            "ungradable"
        ]

    def test_a_clean_session_says_so_rather_than_staying_silent(
        self, style: UniMcqStyle, toy_params
    ) -> None:
        """An absent key would make an old report indistinguishable from a clean one."""
        block = self.block(style, ungradable=0, total=10)

        assert block["count"] == 0
        assert block["rate"] == 0.0
        assert "alert" not in block

    def test_a_few_are_counted_and_named_without_an_alert(
        self, style: UniMcqStyle, toy_params
    ) -> None:
        block = self.block(style, ungradable=1, total=10)

        assert block["count"] == 1
        assert block["rate"] == pytest.approx(0.1)
        assert len(block["item_ids"]) == 1
        assert block["reasons"] == ["no gold answer"]
        assert "alert" not in block

    def test_a_high_rate_is_declared_in_the_report_itself(
        self, style: UniMcqStyle, toy_params
    ) -> None:
        """An implausibly low theta is exactly what a missing dependency also produces."""
        block = self.block(style, ungradable=6, total=10)

        assert block["count"] == 6
        assert block["rate"] == pytest.approx(0.6)
        assert "harness rather than the checkpoint" in block["alert"]

    def test_the_alert_threshold_is_the_documented_one(
        self, style: UniMcqStyle, toy_params
    ) -> None:
        """At the boundary it fires, so the constant and the comparison cannot drift."""
        at = int(UNGRADABLE_ALERT_RATE * 10)

        assert "alert" in self.block(style, ungradable=at, total=10)
        assert "alert" not in self.block(style, ungradable=at - 1, total=10)
