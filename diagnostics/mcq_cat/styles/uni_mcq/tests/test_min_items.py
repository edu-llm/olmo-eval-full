"""``min_items`` per dataset: that BBH gets its floor, and that nothing else moved.

The floor used to be one number for every bank, which is a claim about discriminations
rather than about the harness. Eight items reach SE <= 0.3 on a bank whose median ``a``
is around 1.4; on BBH's, median 3.99, the posterior collapses at once and the session
ends before it has touched a third of 24 unrelated subtasks.

Two halves are worth testing and the second is the larger one. BBH has to get 24 --
covered against the committed bank in :mod:`.test_bbh_bank` -- and the eight banks that
existed before this mechanism have to be measurably untouched by it: same floor, same
resolution, and no trace of the setting in what their manifests hold a run to. A
per-dataset override that leaked into the recorded scoring convention would fail every
one of them at startup over a number that cannot change a single item's outcome, and the
tests below are what says it does not.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from ....common import grading
from .. import convention, datasets
from ..style import (
    CAT_BLOCK,
    DEFAULT_MIN_ITEMS,
    PER_DATASET_CAT_KEYS,
    UniMcqStyle,
    dataset_min_items,
)
from .conftest import CALIBRATED_DATASETS, unblock

#: The floor BBH asks for, which is also its subtask count.
BBH_MIN_ITEMS = 24

#: Every dataset that must keep the shared floor: the four ATLAS banks and the four
#: Route B banks that predate the per-dataset mechanism.
UNCHANGED = (
    "arc_challenge",
    "hellaswag",
    "winogrande",
    "gsm8k",
    "leaderboard_math",
    "ifeval",
    "gpqa",
    "musr",
)


def shipped_config() -> dict[str, Any]:
    """The committed ``config.yaml``, read the way a run reads it."""
    return convention.load_config()


def per_dataset_block() -> dict[str, Any]:
    """The ``datasets`` map at the top level of the shipped config."""
    return dict(shipped_config().get(grading.PER_DATASET_KEY) or {})


def manifest_of(dataset: str) -> dict[str, Any]:
    """A committed manifest, or skip if the bank is not vendored."""
    path = CALIBRATED_DATASETS / dataset / "manifest.json"
    if not path.is_file():
        pytest.skip(f"{dataset} has not been vendored")
    return json.loads(path.read_text(encoding="utf-8"))


class TestTheShippedConfiguration:
    def test_the_shared_floor_is_still_eight(self) -> None:
        assert shipped_config()["min_items"] == DEFAULT_MIN_ITEMS == 8

    def test_bbh_is_the_only_dataset_with_an_override(self) -> None:
        assert set(per_dataset_block()) == {"bbh"}

    def test_the_override_is_the_subtask_count(self) -> None:
        assert per_dataset_block()["bbh"] == {"min_items": BBH_MIN_ITEMS}
        assert len(datasets.SUPPORTED["bbh"].subtasks) == BBH_MIN_ITEMS

    def test_the_floor_stays_under_the_pinned_cap(self) -> None:
        """A floor at or above ``max_items`` would make every session a fixed-length
        test and the standard error a number nobody could act on."""
        assert shipped_config()["max_items"] > BBH_MIN_ITEMS

    def test_the_map_is_yaml_the_loader_and_a_reader_agree_about(self) -> None:
        """Read straight off disk as well, so a key nested under the wrong block -- which
        the loader would simply not find -- cannot pass as an override that applied."""
        raw = yaml.safe_load(Path(convention.CONFIG_PATH).read_text(encoding="utf-8"))
        assert raw[grading.PER_DATASET_KEY]["bbh"]["min_items"] == BBH_MIN_ITEMS


class TestResolution:
    @pytest.mark.parametrize("dataset", UNCHANGED)
    def test_every_other_bank_keeps_the_shared_floor(self, dataset: str) -> None:
        assert dataset_min_items(shipped_config(), dataset) == DEFAULT_MIN_ITEMS

    def test_bbh_resolves_to_its_own(self) -> None:
        assert dataset_min_items(shipped_config(), "bbh") == BBH_MIN_ITEMS

    def test_a_dataset_the_map_never_names_falls_through(self) -> None:
        assert dataset_min_items(shipped_config(), "not_a_dataset") == DEFAULT_MIN_ITEMS

    def test_an_unreadable_config_still_produces_the_pinned_default(self) -> None:
        """``load_config`` returns an empty mapping on a parse failure, and a floor of 0
        there would turn one into a one-item test rather than into an error."""
        assert dataset_min_items({}, "bbh") == DEFAULT_MIN_ITEMS

    def test_an_unknown_per_dataset_key_raises(self) -> None:
        """A misspelling would otherwise leave the shared floor in place under a name
        that reads as though it had been applied."""
        config = {"min_items": 8, grading.PER_DATASET_KEY: {"bbh": {"min_itmes": 24}}}

        with pytest.raises(ValueError, match=CAT_BLOCK):
            dataset_min_items(config, "bbh")

    def test_the_run_level_settings_may_not_be_moved_per_dataset(self) -> None:
        """Both already have a CLI flag and are recorded per run; a second, quieter way
        to set them would make two runs of one bank incomparable and say so nowhere."""
        assert PER_DATASET_CAT_KEYS == ("min_items",)
        for key in ("se_threshold", "max_items"):
            with pytest.raises(ValueError, match=key):
                dataset_min_items(
                    {"min_items": 8, grading.PER_DATASET_KEY: {"bbh": {key: 1}}}, "bbh"
                )


class TestTheStyle:
    def test_a_style_with_no_bank_reports_the_shared_floor(self) -> None:
        """The engine reads it before anything is resolved, so it may not be undefined."""
        assert UniMcqStyle().min_items == DEFAULT_MIN_ITEMS

    @pytest.mark.parametrize("dataset", UNCHANGED)
    def test_resolving_another_bank_does_not_move_the_floor(
        self, monkeypatch, dataset: str
    ) -> None:
        """Each lifts its own blocker, since two of these banks are withheld today.

        The floor is a property of ``config.yaml`` and not of whether a bank is
        available, so it has to read the same for a withheld one -- and has to still
        read the same on the day someone deletes that blocker.
        """
        style = UniMcqStyle()
        if not (CALIBRATED_DATASETS / dataset / "params.json").is_file():
            pytest.skip(f"{dataset} has not been vendored")
        unblock(monkeypatch, dataset)
        style.download_benchmark(dataset)

        assert style.min_items == DEFAULT_MIN_ITEMS

    def test_resolving_bbh_does(self) -> None:
        style = UniMcqStyle()
        if not (CALIBRATED_DATASETS / "bbh" / "params.json").is_file():
            pytest.skip("bbh has not been vendored")
        style.download_benchmark("bbh")

        assert style.min_items == BBH_MIN_ITEMS


class TestItIsNotAScoringConvention:
    """The line ``convention.py`` draws, and why this setting falls on the other side.

    A recorded convention is checked against every run and a difference is refused. That
    is right for a prompt style and wrong for a test length: the floor cannot change
    whether an individual item is answered correctly, so a bank vendored under one and
    run under another is comparable in everything but how many items it took -- which the
    report already carries.
    """

    @pytest.mark.parametrize("dataset", (*UNCHANGED, "bbh"))
    def test_no_manifest_records_it(self, dataset: str) -> None:
        recorded = manifest_of(dataset)[convention.CONVENTION_KEY][convention.RUNTIME_KEY]

        assert "min_items" not in recorded

    @pytest.mark.parametrize("dataset", (*UNCHANGED, "bbh"))
    def test_the_startup_check_still_passes_on_every_committed_bank(self, dataset: str) -> None:
        """The claim "unaffected" reduced to the thing that would actually break."""
        spec = datasets.get_spec(dataset)
        expected = convention.configured_convention(spec, shipped_config())
        recorded = manifest_of(dataset)[convention.CONVENTION_KEY][convention.RUNTIME_KEY]

        assert recorded == expected
