"""Tests for the ATLAS adaptive-testing suite."""

from __future__ import annotations

from olmo_eval.evals.suites.registry import get_suite, suite_exists


def test_atlas_suite_resolves() -> None:
    assert suite_exists("atlas")
    suite = get_suite("atlas")
    assert suite.expanded_tasks == (
        "atlas_arc_challenge",
        "atlas_hellaswag",
        "atlas_winogrande",
        "atlas_csqa",
        "atlas_piqa",
        "atlas_gsm8k",
    )


def test_atlas_suite_includes_gsm8k_excludes_truthfulqa() -> None:
    tasks = set(get_suite("atlas").expanded_tasks)
    assert "atlas_gsm8k" in tasks  # generative benchmark is wired end to end
    assert "atlas_truthfulqa" not in tasks  # no base task yet
