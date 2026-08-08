"""Unit tests for the config-driven ATLAS benchmark registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from olmo_eval.adaptive.bank import ATLAS_INPUTS_DIR, ENV_BANK_DIR
from olmo_eval.adaptive.benchmarks import (
    SCORING_GENERATIVE,
    SCORING_MCQ_LOGLIK,
    SCORING_VARIES,
    bank_dir_for,
    cat_benchmarks,
    get_benchmark,
    list_benchmarks,
    mcq_benchmarks,
)


def test_registry_covers_expected_benchmarks() -> None:
    names = {b.name for b in list_benchmarks()}
    assert names == {
        "arc_challenge",
        "hellaswag",
        "winogrande",
        "csqa",
        "piqa",
        "gsm8k",
        "ifeval",
        "math",
        "truthfulqa",
    }


def test_mcq_benchmarks_excludes_generative_and_taskless() -> None:
    mcq = {b.name for b in mcq_benchmarks()}
    # gsm8k is generative, truthfulqa varies (and has no olmo-eval base task).
    assert mcq == {"arc_challenge", "hellaswag", "winogrande", "csqa", "piqa"}
    assert all(b.scoring == SCORING_MCQ_LOGLIK for b in mcq_benchmarks())


def test_cat_benchmarks_include_gsm8k_exclude_truthfulqa() -> None:
    """cat_benchmarks() drives registration: MCQ set + generative benchmarks, no truthfulqa."""
    wired = {b.name for b in cat_benchmarks()}
    assert wired == {
        "arc_challenge",
        "hellaswag",
        "winogrande",
        "csqa",
        "piqa",
        "gsm8k",
        "ifeval",
        "math",
    }
    assert "truthfulqa" not in wired


def test_registered_flag() -> None:
    assert get_benchmark("gsm8k").registered  # generative + has base task
    assert get_benchmark("arc_challenge").registered  # mcq_loglik + has base task
    assert not get_benchmark("truthfulqa").registered  # varies / no base task


def test_scoring_flags_and_cat_support() -> None:
    assert get_benchmark("gsm8k").scoring == SCORING_GENERATIVE
    assert get_benchmark("truthfulqa").scoring == SCORING_VARIES
    assert get_benchmark("truthfulqa").base_task is None
    # cat_supported is the MCQ-loglik-only flag: gsm8k is generative, so False.
    assert not get_benchmark("gsm8k").cat_supported
    assert not get_benchmark("truthfulqa").cat_supported
    assert get_benchmark("arc_challenge").cat_supported


def test_positional_id_flags() -> None:
    assert get_benchmark("winogrande").positional_id
    assert get_benchmark("piqa").positional_id
    assert get_benchmark("gsm8k").positional_id  # metadata["id"] is a positional index
    assert not get_benchmark("arc_challenge").positional_id
    assert not get_benchmark("hellaswag").positional_id
    assert not get_benchmark("csqa").positional_id


def test_task_and_eval_names() -> None:
    arc = get_benchmark("arc_challenge")
    assert arc.offline_task_name == "atlas_arc_challenge"
    assert arc.online_eval_name == "atlas_arc"  # historical short name preserved
    hs = get_benchmark("hellaswag")
    assert hs.offline_task_name == "atlas_hellaswag"
    assert hs.online_eval_name == "atlas_hellaswag"


def test_unknown_benchmark_raises() -> None:
    with pytest.raises(KeyError):
        get_benchmark("does_not_exist")


def test_bank_dir_defaults_to_subdir(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_BANK_DIR, raising=False)
    assert bank_dir_for(get_benchmark("hellaswag")) == ATLAS_INPUTS_DIR / "hellaswag"
    assert bank_dir_for(get_benchmark("arc_challenge")) == ATLAS_INPUTS_DIR / "arc"


def test_explicit_bank_dir_wins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(ENV_BANK_DIR, "/some/env/path")
    assert bank_dir_for(get_benchmark("hellaswag"), tmp_path) == tmp_path
    assert bank_dir_for(get_benchmark("arc_challenge"), tmp_path) == tmp_path


def test_env_override_is_arc_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """The legacy single-path env override must not hijack non-arc banks."""
    monkeypatch.setenv(ENV_BANK_DIR, "/env/arc/bank")
    assert bank_dir_for(get_benchmark("arc_challenge")) == Path("/env/arc/bank")
    # A non-arc benchmark ignores the env and resolves to its own subdir.
    assert bank_dir_for(get_benchmark("winogrande")) == ATLAS_INPUTS_DIR / "winogrande"
