"""Tests for the ``uni_frq`` FRQ CAT style.

Covers three things without a GPU or network: the graduated bank loads and conforms
to the frozen schema, the frozen-judge selection is intact, and a tiny CAT session
(stub respgen + a simulated coherent judge) terminates, reduces SE, and serializes.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from diagnostics.frq_cat import registry
from diagnostics.frq_cat.base import JudgeVerdict
from diagnostics.frq_cat.common import cat_loop
from diagnostics.frq_cat.common import judge as judge_mod
from diagnostics.frq_cat.styles.uni_frq.style import UniFrqStyle, _prob_correct

_STYLE_DIR = Path(__file__).resolve().parent.parent


def test_style_is_registered() -> None:
    assert "uni_frq" in registry.available_styles()
    style = registry.get_cat("uni_frq")
    assert isinstance(style, UniFrqStyle)
    assert style.name == "uni_frq"


def test_bank_loads_and_conforms() -> None:
    style = UniFrqStyle()
    bank = style.download_bank("tutoreval")
    assert len(bank.scenarios) == 828
    assert len(bank.criteria) == 1186

    irt = style.load_irt_params("")
    assert irt.dimensions == 1
    assert len(irt) == 1186

    # Unidimensional q-matrix and every criterion linked to a present scenario.
    assert bank.criteria[0].q_modeled == {"ability": 1}
    assert all(c.scenario_id in bank.scenarios for c in bank.criteria)

    # Discrimination is loaded as a length-1 vector (skill-keyed -> tuple) at 1D.
    sample = irt.get(bank.criteria[0].criterion_id)
    assert isinstance(sample.discrimination, tuple) and len(sample.discrimination) == 1


def test_bad_irt_params_path_raises_instead_of_silently_using_bundled_bank() -> None:
    style = UniFrqStyle()
    with pytest.raises(FileNotFoundError):
        style.load_irt_params("/nonexistent/dir/params.jsonl")
    # A bare benchmark name (what the runner passes when --irt-params is omitted) still
    # resolves to the bundled bank.
    assert len(style.load_irt_params("tutoreval")) == 1186


def test_frozen_judge_selection_intact() -> None:
    spec = judge_mod.load_spec(_STYLE_DIR / "judge_frozen.yaml")
    assert spec.model_id == "Qwen/Qwen3.5-9B"
    assert spec.revision == "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    assert spec.adapter == "generic-binary"
    assert spec.enable_thinking is False


def test_frontier_judge_template_names_the_model_and_holds_no_secret() -> None:
    spec = judge_mod.load_spec(_STYLE_DIR / "judge_frontier.yaml")
    assert spec.model_id, "the frontier model slot must be filled in"
    assert spec.temperature == 0.0
    # Routing hints survive on the spec (spec_from_config drops unknown top-level keys).
    assert spec.metadata["provider"] and spec.metadata["endpoint"]
    key_env = spec.metadata["api_key_env"]
    assert key_env == key_env.upper(), "api_key_env must be an env var NAME"
    assert "sk-" not in json.dumps(spec.metadata), "no inline API keys in a committed file"


class _StubRespGen:
    """Deterministic stand-in for the served tutor (no network)."""

    def generate(self, scenarios):
        return {s.scenario_id: f"resp:{s.scenario_id}" for s in scenarios}


class _SimJudge:
    """A coherent simulated judge: a fixed-ability responder over the fitted items."""

    def __init__(self, irt, true_theta: float = 0.3) -> None:
        self._irt = irt
        self._true_theta = true_theta

    def evaluate(self, scenario, criterion, response_text):
        p = _prob_correct(self._true_theta, self._irt.get(criterion.criterion_id))
        return JudgeVerdict(criterion_id=criterion.criterion_id, passed=p >= 0.5, rationale="sim")


def test_cat_smoke_terminates_reduces_se_and_serializes() -> None:
    style = UniFrqStyle()
    bank = style.download_bank("tutoreval")
    irt = style.load_irt_params("")

    # Unreachable SE threshold -> the run is bounded only by max_items.
    report = cat_loop.run_cat(
        style,
        bank=bank,
        irt_bank=irt,
        respgen=_StubRespGen(),
        judge=_SimJudge(irt),
        se_threshold=1e-9,
        max_items=12,
    )
    assert report.num_items_administered == 12
    se, theta = report.ability.standard_error, report.ability.theta
    assert isinstance(se, float) and isinstance(theta, float)  # unidimensional -> scalars
    assert math.isfinite(se) and 0.0 < se < 1.0  # narrower than the unit-normal prior
    assert -4.0 <= theta <= 4.0
    json.dumps(report.to_dict())  # must be JSON-serializable

    # More items -> smaller posterior SE than a single-item session.
    one = cat_loop.run_cat(
        style,
        bank=bank,
        irt_bank=irt,
        respgen=_StubRespGen(),
        judge=_SimJudge(irt),
        se_threshold=1e-9,
        max_items=1,
    )
    assert one.num_items_administered == 1
    one_se = one.ability.standard_error
    assert isinstance(one_se, float)
    assert se < one_se


def test_stopping_rule_respects_min_items() -> None:
    style = UniFrqStyle()
    bank = style.download_bank("tutoreval")
    irt = style.load_irt_params("")
    # A generous SE threshold that would trigger immediately is still gated by min_items.
    report = cat_loop.run_cat(
        style,
        bank=bank,
        irt_bank=irt,
        respgen=_StubRespGen(),
        judge=_SimJudge(irt),
        se_threshold=5.0,
        max_items=40,
    )
    assert report.num_items_administered >= style._min_items
