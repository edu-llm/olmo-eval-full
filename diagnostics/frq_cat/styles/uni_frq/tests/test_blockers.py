"""Regression tests for the four defects that made a paid run pointless.

Each was measured on the real 1,186-criterion bank before being fixed, and each is the
kind that produces a well-formed artifact rather than a crash, so nothing else in the
suite noticed. Written against the behaviour, not the implementation: they should keep
failing if the guard is removed by any route.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from diagnostics.frq_cat.base import CATState, JudgeVerdict
from diagnostics.frq_cat.common import bank_loader, irt_params
from diagnostics.frq_cat.styles.uni_frq import run_uni_frq
from diagnostics.frq_cat.styles.uni_frq.respgen_client import TutorUnavailable
from diagnostics.frq_cat.styles.uni_frq.result_sink import SUCCESS_MARKER, ResultSink
from diagnostics.frq_cat.styles.uni_frq.session import run_session
from diagnostics.frq_cat.styles.uni_frq.style import UniFrqStyle

_STYLE_DIR = Path(__file__).resolve().parents[1]


def _real_bank():
    """The shipped bank, because the defect only showed at its scale."""
    bank = bank_loader.load_bank(
        "tutoreval", _STYLE_DIR / "bank/scenarios.jsonl", _STYLE_DIR / "bank/params.jsonl"
    )
    return bank, irt_params.load_irt_params(_STYLE_DIR / "bank/params.jsonl")


class _CountingRespGen:
    def __init__(self, fail_after: int | None = None) -> None:
        self.calls = 0
        self.fail_after = fail_after

    def generate_one(self, scenario) -> str:
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise TutorUnavailable("connection refused")
        return "a tutor answer"


class _AlwaysAbstains:
    """What an unreachable judge endpoint actually produces."""

    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, scenario, criterion, response_text) -> JudgeVerdict:
        self.calls += 1
        return JudgeVerdict(
            criterion_id=criterion.criterion_id,
            passed=False,
            unscorable_reason="judge call failed: UnsupportedProtocol",
        )


class _AlwaysPasses:
    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, scenario, criterion, response_text) -> JudgeVerdict:
        self.calls += 1
        return JudgeVerdict(criterion_id=criterion.criterion_id, passed=True)


def _sink(tmp_path: Path, out: str | None = None) -> ResultSink:
    return ResultSink(
        out or str(tmp_path / "out"),
        run_id="r",
        checkpoint_slug="ckpt",
        local_dir=tmp_path / "local",
    )


# ------------------------------------------------------- 1. the paid work must be bounded


def test_an_unusable_judge_cannot_walk_the_whole_bank(tmp_path: Path) -> None:
    """Measured before the fix: 1,186 judge calls and 702 generations under max_items=5.

    `max_items` counts SCORED criteria, so abstentions never advanced it. An unreachable
    judge therefore paid for the entire bank and still reported the prior.
    """
    bank, irt = _real_bank()
    respgen, judge = _CountingRespGen(), _AlwaysAbstains()

    summary = run_session(
        UniFrqStyle(),
        bank=bank,
        irt_bank=irt,
        respgen=respgen,
        judge=judge,
        sink=_sink(tmp_path),
        se_threshold=0.3,
        max_items=5,
    )

    assert judge.calls < 50, f"unbounded: {judge.calls} judge calls for a 5-item test"
    assert respgen.calls < 50, f"unbounded: {respgen.calls} tutor generations"
    assert summary["criteria_administered"] == 0
    assert summary["stopped_because"].startswith("judge-unusable")


def test_the_streak_breaker_does_not_fire_on_a_healthy_run(tmp_path: Path) -> None:
    bank, irt = _real_bank()
    judge = _AlwaysPasses()
    summary = run_session(
        UniFrqStyle(),
        bank=bank,
        irt_bank=irt,
        respgen=_CountingRespGen(),
        judge=judge,
        sink=_sink(tmp_path),
        se_threshold=0.3,
        max_items=40,
    )
    assert summary["stopped_because"] in ("se-threshold", "max-items")
    assert summary["criteria_administered"] > 0


def test_the_attempt_budget_bounds_a_tutor_that_serves_nothing(tmp_path: Path) -> None:
    """Scenario drops do not advance max_items either, so they need the same bound."""
    bank, irt = _real_bank()

    class _Unservable:
        calls = 0

        def generate_one(self, scenario):
            from diagnostics.frq_cat.styles.uni_frq.respgen_client import ScenarioUnservable

            type(self).calls += 1
            raise ScenarioUnservable("prompt exceeds the context window")

    summary = run_session(
        UniFrqStyle(),
        bank=bank,
        irt_bank=irt,
        respgen=_Unservable(),
        judge=_AlwaysPasses(),
        sink=_sink(tmp_path),
        se_threshold=0.3,
        max_items=10,
    )
    assert _Unservable.calls <= 40, f"{_Unservable.calls} generations for a 10-item test"
    assert summary["attempts_made"] <= summary["attempt_budget"]


# ------------------------------------------- 2. a placeholder endpoint must not reach a GPU


@pytest.mark.parametrize(
    "endpoint",
    [
        "REPLACE_WITH_A_REACHABLE_JUDGE_V1_URL",
        "REPLACE_WITH_THE_TFY_GATEWAY_V1_URL",
        "localhost:8001/v1",  # no scheme: every request fails as UnsupportedProtocol
        "http:///v1",  # no host
    ],
)
def test_an_unusable_endpoint_is_refused_rather_than_retried(endpoint: str) -> None:
    """These all passed the old emptiness check and were then retried, criterion by
    criterion, at roughly 7.5 seconds each across the whole bank."""
    assert run_uni_frq._endpoint_fault("judge", endpoint)


@pytest.mark.parametrize(
    "endpoint", ["http://127.0.0.1:8001/v1", "https://gw.example.com/api/v1", "http://host:80/v1"]
)
def test_a_real_endpoint_is_accepted(endpoint: str) -> None:
    assert run_uni_frq._endpoint_fault("judge", endpoint) == ""


def test_the_shipped_spec_placeholder_is_the_one_that_is_caught() -> None:
    """Pin the guard to the literal string the submission spec ships with."""
    spec = (_STYLE_DIR.parents[3] / ".edullm" / "run-frq-cat.yaml").read_text()
    assert "REPLACE_WITH_A_REACHABLE_JUDGE_V1_URL" in spec
    assert run_uni_frq._endpoint_fault("judge", "REPLACE_WITH_A_REACHABLE_JUDGE_V1_URL")


# --------------------------------------------------------- 3. _SUCCESS must not be a lie


def test_a_stale_success_marker_is_cleared_by_a_failed_run(tmp_path: Path) -> None:
    """A retry into a prefix that already holds a marker used to keep it, leaving
    _SUCCESS beside a crash manifest and the previous run's report."""
    sink = _sink(tmp_path)
    sink.write_json("manifest.json", {"run": 1})
    sink.finalize(ok=True)
    marker = sink.local_dir / SUCCESS_MARKER
    assert marker.exists()

    sink.write_json("manifest.json", {"run": 2, "error": "crashed"})
    sink.finalize(ok=False)
    assert not marker.exists(), "an incomplete run must not leave a completion marker"


def test_losing_the_tutor_mid_run_is_not_a_completed_run() -> None:
    """It scored one criterion, so the old `num_items_administered > 0` test passed and
    the run exited 0 with _SUCCESS."""
    assert "tutor-unavailable".startswith(run_uni_frq._ABORTED_STOP_REASONS)
    for reason in ("tutor-unavailable: te_0375: ConnectError", "judge-unusable: 12 consecutive"):
        assert reason.startswith(run_uni_frq._ABORTED_STOP_REASONS)
    for reason in ("se-threshold", "max-items", "exhausted-bank", "no-scorable-items-left"):
        assert not reason.startswith(run_uni_frq._ABORTED_STOP_REASONS)


def test_a_lost_tutor_stops_the_session_and_says_so(tmp_path: Path) -> None:
    bank, irt = _real_bank()
    summary = run_session(
        UniFrqStyle(),
        bank=bank,
        irt_bank=irt,
        respgen=_CountingRespGen(fail_after=1),
        judge=_AlwaysPasses(),
        sink=_sink(tmp_path),
        se_threshold=0.3,
        max_items=40,
    )
    assert summary["stopped_because"].startswith("tutor-unavailable")
    assert summary["stopped_because"].startswith(run_uni_frq._ABORTED_STOP_REASONS)


# ------------------------------------------------- 4. the report must be parseable JSON


def test_a_report_with_no_scored_criteria_is_still_valid_json(tmp_path: Path) -> None:
    """It used to contain a bare `Infinity`: strict parsers reject the file, and jq
    silently reads it as 1.8e308, the largest finite double."""
    style = UniFrqStyle()
    report = style.report(CATState(benchmark="tutoreval"))
    se = report.ability.standard_error  # Ability is float | tuple for the MIRT styles
    assert isinstance(se, float) and not math.isfinite(se)

    sink = _sink(tmp_path)
    sink.write_json("cat_report.json", report.to_dict())
    raw = (sink.local_dir / "cat_report.json").read_text()

    assert "Infinity" not in raw and "NaN" not in raw
    parsed = json.loads(raw, parse_constant=_reject)
    assert parsed["ability"]["standard_error"] is None
    assert parsed["num_items_administered"] == 0


def test_non_finite_values_survive_as_null_in_ndjson_too(tmp_path: Path) -> None:
    sink = _sink(tmp_path)
    sink.append("trajectory.jsonl", {"theta": 1.0, "standard_error": float("inf")})
    line = (sink.local_dir / "trajectory.jsonl").read_text().strip()
    assert json.loads(line, parse_constant=_reject)["standard_error"] is None


def _reject(constant: str) -> object:
    raise ValueError(f"non-JSON constant in output: {constant}")
