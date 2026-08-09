"""Wire-level tests for the failures the stub-based suite could not see.

Every test here corresponds to a real defect found by review: a judge that crashed the run
on a malformed 200, a tutor whose transient outage was misreported as a bad scenario, an
IRT update that silently reverted to the prior, and a crash path that discarded results.
They exercise the real clients over real HTTP rather than stubs.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from diagnostics.frq_cat.base import CATState, CriterionResponse, JudgeVerdict, Scenario
from diagnostics.frq_cat.common.judge import JudgeSpec
from diagnostics.frq_cat.styles.uni_frq.judge_client import ResilientJudge, redact
from diagnostics.frq_cat.styles.uni_frq.respgen_client import (
    ResilientRespGen,
    RespGenConfig,
    ScenarioUnservable,
    TutorUnavailable,
)
from diagnostics.frq_cat.styles.uni_frq.result_sink import ResultSink
from diagnostics.frq_cat.styles.uni_frq.session import run_session
from diagnostics.frq_cat.styles.uni_frq.style import UniFrqStyle

from .test_pipeline import _tiny_bank

_OK_BODY = {"choices": [{"message": {"content": "PASS\nlooks right"}}]}


@pytest.fixture(autouse=True)
def _no_backoff_sleep(monkeypatch):
    """Retry backoff is real code; waiting for it is not worth the test time."""
    monkeypatch.setattr("time.sleep", lambda *_: None)


@pytest.fixture
def http_stub():
    """Serve a scripted sequence of responses and record the request headers."""
    servers: list[HTTPServer] = []

    class _Calls:
        def __init__(self) -> None:
            self.n = 0
            self.headers: list[dict[str, str]] = []

    def start(responses):
        state = _Calls()

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))
                state.headers.append(dict(self.headers))
                status, body = responses[min(state.n, len(responses) - 1)]
                state.n += 1
                raw = body if isinstance(body, str) else json.dumps(body)
                data = raw.encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, format: str, *args: object) -> None:
                pass

        srv = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers.append(srv)
        return f"http://127.0.0.1:{srv.server_port}/v1", state

    yield start
    for srv in servers:
        srv.shutdown()


def _spec() -> JudgeSpec:
    return JudgeSpec(name="t", model_id="test-model")


def _criterion():
    bank, _ = _tiny_bank()
    return bank.get_scenario("s1"), bank.criteria[0]


# ---------------------------------------------------------------- judge over HTTP


def test_judge_actually_sends_the_bearer_token(http_stub, monkeypatch) -> None:
    monkeypatch.setenv("TEST_KEY", "s3cr3t")
    url, state = http_stub([(200, _OK_BODY)])
    scenario, criterion = _criterion()
    verdict = ResilientJudge(_spec(), endpoint=url, api_key_env="TEST_KEY").evaluate(
        scenario, criterion, "answer"
    )
    assert verdict.passed is True
    assert state.headers[0]["Authorization"] == "Bearer s3cr3t"


def test_judge_refuses_to_start_when_the_key_env_is_empty(monkeypatch) -> None:
    monkeypatch.setenv("TEST_KEY", "   ")
    with pytest.raises(RuntimeError, match="TEST_KEY"):
        ResilientJudge(_spec(), endpoint="https://api.example/v1", api_key_env="TEST_KEY")


def test_judge_does_not_retry_an_auth_failure(http_stub) -> None:
    url, state = http_stub([(401, {"error": "bad key"})])
    scenario, criterion = _criterion()
    verdict = ResilientJudge(_spec(), endpoint=url, max_attempts=4).evaluate(
        scenario, criterion, "answer"
    )
    assert verdict.unscorable_reason and "401" in verdict.unscorable_reason
    assert state.n == 1, "401 will not fix itself; retrying just burns quota"


def test_judge_retries_a_rate_limit_then_succeeds(http_stub) -> None:
    url, state = http_stub([(429, {"error": "slow down"}), (200, _OK_BODY)])
    scenario, criterion = _criterion()
    verdict = ResilientJudge(_spec(), endpoint=url, max_attempts=4).evaluate(
        scenario, criterion, "answer"
    )
    assert verdict.passed is True and not verdict.unscorable_reason
    assert state.n == 2


def test_judge_abstains_when_retries_are_exhausted(http_stub) -> None:
    url, _ = http_stub([(503, {"error": "down"})])
    scenario, criterion = _criterion()
    verdict = ResilientJudge(_spec(), endpoint=url, max_attempts=2).evaluate(
        scenario, criterion, "answer"
    )
    assert verdict.unscorable_reason and "503" in verdict.unscorable_reason


@pytest.mark.parametrize(
    "body",
    [
        {"choices": [{"message": None}]},  # message present but null
        {"choices": [{"message": {"content": None}}]},
        {"choices": []},
        {"choices": "PASS"},
        {"choices": [{"message": {"content": [{"type": "text"}]}}]},  # parts, no text
        "not json at all",
    ],
)
def test_judge_abstains_instead_of_crashing_on_a_malformed_200(http_stub, body) -> None:
    # Any of these used to raise out of evaluate() and kill the whole session.
    url, _ = http_stub([(200, body)])
    scenario, criterion = _criterion()
    verdict = ResilientJudge(_spec(), endpoint=url).evaluate(scenario, criterion, "answer")
    assert verdict.unscorable_reason
    assert verdict.passed is False


def test_judge_reads_content_parts(http_stub) -> None:
    url, _ = http_stub([(200, {"choices": [{"message": {"content": [{"text": "PASS\nok"}]}}]})])
    scenario, criterion = _criterion()
    assert ResilientJudge(_spec(), endpoint=url).evaluate(scenario, criterion, "a").passed is True


def test_error_bodies_are_redacted(http_stub) -> None:
    url, _ = http_stub([(400, {"error": "bad Bearer sk-abcdef123456 supplied"})])
    scenario, criterion = _criterion()
    verdict = ResilientJudge(_spec(), endpoint=url).evaluate(scenario, criterion, "answer")
    assert "sk-abcdef123456" not in (verdict.unscorable_reason or "")
    assert redact("Bearer sk-abcdef123456") == "[redacted]"


# ---------------------------------------------------------------- tutor over HTTP


def test_transient_outage_after_a_400_is_not_reported_as_a_bad_scenario(http_stub) -> None:
    # 400 on the first rung, then the endpoint goes sick. The floor budget was never
    # tried, so nothing was proven about the prompt: this must stop the run, not skip.
    url, _ = http_stub([(400, {"m": "too long"}), (503, {"m": "down"})])
    gen = ResilientRespGen(RespGenConfig(endpoint=url, served_model="t", max_attempts=2))
    with pytest.raises(TutorUnavailable):
        gen.generate_one(Scenario(scenario_id="s1", prompt="x"))


def test_persistent_400_down_to_the_floor_is_permanent(http_stub) -> None:
    url, _ = http_stub([(400, {"m": "maximum context length is 4096"})])
    gen = ResilientRespGen(RespGenConfig(endpoint=url, served_model="t", max_attempts=2))
    with pytest.raises(ScenarioUnservable, match="maximum context length"):
        gen.generate_one(Scenario(scenario_id="s1", prompt="x"))


def test_unreadable_200_is_transient_not_an_empty_answer(http_stub) -> None:
    url, _ = http_stub([(200, {"choices": []})])
    gen = ResilientRespGen(RespGenConfig(endpoint=url, served_model="t", max_attempts=1))
    with pytest.raises(TutorUnavailable):
        gen.generate_one(Scenario(scenario_id="s1", prompt="x"))


def test_budget_ladder_is_deduped_decreasing_and_never_exceeds_max_tokens() -> None:
    for max_tokens in (16, 32, 100, 128, 512, 1024, 4096):
        for attempts in (1, 2, 3, 4, 8):
            ladder = ResilientRespGen(
                RespGenConfig(
                    endpoint="http://x/v1",
                    served_model="t",
                    max_tokens=max_tokens,
                    max_attempts=attempts,
                )
            ).budget_ladder()
            assert ladder == sorted(set(ladder), reverse=True), (max_tokens, attempts, ladder)
            assert max(ladder) <= max_tokens
            assert len(ladder) <= attempts
            if attempts > 1:
                assert ladder[-1] == min(32, max_tokens)


# ---------------------------------------------------------------- session resilience


class _Boom:
    """A judge that raises rather than returning a verdict."""

    def evaluate(self, scenario, criterion, response_text):
        raise ValueError("judge exploded")


class _Resp:
    def __init__(self, *, blank=(), unavailable=()):
        self.blank, self.unavailable = set(blank), set(unavailable)

    def generate_one(self, scenario):
        if scenario.scenario_id in self.unavailable:
            raise TutorUnavailable(f"{scenario.scenario_id}: endpoint down")
        return "" if scenario.scenario_id in self.blank else f"answer for {scenario.scenario_id}"


def _sink(tmp_path):
    return ResultSink(str(tmp_path / "out"), run_id="r", checkpoint_slug="c", local_dir=tmp_path)


def test_a_judge_that_raises_becomes_an_abstention_not_a_dead_run(tmp_path) -> None:
    bank, irt = _tiny_bank()
    sink = _sink(tmp_path)
    summary = run_session(
        UniFrqStyle(),
        bank=bank,
        irt_bank=irt,
        respgen=_Resp(),
        judge=_Boom(),
        sink=sink,
        se_threshold=0.0,
        max_items=10,
    )
    assert summary["criteria_abstained"] == 4  # every criterion abstained, nothing crashed
    assert summary["criteria_administered"] == 0
    assert "judge raised ValueError" in (sink.local_dir / "abstentions.jsonl").read_text()
    assert summary["stopped_because"] == "no-scorable-items-left"


def test_blank_tutor_answer_drops_the_scenario_instead_of_grading_nothing(tmp_path) -> None:
    bank, irt = _tiny_bank()
    sink = _sink(tmp_path)

    class _Pass:
        def evaluate(self, scenario, criterion, response_text):
            return JudgeVerdict(criterion_id=criterion.criterion_id, passed=True)

    summary = run_session(
        UniFrqStyle(),
        bank=bank,
        irt_bank=irt,
        respgen=_Resp(blank={"s1"}),
        judge=_Pass(),
        sink=sink,
        se_threshold=0.0,
        max_items=10,
    )
    assert summary["scenarios_skipped"] == 1
    assert summary["criteria_administered"] == 2  # s1's two criteria are dropped with it
    assert "empty response" in (sink.local_dir / "skipped.jsonl").read_text()


def test_tutor_outage_stops_early_but_keeps_what_was_scored(tmp_path) -> None:
    bank, irt = _tiny_bank()
    sink = _sink(tmp_path)

    class _Pass:
        def evaluate(self, scenario, criterion, response_text):
            return JudgeVerdict(criterion_id=criterion.criterion_id, passed=True)

    # The first selected scenario answers; every other scenario is unavailable.
    summary = run_session(
        UniFrqStyle(),
        bank=bank,
        irt_bank=irt,
        respgen=_Resp(unavailable={"s2", "s3"}),
        judge=_Pass(),
        sink=sink,
        se_threshold=0.0,
        max_items=10,
    )
    assert summary["stopped_because"].startswith("tutor-unavailable")
    assert summary["criteria_administered"] >= 1
    assert (sink.local_dir / "trajectory.jsonl").exists()


def test_skips_are_flushed_even_when_nothing_is_ever_scored(tmp_path, monkeypatch) -> None:
    bank, irt = _tiny_bank()
    sink = _sink(tmp_path)
    calls: list[int] = []
    monkeypatch.setattr(type(sink), "sync", lambda self: calls.append(1))
    run_session(
        UniFrqStyle(),
        bank=bank,
        irt_bank=irt,
        respgen=_Resp(),
        judge=_Boom(),
        sink=sink,
        se_threshold=0.0,
        max_items=10,
    )
    assert calls, "abstention-only runs must still flush to the destination"


# ---------------------------------------------------------------- IRT numerics


def test_a_saturating_item_no_longer_discards_the_observation() -> None:
    style = UniFrqStyle()
    irt = style.load_irt_params("")
    # te_0173_c01 (a=4.67, b=-11.98) saturates p to exactly 1.0 across the grid, so the
    # old linear-space likelihood produced 1-p == 0 and reset theta to the prior.
    est = style.estimate_ability(
        irt, [CriterionResponse(criterion_id="te_0173_c01", correct=False)]
    )
    theta = est.theta
    assert isinstance(theta, float)
    assert theta < -1.0, "a failure on a very easy item must move theta down"
    assert est.metadata["n_responses"] == 1


def test_an_all_pass_run_is_flagged_boundary_limited() -> None:
    style = UniFrqStyle()
    irt = style.load_irt_params("")
    state = CATState(benchmark="t", se_threshold=0.0, max_items=40)
    for _ in range(40):
        nid = style.select_next_item(irt, state)
        if nid is None:
            break
        state.administered.append(CriterionResponse(criterion_id=nid, correct=True))
        state.ability = style.estimate_ability(irt, state.administered)
    assert state.ability is not None
    assert state.ability.metadata.get("boundary_limited") is True
    se = state.ability.standard_error
    assert isinstance(se, float)
    assert se > 0.1, "an off-scale theta must not look precise"


def test_estimate_ability_rejects_a_response_missing_from_the_bank() -> None:
    style = UniFrqStyle()
    _, irt = _tiny_bank()
    with pytest.raises(KeyError):
        style.estimate_ability(irt, [CriterionResponse(criterion_id="not-in-bank", correct=True)])


# ---------------------------------------------------------------- CLI crash path


def test_an_unexpected_crash_still_flushes_and_marks_the_run_incomplete(
    tmp_path, monkeypatch
) -> None:
    from diagnostics.frq_cat.styles.uni_frq import run_uni_frq

    def _explode(*_args, **_kwargs):
        raise RuntimeError("something went wrong mid-session")

    monkeypatch.setattr(run_uni_frq, "run_session", _explode)
    out = tmp_path / "out"
    with pytest.raises(RuntimeError, match="something went wrong"):
        run_uni_frq.main(
            [
                "--checkpoint",
                "ckpt-x",
                "--out",
                str(out),
                "--tutor-endpoint",
                "http://127.0.0.1:1/v1",
                "--judge-endpoint",
                "http://127.0.0.1:1/v1",
                "--judge-config",
                "judge_frozen.yaml",
                "--run-id",
                "crashy",
            ]
        )
    runs = [p for p in out.glob("*/*") if p.is_dir()]
    assert len(runs) == 1
    run_dir = runs[0]
    assert not (run_dir / "_SUCCESS").exists(), "an aborted run must not look complete"
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert "something went wrong" in manifest["error"]
    assert manifest["local_dir"]  # the recoverable copy is recorded
