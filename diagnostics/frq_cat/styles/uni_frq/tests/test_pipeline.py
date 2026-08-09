"""Tests for the uni_frq hardened pipeline (judge parsing, session recovery, result sink).

These cover the failure modes that made the shared runner unusable unattended: a verdict
parser that mis-scored negations, a session that aborted on the first unservable scenario,
and a sink that only produced output if the whole run succeeded.
"""

from __future__ import annotations

import json

import pytest

from diagnostics.frq_cat.base import (
    Criterion,
    FrqBank,
    IRTBank,
    IRTItemParams,
    JudgeVerdict,
    Scenario,
)
from diagnostics.frq_cat.styles.uni_frq.judge_client import parse_verdict
from diagnostics.frq_cat.styles.uni_frq.respgen_client import (
    ResilientRespGen,
    RespGenConfig,
    ScenarioUnservable,
    TutorUnavailable,
)
from diagnostics.frq_cat.styles.uni_frq.result_sink import ResultSink, slugify
from diagnostics.frq_cat.styles.uni_frq.session import run_session
from diagnostics.frq_cat.styles.uni_frq.style import UniFrqStyle


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("PASS", True),
        ("PASS\nthe response defines the term", True),
        ("**PASS** - good", True),
        ("FAIL\nmissing the key step", False),
        ("fail", False),
        # The shared parser scores each of these as a PASS because it searches for
        # "pass" anywhere in the text. They must not be scored as a pass.
        ("The response does not pass the criterion.", None),
        ("does not pass", None),
        ("pass/fail unclear", None),
        ("PASS or FAIL", None),
        # A decided verdict that merely mentions the other word is still decided.
        ("FAIL - does not pass the criterion", False),
        ("", None),
        ("I cannot determine this.", None),
    ],
)
def test_parse_verdict_requires_a_leading_verdict(raw: str, expected: bool | None) -> None:
    passed, _note = parse_verdict(raw)
    assert passed is expected


def _tiny_bank() -> tuple[FrqBank, IRTBank]:
    """Two criteria on one scenario plus two single-criterion scenarios."""
    scenarios = {
        sid: Scenario(scenario_id=sid, prompt=f"prompt for {sid}") for sid in ("s1", "s2", "s3")
    }
    criteria = (
        Criterion(criterion_id="c1a", scenario_id="s1", text="a"),
        Criterion(criterion_id="c1b", scenario_id="s1", text="b"),
        Criterion(criterion_id="c2", scenario_id="s2", text="c"),
        Criterion(criterion_id="c3", scenario_id="s3", text="d"),
    )
    params = {
        "c1a": IRTItemParams(item_id="c1a", difficulty=0.0, discrimination=1.2),
        "c1b": IRTItemParams(item_id="c1b", difficulty=0.4, discrimination=1.0),
        "c2": IRTItemParams(item_id="c2", difficulty=-0.3, discrimination=1.1),
        "c3": IRTItemParams(item_id="c3", difficulty=0.2, discrimination=0.9),
    }
    return FrqBank(name="tiny", scenarios=scenarios, criteria=criteria), IRTBank(
        params=params, dimensions=1
    )


class _RespGen:
    def __init__(self, unservable: set[str] | None = None) -> None:
        self.unservable = unservable or set()

    def generate_one(self, scenario):
        if scenario.scenario_id in self.unservable:
            raise ScenarioUnservable(f"{scenario.scenario_id}: prompt does not fit")
        return f"response for {scenario.scenario_id}"


class _Judge:
    def __init__(self, abstain_on: set[str] | None = None) -> None:
        self.abstain_on = abstain_on or set()

    def evaluate(self, scenario, criterion, response_text):
        if criterion.criterion_id in self.abstain_on:
            return JudgeVerdict(
                criterion_id=criterion.criterion_id, passed=False, unscorable_reason="empty output"
            )
        return JudgeVerdict(criterion_id=criterion.criterion_id, passed=True, rationale="ok")


def _sink(tmp_path, run_id="run-1"):
    return ResultSink(
        str(tmp_path / "out"), run_id=run_id, checkpoint_slug="ckpt", local_dir=tmp_path
    )


def test_session_skips_an_unservable_scenario_and_keeps_going(tmp_path) -> None:
    bank, irt = _tiny_bank()
    sink = _sink(tmp_path)
    summary = run_session(
        UniFrqStyle(),
        bank=bank,
        irt_bank=irt,
        respgen=_RespGen(unservable={"s1"}),
        judge=_Judge(),
        sink=sink,
        se_threshold=0.0,
        max_items=10,
    )
    # s1 is dropped along with BOTH of its criteria; the other two still get scored.
    assert summary["scenarios_skipped"] == 1
    assert summary["criteria_administered"] == 2
    assert summary["report"].num_items_administered == 2
    assert (sink.local_dir / "skipped.jsonl").exists()


def test_session_treats_an_abstention_as_missing_data_not_a_failure(tmp_path) -> None:
    bank, irt = _tiny_bank()
    sink = _sink(tmp_path)
    summary = run_session(
        UniFrqStyle(),
        bank=bank,
        irt_bank=irt,
        respgen=_RespGen(),
        judge=_Judge(abstain_on={"c2"}),
        sink=sink,
        se_threshold=0.0,
        max_items=10,
    )
    assert summary["criteria_abstained"] == 1
    assert summary["criteria_administered"] == 3
    scored = {r.criterion_id for r in summary["report"].responses}
    assert "c2" not in scored


def test_session_persists_incrementally(tmp_path) -> None:
    bank, irt = _tiny_bank()
    sink = _sink(tmp_path)
    run_session(
        UniFrqStyle(),
        bank=bank,
        irt_bank=irt,
        respgen=_RespGen(),
        judge=_Judge(),
        sink=sink,
        se_threshold=0.0,
        max_items=10,
    )
    responses = (sink.local_dir / "responses.jsonl").read_text().strip().splitlines()
    judgments = (sink.local_dir / "judgments.jsonl").read_text().strip().splitlines()
    trajectory = (sink.local_dir / "trajectory.jsonl").read_text().strip().splitlines()
    assert len(responses) == 3  # one generation per scenario, cached across its criteria
    assert len(judgments) == 4
    assert len(trajectory) == 4
    assert json.loads(trajectory[0])["step"] == 1


def test_sink_namespaces_by_checkpoint_and_run_and_writes_success_last(tmp_path) -> None:
    sink = _sink(tmp_path, run_id="r42")
    assert sink.local_dir.name.startswith("r42-")
    assert sink.local_dir.parent.name.startswith("ckpt-")
    sink.write_json("manifest.json", {"a": 1})
    assert not (sink.local_dir / "_SUCCESS").exists()
    sink.finalize(ok=True)
    assert (sink.local_dir / "_SUCCESS").exists()


def test_sink_omits_success_marker_on_failure(tmp_path) -> None:
    sink = _sink(tmp_path, run_id="r43")
    sink.write_json("manifest.json", {"a": 1})
    sink.finalize(ok=False)
    assert not (sink.local_dir / "_SUCCESS").exists()


def test_slugify_is_collision_resistant_and_traversal_safe() -> None:
    assert slugify("s3://bucket/ckpts/step_1000").startswith("s3-bucket-ckpts-step_1000-")
    # Checkpoints sharing a long suffix must not land in the same directory.
    assert slugify("prefix-" + "x" * 80) != slugify("other-" + "x" * 80)
    # Inputs with no usable characters stay distinct instead of all becoming "checkpoint".
    assert slugify("") != slugify("---")
    # Never emits a path separator or a traversal segment.
    for probe in ("..", ".", "../../etc", "/", "\u6a21\u578b"):
        out = slugify(probe)
        assert "/" not in out and out not in {".", ".."} and ".." not in out


def test_sink_rejects_artifact_names_that_escape_the_run_directory(tmp_path) -> None:
    sink = _sink(tmp_path)
    for bad in ("../escape.jsonl", "sub/dir.jsonl", "", ".", ".."):
        with pytest.raises(ValueError):
            sink.append(bad, {"a": 1})
        with pytest.raises(ValueError):
            sink.write_json(bad, {"a": 1})


def test_two_checkpoints_do_not_share_a_run_directory(tmp_path) -> None:
    out = str(tmp_path / "out")
    a = ResultSink(out, run_id="r", checkpoint_slug="s3://b/ckpt/step_1000", local_dir=tmp_path)
    b = ResultSink(out, run_id="r", checkpoint_slug="s3://other/ckpt/step_1000", local_dir=tmp_path)
    assert a.local_dir != b.local_dir


def test_budget_ladder_always_ends_at_the_floor() -> None:
    # If the ladder never reaches the smallest budget, an over-long prompt can never be
    # classified as permanently unservable and is misreported as a transient outage.
    for max_tokens in (1024, 512, 100, 32):
        for attempts in (2, 3, 4):
            ladder = ResilientRespGen(
                RespGenConfig(
                    endpoint="http://x/v1",
                    served_model="t",
                    max_tokens=max_tokens,
                    max_attempts=attempts,
                )
            ).budget_ladder()
            assert ladder[-1] == 32, (max_tokens, attempts, ladder)
            assert len(ladder) <= attempts


@pytest.fixture
def http_server():
    """Serve a fixed status/body on an ephemeral port."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    servers = []

    def start(status: int, body: dict):
        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))
                payload = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format: str, *args: object) -> None:  # noqa: A002
                pass

        srv = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers.append(srv)
        return f"http://127.0.0.1:{srv.server_port}/v1"

    yield start
    for srv in servers:
        srv.shutdown()


def test_persistent_400_is_permanent_not_transient(http_server) -> None:
    url = http_server(400, {"message": "maximum context length is 4096 tokens"})
    gen = ResilientRespGen(
        RespGenConfig(endpoint=url, served_model="tutor", max_attempts=2, timeout=5.0)
    )
    with pytest.raises(ScenarioUnservable) as excinfo:
        gen.generate_one(Scenario(scenario_id="s1", prompt="x" * 100))
    assert "maximum context length" in str(excinfo.value)  # the server's body is preserved


def test_server_error_is_transient(http_server) -> None:
    url = http_server(503, {"message": "service unavailable"})
    gen = ResilientRespGen(
        RespGenConfig(endpoint=url, served_model="tutor", max_attempts=1, timeout=5.0)
    )
    with pytest.raises(TutorUnavailable):
        gen.generate_one(Scenario(scenario_id="s1", prompt="x"))
