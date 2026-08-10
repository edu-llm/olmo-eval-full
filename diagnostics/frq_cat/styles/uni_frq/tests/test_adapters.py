"""Tests for the judge contracts, and for the mismatch that made them necessary.

The defect these exist to prevent is not a crash. Sending the team's frozen JSON prompt
and reading the reply with the text parser abstains on every criterion, so the estimator
sees nothing and the run reports theta 0.0 from an untouched prior: a well-formed report
of a model nobody measured.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from diagnostics.frq_cat.base import CATState
from diagnostics.frq_cat.common.judge import JudgeSpec
from diagnostics.frq_cat.styles.uni_frq import adapters
from diagnostics.frq_cat.styles.uni_frq.judge_client import ResilientJudge, parse_verdict
from diagnostics.frq_cat.styles.uni_frq.style import UniFrqStyle

from .test_pipeline import _tiny_bank

#: What the frozen evidence-first contract actually returns.
_STRICT_REPLY = json.dumps(
    {
        "evidence": "the response states that anhedonia is shared but the disorders differ",
        "rationale": "every required part is directly established",
        "verdict": "pass",
    }
)


@pytest.fixture
def one_shot_judge():
    """Serve a single scripted body and record the request that produced it."""
    servers: list[HTTPServer] = []

    def start(body: object, status: int = 200):
        seen: list[dict] = []

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))
                seen.append(json.loads(raw))
                payload = (body if isinstance(body, str) else json.dumps(body)).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format: str, *args: object) -> None:
                pass

        srv = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers.append(srv)
        return f"http://127.0.0.1:{srv.server_port}/v1", seen

    yield start
    for srv in servers:
        srv.shutdown()


def _case():
    bank, _ = _tiny_bank()
    return bank.get_scenario("s1"), bank.criteria[0]


# ---------------------------------------------------------------------- the registry


def test_every_registered_name_resolves_and_unknown_names_say_what_exists() -> None:
    for name in ("generic-binary", "generic-binary-json", "generic-binary-strict"):
        assert adapters.get_adapter(name).name

    with pytest.raises(KeyError) as excinfo:
        adapters.get_adapter("gpt-4o")
    message = str(excinfo.value)
    assert "generic-binary-strict" in message, "the error must name the valid options"


def test_generic_binary_keeps_its_local_meaning() -> None:
    """The shared JudgeSpec defaults to this name, so it must stay the text contract.

    The pilot on frq/tutorbench uses the same name for JSON. Redefining it here would
    switch every config that omits `adapter` onto a prompt its parser cannot read.
    """
    assert JudgeSpec(name="t", model_id="m").adapter == "generic-binary"
    assert adapters.get_adapter("generic-binary") is adapters.SERVED_PASS_FAIL
    assert adapters.get_adapter("served-pass-fail") is adapters.SERVED_PASS_FAIL
    assert adapters.SERVED_PASS_FAIL.response_format is None
    assert adapters.SERVED_PASS_FAIL.evidence_gated is False


# ------------------------------------------------------------------------ the prompts


def test_strict_prompt_carries_the_gate_that_earned_the_zero_false_pass_rate() -> None:
    scenario, criterion = _case()
    messages = adapters.GENERIC_BINARY_STRICT.build_messages(scenario, criterion, "an answer")
    assert len(messages) == 1 and messages[0]["role"] == "user"
    prompt = messages[0]["content"]

    assert "Evidence-gated decision policy:" in prompt
    assert 'Default to "fail" when uncertain' in prompt
    assert "untrusted data" in prompt, "the prompt-injection guard must survive"
    # Evidence before verdict is the ordering that cut false passes, not a style choice.
    assert prompt.index('"evidence"') < prompt.index('"verdict"')
    for fragment in (scenario.prompt, criterion.text, "an answer"):
        assert fragment in prompt


def test_the_two_json_prompts_differ_only_where_the_pilot_says_they_do() -> None:
    scenario, criterion = _case()
    plain = adapters.GENERIC_BINARY_JSON.build_messages(scenario, criterion, "a")[0]["content"]
    strict = adapters.GENERIC_BINARY_STRICT.build_messages(scenario, criterion, "a")[0]["content"]
    assert "Strictness gate" in strict and "Strictness gate" not in plain
    # The non-strict contract puts the verdict first, which is why a truncated reply is
    # still recoverable there and is not under the strict one.
    assert plain.index('"verdict"') < plain.index('"evidence"')


def test_the_bank_supplies_no_reference_so_both_prompts_say_so() -> None:
    """Every TutorEval scenario has an empty reference_solution; the fallback must fire."""
    scenario, criterion = _case()
    prompt = adapters.GENERIC_BINARY_STRICT.build_messages(scenario, criterion, "a")[0]["content"]
    assert "No reference answer was provided" in prompt


# ------------------------------------------------------------------- the verdict ladder


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        (_STRICT_REPLY, True),
        (json.dumps({"evidence": "NONE", "rationale": "absent", "verdict": "fail"}), False),
        # Markdown fences are forbidden by the prompt and emitted anyway.
        (f"```json\n{_STRICT_REPLY}\n```", True),
        # Unescaped LaTeX makes the object invalid JSON; the field regex still reads it.
        (r'{"evidence": "\frac{1}{2} is given", "verdict": "fail"}', False),
        # Truncated after the verdict, which the non-strict field order permits.
        ('{"verdict": "pass", "rationale": "it cites the ste', True),
        ("[RESULT] 5", True),
        ("[RESULT] 2", False),
        ("pass - the response covers it", True),
        ("fail", False),
    ],
)
def test_json_ladder_reads_every_shape_the_pilot_reads(reply: str, expected: bool) -> None:
    assert adapters.parse_json_verdict(reply).passed is expected


@pytest.mark.parametrize(
    "reply",
    [
        "",
        "   ",
        "I cannot evaluate this criterion.",
        # Strict puts the verdict last, so a reply cut short loses it entirely. This is
        # the case the 6144-token floor exists to prevent.
        '{"evidence": "the response works through the derivation step by step and the',
    ],
)
def test_json_ladder_abstains_rather_than_guessing(reply: str) -> None:
    parsed = adapters.parse_json_verdict(reply)
    assert parsed.passed is None
    assert parsed.reason, "an abstention must carry a reason"


def test_a_scored_json_reply_keeps_its_evidence_and_rationale() -> None:
    parsed = adapters.parse_json_verdict(_STRICT_REPLY)
    assert parsed.passed is True
    assert "anhedonia" in parsed.evidence
    assert parsed.rationale == "every required part is directly established"


def test_the_mismatch_this_whole_module_exists_to_prevent() -> None:
    """The text parser abstains on the frozen judge's reply; the JSON parser scores it.

    Left as one test because the pair is the point: read alone, either line looks like
    correct behaviour, and it is only together that they show a whole run being lost.
    """
    passed, reason = parse_verdict(_STRICT_REPLY)
    assert passed is None, "a JSON reply read by the text parser scores nothing"
    assert "unparseable verdict" in reason

    assert adapters.parse_json_verdict(_STRICT_REPLY).passed is True


# ------------------------------------------------------------------------- the client


def test_a_budget_below_the_floor_is_refused_before_the_run_starts() -> None:
    """Under 6144 the strict reply truncates before its verdict, losing every criterion."""
    spec = JudgeSpec(
        name="frozen", model_id="gemini-3", adapter="generic-binary-strict", max_tokens=1024
    )
    with pytest.raises(RuntimeError, match="max_tokens >= 6144"):
        ResilientJudge(spec, endpoint="http://127.0.0.1:1/v1")


def test_the_text_contract_has_no_floor() -> None:
    spec = JudgeSpec(name="frozen", model_id="qwen", max_tokens=1024)
    assert ResilientJudge(spec, endpoint="http://127.0.0.1:1/v1").adapter.min_max_tokens == 0


def test_json_contracts_ask_the_api_to_return_json(one_shot_judge) -> None:
    url, seen = one_shot_judge({"choices": [{"message": {"content": _STRICT_REPLY}}]})
    spec = JudgeSpec(
        name="frozen", model_id="gemini-3", adapter="generic-binary-strict", max_tokens=6144
    )
    scenario, criterion = _case()
    verdict = ResilientJudge(spec, endpoint=url).evaluate(scenario, criterion, "an answer")

    assert verdict.passed is True
    assert not verdict.unscorable_reason
    assert verdict.evidence, "the evidence field must reach the artifact"
    assert seen[0]["response_format"] == {"type": "json_object"}
    assert seen[0]["max_tokens"] == 6144


def test_the_text_contract_sends_no_response_format(one_shot_judge) -> None:
    url, seen = one_shot_judge({"choices": [{"message": {"content": "PASS\nfine"}}]})
    spec = JudgeSpec(name="frozen", model_id="qwen")
    scenario, criterion = _case()
    ResilientJudge(spec, endpoint=url).evaluate(scenario, criterion, "an answer")
    assert "response_format" not in seen[0], "vLLM need not accept the field"


def test_unparseable_policy_selects_between_abstaining_and_failing_closed(
    one_shot_judge,
) -> None:
    """Both are defensible; the run must record which one it used."""
    garbage = {"choices": [{"message": {"content": "I am unable to assess this."}}]}
    scenario, criterion = _case()

    url, _ = one_shot_judge(garbage)
    abstaining = ResilientJudge(
        JudgeSpec(name="a", model_id="g", adapter="generic-binary-strict", max_tokens=6144),
        endpoint=url,
    )
    verdict = abstaining.evaluate(scenario, criterion, "x")
    assert verdict.unscorable_reason, "an abstention is missing data, not a failed criterion"
    assert abstaining.provenance["unparseable_policy"] == "abstain"

    url, _ = one_shot_judge(garbage)
    fail_closed = ResilientJudge(
        JudgeSpec(
            name="a",
            model_id="g",
            adapter="generic-binary-strict",
            max_tokens=6144,
            metadata={"unparseable_policy": "fail_closed"},
        ),
        endpoint=url,
    )
    verdict = fail_closed.evaluate(scenario, criterion, "x")
    assert verdict.passed is False
    assert not verdict.unscorable_reason, "fail-closed is scored, so it must not read as missing"
    assert verdict.metadata["fail_closed_reason"]


def test_an_unknown_policy_is_refused() -> None:
    spec = JudgeSpec(name="a", model_id="g", metadata={"unparseable_policy": "guess"})
    with pytest.raises(RuntimeError, match="unparseable_policy"):
        ResilientJudge(spec, endpoint="http://127.0.0.1:1/v1")


def test_provenance_names_the_contract_that_graded() -> None:
    spec = JudgeSpec(
        name="frozen", model_id="gemini-3", adapter="generic-binary-strict", max_tokens=6144
    )
    provenance = ResilientJudge(spec, endpoint="http://127.0.0.1:1/v1").provenance
    assert provenance["adapter"] == "generic-binary-strict"
    assert provenance["prompt_version"] == "generic-binary-strict-v1"
    assert provenance["evidence_gated"] is True


# ------------------------------------------------------------------------- the report


def test_the_report_describes_the_contract_that_ran_not_a_constant() -> None:
    style = UniFrqStyle()
    state = CATState(benchmark="tutoreval")
    state.metadata["judge_contract"] = {
        "adapter": "generic-binary-strict",
        "prompt_version": "generic-binary-strict-v1",
        "evidence_gated": True,
        "unparseable_policy": "abstain",
    }
    metadata = style.report(state).metadata

    assert metadata["judge_prompt"]["runtime"] == "generic-binary-strict-v1"
    assert metadata["judge_prompt"]["matches_calibration"] is False
    assert metadata["judge_contract"]["adapter"] == "generic-binary-strict"
    # The old note claimed the runtime prompt "has no evidence gate", which is the
    # opposite of true for this contract and understates the bias.
    note = metadata["scoring_note"]
    assert note.startswith("UNCALIBRATED")
    assert "evidence gate the fitted matrix was not graded under" in note
    assert "biased low" in note


def test_the_note_still_reports_no_gate_for_the_text_contract() -> None:
    style = UniFrqStyle()
    state = CATState(benchmark="tutoreval")
    state.metadata["judge_contract"] = {
        "prompt_version": "generic-binary/v1",
        "evidence_gated": False,
    }
    assert "has no evidence gate" in style.report(state).metadata["scoring_note"]
