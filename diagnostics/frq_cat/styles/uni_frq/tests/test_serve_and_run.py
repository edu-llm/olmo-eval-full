"""Exercises the on-node entrypoint without a GPU.

``VLLM_CMD`` is substituted with a stub OpenAI-compatible server, which lets this test
cover the parts of the script that are not AWS-specific: server launch, the
``/v1/models`` health poll, argument wiring into the pipeline, exit-code propagation and
teardown. Anything that needs real EC2/SSM (instance launch, IAM, billing guard) is out of
scope here and is owned by the shared launcher.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pytest

_STYLE_DIR = Path(__file__).resolve().parent.parent
_SCRIPT = _STYLE_DIR / "aws" / "serve_and_run.sh"
_REPO_ROOT = _STYLE_DIR.parents[3]

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")

_STUB_SERVER = """
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer

port = 8000
for i, a in enumerate(sys.argv):
    if a == "--port":
        port = int(sys.argv[i + 1])

class H(BaseHTTPRequestHandler):
    def _send(self, payload):
        b = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        self._send({"object": "list", "data": [{"id": "tutor"}]})

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))
        self._send({"choices": [{"message": {"content": "PASS\\nmeets the criterion"}}]})

    def log_message(self, format: str, *args: object) -> None:
        pass

HTTPServer(("127.0.0.1", port), H).serve_forever()
"""


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _run(tmp_path: Path, **env_overrides: str) -> subprocess.CompletedProcess[str]:
    stub = tmp_path / "stub_server.py"
    stub.write_text(_STUB_SERVER, encoding="utf-8")
    port = _free_port()
    env = {
        **os.environ,
        "CHECKPOINT": "stub/checkpoint",
        "OUT": str(tmp_path / "out"),
        "VLLM_CMD": f"{sys.executable} {stub}",
        "TUTOR_PORT": str(port),
        "JUDGE_MODE": "api",
        "JUDGE_ENDPOINT": f"http://127.0.0.1:{port}/v1",
        "JUDGE_API_KEY_ENV": "STUB_JUDGE_KEY",
        "STUB_JUDGE_KEY": "not-a-real-key",
        "LOG_DIR": str(tmp_path / "logs"),
        "READY_TIMEOUT": "60",
        "MAX_ITEMS": "2",
        "SE_THRESHOLD": "0.0",
        "RUN_ID": "test-run",
        "REPO_DIR": str(_REPO_ROOT),
        "PYBIN": sys.executable,
        **env_overrides,
    }
    return subprocess.run(
        ["bash", str(_SCRIPT)], env=env, capture_output=True, text=True, timeout=300, check=False
    )


def test_script_has_valid_syntax() -> None:
    assert subprocess.run(["bash", "-n", str(_SCRIPT)], check=False).returncode == 0


def test_dry_run_reports_the_plan_without_starting_anything(tmp_path: Path) -> None:
    result = _run(tmp_path, DRY_RUN="true")
    assert result.returncode == 0, result.stderr
    assert "[dry-run]" in result.stdout


def test_serves_polls_health_and_runs_the_pipeline(tmp_path: Path) -> None:
    result = _run(tmp_path)
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "ready after" in result.stdout  # the /v1/models poll actually succeeded

    run_dir = tmp_path / "out"
    assert (run_dir / "_SUCCESS").exists()
    report = json.loads((run_dir / "cat_report.json").read_text())
    assert report["cat_style"] == "uni_frq"
    assert report["num_items_administered"] == 2
    # The judge identity from judge_frontier.yaml reached the report, authenticated.
    assert report["run"]["judge"]["authenticated"] is True


def test_submission_spec_matches_what_the_flow_actually_needs() -> None:
    """The .edullm spec is how a run reaches the platform; keep it wired to real paths."""
    import yaml

    spec = yaml.safe_load((_REPO_ROOT / ".edullm" / "run-frq-cat.yaml").read_text())
    assert spec["schema_version"] == 1
    command = spec["command"]
    # FRQ serves a model, so it must install the server extra MCQ deliberately omits.
    assert '".[hf,s3,vllm]"' in command
    # It runs the serving entrypoint, not the runner, and that script must exist.
    assert "aws/serve_and_run.sh" in command
    assert _SCRIPT.exists()
    # Results go to the platform-provided prefix, like every other spec in .edullm.
    assert "$EDULLM_OUTPUT_PREFIX" in command
    # curl is required by the readiness probe and is not on the base image.
    assert "curl" in command
    # olmo-eval-full registers olmo-eval-check/olmo-eval-sweep; olmo-core-check is
    # for OLMo-core submissions and would be the wrong workload here.
    assert spec["workload_profile"] == "olmo-eval-sweep"
    # 4096 is the tutor's hard limit, not a preference.
    assert "TUTOR_MAX_MODEL_LEN=4096" in command
    # A --filter=blob:none clone that loses a blob still exits 0, so the checkout has to be
    # audited before the install or a short bank grades as a real one.
    assert "--filter=blob:none" in command
    checkout = command.index("git checkout")
    guard = command.index("git status --porcelain --untracked-files=no")
    install = command.index("pip install")
    assert checkout < guard < install, "the guard must sit between the checkout and the install"
    assert "exit 3" in command


def test_warns_when_the_context_window_is_too_small(tmp_path: Path) -> None:
    result = _run(tmp_path, DRY_RUN="true", TUTOR_MAX_MODEL_LEN="4096")
    assert result.returncode == 0
    assert "below the bank's longest prompt" in result.stderr
