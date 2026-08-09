"""Pin the guard that stops a submission running code the container cannot reach.

The failure this prevents leaves no trace. A spec pinning a sha that was never pushed,
or a sha copied from the previous run, produces a container that clones, checks out
something other than what the submitter is looking at, and returns a report with every
field populated and a plausible ability estimate. Nothing downstream compares the two.

So these tests are mostly about the *refusals*, and one of them is about the instrument
rather than the answer. ``check_pin`` asks the remote with ``git ls-remote`` instead of
reading ``origin/<branch>``, because a remote-tracking ref is a local cache that only a
fetch updates: on a laptop that has committed since its last fetch it reports the branch
tip confidently and wrongly, which is precisely the state a submitter is in at the moment
they submit. ``test_a_stale_remote_tracking_ref_does_not_fool_it`` constructs exactly that
state - remote ahead, cache stale - and would pass against either implementation if the
distinction were merely stylistic.

Each fixture is a real repository with a real ``file://`` remote rather than a mock,
because what is being tested is which of several nearly-identical git facts the script
consults.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from ..scripts import check_submission_pin as pin

SPEC = """\
schema_version: 1
workload_profile: olmo-core-check
command: >-
  bash -lc 'git clone --filter=blob:none {url} /opt/olmo-eval-full
  && cd /opt/olmo-eval-full
  && git checkout {sha}
  && python -m diagnostics.mcq_cat.runner --cat-style uni_mcq'
"""


def git(repo: Path, *args: str) -> str:
    done = subprocess.run(
        ("git", "-c", "user.email=t@t", "-c", "user.name=t", *args),
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout.strip()


@pytest.fixture
def ticket(tmp_path: Path) -> Path:
    """A pushed ticket branch with a `file://` origin, as a submitter would have."""
    origin = tmp_path / "origin.git"
    subprocess.run(("git", "init", "--bare", "-q", str(origin)), check=True)

    repo = tmp_path / "work"
    subprocess.run(
        ("git", "clone", "-q", origin.as_uri(), str(repo)), check=True, capture_output=True
    )
    (repo / "harness.py").write_text("x = 1\n", encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "init")
    git(repo, "checkout", "-qb", "alice-smollm2-ticket")
    git(repo, "push", "-q", "origin", "alice-smollm2-ticket")
    return repo


def write_spec(repo: Path, sha: str) -> Path:
    spec = repo / "spec.yaml"
    url = git(repo, "remote", "get-url", "origin")
    spec.write_text(SPEC.format(url=url, sha=sha), encoding="utf-8")
    return spec


def run(spec: Path, repo: Path, *, do_repin: bool = False, strict: bool = False) -> int:
    return pin.check(spec, repo, do_repin=do_repin, strict_branch=strict)


class TestItAcceptsWhatTheContainerCanActuallyReach:
    def test_a_pushed_sha_that_matches_head_passes(self, ticket: Path) -> None:
        spec = write_spec(ticket, git(ticket, "rev-parse", "HEAD"))

        assert run(spec, ticket) == 0

    def test_repin_writes_head_and_then_agrees_with_itself(self, ticket: Path) -> None:
        spec = write_spec(ticket, "0" * 40)

        assert run(spec, ticket, do_repin=True) == 0
        assert git(ticket, "rev-parse", "HEAD") in spec.read_text(encoding="utf-8")

    def test_an_untracked_file_is_not_a_dirty_tree(self, ticket: Path) -> None:
        """A fresh clone has none, but scratch left by a probe must not block a submit."""
        (ticket / "scratch.log").write_text("noise\n", encoding="utf-8")
        spec = write_spec(ticket, git(ticket, "rev-parse", "HEAD"))

        assert run(spec, ticket) == 0


class TestItRefusesWhatWouldRunSilently:
    def test_an_unpushed_commit_is_refused(self, ticket: Path) -> None:
        (ticket / "harness.py").write_text("x = 2\n", encoding="utf-8")
        git(ticket, "commit", "-qam", "amend the harness")
        spec = write_spec(ticket, git(ticket, "rev-parse", "HEAD"))

        with pytest.raises(SystemExit, match="unpushed or has diverged"):
            run(spec, ticket)

    def test_a_stale_remote_tracking_ref_does_not_fool_it(self, ticket: Path) -> None:
        """The whole reason the check asks the remote rather than reading a local ref.

        Another clone pushes, so the remote moves and this clone's `origin/<branch>` is
        left behind. A check reading the cache sees agreement where there is none.
        """
        other = ticket.parent / "other"
        subprocess.run(
            (
                "git",
                "clone",
                "-q",
                "-b",
                "alice-smollm2-ticket",
                git(ticket, "remote", "get-url", "origin"),
                str(other),
            ),
            check=True,
            capture_output=True,
        )
        (other / "harness.py").write_text("x = 3\n", encoding="utf-8")
        git(other, "commit", "-qam", "someone else moved the branch")
        git(other, "push", "-q", "origin", "alice-smollm2-ticket")

        cached = git(ticket, "rev-parse", "refs/remotes/origin/alice-smollm2-ticket")
        assert cached == git(ticket, "rev-parse", "HEAD"), "fixture must leave the cache stale"

        spec = write_spec(ticket, git(ticket, "rev-parse", "HEAD"))
        with pytest.raises(SystemExit, match="on the remote but HEAD is"):
            run(spec, ticket)

    def test_a_dirty_tracked_file_is_refused(self, ticket: Path) -> None:
        spec = write_spec(ticket, git(ticket, "rev-parse", "HEAD"))
        (ticket / "harness.py").write_text("x = 99\n", encoding="utf-8")

        with pytest.raises(SystemExit, match="uncommitted tracked changes"):
            run(spec, ticket)

    def test_a_branch_name_is_refused_where_a_sha_is_required(self, ticket: Path) -> None:
        """It would resolve in-container to whatever is pushed when the job starts."""
        spec = write_spec(ticket, "alice-smollm2-ticket")

        with pytest.raises(SystemExit, match="not a full 40-character sha"):
            run(spec, ticket)

    def test_a_sha_from_a_previous_run_is_refused(self, ticket: Path) -> None:
        stale = git(ticket, "rev-parse", "HEAD")
        (ticket / "harness.py").write_text("x = 4\n", encoding="utf-8")
        git(ticket, "commit", "-qam", "move on")
        git(ticket, "push", "-q", "origin", "alice-smollm2-ticket")
        spec = write_spec(ticket, stale)

        with pytest.raises(SystemExit, match="but HEAD is"):
            run(spec, ticket)

    def test_a_spec_cloning_elsewhere_is_refused(self, ticket: Path) -> None:
        """Verifying a push here would say nothing about the repo the container reads."""
        spec = ticket / "spec.yaml"
        spec.write_text(
            SPEC.format(
                url="https://github.com/edu-llm/some-other-repo",
                sha=git(ticket, "rev-parse", "HEAD"),
            ),
            encoding="utf-8",
        )

        with pytest.raises(SystemExit, match="but origin is"):
            run(spec, ticket)


class TestItRefusesABranchThatWouldCostSomething:
    def test_an_edullm_prefix_is_refused_because_it_builds_an_image(self, ticket: Path) -> None:
        git(ticket, "checkout", "-qb", "edullm/alice-smollm2-ticket")
        spec = write_spec(ticket, git(ticket, "rev-parse", "HEAD"))

        with pytest.raises(SystemExit, match="fires edullm-platform-build"):
            run(spec, ticket)

    def test_a_non_ticket_branch_is_refused_under_strict(self, ticket: Path) -> None:
        git(ticket, "checkout", "-q", "-b", "flow/uni-mcq")
        spec = write_spec(ticket, git(ticket, "rev-parse", "HEAD"))

        with pytest.raises(SystemExit, match="is not a ticket branch"):
            run(spec, ticket, strict=True)

    def test_a_detached_head_is_refused(self, ticket: Path) -> None:
        git(ticket, "checkout", "-q", "--detach")
        spec = write_spec(ticket, git(ticket, "rev-parse", "HEAD"))

        with pytest.raises(SystemExit, match="HEAD is detached"):
            run(spec, ticket)
