"""Refuse a submission whose spec pins code the container will not run.

Run by the submitting agent, immediately before ``edullm submit``::

    python -m diagnostics.mcq_cat.styles.uni_mcq.scripts.check_submission_pin \\
        --spec .edullm/run-new-banks-cat.yaml --repin

The harness does not travel with a submission. The container clones this repository
from GitHub and checks out one sha, so **the code that runs is whatever was pushed**,
and the spec's ``git checkout`` argument is the only thing that decides which. Three
ways that goes wrong, none of them caught anywhere today:

* the working tree has uncommitted changes, so what ran is not what the submitter is
  looking at when they read the report;
* the sha was never pushed, so the clone cannot reach it -- or, worse, the branch
  exists at an older commit and the run silently evaluates yesterday's harness;
* the spec names something else entirely, most often a sha copied from the previous
  run because the line looked right.

What makes these worth a script rather than a paragraph is that none of them fails
loudly. ``run-native-cat-sweep.yaml`` records the class: a spec pinned several commits
behind "evaluates a harness nobody has and returns a well-formed report with a
plausible theta. That is a wrong answer rather than a failure, and nothing downstream
catches it." The one existing guard that sounds like this one is not: ``edullm check``
refuses ``uncommitted_changes`` against the **OLMo-core** clone it is run in, so a clean
check says nothing about whether this repository's work is pushed.

**``git ls-remote`` rather than a remote-tracking ref.** ``origin/<branch>`` and
``@{upstream}`` are local caches that a fetch updates and nothing else does, so on a
laptop that has not fetched since the last push they answer confidently and wrongly --
which is exactly the failure being guarded. ``ls-remote`` asks the remote itself, in one
round trip, writing no object and touching no ref.

**``--repin`` exists so the guard gets run at all.** A check that must be remembered is a
check that has already been forgotten once; making this the sanctioned way to write the
pin means the pin and its verification cannot drift apart. Neither CI nor a hook
substitutes: ``ci.yml`` fires only on ``main`` so it never sees a ticket branch,
``testpaths`` excludes this directory, ``.git/hooks`` is unversioned and would guard one
laptop, and a pre-push hook fires at the wrong moment anyway -- pushing is never the
fault, submitting against a stale pin is.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import yaml

#: The clone and checkout the spec's shell command must contain, exactly once each.
#:
#: Matched rather than parsed because ``command`` is one folded scalar of shell, not
#: structure. Two clones or two checkouts is not a spec this can reason about, and
#: guessing which one the container would honour is the kind of assumption this file
#: exists to remove.
#:
#: The URL is found by shape rather than by scheme. Every committed spec clones over
#: ``https``, but pinning that here would make the check refuse to reason about any spec
#: written differently -- and would make it untestable against a local remote, which is
#: the only way to exercise the push checks without a network.
_CLONE_SEGMENT = re.compile(r"git clone\b(?P<args>[^&;']*)")
_CHECKOUT = re.compile(r"git checkout\s+(?P<ref>\S+)")


def _clone_url(segment: str) -> str | None:
    """Return the first argument of a `git clone` that names a location, or None."""
    for token in segment.split():
        if "://" in token or token.startswith("git@"):
            return token.removesuffix(".git")
    return None


#: A full object name. An abbreviation cannot be compared for equality against
#: ``rev-parse`` output, and a branch name resolves in the container to whatever was
#: pushed by the time it runs, which is the silent failure this refuses.
_FULL_SHA = re.compile(r"\A[0-9a-f]{40}\Z")

#: Ticket branches are named for one submitter and one model, and are deliberately not
#: under ``edullm/`` -- that prefix is the only thing in this repository that fires an
#: image build, and a CAT run takes its image from OLMo-core, so a ticket named that way
#: would spend several gigabytes building something nothing pulls.
_TICKET = re.compile(r"\A[a-z0-9][a-z0-9._-]*-ticket\Z")


class Refused(SystemExit):
    """Abort naming the fix, in the manner of ``vendor_bank``'s guards."""

    def __init__(self, message: str) -> None:
        super().__init__(f"{message} Nothing was submitted.")


def _git(*args: str, cwd: Path | None = None) -> str:
    """Return stdout for a git command, or raise :class:`Refused` describing the failure."""
    done = subprocess.run(("git", *args), cwd=cwd, capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise Refused(
            f"`git {' '.join(args)}` failed with exit {done.returncode}: "
            f"{done.stderr.strip() or '(no output)'}."
        )
    return done.stdout.strip()


def current_branch(repo: Path) -> str:
    """Return the checked-out branch, refusing a detached HEAD."""
    done = subprocess.run(
        ("git", "symbolic-ref", "--quiet", "--short", "HEAD"),
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    if done.returncode != 0:
        raise Refused(
            "HEAD is detached, so there is no branch to push and no ticket to record the "
            "run onto. Check out the ticket branch for this submission first."
        )
    return done.stdout.strip()


def check_branch_is_a_ticket(branch: str, *, strict: bool) -> None:
    """Refuse an `edullm/` prefix outright, and a non-ticket name unless allowed."""
    if branch.startswith("edullm/"):
        raise Refused(
            f"{branch!r} is under `edullm/`, which is the one prefix that fires "
            f"edullm-platform-build.yml. A CAT run takes its image from OLMo-core, so this "
            f"would build a multi-gigabyte image nothing pulls. Rename the branch to "
            f"<submitter>-<model>-ticket."
        )
    if strict and not _TICKET.match(branch):
        raise Refused(
            f"{branch!r} is not a ticket branch. One submission gets one "
            f"<submitter>-<model>-ticket branch, so that the harness changes a checkpoint "
            f"needed, the spec, the run id and the report are all in one place. Pass "
            f"--any-branch to submit from somewhere else deliberately."
        )


def check_tree_is_clean(repo: Path) -> None:
    """Refuse uncommitted tracked changes outside `.edullm/`.

    ``.edullm/`` is exempt on purpose: ``--spec`` is read off the laptop and compiled
    into the submission, so a spec is never cloned and its edits cannot reach the
    container. That exemption is what lets the pin be written just before submitting
    and the spec committed just after, which is the ordering that makes a spec able to
    record the sha of the very commit the run used.
    """
    dirty = _git(
        "status", "--porcelain", "--untracked-files=no", "--", ".", ":(exclude).edullm", cwd=repo
    )
    if dirty:
        raise Refused(
            "The working tree has uncommitted tracked changes, so the container would run "
            "the last commit rather than what you are looking at:\n"
            + dirty
            + "\nCommit and push them, or stash them."
        )


def read_spec(spec_path: Path) -> tuple[str, str]:
    """Return the (clone url, checkout ref) the spec's command names."""
    try:
        document = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as err:
        raise Refused(f"{spec_path} could not be read as YAML: {err}.") from err

    command = (document or {}).get("command", "")
    clones = [
        url
        for segment in _CLONE_SEGMENT.findall(command)
        if (url := _clone_url(segment)) is not None
    ]
    checkouts = _CHECKOUT.findall(command)
    if len(clones) != 1 or len(checkouts) != 1:
        raise Refused(
            f"{spec_path} names {len(clones)} clone(s) and {len(checkouts)} checkout(s); "
            f"this can only verify a command with exactly one of each, because which one "
            f"the container honours is otherwise a guess."
        )
    return clones[0], checkouts[0]


def check_clone_is_this_repo(url: str, repo: Path) -> None:
    """Refuse a spec that clones somewhere other than this repository's origin."""
    origin = _git("remote", "get-url", "origin", cwd=repo).removesuffix(".git")
    if url.removesuffix(".git") != origin:
        raise Refused(
            f"The spec clones {url!r} but origin is {origin!r}, so verifying a push here "
            f"would say nothing about the repository the container reads."
        )


def check_pin(ref: str, repo: Path, branch: str, url: str) -> str:
    """Refuse unless the spec pins a full sha equal to HEAD and present on the remote."""
    if not _FULL_SHA.match(ref):
        raise Refused(
            f"The spec checks out {ref!r}, which is not a full 40-character sha. A branch "
            f"name resolves in the container to whatever is pushed when the job starts, "
            f"and an abbreviation cannot be compared for equality. Run with --repin."
        )

    head = _git("rev-parse", "HEAD", cwd=repo)
    if ref != head:
        raise Refused(
            f"The spec pins {ref} but HEAD is {head}, so the run would evaluate different "
            f"code than this checkout. Run with --repin."
        )

    listed = _git("ls-remote", url, f"refs/heads/{branch}", cwd=repo)
    if not listed:
        raise Refused(
            f"{branch!r} does not exist on the remote, so the container's clone cannot "
            f"reach {head}. Push the branch."
        )
    remote_tip = listed.split()[0]
    if remote_tip != head:
        behind = subprocess.run(
            ("git", "merge-base", "--is-ancestor", head, remote_tip),
            cwd=repo,
            capture_output=True,
            check=False,
        )
        relation = (
            "the remote is ahead of this checkout"
            if behind.returncode == 0
            else "this checkout is unpushed or has diverged from the remote"
        )
        raise Refused(
            f"{branch!r} is at {remote_tip} on the remote but HEAD is {head}: {relation}. "
            f"Push before submitting -- the container clones from GitHub, not from disk."
        )
    return head


def repin(spec_path: Path, old_ref: str, new_sha: str) -> bool:
    """Rewrite the spec's checkout argument to ``new_sha``. Returns whether it changed."""
    text = spec_path.read_text(encoding="utf-8")
    updated = text.replace(f"git checkout {old_ref}", f"git checkout {new_sha}", 1)
    if updated == text:
        return False
    spec_path.write_text(updated, encoding="utf-8")
    return True


def check(spec_path: Path, repo: Path, *, do_repin: bool, strict_branch: bool) -> int:
    """Run every check in order, repinning first when asked."""
    branch = current_branch(repo)
    check_branch_is_a_ticket(branch, strict=strict_branch)
    check_tree_is_clean(repo)

    url, ref = read_spec(spec_path)
    check_clone_is_this_repo(url, repo)

    if do_repin:
        head = _git("rev-parse", "HEAD", cwd=repo)
        if repin(spec_path, ref, head):
            print(f"repinned {spec_path} to {head}")
        url, ref = read_spec(spec_path)

    sha = check_pin(ref, repo, branch, url)
    print(
        f"branch {branch} at {sha}, present on the remote, and {spec_path.name} pins it. "
        f"Safe to submit."
    )
    print(
        "Not checked here, because this cannot see them: whether this is the ticket, the "
        "checkpoint and the benchmarks you meant, and whether the OLMo-core --commit "
        "published an image. Compile decides the last one and refuses if it did not."
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m diagnostics.mcq_cat.styles.uni_mcq.scripts.check_submission_pin",
        description="Refuse a submission whose spec pins code the container will not run.",
    )
    parser.add_argument("--spec", required=True, type=Path, help="Path to the .edullm spec.")
    parser.add_argument(
        "--repo",
        type=Path,
        default=None,
        help="Repository to check (default: the one holding this file).",
    )
    parser.add_argument(
        "--repin",
        action="store_true",
        help=(
            "Write HEAD into the spec's git checkout line before verifying, so the pin has "
            "one author rather than being copied by hand."
        ),
    )
    parser.add_argument(
        "--any-branch",
        action="store_true",
        help="Allow a branch that is not named <submitter>-<model>-ticket.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo = args.repo or Path(__file__).resolve().parents[5]
    return check(args.spec, repo, do_repin=args.repin, strict_branch=not args.any_branch)


if __name__ == "__main__":
    sys.exit(main())
