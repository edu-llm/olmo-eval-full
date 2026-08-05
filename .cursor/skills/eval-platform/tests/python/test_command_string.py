"""Does the submitter's command survive the platform's own parsing?

`compile_submission` turns the form field into argv with exactly this call:

    command = shlex.split(os.environ.get("FORM_COMMAND", ""))

and `RunManifest.command` requires a first element naming a program rather than a
whole command line. A command that splits wrong is refused after a reviewer has
been asked, so it is worth checking against the same function they use.
"""

import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import REPO as repo, SUBMIT as script, find_bash

bash = find_bash()
if bash is None:
    # 77 is the skip code run_tests.py understands, borrowed from autotools. The
    # submitter is a bash script, so with no bash there is nothing to assert
    # against -- which is a different outcome from the assertions failing.
    print("SKIP: no bash found; this test drives submit_eval_run.sh")
    raise SystemExit(77)

failures = []


def check(label, cond, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'} {label}" + (f"  -- {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(label)


def dry_run(*extra):
    sha = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    argv = [
        bash, str(script),
        "--checkpoint",
        "s3://sbsandbox-intern-edullm-outputs/teams/pre-training/runs/run_019f/checkpoints/step2000",
        "--team", "pre-training", "--experiment", "my-eval",
        "--wandb-project", "edullm-evals", "--eval-ref", sha, "--dry-run",
        *extra,
    ]
    out = subprocess.run(argv, capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"dry run failed:\n{out.stdout}\n{out.stderr}")
    # The command is the line after the bare "command:" label.
    lines = out.stdout.splitlines()
    idx = next(i for i, line in enumerate(lines) if line.strip() == "command:")
    return lines[idx + 1].strip()


print("as the platform will parse it:")
command = dry_run("--group", "smoke")
argv = shlex.split(command)
print(f"  split into {len(argv)} tokens")
check("splits into exactly three tokens", len(argv) == 3, str(len(argv)))
check("first token names a program, not a command line", argv[0] == "bash", argv[0])
check("second token is the shell flag", argv[1] == "-lc", argv[1])
check("not wrapped in outer quotes", not command.startswith(("'", '"')), command[:20])
check("shlex round-trips it", shlex.split(shlex.join(argv)) == argv)

print()
print("the payload the container receives:")
payload = argv[2]
check("installs git conditionally before pip", "command -v git" in payload and "apt-get install" in payload)
check("git install precedes the pip install", payload.index("apt-get") < payload.index("pip install"))
check("pip install precedes the runner", payload.index("pip install") < payload.index("run_eval"))
check("brings transformers via the hf extra", "olmo-eval[hf] @" in payload)
check("tarball is pinned to a 40-hex sha",
      __import__("re").search(r"/archive/[0-9a-f]{40}\.tar\.gz", payload) is not None)
check("invokes the runner by module path", "python -m olmo_eval.platform.run_eval" in payload)
check("passes the checkpoint", "--checkpoint s3://sbsandbox-intern-edullm-outputs/" in payload)
check("passes the resolved benchmark list, quoted",
      '--benchmarks "csqa hellaswag piqa socialiqa arc_easy"' in payload)
check("passes the group's limit", "--limit 2" in payload)

print()
print("the payload is itself a valid shell line:")
inner = shlex.split(payload)
check("payload splits without a quoting error", len(inner) > 10, str(len(inner)))
check("benchmark list survives as ONE argument to --benchmarks",
      inner[inner.index("--benchmarks") + 1] == "csqa hellaswag piqa socialiqa arc_easy",
      inner[inner.index("--benchmarks") + 1])

print()
print("no limit when the group has none:")
plain = shlex.split(dry_run("--group", "default"))[2]
check("default group carries no --limit", "--limit" not in plain)
check("still passes all five benchmarks",
      '--benchmarks "csqa hellaswag piqa socialiqa arc_easy"' in plain)

print()
print("an explicit limit overrides the group's:")
over = shlex.split(dry_run("--group", "smoke", "--limit", "5"))[2]
check("--limit 5 reaches the runner", "--limit 5" in over)
check("the group's 2 is gone", "--limit 2" not in over)

print()
print("tokenizer passthrough:")
tok = shlex.split(dry_run("--group", "smoke", "--tokenizer", "allenai/dolma2-tokenizer"))[2]
check("tokenizer reaches the runner", "--tokenizer allenai/dolma2-tokenizer" in tok)

print()
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("COMMAND STRING SURVIVES THE PLATFORM'S PARSING")
