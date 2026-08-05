"""Where the repo is, for the python half of the suite.

$OLMO_EVAL_ROOT with a relative fallback, which is the same rule
submit_eval_run.sh applies:

    REPO_ROOT="${OLMO_EVAL_ROOT:-$(cd "${SKILL_DIR}/../../.." && pwd)}"

Keeping the two in step matters because several tests invoke the scripts as
subprocesses; if the test resolved the root one way and the script another, a
test could pass against a checkout it was not actually pointed at.

Tests import this by adding the suite root to sys.path, since they are run as
loose scripts rather than as a package:

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from _paths import REPO, SKILL
"""

import os
from pathlib import Path

TESTS = Path(__file__).resolve().parent
SKILL = TESTS.parent
REPO = Path(os.environ.get("OLMO_EVAL_ROOT") or SKILL.parents[2]).resolve()

# Named here rather than rebuilt in each test, so a rename upstream is one edit.
SUBMIT = SKILL / "scripts" / "submit_eval_run.sh"
SWEEP = SKILL / "scripts" / "run_eval_sweep.sh"
SUMMARIZER = SKILL / "scripts" / "summarize_accuracy.py"
RESOLVER = SKILL / "scripts" / "resolve_benchmarks.py"
REGISTRY = SKILL / "scripts" / "benchmarks.json"


def find_bash() -> str | None:
    """A bash that can run the shell half, or None if there is none.

    Git Bash and MSYS2 are tried by their usual install locations before PATH,
    because on Windows the `bash.exe` in System32 is the WSL launcher: it starts
    a Linux VM with its own filesystem, where the repo path handed to it does
    not exist. Preferring it would produce failures that look like the tests are
    broken rather than like the wrong interpreter was chosen.
    """
    import shutil

    candidates = [
        r"C:\Program Files\Git\bin\bash.exe",
        r"C:\Program Files (x86)\Git\bin\bash.exe",
        r"C:\msys64\usr\bin\bash.exe",
    ]
    for path in candidates:
        if Path(path).is_file():
            return path
    found = shutil.which("bash")
    if found and "system32" not in found.lower():
        return found
    return None
