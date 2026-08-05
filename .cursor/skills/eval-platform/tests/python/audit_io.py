"""Audit the eval-platform skill: does documentation match behaviour?

The skill now has two entry points, and they are documented differently:

  submit_eval_run.sh   the platform path, and the one SKILL.md tables in full
  run_eval_sweep.sh    the local path, documented by its own --help header

So SKILL.md's flag table is checked against the submitter, and each script's
header is checked against its own parser.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import SKILL

sweep = (SKILL / "scripts/run_eval_sweep.sh").read_text(encoding="utf-8")
submit = (SKILL / "scripts/submit_eval_run.sh").read_text(encoding="utf-8")
skill_md = (SKILL / "SKILL.md").read_text(encoding="utf-8")

problems = []


def report(section):
    print()
    print("=" * 68)
    print(section)
    print("=" * 68)


def flags_of(source: str) -> tuple[set[str], set[str]]:
    """Flags the parser accepts as inputs, and flags it rejects with guidance.

    A case branch whose body exits non-zero is a rejection handler, not an input,
    and must NOT appear in a "flags you can pass" table.
    """
    case_block = source.split("while [[ $# -gt 0 ]]; do")[1].split("esac")[0]
    parsed, rejected = set(), set()
    for branch in case_block.split(";;"):
        if not re.search(r"(?:^|\|)\s*(--[a-z0-9-]+|-h)\)", branch, re.M):
            continue
        found = set(re.findall(r"(--[a-z0-9-]+)\)", branch))
        (rejected if re.search(r"exit [1-9]", branch) else parsed).update(found)
    return parsed - rejected, rejected


def header_flags(source: str) -> set[str]:
    header = source.split("set -euo pipefail")[0]
    return set(re.findall(r"^#\s+(--[a-z][a-z0-9-]*)", header, re.M))


sweep_inputs, sweep_rejected = flags_of(sweep)
submit_inputs, submit_rejected = flags_of(submit)

for label, source, inputs in [
    ("run_eval_sweep.sh", sweep, sweep_inputs),
    ("submit_eval_run.sh", submit, submit_inputs),
]:
    report(f"{label}: header INPUTS block vs argument parser")
    documented = header_flags(source)
    only_doc = documented - inputs - sweep_rejected - submit_rejected
    only_code = inputs - documented - {"--help"}
    for flag in sorted(only_doc):
        print(f"  DOCUMENTED BUT NOT PARSED: {flag}")
        problems.append(f"{label} header documents {flag} but the parser rejects it")
    for flag in sorted(only_code):
        print(f"  PARSED BUT UNDOCUMENTED:   {flag}")
        problems.append(f"{label} parses {flag} but its header omits it")
    if not only_doc and not only_code:
        print(f"  OK  all {len(inputs)} parsed flags documented, none extra")

report("SKILL.md flag table vs the submitter's parser")
md_flags = set(re.findall(r"\|\s*`(--[a-z0-9-]+)[^`]*`", skill_md))
missing_md = submit_inputs - md_flags - {"--help"}
# The sweep's flags are deliberately not tabled in SKILL.md any more; it points at
# --help instead. So only flags belonging to neither script are a real problem.
extra_md = md_flags - submit_inputs - sweep_inputs
for flag in sorted(missing_md):
    print(f"  IN THE SUBMITTER, NOT IN SKILL.md: {flag}")
    problems.append(f"SKILL.md table omits the submitter's {flag}")
for flag in sorted(extra_md):
    print(f"  IN SKILL.md, NOT A REAL FLAG:      {flag}")
    problems.append(f"SKILL.md documents non-existent {flag}")
if not missing_md and not extra_md:
    print(f"  OK  SKILL.md tables all {len(submit_inputs)} submitter flags, none invented")

report("SKILL.md points at --help for the local path rather than tabling it")
print(f"  {'OK  ' if '--help' in skill_md else 'FAIL'} SKILL.md mentions --help for run_eval_sweep.sh")
if "--help" not in skill_md:
    problems.append("SKILL.md no longer tables the sweep's flags and does not point at --help either")

report("promised outputs vs code that produces them")
checks = [
    ("metrics.json", sweep, r'-O "\$\{OUT\}"', "olmo-eval writes it into -O"),
    ("run_provenance.json", sweep, r"run_provenance\.json", "written by write_provenance"),
    ("_READY", sweep, r"_READY", "marker upload"),
    ("_FAILED", sweep, r"_FAILED", "marker upload"),
    ("accuracy.csv", sweep, r"--out-csv", "summarize_accuracy --out-csv"),
    ("accuracy_wide.csv", sweep, r"--out-wide-csv", "summarize_accuracy --out-wide-csv"),
    ("accuracy.json", sweep, r"--out-json", "summarize_accuracy --out-json"),
    ("sweep.log", sweep, r"\$\{LOG\}", "tee'd and uploaded"),
]
for name, source, pattern, how in checks:
    if re.search(pattern, source):
        print(f"  OK   {name:22s} ({how})")
    else:
        print(f"  MISSING {name:22s} -- promised but no code produces it")
        problems.append(f"output {name} promised but not produced")

report("the submitter refuses what cannot work, before dispatching")
guards = [
    ("checkpoint must be in the outputs bucket", r"OUTPUTS_BUCKET\}/teams/"),
    ("team must be one of the eight", r'in_list "\$\{TEAM\}"'),
    ("compute profile must be a GPU one", r'in_list "\$\{COMPUTE_PROFILE\}" "\$\{GPU_PROFILES\}"'),
    ("eval ref must be a full 40-hex sha", r"\[0-9a-f\]\{40\}"),
    ("experiment must be a lower-case slug", r"\^\[a-z0-9\]\+\(-\[a-z0-9\]\+\)\*\$"),
    ("benchmarks resolved through the shared resolver", r"resolve_benchmarks\.py|RESOLVER"),
    ("gh is required only when actually dispatching", r'DRY_RUN.*!=.*"1"'),
]
for label, pattern in guards:
    if re.search(pattern, submit):
        print(f"  OK   {label}")
    else:
        print(f"  MISSING {label}")
        problems.append(f"submitter lacks a guard: {label}")

report("benchmark resolution has exactly one implementation")
sweep_has_inline = "import json, os, sys" in sweep and "unsupported" in sweep
print(f"  {'FAIL' if sweep_has_inline else 'OK  '} the sweep no longer carries its own resolver")
if sweep_has_inline:
    problems.append("run_eval_sweep.sh still has an inline resolver alongside the shared one")
for label, source in [("sweep", sweep), ("submitter", submit)]:
    uses = "resolve_benchmarks.py" in source
    print(f"  {'OK  ' if uses else 'FAIL'} the {label} calls the shared resolver")
    if not uses:
        problems.append(f"the {label} does not use resolve_benchmarks.py")

report("exit codes")
for source_label, source, codes in [
    ("sweep", sweep, ["2", "3"]),
    ("submitter", submit, ["2", "3", "4"]),
]:
    for code in codes:
        if re.search(rf"exit {code}\b", source):
            print(f"  OK   {source_label} exit {code} present")
        else:
            print(f"  MISSING {source_label} exit {code}")
            problems.append(f"{source_label} documents exit {code} but never uses it")

report("rejected flags stay out of the tables but still guide the user")
for flag in sorted(sweep_rejected | submit_rejected):
    absent = flag not in md_flags
    print(f"  {'OK  ' if absent else 'FAIL'} {flag} absent from SKILL.md's flag table")
    if not absent:
        problems.append(f"{flag} is a rejection handler but appears as an input in SKILL.md")

print()
print("=" * 68)
if problems:
    print(f"{len(problems)} MISMATCH(ES):")
    for problem in problems:
        print(f"  - {problem}")
    sys.exit(1)
print("NO MISMATCHES")
