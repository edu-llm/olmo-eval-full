"""eval-direct-gpu's group resolver, including the `exclude` key smoke depends on.

That skill has no resolver script of its own -- its resolution is a heredoc inside
run_eval_sweep.sh -- and e2e.sh drives *this* skill's copy of the sweep, so nothing
else here exercises it. It is tested from this suite rather than from a second one
because there is one suite, not because the code belongs to this skill.

The resolver is extracted from the heredoc and run as-is, so what is checked is the
code that ships, not a restatement of it.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import REPO

SWEEP = REPO / ".cursor/skills/eval-direct-gpu/scripts/run_eval_sweep.sh"
REGISTRY = REPO / ".cursor/skills/eval-direct-gpu/scripts/benchmarks.json"

failures = []


def check(label, cond, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'} {label}" + (f"  -- {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(label)


def extract_resolver():
    """The heredoc'd python between `python3 - <<'PY'` and its terminator."""
    lines = SWEEP.read_text(encoding="utf-8").split("\n")
    start = next(i for i, line in enumerate(lines) if "python3 - <<'PY'" in line)
    end = next(i for i, line in enumerate(lines) if i > start and line.strip() == "PY")
    return "\n".join(lines[start + 1 : end])


RESOLVER = extract_resolver()
print(f"\nextracted {len(RESOLVER.splitlines())} lines of the shipped resolver")


def resolve(group="", benchmarks="", registry=REGISTRY, limit="", allow_any="0"):
    """Run the resolver the way the sweep does, returning (rc, stdout, stderr)."""
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "resolver.py"
        script.write_text(RESOLVER, encoding="utf-8")
        done = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            # A full env dict, because the resolver reads these with os.environ[...] and
            # would raise KeyError on a missing one rather than treating it as empty.
            env={
                "BENCHMARKS": benchmarks,
                "GROUP": group,
                "ALLOW_ANY": allow_any,
                "REGISTRY": str(registry),
                "CLI_LIMIT": limit,
                "SYSTEMROOT": "C:\\Windows",
                "PATH": "",
            },
        )
        return done.returncode, done.stdout, done.stderr


print("\nsmoke is `all` minus `fact_proxy`:")

rc, out, err = resolve(group="smoke")
lines = out.strip().split("\n")
check("resolves without error", rc == 0, f"rc={rc} {err.strip()}")

names = lines[0].split() if lines else []
expected = ["csqa", "hellaswag", "piqa", "socialiqa", "arc_easy", "popqa", "triviaqa"]
check("seven benchmarks", len(names) == 7, f"{len(names)}: {names}")
check("exactly the expected seven", sorted(names) == sorted(expected), str(names))
check("naturalqs is excluded", "naturalqs" not in names, str(names))
check("jeopardy is excluded", "jeopardy" not in names, str(names))

totals = lines[1].split() if len(lines) > 1 else []
check("506 instances / 1005 prompts / 0 unknown", totals == ["506", "1005", "0"], str(totals))
check("a description is emitted", len(lines) > 2 and lines[2].strip() != "", str(lines[2:]))


print("\nsmoke's prompt budget gives each benchmark its own instance cap:")

# The whole point of a budget is that the caps differ, so a bug that emitted one
# uniform cap would still total correctly if the total were all that was checked
# -- 506 instances is not reachable by any single cap, but a future budget's
# might be. The caps themselves are read out of the resolver's fourth line and
# recomputed here from the registry, which is a second implementation of the
# division over the same input rather than a copy of the answer.
registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
bench = registry["benchmarks"]
budget = registry["groups"]["smoke"]["prompt_budget"]
caps = [int(n) for n in lines[3].split()] if len(lines) > 3 else []
check("one cap per benchmark", len(caps) == len(names), f"{len(caps)} caps for {len(names)} names")
check("the caps are not all the same value", len(set(caps)) > 1, str(caps))

share = budget / len(names)
expected_caps = [max(1, min(bench[n]["instances"], round(share / bench[n]["choices"])))
                 for n in names]
check("each cap is the benchmark's even share of the budget, in instances",
      caps == expected_caps, f"{caps} vs {expected_caps}")
# What the budget is actually for: near-equal prompt counts, from unequal
# instance counts. Nothing else in the suite would notice if the division ran
# the wrong way round and made the instances equal instead.
issued = [cap * bench[n]["choices"] for cap, n in zip(caps, names)]
# Rounding to a whole instance can miss the share by at most half an instance,
# which is half a benchmark's choices in prompts. Anything wider means the
# division ran the wrong way round -- equal instances rather than equal prompts
# would put csqa five times over its share.
check("every benchmark issues within half an instance of an even share",
      all(abs(p - share) <= bench[n]["choices"] / 2 + 1e-9 for p, n in zip(issued, names)),
      str(dict(zip(names, issued))))
check("the realised total is near the budget without claiming to hit it",
      abs(sum(issued) - budget) <= len(names), f"{sum(issued)} against a budget of {budget}")
check("no cap exceeds its split", all(c <= bench[n]["instances"] for c, n in zip(caps, names)),
      str([(n, c, bench[n]["instances"]) for c, n in zip(caps, names)]))


print("\nexcluding does not perturb the groups it derives from:")

for group, count in (("all", 9), ("default", 5), ("reasoning", 5), ("factual", 2), ("fact_proxy", 2)):
    rc, out, _ = resolve(group=group)
    got = len(out.strip().split("\n")[0].split()) if rc == 0 else -1
    check(f"{group} is {count}", got == count, f"got {got}")


print("\nthe registry is what decides, not a hardcoded list:")

with tempfile.TemporaryDirectory() as tmp:
    # A tenth benchmark added to `all` must reach smoke with no edit to smoke. This is
    # the whole reason `like`/`exclude` name groups instead of listing benchmarks.
    grown = json.loads(json.dumps(registry))
    grown["benchmarks"]["madeup"] = {
        "instances": 100,
        "choices": 3,
        "split": "test",
        "metrics": ["accuracy"],
        "kind": "mcq",
    }
    grown["groups"]["all"].append("madeup")
    path = Path(tmp) / "grown.json"
    path.write_text(json.dumps(grown), encoding="utf-8")
    rc, out, err = resolve(group="smoke", registry=path)
    names = out.strip().split("\n")[0].split() if rc == 0 else []
    check("a benchmark added to `all` reaches smoke automatically", "madeup" in names, f"{rc} {names}")

    # And one added to fact_proxy drops out of smoke, equally without editing smoke.
    shrunk = json.loads(json.dumps(registry))
    shrunk["groups"]["fact_proxy"].append("piqa")
    path = Path(tmp) / "shrunk.json"
    path.write_text(json.dumps(shrunk), encoding="utf-8")
    rc, out, _ = resolve(group="smoke", registry=path)
    names = out.strip().split("\n")[0].split() if rc == 0 else []
    check("a benchmark added to `fact_proxy` leaves smoke automatically", "piqa" not in names, str(names))

    # A typo in `exclude` has to say so rather than silently excluding nothing, which
    # would quietly widen the smoke set back to nine.
    typo = json.loads(json.dumps(registry))
    typo["groups"]["smoke"]["exclude"] = "fact_proxies"
    path = Path(tmp) / "typo.json"
    path.write_text(json.dumps(typo), encoding="utf-8")
    rc, out, err = resolve(group="smoke", registry=path)
    check("an unknown exclude target is refused", rc != 0, f"rc={rc}")
    check("and the message names it", "fact_proxies" in err, err.strip()[:120])

    # A group excluding itself must be caught by the same guard that catches a `like`
    # cycle, or the recursion never returns.
    cycle = json.loads(json.dumps(registry))
    cycle["groups"]["smoke"]["exclude"] = "smoke"
    path = Path(tmp) / "cycle.json"
    path.write_text(json.dumps(cycle), encoding="utf-8")
    rc, out, err = resolve(group="smoke", registry=path)
    check("a self-exclude is caught rather than hanging", rc != 0, f"rc={rc}")
    check("and reported as inheriting from itself", "itself" in err, err.strip()[:120])


print("\nan explicit --limit overrides the group's budget, uniformly:")

# A caller who names a number should get that number on every benchmark rather
# than a share of a budget they never mentioned, so this also has to flatten the
# caps back out -- 21 instances is reachable both ways, the equal caps are not.
rc, out, _ = resolve(group="smoke", limit="3")
budgeted_lines = out.strip().split("\n") if rc == 0 else []
totals = budgeted_lines[1].split() if len(budgeted_lines) > 1 else []
check("--limit 3 gives 21 instances / 60 prompts", totals[:2] == ["21", "60"], str(totals))
override_caps = budgeted_lines[3].split() if len(budgeted_lines) > 3 else []
check("and puts the same 3 on every benchmark", override_caps == ["3"] * 7, str(override_caps))


print("\nthe budget's edges:")

with tempfile.TemporaryDirectory() as tmp:
    # Two ways to say how big a group is. Preferring one silently would leave the
    # other sitting in the registry as a false statement about what runs.
    both = json.loads(json.dumps(registry))
    both["groups"]["smoke"]["limit"] = 10
    path = Path(tmp) / "both.json"
    path.write_text(json.dumps(both), encoding="utf-8")
    rc, _, err = resolve(group="smoke", registry=path)
    check("limit and prompt_budget together are refused", rc != 0, f"rc={rc}")
    check("and the message names both keys",
          "limit" in err and "prompt_budget" in err, err.strip()[:160])

    # A budget large enough to outrun a split must cap at the split rather than
    # asking olmo-eval for rows that do not exist.
    huge = json.loads(json.dumps(registry))
    huge["groups"]["smoke"]["prompt_budget"] = 10_000_000
    path = Path(tmp) / "huge.json"
    path.write_text(json.dumps(huge), encoding="utf-8")
    rc, out, err = resolve(group="smoke", registry=path)
    hlines = out.strip().split("\n") if rc == 0 else []
    hnames = hlines[0].split() if hlines else []
    hcaps = [int(n) for n in hlines[3].split()] if len(hlines) > 3 else []
    check("an unreachable budget resolves rather than failing", rc == 0, f"rc={rc} {err.strip()}")
    check("and caps every benchmark at its own split size",
          hcaps == [bench[n]["instances"] for n in hnames], str(dict(zip(hnames, hcaps))))


print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    raise SystemExit(1)
print("EVAL-DIRECT-GPU GROUP RESOLUTION HOLDS")
