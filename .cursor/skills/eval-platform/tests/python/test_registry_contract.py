"""Every name the skill accepts must be a real olmo-eval task.

The skill validates benchmark names against benchmarks.json *before* downloading a
checkpoint. That check is only worth anything if the registry's names actually
exist upstream -- otherwise validation passes and `olmo-eval run` fails later,
after the expensive fetch and conversion the check exists to protect.
"""

import ast
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import REGISTRY, REPO, SKILL as skill

reg = json.loads(REGISTRY.read_text(encoding="utf-8"))
tasks_dir = REPO / "src/olmo_eval/evals/tasks"

# Real task names: @register("x") plus register_variant("base", "suffix") -> base:suffix
real = set()
variants = set()
for py in tasks_dir.rglob("*.py"):
    text = py.read_text(encoding="utf-8", errors="replace")
    real |= set(re.findall(r'@register\(\s*["\']([\w.:-]+)["\']', text))
    for base, suffix in re.findall(
        r'register_variant\(\s*["\']([\w.:-]+)["\']\s*,\s*["\']([\w.:-]+)["\']', text
    ):
        variants.add(f"{base}:{suffix}")

print(f"discovered {len(real)} registered tasks, {len(variants)} variants")

failures = []


def check(label, cond, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'} {label}" + (f"  -- {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(label)


print()
print("1. every benchmark in the registry is a real olmo-eval task")
for name in sorted(reg["benchmarks"]):
    check(f"{name} is registered upstream", name in real or name in variants)

print()
print("2. every group member is defined in the benchmarks block")
known = set(reg["benchmarks"])


def members_of(name, seen=()):
    """Mirror the resolver's group_members(), including 'like' inheritance."""
    assert name not in seen, f"cycle: {seen} -> {name}"
    spec = reg["groups"][name]
    if isinstance(spec, list):
        return list(spec)
    if "benchmarks" in spec:
        return list(spec["benchmarks"])
    return members_of(spec["like"], seen + (name,))


for group in reg["groups"]:
    missing = [m for m in members_of(group) if m not in known]
    check(f"group '{group}' has no undefined members", not missing, str(missing))

print()
print("2b. object-form groups are well-formed")
for group, spec in reg["groups"].items():
    if isinstance(spec, list):
        continue
    check(f"group '{group}' defines members via 'benchmarks' or 'like'",
          ("benchmarks" in spec) ^ ("like" in spec), str(sorted(spec)))
    if "like" in spec:
        check(f"group '{group}' inherits an existing group", spec["like"] in reg["groups"],
              spec["like"])
    if "limit" in spec:
        lim = spec["limit"]
        check(f"group '{group}' limit is a positive int",
              isinstance(lim, int) and not isinstance(lim, bool) and lim >= 1, repr(lim))

print()
print("2c. the smoke group is the default set, capped")
smoke = reg["groups"].get("smoke")
check("smoke group exists", smoke is not None)
check("smoke inherits default rather than duplicating it",
      isinstance(smoke, dict) and smoke.get("like") == "default", repr(smoke))
check("smoke caps at 2 instances", isinstance(smoke, dict) and smoke.get("limit") == 2)
check("smoke resolves to the same benchmarks as default",
      members_of("smoke") == members_of("default"),
      f"{members_of('smoke')} vs {members_of('default')}")

print()
print("3. 'default' group exists (it is what runs with no flags)")
check("default group present", "default" in reg["groups"])

print()
print("4. no alias shadows a real task name")
for alias, target in sorted(reg["aliases"].items()):
    check(f"alias '{alias}' is not itself a runnable task",
          alias not in real and alias not in known,
          f"'{alias}' IS real -- would be wrongly rejected")
    check(f"alias '{alias}' points at a defined benchmark", target in known, target)

print()
print("4b. the unsupported table is honest in both directions")
unsupported = reg.get("unsupported", {})
check("unsupported table exists", bool(unsupported))
for name, reason in sorted(unsupported.items()):
    # A name here must NOT also be offered as runnable, or the two tables fight.
    check(f"'{name}' is not also a runnable registry benchmark", name not in known)
    check(f"'{name}' gives a reason", isinstance(reason, str) and len(reason) > 20)
# Anything claimed to have "no task file" must really have none upstream.
for name, reason in sorted(unsupported.items()):
    if "no task file" in reason:
        check(f"'{name}' genuinely has no @register upstream",
              name not in real and name not in variants,
              "registry says no task file, but it is registered")
    else:
        check(f"'{name}' exists upstream, as its reason implies",
              name in real or name in variants, "reason implies it exists but it does not")

print()
print("5. limit_unsafe agrees with what the loaders actually do when limited")
# The table is empty today, so "every entry is defined" would assert nothing. What
# it stood for is the property worth pinning: a task that reads rows outside the
# split it scores once a limit is set must be flagged, and one that does not must
# not be. hellaswag and socialiqa each carry a `limit_reads_all_splits` class flag
# saying which they are, and a ':mc' variant shares its base task's loader, so the
# base task's answer is the variant's answer too.
unsafe = reg.get("limit_unsafe", {})
for name in unsafe:
    check(f"limit_unsafe '{name}' is defined", name in known)

reads_union: dict[str, bool] = {}
ungated: dict[str, list[str]] = {}
for py in tasks_dir.rglob("*.py"):
    tree = ast.parse(py.read_text(encoding="utf-8", errors="replace"))
    in_file = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "splits" for t in node.targets
        ):
            dumped = ast.dump(node.value)
            if "attr='limit'" in dumped and "attr='limit_reads_all_splits'" not in dumped:
                ungated.setdefault(py.name, [])
        if not isinstance(node, ast.ClassDef):
            continue
        registered = [
            d.args[0].value
            for d in node.decorator_list
            if isinstance(d, ast.Call)
            and getattr(d.func, "id", "") == "register"
            and d.args
            and isinstance(d.args[0], ast.Constant)
        ]
        in_file += registered
        for stmt in node.body:
            if (
                isinstance(stmt, ast.Assign)
                and any(getattr(t, "id", None) == "limit_reads_all_splits" for t in stmt.targets)
                and isinstance(stmt.value, ast.Constant)
            ):
                for task_name in registered:
                    reads_union[task_name] = bool(stmt.value.value)
    if py.name in ungated:
        ungated[py.name] = in_file

check("the flag is declared by the tasks that had the behaviour",
      set(reads_union) == {"hellaswag", "socialiqa"}, str(sorted(reads_union)))
# A `splits` list that varies with the limit and is not gated on the flag is the old
# behaviour, in a task the flag cannot describe. That is a failure only for a task
# this registry offers -- upstream has others, and they are reported rather than
# silently tolerated so that adding one to the registry cannot pass unnoticed.
for filename, task_names in sorted(ungated.items()):
    offered = [n for n in task_names if n in {b.split(":", 1)[0] for b in known}]
    if offered:
        check(f"{filename} gates its limited split choice on the flag", False,
              "offered by this registry as " + " ".join(offered))
    else:
        print(f"  note {filename} still samples across splits when limited "
              f"({' '.join(task_names) or 'no @register'}), and this registry does not offer it")

for name in sorted(known):
    base = name.split(":", 1)[0]
    if base not in reads_union:
        continue
    should_flag = reads_union[base]
    check(f"'{name}' is flagged exactly when {base} samples outside its split",
          (name in unsafe) == should_flag,
          f"flag={should_flag}, listed={name in unsafe}")

print()
print("6. cost-estimate fields are present and sane")
for name, e in sorted(reg["benchmarks"].items()):
    ok = (isinstance(e.get("instances"), int) and e["instances"] > 0
          and isinstance(e.get("choices"), int) and e["choices"] >= 1
          and e.get("kind") in {"mcq", "generative"}
          and isinstance(e.get("metrics"), list) and e["metrics"])
    check(f"{name} has usable instances/choices/kind/metrics", ok, json.dumps(e))
    if e.get("kind") == "generative":
        check(f"{name} (generative) issues 1 prompt per instance", e["choices"] == 1,
              f"choices={e['choices']}")

print()
print("7. BENCHMARKS.md documents exactly the registry's benchmarks")
md = (skill / "BENCHMARKS.md").read_text(encoding="utf-8")
for name in sorted(known):
    check(f"BENCHMARKS.md mentions {name}", re.search(rf"\b{re.escape(name)}\b", md) is not None)
for group in sorted(reg["groups"]):
    check(f"BENCHMARKS.md mentions group '{group}'",
          re.search(rf"\b{re.escape(group)}\b", md) is not None)

print()
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("REGISTRY CONTRACTS HOLD")
