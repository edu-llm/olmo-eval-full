"""The `:mc` variants, and the colon in their names.

Every other benchmark the registry offers is a bare identifier, so the three
multiple-choice-prompting variants are the first names carrying a character with
meaning somewhere. A colon is inert in all four places these names travel --
registry lookup, a `bash -lc` command string, olmo-eval's `-t`, and a CSV header
-- but "inert" is a claim about four separate pieces of code, and the cost of it
being wrong is a submission that is approved and then fails, or a column that
silently overwrites another. So each hop is asserted rather than assumed.

The rest is the arithmetic behind the cost estimate. A `:mc` variant inherits its
split, limit and metrics from the base task and overrides only the formatter, so
the registry's numbers for it are the base task's numbers -- which is true today
and would stop being true the moment someone added a `limit=` upstream. That is
checked against the real `register_variant` call rather than restated here.

The submitter is a bash script; where there is no bash that one section says so
and the rest still runs, since the other three hops need nothing but python.
"""

import argparse
import ast
import csv
import importlib.util
import json
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import REGISTRY, REPO, RESOLVER, SUBMIT, SUMMARIZER, find_bash

MC = ["arc_easy:mc", "csqa:mc", "socialiqa:mc"]
NOT_REGISTERED = ["hellaswag:mc", "piqa:mc"]

failures = []


def check(label, cond, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'} {label}" + (f"  -- {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(label)


def load(name, path):
    """Import a shipped script by path; they are scripts, not an installed package."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


resolver = load("resolver_under_test", RESOLVER)
run_eval = load("run_eval_under_test", REPO / "src/olmo_eval/platform/run_eval.py")
reg = json.loads(REGISTRY.read_text(encoding="utf-8"))


def resolve(**kwargs):
    return resolver.resolve(
        reg,
        **{"group": None, "benchmarks": None, "cli_limit": None, "allow_any_task": False, **kwargs},
    )


print("1. the three names resolve, as a group and as an explicit list")
grouped = resolve(group="mc_format")
check("the group expands to exactly the three", grouped["benchmarks"] == MC, str(grouped["benchmarks"]))
check("the group names itself in the source line", "mc_format" in grouped["source"], grouped["source"])
check("its description survives into the source line",
      "multiple choice prompting" in grouped["source"], grouped["source"])
explicit = resolve(benchmarks=" ".join(MC))
check("an explicit list of the three resolves", explicit["benchmarks"] == MC, str(explicit["benchmarks"]))
check("no name is reported unknown", explicit["unknown"] == [], str(explicit["unknown"]))
check("the estimate is the sum of the three entries",
      grouped["instances"] == sum(reg["benchmarks"][name]["instances"] for name in MC)
      and grouped["prompts"] == sum(reg["benchmarks"][name]["instances"] * reg["benchmarks"][name]["choices"]
                                    for name in MC),
      f"{grouped['instances']} instances / {grouped['prompts']} prompts")

print()
print("2. hellaswag:mc and piqa:mc are in no group and are not runnable")


def members(name):
    return resolver.group_members(reg["groups"], name)


for name in NOT_REGISTERED:
    holding = [group for group in reg["groups"] if name in members(group)]
    check(f"{name} is in no group", not holding, str(holding))
    check(f"{name} is not a registry benchmark", name not in reg["benchmarks"])
    try:
        resolve(benchmarks=name)
        check(f"{name} is refused", False, "it resolved")
    except resolver.Refused as refusal:
        check(f"{name} is refused, by name", any(name in line for line in refusal.lines))

print()
print("3. the three stay out of the sets a normal sweep runs")
for group in ("default", "reasoning", "all", "smoke"):
    overlap = [name for name in MC if name in members(group)]
    check(f"group '{group}' is unchanged by them", not overlap, str(overlap))
check("they are reachable from exactly one group",
      [group for group in reg["groups"] if set(MC) <= set(members(group))] == ["mc_format"])
# `all` used to be every benchmark the registry defined. It stopped being true on
# purpose: these three re-ask benchmarks `all` already contains, so folding them
# in would score three benchmarks twice and quietly raise what a full sweep costs.
# The property worth keeping is the one that identity was standing in for -- that
# nothing is defined and then left unreachable. This is its only home; shell/e2e.sh
# used to assert it too and no longer does, because a copy there would have to
# reimplement `like` inheritance instead of calling group_members.
orphans = [name for name in reg["benchmarks"]
           if not any(name in members(group) for group in reg["groups"])]
check("every registry benchmark belongs to some group", not orphans, str(orphans))
check("'all' is every benchmark that is not a variant of another",
      set(members("all")) == {name for name in reg["benchmarks"] if ":" not in name},
      str(sorted(set(reg["benchmarks"]) - set(members("all")))))

# The refusal message is the only place a caller learns what it could have
# asked for, so a name the registry offers but the error omits is a name nobody
# finds. Checked as a set rather than a literal prefix, which sorting moves.
try:
    resolve(benchmarks="commonsense_qa")
    check("an unknown name is refused", False, "it resolved")
except resolver.Refused as refusal:
    listed = next((line for line in refusal.lines if line.startswith("known: ")), "")
    check("the refusal lists every runnable name, variants included",
          set(listed.removeprefix("known: ").split()) == set(reg["benchmarks"]), listed)

print()
print("4. the registry's numbers are the base task's, because the variant inherits them")
# What `register_variant("x", "mc", ...)` actually overrides upstream. Anything
# beyond `formatter` would change the split, the limit or the metric key, and
# the registry entry beside it would then be describing a different run.
variant_kwargs = {}
for path in (REPO / "src/olmo_eval/evals/tasks").rglob("*.py"):
    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and getattr(node.func, "id", "") == "register_variant"
                and len(node.args) >= 2
                and all(isinstance(arg, ast.Constant) for arg in node.args[:2])):
            variant_kwargs[f"{node.args[0].value}:{node.args[1].value}"] = {
                kw.arg for kw in node.keywords
            }

for name in MC:
    base = name.split(":", 1)[0]
    check(f"{name} is registered upstream", name in variant_kwargs)
    check(f"{name} overrides only the formatter", variant_kwargs.get(name) == {"formatter"},
          str(sorted(variant_kwargs.get(name, ()))))
    entry, base_entry = reg["benchmarks"][name], reg["benchmarks"][base]
    check(f"{name} scores the same split as {base}", entry["split"] == base_entry["split"],
          f"{entry['split']} vs {base_entry['split']}")
    check(f"{name} scores the same instances as {base}",
          entry["instances"] == base_entry["instances"],
          f"{entry['instances']} vs {base_entry['instances']}")
    # One request per option either way: the continuations become single letters
    # rather than fewer of them. Equal counts, and deliberately not equal cost --
    # an mc request carries the whole option list in its prompt and repeats it per
    # option, which without prefix caching is several times the tokens. The
    # registry counts requests; BENCHMARKS.md carries the caveat.
    check(f"{name} issues one prompt per option, as {base} does",
          entry["choices"] == base_entry["choices"], f"{entry['choices']} vs {base_entry['choices']}")
    check(f"{name} reports the same metric key as {base}",
          entry["metrics"] == base_entry["metrics"] == ["accuracy"], str(entry["metrics"]))

# socialiqa:mc used to be flagged limit_unsafe, because it shares socialiqa's
# loader and that loader sampled validation and train together whenever a limit was
# set. The loader now samples the split it scores, so the base task is not flagged
# and neither is the variant -- the inheritance is the same fact, read the other
# way. The resolver is still asked rather than assumed silent, and then asked again
# over a doctored registry, so an empty table is distinguished from a broken path.
check("socialiqa:mc is not flagged, because its base task no longer samples outside its split",
      "socialiqa:mc" not in reg.get("limit_unsafe", {}))
check("and the resolver reports nothing flagged for it under a limit",
      resolve(benchmarks="socialiqa:mc", cli_limit=2)["limit_unsafe"] == {})
doctored = json.loads(json.dumps(reg))
doctored["limit_unsafe"] = {"socialiqa:mc": "a reason invented by this test"}
check("a flag put back would still reach the resolver's output",
      "socialiqa:mc" in resolver.resolve(
          doctored, group=None, benchmarks="socialiqa:mc", cli_limit=2, allow_any_task=False
      )["limit_unsafe"])

print()
print("5. the colon survives olmo-eval's argv")
plan = run_eval.build_plan(
    argparse.Namespace(
        checkpoint="s3://bkt/teams/t/runs/r/checkpoints/step2000/",
        benchmarks=" ".join(MC), limit=None, provider="olmo_core", tokenizer=None,
        output_prefix="s3://b/teams/t/runs/r/", local_out="/tmp/out", override=None, dry_run=True,
    ),
    {},
)
check("the names arrive whole", list(plan.benchmarks) == MC, str(plan.benchmarks))
argv = plan.command
check("one -t per name, colon intact",
      [argv[i + 1] for i, a in enumerate(argv) if a == "-t"] == MC,
      str([argv[i + 1] for i, a in enumerate(argv) if a == "-t"]))
check("shlex.join needs no quoting for these names",
      all(f"-t {name}" in shlex.join(argv) for name in MC), shlex.join(argv))

print()
print("6. the colon survives the submitted command string")
bash = find_bash()
if bash is None:
    print("  SKIPPED  no bash; the submitter is a bash script. The other hops above ran.")
else:
    sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                         capture_output=True, text=True, check=True).stdout.strip()
    out = subprocess.run(
        [bash, str(SUBMIT),
         "--checkpoint", "s3://sbsandbox-intern-edullm-outputs/teams/pre-training/runs/r/checkpoints/step2000",
         "--team", "pre-training", "--experiment", "mc-crosscheck",
         "--wandb-project", "edullm-evals", "--eval-ref", sha,
         "--group", "mc_format", "--dry-run"],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        raise SystemExit(f"dry run failed:\n{out.stdout}\n{out.stderr}")
    lines = out.stdout.splitlines()
    command = lines[next(i for i, line in enumerate(lines) if line.strip() == "command:") + 1].strip()
    # Split as the platform's compile step does, then as bash does inside the
    # container. A colon that broke quoting would show up as a token count.
    outer = shlex.split(command)
    check("the platform's split still yields three tokens", len(outer) == 3, str(len(outer)))
    inner = shlex.split(outer[2])
    passed = inner[inner.index("--benchmarks") + 1]
    check("the three names reach --benchmarks as ONE argument", passed == " ".join(MC), passed)
    check("and each name is intact inside it", passed.split() == MC, str(passed.split()))
    check("the estimate the reviewer sees is the group's",
          f"~{grouped['instances']} instances / ~{grouped['prompts']} prompts" in out.stdout)

print()
print("7. the summarizer gives a variant its own column, beside the base task's")
root = Path(tempfile.mkdtemp()) / "runs"
(root / "step2000").mkdir(parents=True)
(root / "step2000" / "run_provenance.json").write_text(
    json.dumps({"checkpoint": "s3://b/step2000", "status": "ok", "benchmarks": ["arc_easy", *MC]}),
    encoding="utf-8",
)
(root / "step2000" / "metrics.json").write_text(json.dumps({
    "timestamp": "t", "config": {}, "summary": {},
    "tasks": [
        {"task": "arc_easy", "num_instances": 2376, "primary_metric": "accuracy:logprob",
         "metrics": {"accuracy": {"logprob": 0.44}}},
        # Same metric name and same scorer as the base task, on a different
        # benchmark -- so the benchmark name is the only thing keeping the two
        # columns apart.
        {"task": "arc_easy:mc", "num_instances": 2376, "primary_metric": "accuracy:logprob",
         "metrics": {"accuracy": {"logprob": 0.26}}},
    ],
}), encoding="utf-8")

wide_csv, long_csv = root.parent / "accuracy_wide.csv", root.parent / "accuracy.csv"
subprocess.run([sys.executable, str(SUMMARIZER), "--runs-dir", str(root),
                "--out-wide-csv", str(wide_csv), "--out-csv", str(long_csv)],
               capture_output=True, text=True, check=True)

row = list(csv.DictReader(wide_csv.open(encoding="utf-8")))[0]
header = [h for h in row if h not in ("run_id", "step", "checkpoint", "status")]
print("  wide columns:", header)
check("the variant keeps its full name as the column", "arc_easy:mc" in header, str(header))
check("the base task keeps its bare column", "arc_easy" in header, str(header))
check("neither is qualified further -- one metric, one scorer each",
      not any(h.startswith(("arc_easy.", "arc_easy:mc.")) for h in header), str(header))
check("the two scores stay distinct", (row["arc_easy"], row["arc_easy:mc"]) == ("0.44", "0.26"),
      f"{row['arc_easy']} / {row['arc_easy:mc']}")
check("the header round-trips through csv unchanged",
      next(csv.reader(wide_csv.open(encoding="utf-8"))) == list(row), str(list(row)))
long_rows = list(csv.DictReader(long_csv.open(encoding="utf-8")))
check("the long CSV carries the variant under its own benchmark name",
      [r["benchmark"] for r in long_rows] == ["arc_easy", "arc_easy:mc"],
      str([r["benchmark"] for r in long_rows]))
check("the colon in the benchmark name does not confuse the primary flag",
      all(r["is_primary"] == "True" for r in long_rows),
      str([(r["benchmark"], r["is_primary"]) for r in long_rows]))

print()
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("MC VARIANTS RESOLVE, TRAVEL AND REPORT INTACT")
