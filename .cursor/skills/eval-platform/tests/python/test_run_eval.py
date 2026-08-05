"""Exercise olmo_eval.platform.run_eval's real source, without installing olmo-eval.

Loads the module directly by path so the rest of the package (and torch) is not
imported. Checks the argv the platform will actually exec, and the refusals that
are supposed to happen before anything is spent.
"""

import argparse
import importlib.util
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import REPO

target = REPO / "src/olmo_eval/platform/run_eval.py"

spec = importlib.util.spec_from_file_location("run_eval_under_test", target)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
# Registered before executing: `from __future__ import annotations` makes every
# annotation a string, and @dataclass resolves them through sys.modules.
sys.modules["run_eval_under_test"] = mod
spec.loader.exec_module(mod)

failures = []


def check(label, cond, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'} {label}" + (f"  -- {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(label)


def ns(**kw):
    base = dict(
        checkpoint="s3://bkt/teams/t/runs/r/checkpoints/step2000/",
        benchmarks="csqa hellaswag piqa socialiqa arc_easy",
        limit=2,
        provider="olmo_core",
        tokenizer=None,
        output_prefix=None,
        local_out="/tmp/out",
        override=None,
        dry_run=False,
    )
    base.update(kw)
    return argparse.Namespace(**base)


PLATFORM = {
    "EDULLM_OUTPUT_PREFIX": "s3://sbsandbox-intern-edullm-outputs/teams/pre-training/runs/run_019f/",
    "EDULLM_RUN_ID": "run_019f",
    "EDULLM_TEAM": "pre-training",
    "EDULLM_COMMIT_SHA": "a" * 40,
    "WANDB_PROJECT": "edullm-evals",
}

print("plan resolution:")
plan = mod.build_plan(ns(), dict(PLATFORM))
check("output prefix taken from the platform env", plan.output_prefix == PLATFORM["EDULLM_OUTPUT_PREFIX"])
check("benchmarks split into a tuple",
      plan.benchmarks == ("csqa", "hellaswag", "piqa", "socialiqa", "arc_easy"), str(plan.benchmarks))
check("comma separated names also work",
      mod.build_plan(ns(benchmarks="csqa,hellaswag"), dict(PLATFORM)).benchmarks == ("csqa", "hellaswag"))
check("trailing slash stripped from the checkpoint",
      plan.checkpoint == "s3://bkt/teams/t/runs/r/checkpoints/step2000", plan.checkpoint)
check("only populated platform vars are recorded",
      set(plan.environment) == set(PLATFORM), str(sorted(plan.environment)))

print()
print("refusals that must happen before anything is spent:")
for label, kwargs, env in [
    ("no output prefix anywhere", {}, {}),
    ("output prefix that is not s3://", {"output_prefix": "/local/dir"}, {}),
    ("empty benchmark list", {"benchmarks": "   "}, PLATFORM),
    ("zero limit", {"limit": 0}, PLATFORM),
    ("negative limit", {"limit": -1}, PLATFORM),
]:
    try:
        mod.build_plan(ns(**kwargs), dict(env))
        check(f"refuses: {label}", False, "it was accepted")
    except mod.RunEvalError:
        check(f"refuses: {label}", True)

check("a prefix missing its trailing slash gets one",
      mod.build_plan(ns(output_prefix="s3://b/teams/t/runs/r"), {}).output_prefix
      == "s3://b/teams/t/runs/r/")

print()
print("the argv the platform will exec:")
argv = plan.command
print("   ", shlex.join(argv))
check("first word is the olmo-eval entry point", argv[0] == "olmo-eval" and argv[1] == "run")
check("model is the s3 uri, passed straight through", argv[argv.index("-m") + 1] == plan.checkpoint)
check("provider override sits in the harness group, before the first task",
      argv.index("provider.kind=olmo_core") < argv.index("-t"),
      "a provider key after -t is a usage error")
check("one -t per benchmark", argv.count("-t") == 5)
check("one limit override per task", argv.count("limit=2") == 5)
check("every limit follows its own task",
      all(argv[i - 1] == "-o" and argv[i - 2] == b
          for b in plan.benchmarks
          for i in [argv.index(f"limit=2", argv.index(b))]),
      "limit must bind to the preceding -t")
# Compared through Path so the assertion holds on Windows too; the container is
# Linux, where this is the literal /tmp/out.
check("output dir passed with -O",
      argv[argv.index("-O") + 1] == str(Path("/tmp/out")),
      argv[argv.index("-O") + 1])

print()
print("optional flags:")
tok = mod.build_plan(ns(tokenizer="allenai/dolma2-tokenizer"), dict(PLATFORM)).command
check("tokenizer becomes a provider override",
      "provider.tokenizer=allenai/dolma2-tokenizer" in tok)
check("tokenizer override precedes the first task",
      tok.index("provider.tokenizer=allenai/dolma2-tokenizer") < tok.index("-t"))

ovr = mod.build_plan(ns(override=["provider.kwargs.a=1", "provider.kwargs.b=2"]), dict(PLATFORM)).command
check("each extra override gets its own -o",
      ovr.count("-o") == 1 + 2 + 5, f"-o count was {ovr.count('-o')}")
check("extra overrides land before the first task",
      max(ovr.index("provider.kwargs.a=1"), ovr.index("provider.kwargs.b=2")) < ovr.index("-t"))

nolimit = mod.build_plan(ns(limit=None), dict(PLATFORM)).command
check("no limit means no limit override", not any(a.startswith("limit=") for a in nolimit))

print()
print("s3 uri parsing:")
check("bucket and key split", mod._parse_s3_uri("s3://b/teams/t/runs/r/") == ("b", "teams/t/runs/r/"))
check("bucket with no key", mod._parse_s3_uri("s3://b") == ("b", ""))
for bad in ["/local/path", "https://x/y", "s3://"]:
    try:
        mod._parse_s3_uri(bad)
        check(f"rejects {bad!r}", False)
    except mod.RunEvalError:
        check(f"rejects {bad!r}", True)

print()
print("provenance:")
doc = mod.provenance(plan, "started")
check("records the resolved command", doc["olmo_eval_command"] == plan.command)
check("records status", doc["status"] == "started")
check("carries the platform environment for traceability",
      doc["platform_environment"]["EDULLM_RUN_ID"] == "run_019f")
check("records the eval's own version", "olmo_eval_version" in doc)
final = mod.provenance(plan, "ok", exit_code=0, objects_uploaded=12, upload_error=None)
check("terminal record carries exit code and object count",
      final["exit_code"] == 0 and final["objects_uploaded"] == 12)
check("provenance is json serialisable", isinstance(__import__("json").dumps(final), str))

print()
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("RUN_EVAL CONTRACTS HOLD")
