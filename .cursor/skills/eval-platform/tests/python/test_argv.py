"""Validate the argv the sweep script emits against olmo-eval's real parser.

Loads cli/utils.py in isolation with click/console stubbed, since the full
package import needs numpy which is absent here. The parsing logic under test is
pure python and untouched by the stubbing.
"""

import dataclasses
import importlib.util
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import REPO

# --- stub the two imports cli/utils.py needs that we cannot satisfy ----------
click = types.ModuleType("click")


class UsageError(Exception):
    pass


class Option:
    def __init__(self, *a, **k):
        k.pop("save_to", None)


click.UsageError = UsageError
click.Option = Option
sys.modules["click"] = click

console_mod = types.ModuleType("olmo_eval.common.console")
console_mod.console = types.SimpleNamespace(print=lambda *a, **k: None)
sys.modules["olmo_eval.common.console"] = console_mod

# olmo_eval.common.types is imported but only used for annotations here.
pkg = types.ModuleType("olmo_eval")
pkg.__path__ = [str(REPO / "src/olmo_eval")]
sys.modules.setdefault("olmo_eval", pkg)
common = types.ModuleType("olmo_eval.common")
common.__path__ = [str(REPO / "src/olmo_eval/common")]
sys.modules.setdefault("olmo_eval.common", common)
types_mod = types.ModuleType("olmo_eval.common.types")


# process_ordered_args validates task overrides against TaskConfig fields (a real
# constant in utils.py) and SamplingParams fields, so SamplingParams needs to be a
# real dataclass. Its field names only matter for rejecting unknown keys.
@dataclasses.dataclass
class SamplingParams:
    max_tokens: int = 0
    temperature: float = 0.0
    top_p: float = 1.0
    stop_sequences: tuple = ()


types_mod.SamplingParams = SamplingParams
sys.modules.setdefault("olmo_eval.common.types", types_mod)

spec = importlib.util.spec_from_file_location(
    "oe_cli_utils", REPO / "src/olmo_eval/cli/utils.py"
)
utils = importlib.util.module_from_spec(spec)
# Must be registered before exec: a @dataclass in this module resolves
# cls.__module__ through sys.modules during class creation.
sys.modules["oe_cli_utils"] = utils
spec.loader.exec_module(utils)
print("loaded the real cli/utils.py:", hasattr(utils, "process_ordered_args"))
print("  real TASK_CONFIG_FIELDS has 'limit':", "limit" in utils.TASK_CONFIG_FIELDS)
print("  real HARNESS_CONFIG_FIELDS has 'provider':", "provider" in utils.HARNESS_CONFIG_FIELDS)


def parse(argv):
    ordered = utils.reconstruct_ordered_args(argv)
    return utils.process_ordered_args(ordered)


BENCHES = ["hellaswag", "piqa", "arc_easy", "csqa", "socialiqa", "naturalqs", "jeopardy"]

failures = []


def check(label, cond, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'} {label}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        failures.append(label)


def build(tp="1", limit=None, benches=BENCHES, gpu_mem=None):
    """Mirror exactly what run_eval_sweep.sh assembles."""
    args = ["--harness", "default", "-o", "provider.kind=vllm_server"]
    if tp != "1":
        args += ["-o", f"provider.kwargs.tensor_parallel_size={tp}"]
    if gpu_mem is not None:
        args += ["-o", f"provider.kwargs.gpu_memory_utilization={gpu_mem}"]
    for b in benches:
        args += ["-t", b]
        if limit is not None:
            args += ["-o", f"limit={limit}"]
    return ["-m", "/tmp/ck", *args, "-O", "/tmp/out"]


print()
print("=" * 70)
print("CASE 1: default, tp=1, no limit")
print("=" * 70)
argv = build()
print("  argv:", " ".join(argv))
task_ov, harness_ov = parse(argv)
check("provider.kind lands in harness overrides",
      "provider.kind=vllm_server" in harness_ov, str(harness_ov))
check("no task overrides", not any(task_ov.values()), str(dict(task_ov)))
check("all 7 tasks seen", set(task_ov) >= set(BENCHES) or not task_ov,
      str(sorted(task_ov)))

print()
print("=" * 70)
print("CASE 2: tp=2 -- the binding that broke the predecessor")
print("=" * 70)
argv = build(tp="2")
print("  argv:", " ".join(argv))
task_ov, harness_ov = parse(argv)
check("tensor_parallel_size goes to HARNESS, not a task",
      "provider.kwargs.tensor_parallel_size=2" in harness_ov, str(harness_ov))
check("no provider key leaked into task overrides",
      not any("provider" in v for vals in task_ov.values() for v in vals),
      str(dict(task_ov)))
check("uses provider.kwargs.* not provider.tensor_parallel_size",
      not any("provider.tensor_parallel_size" in v for v in harness_ov))

print()
print("=" * 70)
print("CASE 3: limit -- must bind to each task individually")
print("=" * 70)
argv = build(limit=200)
print("  argv:", " ".join(argv))
task_ov, harness_ov = parse(argv)
check("every benchmark got its own limit",
      all(task_ov.get(b) == ["limit=200"] for b in BENCHES),
      str({k: v for k, v in list(task_ov.items())[:3]}))
check("limit did not leak into harness overrides",
      not any("limit" in v for v in harness_ov), str(harness_ov))

print()
print("=" * 70)
print("CASE 4: tp + limit together")
print("=" * 70)
argv = build(tp="4", limit=50)
print("  argv:", " ".join(argv))
task_ov, harness_ov = parse(argv)
check("harness has kind + tp only",
      sorted(harness_ov) == ["provider.kind=vllm_server",
                             "provider.kwargs.tensor_parallel_size=4"],
      str(sorted(harness_ov)))
check("each task has limit only",
      all(task_ov[b] == ["limit=50"] for b in BENCHES))

print()
print("=" * 70)
print("CASE 5: single benchmark subset")
print("=" * 70)
argv = build(benches=["hellaswag"])
print("  argv:", " ".join(argv))
task_ov, harness_ov = parse(argv)
check("subset parses", "provider.kind=vllm_server" in harness_ov)

print()
print("=" * 70)
print("CASE 6: --gpu-memory-utilization, alone and with tp")
print("=" * 70)
argv = build(gpu_mem="0.4")
print("  argv:", " ".join(argv))
task_ov, harness_ov = parse(argv)
check("gpu_memory_utilization lands in HARNESS overrides",
      "provider.kwargs.gpu_memory_utilization=0.4" in harness_ov, str(harness_ov))
check("no provider key leaked into a task",
      not any("provider" in v for vals in task_ov.values() for v in vals))
argv = build(tp="2", gpu_mem="0.35", limit=100)
print("  argv:", " ".join(argv))
task_ov, harness_ov = parse(argv)
check("tp + gpu-mem + limit all bind correctly",
      sorted(harness_ov) == ["provider.kind=vllm_server",
                             "provider.kwargs.gpu_memory_utilization=0.35",
                             "provider.kwargs.tensor_parallel_size=2"]
      and all(task_ov[b] == ["limit=100"] for b in BENCHES),
      str(sorted(harness_ov)))

print()
print("=" * 70)
print("CASE 7: the native provider's two invocations")
print("=" * 70)
# eval-direct-gpu's default path. batch_size travels the same provider.kwargs route as
# tensor_parallel_size, and ProviderConfig.from_dict drops keys it does not recognise
# without complaining, so a wrong route here would look honoured and silently leave
# batch_size None -- which _iter_chunks reads as "one chunk holding everything".
MCQ = ["hellaswag", "piqa", "arc_easy", "csqa", "socialiqa"]
GEN = ["naturalqs", "jeopardy"]


def build_native(batch_size, benches, limit=None):
    """Mirror build_args() on the olmo_core path, for one half of the run."""
    args = ["--harness", "default", "-o", "provider.kind=olmo_core",
            "-o", f"provider.kwargs.batch_size={batch_size}"]
    for b in benches:
        args += ["-t", b]
        if limit is not None:
            args += ["-o", f"limit={limit}"]
    return ["-m", "/tmp/ck", *args, "-O", "/tmp/out"]


argv = build_native(512, MCQ)
print("  mcq argv:", " ".join(argv))
task_ov, harness_ov = parse(argv)
check("batch_size lands in HARNESS overrides",
      "provider.kwargs.batch_size=512" in harness_ov, str(harness_ov))
check("provider.kind=olmo_core is accepted",
      "provider.kind=olmo_core" in harness_ov, str(harness_ov))
check("batch_size did not leak into a task",
      not any("batch_size" in v for vals in task_ov.values() for v in vals),
      str(dict(task_ov)))
check("the multiple-choice half carries only its own benchmarks",
      sorted(task_ov) == sorted(MCQ) or not task_ov, str(sorted(task_ov)))

argv = build_native(192, GEN, limit=20)
print("  generative argv:", " ".join(argv))
task_ov, harness_ov = parse(argv)
check("the generative half carries its own batch size",
      sorted(harness_ov) == ["provider.kind=olmo_core",
                             "provider.kwargs.batch_size=192"],
      str(sorted(harness_ov)))
check("and its own per-benchmark caps",
      all(task_ov[b] == ["limit=20"] for b in GEN), str(dict(task_ov)))
check("and none of the multiple-choice benchmarks",
      not any(b in task_ov for b in MCQ), str(sorted(task_ov)))
# Overrides arrive from argv as strings, and _validate_batch_size demands a real int:
# "512" would be rejected at provider construction. So the coercion is load-bearing,
# not cosmetic, and worth asserting against the real function rather than assumed.
coerced = utils._coerce_value("512")
check("batch_size is coerced to an int, not left a string",
      coerced == 512 and isinstance(coerced, int), f"{coerced!r} ({type(coerced).__name__})")

print()
print("=" * 70)
print("CONTROL: the two mistakes the predecessor made must still be errors")
print("=" * 70)
# The predecessor emitted this: a provider key after -t. It is rejected outright.
bad = ["-m", "x", "--harness", "default", "-t", "hellaswag",
       "-o", "provider.tensor_parallel_size=2"]
try:
    parse(bad)
    check("provider key after -t rejected", False, "no error raised")
except UsageError as e:
    check("provider key after -t rejected", "not a TaskConfig" in str(e), str(e)[:70])
# -o with nothing preceding
try:
    parse(["-m", "x", "-o", "limit=5"])
    check("bare -o rejected", False, "no error raised")
except UsageError as e:
    check("bare -o rejected", True, str(e)[:60])

print()
if failures:
    print(f"{len(failures)} CHECK(S) FAILED: {failures}")
    raise SystemExit(1)
print("ALL ARGV CHECKS PASSED")
