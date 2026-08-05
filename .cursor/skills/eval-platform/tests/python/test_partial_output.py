"""What survives a crash: partial prediction files, the heartbeat, and salvage.

Three pieces, exercised against their real source with no torch and no AWS:

  writers      append/discard, and that a partial file lands beside its final form
  run_eval     the interim uploader thread and the _IN_PROGRESS marker's lifecycle
  salvage      reading a file whose last line was cut off mid-append

The scoring loop that calls the writers cannot run here -- it needs a model -- so what
is pinned instead is the contract it depends on: where the partial file goes, that
discarding removes exactly it, and that a truncated tail is readable.
"""

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import REPO

failures = []


def _stub_package_tree():
    """Make olmo_eval importable enough to load three leaf modules, and no further.

    Importing the package properly runs its task discovery, which imports every task
    and so needs sympy and torch. The modules under test here reach only a logger, two
    filename suffixes and one sanitizer, so their parents are stubbed and those three
    leaves are loaded from their real source -- the point being to exercise the actual
    path arithmetic rather than a restatement of it.
    """
    import logging
    import types as pytypes

    for name in (
        "olmo_eval",
        "olmo_eval.common",
        "olmo_eval.runners",
        "olmo_eval.runners.common",
        "olmo_eval.runners.io",
        "olmo_eval.runners.processing",
    ):
        sys.modules.setdefault(name, pytypes.ModuleType(name))

    log_mod = pytypes.ModuleType("olmo_eval.common.logging")
    log_mod.get_logger = logging.getLogger
    sys.modules["olmo_eval.common.logging"] = log_mod

    common_types = pytypes.ModuleType("olmo_eval.common.types")
    common_types.SamplingParams = object
    sys.modules["olmo_eval.common.types"] = common_types


def _load_real(module_name, relative):
    spec = importlib.util.spec_from_file_location(module_name, REPO / relative)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_stub_package_tree()
_load_real("olmo_eval.runners.common.types", "src/olmo_eval/runners/common/types.py")
_load_real("olmo_eval.runners.processing.utils", "src/olmo_eval/runners/processing/utils.py")


def check(label, cond, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'} {label}" + (f"  -- {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(label)


def load(name, relative):
    return _load_real(name, relative)


# --- writers: where a partial file goes, and that discard removes it ----------------
print("\npartial prediction files:")

writers = load("writers_under_test", "src/olmo_eval/runners/io/writers.py")

with tempfile.TemporaryDirectory() as tmp:
    rows = [{"doc_id": i, "instance_metrics": {"accuracy": {"logprob": 1.0}}} for i in range(3)]
    writers.append_partial_predictions_jsonl(tmp, "hellaswag", rows[:2], "modelA", task_hash="abcdef123456")
    writers.append_partial_predictions_jsonl(tmp, "hellaswag", rows[2:], "modelA", task_hash="abcdef123456")

    found = sorted(Path(tmp).rglob("*.jsonl"))
    check("appending twice makes one file", len(found) == 1, str(found))

    partial = found[0]
    check(
        "named as the final file plus a partial suffix",
        partial.name == "hellaswag_123456" + writers.PARTIAL_PREDICTIONS_SUFFIX,
        partial.name,
    )

    lines = partial.read_text(encoding="utf-8").strip().split("\n")
    check("both appends are present, in order", len(lines) == 3, f"{len(lines)} lines")
    check("rows survive the round trip", [json.loads(x)["doc_id"] for x in lines] == [0, 1, 2])

    # The complete writer must land in the same directory under the same stem, or the
    # discard below would leave the partial file stranded next to it.
    writers.write_predictions_jsonl(tmp, "hellaswag", rows, "modelA", task_hash="abcdef123456")
    both = sorted(p.name for p in Path(tmp).rglob("*.jsonl"))
    check("complete file lands beside the partial", len(both) == 2, str(both))
    check(
        "and shares its stem",
        both[0].startswith("hellaswag_123456") and both[1].startswith("hellaswag_123456"),
        str(both),
    )

    writers.discard_partial_predictions(tmp, "hellaswag", "modelA", task_hash="abcdef123456")
    left = sorted(p.name for p in Path(tmp).rglob("*.jsonl"))
    check("discard removes the partial", len(left) == 1 and not left[0].endswith(".partial.jsonl"), str(left))
    check("and leaves the complete one", left == ["hellaswag_123456-predictions.jsonl"], str(left))

    # Called again on an already-clean directory, as it will be whenever a task is
    # short enough that no partial was ever flushed.
    try:
        writers.discard_partial_predictions(tmp, "hellaswag", "modelA", task_hash="abcdef123456")
        check("discarding a file that is not there is not an error", True)
    except Exception as error:
        check("discarding a file that is not there is not an error", False, repr(error))


# --- run_eval: the uploader thread and the marker lifecycle -------------------------
print("\ninterim upload and the heartbeat marker:")

run_eval = load("run_eval_partial_under_test", "src/olmo_eval/platform/run_eval.py")


class FakeClient:
    """Records what the uploader would have done to S3."""

    def __init__(self):
        self.markers = {}
        self.uploads = 0
        self.deleted = []

    def put_object(self, Bucket, Key, Body, **kw):  # noqa: N803 - boto3's own casing
        self.markers[Key.rsplit("/", 1)[-1]] = Body.decode("utf-8")

    def upload_file(self, filename, bucket, key):
        self.uploads += 1

    def delete_object(self, Bucket, Key):  # noqa: N803
        name = Key.rsplit("/", 1)[-1]
        self.deleted.append(name)
        self.markers.pop(name, None)


def run_with(interval, subprocess_seconds):
    """Run the real run() against a stub subprocess, returning the fake client."""
    import subprocess as real_subprocess
    import time

    client = FakeClient()
    original_client, original_run = run_eval._client, run_eval.subprocess.run
    run_eval._client = lambda: client

    class Completed:
        returncode = 0

    def fake_run(command, check=False):
        time.sleep(subprocess_seconds)
        return Completed()

    run_eval.subprocess.run = fake_run
    try:
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp) / "out"
            local.mkdir()
            (local / "metrics.json").write_text("{}", encoding="utf-8")
            plan = run_eval.Plan(
                checkpoint="s3://b/teams/t/runs/r/checkpoints/step1",
                benchmarks=("csqa",),
                output_prefix="s3://b/teams/t/runs/r/",
                provider="olmo_core",
                limit=None,
                tokenizer=None,
                local_out=local,
            )
            run_eval.run(plan, upload_interval=interval)
            return client
    finally:
        run_eval._client, run_eval.subprocess.run = original_client, original_run
        del real_subprocess


client = run_with(interval=0.2, subprocess_seconds=0.75)
check("heartbeat is written", "_IN_PROGRESS" in client.markers, str(list(client.markers)))
check("it carries a timestamp", "last_upload=" in client.markers.get("_IN_PROGRESS", ""))
check("interim uploads happened while the eval ran", client.uploads >= 1, f"{client.uploads} upload(s)")
check(
    "no terminal marker yet -- run() does not write one",
    "_READY" not in client.markers and "_FAILED" not in client.markers,
)

client = run_with(interval=0, subprocess_seconds=0.1)
check("interval 0 writes no heartbeat", "_IN_PROGRESS" not in client.markers, str(list(client.markers)))
check("interval 0 uploads nothing mid-run", client.uploads == 0, f"{client.uploads} upload(s)")

check(
    "remove_marker deletes the heartbeat by name",
    (lambda c: (setattr(run_eval, "_client", lambda: c), run_eval.remove_marker("s3://b/p/", "_IN_PROGRESS"), c.deleted == ["_IN_PROGRESS"])[-1])(FakeClient()),
)


# --- salvage: a file whose last line was cut off ------------------------------------
print("\nsalvaging a truncated file:")

salvage = load("salvage_under_test", ".cursor/skills/eval-platform/scripts/salvage_partial.py")

with tempfile.TemporaryDirectory() as tmp:
    good = "\n".join(
        json.dumps({"doc_id": i, "instance_metrics": {"accuracy": {"logprob": float(i % 2)}}})
        for i in range(4)
    )
    path = Path(tmp) / "x-predictions.partial.jsonl"

    path.write_text(good + "\n" + '{"doc_id":4,"instance_met', encoding="utf-8")
    rows, note, error = salvage.read_rows(path)
    check("a truncated tail is dropped, not fatal", error is None and len(rows) == 4, f"{error} {len(rows)}")
    check("and is reported", note is not None and "truncated" in note, str(note))
    check("the surviving rows score", salvage.score(rows) == {"accuracy.logprob": 0.5}, str(salvage.score(rows)))

    path.write_text(good + "\n", encoding="utf-8")
    rows, note, error = salvage.read_rows(path)
    check("an intact file reports no truncation", note is None and error is None and len(rows) == 4)

    path.write_text('{"a":1}\n{oops\n{"b":2}\n', encoding="utf-8")
    _, _, error = salvage.read_rows(path)
    check("corruption that is not at the end is an error", error is not None and "not the last" in error, str(error))

    path.write_text("not json at all\n", encoding="utf-8")
    rows, _, error = salvage.read_rows(path)
    check("a wholly unreadable file is an error, not an empty one", error is not None and not rows, str(error))


print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    raise SystemExit(1)
print("PARTIAL OUTPUT CONTRACTS HOLD")
