"""Does summarize_accuracy.py produce the outputs SKILL.md promises?

Builds a runs tree matching olmo_eval.runners.common.models.MetricsOutput
(tasks: list of TaskMetricsEntry dicts, metrics nested {metric: {scorer: value}},
primary_metric in "metric:scorer" form) and checks the emitted CSVs.
"""

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import SUMMARIZER

failures = []


def check(label, cond, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'} {label}" + (f"  -- {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(label)


def task(name, metrics, primary, n):
    return {"task": name, "metrics": metrics, "num_instances": n, "primary_metric": primary}


root = Path(tempfile.mkdtemp()) / "runs"
root.mkdir(parents=True)


def write_run(run_id, ckpt, status, tasks, benchmarks):
    d = root / run_id
    d.mkdir()
    (d / "run_provenance.json").write_text(
        json.dumps({"checkpoint": ckpt, "status": status, "benchmarks": benchmarks})
    )
    if tasks is not None:
        (d / "metrics.json").write_text(
            json.dumps({"timestamp": "t", "config": {}, "tasks": tasks, "summary": {}})
        )


# step1000: a single-metric MCQ task plus a two-metric generative task
write_run(
    "exp-step1000",
    "s3://b/ck/step1000",
    "ok",
    [
        task("hellaswag", {"accuracy": {"default": 0.55}}, "accuracy:default", 10042),
        task(
            "jeopardy",
            {"f1": {"squad": 0.42}, "accuracy": {"squad_em": 0.31}},
            "f1:squad",
            2000,
        ),
    ],
    ["hellaswag", "jeopardy"],
)
# step500: earlier step, must sort first
write_run(
    "exp-step500",
    "s3://b/ck/step500",
    "ok",
    [
        task("hellaswag", {"accuracy": {"default": 0.41}}, "accuracy:default", 10042),
        task("jeopardy", {"f1": {"squad": 0.28}, "accuracy": {"squad_em": 0.19}}, "f1:squad", 2000),
    ],
    ["hellaswag", "jeopardy"],
)
# step1500: eval failed -> provenance but no metrics.json
write_run("exp-step1500", "s3://b/ck/step1500", "conversion_failed", None, ["hellaswag", "jeopardy"])

out_long = root.parent / "accuracy.csv"
out_wide = root.parent / "accuracy_wide.csv"
out_json = root.parent / "accuracy.json"

proc = subprocess.run(
    [sys.executable, str(SUMMARIZER), "--runs-dir", str(root),
     "--out-csv", str(out_long), "--out-wide-csv", str(out_wide), "--out-json", str(out_json)],
    capture_output=True, text=True,
)
print(proc.stdout)
if proc.returncode != 0:
    print(proc.stderr)
    sys.exit("summarizer exited nonzero")

print("--- long CSV (accuracy.csv) ---")
long_rows = list(csv.DictReader(out_long.open(encoding="utf-8")))
expected_fields = ["run_id", "step", "checkpoint", "benchmark", "metric", "scorer",
                   "score", "is_primary", "num_instances", "status"]
check("long CSV header matches documented schema", list(long_rows[0].keys()) == expected_fields,
      str(list(long_rows[0].keys())))
check("long CSV has a row per checkpoint/benchmark/metric",
      len([r for r in long_rows if r["score"]]) == 6, f"{len([r for r in long_rows if r['score']])}")
prim = [r for r in long_rows if r["benchmark"] == "jeopardy" and r["metric"] == "f1"]
check("is_primary True for the primary metric", all(r["is_primary"] == "True" for r in prim))
sec = [r for r in long_rows if r["benchmark"] == "jeopardy" and r["metric"] == "accuracy"]
check("is_primary False for the secondary metric", all(r["is_primary"] == "False" for r in sec))
check("num_instances carried through", prim[0]["num_instances"] == "2000", prim[0]["num_instances"])

print()
print("--- wide CSV (accuracy_wide.csv, the deliverable) ---")
wide_rows = list(csv.DictReader(out_wide.open(encoding="utf-8")))
hdr = list(wide_rows[0].keys())
print("  header:", hdr)
check("single-metric benchmark gets a bare column", "hellaswag" in hdr, str(hdr))
check("multi-metric benchmark gets <bench>.<metric> columns",
      "jeopardy.f1" in hdr and "jeopardy.accuracy" in hdr, str(hdr))
check("no bare 'jeopardy' column colliding with the split ones", "jeopardy" not in hdr)
check("one row per checkpoint", len(wide_rows) == 3, str(len(wide_rows)))
check("rows sorted by step", [r["step"] for r in wide_rows] == ["500", "1000", "1500"],
      str([r["step"] for r in wide_rows]))
check("step parsed from run id", wide_rows[0]["step"] == "500", wide_rows[0]["step"])
check("checkpoint URI preserved", wide_rows[0]["checkpoint"] == "s3://b/ck/step500")

print()
print("--- failure visibility ---")
failed_wide = [r for r in wide_rows if r["run_id"] == "exp-step1500"][0]
check("failed checkpoint still has a wide row", failed_wide is not None)
check("failed checkpoint carries its failure status",
      failed_wide["status"] == "conversion_failed", failed_wide["status"])
check("failed checkpoint has blank scores, not zeros",
      failed_wide["hellaswag"] == "", repr(failed_wide["hellaswag"]))
failed_long = [r for r in long_rows if r["run_id"] == "exp-step1500"]
check("failed checkpoint has placeholder rows per requested benchmark",
      len(failed_long) == 2, str(len(failed_long)))
check("succeeded checkpoints report status ok",
      all(r["status"] == "ok" for r in wide_rows if r["run_id"] != "exp-step1500"))

print()
print("--- accuracy.json ---")
payload = json.loads(out_json.read_text(encoding="utf-8"))
check("accuracy.json has both shapes", "rows" in payload and "wide" in payload, str(list(payload)))
check("accuracy.json wide matches CSV row count", len(payload["wide"]) == len(wide_rows))

print()
if failures:
    print(f"{len(failures)} FAILURE(S): " + "; ".join(failures))
    sys.exit(1)
print("ALL OUTPUT CONTRACTS HOLD")
