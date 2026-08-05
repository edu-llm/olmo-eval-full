"""Exercise summarize_accuracy.py against a realistic metrics.json fixture."""

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import SUMMARIZER as SCRIPT

# The runs tree and the three summary files are *outputs*, so they go to scratch
# rather than next to this file: written into the suite directory they would show
# up as untracked repo state after every run, which is how the artifacts this
# suite was rescued from came to be mistaken for fixtures.
HERE = Path(tempfile.mkdtemp(prefix="evalck_summarizer_"))
RUNS = HERE / "runs"


def metrics_payload(scale: float) -> dict:
    """Shaped exactly like MetricsOutput.to_dict() for a 7-benchmark run."""

    def mcq(task, score, n):
        return {
            "task": task,
            "metrics": {"accuracy": {"logprob": score}},
            "num_instances": n,
            "primary_metric": "accuracy:logprob",
            "duration_seconds": 12.5,
            "task_hash": "abc123",
        }

    tasks = [
        mcq("hellaswag", 0.40 * scale, 10042),
        mcq("piqa", 0.62 * scale, 1838),
        mcq("arc_easy", 0.55 * scale, 2376),
        mcq("csqa", 0.31 * scale, 1221),
        mcq("socialiqa", 0.44 * scale, 1954),
        {
            "task": "naturalqs",
            "metrics": {
                "f1": {"drop_f1": 0.31 * scale},
                "accuracy": {"drop_exact_match": 0.20 * scale},
            },
            "num_instances": 3610,
            "primary_metric": "f1:drop_f1",
        },
        {
            "task": "jeopardy",
            "metrics": {
                "f1": {"f1": 0.40 * scale},
                "accuracy": {"squad_exact_match": 0.28 * scale},
            },
            "num_instances": 2117,
            "primary_metric": "f1:f1",
        },
    ]
    summary = {}
    for t in tasks:
        pm = t["primary_metric"]
        m, s = pm.split(":")
        summary[t["task"]] = {"metric": pm, "score": t["metrics"][m][s]}
    return {
        "timestamp": "2026-08-03T23:00:00Z",
        "config": {"model": "ck", "model_hash": "deadbeef"},
        "tasks": tasks,
        "summary": summary,
        "errors": [],
        "experiment_id": "exp123",
    }


def write_run(run_id, checkpoint, status, scale=None, partial_instances=False):
    d = RUNS / run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "run_provenance.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "checkpoint": checkpoint,
                "benchmarks": [
                    "hellaswag", "piqa", "arc_easy", "csqa",
                    "socialiqa", "naturalqs", "jeopardy",
                ],
                "status": status,
                "limit": None,
                "tensor_parallel_size": 1,
                "git_sha": "cafe123",
            }
        ),
        encoding="utf-8",
    )
    if scale is not None:
        payload = metrics_payload(scale)
        if partial_instances:
            # Simulate dropped instances: num_instances well below split size.
            payload["tasks"][0]["num_instances"] = 4000
        (d / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")


def main() -> int:
    if RUNS.exists():
        import shutil
        shutil.rmtree(RUNS)

    # step9 / step10 exercise numeric-not-lexical ordering.
    write_run("step9", "s3://b/ck/step9", "ok", scale=0.80)
    write_run("step10", "s3://b/ck/step10", "ok", scale=0.95, partial_instances=True)
    write_run("step1000", "s3://b/ck/step1000", "ok", scale=1.0)
    # Failed: provenance but no metrics.json, the case that used to vanish.
    write_run("step2000", "s3://b/ck/step2000", "failed", scale=None)
    # No trailing digits: must sort last, not crash.
    write_run("final", "s3://b/ck/final", "ok", scale=1.0)

    out_csv = HERE / "accuracy.csv"
    out_wide = HERE / "accuracy_wide.csv"
    out_json = HERE / "accuracy.json"
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--runs-dir", str(RUNS),
         "--out-csv", str(out_csv), "--out-wide-csv", str(out_wide),
         "--out-json", str(out_json)],
        capture_output=True, text=True,
    )
    print(r.stdout)
    if r.returncode != 0:
        print("STDERR:", r.stderr)
        return 1

    failures = []

    long_rows = list(csv.DictReader(out_csv.open(encoding="utf-8")))
    wide_rows = list(csv.DictReader(out_wide.open(encoding="utf-8")))

    print("=" * 70)
    print("CHECKS")
    print("=" * 70)

    def check(label, cond, detail=""):
        print(f"  {'OK  ' if cond else 'FAIL'} {label}" + (f"  [{detail}]" if detail else ""))
        if not cond:
            failures.append(label)

    # 5 MCQ x1 metric + 2 generative x2 metrics = 9 rows per successful checkpoint
    ok_runs = [r for r in long_rows if r["run_id"] != "step2000"]
    check("9 metric rows per successful checkpoint",
          len(ok_runs) == 9 * 4, f"{len(ok_runs)} rows / 4 runs")

    # Ordering: numeric by step, non-numeric last
    seen = []
    for r in long_rows:
        if r["run_id"] not in seen:
            seen.append(r["run_id"])
    check("checkpoints ordered numerically, non-numeric last",
          seen == ["step9", "step10", "step1000", "step2000", "final"], str(seen))

    # The failed checkpoint must be present with no score
    failed = [r for r in long_rows if r["run_id"] == "step2000"]
    check("failed checkpoint kept in long CSV", len(failed) == 7, f"{len(failed)} rows")
    check("failed checkpoint has no score", all(r["score"] == "" for r in failed))
    check("failed checkpoint status preserved",
          all(r["status"] == "failed" for r in failed))

    # Both metrics survive on the generative pair
    nq = [r for r in long_rows if r["run_id"] == "step1000" and r["benchmark"] == "naturalqs"]
    check("naturalqs reports 2 metrics", len(nq) == 2,
          str(sorted(r["metric"] for r in nq)))
    check("naturalqs f1 is primary",
          any(r["metric"] == "f1" and r["is_primary"] == "True" for r in nq))
    check("naturalqs exact match is not primary",
          any(r["metric"] == "accuracy" and r["is_primary"] == "False" for r in nq))
    check("naturalqs scorers recorded",
          sorted(r["scorer"] for r in nq) == ["drop_exact_match", "drop_f1"])

    jp = [r for r in long_rows if r["run_id"] == "step1000" and r["benchmark"] == "jeopardy"]
    check("jeopardy exact match scorer is squad_exact_match",
          any(r["scorer"] == "squad_exact_match" for r in jp))

    # Wide shape
    check("wide CSV has one row per checkpoint", len(wide_rows) == 5, f"{len(wide_rows)}")
    cols = set(wide_rows[0].keys())
    check("single-metric benchmarks get bare columns",
          {"hellaswag", "piqa", "arc_easy", "csqa", "socialiqa"} <= cols)
    check("multi-metric benchmarks get qualified columns",
          {"naturalqs.f1", "naturalqs.accuracy", "jeopardy.f1", "jeopardy.accuracy"} <= cols,
          str(sorted(c for c in cols if "." in c)))
    check("no bare naturalqs/jeopardy column", "naturalqs" not in cols and "jeopardy" not in cols)

    wide_by_run = {r["run_id"]: r for r in wide_rows}
    check("wide ordered numerically",
          [r["run_id"] for r in wide_rows] == ["step9", "step10", "step1000", "step2000", "final"],
          str([r["run_id"] for r in wide_rows]))
    check("failed checkpoint present in wide with blank score",
          wide_by_run["step2000"]["hellaswag"] == "")
    check("failed checkpoint status in wide", wide_by_run["step2000"]["status"] == "failed")
    check("successful wide row carries a score",
          abs(float(wide_by_run["step1000"]["hellaswag"]) - 0.40) < 1e-9,
          wide_by_run["step1000"]["hellaswag"])

    # num_instances is surfaced, so a dropped-instance run is detectable
    hs10 = [r for r in long_rows if r["run_id"] == "step10" and r["benchmark"] == "hellaswag"]
    check("num_instances surfaced for dropped-instance detection",
          hs10 and hs10[0]["num_instances"] == "4000", hs10[0]["num_instances"] if hs10 else "-")

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    check("json carries both long and wide", {"rows", "wide"} <= set(payload))

    print()
    if failures:
        print(f"{len(failures)} CHECK(S) FAILED: {failures}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
