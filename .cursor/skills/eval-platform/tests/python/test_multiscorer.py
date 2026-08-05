"""Wide-CSV naming when one metric name carries several scorers.

Models the PopQA shape under consideration: f1, plus two accuracy-family scores
(SQuAD exact match and containment) that would both be metric "accuracy".
Also checks the existing two-metric and single-metric shapes are unchanged.
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


root = Path(tempfile.mkdtemp()) / "runs"
(root / "step100").mkdir(parents=True)
(root / "step100" / "run_provenance.json").write_text(
    json.dumps({"checkpoint": "s3://b/step100", "status": "ok", "benchmarks": []})
)
(root / "step100" / "metrics.json").write_text(json.dumps({
    "timestamp": "t", "config": {}, "summary": {},
    "tasks": [
        # single metric, single scorer -> bare column
        {"task": "hellaswag", "num_instances": 2, "primary_metric": "accuracy:logprob",
         "metrics": {"accuracy": {"logprob": 0.51}}},
        # two metrics, one scorer each -> benchmark.metric
        {"task": "jeopardy", "num_instances": 2, "primary_metric": "f1:f1",
         "metrics": {"f1": {"f1": 0.20}, "accuracy": {"squad_exact_match": 0.11}}},
        # the PopQA shape: "accuracy" carries TWO scorers alongside f1
        {"task": "popqa", "num_instances": 2, "primary_metric": "accuracy:containment",
         "metrics": {"f1": {"f1": 0.44},
                     "accuracy": {"squad_exact_match": 0.28, "containment": 0.53}}},
    ],
}))

wide_csv = root.parent / "accuracy_wide.csv"
long_csv = root.parent / "accuracy.csv"
subprocess.run([sys.executable, str(SUMMARIZER), "--runs-dir", str(root),
                "--out-wide-csv", str(wide_csv), "--out-csv", str(long_csv)],
               capture_output=True, text=True, check=True)

row = list(csv.DictReader(wide_csv.open(encoding="utf-8")))[0]
hdr = [h for h in row if h not in ("run_id", "step", "checkpoint", "status")]
print("  wide columns:", hdr)
print()

print("existing shapes must be unchanged:")
check("single-metric benchmark keeps a bare column", "hellaswag" in hdr, str(hdr))
check("two-metric benchmark keeps benchmark.metric", 
      "jeopardy.f1" in hdr and "jeopardy.accuracy" in hdr, str(hdr))
check("no scorer suffix added where it is not needed",
      not any(h.startswith("jeopardy.accuracy.") or h.startswith("hellaswag.") for h in hdr),
      str(hdr))

print()
print("the new case -- one metric, two scorers:")
check("f1 stays at two levels (single scorer)", "popqa.f1" in hdr, str(hdr))
check("SQuAD exact match gets its own column",
      "popqa.accuracy.squad_exact_match" in hdr, str(hdr))
check("containment gets its own column",
      "popqa.accuracy.containment" in hdr, str(hdr))
check("no ambiguous bare popqa.accuracy column", "popqa.accuracy" not in hdr, str(hdr))

print()
print("no score is lost:")
check("all three popqa scores present and distinct",
      {row.get("popqa.f1"), row.get("popqa.accuracy.squad_exact_match"),
       row.get("popqa.accuracy.containment")} == {"0.44", "0.28", "0.53"},
      str({k: v for k, v in row.items() if k.startswith("popqa")}))
long_rows = list(csv.DictReader(long_csv.open(encoding="utf-8")))
check("long CSV still carries every popqa row",
      len([r for r in long_rows if r["benchmark"] == "popqa"]) == 3)
check("primary flag lands on the containment score",
      [r["is_primary"] for r in long_rows
       if r["benchmark"] == "popqa" and r["scorer"] == "containment"] == ["True"])

print()
if failures:
    print(f"{len(failures)} FAILURE(S): " + "; ".join(failures))
    sys.exit(1)
print("MULTI-SCORER NAMING CORRECT")
