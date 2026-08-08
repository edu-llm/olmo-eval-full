"""Run compare_judge_reliability across the recovered v3 judge waves."""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]  # eduLLM-Evals
SCRIPT = REPO / "scripts" / "compare_judge_reliability.py"
V3 = HERE.parent / "judge-v3-results"
HUMAN = HERE / "human_labels.csv"

judges = ["selene", "flow", "prometheus", "qwen", "gemma"]
waves = ["canonical_r1", "canonical_r2", "canonical_r3",
         "whitespace_r1", "header_synonyms_r1", "instruction_politeness_r1"]

wave_args: list[str] = []
missing: list[str] = []
for j in judges:
    for w in waves:
        p = V3 / j / w / f"{w}.jsonl"
        if not p.is_file():
            missing.append(str(p))
        wave_args += ["--wave", f"{j}:{w}:{p}"]

if missing:
    print("MISSING WAVES:")
    print("\n".join(missing))
    sys.exit(2)

cmd = [sys.executable, str(SCRIPT), str(HUMAN), *wave_args,
       "--json-out", str(HERE / "reliability_report.json"),
       "--csv-out", str(HERE / "reliability_summary.csv")]
print("Running comparison over", len(judges), "judges x", len(waves), "waves ...")
proc = subprocess.run(cmd, cwd=str(REPO))
sys.exit(proc.returncode)
