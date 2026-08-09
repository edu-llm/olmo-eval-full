#!/usr/bin/env python
"""Compare acceptance-gate metrics across multiple models' verdict files (zero API calls).

Prints overall accuracy/false-pass, critical-failure sensitivity + critical false-pass,
and content-skill false-pass for each model, so the lowest-risk judge is visible before
any paid run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_verdicts import _confusion, _load_labels, _load_verdicts  # noqa: E402

RESULTS = Path("eduLLM-Evals/api_judge_pilot/results_strict")
LABELS = Path("eduLLM-Evals/api_judge_pilot/human_labels.csv")
MODELS = [
    "claude-group__claude-sonnet-4-6",
    "claude-group__claude-opus-4-6",
    "gemini-group__gemini-2.5-flash",
    "gemini-group__gemini-3-flash-preview",
    "openai-group__gpt-4.1",
]


def _metrics(verdicts_path: Path, labels: dict) -> dict:
    verdicts = _load_verdicts(verdicts_path)
    allp: list[tuple[str, str]] = []
    critp: list[tuple[str, str]] = []
    contentp: list[tuple[str, str]] = []
    for cid, lab in labels.items():
        v = verdicts.get(cid)
        if v is None:
            continue
        human = lab["human_label"].strip().lower()
        verdict = v["verdict"].strip().lower()
        if human not in ("pass", "fail") or verdict not in ("pass", "fail"):
            continue
        pair = (verdict, human)
        allp.append(pair)
        if (lab.get("criticality") or "").startswith("critical"):
            critp.append(pair)
        if (lab.get("primary_skill") or "") == "content":
            contentp.append(pair)

    def fp_rate(pairs):
        c = _confusion(pairs)
        hf = c["fp"] + c["tn"]
        return (c["fp"] / hf) if hf else None, (c["tn"] / hf) if hf else None

    o = _confusion(allp)
    o_fp, _ = fp_rate(allp)
    crit_fp, crit_sens = fp_rate(critp)
    content_fp, _ = fp_rate(contentp)
    return {
        "acc": (o["tp"] + o["tn"]) / o["n"] if o["n"] else 0.0,
        "fp": o_fp,
        "crit_sens": crit_sens,
        "crit_fp": crit_fp,
        "content_fp": content_fp,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--labels", type=Path, default=LABELS)
    args = parser.parse_args()

    labels = _load_labels(args.labels)
    header = f"{'model':38} {'acc':>6} {'FP%':>6} {'critSens':>9} {'critFP%':>8} {'contentFP%':>11}"
    print(header)
    print("-" * len(header))
    for m in MODELS:
        path = args.results_dir / f"{m}.verdicts.jsonl"
        if not path.is_file():
            print(f"{m:38} (missing)")
            continue
        r = _metrics(path, labels)
        print(
            f"{m[:38]:38} {r['acc']:>6.3f} {r['fp'] * 100:>6.1f} {r['crit_sens']:>9.3f} "
            f"{r['crit_fp'] * 100:>8.1f} {r['content_fp'] * 100:>11.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
