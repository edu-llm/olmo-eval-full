#!/usr/bin/env python
"""Finalize the biggen gold set from packets + human decisions.

Label resolution per case:
  * gold_case_id in --decisions        -> human_adjudicated  (your verdict; also use to OVERRIDE a concordant rec)
  * else if proposers concordant       -> ai_concordant_accepted  (opus-5 + gpt-5.5 agreed)
  * else                               -> needs_human  (gold_label=null; still to resolve)

Emits gold_labels.jsonl (the gold artifact for scoring judge candidates) + gold_manifest.json.
Re-run any time as decisions come in; it is idempotent.

--decisions accepts a JSON object {"biggen__0021": "fail", ...} and/or repeated
--set biggen__0021=fail on the CLI.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--packets", type=Path, required=True)
    ap.add_argument("--sample", type=Path, required=True, help="for _qwen_verdict join")
    ap.add_argument("--decisions", type=Path, default=None)
    ap.add_argument("--set", dest="pairs", action="append", default=[], help="id=label")
    ap.add_argument("--spotchecked", nargs="*", default=[],
                    help="concordant ids the owner reviewed and accepted (verified, not rubber-stamped)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    spotchecked = set(args.spotchecked)

    decisions: dict[str, str] = {}
    if args.decisions and args.decisions.is_file():
        decisions.update(json.loads(args.decisions.read_text(encoding="utf-8")))
    for pair in args.pairs:
        cid, _, lbl = pair.partition("=")
        decisions[cid.strip()] = lbl.strip().lower()
    for cid, lbl in decisions.items():
        if lbl not in ("pass", "fail"):
            raise SystemExit(f"decision for {cid} must be pass/fail, got {lbl!r}")

    qwen = {
        json.loads(l)["gold_case_id"]: json.loads(l).get("_qwen_verdict")
        for l in args.sample.read_text(encoding="utf-8").splitlines() if l.strip()
    }

    rows = [json.loads(l) for l in args.packets.read_text(encoding="utf-8").splitlines() if l.strip()]
    gold = []
    for r in rows:
        cid = r["gold_case_id"]
        if cid in decisions:
            label, prov = decisions[cid], "human_adjudicated"
        elif r["concordant"]:
            label = r["recommendation"]
            prov = "ai_concordant_spotchecked" if cid in spotchecked else "ai_concordant_accepted"
        else:
            label, prov = None, "needs_human"
        gold.append({
            "gold_case_id": cid,
            "model": r["model"],
            "scenario_id": r["scenario_id"],
            "criterion_id": r["criterion_id"],
            "stratum": r.get("stratum", r.get("capability")),
            "criticality": r.get("criticality"),
            "gold_label": label,
            "provenance": prov,
            "_qwen_verdict": qwen.get(cid),
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for g in gold:
            fh.write(json.dumps(g, ensure_ascii=False) + "\n")

    manifest = {
        "n": len(gold),
        "by_provenance": dict(Counter(g["provenance"] for g in gold)),
        "by_label": dict(Counter(g["gold_label"] for g in gold)),
        "by_stratum": dict(Counter(g["stratum"] for g in gold)),
        "resolved": sum(1 for g in gold if g["gold_label"] is not None),
        "pending_needs_human": sum(1 for g in gold if g["gold_label"] is None),
        "human_adjudicated_ids": sorted(g["gold_case_id"] for g in gold if g["provenance"] == "human_adjudicated"),
    }
    (args.out.parent / "gold_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
