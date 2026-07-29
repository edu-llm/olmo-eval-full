"""Apply a curation edit-spec to a CLONE of the rubric bank.

NEVER mutates the source files. Reads data/TutorBench/rubrics_qmatrix_final.jsonl +
data/TutorBench/scenarios.jsonl, applies curation/pilot_edits.json, and writes curated
clones under data/TutorBench/curated/ plus a human-readable change report under curation/.

Supported ops (see curation/pilot_edits.json meta.calibration_baseline):
  * split            -- replace one criterion with N atomic children. Children
                        COPY the parent's criticality / q_mapping / difficulty /
                        discrimination / primary_skill / objectivity /
                        explicitness / irt_params unless overridden per child.
  * soften           -- replace the criterion text in place (wording only).
  * rescope_optional -- replace text, force criticality=not_critical, set
                        optional=true, attach judge_guidance (Q3: judge may mark
                        the criterion N/A / leave blank when it does not apply).
  * keep             -- reviewed no-op (e.g. withhold-answer false positives, Q6).
  * drop             -- remove the criterion entirely.

For every scenario that actually changes, all of its criteria are RENUMBERED
sequentially ({scenario}_c01..cN) and scenario.criterion_ids is regenerated
(Q7). Within edited scenarios, all-zero format/persona "orphan" criteria are
consolidated into ONE standard presentation criterion (Q4). Scenarios whose only
op is `keep` are left byte-identical (no needless ID churn).

Every produced/edited record carries a `curation` provenance block.

Usage:
    python scripts/apply_curation.py
    python scripts/apply_curation.py --spec curation/pilot_edits.json \
        --rubrics data/TutorBench/rubrics_qmatrix_final.jsonl \
        --scenarios data/TutorBench/scenarios.jsonl --out-dir data/TutorBench/curated
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import re
from collections import OrderedDict
from pathlib import Path

# Matches the audit's surface/style detector so consolidation covers exactly the
# criteria the audit flags as surface_format_or_persona_only (all-zero Q + style).
FORMAT_ORPHAN_RE = re.compile(
    r"\b(second person|first person|perspective of|use of \"?you|markdown|latex|"
    r"headings?|bold|bullet|format\w*|demarcate)\b",
    re.I,
)

# --- aspect-aware presentation consolidation (curation_v1) ------------------
# A scenario's style orphans are merged into ONE optional presentation criterion
# whose text covers ONLY the aspects actually present: single-orphan scenarios
# target just their one aspect; multi-aspect scenarios target all present.
_PERSONA_RE = re.compile(
    r"(second person|first person|perspective of|use of \"?you|address(?:ing)? the student|"
    r"conversational|robotic|tutor or|teacher or|from the perspective|as if (?:having|speaking))",
    re.I,
)
_MATH_RE = re.compile(r"(latex|mathematical expression|equation|\bmath\b)", re.I)
_CODE_RE = re.compile(r"(code block|code example|\bcode\b|indentation|syntax)", re.I)
# structure deliberately EXCLUDES bare "format" so "LaTeX formatting" stays math-only.
_STRUCT_RE = re.compile(
    r"(markdown|heading|bullet|section|numbered|paragraph|bold|italic|demarcate|"
    r"structur|organi[sz]|\blist\b|line break|readable|concise|easy to (?:read|follow))",
    re.I,
)

_ASPECT_CLAUSE = {
    "persona": "address the student in the second person (conversational tutor voice)",
    "structure": "use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity",
    "math": "render mathematical expressions in LaTeX",
    "code": "present code in fenced code blocks with correct syntax",
}
_ASPECT_ORDER = ("persona", "structure", "math", "code")


def presentation_aspects(text: str) -> list[str]:
    found = set()
    if _PERSONA_RE.search(text):
        found.add("persona")
    if _MATH_RE.search(text):
        found.add("math")
    if _CODE_RE.search(text):
        found.add("code")
    if _STRUCT_RE.search(text):
        found.add("structure")
    return [a for a in _ASPECT_ORDER if a in found]


def compose_presentation_text(aspects: list[str]) -> str:
    parts = [_ASPECT_CLAUSE[a] for a in aspects]
    body = "; ".join(parts) if parts else "be clearly and readably presented"
    return (f"The response should follow tutoring presentation conventions: {body}. "
            "(Style/presentation only; not a content, diagnosis, or scaffolding skill.)")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def is_all_zero(q: dict) -> bool:
    return not (q.get("content", 0) or q.get("diagnosis", 0) or q.get("scaffolding", 0))


SKILL_DESC = {
    "content": "domain-accurate knowledge or computation",
    "diagnosis": "reading and addressing the student's specific work or stated confusion",
    "scaffolding": "structuring or sequencing the help pedagogically",
}


def regen_rationale(text: str, q: dict) -> str:
    """Build a child-specific q_rationale from the child's own text + q_mapping,
    so split children no longer quote the old bundled criterion. Deterministic
    template (method regenerated_template_v1), NOT an LLM re-label."""
    skills = [s for s in ("content", "diagnosis", "scaffolding") if q.get(s, 0)]
    if not skills:
        return ("This criterion is a tone/format/orphan check; no content, diagnosis, "
                "or scaffolding skill is strictly required.\n"
                "(No skill strictly required for this criterion.)")
    lines = [f"This atomized criterion requires: {', '.join(skills)}.", "Skill justifications:"]
    for s in skills:
        lines.append(
            f"- {s}: evidence: {text} | why required: satisfying this criterion depends on "
            f"{SKILL_DESC[s]}. | counterfactual: a tutor lacking {s} skill could not satisfy it."
        )
    return "\n".join(lines)


def child_from_parent(parent: dict, text: str, override: dict, policy: str) -> dict:
    c = copy.deepcopy(parent)
    c["criterion"] = text
    if "q_mapping" in override:
        c["q_mapping"] = override["q_mapping"]
    if "criticality" in override:
        c["criticality"] = override["criticality"]
    c["q_rationale"] = regen_rationale(text, c.get("q_mapping", {}))
    c["curation"] = {
        "policy_version": policy,
        "op": "split",
        "from": parent["criterion_id"],
        "original_criterion_id": parent["criterion_id"],
        "rationale": "regenerated_template_v1",
    }
    return c


def default_pres_aspects(subject: str | None) -> list[str]:
    """Default presentation aspects for a scenario with no style orphan to learn
    from: persona + structure everywhere, LaTeX for math-heavy subjects, code for CS."""
    subj = (subject or "").lower()
    have = {"persona", "structure"}
    if subj in {"calculus", "statistics", "physics", "chemistry"}:
        have.add("math")
    if subj in {"computer_science", "computer science", "cs"}:
        have.add("code")
    return [a for a in _ASPECT_ORDER if a in have]


def make_standard_format(scenario_id: str, aspects: list[str], removed_ids: list[str],
                         policy: str, op: str = "format_consolidated") -> dict:
    return {
        "criterion_id": f"{scenario_id}_cPENDING",
        "scenario_id": scenario_id,
        "criterion": compose_presentation_text(aspects),
        "expected_evidence": [],
        "scoring_type": "binary",
        "score_anchors": None,
        "primary_skill": None,
        "q_mapping": {"content": 0, "diagnosis": 0, "scaffolding": 0},
        "q_rationale": "Style/presentation only; consolidated from format/persona orphans.",
        "criticality": "not_critical",
        "objectivity": "objective",
        "explicitness": "implicit",
        "source": "TutorBench",
        "status": "approved",
        "version": "1.0",
        "difficulty": 0.0,
        "discrimination": {"content": 0.0, "diagnosis": 0.0, "scaffolding": 0.0},
        "irt_params": {"source": "curation_default", "method": "format_consolidation", "version": "1.0"},
        "optional": True,
        "dimension": "style_surface",
        "judge_guidance": (
            "Presentation/style only. Not a content, diagnosis, or scaffolding skill; "
            "excluded from skill calibration. Non-gating: mark N/A if it does not apply."
        ),
        "curation": {
            "policy_version": policy,
            "op": op,
            "from": removed_ids,
            "aspects": aspects,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="curation/pilot_edits.json")
    ap.add_argument("--rubrics", default="data/TutorBench/rubrics_qmatrix_final.jsonl")
    ap.add_argument("--scenarios", default="data/TutorBench/scenarios.jsonl")
    ap.add_argument("--out-dir", default="data/TutorBench/curated")
    ap.add_argument("--report-dir", default="curation")
    ap.add_argument("--ensure-presentation", action=argparse.BooleanOptionalAction, default=True,
                    help="append one optional style_surface presentation criterion to any "
                         "scenario that lacks one (normalize presentation coverage)")
    args = ap.parse_args()

    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    policy = spec["meta"]["policy_version"]
    std_format_text = spec["meta"]["standard_format_criterion"]
    ops = {o["id"]: o for o in spec["ops"]}

    rubrics = load_jsonl(Path(args.rubrics))
    scenarios = load_jsonl(Path(args.scenarios))
    scen_by_id = {s["scenario_id"]: s for s in scenarios}

    # group rubrics by scenario, preserving file order
    groups: "OrderedDict[str, list[dict]]" = OrderedDict()
    for r in rubrics:
        groups.setdefault(r["scenario_id"], []).append(r)

    # which scenarios actually change (any op that is not a bare 'keep')
    CHANGING = {"split", "soften", "rescope_optional", "drop"}
    edited_scenarios = set()
    for o in spec["ops"]:
        if o["op"] in CHANGING:
            sid = o["id"].rsplit("_c", 1)[0]
            edited_scenarios.add(sid)

    # Bank-wide presentation consolidation: any scenario containing >=1 all-zero
    # style/persona orphan is processed so its orphans consolidate into ONE
    # optional presentation criterion (aspect-aware), not just edited scenarios.
    scenarios_with_orphans = {
        sid for sid, crits in groups.items()
        if any(is_all_zero(c.get("q_mapping", {}) or {})
               and FORMAT_ORPHAN_RE.search(c.get("criterion", "") or "")
               for c in crits)
    }
    processing = edited_scenarios | scenarios_with_orphans

    # validate op ids exist
    all_ids = {r["criterion_id"] for r in rubrics}
    missing = [cid for cid in ops if cid not in all_ids]
    if missing:
        raise SystemExit(f"Spec references unknown criterion_ids: {missing}")

    report_rows = []
    out_rubrics: list[dict] = []

    for sid, crits in groups.items():
        if sid not in processing:
            out_rubrics.extend(crits)  # untouched (includes bare-keep scenarios)
            # still log explicit keeps for traceability
            for c in crits:
                if c["criterion_id"] in ops and ops[c["criterion_id"]]["op"] == "keep":
                    report_rows.append({
                        "scenario_id": sid, "op": "keep", "old_id": c["criterion_id"],
                        "new_id": c["criterion_id"], "old_text": c["criterion"],
                        "new_text": c["criterion"], "note": ops[c["criterion_id"]].get("note", ""),
                    })
            continue

        # build new ordered list for this scenario
        new_list: list[dict] = []
        pending_report: list[tuple[str, str, dict, str]] = []  # (op, old_id, new_dict_or_none, old_text)
        for c in crits:
            cid = c["criterion_id"]
            op = ops.get(cid, {}).get("op")
            if op == "split":
                for ch in ops[cid]["children"]:
                    override = {k: ch[k] for k in ("q_mapping", "criticality") if k in ch}
                    child = child_from_parent(c, ch["text"], override, policy)
                    new_list.append(child)
                    pending_report.append(("split", cid, child, c["criterion"]))
            elif op == "soften":
                nc = copy.deepcopy(c)
                nc["criterion"] = ops[cid]["text"]
                nc["curation"] = {"policy_version": policy, "op": "soften",
                                  "from": cid, "original_criterion_id": cid}
                new_list.append(nc)
                pending_report.append(("soften", cid, nc, c["criterion"]))
            elif op == "rescope_optional":
                nc = copy.deepcopy(c)
                nc["criterion"] = ops[cid]["text"]
                nc["criticality"] = "not_critical"
                nc["optional"] = True
                nc["judge_guidance"] = ops[cid]["judge_guidance"]
                nc["curation"] = {"policy_version": policy, "op": "rescope_optional",
                                  "from": cid, "original_criterion_id": cid}
                new_list.append(nc)
                pending_report.append(("rescope_optional", cid, nc, c["criterion"]))
            elif op == "drop":
                pending_report.append(("drop", cid, None, c["criterion"]))
            elif op == "keep":
                new_list.append(c)  # unchanged content, will still be renumbered
                pending_report.append(("keep", cid, c, c["criterion"]))
            else:
                new_list.append(c)  # untouched criterion in an edited scenario -> renumber only

        # aspect-aware presentation consolidation within this scenario
        removed_fmt = [x for x in new_list if is_all_zero(x.get("q_mapping", {}))
                       and FORMAT_ORPHAN_RE.search(x.get("criterion", ""))]
        if len(removed_fmt) >= 1:
            removed_ids = [x["criterion_id"] for x in removed_fmt]
            seen: set[str] = set()
            for x in removed_fmt:
                seen.update(presentation_aspects(x.get("criterion", "") or ""))
            aspects = [a for a in _ASPECT_ORDER if a in seen]
            new_list = [x for x in new_list if x not in removed_fmt]
            std = make_standard_format(sid, aspects, removed_ids, policy)
            new_list.append(std)
            pending_report.append(("format_consolidated", ",".join(removed_ids), std,
                                   f"{len(removed_ids)} orphan(s) -> aspects={aspects or ['generic']}"))

        # renumber sequentially and update ids
        for idx, c in enumerate(new_list, start=1):
            new_id = f"{sid}_c{idx:02d}"
            old_id = c.get("criterion_id", "")
            if "curation" not in c:
                c["curation"] = {"policy_version": policy, "op": "renumbered",
                                 "original_criterion_id": old_id}
            elif "original_criterion_id" not in c["curation"]:
                c["curation"]["original_criterion_id"] = old_id
            c["criterion_id"] = new_id

        out_rubrics.extend(new_list)

        # update scenario.criterion_ids
        if sid in scen_by_id:
            scen_by_id[sid]["criterion_ids"] = [c["criterion_id"] for c in new_list]
            scen_by_id[sid].setdefault("curation", {})["policy_version"] = policy

        # finalize report rows (map pending new dicts to their final ids)
        for op_name, old_id, ndict, old_text in pending_report:
            report_rows.append({
                "scenario_id": sid, "op": op_name, "old_id": old_id,
                "new_id": (ndict["criterion_id"] if ndict else "(removed)"),
                "old_text": old_text,
                "new_text": (ndict["criterion"] if ndict else ""),
                "note": ops.get(old_id if old_id in ops else "", {}).get("note", ""),
            })

    # normalize presentation coverage: every scenario gets exactly one optional
    # style_surface presentation criterion (appended, existing ids untouched).
    added_pres = 0
    if args.ensure_presentation:
        out_by_scen: "OrderedDict[str, list[dict]]" = OrderedDict()
        for c in out_rubrics:
            out_by_scen.setdefault(c["scenario_id"], []).append(c)
        for sid, clist in out_by_scen.items():
            if any(c.get("dimension") == "style_surface" for c in clist):
                continue
            subject = scen_by_id.get(sid, {}).get("subject")
            idxs = []
            for c in clist:
                m = re.search(r"_c(\d+)$", c["criterion_id"])
                if m:
                    idxs.append(int(m.group(1)))
            nextn = (max(idxs) + 1) if idxs else 1
            pres = make_standard_format(sid, default_pres_aspects(subject), [], policy,
                                        op="presentation_added")
            pres["criterion_id"] = f"{sid}_c{nextn:02d}"
            clist.append(pres)
            added_pres += 1
            if sid in scen_by_id:
                scen_by_id[sid]["criterion_ids"] = [c["criterion_id"] for c in clist]
            report_rows.append({
                "scenario_id": sid, "op": "presentation_added", "old_id": "",
                "new_id": pres["criterion_id"], "old_text": "",
                "new_text": pres["criterion"], "note": "normalize presentation coverage",
            })
        out_rubrics = [c for clist in out_by_scen.values() for c in clist]

    # write curated clones
    out_dir = Path(args.out_dir)
    write_jsonl(out_dir / "rubrics_qmatrix_curated.jsonl", out_rubrics)
    write_jsonl(out_dir / "scenarios_curated.jsonl", scenarios)

    # write report
    rep_dir = Path(args.report_dir)
    rep_dir.mkdir(parents=True, exist_ok=True)
    csv_path = rep_dir / "pilot_change_report.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["scenario_id", "op", "old_id", "new_id",
                                           "old_text", "new_text", "note"])
        w.writeheader()
        for row in report_rows:
            w.writerow({k: str(row.get(k, "")).replace("\n", " ") for k in w.fieldnames})

    md = ["# Pilot curation change report", "",
          f"- policy: `{policy}`",
          f"- source rubrics: `{args.rubrics}` ({len(rubrics)} criteria)",
          f"- curated rubrics: `{out_dir / 'rubrics_qmatrix_curated.jsonl'}` ({len(out_rubrics)} criteria)",
          f"- scenarios edited by ops: {len(edited_scenarios)}",
          f"- scenarios changed (incl. presentation consolidation): {len(processing)}",
          f"- edit operations logged: {len(report_rows)}", ""]
    by_op: dict[str, int] = {}
    for row in report_rows:
        by_op[row["op"]] = by_op.get(row["op"], 0) + 1
    md.append("## Ops summary")
    for k, v in sorted(by_op.items()):
        md.append(f"- {k}: {v}")
    md.append("")
    md.append("## Per-change detail")
    cur_scenario = None
    for row in report_rows:
        if row["scenario_id"] != cur_scenario:
            cur_scenario = row["scenario_id"]
            md.append(f"\n### {cur_scenario}")
        old_t = (row["old_text"][:180] + "...") if len(row["old_text"]) > 180 else row["old_text"]
        md.append(f"- **{row['op']}** `{row['old_id']}` -> `{row['new_id']}`")
        if row["op"] in ("split", "rescope_optional", "soften", "format_consolidated"):
            md.append(f"    - was: {old_t}")
            if row["new_text"]:
                md.append(f"    - now: {row['new_text']}")
    (rep_dir / "pilot_change_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"Source criteria:  {len(rubrics)}")
    print(f"Curated criteria: {len(out_rubrics)}  (delta {len(out_rubrics) - len(rubrics):+d})")
    print(f"Scenarios edited by ops: {len(edited_scenarios)}  "
          f"| changed incl. consolidation: {len(processing)}")
    print(f"Presentation criteria added (coverage normalization): {added_pres}")
    print("Ops:", {k: by_op[k] for k in sorted(by_op)})
    print(f"\nCurated rubrics -> {out_dir / 'rubrics_qmatrix_curated.jsonl'}")
    print(f"Curated scenarios -> {out_dir / 'scenarios_curated.jsonl'}")
    print(f"Report (md)  -> {rep_dir / 'pilot_change_report.md'}")
    print(f"Report (csv) -> {rep_dir / 'pilot_change_report.csv'}")


if __name__ == "__main__":
    main()
