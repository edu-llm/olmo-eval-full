"""Merge two or more EduBench task-type skills into one, and rebuild the bank + graph.

    reads   the raw HuggingFace en_data/<stem>.jsonl files (via ingest_edubench, cached
            locally by huggingface_hub after the first fetch)
    writes  <out-dir>/scenarios.jsonl + .json, rubrics.jsonl + .json, skills_meta.json,
            README.md, skill_graph.json, edubench_skill_graph.png

A merge has two parts:

  q-matrix    for each of the 12 metrics, the merged skill's applicability is the UNION
              of its members' applicability -- if either member matched a criterion, the
              merged skill matches it.
  scenarios   concatenate the members' raw records (e.g. QG.jsonl ++ TMG.jsonl) and give
              EVERY one of them the merged skill's (union) criteria set, uniformly,
              regardless of which member file a scenario actually came from -- because
              Table 7 assigns criteria per task-type column, not per scenario.

This starts from ingest_edubench.py's CURRENT `TASKS`/`METRICS` (i.e. today's Table 7,
including any prior manual edits) and only changes what merging requires. It does not
import huggingface_hub itself, and does not modify ingest_edubench.py -- it reuses that
module's already-correct pure pieces (`resolve_paths`, `read_records`, `build_criterion`,
`write`, the placeholder constants) and re-implements the per-record loop, generalized to
let one logical skill draw from more than one raw stem.

Only `--q-mapping row` semantics are supported (a metric's q_mapping is its Table 7 row).
`onehot`'s per-response one-hot loading has no natural "union" reading across a merge
group, so it is out of scope here.

IMPORTANT positional consequence of "just concatenate the files": scenario_id/criterion_id
numbers are `eb_<i>`, assigned by POSITION in the whole-bank concatenation. A merge group
becomes one contiguous run positioned at its earliest member's slot in TASKS order, so
every original record after that point shifts to a different `eb_<i>` than it had in the
unmerged bank -- e.g. merging QG+TMG makes AG's block (which used to sit between them)
start 1185 (TMG's record count) records later than before. Content for every unmerged
skill is otherwise unchanged; only its numbering shifts if it's sequenced after a merge.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import ingest_edubench as ie
except ModuleNotFoundError as exc:
    sys.exit(
        f"cannot import scripts/ingest_edubench.py ({exc}).\n"
        "It imports huggingface_hub at module level; run this with an interpreter that "
        "has it installed, e.g. llm-from-scratch/Scripts/python.exe"
    )
import build_edubench_skill_graph as skillgraph

ROOT = ie.ROOT
OUT_DIR = ROOT / "data" / "EduBench" / "augmented_qmat" / "merged"
TASKS_BY_SLUG = {t["slug"]: t for t in ie.TASKS}


def parse_groups(specs: list[str]) -> list[list[str]]:
    """--merge a,b --merge c,d -> [[a,b],[c,d]], validated against ie.TASKS."""
    groups = [[s.strip() for s in spec.split(",") if s.strip()] for spec in specs]
    seen: set[str] = set()
    for g in groups:
        if len(g) < 2:
            sys.exit(f"--merge {','.join(g)}: a merge group needs at least 2 slugs")
        for slug in g:
            if slug not in TASKS_BY_SLUG:
                sys.exit(f"--merge: unknown slug {slug!r}, expected one of "
                         f"{sorted(TASKS_BY_SLUG)}")
            if slug in seen:
                sys.exit(f"--merge: {slug!r} appears in more than one group")
            seen.add(slug)
    return groups


def build_skill_defs(groups: list[list[str]]) -> list[dict]:
    """One entry per skill in the OUTPUT axis, in ie.TASKS order (a merge group appears
    once, at its earliest member's position). Each entry has slug/label/display/core_task
    (all consumable by ie.build_criterion unchanged) plus `members`: the original
    ie.TASKS entries that feed it -- 1 for a passthrough, >=2 for an actual merge."""
    group_of = {slug: tuple(g) for g in groups for slug in g}
    emitted: set[tuple[str, ...]] = set()
    defs = []
    for t in ie.TASKS:
        slug = t["slug"]
        gid = group_of.get(slug)
        if gid is None:
            defs.append({"slug": slug, "label": t["stem"], "display": t["display"],
                        "core_task": t["core_task"], "members": [t]})
            continue
        if gid in emitted:
            continue
        emitted.add(gid)
        members = [TASKS_BY_SLUG[s] for s in gid]
        defs.append({
            "slug": "+".join(m["slug"] for m in members),
            "label": "+".join(m["stem"] for m in members),
            "display": " + ".join(m["display"] for m in members),
            "core_task": "; or ".join(m["core_task"] for m in members),
            "members": members,
        })
    return defs


def merge_metrics(skill_defs: list[dict]) -> list[dict]:
    """ie.METRICS with every `applies` set rewritten: original-slug -> merged-slug,
    unioned per merge group. A metric applies to the merged slug iff it applied to ANY
    member -- the union rule."""
    slug_to_merged = {m["slug"]: sd["slug"] for sd in skill_defs for m in sd["members"]}
    merged = []
    for metric in ie.METRICS:
        nm = dict(metric)
        nm["applies"] = {slug_to_merged[s] for s in metric["applies"]}
        merged.append(nm)
    return merged


def metrics_for(slug: str, metrics: list[dict]) -> list[dict]:
    """ie.metrics_for, parameterized on the (possibly merged) metrics list."""
    return [m for m in metrics if slug in m["applies"]]


def build(skill_defs: list[dict], metrics: list[dict]) -> tuple[list[dict], list[dict], dict]:
    """ie.build(), generalized: a skill may draw from more than one raw stem. Mirrors
    ie.build()'s per-record field mapping exactly, substituting the skill def (`sd`) for
    ie.build()'s `task` where criteria/use_case are concerned, but the ORIGINAL per-stem
    task (`member`) for reference_key/grade-band extraction, since that's a property of
    the raw file, not of the (possibly merged) skill it now belongs to."""
    paths = ie.resolve_paths()
    skills = [sd["slug"] for sd in skill_defs]
    scenarios: list[dict] = []
    rubrics: list[dict] = []
    stats = {"counts": {}, "ref_missing": Counter(), "grade_missing": Counter()}

    index = 0
    for sd in skill_defs:
        allocated = metrics_for(sd["slug"], metrics)
        for member in sd["members"]:
            stem = member["stem"]
            records = ie.read_records(paths[stem])
            stats["counts"][stem] = len(records)

            for row in records:
                info = row.get("information")
                if not isinstance(info, dict):
                    sys.exit(f"{stem} record {index}: `information` is "
                             f"{type(info).__name__}")

                grade = None
                for key in ie.GRADE_KEYS:
                    value = info.get(key)
                    if isinstance(value, str) and value.strip():
                        grade = value.strip()
                        break
                if grade is None:
                    stats["grade_missing"][stem] += 1

                subject = info.get(ie.SUBJECT_KEY)
                subject = subject.strip() if isinstance(subject, str) else None

                reference = ""
                if member["reference_key"]:
                    value = info.get(member["reference_key"])
                    if isinstance(value, str) and value.strip():
                        reference = value.strip()
                    else:
                        stats["ref_missing"][stem] += 1

                sid = f"eb_{index}"
                criterion_ids = [f"{sid}_c{n:02d}" for n in range(1, len(allocated) + 1)]

                scenarios.append({
                    "scenario_id": sid,
                    "use_case": sd["slug"],
                    "subject": subject,
                    "grade_band": grade,
                    "modality": "text",
                    "prompt": row.get("prompt") or "",
                    "conversation_context": [],
                    "reference_solution": reference,
                    "criterion_ids": criterion_ids,
                    "source": f"Edubench_{stem} JSONL",
                    "split": ie.SPLIT,
                    "version": ie.VERSION,
                })

                for cid, metric in zip(criterion_ids, allocated):
                    rubrics.append({
                        "criterion_id": cid,
                        "scenario_id": sid,
                        "criterion": ie.build_criterion(metric, sd),
                        "expected_evidence": [],
                        "scoring_type": "binary",
                        "score_anchors": None,
                        "q_mapping": {s: int(s in metric["applies"]) for s in skills},
                        "q_rationale": ie.Q_RATIONALE,
                        "criticality": ie.CRITICALITY,
                        "objectivity": ie.OBJECTIVITY,
                        "explicitness": ie.EXPLICITNESS,
                        "source": ie.SOURCE_URL,
                        "status": "approved",
                        "version": ie.VERSION,
                        "difficulty": None,
                        "discrimination": None,
                    })
                index += 1

    return scenarios, rubrics, stats


def validate(scenarios: list[dict], rubrics: list[dict], stats: dict,
            skill_defs: list[dict], metrics: list[dict]) -> list[str]:
    """ie.validate(), adapted to the merged axis (same checks, same intent)."""
    errs: list[str] = []
    skills = [sd["slug"] for sd in skill_defs]

    for name, records, keys in (("scenario", scenarios, ie.SCENARIO_KEYS),
                                 ("rubric", rubrics, ie.RUBRIC_KEYS)):
        for r in records:
            if list(r.keys()) != keys:
                errs.append(f"{name} {list(r.values())[0]}: key set/order mismatch")
                break

    sids = [s["scenario_id"] for s in scenarios]
    cids = [r["criterion_id"] for r in rubrics]
    if len(set(sids)) != len(sids):
        errs.append("duplicate scenario_id")
    if len(set(cids)) != len(cids):
        errs.append("duplicate criterion_id")

    declared = {c for s in scenarios for c in s["criterion_ids"]}
    actual = set(cids)
    if declared != actual:
        errs.append(f"criterion_ids mismatch: {len(declared - actual)} declared-but-missing, "
                    f"{len(actual - declared)} present-but-undeclared")
    orphans = {r["scenario_id"] for r in rubrics} - set(sids)
    if orphans:
        errs.append(f"{len(orphans)} rubrics reference unknown scenarios")

    for s in scenarios:
        if not (s["prompt"] or "").strip():
            errs.append(f"{s['scenario_id']}: empty prompt")
        if not s["criterion_ids"]:
            errs.append(f"{s['scenario_id']}: no criteria")
        if s["use_case"] not in skills:
            errs.append(f"{s['scenario_id']}: use_case {s['use_case']!r} not in the merged axis")

    for r in rubrics:
        if not (r["criterion"] or "").strip():
            errs.append(f"{r['criterion_id']}: empty criterion")
        if any(t in r["criterion"] for t in ("<", ">", "{", "}")):
            errs.append(f"{r['criterion_id']}: unresolved placeholder in criterion")
        q = r["q_mapping"]
        if list(q.keys()) != skills:
            errs.append(f"{r['criterion_id']}: q_mapping keys/order != the merged axis")
        if set(q.values()) - {0, 1}:
            errs.append(f"{r['criterion_id']}: q_mapping values must be 0/1")
        if sum(q.values()) == 0:
            errs.append(f"{r['criterion_id']}: q_mapping is all zeros")
        if r["difficulty"] is not None or r["discrimination"] is not None:
            errs.append(f"{r['criterion_id']}: difficulty/discrimination must ship as null")

    per_scenario = Counter(r["scenario_id"] for r in rubrics)
    by_skill = {s["scenario_id"]: s["use_case"] for s in scenarios}
    seen: dict[str, set] = {}
    for sid, n in per_scenario.items():
        seen.setdefault(by_skill[sid], set()).add(n)
    for sd in skill_defs:
        want = len(metrics_for(sd["slug"], metrics))
        got = seen.get(sd["slug"], set())
        if got != {want}:
            errs.append(f"{sd['slug']}: criteria per scenario {sorted(got)} != Table 7 "
                       f"total {want}")

    if stats["ref_missing"]:
        errs.append(f"reference_solution key missing for: {dict(stats['ref_missing'])}")

    return errs


def write_skills_meta(skill_defs: list[dict], path: Path) -> None:
    meta = {sd["slug"]: {"label": sd["label"], "display": sd["display"],
                         "members": [m["slug"] for m in sd["members"]]}
            for sd in skill_defs}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8", newline="\n")


def write_readme(skill_defs: list[dict], groups: list[list[str]], out_dir: Path) -> None:
    merged = [sd for sd in skill_defs if len(sd["members"]) > 1]
    lines = [
        "# Merged EduBench skill axis",
        "",
        "Built by [`scripts/build_edubench_merged.py`](../../../../scripts/build_edubench_merged.py) "
        "from `ingest_edubench.py`'s current (already-augmented) `TASKS`/`METRICS`.",
        "",
        "## Merge groups",
        "",
    ]
    for sd in merged:
        members = ", ".join(f"`{m['slug']}`" for m in sd["members"])
        lines.append(f"- **`{sd['slug']}`** ({sd['display']}) = {members}")
    lines += [
        "",
        "## Rule",
        "",
        "- **q-matrix**: a merged skill's applicability for each of the 12 metrics is the "
        "**union** of its members' applicability -- if either member matched a criterion, "
        "the merged skill matches it.",
        "- **scenarios/rubrics**: the members' raw records are concatenated; every "
        "resulting scenario, regardless of which member file it came from, gets the "
        "merged skill's (union) criteria set uniformly.",
        "- `source` keeps each record's ORIGINAL per-stem value; only `use_case` changes.",
        "- `scenario_id`/`criterion_id` numbers are positional (`eb_<i>` over the whole "
        "concatenation), so a merge shifts the numbering of any skill sequenced after it "
        "-- content for every unmerged skill is otherwise unchanged.",
        "- Only `--q-mapping row` semantics are supported.",
        "",
        "See `skills_meta.json` for the full slug -> {label, display, members} mapping.",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def run_graph_pipeline(out_dir: Path) -> int:
    argv = [
        "build_edubench_skill_graph.py",
        "--rubrics", str(out_dir / "rubrics.jsonl"),
        "--scenarios", str(out_dir / "scenarios.jsonl"),
        "--out", str(out_dir / "skill_graph.json"),
        "--plot", str(out_dir / "edubench_skill_graph.png"),
        "--skills-meta", str(out_dir / "skills_meta.json"),
        "--no-qmatrix-out", "--no-grouped-qmatrix-out",
    ]
    old_argv = sys.argv
    sys.argv = argv
    try:
        return skillgraph.main()
    finally:
        sys.argv = old_argv


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--merge", action="append", default=[], metavar="SLUG1,SLUG2[,...]",
                   help="one merge group per flag, e.g. "
                        "--merge question_generation,material_generation")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--name", default=None,
                   help="nest output under <out-dir>/<name>/ instead of writing directly "
                        "into --out-dir, so more than one merge experiment can coexist")
    p.add_argument("--dry-run", action="store_true", help="build and validate, write nothing")
    p.add_argument("--os-trust-store", action="store_true",
                   help="see ingest_edubench.py --os-trust-store")
    args = p.parse_args()
    if not args.merge:
        sys.exit("--merge is required (repeatable): at least one group of >=2 slugs")
    if args.os_trust_store:
        ie.use_os_trust_store()

    out_dir = args.out_dir / args.name if args.name else args.out_dir

    groups = parse_groups(args.merge)
    skill_defs = build_skill_defs(groups)
    metrics = merge_metrics(skill_defs)
    skills = [sd["slug"] for sd in skill_defs]

    print(f"merge groups: {groups}")
    print(f"resulting axis: {len(skills)} skills -- {skills}")

    scenarios, rubrics, stats = build(skill_defs, metrics)
    print(f"\nscenarios: {len(scenarios)}   rubrics: {len(rubrics)}")
    for sd in skill_defs:
        n = sum(1 for s in scenarios if s["use_case"] == sd["slug"])
        expect = sum(m["n_records"] for m in sd["members"])
        flag = "" if n == expect else f"  <-- expected {expect}"
        print(f"  {sd['slug']:<45} {n:>5} scenarios "
             f"({len(metrics_for(sd['slug'], metrics))} criteria each){flag}")

    err = validate(scenarios, rubrics, stats, skill_defs, metrics)
    print(f"\nvalidate: {'PASS' if not err else f'{len(err)} ERROR(S)'}")
    for e in err:
        print(f"  {e}")
    if err:
        return 1

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    ie.write("scenarios", scenarios, out_dir)
    ie.write("rubrics", rubrics, out_dir)
    write_skills_meta(skill_defs, out_dir / "skills_meta.json")
    write_readme(skill_defs, groups, out_dir)
    print(f"wrote {out_dir / 'skills_meta.json'}")
    print(f"wrote {out_dir / 'README.md'}")

    print("\n--- graph pipeline ---")
    return run_graph_pipeline(out_dir)


if __name__ == "__main__":
    sys.exit(main())
