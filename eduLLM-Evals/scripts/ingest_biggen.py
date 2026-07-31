"""Convert the BiGGen-Bench dataset into Scenario + Rubric Schema JSONL.

Source: https://huggingface.co/datasets/prometheus-eval/BiGGen-Bench
(frozen snapshot at data/BiGGen/biggen_source.json, 765 instances).

BiGGen ships one 5-point (polytomous) instance-level rubric per prompt. This repo
scores binary, so each instance is **atomized**: its single holistic criterion is
decomposed into several standalone yes/no criteria (one rubric row each), following
the atomization ruleset in data/BiGGen/ATOMIZATION_SPEC.md (outcome-first,
path-agnostic, derivable targets pinned). The atoms are supplied as overlays:

  - data/BiGGen/atoms_deterministic.jsonl  -- the 18 edited planning/travel_plan and
    tool_usage/item_recommendation instances, decomposed by rule from their
    requirement lists.
  - data/BiGGen/atoms_llm_<capability>.jsonl  -- the remaining English instances,
    authored per the spec (method="authored_v2").

The native holistic rubric (criterion text + the five 1-5 level descriptions) is NOT
discarded: it is preserved per scenario under `native_rubric` as a provenance sidecar
so the polytomous form stays reversible and auditable.

Decisions baked in here (see the handoff discussion):
  - multilingual: BiGGen exposes multilingual as a *capability* (70 instances),
    not a per-row language tag, so it is EXCLUDED at ingest (held for later).
  - excluded instances: two English instances are dropped because they are broken at
    the answer level (not merely a bad anchor): `reasoning_math_proof_4` (theorem
    false for even n) and `reasoning_high_school_mwp_7` (self-contradictory problem).
    They carry no atoms in the overlay.
  - reference answers: HELD. `reference_solution` is null; the native BiGGen
    reference is preserved verbatim under `native_reference_answer`.
  - skill axis: UNDECIDED. `q_mapping`/`primary_skill` null; MIRT
    difficulty/discrimination/irt_params are appended later by assign_irt_params.py.
  - controlled edits: the planning/travel_plan and tool_usage/item_recommendation
    input/anchor reframes (biggen_edit_v1) are overlaid from their *_edited.jsonl
    files; each edited scenario carries its `edit_provenance` block and the
    `native_rubric` reflects the edited anchors.

Run:
    uv run python scripts/edit_biggen_travel_plan.py
    uv run python scripts/edit_biggen_item_recommendation.py
    uv run python scripts/atomize_biggen_deterministic.py
    uv run python scripts/ingest_biggen.py
"""
from __future__ import annotations

import glob
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "BiGGen"
SRC = DATA_DIR / "biggen_source.json"
EDIT_FILES = [
    DATA_DIR / "travel_plan_edited.jsonl",
    DATA_DIR / "item_recommendation_edited.jsonl",
]
DET_ATOMS = DATA_DIR / "atoms_deterministic.jsonl"
LLM_ATOM_GLOB = str(DATA_DIR / "atoms_llm_*.jsonl")

SOURCE_URL = "https://huggingface.co/datasets/prometheus-eval/BiGGen-Bench"
SPLIT = "calibration"   # pipeline-role label, not an upstream split
VERSION = "1.0"
EXCLUDE_CAPABILITY = "multilingual"   # held for later; ingested English-only
# Dropped at ingest: broken at the answer level, not just a bad 1-5 anchor.
EXCLUDED_IDS = {"reasoning_math_proof_4", "reasoning_high_school_mwp_7"}
ANCHOR_LEVELS = [1, 2, 3, 4, 5]
PASS_THRESHOLD = 4      # bar on the preserved native 1-5 scale (sidecar only)

SCENARIO_KEYS = [
    "scenario_id", "source_id", "use_case", "subject", "task", "grade_band",
    "modality", "system_prompt", "prompt", "conversation_context",
    "reference_solution", "native_reference_answer", "criterion_ids",
    "native_rubric", "atomization", "edit_provenance",
    "source", "split", "version",
]
RUBRIC_KEYS = [
    "criterion_id", "scenario_id", "criterion", "expected_evidence",
    "scoring_type", "score_anchors", "primary_skill", "q_mapping", "q_rationale",
    "capability", "task", "criticality", "objectivity", "explicitness",
    "source", "status", "version",
]

# json.dumps(ensure_ascii=False) leaves U+2028/U+2029 raw; BiGGen carries non-Latin
# text, so escape them back (JS JSON.parse / str.splitlines treat them as breaks).
LINE_SEPS = {0x2028: "\\u2028", 0x2029: "\\u2029"}


def scenario_id(index: int) -> str:
    return f"bgb_{index:04d}"


def native_rubric(rubric: dict) -> dict:
    """Preserve the holistic criterion + 5 polytomous levels as a provenance sidecar."""
    return {
        "criteria": rubric["criteria"],
        "scale_min": ANCHOR_LEVELS[0],
        "scale_max": ANCHOR_LEVELS[-1],
        "pass_threshold": PASS_THRESHOLD,
        "levels": {str(n): rubric[f"score{n}_description"] for n in ANCHOR_LEVELS},
    }


def load_edits() -> dict[str, dict]:
    edits: dict[str, dict] = {}
    for path in EDIT_FILES:
        if not path.exists():
            raise SystemExit(f"missing edit overlay {path}; run the edit scripts first")
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                edits[rec["id"]] = rec
    return edits


def load_atoms() -> dict[str, dict]:
    """Merge the deterministic + LLM-authored atom overlays, keyed by source id."""
    atoms: dict[str, dict] = {}
    paths = [DET_ATOMS, *sorted(Path(p) for p in glob.glob(LLM_ATOM_GLOB))]
    for path in paths:
        if not path.exists():
            raise SystemExit(f"missing atom overlay {path}; run the atomize step first")
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                sid = rec["id"]
                if sid in atoms:
                    raise SystemExit(f"duplicate atom overlay entry for {sid}")
                if not rec.get("atoms"):
                    raise SystemExit(f"{sid}: empty atom list")
                atoms[sid] = rec
    return atoms


def build() -> tuple[list[dict], list[dict], dict]:
    source = json.loads(SRC.read_text(encoding="utf-8"))
    edits = load_edits()
    atoms = load_atoms()

    scenarios: list[dict] = []
    rubrics: list[dict] = []
    stats = {
        "excluded_multilingual": 0,
        "excluded_broken": 0,
        "edits_applied": 0,
        "flagged": 0,
    }
    index = 0

    for src in source:
        if src["capability"] == EXCLUDE_CAPABILITY:
            stats["excluded_multilingual"] += 1
            continue
        if src["id"] in EXCLUDED_IDS:
            stats["excluded_broken"] += 1
            continue

        overlay = atoms.get(src["id"])
        if overlay is None:
            raise SystemExit(f"{src['id']}: no atoms and not in EXCLUDED_IDS")

        rec = edits.get(src["id"], src)          # overlay controlled edits
        edited = src["id"] in edits
        if edited:
            stats["edits_applied"] += 1

        flagged = bool(overlay.get("flagged"))
        if flagged:
            stats["flagged"] += 1

        sid = scenario_id(index)
        index += 1
        atom_texts = overlay["atoms"]
        cids = [f"{sid}_c{i:02d}" for i in range(1, len(atom_texts) + 1)]

        scenarios.append({
            "scenario_id": sid,
            "source_id": rec["id"],
            "use_case": rec["capability"],
            "subject": rec["capability"],
            "task": rec["task"],
            "grade_band": None,
            "modality": "text",
            "system_prompt": rec.get("system_prompt") or "",
            "prompt": rec["input"],
            "conversation_context": [],
            "reference_solution": None,                       # held
            "native_reference_answer": rec.get("reference_answer") or "",
            "criterion_ids": cids,
            "native_rubric": native_rubric(rec["score_rubric"]),
            "atomization": {
                "method": overlay["method"],
                "n_atoms": len(atom_texts),
                "flagged": flagged,
                "flag_reason": overlay.get("flag_reason") or "",
            },
            "edit_provenance": rec.get("edit_provenance"),
            "source": SOURCE_URL,
            "split": SPLIT,
            "version": VERSION,
        })

        for cid, atom in zip(cids, atom_texts, strict=True):
            rubrics.append({
                "criterion_id": cid,
                "scenario_id": sid,
                "criterion": atom,
                "expected_evidence": [],                       # references held
                "scoring_type": "binary",
                "score_anchors": None,                         # atoms are natively binary
                "primary_skill": None,                         # skill axis TBD
                "q_mapping": None,                             # left null at ingest
                "q_rationale": None,
                "capability": rec["capability"],
                "task": rec["task"],
                # Labeled by generate_qmatrix.py in the same pass as q_mapping, so they
                # stay null until then. generate_qmatrix feeds any non-None value in as a
                # source-side hint, so a placeholder here would be read as ground truth.
                "criticality": None,
                "objectivity": None,
                "explicitness": None,
                "source": SOURCE_URL,
                "status": "approved",
                "version": VERSION,
            })

    return scenarios, rubrics, stats


def validate(scenarios: list[dict], rubrics: list[dict], edits: dict[str, dict]) -> list[str]:
    errs: list[str] = []

    for name, records, keys in (
        ("scenario", scenarios, SCENARIO_KEYS),
        ("rubric", rubrics, RUBRIC_KEYS),
    ):
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

    declared = [c for s in scenarios for c in s["criterion_ids"]]
    if set(declared) != set(cids) or len(declared) != len(cids):
        errs.append("criterion_ids / rubric id set mismatch")
    orphans = {r["scenario_id"] for r in rubrics} - set(sids)
    if orphans:
        errs.append(f"{len(orphans)} rubrics reference unknown scenarios")

    for s in scenarios:
        if not (s["prompt"] or "").strip():
            errs.append(f"{s['scenario_id']}: empty prompt")
        if not s["criterion_ids"]:
            errs.append(f"{s['scenario_id']}: no criteria")
        if s["atomization"]["n_atoms"] != len(s["criterion_ids"]):
            errs.append(f"{s['scenario_id']}: atomization.n_atoms != criterion count")
        nr = s["native_rubric"]
        if set(nr.get("levels", {})) != {"1", "2", "3", "4", "5"}:
            errs.append(f"{s['scenario_id']}: native_rubric missing 5 levels")
        elif any(not (nr["levels"][k] or "").strip() for k in nr["levels"]):
            errs.append(f"{s['scenario_id']}: blank native anchor level")
        if not (nr.get("criteria") or "").strip():
            errs.append(f"{s['scenario_id']}: empty native_rubric criteria")

    for r in rubrics:
        if not (r["criterion"] or "").strip():
            errs.append(f"{r['criterion_id']}: empty criterion")
        if r["scoring_type"] != "binary":
            errs.append(f"{r['criterion_id']}: scoring_type must be binary")
        if r["score_anchors"] is not None:
            errs.append(f"{r['criterion_id']}: score_anchors must be null (atoms are binary)")
        if r["q_mapping"] is not None or r["primary_skill"] is not None:
            errs.append(f"{r['criterion_id']}: q_mapping/primary_skill must be null at ingest")

    # every controlled edit must have landed, and edit_provenance presence must track it
    edited_source_ids = {s["source_id"] for s in scenarios} & set(edits)
    if edited_source_ids != set(edits):
        errs.append(f"edits lost at ingest: {sorted(set(edits) - edited_source_ids)}")
    for s in scenarios:
        if (s["source_id"] in edits) != (s["edit_provenance"] is not None):
            errs.append(f"{s['scenario_id']}: edit_provenance presence mismatch")

    return errs


def write(name: str, records: list[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = DATA_DIR / f"{name}.jsonl"
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False).translate(LINE_SEPS) + "\n")
    json_path = DATA_DIR / f"{name}.json"
    with json_path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    size = jsonl_path.stat().st_size / 1e6
    print(f"wrote {jsonl_path.relative_to(ROOT)} + .json  ({len(records)} rows, {size:.1f} MB)")


def main() -> None:
    scenarios, rubrics, stats = build()
    edits = load_edits()

    errs = validate(scenarios, rubrics, edits)
    if errs:
        print(f"VALIDATION FAILED ({len(errs)} issues):", file=sys.stderr)
        for e in errs[:20]:
            print(f"  - {e}", file=sys.stderr)
        raise SystemExit(1)
    print(f"validation passed: {len(scenarios)} scenarios, {len(rubrics)} criteria")

    write("scenarios", scenarios)
    write("rubrics", rubrics)

    per_scn = Counter(s["use_case"] for s in scenarios)
    per_cri = Counter(r["capability"] for r in rubrics)
    print("\nper capability (scenarios / criteria):")
    for cap, n in per_scn.most_common():
        print(f"  {cap:<22} {n:>4} / {per_cri[cap]:>4}")
    atoms_per = [len(s["criterion_ids"]) for s in scenarios]
    atoms_per.sort()
    print(f"\natoms/scenario: min {atoms_per[0]} / median {atoms_per[len(atoms_per) // 2]} "
          f"/ max {atoms_per[-1]}   (total criteria {len(rubrics)})")
    print(f"excluded (multilingual, held):  {stats['excluded_multilingual']}")
    print(f"excluded (broken at ingest):    {stats['excluded_broken']}  {sorted(EXCLUDED_IDS)}")
    print(f"controlled edits applied:       {stats['edits_applied']}")
    print(f"flagged scenarios (provenance): {stats['flagged']}")
    print("native 1-5 rubric preserved per scenario under native_rubric (sidecar).")
    print("reference answers: held (native_reference_answer kept; reference_solution null).")
    print("q_mapping/primary_skill: null (skill axis TBD); irt params appended later.")


if __name__ == "__main__":
    main()
