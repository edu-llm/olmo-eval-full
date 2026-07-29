"""Verify judge-verdict ingestion + auditing against a synthetic fixture.

Two independent guarantees are checked here:

  (a) The ingester (``scripts/run_judge_grading.py grade --mode ingest-verdicts``)
      maps agree-pass -> 1, agree-fail -> 0, BOTH void directions (native/PF
      disagreement) and truncation voids -> MISSING (NaN), and cells ABSENT from
      the verdict files -> MISSING. The legacy ``--no-decision-policy fail`` path
      is also exercised to prove it is opt-in only.

  (b) ``scripts/audit_judge_verdicts.py`` reproduces the expected cause-bucket
      counts, no_decision total, finish_reason distribution, per-shard/per-model
      counts, and criteria-set match on the same fixture.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "synthetic_verdicts.jsonl"

FIXTURE_MODEL = "fixture/model-A"
FIXTURE_SCENARIO = "fx_0001"
# c01..c05 are judged in the fixture; c06 is deliberately ABSENT (a hole).
FIXTURE_CRITERIA = [f"fx_0001_c0{i}" for i in range(1, 7)]


def _load_module(name: str, rel: str):
    path = ROOT / rel
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


grading = _load_module("run_judge_grading", "scripts/run_judge_grading.py")
auditor = _load_module("audit_judge_verdicts", "scripts/audit_judge_verdicts.py")


def _write_staging(staging: Path) -> None:
    """Write the judge_inputs + manifest the ingester expects (1 model x 6 crits)."""
    staging.mkdir(parents=True, exist_ok=True)
    with (staging / "judge_inputs.jsonl").open("w", encoding="utf-8") as f:
        for cid in FIXTURE_CRITERIA:
            f.write(json.dumps({
                "model": FIXTURE_MODEL,
                "scenario": FIXTURE_SCENARIO,
                "criterion_id": cid,
                "rubric": f"criterion {cid}",
                "response": "some tutor response",
                "auto_fail": 0,
                "auto_fail_reason": "",
            }) + "\n")
    manifest = {
        "models": [FIXTURE_MODEL],
        "criterion_ids": FIXTURE_CRITERIA,
    }
    with (staging / "judge_inputs_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f)


def _grade_args(staging: Path, policy: str) -> argparse.Namespace:
    return argparse.Namespace(
        staging_dir=staging,
        mode="ingest-verdicts",
        ingest_file=[str(FIXTURE)],
        judge_config=ROOT / "judge_frozen.yaml",
        scenarios=None,
        rubrics=None,
        batch_size=32,
        concurrency=1,
        resume=False,
        dry_run=False,
        no_decision_policy=policy,
    )


def _matrix_row(staging: Path) -> np.ndarray:
    arr = np.load(staging / "response_matrix.npy")
    assert arr.shape == (1, len(FIXTURE_CRITERIA))
    return arr[0]


def test_ingest_void_and_absent_map_to_missing(tmp_path):
    staging = tmp_path / "staging"
    _write_staging(staging)
    rc = grading.cmd_grade(_grade_args(staging, policy="missing"))
    assert rc == 0
    row = _matrix_row(staging)

    # c01 agree-pass -> 1 ; c02 agree-fail -> 0
    assert row[0] == 1.0
    assert row[1] == 0.0
    # c03 native=pass/PF=fail void, c04 native=fail/PF=pass void,
    # c05 truncation void, c06 ABSENT -> all MISSING (NaN)
    for j in (2, 3, 4, 5):
        assert math.isnan(row[j]), f"criterion index {j} should be NaN, got {row[j]}"

    # genuine decisions preserved; voids/absent excluded from filled count
    manifest = json.loads((staging / "response_matrix_manifest.json").read_text())
    assert manifest["coverage"]["n_filled"] == 2
    assert manifest["coverage"]["n_holes"] == 4


def test_legacy_fail_policy_scores_voids_zero(tmp_path):
    staging = tmp_path / "staging"
    _write_staging(staging)
    rc = grading.cmd_grade(_grade_args(staging, policy="fail"))
    assert rc == 0
    row = _matrix_row(staging)
    assert row[0] == 1.0  # agree-pass
    assert row[1] == 0.0  # agree-fail
    assert row[2] == 0.0 and row[3] == 0.0 and row[4] == 0.0  # voids -> 0 (legacy)
    assert math.isnan(row[5])  # absent cell is STILL missing, never fail


def test_normalize_verdict_unit():
    # default policy: void -> y is None (missing)
    y, src, label, _ = grading.normalize_verdict({"verdict": "no_decision"})
    assert y is None and src == "ingest_no_decision" and label == "no_decision"
    # legacy policy: void -> 0
    y, src, _, _ = grading.normalize_verdict({"verdict": "no_decision"}, "fail")
    assert y == 0 and src == "ingest_no_decision"
    # genuine pass/fail unaffected
    assert grading.normalize_verdict({"verdict": "pass"})[0] == 1
    assert grading.normalize_verdict({"verdict": "fail"})[0] == 0


def test_audit_reproduces_expected_buckets(tmp_path):
    # tiny curated/final banks so the criteria-set match is deterministic
    final_rub = tmp_path / "rubrics_qmatrix_final.jsonl"
    with final_rub.open("w", encoding="utf-8") as f:
        for cid in FIXTURE_CRITERIA[:5]:
            f.write(json.dumps({"criterion_id": cid, "scenario_id": FIXTURE_SCENARIO}) + "\n")
    curated_rub = tmp_path / "rubrics_qmatrix_curated.jsonl"
    curated_rub.write_text(json.dumps({"criterion_id": "other_c01"}) + "\n", encoding="utf-8")

    out_json = tmp_path / "audit.json"
    out_csv = tmp_path / "audit.csv"
    rc = auditor.main([
        str(FIXTURE),
        "--out-json", str(out_json),
        "--out-csv", str(out_csv),
        "--final-rubrics", str(final_rub),
        "--curated-rubrics", str(curated_rub),
    ])
    assert rc == 0
    report = json.loads(out_json.read_text())

    t = report["totals"]
    assert t["total_cells"] == 5
    assert t["decided"] == 2
    assert t["no_decision"] == 3
    assert t["no_decision_pct"] == 60.0

    b = report["no_decision_cause_buckets"]
    assert b["native_pass_pf_fail"] == 1
    assert b["native_fail_pf_pass"] == 1
    assert b["truncation"] == 1
    assert b["truncation_and_disagreement"] == 0
    assert report["no_decision_disagreement_total"] == 2
    assert report["no_decision_unclassified"] == 0

    assert report["finish_reason_distribution_all"] == {"stop": 4, "length": 1}
    assert report["finish_reason_distribution_no_decision"] == {"stop": 2, "length": 1}

    assert report["per_shard"]["shard_00001"] == {"total": 5, "no_decision": 3}
    assert report["per_model_no_decision"] == {FIXTURE_MODEL: 3}

    cm = report["criteria_set_match"]
    assert cm["n_verdict_criteria"] == 5
    assert cm["final"]["match_pct"] == 100.0
    assert cm["curated"]["match_pct"] == 0.0
    assert cm["best_match"] == "final"
    assert cm["n_unknown_ids"] == 0

    assert out_csv.is_file()
