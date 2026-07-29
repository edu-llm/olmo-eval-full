"""Unit tests for scripts/ingest_calibration_csv.py on tiny synthetic fixtures.

Builds a small case_index.jsonl + verdicts CSV covering pass / fail / no_decision,
an absent cell, a policy-missing cell, and an unresolved response_id, then checks the
staging matrix shape and 0/1/NaN placement. Also verifies that the path (c) HMAC
recipe reproduces the same response_ids as the synthetic index.
"""

from __future__ import annotations

import csv
import hashlib
import hmac
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "ingest_calibration_csv", ROOT / "scripts" / "ingest_calibration_csv.py"
)
ing = importlib.util.module_from_spec(_SPEC)
sys.modules["ingest_calibration_csv"] = ing
assert _SPEC and _SPEC.loader
_SPEC.loader.exec_module(ing)


ID_KEY_HEX = "a" * 64  # deterministic known secret for the HMAC reproduction test
MODELS = ["org/model-A", "org/model-B"]
SCENARIOS = ["tb_0001", "tb_0002"]
# curated criteria (matrix columns); tb_0002_c02 is deliberately never in the CSV.
CRITERIA = {
    "tb_0001": ["tb_0001_c01", "tb_0001_c02"],
    "tb_0002": ["tb_0002_c01", "tb_0002_c02"],
}
ALL_CRITERIA = [c for s in SCENARIOS for c in CRITERIA[s]]


def _rid(model: str, scenario: str) -> str:
    key = ID_KEY_HEX.encode("utf-8")
    payload = f"response\x1f{model}\x1f{scenario}".encode()
    return "response_" + hmac.new(key, payload, hashlib.sha256).hexdigest()[:24]


@pytest.fixture
def fixtures(tmp_path: Path) -> dict:
    # --- curated bank: 4 nonoptional + 1 optional (must be excluded) ---
    curated = tmp_path / "rubrics_qmatrix_curated.jsonl"
    with curated.open("w", encoding="utf-8") as fh:
        for scenario, crits in CRITERIA.items():
            for cid in crits:
                fh.write(json.dumps({"criterion_id": cid, "scenario_id": scenario}) + "\n")
        fh.write(json.dumps({"criterion_id": "tb_0002_cOPT", "optional": True}) + "\n")

    # --- cohort policy: both models included ---
    cohort = tmp_path / "cohort.json"
    cohort.write_text(
        json.dumps({"schema_version": "t", "included_models": MODELS}), encoding="utf-8"
    )

    # --- private case_index.jsonl (model-major, then criteria) ---
    case_index = tmp_path / "case_index.jsonl"
    position = 0
    rows = []
    for model in MODELS:
        for scenario in SCENARIOS:
            rid = _rid(model, scenario)
            for cid in CRITERIA[scenario]:
                missing = model == "org/model-B" and cid == "tb_0002_c01"
                rows.append(
                    {
                        "position": position,
                        "case_id": f"{rid}__{cid}",
                        "response_id": rid,
                        "model": model,
                        "scenario_id": scenario,
                        "criterion_id": cid,
                        "auto_fail": False,
                        "auto_fail_reason": "",
                        "missing": bool(missing),
                        "missing_reason": "empty_response" if missing else "",
                    }
                )
                position += 1
    with case_index.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    # --- verdicts CSV ---
    # model-A: c01 pass=1, c02 fail=0, tb_0002_c01 no_decision->NaN, tb_0002_c02 ABSENT
    # model-B: c01 fail=0, c02 pass=1, tb_0002_c01 = policy-missing (present but overridden)
    # plus one unresolved response_id row.
    header = [
        "shard",
        "case_id",
        "response_id",
        "scenario_id",
        "criterion_id",
        "judge_model",
        "verdict",
        "model_verdict",
        "criterion_score",
        "decision_source",
        "status",
        "attempt",
        "p_pass",
        "p_fail",
    ]
    csv_rows = []

    def add(model, scenario, cid, verdict):
        rid = _rid(model, scenario)
        score = {"pass": 100.0, "fail": 0.0, "no_decision": ""}[verdict]
        mv = "" if verdict == "no_decision" else verdict
        csv_rows.append(
            [
                "shard_00000",
                f"{rid}__{cid}",
                rid,
                scenario,
                cid,
                "Qwen",
                verdict,
                mv,
                score,
                "max_atomic_p_fail",
                "ok",
                1,
                0.5,
                0.5,
            ]
        )

    add("org/model-A", "tb_0001", "tb_0001_c01", "pass")
    add("org/model-A", "tb_0001", "tb_0001_c02", "fail")
    add("org/model-A", "tb_0002", "tb_0002_c01", "no_decision")
    # tb_0002_c02 for A intentionally absent
    add("org/model-B", "tb_0001", "tb_0001_c01", "fail")
    add("org/model-B", "tb_0001", "tb_0001_c02", "pass")
    add("org/model-B", "tb_0002", "tb_0002_c01", "pass")  # overridden by policy missing
    add("org/model-B", "tb_0002", "tb_0002_c02", "pass")
    # unresolved: a response_id not in the index / archive
    csv_rows.append(
        [
            "shard_00000",
            "response_deadbeefdeadbeefdeadbeef__tb_0001_c01",
            "response_deadbeefdeadbeefdeadbeef",
            "tb_0001",
            "tb_0001_c01",
            "Qwen",
            "pass",
            "pass",
            100.0,
            "max_atomic_p_fail",
            "ok",
            1,
            0.9,
            0.1,
        ]
    )

    csv_path = tmp_path / "run_data.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(csv_rows)

    # --- tutorbench responses archive (for path (c)) ---
    responses = tmp_path / "responses"
    responses.mkdir()
    for model in MODELS:
        fname = model.replace("/", "_") + ".jsonl"
        with (responses / fname).open("w", encoding="utf-8") as fh:
            for scenario in SCENARIOS:
                fh.write(
                    json.dumps(
                        {
                            "Benchmark": "TutorBench",
                            "Scenario": scenario,
                            "Model": model,
                            "Output": "x",
                        }
                    )
                    + "\n"
                )

    # --- fully de-blinded verdicts JSONL (tutor_model per row; no mapping) ---
    jsonl_path = tmp_path / "run_data.jsonl"
    jsonl_rows = [
        ("org/model-A", "tb_0001", "tb_0001_c01", "pass"),
        ("org/model-A", "tb_0001", "tb_0001_c02", "fail"),
        ("org/model-A", "tb_0002", "tb_0002_c01", "no_decision"),
        # model-A tb_0002_c02 intentionally absent
        ("org/model-B", "tb_0001", "tb_0001_c01", "fail"),
        ("org/model-B", "tb_0001", "tb_0001_c02", "pass"),
        ("org/model-B", "tb_0002", "tb_0002_c01", "pass"),
        ("org/model-B", "tb_0002", "tb_0002_c02", "pass"),
        # off-bank criterion (optional) must be excluded from the matrix columns
        ("org/model-A", "tb_0002", "tb_0002_cOPT", "pass"),
    ]
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for model, scenario, cid, verdict in jsonl_rows:
            fh.write(
                json.dumps(
                    {
                        "shard": "shard_00000",
                        "case_id": f"{_rid(model, scenario)}__{cid}",
                        "response_id": _rid(model, scenario),
                        "tutor_model": model,
                        "scenario_id": scenario,
                        "criterion_id": cid,
                        "verdict": verdict,
                        "model_verdict": "" if verdict == "no_decision" else verdict,
                        "criterion_score": "" if verdict == "no_decision" else 0,
                        "decision_source": "max_atomic_p_fail",
                        "status": "ok",
                        "p_pass": 0.5,
                        "p_fail": 0.5,
                    }
                )
                + "\n"
            )

    return {
        "tmp": tmp_path,
        "curated": curated,
        "cohort": cohort,
        "case_index": case_index,
        "csv": csv_path,
        "jsonl": jsonl_path,
        "responses": responses,
    }


def _args(fx, out_dir, **overrides):
    import argparse

    ns = argparse.Namespace(
        csv=fx["csv"],
        curated=fx["curated"],
        cohort_policy=fx["cohort"],
        responses=fx["responses"],
        out_dir=out_dir,
        cohort_only=False,
        jsonl=None,
        case_index=None,
        merged_dir=None,
        id_key=None,
        id_key_file=None,
    )
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


def test_case_index_matrix_shape_and_cells(fixtures):
    out = fixtures["tmp"] / "staging_a"
    ing.run(_args(fixtures, out, case_index=fixtures["case_index"]))

    mat = pd.read_csv(out / "response_matrix.csv", index_col="model")
    mat = mat.apply(pd.to_numeric, errors="coerce")

    # shape: 2 models x 4 nonoptional criteria (optional column excluded)
    assert mat.shape == (2, 4)
    assert list(mat.columns) == ALL_CRITERIA
    assert list(mat.index) == MODELS
    assert "tb_0002_cOPT" not in mat.columns

    # 0/1 placement
    assert mat.loc["org/model-A", "tb_0001_c01"] == 1.0
    assert mat.loc["org/model-A", "tb_0001_c02"] == 0.0
    assert mat.loc["org/model-B", "tb_0001_c01"] == 0.0
    assert mat.loc["org/model-B", "tb_0001_c02"] == 1.0
    assert mat.loc["org/model-B", "tb_0002_c02"] == 1.0

    # no_decision -> NaN
    assert np.isnan(mat.loc["org/model-A", "tb_0002_c01"])
    # absent from CSV -> NaN
    assert np.isnan(mat.loc["org/model-A", "tb_0002_c02"])
    # case-index missing overrides the CSV pass -> NaN
    assert np.isnan(mat.loc["org/model-B", "tb_0002_c01"])


def test_manifest_counts_and_unresolved(fixtures):
    out = fixtures["tmp"] / "staging_a2"
    ing.run(_args(fixtures, out, case_index=fixtures["case_index"]))
    manifest = json.loads((out / "response_matrix_manifest.json").read_text(encoding="utf-8"))

    assert manifest["n_models"] == 2
    assert manifest["n_criteria"] == 4
    # pass: A_c01, B_c02, B_c02(tb_0002) => 3 ; fail: A_c02, B_c01 => 2
    assert manifest["pass_count"] == 3
    assert manifest["fail_count"] == 2
    assert manifest["no_decision_cells"] == 1
    assert manifest["policy_missing_cells"] == 1
    assert manifest["unresolved_response_id_count"] == 1

    audit = json.loads((out / "ingest_audit.json").read_text(encoding="utf-8"))
    assert audit["unresolved_response_id_count"] == 1
    assert "response_deadbeefdeadbeefdeadbeef" in audit["unresolved_response_ids"]


def test_npy_matches_csv(fixtures):
    out = fixtures["tmp"] / "staging_npy"
    ing.run(_args(fixtures, out, case_index=fixtures["case_index"]))
    arr = np.load(out / "response_matrix.npy")
    assert arr.shape == (2, 4)
    assert arr.dtype == np.float32
    assert arr[0, 0] == 1.0
    assert np.isnan(arr[0, 3])


def test_hmac_path_c_reproduces_ids(fixtures):
    out = fixtures["tmp"] / "staging_c"
    ing.run(_args(fixtures, out, id_key=ID_KEY_HEX))

    mat = pd.read_csv(out / "response_matrix.csv", index_col="model")
    mat = mat.apply(pd.to_numeric, errors="coerce")
    assert mat.shape == (2, 4)
    # Path (c) has no case-index policy, so the resolved CSV verdicts land as-is.
    assert mat.loc["org/model-A", "tb_0001_c01"] == 1.0
    assert mat.loc["org/model-A", "tb_0001_c02"] == 0.0
    # tb_0002_c01 pass for B is NOT overridden in path (c) (no policy) -> 1
    assert mat.loc["org/model-B", "tb_0002_c01"] == 1.0

    manifest = json.loads((out / "response_matrix_manifest.json").read_text(encoding="utf-8"))
    # every resolvable distinct response_id in the CSV maps except the deadbeef one
    assert manifest["unresolved_response_id_count"] == 1
    assert manifest["response_id_coverage"] == pytest.approx(4 / 5)


def test_hmac_ids_equal_synthetic_index(fixtures):
    """Path (c) HMAC recipe yields the same response_ids the synthetic index used."""
    mapping = ing.mapping_from_hmac(ID_KEY_HEX.encode("utf-8"), fixtures["responses"])
    index_map = ing.mapping_from_case_index(fixtures["case_index"]).response_to_model
    assert mapping.response_to_model == index_map
    for model in MODELS:
        for scenario in SCENARIOS:
            assert mapping.response_to_model[_rid(model, scenario)] == model


def test_cohort_only_subsets_rows(fixtures):
    # Restrict cohort to just model-A; --cohort-only should drop model-B's row.
    fixtures["cohort"].write_text(
        json.dumps({"schema_version": "t", "included_models": ["org/model-A"]}),
        encoding="utf-8",
    )
    out = fixtures["tmp"] / "staging_cohort"
    ing.run(_args(fixtures, out, case_index=fixtures["case_index"], cohort_only=True))
    mat = pd.read_csv(out / "response_matrix.csv", index_col="model")
    assert list(mat.index) == ["org/model-A"]

    manifest = json.loads((out / "response_matrix_manifest.json").read_text(encoding="utf-8"))
    assert manifest["cohort"]["cohort_only_applied"] is True


# ---------------------------------------------------------------------------
# de-blinded JSONL mode (--jsonl)
# ---------------------------------------------------------------------------


def test_jsonl_matrix_shape_and_cells(fixtures):
    out = fixtures["tmp"] / "staging_jsonl"
    ing.run(_args(fixtures, out, jsonl=fixtures["jsonl"]))

    mat = pd.read_csv(out / "response_matrix.csv", index_col="model")
    mat = mat.apply(pd.to_numeric, errors="coerce")

    # tutor_model row keys, cohort order; 4 nonoptional columns (optional excluded)
    assert mat.shape == (2, 4)
    assert list(mat.columns) == ALL_CRITERIA
    assert list(mat.index) == MODELS
    assert "tb_0002_cOPT" not in mat.columns

    # 0/1 placement
    assert mat.loc["org/model-A", "tb_0001_c01"] == 1.0
    assert mat.loc["org/model-A", "tb_0001_c02"] == 0.0
    assert mat.loc["org/model-B", "tb_0001_c01"] == 0.0
    assert mat.loc["org/model-B", "tb_0001_c02"] == 1.0
    assert mat.loc["org/model-B", "tb_0002_c01"] == 1.0
    assert mat.loc["org/model-B", "tb_0002_c02"] == 1.0

    # no_decision -> NaN ; absent-from-file -> NaN
    assert np.isnan(mat.loc["org/model-A", "tb_0002_c01"])
    assert np.isnan(mat.loc["org/model-A", "tb_0002_c02"])


def test_jsonl_manifest_and_report(fixtures):
    out = fixtures["tmp"] / "staging_jsonl2"
    ing.run(_args(fixtures, out, jsonl=fixtures["jsonl"]))
    manifest = json.loads((out / "response_matrix_manifest.json").read_text(encoding="utf-8"))

    assert manifest["input_mode"] == "jsonl-deblinded"
    assert manifest["n_models"] == 2
    assert manifest["n_criteria"] == 4
    # pass: A_c01, B_c02, B tb_0002_c01, B tb_0002_c02 => 4 ; fail: A_c02, B_c01 => 2
    assert manifest["pass_count"] == 4
    assert manifest["fail_count"] == 2
    assert manifest["no_decision_cells"] == 1
    assert manifest["unresolved_response_id_count"] == 0
    # tb_0002_cOPT is off-bank (optional) and must be reported, not placed
    assert manifest["off_bank_criterion_count"] == 1

    report = manifest["matrix_report"]
    # no criterion column is entirely empty in this fixture (all 4 have >=1 obs)
    assert report["empty_column_count"] == 0
    assert report["tutor_model_unmatched_v2_files"] == []
    # per-model observed cells: A has 2 (c01,c02), B has 4
    per_model = {e["model"]: e["observed_cells"] for e in report["per_model_cell_counts"]}
    assert per_model["org/model-A"] == 2
    assert per_model["org/model-B"] == 4


def test_jsonl_empty_column_detection(fixtures):
    # Drop every row mentioning tb_0002_c02 so that column is entirely absent.
    src = fixtures["jsonl"].read_text(encoding="utf-8").splitlines()
    kept = [ln for ln in src if "tb_0002_c02" not in ln]
    trimmed = fixtures["tmp"] / "run_data_trimmed.jsonl"
    trimmed.write_text("\n".join(kept) + "\n", encoding="utf-8")

    out = fixtures["tmp"] / "staging_jsonl_empty"
    ing.run(_args(fixtures, out, jsonl=trimmed))
    manifest = json.loads((out / "response_matrix_manifest.json").read_text(encoding="utf-8"))
    report = manifest["matrix_report"]
    assert report["empty_column_count"] == 1
    assert "tb_0002_c02" in report["empty_columns_sample"]

    mat = pd.read_csv(out / "response_matrix.csv", index_col="model")
    mat = mat.apply(pd.to_numeric, errors="coerce")
    assert mat["tb_0002_c02"].isna().all()


def test_jsonl_unmatched_model_reported(fixtures):
    # A tutor_model with no matching v2 response file must be flagged.
    extra = fixtures["jsonl"].read_text(encoding="utf-8")
    extra += (
        json.dumps(
            {
                "tutor_model": "ghost/not-a-real-model",
                "scenario_id": "tb_0001",
                "criterion_id": "tb_0001_c01",
                "verdict": "pass",
                "response_id": "response_ghost",
                "decision_source": "x",
            }
        )
        + "\n"
    )
    ghost = fixtures["tmp"] / "run_data_ghost.jsonl"
    ghost.write_text(extra, encoding="utf-8")

    out = fixtures["tmp"] / "staging_ghost"
    ing.run(_args(fixtures, out, jsonl=ghost))
    manifest = json.loads((out / "response_matrix_manifest.json").read_text(encoding="utf-8"))
    assert "ghost/not-a-real-model" in manifest["matrix_report"]["tutor_model_unmatched_v2_files"]
