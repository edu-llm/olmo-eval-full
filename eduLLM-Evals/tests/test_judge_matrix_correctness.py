"""Regression tests for the multi-benchmark judge grading driver's matrix
data-correctness fixes (``scripts/run_all_judge_grading.py``).

Three guarantees are checked with tiny on-disk fixtures:

  (S2a) ``emit_cases`` FAILS FAST when the canonical validator rejects the cases
        (e.g. a duplicate criterion_id yields a duplicate case_id): it raises and
        writes NO cases file, so invalid cases are never shipped to the GPU box.

  (S2b) Rubric lookup and matrix columns are keyed by (scenario_id, criterion_id)
        end-to-end, so the SAME criterion_id appearing under two different
        scenarios does NOT collide: two distinct rubrics / two distinct columns.

  (S3)  A model whose every response errored/blanked (zero gradeable cells) is
        EXCLUDED from the matrix by default and surfaced in the staging manifest,
        instead of being graded as a full all-fail (y=0) row.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


def _load_module(name: str, rel: str):
    path = ROOT / rel
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


driver = _load_module("run_all_judge_grading", "scripts/run_all_judge_grading.py")


# ---------------------------------------------------------------------------
# fixture helpers
# ---------------------------------------------------------------------------
def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _response_row(model: str, scenario: str, output: str, finish: str = "stop") -> dict:
    return {"Model": model, "Scenario": scenario, "Output": output, "Finish Reason": finish}


def _make_bench(
    tmp_path: Path,
    scenarios: list[dict],
    rubrics: list[dict],
    responses: dict[str, list[dict]],
) -> tuple[Path, Path, Path]:
    """Lay out a tiny benchmark on disk; return (responses_dir, scenarios, rubrics)."""
    scenarios_path = tmp_path / "scenarios.jsonl"
    rubrics_path = tmp_path / "rubrics.jsonl"
    responses_dir = tmp_path / "responses"
    _write_jsonl(scenarios_path, scenarios)
    _write_jsonl(rubrics_path, rubrics)
    for model_file, rows in responses.items():
        _write_jsonl(responses_dir / f"{model_file}.jsonl", rows)
    return responses_dir, scenarios_path, rubrics_path


# ---------------------------------------------------------------------------
# S2a: emit fails fast on a duplicate criterion_id (duplicate case_id)
# ---------------------------------------------------------------------------
def test_emit_fails_fast_on_duplicate_criterion_id(tmp_path: Path) -> None:
    # Same criterion_id twice under the SAME scenario -> two staged rows with the
    # same (model, scenario, criterion) -> duplicate case_id -> canonical validator
    # rejects the batch.
    responses_dir, scenarios_path, rubrics_path = _make_bench(
        tmp_path,
        scenarios=[{"scenario_id": "s1", "prompt": "p1"}],
        rubrics=[
            {"criterion_id": "c1", "scenario_id": "s1", "criterion": "first"},
            {"criterion_id": "c1", "scenario_id": "s1", "criterion": "duplicate"},
        ],
        responses={"modelA": [_response_row("M/A", "s1", "an answer")]},
    )
    sb = driver.stage_benchmark("Dup", responses_dir, scenarios_path, rubrics_path)

    out_dir = tmp_path / "out"
    with pytest.raises(driver.CasesValidationError):
        driver.emit_cases(out_dir, sb)

    # FAIL FAST means nothing was shipped: no cases (or index) file on disk.
    assert not (out_dir / driver.CASES_NAME).exists()
    assert not (out_dir / driver.CASES_INDEX_NAME).exists()


# ---------------------------------------------------------------------------
# S2b: same criterion_id under two scenarios must not collide
# ---------------------------------------------------------------------------
def test_shared_criterion_id_across_scenarios_does_not_collide(tmp_path: Path) -> None:
    responses_dir, scenarios_path, rubrics_path = _make_bench(
        tmp_path,
        scenarios=[
            {"scenario_id": "s1", "prompt": "prompt one"},
            {"scenario_id": "s2", "prompt": "prompt two"},
        ],
        rubrics=[
            {"criterion_id": "c1", "scenario_id": "s1", "criterion": "TEXT-S1"},
            {"criterion_id": "c1", "scenario_id": "s2", "criterion": "TEXT-S2"},
        ],
        responses={
            "modelA": [
                _response_row("M/A", "s1", "answer one"),
                _response_row("M/A", "s2", "answer two"),
            ]
        },
    )
    sb = driver.stage_benchmark("Shared", responses_dir, scenarios_path, rubrics_path)

    # Rubric text is keyed by the pair: both survive, neither overwrites the other.
    assert set(sb.bank.rubrics) == {("s1", "c1"), ("s2", "c1")}
    assert sb.column_keys == [("s1", "c1"), ("s2", "c1")]
    assert sb.criterion_ids == ["c1", "c1"]

    out_dir = tmp_path / "out"
    n_cases, _n_af, _n_missing = driver.emit_cases(out_dir, sb)
    assert n_cases == 2

    # Each emitted case must carry the criterion text of ITS scenario.
    cases = [json.loads(ln) for ln in (out_dir / driver.CASES_NAME).read_text().splitlines()]
    text_by_scenario = {c["scenario_id"]: c["criterion"] for c in cases}
    assert text_by_scenario == {"s1": "TEXT-S1", "s2": "TEXT-S2"}

    # Distinct case_ids (blinded response_id folds in the scenario).
    assert len({c["case_id"] for c in cases}) == 2

    # Matrix columns must stay distinct: a pass on s1 and a fail on s2 for the
    # same criterion_id give the row [1, 0], not a single collapsed value.
    rjg = sys.modules["run_judge_grading"]
    verdicts = {
        ("M/A", "s1", "c1"): {"y": 1},
        ("M/A", "s2", "c1"): {"y": 0},
    }
    arr, csv_cells, n_holes = rjg.assemble_matrix(sb.models, sb.column_keys, verdicts)
    assert arr.shape == (1, 2)
    assert csv_cells == [["1", "0"]]
    assert n_holes == 0


# ---------------------------------------------------------------------------
# S3: dead models (zero gradeable cells) are excluded and reported
# ---------------------------------------------------------------------------
def test_dead_model_excluded_and_reported(tmp_path: Path) -> None:
    responses_dir, scenarios_path, rubrics_path = _make_bench(
        tmp_path,
        scenarios=[{"scenario_id": "s1", "prompt": "p1"}],
        rubrics=[{"criterion_id": "c1", "scenario_id": "s1", "criterion": "grade it"}],
        responses={
            "good": [_response_row("M/good", "s1", "a real answer")],
            "dead": [_response_row("M/dead", "s1", "", finish="error")],
        },
    )

    sb = driver.stage_benchmark("Dead", responses_dir, scenarios_path, rubrics_path)
    assert sb.models == ["M/good"]
    assert sb.excluded_models == ["M/dead"]
    assert all(row["model"] == "M/good" for row in sb.staged_rows)

    # The exclusion is surfaced in the staging manifest.
    out_dir = tmp_path / "out"
    driver.write_staging(out_dir, sb)
    manifest = json.loads((out_dir / driver.JUDGE_INPUTS_MANIFEST_NAME).read_text())
    assert manifest["n_excluded_models"] == 1
    assert manifest["excluded_models"] == ["M/dead"]
    assert manifest["models"] == ["M/good"]

    # Opt-out keeps the dead model (behind the clear default).
    sb_keep = driver.stage_benchmark(
        "Dead", responses_dir, scenarios_path, rubrics_path, exclude_dead_models=False
    )
    assert sb_keep.models == ["M/dead", "M/good"]
    assert sb_keep.excluded_models == []


def test_technical_failures_can_be_staged_as_missing(tmp_path: Path) -> None:
    responses_dir, scenarios_path, rubrics_path = _make_bench(
        tmp_path,
        scenarios=[
            {"scenario_id": "s1", "prompt": "p1"},
            {"scenario_id": "s2", "prompt": "p2"},
        ],
        rubrics=[
            {"criterion_id": "c1", "scenario_id": "s1", "criterion": "grade 1"},
            {"criterion_id": "c2", "scenario_id": "s2", "criterion": "grade 2"},
        ],
        responses={
            "mixed": [
                _response_row("M/mixed", "s1", "a real answer"),
                _response_row("M/mixed", "s2", "", finish="error"),
            ]
        },
    )

    sb = driver.stage_benchmark(
        "Mixed",
        responses_dir,
        scenarios_path,
        rubrics_path,
        technical_failure_policy="missing",
    )
    assert sb.models == ["M/mixed"]
    assert sb.gradeable_cells == 1
    assert sb.auto_fail_cells == 0
    assert sb.technical_missing_cells == 1
    assert sb.technical_missing_reasons == {"finish_reason_error": 1}

    out_dir = tmp_path / "out_missing"
    n_cases, n_auto_fail, n_bank_missing = driver.emit_cases(out_dir, sb)
    assert (n_cases, n_auto_fail, n_bank_missing) == (1, 0, 0)
    index_rows = [
        json.loads(line)
        for line in (out_dir / driver.CASES_INDEX_NAME).read_text().splitlines()
    ]
    assert len(index_rows) == 2
    missing_row = next(row for row in index_rows if row["scenario_id"] == "s2")
    assert missing_row["technical_missing"] == 1

    driver.write_staging(out_dir, sb)
    manifest = json.loads((out_dir / driver.JUDGE_INPUTS_MANIFEST_NAME).read_text())
    assert manifest["technical_failure_policy"] == "missing"
    assert manifest["technical_missing_cells"] == 1

    verdict_path = tmp_path / "returned_verdicts.jsonl"
    _write_jsonl(
        verdict_path,
        [
            {
                "case_id": driver.rjg.case_id_for("M/mixed", "s1", "c1"),
                "verdict": "pass",
                "status": "ok",
            }
        ],
    )
    summary = driver.ingest_benchmark(
        out_dir,
        sb,
        [verdict_path],
        driver.rjg.FrozenJudgeConfig(),
        no_decision_policy="missing",
        resume=False,
    )
    assert summary["stats"]["ingested_pass_fail"] == 1
    assert summary["stats"]["technical_missing"] == 1
    assert summary["coverage"] == {
        "complete": False,
        "n_holes": 1,
        "n_filled": 1,
    }

    verdict_rows = {
        row["scenario"]: row
        for row in (
            json.loads(line)
            for line in (out_dir / driver.VERDICTS_NAME).read_text().splitlines()
        )
    }
    assert verdict_rows["s1"]["y"] == 1
    assert verdict_rows["s1"]["source"] == "ingest"
    assert verdict_rows["s2"]["y"] is None
    assert verdict_rows["s2"]["source"] == "technical_missing"
    assert verdict_rows["s2"]["technical_missing_reason"] == "finish_reason_error"

    with (out_dir / driver.MATRIX_CSV_NAME).open(newline="", encoding="utf-8") as f:
        matrix_rows = list(csv.reader(f))
    assert matrix_rows == [["model", "c1", "c2"], ["M/mixed", "1", ""]]


def test_cli_threads_technical_failure_policy_to_staging(tmp_path: Path) -> None:
    responses_root = tmp_path / "responses"
    scenarios_root = tmp_path / "data"
    benchmark = "InFoBench"
    responses_dir, scenarios_path, rubrics_path = _make_bench(
        tmp_path / "fixture",
        scenarios=[
            {"scenario_id": "s1", "prompt": "p1"},
            {"scenario_id": "s2", "prompt": "p2"},
        ],
        rubrics=[
            {"criterion_id": "c1", "scenario_id": "s1", "criterion": "grade 1"},
            {"criterion_id": "c2", "scenario_id": "s2", "criterion": "grade 2"},
        ],
        responses={
            "mixed": [
                _response_row("M/mixed", "s1", "a real answer"),
                _response_row("M/mixed", "s2", "", finish="error"),
            ]
        },
    )
    target_responses = responses_root / benchmark
    target_responses.parent.mkdir(parents=True, exist_ok=True)
    responses_dir.rename(target_responses)
    target_data = scenarios_root / benchmark
    target_data.mkdir(parents=True, exist_ok=True)
    scenarios_path.rename(target_data / "scenarios.jsonl")
    rubrics_map = tmp_path / "rubrics_map.yaml"
    rubrics_map.write_text(f"{benchmark}: {rubrics_path}\n", encoding="utf-8")
    out_root = tmp_path / "judge_out"

    rc = driver.main(
        [
            "--only",
            benchmark,
            "--responses-root",
            str(responses_root),
            "--scenarios-root",
            str(scenarios_root),
            "--rubrics-map",
            str(rubrics_map),
            "--out-root",
            str(out_root),
            "--technical-failure-policy",
            "missing",
        ]
    )

    assert rc == 0
    manifest = json.loads(
        (out_root / benchmark / driver.JUDGE_INPUTS_MANIFEST_NAME).read_text()
    )
    assert manifest["technical_failure_policy"] == "missing"
    assert manifest["technical_missing_cells"] == 1
    cases = (out_root / benchmark / driver.CASES_NAME).read_text().splitlines()
    assert len(cases) == 1
