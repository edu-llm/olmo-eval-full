"""Offline tests for the leave-own-family-out frontier judge comparator."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT_PATH = ROOT / "scripts" / "compare_frontier_judges.py"
SPEC = importlib.util.spec_from_file_location("compare_frontier_judges", SCRIPT_PATH)
assert SPEC and SPEC.loader
frontier = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = frontier
SPEC.loader.exec_module(frontier)


HUMAN_FIELDS = [
    "case_id",
    "case_input_hash",
    "candidate_model",
    "candidate_model_slug",
    "scenario_id",
    "criterion_id",
    "primary_skill",
    "criticality",
    "human_label",
]


def human_rows() -> list[dict[str, str]]:
    models = {
        "openai": ("gpt-5.5", "openai-group/gpt-5.5"),
        "anthropic": ("opus-4.8", "claude-group/claude-opus-4-8"),
        "google": ("gemini-3.5-flash", "gemini-group/gemini-3.5-flash"),
    }
    rows = []
    for short, family in (("o", "openai"), ("a", "anthropic"), ("g", "google")):
        model, slug = models[family]
        for label, scenario in (("pass", "scenario-1"), ("fail", "scenario-2")):
            case_id = f"{short}-{label}"
            rows.append(
                {
                    "case_id": case_id,
                    "case_input_hash": f"input-{case_id}",
                    "candidate_model": model,
                    "candidate_model_slug": slug,
                    "scenario_id": scenario,
                    "criterion_id": f"criterion-{case_id}",
                    "primary_skill": "content",
                    "criticality": "critical" if label == "fail" else "not_critical",
                    "human_label": label,
                }
            )
    return rows


BASE_DECISIONS = {
    "gpt-5.5": {
        "a-pass": "pass",
        "a-fail": "fail",
        "g-pass": "pass",
        "g-fail": "pass",
    },
    "opus-4.8": {
        "o-pass": "pass",
        "o-fail": "fail",
        "g-pass": "pass",
        "g-fail": "fail",
    },
    "gemini-3.6-flash": {
        "o-pass": "pass",
        "o-fail": "pass",
        "a-pass": "pass",
        "a-fail": "fail",
    },
}


def write_humans(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HUMAN_FIELDS)
        writer.writeheader()
        writer.writerows(human_rows())


def _flip(decision: str) -> str:
    return "fail" if decision == "pass" else "pass"


def result_rows(judge: str, wave: str) -> list[dict]:
    family = frontier.JUDGE_FAMILIES[judge]
    variant, replicate = frontier.EXPECTED_WAVES[wave]
    humans = {row["case_id"]: row for row in human_rows()}
    decisions = dict(BASE_DECISIONS[judge])
    first_case = next(iter(decisions))
    # One repeat change for GPT exercises strict repeat metrics.
    if judge == "gpt-5.5" and wave == "canonical_r2":
        decisions[first_case] = _flip(decisions[first_case])
    # One controlled prompt flip per judge exercises prompt stability.
    if wave == "header_synonyms_r1":
        decisions[first_case] = _flip(decisions[first_case])

    rows = []
    for case_id, decision in decisions.items():
        human = humans[case_id]
        candidate_family = frontier._candidate_family(human, location=case_id)
        assert candidate_family != family
        prompt_hash = (
            f"canonical-prompt-{judge}-{case_id}"
            if variant == "canonical"
            else f"{variant}-prompt-{judge}-{case_id}"
        )
        rows.append(
            {
                "case_id": case_id,
                "response_id": f"response-{case_id}",
                "scenario_id": human["scenario_id"],
                "criterion_id": human["criterion_id"],
                "judge_name": judge,
                "judge_family": family,
                "judge_model": f"test/{judge}",
                "resolved_provider_model": f"resolved-test/{judge}",
                "judge_revision": None,
                "adapter": "generic-binary",
                "prompt_version": "judge-validation-v3",
                "evidence_policy_version": "criterion-evidence-gate-v1",
                "normalization_version": "frontier-normalization-v1",
                "configuration_hash": f"configuration-{judge}-{wave}",
                "frozen_configuration_hash": f"frozen-{judge}",
                "prompt_variant": variant,
                "replicate_id": replicate,
                "input_hash": human["case_input_hash"],
                "prompt_hash": prompt_hash,
                "candidate_family": candidate_family,
                "status": "ok",
                "verdict": decision,
            }
        )
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_study(tmp_path: Path) -> tuple[Path, Path]:
    humans = tmp_path / "human_labels.csv"
    root = tmp_path / "results"
    write_humans(humans)
    for judge in frontier.JUDGE_FAMILIES:
        for wave in frontier.EXPECTED_WAVES:
            write_jsonl(
                root / judge / wave / f"{wave}.jsonl",
                result_rows(judge, wave),
            )
    return humans, root


def load_study(
    humans: Path, root: Path
) -> tuple[list, dict[str, dict[str, frontier.WaveData]]]:
    cases, _ = frontier.load_human_labels(humans)
    by_id = {case.case_id: case for case in cases}
    waves_by_judge = {}
    for judge in frontier.JUDGE_FAMILIES:
        waves = {
            wave: frontier.load_wave(
                root / judge / wave / f"{wave}.jsonl",
                judge=judge,
                wave=wave,
                humans_by_id=by_id,
            )
            for wave in frontier.EXPECTED_WAVES
        }
        frontier.validate_judge_waves(judge, waves)
        waves_by_judge[judge] = waves
    return cases, waves_by_judge


def test_report_excludes_diagonal_and_reports_required_metrics(tmp_path: Path) -> None:
    humans, root = write_study(tmp_path)
    cases, waves = load_study(humans, root)

    report, review_rows = frontier.build_report(
        humans,
        cases,
        waves,
        bootstrap_samples=0,
        seed=42,
    )

    assert report["schema_version"] == "frontier-judge-comparison-v1"
    assert report["study_design"]["same_family_skips_are_no_decisions"] is False
    assert "confounded" in report["study_design"]["comparison_warning"]
    assert report["human_gold"]["candidate_family_counts"] == {
        "anthropic": 2,
        "google": 2,
        "openai": 2,
    }

    gpt = report["judges"]["gpt-5.5"]
    assert gpt["canonical"]["n"] == 4
    assert gpt["canonical"]["accuracy"] == 0.75
    assert gpt["canonical"]["coverage"] == 1.0
    assert set(gpt["by_candidate_family"]) == {"anthropic", "google"}
    assert "openai" not in gpt["by_candidate_family"]
    assert gpt["test_retest"]["worst_pairwise_strict_agreement_rate"] == 0.75
    assert gpt["prompt_consistency"]["worst_variant_flip_rate"] == 0.25
    assert gpt["by_candidate_family"]["anthropic"]["canonical"]["accuracy"] == 1.0
    assert gpt["by_candidate_family"]["google"]["canonical"]["accuracy"] == 0.5
    assert report["acceptance"]["thresholds"] == frontier.DEFAULT_THRESHOLDS
    assert report["acceptance"]["judges"]["gpt-5.5"]["passed"] is False
    # Opus is perfect on human agreement/repeats in this fixture, but the one
    # prompt flip deliberately fails the <=10% stability threshold.
    opus_acceptance = report["acceptance"]["judges"]["opus-4.8"]
    assert opus_acceptance["checks"]["macro_f1"]["passed"] is True
    assert opus_acceptance["checks"]["prompt_worst_flip_rate"]["passed"] is False
    assert "not a common-case ranking" in opus_acceptance["scope"]

    pair = report["cross_judge_pairs"]["gpt-5.5_vs_opus-4.8"]
    assert pair["shared_candidate_family"] == "google"
    assert pair["n"] == 2
    assert pair["agreement"]["strict_agreement_rate"] == 0.5
    assert pair["accuracy_delta_first_minus_second"] == -0.5

    consensus = report["two_judge_consensus"]
    assert consensus["n"] == 6
    assert consensus["auto_labeled_n"] == 4
    assert consensus["auto_label_coverage"] == pytest.approx(4 / 6)
    assert consensus["human_review_n"] == 2
    assert consensus["binary_disagreement_n"] == 2
    assert consensus["no_decision_review_n"] == 0
    assert consensus["end_to_end_performance"]["conditional_accuracy"] == 1.0
    assert consensus["critical_failures"]["sensitivity"] == pytest.approx(1 / 3)
    assert {row["case_id"] for row in review_rows} == {"o-fail", "g-fail"}


def test_eligible_api_failure_hurts_coverage_but_diagonal_skip_does_not(
    tmp_path: Path,
) -> None:
    humans, root = write_study(tmp_path)
    path = root / "gpt-5.5" / "canonical_r1" / "canonical_r1.jsonl"
    rows = result_rows("gpt-5.5", "canonical_r1")
    rows[0]["status"] = "api_error"
    rows[0]["verdict"] = "no_decision"
    write_jsonl(path, rows)
    cases, waves = load_study(humans, root)

    report, _ = frontier.build_report(
        humans, cases, waves, bootstrap_samples=0, seed=42
    )
    gpt = report["judges"]["gpt-5.5"]["canonical"]
    assert gpt["n"] == 4
    assert gpt["no_decision_n"] == 1
    assert gpt["coverage"] == 0.75


def test_load_wave_rejects_same_family_and_missing_eligible_rows(tmp_path: Path) -> None:
    humans, root = write_study(tmp_path)
    cases, _ = frontier.load_human_labels(humans)
    by_id = {case.case_id: case for case in cases}
    path = root / "gpt-5.5" / "canonical_r1" / "canonical_r1.jsonl"

    same_family = result_rows("gpt-5.5", "canonical_r1")
    template = dict(same_family[0])
    template.update(
        {
            "case_id": "o-pass",
            "scenario_id": "scenario-1",
            "criterion_id": "criterion-o-pass",
            "input_hash": "input-o-pass",
            "candidate_family": "openai",
            "prompt_hash": "canonical-prompt-gpt-5.5-o-pass",
        }
    )
    same_family.append(template)
    write_jsonl(path, same_family)
    with pytest.raises(frontier.InputValidationError, match="same-family"):
        frontier.load_wave(
            path,
            judge="gpt-5.5",
            wave="canonical_r1",
            humans_by_id=by_id,
        )

    write_jsonl(path, result_rows("gpt-5.5", "canonical_r1")[:-1])
    with pytest.raises(frontier.InputValidationError, match="incomplete eligible coverage"):
        frontier.load_wave(
            path,
            judge="gpt-5.5",
            wave="canonical_r1",
            humans_by_id=by_id,
        )


def test_load_wave_rejects_blank_or_mixed_resolved_provider_model(
    tmp_path: Path,
) -> None:
    humans, root = write_study(tmp_path)
    cases, _ = frontier.load_human_labels(humans)
    by_id = {case.case_id: case for case in cases}
    path = root / "gpt-5.5" / "canonical_r1" / "canonical_r1.jsonl"

    rows = result_rows("gpt-5.5", "canonical_r1")
    for row in rows:
        row["resolved_provider_model"] = ""
    write_jsonl(path, rows)
    with pytest.raises(
        frontier.InputValidationError,
        match="provenance field resolved_provider_model is blank",
    ):
        frontier.load_wave(
            path,
            judge="gpt-5.5",
            wave="canonical_r1",
            humans_by_id=by_id,
        )

    rows = result_rows("gpt-5.5", "canonical_r1")
    rows[0]["resolved_provider_model"] = "resolved-test/gpt-5.5-drifted"
    write_jsonl(path, rows)
    with pytest.raises(
        frontier.InputValidationError,
        match=(
            "rows do not share one provenance value for "
            "resolved_provider_model"
        ),
    ):
        frontier.load_wave(
            path,
            judge="gpt-5.5",
            wave="canonical_r1",
            humans_by_id=by_id,
        )


def test_validate_judge_waves_rejects_resolved_provider_model_drift(
    tmp_path: Path,
) -> None:
    humans, root = write_study(tmp_path)
    path = root / "gpt-5.5" / "canonical_r2" / "canonical_r2.jsonl"
    rows = result_rows("gpt-5.5", "canonical_r2")
    for row in rows:
        row["resolved_provider_model"] = "resolved-test/gpt-5.5-drifted"
    write_jsonl(path, rows)

    with pytest.raises(
        frontier.InputValidationError,
        match=(
            "provenance field resolved_provider_model differs between "
            "canonical_r1 and canonical_r2"
        ),
    ):
        load_study(humans, root)


def test_cli_writes_json_summary_and_human_review_queue(tmp_path: Path) -> None:
    humans, root = write_study(tmp_path)
    json_out = tmp_path / "comparison.json"
    csv_out = tmp_path / "comparison.csv"
    review_out = tmp_path / "review.csv"

    code = frontier.main(
        [
            str(humans),
            "--results-root",
            str(root),
            "--bootstrap-samples",
            "0",
            "--json-out",
            str(json_out),
            "--csv-out",
            str(csv_out),
            "--review-out",
            str(review_out),
        ]
    )

    assert code == 0
    report = json.loads(json_out.read_text(encoding="utf-8"))
    assert report["judges"]["opus-4.8"]["canonical"]["accuracy"] == 1.0
    with csv_out.open(encoding="utf-8", newline="") as handle:
        summary = list(csv.DictReader(handle))
    assert {row["judge"] for row in summary} == set(frontier.JUDGE_FAMILIES)
    with review_out.open(encoding="utf-8", newline="") as handle:
        review = list(csv.DictReader(handle))
    assert {row["case_id"] for row in review} == {"o-fail", "g-fail"}


def test_scenario_cluster_bootstrap_adds_core_intervals(tmp_path: Path) -> None:
    humans, root = write_study(tmp_path)
    cases, waves = load_study(humans, root)

    report, _ = frontier.build_report(
        humans,
        cases,
        waves,
        bootstrap_samples=30,
        seed=7,
    )

    judge_ci = report["judges"]["gpt-5.5"]["confidence_intervals_95"]
    assert judge_ci["canonical.macro_f1"]["requested_resamples"] == 30
    assert judge_ci["retest.worst_pairwise_strict"] is not None
    pair_ci = report["cross_judge_pairs"]["gpt-5.5_vs_opus-4.8"][
        "confidence_intervals_95"
    ]
    assert pair_ci["strict_agreement"] is not None
    assert report["two_judge_consensus"]["confidence_intervals_95"][
        "human_review_rate"
    ] is not None
