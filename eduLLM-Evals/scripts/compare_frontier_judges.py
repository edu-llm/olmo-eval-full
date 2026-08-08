#!/usr/bin/env python3
"""Compare the cross-family frontier-judge study with human labels.

The study deliberately omits the diagonal of the judge-by-tutor-family matrix:

* GPT-5.5 does not grade OpenAI tutor responses.
* Opus 4.8 does not grade Anthropic tutor responses.
* Gemini 3.6 Flash does not grade Google/Gemini tutor responses.

Consequently, a missing same-family judgment is *not* a ``no_decision``.  It is
an ineligible pair that must be excluded.  A missing eligible judgment, on the
other hand, is an invalid/incomplete run.  This script validates that routing
before calculating any metric.

The expected suite has six waves per judge: three canonical repeats and three
semantically invariant prompt variants.  Result rows are the JSONL artifacts
written by ``run_frontier_judge_validation.py``.  The human CSV is the prepared
``human_labels.csv`` from the judge-validation study.

Typical usage::

    python scripts/compare_frontier_judges.py \
      runs/judge_validation_v2/human_labels.csv \
      --results-root runs/frontier_judges \
      --json-out runs/frontier_judges/comparison.json \
      --csv-out runs/frontier_judges/comparison.csv \
      --review-out runs/frontier_judges/human_review.csv

Use repeated ``--wave JUDGE:WAVE:PATH`` arguments instead of
``--results-root`` when the artifacts do not use the standard directory
layout.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Iterable, Mapping, Sequence


# Dict insertion order defines stable report/pair ordering.
JUDGE_FAMILIES: dict[str, str] = {
    "gpt-5.5": "openai",
    "opus-4.8": "anthropic",
    "gemini-3.6-flash": "google",
}
TUTOR_FAMILIES = tuple(JUDGE_FAMILIES.values())

EXPECTED_WAVES: dict[str, tuple[str, str]] = {
    "canonical_r1": ("canonical", "r1"),
    "canonical_r2": ("canonical", "r2"),
    "canonical_r3": ("canonical", "r3"),
    "whitespace_r1": ("whitespace", "r1"),
    "header_synonyms_r1": ("header_synonyms", "r1"),
    "instruction_politeness_r1": ("instruction_politeness", "r1"),
}
CANONICAL_WAVES = ("canonical_r1", "canonical_r2", "canonical_r3")
PROMPT_VARIANT_WAVES = (
    "whitespace_r1",
    "header_synonyms_r1",
    "instruction_politeness_r1",
)

DECISIONS = ("pass", "fail")
ALL_JUDGE_LABELS = ("pass", "fail", "no_decision")
PASS_LABELS = {"pass", "1", "1.0", "true", "yes", "correct", "met"}
FAIL_LABELS = {"fail", "0", "0.0", "false", "no", "incorrect", "not_met"}
CRITICALITIES = {"critical", "critical_negative"}
UNMAPPED_SKILL = "unmapped"
DEFAULT_THRESHOLDS = {
    "macro_f1_min": 0.80,
    "critical_failure_sensitivity_min": 0.90,
    "test_retest_worst_pairwise_strict_agreement_min": 0.90,
    "mapped_primary_skill_macro_f1_min": 0.70,
    "prompt_worst_flip_rate_max": 0.10,
}

PROVENANCE_FIELDS = (
    "judge_name",
    "judge_family",
    "judge_model",
    "resolved_provider_model",
    "judge_revision",
    "adapter",
    "prompt_version",
    "evidence_policy_version",
    "normalization_version",
    "configuration_hash",
    "frozen_configuration_hash",
    "prompt_variant",
    "replicate_id",
)
CROSS_WAVE_IDENTITY_FIELDS = (
    "judge_name",
    "judge_family",
    "judge_model",
    "resolved_provider_model",
    "judge_revision",
    "adapter",
    "prompt_version",
    "evidence_policy_version",
    "normalization_version",
    "frozen_configuration_hash",
)


class InputValidationError(ValueError):
    """Raised when study artifacts cannot be compared safely."""


@dataclass(frozen=True)
class HumanCase:
    case_id: str
    input_hash: str
    human_label: str
    candidate_family: str
    candidate_model: str
    scenario_id: str
    criterion_id: str
    primary_skill: str
    criticality: str
    raw: dict[str, str]


@dataclass(frozen=True)
class WaveData:
    judge: str
    judge_family: str
    wave: str
    path: Path
    file_sha256: str
    decisions: dict[str, str]
    prompt_hashes: dict[str, str]
    metadata: dict[str, object]
    status_counts: dict[str, int]


def _token(value: object) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _canonical_metadata_value(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _normalize_family(value: object, *, location: str) -> str:
    token = _token(value)
    aliases = {
        "openai": "openai",
        "gpt": "openai",
        "anthropic": "anthropic",
        "claude": "anthropic",
        "google": "google",
        "gemini": "google",
    }
    try:
        return aliases[token]
    except KeyError as exc:
        raise InputValidationError(
            f"{location}: unsupported model family {value!r}; expected openai, "
            "anthropic, or google/gemini"
        ) from exc


def _family_from_model(value: object) -> str | None:
    model = str(value or "").strip().lower()
    if not model:
        return None
    exact = {
        "gpt-5.5": "openai",
        "opus-4.8": "anthropic",
        "claude-opus-4-8": "anthropic",
        "gemini-3.5-flash": "google",
        "gemini-3.6-flash": "google",
    }
    if model in exact:
        return exact[model]
    prefix_families = (
        ("openai-group/", "openai"),
        ("openai/", "openai"),
        ("claude-group/", "anthropic"),
        ("anthropic/", "anthropic"),
        ("gemini-group/", "google"),
        ("google/", "google"),
    )
    for prefix, family in prefix_families:
        if model.startswith(prefix):
            return family
    return None


def _candidate_family(row: Mapping[str, str], *, location: str) -> str:
    explicit = str(row.get("candidate_family") or "").strip()
    explicit_family = (
        _normalize_family(explicit, location=f"{location}/candidate_family")
        if explicit
        else None
    )
    inferred_values = []
    for field in ("candidate_model_slug", "candidate_model"):
        value = row.get(field)
        family = _family_from_model(value)
        if value and family is None:
            raise InputValidationError(
                f"{location}: cannot infer candidate family from {field}={value!r}"
            )
        if family is not None:
            inferred_values.append((field, family))
    inferred_families = {family for _, family in inferred_values}
    if len(inferred_families) > 1:
        raise InputValidationError(
            f"{location}: candidate_model and candidate_model_slug imply different "
            f"families: {inferred_values}"
        )
    inferred = next(iter(inferred_families), None)
    if explicit_family and inferred and explicit_family != inferred:
        raise InputValidationError(
            f"{location}: candidate_family={explicit_family!r} conflicts with "
            f"model-derived family={inferred!r}"
        )
    result = explicit_family or inferred
    if result is None:
        raise InputValidationError(
            f"{location}: need candidate_family or a recognized candidate model/slug"
        )
    return result


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise InputValidationError(f"could not read {path}: {exc}") from exc
    return digest.hexdigest()


def _normalize_human_label(value: object, *, location: str) -> str:
    token = _token(value)
    if token in PASS_LABELS:
        return "pass"
    if token in FAIL_LABELS:
        return "fail"
    raise InputValidationError(
        f"{location}: human_label must be binary pass/fail, got {value!r}"
    )


def load_human_labels(path: Path) -> tuple[list[HumanCase], list[str]]:
    required = {
        "case_id",
        "case_input_hash",
        "human_label",
        "scenario_id",
        "criterion_id",
        "primary_skill",
        "criticality",
    }
    try:
        handle = path.open("r", encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise InputValidationError(f"could not open {path}: {exc}") from exc

    with handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise InputValidationError(f"{path}: missing CSV header")
        missing = sorted(required - set(fieldnames))
        if missing:
            raise InputValidationError(
                f"{path}: missing required column(s): {', '.join(missing)}"
            )
        if not {"candidate_family", "candidate_model", "candidate_model_slug"} & set(
            fieldnames
        ):
            raise InputValidationError(
                f"{path}: need candidate_family, candidate_model, or candidate_model_slug"
            )

        cases: list[HumanCase] = []
        seen: set[str] = set()
        for row_number, source in enumerate(reader, 2):
            raw = {name: str(source.get(name) or "") for name in fieldnames}
            location = f"{path}:{row_number}"
            case_id = raw["case_id"].strip()
            if not case_id or case_id in seen:
                raise InputValidationError(
                    f"{location}: blank or duplicate case_id {case_id!r}"
                )
            seen.add(case_id)
            input_hash = raw["case_input_hash"].strip()
            scenario_id = raw["scenario_id"].strip()
            criterion_id = raw["criterion_id"].strip()
            if not input_hash:
                raise InputValidationError(
                    f"{location}: case_input_hash is blank for {case_id}"
                )
            if not scenario_id or not criterion_id:
                raise InputValidationError(
                    f"{location}: scenario_id and criterion_id must be nonblank"
                )
            family = _candidate_family(raw, location=location)
            cases.append(
                HumanCase(
                    case_id=case_id,
                    input_hash=input_hash,
                    human_label=_normalize_human_label(
                        raw["human_label"], location=location
                    ),
                    candidate_family=family,
                    candidate_model=(
                        raw.get("candidate_model", "").strip()
                        or raw.get("candidate_model_slug", "").strip()
                    ),
                    scenario_id=scenario_id,
                    criterion_id=criterion_id,
                    primary_skill=_token(raw["primary_skill"]) or UNMAPPED_SKILL,
                    criticality=_token(raw["criticality"]),
                    raw=raw,
                )
            )
    if not cases:
        raise InputValidationError(f"{path}: contains no human-label rows")

    families = {case.candidate_family for case in cases}
    expected_families = set(TUTOR_FAMILIES)
    if families != expected_families:
        raise InputValidationError(
            "human labels must contain all three tutor families exactly; "
            f"expected={sorted(expected_families)}, got={sorted(families)}"
        )
    return cases, fieldnames


def _load_jsonl(path: Path) -> list[tuple[int, dict]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise InputValidationError(f"could not read {path}: {exc}") from exc
    rows: list[tuple[int, dict]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise InputValidationError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise InputValidationError(
                f"{path}:{line_number}: expected one JSON object per line"
            )
        rows.append((line_number, value))
    if not rows:
        raise InputValidationError(f"{path}: contains no judgment rows")
    return rows


def _uniform_metadata(path: Path, rows: Sequence[tuple[int, dict]]) -> dict[str, object]:
    metadata: dict[str, object] = {}
    for field in PROVENANCE_FIELDS:
        missing_lines = [line for line, row in rows if field not in row]
        if missing_lines:
            raise InputValidationError(
                f"{path}: provenance field {field!r} is missing "
                f"(first at line {missing_lines[0]})"
            )
        values: dict[str, object] = {}
        for _, row in rows:
            values[_canonical_metadata_value(row[field])] = row[field]
        if len(values) != 1:
            raise InputValidationError(
                f"{path}: rows do not share one provenance value for {field}"
            )
        value = next(iter(values.values()))
        if field != "judge_revision" and (
            value is None or (isinstance(value, str) and not value.strip())
        ):
            raise InputValidationError(f"{path}: provenance field {field} is blank")
        metadata[field] = value
    return metadata


def eligible_cases(cases: Sequence[HumanCase], judge: str) -> list[HumanCase]:
    judge_family = JUDGE_FAMILIES[judge]
    return [case for case in cases if case.candidate_family != judge_family]


def load_wave(
    path: Path,
    *,
    judge: str,
    wave: str,
    humans_by_id: Mapping[str, HumanCase],
) -> WaveData:
    if judge not in JUDGE_FAMILIES:
        raise InputValidationError(f"unknown frontier judge {judge!r}")
    if wave not in EXPECTED_WAVES:
        raise InputValidationError(f"unsupported wave {wave!r}")
    rows = _load_jsonl(path)
    metadata = _uniform_metadata(path, rows)
    if str(metadata["judge_name"]).strip() != judge:
        raise InputValidationError(
            f"{path}: judge_name {metadata['judge_name']!r} does not match {judge!r}"
        )
    expected_judge_family = JUDGE_FAMILIES[judge]
    actual_judge_family = _normalize_family(
        metadata["judge_family"], location=f"{path}/judge_family"
    )
    if actual_judge_family != expected_judge_family:
        raise InputValidationError(
            f"{path}: judge_family={actual_judge_family!r} does not match registered "
            f"family={expected_judge_family!r} for {judge}"
        )
    expected_variant, expected_replicate = EXPECTED_WAVES[wave]
    if str(metadata["prompt_variant"]).strip() != expected_variant:
        raise InputValidationError(
            f"{path}: wave {wave} requires prompt_variant={expected_variant!r}, got "
            f"{metadata['prompt_variant']!r}"
        )
    if str(metadata["replicate_id"]).strip() != expected_replicate:
        raise InputValidationError(
            f"{path}: wave {wave} requires replicate_id={expected_replicate!r}, got "
            f"{metadata['replicate_id']!r}"
        )

    decisions: dict[str, str] = {}
    prompt_hashes: dict[str, str] = {}
    status_counts: Counter[str] = Counter()
    for line_number, row in rows:
        location = f"{path}:{line_number}"
        case_id = str(row.get("case_id") or "").strip()
        if not case_id or case_id in decisions:
            raise InputValidationError(
                f"{location}: blank or duplicate case_id {case_id!r}"
            )
        human = humans_by_id.get(case_id)
        if human is None:
            raise InputValidationError(f"{location}: unknown case_id {case_id!r}")
        if human.candidate_family == expected_judge_family:
            raise InputValidationError(
                f"{location}: prohibited same-family judgment: judge {judge} "
                f"({expected_judge_family}) graded {case_id}"
            )
        row_family = _normalize_family(
            row.get("candidate_family"), location=f"{location}/candidate_family"
        )
        if row_family != human.candidate_family:
            raise InputValidationError(
                f"{location}: candidate_family={row_family!r} does not match human "
                f"routing family={human.candidate_family!r} for {case_id}"
            )
        if str(row.get("input_hash") or "").strip() != human.input_hash:
            raise InputValidationError(
                f"{location}: input_hash does not match human case_input_hash for {case_id}"
            )
        for field, expected in (
            ("scenario_id", human.scenario_id),
            ("criterion_id", human.criterion_id),
        ):
            if str(row.get(field) or "").strip() != expected:
                raise InputValidationError(
                    f"{location}: {field} does not match human labels for {case_id}"
                )
        prompt_hash = str(row.get("prompt_hash") or "").strip()
        if not prompt_hash:
            raise InputValidationError(f"{location}: prompt_hash is blank for {case_id}")
        status = _token(row.get("status"))
        if not status:
            raise InputValidationError(f"{location}: status is blank for {case_id}")
        verdict = _token(row.get("verdict"))
        if status == "ok":
            if verdict not in ALL_JUDGE_LABELS:
                raise InputValidationError(
                    f"{location}: unsupported verdict {row.get('verdict')!r} for status=ok"
                )
            decision = verdict
        else:
            decision = "no_decision"
        decisions[case_id] = decision
        prompt_hashes[case_id] = prompt_hash
        status_counts[status] += 1

    expected_ids = {
        case_id
        for case_id, human in humans_by_id.items()
        if human.candidate_family != expected_judge_family
    }
    actual_ids = set(decisions)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        raise InputValidationError(
            f"{path}: incomplete eligible coverage for {judge}; expected "
            f"{len(expected_ids)}, got {len(actual_ids)}; missing={missing[:5]}, "
            f"extra={extra[:5]}"
        )
    return WaveData(
        judge=judge,
        judge_family=expected_judge_family,
        wave=wave,
        path=path,
        file_sha256=file_sha256(path),
        decisions=decisions,
        prompt_hashes=prompt_hashes,
        metadata=metadata,
        status_counts=dict(sorted(status_counts.items())),
    )


def validate_judge_waves(judge: str, waves: Mapping[str, WaveData]) -> None:
    missing = sorted(set(EXPECTED_WAVES) - set(waves))
    extra = sorted(set(waves) - set(EXPECTED_WAVES))
    if missing or extra:
        raise InputValidationError(
            f"judge {judge!r} must provide exactly six waves; missing={missing}, "
            f"extra={extra}"
        )
    baseline = waves["canonical_r1"]
    for wave_name, wave in waves.items():
        for field in CROSS_WAVE_IDENTITY_FIELDS:
            if _canonical_metadata_value(wave.metadata[field]) != _canonical_metadata_value(
                baseline.metadata[field]
            ):
                raise InputValidationError(
                    f"judge {judge!r}: provenance field {field} differs between "
                    f"canonical_r1 and {wave_name}"
                )
        if set(wave.decisions) != set(baseline.decisions):
            raise InputValidationError(
                f"judge {judge!r}: eligible case set differs in {wave_name}"
            )
    for case_id, expected_hash in baseline.prompt_hashes.items():
        for wave_name in ("canonical_r2", "canonical_r3"):
            if waves[wave_name].prompt_hashes[case_id] != expected_hash:
                raise InputValidationError(
                    f"judge {judge!r}: canonical prompt_hash changed for {case_id} "
                    f"between canonical_r1 and {wave_name}"
                )
        for wave_name in PROMPT_VARIANT_WAVES:
            if waves[wave_name].prompt_hashes[case_id] == expected_hash:
                raise InputValidationError(
                    f"judge {judge!r}: prompt variant {wave_name} did not change "
                    f"prompt_hash for {case_id}"
                )


def validate_cross_judge_versions(
    waves_by_judge: Mapping[str, Mapping[str, WaveData]],
) -> None:
    expected_judges = set(JUDGE_FAMILIES)
    actual_judges = set(waves_by_judge)
    if actual_judges != expected_judges:
        raise InputValidationError(
            "comparison requires all three frontier judges; "
            f"missing={sorted(expected_judges - actual_judges)}, "
            f"extra={sorted(actual_judges - expected_judges)}"
        )
    for field in (
        "adapter",
        "prompt_version",
        "evidence_policy_version",
        "normalization_version",
    ):
        values = {
            judge: _canonical_metadata_value(waves["canonical_r1"].metadata[field])
            for judge, waves in waves_by_judge.items()
        }
        if len(set(values.values())) > 1:
            raise InputValidationError(
                f"judges do not share one {field}; refusing mixed-study comparison: "
                f"{values}"
            )


def _safe_div(numerator: int | float, denominator: int | float) -> float | None:
    return numerator / denominator if denominator else None


def _class_metrics(
    gold: Sequence[str], predictions: Sequence[str], label: str
) -> dict[str, int | float | None]:
    support = sum(value == label for value in gold)
    predicted_n = sum(value == label for value in predictions)
    true_positive = sum(
        actual == label and predicted == label
        for actual, predicted in zip(gold, predictions)
    )
    false_positive = predicted_n - true_positive
    false_negative = support - true_positive
    precision = _safe_div(true_positive, true_positive + false_positive)
    recall = _safe_div(true_positive, true_positive + false_negative)
    denominator = 2 * true_positive + false_positive + false_negative
    f1 = (2 * true_positive / denominator) if denominator else 0.0
    return {
        "support": support,
        "predicted_n": predicted_n,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def classification_metrics(
    cases: Sequence[HumanCase], predictions: Mapping[str, str]
) -> dict:
    gold = [case.human_label for case in cases]
    predicted = [predictions[case.case_id] for case in cases]
    n = len(cases)
    confusion = {
        actual: {decision: 0 for decision in ALL_JUDGE_LABELS} for actual in DECISIONS
    }
    for actual, decision in zip(gold, predicted):
        confusion[actual][decision] += 1
    per_class = {label: _class_metrics(gold, predicted, label) for label in DECISIONS}
    macro_f1 = (
        sum(float(per_class[label]["f1"]) for label in DECISIONS) / len(DECISIONS)
        if n
        else None
    )
    weighted_f1 = (
        sum(
            int(per_class[label]["support"]) * float(per_class[label]["f1"])
            for label in DECISIONS
        )
        / n
        if n
        else None
    )
    decided_pairs = [
        (actual, decision)
        for actual, decision in zip(gold, predicted)
        if decision in DECISIONS
    ]
    tp = sum(actual == "fail" and decision == "fail" for actual, decision in decided_pairs)
    tn = sum(actual == "pass" and decision == "pass" for actual, decision in decided_pairs)
    fp = sum(actual == "pass" and decision == "fail" for actual, decision in decided_pairs)
    fn = sum(actual == "fail" and decision == "pass" for actual, decision in decided_pairs)
    mcc_denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / mcc_denominator if mcc_denominator else None
    decided_n = len(decided_pairs)
    correct = confusion["pass"]["pass"] + confusion["fail"]["fail"]
    pass_recall = per_class["pass"]["recall"]
    fail_recall = per_class["fail"]["recall"]
    balanced_accuracy = (
        (float(pass_recall) + float(fail_recall)) / 2
        if pass_recall is not None and fail_recall is not None
        else None
    )
    return {
        "n": n,
        "human_pass_n": int(per_class["pass"]["support"]),
        "human_fail_n": int(per_class["fail"]["support"]),
        "decided_n": decided_n,
        "no_decision_n": n - decided_n,
        "coverage": _safe_div(decided_n, n),
        "correct_n": correct,
        "accuracy": _safe_div(correct, n),
        "conditional_accuracy": _safe_div(correct, decided_n),
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "mcc": mcc,
        "mcc_scope": "decided_cases_only",
        "pass_recall": pass_recall,
        "fail_recall": fail_recall,
        "false_pass_rate": _safe_div(confusion["fail"]["pass"], int(per_class["fail"]["support"])),
        "per_class": per_class,
        "confusion": confusion,
    }


def critical_failure_metrics(
    cases: Sequence[HumanCase], predictions: Mapping[str, str]
) -> dict:
    eligible = [
        case
        for case in cases
        if case.criticality in CRITICALITIES and case.human_label == "fail"
    ]
    detected = sum(predictions[case.case_id] == "fail" for case in eligible)
    no_decision = sum(
        predictions[case.case_id] == "no_decision" for case in eligible
    )
    by_criticality = {}
    for criticality in sorted(CRITICALITIES):
        subset = [case for case in eligible if case.criticality == criticality]
        subset_detected = sum(predictions[case.case_id] == "fail" for case in subset)
        by_criticality[criticality] = {
            "critical_human_failure_n": len(subset),
            "detected_n": subset_detected,
            "sensitivity": _safe_div(subset_detected, len(subset)),
        }
    return {
        "included_criticalities": sorted(CRITICALITIES),
        "critical_human_failure_n": len(eligible),
        "detected_n": detected,
        "missed_n": len(eligible) - detected,
        "no_decision_n": no_decision,
        "sensitivity": _safe_div(detected, len(eligible)),
        "by_criticality": by_criticality,
    }


def performance_report(
    cases: Sequence[HumanCase], predictions: Mapping[str, str]
) -> dict:
    report = classification_metrics(cases, predictions)
    report["critical_failures"] = critical_failure_metrics(cases, predictions)
    by_skill = {}
    for skill in sorted({case.primary_skill for case in cases}):
        subset = [case for case in cases if case.primary_skill == skill]
        by_skill[skill] = classification_metrics(subset, predictions)
        by_skill[skill]["mapped"] = skill != UNMAPPED_SKILL
    report["by_primary_skill"] = by_skill
    return report


def _cohen_kappa(first: Sequence[str], second: Sequence[str]) -> float | None:
    if not first or len(first) != len(second):
        return None
    n = len(first)
    observed = sum(left == right for left, right in zip(first, second)) / n
    first_counts = Counter(first)
    second_counts = Counter(second)
    expected = sum(
        first_counts[label] * second_counts[label] for label in ALL_JUDGE_LABELS
    ) / (n * n)
    return (observed - expected) / (1 - expected) if expected < 1 else None


def _pairwise_reliability(
    cases: Sequence[HumanCase],
    first: Mapping[str, str],
    second: Mapping[str, str],
) -> dict:
    left = [first[case.case_id] for case in cases]
    right = [second[case.case_id] for case in cases]
    n = len(cases)
    raw_agreement = sum(a == b for a, b in zip(left, right))
    strict_agreement = sum(a == b and a in DECISIONS for a, b in zip(left, right))
    both_decided = sum(a in DECISIONS and b in DECISIONS for a, b in zip(left, right))
    binary_disagreement = sum(
        a in DECISIONS and b in DECISIONS and a != b for a, b in zip(left, right)
    )
    return {
        "n": n,
        "raw_agreement_n": raw_agreement,
        "raw_agreement_rate": _safe_div(raw_agreement, n),
        "strict_agreement_n": strict_agreement,
        "strict_agreement_rate": _safe_div(strict_agreement, n),
        "both_decided_n": both_decided,
        "no_decision_involved_n": n - both_decided,
        "binary_disagreement_n": binary_disagreement,
        "binary_disagreement_rate_among_both_decided": _safe_div(
            binary_disagreement, both_decided
        ),
        "cohen_kappa": _cohen_kappa(left, right),
        "cohen_kappa_labels": list(ALL_JUDGE_LABELS),
    }


def test_retest_report(
    cases: Sequence[HumanCase], waves: Mapping[str, WaveData]
) -> dict:
    pair_names = (
        ("canonical_r1", "canonical_r2"),
        ("canonical_r1", "canonical_r3"),
        ("canonical_r2", "canonical_r3"),
    )
    pairwise = {}
    for first, second in pair_names:
        pairwise[f"{first}_vs_{second}"] = _pairwise_reliability(
            cases, waves[first].decisions, waves[second].decisions
        )
    labels = [
        tuple(waves[wave].decisions[case.case_id] for wave in CANONICAL_WAVES)
        for case in cases
    ]
    exact_raw = sum(len(set(values)) == 1 for values in labels)
    exact_strict = sum(
        len(set(values)) == 1 and values[0] in DECISIONS for values in labels
    )
    strict_rates = [pair["strict_agreement_rate"] for pair in pairwise.values()]
    return {
        "n": len(cases),
        "canonical_waves": list(CANONICAL_WAVES),
        "exact_three_repeat_agreement_n": exact_raw,
        "exact_three_repeat_agreement_rate": _safe_div(exact_raw, len(cases)),
        "exact_three_repeat_strict_agreement_n": exact_strict,
        "exact_three_repeat_strict_agreement_rate": _safe_div(
            exact_strict, len(cases)
        ),
        "strict_definition": (
            "labels must agree and be pass/fail; any no_decision is inconsistent"
        ),
        "pairwise": pairwise,
        "worst_pairwise_strict_agreement_rate": (
            min(strict_rates) if strict_rates else None
        ),
    }


def _variant_consistency(
    cases: Sequence[HumanCase],
    canonical: Mapping[str, str],
    variant: Mapping[str, str],
) -> dict:
    pairs = [(canonical[case.case_id], variant[case.case_id]) for case in cases]
    n = len(pairs)
    both_decided = sum(left in DECISIONS and right in DECISIONS for left, right in pairs)
    binary_flips = sum(
        left in DECISIONS and right in DECISIONS and left != right
        for left, right in pairs
    )
    raw_disagreements = sum(left != right for left, right in pairs)
    strict_flips = sum(
        left != right or left == "no_decision" or right == "no_decision"
        for left, right in pairs
    )
    return {
        "n": n,
        "both_decided_n": both_decided,
        "no_decision_involved_n": n - both_decided,
        "binary_flip_n": binary_flips,
        "binary_flip_rate_among_both_decided": _safe_div(binary_flips, both_decided),
        "raw_disagreement_n": raw_disagreements,
        "raw_disagreement_rate": _safe_div(raw_disagreements, n),
        "flip_n": strict_flips,
        "flip_rate": _safe_div(strict_flips, n),
        "flip_definition": (
            "different labels or any no_decision; two no_decisions remain inconsistent"
        ),
    }


def prompt_consistency_report(
    cases: Sequence[HumanCase], waves: Mapping[str, WaveData]
) -> dict:
    canonical = waves["canonical_r1"].decisions
    variants = {
        wave: _variant_consistency(cases, canonical, waves[wave].decisions)
        for wave in PROMPT_VARIANT_WAVES
    }
    rates = [value["flip_rate"] for value in variants.values()]
    any_flip = 0
    for case in cases:
        base = canonical[case.case_id]
        if any(
            base != waves[wave].decisions[case.case_id]
            or base == "no_decision"
            or waves[wave].decisions[case.case_id] == "no_decision"
            for wave in PROMPT_VARIANT_WAVES
        ):
            any_flip += 1
    pooled_n = len(cases) * len(PROMPT_VARIANT_WAVES)
    pooled_flips = sum(value["flip_n"] for value in variants.values())
    return {
        "canonical_wave": "canonical_r1",
        "variants": variants,
        "worst_variant_flip_rate": max(rates) if rates else None,
        "pooled_comparison_n": pooled_n,
        "pooled_flip_n": pooled_flips,
        "pooled_flip_rate": _safe_div(pooled_flips, pooled_n),
        "any_variant_flip_case_n": any_flip,
        "any_variant_flip_case_rate": _safe_div(any_flip, len(cases)),
    }


def _binomial_two_sided_p(discordant_first: int, discordant_second: int) -> float | None:
    """Exact two-sided McNemar/binomial p-value (case-level, descriptive only)."""

    total = discordant_first + discordant_second
    if total == 0:
        return 1.0
    smaller = min(discordant_first, discordant_second)
    lower_tail = sum(math.comb(total, value) for value in range(smaller + 1)) / (2**total)
    return min(1.0, 2 * lower_tail)


def cross_judge_pair_report(
    cases: Sequence[HumanCase],
    waves_by_judge: Mapping[str, Mapping[str, WaveData]],
) -> dict[str, dict]:
    reports: dict[str, dict] = {}
    for first, second in combinations(JUDGE_FAMILIES, 2):
        first_family = JUDGE_FAMILIES[first]
        second_family = JUDGE_FAMILIES[second]
        shared = [
            case
            for case in cases
            if case.candidate_family not in {first_family, second_family}
        ]
        shared_families = sorted({case.candidate_family for case in shared})
        if len(shared_families) != 1:
            raise InputValidationError(
                f"pair {first}/{second} should share exactly one tutor family, got "
                f"{shared_families}"
            )
        first_predictions = waves_by_judge[first]["canonical_r1"].decisions
        second_predictions = waves_by_judge[second]["canonical_r1"].decisions
        first_correct_second_wrong = 0
        second_correct_first_wrong = 0
        both_correct = 0
        both_wrong = 0
        for case in shared:
            first_correct = first_predictions[case.case_id] == case.human_label
            second_correct = second_predictions[case.case_id] == case.human_label
            if first_correct and second_correct:
                both_correct += 1
            elif first_correct:
                first_correct_second_wrong += 1
            elif second_correct:
                second_correct_first_wrong += 1
            else:
                both_wrong += 1
        first_performance = performance_report(shared, first_predictions)
        second_performance = performance_report(shared, second_predictions)
        key = f"{first}_vs_{second}"
        reports[key] = {
            "judges": [first, second],
            "shared_candidate_family": shared_families[0],
            "n": len(shared),
            "agreement": _pairwise_reliability(
                shared, first_predictions, second_predictions
            ),
            "human_performance": {
                first: first_performance,
                second: second_performance,
            },
            "accuracy_delta_first_minus_second": (
                float(first_performance["accuracy"])
                - float(second_performance["accuracy"])
                if first_performance["accuracy"] is not None
                and second_performance["accuracy"] is not None
                else None
            ),
            "correctness_pairing": {
                "both_correct_n": both_correct,
                "first_correct_second_wrong_n": first_correct_second_wrong,
                "second_correct_first_wrong_n": second_correct_first_wrong,
                "both_wrong_n": both_wrong,
                "exact_mcnemar_p_case_level_descriptive": _binomial_two_sided_p(
                    first_correct_second_wrong, second_correct_first_wrong
                ),
                "inference_warning": (
                    "criterion rows are clustered within scenarios; use the "
                    "scenario-clustered bootstrap interval for inference"
                ),
            },
        }
    return reports


def consensus_report(
    cases: Sequence[HumanCase],
    waves_by_judge: Mapping[str, Mapping[str, WaveData]],
    *,
    include_review_rows: bool = False,
) -> tuple[dict, list[dict[str, str]]]:
    canonical = {
        judge: waves["canonical_r1"].decisions
        for judge, waves in waves_by_judge.items()
    }
    predictions: dict[str, str] = {}
    review_rows: list[dict[str, str]] = []
    binary_disagreement_n = 0
    no_decision_review_n = 0
    by_reason: Counter[str] = Counter()
    for case in cases:
        judges = [
            judge
            for judge, family in JUDGE_FAMILIES.items()
            if family != case.candidate_family
        ]
        if len(judges) != 2:
            raise InputValidationError(
                f"{case.case_id}: expected exactly two eligible judges, got {judges}"
            )
        first, second = judges
        left = canonical[first][case.case_id]
        right = canonical[second][case.case_id]
        if left in DECISIONS and left == right:
            predictions[case.case_id] = left
            continue
        predictions[case.case_id] = "no_decision"
        if left == "no_decision" or right == "no_decision":
            reason = "no_decision_involved"
            no_decision_review_n += 1
        else:
            reason = "binary_disagreement"
            binary_disagreement_n += 1
        by_reason[reason] += 1
        if include_review_rows:
            review_rows.append(
                {
                    "case_id": case.case_id,
                    "candidate_model": case.candidate_model,
                    "candidate_family": case.candidate_family,
                    "scenario_id": case.scenario_id,
                    "criterion_id": case.criterion_id,
                    "primary_skill": case.primary_skill,
                    "criticality": case.criticality,
                    "human_label": case.human_label,
                    "judge_1": first,
                    "judge_1_verdict": left,
                    "judge_2": second,
                    "judge_2_verdict": right,
                    "review_reason": reason,
                }
            )
    performance = classification_metrics(cases, predictions)
    critical = critical_failure_metrics(cases, predictions)
    auto_n = int(performance["decided_n"])
    n = len(cases)
    report = {
        "n": n,
        "judges_per_case": 2,
        "policy": (
            "auto-label only when both eligible judges return the same binary "
            "verdict; route every disagreement or no_decision to human review"
        ),
        "auto_labeled_n": auto_n,
        "auto_label_coverage": _safe_div(auto_n, n),
        "human_review_n": n - auto_n,
        "human_review_rate": _safe_div(n - auto_n, n),
        "binary_disagreement_n": binary_disagreement_n,
        "no_decision_review_n": no_decision_review_n,
        "review_reason_counts": dict(sorted(by_reason.items())),
        "end_to_end_performance": performance,
        "critical_failures": critical,
    }
    return report, review_rows


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def _interval(values: Iterable[float | None], requested_samples: int) -> dict | None:
    finite = sorted(
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    )
    if not finite:
        return None
    return {
        "lower": _percentile(finite, 0.025),
        "upper": _percentile(finite, 0.975),
        "confidence_level": 0.95,
        "valid_resamples": len(finite),
        "requested_resamples": requested_samples,
    }


def scenario_cluster_resamples(
    cases: Sequence[HumanCase], *, samples: int, seed: int
) -> list[list[HumanCase]]:
    if samples < 2:
        return []
    clusters: dict[str, list[HumanCase]] = defaultdict(list)
    for case in cases:
        clusters[case.scenario_id].append(case)
    cluster_ids = sorted(clusters)
    if len(cluster_ids) < 2:
        return []
    rng = random.Random(seed)
    result = []
    for _ in range(samples):
        sample: list[HumanCase] = []
        for _ in cluster_ids:
            sample.extend(clusters[rng.choice(cluster_ids)])
        result.append(sample)
    return result


def _bootstrap_intervals(
    resamples: Sequence[Sequence[HumanCase]],
    waves_by_judge: Mapping[str, Mapping[str, WaveData]],
    *,
    requested_samples: int,
) -> dict:
    if not resamples:
        return {"judges": {}, "pairs": {}, "consensus": {}}
    judge_values: dict[str, dict[str, list[float | None]]] = {
        judge: defaultdict(list) for judge in JUDGE_FAMILIES
    }
    pair_values: dict[str, dict[str, list[float | None]]] = defaultdict(
        lambda: defaultdict(list)
    )
    consensus_values: dict[str, list[float | None]] = defaultdict(list)

    for sample in resamples:
        for judge in JUDGE_FAMILIES:
            subset = eligible_cases(sample, judge)
            waves = waves_by_judge[judge]
            canonical = performance_report(subset, waves["canonical_r1"].decisions)
            values = judge_values[judge]
            values["canonical.accuracy"].append(canonical["accuracy"])
            values["canonical.macro_f1"].append(canonical["macro_f1"])
            values["canonical.coverage"].append(canonical["coverage"])
            values["canonical.critical_sensitivity"].append(
                canonical["critical_failures"]["sensitivity"]
            )
            values["retest.worst_pairwise_strict"].append(
                test_retest_report(subset, waves)[
                    "worst_pairwise_strict_agreement_rate"
                ]
            )
            values["prompt.worst_flip"].append(
                prompt_consistency_report(subset, waves)["worst_variant_flip_rate"]
            )

        pairs = cross_judge_pair_report(sample, waves_by_judge)
        for pair_name, pair in pairs.items():
            pair_values[pair_name]["raw_agreement"].append(
                pair["agreement"]["raw_agreement_rate"]
            )
            pair_values[pair_name]["strict_agreement"].append(
                pair["agreement"]["strict_agreement_rate"]
            )
            pair_values[pair_name]["accuracy_delta"].append(
                pair["accuracy_delta_first_minus_second"]
            )

        consensus, _ = consensus_report(sample, waves_by_judge)
        consensus_values["auto_label_coverage"].append(
            consensus["auto_label_coverage"]
        )
        consensus_values["human_review_rate"].append(
            consensus["human_review_rate"]
        )
        consensus_values["accuracy"].append(
            consensus["end_to_end_performance"]["accuracy"]
        )
        consensus_values["conditional_accuracy"].append(
            consensus["end_to_end_performance"]["conditional_accuracy"]
        )
        consensus_values["critical_sensitivity"].append(
            consensus["critical_failures"]["sensitivity"]
        )

    return {
        "judges": {
            judge: {
                metric: _interval(values, requested_samples)
                for metric, values in metrics.items()
            }
            for judge, metrics in judge_values.items()
        },
        "pairs": {
            pair: {
                metric: _interval(values, requested_samples)
                for metric, values in metrics.items()
            }
            for pair, metrics in pair_values.items()
        },
        "consensus": {
            metric: _interval(values, requested_samples)
            for metric, values in consensus_values.items()
        },
    }


def _family_macro(canonical_by_family: Mapping[str, Mapping[str, object]]) -> dict:
    metrics = ("accuracy", "macro_f1", "balanced_accuracy", "coverage")
    result: dict[str, float | None] = {}
    for metric in metrics:
        values = [
            float(report[metric])
            for report in canonical_by_family.values()
            if report[metric] is not None
        ]
        result[metric] = sum(values) / len(values) if values else None
    sensitivity_values = [
        float(report["critical_failures"]["sensitivity"])
        for report in canonical_by_family.values()
        if report["critical_failures"]["sensitivity"] is not None
    ]
    result["critical_failure_sensitivity"] = (
        sum(sensitivity_values) / len(sensitivity_values)
        if sensitivity_values
        else None
    )
    result["definition"] = "unweighted mean across the two eligible tutor families"
    return result


def _threshold_check(
    value: float | None, *, threshold: float, operator: str
) -> dict[str, object]:
    if operator == ">=":
        passed = value is not None and value >= threshold
    elif operator == "<=":
        passed = value is not None and value <= threshold
    else:  # pragma: no cover - internal programming guard
        raise ValueError(f"unsupported threshold operator {operator}")
    return {
        "value": value,
        "operator": operator,
        "threshold": threshold,
        "passed": passed,
    }


def acceptance_report(judge_report: Mapping[str, object]) -> dict:
    """Apply the pre-existing judge-validation thresholds to one eligible set."""

    canonical = judge_report["canonical"]  # type: ignore[index]
    retest = judge_report["test_retest"]  # type: ignore[index]
    prompt = judge_report["prompt_consistency"]  # type: ignore[index]
    mapped_skill_checks = {
        skill: _threshold_check(
            metrics["macro_f1"],
            threshold=DEFAULT_THRESHOLDS["mapped_primary_skill_macro_f1_min"],
            operator=">=",
        )
        for skill, metrics in canonical["by_primary_skill"].items()
        if metrics["mapped"]
    }
    checks = {
        "macro_f1": _threshold_check(
            canonical["macro_f1"],
            threshold=DEFAULT_THRESHOLDS["macro_f1_min"],
            operator=">=",
        ),
        "critical_failure_sensitivity": _threshold_check(
            canonical["critical_failures"]["sensitivity"],
            threshold=DEFAULT_THRESHOLDS["critical_failure_sensitivity_min"],
            operator=">=",
        ),
        "test_retest_worst_pairwise_strict_agreement": _threshold_check(
            retest["worst_pairwise_strict_agreement_rate"],
            threshold=DEFAULT_THRESHOLDS[
                "test_retest_worst_pairwise_strict_agreement_min"
            ],
            operator=">=",
        ),
        "prompt_worst_flip_rate": _threshold_check(
            prompt["worst_variant_flip_rate"],
            threshold=DEFAULT_THRESHOLDS["prompt_worst_flip_rate_max"],
            operator="<=",
        ),
    }
    mapped_skills_passed = bool(mapped_skill_checks) and all(
        bool(check["passed"]) for check in mapped_skill_checks.values()
    )
    checks["mapped_primary_skills"] = {
        "operator": ">=",
        "threshold": DEFAULT_THRESHOLDS["mapped_primary_skill_macro_f1_min"],
        "passed": mapped_skills_passed,
        "skills": mapped_skill_checks,
        "unmapped_excluded_from_acceptance": True,
    }
    return {
        "passed": all(bool(check["passed"]) for check in checks.values()),
        "checks": checks,
        "scope": (
            "the judge's two eligible out-of-family tutor sets; acceptance results "
            "across judges are not a common-case ranking"
        ),
    }


def build_report(
    human_path: Path,
    cases: Sequence[HumanCase],
    waves_by_judge: Mapping[str, Mapping[str, WaveData]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> tuple[dict, list[dict[str, str]]]:
    validate_cross_judge_versions(waves_by_judge)
    resamples = scenario_cluster_resamples(
        cases, samples=bootstrap_samples, seed=seed
    )
    intervals = _bootstrap_intervals(
        resamples,
        waves_by_judge,
        requested_samples=bootstrap_samples,
    )

    judges: dict[str, dict] = {}
    input_waves: dict[str, dict] = {}
    for judge in JUDGE_FAMILIES:
        waves = waves_by_judge[judge]
        subset = eligible_cases(cases, judge)
        canonical = performance_report(subset, waves["canonical_r1"].decisions)
        by_family = {}
        canonical_by_family = {}
        for family in TUTOR_FAMILIES:
            if family == JUDGE_FAMILIES[judge]:
                continue
            family_cases = [case for case in subset if case.candidate_family == family]
            family_canonical = performance_report(
                family_cases, waves["canonical_r1"].decisions
            )
            canonical_by_family[family] = family_canonical
            by_family[family] = {
                "n": len(family_cases),
                "canonical": family_canonical,
                "test_retest": test_retest_report(family_cases, waves),
                "prompt_consistency": prompt_consistency_report(family_cases, waves),
            }
        judges[judge] = {
            "judge_family": JUDGE_FAMILIES[judge],
            "excluded_same_family": JUDGE_FAMILIES[judge],
            "eligible_candidate_families": sorted(canonical_by_family),
            "provenance": {
                field: waves["canonical_r1"].metadata[field]
                for field in CROSS_WAVE_IDENTITY_FIELDS
            },
            "waves": {
                wave: performance_report(subset, waves[wave].decisions)
                for wave in EXPECTED_WAVES
            },
            "canonical": canonical,
            "family_macro": _family_macro(canonical_by_family),
            "test_retest": test_retest_report(subset, waves),
            "prompt_consistency": prompt_consistency_report(subset, waves),
            "by_candidate_family": by_family,
            "confidence_intervals_95": intervals["judges"].get(judge, {}),
        }
        input_waves[judge] = {
            wave: {
                "path": str(waves[wave].path),
                "sha256": waves[wave].file_sha256,
                "configuration_hash": waves[wave].metadata["configuration_hash"],
                "prompt_variant": waves[wave].metadata["prompt_variant"],
                "replicate_id": waves[wave].metadata["replicate_id"],
                "status_counts": waves[wave].status_counts,
                "eligible_case_n": len(waves[wave].decisions),
            }
            for wave in EXPECTED_WAVES
        }

    pairs = cross_judge_pair_report(cases, waves_by_judge)
    for pair_name, pair in pairs.items():
        pair["confidence_intervals_95"] = intervals["pairs"].get(pair_name, {})

    consensus, review_rows = consensus_report(
        cases, waves_by_judge, include_review_rows=True
    )
    consensus["confidence_intervals_95"] = intervals["consensus"]
    consensus["by_candidate_family"] = {}
    for family in TUTOR_FAMILIES:
        family_cases = [case for case in cases if case.candidate_family == family]
        family_report, _ = consensus_report(family_cases, waves_by_judge)
        consensus["by_candidate_family"][family] = family_report

    family_counts = Counter(case.candidate_family for case in cases)
    human_counts = Counter(case.human_label for case in cases)
    human_by_family = {}
    for family in TUTOR_FAMILIES:
        family_cases = [case for case in cases if case.candidate_family == family]
        labels = Counter(case.human_label for case in family_cases)
        human_by_family[family] = {
            "n": len(family_cases),
            "pass_n": labels["pass"],
            "fail_n": labels["fail"],
            "critical_human_failure_n": sum(
                case.human_label == "fail" and case.criticality in CRITICALITIES
                for case in family_cases
            ),
        }
    acceptance_by_judge = {
        judge: acceptance_report(judge_report)
        for judge, judge_report in judges.items()
    }
    report = {
        "schema_version": "frontier-judge-comparison-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": {
            "human_labels": {
                "path": str(human_path),
                "sha256": file_sha256(human_path),
            },
            "waves": input_waves,
        },
        "study_design": {
            "design": "leave-own-tutor-family-out",
            "judge_families": dict(JUDGE_FAMILIES),
            "eligibility_rule": "judge_family != candidate_family",
            "expected_judges_per_case": 2,
            "same_family_skips_are_no_decisions": False,
            "comparison_warning": (
                "each judge omits a different tutor family and each judge pair "
                "overlaps on only one family; pooled judge rankings remain "
                "candidate-family-confounded"
            ),
        },
        "human_gold": {
            "criterion_case_n": len(cases),
            "unique_scenario_n": len({case.scenario_id for case in cases}),
            "pass_n": human_counts["pass"],
            "fail_n": human_counts["fail"],
            "candidate_family_counts": dict(sorted(family_counts.items())),
            "by_candidate_family": human_by_family,
        },
        "settings": {
            "performance_wave": "canonical_r1",
            "bootstrap_method": "scenario_clustered_percentile",
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_valid": bool(resamples),
            "seed": seed,
            "no_decision_policy": {
                "eligible_api_or_parse_failure": (
                    "counts as no_decision and reduces end-to-end performance/coverage"
                ),
                "ineligible_same_family_pair": "excluded; never treated as no_decision",
                "test_retest_strict_agreement": "inconsistent",
                "prompt_flip_rate": "inconsistent, including two no_decisions",
                "consensus": "requires human review",
            },
        },
        "judges": judges,
        "cross_judge_pairs": pairs,
        "two_judge_consensus": consensus,
        "acceptance": {
            "thresholds": dict(DEFAULT_THRESHOLDS),
            "judges": acceptance_by_judge,
            "all_judges_pass": all(
                result["passed"] for result in acceptance_by_judge.values()
            ),
            "comparison_caveat": (
                "thresholds are applied within each judge's eligible two-family "
                "scope; pass/fail outcomes are not an unconfounded pooled ranking"
            ),
        },
    }
    return report, review_rows


def _csv_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "unknown"


def write_csv_summary(path: Path, report: Mapping[str, object]) -> None:
    fields = [
        "judge",
        "judge_family",
        "eligible_case_n",
        "canonical_accuracy",
        "canonical_macro_f1",
        "canonical_balanced_accuracy",
        "canonical_critical_failure_sensitivity",
        "canonical_coverage",
        "canonical_mcc_decided_only",
        "family_macro_accuracy",
        "family_macro_macro_f1",
        "family_macro_critical_failure_sensitivity",
        "exact_three_repeat_strict_agreement_rate",
        "worst_pairwise_strict_agreement_rate",
        "prompt_worst_flip_rate",
        "acceptance_passed",
    ]
    for family in TUTOR_FAMILIES:
        family_name = _csv_name(family)
        fields.extend(
            [
                f"family_{family_name}_n",
                f"family_{family_name}_accuracy",
                f"family_{family_name}_macro_f1",
                f"family_{family_name}_critical_failure_sensitivity",
                f"family_{family_name}_coverage",
            ]
        )
    rows = []
    judges = report["judges"]  # type: ignore[index]
    for judge, metrics in judges.items():  # type: ignore[union-attr]
        canonical = metrics["canonical"]
        family_macro = metrics["family_macro"]
        row = {
            "judge": judge,
            "judge_family": metrics["judge_family"],
            "eligible_case_n": canonical["n"],
            "canonical_accuracy": canonical["accuracy"],
            "canonical_macro_f1": canonical["macro_f1"],
            "canonical_balanced_accuracy": canonical["balanced_accuracy"],
            "canonical_critical_failure_sensitivity": canonical[
                "critical_failures"
            ]["sensitivity"],
            "canonical_coverage": canonical["coverage"],
            "canonical_mcc_decided_only": canonical["mcc"],
            "family_macro_accuracy": family_macro["accuracy"],
            "family_macro_macro_f1": family_macro["macro_f1"],
            "family_macro_critical_failure_sensitivity": family_macro[
                "critical_failure_sensitivity"
            ],
            "exact_three_repeat_strict_agreement_rate": metrics["test_retest"][
                "exact_three_repeat_strict_agreement_rate"
            ],
            "worst_pairwise_strict_agreement_rate": metrics["test_retest"][
                "worst_pairwise_strict_agreement_rate"
            ],
            "prompt_worst_flip_rate": metrics["prompt_consistency"][
                "worst_variant_flip_rate"
            ],
            "acceptance_passed": report["acceptance"]["judges"][judge][
                "passed"
            ],
        }
        for family in TUTOR_FAMILIES:
            prefix = f"family_{_csv_name(family)}"
            family_report = metrics["by_candidate_family"].get(family)
            if family_report is None:
                row[f"{prefix}_n"] = ""
                row[f"{prefix}_accuracy"] = ""
                row[f"{prefix}_macro_f1"] = ""
                row[f"{prefix}_critical_failure_sensitivity"] = ""
                row[f"{prefix}_coverage"] = ""
            else:
                family_canonical = family_report["canonical"]
                row[f"{prefix}_n"] = family_canonical["n"]
                row[f"{prefix}_accuracy"] = family_canonical["accuracy"]
                row[f"{prefix}_macro_f1"] = family_canonical["macro_f1"]
                row[f"{prefix}_critical_failure_sensitivity"] = family_canonical[
                    "critical_failures"
                ]["sensitivity"]
                row[f"{prefix}_coverage"] = family_canonical["coverage"]
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


REVIEW_FIELDS = (
    "case_id",
    "candidate_model",
    "candidate_family",
    "scenario_id",
    "criterion_id",
    "primary_skill",
    "criticality",
    "human_label",
    "judge_1",
    "judge_1_verdict",
    "judge_2",
    "judge_2_verdict",
    "review_reason",
)


def write_review_queue(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _rate(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _print_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    rendered = [[str(value) for value in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in rendered:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]
    print("  ".join(header.ljust(width) for header, width in zip(headers, widths)))
    print("  ".join("-" * width for width in widths))
    for row in rendered:
        print("  ".join(value.ljust(width) for value, width in zip(row, widths)))


def print_report(report: Mapping[str, object]) -> None:
    gold = report["human_gold"]  # type: ignore[index]
    print(
        "Frontier judge comparison "
        f"({gold['criterion_case_n']} cases; two eligible judges per case)"
    )
    rows = []
    for judge, metrics in report["judges"].items():  # type: ignore[index,union-attr]
        canonical = metrics["canonical"]
        rows.append(
            [
                judge,
                canonical["n"],
                _rate(canonical["accuracy"]),
                _rate(canonical["macro_f1"]),
                _rate(canonical["critical_failures"]["sensitivity"]),
                _rate(canonical["coverage"]),
                _rate(
                    metrics["test_retest"][
                        "worst_pairwise_strict_agreement_rate"
                    ]
                ),
                _rate(metrics["prompt_consistency"]["worst_variant_flip_rate"]),
                (
                    "PASS"
                    if report["acceptance"]["judges"][judge]["passed"]
                    else "FAIL"
                ),
            ]
        )
    _print_table(
        [
            "judge",
            "eligible N",
            "accuracy",
            "macro-F1",
            "critical sens.",
            "coverage",
            "retest worst",
            "prompt worst",
            "accept",
        ],
        rows,
    )

    print("\nPer-tutor-family canonical performance")
    family_rows = []
    for judge, metrics in report["judges"].items():  # type: ignore[index,union-attr]
        for family, family_report in metrics["by_candidate_family"].items():
            canonical = family_report["canonical"]
            family_rows.append(
                [
                    judge,
                    family,
                    canonical["n"],
                    _rate(canonical["accuracy"]),
                    _rate(canonical["macro_f1"]),
                    _rate(canonical["critical_failures"]["sensitivity"]),
                ]
            )
    _print_table(
        ["judge", "tutor family", "N", "accuracy", "macro-F1", "critical sens."],
        family_rows,
    )

    print("\nCross-judge agreement on shared eligible cases")
    pair_rows = []
    for pair in report["cross_judge_pairs"].values():  # type: ignore[index,union-attr]
        pair_rows.append(
            [
                " vs ".join(pair["judges"]),
                pair["shared_candidate_family"],
                pair["n"],
                _rate(pair["agreement"]["strict_agreement_rate"]),
                _rate(pair["agreement"]["cohen_kappa"]),
                _rate(pair["accuracy_delta_first_minus_second"]),
            ]
        )
    _print_table(
        ["pair", "shared family", "N", "strict agree", "kappa", "accuracy delta"],
        pair_rows,
    )

    consensus = report["two_judge_consensus"]  # type: ignore[index]
    performance = consensus["end_to_end_performance"]
    print(
        "\nTwo-judge consensus: "
        f"auto-label {consensus['auto_labeled_n']}/{consensus['n']} "
        f"({_rate(consensus['auto_label_coverage'])}); human review "
        f"{consensus['human_review_n']}/{consensus['n']} "
        f"({_rate(consensus['human_review_rate'])}); conditional accuracy "
        f"{_rate(performance['conditional_accuracy'])}."
    )
    print(
        "Caution: each judge pair overlaps on a different tutor family; do not "
        "interpret pooled numbers as an unconfounded universal leaderboard."
    )


def parse_wave_spec(value: str) -> tuple[str, str, Path]:
    parts = value.split(":", 2)
    if len(parts) != 3 or not all(part.strip() for part in parts):
        raise argparse.ArgumentTypeError("--wave must have the form JUDGE:WAVE:PATH")
    judge, wave, path = (part.strip() for part in parts)
    if judge not in JUDGE_FAMILIES:
        raise argparse.ArgumentTypeError(
            f"unknown judge {judge!r}; expected one of {', '.join(JUDGE_FAMILIES)}"
        )
    if wave not in EXPECTED_WAVES:
        raise argparse.ArgumentTypeError(
            f"unsupported wave {wave!r}; expected one of {', '.join(EXPECTED_WAVES)}"
        )
    return judge, wave, Path(path)


def _wave_specs_from_root(root: Path) -> list[tuple[str, str, Path]]:
    return [
        (judge, wave, root / judge / wave / f"{wave}.jsonl")
        for judge in JUDGE_FAMILIES
        for wave in EXPECTED_WAVES
    ]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("human_labels", type=Path, help="prepared human_labels.csv")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--results-root",
        type=Path,
        help="root containing JUDGE/WAVE/WAVE.jsonl artifacts",
    )
    source.add_argument(
        "--wave",
        action="append",
        type=parse_wave_spec,
        metavar="JUDGE:WAVE:PATH",
        help="explicit judge wave; repeat 18 times",
    )
    parser.add_argument("--json-out", type=Path, help="write the complete JSON report")
    parser.add_argument("--csv-out", type=Path, help="write one summary row per judge")
    parser.add_argument(
        "--review-out",
        type=Path,
        help="write cases needing human review under the two-judge policy",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
        help="scenario-clustered bootstrap resamples; 0 disables CIs (default: 2000)",
    )
    parser.add_argument("--seed", type=int, default=42, help="bootstrap seed")
    args = parser.parse_args(argv)
    if args.bootstrap_samples < 0:
        parser.error("--bootstrap-samples must be nonnegative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        cases, _ = load_human_labels(args.human_labels)
        humans_by_id = {case.case_id: case for case in cases}
        wave_specs = (
            _wave_specs_from_root(args.results_root)
            if args.results_root is not None
            else args.wave
        )
        specs: dict[str, dict[str, Path]] = defaultdict(dict)
        for judge, wave, path in wave_specs:
            if wave in specs[judge]:
                raise InputValidationError(
                    f"duplicate wave for judge={judge!r}, wave={wave!r}"
                )
            specs[judge][wave] = path
        waves_by_judge: dict[str, dict[str, WaveData]] = {}
        for judge in JUDGE_FAMILIES:
            judge_specs = specs.get(judge, {})
            missing = sorted(set(EXPECTED_WAVES) - set(judge_specs))
            extra = sorted(set(judge_specs) - set(EXPECTED_WAVES))
            if missing or extra:
                raise InputValidationError(
                    f"judge {judge!r} must provide exactly six waves; "
                    f"missing={missing}, extra={extra}"
                )
            waves = {
                wave: load_wave(
                    judge_specs[wave],
                    judge=judge,
                    wave=wave,
                    humans_by_id=humans_by_id,
                )
                for wave in EXPECTED_WAVES
            }
            validate_judge_waves(judge, waves)
            waves_by_judge[judge] = waves
        unknown_judges = sorted(set(specs) - set(JUDGE_FAMILIES))
        if unknown_judges:
            raise InputValidationError(f"unknown judges in wave specs: {unknown_judges}")

        report, review_rows = build_report(
            args.human_labels,
            cases,
            waves_by_judge,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        print_report(report)
        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(
                json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print(f"Wrote JSON report: {args.json_out}")
        if args.csv_out:
            write_csv_summary(args.csv_out, report)
            print(f"Wrote CSV summary: {args.csv_out}")
        if args.review_out:
            write_review_queue(args.review_out, review_rows)
            print(
                f"Wrote {len(review_rows)} human-review case(s): {args.review_out}"
            )
        return 0
    except InputValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
