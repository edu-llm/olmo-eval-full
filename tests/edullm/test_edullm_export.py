"""Round-trip and fail-closed tests for the fitted-bank exporter."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from olmo_eval.edullm.bank import BankValidationError, load_fitted_bank
from olmo_eval.edullm.calibration import (
    CalibrationFitResult,
    CalibrationProvenance,
)
from olmo_eval.edullm.export import (
    ExportError,
    ExportPolicy,
    SourceBankProvenance,
    build_fitted_bank_export,
    write_fitted_bank_export,
)
from olmo_eval.edullm.skill_structure import SkillStructure


def _jsonl(rows: list[dict]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows
    ).encode()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source_records() -> tuple[list[dict], list[dict]]:
    rubrics = [
        {
            "criterion_id": "c1",
            "scenario_id": "s1",
            "criterion": "States the correct conclusion.",
            "scoring_type": "binary",
            "q_mapping": {"content": 1, "scaffolding": 0},
            # These deliberately unsafe placeholders must never be copied.
            "difficulty": 99.0,
            "discrimination": {"content": -7.0, "scaffolding": 4.0},
            "calibration_version": "synthetic-source",
            "irt_params": {"synthetic": True, "source": "placeholder"},
        },
        {
            "criterion_id": "c2",
            "scenario_id": "s1",
            "criterion": "Explains the reasoning.",
            "q_mapping": {"content": 0, "scaffolding": 1},
        },
        {
            "criterion_id": "c3",
            "scenario_id": "s2",
            "criterion": "Uses both skills.",
            "q_mapping": {"content": 1, "scaffolding": 1},
        },
    ]
    scenarios = [
        {
            "scenario_id": "s1",
            "prompt": "Tutor the first learner.",
            "criterion_ids": ["c1", "c2"],
        },
        {
            "scenario_id": "s2",
            "prompt": "Tutor the second learner.",
            "criterion_ids": ["c3"],
        },
    ]
    return rubrics, scenarios


def _structure(*, one_dimensional: bool = False) -> SkillStructure:
    if one_dimensional:
        return SkillStructure.from_groups(
            "collapsed_1d",
            ("content", "scaffolding"),
            (("instruction_following", ("content", "scaffolding")),),
        )
    return SkillStructure.identity(("content", "scaffolding"), name="native_2d")


def _fit(
    item_ids: tuple[str, ...],
    loadings: list[list[float]],
    difficulties: list[float],
    structure: SkillStructure,
    *,
    correlation: list[list[float]] | None = None,
    converged: bool = True,
) -> CalibrationFitResult:
    n_dims = structure.n_dims
    corr = correlation if correlation is not None else np.eye(n_dims).tolist()
    provenance = CalibrationProvenance(
        source_branch="origin/frq/infobench",
        source_commit="b4ea2e8",
        source_blob="calibration-blob",
        parameterization="sigmoid(A @ theta - b)",
        model_specification=MappingProxyType({"family": "free-2pl"}),
        skill_structure=MappingProxyType(structure.as_dict()),
        quadrature=MappingProxyType({"method": "gauss_hermite", "nodes": 7}),
        convergence_policy=MappingProxyType({"mode": "returned_iterate"}),
        missing_cell_policy="exclude",
        n_persons_supplied=52,
        n_persons_fitted=52,
        n_all_missing_persons_excluded=0,
        n_items=len(item_ids),
        n_observed_cells=100,
        n_missing_cells=4,
        specification_sha256="a" * 64,
    )
    return CalibrationFitResult(
        loadings=np.asarray(loadings, dtype=float),
        difficulties=np.asarray(difficulties, dtype=float),
        latent_correlation=np.asarray(corr, dtype=float),
        marginal_log_likelihood=-12.5,
        penalized_objective=-12.7,
        n_parameters=6,
        n_free_loadings=4,
        n_fixed_loadings=0,
        iterations=8,
        converged=converged,
        termination_reason="converged" if converged else "max_iterations",
        returned_iterate_status=(
            "converged_returned_iterate" if converged else "last_valid_returned_iterate"
        ),
        convergence_diagnostics=MappingProxyType({"objective_delta": 1e-8}),
        provenance=provenance,
    )


def _policy(
    *,
    unfitted: str = "exclude",
    nonpositive: str = "exclude",
    extreme: str = "exclude",
    invalid: str = "error",
    require_converged: bool = True,
) -> ExportPolicy:
    return ExportPolicy(
        unfitted=unfitted,  # type: ignore[arg-type]
        nonpositive=nonpositive,  # type: ignore[arg-type]
        extreme=extreme,  # type: ignore[arg-type]
        invalid=invalid,  # type: ignore[arg-type]
        extreme_a_threshold=6.0,
        off_q_tolerance=1e-10,
        require_converged=require_converged,
    )


def _provenance(rubrics: list[dict], scenarios: list[dict], *paths: Path) -> SourceBankProvenance:
    return SourceBankProvenance(
        rubrics_identifier="memory://source/rubrics.jsonl",
        rubrics_sha256=_sha(_jsonl(rubrics)),
        scenarios_identifier="memory://source/scenarios.jsonl",
        scenarios_sha256=_sha(_jsonl(scenarios)),
        additional_sha256={"response_matrix": "b" * 64},
        metadata={"benchmark": "fixture"},
        protected_input_paths=tuple(paths),
    )


def _build(
    rubrics: list[dict],
    scenarios: list[dict],
    fit: CalibrationFitResult,
    item_ids: tuple[str, ...],
    structure: SkillStructure,
    *,
    policy: ExportPolicy | None = None,
    provenance: SourceBankProvenance | None = None,
):
    return build_fitted_bank_export(
        rubrics,
        scenarios,
        fit,
        item_ids,
        structure,
        source_provenance=provenance or _provenance(rubrics, scenarios),
        calibration_version="fixture-fit-v1",
        calibration_method="confirmatory-m2pl-mml-em",
        policy=policy or _policy(),
        source_q_field="q_mapping",
    )


def test_export_round_trips_through_strict_loader_without_mutating_sources(
    tmp_path: Path,
) -> None:
    rubrics, scenarios = _source_records()
    original_rubrics = copy.deepcopy(rubrics)
    original_scenarios = copy.deepcopy(scenarios)
    structure = _structure()
    item_ids = ("c1", "c3")
    fit = _fit(
        item_ids,
        [[1.2, 0.0], [0.8, 0.9]],
        [-0.2, 0.4],
        structure,
        correlation=[[1.0, 0.25], [0.25, 1.0]],
    )

    exported = _build(rubrics, scenarios, fit, item_ids, structure)

    assert rubrics == original_rubrics
    assert scenarios == original_scenarios
    assert [record["criterion_id"] for record in exported.rubric_records] == ["c1", "c3"]
    assert exported.scenario_records[0]["criterion_ids"] == ("c1",)
    assert exported.manifest["exclusion_reason_counts"] == {"unfitted": 1}
    c1 = exported.rubric_records[0]
    assert c1["difficulty"] == -0.2
    assert c1["discrimination"] == {"content": 1.2, "scaffolding": 0.0}
    assert c1["irt_params"]["synthetic"] is False
    assert c1["irt_params"]["provenance"]["source_bank_sha256"]["rubrics"] == _sha(_jsonl(rubrics))

    written = write_fitted_bank_export(
        exported,
        tmp_path / "fitted-rubrics.jsonl",
        tmp_path / "fitted-scenarios.jsonl",
        tmp_path / "fitted-manifest.json",
    )
    bank = load_fitted_bank(
        written.rubrics_path,
        written.scenarios_path,
        manifest_path=written.manifest_path,
    )

    assert bank.criterion_ids == ("c1", "c3")
    assert np.allclose(bank.A, [[1.2, 0.0], [0.8, 0.9]])
    assert np.allclose(bank.b, [-0.2, 0.4])
    assert np.allclose(bank.latent_correlation, [[1.0, 0.25], [0.25, 1.0]])
    manifest = json.loads(written.manifest_path.read_text())
    assert manifest["inputs"]["source_bank"]["rubrics"]["sha256"] == _sha(_jsonl(rubrics))
    assert manifest["invariants"]["latent_correlation_positive_definite"] is True


def test_export_manifest_detects_output_tampering(tmp_path: Path) -> None:
    rubrics, scenarios = _source_records()
    structure = _structure()
    item_ids = ("c1", "c3")
    exported = _build(
        rubrics,
        scenarios,
        _fit(item_ids, [[1.0, 0.0], [0.8, 0.9]], [0.0, 0.2], structure),
        item_ids,
        structure,
    )
    written = write_fitted_bank_export(
        exported,
        tmp_path / "rubrics.jsonl",
        tmp_path / "scenarios.jsonl",
        tmp_path / "manifest.json",
    )
    written.rubrics_path.write_text(
        written.rubrics_path.read_text() + "\n",
        encoding="utf-8",
    )

    with pytest.raises(BankValidationError, match="SHA-256 mismatch for rubrics"):
        load_fitted_bank(
            written.rubrics_path,
            written.scenarios_path,
            manifest_path=written.manifest_path,
        )


@pytest.mark.parametrize(
    ("unsafe_loading", "policy_name", "reason"),
    [
        (-0.1, "nonpositive", "nonpositive_a"),
        (7.0, "extreme", "extreme_a"),
    ],
)
def test_unsafe_active_loading_obeys_explicit_exclude_or_error_policy(
    unsafe_loading: float,
    policy_name: str,
    reason: str,
) -> None:
    rubrics, scenarios = _source_records()
    structure = _structure()
    item_ids = ("c1", "c3")
    fit = _fit(item_ids, [[unsafe_loading, 0.0], [0.8, 0.9]], [0.0, 0.2], structure)

    excluded = _build(rubrics, scenarios, fit, item_ids, structure, policy=_policy())
    assert [record["criterion_id"] for record in excluded.rubric_records] == ["c3"]
    assert excluded.manifest["exclusion_reason_counts"][reason] == 1

    actions = {"nonpositive": "exclude", "extreme": "exclude"}
    actions[policy_name] = "error"
    with pytest.raises(ExportError, match=reason):
        _build(
            rubrics,
            scenarios,
            fit,
            item_ids,
            structure,
            policy=_policy(**actions),  # type: ignore[arg-type]
        )


def test_nonzero_off_q_loading_is_never_exported() -> None:
    rubrics, scenarios = _source_records()
    structure = _structure()
    item_ids = ("c1", "c3")
    fit = _fit(item_ids, [[1.0, 1e-5], [0.8, 0.9]], [0.0, 0.2], structure)

    with pytest.raises(ExportError, match="invalid_off_q_nonzero"):
        _build(rubrics, scenarios, fit, item_ids, structure, policy=_policy(invalid="error"))

    excluded = _build(
        rubrics,
        scenarios,
        fit,
        item_ids,
        structure,
        policy=_policy(invalid="exclude"),
    )
    assert [record["criterion_id"] for record in excluded.rubric_records] == ["c3"]


def test_export_rejects_singular_correlation_but_accepts_1d_identity() -> None:
    rubrics, scenarios = _source_records()
    two_dimensional = _structure()
    item_ids = ("c1", "c3")
    singular = _fit(
        item_ids,
        [[1.0, 0.0], [0.8, 0.9]],
        [0.0, 0.2],
        two_dimensional,
        correlation=[[1.0, 1.0], [1.0, 1.0]],
    )
    with pytest.raises(ExportError, match="positive definite"):
        _build(rubrics, scenarios, singular, item_ids, two_dimensional)

    one_dimensional = _structure(one_dimensional=True)
    one_dimensional_fit = _fit(
        item_ids,
        [[1.0], [0.8]],
        [0.0, 0.2],
        one_dimensional,
        correlation=[[1.0]],
    )
    exported = _build(
        rubrics,
        scenarios,
        one_dimensional_fit,
        item_ids,
        one_dimensional,
    )
    assert exported.manifest["latent_correlation"] == ((1.0,),)


def test_writer_refuses_source_collision_and_unrequested_overwrite(tmp_path: Path) -> None:
    source_path = tmp_path / "source-rubrics.jsonl"
    source_path.write_text("source must remain", encoding="utf-8")
    rubrics, scenarios = _source_records()
    structure = _structure()
    item_ids = ("c1", "c3")
    exported = _build(
        rubrics,
        scenarios,
        _fit(item_ids, [[1.0, 0.0], [0.8, 0.9]], [0.0, 0.2], structure),
        item_ids,
        structure,
        provenance=_provenance(rubrics, scenarios, source_path),
    )

    with pytest.raises(ExportError, match="source input"):
        write_fitted_bank_export(
            exported,
            source_path,
            tmp_path / "scenarios.jsonl",
            tmp_path / "manifest.json",
            overwrite=True,
        )
    assert source_path.read_text() == "source must remain"

    existing = tmp_path / "existing.jsonl"
    existing.write_text("keep", encoding="utf-8")
    with pytest.raises(ExportError, match="output already exists"):
        write_fitted_bank_export(
            exported,
            existing,
            tmp_path / "new-scenarios.jsonl",
            tmp_path / "new-manifest.json",
        )
    assert existing.read_text() == "keep"
