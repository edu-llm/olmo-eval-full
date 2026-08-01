"""Grade tutor responses across every open-response benchmark with the frozen judge.

This is a thin multi-benchmark DRIVER around the existing single-benchmark judge
plumbing (``scripts/stage_judge_inputs.py`` + ``scripts/run_judge_grading.py`` +
the teammate runner ``aws_judge_handoff/scripts/run_judge_validation.py``). It
loops the in-scope benchmarks x the tutor-model response shards present on disk
and, per benchmark, stages one gradeable cell per (model, scenario, criterion),
emits the frozen judge's blinded CASE inputs, and ingests the returned verdicts
into a per-benchmark verdict log.

It deliberately does NOT use ``tutor_cat.dataio.load_bank``: several benchmark
rubric banks (BiGGen, Bridge, TutorEval, ...) omit the ``q_mapping`` /
``discrimination`` / ``difficulty`` fields that ``Rubric`` requires. The judge
only needs the criterion text, so rubrics are read with a small schema-tolerant
reader; everything downstream of the item bank (case building, verdict
normalization, resume) is reused from ``run_judge_grading``.

==================================================================
GPU / S3 HAND-OFF (frozen judge = Qwen/Qwen3.5-9B, adapter generic-binary)
==================================================================
The judge is the frozen ``qwen`` spec run by the teammate's canonical runner on a
GPU box; this driver never loads the model. The end-to-end flow is a three-move
prepare -> judge -> ingest, run once per benchmark:

  1. PREPARE (local, this script, default phase):
       uv run scripts/run_all_judge_grading.py --emit-cases-only
     -> writes runs/judge/<Benchmark>/cases.jsonl        (SHIP to the GPU box)
              runs/judge/<Benchmark>/cases_index.jsonl   (PRIVATE; de-blinds case_id)
              runs/judge/<Benchmark>/judge_inputs.jsonl  (staged cells, for ingest)

  2. JUDGE (GPU box, teammate runner, once per benchmark). Example (TutorBench):
       python aws_judge_handoff/scripts/run_judge_validation.py run \
         --cases runs/judge/TutorBench/cases.jsonl --judge qwen \
         --output <ingest-root>/TutorBench/canonical_r1.jsonl \
         --backend vllm --prompt-variant canonical --replicate-id r1 --resume \
         --s3-output-prefix <s3-prefix>/TutorBench/canonical_r1 \
         --require-s3-upload
     The exact per-benchmark command is printed by phase 1, with the s3 prefix
     taken from ``--s3-prefix`` / ``$S3_GRADING_PREFIX`` (no bucket is hardcoded).

  3. INGEST (local, this script):
       uv run scripts/run_all_judge_grading.py --ingest <ingest-root>
     -> writes runs/judge/<Benchmark>/verdicts.jsonl  (one row per graded cell)
              runs/judge/<Benchmark>/manifest.json
              runs/judge/_index.json                  (roll-up)
     ``<ingest-root>`` is searched per benchmark as ``<root>/<Benchmark>/*.jsonl``
     (falling back to ``<root>/<Benchmark>.jsonl``). It may be a local path or an
     ``s3://`` root; an ``s3://`` root is mirrored to a temp dir with ``aws s3
     sync`` (via the AWS CLI) before ingesting.

S3 DIVISION OF LABOR
--------------------
The AWS wrapper (``scripts/aws/run_grading_gpu4.sh``) OWNS the on-node S3
transfers: it pushes each benchmark's cases file up and pulls each shard's
verdict file back down into a flat local inbox before ingesting. This driver is
S3-AWARE but does not duplicate that orchestration:
  * ``--s3-prefix`` / ``$S3_GRADING_PREFIX`` is the single source of truth for
    the S3 layout; the printed hand-off command derives ``--s3-output-prefix``
    from it. Because the wrapper exports ``S3_GRADING_PREFIX``, the driver picks
    up the same prefix automatically — the two stay consistent with no copy.
  * ``--ingest`` optionally accepts an ``s3://`` root so a standalone ingest
    (run without the wrapper) can pull verdicts down itself.

Auto-fail cells (tutor Output errored / empty / missing) are never sent to the
judge; they are scored ``verdict=fail`` (y=0) at ingest with the reason recorded.
Resume is keyed on (model, scenario, criterion_id) against verdicts.jsonl.

Pure stdlib + numpy + pyyaml + tutor_cat. Deterministic. The only optional
network call is ``--ingest s3://...``, which shells out to the AWS CLI (no boto3
dependency); the heavy GPU/upload work stays in the out-of-band phase 2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for _p in (str(ROOT), str(SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_judge_grading as rjg  # noqa: E402
import stage_judge_inputs as sji  # noqa: E402

# In-scope open-response (LLM-judge-graded) benchmarks. WildBench is included
# here explicitly even though benchmarks.yaml marks it enabled: false — the
# driver grades whatever response shards exist and skips benchmarks with none.
IN_SCOPE_BENCHMARKS = ("TutorBench", "TutorEval", "InFoBench", "Bridge", "BiGGen", "WildBench")

DEFAULT_RUBRICS_MAP = {
    "TutorBench": "data/TutorBench/rubrics_qmatrix_final.jsonl",
    "TutorEval": "data/TutorEval/rubrics_qmatrix_final.jsonl",
    "InFoBench": "data/InFoBench/rubrics.jsonl",
    "Bridge": "data/Bridge/rubrics.jsonl",
    "BiGGen": "data/BiGGen/rubrics.jsonl",
    "WildBench": "data/WildBench/rubrics.jsonl",
}

DEFAULT_RESPONSES_ROOT = ROOT / "runs" / "responses"
DEFAULT_OUT_ROOT = ROOT / "runs" / "judge"

JUDGE_INPUTS_NAME = "judge_inputs.jsonl"
JUDGE_INPUTS_MANIFEST_NAME = "judge_inputs_manifest.json"
CASES_NAME = "cases.jsonl"
CASES_INDEX_NAME = "cases_index.jsonl"
VERDICTS_NAME = "verdicts.jsonl"
MANIFEST_NAME = "manifest.json"
MATRIX_CSV_NAME = "response_matrix.csv"
INDEX_NAME = "_index.json"

# Ultimate fallback for the GPU hand-off S3 prefix. This is a clearly-labeled
# placeholder, never a real bucket: pass --s3-prefix or set $S3_GRADING_PREFIX.
S3_PREFIX_PLACEHOLDER = "s3://YOUR-BUCKET/edu-tutor-grading"


# =====================================================================
# Schema-tolerant readers (bypass tutor_cat.dataio.load_bank on purpose)
# =====================================================================
@dataclass
class LiteScenario:
    scenario_id: str
    prompt: str = ""
    use_case: str = ""
    subject: str = ""
    conversation_context: list[dict[str, str]] = field(default_factory=list)
    reference_solution: str = ""
    system_prompt: str = ""


@dataclass
class LiteRubric:
    criterion_id: str
    scenario_id: str
    criterion: str
    scoring_type: str = "binary"
    score_anchors: Any | None = None
    verifier: dict[str, Any] | None = None
    primary_skill: str = ""
    criticality: str = "standard"


@dataclass
class LiteBank:
    """Minimal stand-in for tutor_cat.dataio.ItemBank, exposing only the two
    attribute dicts ``run_judge_grading.build_case_dict`` reads."""

    scenarios: dict[str, LiteScenario]
    rubrics: dict[str, LiteRubric]


def load_scenarios(path: Path) -> dict[str, LiteScenario]:
    out: dict[str, LiteScenario] = {}
    for obj in rjg.iter_jsonl(path):
        sid = obj.get("scenario_id")
        if not sid:
            continue
        out[str(sid)] = LiteScenario(
            scenario_id=str(sid),
            prompt=str(obj.get("prompt", "")),
            use_case=str(obj.get("use_case", "")),
            subject=str(obj.get("subject", "")),
            conversation_context=obj.get("conversation_context") or [],
            reference_solution=str(obj.get("reference_solution", "")),
            system_prompt=str(obj.get("system_prompt") or ""),
        )
    return out


def load_rubrics(path: Path) -> list[LiteRubric]:
    out: list[LiteRubric] = []
    for obj in rjg.iter_jsonl(path):
        cid = obj.get("criterion_id")
        sid = obj.get("scenario_id")
        if not cid or not sid:
            continue
        out.append(
            LiteRubric(
                criterion_id=str(cid),
                scenario_id=str(sid),
                criterion=str(obj.get("criterion", "")),
                scoring_type=str(obj.get("scoring_type", "binary")),
                score_anchors=obj.get("score_anchors"),
                verifier=obj.get("verifier"),
                primary_skill=str(obj.get("primary_skill") or ""),
                criticality=str(obj.get("criticality") or "standard"),
            )
        )
    return out


def is_verifier_benchmark(rubrics: list[LiteRubric]) -> bool:
    """A benchmark is verifier-graded (e.g. IFEval) when its rubrics carry a
    deterministic ``verifier`` spec; such banks are skipped by the judge driver."""
    return any(r.verifier for r in rubrics)


# =====================================================================
# Response-shard discovery
# =====================================================================
def discover_shards(responses_dir: Path) -> dict[str, Path]:
    """Map tutor model id -> shard path for every ``<model>.jsonl`` in a
    benchmark's responses dir. The model id is read from the rows' ``Model``
    field (falling back to the file stem), so case_id derivation matches the
    canonical HF repo id rather than the sanitized filename."""
    shards: dict[str, Path] = {}
    for p in sorted(responses_dir.glob("*.jsonl")):
        if p.name.startswith("_"):
            continue
        model_id = p.stem
        for rec in _iter_shard(p):
            m = rec.get("Model")
            if isinstance(m, str) and m.strip():
                model_id = m.strip()
                break
        shards[model_id] = p
    return shards


def _iter_shard(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


# =====================================================================
# Staging (mirrors stage_judge_inputs ordering; reuses classify_cell)
# =====================================================================
@dataclass
class StagedBenchmark:
    benchmark: str
    responses_dir: Path
    scenarios_path: Path
    rubrics_path: Path
    bank: LiteBank
    models: list[str]
    criterion_ids: list[str]
    staged_rows: list[dict]
    total_cells: int = 0
    gradeable_cells: int = 0
    auto_fail_cells: int = 0
    auto_fail_reasons: dict[str, int] = field(default_factory=dict)
    missing_scenarios: int = 0
    malformed_lines: int = 0


def stage_benchmark(
    benchmark: str, responses_dir: Path, scenarios_path: Path, rubrics_path: Path
) -> StagedBenchmark:
    scenarios = load_scenarios(scenarios_path)
    rubrics = load_rubrics(rubrics_path)

    criteria_by_scenario: dict[str, list[LiteRubric]] = {}
    missing_scenarios = 0
    for r in rubrics:
        if r.scenario_id not in scenarios:
            missing_scenarios += 1
            continue
        criteria_by_scenario.setdefault(r.scenario_id, []).append(r)
    for sid in criteria_by_scenario:
        criteria_by_scenario[sid].sort(key=lambda r: r.criterion_id)

    ordered_sids = sorted(criteria_by_scenario)
    ordered_criterion_ids: list[str] = []
    for sid in ordered_sids:
        ordered_criterion_ids.extend(c.criterion_id for c in criteria_by_scenario[sid])

    bank = LiteBank(
        scenarios=scenarios,
        rubrics={r.criterion_id: r for r in rubrics},
    )

    shards = discover_shards(responses_dir)
    models = sorted(shards)

    staged_rows: list[dict] = []
    total = gradeable = auto_fail = 0
    malformed_lines = 0
    reasons: dict[str, int] = {}
    for model in models:
        index, _dupes, skipped = sji.load_response_index(shards[model])
        malformed_lines += skipped
        for sid in ordered_sids:
            rec = index.get(sid)
            af, reason, response = sji.classify_cell(rec, False)
            for crit in criteria_by_scenario[sid]:
                staged_rows.append(
                    {
                        "model": model,
                        "scenario": sid,
                        "criterion_id": crit.criterion_id,
                        "rubric": crit.criterion,
                        "response": response,
                        "auto_fail": af,
                        "auto_fail_reason": reason,
                    }
                )
                total += 1
                if af:
                    auto_fail += 1
                    reasons[reason] = reasons.get(reason, 0) + 1
                else:
                    gradeable += 1

    return StagedBenchmark(
        benchmark=benchmark,
        responses_dir=responses_dir,
        scenarios_path=scenarios_path,
        rubrics_path=rubrics_path,
        bank=bank,
        models=models,
        criterion_ids=ordered_criterion_ids,
        staged_rows=staged_rows,
        total_cells=total,
        gradeable_cells=gradeable,
        auto_fail_cells=auto_fail,
        auto_fail_reasons=reasons,
        missing_scenarios=missing_scenarios,
        malformed_lines=malformed_lines,
    )


def write_staging(out_dir: Path, sb: StagedBenchmark) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / JUDGE_INPUTS_NAME).open("w", encoding="utf-8") as f:
        for row in sb.staged_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": sb.benchmark,
        "responses_dir": _rel(sb.responses_dir),
        "scenarios_path": _rel(sb.scenarios_path),
        "rubrics_path": _rel(sb.rubrics_path),
        "n_models": len(sb.models),
        "n_criteria": len(sb.criterion_ids),
        "matrix_dims": {"rows_models": len(sb.models), "cols_criteria": len(sb.criterion_ids)},
        "total_cells": sb.total_cells,
        "gradeable_cells": sb.gradeable_cells,
        "auto_fail_cells": sb.auto_fail_cells,
        "auto_fail_reason_counts": sb.auto_fail_reasons,
        "missing_scenarios_for_rubrics": sb.missing_scenarios,
        "malformed_response_lines_skipped": sb.malformed_lines,
        "models": sb.models,
        "criterion_ids": sb.criterion_ids,
    }
    with (out_dir / JUDGE_INPUTS_MANIFEST_NAME).open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


# =====================================================================
# Emit blinded cases (reuses build_case_dict + case_id/response_id + validator)
# =====================================================================
def shard_of_response(response_id: str, num_shards: int) -> int:
    """Deterministically map a ``response_id`` (i.e. a whole ``(model, scenario)``
    block) to one of ``num_shards`` buckets via a stable hash. Node-independent and
    reproducible: every node computes the same bucket for a given response_id, and
    all criteria of a ``(model, scenario)`` share it, so scenario blocks stay whole
    and contiguous within their shard (which prefix caching relies on)."""
    if num_shards <= 1:
        return 0
    digest = hashlib.sha1(response_id.encode("utf-8")).hexdigest()
    return int(digest, 16) % num_shards


def cases_filename(num_shards: int, shard_index: int) -> str:
    return CASES_NAME if num_shards <= 1 else f"cases.shard{shard_index}.jsonl"


def cases_index_filename(num_shards: int, shard_index: int) -> str:
    return CASES_INDEX_NAME if num_shards <= 1 else f"cases_index.shard{shard_index}.jsonl"


def emit_cases(
    out_dir: Path, sb: StagedBenchmark, num_shards: int = 1, shard_index: int = 0
) -> tuple[int, int, int]:
    cases: list[dict] = []
    index_rows: list[dict] = []
    n_auto_fail = n_bank_missing = 0
    for row in sb.staged_rows:
        model = str(row["model"])
        sid = str(row["scenario"])
        cid = str(row["criterion_id"])
        response_id = rjg.response_id_for(model, sid)
        if num_shards > 1 and shard_of_response(response_id, num_shards) != shard_index:
            continue
        case_id = rjg.case_id_for(model, sid, cid)
        index_rows.append(
            {
                "case_id": case_id,
                "response_id": response_id,
                "model": model,
                "scenario_id": sid,
                "criterion_id": cid,
                "auto_fail": int(row.get("auto_fail", 0)),
                "auto_fail_reason": row.get("auto_fail_reason", ""),
            }
        )
        if int(row.get("auto_fail", 0)) == 1:
            n_auto_fail += 1
            continue
        case = rjg.build_case_dict(row, sb.bank)
        if case is None:
            n_bank_missing += 1
            continue
        cases.append(case)

    runner = rjg._import_teammate_runner()
    if runner is not None and hasattr(runner, "validate_judge_cases"):
        try:
            runner.validate_judge_cases(cases)
        except Exception as e:  # noqa: BLE001 - surface schema drift, keep going
            print(f"  ! {sb.benchmark}: cases failed teammate validation: {e}", file=sys.stderr)

    out_dir.mkdir(parents=True, exist_ok=True)
    cases_name = cases_filename(num_shards, shard_index)
    index_name = cases_index_filename(num_shards, shard_index)
    with (out_dir / cases_name).open("w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    with (out_dir / index_name).open("w", encoding="utf-8") as f:
        for r in index_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(cases), n_auto_fail, n_bank_missing


def handoff_command(
    benchmark: str,
    cases_path: Path,
    judge: str,
    ingest_root: Path,
    s3_prefix: str,
    num_shards: int = 1,
    shard_index: int = 0,
) -> str:
    # Each shard grades its own cases file into a shard-tagged verdict file so all
    # shards can share one benchmark ingest dir (ingest globs <root>/<Benchmark>/*.jsonl).
    # The s3 output prefix is derived from the caller-supplied --s3-prefix rather
    # than a hardcoded bucket, and is shard-tagged to match the local output.
    suffix = "" if num_shards <= 1 else f".shard{shard_index}"
    out = ingest_root / benchmark / f"canonical_r1{suffix}.jsonl"
    s3_out = f"{s3_prefix.rstrip('/')}/{benchmark}/canonical_r1{suffix}"
    return (
        "python aws_judge_handoff/scripts/run_judge_validation.py run \\\n"
        f"  --cases {_rel(cases_path)} --judge {judge} \\\n"
        f"  --output {_rel(out)} \\\n"
        "  --backend vllm --prompt-variant canonical --replicate-id r1 --resume \\\n"
        f"  --s3-output-prefix {s3_out} \\\n"
        "  --require-s3-upload"
    )


# =====================================================================
# Ingest verdicts (reuses load_ingest_index + normalize_verdict + resume)
# =====================================================================
def resolve_ingest_paths(ingest_root: Path, benchmark: str) -> list[Path]:
    sub = ingest_root / benchmark
    if sub.is_dir():
        return sorted(sub.glob("*.jsonl"))
    flat = ingest_root / f"{benchmark}.jsonl"
    if flat.is_file():
        return [flat]
    return []


def ingest_benchmark(
    out_dir: Path,
    sb: StagedBenchmark,
    ingest_paths: list[Path],
    fj: rjg.FrozenJudgeConfig,
    no_decision_policy: str,
    resume: bool,
) -> dict:
    verdicts_path = out_dir / VERDICTS_NAME
    existing = rjg.load_done_keys(verdicts_path) if resume else {}
    by_case, ingest_prov, conflicts = rjg.load_ingest_index(ingest_paths)
    rjg._cross_check_provenance(fj, ingest_prov)

    stats = {
        "auto_fail": 0,
        "ingested_pass_fail": 0,
        "ingested_no_decision": 0,
        "skipped_existing": 0,
        "ingest_missing": 0,
        "no_decision_policy": no_decision_policy,
        "cross_file_conflicts": conflicts,
    }

    writer = rjg.VerdictWriter(verdicts_path)
    # Carry prior rows forward so the atomic rewrite does not drop skipped cells.
    if resume:
        writer.seed(existing.values())
    try:
        for row in sb.staged_rows:
            key = (str(row["model"]), str(row["scenario"]), str(row["criterion_id"]))
            if resume and rjg.is_resume_done(existing.get(key)):
                stats["skipped_existing"] += 1
                continue
            if int(row.get("auto_fail", 0)) == 1:
                writer.write(
                    rjg.make_verdict_row(
                        key,
                        y=0,
                        source="auto_fail",
                        verdict="fail",
                        extra={"auto_fail_reason": row.get("auto_fail_reason", "")},
                    )
                )
                stats["auto_fail"] += 1
                continue
            cid = rjg.case_id_for(*key)
            vrow = by_case.get(cid)
            if vrow is None:
                stats["ingest_missing"] += 1
                continue
            y, source, verdict_label, extra = rjg.normalize_verdict(vrow, no_decision_policy)
            extra["rationale"] = vrow.get("rationale", "")
            extra["evidence"] = vrow.get("evidence", "")
            extra["native_score"] = vrow.get("native_score")
            writer.write(
                rjg.make_verdict_row(key, y=y, source=source, verdict=verdict_label, extra=extra)
            )
            if source == "ingest":
                stats["ingested_pass_fail"] += 1
            else:
                stats["ingested_no_decision"] += 1
    finally:
        writer.close()

    # Reuse the writer's committed rows rather than re-reading verdicts.jsonl.
    final = writer.rows()
    arr, csv_cells, n_holes = rjg.assemble_matrix(sb.models, sb.criterion_ids, final)
    rjg.write_matrix_csv(out_dir / MATRIX_CSV_NAME, sb.models, sb.criterion_ids, csv_cells)

    by_source: dict[str, int] = {}
    for r in final.values():
        s = r.get("source", "unknown")
        by_source[s] = by_source.get(s, 0) + 1
    total = len(sb.models) * len(sb.criterion_ids)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": sb.benchmark,
        "phase": "ingest",
        "judge_expected": fj.expected_provenance(),
        "judge_observed": {f: sorted(v) for f, v in (ingest_prov or {}).items() if v},
        "policy_versions": {
            "prompt_version": fj.prompt_version,
            "normalization_version": fj.normalization_version,
            "evidence_policy_version": fj.evidence_policy_version,
        },
        "ingest_files": [_rel(p) for p in ingest_paths],
        "counts": {
            "total_cells": total,
            "verdicts_on_disk": len(final),
            "this_run": stats,
            "by_source": by_source,
        },
        "coverage": {"complete": n_holes == 0, "n_holes": n_holes, "n_filled": total - n_holes},
        "models": sb.models,
        "n_criteria": len(sb.criterion_ids),
    }
    with (out_dir / MANIFEST_NAME).open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return {
        "benchmark": sb.benchmark,
        "phase": "ingest",
        "n_models": len(sb.models),
        "n_criteria": len(sb.criterion_ids),
        "stats": stats,
        "coverage": manifest["coverage"],
    }


# =====================================================================
# helpers
# =====================================================================
def _rel(p: Path) -> str:
    p = Path(p)
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def _is_s3_uri(value: str) -> bool:
    return value.startswith("s3://")


def _sync_s3_to_local(s3_uri: str, dest: Path) -> None:
    """Mirror an ``s3://`` root into a local dir with ``aws s3 sync`` before ingest.

    Uses the AWS CLI via subprocess (no boto3 dependency in the local driver);
    credentials come from the environment / instance role, never from code."""
    dest.mkdir(parents=True, exist_ok=True)
    src = s3_uri.rstrip("/") + "/"
    print(f"  aws s3 sync {src} -> {_rel(dest)}")
    subprocess.run(["aws", "s3", "sync", src, str(dest), "--only-show-errors"], check=True)


def _select_benchmarks(only: str | None, exclude: str | None) -> list[str]:
    names = list(IN_SCOPE_BENCHMARKS)
    if only:
        wanted = [x.strip() for x in only.split(",") if x.strip()]
        unknown = [w for w in wanted if w not in IN_SCOPE_BENCHMARKS]
        if unknown:
            raise ValueError(
                f"unknown benchmark(s) {unknown}; in scope: {list(IN_SCOPE_BENCHMARKS)}"
            )
        names = wanted
    if exclude:
        drop = {x.strip() for x in exclude.split(",") if x.strip()}
        names = [n for n in names if n not in drop]
    return names


def _load_rubrics_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return dict(DEFAULT_RUBRICS_MAP)
    import yaml

    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    merged = dict(DEFAULT_RUBRICS_MAP)
    merged.update({str(k): str(v) for k, v in loaded.items()})
    return merged


# =====================================================================
# main
# =====================================================================
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--only", default=None, help="comma-separated subset of in-scope benchmarks")
    ap.add_argument("--exclude", default=None, help="comma-separated benchmarks to skip")
    ap.add_argument("--responses-root", type=Path, default=DEFAULT_RESPONSES_ROOT,
                    help="root of <Benchmark>/<model>.jsonl response shards")
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT,
                    help="root for per-benchmark judge outputs")
    ap.add_argument("--rubrics-map", type=Path, default=None,
                    help="YAML overriding the per-benchmark rubric file map")
    ap.add_argument("--scenarios-root", type=Path, default=ROOT / "data",
                    help="root holding <Benchmark>/scenarios.jsonl")
    ap.add_argument("--judge", default=None,
                    help="frozen judge name (default: judge_frozen.yaml judge_name)")
    ap.add_argument("--judge-config", type=Path, default=None,
                    help="frozen judge config (default: judge_frozen.yaml)")
    ap.add_argument("--emit-cases-only", action="store_true",
                    help="stage + emit blinded cases; do not ingest (default when --ingest absent)")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="partition emit across N shards by (model, scenario) block "
                         "(default 1: no sharding). Each shard grades independently; "
                         "ingest merges all shards back into the full matrix.")
    ap.add_argument("--shard-index", type=int, default=0,
                    help="0-based index of the shard to emit (0 <= shard-index < num-shards); "
                         "e.g. driven by AWS_BATCH_JOB_ARRAY_INDEX")
    ap.add_argument("--ingest", type=str, default=None,
                    help="root of returned verdicts; searched as <root>/<Benchmark>/*.jsonl. "
                         "May be a local path or an s3:// root (mirrored down with "
                         "aws s3 sync before ingesting).")
    ap.add_argument("--s3-prefix", default=os.environ.get("S3_GRADING_PREFIX"),
                    help="base s3:// prefix for the GPU hand-off command "
                         "(also read from $S3_GRADING_PREFIX). The printed runner "
                         "command uploads to <prefix>/<Benchmark>/canonical_r1[.shard<i>]. "
                         f"Placeholder default: {S3_PREFIX_PLACEHOLDER}")
    ap.add_argument("--no-decision-policy", choices=["missing", "fail"], default="missing")
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args(argv)

    if args.num_shards < 1:
        print(f"ERROR: --num-shards must be >= 1 (got {args.num_shards})", file=sys.stderr)
        return 2
    if not (0 <= args.shard_index < args.num_shards):
        print(f"ERROR: --shard-index must satisfy 0 <= shard-index < num-shards "
              f"(got shard-index={args.shard_index}, num-shards={args.num_shards})",
              file=sys.stderr)
        return 2

    try:
        benchmarks = _select_benchmarks(args.only, args.exclude)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    rubrics_map = _load_rubrics_map(args.rubrics_map)
    fj = rjg.load_frozen_judge_config(args.judge_config)
    judge_name = args.judge or fj.judge_name
    phase = "ingest" if args.ingest else "emit"
    s3_prefix = (args.s3_prefix or S3_PREFIX_PLACEHOLDER).rstrip("/")

    ingest_tmp: tempfile.TemporaryDirectory | None = None
    if args.ingest:
        if _is_s3_uri(args.ingest):
            ingest_tmp = tempfile.TemporaryDirectory(prefix="judge_ingest_s3_")
            ingest_root = Path(ingest_tmp.name)
            _sync_s3_to_local(args.ingest, ingest_root)
        else:
            ingest_root = Path(args.ingest)
    else:
        ingest_root = args.out_root / "_verdicts_inbox"

    print("=" * 72)
    print(f"run_all_judge_grading : phase={phase}  judge={judge_name} / {fj.model}")
    if phase == "emit" and args.num_shards > 1:
        print(f"sharding : shard {args.shard_index} of {args.num_shards} "
              "(partitioned by (model, scenario) block)")
    if phase == "emit":
        print(f"s3 prefix: {s3_prefix}")
        if S3_PREFIX_PLACEHOLDER in s3_prefix:
            print("  WARNING: using the placeholder s3 prefix; pass --s3-prefix or set "
                  "$S3_GRADING_PREFIX to a real bucket/prefix.", file=sys.stderr)
    print(f"in scope: {', '.join(benchmarks)}")
    print("=" * 72)

    roll_up: list[dict] = []
    for benchmark in benchmarks:
        responses_dir = args.responses_root / benchmark
        scenarios_path = args.scenarios_root / benchmark / "scenarios.jsonl"
        rubrics_rel = rubrics_map.get(benchmark)
        rubrics_path = ROOT / rubrics_rel if rubrics_rel else None
        out_dir = args.out_root / benchmark

        print(f"\n--- {benchmark} ---")
        if not responses_dir.is_dir():
            print(f"  SKIP: no responses dir {_rel(responses_dir)} (nothing generated yet)")
            roll_up.append({"benchmark": benchmark, "status": "skipped_no_responses"})
            continue
        if rubrics_path is None or not rubrics_path.is_file():
            print(f"  SKIP: rubric file not found: {rubrics_path}")
            roll_up.append({"benchmark": benchmark, "status": "skipped_no_rubrics"})
            continue
        if not scenarios_path.is_file():
            print(f"  SKIP: scenarios file not found: {_rel(scenarios_path)}")
            roll_up.append({"benchmark": benchmark, "status": "skipped_no_scenarios"})
            continue

        probe = load_rubrics(rubrics_path)
        if is_verifier_benchmark(probe):
            print(f"  SKIP: {benchmark} is verifier-graded (rubrics carry 'verifier'); "
                  "not a judge task")
            roll_up.append({"benchmark": benchmark, "status": "skipped_verifier"})
            continue

        sb = stage_benchmark(benchmark, responses_dir, scenarios_path, rubrics_path)
        if not sb.models:
            print(f"  SKIP: responses dir has no model shards: {_rel(responses_dir)}")
            roll_up.append({"benchmark": benchmark, "status": "skipped_no_shards"})
            continue
        write_staging(out_dir, sb)
        print(f"  models={len(sb.models)}  criteria={len(sb.criterion_ids)}  "
              f"cells={sb.total_cells} (gradeable={sb.gradeable_cells}, "
              f"auto_fail={sb.auto_fail_cells})")
        if sb.missing_scenarios:
            print(f"  note: {sb.missing_scenarios} rubric(s) reference a scenario "
                  "absent from scenarios.jsonl")
        if sb.malformed_lines:
            print(f"  note: skipped {sb.malformed_lines} malformed JSON line(s) across "
                  "response shards (see warnings above)")

        if phase == "emit":
            n_cases, n_af, n_missing = emit_cases(
                out_dir, sb, args.num_shards, args.shard_index
            )
            cases_path = out_dir / cases_filename(args.num_shards, args.shard_index)
            shard_note = (f"  (shard {args.shard_index}/{args.num_shards})"
                          if args.num_shards > 1 else "")
            print(f"  wrote {n_cases} blinded cases -> {_rel(cases_path)}{shard_note}"
                  + (f"  (bank_missing={n_missing})" if n_missing else ""))
            print("  GPU hand-off:")
            cmd = handoff_command(benchmark, cases_path, judge_name, ingest_root,
                                  s3_prefix, args.num_shards, args.shard_index)
            for line in cmd.splitlines():
                print(f"    {line}")
            roll_up.append({
                "benchmark": benchmark, "status": "cases_emitted",
                "n_models": len(sb.models), "n_criteria": len(sb.criterion_ids),
                "gradeable_cells": sb.gradeable_cells, "auto_fail_cells": sb.auto_fail_cells,
                "cases": n_cases, "num_shards": args.num_shards,
                "shard_index": args.shard_index,
                "cases_file": _rel(cases_path),
                "malformed_lines_skipped": sb.malformed_lines,
            })
        else:
            ingest_paths = resolve_ingest_paths(ingest_root, benchmark)
            if not ingest_paths:
                print(f"  SKIP ingest: no verdict files under {_rel(ingest_root / benchmark)}")
                roll_up.append({"benchmark": benchmark, "status": "skipped_no_verdicts"})
                continue
            summary = ingest_benchmark(
                out_dir, sb, ingest_paths, fj, args.no_decision_policy, args.resume
            )
            cov = summary["coverage"]
            print(f"  ingested {summary['stats']['ingested_pass_fail']} pass/fail, "
                  f"{summary['stats']['auto_fail']} auto-fail; "
                  f"holes={cov['n_holes']} ({'complete' if cov['complete'] else 'INCOMPLETE'})")
            print(f"  wrote -> {_rel(out_dir / VERDICTS_NAME)}")
            summary["status"] = "ingested"
            roll_up.append(summary)

    index = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "judge": {"name": judge_name, "model": fj.model, "revision": fj.hf_revision},
        "benchmarks": roll_up,
    }
    args.out_root.mkdir(parents=True, exist_ok=True)
    with (args.out_root / INDEX_NAME).open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    print(f"\nwrote roll-up -> {_rel(args.out_root / INDEX_NAME)}")
    if ingest_tmp is not None:
        ingest_tmp.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
