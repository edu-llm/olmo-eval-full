#!/usr/bin/env python3
"""TracingLLM full protocol on OLMoE-7B with exactly three checkpoints.

See ``PLAN_OLMOE7B.md`` and ``checkpoints_olmoe7b.json``. Phases:

  1. Linear probing — early, chinchilla, final × 5 trustworthiness dimensions
  2. HSIC / MI (lite Section 4) — same three checkpoints
  3. Steering — vectors from early + chinchilla applied to final (all 5 dims)
  4. Trustworthiness + general benchmarks on final (baseline + steered)
  5. Figure reproduction (PNG under outputs/.../figures/)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import steering_common as sc
from checkpoint_manifest import ThreeCheckpointPlan, load_manifest, plan_dict
from eval_general import eval_all_general
from mi_probe import run_mi_sweep
from plot_tracingllm_figures import write_figures
from run_probe_dynamics import Config as ProbeConfig
from run_probe_dynamics import probe_checkpoint
from run_steering_eval import Config as SteerConfig
from run_steering_eval import run as run_steering

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [tracingllm_olmoe7b] %(levelname)s: %(message)s",
)
log = logging.getLogger("tracingllm_olmoe7b")

TRUSTWORTHESS = list(sc.LABELED_DATASETS)


def _middle_layer_list(model, explicit: list[int] | None) -> list[int]:
    if explicit:
        return sc.in_range_layers(model, explicit)
    n = len(sc.decoder_layers(model))
    start, end = n // 3, (2 * n) // 3
    return list(range(start, end))


def _prepare(
    uri: str, tmp: Path, tag: str, device: str, seed: int, region: str, endpoint: str | None
):
    raw = sc.materialize_checkpoint(uri, tmp / f"{tag}_raw", region, endpoint)
    hf = sc.ensure_hf_checkpoint(raw, tmp / f"{tag}_hf")
    return sc.load_model(hf, device, seed)


def run_probe_phase(
    plan: ThreeCheckpointPlan, device: str, seed: int, region: str, endpoint: str | None
):
    cfg = ProbeConfig(
        checkpoints=plan.probing_uris,
        datasets=TRUSTWORTHESS,
        layers=None,
        max_statements=plan.max_statements,
        test_ratio=0.2,
        seed=seed,
        device=device,
        aws_region=region,
        s3_endpoint_url=endpoint,
        run_name=plan.run_name,
    )
    rows: list[dict[str, Any]] = []
    for uri in cfg.checkpoints:
        with tempfile.TemporaryDirectory(prefix="probe-") as tmp:
            rows.extend(probe_checkpoint(cfg, uri, Path(tmp)))
    return rows


def run_steering_phase(
    plan: ThreeCheckpointPlan,
    device: str,
    seed: int,
    region: str,
    endpoint: str | None,
    results_s3: str | None,
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="steer-layers-") as tmp:
        _, model = _prepare(plan.final_uri, Path(tmp), "final", device, seed, region, endpoint)
        layers = plan.steering_layers or _middle_layer_list(model, None)
    for source_name, source_uri in plan.steering_sources.items():
        cfg = SteerConfig(
            checkpoint=source_uri,
            target=plan.final_uri,
            dimensions=TRUSTWORTHESS,
            layers=layers,
            alphas=plan.steering_alphas,
            train_ratio=0.5,
            eval_ratio=0.5,
            max_statements=plan.max_statements,
            eval_limit=plan.eval_limit,
            toxigen_limit=plan.toxigen_limit,
            ppl_limit=plan.ppl_limit,
            seed=seed,
            device=device,
            results_s3=results_s3,
            run_name=f"{plan.run_name}-{source_name}",
            aws_region=region,
            s3_endpoint_url=endpoint,
        )
        out = run_steering(cfg)
        out["source_role"] = source_name
        merged.append(out)
    return merged


def run_general_phase(
    plan: ThreeCheckpointPlan, device: str, seed: int, region: str, endpoint: str | None
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="general-") as tmp:
        tmp_path = Path(tmp)
        _tok, model = _prepare(plan.final_uri, tmp_path, "final", device, seed, region, endpoint)
        baseline = eval_all_general(
            model,
            _tok,
            device,
            mmlu_subjects=plan.mmlu_subjects,
            mmlu_per_subject=plan.mmlu_limit_per_subject,
            seed=seed,
        )
    return {"baseline": baseline}


def run_mi_phase(
    plan: ThreeCheckpointPlan, device: str, seed: int, region: str, endpoint: str | None
) -> list[dict[str, Any]]:
    loaded: dict[str, tuple[Any, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="mi-") as tmp:
        tmp_path = Path(tmp)
        for role, uri in (
            ("early", plan.early_uri),
            ("chinchilla", plan.chinchilla_uri),
            ("final", plan.final_uri),
        ):
            loaded[role] = _prepare(uri, tmp_path, role, device, seed, region, endpoint)
        return run_mi_sweep(
            loaded, datasets=TRUSTWORTHESS, device=device, max_statements=plan.max_statements
        )


def run_job(
    plan: ThreeCheckpointPlan,
    *,
    device: str = "cuda:0",
    seed: int = 1234,
    results_s3: str | None = None,
    aws_region: str = "us-east-1",
    s3_endpoint: str | None = None,
    skip_figures: bool = False,
) -> dict[str, Any]:
    log.info("Plan: %s", json.dumps(plan_dict(plan), indent=2))

    probe_rows = run_probe_phase(plan, device, seed, aws_region, s3_endpoint)
    log.info("Probe phase: %d rows", len(probe_rows))

    mi_rows = run_mi_phase(plan, device, seed, aws_region, s3_endpoint)
    log.info("MI phase: %d rows", len(mi_rows))

    steering_results = run_steering_phase(plan, device, seed, aws_region, s3_endpoint, results_s3)
    log.info("Steering phase: %d source runs", len(steering_results))

    general = run_general_phase(plan, device, seed, aws_region, s3_endpoint)

    payload = {
        "run_name": plan.run_name,
        "plan": plan_dict(plan),
        "probe_accuracy": probe_rows,
        "mi_hsic": mi_rows,
        "steering": steering_results,
        "general": general,
        "success": True,
        "timestamp": datetime.now(UTC).isoformat(),
    }

    out_dir = Path("outputs") / plan.run_name
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "results.json": json.dumps(payload, indent=2),
        "probe_accuracy.json": json.dumps({"results": probe_rows}, indent=2),
        "mi_hsic.json": json.dumps({"results": mi_rows}, indent=2),
        "steering_metrics.json": json.dumps({"results": steering_results}, indent=2),
        "general_metrics.json": json.dumps(general, indent=2),
    }
    for name, content in files.items():
        (out_dir / name).write_text(content)

    if not skip_figures:
        write_figures(payload, fig_dir)

    log.info("Wrote results to %s", out_dir)

    if results_s3:
        sc.upload_files(results_s3, plan.run_name, "all", files, aws_region, s3_endpoint)
        if not skip_figures and fig_dir.exists():
            for png in fig_dir.glob("*.png"):
                bucket, prefix = sc.parse_s3_uri(results_s3)
                client = sc.s3_client(aws_region, s3_endpoint)
                key = f"{prefix.rstrip('/')}/{plan.run_name}/all/figures/{png.name}"
                client.put_object(
                    Bucket=bucket,
                    Key=key,
                    Body=png.read_bytes(),
                    ContentType="image/png",
                )

    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="TracingLLM OLMoE-7B three-checkpoint job.")
    parser.add_argument(
        "--manifest",
        default=str(Path(__file__).resolve().parent / "checkpoints_olmoe7b.json"),
        help="JSON manifest with early/chinchilla/final URIs",
    )
    parser.add_argument("--results-s3", help="Upload prefix (e.g. $EDULLM_OUTPUT_PREFIX)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    plan = load_manifest(
        manifest_path,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        preview = plan_dict(plan)
        preview["phases"] = [
            "probe (3 ckpts × 5 dims)",
            "mi_hsic (3 ckpts)",
            "steering (early+chinchilla → final)",
            "general (ARC/MMLU/MathQA/RACE)",
            "figures",
        ]
        log.info("[dry-run]\n%s", json.dumps(preview, indent=2))
        return 0

    try:
        run_job(
            plan,
            device=args.device,
            seed=args.seed,
            results_s3=args.results_s3,
            skip_figures=args.skip_figures,
        )
    except Exception as exc:  # noqa: BLE001
        log.error("job failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
