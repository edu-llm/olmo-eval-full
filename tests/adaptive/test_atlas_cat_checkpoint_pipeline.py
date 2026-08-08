"""ATLAS CAT × olmo-eval checkpoint pipeline (AWS → finetune → Beaker → results).

End-to-end wiring test for the train→diagnose loop:

1. Load a base checkpoint from an S3 URI (AWS download, mocked).
2. Run one finetuning step and write an updated checkpoint.
3. Assemble a Beaker job that runs the online ``atlas_arc`` CAT diagnostic
   against the finetuned checkpoint URI (``inject_aws_credentials`` for S3).
4. Execute the CAT (fake provider — no GPU) and persist ``atlas_arc_results.json``.

Live AWS / Beaker are not contacted.

Training-team hook (one ``g6.xlarge`` / L4 worker → fixed S3 prefix)::

    bash AdaptiveTesting/scripts/atlas_cat_diagnose/launch_g6.sh \\
        --checkpoint s3://BUCKET/checkpoints/EXP/STEP \\
        --run-id EXP-stepSTEP

Known results root::

    s3://edullm-adaptive-inference-056956104102/atlas_cat/<run_id>/
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from olmo_eval.cli.beaker.job_assembler import assemble_external_eval_job
from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType
from olmo_eval.evals.external.benchmarks.atlas_arc.eval import AtlasArcExternalEval
from olmo_eval.launch.beaker.aws import is_s3_path

torch = pytest.importorskip("torch")

KNOWN_ATLAS_CAT_S3_ROOT = "s3://edullm-adaptive-inference-056956104102/atlas_cat"


# ---------------------------------------------------------------------------
# Pipeline steps (thin, testable units mirroring the intended live flow)
# ---------------------------------------------------------------------------


def load_checkpoint_from_s3(s3_uri: str, dest: Path, *, downloader=None) -> Path:
    """Pull a checkpoint from S3 to ``dest``.

    ``downloader`` is injectable so CI can fake boto3 / aws s3 sync. The live
    path would call something like ``aws s3 sync s3://... dest``.
    """
    if not is_s3_path(s3_uri):
        raise ValueError(f"expected s3:// URI, got {s3_uri!r}")
    dest.mkdir(parents=True, exist_ok=True)
    if downloader is None:
        raise RuntimeError(
            "no downloader provided; pass a callable or use the AWS CLI "
            f"(`aws s3 sync {s3_uri} {dest}`)"
        )
    downloader(s3_uri, dest)
    return dest


def finetune_one_step(checkpoint_dir: Path, *, steps: int = 1, lr: float = 1e-3) -> Path:
    """Run a minimal AdamW update and write an updated checkpoint under ``checkpoint_dir``.

    This stands in for an out-of-repo OLMo-core / HF Trainer finetune. The
    important contract for the pipeline is: a new checkpoint directory exists
    and can be addressed as an S3 URI for the Beaker CAT job.
    """
    import torch.nn as nn

    model = nn.Linear(8, 8)
    init_path = checkpoint_dir / "pytorch_model.bin"
    if init_path.exists():
        try:
            state = torch.load(init_path, map_location="cpu", weights_only=True)
        except TypeError:  # torch<2.0 has no weights_only
            state = torch.load(init_path, map_location="cpu")
        model.load_state_dict(state)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for _ in range(steps):
        x = torch.randn(4, 8)
        loss = (model(x) - x).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    out = checkpoint_dir / "finetuned"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "pytorch_model.bin")
    (out / "finetune_meta.json").write_text(
        json.dumps({"steps": steps, "lr": lr, "loss": float(loss.detach())}),
        encoding="utf-8",
    )
    return out


def assemble_atlas_cat_beaker_job(
    *,
    checkpoint_s3_uri: str,
    experiment_name: str = "atlas-cat-ft-diagnostic",
    cluster: str = "ai2/jupiter",
    workspace: str = "ai2/oe-data",
    budget: str = "ai2/oe-base",
    beaker_image: str = "oe-eval-beaker",
    s3_results_bucket: str | None = "edullm-eval-results",
    s3_results_prefix: str | None = "atlas_cat",
    eval_args: dict[str, str] | None = None,
) -> Any:
    """Build the Beaker job that runs online ``atlas_arc`` on the checkpoint."""
    if not is_s3_path(checkpoint_s3_uri):
        raise ValueError(f"CAT Beaker job expects an s3:// checkpoint, got {checkpoint_s3_uri!r}")

    return assemble_external_eval_job(
        name=experiment_name,
        model=checkpoint_s3_uri,
        external_evals=["atlas_arc"],
        cluster=cluster,
        num_gpus=1,
        workspace=workspace,
        beaker_image=beaker_image,
        budget=budget,
        inject_aws_credentials=True,  # required for s3:// model + optional result upload
        store=bool(s3_results_bucket and s3_results_prefix),
        s3_bucket=s3_results_bucket,
        s3_prefix=s3_results_prefix,
        eval_args=eval_args or {"se_stop": "0.3", "min_items": "8", "max_items": "40"},
    )


def _write_bank(tmp_path: Path, n: int) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    with open(tmp_path / "irt_item_parameters_combined.csv", "w", newline="") as fh:
        fh.write("X,a1,d,g,u\n")
        for i in range(1, n + 1):
            fh.write(f"X{i},1.3,{0.6 - i * 0.05:.3f},0.2,1\n")
    with open(tmp_path / "atlas_idx_to_question_id.csv", "w", newline="") as fh:
        fh.write("atlas_idx,question_id\n")
        for i in range(1, n + 1):
            fh.write(f"{i},q{i - 1}\n")
    return tmp_path


class _FakeTask:
    def format_request(self, instance: Instance) -> LMRequest:
        return LMRequest(
            request_type=RequestType.LOGLIKELIHOOD,
            prompt="",
            continuations=(" a", " b"),
        )


class _FakeProvider:
    """Deterministic logprobs so CAT is GPU-free and stable in CI."""

    async def alogprobs(self, requests: list[LMRequest], sampling_params: Any = None):
        return [
            [
                LMOutput(text="a", logprobs=[{"token": "a", "logprob": -1.0}]),
                LMOutput(text="b", logprobs=[{"token": "b", "logprob": -2.0}]),
            ]
        ]


def _items(n: int) -> dict[str, Instance]:
    return {
        f"q{i}": Instance(
            question="q",
            choices=("a", "b"),
            gold_answer="A",
            metadata={"id": f"q{i}", "gold_idx": i % 2},
        )
        for i in range(n)
    }


def run_cat_diagnostic_and_save(
    *,
    bank_dir: Path,
    output_dir: Path,
    n_items: int = 30,
    se_stop: float = 0.0,
    min_items: int = 8,
    max_items: int = 15,
) -> Path:
    """Run online ``atlas_arc`` with a fake provider and save results JSON."""
    ev = AtlasArcExternalEval()
    ev._load_items = lambda args: (_FakeTask(), _items(n_items))  # type: ignore[method-assign]
    result = asyncio.run(
        ev.execute(
            _FakeProvider(),
            {
                "bank_path": str(bank_dir),
                "se_stop": se_stop,
                "min_items": min_items,
                "max_items": max_items,
            },
            output_dir=str(output_dir),
        )
    )
    assert result.success, result.error
    results_path = output_dir / "atlas_arc_results.json"
    assert results_path.is_file()
    return results_path


# ---------------------------------------------------------------------------
# End-to-end test
# ---------------------------------------------------------------------------


def test_atlas_cat_checkpoint_pipeline(tmp_path: Path) -> None:
    """AWS load → finetune step → Beaker CAT job assembly → saved diagnostic."""
    base_s3 = "s3://edullm-checkpoints/owner/base-model/step0"
    ft_s3 = "s3://edullm-checkpoints/owner/base-model/finetuned-step1"
    work = tmp_path / "work"
    results = tmp_path / "cat_results"
    bank_dir = _write_bank(tmp_path / "bank", n=30)

    # --- 1. Load base weights from AWS (mocked S3 sync) --------------------
    def fake_s3_sync(uri: str, dest: Path) -> None:
        assert uri == base_s3
        dest.mkdir(parents=True, exist_ok=True)
        # Seed a tiny state dict as the "downloaded" base checkpoint.
        torch.save(torch.nn.Linear(8, 8).state_dict(), dest / "pytorch_model.bin")
        (dest / "source_uri.txt").write_text(uri, encoding="utf-8")

    local_base = load_checkpoint_from_s3(base_s3, work / "base", downloader=fake_s3_sync)
    assert (local_base / "pytorch_model.bin").is_file()
    assert (local_base / "source_uri.txt").read_text(encoding="utf-8") == base_s3

    # --- 2. Finetuning step → new checkpoint -------------------------------
    ft_dir = finetune_one_step(local_base, steps=1)
    assert (ft_dir / "pytorch_model.bin").is_file()
    meta = json.loads((ft_dir / "finetune_meta.json").read_text(encoding="utf-8"))
    assert meta["steps"] == 1

    # --- 3. Beaker job for CAT on the S3 checkpoint URI --------------------
    job = assemble_atlas_cat_beaker_job(checkpoint_s3_uri=ft_s3)
    assert job.inject_aws_credentials is True
    assert job.command[:2] == ["olmo-eval", "run-external"]
    assert "-m" in job.command and ft_s3 in job.command
    assert "-e" in job.command and "atlas_arc" in job.command
    # Results storage flags for the Beaker worker
    assert "--store" in job.command
    assert "--s3-bucket" in job.command

    # Dry-run launch: record the job that would be submitted (no Beaker API).
    dry_run_record = {
        "dry_run": True,
        "experiment_name": job.name,
        "command": list(job.command),
        "inject_aws_credentials": job.inject_aws_credentials,
        "cluster": job.cluster,
    }
    assert dry_run_record["dry_run"] is True
    assert dry_run_record["inject_aws_credentials"] is True
    assert "atlas_arc" in dry_run_record["command"]

    # --- 4. CAT diagnostic + persist results (stands in for Beaker worker) -
    results_path = run_cat_diagnostic_and_save(bank_dir=bank_dir, output_dir=results)
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    assert payload["name"] == "atlas_arc"
    assert payload["success"] is True
    for key in ("theta", "se", "pirt_accuracy", "n_items", "bank_size"):
        assert key in payload["metrics"]
    assert payload["metrics"]["n_items"] == 15.0
    assert len(payload["metadata"]["selected_question_ids"]) == 15

    # Provenance sidecar: tie CAT outputs to the finetuned checkpoint URI.
    # Mirrors AdaptiveTesting/scripts/atlas_cat_diagnose/ known layout.
    known_dest = f"{KNOWN_ATLAS_CAT_S3_ROOT}/demo-ft-step1"
    provenance = {
        "base_s3": base_s3,
        "finetuned_s3": ft_s3,
        "local_finetuned": str(ft_dir),
        "beaker_command": job.command,
        "cat_results": str(results_path),
        "known_s3_dest": known_dest,
        "launch_hook": "AdaptiveTesting/scripts/atlas_cat_diagnose/launch_g6.sh",
    }
    prov_path = results / "pipeline_provenance.json"
    prov_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    assert prov_path.is_file()
    assert provenance["known_s3_dest"].startswith(KNOWN_ATLAS_CAT_S3_ROOT)


def test_assemble_rejects_non_s3_checkpoint() -> None:
    with pytest.raises(ValueError, match="s3://"):
        assemble_atlas_cat_beaker_job(checkpoint_s3_uri="allenai/OLMo-2-1124-7B")
