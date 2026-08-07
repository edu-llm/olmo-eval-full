# OLMo runner modes for EduLLM

`olmo-eval run-modes` is the supported top-level runner for the integrated
pipeline. A single strict YAML or JSON file selects:

- `standard_olmo` for OLMo tasks and suites;
- `edullm_adaptive` for tutor generation, frozen-Qwen criterion judging,
  joint-EAP stopping, and final EAP/MWLE reporting; or
- both, in the order listed in the file.

OLMo owns the run ID, model-provider startup, cleanup, status, and artifact
root. EduLLM owns its existing judging and adaptive-testing logic. Results from
the two modes remain separate.

## Runtime setup

Install the ordinary runner and EduLLM numerical dependency from the lockfile:

```bash
uv sync --frozen --extra edullm
```

The frozen judge is `Qwen/Qwen3.5-9B` at revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`. It requires vLLM **0.26.0** for
explicit P/F token probabilities, while this repository's ordinary vLLM extra
remains independently pinned. Point `EDULLM_QWEN_VLLM_PYTHON` at a separate environment;
do not upgrade the project-wide vLLM pin:

```bash
uv venv .venv-vllm026 --python 3.12
uv pip install --python .venv-vllm026/bin/python 'vllm==0.26.0'
export EDULLM_QWEN_VLLM_PYTHON="$PWD/.venv-vllm026/bin/python"
```

For a managed local Qwen server, the runner verifies that interpreter and its
required explicit-token-logprob capability before loading a model. For an
already-running server, it verifies the server's `/version` response. The
runner removes `VLLM_BATCH_INVARIANT` only from the Qwen child process because
Qwen's GDN attention does not support that vLLM mode.

GPU jobs must be launched through the repository's `edullm` workflow described
in `AGENTS.md`; the integration contains no AWS or S3 calls.

## Configuration

[`run_modes.example.yaml`](run_modes.example.yaml) shows both modes. Its bank
paths, candidate checkpoint, task list, skill order, and CAT operating point
are placeholders. Replace them with an approved, versioned fitted bank and
benchmark-specific policy. The software intentionally does not install a
universal difficulty, discrimination, skill structure, or stopping threshold.
The candidate provider revision is required and must exactly match
`tutor.model_provenance.revision`; this prevents the manifest from claiming a
different checkpoint than the one the runner loads.

Run read-only validation explicitly when desired:

```bash
uv run olmo-eval run-modes --config examples/edullm/run_modes.example.yaml --check
```

This check validates the fitted bank and CAT configuration as well as provider,
GPU, and frozen-Qwen runtime compatibility, but it does not start inference or
write run artifacts.

Run the configured evaluation (this is the default; no smoke test is inserted):

```bash
uv run olmo-eval run-modes --config /path/to/final-run.yaml
```

The output directory must be new or empty. A completed run has this shape:

```text
<output_dir>/
  manifest.json
  report_index.json
  logs/
  modes/
    standard_olmo/
      metrics.json
      ...
    edullm_adaptive/
      manifest.json
      tutor_responses.jsonl
      judge_rows.jsonl
      cat_result.json
      cat_trace.jsonl
```

`no_decision` judge outputs remain missing observations. They are never
rewritten as criterion failures. The EduLLM result reports EAP and MWLE
separately; ordinary OLMo benchmark metrics are not combined with theta.
