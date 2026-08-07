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

### Using tutor responses generated elsewhere

If tutor responses already exist, use
[`run_precomputed.example.yaml`](run_precomputed.example.yaml) instead of loading
the tutor model again. This option is for an **adaptive-only** run: the primary
provider must be `kind: mock`, `standard_olmo` must not be selected, and
`tutor.generation` must be `null`. The mock provider is only an OLMo lifecycle
placeholder; it does not generate responses or load the tutor model. The frozen
Qwen judge still runs through vLLM on the uploaded responses, after which CAT
performs selection and reports the usual EAP and MWLE estimates.

Use one UTF-8 JSONL file for one tutor model and one run. Each line has this
strict form:

```json
{"scenario_id":"ifb_0001","response":"The tutor's response"}
{"scenario_id":"ifb_0002","response":"","metadata":{"finish_reason":"length"}}
```

`metadata` is optional and, when present, must be a JSON object. It is preserved
for provenance but does not affect judging. Unknown fields, malformed JSON,
duplicate scenario IDs, and blank physical lines are rejected. The file must
contain exactly one row for every scenario in the fitted bank: missing and extra
scenario IDs are both errors because CAT may select any bank scenario.

Declare the actual tutor model and immutable revision under `tutor`, then hash
the exact response-file bytes and put that digest in
`tutor.response_source.sha256`:

```bash
sha256sum /path/to/responses.jsonl
# macOS also provides: shasum -a 256 /path/to/responses.jsonl
```

The runner verifies the digest and complete scenario roster before judging.
Whitespace-only response text is allowed, but it is treated as missing data:
Qwen is not called for that scenario's criteria and each corresponding result is
`no_decision`, never an automatic failure. The adaptive manifest records the
source path, declared and observed hashes, row count, and blank-response count.

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
