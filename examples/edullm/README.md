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
For live provider-generated tutor responses, the candidate provider revision is
required and must exactly match `tutor.model_provenance.revision`; this prevents
the manifest from claiming a different checkpoint than the one the runner
loads. Precomputed workflows declare their provenance as described below.

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

### Uploading responses for multiple tutor models

Use [`run_precomputed_batch.example.yaml`](run_precomputed_batch.example.yaml)
when one file contains responses from several tutor models. This is also an
**adaptive-only** run: the primary provider must be `kind: mock`,
`edullm_adaptive` must be the only selected mode, and `tutor.generation` must be
`null`. The uploaded rows identify the real tutor models; the mock provider is
only a lifecycle placeholder and is never asked to generate a response.

The batch must be a UTF-8 JSONL file with one model-scenario response per line:

```json
{"model_id":"org/model-a","model_family":"family-a","model_revision":"revision-a","scenario_id":"ifb_0001","response":"Model A response"}
{"model_id":"org/model-a","model_family":"family-a","model_revision":"revision-a","scenario_id":"ifb_0002","response":"","metadata":{"finish_reason":"length"}}
{"model_id":"org/model-b","model_family":"family-b","model_revision":"revision-b","scenario_id":"ifb_0001","response":"Model B response"}
{"model_id":"org/model-b","model_family":"family-b","model_revision":"revision-b","scenario_id":"ifb_0002","response":"Model B response"}
```

Every row must contain `model_id`, `model_family`, `model_revision`,
`scenario_id`, and string-valued `response`. `metadata` is an optional JSON
object. All four identity/scenario strings must be non-empty and have no leading
or trailing whitespace. The family and revision must be consistent across all
rows carrying the same `model_id`, and each model must have exactly one row for
every scenario in the fitted bank. The batch `tutor` config contains only
`generation` and `response_source`; the response source provenance contains
exactly non-empty `source` and `revision` strings. The runner rejects unknown
fields, malformed rows, duplicate model-scenario pairs, inconsistent identities,
missing or extra scenarios, and a file whose bytes do not match the declared
SHA-256. This validation completes before Qwen is started.

Qwen is loaded once and reused for every model in the batch. Each model then
gets its own independent CAT session and its own EAP and MWLE estimates. CAT
selects scenarios adaptively, so it judges only the selected responses for each
model; it does **not** exhaustively judge every uploaded response. Blank or
whitespace-only selected responses skip Qwen and remain `no_decision`, exactly
as in the single-model precomputed workflow.

A completed batch run has this adaptive-mode layout:

```text
<output_dir>/modes/edullm_adaptive/
  manifest.json
  batch_summary.json
  model_results.jsonl
  mode_result.json
  models/
    candidate-0000-<stable-hash>/
      manifest.json
      tutor_responses.jsonl
      judge_rows.jsonl
      cat_result.json
      cat_trace.jsonl
    candidate-0001-<stable-hash>/
      ...
```

`model_results.jsonl` maps each original model identity to its safe output
directory, terminal status, CAT metrics, warnings, and error if any.
`batch_summary.json` provides aggregate model, scenario, criterion, and
`no_decision` counts. A runtime failure for one tutor is recorded in that
tutor's directory and does not prevent later tutors from running, although the
overall adaptive mode is marked failed if any tutor failed.

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

The output directory must be new or empty. A completed live-generated or
single-model precomputed run has this shape (batch runs use the layout shown
above):

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
