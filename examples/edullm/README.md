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

#### Preparing a strict batch from response shards

`olmo-eval edullm prepare-response-batch` validates and combines existing
response artifacts without loading a tutor or judge model. It takes an explicit
JSONL source manifest; paths in the manifest are resolved relative to that
manifest. Start with
[`response_batch_sources.example.jsonl`](response_batch_sources.example.jsonl).

Each manifest row uses one named format:

- `single-jsonl-v1` reads strict `{scenario_id,response,metadata?}` rows and
  requires `model_id`, `model_family`, and `model_revision` on the manifest row.
- `tutorbench-output-jsonl-v1` reads the legacy TutorBench Title-Case model-output
  schema and likewise requires an explicit model identity on the manifest row.
  Its original `Rendered Prompt` and digest are retained as generation-request
  provenance; flagged issue/error rows must be blank and internally consistent.
- `batch-jsonl-v1` reads rows already using the strict multi-model schema; model
  identity comes from those rows.

An optional lowercase `sha256` on any manifest row pins the exact source bytes.
Formats and identities are never inferred from filenames.

```bash
# Validate and calculate the exact prospective output hash without writing files.
olmo-eval edullm prepare-response-batch \
  --source-manifest /path/to/response-sources.jsonl \
  --fitted-scenarios /path/to/fitted_bank/scenarios.jsonl \
  --output /path/to/all_tutor_responses.jsonl \
  --check

# Run the same command without --check to write the JSONL and its report.
olmo-eval edullm prepare-response-batch \
  --source-manifest /path/to/response-sources.jsonl \
  --fitted-scenarios /path/to/fitted_bank/scenarios.jsonl \
  --output /path/to/all_tutor_responses.jsonl
```

The default report is
`/path/to/all_tutor_responses.jsonl.report.json`. Both it and the command's JSON
summary contain the exact output SHA-256 and validation counts. Missing
model-scenario pairs, duplicates, identity conflicts, and scenarios outside the
fitted bank are errors by default. `--missing blank` is an explicit opt-in that
synthesizes absent pairs as blank responses; already blank or whitespace-only
responses are preserved and counted. Existing outputs are not replaced unless
`--overwrite` is passed, and source inputs are never valid output targets even
with that flag.
The batch and report must be placed in the same directory. They are published
as one recoverable transaction under an exclusive directory lock: a failed
second write restores both previous files, and the next invocation automatically
recovers a process interruption recorded in the transaction journal. Until that
recovery completes, the precomputed-response loader refuses to consume either
destination named by the unresolved journal; rerun `prepare-response-batch` with
the original arguments to recover the pair. The report records and verifies the
prepared payload digest, so it cannot silently describe different batch bytes.

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
  attempt_history.jsonl
  progress.json
  batch_results.json
  batch_results.csv
  BATCH_REPORT.md
  checkpoints/
    checkpoint-000001.json
    ...
    latest.json
  mode_result.json
  models/
    candidate-0000-<stable-hash>/
      attempt-0001/
        manifest.json
        tutor_responses.jsonl
        judge_rows.jsonl
        cat_result.json
        cat_trace.jsonl
    candidate-0001-<stable-hash>/
      ...
```

`model_results.jsonl` maps each original model identity to its safe output
attempt directory, terminal status, CAT metrics, warnings, and error if any.
`batch_summary.json` provides aggregate model, scenario, criterion, and
`no_decision` counts. `batch_results.json`, `batch_results.csv`, and
`BATCH_REPORT.md` are consolidated derived views; they preserve per-skill EAP
and MWLE estimates and do not invent a global score or model ranking.
`attempt_history.jsonl` and the immutable checkpoint chain retain failed and
superseded attempts. A runtime failure for one tutor does not prevent later
tutors from running, although the overall adaptive mode is marked failed if any
tutor failed.

Batch runs can be resumed only at tutor-model boundaries and only with the
exact same run ID, configuration, fitted bank, response batch, prompt/judge
contract, top-level run metadata, and runtime contract:

```bash
uv run olmo-eval run-modes --config /path/to/final-run.yaml --resume
```

Before Qwen starts, the runner validates the stored resume fingerprint and
checkpoint hash chain. After providers start, it builds the exact in-memory
bank/response snapshot, re-hashes the source inputs once more under the run
lock, and executes that retained snapshot rather than reloading mutable files.
It then reuses only successful attempts whose candidate
manifest and child artifact hashes still match. Failed or interrupted models
receive a new `attempt-NNNN` directory; old attempts are never deleted or
overwritten. A different input or a modified committed artifact makes resume
fail closed. The prompt portion of that fingerprint binds the complete rendered
atomic and classification templates, prompt branches, JSON schema, grading
policies, sampling settings, token IDs, probability policy, and failure
threshold—not only a prompt-version label.
Batch outputs created by the earlier non-checkpointed layout are not resumable;
start those evaluations in a new output directory.

To inspect persisted progress without checking GPUs or starting providers:

```bash
uv run olmo-eval run-modes --config /path/to/final-run.yaml --status
```

`progress.json` includes the active model and scenario, committed counts,
session throughput, and an estimated remaining time once enough work has
completed to estimate it. The status command reconciles both committed JSONL
views—`model_results.jsonl` and `attempt_history.jsonl`—with each other and with
`batch_summary.json`, `manifest.json`, and the newest valid immutable checkpoint.
It rejects mismatched run IDs, resume fingerprints, checkpoint generations, or
states rather than combining stale artifacts. The checkpoint chain is
authoritative: stale root manifests or summaries cannot override it, and the
derived model counts, terminal statuses, attempt count, and result paths must
agree across every view. Inconsistent views must be regenerated by resuming the
run before status is reported.

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
