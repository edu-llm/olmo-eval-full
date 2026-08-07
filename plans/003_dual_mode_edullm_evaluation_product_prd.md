# PRD: Integrating EduLLM into the OLMo Eval Runner

## Document status

| Field | Value |
| --- | --- |
| Status | Implemented and no-GPU verified; real model execution remains explicit |
| Working branch | `eduLLM-Olmo-Integration` |
| Scope | Reusable EduLLM library extraction plus OLMo-runner integration |
| Runner | OLMo Eval |
| EduLLM role | Audited research logic migrated into a reusable library and invoked through a thin adapter |

This revision replaces the earlier assumption that all EduLLM logic was already available
behind production-ready entry points. The branch audit found that the core algorithms do
exist, but they are split across research branches, CLI scripts, offline-study helpers,
and local frozen-judge files. The implementation must migrate and harden those sources,
preserve their behavior with parity tests, and build only the missing runtime boundaries.

Implementation now provides `olmo-eval run-modes --config FILE`, with an explicit
read-only `--check` option. The focused integration suite and the repository-wide
non-Docker test suite pass. No calibration study, CAT experiment, Qwen GPU run, or tutor
model run was performed as part of this software integration.

The component-by-component source decision is recorded in
[`004_edullm_logic_source_audit.md`](004_edullm_logic_source_audit.md). That audit is the
authority for what is reused, what is excluded, and what genuinely needs implementation.

The earlier technical investigation remains in
[`002_edullm_olmo_integration_prd.md`](002_edullm_olmo_integration_prd.md). The completed
feasibility experiment is summarized in
[`reports/edullm_olmo_feasibility_spike/FEASIBILITY_SUMMARY.md`](../reports/edullm_olmo_feasibility_spike/FEASIBILITY_SUMMARY.md).

## Plain-language summary

OLMo Eval will be the runner. Users will select either OLMo's ordinary evaluation path,
EduLLM's custom evaluation path, or both. When EduLLM is selected, the OLMo runner calls a
thin adapter, and that adapter invokes the migrated, tested EduLLM library. The library is
not a second runner: it owns scientific algorithms and state transitions while OLMo owns
execution, providers, lifecycle, configuration, and top-level artifacts.

```text
OLMo runner
    -> standard_olmo
        -> existing OLMo tasks and suites
    -> edullm_adaptive
        -> thin compatibility adapter
            -> reusable EduLLM judging/calibration/CAT library
```

The adapter is not a second runner. It translates OLMo runner inputs into the extracted
EduLLM interfaces and returns EduLLM artifacts and summaries to OLMo. Numerical code is
migrated from audited sources rather than independently reinvented.

## Problem

OLMo Eval already provides a mature runner for model configuration, task and suite
selection, provider execution, status handling, and artifacts. EduLLM already provides
the specialized educational-evaluation logic:

- criterion-level binary Qwen judging;
- pass/fail/no-decision normalization;
- response-matrix and Q-matrix handling;
- IRT/MIRT calibration;
- EAP and MWLE ability estimation;
- adaptive scenario selection and stopping; and
- EduLLM-specific reports and traces.

Today these two systems are not connected through one supported runner interface. The
integration should connect them without changing the scientific behavior that the team
has already implemented and reviewed.

## Product decision

OLMo Eval is the sole top-level runner for both evaluation options.

- `standard_olmo` uses OLMo's existing task/suite path.
- `edullm_adaptive` is an OLMo runner mode backed by a thin EduLLM adapter.
- Users may select either mode or both.
- When both are selected, OLMo runs them in configuration order and records both under
  one run identity.
- Each mode keeps its own metrics and artifacts. OLMo does not combine ordinary benchmark
  accuracy with EduLLM theta estimates into one score.
- Future evaluation designs can register through the same mode interface.

There is no separate “adapter runner” in this design. The adapter is an integration seam
inside the OLMo-controlled run.

## Goals

1. Use OLMo Eval as the runner for the existing EduLLM pipeline.
2. Let users select `standard_olmo`, `edullm_adaptive`, or both through one configuration.
3. Reuse OLMo's provider, task/suite, run-status, and artifact infrastructure.
4. Preserve the existing EduLLM algorithms, policies, prompts, thresholds, and outputs.
5. Pass candidate/tutor and judge providers from OLMo into the EduLLM adapter.
6. Store mode-specific results under one traceable OLMo run.
7. Keep the integration benchmark-agnostic and extensible to future modes.

## Explicit non-goals

This integration task does **not** include:

- gathering new tutor-model responses;
- running the full Qwen judge study;
- changing or selecting a judge model;
- generating new human labels;
- fitting or revising calibration parameters;
- deciding difficulty or discrimination values;
- running a real calibration study;
- running a real CAT study;
- changing the Q-matrix or skill structure;
- independently redesigning the EAP, MWLE, IRT/MIRT, selection, or stopping mathematics;
- changing Qwen prompts, parsers, thresholds, or pass/fail policy;
- repairing benchmark data or criteria;
- proving that an existing bank or judge is scientifically valid; or
- submitting GPU jobs as part of runner implementation.

Fixtures and mocks may exercise the existing functions to prove that the runner wiring is
correct. That is integration testing, not a new scientific run.

This task **does** include extracting the audited algorithms into importable modules,
fixing the documented tri-state/final-stop/runtime-contract defects, adding asynchronous
OLMo provider adapters, and adding regression tests that prove the migrated behavior.

## Audited EduLLM sources remain authoritative

The source audit freezes authoritative implementations for:

- criterion judging;
- calibration;
- calibrated-bank loading;
- EAP/MWLE estimation;
- CAT execution; and
- report/artifact generation.

Some sources are not yet clean entry points. The implementation may extract pure functions,
separate state transitions from CLI/filesystem code, inject providers, and normalize
structured status. It must retain source provenance and parity tests, and it must not
silently alter scientific defaults.

The adapter invokes an importable Python entry point with dependency-injected provider
interfaces. CLI-only numerical code is moved into a reusable module and the CLI becomes a
consumer of that module. Direct-library and OLMo-run execution must produce equivalent
results from the same fixture inputs.

## User-facing mode selection

The exact CLI spelling remains an implementation decision, but the configuration contract
will have this form:

```yaml
evaluation:
  modes:
    - standard_olmo
    - edullm_adaptive
  continue_on_mode_failure: true

model:
  provider: candidate
  checkpoint: <checkpoint-reference>

standard_olmo:
  suites:
    - <explicit-olmo-suite>

edullm_adaptive:
  pipeline_config: <existing-edullm-config>
  judge_provider: judge
  input_artifacts: <existing-input-reference>
```

Rules:

- at least one mode is required;
- no mode, suite, benchmark, or bank is selected implicitly;
- modes run sequentially in the listed order for the first implementation;
- `continue_on_mode_failure` defaults to `true`;
- a mode-specific failure does not erase the other mode's completed results; and
- an invalid shared checkpoint identity or unsafe output root blocks the whole run.

## Architecture

```text
OLMo CLI / configuration
        |
        v
OLMo runner
        |
        +-- shared run ID, checkpoint identity, provider registry, output root
        |
        v
Evaluation mode registry
        |
        +------------------------------+
        |                              |
        v                              v
StandardOlmoMode                 EduLLMAdaptiveMode
        |                              |
native OLMo tasks/suites         EduLLMPipelineAdapter
                                       |
                                       v
                              existing EduLLM entry points
```

### OLMo runner responsibilities

OLMo owns:

- reading and validating the top-level configuration;
- selecting and sequencing modes;
- the shared run and checkpoint identity;
- creation and cleanup of declared candidate and judge providers;
- passing provider handles through the run context;
- top-level status and partial-failure handling;
- safe mode-specific artifact directories;
- the combined manifest and report index; and
- standard OLMo execution when `standard_olmo` is selected.

### EduLLM adapter responsibilities

The thin adapter owns:

- validating the EduLLM-specific portion of the run configuration;
- translating the OLMo run context into the existing EduLLM input contract;
- binding OLMo-managed tutor and judge providers to the existing logic;
- invoking the authoritative existing EduLLM entry point;
- preserving EduLLM's native output files and errors;
- returning status, summary fields, and artifact links to OLMo; and
- refusing incompatible or incomplete inputs before invoking the pipeline.

The adapter does not own a second global run identity, start hidden provider clients, or
reinterpret scientific results.

### Reusable EduLLM library responsibilities

The migrated EduLLM library owns:

- how criteria are judged;
- pass/fail/no-decision semantics;
- Qwen prompting, parsing, and probability decisions;
- matrix and calibrated-bank semantics;
- calibration mathematics;
- IRT/MIRT, EAP, and MWLE;
- scenario selection and stopping; and
- EduLLM-specific trace and report contents.

It does not create provider clients, submit jobs, call cloud APIs, or own a competing
top-level run lifecycle.

## Runner integration contract

### `RunContext`

OLMo supplies a read-only context containing at least:

- run ID and output root;
- code and configuration revision information;
- checkpoint/model identity;
- candidate/tutor provider handle;
- named judge provider handle when requested;
- selected mode configuration; and
- cancellation/status hooks supported by OLMo.

### `ModeResult`

Every mode returns a small OLMo-facing envelope containing:

- mode name and implementation version;
- `succeeded`, `failed`, `blocked_preflight`, or `cancelled` status;
- mode-native headline metrics without reinterpretation;
- warnings and structured error information;
- the number of completed units when known; and
- links to the mode's native artifacts.

`ModeResult` is an index, not a replacement for complete EduLLM or OLMo result files.

### `EvaluationMode`

Each mode implements:

- stable identity;
- configuration validation;
- declared provider requirements;
- read-only preflight;
- execution from `RunContext`; and
- conversion to `ModeResult`.

The dispatcher operates on this interface rather than mode-specific branches. Adding a
future mode requires registration, not a new dispatcher algorithm.

## Provider integration

OLMo must remain responsible for the candidate/tutor and judge provider lifecycle.

- `standard_olmo` receives the candidate provider through its existing path.
- `edullm_adaptive` receives the candidate provider and a named judge provider through
  `RunContext`.
- The adapter passes those handles to existing EduLLM logic.
- The adapter must not instantiate a second untracked vLLM server or API client.
- OLMo closes providers after all selected modes that use them finish.
- Provider errors remain structured infrastructure errors; they do not become failed
  educational criteria.

If the current EduLLM entry point constructs its own provider internally, the integration
may add dependency injection at that boundary. The underlying prompt, parsing, judgment,
calibration, and CAT behavior remains unchanged.

## Artifact integration

Proposed layout:

```text
<olmo-run-root>/
  manifest.json
  report_index.json
  modes/
    standard_olmo/
      ... existing OLMo artifacts ...
    edullm_adaptive/
      ... existing EduLLM artifacts ...
      mode_result.json
```

Requirements:

- OLMo creates and owns the shared run root.
- Each mode receives a collision-safe isolated directory.
- Existing EduLLM artifacts are preserved rather than flattened into `metrics.json`.
- `report_index.json` links to each mode's native output.
- Strict JSON rules apply to newly introduced manifests and envelopes.
- `no_decision` remains missing and is never rewritten as criterion failure.
- Raw outputs and existing provenance fields remain available.
- A failure in one mode cannot overwrite or invalidate the other mode's artifacts.

## Preflight behavior

Preflight performs runner and interface checks without conducting the actual evaluation.

Shared preflight validates:

- selected modes;
- checkpoint identity;
- provider declarations;
- output-root safety; and
- top-level configuration versions.

The EduLLM adapter preflight validates only what is necessary to invoke the existing
pipeline safely, such as required input paths, configuration fields, provider names, and
output compatibility. It does not recompute calibration, evaluate bank quality, or run CAT.

If one mode fails its own preflight and `continue_on_mode_failure` is true, another valid
mode may still run. Shared preflight failures block every mode.

## Failure semantics

- OLMo runner failures are recorded as runner/infrastructure failures.
- Adapter invocation failures are recorded as structured mode failures.
- Existing EduLLM parsing and no-decision rules remain unchanged.
- Generic OLMo scorer error handling must not convert an EduLLM exception or missing
  judgment into numeric `0`.
- Partial results remain accessible when a later mode fails.
- The combined report clearly distinguishes `succeeded`, `failed`,
  `blocked_preflight`, and `partial_failure`.

## Deliverables

1. Importable, provenance-documented EduLLM modules for calibrated-bank loading,
   MIRT/EAP/MWLE, CAT state, Qwen judging, and offline calibration.
2. Ported numerical and behavioral parity tests from the authoritative branch sources.
3. Versioned `EvaluationMode`, `RunContext`, and `ModeResult` contracts.
4. An OLMo-owned mode registry and dispatcher.
5. A `standard_olmo` mode that delegates to an actual OLMo suite unchanged.
6. An `edullm_adaptive` mode containing the thin `EduLLMPipelineAdapter`.
7. OLMo-to-EduLLM provider binding for candidate and judge providers.
8. Safe per-mode artifact namespaces and a combined report index.
9. Structured shared, preflight, mode, and partial-failure statuses.
10. Documentation showing how benchmark banks and EduLLM configurations are passed
    through OLMo.
11. No-GPU fixture tests and OLMo regression tests.

## Testing strategy

### OLMo regression

- Run one actual OLMo suite directly and through `standard_olmo`.
- Confirm requests, metrics, and artifacts are equivalent.
- Confirm no judge provider is created for a standard-only run.

### EduLLM adapter parity

- Run an existing deterministic EduLLM fixture directly.
- Run the same fixture through the OLMo runner and adapter.
- Confirm the same judgments, selection path, EAP/MWLE outputs, errors, and native
  sidecars.
- Confirm candidate and judge calls use the OLMo-supplied provider handles.

### Dispatcher behavior

- standard-only configuration;
- EduLLM-only configuration;
- both modes in configuration order;
- one mode blocked during preflight while the other succeeds;
- one mode fails after the other succeeds;
- shared preflight blocks both;
- artifact directories cannot collide or escape the run root; and
- a future mock mode registers without changing dispatcher logic.

### Scope guard

Tests must demonstrate that integration work does not:

- alter frozen EduLLM outputs from identical fixture inputs;
- create new calibration values;
- change Qwen judgment semantics;
- modify CAT selection or stopping behavior; or
- require a real GPU or a new scientific run.

## Acceptance criteria

The runner integration is complete when:

1. one OLMo configuration can select OLMo evaluation, EduLLM evaluation, or both;
2. OLMo remains the top-level runner in all three cases;
3. `standard_olmo` matches direct OLMo execution;
4. `edullm_adaptive` invokes the authoritative existing EduLLM logic rather than a copied
   implementation;
5. direct and OLMo-wrapped EduLLM fixtures produce equivalent outputs;
6. OLMo supplies and cleans up both tutor and judge providers;
7. native artifacts from both systems remain complete and isolated;
8. failure and partial-failure states are explicit;
9. no scientific parameters, prompts, thresholds, or policies change; and
10. all no-GPU integration and relevant OLMo regression tests pass.

## Implementation sequence

### Step 1: Freeze source-of-truth implementations

Record authoritative files, revisions, behavior, tests, exclusions, and genuine gaps.
Completed in `004_edullm_logic_source_audit.md`.

### Step 2: Extract and test the reusable EduLLM library

Migrate the audited numerical primitives, strict bank contract, CAT state machine,
frozen-Qwen judgment logic, and calibration API. Fix only documented runtime-contract
defects, with source parity tests before integration behavior is added.

### Step 3: Add OLMo mode contracts and dispatcher

Implement `EvaluationMode`, `RunContext`, `ModeResult`, mode registration, sequential
dispatch, status handling, and isolated output roots.

### Step 4: Connect real OLMo execution

Implement `standard_olmo` by delegating to an actual OLMo suite and prove that the wrapper
does not change its behavior.

### Step 5: Add the thin EduLLM adapter

Bind OLMo's candidate and named judge providers, translate configuration, invoke the
existing EduLLM entry point, and return artifact links and status.

### Step 6: Add combined reporting and failure behavior

Write the shared manifest and report index, then cover per-mode preflight, partial failure,
and artifact isolation.

### Step 7: Verify parity

Run direct-versus-wrapped fixture comparisons and the relevant OLMo regression suite. Do
not run new calibration or CAT studies as part of this step.

## Inputs not required for implementation

The runner foundation can be implemented without:

- new tutor responses;
- real Qwen result files;
- new human labels;
- finalized calibration outputs;
- a newly fitted item bank; or
- a completed real CAT run.

Those inputs are needed only when the team later chooses to conduct the corresponding
evaluation through the finished runner.

## Resolved decisions

- OLMo Eval is the sole runner.
- EduLLM is integrated through a thin adapter inside the OLMo-controlled run.
- Existing EduLLM logic remains authoritative and is not reimplemented.
- Users may select OLMo, EduLLM, or both.
- Mode-native metrics remain separate.
- OLMo owns tutor and judge provider lifecycles.
- The adapter calls an importable EduLLM entry point with dependency-injected OLMo
  providers; it does not launch an independent CLI process that owns clients.
- The integration is benchmark-agnostic.
- Implementation uses fixtures and mocks; it does not perform new scientific studies.

## Implemented decisions

- The supported command is `olmo-eval run-modes --config FILE`; `--check` performs
  read-only configuration, fitted-bank, provider, GPU, and frozen-Qwen runtime preflight.
- Reusable EduLLM logic lives in `src/olmo_eval/edullm/`.
- Mode contracts and orchestration live in `src/olmo_eval/runners/`.
- `standard_olmo` delegates to the native `AsyncEvalRunner`.
- `edullm_adaptive` invokes the extracted EduLLM entry point with OLMo-owned candidate
  and judge providers.
- No-GPU direct-versus-wrapped fixtures are the parity oracle; real benchmark and model
  runs remain separate explicit operations.

## Recommended next step

Review the integration diff, prepare one approved benchmark-specific configuration and
fitted bank, run `olmo-eval run-modes --check`, and only then schedule an explicit cluster
evaluation. A real run should not be inferred from this implementation milestone.
