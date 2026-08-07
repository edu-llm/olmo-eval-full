# PRD: Benchmark-Agnostic EduLLM–OLMo Eval Integration

## Document status

| Field | Value |
| --- | --- |
| Status | Feasibility spike complete; production implementation not started |
| Working branch | `eduLLM-Olmo-Integration` |
| Scope | Any criterion-based educational benchmark, not only InFoBench |
| Estimated engineering effort | 12–20 hours for a first vertical slice; 25–40 hours for the full acceptance scope, excluding model/GPU runtime and human data review |

## Summary

Provide two selectable evaluation modes: standard OLMo task/suite evaluation and EduLLM's criterion grading, calibration, and adaptive-testing workflow. OLMo Eval will provide the reusable task registry, benchmark variants and suites, model providers, parallel inference, scoring lifecycle, inspection tools, and run artifacts. EduLLM will retain ownership of its validated binary Qwen grading policy, educational metadata, IRT/MIRT calibration, ability estimators, and CAT policy.

The preferred outcome is a thin EduLLM adaptive plugin using OLMo provider and artifact infrastructure. If OLMo cannot preserve the adaptive semantics cleanly, the supported fallback is a separate EduLLM execution adapter behind the same mode dispatcher and report. Benchmark-specific assumptions must stay in adapters and configuration rather than in the shared pipeline.

## Feasibility spike outcome

The CPU-only spike completed on August 7, 2026. Its report is in `reports/edullm_olmo_feasibility_spike/FEASIBILITY_SUMMARY.md`.

- Sequential CAT selection through OLMo provider interfaces passed.
- Separate tutor and named judge providers passed with an injected provider lookup.
- EAP and MWLE matched provenance-bearing, recorded legacy-derived one-dimensional regression values. This is compatibility evidence, not independent validation.
- A malformed judge output remained `no_decision` and did not update ability.
- EduLLM sidecars were namespaced by evaluation/run and passed full disk round-trip checks; result names also stayed aligned with OLMo task names.
- OLMo's standard external `metrics.json` dropped rich adaptive metadata.
- The external runner did not own the judge lifecycle/GPU planning.
- Third-party evaluation modes were not automatically discoverable.

Conclusion: the adaptive method is technically compatible with OLMo's provider contracts, but the current repository is not yet a production-ready native integration. Start with a two-mode dispatcher and an EduLLM adaptive adapter. Continue toward a native `ExternalEval` plugin only after provider-lifecycle, artifact-fidelity, and plugin-discovery gates pass.

The deterministic spike uses SciPy from OLMo Eval's optional `analysis` dependency group. Reproduce it from the repository root with `uv run --extra analysis python scripts/internal/run_edullm_olmo_feasibility.py`.

The spike deliberately used deterministic top-1 information selection so two response paths could characterize the sequential interface. That is narrower than the legacy CAT default, which makes a seeded-random choice among the top five candidates; production migration must test the frozen deployment selector separately.

## Background and problem

The existing EduLLM work proves the core research workflow, but much of it is spread across benchmark-specific scripts, versioned experiment runners, and custom inference paths. That makes it easy for two runs to use different schemas, prompts, error policies, or provenance fields. It also makes each new benchmark require more custom orchestration than it should.

OLMo Eval already solves many general execution problems:

- task, variant, and suite registration;
- local and API-backed model providers;
- primary and auxiliary providers in one harness;
- asynchronous inference and scoring;
- request, prediction, metric, and configuration artifacts;
- prompt inspection and mock/dry runs;
- instance-aligned statistical comparisons; and
- an `ExternalEval` interface for evaluations that do not fit a fixed task queue.

The integration should reuse those capabilities while preserving the EduLLM semantics that OLMo Eval does not supply: one-criterion-at-a-time educational grading, explicit no-decision states, Q-matrices, calibration, EAP/MWLE, and sequential CAT.

## Goals

1. Expose `standard_olmo` and `edullm_adaptive` as independently selectable modes.
2. Support new educational benchmarks through small, benchmark-owned adapters.
3. Use OLMo Eval for fixed response generation and judging where its contracts preserve EduLLM semantics.
4. Keep tutor and judge providers separate and configurable in the same adaptive run.
5. Preserve the frozen Qwen binary grading path, including evidence requirements and `p_fail >= 0.33` as the fail rule.
6. Preserve raw inputs, raw outputs, parsing outcomes, errors, versions, and hashes for every decision.
7. Produce a strictly aligned model-by-criterion response matrix suitable for IRT/MIRT calibration.
8. Expose EduLLM's EAP/MWLE estimation, item selection, and stopping logic through a sequential CAT evaluation path.
9. Keep deterministic analysis, calibration, and comparison separate from model inference.
10. Allow generation and judging before a benchmark is calibrated, while refusing CAT until a calibrated bank passes its required validation gates.
11. Permit additional evaluation modes later without adding mode-specific branching to the dispatcher.

## Non-goals

- Rewriting OLMo Eval's provider, runner, task, suite, or storage systems.
- Requiring standard OLMo and EduLLM adaptive evaluation to use the same internal runner when doing so loses correctness or provenance.
- Making the shared pipeline specific to InFoBench, TutorBench, or any other single benchmark.
- Automatically rewriting, splitting, or synthesizing benchmark-authored criteria.
- Treating an old TutorBench judge-validation result as proof that the same judge is valid on every new benchmark.
- Migrating historical result archives, ZIP files, caches, virtual environments, or one-off plots into the new runtime.
- Adding direct AWS or S3 calls. GPU work must use the repository's `edullm` workflow.
- Forcing stateful CAT into OLMo Eval's ordinary `Task` queue, which is materialized before inference begins.
- Changing calibration models or CAT policy merely as a side effect of the integration.

## Design principles

1. **One user-facing dispatcher:** Standard OLMo and EduLLM adaptive execution may retain different internal engines, but users select them through one extensible mode interface.
2. **One criterion, one judgment:** A tutor response may have many criteria, but every judgment row concerns exactly one response–criterion pair.
3. **Scenario administration stays distinct from criterion scoring:** One scenario prompt can create several criterion judgments and therefore several calibration observations.
4. **Missing is not failing:** Blank responses, generation failures, over-length inputs, judge failures, parser failures, and ambiguous outputs remain explicit missing/no-decision states.
5. **Configuration over branching code:** Benchmark differences belong in registered adapters, variants, suites, and versioned configuration.
6. **Exact provenance before convenience:** Every reported score must be traceable to data, code, model, prompt, parser, threshold, and configuration versions.
7. **Calibration is a gate for CAT:** Uncalibrated benchmarks can generate and grade responses, but they cannot silently use placeholder item parameters for adaptive testing.

## Architecture and ownership boundary

```text
Evaluation-mode dispatcher
    -> standard_olmo
        -> OLMo Task / variant / suite
        -> fixed benchmark metrics and native OLMo artifacts
    -> edullm_adaptive
        -> benchmark adapter and tutor generation
        -> criterion cases and auxiliary Qwen judging
        -> response matrix + Q-matrix
        -> IRT/MIRT calibration and held-out validation (bank maintenance)
        -> calibrated item bank
        -> sequential CAT + EAP/MWLE (checkpoint evaluation)
        -> adaptive trace, ability estimate, uncertainty, and stop reason
```

| Area | OLMo Eval owns | EduLLM owns |
| --- | --- | --- |
| Benchmark loading | `DataSource`, task registry, variants, suites | Adapter mapping into the canonical educational schema |
| Tutor inference | Provider configuration, lifecycle, batching, retries, runner | Tutor prompt content supplied by the benchmark adapter |
| Judge inference | Named auxiliary-provider lifecycle and concurrency | Frozen Qwen prompt, binary parsing, probability rule, and no-decision policy |
| Scoring | Scorer execution and metric plumbing | Criterion-level judgment meaning and educational aggregates |
| Artifacts | Requests, predictions, metrics, storage hooks | Judge traces and additional policy/parser/data provenance |
| Statistical comparison | Shared-instance alignment and general pairwise machinery | Judge validity metrics such as macro-F1, critical sensitivity, repeat agreement, and prompt flips |
| Psychometrics | None | Q-matrix, IRT/MIRT, calibration diagnostics, EAP/MWLE, uncertainty |
| Adaptive testing | Optional `ExternalEval` adapter/context when integration gates pass | Item selection, exposure rules, stopping, ability updates, and CAT trace |

## Canonical data contracts

All contracts must be versioned, serializable, schema-validated, and use stable deterministic IDs. Field names should be normalized rather than preserving the legacy title-cased record names.

### Benchmark manifest

Defines the adapter name and version, benchmark and dataset revisions, split, scenario and criterion counts, skill-axis ordering, input hashes, and any benchmark-specific options. The manifest must not require calibrated item parameters.

### Scenario record

Represents the unit shown to a tutor model.

Required concepts:

- `benchmark_id`, `scenario_id`, and stable source ID;
- tutor-facing prompt, context, and optional reference material;
- ordered criterion IDs attached to the scenario;
- benchmark-defined metadata; and
- dataset revision and source hash.

### Criterion record

Represents one benchmark-authored pass/fail requirement.

Required concepts:

- `criterion_id` and parent `scenario_id`;
- original criterion text;
- skill mapping or Q-vector in a declared skill order;
- optional critical-failure flag and other benchmark-authored tags; and
- source/version provenance.

Criteria remain uncalibrated records until calibration succeeds. Difficulty and discrimination are therefore optional here and required only in a calibrated item bank.

### Generation record

Represents one tutor model's response to one scenario.

Required concepts:

- run, benchmark, scenario, and tutor-model identities;
- exact formatted request and raw tutor output;
- extracted response, token counts, finish reason, and truncation state;
- generation status and structured error; and
- provider, model revision, sampling configuration, seed, and timestamps.

### Criterion case and judgment

A criterion case joins one generation record to one criterion without including the human label or tutor identity in the judge-facing prompt. A judgment contains:

- stable `case_id`, `generation_id`, and `criterion_id`;
- `pass`, `fail`, or `no_decision`;
- `p_pass` and `p_fail` when available;
- evidence and rationale returned by the judge;
- exact judge prompt and raw judge output;
- parser status, error class, and attempt count;
- prompt, parser, policy, and threshold versions; and
- judge provider, model revision, sampling settings, and timing.

Only an explicit parsed judgment may become pass or fail. Infrastructure and parsing errors must never be silently converted to fail.

### Calibration input and calibrated bank

The calibration input contains an ordered tutor roster, ordered criterion roster, observed/missing response matrix, Q-matrix, and hashes tying every cell back to its judgment. The calibrated bank adds fitted difficulty and discrimination parameters, model form, skill ordering, fit diagnostics, validation results, uncertainty information, and calibration configuration hashes.

### CAT trace

Every adaptive administration must record the model, calibrated-bank hash, policy version, starting prior, eligible items, selected scenario at each step, item-selection objective, criterion outcomes, updated theta and uncertainty, and final stop reason. A trace must be sufficient to replay all deterministic CAT decisions from the stored judgments.

## OLMo Eval components to reuse

### Tasks, variants, and suites

- Build the static benchmark adapter on `Task` in `src/olmo_eval/evals/tasks/common/base.py`.
- Register benchmark tasks and named modes through `src/olmo_eval/evals/tasks/common/registry.py`.
- Use suites for benchmark groupings and display-level aggregation, not for Q-matrix or psychometric aggregation.
- Store benchmark-specific educational metadata in canonical instance metadata or an intentional EduLLM configuration type; do not add hidden globals.

### Providers and harness

- Use OLMo's local/API provider support for tutors.
- Configure Qwen as a named auxiliary provider in `HarnessConfig`.
- Use the provider registry supplied through `ScoringContext`; do not instantiate a separate judge client inside the scorer.
- Retain the mock provider for deterministic no-GPU integration tests and prompt inspection.

### Fixed-task runner and artifacts

- Use the asynchronous runner for the complete fixed benchmark path.
- Reuse its request/prediction JSONL writers, task/model hashes, configuration serialization, concurrency, and retry lifecycle.
- Extend prediction artifacts with structured criterion-judge traces rather than replacing OLMo's artifact format.

### Scorer and metric separation

- Add an EduLLM binary criterion scorer as a context-aware LLM judge scorer.
- Keep per-case judgment creation in the scorer and aggregation in metrics/analysis.
- Do not rely on the generic exception behavior that converts a scorer failure to `0.0`; EduLLM requires a structured no-decision state.

### Inspection and preflight

- Reuse OLMo's shared request formatting and inspection code.
- Add an EduLLM-aware preflight command that validates schemas, IDs, suite expansion, criterion links, Q-matrix alignment, formatted length distribution, auxiliary-judge configuration, and output paths.
- A one-case judge parse smoke test may be opt-in, but a dry run must make no model calls.

### Shared-case analysis

- Reuse OLMo's instance alignment and paired-comparison foundations where applicable.
- Keep EduLLM's deterministic judge-selection statistics in a dedicated analysis layer because they measure judge validity rather than ordinary benchmark accuracy.

### ExternalEval for CAT

`ExternalEval` is the preferred native OLMo integration because the next scenario depends on previous judgments. The feasibility spike added a backwards-compatible context and proved dual-provider lookup with mocks, but the current external runner still does not own the judge lifecycle or preserve rich adaptive metadata. Until those gaps are closed, the EduLLM mode may run through its own adapter behind the shared dispatcher. Neither implementation may construct an untracked one-off provider or lose the native CAT artifacts.

## EduLLM components to retain

### Frozen Qwen binary policy

- Judge: `Qwen/Qwen3.5-9B` at revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
- Mode: zero-shot, one criterion at a time.
- Output contract: strict binary JSON with verdict, evidence, and rationale.
- Decision rule: `p_fail >= 0.33` means fail; otherwise pass when a valid decision exists.
- Runtime contract: temperature `0.0`, seed `42`, probabilities required, and truncated outputs rejected.
- Evidence must come from the tutor response rather than an unstated reference answer.
- Prompt/parser/policy versions must be immutable within a run and stored in the manifest.

This path must never fall through to a Prometheus-style 1–5 prompt or threshold.

### Missing and no-decision policy

Preserve distinct statuses for absent tutor output, generation error, over-length input, judge generation error, parser error, and ambiguous output. Aggregates must report coverage and denominators explicitly.

### Educational and psychometric semantics

Retain benchmark-authored criteria, skill maps, Q-matrices, IRT/MIRT calibration, held-out validation, item-parameter uncertainty, EAP/MWLE ability estimation, information-based selection, exposure rules, stopping rules, and CAT-vs-baseline evaluation.

### Judge-validation analysis

Retain deterministic computation of accuracy, macro-F1, critical-failure sensitivity, per-skill F1, coverage, repeat agreement, prompt flip rate, threshold sweeps, disagreements, and scenario-clustered uncertainty. This code must make no LLM calls.

## End-to-end workflow

### 1. Register and preflight a benchmark

The benchmark adapter loads scenarios, criteria, and skill mappings into canonical contracts. Preflight validates IDs, joins, lengths, Q-matrix ordering, and configuration without requiring calibrated parameters.

### 2. Generate tutor responses

An OLMo task sends one scenario at a time to each configured tutor provider. OLMo writes exact requests, raw outputs, statuses, and run configuration. Failed generations remain explicit records so the expected cohort is auditable.

### 3. Expand response–criterion cases

Each valid tutor response is deterministically paired with all criteria attached to its scenario. Case IDs and input hashes are stable across reruns.

### 4. Judge criteria

The auxiliary Qwen provider receives a blinded, one-criterion prompt. The EduLLM scorer parses the strict binary response, applies the frozen probability rule, and writes both the normalized judgment and full trace. Invalid outputs become no-decision rather than fail.

### 5. Aggregate and validate artifacts

Deterministic checks verify expected/observed counts, uniqueness, matrix alignment, coverage, parser-error rates, hashes, and resume consistency. Optional human-label comparison is a separate analysis step.

### 6. Calibrate

The calibration layer consumes the frozen response matrix and Q-matrix, fits the configured IRT/MIRT model, estimates tutor abilities as needed by fitting/validation, and writes a versioned calibrated bank plus diagnostics. It must not call tutor or judge models.

### 7. Validate and freeze the bank

Held-out prediction, parameter stability, numerical stability, and uncertainty gates determine whether the bank is deployable. Failed validation is recorded as a valid failed result; the pipeline must not loosen a gate automatically.

### 8. Run CAT

The EduLLM adaptive adapter conducts the sequential loop: select scenario, obtain tutor response, judge its criteria, update ability/uncertainty with EAP or MWLE as configured, and evaluate the stop rule. It writes the full adaptive trace and final estimates. Once the remaining integration gates pass, this adapter may be exposed natively as an OLMo `ExternalEval`.

## Proposed code organization

Final names may be adjusted during implementation, but ownership should remain clear:

- `src/olmo_eval/edullm/`: canonical schemas, matrix assembly, calibration, judge-analysis, and CAT domain logic.
- `src/olmo_eval/evals/tasks/edullm.py` or an `edullm/` task package: static benchmark adapter and task registrations.
- `src/olmo_eval/evals/suites/edullm.py`: optional benchmark/split suites.
- `src/olmo_eval/common/scorers/edullm_binary.py`: frozen binary Qwen scorer and parser.
- `src/olmo_eval/common/preflight.py`: reusable benchmark/pipeline preflight checks.
- `src/olmo_eval/runners/io/provenance.py`: local run manifest and additional hashes.
- `src/olmo_eval/evals/external/benchmarks/edullm_cat/`: sequential CAT evaluation.
- `tests/edullm/` plus focused OLMo task/scorer/runner tests: fixtures, contracts, parity, and end-to-end tests.

## Migration strategy

1. Treat this branch's OLMo code as the execution foundation.
2. Treat `AdaptiveEvals` and `frq/infobench` as reviewed reference sources for EduLLM semantics, not as directories to copy wholesale.
3. Re-express reusable fixed-task behavior through OLMo-native task, provider, scorer, and result interfaces; use `ExternalEval` for adaptive execution only where the integration gates pass.
4. Keep benchmark-specific loaders, prompts, experiments, figures, and thresholds in adapters/configuration or as historical references.
5. Do not add the current untracked `eduLLM-Evals/` directory wholesale. It contains local environments, archives, caches, partial source, and historical outputs.
6. Migrate one small frozen fixture first, demonstrate artifact and decision parity, then add full benchmark adapters.
7. Preserve the legacy pipeline until parity and acceptance tests pass; remove or deprecate old execution paths only in a separately reviewed change.

## Implementation phases and deliverables

### Phase 0: Freeze contracts and reference behavior

Deliverables:

- canonical schema definitions and versioning rules;
- a frozen tiny benchmark fixture;
- recorded legacy prompt/parser/threshold behavior; and
- a decision log for missing/no-decision and provenance policy.

### Phase 1: Benchmark adapter and preflight

Deliverables:

- generic educational task adapter;
- adapter registration pattern for arbitrary benchmarks;
- schema, ID, criterion-link, Q-matrix, and token-length checks; and
- mock-provider request inspection.

### Phase 2: OLMo tutor generation and provenance

Deliverables:

- fixed-task generation through OLMo providers/runners;
- normalized generation artifacts;
- stable IDs, hashes, resume behavior, and a local run manifest; and
- no-GPU fixture test for request-to-artifact flow.

### Phase 3: Frozen Qwen criterion scorer

Deliverables:

- named auxiliary Qwen provider configuration;
- strict binary prompt and parser;
- `p_fail >= 0.33` decision rule;
- structured judge trace and no-decision states; and
- tests proving that the Qwen path cannot use a 1–5 Prometheus adapter.

### Phase 4: Aggregation and judge analysis

Deliverables:

- response–criterion matrix builder;
- expected/observed count and coverage reports;
- deterministic judge-validity analysis; and
- shared-case comparison hooks with no inference calls.

### Phase 5: Calibration interfaces

Deliverables:

- benchmark-agnostic IRT/MIRT input contract;
- extracted calibration library rather than hard-coded scripts;
- fitted bank, ability estimates, diagnostics, and held-out validation artifacts; and
- guard that blocks CAT for absent or failed calibration.

### Phase 6: Sequential CAT integration

Deliverables:

- auxiliary-provider-aware external-eval context;
- adaptive selection/update/stop loop;
- EAP/MWLE and uncertainty configuration;
- replayable CAT traces and checkpoint/resume behavior; and
- CAT-vs-random simulation on a fixed judged matrix.

### Phase 7: Benchmark acceptance and documentation

Deliverables:

- at least two adapters or fixtures with different schemas to demonstrate benchmark independence;
- end-to-end runbook and configuration examples;
- parity report against a frozen legacy subset; and
- migration/deprecation recommendations for old scripts.

## Test strategy

### Unit tests

- Schema validation, stable ID generation, serialization, and hashes.
- Scenario-to-criterion joins and exact skill/Q-matrix ordering.
- Valid binary, malformed JSON, conflicting verdict, missing evidence, and ambiguous judge outputs.
- No-decision propagation and denominator/coverage calculations.
- Deterministic matrix construction under shuffled input order.
- Calibration recovery on a synthetic matrix and refusal on invalid/underidentified inputs.
- EAP/MWLE updates, selection objectives, stop rules, and deterministic CAT replay.

### No-GPU integration tests

- Run a small real-format fixture through mock tutor and judge providers.
- Verify exact requests, predictions, judgment traces, metrics, and manifest fields.
- Verify resume is idempotent and does not duplicate cases.
- Verify generation/parser errors remain missing and never become fail.
- Verify a benchmark can generate and judge before calibration, while CAT refuses it.

### Parity tests

- Compare a frozen subset against the legacy Qwen prompt, parsed verdicts, and response matrix.
- Compare calibration output to a frozen numerical fixture within declared tolerances.
- Compare CAT selection and theta traces to the existing deterministic implementation.

### GPU/API smoke tests

- Run only after no-GPU acceptance passes.
- Use `edullm` for GPU execution and never add AWS calls.
- Keep the smoke subset small, then run the full workload from a committed and pushed branch.

## Acceptance criteria

The initial integration is complete when all of the following hold:

1. A benchmark adapter can load scenarios and criteria without calibrated difficulty/discrimination values.
2. The same pipeline runs at least two structurally different benchmark fixtures without shared-code edits.
3. Tutor and Qwen judge providers are configured separately through the OLMo harness.
4. Qwen judgments use only the frozen binary path and store `pass`, `fail`, or `no_decision` plus raw output and probabilities; `no_decision` must never be coerced to a failed (`0`) response.
5. Errors never silently become fails, and coverage is reported with every aggregate.
6. Every matrix cell can be traced back to a generation record, criterion, judgment, prompt/parser policy, and run manifest.
7. The fixed-task path uses OLMo's normal runner and artifacts rather than custom inference orchestration.
8. Calibration consumes a benchmark-neutral matrix/Q-matrix contract and produces a versioned fitted bank with diagnostics.
9. CAT refuses uncalibrated or validation-failed banks.
10. CAT records a replayable sequential trace and can access both tutor and judge providers without constructing hidden clients; a native `ExternalEval` path additionally requires all feasibility gates to pass.
11. Mock no-GPU tests, numerical tests, and the frozen legacy parity suite pass.
12. No new integration code directly calls AWS or S3.

## Provenance and parser-error requirements

Every run manifest must include:

- Git commit and dirty-tree status;
- task, suite, benchmark adapter, and schema versions;
- dataset URI/revision and content hashes;
- tutor and judge model names, immutable revisions, provider configs, and sampling settings;
- prompt, parser, decision-policy, threshold, skill-map, Q-matrix, and calibrated-bank versions/hashes;
- random seeds, start/end times, expected/observed counts, and software environment; and
- parent-run identifiers when judging, calibration, or CAT consumes earlier artifacts.

Every parsing failure must retain the raw output and a stable error class. Recovery, if allowed, must use a versioned deterministic parser and record both the original and recovered result. Ambiguous recovery remains no-decision.

## Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Legacy and OLMo schemas drift during migration | Freeze contracts and parity fixtures before moving behavior |
| Qwen accidentally uses a generic 1–5 judge path | Dedicated scorer, explicit adapter metadata, and a negative regression test |
| Scorer exceptions become false failures | Structured judgment status and no-decision-aware aggregation |
| Dynamic criteria do not fit ordinary fixed scorers cleanly | Expand stable criterion cases deliberately and persist their IDs/traces |
| CAT is forced into a pre-materialized task queue | Keep CAT in the adaptive adapter; use `ExternalEval` only after its named-provider and artifact gates pass |
| Calibration parameters or skills become misaligned | Ordered IDs, Q-matrix hashes, shape checks, and strict preflight |
| A validated judge does not transfer to a new benchmark | Require benchmark-specific judge audit/coverage evidence before research claims |
| New pipeline duplicates old orchestration | Share schemas and the mode dispatcher; reuse OLMo interfaces where compatible and keep any separate adapter thin |
| Historical local artifacts enter version control | Migrate reviewed source and small fixtures only; exclude environments, caches, archives, and bulk results |
| Upstream OLMo changes break extensions | Keep additions narrow, test public contracts, and avoid patching unrelated internals |

## Effort estimate

The first benchmark-neutral vertical slice is estimated at 12–20 engineering hours, while the full acceptance scope is estimated at 25–40 hours. The largest uncertainties are structured judge traces/no-decision handling, auxiliary-provider lifecycle, adaptive artifact storage, and plugin discovery. Full GPU generation/judging time, calibration runtime, benchmark data repair, and human judge validation are outside this estimate.

## Definition of ready for implementation

Implementation may begin after this PRD is reviewed and the following decisions are frozen:

- canonical schema version and required fields;
- exact source revision for the Qwen prompt/parser/policy;
- initial frozen parity fixture;
- first two benchmark adapters/fixtures used to prove generality; and
- whether CAT checkpoint/resume is required in the first release or immediately afterward.
