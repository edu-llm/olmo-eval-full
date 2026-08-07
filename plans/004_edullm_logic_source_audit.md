# EduLLM Logic Source Audit

## Status

| Field | Value |
| --- | --- |
| Audit date | 2026-08-07 |
| Integration branch | `eduLLM-Olmo-Integration` |
| Purpose | Select authoritative existing logic before building the OLMo-runner integration |
| Changes made by audit | Documentation only; no research branch was merged |

## Decision in plain language

EduLLM's logic is **partly built, but not yet assembled as a production library**.
The correct implementation strategy is therefore:

1. migrate the best-tested algorithms from the research branches;
2. preserve their numerical behavior with parity tests;
3. fix a small number of known runtime-contract defects;
4. add OLMo-owned provider, lifecycle, configuration, and artifact boundaries; and
5. build only the pieces that genuinely do not exist.

Do not merge a whole research branch. Those branches also contain benchmark-specific
data, provisional or failed policies, large study outputs, historical scripts, and stale
infrastructure code.

## Audited refs

| Ref | Tip during audit | Role |
| --- | --- | --- |
| `origin/frq/infobench` | `b4ea2e8e4b1da75067dd5e38ea5e83270776a07f` | Newest reusable calibration and adaptive-evaluation algorithms |
| `origin/frq-lab` | `995fa27df757c63dfdd868f1372efe007c832aab` | Live successor to `AdaptiveEvals`; cross-benchmark scientific policy |
| `origin/frq/biggen` | `acd9685` | Older EAP/MWLE experiments and benchmark-specific parity evidence |
| `origin/frq/tutorbench` | `c9062c3` | Historical TutorBench pipeline and artifacts |
| local uncommitted integration files | SHA-256 recorded below | Newest frozen Qwen judging and large-cohort orchestration |

The remote `AdaptiveEvals` branch has been deleted. The stale local remote-tracking ref at
`3a55809` is superseded by `frq-lab`; its relevant core blobs are the same older versions.

Across `frq/ap-ib`, `frq/biggen`, `frq/bridge`, `frq/dr-sci`, `frq/edubench`,
`frq/tutorbench`, `frq/tutoreval`, and `frq/wildbench`, the core `mirt.py`, `engine.py`,
`selector.py`, schemas, calibration script, and judge base runner are identical older
blobs. InFoBench is the only audited branch with the later joint-EAP stopping,
variable-dimensional schemas, strict fitted-bank path, and hardened calibration variants.

## Authoritative source map

| Component | Selected source | Blob / local hash | Reusable APIs | Evidence |
| --- | --- | --- | --- | --- |
| Cross-benchmark scientific policy | `origin/frq-lab:eduLLM-Evals/plans+prds/Calibration Playbook - Cross-Benchmark.md` | `1357de2...` | Policy only | Section 8.7 specifies joint dense-grid EAP posterior-SD stopping and final MWLE reporting |
| MIRT calibration | `origin/frq/infobench:eduLLM-Evals/scripts/calibrate_mirt.py` | `7e2b05ee...` | `fit_m2pl_em`, `build_calibration_quadrature`, `calibration_specification`, strict matrix/Q validation | Calibration, quadrature, returned-iterate, and model-family tests |
| Calibration helpers | `origin/frq/infobench:eduLLM-Evals/scripts/calibrate_partial.py` | `68c4f571...` | Coverage and zero-variance filtering helpers | `test_calibrate_partial.py` |
| Latent skill structures | `origin/frq/infobench:eduLLM-Evals/tutor_cat/skill_structure.py` | `0304fa75...` | `SkillStructure`, `parse_dimension_spec`, Q-column merging | Dynamic-bank and k-fold tests |
| Runtime MIRT math | `origin/frq/infobench:eduLLM-Evals/tutor_cat/mirt.py` | `44c13d50...` | Probability, online update, covariance, joint-grid posterior moments | `test_mirt.py` |
| EAP, MWLE, quadrature | `origin/frq/infobench:eduLLM-Evals/scripts/scenario_cat_lib.py` | `20bb1f61...` | `build_quadrature`, `batch_eap`, `mwle`, `pass_rate_check` | `test_scenario_cat_lib.py` |
| CAT engine and stopping | `origin/frq/infobench:eduLLM-Evals/tutor_cat/engine.py` | `2bc74f68...` | `RunConfig`, `run_evaluation`; online selection state and optional EAP-SD stopping | `test_engine_sim.py` and EAP prototype tests |
| Scenario selector | `origin/frq/infobench:eduLLM-Evals/tutor_cat/selector.py` | `19e634ad...` | Fisher/trace, seeded top-N, fallback, optional D-optimal selection | `test_selector.py` |
| Runtime schemas | `origin/frq/infobench:eduLLM-Evals/tutor_cat/schemas.py` | `b6b8ed1f...` | Scenario, rubric, and verdict records | Dynamic-bank tests |
| Strict fitted-bank loading | InFoBench `scenario_cat_lib.py` plus exporter guarantees | `20bb1f61...` | `load_fitted_bank`, `load_scenario_records` | `test_scenario_cat_lib.py` |
| Safe fitted-bank export | `origin/frq/infobench:eduLLM-Evals/scripts/export_fitted_bank.py` | `fd6a7d47...` | `ExportConfig`, `export_fitted_bank` | `test_export_fitted_bank.py` |
| Frozen Qwen prompt, parsing, probability, and aggregation | local `eduLLM-Evals/scripts/run_local_judge_v4.py` | SHA-256 `64d2af43330861bf05acb04330ea29dbd000452bfff6b553e22dc047958f0ba3` | Atomic prompt construction, native parse, P/F probability normalization, `max_atomic_p_fail` | 40+ focused offline tests |
| Calibration-cohort preparation and merge | local `eduLLM-Evals/scripts/run_calibration_judging.py` | SHA-256 `a0d3ec8d97155ace2c525e05b2e9de859ef54a7cc48ece45ae34ce63f2e4d310` | Blinding, sharding, missing-cell contract, retry/status/merge | 20+ focused offline tests |

## Governing behavior to preserve

- A criterion is binary Pass/Fail only when the judge produced a valid decision.
  Infrastructure or parse failures remain `no_decision`/missing and never become Fail.
- The selected Qwen configuration is zero-shot atomic judging with
  `p_fail >= 0.33` interpreted as Fail.
- A criterion containing reviewed atomic requirements fails when any atomic requirement
  reaches the failure threshold. Criterion probability is `max_atomic_p_fail`.
- Candidate/tutor identity and human labels are absent from judge prompts.
- Missing recorded judgments are excluded from calibration and ability estimation; they
  are not filled with zero.
- Scenario selection uses the online MIRT `theta/U` state. Joint dense-grid EAP marginal
  posterior SD may control stopping. Final EAP and MWLE are reported separately.
- Precision stopping requires all configured marginal SE targets, per-skill criterion
  floors, and the minimum-scenario floor.
- The selector is configured explicitly. Trace/Fisher is not silently interchangeable
  with D-optimal selection.
- The scenario-MIRT parameterization is `sigmoid(A @ theta - b)`.

## Logic that must not be mixed in

- `tutor_cat/mcq_irt/` is a separate older item-level model using
  `sigmoid(a * (theta - b))`; its difficulty parameter is not interchangeable with the
  scenario-MIRT `b`.
- `src/olmo_eval/adaptive/` on the research branches implements ATLAS-style
  unidimensional 3PL CAT. Its session architecture is useful, but its math and banks are
  not EduLLM FRQ/MIRT logic.
- Benchmark study orchestrators (`nested_*`, `finalize_*`, plotting scripts, and complete
  run directories) are evidence-generation tools, not runtime APIs.
- Historical AWS/S3 code must not be migrated. OLMo owns provider execution and the
  supported cluster path is `edullm`.
- Provisional fitted banks and benchmark-specific policy values must not become global
  defaults.

## Genuine gaps and required fixes

### Library extraction

The numerical logic is mostly trapped in CLI and offline-study files. Extract it into an
importable package with no mutable global skill state, dynamic file imports, direct vLLM
construction, subprocess-owned providers, or cloud calls.

### OLMo provider adapters

The Qwen runner currently constructs vLLM itself. The integration needs a judge adapter
that sends both native and P/F scoring requests through an OLMo-managed named provider,
while retaining probability and parser invariants. The tutor/candidate provider must also
come from OLMo.

### Stateful async CAT boundary

The existing engine is synchronous and combines state transitions, provider calls, and
file writing. Extract a provider-independent session/state machine and drive it through
async OLMo providers. Preserve selection and stopping parity.

### Strict bank contract

Combine the strict loader and exporter guarantees into one versioned manifest contract.
Reject source banks with synthetic/unverified parameters, missing hashes, inconsistent
skill order, invalid Q rows, non-finite values, or incompatible parameterization.

### Known correctness defects

- Existing `JudgeVerdict.y` maps non-pass states to zero; tri-state semantics must be
  explicit.
- Existing engine final precision can omit `min_scenarios` after bank exhaustion.
- The permissive loader can return a partially invalid bank.
- Export and runtime disagree on positive-semidefinite versus positive-definite latent
  correlation handling.
- Engine exceptions do not currently produce a structured partial/failure artifact.
- Tensor quadrature grows as `nodes ** dimensions` and requires an explicit size guard.

### Calibration API

Extract the robust returned-iterate calibration path and require the caller to choose a
model family, skill structure, quadrature, convergence policy, and missing-data contract.
Do not silently use the historical free-2PL default.

## Scientific status is not a software default

The source InFoBench bank still contains synthetic difficulty/discrimination values. The
branch does not contain an approved final fitted bank or deployable CAT policy.

The latest InFoBench evidence selected a one-dimensional positive log-shrinkage 2PL
candidate in its inner panels, but the proposed floor-15 / SE-0.20 / trace policy failed
all outer validation panels and all pooled repetitions. Final fit/export was not
authorized. Qwen also has not yet received an InFoBench-specific human validation.

Therefore the integration may ship algorithms and configuration support, but it must:

- require a caller-supplied versioned fitted bank;
- label unvalidated policies as experimental;
- never install the failed InFoBench policy as a universal default; and
- never claim that software wiring validates the bank or judge scientifically.

## Migration sequence

1. Port numerical primitives and their focused tests from InFoBench.
2. Add strict, versioned bank and configuration contracts.
3. Extract an async stateful CAT session with EAP stopping and final MWLE.
4. Extract frozen Qwen prompt/parser/probability logic behind an OLMo provider adapter.
5. Extract calibration into an importable offline API.
6. Connect these components to the OLMo-owned evaluation-mode runner.
7. Add direct-versus-wrapped parity tests and partial-failure/artifact tests.

No calibration, judge study, or CAT experiment is required merely to perform this
software migration. Real scientific runs remain separate, explicit actions.
